import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *
from dassl.utils import (
    MetricMeter, AverageMeter
)
import datetime
import time
import copy

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'NLPrompt',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.NLPROMPT.N_CTX
        ctx_init = cfg.TRAINER.NLPROMPT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.NLPROMPT.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.NLPROMPT.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class GeneralizedCrossEntropy(nn.Module):
    """Computes the generalized cross-entropy loss, from `
    "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"
    <https://arxiv.org/abs/1805.07836>`_
    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`
    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """
    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-6
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        # Avoid undefined gradient for p == 0 by adding epsilon
        p += self.epsilon
        loss = (1 - p ** self.q) / self.q
        return torch.mean(loss)

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class NLPrompt(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.GCE_loss = GeneralizedCrossEntropy(q=1.0)
        self.num_equal = []
        self.confident_rate = []
        self.clean_rate  = []

        self.best_acc = -1
        self.best_epoch = -1
        self.test_acc = []

    def check_cfg(self, cfg):
        assert cfg.TRAINER.NLPROMPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.NLPROMPT.PREC == "fp32" or cfg.TRAINER.NLPROMPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.NLPROMPT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward_ce(self, batch):
        image, label, gt_label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.NLPROMPT.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss_x": loss.item(),
            "acc_x": compute_accuracy(output, label)[0].item(),
        }

        return loss_summary
    
    def forward_backward_mae(self, batch):
        image, label, gt_label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.NLPROMPT.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = self.GCE_loss(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = self.GCE_loss(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss_u": loss.item(),
            "acc_u": compute_accuracy(output, label)[0].item(),
        }

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        gt_label = batch["gttarget"]
        input = input.to(self.device)
        label = label.to(self.device)
        gt_label = gt_label.to(self.device)
        return input, label, gt_label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def before_epoch(self):
        cfg = self.cfg
        if cfg.DATASET.USE_OT == True:
            reg_feat = cfg.DATASET.REG_FEAT
            reg_lab = cfg.DATASET.REG_LAB
            curriclum_epoch = cfg.DATASET.CURRICLUM_EPOCH
            begin_rate = cfg.DATASET.BEGIN_RATE
            curriclum_mode = cfg.DATASET.CURRICLUM_MODE
            Pmode = cfg.DATASET.PMODE
            reg_e = cfg.DATASET.REG_E

            if self.epoch < curriclum_epoch:
                budget, pho = curriculum_scheduler(self.epoch, curriclum_epoch, begin=begin_rate,end=1,mode=curriclum_mode)
            else:
                budget, pho = 1., 1.

            with torch.no_grad():
                pseudo_labels1, noisy_labels, gt_labels, selected_mask, conf1, argmax_plabels = OT_PL(self.model, 
                        self.train_loader_x, num_class=cfg.DATASET.num_class, batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE, budget=budget, reg_feat=reg_feat, 
                        reg_lab=reg_lab,Pmode=Pmode, reg_e=reg_e, load_all=True)

                print("before epoch:data num:", len(gt_labels))
                print("before epoch:different number:", np.sum(gt_labels.cpu().numpy() != argmax_plabels.cpu().numpy()))

                conf_l_mask, conf_u_mask, lowconf_u_mask = get_masks(argmax_plabels, noisy_labels, None, selected_mask)
                selected_rate_conf_l, selected_rate_conf_u, selected_rate_lowconf_u = output_selected_rate(conf_l_mask, conf_u_mask, lowconf_u_mask)
                print("confident_label rate",selected_rate_conf_l)
                unlabeled_mask1 = torch.logical_or(conf_u_mask, lowconf_u_mask)

            if np.sum(conf_l_mask.cpu().numpy()) > 0:
                mask = conf_l_mask.cpu().numpy() 
                self.mask2 = unlabeled_mask1.cpu().numpy()
                pred_idx = mask.nonzero()[0]
                pred_idx2 = self.mask2.nonzero()[0]
                conf = conf1.cpu().numpy()
                plabel = argmax_plabels.cpu().numpy()

                self.tmp_train_loader_x = copy.deepcopy(self.train_loader_x)
                self.train_loader_u = copy.deepcopy(self.train_loader_x)
                

                print("before: len(self.train)",len(self.train_loader_x.dataset.data_source))
                print("before: len of confident samples",len(pred_idx))


                count11=0
                count12=0
                count21=0
                count22=0
                for i in range(len(self.train_loader_x.dataset.data_source)):
                    if mask[i]== True:
                        if plabel[i] == gt_labels[i]:
                            count11 += 1
                        else:
                            count12 += 1
                    elif self.mask2[i]== True:
                        if plabel[i] == gt_labels[i]:
                            count21 += 1
                        else:
                            count22 += 1
                print(f"clean true:{count11}")
                print(f"clean false:{count12}")
                clean_rate=count11/(count11+count12)
                print(f"clean_rate:{clean_rate}")
                self.clean_rate.append(clean_rate)
                print(f"noisy true:{count21}")
                print(f"noisy false:{count22}")

                if self.epoch == 99:
                    print("all clean rate: ", self.clean_rate)

                for index in sorted(pred_idx2,reverse = True):
                    del self.train_loader_x.dataset.data_source[index]
                print("after delete: len(clean_dataset)",len(self.train_loader_x.dataset.data_source))

                for index in sorted(pred_idx,reverse = True):
                    del self.train_loader_u.dataset.data_source[index]
                print("after delete: len(noisy_dataset)",len(self.train_loader_u.dataset.data_source))

    def run_epoch(self):
        self.set_model_mode("train")
        losses_x = MetricMeter()  
        losses_u = MetricMeter()  
        batch_time = AverageMeter()
        data_time = AverageMeter()

        if self.train_loader_x is not None:
            train_loader_x_iter = iter(self.train_loader_x)
            len_train_loader_x = len(self.train_loader_x)
        else:
            len_train_loader_x = 0

        if self.train_loader_u is not None:
            train_loader_u_iter = iter(self.train_loader_u)
            len_train_loader_u = len(self.train_loader_u)
        else:
            len_train_loader_u = 0

        self.num_batches_x = len_train_loader_x
        self.num_batches_u = len_train_loader_u

        end = time.time()
        
        for self.batch_idx in range(self.num_batches_x):
            try:
                batch_x = next(train_loader_x_iter)
                data_time.update(time.time() - end)
                loss_summary_x = self.forward_backward_ce(batch_x)
                losses_x.update(loss_summary_x)
            except StopIteration:
                break  

            batch_time.update(time.time() - end)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches_x < self.cfg.TRAIN.PRINT_FREQ:
                eta_seconds = batch_time.avg * (self.num_batches_x - self.batch_idx - 1)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches_x}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"loss_x {losses_x}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}"
                ]
                print(" ".join(info))

            n_iter = self.epoch * (self.num_batches_x + self.num_batches_u) + self.batch_idx
            for name, meter in losses_x.meters.items():
                self.write_scalar("train_x/" + name, meter.avg, n_iter)

            end = time.time()

        for self.batch_idx in range(self.num_batches_u):
            try:
                batch_u = next(train_loader_u_iter)
                data_time.update(time.time() - end)
                loss_summary_u = self.forward_backward_mae(batch_u)
                losses_u.update(loss_summary_u)
            except StopIteration:
                break  # If the iterator is exhausted, exit the loop

            batch_time.update(time.time() - end)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches_u < self.cfg.TRAIN.PRINT_FREQ:
                eta_seconds = batch_time.avg * (self.num_batches_u - self.batch_idx - 1)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches_u}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"loss_u {losses_u}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}"
                ]
                print(" ".join(info))

            n_iter = self.epoch * (self.num_batches_x + self.num_batches_u) + self.batch_idx
            for name, meter in losses_u.meters.items():
                self.write_scalar("train_u/" + name, meter.avg, n_iter)

            end = time.time()

        self.update_lr()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
        
        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
        
        if self.cfg.DATASET.USE_OT == True:
            self.train_loader_x = copy.deepcopy(self.tmp_train_loader_x)
            self.train_loader_u = copy.deepcopy(self.tmp_train_loader_x)
            print("after epoch: len(clean dataset)", len(self.train_loader_x.dataset.data_source))
            print("after epoch: len(noisy dataset)", len(self.train_loader_u.dataset.data_source))
