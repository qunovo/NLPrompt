import copy
import datetime
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import AverageMeter, MetricMeter, load_checkpoint, load_pretrained_weights
from utils import OT_PL, curriculum_scheduler, get_masks, output_selected_rate

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    """Load a CLIP model on CPU with NLPrompt design details."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    local_model_path = osp.join(osp.dirname(osp.dirname(__file__)), "clip", f"{backbone_name}.pt")

    if osp.isfile(local_model_path):
        model_path = local_model_path
    else:
        model_path = clip._download(clip._MODELS[backbone_name])

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
    """CLIP text encoder wrapper that maps prompts to text features."""
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
    """Learnable prompt context tokens for class names."""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        num_classes = len(classnames)
        num_ctx_tokens = cfg.TRAINER.NLPROMPT.N_CTX
        ctx_init_text = cfg.TRAINER.NLPROMPT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init_text:
            # use given words to initialize context vectors
            ctx_init_text = ctx_init_text.replace("_", " ")
            num_ctx_tokens = len(ctx_init_text.split(" "))
            prompt = clip.tokenize(ctx_init_text)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + num_ctx_tokens, :]
            prompt_prefix = ctx_init_text

        else:
            # random initialization
            if cfg.TRAINER.NLPROMPT.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(num_classes, num_ctx_tokens, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(num_ctx_tokens, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * num_ctx_tokens)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {num_ctx_tokens}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        class_names = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in class_names]
        prompts = [prompt_prefix + " " + name + "." for name in class_names]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + num_ctx_tokens :, :])  # CLS, EOS

        self.n_cls = num_classes
        self.n_ctx = num_ctx_tokens
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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = self.softmax(logits)
        p = p[torch.arange(p.shape[0]), targets]
        # Avoid undefined gradient for p == 0 by adding epsilon
        p += self.epsilon
        loss = (1 - p ** self.q) / self.q
        return torch.mean(loss)


class CustomCLIP(nn.Module):
    """CLIP wrapper with a learnable prompt module."""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, images):
        image_features = self.image_encoder(images.type(self.dtype))

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
    """Trainer for NLPrompt with optional OT-based label selection."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gce_loss = GeneralizedCrossEntropy(q=1.0)
        self.GCE_loss = self.gce_loss
        self.num_equal = []
        self.confident_rate = []
        self.clean_rate = []

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

    def _forward_backward(self, batch, loss_fn, loss_key, acc_key):
        images, labels, _ = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.NLPROMPT.PREC

        if prec == "amp":
            with autocast():
                logits = self.model(images)
                loss = loss_fn(logits, labels)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits = self.model(images)
            loss = loss_fn(logits, labels)
            self.model_backward_and_update(loss)

        return {
            loss_key: loss.item(),
            acc_key: compute_accuracy(logits, labels)[0].item(),
        }

    def forward_backward_ce(self, batch):
        return self._forward_backward(
            batch=batch,
            loss_fn=F.cross_entropy,
            loss_key="loss_x",
            acc_key="acc_x",
        )

    def forward_backward_mae(self, batch):
        return self._forward_backward(
            batch=batch,
            loss_fn=self.gce_loss,
            loss_key="loss_u",
            acc_key="acc_u",
        )

    def parse_batch_train(self, batch):
        images = batch["img"]
        labels = batch["label"]
        gt_labels = batch["gttarget"]
        images = images.to(self.device)
        labels = labels.to(self.device)
        gt_labels = gt_labels.to(self.device)
        return images, labels, gt_labels

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

    @staticmethod
    def _build_train_loader_iter(loader):
        if loader is None:
            return None, 0
        return iter(loader), len(loader)

    def _log_batch_progress(self, losses, batch_time, data_time, num_batches, loss_label):
        eta_seconds = batch_time.avg * (num_batches - self.batch_idx - 1)
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        info = [
            f"epoch [{self.epoch + 1}/{self.max_epoch}]",
            f"batch [{self.batch_idx + 1}/{num_batches}]",
            f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
            f"data {data_time.val:.3f} ({data_time.avg:.3f})",
            f"{loss_label} {losses}",
            f"lr {self.get_current_lr():.4e}",
            f"eta {eta}",
        ]
        print(" ".join(info))

    def _run_train_loader(
        self,
        loader_iter,
        num_batches,
        forward_fn,
        losses,
        batch_time,
        data_time,
        scalar_prefix,
        loss_label,
        end_time,
    ):
        for self.batch_idx in range(num_batches):
            try:
                batch = next(loader_iter)
            except StopIteration:
                break

            data_time.update(time.time() - end_time)
            loss_summary = forward_fn(batch)
            losses.update(loss_summary)
            batch_time.update(time.time() - end_time)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or num_batches < self.cfg.TRAIN.PRINT_FREQ:
                self._log_batch_progress(losses, batch_time, data_time, num_batches, loss_label)

            n_iter = self.epoch * (self.num_batches_x + self.num_batches_u) + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar(f"{scalar_prefix}/{name}", meter.avg, n_iter)

            end_time = time.time()

        return end_time

    @staticmethod
    def _delete_indices(data_source, indices):
        for index in sorted(indices, reverse=True):
            del data_source[index]

    def _compute_ot_budget(self, curriclum_epoch, begin_rate, curriclum_mode):
        if self.epoch < curriclum_epoch:
            budget, _ = curriculum_scheduler(
                self.epoch,
                curriclum_epoch,
                begin=begin_rate,
                end=1,
                mode=curriclum_mode,
            )
        else:
            budget = 1.0
        return budget

    def _get_ot_cfg(self):
        cfg = self.cfg.DATASET
        return {
            "reg_feat": cfg.REG_FEAT,
            "reg_lab": cfg.REG_LAB,
            "curriclum_epoch": cfg.CURRICLUM_EPOCH,
            "begin_rate": cfg.BEGIN_RATE,
            "curriclum_mode": cfg.CURRICLUM_MODE,
            "pmode": cfg.PMODE,
            "reg_e": getattr(cfg, "REG_E", 0.05),
        }

    @staticmethod
    def _print_ot_overview(gt_labels, argmax_plabels):
        print("before epoch:data num:", len(gt_labels))
        print(
            "before epoch:different number:",
            np.sum(gt_labels.cpu().numpy() != argmax_plabels.cpu().numpy()),
        )

    @staticmethod
    def _build_unlabeled_mask(argmax_plabels, noisy_labels, selected_mask):
        conf_l_mask, conf_u_mask, lowconf_u_mask = get_masks(
            argmax_plabels, noisy_labels, None, selected_mask
        )
        selected_rate_conf_l, _, _ = output_selected_rate(
            conf_l_mask, conf_u_mask, lowconf_u_mask
        )
        print("confident_label rate", selected_rate_conf_l)
        unlabeled_mask = torch.logical_or(conf_u_mask, lowconf_u_mask)
        return conf_l_mask, unlabeled_mask

    def _run_ot_pseudolabeling(self, budget, reg_feat, reg_lab, pmode, reg_e):
        with torch.no_grad():
            _, noisy_labels, gt_labels, selected_mask, _, argmax_plabels = OT_PL(
                self.model,
                self.train_loader_x,
                num_class=self.cfg.DATASET.num_class,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                budget=budget,
                reg_feat=reg_feat,
                reg_lab=reg_lab,
                Pmode=pmode,
                reg_e=reg_e,
                load_all=True,
            )
        return noisy_labels, gt_labels, selected_mask, argmax_plabels

    def _apply_ot_split(self, conf_l_mask, unlabeled_mask, pred_labels, gt_labels):
        conf_mask_np = conf_l_mask.cpu().numpy()
        unlabeled_mask_np = unlabeled_mask.cpu().numpy()
        pred_np = pred_labels.cpu().numpy()
        gt_np = gt_labels.cpu().numpy()

        clean_total = int(conf_mask_np.sum())
        clean_correct = int(np.sum((pred_np == gt_np) & conf_mask_np))
        noisy_total = int(unlabeled_mask_np.sum())
        noisy_correct = int(np.sum((pred_np == gt_np) & unlabeled_mask_np))

        clean_rate = clean_correct / clean_total
        self.clean_rate.append(clean_rate)

        confident_indices = np.nonzero(conf_mask_np)[0]
        unlabeled_indices = np.nonzero(unlabeled_mask_np)[0]

        # Backup loaders before deletion for restoration after the epoch.
        self.tmp_train_loader_x = copy.deepcopy(self.train_loader_x)
        self.train_loader_u = copy.deepcopy(self.train_loader_x)

        print("before: len(self.train)", len(self.train_loader_x.dataset.data_source))
        print("before: len of confident samples", len(confident_indices))
        print(f"clean_rate:{clean_rate}")
        print(f"noisy_correct:{noisy_correct}/{noisy_total}")

        self._delete_indices(self.train_loader_x.dataset.data_source, unlabeled_indices)
        print("after delete: len(clean_dataset)", len(self.train_loader_x.dataset.data_source))

        self._delete_indices(self.train_loader_u.dataset.data_source, confident_indices)
        print("after delete: len(noisy_dataset)", len(self.train_loader_u.dataset.data_source))

    def _restore_train_loaders(self):
        self.train_loader_x = copy.deepcopy(self.tmp_train_loader_x)
        self.train_loader_u = copy.deepcopy(self.tmp_train_loader_x)
        print("after epoch: len(clean dataset)", len(self.train_loader_x.dataset.data_source))
        print("after epoch: len(noisy dataset)", len(self.train_loader_u.dataset.data_source))

    def before_epoch(self):
        if not self.cfg.DATASET.USE_OT:
            return

        epoch_start = time.time()
        ot_cfg = self._get_ot_cfg()

        budget = self._compute_ot_budget(
            ot_cfg["curriclum_epoch"],
            ot_cfg["begin_rate"],
            ot_cfg["curriclum_mode"],
        )
        ot_start = time.time()
        noisy_labels, gt_labels, selected_mask, argmax_plabels = self._run_ot_pseudolabeling(
            budget=budget,
            reg_feat=ot_cfg["reg_feat"],
            reg_lab=ot_cfg["reg_lab"],
            pmode=ot_cfg["pmode"],
            reg_e=ot_cfg["reg_e"],
        )
        print(f"before epoch: OT_PL time {time.time() - ot_start:.2f}s")

        self._print_ot_overview(gt_labels, argmax_plabels)

        conf_l_mask, unlabeled_mask = self._build_unlabeled_mask(
            argmax_plabels,
            noisy_labels,
            selected_mask,
        )

        if conf_l_mask.sum().item() > 0:
            self._apply_ot_split(conf_l_mask, unlabeled_mask, argmax_plabels, gt_labels)
        print(f"before epoch total time {time.time() - epoch_start:.2f}s")

    def run_epoch(self):
        self.set_model_mode("train")
        losses_x = MetricMeter()
        losses_u = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        train_loader_x_iter, len_train_loader_x = self._build_train_loader_iter(self.train_loader_x)
        train_loader_u_iter, len_train_loader_u = self._build_train_loader_iter(self.train_loader_u)

        self.num_batches_x = len_train_loader_x
        self.num_batches_u = len_train_loader_u

        end = time.time()
        if train_loader_x_iter is not None:
            end = self._run_train_loader(
                loader_iter=train_loader_x_iter,
                num_batches=self.num_batches_x,
                forward_fn=self.forward_backward_ce,
                losses=losses_x,
                batch_time=batch_time,
                data_time=data_time,
                scalar_prefix="train_x",
                loss_label="loss_x",
                end_time=end,
            )

        if train_loader_u_iter is not None:
            self._run_train_loader(
                loader_iter=train_loader_u_iter,
                num_batches=self.num_batches_u,
                forward_fn=self.forward_backward_mae,
                losses=losses_u,
                batch_time=batch_time,
                data_time=data_time,
                scalar_prefix="train_u",
                loss_label="loss_u",
                end_time=end,
            )

        self.update_lr()

    def after_epoch(self):
        epoch_end_start = time.time()
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            test_start = time.time()
            curr_result = self.test(split="val")
            print(f"after epoch: test time {time.time() - test_start:.2f}s")
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
            save_start = time.time()
            self.save_model(self.epoch, self.output_dir)
            print(f"after epoch: save time {time.time() - save_start:.2f}s")
        
        if self.cfg.DATASET.USE_OT:
            self._restore_train_loaders()
        print(f"after epoch total time {time.time() - epoch_end_start:.2f}s")
