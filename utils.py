import math

import ot
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


def curriculum_scheduler(t, T, begin=0, end=1, mode=None, func=None):
    """
    ratio \in [0,1]
    """
    pho = t / T
    if mode == "linear":
        ratio = pho
    elif mode == "exp":
        ratio = 1 - math.exp(-4 * pho)
    elif mode == "customize":
        ratio = func(t, T)
    budget = begin + ratio * (end - begin)
    return budget, pho


def output_selected_rate(conf_l_mask, conf_u_mask, lowconf_u_mask):
    selected_rate_conf_l = torch.sum(conf_l_mask) / conf_l_mask.size(0)
    selected_rate_conf_u = torch.sum(conf_u_mask) / conf_u_mask.size(0)
    selected_rate_lowconf_u = torch.sum(lowconf_u_mask) / lowconf_u_mask.size(0)
    return selected_rate_conf_l, selected_rate_conf_u, selected_rate_lowconf_u


def get_masks(argmax_plabels, noisy_labels, gt_labels, selected_mask):
    with torch.no_grad():
        equal_label_mask = torch.eq(noisy_labels, argmax_plabels)
        conf_l_mask = torch.logical_and(selected_mask, equal_label_mask)
        conf_u_mask = torch.logical_and(selected_mask, ~equal_label_mask)
        lowconf_u_mask = ~selected_mask
        return conf_l_mask, conf_u_mask, lowconf_u_mask


def _build_label_matrix(labels, num_class, sup_label=None, index=None):
    if sup_label is not None:
        return torch.eye(num_class, dtype=torch.float64)[sup_label[index]].cuda()
    return torch.eye(num_class, dtype=torch.float64)[labels].cuda()


def _select_probability_matrix(out, mode):
    if mode == "out":
        return out
    if mode == "logP":
        return F.log_softmax(out, dim=1)
    if mode == "softmax":
        return F.softmax(out, dim=1)
    raise ValueError(f"Unsupported Pmode: {mode}")


def _compute_pseudo_from_couplings(couplings):
    row_sum = torch.sum(couplings, 1).reshape((-1, 1))
    pseudo_labels = torch.div(couplings, row_sum)
    max_value, argmax_plabels = torch.max(couplings, axis=1)
    conf = max_value / (1 / couplings.size(0))
    conf = torch.clip(conf, min=0, max=1.0)
    return pseudo_labels, conf, argmax_plabels


def _compute_text_features(model):
    prompts = model.prompt_learner()
    tokenized_prompts = model.tokenized_prompts
    text_features = model.text_encoder(prompts, tokenized_prompts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()
    return text_features, logit_scale


def _compute_image_features(model, images):
    image_features = model.image_encoder(images.type(model.dtype))
    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features, image_features_norm


def curriculum_structure_aware_PL(features, P, top_percent, L=None,
                                  reg_feat=2., reg_lab=2., temp=1, device=None, version="fast",
                                  reg_e=0.01, reg_sparsity=None):
    """Select samples via OT coupling and return selection mask."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a = torch.ones((P.shape[0],), dtype=torch.float64).to(device) / P.shape[0]
    top_percent = min(torch.sum(a).item(), top_percent)
    b = torch.ones((P.shape[1],), dtype=torch.float64).to(device) / P.shape[1] * top_percent
    P = P.double()
    coupling = ot.sinkhorn(a, b, M=-P, reg=reg_e, numItermax=1000, stopThr=1e-6)
    total = features.size(0)

    max_values, _ = torch.max(coupling, 1)
    topk_num = int(total * top_percent)
    _, topk_indices = torch.topk(max_values, topk_num)

    selected_mask = torch.zeros((total,), dtype=torch.bool).cuda()
    selected_mask[topk_indices] = True
    return coupling, selected_mask


def OT_PL(model, eval_loader, num_class, batch_size, feat_dim=512, budget=1.0, sup_label=None,
          reg_feat=0.5, reg_lab=0.5, version="fast", Pmode="out", reg_e=0.01,
          reg_sparsity=None, load_all=True):
    """Compute OT-based pseudo labels and selection masks for a loader."""
    model.eval()

    all_pseudo_labels = torch.zeros((len(eval_loader.dataset), num_class), dtype=torch.float64).cuda()
    all_noisy_labels = torch.zeros((len(eval_loader.dataset),), dtype=torch.int64).cuda()
    all_gt_labels = torch.zeros((len(eval_loader.dataset),), dtype=torch.int64).cuda()
    all_selected_mask = torch.zeros((len(eval_loader.dataset)), dtype=torch.bool).cuda()
    all_conf = torch.zeros((len(eval_loader.dataset),), dtype=torch.float64).cuda()
    all_argmax_plabels = torch.zeros((len(eval_loader.dataset),), dtype=torch.int64).cuda()
    if load_all:
        feat_list = []
        out_list = []
        idx_list = []
        label_list = []

    with autocast():
        text_features, logit_scale = _compute_text_features(model)

    # loading given samples
    for batch_idx, batch in enumerate(eval_loader):
        inputs = batch["img"]
        labels = batch["label"]
        gt_labels = batch["gttarget"]
        index = batch["index"]

        with autocast():
            images = inputs.cuda()
            feat, feat_norm = _compute_image_features(model, images)
            logits = logit_scale * feat_norm @ text_features.t()
            out = logits.softmax(dim=-1)

        index = index.cuda()

        all_noisy_labels[index] = labels.cuda()
        all_gt_labels[index] = gt_labels.cuda()

        if load_all:
            feat_list.append(feat)
            out_list.append(out)
            idx_list.append(index)
            label_list.append(labels)
            continue

        L = _build_label_matrix(labels, num_class, sup_label=sup_label, index=index)
        P = _select_probability_matrix(out, Pmode)

        norm_feat = F.normalize(feat)
        couplings, selected_mask = curriculum_structure_aware_PL(
            norm_feat.detach(),
            P.detach(),
            top_percent=budget,
            L=L,
            reg_feat=reg_feat,
            reg_lab=reg_lab,
            version=version,
            reg_e=reg_e,
            reg_sparsity=reg_sparsity,
        )

        all_selected_mask[index] = selected_mask

        pseudo_labels, conf, argmax_plabels = _compute_pseudo_from_couplings(couplings)
        all_conf[index] = conf.double()
        all_pseudo_labels[index, :] = pseudo_labels
        all_argmax_plabels[index] = argmax_plabels

    if load_all:
        feat = torch.cat(feat_list, dim=0)
        out = torch.cat(out_list, dim=0)
        index = torch.cat(idx_list, dim=0).cuda()
        labels = torch.cat(label_list, dim=0)
        L = _build_label_matrix(labels, num_class, sup_label=sup_label, index=index)
        P = _select_probability_matrix(out, Pmode)

        norm_feat = F.normalize(feat)
        couplings, selected_mask = curriculum_structure_aware_PL(
            norm_feat.detach(),
            P.detach(),
            top_percent=budget,
            L=L,
            reg_feat=reg_feat,
            reg_lab=reg_lab,
            version=version,
            reg_e=reg_e,
            reg_sparsity=reg_sparsity,
        )
        all_selected_mask[index] = selected_mask
        pseudo_labels, conf, argmax_plabels = _compute_pseudo_from_couplings(couplings)
        all_conf[index] = conf
        all_pseudo_labels[index, :] = pseudo_labels
        all_argmax_plabels[index] = argmax_plabels

    return all_pseudo_labels, all_noisy_labels, all_gt_labels, all_selected_mask, all_conf, all_argmax_plabels
