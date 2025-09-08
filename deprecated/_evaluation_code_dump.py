# ------------------------------------------------------------------------------------- OG method ---------------------------------------------------------------------------------
import torch

@torch.no_grad()
def spike_gated_modality_attribution(
    model,
    extras,
    feature_dims: dict,
    modalities_order=None,
    class_idx: int = 0,
    return_per_feature: bool = True,
):
    a     = extras['a']           # (T,B)
    s1    = extras['s1']          # (T,B,H)
    s2    = extras['s2']          # (T,B,H)
    x     = extras['x_fused']     # (T,B,F_total)
    den   = extras['z_den']       # (B,1)
    per_t = extras.get('per_t', None)

    T, B, H = s1.shape
    _, _, F_total = x.shape

    W_enc = model.enc.weight            # (H, F_total)
    W_fc  = model.fc.weight             # (H, H)
    W_out_full = model.readout.weight   # (C, H)
    b_out = model.readout.bias          # (C,)

    W_out = W_out_full[class_idx, :]    # (H,)
    G1 = W_enc * W_out.unsqueeze(1)     # (H, F_total)
    u  = torch.matmul(W_fc.t(), W_out)  # (H,)
    G2 = W_enc * u.unsqueeze(1)         # (H, F_total)

    a    = a.to(s1.dtype)
    x    = x.to(s1.dtype)
    den  = den.to(s1.dtype).clamp_min(1e-6)

    eff_w1 = torch.einsum('t b h, h f -> t b f', s1, G1)  # (T,B,F_total)
    eff_w2 = torch.einsum('t b h, h f -> t b f', s2, G2)  # (T,B,F_total)
    eff_w  = eff_w1 + eff_w2

    gate = (a / den.squeeze(1).unsqueeze(0)).unsqueeze(-1)  # (T,B,1)
    contrib_all = gate * x * eff_w                           # (T,B,F_total)

    if modalities_order is None:
        modalities_order = list(model.modalities)
    slices = {}
    start = 0
    for m in modalities_order:
        Fm = int(feature_dims[m])
        slices[m] = slice(start, start + Fm)
        start += Fm
    if start != F_total:
        raise ValueError(f"feature_dims sum ({start}) != F_total ({F_total})")

    contrib_by_modality = {}
    per_time_feature = {} if return_per_feature else None
    per_time_modality, per_modality, feature_saliency = {}, {}, {}

    for m in modalities_order:
        sl = slices[m]
        contrib_m = contrib_all[:, :, sl]                 # (T,B,Fm)
        contrib_by_modality[m] = contrib_m
        if return_per_feature:
            per_time_feature[m] = contrib_m
        per_time_modality[m] = contrib_m.sum(dim=2)       # (T,B)
        per_modality[m]      = per_time_modality[m].sum(dim=0)  # (B,)
        feature_saliency[m]  = contrib_m.sum(dim=0)       # (B,Fm)

    time_saliency = contrib_all.sum(dim=2)                # (T,B)

    bias = b_out[class_idx].expand(B) if b_out is not None else torch.zeros(B, dtype=contrib_all.dtype, device=contrib_all.device)
    logit_reconstruction = time_saliency.sum(dim=0) + bias

    if per_t is not None:
        target_logit = per_t.sum(dim=0)[:, class_idx] + b_out[class_idx]
    else:
        num = extras['z_num']                 # (B,H)
        z = num / den                         # (B,H)
        target_logit = torch.einsum('b h, h -> b', z, W_out) + b_out[class_idx]

    return {
        'contrib_all': contrib_all,                         # (T,B,F_total)
        'contrib_by_modality': contrib_by_modality,         # {m: (T,B,Fm)}
        'per_time_feature': per_time_feature,               # {m: (T,B,Fm)} or None
        'per_time_modality': per_time_modality,             # {m: (T,B)}
        'per_modality': per_modality,                       # {m: (B,)}
        'feature_saliency': feature_saliency,               # {m: (B,Fm)}
        'time_saliency': time_saliency,                     # (T,B)
        'bias': bias,                                       # (B,)
        'logit_reconstruction': logit_reconstruction,       # (B,)
        'target_logit': target_logit,                       # (B,)
        'slices': slices,                                   # {m: slice}
    }

#v2 method
def _safe_eps_like(x, eps=1e-6):
    return torch.full((x.shape[0],), eps, dtype=x.dtype, device=x.device)

@torch.no_grad()
def spike_gated_modality_attribution_v2(
    model, extras, feature_dims, modalities_order=None, class_idx=0, return_per_feature=True
):
    # --- same prelude as yours ...
    a     = extras['a']           # (T,B)
    s1    = extras['s1']          # (T,B,H)
    s2    = extras['s2']          # (T,B,H)
    x     = extras['x_fused']     # (T,B,F_total)
    den   = extras['z_den']       # (B,1)
    per_t = extras.get('per_t', None)

    T, B, H = s1.shape
    _, _, F_total = x.shape

    W_enc = model.enc.weight                  # (H, F_total)
    W_fc  = model.fc.weight                   # (H, H)
    W_out_full = model.readout.weight         # (C, H)
    b_out = model.readout.bias                # (C,)

    W_out = W_out_full[class_idx, :]          # (H,)
    G1 = W_enc * W_out.unsqueeze(1)           # (H, F_total)
    u  = torch.matmul(W_fc.t(), W_out)        # (H,)
    G2 = W_enc * u.unsqueeze(1)               # (H, F_total)

    a    = a.to(s1.dtype)
    x    = x.to(s1.dtype)
    den  = den.to(s1.dtype).clamp_min(1e-6)

    eff_w1 = torch.einsum('t b h, h f -> t b f', s1, G1)  # (T,B,F_total)
    eff_w2 = torch.einsum('t b h, h f -> t b f', s2, G2)  # (T,B,F_total)
    eff_w  = eff_w1 + eff_w2

    gate = (a / den.squeeze(1).unsqueeze(0)).unsqueeze(-1)  # (T,B,1)
    contrib_all = gate * x * eff_w                           # (T,B,F_total)

    # slices/modality aggregation (same as yours) ...
    if modalities_order is None:
        modalities_order = list(model.modalities)
    slices = {}
    start = 0
    for m in modalities_order:
        Fm = int(feature_dims[m]); slices[m] = slice(start, start + Fm); start += Fm
    if start != F_total:
        raise ValueError(f"feature_dims sum ({start}) != F_total ({F_total})")

    contrib_by_modality, per_time_feature = {}, {} if return_per_feature else None
    per_time_modality, per_modality, feature_saliency = {}, {}, {}

    for m in modalities_order:
        sl = slices[m]
        cm = contrib_all[:, :, sl]                         # (T,B,Fm)
        contrib_by_modality[m] = cm
        if return_per_feature: per_time_feature[m] = cm
        per_time_modality[m] = cm.sum(dim=2)               # (T,B)
        per_modality[m]      = per_time_modality[m].sum(dim=0)    # (B,)
        feature_saliency[m]  = cm.sum(dim=0)               # (B,Fm)

    time_saliency = contrib_all.sum(dim=2)                 # (T,B)

    # --- logit targets
    bias = b_out[class_idx].expand(B) if b_out is not None else torch.zeros(B, dtype=contrib_all.dtype, device=contrib_all.device)
    # your reconstruction w/o correction
    logit_reconstruction = time_saliency.sum(dim=0) + bias    # (B,)

    if per_t is not None:
        target_logit = per_t.sum(dim=0)[:, class_idx] + b_out[class_idx]
    else:
        num = extras['z_num']; W_out = W_out                   # (B,H) & (H,)
        z = num / den                                          # (B,H)
        target_logit = torch.einsum('b h, h -> b', z, W_out) + b_out[class_idx]  # (B,)

    # ---------- NEW: completeness correction ----------
    # Force sum_TF(contrib_all) + bias ≈ target_logit
    # scale per sample b: s_b = (target-bias)/(recon-bias)
    recon_wo_bias = (logit_reconstruction - bias)
    tgt_wo_bias   = (target_logit - bias)
    denom = torch.where(recon_wo_bias.abs() < 1e-6, _safe_eps_like(recon_wo_bias), recon_wo_bias)
    scale = (tgt_wo_bias / denom).clamp(min=-10.0, max=10.0)  # guard against blow-ups
    contrib_all = contrib_all * scale.view(1, B, 1)
    # recompute aggregates with the corrected contrib_all
    for m in modalities_order:
        sl = slices[m]
        cm = contrib_all[:, :, sl]
        contrib_by_modality[m] = cm
        if return_per_feature: per_time_feature[m] = cm
        per_time_modality[m] = cm.sum(dim=2)
        per_modality[m]      = per_time_modality[m].sum(dim=0)
        feature_saliency[m]  = cm.sum(dim=0)
    time_saliency = contrib_all.sum(dim=2)
    logit_reconstruction = time_saliency.sum(dim=0) + bias    # now matches target

    return {
        'contrib_all': contrib_all,
        'contrib_by_modality': contrib_by_modality,
        'per_time_feature': per_time_feature,
        'per_time_modality': per_time_modality,
        'per_modality': per_modality,
        'feature_saliency': feature_saliency,
        'time_saliency': time_saliency,
        'bias': bias,
        'logit_reconstruction': logit_reconstruction,
        'target_logit': target_logit,
        'slices': slices,
        'scale': scale,   # for debugging
    }

#----------------------------------------------------------------------------- code for evaluation og -------------------------------------------------------------------------------

import torch
import numpy as np

def _baseline_like_fused(x_fused, mode='zero', data_mean=None, data_std=None):
    """
    Baseline over the full fused vector (after encoders).
    x_fused: (T,B,F_total)
    """
    if mode == 'zero':
        return torch.zeros_like(x_fused)
    elif mode == 'mean':
        if data_mean is None:
            base = x_fused.mean(dim=(0,1), keepdim=True)  # (1,1,F_total)
        else:
            base = data_mean.view(1,1,-1).to(x_fused.device)
        return base.expand_as(x_fused)
    elif mode == 'noise':
        if data_std is None:
            std = x_fused.std(dim=(0,1), keepdim=True)
        else:
            std = data_std.view(1,1,-1).to(x_fused.device)
        return torch.randn_like(x_fused) * std
    else:
        raise ValueError("Unsupported fused baseline mode")

def _apply_mask_with_indices_fused(x0, indices, baseline_fused, mode='remove', mask_alpha=0.0):
    """
    x0: (T,1,F_total)
    indices: list/1D tensor of feature indices in [0..F_total-1]
    baseline_fused: (T,1,F_total)
    mode: 'remove' -> indices -> baseline; 'keep' -> all but indices -> baseline
    mask_alpha: soft mixing towards baseline (0=hard replace)
    """
    T, _, F = x0.shape
    mask = torch.zeros(F, dtype=torch.bool, device=x0.device)
    if len(indices) > 0:
        mask[torch.as_tensor(indices, device=x0.device, dtype=torch.long)] = True
    keep = ~mask if mode == 'remove' else mask
    keep_b = keep.view(1,1,F).expand(T,1,F)

    if mask_alpha == 0.0:
        xk = torch.where(keep_b, x0, baseline_fused)
    else:
        xk = x0.clone()
        unkept = ~keep_b
        xk[unkept] = mask_alpha * x0[unkept] + (1-mask_alpha) * baseline_fused[unkept]
    return xk

def _apply_mask_with_time_indices(x0, t_indices, baseline_fused, mode='remove', mask_alpha=0.0):
    """
    x0: (T,1,F_total)
    t_indices: list/1D tensor of time indices in [0..T-1]
    baseline_fused: (T,1,F_total)
    mode: 'remove' -> selected time indices -> baseline; 'keep' -> all but selected -> baseline
    mask_alpha: soft mixing toward baseline (0=hard replace)
    """
    T = x0.shape[0]
    mask = torch.zeros(T, dtype=torch.bool, device=x0.device)
    if len(t_indices) > 0:
        mask[torch.as_tensor(t_indices, device=x0.device, dtype=torch.long)] = True
    keep = ~mask if 'remove' in mode else mask
    keep_b = keep.view(T, 1, 1).expand(T, 1, x0.shape[2])

    if mask_alpha == 0.0:
        return torch.where(keep_b, x0, baseline_fused)
    else:
        xk = x0.clone()
        unkept = ~keep_b
        xk[unkept] = mask_alpha * x0[unkept] + (1 - mask_alpha) * baseline_fused[unkept]
        return xk

def _apply_mask_with_cell_indices(x0, tf_indices, baseline_fused, mode='remove', mask_alpha=0.0):
    """
    x0: (T,1,F_total)
    tf_indices: 1D tensor/list of flattened (t,f) indices in [0 .. T*F_total-1]
    baseline_fused: (T,1,F_total)
    mode: 'remove' -> selected cells -> baseline
          'keep'   -> all but selected cells -> baseline
    """
    T, _, F = x0.shape

    # Build a boolean mask of selected cells
    mask = torch.zeros((T, 1, F), dtype=torch.bool, device=x0.device)
    if len(tf_indices) > 0:
        idx = torch.as_tensor(tf_indices, device=x0.device, dtype=torch.long)
        t = (idx // F).clamp_(0, T - 1)
        f = (idx %  F).clamp_(0, F - 1)
        mask[t, 0, f] = True

    if 'remove' in mode:
        to_replace = mask
    else:  # 'keep' semantics
        to_replace = ~mask

    if 'keep' in mode and mask.sum() == 0:
        if mask_alpha == 0.0:
            return baseline_fused
        return mask_alpha * x0 + (1 - mask_alpha) * baseline_fused
    if 'remove' in mode and mask.sum() == 0:
        return x0

    if mask_alpha == 0.0:
        xk = x0.clone()
        xk[to_replace] = baseline_fused[to_replace]
    else:
        xk = x0.clone()
        xk[to_replace] = mask_alpha * x0[to_replace] + (1 - mask_alpha) * baseline_fused[to_replace]
    return xk

#move to device and ranking

def tree_map_tensors(x, fn):
    if torch.is_tensor(x):
        return fn(x)
    if isinstance(x, dict):
        return {k: tree_map_tensors(v, fn) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [tree_map_tensors(v, fn) for v in x]
        return type(x)(t) if not isinstance(x, tuple) else tuple(t)
    return x

def move_batch_to_device(batch, device, non_blocking=True):
    def _to_dev(t):
        return t.to(device, non_blocking=non_blocking) if torch.is_tensor(t) else t
    return tree_map_tensors(batch, _to_dev)

def _rank_features_1d(attr_1d, use_abs=False):
    s = attr_1d.abs() if use_abs else attr_1d
    return torch.argsort(s, descending=True)

#   ---------------------------------------------------running evaluation this is just the SGA method-------------------------------------------------------------- 

import torch
import numpy as np
from contextlib import nullcontext
from tqdm.auto import tqdm
import torch.nn.functional as F
from MMMFuse.evaluation import MetricFactory

# -------------------------
# Label & prob utilities
# -------------------------
def _extract_targets(batch, task_type: str, num_labels: int):
    """
    Returns:
      - y_true_ids:   (N,) int for multiclass
      - y_true_multi: (N,C) float 0/1 for binary (C=1) and multilabel (C>=2)
    Only one of these is used depending on task_type.
    """
    y = batch.get('labels', batch.get('y', None))
    if y is None:
        raise KeyError("Batch must contain 'labels' or 'y'.")

    if task_type == 'multiclass':
        # Accept (N,) class indices or one-hot (N,C)
        if y.dim() == 2 and y.size(1) == num_labels:
            return y.argmax(dim=1).long(), None
        return y.long().view(-1), None
    else:
        # binary/multilabel: expect (N,C) or (N,) -> make (N,C)
        if y.dim() == 1:
            y = y.view(-1, 1)
        if y.size(1) != num_labels:
            raise ValueError(f"Expected {num_labels} labels, got {y.size(1)}.")
        return None, y.float()


def _logits_to_probabilities(logits: torch.Tensor, task_type: str):
    """
    logits: (N,C) (C can be 1)
    Returns probabilities with same shape.
    """
    if task_type == 'multiclass':
        return F.softmax(logits, dim=-1)
    elif task_type in ('binary', 'multilabel'):
        return torch.sigmoid(logits)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def _compute_metrics(preds_logits: torch.Tensor,
                     targets_ids: torch.Tensor | None,
                     targets_multi: torch.Tensor | None,
                     task_type: str,
                     num_labels: int,
                     average: str = "macro"):
    """
    preds_logits: (N,C)
    For torchmetrics we pass probabilities; they will threshold internally if needed.
    Targets must be integer/bool for classification metrics.
    """
    preds_probs = _logits_to_probabilities(preds_logits, task_type)

    factory = MetricFactory(task_type=task_type, num_labels=num_labels, average=average)
    metrics = factory.get_metrics()

    results = {}
    with torch.no_grad():
        preds_cpu = preds_probs.detach().cpu()

        if task_type == 'multiclass':
            y_cpu = targets_ids.detach().cpu().long()
            for name, metric in metrics.items():
                results[name] = float(metric(preds_cpu, y_cpu))

        else:
            y_cpu = targets_multi.detach().cpu()
            if y_cpu.dtype.is_floating_point:
                y_cpu = y_cpu.long()

            if y_cpu.ndim == 2 and y_cpu.shape[1] == 1:
                y_cpu = y_cpu.squeeze(1)
                preds_cpu = preds_cpu.squeeze(1)

            for name, metric in metrics.items():
                results[name] = float(metric(preds_cpu, y_cpu))

    return results


@torch.no_grad()
def dataset_masking_eval_fused(
    model, dataloader, k_list,
    feature_dims, modalities_order=None,
    class_idx=0,
    mode='remove_topk',                  # 'remove_topk' or 'keep_topk' (string containing 'remove'/'keep')
    fused_baseline_mode='zero',          # 'zero'|'mean'|'noise'
    use_abs=False,
    label_getter=None,                   # optional; if None we read batch['labels' or 'y']
    amp_dtype=torch.float16,
    mask_alpha=0.0,
    random_control=False,
    rng=None,
    eval_what='feature',                 # 'feature' | 'time' | 'both' | 'joint'
    task_type='binary',                  # 'binary' | 'multiclass' | 'multilabel'
    num_labels=1,
    average='macro'
):
    """
    Faithfulness-by-masking at the fusion boundary.
    Computes **full logits** for each masked input and feeds them to **torchmetrics**.

    Returns a dict with metrics per-k for the requested eval(s) and (optionally) random controls.
    """
    device = next(model.parameters()).device
    model.eval()
    if rng is None:
        rng = np.random.default_rng()

    want_feat  = eval_what in ('feature', 'both')
    want_time  = eval_what in ('time', 'both')
    want_joint = eval_what == 'joint'

    # Collectors: for each k, we collect a list of logits (one per sample), then stack -> (N,C)
    def _collector_dict():
        return {int(k): [] for k in k_list}

    logits_feat_per_k      = _collector_dict() if want_feat  else None
    logits_feat_per_k_rand = _collector_dict() if want_feat and random_control else None

    logits_time_per_k      = _collector_dict() if want_time  else None
    logits_time_per_k_rand = _collector_dict() if want_time and random_control else None

    logits_joint_per_k      = _collector_dict() if want_joint else None
    logits_joint_per_k_rand = _collector_dict() if want_joint and random_control else None

    # Targets (we'll aggregate across all batches)
    targets_ids_all = []    # for multiclass (N,)
    targets_multi_all = []  # for binary/multilabel (N,C)

    # AMP context
    if amp_dtype is not None and device.type == 'cuda':
        amp_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    else:
        amp_ctx = nullcontext()

    for batch in tqdm(dataloader):
        batch = move_batch_to_device(batch, device)

        # Targets
        if label_getter is None:
            y_ids, y_multi = _extract_targets(batch, task_type, num_labels)
        else:
            # If you pass label_getter, it must return the right shape for your task
            y_out = label_getter(batch)
            if task_type == 'multiclass':
                y_ids = y_out.long().view(-1)
                y_multi = None
            else:
                y = y_out if y_out.dim() == 2 else y_out.view(-1, 1)
                y_multi = y.float()
                y_ids = None

        if task_type == 'multiclass':
            targets_ids_all.append(y_ids.detach().cpu())
        else:
            targets_multi_all.append(y_multi.detach().cpu())

        # one full forward to get fused + extras
        with amp_ctx:
            logits_full, extras = model.forward(inputs=batch['inputs'],
                                                present_mask=batch.get('present_mask', None),
                                                return_extras=True)

        # attribution
        attr = spike_gated_modality_attribution_v2(
            model, extras, feature_dims,
            modalities_order=(modalities_order or model.modalities),
            class_idx=class_idx,
            return_per_feature=False
        )

        x_fused = extras['x_fused']                 # (T,B,F_total)
        T, B, F_total = x_fused.shape
        base_fused = _baseline_like_fused(x_fused, fused_baseline_mode)

        contrib_all = attr['contrib_all']           # (T,B,F_total)
        time_scores = attr['time_saliency']         # (T,B)
        feature_scores = contrib_all.sum(dim=0)     # (B,F_total)

        for b in range(B):
            x0 = x_fused[:, b:b+1, :]      # (T,1,F_total)
            base_b = base_fused[:, b:b+1, :]

            # ========= FEATURE MASKING =========
            if want_feat:
                attr_full = feature_scores[b]              # (F_total,)
                order_f = _rank_features_1d(attr_full, use_abs=use_abs).to(device)

                Xs_feat, Xs_feat_rand = [], []
                for k in k_list:
                    k = int(k)
                    k_eff = min(k, F_total)

                    xk = _apply_mask_with_indices_fused(
                        x0,
                        order_f[:k_eff].tolist(),
                        base_b,
                        mode='remove' if 'remove' in mode else 'keep',
                        mask_alpha=mask_alpha
                    )
                    Xs_feat.append(xk)

                    if random_control:
                        rand_idx = rng.choice(F_total, size=k_eff, replace=False)
                        xr = _apply_mask_with_indices_fused(
                            x0, rand_idx, base_b,
                            mode='remove' if 'remove' in mode else 'keep',
                            mask_alpha=mask_alpha
                        )
                        Xs_feat_rand.append(xr)

                X_cat = torch.cat(Xs_feat, dim=1)  # (T,K,F_total)
                with amp_ctx:
                    logits_k = model.forward_from_fusion(X_cat, return_extras=False)  # (K,C)
                for i, k in enumerate(k_list):
                    logits_feat_per_k[int(k)].append(logits_k[i].detach().cpu())

                if random_control:
                    Xr_cat = torch.cat(Xs_feat_rand, dim=1)
                    with amp_ctx:
                        logits_r = model.forward_from_fusion(Xr_cat, return_extras=False)
                    for i, k in enumerate(k_list):
                        logits_feat_per_k_rand[int(k)].append(logits_r[i].detach().cpu())

            # ========= TIME MASKING =========
            if want_time:
                ts = time_scores[:, b]                      # (T,)
                order_t = _rank_features_1d(ts, use_abs=use_abs).to(device)

                Xs_time, Xs_time_rand = [], []
                for k in k_list:
                    k = int(k)
                    k_eff = min(k, T)

                    xk = _apply_mask_with_time_indices(
                        x0,
                        order_t[:k_eff].tolist(),
                        base_b,
                        mode='remove' if 'remove' in mode else 'keep',
                        mask_alpha=mask_alpha
                    )
                    Xs_time.append(xk)

                    if random_control:
                        rand_t = rng.choice(T, size=k_eff, replace=False)
                        xr = _apply_mask_with_time_indices(
                            x0, rand_t, base_b,
                            mode='remove' if 'remove' in mode else 'keep',
                            mask_alpha=mask_alpha
                        )
                        Xs_time_rand.append(xr)

                Xt_cat = torch.cat(Xs_time, dim=1)  # (T,K,F_total)
                with amp_ctx:
                    logits_k_t = model.forward_from_fusion(Xt_cat, return_extras=False)  # (K,C)
                for i, k in enumerate(k_list):
                    logits_time_per_k[int(k)].append(logits_k_t[i].detach().cpu())

                if random_control:
                    Xrt_cat = torch.cat(Xs_time_rand, dim=1)
                    with amp_ctx:
                        logits_r_t = model.forward_from_fusion(Xrt_cat, return_extras=False)
                    for i, k in enumerate(k_list):
                        logits_time_per_k_rand[int(k)].append(logits_r_t[i].detach().cpu())

            # ========= JOINT (cells) MASKING =========
            if want_joint:
                cells = contrib_all[:, b, :].reshape(-1)          # (T*F_total,)
                order_tf = _rank_features_1d(cells, use_abs=use_abs).to(device)

                Xs_joint, Xs_joint_rand = [], []
                TF = T * F_total
                for k in k_list:
                    k = int(k)
                    k_eff = min(k, TF)

                    xk = _apply_mask_with_cell_indices(
                        x0, order_tf[:k_eff].tolist(), base_b,
                        mode='remove' if 'remove' in mode else 'keep',
                        mask_alpha=mask_alpha
                    )
                    Xs_joint.append(xk)

                    if random_control:
                        rand_idx = rng.choice(TF, size=k_eff, replace=False)
                        xr = _apply_mask_with_cell_indices(
                            x0, rand_idx, base_b,
                            mode='remove' if 'remove' in mode else 'keep',
                            mask_alpha=mask_alpha
                        )
                        Xs_joint_rand.append(xr)

                Xj_cat = torch.cat(Xs_joint, dim=1)  # (T,K,F_total)
                with amp_ctx:
                    logits_k_j = model.forward_from_fusion(Xj_cat, return_extras=False)
                for i, k in enumerate(k_list):
                    logits_joint_per_k[int(k)].append(logits_k_j[i].detach().cpu())

                if random_control:
                    Xjr_cat = torch.cat(Xs_joint_rand, dim=1)
                    with amp_ctx:
                        logits_r_j = model.forward_from_fusion(Xjr_cat, return_extras=False)
                    for i, k in enumerate(k_list):
                        logits_joint_per_k_rand[int(k)].append(logits_r_j[i].detach().cpu())

    # Stack targets
    if task_type == 'multiclass':
        y_ids_all = torch.cat(targets_ids_all, dim=0)  # (N,)
        y_multi_all = None
    else:
        y_ids_all = None
        y_multi_all = torch.cat(targets_multi_all, dim=0)  # (N,C)

    # Aggregate metrics per-k
    def _finalize_block(logits_blocks, task_type, num_labels, average):
        out = {}
        for k in k_list:
            if len(logits_blocks[int(k)]) == 0:
                continue
            L = torch.stack(logits_blocks[int(k)], dim=0)  # (N,C)
            out[int(k)] = _compute_metrics(L, y_ids_all, y_multi_all, task_type, num_labels, average)
        return out

    results = {
        'k_list': [int(k) for k in k_list],
        'task_type': task_type,
        'num_labels': num_labels,
        'mode': mode,
        'eval_what': eval_what,
    }

    if want_feat:
        results['feature_metrics_per_k'] = _finalize_block(logits_feat_per_k, task_type, num_labels, average)
        if random_control:
            results['feature_metrics_per_k_random'] = _finalize_block(logits_feat_per_k_rand, task_type, num_labels, average)

    if want_time:
        results['time_metrics_per_k'] = _finalize_block(logits_time_per_k, task_type, num_labels, average)
        if random_control:
            results['time_metrics_per_k_random'] = _finalize_block(logits_time_per_k_rand, task_type, num_labels, average)

    if want_joint:
        results['joint_metrics_per_k'] = _finalize_block(logits_joint_per_k, task_type, num_labels, average)
        if random_control:
            results['joint_metrics_per_k_random'] = _finalize_block(logits_joint_per_k_rand, task_type, num_labels, average)

    return results

#---------------------------------------------------------------------- running this ---------------------------------------------------------------------------------------------------
# k_list = [0, 5, 10, 20, 50, 100, 150, 219, 347]
# # k_list = [0, 500, 1000, 2000, 4000, 8000, 10000]
# test_loader = datamod.test_dataloader()
# model.to(device)

# res_remove = dataset_masking_eval_fused(
#     model, test_loader, k_list, feature_dims,
#     mode='remove_topk',
#     fused_baseline_mode='mean',
#     use_abs=True,
#     random_control=True,  # <-- important
#     eval_what='feature',
#     task_type='multilabel',
#     num_labels=25,
#     class_idx=0
# )

# res_keep = dataset_masking_eval_fused(
#     model, test_loader, k_list, feature_dims,
#     mode='keep_topk',
#     fused_baseline_mode='mean',
#     use_abs=True,
#     random_control=True,
#     eval_what='feature',
#     task_type='multilabel',
#     num_labels=25,
#     class_idx=0
# )

# print("=== REMOVE top-k vs RANDOM ===")
# for k in k_list:
#     print(f"k={k:3d} | attr {res_remove['feature_metrics_per_k'][k]['AUROC']:.3f} "
#           f"| rand {res_remove['feature_metrics_per_k_random'][k]['AUROC']:.3f}")

# print("\n=== KEEP top-k vs RANDOM ===")
# for k in k_list:
#     print(f"k={k:3d} | attr {res_keep['feature_metrics_per_k'][k]['AUROC']:.3f} "
#           f"| rand {res_keep['feature_metrics_per_k_random'][k]['AUROC']:.3f}")
    
# ------------------------------------------------- This is overall evaluation, comparison with other grad based methods -----------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext
from tqdm.auto import tqdm
import torchmetrics.classification as tmc

def _fused_saliency_contrib(
    model,
    x_fused: torch.Tensor,           # (T,B,F)
    class_idx: int,
    method: str = 'grad',            # 'grad' | 'gradxinput' | 'ig'
    ig_steps: int = 32,
    ig_baseline: torch.Tensor | None = None,  # (T,B,F) if method='ig'
    amp_for_grads: bool = False,     # keep False for stability
) -> torch.Tensor:
    """
    Returns |contrib| shaped (T,B,F):
      - 'grad'       -> |∂y_c/∂x|
      - 'gradxinput' -> |x * ∂y_c/∂x|
      - 'ig'         -> |(x - baseline) * average_grad_along_path|
    """
    assert method in ('grad', 'gradxinput', 'ig')
    device = x_fused.device

    if method in ('grad', 'gradxinput'):
        with torch.enable_grad():
            x = x_fused.detach().clone().requires_grad_(True)   # (T,B,F)
            logits = model.forward_from_fusion(x)               # (B,C)
            target = logits[:, class_idx].sum()
            target.backward()
            grad = x.grad                                       # (T,B,F)

            if method == 'grad':
                contrib = grad.abs()
            else:
                contrib = (grad * x).abs()

        return contrib

    # Integrated Gradients
    with torch.enable_grad():
        x = x_fused.detach()
        if ig_baseline is None:
            ig_baseline = torch.zeros_like(x)
        delta = x - ig_baseline
        total_grad = torch.zeros_like(x)

        for i in range(ig_steps):
            alpha = (i + 0.5) / ig_steps
            x_step = (ig_baseline + alpha * delta).detach().clone().requires_grad_(True)
            logits = model.forward_from_fusion(x_step)
            target = logits[:, class_idx].sum()
            grads = torch.autograd.grad(target, x_step, retain_graph=False, create_graph=False)[0]
            total_grad += grads

        avg_grad = total_grad / ig_steps
        contrib = (avg_grad * delta).abs()
        return contrib

@torch.no_grad()
def dataset_masking_eval_fused(
    model, dataloader, k_list,
    feature_dims, modalities_order=None,
    class_idx=0,
    mode='remove_topk',                  # 'remove_topk' or 'keep_topk' (string containing 'remove'/'keep')
    fused_baseline_mode='zero',          # 'zero'|'mean'|'noise'
    use_abs=True,                        # apply |.| to attribution when rank_source='attr'
    label_getter=None,                   # optional; else uses _extract_targets
    amp_dtype=torch.float16,
    mask_alpha=0.0,
    random_control=False,
    rng=None,
    eval_what='feature',                 # 'feature' | 'time' | 'both' | 'joint'
    task_type='binary',                  # 'binary' | 'multiclass' | 'multilabel'
    num_labels=1,
    average='macro',
    # NEW: ranking source
    rank_source='attr',                  # 'attr' | 'grad' | 'gradxinput' | 'ig'
    ig_steps=32
):
    """
    Faithfulness-by-masking at the fusion boundary.
    Ranks units (time/feature/cell) by either your attribution ('attr') or gradient baselines
    ('grad', 'gradxinput', 'ig'). Computes full-task metrics with torchmetrics.

    Returns keys compatible with your print snippet, e.g.:
      results['time_metrics_per_k'][k]['AUROC']
      results['time_metrics_per_k_random'][k]['AUROC']
    """
    device = next(model.parameters()).device
    model.eval()
    if rng is None:
        rng = np.random.default_rng()

    want_feat  = eval_what in ('feature', 'both')
    want_time  = eval_what in ('time', 'both')
    want_joint = eval_what == 'joint'

    def _collector_dict():
        return {int(k): [] for k in k_list}

    logits_feat_per_k      = _collector_dict() if want_feat  else None
    logits_feat_per_k_rand = _collector_dict() if want_feat and random_control else None

    logits_time_per_k      = _collector_dict() if want_time  else None
    logits_time_per_k_rand = _collector_dict() if want_time and random_control else None

    logits_joint_per_k      = _collector_dict() if want_joint else None
    logits_joint_per_k_rand = _collector_dict() if want_joint and random_control else None

    # Targets across all batches
    targets_ids_all = []    # multiclass
    targets_multi_all = []  # binary/multilabel

    # AMP for forward (keep AMP off for grads/IG internally)
    if amp_dtype is not None and device.type == 'cuda':
        amp_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    else:
        amp_ctx = nullcontext()

    for batch in tqdm(dataloader):
        batch = move_batch_to_device(batch, device)

        # Targets
        if label_getter is None:
            y_ids, y_multi = _extract_targets(batch, task_type, num_labels)
        else:
            y_out = label_getter(batch)
            if task_type == 'multiclass':
                y_ids = y_out.long().view(-1)
                y_multi = None
            else:
                y = y_out if y_out.dim() == 2 else y_out.view(-1, 1)
                y_multi = y.long()
                y_ids = None

        if task_type == 'multiclass':
            targets_ids_all.append(y_ids.detach().cpu())
        else:
            targets_multi_all.append(y_multi.detach().cpu())

        # Forward once to get fused features + extras
        with amp_ctx:
            logits_full, extras = model.forward(inputs=batch['inputs'],
                                                present_mask=batch.get('present_mask', None),
                                                return_extras=True)

        x_fused = extras['x_fused']                 # (T,B,F)
        T, B, F_total = x_fused.shape
        base_fused = _baseline_like_fused(x_fused, fused_baseline_mode)

        # Attribution / Saliency for ranking
        if rank_source == 'attr':
            attr = spike_gated_modality_attribution_v2(
                model, extras, feature_dims,
                modalities_order=(modalities_order or model.modalities),
                class_idx=class_idx,
                return_per_feature=False
            )
            contrib_all = attr['contrib_all']       # (T,B,F)
            if use_abs:
                contrib_all = contrib_all.abs()
        elif rank_source in ('grad', 'gradxinput', 'ig'):
            ig_base = base_fused if rank_source == 'ig' else None
            # compute in full precision for stability
            with torch.cuda.amp.autocast(enabled=False):
                contrib_all = _fused_saliency_contrib(
                    model, x_fused.float(), class_idx,
                    method=rank_source, ig_steps=ig_steps, ig_baseline=ig_base.float() if ig_base is not None else None,
                    amp_for_grads=False
                ).to(x_fused.dtype)
        else:
            raise ValueError(f"Unknown rank_source: {rank_source}")

        # Aggregate to time/feature scores for ranking (already abs)
        time_scores    = contrib_all.sum(dim=2)   # (T,B)
        feature_scores = contrib_all.sum(dim=0)   # (B,F)

        for b in range(B):
            x0 = x_fused[:, b:b+1, :]      # (T,1,F)
            base_b = base_fused[:, b:b+1, :]

            # ========= FEATURE MASKING =========
            if want_feat:
                sF = feature_scores[b]                              # (F,)
                order_f = _rank_features_1d(sF, use_abs=False).to(device)

                Xs_feat, Xs_feat_rand = [], []
                for k in k_list:
                    k = int(k)
                    k_eff = min(k, F_total)

                    xk = _apply_mask_with_indices_fused(
                        x0,
                        order_f[:k_eff].tolist(),
                        base_b,
                        mode='remove' if 'remove' in mode else 'keep',
                        mask_alpha=mask_alpha
                    )
                    Xs_feat.append(xk)

                    if random_control:
                        rand_idx = rng.choice(F_total, size=k_eff, replace=False)
                        xr = _apply_mask_with_indices_fused(
                            x0, rand_idx, base_b,
                            mode='remove' if 'remove' in mode else 'keep',
                            mask_alpha=mask_alpha
                        )
                        Xs_feat_rand.append(xr)

                X_cat = torch.cat(Xs_feat, dim=1)  # (T,K,F)
                with amp_ctx:
                    logits_k = model.forward_from_fusion(X_cat, return_extras=False)  # (K,C)
                for i, k in enumerate(k_list):
                    logits_feat_per_k[int(k)].append(logits_k[i].detach().cpu())

                if random_control:
                    Xr_cat = torch.cat(Xs_feat_rand, dim=1)
                    with amp_ctx:
                        logits_r = model.forward_from_fusion(Xr_cat, return_extras=False)
                    for i, k in enumerate(k_list):
                        logits_feat_per_k_rand[int(k)].append(logits_r[i].detach().cpu())

            # ========= TIME MASKING =========
            if want_time:
                sT = time_scores[:, b]                              # (T,)
                order_t = _rank_features_1d(sT, use_abs=False).to(device)

                Xs_time, Xs_time_rand = [], []
                for k in k_list:
                    k = int(k)
                    k_eff = min(k, T)

                    xk = _apply_mask_with_time_indices(
                        x0,
                        order_t[:k_eff].tolist(),
                        base_b,
                        mode='remove' if 'remove' in mode else 'keep',
                        mask_alpha=mask_alpha
                    )
                    Xs_time.append(xk)

                    if random_control:
                        rand_t = rng.choice(T, size=k_eff, replace=False)
                        xr = _apply_mask_with_time_indices(
                            x0, rand_t, base_b,
                            mode='remove' if 'remove' in mode else 'keep',
                            mask_alpha=mask_alpha
                        )
                        Xs_time_rand.append(xr)

                Xt_cat = torch.cat(Xs_time, dim=1)  # (T,K,F)
                with amp_ctx:
                    logits_k_t = model.forward_from_fusion(Xt_cat, return_extras=False)  # (K,C)
                for i, k in enumerate(k_list):
                    logits_time_per_k[int(k)].append(logits_k_t[i].detach().cpu())

                if random_control:
                    Xrt_cat = torch.cat(Xs_time_rand, dim=1)
                    with amp_ctx:
                        logits_r_t = model.forward_from_fusion(Xrt_cat, return_extras=False)
                    for i, k in enumerate(k_list):
                        logits_time_per_k_rand[int(k)].append(logits_r_t[i].detach().cpu())

            # ========= JOINT (cells) MASKING =========
            if want_joint:
                TF = T * F_total
                cells = contrib_all[:, b, :].reshape(-1)            # (T*F,)
                order_tf = _rank_features_1d(cells, use_abs=False).to(device)

                Xs_joint, Xs_joint_rand = [], []
                for k in k_list:
                    k = int(k)
                    k_eff = min(k, TF)

                    xk = _apply_mask_with_cell_indices(
                        x0, order_tf[:k_eff].tolist(), base_b,
                        mode='remove' if 'remove' in mode else 'keep',
                        mask_alpha=mask_alpha
                    )
                    Xs_joint.append(xk)

                    if random_control:
                        rand_idx = rng.choice(TF, size=k_eff, replace=False)
                        xr = _apply_mask_with_cell_indices(
                            x0, rand_idx, base_b,
                            mode='remove' if 'remove' in mode else 'keep',
                            mask_alpha=mask_alpha
                        )
                        Xs_joint_rand.append(xr)

                Xj_cat = torch.cat(Xs_joint, dim=1)  # (T,K,F)
                with amp_ctx:
                    logits_k_j = model.forward_from_fusion(Xj_cat, return_extras=False)
                for i, k in enumerate(k_list):
                    logits_joint_per_k[int(k)].append(logits_k_j[i].detach().cpu())

                if random_control:
                    Xjr_cat = torch.cat(Xs_joint_rand, dim=1)
                    with amp_ctx:
                        logits_r_j = model.forward_from_fusion(Xjr_cat, return_extras=False)
                    for i, k in enumerate(k_list):
                        logits_joint_per_k_rand[int(k)].append(logits_r_j[i].detach().cpu())

    # Stack targets
    if task_type == 'multiclass':
        y_ids_all = torch.cat(targets_ids_all, dim=0)  # (N,)
        y_multi_all = None
    else:
        y_ids_all = None
        y_multi_all = torch.cat(targets_multi_all, dim=0)  # (N,C)

    # Aggregate metrics per-k
    def _finalize_block(logits_blocks, task_type, num_labels, average):
        out = {}
        for k in k_list:
            if len(logits_blocks[int(k)]) == 0:
                continue
            L = torch.stack(logits_blocks[int(k)], dim=0)  # (N,C)
            out[int(k)] = _compute_metrics(L, y_ids_all, y_multi_all, task_type, num_labels, average)
        return out

    results = {
        'k_list': [int(k) for k in k_list],
        'task_type': task_type,
        'num_labels': num_labels,
        'mode': mode,
        'eval_what': eval_what,
        'rank_source': rank_source,
    }

    if want_feat:
        results['feature_metrics_per_k'] = _finalize_block(logits_feat_per_k, task_type, num_labels, average)
        if random_control:
            results['feature_metrics_per_k_random'] = _finalize_block(logits_feat_per_k_rand, task_type, num_labels, average)

    if want_time:
        results['time_metrics_per_k'] = _finalize_block(logits_time_per_k, task_type, num_labels, average)
        if random_control:
            results['time_metrics_per_k_random'] = _finalize_block(logits_time_per_k_rand, task_type, num_labels, average)

    if want_joint:
        results['joint_metrics_per_k'] = _finalize_block(logits_joint_per_k, task_type, num_labels, average)
        if random_control:
            results['joint_metrics_per_k_random'] = _finalize_block(logits_joint_per_k_rand, task_type, num_labels, average)

    return results

# =========================
# Run ALL attribution rankers in one go
# =========================
@torch.no_grad()
def evaluate_all_attributions(
    model, dataloader, k_list, feature_dims,
    # what to mask/evaluate
    eval_what='time',                  # 'feature' | 'time' | 'both' | 'joint'
    # two passes: remove & keep
    do_remove=True,
    do_keep=True,
    # which rankers to compare
    rank_sources=('attr', 'gradxinput', 'grad', 'ig'),
    # shared kwargs forwarded to dataset_masking_eval_fused
    class_idx=0,
    fused_baseline_mode='mean',
    task_type='multilabel',
    num_labels=25,
    average='macro',
    use_abs=True,
    random_control=True,
    ig_steps=32,
    amp_dtype=torch.float16,
    mask_alpha=0.0,
    modalities_order=None,
    rng=None,
):
    """
    Runs dataset_masking_eval_fused for each rank_source in rank_sources,
    for both REMOVE and KEEP (configurable), and returns a nested dict:

    {
      'remove_topk': {
          'attr': {... evaluator results ...},
          'gradxinput': {...},
          ...
      },
      'keep_topk': {
          'attr': {...},
          ...
      }
    }
    """
    results = {}
    common = dict(
        feature_dims=feature_dims,
        eval_what=eval_what,
        class_idx=class_idx,
        fused_baseline_mode=fused_baseline_mode,
        task_type=task_type,
        num_labels=num_labels,
        average=average,
        use_abs=use_abs,
        random_control=random_control,
        ig_steps=ig_steps,
        amp_dtype=amp_dtype,
        mask_alpha=mask_alpha,
        modalities_order=modalities_order,
        rng=rng,
    )

    if do_remove:
        results['remove_topk'] = {}
        for rs in rank_sources:
            out = dataset_masking_eval_fused(
                model, dataloader, k_list,
                rank_source=rs,
                mode='remove_topk',
                **common
            )
            results['remove_topk'][rs] = out

    if do_keep:
        results['keep_topk'] = {}
        for rs in rank_sources:
            out = dataset_masking_eval_fused(
                model, dataloader, k_list,
                rank_source=rs,
                mode='keep_topk',
                **common
            )
            results['keep_topk'][rs] = out

    return results


# =========================
# Pretty-printer (AUROC) for your console summaries
# =========================
def print_auroc_summary(all_results, k_list, eval_what='time'):
    """
    Prints AUROC for attr/grad/ig vs RANDOM, for both remove/keep, mirroring your format.
    Works for eval_what in {'time','feature','joint'}.
    """
    key_map = {
        'time':    ('time_metrics_per_k', 'time_metrics_per_k_random'),
        'feature': ('feature_metrics_per_k', 'feature_metrics_per_k_random'),
        'joint':   ('joint_metrics_per_k', 'joint_metrics_per_k_random'),
    }
    metrics_key, metrics_rand_key = key_map[eval_what]

    def _maybe(val, default='n/a'):
        try:
            return f"{val:.3f}"
        except Exception:
            return default

    for mode in ('remove_topk', 'keep_topk'):
        if mode not in all_results:
            continue
        print(f"\n=== {mode.replace('_',' ').upper()} ===")
        # print header row
        rankers = list(all_results[mode].keys())
        header = "k".ljust(5)
        for rs in rankers:
            header += f"| {rs[:10].ljust(10)} "
            if metrics_rand_key in all_results[mode][rs]:
                header += f"| rand_{rs[:5].ljust(5)} "
        print(header)

        for k in k_list:
            line = f"{k:>3d}  "
            for rs in rankers:
                res = all_results[mode][rs]
                au_attr = res.get(metrics_key, {}).get(int(k), {}).get('AUROC', float('nan'))
                line += f"| { _maybe(au_attr) } "
                if metrics_rand_key in res and res[metrics_rand_key] is not None:
                    au_rand = res.get(metrics_rand_key, {}).get(int(k), {}).get('AUROC', float('nan'))
                    line += f"| { _maybe(au_rand) } "
            print(line)

# ----------------------------------------------- Code for running it ---------------------------------------------------------------------------------------------------
# test_loader = datamod.test_dataloader()
# model.to(device)
# k_list = [0, 5, 10, 20, 50, 100, 150, 219, 347]
# # k_list = [0, 5, 10, 15, 20, 25, 30, 35, 50]

# all_res = evaluate_all_attributions(
#     model, test_loader, k_list, feature_dims,
#     eval_what='feature',                # or 'feature' or 'joint'
#     do_remove=True,
#     do_keep=True,
#     # rank_sources=('attr','gradxinput','grad','ig'),
#     rank_sources=('attr', 'ig'),
#     class_idx=0,
#     fused_baseline_mode='mean',
#     task_type='multilabel',
#     num_labels=25,
#     random_control=True,             # prints the *_random block too
#     ig_steps=64
# )

# # Print like your snippet (AUROC)
# print_auroc_summary(all_res, k_list, eval_what='feature')

