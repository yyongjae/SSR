"""Losses of the factorized planner and its two auxiliary BEV heads.

Planning terms (all go into ``task_losses["plan"]`` of para_ssr_loss.py):

    path      soft CE of the path logits against softmax(-d / tau_path), d = mean squared
              distance between the path anchor and the human path (resampled at the same
              arc-length stations, up to the human's travelled length)
    speed     soft CE of the marginal speed logits against softmax(-d / tau_speed), d = mean
              squared difference of cumulative distance to the human profile
    score     BCE of the NC / DAC / TTC / C logits and the EP logit of every fine candidate
              against the proxy labels (fact/proxy.py)
    list      listwise CE: softmax(PDMS_label / tau_list) vs softmax(log PDMS_pred / tau_list)
    margin    hinge keeping every unsafe candidate's NC / DAC logit ``margin`` below the
              best safe candidate's (FPD-Drive's safety-aware correction)
    im        soft CE of the imitation logit against softmax(-d_min / tau_im), d_min = squared
              distance to the nearest target among {human} + {exact-label anchors with
              PDMS >= target_pdms} (multi-target imitation)
    offset    L1 of the refined top-k candidate nearest the human trajectory (WTA)
    extra     the score / list / margin terms on the 256 WoTE anchors with their EXACT
              official labels (targets ``sim_reward`` [B, 5, 256], SIM_KEYS order)

Auxiliary heads: ``fact_sdf_pred`` L1 against ``fact_sdf_bev`` (task "map") and
``fact_occ_pred`` BCE against ``fact_occ_bev`` (task "det").
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .proxy import composite, proxy_labels
from .vocab import cumulative_distance, resample_path


def _soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


def _bce(logit: torch.Tensor, label: torch.Tensor, weight=None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit.float(), label.float(), weight=weight)


def _listwise(pdms_pred: torch.Tensor, pdms_label: torch.Tensor, tau: float) -> torch.Tensor:
    target = torch.softmax(pdms_label.float() / tau, dim=-1)
    logp = torch.log_softmax(torch.log(pdms_pred.float() + 1e-6) / tau, dim=-1)
    return -(target * logp).sum(-1).mean()


def _margin(logits: torch.Tensor, nc: torch.Tensor, dac: torch.Tensor, pdms: torch.Tensor, margin: float) -> torch.Tensor:
    """logits ``[B, N, 4]`` (nc, dac, ttc, c); labels ``[B, N]``.  Hinge for unsafe candidates
    against the best safe candidate of their scene."""
    safe = (nc > 0) & (dac > 0)
    has_safe = safe.any(dim=1)
    if not has_safe.any():
        return logits.sum() * 0.0
    best = torch.where(safe, pdms, torch.full_like(pdms, -1.0)).argmax(dim=1)        # [B]
    rows = torch.arange(logits.shape[0], device=logits.device)
    ref = logits[rows, best].float()                                                  # [B, 4]
    terms = []
    for col, lab in ((0, nc), (1, dac)):
        unsafe = (lab <= 0) & has_safe.view(-1, 1)
        if unsafe.any():
            gap = ref[:, col].view(-1, 1) - logits[..., col].float()
            terms.append((F.relu(margin - gap) * unsafe.float()).sum() / unsafe.float().sum())
    if not terms:
        return logits.sum() * 0.0
    return sum(terms) / len(terms)


def _score_terms(logits, ep_logit, labels: Dict[str, torch.Tensor], cfg, prefix: str, logs: Dict) -> torch.Tensor:
    bce = (_bce(logits[..., 0], labels["nc"]) + _bce(logits[..., 1], labels["dac"])
           + _bce(logits[..., 2], labels["ttc"]) + _bce(logits[..., 3], labels["comfort"])
           + _bce(ep_logit, labels["ep"]))
    p = torch.sigmoid(logits.float())
    pdms_pred = composite(p[..., 0], p[..., 1], torch.sigmoid(ep_logit.float()), p[..., 2], p[..., 3])
    lst = _listwise(pdms_pred, labels["pdms"], float(cfg.fact_tau_list))
    mar = _margin(logits, labels["nc"], labels["dac"], labels["pdms"], float(cfg.fact_margin))
    logs[f"{prefix}/bce"] = bce.detach()
    logs[f"{prefix}/list"] = lst.detach()
    logs[f"{prefix}/margin"] = mar.detach()
    with torch.no_grad():
        logs[f"{prefix}/label_safe_frac"] = ((labels["nc"] > 0) & (labels["dac"] > 0)).float().mean()
        logs[f"{prefix}/label_pdms_max"] = labels["pdms"].amax(dim=1).mean()
        sel = pdms_pred.argmax(dim=1)
        logs[f"{prefix}/sel_label_pdms"] = labels["pdms"].gather(1, sel.unsqueeze(1)).mean()
        logs[f"{prefix}/nc_acc"] = ((p[..., 0] > 0.5).float() == labels["nc"]).float().mean()
        logs[f"{prefix}/dac_acc"] = ((p[..., 1] > 0.5).float() == labels["dac"]).float().mean()
    return (float(cfg.fact_w_score) * bce + float(cfg.fact_w_list) * lst + float(cfg.fact_w_margin) * mar)


def fact_plan_losses(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                     features: Dict[str, torch.Tensor], cfg, planner) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    logs: Dict[str, torch.Tensor] = {}
    human = targets["trajectory"].float()                                   # [B, T, 3]
    B, T, _ = human.shape
    cand = predictions["fact_cand"]                                          # [B, N, T, 3]
    N = cand.shape[1]
    v0 = features["status_feature"][:, cfg.num_navi_cmd].float()

    # ---- stage-1/2 imitation of the human path and speed profile
    paths = planner.paths                                                    # [P, S+1, 2]
    S = paths.shape[1] - 1
    h_path, h_len = resample_path(human[..., :2], S, planner.station_m)      # [B, S+1, 2], [B]
    n_use = (h_len / planner.station_m).ceil().long().clamp(min=5, max=S)    # stations to compare
    st_mask = (torch.arange(1, S + 1, device=human.device).view(1, S) <= n_use.view(B, 1)).float()
    d_path = ((paths[None, :, 1:] - h_path[:, None, 1:]) ** 2).sum(-1)       # [B, P, S]
    d_path = (d_path * st_mask.view(B, 1, S)).sum(-1) / st_mask.sum(-1).view(B, 1)
    path_loss = _soft_ce(predictions["fact_path_logit"], torch.softmax(-d_path / float(cfg.fact_tau_path), dim=-1))
    h_speed = cumulative_distance(human[..., :2])                            # [B, T]
    d_speed = ((planner.speeds[None] - h_speed[:, None]) ** 2).mean(-1)      # [B, V]
    speed_loss = _soft_ce(predictions["fact_speed_logit"], torch.softmax(-d_speed / float(cfg.fact_tau_speed), dim=-1))
    with torch.no_grad():
        logs["fact/path_gt_rank"] = (predictions["fact_path_logit"] > predictions["fact_path_logit"].gather(
            1, d_path.argmin(1, keepdim=True))).float().sum(1).mean()
        logs["fact/speed_gt_rank"] = (predictions["fact_speed_logit"] > predictions["fact_speed_logit"].gather(
            1, d_speed.argmin(1, keepdim=True))).float().sum(1).mean()

    # ---- fine candidates: proxy labels
    labels = proxy_labels(cand, targets, tuple(cfg.plan_map_extent), v0)
    score_loss = _score_terms(predictions["fact_logits"], predictions["fact_ep_logit"], labels, cfg, "fact_proxy", logs)

    # ---- imitation (multi-target) and the offset head
    tgt = [human[..., :2].unsqueeze(1)]                                       # list of [B, M_i, T, 2]
    anchors = None
    if "sim_reward" in targets and "fact_extra_logits" in predictions and planner.extra_anchors is not None:
        anchors = planner.extra_anchors                                       # [M, T, 3]
        sr = targets["sim_reward"].float()                                    # [B, 5, M]  NC DAC EP TTC C
        exact = {"nc": sr[:, 0], "dac": sr[:, 1], "ep": sr[:, 2], "ttc": sr[:, 3], "comfort": sr[:, 4]}
        exact["pdms"] = composite(exact["nc"], exact["dac"], exact["ep"], exact["ttc"], exact["comfort"])
        valid = targets["sim_reward_valid"].float().view(B, 1)
        good = (exact["pdms"] >= float(cfg.fact_target_pdms)) & (valid > 0)   # [B, M]
        a_xy = anchors[None, :, :, :2].expand(B, -1, -1, -1)
        tgt.append(torch.where(good[:, :, None, None], a_xy, torch.full_like(a_xy, 1e4)))
    tgt = torch.cat(tgt, dim=1)                                               # [B, M, T, 2]
    d_im = ((cand[..., :2].unsqueeze(2) - tgt.unsqueeze(1)) ** 2).sum(-1).mean(-1)      # [B, N, M]
    d_im = d_im.amin(dim=2)
    im_loss = _soft_ce(predictions["fact_im_logit"], torch.softmax(-d_im / float(cfg.fact_tau_im), dim=-1))

    refined = predictions["fact_topk_traj"][..., :2].float()                  # [B, k, T, 2]
    d_ref = ((refined - human[:, None, :, :2]) ** 2).sum(-1).mean(-1)         # [B, k]
    win = d_ref.argmin(dim=1)
    rows = torch.arange(B, device=human.device)
    offset_loss = F.l1_loss(refined[rows, win], human[..., :2])

    total = (float(cfg.fact_w_path) * path_loss + float(cfg.fact_w_speed) * speed_loss + score_loss
             + float(cfg.fact_w_im) * im_loss + float(cfg.fact_w_offset) * offset_loss)
    logs.update({"fact/path": path_loss.detach(), "fact/speed": speed_loss.detach(),
                 "fact/im": im_loss.detach(), "fact/offset": offset_loss.detach()})

    # ---- extra anchors with exact labels
    if anchors is not None:
        valid = targets["sim_reward_valid"].float()
        if valid.sum() > 0:
            keep = valid > 0
            ex = {k: v[keep] for k, v in exact.items()}
            extra_loss = _score_terms(predictions["fact_extra_logits"][keep], predictions["fact_extra_ep_logit"][keep],
                                      ex, cfg, "fact_exact", logs)
            total = total + float(cfg.fact_w_extra) * extra_loss
    with torch.no_grad():
        sel = predictions["fact_topk_index"][:, 0]
        logs["fact/sel_proxy_pdms"] = labels["pdms"].gather(1, sel.unsqueeze(1)).mean()
        logs["fact/oracle_proxy_pdms"] = labels["pdms"].amax(dim=1).mean()
        logs["fact/human_l2_top1"] = d_ref[rows, 0].sqrt().mean()
    return total, logs


def fact_aux_losses(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], cfg) -> Dict[str, torch.Tensor]:
    """-> {"map": sdf L1, "det": occupancy BCE} (unweighted)."""
    out: Dict[str, torch.Tensor] = {}
    clip = float(cfg.plan_map_clip)
    if "fact_sdf_pred" in predictions:
        gt = targets["fact_sdf_bev"].float().unsqueeze(1) / clip
        out["map"] = F.l1_loss(predictions["fact_sdf_pred"].float() / clip, gt)
    if "fact_occ_logit" in predictions:
        gt = targets["fact_occ_bev"].float()
        pos_w = torch.tensor(float(cfg.fact_occ_pos_weight), device=gt.device)
        out["det"] = F.binary_cross_entropy_with_logits(predictions["fact_occ_logit"].float(), gt, pos_weight=pos_w)
    return out
