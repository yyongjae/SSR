"""Command-conditioned anchor refinement and simulator-metric prediction.

All trajectories here use NAVSIM (x_forward, y_left, heading), including the
anchor archive. Simulator labels and candidate coordinates are detached from
the metric loss. The critic can still train the shared image/BEV representation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


METRIC_NAMES = ("NC", "DAC", "DDC", "EP", "TTC", "C", "score")


def load_plan_anchors(path: str, num_candidates: int, num_poses: int) -> torch.Tensor:
    """Load a train-only vocabulary produced by tools/build_para_ssr_anchors.py."""
    with np.load(Path(path).expanduser(), allow_pickle=False) as archive:
        anchors = np.asarray(archive["anchors"], dtype=np.float32)
        interval = float(archive["interval_length"])
        metadata = json.loads(str(archive["metadata_json"].item()))
    if metadata.get("source_split") != "train":
        raise ValueError("planning anchor vocabulary must come from the train split")
    expected_convention = {
        "format_version": 1,
        "coordinates": "navsim_current_ego_x_forward_y_left_heading",
        "pose_representation": "absolute_in_current_ego_frame",
        "includes_current_pose": False,
    }
    for key, expected in expected_convention.items():
        if key not in metadata or metadata[key] != expected:
            raise ValueError(
                f"anchor convention {key} must be {expected!r}, got {metadata.get(key)!r}; "
                "regenerate with tools/build_para_ssr_anchors.py"
            )
    if anchors.shape != (num_candidates, num_poses, 3):
        raise ValueError(
            f"anchor shape {anchors.shape} != {(num_candidates, num_poses, 3)}"
        )
    if not np.isclose(interval, 0.5) or not np.isfinite(anchors).all():
        raise ValueError("anchors must be finite NAVSIM poses sampled at 0.5 seconds")
    if np.unique(anchors[..., :2].reshape(num_candidates, -1), axis=0).shape[0] != num_candidates:
        raise ValueError("planning anchors must have distinct xy trajectories")
    anchors = anchors.copy()
    anchors[..., 2] = np.arctan2(np.sin(anchors[..., 2]), np.cos(anchors[..., 2]))
    return torch.from_numpy(anchors)


def poses_to_offsets(poses: torch.Tensor) -> torch.Tensor:
    offsets = torch.diff(poses, dim=-2, prepend=torch.zeros_like(poses[..., :1, :]))
    heading = torch.atan2(offsets[..., 2].sin(), offsets[..., 2].cos())
    return torch.cat((offsets[..., :2], heading.unsqueeze(-1)), dim=-1)


def offsets_to_poses(offsets: torch.Tensor) -> torch.Tensor:
    poses = offsets.cumsum(dim=-2)
    heading = torch.atan2(poses[..., 2].sin(), poses[..., 2].cos())
    return torch.cat((poses[..., :2], heading.unsqueeze(-1)), dim=-1)


def commanded_candidates(tensor: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    """[B, command, K, ...] -> [B, K, ...], including unknown command."""
    index = command.reshape(tensor.shape[0], -1).argmax(dim=-1)
    return tensor[torch.arange(tensor.shape[0], device=tensor.device), index]


class CandidateMetricHead(nn.Module):
    """Each candidate queries the common BEV to predict seven PDM scores.

    Candidate geometry is a detached input: BCE trains score accuracy, not the
    coordinate generator via a misleading 'make my score easier to predict'
    gradient. Metric-only gradients reach BEV iff detach_bev=False.
    """

    def __init__(self, embed_dims: int, num_heads: int, num_poses: int, detach_bev: bool):
        super().__init__()
        self.detach_bev = detach_bev
        self.pose_encoder = nn.Sequential(
            nn.Linear(num_poses * 4 + 8, embed_dims), nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.attention = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dims)
        self.output = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(),
            nn.Linear(embed_dims, len(METRIC_NAMES)),
        )

    def forward(self, bev, bev_pos, candidates, status):
        poses = candidates.detach()
        # Circular heading encoding; physical coordinates stay in NAVSIM axes.
        encoded = torch.cat((poses[..., :2] / 32.0,
                             poses[..., 2:3].sin(), poses[..., 2:3].cos()), dim=-1)
        status = status.detach().clone()
        status = torch.cat((status[..., :4], status[..., 4:] / 10.0), dim=-1)
        status = status[:, None].expand(-1, candidates.shape[1], -1)
        query = self.pose_encoder(torch.cat((encoded.flatten(-2), status), dim=-1))
        if self.detach_bev:
            bev, bev_pos = bev.detach(), bev_pos.detach()
        attended, _ = self.attention(query, bev + bev_pos, bev, need_weights=False)
        return self.output(self.norm(query + attended))


def rank_candidates(candidates, imitation_logits, metric_logits,
                    imitation_weight: float, metric_weight: float):
    """Select only among the active command's K candidates, without GT."""
    score = (imitation_weight * F.log_softmax(imitation_logits, dim=-1)
             + metric_weight * F.logsigmoid(metric_logits[..., -1]))
    best = score.argmax(dim=-1)
    trajectory = candidates[torch.arange(candidates.shape[0], device=candidates.device), best]
    return trajectory, best, score


def candidate_imitation_loss(predictions, targets, heading_weight: float):
    """Anchor assignment + winner refinement, preserving baseline plan scale.

    Assignment uses immutable anchor xy (WoTE-style) so every anchor owns a
    stable region of the data, rather than moving all targets to one regressor.
    The winner regression is averaged over B*commands*T*D, never over K.
    """
    from ..para_ssr_loss import compute_plan_loss

    offsets = predictions["candidate_offsets"]  # B,commands,K,T,3
    command = targets["command"]
    mask = targets["trajectory_mask"]
    anchors = predictions["plan_anchors"]
    gt_poses = offsets_to_poses(targets["trajectory_offsets"])
    distance = (anchors[None, ..., :2] - gt_poses[:, None, :, :2]).square().sum(-1)
    distance = (distance * mask[:, None]).sum(-1) / mask.sum(-1, keepdim=True).clamp_min(1)
    winner = distance.detach().argmin(-1)
    batch = torch.arange(offsets.shape[0], device=offsets.device)
    # Choose the same anchor index from each command branch; compute_plan_loss
    # applies the one-hot command mask and the original four-mode denominator.
    winner_offsets = offsets.permute(0, 2, 1, 3, 4)[batch, winner]
    regression, metrics = compute_plan_loss(
        winner_offsets, targets["trajectory_offsets"], mask, command, heading_weight
    )
    logits = predictions["candidate_logits"]
    valid = (mask.sum(-1) > 0).to(logits.dtype)
    classification = (F.cross_entropy(logits, winner, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    with torch.no_grad():
        candidates = predictions["trajectory_candidates"]
        ade = torch.linalg.vector_norm(candidates[..., :2] - gt_poses[:, None, :, :2], dim=-1)
        ade = (ade * mask[:, None]).sum(-1) / mask.sum(-1, keepdim=True).clamp_min(1)
        selected_ade = ade[batch, predictions["selected_candidate"]]
        metrics["candidate/oracle_ade"] = (ade.min(-1).values * valid).sum() / valid.sum().clamp_min(1)
        metrics["candidate/selected_ade"] = (selected_ade * valid).sum() / valid.sum().clamp_min(1)
        metrics["candidate/cls_accuracy"] = ((logits.argmax(-1) == winner).float() * valid).sum() / valid.sum().clamp_min(1)
        if candidates.shape[1] > 1:
            # No quadratic batch-wide distance matrix needed for diversity logs.
            spread = candidates[..., :2].std(dim=1, unbiased=False).mean()
            metrics["candidate/xy_spread"] = spread
    return regression, classification, metrics


def candidate_metric_loss(logits, labels, weights):
    if logits.shape != labels.shape or logits.shape[-1] != len(METRIC_NAMES):
        raise ValueError(f"metric logits/labels must be matching [B,K,7], got {logits.shape}/{labels.shape}")
    labels = labels.detach().to(logits)
    if not torch.isfinite(labels).all() or (labels < 0).any() or (labels > 1).any():
        raise ValueError("PDM metric targets must be finite in [0,1]")
    # NC/DDC values of0.5 stay soft labels; do not change evaluation semantics.
    terms = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean((0, 1))
    loss = (terms * logits.new_tensor(weights)).sum()
    logs = {f"loss_metric/{name}": terms[i].detach() for i, name in enumerate(METRIC_NAMES)}
    with torch.no_grad():
        logs.update({f"metric_target/{name}": labels[..., i].mean() for i, name in enumerate(METRIC_NAMES)})
        logs["metric/score_mae"] = (logits[..., -1].sigmoid() - labels[..., -1]).abs().mean()
    return loss, logs
