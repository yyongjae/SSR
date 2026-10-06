"""Is the gap between the chosen candidate and the best of its top 3 predictable at all?

For every scene the model's own top-3 candidates are taken (ranked by the rule it actually deploys)
and a small re-ranker is trained to pick the winner, with increasing information:

  rank1        the model's own choice                      (baseline)
  scores       + its five predicted sub-scores per candidate
  geometry     + the candidate trajectory itself
  entities     + the perception entities (GT or the model's own), attended like the critic
  execution    + how the candidate changes when it is driven: TOAD's inverse kinematics ->
               comfort clamp -> bicycle rollout, i.e. what the tracker can actually follow
  oracle       the best of the three                        (ceiling)

If nothing beats rank1, the remaining gap is not predictable from these inputs.

Labels are the official navtest ones (scripts/evaluation/label_anchors.py, validated against the
evaluation csv), so the numbers are EPDMS (without the two-frame extended comfort, which is not
defined per candidate) and not the WoTE training labels.

    python tools/plan_v2/rerank_probe.py --labels-npz data/planning_vb/navtest_anchor_scores.npz \
        --cache data/critic_cache_navtest --pred-cache data/critic_cache_pred_navtest \
        --dump work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz --out work_dirs/rerank [--gt-entities]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from clean_compare import ENT, EXTRA, INT, epdms, load_cache

from navsim.agents.para_ssr.modules.entity_critic import EntityCritic, mlp
from navsim.agents.para_ssr.modules.kinematics import bicycle_rollout, clamp_controls, poses_to_controls

K = 3


class Probe(Dataset):
    """Keeps the caches as the concatenated arrays they are loaded as -- one dict per scene needs
    over 100 GB for the navtrain cache, which is what an earlier version of this file did."""

    def __init__(self, labels_npz, cache, pred_cache, dump, anchors, use_gt_entities):
        z = np.load(labels_npz, allow_pickle=True)
        self.cols = [str(c) for c in z["columns"]]
        lab_at = {str(t): i for i, t in enumerate(z["tokens"])}
        d = np.load(dump)
        d_at = {str(t): i for i, t in enumerate(d["tokens"])}
        gt_at, self.gt = load_cache(cache, ENT + INT + EXTRA)
        pred_at, self.pred = load_cache(pred_cache, ENT + INT) if pred_cache else ({}, {})
        tokens = sorted(set(lab_at) & set(d_at) & set(gt_at) & (set(pred_at) if pred_at else set(gt_at)))
        self.tokens = tokens
        self.gi = np.array([gt_at[t] for t in tokens])
        self.pi = np.array([pred_at[t] for t in tokens]) if pred_at else self.gi
        self.use_gt = use_gt_entities or not pred_at
        li = np.array([lab_at[t] for t in tokens])
        di = np.array([d_at[t] for t in tokens])
        self.true = epdms(z["scores"][li], self.cols)[:, :-2]            # [N, 256] official EPDMS
        self.sim = d["sim_rewards"][di].astype(np.float32)               # [N, 5, 256]
        self.top = np.argsort(-d["plan_final_rewards"][di].astype(np.float32), axis=1)[:, :K]
        self.anchors = np.load(anchors).astype(np.float32)
        self.offset = d["trajectory_offset"][di].astype(np.float32)      # refined candidates
        print(f"{len(tokens)} 장면, 인지는 {'GT' if self.use_gt else 'v2 예측'}", flush=True)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        top = self.top[i]
        src, si = (self.gt, self.gi[i]) if self.use_gt else (self.pred, self.pi[i])
        out = {k: torch.from_numpy(np.asarray(src[k][si]).astype(np.int64 if k in INT else np.float32))
               for k in ENT + INT}
        for k in EXTRA:
            out[k] = torch.from_numpy(self.gt[k][self.gi[i]].astype(np.float32))
        out["traj"] = torch.from_numpy(self.anchors[top] + self.offset[i][top])      # [K, 8, 3] driven
        out["scores"] = torch.from_numpy(self.sim[i][:, top].T.copy())               # [K, 5]
        out["true"] = torch.from_numpy(self.true[i][top].copy())                     # [K] official EPDMS
        return out


def execution_features(traj, ego_status):
    """What the tracker can actually follow (TOAD kinematics) vs the nominal candidate."""
    v0 = ego_status[:, :1].clamp(min=0).expand(-1, traj.shape[1])                 # vx
    ctrl = clamp_controls(poses_to_controls(traj, v0, 0.5))
    driven, _ = bicycle_rollout(ctrl, v0, 0.5)
    d = (driven[..., :2] - traj[..., :2]).norm(dim=-1)                            # [B, K, T]
    dh = (driven[..., 2] - traj[..., 2]).abs()
    return torch.cat([d.mean(-1, keepdim=True), d.max(-1, keepdim=True).values,
                      dh.mean(-1, keepdim=True), ctrl[..., 0].abs().max(-1, keepdim=True).values,
                      ctrl[..., 1].abs().max(-1, keepdim=True).values], dim=-1)   # [B, K, 5]


class ReRanker(nn.Module):
    def __init__(self, anchors_file, use_scores=True, use_geometry=True, use_entities=False,
                 use_execution=False, embed=128):
        super().__init__()
        self.use = dict(scores=use_scores, geometry=use_geometry, entities=use_entities, execution=use_execution)
        d = 5 * use_scores + 24 * use_geometry + 5 * use_execution
        self.inp = mlp(max(d, 1), embed, embed)
        self.entity = EntityCritic(anchors_file, embed_dims=embed, num_layers=2) if use_entities else None
        self.head = mlp(embed + (embed if use_entities else 0), embed, 1)

    def forward(self, b):
        traj = b["traj"]
        parts = []
        if self.use["scores"]:
            parts.append(b["scores"])
        if self.use["geometry"]:
            parts.append(traj.flatten(2))
        if self.use["execution"]:
            parts.append(execution_features(traj, b["ego_status"]))
        h = self.inp(torch.cat(parts, dim=-1) if parts else traj.new_zeros(*traj.shape[:2], 1))
        if self.entity is not None:
            q = self.entity.cluster_encoder(self.entity.traj_mlp(traj.flatten(2)))
            q = q + self.entity.ego_mlp(torch.cat([b["ego_status"], b["command"]], -1)).unsqueeze(1)
            mem, pad = self.entity.tokens(b)
            for layer in self.entity.layers:
                q = layer(q, mem, memory_key_padding_mask=pad)
            h = torch.cat([h, self.entity.norm(q)], dim=-1)
        return self.head(h).squeeze(-1)                                            # [B, K] logits


def run(name, model, tr, va, device, epochs, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        for b in tr:
            b = {k: v.to(device) for k, v in b.items()}
            loss = nn.functional.cross_entropy(model(b), b["true"].argmax(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    got, best, first = [], [], []
    with torch.no_grad():
        for b in va:
            b = {k: v.to(device) for k, v in b.items()}
            pick = model(b).argmax(-1)
            t = b["true"]
            got.append(t.gather(1, pick[:, None]).squeeze(1).cpu().numpy())
            best.append(t.max(-1).values.cpu().numpy()); first.append(t[:, 0].cpu().numpy())
    got, best, first = map(np.concatenate, (got, best, first))
    print(f"  {name:28s} {100 * got.mean():6.2f}   (1순위 {100 * first.mean():.2f}, 상한 {100 * best.mean():.2f},"
          f" 회복률 {100 * (got.mean() - first.mean()) / max(best.mean() - first.mean(), 1e-9):5.1f}%)", flush=True)
    return float(got.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-npz", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pred-cache", default="")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--anchors", default="data/planning_vb/trajectory_anchors_256.npy")
    ap.add_argument("--out", default="work_dirs/rerank")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--gt-entities", action="store_true", help="feed GT entities instead of the model's own")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = Probe(args.labels_npz, args.cache, args.pred_cache, args.dump, args.anchors, args.gt_entities)
    n_val = max(int(0.2 * len(ds)), 1)
    g = torch.Generator().manual_seed(0)
    train, val = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=g)
    tr = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    va = DataLoader(val, batch_size=args.batch_size, num_workers=2)
    print(f"\n고른 후보의 실제 EPDMS (배포 상위 {K}개 중 선택, held-out {n_val} 장면, EC 제외)")
    t0, res = time.time(), {}
    for name, kw in (("점수만", dict(use_scores=True, use_geometry=False)),
                     ("점수 + 후보 기하", dict(use_scores=True, use_geometry=True)),
                     ("점수 + 기하 + 실행", dict(use_scores=True, use_geometry=True, use_execution=True)),
                     ("점수 + 기하 + 인지", dict(use_scores=True, use_geometry=True, use_entities=True)),
                     ("전부", dict(use_scores=True, use_geometry=True, use_entities=True, use_execution=True))):
        torch.manual_seed(0)
        res[name] = run(name, ReRanker(args.anchors, **kw).to(device), tr, va, device, args.epochs)
    print(f"  ({time.time() - t0:.0f}s)")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(Path(args.out) / "rerank.json", "w"))


if __name__ == "__main__":
    main()
