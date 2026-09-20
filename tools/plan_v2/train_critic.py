"""Train the entity-only critic (modules/entity_critic.py) on the cached GT tokens and report

  1. how well it ranks anchors (AUC per metric on a held-out split) -- can a scorer that sees ONLY
     entities do better than the sensor model's score head (v2: DAC 0.90, NC 0.79 on navtest)?
  2. how much its scores move when one agent is removed -- the reference value the intervention
     diagnostic (tools/plan_v2/perception_use.py) lacked.

    python tools/plan_v2/train_critic.py --cache data/critic_cache --labels data/planning_vb/pdm_score_256 \
        --out work_dirs/entity_critic [--epochs 12]
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from navsim.agents.para_ssr.modules.anchor_planner import SIM_KEYS
from navsim.agents.para_ssr.modules.entity_critic import EntityCritic, critic_loss

KEYS = ("agents", "agent_fut", "agent_fut_mask", "agent_valid", "map_pts", "map_labels", "map_valid",
        "ego_status", "command", "trajectory")


ENT = ("agents", "agent_fut", "agent_fut_mask", "agent_valid", "map_pts", "map_labels", "map_valid")


class Cache(Dataset):
    def __init__(self, cache: str, labels: str, pred_cache: str = "", entities: str = "gt", train: bool = True):
        shards = sorted(glob.glob(f"{cache}/shard_*.npz"))
        if not shards:
            raise SystemExit(f"no shards in {cache}")
        data, toks = {k: [] for k in KEYS}, []
        for s in shards:
            z = np.load(s)
            if len(z["tokens"]) == 0:
                continue
            toks += list(z["tokens"])
            for k in KEYS:
                data[k].append(z[k])
        self.tokens = np.asarray(toks)
        self.data = {k: np.concatenate(v) for k, v in data.items()}
        scores = np.load(labels + ".npy", mmap_mode="r")
        index = {t: i for i, t in enumerate(json.load(open(labels + ".tokens.json")))}
        rows = np.array([index.get(t, -1) for t in self.tokens])
        keep = rows >= 0
        self.tokens, rows = self.tokens[keep], rows[keep]
        self.data = {k: v[keep] for k, v in self.data.items()}
        self.labels = np.asarray(scores[rows], dtype=np.float32)          # [N, 5, K]
        # the model's own detections / map vectors for the same scenes (cache_pred_entities.py)
        self.entities, self.train_mode, self.pred = entities, train, None
        if pred_cache:
            pdata, ptok = {k: [] for k in ENT}, []
            for f in sorted(glob.glob(f"{pred_cache}/shard_*.npz")):
                z = np.load(f)
                ptok += list(z["tokens"])
                for k in ENT:
                    pdata[k].append(z[k])
            pos = {str(t): i for i, t in enumerate(ptok)}
            idx = np.array([pos.get(str(t), -1) for t in self.tokens])
            keep = idx >= 0
            self.tokens, self.labels = self.tokens[keep], self.labels[keep]
            self.data = {k: v[keep] for k, v in self.data.items()}
            idx = idx[keep]
            self.pred = {k: np.concatenate(v)[idx] for k, v in pdata.items()}
            print(f"  with predicted entities: {len(self.tokens)}", flush=True)
        print(f"{len(self.tokens)} scenes with labels", flush=True)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        src = self.data
        if self.pred is not None:
            use_pred = self.entities == "pred" or (self.entities == "mixed" and self.train_mode and np.random.rand() < 0.5)
            if use_pred:
                src = {**self.data, **{k: self.pred[k] for k in ENT}}
        b = {k: torch.from_numpy(np.asarray(v[i])) for k, v in src.items()}
        b["map_labels"] = b["map_labels"].long()
        b["label"] = torch.from_numpy(self.labels[i])
        return b


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(model, dl, device, anchors):
    model.eval()
    preds, labels, sens = [], [], []
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        p = model(b)
        preds.append(p.cpu().numpy()); labels.append(b["label"].cpu().numpy())
        # intervention: drop the agent nearest to the anchors it conflicts with
        boxes = b["agents"]
        pos = torch.stack([boxes[..., 1], -boxes[..., 0]], -1)                         # SSR -> NAVSIM
        d = torch.linalg.norm(anchors[None, :, :, None, :2] - pos[:, None, None], dim=-1)   # [B, K, T, A]
        gap = d.amin(dim=2)                                                             # [B, K, A]
        gap = gap.masked_fill(b["agent_valid"][:, None] < 0.5, 1e6)
        best = gap.amin(dim=1).argmin(dim=1)                                            # [B] closest agent
        g = gap.gather(2, best[:, None, None].expand(-1, gap.shape[1], 1)).squeeze(-1)  # [B, K]
        p2 = model(b, drop_agent=best)
        for i in range(len(g)):
            c, f = g[i] < 2.5, g[i] > 8.0
            if c.sum() >= 3 and f.sum() >= 3:
                dd = (p2[i] - p[i])
                sens.append([float(dd[0][c].mean() - dd[0][f].mean()), float(dd[3][c].mean() - dd[3][f].mean()),
                             float(p[i][0][c].mean()), float(p[i][3][c].mean())])
    pred, lab = np.concatenate(preds), np.concatenate(labels)
    out = {}
    for j, name in enumerate(SIM_KEYS):
        p, l = pred[:, j].ravel(), lab[:, j].ravel()
        out[name] = auc(p[l > 0.5], p[l <= 0.5])
    s = np.array(sens) if sens else np.zeros((0, 4))
    return out, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors", default="data/planning_vb/trajectory_anchors_256.npy")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--pred-cache", default="", help="cache_pred_entities.py output (the model's own perception)")
    ap.add_argument("--entities", default="gt", choices=("gt", "pred", "mixed"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = Cache(args.cache, args.labels, args.pred_cache, args.entities)
    n_val = max(int(0.05 * len(ds)), 1)
    g = torch.Generator().manual_seed(0)
    train, val = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=g)
    if args.pred_cache:   # validate on the model's own perception, which is what matters
        val_ds = Cache(args.cache, args.labels, args.pred_cache, "pred", train=False)
        val = torch.utils.data.Subset(val_ds, val.indices)
    dl = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(val, batch_size=args.batch_size, num_workers=2)
    model = EntityCritic(args.anchors, embed_dims=args.embed, num_layers=args.layers).to(device)
    anchors = model.trajectory_anchors
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train(); t0, tot = time.time(), 0.0
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            loss = critic_loss(model(b), b["label"], torch.ones(len(b["label"]), device=device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += float(loss)
        a, s = evaluate(model, vl, device, anchors)
        print(f"epoch {ep + 1}/{args.epochs} loss {tot / len(dl):.4f} ({time.time() - t0:.0f}s) | AUC " +
              " ".join(f"{k[:3].upper()} {v:.3f}" for k, v in a.items()) +
              (f" | 차량 제거 반응 NC {s[:, 0].mean():+.3f} TTC {s[:, 1].mean():+.3f} (n={len(s)})" if len(s) else ""),
              flush=True)
        torch.save({"model": model.state_dict(), "auc": a}, out / "critic.ckpt")
    a, s = evaluate(model, vl, device, anchors)
    print("\n최종 (held-out)")
    print("  anchor 순위 AUC: " + " ".join(f"{k} {v:.3f}" for k, v in a.items()))
    if len(s):
        print(f"  차량 하나 제거: 충돌 후보 - 먼 후보  NC {s[:, 0].mean():+.4f}  TTC {s[:, 1].mean():+.4f}"
              f"  (충돌 후보의 예측 NC {s[:, 2].mean():.3f}, TTC {s[:, 3].mean():.3f}, n={len(s)})")
    json.dump({"auc": a, "sensitivity": s.tolist()}, open(out / "report.json", "w"))


if __name__ == "__main__":
    main()
