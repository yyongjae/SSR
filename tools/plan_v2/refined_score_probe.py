"""Is "which of my own 256 refined candidates is best" learnable at all?

The re-ranking probe only reordered the model's top 3 and recovered nothing. But the 10-point gap
(88.81 deployed vs 99.18 if the best REFINED candidate were chosen) is not a re-ranking problem: it
needs a different candidate to reach the top at all. This probe therefore trains a small scorer over
ALL 256 candidates with the official refined labels and asks what its own pick is worth.

Features per candidate are only things the deployed model already produces:
  v2's own five predicted sub-scores, its imitation reward, the refined trajectory itself, and
  (optionally) the perception entity tokens, attended like the critic.

Scenes are split 80/20, so the number is "could a scorer trained on such labels pick better",
not a deployable model -- for deployment the labels would have to come from navtrain.

SCOPE. This probe sits on FROZEN v2 features (its own predicted sub-scores and the candidate
geometry). It has no access to the BEV, and it is trained on 9.7k scenes. A negative result
therefore only rules out post-hoc re-selection on top of those features. It says nothing about
retraining the scoring head end to end with BEV access, nor about navtrain-scale supervision.

    python tools/plan_v2/refined_score_probe.py --labels-npz data/planning_vb/navtest_refined_scores.npz \
        --dump work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz --cache data/critic_cache_navtest \
        --pred-cache data/critic_cache_pred_navtest [--entities] [--epochs 12]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from clean_compare import ENT, EXTRA, INT, epdms, load_cache

from navsim.agents.para_ssr.modules.entity_critic import EntityCritic, mlp


class Cand(Dataset):
    def __init__(self, labels_npz, dump, cache, pred_cache, anchors, use_entities):
        z = np.load(labels_npz, allow_pickle=True)
        cols = [str(c) for c in z["columns"]]
        lab_at = {str(t): i for i, t in enumerate(z["tokens"])}
        d = np.load(dump)
        d_at = {str(t): i for i, t in enumerate(d["tokens"])}
        gt_at, self.gt = load_cache(cache, ENT + INT + EXTRA)
        pred_at, self.pred = load_cache(pred_cache, ENT + INT) if pred_cache else ({}, {})
        toks = sorted(set(lab_at) & set(d_at) & set(gt_at) & (set(pred_at) if pred_at else set(gt_at)))
        self.tokens = toks
        li = [lab_at[t] for t in toks]
        di = [d_at[t] for t in toks]
        self.true = epdms(z["scores"][li], cols)[:, :-2].astype(np.float32)      # [N, 256] refined truth
        self.sim = d["sim_rewards"][di].astype(np.float32)                       # [N, 5, 256]
        self.im = d["im_rewards"][di].astype(np.float32)                         # [N, 256]
        self.fin = d["plan_final_rewards"][di].astype(np.float32)                # [N, 256] deployed rule
        self.traj = (np.load(anchors).astype(np.float32)[None] + d["trajectory_offset"][di].astype(np.float32))
        self.gi = np.array([gt_at[t] for t in toks])
        self.pi = np.array([pred_at[t] for t in toks]) if pred_at else self.gi
        self.use_entities = use_entities
        print(f"{len(toks)} 장면 x 256 후보", flush=True)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        out = {"scores": torch.from_numpy(self.sim[i].T.copy()),                 # [256, 5]
               "im": torch.from_numpy(self.im[i][:, None].copy()),
               "fin": torch.from_numpy(self.fin[i].copy()),
               "traj": torch.from_numpy(self.traj[i]),
               "true": torch.from_numpy(self.true[i])}
        if self.use_entities:
            src, si = self.pred, self.pi[i]
            for k in ENT + INT:
                out[k] = torch.from_numpy(np.asarray(src[k][si]).astype(np.int64 if k in INT else np.float32))
            for k in EXTRA:
                out[k] = torch.from_numpy(self.gt[k][self.gi[i]].astype(np.float32))
        return out


class Scorer(nn.Module):
    def __init__(self, anchors, use_entities, embed=128):
        super().__init__()
        self.use_entities = use_entities
        # v2 의 배포 점수도 입력에 넣는다: 최소한 현재 규칙을 재현할 수 있어야 probe 가 성립한다
        self.inp = mlp(5 + 1 + 1 + 24, embed, embed)
        self.entity = EntityCritic(anchors, embed_dims=embed, num_layers=2) if use_entities else None
        self.head = mlp(embed + (embed if use_entities else 0), embed, 1)
        # 마지막 층을 0 으로 시작: 잔차 0 = v2 배포 규칙 그대로. 기준선이 가설 공간 안에 있어야
        # "정보가 있는가" 를 묻는 실험이 된다 (이전 판은 기준선조차 재현하지 못해 무의미했다)
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, b):
        fin = b["fin"][..., None]
        fin = (fin - fin.mean(1, keepdim=True)) / (fin.std(1, keepdim=True) + 1e-6)
        h = self.inp(torch.cat([b["scores"], b["im"], fin, b["traj"].flatten(2)], dim=-1))
        if self.entity is not None:
            q = self.entity.cluster_encoder(self.entity.traj_mlp(b["traj"].flatten(2)))
            q = q + self.entity.ego_mlp(torch.cat([b["ego_status"], b["command"]], -1)).unsqueeze(1)
            mem, pad = self.entity.tokens(b)
            for layer in self.entity.layers:
                q = layer(q, mem, memory_key_padding_mask=pad)
            h = torch.cat([h, self.entity.norm(q)], dim=-1)
        return fin.squeeze(-1) + self.head(h).squeeze(-1)                        # 배포 점수 + 잔차


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-npz", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pred-cache", default="")
    ap.add_argument("--anchors", default="data/planning_vb/trajectory_anchors_256.npy")
    ap.add_argument("--entities", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = Cand(args.labels_npz, args.dump, args.cache, args.pred_cache, args.anchors, args.entities)
    n_val = max(int(0.2 * len(ds)), 1)
    g = torch.Generator().manual_seed(0)
    train, val = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=g)
    tr = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=3, drop_last=True)
    va = DataLoader(val, batch_size=args.batch_size, num_workers=2)
    model = Scorer(args.anchors, args.entities).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(tr))

    def evaluate():
        model.eval()
        got, dep, best, rnd = [], [], [], []
        with torch.no_grad():
            for b in va:
                b = {k: v.to(device) for k, v in b.items()}
                t = b["true"]
                pick = model(b).argmax(-1)
                got.append(t.gather(1, pick[:, None]).squeeze(1).cpu().numpy())
                dep.append(t.gather(1, b["fin"].argmax(-1)[:, None]).squeeze(1).cpu().numpy())
                best.append(t.max(-1).values.cpu().numpy())
                rnd.append(t[:, torch.randint(0, t.shape[1], (1,), device=device)].squeeze(1).cpu().numpy())
        return [100 * np.concatenate(x).mean() for x in (got, dep, best, rnd)]

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for b in tr:
            b = {k: v.to(device) for k, v in b.items()}
            # regression on the true score of every candidate + a listwise term: with 256 candidates
            # the argmax target alone is noisy (many candidates tie near the top)
            pred = model(b)
            t = b["true"]
            loss = nn.functional.cross_entropy(pred, torch.softmax(t / 0.05, dim=-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        if (ep + 1) % 3 == 0 or ep == args.epochs - 1:
            a, d, bst, r = evaluate()
            print(f"epoch {ep + 1:2d} loss {tot / len(tr):.4f} | 학습된 채점기 {a:6.2f}  "
                  f"v2 배포 {d:6.2f}  상한 {bst:6.2f}  무작위 {r:6.2f}", flush=True)
    a, d, bst, r = evaluate()
    print(f"\n({time.time() - t0:.0f}s)  held-out 장면 {n_val}, 인지 {'사용' if args.entities else '미사용'}")
    print(f"  학습된 채점기 {a:.2f} vs v2 배포 {d:.2f}  ->  {a - d:+.2f}")
    print(f"  회복률 {100 * (a - d) / max(bst - d, 1e-9):.1f}% (상한 {bst:.2f}, 무작위 {r:.2f})")


if __name__ == "__main__":
    main()
