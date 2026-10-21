"""minADE / minFDE for the agent motion head — the one perception head we never evaluated.

The model predicts ``fut_mode`` (6) future trajectories per detected agent over ``fut_ts`` (8)
steps of 0.5 s, as PER-STEP OFFSETS (``det_motion_head._last_valid_fde`` accumulates them), and is
trained on them (``para_ssr_loss.py``: task ``motion``). The aux evaluation only ever measured
detection and map mAP, so the failure attribution had to fall back on a constant-velocity proxy.

Predicted agents are matched to GT agents by nearest centre within --match m (class-agnostic, the
GT cache's own ordering), and only GT future steps flagged valid are scored. Reported per class:

    minADE / minFDE   best of the 6 modes (the usual prediction metric)
    ADE1  / FDE1      the mode the model itself ranks first -- what a downstream consumer gets
    MR                miss rate, best-of-6 FDE > 2 m
    CV ADE            a constant-velocity baseline from the predicted box velocity, which is what
                      failure_attribution.py used as a stand-in for this head

    python tools/plan_v2/eval_motion.py --ckpt <ckpt> --out <json> [--bb 34] [--scenes 0] <overrides>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "readout"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: F401,E402
from _env import scene_filter, split_dirs  # noqa: E402
from cache_student_bev import build_agent  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402


class SceneData(Dataset):
    """features AND targets: the GT futures live on the Scene, not on the AgentInput."""

    def __init__(self, tokens, loader, builders, target_builders):
        self.tokens, self.loader = tokens, loader
        self.builders, self.target_builders = builders, target_builders

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        tok = self.tokens[i]
        try:
            feats = {}
            for b in self.builders:
                feats.update(b.compute_features(self.loader.get_agent_input_from_token(tok)))
            scene = self.loader.get_scene_from_token(tok)
            targets = {}
            for b in self.target_builders:
                targets.update(b.compute_targets(scene))
        except (FileNotFoundError, KeyError, ValueError):
            return tok, None, None
        return tok, feats, targets


def collate(batch):
    ok = [b for b in batch if b[1] is not None]
    if not ok:
        return [b[0] for b in batch], None, None
    t, f, g = ok[0]
    return [t], {k: v[None] for k, v in f.items()}, {k: (v[None] if torch.is_tensor(v) else v) for k, v in g.items()}

CLASSES = ("vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier", "czone_sign", "generic_object")


@torch.no_grad()
def scene_metrics(det_out, targets, match, det_thr):
    """One scene -> per-agent rows (class, minADE, minFDE, ADE1, FDE1, cv_ade)."""
    from navsim.agents.para_ssr.modules.losses import denormalize_bbox

    cls = det_out["all_cls_scores"][-1][0].sigmoid()
    score, label = cls.max(-1)
    boxes = denormalize_bbox(det_out["all_bbox_preds"][-1][0].float())
    traj = det_out["traj_preds"][0].float()                      # [Q, M, T, 2] offsets
    mode = det_out["traj_cls_preds"][0].float().argmax(-1)       # [Q] the model's own ranking
    keep = torch.nonzero(score >= det_thr).squeeze(-1)
    if not len(keep):
        return []
    pos = boxes[keep, :2]
    gt_boxes, gt_valid = targets["gt_boxes"][0].float(), targets["gt_valid"][0].bool()
    gt_fut, gt_mask = targets["gt_fut_trajs"][0].float(), targets["gt_fut_masks"][0].float()
    gt_label = targets["gt_labels"][0].long()
    rows = []
    for g in torch.nonzero(gt_valid).squeeze(-1).tolist():
        m = gt_mask[g] > 0.5
        if not m.any():
            continue
        d = torch.linalg.norm(pos - gt_boxes[g, :2][None], dim=-1)
        j = int(torch.argmin(d))
        if float(d[j]) > match:                                   # not detected: no prediction to score
            rows.append(dict(cls=int(gt_label[g]), matched=False))
            continue
        q = int(keep[j])
        p = traj[q].cumsum(-2)                                    # [M, T, 2] positions
        t = gt_fut[g].cumsum(-2)                                  # [T, 2]
        err = torch.linalg.norm(p - t[None], dim=-1)              # [M, T]
        last = int(torch.nonzero(m).max())
        ade = (err * m[None]).sum(-1) / m.sum()
        fde = err[:, last]
        k = int(mode[q])
        b = boxes[q]
        cv = (torch.arange(1, err.shape[1] + 1, device=err.device, dtype=torch.float32) * 0.5)[:, None] * b[7:9][None]
        cv_err = torch.linalg.norm(cv - t, dim=-1)
        rows.append(dict(cls=int(gt_label[g]), matched=True, gap=float(d[j]),
                         min_ade=float(ade.min()), min_fde=float(fde.min()),
                         ade1=float(ade[k]), fde1=float(fde[k]),
                         cv_ade=float((cv_err * m).sum() / m.sum()), cv_fde=float(cv_err[last])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--bb", default="34")
    ap.add_argument("--scenes", type=int, default=0)
    ap.add_argument("--match", type=float, default=2.0, help="centre distance for GT<->prediction matching")
    ap.add_argument("--det-thr", type=float, default=0.3)
    ap.add_argument("--dump-preds", default="", help="save per-scene boxes + motion predictions here")
    args, overrides = ap.parse_known_args()

    from navsim.common.dataloader import SceneLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ov = overrides + [f"agent.config.image_architecture=resnet{args.bb}.tv_in1k"]
    agent = build_agent(args.ckpt, ov).to(device)
    model = agent.para_ssr_model
    logs_dir, blobs_dir = split_dirs("trainval" if args.split == "navtrain" else "test")
    loader = SceneLoader(logs_dir, blobs_dir, scene_filter(args.split), sensor_config=agent.get_sensor_config())
    tokens = sorted(loader.tokens)
    if args.scenes:
        tokens = tokens[: args.scenes]
    if args.dump_preds:
        Path(args.dump_preds).mkdir(parents=True, exist_ok=True)
    builders = agent.get_feature_builders()
    dl = DataLoader(SceneData(tokens, loader, builders, agent.get_target_builders()), batch_size=1,
                    num_workers=4, collate_fn=collate)
    rows, t0 = [], time.time()
    for n, (toks, feats, targets) in enumerate(dl):
        if feats is None or targets is None:
            continue
        feats = {k: v.to(device) for k, v in feats.items()}
        targets = {k: v.to(device) for k, v in targets.items() if torch.is_tensor(v)}
        with torch.no_grad():
            out = model(feats, run_aux=True)
        if "traj_preds" not in out:
            raise SystemExit("이 체크포인트에는 motion head 출력이 없다 (use_det_motion_head=false?)")
        rows += scene_metrics(out, targets, args.match, args.det_thr)
        if args.dump_preds:
            from navsim.agents.para_ssr.modules.losses import denormalize_bbox
            sc = out["all_cls_scores"][-1][0].sigmoid().max(-1).values
            k = torch.nonzero(sc >= args.det_thr).squeeze(-1)
            np.savez_compressed(Path(args.dump_preds) / f"{toks[0]}.npz",
                                boxes=denormalize_bbox(out["all_bbox_preds"][-1][0].float())[k].cpu().numpy().astype(np.float32),
                                scores=sc[k].cpu().numpy().astype(np.float32),
                                traj=out["traj_preds"][0][k].float().cpu().numpy().astype(np.float16),
                                mode=out["traj_cls_preds"][0][k].float().argmax(-1).cpu().numpy().astype(np.int16))
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(tokens)}  {(time.time() - t0) / (n + 1):.3f} s/scene", flush=True)

    report(rows, args, len(tokens))


def report(rows, args, n_scenes):
    import collections
    res = {"checkpoint": args.ckpt, "split": args.split, "scenes": n_scenes,
           "match_m": args.match, "det_thr": args.det_thr, "horizon_s": 4.0, "modes": 6,
           "note": "offsets accumulated to positions; only GT-valid future steps; matched agents only"}
    by = collections.defaultdict(list)
    for r in rows:
        by["all"].append(r)
        by[CLASSES[r["cls"]] if r["cls"] < len(CLASSES) else str(r["cls"])].append(r)
    print(f"\n{args.ckpt}")
    print(f"{'클래스':14s} {'GT':>7s} {'검출률':>7s} {'minADE':>7s} {'minFDE':>7s} {'ADE1':>7s} {'FDE1':>7s} {'MR':>6s} {'등속ADE':>8s}")
    for name, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        m = [r for r in rs if r.get("matched")]
        if len(rs) < 50:
            continue
        v = dict(n_gt=len(rs), detected=len(m) / len(rs))
        if m:
            for k in ("min_ade", "min_fde", "ade1", "fde1", "cv_ade", "cv_fde"):
                v[k] = float(np.mean([r[k] for r in m]))
            v["mr"] = float(np.mean([r["min_fde"] > 2.0 for r in m]))
        res[name] = v
        if m:
            print(f"{name:14s} {v['n_gt']:7d} {100 * v['detected']:6.1f}% {v['min_ade']:7.3f} {v['min_fde']:7.3f}"
                  f" {v['ade1']:7.3f} {v['fde1']:7.3f} {100 * v['mr']:5.1f}% {v['cv_ade']:8.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
