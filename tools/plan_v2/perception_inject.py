"""Give the scorer the objects it MISSED, and see whether its scores change.

The ablation (``perception_use.py``) removes evidence the model already has, which answers "is the
score wired to this evidence" and nothing else. The question that decides whether better perception
would raise the score is the opposite one: if the object the model failed to see were handed to it,
would the score move?

v2's scorer reads perception through ``planner_head.prepare_task_memories``, which gives it three
things per slot: a pooled latent, an ENCODED POSITION and a confidence. The latent cannot be
synthesised for an object the detector never produced, but the position is a learned MLP over plain
coordinates, so a ground-truth position IS in distribution. This script therefore overwrites the
lowest-confidence slots with

    position   <- the GT object / road boundary the model missed
    confidence <- 1.0

and measures the predicted NC (agents) or DAC (map) of the candidates that conflict with that GT
object, against candidates far from it -- the same contrast the ablation uses, so the two numbers
can be read side by side. A placebo injects a GT object far from every candidate.

This is a PARTIAL oracle: the slot's latent still describes whatever the detector saw there. If the
score moves, positional evidence is usable and better perception would help; if it does not move
even with correct positions, the scorer is not reading positions at all.

    python tools/plan_v2/perception_inject.py --ckpt <ckpt> --out <npz> [--tokens <txt>] [--limit N]
"""
import argparse
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
from eval_motion import SceneData, collate  # noqa: E402

ROAD = 0


def ssr_to_nav(p):
    out = np.empty_like(p)
    out[..., 0], out[..., 1] = p[..., 1], -p[..., 0]
    return out


def gaps_to_paths(paths, xy):
    """[K, T, 2] candidate paths (NAVSIM frame) vs one point or polyline -> [K] closest distance."""
    q = np.atleast_2d(xy)
    d = np.linalg.norm(paths[:, :, None, :] - q[None, None], axis=-1)     # [K, T, P]
    return d.min(axis=(1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bb", default="34")
    ap.add_argument("--det-thr", type=float, default=0.3)
    ap.add_argument("--conflict", type=float, default=2.5)
    ap.add_argument("--far", type=float, default=8.0)
    ap.add_argument("--per-scene", type=int, default=3)
    ap.add_argument("--anchors", default="data/planning_vb/trajectory_anchors_256.npy")
    args, overrides = ap.parse_known_args()

    from navsim.common.dataloader import SceneLoader
    from navsim.agents.para_ssr.modules.losses import denormalize_bbox

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ov = overrides + [f"agent.config.image_architecture=resnet{args.bb}.tv_in1k"]
    agent = build_agent(args.ckpt, ov).to(device)
    model = agent.para_ssr_model
    head = model.pts_bbox_head
    anchors = np.load(args.anchors).astype(np.float32)
    x0, x1, y0, y1 = float(head.pc_range[0]), float(head.pc_range[3]), float(head.pc_range[1]), float(head.pc_range[4])

    captured, orig_plan, orig_prep = {}, head.plan_from_bev, head.prepare_task_memories
    inject = {}

    def capture(*a, **k):
        captured["args"], captured["kwargs"] = a, k
        return orig_plan(*a, **k)

    def prep(det_out, map_out):
        mem = orig_prep(det_out, map_out)
        for kind in ("det", "map"):
            spec = inject.get(kind)
            if spec is None:
                continue
            xy, enc = spec, getattr(head, f"{kind}_position_encoder")
            conf_enc = getattr(head, f"{kind}_confidence_encoder")
            pos, cf = mem[f"{kind}_position"], mem[f"{kind}_confidence"]
            # overwrite the least confident slots: the model had nothing there anyway
            order = torch.argsort(cf.norm(dim=-1)[0])[: len(xy)]
            p = torch.as_tensor(np.asarray(xy, np.float32), device=pos.device)[None]
            if kind == "map":
                p = enc(p).mean(dim=2) if p.dim() == 4 else enc(p)
            else:
                p = enc(p)
            pos[:, order] = p.to(pos.dtype)
            cf[:, order] = conf_enc(torch.ones(1, len(xy), 1, device=cf.device)).to(cf.dtype)
        return mem

    head.plan_from_bev, head.prepare_task_memories = capture, prep

    def scores(**spec):
        inject.clear(); inject.update(spec)
        with torch.no_grad():
            out = orig_plan(*captured["args"], **captured["kwargs"])
        inject.clear()
        return out["sim_rewards"][0].float().cpu().numpy()

    logs_dir, blobs_dir = split_dirs("test")
    loader = SceneLoader(logs_dir, blobs_dir, scene_filter("navtest"), sensor_config=agent.get_sensor_config())
    tokens = sorted(loader.tokens)
    if args.tokens:
        want = {l.strip() for l in open(args.tokens) if l.strip()}
        tokens = [t for t in tokens if t in want]
    if args.limit:
        tokens = tokens[: args.limit]
    dl = DataLoader(SceneData(tokens, loader, agent.get_feature_builders(), agent.get_target_builders()),
                    batch_size=1, num_workers=3, collate_fn=collate)
    rows = {"agent": [], "map": [], "placebo": []}
    t0 = 0
    for n, (toks, feats, targets) in enumerate(dl):
        if feats is None:
            continue
        feats = {k: v.to(device) for k, v in feats.items()}
        with torch.no_grad():
            base_out = model(feats, run_aux=False)
        base = base_out["sim_rewards"][0].float().cpu().numpy()
        paths = ssr_to_nav((anchors[None] * 0 + anchors)[..., :2]) if False else anchors[..., :2]
        det_out, map_out = captured["args"][2], captured["args"][3]
        pred = denormalize_bbox(det_out["all_bbox_preds"][-1][0].float()).cpu().numpy()
        pscore = det_out["all_cls_scores"][-1][0].sigmoid().amax(-1).float().cpu().numpy()
        seen = pred[pscore >= args.det_thr, :2]

        # --- GT agents the model missed (no detection within 2 m)
        gt_boxes = targets["gt_boxes"][0].cpu().numpy()
        gt_valid = targets["gt_valid"][0].cpu().numpy() > 0.5
        missed = []
        for g in np.flatnonzero(gt_valid):
            if not len(seen) or np.linalg.norm(seen - gt_boxes[g, :2], axis=1).min() > 2.0:
                missed.append(gt_boxes[g, :2])
        for xy in missed[: args.per_scene]:
            gaps = gaps_to_paths(paths, ssr_to_nav(xy[None])[0])
            c, f = gaps < args.conflict, gaps > args.far
            if c.sum() < 3 or f.sum() < 3:
                continue
            nx = (xy - np.array([x0, y0])) / np.array([x1 - x0, y1 - y0])
            d = scores(det=nx[None]) - base
            rows["agent"].append((toks[0], float(d[0][c].mean() - d[0][f].mean()),
                                  float(d[3][c].mean() - d[3][f].mean()), float(base[0][c].mean()), int(c.sum())))
        far_missed = [xy for xy in missed if gaps_to_paths(paths, ssr_to_nav(xy[None])[0]).min() > 2 * args.far]
        if far_missed:
            nx = (far_missed[0] - np.array([x0, y0])) / np.array([x1 - x0, y1 - y0])
            d = scores(det=nx[None]) - base
            rows["placebo"].append((toks[0], float(np.abs(d[0]).mean()), float(np.abs(d[3]).mean())))

        # --- GT road boundaries the model missed (no predicted road vector within 2 m of it)
        gt_pts = targets["gt_map_pts"][0].cpu().numpy()
        gt_lab = targets["gt_map_labels"][0].cpu().numpy()
        gt_mv = targets["gt_map_valid"][0].cpu().numpy() > 0.5
        if gt_pts.ndim == 4:
            gt_pts = gt_pts[:, 0]
        mp = map_out["all_map_pts_preds"][-1][0].float().cpu().numpy()
        mcls = map_out["all_map_cls_scores"][-1][0].sigmoid().float().cpu().numpy()
        seen_map = mp[(mcls.argmax(-1) == ROAD) & (mcls[:, ROAD] >= args.det_thr)]
        for v in np.flatnonzero(gt_mv & (gt_lab == ROAD))[: args.per_scene]:
            pts = gt_pts[v]                                            # normalised [P, 2]
            if len(seen_map) and np.linalg.norm(seen_map - pts[None], axis=-1).min() < 0.05:
                continue                                               # the model already has it
            phys = pts * np.array([x1 - x0, y1 - y0]) + np.array([x0, y0])
            gaps = gaps_to_paths(paths, ssr_to_nav(phys))
            c, f = gaps < 0.25, gaps > 5.0
            if c.sum() < 3 or f.sum() < 3:
                continue
            d = scores(map=pts[None]) - base
            rows["map"].append((toks[0], float(d[1][c].mean() - d[1][f].mean()),
                                float(base[1][c].mean()), int(c.sum())))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(tokens)}", flush=True)

    head.plan_from_bev, head.prepare_task_memories = orig_plan, orig_prep
    np.savez(args.out, **{k: np.array(v, dtype=object) for k, v in rows.items()}, checkpoint=np.asarray(args.ckpt))
    report(rows)


def report(rows):
    a = np.array([r[1:] for r in rows["agent"]], dtype=float)
    print(f"\n놓친 GT 차량의 위치를 넣어줬을 때 (n={len(a)})")
    if len(a):
        print(f"  충돌 후보 - 먼 후보  NC {a[:, 0].mean():+.4f}  TTC {a[:, 1].mean():+.4f}"
              f"   (원래 예측 NC {a[:, 2].mean():.3f})")
    p = np.array([r[1:] for r in rows["placebo"]], dtype=float)
    if len(p):
        print(f"  위약(모든 후보와 먼 GT 차량 주입) 평균 |d| NC {p[:, 0].mean():.4f} TTC {p[:, 1].mean():.4f}")
    m = np.array([r[1:] for r in rows["map"]], dtype=float)
    print(f"\n놓친 GT road 경계를 넣어줬을 때 (n={len(m)})")
    if len(m):
        print(f"  가로지르는 후보 - 먼 후보  DAC {m[:, 0].mean():+.4f}   (원래 예측 DAC {m[:, 1].mean():.3f})")


if __name__ == "__main__":
    main()
