"""Does the v2 score head USE the perception it is given?  Intervention test.

The planner reads detections and map elements as explicit memory tokens (one per det query /
map vector, planner_head.prepare_task_memories).  For every scene the model is run once; the
planner part is then re-run with tokens removed, and the change of the predicted sub-scores
(NC, DAC, TTC) of all 256 candidates (anchor + offset) is recorded:

  global   all det tokens / all map tokens replaced by one empty token
  agent    one detected agent removed, three ways: its det token only (T), its BEV cells only
           (B, replaced by the scene's mean BEV feature), or both (TB).  Candidates whose path meets the agent's constant-velocity
           extrapolation (centre gap < --conflict m at the same time) vs candidates far from it
           (> --far m).  If the head uses the agent, removing it should raise NC/TTC of the
           conflicting candidates more than of the far ones:  contrast = mean d(conflict) - mean d(far)
  map      one 'road' boundary vector removed.  Candidates whose path crosses it vs far ones, on DAC
  placebo  the same for a detected agent / road vector far from every candidate (noise floor)

    python tools/plan_v2/perception_use.py --ckpt <ckpt> --out <npz> [--tokens <csv with token col>] \
        [--limit N] <agent.config overrides...>
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "readout"))
import _env  # noqa: F401,E402
from _env import scene_filter, split_dirs  # noqa: E402
from cache_student_bev import FeatureData, build_agent, collate  # noqa: E402

REAR_TO_CENTER = 1.461
ROAD = 0          # map label of the closed road contours (failure_attribution.road_region)


def ssr_to_nav(xy):
    return np.stack([xy[..., 1], -xy[..., 0]], axis=-1)


def candidate_paths(anchors, offset):
    """[K, 8, 3] rear-axle poses (NAVSIM frame) -> box centres [K, 9, 2] incl. t = 0."""
    p = anchors + offset
    c = p[..., :2] + REAR_TO_CENTER * np.stack([np.cos(p[..., 2]), np.sin(p[..., 2])], -1)
    return np.concatenate([np.tile([[[REAR_TO_CENTER, 0.0]]], (len(p), 1, 1)), c], axis=1)


def agent_gaps(paths, box):
    """min centre gap [K] between candidates and a CV-extrapolated agent (physical box, SSR frame)."""
    pos = ssr_to_nav(box[:2]); vel = ssr_to_nav(box[7:9])
    t = np.arange(paths.shape[1]) * 0.5
    ag = pos[None] + t[:, None] * vel[None]                       # [9, 2]
    return np.linalg.norm(paths - ag[None], axis=-1).min(axis=1)


def map_gaps(paths, pts):
    """min distance [K] between candidate polylines and one map polyline (NAVSIM frame); 0 = crossing."""
    from shapely.geometry import LineString
    line = LineString(pts)
    return np.array([LineString(p).distance(line) for p in paths])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default=None, help="csv/txt with the scene tokens to use")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--det-thr", type=float, default=0.3)
    ap.add_argument("--map-thr", type=float, default=0.3)
    ap.add_argument("--conflict", type=float, default=2.5)
    ap.add_argument("--far", type=float, default=8.0)
    ap.add_argument("--per-scene", type=int, default=3, help="agents / road vectors intervened per scene")
    args, overrides = ap.parse_known_args()

    from navsim.common.dataloader import SceneLoader
    from navsim.agents.para_ssr.modules.losses import denormalize_bbox

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = build_agent(args.ckpt, overrides).to(device).eval()
    model = agent.para_ssr_model
    head = model.pts_bbox_head
    if head.anchor_planner is None or not head.use_task_interaction:
        raise SystemExit("needs the v2 anchor planner with task-interaction memories")
    anchors = head.anchor_planner.trajectory_anchors.float().cpu().numpy()
    pc = model.cfg.map_pc_range if hasattr(model, "cfg") else (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)
    x0, y0, x1, y1 = float(pc[0]), float(pc[1]), float(pc[3]), float(pc[4])

    captured = {}
    orig_plan, orig_prep = head.plan_from_bev, head.prepare_task_memories

    def capture(*a, **k):
        captured["args"], captured["kwargs"] = a, k
        return orig_plan(*a, **k)

    edit = {}

    def prep(det_out, map_out):
        mem = orig_prep(det_out, map_out)
        for kind, drop in (("det", edit.get("det")), ("map", edit.get("map"))):
            if drop is None:
                continue
            keys = [f"{kind}_memory", f"{kind}_position", f"{kind}_confidence"]
            if drop == "all":
                for k in keys:
                    mem[k] = torch.zeros_like(mem[k][:, :1])
            else:
                keep = torch.ones(mem[keys[0]].shape[1], dtype=torch.bool, device=mem[keys[0]].device)
                keep[drop] = False
                for k in keys:
                    mem[k] = mem[k][:, keep]
        return mem

    head.plan_from_bev, head.prepare_task_memories = capture, prep

    H, W = int(head.bev_h), int(head.bev_w)

    def bev_masked(box, margin=1.0):
        """BEV [B, H*W, C] with the cells around one agent (physical box, SSR frame) set to the mean."""
        bev = captured["args"][0]
        r = max(float(box[3]), float(box[4])) / 2 + margin
        ys = (np.arange(H) + 0.5) / H * (y1 - y0) + y0
        xs = (np.arange(W) + 0.5) / W * (x1 - x0) + x0
        rows = np.flatnonzero(np.abs(ys - box[1]) <= r); cols = np.flatnonzero(np.abs(xs - box[0]) <= r)
        idx = torch.as_tensor((rows[:, None] * W + cols[None]).reshape(-1), device=bev.device)
        out = bev.clone()
        if len(idx):
            out[:, idx] = bev.mean(dim=1, keepdim=True)
        return out, len(idx)

    def scores(bev=None, **e):
        edit.clear(); edit.update(e)
        a = list(captured["args"])
        if bev is not None:
            a[0] = bev
        with torch.no_grad():
            out = orig_plan(*a, **captured["kwargs"])
        edit.clear()
        return out["sim_rewards"][0].float().cpu().numpy(), out

    logs_dir, blobs_dir = split_dirs("test")
    loader = SceneLoader(logs_dir, blobs_dir, scene_filter("navtest"), sensor_config=agent.get_sensor_config())
    tokens = sorted(loader.tokens)
    if args.tokens:
        import pandas as pd
        want = set(pd.read_csv(args.tokens).token) if args.tokens.endswith(".csv") else set(open(args.tokens).read().split())
        tokens = [t for t in tokens if t in want]
    if args.limit:
        tokens = list(np.random.default_rng(0).permutation(tokens)[: args.limit])
    dl = DataLoader(FeatureData(tokens, loader, agent.get_feature_builders()), batch_size=1, num_workers=3,
                    collate_fn=collate)
    rows = {k: [] for k in ("global", "agent", "map", "placebo_agent", "placebo_map")}
    t0, n = time.time(), 0
    for toks, feats, _ in dl:
        if feats is None:
            continue
        feats = {k: v.to(device) for k, v in feats.items()}
        with torch.no_grad():
            base_out = model(feats, run_aux=False)
        base = base_out["sim_rewards"][0].float().cpu().numpy()                 # [5, K]
        offset = base_out["trajectory_offset"][0].float().cpu().numpy()
        paths = candidate_paths(anchors, offset)
        det_out, map_out = captured["args"][2], captured["args"][3]
        # global ablations
        for what, e in (("det", {"det": "all"}), ("map", {"map": "all"})):
            s, _ = scores(**e)
            d = s - base
            rows["global"].append((toks[0], what, *[np.abs(d[i]).mean() for i in (0, 1, 3)],
                                   *[d[i].mean() for i in (0, 1, 3)]))
        # detected agents
        cls = det_out["all_cls_scores"][-1][0].sigmoid().amax(-1).float().cpu().numpy()
        boxes = denormalize_bbox(det_out["all_bbox_preds"][-1][0].float()).cpu().numpy()
        cand = [(agent_gaps(paths, boxes[q]), q) for q in np.flatnonzero(cls >= args.det_thr)]
        near = sorted([c for c in cand if (c[0] < args.conflict).sum() >= 3 and (c[0] > args.far).sum() >= 3],
                      key=lambda c: c[0].min())[: args.per_scene]
        for gaps, q in near:
            c, f = gaps < args.conflict, gaps > args.far
            bev_m, ncell = bev_masked(boxes[q])
            con = []
            for e in ({"det": [int(q)]}, {"bev": bev_m}, {"det": [int(q)], "bev": bev_m}):
                s, _ = scores(**e)
                d = s - base
                con += [d[0][c].mean() - d[0][f].mean(), d[3][c].mean() - d[3][f].mean()]
            rows["agent"].append((toks[0], int(q), float(cls[q]), int(c.sum()), *con,
                                  base[0][c].mean(), base[3][c].mean(), ncell))
        far_agents = [q for g, q in cand if g.min() > 2 * args.far]
        if far_agents:
            s, _ = scores(det=[int(far_agents[0])])
            d = s - base
            rows["placebo_agent"].append((toks[0], np.abs(d[0]).mean(), np.abs(d[3]).mean()))
        # road boundary vectors
        mcls = map_out["all_map_cls_scores"][-1][0].sigmoid().float().cpu().numpy()          # [V, classes]
        mpts = map_out["all_map_pts_preds"][-1][0].float().cpu().numpy().copy()              # [V, P, 2] in [0,1]
        mpts[..., 0] = mpts[..., 0] * (x1 - x0) + x0
        mpts[..., 1] = mpts[..., 1] * (y1 - y0) + y0
        mpts = ssr_to_nav(mpts)
        roads = [v for v in range(len(mcls)) if mcls[v].argmax() == ROAD and mcls[v, ROAD] >= args.map_thr]
        mc = [(map_gaps(paths, mpts[v]), v) for v in roads]
        cross = sorted([c for c in mc if (c[0] < 0.25).sum() >= 3 and (c[0] > 5.0).sum() >= 3],
                       key=lambda c: -(c[0] < 0.25).sum())[: args.per_scene]
        for gaps, v in cross:
            s, _ = scores(map=[int(v)])
            d = s - base
            c, f = gaps < 0.25, gaps > 5.0
            rows["map"].append((toks[0], int(v), float(mcls[v, ROAD]), int(c.sum()),
                                d[1][c].mean() - d[1][f].mean(), base[1][c].mean(), base[1][f].mean()))
        far_roads = [v for g, v in mc if g.min() > 10.0]
        if far_roads:
            s, _ = scores(map=[int(far_roads[0])])
            rows["placebo_map"].append((toks[0], np.abs(s[1] - base[1]).mean()))
        n += 1
        if n % 200 == 0:
            print(f"{n}/{len(tokens)}  {(time.time() - t0) / n:.2f} s/scene", flush=True)

    head.plan_from_bev, head.prepare_task_memories = orig_plan, orig_prep
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{k: np.array(v, dtype=object) for k, v in rows.items()},
             checkpoint=np.asarray(args.ckpt))
    report(rows)


def report(rows):
    g = np.array([r[2:] for r in rows["global"]], dtype=float) if rows["global"] else np.zeros((0, 6))
    kinds = np.array([r[1] for r in rows["global"]])
    print(f"\n장면 {len(rows['global']) // 2}")
    print("전역 제거: 256 후보 예측 점수의 평균 |변화| (NC, DAC, TTC) / 평균 변화")
    for w in ("det", "map"):
        m = g[kinds == w]
        if len(m):
            print(f"  {w} 전부 제거  |d| NC {m[:, 0].mean():.4f} DAC {m[:, 1].mean():.4f} TTC {m[:, 2].mean():.4f}"
                  f"   d NC {m[:, 3].mean():+.4f} DAC {m[:, 4].mean():+.4f} TTC {m[:, 5].mean():+.4f}")
    a = np.array([r[4:] for r in rows["agent"]], dtype=float)
    if len(a):
        print(f"차량 하나 제거 (n={len(a)}): 충돌 후보 - 먼 후보 예측 변화 대비 (클수록 그 차량 정보를 씀)")
        for i, name in enumerate(("det token 만", "BEV 칸만", "둘 다")):
            nc, ttc = a[:, 2 * i], a[:, 2 * i + 1]
            print(f"  {name:10s} NC {nc.mean():+.4f} (>0.05 {100 * (nc > 0.05).mean():3.0f}%)"
                  f"  TTC {ttc.mean():+.4f} (>0.05 {100 * (ttc > 0.05).mean():3.0f}%)")
        print(f"  (충돌 후보의 원래 예측 NC {a[:, 6].mean():.3f}, TTC {a[:, 7].mean():.3f}; 가린 BEV 칸 평균 {a[:, 8].mean():.0f}개)")
    p = np.array([r[1:] for r in rows["placebo_agent"]], dtype=float)
    if len(p):
        print(f"  위약(먼 차량 제거) 평균 |d| NC {p[:, 0].mean():.4f} TTC {p[:, 1].mean():.4f}")
    m = np.array([r[4:] for r in rows["map"]], dtype=float)
    if len(m):
        print(f"road 경계 하나 제거 (n={len(m)}): 가로지르는 후보 - 먼 후보 DAC 변화 대비 {m[:, 0].mean():+.4f}"
              f" (>0.02 {100 * (m[:, 0] > 0.02).mean():.0f}%)  | 원래 예측 DAC 가로지름 {m[:, 1].mean():.3f} / 먼 {m[:, 2].mean():.3f}")
    q = np.array([r[1] for r in rows["placebo_map"]], dtype=float)
    if len(q):
        print(f"  위약(먼 road 경계 제거) 평균 |d| DAC {q.mean():.4f}")


if __name__ == "__main__":
    main()
