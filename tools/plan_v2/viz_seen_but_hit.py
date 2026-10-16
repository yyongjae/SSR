"""Scenes the model PERCEIVED and drove into anyway.

report/25 measured two things that only make sense together: DAC/LK failures are barely fixable by
perception (<2% of the gap), and deleting the colliding vehicle from both the det tokens and the BEV
moves the predicted NC by +0.04. That says the evidence is there and the planner does not act on it.
This draws the individual scenes so the claim can be looked at rather than argued about.

Selected: tokens the official evaluation scored NC=0 (collision) or DAC=0 (off-road), restricted to
those where our OWN detector/mapper produced the relevant object or boundary with high confidence.

    python tools/plan_v2/viz_seen_but_hit.py --kind nc --n 6 --out data/analysis/viz
"""
import argparse
from pathlib import Path
from pathlib import Path as pathlib_Path

import numpy as np

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
REC = ROOT / "work_dirs/eval/para_ssr_v2_r34_aux/records"
CSV = ROOT / "work_dirs/eval_epdms/_exp/diag_para_ssr_v2_r34_refined_r1/2026.09.20.02.16.03/2026.09.20.02.31.25.csv"
DUMP = ROOT / "work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz"
ANCHORS = ROOT / "data/planning_vb/trajectory_anchors_256.npy"
MOTION = ROOT / "data/analysis/motion_preds_r34"
ROAD = 0
EGO_L, EGO_W = 4.049 + 1.127, 2.297          # nuPlan Pacifica: front + rear length, width
CACHE = pathlib_Path("/home/external-user/yongjae/navsim_v2_exp/exp/metric_cache_navtest")


def load_cache(token):
    """Human trajectory and the scorer's own drivable-area polygons, in the ego frame at t=0.

    The map head emits ROAD as POLYLINES; DAC is a point-in-POLYGON test against these. Drawing
    both side by side is the only way to see whether the boundary we predicted is the one crossed.
    """
    import lzma, pickle
    hits = list(CACHE.rglob(f"{token}/metric_cache.pkl"))
    if not hits:
        return None, None, None
    with lzma.open(hits[0], "rb") as f:
        mc = pickle.load(f)
    ra = mc.ego_state.rear_axle
    ox, oy, oh = float(ra.x), float(ra.y), float(ra.heading)
    c, s_ = np.cos(-oh), np.sin(-oh)
    R = np.array([[c, -s_], [s_, c]])

    def to_ego(pts):
        return (np.asarray(pts, np.float64)[:, :2] - [ox, oy]) @ R.T

    human = np.asarray(mc.human_trajectory.poses, np.float64) if mc.human_trajectory is not None else None
    polys = []
    dm = mc.drivable_area_map
    for g in getattr(dm, "_geometries", []):
        try:
            polys.append(to_ego(np.asarray(g.exterior.coords)))
        except Exception:
            continue
    # Where the agents ACTUALLY are at each of the 8 ego steps. The simulator runs at 0.1 s, so ego
    # step t (0.5 s apart) is simulation index 5*(t+1). This is what the scorer collides against.
    fut = []
    for t in range(8):
        try:
            om = mc.observation[5 * (t + 1)]
            fut.append([to_ego(np.asarray(om[k].exterior.coords)) for k in om.tokens])
        except Exception:
            fut.append([])
    return human, polys, fut


def ssr_xy_to_nav(p):
    """SSR BEV axes (x right, y forward) -> NAVSIM ego axes (x forward, y left).

    The aux records store detections and map points in SSR axes: boxes come from
    ``navsim_box_to_ssr`` and map points are denormalised over ``pc_range``. Trajectories, the human
    path and the drivable-area polygons are all NAVSIM. Plotting them together without this is a
    90 degree error that still looks like a plausible scene -- it is what invalidated the first
    version of this figure.
    """
    p = np.asarray(p, np.float64)
    return np.stack([p[..., 1], -p[..., 0]], axis=-1)


def ssr_box_to_nav(b):
    """Inverse of ``para_ssr_targets.navsim_box_to_ssr`` for the columns this plot uses."""
    b = np.asarray(b, np.float64)
    out = np.zeros((len(b), 7), np.float64)
    out[:, 0], out[:, 1] = b[:, 1], -b[:, 0]          # centre
    out[:, 3], out[:, 4] = b[:, 4], b[:, 3]           # ssr (width, length) -> (length, width)
    h = -b[:, 6] - np.pi
    out[:, 6] = np.arctan2(np.sin(h), np.cos(h))
    return out


def ego_corners(x, y, yaw):
    """[N] poses -> [N, 4, 2] footprint corners, rear axle at (x, y)."""
    fl, rl, hw = 4.049, 1.127, EGO_W / 2
    loc = np.array([[fl, hw], [fl, -hw], [-rl, -hw], [-rl, hw]], np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)     # [N, 2, 2]
    return np.einsum("nij,kj->nki", R, loc) + np.stack([x, y], -1)[:, None]


def box_corners(b):
    """[M, >=7] NAVSIM (x, y, z, length, width, h, yaw) -> [M, 4, 2]."""
    x, y, l, w, yaw = b[:, 0], b[:, 1], b[:, 3], b[:, 4], b[:, 6]
    loc = np.stack([np.stack([l / 2, w / 2], -1), np.stack([l / 2, -w / 2], -1),
                    np.stack([-l / 2, -w / 2], -1), np.stack([-l / 2, w / 2], -1)], 1)   # [M,4,2]
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    return np.einsum("mij,mkj->mki", R, loc) + np.stack([x, y], -1)[:, None]


def poly_gap(a, b):
    """Closest distance between two convex polygons [4,2] (0 if they touch/overlap)."""
    from matplotlib.path import Path as MPath
    if MPath(a).contains_points(b).any() or MPath(b).contains_points(a).any():
        return 0.0
    d = np.linalg.norm(a[:, None] - b[None], axis=-1).min()
    return float(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["nc", "dac"], default="nc")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--det-thr", type=float, default=0.3)
    ap.add_argument("--out", default=str(ROOT / "data/analysis/viz"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(CSV)
    df = df[df.token != "average"]
    col = "no_at_fault_collisions" if args.kind == "nc" else "drivable_area_compliance"
    bad = df[df[col] == 0].token.tolist()
    d = np.load(DUMP)
    at = {str(t): i for i, t in enumerate(d["tokens"])}
    anchors = np.load(ANCHORS).astype(np.float32)
    traj_all, off_all, fin_all = d["trajectory"], d["trajectory_offset"], d["plan_final_rewards"]
    print(f"{col} = 0 인 장면 {len(bad)}", flush=True)

    cases = []
    for tok in bad:
        f = REC / f"{tok}.npz"
        if not f.exists() or tok not in at:
            continue
        z0 = np.load(f, allow_pickle=True)
        z = {k: z0[k] for k in z0.files}
        for k in ("det_pred_boxes", "det_gt_boxes"):          # SSR -> NAVSIM, 한 곳에서만
            z[k] = ssr_box_to_nav(z[k]) if len(z[k]) else np.zeros((0, 7))
        for k in ("map_pred_points", "map_gt_points"):
            z[k] = ssr_xy_to_nav(z[k]) if len(z[k]) else np.zeros((0, 20, 2))
        # our motion head: per-step offsets in SSR axes, 6 modes, relative to the box centre
        mz = MOTION / f"{tok}.npz"
        if mz.exists():
            m0 = np.load(mz)
            mb, mt, mm = m0["boxes"].astype(np.float64), m0["traj"].astype(np.float64), m0["mode"]
            pred_fut = ssr_xy_to_nav(mb[:, None, None, :2] + np.cumsum(mt, axis=2))  # [N, 6, T, 2]
            pred_xy = ssr_xy_to_nav(mb[:, :2])
        else:
            pred_fut, pred_xy, mm = np.zeros((0, 6, 8, 2)), np.zeros((0, 2)), np.zeros(0, int)
        i = at[tok]
        # 출력 궤적: 배포가 고른 앵커 + 그 offset
        k = int(np.asarray(fin_all[i], np.float32).argmax())
        tr = (anchors[k] + np.asarray(off_all[i][k], np.float32)).astype(np.float64)   # [8, 3]
        ego = ego_corners(tr[:, 0], tr[:, 1], tr[:, 2])                                # [8, 4, 2]

        if args.kind == "nc":
            gt = z["det_gt_boxes"]
            if not len(gt):
                continue
            gtc = box_corners(gt.astype(np.float64))
            gaps = np.array([[poly_gap(e, g) for g in gtc] for e in ego])               # [8, M]
            j = int(gaps.min(0).argmin())
            if gaps[:, j].min() > 0.5:                     # 실제로 스치지도 않으면 제외
                continue
            pb, ps = z["det_pred_boxes"], z["det_pred_scores"]
            keep = ps >= args.det_thr
            if not keep.any():
                continue
            dist = np.linalg.norm(pb[keep, :2] - gt[j, :2], axis=1)
            m = int(dist.argmin())
            if dist[m] > 2.0:                              # 그 객체를 못 봤으면 "보고도"가 아니다
                continue
            cases.append(dict(token=tok, traj=tr, ego=ego, gt=gt, gtc=gtc, hit=j,
                              pred=pb[keep], score=ps[keep], match=m, conf=float(ps[keep][m]),
                              gap=float(gaps[:, j].min()), t=int(gaps[:, j].argmin()),
                              mp=z["map_pred_points"], ml=z["map_pred_labels"], ms=z["map_pred_scores"],
                              mg=z["map_gt_points"], mgl=z["map_gt_labels"],
                              pred_fut=pred_fut, pred_xy=pred_xy, pred_mode=mm))
        else:
            mg, mgl = z["map_gt_points"], z["map_gt_labels"]
            road = mg[mgl == ROAD]
            if not len(road):
                continue
            pts = np.concatenate(road, 0)
            dist = np.linalg.norm(tr[:, None, :2] - pts[None], axis=-1).min(1)
            if dist.min() > 3.0:
                continue
            mp, ms, ml = z["map_pred_points"], z["map_pred_scores"], z["map_pred_labels"]
            keep = (ms >= args.det_thr) & (ml == ROAD)
            if not keep.any():
                continue
            near = np.linalg.norm(np.concatenate(mp[keep], 0)[:, None] - pts[None], axis=-1).min(0)
            if np.median(near) > 2.0:                      # 경계를 못 봤으면 제외
                continue
            cases.append(dict(token=tok, traj=tr, ego=ego, gt=np.zeros((0, 9)), gtc=np.zeros((0, 4, 2)),
                              hit=-1, pred=np.zeros((0, 9)), score=np.zeros(0), match=-1,
                              conf=float(ms[keep].max()), gap=float(dist.min()), t=int(dist.argmin()),
                              mp=mp, ml=ml, ms=ms, mg=mg, mgl=mgl, seen=float(np.median(near)),
                              pred_fut=pred_fut, pred_xy=pred_xy, pred_mode=mm))
        if len(cases) >= args.n * 4:
            break

    cases.sort(key=lambda c: -c["conf"])
    cases = cases[: args.n]
    print(f"'보고도' 조건을 만족한 장면 {len(cases)}", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for c in cases:                       # 선택된 장면에 대해서만 캐시를 연다 (lzma 라 느리다)
        c["human"], c["polys"], c["fut"] = load_cache(c["token"])

    ncol = 3
    nrow = int(np.ceil(len(cases) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.2 * nrow), squeeze=False)
    for ax, c in zip(axes.ravel(), cases):
        for poly in (c["polys"] or []):    # 채점기가 실제로 쓰는 drivable area 폴리곤
            ax.add_patch(plt.Polygon(poly, closed=True, fc="#dfeaf5", ec="#9ec3e0", lw=.6, zorder=0))
        for p, l, s in zip(c["mp"], c["ml"], c["ms"]):
            if s >= args.det_thr:
                ax.plot(p[:, 0], p[:, 1], color="0.75", lw=1.0, zorder=1)
        for p, l in zip(c["mg"], c["mgl"]):
            ax.plot(p[:, 0], p[:, 1], color="0.35", lw=1.4, ls=(0, (4, 2)), zorder=2)
        for b in c["gtc"]:
            ax.add_patch(plt.Polygon(b, closed=True, fill=False, ec="#1b7837", lw=1.6, zorder=3))
        if len(c["gtc"]) and c["hit"] >= 0:
            ax.add_patch(plt.Polygon(c["gtc"][c["hit"]], closed=True, fc="#1b7837", alpha=.25,
                                     ec="#1b7837", lw=2.2, zorder=4))
        if len(c["pred"]):
            pc = box_corners(c["pred"].astype(np.float64))
            for b in pc:
                ax.add_patch(plt.Polygon(b, closed=True, fill=False, ec="#d95f02", lw=1.1, ls=":", zorder=5))
            if c["match"] >= 0:
                ax.add_patch(plt.Polygon(pc[c["match"]], closed=True, fill=False, ec="#d95f02",
                                         lw=2.4, zorder=6))
        # where the agents ACTUALLY are at the failure step -- what the scorer collided against
        for poly in (c["fut"][c["t"]] if c["fut"] else []):
            ax.add_patch(plt.Polygon(poly, closed=True, fc="#1b7837", alpha=.18,
                                     ec="#1b7837", lw=1.0, ls=(0, (3, 2)), zorder=3))
        # and where WE predicted they would be: motion head, 6 modes, argmax bold
        pf, px, pm = c["pred_fut"], c["pred_xy"], c["pred_mode"]
        if len(pf):
            near = np.linalg.norm(px[:, None] - c["traj"][None, :, :2], axis=-1).min(1) < 12.0
            for q in np.flatnonzero(near):
                for mo in range(pf.shape[1]):
                    ax.plot(pf[q, mo, :, 0], pf[q, mo, :, 1], "-", color="#d95f02",
                            lw=0.7, alpha=.30, zorder=5)
                b = int(pm[q]) if q < len(pm) else 0
                ax.plot(pf[q, b, :, 0], pf[q, b, :, 1], "-", color="#d95f02", lw=1.8, alpha=.95, zorder=6)
                ax.plot(pf[q, b, c["t"], 0], pf[q, b, c["t"], 1], "s", color="#d95f02",
                        ms=6, mfc="none", mew=2, zorder=6)
        for n, e in enumerate(c["ego"]):
            ax.add_patch(plt.Polygon(e, closed=True, fill=False, ec="#2166ac",
                                     lw=2.4 if n == c["t"] else 0.9, alpha=1.0 if n == c["t"] else .5, zorder=7))
        ax.plot(c["traj"][:, 0], c["traj"][:, 1], "-o", color="#2166ac", ms=3, lw=2, zorder=8)
        if c["human"] is not None:         # 사람이 실제로 간 길
            h = c["human"]
            ax.plot(h[:, 0], h[:, 1], "-", color="#e7298a", lw=2.2, zorder=8)
            ax.plot(h[::5, 0], h[::5, 1], "o", color="#e7298a", ms=3.2, zorder=8)
        ax.plot(0, 0, "k*", ms=11, zorder=9)
        extra = f"  det conf {c['conf']:.2f}" if args.kind == "nc" else f"  map conf {c['conf']:.2f}"
        ax.set_title(f"{c['token'][:12]}  t={0.5*(c['t']+1):.1f}s{extra}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xlim(-12, 42); ax.set_ylim(-20, 20)
        ax.grid(alpha=.25, lw=.5)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    lbl = ("collision (NC=0)" if args.kind == "nc" else "off-road (DAC=0)")
    fig.suptitle(f"v2_r34 navtest — {lbl}, and our own perception DID produce the evidence\n"
                 "blue = our output, bold box = failure step   pink = human path   pale blue = drivable area\n"
                 "green solid = GT boxes at t=0, dashed fill = GT agents AT the failure step   "
                 "orange line = our motion prediction (bold = top mode, square = its position at that step)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = out / f"seen_but_hit_{args.kind}.png"
    fig.savefig(p, dpi=130)
    print(f"-> {p}")
    for c in cases:
        print(f"  {c['token']}  최근접거리 {c['gap']:.2f} m  t={0.5*(c['t']+1):.1f}s  conf {c['conf']:.2f}")


if __name__ == "__main__":
    main()
