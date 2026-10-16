"""How much does the LQR tracking change the verdict the geometric cost is trying to teach?

The official DAC is a point-in-polygon test on the SIMULATED ego states -- the plan after an LQR
controller and a kinematic bicycle have tracked it.  The feasibility cost is computed on the PLANNED
trajectory, because that is what the offset head can move.  If the two disagree often, the cost
teaches the wrong thing no matter how accurate the field is.

Same polygon, same test, two trajectories:

    raw       the candidate as emitted (anchor + offset), footprint corners at the 8 plan steps
    tracked   what the scorer actually judged -- read straight from the official per-candidate labels

    python tools/plan_v2/raw_vs_tracked.py [--scenes 300] [--workers 6]
"""
import argparse
import lzma
import pickle
import sys
from pathlib import Path

import numpy as np

# Deliberately imports nothing from this repo: the metric caches were written by the navsim v2
# worktree, and unpickling them needs THAT navsim on the path, not this one.
ROOT = Path("/home/external-user/kyungmin/SSR-v2")
CACHE = Path("/home/external-user/yongjae/navsim_v2_exp/exp/metric_cache_navtest")
EGO = ((4.049, 1.1485), (4.049, -1.1485), (-1.127, -1.1485), (-1.127, 1.1485))


def corners(traj):
    """[K, T, 3] rear-axle poses -> [K, T, 4, 2]."""
    c, s = np.cos(traj[..., 2:3]), np.sin(traj[..., 2:3])
    out = []
    for dx, dy in EGO:
        out.append(np.stack([traj[..., 0] + c[..., 0] * dx - s[..., 0] * dy,
                             traj[..., 1] + s[..., 0] * dx + c[..., 0] * dy], -1))
    return np.stack(out, axis=-2)


def one(args):
    tok, traj = args
    import navsim.common.enums as _e
    import enum as _enum
    if not hasattr(_e, "SceneFrameType"):
        _e.SceneFrameType = _enum.IntEnum("SceneFrameType", {"ORIGINAL": 0, "SYNTHETIC": 1})
    hits = list(CACHE.rglob(f"{tok}/metric_cache.pkl"))
    if not hits:
        return None
    with lzma.open(hits[0], "rb") as f:
        mc = pickle.load(f)
    ra = mc.ego_state.rear_axle
    c, s = np.cos(float(ra.heading)), np.sin(float(ra.heading))
    R = np.array([[c, -s], [s, c]])
    pts = corners(traj).reshape(-1, 2) @ R.T + [float(ra.x), float(ra.y)]      # ego -> global
    inside = mc.drivable_area_map.points_in_polygons(pts[None]).any(0)[0]
    return tok, inside.reshape(traj.shape[0], traj.shape[1], 4).all(axis=(1, 2))   # [K] raw DAC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor

    z = np.load(ROOT / "data/planning_vb/navtest_refined_scores.npz", allow_pickle=True)
    cols = [str(c) for c in z["columns"]]
    ci = cols.index("drivable_area_compliance")
    lab_at = {str(t): i for i, t in enumerate(z["tokens"])}
    d = np.load(ROOT / "work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz")
    d_at = {str(t): i for i, t in enumerate(d["tokens"])}
    anchors = np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy").astype(np.float32)
    toks = sorted(set(lab_at) & set(d_at))
    rng = np.random.default_rng(0)
    toks = [toks[i] for i in rng.permutation(len(toks))[: args.scenes]]
    raw_scores = np.asarray(z["scores"])[[lab_at[t] for t in toks]][:, :256, ci]    # tracked DAC
    offs = d["trajectory_offset"]
    jobs = [(t, (anchors + np.asarray(offs[d_at[t]], np.float32)).astype(np.float64)) for t in toks]

    got = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for n, r in enumerate(ex.map(one, jobs)):
            if r:
                got[r[0]] = r[1]
            if (n + 1) % 100 == 0:
                print(f"  {n + 1}/{len(jobs)}", flush=True)

    keep = [i for i, t in enumerate(toks) if t in got]
    raw = np.stack([got[toks[i]] for i in keep]).astype(np.float64)                # [N, 256]
    trk = raw_scores[keep].astype(np.float64)
    print(f"\n{len(keep)} 장면 x 256 후보.  같은 폴리곤, 같은 판정, 궤적만 다름")
    print(f"  원 궤적 기준 DAC 통과율   {100 * raw.mean():.2f}%")
    print(f"  추종 궤적 기준 (공식)      {100 * trk.mean():.2f}%")
    agree = (raw == trk).mean()
    print(f"\n  후보 단위 일치율          {100 * agree:.2f}%")
    print(f"    원=통과, 추종=실패      {100 * ((raw == 1) & (trk == 0)).mean():.2f}%  <- 추종이 깬 것")
    print(f"    원=실패, 추종=통과      {100 * ((raw == 0) & (trk == 1)).mean():.2f}%  <- 추종이 살린 것")
    fin = np.asarray(d["plan_final_rewards"][[d_at[toks[i]] for i in keep]], np.float64)
    p = fin.argmax(1)
    ar = np.arange(len(keep))
    print(f"\n  배포가 고른 후보만: 원 {100 * raw[ar, p].mean():.2f}%  추종 {100 * trk[ar, p].mean():.2f}%"
          f"  일치 {100 * (raw[ar, p] == trk[ar, p]).mean():.2f}%")


if __name__ == "__main__":
    main()
