"""Why does a run fail lane keeping (LK, EPDMS)?

LK (navsim_v2 ``PDMScorer._calculate_lane_keeping``): the ego box centre, at 0.1 s over the
4 s horizon of the LQR-tracked trajectory, must not stay > 0.5 m from the route centerline
(``metric_cache.centerline``) for 2 s in a row; points inside an INTERSECTION are skipped.
With ``human_penalty_filter`` on, a scene whose human trajectory fails LK scores 1 for every
planner, so every LK failure below is one where the human kept the lane.

Each failing scene's submitted trajectory (not tracked) is replayed with the same rule:
    not_reproduced   the plan itself passes; the failure appears only after LQR tracking
    offset_in_lane   the plan exceeds 0.5 m but its centre stays <= --lane-half m from the
                     centerline: a lateral bias / corner cutting inside the right lane
    other_lane       the centre goes > --lane-half m off: lane change or wrong lane
A random sample of LK-passing scenes checks the replay (it should pass them too).

    python tools/plan_v2/lk_attribution.py --csv <epdms csv> --npz <dump npz> --out <csv>
"""
import argparse
import lzma
import pickle
import sys

import numpy as np
import pandas as pd

NV2 = "/home/external-user/yongjae/navsim_v2"
CACHE = "/home/external-user/yongjae/navsim_v2_exp/exp/metric_cache_navtest"
REAR_AXLE_TO_CENTER = 1.461       # get_pacifica_parameters()
LIMIT, WINDOW, DT = 0.5, 2.0, 0.1  # pdm_scorer.yaml lane_keeping_*, proposal interval


def dense(poses):
    """[8, 3] local rear-axle poses at 0.5 s -> [41, 3] at 0.1 s, origin included."""
    p = np.concatenate([np.zeros((1, 3)), poses], axis=0)
    t, ts = np.arange(len(p)), np.linspace(0, len(p) - 1, 41)
    h = np.unwrap(p[:, 2])
    return np.stack([np.interp(ts, t, p[:, 0]), np.interp(ts, t, p[:, 1]), np.interp(ts, t, h)], axis=1)


def centres_global(poses, origin):
    x0, y0, h0 = origin
    c, s = np.cos(h0), np.sin(h0)
    d = dense(poses)
    gx, gy, gh = x0 + c * d[:, 0] - s * d[:, 1], y0 + s * d[:, 0] + c * d[:, 1], h0 + d[:, 2]
    return np.stack([gx + REAR_AXLE_TO_CENTER * np.cos(gh), gy + REAR_AXLE_TO_CENTER * np.sin(gh)], axis=1)


def replay(centres, mc):
    """Same rule as the scorer -> (fails, lateral deviation [41], in_intersection [41])."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from shapely.geometry import Point

    line = mc.centerline.linestring
    need = int(np.ceil(WINDOW / DT))
    dev, inter = np.zeros(len(centres)), np.zeros(len(centres), bool)
    run, fails = 0, False
    for i, (x, y) in enumerate(centres):
        pt = Point(x, y)
        inter[i] = mc.drivable_area_map.is_in_layer(pt, layer=SemanticMapLayer.INTERSECTION)
        dev[i] = pt.distance(line)
        if inter[i]:
            continue
        run = run + 1 if dev[i] > LIMIT else 0
        fails |= run >= need
    return fails, dev, inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lane-half", type=float, default=1.75)
    ap.add_argument("--calib", type=int, default=300)
    args = ap.parse_args()
    sys.path.insert(0, NV2)

    df = pd.read_csv(args.csv)
    df = df[df.token != "average_all_frames"].set_index("token")
    lk = df.lane_keeping.astype(float)
    fail = sorted(lk.index[lk < 1])
    rng = np.random.default_rng(0)
    calib = sorted(rng.choice(sorted(lk.index[lk == 1]), size=min(args.calib, int((lk == 1).sum())), replace=False))
    z = np.load(args.npz, allow_pickle=True)
    idx = {str(t): i for i, t in enumerate(z["tokens"])}
    cmd = z["command"] if "command" in z.files else None
    meta = pd.read_csv(f"{CACHE}/metadata/metric_cache_navtest_metadata_node_0.csv").file_name
    path = {p.split("/")[-2]: p for p in meta}

    rows = []
    for n, tok in enumerate(fail + calib):
        with lzma.open(path[tok], "rb") as f:
            mc = pickle.load(f)
        ra = mc.ego_state.rear_axle
        origin = (ra.x, ra.y, ra.heading)
        p_fail, p_dev, p_int = replay(centres_global(z["trajectory"][idx[tok]].astype(np.float64), origin), mc)
        h_fail, h_dev, _ = replay(centres_global(np.asarray(mc.human_trajectory.poses, np.float64), origin), mc)
        out = ~p_int
        row = dict(token=tok, lk=float(lk[tok]), plan_fails=p_fail, human_fails=h_fail,
                   plan_max_dev=float(p_dev[out].max()) if out.any() else np.nan,
                   human_max_dev=float(h_dev[out].max()) if out.any() else np.nan,
                   plan_end_dev=float(p_dev[-1]), human_end_dev=float(h_dev[-1]),
                   intersection_frac=float(p_int.mean()),
                   human_turn_deg=float(np.degrees(abs(mc.human_trajectory.poses[-1, 2]))),
                   ego_speed=float(np.hypot(*mc.ego_state.dynamic_car_state.rear_axle_velocity_2d.array)))
        if cmd is not None:
            row["command"] = int(np.argmax(cmd[idx[tok]])) if np.ndim(cmd[idx[tok]]) else int(cmd[idx[tok]])
        if lk[tok] < 1:
            row["cause"] = ("not_reproduced" if not p_fail else
                            "other_lane" if row["plan_max_dev"] > args.lane_half else "offset_in_lane")
        rows.append(row)
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(fail) + len(calib)}", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(args.out, index=False)
    f, c = r[r.lk < 1], r[r.lk == 1]
    print(f"\nLK 평균 {100 * lk.mean():.2f}, 실패 {len(f)} / {len(lk)}")
    print(f"재현 검사: 통과 표본 {len(c)}개 중 궤적 그대로 통과 {100 * (~c.plan_fails).mean():.1f}%")
    for k, v in f.cause.value_counts().items():
        print(f"  {k:16s} {v:4d} ({100 * v / len(f):4.1f}%)")
    for k in ("offset_in_lane", "other_lane", "not_reproduced"):
        g = f[f.cause == k]
        if len(g):
            print(f"  [{k}] plan 최대 이탈 중앙값 {g.plan_max_dev.median():.2f} m, 사람 {g.human_max_dev.median():.2f} m, "
                  f"사람 회전각 중앙값 {g.human_turn_deg.median():.1f} deg, 속도 {g.ego_speed.median():.1f} m/s")
    if "command" in f:
        print("  command 분포 (실패):", f.command.value_counts().sort_index().to_dict(),
              "/ (통과 표본):", c.command.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
