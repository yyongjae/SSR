"""D4: why does a planner fail navhard stage two -- the start state or the driving?

Stage-two scenes are 3DGS renders at perturbed ego states.  For every stage-one / stage-two
scene of a navhard two-stage csv, the start state is read from the navhard metric cache: the
rear-axle distance to the route centerline (the one LK measures) and the heading difference
to the centerline direction.  Failure rates are then broken down by those offsets.  If stage-two
failures concentrate at large start offsets, the collapse is a start-state (covariate) shift;
if they are flat across offsets, it is not.

    python tools/plan_v2/navhard_stage2_attribution.py --csv <navhard csv> --out <csv>
"""
import argparse
import glob
import lzma
import pickle
import sys

import numpy as np
import pandas as pd

NV2 = "/home/external-user/yongjae/navsim_v2"
CACHE = "/home/external-user/yongjae/navsim_v2_exp/exp/metric_cache_navhard"
M = {"lane_keeping": "LK", "drivable_area_compliance": "DAC", "no_at_fault_collisions": "NC",
     "time_to_collision_within_bound": "TTC", "driving_direction_compliance": "DDC"}


def start_state(mc):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from shapely.geometry import Point

    ra = mc.ego_state.rear_axle
    line = mc.centerline.linestring
    pt = Point(ra.x, ra.y)
    lat = pt.distance(line)
    s = line.project(pt)
    if not np.isfinite(s):
        return lat, np.nan, np.nan, False
    p0, p1 = line.interpolate(max(s - 1.0, 0.0)), line.interpolate(s + 1.0)
    dh = (ra.heading - np.arctan2(p1.y - p0.y, p1.x - p0.x) + np.pi) % (2 * np.pi) - np.pi
    speed = float(np.hypot(*mc.ego_state.dynamic_car_state.rear_axle_velocity_2d.array))
    inter = mc.drivable_area_map.is_in_layer(pt, layer=SemanticMapLayer.INTERSECTION)
    return lat, abs(np.degrees(dh)), speed, bool(inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, NV2)
    d = pd.read_csv(args.csv, index_col=0)
    d = d[~d.token.str.startswith("extended_pdm_score")]
    meta = pd.read_csv(glob.glob(f"{CACHE}/metadata/*.csv")[0]).file_name
    path = {p.split("/")[-2]: p for p in meta}
    rows = []
    for n, (_, r) in enumerate(d.iterrows()):
        stage = "stage_one" if pd.notna(r["no_at_fault_collisions_stage_one"]) else "stage_two"
        with lzma.open(path[r.token], "rb") as f:
            mc = pickle.load(f)
        lat, dh, v, inter = start_state(mc)
        row = dict(token=r.token, stage=stage, lateral=lat, heading_deg=dh, speed=v, in_intersection=inter,
                   score=float(r.score))
        row.update({M[c]: float(r[f"{c}_{stage}"]) for c in M})
        rows.append(row)
        if (n + 1) % 1000 == 0:
            print(f"  {n + 1}/{len(d)}", flush=True)
    x = pd.DataFrame(rows)
    x.to_csv(args.out, index=False)

    def rates(g):
        return " ".join(f"{k} {100 * (g[k] < 1).mean():5.1f}" for k in M.values()) + f"  | 점수 {100 * g.score.mean():5.1f}"

    print("\n실패율(%) = 1 - 통과율")
    for st in ("stage_one", "stage_two"):
        g = x[x.stage == st]
        print(f"{st:9s} n={len(g):5d}  시작 횡이탈 중앙값 {g.lateral.median():.2f} m, heading 차 중앙값 {g.heading_deg.median():.1f}°")
        print(f"          {rates(g)}")
    s2 = x[x.stage == "stage_two"]
    print("\nstage two, 시작 횡이탈별 (교차로 밖)")
    o = s2[~s2.in_intersection]
    for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)):
        g = o[(o.lateral >= lo) & (o.lateral < hi)]
        if len(g):
            print(f"  [{lo:.1f}, {hi:.1f}) m  n={len(g):5d} ({100 * len(g) / len(o):4.1f}%)  {rates(g)}")
    print("stage two, 시작 heading 차별 (교차로 밖)")
    for lo, hi in ((0, 5), (5, 15), (15, 999)):
        g = o[(o.heading_deg >= lo) & (o.heading_deg < hi)]
        if len(g):
            print(f"  [{lo}, {hi})°  n={len(g):5d} ({100 * len(g) / len(o):4.1f}%)  {rates(g)}")
    g = s2[s2.in_intersection]
    print(f"stage two, 교차로 안에서 시작  n={len(g)}  {rates(g) if len(g) else ''}")


if __name__ == "__main__":
    main()
