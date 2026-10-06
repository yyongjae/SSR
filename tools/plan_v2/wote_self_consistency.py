"""Does WoTE (public checkpoint, PDMS 88.3) plan against its OWN predicted map?

For every navtest scene where WoTE fails drivable-area compliance (DAC), find where the
planned footprint first leaves the drivable area the metric uses, and ask what WoTE's own
BEV semantic map said about that spot (road-like class or not).  Same question as
SSR/data/analysis/boundary_precision.py asked of PARA-SSR v1.

WoTE's map is a 0.25 m raster (32 m ahead, +-32 m), so exits shallower than ~0.25 m cannot
be told apart; they are reported separately.  The raster's axis convention is found
empirically: the transform (of 8 flips/transposes) with the best IoU against the true road.

    python tools/plan_v2/wote_self_consistency.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V2 / "tools/readout"))
sys.path.insert(0, "/home/external-user/kyungmin/SSR/data/analysis")
import _env  # noqa: F401,E402
from _env import scene_filter, split_dirs  # noqa: E402
from failure_attribution import ego_corners, interp_poses, sdf_at  # noqa: E402

RES = 0.25


def true_road_raster(scene, cur_idx):
    """[128, 256] bool, row = x forward / 0.25, col = (y left + 32) / 0.25, cell centres."""
    import shapely
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from shapely.ops import unary_union
    from navsim.agents.para_ssr.para_ssr_targets import _geometry_local_coords

    pose = scene.frames[cur_idx].ego_status.ego_pose
    origin = StateSE2(float(pose[0]), float(pose[1]), float(pose[2]))
    objs = scene.map_api.get_proximal_map_objects(origin.point, 50.0, [L.LANE, L.INTERSECTION])
    polys = [o.polygon for layer in (L.LANE, L.INTERSECTION) for o in objs.get(layer, [])]
    polys = [p for p in polys if p is not None and not p.is_empty]
    if not polys:
        return np.zeros((128, 256), bool)
    road = _geometry_local_coords(unary_union(polys), origin)
    xs = (np.arange(128) + 0.5) * RES
    ys = -32 + (np.arange(256) + 0.5) * RES
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    shapely.prepare(road)
    return shapely.contains_xy(road, gx.ravel(), gy.ravel()).reshape(128, 256)


TRANSFORMS = {
    "id": lambda m: m, "flip_r": lambda m: m[::-1], "flip_c": lambda m: m[:, ::-1],
    "flip_rc": lambda m: m[::-1, ::-1],
}


def main():
    from dataclasses import replace
    from navsim.common.dataloader import SceneLoader
    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    from navsim.agents.para_ssr.plan_map import DrivableAreaTargetBuilder

    z = np.load(V2 / "data/wote/navtest_dump.npz")
    shape = tuple(z["shape"])
    idx = {t: i for i, t in enumerate(z["tokens"])}
    csv = sorted(glob.glob(str(V2 / "work_dirs/eval/wote_public/*.csv")))[-1]
    pdms = pd.read_csv(csv)
    pdms = pdms[pdms.token != "average"].set_index("token")
    dac = pdms.drivable_area_compliance.astype(float)
    fail = sorted(dac.index[dac < 1])
    rng = np.random.default_rng(0)
    calib = sorted(rng.choice(sorted(dac.index[dac == 1]), size=150, replace=False))
    print(f"WoTE PDMS {100 * pdms.score.astype(float).mean():.2f}, DAC {100 * dac.mean():.2f}, DAC 실패 {len(fail)}", flush=True)

    logs_dir, blobs_dir = split_dirs("test")
    loader = SceneLoader(logs_dir, blobs_dir, scene_filter("navtest", tokens=fail + calib))
    cfg = replace(ParaSSRConfig(), plan_map_weight=1.0)
    db = DrivableAreaTargetBuilder(cfg)
    extent, res = tuple(cfg.plan_map_extent), float(cfg.plan_map_res)

    def pred_mask(tok):
        return np.unpackbits(z["road"][idx[tok]])[: shape[0] * shape[1]].reshape(shape).astype(bool)

    # 1) axis convention
    iou = {k: [] for k in TRANSFORMS}
    for tok in calib:
        scene = loader.get_scene_from_token(tok)
        truth = true_road_raster(scene, scene.scene_metadata.num_history_frames - 1)
        for k, f in TRANSFORMS.items():
            p = f(pred_mask(tok))
            iou[k].append((p & truth).sum() / max((p | truth).sum(), 1))
    best = max(iou, key=lambda k: np.mean(iou[k]))
    print("raster 축 후보별 road IoU:", {k: round(float(np.mean(v)), 3) for k, v in iou.items()}, "->", best, flush=True)
    tf = TRANSFORMS[best]

    # 2) exit points of DAC failures vs WoTE's own map
    rows = []
    for n, tok in enumerate(fail):
        scene = loader.get_scene_from_token(tok)
        sdf = db.compute_targets(scene)["drivable_sdf"].float().numpy()
        plan = interp_poses(z["trajectory"][idx[tok]].astype(np.float64))
        exit_ = None
        for s, pose in enumerate(plan):
            for cx, cy in ego_corners(pose):
                v = sdf_at(sdf, extent, res, cx, cy)
                if np.isfinite(v) and v < -0.1:
                    exit_ = (s, cx, cy, v)
                    break
            if exit_:
                break
        row = dict(token=tok)
        if exit_ is None:
            row["cause"] = "no_exit_found"
        else:
            s, cx, cy, v = exit_
            row.update(t=s * 0.1, depth=-v, x=cx, y=cy)
            i, j = int(cx / RES), int((cy + 32) / RES)
            if not (0 <= i < 128 and 0 <= j < 256):
                row["cause"] = "beyond_map_range"
            else:
                m = tf(pred_mask(tok))
                i0, i1, j0, j1 = max(i - 1, 0), min(i + 2, 128), max(j - 1, 0), min(j + 2, 256)
                row["own_road_3x3"] = float(m[i0:i1, j0:j1].mean())
                row["cause"] = "own_map_says_road" if m[i, j] else "own_map_says_offroad"
        rows.append(row)
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(fail)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(V2 / "data/wote/self_consistency.csv", index=False)
    vc = df.cause.value_counts()
    print(f"\nWoTE DAC 실패 {len(df)}개")
    for k, v in vc.items():
        print(f"  {k:22s} {v:4d} ({100 * v / len(df):4.1f}%)")
    e = df[df.cause.isin(["own_map_says_road", "own_map_says_offroad"])]
    if len(e):
        print(f"\nmap 범위 안에서 재현된 이탈 {len(e)}개: 이탈 깊이 중앙값 {e.depth.median():.2f} m")
        print(f"  자기 map 이 '도로 밖'이라 본 비율: {100 * (e.cause == 'own_map_says_offroad').mean():.1f}%")
        deep = e[e.depth >= 0.3]
        if len(deep):
            print(f"  이탈 깊이 >= 0.3 m (raster 해상도 이상) {len(deep)}개 중: {100 * (deep.cause == 'own_map_says_offroad').mean():.1f}%")


if __name__ == "__main__":
    main()
