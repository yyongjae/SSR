"""Does the geometric EP label (feasibility.ep_label) compute the scorer's quantity?

    P1  project_on_polyline == shapely LineString.project on random NaN-padded polylines
    E1  straight route, human goes 20 m: a candidate going 10 m -> 0.5, 30 m -> 1.0 (capped by the
        max with its own progress), backwards -> 0
    E2  a candidate that leaves the drivable area gets 0 (the corridor verdict).  Control: the same
        candidate on an all-drivable field keeps its progress value
    E3  a scene without a route keeps its anchor label; mode=anchor is the identity
    E4  only the EP column changes, and nothing carries gradient
    R1  on REAL navtest scenes: our ego-frame resampled centerline + rear->centre conversion gives the
        same progress as shapely on the centerline the scorer stored in the metric cache

    python tools/plan_v2/test_ep_label.py
"""
import lzma
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
from navsim.agents.para_ssr.configs.default import ParaSSRConfig                         # noqa: E402
from navsim.agents.para_ssr.feasibility import (EP_COL, REAR_TO_CENTER, ep_label,       # noqa: E402
                                                project_on_polyline, target_shape)

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'통과' if cond else '실패 <<<'}  {name}  {detail}")


def main():
    from shapely.geometry import LineString, Point
    rng = np.random.default_rng(0)

    print("P1  polyline 투영 == shapely")
    errs = []
    for _ in range(50):
        n = rng.integers(3, 30)
        pts = np.cumsum(rng.normal(0, 3, (n, 2)), 0)
        line = np.full((40, 2), np.nan, np.float32); line[:n] = pts
        q = rng.normal(0, 15, (7, 2)).astype(np.float32)
        ours = project_on_polyline(torch.from_numpy(q)[None], torch.from_numpy(line)[None])[0].numpy()
        ls = LineString(pts)
        errs.append(np.abs(ours - np.array([ls.project(Point(*p)) for p in q])).max())
    check("최대 오차 < 1e-3 m", max(errs) < 1e-3, f"{max(errs):.2e}")

    cfg = ParaSSRConfig(feas_ep_target="geometric", feas_dyn_weight=0.0)
    th, tw = target_shape(cfg)
    T, K = 8, 4

    def straight(dist):           # [T, 3] rear-axle poses ending at x = dist
        p = torch.zeros(T, 3); p[:, 0] = torch.linspace(dist / T, dist, T); return p

    route = torch.full((161, 2), float("nan")); route[:120, 0] = torch.arange(-10, 110).float()
    route[:120, 1] = 0.0
    cands = [straight(10.0), straight(30.0), straight(-5.0), straight(20.0)]
    preds = {"trajectory_anchors": torch.stack(cands), "trajectory_offset": torch.zeros(1, K, T, 3)}
    free = torch.full((1, th, tw), 8.0)
    tgt = {"route_centerline": route[None], "trajectory": straight(20.0)[None],
           "drivable_sdf_bev": free, "sim_reward": torch.full((1, 5, K), 0.123)}

    print("E1  진행량 / max(사람 진행량, 자기 진행량)")
    new, logs = ep_label(preds, tgt, cfg, tgt["sim_reward"])
    ep = new[0, EP_COL]
    check("10 m → 0.5, 30 m → 1.0, 후진 → 0, 20 m → 1.0",
          torch.allclose(ep, torch.tensor([0.5, 1.0, 0.0, 1.0]), atol=0.02), f"{[round(float(v), 3) for v in ep]}")

    print("E2  주행영역 밖이면 0 (대조군: 전부 주행 가능하면 유지)")
    xs = torch.linspace(cfg.map_pc_range[1], cfg.map_pc_range[4], th)[:, None]   # rows = forward
    wall = torch.where(xs > 25.0, torch.tensor(-3.0), torch.tensor(8.0)).expand(th, tw)[None].clone()
    new2, _ = ep_label(preds, dict(tgt, drivable_sdf_bev=wall), cfg, tgt["sim_reward"])
    check("30 m 후보(도로 끝 25 m 넘음)는 0, 10 m 후보는 0.5 유지",
          float(new2[0, EP_COL, 1]) == 0.0 and abs(float(new2[0, EP_COL, 0]) - 0.5) < 0.02,
          f"{[round(float(v), 3) for v in new2[0, EP_COL]]}")

    print("E3  경로 없는 장면은 앵커 라벨 유지 / anchor 모드는 항등")
    new3, _ = ep_label(preds, dict(tgt, route_centerline=torch.full((1, 161, 2), float("nan"))), cfg, tgt["sim_reward"])
    check("경로 없음 → 앵커 라벨 그대로", torch.equal(new3, tgt["sim_reward"]))
    ident, lg = ep_label(preds, tgt, ParaSSRConfig(feas_ep_target="anchor"), tgt["sim_reward"])
    check("anchor 모드 항등", ident is tgt["sim_reward"] and lg == {})

    print("E4  EP 열만 바뀌고 기울기 없음")
    others = [i for i in range(5) if i != EP_COL]
    check("다른 열 그대로", torch.equal(new[:, others], tgt["sim_reward"][:, others]))
    check("기울기 없음", not new.requires_grad)

    print("R1  실제 navtest: 우리 중심선 + 중심 변환 == 채점기 중심선 위 진행량")
    f = ROOT / "work_dirs/route_centerline_navtest_limit600.npz"
    if not f.exists():
        print("  (건너뜀: 중심선 파일 없음)")
    else:
        sys.path.insert(0, str(ROOT / "tools/readout"))
        import _env  # noqa: F401
        from navsim.common.dataloader import MetricCacheLoader
        z = np.load(f); mc = MetricCacheLoader(ROOT / "data/exp/metric_cache")
        diffs = []
        for t, line in list(zip([str(x) for x in z["tokens"]], z["line"]))[:150]:
            if t not in mc.metric_cache_paths or np.isnan(line[1, 0]):
                continue
            with lzma.open(mc.metric_cache_paths[t], "rb") as fh:
                cache = pickle.load(fh)
            ego = cache.ego_state; x0, y0, h = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
            c, s = np.cos(h), np.sin(h)
            q_ego = np.array([[REAR_TO_CENTER, 0.0], [15.0 + REAR_TO_CENTER, 0.0], [25.0, 2.0]], np.float32)
            ours = project_on_polyline(torch.from_numpy(q_ego)[None], torch.from_numpy(line)[None])[0].numpy()
            ours = ours[1:] - ours[0]
            g = np.stack([x0 + c * q_ego[:, 0] - s * q_ego[:, 1], y0 + s * q_ego[:, 0] + c * q_ego[:, 1]], -1)
            ref = cache.centerline._linestring
            refp = np.array([ref.project(Point(*p)) for p in g]); refp = refp[1:] - refp[0]
            diffs.append(np.abs(ours - refp).max())
        diffs = np.array(diffs)
        check("진행량 차이 median < 0.2 m", np.median(diffs) < 0.2,
              f"{len(diffs)} 장면  median {np.median(diffs):.3f} m  p90 {np.percentile(diffs, 90):.3f} m")

    print(f"\n{sum(OK)}/{len(OK)} 통과")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
