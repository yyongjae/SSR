"""Of the DAC the corridor still loses, how much could the corridor ever have seen?

The corridor cost is computed on the trajectory the model EMITS.  The official DAC is computed on
what an LQR controller and a bicycle model make of that trajectory.  So a failure falls into one of
two kinds, and they call for opposite work:

    cost > 0   the emitted footprint really does leave the drivable area, and our cost saw it.
               The term exists and did not stop it -> weight, margin, or the gate.
    cost ~ 0   the emitted footprint is inside; only the tracked one leaves.  No field, however
               perfect, can see this -> the cost has to move to the tracked trajectory.

The field here is the GROUND TRUTH static field, built through the same path the training target
uses (`plan_map.DrivableAreaTargetBuilder`, verified cell-for-cell against the official scorer's own
polygons on 80 scenes).  So this measures the mechanism's ceiling, not the predicted field's quality.

    python tools/plan_v2/dac_reachability.py --exp para_ssr_v2_r34_feas_gt_ep29 [--control 300]
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/readout"))
import _env  # noqa: F401,E402
from _env import scene_filter, split_dirs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--control", type=int, default=300, help="DAC-passing scenes, as a control")
    ap.add_argument("--margin", type=float, default=0.0, help="0 = the violation itself")
    args = ap.parse_args()

    import pandas as pd
    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    from navsim.agents.para_ssr.feasibility import (corner_sdf, cost_from_sdf, rasterize_bev_sdf,
                                                    target_shape)
    from navsim.agents.para_ssr.plan_map import DrivableAreaTargetBuilder
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    csv = sorted(glob.glob(str(ROOT / f"work_dirs/eval_epdms/_exp/{args.exp}/*/*.csv")),
                 key=lambda p: Path(p).stat().st_mtime)[-1]
    d = pd.read_csv(csv)
    d = d[d.token != "average_all_frames"]
    dac = {str(t): float(v) for t, v in zip(d.token, d.drivable_area_compliance)}
    print(f"채점 csv: {Path(csv).name}   DAC=0 {sum(v == 0 for v in dac.values())} / {len(dac)}")

    z = np.load(ROOT / f"work_dirs/eval_epdms/{args.exp}/navtest_plan.npz", allow_pickle=True)
    traj_at = {str(t): i for i, t in enumerate(z["tokens"])}
    traj = z["trajectory"]                                   # [N, 8, 3] emitted, NAVSIM ego

    fail = [t for t, v in dac.items() if v == 0 and t in traj_at]
    ok = [t for t, v in dac.items() if v == 1 and t in traj_at]
    rng = np.random.default_rng(0)
    ctrl = [ok[i] for i in rng.permutation(len(ok))[: args.control]]
    print(f"측정 대상: 실패 {len(fail)}개 전부 + 통과 대조군 {len(ctrl)}개")

    cfg = ParaSSRConfig()
    pc = cfg.map_pc_range
    H, W = target_shape(cfg)
    no_sensors = SensorConfig(**{k: False for k in (
        "cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0", "lidar_pc")})
    loader = SceneLoader(*split_dirs("test"), scene_filter("navtest"), sensor_config=no_sensors)
    builder = DrivableAreaTargetBuilder(cfg)

    def cost_of(tok):
        scene = loader.get_scene_from_token(tok)
        cur = scene.scene_metadata.num_history_frames - 1
        poly = builder.drivable_polygon(scene, cur)
        if poly is None or poly.is_empty:
            return None
        f = torch.as_tensor(rasterize_bev_sdf(poly, pc, H, W, 8.0))[None]
        tj = torch.as_tensor(traj[traj_at[tok]], dtype=torch.float32)[None, None]   # [1,1,8,3]
        s = corner_sdf(f, tj, pc)
        return float(cost_from_sdf(s, args.margin, 0.2)[0, 0]), float(s.min())

    out = {}
    for name, toks in (("fail", fail), ("ctrl", ctrl)):
        rows = []
        for n, t in enumerate(toks):
            try:
                r = cost_of(t)
            except Exception:
                r = None
            if r is not None:
                rows.append(r)
            if (n + 1) % 100 == 0:
                print(f"  {name} {n + 1}/{len(toks)}", flush=True)
        out[name] = np.asarray(rows)                      # [M, 2] = (cost, min corner sdf)

    f_cost, f_sdf = out["fail"][:, 0], out["fail"][:, 1]
    c_cost, c_sdf = out["ctrl"][:, 0], out["ctrl"][:, 1]
    saw = f_sdf < 0                                        # a corner really is outside
    print(f"\n=== 공식 DAC=0 장면 {len(f_cost)}개, 배출 궤적을 GT 정적 장에 직접 대본 결과 ===")
    print(f"  코너가 실제로 도로 밖  (우리 비용이 볼 수 있었던 실패)  {saw.sum():4d}  {100*saw.mean():5.1f}%")
    print(f"  코너가 전부 도로 안    (추종 시뮬에서만 이탈)          {(~saw).sum():4d}  {100*(~saw).mean():5.1f}%")
    print(f"\n  볼 수 있었던 실패의 침투 깊이  중앙 {np.median(-f_sdf[saw]):.3f} m"
          f"  90% {np.percentile(-f_sdf[saw], 90):.3f} m")
    print(f"  비용(margin {args.margin})            중앙 {np.median(f_cost[saw]):.2f}")
    print(f"\n  대조군(공식 DAC=1) {len(c_cost)}개 중 코너가 밖인 비율  {100*(c_sdf < 0).mean():5.1f}%"
          f"   <- 낮을수록 위 수치가 진짜")

    n_fail_total = sum(v == 0 for v in dac.values())
    pts = 100.0 * n_fail_total / len(dac)
    print(f"\n=== DAC 100까지 남은 {pts:.2f}점의 분해 (측정 표본 비율로 환산) ===")
    print(f"  기구가 손댈 수 있는 몫 (코너가 밖)      {pts * saw.mean():.2f}점")
    print(f"  추종 궤적으로 옮겨야 닿는 몫            {pts * (~saw).mean():.2f}점")


if __name__ == "__main__":
    main()
