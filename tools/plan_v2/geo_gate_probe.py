"""If the corridor decided admissibility at INFERENCE, how much of the lost DAC comes back?

Today the gate reads a learned scalar (`sim_rewards[DAC]`).  The model also predicts a drivable
field at inference and throws it away.  This asks what happens if the emitted footprint is instead
checked against that field directly, candidate by candidate, and violating candidates are dropped
before the argmax.  Nothing is retrained: the checkpoint, the candidates and the field are the
ones the evaluated model produced.

Three selections are compared on the same dump, all scored against the SAME per-candidate ground
truth (WoTE's per-anchor labels -- see the caveat below):

    deployed     argmax of plan_final_rewards, i.e. what was evaluated
    geo-gate     drop candidates whose emitted footprint violates the PREDICTED field, then argmax
    geo-gate-gt  the same with the GROUND TRUTH field, which separates "the gate idea" from
                 "the predicted field's quality"

CAVEAT, stated because it bounds what this can conclude: the per-candidate labels describe the
ANCHOR, and the selections here are over anchor+offset.  So the numbers are a ranking probe, not a
score prediction.  What the probe is good for is the DIRECTION and the relative size of the three
selections, and the false-reject / false-admit rates of the gate itself, which are exact.

    python tools/plan_v2/geo_gate_probe.py --exp para_ssr_v2_r34_feas_gt_ep29
"""
import argparse
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
    ap.add_argument("--scenes", type=int, default=1500)
    ap.add_argument("--margins", default="0.0,0.1,0.2,0.4",
                    help="게이트 여유를 쓸어본다 — 0에서는 거의 아무도 안 떨어진다")
    args = ap.parse_args()

    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    from navsim.agents.para_ssr.feasibility import (corner_sdf, rasterize_bev_sdf, target_shape)
    from navsim.agents.para_ssr.plan_map import DrivableAreaTargetBuilder
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    z = np.load(ROOT / f"work_dirs/eval_epdms/{args.exp}/cand_field.npz", allow_pickle=True)
    toks = [str(t) for t in z["tokens"]]
    off = z["trajectory_offset"].astype(np.float32)          # [N, K, T, 3]
    fin = z["plan_final_rewards"].astype(np.float32)         # [N, K]
    fld = z["drivable_sdf_pred"].astype(np.float32)          # [N, bev_h, bev_w]
    anchors = np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy").astype(np.float32)
    print(f"덤프 {len(toks)} 장면, 후보 {off.shape[1]}, 예측 장 {fld.shape[1:]}")

    lab = np.load(ROOT / "data/planning_vb/navtest_refined_scores.npz", allow_pickle=True)
    cols = [str(c) for c in lab["columns"]]
    ci, ni = cols.index("drivable_area_compliance"), cols.index("no_at_fault_collisions")
    lab_at = {str(t): i for i, t in enumerate(lab["tokens"])}
    scores = np.asarray(lab["scores"])

    cfg = ParaSSRConfig()
    pc = cfg.map_pc_range
    Ht, Wt = target_shape(cfg)
    no_sensors = SensorConfig(**{k: False for k in (
        "cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0", "lidar_pc")})
    loader = SceneLoader(*split_dirs("test"), scene_filter("navtest"), sensor_config=no_sensors)
    builder = DrivableAreaTargetBuilder(cfg)

    use = [t for t in toks if t in lab_at]
    rng = np.random.default_rng(0)
    use = [use[i] for i in rng.permutation(len(use))[: args.scenes]]
    tok_at = {t: i for i, t in enumerate(toks)}

    MARGINS = [float(x) for x in args.margins.split(",")]
    rows = []
    for n, t in enumerate(use):
        i = tok_at[t]
        traj = torch.as_tensor(anchors + off[i], dtype=torch.float32)[None]      # [1, K, T, 3]
        dac = scores[lab_at[t], :256, ci]
        nc = scores[lab_at[t], :256, ni]
        feasible = (dac > 0.5) & (nc > 0.5)      # 선택 품질용 (배포와 같은 기준)
        feas_dac = dac > 0.5                     # 게이트 오류율용: 정적 장은 DAC만 본다
        if feasible.sum() == 0:
            continue
        try:
            scene = loader.get_scene_from_token(t)
            poly = builder.drivable_polygon(scene, scene.scene_metadata.num_history_frames - 1)
            if poly is None or poly.is_empty:
                continue
            gt = torch.as_tensor(rasterize_bev_sdf(poly, pc, Ht, Wt, 8.0))[None]
        except Exception:
            continue
        pr = torch.as_tensor(fld[i])[None]

        mins = {name: corner_sdf(f, traj, pc).amin(dim=(2, 3))[0].numpy()
                for name, f in (("pred", pr), ("gt", gt))}
        r = {"n_feas": int(feasible.sum()), "deployed": float(dac[fin[i].argmax()])}
        for m in MARGINS:
            for name in ("pred", "gt"):
                a = mins[name] > m
                pick = (fin[i] - 1e4 * (~a)).argmax() if a.any() else fin[i].argmax()
                r[f"{name}@{m}"] = float(dac[pick])
                r[f"{name}@{m}_any"] = float(a.any())
                if name == "pred":
                    r[f"fr@{m}"] = float((~a & feas_dac).sum() / max(feas_dac.sum(), 1))
                    r[f"fa@{m}"] = float((a & ~feas_dac).sum() / max((~feas_dac).sum(), 1))
                    r[f"drop@{m}"] = float((~a).mean())
        rows.append(r)
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(use)}", flush=True)

    d = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    print(f"\n{len(rows)} 장면.  배포(학습된 스칼라 게이트) 기준 실제 DAC 라벨 비율 "
          f"{100*d['deployed'].mean():.2f}%\n")
    print(f"{'margin':>7s}{'예측 장':>10s}{'GT 장':>10s}{'탈락 후보':>11s}"
          f"{'오탈락':>9s}{'오통과':>9s}{'전원탈락':>10s}")
    print("-" * 66)
    base = d["deployed"].mean()
    for m in MARGINS:
        print(f"{m:7.2f}{100*d[f'pred@{m}'].mean():9.2f}%{100*d[f'gt@{m}'].mean():9.2f}%"
              f"{100*d[f'drop@{m}'].mean():10.1f}%{100*d[f'fr@{m}'].mean():8.1f}%"
              f"{100*d[f'fa@{m}'].mean():8.1f}%{100*(1-d[f'pred@{m}_any'].mean()):9.2f}%")
    print(f"\n  오탈락/오통과는 정적 장이 판정하는 DAC 라벨 단독 기준 (동적 장 미포함)")
    best = max(MARGINS, key=lambda m: d[f"pred@{m}"].mean())
    print(f"  최선 margin {best:.2f} -> 예측 장 {100*d[f'pred@{best}'].mean():.2f}% "
          f"({100*(d[f'pred@{best}'].mean()-base):+.2f}%p),  GT 장 천장 "
          f"{100*max(d[f'gt@{m}'].mean() for m in MARGINS):.2f}%")


if __name__ == "__main__":
    main()
