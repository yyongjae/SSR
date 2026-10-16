"""Is refining the PICK worth anything?  Test-time descent on refine.refine_objective, no training.

Takes a dumped arm (all 256 candidates + its predicted fields), picks as deployment does with the
field gate (feasibility.field_gate_select, static margin 0.1), then minimises the stage-2 objective
on the picked trajectory alone -- corridor on the model's OWN predicted field (static + per-step
agents), smoothness, trust region -- and applies the same absolute-corridor fallback as inference.
No human, no route: the progress term needs the human and is left out rather than given an
oracle target, so this measures what the corridor alone buys on the emitted trajectory.

    python tools/plan_v2/refine_upper_bound.py --exp para_ssr_v2_r34_feas_pred_seg_ch_select_ttcband_hold_epgeo_sees_drive
    bash data/logs/score_npz.sh <name> <npz>          # PDMS and EPDMS, both
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
from navsim.agents.para_ssr.configs.default import ParaSSRConfig                         # noqa: E402
from navsim.agents.para_ssr.feasibility import field_gate_select                          # noqa: E402
from navsim.agents.para_ssr.refine import merged_clearance, path_poses, refine_objective  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-dyn", action="store_true", help="refine against the static field only")
    ap.add_argument("--ttc-weight", type=float, default=0.0)
    ap.add_argument("--hinge", action="store_true", help="refine_corridor_hinge: no force inside the margin")
    a = ap.parse_args()
    z = np.load(ROOT / f"work_dirs/eval_epdms/field_{a.exp}.npz")
    anchors = torch.from_numpy(np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy")).float()
    cfg = ParaSSRConfig(feas_select_margin=0.1, feas_select_dyn_margin=-1e9, feas_dyn_weight=1.0,
                        refine_progress_weight=0.0, refine_lateral_weight=0.0, refine_continuity_weight=0.0,
                        refine_ttc_weight=a.ttc_weight, refine_corridor_hinge=a.hinge)
    n = len(z["tokens"])
    has_dyn = "agent_sdf_pred" in z.files and not a.no_dyn
    outs, picks, used, before_in, after_in = [], [], 0, 0, 0
    for s in range(0, n, a.batch):
        e = min(s + a.batch, n)
        static = torch.from_numpy(z["drivable_sdf_pred"][s:e]).float()
        dyn = torch.from_numpy(z["agent_sdf_pred"][s:e]).float() if has_dyn else None
        p = {"trajectory_anchors": anchors, "trajectory_offset": torch.from_numpy(z["trajectory_offset"][s:e]).float(),
             "drivable_sdf_pred": static, "plan_final_rewards": torch.from_numpy(z["plan_final_rewards"][s:e]).float(),
             "plan_topk_index": torch.zeros(e - s, 1, dtype=torch.long)}
        o = field_gate_select(p, cfg)
        base = o["trajectory"][:, None].detach()                                          # [B, 1, T, 3]
        xy = base[..., :2].clone().requires_grad_(True)                                   # heading from the path
        opt = torch.optim.Adam([xy], lr=a.lr)
        for _ in range(a.steps):
            opt.zero_grad()
            c, _ = refine_objective(path_poses(xy, base), base, static, dyn, cfg)
            c.sum().backward()
            opt.step()
        with torch.no_grad():
            tr = path_poses(xy, base)
            # inference's fallback: the gate's own margin, static only (eval runs dyn margin off)
            c_new = merged_clearance(tr, static, None, cfg.map_pc_range, 0.1, 0.0).amin(dim=(2, 3))[:, 0]
            c_old = merged_clearance(base, static, None, cfg.map_pc_range, 0.1, 0.0).amin(dim=(2, 3))[:, 0]
            bad = (c_new < 0) & (c_new < c_old)
            final = torch.where(bad[:, None, None], base[:, 0], tr[:, 0])
            c_fin = merged_clearance(final[:, None], static, None, cfg.map_pc_range, 0.1, 0.0).amin(dim=(2, 3))[:, 0]
        used += int(bad.sum()); before_in += int((c_old >= 0).sum()); after_in += int((c_fin >= 0).sum())
        outs.append(final.numpy()); picks.append(o["plan_topk_index"][:, 0].numpy())
        if s % (a.batch * 20) == 0:
            print(f"{e}/{n}", flush=True)
    traj = np.concatenate(outs).astype(np.float32)
    print(f"예측 장 corridor 안 (margin 0.1): 다듬기 전 {100 * before_in / n:.2f}%  후 {100 * after_in / n:.2f}%   "
          f"fallback {100 * used / n:.2f}%")
    d = ROOT / "work_dirs/eval_epdms/refine_ub"; d.mkdir(parents=True, exist_ok=True)
    name = f"refineub_{a.exp.replace('para_ssr_v2_r34_', '')[:40]}{a.tag}"
    f = d / f"{name}.npz"
    np.savez_compressed(f, tokens=z["tokens"], trajectory=traj, ego_fut_preds=np.zeros((n, 4, 8, 3), np.float32),
                        command=np.zeros((n, 4), np.float32), missing=np.zeros((0,), np.float64),
                        checkpoint=np.array(f"refine_ub {a.exp} steps={a.steps}"), picked_index=np.concatenate(picks))
    print(f"bash data/logs/score_npz.sh {name} {f}")


if __name__ == "__main__":
    main()
