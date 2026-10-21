"""Does a more accurate field produce a better score?  The controlled test.

Everything the corridor line proposes next -- a teacher, a segmentation map head, a finer target --
is an investment in FIELD ACCURACY, and nothing so far shows that accuracy buys score.  Arm-to-arm
comparison cannot show it: `feas_gt` and `feas_pred` finish with the same field error (0.669 vs
0.668 val L1) and different gate gains, so whatever separates them is not the field.

The inference gate makes the clean experiment possible, because it is a selection rule over a
finished checkpoint.  The field can be swapped while the model, the candidates, the offsets and the
ranking stay bit-identical:

    Phi_alpha = alpha * GT + (1 - alpha) * predicted,    alpha in [0, 1]

alpha is the ONLY thing that varies.  alpha=0 is the deployed gate, alpha=1 gates on the ground
truth, and the field error in between falls smoothly.  A monotone rise in score is the evidence;
a flat curve refutes the whole direction and says to stop paying for the field.

The blend is on the head's grid, and the GT is the target builder's own field averaged down exactly
as the grounding loss does (tools/plan_v2/dump_gt_field.py), so alpha=0 and alpha=1 differ in
nothing but accuracy.

    python tools/plan_v2/field_quality_sweep.py --exp para_ssr_v2_r34_feas_gt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))

from navsim.agents.para_ssr.feasibility import field_gate_select      # noqa: E402


class Cfg:
    map_pc_range = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)
    feas_dyn_weight = 0.0            # the static field alone: measured best, and the axis under test
    plan_select_gate = 20.0
    feas_select_margin = 0.1
    feas_select_dyn_margin = -1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="para_ssr_v2_r34_feas_gt")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    Cfg.feas_select_margin = a.margin

    z = np.load(ROOT / f"work_dirs/eval_epdms/field_{a.exp}.npz", allow_pickle=True)
    g = np.load(ROOT / "work_dirs/eval_epdms/gt_field_navtest.npz")
    gt_at = {str(t): i for i, t in enumerate(g["tokens"])}
    toks = [str(t) for t in z["tokens"]]
    idx = np.array([gt_at[t] for t in toks if t in gt_at])
    rows_keep = np.array([i for i, t in enumerate(toks) if t in gt_at])
    print(f"{len(rows_keep)} / {len(toks)} 장면에 GT 장 있음")

    anchors = torch.from_numpy(np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy")).float()
    off_all = z["trajectory_offset"][rows_keep]
    rew_all = z["plan_final_rewards"][rows_keep]
    pred_all = z["drivable_sdf_pred"][rows_keep].astype(np.float32)
    gt_all = g["drivable_sdf_gt"][idx].astype(np.float32)
    names = np.array([toks[i] for i in rows_keep])
    n = len(rows_keep)

    # the error the blend is actually moving, reported the way training reports it
    near = np.abs(gt_all) < 1.0
    mae0 = np.abs(pred_all - gt_all)[near].mean()
    print(f"예측 장의 경계 근처 오차 (|GT|<1m): {mae0:.3f} m\n")

    out_dir = ROOT / "work_dirs/eval_epdms/field_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    print(f"{'alpha':>6s}{'장 오차':>10s}{'선택 변경':>10s}{'전원 탈락':>10s}")
    for tag in a.alphas.split(","):
        al = float(tag)
        blend = (al * gt_all + (1 - al) * pred_all).astype(np.float32)
        err = np.abs(blend - gt_all)[near].mean()
        trajs, picks, changed, none_ok = [], [], 0, 0
        for s in range(0, n, a.batch):
            e = min(s + a.batch, n)
            p = {
                "trajectory_anchors": anchors,
                "trajectory_offset": torch.from_numpy(off_all[s:e]).float(),
                "drivable_sdf_pred": torch.from_numpy(blend[s:e]),
                "plan_final_rewards": torch.from_numpy(rew_all[s:e]).float(),
                "plan_topk_index": torch.zeros(e - s, 1, dtype=torch.long),
            }
            base = p["plan_final_rewards"].argmax(1)
            o = field_gate_select(p, Cfg)
            pick = o["plan_topk_index"][:, 0]
            cl = o["feas_select_clearance"]
            trajs.append(o["trajectory"].numpy())
            picks.append(pick.numpy())
            changed += int((pick != base).sum())
            none_ok += int((cl >= 0).any(dim=-1).logical_not().sum())
        name = f"fq_{a.exp.replace('para_ssr_v2_r34_', '')}_a{tag}"
        f = out_dir / f"{name}.npz"
        np.savez_compressed(f, tokens=names, trajectory=np.concatenate(trajs).astype(np.float32),
                            ego_fut_preds=np.zeros((n, 4, 8, 3), np.float32),
                            command=np.zeros((n, 4), np.float32),
                            missing=np.zeros((0,), np.float64),
                            checkpoint=np.array(f"field blend alpha={tag}"),
                            picked_index=np.concatenate(picks))
        print(f"{tag:>6s}{err:10.3f}{changed:10d}{none_ok:10d}   -> {f.name}")
        made.append((name, f))

    print("\n채점:")
    for name, f in made:
        print(f"  bash data/logs/score_npz.sh {name} {f}")


if __name__ == "__main__":
    main()
