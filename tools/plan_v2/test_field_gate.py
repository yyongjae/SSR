"""Checks for the inference-time field gate (feasibility.field_gate_select).

The gate reads the predicted corridor field at selection instead of the sigmoid head that copied it.
It changes what the model EMITS, on finished checkpoints, so every claim about it has to survive a
control -- a flipped field, an all-admitted set, an all-rejected set -- or the check passes for the
wrong reason.  Two earlier bugs of mine in this file's subject matter were a rotation applied as
R(dtheta)^T and a fill that ignored interior rings, both invisible to a check without a control, so
the axis convention here is tested by an obstacle placed on ONE side of the ego.

    python tools/plan_v2/test_field_gate.py
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/external-user/kyungmin/SSR-v2")

from navsim.agents.para_ssr.feasibility import (          # noqa: E402
    corner_sdf, corridor_clearance, field_gate_select,
)

# SSR BEV: rows along SSR y (navsim x, forward), columns along SSR x (= -navsim y)
PC = [-25.6, -25.6, -5.0, 25.6, 25.6, 3.0]
H, W = 100, 100
T, K = 8, 4


class Cfg:
    map_pc_range = PC
    feas_dyn_weight = 0.0
    feas_select_margin = 0.0
    feas_select_dyn_margin = 0.0
    plan_select_gate = 20.0
    feas_select_gate = True


def straight(dx: float, dy: float) -> torch.Tensor:
    """A [T, 3] straight path advancing (dx, dy) per step in NAVSIM axes."""
    t = torch.arange(1, T + 1, dtype=torch.float32)[:, None]
    return torch.cat([t * dx, t * dy, torch.zeros(T, 1)], dim=1)


def preds(field, trajs, scores, dyn=None):
    """Build the minimal prediction dict the gate consumes.  Anchors carry the trajectory and the
    offset is zero, so `anchors + offset` is exactly `trajs` and the test controls the geometry."""
    anchors = torch.stack(trajs)                                        # [K, T, 3]
    d = {
        "trajectory_anchors": anchors,
        "trajectory_offset": torch.zeros(1, len(trajs), T, 3),
        "drivable_sdf_pred": field[None],
        "plan_final_rewards": torch.tensor(scores, dtype=torch.float32)[None],
        "plan_topk_index": torch.zeros(1, len(trajs), dtype=torch.long),
    }
    if dyn is not None:
        d["agent_sdf_pred"] = dyn[None]
    return d


def main():
    fails = []

    def ck(name, ok, detail=""):
        fails.append(name) if not ok else None
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('   ' + detail) if detail else ''}")

    # --- geometry -----------------------------------------------------------------------------
    # A wall on the ego's LEFT only: navsim +y.  SSR x = -navsim y, so left is SSR x < 0 = low
    # COLUMNS.  Free (+8) to the right, obstacle (-8) to the left.  Mirroring this is the control.
    xs = torch.linspace(PC[0], PC[3], W)                                # SSR x per column
    left = (xs < -3.0).float()[None, :].expand(H, W)                    # 1 where the wall is
    field = 8.0 - 16.0 * left
    left_turn, right_turn = straight(3.0, 1.5), straight(3.0, -1.5)
    p = preds(field, [left_turn, right_turn], [1.0, 0.0])
    cl = corridor_clearance(p, Cfg, 0.0, 0.0)[0]
    ck("C1 left wall rejects the LEFT candidate", bool(cl[0] < 0) and bool(cl[1] > 0),
       f"left {cl[0]:+.2f}  right {cl[1]:+.2f}")
    pm = preds(torch.flip(field, dims=[1]), [left_turn, right_turn], [1.0, 0.0])
    clm = corridor_clearance(pm, Cfg, 0.0, 0.0)[0]
    ck("C1' control: mirrored field flips the verdict", bool(clm[0] > 0) and bool(clm[1] < 0),
       f"left {clm[0]:+.2f}  right {clm[1]:+.2f}")

    # --- min, not mean ------------------------------------------------------------------------
    # One bad STEP out of T must decide the verdict: a candidate that clips a wall at step 4 only.
    free = torch.full((H, W), 8.0)
    ys = torch.linspace(PC[1], PC[4], H)                                # SSR y per row = navsim x
    traj = straight(3.0, 0.0)
    bad_x = float(traj[3, 0])                                           # navsim x at step 4
    band = ((ys - bad_x).abs() < 1.0).float()[:, None].expand(H, W)
    spot = 8.0 - 16.0 * band
    p = preds(spot, [traj], [1.0])
    cl = corridor_clearance(p, Cfg, 0.0, 0.0)[0, 0]
    per = corner_sdf(spot[None], traj[None, None], PC)[0, 0]             # [T, 4]
    ck("C2 clearance is the min over steps and corners",
       abs(float(cl) - float(per.min())) < 1e-4 and float(cl) < 0,
       f"clearance {cl:+.2f}  min {per.min():+.2f}  mean {per.mean():+.2f}")

    # --- margin -------------------------------------------------------------------------------
    p = preds(free, [traj], [1.0])
    c0 = float(corridor_clearance(p, Cfg, 0.0, 0.0)[0, 0])
    c4 = float(corridor_clearance(p, Cfg, 0.4, 0.0)[0, 0])
    ck("C3 margin shifts static clearance by exactly the margin", abs((c0 - c4) - 0.4) < 1e-4,
       f"{c0:+.3f} -> {c4:+.3f}")

    # --- dynamic field takes the worse -------------------------------------------------------
    class CfgD(Cfg):
        feas_dyn_weight = 1.0
    dyn = torch.full((T, H, W), 8.0)
    dyn[3] = -2.0                                                        # agents block step 4 only
    p = preds(free, [traj], [1.0], dyn=dyn)
    cl = float(corridor_clearance(p, CfgD, 0.0, 0.2)[0, 0])
    ck("C4 merged corridor takes the WORSE of static and dynamic", abs(cl - (-2.2)) < 1e-3,
       f"static +8.0, dyn -2.0 - 0.2 -> {cl:+.3f}")
    p = preds(free, [traj], [1.0], dyn=torch.full((T, H, W), 8.0))
    ck("C4' control: a clear dynamic field leaves the static verdict",
       float(corridor_clearance(p, CfgD, 0.0, 0.2)[0, 0]) > 0)

    # --- the gate -----------------------------------------------------------------------------
    # Four candidates.  The top-scoring one drives into the left wall; the runner-up is clear.
    trajs = [left_turn, right_turn, straight(3.0, -0.5), straight(2.0, -2.0)]
    scores = [5.0, 4.0, 3.0, 2.0]
    p = preds(field, trajs, scores)
    out = field_gate_select(p, Cfg)
    ck("C5 gate drops the inadmissible argmax and picks the best admissible",
       int(out["plan_topk_index"][0, 0]) == 1,
       f"ungated argmax 0 -> gated {int(out['plan_topk_index'][0, 0])}")
    exp = (p["trajectory_anchors"][1]).numpy()
    ck("C5a emitted trajectory is the picked anchor + offset",
       np.allclose(out["trajectory"][0].numpy(), exp, atol=1e-5))
    ck("C5b admissible candidates keep their score, rejected drop by the gate",
       abs(float(out["plan_final_rewards"][0, 1]) - 4.0) < 1e-5
       and abs(float(out["plan_final_rewards"][0, 0]) - (5.0 - 20.0)) < 1e-5)

    # control: every candidate clear -> selection must be untouched
    p = preds(free, trajs, scores)
    out = field_gate_select(p, Cfg)
    ck("C6 control: all admissible leaves the argmax alone",
       int(out["plan_topk_index"][0, 0]) == 0
       and torch.allclose(out["plan_final_rewards"], p["plan_final_rewards"]))

    # control: every candidate blocked -> fall back to the ungated order, never an arbitrary pick
    p = preds(torch.full((H, W), -8.0), trajs, scores)
    out = field_gate_select(p, Cfg)
    ck("C7 control: no admissible candidate falls back to the ungated argmax",
       int(out["plan_topk_index"][0, 0]) == 0
       and torch.allclose(out["plan_final_rewards"], p["plan_final_rewards"]))

    # control: no field -> the gate is a no-op rather than a crash (wote runs without the head)
    p = preds(free, trajs, scores)
    del p["drivable_sdf_pred"]
    ck("C8 control: a checkpoint with no field head is untouched", field_gate_select(p, Cfg) == {})

    # --- agreement with the training-time judgement -------------------------------------------
    # c_merge0 > 0 exactly when some corner is inside: the loss's magnitude and the gate's verdict
    # must agree in SIGN on the same trajectories, or they are not the same object.
    from navsim.agents.para_ssr.feasibility import cost_from_sdf
    rng = np.random.default_rng(0)
    tj = torch.stack([straight(float(v), float(l)) for v, l in
                      rng.uniform([-1, -3], [4, 3], size=(64, 2))])
    sdf = corner_sdf(field[None], tj[None], PC)
    c0 = cost_from_sdf(sdf, 0.0, 0.05)[0]
    cl = corridor_clearance(preds(field, list(tj), [0.0] * 64), Cfg, 0.0, 0.0)[0]
    agree = ((c0 > 0.05) == (cl < 0)).float().mean()
    ck("C9 gate verdict agrees in sign with the training cost", float(agree) > 0.95,
       f"agreement {float(agree):.4f} over 64 candidates")

    print(f"\n{'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
