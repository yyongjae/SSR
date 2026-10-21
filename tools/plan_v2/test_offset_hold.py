"""Does the offset hold (anchor_plan_losses offset_hold_radius) restrain only the non-winner offsets?

    H1  off (radius None) -> no offset_hold_loss key, the three WoTE terms unchanged
    H2  offsets inside the radius -> loss exactly 0
    H3  the winner (anchor closest to the GT) gets zero gradient however far its offset goes
    H4  a non-winner beyond the radius: loss = mean over (K-1) x T of (|xy| - r)^2, descent shrinks it
    W1  wiring: ParaSSRLoss adds plan_offset_hold_weight * offset_hold_loss to the plan loss

    python tools/plan_v2/test_offset_hold.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
from navsim.agents.para_ssr.modules.anchor_planner import anchor_plan_losses                # noqa: E402

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'통과' if cond else '실패 <<<'}  {name}  {detail}")


T, K = 8, 4


def setup(off_xy):
    """off_xy [K] lateral offset (m) per candidate, constant over steps; GT = anchor 1 exactly."""
    anchors = torch.zeros(K, T, 3)
    for k in range(K):
        anchors[k, :, 0] = torch.linspace(1, 8, T) * (k + 1)
    off = torch.zeros(1, K, T, 3)
    off[0, :, :, 1] = torch.tensor(off_xy)[:, None]
    off.requires_grad_(True)
    p = {"trajectory_anchors": anchors, "trajectory_offset": off,
         "im_rewards": torch.full((1, K), 1.0 / K), "sim_rewards": torch.full((1, 5, K), 0.5)}
    return p, anchors[1][None], off


def main():
    lab, valid = torch.full((1, 5, K), 1.0), torch.ones(1)
    print("H1  꺼짐")
    p, gt, _ = setup([0.0, 0.0, 3.0, 0.0])
    base = anchor_plan_losses(p, gt, lab, valid)
    on = anchor_plan_losses(p, gt, lab, valid, offset_hold_radius=1.0)
    check("키 없음, WoTE 세 항 동일", "offset_hold_loss" not in base
          and all(torch.equal(base[k], on[k]) for k in base))
    print("H2  반경 안은 0")
    p, gt, _ = setup([0.5, 0.9, -0.9, 0.0])
    check("손실 0", float(anchor_plan_losses(p, gt, lab, valid, offset_hold_radius=1.0)["offset_hold_loss"]) == 0.0)
    print("H3  winner 는 기울기 0")
    p, gt, off = setup([0.0, 5.0, 0.0, 0.0])                   # winner = 1, far outside
    l = anchor_plan_losses(p, gt, lab, valid, offset_hold_radius=1.0)["offset_hold_loss"]
    check("winner 만 멀리 → 손실 0", float(l) == 0.0)
    print("H4  non-winner 초과분")
    p, gt, off = setup([3.0, 0.0, 0.0, 0.0])                   # candidate 0 at 3 m, r = 1 -> excess 2
    l = anchor_plan_losses(p, gt, lab, valid, offset_hold_radius=1.0)["offset_hold_loss"]
    want = (2.0 ** 2) * T / ((K - 1) * T)
    check("값 = 4 / 3", abs(float(l) - want) < 1e-5, f"{float(l):.4f} vs {want:.4f}")
    g, = torch.autograd.grad(l, off)
    check("하강이 후보 0 의 오프셋을 줄임, 다른 후보 기울기 0",
          float(g[0, 0, :, 1].min()) > 0 and float(g[0, 1:].abs().sum()) == 0.0)
    print("W1  배선")
    src = (ROOT / "navsim/agents/para_ssr/para_ssr_loss.py").read_text()
    check("plan += hold_w * offset_hold_loss", 'hold_w * al.get("offset_hold_loss", 0.0)' in src
          and "offset_hold_radius=float(cfg.plan_offset_hold_radius) if hold_w > 0 else None" in src)
    print(f"\n{sum(OK)}/{len(OK)} 통과")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
