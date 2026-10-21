"""Does the geometric TTC label (feasibility.ttc_samples / ttc_label) do exactly what it claims?

    R1  the PROJECTION is real: a plan that brakes to a stop just short of an agent never overlaps
        it, yet at its earlier speed it was heading into it -- the scorer's TTC fails that plan, and
        so must this label.  Control: the same plan's current-footprint samples alone all pass.
    R2  a STOPPED ego is exempt, as in the scorer.  Control: the same overlap while creeping at
        0.2 m/s fails.
    R3  mode=hybrid replaces the TTC column exactly on the refined set S, leaves the rest of that
        column and every other column bit-identical.  Control: mode=anchor is the identity.
    R4  it is a label: no gradient flows from it into the model.

    python tools/plan_v2/test_ttc_label.py
"""
import sys

import torch

sys.path.insert(0, "/home/external-user/kyungmin/SSR-v2")

from navsim.agents.para_ssr.configs.default import ParaSSRConfig                 # noqa: E402
from navsim.agents.para_ssr.feasibility import (TTC_COL, select_weights,         # noqa: E402
                                                ttc_label, ttc_samples)

PC = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)          # navsim x forward in [0, 32], y lateral in [-32, 32]
H, W, T = 100, 200, 8
OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'통과' if cond else '실패 <<<'}  {name}  {detail}")


def agent_field(x0, x1, half_w=3.0):
    """[1, T, H, W]: an agent occupying navsim x in [x0, x1], |y| < half_w, in every plane.
    Inside the lateral band the value is the signed distance along x to the agent's near face."""
    xs = torch.linspace(PC[1], PC[4], H)[:, None]                  # rows = navsim x
    ys = -torch.linspace(PC[0], PC[3], W)[None, :]                 # cols = SSR x = -navsim y
    band = ys.abs() < half_w
    along = torch.where(xs < x0, x0 - xs, torch.where(xs > x1, xs - x1, -torch.minimum(xs - x0, x1 - xs)))
    f = torch.where(band, along.clamp(-3, 8), torch.full_like(along, 8.0))
    return f.expand(T, H, W)[None].clone()


def plan(xs):
    """[1, 1, T, 3]: straight along x through the given rear-axle positions."""
    p = torch.zeros(1, 1, T, 3)
    p[0, 0, :, 0] = torch.tensor(xs, dtype=torch.float32)
    return p


def main():
    print("R1  등속 투영: 멈추기 직전까지 앞차를 향해 달리던 궤적은 실패")
    dyn = agent_field(22.5, 27.0)
    brake = plan([5, 10, 15, 16.5, 17, 17, 17, 17])       # 10 m/s, then brakes; front bumper ends at 21.05
    s = ttc_samples(brake, dyn, PC)                        # [1, 1, T-1, 2, 4]
    now_min, ahead_min = float(s[..., 0, :].min()), float(s[..., 1, :].min())
    check("대조군: 현재 footprint 는 한 번도 안 겹침", now_min > 0, f"min Φ_now {now_min:+.2f}")
    check("투영 footprint 는 겹침 → 실패", ahead_min < 0, f"min Φ_ahead {ahead_min:+.2f}")
    cruise_far = plan([2, 4, 6, 8, 10, 12, 14, 15])        # 4 m/s, front ends at 19.05, projected <= 21.05
    s2 = ttc_samples(cruise_far, dyn, PC)
    check("대조군: 천천히 가서 투영해도 안 닿으면 통과", float(s2.min()) > 0, f"min Φ {float(s2.min()):+.2f}")

    print("R2  정지한 ego 는 면제 (채점기와 동일)")
    on_top = agent_field(-3.0, 6.0)                        # the agent sits on the ego's own footprint
    parked = plan([0] * T)
    creep = plan([0.1 * (i + 1) for i in range(T)])       # 0.2 m/s
    sp, sc = ttc_samples(parked, on_top, PC), ttc_samples(creep, on_top, PC)
    check("정지: 전부 면제 (큰 여유)", float(sp.min()) > 100, f"min {float(sp.min()):.0f}")
    check("대조군: 0.2 m/s 로 움직이면 실패", float(sc.min()) < 0, f"min {float(sc.min()):+.2f}")

    print("R3  hybrid 는 S 에서만 TTC 열을 교체")
    torch.manual_seed(0)
    B, K = 3, 64
    t = torch.linspace(0.2, 1.0, T)
    ex, ey = torch.rand(K) * 20 + 4, torch.rand(K) * 16 - 8
    anchors = torch.stack([ex[:, None] * t, ey[:, None] * t, torch.zeros(K, T)], dim=-1)
    off = (torch.randn(B, K, T, 3) * 0.5).requires_grad_(True)
    preds = {"trajectory_anchors": anchors, "trajectory_offset": off,
             "plan_rank_rewards": torch.randn(B, K), "plan_final_rewards": torch.randn(B, K)}
    tgt = {"agent_sdf_bev": agent_field(10.0, 14.0).expand(B, T, H, W).clone(),
           "trajectory": anchors[torch.randint(0, K, (B,))],
           "sim_reward": torch.rand(B, 5, K), "sim_reward_valid": torch.ones(B)}
    cfg = ParaSSRConfig(feas_ttc_target="hybrid", feas_select_label_k=8, feas_select_model_k=8)
    new, logs = ttc_label(preds, tgt, cfg, tgt["sim_reward"])
    S = select_weights(preds["plan_rank_rewards"], preds, tgt, cfg) > 0
    old = tgt["sim_reward"]
    changed = new[:, TTC_COL] != old[:, TTC_COL]
    check("바뀐 칸은 전부 S 안", bool((changed & ~S).sum() == 0), f"S 밖 변경 {int((changed & ~S).sum())}")
    check("S 밖 TTC 는 원래 값 그대로", torch.equal(new[:, TTC_COL][~S], old[:, TTC_COL][~S]))
    others = [i for i in range(5) if i != TTC_COL]
    check("다른 열은 전부 그대로", torch.equal(new[:, others], old[:, others]))
    check("S 안에서 실제로 바뀜", bool(changed[S].float().mean() > 0.5),
          f"S 중 변경 {100 * float(changed[S].float().mean()):.0f}%   frac_replaced {float(logs['ttc/frac_replaced']):.3f}")
    ident, lg = ttc_label(preds, tgt, ParaSSRConfig(feas_ttc_target="anchor"), tgt["sim_reward"])
    check("대조군: anchor 모드는 항등", ident is tgt["sim_reward"] and lg == {})

    print("R4  라벨이라 기울기가 없음")
    check("반환 텐서가 기울기를 요구하지 않음", not new.requires_grad)

    print("R5  band 는 평균 |offset| 이 [2, 12) m 인 후보에서만 교체")
    mags = torch.tensor([0.0, 1.0, 1.99, 2.0, 5.0, 11.9, 12.0, 20.0])          # 후보별 평균 |offset|
    Kb = len(mags)
    ang = torch.rand(1, Kb, T) * 6.283
    offb = torch.stack([mags[None, :, None] * torch.cos(ang), mags[None, :, None] * torch.sin(ang),
                        torch.zeros(1, Kb, T)], dim=-1)                          # 모든 스텝 |offset| = mags
    pb = {"trajectory_anchors": anchors[:Kb], "trajectory_offset": offb,
          "plan_rank_rewards": torch.randn(1, Kb), "plan_final_rewards": torch.randn(1, Kb)}
    tb = {"agent_sdf_bev": agent_field(10.0, 14.0), "trajectory": anchors[:1],
          "sim_reward": torch.full((1, 5, Kb), 0.123), "sim_reward_valid": torch.ones(1)}
    newb, lb = ttc_label(pb, tb, ParaSSRConfig(feas_ttc_target="band"), tb["sim_reward"])
    want = (mags >= 2.0) & (mags < 12.0)
    got = newb[0, TTC_COL] != 0.123
    check("교체된 후보 = [2, 12) 구간", torch.equal(got, want), f"교체 {got.int().tolist()}")
    check("다른 열은 그대로", torch.equal(newb[:, [i for i in range(5) if i != TTC_COL]],
                                         tb["sim_reward"][:, [i for i in range(5) if i != TTC_COL]]))
    # the logs must describe the labels BEFORE replacement: the anchor label here is 0.123 everywhere,
    # so where a candidate was replaced the logged anchor mean must be 0.123, not the geometric value
    check("로그: 교체 전 앵커 라벨을 기록", abs(float(lb["ttc/target_anchor_replaced"]) - 0.123) < 1e-5,
          f"target_anchor_replaced {float(lb['ttc/target_anchor_replaced']):.4f}  target_geo {float(lb['ttc/target_geo']):.4f}")
    pz = dict(pb, trajectory_offset=torch.zeros_like(offb))
    newz, _ = ttc_label(pz, tb, ParaSSRConfig(feas_ttc_target="band"), tb["sim_reward"])
    check("대조군: offset 0 이면 아무것도 안 바뀜", torch.equal(newz, tb["sim_reward"]))

    print(f"\n{sum(OK)}/{len(OK)} 통과")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
