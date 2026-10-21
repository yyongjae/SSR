"""Does head_mode="corridor" do exactly what it claims, and nothing else?

Four things have to hold, and each is checked against a control rather than asserted:

    T1  four heads, not five, and the three that keep WoTE labels are matched to the RIGHT label
        columns -- shuffling the label columns must make the BCE worse
    T2  the feasible head gets NO anchor label, only the feasibility target
    T3  the merged corridor judges with min(Phi_static - m_s, Phi_dyn - m_d), so a candidate that
        violates only the road, only an agent, or both, is infeasible in all three cases, and the
        merged cost is NOT the sum
    T4  the gate uses the feasible head: a candidate the head rejects ranks below every accepted
        one, and a scene with no accepted candidate keeps its ungated order

    python tools/plan_v2/test_corridor_heads.py
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/external-user/kyungmin/SSR-v2")

from navsim.agents.para_ssr.configs.default import ParaSSRConfig                 # noqa: E402
from navsim.agents.para_ssr.feasibility import (FEAS_INDEX, cost_from_sdf,       # noqa: E402
                                                feasibility_losses, target_shape)
from navsim.agents.para_ssr.modules.anchor_planner import (CORRIDOR_KEYS,        # noqa: E402
                                                           LABEL_COLS, SIM_KEYS,
                                                           anchor_plan_losses)

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'통과' if cond else '실패 <<<'}  {name}  {detail}")


def fake(cfg, heads, K=16, T=8, B=2, seed=0):
    torch.manual_seed(seed)
    H, W = cfg.bev_h, cfg.bev_w
    th, tw = target_shape(cfg)
    off = torch.zeros(B, K, T, 3, requires_grad=True)
    fld = (torch.rand(B, H, W) * 4 - 1).requires_grad_(True)
    dyn = (torch.rand(B, T, H, W) * 4 - 1).requires_grad_(True)
    sim = torch.rand(B, heads, K, requires_grad=True)
    preds = {"trajectory_offset": off,
             "trajectory_anchors": torch.randn(K, T, 3) * 2,
             "drivable_sdf_pred": fld, "agent_sdf_pred": dyn,
             "sim_rewards": sim.clamp(1e-3, 1 - 1e-3),
             "im_rewards": torch.softmax(torch.randn(B, K), -1),
             "plan_rank_rewards": torch.randn(B, K)}
    tgt = {"drivable_sdf_bev": torch.rand(B, th, tw) * 4 - 1,
           "agent_sdf_bev": torch.rand(B, T, th, tw) * 4 - 1,
           "trajectory": torch.randn(B, T, 3),
           "sim_reward": torch.rand(B, 5, K),
           "sim_reward_valid": torch.ones(B)}
    return preds, tgt, off, fld, dyn, sim


def main():
    print("T1/T2  라벨 정렬과 feasible 헤드의 무라벨성")
    cfg = ParaSSRConfig(feas_weight=0.15, feas_field="pred", feas_to_offset=True,
                        feas_to_score=True, feas_dyn_weight=1.0, plan_head_mode="corridor")
    preds, tgt, off, fld, dyn, sim = fake(cfg, len(CORRIDOR_KEYS))
    cols = [None] + [LABEL_COLS[k] for k in CORRIDOR_KEYS[1:]]
    check("헤드 수 4", len(CORRIDOR_KEYS) == 4, f"{CORRIDOR_KEYS}")
    check("label_cols = [None, 2, 3, 4]", cols == [None, 2, 3, 4], f"{cols}")

    al = anchor_plan_losses(preds, tgt["trajectory"], tgt["sim_reward"],
                            tgt["sim_reward_valid"], label_cols=cols)
    # control: point the three labelled heads at the WRONG columns
    bad = [None, LABEL_COLS["comfort"], LABEL_COLS["ego_progress"],
           LABEL_COLS["time_to_collision_within_bound"]]
    al_bad = anchor_plan_losses(preds, tgt["trajectory"], tgt["sim_reward"],
                                tgt["sim_reward_valid"], label_cols=bad)
    same = abs(float(al["sim_reward_loss"].detach()) - float(al_bad["sim_reward_loss"].detach()))
    check("컬럼을 섞으면 BCE가 달라짐 (정렬이 실제로 쓰임)", same > 1e-6,
          f"차이 {same:.6f}")

    # T2: the feasible head must receive no gradient from the anchor BCE
    g = torch.autograd.grad(al["sim_reward_loss"], sim, retain_graph=True)[0]
    per_head = g.abs().sum(dim=(0, 2))
    check("feasible 헤드에 앵커 BCE 기울기 없음", float(per_head[FEAS_INDEX]) == 0.0,
          f"heads={[round(float(x), 4) for x in per_head]}")
    check("나머지 3개 헤드는 기울기 받음", bool((per_head[1:] > 0).all()))

    print("\nT3  통합 장이 min인지, 합이 아닌지")
    # one candidate violating only the road, one only an agent, one both, one neither
    # [B=1, K=4, T=8, corners=4]; one candidate per violation pattern, constant over t and corners
    s_static = torch.tensor([-1.0, 3.0, -1.0, 3.0]).view(1, 4, 1, 1).expand(1, 4, 8, 4).clone()
    s_dyn = torch.tensor([3.0, -1.0, -1.0, 3.0]).view(1, 4, 1, 1).expand(1, 4, 8, 4).clone()
    ms, md = 0.4, 0.2
    phi = torch.minimum(s_static - ms, s_dyn - md)
    c_merge = cost_from_sdf(phi, 0.0, 0.2)[0]
    c_sum = (cost_from_sdf(s_static, ms, 0.2) + cost_from_sdf(s_dyn, md, 0.2))[0]
    names = ["도로만 위반", "에이전트만 위반", "둘 다 위반", "위반 없음"]
    for i, n in enumerate(names):
        print(f"    {n:14s} merge {float(c_merge[i]):8.3f}   sum {float(c_sum[i]):8.3f}")
    check("위반이 하나라도 있으면 merge > 0", bool((c_merge[:3] > 1.0).all()))
    check("위반이 없으면 merge ~ 0", float(c_merge[3]) < 1e-3, f"{float(c_merge[3]):.2e}")
    check("둘 다 위반 = 하나만 위반의 두 배가 아님 (min이지 합이 아님)",
          abs(float(c_merge[2]) - float(c_merge[0])) < 1e-3,
          f"둘다 {float(c_merge[2]):.3f} vs 도로만 {float(c_merge[0]):.3f}")
    check("합 쪽은 두 배가 됨 (대조)", float(c_sum[2]) > 1.8 * float(c_sum[0]),
          f"둘다 {float(c_sum[2]):.3f} vs 도로만 {float(c_sum[0]):.3f}")
    # the two margins must shift the SAME curve, by exactly their difference: the softplus is a
    # soft margin, so "clears the margin" is not "cost exactly zero", and an absolute threshold
    # would only be testing where that arbitrary threshold sits.
    def one(road, agent):
        return float(cost_from_sdf(torch.minimum(torch.full((1, 1, 1, 1), road) - ms,
                                                 torch.full((1, 1, 1, 1), agent) - md), 0.0, 0.2))
    shifts = [abs(one(d + (ms - md), 9.0) - one(9.0, d)) for d in (-0.4, -0.1, 0.0, 0.2, 0.5)]
    check("각자 마진 유지: 도로 곡선 = 에이전트 곡선을 0.2 m 평행이동",
          max(shifts) < 1e-5, f"최대 편차 {max(shifts):.2e}")
    check("같은 여유에서 도로가 더 비쌈 (마진이 0.2 m 크므로)",
          one(0.3, 9.0) > one(9.0, 0.3), f"{one(0.3, 9.0):.4f} vs {one(9.0, 0.3):.4f}")

    print("\nT4  게이트가 feasible 헤드를 쓰는지")
    from navsim.agents.para_ssr.modules.anchor_planner import AnchorPlanner
    pl = AnchorPlanner("/home/external-user/kyungmin/SSR-v2/data/planning_vb/trajectory_anchors_256.npy",
                       head_mode="corridor", select_mode="corridor_first", select_dac_thr=0.5,
                       select_gate=20.0)
    K = 256
    im = torch.softmax(torch.randn(1, K), -1)
    s = torch.rand(1, 4, K) * 0.4 + 0.3            # EP/TTC/comfort mid-range
    s[0, FEAS_INDEX] = torch.rand(K)
    final, ungated = pl.weighted_reward(im, s)
    adm = s[0, FEAS_INDEX] >= 0.5
    check("거부된 후보가 전부 수용된 후보 아래",
          float(final[0][adm].min()) > float(final[0][~adm].max()))
    check("ungated에는 게이트가 없음", not torch.equal(final, ungated))
    s[0, FEAS_INDEX] = 0.1                          # nobody admissible
    f2, u2 = pl.weighted_reward(im, s)
    check("수용 후보가 없으면 게이트 해제 (argmax가 임의가 되지 않도록)", torch.allclose(f2, u2))
    # a MIXED scene is the only one where the gate can show itself: with nobody admissible, or
    # everybody, the gate correctly collapses to the ungated order.
    s_mix = s.clone()
    s_mix[0, FEAS_INDEX] = torch.cat([torch.full((K // 2,), 0.9), torch.full((K - K // 2,), 0.1)])
    s_all = s.clone()
    s_all[0, FEAS_INDEX] = 0.9
    f_mix = pl.weighted_reward(im, s_mix)[0]
    f_all = pl.weighted_reward(im, s_all)[0]
    check("섞인 경우에만 게이트가 순서를 바꿈", not torch.allclose(f_mix, f_all))
    check("게이트가 바꾼 것은 거부된 절반뿐",
          torch.allclose(f_mix[0, : K // 2], f_all[0, : K // 2])
          and not torch.allclose(f_mix[0, K // 2:], f_all[0, K // 2:]))
    check("argmax가 수용된 절반에서 나옴", int(f_mix[0].argmax()) < K // 2)

    print("\nT5  전체 손실 경로 (forward -> loss -> backward)")
    preds, tgt, off, fld, dyn, sim = fake(cfg, 4, seed=1)
    loss, logs = feasibility_losses(preds, tgt, cfg, iteration=10_000)
    loss.backward()
    check("오프셋에 기울기", off.grad is not None and float(off.grad.norm()) > 0,
          f"{float(off.grad.norm()):.4f}")
    check("feasible 헤드에 기울기", float(sim.grad.abs().sum(dim=(0, 2))[FEAS_INDEX]) > 0)
    check("다른 헤드에는 feasibility 기울기 없음",
          float(sim.grad.abs().sum(dim=(0, 2))[1:].sum()) == 0.0)
    check("장에는 grounding 기울기만 (비용 기울기 차단)",
          fld.grad is not None and float(fld.grad.norm()) > 0)
    print("    " + "  ".join(f"{k}={float(v):.4f}" for k, v in sorted(logs.items())
                             if k in ("feas/score", "feas/score_target", "feas/score_pred",
                                      "feas/cost_used", "feas/cost_other")))

    print(f"\n=> {sum(OK)}/{len(OK)} 통과" + ("" if all(OK) else "   << 실패 있음"))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
