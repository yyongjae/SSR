"""Does feas_weighting="select" do exactly what it claims, and nothing else?

    S1  S is exactly (label top-k) U (model top-k) U (WTA winner), and the weights are uniform on S
    S2  control: an invalid label row (navtest, sim_reward_valid = 0) contributes nothing to S
    S3  control: label_k = 0 and model_k = 0 leave only the WTA winner
    S4  the property the whole change exists for: the corridor cost's gradient on the OFFSET is
        exactly zero for every candidate outside S, and non-zero inside it.  Control: the old
        softmax weighting puts gradient on (almost) every candidate
    S5  the score path is untouched: feasible-head targets are still computed for ALL candidates

    python tools/plan_v2/test_select_weights.py
"""
import sys

import torch

sys.path.insert(0, "/home/external-user/kyungmin/SSR-v2")

from navsim.agents.para_ssr.configs.default import ParaSSRConfig                 # noqa: E402
from navsim.agents.para_ssr.feasibility import (feasibility_losses,              # noqa: E402
                                                select_weights, target_shape)

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'통과' if cond else '실패 <<<'}  {name}  {detail}")


def fake(cfg, K=64, T=8, B=3, seed=0):
    torch.manual_seed(seed)
    H, W = cfg.bev_h, cfg.bev_w
    th, tw = target_shape(cfg)
    # anchors ahead of the ego and inside the BEV: a corner outside the grid is clamped to the
    # border and has zero spatial gradient, which would look like a zero weight
    t = torch.linspace(0.2, 1.0, T)
    ex, ey = torch.rand(K) * 20 + 4, torch.rand(K) * 16 - 8
    anchors = torch.stack([ex[:, None] * t, ey[:, None] * t, torch.zeros(K, T)], dim=-1)
    off = torch.zeros(B, K, T, 3, requires_grad=True)
    # NEGATIVE everywhere (every candidate costs something) and SLOPED (a constant field has zero
    # gradient in position -- the first version of this test used one and saw no gradient at all)
    def ramp(h, w):
        r = torch.arange(h).float()[:, None] / h
        c = torch.arange(w).float()[None, :] / w
        return (-1.0 - 2.0 * r - 2.0 * c).expand(B, h, w).clone()
    tgt_field = ramp(th, tw)
    preds = {"trajectory_offset": off, "trajectory_anchors": anchors,
             "drivable_sdf_pred": ramp(H, W),
             "sim_rewards": torch.rand(B, 4, K).clamp(1e-3, 1 - 1e-3),
             "im_rewards": torch.softmax(torch.randn(B, K), -1),
             "plan_rank_rewards": torch.randn(B, K)}
    gt = anchors[torch.randint(0, K, (B,))] + 0.1 * torch.randn(B, T, 3)
    tgt = {"drivable_sdf_bev": tgt_field, "trajectory": gt,
           "sim_reward": torch.rand(B, 5, K), "sim_reward_valid": torch.ones(B)}
    return preds, tgt, off


def expected_set(preds, tgt, lk, mk):
    logits = preds["plan_rank_rewards"]
    B, K = logits.shape
    nc, dac, ep, ttc, cf = tgt["sim_reward"].unbind(1)
    pdm = nc * dac * (5 * ep + 5 * ttc + 2 * cf) / 12
    flat = preds["trajectory_anchors"].reshape(1, K, -1)
    win = torch.norm(flat - tgt["trajectory"].reshape(B, 1, -1), dim=2).argmin(1)
    out = []
    for b in range(B):
        s = set(logits[b].topk(mk).indices.tolist()) if mk else set()
        if lk and tgt["sim_reward_valid"][b] > 0:
            s |= set(pdm[b].topk(lk).indices.tolist())
        s.add(int(win[b]))
        out.append(s)
    return out


def main():
    cfg = ParaSSRConfig(feas_weight=0.15, feas_field="gt", feas_to_offset=True, feas_to_score=True,
                        feas_weighting="select", feas_select_label_k=8, feas_select_model_k=8,
                        plan_head_mode="corridor")

    print("S1  S = 라벨 top-k ∪ 모델 top-k ∪ WTA 승자, S 위에서 균등")
    preds, tgt, _ = fake(cfg)
    w = select_weights(preds["plan_rank_rewards"], preds, tgt, cfg)
    exp = expected_set(preds, tgt, 8, 8)
    got = [set(torch.nonzero(w[b] > 0).flatten().tolist()) for b in range(w.shape[0])]
    check("선택 집합이 정의와 일치", got == exp, f"|S| = {[len(s) for s in got]}")
    uni = all(torch.allclose(w[b][w[b] > 0], torch.full((len(got[b]),), 1.0 / len(got[b])))
              for b in range(w.shape[0]))
    check("S 위에서 균등, 합 1", uni and torch.allclose(w.sum(-1), torch.ones(w.shape[0])))

    print("S2  대조군: 라벨이 무효인 행은 라벨 top-k 가 빠짐")
    tgt2 = dict(tgt, sim_reward_valid=torch.tensor([1.0, 0.0, 1.0]))
    w2 = select_weights(preds["plan_rank_rewards"], preds, tgt2, cfg)
    exp2 = expected_set(preds, tgt2, 8, 8)
    got2 = [set(torch.nonzero(w2[b] > 0).flatten().tolist()) for b in range(3)]
    check("무효 행 = 모델 top-k ∪ 승자", got2 == exp2 and len(got2[1]) <= 9, f"|S| = {[len(s) for s in got2]}")

    print("S3  대조군: k 둘 다 0 이면 WTA 승자 하나만")
    cfg0 = ParaSSRConfig(feas_weighting="select", feas_select_label_k=0, feas_select_model_k=0)
    w0 = select_weights(preds["plan_rank_rewards"], preds, tgt, cfg0)
    check("|S| = 1", bool(((w0 > 0).sum(-1) == 1).all()))

    print("S4  offset 기울기: S 밖은 정확히 0, S 안은 0 아님 (대조군: softmax)")
    preds, tgt, off = fake(cfg, seed=1)
    loss, logs = feasibility_losses(preds, tgt, cfg, iteration=10_000)
    loss.backward()
    g = off.grad.abs().sum(dim=(2, 3))                                  # [B, K]
    S = expected_set(preds, tgt, 8, 8)
    inside = all(bool((g[b, list(S[b])] > 0).all()) for b in range(g.shape[0]))
    outside = all(float(g[b, [k for k in range(g.shape[1]) if k not in S[b]]].abs().max()) == 0.0
                  for b in range(g.shape[0]))
    check("S 밖 기울기 = 0", outside)
    check("S 안 기울기 > 0", inside)
    print(f"        weight_support {float(logs['feas/weight_support']):.1f}   eff_n {float(logs['feas/weight_eff_n']):.1f}")

    cfgs = ParaSSRConfig(feas_weight=0.15, feas_field="gt", feas_to_offset=True, feas_to_score=True,
                         feas_weighting="softmax", plan_head_mode="corridor")
    preds, tgt, off = fake(cfgs, seed=1)
    loss, logs_s = feasibility_losses(preds, tgt, cfgs, iteration=10_000)
    loss.backward()
    gs = off.grad.abs().sum(dim=(2, 3))
    frac = float((gs > 0).float().mean())
    check("대조군: softmax 는 거의 전 후보에 기울기", frac > 0.95, f"기울기 받는 후보 비율 {frac:.2f}")

    print("S5  점수 경로는 그대로: feasible 타깃은 전 후보에서 계산")
    preds, tgt, _ = fake(cfg, seed=2)
    sim = preds["sim_rewards"].detach().requires_grad_(True)
    preds["sim_rewards"] = sim
    loss, logs = feasibility_losses(preds, tgt, cfg, iteration=10_000)
    loss.backward()
    frac = float((sim.grad[:, 0].abs() > 0).float().mean())
    check("feasible 헤드 기울기가 전 후보에", frac == 1.0, f"비율 {frac:.2f}")

    print("S6  S 밖 앵커 유지(hold): S 밖·반경 밖에만 기울기, 방향은 |offset| 을 줄이는 쪽")
    cfgh = ParaSSRConfig(feas_weight=0.15, feas_field="gt", feas_to_offset=False, feas_to_score=False,
                         feas_weighting="select", feas_select_label_k=8, feas_select_model_k=8,
                         feas_offset_anchor_weight=0.05, feas_offset_anchor_radius=2.0,
                         plan_head_mode="wote")
    preds, tgt, off = fake(cfgh, seed=3)
    B, K = off.shape[:2]
    with torch.no_grad():                         # half the candidates 5 m out, half 1 m out (inside r0)
        ang = torch.rand(B, K, 8) * 6.283
        r = torch.where(torch.arange(K) % 2 == 0, 5.0, 1.0)[None, :, None]
        off[..., 0] = r * torch.cos(ang); off[..., 1] = r * torch.sin(ang)
    loss, logs_h = feasibility_losses(preds, tgt, cfgh, iteration=10_000)
    loss.backward()
    g = off.grad[..., :2].abs().sum(dim=(2, 3))
    S = expected_set(preds, tgt, 8, 8)
    far = (torch.arange(K) % 2 == 0)
    ok_out_far = all(bool((g[b, [k for k in range(K) if k not in S[b] and far[k]]] > 0).all()) for b in range(B))
    ok_out_near = all(float(g[b, [k for k in range(K) if k not in S[b] and not far[k]]].max()) == 0.0 for b in range(B))
    ok_in = all(float(g[b, list(S[b])].max()) == 0.0 for b in range(B))
    check("S 밖·반경 밖(5 m) 후보에 기울기", ok_out_far)
    check("S 밖·반경 안(1 m) 후보는 기울기 0 (자유)", ok_out_near)
    check("S 안 후보는 hold 기울기 0", ok_in)
    # descent shrinks |offset| iff the gradient points ALONG the offset.  (A step test is useless
    # here: the gradient is ~1e-4 and float32 cannot resolve that step on a 5 m vector -- the first
    # version of this check failed for exactly that reason.)
    with torch.no_grad():
        dot = (off.grad[..., :2] * off[..., :2]).sum(-1)                          # [B, K, T]
        has = off.grad[..., :2].abs().sum(-1) > 0
    check("기울기가 offset 방향과 같음 → 하강이 |offset| 을 줄임", bool((dot[has] > 0).all()),
          f"기울기 있는 칸 {int(has.sum())}")
    check("로그: offset_mag_in / out / hold", all(k in logs_h for k in
          ("feas/offset_mag_in", "feas/offset_mag_out", "feas/offset_hold")),
          f"out {float(logs_h['feas/offset_mag_out']):.2f} m")
    cfg0h = ParaSSRConfig(feas_weight=0.15, feas_field="gt", feas_to_offset=False, feas_to_score=False,
                          feas_weighting="select", feas_offset_anchor_weight=0.0, plan_head_mode="wote")
    preds, tgt, off = fake(cfg0h, seed=3)
    with torch.no_grad():
        off[..., 0] = 5.0
    loss, logs0 = feasibility_losses(preds, tgt, cfg0h, iteration=10_000)
    check("대조군: 가중치 0 이면 hold 항 없음", "feas/offset_hold" not in logs0
          and not (loss.requires_grad and torch.autograd.grad(loss, off, allow_unused=True)[0] is not None
                   and float(torch.autograd.grad(loss, off, allow_unused=True, retain_graph=True)[0].abs().sum()) > 0))

    n = len(OK)
    print(f"\n{sum(OK)}/{n} 통과")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
