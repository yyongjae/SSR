"""Does stage 2 (refine.py) do what it claims, and only that?

    R0  the refiner starts as the identity (zero last layer): refined == picked candidate
    R1  stage 2 cannot move stage 1: no gradient reaches the offsets or the planner query
    R2  training refines [pick, WTA winner]; inference refines the pick only and emits it
    O1  corridor: a pick whose corner sits inside a wall gets pushed OUT (descent lowers the cost);
        control: the same pick on an all-free field has zero corridor gradient
    O2  progress: a pick short of the human is pushed forward; one at the human is not
    O3  smooth: a refinement that adds a zig-zag costs more than the same mean shift without one
    O4  test-time descent on the objective alone moves a wall-violating pick into the corridor
    HG  refine_corridor_hinge: a pick inside the margin of a 12 m road gets EXACTLY zero corridor force
        (softplus pushes it toward the middle); a pick whose corner crosses the edge is pushed back in
    FR  finetune_refiner_only: only refiner.* trains, frozen modules stay in eval mode
    F1  fallback: a refinement that leaves the corridor where the pick was inside it is discarded;
        control: a refinement that stays inside is kept
    W1  wiring: ParaSSRLoss calls refine_select/refine_losses; the model calls refine_select in eval

    python tools/plan_v2/test_refine.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
from navsim.agents.para_ssr.configs.default import ParaSSRConfig                          # noqa: E402
from navsim.agents.para_ssr.refine import (SelectedRefiner, merged_clearance,             # noqa: E402
                                           refine_objective, refine_select)

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'통과' if cond else '실패 <<<'}  {name}  {detail}")


T, K, C = 8, 4, 16


def straight(dist, y=0.0):
    p = torch.zeros(T, 3); p[:, 0] = torch.linspace(dist / T, dist, T); p[:, 1] = y; return p


def grid_x(cfg, h=50):
    return torch.linspace(cfg.map_pc_range[1], cfg.map_pc_range[4], h)[:, None]     # rows = forward


def main():
    cfg = ParaSSRConfig(refine_progress_weight=1.0, refine_lateral_weight=0.0, refine_continuity_weight=0.0)
    free = torch.full((1, 50, 100), 8.0)
    route = torch.full((1, 161, 2), float("nan")); route[0, :120, 0] = torch.arange(-10, 110).float(); route[0, :120, 1] = 0

    print("R0/R1/R2  항등 시작, 1단계 불변, 학습=[선택, WTA] / 추론=선택")
    ref = SelectedRefiner(C, cfg.map_pc_range)
    anchors = torch.stack([straight(10.0), straight(20.0), straight(30.0), straight(5.0)])
    off = torch.zeros(1, K, T, 3, requires_grad=True)
    h = torch.randn(1, K, C, requires_grad=True)
    preds = {"trajectory_anchors": anchors, "trajectory_offset": off, "plan_topk_index": torch.tensor([[2, 0, 1]]),
             "drivable_sdf_pred": free, "trajectory": (anchors[2] + off[0, 2]).unsqueeze(0)}
    st = torch.zeros(1, 4)
    out = refine_select(ref, preds, h.detach(), st, cfg, gt_trajectory=straight(19.0)[None])
    check("학습: refine_index = [선택 2, WTA 1]", out["refine_index"].tolist() == [[2, 1]], f"{out['refine_index'].tolist()}")
    check("항등 시작: refined == base", torch.allclose(out["refine_trajectory"], out["refine_base"]))
    loss = out["refine_trajectory"].sum()
    loss.backward()
    check("offset·query 로 기울기 0 (1단계 불변)", off.grad is None and h.grad is None)
    check("refiner 파라미터에는 기울기", ref.mlp[-1].weight.grad is not None and ref.mlp[-1].weight.grad.abs().sum() > 0)
    ref.eval()
    out = refine_select(ref, preds, h.detach(), st, cfg)
    check("추론: 선택만 다듬어 trajectory 로 내보냄", out["refine_index"].shape == (1, 1)
          and torch.allclose(out["trajectory"], anchors[2][None]))

    print("O1  corridor: 벽 안 모서리는 밖으로")
    xs = grid_x(cfg)
    wall = (12.0 - xs).clamp(-8.0, 8.0).expand(50, 100)[None].clone()      # a real SDF: slope 1 across the wall
    base = straight(12.0)[None, None]
    tr = base.clone().requires_grad_(True)
    c, _ = refine_objective(tr, base, wall, None, ParaSSRConfig(refine_progress_weight=0.0, refine_trust_weight=0.0,
                                                                refine_smooth_weight=0.0))
    g, = torch.autograd.grad(c.sum(), tr)
    check("끝 스텝 x 기울기 > 0 (뒤로 당김 = 비용 감소 방향 -x)", float(g[0, 0, -1, 0]) > 0, f"{float(g[0,0,-1,0]):.3f}")
    tr2 = straight(5.0)[None, None].clone().requires_grad_(True)
    c2, _ = refine_objective(tr2, tr2.detach(), free, None, ParaSSRConfig(refine_progress_weight=0.0, refine_trust_weight=0.0))
    g2, = torch.autograd.grad(c2.sum(), tr2)
    check("대조군: 전부 주행 가능하면 corridor 기울기 ≈ 0", float(g2.abs().max()) < 1e-6, f"{float(g2.abs().max()):.1e}")

    print("HG  hinge: 안쪽이면 힘 0, 테두리 넘으면 안으로")
    ys = torch.linspace(cfg.map_pc_range[0], cfg.map_pc_range[3], 100)[None, :]
    road = (6.0 - ys.abs()).clamp(-8.0, 8.0).expand(50, 100)[None].clone()   # |y| < 6 drivable, slope 1
    def corr_grad(y, hinge):
        tr = straight(10.0, y)[None, None].clone().requires_grad_(True)
        c, _ = refine_objective(tr, tr.detach(), road, None, ParaSSRConfig(
            refine_progress_weight=0.0, refine_trust_weight=0.0, refine_smooth_weight=0.0,
            refine_continuity_weight=0.0, refine_corridor_hinge=hinge))
        return torch.autograd.grad(c.sum(), tr)[0][0, 0, :, 1]
    gs, gh = corr_grad(4.0, False), corr_grad(4.0, True)
    check("y=4 (모서리 5 m, margin 안쪽): softplus 는 가운데로 밂", float(gs.abs().max()) > 1e-3, f"{float(gs.abs().max()):.2e}")
    check("y=4: hinge 는 정확히 0", float(gh.abs().max()) == 0.0)
    go = corr_grad(5.4, True)
    check("y=5.4 (모서리 6.4 m, 테두리 넘음): hinge 가 안으로 (|y| 감소 방향)", float((go * 5.4).sum()) > 0,
          f"{float(go.sum()):+.3f}")

    print("FR  stage 2 만 학습")
    from navsim.agents.para_ssr.para_ssr_model import ParaSSRModel
    import numpy as np
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    mc = ParaSSRConfig(plan_anchor=True, plan_anchor_file=str(ROOT / "data/planning_vb/trajectory_anchors_256.npy"),
                       image_architecture="resnet34.tv_in1k", backbone_pretrained=False, finetune_refiner_only=True,
                       plan_head_mode="corridor", plan_score_sees_offset=True, refine_enable=True, feas_weight=1.0)
    mc.trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    m = ParaSSRModel(mc)
    n = m.freeze_all_but(m.REFINER_PREFIXES)
    train = sorted({k.split(".")[0] for k, p in m.named_parameters() if p.requires_grad})
    check("학습 대상 = refiner", train == ["refiner"], f"{train}, {n} params")
    m.train()
    bns = [x.training for k, x in m.named_modules() if isinstance(x, torch.nn.BatchNorm2d)]
    ref = [x.training for k, x in m.named_modules() if k.startswith("refiner")]
    check("고정된 BatchNorm 은 eval, refiner 는 train", bns and not any(bns) and all(ref))

    print("O2  progress: 사람보다 덜 가면 앞으로")
    cf = ParaSSRConfig(refine_progress_weight=1.0, refine_corridor_weight=0.0, refine_trust_weight=0.0,
                       refine_smooth_weight=0.0, refine_lateral_weight=0.0)
    for d, expect in ((10.0, True), (20.0, False)):
        tr = straight(d)[None, None].clone().requires_grad_(True)
        c, _ = refine_objective(tr, tr.detach(), free, None, cf, route=route, human=straight(20.0)[None])
        g, = torch.autograd.grad(c.sum(), tr)
        ok = float(-g[0, 0, -1, 0]) > 0 if expect else float(g.abs().max()) == 0
        check(f"{d:.0f} m → {'전진' if expect else '기울기 0'}", ok, f"{float(-g[0,0,-1,0]):.3f}")

    print("O3  smooth: 지그재그 refinement 가 더 비쌈")
    cs = ParaSSRConfig(refine_progress_weight=0.0, refine_corridor_weight=0.0, refine_trust_weight=0.0)
    b = straight(20.0)[None, None]
    zz = b.clone(); zz[..., 1] += torch.tensor([0.5, -0.5] * 4)
    sh = b.clone(); sh[..., 1] += 0.0
    c_zz, _ = refine_objective(zz, b, free, None, cs); c_sh, _ = refine_objective(sh, b, free, None, cs)
    check("지그재그 > 이동 없음", float(c_zz) > float(c_sh) + 1.0, f"{float(c_zz):.2f} vs {float(c_sh):.2f}")

    print("O4  목적함수만으로 내리면 corridor 안으로")
    from navsim.agents.para_ssr.refine import path_poses
    base = straight(13.0)[None, None]
    xy = base[..., :2].clone().requires_grad_(True)
    opt = torch.optim.Adam([xy], lr=0.05)
    co = ParaSSRConfig(refine_progress_weight=0.0, refine_trust_weight=0.01)
    before = float(merged_clearance(base, wall, None, cfg.map_pc_range, 0.0, 0.0).amin())
    for _ in range(300):
        opt.zero_grad(); c, _ = refine_objective(path_poses(xy, base), base, wall, None, co); c.sum().backward(); opt.step()
    after = float(merged_clearance(path_poses(xy, base).detach(), wall, None, cfg.map_pc_range, 0.0, 0.0).amin())
    check("최소 여유 음수 → 0 이상", before < 0 <= after, f"{before:+.2f} → {after:+.2f}")

    print("F1  fallback: corridor 를 벗어나는 refinement 는 버림")
    ref2 = SelectedRefiner(C, cfg.map_pc_range).eval()
    with torch.no_grad():
        ref2.mlp[-1].bias.view(T, 2)[:, 0] = 6.0           # push every step 6 m forward
    an = torch.stack([straight(5.0)] * K)
    p2 = dict(preds, trajectory_anchors=an, trajectory=an[2][None], drivable_sdf_pred=wall,
              trajectory_offset=torch.zeros(1, K, T, 3))
    cfg_f = ParaSSRConfig(feas_select_margin=0.0, feas_select_dyn_margin=-1e9)
    o = refine_select(ref2, p2, torch.zeros(1, K, C), st, cfg_f)
    check("벽으로 들어가는 refinement → 선택 원본 유지", bool(o["refine_fallback_used"][0])
          and torch.allclose(o["trajectory"], an[2][None]))
    o = refine_select(ref2, dict(p2, drivable_sdf_pred=free), torch.zeros(1, K, C), st, cfg_f)
    check("대조군: 전부 주행 가능하면 refinement 채택", not bool(o["refine_fallback_used"][0])
          and float((o["trajectory"] - an[2][None])[..., 0].mean().detach()) > 5.0)

    print("H1  heading 은 경로에서: 자유 변수가 아님")
    ref3 = SelectedRefiner(C, cfg.map_pc_range).eval()
    with torch.no_grad():
        ref3.mlp[-1].bias.view(T, 2)[:, 1] = torch.linspace(0.2, 1.6, T)   # bend left
    o = refine_select(ref3, dict(preds, drivable_sdf_pred=free, trajectory_offset=torch.zeros(1, K, T, 3)),
                      torch.zeros(1, K, C), st, cfg_f)
    tr = o["trajectory"][0]
    from navsim.agents.para_ssr.modules.kinematics import bezier_xyyaw
    check("출력 heading == 출력 xy 의 bezier 접선", torch.allclose(tr, bezier_xyyaw(tr[None, :, :2], tr[None, :, 2])[0], atol=1e-5))
    check("왼쪽으로 휘면 heading 도 양수", float(tr[-1, 2]) > 0.02, f"{float(tr[-1, 2]):.3f}")

    print("W1  배선")
    loss_src = (ROOT / "navsim/agents/para_ssr/para_ssr_loss.py").read_text()
    model_src = (ROOT / "navsim/agents/para_ssr/para_ssr_model.py").read_text()
    check("손실: refine_select(gt) + refine_losses", "gt_trajectory=targets[\"trajectory\"]" in loss_src
          and "refine_losses(predictions, targets, features, cfg)" in loss_src)
    check("모델: 추론에서 refine_select, query detach", "refine_select(self.refiner" in model_src
          and 'outs["scene_query"].transpose(0, 1).detach()' in model_src)

    print(f"\n{sum(OK)}/{len(OK)} 통과")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
