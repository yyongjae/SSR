"""Does the corridor-conditioned ranking loss (feasibility.admissible_rank_loss) rank only inside the corridor?

    K1  corridor mode: a candidate the GT corridor rejects gets EXACTLY zero gradient on its score
        heads, however good its label; control: rank_set=all gives it a gradient
    K2  inside the corridor the descent raises the label-best candidate's score and lowers the others
    K3  scores ordered like the labels -> lower loss than the reverse order; top1_hit/regret logs agree
    K4  a scene with < 2 admissible candidates, or without labels (sim_reward_valid=0), contributes nothing
    K5  weight 0 -> (None, {})
    F1  finetune_score_only: only reward/sim heads (+offset_embed) stay trainable, frozen modules stay
        in eval mode after model.train()

    python tools/plan_v2/test_rank_loss.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
from navsim.agents.para_ssr.configs.default import ParaSSRConfig                          # noqa: E402
from navsim.agents.para_ssr.feasibility import admissible_rank_loss, target_shape         # noqa: E402

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'통과' if cond else '실패 <<<'}  {name}  {detail}")


T, K = 8, 4


def straight(dist):
    p = torch.zeros(T, 3); p[:, 0] = torch.linspace(dist / T, dist, T); return p


def setup(ep_heads, ep_labels, wall_at=None, valid=1.0, **kw):
    kw.setdefault("rank_set", "corridor")
    cfg = ParaSSRConfig(rank_weight=1.0, plan_head_mode="corridor", **kw)
    th, tw = target_shape(cfg)
    xs = torch.linspace(cfg.map_pc_range[1], cfg.map_pc_range[4], th)[:, None]
    field = torch.full((1, th, tw), 8.0) if wall_at is None else (wall_at - xs).clamp(-8, 8).expand(th, tw)[None].clone()
    sim = torch.full((1, 4, K), 0.9)
    sim[0, 1] = torch.tensor(ep_heads)
    sim.requires_grad_(True)
    im = torch.full((1, K), 1.0 / K)
    preds = {"im_rewards": im, "sim_rewards": sim,
             "trajectory_anchors": torch.stack([straight(5.0), straight(8.0), straight(25.0), straight(6.0)]),
             "trajectory_offset": torch.zeros(1, K, T, 3)}
    lab = torch.full((1, 5, K), 1.0)
    lab[0, 2] = torch.tensor(ep_labels)
    preds["drivable_sdf_pred"] = field[:, ::4, ::4].clone()      # the model's own field, BEV grid
    tgt = {"drivable_sdf_bev": field, "sim_reward_valid": torch.tensor([valid])}
    return cfg, preds, tgt, lab, sim


def main():
    print("K1  corridor 밖 후보는 기울기 0")
    # candidate 2 (25 m) crosses the wall at 15 m and has the BEST label
    cfg, p, t, lab, sim = setup([0.5, 0.5, 0.5, 0.5], [0.3, 0.5, 1.0, 0.4], wall_at=15.0)
    loss, logs = admissible_rank_loss(p, t, lab, cfg)
    g, = torch.autograd.grad(loss, sim)
    check("후보 2 (벽 넘음, 라벨 최고): EP 헤드 기울기 = 0", float(g[0, 1, 2].abs()) == 0.0)
    check("admissible_n = 3", abs(float(logs["rank/admissible_n"]) - 3) < 1e-6, f"{float(logs['rank/admissible_n'])}")
    cfg, p, t, lab, sim = setup([0.5, 0.5, 0.5, 0.5], [0.3, 0.5, 1.0, 0.4], wall_at=15.0, rank_set="all")
    loss, _ = admissible_rank_loss(p, t, lab, cfg)
    g_all, = torch.autograd.grad(loss, sim)
    check("대조군 rank_set=all: 후보 2 에 기울기 (올리는 방향)", float(-g_all[0, 1, 2]) > 0)

    print("K2  corridor 안: 라벨 1위를 올리고 나머지를 내림")
    d = -g[0, 1]                                  # descent direction on the EP head outputs
    check("후보 1 (안쪽 라벨 1위 0.5) 상승, 0·3 하강", float(d[1]) > 0 and float(d[0]) < 0 and float(d[3]) < 0,
          f"{[round(float(v), 4) for v in d]}")

    print("K3  라벨 순서와 같은 점수가 손실이 낮음")
    cfg, p, t, lab, _ = setup([0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8])
    good, lg = admissible_rank_loss(p, t, lab, cfg)
    cfg, p, t, lab, _ = setup([0.8, 0.6, 0.4, 0.2], [0.2, 0.4, 0.6, 0.8])
    bad, lb = admissible_rank_loss(p, t, lab, cfg)
    check("같은 순서 < 반대 순서", float(good) < float(bad), f"{float(good):.3f} < {float(bad):.3f}")
    check("top1_hit 1 / 0, regret 0 / >0", float(lg["rank/top1_hit"]) == 1 and float(lb["rank/top1_hit"]) == 0
          and float(lg["rank/top1_regret"]) == 0 and float(lb["rank/top1_regret"]) > 0)

    print("K4  집합이 작거나 라벨이 없으면 0")
    cfg, p, t, lab, sim = setup([0.5] * 4, [0.3, 0.5, 1.0, 0.4], wall_at=3.0)   # everything crosses
    loss, logs = admissible_rank_loss(p, t, lab, cfg)
    check("전부 corridor 밖 → 손실 0", float(loss) == 0.0 and float(logs["rank/scenes"]) == 0)
    cfg, p, t, lab, sim = setup([0.5] * 4, [0.3, 0.5, 1.0, 0.4], valid=0.0)
    loss, _ = admissible_rank_loss(p, t, lab, cfg)
    check("sim_reward_valid=0 → 손실 0", float(loss) == 0.0)

    print("G1  배포 게이트 집합 + 전체 라벨")
    # the predicted field walls off at 15 m: candidate 2 (25 m) is out of the deployment set
    cfg, p, t, lab, sim = setup([0.5] * 4, [0.3, 0.5, 1.0, 0.4], wall_at=15.0, rank_set="gate")
    loss, logs = admissible_rank_loss(p, t, lab, cfg)
    g, = torch.autograd.grad(loss, sim)
    check("게이트가 거른 후보 2: 기울기 0", float(g[0, 1, 2].abs()) == 0.0)
    # the gate admits candidate 1, but its label says it leaves the drivable area (DAC 0): it must rank low
    cfg, p, t, lab, sim = setup([0.5] * 4, [0.3, 0.9, 1.0, 0.4], rank_set="gate")
    lab[0, 1, 1] = 0.0
    loss, _ = admissible_rank_loss(p, t, lab, cfg)
    g, = torch.autograd.grad(loss, sim)
    d = -g[0, 1]
    check("게이트 통과·DAC 라벨 0 후보 1: 하강 / 실제 1위 후보 2: 상승", float(d[1]) < 0 and float(d[2]) > 0,
          f"{[round(float(v), 4) for v in d]}")
    print("G2  대조군 all 은 전체 라벨: DAC 0 인 후보를 올리지 않음")
    cfg, p, t, lab, sim = setup([0.5] * 4, [0.3, 0.5, 1.0, 0.4], rank_set="all")
    lab[0, 1, 2] = 0.0                                   # the farthest candidate is off-road by label
    loss, _ = admissible_rank_loss(p, t, lab, cfg)
    g, = torch.autograd.grad(loss, sim)
    check("후보 2 (EP 최고지만 DAC 0): 하강", float(-g[0, 1, 2]) < 0)

    print("K5  꺼짐")
    cfg, p, t, lab, _ = setup([0.5] * 4, [0.5] * 4)
    cfg.rank_weight = 0.0
    check("(None, {})", admissible_rank_loss(p, t, lab, cfg) == (None, {}))

    print("F1  점수 헤드만 학습")
    from navsim.agents.para_ssr.para_ssr_model import ParaSSRModel
    import numpy as np
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    mc = ParaSSRConfig(plan_anchor=True, plan_anchor_file=str(ROOT / "data/planning_vb/trajectory_anchors_256.npy"),
                       image_architecture="resnet34.tv_in1k", backbone_pretrained=False, finetune_score_only=True,
                       plan_head_mode="corridor", plan_score_sees_offset=True)
    mc.trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    m = ParaSSRModel(mc)
    n = m.freeze_all_but_score_heads()
    train = sorted({k.split(".")[2] for k, p in m.named_parameters() if p.requires_grad})
    check("학습 대상 = reward_head / sim_reward_heads / offset_embed", train == ["offset_embed", "reward_head", "sim_reward_heads"],
          f"{train}, {n} params")
    m.train()
    bns = [(k, x.training) for k, x in m.named_modules() if isinstance(x, torch.nn.BatchNorm2d)]
    heads = [x.training for k, x in m.named_modules() if k.startswith("pts_bbox_head.anchor_planner.sim_reward_heads")]
    check("고정된 BatchNorm 은 eval, 점수 헤드는 train", bns and not any(t for _, t in bns) and all(heads))
    check("루트 모델은 train (학습 경로 유지)", m.training)

    print(f"\n{sum(OK)}/{len(OK)} 통과")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
