"""Does feasibility.selection_curse_loss correct only the would-be picks, and only downward-biased?

    C1  weight 0 -> (None, {})
    C2  curse_set=topk: S = top-k under the DEPLOYED rule (plan_final_rewards, then the static
        predicted-field gate).  Candidates outside S get EXACTLY zero gradient; a candidate with the
        highest plan_final_rewards that the field gate rejects is outside S
    C3  asymmetry: a head over-rating its label by 0.2 gets (1-tau)/tau times the gradient of one
        under-rating by 0.2, and descent lowers the over-rated one
    C4  curse_set=all: every candidate gets a gradient (the control)
    C5  only the EP / TTC / C heads learn; the feasible head gets zero gradient
    C6  a scene with sim_reward_valid=0 contributes nothing
    C7  logs: pick_over_* is the picked candidate's (head - label)
    V1  v2+seg layout (wote heads, no field): reads EP/TTC/C at sim[:, 2:5], NC/DAC get zero gradient
    W1  wiring: ParaSSRLoss adds it to the plan loss

    python tools/plan_v2/test_curse_loss.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
from navsim.agents.para_ssr.configs.default import ParaSSRConfig                          # noqa: E402
from navsim.agents.para_ssr.feasibility import selection_curse_loss, target_shape         # noqa: E402

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'통과' if cond else '실패 <<<'}  {name}  {detail}")


T, K = 8, 4


def straight(dist):
    p = torch.zeros(T, 3); p[:, 0] = torch.linspace(dist / T, dist, T); return p


def setup(final, heads, labels, wall_at=None, valid=1.0, **kw):
    """final [K] deployed score; heads/labels [3, K] for EP, TTC, C."""
    cfg = ParaSSRConfig(curse_weight=1.0, plan_head_mode="corridor", curse_topk=2, **kw)
    th, tw = target_shape(cfg)
    xs = torch.linspace(cfg.map_pc_range[1], cfg.map_pc_range[4], th)[:, None]
    field = torch.full((1, th, tw), 8.0) if wall_at is None else (wall_at - xs).clamp(-8, 8).expand(th, tw)[None].clone()
    sim = torch.full((1, 4, K), 0.9)
    sim[0, 1:] = torch.tensor(heads)
    sim.requires_grad_(True)
    lab = torch.full((1, 5, K), 1.0)
    lab[0, 2], lab[0, 3], lab[0, 4] = (torch.tensor(r) for r in labels)        # EP, TTC, C columns
    preds = {"sim_rewards": sim, "plan_final_rewards": torch.tensor([final]).float(),
             "trajectory_anchors": torch.stack([straight(5.0), straight(8.0), straight(25.0), straight(6.0)]),
             "trajectory_offset": torch.zeros(1, K, T, 3), "drivable_sdf_pred": field[:, ::4, ::4].clone()}
    tgt = {"sim_reward_valid": torch.tensor([valid])}
    return cfg, preds, tgt, lab, sim


def main():
    H = [[0.5] * K] * 3
    print("C1  꺼짐")
    cfg, p, t, lab, _ = setup([1, 2, 3, 4], H, H)
    cfg.curse_weight = 0.0
    check("(None, {})", selection_curse_loss(p, t, lab, cfg) == (None, {}))

    print("C2  배포 규칙 상위 k 만 기울기, 장 게이트가 거른 후보는 제외")
    over = [[0.9] * K] * 3; y = [[0.5] * K] * 3
    # candidate 2 (25 m) has the highest final score but crosses the wall at 15 m -> gated out
    cfg, p, t, lab, sim = setup([1.0, 3.0, 9.0, 2.0], over, y, wall_at=15.0)
    loss, logs = selection_curse_loss(p, t, lab, cfg)
    g, = torch.autograd.grad(loss, sim)
    nz = (g[0, 1:].abs().sum(0) > 0).tolist()
    check("S = {1, 3} (k=2, 후보 2 는 게이트 탈락)", nz == [False, True, False, True], f"{nz}")
    check("S 밖 후보 0·2 기울기 정확히 0", float(g[0, 1:, 0].abs().sum()) == 0 and float(g[0, 1:, 2].abs().sum()) == 0)

    print("C3  비대칭: 과대평가가 더 비쌈")
    cfg, p, t, lab, sim = setup([4.0, 3.0, 1.0, 2.0], [[0.7, 0.3, 0.5, 0.5]] * 3, [[0.5, 0.5, 0.5, 0.5]] * 3)
    loss, _ = selection_curse_loss(p, t, lab, cfg)
    g, = torch.autograd.grad(loss, sim)
    go, gu = float(g[0, 2, 0]), float(g[0, 2, 1])
    check("과대(+0.2) 기울기 / 과소(-0.2) 기울기 = (1-τ)/τ = 4", abs(go / -gu - 4.0) < 1e-4, f"{go:.4f} / {gu:.4f}")
    check("하강 방향: 과대 후보는 내리고 과소 후보는 올림", go > 0 and gu < 0)

    print("C4  대조군 all: 전 후보 기울기")
    cfg, p, t, lab, sim = setup([1.0, 3.0, 9.0, 2.0], over, y, wall_at=15.0, curse_set="all")
    loss, _ = selection_curse_loss(p, t, lab, cfg)
    g, = torch.autograd.grad(loss, sim)
    check("4 후보 모두 기울기", bool((g[0, 1:].abs().sum(0) > 0).all()))

    print("C5  feasible 헤드는 기울기 0")
    check("sim[:, 0] 기울기 0", float(g[0, 0].abs().sum()) == 0.0)

    print("C6  라벨 없는 장면은 0")
    cfg, p, t, lab, sim = setup([1.0, 3.0, 9.0, 2.0], over, y, valid=0.0)
    loss, _ = selection_curse_loss(p, t, lab, cfg)
    check("손실 0", float(loss) == 0.0)

    print("C7  로그")
    heads = [[0.9, 0.2, 0.5, 0.5], [0.6, 0.2, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]
    labels = [[0.7, 0.5, 0.5, 0.5], [0.9, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]
    cfg, p, t, lab, sim = setup([5.0, 1.0, 0.0, 0.0], heads, labels)
    _, logs = selection_curse_loss(p, t, lab, cfg)
    check("선택 = 후보 0: EP 과대 +0.2, TTC 과대 -0.3",
          abs(float(logs["curse/pick_over_ep"]) - 0.2) < 1e-5 and abs(float(logs["curse/pick_over_ttc"]) + 0.3) < 1e-5,
          f"{float(logs['curse/pick_over_ep']):+.3f} / {float(logs['curse/pick_over_ttc']):+.3f}")

    print("V1  v2+seg (wote 헤드 NC, DAC, EP, TTC, C; 장 없음)")
    cfg = ParaSSRConfig(curse_weight=1.0, plan_head_mode="wote", curse_topk=2)
    sim = torch.full((1, 5, K), 0.5)
    sim[0, 2:] = torch.tensor([[0.9, 0.2, 0.5, 0.5], [0.6, 0.2, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
    sim.requires_grad_(True)
    lab = torch.full((1, 5, K), 0.5)
    lab[0, 2], lab[0, 3] = torch.tensor([0.7, 0.5, 0.5, 0.5]), torch.tensor([0.9, 0.5, 0.5, 0.5])
    p = {"sim_rewards": sim, "plan_final_rewards": torch.tensor([[5.0, 4.0, 0.0, 0.0]])}
    loss, logs = selection_curse_loss(p, {"sim_reward_valid": torch.tensor([1.0])}, lab, cfg)
    g, = torch.autograd.grad(loss, sim)
    check("EP 과대 +0.2 / TTC 과대 -0.3 (열 2, 3 을 읽음)",
          abs(float(logs["curse/pick_over_ep"]) - 0.2) < 1e-5 and abs(float(logs["curse/pick_over_ttc"]) + 0.3) < 1e-5)
    check("NC·DAC 헤드 기울기 0, S = 상위 2 {0, 1}",
          float(g[0, :2].abs().sum()) == 0 and (g[0, 2:].abs().sum(0) > 0).tolist() == [True, True, False, False])

    print("W1  배선")
    src = (ROOT / "navsim/agents/para_ssr/para_ssr_loss.py").read_text()
    check("손실: selection_curse_loss(predictions, targets, sim_reward, cfg) 를 plan 에 더함",
          "selection_curse_loss(predictions, targets, sim_reward, cfg)" in src and 'logs["loss_curse"]' in src)

    print(f"\n{sum(OK)}/{len(OK)} 통과")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
