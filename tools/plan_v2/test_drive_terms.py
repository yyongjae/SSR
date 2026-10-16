"""Does feasibility.drive_terms push and resist where it should, and nowhere else?

    D0  all weights 0 -> (None, {})  (the default arms are untouched)
    P1  a candidate short of the human's route progress gets a FORWARD gradient on its offset;
        control: one at/beyond the human gets exactly zero
    P2  a candidate the corridor verdict rejects (end inside a wall) is not pushed; control: the
        same candidate on an all-drivable field is
    P3  a scene whose human moves < 2 m (red light) pushes nothing; a scene without a route neither
    P4  a candidate outside S gets no gradient from any term
    L1  lateral: an end 3 m off the centerline is pulled TOWARD it; one within `free` gets zero
    C1  continuity: a first step matching v + a dt/2 gets zero gradient; a first step at 0 m/s while
        driving 10 m/s is pushed to speed up, and nothing beyond step 0 is touched
    W1  wired into ParaSSRLoss: loss_drive appears in the logs only when a weight is > 0

    python tools/plan_v2/test_drive_terms.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
from navsim.agents.para_ssr.configs.default import ParaSSRConfig                         # noqa: E402
from navsim.agents.para_ssr.feasibility import drive_terms, target_shape                  # noqa: E402

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'통과' if cond else '실패 <<<'}  {name}  {detail}")


T, K = 8, 4


def straight(dist, y=0.0):
    p = torch.zeros(T, 3); p[:, 0] = torch.linspace(dist / T, dist, T); p[:, 1] = y; return p


def setup(cands, human=20.0, field=None, route=True, **kw):
    base = dict(feas_weight=1.0, feas_field="gt", feas_dyn_weight=0.0, feas_select_model_k=K,
                feas_select_label_k=0)
    base.update(kw)
    cfg = ParaSSRConfig(**base)
    th, tw = target_shape(cfg)
    r = torch.full((161, 2), float("nan"))
    if route:
        r[:120, 0] = torch.arange(-10, 110).float(); r[:120, 1] = 0.0
    off = torch.zeros(1, K, T, 3, requires_grad=True)
    preds = {"trajectory_anchors": torch.stack(cands), "trajectory_offset": off,
             "plan_final_rewards": torch.zeros(1, K)}
    tgt = {"route_centerline": r[None], "trajectory": straight(human)[None],
           "drivable_sdf_bev": field if field is not None else torch.full((1, th, tw), 8.0)}
    feats = {"status_feature": torch.zeros(1, 8)}
    return cfg, preds, tgt, feats, off


def grad_of(cfg, preds, tgt, feats, off):
    loss, logs = drive_terms(preds, tgt, feats, cfg)
    if loss is None or not loss.requires_grad:
        return torch.zeros_like(off), logs
    g, = torch.autograd.grad(loss, off, allow_unused=True)
    return (g if g is not None else torch.zeros_like(off)), logs


def main():
    print("D0  가중치 0 → 꺼짐")
    cfg, p, t, f, off = setup([straight(10.0)] * K)
    check("(None, {})", drive_terms(p, t, f, cfg) == (None, {}))

    print("P1  사람보다 덜 간 후보만 앞으로")
    cands = [straight(10.0), straight(20.0), straight(30.0), straight(5.0)]
    cfg, p, t, f, off = setup(cands, feas_progress_weight=1.0)
    g, logs = grad_of(cfg, p, t, f, off)
    gx = -g[0, :, -1, 0]                                     # descent direction on the end x
    check("10 m, 5 m 후보: 끝점 전진 방향(+x)", bool(gx[0] > 0 and gx[3] > 0), f"{gx.tolist()}")
    check("대조군 20 m(=사람), 30 m: 기울기 0", float(g[0, 1].abs().max()) == 0 and float(g[0, 2].abs().max()) == 0)
    check("마지막 스텝 외엔 기울기 0 (진행은 끝점에서 측정)", float(g[0, :, :-1].abs().max()) == 0)

    print("P2  corridor가 거부한 후보는 밀지 않음")
    cfg0 = ParaSSRConfig()
    th, tw = target_shape(cfg0)
    xs = torch.linspace(cfg0.map_pc_range[1], cfg0.map_pc_range[4], th)[:, None]
    wall = torch.where(xs > 12.0, torch.tensor(-3.0), torch.tensor(8.0)).expand(th, tw)[None].clone()
    cfg, p, t, f, off = setup(cands, field=wall, feas_progress_weight=1.0)
    g, logs = grad_of(cfg, p, t, f, off)
    check("10 m 후보(앞범퍼 ~13.9 m, 벽 12 m 넘음): 기울기 0", float(g[0, 0].abs().max()) == 0)
    check("5 m 후보(앞범퍼 ~8.9 m, 벽 안): 여전히 전진", float(-g[0, 3, -1, 0]) > 0)
    check("gate_pass_S = 0.25 (5 m 후보만 통과)",
          abs(float(logs["drive/gate_pass_S"]) - 0.25) < 1e-6, f"{float(logs['drive/gate_pass_S']):.3f}")

    print("P3  사람이 거의 안 움직이거나 경로가 없으면 0")
    cfg, p, t, f, off = setup(cands, human=1.0, feas_progress_weight=1.0)
    g, _ = grad_of(cfg, p, t, f, off)
    check("사람 1 m: 전부 0", float(g.abs().max()) == 0)
    cfg, p, t, f, off = setup(cands, route=False, feas_progress_weight=1.0, feas_lateral_weight=1.0)
    g, _ = grad_of(cfg, p, t, f, off)
    check("경로 없음: 전부 0 (NaN 없음)", float(g.abs().max()) == 0 and not torch.isnan(g).any())

    print("P4  S 밖 후보는 기울기 0")
    cfg, p, t, f, off = setup(cands, feas_progress_weight=1.0, feas_lateral_weight=1.0,
                              feas_continuity_weight=1.0, feas_select_model_k=1)
    p["plan_final_rewards"] = torch.tensor([[0.0, 0.0, 0.0, 5.0]])   # model top-1 = cand 3; WTA = cand 1
    g, _ = grad_of(cfg, p, t, f, off)
    check("후보 0, 2 (S 밖): 0", float(g[0, 0].abs().max()) == 0 and float(g[0, 2].abs().max()) == 0)
    check("후보 3 (S 안, 5 m): 기울기 있음", float(g[0, 3].abs().max()) > 0)

    print("L1  차선 중심선 쪽으로")
    cands = [straight(20.0, 3.0), straight(20.0, -3.0), straight(20.0, 0.5), straight(20.0, 0.0)]
    cfg, p, t, f, off = setup(cands, feas_lateral_weight=1.0)
    g, _ = grad_of(cfg, p, t, f, off)
    gy = -g[0, :, -1, 1]
    check("+3 m → -y, -3 m → +y", bool(gy[0] < 0 and gy[1] > 0), f"{gy.tolist()}")
    check("대조군 0.5 m, 0 m (free 1 m 이내): 0", float(g[0, 2:].abs().max()) == 0)

    print("C1  연속성: 첫 스텝 속도 vs 측정 운동")
    cands = [straight(40.0), straight(4.0), straight(40.0), straight(40.0)]   # first step 5 m / 0.5 m
    cfg, p, t, f, off = setup(cands, feas_continuity_weight=1.0, feas_beta=0.01)
    f["status_feature"][0, 4] = 10.0                                         # vx = 10 m/s, a = 0
    g, logs = grad_of(cfg, p, t, f, off)
    check("10 m/s 로 출발하는 후보: 0 (데드존)", float(g[0, 0].abs().max()) < 1e-6, f"{float(g[0,0].abs().max()):.1e}")
    check("1 m/s 로 출발하는 후보: 첫 스텝 앞으로", float(-g[0, 1, 0, 0]) > 0)
    check("스텝 1 이후는 건드리지 않음", float(g[0, :, 1:].abs().max()) == 0)
    f["status_feature"][0, 4] = 8.0; f["status_feature"][0, 6] = 8.0          # v + a dt/2 = 10 m/s
    g, _ = grad_of(cfg, p, t, f, off)
    check("가속도 반영: v 8 + a 8 → 10 m/s 출발 후보 0", float(g[0, 0].abs().max()) < 1e-6)

    print("W1  ParaSSRLoss 배선")
    src = (ROOT / "navsim/agents/para_ssr/para_ssr_loss.py").read_text()
    check("drive_terms 호출 + loss_drive 로그", "drive_terms(predictions, targets, features, cfg)" in src
          and 'logs["loss_drive"]' in src)

    print(f"\n{sum(OK)}/{len(OK)} 통과")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
