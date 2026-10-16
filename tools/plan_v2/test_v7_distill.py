"""Does the ported BEV selector v7 distillation (distill/) do what the teammate's does, and only when on?

    D1  the Gaussian spatial bias peaks ON the anchor path (0 at a waypoint) and falls off with distance
    D2  identical teacher and student BEV -> feature loss 0 and attention KL 0 (identity student_proj)
    D3  the teacher branch gets NO gradient; the student BEV and the module's parameters do
    D4  sample_valid: all frames valid == the teammate's unmasked loss; a masked frame's teacher
        cannot change the loss
    S1  store: a real navtrain token loads (256, 50, 100) float16 from both caches; BEVFusion = the
        stored map with W flipped (manifest), ReSMap = BevCache.bev (plain transpose, measured)
    W1  wiring: loss owns v7_distill only when on; the target builder appears only when on; checkpoint
        loading treats _loss.v7_distill.* as training-only; launcher switch V7=1
    E1  end to end on CPU, v2+seg config, 2 real frames: flag off -> no v7 keys; flag on -> total =
        off total + weight * sum(v7 terms) exactly, and the v7 module + student BEV get gradient

    python tools/plan_v2/test_v7_distill.py [--no-e2e]
"""
import pickle
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/readout"))
import _env  # noqa: F401,E402
from navsim.agents.para_ssr.configs.default import ParaSSRConfig                          # noqa: E402
from navsim.agents.para_ssr.distill import TrajectoryAnchorDistillation, v7_stores        # noqa: E402

OK = []
ANCHORS = ROOT / "data/planning_vb/trajectory_anchors_256.npy"


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'통과' if cond else '실패 <<<'}  {name}  {detail}")


def module(**kw):
    torch.manual_seed(0)
    return TrajectoryAnchorDistillation(trajectory_anchors_file=str(ANCHORS), **kw)


def main():
    cfg = ParaSSRConfig()
    H, W, C = cfg.bev_h, cfg.bev_w, cfg.embed_dims
    A = torch.from_numpy(np.load(ANCHORS)).float()
    m = module()

    print("D1  spatial bias 는 앵커 경로 위에서 최대")
    inside = ((A[:, -1, 0] > 10) & (A[:, -1, 0] < 30) & (A[:, -1, 1].abs() < 20)).nonzero()[:, 0]
    k = int(inside[0])                                           # an anchor ending inside the grid
    x, y = float(A[k, -1, 0]), float(A[k, -1, 1])
    r, c = int(x / 0.64), int((-y + 32.0) / 0.64)
    b = m.spatial_bias[k].view(H, W)
    check(f"앵커 {k} 끝점 ({x:.1f}, {y:.1f}) 칸의 bias ≈ 0", float(b[r, c]) > -0.05, f"{float(b[r, c]):.3f}")
    far = float(b[0, 0]) if abs(c) > 40 else float(b[0, W - 1])
    check("경로에서 먼 칸은 강하게 음수 (< -5)", far < -5.0, f"{far:.1f}")
    check("bias ≤ 0 전부, 모양 [256, 5000]", float(m.spatial_bias.max()) <= 0 and tuple(m.spatial_bias.shape) == (256, H * W))

    print("D2  teacher == student → 손실 0")
    torch.manual_seed(1)
    bev = torch.randn(2, H * W, C); pos = torch.randn(2, H * W, C) * 0.1
    gt = A[[3, 100]].clone()
    cmd = torch.eye(4)[[0, 2]]; st = torch.randn(2, 4)
    losses, met = m(bev, (bev.clone(), bev.clone()), A, gt_trajectory=gt, command=cmd, ego_status=st, bev_pos=pos)
    check("loss_distill_tok ≈ 0, distill_cos ≈ 1", float(losses["loss_distill_tok"]) < 1e-4
          and float(met["distill_cos/resmap"]) > 0.999, f"{float(losses['loss_distill_tok']):.2e}")

    print("D3  teacher 는 기울기 없음, student·모듈은 있음")
    s = torch.randn(2, H * W, C, requires_grad=True)
    t1 = torch.randn(2, H * W, C, requires_grad=True); t2 = torch.randn(2, H * W, C, requires_grad=True)
    losses, _ = m(s, (t1, t2), A, gt_trajectory=gt, command=cmd, ego_status=st, bev_pos=pos)
    total = sum(v for v in losses.values())                       # the teammate's summation
    total.backward()
    check("teacher 입력 grad 없음", t1.grad is None and t2.grad is None)
    check("student BEV grad > 0", s.grad is not None and float(s.grad.abs().sum()) > 0)
    check("q/k/v/out/anchor_mlp 모두 grad", all(p.grad is not None for n, p in m.named_parameters()),
          f"{[n for n, p in m.named_parameters() if p.grad is None][:3]}")
    ratio = float(total) / float(losses["loss_distill_bevfusion"] + losses["loss_distill_resmap"]
                                 + losses["loss_attn_bevfusion"] + losses["loss_attn_resmap"] + losses["loss_distill_tok"])
    check("팀원 합산 그대로 (bank 두 개 + attn 로그 + tok)", abs(ratio - 1) < 1e-6)

    print("D4  sample_valid")
    m.zero_grad()
    with torch.no_grad():
        l0, _ = m(s, (t1, t2), A, gt_trajectory=gt, command=cmd, ego_status=st, bev_pos=pos)
        l1, _ = m(s, (t1, t2), A, gt_trajectory=gt, command=cmd, ego_status=st, bev_pos=pos,
                  sample_valid=torch.ones(2))
        t1b, t2b = t1.clone(), t2.clone(); t1b[1] += 5.0; t2b[1] -= 5.0
        la, _ = m(s, (t1, t2), A, gt_trajectory=gt, command=cmd, ego_status=st, bev_pos=pos,
                  sample_valid=torch.tensor([1.0, 0.0]))
        lb, _ = m(s, (t1b, t2b), A, gt_trajectory=gt, command=cmd, ego_status=st, bev_pos=pos,
                  sample_valid=torch.tensor([1.0, 0.0]))
    check("전부 valid == 마스크 없음 (팀원 손실)", abs(float(l0["loss_distill_tok"]) - float(l1["loss_distill_tok"])) < 1e-6)
    check("무효 프레임의 teacher 를 바꿔도 손실 불변", abs(float(la["loss_distill_tok"]) - float(lb["loss_distill_tok"])) < 1e-6)

    print("S1  실제 캐시 토큰")
    bf, rs = v7_stores(cfg)
    bf.validate_manifest(cfg); rs.validate_manifest(cfg)
    from navsim.agents.para_ssr.readout.bev_cache import BevCache
    tok = next(t for t in list(rs.index())[:5000] if bf.has_token(t))
    a, r_ = bf.load(tok), rs.load(tok)
    check("두 teacher 모두 (256, 50, 100) float16", a.shape == r_.shape == (C, H, W) and a.dtype == r_.dtype == np.float16)
    with np.load(bf.path_for(tok)) as z:
        raw = z["bev_feature"]
    check("BEVFusion = 저장본[:, :, ::-1] (manifest)", np.array_equal(a, raw[:, :, ::-1]))
    check("ReSMap = BevCache.bev (평면 전치, 뒤집기 없음)", np.array_equal(r_, BevCache(rs.teacher_root()).bev(tok)))
    check("없는 토큰 → None", bf.load("ffffffffffffffff") is None and rs.load("ffffffffffffffff") is None)

    print("W1  배선")
    loss_src = (ROOT / "navsim/agents/para_ssr/para_ssr_loss.py").read_text()
    agent_src = (ROOT / "navsim/agents/para_ssr/para_ssr_agent.py").read_text()
    sh = (ROOT / "data/logs/train_v2_arm.sh").read_text()
    check("손실: v7_distill 은 켤 때만, 균형 뒤에 total_loss 에 가산", "self.v7_distill = None" in loss_src
          and "total_loss = total_loss + float(cfg.distill_v7_weight) * value" in loss_src)
    check("에이전트: V7TeacherTargetBuilder (켤 때만), ckpt 의 _loss.v7_distill. 무시",
          "V7TeacherTargetBuilder(cfg)" in agent_src and '"_loss.v7_distill."' in agent_src)
    check("런처: V7=1 → distill_v7=true, 접미사 _v7", "agent.config.distill_v7=true" in sh and 'SFX="${SFX}_v7"' in sh)
    from navsim.agents.para_ssr.para_ssr_loss import ParaSSRLoss
    check("기본 설정: loss.v7_distill is None", ParaSSRLoss(cfg).v7_distill is None)

    if "--no-e2e" not in sys.argv:
        e2e(cfg)
    print(f"\n{sum(OK)}/{len(OK)} 통과")
    sys.exit(0 if all(OK) else 1)


def e2e(cfg0):
    print("E1  CPU 끝단 (v2+seg 설정, 실제 프레임 2개)")
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
    from navsim.common.dataloader import SceneLoader
    from _env import scene_filter, split_dirs
    ts = TrajectorySampling(time_horizon=4, interval_length=0.5)
    base = dict(use_task_interaction=True, use_det_motion_head=True, use_map_head=True,
                grad_balance_target={"plan": 0.4, "det": 0.3, "map": 0.3}, plan_anchor=True,
                plan_anchor_file=str(ANCHORS), plan_score_file=str(ROOT / "data/planning_vb/pdm_score_256"),
                image_architecture="resnet34.tv_in1k", backbone_pretrained=False, plan_heading_from_xy=True,
                map_head_type="seg", bev_seg_weight=3.0)
    cfg_off = replace(cfg0, **base); cfg_off.trajectory_sampling = ts
    cfg_on = replace(cfg_off, distill_v7=True); cfg_on.trajectory_sampling = ts
    off, on = ParaSSRAgent(cfg_off, ts), ParaSSRAgent(cfg_on, ts)
    names_off = [type(b).__name__ for b in off.get_target_builders()]
    names_on = [type(b).__name__ for b in on.get_target_builders()]
    check("target builder: 끌 때 없음 / 켤 때 V7TeacherTargetBuilder", "V7TeacherTargetBuilder" not in names_off
          and "V7TeacherTargetBuilder" in names_on)
    bf, rs = v7_stores(cfg_on)
    logs_dir, blobs_dir = split_dirs("trainval")
    rng = random.Random(0)
    toks, log = [], None
    for lp in rng.sample(sorted(logs_dir.glob("*.pkl")), 200):
        cand = [f["token"] for f in pickle.load(open(lp, "rb")) if bf.has_token(f["token"]) and rs.has_token(f["token"])]
        if len(cand) >= 20:
            toks, log = cand[10:12], lp.stem
            break
    loader = SceneLoader(logs_dir, blobs_dir, scene_filter("navtrain", log_names=[log], tokens=toks),
                         sensor_config=on.get_sensor_config())
    toks = loader.tokens[:2]
    check("프레임 2개 확보", len(toks) == 2, f"{log}")
    fb = on.get_feature_builders()
    feats, tgts = [], []
    for t in toks:
        ai, sc = loader.get_agent_input_from_token(t), loader.get_scene_from_token(t)
        f = {}
        for b in fb:
            f.update(b.compute_features(ai))
        g = {}
        for b in on.get_target_builders():
            g.update(b.compute_targets(sc))
        feats.append(f); tgts.append(g)
    stack = lambda ds: {k: torch.stack([d[k] for d in ds]) for k in ds[0] if torch.is_tensor(ds[0][k])}  # noqa: E731
    F_, T_ = stack(feats), stack(tgts)
    check("v7 타깃 모양·valid", tuple(T_["v7_teacher_bevfusion"].shape) == (2, 256, 50, 100)
          and float(T_["v7_teacher_valid"].min()) == 1.0)
    on.para_ssr_model.load_state_dict(off.para_ssr_model.state_dict())
    model = off.para_ssr_model
    model.train()
    torch.manual_seed(0)
    pred = model(F_)
    T_off = {k: v for k, v in T_.items() if not k.startswith("v7_")}
    l_off, logs_off = off._loss(model, F_, T_off, dict(pred))
    l_on, logs_on = on._loss(model, F_, T_, dict(pred))
    v7_keys = [k for k in logs_on if k.startswith("v7/loss_")]
    check("끌 때 v7 로그 없음", not any(k.startswith("v7/") for k in logs_off))
    extra = sum(float(logs_on[k]) for k in v7_keys) * cfg_on.distill_v7_weight
    check("켤 때 total = 끈 total + w·Σ v7 항", abs(float(l_on) - float(l_off) - extra) < 1e-3 * max(1.0, abs(extra)),
          f"{float(l_off):.4f} + {extra:.4f} vs {float(l_on):.4f}")
    l_on.backward()
    g_mod = [n for n, p in on._loss.v7_distill.named_parameters() if p.grad is None]
    check("v7 모듈 파라미터 전부 grad", not g_mod, f"{g_mod[:3]}")
    enc = [p for n, p in model.named_parameters() if n.startswith("pts_bbox_head") and p.grad is not None]
    check("학생 쪽 파라미터로 기울기 전달", len(enc) > 0)
    opt = on.get_optimizers()["optimizer"]
    ids = {id(p) for g in opt.param_groups for p in g["params"]}
    check("v7 모듈 파라미터가 optimizer 에 포함", all(id(p) in ids for p in on._loss.v7_distill.parameters()))
    print("   ", {k: round(float(logs_on[k]), 4) for k in sorted(logs_on) if k.startswith("v7/")})


if __name__ == "__main__":
    main()
