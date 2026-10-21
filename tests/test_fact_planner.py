"""Factorized planner (navsim/agents/para_ssr/fact/): vocabulary, proxy labels, planner, losses (CPU)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from shapely.geometry import box

from navsim.agents.para_ssr.fact import losses as fl
from navsim.agents.para_ssr.fact.planner import FactorizedPlanner
from navsim.agents.para_ssr.fact.proxy import proxy_labels, route_progress
from navsim.agents.para_ssr.fact.targets import (
    rasterize_bev_grid, rasterize_field_grid, resample_from_projection, sdf_on_bev_grid,
)
from navsim.agents.para_ssr.fact.vocab import compose, cumulative_distance, path_headings, resample_path
from navsim.agents.para_ssr.plan_map import rasterize_sdf

PC_RANGE = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)
EXTENT = (-8.0, 72.0, -32.0, 32.0)
RES = 0.25
T = 8


def _vocab(tmp_path, P=12, V=6, S=50):
    """Synthetic vocabulary: curved paths of various curvature, speed profiles of various speeds."""
    s = np.arange(S + 1, dtype=np.float32)
    paths = np.zeros((P, S + 1, 2), np.float32)
    for i, k in enumerate(np.linspace(-0.02, 0.02, P)):
        paths[i, :, 0] = s
        paths[i, :, 1] = 0.5 * k * s ** 2
    speeds = np.stack([np.arange(1, T + 1) * 0.5 * v for v in np.linspace(0.0, 15.0, V)]).astype(np.float32)
    pf, sf = tmp_path / "paths.npy", tmp_path / "speeds.npy"
    np.save(pf, paths)
    np.save(sf, speeds)
    return str(pf), str(sf), paths, speeds


def _cfg():
    return SimpleNamespace(
        plan_map_extent=EXTENT, plan_map_res=RES, plan_map_clip=10.0, pc_range=PC_RANGE, bev_h=10, bev_w=20,
        fut_ts=T, num_navi_cmd=4, fact_w_path=1.0, fact_w_speed=1.0, fact_w_score=1.0, fact_w_list=1.0,
        fact_w_margin=0.5, fact_w_im=1.0, fact_w_offset=1.0, fact_w_extra=1.0, fact_tau_path=1.0,
        fact_tau_speed=2.0, fact_tau_im=1.0, fact_tau_list=0.05, fact_margin=2.0, fact_target_pdms=0.95,
        fact_sdf_weight=1.0, fact_occ_weight=1.0, fact_occ_pos_weight=3.0,
    )


def test_compose_and_resample_roundtrip():
    S = 50
    s = torch.arange(S + 1, dtype=torch.float32)
    paths = torch.stack([s, 0.1 * s], dim=-1).unsqueeze(0)                 # straight line, slope 0.1
    hd = path_headings(paths)
    speeds = torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]])
    poses = compose(paths, hd, speeds, 1.0)                                 # [1, T, 3]
    assert poses.shape == (1, T, 3)
    assert torch.allclose(poses[0, :, 0], speeds[0], atol=1e-5)
    assert torch.allclose(poses[0, :, 2], torch.full((T,), float(np.arctan(0.1))), atol=1e-5)
    st, length = resample_path(poses[..., :2], S, 1.0)
    assert torch.allclose(st[0, :17], paths[0, :17], atol=0.3)              # within the travelled 16 m
    assert length[0] == pytest.approx(16.0, abs=0.1)
    assert torch.allclose(cumulative_distance(poses[..., :2])[0], speeds[0] * np.hypot(1, 0.1) / 1.0, atol=0.05)


def test_rasters_and_route():
    boxes = np.array([[10.0, 0.0, 4.0, 2.0, 0.0]])                           # a car 10 m ahead
    occ = rasterize_field_grid(boxes, EXTENT, RES)
    assert occ.shape == (320, 256) and occ[(10 + 8) * 4, 32 * 4] == 1 and occ[(30 + 8) * 4, 32 * 4] == 0
    bev = rasterize_bev_grid(boxes, PC_RANGE, 50, 100)
    assert bev.shape == (50, 100) and bev[int(10 / 0.64), 50] == 1 and bev[int(30 / 0.64), 50] == 0
    sdf = rasterize_sdf(box(-8, -3, 72, 3), EXTENT, RES, 10.0)
    sdf_bev = sdf_on_bev_grid(sdf, EXTENT, RES, PC_RANGE, 50, 100)
    assert sdf_bev[5, 50] > 2.0 and sdf_bev[5, 20] < -5.0
    poly = np.stack([np.linspace(-5, 100, 50), np.zeros(50)], axis=-1)
    route, valid = resample_from_projection(poly, 128, 1.0)
    assert np.allclose(route[0], [0, 0], atol=1e-6) and valid[:100].all() and not valid[110:].any()


def test_proxy_labels_detect_collision_offroad_and_progress():
    B, N = 1, 3
    cand = torch.zeros(B, N, T, 3)
    t = torch.arange(1, T + 1).float() * 0.5
    cand[0, 0, :, 0] = 5.0 * t                   # drives straight into the car at 20 m
    cand[0, 1, :, 0] = 2.0 * t                   # stops short (8 m)
    cand[0, 2, :, 0] = 2.0 * t
    cand[0, 2, :, 1] = -2.0 * t                  # off the 6 m wide road
    boxes = np.array([[20.0, 0.0, 4.0, 2.0, 0.0]])
    occ = torch.from_numpy(np.stack([rasterize_field_grid(boxes, EXTENT, RES)] * T)).float()[None]
    sdf = torch.from_numpy(rasterize_sdf(box(-8, -3, 72, 3), EXTENT, RES, 10.0))[None]
    poly = np.stack([np.linspace(-5, 100, 50), np.zeros(50)], axis=-1)
    route, valid = resample_from_projection(poly, 128, 1.0)
    targets = {"drivable_sdf": sdf, "fact_occ": occ, "fact_route": torch.from_numpy(route)[None],
               "fact_route_valid": torch.from_numpy(valid)[None], "fact_route_ok": torch.ones(1)}
    lab = proxy_labels(cand, targets, EXTENT, torch.zeros(1))
    assert lab["nc"].tolist() == [[0.0, 1.0, 1.0]]
    assert lab["dac"].tolist() == [[1.0, 1.0, 0.0]]
    assert lab["ttc"][0, 1] == 1.0
    assert lab["ep_raw"][0, 1] == pytest.approx(8.0, abs=0.2)
    assert lab["ep"][0, 1] == pytest.approx(1.0)                 # best safe candidate normalises to 1
    assert 0 <= lab["pdms"].min() and lab["pdms"].max() <= 1
    assert lab["pdms"][0, 1] > lab["pdms"][0, 0] and lab["pdms"][0, 1] > lab["pdms"][0, 2]


def test_planner_forward_backward_and_losses(tmp_path):
    torch.manual_seed(0)
    pf, sf, paths_np, speeds_np = _vocab(tmp_path)
    anchors = np.zeros((5, T, 3), np.float32)
    anchors[:, :, 0] = np.arange(1, T + 1)[None] * np.linspace(0.5, 4.0, 5)[:, None]
    af = tmp_path / "anchors.npy"
    np.save(af, anchors)
    C, H, W, B = 32, 10, 20, 2
    planner = FactorizedPlanner(pf, sf, PC_RANGE, embed_dims=C, fut_ts=T, num_heads=4, num_stations=8,
                                topk_path=4, topk_speed=3, topk=2, fine_layers=1, extra_anchor_file=str(af))
    planner.train()
    bev = torch.randn(B, C, H, W, requires_grad=True)
    ego = torch.randn(B, C)
    status = torch.tensor([[3.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.5, 0.0]])
    sdf = torch.randn(B, 1, H, W)
    occ = torch.rand(B, T, H, W)
    gt_path = torch.tensor([11, 0])
    gt_speed = torch.tensor([5, 0])
    out = planner(bev, ego, status, sdf, occ, use_extra=True, gt_path=gt_path, gt_speed=gt_speed)
    N = 4 * 3
    assert out["trajectory"].shape == (B, T, 3)
    assert out["fact_cand"].shape == (B, N, T, 3)
    assert out["fact_logits"].shape == (B, N, 4) and out["fact_extra_logits"].shape == (B, 5, 4)
    assert (out["fact_path_idx"] == gt_path[:, None]).any(1).all()
    assert (out["fact_speed_idx"] == gt_speed[:, None]).any(1).all()
    assert out["sim_rewards"].shape == (B, 5, N) and out["plan_topk_trajectory"].shape == (B, 2, T, 3)
    # compositions agree with the vocabulary
    p = torch.from_numpy(paths_np[out["fact_path_idx"][0, 0]])
    s = torch.from_numpy(speeds_np[out["fact_speed_idx"][0, 0]])
    exp = compose(p[None], path_headings(p[None]), s[None], 1.0)[0]
    assert torch.allclose(out["fact_cand"][0, 0], exp, atol=1e-5)

    human = torch.zeros(B, T, 3)
    human[:, :, 0] = torch.arange(1, T + 1).float() * 2.0
    occ_field = torch.zeros(B, T, 320, 256)
    sdf_field = torch.from_numpy(rasterize_sdf(box(-8, -3, 72, 3), EXTENT, RES, 10.0))[None].expand(B, -1, -1)
    poly = np.stack([np.linspace(-5, 100, 50), np.zeros(50)], axis=-1)
    route, valid = resample_from_projection(poly, 128, 1.0)
    sim = torch.ones(B, 5, 5)
    sim[:, 2] = torch.linspace(0.2, 1.0, 5)
    targets = {"trajectory": human, "drivable_sdf": sdf_field, "fact_occ": occ_field,
               "fact_route": torch.from_numpy(route)[None].expand(B, -1, -1),
               "fact_route_valid": torch.from_numpy(valid)[None].expand(B, -1), "fact_route_ok": torch.ones(B),
               "sim_reward": sim, "sim_reward_valid": torch.ones(B),
               "fact_sdf_bev": torch.randn(B, H, W).half(), "fact_occ_bev": torch.zeros(B, T, H, W, dtype=torch.uint8)}
    features = {"status_feature": torch.cat([torch.zeros(B, 4), status], dim=1)}
    cfg = _cfg()
    loss, logs = fl.fact_plan_losses(out, targets, features, cfg, planner)
    assert torch.isfinite(loss)
    preds = dict(out, fact_sdf_pred=torch.randn(B, 1, H, W, requires_grad=True), fact_occ_logit=torch.randn(B, T, H, W))
    aux = fl.fact_aux_losses(preds, targets, cfg)
    total = loss + aux["map"] + aux["det"]
    total.backward()
    assert bev.grad is not None and torch.isfinite(bev.grad).all()
    grads = [p.grad for p in planner.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), "every planner parameter must receive a gradient (DDP)"
    for k in ("fact/path", "fact/speed", "fact_proxy/bce", "fact_exact/bce", "fact/offset", "fact/sel_proxy_pdms"):
        assert k in logs and torch.isfinite(logs[k])

    planner.eval()
    with torch.no_grad():
        out_eval = planner(bev, ego, status, sdf, occ)
    assert "fact_extra_logits" not in out_eval and out_eval["trajectory"].shape == (B, T, 3)


def test_route_progress_fallback_without_route():
    cand = torch.zeros(1, 2, T, 3)
    cand[0, 0, :, 0] = torch.arange(1, T + 1).float()
    cand[0, 1, :, 0] = -torch.arange(1, T + 1).float()
    r = torch.zeros(1, 4, 2)
    prog = route_progress(cand, r, torch.zeros(1, 4), torch.zeros(1))
    assert prog.tolist() == [[8.0, 0.0]]
