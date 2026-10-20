"""km 10/19 (C): fixed LQR port == official PDMSimulator, exact verdicts == official scorer, candidate scorer, selection."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.modules import exact_verdict as EV
from navsim.agents.para_ssr.modules.candidate_scorer import CandidateScorer, cand_scorer_loss, expected_pdms
from navsim.agents.para_ssr.modules.torch_tracker import track

MC = Path("/home/external-user/yongjae/SSR/data/exp/metric_cache")
EXACT = Path(os.environ.get("EXACT_NAVTEST", str(Path(os.environ.get("KM_ROOT", "/home/external-user/ssd/kyungmin_ck")) / "exact" / "navtest")))
TOKENS = Path(__file__).resolve().parents[2] / "resources/km_corridor/packed/navtest/tokens.parquet"
ANCHORS = Path(__file__).resolve().parents[2] / "resources/km_corridor/trajectory_anchors_256.npy"


def _mc_paths(n):
    out = []
    for log in sorted(os.scandir(MC), key=lambda e: e.name):
        u = Path(log.path) / "unknown"
        if u.is_dir():
            for t in sorted(os.scandir(u), key=lambda e: e.name):
                out.append((t.name, Path(t.path) / "metric_cache.pkl"))
                if len(out) >= n:
                    return out
    return out


def _official(mc, traj):
    """official states [K, 41, StateIndex.size()] of ego-frame poses traj [K, 8, 3]"""
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    fs = TrajectorySampling(num_poses=40, interval_length=0.1)
    ies = mc.ego_state
    arr = np.stack([get_trajectory_as_array(transform_trajectory(Trajectory(t.astype(np.float32)), ies), fs, ies.time_point)
                    for t in traj])
    return PDMSimulator(fs).simulate_proposals(arr, ies)


def _init(mc):
    d = mc.ego_state.dynamic_car_state
    return torch.tensor([d.rear_axle_velocity_2d.x, d.rear_axle_velocity_2d.y, d.rear_axle_acceleration_2d.x,
                         d.rear_axle_acceleration_2d.y, 0.0, 0.0, 0.0], dtype=torch.float64)


@pytest.mark.skipif(not MC.exists(), reason="metric cache")
def test_tracker_equals_official_simulator():
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    from tools.corridor.build_corridor_dt01 import load_mc
    anc = np.load(ANCHORS).astype(np.float64)
    rng = np.random.default_rng(0)
    for _, p in _mc_paths(3):
        mc = load_mc(p)
        traj = anc + rng.normal(0, 1.0, anc.shape) * np.array([1.0, 0.5, 0.05])          # off-anchor shapes too
        S = _official(mc, traj)
        ra = mc.ego_state.rear_axle
        c, s = np.cos(ra.heading), np.sin(ra.heading)
        gx, gy = S[..., StateIndex.X] - ra.x, S[..., StateIndex.Y] - ra.y
        st = track(torch.tensor(traj), _init(mc)[None].repeat(len(traj), 1), full=True).numpy()
        err = np.hypot(st[..., 0] - (c * gx + s * gy), st[..., 1] - (-s * gx + c * gy))
        assert err.max() < 1e-3, err.max()


@pytest.mark.skipif(not (MC.exists() and EXACT.exists()), reason="metric cache / exact GT")
def test_exact_verdicts_equal_official_scorer():
    import pandas as pd
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_comfortable
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from tools.corridor.build_corridor_dt01 import load_mc
    toks = pd.read_parquet(TOKENS).token.astype(str).to_numpy()
    row = {t: i for i, t in enumerate(toks)}
    tab = {k: (np.load(EXACT / f"{k}.npy", mmap_mode="r"), np.load(EXACT / f"{k}_off.npy")) for k in EV.TABLES}
    anc = np.load(ANCHORS).astype(np.float64)
    agree = {k: [] for k in ("nc", "dac", "ttc", "comfort")}
    for tok, p in _mc_paths(6):
        if tok not in row:
            continue
        mc = load_mc(p)
        r = row[tok]
        S = _official(mc, anc)
        sc = PDMScorer(TrajectorySampling(num_poses=40, interval_length=0.1))
        sc._reset(S, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
        sc._calculate_ego_area()
        sc._calculate_no_at_fault_collision()
        sc._calculate_drivable_area_compliance()
        sc._calculate_ttc()
        st = track(torch.tensor(anc), _init(mc)[None].repeat(len(anc), 1), full=True).float()[None]
        t = {k: v[None] for k, v in EV.pack_token(*(np.asarray(v[o[r]:o[r + 1]], np.float32) for v, o in tab.values())).items()}
        ex = EV.exact_verdicts(st, t)
        agree["nc"].append((ex["nc"][0].numpy() < 1) == (sc._multi_metrics[MultiMetricIndex.NO_COLLISION] < 1))
        agree["dac"].append(ex["dac"][0].numpy() == sc._multi_metrics[MultiMetricIndex.DRIVABLE_AREA])
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
        agree["ttc"].append(ex["ttc"][0].numpy() == sc._weighted_metrics[WeightedMetricIndex.TTC])
        agree["comfort"].append(ex["comfort"][0].numpy() == ego_is_comfortable(S, np.arange(41) * 0.1).all(-1))
    assert agree["dac"], "no navtest token among the first metric caches"
    for k, v in agree.items():
        rate = np.concatenate(v).mean()
        print(k, rate)
    for k, v in agree.items():
        rate = np.concatenate(v).mean()
        assert rate >= (0.995 if k in ("nc", "ttc") else 0.9999), (k, rate)


def _toy(b=2, k=16, cin=32, bev_h=10, bev_w=10):
    torch.manual_seed(0)
    anc = torch.zeros(k, 8, 3)
    anc[:, :, 0] = torch.linspace(1, 8, 8)[None] * torch.linspace(0.5, 2.0, k)[:, None]
    st = track(anc.double(), torch.zeros(k, 7, dtype=torch.float64), full=True).float()[None].repeat(b, 1, 1, 1)
    det = {"all_cls_scores": torch.randn(6, b, 50, 7), "all_bbox_preds": torch.randn(6, b, 50, 10),
           "traj_preds": torch.randn(b, 50, 6, 8, 2), "traj_cls_preds": torch.randn(b, 50, 6),
           "motion_hidden": torch.randn(b, 50, 6, cin)}
    fields = {"corridor_sdf": torch.full((b, 200, 200), 5.0), "corridor_occ": torch.full((b, 49, 200, 200), -20.0)}
    return st, torch.randn(b, bev_h * bev_w, cin, requires_grad=True), det, fields


def test_scorer_forward_backward_and_loss():
    st, bev, det, fields = _toy()
    m = CandidateScorer(32, 10, 10, 64.0, hid=64, use_fields=True)
    out = m(st, bev, det, fields)
    assert out["cand_nc_logits"].shape == (2, 16, 3) and out["cand_progress"].shape == (2, 16)
    ex = {"nc": torch.ones(2, 16), "dac": torch.ones(2, 16), "ttc": torch.ones(2, 16), "progress": torch.rand(2, 16) * 20}
    ex["nc"][0, 3], ex["dac"][1, 4], ex["ttc"][0, 5] = 0.5, 0.0, 0.0
    loss, logs = cand_scorer_loss(out, ex, torch.ones(2))
    loss.backward()
    assert torch.isfinite(loss) and bev.grad.abs().sum() > 0
    assert all(p.grad is not None for n, p in m.named_parameters()), [n for n, p in m.named_parameters() if p.grad is None]


def test_expected_pdms_perfect_verdict_is_furthest_safe():
    st, *_ = _toy(b=1)
    k = st.shape[1]
    big = 30.0
    prog = torch.linspace(1, 20, k)[None]
    unsafe = torch.zeros(1, k, dtype=torch.bool)
    unsafe[0, -3:] = True                                     # the three furthest fail
    pred = {"cand_nc_logits": torch.tensor([big, -big, -big]).repeat(1, k, 1),
            "cand_dac_logit": torch.where(unsafe, big, -big), "cand_ttc_logit": torch.full((1, k), -big),
            "cand_progress": prog}
    ex = expected_pdms(pred, st)
    assert int(ex["score"].argmax(1)) == k - 4
