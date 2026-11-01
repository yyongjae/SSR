"""Tests for navsim/agents/para_ssr/refiner/refiner_net.py (M5; CPU, < 1 min).

Parameter-count parity across arms (only the adapter differs), identical trunk initialisation for the same seed,
identity at initialisation, gate detachment, token-batching consistency, u_d layout.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_refiner_net.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from navsim.agents.para_ssr.refiner import refiner_net as RN  # noqa: E402
from navsim.agents.para_ssr.refiner.corridor import corridor_geometry  # noqa: E402
from navsim.agents.para_ssr.refiner.decoder import decode  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "decoder_navtest_sample.npz"
NORM = (np.full(256, 0.3, np.float32), np.full(256, 0.6, np.float32))


@pytest.fixture(scope="module")
def drafts():
    z = np.load(FIX)
    return torch.as_tensor(z["tau_h"][:12].reshape(3, 4, 8, 3)), torch.as_tensor(z["v0"][:12:4])


def ego(T, v0=None):
    v0 = torch.full((T,), 5.0) if v0 is None else v0
    return v0, torch.zeros(T), torch.stack([v0, torch.zeros(T), torch.zeros(T), torch.zeros(T)], -1), torch.arange(T) % 4


def test_param_count_parity():
    nT = RN.RefinerNet("T", *NORM, seed=0)
    nN = RN.RefinerNet("none", seed=0)
    cT, cN = nT.param_counts(), nN.param_counts()
    assert cT["adapter"] == 41152 and cN["adapter"] == 0
    assert cT["trunk"] == cN["trunk"]
    shapes = lambda n: [(k, tuple(p.shape)) for k, p in n.trunk_named_parameters()]
    assert shapes(nT) == shapes(nN)
    assert 3.3e6 < cT["trunk"] < 3.45e6                          # spec / design: ~3.35 M
    # component sizes of the spec (d = 192)
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(nT.station) == 17 * 128 * 192 + 192
    assert count(nT.draft_mlp) == 69 * 192 + 192 + 192 * 192 + 192
    assert count(nT.global_proj) + nT.global_pos.numel() == 64 * 192 + 192 + 200 * 192
    assert len(nT.layers) == 4 and nT.layers[0].sa.num_heads == 6 and nT.layers[0].ffn[0].out_features == 768
    # buffers (normalisation) are not parameters
    assert not any(n.startswith("adapter.norm") for n, _ in nT.named_parameters())


def test_trunk_init_identical_across_arms():
    a = dict(RN.RefinerNet("T", *NORM, seed=3).trunk_named_parameters())
    b = dict(RN.RefinerNet("none", seed=3).trunk_named_parameters())
    c = dict(RN.RefinerNet("none", seed=4).trunk_named_parameters())
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a if a[k].abs().sum() > 0)
    # construction does not disturb the global RNG
    torch.manual_seed(11)
    x0 = torch.rand(3)
    torch.manual_seed(11)
    RN.RefinerNet("T", *NORM, seed=0)
    assert torch.equal(torch.rand(3), x0)


def test_forward_shapes_and_identity_at_init(drafts):
    tau, v0 = drafts
    T, K = tau.shape[:2]
    torch.manual_seed(0)
    bev = torch.relu(torch.randn(T, 256, 50, 100)).half()
    for arm, b in (("T", bev), ("none", None)):
        net = RN.RefinerNet(arm, *(NORM if arm == "T" else (None, None)), seed=0).eval()
        with torch.no_grad():
            o = net(b, tau, *ego(T, v0))
        assert o["gate_logit"].shape == (T, K) and o["z_lon"].shape == (T, K, 6) and o["w_lat"].shape == (T, K, 6)
        assert o["ud"].shape == (T * K, RN.UD_DIM)
        assert not o["z_lon"].any() and not o["w_lat"].any()          # zero-initialised heads
        dec = decode(tau.reshape(-1, 8, 3), o["z_lon"].reshape(-1, 6), o["w_lat"].reshape(-1, 6))
        assert torch.equal(dec["traj"], tau.reshape(-1, 8, 3))        # identity at init (bitwise)


def test_gate_does_not_reach_trunk(drafts):
    tau, v0 = drafts
    T = tau.shape[0]
    net = RN.RefinerNet("T", *NORM, seed=0)
    bev = torch.relu(torch.randn(T, 256, 50, 100))
    o = net(bev, tau, *ego(T, v0))
    o["gate_logit"].sum().backward()
    for n, p in net.named_parameters():
        has = p.grad is not None and bool(p.grad.abs().sum() > 0)
        assert has == n.startswith("gate_head."), n
    net.zero_grad()
    o = net(bev, tau, *ego(T, v0))
    (o["z_lon"].sum() + o["w_lat"].sum()).backward()
    assert net.adapter.conv1.weight.grad is not None and net.gate_head[0].weight.grad is None


def test_token_batching_consistency(drafts):
    """A draft's outputs depend only on its own token's features: K-grouping and token grouping do not matter."""
    tau, v0 = drafts
    T, K = tau.shape[:2]
    torch.manual_seed(1)
    net = RN.RefinerNet("T", *NORM, seed=1).eval()
    for m in (net.lon_head[-1], net.lat_head[-1]):                     # non-trivial outputs
        torch.nn.init.normal_(m.weight, std=0.1)
    bev = torch.relu(torch.randn(T, 256, 50, 100))
    e = ego(T, v0)
    with torch.no_grad():
        full = net(bev, tau, *e)
        for t in range(T):
            one = net(bev[t:t + 1], tau[t:t + 1], *[x[t:t + 1] for x in e])
            for k in range(K):
                single = net(bev[t:t + 1], tau[t:t + 1, k:k + 1], *[x[t:t + 1] for x in e])
                for key in ("gate_logit", "z_lon", "w_lat"):
                    torch.testing.assert_close(single[key][0, 0], full[key][t, k], atol=2e-5, rtol=1e-4)
            torch.testing.assert_close(one["z_lon"][0], full["z_lon"][t], atol=2e-5, rtol=1e-4)
        # a different token's BEV changes the output (the scene is read)
        swapped = net(bev.flip(0), tau, *e)
        assert not torch.allclose(swapped["z_lon"], full["z_lon"])


def test_ud_layout():
    t = 0.5 * torch.arange(1, 9, dtype=torch.float32)
    tau = torch.stack([6.0 * t, torch.zeros(8), torch.zeros(8)], -1)[None]          # 6 m/s straight
    g = corridor_geometry(tau)
    v0, a0 = torch.tensor([5.0]), torch.tensor([0.4])
    eds = torch.tensor([[5.0, 0.1, 0.4, -0.2]])
    u = RN.draft_features(tau, v0, a0, eds, torch.tensor([2]), g)[0]
    N = {n: i for i, n in enumerate(RN.UD_NAMES)}
    assert len(RN.UD_NAMES) == u.shape[0] == 69
    assert u[N["v0_15"]] == pytest.approx(5 / 15) and u[N["a0_4"]] == pytest.approx(0.1)
    assert u[N["vx_15"]] == pytest.approx(5 / 15) and u[N["vy_15"]] == pytest.approx(0.1 / 15)
    assert u[N["ax_4"]] == pytest.approx(0.1) and u[N["ay_4"]] == pytest.approx(-0.05)
    assert u[6:10].tolist() == [0, 0, 1, 0]
    for k in range(1, 9):
        assert u[N[f"p{k}_x_40"]] == pytest.approx(6.0 * 0.5 * k / 40) and u[N[f"p{k}_cos_h"]] == 1.0
        assert N[f"p{k}_x_40"] == 10 + 4 * (k - 1)
    assert torch.allclose(u[42:50], torch.full((8,), 6 / 15))
    assert u[N["acc0_4"]] == pytest.approx((6 - 5) / 0.5 / 4) and torch.allclose(u[51:58], torch.zeros(7), atol=1e-5)
    assert torch.allclose(u[58:66], torch.zeros(8), atol=1e-5)
    assert u[N["S8_40"]] == pytest.approx(24 / 40) and u[67] == 0 and u[68] == 0
    # unknown command index -> all zeros; NaN ego -> 0
    u2 = RN.draft_features(tau, torch.tensor([float("nan")]), a0, eds, torch.tensor([-1]), g)[0]
    assert u2[6:10].sum() == 0 and u2[0] == 0 and torch.isfinite(u2).all()
