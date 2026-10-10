"""CK core (navsim/agents/para_ssr/ck) unit tests, CPU only: model, correction decode, KD combination, selection.

  keys      : CK_KEYS == anchor_planner.SIM_KEYS order; ck_final == AnchorPlanner.weighted_reward
  trunk     : CKTrunk.forward == RefinerNet.forward bit-identical (arms T, M); student adapter == e2e.build_student
  shapes    : arms S / T / M with K = 16 candidates and K2 = 16 extra candidates (+ lead head)
  identity  : zero-initialised lon / lat heads -> corrected == candidate bit-exact; extra / corr rescoring reuse the
              scene features (same scores for the same trajectories)
  gradients : score head (not detached) reaches the trunk and the adapter; correction KD reaches lon / lat heads
  init      : build_ck from the stage-T4 teacher snapshots (T, M, S-from-T); save_ck / load_ck round trip
  decode    : correct == flat decode; KD space bounds; controls -> KD space -> controls round trip
  kd        : combine_teacher source rule (map None, map not ok); kd_corrected = decode(cand, z_DET, w_MAP)
  select    : select_all on a toy with a known answer
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.ck import constants as C
from navsim.agents.para_ssr.ck import kd as KD
from navsim.agents.para_ssr.ck import losses as L
from navsim.agents.para_ssr.ck import model as M
from navsim.agents.para_ssr.ck import select as S
from navsim.agents.para_ssr.ck.correct import controls_from_kd, correct, kd_space
from navsim.agents.para_ssr.refiner import e2e as E
from navsim.agents.para_ssr.refiner.adapters import save_norm
from navsim.agents.para_ssr.refiner.decoder import decode
from navsim.agents.para_ssr.refiner.refiner_net import RefinerNet

TEACH_T = C.TEACHER_INIT["T"]
TEACH_M = C.TEACHER_INIT["M"]


# ----------------------------------------------------------------------------------------------- fixtures
def _cand(T=2, K=4, v=7.0, seed=0, noise=0.15):
    """Straight-ish candidates at speed v with small seeded perturbations (x fwd, y left, heading)."""
    g = torch.Generator().manual_seed(seed)
    s = v * torch.arange(1, 9, dtype=torch.float32) * 0.5
    base = torch.stack([s, torch.zeros(8), torch.zeros(8)], -1)
    tau = base[None, None].repeat(T, K, 1, 1)
    scale = torch.linspace(0.0, 1.0, 8)[:, None] * torch.tensor([noise, noise, 0.02])
    return (tau + torch.randn(T, K, 8, 3, generator=g) * scale).contiguous()


def _status(T=2, v=7.0):
    sf = torch.zeros(T, 8)
    sf[:, 1] = 1.0
    sf[:, 4] = v
    return sf


def _bev(T=2, seed=1):
    return torch.randn(T, 256, 50, 100, generator=torch.Generator().manual_seed(seed)).half()


def _norm(seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.3, 256).astype(np.float32), rng.uniform(0.5, 2.0, 256).astype(np.float32)


def _randomize_heads(trunk, seed=5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for h in (trunk.lon_head, trunk.lat_head):
            h[-1].weight.copy_(torch.randn(h[-1].weight.shape, generator=g) * 0.05)
            h[-1].bias.copy_(torch.randn(h[-1].bias.shape, generator=g) * 0.1 - 0.2)


# ----------------------------------------------------------------------------------------------- keys / selection rule
def test_ck_keys_follow_v2_sim_keys_and_label_cols():
    from navsim.agents.para_ssr.modules.anchor_planner import SIM_KEYS
    assert tuple(C.CK_KEY_TO_SIM[k] for k in C.CK_KEYS) == tuple(SIM_KEYS)
    assert C.LABEL_COLS[:5] == C.CK_KEYS and C.CK_LABEL_IDX == (0, 1, 2, 3, 4)
    assert C.K_CAND == 16 and C.N_CTRL == 6 and set(C.SPLITS) == {"navtrain_train", "navtrain_val", "navtest"}


def test_ck_final_equals_anchor_planner_weighted_reward():
    from navsim.agents.para_ssr.modules.anchor_planner import AnchorPlanner
    g = torch.Generator().manual_seed(0)
    im = torch.softmax(torch.randn(3, 16, generator=g), -1)
    sim = torch.rand(3, 5, 16, generator=g)                                   # [B, 5, K], SIM_KEYS order
    ref = AnchorPlanner.weighted_reward(SimpleNamespace(reward_weights=C.SEL_W), im, sim)
    got = S.ck_final(sim.permute(0, 2, 1), im)                                # [B, K, 5], CK_KEYS order
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    got_np = S.ck_final(sim.permute(0, 2, 1).numpy().astype(np.float64), im.numpy().astype(np.float64))
    np.testing.assert_allclose(got_np, ref.numpy(), rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------------------------------------- trunk
@pytest.mark.parametrize("arm", ["T", "M"])
def test_cktrunk_bit_identical_to_refinernet(arm):
    norm = _norm()
    ref = (RefinerNet("T", *norm, seed=0) if arm == "T"
           else RefinerNet("M", seed=0, map_norm_mean=norm[0], map_norm_std=norm[1]))
    tr = M.build_trunk(arm, 0, norm)
    assert list(tr.state_dict()) == list(ref.state_dict())
    for a, b in zip(tr.state_dict().values(), ref.state_dict().values()):
        assert torch.equal(a, b)                                               # same init for the same seed
    _randomize_heads(ref)
    tr.load_state_dict(ref.state_dict())
    tau, sf = _cand(2, 3), _status(2)
    v0, a0, eds, cmd = E.ego_inputs(sf)
    bev = _bev(2).float()
    with torch.no_grad():
        o_ref = ref(bev, tau, v0, a0, eds, cmd)
        o_ck = tr(bev, tau, v0, a0, eds, cmd)
    for k in ("gate_logit", "z_lon", "w_lat", "ud"):
        assert torch.equal(o_ref[k], o_ck[k]), k
    assert o_ck["h"].shape == (6, 384)


def test_student_adapter_matches_build_student():
    net = M.CKNet("S", seed=0)
    st = E.build_student(0, 0.1)
    assert list(net.trunk.state_dict()) == list(st.state_dict())
    for a, b in zip(net.trunk.state_dict().values(), st.state_dict().values()):
        assert torch.equal(a, b)
    assert net.trunk.arm == "S" and isinstance(net.trunk.adapter, M.AdapterSGrid)
    # S grid input == bev_embed [T, 5000, 256] (index row * 100 + col) through AdapterS
    emb = torch.randn(2, 5000, 256, generator=torch.Generator().manual_seed(4))
    grid = emb.transpose(1, 2).reshape(2, 256, 50, 100)
    with torch.no_grad():
        assert torch.equal(net.trunk.adapter(grid), st.adapter(emb))


def test_builders_do_not_consume_global_rng_and_share_score_head_init():
    torch.manual_seed(123)
    x0 = torch.rand(3)
    torch.manual_seed(123)
    nets = {a: M.CKNet(a, seed=0, norm=_norm() if a in ("T", "M") else None, lead_aux=(a == "S")) for a in "TMS"}
    assert torch.equal(torch.rand(3), x0)
    for a in "MS":
        for p, q in zip(nets["T"].score_head.parameters(), nets[a].score_head.parameters()):
            assert torch.equal(p, q)
        tk = {k: v for k, v in nets["T"].trunk.state_dict().items() if not k.startswith("adapter.")}
        ok = {k: v for k, v in nets[a].trunk.state_dict().items() if not k.startswith("adapter.")}
        assert list(tk) == list(ok) and all(torch.equal(tk[k], ok[k]) for k in tk)
    assert nets["S"].lead_head is not None and nets["T"].lead_head is None


# ----------------------------------------------------------------------------------------------- shapes / identity
@pytest.mark.parametrize("arm", ["S", "T", "M"])
def test_shapes_k16_extra16(arm):
    T, K = 2, C.K_CAND
    net = M.CKNet(arm, seed=0, norm=_norm() if arm != "S" else None, lead_aux=(arm == "S")).eval()
    cand = _cand(T, K)
    with torch.no_grad():
        o = net(_bev(T), cand, _status(T), extra=_cand(T, K, seed=9), decode=True, slope=0.0)
    assert o["score_logit"].shape == (T, K, 5) and o["score_logit"].dtype == torch.float32
    assert o["z_lon"].shape == (T, K, 6) and o["w_lat"].shape == (T, K, 6)
    assert o["extra_score_logit"].shape == (T, K, 5)
    assert o["corr"]["traj"].shape == (T, K, 8, 3) and o["corr"]["c_lon"].shape == (T, K, 8)
    assert o["corr"]["e_lat"].shape == (T, K, 8) and o["corr"]["raw"]["traj"].shape == (T * K, 8, 3)
    assert len(o["ego"]) == 4 and o["ego"][0].shape == (T,)
    if arm == "S":
        assert o["lead_logit"].shape == (T,)
    assert all(torch.isfinite(o[k]).all() for k in ("score_logit", "z_lon", "w_lat", "extra_score_logit"))


def test_init_identity_and_scene_reuse():
    net = M.CKNet("S", seed=0).eval()
    T, K = 2, 5
    cand = _cand(T, K)
    with torch.no_grad():
        o = net(_bev(T), cand, _status(T), extra=cand.clone(), rescore_corr=True)
        assert torch.equal(o["z_lon"], torch.zeros_like(o["z_lon"])) and torch.equal(o["w_lat"], torch.zeros_like(o["w_lat"]))
        assert torch.equal(o["corr"]["traj"], cand)                                   # identity, bit-exact
        assert torch.equal(o["extra_score_logit"], o["score_logit"])
        assert torch.equal(o["corr_score_logit"], o["score_logit"])
        # candidates are scored independently: a permutation of the candidates permutes the scores
        perm = torch.tensor([3, 0, 4, 1, 2])
        o2 = net(_bev(T), cand[:, perm], _status(T), decode=False)
    torch.testing.assert_close(o2["score_logit"], o["score_logit"][:, perm], rtol=1e-5, atol=1e-5)
    assert "corr" not in o2


# ----------------------------------------------------------------------------------------------- gradients
def test_score_loss_trains_trunk_and_adapter_not_controls():
    net = M.CKNet("S", seed=0)
    T, K = 2, 3
    o = net(_bev(T), _cand(T, K), _status(T), decode=False)
    y = torch.rand(T, K, 5, generator=torch.Generator().manual_seed(2)).round()
    loss, per = L.score_bce(o["score_logit"], y, torch.ones(T, K, dtype=torch.bool))
    loss.backward()
    nz = lambda p: p.grad is not None and bool(torch.count_nonzero(p.grad) > 0)
    assert nz(net.score_head[0].weight) and nz(net.trunk.enc[0].weight) and nz(net.trunk.station.weight)
    assert nz(net.trunk.adapter.conv1.weight) and nz(net.trunk.layers[0].sa.in_proj_weight)
    assert nz(net.trunk.draft_mlp[0].weight) and nz(net.trunk.global_pos)
    assert not nz(net.trunk.lon_head[-1].weight) and not nz(net.trunk.gate_head[-1].weight)
    assert set(per) == set(C.CK_KEYS)


def test_ctrl_kd_reaches_correction_heads_at_identity():
    net = M.CKNet("S", seed=0)
    T, K = 2, 3
    cand = _cand(T, K)
    o = net(_bev(T), cand, _status(T), slope=C.LON_ST_SLOPE["train"])
    ct = -0.5 * torch.ones(T, K, 6)
    et = 0.3 * torch.ones(T, K, 6)
    cs, es = kd_space(o["corr"])
    loss, parts = L.ctrl_kd_l1(cs, es, ct, et, torch.ones(T, K, dtype=torch.bool))
    loss.backward()
    assert parts["lon"] == pytest.approx(0.5) and parts["lat"] == pytest.approx(0.3)
    for h in (net.trunk.lon_head, net.trunk.lat_head):
        assert torch.count_nonzero(h[-1].weight.grad) > 0


# ----------------------------------------------------------------------------------------------- init / io
@pytest.mark.skipif(not (TEACH_T / "ckpt_best.pt").exists(), reason="stage-T4 teacher snapshots not available")
@pytest.mark.parametrize("arm", ["T", "M"])
def test_build_ck_from_v1_teacher_snapshot(arm):
    tr = E.train_refiner_module()
    src = TEACH_T if arm == "T" else TEACH_M
    ref, _ = tr.load_run_model(src, "best", "cpu")
    net = M.build_ck(arm, seed=0, init_from=src).eval()
    rep = net.init_report
    assert rep["kind"] == "v1" and rep["missing"] == [] and rep["unexpected"] == [] and rep["source_arm"] == arm
    assert rep["n_loaded"] == len(ref.state_dict())
    T, K = 2, 3
    tau, sf = _cand(T, K), _status(T)
    bev = (_bev(T).float() * 0.5).abs().half()
    with torch.no_grad():
        o_ref = ref(bev, tau, *E.ego_inputs(sf))
        o = net(bev, tau, sf)
    assert torch.equal(o["z_lon"], o_ref["z_lon"]) and torch.equal(o["w_lat"], o_ref["w_lat"])
    with pytest.raises(ValueError):
        M.build_ck("M" if arm == "T" else "T", seed=0, init_from=src)


@pytest.mark.skipif(not (TEACH_T / "ckpt_best.pt").exists(), reason="stage-T4 teacher snapshots not available")
def test_build_ck_student_from_teacher_trunk_skips_adapter():
    net = M.build_ck("S", seed=0, init_from=TEACH_T)
    sd = torch.load(TEACH_T / "ckpt_best.pt", map_location="cpu", weights_only=False)["model"]
    assert net.init_report["skipped"] == sorted(k for k in sd if k.startswith("adapter."))
    for k, v in net.trunk.state_dict().items():
        if not k.startswith("adapter."):
            assert torch.equal(v, sd[k]), k
    fresh = M.CKNet("S", seed=0)
    for a, b in zip(net.trunk.adapter.state_dict().values(), fresh.trunk.adapter.state_dict().values()):
        assert torch.equal(a, b)


def test_save_load_ck_round_trip(tmp_path):
    norm = _norm()
    net = M.CKNet("T", seed=2, norm=norm, lead_aux=True)
    _randomize_heads(net.trunk)
    net.set_score_prior([0.97, 0.95, 0.8, 0.9, 0.99])
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"arm": "T", "seed": 2, "lead_aux": 1}))
    save_norm(run / "norm.npz", *norm)
    M.save_ck(run / "ckpt_last.pt", net, {"arm": "T"}, epoch=1, step=10)
    assert not list(run.glob(".*tmp*"))
    got, cfg = M.load_ck(run, "last")
    assert cfg["arm"] == "T" and not got.training
    for (k, a), (k2, b) in zip(net.state_dict().items(), got.state_dict().items()):
        assert k == k2 and torch.equal(a, b)
    T, K = 2, 3
    with torch.no_grad():
        o1, o2 = net.eval()(_bev(T), _cand(T, K), _status(T)), got(_bev(T), _cand(T, K), _status(T))
    assert torch.equal(o1["score_logit"], o2["score_logit"]) and torch.equal(o1["lead_logit"], o2["lead_logit"])
    again = M.build_ck("T", seed=2, init_from=run, lead_aux=True)
    assert again.init_report["kind"] == "ck"
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), again.state_dict().values()))
    p = torch.sigmoid(net.score_head[-1].bias)
    torch.testing.assert_close(p, torch.tensor([0.97, 0.95, 0.8, 0.9, 0.99]), rtol=0, atol=1e-6)


# ----------------------------------------------------------------------------------------------- decode
def test_correct_equals_flat_decode_and_kd_space_bounds():
    T, K = 2, 4
    g = torch.Generator().manual_seed(7)
    tau = _cand(T, K, v=8.0)
    z = torch.randn(T, K, 6, generator=g)
    w = torch.randn(T, K, 6, generator=g)
    v0 = torch.full((T,), 8.0)
    out = correct(tau, z, w, v0, slope=0.0)
    ref = decode(tau.reshape(-1, 8, 3), z.reshape(-1, 6), w.reshape(-1, 6), v0=v0.repeat_interleave(K), mode="A")
    assert torch.equal(out["traj"].reshape(-1, 8, 3), ref["traj"])
    assert torch.equal(out["c_lon"].reshape(-1, 8), ref["c_lon"])
    cl, el = kd_space(out)
    assert cl.shape == (T, K, 6) and el.shape == (T, K, 6)
    assert torch.equal(out["c_lon"][..., :2], torch.zeros(T, K, 2)) and torch.equal(out["e_lat"][..., :2], torch.zeros(T, K, 2))
    assert bool((cl <= 0).all()) and bool((el.abs() <= 2.0 + 1e-6).all())
    # the straight-through slope changes the backward only
    out2 = correct(tau, z, w, v0, slope=0.1)
    assert torch.equal(out2["traj"], out["traj"])


def test_decode_round_trip_controls_kd_space_controls():
    """controls -> decoded KD space (c_lon[2:], e_lat[2:]) -> controls -> the same trajectory (no clamp / projection
    active: decelerating z, moderate lateral w on a straight 8 m/s candidate)."""
    T, K = 2, 3
    g = torch.Generator().manual_seed(0)
    tau = _cand(T, K, v=8.0, noise=0.0)
    z = -torch.rand(T, K, 6, generator=g) * 0.5 - 0.05
    w = (torch.rand(T, K, 6, generator=g) - 0.5) * 0.6
    v0 = torch.full((T,), 8.0)
    c = correct(tau, z, w, v0)
    assert bool((c["raw"]["flags"]["alpha"] == 1).all())
    z2, w2 = controls_from_kd(*kd_space(c))
    torch.testing.assert_close(z2, z, rtol=0, atol=1e-5)
    torch.testing.assert_close(w2, w, rtol=0, atol=1e-5)
    c2 = correct(tau, z2, w2, v0)
    torch.testing.assert_close(c2["traj"], c["traj"], rtol=0, atol=1e-4)
    # mode A: corrected candidates are never ahead of the original along the path
    assert bool((c["raw"]["s"] <= c["raw"]["s0"] + 1e-5).all())


# ----------------------------------------------------------------------------------------------- kd combination
def _teacher(seed, N=3, K=4, ok=None):
    g = np.random.default_rng(seed)
    return {"score_logit": g.normal(0, 2, (N, K, 5)).astype(np.float16), "ok": np.ones((N, K), bool) if ok is None else ok,
            "c_lon": -np.abs(g.normal(0, 1, (N, K, 8))).astype(np.float32),
            "e_lat": g.normal(0, 0.5, (N, K, 6)).astype(np.float32)}


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_combine_teacher_source_rule(backend):
    det = _teacher(0)
    mp = _teacher(1)
    mp["ok"][1, 2] = False
    if backend == "torch":
        det = {k: torch.as_tensor(v) for k, v in det.items()}
        mp = {k: torch.as_tensor(v) for k, v in mp.items()}
    sig = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))
    arr = lambda x: x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    out = KD.combine_teacher(det, mp)
    pd, pm = sig(arr(det["score_logit"])), sig(arr(mp["score_logit"]))
    P = arr(out["kd_score_prob"])
    I = {k: i for i, k in enumerate(C.CK_KEYS)}
    okm = arr(mp["ok"])
    np.testing.assert_allclose(P[..., I["nc"]], pd[..., I["nc"]], atol=1e-6)
    np.testing.assert_allclose(P[..., I["ttc"]], pd[..., I["ttc"]], atol=1e-6)
    np.testing.assert_allclose(P[..., I["dac"]], np.where(okm, pm[..., I["dac"]], pd[..., I["dac"]]), atol=1e-6)
    for k in ("ep", "comfort"):
        np.testing.assert_allclose(P[..., I[k]], np.where(okm, 0.5 * (pd[..., I[k]] + pm[..., I[k]]), pd[..., I[k]]),
                                   atol=1e-6)
    np.testing.assert_array_equal(arr(out["kd_c_lon"]), arr(det["c_lon"])[..., 2:])
    np.testing.assert_array_equal(arr(out["kd_e_lat"]), np.where(okm[..., None], arr(mp["e_lat"]), arr(det["e_lat"])))
    np.testing.assert_array_equal(arr(out["kd_ok"]), okm)
    assert isinstance(out["kd_score_prob"], torch.Tensor) == (backend == "torch")
    # no MAP teacher -> DET for everything
    solo = KD.combine_teacher(det, None)
    np.testing.assert_allclose(arr(solo["kd_score_prob"]), pd, atol=1e-6)
    np.testing.assert_array_equal(arr(solo["kd_e_lat"]), arr(det["e_lat"]))
    np.testing.assert_array_equal(arr(solo["kd_ok"]), arr(det["ok"]))
    # scores only (teacher scores on the corrected candidates)
    so = KD.combine_teacher({k: det[k] for k in ("score_logit", "ok")}, {k: mp[k] for k in ("score_logit", "ok")})
    assert set(so) == {"kd_score_prob", "kd_ok"}


def test_kd_corrected_uses_det_lon_and_map_lat():
    T, K = 2, 3
    g = torch.Generator().manual_seed(1)
    cand = _cand(T, K, v=8.0)
    z_det, w_det = -torch.rand(T, K, 6, generator=g), torch.randn(T, K, 6, generator=g) * 0.3
    z_map, w_map = -torch.rand(T, K, 6, generator=g), torch.randn(T, K, 6, generator=g) * 0.3
    v0 = torch.full((T,), 8.0)
    got = KD.kd_corrected(cand, z_det, w_map, v0)
    assert torch.equal(got, correct(cand, z_det, w_map, v0, 0.0)["traj"])
    assert not torch.equal(got, correct(cand, z_map, w_det, v0, 0.0)["traj"])
    assert KD.combine_controls(z_det, None, w_det)[1] is w_det


# ----------------------------------------------------------------------------------------------- selection
def test_select_all_toy():
    N, K = 3, 4
    v2_final = np.tile(np.array([0.0, -1.0, -2.0, -3.0]), (N, 1))           # sorted descending (top-1 = index 0)
    v2_im = np.full((N, K), 1.0 / K)
    prob = np.full((N, K, 5), 0.9)
    prob[:, 0, 0] = 0.05                                                       # candidate 0 collides
    prob[:, 2, :] = 0.99                                                       # candidate 2 best
    pc = prob.copy()
    pc[1, 1, :] = 0.999                                                        # token 1: corrected cand 1 is best
    out = S.select_all(v2_final, v2_im, prob, pc, beta=1.0)
    np.testing.assert_array_equal(out["v2"], 0)
    np.testing.assert_array_equal(out["a"], [2, 2, 2])
    np.testing.assert_array_equal(out["b"], out["a"])
    np.testing.assert_array_equal(out["c"], [2, K + 1, 2])                    # ties keep the original
    assert S.select_all(v2_final, v2_im, prob, pc, beta=0.0)["a"].tolist() == [0, 0, 0]
    assert "c" not in S.select_all(v2_final, v2_im, prob, None, beta=1.0)
    t = S.select_all(torch.as_tensor(v2_final), torch.as_tensor(v2_im), torch.as_tensor(prob), None, beta=0.5)
    assert t["a"].dtype == np.int64
    np.testing.assert_allclose(S.blend(np.ones(2), np.zeros(2), 0.25), [0.75, 0.75])


# ----------------------------------------------------------------------------------------------- ema (same math as E2)
def test_ema_balancer_matches_stage_e():
    cfg = SimpleNamespace(refiner_mode="E2", ref_term_weights=None, kd_balance="ema", kd_draft_source="tau0",
                          kd_ratio=1.0, kd_ema_m=0.99, kd_ema_floor=1e-4, kd_weight_max=10.0,
                          kd_ratio_ramp_epochs=None, kd_start_epoch=0.0, kd_ramp=(0.0, 0.0))
    st = E.StageE(cfg)
    eb = L.EmaBalancer(ratio=1.0, m=0.99, floor=1e-4, cap=10.0, start_step=3)
    rng = np.random.default_rng(0)
    for step in range(20):
        s, k = float(rng.uniform(0.1, 2)), float(rng.uniform(0.01, 0.5))
        if step == 7:
            s = float("nan")                                                   # skipped by both
        st.ema_update(s, k)
        eb.update(s, k)
        want = st.ema_weight() if step >= 3 else 0.0
        assert eb.weight(step) == pytest.approx(want, rel=1e-12, abs=0)
    assert eb.n == 19 and eb.hat() == pytest.approx(st.ema_hat())
    eb2 = L.EmaBalancer(start_step=3)
    eb2.load_state_dict(eb.state_dict())
    assert eb2.weight(30) == eb.weight(30)
    assert L.EmaBalancer(cap=None, start_step=0).weight(0) == 0.0             # before the first update
