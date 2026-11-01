"""Stage E (navsim/agents/para_ssr/refiner/e2e.py) unit tests, CPU only.

  grid        : student bev_embed index r * 100 + c == stage-T S grid cell (r, c) (BEV encoder reference points)
  identity    : an untrained student refiner decodes to tau0 exactly (zero-initialised lon / lat heads)
  kd zero     : L_KD = 0 when student and teacher controls are equal; mean |diff| otherwise
  grad paths  : refiner losses give no gradient to tau0 / the planner; the BEV gradient is scaled by ref_bev_grad_scale;
                every student parameter gets a gradient (DDP-safe); a GT-missing sample is excluded (index-select)
  teachers    : frozen, eval, not registered in the agent (not in state_dict / parameters / optimiser)
  config      : refiner_mode off builds nothing (no ref_student, no extra optimiser group / callback / targets)
  perturbation: deterministic for a seed, invalid perturbations keep tau0
  E0 parity   : tools/refiner/stageE_parity.py check (golden taken before the first code edit), if the golden exists
"""
from __future__ import annotations

import math
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(_REPO / "data/dataset/maps"))   # read by navsim at import time
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(_REPO / "data/dataset"))
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.refiner import e2e as E
from navsim.agents.para_ssr.refiner.adapters import s_grid_cell
from navsim.agents.para_ssr.refiner.decoder import decode

DATA = Path("/home/external-user/ssd/yongjae_refiner")
GT_TOKENS = ("1aa44d46e4ab5bc7", "153c6b07f09d53d1")
TEACHER_RUN = DATA / "runs/stageT3_T_fold0_seed0"          # same config / ckpt format as the stage-T4 teachers


def _cfg(**kw):
    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    c = ParaSSRConfig(backbone_pretrained=False)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _tau(B=2, v=8.0, curv=0.0):
    s = v * np.arange(1, 9) * 0.5
    h = curv * s
    if curv == 0.0:
        x, y = s, np.zeros_like(s)
    else:
        x, y = np.sin(h) / curv, (1 - np.cos(h)) / curv
    tau = torch.as_tensor(np.stack([x, y, h], -1), dtype=torch.float32)
    return tau[None].repeat(B, 1, 1)


def _status(B=2, v=8.0):
    sf = torch.zeros(B, 8)
    sf[:, 1] = 1.0            # straight
    sf[:, 4] = v
    return sf


# ----------------------------------------------------------------------------------------------- grid
def test_student_bev_index_is_s_grid():
    from navsim.agents.para_ssr.modules.bevformer import BEVFormerEncoder

    ref = BEVFormerEncoder.get_reference_points(50, 100, dim="2d", bs=1, device="cpu")[0, :, 0]       # [5000, 2] (x_n, y_n)
    cfg = _cfg()
    pc = cfg.pc_range
    x_right = ref[:, 0] * (pc[3] - pc[0]) + pc[0]
    y_fwd = ref[:, 1] * (pc[4] - pc[1]) + pc[1]
    r, c = s_grid_cell(y_fwd.numpy(), -x_right.numpy())         # N frame: x = forward, y_left = -x_right
    q = np.arange(5000)
    np.testing.assert_allclose(r, q // 100, atol=1e-4)
    np.testing.assert_allclose(c, q % 100, atol=1e-4)


# ----------------------------------------------------------------------------------------------- identity
@pytest.mark.parametrize("curv", [0.0, 0.02])
def test_student_identity_at_init(curv):
    net = E.build_student(0, 0.1).eval()
    B = 2
    bev = torch.randn(B, 5000, 256)
    tau0 = _tau(B, 7.0, curv)
    v0, a0, eds, cmd = E.ego_inputs(_status(B, 7.0))
    with torch.no_grad():
        out = net(bev, tau0[:, None], v0, a0, eds, cmd)
    assert torch.count_nonzero(out["z_lon"]) == 0 and torch.count_nonzero(out["w_lat"]) == 0
    dec = decode(tau0, out["z_lon"][:, 0], out["w_lat"][:, 0], v0=v0, mode="A")
    assert torch.equal(dec["traj"], tau0) or torch.allclose(dec["traj"], tau0, atol=1e-5, rtol=0)


def test_build_student_uses_no_global_rng_and_is_seeded():
    torch.manual_seed(123)
    s0 = torch.random.get_rng_state().clone()
    a = E.build_student(0, 0.1)
    assert torch.equal(torch.random.get_rng_state(), s0)
    b = E.build_student(0, 0.1)
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p, q), n
    assert isinstance(a.adapter, E.AdapterS)


# ----------------------------------------------------------------------------------------------- KD
def test_kd_loss_zero_when_equal_and_l1_otherwise():
    z, w = torch.randn(3, 1, 6), torch.randn(3, 1, 6)
    s = {"z_lon": z, "w_lat": w}
    cs = E.kd_controls(s, None, "raw")
    ok = torch.ones(3, dtype=torch.bool)
    l, per = E.kd_loss(cs, [cs.clone(), cs.clone()], [ok, ok])
    assert float(l) == 0.0 and all(float(p) == 0.0 for p in per)
    t = E.kd_controls({"z_lon": z + 0.5, "w_lat": w - 0.25}, None, "raw")
    l, per = E.kd_loss(cs, [t, cs.clone()], [ok, ok])
    assert math.isclose(float(per[0]), 0.375, rel_tol=1e-6)
    assert math.isclose(float(l), 0.1875, rel_tol=1e-6)
    ok0 = torch.tensor([True, False, False])
    t2 = cs.clone()
    t2[1:, :6] += 10.0                                             # masked samples do not count
    l, _ = E.kd_loss(cs, [t2], [ok0])
    assert float(l) == 0.0


def test_kd_spaces_bounded():
    """tanh / decoded controls stay bounded for saturated teacher z_lon (+10): no pull of the student to z = +10."""
    B = 2
    tau = _tau(B, 8.0)
    zs = {"z_lon": torch.full((B, 1, 6), 10.0), "w_lat": torch.zeros(B, 1, 6)}
    z0 = {"z_lon": torch.full((B, 1, 6), 1.0, requires_grad=True), "w_lat": torch.zeros(B, 1, 6, requires_grad=True)}
    dec = lambda o: decode(tau, o["z_lon"][:, 0], o["w_lat"][:, 0], v0=torch.full((B,), 8.0), mode="A")
    ok = torch.ones(B, dtype=torch.bool)
    assert E.kd_controls(zs, None, "tanh").abs().max() <= 1.0
    ct = E.kd_controls(zs, dec(zs), "decoded")
    assert ct.shape == (B, 12) and float(ct.abs().max()) == 0.0     # a non-live teacher decodes to c = 0, e = 0
    l, _ = E.kd_loss(E.kd_controls(z0, dec(z0), "decoded"), [ct], [ok])
    assert float(l) == 0.0                                          # a non-live student already matches it
    with pytest.raises(ValueError):
        E.kd_controls(zs, None, "")


def test_e2_training_requires_kd_space():
    st, _ = _stage("E2", kd_teacher_runs=(str(TEACHER_RUN),), kd_space="")
    with pytest.raises(ValueError, match="kd_space"):
        st.teachers(torch.device("cpu"))


def test_gt_loader_finds_dev_object_tokens():
    """objects/dev holds the 6,634 E2E train_logs tokens that were stage-T dev tokens (reviewer finding 1)."""
    tok = "32b4934cb70c50a7"
    if not (DATA / "objects/dev" / f"{tok}.npz").exists():
        pytest.skip("objects/dev not available")
    ld = E.GTLoader(DATA)
    from navsim.agents.para_ssr.refiner import data as RD
    assert RD._read_objects(ld.src, tok) is not None
    if (DATA / "sdf/navtrain" / f"{tok}.npz").exists() and (DATA / E.SIDE_DIR / tok[:2] / f"{tok}.npz").exists():
        assert bool(ld.load(tok)["ref_gt_ok"])


def test_nonfinite_surrogate_term_is_dropped(monkeypatch):
    B = 2
    tg = _gt_targets(GT_TOKENS, B)
    st, _ = _stage("E1", ref_perturb_frac=0.0)
    net = E.build_student(0, 0.1)
    tr = E.train_refiner_module()
    orig = tr.surrogate_terms_batch

    def bad(*a, **k):
        t = dict(orig(*a, **k))
        t["col"] = t["col"].clone()
        t["col"][0] = float("nan")
        return t
    monkeypatch.setattr(tr, "surrogate_terms_batch", bad)
    loss, logs = st.loss(net, {"status_feature": _status(B, 6.0)}, tg, _Toy(B)(_tau(B, 6.0)), iteration=0)
    assert float(logs["ref/n_nonfinite"]) == 1.0 and float(logs["ref/L_sur"]) == 0.0 and torch.isfinite(loss)


# ----------------------------------------------------------------------------------------------- gradient paths
def _gt_targets(tokens, B):
    ld = E.GTLoader(DATA)
    items = [ld.load(t) for t in tokens]
    if not all(bool(it["ref_gt_ok"]) for it in items):
        pytest.skip("GT for the test tokens not available")
    tg = {k: torch.stack([it[k] for it in items]) for k in items[0]}
    tg["trajectory"] = _tau(B, 6.0)
    return tg


class _Toy(torch.nn.Module):
    """Stand-in for PARA-SSR: a planner parameter producing tau0 and an encoder parameter producing bev_embed."""

    def __init__(self, B):
        super().__init__()
        self.enc = torch.nn.Parameter(torch.randn(B, 5000, 256) * 0.5)
        self.plan = torch.nn.Parameter(torch.zeros(B, 8, 3))

    def forward(self, tau):
        return {"bev_embed": self.enc * 1.0, "trajectory": tau + self.plan}


def _stage(mode="E1", **kw):
    cfg = _cfg(refiner_mode=mode, **kw)
    return E.StageE(cfg), cfg


@pytest.mark.parametrize("perturb", [0.0, 1.0])
def test_gradient_paths(perturb):
    B = 2
    tg = _gt_targets(GT_TOKENS, B)
    feats = {"status_feature": _status(B, 6.0)}
    grads = {}
    for scale in (0.1, 1.0):
        st, cfg = _stage("E1", ref_bev_grad_scale=scale, ref_perturb_frac=perturb)
        net = E.build_student(0, scale)
        # make the (zero-initialised) heads non-trivial so the surrogate has a gradient
        with torch.no_grad():
            for h in (net.lon_head, net.lat_head):
                h[-1].weight.normal_(0, 0.05, generator=torch.Generator().manual_seed(1))
                h[-1].bias.fill_(-0.3)
        toy = _Toy(B)
        with torch.no_grad():
            toy.enc.copy_(torch.randn(B, 5000, 256, generator=torch.Generator().manual_seed(2)))
        preds = toy(_tau(B, 6.0))
        loss, logs = st.loss(net, feats, tg, preds, iteration=7, log_bev_grad=True)
        assert torch.isfinite(loss) and float(logs["ref/n_gt_ok"]) == B
        loss.backward()
        assert toy.plan.grad is None or torch.count_nonzero(toy.plan.grad) == 0      # tau0 is stop-grad
        missing = [n for n, p in net.named_parameters() if p.grad is None]
        assert not missing, missing                                                   # DDP: every param in graph
        grads[scale] = toy.enc.grad.clone()
    assert torch.count_nonzero(grads[1.0]) > 0
    torch.testing.assert_close(grads[0.1], 0.1 * grads[1.0], rtol=1e-4, atol=1e-10)


def test_gt_missing_sample_is_excluded():
    B = 2
    tg = _gt_targets(GT_TOKENS, B)
    feats = {"status_feature": _status(B, 6.0)}
    net = E.build_student(0, 0.1)
    st, _ = _stage("E1", ref_perturb_frac=0.0)
    toy = _Toy(B)
    l_both, _ = st.loss(net, feats, tg, toy(_tau(B, 6.0)), iteration=0)
    tg1 = dict(tg)
    tg1["ref_gt_ok"] = torch.tensor([True, False])
    for k in tg1:                                   # garbage in the missing sample must not matter
        if k.startswith("ref_") and k != "ref_gt_ok" and tg1[k].is_floating_point():
            tg1[k] = tg1[k].clone()
            tg1[k][1] = float("nan")
    l1, logs1 = st.loss(net, feats, tg1, toy(_tau(B, 6.0)), iteration=0)
    assert float(logs1["ref/n_gt_missing"]) == 1.0 and torch.isfinite(l1)
    tg0 = {k: (v[:1] if isinstance(v, torch.Tensor) and v.dim() > 0 else v) for k, v in tg.items()}
    toy0 = _Toy(1)
    l_single, _ = st.loss(net, {"status_feature": _status(1, 6.0)}, tg0, toy0(_tau(1, 6.0)), iteration=0)
    torch.testing.assert_close(l1, l_single)
    assert torch.isfinite(l_both)


# ----------------------------------------------------------------------------------------------- teachers
@pytest.mark.skipif(not (TEACHER_RUN / "ckpt_best.pt").exists(), reason="stage-T teacher run not available")
@pytest.mark.parametrize("space", ["raw", "decoded"])
def test_teachers_frozen_and_not_registered(space):
    B = 2
    st, cfg = _stage("E2", kd_teacher_runs=(str(TEACHER_RUN),), kd_lambda=1.0, kd_ramp=(0.0, 0.0),
                     ref_perturb_frac=0.0, kd_space=space)
    tg = {"ref_gt_ok": torch.zeros(B, dtype=torch.bool), "trajectory": _tau(B),
          E.KD_BEV_KEY.format(0): torch.randn(B, 256, 50, 100).half(), E.KD_OK_KEY.format(0): torch.ones(B, dtype=torch.bool)}
    net = E.build_student(0, 0.1)
    toy = _Toy(B)
    loss, logs = st.loss(net, {"status_feature": _status(B)}, tg, toy(_tau(B)), iteration=0)
    ts = st.teachers(torch.device("cpu"))
    assert len(ts) == 1 and not ts[0].training
    assert all(not p.requires_grad for p in ts[0].parameters())
    loss.backward()
    assert all(p.grad is None for p in ts[0].parameters())
    assert float(logs["kd/lambda"]) == 1.0 and (float(logs["kd/loss"]) > 0 if space == "raw" else
                                                 math.isfinite(float(logs["kd/loss"])))
    assert all(p.grad is not None for p in net.parameters())
    assert not isinstance(st, torch.nn.Module)                  # the holder is not a module -> not in state_dict


def test_kd_lambda_schedule():
    st, _ = _stage("E2", kd_teacher_runs=("x",), kd_lambda=2.0, kd_ramp=(5.0, 10.0))
    for ep, lam in ((0.0, 0.0), (4.99, 0.0), (5.0, 0.0), (7.5, 1.0), (10.0, 2.0), (29.0, 2.0)):
        st.epoch_frac = ep
        assert math.isclose(st.kd_lambda(), lam, abs_tol=1e-9), (ep, st.kd_lambda())


# ----------------------------------------------------------------------------------------------- agent wiring
def _agent(**kw):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
    cfg = _cfg(**kw)
    return ParaSSRAgent(cfg, TrajectorySampling(time_horizon=4, interval_length=0.5))


def test_off_builds_nothing_and_on_registers_student():
    torch.manual_seed(0)
    a0 = _agent()
    s_off = torch.random.get_rng_state().clone()
    assert not hasattr(a0, "ref_student") and a0._stage_e is None
    assert len(a0.get_training_callbacks()) == 1
    assert len(a0.get_optimizers()["optimizer"].param_groups) == 2
    torch.manual_seed(0)
    a1 = _agent(refiner_mode="E1")
    assert torch.equal(torch.random.get_rng_state(), s_off)       # E1 construction leaves the global RNG as E0's
    assert any(k.startswith("ref_student.") for k in a1.state_dict())
    for (n, p), (m, q) in zip(a0.para_ssr_model.named_parameters(), a1.para_ssr_model.named_parameters()):
        assert n == m and torch.equal(p, q), n                        # same PARA-SSR initialisation (seeded builds)
    g = a1.get_optimizers()["optimizer"].param_groups
    assert len(g) == 3 and g[2]["lr_scale"] == 3.0 and g[2]["weight_decay"] == 0.01
    assert len(a1.get_training_callbacks()) == 2


def test_hydra_yaml_defaults_equal_dataclass():
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    y = OmegaConf.load(Path(__file__).resolve().parents[3] / "navsim/planning/script/config/common/agent/para_ssr_agent.yaml")
    cfg = instantiate(y.config)
    ref = _cfg()
    for k in ("refiner_mode", "ref_bev_grad_scale", "ref_w", "ref_m_col", "ref_m_dac", "ref_m_ttc", "ref_lon_st_slope",
              "ref_perturb_frac", "ref_seed", "ref_lr_mult", "ref_weight_decay", "ref_clip", "kd_lambda",
              "ref_eval_traj", "ref_data_root", "kd_space", "kd_balance", "kd_ratio", "kd_ema_m", "kd_ema_floor",
              "kd_start_epoch", "kd_draft_source", "grad_share_every", "ref_human_only_until", "kd_ratio_ramp_epochs",
              "kd_weight_max"):
        assert getattr(cfg, k) == getattr(ref, k), k
    assert dict(cfg.ref_term_weights) == dict(ref.ref_term_weights)
    assert tuple(cfg.kd_ramp) == tuple(ref.kd_ramp) and tuple(cfg.kd_teacher_runs) == ()


# ----------------------------------------------------------------------------------------------- perturbation
def test_perturbation_deterministic_and_counts():
    tau0 = torch.cat([_tau(3, 8.0), _tau(3, 5.0, 0.02)])
    v0 = torch.tensor([8.0] * 3 + [5.0] * 3)
    a0 = torch.zeros(6)
    r1 = E.perturb_batch(tau0, v0, a0, 1.0, np.random.default_rng([0, 5, 0]))
    r2 = E.perturb_batch(tau0, v0, a0, 1.0, np.random.default_rng([0, 5, 0]))
    assert torch.equal(r1[0], r2[0]) and torch.equal(r1[1], r2[1])
    assert bool((r1[1] | r1[2]).all())                                   # frac 1: every sample tried
    changed = (r1[0] - tau0).abs().amax((1, 2)) > 0
    assert torch.equal(changed | (r1[1] & ~changed), r1[1]) and not bool(changed[r1[2]].any())
    r0 = E.perturb_batch(tau0, v0, a0, 0.0, np.random.default_rng(0))
    assert torch.equal(r0[0], tau0) and not bool(r0[1].any())
    ext = E.extend_straight(_tau(1, 8.0)[0].numpy())
    assert ext.shape == (16, 3) and np.allclose(np.diff(ext[7:, 0]), 4.0, atol=1e-5)


# ----------------------------------------------------------------------------------------------- E0 parity
@pytest.mark.skipif(not Path("/home/external-user/ssd/yongjae_refiner/stageE/parity_golden.pt").exists(),
                    reason="golden dump not available")
def test_e0_parity_off_bit_identical():
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "stageE_parity.py"
    spec = importlib.util.spec_from_file_location("stageE_parity", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.check(m.GOLDEN) == []


def test_callback_skips_nonfinite_optimizer_step():
    st, cfg = _stage("E1")
    net = E.build_student(0, 0.1)
    agent = SimpleNamespace(config=cfg, _stage_e=st, ref_student=net)

    class _PL(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net, self.other = net, torch.nn.Linear(2, 2)
            self.logged = {}

        def log(self, k, v, **kw):
            self.logged[k] = v
    pl = _PL()
    cb = E.make_callback(agent)
    tr = SimpleNamespace(global_step=3, is_global_zero=True)
    for p in pl.parameters():
        p.grad = torch.ones_like(p)
    cb.on_before_optimizer_step(tr, pl, None)                     # finite: refiner clipped, nothing skipped
    assert st.skipped_steps == 0 and all(p.grad is not None for p in pl.parameters())
    assert "train/ref/grad_norm_preclip" in pl.logged
    pl.other.weight.grad[0, 0] = float("nan")
    cb.on_before_optimizer_step(tr, pl, None)
    assert st.skipped_steps == 1 and all(p.grad is None for p in pl.parameters())


# ----------------------------------------------------------------------------------------------- post-pilot options
@pytest.mark.skipif(not Path("/home/external-user/ssd/yongjae_refiner/stageE/parity_golden_e12.pt").exists(),
                    reason="E1/E2 golden not available")
def test_e1_e2_defaults_bit_identical_to_pre_option_code():
    """kd_balance / kd_draft_source / grad_share_every at their defaults: E1 and E2 (fixed) losses, logs and every
    parameter gradient torch.equal to the golden taken before the options existed (real CPU batch, 2 steps)."""
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "stageE_parity_e12.py"
    spec = importlib.util.spec_from_file_location("stageE_parity_e12", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    n = torch.get_num_threads()
    try:
        assert m.check() == []
    finally:
        torch.set_num_threads(n)


def test_ema_weight_math_synthetic():
    st, cfg = _stage("E2", kd_teacher_runs=("x",), kd_balance="ema", kd_ratio=1.0, kd_ema_m=0.99, kd_ema_floor=1e-4)
    rng = np.random.default_rng(0)
    sur = 0.6 + 0.2 * rng.standard_normal(400)
    kd = 0.03 + 0.01 * rng.standard_normal(400)
    es = ek = 0.0
    for n, (s, k) in enumerate(zip(sur, kd), 1):
        st.ema_update(s, k)
        es, ek = 0.99 * es + 0.01 * s, 0.99 * ek + 0.01 * k
        c = 1 - 0.99 ** n
        assert math.isclose(st.ema_weight(), (es / c) / max(ek / c, 1e-4), rel_tol=1e-12)
    # bias correction: the first weight is exactly sur_1 / kd_1
    st1, _ = _stage("E2", kd_teacher_runs=("x",), kd_balance="ema")
    assert st1.ema_weight() == 0.0
    st1.ema_update(0.5, 0.02)
    assert math.isclose(st1.ema_weight(), 25.0, rel_tol=1e-12)
    # constant inputs -> weighted KD equals the surrogate exactly (ratio r scales it)
    st2, _ = _stage("E2", kd_teacher_runs=("x",), kd_balance="ema", kd_ratio=0.5)
    for _ in range(50):
        st2.ema_update(0.64, 0.032)
    assert math.isclose(st2.ema_weight() * 0.032, 0.5 * 0.64, rel_tol=1e-9)
    # floor on the denominator, non-finite values skip the update
    st3, _ = _stage("E2", kd_teacher_runs=("x",), kd_balance="ema", kd_ema_floor=1e-4)
    st3.ema_update(0.6, 0.0)
    assert math.isclose(st3.ema_weight(), 0.6 / 1e-4, rel_tol=1e-12)
    st3.ema_update(float("nan"), 0.1)
    assert st3.ema["n"] == 1
    # start epoch: None -> kd_ramp[0]
    assert st3.kd_start() == 5.0
    st4, _ = _stage("E2", kd_teacher_runs=("x",), kd_balance="ema", kd_start_epoch=0.0)
    assert st4.kd_start() == 0.0
    with pytest.raises(ValueError):
        _stage("E2", kd_balance="bogus")
    with pytest.raises(ValueError):
        _stage("E2", kd_draft_source="bogus")


def test_ema_state_saved_and_restored_through_lightning_callback_state():
    from pytorch_lightning.trainer import call
    import pytorch_lightning as pl

    def mk():
        st, cfg = _stage("E2", kd_teacher_runs=("x",), kd_balance="ema")
        agent = SimpleNamespace(config=cfg, _stage_e=st, ref_student=None)
        return st, E.make_callback(agent)
    a, cba = mk()
    for s, k in ((0.7, 0.03), (0.5, 0.05), (0.9, 0.02)):
        a.ema_update(s, k)
    tr = pl.Trainer(callbacks=[cba], logger=False, enable_checkpointing=False, accelerator="cpu",
                    enable_progress_bar=False, enable_model_summary=False)
    ckpt = {"callbacks": call._call_callbacks_state_dict(tr)}
    assert ckpt["callbacks"][cba.state_key] == a.ema_state()
    b, cbb = mk()
    tr2 = pl.Trainer(callbacks=[cbb], logger=False, enable_checkpointing=False, accelerator="cpu",
                     enable_progress_bar=False, enable_model_summary=False)
    call._call_callbacks_load_state_dict(tr2, ckpt)
    assert b.ema == a.ema
    a.ema_update(0.4, 0.04)
    b.ema_update(0.4, 0.04)
    assert a.ema_weight() == b.ema_weight()
    # fixed (default): nothing is stored, as before
    st, cfg = _stage("E2", kd_teacher_runs=("x",))
    cb = E.make_callback(SimpleNamespace(config=cfg, _stage_e=st, ref_student=None))
    tr3 = pl.Trainer(callbacks=[cb], logger=False, enable_checkpointing=False, accelerator="cpu",
                     enable_progress_bar=False, enable_model_summary=False)
    assert call._call_callbacks_state_dict(tr3) == {}


def _human(B=2):
    t = _tau(B, 5.0, 0.02)
    t[..., 1] += 0.05 * torch.arange(1, 9, dtype=torch.float32)       # differs from tau0
    return t


@pytest.mark.skipif(not (TEACHER_RUN / "ckpt_best.pt").exists(), reason="stage-T teacher run not available")
def test_human_mix_drafts_valid_and_shared_by_student_and_teachers(monkeypatch):
    B = 6
    tau0 = torch.cat([_tau(3, 6.0), _tau(3, 6.0, 0.01)])
    human = torch.cat([_human(3), _human(3)])
    tg = {"ref_gt_ok": torch.zeros(B, dtype=torch.bool), "trajectory": human,
          E.KD_BEV_KEY.format(0): torch.randn(B, 256, 50, 100).half(), E.KD_OK_KEY.format(0): torch.ones(B, dtype=torch.bool)}
    seen = {}
    orig = E.StageE._run

    def rec(net, bev, tau_in, ego):
        seen.setdefault(arm, []).append(tau_in.detach().clone())
        return orig(net, bev, tau_in, ego)
    monkeypatch.setattr(E.StageE, "_run", staticmethod(rec))
    net = E.build_student(0, 0.1)
    toy = _Toy(B)
    logs = {}
    for arm, mode in (("E2", "E2"), ("E1", "E1")):
        st, _ = _stage(mode, kd_teacher_runs=(str(TEACHER_RUN),), kd_lambda=1.0, kd_ramp=(0.0, 0.0),
                       kd_space="decoded", ref_perturb_frac=0.5, kd_draft_source="human_mix")
        _, logs[arm] = st.loss(net, {"status_feature": _status(B, 6.0)}, tg, toy(tau0), iteration=11)
    s_in, t_in = seen["E2"]                                     # student then teacher
    assert torch.equal(s_in, t_in) and torch.equal(seen["E1"][0], s_in)    # same drafts: student / teacher / E1
    exp, pert, inval = E.perturb_batch(tau0, *E.ego_inputs(_status(B, 6.0))[:2], 0.5,
                                       np.random.default_rng([0, 11, 0]), src=human)
    assert torch.equal(exp, s_in) and bool(torch.isfinite(s_in).all())
    assert float(logs["E2"]["ref/n_human"]) == float(pert.sum()) and float(logs["E2"]["ref/n_human_invalid"]) == float(inval.sum())
    assert math.isclose(float(logs["E2"]["ref/frac_human"]), float(pert.sum()) / B, rel_tol=1e-6)
    keep = ~pert
    assert torch.equal(s_in[keep], tau0[keep])                  # unselected / invalid samples feed sg(tau0)
    assert int(pert.sum()) >= 1
    # a perturbed human draft starts where the human starts (continuity of the bank generator) and is not tau0
    for i in torch.nonzero(pert)[:, 0].tolist():
        assert not torch.equal(s_in[i], tau0[i])
    # non-finite human trajectory -> invalid -> sg(tau0)
    bad = human.clone()
    bad[:] = float("nan")
    r = E.perturb_batch(tau0, *E.ego_inputs(_status(B, 6.0))[:2], 1.0, np.random.default_rng(0), src=bad)
    assert torch.equal(r[0], tau0) and bool(r[2].all()) and not bool(r[1].any())


@pytest.mark.skipif(not (TEACHER_RUN / "ckpt_best.pt").exists(), reason="stage-T teacher run not available")
@pytest.mark.parametrize("balance", ["fixed", "ema"])
def test_grad_share_logging_leaves_gradients_bit_identical(balance):
    B = 2
    tg = _gt_targets(GT_TOKENS, B)
    tg[E.KD_BEV_KEY.format(0)] = torch.randn(B, 256, 50, 100, generator=torch.Generator().manual_seed(3)).half()
    tg[E.KD_OK_KEY.format(0)] = torch.ones(B, dtype=torch.bool)
    feats = {"status_feature": _status(B, 6.0)}
    res = {}
    for every in (0, 1):
        st, _ = _stage("E2", kd_teacher_runs=(str(TEACHER_RUN),), kd_lambda=5.0, kd_ramp=(0.0, 0.0), kd_space="decoded",
                       ref_perturb_frac=0.0, grad_share_every=every, kd_balance=balance, kd_start_epoch=0.0)
        net = E.build_student(0, 0.1)
        with torch.no_grad():
            for h in (net.lon_head, net.lat_head):
                h[-1].weight.normal_(0, 0.05, generator=torch.Generator().manual_seed(1))
                h[-1].bias.fill_(-0.3)
        toy = _Toy(B)
        with torch.no_grad():
            toy.enc.copy_(torch.randn(B, 5000, 256, generator=torch.Generator().manual_seed(2)))
        preds = toy(_tau(B, 6.0))
        e0 = (preds["bev_embed"] ** 2).mean() * 3.0 + preds["trajectory"].abs().mean()
        loss, logs = st.loss(net, feats, tg, preds, iteration=0, e0_loss=e0)
        (e0 + loss).backward()
        res[every] = (float(loss), {n: p.grad.clone() for n, p in net.named_parameters()},
                      toy.enc.grad.clone(), toy.plan.grad.clone(), logs)
    assert res[0][0] == res[1][0]
    for n in res[0][1]:
        assert torch.equal(res[0][1][n], res[1][1][n]), n
    assert torch.equal(res[0][2], res[1][2]) and torch.equal(res[0][3], res[1][3])
    lg = res[1][4]
    assert "gnorm/bev_e0" not in res[0][4] and "ref/vshare_e0" in res[0][4]
    assert all(float(lg[f"gnorm/bev_{k}"]) > 0 for k in ("e0", "sur", "kd"))
    assert math.isclose(sum(float(lg[f"ref/gshare_{k}"]) for k in ("e0", "sur", "kd")), 1.0, rel_tol=1e-5)
    assert math.isclose(sum(float(lg[f"ref/vshare_{k}"]) for k in ("e0", "sur", "kd")), 1.0, rel_tol=1e-5)
    if balance == "ema":             # first step: weighted KD == weighted surrogate (r = 1)
        assert math.isclose(float(lg["kd/weighted"]), float(lg["ref/L_sur_weighted"]), rel_tol=1e-5)


HUMAN_E2E = DATA / E.HUMAN_NPZ


@pytest.mark.skipif(not (HUMAN_E2E.exists() and (DATA / "human/train.npz").exists()), reason="human npz not available")
def test_human_mix_uses_logged_path_bitwise_as_draft_bank(monkeypatch):
    """perturb_one with the logged path (GTLoader human_path=True) == decoder.sample_bank (make_draft_bank's generator,
    same HumanContext(tau_h, path, n_reg, v0, a0) and fallbacks) for the same family and rng stream; the stage-E human
    npz agrees with the stage-T bank source human/train.npz; default GTLoader adds no keys."""
    from navsim.agents.para_ssr.refiner import decoder as D

    H = np.load(DATA / "human/train.npz", allow_pickle=False)
    gl = E.GTLoader(DATA, human_path=True)
    gl._human_path("x")
    both = [t for t in H["tokens"].tolist() if t in gl._human[0]]          # stage-T train tokens of the E2E train set
    assert len(both) > 15000
    toks = np.asarray(both[:: len(both) // 60][:60])
    n_diff_straight, n_cmp = 0, 0
    for j, tok in enumerate(toks.tolist()):
        i = int(np.nonzero(H["tokens"] == tok)[0][0])
        p, nr = gl._human_path(tok)
        assert int(nr) == int(H["n_reg"][i]) and np.array_equal(p, H["path"][i], equal_nan=True)
        tau, v0, a0 = H["traj"][i].astype(np.float64), float(H["v0"][i]), float(H["a0"][i])
        for seed in range(3):
            got, fam = E.perturb_one(tau, v0, a0, np.random.default_rng([seed, j]), p, int(nr))
            rng_b = np.random.default_rng([seed, j])
            assert D.BANK_LAYOUT[1 + int(rng_b.integers(len(D.BANK_LAYOUT) - 1))] == fam
            monkeypatch.setattr(D, "rng_for_token", lambda _t, r=rng_b: r)
            b = D.sample_bank(H["traj"][i], tok, path_long=H["path"][i], n_valid=int(H["n_reg"][i]), v0=H["v0"][i],
                              a0=H["a0"][i], layout=(fam,))
            if got is None:
                assert not bool(b["valid"][0])
            else:
                assert bool(b["valid"][0]) and np.array_equal(got, b["drafts"][0])
                n_cmp += 1
            st, _ = E.perturb_one(tau, v0, a0, np.random.default_rng([seed, j]))
            if got is not None and st is not None and float(np.abs(st - got).max()) > 0.5:
                n_diff_straight += 1
    assert n_cmp >= 60 and n_diff_straight >= 1
    # a token outside the npz -> n_reg -1 -> extend_straight, per sample
    assert int(gl._human_path("0000000000000000")[1]) == -1
    tau0 = _tau(2, 6.0)
    hp = torch.as_tensor(np.stack([H["path"][0], np.zeros((16, 3), np.float32)]))
    hn = torch.tensor([int(H["n_reg"][0]), -1])
    src = torch.as_tensor(H["traj"][[0, 0]])
    v, a = E.ego_inputs(_status(2, 6.0))[:2]
    out, pert, _ = E.perturb_batch(tau0, v, a, 1.0, np.random.default_rng(5), src=src, src_path=hp, src_nreg=hn)
    r = np.random.default_rng(5)
    exp = []
    for k in range(2):
        r.random()
        d, _ = (E.perturb_one(H["traj"][0].astype(np.float64), float(v[k]), float(a[k]), r, H["path"][0].astype(np.float64),
                              int(H["n_reg"][0])) if k == 0 else
                E.perturb_one(H["traj"][0].astype(np.float64), float(v[k]), float(a[k]), r))
        exp.append(tau0[k].numpy() if d is None else d)
    assert np.array_equal(out.numpy(), np.stack(exp))
    # defaults: no human keys
    assert not any(k.startswith("ref_human") for k in E.GTLoader(DATA).load(GT_TOKENS[0]))
    assert {"ref_human_path", "ref_human_nreg"} <= set(gl.load(GT_TOKENS[0]))


# ----------------------------------------------------------------------------------------------- warm-up / ramp / cap
def test_new_option_defaults_are_off():
    c = _cfg()
    assert c.ref_human_only_until is None and c.kd_ratio_ramp_epochs is None and c.kd_weight_max is None
    st, _ = _stage("E2", kd_teacher_runs=("x",), kd_balance="ema", kd_ratio=0.7)
    st.ema_update(0.5, 0.02)
    for ef in (0.0, 3.3, 29.0):
        st.epoch_frac = ef
        assert st.kd_ratio_now() == 0.7 and st.ema_weight() == 0.7 * 25.0 and st.ema_weight(capped=False) == st.ema_weight()


def test_kd_ratio_ramp_and_weight_cap():
    st, _ = _stage("E2", kd_teacher_runs=("x",), kd_balance="ema", kd_ratio=2.0, kd_start_epoch=1.0,
                   kd_ratio_ramp_epochs=4.0)
    st.ema_update(0.5, 0.02)                                     # full-ratio weight 2 * 25 = 50
    for ef, r in ((0.0, 0.0), (0.99, 0.0), (1.0, 0.0), (1.5, 0.25), (2.0, 0.5), (3.0, 1.0), (4.75, 1.875),
                  (5.0, 2.0), (12.3, 2.0)):
        st.epoch_frac = ef
        assert math.isclose(st.kd_ratio_now(), r, rel_tol=0, abs_tol=1e-12), ef
        assert math.isclose(st.ema_weight(), r * 25.0, rel_tol=1e-12, abs_tol=1e-12), ef
    stc, _ = _stage("E2", kd_teacher_runs=("x",), kd_balance="ema", kd_ratio=2.0, kd_start_epoch=0.0,
                    kd_ratio_ramp_epochs=4.0, kd_weight_max=30.0)
    stc.ema_update(0.5, 0.02)
    for ef, w in ((1.0, 12.5), (2.0, 25.0), (2.4, 30.0), (3.0, 30.0), (9.0, 30.0)):   # cap after the ratio
        stc.epoch_frac = ef
        assert math.isclose(stc.ema_weight(), w, rel_tol=1e-12), ef
        assert math.isclose(stc.ema_weight(capped=False), stc.kd_ratio_now() * 25.0, rel_tol=1e-12)
    stn, _ = _stage("E2", kd_teacher_runs=("x",), kd_balance="ema", kd_ratio=1.0, kd_weight_max=10.0)
    stn.ema_update(0.5, 0.02)
    assert stn.ema_weight() == 10.0 and stn.ema_weight(capped=False) == 25.0         # cap alone, no ramp


def _draft_run(monkeypatch, epoch, B=6, bad_human=None, fail_first=False, **kw):
    """E1 loss at epoch_frac = epoch; returns (student draft tau_in, logs, tau0, human, ego)."""
    tau0 = torch.cat([_tau(3, 6.0), _tau(B - 3, 6.0, 0.01)])
    human = torch.cat([_human(3), _human(B - 3)])
    if bad_human is not None:
        human[bad_human] = float("nan")
    tg = {"ref_gt_ok": torch.zeros(B, dtype=torch.bool), "trajectory": human}
    seen = []
    orig = E.StageE._run

    def rec(net, bev, tau_in, ego):
        seen.append(tau_in.detach().clone())
        return orig(net, bev, tau_in, ego)
    monkeypatch.setattr(E.StageE, "_run", staticmethod(rec))
    if fail_first:                                                # first perturbation of the batch fails
        po, calls = E.perturb_one, []

        def one(*a, **k):
            calls.append(1)
            return (None, "x") if len(calls) == 1 else po(*a, **k)
        monkeypatch.setattr(E, "perturb_one", one)
    st, _ = _stage("E1", **kw)
    st.epoch_frac = epoch
    net = E.build_student(0, 0.1)
    _, logs = st.loss(net, {"status_feature": _status(B, 6.0)}, tg, _Toy(B)(tau0), iteration=7)
    return seen[0], logs, tau0, human, E.ego_inputs(_status(B, 6.0))[:2]


@pytest.mark.parametrize("src", ["human_mix", "tau0"])
def test_human_only_until_switches_draft_source_at_epoch(monkeypatch, src):
    kw = dict(ref_perturb_frac=0.5, kd_draft_source=src, ref_human_only_until=5.0)
    rng = lambda: np.random.default_rng([0, 7, 0])
    for ep in (0.0, 4.999):                                       # warm-up: every draft = perturbed GT human
        d, lg, tau0, human, (v0, a0) = _draft_run(monkeypatch, ep, **kw)
        exp, pert, inval = E.perturb_batch(human, v0, a0, 1.0, rng(), src=human)
        assert torch.equal(d, exp) and bool(pert.all())
        assert float(lg["ref/human_only"]) == 1.0 and float(lg["ref/frac_draft_human"]) == 1.0
        assert float(lg["ref/frac_draft_tau0"]) == 0.0 and float(lg["ref/n_human"]) == 6.0
        assert all(not torch.equal(d[i], tau0[i]) and not torch.equal(d[i], human[i]) for i in range(6))
        monkeypatch.undo()
    # invalid perturbation -> the UNPERTURBED GT human trajectory (not tau0)
    monkeypatch.undo()
    d, lg, tau0, human, _ = _draft_run(monkeypatch, 1.0, fail_first=True, **kw)
    assert torch.equal(d[0], human[0]) and float(lg["ref/n_human_invalid"]) == 1.0
    assert float(lg["ref/frac_draft_human"]) == 1.0 and math.isclose(float(lg["ref/frac_human"]), 5 / 6, rel_tol=1e-6)
    monkeypatch.undo()
    # non-finite GT human (only case) -> sg(tau0), logged as a tau0 draft
    d, lg, tau0, human, _ = _draft_run(monkeypatch, 1.0, bad_human=2, **kw)
    assert torch.equal(d[2], tau0[2]) and math.isclose(float(lg["ref/frac_draft_tau0"]), 1 / 6, rel_tol=1e-6)
    monkeypatch.undo()
    # from the switch epoch on: the kd_draft_source behaviour, identical to the run without the option
    for ep in (5.0, 12.0):
        d, lg, tau0, human, (v0, a0) = _draft_run(monkeypatch, ep, **kw)
        monkeypatch.undo()
        d_ref, lg_ref, *_ = _draft_run(monkeypatch, ep, ref_perturb_frac=0.5, kd_draft_source=src)
        monkeypatch.undo()
        assert torch.equal(d, d_ref) and float(lg["ref/loss"]) == float(lg_ref["ref/loss"])
        assert float(lg["ref/human_only"]) == 0.0
        if src == "human_mix":
            exp, pert, _ = E.perturb_batch(tau0, v0, a0, 0.5, rng(), src=human)
            assert torch.equal(d, exp) and 0 < int(pert.sum()) < 6
            assert math.isclose(float(lg["ref/frac_draft_human"]), float(pert.sum()) / 6, rel_tol=1e-6)
            assert torch.equal(d[~pert], tau0[~pert])             # the other samples: UNPERTURBED sg(tau0)
        else:
            assert float(lg["ref/frac_draft_human"]) == 0.0 and float(lg["ref/frac_draft_tau0"]) == 1.0
        assert math.isclose(float(lg["ref/frac_draft_human"]) + float(lg["ref/frac_draft_tau0"]), 1.0, rel_tol=1e-6)
        new = {"ref/human_only", "ref/frac_draft_human", "ref/frac_draft_tau0"}
        assert set(lg) - set(lg_ref) == new and not (new & set(lg_ref))   # option off: no new log keys


def test_human_only_needs_human_path_loader(monkeypatch):
    from navsim.agents.para_ssr import para_ssr_targets as T
    seen = {}

    class _GL:
        def __init__(self, root, runs, human_path=False):
            seen["hp"] = human_path

        def load(self, tok):
            return {}
    monkeypatch.setattr(E, "GTLoader", _GL)
    scene = SimpleNamespace(scene_metadata=SimpleNamespace(initial_token="t"))
    for kw, hp in (({}, False), ({"ref_human_only_until": 5.0}, True), ({"kd_draft_source": "human_mix"}, True)):
        obj = SimpleNamespace(_config=_cfg(refiner_mode="E1", **kw))
        T.ParaSSRTargetBuilder._stage_e_targets(obj, scene)
        assert seen.pop("hp", None) == hp, kw


@pytest.mark.skipif(not (TEACHER_RUN / "ckpt_best.pt").exists() or not (DATA / "human/train.npz").exists(),
                    reason="stage-T teacher run / human npz not available")
def test_human_only_e2_teachers_see_gt_human_with_logged_path_and_ramped_capped_weight(monkeypatch):
    """E2, two teachers, logged human path keys present: while epoch_frac < ref_human_only_until the student and every
    teacher get the SAME all-GT-human drafts (no sg(tau0)); after it the human_mix split; kd/lambda = ramped, capped."""
    H = np.load(DATA / "human/train.npz", allow_pickle=False)
    B = 4
    tau0 = _tau(B, 6.0, 0.01)
    human = torch.as_tensor(H["traj"][:B]).float()
    hp, hn = torch.as_tensor(H["path"][:B]).float(), torch.as_tensor(H["n_reg"][:B]).long()
    hn[3] = -1                                                     # one token without a logged path
    tg = {"ref_gt_ok": torch.zeros(B, dtype=torch.bool), "trajectory": human, "ref_human_path": hp,
          "ref_human_nreg": hn}
    for i in range(2):
        tg[E.KD_BEV_KEY.format(i)] = torch.randn(B, 256, 50, 100).half()
        tg[E.KD_OK_KEY.format(i)] = torch.ones(B, dtype=torch.bool)
    seen = []
    orig = E.StageE._run

    def rec(net, bev, tau_in, ego):
        seen.append(tau_in.detach().clone())
        return orig(net, bev, tau_in, ego)
    monkeypatch.setattr(E.StageE, "_run", staticmethod(rec))
    st, _ = _stage("E2", kd_teacher_runs=(str(TEACHER_RUN), str(TEACHER_RUN)), kd_space="decoded", kd_balance="ema",
                   kd_ratio=1.0, kd_start_epoch=0.0, kd_ratio_ramp_epochs=5.0, kd_weight_max=100.0,
                   ref_perturb_frac=0.5, kd_draft_source="human_mix", ref_human_only_until=5.0)
    net = E.build_student(0, 0.1)
    v0, a0 = E.ego_inputs(_status(B, 6.0))[:2]
    for ep, it in ((2.5, 3), (4.999, 4), (5.0, 5), (7.0, 6)):
        seen.clear()
        st.epoch_frac = ep
        st.load_ema_state({"ema_sur": 5.0, "ema_kd": 1e-3, "ema_n": 50})          # uncapped weight >> 100
        _, lg = st.loss(net, {"status_feature": _status(B, 6.0)}, tg, _Toy(B)(tau0), iteration=it)
        assert len(seen) == 3 and all(torch.equal(s, seen[0]) for s in seen)        # student + 2 teachers
        d = seen[0]
        if ep < 5.0:
            exp, pert, _ = E.perturb_batch(human, v0, a0, 1.0, np.random.default_rng([0, it, 0]), src=human,
                                           src_path=hp, src_nreg=hn)
            assert torch.equal(d, exp) and float(lg["ref/frac_draft_tau0"]) == 0.0
            assert not any(torch.allclose(d[i], tau0[i]) for i in range(B))
        else:
            exp, pert, _ = E.perturb_batch(tau0, v0, a0, 0.5, np.random.default_rng([0, it, 0]), src=human,
                                           src_path=hp, src_nreg=hn)
            assert torch.equal(d, exp) and torch.equal(d[~pert], tau0[~pert])
            assert math.isclose(float(lg["ref/frac_draft_human"]), float(pert.sum()) / B, rel_tol=1e-6)
        r = min(ep / 5.0, 1.0)
        assert math.isclose(float(lg["kd/ratio"]), r, rel_tol=1e-6)
        assert float(lg["kd/w_ema_uncapped"]) > 100.0 and float(lg["kd/lambda"]) == 100.0


# ----------------------------------------------------------------------------------------------- logger route (W&B / TB)
def _stage_logs(mode, **kw):
    """Real StageE.loss logs (GT surrogate, e0_loss -> vshare, grad_share_every=1 -> gnorm/bev_* / gshare, E2: KD)."""
    B = 2
    tg = _gt_targets(GT_TOKENS, B)
    if mode == "E2":
        tg[E.KD_BEV_KEY.format(0)] = torch.randn(B, 256, 50, 100, generator=torch.Generator().manual_seed(3)).half()
        tg[E.KD_OK_KEY.format(0)] = torch.ones(B, dtype=torch.bool)
        kw = dict(kd_teacher_runs=(str(TEACHER_RUN),), kd_space="decoded", kd_ramp=(0.0, 0.0), kd_lambda=1.0, **kw)
    st, cfg = _stage(mode, grad_share_every=1, **kw)
    net = E.build_student(0, 0.1)
    preds = _Toy(B)(_tau(B, 6.0))
    e0 = (preds["bev_embed"] ** 2).mean() + preds["trajectory"].abs().mean()
    _, logs = st.loss(net, {"status_feature": _status(B, 6.0)}, tg, preds, iteration=0, log_bev_grad=True, e0_loss=e0)
    logs = {k: v.detach() for k, v in logs.items()}
    logs["loss"], logs["loss_e0"] = e0.detach(), e0.detach()
    return st, cfg, net, logs


def _fit_with_loggers(st, cfg, net, logs, tmp_path, log_scalars=True):
    """Lightning fit (CPU, accumulate 2, log_every_n_steps 1) of a toy module whose agent publishes `logs` as
    latest_logs, with the real ParaSSRLoggingCallback + StageE callback and a capturing logger."""
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import Logger
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRLoggingCallback

    class _Cap(Logger):
        def __init__(self):
            super().__init__()
            self.rows = []

        name, version = "cap", 0
        save_dir = property(lambda self: str(tmp_path))

        def log_hyperparams(self, *a, **k):
            pass

        def log_metrics(self, metrics, step=None):
            self.rows.append((step, dict(metrics)))

    agent = SimpleNamespace(config=cfg, _stage_e=st, ref_student=net, latest_logs={})

    class _LM(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.agent = agent
            self.w = torch.nn.Parameter(torch.zeros(3))

        def training_step(self, batch, batch_idx):
            agent.latest_logs = logs
            loss = ((self.w - batch[0]) ** 2).sum()
            self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    torch.manual_seed(0)
    data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(8, 3)), batch_size=2, shuffle=True)
    cb = E.make_callback(agent)
    if not log_scalars:
        cb._log_scalars = lambda *a, **k: None
    cap = _Cap()
    lm = _LM()
    tr = pl.Trainer(callbacks=[ParaSSRLoggingCallback(), cb], logger=cap, enable_checkpointing=False,
                    accelerator="cpu", devices=1, max_epochs=1, accumulate_grad_batches=2, log_every_n_steps=1,
                    enable_progress_bar=False, enable_model_summary=False, default_root_dir=str(tmp_path))
    tr.fit(lm, data)
    keys = set().union(*(m for _, m in cap.rows))
    return keys, lm.w.detach().clone(), torch.random.get_rng_state()


@pytest.mark.skipif(not (TEACHER_RUN / "ckpt_best.pt").exists(), reason="stage-T teacher run not available")
@pytest.mark.parametrize("mode,kw", [("E1", {}), ("E2", {}),
                                     ("E2", dict(kd_balance="ema", kd_start_epoch=0.0, kd_weight_max=10.0))])
def test_stage_metrics_reach_the_lightning_logger(mode, kw, tmp_path):
    """Every ref/*, kd/*, gnorm/*, time/ref_* scalar of StageE.loss reaches the logger (the same log_metrics call W&B and
    TensorBoard get) as train/<key>_step and train/<key>_epoch, plus the jsonl-only stageE/* fields; the jsonl is still
    written; the extra logging changes neither the parameters nor the RNG state."""
    st, cfg, net, logs = _stage_logs(mode, **kw)
    keys, w1, rng1 = _fit_with_loggers(st, cfg, net, logs, tmp_path / "a")
    want = [k for k in logs if k.startswith(("ref/", "kd/", "gnorm/", "time/"))]
    if mode == "E2":
        assert {"kd/loss", "kd/lambda", "kd/weighted", "kd/l1_0", "kd/l1_lon_0", "kd/n_ok_0",
                "kd/teacher_live_0"} <= set(want)
        if kw:
            assert {"kd/w_ema", "kd/ema_sur", "kd/ema_kd", "kd/ratio", "kd/w_ema_uncapped"} <= set(want)
    assert {"ref/L_sur", "ref/t_prog", "ref/vshare_kd", "ref/gshare_sur", "gnorm/bev_e0", "time/ref_ms"} <= set(want)
    missing = [k for k in want for s in ("_step", "_epoch") if f"train/{k}{s}" not in keys]
    assert not missing, missing
    assert "train/loss_e0_step" in keys and "train/loss_e0_epoch" in keys
    for k in ("stageE/sec_step", "stageE/sec_wait", "stageE/skipped_steps"):
        assert f"{k}_step" in keys and f"{k}_epoch" in keys, k
    assert "stageE/epoch_frac" in keys
    lines = (tmp_path / "a" / "stageE_steps.jsonl").read_text().splitlines()
    assert len(lines) == 4 and "ref/L_sur" in lines[0]            # 4 micro-batches = 2 optimiser steps
    st2, cfg2, net2, logs2 = _stage_logs(mode, **kw)
    keys2, w2, rng2 = _fit_with_loggers(st2, cfg2, net2, logs2, tmp_path / "b", log_scalars=False)
    assert torch.equal(w1, w2) and torch.equal(rng1, rng2)
    assert not any(k.startswith("stageE/") for k in keys2)
