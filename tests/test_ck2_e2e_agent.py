"""CK2 e2e hooks on the real v2 ParaSSRAgent (CPU; SPEC s8 T1 / T9).

T1  flag off (ck_e2e2 {} and {'enabled': False}): state_dict keys / values, optimiser groups, target builders, callbacks,
    training forward / loss / logs / gradients and the eval forward are identical to an agent built from the
    pre-edit copy <scratch>/ck2e2e/para_ssr_agent.orig.py (same seed, CK2 fixture).
T9  flag on (main and BEV-KD arm): v2 part unchanged (student / adapter built under fork_rng), CK2 loss added after the
    balancer, gradients, lon / gate heads frozen, optimiser groups (CK2 x3, BEV-KD x3), target builder / callback,
    eval ck2_* outputs (trajectory stays v2's), on-policy step through compute_loss (no generation -> fallback set),
    strict checkpoint round trip incl. ck_bev_kd.*.
Inputs: the CK2 2-token fixture (D/ck2/e2e_smoke_cpu/fixtures/real_b2_ck2.pt, packed rows 0 / 40000) and the CK2 smoke
teachers (D/ck2/smoke/smoke_ck2{T,M}_ddp2).  CPU, 1 thread:
  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_agent.py \
    --basetemp=/tmp/claude-0/-workspace/f9196902-3a88-4418-a505-cd713226ed4a/scratchpad/ck2e2e/pytest/agent
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from test_ck_e2e_model_agent import sampling, v2_config  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-workspace/f9196902-3a88-4418-a505-cd713226ed4a/scratchpad/ck2e2e")
ORIG_AGENT = SCRATCH / "para_ssr_agent.orig.py"
D = Path("/home/external-user/ssd/yongjae_refiner/ck")
FIX2 = D / "ck2/e2e_smoke_cpu/fixtures/real_b2_ck2.pt"
SMOKE_T = D / "ck2/smoke/smoke_ck2T_ddp2"
SMOKE_M = D / "ck2/smoke/smoke_ck2M_ddp2"

pytestmark = pytest.mark.skipif(not (FIX2.is_file() and (SMOKE_T / "done.json").is_file()
                                     and (SMOKE_M / "done.json").is_file()),
                                reason="CK2 fixture / smoke teachers absent")


def ck2_cfg(tmp: Path, **over):
    d = {"enabled": True, "io_dir": str(Path(tmp) / "ck_e2e2"), "teacher_det_run": str(SMOKE_T),
         "teacher_map_run": str(SMOKE_M), "teacher_amp": False, "score_prior_rows": 200,
         "teacher_ep_check": False,      # smoke teachers are 'official' (predate --ep-target), the student 'decoupled'
         "kd_calib": {"enabled": False}}  # no KD calibration files for the smoke teachers
    d.update(over)
    return d


def arm_cfg(tmp: Path, **over):
    return ck2_cfg(tmp, bev_kd={"enabled": True, "teachers": ["det", "map"]}, **over)   # = bevkd_arm.set teachers


def build(seed: int = 0, cls=None, **cfg_over):
    if cls is None:
        from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent as cls
    torch.manual_seed(seed)
    return cls(v2_config(**cfg_over), sampling(), lr=1e-4)


def orig_agent_class():
    """ParaSSRAgent of the pre-edit copy of para_ssr_agent.py (SPEC s0: copied before the CK2 hooks were added)."""
    assert ORIG_AGENT.is_file(), ORIG_AGENT
    spec = importlib.util.spec_from_file_location("navsim.agents.para_ssr._orig_para_ssr_agent_ck2", ORIG_AGENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ParaSSRAgent


@pytest.fixture(scope="module")
def batch():
    return torch.load(FIX2, map_location="cpu", weights_only=False)


def _eq(a, b) -> bool:
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b


def _fresh(d):
    return ({k: v.clone() for k, v in d["features"].items()}, {k: v.clone() for k, v in d["targets"].items()})


def _v2_targets(t):
    drop = ("ck_", "ck2_", "ref_", "kd_bev_", "kd_ok_")
    return {k: v for k, v in t.items() if not k.startswith(drop)}


def _train_step(agent, batch, seed: int, v2_only: bool = False):
    agent.train()
    f, t = _fresh(batch)
    if v2_only:
        t = _v2_targets(t)
    torch.manual_seed(seed)
    pred = agent.forward(f)
    loss = agent.compute_loss(f, t, pred)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in agent.named_parameters() if p.grad is not None}
    return pred, loss.detach(), {k: v.detach().clone() for k, v in agent.latest_logs.items()}, grads


# ----------------------------------------------------------------------------------------------- T1 flag off
def test_off_bit_identical(batch):
    Orig = orig_agent_class()
    ref = build(0, cls=Orig)
    agents = [build(0), build(0, ck_e2e2={"enabled": False})]
    sd0 = ref.state_dict()
    for a in agents:
        assert a._ck_e2e2 is None and getattr(a, "ck_bev_kd", None) is None and a._ck_e2e is None and not hasattr(a, "ck_student")
        sd = a.state_dict()
        assert list(sd) == list(sd0) and all(_eq(sd[k], sd0[k]) for k in sd0)
        g0, g1 = ref.get_optimizers()["optimizer"].param_groups, a.get_optimizers()["optimizer"].param_groups
        assert len(g0) == len(g1) == 2
        assert [len(g["params"]) for g in g0] == [len(g["params"]) for g in g1]
        assert [{k: v for k, v in g.items() if k != "params"} for g in g0] == \
            [{k: v for k, v in g.items() if k != "params"} for g in g1]
        assert [type(b).__name__ for b in ref.get_target_builders()] == [type(b).__name__ for b in
                                                                           a.get_target_builders()]
        assert [type(c).__name__ for c in ref.get_training_callbacks()] == [type(c).__name__ for c in
                                                                             a.get_training_callbacks()]
    res = [_train_step(a, batch, 123, v2_only=True) for a in [ref] + agents]
    p0, l0, lg0, g0 = res[0]
    for p, l, lg, g in res[1:]:
        assert torch.equal(l, l0)
        assert set(p) == set(p0) and all(_eq(p[k], p0[k]) for k in p0)
        assert set(lg) == set(lg0) and all(_eq(lg[k], lg0[k]) for k in lg0)
        assert set(g) == set(g0) and all(torch.equal(g[k], g0[k]) for k in g0)
    for a in [ref] + agents:
        a.eval()
    with torch.no_grad():
        outs = [a.forward(_fresh(batch)[0]) for a in [ref] + agents]
    for o in outs[1:]:
        assert set(o) == set(outs[0]) and all(_eq(o[k], outs[0][k]) for k in o)


def test_validate_rules(tmp_path):
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
    with pytest.raises(ValueError, match="cannot both be enabled"):
        ParaSSRAgent(v2_config(ck_e2e={"enabled": True, "io_dir": str(tmp_path / "a")}, ck_e2e2=ck2_cfg(tmp_path)),
                     sampling())
    with pytest.raises(ValueError, match="unknown key"):
        ParaSSRAgent(v2_config(ck_e2e2=ck2_cfg(tmp_path, nope=1)), sampling())
    with pytest.raises(ValueError, match="io_dir"):
        ParaSSRAgent(v2_config(ck_e2e2=ck2_cfg(tmp_path, io_dir="rel/dir")), sampling())
    with pytest.raises(ValueError, match="plan_anchor"):
        ParaSSRAgent(v2_config(ck_e2e2=ck2_cfg(tmp_path), plan_anchor=False), sampling())


# ----------------------------------------------------------------------------------------------- T9 flag on
def test_on_v2_part_unchanged_and_ck2_trains(batch, tmp_path):
    from navsim.agents.para_ssr.ck.e2e_data2 import ck2_warmup_prior
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
    off = build(0)
    on = build(0, ck_e2e2=ck2_cfg(tmp_path))
    sd_off, sd_on = off.state_dict(), on.state_dict()
    ck_keys = [k for k in sd_on if k.startswith("ck_student.")]
    assert ck_keys and [k for k in sd_on if not k.startswith("ck_student.")] == list(sd_off)
    assert all(_eq(sd_on[k], sd_off[k]) for k in sd_off)
    assert not any("teacher" in k or k.startswith("ck_bev_kd.") for k in sd_on)
    st = on.ck_student
    assert st.trunk.adapter.bev_grad_scale == 1.0
    assert float(st.trunk.lon_head[-1].weight.abs().max()) == 0.0 and float(st.trunk.lon_head[-1].bias.abs().max()) == 0
    frozen = [n for n, p in st.named_parameters() if not p.requires_grad]
    assert frozen and all(n.startswith(("trunk.lon_head.", "trunk.gate_head.")) for n in frozen)
    # score prior = ck2_warmup_prior of the student's warm-up mix (NU20)
    prior = ck2_warmup_prior(CKE2E2Config.from_any(ck2_cfg(tmp_path)), n_max=200)
    b = st.score_head[-1].bias.detach().double().numpy()
    np.testing.assert_allclose(1 / (1 + np.exp(-b)), np.clip(prior, 1e-4, 1 - 1e-4), rtol=1e-5)
    # optimiser: v2 groups unchanged + CK2 group lr x3, wd 0.01, trainable student params only
    groups = on.get_optimizers()["optimizer"].param_groups
    assert len(groups) == 3 and groups[2]["lr_scale"] == 3.0 and groups[2]["weight_decay"] == 0.01
    assert abs(groups[2]["initial_lr"] - 3e-4) < 1e-12 and abs(groups[2]["lr"] - 3.0 * groups[0]["lr"]) < 1e-12
    assert {id(p) for p in groups[2]["params"]} == {id(p) for p in st.parameters() if p.requires_grad}
    assert type(on.get_target_builders()[-1]).__name__ == "CK2E2ETargetBuilder"
    assert type(on.get_training_callbacks()[-1]).__name__ == "CKE2E2Callback"
    # warm-up train step (epoch 0)
    (_, l_off, lg_off, g_off), (_, l_on, lg_on, g_on) = (_train_step(off, batch, 7, v2_only=True),
                                                         _train_step(on, batch, 7))
    assert torch.equal(lg_on["loss_v2"], l_off)
    assert torch.isfinite(l_on) and float(lg_on["ck2/loss"]) > 0 and float(lg_on["ck2/loss_nonfinite"]) == 0
    assert abs(float(l_on) - float(l_off) - float(lg_on["ck2/loss"])) < 1e-4 * max(1.0, float(l_on))
    for key in lg_off:
        if key.startswith("loss_") and key in lg_on:
            assert torch.equal(lg_on[key], lg_off[key]), key
    assert float(lg_on["ck2/phase"]) == 0.0 and float(lg_on["ck2/n_cand"]) == 64.0
    assert float(lg_on["ck2/kd_ok_frac"]) > 0.5 and float(lg_on["ck2/n_gt"]) >= 1
    assert float(lg_on["ck2/w_lat"]) == 0.0                     # lateral KD off before lat_kd.start_mb (1000)
    missing = [n for n, p in st.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
    assert all(p.grad is None for p in st.parameters() if not p.requires_grad)
    assert all(torch.isfinite(g).all() for g in g_on.values())
    gb = [g_on[n] for n in g_on if n.startswith("para_ssr_model.")]
    assert gb and all(torch.isfinite(g).all() for g in gb)
    # the CK2 loss reaches the v2 BEV: v2 gradients differ from the off run
    assert any(not torch.equal(g_on[n], g_off[n]) for n in g_off if n.startswith("para_ssr_model.") and n in g_on)
    for k, v in lg_on.items():
        assert torch.isfinite(v).all(), k
    assert on._ck_e2e2.mb == 1


def test_on_eval_outputs(batch, tmp_path):
    on = build(0, ck_e2e2=ck2_cfg(tmp_path)).eval()
    off = build(0).eval()
    with torch.no_grad():
        pred = on.forward(_fresh(batch)[0])
        p_off = off.forward(_fresh(batch)[0])
    B = 2
    shapes = {"ck2_cand_idx": (B, 16), "ck2_v2_final": (B, 16), "ck2_v2_im": (B, 16), "ck2_v2_sim": (B, 16, 5),
              "ck2_cand96": (B, 96, 8, 3), "ck2_valid96": (B, 96), "ck2_score_logit": (B, 96, 5),
              "ck2_w_lat": (B, 96, 6), "ck2_e_lat": (B, 96, 6), "ck2_lat_traj": (B, 96, 8, 3)}
    for k, s in shapes.items():
        assert tuple(pred[k].shape) == s, k
        assert torch.isfinite(pred[k].float()).all(), k
    assert "ck2_sel_idx" not in pred and not any(k.startswith("ck_") for k in pred)
    assert set(p_off) == set(pred) - set(shapes)
    assert torch.equal(p_off["trajectory"], pred["trajectory"])          # NU30
    assert float((pred["ck2_cand96"][:, 0] - pred["trajectory"].float()).abs().max()) < 1e-4
    assert pred["ck2_valid96"][:, ::6].all() and float(pred["ck2_valid96"][:, 1:].float().mean()) > 0.5
    # fresh student: zero-initialised lateral head -> identity correction
    assert torch.allclose(pred["ck2_lat_traj"], pred["ck2_cand96"], atol=1e-5)


def test_onpolicy_step_through_agent(batch, tmp_path):
    """epoch 5 (on-policy) on the real v2 predictions: current top-16 + 32 variants (KD) and, without any generation,
    the warm-up-style fallback label set (G_src 0) -> 96 student candidates per token; finite loss and gradients;
    epoch 4 (warmup_record) writes the 96-column record of both tokens."""
    from navsim.agents.para_ssr.ck import e2e_data as ED
    from navsim.agents.para_ssr.ck import e2e_data2 as D2
    from navsim.agents.para_ssr.ck.online2 import CandRecorder2
    on = build(0, ck_e2e2=ck2_cfg(tmp_path, grad_share_every=1))
    ck = on._ck_e2e2
    ck.epoch, ck.epoch_frac = 4, 4.0
    ck.recorder = CandRecorder2(ck.cfg.io_dir, 0, 1, 4, ck.cfg.rec_chunk_tokens)
    _, l4, lg4, _ = _train_step(on, batch, 11)
    ck.recorder.close_epoch()
    ck.recorder = None
    assert float(lg4["ck2/phase"]) == 1.0 and float(lg4["ck2/rec_tokens"]) == 2.0 and torch.isfinite(l4)
    chunks = ED.list_rec_chunks(ck.cfg.io_dir, 4)
    assert len(chunks) == 1
    d = D2.read_rec2_chunk(chunks[0]["path"])
    assert d["traj"].shape == (2, 96, 8, 3) and sorted(d["row"].tolist()) == sorted(batch["rows"])
    on.zero_grad(set_to_none=True)
    ck.epoch, ck.epoch_frac = 5, 5.0
    _, l5, lg5, g5 = _train_step(on, batch, 12)
    assert float(lg5["ck2/phase"]) == 2.0 and torch.isfinite(l5)
    assert float(lg5["ck2/n_cand"]) == 96.0 and float(lg5["ck2/G_src_fallback"]) == 1.0
    assert float(lg5["ck2/G_ok_frac"]) > 0.5 and float(lg5["ck2/kd_ok_frac"]) > 0.5
    for k in ("gnorm/bev_v2", "gnorm/bev_ck", "ck2/gshare"):
        assert k in lg5 and math.isfinite(float(lg5[k])), k
    assert 0.0 < float(lg5["ck2/gshare"]) < 1.0
    assert all(torch.isfinite(g).all() for g in g5.values())
    missing = [n for n, p in on.ck_student.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_bevkd_arm_agent(batch, tmp_path):
    main = build(0, ck_e2e2=ck2_cfg(tmp_path / "m"))
    arm = build(0, ck_e2e2=arm_cfg(tmp_path / "a"))
    sd_m, sd_a = main.state_dict(), arm.state_dict()
    bk = [k for k in sd_a if k.startswith("ck_bev_kd.")]
    assert sorted(bk) == sorted([f"ck_bev_kd.kd.{a}.{t}.{p}" for t in ("det", "map")
                                 for a, p in (("adapters", "proj.weight"), ("adapters", "proj.bias"),
                                              ("zscore", "mean"), ("zscore", "std"))]), bk
    assert [k for k in sd_a if not k.startswith("ck_bev_kd.")] == list(sd_m)
    assert all(_eq(sd_a[k], sd_m[k]) for k in sd_m)                     # same v2 / student init (fork_rng)
    for t in ("det", "map"):
        assert float(arm.ck_bev_kd.kd.adapters[t].proj.weight.abs().max()) == 0.0      # zero init
    groups = arm.get_optimizers()["optimizer"].param_groups
    assert len(groups) == 4 and groups[3]["lr_scale"] == 3.0 and groups[3]["weight_decay"] == 0.01
    assert {id(p) for p in groups[3]["params"]} == {id(p) for p in arm.ck_bev_kd.parameters() if p.requires_grad}
    assert {id(p) for p in groups[2]["params"]}.isdisjoint({id(p) for p in groups[3]["params"]})
    (_, l_m, lg_m, g_m), (_, l_a, lg_a, g_a) = _train_step(main, batch, 5), _train_step(arm, batch, 5)
    assert torch.isfinite(l_a) and float(lg_a["bevkd/det/lam"]) == 0.0 == float(lg_a["bevkd/map/lam"])  # mb 0 < 10600
    for k in ("bevkd/det/loss", "bevkd/det/cos", "bevkd/map/loss", "bevkd/map/ok_frac", "bevkd/term",
              "bevkd/nonfinite"):
        assert k in lg_a, k
    assert float(lg_a["bevkd/map/loss"]) > 0 and float(lg_a["bevkd/map/ok_frac"]) == 1.0
    assert float(lg_a["bevkd/term"]) > 0 and float(lg_a["bevkd/nonfinite"]) == 0
    assert abs(float(l_a) - float(l_m) - float(lg_a["bevkd/term"])) < 1e-4 * max(1.0, float(l_a))
    # lambda = 0: the BEV gets no KD gradient -> v2 / student gradients equal the main run's; the adapter learns
    for n in g_m:
        assert torch.allclose(g_a[n], g_m[n], rtol=0, atol=1e-7), n
    ga = [p.grad for p in arm.ck_bev_kd.parameters() if p.requires_grad]
    assert ga and all(g is not None and torch.isfinite(g).all() for g in ga)
    assert any(float(g.abs().max()) > 0 for g in ga)


def test_initialize_strict_roundtrip(tmp_path):
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
    a = build(0, ck_e2e2=arm_cfg(tmp_path))
    with torch.no_grad():
        for p in list(a.ck_student.parameters()) + list(a.ck_bev_kd.parameters()):
            p.add_(0.01)
    ck = tmp_path / "x.ckpt"
    torch.save({"state_dict": {f"agent.{k}": v for k, v in a.state_dict().items()}}, ck)
    b = ParaSSRAgent(v2_config(ck_e2e2=arm_cfg(tmp_path)), sampling(), checkpoint_path=str(ck))
    assert b.ck_student.init_report["score_prior"] is None             # apply_prior off with a checkpoint
    b.initialize()
    for m_a, m_b in ((a.ck_student, b.ck_student), (a.ck_bev_kd, b.ck_bev_kd)):
        sa, sb = m_a.state_dict(), m_b.state_dict()
        assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    # a CK2 checkpoint without the arm does not load into an arm agent, and a v2-only one not into a CK2 agent
    m = build(0, ck_e2e2=ck2_cfg(tmp_path))
    ck_m = tmp_path / "m.ckpt"
    torch.save({"state_dict": {f"agent.{k}": v for k, v in m.state_dict().items()}}, ck_m)
    c = ParaSSRAgent(v2_config(ck_e2e2=arm_cfg(tmp_path)), sampling(), checkpoint_path=str(ck_m))
    with pytest.raises(RuntimeError, match="Missing key"):
        c.initialize()
    off = build(0)
    ck_o = tmp_path / "off.ckpt"
    torch.save({"state_dict": {f"agent.{k}": v for k, v in off.state_dict().items()}}, ck_o)
    e = ParaSSRAgent(v2_config(ck_e2e2=ck2_cfg(tmp_path)), sampling(), checkpoint_path=str(ck_o))
    with pytest.raises(RuntimeError, match="Missing key"):
        e.initialize()
