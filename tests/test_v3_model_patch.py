"""refiner-v3 CKNet integration of the temporal branch (SPEC 4.5 / 4.7) -- tests of the proposed model.py edit.

model.py is owned by the integration step; the MODEL builder ships the edit as docs/refiner_v3/model_py_temporal.patch.
While model.py is unpatched (no `tfeat` in it), the patch is applied to a copy under D3/test_tmp and that copy is
imported as navsim.agents.para_ssr.ck._model_v3patched (same package, so relative imports resolve); once model.py is
patched the real module is tested.  The unpatched 8ae8b1d model.py is the OFF reference.

  OFF   : identical state_dict / param_counts / forward (bitwise) to 8ae8b1d model.py; ck2T10 ckpt_last loads strictly
          with param_count 3,516,178 (config.json); trainable after train_ck2.freeze_unused == 3,391,243 if importable
  ON    : init == OFF bitwise through CKNet.forward(objs=...) (cand + extra + rescore_corr), arms a and b; every
          trainable parameter a finite gradient (score loss + forecast loss); out['fc'] only for the learned arm
  guards: temporal with arm M; forward without objs / objs to an OFF net; trunk.candidates without tfeat;
          build_ck(init_from=ck2T10, temporal on); load_ck of a PRIVILEGED run -> PermissionError (allowed: strict
          load, ca_b restored from the checkpoint)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
CK = str(Path(__file__).resolve().parents[1])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import uuid  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck import model as M0  # noqa: E402
from navsim.agents.para_ssr.ck.temporal import config as TC  # noqa: E402
from navsim.agents.para_ssr.ck.temporal.loss import forecast_loss  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_temporal_testlib as TL  # noqa: E402

PATCH = Path(CK) / "docs" / "refiner_v3" / "model_py_temporal.patch"
CK2T10 = Path("/home/external-user/ssd/yongjae_refiner/ck/ck2/train/ck2T10")
TMP_ROOT = TC.D3 / "test_tmp"
CA_B = torch.full((2, 10), 0.4)


@pytest.fixture(scope="module")
def tmpdir_d3():
    d = TMP_ROOT / f"model_patch_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def MP(tmpdir_d3):
    if "tfeat" in Path(M0.__file__).read_text():
        return M0                                                          # patch already applied
    if not PATCH.is_file() or shutil.which("patch") is None:
        pytest.skip("patch file / patch tool not available")
    out = tmpdir_d3 / "model_patched.py"
    r = subprocess.run(["patch", "-s", "-o", str(out), M0.__file__, str(PATCH)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    name = "navsim.agents.para_ssr.ck._model_v3patched"
    spec = importlib.util.spec_from_file_location(name, out)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _norm():
    rng = np.random.default_rng(3)
    return rng.normal(0, 0.3, 256).astype(np.float32), rng.uniform(0.5, 2.0, 256).astype(np.float32)


def _inputs(T=2, K=4, K2=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    cand = TL.rand_cands(T, K, seed=seed).reshape(T, K, 8, 3)
    extra = TL.rand_cands(T, K2, seed=seed + 1).reshape(T, K2, 8, 3)
    sf = torch.zeros(T, 8)
    sf[:, 1] = 1.0
    sf[:, 4] = 8.0
    bev = torch.randn(T, 256, 50, 100, generator=g)
    (obj, mask, fv), _ = TL.rand_objs(T, (11, 6), seed=seed)
    return bev, cand, extra, sf, {"obj_in": obj, "obj_mask": mask, "frame_valid": fv}


def _cfg(arm, fc_train="joint"):
    f = {"a": "learned", "b": "ca", "d": "oracle_gt"}[arm]
    return TC.TemporalCfg(on=True, forecaster=f, n_modes=6 if f == "learned" else 1,
                          fc_train=fc_train if f == "learned" else "frozen")


def _same_outputs(o1, o2):
    for k in ("score_logit", "z_lon", "w_lat", "gate_logit", "extra_score_logit", "corr_score_logit"):
        assert torch.equal(o1[k], o2[k]), k


# ----------------------------------------------------------------------------------------------- OFF
def test_off_identical_to_8ae8b1d(MP):
    ref = M0.CKNet("T", seed=0, norm=_norm()).eval()
    net = MP.CKNet("T", seed=0, norm=_norm()).eval()
    assert net.temporal is None and not getattr(net.trunk, "_temporal_required", False)
    sr, sn = ref.state_dict(), net.state_dict()
    assert list(sr) == list(sn) and all(torch.equal(sr[k], sn[k]) for k in sr)
    assert ref.param_counts() == net.param_counts()
    bev, cand, extra, sf, _ = _inputs()
    with torch.no_grad():
        _same_outputs(ref(bev, cand, sf, extra=extra, rescore_corr=True), net(bev, cand, sf, extra=extra,
                                                                              rescore_corr=True))
    with pytest.raises(ValueError):
        net(bev, cand, sf, objs=_inputs()[4])


@pytest.mark.skipif(not (CK2T10 / "ckpt_last.pt").is_file(), reason="ck2T10 checkpoint not available")
def test_off_loads_ck2T10_strictly(MP):
    net, cfg = MP.load_ck(CK2T10, "last")
    assert net.temporal is None and "temporal" not in cfg
    assert net.param_counts()["total"] == cfg["param_count"] == 3516178
    try:
        from tools.ck.train_ck2 import freeze_unused
    except Exception:
        return                                                            # trainer deps not importable here
    freeze_unused(net)
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) == cfg["param_count_trainable"] == 3391243


# ----------------------------------------------------------------------------------------------- ON
@pytest.mark.parametrize("arm", ["a", "b"])
def test_on_at_init_equals_off_and_gradients(MP, arm):
    off = MP.CKNet("T", seed=0, norm=_norm()).eval()
    on = MP.CKNet("T", seed=0, norm=_norm(), temporal=_cfg(arm), ca_b=None if arm == "a" else CA_B).eval()
    so, sn = off.state_dict(), on.state_dict()
    assert all(torch.equal(so[k], sn[k]) for k in so)                     # CK2 modules: same init
    extra_keys = set(sn) - set(so)
    assert extra_keys and all(k.startswith("temporal.") for k in extra_keys)
    pc = on.param_counts()
    assert pc["temporal_shared"] == 237888 and pc["total"] == off.param_counts()["total"] + 237888 + pc["forecaster"]
    bev, cand, extra, sf, objs = _inputs()
    with torch.no_grad():
        o_off = off(bev, cand, sf, extra=extra, rescore_corr=True)
        o_on = on(bev, cand, sf, extra=extra, rescore_corr=True, objs=objs)
    _same_outputs(o_off, o_on)                                            # zero-init projections: bit-identical
    assert ("fc" in o_on) == (arm == "a") and o_on["t_stats"]["n_obj"] >= 1
    with pytest.raises(ValueError):
        on(bev, cand, sf)
    with pytest.raises(RuntimeError):
        v0, a0, eds, cmd = o_on["ego"]
        feat, mem = on.trunk.scene(bev, 2)
        on.trunk.candidates(feat, mem, cand, v0, a0, eds, cmd)
    # gradients: score loss (+ forecast loss in arm a), projections randomised so the branch is in the score path
    on.train()
    with torch.no_grad():
        for m in (on.temporal.st_proj, on.temporal.gap_proj):
            m.weight.normal_(0, 0.02)
    out = on(bev, cand, sf, extra=extra, objs=objs, decode=False)
    loss = out["score_logit"].sum() + out["extra_score_logit"].sum()
    if arm == "a":
        fl, _ = forecast_loss(out["fc"], torch.randn(2, TC.K_OBJ, 10, 3) * 5, torch.ones(2, TC.K_OBJ, 10, dtype=torch.int8),
                              objs["obj_mask"][:, 0])
        loss = loss + fl
    loss.backward()
    skip = ("trunk.gate_head", "trunk.lon_head", "trunk.lat_head", "lead_head")   # not in this loss
    for n, p in on.named_parameters():
        if n.startswith(skip):
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert float(on.temporal.st_enc[0].weight.grad.abs().sum()) > 0


def test_guards(MP, tmpdir_d3):
    with pytest.raises(ValueError):
        MP.CKNet("M", seed=0, norm=_norm(), temporal=_cfg("a"))
    if (CK2T10 / "ckpt_last.pt").is_file():
        with pytest.raises(ValueError, match="temporal"):
            MP.build_ck("T", 0, init_from=CK2T10, temporal=_cfg("b"))
    # privileged run: refused by load_ck unless allowed; allowed load restores ca_b from the checkpoint
    run = tmpdir_d3 / "v3T_test_d_oracle_PRIV"
    run.mkdir()
    cb = torch.arange(20, dtype=torch.float32).reshape(2, 10) * 0.01 + 0.3
    net = MP.CKNet("T", seed=0, norm=_norm(), temporal=_cfg("d"), ca_b=cb)
    (run / "config.json").write_text(json.dumps({"arm": "T", "seed": 0, "temporal": _cfg("d").to_dict(),
                                                 "privileged": 1}))
    MP.save_ck(run / "ckpt_last.pt", net, {"arm": "T"})
    with pytest.raises(PermissionError, match="PRIVILEGED"):
        MP.load_ck(run)
    n2, _ = MP.load_ck(run, allow_privileged=True)
    assert torch.equal(n2.temporal.ca_b, cb) and n2.temporal_cfg.privileged
    bev, cand, extra, sf, objs = _inputs()
    with pytest.raises(ValueError):
        n2(bev, cand, sf, objs=objs)                                      # oracle needs the privileged keys
    objs = dict(objs, priv_fc_xyh=torch.randn(2, TC.K_OBJ, 10, 3), priv_fc_state=torch.ones(2, TC.K_OBJ, 10, dtype=torch.int8))
    with torch.no_grad():
        o = n2(bev, cand, sf, objs=objs)
    assert "fc" not in o and torch.isfinite(o["score_logit"]).all()
