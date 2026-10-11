"""refiner-v3 review fixes (docs/refiner_v3/REVIEW_FIXES.md), CPU only.

  * D1 K_OBJ = 96, pack root objpack_k96; the real packs have that layout and pass the 1 % truncation rule;
  * singleton rule of match_targets (FC_SINGLE; tracks first annotated after t0 never matched); oracle reads
    FC_SINGLE as OBS; the forecast loss ignores it;
  * D2 CA scales fitted on the augmented stream (b > b_clean, records aug / source / k_obj); load_ca_b checks source;
  * D3 TemporalCfg fc_init / fc_train rules and legacy mapping; frozen learned branch runs without gradient;
    forecaster pre-training stage on real packs -> forecaster.pt loads strictly into arm a's branch;
  * correctness finding 3: OFF configs compare equal after normalisation;
  * D5 provenance.compare; D6 infer logit check tolerance.

  cd /workspace/yongjae/SSR-v3 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-v3 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 /venv/ssr/bin/python -B -m pytest -q -p no:cacheprovider \
    tests/test_v3_review_fixes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
CK = str(Path(__file__).resolve().parents[1])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import json  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck.temporal import config as TC  # noqa: E402
from navsim.agents.para_ssr.ck.temporal import objects as O  # noqa: E402
from navsim.agents.para_ssr.refiner import gt_future as GF  # noqa: E402

PK = TC.PACK_ROOT / "bevfusion"
have_packs = pytest.mark.skipif(not all((PK / s / "build.json").is_file() for s in
                                        ("navtrain_train", "navtrain_val", "navtest")), reason="K96 packs absent")


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------------- D1
def test_k_obj_and_pack_root():
    assert TC.K_OBJ == 96 and TC.PACK_ROOT.name == "objpack_k96"
    assert TC.TemporalCfg().pack_root == str(TC.PACK_ROOT)


@have_packs
def test_real_packs_layout_and_truncation():
    from tools.ck.data.obj_pack import ObjPack
    for s, n in (("navtrain_train", 85109), ("navtrain_val", 18179), ("navtest", 12146)):
        b = json.loads((PK / s / "build.json").read_text())
        assert b["status"] == "complete" and b["K_OBJ"] == 96 and b["n"] == n
        assert b["counts"]["truncated_frac_of_valid_frames"] <= 0.01, s
        assert b["targets"] is True                                    # navtest has GT targets now (D3 gt_objects)
        assert b["provenance"]["gt"]["dir"] == str(TC.D3 / "gt_objects" / s)
        it = ObjPack(s, root=TC.PACK_ROOT, targets=True).item(0)
        assert it["obj_in"].shape == (3, 96, 17) and it["fc_state"].shape == (96, 10)
        v = json.loads((PK / s / "verify.json").read_text())
        assert v["ok"] is True


# ----------------------------------------------------------------------------------------------- singleton rule
def _gt(tracks):
    """minimal gt_future object dict: tracks = [(cls, agent, fk, lk, xy_fn(k) -> (x, y))]."""
    A = len(tracks)
    kf = np.zeros((A, GF.N_KF, 6), np.float32)
    first = np.zeros((A, 6), np.float32)
    meta = np.zeros((A, 5), np.int16)
    for i, (cls, ag, fk, lk, fn) in enumerate(tracks):
        for k in range(fk, lk + 1):
            x, y = fn(k)
            kf[i, k, :3] = (x, y, 0.0)
            kf[i, k, GF.KF_PRESENT] = 1.0
        first[i] = (4.6, 2.0, 0.0, 0.0, 0.0, fk)
        meta[i] = (cls, ag, fk, lk, int(fk == lk))
    return dict(kf=kf, first=first, meta=meta, track=np.array([f"t{i}" for i in range(A)], "<U16"),
                n_kf=np.int32(GF.N_KF))


def _det(xys):
    obj0 = np.zeros((TC.K_OBJ, TC.D_OBJ), np.float32)
    m = np.zeros(TC.K_OBJ, bool)
    for i, (x, y) in enumerate(xys):
        obj0[i, :2] = (x, y)
        obj0[i, 2] = 1.0
        obj0[i, 4:6] = (4.6, 2.0)
        obj0[i, 8] = 0.9
        obj0[i, 9] = 1.0                                                # vehicle
        m[i] = True
    return obj0, m


def test_match_targets_singleton_rule():
    gt = _gt([(0, 1, 0, 10, lambda k: (10 + 2.5 * k, 0.0)),            # moving vehicle, annotated 0..5 s
              (0, 1, 0, 0, lambda k: (20.0, 3.0)),                      # singleton at t0
              (0, 1, 2, 2, lambda k: (30.0, -3.0))])                    # singleton first annotated at t = 1 s
    obj0, m = _det([(10.1, 0.0), (20.0, 3.1), (30.0, -3.0)])
    xyh, st, tr, s = O.match_targets(obj0, m, gt, return_stats=True)
    assert tr[:3].tolist() == [0, 1, -1]                                 # the future-first singleton is not matched
    assert (st[0] == TC.FC_OBS).all() and np.allclose(xyh[0, :, 0], 10 + 2.5 * np.arange(1, 11))
    assert (st[1] == TC.FC_SINGLE).all() and np.allclose(xyh[1, :, :2], [20.0, 3.0])
    assert (st[2] == TC.FC_NONE).all()
    assert s["n_single_matched"] == 1 and s["n_gt_future_first"] == 1 and s["n_matched"] == 2


def test_oracle_reads_single_as_obs_and_loss_ignores_it():
    from navsim.agents.para_ssr.ck.temporal.forecast import LearnedForecaster, oracle_forecast, trim_objects
    from navsim.agents.para_ssr.ck.temporal.loss import forecast_valid
    obj0, m = _det([(10.0, 0.0), (20.0, 3.0)])
    obj = np.zeros((1, 3, TC.K_OBJ, TC.D_OBJ), np.float32)
    obj[0, 0] = obj0
    mask = np.zeros((1, 3, TC.K_OBJ), bool)
    mask[0, 0] = m
    fv = torch.tensor([[True, False, False]])
    xyh = torch.zeros(1, TC.K_OBJ, 10, 3)
    xyh[0, 0, :, 0], xyh[0, 1, :, :2] = 50.0, torch.tensor([20.0, 3.0])
    st = torch.zeros(1, TC.K_OBJ, 10, dtype=torch.int8)
    st[0, 0], st[0, 1] = TC.FC_OBS, TC.FC_SINGLE
    oi, om, f = trim_objects(torch.from_numpy(obj), torch.from_numpy(mask), fv)
    fc = oracle_forecast(oi, om, f, xyh, st, torch.ones(2, 10))
    assert torch.allclose(fc.mu[0, 0, 0, :, 0], torch.full((10,), 50.0))
    assert torch.allclose(fc.mu[0, 1, 0, :, :], torch.tensor([20.0, 3.0]).expand(10, 2))
    assert float(fc.b[0, 1, 0].max()) == TC.B_ORACLE
    lf = LearnedForecaster(n_modes=2, seed=0)(oi, om, f)
    v = forecast_valid(lf, st, om[:, 0])
    assert v[0, 0].all() and not v[0, 1].any()


# ----------------------------------------------------------------------------------------------- D3 config rules
def test_temporal_cfg_fc_rules():
    a = TC.TemporalCfg(on=True)
    assert a.fc_train == "frozen" and a.fc_frozen and a.fc_init == ""
    a.validate("T")                                                    # building is fine without fc_init
    with pytest.raises(ValueError, match="fc_init"):
        a.check_trainable()                                            # training a frozen random forecaster: no
    TC.TemporalCfg(on=True, fc_init="/x.pt").check_trainable()
    TC.TemporalCfg(on=True, fc_train="joint").check_trainable()
    with pytest.raises(ValueError, match="joint"):
        TC.TemporalCfg(on=True, fc_backprop=True).validate("T")
    TC.TemporalCfg(on=True, fc_backprop=True, fc_train="joint").validate("T")
    with pytest.raises(ValueError, match="learned forecaster only"):
        TC.TemporalCfg(on=True, forecaster="ca", n_modes=1, fc_init="/x.pt").validate("T")
    with pytest.raises(ValueError):
        TC.TemporalCfg(on=True, fc_train="bogus").validate("T")
    # legacy: learned configs written before fc_train existed trained jointly
    old = {k: v for k, v in TC.TemporalCfg(on=True).to_dict().items() if k not in ("fc_train", "fc_init")}
    assert TC.TemporalCfg.from_dict(old).fc_train == "joint"
    assert TC.TemporalCfg.from_dict(dict(old, forecaster="ca", n_modes=1)).fc_train == "frozen"
    # correctness finding 3: OFF configs are equal after normalisation whatever their other fields
    off2 = TC.TemporalCfg(p_obj_drop=0.2, pack_root="/elsewhere")
    assert off2 != TC.TemporalCfg() and off2.normalised() == TC.TemporalCfg()
    assert TC.TemporalCfg(on=True).normalised() == TC.TemporalCfg(on=True)


def test_frozen_branch_has_no_forecaster_gradient():
    from navsim.agents.para_ssr.ck.temporal.branch import TemporalBranch
    sys.path.insert(1, str(Path(__file__).resolve().parent))
    import v3_temporal_testlib as TL
    (obj, mask, fv), _ = TL.rand_objs(2, (6, 3), seed=0)
    br = TemporalBranch(TC.TemporalCfg(on=True, fc_init="/unused.pt"), 192, 0)
    ctx = br.scene({"obj_in": obj, "obj_mask": mask, "frame_valid": fv})
    assert ctx.fc_raw is not None and not ctx.fc_raw.mu.requires_grad and not ctx.fc.mu.requires_grad
    assert ctx.n_valid == int(ctx.fc.valid.sum()) and ctx.n_obj >= 1
    joint = TemporalBranch(TC.TemporalCfg(on=True, fc_train="joint"), 192, 0)
    cj = joint.scene({"obj_in": obj, "obj_mask": mask, "frame_valid": fv})
    assert cj.fc_raw.mu.requires_grad and not cj.fc.mu.requires_grad   # fc_backprop False: raster reads detached
    assert torch.equal(cj.fc_raw.mu.detach(), ctx.fc_raw.mu)            # same init, same forecast
    empty = br.scene({"obj_in": obj, "obj_mask": torch.zeros_like(mask), "frame_valid": fv})
    assert empty.n_valid == 0 and empty.n_obj == 1                     # cosmetic finding 9: n_valid is the count


# ----------------------------------------------------------------------------------------------- D2 CA scales
@have_packs
def test_fit_ca_scales_augmented(tmp_path):
    from tools.ck.v3 import fit_ca_scales as F
    out = tmp_path / "ca.json"
    F.main(["--pack", str(PK / "navtrain_val"), "--allow-split", "--limit", "600", "--aug-epochs", "0-1",
            "--workers", "2", "--out", str(out)])
    d = json.loads(out.read_text())
    assert d["source"] == "bevfusion" and d["k_obj"] == 96 and d["aug"]["epochs"] == [0, 1]
    assert d["aug"]["v_noise"] == 0.3 and d["aug"]["p_obj_drop"] == 0.05 and d["aug"]["stream"] == "objaug"
    for g in ("vehicle", "vru"):
        assert all(x > y for x, y in zip(d["b"][g], d["b_clean"][g])), g   # noisy velocities: wider scales
        assert sum(d["n"][g]) > sum(d["n_clean"][g])                      # 2 epochs of the stream vs 1 clean pass
    with pytest.raises(SystemExit):
        F.main(["--pack", str(PK / "navtrain_val"), "--allow-split", "--limit", "10", "--out", str(out)])  # no overwrite
    # the real Step-1 file: augmented fit on the K96 train pack
    real = json.loads(TC.CA_SCALES.read_text())
    bj = PK / "navtrain_train" / "build.json"
    import hashlib
    assert real["pack_sha256"] == hashlib.sha256(bj.read_bytes()).hexdigest()
    assert real["aug"]["epochs"] == list(range(10)) and real["aug"]["seed"] == 0 and real["k_obj"] == 96


def test_load_ca_b_source(tmp_path):
    from navsim.agents.para_ssr.ck.temporal.branch import load_ca_b
    b = {g: [0.1 * (k + 1) for k in range(10)] for g in ("vehicle", "vru")}
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"b": b, "source": "v2det"}))
    assert load_ca_b(p).shape == (2, 10)
    with pytest.raises(ValueError, match="source"):
        load_ca_b(p, source="bevfusion")
    p.write_text(json.dumps({"b": b}))                                  # pre-fix file: bevfusion
    load_ca_b(p, source="bevfusion")


# ----------------------------------------------------------------------------------------------- D3 pre-training
@have_packs
def test_pretrain_stage_cpu_and_strict_load(tmp_path):
    from navsim.agents.para_ssr.ck.temporal.branch import TemporalBranch
    from tools.ck.v3 import pretrain_forecaster as PF
    out = tmp_path / "pre"
    PF.main(["--out", str(out), "--device", "cpu", "--epochs", "1", "--max-steps", "2", "--batch", "8",
             "--limit-tokens", "32", "--val-limit", "16", "--eval-splits", "navtrain_val,navtest",
             "--train-eval-rows", "0", "--workers", "0", "--log-every", "1"])
    m = torch.load(out / "forecaster.pt", map_location="cpu", weights_only=False)
    assert m["n_modes"] == 6 and m["step"] == 2
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["aug"]["aug_seed"] == PF.AUG_SEED_OFFSET and cfg["provenance"]["data"]["packs"]["navtrain_train"][
        "k_obj"] == 96
    val = [json.loads(x) for x in (out / "val_metrics.jsonl").read_text().splitlines()]
    assert {r["split"] for r in val} == {"navtrain_val", "navtest"}
    assert all(np.isfinite(r["learned"]["fc_minade"]) and np.isfinite(r["ca"]["fc_minade"]) for r in val)
    br = TemporalBranch(TC.TemporalCfg(on=True, fc_init=str(out / "forecaster.pt")), 192, 0)
    br.forecaster.load_state_dict(m["forecaster"], strict=True)
    # the pre-training init equals the in-branch init of arm a (same seed): untrained weights compare equal
    from navsim.agents.para_ssr.ck.temporal.forecast import LearnedForecaster
    f0 = LearnedForecaster(n_modes=6, seed=0)
    b0 = TemporalBranch(TC.TemporalCfg(on=True, fc_train="joint"), 192, 0).forecaster
    assert all(torch.equal(x, y) for x, y in zip(f0.state_dict().values(), b0.state_dict().values()))
    with pytest.raises(SystemExit, match="resume"):
        PF.main(["--out", str(out), "--device", "cpu", "--epochs", "1", "--max-steps", "2", "--limit-tokens", "32",
                 "--val-limit", "16", "--eval-splits", "navtrain_val", "--train-eval-rows", "0", "--workers", "0"])


# ----------------------------------------------------------------------------------------------- D5 provenance
def _prov(code="c", tr="t", va="v", ca="s", wt="w", fc=None):
    p = {"k_obj": 96, "code": {"train": {"sha256": code}, "worktree": {"sha256": wt}},
         "data": {"packs": {"navtrain_train": {"build_sha256": tr}, "navtrain_val": {"build_sha256": va}},
                  "ca_scales": {"sha256": ca}}}
    if fc is not None:
        p["data"]["fc_init"] = fc
    return p


def test_provenance_compare():
    from tools.ck.v3 import provenance as PV
    fc = {"path": "/x", "sha256": "f", "pretrain_train_pack_sha256": "t", "pretrain_code_sha256": "c"}
    r = PV.compare({"a": _prov(fc=fc), "b": _prov(), "d": _prov()})
    assert r == {"errors": [], "warnings": []}
    assert PV.compare({"a": _prov(), "b": _prov(tr="T2")})["errors"]
    assert PV.compare({"a": _prov(), "b": _prov(code="c2")})["errors"]
    assert PV.compare({"a": _prov(), "b": _prov(ca="s2")})["errors"]
    r = PV.compare({"a": _prov(), "b": _prov(wt="w2")})
    assert not r["errors"] and r["warnings"]                            # worktree sha: warning only
    r = PV.compare({"a": _prov(fc=dict(fc, pretrain_train_pack_sha256="other")), "b": _prov()})
    assert any("pre-trained on train pack" in e for e in r["errors"])
    r = PV.compare({"a": _prov(fc=dict(fc, pretrain_code_sha256="old")), "b": _prov()})
    assert not r["errors"] and r["warnings"]
    # single-run check (train_ck2 refuses a fresh run on it)
    ok = _prov(fc=fc)
    for s_ in ("navtrain_train", "navtrain_val"):
        ok["data"]["packs"][s_]["status"] = "complete"
    assert PV.check_self(ok) == []
    bad = json.loads(json.dumps(ok))
    bad["data"]["fc_init"]["pretrain_train_pack_sha256"] = "other"
    assert PV.check_self(bad)
    bad = json.loads(json.dumps(ok))
    bad["data"]["packs"]["navtrain_val"]["status"] = "failed_truncation"
    assert PV.check_self(bad)
    real = PV.run_provenance(TC.TemporalCfg(on=True, forecaster="ca", n_modes=1))
    assert PV.check_self(real) == []
    assert real["code"]["train"]["n_files"] > 20 and len(real["code"]["train"]["sha256"]) == 64
    assert real["k_obj"] == 96 and "navtrain_val" in real["data"]["packs"]
    assert PV.run_provenance(TC.TemporalCfg())["data"] == {}


# ----------------------------------------------------------------------------------------------- D6 infer check
def test_infer_logit_check_tolerance(tmp_path):
    from tools.ck.v3 import infer_val_v3 as IV
    rng = np.random.default_rng(0)
    N, n = 18179, 400
    rows = np.arange(n)
    tl = rng.normal(0, 3, (n, 16, 5)).astype(np.float32)
    np.savez(tmp_path / "eval_r34_navtrain_val_last.npz", logit=tl, rows=rows, epoch=9, step=100)
    done = np.zeros(N, bool)
    done[rows] = True
    L = np.full((N, 16, 5), np.nan, np.float32)
    L[rows] = tl + rng.normal(0, 1e-5, tl.shape)
    c = IV.logit_check(tmp_path, L, done, 100)
    assert c["ok"] and c["n_rows"] == n and c["q999_abs_dlogit"] < 2e-4
    assert IV.logit_check(tmp_path, L, done, 99)["ok"] is False           # other weights (step) than the reference
    L2 = L.copy()
    L2[rows[:40]] += 0.5                                                  # 10 % of the tokens far off
    c2 = IV.logit_check(tmp_path, L2, done, 100)
    assert not c2["ok"] and c2["q999_abs_dlogit"] > 2e-4
    assert IV.logit_check(tmp_path / "none", L, done, 100) is None
