"""Online KD calibration (ck_e2e2.kd_calib.mode 'online', user 2026-11-02 option B; navsim ck/kd_calib.OnlineKDCalib):
the weighted / binned Platt fit equals the plain fit, the refit schedule (slots, EMA decay, on-policy reset, empty
window, identity before the first refit), the state round trip (weights_only), config / launcher rules, the teachers'
extra no-grad pass on the label set leaving every KD output bit-identical, the student loss before the first refit
bit-identical to the uncalibrated run, and the (logit, label) pairs taken from the right candidates (warm-up: the KD
set; on-policy: the label set of rows WITH a generation only).  CPU only:
  cd /workspace/yongjae/SSR-v7c && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_kd_calib_online.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import kd_calib as KC  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck.constants import CK_KEYS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
need_inputs = pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")
ONLINE = {"enabled": True, "mode": "online", "det": None, "map": None, "min_n": 10.0}
J = {k: i for i, k in enumerate(CK_KEYS)}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pairs(n, ab, seed, soft=True):
    """z [n, 5] ~ N(0, 2), y [n, 5] = sigmoid(a_k z_k + b_k) (soft) or Bernoulli of it, per key (ab: {key: (a, b)},
    other keys a = 1, b = 0)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, (n, len(CK_KEYS)))
    y = np.empty_like(z)
    for k, j in J.items():
        a, b = ab.get(k, (1.0, 0.0))
        p = _sig(a * z[:, j] + b)
        y[:, j] = p if soft else (rng.random(n) < p)
    return torch.from_numpy(z).float(), torch.from_numpy(y).float()


# ----------------------------------------------------------------------------------------------- numerics
def test_weighted_fit_equals_repeated_samples_and_unweighted_path_unchanged():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 3000)
    y = (rng.random(3000) < _sig(0.7 * z + 0.5)).astype(np.float64)
    w = rng.integers(1, 4, 3000)
    rep = KC.fit_platt(np.repeat(z, w), np.repeat(y, w))
    wf = KC.fit_platt_weighted(z, y, w)
    assert abs(rep["a"] - wf["a"]) < 1e-8 and abs(rep["b"] - wf["b"]) < 1e-8
    assert abs(rep["nll"] - wf["nll"]) < 1e-10 and wf["n"] == float(w.sum())
    one = KC.fit_platt_weighted(z, y, np.ones_like(z))
    plain = KC.fit_platt(z, y)
    assert abs(one["a"] - plain["a"]) < 1e-10 and abs(one["b"] - plain["b"]) < 1e-10
    # zero weights are dropped, labels outside [0, 1] refused
    w0 = np.where(np.arange(3000) < 1500, 1.0, 0.0)
    h = KC.fit_platt_weighted(z, y, w0)
    p = KC.fit_platt(z[:1500], y[:1500])
    assert abs(h["a"] - p["a"]) < 1e-10 and abs(h["b"] - p["b"]) < 1e-10
    with pytest.raises(ValueError, match="0, 1"):
        KC.fit_platt_weighted(np.array([0.0, 1.0]), np.array([0.0, 1.5]), np.array([1.0, 1.0]))


def test_binned_online_fit_recovers_the_exact_fit_per_teacher_and_key():
    co = KC.OnlineKDCalib(min_n=10)
    assert co.fit == {"det": ["nc", "ep", "ttc"], "map": ["dac", "ep"]}
    true = {"det": {"nc": (0.8, 1.5), "ep": (0.6, 0.6), "ttc": (0.65, 1.5)}, "map": {"dac": (0.4, 1.6), "ep": (0.6, 0.7)}}
    raw = {}
    for s, t in enumerate(("det", "map")):
        z, y = _pairs(200_000, true[t], seed=s, soft=False)
        m = torch.ones(200_000, dtype=torch.bool)
        m[:1000] = False                                         # masked candidates are not counted
        assert co.add(t, z, y, m) == 199_000 * len(co.fit[t])
        raw[t] = (z[1000:].double().numpy(), y[1000:].double().numpy())
    rec = co.refit("cpu", "warmup", reduce=False)
    assert rec["refit"] == 1 and rec["reset"] is False and co.version == 1
    for i, t in enumerate(("det", "map")):
        for k in CK_KEYS:
            j = J[k]
            if k in co.fit[t]:
                exact = KC.fit_platt(raw[t][0][:, j], raw[t][1][:, j])
                assert abs(co.a[i, j] - exact["a"]) < 2e-3 and abs(co.b[i, j] - exact["b"]) < 2e-3, (t, k)
                assert abs(co.a[i, j] - true[t][k][0]) < 0.03 and abs(co.b[i, j] - true[t][k][1]) < 0.03, (t, k)
                r = rec["keys"][f"{t}/{k}"]
                assert r["fitted"] and r["nll_new_win"] <= r["nll_raw_win"] + 1e-12
                assert r["a_prev"] == 1.0 and r["b_prev"] == 0.0 and r["nll_prev_win"] == r["nll_raw_win"]
            else:                                                 # not a KD key of this teacher: identity
                assert co.a[i, j] == 1.0 and co.b[i, j] == 0.0 and f"{t}/{k}" not in rec["keys"]
    a, b = co.params("det")
    assert a.dtype == np.float32 and a[J["dac"]] == 1.0 and b[J["comfort"]] == 0.0


def test_refit_schedule_ema_reset_and_empty_window():
    co = KC.OnlineKDCalib(refits_per_epoch=4, ema_epochs=0.5, min_n=10)
    phase = lambda e: "onpolicy" if e >= 2 else "warmup"          # noqa: E731
    decay = math.exp(-1.0 / (4 * 0.5))
    assert abs(co.decay - decay) < 1e-15
    assert co.maybe_refit(0.0, "cpu", phase, reduce=False) is None and co.slot == 0
    assert (co.a == 1).all() and (co.b == 0).all() and co.version == 0      # identity before the first refit
    za, ya = _pairs(20_000, {"nc": (1.0, 1.0)}, 0)
    co.add("det", za, ya, torch.ones(20_000, dtype=torch.bool))
    assert co.maybe_refit(0.2, "cpu", phase, reduce=False) is None           # same slot
    r1 = co.maybe_refit(0.25, "cpu", phase, reduce=False)
    assert r1["window_slot"] == 0 and r1["window_phase"] == "warmup" and r1["keys"]["det/nc"]["fitted"]
    win_a = co.ema.copy()
    zb, yb = _pairs(20_000, {"nc": (0.5, -1.0)}, 1)
    co.add("det", zb, yb, torch.ones(20_000, dtype=torch.bool))
    r2 = co.maybe_refit(0.5, "cpu", phase, reduce=False)
    win_b = co.ema - decay * win_a
    assert np.allclose(win_b[0, J["nc"], :, 0].sum(), 20_000) and r2["refit"] == 2 and co.version == 2
    # the out-of-sample record of window B uses the parameters fitted on window A
    assert r2["keys"]["det/nc"]["a_prev"] == r1["keys"]["det/nc"]["a"]
    # empty window: nothing changes
    a, b, ema, v = co.a.copy(), co.b.copy(), co.ema.copy(), co.version
    r3 = co.maybe_refit(0.75, "cpu", phase, reduce=False)
    assert r3["skipped"] == "empty window" and np.array_equal(co.ema, ema) and np.array_equal(co.a, a)
    assert np.array_equal(co.b, b) and co.version == v and co.n_refits == 2
    # the window that ends at the epoch-2 boundary is still warm-up; the first on-policy window resets the EMA
    co.add("det", za, ya, torch.ones(20_000, dtype=torch.bool))
    r4 = co.maybe_refit(2.0, "cpu", phase, reduce=False)
    assert r4["window_phase"] == "warmup" and r4["reset"] is False
    zc, yc = _pairs(30_000, {"nc": (2.0, 0.0)}, 2)
    co.add("det", zc, yc, torch.ones(30_000, dtype=torch.bool))
    r5 = co.maybe_refit(2.25, "cpu", phase, reduce=False)
    assert r5["window_phase"] == "onpolicy" and r5["reset"] is True and co.ema_phase == "onpolicy"
    assert np.allclose(co.ema[0, J["nc"], :, 0].sum(), 30_000)                 # warm-up pairs gone
    assert abs(co.a[0, J["nc"]] - 2.0) < 0.02
    co.add("det", zc, yc, torch.ones(30_000, dtype=torch.bool))
    r6 = co.maybe_refit(2.5, "cpu", phase, reduce=False)
    assert r6["reset"] is False


def test_min_n_keeps_identity_and_state_round_trip(tmp_path):
    co = KC.OnlineKDCalib(min_n=5000)
    z, y = _pairs(1000, {}, 0)
    co.add("map", z, y, torch.ones(1000, dtype=torch.bool))
    rec = co.refit("cpu", "warmup", reduce=False)
    assert not rec["keys"]["map/dac"]["fitted"] and (co.a == 1).all() and co.version == 0
    z, y = _pairs(10_000, {"dac": (0.4, 1.6)}, 1)
    co.add("map", z, y, torch.ones(10_000, dtype=torch.bool))
    co.refit("cpu", "warmup", reduce=False)
    assert rec is not co.last and co.last["keys"]["map/dac"]["fitted"]
    co.slot = 7
    p = tmp_path / "s.pt"
    torch.save(co.state_dict(), p)
    sd = torch.load(p, weights_only=True)
    co2 = KC.OnlineKDCalib(min_n=5000)
    co2.load_state_dict(sd)
    assert np.array_equal(co2.ema, co.ema) and np.array_equal(co2.a, co.a) and np.array_equal(co2.b, co.b)
    assert co2.slot == 7 and co2.n_refits == co.n_refits and co2.ema_phase == "warmup" and co2.version > co.version
    with pytest.raises(ValueError, match="options"):
        KC.OnlineKDCalib(min_n=10).load_state_dict(sd)
    with pytest.raises(ValueError, match="OnlineKDCalib"):
        KC.OnlineKDCalib(bins=8)


# ----------------------------------------------------------------------------------------------- config / launcher
def test_config_rules_and_defaults(tmp_path):
    d = O.CKE2E2Config.from_any({}).kd_calib
    assert d["mode"] == "online" and d["enabled"] is True and d["det"] is None and d["map"] is None
    c = TL.cfg2(tmp_path, kd_calib=ONLINE)
    c.validate()
    assert c.kd_calib["mode"] == "online" and c.kd_calib["det"] is None and c.kd_calib["min_n"] == 10.0
    for bad, msg in (({"det": "/x/a.json"}, "must be null"), ({"mode": "both"}, "mode must be"),
                     ({"bins": 8}, "bins >= 16")):
        with pytest.raises(ValueError, match=msg):
            TL.cfg2(tmp_path, kd_calib=dict(ONLINE, **bad)).validate()
    TL.cfg2(tmp_path, kd_calib=dict(ONLINE, enabled=False)).validate()     # calibration off needs no other key
    with pytest.raises(ValueError, match="unknown"):
        O.CKE2E2Config.from_any({"kd_calib": {"modes": "online"}})
    # the yaml defaults carry the new keys with the dataclass defaults
    sys.path.insert(0, str(REPO))
    from tools.ck.e2e2 import launch_util2 as LU2
    y = LU2.load_ck2_yaml()["kd_calib"]
    assert {k: y[k] for k in ("mode",) + O.KD_CALIB_ONLINE_KEYS} == \
        {k: d[k] for k in ("mode",) + O.KD_CALIB_ONLINE_KEYS}


def test_launcher_record_and_v7_set_file(tmp_path):
    sys.path.insert(0, str(REPO))
    from tools.ck.e2e.launch_util import LaunchError
    from tools.ck.e2e2 import launch_util2 as LU2
    from tools.ck.e2e import launch_util as LU
    kc = O.CKE2E2Config.from_any({"kd_calib": ONLINE}).kd_calib
    rec = LU2.kd_calib_record(kc, {}, str(TL.SMOKE_T), str(TL.SMOKE_M), "last")
    assert rec["mode"] == "online" and rec["det"] is None and rec["online"]["min_n"] == 10.0
    with pytest.raises(LaunchError, match="must be null"):
        LU2.kd_calib_record(dict(kc, det="/x/a.json"), {}, str(TL.SMOKE_T), str(TL.SMOKE_M), "last")
    assert LU2.kd_calib_record({"enabled": False}, {}, "", "", "last") == {"enabled": False}
    # this branch's set file switches the student to the online mode (no file, no r34 candidates)
    ck = LU.apply_sets(LU2.load_ck2_yaml(), LU.read_lines(REPO / "tools/ck/e2e2/v2_v15_v7.set"))
    c = O.CKE2E2Config.from_any(ck)
    assert c.kd_calib["mode"] == "online" and c.kd_calib["det"] is None and c.kd_calib["map"] is None
    flat = dict(LU.flatten(ck))
    assert LU.hydra_value(flat["kd_calib.det"]) == "null" and flat["kd_calib.mode"] == "online"


# ----------------------------------------------------------------------------------------------- teachers
@need_inputs
def test_teacher_extra_pass_leaves_kd_outputs_bit_identical():
    tp = O.TeacherPair2(str(TL.SMOKE_T), str(TL.SMOKE_M), "last", "cpu", amp=False)
    f, t = TL.fresh(2)
    st = f["status_feature"]
    a = torch.from_numpy(np.load(TL.ANCHORS)).float()
    g = torch.Generator().manual_seed(0)
    cand = a[torch.randint(0, 256, (2, 48), generator=g)] + 0.05 * torch.randn(2, 48, 8, 3, generator=g)
    extra = a[torch.randint(0, 256, (2, 40), generator=g)] + 0.05 * torch.randn(2, 40, 8, 3, generator=g)
    args = (t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"])
    T0 = tp.run(*args, cand, st)
    T1 = tp.run(*args, cand, st, extra=extra)
    for k in ("kd_score_prob", "kd_e_lat", "kd_ok", "z_det", "z_map", "ok_det", "ok_map", "p_det", "p_map"):
        assert torch.equal(T0[k], T1[k]), k
    assert "z_det_x" not in T0 and T1["z_det_x"].shape == (2, 40, 5) and T1["ok_map_x"].shape == (2, 40)
    Tx = tp.run(*args, extra, st)
    assert torch.allclose(T1["z_det_x"], Tx["z_det"], atol=1e-6) and torch.allclose(T1["z_map_x"], Tx["z_map"],
                                                                                     atol=1e-6)
    # identity online parameters == no calibration, bit for bit; other parameters: sigmoid(a z + b) per source key
    co = KC.OnlineKDCalib()
    tp.set_calib_arrays({tn: co.params(tn) for tn in KC.CALIB_TEACHERS})
    T2 = tp.run(*args, cand, st)
    assert torch.equal(T2["kd_score_prob"], T0["kd_score_prob"]) and torch.equal(T2["z_det"], T0["z_det"])
    co.a[0, J["nc"]], co.b[0, J["nc"]] = 0.5, 1.0
    co.a[1, J["dac"]], co.b[1, J["dac"]] = 0.4, 1.5
    tp.set_calib_arrays({tn: co.params(tn) for tn in KC.CALIB_TEACHERS})
    T3 = tp.run(*args, cand, st)
    ok = T3["kd_ok"]
    assert torch.allclose(T3["kd_score_prob"][..., J["nc"]][ok], torch.sigmoid(0.5 * T0["z_det"][..., J["nc"]] + 1.0)[ok])
    assert torch.allclose(T3["kd_score_prob"][..., J["dac"]][ok], torch.sigmoid(0.4 * T0["z_map"][..., J["dac"]] + 1.5)[ok])
    assert torch.equal(T3["z_det"], T0["z_det"])                           # raw logits stay raw


# ----------------------------------------------------------------------------------------------- student loss
def _student(c, seed=0):
    torch.manual_seed(seed)
    return O.build_student_ck2(c)


@need_inputs
def test_warmup_loss_identity_before_first_refit_then_refit(tmp_path):
    c_on = TL.cfg2(tmp_path / "on", kd_calib=ONLINE, lat_kd={"start_mb": 0})
    c_off = TL.cfg2(tmp_path / "off", lat_kd={"start_mb": 0})
    ck_on, ck_off = O.CKE2E2(c_on), O.CKE2E2(c_off)
    st_on, st_off = _student(c_on), _student(c_off)
    f, t = TL.fresh(2)
    for ck in (ck_on, ck_off):
        ck.epoch, ck.epoch_frac = 0, 0.0
    l_on, g_on = ck_on.loss(st_on, f, t, TL.fake_predictions(2, seed=0))
    l_off, g_off = ck_off.loss(st_off, *TL.fresh(2), TL.fake_predictions(2, seed=0))
    assert torch.equal(l_on, l_off) and torch.equal(g_on["ck2/kd_score"], g_off["ck2/kd_score"])
    assert float(g_on["ck2/cal_refits"]) == 0 and float(g_on["ck2/cal_det_nc_a"]) == 1.0
    assert float(g_on["ck2/cal_map_dac_b"]) == 0.0 and "ck2/cal_det_dac_a" not in g_on
    n1 = int(g_on["ck2/cal_pairs"])
    S = ck_on.warmup_batch(t, ck_on.tokens_of(t["ck_row"].reshape(-1).long()),
                           TL.fake_predictions(2, seed=0)["trajectory"], torch.device("cpu"))
    assert n1 > 0 and n1 <= int((S["ok_lab"] & S["cand_ok"]).sum()) * 5      # <= 3 DET + 2 MAP keys per candidate
    # next slot: refit on the first window's pairs (warm-up), applied to this micro-batch's KD targets
    ck_on.epoch_frac = 0.1
    _, g2 = ck_on.loss(st_on, *TL.fresh(2), TL.fake_predictions(2, seed=1))
    assert float(g2["ck2/cal_refits"]) == 1 and float(g2["ck2/cal_det_nc_a"]) != 1.0
    lines = (Path(c_on.io_dir) / "kd_calib_online.jsonl").read_text().splitlines()
    rec = json.loads(lines[-1])
    assert rec["window_phase"] == "warmup" and rec["keys"]["det/nc"]["n_win"] > 0 and rec["mb"] == 1
    a, b = ck_on.calib_online.params("det")
    assert float(g2["ck2/cal_det_nc_a"]) == float(a[J["nc"]])
    tp = ck_on.teachers(torch.device("cpu"))
    assert torch.equal(tp.calib["det"][0], torch.from_numpy(a))
    # state in the CKE2E2 checkpoint dict, and the cross-mode refusals
    sd = ck_on.state_dict()
    ck3 = O.CKE2E2(c_on)
    ck3.load_state_dict(sd)
    assert np.array_equal(ck3.calib_online.a, ck_on.calib_online.a) and ck3.calib_online.slot == 1
    with pytest.raises(ValueError, match="no online KD calibration state"):
        O.CKE2E2(c_on).load_state_dict(ck_off.state_dict())
    with pytest.raises(ValueError, match="not 'online'"):
        O.CKE2E2(c_off).load_state_dict(sd)


@need_inputs
def test_onpolicy_pairs_come_from_the_label_set_of_rows_with_a_generation(tmp_path):
    c = TL.cfg2(tmp_path, kd_calib=ONLINE)
    ck = O.CKE2E2(c, max_epochs=30)
    ck._labels = D2.LabelStore2(c.io_dir, n_rows=16, packed=c.warmup["packed"], var_sampling=c.var_sampling,
                                ep_target=c.ep_target)
    TL.make_gen(c.io_dir, 4, [3], 16)
    st = _student(c)
    spy = {}
    cb = ck.current_batch

    def w(*a, **k):
        spy["S"] = cb(*a, **k)
        return spy["S"]
    ck.current_batch = w
    tp = ck.teachers(torch.device("cpu"))
    run = tp.run

    def wr(*a, **k):
        spy["T"] = run(*a, **k)
        spy["extra"] = k.get("extra")
        return spy["T"]
    tp.run = wr
    ck.epoch, ck.epoch_frac = 5, 5.0
    f, t = TL.fresh(2, rows=[3, 7])                      # row 3: generation 4; row 7: none -> GT-based fallback
    loss, logs = ck.loss(st, f, t, TL.fake_predictions(2, seed=3))
    assert torch.isfinite(loss)
    S, T = spy["S"], spy["T"]
    assert spy["extra"] is S["extra"] and T["kd_ok"].shape == (2, 48) and T["z_det_x"].shape == (2, 48, 5)
    assert int(S["src"][0]) == O.SRC_PREV and int(S["src"][1]) == O.SRC_FALLBACK and S["ok_lab"][1].any()
    n_det = int((S["ok_lab"][0] & T["ok_det_x"][0]).sum()) * 3
    n_map = int((S["ok_lab"][0] & T["ok_map_x"][0]).sum()) * 2
    assert int(logs["ck2/cal_pairs"]) == n_det + n_map and n_det > 0          # the fallback row adds nothing
    # the window's histogram holds exactly these pairs (det nc: one count per usable label candidate of row 3)
    w_ = ck.calib_online._win.reshape(2, len(CK_KEYS), -1, 2)
    assert float(w_[0, J["nc"], :, 0].sum()) == n_det / 3 and float(w_[0, J["dac"], :, 0].sum()) == 0.0
    ys = S["y"][0][S["ok_lab"][0] & T["ok_det_x"][0], J["nc"]].double().sum()
    assert abs(float(w_[0, J["nc"], :, 1].sum()) - float(ys)) < 1e-9
    # first on-policy window -> refit with an EMA reset
    ck.epoch_frac = 5.1
    _, logs2 = ck.loss(st, *TL.fresh(2, rows=[3, 7]), TL.fake_predictions(2, seed=4))
    assert ck.calib_online.last["window_phase"] == "onpolicy" and ck.calib_online.last["reset"] is True
    assert float(logs2["ck2/cal_refits"]) == 1


# ----------------------------------------------------------------------------------------------- DDP (gloo, CPU)
def _ddp_worker(rank, world, init, out):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    co = KC.OnlineKDCalib(min_n=10)
    z, y = _pairs(30_000, {"nc": (0.7, 1.2), "ttc": (0.5, 1.5)}, seed=10 + rank, soft=False)
    co.add("det", z, y, torch.ones(30_000, dtype=torch.bool))
    phase = lambda e: "warmup"                                       # noqa: E731
    co.maybe_refit(0.0, "cpu", phase)
    rec = co.maybe_refit(0.1, "cpu", phase)                          # all-reduce of the two ranks' windows
    torch.save({"a": torch.from_numpy(co.a), "b": torch.from_numpy(co.b), "n": rec["keys"]["det/nc"]["n_win"]},
               f"{out}/r{rank}.pt")
    dist.destroy_process_group()


def test_ddp_refit_is_identical_on_every_rank_and_uses_all_pairs(tmp_path):
    import torch.multiprocessing as mp
    init = f"file://{tmp_path}/pg"
    mp.spawn(_ddp_worker, args=(2, init, str(tmp_path)), nprocs=2, join=True)
    r0, r1 = (torch.load(tmp_path / f"r{r}.pt", weights_only=True) for r in (0, 1))
    assert torch.equal(r0["a"], r1["a"]) and torch.equal(r0["b"], r1["b"]) and r0["n"] == r1["n"] == 60_000
    co = KC.OnlineKDCalib(min_n=10)                                  # single process on the union = the same fit
    for r in (0, 1):
        z, y = _pairs(30_000, {"nc": (0.7, 1.2), "ttc": (0.5, 1.5)}, seed=10 + r, soft=False)
        co.add("det", z, y, torch.ones(30_000, dtype=torch.bool))
    co.refit("cpu", "warmup", reduce=False)
    assert np.allclose(r0["a"].numpy(), co.a, atol=1e-12) and np.allclose(r0["b"].numpy(), co.b, atol=1e-12)
