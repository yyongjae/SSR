"""Tests for the mode-A dead-zone options of stage-T run 2 (PRESTATED_DECISION_RULE AMENDMENT 3 (1), (3)).  CPU, < 1 min.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python -m pytest -q tools/refiner/tests/test_dead_zone.py

- decoder.lon_clamp_a: forward bit-identical to torch.clamp(c, max=0.0) for every value (exact zeros, -0.0, float16 /
  bfloat16 / autocast included); backward 1 (c <= 0) / lam (c > 0) exactly; lam = 0 is torch.clamp's gradient;
- decode(..., lon_st_slope=lam): every output bitwise equal to lam = 0, identity bytes at z = w = 0, modes B / P ignore
  it, z_lon gets a gradient when all c > 0 only with lam > 0;
- decoder.lon_live == (decode()['c_lon'] < 0).any(-1);
- train_refiner: a collision-only loss on a net whose z_lon > 0 everywhere gives zero lon-head gradient with lam = 0 and
  a nonzero one (pushing z down) with lam > 0; the hinge penalty value / gradient; config.json records, resume refusal,
  liveness / zdead log records; argument checks;
- tools/refiner/liveness.py on a synthetic pred.npz (requires --eval; writes only with --out).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(HERE), str(HERE.parent)):
    sys.path.insert(0, p)

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner import decoder as D  # noqa: E402
import liveness as LV  # noqa: E402
import refiner_synth as SY  # noqa: E402
import train_refiner as TR  # noqa: E402

QUIET = lambda *a, **k: None  # noqa: E731
FIX = HERE / "fixtures" / "decoder_navtest_sample.npz"
LAMS = (0.0, 0.1, 1.0)


def _bits(x: torch.Tensor) -> torch.Tensor:
    """Raw bit pattern (so -0.0 != +0.0 and NaN payloads are compared)."""
    ity = {torch.float64: torch.int64, torch.float32: torch.int32, torch.float16: torch.int16,
           torch.bfloat16: torch.int16}[x.dtype]
    return x.contiguous().view(ity)


def _cvals(dtype, n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(n, 8, generator=g, dtype=torch.float64) * 3.0
    c[::7] = 0.0
    c[1::11] = -0.0
    c[2::13, 3] = 1e-30
    c[3::17, 5] = -1e-30
    return c.to(dtype)


# ---------------------------------------------------------------------------------------------------- clamp op
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lam", LAMS)
def test_forward_bit_identical_to_clamp(dtype, lam):
    c = _cvals(dtype)
    assert bool((_bits(c) == _bits(torch.tensor(-0.0, dtype=dtype))).any())            # -0.0 really present
    ref = torch.clamp(c, max=0.0)
    got = D.lon_clamp_a(c, lam)
    assert got.dtype == ref.dtype and torch.equal(_bits(got), _bits(ref))
    got_g = D.lon_clamp_a(c.clone().requires_grad_(True), lam)                          # autograd path, same bytes
    assert torch.equal(_bits(got_g.detach()), _bits(ref))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):                       # autocast region (CPU)
        got_a = D.lon_clamp_a(c.clone().requires_grad_(True), lam)
    assert torch.equal(_bits(got_a.detach()), _bits(ref))


def test_naive_relu_form_is_not_bitwise():
    """Why lon_clamp_a is a custom Function: c - relu(c) + lam (relu(c) - relu(c).detach()) turns -0.0 into +0.0."""
    c = torch.tensor([-0.0, 0.0, -1.0, 2.0])
    pos = torch.relu(c)
    naive = c - pos + 0.5 * (pos - pos.detach())
    assert not torch.equal(_bits(naive), _bits(torch.clamp(c, max=0.0)))
    assert torch.equal(_bits(D.lon_clamp_a(c, 0.5)), _bits(torch.clamp(c, max=0.0)))


@pytest.mark.parametrize("lam", LAMS)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16])
def test_gradient_values(lam, dtype):
    c = _cvals(dtype, 500, seed=1)
    c[0, :3] = torch.tensor([0.0, -0.0, float("nan")], dtype=dtype)
    c = c.requires_grad_(True)
    g = torch.randn(c.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64).to(dtype)
    D.lon_clamp_a(c, lam).backward(g)
    cd = c.detach()
    exp = torch.where(cd > 0, g * lam, torch.where(cd <= 0, g, torch.zeros_like(g)))
    assert torch.equal(c.grad, exp)
    assert torch.equal(c.grad[cd < 0], g[cd < 0])                                       # 1 on the live side
    assert torch.equal(c.grad[cd == 0], g[cd == 0])                                     # boundary (+-0) passes
    assert float(c.grad[0, 2]) == 0.0                                                   # NaN: as torch.clamp
    # the torch.clamp gradient (run 1) == lam = 0
    c2 = cd.clone().requires_grad_(True)
    torch.clamp(c2, max=0.0).backward(g)
    if lam == 0.0:
        assert torch.equal(c.grad, c2.grad)
    assert torch.equal(c2.grad[cd <= 0], c.grad[cd <= 0])


def test_slope_argument_checked():
    c = torch.zeros(2, 8)
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            D.lon_clamp_a(c, bad)


# ---------------------------------------------------------------------------------------------------- decode
@pytest.fixture(scope="module")
def taus():
    fx = dict(np.load(FIX, allow_pickle=True))
    return torch.as_tensor(np.concatenate([fx["tau_h"], fx["tau_student"]])[:120])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("lam", (0.1, 1.0))
def test_decode_identity_bytes_with_slope(taus, dtype, lam):
    tau = taus.to(dtype)
    z = torch.zeros(len(tau), 6, dtype=dtype)
    o = D.decode(tau, z, z, None, "A", lon_st_slope=lam)
    assert torch.equal(_bits(o["traj"]), _bits(tau))
    zr = z.clone().requires_grad_(True)
    o2 = D.decode(tau, zr, z, None, "A", lon_st_slope=lam)
    assert torch.equal(_bits(o2["traj"].detach()), _bits(tau))


@pytest.mark.parametrize("mode", ["A", "B", "P"])
def test_decode_outputs_independent_of_slope(taus, mode):
    tau = taus[:60].double()
    g = torch.Generator().manual_seed(3)
    z = 0.8 * torch.randn(len(tau), 6, generator=g, dtype=torch.float64)
    w = 0.3 * torch.randn(len(tau), 6, generator=g, dtype=torch.float64)
    v0 = torch.full((len(tau),), 6.0, dtype=torch.float64)
    outs, grads = [], []
    for lam in LAMS:
        zr, wr = z.clone().requires_grad_(True), w.clone().requires_grad_(True)
        o = D.decode(tau, zr, wr, v0, mode, lon_st_slope=lam)
        (o["traj"] * torch.randn(o["traj"].shape, generator=torch.Generator().manual_seed(4),
                                 dtype=torch.float64)).sum().backward()
        outs.append(o)
        grads.append((zr.grad.clone(), wr.grad.clone()))
    for o in outs[1:]:
        for k, v in outs[0].items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(_bits(o[k].detach()), _bits(v.detach())), k
        for k, v in outs[0]["flags"].items():
            assert torch.equal(o["flags"][k], v), k
    assert all(torch.equal(gr[1], grads[0][1]) for gr in grads)                         # lateral gradient unchanged
    if mode in ("B", "P"):                                                              # ignored outside mode A
        assert all(torch.equal(gr[0], grads[0][0]) for gr in grads)


def test_decode_dead_zone_gradient(taus):
    """All c > 0 (z > 0 everywhere): no z gradient with lam = 0 (run 1's dead zone), a nonzero one with lam > 0 that
    scales linearly with lam (the clamped part is the only path from z)."""
    tau = taus[:40].double()
    S = D.DraftPath(tau).knots()
    moving = (S[:, 1:] - S[:, :-1]).min(1).values > 0.5
    z0 = torch.full((len(tau), 6), 0.4, dtype=torch.float64)
    w = torch.zeros(len(tau), 6, dtype=torch.float64)
    assert not D.lon_live(z0).any()
    gw = torch.randn(len(tau), 8, 2, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    grads = {}
    for lam in LAMS:
        z = z0.clone().requires_grad_(True)
        o = D.decode(tau, z, w, None, "A", lon_st_slope=lam)
        (o["traj"][..., :2] * gw).sum().backward()
        grads[lam] = z.grad
    assert torch.equal(grads[0.0], torch.zeros_like(z0))
    assert (grads[1.0][moving].abs().sum(1) > 0).all()
    torch.testing.assert_close(grads[0.1], 0.1 * grads[1.0], rtol=1e-12, atol=1e-15)


def test_lon_live_matches_decode(taus):
    tau = taus[:80].float()
    g = torch.Generator().manual_seed(6)
    z = torch.randn(len(tau), 6, generator=g)
    z[:10] = torch.rand(10, 6, generator=g) + 0.01        # all > 0: never live
    z[10:20] = 0.0                                        # identity: c == 0, not live
    z[20:30, 0] = -1e-3                                    # a single slightly negative control: live
    z[20:30, 1:] = 0.0
    o = D.decode(tau, z, torch.zeros_like(z), None, "A")
    live = D.lon_live(z)
    assert torch.equal(live, (o["c_lon"] < 0).any(-1))
    assert not live[:20].any() and live[20:30].all()
    assert torch.equal(D.lon_live(z.reshape(8, 10, 6)), live.reshape(8, 10))           # any leading shape
    assert torch.equal(D.lon_live(z.numpy()), live)


# ---------------------------------------------------------------------------------------------------- training
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dead_zone")
    df, src = SY.make_sources(tmp, 10)
    RD.pack_split("train", df, tmp / "packed", sources=src, workers=1, log_fn=QUIET)
    teacher = SY.make_fake_teacher(tmp, df.token)
    return tmp, df, teacher


def _batch(env, rows=(8, 9)):
    tmp = env[0]
    P = RD.PackedSplit("train", tmp / "packed")
    return RD.collate_tokens([RD.TokenDataset(P, list(rows), None)[i] for i in range(len(rows))])


@pytest.mark.parametrize("sur", ["real", "stub"])
def test_training_step_collision_gradient(env, sur):
    """Net with z_lon = +0.5 on every draft (all c > 0): a collision-only loss (col = 1, the rest 0, no gate) gives
    lon-head gradients exactly 0 with lam = 0 and nonzero with lam = 1, pushing z_lon down (toward deceleration)."""
    b = _batch(env)
    T, K = b["tau0"].shape[:2]
    fn = TR.resolve_surrogate(sur)[0]
    W = {"col": 1.0, "dac": 0.0, "prog": 0.0, "cmf": 0.0, "mod": 0.0}
    res = {}
    for lam in (0.0, 1.0):
        net = TR.build_net("none", 0)
        with torch.no_grad():
            net.lon_head[-1].bias.fill_(0.5)
        out = net(None, b["tau0"], b["v0"], b["a0"], b["eds"], b["cmd"])
        assert bool((out["z_lon"] > 0).all()) and not D.lon_live(out["z_lon"].detach()).any()
        out["z_lon"].retain_grad()
        loss, st, dec = TR.compute_loss(out, b, fn, W, 1.0, "A", False, 0.0, lon_st_slope=lam)
        loss.backward()
        res[lam] = (float(loss.detach()), st, out["z_lon"].grad.clone(), net.lon_head[-1].bias.grad.clone())
        assert st["live"] == 0.0 and st["t_col"] > 1e-3                              # the cone is hit, nobody brakes
    (l0, s0, gz0, gb0), (l1, s1, gz1, gb1) = res[0.0], res[1.0]
    assert l0 == l1 and s0 == s1                                                      # forward identical
    assert torch.equal(gz0, torch.zeros_like(gz0)) and torch.equal(gb0, torch.zeros_like(gb0))
    assert float(gz1.abs().sum()) > 0 and float(gb1.abs().sum()) > 0
    assert float(gz1.sum()) > 0                                                       # descent lowers z -> brakes


def test_zdead_penalty_value_and_gradient(env):
    b = _batch(env)
    T, K = b["tau0"].shape[:2]
    g = torch.Generator().manual_seed(7)
    z = torch.randn(T, K, 6, generator=g).requires_grad_(True)
    valid = b["draft_valid"]
    m = valid.reshape(-1).float()
    zf = z.detach().reshape(-1, 6)
    exp = float(((torch.relu(zf) ** 2).mean(1) * m).sum() / m.sum())
    assert float(TR.zdead_penalty(z.detach(), valid)) == pytest.approx(exp, rel=1e-6)
    assert float(TR.zdead_penalty(-z.detach().abs(), valid)) == 0.0
    assert float(TR.zdead_penalty(z.detach(), torch.zeros_like(valid))) == 0.0
    out = {"z_lon": z, "w_lat": torch.zeros(T, K, 6), "gate_logit": torch.zeros(T, K)}
    W = dict(TR.DEFAULT_W)
    l0, s0, _ = TR.compute_loss(out, b, TR.stub_terms, W, 1.0)
    l1, s1, _ = TR.compute_loss(out, b, TR.stub_terms, W, 1.0, w_zdead=0.01)
    assert s0["zdead"] == s1["zdead"] == pytest.approx(exp, rel=1e-6)
    assert s1["loss_ex_zd"] == s0["loss"] == s0["loss_ex_zd"]
    assert float(l1) == pytest.approx(float(l0) + 0.01 * s1["zdead"], rel=1e-6)
    # gradient of the hinge alone: w * 2 relu(z) / 6 / n_valid on valid drafts
    z.grad = None
    (TR.zdead_penalty(z, valid) * 0.01).backward()
    gexp = 0.01 * 2 * torch.relu(z.detach()) / 6 / m.sum() * valid[..., None].float()
    torch.testing.assert_close(z.grad, gexp)


def _args(tmp, teacher, extra=()):
    a = ["--arm", "none", "--fold", "0", "--seed", "0", "--gpu", "-1", "--packed-root", str(tmp / "packed"),
         "--runs", str(tmp / "runs"), "--teacher-root", str(teacher), "--tokens-per-batch", "3", "--workers", "0",
         "--n-norm", "4", "--log-every", "1", "--inner-val-frac", "0.3", "--max-steps", "1", "--epochs", "1",
         "--max-val-batches", "1", "--surrogate", "stub", *extra]
    return TR.get_parser().parse_args(a)


def test_defaults_are_off():
    a = TR.get_parser().parse_args(["--arm", "none"])
    assert a.lon_st_slope == 0.0 and a.w_zdead == 0.0


def test_train_records_logs_and_refuses_mismatch(env):
    tmp, _, teacher = env
    run = TR.train(_args(tmp, teacher, ["--lon-st-slope", "1.0", "--w-zdead", "0.01", "--tag", "dz"]))
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["lon_st_slope"] == 1.0 and cfg["w_zdead"] == 0.01
    log = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    st = [r for r in log if r["kind"] == "step"][0]
    assert {"zdead", "live", "loss_ex_zd"} <= set(st) and st["loss"] >= st["loss_ex_zd"]
    ep = [r for r in log if r["kind"] == "epoch"][0]
    assert 0.0 <= ep["val_liveness"] <= 1.0 and 0.0 <= ep["train_liveness"] <= 1.0
    assert ep["val_loss"] == pytest.approx(ep["val"]["loss"], abs=1e-6) and "live" in ep["val"] and "zdead" in ep["val"]
    for extra in (["--lon-st-slope", "0.1", "--w-zdead", "0.01"], ["--lon-st-slope", "1.0"], []):
        with pytest.raises(SystemExit):
            TR.train(_args(tmp, teacher, [*extra, "--tag", "dz"]))
    TR.train(_args(tmp, teacher, ["--lon-st-slope", "1", "--w-zdead", "0.01", "--tag", "dz"]))       # same: resumes
    # a run-1 style config.json (no dead-zone keys) resumes with the defaults and refuses a slope
    run0 = TR.train(_args(tmp, teacher, ["--tag", "dz0"]))
    c0 = json.loads((run0 / "config.json").read_text())
    assert c0["lon_st_slope"] == 0.0 and c0["w_zdead"] == 0.0
    for k in ("lon_st_slope", "w_zdead"):
        c0.pop(k)
    (run0 / "config.json").write_text(json.dumps(c0))
    TR.train(_args(tmp, teacher, ["--tag", "dz0"]))
    with pytest.raises(SystemExit):
        TR.train(_args(tmp, teacher, ["--lon-st-slope", "1", "--tag", "dz0"]))


def test_finished_run_refuses_mismatch(env):
    """A finished (DONE) run-1 style directory: the same call exits cleanly (returns, nothing rewritten); other
    dead-zone options / margins raise SystemExit BEFORE the DONE early return (so stageT_gpu_commands.sh stops under
    set -e instead of going on to overwrite the run's eval outputs)."""
    tmp, _, teacher = env
    run = TR.train(_args(tmp, teacher, ["--tag", "dzdone"]))
    c0 = json.loads((run / "config.json").read_text())
    for k in ("lon_st_slope", "w_zdead"):
        c0.pop(k)
    (run / "config.json").write_text(json.dumps(c0))
    (run / "DONE").write_text("x")
    before = {f.name: f.read_bytes() for f in run.iterdir() if f.is_file()}
    assert TR.train(_args(tmp, teacher, ["--tag", "dzdone"])) == run                  # same options: DONE, exit 0
    for extra in (["--lon-st-slope", "1.0", "--w-zdead", "0.1"], ["--lon-st-slope", "1.0"], ["--w-zdead", "0.1"],
                  ["--m-col", "0.5"]):
        with pytest.raises(SystemExit, match="config.json has"):
            TR.train(_args(tmp, teacher, [*extra, "--tag", "dzdone"]))
    assert {f.name: f.read_bytes() for f in run.iterdir() if f.is_file()} == before    # nothing touched


def test_gpu_script_refuses_new_flags_with_default_tag(tmp_path):
    """stageT_gpu_commands.sh: --lon-st-slope / --w-zdead with TAG unset (= stageT, the run-1 tag) is refused before
    anything is run or written."""
    src = (Path(TR.__file__).resolve().parent / "stageT_gpu_commands.sh").read_text()
    # sandboxed copy: DATA -> tmp, PY -> a recorder, so a broken guard can never reach the real run directories
    rec = tmp_path / "called"
    py = tmp_path / "py.sh"
    py.write_text(f"#!/usr/bin/env bash\necho \"$@\" >> {rec}\n")
    py.chmod(0o755)
    lines = []
    for ln in src.splitlines():
        if ln.startswith("DATA="):
            ln = f"DATA={tmp_path / 'data'}"
        elif ln.startswith("PY="):
            ln = f"PY={py}"
        lines.append(ln)
    assert sum(x.startswith(("DATA=", "PY=")) for x in src.splitlines()) == 2
    sh = tmp_path / "gpu.sh"
    sh.write_text("\n".join(lines) + "\n")
    env = {k: v for k, v in os.environ.items() if k != "TAG"}
    for extra in (["--lon-st-slope", "1.0"], ["--w-zdead=0.1"]):
        r = subprocess.run(["bash", str(sh), "0", "none", "0", "0", *extra], env=env, capture_output=True, text=True)
        assert r.returncode == 2 and "TAG=stageT" in r.stderr
    assert not rec.exists() and not (tmp_path / "data").exists()
    r = subprocess.run(["bash", str(sh), "0", "none", "0", "0", "--lon-st-slope", "1.0"], env={**env, "TAG": "run2"},
                       capture_output=True, text=True)
    assert r.returncode == 0 and "--tag run2 --lon-st-slope 1.0" in rec.read_text()


@pytest.mark.parametrize("extra", [["--lon-st-slope", "-1"], ["--w-zdead", "-0.1"], ["--w-zdead", "nan"],
                                   ["--lon-st-slope", "1", "--mode", "B"]])
def test_bad_dead_zone_args_refused(env, extra):
    tmp, _, teacher = env
    with pytest.raises(SystemExit):
        TR.train(_args(tmp, teacher, [*extra, "--tag", "bad"]))


# ---------------------------------------------------------------------------------------------------- liveness.py
def _pred(tmp: Path, name="eval_train_fold0"):
    N, K = 4, 13
    g = np.random.default_rng(0)
    z = np.abs(g.normal(size=(N, K, 6))).astype(np.float32) + 0.05          # all > 0: dead
    z[0, :3] = -0.5                                                         # 3 live drafts (token 0, k 0..2)
    z[1, 0, 0] = -0.3                                                       # 1 live draft with mixed signs (c_2 < 0)
    valid = np.ones((N, K), bool)
    valid[3] = False
    valid[0, 2] = False                                                     # a live but invalid draft
    fam = np.tile(np.arange(K) % 9, (N, 1)).astype(np.int8)
    t = 0.5 * np.arange(1, 9)
    tau0 = np.zeros((N, K, 8, 3), np.float32)
    tau0[..., 0] = 10.0 * t
    tau1 = tau0.copy()
    tau1[0, 0, :, 0] = 10.0 * t - 0.2 * t                                   # shortened by 0.8 m at 4 s
    tau1[0, 1, :, 0] = 10.0 * t - 0.1 * t                                   # 0.4 m: below the threshold
    d = tmp / "run" / name
    d.mkdir(parents=True)
    np.savez(d / "pred.npz", tokens=np.array([f"t{i}" for i in range(N)]), z_lon=z, draft_valid=valid, family=fam,
             tau0=tau0, tau1=tau1, p_g=np.zeros((N, K), np.float32))
    return tmp / "run", z, valid


def test_liveness_tool(tmp_path, capsys):
    run, z, valid = _pred(tmp_path)
    n_valid = int(valid.sum())
    res = LV.main(["--run", str(run), "--eval", "eval_train_fold0"])
    assert json.loads(capsys.readouterr().out)["liveness"] == res["liveness"]
    assert sorted(p.name for p in (run / "eval_train_fold0").iterdir()) == ["pred.npz"]   # nothing written
    assert res["n_valid"] == n_valid and res["n_live"] == 3                              # (0,0), (0,1), (1,0)
    assert res["liveness"] == pytest.approx(3 / n_valid)
    assert res["frac_zlon_all_pos"] == pytest.approx((n_valid - 3) / n_valid)
    assert res["frac_short_gt_0p5m"] == pytest.approx(1 / n_valid)
    assert res["mean_short_m"] == pytest.approx((0.8 + 0.4) / n_valid, rel=1e-5)
    fam = res["by_family"]
    assert fam["identity"]["n_live"] == 2 and fam["small"]["n_live"] == 1 and fam["lconst"]["n_live"] == 0
    assert sum(f["n"] for f in fam.values()) == n_valid
    assert res["gate_pass"] and res["min_liveness"] == 0.01
    assert not LV.main(["--run", str(run), "--eval", "eval_train_fold0", "--min-liveness", "0.2"])["gate_pass"]
    out = tmp_path / "o" / "live.json"
    LV.main(["--run", str(run), "--eval", "eval_train_fold0", "--out", str(out)])
    assert json.loads(out.read_text())["n_live"] == 3
    with pytest.raises(SystemExit):                                                       # --eval is required
        LV.main(["--run", str(run)])
    # the tool's liveness == decode()['c_lon'] < 0 on the same float32 controls
    zt = torch.as_tensor(z.reshape(-1, 6))
    tau = torch.as_tensor(np.zeros((len(zt), 8, 3), np.float32) + np.stack(
        [10.0 * 0.5 * np.arange(1, 9), np.zeros(8), np.zeros(8)], -1).astype(np.float32))
    o = D.decode(tau, zt, torch.zeros_like(zt), None, "A")
    assert int(((o["c_lon"] < 0).any(-1).reshape(valid.shape) & torch.as_tensor(valid)).sum()) == 3
