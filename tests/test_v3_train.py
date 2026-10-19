"""refiner-v3 pipeline integration tests (docs/refiner_v3/SPEC.md 5.2-5.3, 4.7; Integrate phase), CPU only.

Synthetic (in-memory CK2 data of tests/test_ck2_train.py + synthetic object rows through the real
obj_pack.augment_objects):
  * arms a / b / d each run 3 CPU train steps (train_ck2.train): config.json temporal / privileged / parameter
    counts, forecast stats and grad_norm_fc only in arm a, privileged val rows only in d, strict load_ck (d needs
    allow_privileged), forecaster param group;
  * the model sees only its inputs in arms a and b: perturbing fc_xyh / fc_state, lead flags, anc_* / var_*
    metadata, labels and gt_traj leaves the forward bitwise unchanged (while perturbing obj_in does change it);
  * resume refuses a changed temporal cfg; an old config.json without 'temporal' resumes as OFF only;
  * privileged run-name rule, ca-scales presence, obj_ok threshold, dummy_backward with a forecast, param groups;
  * 2-process gloo DDP == 1 process for arm a.
Real data (skipped when absent): batch streams of arms a and b identical (same bytes incl. augmentation) on the
navtrain_val pack; OFF datasets emit no new keys and never call the augmentation; aug touches only the object
inputs; CKDataset refuses aug; a pack shorter than the dataset rows is refused; the oracle's privileged keys.

  cd /workspace/yongjae/SSR-v3 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-v3 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 /venv/ssr/bin/python -B -m pytest -q -p no:cacheprovider \
    tests/test_v3_train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
CK = str(Path(__file__).resolve().parents[1])
if sys.path[0] != CK:
    sys.path.insert(0, CK)
sys.path.insert(1, str(Path(__file__).resolve().parent))

import json  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import test_ck2_train as T2T  # noqa: E402
from navsim.agents.para_ssr.ck.temporal import config as TCF  # noqa: E402
from tools.ck import ckutil as U  # noqa: E402
from tools.ck import train_ck2 as T2  # noqa: E402
from tools.ck.data import obj_pack as OP  # noqa: E402
from tools.ck.data.ck_dataset import collate_ck  # noqa: E402

REAL_PACK = TCF.PACK_ROOT / "bevfusion" / "navtrain_val"
SMOKE_ROOT = TCF.D3 / "test_tmp" / "objpack_k96_limit256"     # built on demand (a --limit 256 train pack, K_OBJ 96)
have_val_pack = pytest.mark.skipif(not (REAL_PACK / "build.json").is_file(), reason="navtrain_val object pack absent")


@pytest.fixture(autouse=True)
def _threads(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------------- synthetic objects
def fake_objects(r: int, n_agents: int = 5, seed: int = 0) -> dict:
    """Causal object rows of a synthetic token (t0 / t-0.5 / t-1.0 constant velocity, one VRU, two statics, t-1.0
    missing for every third row) + constant-velocity GT targets of the agent slots (last 2 steps BEYOND for slot 0)."""
    rng = np.random.default_rng([seed, int(r), 7])
    cls = np.array([0] * (n_agents - 1) + [1, 3, 6])
    n = len(cls)
    c0 = np.stack([rng.uniform(4, 30, n), rng.uniform(-5, 5, n)], 1)
    v = np.stack([rng.uniform(-2, 8, n), rng.normal(0, 0.5, n)], 1)
    v[cls >= 3] = 0.0
    h = np.arctan2(v[:, 1], v[:, 0] + 1e-3)
    size = {0: (4.6, 2.0), 1: (0.8, 0.8), 3: (0.4, 0.4), 6: (1.0, 1.0)}
    obj = np.zeros((3, TCF.K_OBJ, TCF.D_OBJ), np.float32)
    mask = np.zeros((3, TCF.K_OBJ), bool)
    fv = np.ones(3, bool)
    if r % 3 == 2:
        fv[2] = False
    for f, age in enumerate((0.0, 0.5, 1.0)):
        if not fv[f]:
            continue
        c = c0 - v * age
        obj[f, :n, 0:2] = c
        obj[f, :n, 2], obj[f, :n, 3] = np.cos(h), np.sin(h)
        obj[f, :n, 4] = [size[int(k)][0] for k in cls]
        obj[f, :n, 5] = [size[int(k)][1] for k in cls]
        obj[f, :n, 6:8] = v
        obj[f, :n, 8] = 0.8
        obj[f, np.arange(n), 9 + cls] = 1.0
        obj[f, :n, 16] = age
        mask[f, :n] = True
    t = 0.5 * np.arange(1, 11)
    fc_xyh = np.full((TCF.K_OBJ, 10, 3), np.nan, np.float32)
    fc_state = np.zeros((TCF.K_OBJ, 10), np.int8)
    for i in range(n):
        if cls[i] <= 2:
            fc_xyh[i, :, 0:2] = c0[i] + v[i] * t[:, None]
            fc_xyh[i, :, 2] = h[i]
            fc_state[i] = TCF.FC_OBS
    fc_state[0, 8:] = TCF.FC_BEYOND
    fc_xyh[0, 8:] = np.nan
    return dict(obj_in=obj, obj_mask=mask, frame_valid=fv, obj_ok=np.array(True), fc_xyh=fc_xyh, fc_state=fc_state)


def _add_objects(it: dict, row: int, tcfg, aug: bool, epoch: int, seed: int, n_agents: int = 5) -> dict:
    o = fake_objects(row, n_agents)
    if aug:
        o = OP.augment_objects(o, it["token"], epoch, seed, tcfg)
    if tcfg.forecaster == "oracle_gt":
        o["priv_fc_xyh"], o["priv_fc_state"] = o["fc_xyh"].copy(), o["fc_state"].copy()
    it.update(o)
    return it


class FakeCK2T(T2T.FakeCK2):
    def __init__(self, tcfg, aug=True, n_agents=5, **kw):
        super().__init__(**kw)
        self.tcfg, self.aug, self.n_agents = tcfg, aug, n_agents

    def __getitem__(self, i):
        it = super().__getitem__(i)
        return _add_objects(it, int(self.rows[i]), self.tcfg, self.aug, self.epoch, self.seed, self.n_agents)


class FakeR34T(T2T.FakeR34):
    def __init__(self, tcfg, n_agents=5, **kw):
        super().__init__(**kw)
        self.tcfg, self.n_agents = tcfg, n_agents

    def __getitem__(self, i):
        return _add_objects(super().__getitem__(i), i, self.tcfg, False, 0, 0, self.n_agents)


def _scales(tmp_path) -> str:
    p = Path(tmp_path) / "ca_scales_test.json"
    if not p.is_file():
        b = {g: [round(0.1 + 0.25 * k, 3) for k in range(10)] for g in ("vehicle", "vru")}
        p.write_text(json.dumps({"version": "test", "split": "synthetic", "b": b}))
    return str(p)


# arm "a" here = the learned forecaster trained JOINTLY (fc_train joint: forecast loss, own param group), the pre-fix
# Step-1 behaviour these tests were written for; the frozen pre-trained default (review fix D3) is "af" below.
ARM_FLAGS = {"a": ["--t-forecaster", "learned", "--t-fc-train", "joint"], "b": ["--t-forecaster", "ca"],
             "d": ["--t-forecaster", "oracle_gt"]}


def _targs(tmp_path, arm, run, extra=()):
    return T2T._args(tmp_path, run, ["--temporal", "1", *ARM_FLAGS[arm], "--t-ca-scales", _scales(tmp_path),
                                     *extra])


def _tdatasets(tcfg, n_train=8, n_eval=4, n_agents=5, aug=True):
    return {"train": FakeCK2T(tcfg, aug=aug, n_agents=n_agents, n=n_train),
            "evals": [("navtrain_val", "r34", FakeR34T(tcfg, n_agents=n_agents, n=n_eval)),
                      ("navtrain_val_raw", "raw", FakeCK2T(tcfg, aug=False, n_agents=n_agents, n=n_eval, gt=False,
                                                           seed=5))]}


def _read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# ----------------------------------------------------------------------------------------------- args / guards
def test_resolve_temporal_flags_and_guards(tmp_path):
    off = T2T._args(tmp_path)
    assert off.temporal_cfg == TCF.TemporalCfg() and not off.temporal_cfg.on
    a = _targs(tmp_path, "a", "x")
    assert a.temporal_cfg.on and a.temporal_cfg.forecaster == "learned" and a.temporal_cfg.n_modes == 6
    b = _targs(tmp_path, "b", "x")
    assert b.temporal_cfg.forecaster == "ca" and b.temporal_cfg.n_modes == 1      # --t-modes default per forecaster
    with pytest.raises(SystemExit, match="n_modes"):
        _targs(tmp_path, "b", "x", ["--t-modes", "6"])
    with pytest.raises(SystemExit, match="_PRIV"):
        _targs(tmp_path, "d", "oracle_run")
    d = _targs(tmp_path, "d", "oracle_run_PRIV")
    assert d.temporal_cfg.privileged
    with pytest.raises(SystemExit, match="ca-scales"):
        T2T._args(tmp_path, "x", ["--temporal", "1", "--t-forecaster", "ca", "--t-ca-scales",
                                  str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit, match="arm 'T'"):
        T2.resolve_args(T2.get_parser().parse_args(["--arm", "M", "--run", "m", "--temporal", "1"]))
    with pytest.raises(SystemExit, match="not built"):
        T2T._args(tmp_path, "x", ["--temporal", "1", "--t-source", "v2det"])


def test_obj_ok_threshold():
    class O:
        def __init__(self, f):
            self.frac_obj_bad, self.n_obj_bad, self.pack = f, int(f * 1000), type("P", (), {"dir": "pack"})()

    class D:
        def __init__(self, f):
            self.objects = O(f)

        def __len__(self):
            return 1000

    T2.check_obj_bad("x", D(0.005))                       # == 0.5 % is allowed
    with pytest.raises(SystemExit, match="obj_ok"):
        T2.check_obj_bad("x", D(0.006))
    T2.check_obj_bad("x", object())                       # dataset without objects (OFF)


def test_param_groups_and_dummy_backward():
    from navsim.agents.para_ssr.ck.model import build_ck
    off = build_ck("T", seed=0, norm=T2T._norm())
    T2.freeze_unused(off)
    c, f = T2.param_groups(off)
    assert f == [] and [id(p) for p in c] == [id(p) for p in T2.trainable(off)]
    ta = TCF.TemporalCfg(on=True, forecaster="learned", fc_train="joint")
    net = build_ck("T", seed=0, norm=T2T._norm(), temporal=ta)
    T2.freeze_unused(net)
    c, f = T2.param_groups(net)
    names = {id(p): n for n, p in net.named_parameters()}
    assert f and all(names[id(p)].startswith("temporal.forecaster.") for p in f)
    assert not any(names[id(p)].startswith("temporal.forecaster.") for p in c)
    assert len(c) + len(f) == len(T2.trainable(net))
    assert sum(p.numel() for p in f) == net.temporal.param_counts()["forecaster"] == 472085
    b = collate_ck([FakeCK2T(ta, n=2)[i] for i in range(2)])
    out = T2.forward(net, b, False, torch.device("cpu"), True, ta)
    assert "fc" in out and out["t_stats"]["n_obj"] >= 1
    T2.dummy_backward(out)
    for n, p in net.named_parameters():
        if n.startswith("temporal.forecaster."):
            assert p.grad is not None and float(p.grad.abs().max()) == 0.0, n


# ----------------------------------------------------------------------------------------------- model inputs only
@pytest.mark.parametrize("arm", ["a", "b"])
def test_model_sees_only_model_inputs_temporal(tmp_path, arm):
    from navsim.agents.para_ssr.ck.model import build_ck
    tc = _targs(tmp_path, arm, "x").temporal_cfg
    net = build_ck("T", seed=0, norm=T2T._norm(), temporal=tc).eval()
    with torch.no_grad():                                  # make the zero-init projections act on the trunk
        g = torch.Generator().manual_seed(5)
        for m in (net.temporal.st_proj, net.temporal.gap_proj):
            m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.05)
            m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.05)
    ds = FakeCK2T(tc, aug=False, n=2)
    b = collate_ck([ds[0], ds[1]])
    b["lead_has"], b["lead_d1"] = torch.tensor([1.0, 0.0]), torch.tensor([12.0, float("nan")])
    dev = torch.device("cpu")
    with torch.no_grad():
        ref = T2.forward(net, b, False, dev, True, tc)
    p = dict(b)
    p["fc_xyh"] = torch.randn_like(b["fc_xyh"].nan_to_num(0.0)) * 30
    p["fc_state"] = torch.full_like(b["fc_state"], TCF.FC_OBS)
    p["lead_has"], p["lead_d1"] = 1 - b["lead_has"], torch.tensor([3.0, 4.0])
    for k in ("anc_idx", "anc_group", "anc_pi", "anc_w", "var_anchor", "var_v", "var_col", "var_pi", "y", "y_ok",
              "y_pdms", "var_y", "var_ok", "var_pdms", "gt_traj", "obj_ok"):
        v = b[k]
        p[k] = (~v) if v.dtype == torch.bool else (v.flip(-1) if v.dim() and v.shape[-1] > 1 else v * 0 + 7)
    for k in [k for k in b if k.startswith("ref_")]:
        p[k] = b[k].flip(0) if b[k].dim() else b[k]
    with torch.no_grad():
        got = T2.forward(net, p, False, dev, True, tc)
    for k in ("score_logit", "extra_score_logit", "w_lat", "z_lon"):
        assert torch.equal(ref[k], got[k]), k
    # the object inputs DO reach the model (sanity: the branch is live)
    q = dict(b)
    q["obj_in"] = b["obj_in"].clone()
    q["obj_in"][..., 0:2] += 1.5
    with torch.no_grad():
        moved = T2.forward(net, q, False, dev, True, tc)
    assert not torch.equal(ref["score_logit"], moved["score_logit"])


# ----------------------------------------------------------------------------------------------- 3 CPU steps per arm
@pytest.mark.parametrize("arm", ["a", "b", "d"])
def test_arm_three_cpu_steps(tmp_path, arm):
    from navsim.agents.para_ssr.ck.model import load_ck
    run_name = f"t3_{arm}" + ("_PRIV" if arm == "d" else "")
    a = _targs(tmp_path, arm, run_name, ["--max-steps", "3"])
    run = T2.train(a, datasets=_tdatasets(a.temporal_cfg), norm_override=T2T._norm())
    done = json.loads((run / "done.json").read_text())
    assert done["steps"] == 3 and done["n_skip"] == {"loss": 0, "grad": 0}
    cfg = json.loads((run / "config.json").read_text())
    t = TCF.TemporalCfg.from_dict(cfg["temporal"])
    assert t == a.temporal_cfg and cfg["privileged"] == int(arm == "d")
    assert cfg["param_count_temporal_shared"] == 237888
    assert cfg["param_count_forecaster"] == (472085 if arm == "a" else 0)
    assert cfg["param_count"] == 3516178 + 237888 + cfg["param_count_forecaster"]
    steps = [r for r in _read_jsonl(run / "train_log.jsonl") if r["kind"] == "step"]
    assert [r["step"] for r in steps] == [1, 2, 3] and all(np.isfinite(r["loss"]) for r in steps)
    for r in steps:
        assert ("fc_loss" in r) == (arm == "a") and ("grad_norm_fc" in r) == (arm == "a")
        assert r["t_n_obj"] >= 1 and "loss_ck" in r and r["n_obj_bad"] == 0
        if arm == "a":
            assert r["loss"] == pytest.approx(r["loss_ck"] + r["fc_lambda"] * r["fc_loss"], rel=1e-4, abs=1e-5)
            assert r["fc_n_obj"] > 0
        else:
            assert r["loss"] == r["loss_ck"]
    val = _read_jsonl(run / "val_metrics.jsonl")
    assert {r["split"] for r in val} == {"navtrain_val", "navtrain_val_raw"}
    for r in val:
        assert r.get("privileged") == (1 if arm == "d" else None)
        assert r["n_obj_bad"] == 0
        assert ("fc_nll" in r) == (arm == "a")
        if arm == "a":
            assert np.isfinite(r["fc_nll"]) and np.isfinite(r["fc_minade"]) and np.isfinite(r["fc_minfde"])
    ck = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert len(ck["optim"]["param_groups"]) == (2 if arm == "a" else 1)
    if arm == "d":
        with pytest.raises(PermissionError):
            load_ck(run)
        net, _ = load_ck(run, allow_privileged=True)
    else:
        net, _ = load_ck(run)
    assert net.temporal is not None and net.temporal_cfg == a.temporal_cfg
    if arm != "a":
        assert torch.equal(net.temporal.ca_b, ck["model"]["temporal.ca_b"])
    # the temporal branch learned something (zero-init projections moved), lon head still frozen at 0
    assert float(net.temporal.st_proj.weight.detach().abs().max()) > 0
    assert float(net.trunk.lon_head[-1].weight.abs().max()) == 0.0


def _fc_init(tmp_path, n_modes=6, seed=123) -> str:
    """a 'pre-trained' forecaster file in the pretrain_forecaster.py format (random weights of another seed)."""
    from navsim.agents.para_ssr.ck.temporal.forecast import LearnedForecaster
    p = Path(tmp_path) / f"pre_m{n_modes}_s{seed}" / "forecaster.pt"
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        f = LearnedForecaster(n_modes=n_modes, seed=seed)
        torch.save({"forecaster": f.state_dict(), "n_modes": n_modes, "seed": seed, "step": 0}, p)
    return str(p)


def test_frozen_pretrained_arm_three_cpu_steps(tmp_path):
    """review fix D3: arm a with a pre-trained forecaster FROZEN -- weights loaded from fc_init and unchanged after
    training, no forecast loss term (stats logged only), one optimiser group, trainable count == arm b's."""
    from navsim.agents.para_ssr.ck.model import load_ck
    fi = _fc_init(tmp_path)
    a = T2T._args(tmp_path, "t3_af", ["--temporal", "1", "--t-forecaster", "learned", "--t-fc-init", fi,
                                      "--t-ca-scales", _scales(tmp_path), "--max-steps", "3"])
    assert a.temporal_cfg.fc_frozen and a.temporal_cfg.fc_init == fi
    run = T2.train(a, datasets=_tdatasets(a.temporal_cfg), norm_override=T2T._norm())
    cfg = json.loads((run / "config.json").read_text())
    b = _targs(tmp_path, "b", "t3_b_ref", ["--max-steps", "1"])
    runb = T2.train(b, datasets=_tdatasets(b.temporal_cfg), norm_override=T2T._norm())
    cfgb = json.loads((runb / "config.json").read_text())
    assert cfg["param_count_trainable"] == cfgb["param_count_trainable"]       # frozen forecaster: as arm b
    assert cfg["param_count_forecaster"] == 472085
    pv = cfg["provenance"]
    assert pv["data"]["fc_init"]["path"] == fi and len(pv["data"]["fc_init"]["sha256"]) == 64
    assert len(pv["code"]["train"]["sha256"]) == 64 and pv["k_obj"] == TCF.K_OBJ
    steps = [r for r in _read_jsonl(run / "train_log.jsonl") if r["kind"] == "step"]
    for r in steps:
        assert r["loss"] == r["loss_ck"] and "fc_loss" in r and "grad_norm_fc" not in r and "fc_lambda" not in r
        assert r["t_n_valid"] >= 1
    val = [r for r in _read_jsonl(run / "val_metrics.jsonl") if r["split"] == "navtrain_val"]
    assert val and np.isfinite(val[-1]["fc_minade"])
    ck = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert len(ck["optim"]["param_groups"]) == 1
    ref = torch.load(fi, map_location="cpu", weights_only=False)["forecaster"]
    for k, v in ref.items():
        assert torch.equal(ck["model"][f"temporal.forecaster.{k}"], v), k     # loaded AND unchanged
    net, _ = load_ck(run)
    assert all(torch.equal(net.temporal.forecaster.state_dict()[k], v) for k, v in ref.items())
    z = np.load(run / "eval_r34_navtrain_val_last.npz")                       # D6: per-token logits saved
    assert z["logit"].ndim == 3 and z["logit"].shape[-1] == 5 and int(z["step"]) == 3 and len(z["rows"]) == 4
    # guards: wrong n_modes, missing file, frozen without a pre-trained forecaster
    with pytest.raises(SystemExit, match="n_modes"):
        T2T._args(tmp_path, "x", ["--temporal", "1", "--t-fc-init", _fc_init(tmp_path, n_modes=3),
                                  "--t-ca-scales", _scales(tmp_path)])
    with pytest.raises(SystemExit, match="missing"):
        T2T._args(tmp_path, "x", ["--temporal", "1", "--t-fc-init", str(tmp_path / "nope.pt")])
    with pytest.raises(SystemExit, match="fc_init"):
        T2T._args(tmp_path, "x", ["--temporal", "1", "--t-forecaster", "learned"])
    with pytest.raises(SystemExit, match="learned forecaster only"):
        T2T._args(tmp_path, "x", ["--temporal", "1", "--t-forecaster", "ca", "--t-fc-init", fi,
                                  "--t-ca-scales", _scales(tmp_path)])
    # D5: resuming with another pre-trained forecaster file is refused (provenance)
    (run / "done.json").unlink()
    other = _fc_init(tmp_path, seed=7)
    c = T2T._args(tmp_path, "t3_af", ["--temporal", "1", "--t-forecaster", "learned", "--t-fc-init", other,
                                      "--t-ca-scales", _scales(tmp_path), "--max-steps", "4", "--resume"])
    with pytest.raises(SystemExit, match="temporal"):
        T2.train(c, datasets=_tdatasets(c.temporal_cfg), norm_override=T2T._norm())


def test_resume_refuses_changed_temporal_cfg(tmp_path):
    a = _targs(tmp_path, "a", "rs", ["--max-steps", "1", "--best-metric", "none"])
    run = T2.train(a, datasets=_tdatasets(a.temporal_cfg), norm_override=T2T._norm())
    (run / "done.json").unlink()
    for extra, base in ((["--t-lambda-fc", "0.5"], "a"), (["--t-obj-drop", "0.2"], "a"), ([], "b")):
        b = _targs(tmp_path, base, "rs", ["--resume", "--max-steps", "2", "--best-metric", "none", *extra])
        with pytest.raises(SystemExit, match="temporal"):
            T2.train(b, datasets=_tdatasets(b.temporal_cfg), norm_override=T2T._norm())
    with pytest.raises(SystemExit, match="temporal"):
        T2.train(T2T._args(tmp_path, "rs", ["--resume", "--max-steps", "2", "--best-metric", "none"]),
                 datasets=T2T._datasets(), norm_override=T2T._norm())
    b = _targs(tmp_path, "a", "rs", ["--resume", "--max-steps", "2", "--best-metric", "none"])
    T2.train(b, datasets=_tdatasets(b.temporal_cfg), norm_override=T2T._norm())
    assert json.loads((run / "done.json").read_text())["steps"] == 2


def test_old_config_without_temporal_resumes_as_off(tmp_path):
    a = T2T._args(tmp_path, "old", ["--max-steps", "1", "--best-metric", "none"])
    run = T2.train(a, datasets=T2T._datasets(), norm_override=T2T._norm())
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["temporal"] == TCF.TemporalCfg().to_dict() and cfg["privileged"] == 0
    assert cfg["param_count_temporal_shared"] == 0 and cfg["param_count_forecaster"] == 0
    for k in ("temporal", "privileged", "param_count_temporal_shared", "param_count_forecaster"):
        cfg.pop(k)
    U.write_json(run / "config.json", cfg)                 # an 8ae8b1d-era config.json
    (run / "done.json").unlink()
    with pytest.raises(SystemExit, match="temporal"):
        b = _targs(tmp_path, "a", "old", ["--resume", "--max-steps", "2", "--best-metric", "none"])
        T2.train(b, datasets=_tdatasets(b.temporal_cfg), norm_override=T2T._norm())
    T2.train(T2T._args(tmp_path, "old", ["--resume", "--max-steps", "2", "--best-metric", "none"]),
             datasets=T2T._datasets(), norm_override=T2T._norm())
    assert json.loads((run / "done.json").read_text())["steps"] == 2


# ----------------------------------------------------------------------------------------------- DDP (gloo, CPU)
DDP_T = ["--temporal", "1", "--t-forecaster", "learned", "--t-fc-train", "joint", "--t-frame-drop", "0",
         "--t-obj-drop", "0", "--t-vnoise", "0"]


def _ddp_worker_t(rank, world, port, out_root, run):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world), RANK=str(rank),
                      LOCAL_RANK=str(rank), OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    _ddp_single_t(out_root, run)


def _ddp_single_t(out_root, run):
    a = T2.resolve_args(T2.get_parser().parse_args(
        ["--arm", "T", "--run", run, "--device", "cpu", "--out-root", out_root, "--workers", "0",
         "--tokens-per-batch", "4", "--epochs", "1", "--log-every", "1", "--ckpt-every", "0",
         "--train-eval-rows", "0", "--lr", "1e-4", "--warmup-steps", "1", "--clip", "0", *DDP_T]))
    T2.train(a, datasets=_tdatasets(a.temporal_cfg, n_train=8, n_eval=5, aug=False), norm_override=T2T._norm())


def test_ddp_two_ranks_equals_single_process_arm_a(tmp_path):
    import subprocess
    import torch.multiprocessing as mp
    if torch.cuda.is_available():
        r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp",
                            str(tmp_path / "child"), f"{__file__}::test_ddp_two_ranks_equals_single_process_arm_a"],
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), cwd=CK, capture_output=True, text=True,
                           timeout=900)
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        return
    from navsim.agents.para_ssr.ck.model import load_ck
    for t in T2T.TRAIN_TOKENS:
        T2T._gt_cached(t)
    if not all(bool(T2T._GT_CACHE[t]["ref_gt_ok"]) for t in T2T.TRAIN_TOKENS):
        pytest.skip("surrogate GT of the fixture tokens absent")
    out = str(tmp_path)
    mp.start_processes(_ddp_worker_t, args=(2, T2T._free_port(), out, "ddp2"), nprocs=2, join=True,
                       start_method="fork")
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        os.environ.pop(k, None)
    _ddp_single_t(out, "ddp1")
    r2, r1 = Path(out) / "ddp2", Path(out) / "ddp1"
    s2 = [r for r in _read_jsonl(r2 / "train_log.jsonl") if r["kind"] == "step"]
    s1 = [r for r in _read_jsonl(r1 / "train_log.jsonl") if r["kind"] == "step"]
    assert len(s2) == len(s1) == 2 and all(r["world"] == 2 for r in s2)
    for x, y in zip(s1, s2):
        for k in ("loss", "loss_ck", "fc_loss", "fc_nll", "fc_ce"):
            assert x[k] == pytest.approx(y[k], rel=1e-4, abs=1e-6), k
        assert x["fc_n_obj"] == y["fc_n_obj"] == 20.0                     # counts are summed over ranks
        assert x["grad_norm"] == pytest.approx(y["grad_norm"], rel=1e-3)
        assert x["grad_norm_fc"] == pytest.approx(y["grad_norm_fc"], rel=1e-3)
    n2, _ = load_ck(r2, "last")
    n1, _ = load_ck(r1, "last")
    # temporal.forecaster.dec_pi.bias is shared by every mode logit: softmax shift invariance makes its gradient
    # exactly 0 in theory (sum over modes of p - onehot), so Adam turns pure reduction-order noise into +-lr steps
    # [실측 1.9e-5 at lr 1e-4]; it cannot change any output (the mode probabilities are shift invariant)
    sd2, sd1 = n2.state_dict(), n1.state_dict()
    d = max(float((sd2[k] - sd1[k]).abs().max()) for k in sd2
            if sd2[k].dtype.is_floating_point and k != "temporal.forecaster.dec_pi.bias")
    assert d < 1e-5, d
    v2, v1 = _read_jsonl(r2 / "val_metrics.jsonl"), _read_jsonl(r1 / "val_metrics.jsonl")
    for x, y in zip(v1, v2):
        assert x["split"] == y["split"] and x["n_tokens"] == y["n_tokens"] == 5
        for k in ("fc_nll", "fc_minade", "fc_minfde", "fc_n_obj"):
            assert x[k] == pytest.approx(y[k], rel=1e-4, abs=1e-6), k


# ----------------------------------------------------------------------------------------------- real datasets
def _cfg(forecaster="learned", **kw):
    if forecaster == "learned":
        kw.setdefault("fc_train", "joint")
    return TCF.TemporalCfg(on=True, forecaster=forecaster, n_modes=6 if forecaster == "learned" else 1, **kw)


def _limit_pack():
    """a 256-row navtrain_train pack (K_OBJ 96) under D3/test_tmp, built once by the real builder."""
    if not (SMOKE_ROOT / "bevfusion" / "navtrain_train" / "build.json").is_file():
        from tools.ck.v3 import build_obj_pack as B
        assert B.main(["--split", "navtrain_train", "--limit", "256", "--workers", "4", "--out-root",
                       str(SMOKE_ROOT)]) == 0
    return SMOKE_ROOT


@have_val_pack
def test_real_batch_streams_identical_between_arms_and_off_unchanged(monkeypatch):
    from tools.ck.data.ck2_dataset import CK2Dataset
    n = 16
    ta, tb = _cfg("learned"), _cfg("ca")
    da = CK2Dataset("navtrain_val", bev="none", n_var=32, limit=n, objects={"cfg": ta.to_dict(), "aug": True,
                                                                             "targets": True})
    db = CK2Dataset("navtrain_val", bev="none", n_var=32, limit=n, objects={"cfg": tb.to_dict(), "aug": True,
                                                                             "targets": True})
    dn = CK2Dataset("navtrain_val", bev="none", n_var=32, limit=n, objects={"cfg": ta.to_dict(), "aug": False,
                                                                             "targets": True})
    calls = []
    real_aug = OP.augment_objects
    monkeypatch.setattr(OP, "augment_objects", lambda *x, **k: calls.append(1) or real_aug(*x, **k))
    off = CK2Dataset("navtrain_val", bev="none", n_var=32, limit=n)
    changed_obj = 0
    for e in (0, 1):
        for d in (da, db, dn, off):
            d.set_epoch(e)
        for i in range(n):
            ia, ib, inn, io = da[i], db[i], dn[i], off[i]
            assert sorted(ia) == sorted(ib)
            for k in ia:
                va, vb = ia[k], ib[k]
                assert (va == vb) if isinstance(va, (str, int)) else np.asarray(va).tobytes() == np.asarray(vb).tobytes(), k
            assert set(ia) - set(io) == {"obj_in", "obj_mask", "frame_valid", "obj_ok", "fc_xyh", "fc_state"}
            for k in io:                                   # every CK2 key bitwise equal to the OFF dataset
                assert (io[k] == ia[k]) if isinstance(io[k], (str, int)) else \
                    np.asarray(io[k]).tobytes() == np.asarray(ia[k]).tobytes(), k
            for k in ("fc_xyh", "fc_state", "obj_ok"):     # augmentation never touches the targets
                assert np.asarray(inn[k]).tobytes() == np.asarray(ia[k]).tobytes(), k
            changed_obj += int(not np.array_equal(inn["obj_in"], ia["obj_in"]))
    assert changed_obj > 0
    n_calls_on = len(calls)
    calls.clear()
    for i in range(n):
        off[i]
    assert calls == [] and n_calls_on == 2 * 2 * n         # da + db per item, never for OFF or aug False
    da.set_epoch(1)                                        # the augmentation stream is keyed by the epoch
    a1 = [da[i]["obj_in"].tobytes() for i in range(4)]
    da.set_epoch(0)
    a0 = [da[i]["obj_in"].tobytes() for i in range(4)]
    assert a1 != a0


@have_val_pack
def test_real_ckdataset_objects_and_guards():
    from tools.ck.data.ck2_dataset import CK2Dataset
    from tools.ck.data.ck_dataset import CKDataset
    td = _cfg("oracle_gt")
    ds = CKDataset("navtrain_val", bev="none", k=16, labels="cand", limit=8,
                   objects={"cfg": td.to_dict(), "aug": False, "targets": True})
    it = ds[3]
    for k in TCF.OBJ_INPUT_KEYS + TCF.PRIV_INPUT_KEYS + TCF.TARGET_KEYS:
        assert k in it, k
    assert np.array_equal(it["priv_fc_xyh"], it["fc_xyh"], equal_nan=True)
    assert np.array_equal(it["priv_fc_state"], it["fc_state"])
    pack = OP.ObjPack("navtrain_val")
    r = int(ds.rows[3])
    assert np.array_equal(it["obj_in"], pack.item(r)["obj_in"])          # eval: clean rows
    with pytest.raises(ValueError, match="aug"):
        CKDataset("navtrain_val", bev="none", k=16, labels="cand", limit=8,
                  objects={"cfg": td.to_dict(), "aug": True, "targets": True})
    with pytest.raises(ValueError, match="off"):
        CKDataset("navtrain_val", bev="none", k=16, labels="cand", limit=8,
                  objects={"cfg": TCF.TemporalCfg().to_dict(), "aug": False, "targets": False})
    with pytest.raises(ValueError, match="targets"):
        CKDataset("navtrain_val", bev="none", k=16, labels="cand", limit=8,
                  objects={"cfg": td.to_dict(), "aug": False, "targets": False})
    with pytest.raises(ValueError):
        CKDataset("navtrain_val", bev="none", k=16, labels="cand", limit=8,
                  objects={"cfg": td.to_dict(), "aug": False, "targets": True, "bogus": 1})
    if (TCF.D3 / "gt_objects" / "navtrain_train" / "index.parquet").is_file():   # 256-row limit pack
        ts = _cfg("learned", pack_root=str(_limit_pack()))
        CK2Dataset("navtrain_train", bev="none", n_var=0, limit=64, objects={"cfg": ts.to_dict(), "aug": False,
                                                                             "targets": True})
        with pytest.raises(ValueError, match="rows"):
            CK2Dataset("navtrain_train", bev="none", n_var=0, objects={"cfg": ts.to_dict(), "aug": False,
                                                                       "targets": True})
