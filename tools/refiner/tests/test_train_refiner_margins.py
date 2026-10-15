"""Tests for the optional surrogate-margin arguments of tools/refiner/train_refiner.py (--m-col / --m-dac).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python -m pytest -q tools/refiner/tests/test_train_refiner_margins.py

- defaults: no flag -> no override (SurrogateConfig() code defaults 0.3 / 0.2, unchanged);
- an override replaces only the given field(s);
- compute_loss: cfg=None == explicit SurrogateConfig() (bitwise); an override == surrogate_terms_batch with that config,
  and a smaller margin gives a smaller (or equal) col / dac term;
- train(): the effective config is recorded in config.json; a resume with other margins and the stub are refused.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(HERE), str(HERE.parent)):
    sys.path.insert(0, p)

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner import surrogate as SU  # noqa: E402
import refiner_synth as SY  # noqa: E402
import train_refiner as TR  # noqa: E402

QUIET = lambda *a, **k: None  # noqa: E731


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("train_margins")
    df, src = SY.make_sources(tmp, 10)
    RD.pack_split("train", df, tmp / "packed", sources=src, workers=1, log_fn=QUIET)
    teacher = SY.make_fake_teacher(tmp, df.token)
    return tmp, df, teacher


def _args(tmp, teacher, extra=()):
    a = ["--arm", "none", "--fold", "0", "--seed", "0", "--gpu", "-1", "--packed-root", str(tmp / "packed"),
         "--runs", str(tmp / "runs"), "--teacher-root", str(teacher), "--tokens-per-batch", "3", "--workers", "0",
         "--n-norm", "4", "--log-every", "1", "--inner-val-frac", "0.3", "--max-steps", "1", "--epochs", "1",
         "--max-val-batches", "1", *extra]
    return TR.get_parser().parse_args(a)


def _base():
    return ["--arm", "none", "--fold", "0", "--seed", "0"]


def test_defaults_unchanged():
    c = SU.SurrogateConfig()
    assert c.m_col == 0.3 and c.m_dac == 0.2 and c.use_human_mask is True
    a = TR.get_parser().parse_args(_base())
    assert a.m_col is None and a.m_dac is None
    assert TR.surrogate_config(a) is None
    assert TR.surrogate_config_record(None, "surrogate.surrogate_terms") == dataclasses.asdict(SU.SurrogateConfig())
    assert TR.surrogate_config_record(None, "stub") is None


@pytest.mark.parametrize("extra,mc,md", [(["--m-col", "0.15", "--m-dac", "0.1"], 0.15, 0.1),
                                         (["--m-col", "0.15"], 0.15, 0.2), (["--m-dac", "0.05"], 0.3, 0.05),
                                         (["--m-col", "0", "--m-dac", "0"], 0.0, 0.0)])
def test_override_only_given_fields(extra, mc, md):
    a = TR.get_parser().parse_args(_base() + extra)
    c = TR.surrogate_config(a)
    assert isinstance(c, SU.SurrogateConfig) and c.m_col == mc and c.m_dac == md
    d, d0 = dataclasses.asdict(c), dataclasses.asdict(SU.SurrogateConfig())
    assert {k: v for k, v in d.items() if k not in ("m_col", "m_dac")} == \
           {k: v for k, v in d0.items() if k not in ("m_col", "m_dac")}


def _batch(env):
    tmp, df, _ = env
    P = RD.PackedSplit("train", tmp / "packed")
    return RD.collate_tokens([RD.TokenDataset(P, [3, 8, 9], None)[i] for i in range(3)])


def test_compute_loss_uses_cfg(env):
    b = _batch(env)
    T, K = b["tau0"].shape[:2]
    torch.manual_seed(0)
    out = {"z_lon": 0.3 * torch.randn(T, K, 6), "w_lat": 0.3 * torch.randn(T, K, 6), "gate_logit": torch.zeros(T, K)}
    W = dict(TR.DEFAULT_W)
    fn = TR.surrogate_terms_batch
    l0, s0, _ = TR.compute_loss(out, b, fn, W, 1.0)
    l1, s1, _ = TR.compute_loss(out, b, fn, W, 1.0, cfg=SU.SurrogateConfig())
    assert float(l0) == float(l1) and s0 == s1                           # None == code defaults, bitwise
    cfg = SU.SurrogateConfig(m_col=0.15, m_dac=0.1)
    l2, s2, dec = TR.compute_loss(out, b, fn, W, 1.0, cfg=cfg)
    t = TR.surrogate_terms_batch(dec, b, T, K, cfg)
    m = b["draft_valid"].reshape(-1).float()
    for n in ("col", "dac"):
        ref = float((torch.nan_to_num(t[n].float(), nan=0.0, posinf=1e3) * m).sum() / m.sum().clamp(min=1))
        assert s2[f"t_{n}"] == pytest.approx(ref, rel=1e-6, abs=1e-12)
        assert s2[f"t_{n}"] <= s0[f"t_{n}"] + 1e-12                      # smaller margin -> smaller hinge cost
    assert s2["t_col"] < s0["t_col"]                                     # the synthetic cone is within 0.3 m for some draft
    for n in ("prog", "cmf", "mod"):
        assert s2[f"t_{n}"] == s0[f"t_{n}"]                              # margins touch only col / dac


def test_train_records_and_guards(env):
    tmp, _, teacher = env
    a = _args(tmp, teacher, ["--surrogate", "real", "--m-col", "0.15", "--m-dac", "0.1", "--tag", "mrg"])
    run = TR.train(a)
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["m_col"] == 0.15 and cfg["m_dac"] == 0.1
    assert cfg["surrogate_cfg"]["m_col"] == 0.15 and cfg["surrogate_cfg"]["m_dac"] == 0.1
    assert cfg["surrogate_cfg"]["use_human_mask"] is True
    # resume with other margins is refused; the same margins resume
    with pytest.raises(SystemExit):
        TR.train(_args(tmp, teacher, ["--surrogate", "real", "--m-col", "0.3", "--tag", "mrg"]))
    with pytest.raises(SystemExit):
        TR.train(_args(tmp, teacher, ["--surrogate", "real", "--tag", "mrg"]))
    TR.train(_args(tmp, teacher, ["--surrogate", "real", "--m-col", "0.15", "--m-dac", "0.1", "--tag", "mrg"]))
    # default run records the code defaults
    run_d = TR.train(_args(tmp, teacher, ["--surrogate", "real", "--tag", "dflt"]))
    cd = json.loads((run_d / "config.json").read_text())
    assert cd["m_col"] is None and cd["m_dac"] is None
    # (JSON round trip: the TTC field ttc_deltas is a tuple in the dataclass and a list in config.json)
    assert cd["surrogate_cfg"] == json.loads(json.dumps(dataclasses.asdict(SU.SurrogateConfig())))
    # the stub ignores margins -> refused
    with pytest.raises(SystemExit):
        TR.train(_args(tmp, teacher, ["--surrogate", "stub", "--m-col", "0.15", "--tag", "stubm"]))
