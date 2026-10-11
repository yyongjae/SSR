"""refiner-v3 forecast loss + forecast metrics (SPEC sections 4.6, 5.3; builder LOSS+EVAL).  CPU, synthetic inputs.

temporal/loss.py: written by LOSS+EVAL, overwritten and rebuilt to the same contract by the MODEL builder (2026-10-11
06:15 UTC); these tests pin that contract with hand-computed answers.  temporal/metrics.py is self-contained (same
validity / WTA rule).

  cd /workspace/yongjae/SSR-v3 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-v3 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 10 /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v3_loss_metrics.py

Known answers (M = 2 toy, hand-computed), masking (state / agent / slot / augmented mask / trimming), NaN targets,
zero-target batches (finite 0 that stays in the graph), per-object averaging, WTA tie rule, analytic gradients,
autocast independence, metric accumulation (merge / tensor round trip), generator independence (CA-style Forecast
without log_b), trainer hook (add_forecast_loss, forecast_touch), and end-to-end with the model builder's
LearnedForecaster / ca_forecast when built.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CK = str(Path(__file__).resolve().parents[1])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck.temporal import config as TC  # noqa: E402
from navsim.agents.para_ssr.ck.temporal import loss as TL  # noqa: E402
from navsim.agents.para_ssr.ck.temporal import metrics as TM  # noqa: E402


@dataclass
class FC:
    """Duck-typed stand-in of forecast.Forecast (only the fields loss / metrics read)."""
    mu: torch.Tensor
    b: torch.Tensor
    pi: torch.Tensor
    group: torch.Tensor
    valid: torch.Tensor
    log_b: Optional[torch.Tensor] = None
    logit_pi: Optional[torch.Tensor] = None

    @property
    def agent(self) -> torch.Tensor:                 # = forecast.Forecast.agent
        return self.valid & (self.group < 2)


def learned(mu, log_b, logit_pi, group, valid):
    mu = torch.as_tensor(mu, dtype=torch.float32).clone().requires_grad_(True)
    log_b = torch.as_tensor(log_b, dtype=torch.float32).clone().requires_grad_(True)
    logit_pi = torch.as_tensor(logit_pi, dtype=torch.float32).clone().requires_grad_(True)
    return FC(mu=mu, b=log_b.exp(), pi=torch.softmax(logit_pi, -1), group=torch.as_tensor(group).long(),
              valid=torch.as_tensor(valid).bool(), log_b=log_b, logit_pi=logit_pi)


def targets(T, K, kpack=64, seed=0):
    g = np.random.default_rng(seed)
    xyh = np.full((T, kpack, 10, 3), np.nan, np.float32)
    xyh[:, :K] = g.normal(0, 5, (T, K, 10, 3)).astype(np.float32)
    st = np.zeros((T, kpack, 10), np.int8)
    m0 = np.zeros((T, kpack), bool)
    m0[:, :K] = True
    return torch.from_numpy(xyh), torch.from_numpy(st), torch.from_numpy(m0)


def toy():
    """T = 1, K = 2: slot 0 vehicle with 4 OBS steps (others beyond), slot 1 static (no target).
    mode 0 = y + 0.1, b 0.5; mode 1 = y + 1.0, b 1.0; pi = (0.25, 0.75)."""
    xyh, st, m0 = targets(1, 2)
    st[0, 0, :4] = TC.FC_OBS
    st[0, 0, 4:] = TC.FC_BEYOND
    st[0, 1, :] = TC.FC_OBS                          # static slot: has OBS states but is not an agent
    y = torch.nan_to_num(xyh[0, :2, :, :2])
    mu = torch.zeros(1, 2, 2, 10, 2)
    mu[0, :, 0] = y + 0.1
    mu[0, :, 1] = y + 1.0
    log_b = torch.zeros(1, 2, 2, 10, 2)
    log_b[0, :, 0] = math.log(0.5)
    logit_pi = torch.tensor([[[0.0, math.log(3.0)], [0.0, 0.0]]])
    fc = learned(mu, log_b, logit_pi, [[0, 2]], [[True, True]])
    return fc, xyh, st, m0


# ----------------------------------------------------------------------------------------------- constants
def test_constants_follow_config():
    assert TC.FC_OBS == 1 and TC.FC_STEPS == 10 and TM.AGENT_GROUPS == (0, 1) == tuple(TL.AGENT_GROUPS)
    assert TL.FC_OBS == TC.FC_OBS and TM.B_FLOOR == TC.B_MIN


# ----------------------------------------------------------------------------------------------- known answers
def test_known_answer_m2():
    fc, xyh, st, m0 = toy()
    loss, s = TL.forecast_loss(fc, xyh, st, m0)
    nll = math.log(2 * 0.5) + 0.1 / 0.5             # 0.2 at m* = 0
    ce = math.log(4.0)                               # -log 0.25
    assert loss.item() == pytest.approx(nll + ce, abs=1e-6)
    assert s["fc_nll"] == pytest.approx(nll, abs=1e-6) and s["fc_ce"] == pytest.approx(ce, abs=1e-6)
    assert s["fc_n_obj"] == 1.0
    assert s["fc_minade"] == pytest.approx(math.sqrt(0.02), abs=1e-6)
    assert s["fc_minfde"] == pytest.approx(math.sqrt(0.02), abs=1e-6)
    assert set(s) == {"fc_loss", "fc_nll", "fc_ce", "fc_minade", "fc_minfde", "fc_n_obj"}


def test_known_gradients():
    fc, xyh, st, m0 = toy()
    loss, _ = TL.forecast_loss(fc, xyh, st, m0)
    loss.backward()
    # CE gradient wrt logits = softmax - onehot(m* = 0); static slot 1 gets exactly 0
    assert torch.allclose(fc.logit_pi.grad[0, 0], torch.tensor([0.25 - 1.0, 0.75]), atol=1e-6)
    assert torch.equal(fc.logit_pi.grad[0, 1], torch.zeros(2))
    # NLL gradient wrt mu (m* = 0): sign(mu - y) / b / (2 * n_valid) = 1 / 0.5 / 8 on the 4 valid steps
    g = fc.mu.grad[0, 0, 0]
    assert torch.allclose(g[:4], torch.full((4, 2), 0.25), atol=1e-6)
    assert torch.equal(g[4:], torch.zeros(6, 2))
    assert torch.equal(fc.mu.grad[0, 0, 1], torch.zeros(10, 2))        # loser mode: no NLL gradient
    assert torch.equal(fc.mu.grad[0, 1], torch.zeros(2, 10, 2))
    # d/dlog_b of log 2b + |d| / b = 1 - |d| / b = 1 - 0.2 = 0.8, / (2 * 4)
    assert torch.allclose(fc.log_b.grad[0, 0, 0, :4], torch.full((4, 2), 0.8 / 8), atol=1e-6)
    assert torch.equal(fc.log_b.grad[0, 0, 1], torch.zeros(10, 2))


def test_per_object_average_not_per_step():
    """Two vehicles with 1 and 9 valid steps: the loss is the mean of the two per-object values."""
    T, K, M = 1, 2, 1
    xyh, st, m0 = targets(T, K, seed=3)
    st[0, 0, :1] = TC.FC_OBS
    st[0, 1, :9] = TC.FC_OBS
    y = torch.nan_to_num(xyh[0, :K, :, :2])
    mu = torch.zeros(T, K, M, 10, 2)
    mu[0, 0, 0] = y[0] + 0.3
    mu[0, 1, 0] = y[1] - 0.6
    fc = learned(mu, torch.zeros(T, K, M, 10, 2), torch.zeros(T, K, M), [[0, 1]], [[True, True]])
    loss, s = TL.forecast_loss(fc, xyh, st, m0)
    per = [math.log(2) + 0.3, math.log(2) + 0.6]                     # b = 1, CE = 0 for M = 1
    assert loss.item() == pytest.approx(np.mean(per), abs=1e-6)
    assert s["fc_n_obj"] == 2.0 and s["fc_ce"] == pytest.approx(0.0, abs=1e-7)


def test_wta_tie_takes_first_mode_and_l1_rule():
    T, K, M = 1, 1, 3
    xyh, st, m0 = targets(T, K, seed=4)
    st[0, 0, :] = TC.FC_OBS
    y = xyh[0, 0, :, :2]
    mu = torch.zeros(T, K, M, 10, 2)
    mu[0, 0, 0] = y + torch.tensor([0.5, 0.0])            # |.|_1 = 0.5 (L2 0.5)
    mu[0, 0, 1] = y + torch.tensor([0.0, 0.5])            # tie with mode 0
    mu[0, 0, 2] = y + torch.tensor([0.3, 0.3])            # |.|_1 = 0.6 but L2 0.42: L1 rule must not pick it
    lb = torch.zeros(T, K, M, 10, 2)
    lb[0, 0, 1] = 2.0                                       # modes 0 / 1 differ only in b -> m* reveals the choice
    fc = learned(mu, lb, torch.zeros(T, K, M), [[0]], [[True]])
    loss, s = TL.forecast_loss(fc, xyh, st, m0)
    assert s["fc_nll"] == pytest.approx(math.log(2) + 0.25, abs=1e-6)    # mode 0: mean(0.5, 0) per axis
    assert s["fc_minade"] == pytest.approx(math.sqrt(0.18), abs=1e-6)     # min over ALL modes of the L2 ADE


# ----------------------------------------------------------------------------------------------- masks
@pytest.mark.parametrize("what", ["state", "static", "slot_invalid", "aug_mask"])
def test_masks_exclude(what):
    T, K, M = 2, 3, 2
    xyh, st, m0 = targets(T, K, seed=5)
    st[:, :K, :] = TC.FC_OBS
    grp = torch.zeros(T, K, dtype=torch.long)
    val = torch.ones(T, K, dtype=torch.bool)
    if what == "state":
        st[1, 2, :] = TC.FC_ABSENT
    elif what == "static":
        grp[1, 2] = 2
    elif what == "slot_invalid":
        val[1, 2] = False
    else:
        m0[1, 2] = False
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(T, K, M, 10, 2, generator=g)
    fc = learned(mu, torch.randn(T, K, M, 10, 2, generator=g) * 0.1, torch.randn(T, K, M, generator=g), grp, val)
    loss, s = TL.forecast_loss(fc, xyh, st, m0)
    assert s["fc_n_obj"] == T * K - 1
    loss.backward()
    assert torch.equal(fc.mu.grad[1, 2], torch.zeros(M, 10, 2))
    assert torch.equal(fc.logit_pi.grad[1, 2], torch.zeros(M))
    # changing the excluded object's target / prediction leaves the loss unchanged
    xyh2 = xyh.clone()
    xyh2[1, 2] += 100.0
    l2, _ = TL.forecast_loss(fc, xyh2, st, m0)
    assert l2.item() == loss.item()


def test_nan_targets_and_trim():
    """NaN everywhere the state is not OBS (and in the 61 padding slots of the 64-slot pack) -> finite loss and
    gradients; only the first K slots of the targets are read."""
    T, K, M = 2, 3, 6
    xyh, st, m0 = targets(T, K, seed=6)
    st[:, :K, :5] = TC.FC_OBS
    xyh[:, :K, 5:] = float("nan")
    g = torch.Generator().manual_seed(1)
    fc = learned(torch.randn(T, K, M, 10, 2, generator=g), torch.zeros(T, K, M, 10, 2), torch.zeros(T, K, M),
                 torch.zeros(T, K, dtype=torch.long), torch.ones(T, K, dtype=torch.bool))
    loss, s = TL.forecast_loss(fc, xyh, st, m0)
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in s.values())
    loss.backward()
    for t in (fc.mu, fc.log_b, fc.logit_pi):
        assert torch.isfinite(t.grad).all()
    st_bad = st.clone()
    st_bad[:, K:] = TC.FC_OBS                       # states beyond K are never read
    l2, _ = TL.forecast_loss(fc, xyh, st_bad, m0)
    assert l2.item() == loss.item()


def test_zero_targets_finite_zero_in_graph():
    T, K, M = 2, 4, 6
    xyh, st, m0 = targets(T, K, seed=7)            # every state 0 (no target)
    fc = learned(torch.randn(T, K, M, 10, 2), torch.zeros(T, K, M, 10, 2), torch.randn(T, K, M),
                 torch.zeros(T, K, dtype=torch.long), torch.ones(T, K, dtype=torch.bool))
    loss, s = TL.forecast_loss(fc, xyh, st, m0)
    assert loss.item() == 0.0 and loss.requires_grad and loss.grad_fn is not None
    assert s["fc_n_obj"] == 0.0 and s["fc_minade"] == 0.0
    loss.backward()
    for t in (fc.mu, fc.log_b, fc.logit_pi):
        assert t.grad is not None and torch.equal(t.grad, torch.zeros_like(t))


def test_autocast_independent():
    T, K, M = 2, 5, 6
    xyh, st, m0 = targets(T, K, seed=8)
    st[:, :K, :7] = TC.FC_OBS
    g = torch.Generator().manual_seed(2)
    args = (torch.randn(T, K, M, 10, 2, generator=g), torch.randn(T, K, M, 10, 2, generator=g) * 0.2,
            torch.randn(T, K, M, generator=g), torch.zeros(T, K, dtype=torch.long), torch.ones(T, K, dtype=torch.bool))
    l0, _ = TL.forecast_loss(learned(*args), xyh, st, m0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        l1, _ = TL.forecast_loss(learned(*args), xyh, st, m0)
    assert l1.dtype == torch.float32 and l1.item() == l0.item()


def test_shape_errors_and_ca_style_forecast():
    fc, xyh, st, m0 = toy()
    with pytest.raises(ValueError):
        TL.forecast_loss(fc, xyh[:, :1], st, m0)                     # fewer target slots than K
    with pytest.raises(ValueError):
        TL.forecast_loss(fc, xyh[:, :, :9], st, m0)
    with pytest.raises(ValueError):
        TM.forecast_errors(fc, xyh[:, :, :9], st, m0)
    ca = FC(mu=fc.mu.detach(), b=fc.b.detach(), pi=fc.pi.detach(), group=fc.group, valid=fc.valid)
    with pytest.raises(ValueError, match="learned"):              # CA / oracle arms have no forecast loss
        TL.forecast_loss(ca, xyh, st, m0)


# ----------------------------------------------------------------------------------------------- metrics
def test_errors_known_answer_and_mixture():
    fc, xyh, st, m0 = toy()
    e = TM.forecast_errors(fc, xyh, st, m0)
    assert e["has"].tolist() == [[True, False]]
    assert e["n_step"].tolist() == [[4, 0]]
    assert e["minade"][0, 0].item() == pytest.approx(math.sqrt(0.02), abs=1e-6)
    assert e["fde"][0, 0, 1].item() == pytest.approx(math.sqrt(2.0), abs=1e-6)    # last valid step = 3
    assert e["wta_nll"][0, 0].item() == pytest.approx(0.2, abs=1e-6)
    # mixture: -log(0.25 exp(-S0) + 0.75 exp(-S1)) / 8, S_m = sum over 4 steps x 2 axes of the Laplace NLL
    S0 = 8 * (math.log(1.0) + 0.2)
    S1 = 8 * (math.log(2.0) + 1.0)
    mix = -math.log(0.25 * math.exp(-S0) + 0.75 * math.exp(-S1)) / 8
    assert e["mix_nll"][0, 0].item() == pytest.approx(mix, abs=1e-5)
    assert e["minade"][0, 1].item() == 0.0 and e["mix_nll"][0, 1].item() == 0.0


def test_mixture_equals_wta_for_one_mode_and_ca_style_forecast():
    """CA / oracle forecasts carry b and pi but no log_b / logit_pi: metrics read b (floored at B_MIN)."""
    T, K = 3, 4
    xyh, st, m0 = targets(T, K, seed=9)
    st[:, :K, :6] = TC.FC_OBS
    g = torch.Generator().manual_seed(3)
    mu = torch.randn(T, K, 1, 10, 2, generator=g)
    b = torch.rand(T, K, 1, 10, 2, generator=g) + 0.01            # some below the floor
    ca = FC(mu=mu, b=b, pi=torch.ones(T, K, 1), group=torch.zeros(T, K, dtype=torch.long),
            valid=torch.ones(T, K, dtype=torch.bool))
    e = TM.forecast_errors(ca, xyh, st, m0)
    assert torch.allclose(e["mix_nll"], e["wta_nll"], atol=1e-5)
    y = xyh[:, :K, :6, :2]
    bb = b[:, :, 0, :6].clamp(min=TC.B_MIN)
    ref = (torch.log(2 * bb) + (mu[:, :, 0, :6] - y).abs() / bb).mean((-1, -2))
    assert torch.allclose(e["wta_nll"], ref, atol=1e-5)


def test_metric_accumulator_merge_and_tensor_roundtrip():
    T, K, M = 4, 6, 6
    xyh, st, m0 = targets(T, K, seed=10)
    rng = np.random.default_rng(0)
    st[:, :K] = torch.from_numpy(rng.integers(0, 4, (T, K, 10)).astype(np.int8))
    g = torch.Generator().manual_seed(4)
    grp = torch.from_numpy(rng.integers(0, 3, (T, K)))
    fc = learned(torch.randn(T, K, M, 10, 2, generator=g) * 3, torch.zeros(T, K, M, 10, 2),
                 torch.randn(T, K, M, generator=g), grp, torch.ones(T, K, dtype=torch.bool))
    full = TM.ForecastMetrics()
    full.update(fc, xyh, st, m0)
    parts = []
    for sl in (slice(0, 1), slice(1, 4)):
        f = learned(fc.mu.detach()[sl], fc.log_b.detach()[sl], fc.logit_pi.detach()[sl], grp[sl], fc.valid[sl])
        m = TM.ForecastMetrics()
        m.update(f, xyh[sl], st[sl], m0[sl])
        parts.append(m)
    merged = TM.summarize(parts)
    r = full.result()
    for k, v in r.items():
        assert merged[k] == pytest.approx(v, rel=1e-6, abs=1e-6), k
    rt = TM.ForecastMetrics.from_tensor(full.to_tensor()).result()
    assert rt == r
    e = TM.forecast_errors(fc, xyh, st, m0)
    h = e["has"]
    assert r["fc_n_obj"] == float(h.sum())
    assert r["fc_minade"] == pytest.approx(float(e["minade"][h].mean()), rel=1e-6)
    assert r["fc_miss2m"] == pytest.approx(float((e["minfde"][h] > 2.0).float().mean()), rel=1e-6)
    assert r["fc_veh_n_obj"] + r["fc_vru_n_obj"] == r["fc_n_obj"]
    v5 = e["valid"][..., 9]
    assert r["fc_minde_5s"] == pytest.approx(float(e["l2"][..., 9].amin(2)[v5].mean()), rel=1e-6)
    # the training stat and the eval accumulator agree on one batch
    _, s = TL.forecast_loss(fc, xyh, st, m0)
    assert s["fc_minade"] == pytest.approx(r["fc_minade"], rel=1e-6)
    assert s["fc_nll"] == pytest.approx(r["fc_nll"], rel=1e-5)


def test_empty_metrics_are_nan_not_error():
    r = TM.ForecastMetrics().result()
    assert r["fc_n_obj"] == 0.0 and math.isnan(r["fc_minade"]) and math.isnan(r["fc_nll"])


# ----------------------------------------------------------------------------------------------- trainer hook
def test_add_forecast_loss_and_touch():
    fc, xyh, st, m0 = toy()
    base = torch.tensor(2.0, requires_grad=True)
    st_d = {}
    batch = {"fc_xyh": xyh, "fc_state": st, "obj_mask": torch.stack([m0, m0, m0], 1)}
    out = TL.add_forecast_loss(base, st_d, {"fc": fc}, batch, lambda_fc=0.5)
    l_fc, _ = TL.forecast_loss(fc, xyh, st, m0)
    assert out.item() == pytest.approx(2.0 + 0.5 * l_fc.item(), abs=1e-6)
    assert st_d["fc_lambda"] == 0.5 and "fc_loss" in st_d
    st_e = {}
    assert TL.add_forecast_loss(base, st_e, {"score_logit": None}, {}, 1.0) is base and st_e == {}
    with pytest.raises(KeyError):
        TL.add_forecast_loss(base, {}, {"fc": fc}, {"fc_state": st}, 1.0)
    z = TL.forecast_touch(fc)
    assert z.item() == 0.0
    z.backward()
    assert torch.equal(fc.mu.grad, torch.zeros_like(fc.mu)) and fc.logit_pi.grad is not None
    assert TL.forecast_touch(None).item() == 0.0


def test_real_forecast_dataclass_if_built():
    """The pinned Forecast dataclass (forecast.py, model builder) works with the loss unchanged."""
    F = pytest.importorskip("navsim.agents.para_ssr.ck.temporal.forecast")
    if not hasattr(F, "Forecast"):
        pytest.skip("forecast.Forecast not built yet")
    fc0, xyh, st, m0 = toy()
    T, K, M = fc0.mu.shape[:3]
    kw = dict(mu=fc0.mu, b=fc0.b, pi=fc0.pi, present=torch.ones(T, K, M, 10), c0=torch.zeros(T, K, 2),
              v0=torch.zeros(T, K, 2), h0=torch.zeros(T, K), L=torch.ones(T, K), W=torch.ones(T, K), group=fc0.group,
              valid=fc0.valid, log_b=fc0.log_b, logit_pi=fc0.logit_pi)
    fc = F.Forecast(**kw)
    l1, _ = TL.forecast_loss(fc, xyh, st, m0)
    l0, _ = TL.forecast_loss(fc0, xyh, st, m0)
    assert l1.item() == l0.item()


def _synthetic_objects(T=3, K=64, n=(12, 0, 30), seed=0):
    """obj_in [T, 3, K, 17] canonical rows (SPEC 3.2): token t has n[t] t0 objects (and past copies moved back by
    their velocity); classes cycle through 0..6; frame_valid all True except token 1 (no objects, no past)."""
    g = np.random.default_rng(seed)
    obj = np.zeros((T, 3, K, 17), np.float32)
    msk = np.zeros((T, 3, K), bool)
    fv = np.ones((T, 3), bool)
    fv[1, 1:] = False
    for t in range(T):
        m = n[t]
        if m == 0:
            continue
        xy = np.stack([g.uniform(-5, 30, m), g.uniform(-15, 15, m)], 1)
        h = g.uniform(-np.pi, np.pi, m)
        v = g.normal(0, 4, (m, 2))
        cls = np.arange(m) % 7
        for f, age in enumerate((0.0, 0.5, 1.0)):
            r = obj[t, f, :m]
            r[:, 0:2] = xy - v * age
            r[:, 2], r[:, 3] = np.cos(h), np.sin(h)
            r[:, 4], r[:, 5] = 4.5, 1.9
            r[:, 6:8] = v
            r[:, 8] = g.uniform(0.3, 1.0, m)
            r[np.arange(m), 9 + cls] = 1.0
            r[:, 16] = age
            msk[t, f, :m] = True
    return torch.from_numpy(obj), torch.from_numpy(msk), torch.from_numpy(fv)


def _cv_targets(obj, msk, noise=0.5, seed=1):
    T, _, K, _ = obj.shape
    g = torch.Generator().manual_seed(seed)
    t = 0.5 * torch.arange(1, 11, dtype=torch.float32)
    xy = obj[:, 0, :, None, 0:2] + obj[:, 0, :, None, 6:8] * t[None, None, :, None]
    xy = xy + noise * torch.randn(xy.shape, generator=g)
    xyh = torch.cat([xy, torch.zeros(T, K, 10, 1)], -1)
    st = torch.where(msk[:, 0, :, None].expand(T, K, 10), torch.tensor(TC.FC_OBS, dtype=torch.int8),
                     torch.tensor(TC.FC_NONE, dtype=torch.int8))
    return xyh, st


def test_learned_forecaster_end_to_end():
    """The model builder's LearnedForecaster (if built) -> forecast_loss: finite loss, every forecaster parameter gets
    a finite gradient (non-zero for the decoder), empty-object tokens and zero-target batches stay finite / in the
    graph; trimming the 64-slot pack to the batch maximum leaves loss and stats unchanged."""
    F = pytest.importorskip("navsim.agents.para_ssr.ck.temporal.forecast")
    if not hasattr(F, "LearnedForecaster"):
        pytest.skip("LearnedForecaster not built yet")
    net = F.LearnedForecaster(n_modes=6, seed=0)
    obj, msk, fv = _synthetic_objects()
    xyh, st = _cv_targets(obj, msk)
    fc = net(obj, msk, fv)
    loss, s = TL.forecast_loss(fc, xyh, st, msk[:, 0])
    assert torch.isfinite(loss) and s["fc_n_obj"] > 0 and all(np.isfinite(v) for v in s.values())
    loss.backward()
    names = []
    for n_, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n_
        names.append(n_)
    assert any(p.grad.abs().sum() > 0 for n_, p in net.named_parameters() if n_.startswith("dec_out"))
    assert any(p.grad.abs().sum() > 0 for n_, p in net.named_parameters() if n_.startswith("dec_pi"))
    # agent count: classes 0-2 of the t0 objects of tokens 0 and 2
    n_agent = sum(int(((torch.arange(m) % 7) < 3).sum()) for m in (12, 0, 30))
    assert s["fc_n_obj"] == n_agent
    # trimmed inputs (K' = 30) give the same loss
    oi, om, ofv = F.trim_objects(obj, msk, fv)
    assert oi.shape[2] == 30
    fc2 = net(oi, om, ofv)
    l2, s2 = TL.forecast_loss(fc2, xyh, st, msk[:, 0])
    assert l2.item() == pytest.approx(loss.item(), rel=1e-5, abs=1e-6) and s2["fc_n_obj"] == s["fc_n_obj"]
    # zero targets: finite 0 with (zero) gradients everywhere
    net.zero_grad()
    fc3 = net(obj, msk, fv)
    l3, s3 = TL.forecast_loss(fc3, xyh, torch.zeros_like(st), msk[:, 0])
    assert l3.item() == 0.0 and s3["fc_n_obj"] == 0.0
    l3.backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) == 0.0 for p in net.parameters())


def test_ca_forecast_metrics_if_built():
    """forecast.ca_forecast (no log_b / logit_pi) goes through ForecastMetrics; exact constant-velocity targets
    give ~0 error for objects without history and the forecast_loss refuses it (no trainable forecast)."""
    F = pytest.importorskip("navsim.agents.para_ssr.ck.temporal.forecast")
    if not hasattr(F, "ca_forecast"):
        pytest.skip("ca_forecast not built yet")
    obj, msk, fv = _synthetic_objects()
    fv[:, 1:] = False                                       # no history -> a = 0 -> constant velocity
    xyh, st = _cv_targets(obj, msk, noise=0.0)
    ca_b = torch.full((2, 10), 0.7)
    fc = F.ca_forecast(obj, msk, fv, ca_b)
    m = TM.ForecastMetrics()
    m.update(fc, xyh, st, msk[:, 0])
    r = m.result()
    assert r["fc_n_obj"] > 0 and r["fc_minade"] < 1e-4 and r["fc_minfde"] < 1e-4
    assert r["fc_nll"] == pytest.approx(math.log(2 * 0.7), abs=1e-4)
    with pytest.raises(ValueError):
        TL.forecast_loss(fc, xyh, st, msk[:, 0])
