"""Teacher navtest summary (tools/ck/e2e2/summarize_teacher_navtest.py, used by summarize_teacher_navtest_ep.py): the
score-quality metrics of the CK EP head use the teacher run's ep_target (report 48 task 5c).  Synthetic arrays, CPU:
  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_teacher_summary_ep.py

'official' reproduces the previous formulas exactly (lab[..., CK_LABEL_IDX], the label 'ep' column); 'decoupled'
scores the EP head against ep_target.ck_targets (EP without the NC * DAC * DDC factor), every other key and the PDMS
pair accuracy unchanged; teacher_ep_target reads the run's config.json (missing key = 'official').
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from navsim.agents.para_ssr.ck.ep_target import EP_KEY_IDX, ck_targets, decoupled_ep  # noqa: E402
from tools.ck.e2e2 import summarize_teacher_navtest as S  # noqa: E402
from tools.ck.train_ck2 import key_metrics  # noqa: E402

L = Cn.LBL
CKI = list(Cn.CK_LABEL_IDX)


def _labels(n=30, k=96, seed=0):
    """official-like labels: ~30 % candidates fail NC or DAC (official EP 0 there), progress r / p > 5 m."""
    rng = np.random.default_rng(seed)
    lab = np.zeros((n, k, 9))
    nc = (rng.random((n, k)) > 0.15).astype(float)
    dac = (rng.random((n, k)) > 0.15).astype(float)
    lab[..., L["nc"]], lab[..., L["dac"]], lab[..., L["ddc"]] = nc, dac, 1.0
    lab[..., L["ttc"]] = (rng.random((n, k)) > 0.1).astype(float)
    lab[..., L["comfort"]] = (rng.random((n, k)) > 0.05).astype(float)
    r = rng.uniform(6, 40, (n, k))
    p = rng.uniform(6, 40, (n, k))
    lab[..., L["raw_progress"]], lab[..., L["pdm_progress_eff"]] = r, p
    m = nc * dac
    lab[..., L["ep"]] = np.clip(r * m / np.maximum(p, r * m), 0, 1)               # official EP (0 on a fail)
    lab[..., L["pdms"]] = m * (5 * lab[..., L["ttc"]] + 2 * lab[..., L["comfort"]] + 5 * lab[..., L["ep"]]) / 12
    return lab


def _prob(lab, ep_source, seed=1):
    rng = np.random.default_rng(seed)
    p = np.clip(lab[..., CKI] * 0.8 + 0.1 + 0.05 * rng.standard_normal(lab.shape[:-1] + (5,)), 0.01, 0.99)
    p[..., EP_KEY_IDX] = np.clip(ep_source, 0.0, 1.0)
    return p


def test_official_is_unchanged_and_decoupled_scores_the_decoupled_ep():
    lab = _labels()
    dec = decoupled_ep(lab)
    assert (np.abs(dec - lab[..., L["ep"]]) > 0.05).any()            # failing candidates differ, else equal
    np.testing.assert_allclose(dec[(lab[..., L["nc"]] * lab[..., L["dac"]]) == 1],
                               lab[..., L["ep"]][(lab[..., L["nc"]] * lab[..., L["dac"]]) == 1], atol=1e-12)
    prob = _prob(lab, dec)                                            # an EP head that learned the DECOUPLED EP
    mask = np.ones(lab.shape[:2], bool)
    mask[0, :5] = False
    # km: 'official' == the old lab[..., CK_LABEL_IDX] formula exactly; 'decoupled' = perfect EP, other keys equal
    old = key_metrics(prob[mask].reshape(-1, 5), lab[..., CKI][mask].reshape(-1, 5), np.ones(int(mask.sum()), bool))
    k_off = S.km(prob, lab, mask)
    assert k_off == old and S.km(prob, lab, mask, ep_target="official") == old
    k_dec = S.km(prob, lab, mask, ep_target="decoupled")
    assert k_dec["ep_mae"] < 1e-12 and k_dec["ep_corr"] == pytest.approx(1.0) and k_off["ep_mae"] > 0.05
    for key in old:
        if "ep" not in key.split("_"):
            assert k_dec[key] == old[key], key
    # within-token EP Pearson against the teacher's EP target; Spearman vs the official PDMS unchanged
    w_off = S.within_token(prob, lab)
    w_dec = S.within_token(prob, lab, ep_target="decoupled")
    old_ep = S.rowcorr(prob[..., EP_KEY_IDX], lab[..., L["ep"]])
    assert w_off["ep_pearson_within_mean"] == float(np.nanmean(old_ep))
    assert w_dec["ep_pearson_within_mean"] == pytest.approx(1.0) and w_off["ep_pearson_within_mean"] < 0.99
    assert w_dec["spearman_sel_pdms_mean"] == w_off["spearman_sel_pdms_mean"]
    # pair metrics: per-key sign agreement on the CK targets; official == the old label-column formula
    valid = np.ones(lab.shape[:2], bool)
    p_off = S.pair_metrics(prob, lab, valid)
    p_dec = S.pair_metrics(prob, lab, valid, ep_target="decoupled")
    c = np.arange(96)
    parent = (c // 6) * 6
    pair = valid & (c % 6 != 0)[None]
    for j, kname in enumerate(Cn.CK_KEYS):
        dy = lab[..., L[kname]] - lab[:, parent][..., L[kname]]
        dp = prob[..., j] - prob[:, parent][..., j]
        m = pair & (np.abs(dy) > 1e-6)
        assert p_off[f"pair_acc_{kname}"] == float((np.sign(dp[m]) == np.sign(dy[m])).mean()), kname
    y5 = ck_targets(lab, "decoupled")
    dy = y5[..., EP_KEY_IDX] - y5[:, parent][..., EP_KEY_IDX]
    m = pair & (np.abs(dy) > 1e-6)
    assert p_dec["pair_acc_ep"] == 1.0 == float((np.sign(prob[..., 2] - prob[:, parent][..., 2])[m] == np.sign(dy[m])).mean())
    assert p_off["pair_acc_ep"] < 1.0 and p_dec["ep_target"] == "decoupled" and p_off["ep_target"] == "official"
    for kname in ("nc", "dac", "ttc", "comfort", "pdms"):
        assert p_dec[f"pair_acc_{kname}"] == p_off[f"pair_acc_{kname}"], kname


def test_teacher_ep_target_from_run_config(tmp_path):
    for name, cfg, want in (("old", {"arm": "T"}, "official"), ("dep", {"arm": "T", "ep_target": "decoupled"},
                                                                       "decoupled"),
                            ("off", {"arm": "M", "ep_target": "official"}, "official")):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps(cfg))
        assert S.teacher_ep_target(d / "ckpt_last.pt") == want
    with pytest.raises(SystemExit, match="config.json"):
        S.teacher_ep_target(tmp_path / "missing" / "ckpt_last.pt")
    real = Path("/home/external-user/ssd/yongjae_refiner/ck/ck2/train")
    for run, want in (("ck2T", "official"), ("ck2T10dep", "decoupled"), ("ck2M10dep", "decoupled")):
        if (real / run / "config.json").is_file():
            assert S.teacher_ep_target(real / run / "ckpt_last.pt") == want, run
