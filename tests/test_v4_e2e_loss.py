"""refiner v4 student inside CKE2E2.loss (the v2-free harness of tests/test_ck2_e2e_loss.py): hazard readout + P_ref
head + corridor version 2 + parent-only score KD, warm-up and on-policy steps -> finite loss, gradients on every
trainable student parameter (v4 heads included), the parents of every candidate set, the KD mask.  CPU only:
  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_e2e_loss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
import test_ck2_e2e_loss as TLOSS  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import losses as L  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck.online import student_topk  # noqa: E402

pytestmark = pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")
V4 = {"readout": "hazard", "pref_head": True, "corridor_version": 2, "kd_parent_only": True}
FROZEN_V4 = ("trunk.lon_head.", "trunk.gate_head.", "trunk.lat_head.", "score_head.")


def _check_grads_v4(student, bev):
    for n, p in student.named_parameters():
        if n.startswith(FROZEN_V4):
            assert not p.requires_grad and p.grad is None, n
        elif n.startswith("pref_head."):
            continue                       # no P_ref target in this harness (the head is only touched, 0 weight)
        else:
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()), n
    assert bev.grad is not None and bool(torch.isfinite(bev.grad).all()) and float(bev.grad.abs().sum()) > 0


def _spy_kd(monkeypatch):
    seen = {}
    fn = L.score_kd_bce

    def w(s, p, ok):
        seen["ok"] = ok.clone()
        return fn(s, p, ok)
    monkeypatch.setattr(L, "score_kd_bce", w)
    return seen


def test_v4_warmup_step(tmp_path, monkeypatch):
    c = TL.cfg2(tmp_path, v4=V4)
    ck = O.CKE2E2(c)
    st = TLOSS._student(c)
    assert st.v4_readout is not None and st.corridor_version == 2
    TLOSS._spy(ck, "warmup_batch")
    TLOSS._spy_teacher(ck)
    seen = _spy_kd(monkeypatch)
    f, t = TL.fresh(2)
    pred = TL.fake_predictions(2, seed=0)
    loss, logs = ck.loss(st, f, t, pred)
    assert torch.isfinite(loss) and float(logs["ck2/n_cand"]) == 64.0
    S = ck._spy["warmup_batch"]
    # raw anchors are their own parents; every variant's parent is an anchor of the table
    assert torch.equal(S["cand_parent"][:, :32], S["cand"][:, :32])
    anc = ck.sampler().anchors_t("cpu")
    vp = S["cand_parent"][:, 32:].reshape(-1, 24)
    hit = (vp[:, None] == anc.reshape(1, 256, 24)).all(-1).any(-1)
    ok_v = S["cand_ok"][:, 32:].reshape(-1)
    assert bool(hit[ok_v].all())
    # parent-only KD: the KD mask is the teachers' ok on the raw anchors only (vtype 0)
    assert torch.equal(seen["ok"], ck._spy["T"]["kd_ok"].bool() & S["kd_tok_ok"][:, None] & (S["vtype_kd"] == 0))
    loss.backward()
    _check_grads_v4(st, pred["bev_embed"])


def test_v4_onpolicy_step(tmp_path, monkeypatch):
    c, ck, gtraj, glab = TLOSS._onpolicy(tmp_path, lat_kd={"start_mb": 0}, v4=V4)
    st = TLOSS._student(c)
    ck.epoch, ck.epoch_frac, ck.gstep = 5, 5.5, 3
    TLOSS._spy(ck, "current_batch")
    TLOSS._spy_teacher(ck)
    seen = _spy_kd(monkeypatch)
    f, t = TL.fresh(2, rows=[3, 7])                     # row 3 has generation 4, row 7 none -> fallback
    pred = TL.fake_predictions(2, seed=3)
    v2 = (pred["bev_embed"] ** 2).mean()
    loss, logs = ck.loss(st, f, t, pred, v2_loss=v2)
    assert torch.isfinite(loss) and float(logs["ck2/n_cand"]) == 96.0
    S = ck._spy["current_batch"]
    top = student_topk(pred, 16)
    # KD set: the 16 identities are their own parents; a variant's parent is its top-16 rank's draft
    assert torch.equal(S["cand_parent"][:, :16], top["cand"].float())
    assert torch.equal(S["vtype_kd"][:, :16], torch.zeros(2, 16, dtype=torch.long))
    # label set, generation row: identities self, variant column c -> generation identity c // 6
    G0 = S["extra"][0]
    assert torch.equal(S["extra_parent"][0, :16], G0[:16])
    for i in range(16, 48):
        if bool(S["ok_lab"][0, i]):
            k = None
            for kk in range(16):
                if torch.equal(S["extra_parent"][0, i], G0[kk]):
                    k = kk
            assert k is not None, i
    assert torch.equal(seen["ok"], ck._spy["T"]["kd_ok"].bool() & (S["vtype_kd"] == 0))
    (v2 + loss).backward()
    _check_grads_v4(st, pred["bev_embed"])
    # inference over the 96 pool with parents: finite, same layout
    st.eval()
    with torch.no_grad():
        res = ck.infer(st, f, {k: (v.detach() if torch.is_tensor(v) else v) for k, v in pred.items()})
    assert res["ck2_score_logit"].shape == (2, 96, 5) and torch.isfinite(res["ck2_score_logit"]).all()
