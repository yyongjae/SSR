"""CK2 e2e T7: CKE2E2.loss with a v2-free harness (random bev_embed requiring grad, the CK2 2-token fixture targets,
the CK2 smoke teachers, the real CKE2E2 / cands2 / e2e_data2 pieces).  CPU only:
  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_loss.py \
    --basetemp=/tmp/claude-0/-workspace/f9196902-3a88-4418-a505-cd713226ed4a/scratchpad/ck2e2e/pytest/loss

Covers: warm-up (64 candidates) / warmup_record (records the 96 of the student's top-16) / on-policy (current 48 for
KD + 48 of the label generation for BCE = 96 in one pass; rows without a generation -> fallback G_src 0) losses finite
with gradients on every trainable student parameter and on bev_embed (x bev_grad_scale); lon / gate heads get no
gradient; z_lon = 0 in every decode; the lateral KD = ONE L1 to the mean DET / MAP target; lateral-KD weight 0 before
lat_kd.start_mb; a token without warm-up data contributes nothing; non-finite CK loss dropped; grad-share logging.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck.constants import SUR_TERMS  # noqa: E402
from navsim.agents.para_ssr.ck.online import student_topk  # noqa: E402

pytestmark = pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")
FROZEN = ("trunk.lon_head.", "trunk.gate_head.")


def _student(c, seed=0):
    torch.manual_seed(seed)
    return O.build_student_ck2(c)


def _spy(ck, name):
    """wrap a CKE2E2 method and keep its last return value in ck._spy[name]."""
    fn = getattr(ck, name)
    ck.__dict__.setdefault("_spy", {})

    def w(*a, **k):
        out = fn(*a, **k)
        ck._spy[name] = out
        return out
    setattr(ck, name, w)


def _spy_teacher(ck):
    tp = ck.teachers(torch.device("cpu"))
    run = tp.run
    ck.__dict__.setdefault("_spy", {})

    def w(*a, **k):
        out = run(*a, **k)
        ck._spy["T"] = out
        return out
    tp.run = w


def _check_grads(student, bev):
    for n, p in student.named_parameters():
        if n.startswith(FROZEN):
            assert not p.requires_grad and p.grad is None, n
        else:
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()), n
    assert bev.grad is not None and bool(torch.isfinite(bev.grad).all()) and float(bev.grad.abs().sum()) > 0


# ----------------------------------------------------------------------------------------------- warm-up
def test_warmup_loss_backward_and_masks(tmp_path):
    c = TL.cfg2(tmp_path)
    ck = O.CKE2E2(c)
    st = _student(c)
    _spy(ck, "warmup_batch")
    _spy_teacher(ck)
    f, t = TL.fresh(2)
    pred = TL.fake_predictions(2, seed=0)
    loss, logs = ck.loss(st, f, t, pred)
    assert torch.isfinite(loss) and float(logs["ck2/loss"]) > 0 and ck.mb == 1
    for k, v in logs.items():
        assert torch.isfinite(v).all(), k
    assert float(logs["ck2/phase"]) == 0.0 and float(logs["ck2/n_cand"]) == 64.0 and float(logs["ck2/r"]) == 1.0
    assert float(logs["ck2/w_lat"]) == 0.0 and float(logs["ck2/wt_lat_kd"]) == 0.0     # before lat_kd.start_mb 1000
    assert float(logs["ck2/kd_score"]) > 0 and float(logs["ck2/lat_kd"]) > 0 and float(logs["ck2/n_gt"]) == 2.0
    for k in SUR_TERMS:
        assert f"ck2/t_{k}" in logs
    S = ck._spy["warmup_batch"]
    assert S["cand"].shape == (2, 64, 8, 3) and S["extra"] is None and S["sur_pos"].shape == (2, 16)
    g = S["grp"]
    assert ((g[:, :32] <= 1).sum(1) == 16).all() and (g[:, 32:] == 3).all()
    assert torch.equal(S["lat_mask"], g != 2)                                         # strat anchors excluded (NU16)
    assert (g.gather(1, S["sur_pos"]) <= 1).all()                                     # surrogate on near + mid only
    T = ck._spy["T"]
    assert T["kd_ok"].shape == (2, 64)
    expect = (sum(float(logs[k]) for k in ("ck2/wt_bce", "ck2/wt_sur", "ck2/wt_kd_score", "ck2/wt_lat_kd")))
    assert abs(float(loss) - expect) < 1e-5
    loss.backward()
    _check_grads(st, pred["bev_embed"])


def test_bev_gradient_proportional_to_scale(tmp_path):
    f, t = TL.fresh(2)
    grads = []
    for g in (1.0, 0.25):
        c = TL.cfg2(tmp_path, bev_grad_scale=g)
        ck = O.CKE2E2(c)
        st = _student(c)
        assert st.trunk.adapter.bev_grad_scale == g
        pred = TL.fake_predictions(2, seed=0)
        loss, _ = ck.loss(st, f, t, pred)
        loss.backward()
        grads.append(pred["bev_embed"].grad.clone())
    assert grads[0].abs().sum() > 0
    torch.testing.assert_close(grads[1], 0.25 * grads[0], rtol=1e-5, atol=1e-10)


def test_lateral_kd_is_one_l1_on_the_mean_target_and_z_lon_zero(tmp_path):
    from navsim.agents.para_ssr.ck.correct import correct
    from navsim.agents.para_ssr.ck.online import bev_sgrid
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    c = TL.cfg2(tmp_path, lat_kd={"start_mb": 0})
    ck = O.CKE2E2(c)
    st = _student(c)
    with torch.no_grad():                                     # a non-identity lateral head so e_s != 0
        for p in st.trunk.lat_head.parameters():
            p.add_(0.05 * torch.randn_like(p))
    _spy(ck, "warmup_batch")
    _spy_teacher(ck)
    f, t = TL.fresh(2)
    pred = TL.fake_predictions(2, seed=4)
    _, logs = ck.loss(st, f, t, pred)
    S, T = ck._spy["warmup_batch"], ck._spy["T"]
    with torch.no_grad():
        out = O.student_forward2(st, bev_sgrid(pred["bev_embed"]), S["cand"], f["status_feature"].float())
        v0 = ego_inputs(f["status_feature"].float())[0]
        assert float(out["w_lat"].abs().max()) > 0
        z = st.trunk.lon_head(torch.randn(3, 2 * st.trunk.d_model))
        assert float(z.abs().max()) == 0.0                                            # z_lon == 0
        e_s = correct(S["cand"], torch.zeros_like(out["w_lat"]), out["w_lat"], v0, 0.1)["e_lat"][..., 2:]
        tgt = 0.5 * (T["e_det"] + T["e_map"])
        m = T["kd_ok"] & S["kd_tok_ok"][:, None] & S["lat_mask"]
        manual = ((e_s - tgt).abs().mean(-1) * m.float()).sum() / m.float().sum()
        mean_of_two = 0.5 * (((e_s - T["e_det"]).abs().mean(-1) * m).sum() + ((e_s - T["e_map"]).abs().mean(-1)
                                                                             * m).sum()) / m.float().sum()
    assert abs(float(logs["ck2/lat_kd"]) - float(manual)) < 1e-6
    assert abs(float(manual) - float(mean_of_two)) > 1e-4          # differs from the mean of two L1s
    assert float(logs["ck2/w_lat"]) > 0                            # start_mb 0: EMA weight active


def test_lat_kd_weight_schedule(tmp_path):
    c = TL.cfg2(tmp_path)
    ck = O.CKE2E2(c)
    ck.ema.update(0.5, 0.25)
    # default lat_kd.ratio 0.5 (user 2026-10-08): w = 0.5 * (0.5 / 0.25) = 1.0
    assert ck.ema.weight(999) == 0.0 and ck.ema.weight(1000) == pytest.approx(1.0)
    c2 = TL.cfg2(tmp_path, lat_kd={"balance": "fixed", "fixed": 0.3, "start_mb": 2})
    ck2 = O.CKE2E2(c2)
    st = _student(c2)
    f, t = TL.fresh(1)
    ws = []
    for i in range(3):
        _, logs = ck2.loss(st, f, t, TL.fake_predictions(1, seed=i))
        ws.append(float(logs["ck2/w_lat"]))
    assert ws[:2] == [0.0, 0.0] and ws[2] == pytest.approx(0.3)


def test_token_without_warmup_data_contributes_nothing(tmp_path):
    """ck2_wu_ok False -> every CK term masked for that token (NU44): the loss terms equal those of the other token
    alone (per-token computations are independent)."""
    c = TL.cfg2(tmp_path, lat_kd={"start_mb": 0})
    f, t = TL.fresh(2)
    t["ck2_wu_ok"][1] = False
    pred = TL.fake_predictions(2, seed=2)
    ck = O.CKE2E2(c)
    st = _student(c)
    _, l2 = ck.loss(st, f, t, pred)
    f1, t1 = TL.fresh(1)
    ck1 = O.CKE2E2(c)
    _, l1 = ck1.loss(st, f1, t1, TL.slice_pred(pred, slice(0, 1)))
    for k in ("ck2/bce", "ck2/kd_score", "ck2/sur", "ck2/lat_kd", "ck2/loss"):
        assert abs(float(l2[k]) - float(l1[k])) < 2e-5 * max(1.0, abs(float(l1[k]))), (k, float(l2[k]), float(l1[k]))
    assert float(l2["ck2/n_gt"]) == 1.0 and float(l2["ck2/kd_ok_frac"]) == pytest.approx(0.5)


def test_no_map_bev_kd_off(tmp_path):
    c = TL.cfg2(tmp_path)
    f, t = TL.fresh(2)
    t["kd_ok_1"][0] = False
    ck = O.CKE2E2(c)
    _, logs = ck.loss(_student(c), f, t, TL.fake_predictions(2, seed=1))
    assert float(logs["ck2/kd_ok_frac"]) == pytest.approx(0.5)


# ----------------------------------------------------------------------------------------------- recording
def test_warmup_record_writes_the_96_of_the_top16(tmp_path):
    c = TL.cfg2(tmp_path, rec_chunk_tokens=2)
    ck = O.CKE2E2(c)
    st = _student(c)
    ck.epoch, ck.epoch_frac, ck.gstep = 4, 4.0, 17
    assert ck.phase(4) == "warmup_record" and ck.recording(4)
    ck.recorder = D2.CandRecorder2(c.io_dir, 0, 2, 4, c.rec_chunk_tokens)
    f, t = TL.fresh(2, rows=[11, -1])
    pred = TL.fake_predictions(2, seed=10)
    _, logs = ck.loss(st, f, t, pred)
    assert float(logs["ck2/phase"]) == 1.0 and float(logs["ck2/rec_tokens"]) == 1.0
    assert float(logs["ck2/n_cand"]) == 64.0                                   # the loss is still the warm-up one
    _, _ = ck.loss(st, f, t, TL.fake_predictions(2, seed=11))
    rec = ck.recorder
    assert rec.n_tokens == 2 and len(rec.chunks) == 1
    from navsim.agents.para_ssr.ck import e2e_data as ED
    a = D2.read_rec2_chunk(Path(ED.rec_rank_dir(c.io_dir, 4, 0)) / rec.chunks[0])
    assert a["row"].tolist() == [11, 11] and a["gstep"].tolist() == [17, 17] and int(a["fmt"]) == 2
    top = student_topk(pred, 16)
    V = O.online_variants(top["cand"], O.v0_from_status(f["status_feature"]), c.variants)
    assert np.array_equal(a["traj"][0], V["traj96"][0].numpy())                   # exactly the step's f32 values
    assert np.array_equal(a["traj"][0, ::6], top["cand"][0].numpy())
    assert np.array_equal(a["valid"][0], V["valid96"][0].numpy())
    assert np.array_equal(a["cand_idx"][0], top["idx"][0].numpy().astype(np.int16))
    assert np.array_equal(a["v2_final"][0], top["final"][0].numpy())


# ----------------------------------------------------------------------------------------------- on-policy
def _onpolicy(tmp_path, **over):
    c = TL.cfg2(tmp_path, **over)
    ck = O.CKE2E2(c, max_epochs=30)
    n_rows = 16
    ck._labels = D2.LabelStore2(c.io_dir, n_rows=n_rows, packed=c.warmup["packed"], var_sampling=c.var_sampling,
                                ep_target=c.ep_target)
    gtraj, glab, _ = TL.make_gen(c.io_dir, 4, [3], n_rows)
    return c, ck, gtraj, glab


def test_onpolicy_loss_label_set_and_current_candidates(tmp_path):
    c, ck, gtraj, glab = _onpolicy(tmp_path, lat_kd={"start_mb": 0}, rec_chunk_tokens=1)
    st = _student(c)
    ck.epoch, ck.epoch_frac, ck.gstep = 5, 5.5, 3
    ck.recorder = D2.CandRecorder2(c.io_dir, 0, 1, 5, c.rec_chunk_tokens)
    _spy(ck, "current_batch")
    _spy_teacher(ck)
    f, t = TL.fresh(2, rows=[3, 7])                     # row 3 has generation 4, row 7 none -> fallback
    pred = TL.fake_predictions(2, seed=3)
    v2 = (pred["bev_embed"] ** 2).mean()
    loss, logs = ck.loss(st, f, t, pred, v2_loss=v2)
    assert ck._labels.max_epoch == 4                                   # epoch 5, lag 1
    assert torch.isfinite(loss) and float(logs["ck2/phase"]) == 2.0 and float(logs["ck2/n_cand"]) == 96.0
    assert float(logs["ck2/G_src_prev"]) == 0.5 and float(logs["ck2/G_src_fallback"]) == 0.5
    for k, v in logs.items():
        assert torch.isfinite(v).all(), k
    S = ck._spy["current_batch"]
    top = student_topk(pred, 16)
    assert S["cand"].shape == (2, 48, 8, 3) and S["extra"].shape == (2, 48, 8, 3)
    assert torch.equal(S["cand"][:, :16], top["cand"])                 # KD candidates: current top-16 first
    assert ck._spy["T"]["kd_ok"].shape == (2, 48)                     # teachers only on the current 48
    # BCE set of row 3 = generation 4: its 16 identity columns + 32 variants
    assert np.array_equal(S["extra"][0, :16].numpy(), gtraj[0, ::6])
    assert torch.equal(S["vtype_lab"][0, :16], torch.zeros(16, dtype=torch.long))
    assert (S["vtype_lab"][0, 16:] > 0).all()
    np.testing.assert_array_equal(S["y"][0, :16].numpy(), D2.ck_targets(glab[0, ::6], c.ep_target))
    # fallback row (no generation): 16 near+mid anchors + 32 variants of the warm-up files, official labels
    assert (S["vtype_lab"][1, :16] == 0).all() and S["ok_lab"][1].any()
    assert int(S["src"][1]) == O.SRC_FALLBACK and int(S["src"][0]) == O.SRC_PREV
    # surrogate on the 16 current originals
    assert torch.equal(S["sur_pos"][0], torch.arange(16))
    # recorded: the 96 of this step's top-16
    assert ck.recorder.n_tokens == 2
    (v2 + loss).backward()
    _check_grads(st, pred["bev_embed"])


def test_onpolicy_fallback_none_and_kd_ramp(tmp_path):
    c, ck, _, _ = _onpolicy(tmp_path, onpolicy_label={"fallback": "none"}, kd_ramp_epochs=2.0)
    st = _student(c)
    ck.epoch, ck.epoch_frac = 5, 5.5
    assert ck.r(5.5) == pytest.approx(0.25) and ck.r(4.5) == 1.0 and ck.r(8.0) == 1.0
    f, t = TL.fresh(2, rows=[3, 7])
    _, logs = ck.loss(st, f, t, TL.fake_predictions(2, seed=5))
    assert float(logs["ck2/G_src_nodata"]) == 0.5 and float(logs["ck2/r"]) == pytest.approx(0.25)
    assert float(logs["ck2/G_ok_frac"]) <= 0.5 + 1e-6


# ----------------------------------------------------------------------------------------------- non-finite / gshare
def test_nonfinite_ck_loss_is_dropped(tmp_path):
    c = TL.cfg2(tmp_path, lat_kd={"start_mb": 0})
    ck = O.CKE2E2(c)
    st = _student(c)
    f, t = TL.fresh(2)
    pred = TL.fake_predictions(2, requires_grad=False)
    bev = pred["bev_embed"].clone()
    bev[0, 10, 3] = float("inf")
    pred["bev_embed"] = bev.requires_grad_(True)
    loss, logs = ck.loss(st, f, t, pred)
    assert float(loss) == 0.0 and float(logs["ck2/loss_nonfinite"]) == 1.0 and ck.cum["nonfinite_loss"] == 1
    loss.backward()
    assert all(p.grad is not None for p in O.ck2_parameters(type("A", (), {"ck_student": st})))


def test_grad_share_logged_every_n(tmp_path):
    c = TL.cfg2(tmp_path, grad_share_every=2)
    ck = O.CKE2E2(c)
    st = _student(c)
    f, t = TL.fresh(1)
    for i in range(3):
        pred = TL.fake_predictions(1, seed=i)
        v2 = (pred["bev_embed"] ** 2).mean()
        _, logs = ck.loss(st, f, t, pred, v2_loss=v2)
        has = "ck2/gshare" in logs
        assert has == (i % 2 == 0)
        if has:
            assert 0 < float(logs["ck2/gshare"]) < 1 and float(logs["gnorm/bev_v2"]) > 0
            assert float(logs["gnorm/bev_ck"]) > 0 and "gnorm/bev_bevkd" not in logs
