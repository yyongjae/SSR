"""CK2 e2e inference outputs (SPEC s5-1): CKE2E2.infer over the 96-candidate pool (v2 top-16 x 6 variants, c = k*6 + v).
CPU only:
  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_infer.py \
    --basetemp=/tmp/claude-0/-workspace/f9196902-3a88-4418-a505-cd713226ed4a/scratchpad/ck2e2e/pytest/infer

Shapes / dtypes of every ck2_* key, pool layout (identity columns = top-16, column 0 = v2's trajectory), the lateral
head applied with z_lon = 0 (slope 0) to every column, identity decode when w_lat = 0, the CK2 logits of a column equal
the training-time logits of the same trajectory (one scene, independent candidates), and (when ck/select2.py exists)
the optional infer_select path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck.online import bev_sgrid, student_topk  # noqa: E402

pytestmark = pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")
SHAPES = {"ck2_cand_idx": ((16,), torch.int64), "ck2_v2_final": ((16,), torch.float32),
          "ck2_v2_im": ((16,), torch.float32), "ck2_v2_sim": ((16, 5), torch.float32),
          "ck2_cand96": ((96, 8, 3), torch.float32), "ck2_valid96": ((96,), torch.bool),
          "ck2_score_logit": ((96, 5), torch.float32), "ck2_w_lat": ((96, 6), torch.float32),
          "ck2_e_lat": ((96, 6), torch.float32), "ck2_lat_traj": ((96, 8, 3), torch.float32)}


def _setup(tmp_path, perturb=True, **over):
    c = TL.cfg2(tmp_path, **over)
    ck = O.CKE2E2(c)
    torch.manual_seed(0)
    st = O.build_student_ck2(c).eval()
    if perturb:
        with torch.no_grad():
            for p in st.trunk.lat_head.parameters():
                p.add_(0.05 * torch.randn_like(p))
    return c, ck, st


def test_infer_outputs_layout_and_lateral(tmp_path):
    from navsim.agents.para_ssr.ck.correct import correct
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    c, ck, st = _setup(tmp_path)
    f, _ = TL.fresh(2)
    pred = TL.fake_predictions(2, seed=6, requires_grad=False)
    out = ck.infer(st, f, pred)
    for k, (shape, dt) in SHAPES.items():
        assert out[k].shape == (2,) + shape and out[k].dtype == dt, (k, out[k].shape, out[k].dtype)
    assert set(out) == set(SHAPES)                                   # infer_select off: no selection keys
    top = student_topk(pred, 16)
    assert torch.equal(out["ck2_cand96"][:, ::6], top["cand"]) and torch.equal(out["ck2_cand_idx"], top["idx"])
    assert torch.equal(out["ck2_cand96"][:, 0], pred["trajectory"])          # column 0 = v2's submitted trajectory
    assert out["ck2_valid96"][:, ::6].all()
    V = O.online_variants(top["cand"], O.v0_from_status(f["status_feature"]), c.variants)
    assert torch.equal(out["ck2_cand96"], V["traj96"]) and torch.equal(out["ck2_valid96"], V["valid96"])
    v0 = ego_inputs(f["status_feature"].float())[0]
    dec = correct(out["ck2_cand96"], torch.zeros_like(out["ck2_w_lat"]), out["ck2_w_lat"], v0, 0.0)
    torch.testing.assert_close(out["ck2_lat_traj"], dec["traj"], rtol=0, atol=0)
    torch.testing.assert_close(out["ck2_e_lat"], dec["e_lat"][..., 2:], rtol=0, atol=0)
    assert float(out["ck2_w_lat"].abs().max()) > 0
    # the CK2 logits of every column = a forward on that column set (candidates are independent)
    with torch.no_grad():
        o = O.student_forward2(st, bev_sgrid(pred["bev_embed"]), out["ck2_cand96"][:, 10:30], f["status_feature"])
    torch.testing.assert_close(o["score_logit"], out["ck2_score_logit"][:, 10:30], rtol=1e-4, atol=1e-5)
    # w_lat = 0 (fresh student): the lateral decode is the identity
    _, ck0, st0 = _setup(tmp_path, perturb=False)
    o0 = ck0.infer(st0, f, pred)
    assert torch.equal(o0["ck2_lat_traj"], o0["ck2_cand96"]) and float(o0["ck2_e_lat"].abs().max()) == 0.0


def test_training_kd_columns_match_the_inference_pool(tmp_path):
    """The on-policy KD candidates (top-16 + 32 sampled variants) are columns of the inference pool (same f32 values),
    so the critic trained on them scores the same trajectories at inference."""
    c, ck, st = _setup(tmp_path)
    ck.epoch = 5
    f, t = TL.fresh(2, rows=[3, 7])
    pred = TL.fake_predictions(2, seed=8, requires_grad=False)
    toks = ck.tokens_of(t["ck_row"])
    fb = student_topk(pred, 1)["cand"][:, 0]
    top = student_topk(pred, 16)
    v0v = O.v0_from_status(f["status_feature"])
    V = O.online_variants(top["cand"], v0v, c.variants)
    P = O.C2.onpolicy_cands(top["cand"], v0v, toks, 5, c.seed, c.n_var_step, c.variants, c.var_sampling, O.STREAM_NOW,
                            V=V)
    out = ck.infer(st, f, pred)
    cols = P["cols"]
    pool = out["ck2_cand96"].gather(1, cols[..., None, None].expand(-1, -1, 8, 3))
    assert torch.equal(P["cand"][:, 16:], pool) and torch.equal(P["cand"][:, :16], out["ck2_cand96"][:, ::6])
    with torch.no_grad():
        o = O.student_forward2(st, bev_sgrid(pred["bev_embed"]), P["cand"], f["status_feature"])
    lg = out["ck2_score_logit"].gather(1, cols[..., None].expand(-1, -1, 5))
    torch.testing.assert_close(o["score_logit"][:, 16:], lg, rtol=1e-4, atol=1e-5)
    assert fb.shape == (2, 8, 3)


@pytest.mark.skipif(not (Path(O.__file__).parent / "select2.py").is_file(), reason="ck/select2.py not written yet")
def test_infer_select_optional(tmp_path):
    c, ck, st = _setup(tmp_path, infer_select={"beta": 0.0, "set": "all", "lat_mode": "off"})
    f, _ = TL.fresh(2)
    pred = TL.fake_predictions(2, seed=9, requires_grad=False)
    out = ck.infer(st, f, pred)
    assert out["ck2_sel_idx"].shape == (2,) and out["ck2_traj"].shape == (2, 8, 3)
    # beta 0 = v2 only; variants inherit the parent's v2 score and ties go to the identity -> column 0, v2 trajectory
    assert out["ck2_sel_idx"].tolist() == [0, 0]
    assert torch.equal(out["ck2_traj"], pred["trajectory"])
