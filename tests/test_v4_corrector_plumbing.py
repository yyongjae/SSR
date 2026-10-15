"""refiner v4 corrector A training plumbing: rec2 edit block, CandRecorder2 with edits, labeler2 scoring the 96 + 8
in one call into the optional generation edit files, LabelStore2(with_edits), and the student's on-policy loss with
the recorded edits (critic BCE on the edits + AWR).  CPU only (fake scorer, smoke teachers):
  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_corrector_plumbing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
import test_ck2_e2e_labeler as TLAB  # noqa: E402
import test_ck2_e2e_loss as TLOSS  # noqa: E402
from tools.ck.e2e import labeler as L1  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data as ED  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402

KE = D2.EDIT_K


def _edits(n, seed=0):
    g = np.random.default_rng(seed)
    return {"traj": g.normal(size=(n, KE, 8, 3)).astype(np.float32),
            "src": g.integers(0, 96, (n, KE)).astype(np.int16), "u": g.normal(size=(n, KE, 12)).astype(np.float32),
            "valid": g.random((n, KE)) > 0.2}


def test_rec2_edit_block_roundtrip_and_refusals(tmp_path):
    traj, valid = TLAB._rec(3)
    e = _edits(3)
    p = tmp_path / "c.npz"
    args = ([0, 1, 2], traj, valid, np.zeros((3, 16), np.int16), np.ones((3, 16), np.float32), np.ones(3, np.float32),
            0, 5, 0)
    D2.write_rec2_chunk(p, *args, edits=e)
    d = D2.read_rec2_chunk(p)
    assert np.array_equal(d["edit_traj"], e["traj"]) and np.array_equal(d["edit_src"], e["src"])
    D2.write_rec2_chunk(tmp_path / "plain.npz", *args)                        # no block: the old chunk exactly
    assert "edit_traj" not in D2.read_rec2_chunk(tmp_path / "plain.npz")
    a = D2._rec2_arrays(*args, edits=e)
    a.pop("edit_u")
    with pytest.raises(ValueError, match="partial edit block"):
        D2._check_rec2(a)
    rec = D2.CandRecorder2(tmp_path / "io", 0, 1, 5, chunk_tokens=2)
    rec.add([0, 1], traj[:2], valid[:2], np.zeros((2, 16)), np.ones((2, 16)), np.ones(2), 1,
            edits={k: v[:2] for k, v in e.items()})
    with pytest.raises(ValueError, match="edits"):
        rec.add([2], traj[2:], valid[2:], np.zeros((1, 16)), np.ones((1, 16)), np.ones(1), 1)


def test_labeler2_scores_edits_with_the_96(tmp_path):
    io = tmp_path / "ck_e2e2"
    tok = TLAB._tokens(tmp_path)
    rows = [0, 2, 4]
    traj, valid = TLAB._rec(3, seed=7)
    e = _edits(3, seed=8)
    p = ED.rec_chunk_path(io, 6, 0, "eeee0000", 0)
    D2.write_rec2_chunk(p, rows, traj, valid, np.zeros((3, 16), np.int16), np.ones((3, 16), np.float32),
                        np.full(3, 3.0, np.float32), np.arange(3), 6, 0, edits=e)
    assert TLAB._lab(io, tok).run() == 0
    g = D2.open_generation2(io, 6, TLAB.N_FAKE, "r", optional=D2.GEN2_EDIT_FILES)
    for i, r in enumerate(rows):
        # the 96 exactly as without edits (fake scorer, batch-independent)
        exp = L1.fake_scores(traj[i])
        assert np.array_equal(TLAB._bits(np.asarray(g["labels"][r])), TLAB._bits(exp))
        assert np.array_equal(np.asarray(g["edit_traj"][r]), e["traj"][i])
        v = e["valid"][i]
        assert np.array_equal(np.asarray(g["edit_ok"][r]), v)
        assert np.array_equal(TLAB._bits(np.asarray(g["edit_labels"][r])[v]),
                              TLAB._bits(L1.fake_scores(e["traj"][i][v])))
        assert np.isnan(np.asarray(g["edit_labels"][r])[~v]).all()
        assert np.array_equal(np.asarray(g["edit_src"][r]), e["src"][i])
    # the label store returns them (with the sources and parents from the generation's 96)
    ls = D2.LabelStore2(io, n_rows=TLAB.N_FAKE, packed=D2.WARMUP_DEFAULT["packed"], with_edits=True)
    ls.refresh(6)
    G = ls.lookup(torch.tensor(rows), [f"tok{r:04d}" for r in rows], 7, 0)
    assert G["E_ok"].shape == (3, KE) and torch.equal(G["E_ok"][0], torch.from_numpy(e["valid"][0]))
    s0 = int(e["src"][0][0])
    assert torch.equal(G["E_src_traj"][0, 0], torch.from_numpy(traj[0, s0]))
    assert torch.equal(G["E_src_par"][0, 0], torch.from_numpy(traj[0, (s0 // 6) * 6]))
    assert G["G_pdms"].shape == (3, 48)


@pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")
def test_onpolicy_loss_with_edits_and_awr(tmp_path):
    V4 = {"readout": "hazard", "pref_head": True, "corridor_version": 2, "kd_parent_only": True, "corrector": True,
          "lambda_awr": 1.0}
    c, ck, gtraj, glab = TLOSS._onpolicy(tmp_path, lat_kd={"start_mb": 0}, v4=V4, rec_chunk_tokens=1)
    # add an edit block (fake official labels) to the row-3 generation used by _onpolicy
    g = D2.open_generation2(c.io_dir, 4, 16, "r+", optional=D2.GEN2_EDIT_FILES)
    rng = np.random.default_rng(0)
    src = np.arange(KE, dtype=np.int16) * 6 + 1
    g["edit_traj"][3] = gtraj[0][src] + 0.1
    lab = np.array(g["labels"][3][src])
    lab[:, 6] = np.clip(lab[:, 6] + rng.normal(0, 0.2, KE), 0, 1)                 # edits better / worse in PDMS
    g["edit_labels"][3] = lab
    g["edit_ok"][3] = True
    g["edit_src"][3] = src
    g["edit_u"][3] = rng.normal(0, 0.3, (KE, 12)).astype(np.float32)
    for k in D2.GEN2_EDIT_FILES:
        g[k].flush()
    # the harness builds its own 16-row store; a corrector run's CKE2E2.labels() sets with_edits from the config
    ck._labels = D2.LabelStore2(c.io_dir, n_rows=16, packed=c.warmup["packed"], var_sampling=c.var_sampling,
                                ep_target=c.ep_target, with_edits=True, with_pref=True)
    st = TLOSS._student(c)
    assert st.corrector is not None
    ck.epoch, ck.epoch_frac, ck.gstep = 5, 5.5, 3
    ck.recorder = D2.CandRecorder2(c.io_dir, 0, 1, 5, c.rec_chunk_tokens)
    TLOSS._spy(ck, "current_batch")
    f, t = TL.fresh(2, rows=[3, 7])
    pred = TL.fake_predictions(2, seed=3)
    loss, logs = ck.loss(st, f, t, pred)
    assert torch.isfinite(loss)
    S = ck._spy["current_batch"]
    assert S["extra"].shape == (2, 48 + KE, 8, 3) and bool(S["ok_lab"][0, 48:].all())
    assert not bool(S["ok_lab"][1, 48:].any())                                    # fallback row: no edits
    assert float(logs["v4/awr_n"]) == KE and torch.isfinite(torch.as_tensor(logs["v4/awr"]))
    loss.backward()
    assert float(sum(p.grad.abs().sum() for p in st.corrector.parameters() if p.grad is not None)) > 0
    # the step's record carries the corrector's edits
    ck.recorder.close_epoch()
    d = D2.read_rec2_chunk(Path(c.io_dir) / "rec" / "ep005" / "r0" / ck.recorder.chunks[0])
    assert d["edit_traj"].shape == (1, KE, 8, 3) and (d["edit_src"] >= 0).all()
