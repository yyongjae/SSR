"""CK2 e2e T6: TeacherPair2 on the CK2 smoke teacher checkpoints (CPU fp32): load-time guards (arm, trainer
train_ck2, done.json, zero lon head), output shapes, lateral KD target = MEAN of the DET and MAP decoded offsets
(z_lon = 0, slope 0), score KD source per key (nc / ttc DET, dac MAP, ep / comfort mean), kd_ok rules (no MAP BEV ->
KD off for the token; cand_ok).
  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_teacher.py \
    --basetemp=/tmp/claude-0/-workspace/f9196902-3a88-4418-a505-cd713226ed4a/scratchpad/ck2e2e/pytest/teacher
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck.constants import CK_KEYS  # noqa: E402

pytestmark = pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")


@pytest.fixture(scope="module")
def pair():
    return O.TeacherPair2(str(TL.SMOKE_T), str(TL.SMOKE_M), "last", "cpu", amp=False, require_done=True)


def _cands(t, n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.from_numpy(np.load(TL.ANCHORS)).float()
    idx = torch.randint(0, 256, (t["ck2_gt"].shape[0], n), generator=g)
    return a[idx] + 0.05 * torch.randn(idx.shape + (8, 3), generator=g)


def test_load_guards_and_frozen(pair):
    for net, arm in ((pair.det, "T"), (pair.map, "M")):
        assert net.arm == arm and not net.training
        assert all(not p.requires_grad for p in net.parameters())
        lh = net.trunk.lon_head[-1]
        assert float(lh.weight.abs().max()) == 0.0 and float(lh.bias.abs().max()) == 0.0
    assert pair.det_cfg["trainer"] == "train_ck2" and pair.map_cfg["trainer"] == "train_ck2"


def _clone_run(src: Path, dst: Path, drop=(), cfg_over=None):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name in drop or p.suffix == ".jsonl":
            continue
        if p.name == "config.json" and cfg_over:
            c = json.loads(p.read_text())
            c.update(cfg_over)
            (dst / p.name).write_text(json.dumps(c))
        elif p.suffix == ".pt":
            os.symlink(p, dst / p.name)
        else:
            shutil.copyfile(p, dst / p.name)
    return dst


def test_refusals(tmp_path):
    T, M = str(TL.SMOKE_T), str(TL.SMOKE_M)
    nodone = _clone_run(TL.SMOKE_T, tmp_path / "nodone", drop=("done.json",))
    with pytest.raises(ValueError, match="done.json"):
        O.TeacherPair2(str(nodone), M, require_done=True)
    O.TeacherPair2(str(nodone), M, require_done=False)                     # smoke / test override
    with pytest.raises(ValueError, match="arm"):
        O.TeacherPair2(M, T)                                                # swapped arms
    ck1 = _clone_run(TL.SMOKE_T, tmp_path / "ck1", cfg_over={"trainer": None})
    with pytest.raises(ValueError, match="train_ck2"):
        O.TeacherPair2(str(ck1), M)
    if TL.CK1_T.is_dir():                                                    # the real CK1 teacher run
        with pytest.raises(ValueError, match="train_ck2"):
            O.TeacherPair2(str(TL.CK1_T), M)
    # a train_ck2-labelled run whose lon head is not zero (not a CK2 checkpoint)
    from navsim.agents.para_ssr.ck.model import load_ck, save_ck
    bad = _clone_run(TL.SMOKE_T, tmp_path / "lon", drop=("ckpt_last.pt", "ckpt_best.pt", "ckpt_ep0.pt"))
    net, cfg = load_ck(str(TL.SMOKE_T), "last")
    with torch.no_grad():
        net.trunk.lon_head[-1].bias.add_(0.1)
    save_ck(bad / "ckpt_last.pt", net, cfg)
    with pytest.raises(ValueError, match="lon_head"):
        O.TeacherPair2(str(bad), M)
    with pytest.raises(ValueError, match="both"):
        O.TeacherPair2(T, "")


def test_ep_target_runtime_guard(tmp_path):
    """report 48 S56-7 / task 5a: the student runtime (TeacherPair2, built by CKE2E2.teachers) refuses teacher runs
    whose config.json ep_target (missing = 'official') differs from the student's ck_e2e2.ep_target -- not only the
    launcher (launch_util2.teacher_check).  ck_e2e2.teacher_ep_check (default true) can switch it off for the
    official-EP smoke fixtures only."""
    T, M = str(TL.SMOKE_T), str(TL.SMOKE_M)
    assert O.CKE2E2Config.from_any(None).teacher_ep_check is True
    assert O.TeacherPair2(T, M, ep_target="official").ep_target == {"det": "official", "map": "official"}
    with pytest.raises(ValueError, match="ep_target"):
        O.TeacherPair2(T, M, ep_target="decoupled")                        # smoke teachers predate --ep-target
    dT = _clone_run(TL.SMOKE_T, tmp_path / "depT", cfg_over={"ep_target": "decoupled"})
    dM = _clone_run(TL.SMOKE_M, tmp_path / "depM", cfg_over={"ep_target": "decoupled"})
    assert O.TeacherPair2(str(dT), str(dM), ep_target="decoupled").ep_target == {"det": "decoupled", "map": "decoupled"}
    with pytest.raises(ValueError, match="map .*ep_target 'official'"):
        O.TeacherPair2(str(dT), M, ep_target="decoupled")                  # one official teacher is enough to refuse
    with pytest.raises(ValueError, match="det .*ep_target 'decoupled'"):
        O.TeacherPair2(str(dT), M, ep_target="official")
    O.TeacherPair2(str(dT), M)                                              # no ep_target given: not checked
    # through CKE2E2.teachers(): the student's ep_target, check on unless teacher_ep_check false
    with pytest.raises(ValueError, match="ep_target"):
        O.CKE2E2(TL.cfg2(tmp_path / "a", teacher_ep_check=True, ep_target="decoupled")).teachers("cpu")  # vs official
    O.CKE2E2(TL.cfg2(tmp_path / "b", teacher_ep_check=True)).teachers("cpu")  # default 'official' (2026-10-09)
    O.CKE2E2(TL.cfg2(tmp_path / "b2", teacher_ep_check=True, ep_target="official")).teachers("cpu")
    with pytest.raises(ValueError, match="ep_target"):                     # decoupled teachers vs default official
        O.CKE2E2(TL.cfg2(tmp_path / "c0", teacher_ep_check=True, teacher_det_run=str(dT),
                         teacher_map_run=str(dM))).teachers("cpu")
    tp = O.CKE2E2(TL.cfg2(tmp_path / "c", teacher_ep_check=True, teacher_det_run=str(dT),
                          teacher_map_run=str(dM), ep_target="decoupled")).teachers("cpu")
    assert tp.ep_target == {"det": "decoupled", "map": "decoupled"}
    O.CKE2E2(TL.cfg2(tmp_path / "d")).teachers("cpu")                     # fixtures: check off (TL.cfg2)


def test_run_outputs_mean_target_and_key_source(pair):
    f, t = TL.fresh(2)
    cand = _cands(t)
    B, K = cand.shape[:2]
    cok = torch.ones(B, K, dtype=torch.bool)
    cok[1, 3] = False
    out = pair.run(t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], cand, f["status_feature"], cand_ok=cok)
    assert out["kd_score_prob"].shape == (B, K, 5) and out["kd_e_lat"].shape == (B, K, 6)
    assert out["kd_ok"].shape == (B, K) and out["kd_ok"].dtype == torch.bool
    assert not bool(out["kd_ok"][1, 3]) and bool(out["kd_ok"].sum() == B * K - 1)
    # mean lateral target (user decision): one target = 0.5 (e_DET + e_MAP)
    torch.testing.assert_close(out["kd_e_lat"], 0.5 * (out["e_det"] + out["e_map"]), rtol=0, atol=0)
    assert float((out["e_det"] - out["e_map"]).abs().max()) > 0          # the two teachers differ
    # per-key score source
    pd_, pm = out["p_det"], out["p_map"]
    src = {"nc": pd_[..., 0], "dac": pm[..., 1], "ep": 0.5 * (pd_[..., 2] + pm[..., 2]), "ttc": pd_[..., 3],
           "comfort": 0.5 * (pd_[..., 4] + pm[..., 4])}
    for i, k in enumerate(CK_KEYS):
        torch.testing.assert_close(out["kd_score_prob"][..., i], src[k], rtol=0, atol=1e-7)


def test_run_equals_direct_forward_with_zero_lon(pair):
    """Each teacher's decoded e_lat = correct(cand, z = 0, w_lat, v0, 0) of a plain CKNet forward (no lon control)."""
    from navsim.agents.para_ssr.ck.correct import correct
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    f, t = TL.fresh(2)
    cand = _cands(t, 6, seed=3)
    out = pair.run(t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], cand, f["status_feature"])
    v0 = ego_inputs(f["status_feature"].float())[0]
    for net, bev, key, pk in ((pair.det, t["kd_bev_0"], "e_det", "p_det"), (pair.map, t["kd_bev_1"], "e_map", "p_map")):
        with torch.no_grad():
            o = net(bev.float(), cand, f["status_feature"].float(), decode=False)
        assert float(o["z_lon"].abs().max()) == 0.0                         # lon head removed
        e = correct(cand, torch.zeros_like(o["w_lat"]), o["w_lat"], v0, 0.0)["e_lat"][..., 2:]
        torch.testing.assert_close(out[key], e, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(out[pk], torch.sigmoid(o["score_logit"]), rtol=1e-5, atol=1e-6)


def test_no_map_bev_turns_kd_off_for_the_token(pair):
    f, t = TL.fresh(2)
    cand = _cands(t, 5, seed=1)
    ok1 = t["kd_ok_1"].clone()
    ok1[0] = False
    out = pair.run(t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], ok1, cand, f["status_feature"])
    assert not out["kd_ok"][0].any() and out["kd_ok"][1].all()
    ok0 = t["kd_ok_0"].clone()
    ok0[1] = False
    out = pair.run(t["kd_bev_0"], ok0, t["kd_bev_1"], t["kd_ok_1"], cand, f["status_feature"])
    assert out["kd_ok"][0].all() and not out["kd_ok"][1].any()
