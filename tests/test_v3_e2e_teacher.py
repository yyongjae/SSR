"""refiner-v3 Step 2: the CK2 e2e with the temporal DET teacher v3T10_a_learned (docs/refiner_v3/STEP2_E2E.md).  CPU:
  cd /workspace/yongjae/SSR-v3 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -B -m pytest -q -p no:cacheprovider tests/test_v3_e2e_teacher.py \
    --basetemp=/tmp/v3e2e_pytest/teacher
(i)   non-temporal teachers: TeacherPair2 outputs, the e2e target-builder batches and a warm-up CKE2E2.loss step are
      BIT-identical to the HEAD code (git archive HEAD in tmp, tests/v3e2e_probe.py in two fresh processes).
(ii)  the temporal DET teacher in TeacherPair2 = the trainer forward (train_ck2.forward) and the infer forward
      (CKNet.forward with the 80 variants as extra, infer_val_v3) for the same tokens / candidates; vs the trainer's own
      saved GPU-fp16 train-eval logits within fp16 noise.
(iii) kd_obj_* of the e2e target builder = ObjPack.item (and the trainer's eval ObjectRows) for the token's row; missing
      token -> zeros + kd_obj_ok_0 False; picklable for DataLoader workers; a pack changed / removed after the start
      or a row read error RAISES (per-process fingerprint re-check; review finding 1).
(iv)  causality: only OBJ_INPUT_KEYS reach TemporalBranch.scene, no fc_* / priv_* key is ever emitted or opened, valid
      t0 rows have age 0 and past rows age > 0, a non-causal row is refused; obj_ok False -> KD off for that token;
      privileged teachers refused (TeacherPair2, validate, KDObjRows, fit_kd_calib).
(v)   one e2e step with the v3 teacher set (v3teacher_arm.set on ck_e2e2.yaml; kd_calib.det = a CPU calibration fit of
      the temporal teacher with fit_kd_calib.py --device cpu --limit 16) runs with finite KD; the set file changes
      exactly teacher_det_run and kd_calib.det.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck.temporal.config import COL, OBJ_INPUT_KEYS  # noqa: E402

D = Path("/home/external-user/ssd/yongjae_refiner/ck")
V3 = D / "v3/train/v3T10_a_learned"
MAIN_T, MAIN_M = D / "ck2/train/ck2T10", D / "ck2/train/ck2M10"
CAL_M = D / "ck2/kd_calib/ck2M10__last.json"
PACKED = D2.WARMUP_DEFAULT["packed"]
SET_FILE = REPO / "tools/ck/e2e2/v3teacher_arm.set"
PY = "/venv/ssr/bin/python"


def _have_v3() -> bool:
    return ((V3 / "done.json").is_file() and (V3 / "ckpt_last.pt").is_file() and (MAIN_M / "done.json").is_file()
            and TL.have_inputs()
            and (D / "v3/objpack_k96/bevfusion/navtrain_train/build.json").is_file())


need_v3 = pytest.mark.skipif(not _have_v3(), reason="v3 teacher / object pack / CK2 fixture absent")


# ----------------------------------------------------------------------------------------------- helpers
@pytest.fixture(scope="module")
def pair_v3():
    return O.TeacherPair2(str(V3), str(MAIN_M), "last", "cpu", amp=False, require_done=True, ep_target="official")


@pytest.fixture(scope="module")
def kdobjs():
    return D2.KDObjRows(D2.teacher_temporal(V3), PACKED)


def _with_objs(t, ko):
    t = dict(t)
    its = [ko.item(int(r)) for r in t["ck_row"]]
    for k in its[0]:
        t[k] = torch.stack([it[k] for it in its])
    return t


def _cands(B, n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.from_numpy(np.load(TL.ANCHORS)).float()
    return a[torch.randint(0, 256, (B, n), generator=g)] + 0.05 * torch.randn(B, n, 8, 3, generator=g)


def _clone_run(src: Path, dst: Path, cfg_over=None):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.suffix == ".jsonl" or p.name.startswith("eval_") or (p.suffix == ".pt" and p.name != "ckpt_last.pt"):
            continue
        if p.name == "config.json" and cfg_over:
            c = json.loads(p.read_text())
            for k, v in cfg_over.items():
                c[k] = v
            (dst / p.name).write_text(json.dumps(c))
        elif p.suffix == ".pt":
            os.symlink(p, dst / p.name)
        else:
            shutil.copyfile(p, dst / p.name)
    return dst


def _priv_temporal():
    t = dict(json.loads((V3 / "config.json").read_text())["temporal"])
    t.update(forecaster="oracle_gt", n_modes=1, fc_init="", fc_train="frozen")
    t.pop("privileged", None)
    return t


# ----------------------------------------------------------------------------------------------- (i) bit-identity
@pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")
def test_non_temporal_path_bit_identical_to_head(tmp_path):
    head = tmp_path / "head"
    head.mkdir()
    arc = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD", "navsim", "tools"], capture_output=True,
                         timeout=300)
    if arc.returncode != 0:
        pytest.skip(f"git archive HEAD failed: {arc.stderr[-300:]}")
    subprocess.run(["tar", "-x", "-C", str(head)], input=arc.stdout, check=True, timeout=300)
    # HEAD has no temporal KD (the e2e files changed only in the working tree)
    assert "KDObjRows" not in (head / "navsim/agents/para_ssr/ck/e2e_data2.py").read_text()
    assert "KDObjRows" in (REPO / "navsim/agents/para_ssr/ck/e2e_data2.py").read_text()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    outs = {}
    for name, root in (("head", head), ("cur", REPO)):
        io = tmp_path / f"io_{name}"
        io.mkdir()
        r = subprocess.run([PY, "-B", str(REPO / "tests/v3e2e_probe.py"), str(root), str(tmp_path / f"{name}.pt"),
                            str(io)], env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=1800)
        assert r.returncode == 0, r.stderr[-3000:]
        outs[name] = torch.load(tmp_path / f"{name}.pt", map_location="cpu", weights_only=False)
    a, b = outs["head"], outs["cur"]
    assert set(a) == set(b)
    groups = {k.split("/")[0] for k in a}
    assert groups == {"pair", "builder", "loss"}
    assert any(k.startswith("pair/main/") for k in a) or not CAL_M.is_file()     # the ck2e2e_main teachers + calib
    assert not any("kd_obj" in k for k in b)                                      # CK2 teacher: no object keys
    for k in a:
        x, y = a[k], b[k]
        assert x.dtype == y.dtype and x.shape == y.shape, k
        if x.is_floating_point():
            assert torch.equal(torch.isnan(x), torch.isnan(y)) and torch.equal(x.nan_to_num(), y.nan_to_num()), k
        else:
            assert torch.equal(x, y), k


# ----------------------------------------------------------------------------------------------- (ii) forward
@need_v3
def test_temporal_teacher_equals_trainer_and_infer_forward(pair_v3, kdobjs):
    from navsim.agents.para_ssr.ck.correct import correct
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    from tools.ck import train_ck2 as TR
    assert pair_v3.det_temporal and pair_v3.map.temporal is None
    ti = pair_v3.temporal_info
    assert ti["cfg"]["on"] and not ti["cfg"]["privileged"] and ti["cfg"]["forecaster"] == "learned"
    assert ti["fc_init"]["sha16"] and ti["forecaster_state_sha16"] and ti["branch_state_sha16"]
    f, t = TL.fresh(2)
    t = _with_objs(t, kdobjs)
    objs, ok = O.kd_objs_of(t)
    assert tuple(objs) == OBJ_INPUT_KEYS and bool(ok.all())
    cand = _cands(2, 16, seed=1)
    status = f["status_feature"].float()
    R = pair_v3.run(t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], cand, status, objs_det=objs,
                    obj_ok_det=ok)
    net, tc = pair_v3.det, pair_v3.det.temporal_cfg
    batch = {"bev": t["kd_bev_0"], "cand": cand, "status": status, **objs}
    ref = TR.forward(net, batch, False, torch.device("cpu"), False, tc)              # trainer forward (fp32 on CPU)
    assert torch.equal(torch.sigmoid(ref["score_logit"].float()), R["p_det"])
    dec = correct(cand.float(), torch.zeros_like(ref["w_lat"].float()), ref["w_lat"].float(),
                  ego_inputs(status)[0], 0.0)
    torch.testing.assert_close(dec["e_lat"][..., 2:].float(), R["e_det"], rtol=0, atol=1e-6)
    with torch.no_grad():                                                           # infer_val_v3: + 80 variants
        var = _cands(2, 80, seed=2)
        inf = net(t["kd_bev_0"].float(), cand, status, extra=var, decode=False, objs=objs)
        z, _ = O.teacher_forward(net, t["kd_bev_0"].float(), cand, ego_inputs(status), objs)
    d = (inf["score_logit"].float() - z).abs().reshape(-1)
    assert float(torch.quantile(d, 0.999)) <= 2e-4 and float(d.max()) <= 1e-3      # infer-check tolerance
    # the MAP teacher is untouched by the temporal DET teacher
    pm = O.TeacherPair2(str(MAIN_T), str(MAIN_M), "last", "cpu", amp=False, require_done=True, ep_target="official")
    R2 = pm.run(t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], cand, status)
    assert torch.equal(R2["p_map"], R["p_map"]) and torch.equal(R2["e_map"], R["e_map"])
    assert not torch.equal(R2["p_det"], R["p_det"])                                 # another DET teacher


@need_v3
def test_temporal_teacher_matches_saved_trainer_eval_logits(pair_v3):
    """CPU fp32 TeacherPair2 forward vs the logits train_ck2 saved at its last navtrain_train train-eval (GPU, fp16
    autocast, same ckpt_last weights): equal within fp16 noise [measured max ~0.024 on 16 rows]."""
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    from tools.ck.data.ck_dataset import CKDataset, collate_ck
    f = V3 / "eval_r34_navtrain_train_train_eval_last.npz"
    if not f.is_file():
        pytest.skip("no saved train-eval logits")
    z = np.load(f)
    rows = z["rows"].astype(np.int64)
    pick = np.sort(rows[np.linspace(0, len(rows) - 1, 6).astype(int)])
    pos = {int(r): i for i, r in enumerate(rows)}
    tc = pair_v3.det.temporal_cfg
    ds = CKDataset("navtrain_train", bev="T", k=16, labels=None, gt=False, rows=pick,
                   objects={"cfg": tc.to_dict(), "aug": False, "targets": False})
    b = collate_ck([ds[i] for i in range(len(ds))])
    objs = {k: b[k] for k in OBJ_INPUT_KEYS}
    R = pair_v3.run(b["bev"], b["bev_ok"], b["bev"], b["bev_ok"], b["cand"].float(), b["status"].float(),
                    objs_det=objs, obj_ok_det=b["obj_ok"])
    with torch.no_grad():
        zz, _ = O.teacher_forward(pair_v3.det, b["bev"].float(), b["cand"].float(), ego_inputs(b["status"].float()),
                                  objs)
    assert torch.equal(torch.sigmoid(zz), R["p_det"])
    ref = np.stack([z["logit"][pos[int(r)]] for r in b["rows"].numpy()])
    d = np.abs(zz.numpy() - ref)
    assert np.isfinite(d).all() and float(np.median(d)) < 5e-3 and float(d.max()) < 0.1, (np.median(d), d.max())


# ----------------------------------------------------------------------------------------------- (iii) rows
@need_v3
def test_builder_obj_rows_equal_objpack_item(tmp_path):
    from tools.ck.data.ck2_dataset import ObjectRows
    from tools.ck.data.obj_pack import ObjPack
    b = D2.CK2E2ETargetBuilder(ck_cfg={"teacher_det_run": str(V3), "teacher_map_run": str(MAIN_M)})
    assert b.kd_objs is not None and not b.kd_objs.pack.targets
    pk = ObjPack("navtrain_train", source="bevfusion", root=str(D / "v3/objpack_k96"), targets=False)
    d = TL.batch()
    tc = D2.teacher_temporal(V3)
    orows = ObjectRows("navtrain_train", {"cfg": tc.to_dict(), "aug": False, "targets": False}, pk.tokens,
                       np.asarray(d["rows"], np.int64))
    b2 = pickle.loads(pickle.dumps(b))                                               # DataLoader worker copy
    b0 = D2.CK2E2ETargetBuilder(ck_cfg={"teacher_det_run": str(MAIN_T), "teacher_map_run": str(MAIN_M)})
    assert b0.kd_objs is None                                                        # CK2 DET teacher: no pack
    for tok, row in zip(d["tokens"], d["rows"]):
        o = b.compute_for_token(tok)
        o2 = b2.compute_for_token(tok)
        assert int(o["ck_row"]) == int(row)
        it = pk.item(int(row))
        er = orows.item(int(row), tok, 0, 0)                                        # trainer eval path (no aug)
        for k, src in D2.KD_OBJ_INPUT.items():
            assert np.array_equal(o[k].numpy(), it[src]) and np.array_equal(o[k].numpy(), er[src]), k
            assert torch.equal(o[k], o2[k])
            assert o[k].dtype == {"obj_in": torch.float32}.get(src, torch.bool)
        assert bool(o[D2.KD_OBJ_OK]) == bool(it["obj_ok"]) is True
        # every other key = the CK2-teacher builder's item (both DET teachers read the BEVFusion t0 cache)
        o0 = b0.compute_for_token(tok)
        assert set(o) - set(o0) == set(D2.KD_OBJ_KEYS) | {D2.KD_OBJ_OK}
        for k, v in o0.items():
            assert torch.equal(o[k], v), k
    miss = b.compute_for_token("ffffffffffffffff")
    assert int(miss["ck_row"]) == -1 and not bool(miss[D2.KD_OBJ_OK])
    assert all(float(miss[k].float().abs().sum()) == 0 for k in D2.KD_OBJ_KEYS)
    assert tuple(miss["kd_obj_in_0"].shape) == (3, 96, 17)
    assert not any(k.startswith("kd_obj") for k in b0.compute_for_token(d["tokens"][0]))
    # a temporal MAP teacher is refused
    with pytest.raises(ValueError, match="MAP teacher"):
        D2.CK2E2ETargetBuilder(ck_cfg={"teacher_det_run": str(MAIN_T), "teacher_map_run": str(V3)})


def _probe_item(ko_pickle: bytes, row: int) -> str:
    """child process: unpickle a KDObjRows (DataLoader worker copy) and read one row -> 'ok:<obj_ok>' or 'err:<msg>'."""
    code = ("import pickle, sys; ko = pickle.loads(sys.stdin.buffer.read())\n"
            "try:\n    it = ko.item(%d); print('ok:' + str(bool(it['kd_obj_ok_0'])))\n"
            "except RuntimeError as e:\n    print('err:' + str(e))\n" % int(row))
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    r = subprocess.run([PY, "-B", "-c", code], input=ko_pickle, capture_output=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr.decode()[-2000:]
    return r.stdout.decode().strip().splitlines()[-1]


@need_v3
def test_kd_obj_rows_fail_loud_on_pack_change_and_read_error(tmp_path, monkeypatch):
    """review finding 1 (unattended run): a pack changed / removed after the e2e start, or a read error of a packed row,
    RAISES in the reading process (never a silent kd_obj_ok False); r = -1 (token without a packed row) stays zeros."""
    import dataclasses
    tc = D2.teacher_temporal(V3)
    src = D / "v3/objpack_k96/bevfusion/navtrain_train"
    root = tmp_path / "packroot"
    dst = root / tc.source / "navtrain_train"
    dst.mkdir(parents=True)
    for f in src.iterdir():                                      # symlinked copy; build.json a real copy (editable)
        if f.name == "build.json":
            shutil.copyfile(f, dst / f.name)
        elif f.is_file():
            os.symlink(f, dst / f.name)
    ko = D2.KDObjRows(dataclasses.replace(tc, pack_root=str(root)), PACKED)
    assert ko.fingerprint["build_sha256"] and "obj.npy" in ko.fingerprint and "fc_xyh.npy" not in ko.fingerprint
    ref = D2.KDObjRows(tc, PACKED)
    a, b = ko.item(5), ref.item(5)                                # same rows as the real pack
    assert all(torch.equal(a[k], b[k]) for k in a) and bool(a[D2.KD_OBJ_OK])
    blob = pickle.dumps(ko)
    assert _probe_item(blob, 5) == "ok:True"                      # a fresh worker process re-checks and reads
    # build.json rewritten after the start -> the next process to read raises (main: once its pid check re-runs)
    bj = dst / "build.json"
    bj.write_text(bj.read_text().replace('"complete"', '"complete" ', 1))
    out = _probe_item(blob, 5)
    assert out.startswith("err:") and "changed since the e2e start" in out and "build_sha256" in out, out
    ko._pid = None
    with pytest.raises(RuntimeError, match="changed since the e2e start"):
        ko.item(5)
    # a removed array -> unreadable -> raises
    shutil.copyfile(src / "build.json", bj)
    ko._pid = None
    ko.item(5)                                                    # restored: fine again
    (dst / "obj_mask.npy").unlink()
    ko._pid = None
    with pytest.raises(RuntimeError, match="unreadable since the e2e start"):
        ko.item(5)
    # r = -1 / out of range: zeros + kd_obj_ok False without touching the pack
    assert not bool(ko.item(-1)[D2.KD_OBJ_OK]) and not bool(ko.item(10 ** 9)[D2.KD_OBJ_OK])
    # a read error of a packed row is re-raised (was: swallowed -> silent KD off)
    def boom(r):
        raise OSError("simulated read error")
    monkeypatch.setattr(ref.pack, "item", boom)
    with pytest.raises(OSError, match="simulated read error"):
        ref.item(5)


# ----------------------------------------------------------------------------------------------- (iv) causality
@need_v3
def test_causality_only_object_inputs_reach_the_teacher(pair_v3, kdobjs, monkeypatch):
    f, t = TL.fresh(2)
    t = _with_objs(t, kdobjs)
    assert not any(k.startswith(("fc_", "priv_")) or "fc_" in k for k in t)        # never a forecast target
    assert all(not n.startswith("fc_") for n in kdobjs.pack._names)               # targets never opened
    objs, ok = O.kd_objs_of(t)
    m = objs["obj_mask"] & objs["frame_valid"][..., None]
    age = objs["obj_in"][..., COL["age"]]
    assert bool(m[:, 0].any()) and bool((age[:, 0][m[:, 0]] == 0).all())
    assert bool((age[:, 1:][m[:, 1:]] > 0).all())                                  # past frames strictly earlier
    assert float(objs["obj_in"][~m].abs().sum()) == 0                               # padding rows are zero
    seen = []
    br = pair_v3.det.temporal
    orig = br.scene

    def spy(o):
        seen.append(tuple(o))
        return orig(o)

    monkeypatch.setattr(br, "scene", spy)
    cand = _cands(2, 8, seed=4)
    status = f["status_feature"].float()
    R = pair_v3.run(t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], cand, status, objs_det=objs,
                    obj_ok_det=ok)
    assert seen == [OBJ_INPUT_KEYS]
    assert bool(R["kd_ok"].all())
    # obj_ok False for token 1 -> DET ok False -> KD off for that token only
    R1 = pair_v3.run(t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], cand, status, objs_det=objs,
                     obj_ok_det=torch.tensor([True, False]))
    assert bool(R1["kd_ok"][0].all()) and not bool(R1["kd_ok"][1].any())
    # a non-causal row (valid past row with age 0) is refused by the branch's input guard
    bad = {k: v.clone() for k, v in objs.items()}
    i = int(torch.nonzero(bad["obj_mask"][0, 1] & bad["frame_valid"][0, 1])[0]) if bool(
        (bad["obj_mask"][0, 1] & bad["frame_valid"][0, 1]).any()) else None
    if i is None:                                                                    # make one valid past row
        bad["frame_valid"][0, 1] = True
        bad["obj_mask"][0, 1, 0] = True
        i = 0
    bad["obj_in"][0, 1, i, COL["age"]] = 0.0
    with pytest.raises(ValueError, match="non-causal"):
        pair_v3.run(t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], cand, status, objs_det=bad,
                    obj_ok_det=ok)
    # wiring errors are loud
    with pytest.raises(ValueError, match="required"):
        pair_v3.run(t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], cand, status)
    with pytest.raises(KeyError, match="kd_obj"):
        O.kd_objs_of({k: v for k, v in t.items() if k != "kd_obj_ok_0"})


@need_v3
def test_privileged_and_map_temporal_teachers_refused(tmp_path):
    from navsim.agents.para_ssr.ck.temporal.config import TemporalCfg
    from tools.ck.e2e2 import fit_kd_calib as F
    priv = _clone_run(V3, tmp_path / "v3T_PRIV", cfg_over={"temporal": _priv_temporal()})
    assert TemporalCfg.from_dict(json.loads((priv / "config.json").read_text())["temporal"]).privileged
    with pytest.raises(PermissionError, match="PRIVILEGED"):
        O.TeacherPair2(str(priv), str(MAIN_M), "last", "cpu", amp=False)
    with pytest.raises(ValueError, match="PRIVILEGED"):
        D2.KDObjRows(D2.teacher_temporal(priv), PACKED)
    with pytest.raises(ValueError, match="PRIVILEGED"):
        O.CKE2E2Config.from_any({"enabled": True, "io_dir": str(tmp_path / "io"), "teacher_det_run": str(priv),
                                 "teacher_map_run": str(MAIN_M)}).validate(None)
    with pytest.raises(SystemExit, match="PRIVILEGED"):
        F.run_info(priv, "last")
    # a temporal run given as the MAP teacher (arm T run in the MAP slot is refused by the arm check already)
    with pytest.raises(ValueError):
        O.TeacherPair2(str(MAIN_T), str(V3), "last", "cpu", amp=False)


# ----------------------------------------------------------------------------------------------- (v) e2e step
@need_v3
def test_e2e_step_with_v3_teacher_set_and_cpu_calibration(tmp_path, kdobjs):
    from tools.ck.e2e import launch_util as LU
    from tools.ck.e2e2 import fit_kd_calib as F
    from tools.ck.e2e2 import launch_util2 as L2
    # the arm set file changes exactly teacher_det_run and kd_calib.det of ck_e2e2.yaml
    base = L2.load_ck2_yaml()
    lines = LU.read_lines(str(SET_FILE))
    arm = LU.apply_sets(base, lines)
    diff = {k for (k, v), (k2, v2) in zip(LU.flatten(base), LU.flatten(arm)) if k == k2 and v != v2}
    assert diff == {"teacher_det_run", "kd_calib.det"} and len(LU.flatten(base)) == len(LU.flatten(arm))
    assert arm["teacher_det_run"] == str(V3) and arm["bev_kd"]["enabled"] is False
    assert arm["kd_calib"]["det"].endswith("/ck2/kd_calib/v3T10_a_learned__last.json")
    # calibration of the temporal teacher (CPU dry fit, tiny; the real fit runs on GPU 0, STEP2_E2E.md)
    cal = tmp_path / "kd_calib" / "v3T10_a_learned__last.json"
    rec = F.main(["--run", str(V3), "--which", "last", "--device", "cpu", "--limit", "16", "--workers", "0",
                  "--batch", "8", "--out", str(cal)])
    assert rec["fitted_keys"] == ["nc", "ep", "ttc"] and rec["inference"]["temporal"]["on"]
    assert rec["inference"]["sanity"]["fwd_score_maxdiff_vs_cknet"] == 0.0
    assert rec["data"]["objects"]["build_sha16"] and rec["data"]["n_tokens_obj_ok"] == rec["data"]["n_tokens"]
    rec2 = F.main(["--run", str(V3), "--which", "last", "--device", "cpu", "--limit", "16", "--workers", "0",
                   "--batch", "8", "--out", str(cal)])
    assert rec2["inference"]["cached"] and rec2["params"] == rec["params"]          # cache key incl. temporal / pack
    # one warm-up e2e step: v3 DET teacher (calibrated by that file) + ck2M10 (its real calibration)
    ck = dict(arm, io_dir=str(tmp_path / "ck_e2e2"), teacher_amp=False, score_prior="none", strict_rows=0.0,
              grad_share_every=1)
    ck["kd_calib"] = {"enabled": True, "det": str(cal), "map": str(CAL_M)}
    c = O.CKE2E2Config.from_any(ck)
    c.validate(None)
    E = O.CKE2E2(c, max_epochs=30)
    student = O.build_student_ck2(c, apply_prior=False)
    f, t = TL.fresh(2)
    t = _with_objs(t, kdobjs)
    pred = TL.fake_predictions(2, seed=5)
    v2_loss = (pred["bev_embed"] ** 2).mean()
    loss, logs = E.loss(student, f, t, pred, v2_loss=v2_loss)
    lv = {k: float(v) for k, v in logs.items()}
    assert np.isfinite(float(loss.detach())) and lv["ck2/loss_nonfinite"] == 0.0
    assert lv["ck2/kd_ok_frac"] > 0.5 and lv["ck2/kd_obj_ok_frac"] == 1.0
    assert np.isfinite(lv["ck2/kd_score"]) and lv["ck2/kd_score"] > 0 and np.isfinite(lv["ck2/lat_kd"])
    assert E._teachers.det_temporal and set(E._teachers.calib_info) == {"det", "map"}
    loss.backward()
    assert TL.all_grads_finite(student) and pred["bev_embed"].grad is not None
    # without object rows the temporal-teacher step refuses (never a silent CK2-path KD)
    t2 = {k: v for k, v in t.items() if not k.startswith("kd_obj")}
    with pytest.raises(KeyError, match="kd_obj"):
        E.loss(student, f, t2, TL.fake_predictions(2, seed=5))
