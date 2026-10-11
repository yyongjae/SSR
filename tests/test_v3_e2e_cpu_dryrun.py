"""refiner-v3 Step 2: tiny CPU dry run of the REAL e2e training step with the temporal DET teacher (docs/refiner_v3/
STEP2_E2E.md): the v2 ParaSSRAgent + CK2 student through pytorch_lightning on CPU (tools/ck/e2e2/cpu_dryrun.py's
build_agent / fit; 2 fixture tokens = packed navtrain_train rows 0 / 40000, batch 1, accumulate 2), ck_e2e2 =
cpu_dryrun's compressed schedule + the v3 teacher set (teacher_det_run v3T10_a_learned, MAP ck2M10, KD calibration on:
a CPU dry fit for the v3 teacher, the real ck2M10 file), epochs 0 'warmup' and 1 'warmup_record'.  The dataset items
carry the DET teacher's object rows exactly as the e2e target builder emits them (KDObjRows).  NOT a training run;
every output under the pytest tmp dir:
  cd /workspace/yongjae/SSR-v3 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -B -m pytest -q -p no:cacheprovider tests/test_v3_e2e_cpu_dryrun.py \
    --basetemp=/tmp/v3e2e_pytest/dryrun
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

D = Path("/home/external-user/ssd/yongjae_refiner/ck")
V3 = D / "v3/train/v3T10_a_learned"
MAIN_M = D / "ck2/train/ck2M10"
CAL_M = D / "ck2/kd_calib/ck2M10__last.json"
FIX2 = D / "ck2/e2e_smoke_cpu/fixtures/real_b2_ck2.pt"

pytestmark = pytest.mark.skipif(
    not ((V3 / "done.json").is_file() and (MAIN_M / "done.json").is_file() and CAL_M.is_file() and FIX2.is_file()),
    reason="v3 teacher / ck2M10 / fixture absent")


def test_cpu_dryrun_e2e_with_temporal_det_teacher(tmp_path):
    from navsim.agents.para_ssr.ck import e2e_data2 as D2
    from tools.ck.e2e import launch_util as LU
    from tools.ck.e2e2 import cpu_dryrun as CD
    from tools.ck.e2e2 import fit_kd_calib as F
    cal = tmp_path / "kd_calib" / "v3T10_a_learned__last.json"
    F.main(["--run", str(V3), "--which", "last", "--device", "cpu", "--limit", "16", "--workers", "0", "--batch", "8",
            "--out", str(cal)])
    io = tmp_path / "ck_e2e2"
    ck2 = CD.ck2_dict(io, "main")
    sets = [ln if not ln.startswith("kd_calib.det=") else f"kd_calib.det={cal}"
            for ln in LU.read_lines(str(REPO / "tools/ck/e2e2/v3teacher_arm.set"))]
    ck2 = LU.apply_sets(dict(ck2, kd_calib={"enabled": True, "det": None, "map": None}), sets)
    ck2.update(teacher_map_run=str(MAIN_M), teacher_ep_check=True)
    ck2["kd_calib"]["map"] = str(CAL_M)
    assert ck2["teacher_det_run"] == str(V3) and ck2["kd_calib"]["det"] == str(cal)
    ko = D2.KDObjRows(D2.teacher_temporal(V3), D2.WARMUP_DEFAULT["packed"])

    class ObjFixtureDS(CD.FixtureDS):
        def __getitem__(self, i):
            f, t = super().__getitem__(i)
            t = dict(t)
            t.update(ko.item(int(t["ck_row"])))
            return f, t

    CD.FixtureDS = ObjFixtureDS                       # fit() builds its DataLoader from CD.FixtureDS
    try:
        agent = CD.build_agent(ck2)
        CD.fit(agent, tmp_path, 2)
    finally:
        CD.FixtureDS = ObjFixtureDS.__mro__[1]
    tp = agent._ck_e2e2._teachers
    assert tp is not None and tp.det_temporal and tp.temporal_info["cfg"]["forecaster"] == "learned"
    assert set(tp.calib_info) == {"det", "map"} and tp.calib_info["det"]["path"] == str(cal)
    recs = [json.loads(x) for x in (io / "steps_rank0.jsonl").read_text().splitlines() if x.strip()]
    assert [int(r["ck2/phase"]) for r in recs] == [0, 0, 1, 1]
    for r in recs:
        assert r["ck2/kd_obj_ok_frac"] == 1.0 and r["ck2/kd_ok_frac"] > 0.5
        assert math.isfinite(r["ck2/kd_score"]) and r["ck2/kd_score"] > 0 and math.isfinite(r["ck2/lat_kd"])
        assert math.isfinite(r["ck2/loss"]) and r["skipped_steps"] == 0
        assert all(math.isfinite(v) for k, v in r.items() if k.startswith("ck2/") and isinstance(v, float))
    assert (io / "rec/ep001").is_dir()                                       # warmup_record recorded the student pool
    ck = sorted((tmp_path / "train").rglob("last.ckpt"))
    assert ck, "no checkpoint"
    sd = torch.load(ck[-1], map_location="cpu", weights_only=False)["state_dict"]
    stu = [k for k in sd if k.startswith("agent.ck_student.")]
    assert stu and not any(k.startswith("agent.ck_student.temporal.") for k in stu)   # student stays CK2
    assert not any(("forecaster" in k or "st_enc" in k) for k in sd)        # frozen teachers are not in the state
