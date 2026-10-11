"""refiner-v3 Step 2 launcher: the CK2 e2e with the temporal DET teacher (tools/ck/e2e2/v3teacher_arm.set,
docs/refiner_v3/STEP2_E2E.md).  CPU, nothing launched (E2E_DRY_RUN=1), every output under the pytest tmp dir:
  cd /workspace/yongjae/SSR-v3 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -B -m pytest -q -p no:cacheprovider tests/test_v3_e2e_launch.py \
    --basetemp=/tmp/v3e2e_pytest/launch
  * launch_util2.teacher_check records det.temporal (cfg, branch / forecaster weight sha16 = what TeacherPair2 loads,
    fc_init / ca_scales shas, the teacher's training provenance); the MAP record and CK2 records carry no temporal key;
  * kd_objects_check opens the e2e object pack and refuses a pack that is not the teacher's training pack; privileged
    teachers are refused;
  * build_overrides2 with the arm set differs from the ck2e2e_main configuration ONLY in teacher_det_run and
    kd_calib.det, and a train_e2e2.sh dry run reproduces ck2e2e_main's launch/overrides.txt line for line except those
    two keys (run name / run dir normalised);
  * smoke-check2's temporal gate (temporal_smoke_checks) passes on those records only with ck2/kd_obj_ok_frac logged
    and is absent for a CK2 run.
The KD calibration file of the temporal teacher is a CPU dry fit (fit_kd_calib.py --device cpu --limit 16) in tmp.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.ck.e2e import launch_util as LU  # noqa: E402
from tools.ck.e2e2 import launch_util2 as L2  # noqa: E402

D = Path("/home/external-user/ssd/yongjae_refiner/ck")
V3 = D / "v3/train/v3T10_a_learned"
MAIN_T, MAIN_M = D / "ck2/train/ck2T10", D / "ck2/train/ck2M10"
MAIN_RUN = D / "ck2/e2e/ck2e2e_main"
SET_FILE = REPO / "tools/ck/e2e2/v3teacher_arm.set"
SH = REPO / "tools/ck/e2e2/train_e2e2.sh"
CAL_REAL_DET = "/home/external-user/ssd/yongjae_refiner/ck/ck2/kd_calib/v3T10_a_learned__last.json"

pytestmark = pytest.mark.skipif(
    not ((V3 / "done.json").is_file() and (MAIN_T / "done.json").is_file() and (MAIN_M / "done.json").is_file()
         and (D / "ck2/kd_calib/ck2M10__last.json").is_file()),
    reason="v3 teacher / ck2e2e_main teachers / calibration absent")


@pytest.fixture(scope="module")
def cal(tmp_path_factory):
    """CPU dry calibration file of v3T10_a_learned (16 tokens), valid for its ckpt sha16 / ep_target / arm."""
    from tools.ck.e2e2 import fit_kd_calib as F
    out = tmp_path_factory.mktemp("kd_calib") / "v3T10_a_learned__last.json"
    F.main(["--run", str(V3), "--which", "last", "--device", "cpu", "--limit", "16", "--workers", "0", "--batch", "8",
            "--out", str(out)])
    return str(out)


def _arm_sets(cal_path: str):
    """v3teacher_arm.set lines with kd_calib.det redirected to the test calibration file."""
    lines = LU.read_lines(str(SET_FILE))
    assert [ln.split("=", 1)[0] for ln in lines] == ["teacher_det_run", "kd_calib.det"]
    return [ln if not ln.startswith("kd_calib.det=") else f"kd_calib.det={cal_path}" for ln in lines]


def _clone_run(src: Path, dst: Path, cfg_edit=None):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.suffix == ".jsonl" or p.name.startswith("eval_") or (p.suffix == ".pt" and p.name != "ckpt_last.pt"):
            continue
        if p.name == "config.json" and cfg_edit:
            c = json.loads(p.read_text())
            cfg_edit(c)
            (dst / p.name).write_text(json.dumps(c))
        elif p.suffix == ".pt":
            os.symlink(p, dst / p.name)
        else:
            shutil.copyfile(p, dst / p.name)
    return dst


# ----------------------------------------------------------------------------------------------- teacher records
def test_teacher_check_records_temporal_teacher(cal):
    from navsim.agents.para_ssr.ck import online2 as O
    rec = L2.teacher_check(str(V3), str(MAIN_M), "last", "strict", True, ep_target="official",
                           kd_calib={"enabled": True, "det": cal, "map": str(D / "ck2/kd_calib/ck2M10__last.json")})
    t = rec["det"]["temporal"]
    assert t["cfg"]["on"] and t["cfg"]["forecaster"] == "learned" and not t["privileged"]
    assert "temporal" not in rec["map"]
    pair = O.TeacherPair2(str(V3), str(MAIN_M), "last", "cpu", amp=False)
    ti = pair.temporal_info
    assert t["branch_state_sha16"] == ti["branch_state_sha16"]                      # the weights the KD uses
    assert t["forecaster_state_sha16"] == ti["forecaster_state_sha16"]
    assert t["fc_init"]["sha256"][:16] == ti["fc_init"]["sha16"]
    assert t["fc_init"]["sha256"] == t["train_provenance"]["fc_init_sha256"]
    assert t["forecaster_equals_fc_init"] is True                                  # frozen pre-trained forecaster
    assert t["ca_scales"]["sha256"] == t["train_provenance"]["ca_scales_sha256"] and t["warnings"] == []
    assert t["train_provenance"]["packs"]["navtrain_train"]
    assert rec["kd_calib"]["det"]["ckpt_sha16"] == rec["det"]["ckpt_sha16"]
    # CK2 teachers: records without any temporal key (unchanged)
    rec0 = L2.teacher_check(str(MAIN_T), str(MAIN_M), "last", "strict", False, ep_target="official")
    assert "temporal" not in rec0["det"] and "temporal" not in rec0["map"]


def test_overrides_differ_from_main_only_in_teacher_keys(tmp_path, cal):
    ov0, ck0, te0 = L2.build_overrides2(str(tmp_path / "ck2e2e_v3T"), ["0", "1"], wandb="online",
                                        teacher_deep=False)
    ov1, ck1, te1 = L2.build_overrides2(str(tmp_path / "ck2e2e_v3T"), ["0", "1"], ck_sets=_arm_sets(cal),
                                        wandb="online", teacher_deep=False)
    d = [(a, b) for a, b in zip(ov0, ov1) if a != b]
    assert len(ov0) == len(ov1) and [a.split("=", 1)[0] for a, _ in d] == [
        "++agent.config.ck_e2e2.teacher_det_run", "++agent.config.ck_e2e2.kd_calib.det"]
    assert d[0][1].endswith(f"={V3}") and d[1][1].endswith(f"={cal}")
    assert "kd_objects" not in te0 and "temporal" not in te0["det"]
    ko = te1["kd_objects"]
    assert ko["same_as_teacher_training"] and ko["build_sha256"] == ko["teacher_train_pack_sha256"]
    assert ko["split"] == "navtrain_train" and ko["n"] == 85109 and ko["targets_opened"] is False
    assert te1["det"]["temporal"]["cfg"]["on"]
    # the ck2e2e_main effective config (when present) = ck0 apart from io_dir
    eff = MAIN_RUN / "launch/ck_e2e2_effective.json"
    if eff.is_file():
        e = {k: v for k, v in json.loads(eff.read_text()).items() if not k.startswith("_")}
        e.pop("io_dir")
        assert e == {k: v for k, v in ck0.items() if k != "io_dir"}


def test_kd_objects_refusals(tmp_path):
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config

    def bad_pack(c):
        c["provenance"]["data"]["packs"]["navtrain_train"]["build_sha256"] = "0" * 64

    clone = _clone_run(V3, tmp_path / "v3T_otherpack", bad_pack)
    sets = [f"teacher_det_run={clone}", "kd_calib.enabled=false"]
    with pytest.raises(L2.LaunchError, match="pack rebuilt"):
        L2.build_overrides2(str(tmp_path / "ck2e2e_x"), ["0", "1"], ck_sets=sets, wandb="disable",
                            teacher_deep=False)

    def no_prov(c):
        c.pop("provenance", None)

    clone2 = _clone_run(V3, tmp_path / "v3T_noprov", no_prov)
    c2 = CKE2E2Config.from_any({"teacher_det_run": str(clone2), "warmup": {}})
    with pytest.raises(L2.LaunchError, match="provenance"):
        L2.kd_objects_check(c2, {})

    def priv(c):
        t = c["temporal"]
        t.update(forecaster="oracle_gt", n_modes=1, fc_init="", fc_train="frozen")
        t.pop("privileged", None)

    clone3 = _clone_run(V3, tmp_path / "v3T_PRIV", priv)
    with pytest.raises(L2.LaunchError, match="PRIVILEGED"):
        L2.teacher_check(str(clone3), str(MAIN_M), "last", "strict", False, ep_target="official")
    with pytest.raises(L2.LaunchError, match="(MAP|arm)"):
        L2.teacher_check(str(MAIN_T), str(V3), "last", "strict", False, ep_target="official")


# ----------------------------------------------------------------------------------------------- shell dry run
def test_train_e2e2_dry_run_matches_main_overrides(tmp_path, cal):
    sets = tmp_path / "v3teacher_tmpcal.set"
    sets.write_text("\n".join(_arm_sets(cal)) + "\n")
    root = tmp_path / "root"
    env = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
               E2E_WANDB="online", E2E_ROOT=str(root), E2E_DRY_RUN="1", CK2_E2E_APPROVED="1", RUN="ck2e2e_v3T",
               E2E_CK_SET=str(sets))
    for k in ("ARM", "E2E_SMOKE", "E2E_EXTRA", "E2E_NO_DEEP_TEACHER_CHECK"):
        env.pop(k, None)
    r = subprocess.run(["bash", str(SH), "start", "--gpus", "0,1"], env=env, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "dry run: nothing launched (arm main" in r.stdout
    R = root / "ck2e2e_v3T"
    assert not (R / "train").exists() and not (R / "launch").exists()
    ov = (R / "launch_dryrun/overrides.txt").read_text().splitlines()
    tj = json.loads((R / "launch_dryrun/teachers.json").read_text())
    assert tj["det"]["temporal"]["cfg"]["on"] and tj["kd_objects"]["same_as_teacher_training"]
    assert tj["kd_calib"]["det"]["path"] == cal and tj["det"]["lon_head"].endswith("max |w| 0")
    main_ov = MAIN_RUN / "launch/overrides.txt"
    if not main_ov.is_file():
        pytest.skip("ck2e2e_main launch/overrides.txt absent")

    def norm(lines, run_dir, run):
        return [ln.replace(str(run_dir), "<R>").replace(run, "<RUN>") for ln in lines]

    a = norm(main_ov.read_text().splitlines(), MAIN_RUN, "ck2e2e_main")
    b = norm(ov, R, "ck2e2e_v3T")
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert [x.split("=", 1)[0] for x, _ in diff] == ["++agent.config.ck_e2e2.teacher_det_run",
                                                     "++agent.config.ck_e2e2.kd_calib.det"], diff
    assert diff[0][0].endswith("/ck2/train/ck2T10") and diff[0][1].endswith(f"={V3}")
    assert diff[1][0].endswith("/ck2/kd_calib/ck2T10__last.json") and diff[1][1].endswith(f"={cal}")
    # the real arm set points kd_calib.det at the file the GPU fit writes (fit_kd_calib default path)
    from navsim.agents.para_ssr.ck.kd_calib import calib_path
    assert str(calib_path(str(V3), "last")) == CAL_REAL_DET
    assert f"kd_calib.det={CAL_REAL_DET}" in LU.read_lines(str(SET_FILE))
    _check_temporal_smoke_gate(tmp_path, R / "launch_dryrun")


def _check_temporal_smoke_gate(tmp_path, launch_dir):
    """review finding 2: smoke-check2's temporal gate (launch_util2.temporal_smoke_checks) on the dry run's real
    launch records: pass only with det.temporal + kd_objects recorded AND ck2/kd_obj_ok_frac logged >= 0.999; a CK2
    run (ck2e2e_main's real launch dir) gets no new check (its smoke result is unchanged)."""
    if (MAIN_RUN / "launch/ck_e2e2_effective.json").is_file():
        assert L2.temporal_smoke_checks(MAIN_RUN, [{"ck2/kd_ok_frac": 1.0}]) == {}
    R = tmp_path / "smoke_like"
    shutil.copytree(launch_dir, R / "launch")
    good = [{"ck2/kd_obj_ok_frac": 1.0}, {"ck2/kd_obj_ok_frac": 1.0}]
    c = L2.temporal_smoke_checks(R, good)
    info = c.pop("_info")
    assert c == {"temporal_teacher_recorded": True, "kd_objects_same_as_teacher_training": True,
                 "kd_obj_ok_frac_logged_ge_0999": True}, (c, info)
    assert info["kd_obj_ok_frac_n"] == 2 and info["teacher_det_run"] == str(V3)
    # the temporal KD path never logged (e.g. teacher silently non-temporal) / KD objects mostly off -> gate fails
    assert L2.temporal_smoke_checks(R, [{"ck2/kd_ok_frac": 1.0}])["kd_obj_ok_frac_logged_ge_0999"] is False
    assert L2.temporal_smoke_checks(R, [{"ck2/kd_obj_ok_frac": 0.5}])["kd_obj_ok_frac_logged_ge_0999"] is False
    # teachers.json without the temporal / kd_objects records -> gate fails
    tj = json.loads((R / "launch/teachers.json").read_text())
    tj.pop("kd_objects")
    tj["det"].pop("temporal")
    (R / "launch/teachers.json").write_text(json.dumps(tj))
    c2 = L2.temporal_smoke_checks(R, good)
    assert c2["temporal_teacher_recorded"] is False and c2["kd_objects_same_as_teacher_training"] is False
