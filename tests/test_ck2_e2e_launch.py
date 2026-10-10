"""CK2 e2e launcher (SPEC s8 T12): tools/ck/e2e2/launch_util2.py overrides (2-GPU r34 recipe, doubled CK constants,
every ck_e2e2 leaf, BEV-KD arm set file), refusals (GPUs, teachers), Hydra round trip of the ck_e2e2 overrides into
CKE2E2Config, and a train_e2e2.sh dry run (E2E_DRY_RUN=1: prints the overrides, launches nothing).  CPU, no GPU:
  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_launch.py \
    --basetemp=/tmp/claude-0/-workspace/f9196902-3a88-4418-a505-cd713226ed4a/scratchpad/ck2e2e/pytest/launch
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.ck.e2e2 import launch_util2 as L2  # noqa: E402

D = Path("/home/external-user/ssd/yongjae_refiner/ck")
SMOKE_T = D / "ck2/smoke/smoke_ck2T_ddp2"
SMOKE_M = D / "ck2/smoke/smoke_ck2M_ddp2"
REAL_T, REAL_M = D / "ck2/train/ck2T10", D / "ck2/train/ck2M10"              # ck_e2e2.yaml defaults (official EP)
SH = REPO / "tools/ck/e2e2/train_e2e2.sh"
pytestmark = pytest.mark.skipif(not ((SMOKE_T / "done.json").is_file() and (SMOKE_M / "done.json").is_file()),
                                reason="CK2 smoke teachers absent")
# the smoke teachers were trained before --ep-target (config.json without the key = 'official'): the student must match
# and have no KD calibration files (kd_calib guard: tests/test_ck2_e2e_kd_calib.py)
SMOKE_SETS = [f"teacher_det_run={SMOKE_T}", f"teacher_map_run={SMOKE_M}", "ep_target=official",
              "kd_calib.enabled=false"]


def _ov(tmp, sets=(), gpus=("0", "1"), **kw):
    return L2.build_overrides2(str(tmp / "ck2e2e_test"), list(gpus), ck_sets=list(SMOKE_SETS) + list(sets),
                               wandb=kw.pop("wandb", "offline"), **kw)


def _get(ov, key):
    hits = [o.split("=", 1)[1] for o in ov if o.split("=", 1)[0].lstrip("+~") == key]
    assert hits, key
    return hits[-1]


def test_recipe_and_ck_leaves(tmp_path):
    ov, ck, teachers = _ov(tmp_path)
    assert _get(ov, "trainer.params.devices") == "2" and "+trainer.params.devices=2" in ov
    assert _get(ov, "trainer.params.strategy") == "ddp"
    assert _get(ov, "dataloader.params.batch_size") == "4"
    assert _get(ov, "trainer.params.accumulate_grad_batches") == "16"
    assert _get(ov, "agent.config.grad_balance_warmup_iters") == "10600"
    assert _get(ov, "agent.config.grad_balance_interval") == "200"
    assert _get(ov, "agent.config.grad_norm_log_interval") == "200"
    assert _get(ov, "agent.config.max_epochs") == "30" and _get(ov, "trainer.params.max_epochs") == "30"
    assert _get(ov, "agent.config.image_architecture") == "resnet34.tv_in1k" and _get(ov, "agent.lr") == "1e-4"
    assert _get(ov, "agent.config.plan_score_file").endswith("/planning_vb/pdm_score_256")      # v2 keeps WoTE labels
    assert _get(ov, "experiment_name") == "ck2e2e_test" and _get(ov, "wandb.name") == "ck2e2e_test"
    assert _get(ov, "wandb.project") == "para-ssr-v2" and _get(ov, "wandb.group") == "ck2e2e"
    assert _get(ov, "wandb.tags") == "[ck2e2e,main]"
    assert _get(ov, "output_dir") == f"{tmp_path}/ck2e2e_test/train/"
    # doubled per-mb constants and the user decisions
    c = "agent.config.ck_e2e2."
    want = {"enabled": "true", "lat_kd.start_mb": "1000", "label_refresh_every_mb": "1000", "grad_share_every": "100",
            "n_var_step": "32", "lambda_score": "1.0", "lambda_kd_score": "0.5", "onpolicy_from_epoch": "5",
            "label_lag": "1", "variants.ext": "straight", "variants.speeds": "[-1.0,-0.5,0.5]",
            "variants.lats": "[-0.5,0.5]", "variants.combine": "separate", "teacher_amp": "true",
            "teacher_which": "last", "bev_kd.enabled": "false", "infer_select": "null",
            "io_dir": f"{tmp_path}/ck2e2e_test/ck_e2e2", "teacher_det_run": str(SMOKE_T)}
    for k, v in want.items():
        assert _get(ov, c + k) == v, k
    assert ck["io_dir"] == f"{tmp_path}/ck2e2e_test/ck_e2e2"
    assert teachers["det"]["ckpt_sha16"] and teachers["map"]["arm"] == "M" and "lon_head" in teachers["det"]
    # every CKE2E2Config leaf is in the yaml and the yaml equals the dataclass defaults (enabled / io_dir aside)
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
    y = L2.load_ck2_yaml()
    dflt = CKE2E2Config().to_dict()
    assert set(y) == set(dflt)
    y2 = dict(y, enabled=False)
    assert CKE2E2Config.from_any(y2).to_dict() == dflt


def test_hydra_roundtrip_into_config(tmp_path):
    """the '++agent.config.ck_e2e2.*' overrides parsed by Hydra's grammar rebuild exactly the effective config"""
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
    for sets in ((), ("bev_kd.enabled=true",)):
        ov, ck, _ = _ov(tmp_path, sets)
        parsed = OverridesParser.create().parse_overrides([o for o in ov if o.startswith("++agent.config.ck_e2e2.")])
        d: dict = {}
        for p in parsed:
            key = p.key_or_group.split("agent.config.ck_e2e2.", 1)[1].split(".")
            cur = d
            for k in key[:-1]:
                cur = cur.setdefault(k, {})
            cur[key[-1]] = p.value()
        assert CKE2E2Config.from_any(d).to_dict() == CKE2E2Config.from_any(ck).to_dict()


def test_bevkd_arm_set_file(tmp_path):
    sets = L2.LU.read_lines(str(L2.BEVKD_SET))
    ov, ck, _ = _ov(tmp_path, sets)
    c = "agent.config.ck_e2e2.bev_kd."
    for k, v in {"enabled": "true", "teachers": "[det,map]", "distance": "mse", "init": "zero", "ratio": "0.1",
                 "cap": "0.25", "start_mb": "10600", "ratio_det": "null", "ratio_map": "null", "cap_det": "null",
                 "cap_map": "null"}.items():
        assert _get(ov, c + k) == v, k
    assert _get(ov, "wandb.tags") == "[ck2e2e,bevkd]" and L2.arm_of(ck) == "bevkd"
    # everything else identical to the main run (same recipe / seed / data order)
    ov_m, _, _ = _ov(tmp_path)
    diff = sorted(set(ov) ^ set(ov_m))
    assert all(o.startswith(("++agent.config.ck_e2e2.bev_kd.enabled=", "++agent.config.ck_e2e2.bev_kd.teachers=",
                             "wandb.tags=")) for o in diff), diff
    # the BEV-KD guard: both teachers' frozen z-score files (det norm.npz, map norm_map.npz) recorded in teachers.json
    _, _, teachers = _ov(tmp_path, sets)
    bk = teachers["bev_kd"]
    assert set(bk) == {"det", "map"} and bk["det"]["norm"] == str(SMOKE_T / "norm.npz") \
        and bk["map"]["norm"] == str(SMOKE_M / "norm_map.npz") and len(bk["map"]["sha16"]) == 16
    assert "bev_kd" not in _ov(tmp_path)[2]                                     # main run: no arm, no record
    # a MAP teacher run without norm_map.npz: refused before anything starts
    nm = tmp_path / "noNormM"
    nm.mkdir()
    for f in ("config.json", "done.json"):
        shutil.copy(SMOKE_M / f, nm / f)
    os.symlink(SMOKE_M / "ckpt_last.pt", nm / "ckpt_last.pt")
    with pytest.raises(L2.LaunchError, match="norm_map.npz"):
        L2.build_overrides2(str(tmp_path / "ck2e2e_nonorm"), ["2", "3"], wandb="offline", teacher_deep=False,
                            ck_sets=list(SMOKE_SETS) + sets + [f"teacher_map_run={nm}"])
    L2.build_overrides2(str(tmp_path / "ck2e2e_detonly"), ["2", "3"], wandb="offline", teacher_deep=False,
                        ck_sets=list(SMOKE_SETS) + sets + [f"teacher_map_run={nm}", "bev_kd.teachers=[det]"])


def test_smoke_check2_and_status_per_teacher(tmp_path):
    R = tmp_path / "g1_bevkd"
    (R / L2.IO_SUB).mkdir(parents=True)
    (R / "launch").mkdir()
    (R / "launch/ck_e2e2_effective.json").write_text(json.dumps(
        {"bev_kd": {"enabled": True, "teachers": ["det", "map"], "start_mb": 2}}))
    recs = []
    for mb in range(6):
        r = {"epoch": 0, "mb": mb, "ck2/phase": 0.0, "sec_step": 0.5}
        for t, lam, ok in (("det", 0.0 if mb < 3 else 2.0, 1.0), ("map", 0.0 if mb < 3 else 5.0, 0.75)):
            r.update({f"bevkd/{t}/lam": lam, f"bevkd/{t}/ok_frac": ok, f"bevkd/{t}/loss": 1.0})
            if mb % 2 == 0:
                r.update({f"bevkd/{t}/g_kd_unit": 0.01, f"bevkd/{t}/ratio_now": lam * 0.01 / 0.2,
                          f"gnorm/bev_bevkd_{t}": lam * 0.01, "gnorm/bev_v2": 0.2})
        recs.append(r)
    (R / L2.IO_SUB / "steps_rank0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    res = L2.smoke_check2(str(R), gen_epochs=())
    assert res["arm"] == "bevkd" and set(res["bevkd"]) == {"det", "map"}
    m = res["bevkd"]["map"]
    assert m["lam_max"] == 5.0 and m["ok_frac_mean"] == 0.75 and m["ratio_now_last10"][-1] == pytest.approx(0.25)
    assert m["gnorm_bev_over_v2_mean"] == pytest.approx((0.0 + 0.25) / 2)              # mb 2, 4 (>= start_mb 2)
    assert res["checks"]["bevkd_teachers_logged"] and res["checks"]["bevkd_lam_pos_every_teacher"]
    for r in recs:                                                                    # MAP never switched on
        r["bevkd/map/lam"] = 0.0
    (R / L2.IO_SUB / "steps_rank0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    res = L2.smoke_check2(str(R), gen_epochs=())
    assert not res["checks"]["bevkd_lam_pos_every_teacher"] and not res["pass"]
    st = L2.run_status2(str(R))
    assert st["arm"] == "bevkd" and st["steps"]["bevkd_lam"] == {"det": 2.0, "map": 0.0}


def test_refusals(tmp_path):
    for g in (("0",), ("0", "1", "2"), ("4", "5"), ("0", "0")):
        with pytest.raises(L2.LaunchError):
            _ov(tmp_path, gpus=g)
    with pytest.raises(L2.LaunchError, match="unknown key"):
        _ov(tmp_path, ["no_such_key=1"])
    with pytest.raises(L2.LaunchError, match="smoke"):
        _ov(tmp_path, ["teacher_require_done=false"])
    with pytest.raises(L2.LaunchError):                            # 'separate' 6-column table only
        _ov(tmp_path, ["variants.combine=grid"])
    # a teacher without done.json (copy of the smoke run minus done.json): strict refuses, smoke mode accepts
    fake = tmp_path / "fakeT"
    fake.mkdir()
    shutil.copy(SMOKE_T / "config.json", fake / "config.json")
    for f in ("norm.npz",):
        if (SMOKE_T / f).is_file():
            shutil.copy(SMOKE_T / f, fake / f)
    os.symlink(SMOKE_T / "ckpt_last.pt", fake / "ckpt_last.pt")
    with pytest.raises(L2.LaunchError, match="done.json"):
        L2.teacher_check(str(fake), str(SMOKE_M), mode="strict", ep_target="official")
    assert L2.teacher_check(str(fake), str(SMOKE_M), mode="smoke", ep_target="official")["det"]["done"] is None
    # wrong arm / not a CK2 trainer
    with pytest.raises(L2.LaunchError, match="arm"):
        L2.teacher_check(str(SMOKE_M), str(SMOKE_M), mode="smoke", deep=False, ep_target="official")
    ck1 = D / "train/ckT_p1"
    if (ck1 / "config.json").is_file():
        with pytest.raises(L2.LaunchError, match="train_ck2"):
            L2.teacher_check(str(ck1), str(SMOKE_M), mode="smoke", deep=False, ep_target="official")
    # the real CK2 teachers while they are still training: refused (the e2e runs start only after both finish)
    if (REAL_T / "config.json").is_file() and not ((REAL_T / "done.json").is_file() and (REAL_M / "done.json").is_file()):
        with pytest.raises(L2.LaunchError, match="done.json"):
            L2.teacher_check(str(REAL_T), str(REAL_M), mode="strict", deep=False)


def test_teacher_ep_target_must_match_student(tmp_path):
    """user 2026-10-08: the teachers' config.json ep_target (missing = 'official') must equal ck_e2e2.ep_target, in
    both modes; the default student target is 'official' since 2026-10-09 00:30 KST (ck_e2e2.yaml == CKE2E2Config
    default; was 'decoupled')."""
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
    assert CKE2E2Config().ep_target == "official" == L2.load_ck2_yaml()["ep_target"]
    for mode in ("smoke", "strict"):
        with pytest.raises(L2.LaunchError, match="ep_target"):
            L2.teacher_check(str(SMOKE_T), str(SMOKE_M), mode=mode, deep=False, ep_target="decoupled")
    for kw in ({}, {"ep_target": "official"}):                                              # default: official
        rec = L2.teacher_check(str(SMOKE_T), str(SMOKE_M), mode="smoke", deep=False, **kw)
        assert rec["ep_target"] == "official" and rec["det"]["ep_target"] == rec["map"]["ep_target"] == "official"
    # decoupled copies of the smoke runs (config.json + ckpt link): accepted for a decoupled student, refused for an
    # official one; one decoupled + one official teacher: refused either way
    runs = {}
    for name, src in (("T", SMOKE_T), ("M", SMOKE_M)):
        d = tmp_path / f"dep{name}"
        d.mkdir()
        cfg = json.loads((src / "config.json").read_text())
        cfg["ep_target"] = "decoupled"
        (d / "config.json").write_text(json.dumps(cfg))
        os.symlink(src / "ckpt_last.pt", d / "ckpt_last.pt")
        shutil.copy(src / "done.json", d / "done.json")
        runs[name] = d
    rec = L2.teacher_check(str(runs["T"]), str(runs["M"]), mode="strict", deep=False, ep_target="decoupled")
    assert rec["det"]["ep_target"] == rec["map"]["ep_target"] == "decoupled"
    for kw in ({}, {"ep_target": "official"}):
        with pytest.raises(L2.LaunchError, match="ep_target"):
            L2.teacher_check(str(runs["T"]), str(runs["M"]), mode="strict", deep=False, **kw)
    for ep in ("official", "decoupled"):
        with pytest.raises(L2.LaunchError, match="ep_target"):
            L2.teacher_check(str(runs["T"]), str(SMOKE_M), mode="smoke", deep=False, ep_target=ep)
    # through the override builder: the student's ck_e2e2.ep_target is the one checked
    sets = [f"teacher_det_run={runs['T']}", f"teacher_map_run={runs['M']}", "kd_calib.enabled=false"]
    ov, ck, teachers = L2.build_overrides2(str(tmp_path / "ck2e2e_dep"), ["0", "1"], ck_sets=sets + ["ep_target=decoupled"],
                                           wandb="offline", teacher_deep=False)
    assert ck["ep_target"] == "decoupled" and teachers["ep_target"] == "decoupled"
    assert "++agent.config.ck_e2e2.ep_target=decoupled" in ov
    with pytest.raises(L2.LaunchError, match="ep_target"):
        L2.build_overrides2(str(tmp_path / "ck2e2e_dep2"), ["0", "1"], ck_sets=sets + ["ep_target=official"],
                            wandb="offline", teacher_deep=False)
    # the CLI subcommand (exit 3 = refused)
    assert L2.main(["teacher-check", "--det", str(runs["T"]), "--map", str(runs["M"])]) == 3
    assert L2.main(["teacher-check", "--det", str(runs["T"]), "--map", str(runs["M"]), "--ep-target", "decoupled"]) == 0
    assert L2.main(["teacher-check", "--det", str(SMOKE_T), "--map", str(SMOKE_M), "--mode", "smoke"]) == 0
    assert L2.main(["teacher-check", "--det", str(SMOKE_T), "--map", str(SMOKE_M), "--mode", "smoke",
                    "--ep-target", "decoupled"]) == 3


def test_labeler_health_and_status(tmp_path):
    R = tmp_path / "run"
    (R / L2.IO_SUB / "lab").mkdir(parents=True)
    st = tmp_path / "wd.state"
    assert L2.labeler_health2(str(R), str(st))[0]
    sp = R / L2.IO_SUB / "lab/status.json"
    sp.write_text(json.dumps({"pid": 1, "tokens_ok": 5, "backlog_chunks": 2}))
    assert L2.labeler_health2(str(R), str(st))[0]
    ok, why = L2.labeler_health2(str(R), str(st), stall_s=10, now=time.time() + 100)
    assert not ok and "not written" in why
    s = L2.run_status2(str(R))
    assert s["train"]["alive"] is False and s["generations_done"] == [] and isinstance(s["warnings"], list)


def _sh(env_extra, *args, timeout=600):
    env = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
               E2E_WANDB="disable", **env_extra)
    for k in ("CK2_E2E_APPROVED", "E2E_SMOKE", "ARM", "E2E_CK_SET"):
        if k not in env_extra:
            env.pop(k, None)
    return subprocess.run(["bash", str(SH)] + list(args), env=env, capture_output=True, text=True, timeout=timeout)


def test_train_e2e2_dry_run(tmp_path):
    sets = tmp_path / "smoke_teachers.set"
    sets.write_text("\n".join(SMOKE_SETS) + "\n")
    base = {"E2E_ROOT": str(tmp_path / "root"), "E2E_DRY_RUN": "1", "E2E_CK_SET": str(sets), "RUN": "dry_main"}
    r = _sh(dict(base, CK2_E2E_APPROVED="1"), "start", "--gpus", "0,1")
    assert r.returncode == 0, r.stderr[-2000:]
    out = r.stdout
    assert "+trainer.params.devices=2" in out and "trainer.params.accumulate_grad_batches=16" in out
    assert "++agent.config.ck_e2e2.bev_kd.enabled=false" in out and "dry run: nothing launched (arm main" in out
    R = tmp_path / "root/dry_main"
    assert not (R / "launch/train.pid").exists() and not (R / "train").exists() and not (R / "labeler").exists()
    assert (R / "launch_dryrun/teachers.json").is_file() and (R / "launch_dryrun/ck_e2e2_effective.json").is_file()
    r = _sh(dict(base, CK2_E2E_APPROVED="1", ARM="bevkd", RUN="dry_bevkd"), "start", "--gpus", "2,3")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "++agent.config.ck_e2e2.bev_kd.enabled=true" in r.stdout and "arm bevkd" in r.stdout
    assert "++agent.config.ck_e2e2.bev_kd.teachers=[det,map]" in r.stdout
    tj = json.loads((tmp_path / "root/dry_bevkd/launch_dryrun/teachers.json").read_text())
    assert set(tj["bev_kd"]) == {"det", "map"}
    # refusals: no approval, no / wrong GPU list, real (unfinished) teachers
    assert _sh(base, "start", "--gpus", "0,1").returncode == 3
    assert _sh(dict(base, CK2_E2E_APPROVED="1"), "start").returncode == 3
    assert _sh(dict(base, CK2_E2E_APPROVED="1"), "start", "--gpus", "0").returncode == 3
    assert _sh(dict(base, CK2_E2E_APPROVED="1"), "start", "--gpus", "1,4").returncode == 3
    if not ((REAL_T / "done.json").is_file() and (REAL_M / "done.json").is_file()):
        nb = {k: v for k, v in base.items() if k != "E2E_CK_SET"}
        r = _sh(dict(nb, CK2_E2E_APPROVED="1", E2E_NO_DEEP_TEACHER_CHECK="1", RUN="dry_real"), "start", "--gpus", "0,1")
        # yaml-default teachers (ck2T10 / ck2M10) not finished (or not started yet): refused
        assert r.returncode == 3 and ("done.json" in r.stderr or "no config.json" in r.stderr), r.stderr[-1000:]
    # E2E_SMOKE only below the smoke prefix
    assert _sh(dict(base, E2E_SMOKE="1"), "start", "--gpus", "0,1").returncode == 3


def test_hydra_compose_full_override_list(tmp_path):
    """the complete override list composes with the real training config (default_training + para_ssr_agent) and the
    agent config instantiates to the effective CKE2E2Config (main and BEV-KD arm)"""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
    cfg_dir = str(REPO / "navsim/planning/script/config/training")
    for sets in ((), tuple(L2.LU.read_lines(str(L2.BEVKD_SET)))):
        ov, ck, _ = _ov(tmp_path, sets, wandb="online")
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=cfg_dir, version_base="1.2"):
            cfg = compose(config_name="default_training", overrides=ov)
        assert int(cfg.trainer.params.devices) == 2 and int(cfg.trainer.params.accumulate_grad_batches) == 16
        assert int(cfg.dataloader.params.batch_size) == 4 and cfg.trainer.params.strategy == "ddp"
        assert int(cfg.agent.config.grad_balance_warmup_iters) == 10600
        assert int(cfg.agent.config.grad_balance_interval) == 200 and int(cfg.agent.config.grad_norm_log_interval) == 200
        assert list(cfg.wandb.tags) == ["ck2e2e", L2.arm_of(ck)] and cfg.wandb.group == "ck2e2e"
        assert cfg.wandb.project == "para-ssr-v2" and cfg.wandb.name == "ck2e2e_test"
        assert dict(OmegaConf.to_container(cfg.agent.config.grad_balance_target)) == {"plan": 0.4, "det": 0.3, "map": 0.3}
        pc = instantiate(cfg.agent.config)
        assert CKE2E2Config.from_any(pc.ck_e2e2).to_dict() == CKE2E2Config.from_any(ck).to_dict()
        assert pc.ck_e2e == {} and pc.plan_score_file.endswith("pdm_score_256")
    GlobalHydra.instance().clear()


def test_smoke_token_override_quoted(tmp_path):
    """smoke_e2e2.sh write_sets: the fixed smoke token set is passed QUOTED (an unquoted digit-only token such as
    6540354015965607 or 01170848407050e2 would be parsed by Hydra as a number) and composes with the training config"""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    toks = ["6540354015965607", "01170848407050e2", "cc9b708a380b5a8a"]
    logs = ["2021.05.12.19.36.12_veh-35_00215_00405"]
    q = lambda xs: "[" + ",".join("'" + x + "'" for x in xs) + "]"   # noqa: E731  (= smoke_e2e2.sh write_sets)
    ov, _, _ = _ov(tmp_path, extra=[f"scene_filter.tokens={q(toks)}", f"scene_filter.log_names={q(logs)}"])
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "navsim/planning/script/config/training"), version_base="1.2"):
        cfg = compose(config_name="default_training", overrides=ov)
    GlobalHydra.instance().clear()
    assert list(cfg.scene_filter.tokens) == toks and all(isinstance(t, str) for t in cfg.scene_filter.tokens)
    assert list(cfg.scene_filter.log_names) == logs


def test_gen_roundtrip2_matches_any_record(tmp_path):
    """smoke check: a labelled row must equal ANY of its records (a resumed epoch leaves two attempts that recorded
    the same row differently; the labeler keeps the first one IT scored, not the first chunk in listing order)"""
    import numpy as np
    from navsim.agents.para_ssr.ck import e2e_data as ED
    from navsim.agents.para_ssr.ck import e2e_data2 as D2
    io = tmp_path / "ck_e2e2"
    rng = np.random.default_rng(0)
    t_a = rng.normal(size=(2, 96, 8, 3)).astype(np.float32)          # attempt a (killed): rows 0, 1
    t_b = t_a.copy()
    t_b[0] += 1.0                                                    # attempt b (resumed): row 0 differs
    valid = np.ones((2, 96), bool)
    for att, t in (("aaaaaaaa", t_a), ("bbbbbbbb", t_b)):
        D2.write_rec2_chunk(ED.rec_chunk_path(io, 1, 0, att, 0), [0, 1], t, valid, np.zeros((2, 16)),
                            np.zeros((2, 16)), np.zeros(2), 0, 1, 0)
    g = D2.open_generation2(io, 1, n_rows=4, mode="w+")
    lab = np.zeros((2, 96, 9), np.float32)
    D2.write_generation_rows2(g, [0, 1], np.stack([t_b[0], t_a[1]]), lab, valid, valid)   # row 0 from attempt b
    r = L2.gen_roundtrip2(io, [1])[1]
    assert r["rows_recorded"] == 2 and r["rows_labelled"] == 2 and r["mismatch"] == 0
    D2.write_generation_rows2(g, [0], (t_b[0] + 5.0)[None], lab[:1], valid[:1], valid[:1], overwrite=True)
    assert L2.gen_roundtrip2(io, [1])[1]["mismatch"] == 1          # matches no record


# ----------------------------------------------------------------------------------------------- smoke exit code (F2)
_FAKEPY = r"""#!/bin/bash
# stub python for smoke_e2e2.sh: heavy commands pass / fail per STUB_* env; '-' (heredoc) runs the REAL python
case "$1" in
  -) shift; exec /venv/ssr/bin/python - "$@";;
  -m) echo "== 1 failed, 41 passed in 3.21s =="; exit "${STUB_PYTEST_RC:-0}";;
  */launch_util2.py)
    case "$2" in
      gpu-check) exit 0;;
      smoke-check2)
        shift 2; rd=""; out=""
        while [ $# -gt 0 ]; do case "$1" in --run-dir) rd=$2; shift 2;; --out) out=$2; shift 2;; *) shift;; esac; done
        n=$(basename "$rd")
        case ",${STUB_CHECK_CRASH:-}," in *",$n,"*) echo "Traceback: KeyError (stub crash)" >&2; exit 1;; esac
        case ",${STUB_CHECK_FAIL:-}," in
          *",$n,"*) echo '{"pass": false}' > "$out"; echo '{"pass": false}'; exit 4;;
          *) echo '{"pass": true}' > "$out"; echo '{"pass": true}'; exit 0;;
        esac;;
    esac;;
  tools/ck/e2e2/eval_e2e2.py)
    shift; rd=""; sp=""
    while [ $# -gt 0 ]; do case "$1" in --run-dir) rd=$2; shift 2;; --split) sp=$2; shift 2;; *) shift;; esac; done
    case "${STUB_EVAL:-ok}" in
      fail) echo "SystemExit: stub eval failure ($sp)"; exit 19;;
      noop) exit 0;;
      ok|icfalse)
        n=$(basename "$rd"); r=$rd/eval
        mkdir -p "$r/root/eval/$n/$sp" "$r/root/infer/$n/$sp"
        echo '{}' > "$r/root/eval/$n/$sp/metrics.json"; echo '{}' > "$r/summary.json"
        [ "${STUB_EVAL}" = icfalse ] && ic=false || ic=true
        echo "{\"pass\": $ic}" > "$r/root/infer/$n/$sp/infer_check.json"; exit 0;;
    esac;;
esac
echo "fakepy: unhandled $*" >&2; exit 99
"""
_FAKETE = r"""#!/bin/bash
# stub train_e2e2.sh: start writes train.exit = 0 at once (STUB_START_FAIL lists refused runs; g2 also gets its
# epoch-1 checkpoint unless STUB_G2_NOCKPT=1); wait (train or labeler) -> STUB_LAB_RC
echo "$RUN $ARM $*" >> "$STUB_CALLS"
case "$1" in
  start) case ",${STUB_START_FAIL:-}," in *",$RUN,"*) exit 3;; esac
         if [ "$RUN" = g2 ] && [ "${STUB_G2_NOCKPT:-0}" != 1 ]; then
           c="$E2E_ROOT/g2/train/lightning_logs/version_0/checkpoints"; mkdir -p "$c"; : > "$c/epoch=1-step=10.ckpt"
         fi
         mkdir -p "$E2E_ROOT/$RUN/launch"; echo 0 > "$E2E_ROOT/$RUN/launch/train.exit"; exit 0;;
  wait) exit "${STUB_LAB_RC:-0}";;
  *) exit 0;;
esac
"""


def _smoke(tmp, tag, stages, stale=False, **stub):
    """run the REAL smoke_e2e2.sh with the python / train_e2e2.sh stubs (SMOKE_PY / SMOKE_TE / SMOKE_ROOT) ->
    (exit code, {stage: pass}, call log)."""
    bindir = tmp / "bin"
    if not bindir.is_dir():
        bindir.mkdir()
        for name, body in (("fakepy", _FAKEPY), ("fakete.sh", _FAKETE)):
            (bindir / name).write_text(body)
            (bindir / name).chmod(0o755)
    root = tmp / "sr"
    sr = root / tag
    sr.mkdir(parents=True)
    (sr / "g_tokens.txt").write_text("scene_filter.tokens=['a']\n")      # write_sets: token set already present
    if stale:                                  # outputs of an earlier PASSING G3 under the same TAG (1 h old)
        r = sr / "g1_main/eval"
        old = time.time() - 3600
        files = [r / "summary.json"]
        for s in ("navtrain_val", "navtest"):
            for d, f, txt in (("eval", "metrics.json", "{}"), ("infer", "infer_check.json", '{"pass": true}')):
                p = r / "root" / d / "g1_main" / s / f
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(txt)
                files.append(p)
        files[0].write_text("{}")
        for p in files:
            os.utime(p, (old, old))
    env = dict(os.environ, TAG=tag, SMOKE_PY=str(bindir / "fakepy"), SMOKE_TE=str(bindir / "fakete.sh"),
               SMOKE_ROOT=str(root), STUB_CALLS=str(sr / "calls.txt"), OMP_NUM_THREADS="1", G2_STOP_DELAY="0",
               **{f"STUB_{k.upper()}": str(v) for k, v in stub.items()})
    rc = subprocess.run(["bash", str(REPO / "tools/ck/e2e2/smoke_e2e2.sh"), *stages], env=env, capture_output=True,
                        text=True, timeout=120).returncode
    res = {f.stem[len("result_"):]: json.loads(f.read_text()).get("pass") for f in sr.glob("result_*.json")}
    calls = (sr / "calls.txt").read_text() if (sr / "calls.txt").is_file() else ""
    return rc, res, calls


@pytest.mark.parametrize("tag, stages, stale, stub, want", [
    ("allpass", ["G0", "G1", "G2", "G3"], False, {},
     {"G0": True, "G1_main": True, "G1_bevkd": True, "G2": True, "G3": True}),
    ("g0_pytest", ["G0"], False, {"pytest_rc": 1}, {"G0": False}),
    ("g1_mainfail", ["G1"], False, {"check_fail": "g1_main"}, {"G1_main": False, "G1_bevkd": True}),
    ("g1_maincrash", ["G1"], False, {"check_crash": "g1_main"}, {"G1_bevkd": True}),
    ("g1_bevkdfail", ["G1"], False, {"check_fail": "g1_bevkd"}, {"G1_main": True, "G1_bevkd": False}),
    ("g1_labtimeout", ["G1"], False, {"lab_rc": 124}, {"G1_main": True, "G1_bevkd": True}),
    ("g1_bevkd_start", ["G1"], False, {"start_fail": "g1_bevkd"}, {}),
    ("g2_waitfail", ["G2"], False, {"lab_rc": 124}, {"G2": True}),
    ("g2_checkfail", ["G2"], False, {"check_fail": "g2"}, {"G2": False}),
    ("g2_nockpt", ["G2"], False, {"g2_nockpt": 1}, {"G2": True}),
    ("g3_evalfail", ["G3"], False, {"eval": "fail"}, {"G3": False}),
    ("g3_icfalse", ["G3"], False, {"eval": "icfalse"}, {"G3": False}),
    ("g3_failstale", ["G3"], True, {"eval": "fail"}, {"G3": False}),
    ("g3_noopstale", ["G3"], True, {"eval": "noop"}, {"G3": False}),
    ("g3_rerun_ok", ["G3"], True, {}, {"G3": True}),
    ("all3fail", ["G0", "G1", "G3"], False, {"pytest_rc": 1, "check_fail": "g1_main", "eval": "fail"},
     {"G0": False, "G1_main": False, "G1_bevkd": True, "G3": False}),
])
def test_smoke_exit_code_failure_injection(tmp_path, tag, stages, stale, stub, want):
    """report 48 F2: the smoke's exit code is nonzero for every injected failure (G0 pytest rc, G1 main check failing
    or crashing while BEV-KD passes, BEV-KD failing, a labeler wait timeout, a refused BEV-KD start, G2 train / labeler
    wait, check or missing epoch-1 checkpoint, G3 eval rc, G3 infer_check false, G3 outputs left by an earlier attempt
    = stale) and 0 when everything passes; the result_*.json files agree with the exit code."""
    rc, res, calls = _smoke(tmp_path, tag, stages, stale, **stub)
    ok = all(want.values()) and tag in ("allpass", "g3_rerun_ok")
    assert (rc == 0) == ok, (tag, rc, res)
    assert res == want, (tag, res)
    if tag == "g1_bevkd_start":                # F2f: the already started main arm is stopped, not left orphaned
        assert "g1_main main stop" in calls and "g1_bevkd bevkd start" in calls, calls
    if "G1" in stages and tag != "g1_bevkd_start":                 # both labelers waited for, both checks run
        assert sum(1 for x in calls.splitlines() if x.startswith("g1_") and "wait --labeler" in x) == 2
