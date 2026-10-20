#!/usr/bin/env python
"""CK2 e2e launch helpers (SPEC ck2e2e s7).  Used by train_e2e2.sh / smoke_e2e2.sh; importable for tests.
tools/ck/e2e/launch_util.py is imported unchanged (recipe, hydra_value, flatten, apply_sets, read_lines, GPU / pid / ckpt
helpers, W&B mode); this file adds the CK2 recipe, the ck_e2e2 dict, the teacher guards and the ck_e2e2 io-dir status.

  launch_util2.py overrides  --run-dir R --gpus 0,1 [--workers 6] [--max-epochs 30] [--ck-set k=v ...] [--ck-set-file F]
                             [--extra-file F] [--resume-ckpt P] [--wandb online|offline|disable]
                             [--teacher-check strict|smoke] --out R/launch/overrides.txt
  launch_util2.py teacher-check --det RUN --map RUN [--mode strict|smoke] [--ep-target decoupled|official] [--out F.json]
                                [--kd-calib-det F.json] [--kd-calib-map F.json]            exit 3 when not usable
  launch_util2.py gpu-check  --gpus 0,1 [--max-used-mib 1024] [--list-only]               2 or 4 GPUs of 0-3
  launch_util2.py labeler-health --run-dir R --state F [--stall-s 1200]                   exit 5 when the labeler hangs
  launch_util2.py status     --run-dir R [--json]
  launch_util2.py smoke-check2 --run-dir R --out F [--ngpu 2] [--onpolicy-epochs 2] [--gen-epochs 1]   GPU smoke summary
  (find-last / wandb-mode: tools/ck/e2e/launch_util.py)

Recipe (user): v2 r34 source on 2 GPUs = the old 4-GPU recipe with devices 2, batch 4 / GPU, accumulate_grad_batches 16
(global 128), grad_balance_warmup_iters 10600, grad_balance_interval 200, grad_norm_log_interval 200.  4 GPUs (0-3, user
2026-10-13, fusion_v1) = the old 4-GPU counters: devices 4, accumulate 8 (global 128), 5300 / 100 / 100, and the
ck_e2e2 per-rank micro-batch counters halved (MB_COUNTERS: lat_kd.start_mb, label_refresh_every_mb, grad_share_every,
bev_kd.start_mb), so every schedule keeps its meaning in samples / optimizer steps; experiment_name
<run> (prefixed 'ck2e2e_' unless it already starts with 'ck2e2e'), W&B project para-ssr-v2, name <run>, group ck2e2e, tags [ck2e2e, main | bevkd].  plan_score_file stays the
WoTE pdm_score_256 (user: v2 keeps its labels).  Then '++agent.config.ck_e2e2.<leaf>=<value>' for every leaf of
ck_e2e2.yaml (+ --ck-set / --ck-set-file lines) and output_dir=<R>/train/.
Teacher guard (teacher_check, both modes): config.json arm T / M, trainer train_ck2, ckpt present, lon head zero,
ep_target == the student's ck_e2e2.ep_target (a run without the key = 'official'; user 2026-10-08: label BCE and EP KD
must mean the same EP), and in strict mode done.json (train_e2e2.sh start refuses until both teachers finished).
BEV-KD arm guard (bevkd_norms): every bev_kd.teachers entry has its frozen z-score file in its teacher run (det ->
teacher_det_run/norm.npz, map -> teacher_map_run/norm_map.npz; bev_kd.NORM_FILE), loadable and finite; recorded with
its sha16 in teachers.json ('bev_kd').
KD-calibration guard (teacher_check(kd_calib=...) -> kd_calib_record, both modes; user 2026-10-08 ~22:35 KST): with
ck_e2e2.kd_calib.enabled every configured file (kd_calib.det / .map) must exist and match its teacher (navsim
ck/kd_calib.check_calib: arm, run, which, the ckpt sha16 of teacher_check, ep_target, every KD key of that teacher
fitted); a missing file is refused with the fit_kd_calib.py command to run after the teacher has finished.  Recorded
(file sha16, a / b per key, held-out NLL / ECE) in teachers.json ('kd_calib').
Temporal DET teacher guard (refiner-v3, docs/refiner_v3/STEP2_E2E.md; tools/ck/e2e2/v3teacher_arm.set): a teacher whose
config.json has temporal.on must be the DET teacher (arm T) and NOT privileged (oracle forecaster / gt source refused
in both modes); teacher_check records under det['temporal'] its cfg, the sha16 of the temporal-branch / forecaster
weights in the checkpoint (deep), the pre-trained forecaster file (fc_init sha), the ca_scales file and the training
provenance of the teacher.  kd_objects_check (both modes) opens the object pack the e2e target builder will read
(e2e_data2.KDObjRows: split = basename of warmup.packed, ObjPack with the packed token order) and REFUSES a pack whose
build.json sha256 differs from the one the teacher was trained on (config.json provenance data.packs.<split>) -> recorded
as teachers.json 'kd_objects' (pack record + equality).  CK2 teachers: no new key, nothing opened (records unchanged).
smoke-check2 with a temporal DET teacher (effective teacher_det_run temporal.on) adds the gate temporal_smoke_checks:
teachers.json det.temporal + kd_objects recorded and ck2/kd_obj_ok_frac logged with mean >= 0.999 (CK2: unchanged).
refiner v4_3 (tools/ck/e2e2/v4_3_arm.set, ARM=v4_3; arm_of 'v4_3' for a range readout): apply_sets2 declares the
optional v4 keys a set line names (absent from ck_e2e2.yaml: older arms unchanged); placeholder_check refuses any
ck_e2e2 value still holding the arm file's /PLACEHOLDER/ (teacher runs / KD calibration files of the range teachers,
set by the user's E2E_CK_SET file; refiner v5: the Z adapters' teacher-statistics records v4.v5.z_stats_det / _map,
refused only while v4.v5.z_adapt is on, as kd_calib.* only while kd_calib is enabled; z_stats_check then ties each
record to its teacher -- run, which, ckpt sha16, z_stride; strict mode refuses dry-run records -- and records them as
teachers.json 'z_stats'); kd_constraint_check (v4.kd_constraint.enabled) refuses non-range teachers, teachers on
another grid (corridor version, time bins) than the student, teachers whose rg_stopped_aware differs from the student's
(Phase 0b R2 B semantics: an R2 student needs R2 teachers and vice versa) and an occ_channels group missing on the
student or the DET teacher -- the runtime rule and message (online2.constraint_teacher_problems / CONSTRAINT_REFUSAL) on
the runs' config.json -- and records them as teachers.json 'kd_constraint' (rg_read of both sides recorded; it may
differ, rg_read_differs names the teachers whose read is not the student's).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = str(HERE.parents[2])
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from tools.ck.e2e import launch_util as LU  # noqa: E402  (imported unchanged)

LaunchError = LU.LaunchError
PY = LU.PY
CK2_YAML = HERE / "ck_e2e2.yaml"
BEVKD_SET = HERE / "bevkd_arm.set"
MAIN_SET = HERE / "main_arm.set"          # refiner-v4: the ck2e2e_main arm (BEV KD off; the yaml default is bevkd)
V4_SET = HERE / "v4_arm.set"              # refiner v4 student
V4_3_SET = HERE / "v4_3_arm.set"          # refiner v4_3 student (range critic + computed edits + constraint KD)
PLACEHOLDER = "/PLACEHOLDER/"             # an arm-file value that must be replaced before a launch (refused)
CK_DATA = "/home/external-user/ssd/yongjae_refiner/ck"
E2E_ROOT = f"{CK_DATA}/ck2/e2e"
SMOKE_ROOT = f"{CK_DATA}/ck2/e2e_smoke"
IO_SUB = "ck_e2e2"
NGPU = 2                          # user: 2 GPUs per run (devices 2)
NGPU_ALLOWED = (2, 4)             # user 2026-10-13: 4 GPUs for the fusion_v1 / fusion_v10 runs
EXTRA_GPUS = ("4", "5", "6", "7")  # only with CK_ALLOW_GPU45=1 (2,3,4,5 or 4,5,6,7)
ACCUMULATE = 16                   # user: 2 x 4 x 16 = 128
BALANCE_WARMUP = 10600            # user
BALANCE_INTERVAL = 200            # user
GRAD_NORM_LOG = 200               # user
TOKENS_PER_MB = 4
GLOBAL_BATCH = NGPU * TOKENS_PER_MB * ACCUMULATE            # 128
# ck_e2e2 leaves counted in per-rank micro-batches (the 2-GPU values doubled the 4-GPU run's, ck_e2e2.yaml)
MB_COUNTERS = ("lat_kd.start_mb", "label_refresh_every_mb", "grad_share_every", "bev_kd.start_mb")
WANDB_PROJECT = "para-ssr-v2"
WANDB_GROUP = "ck2e2e"
GSHARE_WARN = 0.5
KST = 9 * 3600


def kst(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime((time.time() if t is None else t) + KST))


# ----------------------------------------------------------------------------------------------- ck_e2e2 dict
def load_ck2_yaml(path=CK2_YAML) -> Dict[str, Any]:
    return LU.load_ck_yaml(path)


def ck2_from_any(ck: Dict[str, Any]):
    """CKE2E2Config.from_any + validate(parent=None) (the agent repeats the parent rules); LaunchError on failure."""
    from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
    try:
        c = CKE2E2Config.from_any(ck)
        c.validate(None)
    except ValueError as e:
        raise LaunchError(f"ck_e2e2: {e}") from None
    return c


def check_paths(ck: Dict[str, Any]) -> None:
    for k, v in LU.flatten(ck):
        if isinstance(v, str) and v.startswith("/"):
            LU.no_equals(v, f"ck_e2e2.{k}")


def arm_of(ck: Dict[str, Any]) -> str:
    v4 = ck.get("v4") or {}
    if str(v4.get("readout", "mean")) == "range":
        return "v4_3"
    if str(v4.get("readout", "mean")) != "mean" or str(v4.get("pref_head", False)).lower() in ("true", "1"):
        return "v4"
    return "bevkd" if bool((ck.get("bev_kd") or {}).get("enabled")) else "main"


# ----------------------------------------------------------------------------------------------- teachers
def lon_head_zero(ckpt_path: Path) -> Tuple[bool, str]:
    """last layer of trunk.lon_head all zero in ckpt['model'] (CK2 teacher: lon head removed)."""
    import re
    import torch
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    idx = sorted({int(m.group(1)) for k in sd for m in [re.match(r"trunk\.lon_head\.(\d+)\.weight$", k)] if m})
    if not idx:
        return False, "no trunk.lon_head.<i>.weight in the checkpoint"
    w, b = sd[f"trunk.lon_head.{idx[-1]}.weight"], sd.get(f"trunk.lon_head.{idx[-1]}.bias")
    mx = max(float(w.abs().max()), float(b.abs().max()) if b is not None else 0.0)
    return mx == 0.0, f"trunk.lon_head.{idx[-1]} max |w| {mx:g}"


def _state_sha16(sd: Dict[str, Any], prefix: str) -> Optional[str]:
    """sha16 of the tensors under `prefix` of a state dict (sorted names, dtype, shape, bytes); None if none
    (= online2._tensor_sha16 of the module's own state dict, whose keys lack the prefix)."""
    import hashlib
    ks = sorted(k for k in sd if k.startswith(prefix))
    if not ks:
        return None
    h = hashlib.sha256()
    for k in ks:
        v = sd[k].detach().cpu().contiguous()
        h.update(k[len(prefix):].encode())
        h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()[:16]


def temporal_record(rd: Path, cfg: Dict[str, Any], ck: Path, deep: bool = True) -> Optional[Dict[str, Any]]:
    """Temporal-branch record of a teacher run (None when its config.json has no temporal.on): cfg, privileged flag,
    checkpoint state sha16 of the branch / forecaster (deep), fc_init / ca_scales file records
    (tools/ck/v3/provenance) and the teacher's training provenance (packs, ca_scales, fc_init) for the comparisons."""
    from navsim.agents.para_ssr.ck.temporal.config import TemporalCfg
    tc = TemporalCfg.from_dict(cfg.get("temporal"))
    if not tc.on:
        return None
    from tools.ck.v3 import provenance as PV
    rec: Dict[str, Any] = {"cfg": tc.to_dict(), "privileged": tc.privileged}
    if deep:
        import torch
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        rec["branch_state_sha16"] = _state_sha16(sd, "temporal.")
        rec["forecaster_state_sha16"] = _state_sha16(sd, "temporal.forecaster.")
    rec["fc_init"] = PV.fc_init_record(tc.fc_init) if tc.fc_init else None
    if deep and tc.fc_init and Path(tc.fc_init).is_file() and rec.get("forecaster_state_sha16"):
        # frozen pre-trained forecaster: the checkpoint's forecaster must be the fc_init weights (sha16 of its
        # 'forecaster' state dict, same scheme)
        import torch
        fi = torch.load(tc.fc_init, map_location="cpu", weights_only=False)
        fsd = fi.get("forecaster") if isinstance(fi, dict) else None
        if isinstance(fsd, dict):
            rec["fc_init"]["forecaster_state_sha16"] = _state_sha16(fsd, "")
            rec["forecaster_equals_fc_init"] = rec["fc_init"]["forecaster_state_sha16"] == rec["forecaster_state_sha16"]
    rec["ca_scales"] = PV.ca_record(tc.ca_scales)
    prov = (cfg.get("provenance") or {}).get("data") or {}
    rec["train_provenance"] = {"packs": {s: (r or {}).get("build_sha256") for s, r in (prov.get("packs") or {}).items()},
                               "ca_scales_sha256": (prov.get("ca_scales") or {}).get("sha256"),
                               "fc_init_sha256": (prov.get("fc_init") or {}).get("sha256")}
    warn = []
    tp = rec["train_provenance"]
    if rec["fc_init"] and tp["fc_init_sha256"] and rec["fc_init"].get("sha256") != tp["fc_init_sha256"]:
        warn.append("fc_init file changed since the teacher was trained (the checkpoint's own forecaster weights are "
                    "used; recorded only)")
    if tc.fc_frozen and rec.get("forecaster_equals_fc_init") is False:
        warn.append("frozen forecaster weights in the checkpoint differ from the fc_init file (recorded only)")
    if tp["ca_scales_sha256"] and rec["ca_scales"].get("sha256") != tp["ca_scales_sha256"]:
        warn.append("ca_scales file changed since the teacher was trained (a CA branch carries its scales in the "
                    "checkpoint; recorded only)")
    rec["warnings"] = warn
    return rec


def kd_objects_check(c, teachers: Dict[str, Any]) -> Dict[str, Any]:
    """Object pack of a temporal DET teacher (empty dict for a CK2 teacher): opens e2e_data2.KDObjRows(teacher cfg,
    warmup.packed) exactly as the target builder will (structure, token order), records it (tools/ck/v3/provenance
    pack_record) and REFUSES (LaunchError) a pack whose build.json sha256 differs from the one recorded in the
    teacher's training provenance for that split (the KD objects must be the rows the teacher was trained on)."""
    from navsim.agents.para_ssr.ck import e2e_data2 as D2
    tc = D2.teacher_temporal(c.teacher_det_run)
    if not tc.on:
        return {}
    from tools.ck.v3 import provenance as PV
    if tc.privileged:
        raise LaunchError(f"teacher det {c.teacher_det_run}: PRIVILEGED temporal run")
    try:
        ko = D2.KDObjRows(tc, c.warmup["packed"])
    except (ValueError, FileNotFoundError, ImportError) as e:
        raise LaunchError(f"temporal DET teacher {c.teacher_det_run}: object pack for the e2e rows: {e}") from None
    rec = PV.pack_record(ko.pack.dir)
    rec.update(ko.describe())
    want = (((teachers.get("det") or {}).get("temporal") or {}).get("train_provenance") or {}).get("packs", {}).get(
        ko.split)
    if want is None:
        cfg = json.loads((Path(c.teacher_det_run) / "config.json").read_text())
        want = ((((cfg.get("provenance") or {}).get("data") or {}).get("packs") or {}).get(ko.split) or {}).get(
            "build_sha256")
    rec["teacher_train_pack_sha256"] = want
    if want is None:
        raise LaunchError(f"temporal DET teacher {c.teacher_det_run}: config.json provenance records no object pack "
                          f"for {ko.split} (cannot prove the KD objects are the teacher's training rows)")
    if rec.get("build_sha256") != want:
        raise LaunchError(f"temporal DET teacher {c.teacher_det_run}: object pack {ko.pack.dir} build.json sha256 "
                          f"{rec.get('build_sha256')} != {want} recorded when the teacher was trained (pack rebuilt?)")
    rec["same_as_teacher_training"] = True
    return rec


def teacher_check(det_run: str, map_run: str, which: str = "last", mode: str = "strict",
                  deep: bool = True, ep_target: Optional[str] = None, kd_calib: Optional[Dict[str, Any]] = None,
                  kd_keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Refuse a teacher run that is not a finished CK2 run of the right arm (SPEC s7-3 guards):
    config.json arm T / M, trainer 'train_ck2', ckpt_<which>.pt present, lon head last layer zero (deep), config.json
    ep_target == ep_target (the student's ck_e2e2.ep_target; None = the CKE2E2Config default; a run without the key
    is 'official'), and in mode 'strict' done.json present (training finished).  -> record of both runs (ckpt sha16,
    done.json, mtimes, ep_target).  kd_calib (the student's ck_e2e2.kd_calib dict; None = not checked): when enabled,
    every configured calibration file must exist and match its teacher (kd_calib_record; KD keys kd_keys, None = the
    default nc / dac / ep / ttc) -> record['kd_calib']."""
    import hashlib
    from navsim.agents.para_ssr.ck.ep_target import check_ep_target, run_ep_target
    if ep_target is None:
        from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
        ep_target = CKE2E2Config().ep_target
    try:
        want_ep = check_ep_target(ep_target)
    except ValueError as e:
        raise LaunchError(f"student {e}") from None
    out: Dict[str, Any] = {"mode": mode, "which": which, "checked": kst(), "ep_target": want_ep}
    for name, run, arm in (("det", det_run, "T"), ("map", map_run, "M")):
        rd = Path(str(run))
        LU.no_equals(str(rd), f"teacher {name} run")
        cj = rd / "config.json"
        if not cj.is_file():
            raise LaunchError(f"teacher {name} {rd}: no config.json")
        cfg = json.loads(cj.read_text())
        if cfg.get("arm") != arm:
            raise LaunchError(f"teacher {name} {rd}: arm {cfg.get('arm')!r} != {arm!r}")
        if cfg.get("trainer") != "train_ck2":
            raise LaunchError(f"teacher {name} {rd}: trainer {cfg.get('trainer')!r} != 'train_ck2'")
        try:
            t_ep = run_ep_target(cfg)
        except ValueError as e:
            raise LaunchError(f"teacher {name} {rd}: config.json {e}") from None
        if t_ep != want_ep:
            raise LaunchError(f"teacher {name} {rd}: ep_target {t_ep!r} (config.json; missing = 'official') != the "
                              f"student's ck_e2e2.ep_target {want_ep!r} (label BCE and EP KD would disagree)")
        done = rd / "done.json"
        if mode == "strict" and not done.is_file():
            raise LaunchError(f"teacher {name} {rd}: no done.json (CK2 teacher still training; the e2e runs start only "
                              f"after both teachers have finished)")
        ck = rd / (which if which.endswith(".pt") else f"ckpt_{which}.pt")
        if not ck.is_file():
            raise LaunchError(f"teacher {name} {rd}: {ck.name} missing")
        rec: Dict[str, Any] = {"run": str(rd), "arm": arm, "ckpt": str(ck), "ckpt_mtime": kst(ck.stat().st_mtime),
                               "ckpt_bytes": ck.stat().st_size, "done": json.loads(done.read_text())
                               if done.is_file() else None, "epochs": cfg.get("epochs"), "ep_target": t_ep}
        h = hashlib.sha256()
        with open(ck, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        rec["ckpt_sha16"] = h.hexdigest()[:16]
        if deep:
            ok, why = lon_head_zero(ck)
            rec["lon_head"] = why
            if not ok:
                raise LaunchError(f"teacher {name} {rd}: {why} (not a CK2 run: z_lon must be 0)")
        # refiner-v3 temporal teacher (its config.json; CK2 runs: no key added)
        try:
            from navsim.agents.para_ssr.ck.temporal.config import TemporalCfg
            tcfg = TemporalCfg.from_dict(cfg.get("temporal"))
        except ValueError as e:
            raise LaunchError(f"teacher {name} {rd}: config.json temporal: {e}") from None
        if tcfg.on:
            if tcfg.privileged:
                raise LaunchError(f"teacher {name} {rd}: PRIVILEGED temporal run (forecaster {tcfg.forecaster!r}, "
                                  f"source {tcfg.source!r}): analysis only, never a KD teacher")
            if name != "det":
                raise LaunchError(f"teacher {name} {rd}: temporal branch on a MAP teacher is not supported")
            try:
                tcfg.validate(arm)
            except (ValueError, NotImplementedError) as e:
                raise LaunchError(f"teacher {name} {rd}: temporal cfg: {e}") from None
            rec["temporal"] = temporal_record(rd, cfg, ck, deep)
        out[name] = rec
    if kd_calib is not None:
        out["kd_calib"] = kd_calib_record(kd_calib, out, det_run, map_run, which, kd_keys)
    return out


def warmup_check(c) -> Dict[str, Any]:
    """Warm-up label sources of a CKE2E2Config (e2e_data2._WUArrays.check): structure, row counts, variant table and
    the provenance guard (packed token order == the variant build / packed meta / raw-label plan token_sha16, anchors
    sha16, anchor-sampler config of the build; report 48 S56-5).  LaunchError on a mismatch; -> record (teachers.json
    'warmup_sources')."""
    from navsim.agents.para_ssr.ck import e2e_data2 as D2
    try:
        return D2._WUArrays(D2.warmup_cfg(c.to_dict()), c.ep_target).check()
    except (D2.WUSourceError, FileNotFoundError) as e:
        raise LaunchError(f"ck_e2e2.warmup sources: {e}") from None


def bevkd_norms(c) -> Dict[str, Any]:
    """BEV-KD arm z-score files of a CKE2E2Config (empty dict when the arm is off): {t: {run, norm, sha16, mean_abs,
    std_min}}; LaunchError when a teacher's file is missing or bad (the arm would fail at agent construction)."""
    import hashlib
    from navsim.agents.para_ssr.ck.bev_kd import NORM_FILE, load_teacher_norm
    bk = c.bev_kd
    if not bk["enabled"]:
        return {}
    runs = {"det": c.teacher_det_run, "map": c.teacher_map_run}
    out: Dict[str, Any] = {}
    for t in bk["teachers"]:
        if not runs.get(t):
            raise LaunchError(f"bev_kd teacher {t}: no teacher run")
        p = Path(runs[t]) / NORM_FILE[t]
        if not p.is_file():
            raise LaunchError(f"bev_kd teacher {t}: {p} missing (frozen z-score of the teacher BEV)")
        try:
            mean, std = load_teacher_norm(p)
        except Exception as e:      # noqa: BLE001
            raise LaunchError(f"bev_kd teacher {t}: {p}: {e}") from None
        out[t] = {"run": str(runs[t]), "norm": str(p), "sha16": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
                  "mean_abs": float(abs(mean).mean()), "std_min": float(std.min())}
    return out


def kd_calib_record(kd_calib: Optional[Dict[str, Any]], teachers: Dict[str, Any], det_run: str, map_run: str,
                    which: str, kd_keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """KD calibration files (ck_e2e2.kd_calib dict {enabled, det, map}) against the teacher_check record (ckpt sha16 /
    ep_target of the runs).  -> {'enabled': False} when off / None, else {'enabled': True, 'det': info | None, 'map':
    info | None} (kd_calib.load_teacher_calib info: path, sha16, run, which, ckpt_sha16, ep_target, fitted_keys,
    params, heldout).  LaunchError when a configured file is missing or does not match its teacher."""
    from navsim.agents.para_ssr.ck.kd_calib import load_teacher_calib
    if not kd_calib or not kd_calib.get("enabled"):
        return {"enabled": False}
    out: Dict[str, Any] = {"enabled": True}
    for name, arm, run in (("det", "T", det_run), ("map", "M", map_run)):
        p = kd_calib.get(name)
        if not p:
            out[name] = None
            continue
        LU.no_equals(str(p), f"ck_e2e2.kd_calib.{name}")
        if not Path(p).is_file():
            raise LaunchError(
                f"ck_e2e2.kd_calib.{name}: calibration file {p} missing (KD calibration of teacher {run} "
                f"'{which}' not fitted yet): after the teacher has finished run on one free GPU  "
                f"CUDA_VISIBLE_DEVICES=<g> PYTHONPATH={REPO} {PY} {REPO}/tools/ck/e2e2/fit_kd_calib.py --run {run} "
                f"--which {which} --out {p}   (or set kd_calib.enabled=false)")
        rec = teachers.get(name) or {}
        try:
            _, _, info = load_teacher_calib(p, run=run, which=which, ckpt_sha16=rec.get("ckpt_sha16"),
                                            ep_target=rec.get("ep_target") or teachers.get("ep_target"), arm=arm,
                                            kd_keys=kd_keys)
        except (ValueError, FileNotFoundError) as e:
            raise LaunchError(f"ck_e2e2.kd_calib.{name}: {e}") from None
        out[name] = info
    return out


def apply_sets2(ck: Dict[str, Any], sets: Sequence[str]) -> Dict[str, Any]:
    """launch_util.apply_sets with the OPTIONAL v4_3 keys of the v4 block (online2._d_v4_opt): a set line naming one
    (v4.<key>=..., v4.kd_constraint.<key>=...) first declares it at its default -- they are absent from ck_e2e2.yaml so
    the older arms' effective dicts / override lists stay unchanged -- then the lines are applied as before."""
    import copy
    from navsim.agents.para_ssr.ck.online2 import _d_v4_opt
    opt = _d_v4_opt()
    out = copy.deepcopy(ck)
    for ln in sets:
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        parts = ln.split("=", 1)[0].strip().split(".")
        if len(parts) >= 2 and parts[0] == "v4" and parts[1] in opt and isinstance(out.get("v4"), dict):
            out["v4"].setdefault(parts[1], copy.deepcopy(opt[parts[1]]))
    return LU.apply_sets(out, sets)


def placeholder_check(ck: Dict[str, Any]) -> None:
    """Refuse an effective ck_e2e2 dict that still holds an arm file's PLACEHOLDER value (v4_3_arm.set: the KD
    calibration files of its R2 teachers v4T10_rng2 / v4M10_rng2 are set by the user's E2E_CK_SET file once those
    teachers exist; the teacher runs themselves are named and refused by teacher_check until they have finished)."""
    kc_on = str((ck.get("kd_calib") or {}).get("enabled", True)).lower() in ("true", "1")
    # refiner v5 'zkd' group (v5_arm.set): the Z adapters' teacher statistics, refused exactly like kd_calib -- only
    # while v4.v5.z_adapt is on (unused statistics paths when it is off)
    v5 = (ck.get("v4") or {}).get("v5") or {}
    za_on = str(v5.get("z_adapt", False)).lower() in ("true", "1")
    bad = [k for k, v in LU.flatten(ck) if isinstance(v, str) and PLACEHOLDER in v
           and (kc_on or not k.startswith("kd_calib."))                # unused calibration paths when it is off
           and (za_on or not k.startswith("v4.v5.z_stats_"))]
    if bad:
        raise LaunchError(f"ck_e2e2 {bad} still hold the arm file's placeholder ({PLACEHOLDER}...): set them (e.g. "
                          f"E2E_CK_SET=<file> with teacher_det_run=<range DET run> / teacher_map_run=<range MAP run> / "
                          f"kd_calib.det|map=<their calibration files> or kd_calib.enabled=false / "
                          f"v4.v5.z_stats_det|map=<the teachers' tools/ck/v5/fit_z_stats.py records> or "
                          f"v4.v5.z_adapt=false) before a launch")


def z_stats_check(c, teachers: Dict[str, Any], mode: str = "strict") -> Dict[str, Any]:
    """refiner v5 'zkd' group (ck_e2e2.v4.v5.z_adapt; {} when off): both teacher-statistics records (v4.v5.z_stats_det
    / _map, tools/ck/v5/fit_z_stats.py) must exist and belong to the teachers of teacher_check (navsim
    ck/v5/zadapt.check_z_stats: arm, run, which, the ckpt sha16 of the teacher record, the student's z_stride; an
    unfinished record refused; strict mode also refuses a DRY-RUN record: fewer than zadapt.DRY_RUN_BELOW tokens or
    --limit) -> record {det / map: {path, sha16, n_tokens, channels}} (teachers.json 'z_stats').  A missing file is
    refused with the fit command."""
    v5 = c.v5
    if not (v5.get("enabled") and v5.get("z_adapt")):
        return {}
    from navsim.agents.para_ssr.ck.v5 import zadapt as ZA
    out: Dict[str, Any] = {}
    for name, arm, run in (("det", "T", c.teacher_det_run), ("map", "M", c.teacher_map_run)):
        p = str(v5.get(f"z_stats_{name}") or "")
        LU.no_equals(p, f"ck_e2e2.v4.v5.z_stats_{name}")
        if not Path(p).is_file():
            raise LaunchError(
                f"ck_e2e2.v4.v5.z_stats_{name}: teacher-statistics file {p!r} missing (Z adapters of teacher {run} "
                f"'{c.teacher_which}' not fitted yet): after the teacher has finished run (CPU)  CUDA_VISIBLE_DEVICES= "
                f"OMP_NUM_THREADS=8 PYTHONPATH={REPO} {PY} {REPO}/tools/ck/v5/fit_z_stats.py --run {run} --which "
                f"{c.teacher_which} --tokens 4096 --zq-exec {c.v4_net().exec_model} --out {p}   (or set v4.v5.z_adapt=false)")
        try:
            rec, info = ZA.load_z_stats(p)
            ZA.check_z_stats(rec, arm, run=run, which=c.teacher_which,
                             ckpt_sha16=(teachers.get(name) or {}).get("ckpt_sha16"), z_stride=int(v5["z_stride"]),
                             allow_dry_run=mode != "strict", where=f"ck_e2e2.v4.v5.z_stats_{name}",
                             exec_model=c.v4_net().exec_model)
        except (ValueError, FileNotFoundError) as e:
            raise LaunchError(str(e)) from None
        out[name] = dict(info, n_tokens=(rec.get("data") or {}).get("n_tokens"), dry_run=bool(rec.get("dry_run")),
                         ckpt_sha16=rec.get("ckpt_sha16"),
                         channels={k: {"mean": v.get("mean"), "std": v.get("std"), "n": v.get("n")}
                                   for k, v in (rec.get("channels") or {}).items()})
    return out


def kd_constraint_check(c) -> Dict[str, Any]:
    """refiner v4_3 constraint KD (ck_e2e2.v4.kd_constraint.enabled; {} when off): both teacher runs must have the 'range'
    readout (config.json v4), and the runtime rule online2.constraint_teacher_problems must hold on their config.json:
    the student's grid (corridor_version, ev_tbins, ev_tbin_s: the fields are distilled cell by cell), the student's
    B semantics (rg_stopped_aware) on both teachers, occ_channels' groups on the student and the DET teacher -- else
    LaunchError with the runtime message (online2.CONSTRAINT_REFUSAL + the mismatches, the text
    online2.check_constraint_teachers raises when the teachers are loaded).  rg_read is not part of the rule (the
    recorded v4_3 decision, constraint_teacher_problems): recorded per side, rg_read_differs = the teachers whose read
    is not the student's.  -> record (teachers.json 'kd_constraint')."""
    if not c.kd_constraint["enabled"]:
        return {}
    from navsim.agents.para_ssr.ck.online2 import CONSTRAINT_REFUSAL, constraint_teacher_problems
    from navsim.agents.para_ssr.ck.v4.config import V4Cfg
    sn = c.v4_net()
    want = {"corridor_version": int(c.v4["corridor_version"]), "ev_tbins": int(sn.ev_tbins),
            "ev_tbin_s": float(sn.ev_tbin_s)}
    rec: Dict[str, Any] = {"cfg": dict(c.kd_constraint), "student_grid": want,
                           "student_r2": {"rg_read": str(sn.rg_read), "rg_stopped_aware": bool(sn.rg_stopped_aware)}}
    tlist = []
    for name, run in (("det", c.teacher_det_run), ("map", c.teacher_map_run)):
        cfg = json.loads((Path(run) / "config.json").read_text())
        try:
            v4 = V4Cfg.from_dict(cfg.get("v4"))
        except ValueError as e:
            raise LaunchError(f"teacher {name} {run}: config.json v4: {e}") from None
        got = {"corridor_version": int(cfg.get("corridor_version") or 1), "ev_tbins": int(v4.ev_tbins),
               "ev_tbin_s": float(v4.ev_tbin_s)}
        if not v4.range:
            raise LaunchError(f"teacher {name} {run}: v4.readout {v4.readout!r}; ck_e2e2.v4.kd_constraint needs range "
                              f"teachers (train_ck2.py --v4-readout range, tools/ck/v4/run_teachers.sh READOUT=range)")
        tlist.append((name, run, v4, got["corridor_version"]))
        rec[name] = dict(got, readout=v4.readout, exec_model=v4.exec_model, rg_read=str(v4.rg_read),
                         rg_stopped_aware=bool(v4.rg_stopped_aware))
    bad = constraint_teacher_problems(sn, want["corridor_version"], str(c.kd_constraint["occ_channels"]), tlist)
    if bad:
        raise LaunchError(CONSTRAINT_REFUSAL + "; ".join(bad))
    rec["rg_read_differs"] = [n for n, _, v4, _ in tlist if str(v4.rg_read) != str(sn.rg_read)]
    return rec


def kd_calib_check(c, teachers: Dict[str, Any]) -> Dict[str, Any]:
    """kd_calib_record for a CKE2E2Config (its kd_calib, teacher runs / which and kd_score_keys)."""
    return kd_calib_record(c.kd_calib, teachers, c.teacher_det_run, c.teacher_map_run, c.teacher_which,
                           c.kd_score_keys)


# ----------------------------------------------------------------------------------------------- overrides
def parse_gpus2(s: str, ngpu: Optional[int] = None) -> List[str]:
    """2 or 4 GPUs of 0-3 (or exactly ngpu when given).  CK_ALLOW_GPU45=1 (user 2026-10-13: the fusion_v1 run on
    GPUs 2,3,4,5; fusion_v10 on GPUs 4,5,6,7) also admits GPUs 4,5 as a pair and 6,7 as 4,5,6,7."""
    if os.environ.get("CK_ALLOW_GPU45") == "1" or os.environ.get("CK_ALLOW_GPU4567") == "1":
        g = [x.strip() for x in str(s).split(",") if x.strip()]
        pool = tuple(LU.ALLOWED_GPUS) + EXTRA_GPUS
        bad = [x for x in g if x not in pool]
        if not g or bad or len(set(g)) != len(g):
            raise LaunchError(f"GPU list {s!r}: distinct GPUs of {','.join(pool)} (CK_ALLOW_GPU45=1)")
        if ("4" in g) != ("5" in g):
            raise LaunchError(f"GPU list {s!r}: GPUs 4 and 5 only as a pair (CK_ALLOW_GPU45=1)")
        if ("6" in g) != ("7" in g) or (("6" in g) and not ("4" in g)):
            raise LaunchError(f"GPU list {s!r}: GPUs 6 and 7 only as 4,5,6,7 (CK_ALLOW_GPU45=1)")
    else:
        g = LU.parse_gpus(s)
    ok = (len(g) == ngpu) if ngpu is not None else (len(g) in NGPU_ALLOWED)
    if not ok:
        want = ngpu if ngpu is not None else " or ".join(map(str, NGPU_ALLOWED))
        raise LaunchError(f"CK2 e2e runs on {want} GPUs of 0-3 (user recipe), got {s!r}")
    return g


def gpu_check2(gpus: Sequence[str], max_used_mib: int = 1024, used: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """= launch_util.gpu_check (every GPU <= max_used_mib) on a list already validated by parse_gpus2.  LU.gpu_check
    re-validates against GPUs 0-3 only and so refused 4,5 under CK_ALLOW_GPU45=1 whatever their memory use."""
    used = LU.nvidia_smi_used() if used is None else used
    busy = {g: used.get(g) for g in gpus if used.get(g) is None or used[g] > max_used_mib}
    if busy:
        raise LaunchError(f"GPU busy or unknown (MiB used > {max_used_mib}): {busy}")
    return {g: used[g] for g in gpus}


def counters2(ngpu: int) -> Dict[str, int]:
    """Per-GPU-count trainer / balancer counters: the global batch (128) and every per-rank micro-batch counter keep
    their meaning in samples / optimizer steps (2 GPUs: the user values; 4 GPUs: the old 4-GPU recipe)."""
    if ngpu not in NGPU_ALLOWED:
        raise LaunchError(f"unsupported GPU count {ngpu} (allowed {NGPU_ALLOWED})")
    acc = GLOBAL_BATCH // (TOKENS_PER_MB * ngpu)
    if acc * TOKENS_PER_MB * ngpu != GLOBAL_BATCH:
        raise LaunchError(f"global batch {GLOBAL_BATCH} is not divisible over {ngpu} GPUs")
    return {"accumulate": acc, "balance_warmup": BALANCE_WARMUP * NGPU // ngpu,
            "balance_interval": BALANCE_INTERVAL * NGPU // ngpu, "grad_norm_log": GRAD_NORM_LOG * NGPU // ngpu}


def scale_mb_counters(ck: Dict[str, Any], ngpu: int) -> Dict[str, Any]:
    """ck_e2e2 yaml defaults counted in per-rank micro-batches, rescaled from the 2-GPU values to ngpu ranks (applied
    BEFORE --ck-set lines, which stay absolute)."""
    if ngpu == NGPU:
        return ck
    out = json.loads(json.dumps(ck))
    for dotted in MB_COUNTERS:
        node, *path = dotted.split(".")
        parent, key = (out, node) if not path else (out[node], path[0])
        v = parent[key]
        if not isinstance(v, int) or (v * NGPU) % ngpu:
            raise LaunchError(f"ck_e2e2.{dotted}={v!r} does not rescale to {ngpu} GPUs")
        parent[key] = v * NGPU // ngpu
    return out


def recipe2(run: str, run_dir: str, ngpu: int, workers: int, max_epochs: int, wandb: str, arm: str = "main"
            ) -> List[str]:
    """launch_util.recipe (v2 r34 source) with the CK2 counters for ngpu GPUs (counters2) / names (SPEC s7-2)."""
    c = counters2(ngpu)
    rep = {
        "+trainer.params.devices": f"+trainer.params.devices={ngpu}",
        "trainer.params.strategy": "trainer.params.strategy=ddp",
        "trainer.params.accumulate_grad_batches": f"trainer.params.accumulate_grad_batches={c['accumulate']}",
        "agent.config.grad_balance_warmup_iters": f"agent.config.grad_balance_warmup_iters={c['balance_warmup']}",
        "agent.config.grad_balance_interval": f"agent.config.grad_balance_interval={c['balance_interval']}",
        "agent.config.grad_norm_log_interval": f"agent.config.grad_norm_log_interval={c['grad_norm_log']}",
        "experiment_name": f"experiment_name={run if run.startswith('ck2e2e') else 'ck2e2e_' + run}",
    }
    out, seen = [], set()
    for o in LU.recipe(run, run_dir, ngpu, workers, max_epochs, wandb):
        k = o.split("=", 1)[0]
        if k in rep:
            out.append(rep[k])
            seen.add(k)
        else:
            out.append(o)
    missing = set(rep) - seen
    if missing:
        raise LaunchError(f"launch_util.recipe no longer has {sorted(missing)}")
    if wandb != "disable":
        tags = f"[ck2e2e,{arm}]" if ngpu == NGPU else f"[ck2e2e,{arm},gpu{ngpu}]"
        out += [f"wandb.project={WANDB_PROJECT}", f"wandb.group={WANDB_GROUP}", f"wandb.tags={tags}"]
    return out


def build_overrides2(run_dir: str, gpus: Sequence[str], workers: int = 6, max_epochs: int = 30,
                     ck_sets: Sequence[str] = (), extra: Sequence[str] = (), resume_ckpt: Optional[str] = None,
                     wandb: str = "online", ck_yaml=CK2_YAML, teacher_mode: str = "strict",
                     teacher_deep: bool = True) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    """-> (ordered Hydra overrides, effective ck_e2e2 dict, teacher record).  extra overrides replace recipe entries
    of the same key (smoke: limit_train_batches, scene_filter ...)."""
    run_dir = LU.no_equals(str(Path(run_dir)), "run dir")
    if not os.path.isabs(run_dir):
        raise LaunchError(f"run dir must be absolute: {run_dir}")
    gpus = parse_gpus2(",".join(gpus))
    run = Path(run_dir).name
    ck = scale_mb_counters(load_ck2_yaml(ck_yaml), len(gpus))
    ck["io_dir"] = f"{run_dir}/{IO_SUB}"
    ck = apply_sets2(ck, ck_sets)
    check_paths(ck)
    placeholder_check(ck)
    c = ck2_from_any(ck)
    if not c.enabled:
        raise LaunchError("ck_e2e2.enabled is false")
    if teacher_mode == "strict" and not c.teacher_require_done:
        raise LaunchError("ck_e2e2.teacher_require_done=false is only for smoke runs (E2E_SMOKE=1)")
    if max_epochs <= c.onpolicy_from_epoch and teacher_mode == "strict":
        raise LaunchError(f"max_epochs {max_epochs} <= onpolicy_from_epoch {c.onpolicy_from_epoch}")
    teachers = teacher_check(c.teacher_det_run, c.teacher_map_run, c.teacher_which, teacher_mode, teacher_deep,
                             ep_target=c.ep_target, kd_calib=c.kd_calib, kd_keys=c.kd_score_keys)
    bk = bevkd_norms(c)
    if bk:
        teachers["bev_kd"] = bk
    zs = z_stats_check(c, teachers, teacher_mode)      # refiner v5 Z adapters only (off: no key)
    if zs:
        teachers["z_stats"] = zs
    kq = kd_constraint_check(c)                        # refiner v4_3 constraint KD only (off: no key)
    if kq:
        teachers["kd_constraint"] = kq
    ko = kd_objects_check(c, teachers)                 # temporal DET teacher only (refiner-v3)
    if ko:
        teachers["kd_objects"] = ko
    teachers["warmup_sources"] = warmup_check(c)
    base = recipe2(run, run_dir, len(gpus), workers, max_epochs, wandb, arm_of(ck))
    base += [f"++agent.config.ck_e2e2.{k}={LU.hydra_value(v)}" for k, v in LU.flatten(ck)]
    base.append(f"output_dir={run_dir}/train/")
    if resume_ckpt:
        base.append(f"++resume_checkpoint={LU.no_equals(resume_ckpt, 'resume checkpoint')}")
    merged: Dict[str, str] = {}
    for o in base + list(extra):
        k = LU._okey(o)
        merged[("~" + k) if o.startswith("~") else k] = o
    return list(merged.values()), ck, teachers


# ----------------------------------------------------------------------------------------------- labeler health
def labeler_health2(run_dir: str, state_path: str, stall_s: float = 1200.0, now: Optional[float] = None
                    ) -> Tuple[bool, str]:
    """= launch_util.labeler_health with the ck_e2e2 io dir (lab/status.json of labeler2)."""
    now = time.time() if now is None else float(now)
    sp = Path(run_dir) / IO_SUB / "lab/status.json"
    pidf = Path(run_dir) / "labeler/labeler.pid"
    if pidf.is_file() and now - pidf.stat().st_mtime < stall_s:
        try:
            cur = int(pidf.read_text().strip())
            if not sp.is_file() or int(json.loads(sp.read_text()).get("pid") or -1) != cur:
                return True, "labeler starting (status.json not yet from this pid)"
        except (ValueError, json.JSONDecodeError):
            return True, "labeler starting"
    if not sp.is_file():
        return True, "no status.json yet"
    age = now - sp.stat().st_mtime
    if age > stall_s:
        return False, f"lab/status.json not written for {age:.0f} s (> {stall_s:.0f}): labeler main loop hung"
    try:
        d = json.loads(sp.read_text())
    except json.JSONDecodeError:
        return True, "status.json being rewritten"
    prog = sum(int(d.get(k) or 0) for k in ("tokens_ok", "tokens_err", "tokens_skip"))
    busy = int(d.get("backlog_chunks") or 0) > 0 or int(d.get("inflight") or 0) > 0
    st_p = Path(state_path)
    try:
        st = json.loads(st_p.read_text()) if st_p.is_file() else {}
    except json.JSONDecodeError:
        st = {}
    if st.get("pid") != d.get("pid") or st.get("prog") != prog or not busy:
        LU._write_json(st_p, {"pid": d.get("pid"), "prog": prog, "t": now})
        return True, f"progress {prog}" + ("" if busy else " (idle)")
    stalled = now - float(st.get("t", now))
    if stalled > stall_s:
        return False, (f"no scored token for {stalled:.0f} s (> {stall_s:.0f}) with backlog {d.get('backlog_chunks')} "
                       f"chunks / inflight {d.get('inflight')}: pool stuck")
    return True, f"busy, no new token for {stalled:.0f} s"


# ----------------------------------------------------------------------------------------------- status
PHASES2 = {0: "warmup", 1: "warmup_record", 2: "onpolicy"}


def run_status2(run_dir: str, max_epochs: int = 30) -> Dict[str, Any]:
    R = Path(run_dir)
    IO = R / IO_SUB
    st: Dict[str, Any] = {"run_dir": str(R), "now": kst()}
    tp = LU.pid_alive(R / "launch/train.pid", "run_training")
    lp = LU.pid_alive(R / "labeler/labeler.pid", "labeler2")
    rd = lambda p: p.read_text().strip() if p.is_file() else None  # noqa: E731
    st["train"] = {"pid": tp, "alive": tp is not None, "exit": rd(R / "launch/train.exit")}
    st["labeler"] = {"pid": lp, "alive": lp is not None, "exit": rd(R / "labeler/labeler.exit")}
    wp = LU.pid_alive(R / "labeler/watchdog.pid", "_watchdog")
    rs = R / "labeler/watchdog.restarts"
    st["watchdog"] = {"pid": wp, "alive": wp is not None,
                      "restarts": len(rs.read_text().splitlines()) if rs.is_file() else 0}
    st["labeler_disabled"] = not (R / "labeler").is_dir()
    st["train_done"] = (IO / "TRAIN_DONE").exists()
    eff = R / "launch/ck_e2e2_effective.json"
    if eff.is_file():
        try:
            e = json.loads(eff.read_text())
            max_epochs = int(e.get("_trainer_max_epochs", max_epochs))
            st["arm"] = arm_of(e)
        except Exception:       # noqa: BLE001
            pass
    recs = LU.read_jsonl(IO / "steps_rank0.jsonl", tail_bytes=8 << 20)
    if recs:
        last = recs[-1]
        sec = LU._median([LU._num(r.get("sec_step")) for r in recs[-300:]])
        ep, frac = LU.rec_epoch(last), LU.rec_frac(last)
        ph = LU._num(last.get("ck2/phase"))
        st["steps"] = {"n_tail": len(recs), "epoch": ep, "epoch_frac": frac,
                       "phase": PHASES2.get(int(ph)) if ph is not None else None, "sec_step_median": sec,
                       "mem_gb": last.get("mem_gb"), "ck2_loss": last.get("ck2/loss"),
                       "G_src_fallback": last.get("ck2/G_src_fallback"), "w_lat": last.get("ck2/w_lat")}
        lam = {t: last.get(f"bevkd/{t}/lam") for t in bevkd_teachers(recs[-50:])}
        if lam:
            st["steps"]["bevkd_lam"] = lam
        gs = [LU._num(r.get("ck2/gshare")) for r in recs if "ck2/gshare" in r][-20:]
        gs = [g for g in gs if g is not None and math.isfinite(g)]
        if gs:
            st["gshare"] = {"n": len(gs), "mean": sum(gs) / len(gs), "max": max(gs),
                            "warn": bool(sum(gs) / len(gs) > GSHARE_WARN)}
        ws = next((int(r["world_size"]) for r in reversed(LU.read_jsonl(IO / "epochs.jsonl")) if r.get("world_size")),
                  NGPU)
        mb_ep = LU.mb_per_epoch(ws)
        if sec and frac is not None and st["train"]["alive"]:
            remain = max(0.0, max_epochs - frac) * mb_ep * sec
            st["eta"] = {"hours_left": round(remain / 3600, 2), "finish": kst(time.time() + remain),
                         "basis": f"median sec_step of the last 300 mb x {mb_ep} mb/epoch x "
                                  f"{max(0.0, max_epochs - frac):.2f} epochs left [추정]"}
    ep_recs = LU.read_jsonl(IO / "epochs.jsonl")
    if ep_recs:
        st["last_epoch_record"] = ep_recs[-1]
        es = [r for r in ep_recs if r.get("event") == "epoch_start"]
        if es:
            st["last_epoch_start"] = {k: es[-1].get(k) for k in ("epoch", "phase", "label_warning")}
    st["checkpoints"] = [p.name for _, p in LU.epoch_ckpts(str(R))]
    s = IO / "lab/status.json"
    if s.is_file():
        try:
            st["labeler_status"] = json.loads(s.read_text())
        except json.JSONDecodeError:
            st["labeler_status"] = "unreadable"
    lab = IO / "lab"
    st["generations_done"] = sorted(int(p.parent.name[2:]) for p in lab.glob("ep*/DONE.json")) if lab.is_dir() else []
    w = []
    if st["train"]["alive"] and not st["labeler"]["alive"] and not st["labeler_disabled"]:
        w.append(f"labeler not running while training is alive (exit {st['labeler']['exit']}); restart with "
                 f"'RUN={R.name} bash tools/ck/e2e2/train_e2e2.sh labeler --gpus <same>'")
    lw = (st.get("last_epoch_start") or {}).get("label_warning") or {}
    if lw.get("warn"):
        w.append(f"epoch {st['last_epoch_start'].get('epoch')} label supply: {lw.get('msg')}")
    if (st.get("gshare") or {}).get("warn"):
        w.append(f"CK2 share of the BEV gradient mean {st['gshare']['mean']:.2f} > {GSHARE_WARN} (last "
                 f"{st['gshare']['n']} records)")
    st["warnings"] = w
    return st


def print_status2(st: Dict[str, Any]) -> None:
    print(f"[{st['now']}] {st['run_dir']} (arm {st.get('arm', '?')})")
    t, lb, wd = st["train"], st["labeler"], st.get("watchdog") or {}
    print(f"  train   : {'alive pid ' + str(t['pid']) if t['alive'] else 'not running'}"
          f"{'  exit ' + t['exit'] if t['exit'] is not None else ''}{'  TRAIN_DONE' if st['train_done'] else ''}")
    print(f"  labeler : {'alive pid ' + str(lb['pid']) if lb['alive'] else 'not running'}"
          f"{'  exit ' + lb['exit'] if lb['exit'] is not None else ''}  watchdog "
          f"{'alive' if wd.get('alive') else 'not running'} (restarts {wd.get('restarts', 0)})")
    if "steps" in st:
        print(f"  steps   : {json.dumps(st['steps'])}")
    if "eta" in st:
        print(f"  ETA     : {st['eta']['hours_left']} h -> {st['eta']['finish']} ({st['eta']['basis']})")
    if "gshare" in st:
        print(f"  gshare  : {json.dumps(st['gshare'])}")
    if "last_epoch_record" in st:
        print(f"  epoch   : {json.dumps(st['last_epoch_record'])[:300]}")
    print(f"  ckpts   : {len(st['checkpoints'])} {st['checkpoints'][-3:]}")
    if "labeler_status" in st:
        print(f"  labels  : {json.dumps(st['labeler_status'])[:400]}")
    print(f"  gen DONE: {st['generations_done']}")
    for w in st.get("warnings") or []:
        print(f"  WARNING : {w}")


# ----------------------------------------------------------------------------------------------- GPU smoke check
def bevkd_teachers(recs: Sequence[Dict[str, Any]]) -> List[str]:
    """BEV-KD teachers present in step records (keys bevkd/<t>/lam), in bev_kd.TEACHERS order."""
    seen = {k.split("/")[1] for r in recs for k in r if k.startswith("bevkd/") and k.endswith("/lam")
            and k.count("/") == 2}
    return [t for t in ("det", "map") if t in seen] + sorted(seen - {"det", "map"})


def bevkd_summary(recs: Sequence[Dict[str, Any]], start_mb: Optional[int] = None) -> Dict[str, Any]:
    """per BEV-KD teacher over step records: lam max / last, last 10 ratio_now (lam_t g_t / g_plan at the measurement),
    mean ok_frac (masked tokens), last g_kd_unit, mean weighted BEV-gradient norm (gnorm/bev_bevkd_<t>) and its ratio to
    gnorm/bev_v2 (records at mb >= start_mb when given)."""
    out: Dict[str, Any] = {}
    late = [r for r in recs if start_mb is None or (LU._num(r.get("mb")) or 0) >= start_mb]
    for t in bevkd_teachers(recs):
        lam = [LU._num(r.get(f"bevkd/{t}/lam")) for r in recs if f"bevkd/{t}/lam" in r]
        lam = [v for v in lam if v is not None]
        rn = [LU._num(r.get(f"bevkd/{t}/ratio_now")) for r in recs if f"bevkd/{t}/ratio_now" in r]
        g = [LU._num(r.get(f"bevkd/{t}/g_kd_unit")) for r in recs if f"bevkd/{t}/g_kd_unit" in r]
        gk = [(LU._num(r.get(f"gnorm/bev_bevkd_{t}")), LU._num(r.get("gnorm/bev_v2"))) for r in late
              if f"gnorm/bev_bevkd_{t}" in r]
        gk = [(a, b) for a, b in gk if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
        out[t] = {"lam_max": max(lam) if lam else None, "lam_last": lam[-1] if lam else None,
                  "ratio_now_last10": rn[-10:], "g_kd_unit_last": g[-1] if g else None,
                  "ok_frac_mean": LU._mean([LU._num(r.get(f"bevkd/{t}/ok_frac")) for r in recs
                                            if f"bevkd/{t}/ok_frac" in r]),
                  "gnorm_bev_mean": LU._mean([a for a, _ in gk]) if gk else None,
                  "gnorm_bev_over_v2_mean": LU._mean([a / b for a, b in gk if b > 0]) if gk else None}
    return out


def phase_timing2(recs: Sequence[Dict[str, Any]], skip_first: int = 20) -> Dict[str, Any]:
    """per CK2 phase (ck2/phase), records after the first skip_first of every epoch: median sec_step, mean wall
    (sec_step + sec_wait), peak mem, median variants / teacher / student ms."""
    by_ep: Dict[int, List[Dict[str, Any]]] = {}
    for r in recs:
        e = LU.rec_epoch(r)
        by_ep.setdefault(-1 if e is None else e, []).append(r)
    acc: Dict[str, Dict[str, list]] = {}
    for _, rs in sorted(by_ep.items()):
        for r in rs[skip_first:]:
            ph = LU._num(r.get("ck2/phase"))
            if ph is None:
                continue
            d = acc.setdefault(PHASES2.get(int(ph), str(int(ph))), {k: [] for k in ("step", "wall", "mem", "var",
                                                                                     "teach", "stud")})
            st, wt = LU._num(r.get("sec_step")), LU._num(r.get("sec_wait")) or 0.0
            if st is not None:
                d["step"].append(st)
                d["wall"].append(st + wt)
            d["mem"].append(LU._num(r.get("mem_gb")) or 0.0)
            d["var"].append(LU._num(r.get("ck2/variants_ms")))
            d["teach"].append(LU._num(r.get("ck2/teacher_ms")))
            d["stud"].append(LU._num(r.get("ck2/student_ms")))
    return {k: {"n": len(d["step"]), "sec_step_median": LU._median(d["step"]), "wall_per_mb_mean": LU._mean(d["wall"]),
                "mem_gb_max": max(d["mem"]) if d["mem"] else None, "variants_ms_median": LU._median(d["var"]),
                "teacher_ms_median": LU._median(d["teach"]), "student_ms_median": LU._median(d["stud"])}
            for k, d in acc.items()}


def estimate_hours2(timing: Dict[str, Any], world_size: int = NGPU, max_epochs: int = 30, record_from: int = 4,
                    onpolicy_from: int = 5, key: str = "wall_per_mb_mean") -> Dict[str, Any]:
    """30-epoch wall time from the measured steady s/mb per phase (epoch fixed costs not included) [추정]."""
    mb = LU.mb_per_epoch(world_size)
    rows, tot = [], 0.0
    for ph in ("warmup", "warmup_record", "onpolicy"):
        n_ep = sum(1 for e in range(max_epochs) if (("onpolicy" if e >= onpolicy_from else "warmup_record"
                                                     if e >= record_from else "warmup") == ph))
        spm = (timing.get(ph) or {}).get(key)
        if spm is None:
            return {"error": f"no {ph} timing"}
        rows.append({"phase": ph, "epochs": n_ep, "s_per_mb": spm, "hours": n_ep * mb * spm / 3600})
        tot += n_ep * mb * spm
    return {"mb_per_epoch": mb, "rows": rows, "total_hours": tot / 3600, "basis": key + " (no epoch overhead)"}


def gen_roundtrip2(io: Path, epochs: Sequence[int]) -> Dict[int, Any]:
    import numpy as np
    from navsim.agents.para_ssr.ck import e2e_data as ED
    from navsim.agents.para_ssr.ck import e2e_data2 as D2
    out: Dict[int, Any] = {}
    for e in epochs:
        try:
            meta = json.loads((ED.gen_dir(str(io), e) / "meta.json").read_text())
            g = D2.open_generation2(str(io), e, int(meta["n_rows"]), mode="r")
        except Exception as ex:      # noqa: BLE001
            out[e] = {"error": f"{type(ex).__name__}: {ex}"}
            continue
        # every record of a row (a resumed epoch leaves the dead attempt's chunks too, and the DistributedSampler pads
        # a token onto both ranks): the labeler keeps the FIRST record it scored, which need not be the first chunk
        # in listing order -> a labelled row must equal ANY of its records (= launch_util.gen_roundtrip)
        rec: Dict[int, List[Any]] = {}
        for c in ED.list_rec_chunks(str(io), e):
            d = D2.read_rec2_chunk(c["path"])
            for i, r in enumerate(d["row"]):
                rec.setdefault(int(r), []).append(np.nan_to_num(d["traj"][i], nan=-9.0))
        done = np.flatnonzero(np.asarray(g["row_state"]) == 1)
        bad = sum(1 for r in done if not any(np.array_equal(np.nan_to_num(np.asarray(g["traj"][r]), nan=-9.0), x)
                                             for x in rec.get(int(r), [])))
        out[e] = {"rows_recorded": len(rec), "rows_labelled": int(len(done)), "mismatch": int(bad),
                  "labelled_of_recorded": len(set(map(int, done)) & set(rec)) / max(len(rec), 1),
                  "cand_ok_frac": float(np.asarray(g["cand_ok"])[done].mean()) if len(done) else None}
    return out


def temporal_smoke_checks(R: Path, recs: Sequence[Dict[str, Any]], min_obj_ok: float = 0.999) -> Dict[str, Any]:
    """Smoke gate of the refiner-v3 temporal DET teacher (review finding 2): {} unless the run's effective
    teacher_det_run (launch/ck_e2e2_effective.json) has temporal.on.  Then teachers.json must record det.temporal
    (non-privileged, branch / forecaster sha16, no warnings) and kd_objects.same_as_teacher_training, and the steps must
    log ck2/kd_obj_ok_frac (= the temporal KD path ran) with mean >= min_obj_ok (the train pack has no obj_ok False
    row; tokens without a packed row are bounded by strict_rows)."""
    eff = R / "launch/ck_e2e2_effective.json"
    try:
        det = json.loads(eff.read_text()).get("teacher_det_run") if eff.is_file() else None
    except Exception:       # noqa: BLE001
        det = None
    try:
        from navsim.agents.para_ssr.ck import e2e_data2 as D2
        on = bool(det) and D2.teacher_temporal(det).on
    except Exception as e:  # noqa: BLE001  (an unreadable temporal cfg must fail the gate, not skip it)
        return {"_info": {"teacher_det_run": det, "error": f"{type(e).__name__}: {e}"}, "temporal_cfg_readable": False}
    if not on:
        return {}
    tj = R / "launch/teachers.json"
    try:
        tch = json.loads(tj.read_text()) if tj.is_file() else {}
    except Exception:       # noqa: BLE001
        tch = {}
    tr = (tch.get("det") or {}).get("temporal") or {}
    ko = tch.get("kd_objects") or {}
    vals = [LU._num(r.get("ck2/kd_obj_ok_frac")) for r in recs if "ck2/kd_obj_ok_frac" in r]
    vals = [v for v in vals if v is not None]
    info = {"teacher_det_run": det, "kd_obj_ok_frac_n": len(vals), "kd_obj_ok_frac_mean": LU._mean(vals),
            "kd_obj_ok_frac_min": min(vals) if vals else None, "kd_objects_build_sha256": ko.get("build_sha256"),
            "forecaster_state_sha16": tr.get("forecaster_state_sha16"), "warnings": tr.get("warnings")}
    return {"_info": info,
            "temporal_teacher_recorded": bool(tr) and (tr.get("cfg") or {}).get("on") is True
            and tr.get("privileged") is False and bool(tr.get("branch_state_sha16"))
            and bool(tr.get("forecaster_state_sha16")) and not tr.get("warnings"),
            "kd_objects_same_as_teacher_training": ko.get("same_as_teacher_training") is True,
            "kd_obj_ok_frac_logged_ge_0999": bool(vals) and (LU._mean(vals) or 0.0) >= min_obj_ok}


def smoke_check2(run_dir: str, ngpu: int = NGPU, onpolicy_epochs: Sequence[int] = (2,),
                 gen_epochs: Sequence[int] = (1,)) -> Dict[str, Any]:
    R = Path(run_dir)
    IO = R / IO_SUB
    recs = LU.read_jsonl(IO / "steps_rank0.jsonl")
    res: Dict[str, Any] = {"run_dir": str(R), "checked": kst(), "n_records": len(recs)}
    res["phase_timing"] = phase_timing2(recs)
    nonfin = sum(1 for r in recs for k, v in r.items() if k.startswith(("ck2/", "bevkd/", "gnorm/"))
                 and isinstance(v, float) and not math.isfinite(v) and not k.endswith("ratio_now"))
    res["nonfinite"] = nonfin
    res["skipped_steps"] = max([int(LU._num(r.get("skipped_steps")) or 0) for r in recs], default=0)
    onp = [r for r in recs if LU.rec_epoch(r) in set(onpolicy_epochs)]
    res["onpolicy_G_src"] = {k: LU._mean([LU._num(r.get(f"ck2/G_src_{k}")) for r in onp])
                             for k in ("fallback", "prev", "older", "nodata")}
    res["kd_ok_frac"] = LU._mean([LU._num(r.get("ck2/kd_ok_frac")) for r in recs])
    res["gshare_mean"] = LU._mean([LU._num(r.get("ck2/gshare")) for r in recs if "ck2/gshare" in r])
    res["e_abs_T_mean"] = LU._mean([LU._num(r.get("ck2/e_abs_T")) for r in recs])
    arm, bk_cfg = None, {}
    eff = R / "launch/ck_e2e2_effective.json"
    if eff.is_file():
        try:
            e = json.loads(eff.read_text())
            arm, bk_cfg = arm_of(e), dict(e.get("bev_kd") or {})
        except Exception:       # noqa: BLE001
            pass
    res["arm"] = arm
    res["bevkd"] = bevkd_summary(recs, bk_cfg.get("start_mb"))
    res["gen_roundtrip"] = gen_roundtrip2(IO, gen_epochs)
    s = IO / "lab/status.json"
    res["labeler_status"] = json.loads(s.read_text()) if s.is_file() else None
    res["train_exit"] = LU._exit_code(R / "launch/train.exit")
    res["estimate_30ep_2gpu"] = estimate_hours2(res["phase_timing"], ngpu)
    mem = max([(v or {}).get("mem_gb_max") or 0.0 for v in res["phase_timing"].values()], default=0.0)
    checks = {"train_exit_0": res["train_exit"] == 0, "nonfinite_0": nonfin == 0,
              "skipped_steps_0": res["skipped_steps"] == 0,
              "phases_all": all(k in res["phase_timing"] for k in ("warmup", "warmup_record", "onpolicy")),
              "mem_le_22gb": 0 < mem <= 22.0,
              "onpolicy_labels_from_prev_ge_90pct": (res["onpolicy_G_src"]["prev"] or 0.0) >= 0.9,
              "gen_roundtrip_bitexact": all(isinstance(v, dict) and "error" not in v and v["rows_labelled"] > 0
                                            and v["mismatch"] == 0 for v in res["gen_roundtrip"].values())}
    if arm == "bevkd":
        want = [str(t) for t in (bk_cfg.get("teachers") or [])]
        checks["bevkd_teachers_logged"] = bool(want) and sorted(res["bevkd"]) == sorted(want)
        checks["bevkd_lam_pos_every_teacher"] = bool(want) and all(
            ((res["bevkd"].get(t) or {}).get("lam_max") or 0.0) > 0.0 for t in want)
    tchk = temporal_smoke_checks(R, recs)               # refiner-v3 temporal DET teacher only (CK2: {} -> unchanged)
    if tchk:
        res["temporal"] = tchk.pop("_info")
        checks.update(tchk)
    res["checks"] = checks
    res["pass"] = all(checks.values())
    return res


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overrides")
    o.add_argument("--run-dir", required=True)
    o.add_argument("--gpus", required=True)
    o.add_argument("--workers", type=int, default=6)
    o.add_argument("--max-epochs", type=int, default=30)
    o.add_argument("--ck-set", action="append", default=[])
    o.add_argument("--ck-set-file", action="append", default=[])
    o.add_argument("--extra-file", default="")
    o.add_argument("--resume-ckpt", default="")
    o.add_argument("--wandb", default="online", choices=["online", "offline", "disable"])
    o.add_argument("--teacher-check", default="strict", choices=["strict", "smoke"])
    o.add_argument("--no-deep", action="store_true", help="skip the lon-head zero check (tests)")
    o.add_argument("--out", required=True)
    t = sub.add_parser("teacher-check")
    t.add_argument("--det", required=True)
    t.add_argument("--map", required=True)
    t.add_argument("--which", default="last")
    t.add_argument("--mode", default="strict", choices=["strict", "smoke"])
    t.add_argument("--ep-target", default=None, choices=["official", "decoupled"],
                   help="the student's ck_e2e2.ep_target (default: the CKE2E2Config / ck_e2e2.yaml default)")
    t.add_argument("--kd-calib-det", default="", help="KD calibration file of the DET teacher (checked when given)")
    t.add_argument("--kd-calib-map", default="", help="KD calibration file of the MAP teacher (checked when given)")
    t.add_argument("--out", default="")
    g = sub.add_parser("gpu-check")
    g.add_argument("--gpus", required=True)
    g.add_argument("--max-used-mib", type=int, default=1024)
    g.add_argument("--list-only", action="store_true")
    lh = sub.add_parser("labeler-health")
    lh.add_argument("--run-dir", required=True)
    lh.add_argument("--state", required=True)
    lh.add_argument("--stall-s", type=float, default=1200.0)
    s = sub.add_parser("status")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--json", action="store_true")
    sc = sub.add_parser("smoke-check2")
    sc.add_argument("--run-dir", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--ngpu", type=int, default=NGPU)
    sc.add_argument("--onpolicy-epochs", default="2")
    sc.add_argument("--gen-epochs", default="1")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "overrides":
            sets = list(a.ck_set)
            for f in a.ck_set_file:
                sets += LU.read_lines(f)
            ov, ck, teachers = build_overrides2(a.run_dir, parse_gpus2(a.gpus), a.workers, a.max_epochs, sets,
                                                LU.read_lines(a.extra_file), a.resume_ckpt or None, a.wandb,
                                                teacher_mode=a.teacher_check, teacher_deep=not a.no_deep)
            outp = Path(a.out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            for o_ in ov:
                if "\n" in o_:
                    raise LaunchError(f"newline in override {o_!r}")
            outp.write_text("\n".join(ov) + "\n")
            eff = dict(ck)
            m = [x for x in ov if LU._okey(x) == "trainer.params.max_epochs"]
            eff["_trainer_max_epochs"] = int(m[-1].split("=", 1)[1]) if m else a.max_epochs
            eff["_arm"] = arm_of(ck)
            eff["_ck_sets"] = sets
            LU._write_json(outp.parent / "ck_e2e2_effective.json", eff)
            LU._write_json(outp.parent / "teachers.json", teachers)
        elif a.cmd == "teacher-check":
            kc = ({"enabled": True, "det": a.kd_calib_det or None, "map": a.kd_calib_map or None}
                  if (a.kd_calib_det or a.kd_calib_map) else None)
            rec = teacher_check(a.det, a.map, a.which, a.mode, ep_target=a.ep_target, kd_calib=kc)
            if a.out:
                LU._write_json(a.out, rec)
            print(json.dumps(rec, indent=1, default=str))
        elif a.cmd == "gpu-check":
            g2 = parse_gpus2(a.gpus)
            print(",".join(g2) if a.list_only else json.dumps(gpu_check2(g2, a.max_used_mib)))
        elif a.cmd == "labeler-health":
            ok, why = labeler_health2(a.run_dir, a.state, a.stall_s)
            print(why)
            return 0 if ok else 5
        elif a.cmd == "status":
            st = run_status2(a.run_dir)
            print(json.dumps(st, indent=1, default=str)) if a.json else print_status2(st)
        elif a.cmd == "smoke-check2":
            res = smoke_check2(a.run_dir, a.ngpu, [int(x) for x in a.onpolicy_epochs.split(",") if x],
                               [int(x) for x in a.gen_epochs.split(",") if x])
            LU._write_json(a.out, res)
            print(json.dumps({"pass": res["pass"], "checks": res["checks"], "phase_timing": res["phase_timing"],
                              "estimate_30ep_2gpu": res["estimate_30ep_2gpu"]}, indent=1, default=str))
            return 0 if res["pass"] else 4
    except LaunchError as e:
        print(f"[launch_util2] REFUSED: {e.msg}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
