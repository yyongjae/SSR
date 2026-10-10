#!/usr/bin/env python
"""CK2 e2e launch helpers (SPEC ck2e2e s7).  Used by train_e2e2.sh / smoke_e2e2.sh; importable for tests.
tools/ck/e2e/launch_util.py is imported unchanged (recipe, hydra_value, flatten, apply_sets, read_lines, GPU / pid / ckpt
helpers, W&B mode); this file adds the CK2 recipe, the ck_e2e2 dict, the teacher guards and the ck_e2e2 io-dir status.

  launch_util2.py overrides  --run-dir R --gpus 0,1 [--workers 6] [--max-epochs 30] [--ck-set k=v ...] [--ck-set-file F]
                             [--extra-file F] [--resume-ckpt P] [--wandb online|offline|disable]
                             [--teacher-check strict|smoke] --out R/launch/overrides.txt
  launch_util2.py teacher-check --det RUN --map RUN [--mode strict|smoke] [--ep-target decoupled|official] [--out F.json]
                                [--kd-calib-det F.json] [--kd-calib-map F.json]            exit 3 when not usable
  launch_util2.py gpu-check  --gpus 0,1 [--max-used-mib 1024] [--list-only]               exactly 2 GPUs of 0-3
  launch_util2.py labeler-health --run-dir R --state F [--stall-s 1200]                   exit 5 when the labeler hangs
  launch_util2.py status     --run-dir R [--json]
  launch_util2.py smoke-check2 --run-dir R --out F [--ngpu 2] [--onpolicy-epochs 2] [--gen-epochs 1]   GPU smoke summary
  (find-last / wandb-mode: tools/ck/e2e/launch_util.py)

Recipe (user): v2 r34 source on 2 GPUs = the old 4-GPU recipe with devices 2, batch 4 / GPU, accumulate_grad_batches 16
(global 128), grad_balance_warmup_iters 10600, grad_balance_interval 200, grad_norm_log_interval 200; experiment_name
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
CK_DATA = "/home/external-user/ssd/yongjae_refiner/ck"
E2E_ROOT = f"{CK_DATA}/ck2/e2e"
SMOKE_ROOT = f"{CK_DATA}/ck2/e2e_smoke"
IO_SUB = "ck_e2e2"
NGPU = 2                          # user: 2 GPUs per run (devices 2)
ACCUMULATE = 16                   # user: 2 x 4 x 16 = 128
BALANCE_WARMUP = 10600            # user
BALANCE_INTERVAL = 200            # user
GRAD_NORM_LOG = 200               # user
TOKENS_PER_MB = 4
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


def kd_calib_check(c, teachers: Dict[str, Any]) -> Dict[str, Any]:
    """kd_calib_record for a CKE2E2Config (its kd_calib, teacher runs / which and kd_score_keys)."""
    return kd_calib_record(c.kd_calib, teachers, c.teacher_det_run, c.teacher_map_run, c.teacher_which,
                           c.kd_score_keys)


# ----------------------------------------------------------------------------------------------- overrides
def parse_gpus2(s: str, ngpu: int = NGPU) -> List[str]:
    g = LU.parse_gpus(s)
    if len(g) != ngpu:
        raise LaunchError(f"CK2 e2e runs on exactly {ngpu} GPUs of 0-3 (devices {ngpu}, user recipe), got {s!r}")
    return g


def recipe2(run: str, run_dir: str, ngpu: int, workers: int, max_epochs: int, wandb: str, arm: str = "main"
            ) -> List[str]:
    """launch_util.recipe (v2 r34 source) with the CK2 2-GPU counters / names (SPEC s7-2)."""
    rep = {
        "+trainer.params.devices": f"+trainer.params.devices={ngpu}",
        "trainer.params.strategy": "trainer.params.strategy=ddp",
        "trainer.params.accumulate_grad_batches": f"trainer.params.accumulate_grad_batches={ACCUMULATE}",
        "agent.config.grad_balance_warmup_iters": f"agent.config.grad_balance_warmup_iters={BALANCE_WARMUP}",
        "agent.config.grad_balance_interval": f"agent.config.grad_balance_interval={BALANCE_INTERVAL}",
        "agent.config.grad_norm_log_interval": f"agent.config.grad_norm_log_interval={GRAD_NORM_LOG}",
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
        out += [f"wandb.project={WANDB_PROJECT}", f"wandb.group={WANDB_GROUP}", f"wandb.tags=[ck2e2e,{arm}]"]
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
    ck = load_ck2_yaml(ck_yaml)
    ck["io_dir"] = f"{run_dir}/{IO_SUB}"
    ck = LU.apply_sets(ck, ck_sets)
    check_paths(ck)
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
            print(",".join(g2) if a.list_only else json.dumps(LU.gpu_check(g2, a.max_used_mib)))
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
