#!/usr/bin/env python
"""Checks of a finished GPU smoke (scripts/experiment.sh smoke): 2 epochs x 60 micro-batches of one composed run.  Port
of scripts/training/smoke_v2_final.sh and smoke_v2_nokd.sh; what is expected follows the run's settings.json, so it also
covers refiner off, refiner KD off or partial, BEV KD off, and one teacher off.

  smoke_check.py --run-dir <run> [--mem-limit-gb 22.5]

Writes <run>/launch/smoke_check.json, prints each check, exit 1 when one fails.  Peak memory: torch's peak per GPU from
steps_rank0.jsonl when the refiner writes one; with the refiner off, the peak of nvidia-smi's memory.used sampled by
scripts/experiment.sh into launch/gpu_mem.csv, which counts reserved memory too, so it reads higher.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BANKS = {"det": "bevfusion", "map": "resmap"}
KD_KEYS_OF = {"det": ("nc", "ep", "ttc"), "map": ("dac", "ep")}
KD_SCORE_KEYS_ALL = ("nc", "dac", "ep", "ttc")     # ck_e2e2.yaml kd_score_keys


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()] if path.is_file() else []


def _gpu_mem_peak_gb(path: Path) -> Optional[float]:
    vals = []
    if path.is_file():
        for ln in path.read_text().splitlines():
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) == 2 and parts[1].replace(".", "", 1).isdigit():
                vals.append(float(parts[1]) / 1024.0)
    return max(vals) if vals else None


def _events(run: Path) -> Optional[Dict[str, List[float]]]:
    """{scalar tag: values} over the run's TensorBoard files; None when tensorboard cannot be imported or no file."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    files = sorted(glob.glob(str(run / "train/lightning_logs/version_*/events.out.tfevents.*")))
    if not files:
        return None
    out: Dict[str, List[float]] = {}
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            out.setdefault(tag, []).extend(s.value for s in ea.Scalars(tag))
    return out


def _ckpt_keys(path: str) -> List[str]:
    import torch
    return list(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])


def _calib_checks(run: Path, s: Dict[str, str], steps, teachers, checks, info) -> None:
    """The online KD calibration of smoke_v2_final.sh, on the keys of this arm."""
    kc = teachers.get("kd_calib") or {}
    checks["kd_calib_mode_online"] = kc.get("mode") == "online"
    kd_keys = list(KD_SCORE_KEYS_ALL)
    src = [r for r, k in (("det", "DET_KD"), ("map", "MAP_KD")) if s[k] not in ("off", "off_ref")]
    if len(src) == 1:
        kd_keys = list(KD_KEYS_OF[src[0]])
    keys = [f"{t}/{k}" for t in ("det", "map") for k in KD_KEYS_OF[t] if k in kd_keys]
    cal = read_jsonl(run / "refiner/kd_calib_online.jsonl")
    wu = [c for c in cal if c.get("window_phase") == "warmup_record" and "keys" in c]
    checks["kd_calib_ge_5_warmup_refits"] = len(wu) >= 5
    good = True
    for c in wu:
        for k in keys:
            r = c["keys"].get(k) or {}
            if not (r.get("fitted") and r.get("a", 0) > 0 and math.isfinite(r.get("a", math.nan))
                    and math.isfinite(r.get("b", math.nan))):
                good = False
    checks["kd_calib_every_key_fitted"] = good and bool(wu)
    checks["kd_calib_onpolicy_window"] = any(c.get("window_phase") == "onpolicy" for c in cal)
    checks["kd_calib_in_step_log"] = any(k.startswith("ck2/cal_") for r in steps for k in r)
    info["kd_calib_keys"] = keys
    info["kd_calib_warmup_refits"] = len(wu)


def _refiner_checks(run: Path, s: Dict[str, str], checks: Dict[str, bool], info: Dict[str, Any]) -> List[Dict[str, Any]]:
    io = run / "refiner"
    ep = read_jsonl(io / "epochs.jsonl")
    ends = [e for e in ep if e.get("event") == "epoch_end"]
    phases = [e.get("phase") for e in ep if e.get("event") == "epoch_start"]
    checks["two_epochs_warmup_record_onpolicy"] = len(ends) == 2 and phases == ["warmup_record", "onpolicy"]
    cum = ends[-1]["cum"] if ends else {}
    checks["no_nonfinite_loss"] = bool(ends) and cum.get("nonfinite_loss") == 0 and cum.get("sur_nonfinite") == 0
    st = read_jsonl(io / "steps_rank0.jsonl")
    bad = sorted({k for r in st for k, v in r.items() if isinstance(v, float) and not math.isfinite(v)})
    checks["no_nonfinite_step_value"] = bool(st) and not bad
    info["nonfinite_step_keys"] = bad[:10]
    checks["no_skipped_step"] = bool(st) and max(int(r.get("skipped_steps") or 0) for r in st) == 0
    lab_p = io / "lab/status.json"
    lab = json.loads(lab_p.read_text()) if lab_p.is_file() else {}
    g0 = (lab.get("epochs") or {}).get("0") or {}
    checks["labeler_generation_0_done"] = (lab.get("tokens_ok", 0) > 0 and lab.get("tokens_err", 1) == 0 and bool(g0.get("done"))
                                           and g0.get("rows_done", 0) > 0)
    tj_p = run / "launch/teachers.json"
    teachers = json.loads(tj_p.read_text()) if tj_p.is_file() else {}
    wt_kd = [float(r.get("ck2/wt_kd_score", 0.0)) for r in st]
    wt_lat = [float(r.get("ck2/wt_lat_kd", 0.0)) for r in st]
    teacher_ms = [float(r["ck2/teacher_ms"]) for r in st if "ck2/teacher_ms" in r]
    rkd = s["REFINER_KD"]
    if rkd == "off":
        checks["no_teacher_record"] = teachers.get("mode") == "none" and "det" not in teachers and "map" not in teachers
        checks["no_teacher_log"] = not any("ck2/e_abs_T" in r for r in st)
        checks["no_teacher_time"] = bool(teacher_ms) and max(teacher_ms) < 5.0
    else:
        checks["both_teachers_recorded"] = "det" in teachers and "map" in teachers
        ok = [float(r["ck2/kd_ok_frac"]) for r in st if "ck2/kd_ok_frac" in r]
        checks["kd_ok_frac_ge_0.999"] = bool(ok) and sum(ok) / len(ok) >= 0.999
        info["kd_ok_frac_mean"] = sum(ok) / len(ok) if ok else None
    # each KD term is on exactly when the settings say so
    checks["score_kd_as_set"] = (max(wt_kd, default=0.0) > 0.0) if rkd in ("on", "score") else (max(wt_kd, default=0.0) == 0.0)
    checks["lateral_kd_as_set"] = (max(wt_lat, default=0.0) > 0.0) if rkd in ("on", "lat") else (max(wt_lat, default=0.0) == 0.0)
    if rkd in ("on", "score"):
        _calib_checks(run, s, st, teachers, checks, info)
    else:
        checks["kd_calib_off_as_set"] = not (teachers.get("kd_calib") or {}).get("enabled")
    return st


def check_run(run_dir: str, mem_limit_gb: float = 22.5) -> Dict[str, Any]:
    run = Path(run_dir)
    settings = json.loads((run / "launch/settings.json").read_text())
    s = settings["settings"]
    checks: Dict[str, bool] = {}
    info: Dict[str, Any] = {"run": settings.get("run"), "mem_limit_gb": mem_limit_gb}
    ex = run / "launch/train.exit"
    checks["train_exit_0"] = ex.is_file() and ex.read_text().strip() == "0"
    tt = run / "train/train_tokens.json"
    checks["train_tokens_recorded"] = tt.is_file() and json.loads(tt.read_text()).get("count", 0) > 0
    tm = run / "train/train_time.json"
    sess = (json.loads(tm.read_text()).get("sessions") or [{}])[-1] if tm.is_file() else {}
    checks["two_epochs_completed"] = sess.get("status") == "completed" and sess.get("epochs_completed") == 2
    last = sorted(glob.glob(str(run / "train/lightning_logs/version_*/checkpoints/last.ckpt")))
    checks["last_ckpt"] = bool(last)
    checks["epoch_1_ckpt"] = bool(glob.glob(str(run / "train/lightning_logs/version_*/checkpoints/epoch=1-*.ckpt")))

    steps: List[Dict[str, Any]] = []
    if s["REFINER"] == "on":
        steps = _refiner_checks(run, s, checks, info)
    else:
        checks["no_refiner_files"] = not (run / "refiner/steps_rank0.jsonl").exists()

    # memory
    torch_peak = max((float(r.get("mem_gb", 0.0)) for r in steps), default=0.0) or None
    smi_peak = _gpu_mem_peak_gb(run / "launch/gpu_mem.csv")
    peak = torch_peak if torch_peak is not None else smi_peak
    info.update(peak_mem_torch_gb=torch_peak, peak_mem_nvidia_smi_gb=smi_peak)
    checks["peak_memory_below_limit"] = peak is not None and 0 < peak < mem_limit_gb

    # losses in the TensorBoard record, every arm
    ev = _events(run)
    if ev is None:
        info["events"] = "not read (no tensorboard or no event file)"
    else:
        bad = sorted(t for t, vals in ev.items() if not all(math.isfinite(v) for v in vals))
        checks["no_nonfinite_event"] = not bad
        info["nonfinite_events"] = bad[:10]
        bev_on = s["BEV_KD"] == "on"
        for role, bank in BANKS.items():
            want = bev_on and s[f"{role.upper()}_KD"] not in ("off", "off_bev")
            has = any(t.startswith(f"train/loss_distill_{bank}") for t in ev)
            checks[f"bev_kd_{role}_as_set"] = has == want
    # the checkpoint holds what the settings say
    if last:
        keys = _ckpt_keys(last[-1])
        checks["ckpt_refiner_as_set"] = any(k.startswith("agent.ck_student.") for k in keys) == (s["REFINER"] == "on")
        checks["ckpt_bev_kd_as_set"] = any("_distill." in k for k in keys) == (s["BEV_KD"] == "on")
        checks["ckpt_no_teacher_tensor"] = not any(k.startswith("agent.ck_student.") and "teacher" in k for k in keys)
    return {"pass": all(checks.values()), "checks": checks, "info": info}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mem-limit-gb", type=float, default=22.5)
    a = ap.parse_args(argv)
    res = check_run(a.run_dir, a.mem_limit_gb)
    Path(a.run_dir, "launch/smoke_check.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    for k, v in res["checks"].items():
        print(f"[smoke_check] {'ok  ' if v else 'FAIL'} {k}")
    print(f"[smoke_check] {json.dumps(res['info'], default=str)}")
    print(f"[smoke_check] {'PASS' if res['pass'] else 'FAIL'} {a.run_dir}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
