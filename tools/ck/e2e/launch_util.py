#!/usr/bin/env python
"""CK Phase 2 launch helpers (owner: launch).  Used by train_e2e.sh / smoke_e2e.sh; importable for tests.

  launch_util.py overrides  --run-dir R --gpus 0,1,2,3 [--workers 6] [--max-epochs 30] [--ck-set k=v ...]
                            [--ck-set-file F] [--extra-file F] [--resume-ckpt P] [--wandb online|offline|disable]
                            --out R/launch/overrides.txt
  launch_util.py gpu-check  --gpus 0,1,2,3 [--max-used-mib 1024]       exit 3 when a GPU is outside 0-3 or busy
  launch_util.py pick-gpus  --n 1 --among 1,2,3                         first free GPUs (stdout, comma list)
  launch_util.py find-last  --run-dir R                                 newest <R>/train/**/checkpoints/last.ckpt
  launch_util.py wandb-mode                                             online if the W&B API answers, else offline
  launch_util.py labeler-health --run-dir R --state F [--stall-s 1200]  exit 5 when an alive labeler is hung
  launch_util.py status     --run-dir R [--json]                        train / labeler / steps / labels / ETA
  launch_util.py smoke-tokens --n 240 --out F.json --extra-out F.txt    fixed S3 token set (whole logs, sorted)
  launch_util.py smoke-check --stage S1..S5 --run-dir R --out result.json [...]
  launch_util.py smoke-report --root SMOKE_ROOT                         smoke_report.json + kd_loss_table.{json,md}

Hydra overrides follow contract config.hydra with '++' instead of '+' (works whether or not the agent yaml already
lists the key); the trainer recipe is contract config.trainer_recipe (= v2 r34 source + 4-GPU counters).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO = str(Path(__file__).resolve().parents[3])   # repo root of this worktree
PY = "/venv/ssr/bin/python"
HERE = Path(__file__).resolve().parent
CK_YAML = HERE / "ck_e2e.yaml"
E2E_ROOT = "/home/external-user/ssd/yongjae_refiner/ck/phase2/e2e"
SMOKE_ROOT = "/home/external-user/ssd/yongjae_refiner/ck/phase2/e2e_smoke"
CK_DATA = "/home/external-user/ssd/yongjae_refiner/ck"
PACKED_TRAIN = f"{CK_DATA}/packed/navtrain_train"
ALLOWED_GPUS = ("0", "1", "2", "3")
N_TRAIN_TOKENS = 85109
TOKENS_PER_MB = 4

TRAIN_ENV = {
    "PYTHONPATH": REPO,
    "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
    "NUPLAN_MAPS_ROOT": "/home/external-user/yongjae/SSR/data/dataset/maps",
    "OPENSCENE_DATA_ROOT": "/home/external-user/yongjae/SSR/data/dataset",
    "NAVSIM_DEVKIT_ROOT": REPO,
    "NCCL_SOCKET_IFNAME": "lo",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}

# report 45 monitors (status only): §4-3 transition metric = student top-16 raw (pdm_score_256) failure rate vs v2 r34
# on navtrain_train (same numbers as online.RAW_FAIL_R34; within 1.5x is the target), and decision #4 = CK share of
# the BEV gradient (re-decide g = 0.1 if it stays above 0.5).
RAW_FAIL_R34 = {"nc": 0.024, "dac": 0.034, "ttc": 0.074}
RAW_FAIL_MAX_RATIO = 1.5
GSHARE_WARN = 0.5
MONITOR_N_GSHARE = 20
MONITOR_N_STEPS = 500

PLAN_ANCHOR_FILE = "/home/external-user/kyungmin/SSR-v2/data/planning_vb/trajectory_anchors_256.npy"
PLAN_SCORE_FILE = "/home/external-user/kyungmin/SSR-v2/data/planning_vb/pdm_score_256"


class LaunchError(SystemExit):
    """Refusal with a message (exit code 3)."""

    def __init__(self, msg: str):
        super().__init__(3)
        self.msg = msg

    def __str__(self):
        return self.msg


# ----------------------------------------------------------------------------------------------- ck_e2e dict
def load_ck_yaml(path=CK_YAML) -> Dict[str, Any]:
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise LaunchError(f"{path}: not a mapping")
    return d


def _set_dotted(d: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            if p in cur:
                raise LaunchError(f"ck_e2e.{key}: {p} is not a mapping")
            raise LaunchError(f"ck_e2e.{key}: unknown key {p!r}")
        cur = cur[p]
    if parts[-1] not in cur:
        raise LaunchError(f"ck_e2e.{key}: unknown key (not in {CK_YAML.name})")
    cur[parts[-1]] = value


def apply_sets(d: Dict[str, Any], sets: Iterable[str]) -> Dict[str, Any]:
    """'key=value' (dotted key for nested dicts; value parsed as YAML) applied onto a copy of d."""
    import copy
    import yaml
    out = copy.deepcopy(d)
    for s in sets:
        s = s.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            raise LaunchError(f"--ck-set {s!r}: expected key=value")
        k, v = s.split("=", 1)
        _set_dotted(out, k.strip(), yaml.safe_load(v) if v.strip() != "" else "")
    return out


def read_lines(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        raise LaunchError(f"{path}: not a file")
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]


def flatten(d: Dict[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    out = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out += flatten(v, key + ".")
        else:
            out.append((key, v))
    return out


_PLAIN = re.compile(r"^[A-Za-z0-9_./\-+@]+$")


def hydra_value(v: Any) -> str:
    """Python value -> Hydra override value text (round-trips through the override grammar)."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise LaunchError(f"non-finite value {v}")
        r = repr(v)
        return r if ("." in r or "e" in r) else r + ".0"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(hydra_value(x) for x in v) + "]"
    s = str(v)
    if s == "":
        return "''"
    if _PLAIN.match(s) and s.lower() not in ("true", "false", "null", "none") and not _looks_numeric(s):
        return s
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def no_equals(path: str, what: str) -> str:
    if "=" in str(path):
        raise LaunchError(f"{what} {path!r} contains '=' (Hydra override paths must not)")
    return str(path)


def validate_ck(ck: Dict[str, Any]) -> None:
    """Launch-side pre-check (the agent repeats these in _validate_config); also CKE2EConfig.from_any if present."""
    io = str(ck.get("io_dir", ""))
    if ck.get("enabled"):
        if not io or not os.path.isabs(io):
            raise LaunchError(f"ck_e2e.io_dir must be an absolute path, got {io!r}")
        no_equals(io, "ck_e2e.io_dir")
    if not (1 <= int(ck["topk"]) <= 256):
        raise LaunchError(f"ck_e2e.topk {ck['topk']} not in [1, 256]")
    if int(ck["record_from_epoch"]) > int(ck["onpolicy_from_epoch"]):
        raise LaunchError("ck_e2e.record_from_epoch > onpolicy_from_epoch")
    if int(ck["label_lag"]) < 1:
        raise LaunchError("ck_e2e.label_lag < 1")
    for key, arm in (("teacher_det_run", "T"), ("teacher_map_run", "M")):
        run = str(ck.get(key) or "")
        if not run:
            continue
        no_equals(run, f"ck_e2e.{key}")
        cfg = Path(run) / "config.json"
        if cfg.is_file():
            got = json.loads(cfg.read_text()).get("arm")
            if got != arm:
                raise LaunchError(f"ck_e2e.{key}={run}: config.json arm {got!r} != {arm!r}")
    for k, v in (ck.get("phase1") or {}).items():
        no_equals(str(v), f"ck_e2e.phase1.{k}")


def try_from_any(ck: Dict[str, Any]) -> Optional[str]:
    """Validate with the model owner's CKE2EConfig.from_any when importable; returns None or an error string."""
    try:
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from navsim.agents.para_ssr.ck.online import CKE2EConfig  # noqa: WPS433
    except Exception as e:      # noqa: BLE001  (model owner's module not there yet)
        return f"skip: {type(e).__name__}: {e}"
    CKE2EConfig.from_any(ck)
    return None


# ----------------------------------------------------------------------------------------------- overrides
def _okey(o: str) -> str:
    k = o.split("=", 1)[0]
    return k.lstrip("+~")


def recipe(run: str, run_dir: str, ngpu: int, workers: int, max_epochs: int, wandb: str) -> List[str]:
    """contract config.trainer_recipe (v2 r34 source overrides + 4-GPU counters)."""
    ov = [
        "agent=para_ssr_agent", "agent.lr=1e-4", "agent.config.max_epochs=30",
        f"experiment_name=ck_e2e_{run}", "scene_filter=navtrain", "split=trainval",
        "dataloader.params.batch_size=4", f"dataloader.params.num_workers={workers}",
        f"trainer.params.max_epochs={max_epochs}", "trainer.params.accumulate_grad_batches=8",
        "trainer.params.check_val_every_n_epoch=5", "trainer.params.precision=32",
        f"+trainer.params.devices={ngpu}", f"trainer.params.strategy={'ddp' if ngpu > 1 else 'auto'}",
        "trainer.params.gradient_clip_val=35.0", "trainer.params.gradient_clip_algorithm=norm",
        "agent.config.use_task_interaction=true", "agent.config.use_det_motion_head=true",
        "agent.config.use_map_head=true",
        "~agent.config.grad_balance_target", "+agent.config.grad_balance_target={plan:0.4,det:0.3,map:0.3}",
        "agent.config.plan_anchor=true", f"agent.config.plan_anchor_file={PLAN_ANCHOR_FILE}",
        f"agent.config.plan_score_file={PLAN_SCORE_FILE}",
        "agent.config.image_architecture=resnet34.tv_in1k", "agent.config.plan_heading_from_xy=true",
        "agent.config.grad_balance_warmup_iters=5300", "agent.config.grad_balance_interval=100",
        "agent.config.grad_norm_log_interval=100",
        "trainer.params.limit_val_batches=0", "trainer.params.num_sanity_val_steps=0",
        "++agent.config.log_sync_dist=false",
    ]
    if wandb == "disable":
        ov.append("wandb.enable=false")
    else:
        ov += ["wandb.enable=true", "wandb.project=para-ssr-v2", f"wandb.name={run}", f"wandb.mode={wandb}"]
    return ov


def build_overrides(run_dir: str, gpus: Sequence[str], workers: int = 6, max_epochs: int = 30,
                    ck_sets: Sequence[str] = (), extra: Sequence[str] = (), resume_ckpt: Optional[str] = None,
                    wandb: str = "online", ck_yaml=CK_YAML) -> Tuple[List[str], Dict[str, Any]]:
    """-> (ordered Hydra overrides, effective ck_e2e dict).  extra overrides replace recipe entries of the same key."""
    run_dir = no_equals(str(Path(run_dir)), "run dir")
    if not os.path.isabs(run_dir):
        raise LaunchError(f"run dir must be absolute: {run_dir}")
    run = Path(run_dir).name
    ck = load_ck_yaml(ck_yaml)
    ck["io_dir"] = f"{run_dir}/ck_e2e"
    ck = apply_sets(ck, ck_sets)
    validate_ck(ck)
    base = recipe(run, run_dir, len(gpus), workers, max_epochs, wandb)
    base += [f"++agent.config.ck_e2e.{k}={hydra_value(v)}" for k, v in flatten(ck)]
    base.append(f"output_dir={run_dir}/train/")
    if resume_ckpt:
        base.append(f"++resume_checkpoint={no_equals(resume_ckpt, 'resume checkpoint')}")
    merged: Dict[str, str] = {}
    for o in base + list(extra):
        k = _okey(o)
        if o.startswith("~"):
            merged["~" + k] = o
        else:
            merged[k] = o
    return list(merged.values()), ck


# ----------------------------------------------------------------------------------------------- GPUs
def nvidia_smi_used() -> Dict[str, int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise LaunchError(f"nvidia-smi failed: {out.stderr.strip()}")
    used = {}
    for ln in out.stdout.strip().splitlines():
        i, m = [x.strip() for x in ln.split(",")]
        used[i] = int(float(m))
    return used


def parse_gpus(s: str) -> List[str]:
    g = [x.strip() for x in str(s).split(",") if x.strip()]
    if not g:
        raise LaunchError("empty GPU list")
    bad = [x for x in g if x not in ALLOWED_GPUS]
    if bad:
        raise LaunchError(f"GPU {','.join(bad)} not allowed: CK Phase 2 uses only GPUs {','.join(ALLOWED_GPUS)}")
    if len(set(g)) != len(g):
        raise LaunchError(f"duplicate GPU in {s}")
    return g


def gpu_check(gpus: Sequence[str], max_used_mib: int = 1024, used: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    gpus = parse_gpus(",".join(gpus))
    used = nvidia_smi_used() if used is None else used
    busy = {g: used.get(g) for g in gpus if used.get(g) is None or used[g] > max_used_mib}
    if busy:
        raise LaunchError(f"GPU busy or unknown (MiB used > {max_used_mib}): {busy}")
    return {g: used[g] for g in gpus}


def pick_gpus(n: int, among: Sequence[str], max_used_mib: int = 1024, used=None) -> List[str]:
    among = parse_gpus(",".join(among))
    used = nvidia_smi_used() if used is None else used
    free = [g for g in among if used.get(g, 10 ** 9) <= max_used_mib]
    if len(free) < n:
        raise LaunchError(f"need {n} free GPU(s) among {among}, free: {free} (used {used})")
    return free[:n]


# ----------------------------------------------------------------------------------------------- ckpt / pids
def find_last_ckpt(run_dir: str) -> Path:
    cands = [Path(p) for p in glob.glob(f"{run_dir}/train/lightning_logs/version_*/checkpoints/last.ckpt")]
    cands = [p for p in cands if p.is_file()]
    if not cands:
        raise LaunchError(f"no last.ckpt under {run_dir}/train/lightning_logs/version_*/checkpoints")
    p = max(cands, key=lambda q: q.stat().st_mtime)
    return Path(no_equals(str(p), "last.ckpt path"))


def epoch_ckpts(run_dir: str) -> List[Tuple[int, Path]]:
    out = []
    for p in glob.glob(f"{run_dir}/train/lightning_logs/version_*/checkpoints/epoch=*-step=*.ckpt"):
        m = re.search(r"epoch=(\d+)-step=(\d+)\.ckpt$", p)
        if m:
            out.append((int(m.group(1)), Path(p)))
    return sorted(out)


def pid_alive(pid_file, needle: str = "") -> Optional[int]:
    """pid from <pid_file> if that process exists (and its cmdline contains needle); else None."""
    p = Path(pid_file)
    if not p.is_file():
        return None
    try:
        pid = int(p.read_text().strip())
    except ValueError:
        return None
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return None
    if needle and needle not in cmd:
        return None
    return pid


# ----------------------------------------------------------------------------------------------- labeler health
def labeler_health(run_dir: str, state_path: str, stall_s: float = 1200.0, now: Optional[float] = None
                   ) -> Tuple[bool, str]:
    """Watchdog check of a labeler that is alive: (healthy, reason).  Unhealthy = lab/status.json not rewritten for
    stall_s (main loop hung; it is rewritten at least every poll / --status-every), or no scored token
    (tokens_ok + err + skip unchanged) for stall_s while it has a backlog or tasks in flight (pool stuck).
    The progress counter / time of the last change are kept in state_path between calls."""
    now = time.time() if now is None else float(now)
    sp = Path(run_dir) / "ck_e2e/lab/status.json"
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
        _write_json(st_p, {"pid": d.get("pid"), "prog": prog, "t": now})
        return True, f"progress {prog}" + ("" if busy else " (idle)")
    stalled = now - float(st.get("t", now))
    if stalled > stall_s:
        return False, (f"no scored token for {stalled:.0f} s (> {stall_s:.0f}) with backlog {d.get('backlog_chunks')} "
                       f"chunks / inflight {d.get('inflight')}: pool stuck")
    return True, f"busy, no new token for {stalled:.0f} s"


# ----------------------------------------------------------------------------------------------- W&B
def wandb_mode(timeout: int = 45) -> str:
    """'online' when the W&B API answers with the stored credentials, else 'offline' (same project / name)."""
    env = os.environ.get("WANDB_MODE")
    if env in ("online", "offline", "disabled"):
        return "disable" if env == "disabled" else env
    code = "import wandb; v = wandb.Api(timeout=20).viewer; print('ok', bool(v))"
    try:
        r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=timeout)
        return "online" if r.returncode == 0 and "ok" in r.stdout else "offline"
    except Exception:       # noqa: BLE001
        return "offline"


# ----------------------------------------------------------------------------------------------- jsonl
def read_jsonl(path, tail_bytes: Optional[int] = None) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    with open(p, "rb") as f:
        if tail_bytes:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            data = f.read()
            if size > tail_bytes:
                data = data.split(b"\n", 1)[-1]
        else:
            data = f.read()
    out = []
    for ln in data.decode(errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v


def _median(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(statistics.median(xs)) if xs else None


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(sum(xs) / len(xs)) if xs else None


def rec_epoch(r: Dict[str, Any]) -> Optional[int]:
    for k in ("epoch", "ck/epoch", "trainer/epoch"):
        if k in r and _num(r[k]) is not None:
            return int(_num(r[k]))
    if "epoch_frac" in r and _num(r["epoch_frac"]) is not None:
        return int(math.floor(_num(r["epoch_frac"])))
    return None


def rec_frac(r: Dict[str, Any]) -> Optional[float]:
    for k in ("epoch_frac", "ck/epoch_frac"):
        if k in r and _num(r[k]) is not None:
            return _num(r[k])
    return None


# ----------------------------------------------------------------------------------------------- status
def monitors(recs: Sequence[Dict[str, Any]], n_gshare: int = MONITOR_N_GSHARE, n_steps: int = MONITOR_N_STEPS
             ) -> Dict[str, Any]:
    """report 45 watch metrics from the step log tail: CK share of the BEV gradient (last n_gshare records that carry
    ck/gshare_ck) and the student top-16 raw failure rate per key (last n_steps records) vs RAW_FAIL_R34."""
    out: Dict[str, Any] = {}
    gs = [_num(r.get("ck/gshare_ck")) for r in recs if "ck/gshare_ck" in r]
    gs = [g for g in gs if g is not None and math.isfinite(g)][-n_gshare:]
    if gs:
        out["gshare_ck"] = {"n": len(gs), "mean": float(sum(gs) / len(gs)), "max": float(max(gs)),
                            "warn": bool(sum(gs) / len(gs) > GSHARE_WARN)}
    tail = list(recs)[-n_steps:]
    rf = {}
    for k, ref in RAW_FAIL_R34.items():
        m = _mean([_num(r.get(f"ck/raw_fail_{k}")) for r in tail])
        if m is not None:
            rf[k] = {"mean": m, "r34": ref, "ratio": m / ref, "over": bool(m / ref > RAW_FAIL_MAX_RATIO)}
    if rf:
        out["raw_fail"] = {"n": len(tail), "keys": rf, "within_1.5x": not any(v["over"] for v in rf.values())}
    return out


def status_warnings(st: Dict[str, Any]) -> List[str]:
    w = []
    t, lb = st["train"], st["labeler"]
    if t["alive"] and not lb["alive"] and not st.get("labeler_disabled"):
        w.append(f"labeler not running while training is alive (labeler exit {lb['exit']}); watchdog "
                 f"{'alive' if (st.get('watchdog') or {}).get('alive') else 'NOT running'}; see labeler/labeler.log, "
                 "restart with 'RUN=<run> bash tools/ck/e2e/train_e2e.sh labeler'")
    lw = (st.get("last_epoch_start") or {}).get("label_warning") or {}
    if lw.get("warn"):
        w.append(f"epoch {st['last_epoch_start'].get('epoch')} label supply: {lw.get('msg')}")
    mon = st.get("monitor") or {}
    g = mon.get("gshare_ck")
    if g and g["warn"]:
        w.append(f"CK share of the BEV gradient mean {g['mean']:.2f} > {GSHARE_WARN} over the last {g['n']} records "
                 "(report 45 decision #4: re-decide g = 0.1 if it stays there)")
    rf = mon.get("raw_fail")
    if rf and not rf["within_1.5x"]:
        over = ", ".join(f"{k} {v['mean']:.3f} ({v['ratio']:.1f}x)" for k, v in rf["keys"].items() if v["over"])
        w.append(f"student top-16 raw fail rate above 1.5x r34: {over} (report 45 §4-3 transition metric; logged only)")
    return w


def run_status(run_dir: str, max_epochs: int = 30) -> Dict[str, Any]:
    R = Path(run_dir)
    st: Dict[str, Any] = {"run_dir": str(R), "now_utc": time.strftime("%F %T", time.gmtime())}
    tp = pid_alive(R / "launch/train.pid", "run_training")
    lp = pid_alive(R / "labeler/labeler.pid", "labeler")
    st["train"] = {"pid": tp, "alive": tp is not None,
                   "exit": (R / "launch/train.exit").read_text().strip() if (R / "launch/train.exit").is_file()
                   else None}
    st["labeler"] = {"pid": lp, "alive": lp is not None,
                     "exit": (R / "labeler/labeler.exit").read_text().strip()
                     if (R / "labeler/labeler.exit").is_file() else None}
    wp = pid_alive(R / "labeler/watchdog.pid", "_watchdog")
    rs = R / "labeler/watchdog.restarts"
    st["watchdog"] = {"pid": wp, "alive": wp is not None,
                      "restarts": len(rs.read_text().splitlines()) if rs.is_file() else 0}
    st["labeler_disabled"] = not (R / "labeler").is_dir()
    st["train_done"] = (R / "ck_e2e/TRAIN_DONE").exists()
    eff = R / "launch/ck_e2e_effective.json"
    if eff.is_file():
        try:
            max_epochs = int(json.loads(eff.read_text()).get("_trainer_max_epochs", max_epochs))
        except Exception:       # noqa: BLE001
            pass
    recs = read_jsonl(R / "ck_e2e/steps_rank0.jsonl", tail_bytes=8 << 20)
    if recs:
        st["monitor"] = monitors(recs)
        last = recs[-1]
        sec = _median([_num(r.get("sec_step")) for r in recs[-300:]])
        ep = rec_epoch(last)
        frac = rec_frac(last)
        st["steps"] = {"n_tail": len(recs), "epoch": ep, "epoch_frac": frac,
                       "phase": last.get("ck/phase"), "sec_step_median": sec,
                       "mem_gb": last.get("mem_gb"), "ck_loss": last.get("ck/loss"),
                       "G_src_phase1": last.get("ck/G_src_phase1"), "t": last.get("t") or last.get("time")}
        ep_recs = read_jsonl(R / "ck_e2e/epochs.jsonl")
        ws = next((int(r["world_size"]) for r in reversed(ep_recs) if r.get("world_size")), 4)
        mb_per_epoch, basis = N_TRAIN_TOKENS / (TOKENS_PER_MB * ws), "85,109 tokens / (4 x world size)"
        for r in reversed(recs):                # batch_idx / (epoch_frac - epoch) = num_training_batches
            b, fr, e = _num(r.get("batch")), rec_frac(r), rec_epoch(r)
            if b and fr is not None and e is not None and fr - e > 1e-6:
                mb_per_epoch, basis = b / (fr - e), "num_training_batches from the step log"
                break
        if frac is None and ep is not None:
            frac = float(ep)
        if sec and frac is not None and st["train"]["alive"]:
            remain = max(0.0, max_epochs - frac) * mb_per_epoch * sec
            st["eta"] = {"hours_left": round(remain / 3600, 2),
                         "finish_utc": time.strftime("%F %T", time.gmtime(time.time() + remain)),
                         "basis": f"median sec_step of last 300 mb x {mb_per_epoch:.0f} mb/epoch ({basis}) x "
                                  f"{max(0.0, max_epochs - frac):.2f} epochs left of {max_epochs} [추정]"}
    ep_recs = read_jsonl(R / "ck_e2e/epochs.jsonl")
    if ep_recs:
        st["last_epoch_record"] = ep_recs[-1]
        es = [r for r in ep_recs if r.get("event") == "epoch_start"]
        if es:
            st["last_epoch_start"] = {k: es[-1].get(k) for k in ("epoch", "phase", "label_warning")}
    st["checkpoints"] = [str(p.name) for _, p in epoch_ckpts(str(R))]
    lab = R / "ck_e2e/lab"
    s = lab / "status.json"
    if s.is_file():
        try:
            st["labeler_status"] = json.loads(s.read_text())
        except json.JSONDecodeError:
            st["labeler_status"] = "unreadable"
    st["generations_done"] = sorted(int(p.parent.name[2:]) for p in lab.glob("ep*/DONE.json")) if lab.is_dir() else []
    st["warnings"] = status_warnings(st)
    return st


def print_status(st: Dict[str, Any]) -> None:
    print(f"[{st['now_utc']} UTC] {st['run_dir']}")
    t, lb = st["train"], st["labeler"]
    print(f"  train   : {'alive pid ' + str(t['pid']) if t['alive'] else 'not running'}"
          f"{'  exit ' + t['exit'] if t['exit'] is not None else ''}{'  TRAIN_DONE' if st['train_done'] else ''}")
    wd = st.get("watchdog") or {}
    print(f"  labeler : {'alive pid ' + str(lb['pid']) if lb['alive'] else 'not running'}"
          f"{'  exit ' + lb['exit'] if lb['exit'] is not None else ''}  watchdog "
          f"{'alive' if wd.get('alive') else 'not running'} (restarts {wd.get('restarts', 0)})")
    if "steps" in st:
        s = st["steps"]
        print(f"  steps   : epoch {s['epoch']} frac {s['epoch_frac']} phase {s['phase']} sec/mb {s['sec_step_median']} "
              f"mem {s['mem_gb']} GB ck/loss {s['ck_loss']} G_src_phase1 {s['G_src_phase1']}")
    if "eta" in st:
        print(f"  ETA     : {st['eta']['hours_left']} h -> {st['eta']['finish_utc']} UTC ({st['eta']['basis']})")
    mon = st.get("monitor") or {}
    if mon.get("gshare_ck"):
        g = mon["gshare_ck"]
        print(f"  gshare  : CK share of BEV grad, last {g['n']} records mean {g['mean']:.3f} max {g['max']:.3f} "
              f"(warn > {GSHARE_WARN})")
    if mon.get("raw_fail"):
        rf = mon["raw_fail"]
        print(f"  rawfail : student top-16 raw fail, last {rf['n']} mb: " + ", ".join(
            f"{k} {v['mean']:.3f} = {v['ratio']:.2f}x r34{' (>1.5x)' if v['over'] else ''}" for k, v in rf["keys"].items()))
    les = st.get("last_epoch_start") or {}
    if (les.get("label_warning") or {}).get("msg"):
        print(f"  labels@e: epoch {les.get('epoch')}: {les['label_warning']['msg']}")
    if "last_epoch_record" in st:
        print(f"  epoch   : {json.dumps(st['last_epoch_record'])[:300]}")
    print(f"  ckpts   : {len(st['checkpoints'])} {st['checkpoints'][-3:]}")
    if "labeler_status" in st:
        print(f"  labels  : {json.dumps(st['labeler_status'])[:400]}")
    print(f"  gen DONE: {st['generations_done']}")
    for w in st.get("warnings") or []:
        print(f"  WARNING : {w}")


# ----------------------------------------------------------------------------------------------- smoke: tokens
def smoke_tokens(n: int = 240, packed: str = PACKED_TRAIN, min_log: int = 20) -> Tuple[List[str], List[str]]:
    """n tokens = whole navtrain_train logs (sorted by name, logs with >= min_log tokens) cut to exactly n, all with a
    metric cache; returns (tokens, logs).  Same set every call (deterministic)."""
    import pandas as pd
    df = pd.read_parquet(Path(packed) / "tokens.parquet")
    mc_root = Path("/home/external-user/ssd/yongjae_refiner/metric_cache")
    cnt = df.groupby("log").size()
    toks, logs = [], []
    for lg in sorted(cnt.index):
        if cnt[lg] < min_log:
            continue
        sub = df[df.log == lg].sort_values("row")
        good = [t for t in sub.token if (mc_root / lg / "unknown" / t / "metric_cache.pkl").is_file()]
        if not good:
            continue
        logs.append(lg)
        toks += good[: n - len(toks)]
        if len(toks) >= n:
            break
    if len(toks) < n:
        raise LaunchError(f"only {len(toks)} smoke tokens found")
    return toks, logs


# ----------------------------------------------------------------------------------------------- smoke: checks
def _exit_code(path) -> Optional[int]:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return int(p.read_text().strip())
    except ValueError:
        return None


def _nonfinite_count(recs: Sequence[Dict[str, Any]]) -> int:
    n = 0
    for r in recs:
        for k, v in r.items():
            if k.startswith("ck/") and isinstance(v, (int, float)) and not math.isfinite(float(v)):
                n += 1
        if _num(r.get("ck/sur_nonfinite")):
            n += int(_num(r.get("ck/sur_nonfinite")) > 0)
    return n


def ckpt_callback_state(path) -> Optional[Dict[str, Any]]:
    """The CK callback's state in a Lightning ckpt: the 'callbacks' entry holding 'mb' and 'ema'."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k, v in (ck.get("callbacks") or {}).items():
        if isinstance(v, dict) and "mb" in v and "ema" in v:
            return dict(v, _key=str(k))
    return None


def labeler_tok_s(io_dir: Path, epoch: Optional[int] = None) -> Tuple[Optional[float], str]:
    """Labeler throughput [tok/s]: lab/status.json recent rate if present, else median chunk-marker tok_s."""
    s = io_dir / "lab/status.json"
    if s.is_file():
        try:
            d = json.loads(s.read_text())
            for k in ("recent_tok_s", "tok_s", "tok_per_s", "rate_tok_s"):
                if _num(d.get(k)):
                    return float(d[k]), f"status.json:{k}"
        except json.JSONDecodeError:
            pass
    pat = f"lab/ep{epoch:03d}/chunks/*.json" if epoch is not None else "lab/ep*/chunks/*.json"
    vals, n_tot, sec_tot = [], 0, 0.0
    for p in io_dir.glob(pat):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if _num(d.get("tok_s")):
            vals.append(float(d["tok_s"]))
        n_tot += int(d.get("n", 0) or 0)
        sec_tot += float(d.get("sec", 0) or 0)
    if vals:
        return _median(vals), f"median chunk tok_s over {len(vals)} chunks"
    if sec_tot > 0:
        return n_tot / sec_tot, "sum n / sum sec of chunk markers"
    return None, "no labeler throughput record"


PHASE_NAMES = {0: "replay", 1: "replay_record", 2: "onpolicy"}


def phase_timing(recs: Sequence[Dict[str, Any]], skip_first: int = 20) -> Dict[str, Any]:
    """Per CK phase (ck/phase code), records after the first skip_first of every epoch (warm-up, worker start):
    median sec_step, mean (sec_step + sec_wait) = steady wall time per micro-batch, peak mem."""
    by_ep: Dict[int, List[Dict[str, Any]]] = {}
    for r in recs:
        e = rec_epoch(r)
        by_ep.setdefault(-1 if e is None else e, []).append(r)
    out: Dict[str, Any] = {}
    for e, rs in sorted(by_ep.items()):
        for r in rs[skip_first:]:
            ph = _num(r.get("ck/phase"))
            if ph is None:
                continue
            d = out.setdefault(PHASE_NAMES.get(int(ph), str(int(ph))), {"step": [], "wall": [], "mem": [], "epochs": set()})
            st, wt = _num(r.get("sec_step")), _num(r.get("sec_wait")) or 0.0
            if st is not None:
                d["step"].append(st)
                d["wall"].append(st + wt)
            d["mem"].append(_num(r.get("mem_gb")) or 0.0)
            d["epochs"].add(e)
    res = {}
    for k, d in out.items():
        res[k] = {"n": len(d["step"]), "epochs": sorted(d["epochs"]), "sec_step_median": _median(d["step"]),
                  "sec_step_mean": _mean(d["step"]), "wall_per_mb_mean": _mean(d["wall"]),
                  "mem_gb_max": max(d["mem"]) if d["mem"] else None}
    return res


def epoch_overhead(run_dir: Path, recs: Sequence[Dict[str, Any]], skip_first: int = 20) -> Dict[str, Any]:
    """Fixed cost per epoch beyond the steady micro-batches: epoch wall (epochs.jsonl start -> end) minus n_mb x the
    steady per-mb wall of that epoch, plus the gap to the next epoch start (checkpoint save, loader restart).  Also the
    one-time start-up (launch.log 'start' line -> first epoch_start)."""
    ev = read_jsonl(Path(run_dir) / "ck_e2e/epochs.jsonl")
    starts = {}
    ends = {}
    for r in ev:
        if r.get("event") == "epoch_start":
            starts[int(r["epoch"])] = float(r["time"])            # the last attempt of an epoch wins
        elif r.get("event") == "epoch_end":
            ends[int(r["epoch"])] = float(r["time"])
    per = []
    for e in sorted(ends):
        if e not in starts:
            continue
        rs = [r for r in recs if rec_epoch(r) == e]
        steady = _mean([(_num(r.get("sec_step")) or 0.0) + (_num(r.get("sec_wait")) or 0.0) for r in rs[skip_first:]])
        if not rs or steady is None:
            continue
        inside = (ends[e] - starts[e]) - len(rs) * steady
        gap = (starts[e + 1] - ends[e]) if (e + 1) in starts else None
        per.append({"epoch": e, "n_mb": len(rs), "wall_s": ends[e] - starts[e], "steady_s_per_mb": steady,
                    "inside_overhead_s": inside, "gap_to_next_s": gap})
    gaps = [p["gap_to_next_s"] for p in per if p["gap_to_next_s"] is not None]
    out = {"epochs": per, "inside_overhead_s_median": _median([p["inside_overhead_s"] for p in per]),
           "gap_s_median": _median(gaps)}
    ll = Path(run_dir) / "launch/launch.log"
    if ll.is_file() and starts:
        m = re.search(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC\] start RUN=", ll.read_text(), re.M)
        if m:
            import calendar
            t_launch = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            out["startup_s"] = min(starts.values()) - t_launch
    return out


def mb_per_epoch(world_size: int, n_tokens: int = N_TRAIN_TOKENS, batch: int = TOKENS_PER_MB) -> int:
    """DistributedSampler pads to a multiple of world_size; the last micro-batch of a rank may be partial."""
    per_rank = math.ceil(n_tokens / world_size)
    return math.ceil(per_rank / batch)


def schedule_phases(max_epochs: int = 30, record_from: int = 4, onpolicy_from: int = 5) -> List[str]:
    return ["replay" if e < record_from else "replay_record" if e < onpolicy_from else "onpolicy"
            for e in range(max_epochs)]


def estimate_hours(timing: Dict[str, Any], overhead: Dict[str, Any], world_size: int = 4, max_epochs: int = 30,
                   record_from: int = 4, onpolicy_from: int = 5, key: str = "wall_per_mb_mean") -> Dict[str, Any]:
    """30-epoch wall time from measured steady s/mb per phase (DDP smoke) + per-epoch fixed cost + start-up [추정]."""
    mb = mb_per_epoch(world_size)
    per_epoch_fixed = (overhead.get("inside_overhead_s_median") or 0.0) + (overhead.get("gap_s_median") or 0.0)
    rows, tot = [], float(overhead.get("startup_s") or 0.0)
    for ph in ("replay", "replay_record", "onpolicy"):
        n_ep = sum(1 for x in schedule_phases(max_epochs, record_from, onpolicy_from) if x == ph)
        spm = (timing.get(ph) or {}).get(key)
        if spm is None:
            return {"error": f"no {ph} timing"}
        sec = n_ep * (mb * spm + per_epoch_fixed)
        tot += sec
        rows.append({"phase": ph, "epochs": n_ep, "s_per_mb": spm, "min_per_epoch": (mb * spm + per_epoch_fixed) / 60,
                     "hours": sec / 3600})
    return {"mb_per_epoch": mb, "world_size": world_size, "per_epoch_fixed_s": per_epoch_fixed,
            "startup_s": overhead.get("startup_s"), "rows": rows, "total_hours": tot / 3600, "basis": key}


def gen_roundtrip(io_dir: Path, epochs: Sequence[int]) -> Dict[str, Any]:
    """Every row labelled in generation lab/epE (row_state 1) carries exactly the trajectories recorded for it in
    rec/epE (cat(cand, kd_corr), bit-exact), and how many recorded rows are labelled."""
    import numpy as np
    from navsim.agents.para_ssr.ck import e2e_data as D
    out = {}
    for e in epochs:
        try:
            n_rows = int(json.loads((D.gen_dir(str(io_dir), e) / "meta.json").read_text())["n_rows"])
            g = D.open_generation(str(io_dir), e, n_rows, mode="r")
        except Exception as ex:      # noqa: BLE001
            out[e] = {"error": f"{type(ex).__name__}: {ex}"}
            continue
        rs = np.asarray(g["row_state"])
        recs: Dict[int, List[Any]] = {}
        for c in D.list_rec_chunks(str(io_dir), e):
            d = D.read_rec_chunk(c["path"])
            for i, r in enumerate(d["row"]):
                recs.setdefault(int(r), []).append(np.concatenate([d["cand"][i], d["kd_corr"][i]], 0))
        done = np.flatnonzero(rs == 1)
        bad = 0
        for r in done:
            t = np.nan_to_num(np.asarray(g["traj"][r]), nan=-9.0)
            if not any(np.array_equal(t, np.nan_to_num(x, nan=-9.0)) for x in recs.get(int(r), [])):
                bad += 1
        lab = np.asarray(g["labels"])[done]
        out[e] = {"rows_recorded": len(recs), "rows_labelled": int(len(done)),
                  "labelled_of_recorded": len(set(map(int, done)) & set(recs)) / max(len(recs), 1),
                  "labelled_without_matching_record": bad,
                  "cand_ok_frac": float(np.asarray(g["cand_ok"])[done].mean()) if len(done) else None,
                  "label_finite_frac": float(np.isfinite(lab).mean()) if len(done) else None}
    return out


def check_stage(stage: str, run_dir: Optional[str], **kw) -> Dict[str, Any]:
    R = Path(run_dir) if run_dir else None
    res: Dict[str, Any] = {"stage": stage, "run_dir": str(R) if R else None, "checked_utc": time.strftime(
        "%F %T", time.gmtime())}
    checks: Dict[str, Any] = {}
    if stage == "S1":
        rc = int(kw.get("rc", 1))
        checks["pytest_rc_0"] = rc == 0
        res["pytest_summary"] = kw.get("summary", "")
    elif stage in ("S2", "S3", "S4"):
        recs = read_jsonl(R / "ck_e2e/steps_rank0.jsonl")
        res["n_step_records"] = len(recs)
        res["phase_timing"] = phase_timing(recs)
        checks["step_records"] = len(recs) > 0
        sec = [_num(r.get("sec_step")) for r in recs]
        skip = 20 if len(sec) > 60 else max(0, len(sec) // 4)
        res["sec_per_mb_median"] = _median(sec[skip:])
        res["mem_gb_max"] = max([_num(r.get("mem_gb")) or 0.0 for r in recs], default=None)
        res["nonfinite"] = _nonfinite_count(recs)
        res["ck_keys"] = sorted({k for r in recs for k in r if k.startswith("ck/")})
        checks["ck_logs"] = len(res["ck_keys"]) > 0
        checks["nonfinite_0"] = res["nonfinite"] == 0
        res["skipped_steps"] = max([int(_num(r.get("skipped_steps")) or 0) for r in recs], default=0)
        checks["skipped_steps_0"] = res["skipped_steps"] == 0
        if stage == "S2":
            res["train_exit"] = _exit_code(R / "launch/train.exit")
            checks["train_exit_0"] = res["train_exit"] == 0
            checks["mem_le_20gb"] = res["mem_gb_max"] is not None and res["mem_gb_max"] <= 20.0
            pt = res["phase_timing"]
            checks["phases_replay_and_onpolicy"] = all((pt.get(k) or {}).get("n", 0) >= 50 for k in ("replay", "onpolicy"))
        if stage == "S3":
            res["train_exit_a"] = kw.get("exit_a")
            res["train_exit"] = _exit_code(R / "launch/train.exit")
            checks["train_exit_0_after_resume"] = res["train_exit"] == 0
            e3 = [r for r in recs if (rec_epoch(r) or -1) >= 3]
            res["ep3_records"] = len(e3)
            res["ep3_G_src_phase1"] = _mean([_num(r.get("ck/G_src_phase1")) for r in e3])
            res["ep3_G_src_prev"] = _mean([_num(r.get("ck/G_src_prev")) for r in e3])
            res["ep3_G_src_older"] = _mean([_num(r.get("ck/G_src_older")) for r in e3])
            onp = [r for r in recs if _num(r.get("ck/phase")) == 2]
            res["onpolicy_kd_ok_frac"] = _mean([_num(r.get("ck/kd_ok_frac")) for r in onp])
            checks["ep3_G_src_phase1_lt_1pct"] = res["ep3_G_src_phase1"] is not None and res["ep3_G_src_phase1"] < 0.01
            checks["kd_ok_frac_ge_99pct"] = (res["onpolicy_kd_ok_frac"] is not None
                                             and res["onpolicy_kd_ok_frac"] >= 0.99)
            gens = {e: (R / f"ck_e2e/lab/ep{e:03d}/DONE.json").is_file() for e in (1, 2)}
            res["gen_done"] = gens
            checks["lab_ep001_ep002_DONE"] = all(gens.values())
            e2 = [r for r in recs if rec_epoch(r) == 2]
            res["ep2_G_src"] = {k: _mean([_num(r.get(f"ck/G_src_{k}")) for r in e2]) for k in ("phase1", "prev", "older")}
            checks["ep2_G_from_ep1_ge_90pct"] = (res["ep2_G_src"]["prev"] or 0.0) >= 0.9
            try:
                rt = gen_roundtrip(R / "ck_e2e", (1, 2))
            except Exception as ex:      # noqa: BLE001
                rt = {"error": f"{type(ex).__name__}: {ex}"}
            res["gen_roundtrip"] = rt
            checks["gen_roundtrip_bitexact"] = all(isinstance(v, dict) and "error" not in v and v["rows_labelled"] > 0
                                                   and v["labelled_without_matching_record"] == 0
                                                   and v["labelled_of_recorded"] >= 0.99 for v in rt.values()) \
                and "error" not in rt
            # resume continuity: callback state at the epoch-2 ckpt vs the final last.ckpt
            e2 = [p for e, p in epoch_ckpts(str(R)) if e == 2]
            try:
                fin = find_last_ckpt(str(R))
            except LaunchError:
                fin = None
            if e2 and fin is not None:
                a, b = ckpt_callback_state(e2[-1]), ckpt_callback_state(fin)
                if a and b:
                    n_a = int((a.get("ema") or {}).get("n", -1))
                    n_b = int((b.get("ema") or {}).get("n", -1))
                    res["resume_state"] = {"ep2_mb": a["mb"], "final_mb": b["mb"], "ep2_ema_n": n_a,
                                           "final_ema_n": n_b, "mb_per_epoch_expected": kw.get("mb_per_epoch", 60)}
                    d_mb = int(b["mb"]) - int(a["mb"])
                    checks["resume_mb_continuous"] = d_mb == int(kw.get("mb_per_epoch", 60))
                    checks["resume_ema_continuous"] = n_a >= 0 and 0 <= n_b - n_a <= d_mb and n_b - n_a >= d_mb - 2
                else:
                    checks["resume_state_found"] = False
            else:
                checks["resume_ckpts_found"] = False
        if stage == "S4":
            res["train_exit"] = _exit_code(R / "launch/train.exit")
            checks["train_exit_0_no_hang"] = res["train_exit"] == 0
            onp = res["phase_timing"].get("onpolicy") or {}
            spm = onp.get("sec_step_median") if onp else res["sec_per_mb_median"]
            res["onpolicy_sec_per_mb_median"] = spm
            checks["sec_per_mb_le_0.55"] = spm is not None and spm <= 0.55
            ngpu = int(kw.get("ngpu", 4))
            res["ngpu"] = ngpu
            wall = onp.get("wall_per_mb_mean") or spm
            consume = ngpu * TOKENS_PER_MB / wall if wall else None
            res["epoch_overhead"] = epoch_overhead(R, recs)
            wt = R / "watchdog_test.json"
            if wt.is_file():
                w = json.loads(wt.read_text())
                rs_ = R / "labeler/watchdog.restarts"
                w["restarts"] = rs_.read_text().splitlines() if rs_.is_file() else []
                gens = {e: (R / f"ck_e2e/lab/ep{e:03d}/DONE.json").is_file() for e in (1, 2)}
                w["gen_done"] = gens
                res["watchdog_test"] = w
                checks["watchdog_restarted_labeler"] = len(w["restarts"]) >= 1 and all(gens.values())
                checks["killed_labeler_exit_143"] = bool(w["restarts"]) and w["restarts"][0].endswith("exit 143")
            try:
                res["gen_roundtrip"] = gen_roundtrip(R / "ck_e2e", (1, 2))
            except Exception as ex:      # noqa: BLE001
                res["gen_roundtrip"] = {"error": f"{type(ex).__name__}: {ex}"}
            drain = kw.get("drain_dir")
            tok_s, src = None, ""
            if drain and (Path(drain) / "lab/status.json").is_file():
                d = json.loads((Path(drain) / "lab/status.json").read_text())
                if _num(d.get("total_tok_s")) and int(d.get("tokens_ok") or 0) > 0:
                    tok_s, src = float(d["total_tok_s"]), (f"drain run total_tok_s ({d.get('tokens_ok')} tokens, "
                                                           f"uptime {d.get('uptime_s')} s incl. pool start)")
                    res["drain_recent_tok_s"] = _num(d.get("recent_tok_s"))
            if tok_s is None:
                tok_s, src = labeler_tok_s(R / "ck_e2e", 0)
                src += " (during training: supply-bound lower bound)"
            snaps = read_jsonl(R / "lab_snapshots.jsonl")
            busy = [_num(x.get("recent_tok_s")) for x in snaps if int(x.get("backlog_chunks") or 0) > 0]
            res["during_training_max_tok_s_with_backlog"] = max([b for b in busy if b], default=None)
            res["during_training_snapshots"] = len(snaps)
            res.update(train_consume_tok_s=consume, labeler_tok_s=tok_s, labeler_tok_s_source=src)
            checks["labeler_ge_1.5x_consume"] = bool(tok_s and consume and tok_s >= 1.5 * consume)
            res["gshare_records"] = sum(1 for r in recs if "ck/gshare_ck" in r)
            res["gshare_ck_mean"] = _mean([_num(r.get("ck/gshare_ck")) for r in recs if "ck/gshare_ck" in r])
            checks["gshare_logged"] = res["gshare_records"] > 0
            res["labeler_exit"] = _exit_code(R / "labeler/labeler.exit")
            checks["labeler_exit_0"] = res["labeler_exit"] == 0
    elif stage == "S5":
        run = R.name
        ev = R / "eval"
        tabs = {s: (ev / "root/eval" / run / s / "table.md").is_file() for s in ("navtrain_val", "navtest")}
        res["tables"] = tabs
        checks["eval_ck_tables"] = all(tabs.values())
        checks["summary"] = (ev / "summary.json").is_file()
        ic = ev / "root/infer" / run / "navtest/infer_check.json"
        if ic.is_file():
            res["infer_check"] = json.loads(ic.read_text())
            checks["infer_check"] = bool(res["infer_check"].get("pass"))
        vm = ev / "root/eval" / run / "navtrain_val/metrics.json"
        if vm.is_file():
            res["best_variant"] = json.loads(vm.read_text()).get("best_variant")
            checks["variant_chosen_on_val"] = bool(res["best_variant"])
        if (ev / "summary.json").is_file():
            sm = json.loads((ev / "summary.json").read_text())
            res["representative"] = {k: v.get("representative") for k, v in sm.get("splits", {}).items()}
            checks["navtest_representative_row"] = bool(res["representative"].get("navtest"))
    res["checks"] = checks
    res["pass"] = bool(checks) and all(bool(v) for v in checks.values())
    return res


# ----------------------------------------------------------------------------------------------- smoke: report
KD_TERMS = [
    # key, label, formula, weight text, weight fn(record, cfg), source
    ("ck/bce_cand", "GT 점수 BCE(후보)", "BCE(ŝ(G[:, :16]), y)", "λ_score",
     lambda r, c: c["lambda_score"], "공식 라벨 5키(replay: Phase 1 cand / on-policy: 직전 세대 → Phase 1 대체)"),
    ("ck/bce_tkd", "GT 점수 BCE(τ'_KD)", "BCE(ŝ(G[:, 16:]), y)", "λ_corr_score",
     lambda r, c: c["lambda_corr_score"], "τ'_KD 공식 라벨(replay: Phase 1 kd_corr / on-policy: 직전 세대)"),
    ("ck/kd_score", "점수 KD", "KL(p_T‖ŝ(τ)) + KL(p_T,corr‖ŝ(τ'_KD))", "r·λ_kd_score",
     lambda r, c: c["lambda_kd_score"] * (_num(r.get("ck/r")) if _num(r.get("ck/r")) is not None else 1.0),
     "NC·TTC=DET, DAC=MAP, EP·C=두 teacher 평균"),
    ("ck/sur", "교정 surrogate", "L_sur(decode(τ, z, w; slope 0.1))", "λ_sur",
     lambda r, c: c["lambda_sur"], "surrogate GT(GTLoader ref_*), SUR_WEIGHTS"),
    ("ck/kd_ctrl", "교정 KD", "0.25·|c_lon − c_lon,DET| + 1.0·|e_lat − e_lat,MAP|", "r·w_ema",
     lambda r, c: (_num(r.get("ck/r")) if _num(r.get("ck/r")) is not None else 1.0) * (_num(r.get("ck/w_ema")) or 0.0),
     "세로=DET, 측방=MAP (EMA 균형, cap 10)"),
]
AUX_KEYS = ["ck/ratio_v2", "ck/bce_nc", "ck/bce_dac", "ck/bce_ep", "ck/bce_ttc", "ck/bce_comfort", "ck/kd_lon",
            "ck/kd_lat", "ck/w_ema", "ck/r", "ck/kd_ok_frac", "ck/teacher_kd_ok_frac", "ck/G_ok_frac",
            "ck/G_src_phase1", "ck/G_src_prev", "ck/G_src_older", "ck/gshare_ck", "ck/live", "ck/e_abs",
            "ck/raw_fail_nc", "ck/raw_fail_dac", "ck/raw_fail_ttc", "ck/teacher_ms", "ck/student_ms", "ck/total_ms",
            "sec_step", "mem_gb"]
V2_LOSS_KEYS = ("loss_v2", "ck/loss_v2", "v2/loss", "train/loss_v2")


def _v2_loss(r: Dict[str, Any]) -> Optional[float]:
    for k in V2_LOSS_KEYS:
        if _num(r.get(k)) is not None:
            return _num(r[k])
    return None


def kd_table(phases: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, Any]]]) -> Dict[str, Any]:
    """phases: name -> (step records, effective ck config).  Per term: raw mean, weighted mean (weight per record),
    and the reconstruction check sum(weighted) vs ck/loss."""
    out: Dict[str, Any] = {"phases": {}, "terms": [], "aux": {}}
    for name, (recs, cfg) in phases.items():
        lw = float(cfg.get("loss_weight", 1.0))
        ph = {"n_records": len(recs)}
        tot_w = []
        def wt(r, key, wfn):
            """logged weighted value ck/wt_<term> (model's own weighting) else weight x raw from the config"""
            v = _num(r.get("ck/wt_" + key[3:]))
            if v is not None:
                return v
            x = _num(r.get(key))
            return lw * wfn(r, cfg) * x if x is not None else None

        for key, label, formula, wtxt, wfn, src in KD_TERMS:
            raw = [_num(r.get(key)) for r in recs]
            wtd = [wt(r, key, wfn) for r in recs]
            calc = [lw * wfn(r, cfg) * x if x is not None else None for r, x in zip(recs, raw)]
            ph[key] = {"raw": _mean(raw), "weighted": _mean(wtd), "weighted_from_config": _mean(calc),
                       "weighted_source": "log ck/wt_*" if any("ck/wt_" + key[3:] in r for r in recs) else "config"}
        for r in recs:
            parts = [wt(r, key, wfn) for key, _, _, _, wfn, _ in KD_TERMS]
            parts = [x for x in parts if x is not None]
            if parts and _num(r.get("ck/loss")) is not None:
                tot_w.append((sum(parts), _num(r["ck/loss"])))
        ph["ck/loss"] = _mean([_num(r.get("ck/loss")) for r in recs])
        ph["loss_v2"] = _mean([_v2_loss(r) for r in recs])
        ph["ck_over_v2"] = (ph["ck/loss"] / ph["loss_v2"]) if ph["ck/loss"] is not None and ph["loss_v2"] else None
        if tot_w:
            rec_sum = sum(a for a, _ in tot_w) / len(tot_w)
            logged = sum(b for _, b in tot_w) / len(tot_w)
            ph["reconstruct"] = {"sum_weighted_terms": rec_sum, "ck_loss": logged,
                                 "rel_diff": abs(rec_sum - logged) / max(abs(logged), 1e-9)}
        for k in AUX_KEYS:
            ph.setdefault("aux", {})[k] = _mean([_num(r.get(k)) for r in recs])
        out["phases"][name] = ph
    for key, label, formula, wtxt, _, src in KD_TERMS:
        out["terms"].append({"key": key, "label": label, "formula": formula, "weight": wtxt, "source": src})
    return out


def _f(x, nd=4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def kd_table_md(t: Dict[str, Any], cfgs: Dict[str, Dict[str, Any]], notes: Sequence[str]) -> str:
    names = list(t["phases"])
    L = ["# CK Phase 2 smoke — KD loss 표", "",
         "학생 CK loss 항별 실측 평균(smoke 로그 steps_rank0.jsonl) [실측]. raw = 가중 전 값, 가중 = 모델이 로그한 "
         "ck/wt_*(없으면 설정 가중×raw)의 step 평균. 30 epoch 본 학습 값과는 다르다(BEV가 처음부터 학습 중, smoke는 "
         "수백 mb) [추정].", ""]
    hdr = "| 항 | 식 | 가중 | 출처(teacher 키 규칙) | " + " | ".join(f"{n} raw | {n} 가중" for n in names) + " |"
    L += [hdr, "|" + "---|" * (4 + 2 * len(names))]
    for term in t["terms"]:
        k = term["key"]
        cells = []
        for n in names:
            v = t["phases"][n].get(k, {})
            cells += [_f(v.get("raw")), _f(v.get("weighted"))]
        esc = lambda x: str(x).replace("|", "\\|")  # noqa: E731
        L.append(f"| {term['label']} (`{k}`) | {esc(term['formula'])} | {term['weight']} | {esc(term['source'])} | "
                 + " | ".join(cells) + " |")
    L += ["", "| 지표 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    rows = [("ck/loss (전체 CK loss)", lambda p: p.get("ck/loss")),
            ("v2 loss", lambda p: p.get("loss_v2")),
            ("CK / v2 loss 비율", lambda p: p.get("ck_over_v2")),
            ("Σ(가중 항) vs ck/loss 상대차", lambda p: (p.get("reconstruct") or {}).get("rel_diff"))]
    rows += [(k, (lambda kk: (lambda p: p["aux"].get(kk)))(k)) for k in AUX_KEYS]
    for label, fn in rows:
        L.append(f"| {label} | " + " | ".join(_f(fn(t["phases"][n])) for n in names) + " |")
    L += ["", "가중 설정(실행별 ck_e2e):", ""]
    for n in names:
        c = cfgs.get(n, {})
        L.append(f"- {n}: λ_score {c.get('lambda_score')}, λ_corr_score {c.get('lambda_corr_score')}, "
                 f"λ_kd_score {c.get('lambda_kd_score')}, λ_sur {c.get('lambda_sur')}, kd_ctrl_w {c.get('kd_ctrl_w')}, "
                 f"kd_ema {c.get('kd_ema')}, loss_weight {c.get('loss_weight')}, bev_grad_scale {c.get('bev_grad_scale')}, "
                 f"kd_ramp_epochs {c.get('kd_ramp_epochs')}, onpolicy_from {c.get('onpolicy_from_epoch')}")
    if notes:
        L += ["", "주의:", ""] + [f"- {x}" for x in notes]
    return "\n".join(L) + "\n"


def smoke_report(root: str) -> Dict[str, Any]:
    root_p = Path(root)
    stages = {}
    for s in ("S1", "S2", "S3", "S4", "S5"):
        p = root_p / f"result_{s}.json"
        stages[s] = json.loads(p.read_text()) if p.is_file() else {"stage": s, "pass": False, "missing": True}
    rep = {"root": str(root_p), "created_utc": time.strftime("%F %T", time.gmtime()), "stages": stages,
           "all_pass": all(v.get("pass") for v in stages.values())}
    rep["summary"] = {s: {"pass": v.get("pass"), "sec_per_mb": v.get("sec_per_mb_median"),
                          "mem_gb_max": v.get("mem_gb_max"), "labeler_tok_s": v.get("labeler_tok_s"),
                          "train_consume_tok_s": v.get("train_consume_tok_s"),
                          "ep3_G_src": [v.get("ep3_G_src_phase1"), v.get("ep3_G_src_prev"), v.get("ep3_G_src_older")]
                          if s == "S3" else None} for s, v in stages.items()}
    phases, cfgs = {}, {}
    for name, run, code in (("replay(S4 ep0)", "s4", 0), ("replay+기록(S4 ep1)", "s4", 1),
                            ("on-policy(S4 ep2)", "s4", 2)):
        R = root_p / run
        recs = read_jsonl(R / "ck_e2e/steps_rank0.jsonl")
        eff = R / "launch/ck_e2e_effective.json"
        cfg = json.loads(eff.read_text()) if eff.is_file() else load_ck_yaml()
        recs = [r for r in recs if _num(r.get("ck/phase")) == code]
        if recs:
            phases[name] = (recs, cfg)
            cfgs[name] = cfg
    notes = ["smoke는 kd_ema.start_mb를 50으로 앞당겼다(본 학습 500). S4 ep2는 kd_ramp_epochs 0.01로 r≈1이다(본 학습은 "
             "ep 5→8 ramp).",
             "S4는 3 epoch × 200 mb를 처음부터 돌렸다. ep2 on-policy 후보는 학습 초기(약 400 mb) v2 후보라 teacher 분포 밖이다"
             "(report 45 §4-3). 본 학습 ep 5의 후보는 이보다 낫다 [추정].",
             "'Σ(가중 항) vs ck/loss 상대차'는 가중 항의 합이 ck/loss와 맞는지 보는 점검이다(0에 가까워야 한다).",
             "w_ema는 EMA(L_sur)/EMA(L_KD)이고 상한 10이다. smoke처럼 짧으면 상한에 붙을 수 있다."]
    if phases:
        t = kd_table(phases)
        rep["kd_table"] = t
        _write_json(root_p / "kd_loss_table.json", t)
        (root_p / "kd_loss_table.md").write_text(kd_table_md(t, cfgs, notes))
    s4 = stages.get("S4") or {}
    if s4.get("phase_timing") and s4.get("epoch_overhead"):
        ws = int(s4.get("ngpu") or 4)
        est = {"measured_world_size": ws}
        for key in ("wall_per_mb_mean", "sec_step_median"):
            est[key] = estimate_hours(s4["phase_timing"], s4["epoch_overhead"], world_size=4, key=key)
        rep["estimate_30ep"] = est
    _write_json(root_p / "smoke_report.json", rep)
    (root_p / "smoke_summary.md").write_text(smoke_summary_md(rep))
    return rep


def smoke_summary_md(rep: Dict[str, Any]) -> str:
    st = rep["stages"]
    L = ["# CK Phase 2 smoke 요약", "", f"root `{rep['root']}`, 생성 {rep['created_utc']} UTC. all_pass = {rep['all_pass']}.", "",
         "| 단계 | 통과 | 확인 항목 |", "|---|---|---|"]
    for k, v in st.items():
        ch = v.get("checks") or {}
        L.append(f"| {k} | {v.get('pass')} | " + ", ".join(f"{c} {'O' if ok else 'X'}" for c, ok in ch.items()) + " |")
    L += ["", "## step 시간·메모리 [실측]", "", "warm-up(epoch마다 처음 20 mb)을 뺀 값. wall = sec_step + sec_wait(데이터 대기·hook).",
          "", "| 실행 | GPU | phase | n | sec_step 중앙값 | wall/mb 평균 | peak mem GB |", "|---|---|---|---|---|---|---|"]
    for k in ("S2", "S3", "S4"):
        v = st.get(k) or {}
        ng = 1 if k != "S4" else v.get("ngpu", 4)
        for ph, d in (v.get("phase_timing") or {}).items():
            L.append(f"| {k} | {ng} | {ph} | {d['n']} | {_f(d['sec_step_median'])} | {_f(d['wall_per_mb_mean'])} | "
                     f"{_f(d['mem_gb_max'])} |")
    s4 = st.get("S4") or {}
    if s4.get("epoch_overhead"):
        o = s4["epoch_overhead"]
        L += ["", f"S4 epoch 고정비 [실측]: epoch 안 {_f(o.get('inside_overhead_s_median'))} s + epoch 사이 "
                  f"{_f(o.get('gap_s_median'))} s, 시작 {_f(o.get('startup_s'))} s."]
    if s4:
        L += ["", f"S4 labeler: 처리량 {_f(s4.get('labeler_tok_s'))} tok/s ({s4.get('labeler_tok_s_source')}), 학습 소비 "
                  f"{_f(s4.get('train_consume_tok_s'))} tok/s. gshare_ck 평균 {_f(s4.get('gshare_ck_mean'))}. watchdog: "
                  f"{json.dumps(s4.get('watchdog_test'), ensure_ascii=False)}"]
    est = rep.get("estimate_30ep")
    if est:
        L += ["", "## 30 epoch 예상 [추정]", "",
              f"S4 실측(DDP {est['measured_world_size']} GPU)의 phase별 s/mb × 4 GPU 기준 epoch당 "
              f"{est['wall_per_mb_mean'].get('mb_per_epoch')} mb, 일정 replay 0-3 / replay+기록 4 / on-policy 5-29.", "",
              "| 기준 | replay h | replay+기록 h | on-policy h | 고정비·시작 포함 합계 h |", "|---|---|---|---|---|"]
        for key, lab in (("wall_per_mb_mean", "wall/mb 평균(대기 포함)"), ("sec_step_median", "sec_step 중앙값")):
            e = est.get(key) or {}
            if "rows" in e:
                hs = {r["phase"]: r["hours"] for r in e["rows"]}
                L.append(f"| {lab} | {hs['replay']:.2f} | {hs['replay_record']:.2f} | {hs['onpolicy']:.2f} | "
                         f"{e['total_hours']:.2f} |")
        if est["measured_world_size"] != 4:
            L.append("")
            L.append(f"주의: S4를 {est['measured_world_size']} GPU로 쟀다. 4 GPU의 s/mb는 조금 더 클 수 있다 [추정].")
    return "\n".join(L) + "\n"


def _write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overrides")
    o.add_argument("--run-dir", required=True)
    o.add_argument("--gpus", default="0,1,2,3")
    o.add_argument("--workers", type=int, default=6)
    o.add_argument("--max-epochs", type=int, default=30)
    o.add_argument("--ck-set", action="append", default=[])
    o.add_argument("--ck-set-file", default="")
    o.add_argument("--extra-file", default="")
    o.add_argument("--resume-ckpt", default="")
    o.add_argument("--wandb", default="online", choices=["online", "offline", "disable"])
    o.add_argument("--out", required=True)
    g = sub.add_parser("gpu-check")
    g.add_argument("--gpus", required=True)
    g.add_argument("--max-used-mib", type=int, default=1024)
    g.add_argument("--list-only", action="store_true", help="only check the indices (no nvidia-smi)")
    pg = sub.add_parser("pick-gpus")
    pg.add_argument("--n", type=int, default=1)
    pg.add_argument("--among", default="1,2,3")
    pg.add_argument("--max-used-mib", type=int, default=1024)
    f = sub.add_parser("find-last")
    f.add_argument("--run-dir", required=True)
    sub.add_parser("wandb-mode")
    lh = sub.add_parser("labeler-health")
    lh.add_argument("--run-dir", required=True)
    lh.add_argument("--state", required=True)
    lh.add_argument("--stall-s", type=float, default=1200.0)
    s = sub.add_parser("status")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--json", action="store_true")
    t = sub.add_parser("smoke-tokens")
    t.add_argument("--n", type=int, default=240)
    t.add_argument("--out", required=True)
    t.add_argument("--extra-out", required=True)
    c = sub.add_parser("smoke-check")
    c.add_argument("--stage", required=True, choices=["S1", "S2", "S3", "S4", "S5"])
    c.add_argument("--run-dir", default="")
    c.add_argument("--out", required=True)
    c.add_argument("--rc", type=int, default=1)
    c.add_argument("--summary", default="")
    c.add_argument("--exit-a", default="")
    c.add_argument("--ngpu", type=int, default=4)
    c.add_argument("--mb-per-epoch", type=int, default=60)
    c.add_argument("--drain-dir", default="")
    r = sub.add_parser("smoke-report")
    r.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "overrides":
            sets = list(a.ck_set) + read_lines(a.ck_set_file)
            ov, ck = build_overrides(a.run_dir, parse_gpus(a.gpus), a.workers, a.max_epochs, sets,
                                     read_lines(a.extra_file), a.resume_ckpt or None, a.wandb)
            err = try_from_any(ck)
            if err:
                print(f"[launch_util] CKE2EConfig.from_any {err}", file=sys.stderr)
            outp = Path(a.out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            for o_ in ov:
                if "\n" in o_:
                    raise LaunchError(f"newline in override {o_!r}")
            outp.write_text("\n".join(ov) + "\n")
            eff = dict(ck)
            m = [x for x in ov if _okey(x) == "trainer.params.max_epochs"]
            eff["_trainer_max_epochs"] = int(m[-1].split("=", 1)[1]) if m else a.max_epochs
            _write_json(outp.parent / "ck_e2e_effective.json", eff)
        elif a.cmd == "gpu-check":
            if a.list_only:
                print(",".join(parse_gpus(a.gpus)))
            else:
                print(json.dumps(gpu_check(a.gpus.split(","), a.max_used_mib)))
        elif a.cmd == "pick-gpus":
            print(",".join(pick_gpus(a.n, a.among.split(","), a.max_used_mib)))
        elif a.cmd == "find-last":
            print(find_last_ckpt(a.run_dir))
        elif a.cmd == "wandb-mode":
            print(wandb_mode())
        elif a.cmd == "labeler-health":
            ok, why = labeler_health(a.run_dir, a.state, a.stall_s)
            print(why)
            return 0 if ok else 5
        elif a.cmd == "status":
            st = run_status(a.run_dir)
            print(json.dumps(st, indent=1, default=str)) if a.json else print_status(st)
        elif a.cmd == "smoke-tokens":
            toks, logs = smoke_tokens(a.n)
            _write_json(a.out, {"tokens": toks, "logs": logs, "n": len(toks)})
            Path(a.extra_out).write_text(f"scene_filter.tokens=[{','.join(toks)}]\n"
                                         f"scene_filter.log_names=[{','.join(logs)}]\n")
            print(f"{len(toks)} tokens from {len(logs)} logs")
        elif a.cmd == "smoke-check":
            res = check_stage(a.stage, a.run_dir or None, rc=a.rc, summary=a.summary,
                              exit_a=_num(a.exit_a), ngpu=a.ngpu, mb_per_epoch=a.mb_per_epoch,
                              drain_dir=a.drain_dir or None)
            _write_json(a.out, res)
            print(json.dumps({"stage": a.stage, "pass": res["pass"], "checks": res["checks"]}))
            return 0 if res["pass"] else 4
        elif a.cmd == "smoke-report":
            rep = smoke_report(a.root)
            print(json.dumps(rep["summary"], indent=1, default=str))
            print(f"all_pass={rep['all_pass']}")
    except LaunchError as e:
        print(f"[launch_util] REFUSED: {e.msg}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
