#!/usr/bin/env python
"""Settings -> one training run: canonical settings, run name, merged set file, merged extra file, environment, record.

compose() is a pure function of the settings, tools/release/registry.yaml, the checked-in recipe (ck_e2e2.yaml,
v2_v15_v7_arm.extra) and the roots of navsim/common/ssr_paths.py.  It writes nothing and needs no torch, no GPU and no
data on disk.  scripts/experiment.sh is its one caller; the CLI below writes what compose() returns.

  compose.py name KEY=VALUE ...                     print the run name
  compose.py emit --out-dir D [--run-root R] [--smoke] KEY=VALUE ...
                                                    write D/{refiner.set,arm.extra,env.sh,composition.json}, print the name
  compose.py verify --composition J --overrides F [--teachers-json T]
                                                    launcher output == prediction, teacher sha pins, reference diff
  compose.py reference --overrides F --run-dir R --plan-data P --teacher-cache T --data-root D
                                                    normalise a run's overrides into tools/release/reference/ form
  compose.py settings-args --settings-json F        the KEY=VALUE lines of a run's launch/settings.json
  compose.py record --composition J --out F [--asset-profile P --asset-exit N --asset-log L] [--extra-json S]
                                                    write launch/settings.json (settings + commit, host, versions)
  compose.py dirty-record --out-dir D               save the uncommitted diff into the run folder
  compose.py resume-check --composition J --settings-json F [--allow-commit-change]

Run name: only what differs from the defaults, in a fixed order, joined with _, then the hash (main_<hash6> by default):
  [<bb>] [<det>-<map>] [ref-off] [rkd-off|score|lat] [bkd-off] [detkd-off|offbev|offref] [mapkd-off|offbev|offref]
  [f1|f4 dist headfree z<a>_<b> r32] [bn-frozen] [e<N>] [g<N>] [s<N>] [<TAG>] [x] <hash6>
A setting implied by another one gets no part: REFINER=off is ref-off only, REFINER_KD=off BEV_KD=off is rkd-off_bkd-off.
hash6 = sha256 over the resolved overrides, minus run-path and wandb lines, with data roots written as placeholders so
one config has one name on every server, plus the registry pins of the teachers in use.  Never the commit.  A changed
default changes the overrides, so it changes the hash; launch/settings.json of a run has every setting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from navsim.common import ssr_paths as SP  # noqa: E402
from tools.ck.e2e import launch_util as LU  # noqa: E402
from tools.ck.e2e2 import launch_util2 as L2  # noqa: E402

REGISTRY = REPO / "tools/release/registry.yaml"
REFERENCE = REPO / "tools/release/reference/main_overrides.txt"
ARM_EXTRA = REPO / "tools/ck/e2e2/v2_v15_v7_arm.extra"
AGENT_YAML = REPO / "navsim/planning/script/config/common/agent/para_ssr_agent.yaml"
ANCHORS_REL = "resources/planning_vb/trajectory_anchors_256.npy"

# every setting and its default; scripts/experiment.sh has the same block at its top (a test compares them)
DEFAULTS: Dict[str, str] = {
    "REFINER": "on", "REFINER_KD": "on", "BEV_KD": "on", "DET_KD": "on", "MAP_KD": "on",
    "BACKBONE": "r34", "DET_TEACHER": "bevfusion", "MAP_TEACHER": "resmap",
    "FRAMES": "2,3", "UNDISTORT": "on", "HEADING": "xy", "PILLAR_Z": "-1,3", "BEV_RANGE": "64", "BN": "moving",
    "SEED": "0", "EPOCHS": "30", "GPUS": "0,1,2,3,4,5,6,7", "SAVE_EVERY": "1",
    "RUN_ROOT": "", "TAG": "", "WANDB": "offline", "EXTRA": "", "DRY": "0",
}
CHOICES: Dict[str, Tuple[str, ...]] = {
    "REFINER": ("on", "off"), "REFINER_KD": ("on", "off", "score", "lat"), "BEV_KD": ("on", "off"),
    "DET_KD": ("on", "off", "off_bev", "off_ref"), "MAP_KD": ("on", "off", "off_bev", "off_ref"),
    "FRAMES": ("2,3", "3", "0,1,2,3"), "UNDISTORT": ("on", "off"), "HEADING": ("xy", "free"),
    "BEV_RANGE": ("64", "32"), "BN": ("moving", "frozen"), "WANDB": ("online", "offline", "disable"), "DRY": ("0", "1"),
}
# settings that describe where or how a run is launched, not what it trains
NOT_RECIPE = ("GPUS", "SAVE_EVERY", "RUN_ROOT", "TAG", "WANDB", "DRY")
GPU_COUNTS = (2, 4, 8)
NAME_DEFAULT = "main"                      # the run name's first part when no setting differs from the defaults
NAME_SMOKE = "smoke"                       # prefix of a smoke run (scripts/experiment.sh looks for smoke_*)
# the run name is what people read: no old branch words in it.  One part of the name (split on _ . -), any case: ck, ck2, ck2e2e,
# v2, v2f, v2final, v7c, v15x.  ckpt and v20 are other words.
NAME_REFUSED_PART = re.compile(r"(?i)(ck|ck[0-9].*|v(2|7|15)|v(2|7|15)[a-z].*)")
KD_KEYS_OF ={"det": ["nc", "ep", "ttc"], "map": ["dac", "ep"]}   # kd_calib.fit_keys: a single teacher teaches its own keys
ONPOLICY_FROM_EPOCH = 5           # ck_e2e2.yaml; the launcher repeats this refusal with the merged value
# Refused in EXTRA: the locked values of the release (docs/EXPERIMENTS.md).
EXTRA_REFUSED = ("infer_select", "submit_selection", "kd_calib", "teacher_", "warmup.", "distill_lambda_", "distill_tau_",
                 "plan_anchor_file", "dataloader.params.batch_size", "io_dir", "output_dir", "seed",
                 "use_task_interaction", "use_det_motion_head", "use_map_head", "map_head_type", "bev_seg_weight")
# Keys a setting owns: through EXTRA the run name would say something else than the run does.
EXTRA_OWNED = {
    "agent.config.ck_e2e2.enabled": "REFINER", "agent.config.ck_e2e2.use_teachers": "REFINER_KD",
    "agent.config.ck_e2e2.lambda_kd_score": "REFINER_KD", "agent.config.ck_e2e2.lat_kd": "REFINER_KD",
    "agent.config.ck_e2e2.kd_sources": "DET_KD / MAP_KD", "agent.config.ck_e2e2.kd_score_keys": "DET_KD / MAP_KD",
    "agent.config.distill_off_banks": "DET_KD / MAP_KD", "agent.config.use_distill": "BEV_KD",
    "agent.config.image_architecture": "BACKBONE", "agent.config.frozen_stages": "BACKBONE",
    "agent.config.backbone_out_indices": "BACKBONE", "agent.config.distill_branches": "DET_TEACHER / MAP_TEACHER",
    "agent.config.distill_norm_files": "DET_TEACHER / MAP_TEACHER",
    "agent.config.frame_indices": "FRAMES", "agent.config.undistort": "UNDISTORT",
    "agent.config.plan_heading_from_xy": "HEADING", "agent.config.pc_range": "PILLAR_Z / BEV_RANGE",
    "agent.config.map_pc_range": "PILLAR_Z / BEV_RANGE", "agent.config.bev_h": "BEV_RANGE",
    "agent.config.bev_w": "BEV_RANGE", "agent.config.backbone_bn_mode": "BN", "agent.config.max_epochs": "EPOCHS",
    "trainer.params.max_epochs": "EPOCHS", "trainer.params.devices": "GPUS", "checkpoint.every_n_epochs": "SAVE_EVERY",
}
# overrides that do not enter hash6: run paths, wandb, how often to save, and roots that are lines only on some servers
HASH_SKIP = ("experiment_name", "output_dir", "agent.config.ck_e2e2.io_dir", "resume_checkpoint",
             "checkpoint.every_n_epochs", "agent.config.ref_data_root", "agent.config.distill_v7_root")
# variables this script's launches set; a value inherited from the calling shell must not reach them
CONTROL_ENV = ("ARM", "RUN", "E2E_ROOT", "E2E_CK_SET", "E2E_EXTRA", "E2E_NO_LABELER", "E2E_NO_WATCHDOG", "E2E_SMOKE",
               "E2E_SMOKE_PREFIX", "E2E_DRY_RUN", "E2E_WANDB", "E2E_NO_DEEP_TEACHER_CHECK", "MAX_EPOCHS",
               "LABELER_MAX_EPOCH", "TRAIN_WORKERS", "CK_BEV64", "CK_ALLOW_GPU45", "CK_ALLOW_GPU4567", "SSR_ALLOW_DIRECT",
               "SSR_EXPERIMENT_SHA", "SSR_GPU_POLICY", "SSR_ALLOW_FILE_CALIB", "SSR_DET_CACHE_TRAIN", "SSR_DET_CACHE_VAL",
               "SSR_DET_SHA_HEAD", "SSR_MAP_CACHE_ROOT", "SSR_MAP_SHA256", "SSR_MAP_REQUIRE_SEG")


class Refused(ValueError):
    """A setting or a combination that cannot run; the message says why."""


# ----------------------------------------------------------------------------------------------- registry
def load_registry(path: Optional[str] = None) -> Dict[str, Any]:
    import yaml
    p = Path(path or os.environ.get("SSR_RELEASE_REGISTRY") or REGISTRY)
    if not p.is_file():
        raise Refused(f"registry {p} not found")
    d = yaml.safe_load(p.read_text())
    if not isinstance(d, dict) or "backbones" not in d or "teachers" not in d:
        raise Refused(f"registry {p}: needs backbones and teachers")
    return d


def _expand(value: Any, roots: Mapping[str, str]) -> Any:
    """${NAME} -> root value, in the strings of a registry entry."""
    if isinstance(value, str):
        def sub(m):
            if m.group(1) not in roots:
                raise Refused(f"registry: unknown root ${{{m.group(1)}}} in {value!r}")
            return roots[m.group(1)]
        return re.sub(r"\$\{([A-Z0-9_]+)\}", sub, value)
    if isinstance(value, dict):
        return {k: _expand(v, roots) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, roots) for v in value]
    return value


READY_FIELDS = {"det": ("layout", "cache_name", "cache_train", "cache_val", "sha_head", "norm_v15", "refiner_run",
                        "refiner_ckpt_sha16"),
                "map": ("layout", "cache_name", "cache_root", "sha256", "norm_v15", "refiner_run", "refiner_ckpt_sha16")}
LAYOUT_OF = {"det": "per_token_npz", "map": "sharded"}


def teacher_entry(reg: Mapping[str, Any], role: str, name: str) -> Dict[str, Any]:
    """The registry entry of a ready teacher; Refused for anything else, listing what is missing."""
    known = reg["teachers"].get(role) or {}
    if name not in known:
        raise Refused(f"{role.upper()}_TEACHER={name}: not in the registry ({', '.join(known)})")
    e = known[name]
    if e.get("status") != "ready":
        needs = "; ".join(str(x) for x in e.get("needs") or ["(the entry does not say)"])
        raise Refused(f"{role.upper()}_TEACHER={name}: status {e.get('status')}. Missing: {needs}")
    absent = [k for k in READY_FIELDS[role] if not e.get(k)]
    if absent:
        raise Refused(f"{role.upper()}_TEACHER={name}: registry entry lacks {', '.join(absent)}")
    for k in ("sha_head", "sha256", "refiner_ckpt_sha16"):
        if k in e and not re.fullmatch(r"[0-9a-f]{16,64}", str(e[k]) if isinstance(e[k], str) else ""):
            raise Refused(f"{role.upper()}_TEACHER={name}: {k} {e[k]!r} is not a quoted hex sha")
    if e["layout"] != LAYOUT_OF[role]:
        raise Refused(f"{role.upper()}_TEACHER={name}: layout {e['layout']}: the {role} reader needs {LAYOUT_OF[role]}")
    return e


def default_teacher(role: str) -> str:
    return DEFAULTS["DET_TEACHER" if role == "det" else "MAP_TEACHER"]


# ----------------------------------------------------------------------------------------------- canonical settings
def _z_text(x: float) -> str:
    return f"{int(x)}" if float(x).is_integer() else repr(float(x))


def parse_pillar_z(text: str) -> Tuple[float, float]:
    parts = str(text).split(",")
    try:
        a, b = float(parts[0]), float(parts[1])
        if len(parts) != 2 or not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError
    except (ValueError, IndexError):
        raise Refused(f"PILLAR_Z={text}: expected a,b in metres") from None
    if a >= b:
        raise Refused(f"PILLAR_Z={text}: a must be below b")
    return a, b


def extra_lines_of(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(";") if x.strip()]


def check_extra(text: str) -> None:
    for line in extra_lines_of(text):
        for bad in EXTRA_REFUSED:
            if bad in line:
                raise Refused(f"EXTRA line {line!r}: {bad!r} is locked (docs/EXPERIMENTS.md)")
        if "=" not in line and not line.startswith("~"):
            raise Refused(f"EXTRA line {line!r}: expected key=value")
        key = LU._okey(line)
        for owned, setting in EXTRA_OWNED.items():
            if key == owned or key.startswith(owned + "."):
                raise Refused(f"EXTRA line {line!r}: use {setting}, so the run name stays true")


def _int(key: str, value: str, low: int) -> str:
    if not re.fullmatch(r"[0-9]+", str(value)) or int(value) < low:
        raise Refused(f"{key}={value}: expected an integer >= {low}")
    return str(int(value))


def canonicalize(raw: Mapping[str, str], reg: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """Filled, checked settings where one config has one spelling; Refused says what is wrong.  Idempotent."""
    reg = load_registry() if reg is None else reg
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise Refused(f"unknown setting(s) {', '.join(unknown)}")
    s = {k: str(raw.get(k, v)) for k, v in DEFAULTS.items()}
    for k, allowed in CHOICES.items():
        if s[k] not in allowed:
            raise Refused(f"{k}={s[k]}: one of {' | '.join(allowed)}")
    s["SEED"] = _int("SEED", s["SEED"], 0)
    s["EPOCHS"] = _int("EPOCHS", s["EPOCHS"], 1)
    s["SAVE_EVERY"] = _int("SAVE_EVERY", s["SAVE_EVERY"], 1)
    if int(s["EPOCHS"]) % int(s["SAVE_EVERY"]):
        raise Refused(f"SAVE_EVERY={s['SAVE_EVERY']} does not divide EPOCHS={s['EPOCHS']}: the last epoch, the one eval "
                      f"reads, would have no checkpoint")
    gpus = [g.strip() for g in s["GPUS"].split(",") if g.strip()]
    if not all(g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus) or len(gpus) not in GPU_COUNTS:
        raise Refused(f"GPUS={s['GPUS']}: {' | '.join(map(str, GPU_COUNTS))} distinct GPU indices")
    s["GPUS"] = ",".join(gpus)
    # a tag can be the first part of the run name
    if s["TAG"] and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", s["TAG"]):
        raise Refused(f"TAG={s['TAG']}: characters A-Za-z0-9.- only, a letter or digit first")
    if s["TAG"] in (NAME_DEFAULT, NAME_SMOKE):
        raise Refused(f"TAG={s['TAG']}: the run name would read as the default recipe or as a smoke run")
    if s["RUN_ROOT"] and (not os.path.isabs(s["RUN_ROOT"]) or "=" in s["RUN_ROOT"]):
        raise Refused(f"RUN_ROOT={s['RUN_ROOT']}: an absolute path without '='")
    a, b = parse_pillar_z(s["PILLAR_Z"])
    s["PILLAR_Z"] = f"{_z_text(a)},{_z_text(b)}"
    bb = (reg["backbones"] or {}).get(s["BACKBONE"])
    if bb is None:
        raise Refused(f"BACKBONE={s['BACKBONE']}: not in the registry ({', '.join(reg['backbones'])})")
    if bb.get("status") == "refused":
        raise Refused(f"BACKBONE={s['BACKBONE']}: {bb.get('reason')}")
    teacher_entry(reg, "det", s["DET_TEACHER"])
    teacher_entry(reg, "map", s["MAP_TEACHER"])

    # refiner / KD: one config, one spelling
    if s["REFINER"] == "off":
        s["REFINER_KD"] = "off"
    no_ref, no_bev = s["REFINER_KD"] == "off", s["BEV_KD"] == "off"
    if no_ref and no_bev:
        s["DET_KD"] = s["MAP_KD"] = "off"
    else:
        if s["DET_KD"] == "off" and s["MAP_KD"] == "off":
            raise Refused("DET_KD=off MAP_KD=off with a KD still on: use REFINER_KD=off BEV_KD=off")
        for k in ("DET_KD", "MAP_KD"):
            if no_ref and s[k] == "off_ref":
                raise Refused(f"{k}=off_ref means nothing: the refiner KD is off (REFINER_KD=off or REFINER=off)")
            if no_bev and s[k] == "off_bev":
                raise Refused(f"{k}=off_bev means nothing: BEV_KD=off")
            if s[k] == "off":
                s[k] = "off_bev" if no_ref else ("off_ref" if no_bev else "off")
        if not no_bev and all(s[k] in ("off", "off_bev") for k in ("DET_KD", "MAP_KD")):
            raise Refused("both BEV KD banks off: use BEV_KD=off")
        if not no_ref and all(s[k] in ("off", "off_ref") for k in ("DET_KD", "MAP_KD")):
            raise Refused("both refiner KD sources off: use REFINER_KD=off")

    # a teacher nothing reads is not part of the config
    for role, tk, kd in (("det", "DET_TEACHER", "DET_KD"), ("map", "MAP_TEACHER", "MAP_KD")):
        if not teacher_read(s, kd):
            s[tk] = default_teacher(role)

    # student
    if s["BEV_RANGE"] == "32" and (s["REFINER"] == "on" or s["BEV_KD"] == "on"):
        raise Refused("BEV_RANGE=32 only with REFINER=off BEV_KD=off: the 32 m teacher caches no longer exist")
    if s["REFINER"] == "on" and int(s["EPOCHS"]) <= ONPOLICY_FROM_EPOCH:
        raise Refused(f"EPOCHS={s['EPOCHS']}: the refiner needs more than {ONPOLICY_FROM_EPOCH} (on-policy labels start "
                      f"at epoch {ONPOLICY_FROM_EPOCH})")
    check_extra(s["EXTRA"])
    s["EXTRA"] = ";".join(extra_lines_of(s["EXTRA"]))
    return s


def teacher_read(s: Mapping[str, str], kd_key: str) -> bool:
    """Something reads this teacher's cache: the refiner with teachers loads both, a BEV KD bank that is on reads its own."""
    if s["REFINER"] == "on" and s["REFINER_KD"] != "off":
        return True
    return s["BEV_KD"] == "on" and s[kd_key] not in ("off", "off_bev")


def profile_of(s: Mapping[str, str]) -> str:
    """verify_assets profile a run needs."""
    if s["BEV_KD"] == "off" and s["REFINER"] == "off":
        return "base"
    if s["BEV_KD"] == "off" and s["REFINER_KD"] == "off":
        return "refiner_noteacher"
    return "full"


def is_default_recipe(s: Mapping[str, str]) -> bool:
    """The main recipe on the reference GPU count: what tools/release/reference/main_overrides.txt describes."""
    return all(s[k] == v for k, v in DEFAULTS.items() if k not in NOT_RECIPE) and len(s["GPUS"].split(",")) == 8


# ----------------------------------------------------------------------------------------------- the run's lines
def _st_parts(s: Mapping[str, str]) -> List[str]:
    p: List[str] = []
    if s["FRAMES"] != DEFAULTS["FRAMES"]:
        p.append("f1" if s["FRAMES"] == "3" else "f4")
    if s["UNDISTORT"] == "off":
        p.append("dist")
    if s["HEADING"] == "free":
        p.append("headfree")
    if s["PILLAR_Z"] != DEFAULTS["PILLAR_Z"]:
        a, b = s["PILLAR_Z"].split(",")
        p.append(f"z{a}_{b}".replace(".", "p"))
    if s["BEV_RANGE"] == "32":
        p.append("r32")
    return p


def _teacher_kd_part(s: Mapping[str, str], key: str, label: str) -> List[str]:
    """The name part of DET_KD / MAP_KD on canonical settings: nothing when on, nothing when no KD runs at all."""
    if s[key] == "on" or (s["REFINER_KD"] == "off" and s["BEV_KD"] == "off"):
        return []
    return [f"{label}-{s[key].replace('_', '')}"]


def run_name(s: Mapping[str, str], hash6: str) -> str:
    parts: List[str] = []
    if s["BACKBONE"] != DEFAULTS["BACKBONE"]:
        parts.append(s["BACKBONE"])
    if s["DET_TEACHER"] != DEFAULTS["DET_TEACHER"] or s["MAP_TEACHER"] != DEFAULTS["MAP_TEACHER"]:
        parts.append(f"{s['DET_TEACHER']}-{s['MAP_TEACHER']}")
    if s["REFINER"] == "off":
        parts.append("ref-off")           # the refiner KD goes with it
    elif s["REFINER_KD"] != "on":
        parts.append(f"rkd-{s['REFINER_KD']}")
    if s["BEV_KD"] == "off":
        parts.append("bkd-off")
    parts += _teacher_kd_part(s, "DET_KD", "detkd") + _teacher_kd_part(s, "MAP_KD", "mapkd")
    parts += _st_parts(s)
    if s["BN"] != DEFAULTS["BN"]:
        parts.append(f"bn-{s['BN']}")
    if s["EPOCHS"] != DEFAULTS["EPOCHS"]:
        parts.append(f"e{s['EPOCHS']}")
    if len(s["GPUS"].split(",")) != len(DEFAULTS["GPUS"].split(",")):
        parts.append(f"g{len(s['GPUS'].split(','))}")
    if s["SEED"] != DEFAULTS["SEED"]:
        parts.append(f"s{s['SEED']}")
    if s["TAG"]:
        parts.append(s["TAG"])
    if s["EXTRA"]:
        parts.append("x")
    name = "_".join((parts or [NAME_DEFAULT]) + [hash6])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise Refused(f"run name {name!r} has a character outside A-Za-z0-9_.-")
    bad = sorted({p for p in re.split(r"[_.-]", name) if NAME_REFUSED_PART.fullmatch(p)})
    if bad:
        raise Refused(f"run name {name!r} has the part(s) {', '.join(bad)}: choose another TAG or registry name")
    return name


def _kd_lists(s: Mapping[str, str]) -> Tuple[List[str], List[str]]:
    """(BEV KD banks off, refiner KD sources), from the canonical DET_KD / MAP_KD."""
    bev_off = [r for r, k in (("det", "DET_KD"), ("map", "MAP_KD")) if s[k] in ("off", "off_bev")]
    src = [r for r, k in (("det", "DET_KD"), ("map", "MAP_KD")) if s[k] not in ("off", "off_ref")]
    return bev_off, src


def set_lines(s: Mapping[str, str], roots: Mapping[str, str], teachers: Mapping[str, Dict[str, Any]],
              smoke: bool = False) -> List[str]:
    """Lines of the merged set file (ck_e2e2 key=value)."""
    data = roots["SSR_DATA_ROOT"]
    lines = [f"warmup.packed={data}/ck/packed/navtrain_train", f"warmup.raw_labels={data}/ck/labels/navtrain_train/raw256",
             f"warmup.var_dir={data}/ck/ck2/labels/navtrain_train/var_separate_sampler16_accstraight",
             f"seed={s['SEED']}"]
    if s["REFINER"] == "off":
        return ["enabled=false"] + lines
    if s["REFINER_KD"] == "off":     # tools/ck/e2e2/v2_nokd.set
        lines += ["use_teachers=false", "teacher_det_run=null", "teacher_map_run=null", "lambda_kd_score=0.0",
                  "lat_kd.balance=fixed", "lat_kd.fixed=0.0", "kd_calib.enabled=false"]
    else:
        lines += [f"teacher_det_run={teachers['det']['refiner_run']}", f"teacher_map_run={teachers['map']['refiner_run']}"]
        _, src = _kd_lists(s)
        if len(src) == 1:            # a single teacher teaches its own keys and its own lateral offset
            lines += [f"kd_sources=[{src[0]}]", f"kd_score_keys=[{','.join(KD_KEYS_OF[src[0]])}]"]
        if s["REFINER_KD"] == "score":
            lines += ["lat_kd.balance=fixed", "lat_kd.fixed=0.0"]
        if s["REFINER_KD"] == "lat":  # the Platt calibration acts on score KD logits only: nothing would read it
            lines += ["lambda_kd_score=0.0", "kd_calib.enabled=false"]
    if smoke:                        # scripts/training/smoke_v2_final.sh
        lines += ["record_from_epoch=0", "record_until_epoch=0", "onpolicy_from_epoch=1", "label_refresh_every_mb=20"]
        if s["REFINER_KD"] in ("on", "lat"):
            lines.append("lat_kd.start_mb=30")
        if s["REFINER_KD"] in ("on", "score"):
            lines.append("kd_calib.min_n=50")
    return lines


def _grid(s: Mapping[str, str]) -> str:
    a, b = parse_pillar_z(s["PILLAR_Z"])
    return f"[-32.0,0.0,{a!r},32.0,{'32.0' if s['BEV_RANGE'] == '32' else '64.0'},{b!r}]"


def extra_lines(s: Mapping[str, str], roots: Mapping[str, str], teachers: Mapping[str, Dict[str, Any]],
                reg: Mapping[str, Any], smoke: bool = False) -> List[str]:
    """Lines of the merged extra file (Hydra overrides, merged by key into the launcher recipe)."""
    ng = len(s["GPUS"].split(","))
    x: Dict[str, str] = {}

    def put(line: str) -> None:
        x[LU._okey(line)] = line

    bev = s["BEV_KD"] == "on"
    if bev:                          # v15 probe schedule in per-rank micro-batches (train_v2_v15_v7.sh)
        put(f"++agent.config.distill_probe_warmup_steps={20 if smoke else 21200 // ng}")
        put(f"++agent.config.distill_probe_ramp_steps={20 if smoke else 10600 // ng}")
    for line in LU.read_lines(str(ARM_EXTRA)):
        key = LU._okey(line)
        if key.startswith("wandb."):
            continue
        if not bev and (key == "agent.config.use_distill" or key.startswith("agent.config.distill_")):
            continue
        put(line)
    put(f"++agent.config.use_distill={'true' if bev else 'false'}")
    grid = _grid(s)
    put(f"++agent.config.pc_range={grid}")
    put(f"++agent.config.map_pc_range={grid}")
    if s["BEV_RANGE"] == "32":
        put("++agent.config.bev_h=50")
    if s["FRAMES"] != DEFAULTS["FRAMES"]:
        put(f"++agent.config.frame_indices=[{s['FRAMES']}]")
    if s["UNDISTORT"] == "off":
        put("++agent.config.undistort=false")
    if s["HEADING"] == "free":
        put("agent.config.plan_heading_from_xy=false")
    bb = reg["backbones"][s["BACKBONE"]]
    if s["BACKBONE"] != DEFAULTS["BACKBONE"]:
        put(f"agent.config.image_architecture={bb['timm']}")
    for k, v in (bb.get("flags") or {}).items():
        put(f"++agent.config.{k}={LU.hydra_value(v)}")
    if s["SAVE_EVERY"] != "1" and not smoke:
        put(f"checkpoint.every_n_epochs={s['SAVE_EVERY']}")
    put(f"seed={s['SEED']}")
    put(f"++agent.config.backbone_bn_mode={s['BN']}")
    if s["SEED"] != "0":
        put(f"++agent.config.distill_probe_seed={s['SEED']}")
    if bev:
        off, _ = _kd_lists(s)
        put(f"++agent.config.distill_feature_root={roots['SSR_TEACHER_CACHE']}")
        put(f"++agent.config.distill_off_banks=[{','.join(off)}]")
        banks = [(b, r) for b, r in (("bevfusion", "det"), ("resmap", "map")) if r not in off]
        put("++agent.config.distill_norm_files={" + ",".join(f"{b}:{LU.hydra_value(teachers[r]['norm_v15'])}"
                                                             for b, r in banks) + "}")
        if s["DET_TEACHER"] != DEFAULTS["DET_TEACHER"] or s["MAP_TEACHER"] != DEFAULTS["MAP_TEACHER"]:
            # one line would replace the whole dict: both roles (branch names are role names, tools/release/registry.yaml)
            reg_t = reg["teachers"]
            put(f"++agent.config.distill_branches.bevfusion.cache_name={reg_t['det'][s['DET_TEACHER']]['cache_name']}")
            put(f"++agent.config.distill_branches.resmap.cache_name={reg_t['map'][s['MAP_TEACHER']]['cache_name']}")
    # the agent yaml's literal roots: lines only when this server's root is another one
    for key, value in (("ref_data_root", roots["SSR_DATA_ROOT"]), ("distill_v7_root", roots["SSR_TEACHER_CACHE"])):
        if _agent_yaml_value(key) != value:
            put(f"++agent.config.{key}={value}")
    if smoke:                        # scripts/training/smoke_v2_final.sh
        put("trainer.params.limit_train_batches=60")
        put("agent.config.grad_balance_warmup_iters=20")
        put("agent.config.grad_balance_interval=10")
        put("agent.config.grad_norm_log_interval=10")
    for line in extra_lines_of(s["EXTRA"]):
        put(line)
    return list(x.values())


_AGENT_YAML_CACHE: Dict[str, Any] = {}


def _agent_yaml_value(key: str) -> Any:
    if not _AGENT_YAML_CACHE:
        import yaml
        _AGENT_YAML_CACHE.update(yaml.safe_load(AGENT_YAML.read_text())["config"])
    return _AGENT_YAML_CACHE.get(key)


def _env_lines(s: Mapping[str, str], teachers: Mapping[str, Dict[str, Any]], run: str, run_root: str, hash6: str,
               smoke: bool) -> Dict[str, str]:
    env = {"RUN": run, "E2E_ROOT": run_root, "ARM": "main", "CK_BEV64": "0" if s["BEV_RANGE"] == "32" else "1",
           "SSR_EXPERIMENT_SHA": hash6, "CK2_E2E_APPROVED": "1", "SSR_GPU_POLICY": "any",
           "MAX_EPOCHS": "2" if smoke else s["EPOCHS"], "E2E_WANDB": s["WANDB"]}
    if s["REFINER"] == "off":
        env["E2E_NO_LABELER"] = "1"
    if smoke:
        env.update(E2E_SMOKE="1", E2E_SMOKE_PREFIX=run_root.rstrip("/") + "/", LABELER_MAX_EPOCH="1")
    d, m = teachers.get("det"), teachers.get("map")
    if d is not None:
        env.update(SSR_DET_CACHE_TRAIN=d["cache_train"], SSR_DET_CACHE_VAL=d["cache_val"], SSR_DET_SHA_HEAD=d["sha_head"])
    if m is not None:
        env.update(SSR_MAP_CACHE_ROOT=m["cache_root"], SSR_MAP_SHA256=m["sha256"])
        if m.get("require_seg") is False:
            env["SSR_MAP_REQUIRE_SEG"] = "0"
    return env


# ----------------------------------------------------------------------------------------------- hash + reference
def normalize_line(line: str, roots: Mapping[str, str], run_dir: Optional[str] = None, repo: str = str(REPO)) -> str:
    """Absolute paths of this server -> ${NAME}, longest first, on whole path components."""
    subs = {"SSR_DATA_ROOT": roots["SSR_DATA_ROOT"], "SSR_TEACHER_CACHE": roots["SSR_TEACHER_CACHE"],
            "SSR_PLAN_DATA": roots["SSR_PLAN_DATA"], "REPO": repo}
    if run_dir:
        subs["RUN_DIR"] = run_dir
    for name, path in sorted(subs.items(), key=lambda kv: -len(kv[1])):
        line = re.sub(re.escape(path.rstrip("/")) + r"(?![A-Za-z0-9_.-])", "${" + name + "}", line)
    return line


def _key(line: str) -> str:
    k = LU._okey(line)
    return ("~" + k) if line.startswith("~") else k


def hash_lines(overrides: Sequence[str], roots: Mapping[str, str]) -> List[str]:
    return [normalize_line(o, roots) for o in overrides
            if LU._okey(o) not in HASH_SKIP and not LU._okey(o).startswith("wandb.")]


def _sha(lines: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


# the only lines of the main recipe allowed to differ from the reference run (docs/EXPERIMENTS.md section 4)
_RUN_DERIVED = ("experiment_name", "output_dir", "agent.config.ck_e2e2.io_dir")
_CK = "++agent.config.ck_e2e2."
ALLOWED_CHANGED = {
    "agent.config.ck_e2e2.kd_calib.det": _CK + "kd_calib.det=null",
    "agent.config.ck_e2e2.kd_calib.map": _CK + "kd_calib.map=null",
    "agent.config.plan_anchor_file": "agent.config.plan_anchor_file=${REPO}/" + ANCHORS_REL,
    "agent.config.plan_score_file": "agent.config.plan_score_file=${SSR_DATA_ROOT}/ck/raw256/pdm_score_256_officialEP",
}
ALLOWED_NEW = {
    "seed": "seed=0",
    "agent.config.backbone_bn_mode": "++agent.config.backbone_bn_mode=moving",
    "agent.config.distill_off_banks": "++agent.config.distill_off_banks=[]",
    "agent.config.distill_norm_files": "++agent.config.distill_norm_files={bevfusion:resources/km_corridor/"
                                       "norm_bevfusion_100x100.npz,resmap:resources/km_corridor/norm_resmap_100x100.npz}",
    "agent.config.ck_e2e2.use_teachers": _CK + "use_teachers=true",
    "agent.config.ck_e2e2.kd_sources": _CK + "kd_sources=[det,map]",
    "agent.config.ck_e2e2.submit_selection": _CK + "submit_selection=false",
    "agent.config.ck_e2e2.kd_calib.mode": _CK + "kd_calib.mode=online",
    "agent.config.ck_e2e2.kd_calib.refits_per_epoch": _CK + "kd_calib.refits_per_epoch=10",
    "agent.config.ck_e2e2.kd_calib.ema_epochs": _CK + "kd_calib.ema_epochs=1.0",
    "agent.config.ck_e2e2.kd_calib.bins": _CK + "kd_calib.bins=4096",
    "agent.config.ck_e2e2.kd_calib.zmax": _CK + "kd_calib.zmax=20.0",
    "agent.config.ck_e2e2.kd_calib.min_n": _CK + "kd_calib.min_n=2000.0",
    "agent.config.ck_e2e2.kd_calib.reset_onpolicy": _CK + "kd_calib.reset_onpolicy=true",
}
# keys that may appear or differ for a reason outside the recipe: how often to save, roots on another server
ALLOWED_ANY = ("checkpoint.every_n_epochs", "agent.config.ref_data_root", "agent.config.distill_v7_root")


def reference_diff(overrides: Sequence[str], roots: Mapping[str, str], run_dir: str, repo: str = str(REPO),
                   ref_path: Path = REFERENCE) -> Tuple[bool, List[str]]:
    """Main recipe overrides against the reference run: only the ALLOWED_ lines may differ.  -> (pass, findings)."""
    ours = {_key(o): normalize_line(o, roots, run_dir, repo) for o in overrides}
    ref = {_key(o): o for o in LU.read_lines(str(ref_path))}
    bad: List[str] = []

    def allowed_any(k: str) -> bool:
        return k in _RUN_DERIVED or k.startswith("wandb.") or k in ALLOWED_ANY

    for k, line in ref.items():
        if allowed_any(k):
            continue
        if k not in ours:
            bad.append(f"missing  {line}")
        elif ours[k] != line and ours[k] != ALLOWED_CHANGED.get(k):
            bad.append(f"changed  {line}  ->  {ours[k]}")
    for k, line in ours.items():
        if k in ref or allowed_any(k):
            continue
        if ALLOWED_NEW.get(k) != line:
            bad.append(f"new      {line}")
    common = [k for k in ref if k in ours and not allowed_any(k)]
    if common != [k for k in ours if k in ref and not allowed_any(k)]:
        bad.append("order of the common keys differs from the reference")
    return not bad, bad


# ----------------------------------------------------------------------------------------------- compose
@dataclass
class Composition:
    settings: Dict[str, str]
    run: str
    hash6: str
    run_root: str
    run_dir: str
    set_lines: List[str]
    extra_lines: List[str]
    env: Dict[str, str]
    overrides: List[str]
    pins: Dict[str, Any]
    profile: str
    smoke: bool = False
    default_recipe: bool = False
    roots: Dict[str, str] = field(default_factory=dict)

    def record(self) -> Dict[str, Any]:
        """The part of settings.json that needs nothing but the settings."""
        return {"run": self.run, "hash6": self.hash6, "smoke": self.smoke, "settings": self.settings,
                "run_root": self.run_root, "overrides_sha256": _sha(self.overrides), "teacher_pins": self.pins,
                "profile": self.profile, "registry": str(os.environ.get("SSR_RELEASE_REGISTRY") or REGISTRY)}


def _resolved(s, roots, teachers, reg, run_dir, smoke) -> Tuple[List[str], List[str], List[str]]:
    sets = set_lines(s, roots, teachers, smoke)
    extra = extra_lines(s, roots, teachers, reg, smoke)
    ng = len(s["GPUS"].split(","))
    ck = L2.effective_ck(run_dir, ng, sets, extra)
    ov = L2.merge_overrides(run_dir, ng, ck, 6, 2 if smoke else int(s["EPOCHS"]), extra, None, s["WANDB"])
    return sets, extra, ov


def compose(raw: Mapping[str, str], registry: Optional[str] = None, run_root: Optional[str] = None, smoke: bool = False,
            env: Optional[Mapping[str, str]] = None) -> Composition:
    """raw settings -> everything a launch needs.  run_root replaces RUN_ROOT for this composition (the check folder);
    the name never depends on it."""
    reg = load_registry(registry)
    s = canonicalize(raw, reg)
    roots = SP.resolve(os.environ if env is None else env)
    reg_x = {"backbones": reg["backbones"],
             "teachers": {r: {n: _expand(e, roots) for n, e in (reg["teachers"].get(r) or {}).items()}
                          for r in ("det", "map")}}
    used: Dict[str, Dict[str, Any]] = {}
    for role, tk, kd in (("det", "DET_TEACHER", "DET_KD"), ("map", "MAP_TEACHER", "MAP_KD")):
        if teacher_read(s, kd):
            used[role] = teacher_entry(reg_x, role, s[tk])
    pins = {}
    for role, e in used.items():
        raw_e = reg["teachers"][role][s["DET_TEACHER" if role == "det" else "MAP_TEACHER"]]
        pins[role] = {"name": s["DET_TEACHER" if role == "det" else "MAP_TEACHER"], "cache_name": e["cache_name"],
                      "sha": e.get("sha_head") or e.get("sha256"), "refiner_ckpt_sha16": e["refiner_ckpt_sha16"],
                      "norm_v15": raw_e["norm_v15"]}
    # the name comes from the full recipe, also for a smoke of it, on any root
    _, _, ov0 = _resolved(s, roots, used, reg_x, "/_run_root_/_run_", False)
    hash6 = hashlib.sha256(json.dumps({"overrides": hash_lines(ov0, roots), "pins": pins}, sort_keys=True).encode()
                           ).hexdigest()[:6]
    name = run_name(s, hash6)
    root = (run_root or s["RUN_ROOT"] or roots["SSR_RUN_ROOT"]).rstrip("/")
    if smoke:
        name = f"{NAME_SMOKE}_{name}"
        root = root if run_root else root + "/_smoke"
        s = dict(s, WANDB="disable")
    run_dir = f"{root}/{name}"
    sets, extra, ov = _resolved(s, roots, used, reg_x, run_dir, smoke)
    return Composition(settings=s, run=name, hash6=hash6, run_root=root, run_dir=run_dir, set_lines=sets, extra_lines=extra,
                       env=_env_lines(s, used, name, root, hash6, smoke), overrides=ov, pins=pins, profile=profile_of(s),
                       smoke=smoke, default_recipe=is_default_recipe(s) and not smoke, roots=dict(roots))


# ----------------------------------------------------------------------------------------------- CLI helpers
def parse_kv(items: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise Refused(f"{it!r}: expected KEY=VALUE")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_composition(c: Composition, out_dir: str) -> None:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    _write(d / "refiner.set", f"# generated by tools/release/compose.py for {c.run}\n" + "\n".join(c.set_lines) + "\n")
    _write(d / "arm.extra", f"# generated by tools/release/compose.py for {c.run}\n" + "\n".join(c.extra_lines) + "\n")
    env = dict(c.env, E2E_CK_SET=str(d / "refiner.set"), E2E_EXTRA=str(d / "arm.extra"))
    lines = ["unset " + " ".join(CONTROL_ENV)]
    lines += [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
    lines += [f"CP_RUN={shlex.quote(c.run)}", f"CP_RUN_DIR={shlex.quote(c.run_dir)}", f"CP_RUN_ROOT={shlex.quote(c.run_root)}",
              f"CP_HASH={c.hash6}", f"CP_PROFILE={c.profile}", f"CP_GPUS={c.settings['GPUS']}",
              f"CP_DEFAULT_RECIPE={int(c.default_recipe)}", f"CP_DIR={shlex.quote(str(d))}"]
    _write(d / "env.sh", "\n".join(lines) + "\n")
    _write(d / "composition.json", json.dumps({**c.record(), "default_recipe": c.default_recipe, "env": env,
                                               "set_lines": c.set_lines, "extra_lines": c.extra_lines,
                                               "overrides": c.overrides, "roots": c.roots, "run_dir": c.run_dir}, indent=1) + "\n")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout


def git_state() -> Tuple[str, bool]:
    return _git("rev-parse", "HEAD").strip(), bool(_git("status", "--porcelain").strip())


def kst() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime(time.time() + 9 * 3600))


def _version(pkg: str) -> Optional[str]:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def verify(comp: Mapping[str, Any], overrides_path: str, teachers_json: Optional[str]) -> int:
    """Launcher output against the prediction, teacher checkpoints against the registry, reference diff on the main recipe."""
    rc = 0
    got = LU.read_lines(overrides_path)
    if got != comp["overrides"]:
        rc = 1
        print("[check] FAIL launcher overrides differ from the composition:")
        for line in sorted(set(got) ^ set(comp["overrides"])):
            print(f"   {'launcher ' if line in got else 'composed '} {line}")
    else:
        print(f"[check] launcher overrides == composition ({len(got)} lines, sha {_sha(got)[:16]})")
    if teachers_json and comp["teacher_pins"]:
        rec = json.loads(Path(teachers_json).read_text())
        for role, pin in comp["teacher_pins"].items():
            if role not in rec:     # no refiner teacher in this arm: only the BEV caches are read
                continue
            if rec[role].get("ckpt_sha16") != pin["refiner_ckpt_sha16"]:
                rc = 1
                print(f"[check] FAIL {role} refiner teacher {rec[role].get('run')}: ckpt sha16 "
                      f"{rec[role].get('ckpt_sha16')} != registry {pin['refiner_ckpt_sha16']}")
            else:
                print(f"[check] {role} refiner teacher ckpt sha16 {pin['refiner_ckpt_sha16']} matches the registry")
    if comp.get("default_recipe"):
        ok, lines = reference_diff(got, comp["roots"], comp["run_dir"])
        print(f"[check] reference diff ({REFERENCE.relative_to(REPO)}): {'PASS' if ok else 'FAIL'}")
        for line in lines:
            print("   " + line)
        rc = rc if ok else 1
    return rc


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("name")
    n.add_argument("kv", nargs="*")
    e = sub.add_parser("emit")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--run-root", default=None)
    e.add_argument("--smoke", action="store_true")
    e.add_argument("kv", nargs="*")
    v = sub.add_parser("verify")
    v.add_argument("--composition", required=True)
    v.add_argument("--overrides", required=True)
    v.add_argument("--teachers-json", default=None)
    rf = sub.add_parser("reference")
    rf.add_argument("--overrides", required=True)
    rf.add_argument("--run-dir", required=True)
    rf.add_argument("--plan-data", required=True)
    rf.add_argument("--teacher-cache", required=True)
    rf.add_argument("--data-root", required=True)
    a = sub.add_parser("settings-args")
    a.add_argument("--settings-json", required=True)
    r = sub.add_parser("record")
    r.add_argument("--composition", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--asset-profile", default=None)
    r.add_argument("--asset-exit", type=int, default=None)
    r.add_argument("--asset-log", default=None)
    d = sub.add_parser("dirty-record")
    d.add_argument("--out-dir", required=True)
    c = sub.add_parser("resume-check")
    c.add_argument("--composition", required=True)
    c.add_argument("--settings-json", required=True)
    c.add_argument("--allow-commit-change", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "name":
            print(compose(parse_kv(args.kv)).run)
        elif args.cmd == "emit":
            cp = compose(parse_kv(args.kv), run_root=args.run_root, smoke=args.smoke)
            write_composition(cp, args.out_dir)
            print(cp.run)
        elif args.cmd == "verify":
            return verify(json.loads(Path(args.composition).read_text()), args.overrides, args.teachers_json)
        elif args.cmd == "reference":
            roots = {"SSR_DATA_ROOT": args.data_root, "SSR_TEACHER_CACHE": args.teacher_cache,
                     "SSR_PLAN_DATA": args.plan_data}
            for line in LU.read_lines(args.overrides):
                print(normalize_line(line, roots, args.run_dir))
        elif args.cmd == "settings-args":
            s = json.loads(Path(args.settings_json).read_text())["settings"]
            for k, val in s.items():
                print(f"{k}={val}")
        elif args.cmd == "record":
            comp = json.loads(Path(args.composition).read_text())
            commit, dirty = git_state()
            rec = {k: comp[k] for k in ("run", "hash6", "smoke", "settings", "run_root", "overrides_sha256", "teacher_pins",
                                        "profile", "registry")}
            rec.update(commit=commit, git_dirty=dirty, gpus=comp["settings"]["GPUS"].split(","), hostname=socket.gethostname(),
                       python=sys.version.split()[0], torch=_version("torch"), created_kst=kst(),
                       labeler_workers=os.environ.get("LABELER_WORKERS"))
            if args.asset_exit is not None:
                rec["asset_check"] = {"profile": args.asset_profile, "quick": True, "exit": args.asset_exit,
                                      "log": args.asset_log}
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            _write(Path(args.out), json.dumps(rec, indent=1) + "\n")
        elif args.cmd == "dirty-record":
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            _write(out / "git_status.txt", _git("status", "--porcelain"))
            _write(out / "git_dirty.diff", _git("diff", "HEAD"))
            # untracked files are not in that diff: keep the small ones
            names = [x for x in _git("ls-files", "--others", "--exclude-standard", "-z").split("\0") if x]
            small = [x for x in names if (REPO / x).is_file() and (REPO / x).stat().st_size < 5 << 20]
            if small:
                subprocess.run(["tar", "-C", str(REPO), "--null", "-T", "-", "-czf", str(out / "git_untracked.tar.gz")],
                               input="\0".join(small), text=True, check=True)
        elif args.cmd == "resume-check":
            comp = json.loads(Path(args.composition).read_text())
            old = json.loads(Path(args.settings_json).read_text())
            bad = []
            if comp["hash6"] != old["hash6"] or comp["run"] != old["run"]:
                bad.append(f"settings: this code composes {comp['run']}, the run is {old['run']}")
            n_new, n_old = len(comp["settings"]["GPUS"].split(",")), len(old["settings"]["GPUS"].split(","))
            if n_new != n_old:
                bad.append(f"GPU count {n_new} != {n_old} of the run")
            commit, _ = git_state()
            if commit != old.get("commit") and not args.allow_commit_change:
                bad.append(f"commit {commit[:8]} != {str(old.get('commit'))[:8]} of the run (ALLOW_COMMIT_CHANGE=1 to accept)")
            if bad:
                raise Refused("; ".join(bad))
    except Refused as ex:
        print(f"[compose] REFUSED: {ex}", file=sys.stderr)
        return 3
    except LU.LaunchError as ex:
        print(f"[compose] REFUSED: {ex.msg}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
