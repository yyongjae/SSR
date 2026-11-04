#!/usr/bin/env python
"""Check that the assets one profile needs are on this server, and that they are the right ones.

  verify_assets.py --profile {base,refiner_noteacher,full} [--quick] [--strict] [--only GROUP]
                   [--det-teacher NAME] [--map-teacher NAME] [--registry FILE] [--workers 12] [--json FILE]

Profiles (docs/SETUP.md has the disk and hours of each):
  base               REFINER=off BEV_KD=off: dataset, maps, anchors, pdm_score_256
  refiner_noteacher  base + navtrain metric cache, GT stores (objects, sdf, e2e_side), packed, warm-up labels
  full               refiner_noteacher + the two BEV teacher caches and the two refiner teacher runs

--quick is the cheap pass (a few seconds, what `experiment.sh train` runs): manifests, directory listings, counts from
index files, a fixed sample of tokens.  Without it every token of a coverage check is looked at, the arrays are hashed
and all 85,109 tokens go through the training GT loader, whose statistics must equal the reference (a few minutes, CPU).
--only runs one group of the profile (dataset, metric_cache, gt, packed, labels, teacher_cache, teachers): the setup scripts
use it to check their own step.

Each line says PASS, WARN or FAIL, and a WARN or FAIL names the fix.
  FAIL  the training this profile describes would stop, or would silently run without the input.  Exit code 1.
  WARN  needed only by another job (evaluation, retraining the refiner teachers), or a difference that does not change
        the inputs read.  Exit code 0 unless --strict.

No file of a trained planner is needed by any check: cand.npy, v2_final.npy, labels/*/cand and v2_source are not looked
at.  A packed directory that came from a planner dump is accepted when its arrays equal the dataset pins, with a WARN.
Nothing is written except --json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""        # no check needs a GPU, and the torch import must not claim one

ROOT = Path(__file__).resolve().parents[2]     # repo root of this worktree
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from navsim.common import ssr_paths as SP  # noqa: E402

PROFILES = ("base", "refiner_noteacher", "full")
SETUP = "tools/release/setup"
N_SAMPLE = 200                 # tokens of a --quick coverage check
N_READ = 8                     # sampled files opened and read

# ----------------------------------------------------------------------------------------------- pins
# sha256(array bytes)[:16] unless it says file (sha256 of the file)[:16].  tokens = np.array(tokens, "U16").
N_TOKENS = {"navtrain_train": 85109, "navtrain_val": 18179}
PINS: Dict[str, Dict[str, str]] = {
    "packed/navtrain_train": {"token_sha16": "0b56b9aecbe1f094", "gt_traj": "ad3654bd81bdb82f",
                              "status": "d0d6f908e8be4f3c", "ok": "1f8f6dbc2f6c7496"},
    "packed/navtrain_val": {"token_sha16": "0fea22c41a4106b0", "gt_traj": "adb06837415973fd",
                            "status": "2309a38c18a61c48", "ok": "0bcf0f7a51703e00"},
    "raw256/navtrain_train": {"labels": "b0f6e572d72692c3", "ok": "d7bbcbf0fb3bb9e3"},
    "raw256/navtrain_val": {"labels": "dad2da663bebdb57", "ok": "89612aae9df7937a"},
    # traj: array bytes of traj.npy.  The older build.json on the reference server records another traj_sha16: that
    # value was taken before its accel columns were merged in, so it does not describe the file.
    "var/navtrain_train": {"labels": "4d042a39ebd1fd8a", "ok": "17efef5f358f3ac2", "traj": "9969cf98fb7aa9ac",
                           "anchor_idx": "47744cd2f2076785", "valid": "ceb5ebd777f6d4ef"},
    "var/navtrain_val": {"labels": "3db198fbd7a00d07", "ok": "c6ed18f987be86f3", "traj": "9a9e488338450b24",
                         "anchor_idx": "c8e5c5a8893e3a99", "valid": "ac4f8db652148258"},
}
SHAPES = {"raw256": {"labels": (256, 9), "ok": (256,)}, "var": {"labels": (96, 9), "ok": (96,), "traj": (96, 8, 3)}}
VAR_NAME = "var_separate_sampler16_accstraight"
REPO_FILES = {                 # path -> file sha16
    "resources/planning_vb/trajectory_anchors_256.npy": "44f64a763473c3a8",
    "navsim/planning/script/config/common/scene_filter/navtrain.yaml": "c37fea567a0cfdbc",
    "navsim/planning/script/config/common/scene_filter/navtest.yaml": "61284edf5003c029",
    "navsim/planning/script/config/training/default_train_val_test_log_split.yaml": "3c2182f5ed604427",
}
REPO_FILES_REFINER = {
    "resources/metric_cache_saved_config.yaml": "92d42853253622b2",
    "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml": "8d097e448ba6975c",
}
ANCHORS_SHA16 = "6bdf549edc25a959"     # float32 [256, 8, 3], the value anchor_sampler checks
PDM_SCORE = {"npy": "3b9e6c3e75bc698e", "tokens": "b1278ec613611037", "shape": (103288, 5, 256)}
NAVTRAIN_LOGS, TRAINVAL_LOG_PICKLES, TEST_LOG_PICKLES = 1192, 1310, 147
NAVTEST_TOKENS = 12146
CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0")   # the sensors the student reads
HISTORY_FRAMES = (0, 1)                    # frames back from t0 it reads by default (frame_indices 2, 3)
GT_EXPECT = {"gt_n": 85109, "gt_ok": 85109, "p_pdm_nonfinite": 0, "cl_n_lt2": 0, "sdf_absmax_zero": 0,
             "sdf_absmax_max": 10.0, "obj_n_max": 800, "obj_n_zero": 68, "p_pdm_zero": 4370, "p_pdm_max": 67.8193436381719}
ENV_PINS = {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "pytorch-lightning": "2.2.1", "timm": "1.0.28",
            "numpy": "1.23.4", "hydra-core": "1.2.0", "omegaconf": "2.3.1", "nuplan-devkit": "1.2.0"}
NAVSIM_V11_COMMIT = "0811876c274e8b058ab2be9b3dcd4d37bd23f177"
TRACKED_TOKENS = "resources/km_corridor/packed/{split}/tokens.parquet"
REGISTRY = "tools/release/registry.yaml"
RESMAP_CHANNELS_HWC = (256, 100, 100)


# ----------------------------------------------------------------------------------------------- report
class Report:
    def __init__(self):
        self.rows: List[Dict[str, str]] = []

    def add(self, status: str, name: str, detail: str, fix: str = "") -> None:
        self.rows.append({"status": status, "name": name, "detail": detail, "fix": fix})
        line = f"{status:<4}  {name}: {detail}"
        if fix and status != "PASS":
            line += f"\n      fix: {fix}"
        print(line, flush=True)

    def n(self, status: str) -> int:
        return sum(r["status"] == status for r in self.rows)


class Ctx:
    def __init__(self, paths: Dict[str, str], quick: bool, workers: int, rep: Report, det: str, mapt: str,
                 registry: Optional[dict]):
        self.p, self.quick, self.workers, self.rep = paths, quick, workers, rep
        self.det, self.mapt, self.registry = det, mapt, registry
        self.data = Path(paths["SSR_DATA_ROOT"])
        self.ck = self.data / "ck"
        self._tokens: Dict[str, object] = {}

    # one verdict per check: a list of problems -> FAIL (or WARN when soft), an empty list -> PASS with `good`
    def verdict(self, name: str, problems: Sequence[str], good: str, fix: str, soft: bool = False) -> None:
        if problems:
            head = "; ".join(list(problems)[:4])
            more = f" (+{len(problems) - 4} more)" if len(problems) > 4 else ""
            self.rep.add("WARN" if soft else "FAIL", name, head + more, fix)
        else:
            self.rep.add("PASS", name, good)

    def tokens(self, split: str):
        """The split's table (token, log), from the copy tracked in this repo, so a check does not depend on the
        directory it checks."""
        if split not in self._tokens:
            import pandas as pd
            df = pd.read_parquet(ROOT / TRACKED_TOKENS.format(split=split), columns=["token", "log"])
            df["token"] = df.token.astype(str)
            self._tokens[split] = df
        return self._tokens[split]

    def sample(self, split: str, n: int = N_SAMPLE) -> List[Tuple[str, str]]:
        df = self.tokens(split)
        rows = np.random.default_rng(0).choice(len(df), min(n, len(df)), replace=False)
        return [(df.token.iat[int(i)], df.log.iat[int(i)]) for i in sorted(rows)]

    def coverage(self, split: str) -> List[Tuple[str, str]]:
        """Tokens a coverage check looks at: a fixed sample with --quick, else all."""
        if self.quick:
            return self.sample(split)
        df = self.tokens(split)
        return list(zip(df.token.tolist(), df.log.tolist()))


# ----------------------------------------------------------------------------------------------- helpers
def sha16_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()[:16]


def sha16_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def missing_paths(paths: Iterable[Path], workers: int) -> List[Path]:
    """Paths that are not files (threads: a stat of 85,109 files is IO, not CPU)."""
    paths = list(paths)
    with ThreadPoolExecutor(max(1, workers)) as ex:
        ok = list(ex.map(os.path.isfile, paths, chunksize=256))
    return [p for p, k in zip(paths, ok) if not k]


def npy_header(p) -> Tuple[Tuple[int, ...], str]:
    a = np.load(p, mmap_mode="r", allow_pickle=False)
    return tuple(a.shape), str(a.dtype)


def count_entries(d: Path, suffix: str = "") -> int:
    with os.scandir(d) as it:
        return sum(e.name.endswith(suffix) for e in it)


def hash_problems(rel: str, base: Path, pins: Dict[str, str], arrays: Sequence[str]) -> List[str]:
    out = []
    for k in arrays:
        p = base / f"{k}.npy"
        if not p.is_file():
            out.append(f"{rel}/{k}.npy missing")
        elif sha16_array(np.load(p, mmap_mode="r", allow_pickle=False)) != pins[k]:
            out.append(f"{rel}/{k}.npy differs from pin {pins[k]}")
    return out


# ----------------------------------------------------------------------------------------------- registry
def load_registry(path: Path, paths: Dict[str, str]) -> Optional[dict]:
    """tools/release/registry.yaml with ${VAR} replaced by the resolved roots (then the environment)."""
    if not path.is_file():
        return None
    import yaml
    txt = path.read_text()

    def sub(m):
        return str(paths.get(m.group(1)) or os.environ.get(m.group(1)) or m.group(0))

    return yaml.safe_load(re.sub(r"\$\{(\w+)\}", sub, txt))


def teacher_entry(c: Ctx, role: str, name: str) -> Optional[dict]:
    """Registry entry of a selected teacher, or None after a FAIL that lists what is missing."""
    fix = f"add the {role} teacher to {REGISTRY} (status ready + its fields) or choose a ready one"
    if c.registry is None:
        c.rep.add("FAIL", f"teacher.{role}", f"{REGISTRY} not found", fix)
        return None
    e = (c.registry.get("teachers", {}).get(role) or {}).get(name)
    if e is None:
        c.rep.add("FAIL", f"teacher.{role}", f"{name!r} is not in the registry", fix)
    elif e.get("status") != "ready":
        c.rep.add("FAIL", f"teacher.{role}", f"{name!r} has status {e.get('status')!r}, missing: {e.get('needs')}", fix)
    else:
        return e
    return None


# ----------------------------------------------------------------------------------------------- checks: base
def check_env(c: Ctx) -> None:
    from importlib import metadata
    bad = []
    for k, want in ENV_PINS.items():
        try:
            got = metadata.version(k)
        except metadata.PackageNotFoundError:
            got = "not installed"
        if got != want:
            bad.append(f"{k} {got} (reference {want})")
    if sys.version_info[:2] != (3, 9):
        bad.append(f"python {sys.version.split()[0]} (reference 3.9)")
    c.verdict("env.versions", bad, f"python {sys.version.split()[0]}, {len(ENV_PINS)} packages at the reference version",
              f"{SETUP}/install_env.sh; numbers from another version are not comparable", soft=True)


def check_navsim_import(c: Ctx) -> None:
    import navsim
    where = Path(navsim.__file__).resolve()
    c.verdict("env.navsim_import", [] if ROOT.resolve() in where.parents else [f"navsim resolves to {where}"],
              f"navsim is imported from this checkout", f"PYTHONPATH={ROOT} (an editable install of another checkout "
              "takes precedence otherwise)")


def check_repo_files(c: Ctx, files: Dict[str, str], name: str) -> None:
    bad = []
    for rel, want in files.items():
        p = ROOT / rel
        if not p.is_file():
            bad.append(f"{rel} missing")
        elif sha16_file(p) != want:
            bad.append(f"{rel} differs from pin {want}")
    c.verdict(name, bad, f"{len(files)} files equal their pins", "git checkout of these files")


def check_anchors(c: Ctx) -> None:
    a = np.load(ROOT / "resources/planning_vb/trajectory_anchors_256.npy", allow_pickle=False)
    got = sha16_array(a.astype(np.float32))
    c.verdict("anchors", [] if got == ANCHORS_SHA16 else [f"array sha16 {got} != {ANCHORS_SHA16}"],
              f"[256, 8, 3] array sha16 {got}", "git checkout resources/planning_vb/trajectory_anchors_256.npy")


def check_tokens(c: Ctx) -> None:
    bad = []
    for split, n in N_TOKENS.items():
        df = c.tokens(split)
        got = sha16_array(np.array(df.token.tolist(), "U16"))
        want = PINS[f"packed/{split}"]["token_sha16"]
        if len(df) != n or got != want:
            bad.append(f"{split}: {len(df)} tokens, sha16 {got} (expected {n}, {want})")
    c.verdict("tokens.tracked", bad, ", ".join(f"{s} {n}" for s, n in N_TOKENS.items()) + " tokens equal their pins",
              f"git checkout {TRACKED_TOKENS}")


def check_dataset(c: Ctx) -> None:
    fix = f"{SETUP}/10_download_dataset.sh"
    root = Path(c.p["OPENSCENE_DATA_ROOT"])
    logs, sensors, maps = root / "navsim_logs/trainval", root / "sensor_blobs/trainval", Path(c.p["NUPLAN_MAPS_ROOT"])
    bad = []
    if not (maps / "nuplan-maps-v1.0.json").is_file():
        bad.append(f"{maps}/nuplan-maps-v1.0.json missing")
    n = count_entries(logs, ".pkl") if logs.is_dir() else 0
    if n < TRAINVAL_LOG_PICKLES:
        bad.append(f"{logs}: {n} log pickles (expected {TRAINVAL_LOG_PICKLES}; they must be directly in this directory)")
    n_s = count_entries(sensors) if sensors.is_dir() else 0
    if n_s < NAVTRAIN_LOGS:
        bad.append(f"{sensors}: {n_s} log directories (expected {NAVTRAIN_LOGS})")
    used = sorted(set(c.tokens("navtrain_train").log) | set(c.tokens("navtrain_val").log))
    no_log = [g for g in used if not (logs / f"{g}.pkl").is_file()]
    if no_log:
        bad.append(f"{len(no_log)} of {len(used)} navtrain logs have no pickle, e.g. {no_log[0]}")
    no_cam = [g for g in used if any(not _has_entry(sensors / g / cam) for cam in CAMERAS)]
    if no_cam:
        bad.append(f"{len(no_cam)} of {len(used)} navtrain logs lack {'/'.join(CAMERAS)}, e.g. {no_cam[0]}")
    # the frames a few logs' tokens read must have their image files: partial extraction leaves the directories in place
    pick = [g for g in used[:: max(1, len(used) // 8)] if g not in set(no_log)][:8]
    all_tokens = set(c.tokens("navtrain_train").token) | set(c.tokens("navtrain_val").token)
    n_files = 0
    for g in pick:
        with open(logs / f"{g}.pkl", "rb") as f:
            frames = pickle.load(f)
        for i, fr in enumerate(frames):
            if fr["token"] not in all_tokens:
                continue
            for k in HISTORY_FRAMES:
                for cam in CAMERAS:
                    n_files += 1
                    if not (sensors / frames[i - k]["cams"][cam]["data_path"]).is_file():
                        bad.append(f"{sensors}/{frames[i - k]['cams'][cam]['data_path']} missing")
    c.verdict("dataset.trainval", bad, f"{n} log pickles, {n_s} sensor directories, maps; {n_files} image files of "
              f"{len(pick)} logs present", fix)


def _has_entry(d: Path) -> bool:
    try:
        with os.scandir(d) as it:
            return any(True for _ in it)
    except OSError:
        return False


def check_dataset_test(c: Ctx) -> None:
    root = Path(c.p["SSR_TEST_DATA_ROOT"])
    logs, sensors = root / "navsim_logs/test/test", root / "sensor_blobs/test/test"
    bad = []
    n = count_entries(logs, ".pkl") if logs.is_dir() else 0
    if n < TEST_LOG_PICKLES:
        bad.append(f"{logs}: {n} log pickles (expected {TEST_LOG_PICKLES})")
    if not sensors.is_dir():
        bad.append(f"{sensors} missing")
    c.verdict("dataset.test", bad, f"{n} test log pickles, sensors present",
              f"{SETUP}/10_download_dataset.sh (only evaluation reads these)", soft=True)


def check_pdm_score(c: Ctx) -> None:
    d = Path(c.p["SSR_PLAN_DATA"])
    fix = (f"copy pdm_score_256.npy and pdm_score_256.tokens.json into SSR_PLAN_DATA={d} (no public packed copy; "
           "docs/SETUP.md says how to rebuild it)")
    npy, tok = d / "pdm_score_256.npy", d / "pdm_score_256.tokens.json"
    bad = []
    if not npy.is_file():
        bad.append(f"{npy} missing")
    elif npy_header(npy)[0] != PDM_SCORE["shape"]:
        bad.append(f"{npy}: shape {npy_header(npy)[0]} != {PDM_SCORE['shape']}")
    elif not c.quick and sha16_file(npy) != PDM_SCORE["npy"]:
        bad.append(f"{npy} differs from pin {PDM_SCORE['npy']}")
    if not tok.is_file():
        bad.append(f"{tok} missing")
    elif sha16_file(tok) != PDM_SCORE["tokens"]:
        bad.append(f"{tok} differs from pin {PDM_SCORE['tokens']}")
    c.verdict("pdm_score_256", bad, f"{'shape + tokens' if c.quick else 'file + tokens'} equal their pins", fix)


def check_timm(c: Ctx) -> None:
    hub = Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface") / "hub"
    d = hub / "models--timm--resnet34.tv_in1k/snapshots"
    c.verdict("weights.resnet34", [] if _has_entry(d) else [f"not cached under {hub}"], f"cached under {hub}",
              "first training start downloads it (ImageNet weights, needs network); other backbones have their own "
              "file", soft=True)


def check_eval_assets(c: Ctx) -> None:
    v11, cache = Path(c.p["NAVSIM_V11"]), Path(c.p["NAVSIM_V11_CACHE"])
    bad = []
    if not (v11 / "navsim").is_dir():
        bad.append(f"NAVSIM_V11={v11} has no navsim/")
    if not cache.is_dir():
        bad.append(f"NAVSIM_V11_CACHE={cache} missing")
    else:
        n = sum(1 for _ in cache.glob("*/unknown/*/metric_cache.pkl")) if not c.quick else None
        if n is not None and n != NAVTEST_TOKENS:
            bad.append(f"{cache}: {n} metric caches (expected {NAVTEST_TOKENS})")
    c.verdict("eval.pristine_v11", bad, f"tree and navtest cache present (commit {NAVSIM_V11_COMMIT[:7]} is checked by "
              "the eval preflight)", f"{SETUP}/70_pristine_navsim.sh (only evaluation reads these)", soft=True)
    bad = [f"{k}={c.p[k]} missing" for k in ("NAVSIM_V22_POST", "NAVSIM_V22_PRE", "NAVSIM_V22_CACHE")
           if not Path(c.p[k]).is_dir()]
    c.verdict("eval.epdms_v22", bad, "both trees and the navtest cache present",
              f"{SETUP}/70_pristine_navsim.sh (only EPDMS reads these)", soft=True)


# ----------------------------------------------------------------------------------------------- checks: refiner
def check_scorer(c: Ctx) -> None:
    p = ROOT / "tools/refiner/rescore_attr.py"
    c.verdict("scorer.vendored", [] if p.is_file() else [f"{p} missing"], "tools/refiner/rescore_attr.py present",
              "git checkout tools/refiner/rescore_attr.py")


def check_metric_cache(c: Ctx, split: str = "navtrain_train", soft: bool = False) -> None:
    root = c.data / "metric_cache"
    fix = f"{SETUP}/20_metric_cache_navtrain.sh"
    bad = []
    done = root / "_state/done_tokens.txt"
    if not root.is_dir():
        c.verdict(f"metric_cache.{split}", [f"{root} missing"], "", fix, soft)
        return
    if done.is_file():
        have = set(done.read_text().split())
        lack = [t for t in c.tokens(split).token if t not in have]
        if lack:
            bad.append(f"{len(lack)} tokens not in _state/done_tokens.txt, e.g. {lack[0]}")
    else:
        bad.append(f"{done} missing")
    toks = c.coverage(split)
    paths = [root / g / "unknown" / t / "metric_cache.pkl" for t, g in toks]
    lost = missing_paths(paths, c.workers)
    if lost:
        bad.append(f"{len(lost)} of {len(paths)} metric_cache.pkl missing, e.g. {lost[0]}")
    n_read = 0
    for p, _ in zip([q for q in paths if q not in set(lost)][:: max(1, len(paths) // N_READ)], range(N_READ)):
        n_read += 1
        with open(p, "rb") as f:
            if f.read(6) != b"\xfd7zXZ\x00":
                bad.append(f"{p}: not an xz stream")
    c.verdict(f"metric_cache.{split}", bad, f"{len(paths)} of {len(c.tokens(split))} tokens checked present; "
              f"{n_read} headers read", fix, soft)


def check_gt_stores(c: Ctx) -> None:
    fix = f"{SETUP}/30_gt_stores.sh"
    split = "navtrain_train"
    toks = c.coverage(split)
    obj_dirs = [c.data / "objects" / d for d in ("train", "navtrain", "dev")]
    have_obj = set()
    for d in obj_dirs:
        if d.is_dir():
            with os.scandir(d) as it:
                have_obj |= {e.name[:-4] for e in it if e.name.endswith(".npz")}
    lack = [t for t, _ in toks if t not in have_obj]
    c.verdict("gt.objects", [f"{len(lack)} of {len(toks)} tokens missing, e.g. {lack[0]}"] if lack else [],
              f"{len(toks)} tokens present", fix)
    lost = missing_paths((c.data / "sdf/navtrain" / f"{t}.npz" for t, _ in toks), c.workers)
    c.verdict("gt.sdf", [f"{len(lost)} of {len(toks)} tokens missing, e.g. {lost[0]}"] if lost else [],
              f"{len(toks)} tokens present", fix)
    lost = missing_paths((c.data / "e2e_side" / t[:2] / f"{t}.npz" for t, _ in toks), c.workers)
    c.verdict("gt.e2e_side", [f"{len(lost)} of {len(toks)} tokens missing, e.g. {lost[0]}"] if lost else [],
              f"{len(toks)} tokens present", fix)
    if c.quick:
        bad, n_read = [], 0
        for t, _ in c.sample(split, N_READ):
            p = c.data / "e2e_side" / t[:2] / f"{t}.npz"
            if p.is_file():
                n_read += 1
                with np.load(p, allow_pickle=False) as z:
                    if not np.isfinite(z["p_pdm"]) or int(z["cl_n"]) < 2:
                        bad.append(f"{p}: p_pdm {z['p_pdm']}, cl_n {z['cl_n']}")
        c.verdict("gt.read_sample", bad, f"{n_read} side files read", fix)
    else:
        check_gt_loader(c, fix)


def _gt_chunk(a):
    """Runs in a worker: the loader the training uses (report 39 section 5.10)."""
    data, toks = a
    from navsim.agents.para_ssr.refiner.e2e import GTLoader
    loader = _GT_LOADER.setdefault(data, GTLoader(Path(data)))
    out = []
    for t in toks:
        o = loader.load(t)
        out.append((bool(o["ref_gt_ok"]), int(o["ref_obj_n"]), float(o["ref_p_pdm"]), int(o["ref_cl_n"]),
                    float(o["ref_sdf"].float().abs().max())))
    return out


_GT_LOADER: Dict[str, object] = {}


def gt_stats(rows: Sequence[Tuple[bool, int, float, int, float]]) -> Dict[str, float]:
    ok = np.array([r[0] for r in rows], bool)
    n_obj = np.array([r[1] for r in rows], np.int64)
    pdm = np.array([r[2] for r in rows], np.float64)
    cl = np.array([r[3] for r in rows], np.int64)
    sdf = np.array([r[4] for r in rows], np.float64)
    return dict(gt_n=len(rows), gt_ok=int(ok.sum()), p_pdm_nonfinite=int((~np.isfinite(pdm)).sum()),
                cl_n_lt2=int((cl < 2).sum()), sdf_absmax_zero=int((sdf == 0).sum()), sdf_absmax_max=float(sdf.max()),
                obj_n_max=int(n_obj.max()), obj_n_zero=int((n_obj == 0).sum()), p_pdm_zero=int((pdm == 0).sum()),
                p_pdm_max=float(pdm.max()))


def compare_gt(got: Dict[str, float], want: Dict[str, float]) -> List[str]:
    return [f"{k} {got.get(k)} != {v}" for k, v in want.items() if abs(float(got.get(k, np.nan)) - float(v)) > 1e-6
            or not np.isfinite(float(got.get(k, np.nan)))]


def check_gt_loader(c: Ctx, fix: str) -> None:
    from multiprocessing import get_context
    toks = c.tokens("navtrain_train").token.tolist()
    chunks = [(str(c.data), toks[i:i + 500]) for i in range(0, len(toks), 500)]
    with get_context("fork").Pool(max(1, c.workers)) as p:
        rows = [r for rr in p.imap_unordered(_gt_chunk, chunks) for r in rr]
    got = gt_stats(rows)
    c.verdict("gt.loader_all_tokens", compare_gt(got, GT_EXPECT), json.dumps(got), fix)


def check_packed(c: Ctx, split: str = "navtrain_train", soft: bool = False) -> None:
    d = c.ck / "packed" / split
    fix = f"{SETUP}/40_packed_from_dataset.sh"
    name = f"packed.{split}"
    need = ["tokens.parquet", "gt_traj.npy", "ok.npy", "status.npy"]
    lack = [f for f in need if not (d / f).is_file()]
    if lack:
        c.verdict(name, [f"{d}: {lack} missing"], "", fix, soft)
        return
    pins = PINS[f"packed/{split}"]
    import pandas as pd
    toks = pd.read_parquet(d / "tokens.parquet", columns=["token"]).token.astype(str).tolist()
    bad = []
    got = sha16_array(np.array(toks, "U16"))
    if got != pins["token_sha16"]:
        bad.append(f"token_sha16 {got} != {pins['token_sha16']}")
    if npy_header(d / "gt_traj.npy") != ((N_TOKENS[split], 8, 3), "float32"):
        bad.append(f"gt_traj.npy {npy_header(d / 'gt_traj.npy')}")
    # small arrays (13 MB): hashed in both modes
    bad += hash_problems(str(d), d, pins, ("gt_traj", "status", "ok"))
    c.verdict(name, bad, f"{len(toks)} tokens; tokens, gt_traj, status, ok equal the dataset pins", fix, soft)
    meta = d / "meta.json"
    if not bad and meta.is_file():
        m = json.loads(meta.read_text())
        c.verdict(f"{name}.provenance", [] if m.get("source") == "dataset" else
                  ["built from a planner dump directory; the arrays read are equal to the dataset's, so no value depends "
                   "on that planner"], "built from the dataset", f"{fix} replaces it (move the old directory away first)",
                  soft=True)


def check_labels(c: Ctx, split: str = "navtrain_train", soft: bool = False) -> None:
    fix = f"{SETUP}/50_warmup_labels.sh"
    n = N_TOKENS[split]
    for kind, d, arrays in (("raw256", c.ck / "labels" / split / "raw256", ("labels", "ok")),
                            ("var", c.ck / "ck2/labels" / split / VAR_NAME, ("labels", "ok", "traj"))):
        bad, name = [], f"labels.{kind}.{split}"
        pins = PINS[f"{kind}/{split}"]
        for a in arrays:
            p = d / f"{a}.npy"
            if not p.is_file():
                bad.append(f"{p} missing")
            elif npy_header(p)[0] != (n,) + SHAPES[kind][a]:
                bad.append(f"{p}: shape {npy_header(p)[0]} != {(n,) + SHAPES[kind][a]}")
        if kind == "var":
            b = d / "build.json"
            if not b.is_file() or not (d / "index.npz").is_file():
                bad.append(f"{d}: build.json or index.npz missing")
            else:
                j = json.loads(b.read_text())
                if j.get("token_sha16") != PINS[f"packed/{split}"]["token_sha16"]:
                    bad.append(f"build.json token_sha16 {j.get('token_sha16')}")
                if j.get("anchors_sha16") != ANCHORS_SHA16:
                    bad.append(f"build.json anchors_sha16 {j.get('anchors_sha16')}")
                if int(j.get("n_built", -1)) != n:
                    bad.append(f"build.json n_built {j.get('n_built')} != {n}")
        else:
            plan = d / "shards/_plan.json"
            if plan.is_file() and json.loads(plan.read_text()).get("token_sha16") != PINS[f"packed/{split}"]["token_sha16"]:
                bad.append(f"{plan}: token_sha16 differs")
        if not c.quick and not bad:
            bad += hash_problems(str(d), d, pins, arrays)
            if kind == "var":
                with np.load(d / "index.npz", allow_pickle=False) as z:
                    for k in ("anchor_idx", "valid"):
                        if sha16_array(z[k]) != pins[k]:
                            bad.append(f"index.npz {k} differs from pin {pins[k]}")
        c.verdict(name, bad, "shapes and provenance fine" if c.quick else "arrays equal their pins", fix, soft)


def check_teacher_retrain_inputs(c: Ctx) -> None:
    """Only retraining the refiner teachers reads these: they monitor navtrain_val.  Finished teacher runs do not."""
    check_packed(c, "navtrain_val", soft=True)
    check_labels(c, "navtrain_val", soft=True)
    check_metric_cache(c, "navtrain_val", soft=True)


# ----------------------------------------------------------------------------------------------- checks: teachers
def check_bev_per_token(c: Ctx, role: str, name: str, e: dict) -> None:
    root = Path(e["cache_train"])
    tag = f"teacher.{role}.{name}.cache"
    fix = f"{SETUP}/60_teacher_bev.sh"
    mf = root / "manifest.json"
    if not mf.is_file():
        c.verdict(tag, [f"{mf} missing"], "", fix)
        return
    m = json.loads(mf.read_text())
    bad = []
    if str(m.get("checkpoint_sha256_head")) != str(e.get("sha_head")):
        bad.append(f"checkpoint_sha256_head {m.get('checkpoint_sha256_head')} != registry {e.get('sha_head')}")
    if m.get("layout") != "samples/<token[:2]>/<token>.npz":
        bad.append(f"layout {m.get('layout')}")
    if "_future" in str(root.resolve()):
        bad.append("a *_future cache holds future-frame teacher runs")
    h, w = m.get("target_bev_shape", [0, 0])
    ch = int(m.get("bev_channels", 0))
    if int(m.get("num_samples_written", -1)) < N_TOKENS["navtrain_train"]:
        bad.append(f"num_samples_written {m.get('num_samples_written')}")
    toks = c.coverage("navtrain_train")
    paths = [root / "samples" / t[:2] / f"{t}.npz" for t, _ in toks]
    lost = missing_paths(paths, c.workers)
    if lost:
        bad.append(f"{len(lost)} of {len(paths)} tokens missing, e.g. {lost[0]}")
    min_bytes = ch * h * w * 2              # bev_feature alone, float16
    with ThreadPoolExecutor(max(1, c.workers)) as ex:
        sizes = list(ex.map(lambda p: os.path.getsize(p) if os.path.isfile(p) else min_bytes, paths, chunksize=256))
    short = [p for p, s in zip(paths, sizes) if s < min_bytes]
    if short:
        bad.append(f"{len(short)} files smaller than the bev tensor ({min_bytes} B), e.g. {short[0]}")
    alive = [p for p in paths if p not in set(lost) and p not in set(short)]
    if alive:
        with np.load(alive[0], allow_pickle=False) as z:
            shape = tuple(z["bev_feature"].shape)
            if shape != (ch, h, w) or z["bev_feature"].dtype != np.float16:
                bad.append(f"{alive[0]}: bev_feature {shape} {z['bev_feature'].dtype}")
    c.verdict(tag, bad, f"manifest head {m.get('checkpoint_sha256_head')}; {len(paths)} tokens present, sizes fine", fix)


def check_bev_sharded(c: Ctx, role: str, name: str, e: dict) -> None:
    root = Path(e["cache_root"])
    tag = f"teacher.{role}.{name}.cache"
    fix = f"{SETUP}/60_teacher_bev.sh"
    if not (root / "meta.json").is_file() or not (root / "index.json").is_file():
        c.verdict(tag, [f"{root}: meta.json or index.json missing"], "", fix)
        return
    m = json.loads((root / "meta.json").read_text())
    idx = json.loads((root / "index.json").read_text())
    bad = []
    if str(m.get("checkpoint_sha256")) != str(e.get("sha256")):
        bad.append(f"checkpoint_sha256 {m.get('checkpoint_sha256')} != registry {e.get('sha256')}")
    if int(m.get("num_frames", -1)) != len(idx):
        bad.append(f"num_frames {m.get('num_frames')} != {len(idx)} rows of index.json")
    t = m.get("tensors", {}).get("bev", {})
    shape = tuple(t.get("shape", ()))
    if t.get("dtype") != "float16":
        bad.append(f"bev dtype {t.get('dtype')}")
    lack = [tk for tk in c.tokens("navtrain_train").token if tk not in idx]
    if lack:
        bad.append(f"{len(lack)} navtrain_train tokens not in index.json, e.g. {lack[0]}")
    toks = c.coverage("navtrain_train")
    need: Dict[str, int] = {}
    for tk, _ in toks:
        if tk in idx:
            sh, off = idx[tk]
            need[sh] = max(need.get(sh, -1), int(off))
    n_read = 0
    for sh, top in need.items():
        p = root / "bev" / f"{sh}.npy"
        if not p.is_file():
            bad.append(f"{p} missing")
            continue
        n_read += 1
        got, dtype = npy_header(p)
        if got[1:] != shape or dtype != "float16":
            bad.append(f"{p}: {got} {dtype}, expected [N, {shape}] float16")
        elif top >= got[0]:
            bad.append(f"{p}: row {top} beyond its {got[0]} rows")
    c.verdict(tag, bad, f"meta sha {str(m.get('checkpoint_sha256'))[:16]}; all {len(c.tokens('navtrain_train'))} tokens in "
              f"the index; {n_read} shards fit", fix)


BEV_LAYOUTS: Dict[str, Callable] = {"per_token_npz": check_bev_per_token, "sharded": check_bev_sharded}


def check_norm(c: Ctx, role: str, name: str, e: dict) -> None:
    p = Path(str(e.get("norm_v15", "")))
    p = p if p.is_absolute() else ROOT / p
    c.verdict(f"teacher.{role}.{name}.norm", [] if p.is_file() else [f"{p} missing"], f"{p.relative_to(ROOT) if ROOT in p.parents else p}",
              "the registry's norm_v15 must point at the teacher's z-score file (refiner/data.py compute_teacher_norm "
              "makes one)")


def check_refiner_teacher(c: Ctx, role: str, name: str, e: dict) -> None:
    arm = {"det": "T", "map": "M"}[role]
    d = Path(e["refiner_run"])
    tag = f"teacher.{role}.{name}.refiner_run"
    fix = (f"{SETUP}/run_refiner_teachers.sh, or unpack the teacher tarball; a retrained run has a new sha: put it in "
           f"refiner_ckpt_sha16 of {REGISTRY}")
    norm = {"T": "norm.npz", "M": "norm_map.npz"}[arm]
    lack = [f for f in ("config.json", "done.json", "ckpt_last.pt", norm) if not (d / f).is_file()]
    if lack:
        c.verdict(tag, [f"{d}: {lack} missing"], "", fix)
        return
    cfg = json.loads((d / "config.json").read_text())
    want = dict(arm=arm, trainer="train_ck2", split="navtrain_train", init_from="none")
    bad = [f"config.json {k} {cfg.get(k)!r} != {v!r}" for k, v in want.items() if cfg.get(k) != v]
    got = sha16_file(d / "ckpt_last.pt")
    if got != e.get("refiner_ckpt_sha16"):
        bad.append(f"ckpt_last.pt sha16 {got} != registry {e.get('refiner_ckpt_sha16')}")
    c.verdict(tag, bad, f"arm {arm}, ckpt sha16 {got}, split navtrain_train, from scratch", fix)


def check_teacher_caches(c: Ctx) -> None:
    for role, name in (("det", c.det), ("map", c.mapt)):
        e = teacher_entry(c, role, name)
        if e is None:
            continue
        layout = e.get("layout")
        if layout not in BEV_LAYOUTS:
            c.rep.add("FAIL", f"teacher.{role}.{name}.cache", f"unknown layout {layout!r}", f"one of {sorted(BEV_LAYOUTS)}")
        else:
            BEV_LAYOUTS[layout](c, role, name, e)
        check_norm(c, role, name, e)


def check_refiner_teachers(c: Ctx) -> None:
    for role, name in (("det", c.det), ("map", c.mapt)):
        e = teacher_entry(c, role, name)
        if e is not None:
            check_refiner_teacher(c, role, name, e)


# ----------------------------------------------------------------------------------------------- profiles
BASE_CHECKS: Sequence[Callable[[Ctx], None]] = (
    check_env, check_navsim_import,
    lambda c: check_repo_files(c, REPO_FILES, "repo.pinned_files"),
    check_anchors, check_tokens, check_dataset, check_pdm_score, check_timm, check_dataset_test, check_eval_assets)
REFINER_CHECKS: Sequence[Callable[[Ctx], None]] = (
    lambda c: check_repo_files(c, REPO_FILES_REFINER, "repo.pinned_files_refiner"),
    check_scorer, check_metric_cache, check_gt_stores, check_packed, check_labels)
TEACHER_CHECKS: Sequence[Callable[[Ctx], None]] = (check_teacher_caches, check_refiner_teachers, check_teacher_retrain_inputs)
CHECKS = {"base": BASE_CHECKS, "refiner_noteacher": tuple(BASE_CHECKS) + tuple(REFINER_CHECKS),
          "full": tuple(BASE_CHECKS) + tuple(REFINER_CHECKS) + tuple(TEACHER_CHECKS)}
GROUPS: Dict[str, Sequence[Callable[[Ctx], None]]] = {
    "dataset": (check_tokens, check_dataset, check_dataset_test), "metric_cache": (check_metric_cache,),
    "gt": (check_gt_stores,), "packed": (check_packed,), "labels": (check_labels,),
    "teacher_cache": (check_teacher_caches,), "teachers": (check_refiner_teachers,)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, choices=PROFILES)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--strict", action="store_true", help="a WARN also gives exit code 1")
    ap.add_argument("--only", default="", choices=[""] + sorted(GROUPS), help="one group of checks of this profile")
    ap.add_argument("--det-teacher", default=os.environ.get("DET_TEACHER") or "bevfusion")
    ap.add_argument("--map-teacher", default=os.environ.get("MAP_TEACHER") or "resmap")
    ap.add_argument("--registry", default=os.environ.get("SSR_RELEASE_REGISTRY") or str(ROOT / REGISTRY))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--json", default="", help="also write the rows to this file")
    a = ap.parse_args(argv)
    paths = SP.resolve()
    rep = Report()
    registry = load_registry(Path(a.registry), paths)
    c = Ctx(paths, a.quick, a.workers, rep, a.det_teacher, a.map_teacher, registry)
    print(f"[verify_assets] profile {a.profile}, {'quick' if a.quick else 'full check'}, data {paths['SSR_DATA_ROOT']}, "
          f"teachers {paths['SSR_TEACHER_CACHE']}", flush=True)
    for fn in [f for f in CHECKS[a.profile] if not a.only or f in GROUPS[a.only]]:
        try:
            fn(c)
        except Exception as e:      # one broken input must not hide the others
            rep.add("FAIL", getattr(fn, "__name__", "check"), f"{type(e).__name__}: {e}",
                    "see the traceback line: " + traceback.format_exc(limit=-1).strip().splitlines()[-2].strip())
    code = 1 if rep.n("FAIL") or (a.strict and rep.n("WARN")) else 0
    print(f"[verify_assets] {rep.n('PASS')} pass, {rep.n('WARN')} warn, {rep.n('FAIL')} fail -> exit {code}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(dict(profile=a.profile, quick=a.quick, exit=code, rows=rep.rows), indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
