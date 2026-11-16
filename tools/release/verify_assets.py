#!/usr/bin/env python
"""Check that the assets one profile needs are on this server, and that they are the right ones.

  verify_assets.py --profile {base,refiner_noteacher,full} [--data {navtrain,trainval_simscale}] [--quick] [--strict]
                   [--only GROUP] [--det-teacher NAME] [--map-teacher NAME] [--registry FILE] [--workers 12] [--json FILE]

Profiles (docs/SETUP.md has the disk and hours of each):
  base               REFINER=off BEV_KD=off: dataset, maps, anchors, pdm_score_256_officialEP
  refiner_noteacher  base + navtrain metric cache, GT stores (objects, sdf, e2e_side), packed, warm-up labels
  full               refiner_noteacher + the two BEV teacher caches and the two refiner teacher runs

--quick is the cheap pass (a few seconds, what `experiment.sh train` runs): manifests, directory listings, counts from
index files, a fixed sample of tokens.  Without it every token of a coverage check is looked at, the arrays are hashed
and all 85,109 tokens go through the training GT loader, whose statistics must equal the reference (a few minutes, CPU).
--only runs one group of the profile (dataset, metric_cache, gt, packed, labels, teacher_cache, teachers): the setup scripts
use it to check their own step.

--data trainval_simscale (default navtrain: the checks above, unchanged) checks the inputs of that recipe instead: its token
table (splits/trainval_simscale.parquet), the SimScale scene pickles and images, and for EVERY row of the table the metric
cache (SimScale: 41 GT maps extended at constant velocity to 51), the GT stores, the packed / raw256 / variant rows (row order = the table, token_sha16
of tokens.token_sha16), pdm_score_256_officialEP_trainval_simscale, the DET cache files and the MAP index (default teachers
bevformer + mask2map: BEVFusion and ReSMap need LiDAR or satellite images, which SimScale does not have) and the refiner
teacher runs.  A missing row is a FAIL.  --quick: a fixed sample of real and SimScale rows instead of every row.  A table
built in mini mode (development) is a WARN.

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
from navsim.agents.para_ssr.ck.tokens import token_sha16  # noqa: E402

PROFILES = ("base", "refiner_noteacher", "full")
SETUP = "tools/release/setup"
N_SAMPLE = 200                 # tokens of a --quick coverage check
N_READ = 8                     # sampled files opened and read

# ----------------------------------------------------------------------------------------------- pins
# sha256(array bytes)[:16] unless it says file (sha256 of the file)[:16].  tokens: navsim/agents/para_ssr/ck/tokens.token_sha16
# (= np.array(tokens, "U16") for 16-character tokens; a SimScale token is never truncated).
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
OFFICIAL_EP = {"npy": "a707bd3508ff04e9", "tokens": "b1278ec613611037", "shape": (103288, 5, 256)}   # agent.config.plan_score_file
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
# --data trainval_simscale
DATA = ("navtrain", "trainval_simscale")
TVSIM = "trainval_simscale"
TVSIM_TEACHERS = {"det": "bevformer", "map": "mask2map"}     # camera-only: the only teachers a SimScale scene can have
TVSIM_SCORE = "pdm_score_256_officialEP_trainval_simscale"
TABLE_COLUMNS = ("token", "log", "city", "row", "source", "round", "pkl")    # = navsim.common.simscale.TABLE_COLUMNS
SOURCES = ("navtrain_train", "navtrain_val", "simscale")      # the table's blocks, in row order
REPO_FILES_TVSIM = {"resources/metric_cache_saved_config_simscale.yaml": "12c573f170b3aa71"}
OBS_MAPS = 51                     # metric cache observation: proposals + 1 s of TTC look-ahead (5 s), every cache
OBS_EXTRAPOLATED_FROM = {"real": None, "simscale": 41}   # SimScale: 4 s GT, then constant velocity (PDMObservation)
SIM_CAMERA_FRAMES = (0, 1, 2, 3)                              # the camera teachers read all four history frames
VAR_NAMES = ["id", "a-1.0", "a-0.5", "a+0.5", "l-0.5", "l+0.5"]   # = e2e_data2.VNAMES


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
                 registry: Optional[dict], data: str = "navtrain"):
        self.p, self.quick, self.workers, self.rep = paths, quick, workers, rep
        self.det, self.mapt, self.registry = det, mapt, registry
        self.data = Path(paths["SSR_DATA_ROOT"])
        self.ck = self.data / "ck"
        self.data_key = data
        self._tokens: Dict[str, object] = {}

    # the tokens a training of --data reads: navtrain_train, or every row of the trainval_simscale table
    @property
    def train_split(self) -> str:
        return TVSIM if self.data_key == TVSIM else "navtrain_train"

    def train_tokens(self):
        return self.table() if self.data_key == TVSIM else self.tokens("navtrain_train")

    def train_coverage(self) -> List[Tuple[str, str]]:
        """train_tokens to look at: all, or with --quick a fixed sample (trainval_simscale: half of it SimScale rows)."""
        if self.data_key != TVSIM:
            return self.coverage("navtrain_train")
        df = self.table()
        if not self.quick:
            return list(zip(df.token.tolist(), df.log.tolist()))
        rng, pick = np.random.default_rng(0), []
        for m in ((df.source != "simscale").to_numpy(), (df.source == "simscale").to_numpy()):
            idx = np.flatnonzero(m)
            pick += sorted(rng.choice(idx, min(N_SAMPLE // 2, len(idx)), replace=False).tolist())
        return [(df.token.iat[i], df.log.iat[i]) for i in sorted(pick)]

    def table(self):
        """The trainval_simscale token table (TABLE_COLUMNS, row order = every row-indexed asset)."""
        if "_table" not in self._tokens:
            import pandas as pd
            p = self.data / "splits/trainval_simscale.parquet"
            if not p.is_file():
                raise FileNotFoundError(f"{p} missing ({SETUP}/16_trainval_simscale_table.sh)")
            df = pd.read_parquet(p)
            for k in ("token", "log", "source", "pkl"):
                df[k] = df[k].astype(str)
            self._tokens["_table"] = df
        return self._tokens["_table"]

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
        got = token_sha16(df.token.tolist())
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


def check_official_ep(c: Ctx) -> None:
    d = c.ck / "raw256"
    fix = (f"copy pdm_score_256_officialEP.npy and pdm_score_256_officialEP.tokens.json into {d}, or run "
           "tools/release/setup/assemble_official_ep.py (needs the raw256 labels of both splits)")
    npy, tok = d / "pdm_score_256_officialEP.npy", d / "pdm_score_256_officialEP.tokens.json"
    bad = []
    if not npy.is_file():
        bad.append(f"{npy} missing")
    elif npy_header(npy)[0] != OFFICIAL_EP["shape"]:
        bad.append(f"{npy}: shape {npy_header(npy)[0]} != {OFFICIAL_EP['shape']}")
    elif not c.quick and sha16_file(npy) != OFFICIAL_EP["npy"]:
        bad.append(f"{npy} differs from pin {OFFICIAL_EP['npy']}")
    if not tok.is_file():
        bad.append(f"{tok} missing")
    elif sha16_file(tok) != OFFICIAL_EP["tokens"]:
        bad.append(f"{tok} differs from pin {OFFICIAL_EP['tokens']}")
    c.verdict("official_ep", bad, f"{'shape + tokens' if c.quick else 'file + tokens'} equal their pins", fix)


def check_timm(c: Ctx) -> None:
    hub = Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface") / "hub"
    d = hub / "models--timm--resnet34.tv_in1k/snapshots"
    c.verdict("weights.resnet34", [] if _has_entry(d) else [f"not cached under {hub}"], f"cached under {hub}",
              "first training start downloads it (ImageNet weights, needs network); other backbones have their own "
              "file", soft=True)
    d = hub / "models--timm--vit_small_patch14_reg4_dinov2.lvd142m/snapshots"
    c.verdict("weights.dino_vits14", [] if _has_entry(d) else [f"not cached under {hub}"], f"cached under {hub}",
              "first training start downloads it (needs network): hf download timm/vit_small_patch14_reg4_dinov2.lvd142m "
              "model.safetensors", soft=True)


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
    if split == "navtrain_train":           # the training tokens of --data (trainval_simscale: every table row)
        split, tab, toks = c.train_split, c.train_tokens(), c.train_coverage()
        fix += " (SPLITS=trainval_simscale)" if split == TVSIM else ""
    else:
        tab, toks = c.tokens(split), c.coverage(split)
    bad = []
    done = root / "_state/done_tokens.txt"
    if not root.is_dir():
        c.verdict(f"metric_cache.{split}", [f"{root} missing"], "", fix, soft)
        return
    if done.is_file():
        have = set(done.read_text().split())
        lack = [t for t in tab.token if t not in have]
        if lack:
            bad.append(f"{len(lack)} tokens not in _state/done_tokens.txt, e.g. {lack[0]}")
    else:
        bad.append(f"{done} missing")
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
    c.verdict(f"metric_cache.{split}", bad, f"{len(paths)} of {len(tab)} tokens checked present; "
              f"{n_read} headers read", fix, soft)


def check_gt_stores(c: Ctx) -> None:
    fix = f"{SETUP}/30_gt_stores.sh" + (" (SPLIT=trainval_simscale)" if c.data_key == TVSIM else "")
    split = "navtrain_train"
    toks = c.train_coverage()
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
        for t, _ in (c.sample(split, N_READ) if c.data_key != TVSIM else toks[:: max(1, len(toks) // N_READ)][:N_READ]):
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
    toks = c.train_tokens().token.tolist()
    chunks = [(str(c.data), toks[i:i + 500]) for i in range(0, len(toks), 500)]
    with get_context("fork").Pool(max(1, c.workers)) as p:
        rows = [r for rr in p.imap_unordered(_gt_chunk, chunks) for r in rr]
    got = gt_stats(rows)
    if c.data_key == TVSIM:     # no reference statistics: every row loads, with a finite p_pdm, a centerline, an sdf
        want = dict(gt_n=len(toks), gt_ok=len(toks), p_pdm_nonfinite=0, cl_n_lt2=0, sdf_absmax_zero=0)
    else:
        want = GT_EXPECT
    c.verdict("gt.loader_all_tokens", compare_gt(got, want), json.dumps(got), fix)


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
    got = token_sha16(toks)
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
    fix = f"{SETUP}/60_teacher_bev.sh" if c.data_key != TVSIM else f"{SETUP}/65_cam_teacher_bev.sh (real + SimScale rows)"
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
    if int(m.get("num_samples_written", -1)) < len(c.train_tokens()):
        bad.append(f"num_samples_written {m.get('num_samples_written')}")
    toks = c.train_coverage()
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
    fix = f"{SETUP}/60_teacher_bev.sh" if c.data_key != TVSIM else f"{SETUP}/65_cam_teacher_bev.sh (real + SimScale rows)"
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
    lack = [tk for tk in c.train_tokens().token if tk not in idx]
    if lack:
        bad.append(f"{len(lack)} {c.train_split} tokens not in index.json, e.g. {lack[0]}")
    toks = c.train_coverage()
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
    c.verdict(tag, bad, f"meta sha {str(m.get('checkpoint_sha256'))[:16]}; all {len(c.train_tokens())} tokens in "
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


# ----------------------------------------------------------------------------------------------- checks: trainval_simscale
SORTED_SHA16 = {"navtrain_train": "c3fa309cbde971e8", "navtrain_val": "05b78ec43b95d954"}   # = constants.SPLITS pins


def _setup_module(name: str):
    """A script of tools/release/setup as a module (that directory is not a package)."""
    import importlib.util
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / SETUP / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


def _real_sim(df) -> Tuple[np.ndarray, np.ndarray]:
    sim = (df.source == "simscale").to_numpy()
    return ~sim, sim


def check_table(c: Ctx) -> None:
    fix = f"{SETUP}/16_trainval_simscale_table.sh"
    p = c.data / "splits/trainval_simscale.parquet"
    if not p.is_file():
        c.verdict("tvsim.table", [f"{p} missing"], "", fix)
        return
    df = c.table()
    bad: List[str] = []
    if tuple(df.columns) != TABLE_COLUMNS:
        c.verdict("tvsim.table", [f"columns {list(df.columns)} != {list(TABLE_COLUMNS)}"], "", fix)
        return
    real, sim = _real_sim(df)
    if not (df.row.to_numpy() == np.arange(len(df))).all():
        bad.append("row is not 0..N-1")
    if df.token.duplicated().any():
        bad.append(f"{int(df.token.duplicated().sum())} duplicate tokens")
    blocks = [SOURCES.index(x) if x in SOURCES else -1 for x in df.source]
    if min(blocks, default=0) < 0 or blocks != sorted(blocks):
        bad.append(f"sources are not the blocks {SOURCES} in this order")
    rnd = df["round"].to_numpy().astype(np.int64)
    if (rnd[real] != -1).any() or (df.pkl[real] != "").any() or (df.token[real].str.len() != 16).any():
        bad.append("a real row with a round, a pkl or a token that is not 16 characters")
    st = df[sim]
    rr = st.token.str.extract(r"^[0-9a-f]{16}-(\d{3})$")[0]
    suffix_ok = rr.notna() & (rr.fillna("-1").astype(int) == st["round"].astype(int))
    pkl_ok = st.pkl == [f"synthetic_reaction_pdm_v1.0-{int(r)}/meta_datas/{lg}.pkl" for r, lg in zip(st["round"], st.log)]
    if not (suffix_ok.all() and pkl_ok.all()):
        bad.append(f"{int((~(suffix_ok & pkl_ok)).sum())} SimScale rows whose token is not '<16 hex>-<round>' or whose "
                   f"pkl is not their round's scene pickle")
    if not (st.sort_values(["round", "token"], kind="stable").index == st.index).all():
        bad.append("SimScale rows are not in round order, sorted by token")
    T = _setup_module("build_trainval_simscale_table")
    src_logs = [T.source_log(lg) if s_ else lg for lg, s_ in zip(df.log, sim)]
    test = T.test_logs()
    leak = {lg for lg in src_logs if lg in test}
    if leak:
        bad.append(f"{len(leak)} test / navtest logs are training rows, e.g. {sorted(leak)[0]}")
    meta_p = p.with_suffix(".meta.json")
    mode = json.loads(meta_p.read_text()).get("mode") if meta_p.is_file() else "unknown (no .meta.json)"
    if mode == "full":
        for part, want in SORTED_SHA16.items():
            toks = df.token[df.source == part].tolist()
            got = hashlib.sha256("\n".join(sorted(toks)).encode()).hexdigest()[:16]
            if len(toks) != N_TOKENS[part] or got != want:
                bad.append(f"{part}: {len(toks)} tokens, sorted sha16 {got} (pinned {N_TOKENS[part]}, {want})")
    n = {s_: int((df.source == s_).sum()) for s_ in SOURCES}
    c.verdict("tvsim.table", bad, f"{len(df)} rows {n}, rounds {sorted(set(rnd[sim].tolist()))}, token_sha16 "
              f"{token_sha16(df.token.tolist())}", fix)
    c.verdict("tvsim.table.mode", [] if mode == "full" else [f"mode {mode}: a development selection, not the full split"],
              "full table, real parts equal the navtrain pins", f"{fix} without MINI=1 on the full download", soft=True)


def _scene_images(a) -> List[str]:
    """Runs in a worker: (sim_root, [pkl]) -> image paths (CAM_F0/L0/R0, frames 0..3) that are missing or unreadable."""
    root, pkls = a
    T = _setup_module("build_trainval_simscale_table")
    out = []
    for pk in pkls:
        r = Path(root) / pk
        try:
            frames = T.load_scene_pickle(r)
            files = [Path(root) / Path(pk).parts[0] / T.HIST_DIR / frames[i]["cams"][cam]["data_path"]
                     for i in SIM_CAMERA_FRAMES for cam in CAMERAS]
        except Exception as e:  # noqa: BLE001
            out.append(f"{r}: {type(e).__name__}")
            continue
        out += [str(f) for f in files if not f.is_file()]
    return out


def check_dataset_simscale(c: Ctx) -> None:
    fix = f"{SETUP}/12_download_simscale.sh (a table row whose scene is gone: rebuild the table with 16_*)"
    df = c.table()
    st = df[df.source == "simscale"]
    root = Path(c.p["SSR_SIMSCALE_ROOT"])
    bad = []
    names: Dict[str, set] = {}               # every row, both modes: one listing per round directory, no stat per scene
    for parent in sorted({os.path.dirname(pk) for pk in st.pkl}):
        d = root / parent
        names[parent] = {e.name for e in os.scandir(d)} if d.is_dir() else set()
    lost = [root / pk for pk in st.pkl if os.path.basename(pk) not in names[os.path.dirname(pk)]]
    if lost:
        bad.append(f"{len(lost)} of {len(st)} scene pickles missing, e.g. {lost[0]}")
    pick = st.pkl.tolist() if not c.quick else st.pkl.sample(min(N_SAMPLE, len(st)), random_state=0).tolist()
    lost_set = set(lost)
    pick = [pk for pk in pick if root / pk not in lost_set]
    chunks = [(str(root), pick[i:i + 256]) for i in range(0, len(pick), 256)]
    from multiprocessing import get_context
    with get_context("fork").Pool(max(1, min(c.workers, len(chunks) or 1))) as pool:
        miss = [x for xs in pool.imap_unordered(_scene_images, chunks) for x in xs]
    if miss:
        bad.append(f"{len(miss)} images of {len(pick)} scenes missing, e.g. {miss[0]}")
    c.verdict("dataset.simscale", bad, f"{len(st)} scene pickles; {len(SIM_CAMERA_FRAMES) * len(CAMERAS) * len(pick)} "
              f"images of {len(pick)} scenes present", fix)


def check_metric_cache_window(c: Ctx) -> None:
    """Full mode: a few metric caches unpickled; every one observes 5 s (51 maps): real ones the GT future, SimScale ones
    4 s of it (maps 0-40) and constant velocity after it (extend_constant_velocity, maps 41-50)."""
    if c.quick:
        return
    fix = f"{SETUP}/20_metric_cache_navtrain.sh SPLITS=trainval_simscale (move the wrong caches away first)"
    os.environ.setdefault("NUPLAN_MAPS_ROOT", c.p["NUPLAN_MAPS_ROOT"])
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    import lzma
    df = c.table()
    bad, n = [], 0
    for kind, m in zip(("real", "simscale"), _real_sim(df)):
        rows = df[m]
        for t, lg in list(zip(rows.token, rows.log))[:: max(1, len(rows) // (N_READ // 2))][: N_READ // 2]:
            p = c.data / "metric_cache" / lg / "unknown" / t / "metric_cache.pkl"
            if not p.is_file():
                continue
            with lzma.open(p, "rb") as f:
                obs = pickle.load(f).observation
            n += 1
            got = (len(obs._occupancy_maps), getattr(obs, "_extrapolated_from", None))
            if got != (OBS_MAPS, OBS_EXTRAPOLATED_FROM[kind]):
                bad.append(f"{p}: {got[0]} observation maps, extrapolated from {got[1]}; a {kind} cache has {OBS_MAPS}, "
                           f"extrapolated from {OBS_EXTRAPOLATED_FROM[kind]}")
    c.verdict("metric_cache.window", bad, f"{n} caches read: 51 maps, SimScale ones extrapolated from map 41", fix)


def _rows_equal_by_token(tokens: Sequence[str], arrays: Dict[str, np.ndarray], other: Path, names) -> Tuple[int, int]:
    """(shared rows, rows that differ) between this split's arrays and another packed-style directory, by token."""
    import pandas as pd
    o = pd.read_parquet(other / "tokens.parquet", columns=["token"]).token.astype(str).tolist()
    row = {t: i for i, t in enumerate(tokens)}
    pairs = [(i, row[t]) for i, t in enumerate(o) if t in row]
    if not pairs:
        return 0, 0
    oi, ni = (np.array(x, np.int64) for x in zip(*pairs))
    srt = np.argsort(oi)
    n_bad = 0
    for k in names:
        theirs = np.asarray(np.load(other / f"{k}.npy", mmap_mode="r")[oi[srt]])
        mine = np.asarray(arrays[k][ni[srt]])
        n_bad = max(n_bad, int((theirs != mine).reshape(len(ni), -1).any(1).sum()))
    return len(ni), n_bad


def check_packed_tvsim(c: Ctx) -> None:
    import pandas as pd
    d = c.ck / "packed" / TVSIM
    fix = f"{SETUP}/40_packed_from_dataset.sh (SPLITS=trainval_simscale)"
    need = ["tokens.parquet", "gt_traj.npy", "ok.npy", "status.npy", "meta.json"]
    lack = [f for f in need if not (d / f).is_file()]
    if lack:
        c.verdict(f"packed.{TVSIM}", [f"{d}: {lack} missing"], "", fix)
        return
    tab = c.table()
    n = len(tab)
    t = pd.read_parquet(d / "tokens.parquet")
    bad = []
    if t.token.astype(str).tolist() != tab.token.tolist() or t.log.astype(str).tolist() != tab.log.tolist():
        bad.append("tokens.parquet token / log are not the table's (row order)")
    if not (t["row"].to_numpy() == np.arange(len(t))).all():
        bad.append("tokens.parquet row is not 0..N-1")
    want = {"gt_traj": ((n, 8, 3), "float32"), "status": ((n, 8), "float32"), "ok": ((n,), "bool")}
    for k, w in want.items():
        if npy_header(d / f"{k}.npy") != w:
            bad.append(f"{k}.npy {npy_header(d / f'{k}.npy')}, expected {w}")
    m = json.loads((d / "meta.json").read_text())
    if m.get("token_sha16") != token_sha16(tab.token.tolist()):
        bad.append(f"meta.json token_sha16 {m.get('token_sha16')} != the table's")
    if not bad:
        ok = np.load(d / "ok.npy")
        if not ok.all():
            bad.append(f"{int((~ok).sum())} rows without a finite GT trajectory (the training would drop them)")
        if not c.quick:
            arrays = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("gt_traj", "status", "ok")}
            for part in ("navtrain_train", "navtrain_val"):
                if (c.ck / "packed" / part / "tokens.parquet").is_file():
                    shared, diff = _rows_equal_by_token(tab.token.tolist(), arrays, c.ck / "packed" / part, arrays)
                    if diff:
                        bad.append(f"{diff} of {shared} rows differ from packed/{part} (same token, other values)")
    c.verdict(f"packed.{TVSIM}", bad, f"{n} rows in table order, all ok; real rows {m.get('real_rows')}", fix)


def check_labels_tvsim(c: Ctx) -> None:
    fix = f"{SETUP}/50_warmup_labels.sh (SPLITS=trainval_simscale)"
    tab = c.table()
    n, sha = len(tab), token_sha16(tab.token.tolist())
    d = c.ck / "labels" / TVSIM / "raw256"
    bad = []
    for a, shape in (("labels", (n, 256, 9)), ("ok", (n, 256))):
        p = d / f"{a}.npy"
        if not p.is_file():
            bad.append(f"{p} missing")
        elif npy_header(p)[0] != shape:
            bad.append(f"{p}: shape {npy_header(p)[0]} != {shape}")
    plan = d / "shards/_plan.json"
    if not plan.is_file() or json.loads(plan.read_text()).get("token_sha16") != sha:
        bad.append(f"{plan}: missing or token_sha16 is not the table's")
    if not bad and not np.load(d / "ok.npy").all():
        bad.append(f"{int((~np.load(d / 'ok.npy')).sum())} anchors not scored (the training keeps a row only with all 256)")
    c.verdict(f"labels.raw256.{TVSIM}", bad, f"[{n}, 256, 9], every anchor scored, shard plan of this table", fix)
    vd = c.ck / "ck2/labels" / TVSIM / VAR_NAME
    bad = []
    for a, shape in (("labels", (n, 96, 9)), ("ok", (n, 96)), ("traj", (n, 96, 8, 3))):
        p = vd / f"{a}.npy"
        if not p.is_file():
            bad.append(f"{p} missing")
        elif npy_header(p)[0] != shape:
            bad.append(f"{p}: shape {npy_header(p)[0]} != {shape}")
    if not (vd / "build.json").is_file() or not (vd / "index.npz").is_file():
        bad.append(f"{vd}: build.json or index.npz missing")
    else:
        b = json.loads((vd / "build.json").read_text())
        for k, want in (("token_sha16", sha), ("anchors_sha16", ANCHORS_SHA16), ("n_built", n), ("n_tokens", n),
                        ("names", VAR_NAMES), ("n_anchors", 16)):
            if b.get(k) != want:
                bad.append(f"build.json {k} {b.get(k)} != {want}")
        own = str((b.get("label_source") or {}).get("path", ""))
        if not own.endswith(f"{TVSIM_SCORE}.npy"):
            bad.append(f"build.json label_source {own or 'not recorded'} is not {TVSIM_SCORE}.npy")
        sm = c.ck / "raw256" / f"{TVSIM_SCORE}.meta.json"
        want = json.loads(sm.read_text()).get("npy_sha16") if sm.is_file() else "(no score meta)"
        if (b.get("label_source") or {}).get("npy_sha16") != want:
            bad.append(f"build.json label_source npy_sha16 {(b.get('label_source') or {}).get('npy_sha16')} != the "
                       f"score file's {want}: the variants were built from another score file")
        from navsim.agents.para_ssr.ck.anchor_sampler import AnchorSampler
        drop = ("label_source",)
        if {k: v for k, v in (b.get("sampler") or {}).items() if k not in drop} != \
                {k: v for k, v in AnchorSampler(seed=int(b.get("seed", 0))).config().items() if k not in drop}:
            bad.append("build.json sampler differs from the current AnchorSampler")
        if not bad:
            with np.load(vd / "index.npz", allow_pickle=False) as z:
                if not z["built"].all():
                    bad.append(f"{int((~z['built']).sum())} rows not built")
            ok = np.load(vd / "ok.npy")
            if not ok.any(1).all():
                bad.append(f"{int((~ok.any(1)).sum())} rows without a scored variant (the training would drop them)")
    c.verdict(f"labels.var.{TVSIM}", bad, f"[{n}, 96, ...], build.json of this table and sampler", fix)


def check_score_tvsim(c: Ctx) -> None:
    d = c.ck / "raw256"
    fix = f"{SETUP}/assemble_trainval_simscale_score.py (50_warmup_labels.sh SPLITS=trainval_simscale)"
    npy, tok, meta = (d / f"{TVSIM_SCORE}{x}" for x in (".npy", ".tokens.json", ".meta.json"))
    tab = c.table()
    bad = []
    if not npy.is_file() or not tok.is_file():
        c.verdict("official_ep.trainval_simscale", [f"{npy} or its .tokens.json missing"], "", fix)
        return
    if npy_header(npy) != ((len(tab), 5, 256), "float16"):
        bad.append(f"{npy}: {npy_header(npy)}, expected [{len(tab)}, 5, 256] float16")
    if json.loads(tok.read_text()) != tab.token.tolist():
        bad.append(f"{tok}: not the table's tokens in row order")
    m = json.loads(meta.read_text()) if meta.is_file() else {}
    for k in ("real_vs_pinned_officialEP", "real_vs_wote_nc_dac_ttc_c"):
        if isinstance(m.get(k), dict) and m[k].get("n_differ"):
            bad.append(f"{meta.name}: {k} {m[k]}")
    raw = c.ck / "labels" / TVSIM / "raw256" / "labels.npy"
    if not c.quick and (not raw.is_file() or m.get("labels_sha16") != sha16_file(raw)):
        bad.append(f"{meta.name}: made from raw256 labels sha16 {m.get('labels_sha16')}, {raw} is not those (a raw256 "
                   f"rebuilt after it): move the score file away and rerun")
    if not bad and not c.quick:
        a = np.load(npy, mmap_mode="r")
        if not np.isfinite(np.asarray(a[:, [0, 1, 3, 4]])).all():
            bad.append("non-finite NC / DAC / TTC / C")
    ref = m.get("real_vs_pinned_officialEP", "not recorded")
    c.verdict("official_ep.trainval_simscale", bad, f"[{len(tab)}, 5, 256] in table order; real rows vs the pinned file: "
              f"{ref}", fix)


# ----------------------------------------------------------------------------------------------- profiles
BASE_CHECKS: Sequence[Callable[[Ctx], None]] = (
    check_env, check_navsim_import,
    lambda c: check_repo_files(c, REPO_FILES, "repo.pinned_files"),
    check_anchors, check_tokens, check_dataset, check_official_ep, check_timm, check_dataset_test,
    check_eval_assets)
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
# --data trainval_simscale: the table, the SimScale scenes and that split's assets; the navtrain_val retraining inputs
# and the pinned 103,288-row score file are not read by that training
TVSIM_BASE: Sequence[Callable[[Ctx], None]] = (
    check_env, check_navsim_import,
    lambda c: check_repo_files(c, REPO_FILES, "repo.pinned_files"),
    check_anchors, check_tokens, check_dataset, check_table, check_dataset_simscale, check_score_tvsim, check_timm,
    check_dataset_test, check_eval_assets)
TVSIM_REFINER: Sequence[Callable[[Ctx], None]] = (
    lambda c: check_repo_files(c, dict(REPO_FILES_REFINER, **REPO_FILES_TVSIM), "repo.pinned_files_refiner"),
    check_scorer, check_metric_cache, check_metric_cache_window, check_gt_stores, check_packed_tvsim, check_labels_tvsim)
TVSIM_TEACHER: Sequence[Callable[[Ctx], None]] = (check_teacher_caches, check_refiner_teachers)
CHECKS_BY_DATA = {"navtrain": CHECKS, TVSIM: {
    "base": TVSIM_BASE, "refiner_noteacher": tuple(TVSIM_BASE) + tuple(TVSIM_REFINER),
    "full": tuple(TVSIM_BASE) + tuple(TVSIM_REFINER) + tuple(TVSIM_TEACHER)}}
GROUPS_BY_DATA = {"navtrain": GROUPS, TVSIM: dict(
    GROUPS, dataset=(check_tokens, check_dataset, check_dataset_test, check_table, check_dataset_simscale),
    metric_cache=(check_metric_cache, check_metric_cache_window), packed=(check_packed_tvsim,),
    labels=(check_labels_tvsim, check_score_tvsim))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, choices=PROFILES)
    ap.add_argument("--data", default="navtrain", choices=DATA, help="the training data the profile is for")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--strict", action="store_true", help="a WARN also gives exit code 1")
    ap.add_argument("--only", default="", choices=[""] + sorted(GROUPS), help="one group of checks of this profile")
    ap.add_argument("--det-teacher", default=os.environ.get("DET_TEACHER") or "",
                    help="default bevfusion (--data trainval_simscale: bevformer)")
    ap.add_argument("--map-teacher", default=os.environ.get("MAP_TEACHER") or "",
                    help="default resmap (--data trainval_simscale: mask2map)")
    ap.add_argument("--registry", default=os.environ.get("SSR_RELEASE_REGISTRY") or str(ROOT / REGISTRY))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--json", default="", help="also write the rows to this file")
    a = ap.parse_args(argv)
    paths = SP.resolve()
    rep = Report()
    registry = load_registry(Path(a.registry), paths)
    tv = a.data == TVSIM
    det = a.det_teacher or (TVSIM_TEACHERS["det"] if tv else "bevfusion")
    mapt = a.map_teacher or (TVSIM_TEACHERS["map"] if tv else "resmap")
    c = Ctx(paths, a.quick, a.workers, rep, det, mapt, registry, data=a.data)
    print(f"[verify_assets] profile {a.profile}, {'quick' if a.quick else 'full check'}, data {paths['SSR_DATA_ROOT']}, "
          f"teachers {paths['SSR_TEACHER_CACHE']}" + (f", --data {a.data}" if tv else ""), flush=True)
    checks, groups = CHECKS_BY_DATA[a.data], GROUPS_BY_DATA[a.data]
    for fn in [f for f in checks[a.profile] if not a.only or f in groups[a.only]]:
        try:
            fn(c)
        except Exception as e:      # one broken input must not hide the others
            rep.add("FAIL", getattr(fn, "__name__", "check"), f"{type(e).__name__}: {e}",
                    "see the traceback line: " + traceback.format_exc(limit=-1).strip().splitlines()[-2].strip())
    code = 1 if rep.n("FAIL") or (a.strict and rep.n("WARN")) else 0
    print(f"[verify_assets] {rep.n('PASS')} pass, {rep.n('WARN')} warn, {rep.n('FAIL')} fail -> exit {code}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(dict(profile=a.profile, quick=a.quick, exit=code, rows=rep.rows,
                                                **({"data": a.data} if tv else {})), indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
