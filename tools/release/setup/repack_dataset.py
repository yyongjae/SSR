#!/usr/bin/env python
"""Build ck/packed/<split> from the dataset alone: no planner checkpoint and no dump directory.

  repack_dataset.py --split {navtrain_train,navtrain_val,trainval_simscale} [--root R | --out DIR] [--tokens PARQUET]
                    [--logs DIR] [--workers 2] [--expect-from OLD_PACKED_DIR] [--skip-done]

The refiner reads only tokens, ok, gt_traj (student) and status (teachers, label builders) from this directory.  These
are ground truth and ego state from the logs, so tools/refiner/extract_human.py gives them directly.

Output <root>/packed/<split>/ (default root = $SSR_DATA_ROOT/ck), or --out as is:
  tokens.parquet  token, log, city, row; row order of the split table
  gt_traj.npy     f32 [N, 8, 3]  extract_human traj (= Scene.get_future_trajectory(8).poses)
  status.npy      f32 [N, 8]     driving_command (4) + ego_dynamic_state vx, vy, ax, ay
  ok.npy          bool [N]       gt_traj finite
  meta.json       split, n, n_ok, source "dataset", sha16 of each array, anchors_sha16

The arrays are compared with the pins below before the directory appears: it is written to <out>.tmp<pid> and renamed at
the end, so a failed run leaves no usable directory (the temp directory stays, for diagnosis).  An existing output is
never overwritten (--skip-done: one whose meta.json token_sha16 is that of the token table is kept as finished).
--expect-from also compares row by row with an existing packed directory.  CPU only, ~1 min.

trainval_simscale (no pins): the rows of its token table (build_trainval_simscale_table.py); a SimScale row is its scene
pickle, whose 8 future frames are the scene window (extract_human min_future 8; gt_traj = frames 4..11).  Instead of
pins, its real rows must equal the packed navtrain_train / navtrain_val rows token by token where those directories
exist (<root>/packed/<split>), which meta.json records.  token_sha16 = navsim/agents/para_ssr/ck/tokens.token_sha16
(full-length tokens; U16 for 16-character tokens, so the navtrain pins are unchanged).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]      # repo root of this worktree
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/refiner"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from navsim.common import ssr_paths as SP  # noqa: E402

os.environ.setdefault("NUPLAN_MAPS_ROOT", SP.NUPLAN_MAPS_ROOT)        # read when nuplan is imported
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

# sha256(array bytes)[:16]; tokens = tokens.token_array (U16 here), the value e2e_data2 compares with meta.json token_sha16
PINS = {
    "navtrain_train": {"n": 85109, "token_sha16": "0b56b9aecbe1f094", "gt_traj_sha16": "ad3654bd81bdb82f",
                       "status_sha16": "d0d6f908e8be4f3c", "ok_sha16": "1f8f6dbc2f6c7496"},
    "navtrain_val": {"n": 18179, "token_sha16": "0fea22c41a4106b0", "gt_traj_sha16": "adb06837415973fd",
                     "status_sha16": "2309a38c18a61c48", "ok_sha16": "0bcf0f7a51703e00"},
}
UNPINNED = ("trainval_simscale",)            # table-driven splits, checked against the navtrain parts instead
REAL_PARTS = ("navtrain_train", "navtrain_val")
ANCHORS = ROOT / "resources/planning_vb/trajectory_anchors_256.npy"
TRACKED_TOKENS = ROOT / "resources/km_corridor/packed/{split}/tokens.parquet"
ARRAYS = ("gt_traj", "status", "ok")


def sha16(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def token_sha16(tokens) -> str:
    from navsim.agents.para_ssr.ck.tokens import token_sha16 as _token_sha16
    return _token_sha16(tokens)


def read_tokens(split: str, path: Optional[str] = None) -> pd.DataFrame:
    """token, log, city in row order: --tokens, else the split table, else the copy tracked in this repo."""
    if path:
        src = Path(path)
    else:
        from navsim.agents.para_ssr.ck import constants as C
        src = Path(C.SPLITS[split]["tokens"])
        if not src.is_file():
            src = Path(str(TRACKED_TOKENS).format(split=split))
    df = pd.read_parquet(src)
    if "city" not in df.columns:
        df["city"] = df["map_location"] if "map_location" in df.columns else ""
    df = df[["token", "log", "city"] + (["pkl"] if "pkl" in df.columns else [])].reset_index(drop=True)
    df["token"] = df.token.astype(str)
    if df.token.duplicated().any():
        raise SystemExit(f"{src}: duplicate tokens")
    return df


def build_arrays(df: pd.DataFrame, logs_dir: Path, workers: int = 2) -> Dict[str, np.ndarray]:
    """df: token, log[, pkl] -> gt_traj, status, ok, row aligned with df (pkl: a SimScale scene, under SSR_SIMSCALE_ROOT)."""
    import extract_human as H
    d = H.extract_split(df, Path(logs_dir), workers, sim_root=Path(SP.SSR_SIMSCALE_ROOT))
    if not (d["tokens"] == df.token.to_numpy()).all():
        raise SystemExit("extract_human changed the row order")
    gt = d["traj"].astype(np.float32)
    status = np.concatenate([d["cmd_raw"].astype(np.float32), d["eds"].astype(np.float32)], 1)
    return {"gt_traj": gt, "status": status, "ok": np.isfinite(gt).all((1, 2))}


def array_shas(tokens, arrays: Dict[str, np.ndarray]) -> Dict[str, object]:
    out = {"n": len(tokens), "token_sha16": token_sha16(tokens)}
    out.update({f"{k}_sha16": sha16(arrays[k]) for k in ARRAYS})
    return out


def compare_pins(got: Dict[str, object], want: Dict[str, object]) -> List[str]:
    return [f"{k}: got {got.get(k)} expected {v}" for k, v in want.items() if got.get(k) != v]


def compare_with_dir(df: pd.DataFrame, arrays: Dict[str, np.ndarray], old: Path) -> List[str]:
    """Row-by-row comparison with an existing packed directory."""
    bad = []
    o = pd.read_parquet(old / "tokens.parquet")
    for col in ("token", "log", "city"):
        if len(o) != len(df) or not (o[col].astype(str).to_numpy() == df[col].astype(str).to_numpy()).all():
            bad.append(f"tokens.parquet column {col} differs from {old}")
    if not (o["row"].to_numpy() == np.arange(len(o))).all():
        bad.append(f"{old}/tokens.parquet: row is not 0..N-1")
    for k in ARRAYS:
        a = np.load(old / f"{k}.npy", mmap_mode="r")
        if a.dtype != arrays[k].dtype or a.shape != arrays[k].shape:
            bad.append(f"{k}: {a.dtype} {a.shape} in {old}, {arrays[k].dtype} {arrays[k].shape} here")
        elif not np.array_equal(np.asarray(a), arrays[k]):
            bad.append(f"{k}: {int((np.asarray(a) != arrays[k]).reshape(len(a), -1).any(1).sum())} rows differ from {old}")
    return bad


def compare_real_parts(df: pd.DataFrame, arrays: Dict[str, np.ndarray], packed_root: Path):
    """A table-driven split's rows against the packed navtrain parts, token by token, where a part exists.
    -> (problems, {part: what was compared})."""
    bad, seen = [], {}
    row = {t: i for i, t in enumerate(df.token.tolist())}
    for part in REAL_PARTS:
        d = Path(packed_root) / part
        if not (d / "tokens.parquet").is_file():
            seen[part] = "not compared (no packed directory)"
            continue
        o = pd.read_parquet(d / "tokens.parquet", columns=["token"]).token.astype(str).tolist()
        mine = [(i, row[t]) for i, t in enumerate(o) if t in row]
        if not mine:
            seen[part] = "not compared (no shared token)"
            continue
        oi, ni = (np.array(x, np.int64) for x in zip(*mine))
        n_bad = 0
        for k in ARRAYS:
            a = np.asarray(np.load(d / f"{k}.npy", mmap_mode="r")[np.sort(oi)])[np.argsort(np.argsort(oi))]
            n_bad = max(n_bad, int((a != arrays[k][ni]).reshape(len(ni), -1).any(1).sum()))
        if n_bad:
            bad.append(f"{n_bad} of {len(ni)} rows differ from {d} (the same token must give the same row)")
        seen[part] = f"{len(ni) - n_bad} of {len(ni)} shared rows equal"
    return bad, seen


def write_dir(out: Path, df: pd.DataFrame, arrays: Dict[str, np.ndarray], meta: Dict[str, object]) -> None:
    out.mkdir(parents=True)
    d = df[["token", "log", "city"]].copy()
    d["row"] = np.arange(len(d))
    d.to_parquet(out / "tokens.parquet", index=False)
    for k in ARRAYS:
        np.save(out / f"{k}.npy", arrays[k], allow_pickle=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def run(split: str, out: Path, df: pd.DataFrame, logs_dir: Path, workers: int = 2, expect_from: Optional[str] = None,
        pins: Optional[Dict[str, object]] = None, packed_root: Optional[Path] = None) -> Dict[str, object]:
    """Build, compare, publish.  SystemExit(1) when a comparison fails; nothing appears at `out` in that case.
    An unpinned split is compared with the packed navtrain parts under packed_root (default: next to `out`)."""
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}: choose another --out or move it away first")
    t0 = time.time()
    arrays = build_arrays(df, logs_dir, workers)
    shas = array_shas(df.token.tolist(), arrays)
    meta = dict(split=split, source="dataset", n_ok=int(arrays["ok"].sum()), logs_dir=str(logs_dir), **shas)
    if ANCHORS.is_file():
        from navsim.agents.para_ssr.ck.anchor_sampler import ANCHORS_SHA16
        meta["anchors_sha16"] = ANCHORS_SHA16
    meta["created"] = time.strftime("%F %T")
    meta["seconds"] = round(time.time() - t0, 1)
    pins = (PINS[split] if split not in UNPINNED else {}) if pins is None else pins
    bad = compare_pins(shas, pins)
    if split in UNPINNED:
        real_bad, meta["real_rows"] = compare_real_parts(df, arrays, packed_root or out.parent)
        meta["n_simscale"] = int((df["pkl"].astype(str) != "").sum()) if "pkl" in df else 0
        bad += real_bad
    if expect_from:
        bad += compare_with_dir(df, arrays, Path(expect_from))
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}")
    write_dir(tmp, df, arrays, meta)
    if bad:
        print(json.dumps({"split": split, "result": "FAIL", "mismatch": bad, "kept_for_diagnosis": str(tmp)}, indent=1))
        raise SystemExit(1)
    os.replace(tmp, out)
    report = dict(meta, result="PASS", pins="equal" if split not in UNPINNED else "none (real rows compared)",
                  expect_from=expect_from or "not given", out=str(out))
    print(json.dumps(report, indent=1))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(PINS) + list(UNPINNED))
    ap.add_argument("--root", default="", help="CK data root (default $SSR_DATA_ROOT/ck); output <root>/packed/<split>")
    ap.add_argument("--out", default="", help="output directory, used as is")
    ap.add_argument("--tokens", default="", help="token table (default: the split table, else the tracked copy)")
    ap.add_argument("--logs", default="", help="log pickles (default $OPENSCENE_DATA_ROOT/navsim_logs/trainval)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--expect-from", default="", help="existing packed directory to compare row by row")
    ap.add_argument("--skip-done", action="store_true", help="an existing output of this token table is finished")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(a.root or Path(SP.SSR_DATA_ROOT) / "ck") / "packed" / a.split
    logs = Path(a.logs or Path(SP.OPENSCENE_DATA_ROOT) / "navsim_logs/trainval")
    df = read_tokens(a.split, a.tokens or None)
    if a.skip_done and (out / "meta.json").is_file():
        got = json.loads((out / "meta.json").read_text()).get("token_sha16")
        if got == token_sha16(df.token.tolist()):
            print(f"{out}: finished for this token table (token_sha16 {got}), kept")
            return 0
        raise SystemExit(f"{out} was built from another token table (token_sha16 {got}): move it away first")
    run(a.split, out, df, logs, a.workers, a.expect_from or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
