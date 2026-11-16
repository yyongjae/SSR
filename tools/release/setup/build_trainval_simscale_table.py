#!/usr/bin/env python
"""Token table of the trainval_simscale split: navtrain_train, navtrain_val, then the SimScale rounds.

  build_trainval_simscale_table.py [--train PARQUET] [--val PARQUET] [--sim-root DIR] [--rounds 0 1 2 3 4]
                                   [--max-per-source N | SOURCE=N ...] [--out FILE] [--workers 16]

Columns navsim/common/simscale.TABLE_COLUMNS: token (full length), log, city, row (= position), source (navtrain_train,
navtrain_val, simscale), round (int8, -1 for a real row), pkl (SimScale: the scene pickle relative to SSR_SIMSCALE_ROOT,
simscale.scene_pkl(); real: '').  Rows: navtrain_train in the order of --train (default the split table of
constants.SPLITS, made by 15_token_lists.sh), navtrain_val in the order of --val, then rounds 0..4, each sorted by token.
This row order is the row order of every row-indexed asset of the split.

A SimScale scene <log>-<scene>-<rrr>.pkl is kept when it has 12 frames, its frame 3 token is '<16 hex>-<rrr>' of its
round (that token is the table token), frame 3 has the pickle's log name and a route, no object of frame 3 overlaps the
ego box (a simulated start inside another object: NAVSIM v2 / SimScale scoring drops such objects, a v1 metric cache
keeps them and scores every trajectory of the scene a collision; ~0.16 % of round 4), and the CAM_F0 / CAM_L0 / CAM_R0
images of frames 0..3 are in the round's sensor_blobs_hist.  Otherwise it is dropped and counted per reason.  The
pickles are downloaded data: they are read with an unpickler that only resolves the numpy names a frame list needs.

Refused (nothing is written): a scene whose source log (the pickle name without '-<scene>-<rrr>') is a test or navtest
log; a duplicate token; a round directory that is missing.  In full mode also: a real part that is not its pinned split
(n and the sha16 of the sorted tokens, constants.SPLITS), a round with another number of scene pickles than published,
a round that loses more than --max-drop-frac of its scenes to the filter (an incomplete download; ego_overlap drops,
a property of the data, are not counted there).
Mini mode (--max-per-source): the first N rows of a source (SimScale: after the filter), no pins; for development.

Output: --out (default constants.SPLITS['trainval_simscale']['tokens']), <out>.meta.json (counts, drop reasons, input
hashes, token_sha16), and the SimScale rows alone as constants.SPLITS['simscale']['tokens'] (row = position there).  The
table is written last: it marks the output complete and is never overwritten (the other two files of a run that stopped
before it are replaced).  CPU only; reading the pickles takes ~1 min per 50k scenes at 16 workers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]      # repo root of this worktree
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from navsim.common import ssr_paths as SP  # noqa: E402
from navsim.agents.para_ssr.ck.tokens import token_sha16  # noqa: E402

# navsim.common.simscale names, repeated here: that module imports navsim.common.dataclasses (nuplan), and this tool
# runs before the environment has the maps (tests/test_simscale_assets.py checks they are equal)
TABLE_COLUMNS = ("token", "log", "city", "row", "source", "round", "pkl")
SOURCE_REAL = ("navtrain_train", "navtrain_val")
SOURCE_SIM = "simscale"
ROUNDS = (0, 1, 2, 3, 4)
NUM_FRAMES, CURRENT = 12, 3
META_DIR, HIST_DIR = "meta_datas", "sensor_blobs_hist"
CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0")         # the student's and the camera teachers' cameras
CAMERA_FRAMES = (0, 1, 2, 3)                     # history frames with images (BEVFormer reads all four)
# published scene pickles per round (HF simscale_pdm_v1.0-<r>_sensor_blobs_hist.json and the SimScale scene filters)
PUBLISHED_SCENES = {0: 65026, 1: 55471, 2: 46261, 3: 38412, 4: 32202}
LOG_SPLIT = ROOT / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
NAVTEST_FILTER = ROOT / "navsim/planning/script/config/common/scene_filter/navtest.yaml"
TRACKED = ROOT / "resources/km_corridor/packed/{split}/tokens.parquet"
DROP_REASONS = ("unpickle", "frames", "token", "log_name", "route", "ego_overlap", "camera")
DATA_REASONS = ("ego_overlap",)                  # drops that a complete download has too (not --max-drop-frac)
# the ego box of every NAVSIM metric cache (nuplan get_pacifica_parameters): length, width, rear axle to centre [m]
EGO_LENGTH, EGO_WIDTH, EGO_REAR_AXLE_TO_CENTER = 5.176, 2.297, 1.461
TOKEN_RE = re.compile(r"^[0-9a-f]{16}-(\d{3})$")


def round_dir(r: int) -> str:
    return f"synthetic_reaction_pdm_v1.0-{int(r)}"


def scene_pkl(r: int, log_stem: str) -> str:
    return f"{round_dir(r)}/{META_DIR}/{log_stem}.pkl"


def source_log(stem: str) -> str:
    """The real log a SimScale scene was branched from: '<log>-<scene>-<rrr>' -> '<log>' (log names hold '-' too)."""
    return stem.rsplit("-", 2)[0]


# ----------------------------------------------------------------------------------------------- scene pickles
class SceneUnpickler(pickle.Unpickler):
    """A SimScale scene pickle is downloaded data: only the numpy names an OpenScene frame list is made of resolve."""
    ALLOWED = {(m, n) for m in ("numpy.core.multiarray", "numpy._core.multiarray") for n in ("_reconstruct", "scalar")} | \
        {("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"{module}.{name} is not allowed in a scene pickle")
        return super().find_class(module, name)


def load_scene_pickle(path) -> list:
    with open(path, "rb") as f:
        return SceneUnpickler(f).load()


def ego_overlaps(boxes) -> int:
    """Number of annotation boxes ([x, y, z, length, width, height, heading] in the frame's ego coordinates, whose origin
    is the rear axle) that intersect the ego box: the objects a metric cache's first occupancy map has on the ego."""
    import shapely
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    if not len(b):
        return 0
    corners = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]]) * b[:, None, 3:5]     # [N, 4, 2]
    cos, sin = np.cos(b[:, 6])[:, None], np.sin(b[:, 6])[:, None]
    xy = np.stack([corners[..., 0] * cos - corners[..., 1] * sin, corners[..., 0] * sin + corners[..., 1] * cos], -1)
    ego = shapely.box(EGO_REAR_AXLE_TO_CENTER - EGO_LENGTH / 2, -EGO_WIDTH / 2,
                      EGO_REAR_AXLE_TO_CENTER + EGO_LENGTH / 2, EGO_WIDTH / 2)
    return int(shapely.intersects(shapely.polygons(xy + b[:, None, :2]), ego).sum())


def check_scene(a) -> Tuple[str, str, str, str]:
    """a = (sim_root, round, pickle name) -> (status, token, stem, city); status 'ok' or one of DROP_REASONS."""
    root, r, name = a
    stem = name[:-4]
    try:
        frames = load_scene_pickle(Path(root) / round_dir(r) / META_DIR / name)
    except Exception:  # noqa: BLE001  (corrupt or not a frame list: dropped, counted)
        return "unpickle", "", stem, ""
    if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
        return "frames", "", stem, ""
    cur = frames[CURRENT]
    tok = str(cur.get("token", ""))
    m = TOKEN_RE.match(tok)
    if m is None or int(m.group(1)) != int(r):
        return "token", tok, stem, ""
    if cur.get("log_name") != stem:
        return "log_name", tok, stem, ""
    if len(cur.get("roadblock_ids") or []) == 0:
        return "route", tok, stem, ""
    try:
        if ego_overlaps(cur["anns"]["gt_boxes"]):
            return "ego_overlap", tok, stem, ""
    except (KeyError, TypeError, ValueError):
        return "frames", tok, stem, ""
    sensors = Path(root) / round_dir(r) / HIST_DIR
    try:
        files = [sensors / frames[i]["cams"][c]["data_path"] for i in CAMERA_FRAMES for c in CAMERAS]
    except (KeyError, TypeError):
        return "camera", tok, stem, ""
    if not all(os.path.isfile(p) for p in files):
        return "camera", tok, stem, ""
    return "ok", tok, stem, str(cur.get("map_location", ""))


def scan_round(sim_root: Path, r: int, workers: int) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    """Kept scenes of a round (token, log, city, round, pkl; sorted by token), drop counts, number of pickles."""
    d = Path(sim_root) / round_dir(r) / META_DIR
    if not d.is_dir():
        raise SystemExit(f"round {r}: {d} missing (12_download_simscale.sh ROUNDS=..., or leave the round out with --rounds)")
    with os.scandir(d) as it:
        names = sorted(e.name for e in it if e.name.endswith(".pkl"))
    with Pool(max(1, workers)) as p:
        res = p.map(check_scene, [(str(sim_root), r, n) for n in names], chunksize=64)
    drop = {k: 0 for k in DROP_REASONS}
    rows = []
    for st, tok, stem, city in res:
        if st != "ok":
            drop[st] += 1
            continue
        rows.append((tok, stem, city))
    df = pd.DataFrame(rows, columns=["token", "log", "city"]).sort_values("token", kind="stable").reset_index(drop=True)
    df["round"] = np.int8(r)
    df["pkl"] = [scene_pkl(r, s) for s in df.log]
    return df, drop, len(names)


# ----------------------------------------------------------------------------------------------- real part
def read_real(path: Path, split: str) -> pd.DataFrame:
    """token, log, city of a navtrain split table, in file order; city from the copy tracked in this repo when the
    table has none (an older table of token and log only)."""
    df = pd.read_parquet(path)
    if "city" not in df.columns:
        if "map_location" in df.columns:
            df["city"] = df["map_location"]
        else:
            tr = pd.read_parquet(str(TRACKED).format(split=split), columns=["token", "city"])
            city = dict(zip(tr.token.astype(str), tr.city.astype(str)))
            lack = [t for t in df.token.astype(str) if t not in city]
            if lack:
                raise SystemExit(f"{path}: no city column and {len(lack)} tokens not in {TRACKED.format(split=split)}")
            df["city"] = [city[t] for t in df.token.astype(str)]
    df = df[["token", "log", "city"]].astype(str).reset_index(drop=True)
    if df.token.duplicated().any():
        raise SystemExit(f"{path}: duplicate tokens")
    return df


def sorted_sha16(tokens: Iterable[str]) -> str:
    """constants.SPLITS[...]['sha16_sorted_tokens'] (= derive_tokens.py): sha256 of the sorted tokens, one per line."""
    return hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest()[:16]


def real_pin_problems(split: str, df: pd.DataFrame, splits: dict) -> List[str]:
    want_n, want_sha = splits[split]["n"], splits[split]["sha16_sorted_tokens"]
    got = sorted_sha16(df.token.tolist())
    bad = []
    if len(df) != want_n:
        bad.append(f"{split}: {len(df)} tokens, pinned {want_n}")
    if got != want_sha:
        bad.append(f"{split}: sorted token sha16 {got}, pinned {want_sha}")
    return bad


def test_logs() -> set:
    """test_logs of the official log split and the navtest filter's logs: no training row may come from them."""
    logs = set(yaml.safe_load(LOG_SPLIT.read_text())["test_logs"])
    return logs | set(yaml.safe_load(NAVTEST_FILTER.read_text())["log_names"])


# ----------------------------------------------------------------------------------------------- table
def parse_caps(items: Sequence[str]) -> Dict[str, int]:
    """--max-per-source: 'N' (every source) and / or 'SOURCE=N' -> {source: N}."""
    caps: Dict[str, int] = {}
    for it in items:
        for x in str(it).split(","):
            if not x:
                continue
            k, _, v = x.rpartition("=")
            names = [k] if k else list(SOURCE_REAL) + [SOURCE_SIM]
            for s in names:
                if s not in SOURCE_REAL + (SOURCE_SIM,):
                    raise SystemExit(f"--max-per-source: unknown source {s!r}")
                caps[s] = int(v)
    return caps


def assemble(parts: Sequence[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """[(source, df token/log/city[/round/pkl])] in table order -> the table, TABLE_COLUMNS, row = position."""
    out = []
    for source, df in parts:
        d = df.copy()
        d["source"] = source
        if source != SOURCE_SIM:
            d["round"], d["pkl"] = np.int8(-1), ""
        out.append(d)
    t = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=list(TABLE_COLUMNS))
    t["row"] = np.arange(len(t), dtype=np.int64)
    t["round"] = t["round"].astype(np.int8)
    for c in ("token", "log", "city", "source", "pkl"):
        t[c] = t[c].astype(str)
    return t[list(TABLE_COLUMNS)]


def table_problems(t: pd.DataFrame, bad_logs: set) -> List[str]:
    bad = []
    dup = t.token[t.token.duplicated()]
    if len(dup):
        bad.append(f"{len(dup)} duplicate tokens, e.g. {dup.iloc[0]}")
    src = [source_log(lg) if s == SOURCE_SIM else lg for lg, s in zip(t.log, t.source)]
    leak = sorted({lg for lg in src if lg in bad_logs})
    if leak:
        n = sum(lg in bad_logs for lg in src)
        bad.append(f"{n} rows come from {len(leak)} test / navtest logs (a leak of the evaluation logs), e.g. {leak[0]}")
    return bad


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def file_sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def build(train: Path, val: Path, sim_root: Path, rounds: Sequence[int], caps: Dict[str, int], workers: int,
          max_drop_frac: float, splits: dict) -> Tuple[pd.DataFrame, dict]:
    t0 = time.time()
    mini = bool(caps)
    bad: List[str] = []
    real = {"navtrain_train": read_real(train, "navtrain_train"), "navtrain_val": read_real(val, "navtrain_val")}
    meta: Dict[str, object] = dict(mode="mini" if mini else "full", caps=caps, rounds=[int(r) for r in rounds],
                                   sim_root=str(sim_root), inputs={})
    for s, p in (("navtrain_train", train), ("navtrain_val", val)):
        meta["inputs"][s] = dict(path=str(p), n=len(real[s]), file_sha16=file_sha16(p),
                                 sorted_sha16=sorted_sha16(real[s].token.tolist()))
        if not mini:
            bad += real_pin_problems(s, real[s], splits)
    sims, rounds_meta = [], {}
    for r in rounds:
        df, drop, n_pkl = scan_round(sim_root, r, workers)
        rounds_meta[int(r)] = dict(pickles=n_pkl, kept=len(df), dropped=drop)
        n_drop = n_pkl - len(df)
        n_lost = n_drop - sum(drop[k] for k in DATA_REASONS)
        print(f"round {r}: {n_pkl} scene pickles, kept {len(df)}, dropped {n_drop} {drop}", flush=True)
        if not mini:
            if n_pkl != PUBLISHED_SCENES[int(r)]:
                bad.append(f"round {r}: {n_pkl} scene pickles, published {PUBLISHED_SCENES[int(r)]} (meta_datas incomplete)")
            if n_pkl and n_lost / n_pkl > max_drop_frac:
                bad.append(f"round {r}: {n_lost} of {n_pkl} scenes lost to the filter {drop} (> {max_drop_frac:.0%}: are all "
                           f"sensor_blobs_hist parts there?)")
        sims.append(df)
    meta["rounds_scanned"] = rounds_meta
    sim = pd.concat(sims, ignore_index=True) if sims else pd.DataFrame(columns=["token", "log", "city", "round", "pkl"])
    parts = [(s, real[s]) for s in SOURCE_REAL] + [(SOURCE_SIM, sim)]
    parts = [(s, df.iloc[: caps[s]] if s in caps else df) for s, df in parts]
    t = assemble(parts)
    bad += table_problems(t, test_logs())
    if bad:
        raise SystemExit("refusing to write the table: " + "; ".join(bad))
    meta.update(n=len(t), n_by_source={s: int((t.source == s).sum()) for s, _ in parts},
                n_by_round={int(r): int((t["round"] == r).sum()) for r in rounds}, token_sha16=token_sha16(t.token),
                sorted_sha16_real={s: sorted_sha16(t.token[t.source == s]) for s in SOURCE_REAL},
                pinned_real="checked" if not mini else "not checked (mini)", test_log_leak="none",
                created=time.strftime("%F %T"), seconds=round(time.time() - t0, 1))
    return t, meta


def main(argv=None) -> int:
    from navsim.agents.para_ssr.ck import constants as C
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=C.SPLITS["navtrain_train"]["tokens"], help="navtrain_train table, rows in order")
    ap.add_argument("--val", default=C.SPLITS["navtrain_val"]["tokens"], help="navtrain_val table, rows in order")
    ap.add_argument("--sim-root", default=SP.SSR_SIMSCALE_ROOT)
    ap.add_argument("--rounds", type=int, nargs="+", default=list(ROUNDS))
    ap.add_argument("--max-per-source", nargs="*", default=[], help="mini mode: N, or SOURCE=N (navtrain_train, "
                    "navtrain_val, simscale)")
    ap.add_argument("--max-drop-frac", type=float, default=0.01, help="full mode: largest share of a round the filter "
                    "may drop")
    ap.add_argument("--out", default=C.SPLITS["trainval_simscale"]["tokens"])
    ap.add_argument("--sim-out", default=C.SPLITS["simscale"]["tokens"], help="the SimScale rows alone")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    out, sim_out = Path(a.out), Path(a.sim_out)
    meta_p = out.with_suffix(".meta.json")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    if sorted(set(a.rounds)) != sorted(a.rounds) or not set(a.rounds) <= set(ROUNDS):
        raise SystemExit(f"--rounds: distinct values of {ROUNDS}")
    caps = parse_caps(a.max_per_source)
    t, meta = build(Path(a.train), Path(a.val), Path(a.sim_root), sorted(a.rounds), caps, a.workers, a.max_drop_frac,
                    C.SPLITS)
    sim = t[t.source == SOURCE_SIM].reset_index(drop=True)
    sim["row"] = np.arange(len(sim), dtype=np.int64)
    meta.update(out=str(out), sim_out=str(sim_out), sim_token_sha16=token_sha16(sim.token))
    write_atomic(sim, sim_out)
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_p.with_name(f".{meta_p.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, meta_p)
    write_atomic(t, out)                                    # last: the table marks the output complete
    print(f"trainval_simscale ({meta['mode']}): {len(t)} rows {meta['n_by_source']}, token_sha16 {meta['token_sha16']} "
          f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
