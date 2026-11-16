#!/usr/bin/env python
"""SSR side of the camera-teacher caches (BEVFormer DET, Mask2Map MAP): token lists, SimScale infos, manifests, checks.

The GPU passes run in the teacher env (setup_teacher_env.sh) from the get_teachers.sh checkouts: BEVFormer
tools/navsim_cache_teacher.py and Mask2Map tools/navsim/cache_teacher_m2m.py.  This file runs with the SSR python, CPU
only; tools/release/setup/65_cam_teacher_bev.sh calls it between the passes.

  tokens        --table T --work W                 token lists of a token table (navsim/common/simscale.TABLE_COLUMNS)
  pass-tokens   --list L --out F [--det D] [--map D]   the tokens of L that no cache holds yet (a pass's frozen list)
  bev-infos     --table T --round R --converter C --out F   BEVFormer frame pool of the SimScale tokens of round R
  m2m-infos     --table T --round R --out F        Mask2Map infos of the SimScale tokens of round R
  empty-gt      --tokens F --out G                 packed map GT without geometry (the test pipeline reads none)
  det-manifest  --root D --template M              DET manifest.json over D's passes
  map-merge     --root D --template M              MAP index.json + meta.json over D/passes/*
  verify        --table T [--det D] [--map D]      every table token in the caches, read through the SSR readers

Layouts (the readers: refiner/data.py TeacherCache, refiner/resmap_cache.py ResmapCache, distill/teacher_store.py):
  DET  <root>/samples/<tok[:2]>/<tok>.npz (bev_feature f16 [256,100,100], (C, x_forward, y_left ascending), boxes),
       <root>/manifest.json, <root>/passes/<pass>/{provenance,done_r*}.json.  Real and SimScale tokens share the tree.
  MAP  <root>/passes/<pass>/ = one cache_teacher_m2m.py run ({meta,index,index_r*,provenance}.json, bev/, seg/);
       <root>/bev/<pass> -> ../passes/<pass>/bev (and seg) relative links, <root>/index.json {token: ["<pass>/<shard>",
       row]} over every pass and <root>/meta.json = the bundle meta with num_frames = len(index) and the passes.  Shard
       names are unique per pass by construction; no pass may be named 'navtest' (TeacherFeatureStore maps that
       prefix to root/navtest/).  A reference cache copied from another server is a pass too (60_teacher_bev.sh).
Token keys are the full table tokens ('<16 hex>-<rrr>' for SimScale), never truncated.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]      # repo root of this worktree
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from navsim.common import simscale as SS  # noqa: E402

DET_BEV = (256, 100, 100)
MAP_BEV = (256, 100, 100)
MAP_SEG = (4, 200, 200)
M2M_CAMS = ("CAM_L0", "CAM_F0", "CAM_R0")           # the cameras the Mask2Map config feeds, in its order
M2M_CLASSES = ("road", "walkway", "centerline", "crosswalk")
M2M_ROI = ((64.0, 64.0), (32.0, 0.0))                # roi_size, roi_center of the front64 config and map GT
RESERVED_PASSES = ("navtest",)


# ----------------------------------------------------------------------------------------------- small helpers
def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")


def write_text(path: Path, text: str) -> bool:
    """Write atomically; a file that already holds `text` is left alone (its mtime too).  -> True if written."""
    path = Path(path)
    if path.is_file() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return True


def write_json(path: Path, obj, indent: Optional[int] = 2) -> bool:
    return write_text(path, json.dumps(obj, indent=indent) + "\n")


def read_tokens(path) -> List[str]:
    return [t.strip() for t in Path(path).read_text().splitlines() if t.strip()]


def sha16(tokens: Sequence[str]) -> str:
    from navsim.agents.para_ssr.ck.tokens import token_sha16
    return token_sha16(tokens)


def load_table(path) -> "pd.DataFrame":
    import pandas as pd
    df = pd.read_parquet(path)
    lack = [c for c in SS.TABLE_COLUMNS if c not in df.columns]
    if lack:
        raise SystemExit(f"{path}: no columns {lack} (navsim/common/simscale.TABLE_COLUMNS)")
    df["token"] = df["token"].astype(str)
    if df.token.duplicated().any():
        raise SystemExit(f"{path}: duplicate tokens, e.g. {df.token[df.token.duplicated()].iloc[0]}")
    return df


def real_rows(df):
    return df[df.source.isin(SS.SOURCE_REAL)]


def sim_rows(df, r: Optional[int] = None):
    s = df[df.source == SS.SOURCE_SIM]
    return s if r is None else s[s["round"].astype(int) == int(r)]


# ----------------------------------------------------------------------------------------------- tokens
def table_lists(df) -> Dict[str, List[str]]:
    """{'real': tokens, 'sim_r<r>': tokens} in table order; only the rounds the table holds."""
    out = {"real": real_rows(df).token.tolist()}
    for r in sorted(set(sim_rows(df)["round"].astype(int))):
        out[f"sim_r{r}"] = sim_rows(df, r).token.tolist()
    return out


def cmd_tokens(a) -> None:
    df = load_table(a.table)
    work = Path(a.work)
    lists = table_lists(df)
    summary = {"table": str(Path(a.table).resolve()), "n_rows": len(df), "lists": {}}
    for name, toks in lists.items():
        write_text(work / "tokens" / f"{name}.txt", "".join(t + "\n" for t in toks))
        summary["lists"][name] = {"n": len(toks), "token_sha16": sha16(toks)}
    write_json(work / "tokens" / "lists.json", summary)
    print(" ".join(f"{k}={v['n']}" for k, v in summary["lists"].items()))


# ----------------------------------------------------------------------------------------------- BEVFormer infos
BEV_CAMS = ("CAM_F0", "CAM_L0", "CAM_R0")
BEV_HISTORY = 4                                      # navsim_converter --history-frames of the reference pool


def pool_key(token: str, log: str) -> str:
    """Frame-pool key of a SimScale history frame (the current frame keeps the table token, the cache key)."""
    return f"{token}#{log}"


def _bev_scene(job):
    conv_path, root, pkl, token, r = job
    conv = _converter(conv_path)
    frames = SS.read_scene(root, pkl, token)
    log = Path(pkl).stem
    sensor = str(SS.sensor_root(root, r))
    in_split = [k == SS.CURRENT for k in range(len(frames))]
    infos, kept = [], []
    for idx in conv.select_frame_indices(in_split, BEV_HISTORY):
        info, _ = conv.convert_frame(frames[idx], log, idx, "trainval", sensor, 1.0, sensor, "lidar", ".pcd",
                                     list(BEV_CAMS), True, in_split[idx], [])
        if info is None:
            return token, None, f"{pkl} frame {idx}: a camera image is missing"
        infos.append(info)
        kept.append(idx)
    stats = conv.link_prev_next(infos, kept, frames)
    key = {i["token"]: (i["token"] if i["in_split_tokens"] else pool_key(i["token"], log)) for i in infos}
    for i in infos:
        i["token"] = key[i["token"]]
        i["prev"] = key.get(i["prev"]) if i["prev"] is not None else None
        i["next"] = key.get(i["next"]) if i["next"] is not None else None
    return token, infos, stats


_CONV = {}


def _converter(path: str):
    if path not in _CONV:
        _CONV[path] = _load_by_path("bevformer_navsim_converter", Path(path))
    return _CONV[path]


def _load_by_path(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bev_sim_infos(converter: Path, root, rows, workers: int = 8) -> dict:
    """BEVFormer frame pool of SimScale table rows (token, pkl, round), in navsim_converter.py's output format.

    Built per scene pickle with the converter's own functions (select_frame_indices --history-frames 4, convert_frame
    with --check-files on CAM_F0/L0/R0 and --sensor-prefix = the round's sensor root, link_prev_next), so every info is
    the one `navsim_converter.py --history-frames 4 --cameras CAM_F0 CAM_L0 CAM_R0 --check-files` writes, and the
    4 frames of a scene are one prev chain.  The converter CLI itself cannot take a whole round: it flags a frame
    in-split by token, and a SimScale frame token is not unique across the scenes of a round (clips of one source log
    overlap in time and render the same time stamp from different ego poses under the same '<hex>-<rrr>' token), so
    another scene's frame would be taken for a target and history frames would alias in the pool.  Here the in-split
    frame is frame 3 of its own scene, and a history frame is keyed '<token>#<scene pickle stem>' (pool_key)."""
    jobs = [(str(converter), str(root), str(p), str(t), int(r)) for t, p, r in rows]
    with Pool(max(1, workers)) as pool:
        res = pool.map(_bev_scene, jobs, chunksize=32)
    bad = [s for _, i, s in res if i is None]
    if bad:
        raise SystemExit(f"{len(bad)} scenes without their images, e.g. {bad[0]}")
    infos = [x for _, i, _ in res for x in i]
    infos.sort(key=lambda i: (i["log_name"], i["timestamp"]))
    keys = [i["token"] for i in infos]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate frame keys in the SimScale pool")
    links = {}
    for _, _, s in res:
        for k, v in s.items():
            links[k] = links.get(k, 0) + v
    conv = _converter(str(converter))
    meta = dict(version="trainval", navsim_converter=conv.CONVERTER_VERSION, image_width=conv.ORIGINAL_IMAGE_WIDTH,
                history_frames=BEV_HISTORY, keyframes_only=False, split="trainval", cameras=list(BEV_CAMS),
                max_sweeps=0, scene_filter=None, num_logs=len(res), num_frames=len(infos),
                num_in_split=sum(bool(i["in_split_tokens"]) for i in infos), links=links,
                simscale="per scene pickle (teacher_caches.bev_sim_infos): in-split = frame 3 of the scene, history "
                         "frames keyed <token>#<scene>")
    return dict(infos=infos, metadata=meta)


def cmd_bev_infos(a) -> None:
    from navsim.common import ssr_paths as SP
    df = load_table(a.table)
    rows = sim_rows(df, a.round)
    toks = rows.token.tolist()
    out = Path(a.out)
    done = out.with_suffix(".tokens.json")
    if out.is_file() and done.is_file() and json.loads(done.read_text()).get("token_sha16") == sha16(toks):
        print(f"{out}: kept ({len(toks)} scenes)")
        return
    root = a.simscale_root or SP.SSR_SIMSCALE_ROOT
    t0 = time.time()
    d = bev_sim_infos(Path(a.converter), root, zip(toks, rows.pkl.astype(str), rows["round"].astype(int)), a.workers)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(d, fh, protocol=4)
    tmp.replace(out)
    m = d["metadata"]
    write_json(done, {"n": len(toks), "token_sha16": sha16(toks), "round": int(a.round), "simscale_root": str(root),
                      "frames": m["num_frames"], "links": m["links"], "created": now()})
    print(f"{out}: {m['num_frames']} frames of {m['num_logs']} scenes, {m['num_in_split']} in-split, links "
          f"{m['links']} ({time.time() - t0:.0f}s)")


def det_has(root: Path, token: str) -> bool:
    p = Path(root) / "samples" / token[:2] / f"{token}.npz"
    return p.is_file() and p.stat().st_size >= DET_BEV[0] * DET_BEV[1] * DET_BEV[2] * 2


def map_index(root: Path) -> Dict[str, list]:
    """Union of the indexes of every pass of a MAP root (what map-merge writes), without writing anything."""
    idx: Dict[str, list] = {}
    for name, part in pass_indexes(Path(root)).items():
        for t, (sh, row) in part.items():
            idx.setdefault(t, [f"{name}/{sh}", row])
    return idx


def cmd_pass_tokens(a) -> None:
    """The tokens of --list that no cache holds yet, frozen at the first call: a pass keeps the token list it started
    with (cache_teacher_m2m.py shards a pass round-robin over its dataset, so a list that changes under a resumed pass
    would hand a token to two ranks).  An existing --out is kept and reported."""
    out = Path(a.out)
    if out.is_file():
        print(f"{out}: kept ({len(read_tokens(out))} tokens)")
        return
    toks = read_tokens(a.list)
    have = set()
    if a.map:
        have |= set(map_index(Path(a.map)))
    if a.det:
        with ThreadPoolExecutor(16) as ex:
            have |= {t for t, ok in zip(toks, ex.map(lambda t: det_has(Path(a.det), t), toks)) if ok}
    left = [t for t in toks if t not in have]
    write_text(out, "".join(t + "\n" for t in left))
    print(f"{out}: {len(left)} of {len(toks)} tokens to cache")


# ----------------------------------------------------------------------------------------------- Mask2Map infos
def m2m_info(frame: dict, sensor_root: Path, scene_name: str, cams: Sequence[str] = M2M_CAMS) -> dict:
    """One frame of a scene dict -> one Mask2Map NAVSIM info, the fields and conventions of Mask2Map's
    tools/navsim/create_navsim_infos.py (scan_log).  ego2global from translation + quaternion (wxyz), never from the
    4x4 matrices, whose bottom row is zero in SimScale; cam2ego = sensor2lidar (lidar2ego is the identity) and
    extrinsics = ego2cam = inv(cam2ego); intrinsics calibrated at 1920x1080; img_fpath absolute."""
    cam_dict = {}
    for c in cams:
        cm = frame["cams"][c]
        cam2ego = np.eye(4)
        cam2ego[:3, :3] = np.asarray(cm["sensor2lidar_rotation"], dtype=np.float64)
        cam2ego[:3, 3] = np.asarray(cm["sensor2lidar_translation"], dtype=np.float64)
        cam_dict[c] = dict(
            img_fpath=str(Path(sensor_root) / cm["data_path"]),
            intrinsics=np.asarray(cm["cam_intrinsic"], dtype=np.float64),
            extrinsics=np.linalg.inv(cam2ego),
            distortion=np.asarray(cm["distortion"], dtype=np.float64),
        )
    return dict(
        token=str(frame["token"]),
        cams=cam_dict,
        lidar2ego_translation=list(map(float, frame["lidar2ego_translation"])),
        lidar2ego_rotation=list(map(float, frame["lidar2ego_rotation"])),
        e2g_translation=list(map(float, frame["ego2global_translation"])),
        e2g_rotation=list(map(float, frame["ego2global_rotation"])),
        timestamp=int(frame["timestamp"]),
        location=str(frame["map_location"]),
        scene_name=scene_name,
        log_name=str(frame["log_name"]),
        scene_token=str(frame["scene_token"]),
        frame_idx=int(frame["frame_idx"]),
        in_split_tokens=True,
        sample_idx=0,
        prev=-1,
        next=-1,
        lidar_path=None,
    )


def _sim_info(job):
    root, pkl, token, r = job
    frames = SS.read_scene(root, pkl, token)
    info = m2m_info(frames[SS.CURRENT], SS.sensor_root(root, r), f"{Path(pkl).stem}#0")
    lost = [c["img_fpath"] for c in info["cams"].values() if not os.path.isfile(c["img_fpath"])]
    return info, lost


def sim_infos(root, rows, workers: int = 8) -> List[dict]:
    """Mask2Map infos of SimScale table rows (token, pkl, round): the current frame (3) only, the model is single-frame
    (queue_length 1, video_test_mode off), one scene per pickle; sample_idx is the position in the list."""
    jobs = [(str(root), str(p), str(t), int(r)) for t, p, r in rows]
    with Pool(max(1, workers)) as pool:
        res = pool.map(_sim_info, jobs, chunksize=64)
    lost = [p for _, ls in res for p in ls]
    if lost:
        raise SystemExit(f"{len(lost)} camera images missing, e.g. {lost[0]}")
    infos = [i for i, _ in res]
    for k, s in enumerate(infos):
        s["sample_idx"] = k
    return infos


def cmd_m2m_infos(a) -> None:
    from navsim.common import ssr_paths as SP
    df = load_table(a.table)
    rows = sim_rows(df, a.round)
    toks = rows.token.tolist()
    out = Path(a.out)
    done = out.with_suffix(".tokens.json")
    if out.is_file() and done.is_file() and json.loads(done.read_text()).get("token_sha16") == sha16(toks):
        print(f"{out}: kept ({len(toks)} frames)")
        return
    root = a.simscale_root or SP.SSR_SIMSCALE_ROOT
    t0 = time.time()
    infos = sim_infos(root, zip(toks, rows.pkl.astype(str), rows["round"].astype(int)), a.workers)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(infos, fh, protocol=4)
    tmp.replace(out)
    write_json(done, {"n": len(infos), "token_sha16": sha16(toks), "round": int(a.round), "simscale_root": str(root),
                      "cams": list(M2M_CAMS), "frame": SS.CURRENT, "created": now()})
    print(f"{out}: {len(infos)} frames ({time.time() - t0:.0f}s)")


def empty_map_gt(tokens: Sequence[str], out: Path) -> None:
    """A packed map GT (Mask2Map tools/navsim/pack_map_gt.py layout and dtypes) that lists `tokens` with no geometry.
    NavsimOfflineMapDataset requires a map GT row for every frame, but only its train pipeline and evaluate() read it:
    the test pipeline the caches run (load images, normalise, pad, collect) never does, so the cached tensors do not
    depend on it (shown on real tokens in tests/test_simscale_teachers.py and the fidelity report)."""
    toks = sorted(set(str(t) for t in tokens))
    if any(len(t) > 32 for t in toks):
        raise SystemExit("map GT tokens are stored as S32: a token is longer than 32 characters")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, coords=np.zeros((0, 2), np.float32), ring_off=np.zeros(1, np.int64), ring_cls=np.zeros(0, np.int8),
             ring_poly=np.zeros(0, np.int32), tok_off=np.zeros(len(toks) + 1, np.int64),
             tokens=np.asarray(toks, dtype="S32"), classes=np.asarray(M2M_CLASSES, dtype="S16"),
             area_classes=np.asarray(sorted(("road", "walkway", "crosswalk")), dtype="S16"),
             roi_size=np.asarray(M2M_ROI[0], np.float64), roi_center=np.asarray(M2M_ROI[1], np.float64))
    tmp.replace(out)


def cmd_empty_gt(a) -> None:
    toks = [t for f in a.tokens for t in read_tokens(f)]
    empty_map_gt(toks, Path(a.out))
    print(f"{a.out}: {len(set(toks))} tokens, no geometry")


# ----------------------------------------------------------------------------------------------- DET manifest
def det_count(root: Path, workers: int = 16) -> int:
    """Complete-looking npz files under root/samples (size >= the bev tensor)."""
    sub = [p for p in (Path(root) / "samples").iterdir() if p.is_dir()] if (Path(root) / "samples").is_dir() else []
    min_b = DET_BEV[0] * DET_BEV[1] * DET_BEV[2] * 2

    def one(d):
        n = 0
        with os.scandir(d) as it:
            for e in it:
                if e.name.endswith(".npz") and not e.name.startswith(".") and e.stat().st_size >= min_b:
                    n += 1
        return n

    with ThreadPoolExecutor(workers) as ex:
        return sum(ex.map(one, sub))


def det_passes(root: Path) -> Dict[str, dict]:
    out = {}
    for d in sorted((Path(root) / "passes").glob("*/provenance.json")):
        prov = json.loads(d.read_text())
        done = [json.loads(p.read_text()) for p in sorted(d.parent.glob("done_r*.json"))]
        out[d.parent.name] = dict(prov, done=done)
    return out


def cmd_det_manifest(a) -> None:
    """manifest.json of a DET root: the bundle manifest (reader-checked fields: checkpoint_sha256_head, layout, shapes,
    point_cloud_range, to_student_transform, ...) with this cache's counts and passes; the reference run's own
    provenance stays under 'template'."""
    root = Path(a.root)
    tpl = json.loads(Path(a.template).read_text())
    passes = det_passes(root)
    bad = {k: v.get("checkpoint_sha256_head") for k, v in passes.items()
           if v.get("checkpoint_sha256_head") != tpl["checkpoint_sha256_head"]}
    if bad:
        raise SystemExit(f"passes with another checkpoint than the template's {tpl['checkpoint_sha256_head']}: {bad}")
    man = dict(tpl)
    man["template"] = {k: tpl.get(k) for k in ("created", "git_commit", "config", "checkpoint", "split", "ann_file",
                                               "num_samples_expected", "num_samples_written", "world_size")}
    first = next(iter(passes.values()), {})
    man.update(created=now(), git_commit=(first.get("git_commit") or "")[:7] or tpl.get("git_commit"),
               config=first.get("config", tpl.get("config")), checkpoint=first.get("checkpoint", tpl.get("checkpoint")),
               ann_file={k: v.get("ann_file") for k, v in passes.items()},
               num_samples_expected=sum(int(v.get("n_targets", 0)) for v in passes.values()),
               num_samples_written=det_count(root), world_size=max([v.get("world_size", 1) for v in passes.values()] or
                                                                   [tpl.get("world_size")]),
               passes={k: {f: v.get(f) for f in ("ann_file", "token_list", "token_list_sha256", "n_pool_frames",
                                                 "n_targets", "n_frames_planned", "world_size", "git_commit",
                                                 "git_diff_sha256", "torch", "gpu", "cudnn_allow_tf32",
                                                 "matmul_allow_tf32", "created", "done")}
                       for k, v in passes.items()})
    write_json(root / "manifest.json", man)
    print(f"{root}/manifest.json: {man['num_samples_written']} samples, passes {sorted(passes)}")


# ----------------------------------------------------------------------------------------------- MAP merge
def pass_indexes(root: Path) -> Dict[str, Dict[str, list]]:
    """{pass: {token: [shard, row]}} of root/passes/*, each the union of its index_r<rank>.json over the world size of
    its provenance (what cache_teacher_m2m.py's final step writes as its index.json); a pass without them (a reference
    cache copied with only its index.json) uses its index.json."""
    out = {}
    for d in sorted(p for p in (Path(root) / "passes").glob("*") if p.is_dir()):
        if d.name in RESERVED_PASSES:
            raise SystemExit(f"{d}: a pass must not be named {d.name!r} (TeacherFeatureStore maps it to root/navtest)")
        prov = json.loads((d / "provenance.json").read_text()) if (d / "provenance.json").is_file() else {}
        parts = [d / f"index_r{r}.json" for r in range(int(prov.get("world_size", 0)))]
        if parts and all(p.is_file() for p in parts):
            idx: Dict[str, list] = {}
            for p in parts:
                part = json.loads(p.read_text())
                dup = set(part) & set(idx)
                if dup:
                    raise SystemExit(f"{d}: {len(dup)} tokens in two ranks' indexes, e.g. {sorted(dup)[:3]}")
                idx.update(part)
        elif (d / "index.json").is_file():
            idx = json.loads((d / "index.json").read_text())
        else:
            continue
        out[d.name] = idx
    return out


def link(src_rel: str, dst: Path) -> None:
    """dst -> src_rel relative symlink; an existing equal link is kept, anything else there is refused."""
    if dst.is_symlink():
        if os.readlink(dst) == src_rel:
            return
        raise SystemExit(f"{dst} links to {os.readlink(dst)}, not {src_rel}")
    if dst.exists():
        raise SystemExit(f"{dst} exists and is not a link: move it away")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src_rel, dst)


def cmd_map_merge(a) -> None:
    """index.json + meta.json of a MAP root over all its passes.  Every pass must be the template's checkpoint and
    tensors; a token in two passes is refused.  meta.json = the bundle meta (the fields ResmapCache checks) with
    num_frames = len(index) and the passes' provenance; index.json is written before meta.json."""
    root = Path(a.root)
    tpl = json.loads(Path(a.template).read_text())
    idxs = pass_indexes(root)
    if not idxs:
        raise SystemExit(f"{root}/passes holds no finished pass")
    merged: Dict[str, list] = {}
    passes = {}
    for name, idx in idxs.items():
        d = root / "passes" / name
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
        prov = json.loads((d / "provenance.json").read_text()) if (d / "provenance.json").is_file() else {}
        sha = meta.get("checkpoint_sha256") or prov.get("checkpoint_sha256")
        if sha != tpl["checkpoint_sha256"]:
            raise SystemExit(f"{d}: checkpoint_sha256 {sha} != template {tpl['checkpoint_sha256']}")
        for f in ("tensors", "classes", "pc_range", "roi_size_m"):
            if meta and meta.get(f) != tpl.get(f):
                raise SystemExit(f"{d}: meta {f} {meta.get(f)} != template {tpl.get(f)}")
        dup = set(idx) & set(merged)
        if dup:
            raise SystemExit(f"{len(dup)} tokens in pass {name} and an earlier pass, e.g. {sorted(dup)[:3]}")
        for field in ("bev", "seg"):
            if (d / field).is_dir():
                link(f"../passes/{name}/{field}", root / field / name)
        merged.update({t: [f"{name}/{sh}", int(row)] for t, (sh, row) in idx.items()})
        passes[name] = dict(num_frames=len(idx), **{k: prov.get(k) for k in (
            "cfg", "world_size", "split", "ann_file", "map_gt_file", "only_split_tokens", "token_list", "bev_tap")})
    meta = dict(tpl, num_frames=len(merged), passes=passes, merged=now(),
                template={k: tpl.get(k) for k in ("teacher_config", "teacher_checkpoint", "ann_file", "map_gt_file",
                                                  "num_frames")})
    write_json(root / "index.json", merged, indent=None)
    write_json(root / "meta.json", meta)
    print(f"{root}: index.json {len(merged)} frames over passes " + ", ".join(f"{k}={v['num_frames']}"
                                                                            for k, v in passes.items()))


# ----------------------------------------------------------------------------------------------- verify
def _reader_env() -> None:
    """The 64 m readers (CK_BEV64=1, as compose sets for BEV_RANGE=64); set before the refiner modules are imported."""
    os.environ["CK_BEV64"] = "1"


def verify_det(root: Path, tokens: Sequence[str], sha_head: str, workers: int = 8) -> List[str]:
    """Every token through TeacherCache.load_bev and TeacherFeatureStore.load_one_token; -> problems."""
    _reader_env()
    from types import SimpleNamespace
    from navsim.agents.para_ssr.refiner.data import TeacherCache
    from navsim.agents.para_ssr.distill.teacher_store import TeacherFeatureStore
    tc = TeacherCache(root, expect_sha_head=sha_head)
    st = TeacherFeatureStore(str(root.parent), root.parent.name, cache_subdirs=(str(root),), expect_sha=sha_head)
    st.validate_manifest(SimpleNamespace(bev_h=100, bev_w=100, embed_dims=256, pc_range=[-32, 0, -3, 32, 64, 5]))

    def one(t):
        try:
            x = tc.load_bev(t)
            y = st.load_one_token(t)
            if y is None or y.shape != DET_BEV or not np.array_equal(x, y):
                return f"{t}: TeacherFeatureStore {None if y is None else y.shape} differs from TeacherCache"
            if not np.isfinite(x.astype(np.float32)).all():
                return f"{t}: non-finite bev_feature"
        except Exception as e:  # noqa: BLE001  every reader failure is a finding
            return f"{t}: {type(e).__name__}: {e}"
        return None

    with ThreadPoolExecutor(workers) as ex:
        return [p for p in ex.map(one, tokens) if p]


def verify_map(root: Path, tokens: Sequence[str], sha256: str, workers: int = 8) -> List[str]:
    """Every token through ResmapCache.load_bev / load_seg and TeacherFeatureStore.load_one_token; -> problems."""
    _reader_env()
    from types import SimpleNamespace
    from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache
    from navsim.agents.para_ssr.distill.teacher_store import TeacherFeatureStore
    rc = ResmapCache(root, expect_sha256=sha256, expect_split="none")
    st = TeacherFeatureStore(str(root.parent), root.name, cache_dir=str(root), expect_sha=sha256)
    st.validate_manifest(SimpleNamespace(bev_h=100, bev_w=100, embed_dims=256, pc_range=[-32, 0, -3, 32, 64, 5]))
    if not st.is_sharded() or st.flip_w:
        return [f"{root}: TeacherFeatureStore sees no sharded cache"]

    def one(t):
        try:
            x = rc.load_bev(t)
            s = rc.load_seg(t)
            y = st.load_one_token(t)
            if y is None or y.shape != MAP_BEV or not np.array_equal(x, y):
                return f"{t}: TeacherFeatureStore {None if y is None else y.shape} differs from ResmapCache"
            if s.shape != MAP_SEG or not np.isfinite(x.astype(np.float32)).all():
                return f"{t}: seg {s.shape} or non-finite bev"
        except Exception as e:  # noqa: BLE001
            return f"{t}: {type(e).__name__}: {e}"
        return None

    with ThreadPoolExecutor(workers) as ex:
        return [p for p in ex.map(one, tokens) if p]


def coverage(df, det: Optional[Path], map_root: Optional[Path], workers: int = 16) -> Dict[str, dict]:
    """Per cache and source: table tokens without a cache entry (DET: no complete-sized npz; MAP: not in index.json)."""
    out = {}
    toks = df.token.tolist()
    src = df.source.tolist()
    if det is not None:
        with ThreadPoolExecutor(workers) as ex:
            ok = list(ex.map(lambda t: det_has(det, t), toks, chunksize=512))
        out["det"] = _missing_by_source(toks, src, ok)
    if map_root is not None:
        idx = json.loads((Path(map_root) / "index.json").read_text()) if (Path(map_root) / "index.json").is_file() else {}
        out["map"] = _missing_by_source(toks, src, [t in idx for t in toks])
    return out


def _missing_by_source(toks, src, ok) -> dict:
    res: Dict[str, dict] = {}
    for t, s, k in zip(toks, src, ok):
        r = res.setdefault(s, {"n": 0, "missing": 0, "example": None})
        r["n"] += 1
        if not k:
            r["missing"] += 1
            r["example"] = r["example"] or t
    return res


def cmd_verify(a) -> None:
    df = load_table(a.table)
    toks = df.token.tolist()
    report = {"table": str(a.table), "n_table": len(df)}
    bad = []
    if a.coverage:
        report["coverage"] = coverage(df, Path(a.det) if a.det else None, Path(a.map) if a.map else None, a.workers)
        for cache, by in report["coverage"].items():
            bad += [f"{cache}: {v['missing']} of {v['n']} {s} tokens missing, e.g. {v['example']}"
                    for s, v in by.items() if v["missing"]]
    if a.sample and a.sample < len(toks):
        rng = np.random.default_rng(0)
        toks = [toks[i] for i in sorted(rng.choice(len(toks), a.sample, replace=False))]
    report["n_read"] = len(toks)
    if a.det:
        t0 = time.time()
        p = verify_det(Path(a.det), toks, a.det_sha_head, a.workers)
        report["det"] = {"root": a.det, "problems": len(p), "sec": round(time.time() - t0, 1)}
        bad += p
    if a.map:
        t0 = time.time()
        p = verify_map(Path(a.map), toks, a.map_sha256, a.workers)
        idx = json.loads((Path(a.map) / "index.json").read_text())
        report["map"] = {"root": a.map, "problems": len(p), "index": len(idx), "sec": round(time.time() - t0, 1)}
        bad += p
    by_src = df.groupby("source").size().to_dict()
    report["sources"] = {k: int(v) for k, v in by_src.items()}
    report["problems"] = bad[:20]
    print(json.dumps(report, indent=2))
    if a.out:
        write_json(Path(a.out), report)
    if bad:
        raise SystemExit(f"{len(bad)} problems, e.g. {bad[0]}")


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("tokens")
    s.add_argument("--table", required=True)
    s.add_argument("--work", required=True)
    s = sub.add_parser("pass-tokens")
    s.add_argument("--list", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--det", default=None)
    s.add_argument("--map", default=None)
    s = sub.add_parser("bev-infos")
    s.add_argument("--table", required=True)
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--converter", required=True, help="BEVFormer tools/data_converter/navsim_converter.py")
    s.add_argument("--out", required=True)
    s.add_argument("--simscale-root", default=None)
    s.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("m2m-infos")
    s.add_argument("--table", required=True)
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--simscale-root", default=None)
    s.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("empty-gt")
    s.add_argument("--tokens", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("det-manifest")
    s.add_argument("--root", required=True)
    s.add_argument("--template", required=True)
    s = sub.add_parser("map-merge")
    s.add_argument("--root", required=True)
    s.add_argument("--template", required=True)
    s = sub.add_parser("verify")
    s.add_argument("--table", required=True)
    s.add_argument("--det", default=None)
    s.add_argument("--map", default=None)
    s.add_argument("--det-sha-head", default="4ca4f1aa55011beb")
    s.add_argument("--map-sha256", default="579665a7309beb17621dcbad511031776015f0a58c65a2beee1ee58ed45771e4")
    s.add_argument("--sample", type=int, default=0, help="read a seeded sample of N tokens (0 = all)")
    s.add_argument("--coverage", action="store_true", help="also check that every table token has an entry")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--out", default=None)
    a = p.parse_args(argv)
    {"tokens": cmd_tokens, "pass-tokens": cmd_pass_tokens, "bev-infos": cmd_bev_infos, "m2m-infos": cmd_m2m_infos,
     "empty-gt": cmd_empty_gt, "det-manifest": cmd_det_manifest, "map-merge": cmd_map_merge,
     "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
