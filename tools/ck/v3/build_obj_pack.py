#!/usr/bin/env python
"""Build the refiner-v3 object pack of one split (SPEC 3.4; owner B1).  CPU only, read-only on every input.

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/yongjae/SSR-v3 \
    nice -n 10 /venv/ssr/bin/python tools/ck/v3/build_obj_pack.py --split navtrain_val --workers 16
  ... --split navtrain_train --workers 16
  ... --split navtest --workers 16            (targets from D3/gt_objects/navtest; --no-targets without them)

Output P = <out-root>/<source>/<split>/ (default out-root D3/objpack): the files of tools/ck/data/obj_pack.py
(tokens.parquet, obj.npy, obj_mask.npy, frame_valid.npy, frame_dt.npy, frame_src.npy, n_raw.npy, obj_ok.npy,
frame_why.npy, frame_token.npy, fc_xyh.npy, fc_state.npy, fc_track.npy) + build.json.  Rows follow
CK_DATA/packed/<split>/tokens.parquet.

Resumable: work is split into chunks of --chunk tokens (default 64); each chunk is written atomically to
P/_parts/c<i>.npz (tmp + rename) and skipped on a re-run if its token list and part key match; otherwise it is
rebuilt.  Part key = layout (version, K_OBJ, D_OBJ, thresholds, targets) + an INPUT FINGERPRINT (tooling fix F4,
2026-10-11): sha256 of the build provenance without git head (BEVFusion cache manifest shas, frames parquet sha,
code shas of objects / obj_pack / build_obj_pack / config / future_teacher / gt_future, GT dir + index sha),
MATCH_GATE_M, FC_T, N_FRAMES and the sha256 of the GT per-file manifest (D3/gt_objects/_manifests/<split>.json).
Before F4 a build interrupted and resumed after such an input changed mixed old and new chunks while build.json
recorded only the new provenance.  The assembly re-checks every part's key.  build.json records the fingerprint
(inputs_sha256, inputs); packs built before F4 carry neither and load unchanged (ObjPack does not read them).
The final arrays are assembled into *.npy.tmp and renamed only after every chunk is in; build.json (status complete)
is written last; _parts is then removed (--keep-parts keeps it).  A complete pack is never overwritten.

Fails (exit 2, build.json status failed_*) if a CausalityError occurs (raised by the worker, never swallowed), if more
than 1 % of the valid frames of a FULL split are truncated at K_OBJ (96 since the review fix) (SPEC 3.4: K_OBJ is then raised in a spec
change; --limit builds record the fraction but skip the rule), or if a GT file is missing for a split with targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[3]
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))

from navsim.agents.para_ssr.ck.temporal import config as TC  # noqa: E402
from navsim.agents.para_ssr.ck.temporal import objects as O  # noqa: E402
from navsim.agents.para_ssr.refiner import future_teacher as FT  # noqa: E402
from navsim.agents.para_ssr.refiner import gt_future as GF  # noqa: E402
from tools.ck.data import obj_pack as OP  # noqa: E402

KST = timezone(timedelta(hours=9))
SUBSET = {"navtrain_train": "navtrain", "navtrain_val": "navtrain", "navtest": "navtest"}
# GT objects: ONLY the D3 copies / links (review fix: no silent fallback to an editable report folder).  Built by
# tools/ck/v3/setup_gt_objects.sh: navtrain_val = a copy of SSR/report/53.../A_full16/gt_objects/navtrain_val,
# navtrain_train = a symlink to /home/external-user/ssd/yongjae_refiner/objects/navtrain, navtest = built from
# test_navsim_logs with tools/refiner/build_future_objects.py; each with a per-file sha256 manifest (manifest.json).
GT_DIRS = {s: TC.D3 / "gt_objects" / s for s in ("navtrain_train", "navtrain_val", "navtest")}
TRUNC_MAX_FRAC = 0.01
NAMES = list(OP.PACK_FILES)
TNAMES = list(OP.TARGET_FILES)


def now_kst() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def gt_dir_for(split: str) -> Path:
    p = GT_DIRS[split]
    if not (p / "index.parquet").is_file():
        raise FileNotFoundError(f"no GT objects for {split}: {p}/index.parquet missing (run tools/ck/v3/setup_gt_objects.sh)")
    return p


# ----------------------------------------------------------------------------------------------- worker
_W = {}


def part_key(targets: bool, inputs_sha256: str = "") -> str:
    """identifies a part file: layout + the input fingerprint (F4); a resumed build reuses a part only if it matches."""
    return json.dumps(dict(version=OP.PACK_VERSION, objects=O.OBJECTS_VERSION, K_OBJ=TC.K_OBJ, D_OBJ=TC.D_OBJ,
                           SCORE_THR=list(TC.SCORE_THR), DT_RANGE=list(TC.DT_RANGE), targets=bool(targets),
                           inputs=str(inputs_sha256)), sort_keys=True)


def gt_manifest(gt_dir):
    """the per-file sha256 manifest of a GT dir (tools/ck/v3/setup_gt_objects.sh), or None."""
    if not gt_dir:
        return None
    p = Path(gt_dir).parent / "_manifests" / f"{Path(gt_dir).name}.json"
    return dict(path=str(p), sha256=sha256_file(p)) if p.is_file() else dict(path=str(p), sha256=None)


def input_fingerprint(prov, gt_dir):
    """F4: (sha256, dict) of every input a chunk's content depends on besides the layout of part_key: the provenance
    (minus git head / note: the code shas cover the code), MATCH_GATE_M, FC_T, N_FRAMES and the GT manifest sha."""
    d = dict(provenance={k: v for k, v in prov.items() if k not in ("git_head", "git_note")},
             match_gate_m=float(O.MATCH_GATE_M), FC_T=[float(x) for x in getattr(O, "FC_T", TC.FC_T)],
             N_FRAMES=int(TC.N_FRAMES),
             gt_manifest=gt_manifest(gt_dir))
    d = json.loads(json.dumps(d, sort_keys=True))
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(), d


def _init(subset, gt_dir, targets, pkey=None):
    _W["part_key"] = pkey if pkey is not None else part_key(targets)
    _W["src"] = O.BevfusionObjectSource(subset)
    _W["gt_dir"] = Path(gt_dir) if gt_dir else None
    _W["targets"] = bool(targets)


def one_token(tok: str):
    """-> (arrays dict for one row, stats dict).  CausalityError propagates."""
    src = _W["src"]
    rec = {k: np.zeros(shp, dt) for k, (dt, shp) in OP.PACK_FILES.items()}
    rec["frame_dt"][:] = np.nan
    st = dict(err=None, gt_missing=0, n_agent=0, n_matched=0, n_cross=0, n_static_matched=0, n_nonfinite=0,
              n_gt_future_first=0, n_single_matched=0)
    try:
        fr = src.frames(tok, dtype=np.float64)
    except O.CausalityError:
        raise
    except Exception as e:  # noqa: BLE001  (recorded: obj_ok False, listed in build.json)
        st["err"] = f"{type(e).__name__}: {e}"
        fr = None
    if fr is not None:
        obj32 = fr["obj"].astype(np.float32)
        rec["obj"] = obj32
        rec["obj_mask"] = fr["mask"]
        rec["frame_valid"] = fr["valid"]
        rec["frame_dt"] = fr["dt"].astype(np.float32)
        rec["frame_src"] = fr["src"].astype(np.int8)
        rec["n_raw"] = fr["n_raw"].astype(np.int16)
        rec["frame_why"] = np.array([O.WHY_CODE[w] for w in fr["why"]], np.int8)
        rec["frame_token"] = fr["frame_token"].astype("<U16")
        rec["obj_ok"] = np.array(bool(fr["valid"][0]))
        st["n_nonfinite"] = int(fr["n_nonfinite"].sum())
        st["dt_raw"] = [float(x) for x in fr["dt_raw"]]
        if not fr["valid"][0]:
            st["err"] = f"t0 frame invalid: {fr['why'][0]}"
    if _W["targets"]:
        xyh = np.zeros(OP.TARGET_FILES["fc_xyh"][1], np.float32)
        state = np.zeros(OP.TARGET_FILES["fc_state"][1], np.int8)
        track = np.full(OP.TARGET_FILES["fc_track"][1], -1, np.int32)
        if fr is not None and fr["valid"][0]:
            gp = _W["gt_dir"] / f"{tok}.npz"
            if gp.is_file():
                gt = GF.load_objects(gp)
                # match on the CLEAN pack rows (float32 values, as stored)
                xyh, state, track, ms = O.match_targets(rec["obj"][0], rec["obj_mask"][0], gt, return_stats=True)
                for k in ("n_agent", "n_matched", "n_cross", "n_static_matched", "n_gt_future_first", "n_single_matched"):
                    st[k] = ms[k]
            else:
                st["gt_missing"] = 1
        rec["fc_xyh"], rec["fc_state"], rec["fc_track"] = xyh, state, track
    return rec, st


def run_chunk(job):
    ci, rows, toks, part_path = job
    recs, sts = [], []
    for tok in toks:
        r, s = one_token(tok)
        recs.append(r)
        sts.append(s)
    names = NAMES + (TNAMES if _W["targets"] else [])
    arrs = {k: np.stack([r[k] for r in recs]) for k in names}
    tmp = str(part_path) + ".tmp.npz"
    np.savez(tmp, rows=np.asarray(rows, np.int64), tokens=np.asarray(toks, "<U16"),
             stats=np.array(json.dumps(sts)), part_key=np.array(_W["part_key"]), **arrs)
    os.replace(tmp, part_path)
    return ci, len(toks)


# ----------------------------------------------------------------------------------------------- stats
def summarise(P, n, targets, sts, frames):
    """counts for build.json from the assembled arrays + per-token stats."""
    fv, why, src, nraw = frames["frame_valid"], frames["frame_why"], frames["frame_src"], frames["n_raw"]
    out = dict(n_tokens=int(n), obj_ok_false=int((~frames["obj_ok"]).sum()))
    per = []
    for f in range(TC.N_FRAMES):
        v = fv[:, f]
        d = dict(frame=f, n_valid=int(v.sum()), frac_valid=round(float(v.mean()), 6),
                 invalid_by_reason={O.WHY[c] or "ok": int((why[:, f] == c).sum()) for c in range(len(O.WHY))},
                 src_current=int((src[:, f] == O.SRC_CURRENT).sum()), src_future=int((src[:, f] == O.SRC_FUTURE).sum()),
                 truncated=int((nraw[:, f] > TC.K_OBJ).sum()),
                 n_raw_mean=round(float(nraw[v, f].mean()), 3) if v.any() else None,
                 n_raw_p99=float(np.quantile(nraw[v, f], .99)) if v.any() else None,
                 n_raw_max=int(nraw[v, f].max()) if v.any() else None,
                 dt_median=float(np.nanmedian(frames["frame_dt"][v, f])) if v.any() else None,
                 dt_min=float(np.nanmin(frames["frame_dt"][v, f])) if v.any() else None,
                 dt_max=float(np.nanmax(frames["frame_dt"][v, f])) if v.any() else None,
                 n_slots_mean=round(float(frames["obj_mask"][v, f].sum(1).mean()), 3) if v.any() else None)
        per.append(d)
    out["frames"] = per
    nvalid = int(fv.sum())
    ntr = int(sum(p["truncated"] for p in per))
    out["truncated_frames"] = ntr
    out["truncated_frac_of_valid_frames"] = ntr / max(nvalid, 1)
    out["no_past_frame"] = round(float((~fv[:, 1] & ~fv[:, 2]).mean()), 6)
    out["no_t-1.0"] = round(float((~fv[:, 2]).mean()), 6)
    out["all_frames_valid"] = round(float(fv.all(1).mean()), 6)
    out["n_nonfinite_boxes"] = int(sum(s["n_nonfinite"] for s in sts))
    errs = [(i, s["err"]) for i, s in enumerate(sts) if s["err"]]
    out["n_errors"] = len(errs)
    out["errors_first20"] = errs[:20]
    if targets:
        na = int(sum(s["n_agent"] for s in sts))
        nm = int(sum(s["n_matched"] for s in sts))
        out["targets"] = dict(
            n_agent_slots=na, n_matched=nm, match_rate=nm / max(na, 1), n_cross_vehicle_vru=int(sum(s["n_cross"] for s in sts)),
            n_static_matched=int(sum(s["n_static_matched"] for s in sts)), n_gt_missing=int(sum(s["gt_missing"] for s in sts)),
            n_gt_future_first_excluded=int(sum(s.get("n_gt_future_first", 0) for s in sts)),
            n_single_matched=int(sum(s.get("n_single_matched", 0) for s in sts)),
            singleton_rule="GT tracks first annotated after t0 are never matched; matched singletons -> FC_SINGLE (4)")
    return out


def target_stats(fc_state, obj, mask):
    """fc_state histogram per step over valid agent t0 slots."""
    ag = O.row_agent(obj[:, 0]) & mask[:, 0]
    s = fc_state[ag]                                                    # [n, 10]
    return dict(n_agent_slots=int(ag.sum()),
                frac_obs_by_k=[round(float((s[:, k] == TC.FC_OBS).mean()), 4) for k in range(TC.FC_STEPS)],
                frac_absent_by_k=[round(float((s[:, k] == TC.FC_ABSENT).mean()), 4) for k in range(TC.FC_STEPS)],
                frac_beyond_by_k=[round(float((s[:, k] == TC.FC_BEYOND).mean()), 4) for k in range(TC.FC_STEPS)],
                frac_none_by_k=[round(float((s[:, k] == TC.FC_NONE).mean()), 4) for k in range(TC.FC_STEPS)],
                frac_single_by_k=[round(float((s[:, k] == TC.FC_SINGLE).mean()), 4) for k in range(TC.FC_STEPS)],
                steps_supervised=round(float((s == TC.FC_OBS).mean()), 4))


def provenance(subset, gt_dir, targets):
    src = O.BevfusionObjectSource(subset)
    files = {
        "objects.py": Path(O.__file__), "obj_pack.py": Path(OP.__file__), "build_obj_pack.py": Path(__file__).resolve(),
        "config.py": Path(TC.__file__), "future_teacher.py": Path(FT.__file__), "gt_future.py": Path(GF.__file__)}
    try:
        head = subprocess.run(["git", "-C", str(V3), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        head = None
    d = dict(
        caches={k: dict(root=str(c.root), manifest_sha256=sha256_file(c.manifest_path), sha_head=c.sha_head,
                        git_commit=c.manifest.get("git_commit"), point_cloud_range=list(c.pcr))
                for k, c in (("current", src.current), ("future", src.future))},
        frames_parquet=dict(path=str(src.frames_path), sha256=sha256_file(src.frames_path)),
        code={k: dict(path=str(p), sha256=sha256_file(p)) for k, p in files.items()},
        git_head=head, git_note="code is uncommitted on branch refiner-v3; see code sha256")
    if targets:
        d["gt"] = dict(dir=str(gt_dir), resolved=os.path.realpath(str(gt_dir)),
                       index_sha256=sha256_file(Path(gt_dir) / "index.parquet"))
    return d


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SUBSET))
    ap.add_argument("--source", default="bevfusion", choices=("bevfusion",))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-root", default=str(TC.PACK_ROOT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-targets", action="store_true")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--keep-parts", action="store_true")
    ap.add_argument("--max-trunc-frac", type=float, default=TRUNC_MAX_FRAC,
                    help="SPEC 3.4 rule (default 0.01); a different value is an explicit, recorded spec decision")
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= 16:
        raise SystemExit("--workers must be in 1..16 (CPU rule)")
    t_start, kst_start = time.time(), now_kst()
    targets = not a.no_targets
    subset = SUBSET[a.split]
    gt_dir = gt_dir_for(a.split) if targets else None
    P = OP.pack_dir(a.source, a.split, a.out_root)
    bj = P / "build.json"
    if bj.is_file() and json.loads(bj.read_text()).get("status") == "complete":
        raise SystemExit(f"{P}: a complete pack exists; refusing to overwrite (remove it deliberately first)")
    parts = P / "_parts"
    parts.mkdir(parents=True, exist_ok=True)
    toks = OP.packed_tokens(a.split)
    if a.limit:
        toks = toks[: a.limit]
    n = len(toks)
    chunks = [(i, list(range(s, min(s + a.chunk, n)))) for i, s in enumerate(range(0, n, a.chunk))]
    # F4: the provenance is taken BEFORE any chunk is built and is the one build.json records
    prov = provenance(subset, gt_dir, targets)
    inputs_sha, inputs = input_fingerprint(prov, gt_dir)
    pkey = part_key(targets, inputs_sha)
    todo = []
    for ci, rows in chunks:
        pp = parts / f"c{ci:05d}.npz"
        if pp.is_file():
            try:
                with np.load(pp, allow_pickle=False) as z:
                    if np.array_equal(z["tokens"].astype(str), toks[rows]) and np.array_equal(z["rows"], rows) \
                            and "part_key" in z.files and str(z["part_key"]) == pkey:
                        continue
            except Exception:  # noqa: BLE001  (corrupt / partial part: rebuild it)
                pass
        todo.append((ci, rows, [str(t) for t in toks[rows]], pp))
    print(f"[{now_kst()}] {a.split}: {n} tokens, {len(chunks)} chunks, {len(todo)} to build, workers {a.workers}, "
          f"targets {targets}, inputs {inputs_sha[:16]}, out {P}", flush=True)
    status = "complete"
    fail_msg = None
    if todo:
        ctx = get_context("fork")
        done = 0
        try:
            with ctx.Pool(a.workers, initializer=_init,
                          initargs=(subset, str(gt_dir) if gt_dir else "", targets, pkey)) as pool:
                for ci, k in pool.imap_unordered(run_chunk, todo, chunksize=1):
                    done += 1
                    if done % 50 == 0 or done == len(todo):
                        el = time.time() - t_start
                        print(f"[{now_kst()}] chunks {done}/{len(todo)}  {el:.0f}s", flush=True)
        except O.CausalityError as e:
            status, fail_msg = "failed_causality", f"{e}"
            traceback.print_exc()
    if status == "complete":
        # assemble
        names = NAMES + (TNAMES if targets else [])
        specs = {**OP.PACK_FILES, **(OP.TARGET_FILES if targets else {})}
        mm = {k: np.lib.format.open_memmap(str(P / f"{k}.npy.tmp"), mode="w+", dtype=specs[k][0],
                                           shape=(n,) + tuple(specs[k][1])) for k in names}
        sts = [None] * n
        for ci, rows in chunks:
            with np.load(parts / f"c{ci:05d}.npz", allow_pickle=False) as z:
                if not np.array_equal(z["tokens"].astype(str), toks[rows]):
                    raise RuntimeError(f"chunk {ci}: token mismatch")
                if "part_key" not in z.files or str(z["part_key"]) != pkey:      # F4: never assemble mixed inputs
                    raise RuntimeError(f"chunk {ci}: part key != this build's (inputs {inputs_sha[:16]})")
                for k in names:
                    mm[k][rows[0]:rows[-1] + 1] = z[k]
                for r, s in zip(rows, json.loads(str(z["stats"]))):
                    sts[r] = s
        for k in names:
            mm[k].flush()
        frames = {k: np.asarray(mm[k]) for k in ("frame_valid", "frame_why", "frame_src", "n_raw", "obj_ok",
                                                 "frame_dt", "obj_mask")}
        counts = summarise(P, n, targets, sts, frames)
        if targets:
            counts["target_steps"] = target_stats(np.asarray(mm["fc_state"]), np.asarray(mm["obj"]),
                                                  np.asarray(mm["obj_mask"]))
        del frames
        for k in names:
            del mm[k]
        import pandas as pd
        tmp_tok = P / "tokens.parquet.tmp"
        pd.DataFrame({"token": toks.astype(str), "row": np.arange(n, dtype=np.int64)}).to_parquet(tmp_tok, index=False)
        if a.limit:
            counts["truncation_check"] = "skipped (--limit build: the 1 % rule applies to full splits)"
        elif counts["truncated_frac_of_valid_frames"] > a.max_trunc_frac:
            status = "failed_truncation"
            fail_msg = f"{counts['truncated_frac_of_valid_frames']:.4f} of valid frames truncated at K_OBJ={TC.K_OBJ}"
        if targets and counts["targets"]["n_gt_missing"] > 0:
            status = "failed_gt_missing"
            fail_msg = f"{counts['targets']['n_gt_missing']} tokens without a GT file"
        if status == "complete":
            for k in names:
                os.replace(P / f"{k}.npy.tmp", P / f"{k}.npy")
            os.replace(tmp_tok, P / "tokens.parquet")
    else:
        counts = None
    out = dict(version=OP.PACK_VERSION, objects_version=O.OBJECTS_VERSION, status=status, fail=fail_msg,
               source=a.source, split=a.split, subset=subset, targets=targets, limit=int(a.limit), n=int(n),
               K_OBJ=TC.K_OBJ, max_trunc_frac=a.max_trunc_frac, D_OBJ=TC.D_OBJ, N_FRAMES=TC.N_FRAMES, SCORE_THR=list(TC.SCORE_THR),
               DT_RANGE=list(TC.DT_RANGE), FC_T=list(TC.FC_T), match_gate_m=O.MATCH_GATE_M,
               columns=["x", "y", "cos_h", "sin_h", "L", "W", "vx", "vy", "score"] + [f"cls_{c}" for c in FT.CLASS_NAMES]
               + ["age_s"], frames=["t0", "t-0.5", "t-1.0"], why_codes=list(O.WHY),
               src_codes={"0": "none", "1": "current-frame cache", "2": "_future-named cache"},
               fc_state_codes={"0": "no target", "1": "OBS", "2": "matched, ABSENT_OFFICIAL", "3": "t > t_avail",
                               "4": "matched singleton GT track (official hold; oracle only, never a motion target)"},
               provenance=prov, inputs_sha256=inputs_sha, inputs={k: v for k, v in inputs.items() if k != "provenance"},
               counts=counts, workers=a.workers, chunk=a.chunk,
               kst_start=kst_start, kst_end=now_kst(), wall_s=round(time.time() - t_start, 1))
    if status == "complete":
        out["files"] = {f.name: dict(sha256=sha256_file(f), bytes=f.stat().st_size)
                        for f in sorted(P.iterdir()) if f.is_file() and f.suffix in (".npy", ".parquet")}
        out["kst_end"] = now_kst()
        out["wall_s"] = round(time.time() - t_start, 1)
    tmp = P / "build.json.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, bj)
    if status == "complete" and not a.keep_parts:
        shutil.rmtree(parts)
    print(f"[{now_kst()}] {a.split}: status {status} {fail_msg or ''} wall {out['wall_s']}s", flush=True)
    if counts:
        print(json.dumps({k: v for k, v in counts.items() if k not in ("errors_first20",)}, indent=0)[:4000], flush=True)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
