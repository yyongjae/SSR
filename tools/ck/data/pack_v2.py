#!/usr/bin/env python
"""CK Phase 1: pack the per-token v2 dump (plan npz) of a split into row-aligned arrays.

  pack_v2.py --split S [--root R] [--k 16] [--tokens split|dumped|<parquet>] [--allow-missing] [--threads 16]

Output <root>/packed/<split>/ (np.lib.format .npy, load with mmap_mode='r'):
  tokens.parquet  token, log, city, row   (row order = split tokens parquet order; 'dumped' keeps that order)
  cand.npy        f32 [N,K,8,3]  v2 top-K executed candidates (anchor + offset), descending plan_final_rewards
  cand_idx.npy    i16 [N,K]      anchor index of each candidate (-1 if missing)
  v2_final.npy    f32 [N,K]      plan_final_rewards of the candidates
  v2_im.npy       f32 [N,K]      im_rewards (softmax over 256)
  v2_sim.npy      f32 [N,K,5]    sim_rewards sigmoid, CK_KEYS order (nc, dac, ep, ttc, comfort)
  status.npy      f32 [N,8]      status_feature (cmd one-hot 4 + vx, vy, ax, ay)
  gt_traj.npy     f32 [N,8,3]    human future (N frame), NaN if unavailable
  sub_traj.npy    f32 [N,8,3]    v2 submitted trajectory (== cand[:, 0])
  ok.npy          bool [N]       token dumped and consistent (cand[0] == trajectory, finite, BEV present)
  meta.json       n, k, n_ok, v2 ckpt sha16, dump dir, anchors sha16, created, per-check counts

K > dump K (e.g. 32 for the optimism-bias check): candidates 0..15 are the stored f32 cand, 16..K-1 are rebuilt from
anchors + stored f16 offset (rank by the stored f32 final); meta records 'k_rebuilt_from_f16_offset'.
Missing tokens fail the pack unless --allow-missing (then ok[row] = False and the rows are NaN / -1).
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root of this worktree
from tools.ck.data import common as CM  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ARRAYS = ("cand", "cand_idx", "v2_final", "v2_im", "v2_sim", "status", "gt_traj", "sub_traj", "ok")


def resolve_tokens(split_name: str, spec: str, dump: Path) -> pd.DataFrame:
    df = CM.split_tokens(split_name)
    if spec in ("", "split"):
        return df
    if spec == "dumped":
        m = [CM.plan_path(dump, t).is_file() for t in df.token]
        return df[m].reset_index(drop=True)
    sub = pd.read_parquet(spec)
    keep = set(sub.token.astype(str))
    out = df[df.token.isin(keep)].reset_index(drop=True)
    assert len(out) == len(keep), f"{len(keep) - len(out)} tokens of {spec} are not in split {split_name}"
    return out


def read_token(tok: str, dump: Path, k: int, anchors: np.ndarray, need_bev: bool):
    """-> (dict of per-token arrays, reason or '')."""
    p = CM.plan_path(dump, tok)
    if not p.is_file():
        return None, "missing_plan"
    if need_bev and not CM.bev_path(dump, tok).is_file():
        return None, "missing_bev"
    with np.load(p, allow_pickle=False) as z:
        kd = [int(n[len("cand_top"):]) for n in z.files if n.startswith("cand_top") and not n.endswith("_idx")]
        assert len(kd) == 1, z.files
        kd = kd[0]
        cand_d = np.asarray(z[f"cand_top{kd}"], np.float32)
        idx_d = np.asarray(z[f"cand_top{kd}_idx"], np.int64)
        final = np.asarray(z["final"], np.float32)
        im = np.asarray(z["im"], np.float32)
        sim = np.asarray(z["sim"], np.float32)              # [5, 256]
        traj = np.asarray(z["trajectory"], np.float32)
        status = np.asarray(z["status"], np.float32)
        gt = np.asarray(z["gt_traj"], np.float32) if "gt_traj" in z.files else np.full((8, 3), np.nan, np.float32)
        offset = np.asarray(z["offset"], np.float32) if k > kd else None
    if k <= kd:
        cand, idx = cand_d[:k], idx_d[:k]
    else:
        order = np.argsort(-final, kind="stable")[:k]
        if not np.array_equal(order[:kd], idx_d):         # ties: keep the stored order for the first kd
            rest = [i for i in order if i not in set(idx_d.tolist())][: k - kd]
            order = np.concatenate([idx_d, np.asarray(rest, np.int64)])
        idx = order
        extra = anchors[idx[kd:]] + offset[idx[kd:]]
        cand = np.concatenate([cand_d, extra.astype(np.float32)], 0)
    if not np.isfinite(cand).all():
        return None, "nonfinite_cand"
    if not np.array_equal(cand[0], traj):
        return None, "top1_ne_trajectory"
    return dict(cand=cand, cand_idx=idx.astype(np.int16), v2_final=final[idx], v2_im=im[idx],
                v2_sim=sim[:, idx].T.copy(), status=status, gt_traj=gt, sub_traj=traj), ""


def pack(split_name: str, root=None, k: int = CM.K_CAND, tokens: str = "split", allow_missing: bool = False,
         threads: int = 16, need_bev: bool = True, dump_dir=None) -> dict:
    t0 = time.time()
    dump = Path(dump_dir) if dump_dir else CM.dump_dir(split_name, root)
    out = CM.packed_dir(split_name, root)
    df = resolve_tokens(split_name, tokens, dump)
    n = len(df)
    anchors = np.load(CM.C.ANCHORS).astype(np.float32)
    shapes = dict(cand=((n, k, 8, 3), np.float32, np.nan), cand_idx=((n, k), np.int16, -1),
                  v2_final=((n, k), np.float32, np.nan), v2_im=((n, k), np.float32, np.nan),
                  v2_sim=((n, k, 5), np.float32, np.nan), status=((n, 8), np.float32, np.nan),
                  gt_traj=((n, 8, 3), np.float32, np.nan), sub_traj=((n, 8, 3), np.float32, np.nan))
    W = {name: CM.MemmapWriter(out / f"{name}.npy", dt, shp, fill) for name, (shp, dt, fill) in shapes.items()}
    ok = np.zeros(n, bool)
    reasons = {}
    toks = df.token.tolist()

    def job(i):
        return i, read_token(toks[i], dump, k, anchors, need_bev)

    with ThreadPoolExecutor(max(1, threads)) as ex:
        for j, (i, (rec, why)) in enumerate(ex.map(job, range(n), chunksize=64)):
            if rec is None:
                reasons[why] = reasons.get(why, 0) + 1
                continue
            for name in shapes:
                W[name].arr[i] = rec[name]
            ok[i] = True
            if (j + 1) % 20000 == 0:
                print(f"  {j + 1}/{n} {time.time() - t0:.0f}s", flush=True)
    n_bad = int((~ok).sum())
    if n_bad and not allow_missing:
        for w in W.values():
            w.arr.flush()
            del w.arr
            w.tmp.unlink(missing_ok=True)
        raise SystemExit(f"pack {split_name}: {n_bad}/{n} tokens not usable {reasons} (use --allow-missing)")
    for w in W.values():
        w.finalize()
    CM.atomic_save_npy(out / "ok.npy", ok)
    tdf = df.assign(row=np.arange(n))
    CM.atomic_to_parquet(tdf, out / "tokens.parquet")
    meta = dict(split=split_name, n=n, k=k, n_ok=int(ok.sum()), not_ok=reasons, tokens_spec=tokens,
                v2_ckpt_sha16=CM.V2_CKPT_SHA16, dump_dir=str(dump), anchors_sha16=CM.sha16_array(anchors),
                k_rebuilt_from_f16_offset=max(0, k - CM.K_CAND), token_sha16=CM.sha16_array(np.array(toks, "U16")),
                cand_sha16=CM.sha16_array(np.load(out / "cand.npy", mmap_mode="r")),
                created=time.strftime("%F %T"), seconds=round(time.time() - t0, 1))
    CM.atomic_write_json(out / "meta.json", meta)
    print(f"packed {split_name}: n {n} ok {meta['n_ok']} k {k} {reasons} in {meta['seconds']}s -> {out}", flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=sorted(CM.SPLITS))
    ap.add_argument("--root", default="", help="CK data root (default CK_DATA or $CK_DATA_ROOT)")
    ap.add_argument("--k", type=int, default=CM.K_CAND)
    ap.add_argument("--tokens", default="split", help="split | dumped | <parquet with token>")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--no-bev-check", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    pack(a.split, a.root or None, a.k, a.tokens, a.allow_missing, a.threads, need_bev=not a.no_bev_check)


if __name__ == "__main__":
    main()
