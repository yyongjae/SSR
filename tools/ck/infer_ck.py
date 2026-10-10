#!/usr/bin/env python
"""Run a trained CK net (teacher or student) on a split's packed v2 candidates (contract pipeline.infer_ck).

  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 /venv/ssr/bin/python tools/ck/infer_ck.py --run CK_DATA/train/ckS_p1 --split navtest

Per output row (packed row order, or the --rows subset order with a 'row' column in tokens.parquet):
  score_logit f16 [N,K,5] (CK_KEYS), z_lon / w_lat f32 [N,K,6], c_lon / e_lat f32 [N,K,6] (decoded [2:], slope 0.0),
  corr_traj f32 [N,K,8,3], corr_score_logit f16 [N,K,5] (CK rescoring of its own corr_traj, same scene features),
  extra_score_logit f16 [N,K2,5] (--extra), done bool [N].
--decode 0 (kd_targets rescoring): only <score-name>.npy + <done-name>.npy.
Shards (--shard i --nshard n) write disjoint rows of shared memmaps (created under a file lock); resumable via done.
Rows whose packed ok is False are skipped (done stays False, arrays NaN).
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[2])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)
import navsim  # noqa: E402

assert navsim.__file__.startswith(CK), navsim.__file__

import argparse  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional, Union  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402


def _arr(x, mmap=True):
    if x is None:
        return None
    if isinstance(x, (str, Path)):
        return np.load(x, mmap_mode="r" if mmap else None)
    return x


def packed_info(split: str):
    import pandas as pd
    d = U.ck_data() / "packed" / split
    tdf = pd.read_parquet(d / "tokens.parquet")
    ok = np.load(d / "ok.npy") if (d / "ok.npy").is_file() else np.ones(len(tdf), bool)
    return d, tdf, ok


def run_infer(run_dir: Union[str, Path], split: str, traj=None, extra=None, rows=None, out=None, which: str = "last",
              batch: int = 16, workers: int = 8, shard: int = 0, nshard: int = 1, decode: bool = True,
              rescore_corr: bool = True, score_name: str = "score_logit", done_name: str = "done",
              device: str = "cuda", limit: int = 0, net_cfg=None, dataset=None, collate_fn=None,
              log_every: int = 200) -> Path:
    """Infer one shard; returns the output dir.  traj / extra: npy path or array [N_packed, K(2), 8, 3] aligned to
    packed rows (traj replaces the packed candidates).  rows: packed row subset (array or npy path).
    net_cfg / dataset / collate_fn: injection for tests ((net, cfg), a CKDataset-like object, collate)."""
    from navsim.agents.para_ssr.ck import constants as Cn

    run_dir = Path(run_dir)
    if device.startswith("cuda"):
        U.gpu_guard(device)
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if net_cfg is None:
        from navsim.agents.para_ssr.ck.model import load_ck
        net, cfg = load_ck(run_dir, which=which, device=str(dev))
    else:
        net, cfg = net_cfg
    net = net.to(dev).eval()
    arm = cfg["arm"]
    K = int(cfg.get("k", Cn.K_CAND))
    out = Path(out) if out else U.ck_data() / "infer" / run_dir.name / split
    out.mkdir(parents=True, exist_ok=True)
    if dataset is None:
        pdir, tdf, pok = packed_info(split)
        n_packed = len(tdf)
    else:
        tdf, pok, n_packed = dataset.tokens_df, dataset.packed_ok, len(dataset.packed_ok)
    rows = _arr(rows, mmap=False)
    if rows is not None:
        rows = np.asarray(rows, np.int64)
    elif limit:
        rows = np.arange(min(limit, n_packed), dtype=np.int64)
    N = len(rows) if rows is not None else n_packed
    out_rows = rows if rows is not None else np.arange(N, dtype=np.int64)
    traj_a, extra_a = _arr(traj), _arr(extra)
    K2 = int(extra_a.shape[1]) if extra_a is not None else 0
    specs = {done_name: ((N,), np.bool_, False), score_name: ((N, K, 5), np.float16, np.nan)}
    if decode:
        specs.update({"z_lon": ((N, K, 6), np.float32, np.nan), "w_lat": ((N, K, 6), np.float32, np.nan),
                      "c_lon": ((N, K, 6), np.float32, np.nan), "e_lat": ((N, K, 6), np.float32, np.nan),
                      "corr_traj": ((N, K, 8, 3), np.float32, np.nan)})
        if rescore_corr:
            specs["corr_score_logit"] = ((N, K, 5), np.float16, np.nan)
    if K2:
        specs["extra_score_logit"] = ((N, K2, 5), np.float16, np.nan)
    with U.FileLock(out / ".lock"):
        A = {k: U.open_memmap(out / f"{k}.npy", s, d, fill=f) for k, (s, d, f) in specs.items()}
        if not (out / "tokens.parquet").is_file():
            sub = tdf.iloc[out_rows].copy().reset_index(drop=True)
            sub["row"] = out_rows
            tmp = out / f".tokens.tmp{os.getpid()}.parquet"
            sub.to_parquet(tmp, index=False)
            os.replace(tmp, out / "tokens.parquet")
        meta_p = out / "meta.json"
        meta = U.read_json(meta_p, {}) or {}
        ck_file = run_dir / f"ckpt_{which}.pt"
        meta.update(run=str(run_dir), which=which, arm=arm, split=split, n=N, k=K, k2=K2, slope=Cn.LON_ST_SLOPE["eval"],
                    ckpt_sha256=U.sha256_file(ck_file, 64) if ck_file.is_file() else None,
                    rows=None if rows is None else int(len(rows)))
        meta.setdefault("passes", {})[score_name] = dict(
            traj=str(traj) if isinstance(traj, (str, Path)) else ("array" if traj is not None else "packed cand"),
            extra=str(extra) if isinstance(extra, (str, Path)) else ("array" if extra is not None else None),
            decode=bool(decode), rescore_corr=bool(rescore_corr and decode), done=done_name)
        U.write_json(meta_p, meta)
    done = A[done_name]
    pos = U.shard_rows(N, shard, nshard)
    pos = pos[~np.asarray(done[pos], bool)]
    pos = pos[np.asarray(pok)[out_rows[pos]]]                      # packed-ok rows only
    print(f"[infer] {run_dir.name} arm {arm} split {split} -> {out}: shard {shard}/{nshard} todo {len(pos)} of {N}",
          flush=True)
    if len(pos) == 0:
        return out
    if dataset is None:
        from tools.ck.data.ck_dataset import CKDataset, collate_ck
        ds = CKDataset(split, bev=arm, k=K, labels=None, gt=False, rows=out_rows[pos])
        collate_fn = collate_fn or collate_ck
    else:
        ds = dataset.subset(out_rows[pos])
    pos_of_row = {int(r): int(p) for p, r in zip(pos, out_rows[pos])}
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_fn, num_workers=workers,
                    pin_memory=dev.type == "cuda", prefetch_factor=4 if workers > 0 else None)
    use_amp = dev.type == "cuda"
    slope = Cn.LON_ST_SLOPE["eval"]
    t0, n, n_bad = time.time(), 0, 0
    with torch.no_grad():
        for b in dl:
            rr = b["rows"] if "rows" in b else b["row"]                 # collate_ck: 'rows' int64 [T]
            prow = [int(r) for r in (rr.tolist() if torch.is_tensor(rr) else rr)]
            p = np.array([pos_of_row[r] for r in prow], np.int64)
            bok = (np.asarray(b["bev_ok"].cpu().numpy() if torch.is_tensor(b["bev_ok"]) else b["bev_ok"], bool)
                   .reshape(-1) if "bev_ok" in b else np.ones(len(p), bool))
            n_bad += int((~bok).sum())
            bev = b.get("bev")
            bev = bev.to(dev, non_blocking=True) if bev is not None else None
            status = b["status"].to(dev).float()
            cand = (torch.from_numpy(np.asarray(traj_a[prow], np.float32)) if traj_a is not None
                    else b["cand"]).to(dev).float()
            ex = torch.from_numpy(np.asarray(extra_a[prow], np.float32)).to(dev) if extra_a is not None else None
            rc = bool(decode and rescore_corr)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                o = net(bev, cand, status, extra=ex, decode=decode, slope=slope, rescore_corr=rc)
            g = lambda x: x.float().cpu().numpy()[bok]
            q = p[bok]                     # rows whose BEV is missing (bev_ok False) stay NaN / not done
            A[score_name][q] = g(o["score_logit"]).astype(np.float16)
            if K2:
                A["extra_score_logit"][q] = g(o["extra_score_logit"]).astype(np.float16)
            if decode:
                c = o["corr"]
                A["z_lon"][q] = g(o["z_lon"])
                A["w_lat"][q] = g(o["w_lat"])
                A["c_lon"][q] = g(c["c_lon"][..., 2:])
                A["e_lat"][q] = g(c["e_lat"][..., 2:])
                A["corr_traj"][q] = g(c["traj"])
                if rc:          # CK rescoring of its own corrections, same scene features (CKNet rescore_corr)
                    A["corr_score_logit"][q] = g(o["corr_score_logit"]).astype(np.float16)
            done[q] = True
            n += len(p)
            if n % log_every < len(p) or n == len(pos):
                el = time.time() - t0
                print(f"[infer] {n}/{len(pos)} {el:.0f}s {n / max(el, 1e-6):.1f} tok/s", flush=True)
    for m in A.values():
        m.flush()
    print(f"[infer] done {n} rows in {time.time() - t0:.0f}s ({n_bad} without BEV left undone)", flush=True)
    return out


def get_parser():
    ap = argparse.ArgumentParser(description="CK inference on packed candidates",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--run", required=True, help="CK run dir (or a run name under CK_DATA/train)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--which", default="last")
    ap.add_argument("--out", default=None, help="default CK_DATA/infer/<run>/<split>")
    ap.add_argument("--traj", default="packed", help="packed | <npy [N,K,8,3] aligned to packed rows>")
    ap.add_argument("--extra", default=None, help="<npy [N,K2,8,3] aligned to packed rows>")
    ap.add_argument("--rows", default=None, help="<npy int> packed row subset")
    ap.add_argument("--limit", type=int, default=0, help="first N packed rows (smoke)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--decode", type=int, default=1)
    ap.add_argument("--rescore-corr", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    return ap


def resolve_run(run: str) -> Path:
    p = Path(run)
    if p.is_dir():
        return p
    q = U.ck_data() / "train" / run
    if q.is_dir():
        return q
    raise SystemExit(f"run dir not found: {run} (nor {q})")


def main(argv=None):
    a = get_parser().parse_args(argv)
    if a.workers > 16:
        raise SystemExit("--workers <= 16 per process")
    run_infer(resolve_run(a.run), a.split, traj=None if a.traj == "packed" else a.traj, extra=a.extra,
              rows=a.rows, out=a.out, which=a.which, batch=a.batch, workers=a.workers, shard=a.shard,
              nshard=a.nshard, decode=bool(a.decode), rescore_corr=bool(a.rescore_corr), device=a.device,
              limit=a.limit)


if __name__ == "__main__":
    main()
