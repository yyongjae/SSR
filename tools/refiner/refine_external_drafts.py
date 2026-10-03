#!/usr/bin/env python
"""Apply a frozen stage-T refiner to ONE external draft per token (e.g. a planner's navtest trajectories) instead of
the 13-draft bank; theta 0 (the correction is always applied, the gate is recorded but ignored).

  python tools/refiner/refine_external_drafts.py predict --run <runs>/<run> --drafts-pkl <pkl> --out <dir> --gpu 0
  python tools/refiner/refine_external_drafts.py original --drafts-pkl <pkl> --out <dir>     (unrefined drafts, same rows)
  then: python tools/refiner/score_trajectories.py --drafts <dir>/refined.npz --tokens <dir>/tokens.parquet \
            --out <dir>/scores.parquet --workers 4

Same inference path as eval_refiner.py predict (STAGE E — E2 EVALUATION SPEC, side experiment):
  * the run's net + norms from ckpt_<ckpt>.pt (train_refiner.load_run_model), the run's decoder mode (A), autocast fp16
    on CUDA when the run trained with amp (as eval_refiner);
  * the teacher cache of the evaluated split exactly as eval_refiner.predict selects it (arm T: BEVFusion
    TeacherCache.for_subset(SPLIT_SUBSET[split]) -> navtest: cache_val_50x100, sha head checked against the run; arm M:
    eval_refiner.run4_teacher -> ReSMap navtest cache, sha checked, coverage checked; arm none: no BEV);
  * ego state v0 / a0 / eds / cmd from the packed split (PackedSplit part 'human'), as stage T;
  * the loader item of the packed row (data.TokenDataset) with the bank replaced by the single external draft:
    drafts [1, 8, 3] (float32), draft_valid [True], family [-1], params 0, labels NaN.  The refiner has no interaction
    between the drafts of a token (refiner_net: per-draft self-attention over [draft + 48 stations], cross-attention
    per query), so the result for a draft does not depend on the other drafts of the token
    (tests/test_refine_external_drafts.py: bank slot k fed alone == eval_refiner bank output for slot k).
Rows: every packed row whose token is in the pkl and whose 'human' part is written (ego state), frame_gap rows
INCLUDED (frame_gap concerns the future human frames, not the t0 ego state; stage T excluded them only because the
bank / labels are built from the human trajectory); recorded per token in tokens.parquet (column frame_gap).
Outputs (<out>): pred.npz (tokens, rows, tau0 [N,1,8,3], tau1, p_g [N,1], z_lon [N,1,6], w_lat [N,1,6], c_lon [N,1,8],
e_lat [N,1,8], alpha, beta, lon_live [N,1]), refined.npz (tokens, drafts = tau1 [N,1,8,3] float32; for 'original'
drafts = tau0), tokens.parquet (token, log, frame_gap), predict_meta.json.
Draft pkl format: {token: [8, 3]} or {'trajectories': {token: [8, 3]}, 'meta': ...} (N frame, t = 0.5 .. 4 s).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner.decoder import lon_live  # noqa: E402
import eval_refiner as EV  # noqa: E402
import train_refiner as TR  # noqa: E402


def load_drafts_pkl(path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict) and "trajectories" in d and isinstance(d["trajectories"], dict):
        d = d["trajectories"]
    out = {}
    for t, v in d.items():
        v = np.asarray(getattr(v, "poses", v), np.float32)
        if v.shape != (8, 3) or not np.isfinite(v).all():
            raise ValueError(f"{t}: draft {v.shape} (need finite [8, 3])")
        out[str(t)] = v
    return out


class ExternalDraftDataset(RD.TokenDataset):
    """TokenDataset whose bank is replaced by one external draft per token (module docstring)."""

    def __init__(self, packed, rows, teacher, drafts: Dict[str, np.ndarray]):
        super().__init__(packed, rows, teacher)
        self.ext = drafts

    def __getitem__(self, i: int) -> Dict:
        d = super().__getitem__(i)
        d["drafts"] = np.asarray(self.ext[d["token"]], np.float32)[None]
        d["draft_valid"] = np.ones(1, bool)
        d["family"] = np.full(1, -1, np.int8)
        d["params"] = np.zeros((1, 6), np.float32)
        d["labels"] = np.full((1, len(RD.LABEL_COLS)), np.nan)
        return d


def select_rows(packed, drafts: Dict[str, np.ndarray]) -> np.ndarray:
    ok = np.asarray(packed.done[:, RD.PART_ID["human"]]) == 1
    ok &= packed.index.token.astype(str).isin(set(drafts)).values
    return np.flatnonzero(ok)


def teacher_for(run_cfg, split, rows, packed, teacher_root=None, resmap_root=None):
    """The loader teacher exactly as eval_refiner.predict selects it (no shuffle / subset / branch drop)."""
    arm = run_cfg["arm"]
    if arm == "none":
        return None
    if arm == "T":
        t = RD.TeacherCache(teacher_root) if teacher_root else RD.TeacherCache.for_subset(RD.SPLIT_SUBSET[split])
        if t.sha_head != run_cfg.get("teacher_sha_head", RD.TEACHER_SHA_HEAD):
            raise RuntimeError(f"teacher sha head {t.sha_head} != the run's {run_cfg.get('teacher_sha_head')}")
        return t
    a = SimpleNamespace(split=split, teacher_root=teacher_root, resmap_root=resmap_root, shuffle_teacher_seed=None,
                        shuffle_which=None)
    teacher, _, _ = EV.run4_teacher(a, run_cfg, rows, packed)
    return teacher


def _write_common(out: Path, tokens, rows, packed, tau0, tau1, extra: Dict, meta: Dict):
    out.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray(tokens).astype(str)
    np.savez(out / "pred.npz", tokens=tokens, rows=rows, tau0=tau0, tau1=tau1, **extra)
    np.savez(out / "refined.npz", tokens=tokens, drafts=tau1.astype(np.float32))
    pd.DataFrame({"token": tokens, "log": packed.index.log.values[rows],
                  "frame_gap": np.asarray(packed.arrays["frame_gap"], bool)[rows]}).to_parquet(out / "tokens.parquet",
                                                                                                 index=False)
    (out / "predict_meta.json").write_text(json.dumps(meta, indent=1))


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@torch.no_grad()
def predict(a) -> Path:
    dev = TR._device(a.gpu)
    net, cfg = TR.load_run_model(Path(a.run), a.ckpt, dev)
    drafts = load_drafts_pkl(a.drafts_pkl)
    packed = RD.PackedSplit(a.split, a.packed_root)
    rows = select_rows(packed, drafts)
    if a.limit:
        rows = rows[: a.limit]
    teacher = teacher_for(cfg, a.split, rows, packed, a.teacher_root, a.resmap_root)
    ds = ExternalDraftDataset(packed, rows, teacher, drafts)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=a.tokens_per_batch, shuffle=False, collate_fn=RD.collate_tokens,
                        num_workers=min(int(a.loader_workers), 4), persistent_workers=False,
                        prefetch_factor=4 if a.loader_workers > 0 else None)
    mode = cfg.get("mode", "A")
    use_amp = bool(cfg.get("amp", 1)) and dev.type == "cuda"
    keys = ("rows", "tau0", "tau1", "p_g", "z_lon", "w_lat", "c_lon", "e_lat", "alpha", "beta", "lon_live")
    rec = {k: [] for k in keys}
    toks = []
    t0 = time.time()
    for i, batch in enumerate(loader):
        batch = RD.batch_to(batch, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            o = net(batch["bev"], batch["tau0"], batch["v0"], batch["a0"], batch["eds"], batch["cmd"])
        dec = TR.decode_batch(o, batch, mode)
        T, K = batch["tau0"].shape[:2]
        rec["rows"].append(batch["rows"].cpu().numpy())
        rec["tau0"].append(batch["tau0"].cpu().numpy())
        rec["tau1"].append(dec["traj"].float().reshape(T, K, 8, 3).cpu().numpy())
        rec["p_g"].append(torch.sigmoid(o["gate_logit"].float()).cpu().numpy())
        rec["z_lon"].append(o["z_lon"].float().cpu().numpy())
        rec["w_lat"].append(o["w_lat"].float().cpu().numpy())
        rec["c_lon"].append(dec["c_lon"].float().reshape(T, K, -1).cpu().numpy())
        rec["e_lat"].append(dec["e_lat"].float().reshape(T, K, -1).cpu().numpy())
        rec["alpha"].append(dec["flags"]["alpha"].reshape(T, K).float().cpu().numpy())
        rec["beta"].append(dec["flags"]["beta"].reshape(T, K).float().cpu().numpy())
        rec["lon_live"].append(lon_live(o["z_lon"].float()).cpu().numpy())
        toks += batch["tokens"]
        if (i + 1) % 50 == 0:
            print(f"[predict] {len(toks)}/{len(rows)} tokens {time.time() - t0:.0f}s", flush=True)
    R = {k: np.concatenate(v) for k, v in rec.items()}
    tokens = np.array(toks)
    assert np.array_equal(R["rows"], rows)
    for i, t in enumerate(tokens):  # the fed draft is the pkl draft, bitwise
        if not np.array_equal(R["tau0"][i, 0], drafts[t]):
            raise AssertionError(f"{t}: fed draft != pkl draft")
    meta = dict(run=str(a.run), ckpt=a.ckpt, arm=cfg["arm"], mode=mode, amp=use_amp, split=a.split,
                drafts_pkl=str(a.drafts_pkl), drafts_pkl_sha256=_sha(a.drafts_pkl), n_pkl=len(drafts),
                n_tokens=int(len(tokens)), n_frame_gap=int(np.asarray(packed.arrays["frame_gap"], bool)[rows].sum()),
                theta=0.0, teacher=str(getattr(teacher, "root", None)), sec=round(time.time() - t0, 1),
                tau1_eq_tau0_frac=float((R["tau1"] == R["tau0"]).all((2, 3)).mean()),
                created=time.strftime("%Y-%m-%dT%H:%M:%S"), gpu=a.gpu)
    _write_common(Path(a.out), tokens, R.pop("rows"), packed, R.pop("tau0"), R.pop("tau1"), R, meta)
    print(json.dumps(meta), flush=True)
    return Path(a.out)


def original(a) -> Path:
    drafts = load_drafts_pkl(a.drafts_pkl)
    packed = RD.PackedSplit(a.split, a.packed_root)
    rows = select_rows(packed, drafts)
    if a.limit:
        rows = rows[: a.limit]
    tokens = packed.index.token.astype(str).values[rows]
    tau0 = np.stack([drafts[t] for t in tokens])[:, None]
    meta = dict(kind="original (unrefined)", split=a.split, drafts_pkl=str(a.drafts_pkl),
                drafts_pkl_sha256=_sha(a.drafts_pkl), n_pkl=len(drafts), n_tokens=int(len(tokens)),
                n_frame_gap=int(np.asarray(packed.arrays["frame_gap"], bool)[rows].sum()),
                created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _write_common(Path(a.out), tokens, rows, packed, tau0, tau0.copy(), {}, meta)
    print(json.dumps(meta), flush=True)
    return Path(a.out)


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["predict", "original"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--drafts-pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--packed-root", default=str(RD.DATA_ROOT / "packed"))
    ap.add_argument("--teacher-root", default=None)
    ap.add_argument("--resmap-root", default=None)
    ap.add_argument("--tokens-per-batch", type=int, default=32)
    ap.add_argument("--loader-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    return ap


def main(argv=None):
    a = get_parser().parse_args(argv)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if a.cmd == "predict":
        if not a.run:
            raise SystemExit("predict needs --run")
        predict(a)
    else:
        original(a)


if __name__ == "__main__":
    main()
