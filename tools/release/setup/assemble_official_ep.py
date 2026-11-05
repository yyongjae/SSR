#!/usr/bin/env python
"""The official-EP score file: pdm_score_256 with the EP column from the raw256 labels, in the same layout.

  assemble_official_ep.py [--ck DIR] [--plan-data DIR] [--out-dir DIR]

Reads   <ck>/labels/{navtrain_train,navtrain_val}/raw256/{labels,ok}.npy (50_warmup_labels.sh) and
        <ck>/packed/<split>/tokens.parquet; <plan-data>/pdm_score_256.tokens.json gives the token order.
Checks  every anchor of every token is scored; NC, DAC, TTC, C equal <plan-data>/pdm_score_256.npy; the two files equal their pins
        before they appear.
Writes  <out-dir>/pdm_score_256_officialEP.npy f16 [103288, 5, 256], keys NC, DAC, EP, TTC, C, and its .tokens.json.
        <out-dir> defaults to <ck>/raw256.  An existing file is never overwritten.  CPU only, about 2 minutes, 0.25 GiB.

agent.config.plan_score_file reads this file (tools/ck/e2e/launch_util.py).  50_warmup_labels.sh runs this script at its end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]      # repo root of this worktree
sys.path.insert(0, str(ROOT))
from navsim.common import ssr_paths as SP  # noqa: E402

SPLITS = ("navtrain_train", "navtrain_val")
LABEL_COLS = ("nc", "dac", "ep", "ttc", "comfort", "ddc", "pdms", "raw_progress", "pdm_progress_eff")
KEYS = ("nc", "dac", "ep", "ttc", "comfort")    # axis 1 of pdm_score_256
NAME = "pdm_score_256_officialEP"
PINS = {".npy": "a707bd3508ff04e9", ".tokens.json": "b1278ec613611037"}     # sha16 of the two files


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def assemble(ck: Path, plan: Path) -> np.ndarray:
    tokens = json.loads((plan / "pdm_score_256.tokens.json").read_text())
    index = {t: i for i, t in enumerate(tokens)}
    wote = np.load(plan / "pdm_score_256.npy", mmap_mode="r")
    out = np.full(wote.shape, np.nan, np.float16)
    cols = [LABEL_COLS.index(k) for k in KEYS]
    for split in SPLITS:
        d = ck / "labels" / split / "raw256"
        lab, ok = np.load(d / "labels.npy", mmap_mode="r"), np.load(d / "ok.npy")
        if not ok.all():
            raise SystemExit(f"{split}: {int((~ok).sum())} anchors are not scored in {d}")
        rows = np.array([index[t] for t in pd.read_parquet(ck / "packed" / split / "tokens.parquet").token.tolist()])
        off = np.asarray(lab[..., cols], np.float32).transpose(0, 2, 1).astype(np.float16)      # [N, 5, 256]
        same = np.asarray(wote[np.sort(rows)])[np.argsort(np.argsort(rows))]                    # packed order
        for j, k in enumerate(KEYS):
            if k != "ep" and not np.array_equal(off[:, j], same[:, j]):
                raise SystemExit(f"{split}: {k} differs from pdm_score_256")
        out[rows] = off
        print(f"{split}: {len(rows)} tokens, NC DAC TTC C equal pdm_score_256")
    if np.isnan(out).any():
        raise SystemExit("some tokens of pdm_score_256.tokens.json are in neither split")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ck", default=str(Path(SP.SSR_DATA_ROOT) / "ck"))
    ap.add_argument("--plan-data", default=str(SP.SSR_PLAN_DATA))
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args(argv)
    ck, plan = Path(a.ck), Path(a.plan_data)
    dest = Path(a.out_dir) if a.out_dir else ck / "raw256"
    outs = {suffix: dest / (NAME + suffix) for suffix in PINS}
    for p in outs.values():
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    dest.mkdir(parents=True, exist_ok=True)
    tmp = {suffix: dest / f".{NAME}.tmp{os.getpid()}{suffix}" for suffix in PINS}
    np.save(tmp[".npy"], assemble(ck, plan))
    shutil.copyfile(plan / "pdm_score_256.tokens.json", tmp[".tokens.json"])
    got = {suffix: sha16(p) for suffix, p in tmp.items()}
    bad = [f"{NAME}{s}: sha16 {got[s]} != pin {PINS[s]}" for s in PINS if got[s] != PINS[s]]
    if bad:
        for p in tmp.values():
            p.unlink()
        raise SystemExit("; ".join(bad) + " (check the raw256 labels with verify_assets)")
    for suffix, p in tmp.items():
        os.replace(p, outs[suffix])
    print(f"{outs['.npy']} sha16 {got['.npy']}, {outs['.tokens.json']} sha16 {got['.tokens.json']}: equal their pins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
