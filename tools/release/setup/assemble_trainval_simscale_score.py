#!/usr/bin/env python
"""The official-EP score file of the trainval_simscale split: the raw256 labels of its rows, in the layout of
pdm_score_256_officialEP, in the row order of its token table.

  assemble_trainval_simscale_score.py [--ck DIR] [--table PARQUET] [--plan-data DIR] [--out-dir DIR] [--skip-done]

Reads   <ck>/labels/trainval_simscale/raw256/{labels,ok}.npy (50_warmup_labels.sh) and <ck>/packed/trainval_simscale/
        tokens.parquet, whose tokens must be the table's (--table, default constants.SPLITS['trainval_simscale']).
Checks  the raw256 labels are this table's (their shard plan has its token_sha16) and every anchor of every row is
        scored.  Real rows: equal, as float16 bytes, to the rows of the pinned
        <ck>/raw256/pdm_score_256_officialEP when that file is there (and is the pinned file); else their NC, DAC, TTC,
        C equal <plan-data>/pdm_score_256 when that is there.  SimScale rows have no reference (WoTE never scored them).
Writes  <out-dir>/pdm_score_256_officialEP_trainval_simscale.npy f16 [N, 5, 256], keys NC, DAC, EP, TTC, C, its
        .tokens.json (table tokens in row order) and .meta.json (counts, what the real rows were compared with).
        <out-dir> defaults to <ck>/raw256.  The pinned 103,288-row file is never touched.  An existing output is never
        overwritten (--skip-done: one with the table's tokens, made from the raw256 labels.npy there now, i.e. the
        sha16 its meta records, is kept).  CPU only, ~1 min per 100k rows.

agent.config.plan_score_file of DATA=trainval_simscale reads this file; label_variants.py reads it as the split's label
source (constants.SPLITS['trainval_simscale']['label_source']).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]      # repo root of this worktree
sys.path.insert(0, str(ROOT))
from navsim.common import ssr_paths as SP  # noqa: E402
from navsim.agents.para_ssr.ck.tokens import token_sha16  # noqa: E402

SPLIT = "trainval_simscale"
NAME = "pdm_score_256_officialEP_trainval_simscale"
LABEL_COLS = ("nc", "dac", "ep", "ttc", "comfort", "ddc", "pdms", "raw_progress", "pdm_progress_eff")
KEYS = ("nc", "dac", "ep", "ttc", "comfort")    # axis 1 of pdm_score_256
PINNED = "pdm_score_256_officialEP"             # the 103,288-row file of the navtrain recipe
PINNED_SHA16 = {".npy": "a707bd3508ff04e9", ".tokens.json": "b1278ec613611037"}    # = assemble_official_ep.PINS
CHUNK = 8192


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def score_rows(lab: np.ndarray) -> np.ndarray:
    """labels [n, 256, 9] (LABEL_COLS) -> [n, 5, 256] float16 (KEYS), as assemble_official_ep."""
    cols = [LABEL_COLS.index(k) for k in KEYS]
    return np.asarray(lab[..., cols], np.float32).transpose(0, 2, 1).astype(np.float16)


def read_reference(path_prefix: Path):
    """(scores [M, 5, 256] mmap, {token: row}) of a score file, or None when it is not there."""
    npy, tok = Path(str(path_prefix) + ".npy"), Path(str(path_prefix) + ".tokens.json")
    if not (npy.is_file() and tok.is_file()):
        return None
    return np.load(npy, mmap_mode="r"), {t: i for i, t in enumerate(json.loads(tok.read_text()))}


def compare_rows(out: np.ndarray, tokens: List[str], ref, keys=KEYS) -> Dict[str, int]:
    """Rows of `out` whose token the reference has: n compared, n differing (exact float16 bytes, NaN included) over
    the given keys."""
    scores, index = ref
    have = [(i, index[t]) for i, t in enumerate(tokens) if t in index]
    if not have:
        return dict(n_compared=0, n_differ=0)
    mine, theirs = (np.array(x, np.int64) for x in zip(*have))
    kk = [KEYS.index(k) for k in keys]
    n_bad = 0
    for s in range(0, len(mine), CHUNK):
        a, b = mine[s:s + CHUNK], theirs[s:s + CHUNK]
        o = np.argsort(b)
        ref_rows = np.asarray(scores[b[o]])[np.argsort(o)][:, kk]
        got = np.asarray(out[a])[:, kk]
        n_bad += int((got.view(np.uint16) != ref_rows.view(np.uint16)).reshape(len(a), -1).any(1).sum())
    return dict(n_compared=len(mine), n_differ=n_bad)


def assemble(ck: Path, table: pd.DataFrame, plan: Path, tmp_npy: Path) -> Dict[str, object]:
    """Writes the [N, 5, 256] array to tmp_npy; -> meta (raises SystemExit on a failed check)."""
    toks = table.token.astype(str).tolist()
    pk = pd.read_parquet(ck / "packed" / SPLIT / "tokens.parquet", columns=["token"]).token.astype(str).tolist()
    if pk != toks:
        raise SystemExit(f"{ck}/packed/{SPLIT}/tokens.parquet is not the token table (rebuild it from this table)")
    d = ck / "labels" / SPLIT / "raw256"
    plan_f = d / "shards" / "_plan.json"
    got = json.loads(plan_f.read_text()).get("token_sha16") if plan_f.is_file() else None
    if got != token_sha16(toks):
        raise SystemExit(f"{plan_f}: token_sha16 {got}, the table's is {token_sha16(toks)} (raw256 labels of another "
                         f"table: rebuild them, 50_warmup_labels.sh SPLITS={SPLIT})")
    lab, ok = np.load(d / "labels.npy", mmap_mode="r"), np.load(d / "ok.npy", mmap_mode="r")
    if lab.shape != (len(toks), 256, len(LABEL_COLS)) or ok.shape != (len(toks), 256):
        raise SystemExit(f"{d}: labels {lab.shape} / ok {ok.shape}, expected [{len(toks)}, 256, ...]")
    n_unscored = int((~np.asarray(ok)).sum())
    if n_unscored:
        raise SystemExit(f"{d}: {n_unscored} anchors are not scored")
    out = np.lib.format.open_memmap(tmp_npy, mode="w+", dtype=np.float16, shape=(len(toks), len(KEYS), 256))
    for s in range(0, len(toks), CHUNK):
        out[s:s + CHUNK] = score_rows(lab[s:s + CHUNK])
    out.flush()
    if not np.isfinite(np.asarray(out[:, [KEYS.index(k) for k in ("nc", "dac", "ttc", "comfort")]])).all():
        raise SystemExit("non-finite NC / DAC / TTC / C in the raw256 labels")
    real = (table.source.astype(str) != "simscale").to_numpy()
    meta: Dict[str, object] = dict(split=SPLIT, n=len(toks), n_real=int(real.sum()), n_simscale=int((~real).sum()),
                                   keys=list(KEYS), token_sha16=token_sha16(toks), labels=str(d),
                                   labels_sha16=sha16(d / "labels.npy"))
    bad: List[str] = []
    pinned = read_reference(ck / "raw256" / PINNED)
    real_toks = [t if r else "" for t, r in zip(toks, real)]
    if pinned is not None:
        got = {sfx: sha16(Path(str(ck / "raw256" / PINNED) + sfx)) for sfx in PINNED_SHA16}
        if got != PINNED_SHA16:
            raise SystemExit(f"{ck}/raw256/{PINNED}: sha16 {got} != pins {PINNED_SHA16} (not the pinned file)")
        c = compare_rows(out, real_toks, pinned)
        meta["real_vs_pinned_officialEP"] = c
        if c["n_differ"]:
            bad.append(f"{c['n_differ']} of {c['n_compared']} real rows differ from {PINNED}")
    else:
        meta["real_vs_pinned_officialEP"] = "not compared (file absent)"
        wote = read_reference(plan / "pdm_score_256")
        if wote is not None:
            c = compare_rows(out, real_toks, wote, keys=("nc", "dac", "ttc", "comfort"))
            meta["real_vs_wote_nc_dac_ttc_c"] = c
            if c["n_differ"]:
                bad.append(f"{c['n_differ']} of {c['n_compared']} real rows: NC / DAC / TTC / C differ from pdm_score_256")
        else:
            meta["real_vs_wote_nc_dac_ttc_c"] = "not compared (file absent)"
    del out
    if bad:
        raise SystemExit("; ".join(bad))
    return meta


def main(argv=None) -> int:
    from navsim.agents.para_ssr.ck import constants as C
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ck", default=str(Path(SP.SSR_DATA_ROOT) / "ck"))
    ap.add_argument("--table", default=C.SPLITS[SPLIT]["tokens"])
    ap.add_argument("--plan-data", default=str(SP.SSR_PLAN_DATA))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--skip-done", action="store_true", help="an existing output with the table's tokens is kept")
    a = ap.parse_args(argv)
    ck = Path(a.ck)
    dest = Path(a.out_dir) if a.out_dir else ck / "raw256"
    outs = {sfx: dest / (NAME + sfx) for sfx in (".npy", ".tokens.json", ".meta.json")}
    table = pd.read_parquet(a.table, columns=["token", "source"])
    toks = table.token.astype(str).tolist()
    if any(p.exists() for p in outs.values()):
        if a.skip_done and all(p.is_file() for p in outs.values()) \
                and json.loads(outs[".tokens.json"].read_text()) == toks \
                and np.load(outs[".npy"], mmap_mode="r").shape == (len(toks), len(KEYS), 256):
            done_from = json.loads(outs[".meta.json"].read_text()).get("labels_sha16")
            now = ck / "labels" / SPLIT / "raw256" / "labels.npy"
            if now.is_file() and done_from == sha16(now):
                print(f"{outs['.npy']}: finished for this token table and these raw256 labels, kept")
                return 0
            raise SystemExit(f"{dest}/{NAME}.*: made from other raw256 labels (sha16 {done_from}, {now} now): move it "
                             f"away, then rerun")
        raise SystemExit(f"refusing to overwrite {dest}/{NAME}.*: move it away first")
    dest.mkdir(parents=True, exist_ok=True)
    tmp = {sfx: dest / f".{NAME}.tmp{os.getpid()}{sfx}" for sfx in outs}
    try:
        meta = assemble(ck, table, Path(a.plan_data), tmp[".npy"])
    except BaseException:
        tmp[".npy"].unlink(missing_ok=True)
        raise
    tmp[".tokens.json"].write_text(json.dumps(toks))
    meta.update(npy_sha16=sha16(tmp[".npy"]), tokens_sha16=sha16(tmp[".tokens.json"]))
    tmp[".meta.json"].write_text(json.dumps(meta, indent=1))
    for sfx in (".tokens.json", ".meta.json", ".npy"):          # the .npy last: its presence means complete
        os.replace(tmp[sfx], outs[sfx])
    print(f"{outs['.npy']} [{meta['n']}, 5, 256] f16 ({meta['n_real']} real, {meta['n_simscale']} SimScale rows); real "
          f"rows vs pinned: {meta['real_vs_pinned_officialEP']}; vs WoTE: {meta.get('real_vs_wote_nc_dac_ttc_c', '-')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
