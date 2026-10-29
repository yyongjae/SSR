#!/usr/bin/env python
"""Official scores of a select_npz.py output ({tokens, trajectory} npz): NAVSIM v1 PDMS and NAVSIM v2.2 one-stage
EPDMS.  Replaces the SSR-v2 checkout's data/logs/score_npz.sh, absent on this host (A50), with the scorer calls of km's
corr30_cf_v15_fix/eval_ep5/run_eval_ep5.sh (navtest PDMS 88.88 / EPDMS 86.81 for its sel_model.npz):
  (1) v1 PDMS: tools.corridor.corr_select.work (official simulator + scorer, = the single-trajectory pdm_score) on the
      metric cache of every token (tools.ck.data.common.mc_path; logs from <KM_CORR_ROOT>/packed/<split>/tokens.parquet)
      -> <out>/pdms_<name>.json (x 100 means over the scored tokens, error count)
  (2) v2.2 EPDMS: <out>/sel_<name>.pkl {token: f32 [8, 3]} -> run_pdm_score_one_stage.py with the precomputed agent
      (tools/refiner_v6c/val_epdms.scorer_cmd on its SPLITS[split]: navtest = metric_cache_v22_navtest, test logs,
      the stock navtest split; fewer tokens = a quoted subset split under <out>/config) -> <out>/epdms_<name>.json
      (EPDMS and the 9 terms, 100 x mean over the valid rows; EC over the rows that have a previous frame)
CPU-heavy (process pool + ray): run it under nice, one scoring at a time (the container memory is shared).
A missing input, a scoring error or a non-zero scorer exit returns 1.

  $PY tools/corridor/score_npz.py --npz <sel.npz> --split navtest --out <dir> --name <tag> [--workers 32] [--threads 32]
      [--nice 5] [--skip-epdms]
  $PY tools/corridor/score_npz.py --check --split navtest      # exit 0 only when every scorer input exists
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

CK = str(Path(__file__).resolve().parents[2])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

TERMS = ("pdms", "nc", "dac", "ep", "ttc", "comfort")                    # the columns of corr_select.work
EP_TERMS = ("NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC")      # val_epdms.SHORT, printed after EPDMS


def token_logs(split: str) -> Dict[str, str]:
    import pandas as pd
    root = os.environ.get("KM_CORR_ROOT", str(Path(CK) / "resources/km_corridor"))
    tdf = pd.read_parquet(f"{root}/packed/{split}/tokens.parquet")
    return dict(zip(tdf.token.astype(str), tdf.log.astype(str)))


def check(split: str) -> List[str]:
    """the missing scorer inputs of `split` (empty: all there)"""
    from tools.ck.data.common import MC_ROOT
    from tools.refiner_v6c import val_epdms as VE
    if split not in VE.SPLITS:
        return [f"no EPDMS table for split {split!r} in tools/refiner_v6c/val_epdms.SPLITS ({sorted(VE.SPLITS)})"]
    T = VE.table(split)
    root = os.environ.get("KM_CORR_ROOT", str(Path(CK) / "resources/km_corridor"))
    need = {"v1 metric cache (tools.ck.data.common.MC_ROOT)": MC_ROOT.get(split),
            "v2.2 metric cache (val_epdms.SPLITS)": T["metric_cache"], "navsim logs (val_epdms.SPLITS)": T["logs"],
            "adjacency table (val_epdms.SPLITS)": T["adj"], "NAVSIM v2.2 checkout (val_epdms.NAVSIM22)": VE.NAVSIM22,
            "precomputed agent config": VE.PE_EPDMS / "config/agent/precomputed_agent.yaml",
            "token table (KM_CORR_ROOT)": Path(root) / "packed" / split / "tokens.parquet"}
    return [f"{k}: {v}" for k, v in need.items() if v is None or not Path(v).exists()]


def score_pdms(split: str, toks: List[str], traj: np.ndarray, log_of: Dict[str, str], workers: int) -> dict:
    from tools.ck.data import common as CM
    from tools.corridor.corr_select import work
    items = [(str(CM.mc_path(split, log_of[t], t)), x[None].astype(np.float32)) for t, x in zip(toks, traj)]
    jobs = [items[i:i + 20] for i in range(0, len(items), 20)]
    res = []
    with ProcessPoolExecutor(workers) as ex:
        for x in ex.map(work, jobs):
            res += x
    ok = np.array([not isinstance(x, str) for x in res])
    R = np.array([x[0] for x in res if not isinstance(x, str)], np.float64).reshape(-1, len(TERMS))
    summ = {"split": split, "tokens": len(toks), "scored": int(ok.sum()), "errors": int((~ok).sum()),
            **{k: float(100 * R[:, i].mean()) if len(R) else float("nan") for i, k in enumerate(TERMS)}}
    if (~ok).any():
        summ["first_error"] = [x for x in res if isinstance(x, str)][0]
    return summ


def score_epdms(a, toks: List[str], traj: np.ndarray, log_of: Dict[str, str]) -> dict:
    """one official v2.2 one-stage run (km's run_eval_ep5.sh call); rc != 0 or no csv -> 'error' in the result"""
    from tools.refiner_v6c import val_epdms as VE
    out = Path(a.out).resolve()
    T = {t: np.ascontiguousarray(x, np.float32).reshape(8, 3) for t, x in zip(toks, traj)}
    VE.check_traj(a.name, T)
    assert len(T) == len(toks), f"duplicate tokens in {a.npz}"
    pkl = out / f"sel_{a.name}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(T, f)
    tts = VE.split_for(out, list(toks), [log_of[t] for t in toks], a.split)       # full split, or a quoted subset
    extra = [] if tts == VE.table(a.split)["full_split"] else [f"file://{out}/config"]
    exp_root = out / "epdms_exp"
    exp_name = f"epdms_{a.name}"
    od = exp_root / exp_name / time.strftime("%Y%m%d%H%M%S")
    cmd, env, cwd = VE.scorer_cmd(a.split, pkl, exp_root, exp_name, od, a.threads, a.nice, extra, tts=tts)
    t0 = time.time()
    print(f"[{VE.kst()}] EPDMS {a.name}: {len(toks)} tokens, split {tts}, {a.threads} threads", flush=True)
    with open(out / f"epdms_{a.name}.log", "w") as lf:
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    csvs = sorted(od.rglob("*.csv"), key=lambda x: x.stat().st_mtime)
    res = {"split": a.split, "train_test_split": tts, "tokens": len(toks), "rc": rc, "sec": round(time.time() - t0, 1),
           "csv": str(csvs[-1]) if csvs else None, "log": str(out / f"epdms_{a.name}.log"), "cmd": " ".join(cmd)}
    if rc != 0 or not csvs:
        res["error"] = f"scorer rc {rc}, csv {'found' if csvs else 'missing'} (see {res['log']})"
        return res
    df, _ = VE.load_csv(csvs[-1])
    if set(df.index) != set(toks):
        res["error"] = f"csv token set differs from the npz ({len(df)} rows vs {len(toks)} tokens)"
        return res
    v = df[df.valid]
    res.update({"valid": int(len(v)), "n_EC": int(v[VE.MET[8]].notna().sum()),
                "EPDMS": float(100 * v.score.mean()), **{VE.SHORT[m]: float(100 * v[m].mean()) for m in VE.MET}})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="")
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--out", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--workers", type=int, default=32, help="v1 PDMS process pool")
    ap.add_argument("--threads", type=int, default=32, help="v2.2 EPDMS ray threads")
    ap.add_argument("--nice", type=int, default=5, help="nice of the EPDMS scorer process")
    ap.add_argument("--skip-epdms", action="store_true")
    ap.add_argument("--check", action="store_true", help="only check the scorer inputs of --split")
    a = ap.parse_args(argv)
    miss = check(a.split)
    for m in miss:
        print(f"score_npz: missing {m}", file=sys.stderr)
    if a.check or miss:
        if not miss:
            print(f"score_npz: every scorer input of {a.split} exists")
        return 1 if miss else 0
    if not (a.npz and a.out and a.name):
        ap.error("--npz, --out and --name are required (or --check)")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.npz, allow_pickle=True)                     # select_npz.py stores tokens as an object array
    toks = [str(t) for t in z["tokens"]]
    traj = np.asarray(z["trajectory"], np.float32)
    assert traj.shape == (len(toks), 8, 3), f"{a.npz}: trajectory {traj.shape} for {len(toks)} tokens"
    log_of = token_logs(a.split)
    lack = [t for t in toks if t not in log_of]
    if lack:
        print(f"score_npz: {len(lack)} tokens of {a.npz} are not in the {a.split} token table (e.g. {lack[:3]})",
              file=sys.stderr)
        return 1
    rc = 0
    pm = {"npz": str(a.npz), **score_pdms(a.split, toks, traj, log_of, a.workers)}
    json.dump(pm, open(out / f"pdms_{a.name}.json", "w"), indent=1)
    print(f"{a.name} {a.split} PDMS: {pm['tokens']} tokens, scored {pm['scored']} (errors {pm['errors']})  "
          + " | ".join(f"{k.upper() if k != 'comfort' else 'C'} {pm[k]:6.2f}" for k in TERMS), flush=True)
    if pm["errors"]:
        print(f"score_npz: {pm['errors']} PDMS scoring errors, first: {pm['first_error']}", file=sys.stderr)
        rc = 1
    if not a.skip_epdms:
        ep = {"npz": str(a.npz), **score_epdms(a, toks, traj, log_of)}
        json.dump(ep, open(out / f"epdms_{a.name}.json", "w"), indent=1)
        if "error" in ep:
            print(f"score_npz: EPDMS {a.name}: {ep['error']}", file=sys.stderr)
            return 1
        print(f"{a.name} {a.split} EPDMS (v2.2 one-stage, {ep['valid']}/{ep['tokens']} valid): "
              f"EPDMS {ep['EPDMS']:6.2f} | " + " | ".join(f"{k} {ep[k]:6.2f}" for k in EP_TERMS), flush=True)
        if ep["valid"] < ep["tokens"]:
            print(f"WARNING: EPDMS {a.name}: {ep['tokens'] - ep['valid']} invalid rows (left out of the means)",
                  file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
