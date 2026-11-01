#!/usr/bin/env python
"""End-to-end CPU smoke test of the stage-T refiner pipeline on REAL inputs (a few train + dev tokens).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python tools/refiner/smoke_refiner.py [--n-train 12 --n-dev 8]

Steps (everything written under <data>/_smoke/, resumable; report -> report/refiner_T/refiner_smoke.json)
  1. tokens: train / dev split tokens that already have a metric cache, a drivable SDF (sdf/navtrain), a teacher npz
     and a human row without frame gap, spread over several logs (seeded).
  2. drafts: decoder.sample_bank per token (human/<split>.npz: traj, path, n_reg, v0, a0) -> _smoke/drafts/<split>/.
     (stand-in for make_draft_bank.py, which did not exist when this was written)
  3. labels: score_trajectories.py (2 workers) -> _smoke/scores/<split>.parquet.
  4. objects: build_future_objects.py (2 workers) -> _smoke/objects/<split>/.
  5. pack: data.pack_split (human and SDF read from the real data root) -> _smoke/packed/<split>/.
  5b. surrogate adapter check on the packed dev tokens: train_refiner.scene_from_batch + surrogate_terms_batch vs the
     surrogate's reference builder (scene_from_numpy on the per-token objects / SDF npz + centerline_from_metric_cache
     on the metric cache itself) for decoded random corrections -> max |term difference|.
  6. train: train_refiner.py --arm T and --arm none, 2 steps each (CPU, --surrogate real), parameter counts recorded.
  7. eval: eval_refiner.py predict / score / report on the dev tokens, theta = median p_g (so that both gate branches
     occur), --direct-check on every dev token (gated trajectories scored directly == mixed scores).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402

PY = "/home/external-user/miniconda3/envs/ssr/bin/python"
SMOKE = RD.DATA_ROOT / "_smoke"
LOGS = Path("/home/external-user/yongjae/SSR/data/dataset/navsim_logs/trainval")
REPORT = REPO / "report/refiner_T/refiner_smoke.json"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")


def sh(cmd, log=None):
    t = time.time()
    print("$", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd], env=ENV, cwd=str(REPO), capture_output=True, text=True)
    if log:
        Path(log).write_text(r.stdout + "\n---stderr---\n" + r.stderr)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r.stdout, time.time() - t


def pick_tokens(split: str, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(RD.DATA_ROOT / "splits" / f"{split}.parquet")
    with np.load(RD.DATA_ROOT / "human" / f"{split}.npz") as z:
        gap = dict(zip(z["tokens"], z["frame_gap"]))
    tc = RD.TeacherCache.for_subset("navtrain")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    out, logs = [], {}
    for i in order:
        r = df.iloc[i]
        if gap.get(r.token, True) or logs.get(r.log, 0) >= 2:
            continue
        if not (RD.DATA_ROOT / "sdf" / "navtrain" / f"{r.token}.npz").is_file():
            continue
        if RD.locate_metric_cache(r.token, r.log) is None or not tc.has(r.token):
            continue
        out.append(i)
        logs[r.log] = logs.get(r.log, 0) + 1
        if len(out) >= n:
            break
    return df.iloc[sorted(out)].reset_index(drop=True)


def make_drafts(split: str, tok: pd.DataFrame) -> float:
    from navsim.agents.para_ssr.refiner.decoder import sample_bank

    t = time.time()
    d = SMOKE / "drafts" / split
    d.mkdir(parents=True, exist_ok=True)
    with np.load(RD.DATA_ROOT / "human" / f"{split}.npz") as z:
        pos = {tk: i for i, tk in enumerate(z["tokens"])}
        H = {k: z[k] for k in ("traj", "path", "n_reg", "v0", "a0")}
    for tk in tok.token:
        p = d / f"{tk}.npz"
        if p.exists():
            continue
        i = pos[tk]
        b = sample_bank(H["traj"][i], tk, path_long=H["path"][i], n_valid=int(H["n_reg"][i]), v0=float(H["v0"][i]),
                        a0=float(H["a0"][i]))
        np.savez(p, drafts=b["drafts"], family=b["family"], params=b["params"], z_lon=b["z_lon"], w_lat=b["w_lat"],
                 valid=b["valid"])
    return time.time() - t


def check_scene_equivalence(split: str, od: Path) -> dict:
    import torch
    import train_refiner as TR
    from navsim.agents.para_ssr.refiner import surrogate as S
    from navsim.agents.para_ssr.refiner.decoder import decode
    from navsim.agents.para_ssr.refiner.gt_future import load_objects
    from navsim.agents.para_ssr.refiner.sdf import load_sdf

    P = RD.PackedSplit(split, SMOKE / "packed")
    rows = P.rows_with()
    b = RD.collate_tokens([RD.TokenDataset(P, rows, None)[i] for i in range(len(rows))])
    scenes = []
    for r in rows:
        tk, lg = str(P.index.token.values[r]), str(P.index.log.values[r])
        d = P.row(r)
        mc = RD.load_metric_cache(RD.locate_metric_cache(tk, lg))
        scenes.append(S.scene_from_numpy(objs=load_objects(od / f"{tk}.npz"),
                                         sdf=load_sdf(RD.DATA_ROOT / "sdf" / "navtrain" / f"{tk}.npz"),
                                         centerline=S.centerline_from_metric_cache(mc), human_traj=d["human_traj"],
                                         p_pdm=d["pdm_progress_eff"], v0=d["v0"], a0=d["a0"]))
    ref = S.collate_scenes(scenes)
    T, K = b["tau0"].shape[:2]
    g = torch.Generator().manual_seed(0)
    z = torch.randn(T * K, 6, generator=g, dtype=torch.float64)
    w = torch.randn(T * K, 6, generator=g, dtype=torch.float64) * 0.5
    tau = b["tau0"].reshape(-1, 8, 3).double()
    dec = decode(tau, z, w, v0=b["v0"].double().repeat_interleave(K))
    t_ref = S.surrogate_terms(dec, tau, ref, index=torch.arange(T).repeat_interleave(K))
    t_got = TR.surrogate_terms_batch(dec, b, T, K)
    out = {k: float((t_got[k].double() - t_ref[k].double()).abs().max()) for k in TR.TERMS + ("P1", "P0")}
    out["unknown_equal"] = bool(torch.equal(t_got["unknown"], t_ref["unknown"]))
    out["mean_terms"] = {k: float(t_got[k].double().mean()) for k in TR.TERMS}
    out["n_drafts"] = int(T * K)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-dev", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    SMOKE.mkdir(parents=True, exist_ok=True)
    rep = dict(started=time.strftime("%Y-%m-%dT%H:%M:%S"), smoke_root=str(SMOKE))
    tokens = {}
    for split, n in (("train", a.n_train), ("dev", a.n_dev)):
        sp = SMOKE / "splits" / f"{split}.parquet"
        if sp.exists():
            tok = pd.read_parquet(sp)
        else:
            tok = pick_tokens(split, n, a.seed)
            sp.parent.mkdir(parents=True, exist_ok=True)
            tok.to_parquet(sp, index=False)
        tokens[split] = tok
        rep[f"{split}_tokens"] = tok.token.tolist()
        rep[f"{split}_logs"] = int(tok.log.nunique())
        rep[f"sec_drafts_{split}"] = round(make_drafts(split, tok), 1)
        sc = SMOKE / "scores" / f"{split}.parquet"
        if not sc.exists():
            _, dt = sh([PY, HERE / "score_trajectories.py", "--drafts", SMOKE / "drafts" / split, "--tokens", sp,
                        "--out", sc, "--workers", "2"], SMOKE / f"score_{split}.log")
            rep[f"sec_score_{split}"] = round(dt, 1)
        s = pd.read_parquet(sc)
        rep[f"bank_fail_rate_{split}"] = float(((s.nc < 1) | (s.dac < 1) | (s.ddc < 1)).mean())
        od = SMOKE / "objects" / split
        if not od.exists() or len(list(od.glob("*.npz"))) < len(tok):
            _, dt = sh([PY, HERE / "build_future_objects.py", "--tokens", sp, "--logs", LOGS, "--out", od,
                        "--workers", "2"], SMOKE / f"objects_{split}.log")
            rep[f"sec_objects_{split}"] = round(dt, 1)
        src = RD.Sources(split, root=SMOKE, human=RD.DATA_ROOT / "human" / f"{split}.npz",
                         sdf=RD.DATA_ROOT / "sdf" / "navtrain", objects=[od])
        t = time.time()
        rep[f"pack_{split}"] = RD.pack_split(split, tok, SMOKE / "packed", sources=src, workers=2, chunk=4)
        rep[f"sec_pack_{split}"] = round(time.time() - t, 1)
        P = RD.PackedSplit(split, SMOKE / "packed")
        rep[f"packed_rows_ok_{split}"] = int(len(P.rows_with()))
        if split == "dev":
            rep["surrogate_adapter_check_dev"] = check_scene_equivalence(split, od)
    # ---- train both arms, 2 steps
    runs = {}
    for arm in ("T", "none"):
        cmd = [PY, HERE / "train_refiner.py", "--arm", arm, "--fold", "-1", "--seed", "0", "--gpu", "-1",
               "--packed-root", SMOKE / "packed", "--runs", SMOKE / "runs", "--tokens-per-batch", "4", "--workers", "1",
               "--max-steps", "2", "--epochs", "1", "--n-norm", "8", "--log-every", "1", "--tag", "smoke", "--restart",
               "--surrogate", "real"]
        out, dt = sh(cmd, SMOKE / f"train_{arm}.log")
        run = SMOKE / "runs" / f"smoke_{arm}_fold-1_seed0"
        cfg = json.loads((run / "config.json").read_text())
        log = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
        runs[arm] = run
        rep[f"train_{arm}"] = dict(sec=round(dt, 1), param_counts=cfg["param_counts"], surrogate=cfg["surrogate"],
                                   pi=cfg["pi"], pos_weight=cfg["pos_weight"], n_train_tokens=cfg["n_train_tokens"],
                                   n_ival_tokens=cfg["n_ival_tokens"],
                                   steps=[{k: r.get(k) for k in ("step", "loss", "gate_bce", "corr", "t_col", "t_dac",
                                                                  "t_prog", "t_cmf", "t_mod", "t_unknown", "grad_norm",
                                                                  "sec")}
                                          for r in log if r["kind"] == "step"],
                                   val=[r["val"] for r in log if r["kind"] == "epoch"])
    pc = {arm: rep[f"train_{arm}"]["param_counts"] for arm in runs}
    rep["param_parity"] = dict(trunk_equal=pc["T"]["trunk"] == pc["none"]["trunk"], **{f"{k}_{arm}": v for arm in pc
                                                                                         for k, v in pc[arm].items()})
    # ---- eval on dev
    for arm, run in runs.items():
        base = ["--run", run, "--split", "dev", "--packed-root", SMOKE / "packed", "--loader-workers", "0"]
        _, dt_p = sh([PY, HERE / "eval_refiner.py", "predict"] + base, SMOKE / f"eval_predict_{arm}.log")
        outd = run / "eval_dev"
        P = np.load(outd / "pred.npz")
        v = P["draft_valid"].astype(bool)
        theta = float(np.median(P["p_g"][v]))
        _, dt_s = sh([PY, HERE / "eval_refiner.py", "score"] + base + ["--workers", "2"], SMOKE / f"eval_score_{arm}.log")
        _, dt_r = sh([PY, HERE / "eval_refiner.py", "report"] + base + ["--theta", f"{theta:.6f}", "--sweep",
                                                                           "--direct-check", str(len(P["tokens"]))],
                     SMOKE / f"eval_report_{arm}.log")
        R = json.loads((outd / "report.json").read_text())
        th = R["theta"][f"{theta:.4f}"] if f"{theta:.4f}" in R["theta"] else R["theta"][sorted(R["theta"])[0]]
        rep[f"eval_{arm}"] = dict(sec_predict=round(dt_p, 1), sec_score=round(dt_s, 1), sec_report=round(dt_r, 1),
                                  theta=theta, n_tokens=R["n_tokens"], missing_tau1_scores=R["missing_tau1_scores"],
                                  tau1_max_abs_change=float(np.abs(P["tau1"] - P["tau0"]).max()),
                                  tau1_eq_tau0_frac=float((P["tau1"] == P["tau0"]).all((2, 3)).mean()),
                                  at_theta={k: v for k, v in th.items() if k != "by_family"},
                                  no_correction_pdms=R["no_correction"]["pdms"], oracle_all=R["oracle_all"],
                                  theta_at_budget=R.get("theta_at_budget"), direct_check=R.get("direct_check"))
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("param_parity", "eval_T", "eval_none")}, indent=1, default=float))


if __name__ == "__main__":
    main()
