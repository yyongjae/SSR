#!/usr/bin/env python
"""Stage-T integration smoke (CPU): the REAL upstream products on ~20 tokens, end to end.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python tools/refiner/integrate_smoke.py [--n-train 12 --n-dev 8]

Unlike smoke_refiner.py (written before the draft bank existed; sample_bank stand-in drafts, its own labels/objects),
every input here is the product of the real pipeline:
  drafts   drafts/<split>/<token>.npz          make_draft_bank.py (13 drafts, family, params, valid)
  labels   scores/<split>.parquet or, while the bank job is still scoring, scores/<split>.shards/part-*.parquet
           (score_trajectories.py; the labels of the smoke tokens are copied to _integrate/scores/<split>.parquet)
  objects  objects/<split>/<token>.npz         build_future_objects.py (full build)
  sdf      sdf/navtrain/<token>.npz            build_sdf.py
  human    human/<split>.npz                   extract_human.py
  metric cache (centerline, official scoring), teacher bev_feature (cache_train_50x100, manifest-checked).
Tokens: train / dev split tokens with all of the above, no frame gap, at most one token per log (seeded).

Checks (report -> report/refiner_T/integration_smoke.json; data -> <data>/_integrate/)
  A  pack         data.pack_split of all six parts from the real sources; packed drafts == bank npz bytes,
                  packed labels == scores rows (float64).
  B  identity     "no modification returns the original official scores exactly":
                  B1 score_trajectories.score_token on the packed tau0 == the bank labels (nc dac ddc ep ttc comfort pdms
                     raw_progress, pdm_progress_eff), np.array_equal in float64;
                  B2 an UNTRAINED RefinerNet (both arms; zero-initialised correction heads) decodes every draft to the
                     tau0 bytes, and eval_refiner (predict -> score_trajectories CLI -> report) on an untrained run with
                     theta = 0 (every draft "modified") gives per-draft official scores == the bank labels exactly.
  C  surrogate    train_refiner.scene_from_batch + surrogate_terms_batch == the surrogate's reference builder
                  (scene_from_numpy on the per-token npz + centerline_from_metric_cache on the metric cache).
  D  gradients    one batch (all train tokens x 13 drafts), both arms, real surrogate: loss finite, every parameter
                  gradient finite, gradient norm per module group; then --opt-steps AdamW steps (lr 3e-4, wd 0.01,
                  clip 1.0) on that fixed batch: loss / correction terms per step, drafts changed.
  D2 overfit     (--overfit-steps N, or --only-overfit to add it to an existing report) arm none, gate weight 0,
                  AdamW lr --overfit-lr (3e-4) for N steps on the fixed train batch: does the whole chain (net -> decoder ->
                  surrogate) learn per-draft corrections that lower the collision / DAC surrogate?
  E  CLI train    train_refiner.py --arm T / none (CPU, --max-steps, real surrogate, log every step).
  F  CLI eval     eval_refiner.py predict / score / report on the dev tokens (theta = median p_g so both gate branches
                  occur, --sweep, --direct-check on every dev token: gated trajectories scored directly == mixed).
Resumable per stage only coarsely (pack and scoring shards resume; the rest re-runs, ~5-10 min in total).
CPU only, <= 2 worker processes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402

PY = "/home/external-user/miniconda3/envs/ssr/bin/python"
OUT = RD.DATA_ROOT / "_integrate"
REPORT = REPO / "report/refiner_T/integration_smoke.json"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
CHECK_COLS = ("nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms", "raw_progress")


def sh(cmd, log=None):
    t = time.time()
    print("$", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd], env=ENV, cwd=str(REPO), capture_output=True, text=True)
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(r.stdout + "\n---stderr---\n" + r.stderr)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r.stdout, time.time() - t


# ----------------------------------------------------------------------------------------------- sources
def labels_table(split: str) -> pd.DataFrame:
    """All scored bank rows of a split: the merged parquet if the bank job wrote it, else its shards."""
    p = RD.DATA_ROOT / "scores" / f"{split}.parquet"
    shards = sorted(glob.glob(str(RD.DATA_ROOT / "scores" / f"{split}.shards" / "part-*.parquet")))
    if p.is_file():
        df = pd.read_parquet(p)
        src = str(p)
        # tokens scored after the last merge are only in the shards
        if shards:
            extra = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
            extra = extra[~extra.token.isin(set(df.token))]
            if len(extra):
                df = pd.concat([df, extra], ignore_index=True)
                src += f" + {len(shards)} shards"
    else:
        df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
        src = f"{len(shards)} shards"
    df = df[df.k >= 0]
    err = df["error"].fillna("").astype(str) != "" if "error" in df else np.zeros(len(df), bool)
    bad = set(df.token[err])
    cnt = df[~df.token.isin(bad)].groupby("token").k.nunique()
    full = set(cnt.index[cnt == RD.K_DRAFTS])
    df = df[df.token.isin(full)].copy()
    df.attrs["source"] = src
    return df


def pick_tokens(split: str, n: int, seed: int, lab: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(RD.DATA_ROOT / "splits" / f"{split}.parquet")
    with np.load(RD.DATA_ROOT / "human" / f"{split}.npz") as z:
        gap = dict(zip(z["tokens"], z["frame_gap"]))
    tc = RD.TeacherCache.for_subset("navtrain")
    have_lab = set(lab.token)
    rng = np.random.default_rng(seed)
    out, logs = [], set()
    for i in rng.permutation(len(df)):
        r = df.iloc[i]
        if gap.get(r.token, True) or r.log in logs or r.token not in have_lab:
            continue
        need = [RD.DATA_ROOT / "drafts" / split / f"{r.token}.npz", RD.DATA_ROOT / "objects" / split / f"{r.token}.npz",
                RD.DATA_ROOT / "sdf" / "navtrain" / f"{r.token}.npz"]
        if not all(p.is_file() for p in need):
            continue
        if RD.locate_metric_cache(r.token, r.log) is None or not tc.has(r.token):
            continue
        out.append(i)
        logs.add(r.log)
        if len(out) >= n:
            break
    return df.iloc[sorted(out)].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- A: pack
def check_pack(split: str, tok: pd.DataFrame, lab: pd.DataFrame) -> Dict:
    P = RD.PackedSplit(split, OUT / "packed")
    rows = P.rows_with()
    res = dict(rows_all_parts=int(len(rows)), n=int(P.N), drafts_equal_npz=0, labels_equal_rows=0,
               obj_dropped=int(np.asarray(P.arrays["obj_dropped"]).sum()),
               cl_n_max=int(np.asarray(P.arrays["cl_n"]).max()), cl_truncated=int((np.asarray(P.arrays["cl_n"]) > RD.CL_MAX).sum()),
               obj_n_max=int(np.asarray(P.arrays["obj_n"]).max()))
    for r in rows:
        tk = str(P.index.token.values[r])
        with np.load(RD.DATA_ROOT / "drafts" / split / f"{tk}.npz") as z:
            same = np.array_equal(z["drafts"], P.arrays["drafts"][r]) and np.array_equal(z["valid"], P.arrays["draft_valid"][r])
        res["drafts_equal_npz"] += int(same)
        L = lab[lab.token == tk].sort_values("k")
        got = np.asarray(P.arrays["labels"][r])
        ok = all(np.array_equal(L[c].to_numpy(np.float64), got[:, RD.LBL[c]]) for c in RD.LABEL_COLS)
        ok &= float(P.arrays["pdm_progress_eff"][r]) == float(L.pdm_progress_eff.iloc[0])
        res["labels_equal_rows"] += int(ok)
    res["bank_fail_rate_valid"] = float(_bank_fail(P, rows))
    return res


def _bank_fail(P, rows) -> float:
    lab = np.asarray(P.arrays["labels"][rows])
    val = np.asarray(P.arrays["draft_valid"][rows])
    f = (lab[..., RD.LBL["nc"]] < 1) | (lab[..., RD.LBL["dac"]] < 1) | (lab[..., RD.LBL["ddc"]] < 1)
    return f[val].mean()


# ----------------------------------------------------------------------------------------------- B: identity
def check_score_identity(split: str) -> Dict:
    """B1: official scores of the packed original drafts (score_token) == the packed bank labels, exactly."""
    import score_trajectories as ST

    P = RD.PackedSplit(split, OUT / "packed")
    sim, scorer = ST.get_simulator_scorer()
    n, mism, worst, ppe_mism = 0, 0, 0.0, 0
    for r in P.rows_with():
        tk, lg = str(P.index.token.values[r]), str(P.index.log.values[r])
        mc = ST.load_metric_cache(ST.locate_metric_cache(tk, lg))
        got = ST.score_token(mc, np.asarray(P.arrays["drafts"][r]), sim, scorer)
        lab = np.asarray(P.arrays["labels"][r])
        for k, g in enumerate(got):
            n += 1
            v = np.array([g[c] for c in RD.LABEL_COLS], np.float64)
            if not np.array_equal(v, lab[k]):
                mism += 1
                worst = max(worst, float(np.nanmax(np.abs(v - lab[k]))))
            ppe_mism += int(g["pdm_progress_eff"] != float(P.arrays["pdm_progress_eff"][r]))
    return dict(n_traj=n, mismatches=mism, max_abs_diff=worst, pdm_progress_eff_mismatches=ppe_mism)


def untrained_net(arm: str, norm=None):
    import train_refiner as TR
    return TR.build_net(arm, 0, norm)


def check_decode_identity(split: str, teacher, norm) -> Dict:
    """B2 (in-process part): untrained net of each arm -> decode -> tau1 bytes == tau0 bytes; apply_gate keeps bytes."""
    import train_refiner as TR
    from navsim.agents.para_ssr.refiner.decoder import apply_gate

    P = RD.PackedSplit(split, OUT / "packed")
    rows = P.rows_with()
    out = {}
    for arm in ("T", "none"):
        net = untrained_net(arm, norm if arm == "T" else None).eval()
        b = RD.collate_tokens([RD.TokenDataset(P, rows, teacher if arm == "T" else None)[i] for i in range(len(rows))])
        with torch.no_grad():
            o = net(b["bev"], b["tau0"], b["v0"], b["a0"], b["eds"], b["cmd"])
            dec = TR.decode_batch(o, b, "A")
        T, K = b["tau0"].shape[:2]
        tau0 = b["tau0"].reshape(T * K, 8, 3)
        tau1 = dec["traj"].reshape(T * K, 8, 3)
        g_keep = apply_gate(tau0, tau1, torch.zeros(T * K, dtype=torch.bool))
        g_mod = apply_gate(tau0, tau1, torch.ones(T * K, dtype=torch.bool))
        out[arm] = dict(n_drafts=int(T * K), z_abs_max=float(o["z_lon"].abs().max()), w_abs_max=float(o["w_lat"].abs().max()),
                        tau1_bytes_equal_tau0=bool(tau1.numpy().tobytes() == tau0.numpy().tobytes()),
                        gate_keep_bytes_equal=bool(g_keep.numpy().tobytes() == tau0.numpy().tobytes()),
                        gate_modify_bytes_equal=bool(g_mod.numpy().tobytes() == tau0.numpy().tobytes()))
    return out


def write_untrained_run(arm: str, norm, cfg_extra: Dict) -> Path:
    """A run directory eval_refiner can load (config.json, norm.npz for arm T, ckpt_last.pt) holding an UNTRAINED net."""
    from navsim.agents.para_ssr.refiner.adapters import save_norm

    run = OUT / "runs" / f"untrained_{arm}"
    run.mkdir(parents=True, exist_ok=True)
    net = untrained_net(arm, norm if arm == "T" else None)
    if arm == "T":
        save_norm(run / "norm.npz", norm[0], norm[1], dict(note="integration smoke, untrained"))
    cfg = dict(arm=arm, seed=0, mode="A", amp=1, net_kw={}, surrogate="none (untrained)", fold=-1,
               teacher_sha_head=RD.TEACHER_SHA_HEAD, **cfg_extra)
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    torch.save({"model": net.state_dict(), "epoch": -1, "step": 0}, run / "ckpt_last.pt")
    return run


def eval_untrained(arm: str, norm) -> Dict:
    run = write_untrained_run(arm, norm, {})
    base = ["--run", run, "--split", "dev", "--packed-root", OUT / "packed", "--loader-workers", "0", "--ckpt", "last"]
    sh([PY, HERE / "eval_refiner.py", "all"] + base + ["--workers", "2", "--theta", "0.0"],
       OUT / "logs" / f"eval_untrained_{arm}.log")
    rows = pd.read_parquet(run / "eval_dev" / "report_rows.parquet")
    R = json.loads((run / "eval_dev" / "report.json").read_text())
    eq = {m: bool(np.array_equal(rows[f"{m}_tau1"].to_numpy(np.float64), rows[f"{m}_orig"].to_numpy(np.float64)))
          for m in ("nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms")}
    th0 = R["theta"]["0.0000"]
    return dict(n_drafts=int(len(rows)), tau1_scores_equal_bank_labels=eq, all_equal=all(eq.values()),
                theta0_modified=th0["modified"], theta0_pdms=th0["pdms"], no_correction_pdms=R["no_correction"]["pdms"],
                pdms_equal=th0["pdms"] == R["no_correction"]["pdms"], missing_tau1_scores=R["missing_tau1_scores"])


# ----------------------------------------------------------------------------------------------- C: surrogate adapter
def check_scene_equivalence(split: str) -> Dict:
    import train_refiner as TR
    from navsim.agents.para_ssr.refiner import surrogate as S
    from navsim.agents.para_ssr.refiner.decoder import decode
    from navsim.agents.para_ssr.refiner.gt_future import load_objects
    from navsim.agents.para_ssr.refiner.sdf import load_sdf

    P = RD.PackedSplit(split, OUT / "packed")
    rows = P.rows_with()
    b = RD.collate_tokens([RD.TokenDataset(P, rows, None)[i] for i in range(len(rows))])
    scenes = []
    for r in rows:
        tk, lg = str(P.index.token.values[r]), str(P.index.log.values[r])
        d = P.row(r)
        mc = RD.load_metric_cache(RD.locate_metric_cache(tk, lg))
        scenes.append(S.scene_from_numpy(objs=load_objects(RD.DATA_ROOT / "objects" / split / f"{tk}.npz"),
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
    out["unknown_rate"] = float(t_got["unknown"].float().mean())
    out["n_drafts"] = int(T * K)
    return out


# ----------------------------------------------------------------------------------------------- D: gradients
GROUPS = ("adapter", "enc", "station", "draft_mlp", "global_proj", "global_pos", "layers", "norm_out", "gate_head",
          "lon_head", "lat_head")


def check_grad_and_steps(split: str, teacher, norm, n_steps: int) -> Dict:
    import train_refiner as TR

    P = RD.PackedSplit(split, OUT / "packed")
    rows = P.rows_with()
    pi = TR.label_prevalence(P, rows)
    pos_weight = float(min((1 - pi) / pi, TR.POS_WEIGHT_CAP)) if 0 < pi < 1 else 1.0
    terms_fn, label = TR.resolve_surrogate("real")
    res = dict(surrogate=label, pi=pi, pos_weight=pos_weight)
    for arm in ("T", "none"):
        torch.manual_seed(0)
        net = untrained_net(arm, norm if arm == "T" else None).train()
        b = RD.collate_tokens([RD.TokenDataset(P, rows, teacher if arm == "T" else None)[i] for i in range(len(rows))])
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
        steps = []
        first = None
        t0 = time.time()
        for s in range(n_steps + 1):
            out = net(b["bev"], b["tau0"], b["v0"], b["a0"], b["eds"], b["cmd"])
            loss, st, dec = TR.compute_loss(out, b, terms_fn, TR.DEFAULT_W, pos_weight, "A", False, TR.DEFAULT_W_GATE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = {}
            fin = True
            nz = 0
            ntot = 0
            for g in GROUPS:
                ps = [p for n, p in net.named_parameters() if n.split(".")[0] == g and p.grad is not None]
                if ps:
                    fin &= all(bool(torch.isfinite(p.grad).all()) for p in ps)
                    gn[g] = float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in ps)))
            for p in net.parameters():
                ntot += 1
                nz += int(p.grad is not None and bool((p.grad != 0).any()))
            total = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
            T, K = b["tau0"].shape[:2]
            chg = (dec["traj"].detach().reshape(T, K, 8, 3) != b["tau0"]).any(-1).any(-1)
            rec = dict(step=s, loss=st["loss"], corr=st["corr"], gate_bce=st["gate_bce"],
                       **{k: st[f"t_{k}"] for k in TR.TERMS}, grad_finite=bool(fin), grad_norm=total,
                       params_with_nonzero_grad=f"{nz}/{ntot}", drafts_changed=int(chg.sum()),
                       max_abs_change_m=float((dec["traj"].detach().reshape(T, K, 8, 3) - b["tau0"])[..., :2].abs().max()))
            if s == 0:
                first = dict(rec, grad_norm_by_group=gn)
            steps.append(rec)
            if s < n_steps:
                opt.step()
        res[arm] = dict(first=first, steps=steps, sec=round(time.time() - t0, 1), n_drafts=int(b["tau0"].shape[0] * b["tau0"].shape[1]),
                        corr_first=steps[0]["corr"], corr_last=steps[-1]["corr"], all_finite=all(r["grad_finite"] for r in steps)
                        and all(np.isfinite(r["loss"]) for r in steps), param_counts=net.param_counts())
    res["step0_corr_equal_across_arms"] = res["T"]["steps"][0]["corr"] == res["none"]["steps"][0]["corr"]
    return res


def overfit_probe(split: str, n_steps: int, lr: float = 3e-4, every: int = 5) -> Dict:
    """D2: arm none, correction loss only (w_gate = 0), n_steps AdamW steps on ONE fixed batch (all packed tokens)."""
    import train_refiner as TR

    P = RD.PackedSplit(split, OUT / "packed")
    rows = P.rows_with()
    b = RD.collate_tokens([RD.TokenDataset(P, rows, None)[i] for i in range(len(rows))])
    terms_fn, label = TR.resolve_surrogate("real")
    torch.manual_seed(0)
    net = untrained_net("none").train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    T, K = b["tau0"].shape[:2]
    log, t0 = [], time.time()
    for s in range(n_steps + 1):
        out = net(b["bev"], b["tau0"], b["v0"], b["a0"], b["eds"], b["cmd"])
        loss, st, dec = TR.compute_loss(out, b, terms_fn, TR.DEFAULT_W, 1.0, "A", False, 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        if s % every == 0 or s == n_steps:
            ch = (dec["traj"].detach().reshape(T, K, 8, 3) - b["tau0"])[..., :2].norm(dim=-1).amax(-1)
            log.append(dict(step=s, corr=st["corr"], **{k: st[f"t_{k}"] for k in TR.TERMS}, grad_norm=gn,
                            finite=bool(np.isfinite(st["loss"]) and np.isfinite(gn)),
                            change_median_m=float(ch.median()), change_max_m=float(ch.max()),
                            sec=round(time.time() - t0, 1)))
            print(json.dumps(log[-1]), flush=True)
        if s < n_steps:
            opt.step()
    f, l = log[0], log[-1]
    return dict(split=split, arm="none", lr=lr, steps=n_steps, n_drafts=int(T * K), surrogate=label, log=log,
                corr_rel_change=(l["corr"] - f["corr"]) / f["corr"], col_rel_change=(l["col"] - f["col"]) / f["col"],
                dac_rel_change=(l["dac"] - f["dac"]) / f["dac"], all_finite=all(r["finite"] for r in log))


# ----------------------------------------------------------------------------------------------- E/F: CLIs
def cli_train(arm: str, max_steps: int) -> Dict:
    cmd = [PY, HERE / "train_refiner.py", "--arm", arm, "--fold", "-1", "--seed", "0", "--gpu", "-1",
           "--packed-root", OUT / "packed", "--runs", OUT / "runs", "--tokens-per-batch", "4", "--workers", "1",
           "--max-steps", str(max_steps), "--epochs", "2", "--n-norm", "16", "--log-every", "1", "--tag", "integ",
           "--restart"]
    _, dt = sh(cmd, OUT / "logs" / f"train_{arm}.log")
    run = OUT / "runs" / f"integ_{arm}_fold-1_seed0"
    cfg = json.loads((run / "config.json").read_text())
    log = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    st = [r for r in log if r["kind"] == "step"]
    return dict(run=str(run), sec=round(dt, 1), surrogate=cfg["surrogate"], pi=cfg["pi"], pos_weight=cfg["pos_weight"],
                n_train_tokens=cfg["n_train_tokens"], n_ival_tokens=cfg["n_ival_tokens"], param_counts=cfg["param_counts"],
                steps=[{k: r.get(k) for k in ("step", "loss", "corr", "gate_bce", "t_col", "t_dac", "t_prog", "t_cmf",
                                               "t_mod", "grad_norm")} for r in st],
                all_finite=all(np.isfinite(r["loss"]) and np.isfinite(r.get("grad_norm", 0.0)) for r in st),
                val=[r["val"] for r in log if r["kind"] == "epoch"])


def cli_eval(run: Path, arm: str) -> Dict:
    base = ["--run", run, "--split", "dev", "--packed-root", OUT / "packed", "--loader-workers", "0", "--ckpt", "last"]
    _, dt_p = sh([PY, HERE / "eval_refiner.py", "predict"] + base, OUT / "logs" / f"eval_predict_{arm}.log")
    P = np.load(run / "eval_dev" / "pred.npz")
    v = P["draft_valid"].astype(bool)
    theta = float(np.median(P["p_g"][v]))
    _, dt_s = sh([PY, HERE / "eval_refiner.py", "score"] + base + ["--workers", "2"], OUT / "logs" / f"eval_score_{arm}.log")
    _, dt_r = sh([PY, HERE / "eval_refiner.py", "report"] + base + ["--theta", f"{theta:.6f}", "--sweep",
                                                                       "--direct-check", str(len(P["tokens"]))],
                 OUT / "logs" / f"eval_report_{arm}.log")
    R = json.loads((run / "eval_dev" / "report.json").read_text())
    key = f"{theta:.4f}"
    th = R["theta"][key]
    return dict(sec_predict=round(dt_p, 1), sec_score=round(dt_s, 1), sec_report=round(dt_r, 1), theta=theta,
                n_tokens=R["n_tokens"], missing_tau1_scores=R["missing_tau1_scores"],
                tau1_max_abs_change=float(np.abs(P["tau1"] - P["tau0"]).max()),
                tau1_eq_tau0_frac=float((P["tau1"] == P["tau0"]).all((2, 3)).mean()),
                at_theta={k: v for k, v in th.items() if k != "by_family"}, no_correction=R["no_correction"],
                oracle_all={k: v for k, v in R["oracle_all"].items() if k != "by_family"},
                theta_at_budget=R.get("theta_at_budget"), direct_check=R.get("direct_check"))


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-dev", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opt-steps", type=int, default=8)
    ap.add_argument("--cli-steps", type=int, default=6)
    ap.add_argument("--fresh", action="store_true", help="delete <data>/_integrate first")
    ap.add_argument("--overfit-steps", type=int, default=0)
    ap.add_argument("--overfit-lr", type=float, default=3e-4)
    ap.add_argument("--only-overfit", action="store_true", help="run D2 only and add it to the existing report")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    if a.only_overfit:
        rep = json.loads(REPORT.read_text())
        rep["D2_overfit_train"] = overfit_probe("train", a.overfit_steps or 60, a.overfit_lr)
        REPORT.write_text(json.dumps(rep, indent=1, default=float))
        print("updated", REPORT, flush=True)
        return
    if a.fresh and OUT.exists():
        import shutil
        shutil.rmtree(OUT / "packed", ignore_errors=True)
        shutil.rmtree(OUT / "runs", ignore_errors=True)
        shutil.rmtree(OUT / "splits", ignore_errors=True)
        shutil.rmtree(OUT / "scores", ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rep = dict(started=time.strftime("%Y-%m-%dT%H:%M:%S"), out=str(OUT), a_max=RD.A_MAX, cl_max=RD.CL_MAX,
               pack_version=RD.PACK_VERSION)
    tokens = {}
    t_all = time.time()
    for split, n in (("train", a.n_train), ("dev", a.n_dev)):
        sp = OUT / "splits" / f"{split}.parquet"
        lab_all = labels_table(split)
        rep[f"labels_source_{split}"] = lab_all.attrs["source"]
        if sp.exists():
            tok = pd.read_parquet(sp)
        else:
            tok = pick_tokens(split, n, a.seed, lab_all)
            sp.parent.mkdir(parents=True, exist_ok=True)
            tok.to_parquet(sp, index=False)
        tokens[split] = tok
        lab = lab_all[lab_all.token.isin(set(tok.token))].sort_values(["token", "k"]).reset_index(drop=True)
        lp = OUT / "scores" / f"{split}.parquet"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lab.to_parquet(lp, index=False)
        rep[f"{split}_tokens"] = tok.token.tolist()
        rep[f"{split}_logs"] = int(tok.log.nunique())
        src = RD.Sources(split, root=RD.DATA_ROOT, scores=lp)
        t = time.time()
        rep[f"pack_{split}"] = RD.pack_split(split, tok, OUT / "packed", sources=src, workers=2, chunk=4)
        rep[f"sec_pack_{split}"] = round(time.time() - t, 1)
        rep[f"A_pack_{split}"] = check_pack(split, tok, lab)
        print(json.dumps({f"A_pack_{split}": rep[f"A_pack_{split}"]}), flush=True)
    # ---- B1: bank labels reproduce from the packed original drafts
    for split in ("train", "dev"):
        t = time.time()
        rep[f"B1_score_identity_{split}"] = dict(check_score_identity(split), sec=round(time.time() - t, 1))
        print(json.dumps({f"B1_{split}": rep[f"B1_score_identity_{split}"]}), flush=True)
    # ---- teacher norm for the in-process arm-T nets (train tokens)
    teacher = RD.TeacherCache.for_subset("navtrain")
    mean, std, ninfo = RD.compute_teacher_norm(teacher, tokens["train"].token.tolist(), 64, seed=0)
    norm = (mean, std)
    # ---- B2: untrained nets = identity (in-process + through eval_refiner and the scoring CLI)
    rep["B2_decode_identity_dev"] = check_decode_identity("dev", teacher, norm)
    for arm in ("T", "none"):
        rep[f"B2_eval_untrained_{arm}"] = eval_untrained(arm, norm)
        print(json.dumps({f"B2_{arm}": rep[f"B2_eval_untrained_{arm}"]}), flush=True)
    # ---- C: surrogate adapter
    rep["C_surrogate_adapter_dev"] = check_scene_equivalence("dev")
    rep["C_surrogate_adapter_train"] = check_scene_equivalence("train")
    print(json.dumps({"C": [rep["C_surrogate_adapter_dev"], rep["C_surrogate_adapter_train"]]}), flush=True)
    # ---- D: gradients + optimisation steps on a fixed batch
    rep["D_grad_steps_train"] = check_grad_and_steps("train", teacher, norm, a.opt_steps)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("steps", "first")} if isinstance(v, dict) else v
                      for k, v in rep["D_grad_steps_train"].items()}), flush=True)
    if a.overfit_steps:
        rep["D2_overfit_train"] = overfit_probe("train", a.overfit_steps, a.overfit_lr)
    # ---- E / F: CLI train + eval, both arms
    for arm in ("T", "none"):
        rep[f"E_train_{arm}"] = cli_train(arm, a.cli_steps)
        rep[f"F_eval_{arm}"] = cli_eval(Path(rep[f"E_train_{arm}"]["run"]), arm)
        print(json.dumps({f"F_eval_{arm}": {k: rep[f'F_eval_{arm}'][k] for k in ('theta', 'direct_check', 'at_theta')}},
                         default=float), flush=True)
    pc = {arm: rep[f"E_train_{arm}"]["param_counts"] for arm in ("T", "none")}
    rep["param_parity"] = dict(trunk_equal=pc["T"]["trunk"] == pc["none"]["trunk"], T=pc["T"], none=pc["none"])
    rep["sec_total"] = round(time.time() - t_all, 1)
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    REPORT.write_text(json.dumps(rep, indent=1, default=float))
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
