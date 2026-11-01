#!/usr/bin/env python
"""Apply a trained stage-T refiner to a packed split and score the result officially (IMPL_SPEC §3.8, §4).

  python tools/refiner/eval_refiner.py all --run <runs>/<run> --split dev --gpu 3 --workers 4 --theta 0.5
  (sub-commands: predict | score | report | all)

Inference rule (PRESTATED_DECISION_RULE.txt): final_k = decode(tau0_k, z_k, w_k) if p_g,k >= theta else the ORIGINAL
tau0_k bytes (decoder.apply_gate).  Implementation:
  predict : net + decoder (mode of the run) on every draft of the selected rows -> <out>/pred.npz (tokens, rows,
            tau0, tau1, p_g, z_lon, w_lat, draft_valid, family, alpha, beta) and <out>/refined.npz (tokens, drafts =
            tau1 [N, K, 8, 3]) + <out>/tokens.parquet (token, log).
  score   : tools/refiner/score_trajectories.py on refined.npz (<= 4 workers, resumable) -> <out>/scores_tau1.parquet.
  report  : per draft, the official scores of the final trajectory = scores(tau1) where p_g >= theta else the bank
            labels of tau0 (scores/<split>.parquet via the pack).  This equals scoring the gated trajectories directly
            because score_trajectories is per-trajectory exact (batch == single pdm_score, bitwise) and the gate
            returns the original tau0 bytes; --direct-check N re-scores the gated trajectories of N tokens directly and
            asserts equality.  Writes <out>/report_rows.parquet (one row per draft) and <out>/report.json.
Teacher-shuffle control (PRESTATED_DECISION_RULE.txt AMENDMENT 5, descriptive): --shuffle-teacher-seed S (arm T only)
           remaps the BEV lookup by a fixed derangement of the evaluated tokens (seeded; each token reads ANOTHER
           token's bev_feature); tokens, drafts, rows and everything else are unchanged.  Recorded in predict_meta.json.
Selection: --split dev (all packed dev rows), or --split train --fold k (out-of-fold rows of fold k; default = the
           run's own held-out fold).  Only valid drafts (draft_valid) of tokens with labels are counted.
RUN 4 (PRESTATED_DECISION_RULE AMENDMENT 6; all off by default, runs 1-3 unchanged):
  arms M / TM: the BEV comes from data.arm_teachers (ReSMap for M; BEVFusion + ReSMap concatenated for TM), the caches
           of the EVALUATED split's subset (navtrain for train / dev, navtest for navtest); sha heads checked against
           the run's config.json (teacher_sha_head for the det cache, resmap_sha_head for the map cache).
  token subset: --split train -> the run's config.json 'token_subset' (sha256 re-checked; an explicit --token-subset
           must be the same file content); --split dev -> --token-subset, default the run's 'dev_token_subset' if
           recorded; --split navtest -> never subset (--token-subset refused).  Arms M / TM refuse any selection with a
           token outside the ReSMap cache.  Recorded in predict_meta.json 'token_subset'.
  --shuffle-teacher-seed S [--shuffle-which det|map|both]: default = the arm's teacher(s) (T det, M map, TM both);
           'both' uses ONE derangement for the two branches (each token reads another token's full input).
  --drop-branch det|map (TM only): that branch's NORMALISED input is 0 (= the per-channel training-mean feature).
  --eval-name NAME: output directory <run>/NAME (instead of the default eval_<split>[_fold<k>]); shuffle / branch-drop
           predictions must go to a non-default directory (--eval-name or --out), so eval_dev / eval_navtest /
           eval_train_fold<k> are never overwritten by a control.  All recorded in predict_meta.json.
Metrics (report.json, per theta; valid drafts; 'points' = x100):
  pdms / pdms_orig / d_pdms_points, ep_loss_points = 100 * mean(ep_orig - ep_final), fail counts nc/dac/ddc/ttc/comfort
  (< 1) final vs orig, fail_any (nc|dac|ddc|ttc), fixed (orig fail_any -> final pass), new_fail (orig passes nc, dac,
  ddc, ttc, comfort -> final fails any of them), modified, unnecessary_mod (modified while orig passes nc, dac, ttc),
  harm (final pdms < orig pdms), per family at --theta; theta_at_budget = smallest theta of the sweep with
  ep_loss_points <= --budget-ep (pre-stated 0.5 points).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner.decoder import FAMILY_NAME, apply_gate  # noqa: E402
import train_refiner as TR  # noqa: E402

METRICS = ("nc", "dac", "ddc", "ep", "ttc", "comfort", "pdms")
PYTHON = "/home/external-user/miniconda3/envs/ssr/bin/python"
SWEEP = tuple(np.round(np.linspace(0.0, 1.0, 101), 4))


def default_eval_name(split: str, fold: Optional[int]) -> str:
    return f"eval_{split}" + (f"_fold{fold}" if fold is not None and fold >= 0 else "")


def out_dir_for(a, cfg) -> Path:
    if a.out:
        return Path(a.out)
    if getattr(a, "eval_name", None):
        return Path(a.run) / a.eval_name
    return Path(a.run) / default_eval_name(a.split, a.fold)


def select_eval_rows(packed: RD.PackedSplit, split: str, fold: Optional[int], limit: Optional[int] = None,
                     subset_tokens=None):
    parts = ("human", "drafts", "labels")
    rows = packed.select(parts, folds=[fold]) if (split == "train" and fold is not None and fold >= 0) else packed.select(parts)
    if subset_tokens is not None:
        rows = RD.restrict_rows(packed, rows, subset_tokens)
    return rows[:limit] if limit else rows


# ----------------------------------------------------------------------------------------------- teacher shuffle
def derangement(tokens: Sequence[str], seed: int) -> Dict[str, str]:
    """token -> source token of its BEV: a seeded random cyclic order (one n-cycle), so no token maps to itself."""
    tokens = [str(t) for t in tokens]
    if len(set(tokens)) != len(tokens) or len(tokens) < 2:
        raise ValueError("derangement needs >= 2 distinct tokens")
    order = np.random.default_rng(int(seed)).permutation(len(tokens))
    return {tokens[order[i]]: tokens[order[(i + 1) % len(order)]] for i in range(len(order))}


class ShuffledTeacher:
    """TeacherCache view whose load_bev(token) returns the bev_feature of mapping[token] (picklable)."""

    def __init__(self, base, mapping: Dict[str, str]):
        self.base, self.mapping = base, dict(mapping)
        self.root, self.sha_head = base.root, base.sha_head

    def load_bev(self, token: str, s_grid: bool = True) -> np.ndarray:
        return self.base.load_bev(self.mapping[token], s_grid=s_grid)


ARM_TEACHERS = {"T": ("det",), "M": ("map",), "TM": ("det", "map"), "none": ()}
SHUFFLE_WHICH = {"det": ("det",), "map": ("map",), "both": ("det", "map")}


def check_run4_eval_args(a, cfg) -> None:
    """Refuse run-4 evaluation options that do not apply (before anything is written)."""
    arm = cfg["arm"]
    if getattr(a, "shuffle_which", None) is not None and a.shuffle_teacher_seed is None:
        raise SystemExit("--shuffle-which needs --shuffle-teacher-seed")
    if a.shuffle_teacher_seed is not None:
        if not ARM_TEACHERS[arm]:
            raise SystemExit(f"--shuffle-teacher-seed needs a teacher arm (T, M, TM), not {arm!r}")
        which = getattr(a, "shuffle_which", None) or {"T": "det", "M": "map", "TM": "both"}[arm]
        if not set(SHUFFLE_WHICH[which]) <= set(ARM_TEACHERS[arm]):
            raise SystemExit(f"--shuffle-which {which} does not apply to arm {arm} (teachers {ARM_TEACHERS[arm]})")
    if getattr(a, "drop_branch", None) is not None and arm != "TM":
        raise SystemExit(f"--drop-branch is for arm TM only (run arm {arm})")
    if a.split == "navtest" and getattr(a, "token_subset", None):
        raise SystemExit("navtest is never subset (all evaluable tokens): --token-subset refused for --split navtest")
    if getattr(a, "out", None) and getattr(a, "eval_name", None):
        raise SystemExit("--out and --eval-name are mutually exclusive")
    control = a.shuffle_teacher_seed is not None or getattr(a, "drop_branch", None) is not None
    if control:
        name = Path(a.out).name if a.out else getattr(a, "eval_name", None)
        if not name:
            raise SystemExit("a shuffle / branch-drop prediction needs --eval-name (or --out): it must not write into "
                             f"{default_eval_name(a.split, a.fold)}")
        defaults = {default_eval_name(a.split, a.fold), "eval_dev", "eval_navtest", "eval_train"} | \
            {f"eval_train_fold{k}" for k in range(5)}
        if name in defaults:
            raise SystemExit(f"refused: control output name {name!r} is a default evaluation directory")


def resolve_token_subset(a, cfg):
    """-> (subset tokens | None, info | None) for the evaluated split (module docstring, RUN 4)."""
    if a.split == "navtest":
        return None, None
    rec = cfg.get("token_subset") if a.split == "train" else cfg.get("dev_token_subset")
    path = getattr(a, "token_subset", None)
    if a.split == "train":
        if rec is None and path is None:
            return None, None
        if rec is None:
            raise SystemExit("--split train --token-subset: the run was trained without a token subset (OOF rows would "
                             "then not be the run's own held-out rows)")
        tok, info = RD.load_token_subset(path or rec["path"])
        if info["sha256"] != rec["sha256"]:
            raise SystemExit(f"train token subset {info['path']} sha256 {info['sha256'][:16]} != the run's "
                             f"{rec['sha256'][:16]} ({rec['path']})")
        return tok, dict(info, source="run config.json token_subset" + (" (+ --token-subset, same sha256)" if path else ""))
    if path is None and rec is None:
        return None, None
    tok, info = RD.load_token_subset(path or rec["path"])
    if path is None and info["sha256"] != rec["sha256"]:
        raise SystemExit(f"dev token subset {info['path']} changed since training (sha256 {info['sha256'][:16]} != "
                         f"{rec['sha256'][:16]})")
    return tok, dict(info, source="--token-subset" if path else "run config.json dev_token_subset")


def run4_teacher(a, cfg, rows, packed):
    """Loader teacher for arms M / TM (+ shuffle) -> (teacher, shuffle_info, mapping | None)."""
    arm = cfg["arm"]
    _, det, mp = RD.arm_teachers(arm, a.split, a.teacher_root, getattr(a, "resmap_root", None))
    if det is not None and det.sha_head != cfg.get("teacher_sha_head", RD.TEACHER_SHA_HEAD):
        raise RuntimeError(f"teacher sha head {det.sha_head} != the run's {cfg.get('teacher_sha_head')}")
    if mp is not None and mp.sha_head != cfg.get("resmap_sha_head"):
        raise RuntimeError(f"ReSMap sha head {mp.sha_head} != the run's {cfg.get('resmap_sha_head')}")
    toks_eval = packed.index.token.values[np.asarray(rows, np.int64)]
    if mp is not None:
        miss = [t for t in toks_eval if not mp.has(str(t))]
        if miss:
            raise SystemExit(f"arm {arm}: {len(miss)} of {len(toks_eval)} evaluated tokens are not in the ReSMap cache "
                             f"{mp.root} (e.g. {miss[:3]}); pass a covered --token-subset")
    shuffle_info, mapping = None, None
    if a.shuffle_teacher_seed is not None:
        which = getattr(a, "shuffle_which", None) or {"M": "map", "TM": "both"}[arm]
        mapping = derangement(toks_eval, a.shuffle_teacher_seed)
        log_of = dict(zip(map(str, toks_eval), packed.index.log.values[np.asarray(rows, np.int64)]))
        if "det" in SHUFFLE_WHICH[which]:
            det = ShuffledTeacher(det, mapping)
        if "map" in SHUFFLE_WHICH[which]:
            mp = ShuffledTeacher(mp, mapping)
        shuffle_info = dict(seed=int(a.shuffle_teacher_seed), n=len(mapping), kind="seeded cyclic derangement",
                            which=which, fixed_points=int(sum(k == v for k, v in mapping.items())),
                            same_log_frac=float(np.mean([log_of[t] == log_of[mapping[t]] for t in mapping])))
    teacher = mp if arm == "M" else RD.ConcatTeacher(det, mp)
    return teacher, shuffle_info, mapping


# ----------------------------------------------------------------------------------------------- predict
@torch.no_grad()
def predict(a) -> Path:
    dev = TR._device(a.gpu)
    net, cfg = TR.load_run_model(Path(a.run), a.ckpt, dev)
    check_run4_eval_args(a, cfg)
    sub_tok, sub_info = resolve_token_subset(a, cfg)
    out = out_dir_for(a, cfg)
    out.mkdir(parents=True, exist_ok=True)
    packed = RD.PackedSplit(a.split, a.packed_root)
    rows = select_eval_rows(packed, a.split, a.fold, a.limit, subset_tokens=sub_tok)
    teacher = None
    shuffle_info = None
    run4 = {}
    if sub_info is not None:
        run4["token_subset"] = dict(sub_info, n_eval_tokens=int(len(rows)))
    if getattr(a, "drop_branch", None) is not None:
        net.adapter.set_drop_branch(a.drop_branch)
        run4["drop_branch"] = a.drop_branch
    if cfg["arm"] in ("M", "TM"):
        teacher, shuffle_info, mapping = run4_teacher(a, cfg, rows, packed)
        if mapping is not None:
            (out / "teacher_shuffle_map.json").write_text(json.dumps(mapping))
    if cfg["arm"] == "T":
        # the cache of the EVALUATED split (navtrain for train/dev, navtest -> cache_val_50x100), manifest-checked
        teacher = RD.TeacherCache(a.teacher_root) if a.teacher_root else RD.TeacherCache.for_subset(RD.SPLIT_SUBSET[a.split])
        if teacher.sha_head != cfg.get("teacher_sha_head", RD.TEACHER_SHA_HEAD):
            raise RuntimeError(f"teacher sha head {teacher.sha_head} != the run's {cfg.get('teacher_sha_head')}")
    if a.shuffle_teacher_seed is not None and cfg["arm"] in ("T", "none"):
        if teacher is None:
            raise SystemExit("--shuffle-teacher-seed needs an arm-T run")
        toks_eval = packed.index.token.values[np.asarray(rows, np.int64)]
        mapping = derangement(toks_eval, a.shuffle_teacher_seed)
        log_of = dict(zip(map(str, toks_eval), packed.index.log.values[np.asarray(rows, np.int64)]))
        teacher = ShuffledTeacher(teacher, mapping)
        shuffle_info = dict(seed=int(a.shuffle_teacher_seed), n=len(mapping), kind="seeded cyclic derangement",
                            fixed_points=int(sum(k == v for k, v in mapping.items())),
                            same_log_frac=float(np.mean([log_of[t] == log_of[mapping[t]] for t in mapping])))
        if getattr(a, "shuffle_which", None) is not None:
            shuffle_info["which"] = a.shuffle_which
        (out / "teacher_shuffle_map.json").write_text(json.dumps(mapping))
    loader = RD.make_loader(packed, rows, teacher, a.tokens_per_batch, shuffle=False, workers=a.loader_workers,
                            drop_last=False)
    mode = cfg.get("mode", "A")
    use_amp = bool(cfg.get("amp", 1)) and dev.type == "cuda"
    rec = {k: [] for k in ("rows", "tau0", "tau1", "p_g", "z_lon", "w_lat", "draft_valid", "family", "alpha", "beta")}
    toks: List[str] = []
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
        rec["draft_valid"].append(batch["draft_valid"].cpu().numpy())
        rec["family"].append(batch["family"].cpu().numpy())
        rec["alpha"].append(dec["flags"]["alpha"].reshape(T, K).cpu().numpy())
        rec["beta"].append(dec["flags"]["beta"].reshape(T, K).cpu().numpy())
        toks += batch["tokens"]
        if (i + 1) % 100 == 0:
            print(f"[predict] {len(toks)}/{len(rows)} tokens {time.time() - t0:.0f}s", flush=True)
    R = {k: np.concatenate(v) for k, v in rec.items()}
    tokens = np.array(toks)
    np.savez(out / "pred.npz", tokens=tokens, **R)
    np.savez(out / "refined.npz", tokens=tokens, drafts=R["tau1"].astype(np.float32))
    logs = packed.index.log.values[R["rows"]]
    pd.DataFrame({"token": tokens, "log": logs}).to_parquet(out / "tokens.parquet", index=False)
    meta = dict(run=str(a.run), ckpt=a.ckpt, split=a.split, fold=a.fold, n_tokens=int(len(tokens)), mode=mode,
                arm=cfg["arm"], sec=round(time.time() - t0, 1), created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                tau1_eq_tau0_frac=float((R["tau1"] == R["tau0"]).all((2, 3)).mean()), teacher_shuffle=shuffle_info,
                **run4)
    if getattr(a, "eval_name", None):
        meta["eval_name"] = a.eval_name
    (out / "predict_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    return out


# ----------------------------------------------------------------------------------------------- score
def score(a) -> Path:
    cfg = json.loads((Path(a.run) / "config.json").read_text())
    out = out_dir_for(a, cfg)
    cmd = [PYTHON, str(HERE / "score_trajectories.py"), "--drafts", str(out / "refined.npz"), "--tokens",
           str(out / "tokens.parquet"), "--out", str(out / "scores_tau1.parquet"), "--workers", str(min(a.workers, 4))]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(REPO))
    return out


# ----------------------------------------------------------------------------------------------- report
def _scores_array(df: pd.DataFrame, tokens: np.ndarray, K: int) -> np.ndarray:
    """scores parquet -> [N, K, len(METRICS)] (NaN where missing)."""
    pos = {t: i for i, t in enumerate(tokens)}
    arr = np.full((len(tokens), K, len(METRICS)), np.nan)
    df = df[(df.k >= 0) & df.token.isin(pos)]
    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str) == ""]
    i = df.token.map(pos).to_numpy()
    k = df.k.to_numpy().astype(int)
    arr[i, k] = df[list(METRICS)].to_numpy(np.float64)
    return arr


def mix_metrics(orig: np.ndarray, ref: np.ndarray, modify: np.ndarray, valid: np.ndarray, family=None) -> Dict:
    """orig / ref [N, K, 7] (METRICS order), modify / valid [N, K] bool -> metric dict (see module docstring)."""
    M = {m: i for i, m in enumerate(METRICS)}
    fin = np.where(modify[..., None], ref, orig)
    v = valid & np.isfinite(orig).all(-1) & np.isfinite(fin).all(-1)
    o, f, md = orig[v], fin[v], modify[v]
    fail = lambda x, keys: np.any(np.stack([x[:, M[k]] < 1 for k in keys], -1), -1)
    core = ("nc", "dac", "ddc", "ttc")
    allk = core + ("comfort",)
    of, ff = fail(o, core), fail(f, core)
    oa, fa = fail(o, allk), fail(f, allk)
    res = dict(n=int(v.sum()), modified=int(md.sum()), modified_frac=float(md.mean()) if len(md) else float("nan"),
               pdms=float(f[:, M["pdms"]].mean()), pdms_orig=float(o[:, M["pdms"]].mean()),
               d_pdms_points=float(100 * (f[:, M["pdms"]] - o[:, M["pdms"]]).mean()),
               ep_loss_points=float(100 * (o[:, M["ep"]] - f[:, M["ep"]]).mean()),
               fail_any=int(ff.sum()), fail_any_orig=int(of.sum()), fixed=int((of & ~ff).sum()),
               new_fail=int((~oa & fa).sum()),
               unnecessary_mod=int((md & ~fail(o, ("nc", "dac", "ttc"))).sum()),
               harm=int((f[:, M["pdms"]] < o[:, M["pdms"]]).sum()))
    for k in ("nc", "dac", "ddc", "ttc", "comfort"):
        res[f"fail_{k}"] = int((f[:, M[k]] < 1).sum())
        res[f"fail_{k}_orig"] = int((o[:, M[k]] < 1).sum())
    if family is not None:
        fam = family[v]
        res["by_family"] = {}
        for c in np.unique(fam):
            s = fam == c
            res["by_family"][FAMILY_NAME.get(int(c), str(int(c)))] = dict(
                n=int(s.sum()), modified=int(md[s].sum()), d_pdms_points=float(100 * (f[s, M["pdms"]] - o[s, M["pdms"]]).mean()),
                fixed=int((of[s] & ~ff[s]).sum()), new_fail=int((~oa[s] & fa[s]).sum()),
                fail_any=int(ff[s].sum()), fail_any_orig=int(of[s].sum()))
    return res


def report(a) -> Dict:
    cfg = json.loads((Path(a.run) / "config.json").read_text())
    out = out_dir_for(a, cfg)
    P = np.load(out / "pred.npz", allow_pickle=False)
    tokens, rows = P["tokens"], P["rows"]
    N, K = P["p_g"].shape
    packed = RD.PackedSplit(a.split, a.packed_root)
    lab = np.asarray(packed.arrays["labels"][rows])                           # [N, 13, L]
    orig = np.stack([lab[..., RD.LBL[m]] for m in METRICS], -1).astype(np.float64)
    sc = pd.read_parquet(out / "scores_tau1.parquet")
    ref = _scores_array(sc, tokens, K)
    valid = P["draft_valid"].astype(bool)
    missing = int((valid & ~np.isfinite(ref).all(-1)).sum())
    thetas = sorted(set([float(t) for t in (a.theta or [0.5])] + (list(SWEEP) if a.sweep else [])))
    res = {}
    for th in thetas:
        res[f"{th:.4f}"] = mix_metrics(orig, ref, P["p_g"] >= th, valid, P["family"] if th in (a.theta or [0.5]) else None)
    rep = dict(run=str(a.run), split=a.split, fold=a.fold, n_tokens=int(N), K=int(K), missing_tau1_scores=missing,
               arm=cfg["arm"], surrogate=cfg.get("surrogate"), no_correction=mix_metrics(orig, ref, np.zeros_like(valid), valid),
               oracle_all=mix_metrics(orig, ref, np.ones_like(valid), valid), theta=res)
    if a.sweep:
        ok = [th for th in thetas if res[f"{th:.4f}"]["ep_loss_points"] <= a.budget_ep]
        rep["budget_ep_points"] = a.budget_ep
        rep["theta_at_budget"] = min(ok) if ok else None
        if ok:
            rep["at_budget"] = res[f"{min(ok):.4f}"]
    if a.direct_check:
        rep["direct_check"] = direct_check(P, orig, ref, packed, float((a.theta or [0.5])[0]), a.direct_check)
    rows_df = pd.DataFrame({"token": np.repeat(tokens, K), "k": np.tile(np.arange(K), N),
                            "family": P["family"].reshape(-1), "valid": valid.reshape(-1), "p_g": P["p_g"].reshape(-1),
                            "alpha": P["alpha"].reshape(-1)})
    for i, m in enumerate(METRICS):
        rows_df[f"{m}_orig"] = orig[..., i].reshape(-1)
        rows_df[f"{m}_tau1"] = ref[..., i].reshape(-1)
    rows_df.to_parquet(out / "report_rows.parquet", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    th0 = f"{float((a.theta or [0.5])[0]):.4f}"
    print(json.dumps({k: v for k, v in res[th0].items() if k != "by_family"}), flush=True)
    return rep


def direct_check(P, orig, ref, packed, theta: float, n_tokens: int) -> Dict:
    """Score apply_gate(tau0, tau1, p_g >= theta) of the first n_tokens directly and compare with the mixed values."""
    import score_trajectories as ST

    tokens, rows = P["tokens"], P["rows"]
    sim, scorer = ST.get_simulator_scorer()
    mism, n = 0, 0
    worst = 0.0
    for i in range(min(n_tokens, len(tokens))):
        tk = str(tokens[i])
        p = ST.locate_metric_cache(tk, str(packed.index.log.values[rows[i]]))
        if p is None:
            continue
        mc = ST.load_metric_cache(p)
        tau0 = torch.as_tensor(P["tau0"][i])
        fin = apply_gate(tau0, torch.as_tensor(P["tau1"][i]), torch.as_tensor(P["p_g"][i] >= theta)).numpy()
        got = ST.score_token(mc, fin, sim, scorer)
        mixed = np.where((P["p_g"][i] >= theta)[:, None], ref[i], orig[i])
        for k, g in enumerate(got):
            n += 1
            vals = np.array([g[m] for m in METRICS], np.float64)
            if not np.array_equal(vals, mixed[k]):
                mism += 1
                worst = max(worst, float(np.nanmax(np.abs(vals - mixed[k]))))
        # the unmodified drafts must be the original bytes
        keep = ~(P["p_g"][i] >= theta)
        assert np.array_equal(fin[keep], P["tau0"][i][keep])
    return dict(n_traj=n, mismatches=mism, max_abs_diff=worst, theta=theta)


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["predict", "score", "report", "all"])
    ap.add_argument("--run", required=True, help="run directory (runs/<run>)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--fold", type=int, default=None, help="train split: evaluate rows of this fold (out-of-fold)")
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--packed-root", default=str(RD.DATA_ROOT / "packed"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--teacher-root", default=None, help="override the teacher cache (default: by split subset)")
    ap.add_argument("--tokens-per-batch", type=int, default=8)
    ap.add_argument("--loader-workers", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4, help="scoring workers (<= 4)")
    ap.add_argument("--theta", type=float, nargs="*", default=None)
    ap.add_argument("--sweep", action="store_true", help="also report theta = 0.00 .. 1.00 (step 0.01)")
    ap.add_argument("--budget-ep", type=float, default=0.5, help="EP-loss budget [points] for theta_at_budget")
    ap.add_argument("--direct-check", type=int, default=0, help="re-score the gated trajectories of N tokens directly")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle-teacher-seed", type=int, default=None,
                    help="control: each token reads another token's BEV (seeded derangement of the evaluated tokens)")
    ap.add_argument("--shuffle-which", choices=["det", "map", "both"], default=None,
                    help="run 4: which teacher input is shuffled (default: the arm's teacher(s): T det, M map, TM both)")
    ap.add_argument("--drop-branch", choices=["det", "map"], default=None,
                    help="run 4, arm TM only: the branch's normalised input is 0 (= training-mean feature)")
    ap.add_argument("--eval-name", default=None,
                    help="output subdirectory of the run (default eval_<split>[_fold<k>]); required for controls")
    ap.add_argument("--token-subset", default=None,
                    help="run 4: token parquet restricting the evaluated rows (train: must equal the run's; dev: default "
                         "the run's dev_token_subset; navtest: refused)")
    ap.add_argument("--resmap-root", default=None, help="override the ReSMap cache root (meta-checked; tests)")
    return ap


def resolve_fold(a) -> None:
    """--split train without --fold -> the run's own held-out fold (out-of-fold evaluation)."""
    if a.split == "train" and a.fold is None:
        cfg = json.loads((Path(a.run) / "config.json").read_text())
        a.fold = int(cfg.get("fold", -1))
        if a.fold < 0:
            raise SystemExit("run trained on all folds: evaluating it on train is in-sample; pass --fold explicitly")


def main(argv=None):
    a = get_parser().parse_args(argv)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    resolve_fold(a)
    if a.cmd in ("predict", "all"):
        predict(a)
    if a.cmd in ("score", "all"):
        score(a)
    if a.cmd in ("report", "all"):
        report(a)


if __name__ == "__main__":
    main()
