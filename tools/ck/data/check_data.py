#!/usr/bin/env python
"""CK Phase 1 data integrity checks -> <root>/impl-data/check_<split>.json (+ check_summary.json).

  check_data.py [splits...] [--root R] [--sample 300] [--labels cand]

Per split (each block is skipped with a note when its inputs are not there yet):
  dump    plan / bev file counts vs split tokens; shard metas (ckpt sha16, max fp16 rel err, max top-1 err, tok/s);
          a random sample of BEV files: shape [256,50,100], float16, finite, file size 2,560,128 B.
  packed  n == split n, ok all, cand[:,0] == sub_traj, finite, v2_final descending, distinct cand_idx, v2_sim/v2_im in
          [0,1]; packed == dump on the sample (bitwise).
  labels  ok fraction; value domains (nc in {0,.5,1}, dac/ttc/comfort/ddc in {0,1}, ep/pdms in [0,1]); PDMS of rank 0
          (= v2 submitted), oracle-16, mean over all 16; failure rates (nc<1, dac<1, ttc<1, any) by rank.
          navtest: per-token rank-0 vs the official v2 CSV (sub-scores equal, |dPDMS| max) and the mean vs 0.88137
          (+-0.002, else 'v2_mismatch').
  lead    coverage; v2 rank-0 NC|TTC failure rate in lead-decel scenes (has_lead & D_1, censored excluded) vs moving
          leads without decel, with n.
Cross-split: candidate failure-rate ratio train / val (optimism bias of in-sample v2 candidates; report 44 §11:
  < 0.5 -> add top-17..32 candidates).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root of this worktree
from tools.ck.data import common as CM  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LC = {k: i for i, k in enumerate(CM.LABEL_COLS)}
BEV_BYTES = 2_560_128
CSV_COLS = dict(nc="no_at_fault_collisions", dac="drivable_area_compliance", ddc="driving_direction_compliance",
                ep="ego_progress", ttc="time_to_collision_within_bound", comfort="comfort", pdms="score")


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def check_dump(split_name, root, toks, rng, n_sample):
    import json
    d = CM.dump_dir(split_name, root)
    if not d.exists():
        return dict(skipped="no dump dir")
    have_plan = np.array([CM.plan_path(d, t).is_file() for t in toks.token])
    have_bev = np.array([CM.bev_path(d, t).is_file() for t in toks.token])
    metas = [json.loads(p.read_text()) for p in sorted(d.glob("meta_shard*.json"))]
    r = dict(n=len(toks), n_plan=int(have_plan.sum()), n_bev=int(have_bev.sum()), n_shard_meta=len(metas),
             ckpt_sha16=sorted({m.get("ckpt_sha256", "")[:16] for m in metas}),
             max_fp16_rel_err=max([m.get("max_fp16_rel_err", 0.0) for m in metas] or [None] if metas else [0.0]),
             max_top1_err=max([m.get("max_top1_err", 0.0) for m in metas] or [0.0]),
             tok_per_s=[m.get("tok_per_s") for m in metas])
    idx = np.flatnonzero(have_bev)
    smp = rng.choice(idx, min(n_sample, len(idx)), replace=False) if len(idx) else []
    bad = []
    for i in smp:
        t = toks.token.iat[i]
        p = CM.bev_path(d, t)
        a = np.load(p, mmap_mode="r")
        if a.shape != (256, 50, 100) or a.dtype != np.float16 or p.stat().st_size != BEV_BYTES \
                or not np.isfinite(np.asarray(a)).all():
            bad.append(t)
    r.update(bev_sample=len(smp), bev_sample_bad=len(bad), bev_bad_head=bad[:5],
             ok=bool(have_plan.all() and have_bev.all() and not bad and r["ckpt_sha16"] == [CM.V2_CKPT_SHA16]
                     and r["max_top1_err"] < 1e-4 and (r["max_fp16_rel_err"] or 0) < 1e-3))
    return r


def check_packed(split_name, root, toks, rng, n_sample):
    p = CM.packed_dir(split_name, root)
    if not (p / "meta.json").is_file():
        return dict(skipped="not packed")
    L = lambda n: np.load(p / f"{n}.npy", mmap_mode="r")  # noqa: E731
    pt = pd.read_parquet(p / "tokens.parquet")
    cand, idx, fin, im, sim = L("cand"), L("cand_idx"), L("v2_final"), L("v2_im"), L("v2_sim")
    sub, ok = L("sub_traj"), L("ok")
    r = dict(n=len(pt), n_split=len(toks), same_order=bool((pt.token.values == toks.token.values).all())
             if len(pt) == len(toks) else False, n_ok=int(ok.sum()), k=int(cand.shape[1]))
    c = np.asarray(cand[ok])
    r["cand0_eq_sub"] = bool(np.array_equal(c[:, 0], np.asarray(sub)[ok]))
    r["cand_finite"] = bool(np.isfinite(c).all())
    f = np.asarray(fin)[ok]
    r["final_descending"] = bool((np.diff(f, axis=1) <= 0).all())
    ii = np.asarray(idx)[ok]
    r["idx_distinct"] = bool(all(len(set(row)) == len(row) for row in ii[: min(len(ii), 5000)]))
    s, m = np.asarray(sim)[ok], np.asarray(im)[ok]
    r["sim_in_01"] = bool(((s >= 0) & (s <= 1)).all())
    r["im_in_01"] = bool(((m >= 0) & (m <= 1)).all())
    d = CM.dump_dir(split_name, root)
    smp = rng.choice(np.flatnonzero(ok), min(n_sample, int(ok.sum())), replace=False) if ok.any() else []
    mism = 0
    for i in smp:
        t = pt.token.iat[i]
        with np.load(CM.plan_path(d, t)) as z:
            kd = CM.K_CAND
            if not np.array_equal(z[f"cand_top{kd}"][: min(kd, r["k"])], cand[i][: min(kd, r["k"])]):
                mism += 1
    r["packed_vs_dump_sample"] = len(smp)
    r["packed_vs_dump_mismatch"] = mism
    r["ok"] = bool(r["n_ok"] == r["n"] == r["n_split"] and r["same_order"] and r["cand0_eq_sub"] and r["cand_finite"]
                   and r["final_descending"] and r["idx_distinct"] and r["sim_in_01"] and r["im_in_01"] and mism == 0)
    return r


def label_stats(lab, ok):
    """lab [N,K,9], ok [N,K] -> dict of headline numbers."""
    allok = ok.all(1)
    L = lab[allok]
    nc, dac, ttc, pd_ = L[..., LC["nc"]], L[..., LC["dac"]], L[..., LC["ttc"]], L[..., LC["pdms"]]
    fail = (nc < 1) | (dac < 1) | (ttc < 1)
    return dict(n_tokens_all_ok=int(allok.sum()), pdms_rank0=_f(pd_[:, 0].mean()), pdms_oracle=_f(pd_.max(1).mean()),
                pdms_mean_k=_f(pd_.mean()), fail_rank0=_f(fail[:, 0].mean()), fail_all_k=_f(fail.mean()),
                fail_by_rank=[round(float(x), 4) for x in fail.mean(0)],
                nc_fail_all_k=_f((nc < 1).mean()), dac_fail_all_k=_f((dac < 1).mean()),
                ttc_fail_all_k=_f((ttc < 1).mean()),
                rank0_means={k: _f(L[:, 0, LC[k]].mean()) for k in CM.LABEL_COLS})


def check_labels(split_name, root, name, toks):
    d = CM.labels_dir(split_name, name, root)
    if not (d / "labels.npy").is_file():
        return dict(skipped=f"no labels/{name}")
    lab = np.load(d / "labels.npy")
    ok = np.load(d / "ok.npy")
    r = dict(n=int(lab.shape[0]), k=int(lab.shape[1]), ok_frac=_f(ok.mean()), tokens_all_ok=int(ok.all(1).sum()))
    v = lab[ok]
    dom = {}
    for k, allowed in (("nc", (0.0, 0.5, 1.0)), ("dac", (0.0, 1.0)), ("ttc", (0.0, 1.0)), ("comfort", (0.0, 1.0)),
                       ("ddc", (0.0, 0.5, 1.0))):
        dom[k] = bool(np.isin(v[:, LC[k]], allowed).all())
    for k in ("ep", "pdms"):
        x = v[:, LC[k]]
        dom[k] = bool(((x >= 0) & (x <= 1 + 1e-6)).all())
    r["domains_ok"] = dom
    r.update(label_stats(lab, ok))
    if split_name == "navtest":
        pk = pd.read_parquet(CM.packed_dir(split_name, root) / "tokens.parquet")
        csv = pd.read_csv(CM.NAVTEST_CSV)
        csv = csv[csv.token != "average"].set_index("token")
        common = [t for t in pk.token if t in csv.index]
        rows = pk.set_index("token").loc[common, "row"].values
        c = csv.loc[common]
        cmp = {}
        for k, col in CSV_COLS.items():
            a = lab[rows, 0, LC[k]].astype(np.float64)
            b = c[col].values.astype(np.float64)
            dd = np.abs(a - b)
            cmp[k] = dict(n_diff_gt_1e4=int((dd > 1e-4).sum()), max_abs=_f(np.nanmax(dd)))
        r["navtest_csv"] = dict(n_common=len(common), per_key=cmp, csv_mean_pdms=_f(c.score.mean()),
                                rank0_mean_pdms=_f(lab[rows, 0, LC["pdms"]].mean()),
                                contract_pdms=CM.NAVTEST_CSV_PDMS)
        diff = abs(r["navtest_csv"]["rank0_mean_pdms"] - CM.NAVTEST_CSV_PDMS)
        r["navtest_csv"]["v2_mismatch"] = bool(diff > CM.NAVTEST_CSV_TOL or len(common) < len(pk))
    r["ok"] = bool(all(dom.values()) and r["ok_frac"] == 1.0
                   and not r.get("navtest_csv", {}).get("v2_mismatch", False))
    return r


def check_lead(split_name, root, name):
    p = CM.lead_path(split_name, root)
    if not p.is_file():
        return dict(skipped="no lead file")
    ld = pd.read_parquet(p)
    r = dict(n=len(ld), has_lead=int((ld.has_lead == 1).sum()), lead_decel=int((ld.lead_decel == 1).sum()),
             errors=int((ld.error != "").sum()))
    d = CM.labels_dir(split_name, name, root)
    pk = CM.packed_dir(split_name, root) / "tokens.parquet"
    if (d / "labels.npy").is_file() and pk.is_file():
        lab = np.load(d / "labels.npy")
        ok = np.load(d / "ok.npy")
        t = pd.read_parquet(pk)
        m = ld.set_index("token").loc[t.token]
        f0 = (lab[:, 0, LC["nc"]] < 1) | (lab[:, 0, LC["ttc"]] < 1)
        dec = (m.has_lead.values == 1) & (m.D_1.values == 1) & ok[:, 0]
        nod = (m.has_lead.values == 1) & (m.D_1.values == 0) & (m.lead_v.values >= 0.5) & ok[:, 0]
        r["v2_rank0_nc_or_ttc_fail"] = dict(lead_decel=_f(f0[dec].mean()), n_lead_decel=int(dec.sum()),
                                            moving_no_decel=_f(f0[nod].mean()), n_moving_no_decel=int(nod.sum()),
                                            all=_f(f0[ok[:, 0]].mean()))
    r["ok"] = bool(r["n"] > 0 and r["errors"] == 0)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("splits", nargs="*", default=["navtrain_val", "navtrain_train", "navtest"])
    ap.add_argument("--root", default="")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--labels", default="cand")
    a = ap.parse_args(argv)
    root = a.root or None
    out = CM.root_dir(root) / "impl-data"
    summ = {}
    for s in a.splits:
        t0 = time.time()
        rng = np.random.default_rng(0)
        toks = CM.split_tokens(s)
        pk = CM.packed_dir(s, root) / "tokens.parquet"
        if pk.is_file() and len(pd.read_parquet(pk)) != len(toks):     # subset pack (smoke): check that subset
            toks = toks.set_index("token").loc[pd.read_parquet(pk).token].reset_index()[["token", "log", "city"]]
        r = dict(split=s, dump=check_dump(s, root, toks, rng, a.sample), packed=check_packed(s, root, toks, rng, a.sample),
                 labels=check_labels(s, root, a.labels, toks), lead=check_lead(s, root, a.labels))
        r["seconds"] = round(time.time() - t0, 1)
        r["checked"] = time.strftime("%F %T")
        CM.atomic_write_json(out / f"check_{s}.json", r)
        summ[s] = {k: (v.get("ok") if isinstance(v, dict) and "ok" in v else v.get("skipped") if isinstance(v, dict)
                       else v) for k, v in r.items() if k in ("dump", "packed", "labels", "lead")}
        lb = r["labels"]
        if "pdms_rank0" in lb:
            summ[s].update(pdms_rank0=lb["pdms_rank0"], pdms_oracle=lb["pdms_oracle"], fail_all_k=lb["fail_all_k"])
        print(s, summ[s], flush=True)
    fr = {s: summ[s].get("fail_all_k") for s in summ}
    if fr.get("navtrain_train") is not None and fr.get("navtrain_val"):
        ratio = fr["navtrain_train"] / fr["navtrain_val"]
        summ["optimism_ratio_train_over_val"] = dict(ratio=ratio, flag_add_top32=bool(ratio < 0.5),
                                                     fail_train=fr["navtrain_train"], fail_val=fr["navtrain_val"])
        print("candidate failure rate train/val", summ["optimism_ratio_train_over_val"], flush=True)
    CM.atomic_write_json(out / "check_summary.json", dict(summ, checked=time.strftime("%F %T")))


if __name__ == "__main__":
    main()
