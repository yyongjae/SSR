#!/usr/bin/env python
"""Validate the stored drivable-area SDFs against the official DAC on navtest (IMPL_SPEC section 3.5).

Trajectories: the archived PARA-SSR interaction_final navtest submissions (work_dirs/eval/
para_ssr_interaction_final_navtest_trajectories.pkl, = B_read_vs_generate base.npz ``traj``; official labels in
pdm_attr/table.parquet ``re_drivable_area_compliance``) and, as a false-alarm check, the human (GT) trajectory of the
same token where it is valid.  Each trajectory is re-scored with the official pdm_score (cf_common.score) to get the
official DAC and the scorer's own 41 LQR-tracked ego corners (``scorer._ego_coords[1][:, :4]``, FL/RL/RR/FR).

Per trajectory and corner set (``tracked`` = the 41 LQR-tracked footprints the official DAC judges; ``raw`` = the
41-point 0.1 s reference obtained by linear time interpolation of [origin; 8 poses], sf_common.Path):
  sdf_min   : min over in-grid corners of the stored float16 SDF, bilinear (sdf.sample_sdf, float32 torch)
  n_oog     : corners outside the SDF grid (excluded)
  exact_min : min exact signed distance of the corners to the official-layer union (uncropped)
  exact_fail: any corner outside the union (for ``tracked`` this must equal the official DAC failure)
SDF prediction of a DAC failure: sdf_min < 0 (also reported at margins m: sdf_min < m).

Steps (CPU only, <= 2 workers):
  validate_sdf.py tokens [--n-random 300 --n-fail 250 --seed 0]   # -> <DATA>/sdf/validation/navtest_tokens.parquet
  build_sdf.py --subset navtest --tokens <DATA>/sdf/validation/navtest_tokens.parquet --workers 2
  validate_sdf.py run --workers 2      # -> <DATA>/sdf/validation/navtest_rows.parquet + summary json
  validate_sdf.py summarize            # recompute the summary from the rows
Summary: report/refiner_T/sdf_validation.json (copy under <DATA>/sdf/validation/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path("/home/external-user/yongjae/SSR")
for p in (ROOT, ROOT / "report/collision_counterfactual/counterfactual",
          ROOT / "report/planner_vs_perception_tests/safety_filter"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from navsim.agents.para_ssr.refiner import sdf as S  # noqa: E402

DATA = Path("/home/external-user/ssd/yongjae_refiner")
VDIR = S.SDF_ROOT / "validation"
TOKENS = VDIR / "navtest_tokens.parquet"
ROWS = VDIR / "navtest_rows.parquet"
ERRS = VDIR / "navtest_corner_errors.npz"
SUMMARY = ROOT / "report/refiner_T/sdf_validation.json"
TABLE = ROOT / "report/perception_reliability/pdm_attr/table.parquet"
MARGINS = (-0.1, 0.0, 0.1, 0.2)

G = {}


# ----------------------------------------------------------------------------------------- sampling
def make_tokens(n_random: int, n_fail: int, seed: int) -> pd.DataFrame:
    """n_random uniform navtest tokens + n_fail further tokens drawn from the official DAC failures."""
    t = pd.read_parquet(TABLE, columns=["token", "log", "re_drivable_area_compliance"])
    rng = np.random.default_rng(seed)
    r_idx = rng.choice(len(t), n_random, replace=False)
    pool = np.setdiff1d(np.nonzero(t.re_drivable_area_compliance.values < 1)[0], r_idx)
    f_idx = rng.choice(pool, min(n_fail, len(pool)), replace=False)
    df = pd.concat([t.iloc[r_idx].assign(random=True), t.iloc[f_idx].assign(random=False)])
    df = df.rename(columns={"re_drivable_area_compliance": "dac_table"}).reset_index(drop=True)
    VDIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TOKENS, index=False)
    print(f"[validate_sdf] {len(df)} tokens ({n_random} random, {len(f_idx)} DAC-fail enriched; "
          f"random DAC fail {int((df[df.random].dac_table < 1).sum())}) -> {TOKENS}")
    return df


# ----------------------------------------------------------------------------------------- worker
def _init():
    import cf_common as CF
    CF.init_worker()
    G["CF"] = CF
    G["human"] = CF.G["RA"].G["human"]


def _corner_stats(corners, f16_t, union, kind):
    import shapely
    import torch
    v, m = S.sample_sdf(f16_t, torch.from_numpy(corners.astype(np.float32)))
    v, m = v.numpy().astype(np.float64), m.numpy()
    vn, mn = S.sample_sdf_np(f16_t.numpy(), corners.astype(np.float32).astype(np.float64))
    ex = S.signed_distance_exact(union, corners)
    inside = shapely.contains_xy(union, corners[..., 0], corners[..., 1])
    vv = np.where(m, v, np.inf)
    t_first = int(np.argmax((vv < 0).any(1))) if (vv < 0).any() else -1
    t_first_ex = int(np.argmax((~inside).any(1))) if (~inside).any() else -1
    rec = {f"{kind}_sdf_min": float(vv.min()), f"{kind}_n_oog": int((~m).sum()),
           f"{kind}_exact_min": float(ex.min()), f"{kind}_exact_fail": bool((~inside).any()),
           f"{kind}_sdf_first_t": t_first, f"{kind}_exact_first_t": t_first_ex,
           f"{kind}_torch_np_maxdiff": float(np.abs(v - vn)[m].max()) if m.any() else 0.0,
           f"{kind}_x_max": float(corners[..., 0].max()), f"{kind}_absy_max": float(np.abs(corners[..., 1]).max())}
    sel = m & (np.abs(ex) < 2.0)
    return rec, np.stack([ex[sel], v[sel]], -1)


def one_token(row):
    import shapely
    import torch
    import sf_common as SF
    CF = G["CF"]
    tok, log = row["token"], row["log"]
    out_rows, errs = [], []
    try:
        mc = CF.load_mc(log, tok)
        ra = mc.ego_state.rear_axle
        xyh = (float(ra.x), float(ra.y), float(ra.heading))
        f16 = S.load_sdf(S.sdf_path(tok, "navtest"))
        with np.load(S.sdf_path(tok, "navtest")) as z:
            assert np.allclose(z["ego_xyh"], xyh), "stored ego pose differs from the metric cache"
        f16_t = torch.from_numpy(f16)
        union, _ = S.drivable_union(mc.drivable_area_map, xyh, crop=False)
        shapely.prepare(union)
        trajs = {"orig": np.asarray(CF.G["traj"][tok], np.float64)}
        h, hv = G["human"].get(tok, (None, False))
        if hv:
            trajs["human"] = np.asarray(h, np.float64)
        for name, poses in trajs.items():
            res, _, _, _ = CF.score(mc, poses)
            tracked = S.global_to_n(np.asarray(CF.G["scorer"]._ego_coords[1][:, :4], np.float64), xyh)  # [41,4,2]
            P = SF.Path(poses)
            xr, yr, hr = P.ref_poses_time()
            raw = SF.ego_corners(xr, yr, hr, 0.0)                                                     # [41,4,2]
            rec = dict(token=tok, log=log, traj=name, random=bool(row["random"]),
                       dac_official=float(res["dac"]), dac_table=float(row["dac_table"]) if name == "orig" else np.nan,
                       lqr_max_dev=float(np.abs(tracked.mean(-2) - raw.mean(-2)).max()), ok=True)
            for kind, C in (("tracked", tracked), ("raw", raw)):
                r, e = _corner_stats(C, f16_t, union, kind)
                rec.update(r)
                errs.append(np.concatenate([e, np.full((len(e), 1), 0 if kind == "tracked" else 1)], 1))
            out_rows.append(rec)
    except Exception as ex:  # recorded
        out_rows.append(dict(token=tok, log=log, traj="orig", ok=False, error=f"{type(ex).__name__}: {ex}"))
    return out_rows, (np.concatenate(errs) if errs else np.zeros((0, 3)))


# ----------------------------------------------------------------------------------------- summary
def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def _conf(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp = int((y & pred).sum()), int((~y & pred).sum())
    fn, tn = int((y & ~pred).sum()), int((~y & ~pred).sum())
    return dict(n=len(y), pos=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn, agree=(tp + tn) / max(len(y), 1),
                recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1), fpr=fp / max(fp + tn, 1))


def summarize(rows: pd.DataFrame, errs: np.ndarray) -> dict:
    rows = rows[rows.ok]
    out = dict(n_tokens=int(rows.token.nunique()), n_rows=int(len(rows)))
    orig = rows[rows.traj == "orig"]
    out["sanity"] = dict(
        official_vs_table_mismatch=int((orig.dac_official != orig.dac_table).sum()),
        official_vs_exact_union_on_tracked_mismatch=int(((rows.dac_official < 1) != rows.tracked_exact_fail).sum()),
        torch_vs_numpy_maxdiff=float(max(rows.tracked_torch_np_maxdiff.max(), rows.raw_torch_np_maxdiff.max())),
        tracked_n_oog_total=int(rows.tracked_n_oog.sum()), raw_n_oog_total=int(rows.raw_n_oog.sum()),
        rows_with_oog_tracked=int((rows.tracked_n_oog > 0).sum()), rows_with_oog_raw=int((rows.raw_n_oog > 0).sum()),
        corner_x_max=float(max(rows.tracked_x_max.max(), rows.raw_x_max.max())),
        corner_absy_max=float(max(rows.tracked_absy_max.max(), rows.raw_absy_max.max())))
    subsets = {"orig_random": orig[orig.random], "orig_all": orig, "human_all": rows[rows.traj == "human"]}
    for sname, d in subsets.items():
        y = d.dac_official.values < 1
        blk = dict(n=int(len(d)), official_fail=int(y.sum()))
        for kind in ("tracked", "raw"):
            k = dict(sdf=_conf(y, d[f"{kind}_sdf_min"].values < 0),
                     exact_union=_conf(y, d[f"{kind}_exact_fail"].values),
                     sdf_vs_exact_union=_conf(d[f"{kind}_exact_fail"].values, d[f"{kind}_sdf_min"].values < 0),
                     auc_sdf=_auc(y, -d[f"{kind}_sdf_min"].values),
                     auc_exact=_auc(y, -d[f"{kind}_exact_min"].values),
                     margins={str(m): _conf(y, d[f"{kind}_sdf_min"].values < m) for m in MARGINS})
            dis = d[(d[f"{kind}_sdf_min"].values < 0) != y]
            k["disagreements"] = [dict(token=r.token, official_fail=bool(r.dac_official < 1),
                                       sdf_min=round(float(r[f"{kind}_sdf_min"]), 4),
                                       exact_min=round(float(r[f"{kind}_exact_min"]), 4),
                                       exact_fail=bool(r[f"{kind}_exact_fail"]))
                                  for _, r in dis.iterrows()][:40]
            blk[kind] = k
        out[sname] = blk
    # corner-level accuracy of the stored float16 bilinear SDF vs the exact signed distance (|d| < 2 m)
    acc = {}
    for kid, kind in ((0, "tracked"), (1, "raw")):
        e = errs[errs[:, 2] == kid]
        if not len(e):
            continue
        ae = np.abs(e[:, 1] - e[:, 0])
        clear = np.abs(e[:, 0]) > 0.05
        acc[kind] = dict(n_corners=int(len(e)), abs_err_median=float(np.median(ae)),
                         abs_err_p95=float(np.quantile(ae, 0.95)), abs_err_p99=float(np.quantile(ae, 0.99)),
                         abs_err_max=float(ae.max()), bias=float((e[:, 1] - e[:, 0]).mean()),
                         sign_mismatch_clear=int((np.sign(e[:, 1]) != np.sign(e[:, 0]))[clear].sum()),
                         sign_mismatch_within_5cm=int(((e[:, 1] >= 0) != (e[:, 0] > 0))[~clear].sum()),
                         n_within_5cm=int((~clear).sum()))
    out["corner_accuracy_abs_d_lt_2m"] = acc
    # build cost / storage of the validated files (from build_sdf stats)
    sf = S.SDF_ROOT / "navtest/_build/stats.jsonl"
    if sf.exists():
        st = pd.DataFrame([json.loads(x) for x in sf.read_text().splitlines()])
        st = st[(st.status == "built") & st.token.isin(set(rows.token))].drop_duplicates("token", keep="last")
        out["build"] = dict(n=int(len(st)),
                            t_build_s=dict(median=float(st.t_build.median()), mean=float(st.t_build.mean()),
                                           p95=float(st.t_build.quantile(0.95))),
                            t_load_s=dict(median=float(st.t_load.median()), mean=float(st.t_load.mean()),
                                          p95=float(st.t_load.quantile(0.95))),
                            bytes=dict(median=float(st.bytes.median()), mean=float(st.bytes.mean()),
                                       max=float(st.bytes.max())),
                            n_invalid_polys=int(st.n_invalid.sum()), frac_inside_median=float(st.frac_inside.median()))
    return out


def run(workers: int) -> dict:
    df = pd.read_parquet(TOKENS)
    missing = [t for t in df.token if not S.sdf_path(t, "navtest").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} navtest SDFs missing; run build_sdf.py --subset navtest --tokens {TOKENS}")
    t0 = time.time()
    rows, errs = [], []
    recs = df.to_dict("records")
    with Pool(workers, initializer=_init) as pool:
        for k, (r, e) in enumerate(pool.imap_unordered(one_token, recs, chunksize=4), 1):
            rows.extend(r)
            errs.append(e)
            if k % 50 == 0 or k == len(recs):
                print(f"[validate_sdf] {k}/{len(recs)} {time.time() - t0:.0f}s", flush=True)
    rows = pd.DataFrame(rows)
    rows.to_parquet(ROWS, index=False)
    errs = np.concatenate(errs)
    np.savez_compressed(ERRS, err=errs.astype(np.float32))
    return write_summary(rows, errs, time.time() - t0, workers)


def write_summary(rows, errs, wall=None, workers=None) -> dict:
    s = summarize(rows, errs)
    s.update(wall_s=wall, workers=workers, sdf_version=S.SDF_VERSION, rows=str(ROWS), tokens=str(TOKENS),
             n_errors=int((~rows.ok).sum()) if "ok" in rows else 0)
    txt = json.dumps(s, indent=1, default=float)
    SUMMARY.write_text(txt)
    (VDIR / "sdf_validation.json").write_text(txt)
    print(txt[:6000])
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a1 = sp.add_parser("tokens")
    a1.add_argument("--n-random", type=int, default=300)
    a1.add_argument("--n-fail", type=int, default=250)
    a1.add_argument("--seed", type=int, default=0)
    a2 = sp.add_parser("run")
    a2.add_argument("--workers", type=int, default=2)
    sp.add_parser("summarize")
    a = ap.parse_args()
    if a.cmd == "tokens":
        make_tokens(a.n_random, a.n_fail, a.seed)
    elif a.cmd == "run":
        if not 1 <= a.workers <= 2:
            ap.error("validation uses at most 2 workers (shared machine)")
        run(a.workers)
    else:
        write_summary(pd.read_parquet(ROWS), np.load(ERRS)["err"].astype(np.float64))


if __name__ == "__main__":
    main()
