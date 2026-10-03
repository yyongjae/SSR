#!/usr/bin/env python
"""M6 builder (IMPL_SPEC §3.4): per-token GT future objects from the raw NAVSIM logs + metric-cache validation (V2).

  python tools/refiner/build_future_objects.py --tokens <parquet: token, log[, frame_idx]> --logs <log dir> \
      --out /home/external-user/ssd/yongjae_refiner/objects/<split> --workers 2
  # validation against an official metric cache (+ UNKNOWN-rate measurement):
  python tools/refiner/build_future_objects.py ... --mc-root <metric cache root> [--report <summary.json>]

What it does (conventions: navsim/agents/para_ssr/refiner/gt_future.py docstring)
  * one task per log: the log pickle is read once; each token is located by its ``token`` (list position in the log;
    the table's ``frame_idx`` is the frame dict's own ``frame_idx`` field -- NOT the list position -- and is only
    cross-checked); frames log[i : i + 17] -> gt_future.build_from_frames -> <out>/<token>.npz (atomic write).
  * resumable: an existing npz is not rebuilt (``--overwrite`` to rebuild); with ``--mc-root`` the existing npz is
    loaded and validated.  <out>/index.parquet is rewritten at the end (rows of this run merged over older rows).
  * validation (``--mc-root``; per token, npz as stored, i.e. float32):
      - query(t = 0.1 * arange(51)) corners vs metric-cache occupancy-map polygons, both in N (MC polygons moved to N
        with the metric cache's own t0 ego pose): max corner error [m];
      - presence: my state == OBS  <=>  track in map[ti], over all my tracks x 51 maps;
      - metric-cache tracks missing from my set must all be explained by the radius R (centre > R at every keyframe);
        their minimum distance to the GT ego is reported;
      - unique_objects (first appearance): L, W, heading, velocity (agents), class;
      - red-light tokens in the maps (expected 0).
    UNKNOWN rates (gt_future.unknown_space, spec definition = reach 75 m / t > 5 s, and extended = + outside R - 10 m)
    for the Pacifica footprint corners at the 41 dense steps + TTC projections (+0.3/0.6/0.9 s, constant velocity)
    of three reference trajectories: human (GT log), constant velocity (v0, straight), 1.4x speed along the 8 s log path.
  * logs: stdout (run under nohup into /home/external-user/ssd/yongjae_refiner/objects/logs/).
CPU only; <= 4 workers (machine shared).
"""
from __future__ import annotations

import argparse
import json
import lzma
import os
import pickle
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "report/planner_vs_perception_tests/safety_filter"))

from navsim.agents.para_ssr.refiner import gt_future as G  # noqa: E402

N_MC = 51                       # metric-cache occupancy maps, t = 0.1 * ti, ti = 0..50
T41 = np.arange(41) * 0.1       # dense ego steps 0..4 s
TTC_DT = (0.3, 0.6, 0.9)        # official TTC constant-velocity projections [s]
N_LOG_FRAMES = G.N_PATH_FRAMES + 1


# ------------------------------------------------------------------------------------------------ helpers
def load_mc(mc_root, log, token):
    p = Path(mc_root) / log / "unknown" / token / "metric_cache.pkl"
    if not p.exists():
        c = list((Path(mc_root) / log).glob(f"*/{token}/metric_cache.pkl"))
        if not c:
            return None
        p = c[0]
    with lzma.open(p, "rb") as f:
        return pickle.load(f)


def _to_n(x, y, x0, y0, h0):
    c, s = np.cos(h0), np.sin(h0)
    dx, dy = np.asarray(x, np.float64) - x0, np.asarray(y, np.float64) - y0
    return c * dx + s * dy, -s * dx + c * dy


def validate_mc(obj, mc):
    """compare the stored objects of one token with its metric cache; returns a flat dict of statistics."""
    obs = mc.observation
    ra = mc.ego_state.rear_axle
    x0, y0, h0 = float(ra.x), float(ra.y), float(ra.heading)
    p0 = obj["pose0"]
    r = dict(pose0_err=float(max(abs(p0[0] - x0), abs(p0[1] - y0), abs(G.wrap(p0[2] - h0)))))
    maps = obs._occupancy_maps
    r["mc_n_maps"] = len(maps)
    t = np.arange(len(maps)) * 0.1                      # same float grid as the metric cache
    boxes, state = G.query(obj, t, dtype=np.float64)
    C = G.box_corners(boxes)                            # [A, T, 4, 2]
    idx = {str(k): i for i, k in enumerate(obj["track"])}
    A = len(idx)
    pres_mc = np.zeros((A, len(maps)), bool)
    err = np.zeros((A, len(maps)))
    n_red = 0
    excl = {}
    for ti, m in enumerate(maps):
        for tk, g in zip(m.tokens, m._geometries):
            if obs.red_light_token in tk:
                n_red += 1
                continue
            co = np.asarray(g.exterior.coords)[:4]
            xn, yn = _to_n(co[:, 0], co[:, 1], x0, y0, h0)
            i = idx.get(tk)
            if i is None:
                excl.setdefault(tk, []).append((ti, xn.mean(), yn.mean(), float(np.hypot(xn, yn).min())))
                continue
            pres_mc[i, ti] = True
            err[i, ti] = np.hypot(C[i, ti, :, 0] - xn, C[i, ti, :, 1] - yn).max()
    mine = state == G.OBS
    both = mine & pres_mc
    r.update(n_mine=A, n_pairs=int(mine.size), n_pairs_obs=int(both.sum()),
             presence_mismatch=int((mine != pres_mc).sum()),
             mine_not_in_mc=int((~pres_mc.any(1)).sum()) if A else 0,
             max_corner_err=float(err[both].max()) if both.any() else 0.0,
             red_light_tokens=n_red, n_mc_tracks=int(pres_mc.any(1).sum()) + len(excl), n_excl=len(excl))
    # excluded (radius) tracks: must be > R at every keyframe; distance to the GT ego
    R = float(obj["R"])
    unexpl, dmin_ego, dmin_org, corner_minus_R = 0, np.inf, np.inf, np.inf
    for tk, lst in excl.items():
        a = np.asarray(lst)
        kfm = (a[:, 0].astype(int) % 5) == 0
        d0 = np.hypot(a[kfm, 1], a[kfm, 2]) if kfm.any() else np.hypot(a[:, 1], a[:, 2])
        unexpl += int(d0.min() <= R + 1e-6)
        dmin_org = min(dmin_org, float(np.hypot(a[:, 1], a[:, 2]).min()))
        corner_minus_R = min(corner_minus_R, float(a[:, 3].min()) - R)
        ego = G.gt_ego_xy(obj, a[:, 0] * 0.1)
        dmin_ego = min(dmin_ego, float(np.hypot(a[:, 1] - ego[:, 0], a[:, 2] - ego[:, 1]).min()))
    r.update(excl_unexplained=unexpl, excl_min_d_origin=dmin_org, excl_min_d_ego=dmin_ego,
             excl_min_corner_minus_R=corner_minus_R)
    # unique objects (first appearance)
    uo = obs.unique_objects
    eh = elw = ev = 0.0
    ecls = 0
    stat_v = 0.0
    c, s = np.cos(h0), np.sin(h0)
    for tk, i in idx.items():
        o = uo.get(tk)
        if o is None:
            continue
        f = obj["first"][i]
        eh = max(eh, abs(float(G.wrap(o.center.heading - h0 - f[G.FI_H]))))
        elw = max(elw, abs(o.box.length - f[G.FI_L]), abs(o.box.width - f[G.FI_W]))
        ecls += int(o.tracked_object_type.value != int(obj["meta"][i, G.MT_CLASS]))
        if obj["meta"][i, G.MT_AGENT]:
            vx, vy = o.velocity.x, o.velocity.y
            ev = max(ev, abs(c * vx + s * vy - f[G.FI_VX]), abs(-s * vx + c * vy - f[G.FI_VY]))
        else:
            stat_v = max(stat_v, abs(f[G.FI_VX]), abs(f[G.FI_VY]))
    r.update(first_heading_err=eh, first_LW_err=float(elw), first_v_err=float(ev), first_class_mismatch=ecls,
             static_v_max=float(stat_v))
    return r


def _dense(poses8):
    """[8, 3] poses at 0.5..4 s -> [41, 3] at 0.1 s, linear in time through the origin (scorer interpolation)."""
    p = np.concatenate([np.zeros((1, 3)), np.asarray(poses8, np.float64)], 0)
    p[:, 2] = np.unwrap(p[:, 2])
    tk = np.arange(9) * 0.5
    return np.stack([np.interp(T41, tk, p[:, j]) for j in range(3)], -1)


def _footprints(dense):
    """ego corners at the 41 steps and their TTC projections -> list of (corners [T, 4, 2], t [T])."""
    import sf_common as SF
    x, y, h = dense[:, 0], dense[:, 1], dense[:, 2]
    out = [(SF.ego_corners(x, y, h, 0.0), T41)]
    vx, vy = np.gradient(x, 0.1), np.gradient(y, 0.1)
    for d in TTC_DT:
        out.append((SF.ego_corners(x + vx * d, y + vy * d, h, 0.0), T41 + d))
    return out


def reference_trajectories(obj, frames):
    """human / cv / su14 reference trajectories [8, 3] in N (see module docstring)."""
    trajs = {}
    n = int(obj["n_kf"])
    if n >= 9:
        trajs["human"] = np.asarray(obj["ego_kf"][1:9], np.float64)
    v0 = float(np.hypot(*np.asarray(frames[0]["ego_dynamic_state"][:2], np.float64)))
    tk = np.arange(1, 9) * 0.5
    trajs["cv"] = np.stack([v0 * tk, 0 * tk, 0 * tk], -1)
    poses = [G.frame_pose(f) for f in frames[:N_LOG_FRAMES]]
    xs, ys = _to_n([p[0] for p in poses], [p[1] for p in poses], *poses[0])
    hs = np.unwrap(G.wrap(np.array([p[2] for p in poses]) - poses[0][2]))
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    if len(s) >= 9 and s[-1] > 0:
        s_new = np.minimum(1.4 * s[1:9], s[-1])
        # np.interp needs increasing s: stationary segments -> tiny monotone jitter
        sm = s + np.arange(len(s)) * 1e-9
        trajs["su14"] = np.stack([np.interp(s_new, sm, xs), np.interp(s_new, sm, ys), np.interp(s_new, sm, hs)], -1)
    return trajs


def unknown_stats(obj, frames):
    """UNKNOWN counts per reference trajectory and definition, plus the headroom to the boundary:
    margin_reach = min over points of (REACH_M - |p - GT ego(t)|), margin_R = min of (R - RADIUS_MARGIN - |p|)."""
    r = {}
    R = float(obj["R"])
    for name, p8 in reference_trajectories(obj, frames).items():
        fps = _footprints(_dense(p8))
        m_reach = m_R = np.inf
        for corners, t in fps:
            ego = G.gt_ego_xy(obj, t)
            m_reach = min(m_reach, float((G.REACH_M - np.linalg.norm(corners - ego[:, None], axis=-1)).min()))
            m_R = min(m_R, float((R - G.RADIUS_MARGIN - np.linalg.norm(corners, axis=-1)).min()))
        r[f"unk_{name}_margin_reach"], r[f"unk_{name}_margin_R"] = m_reach, m_R
        for dname, marg in (("spec", None), ("ext", G.RADIUS_MARGIN)):
            n_pts = n_unk = 0
            for corners, t in fps:
                u = G.unknown_space(obj, np.transpose(corners, (1, 0, 2)), t, radius_margin=marg)   # [4, T]
                n_pts += u.size
                n_unk += int(u.sum())
            r[f"unk_{name}_{dname}_pts"] = n_unk
            r[f"unk_{name}_{dname}_any"] = int(n_unk > 0)
        r[f"unk_{name}_npts"] = n_pts
    return r


# ------------------------------------------------------------------------------------------------ worker
def process_log(task):
    log, rows, cfg = task
    out = Path(cfg["out"])
    res = []
    todo = [(t, fi) for t, fi in rows if cfg["overwrite"] or cfg["mc_root"] or not (out / f"{t}.npz").exists()]
    if not todo:
        return [dict(token=t, log=log, status="exists") for t, _ in rows]
    t_load = time.perf_counter()
    try:
        with open(Path(cfg["logs"]) / f"{log}.pkl", "rb") as f:
            d = pickle.load(f)
    except Exception as e:  # noqa: BLE001
        return [dict(token=t, log=log, status="error", error=f"log load: {e!r}") for t, _ in rows]
    t_load = time.perf_counter() - t_load
    pos = {f["token"]: i for i, f in enumerate(d)}
    done = {t for t, _ in todo}
    res += [dict(token=t, log=log, status="exists") for t, _ in rows if t not in done]
    for tok, fi in todo:
        row = dict(token=tok, log=log, log_load_s=t_load / len(todo))
        try:
            i = pos.get(tok)
            if i is None:
                raise KeyError("token not in log")
            row["log_pos"] = i
            row["frame_idx_mismatch"] = int(fi is not None and fi >= 0 and int(d[i].get("frame_idx", fi)) != fi)
            frames = d[i: i + N_LOG_FRAMES]
            path = out / f"{tok}.npz"
            if path.exists() and not cfg["overwrite"]:
                obj = G.load_objects(path)
                row["status"] = "exists"
            else:
                t0 = time.perf_counter()
                obj = G.build_from_frames(frames)
                G.save_objects(path, obj)
                row["build_s"] = time.perf_counter() - t0
                row["status"] = "built"
                obj = G.load_objects(path)                     # validate exactly what is stored
            meta = obj["meta"]
            row.update(A=int(meta.shape[0]), n_tracks_all=int(obj["n_tracks_all"]), n_kf=int(obj["n_kf"]),
                       dt_max=float(obj["dt_max"]), S_avail=float(obj["S_avail"]), R=float(obj["R"]),
                       n_dup=int(obj["n_dup"]), n_single=int(meta[:, G.MT_SINGLE].sum()) if len(meta) else 0,
                       n_agent=int(meta[:, G.MT_AGENT].sum()) if len(meta) else 0,
                       npz_bytes=int(path.stat().st_size))
            if cfg["mc_root"]:
                t0 = time.perf_counter()
                mc = load_mc(cfg["mc_root"], log, tok)
                row["mc_load_s"] = time.perf_counter() - t0
                if mc is None:
                    row["mc_missing"] = 1
                else:
                    row["mc_missing"] = 0
                    row.update(validate_mc(obj, mc))
                row.update(unknown_stats(obj, frames))
        except Exception as e:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = f"{e!r} {traceback.format_exc(limit=3)}"
        res.append(row)
    return res


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ------------------------------------------------------------------------------------------------ summary
def summarize(df: pd.DataFrame) -> dict:
    v = df[(df.status != "error") & (df["mc_missing"] == 0)] if "mc_missing" in df else df.iloc[:0]
    s = dict(n_tokens=int(len(df)), n_error=int((df.status == "error").sum()), n_built=int((df.status == "built").sum()))
    if "build_s" in df and df.build_s.notna().any():
        b = df.build_s.dropna()
        s.update(build_s_mean=float(b.mean()), build_s_p50=float(b.median()), build_s_p95=float(b.quantile(.95)),
                 log_load_s_per_token_mean=float(df.log_load_s.dropna().mean()))
    if "A" in df:
        a = df.A.dropna()
        s.update(A_mean=float(a.mean()), A_p95=float(a.quantile(.95)), A_max=int(a.max()),
                 R_mean=float(df.R.mean()), R_max=float(df.R.max()), frac_R_gt_80=float((df.R > 80).mean()),
                 tokens_gap=int((df.dt_max > 0.75).sum()), n_dup_total=int(df.n_dup.sum()),
                 npz_kb_mean=float(df.npz_bytes.mean() / 1024))
    if len(v):
        s.update(val_tokens=int(len(v)), val_max_corner_err=float(v.max_corner_err.max()),
                 val_pairs_obs=int(v.n_pairs_obs.sum()), val_pairs=int(v.n_pairs.sum()),
                 val_presence_mismatch=int(v.presence_mismatch.sum()),
                 val_tokens_presence_identical=int((v.presence_mismatch == 0).sum()),
                 val_mine_not_in_mc=int(v.mine_not_in_mc.sum()), val_red_light_tokens=int(v.red_light_tokens.sum()),
                 val_mc_tracks=int(v.n_mc_tracks.sum()), val_excl_tracks=int(v.n_excl.sum()),
                 val_excl_unexplained=int(v.excl_unexplained.sum()),
                 val_excl_min_d_ego=float(v.excl_min_d_ego.replace(np.inf, np.nan).min()),
                 val_excl_min_d_origin=float(v.excl_min_d_origin.replace(np.inf, np.nan).min()),
                 val_excl_min_corner_minus_R=float(v.excl_min_corner_minus_R.replace(np.inf, np.nan).min()),
                 val_pose0_err=float(v.pose0_err.max()), val_first_heading_err=float(v.first_heading_err.max()),
                 val_first_LW_err=float(v.first_LW_err.max()), val_first_v_err=float(v.first_v_err.max()),
                 val_first_class_mismatch=int(v.first_class_mismatch.sum()), val_static_v_max=float(v.static_v_max.max()),
                 mc_load_s_mean=float(v.mc_load_s.mean()))
        unk = {}
        for name in ("human", "cv", "su14"):
            if f"unk_{name}_npts" not in v:
                continue
            w = v[v[f"unk_{name}_npts"].notna()]
            for dn in ("spec", "ext"):
                unk[f"{name}_{dn}_token_rate"] = float(w[f"unk_{name}_{dn}_any"].mean())
                unk[f"{name}_{dn}_point_rate"] = float(w[f"unk_{name}_{dn}_pts"].sum() / w[f"unk_{name}_npts"].sum())
            unk[f"{name}_n_tokens"] = int(len(w))
            for mk in ("margin_reach", "margin_R"):
                col = w[f"unk_{name}_{mk}"]
                unk[f"{name}_{mk}_min"] = float(col.min())
                unk[f"{name}_{mk}_p1"] = float(col.quantile(0.01))
        s["unknown"] = unk
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", required=True, help="parquet with columns token, log[, frame_idx]")
    ap.add_argument("--logs", required=True, help="dir with <log>.pkl raw NAVSIM logs")
    ap.add_argument("--out", required=True, help="output dir (<token>.npz, index.parquet)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tokens (after --sample)")
    ap.add_argument("--sample", type=int, default=0, help="random sample of N tokens (seed --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--mc-root", default="", help="validate against this metric cache root (+ UNKNOWN rates)")
    ap.add_argument("--report", default="", help="summary json path (default <out>/summary[_validation].json)")
    a = ap.parse_args(argv)
    assert 1 <= a.workers <= 4, "machine is shared: <= 4 workers"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tk = pd.read_parquet(a.tokens)
    if a.sample:
        tk = tk.sample(min(a.sample, len(tk)), random_state=a.seed)
    if a.limit:
        tk = tk.iloc[: a.limit]
    has_fi = "frame_idx" in tk
    cfg = dict(out=str(out), logs=a.logs, overwrite=a.overwrite, mc_root=a.mc_root)
    tasks = [(log, [(r.token, int(r.frame_idx) if has_fi else None) for r in g.itertuples()], cfg)
             for log, g in tk.groupby("log", sort=True)]
    tasks.sort(key=lambda x: -len(x[1]))
    t0 = time.time()
    print(f"[build_future_objects] {len(tk)} tokens / {len(tasks)} logs -> {out} workers={a.workers} "
          f"mc_root={a.mc_root or '-'}", flush=True)
    rows, n = [], 0
    with Pool(a.workers, initializer=_init) as pool:
        for res in pool.imap_unordered(process_log, tasks):
            rows += res
            n += len(res)
            err = sum(r.get("status") == "error" for r in res)
            print(f"  {n}/{len(tk)} tokens  {time.time() - t0:.0f}s  errors_in_chunk={err}", flush=True)
    df = pd.DataFrame(rows)
    idx_path = out / "index.parquet"
    new = df[df.status != "exists"] if not a.mc_root else df
    if idx_path.exists():
        old = pd.read_parquet(idx_path)
        new = pd.concat([old[~old.token.isin(new.token)], new], ignore_index=True)
    if len(new):
        new.to_parquet(idx_path, index=False)
    s = summarize(df)
    s.update(wall_s=time.time() - t0, workers=a.workers, tokens_file=a.tokens, logs=a.logs, mc_root=a.mc_root,
             version=G.VERSION)
    rep = Path(a.report) if a.report else out / ("summary_validation.json" if a.mc_root else "summary.json")
    rep.write_text(json.dumps(s, indent=1, default=float))
    if a.mc_root:
        df.to_parquet(rep.with_suffix(".parquet"), index=False)
    print(json.dumps(s, indent=1, default=float), flush=True)
    errs = df[df.status == "error"]
    for r in errs.head(5).itertuples():
        print("ERROR", r.token, r.error, flush=True)
    return s


if __name__ == "__main__":
    main()
