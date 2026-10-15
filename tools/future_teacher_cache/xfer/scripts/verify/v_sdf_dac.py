"""SDF vs official DAC on 30 new tokens x (13-draft bank of make_draft_bank + 2 lateral-shift probes), following
validate_sdf.py conventions (tracked = scorer's 41 LQR corners; raw = linear-interp reference corners)."""
import json, os, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path("/home/external-user/yongjae/SSR")
for p in (ROOT, ROOT / "tools/refiner"):
    sys.path.insert(0, str(p))
import numpy as np
import pandas as pd
import validate_sdf as VS          # adds the cf_common / sf_common paths
import build_metric_cache as BMC
from navsim.agents.para_ssr.refiner import sdf as S

HERE = Path(__file__).parent
LOGS = ROOT / "data/dataset/navsim_logs/trainval"
NTOK = 30


def one_log(task):
    import shapely, torch
    import sf_common as SF
    import extract_human as EH
    import make_draft_bank as MB
    torch.set_num_threads(1)
    CF = VS.G["CF"]
    log, toks = task
    frames = pickle.load(open(LOGS / f"{log}.pkl", "rb"))
    pos = {f["token"]: i for i, f in enumerate(frames)}
    rows, errs = [], []
    for tok in toks:
        try:
            mc = BMC.load_mc(BMC.mc_path(BMC.OUT, log, tok))
            ra = mc.ego_state.rear_axle
            xyh = (float(ra.x), float(ra.y), float(ra.heading))
            f16 = S.load_sdf(S.sdf_path(tok, "navtrain"))
            f16_t = torch.from_numpy(f16)
            union, _ = S.drivable_union(mc.drivable_area_map, xyh, crop=False)
            shapely.prepare(union)
            h = EH.extract_token(frames, pos[tok])
            bank = MB.make_bank(h["traj"], tok, path_long=h["path"], n_valid=int(h["n_reg"]), v0=float(h["v0"]),
                                a0=float(h["a0"]), centerline=lambda: MB.centerline_n(mc))
            trajs = {f"k{k:02d}_{int(bank['family'][k])}": bank["drafts"][k] for k in range(len(bank["drafts"]))}
            t = np.arange(1, 9) * 0.5
            v = float(h["v0"])
            for side, sgn in (("latL", 1), ("latR", -1)):
                y = sgn * 3.5 * (t / 4.0) ** 2
                trajs[side] = np.stack([v * t, y, np.arctan2(sgn * 7.0 * t / 16.0, max(v, 0.5))], -1).astype(np.float32)
            for name, poses in trajs.items():
                res, _, _, _ = CF.score(mc, np.asarray(poses, np.float64))
                tracked = S.global_to_n(np.asarray(CF.G["scorer"]._ego_coords[1][:, :4], np.float64), xyh)
                P = SF.Path(np.asarray(poses, np.float64))
                xr, yr, hr = P.ref_poses_time()
                raw = SF.ego_corners(xr, yr, hr, 0.0)
                rec = dict(token=tok, log=log, traj=name, random=True, frame_gap=bool(h["frame_gap"]),
                           dac_official=float(res["dac"]), dac_table=np.nan, ok=True)
                for kind, C in (("tracked", tracked), ("raw", raw)):
                    r, e = VS._corner_stats(C, f16_t, union, kind)
                    rec.update(r)
                    errs.append(np.concatenate([e, np.full((len(e), 1), 0 if kind == "tracked" else 1)], 1))
                rows.append(rec)
        except Exception as ex:
            import traceback
            rows.append(dict(token=tok, log=log, traj="?", ok=False, error=repr(ex) + traceback.format_exc()[-500:]))
    return rows, (np.concatenate(errs) if errs else np.zeros((0, 3)))


if __name__ == "__main__":
    s = pd.read_parquet(HERE / "sample100.parquet").iloc[:NTOK]
    tasks = [(lg, list(g.token)) for lg, g in s.groupby("log")]
    t0 = time.time()
    rows, errs = [], []
    with Pool(4, initializer=VS._init) as p:
        for r, e in p.imap_unordered(one_log, tasks):
            rows += r; errs.append(e)
    df = pd.DataFrame(rows)
    df.to_parquet(HERE / "sdf_dac_rows.parquet", index=False)
    errs = np.concatenate(errs)
    np.savez_compressed(HERE / "sdf_dac_errs.npz", err=errs.astype(np.float32))
    if "error" in df and (~df.ok).any():
        print(df[~df.ok][["token", "error"]].to_string())
    df["orig_dummy"] = 0
    ok = df[df.ok]
    out = dict(n_tokens=int(ok.token.nunique()), n_rows=int(len(ok)), n_err_tokens=int((~df.ok).sum()), wall_s=time.time() - t0)
    y = ok.dac_official.values < 1
    out["official_fail"] = int(y.sum())
    out["official_vs_exact_union_tracked_mismatch"] = int((y != ok.tracked_exact_fail.values).sum())
    out["torch_vs_numpy_maxdiff"] = float(max(ok.tracked_torch_np_maxdiff.max(), ok.raw_torch_np_maxdiff.max()))
    out["tracked_n_oog_total"] = int(ok.tracked_n_oog.sum()); out["raw_n_oog_total"] = int(ok.raw_n_oog.sum())
    for kind in ("tracked", "raw"):
        out[kind] = dict(sdf=VS._conf(y, ok[f"{kind}_sdf_min"].values < 0),
                         exact_union=VS._conf(y, ok[f"{kind}_exact_fail"].values),
                         sdf_vs_exact_union=VS._conf(ok[f"{kind}_exact_fail"].values, ok[f"{kind}_sdf_min"].values < 0),
                         auc_sdf=VS._auc(y, -ok[f"{kind}_sdf_min"].values),
                         margins={str(m): VS._conf(y, ok[f"{kind}_sdf_min"].values < m) for m in VS.MARGINS})
        dis = ok[(ok[f"{kind}_sdf_min"].values < 0) != y]
        out[kind]["disagreements"] = [dict(token=r.token, traj=r.traj, official_fail=bool(r.dac_official < 1),
                                           sdf_min=round(float(r[f"{kind}_sdf_min"]), 4),
                                           exact_min=round(float(r[f"{kind}_exact_min"]), 4),
                                           n_oog=int(r[f"{kind}_n_oog"])) for _, r in dis.iterrows()][:30]
    acc = {}
    for kid, kind in ((0, "tracked"), (1, "raw")):
        e = errs[errs[:, 2] == kid]
        ae = np.abs(e[:, 1] - e[:, 0])
        clear = np.abs(e[:, 0]) > 0.05
        acc[kind] = dict(n_corners=int(len(e)), abs_err_median=float(np.median(ae)), abs_err_p99=float(np.quantile(ae, 0.99)),
                         abs_err_max=float(ae.max()), bias=float((e[:, 1] - e[:, 0]).mean()),
                         sign_mismatch_clear=int((np.sign(e[:, 1]) != np.sign(e[:, 0]))[clear].sum()),
                         n_within_5cm=int((~clear).sum()))
    out["corner_accuracy_abs_d_lt_2m"] = acc
    json.dump(out, open(HERE / "sdf_dac_summary.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
