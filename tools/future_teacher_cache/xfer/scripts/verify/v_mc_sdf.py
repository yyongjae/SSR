"""Per sampled token: metric cache readable (xz + unpickle), structural sanity vs the raw log, and the stored SDF
re-derived bit-exactly from the metric cache (build_sdf_from_metric_cache)."""
import json, os, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools/refiner"))
import build_metric_cache as BMC
from navsim.agents.para_ssr.refiner import sdf as S

HERE = Path(__file__).parent
MC = Path("/home/external-user/ssd/yongjae_refiner/metric_cache")
LOGS = ROOT / "data/dataset/navsim_logs/trainval"


def one_log(task):
    import extract_human as EH
    from navsim.agents.para_ssr.refiner import surrogate as SU
    log, toks = task
    frames = pickle.load(open(LOGS / f"{log}.pkl", "rb"))
    pos = {f["token"]: i for i, f in enumerate(frames)}
    out = []
    for tok in toks:
        r = dict(token=tok, log=log)
        try:
            p = BMC.mc_path(MC, log, tok)
            r["xz_ok"] = BMC.xz_ok(p)
            mc = BMC.load_mc(p)
            ra = mc.ego_state.rear_axle
            h = EH.extract_token(frames, pos[tok])
            g = h["ego_global"]
            r["ego_vs_log"] = float(max(abs(ra.x - g[0]), abs(ra.y - g[1]), abs(np.angle(np.exp(1j * (ra.heading - g[2]))))))
            r["v0_vs_log"] = float(abs(mc.ego_state.dynamic_car_state.speed - h["v0"]))
            r["time_us_vs_log"] = int(mc.ego_state.time_us - frames[pos[tok]]["timestamp"])
            r["n_occ_maps"] = len(mc.observation._occupancy_maps)
            r["n_unique"] = len(mc.observation.unique_objects)
            r["n_pdm_states"] = len(mc.trajectory.get_sampled_trajectory())
            r["n_centerline"] = int(mc.centerline._states_se2_array.shape[0])
            r["n_route"] = len(mc.route_lane_ids)
            r["n_drivable"] = len(mc.drivable_area_map._geometries)
            r["drivable_types"] = sorted({str(t) for t in mc.drivable_area_map._map_types})
            r["log_roadblock_route_nonempty"] = len(frames[pos[tok]]["roadblock_ids"]) > 0
            cl = SU.centerline_from_metric_cache(mc)
            r["cl_n"] = int(np.asarray(cl).shape[0]); r["cl_finite"] = bool(np.isfinite(np.asarray(cl)).all())
            # SDF bit-exact re-derivation
            sp = S.sdf_path(tok, "navtrain")
            stored = S.load_sdf(sp)
            with np.load(sp) as z:
                r["sdf_ego_xyh_diff"] = float(np.abs(z["ego_xyh"] - np.array([ra.x, ra.y, ra.heading])).max())
                r["sdf_version"] = str(z["version"]); r["sdf_token_ok"] = str(z["token"]) == tok
            new, _ = S.build_sdf_from_metric_cache(mc)
            r["sdf_bit_equal"] = bool(np.array_equal(np.asarray(new, np.float32).astype(np.float16).view(np.uint16), stored.view(np.uint16)))
            r["sdf_maxdiff"] = float(np.abs(new.astype(np.float32) - stored.astype(np.float32)).max())
            r["sdf_frac_inside"] = float((stored > 0).mean())
            r["ok"] = True
        except Exception as e:
            import traceback
            r.update(ok=False, error=repr(e) + traceback.format_exc()[-400:])
        out.append(r)
    return out


if __name__ == "__main__":
    s = pd.read_parquet(HERE / "sample100.parquet")
    tasks = [(lg, list(g.token)) for lg, g in s.groupby("log")]
    t0 = time.time()
    rows = []
    with Pool(4) as p:
        for r in p.imap_unordered(one_log, tasks):
            rows += r
    df = pd.DataFrame(rows)
    df.to_parquet(HERE / "mc_sdf_rows.parquet", index=False)
    print("wall", time.time() - t0)
    print(df.drop(columns=[c for c in ("error", "drivable_types") if c in df]).describe().T.to_string())
    for c in ("ok", "xz_ok", "sdf_bit_equal", "log_roadblock_route_nonempty", "cl_finite"):
        print(c, df[c].value_counts(dropna=False).to_dict())
    print("types", df.drivable_types.astype(str).value_counts().to_dict())
    print("versions", df.sdf_version.value_counts().to_dict())
    if "error" in df:
        print(df[~df.ok][["token", "error"]].to_string())
