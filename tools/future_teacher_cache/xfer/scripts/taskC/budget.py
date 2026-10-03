"""Task C: disk/time budget for 'stream -> extract needed -> infer -> delete' (read-only, CPU).
Inputs: ../frames_avail.parquet (all trainval/test frames, on-disk flags, cache flags; built by ../disk_index.py),
        ../hf/map_trainval.json (archive -> logs), ../hf/sizes.json (tgz bytes), splits parquets.
Output: budget.json"""
import json, os, numpy as np, pandas as pd
S = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(S)
SP = "/home/external-user/ssd/yongjae_refiner/splits"
df = pd.read_parquet(f"{P}/frames_avail.parquet")
df = df[df.split == "trainval"].sort_values(["log", "idx"]).reset_index(drop=True)
amap = json.load(open(f"{P}/hf/map_trainval.json")); sizes = json.load(open(f"{P}/hf/sizes.json"))
log2arc = {l: int(k.rsplit("_", 1)[1]) for k, v in amap.items() for l in v}
df["arc"] = df.log.map(log2arc).astype(int)
N = len(df); pos = np.arange(N); idx = df.idx.values
logn = df.groupby("log").idx.transform("size").values
def shift(k):
    ok = (idx + k >= 0) & (idx + k < logn); return np.where(ok, pos + k, -1)
SH = {k: shift(k) for k in range(-2, 11)}
cam3 = (df.d_CAM_F0 & df.d_CAM_L0 & df.d_CAM_R0).values; lid = df.d_lidar.values
bev = df.c_bev_train.values | df.c_bev.values
arc = df.arc.values
# per-file sizes [measured, task A archive-0 listing]
F0, L0, R0, PCD = 220578, 201296, 203852, 1403088
CAM3 = F0 + L0 + R0; CAM8 = 3091000 - PCD  # 8-cam total from task A (3.091 MB/frame incl. lidar)
# per-archive scaled sizes from tgz bytes (camera ratio 1.003, lidar 1.134; task A)
fr_per_arc = np.bincount(arc, minlength=200)
cam8_arc = np.array([sizes["cam"][str(i)] * 1.003 / fr_per_arc[i] for i in range(200)])
pcd_arc = np.array([sizes["lidar"][str(i)] * 1.134 / fr_per_arc[i] for i in range(200)])
cam3_arc = cam8_arc * CAM3 / CAM8
NPZ_BEV, NPZ_DROP = 2648014, 87886
tok2row = pd.Series(pos, index=df.token)
def rows(name): return tok2row.reindex(pd.read_parquet(f"{SP}/{name}.parquet").token).dropna().astype(int).values
navtrain_tok = np.where(df.c_bev_train.values)[0]
navtrain_logs = set(df.log.values[navtrain_tok])
sets_tok = {"stageT": np.union1d(rows("train_trainlogs"), rows("dev_trainlogs")), "e2e": rows("e2e_train_trainlogs"), "navtrain": navtrain_tok}
def run_set(toks, H):
    F = set()
    for k in range(1, 2 * H + 1):
        r = SH[k][toks]; F.update(r[r >= 0].tolist())
    F = np.array(sorted(F), dtype=int); return F[~bev[F]]
RUN = {f"{n}_4s": run_set(t, 4) for n, t in sets_tok.items()}
RUN["navtrain_5s"] = run_set(navtrain_tok, 5)
allnav = np.where(df.log.isin(navtrain_logs).values)[0]; RUN["navtrain_logs_all_frames"] = allnav[~bev[allnav]]
RUN["trainval_all_frames"] = pos[~bev]
def need(run, scen):
    """bytes per archive of files to extract. scen S1: navtrain current+history sensors already on disk; S2: no sensors on disk."""
    camB = np.zeros(200); lidB = np.zeros(200); camB_s = np.zeros(200); lidB_s = np.zeros(200)
    need_cam = np.zeros(N, bool); need_lid = np.zeros(N, bool)
    have_c = cam3 if scen == "S1" else np.zeros(N, bool); have_l = lid if scen == "S1" else np.zeros(N, bool)
    need_cam[run] = ~have_c[run]; need_lid[run] = ~have_l[run]
    for k in (1, 2):
        j = SH[-k][run]; j = j[j >= 0]; need_lid[j[~have_l[j]]] = True
    nc, nl = np.where(need_cam)[0], np.where(need_lid)[0]
    camB = np.bincount(arc[nc], minlength=200) * CAM3; lidB = np.bincount(arc[nl], minlength=200) * PCD
    camB_s = np.bincount(arc[nc], weights=cam3_arc[arc[nc]], minlength=200); lidB_s = np.bincount(arc[nl], weights=pcd_arc[arc[nl]], minlength=200)
    sweep_only = int(need_lid.sum() - (need_lid & np.isin(pos, run)).sum())
    return dict(n_cam_frames=len(nc), n_lid_frames=len(nl), n_sweep_only_lidar=sweep_only), camB + lidB, camB_s + lidB_s
def groups(per, g):
    gs = np.array([per[i:i + g].sum() for i in range(0, 200, g)])
    pair = max(gs[i] + gs[i + 1] for i in range(len(gs) - 1)) if len(gs) > 1 else gs[0]
    return gs.max(), pair
out = {}
GS = (1, 2, 5, 10, 20, 40, 100, 200)
for name, run in RUN.items():
    o = dict(run_frames=len(run), run_logs=int(len(set(df.log.values[run]))), run_archives=int(len(set(arc[run]))),
             npz_drop_bev_GB=len(run) * NPZ_DROP / 1e9, npz_bev_GB=len(run) * NPZ_BEV / 1e9,
             gpu_h_1gpu_11p6=len(run) / 11.6 / 3600, gpu_h_1gpu_5p8=len(run) / 5.8 / 3600, gpu_h_4gpu_11p6=len(run) / 46.4 / 3600)
    for scen in ("S1", "S2"):
        info, per, per_s = need(run, scen)
        d = dict(info, total_GB=per.sum() / 1e9, total_GB_arcscaled=per_s.sum() / 1e9)
        for g in GS:
            m, pr = groups(per, g); ms, prs = groups(per_s, g)
            d[f"g{g}"] = dict(peak_serial_GB=m / 1e9, peak_overlap2_GB=pr / 1e9, peak_serial_GB_arcscaled=ms / 1e9, peak_overlap2_GB_arcscaled=prs / 1e9)
        o[scen] = d
    out[name] = o
    print(name, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in o.items() if not isinstance(v, dict)}),
          "| S1 total", round(o["S1"]["total_GB"], 1), "g1", round(o["S1"]["g1"]["peak_serial_GB"], 2), "g10", round(o["S1"]["g10"]["peak_serial_GB"], 1), "g40", round(o["S1"]["g40"]["peak_serial_GB"], 1),
          "| S2 total", round(o["S2"]["total_GB"], 1), "g1", round(o["S2"]["g1"]["peak_serial_GB"], 2), "g10", round(o["S2"]["g10"]["peak_serial_GB"], 1), "g40", round(o["S2"]["g40"]["peak_serial_GB"], 1), flush=True)
# extra info
out["meta"] = dict(trainval_frames=int(N), trainval_logs=int(df.log.nunique()), navtrain_logs=len(navtrain_logs),
    navtrain_logs_frames=int(len(allnav)), cached_current_npz_train=int(bev.sum()),
    ondisk_frames_navtrain_pkg=int((cam3 & lid).sum()), ondisk_8cam_lidar_GB_est=float((cam3 & lid).sum() * 3.091e6 / 1e9),
    frames_not_on_disk=int((~(cam3 & lid)).sum()),
    keep_front3_all_trainval_not_on_disk_GB=float((~cam3).sum() * CAM3 / 1e9),
    keep_front3_navtrain_logs_not_on_disk_GB=float((~cam3[allnav]).sum() * CAM3 / 1e9),
    P_full_extracted_GB=float(sum(sizes["cam"].values()) * 1.003 / 1e9 + sum(sizes["lidar"].values()) * 1.134 / 1e9),
    P_full_largest_tgz_GB=float(max(max(sizes["cam"].values()), max(sizes["lidar"].values())) / 1e9),
    tgz_total_GB=float((sum(sizes["cam"].values()) + sum(sizes["lidar"].values())) / 1e9),
    group_tgz_GB={f"g{g}": float(max(sum(sizes["cam"][str(i)] + sizes["lidar"][str(i)] for i in range(s, min(200, s + g))) for s in range(0, 200, g)) / 1e9) for g in GS})
print(json.dumps(out["meta"], indent=1))
json.dump(out, open(f"{S}/budget.json", "w"), indent=1)
