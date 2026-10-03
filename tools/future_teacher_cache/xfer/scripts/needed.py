import glob, json, os, pickle, collections
import pandas as pd
S = os.path.dirname(os.path.abspath(__file__))
LOGD = "/home/external-user/navsim/download/trainval_navsim_logs/trainval"
BLOB = "/home/external-user/navsim/download/trainval_sensor_blobs/trainval"
CACHE = "/home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100"
SP = "/home/external-user/ssd/yongjae_refiner/splits"
W = "/home/external-user/yongjae/SSR/tools/future_teacher_cache"
cached = set(os.path.basename(p)[:-4] for p in glob.glob(f"{CACHE}/samples/*/*.npz"))
idx = json.load(open(f"{W}/future_index_train.json"))
amap = json.load(open(f"{S}/hf/map_trainval.json"))
log2arc = {l: int(k.rsplit('_', 1)[1]) for k, v in amap.items() for l in v}
sizes = json.load(open(f"{S}/hf/sizes.json"))
# per-log frames
cache_f = f"{S}/logframes.pkl"
if os.path.exists(cache_f):
    LF = pickle.load(open(cache_f, "rb"))
else:
    LF = {}
    for p in sorted(glob.glob(f"{LOGD}/*.pkl")):
        log = os.path.basename(p)[:-4]
        fr = pickle.load(open(p, "rb"))
        LF[log] = [(f["token"], os.path.basename(f["lidar_path"]) if f.get("lidar_path") else None,
                    {c: os.path.basename(f["cams"][c]["data_path"]) for c in ("CAM_F0", "CAM_L0", "CAM_R0")}) for f in fr]
    pickle.dump(LF, open(cache_f, "wb"))
tok2 = {}
for log, fr in LF.items():
    for i, f in enumerate(fr): tok2[f[0]] = (log, i)
disk = {}
def ondisk(log):
    if log not in disk:
        d = {}
        for sub in ("CAM_F0", "CAM_L0", "CAM_R0", "MergedPointCloud"):
            try: d[sub] = set(os.listdir(f"{BLOB}/{log}/{sub}"))
            except FileNotFoundError: d[sub] = set()
        disk[log] = d
    return disk[log]
def cams_ok(log, i):
    f = LF[log][i]; d = ondisk(log)
    return all(f[2][c] in d[c] for c in ("CAM_F0", "CAM_L0", "CAM_R0"))
def lidar_ok(log, i):
    f = LF[log][i]; return f[1] is not None and f[1] in ondisk(log)["MergedPointCloud"]
sets = {
  "stageT": set(pd.read_parquet(f"{SP}/train_trainlogs.parquet").token) | set(pd.read_parquet(f"{SP}/dev_trainlogs.parquet").token),
  "e2e": set(pd.read_parquet(f"{SP}/e2e_train_trainlogs.parquet").token),
  "navtrain": set(idx),
}
res = {}
for name, toks in sets.items():
    for H in (4, 2):
        need = set()
        for t in toks:
            fut = idx.get(t)
            if fut is None: continue
            for f in fut[:2 * H]:
                if f is not None and f not in cached: need.add(f)
        run = need
        frames_disk = {f for f in run if cams_ok(*tok2[f]) and lidar_ok(*tok2[f])}
        dl = run - frames_disk
        # files needed: cams for dl frames (if missing), lidar for dl frames, sweep lidar f-1,f-2 for all run frames
        cam_logs, lid_logs = collections.Counter(), collections.Counter()
        n_cam_files = n_lid_files = 0; sweep_extra = set()
        for f in run:
            log, i = tok2[f]
            if not cams_ok(log, i): cam_logs[log] += 1
            lid_need = set()
            if not lidar_ok(log, i): lid_need.add(i)
            for k in (1, 2):
                j = i - k
                if j >= 0 and LF[log][j][1] is not None and not lidar_ok(log, j): lid_need.add(j)
            for j in lid_need:
                if j != i and LF[log][j][0] not in run: sweep_extra.add((log, j))
                lid_logs[log] += 1
        sweep_extra_notdl = {x for x in sweep_extra}
        ca = sorted({log2arc[l] for l in cam_logs}); la = sorted({log2arc[l] for l in lid_logs})
        res[f"{name}_{H}s"] = dict(run=len(run), on_disk=len(frames_disk), dl=len(dl),
            run_frames_missing_some_sweep=sum(1 for f in frames_disk if any((tok2[f][1]-k)>=0 and LF[tok2[f][0]][tok2[f][1]-k][1] and not lidar_ok(tok2[f][0], tok2[f][1]-k) for k in (1,2))),
            sweep_frames_outside_run=len(sweep_extra),
            sweep_frames_outside_run_and_outside_dl_cams=len(sweep_extra),
            cam_logs=len(cam_logs), lidar_logs=len(lid_logs),
            cam_archives=len(ca), lidar_archives=len(la),
            cam_GB=sum(sizes["cam"][str(a)] for a in ca) / 1e9, lidar_GB=sum(sizes["lidar"][str(a)] for a in la) / 1e9,
            union_archives=len(set(ca) | set(la)), lidar_archives_not_in_cam=len(set(la) - set(ca)))
        res[f"{name}_{H}s"]["total_GB"] = res[f"{name}_{H}s"]["cam_GB"] + res[f"{name}_{H}s"]["lidar_GB"]
        if H == 4:
            json.dump({"cam": ca, "lidar": la}, open(f"{S}/archives_{name}_4s.json", "w"))
            # archives NOT needed
        print(name, H, json.dumps(res[f"{name}_{H}s"]), flush=True)
json.dump(res, open(f"{S}/needed_result.json", "w"), indent=1)
