import json, pickle, os, sys, collections, re
S = os.path.dirname(os.path.abspath(__file__))
LF = pickle.load(open(f"{S}/logframes.pkl", "rb"))
amap = json.load(open(f"{S}/hf/map_trainval.json")); sizes = json.load(open(f"{S}/hf/sizes.json"))
LOGD = "/home/external-user/navsim/download/trainval_navsim_logs/trainval"
I = int(sys.argv[1]) if len(sys.argv) > 1 else 0
mlogs = amap[f"openscene_sensor_trainval_{I}"]
full = {}
for log in mlogs:
    fr = pickle.load(open(f"{LOGD}/{log}.pkl", "rb"))
    full[log] = fr
res = {}
for mod in ("camera", "lidar"):
    p = f"{S}/full/list_{mod}_{I}.txt"
    if not os.path.exists(p): continue
    files = {}; dirs = 0; prefixes = collections.Counter(); owners = collections.Counter()
    for line in open(p):
        parts = line.split(None, 6)  # perms owner size date time path  (with --full-time: date time)
        perms, owner, size, date, time_, path = parts[0], parts[1], int(parts[2]), parts[3], parts[4], parts[5].strip()
        owners[owner] += 1
        if perms.startswith('d'): dirs += 1; continue
        seg = path.split('/'); prefixes['/'.join(seg[:3])] += 1
        files[path] = size
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    for path, size in files.items():
        seg = path.split('/'); by[seg[3]][seg[4]][seg[5]] = size
    r = dict(n_files=len(files), n_dirs=dirs, prefixes=dict(prefixes), owners=dict(owners), logs=sorted(by), uncompressed_total=sum(files.values()),
             compressed=sizes['cam' if mod == 'camera' else 'lidar'][str(I)])
    r['ratio_uncomp_over_comp'] = r['uncompressed_total'] / r['compressed']
    r['logs_match_map'] = sorted(by) == sorted(mlogs)
    sub_sizes = collections.defaultdict(list); comp = {}
    for log in by:
        fr = full.get(log)
        if fr is None: comp[log] = 'NOT IN MAP'; continue
        c = {}
        if mod == "camera":
            for cam in ("CAM_F0","CAM_L0","CAM_R0","CAM_L1","CAM_R1","CAM_L2","CAM_R2","CAM_B0"):
                exp = {os.path.basename(f["cams"][cam]["data_path"]) for f in fr}
                have = set(by[log].get(cam, {}))
                c[cam] = dict(exp=len(exp), have=len(have), missing=len(exp - have), extra=len(have - exp))
                sub_sizes[cam] += [by[log][cam][x] for x in have & exp]
        else:
            exp = {os.path.basename(f["lidar_path"]) for f in fr if f.get("lidar_path")}
            have = set(by[log].get("MergedPointCloud", {}))
            c["MergedPointCloud"] = dict(exp=len(exp), have=len(have), missing=len(exp - have), extra=len(have - exp))
            sub_sizes["MergedPointCloud"] += [by[log]["MergedPointCloud"][x] for x in have & exp]
            c['subdirs'] = sorted(by[log])
        comp[log] = c
    r['completeness'] = comp
    r['mean_member_bytes'] = {k: sum(v) / len(v) for k, v in sub_sizes.items() if v}
    res[mod] = r
if "camera" in res and "lidar" in res:
    res['same_logs_cam_lidar'] = res['camera']['logs'] == res['lidar']['logs']
    mb = {**res['camera']['mean_member_bytes'], **res['lidar']['mean_member_bytes']}
    if len(mb) == 9:
        res['per_frame_bytes_F0L0R0_lidar'] = mb['CAM_F0'] + mb['CAM_L0'] + mb['CAM_R0'] + mb['MergedPointCloud']
        res['per_frame_bytes_8cam_lidar'] = sum(mb.values())
        res['per_frame_bytes_F0L0R0'] = mb['CAM_F0'] + mb['CAM_L0'] + mb['CAM_R0']
json.dump(res, open(f"{S}/listing_result_{I}.json", "w"), indent=1)
print(json.dumps({k: (v if k not in ('camera','lidar') else {kk: vv for kk, vv in v.items() if kk != 'completeness'}) for k, v in res.items()}, indent=1))
for mod in ('camera','lidar'):
    if mod in res:
        for log, c in res[mod]['completeness'].items(): print(mod, log, c)
