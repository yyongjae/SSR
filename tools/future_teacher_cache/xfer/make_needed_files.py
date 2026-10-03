#!/usr/bin/env python
"""Per-archive lists of the sensor files to extract while streaming OpenScene v1.1 trainval (report 40 §2, §9).

Definitions (report 40 §2-1, same as scripts/needed.py and scripts/taskC/budget.py):
  future frames of a token t at position i of its log pkl = log frames i+1 .. i+2*H (2 Hz, same log, clipped at log end)
  run frames  = union of future frames over all input tokens, minus frames whose current-frame npz exists in --cache-dir
  on disk     = all --cams .jpg AND the lidar .pcd of the frame exist under --sensor-root
  DL frames   = run frames that are not on disk
  sweeps      = lidar of frames f-1, f-2 of the same log (navsim_converter.build_sweeps; stops at the first frame
                without lidar_path or at the log start)
  needed files = (cams + lidar of every run frame) + (lidar of sweeps of every run frame), minus files already
                 present under --sensor-root, grouped by archive index through the official archive->log map.

ReSMap (camera CAM_L0/F0/R0 + satellite, no lidar, temporal memory run in scene order; report 40 §10-7):
  mode A (default) = ReSMap re-runs the sensor-covered frames of each run log in scene order, as its existing caches
                     were made. The --cams files above already cover it; nothing extra is extracted. summary.json
                     'resmap' reports the frame counts.
  mode B (--resmap-full-logs) = gap-free 2 Hz sequences: also extract --cams for EVERY frame of every run log that is
                     not on disk (cameras only). Decide before streaming; widening later needs the camera archives again.

Outputs in --out:
  needed_camera_<i>.txt / needed_lidar_<i>.txt  full tar member names (openscene-v1.1/sensor_blobs/trainval/<log>/...)
                                                 for every i in 0..199 (empty file when nothing is needed)
  run_frames.txt                                 token<TAB>log<TAB>frame_idx<TAB>dl(0/1)  (frames to infer)
  run_frames.yaml                                navsim scene_filter style {log_names, tokens}
  summary.json                                   counts per archive and totals, estimated extracted bytes
Nothing outside --out is written (except --lf-cache when given).
"""
import argparse, glob, json, os, pickle, re, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "openscene-v1.1/sensor_blobs/trainval/"
# archive-0 listing means (report 40 §9-1), used when no exact listing is given
MEAN_SIZE = {"CAM_F0": 220578, "CAM_L0": 201296, "CAM_R0": 203852, "lidar": 1403088}
MEAN_OTHER_CAM = (3091000 - 1403088 - 220578 - 201296 - 203852) // 5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", nargs="+", required=True,
                   help="split parquet(s) (column 'token'), yaml ({tokens: [...]} or list), json (list or dict keys) or txt (one token per line); union of all")
    p.add_argument("--horizon-s", type=float, default=4.0, help="future horizon in seconds; frames k=1..round(2*h)")
    p.add_argument("--logs-dir", default="/home/external-user/navsim/download/trainval_navsim_logs/trainval")
    p.add_argument("--sensor-root", default="/home/external-user/navsim/download/trainval_sensor_blobs",
                   help="root holding trainval/<log>/<CAM|MergedPointCloud>/ (checked for already-present files; read only)")
    p.add_argument("--cache-dir", nargs="+", default=["/home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100"],
                   help="existing BEVFusion cache dir(s) with samples/<tok[:2]>/<tok>.npz; those frames are not re-run")
    p.add_argument("--map", default=f"{HERE}/hf/map_trainval.json")
    p.add_argument("--tree-dir", default=f"{HERE}/hf", help="dir with tree_openscene_sensor_trainval_{camera,lidar}.json (tgz sizes)")
    p.add_argument("--cams", nargs="+", default=["CAM_F0", "CAM_L0", "CAM_R0"])
    p.add_argument("--max-sweeps", type=int, default=2)
    p.add_argument("--no-sweeps", action="store_true", help="do not add sweep lidar (debug)")
    p.add_argument("--resmap-full-logs", action="store_true",
                   help="ReSMap mode B: also extract --cams for every frame of the run logs that is not on disk (gap-free sequences)")
    p.add_argument("--exact-listings", nargs="*", default=[],
                   help="tar -tv / -xvv listings (e.g. results/full/list_*_0.txt or state/members/*.txt) giving exact member sizes")
    p.add_argument("--lf-cache", default=None, help="optional pickle caching the parsed log pkls (speeds up reruns)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    return p.parse_args()


def read_tokens(paths):
    toks = []
    for path in paths:
        if path.endswith(".parquet"):
            import pandas as pd
            toks += pd.read_parquet(path, columns=["token"]).token.astype(str).tolist()
        elif path.endswith((".yaml", ".yml")):
            import yaml
            d = yaml.safe_load(open(path))
            toks += list(d["tokens"] if isinstance(d, dict) else d)
        elif path.endswith(".json"):
            d = json.load(open(path))
            toks += list(d.keys() if isinstance(d, dict) else d)
        else:
            toks += [l.split()[0] for l in open(path) if l.strip()]
    return toks


def rel_of(path):
    """navsim data_path / lidar_path -> '<log>/<SENSOR>/<file>' (same normalisation as navsim_converter)."""
    rel = str(path).replace("\\", "/").lstrip("/")
    if "sensor_blobs/" in rel:
        rel = rel.split("sensor_blobs/")[-1]
    if rel.startswith("trainval/"):
        rel = rel[len("trainval/"):]
    return rel


def load_log(args):
    p, cams = args
    fr = pickle.load(open(p, "rb"))
    out = []
    for f in fr:
        lid = f.get("lidar_path")
        lid = os.path.splitext(rel_of(lid))[0] + ".pcd" if lid else None
        out.append((f["token"], lid, {c: rel_of(f["cams"][c]["data_path"]) if c in f.get("cams", {}) and f["cams"][c].get("data_path") else None for c in cams}))
    return os.path.basename(p)[:-4], out


def load_logs(logs_dir, cams, workers, cache):
    if cache and os.path.exists(cache):
        LF = pickle.load(open(cache, "rb"))
        if LF.get("__cams__") == list(cams):
            del LF["__cams__"]
            return LF
    paths = sorted(glob.glob(f"{logs_dir}/*.pkl"))
    with Pool(workers) as pool:
        LF = dict(pool.imap_unordered(load_log, [(p, cams) for p in paths], chunksize=4))
    if cache:
        pickle.dump(dict(LF, __cams__=list(cams)), open(cache + ".tmp", "wb"))
        os.replace(cache + ".tmp", cache)
    return LF


def cached_tokens(cache_dirs):
    s = set()
    for c in cache_dirs:
        for sub in os.scandir(f"{c}/samples"):
            if sub.is_dir():
                s.update(e.name[:-4] for e in os.scandir(sub.path) if e.name.endswith(".npz") and not e.name.endswith(".tmp.npz"))
    return s


def read_listing(path, sizes):
    rx = re.compile(r"^\S+\s+\S+\s+(\d+)\s+\S+\s+\S+\s+(\S+)$")
    for line in open(path):
        m = rx.match(line.rstrip("\n"))
        if m and not m.group(2).endswith("/"):  # tar -tv / -xvv print the original (unstripped) member name
            sizes[m.group(2)] = int(m.group(1))


def main():
    a = parse_args()
    t0 = time.time()
    H = int(round(2 * a.horizon_s))
    os.makedirs(a.out, exist_ok=True)
    amap = json.load(open(a.map))
    log2arc = {l: int(k.rsplit("_", 1)[1]) for k, v in amap.items() for l in v}
    n_arc = len(amap)
    tgz = {}
    for mod in ("camera", "lidar"):
        tgz[mod] = {int(x["path"].rsplit("_", 1)[1][:-4]): int(x["size"])
                    for x in json.load(open(f"{a.tree_dir}/tree_openscene_sensor_trainval_{mod}.json")) if x["path"].endswith(".tgz")}

    LF = load_logs(a.logs_dir, a.cams, a.workers, a.lf_cache)
    tok2 = {f[0]: (log, i) for log, fr in LF.items() for i, f in enumerate(fr)}
    missing_logs_in_map = sorted(set(LF) - set(log2arc))
    print(f"[needed] {len(LF)} logs, {len(tok2)} frames loaded ({time.time()-t0:.0f}s)", flush=True)

    toks = set(read_tokens(a.tokens))
    tok_unknown = sorted(t for t in toks if t not in tok2)
    cached = cached_tokens(a.cache_dir)
    tok_not_cached = sum(1 for t in toks if t not in cached)
    print(f"[needed] {len(toks)} input tokens ({len(tok_unknown)} not in any log, {tok_not_cached} without current npz), {len(cached)} cached npz", flush=True)

    # run frames
    fut = set()
    for t in toks:
        if t not in tok2:
            continue
        log, i = tok2[t]
        n = len(LF[log])
        for k in range(1, H + 1):
            if i + k < n:
                fut.add((log, i + k))
    run = {x for x in fut if LF[x[0]][x[1]][0] not in cached}

    # disk state per log (listdir once per needed sensor dir)
    disk = {}
    def present(rel):
        d, fn = rel.rsplit("/", 1)
        if d not in disk:
            try:
                disk[d] = set(os.listdir(f"{a.sensor_root}/trainval/{d}"))
            except FileNotFoundError:
                disk[d] = set()
        return fn in disk[d]

    def frame_files(log, i):
        _, lid, cams = LF[log][i]
        cam_files = [cams[c] for c in a.cams]
        return cam_files, lid

    need = {"camera": defaultdict(set), "lidar": defaultdict(set)}
    dl, ondisk_missing_sweep, bad = set(), 0, Counter()
    sweep_only_lidar = set()
    for (log, i) in run:
        arc = log2arc[log]
        cam_files, lid = frame_files(log, i)
        frame_missing = False
        for rel in cam_files:
            if rel is None:
                bad["run_frame_without_cam_path"] += 1; continue
            if not present(rel):
                need["camera"][arc].add(rel); frame_missing = True
        if lid is None:
            bad["run_frame_without_lidar_path"] += 1
        elif not present(lid):
            need["lidar"][arc].add(lid); frame_missing = True
        if frame_missing:
            dl.add((log, i))
        if a.no_sweeps:
            continue
        sweep_missing = False
        for k in range(1, a.max_sweeps + 1):
            j = i - k
            if j < 0:
                break
            slid = LF[log][j][1]
            if not slid:
                break
            if not present(slid):
                need["lidar"][arc].add(slid); sweep_missing = True
                if (log, j) not in run:
                    sweep_only_lidar.add((log, j))
        if sweep_missing and not frame_missing:
            ondisk_missing_sweep += 1

    # ReSMap: frames per run log that have all --cams after extraction (mode A), and mode B additions
    run_logs = sorted({l for l, _ in run})
    rs = Counter()
    for log in run_logs:
        arc = log2arc[log]
        for i in range(len(LF[log])):
            rs["frames_in_run_logs"] += 1
            cam_files = frame_files(log, i)[0]
            if any(c is None for c in cam_files):
                rs["frames_without_cam_path"] += 1; continue
            missing = [c for c in cam_files if not present(c) and c not in need["camera"][arc]]
            if not missing:
                rs["mode_a_sensor_covered_frames"] += 1
                continue
            rs["mode_b_extra_frames"] += 1
            rs["mode_b_extra_cam_files"] += len(missing)
            rs["mode_b_extra_est_bytes"] += sum(MEAN_SIZE.get(c.split("/")[1], MEAN_OTHER_CAM) for c in missing)
            if a.resmap_full_logs:
                need["camera"][arc].update(missing)
    rs["mode_b_frames"] = rs["mode_a_sensor_covered_frames"] + rs["mode_b_extra_frames"]

    # sizes
    exact = {}
    for pth in a.exact_listings:
        read_listing(pth, exact)
    fr_per_arc = Counter(log2arc[l] for l in LF for _ in LF[l])
    def scale(mod, arc):  # per-frame compressed size of archive relative to archive 0 (report 40 §9-2 'arcscaled')
        return (tgz[mod][arc] / fr_per_arc[arc]) / (tgz[mod][0] / fr_per_arc[0])
    def est(rel):
        sensor = rel.split("/")[1]
        return MEAN_SIZE.get("lidar" if sensor == "MergedPointCloud" else sensor, MEAN_OTHER_CAM)

    per_arc, tot = {}, Counter()
    run_per_arc, dl_per_arc = Counter(log2arc[l] for l, _ in run), Counter(log2arc[l] for l, _ in dl)
    for arc in range(n_arc):
        row = {"run_frames": run_per_arc[arc], "dl_frames": dl_per_arc[arc]}
        for mod in ("camera", "lidar"):
            rels = sorted(need[mod][arc])
            with open(f"{a.out}/needed_{mod}_{arc}.txt", "w") as fh:
                fh.writelines(PREFIX + r + "\n" for r in rels)
            b_mean = sum(est(r) for r in rels)
            b_exact = [exact.get(PREFIX + r) for r in rels]
            n_exact = sum(x is not None for x in b_exact)
            row[mod] = {"files": len(rels), "est_bytes": b_mean, "est_bytes_arcscaled": int(b_mean * scale(mod, arc)),
                        "exact_bytes_known_files": n_exact, "exact_bytes": sum(x for x in b_exact if x is not None),
                        "tgz_bytes": tgz[mod][arc]}
            tot[f"{mod}_files"] += len(rels); tot[f"{mod}_est_bytes"] += b_mean
            tot[f"{mod}_est_bytes_arcscaled"] += row[mod]["est_bytes_arcscaled"]; tot[f"{mod}_tgz_bytes"] += tgz[mod][arc]
        per_arc[arc] = row

    with open(f"{a.out}/run_frames.txt", "w") as fh:
        for log, i in sorted(run):
            fh.write(f"{LF[log][i][0]}\t{log}\t{i}\t{int((log, i) in dl)}\n")
    import yaml
    yaml.safe_dump({"log_names": sorted({l for l, _ in run}), "tokens": sorted(LF[l][i][0] for l, i in run)},
                   open(f"{a.out}/run_frames.yaml", "w"))

    est_total = tot["camera_est_bytes"] + tot["lidar_est_bytes"]
    est_total_s = tot["camera_est_bytes_arcscaled"] + tot["lidar_est_bytes_arcscaled"]
    tgz_total = tot["camera_tgz_bytes"] + tot["lidar_tgz_bytes"]
    nonempty = {m: sum(1 for r in per_arc.values() if r[m]["files"]) for m in ("camera", "lidar")}
    summary = {
        "args": {k: v for k, v in vars(a).items()},
        "frames_per_token": H,
        "input_tokens": len(toks), "input_tokens_not_in_logs": len(tok_unknown), "input_tokens_without_current_npz": tok_not_cached,
        "future_frames_union": len(fut), "already_cached": len(fut) - len(run),
        "run_frames": len(run), "run_frames_on_disk": len(run) - len(dl), "dl_frames": len(dl),
        "run_frames_on_disk_missing_some_sweep": ondisk_missing_sweep,
        "sweep_only_lidar_frames_outside_run": len(sweep_only_lidar),
        "run_logs": len({l for l, _ in run}), "anomalies": dict(bad),
        "resmap": dict(rs, mode_b_enabled=a.resmap_full_logs, mode_b_extra_est_GB=rs["mode_b_extra_est_bytes"] / 1e9), "logs_not_in_map": missing_logs_in_map,
        "totals": dict(tot), "est_extracted_GB": est_total / 1e9, "est_extracted_GB_arcscaled": est_total_s / 1e9,
        "tgz_total_GB": tgz_total / 1e9,
        # compressed-equivalent of what is kept (archive-0 expansion ratios camera 1.003, lidar 1.134; report 40 §9-1)
        "discard_fraction_compressed_est": 1 - (tot["camera_est_bytes_arcscaled"] / 1.003 + tot["lidar_est_bytes_arcscaled"] / 1.134) / tgz_total,
        "archives_with_nonempty_list": nonempty, "archives": n_arc,
        "per_archive": per_arc, "seconds": time.time() - t0,
    }
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
    short = {k: v for k, v in summary.items() if k not in ("per_archive", "args", "logs_not_in_map")}
    print(json.dumps(short, indent=1))


if __name__ == "__main__":
    main()
