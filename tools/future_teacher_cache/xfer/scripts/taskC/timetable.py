"""Task C time table from budget.json + bench_tar.json (arithmetic only)."""
import json, os
S = os.path.dirname(os.path.abspath(__file__))
b = json.load(open(f"{S}/budget.json")); bt = json.load(open(f"{S}/bench_tar.json"))
TGZ = b["meta"]["tgz_total_GB"]; GZ = bt["gzip_dc"]["MBps_compressed"]
out = {"transfer_GB": TGZ, "gzip_dc_MBps_per_core": GZ, "extract_core_h": TGZ * 1e3 / GZ / 3600, "download_h": {}, "infer_h": {}, "wall_h": {}, "bw_where_gpu_becomes_bottleneck_MBps": {}}
for r in (10, 50, 100, 300): out["download_h"][r] = TGZ * 1e3 / r / 3600
sets = ["stageT_4s", "e2e_4s", "navtrain_4s", "navtrain_5s", "navtrain_logs_all_frames", "trainval_all_frames"]
for s in sets:
    n = b[s]["run_frames"]; d = {}
    for fps, tag in ((11.6, "ded"), (5.8, "shared")):
        for g in (1, 4): d[f"{g}gpu_{tag}"] = n / (fps * g) / 3600
    out["infer_h"][s] = d
    out["bw_where_gpu_becomes_bottleneck_MBps"][s] = {k: TGZ * 1e3 / (v * 3600) for k, v in d.items()}
    w = {}
    for r in (10, 50, 100, 300):
        dl = out["download_h"][r]
        for k in ("1gpu_ded", "4gpu_ded", "1gpu_shared"):
            inf = d[k]
            # overlapped (chunked g~10: tail = one group's inference ~ inf/20) vs serial (stream-all then infer)
            w[f"{r}MBps_{k}"] = dict(overlap=max(dl, inf) + inf / 20, serial=dl + inf)
    out["wall_h"][s] = w
print(json.dumps({k: v for k, v in out.items() if k != "wall_h"}, indent=1))
for s in ("e2e_4s", "trainval_all_frames"):
    for k, v in out["wall_h"][s].items(): print(s, k, {a: round(x, 1) for a, x in v.items()})
json.dump(out, open(f"{S}/timetable.json", "w"), indent=1)
