#!/usr/bin/env python
"""Token lists for running the unchanged BEVFusion teacher on future frames.

For every navtrain / navtest scene token, its future frames are the next 8 log frames (t+0.5 .. t+4.0 s,
logs are 2 Hz).  Frames already in the existing teacher caches are skipped.  Writes
  future_index_<split>.json   token -> [8 future frame tokens or null]   (all scene tokens)
  future_<split>.yaml         navsim scene_filter with log_names + tokens still to run
  validate_test.yaml          24 navtest tokens that ARE already cached (pipeline reproduction check)
  counts.json
"""
import glob, json, os, pickle
import yaml

C = "/home/external-user/datasets/teacher_cache/bevfusion"
SPLITS = {
    "train": ("/home/external-user/navsim/download/trainval_navsim_logs/trainval", f"{C}/cache_train_50x100"),
    "test": ("/home/external-user/navsim/download/test_navsim_logs/test", f"{C}/cache_val_50x100"),
}
OUT = os.path.dirname(os.path.abspath(__file__))
counts = {}
for split, (logdir, cache) in SPLITS.items():
    cached = set(os.path.basename(p)[:-4] for p in glob.glob(f"{cache}/samples/*/*.npz"))
    index, need, need_logs = {}, set(), set()
    for p in sorted(glob.glob(f"{logdir}/*.pkl")):
        log = os.path.basename(p)[:-4]
        toks = [f["token"] for f in pickle.load(open(p, "rb"))]
        for i, t in enumerate(toks):
            if t not in cached:
                continue
            fut = [toks[i + k] if i + k < len(toks) else None for k in range(1, 9)]
            index[t] = fut
            for f in fut:
                if f is not None and f not in cached:
                    need.add(f); need_logs.add(log)
    json.dump(index, open(f"{OUT}/future_index_{split}.json", "w"))
    yaml.safe_dump({"log_names": sorted(need_logs), "tokens": sorted(need)}, open(f"{OUT}/future_{split}.yaml", "w"))
    counts[split] = {"scene_tokens": len(index), "future_frames_to_run": len(need), "logs": len(need_logs),
                     "cached_existing": len(cached)}
    if split == "test":
        val = sorted(cached)[:24]
        vlogs = set()
        for p in sorted(glob.glob(f"{logdir}/*.pkl")):
            toks = {f["token"] for f in pickle.load(open(p, "rb"))}
            if toks & set(val): vlogs.add(os.path.basename(p)[:-4])
        yaml.safe_dump({"log_names": sorted(vlogs), "tokens": val}, open(f"{OUT}/validate_test.yaml", "w"))
json.dump(counts, open(f"{OUT}/counts.json", "w"), indent=1)
print(json.dumps(counts, indent=1))
