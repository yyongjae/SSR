#!/usr/bin/env python
"""Compare two off_equiv_probe.py outputs bitwise (ref = CK2 8ae8b1d tree, new = refiner-v3 worktree).

  python -B tools/ck/v3/off_equiv_compare.py <ref.npz> <new.npz>  -> JSON report; exit 1 on any difference.
Arrays: identical key sets, dtypes, shapes and bytes (NaN == NaN at the same place).  Facts (the .json sidecar):
param counts, state_dict keys, data hashes / item keys, step-stat keys, optimiser group count must be equal;
config.json keys may only GAIN the documented temporal keys (SPEC 4.7: 'temporal' dict + the new CLI flags).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

NEW_CONFIG_KEYS = {"temporal", "privileged", "param_count_temporal_shared", "param_count_forecaster", "temporal_on",
                   "t_source", "t_forecaster", "t_modes", "t_lambda_fc", "t_fc_backprop", "t_frame_drop",
                   "t_obj_drop", "t_vnoise", "t_pack_root", "t_ca_scales",
                   # review fixes D3 / D5: pre-trained forecaster flags and the run provenance record
                   "t_fc_init", "t_fc_train", "provenance"}
EQUAL_FACTS = ("param_counts", "sd_keys", "ck2T10_param_count", "ck2T10_config_param_count", "ck2T10_trainable",
               "data_hashes", "data_keys", "ck2_describe", "train1_step_keys", "train3_step_keys",
               "train1_optim_groups", "train3_optim_groups", "train1_val_keys", "train3_val_keys",
               "train1_param_count", "train3_param_count")


def compare(ref_path: str, new_path: str) -> dict:
    A, B = np.load(ref_path), np.load(new_path)
    ka, kb = set(A.files), set(B.files)
    rep = {"n_arrays": len(ka), "only_ref": sorted(ka - kb), "only_new": sorted(kb - ka), "diff": [],
           "max_abs": 0.0}
    for k in sorted(ka & kb):
        a, b = A[k], B[k]
        if a.dtype != b.dtype or a.shape != b.shape:
            rep["diff"].append(f"{k}: {a.dtype}{a.shape} vs {b.dtype}{b.shape}")
            continue
        same = a.tobytes() == b.tobytes()
        if not same and a.dtype.kind == "f":
            same = bool(np.array_equal(a, b, equal_nan=True))
        if not same:
            d = float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.dtype.kind in "fiub" \
                else float("nan")
            rep["diff"].append(f"{k}: max |d| {d:g}")
            rep["max_abs"] = max(rep["max_abs"], d)
    ma = json.loads(Path(ref_path).with_suffix(".json").read_text())
    mb = json.loads(Path(new_path).with_suffix(".json").read_text())
    rep["facts_diff"] = [k for k in EQUAL_FACTS if ma.get(k) != mb.get(k)]
    for s in ("train1", "train3"):
        ca, cb = set(ma.get(f"{s}_config_keys", [])), set(mb.get(f"{s}_config_keys", []))
        rep[f"{s}_config_only_ref"] = sorted(ca - cb)
        rep[f"{s}_config_new_keys"] = sorted(cb - ca)
        if (ca - cb) or not (cb - ca) <= NEW_CONFIG_KEYS:
            rep["facts_diff"].append(f"{s}_config_keys")
    rep["n_data_hashes"] = len(ma.get("data_hashes", {}))
    rep["param_counts"] = mb.get("param_counts")
    rep["ck2T10"] = [mb.get("ck2T10_param_count"), mb.get("ck2T10_trainable")]
    rep["identical"] = not (rep["only_ref"] or rep["only_new"] or rep["diff"] or rep["facts_diff"])
    return rep


if __name__ == "__main__":
    r = compare(sys.argv[1], sys.argv[2])
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["identical"] else 1)
