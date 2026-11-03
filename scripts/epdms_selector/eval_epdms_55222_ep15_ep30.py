#!/usr/bin/env python
"""Evaluate official EPDMS (5/5/2/2/2) selector variants on ep30 and ep15 for navtest & navhard EPDMS."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

NAVSIM22 = "/workspace/navsim_workspace/navsim"
if sys.path[0] != NAVSIM22:
    sys.path.insert(0, NAVSIM22)

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "/workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main")
from optimize_epdms_v7 import (
    _init_worker,
    evaluate_config_navhard,
    evaluate_config_navtest,
)

TIE_ORDER = np.concatenate([np.arange(0, 96, 6), np.array([c for c in range(96) if c % 6], np.int64)])


def pick_tie(score: np.ndarray, ok_mask: np.ndarray) -> np.ndarray:
    s = np.where(ok_mask & np.isfinite(score), score, -np.inf)
    j = np.argmax(s[:, TIE_ORDER], axis=1)
    return TIE_ORDER[j].astype(np.int64)


def load_pool_any(ep_tag: str, split: str):
    """ep_tag in ('ep30', 'ep15'), split in ('navtest', 'navhard')."""
    base = Path("/workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main")
    if split == "navtest":
        er = Path(f"/workspace/yongjae/ssd/yongjae_refiner/v7/eval/v2_v15_v7_main_{ep_tag}/eval/root")
        pk = er / "packed/navtest"
        inf = er / f"infer/v2_v15_v7_main_{ep_tag}/navtest"
        tdf = pd.read_parquet(pk / "tokens.parquet")
        toks = [str(t) for t in tdf.token.to_numpy()]
        cand96 = np.load(pk / "cand96.npy").astype(np.float32)
        lat96 = np.load(inf / "lat_traj.npy").astype(np.float32)
        v2_final = np.load(pk / "v2_final.npy").astype(np.float64)
        v2_im = np.load(pk / "v2_im.npy").astype(np.float64)
        v2_sim = np.load(pk / "v2_sim.npy").astype(np.float64)
        logit96 = np.load(inf / "score_logit.npy").astype(np.float64)
        valid96 = np.load(pk / "valid96.npy").astype(bool)
    else:
        pdir_name = "pool96_ep29_navhard" if ep_tag == "ep30" else f"pool96_{ep_tag}_navhard"
        pdir = base / pdir_name
        toks = json.load(open(pdir / "tokens.json"))
        cand96 = np.load(pdir / "cand96.npy").astype(np.float32)
        lat96 = np.load(pdir / "lat_traj.npy").astype(np.float32)
        v2_final = np.load(pdir / "v2_final.npy").astype(np.float64)
        v2_im = np.load(pdir / "v2_im.npy").astype(np.float64)
        v2_sim = np.load(pdir / "v2_sim.npy").astype(np.float64)
        logit96 = np.load(pdir / "score_logit.npy").astype(np.float64)
        valid96 = np.load(pdir / "valid96.npy").astype(bool)
    return toks, cand96, lat96, v2_final, v2_im, v2_sim, logit96, valid96


def build_selector_trajs(ep_tag: str, split: str) -> Dict[str, Dict[str, np.ndarray]]:
    toks, cand96, lat96, v2_final, v2_im, v2_sim, logit96, valid96 = load_pool_any(ep_tag, split)
    N = len(toks)
    ar = np.arange(N)
    vtype = np.arange(96) % 6
    ok_all = valid96 | (vtype[None, :] == 0)

    prob96 = 1.0 / (1.0 + np.exp(-logit96))
    im96 = np.repeat(v2_im, 6, axis=1)
    v2f96 = np.repeat(v2_final, 6, axis=1)
    v2sim96 = np.repeat(v2_sim, 6, axis=1)

    eps = 1e-6
    nc_c, dac_c, ep_c, ttc_c, cmf_c = [prob96[..., i] for i in range(5)]
    nc_v, dac_v, ep_v, ttc_v, cmf_v = [v2sim96[..., i] for i in range(5)]

    # Kinematic / structural LK and EC indicators on the 96 candidates:
    # v=0: id, v=1: a-1.0, v=2: a-0.5, v=3: a+0.5, v=4: l-0.5, v=5: l+0.5
    lk_kin = np.broadcast_to((~np.isin(vtype, [4, 5])).astype(np.float64)[None, :], (N, 96))
    ec_kin_id = np.broadcast_to((vtype == 0).astype(np.float64)[None, :], (N, 96))
    ec_kin_noacc = np.broadcast_to(np.isin(vtype, [0, 1, 2]).astype(np.float64)[None, :], (N, 96))

    # 1. Official v1 PDMS (5*EP + 5*TTC + 2*C) / 12 (P in sel_pdms.py)
    s_pdms_12 = nc_c * dac_c * (5.0 * ep_c + 5.0 * ttc_c + 2.0 * cmf_c) / 12.0

    # 2. Official EPDMS (5/5/2/2/2)/16 with unpredicted heads = 1 (LK=1, EC=1, DDC=1, TLC=1)
    s_epdms_55222_const1 = nc_c * dac_c * (5.0 * ep_c + 5.0 * ttc_c + 2.0 * cmf_c + 2.0 * 1.0 + 2.0 * 1.0) / 16.0

    # 3. Official EPDMS (5/5/2/2/2)/16 with EC=C, LK=1
    s_epdms_55222_ecC_lk1 = nc_c * dac_c * (5.0 * ep_c + 5.0 * ttc_c + 2.0 * cmf_c + 2.0 * 1.0 + 2.0 * cmf_c) / 16.0

    # 4. Official EPDMS (5/5/2/2/2)/16 with LK=DAC, EC=C
    s_epdms_55222_lkDAC_ecC = nc_c * dac_c * (5.0 * ep_c + 5.0 * ttc_c + 2.0 * cmf_c + 2.0 * dac_c + 2.0 * cmf_c) / 16.0

    # 5. Official EPDMS (5/5/2/2/2)/16 with Kinematic LK (non-lat=1) & EC (id=1)
    s_epdms_55222_kin_id = nc_c * dac_c * (5.0 * ep_c + 5.0 * ttc_c + 2.0 * cmf_c + 2.0 * lk_kin + 2.0 * ec_kin_id) / 16.0

    # 6. Official EPDMS (5/5/2/2/2)/16 with Kinematic LK (non-lat=1) & EC (no-accel=1)
    s_epdms_55222_kin_noacc = nc_c * dac_c * (5.0 * ep_c + 5.0 * ttc_c + 2.0 * cmf_c + 2.0 * lk_kin + 2.0 * ec_kin_noacc) / 16.0

    # 7. Two-Head (V2 + CK2) Official EPDMS (5/5/2/2/2)/16
    s_v2_55222_const1 = nc_v * dac_v * (5.0 * ep_v + 5.0 * ttc_v + 2.0 * cmf_v + 4.0) / 16.0
    s_twohead_55222_const1 = 0.5 * np.log(s_v2_55222_const1 + eps) + 0.5 * np.log(s_epdms_55222_const1 + eps)
    s_twohead_55222_kin_noacc = 0.5 * np.log(s_v2_55222_const1 + eps) + 0.5 * np.log(s_epdms_55222_kin_noacc + eps)

    # 8. Blend with v2_final: (1-b)*v2_final + b*log(EPDMS_55222)
    s_blend_b050_55222_const1 = 0.5 * v2f96 + 0.5 * np.log(nc_c + eps) + 0.5 * np.log(dac_c + eps) + 0.5 * np.log(5.0 * ep_c + 5.0 * ttc_c + 2.0 * cmf_c + 4.0 + eps)
    s_blend_b075_55222_const1 = 0.25 * v2f96 + 0.75 * np.log(nc_c + eps) + 0.75 * np.log(dac_c + eps) + 0.75 * np.log(5.0 * ep_c + 5.0 * ttc_c + 2.0 * cmf_c + 4.0 + eps)
    s_blend_b050_55222_kin = 0.5 * v2f96 + 0.5 * np.log(nc_c + eps) + 0.5 * np.log(dac_c + eps) + 0.5 * np.log(5.0 * ep_c + 5.0 * ttc_c + 2.0 * cmf_c + 2.0 * lk_kin + 2.0 * ec_kin_noacc + eps)

    # 9. Stored v7_default (b=0.75 for ep30, b=1.0 for ep15)
    b_def = 0.75 if ep_tag == "ep30" else 1.0
    s_v7_default = (1.0 - b_def) * v2f96 + b_def * (
        0.1 * np.log(im96 + eps) + 0.5 * np.log(nc_c + eps) + 0.5 * np.log(dac_c + eps) + 1.0 * np.log(5.0 * ttc_c + 2.0 * cmf_c + 5.0 * ep_c + eps)
    )

    # 10. Our Best EPDMS Selectors (navtest: tune_31 / tune_41; navhard: tune_37)
    s_our_best_nt = (
        0.3 * v2f96
        + 1.2 * np.log(nc_c + eps) + 1.2 * np.log(dac_c + eps)
        + 0.5 * np.log(5.0 * ttc_c + 2.0 * cmf_c + 5.0 * ep_c + eps)
        + 0.3 * np.log(nc_v + eps) + 0.3 * np.log(dac_v + eps)
        + 2.0 * s_pdms_12
    )
    dv_nt = [0.04, -0.02, 0.01, -0.05, 0.0, 0.0]
    for v in range(6):
        s_our_best_nt[:, vtype == v] += dv_nt[v]

    s_our_best_nh = (
        0.6 * v2f96
        + 0.8 * np.log(nc_c + eps) + 0.8 * np.log(dac_c + eps) + 0.3 * np.log(ttc_c + eps)
        + 0.5 * np.log(5.0 * ttc_c + 2.0 * cmf_c + 5.0 * ep_c + eps)
        + 0.3 * np.log(nc_v + eps) + 0.3 * np.log(dac_v + eps)
    )
    dv_nh = [0.0, 0.01, 0.02, -0.02, 0.0, 0.0]
    for v in range(6):
        s_our_best_nh[:, vtype == v] += dv_nh[v]

    score_map = {
        "v7_default": s_v7_default,
        "P_pdms_552_div12": s_pdms_12,
        "EPDMS_55222_const1": s_epdms_55222_const1,
        "EPDMS_55222_ecC_lk1": s_epdms_55222_ecC_lk1,
        "EPDMS_55222_lkDAC_ecC": s_epdms_55222_lkDAC_ecC,
        "EPDMS_55222_kin_id": s_epdms_55222_kin_id,
        "EPDMS_55222_kin_noacc": s_epdms_55222_kin_noacc,
        "TwoHead_55222_const1": s_twohead_55222_const1,
        "TwoHead_55222_kin_noacc": s_twohead_55222_kin_noacc,
        "Blend_b050_55222_const1": s_blend_b050_55222_const1,
        "Blend_b075_55222_const1": s_blend_b075_55222_const1,
        "Blend_b050_55222_kin": s_blend_b050_55222_kin,
        "Our_Best_NT_tune31": s_our_best_nt,
        "Our_Best_NH_tune37": s_our_best_nh,
    }

    out_trajs = {}
    for name, sc in score_map.items():
        idx = pick_tie(sc, ok_all)
        chosen = lat96[ar, idx]
        out_trajs[name] = {toks[i]: chosen[i] for i in range(N)}
    return out_trajs


def run_eval(ep_tag: str, split: str, workers: int = 36):
    mc_dir = (
        "/workspace/yongjae/perception_eval/epdms/metric_cache_v22_navtest"
        if split == "navtest"
        else "/workspace/navsim_workspace/exp/metric_cache_navhard_two_stage_v22"
    )
    out_json = Path(f"/workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/epdms_55222_{split}_{ep_tag}.json")
    if out_json.is_file():
        saved = json.load(open(out_json))
    else:
        saved = {}

    trajs_by_name = build_selector_trajs(ep_tag, split)
    Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    if split == "navhard":
        sf_raw = yaml.load(
            open(f"{NAVSIM22}/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"),
            Loader=Loader,
        )
        split_raw = yaml.load(
            open(f"{NAVSIM22}/navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml"),
            Loader=Loader,
        )
        all_tokens = [str(t) for t in sf_raw["tokens"]] + [str(t) for t in sf_raw["reactive_synthetic_initial_tokens"]]
        all_mappings = {}
        token_set = set(all_tokens)
        for orig_token, prev_token, two_stage_pairs in split_raw["reactive_all_mapping"]:
            if prev_token in token_set or orig_token in token_set:
                all_mappings[(str(orig_token), str(prev_token))] = [(str(p[0]), str(p[1])) for p in two_stage_pairs]
    else:
        sf_raw = yaml.load(
            open(f"{NAVSIM22}/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"),
            Loader=Loader,
        )
        all_tokens = [str(t) for t in sf_raw["tokens"]]
        all_mappings = {}

    row_cache: Dict[Tuple[str, bytes], Dict[str, Any]] = {}
    ec_cache: Dict[Tuple[str, bytes, str, bytes], float] = {}
    start_adj_cache: Dict[str, str] = {}

    print(f"=== Evaluating {ep_tag} on {split} (workers={workers}) ===", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(mc_dir, split),
    ) as ex:
        for name, traj_dict in trajs_by_name.items():
            if name in saved:
                continue
            if split == "navhard":
                res = evaluate_config_navhard(name, traj_dict, all_tokens, all_mappings, row_cache, ec_cache, ex, workers)
            else:
                res = evaluate_config_navtest(name, traj_dict, all_tokens, start_adj_cache, row_cache, ec_cache, ex, workers)
            saved[name] = res
            out_json.parent.mkdir(parents=True, exist_ok=True)
            json.dump(saved, open(out_json, "w"), indent=2)
    print(f"=== Saved {out_json} ===", flush=True)


if __name__ == "__main__":
    ep = sys.argv[1]
    sp = sys.argv[2]
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 36
    run_eval(ep, sp, w)
