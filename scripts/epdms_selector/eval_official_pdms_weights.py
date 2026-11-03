#!/usr/bin/env python3
"""
Evaluate strictly using Official PDMS Formula / Weights (as in iPAD, SparseDriveV2, Hydra-MDP):
  PDMS(p) = NC * DAC * (5 * TTC + 5 * EP + 2 * C) / 12
  log PDMS(p) = 1.0 * log(NC) + 1.0 * log(DAC) + 1.0 * log((5 * TTC + 5 * EP + 2 * C) / 12)
ONLY on v2_v15_v7_main (64m BEV): 23ep (ep22) & 30ep (ep29/ep30)
across:
  1) navtest v1 PDMS (N=12,146)
  2) navtest v2.2 One-Stage EPDMS (After & Before, N=12,146)
  3) navhard_two_stage v2.2 Two-Stage EPDMS (After & Before, Stage1=450, Stage2=5462)
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

V7_DIR = Path("/workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main")
sys.path.insert(0, str(V7_DIR))

import optimize_v1_pdms_v7 as v1_v7
import optimize_epdms_v7 as ep_v7

EPS = 1e-6


def official_pdms_val(prob: np.ndarray) -> np.ndarray:
    """prob [..., 5] in order (nc, dac, ep, ttc, comfort) -> Official PDMS in [0, 1]"""
    nc, dac, ep, ttc, cmf = [prob[..., i] for i in range(5)]
    return nc * dac * (5.0 * ttc + 5.0 * ep + 2.0 * cmf) / 12.0


def official_pdms_log(prob: np.ndarray, im: np.ndarray = None, w_im: float = 0.0) -> np.ndarray:
    """Log-space official PDMS: w_im*log(im) + 1*log(nc) + 1*log(dac) + 1*log((5*ttc + 5*ep + 2*cmf)/12)"""
    nc, dac, ep, ttc, cmf = [prob[..., i] for i in range(5)]
    s = (
        np.log(nc + EPS)
        + np.log(dac + EPS)
        + np.log((5.0 * ttc + 5.0 * ep + 2.0 * cmf) / 12.0 + EPS)
    )
    if im is not None and w_im != 0.0:
        s = s + w_im * np.log(im + EPS)
    return s


def compute_official_indices(
    v2_final: np.ndarray,
    v2_im: np.ndarray,
    v2_sim: np.ndarray,
    logit96: np.ndarray,
    valid96: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    96 candidates are ordered [16 parents, 6 variants]:
      prank = idx // 6 in {0..15}
      vtype = idx % 6  in {0..5} (0 is identity v=0)
    """
    N = len(v2_final)
    vtype = np.arange(96) % 6
    ok_all = valid96 | (vtype[None, :] == 0)
    ok_v0 = (vtype[None, :] == 0)  # Only the 16 original anchor modes (v=0)

    prob96 = 1.0 / (1.0 + np.exp(-np.clip(logit96, -30.0, 30.0)))
    im96 = np.repeat(v2_im, 6, axis=1)
    v2f96 = np.repeat(v2_final, 6, axis=1)
    v2sim96 = np.repeat(v2_sim, 6, axis=1)

    # Official PDMS on CK2 [N, 96] and V2 [N, 96]
    pdms_ck = official_pdms_val(prob96)
    pdms_v2 = official_pdms_val(v2sim96)
    log_ck_noim = official_pdms_log(prob96)
    log_ck_im01 = official_pdms_log(prob96, im96, 0.1)
    log_v2_noim = official_pdms_log(v2sim96)
    log_v2_im01 = official_pdms_log(v2sim96, im96, 0.1)

    # WoTE / SSR-v7 default score on CK2 (w = 0.1, 0.5, 0.5, 1.0)
    wote_ck = (
        0.1 * np.log(im96 + EPS)
        + 0.5 * np.log(prob96[..., 0] + EPS)
        + 0.5 * np.log(prob96[..., 1] + EPS)
        + 1.0 * np.log((5.0 * prob96[..., 3] + 2.0 * prob96[..., 4] + 5.0 * prob96[..., 2]) / 12.0 + EPS)
    )

    # Probability-averaged (0.5 * p_v2 + 0.5 * p_ck) plugged into Official PDMS formula
    p_arith50 = 0.5 * v2sim96 + 0.5 * prob96
    p_arith60 = 0.4 * v2sim96 + 0.6 * prob96

    def pick(score: np.ndarray, mask: np.ndarray = ok_all) -> np.ndarray:
        return np.argmax(np.where(mask, score, -1e9), axis=1).astype(np.int64)

    sels = {}
    # 0. Existing WoTE-weighted baselines (w_nc=0.5, w_dac=0.5, w_rest=1.0)
    sels["0_v2_wote_16mode (w=0.1,0.5,0.5,1)"] = np.zeros(N, dtype=np.int64)
    sels["0_v7_default_b0.75 (w=0.1,0.5,0.5,1)"] = pick(0.25 * v2f96 + 0.75 * wote_ck)
    sels["0_v7_default_b0.50 (w=0.1,0.5,0.5,1)"] = pick(0.50 * v2f96 + 0.50 * wote_ck)

    # 1. 16-Mode Only (v=0) with Official PDMS Formula (iPAD / SparseDriveV2 on 16 proposals)
    sels["1_v2_16mode_official_pdms (no_im)"] = pick(pdms_v2, ok_v0)
    sels["1_v2_16mode_official_pdms (w_im=0.1)"] = pick(log_v2_im01, ok_v0)
    sels["1_ck_16mode_official_pdms (no_im)"] = pick(pdms_ck, ok_v0)
    sels["1_ck_16mode_official_pdms (w_im=0.1)"] = pick(log_ck_im01, ok_v0)
    sels["1_twohead_16mode_official_pdms (no_im)"] = pick(0.5 * log_v2_noim + 0.5 * log_ck_noim, ok_v0)
    sels["1_twohead_16mode_official_pdms (w_im=0.1)"] = pick(0.5 * log_v2_im01 + 0.5 * log_ck_im01, ok_v0)

    # 2. 96-Pool CK2-Only with Official PDMS Formula (beta=1.0)
    sels["2_ck96_official_pdms_pure (plugin_b1.0)"] = pick(pdms_ck)
    sels["2_ck96_official_pdms_im0.1 (w=0.1,1,1,1, b=1.0)"] = pick(log_ck_im01)

    # 3. 96-Pool Standard Blend: (1-b)*v2_final + b*CK_official_pdms
    sels["3_blend_v2f_ck_official_b0.50 (plugin_b0.5)"] = pick(0.50 * v2f96 + 0.50 * log_ck_noim)
    sels["3_blend_v2f_ck_official_b0.75 (plugin_b0.75)"] = pick(0.25 * v2f96 + 0.75 * log_ck_noim)
    sels["3_blend_v2f_ck_official_im0.1_b0.50 (f15_def_b0.5)"] = pick(0.50 * v2f96 + 0.50 * log_ck_im01)
    sels["3_blend_v2f_ck_official_im0.1_b0.75 (f15_def_b0.75)"] = pick(0.25 * v2f96 + 0.75 * log_ck_im01)

    # 4. 96-Pool Two-Head Official PDMS (BOTH V2 and CK2 use exact Official PDMS Formula: NC * DAC * (5*TTC+5*EP+2*C)/12)
    sels["4_twohead_official_log_b0.50 (no_im)"] = pick(0.50 * log_v2_noim + 0.50 * log_ck_noim)
    sels["4_twohead_official_log_b0.50 (w_im=0.1)"] = pick(0.50 * log_v2_im01 + 0.50 * log_ck_im01)
    sels["4_twohead_official_log_b0.60 (no_im)"] = pick(0.40 * log_v2_noim + 0.60 * log_ck_noim)
    sels["4_twohead_official_log_b0.60 (w_im=0.1)"] = pick(0.40 * log_v2_im01 + 0.60 * log_ck_im01)
    sels["4_twohead_official_lin_b0.50 (no_im)"] = pick(0.50 * pdms_v2 + 0.50 * pdms_ck)
    sels["4_twohead_official_lin_b0.60 (no_im)"] = pick(0.40 * pdms_v2 + 0.60 * pdms_ck)

    # 5. Probability-Averaged Official PDMS: PDMS(0.5 * p_v2 + 0.5 * p_ck)
    sels["5_prob_avg50_official_pdms (no_im)"] = pick(official_pdms_val(p_arith50))
    sels["5_prob_avg50_official_pdms (w_im=0.1)"] = pick(official_pdms_log(p_arith50, im96, 0.1))
    sels["5_prob_avg60_official_pdms (no_im)"] = pick(official_pdms_val(p_arith60))
    sels["5_prob_avg60_official_pdms (w_im=0.1)"] = pick(official_pdms_log(p_arith60, im96, 0.1))

    return sels


def eval_v1_labels(lab: np.ndarray, idx: np.ndarray) -> Dict[str, float]:
    ar = np.arange(len(idx))
    c = lab[ar, idx]
    return {
        "nc": round(float(c[:, 0].mean() * 100), 2),
        "dac": round(float(c[:, 1].mean() * 100), 2),
        "ep": round(float(c[:, 2].mean() * 100), 2),
        "ttc": round(float(c[:, 3].mean() * 100), 2),
        "comfort": round(float(c[:, 4].mean() * 100), 2),
        "ddc": round(float(c[:, 5].mean() * 100), 2),
        "pdms": round(float(c[:, 6].mean() * 100), 2),
    }


def main():
    summary_out = {}

    # 1. Evaluate v1 PDMS on v2_v15_v7_main (64m) ONLY
    for tag in ["ep22", "ep30", "ep20"]:
        v2_final, v2_im, v2_sim, valid96, logit96, cand96, lat96, lab_pool, lab_lat = v1_v7.load_split_data(tag, "navtest")
        sels = compute_official_indices(v2_final, v2_im, v2_sim, logit96, valid96)
        res = {}
        print(f"\n==================== v2_v15_v7_main (64m) {tag} : navtest v1 PDMS ====================")
        for k, idx in sels.items():
            m = eval_v1_labels(lab_lat, idx)
            res[k] = m
            print(f"{k:50s} | PDMS={m['pdms']:6.2f}% | NC={m['nc']:5.2f} DAC={m['dac']:5.2f} EP={m['ep']:5.2f} TTC={m['ttc']:5.2f} C={m['comfort']:5.2f}")
        summary_out[f"v7_64m_{tag}_v1_pdms"] = res

    # 2. Evaluate EPDMS (navtest & navhard, After & Before) on v2_v15_v7_main (64m) for key official PDMS selectors
    # Map already-computed keys in epdms_{split}_{mode}.json to save time
    precomp_map = {
        "2_ck96_official_pdms_pure (plugin_b1.0)": "plugin_b100",
        "3_blend_v2f_ck_official_b0.50 (plugin_b0.5)": "plugin_b050",
        "3_blend_v2f_ck_official_im0.1_b0.50 (f15_def_b0.5)": "f15_default_b050",
    }

    epdms_keys_to_eval = [
        "1_v2_16mode_official_pdms (no_im)",
        "1_v2_16mode_official_pdms (w_im=0.1)",
        "1_ck_16mode_official_pdms (no_im)",
        "1_twohead_16mode_official_pdms (no_im)",
        "2_ck96_official_pdms_pure (plugin_b1.0)",
        "2_ck96_official_pdms_im0.1 (w=0.1,1,1,1, b=1.0)",
        "3_blend_v2f_ck_official_b0.50 (plugin_b0.5)",
        "3_blend_v2f_ck_official_b0.75 (plugin_b0.75)",
        "3_blend_v2f_ck_official_im0.1_b0.50 (f15_def_b0.5)",
        "4_twohead_official_log_b0.50 (no_im)",
        "4_twohead_official_log_b0.50 (w_im=0.1)",
        "4_twohead_official_log_b0.60 (no_im)",
        "4_twohead_official_lin_b0.50 (no_im)",
        "4_twohead_official_lin_b0.60 (no_im)",
        "5_prob_avg50_official_pdms (no_im)",
        "5_prob_avg60_official_pdms (no_im)",
    ]

    Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    NAVSIM22 = "/workspace/navsim_workspace/navsim"

    for split, mc_dir in [
        ("navhard", "/workspace/navsim_workspace/exp/metric_cache_navhard_two_stage_v22"),
        ("navtest", "/home/external-user/yongjae/SSR/data/exp/metric_cache"),
    ]:
        if split == "navhard":
            sf_raw = yaml.load(open(f"{NAVSIM22}/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"), Loader=Loader)
            split_raw = yaml.load(open(f"{NAVSIM22}/navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml"), Loader=Loader)
            all_tokens = [str(t) for t in sf_raw["tokens"]] + [str(t) for t in sf_raw["reactive_synthetic_initial_tokens"]]
            all_mappings = {}
            token_set = set(all_tokens)
            for orig_token, prev_token, two_stage_pairs in split_raw["reactive_all_mapping"]:
                if prev_token in token_set or orig_token in token_set:
                    all_mappings[(str(orig_token), str(prev_token))] = [(str(p[0]), str(p[1])) for p in two_stage_pairs]
        else:
            sf_raw = yaml.load(open(f"{NAVSIM22}/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"), Loader=Loader)
            all_tokens = [str(t) for t in sf_raw["tokens"]]
            all_mappings = {}

        row_cache: Dict[Tuple[str, bytes], Dict[str, Any]] = {}
        ec_cache: Dict[Tuple[str, bytes, str, bytes], float] = {}
        start_adj_cache: Dict[str, str] = {}
        workers = 36

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=ep_v7._init_worker,
            initargs=(str(mc_dir), split),
        ) as ex:
            for mode in ["ep22", "ep29"]:
                prev_json_path = V7_DIR / f"opt_results/epdms_{split}_{mode}.json"
                prev_res = json.load(open(prev_json_path))["all_results"] if prev_json_path.exists() else {}

                toks, cand96, lat96, v2_final, v2_im, v2_sim, logit96, valid96 = ep_v7.load_pool(mode, split)
                sels = compute_official_indices(v2_final, v2_im, v2_sim, logit96, valid96)
                ar = np.arange(len(toks))
                res_mode = {}
                print(f"\n==================== v2_v15_v7_main (64m) {mode} : {split} EPDMS ====================", flush=True)
                for k in epdms_keys_to_eval:
                    if k in precomp_map and precomp_map[k] in prev_res:
                        r = prev_res[precomp_map[k]]
                        res_mode[k] = r
                        if split == "navhard":
                            g_aft, s1_aft, s2_aft = r["after"]["combined"], r["after"]["stage1"], r["after"]["stage2"]
                            g_bef, s1_bef, s2_bef = r["before"]["combined"], r["before"]["stage1"], r["before"]["stage2"]
                            print(f"[navhard:{k:48s}] (cached) | After={g_aft['score']:.2f}% (S1={s1_aft['score']:.2f}%, S2={s2_aft['score']:.2f}%) | Before={g_bef['score']:.2f}% (S1={s1_bef['score']:.2f}%, S2={s2_bef['score']:.2f}%)", flush=True)
                        else:
                            avg_aft, avg_bef = r["after"], r["before"]
                            print(f"[navtest:{k:48s}] (cached) | After={avg_aft['score']:.2f}% (NC={avg_aft['no_at_fault_collisions']:.2f}, DAC={avg_aft['drivable_area_compliance']:.2f}, EP={avg_aft['ego_progress']:.2f}, TTC={avg_aft['time_to_collision_within_bound']:.2f}, LK={avg_aft['lane_keeping']:.2f}, EC={avg_aft['two_frame_extended_comfort']:.2f}) | Before={avg_bef['score']:.2f}%", flush=True)
                        continue
                    idx = sels[k]
                    chosen = lat96[ar, idx]
                    traj_dict = {toks[i]: chosen[i] for i in range(len(toks))}
                    if split == "navhard":
                        r = ep_v7.evaluate_config_navhard(k, traj_dict, all_tokens, all_mappings, row_cache, ec_cache, ex, workers)
                    else:
                        r = ep_v7.evaluate_config_navtest(k, traj_dict, all_tokens, start_adj_cache, row_cache, ec_cache, ex, workers)
                    res_mode[k] = r
                summary_out[f"v7_64m_{mode}_{split}_epdms"] = res_mode

    out_path = V7_DIR / "opt_results/official_pdms_weights_v7.json"
    with open(out_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved all results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
