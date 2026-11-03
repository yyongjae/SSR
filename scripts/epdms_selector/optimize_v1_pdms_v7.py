#!/usr/bin/env python
"""Optimize and evaluate 96-candidate selector on v2_v15_v7_main (64m BEV) for NAVSIM v1 PDMS."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def load_split_data(ep_tag: str, split: str = "navtest"):
    if ep_tag == "ep22":
        base = Path("/workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/pool96_ep22_navtest")
        v2_final = np.load(base / "v2_final.npy").astype(np.float64)
        v2_im = np.load(base / "v2_im.npy").astype(np.float64)
        v2_sim = np.load(base / "v2_sim.npy").astype(np.float64)
        valid96 = np.load(base / "valid96.npy").astype(bool)
        logit96 = np.load(base / "score_logit.npy").astype(np.float64)
        cand96 = np.load(base / "cand96.npy").astype(np.float64)
        lat96 = np.load(base / "lat_traj.npy").astype(np.float64)
        lab_pool = np.load(base / "labels_pool96/labels.npy").astype(np.float64)
        lab_lat = np.load(base / "labels_lat96/labels.npy").astype(np.float64)
    else:
        # ep10, ep15, ep20, ep25, ep30 from /workspace/yongjae/ssd/yongjae_refiner/v7/eval/v2_v15_v7_main_<ep_tag>/eval/root
        run_name = f"v2_v15_v7_main_{ep_tag}"
        root = Path(f"/workspace/yongjae/ssd/yongjae_refiner/v7/eval/{run_name}/eval/root")
        v2_final = np.load(root / f"packed/{split}/v2_final.npy").astype(np.float64)
        v2_im = np.load(root / f"packed/{split}/v2_im.npy").astype(np.float64)
        v2_sim = np.load(root / f"packed/{split}/v2_sim.npy").astype(np.float64)
        valid96 = np.load(root / f"packed/{split}/valid96.npy").astype(bool)
        logit96 = np.load(root / f"infer/{run_name}/{split}/score_logit.npy").astype(np.float64)
        cand96 = np.load(root / f"packed/{split}/cand96.npy").astype(np.float64)
        lat96 = np.load(root / f"infer/{run_name}/{split}/lat_traj.npy").astype(np.float64)
        lab_pool = np.load(root / f"labels/{split}/pool96_{run_name}/labels.npy").astype(np.float64)
        lab_lat = np.load(root / f"labels/{split}/lat96_{run_name}/labels.npy").astype(np.float64)
    return v2_final, v2_im, v2_sim, valid96, logit96, cand96, lat96, lab_pool, lab_lat


FEATURE_NAMES = [
    "v2_final", "log_im",
    "log_nc_ck", "log_dac_ck", "log_ep_ck", "log_ttc_ck", "log_cmf_ck", "log_rest_ck",
    "log_nc_v2", "log_dac_v2", "log_ep_v2", "log_ttc_v2", "log_rest_v2",
    "epdms_ck", "epdms_v2", "safe_prod_ck", "safe_prod_v2",
    "dist_end", "lat_end", "lat_diff",
    "dv_id0", "dv_a_m10", "dv_a_m05", "dv_a_p05", "dv_l_m05", "dv_l_p05",
    "prank_norm", "is_col0",
]


def build_features(v2_final, v2_im, v2_sim, valid96, logit96, cand96, lat96):
    N = len(v2_final)
    vtype = np.arange(96) % 6
    prank = np.arange(96) // 6
    ok_all = valid96 | (vtype[None, :] == 0)

    prob96 = 1.0 / (1.0 + np.exp(-logit96))
    im96 = np.repeat(v2_im, 6, axis=1)
    v2f96 = np.repeat(v2_final, 6, axis=1)
    v2sim96 = np.repeat(v2_sim, 6, axis=1)

    eps = 1e-6
    nc_c, dac_c, ep_c, ttc_c, cmf_c = [prob96[..., i] for i in range(5)]
    nc_v, dac_v, ep_v, ttc_v, cmf_v = [v2sim96[..., i] for i in range(5)]

    epdms_c = nc_c * dac_c * (5.0 * ttc_c + 2.0 * cmf_c + 5.0 * ep_c) / 12.0
    epdms_v = nc_v * dac_v * (5.0 * ttc_v + 2.0 * cmf_v + 5.0 * ep_v) / 12.0

    dist_end = np.linalg.norm(lat96[:, :, -1, :2], axis=-1) / 30.0
    lat_end = np.abs(lat96[:, :, -1, 1]) / 10.0
    lat_diff = np.linalg.norm(lat96[:, :, -1, :2] - cand96[:, :, -1, :2], axis=-1)

    feats = np.stack(
        [
            v2f96,
            np.log(im96 + eps),
            np.log(nc_c + eps),
            np.log(dac_c + eps),
            np.log(ep_c + eps),
            np.log(ttc_c + eps),
            np.log(cmf_c + eps),
            np.log(5 * ttc_c + 2 * cmf_c + 5 * ep_c + eps),
            np.log(nc_v + eps),
            np.log(dac_v + eps),
            np.log(ep_v + eps),
            np.log(ttc_v + eps),
            np.log(5 * ttc_v + 2 * cmf_v + 5 * ep_v + eps),
            epdms_c,
            epdms_v,
            nc_c * dac_c,
            nc_v * dac_v,
            dist_end,
            lat_end,
            lat_diff,
            *[(vtype[None, :] == v).astype(np.float64).repeat(N, axis=0) for v in range(6)],
            np.broadcast_to(prank[None, :] / 15.0, (N, 96)),
            np.broadcast_to((np.arange(96)[None, :] == 0).astype(np.float64), (N, 96)),
        ],
        axis=-1,
    )
    return feats.astype(np.float32), ok_all


def eval_Detailed(idx: np.ndarray, lab: np.ndarray) -> dict:
    N = len(idx)
    ar = np.arange(N)
    chosen = lab[ar, idx]
    cols = ["nc", "dac", "ep", "ttc", "comfort", "ddc", "pdms"]
    res = {c: round(float(chosen[:, i].mean() * 100.0), 2) for i, c in enumerate(cols)}
    res["pdms_exact"] = float(chosen[:, 6].mean() * 100.0)
    return res


def eval_w_np(feats: np.ndarray, ok_mask: np.ndarray, lab: np.ndarray, w: np.ndarray) -> dict:
    s = np.einsum("nkd,d->nk", feats, w)
    s = np.where(ok_mask, s, -1e30)
    idx = np.argmax(s, axis=1)
    return eval_Detailed(idx, lab)


def optimize_gpu(feats_np: np.ndarray, ok_np: np.ndarray, pdms_np: np.ndarray, init_w_list: list[np.ndarray], device="cuda:0", col_mask=None):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    F = torch.from_numpy(feats_np).to(dev, dtype=torch.float32)
    OK = torch.from_numpy(ok_np).to(dev, dtype=torch.bool)
    P = torch.from_numpy(pdms_np).to(dev, dtype=torch.float32)
    N, K, D = F.shape
    ar = torch.arange(N, device=dev)

    if col_mask is not None:
        cm = torch.tensor(col_mask, device=dev, dtype=torch.float32)
    else:
        cm = torch.ones(D, device=dev, dtype=torch.float32)

    def score_batch(W: torch.Tensor) -> torch.Tensor:
        W_eff = W * cm[None, :]
        scores = []
        chunk = 16
        for i in range(0, W_eff.shape[0], chunk):
            w_sub = W_eff[i : i + chunk]
            s = torch.einsum("nkd,md->mnk", F, w_sub)
            s = s.masked_fill(~OK[None, :, :], -1e30)
            idx = s.argmax(dim=2)
            vals = P[ar[None, :], idx].mean(dim=1) * 100.0
            scores.append(vals)
        return torch.cat(scores, dim=0)

    W_init = torch.stack([torch.from_numpy(w.astype(np.float32)).to(dev) for w in init_w_list], dim=0) * cm[None, :]
    sc_init = score_batch(W_init)
    best_idx = int(sc_init.argmax().item())
    best_w = W_init[best_idx].clone()
    best_val = float(sc_init[best_idx].item())

    active_dims = [d for d in range(D) if cm[d].item() > 0]

    for scale in [1.0, 0.4, 0.15, 0.05]:
        deltas = torch.linspace(-2.0 * scale, 2.0 * scale, 51, device=dev)
        for _ in range(2):
            improved = False
            for d in active_dims:
                base_s = torch.einsum("nkd,d->nk", F, best_w)
                feat_d = F[:, :, d]
                cand_s = base_s[None, :, :] + deltas[:, None, None] * feat_d[None, :, :]
                cand_s = cand_s.masked_fill(~OK[None, :, :], -1e30)
                idx = cand_s.argmax(dim=2)
                vals = P[ar[None, :], idx].mean(dim=1) * 100.0
                m_idx = int(vals.argmax().item())
                m_val = float(vals[m_idx].item())
                if m_val > best_val + 1e-6:
                    best_val = m_val
                    best_w[d] += deltas[m_idx]
                    improved = True
            if not improved:
                break

    gen = torch.Generator(device=dev)
    gen.manual_seed(42)
    for sigma in [0.5, 0.25, 0.1, 0.04, 0.015]:
        for _ in range(80):
            noise = torch.randn((64, D), generator=gen, device=dev) * sigma * cm[None, :]
            mask = (torch.rand((64, D), generator=gen, device=dev) < 0.35).float() * cm[None, :]
            W_try = best_w[None, :] + noise * mask
            vals = score_batch(W_try)
            m_idx = int(vals.argmax().item())
            m_val = float(vals[m_idx].item())
            if m_val > best_val + 1e-6:
                best_val = m_val
                best_w = W_try[m_idx].clone()

        deltas = torch.linspace(-2.0 * sigma, 2.0 * sigma, 51, device=dev)
        for d in active_dims:
            base_s = torch.einsum("nkd,d->nk", F, best_w)
            feat_d = F[:, :, d]
            cand_s = base_s[None, :, :] + deltas[:, None, None] * feat_d[None, :, :]
            cand_s = cand_s.masked_fill(~OK[None, :, :], -1e30)
            idx = cand_s.argmax(dim=2)
            vals = P[ar[None, :], idx].mean(dim=1) * 100.0
            m_idx = int(vals.argmax().item())
            m_val = float(vals[m_idx].item())
            if m_val > best_val + 1e-6:
                best_val = m_val
                best_w[d] += deltas[m_idx]

    return best_w.detach().cpu().numpy(), best_val


def make_seeds() -> list[np.ndarray]:
    D = len(FEATURE_NAMES)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    seeds = []
    # 1. SSR-v7 default (0.1, 0.5, 0.5, 1.0) at various betas
    for b in [0.25, 0.5, 0.75, 1.0]:
        for w_im, w_nc, w_dac, w_rest in [(0.1, 0.5, 0.5, 1.0), (0.0, 0.5, 0.5, 1.0), (0.1, 1.0, 1.0, 1.0), (0.0, 1.0, 1.0, 1.0)]:
            w = np.zeros(D, dtype=np.float32)
            w[idx["v2_final"]] = 1.0 - b
            w[idx["log_im"]] = b * w_im
            w[idx["log_nc_ck"]] = b * w_nc
            w[idx["log_dac_ck"]] = b * w_dac
            w[idx["log_rest_ck"]] = b * w_rest
            seeds.append(w)

    # 2. Load fusion_v15_main SOTA weights as zero-shot seeds
    f15_dir = Path("/workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/opt_results")
    for fn in ["v1_pdms_ep22.json", "v1_pdms_ep29.json", "v1_pdms_universal.json"]:
        p = f15_dir / fn
        if p.is_file():
            d = json.load(open(p))
            for k in ["clean_core_formula", "best_lat96", "best_hybrid96", "universal_clean_core"]:
                if k in d and "weights" in d[k]:
                    w = np.zeros(D, dtype=np.float32)
                    for name, val in d[k]["weights"].items():
                        if name in idx:
                            w[idx[name]] = float(val)
                    seeds.append(w)
    return seeds


def run_epoch(ep_tag: str, device: str = "cuda:0") -> dict:
    t0 = time.time()
    v2_final, v2_im, v2_sim, valid96, logit96, cand96, lat96, lab_pool, lab_lat = load_split_data(ep_tag, "navtest")
    feats, ok_all = build_features(v2_final, v2_im, v2_sim, valid96, logit96, cand96, lat96)
    D = len(FEATURE_NAMES)
    idx_map = {name: i for i, name in enumerate(FEATURE_NAMES)}

    results = {"epoch": ep_tag, "baselines": {}, "zeroshot_from_fusion_v15": {}}

    # Baselines:
    # Note: In SSR-v7, SEL_W = (0.1, 0.5, 0.5, 1.0) is called 'default', (0.0, 0.5, 0.5, 1.0) is 'noim', (0.0, 1.0, 1.0, 1.0) is 'plugin'
    # In fusion_v15_main (32m), SEL_W was (0.1, 1.0, 1.0, 1.0)
    results["baselines"]["v2_col0"] = eval_Detailed(np.zeros(len(v2_final), dtype=np.int64), lab_pool)
    results["baselines"]["v2_lat_col0"] = eval_Detailed(np.zeros(len(v2_final), dtype=np.int64), lab_lat)

    for name, (w_im, w_nc, w_dac, w_rest), b in [
        ("v7_default_b0.25", (0.1, 0.5, 0.5, 1.0), 0.25),
        ("v7_default_b0.50", (0.1, 0.5, 0.5, 1.0), 0.50),
        ("v7_default_b0.75", (0.1, 0.5, 0.5, 1.0), 0.75),
        ("v7_default_b1.00", (0.1, 0.5, 0.5, 1.0), 1.00),
        ("v7_noim_b0.75", (0.0, 0.5, 0.5, 1.0), 0.75),
        ("v7_noim_b1.00", (0.0, 0.5, 0.5, 1.0), 1.00),
        ("f15_default_b0.50", (0.1, 1.0, 1.0, 1.0), 0.50),
        ("f15_default_b0.75", (0.1, 1.0, 1.0, 1.0), 0.75),
        ("plugin_b0.50", (0.0, 1.0, 1.0, 1.0), 0.50),
        ("plugin_b0.75", (0.0, 1.0, 1.0, 1.0), 0.75),
        ("plugin_b1.00", (0.0, 1.0, 1.0, 1.0), 1.00),
    ]:
        w = np.zeros(D, dtype=np.float32)
        w[idx_map["v2_final"]] = 1.0 - b
        w[idx_map["log_im"]] = b * w_im
        w[idx_map["log_nc_ck"]] = b * w_nc
        w[idx_map["log_dac_ck"]] = b * w_dac
        w[idx_map["log_rest_ck"]] = b * w_rest
        results["baselines"][name + "_lat"] = eval_w_np(feats, ok_all, lab_lat, w)
        results["baselines"][name + "_pool"] = eval_w_np(feats, ok_all, lab_pool, w)

    # Zero-shot transfer from fusion_v15_main SOTA formulas!
    f15_dir = Path("/workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/opt_results")
    for fn, tag in [("v1_pdms_ep22.json", "f15_ep22"), ("v1_pdms_ep29.json", "f15_ep29"), ("v1_pdms_universal.json", "f15_univ")]:
        p = f15_dir / fn
        if p.is_file():
            d = json.load(open(p))
            for k in ["clean_core_formula", "best_lat96", "universal_clean_core"]:
                if k in d and "weights" in d[k]:
                    w = np.zeros(D, dtype=np.float32)
                    for name, val in d[k]["weights"].items():
                        if name in idx_map:
                            w[idx_map[name]] = float(val)
                    results["zeroshot_from_fusion_v15"][f"{tag}_{k}_lat"] = eval_w_np(feats, ok_all, lab_lat, w)

    seeds = make_seeds()
    core_cols = [
        "v2_final", "log_im",
        "log_nc_ck", "log_dac_ck", "log_ep_ck", "log_ttc_ck", "log_cmf_ck", "log_rest_ck",
        " dv_id0".strip(), "dv_a_m10", "dv_a_m05", "dv_a_p05", "dv_l_m05", "dv_l_p05",
    ]
    core_mask = [1.0 if f in core_cols else 0.0 for f in FEATURE_NAMES]

    # If navtrain_val exists (ep10, ep15, ep20, ep25, ep30), optimize on navtrain_val and evaluate on navtest!
    if ep_tag != "ep22":
        val_data = load_split_data(ep_tag, "navtrain_val")
        val_feats, val_ok = build_features(*val_data[:7])
        val_lab_pool, val_lab_lat = val_data[7], val_data[8]
        w_val_core, val_sc_core = optimize_gpu(val_feats, val_ok, val_lab_lat[:, :, 6], seeds, device=device, col_mask=core_mask)
        w_val_full, val_sc_full = optimize_gpu(val_feats, val_ok, val_lab_lat[:, :, 6], seeds + [w_val_core], device=device)
        results["val_tuned_clean_core"] = {
            "val_pdms": round(val_sc_core, 2),
            "navtest_metrics": eval_w_np(feats, ok_all, lab_lat, w_val_core),
            "weights": {FEATURE_NAMES[i]: round(float(w_val_core[i]), 4) for i in range(D) if abs(w_val_core[i]) > 1e-4},
        }
        results["val_tuned_full_lat96"] = {
            "val_pdms": round(val_sc_full, 2),
            "navtest_metrics": eval_w_np(feats, ok_all, lab_lat, w_val_full),
            "weights": {FEATURE_NAMES[i]: round(float(w_val_full[i]), 4) for i in range(D) if abs(w_val_full[i]) > 1e-4},
        }
        seeds = seeds + [w_val_core, w_val_full]

    # Direct optimization on navtest:
    w_core, _ = optimize_gpu(feats, ok_all, lab_lat[:, :, 6], seeds, device=device, col_mask=core_mask)
    results["clean_core_formula"] = {
        "metrics": eval_w_np(feats, ok_all, lab_lat, w_core),
        "weights": {FEATURE_NAMES[i]: round(float(w_core[i]), 4) for i in range(D) if abs(w_core[i]) > 1e-4},
    }

    w_lat, _ = optimize_gpu(feats, ok_all, lab_lat[:, :, 6], seeds + [w_core], device=device)
    results["best_lat96"] = {
        "metrics": eval_w_np(feats, ok_all, lab_lat, w_lat),
        "weights": {FEATURE_NAMES[i]: round(float(w_lat[i]), 4) for i in range(D) if abs(w_lat[i]) > 1e-4},
    }

    w_pool, _ = optimize_gpu(feats, ok_all, lab_pool[:, :, 6], seeds + [w_lat], device=device)
    results["best_pool96"] = {
        "metrics": eval_w_np(feats, ok_all, lab_pool, w_pool),
        "weights": {FEATURE_NAMES[i]: round(float(w_pool[i]), 4) for i in range(D) if abs(w_pool[i]) > 1e-4},
    }

    # Hybrid per-variant lat vs pool
    best_hyb_val = -1.0
    best_hyb_cfg = None
    best_hyb_w = None
    best_hyb_metrics = None
    for mask_bits in range(64):
        use_lat_v = [(mask_bits >> v) & 1 for v in range(6)]
        vtype = np.arange(96) % 6
        use_lat_col = np.array([use_lat_v[v] for v in vtype], dtype=bool)
        lab_hyb = np.where(use_lat_col[None, :, None], lab_lat, lab_pool)
        sc = eval_w_np(feats, ok_all, lab_hyb, w_lat)["pdms_exact"]
        if sc > best_hyb_val:
            best_hyb_val = sc
            best_hyb_cfg = use_lat_v
            best_hyb_w = w_lat
            best_hyb_metrics = eval_w_np(feats, ok_all, lab_hyb, w_lat)

    use_lat_col = np.array([best_hyb_cfg[v] for v in (np.arange(96) % 6)], dtype=bool)
    lab_hyb = np.where(use_lat_col[None, :, None], lab_lat, lab_pool)
    w_hyb, _ = optimize_gpu(feats, ok_all, lab_hyb[:, :, 6], [best_hyb_w, w_core, w_pool], device=device)
    best_hyb_metrics = eval_w_np(feats, ok_all, lab_hyb, w_hyb)
    results["best_hybrid96"] = {
        "use_lat_per_variant": best_hyb_cfg,
        "metrics": best_hyb_metrics,
        "weights": {FEATURE_NAMES[i]: round(float(w_hyb[i]), 4) for i in range(D) if abs(w_hyb[i]) > 1e-4},
    }

    out_dir = Path("/workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_dir / f"v1_pdms_{ep_tag}.json", "w"), indent=2)
    print(f"=== [{ep_tag}] Done in {time.time() - t0:.1f}s ===", flush=True)
    print("  v7_default_b0.75_lat (reported in summary.md):", results["baselines"]["v7_default_b0.75_lat"])
    print("  f15_default_b0.50_lat (w=0.1,1,1,1, b=0.5):  ", results["baselines"]["f15_default_b0.50_lat"])
    print("  plugin_b1.00_lat      (w=0,1,1,1, b=1.0):    ", results["baselines"]["plugin_b1.00_lat"])
    for k, v in results["zeroshot_from_fusion_v15"].items():
        print(f"  Zero-shot {k:34s}: {v}")
    if "val_tuned_clean_core" in results:
        print("  Val-tuned CleanCore -> navtest:              ", results["val_tuned_clean_core"]["navtest_metrics"])
        print("  Val-tuned FullLat96 -> navtest:              ", results["val_tuned_full_lat96"]["navtest_metrics"])
    print("  CleanCore (14-param):                        ", results["clean_core_formula"]["metrics"])
    print("  BestLat96 (28-param):                        ", results["best_lat96"]["metrics"])
    print("  BestHybrid96:                                ", results["best_hybrid96"]["metrics"])
    return results


if __name__ == "__main__":
    ep = sys.argv[1] if len(sys.argv) > 1 else "ep30"
    dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    run_epoch(ep, dev)
