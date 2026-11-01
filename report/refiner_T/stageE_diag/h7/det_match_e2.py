#!/usr/bin/env python3
"""H7: per-GT-object detection matching for E2's own det head (and E0 recomputed as a check against
report/perception_reliability/build_det/gt_objects.parquet 'inter.*'), with the build_det definitions:
  class-agnostic greedy (score desc, closest untaken GT, strict <) at 0.5/1/2 m, prediction kept iff score >= 0.3;
  class-aware 2 m; nn_dist (nearest tau-prediction, no 1:1); err_pos / err_vel / err_speed on the agn-2 m pair.
Also per-map-GT-point distance to nearest predicted road boundary polyline (tau 0.3, same class 0) for the DAC part
is done elsewhere (dac_map_e2.py).
Output: <OUT>/det_gt_e2.parquet (token, gt_idx, e2.*, e0re.*)
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

E0_REC = Path("/home/external-user/yongjae/SSR/work_dirs/eval/para_ssr_interaction_final_aux/records")
E2_REC = Path("/home/external-user/ssd/yongjae_refiner/stageE_diag/h7/e2_aux/records")
OUT = Path("/home/external-user/ssd/yongjae_refiner/stageE_diag/h7")
TAU = 0.3
AGN_THR = (0.5, 1.0, 2.0)


def greedy_match(D, scores, thresholds):  # verbatim logic of build_det_tables.greedy_match
    P, G = D.shape
    gt_match = np.full((len(thresholds), G), -1, np.int32)
    if P == 0 or G == 0:
        return gt_match
    order = np.lexsort((np.arange(P), -scores))
    row_min = D.min(axis=1)
    for ti, thr in enumerate(thresholds):
        taken = np.zeros(G, bool)
        for p in order:
            if row_min[p] >= thr:
                continue
            cand = np.flatnonzero(~taken)
            if not len(cand):
                break
            g = cand[int(np.argmin(D[p, cand]))]
            if D[p, g] < thr:
                taken[g] = True
                gt_match[ti, g] = p
    return gt_match


def per_model(pb, ps, pl, gb, gl, tag):
    P, G = len(pb), len(gb)
    D = (np.linalg.norm(pb[:, None, :2] - gb[None, :, :2], axis=-1).astype(np.float32)
         if P and G else np.zeros((P, G), np.float32))
    Dca = np.where(pl[:, None] == gl[None, :], D, np.inf).astype(np.float32)
    gm_ag = greedy_match(D, ps, AGN_THR)
    gm_ca = greedy_match(Dca, ps, (2.0,))
    above = ps >= TAU

    def matched(gm):
        return (gm >= 0) & np.where(gm >= 0, above[np.maximum(gm, 0)] if P else False, False)

    out = {}
    for ti, thr in enumerate(AGN_THR):
        out[f"{tag}.matched_agn@{thr:g}"] = matched(gm_ag[ti])
    out[f"{tag}.matched@2"] = matched(gm_ca[0])
    pair = np.where(matched(gm_ag[2]), gm_ag[2], -1)
    ok = pair >= 0
    q = np.maximum(pair, 0)
    nan = np.full(G, np.nan, np.float32)
    epos, evel, espd, msc, mcl = nan.copy(), nan.copy(), nan.copy(), nan.copy(), np.full(G, -1, np.int8)
    if ok.any():
        epos[ok] = np.linalg.norm(pb[q[ok], :2].astype(np.float64) - gb[ok, :2], axis=1)
        evel[ok] = np.linalg.norm(pb[q[ok], 7:9].astype(np.float64) - gb[ok, 7:9], axis=1)
        espd[ok] = np.linalg.norm(pb[q[ok], 7:9], axis=1) - np.linalg.norm(gb[ok, 7:9], axis=1)
        msc[ok] = ps[q[ok]]
        mcl[ok] = pl[q[ok]]
    out[f"{tag}.err_pos"], out[f"{tag}.err_vel"], out[f"{tag}.err_speed"] = epos, evel, espd
    out[f"{tag}.match_score"], out[f"{tag}.match_cls"] = msc, mcl
    out[f"{tag}.nn_dist"] = D[above].min(axis=0) if (P and G and above.any()) else np.full(G, np.inf, np.float32)
    # best score of ANY prediction within 2 m (no tau, no 1:1): 'did the head put any mass there'
    out[f"{tag}.max_score_2m"] = (np.where(D < 2.0, ps[:, None], 0.0).max(0) if (P and G)
                                  else np.zeros(G, np.float32))
    return out


def process(token):
    with np.load(E0_REC / f"{token}.npz") as z:
        gb = z["det_gt_boxes"].astype(np.float32)
        gl = z["det_gt_labels"].astype(np.int64)
        e0 = (z["det_pred_boxes"].astype(np.float32), z["det_pred_scores"].astype(np.float32),
              z["det_pred_labels"].astype(np.int64))
    with np.load(E2_REC / f"{token}.npz") as z:
        assert np.array_equal(z["det_gt_boxes"], gb)
        e2 = (z["det_pred_boxes"].astype(np.float32), z["det_pred_scores"].astype(np.float32),
              z["det_pred_labels"].astype(np.int64))
    G = len(gb)
    row = {"token": np.full(G, token), "gt_idx": np.arange(G, dtype=np.int32)}
    row.update(per_model(*e2, gb, gl, "e2"))
    row.update(per_model(*e0, gb, gl, "e0re"))
    npred = {"token": token, "e2.n_tau": int((e2[1] >= TAU).sum()), "e0.n_tau": int((e0[1] >= TAU).sum()), "n_gt": G}
    return row, npred


def main():
    tokens = sorted(p.stem for p in E2_REC.glob("*.npz"))
    print(len(tokens), "tokens", flush=True)
    rows, npr = [], []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        for r, n in pool.imap(process, tokens, chunksize=64):
            if len(r["token"]):
                rows.append(r)
            npr.append(n)
    df = pd.concat([pd.DataFrame(r) for r in rows], ignore_index=True)
    df.to_parquet(OUT / "det_gt_e2.parquet")
    pd.DataFrame(npr).to_parquet(OUT / "det_tok_e2.parquet")
    print(df.shape, flush=True)


if __name__ == "__main__":
    main()
