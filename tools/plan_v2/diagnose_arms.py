"""Where does each arm lose its points?  Pool vs filter vs ranking, from the real scorer's per-candidate table.

For every arm with a dump (all 256 candidates + head outputs) and a candidate table
(pdms_all_candidates.py: the scorer's own NC/DAC/DDC/EP/TTC/C/PDMS for every candidate), this splits
the gap between the pool's ceiling and the deployed pick into

    filter loss   the pick is infeasible (NC < 1 or DAC < 1): the model let a bad candidate through
    rank loss     the pick is feasible but a better feasible candidate existed

and measures the parts that produce them: the feasibility verdict (head and field, AUROC against the
real NC*DAC), the ranking (per-scene Spearman of the final score with real PDMS among the really
feasible), and two swaps that say which part is worth fixing:

    perfect filter + model ranking   argmax of the model's score among the REALLY feasible
    field-gate filter + perfect rank argmax of real PDMS among candidates the field gate admits

The field gate is feasibility.field_gate_select's rule (static margin 0.1, dynamic off), applied with
the predicted field and with the GT field (the latter also on arms that never trained a field).

    python tools/plan_v2/diagnose_arms.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
from navsim.agents.para_ssr.feasibility import corridor_clearance            # noqa: E402

R = ROOT / "work_dirs/eval_epdms"
ARMS = [  # name, dump, candidate table, sim_rewards layout
    ("v2_r34", "diag_para_ssr_v2_r34/dump.npz", "para_ssr_v2_r34", "wote"),
    ("feas_gt", "field_para_ssr_v2_r34_feas_gt.npz", "para_ssr_v2_r34_feas_gt", "wote"),
    ("feas_pred_seg_ch", "field_para_ssr_v2_r34_feas_pred_seg_ch.npz", "feas_pred_seg_ch", "corridor"),
    ("B3", "field_para_ssr_v2_r34_feas_gt_select_ttcband_hold_ep23.npz", "b3_ep23", "wote"),
    ("D+", "field_para_ssr_v2_r34_feas_pred_seg_ch_select_ttcband_hold_epgeo_sees_drive.npz", "dplus", "corridor"),
]


class Cfg:
    map_pc_range = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)
    feas_dyn_weight = 0.0


def clearance(off, field, anchors, bs=64):
    out = []
    for s in range(0, len(off), bs):
        p = {"trajectory_anchors": anchors, "trajectory_offset": torch.from_numpy(off[s:s + bs]).float(),
             "drivable_sdf_pred": torch.from_numpy(field[s:s + bs]).float()}
        out.append(corridor_clearance(p, Cfg, 0.1, -1e9).numpy())
    return np.concatenate(out)


def auroc(score, label):
    s, y = score.ravel(), label.ravel().astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.empty(len(s)); r[np.argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
    npos = y.sum()
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * (len(s) - npos))


def scene_spearman(a, b, mask, min_n=5):
    rho = []
    for i in range(len(a)):
        m = mask[i]
        if m.sum() < min_n:
            continue
        x, y = a[i, m], b[i, m]
        if np.ptp(y) == 0 or np.ptp(x) == 0:
            continue
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        rho.append(np.corrcoef(rx, ry)[0, 1])
    return np.median(rho), len(rho)


def masked_argmax(v, m):
    return np.where(m, v, -np.inf).argmax(1)


def main():
    anchors = torch.from_numpy(np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy")).float()
    g = np.load(R / "gt_field_navtest.npz")
    gt_tok = {str(t): i for i, t in enumerate(g["tokens"])}
    rows = []
    for name, dump, table, layout in ARMS:
        f = R / f"candidate_pdms_{table}.npz"
        if not f.exists():
            print(f"{name}: 후보 표 없음, 건너뜀"); continue
        z, c = np.load(R / dump), np.load(f)
        assert (z["tokens"] == c["tokens"]).all()
        S = c["scores"].astype(np.float64)                      # NC DAC DDC EP TTC C PDMS
        n = len(S); ix = np.arange(n)
        NC, DAC, EP, TTC, C, P = S[..., 0], S[..., 1], S[..., 3], S[..., 4], S[..., 5], S[..., 6]
        feas = (NC == 1) & (DAC == 1)
        pick = z["plan_topk_index"][:, 0]
        final = z["plan_final_rewards"].astype(np.float64)
        sim = z["sim_rewards"].astype(np.float64)
        head_feas = sim[:, 0] if layout == "corridor" else sim[:, 0] * sim[:, 1]
        head_ep = sim[:, 1] if layout == "corridor" else sim[:, 2]
        off = z["trajectory_offset"]
        o = {"arm": name}
        # ---- pool
        traj_end = (anchors.numpy()[None, :, -1, :2] + off[:, :, -1, :2].astype(np.float32))
        spread = []
        for i in range(0, n, 7):                                # subsample for the spread
            m = feas[i]
            if m.sum() >= 2:
                spread.append(traj_end[i, m].std(0).sum())
        o.update(ceiling=100 * P.max(1).mean(), pool_mean=100 * P.mean(), has_perfect=100 * (P.max(1) >= 0.999).mean(),
                 n_feas=np.median(feas.sum(1)), spread_feas=np.median(spread))
        # ---- deployed + decomposition
        pp = P[ix, pick]; gap = P.max(1) - pp
        pf = feas[ix, pick]
        o.update(deployed=100 * pp.mean(), filter_loss=100 * (gap * ~pf).sum() / n, rank_loss=100 * (gap * pf).sum() / n,
                 pick_nc_fail=100 * (NC[ix, pick] < 1).mean(), pick_dac_fail=100 * (DAC[ix, pick] < 1).mean(),
                 pick_ep=100 * EP[ix, pick].mean(), pick_ttc=100 * TTC[ix, pick].mean(), pick_c=100 * C[ix, pick].mean())
        best_f = masked_argmax(P, feas)
        okf = feas.any(1)
        o["ep_gap_feasible_pick"] = 100 * (EP[ix, best_f] - EP[ix, pick])[pf & okf].mean()
        # ---- heads
        o["auroc_head_feas"] = auroc(head_feas, feas)
        o["rho_final_vs_pdms_in_feas"], _ = scene_spearman(final, P, feas)
        o["rho_ep_head_vs_ep_in_feas"], _ = scene_spearman(head_ep, EP, feas)
        # ---- swaps
        o["perfect_filter_model_rank"] = 100 * P[ix, masked_argmax(final, feas | ~okf[:, None])].mean()
        # ---- field gate (pred field where the arm has one; GT field always)
        gi = np.array([gt_tok[str(t)] for t in z["tokens"]])
        fields = {"gt": g["drivable_sdf_gt"][gi]}
        if "drivable_sdf_pred" in z.files:
            fields["pred"] = z["drivable_sdf_pred"]
        for fk, fld in fields.items():
            cl = clearance(off, fld, anchors)
            adm = cl >= 0
            anyadm = adm.any(1, keepdims=True)
            gated = final - 20.0 * (~adm & anyadm)
            gp = gated.argmax(1)
            o[f"gate_{fk}"] = 100 * P[ix, gp].mean()
            o[f"gate_{fk}_dac"] = 100 * DAC[ix, gp].mean()
            o[f"auroc_field_{fk}_vs_dac"] = auroc(cl, DAC == 1)
            o[f"gate_{fk}_perfect_rank"] = 100 * P[ix, masked_argmax(P, adm | ~anyadm)].mean()
            o[f"gate_{fk}_false_reject"] = 100 * ((~adm) & (DAC == 1)).sum() / max((DAC == 1).sum(), 1)
            o[f"gate_{fk}_false_accept"] = 100 * (adm & (DAC < 1)).sum() / max((DAC < 1).sum(), 1)
        if "drivable_sdf_pred" in z.files and "pred" in fields:
            gf, pf_ = fields["gt"], fields["pred"]
            band = np.abs(gf) < 2.0
            o["field_mae_band"] = np.abs(pf_ - gf)[band].mean()
            o["field_sign_acc_band"] = 100 * ((pf_ >= 0) == (gf >= 0))[band].mean()
        rows.append(o)
        print(f"{name} 완료", flush=True)

    keys = []
    for o in rows:
        keys += [k for k in o if k not in keys and k != "arm"]
    print("\n" + "지표".ljust(30) + "".join(o["arm"].rjust(17) for o in rows))
    for k in keys:
        print(k.ljust(30) + "".join((f"{o[k]:17.3f}" if k in o else " " * 16 + "-") for o in rows))
    np.save(R / "diagnose_arms.npy", rows, allow_pickle=True)


if __name__ == "__main__":
    main()
