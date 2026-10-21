"""How well does each TTC label predict the scorer's TTC of the EMITTED trajectory?

Calls feasibility.ttc_samples itself, so what is measured is what trains -- the first version of
this check re-implemented the geometry inline and could have drifted from the code.

Labels compared, on the same 3000 navtest scenes x 256 candidates:
    anchor(v2)   navtest_anchor_scores.npz TTC column -- produced by the v2.2 scorer
    anchor(v1)   the anchors driven verbatim, scored by the v1 scorer (pdms_all_candidates.py on a
                 zero-offset dump) -- the SAME scorer as the training labels (pdm_score_256)
    geometric    ttc_samples on the ground-truth agent field, border / exempt outside the BEV

    python tools/plan_v2/ttc_label_fidelity.py --model v2_r34|feas_gt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
from navsim.agents.para_ssr.feasibility import ttc_samples      # noqa: E402

PC = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)
DUMPS = {"v2_r34": ("diag_para_ssr_v2_r34/dump.npz", "candidate_pdms_para_ssr_v2_r34.npz"),
         "feas_gt": ("field_para_ssr_v2_r34_feas_gt.npz", "candidate_pdms_para_ssr_v2_r34_feas_gt.npz")}


def auc(s, y):
    s, y = s.ravel().astype(np.float64), y.ravel()
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = y.sum(); n = len(y) - p
    return (r[y].sum() - p * (p + 1) / 2) / (p * n)


def report(name, score, y, thr):
    pred = score >= thr
    print(f"  {name:30s} AUC {auc(score, y):.4f}   정확도 {100 * (pred == y).mean():5.2f}%   "
          f"오판정 {100 * ((~pred) & y).mean():5.2f}%   놓침 {100 * (pred & ~y).mean():5.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v2_r34", choices=list(DUMPS))
    a = ap.parse_args()
    E = ROOT / "work_dirs/eval_epdms"
    sel = np.load(E / "anchors0_3000.npz")
    toks = [str(t) for t in sel["tokens"]]
    dump, cand = DUMPS[a.model]
    d = np.load(E / dump); dat = {str(t): i for i, t in enumerate(d["tokens"])}
    rows = np.array([dat[t] for t in toks])
    true = np.load(E / cand)["scores"][rows, :, 4]                      # v1 TTC of the EMITTED candidates
    y = true >= 0.5
    print(f"{a.model}: {len(rows)} 장면 x 256,  진짜 TTC=1 비율 {100 * y.mean():.1f}%\n")

    av = np.load(ROOT / "data/planning_vb/navtest_anchor_scores.npz", allow_pickle=True)
    ci = [str(c) for c in av["columns"]].index("time_to_collision_within_bound")
    aat = {str(t): i for i, t in enumerate(av["tokens"])}
    lab_v2 = np.asarray(av["scores"])[[aat[t] for t in toks], :256, ci]
    rng = np.random.default_rng(1)
    report("앵커 라벨 (v2 채점기)", lab_v2 + 1e-6 * rng.random(lab_v2.shape), y, 0.5)
    p1 = E / "candidate_pdms_anchors0_3000.npz"
    if p1.exists():
        z = np.load(p1)
        lab_v1 = z["scores"][:, :, 4]
        report("앵커 라벨 (v1 채점기, 학습과 동일)", lab_v1 + 1e-6 * rng.random(lab_v1.shape), y, 0.5)
    else:
        print("  앵커 라벨 (v1): 아직 없음")

    g = np.load(E / "gt_field_navtest.npz"); gat = {str(t): i for i, t in enumerate(g["tokens"])}
    dyn = torch.from_numpy(g["agent_sdf_gt"][[gat[t] for t in toks]].astype(np.float32))
    anc = torch.from_numpy(np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy")).float()
    traj = anc[None] + torch.from_numpy(d["trajectory_offset"][rows].astype(np.float32))
    for mode in ("border", "exempt"):
        s = ttc_samples(traj, dyn, PC, outside=mode)
        score = s.amin(dim=(2, 3, 4)).numpy()
        for thr in (0.0, 0.3):
            report(f"기하 ({mode}, margin {thr})", score, y, thr)


if __name__ == "__main__":
    main()
