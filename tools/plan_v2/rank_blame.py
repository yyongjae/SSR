"""Which head loses the ranking?  For scenes where the deployed (field-gated) pick is feasible but a
better feasible candidate existed, attribute the miss to the terms of the selection rule.

    final = w_im log im + w_r log(5 TTC + 2 C + 5 EP)   - gate(feasible head < thr)   - gate(field)

For each such scene, with b = the best really-feasible candidate (real PDMS) and p = the pick:

    feasible-head veto   b was pushed down by the FEASIBLE HEAD's gate (head < 0.5) although it is
                         really feasible -- the learned copy of the corridor overrules the field
    field veto           b was rejected by the predicted-field gate although really drivable
    preference miss      b passed both gates, and the preference terms ranked p above b; split by
                         which term carried the wrong sign: im (imitation), EP head, TTC head, C head

    python tools/plan_v2/rank_blame.py --dump <field_*.npz> --table <candidate_pdms_*.npz>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
from tools.plan_v2.diagnose_arms import clearance                                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--layout", default="corridor")
    a = ap.parse_args()
    z = np.load(a.dump)
    S = np.load(a.table)["scores"].astype(np.float64)
    NC, DAC, EP, TTC, C, P = (S[..., i] for i in (0, 1, 3, 4, 5, 6))
    n = len(P); ix = np.arange(n)
    A = torch.from_numpy(np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy")).float()
    sim = z["sim_rewards"].astype(np.float64); im = z["im_rewards"].astype(np.float64)
    if a.layout == "corridor":
        feas, ep, ttc, c = sim[:, 0], sim[:, 1], sim[:, 2], sim[:, 3]
    else:
        feas, ep, ttc, c = sim[:, 1], sim[:, 2], sim[:, 3], sim[:, 4]
    eps = 1e-6
    t_im, t_rest = 0.1 * np.log(im + eps), np.log(5 * ttc + 2 * c + 5 * ep + eps)
    head_ok = feas >= 0.5
    cl = clearance(z["trajectory_offset"], z["drivable_sdf_pred"].astype(np.float32), A)
    field_ok = cl >= 0
    pref = t_im + t_rest
    any_h, any_f = head_ok.any(1, keepdims=True), field_ok.any(1, keepdims=True)
    final = pref - 20.0 * (~head_ok & any_h) - 20.0 * (~field_ok & any_f)
    p = final.argmax(1)
    real_ok = (NC == 1) & (DAC == 1)
    b = np.where(real_ok, P, -1).argmax(1)
    has = real_ok.any(1)
    gap = np.where(has, P[ix, b] - P[ix, p], 0.0)
    pick_ok = real_ok[ix, p]
    miss = has & pick_ok & (gap > 1e-6)
    print(f"배포 선택 PDMS {100 * P[ix, p].mean():.2f}   가능 후보 최고 {100 * np.where(has, P[ix, b], P[ix, p]).mean():.2f}")
    print(f"선택은 가능하지만 더 좋은 가능 후보가 있던 장면 {100 * miss.mean():.1f}%  → 손실 {100 * gap[miss].sum() / n:.2f}점")
    hv = miss & ~head_ok[ix, b] & head_ok[ix, p]
    fv = miss & ~hv & ~field_ok[ix, b] & field_ok[ix, p]
    pm = miss & ~hv & ~fv
    for nm, m in (("feasible 헤드가 최고 후보를 거부", hv), ("예측 장 게이트가 최고 후보를 거부", fv), ("둘 다 통과, 선호 점수가 틀림", pm)):
        print(f"  {nm:30s} 장면 {100 * m.mean():5.1f}%   손실 {100 * gap[m].sum() / n:5.2f}점")
    # preference misses: which term put p above b?
    d_im = t_im[ix, p] - t_im[ix, b]
    d_rest = t_rest[ix, p] - t_rest[ix, b]
    print(f"  └ 선호 오류 중 im 항이 주범(|d_im|>|d_rest| 이고 p 편) {100 * (pm & (d_im > 0) & (np.abs(d_im) > np.abs(d_rest))).sum() / max(pm.sum(), 1):.1f}%, "
          f"rest 항이 주범 {100 * (pm & (d_rest > 0) & (np.abs(d_rest) >= np.abs(d_im))).sum() / max(pm.sum(), 1):.1f}%")
    for nm, pred, true in (("EP", ep, EP), ("TTC", ttc, TTC), ("C", c, C)):
        dp, dt = pred[ix, p] - pred[ix, b], true[ix, p] - true[ix, b]
        wrong = pm & (dp > 0) & (dt < 0)
        print(f"  └ {nm:3s} 헤드: p 를 더 높게 예측했지만 실제로는 b 가 더 높음 {100 * wrong.sum() / max(pm.sum(), 1):5.1f}%   "
              f"(실제 차이 평균 {np.mean(dt[pm]):+.3f}, 예측 차이 평균 {np.mean(dp[pm]):+.3f})")
    # is the true best even close in score?  rank of b under the deployed rule
    rank_b = (final[pm] > final[pm, None].squeeze(1)[np.arange(pm.sum()), b[pm]][:, None]).sum(1) if pm.any() else np.array([0])
    print(f"  └ 선호 오류 장면에서 최고 후보의 순위 중앙값 {np.median(rank_b):.0f} / 256 (상위 5 안 {100 * (rank_b < 5).mean():.1f}%)")


if __name__ == "__main__":
    main()
