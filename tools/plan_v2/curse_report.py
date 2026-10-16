"""Selection-curse report for one arm: does the deployed pick's score head over-rate it, and by how much
more than a random candidate of the same deployment set?  Linear scale only (the log preference blows
up near 0 and inflates means), on the deployed rule (plan_final_rewards + static field gate, margin 0.1).

    pick over    head - true on the deployed pick, TTC and EP (and the linear preference (5T+2C+5E)/12)
    random over  the same for a uniformly random candidate of the field-gate-admitted set (seed 0)
    feasible-only rows restrict both to candidates with true NC = DAC = 1 (removes collision mixing)
    im-weight    PRE-DECLARED analysis grid {0.1 (deployed), 0.05, 0}: PDMS of the field-gated argmax of
                 w log im + log(5 TTC + 2 C + 5 EP) with the feasible-head gate as deployed

    python tools/plan_v2/curse_report.py --dump <field_*.npz> --table <candidate_pdms_*.npz>
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
    a = ap.parse_args()
    z = np.load(a.dump)
    S = np.load(a.table)["scores"].astype(np.float64)
    NC, DAC, EP, TTC, C, P = (S[..., i] for i in (0, 1, 3, 4, 5, 6))
    n = len(P); ix = np.arange(n)
    A = torch.from_numpy(np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy")).float()
    sim = z["sim_rewards"].astype(np.float64); im = z["im_rewards"].astype(np.float64)
    feas, ep, ttc, c = sim[:, 0], sim[:, 1], sim[:, 2], sim[:, 3]
    f = clearance(z["trajectory_offset"], z["drivable_sdf_pred"].astype(np.float32), A) >= 0
    fgate = -20.0 * (~f & f.any(1, keepdims=True))
    hgate = -20.0 * ((feas < 0.5) & (feas >= 0.5).any(1, keepdims=True))
    eps = 1e-6
    deployed = z["plan_final_rewards"].astype(np.float64) + fgate
    p = deployed.argmax(1)
    adm = np.where(f.any(1, keepdims=True), f, True)
    rng = np.random.default_rng(0)
    r = np.array([rng.choice(np.flatnonzero(adm[i])) for i in range(n)])
    lin_p, lin_t = (5 * ttc + 2 * c + 5 * ep) / 12, (5 * TTC + 2 * C + 5 * EP) / 12
    print(f"배포 선택 PDMS {100 * P[ix, p].mean():.2f}  (NC 실패 {100 * (NC[ix, p] < 1).mean():.2f}%, EP {100 * EP[ix, p].mean():.1f})")
    print("과대평가 (헤드 - 실제), 선형        선택      무작위(게이트 통과 집합)")
    for nm, h, t in (("TTC", ttc, TTC), ("EP", ep, EP), ("선호 (5T+2C+5E)/12", lin_p, lin_t)):
        print(f"  {nm:20s}  {np.mean(h[ix, p] - t[ix, p]):+.3f}    {np.mean(h[ix, r] - t[ix, r]):+.3f}")
    ok = (NC == 1) & (DAC == 1)
    m = ok[ix, p] & ok[ix, r]
    print(f"정답 NC·DAC=1 인 선택·무작위 쌍만 ({100 * m.mean():.1f}% 장면)")
    for nm, h, t in (("TTC", ttc, TTC), ("EP", ep, EP)):
        print(f"  {nm:20s}  {np.mean((h - t)[ix, p][m]):+.3f}    {np.mean((h - t)[ix, r][m]):+.3f}")
    print("im 가중 (사전 선언 격자, 장 게이트 + feasible 헤드 게이트 = 배포 방식)")
    for w in (0.1, 0.05, 0.0):
        s = w * np.log(im + eps) + np.log(5 * ttc + 2 * c + 5 * ep + eps) + hgate + fgate
        q = s.argmax(1)
        print(f"  w_im={w:<5}  PDMS {100 * P[ix, q].mean():.2f}  NC 실패 {100 * (NC[ix, q] < 1).mean():.2f}%  "
              f"TTC {100 * TTC[ix, q].mean():.1f}  EP {100 * EP[ix, q].mean():.1f}  DAC {100 * DAC[ix, q].mean():.1f}   "
              f"선택 과대 TTC {np.mean(ttc[ix, q] - TTC[ix, q]):+.3f}  EP {np.mean(ep[ix, q] - EP[ix, q]):+.3f}")


if __name__ == "__main__":
    main()
