"""Selection-curse report for a plain v2 arm (wote heads NC, DAC, EP, TTC, C; no field, no gate).

The deployed rule is WoTE's  0.1 log im + 0.5 log NC + 0.5 log DAC + log(5 TTC + 2 C + 5 EP)
(anchor_planner.weighted_reward, select_mode=weighted).  Linear scale only.

    control      the rule rebuilt from the dumped heads must reproduce plan_topk_index[:, 0], and the
                 table's PDMS of that pick must reproduce the published score
    loss split   best really-feasible (true NC = DAC = 1) candidate vs the pick: how many points, and in
                 the scenes where the pick is feasible but worse, is the im term or the head term the
                 larger of the two differences
    over-rating  head - true on the pick vs a uniformly random candidate of the deployed top-k
                 (k = 8, 64, 256; seed 0), on pairs where both are really feasible.  If the pick's
                 over-rating is not above the random one's, there is no selection curse to correct.
    im-weight    PRE-DECLARED grid {0.1 (deployed), 0.05, 0}

    python tools/plan_v2/curse_report_v2.py --dump <dump.npz> --table <candidate_pdms_*.npz>
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--table", required=True)
    a = ap.parse_args()
    z = np.load(a.dump)
    t = np.load(a.table)
    assert (t["tokens"] == z["tokens"]).all(), "dump / table token order differ"
    S = t["scores"].astype(np.float64)
    NC, DAC, EP, TTC, C, P = (S[..., i] for i in (0, 1, 3, 4, 5, 6))
    n = len(P); ix = np.arange(n)
    sim = z["sim_rewards"].astype(np.float64); im = z["im_rewards"].astype(np.float64)
    nc, dac, ep, ttc, c = (sim[:, i] for i in range(5))
    eps = 1e-6
    t_im = 0.1 * np.log(im + eps)
    t_head = 0.5 * np.log(nc + eps) + 0.5 * np.log(dac + eps) + np.log(5 * ttc + 2 * c + 5 * ep + eps)
    final = t_im + t_head
    p = final.argmax(1)
    print(f"대조: 재구성 선택 = 덤프 선택 {100 * (p == z['plan_topk_index'][:, 0]).mean():.2f}%   "
          f"덤프 선택의 PDMS {100 * P[ix, z['plan_topk_index'][:, 0]].mean():.2f}")
    print(f"배포 선택 PDMS {100 * P[ix, p].mean():.2f}  NC 실패 {100 * (NC[ix, p] < 1).mean():.2f}%  "
          f"DAC 실패 {100 * (DAC[ix, p] < 1).mean():.2f}%  EP {100 * EP[ix, p].mean():.1f}  TTC {100 * TTC[ix, p].mean():.1f}")

    ok = (NC == 1) & (DAC == 1)
    has = ok.any(1)
    b = np.where(ok, P, -1).argmax(1)
    best = np.where(has, P[ix, b], P[ix, p])
    print(f"\n손실 분해: 가능 후보 최고 {100 * best.mean():.2f}  (차 {100 * (best - P[ix, p]).mean():.2f}점)")
    bad = ~ok[ix, p] & has
    miss = has & ok[ix, p] & (P[ix, b] - P[ix, p] > 1e-6)
    print(f"  선택이 불가능 (NC 또는 DAC 0) {100 * bad.mean():5.1f}% 장면   {100 * (best - P[ix, p])[bad].sum() / n:5.2f}점")
    print(f"  선택은 가능, 더 좋은 가능 후보 {100 * miss.mean():5.1f}% 장면   {100 * (best - P[ix, p])[miss].sum() / n:5.2f}점")
    d_im, d_h = t_im[ix, p] - t_im[ix, b], t_head[ix, p] - t_head[ix, b]
    for nm, m in (("im 차이가 더 큼", miss & (np.abs(d_im) > np.abs(d_h))),
                  ("헤드 차이가 더 큼", miss & (np.abs(d_im) <= np.abs(d_h)))):
        print(f"    └ {nm:12s} {100 * m.sum() / max(miss.sum(), 1):5.1f}%   {100 * (best - P[ix, p])[m].sum() / n:5.2f}점")
    for nm, h, tr in (("EP", ep, EP), ("TTC", ttc, TTC), ("C", c, C)):
        dp, dt = h[ix, p] - h[ix, b], tr[ix, p] - tr[ix, b]
        print(f"    └ {nm:3s} 헤드가 순서를 뒤집음 {100 * (miss & (dp > 0) & (dt < 0)).sum() / max(miss.sum(), 1):5.1f}%")

    print("\n과대평가 (헤드 - 실제), 선택 vs 같은 상위 k 의 무작위 후보, 둘 다 실제 NC·DAC=1 인 장면")
    order = np.argsort(-final, axis=1)
    rng = np.random.default_rng(0)
    heads = (("NC", nc, NC), ("DAC", dac, DAC), ("EP", ep, EP), ("TTC", ttc, TTC), ("C", c, C),
             ("선호 (5T+2C+5E)/12", (5 * ttc + 2 * c + 5 * ep) / 12, (5 * TTC + 2 * C + 5 * EP) / 12))
    for k in (8, 64, 256):
        r = order[ix, rng.integers(0, k, n)]
        m = ok[ix, p] & ok[ix, r]
        row = "  ".join(f"{nm.split()[0]} {np.mean((h - tr)[ix, p][m]):+.3f}/{np.mean((h - tr)[ix, r][m]):+.3f}"
                        for nm, h, tr in heads)
        print(f"  k={k:<3d} ({100 * m.mean():4.1f}%)  선택/무작위  {row}")

    print("\nim 가중 (사전 선언 격자)")
    for w in (0.1, 0.05, 0.0):
        q = (w * np.log(im + eps) + t_head).argmax(1)
        print(f"  w_im={w:<5} PDMS {100 * P[ix, q].mean():.2f}  NC 실패 {100 * (NC[ix, q] < 1).mean():.2f}%  "
              f"DAC 실패 {100 * (DAC[ix, q] < 1).mean():.2f}%  EP {100 * EP[ix, q].mean():.1f}  TTC {100 * TTC[ix, q].mean():.1f}   "
              f"선택 과대 EP {np.mean(ep[ix, q] - EP[ix, q]):+.3f}  TTC {np.mean(ttc[ix, q] - TTC[ix, q]):+.3f}")


if __name__ == "__main__":
    main()
