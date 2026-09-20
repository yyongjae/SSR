"""Is WoTE's score head confidently wrong too?  (the v2 D2 report, for WoTE)

Reads the WoTE dump with per-candidate scores (dump_wote.py) and the EPDMS csv of its top-3
candidates (wote_diag.sh), and prints the same three numbers we measured on v2:
  1. EPDMS of rank 1..3 and of the oracle among them (how much the selection loses)
  2. how often a failure was scored >= 0.9 by the head that picked it (confidently wrong)
  3. how well the predicted NC / DAC separate the real pass/fail of the trajectory it output (AUC)

    python tools/plan_v2/wote_scorer_report.py
"""
import glob

import numpy as np
import pandas as pd

ROOT = "/home/external-user/kyungmin/SSR-v2/work_dirs/eval_epdms/_exp"
DUMP = "/home/external-user/kyungmin/SSR-v2/work_dirs/eval_epdms/diag_wote/dump.npz"
SIM = {"NC": (0, "no_at_fault_collisions"), "DAC": (1, "drivable_area_compliance"),
       "EP": (2, "ego_progress"), "TTC": (3, "time_to_collision_within_bound")}
M = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "ego_progress": "EP",
     "time_to_collision_within_bound": "TTC", "lane_keeping": "LK", "two_frame_extended_comfort": "EC",
     "score": "EPDMS"}


def load(name):
    f = sorted(glob.glob(f"{ROOT}/{name}/*/*.csv"))[-1]
    d = pd.read_csv(f)
    return d[d.token != "average_all_frames"].set_index("token")


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    z = np.load(DUMP)
    idx = {t: i for i, t in enumerate(z["tokens"])}
    v = {r: load(f"diag_wote_refined_r{r}") for r in (1, 2, 3)}
    common = sorted(set.intersection(*[set(x.index) for x in v.values()]) & set(idx))
    v = {r: x.loc[common] for r, x in v.items()}
    print(f"장면 {len(common)}\n1. 후보별 EPDMS")
    print("            " + " ".join(f"{k:>6s}" for k in M.values()))
    for r, x in v.items():
        print(f"  rank {r}     " + " ".join(f"{100 * x[c].astype(float).mean():6.2f}" for c in M))
    best = np.max(np.stack([v[r].score.astype(float).to_numpy() for r in v]), 0)
    print(f"  상위 3개 중 최선 EPDMS {100 * best.mean():.2f}  (1순위 {100 * v[1].score.astype(float).mean():.2f})")

    rows = np.array([idx[t] for t in common])
    sim = z["sim_rewards"][rows].astype(np.float32)          # [N, 5, K]
    top1 = z["topk_index"][rows, 0]
    print("\n2~3. 고른 후보의 예측 점수 vs 실제 (출력 궤적 기준)")
    for name, (j, col) in SIM.items():
        if col not in v[1]:
            continue
        p = sim[np.arange(len(rows)), j, top1]
        actual = v[1][col].astype(float).to_numpy()
        ok = actual == 1 if name != "EP" else actual >= 0.8
        fail = ~ok
        line = (f"  {name:4s} 예측 평균 {p.mean():.3f} | 실제 통과율 {ok.mean():.3f}"
                f" | 통과/실패 구분 AUC {auc(p[ok], p[fail]):.3f}")
        if name in ("NC", "DAC", "TTC"):
            line += f" | 실패인데 예측 >= 0.9 인 비율 {100 * (p[fail] >= 0.9).mean():.0f}%"
        print(line)


if __name__ == "__main__":
    main()
