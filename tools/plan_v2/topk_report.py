"""D2 report: does the score head rank what is actually output?

Reads the EPDMS csv of each top-k variant (topk_variants.py + epdms_from_npz.sh, run names
<prefix>_{anchor,refined}_r{k}) and the dump's predicted sub-scores, and prints
  1. EPDMS and sub-scores per variant
  2. how often the offset flips a metric of the chosen candidate (anchor_r1 vs refined_r1)
  3. the oracle among the top-k refined / anchors (best EPDMS per scene)
  4. score-head calibration: predicted DAC / NC of the chosen anchor vs the real DAC / NC of
     the anchor (what it was trained on) and of the output (what it is used for)

    python tools/plan_v2/topk_report.py <dump.npz> <prefix> [k=3]
"""
import glob
import sys

import numpy as np
import pandas as pd

ROOT = "/home/external-user/kyungmin/SSR-v2/work_dirs/eval_epdms/_exp"
M = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "driving_direction_compliance": "DDC",
     "ego_progress": "EP", "time_to_collision_within_bound": "TTC", "lane_keeping": "LK",
     "two_frame_extended_comfort": "EC", "score": "EPDMS"}


def load(name):
    f = sorted(glob.glob(f"{ROOT}/{name}/*/*.csv"))[-1]
    d = pd.read_csv(f)
    return d[d.token != "average_all_frames"].set_index("token")


def auc(pos_score, neg_score):
    """P(score of a passing scene > score of a failing scene)."""
    if len(pos_score) == 0 or len(neg_score) == 0:
        return float("nan")
    s = np.concatenate([pos_score, neg_score]); r = s.argsort().argsort() + 1
    return (r[: len(pos_score)].sum() - len(pos_score) * (len(pos_score) + 1) / 2) / (len(pos_score) * len(neg_score))


def main():
    dump, prefix = np.load(sys.argv[1]), sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    v = {f"{kind}_r{r}": load(f"{prefix}_{kind}_r{r}") for kind in ("refined", "anchor") for r in range(1, k + 1)}
    common = sorted(set.intersection(*[set(x.index) for x in v.values()]))
    v = {n: x.loc[common] for n, x in v.items()}
    print(f"장면 {len(common)}\n")
    print("1. 후보별 EPDMS")
    print("   " + " ".join(f"{c:>6s}" for c in M.values()))
    for n, x in v.items():
        print(f"{n:11s} " + " ".join(f"{100 * x[c].astype(float).mean():6.2f}" for c in M))

    a, r = v["anchor_r1"], v["refined_r1"]
    print("\n2. 선택된 후보에서 offset 이 바꾼 것 (anchor_r1 -> refined_r1, 장면 수)")
    for c in ("drivable_area_compliance", "no_at_fault_collisions", "time_to_collision_within_bound", "lane_keeping"):
        fa, fr = a[c].astype(float) < 1, r[c].astype(float) < 1
        print(f"   {M[c]:4s} 통과->실패 {int((~fa & fr).sum()):5d}   실패->통과 {int((fa & ~fr).sum()):5d}")
    d_ep = (r.ego_progress - a.ego_progress).astype(float)
    print(f"   EP 평균 변화 {100 * d_ep.mean():+.2f}")

    print(f"\n3. 상위 {k}개 중 최선 (정답을 아는 선택)")
    for kind in ("refined", "anchor"):
        best = np.max(np.stack([v[f"{kind}_r{i}"].score.astype(float).to_numpy() for i in range(1, k + 1)]), 0)
        print(f"   {kind:8s} 최선 EPDMS {100 * best.mean():.2f}  (1순위 {100 * v[f'{kind}_r1'].score.astype(float).mean():.2f})")

    idx = {t: i for i, t in enumerate(dump["tokens"])}
    rows = np.array([idx[t] for t in common])
    top1 = dump["plan_topk_index"][rows, 0]
    sim = dump["sim_rewards"][rows].astype(np.float32)          # [N, 5(NC, DAC, EP, TTC, C), K]
    print("\n4. 점수 head 의 예측 vs 실제 (선택된 후보)")
    for j, c in ((1, "drivable_area_compliance"), (0, "no_at_fault_collisions")):
        p = sim[np.arange(len(rows)), j, top1]
        for kind, x in (("anchor", a), ("output", r)):
            ok = x[c].astype(float).to_numpy() == 1
            print(f"   {M[c]:4s} 예측 평균 {p.mean():.3f} | 실제({kind}) 통과율 {ok.mean():.3f} | "
                  f"통과/실패 구분 AUC {auc(p[ok], p[~ok]):.3f}")


if __name__ == "__main__":
    main()
