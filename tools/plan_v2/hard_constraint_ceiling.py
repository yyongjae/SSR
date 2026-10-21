"""If a corridor removed the infeasible candidates outright, what would that be worth?

The corridor argument is not "perceive better" -- report/25 measured that only 2.1% of DAC failures
have a perception cause, and DAC 96.07 is already reached by imitation alone (picking the top
imitation anchor gives 86.95, adding the offset 95.16, the whole five-head scorer only +0.91). The
argument is that perception is never turned into a HARD constraint, so the planner is free to pick a
candidate its own map says is illegal.

This puts a ceiling on that idea without training anything. For every navtest scene the candidates
whose TRUE sub-score is 0 are struck out, and the deployed rule then picks among the survivors.
A perfect corridor cannot do better than this, because it cannot know more than the ground truth.

Controls matter here: striking out candidates also shrinks the choice set, which changes the pick
even when the filter is uninformative. So each arm is paired with a random filter that removes the
same number of candidates per scene.

Scale note: the candidate labels carry no two_frame_extended_comfort (it depends on the previous
frame's emitted trajectory), so numbers here are EPDMS WITHOUT EC. Official EPDMS is 87.80 and
official PDMS is 88.14; on this scale the deployed model is 88.81.

    python tools/plan_v2/hard_constraint_ceiling.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "readout"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: F401,E402
from clean_compare import epdms  # noqa: E402

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
NEG = -1e9


def main():
    z = np.load(ROOT / "data/planning_vb/navtest_refined_scores.npz", allow_pickle=True)
    cols = [str(c) for c in z["columns"]]
    ci = {c: k for k, c in enumerate(cols)}
    lab_at = {str(t): i for i, t in enumerate(z["tokens"])}
    raw = np.asarray(z["scores"])
    d = np.load(ROOT / "work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz")
    d_at = {str(t): i for i, t in enumerate(d["tokens"])}
    toks = sorted(set(lab_at) & set(d_at))
    R = raw[[lab_at[t] for t in toks]].astype(np.float32)
    fin = np.asarray(d["plan_final_rewards"][[d_at[t] for t in toks]], np.float64)
    del raw
    true = epdms(R, cols)[:, :256].astype(np.float64)       # 평가가 매기는 점수 (EC 제외 척도)
    N = len(toks)
    ar = np.arange(N)
    rng = np.random.default_rng(0)

    base = 100 * true[ar, fin.argmax(1)].mean()
    print(f"navtest {N} 장면 x 후보 256   (척도: EPDMS, EC 제외 — 공식 EPDMS 87.80 / PDMS 88.14)")
    print(f"배포 선택 {base:.2f},  오라클 {100 * true.max(1).mean():.2f}\n")
    print(f"{'하드 제약':34s} {'제거 후보':>9s} {'점수':>8s} {'이득':>7s} {'무작위 통제군':>13s}")

    filters = [("drivable_area_compliance",), ("no_at_fault_collisions",),
               ("drivable_area_compliance", "no_at_fault_collisions"),
               ("no_at_fault_collisions", "drivable_area_compliance",
                "driving_direction_compliance", "traffic_light_compliance")]
    names = ["DAC=0 후보 제거", "NC=0 후보 제거", "DAC·NC=0 제거", "곱셈항 4개 전부 =0 제거"]
    for name, keys in zip(names, filters):
        bad = np.zeros((N, 256), bool)
        for k in keys:
            bad |= R[:, :256, ci[k]] == 0
        keep = ~bad
        none = ~keep.any(1)                       # 전부 불가능한 장면은 필터를 적용하지 않는다
        sc = np.where(keep | none[:, None], fin, NEG)
        got = 100 * true[ar, sc.argmax(1)].mean()
        # 통제군: 장면마다 같은 개수를 무작위로 제거
        nrm = bad.sum(1)
        ctrl = np.zeros((N, 256), bool)
        for i in range(N):
            if nrm[i]:
                ctrl[i, rng.choice(256, min(nrm[i], 255), replace=False)] = True
        cs = np.where(~ctrl, fin, NEG)
        cgot = 100 * true[ar, cs.argmax(1)].mean()
        print(f"{name:34s} {bad.mean() * 256:9.1f} {got:8.2f} {got - base:+7.2f} "
              f"{cgot:9.2f} ({cgot - base:+.2f})")

    # 우리 모델 자신의 예측으로 같은 필터를 걸면? (corridor 가 우리 인지 수준일 때)
    sim = np.asarray(d["sim_rewards"][[d_at[t] for t in toks]], np.float64)   # [N, 5, 256]
    print(f"\n같은 필터를 우리 '예측' sub-score 로 걸었을 때 (임계값별)")
    for thr in (0.1, 0.3, 0.5):
        keep = (sim[:, 1] > thr)                                 # 예측 DAC
        none = ~keep.any(1)
        got = 100 * true[ar, np.where(keep | none[:, None], fin, NEG).argmax(1)].mean()
        print(f"  예측 DAC > {thr}: 평균 {keep.sum(1).mean():5.1f}개 생존, 점수 {got:6.2f} ({got - base:+.2f})")


if __name__ == "__main__":
    main()
