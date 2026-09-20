"""Choosing vs using: the score head ranks bare anchors, but the car drives anchor + offset.

Both are labelled with the official scorer on all 12,146 navtest scenes (label_anchors.py, once with
the anchors and once with the model's own refined candidates), so the two can be put side by side:

  choosing   pick an anchor, keep it          -> what a better scorer alone could win
  using      pick an anchor, drive it refined -> what the offset head does to that choice
  mismatch   the anchor the score head likes is not the refined candidate that actually scores best

    python tools/plan_v2/use_report.py --anchor-labels <anchors.npz> --refined-labels <refined.npz> \
        --dump work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz
"""
import argparse

import numpy as np

from clean_compare import epdms, pdms_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-labels", required=True)
    ap.add_argument("--refined-labels", required=True)
    ap.add_argument("--dump", required=True)
    args = ap.parse_args()

    a, r = np.load(args.anchor_labels, allow_pickle=True), np.load(args.refined_labels, allow_pickle=True)
    cols = [str(c) for c in a["columns"]]
    at_a = {str(t): i for i, t in enumerate(a["tokens"])}
    at_r = {str(t): i for i, t in enumerate(r["tokens"])}
    dump = np.load(args.dump)
    at_d = {str(t): i for i, t in enumerate(dump["tokens"])}
    tok = sorted(set(at_a) & set(at_r) & set(at_d))
    sa, sr = a["scores"], r["scores"]        # materialise once (NpzFile indexing decompresses it all)
    dsim, dfin = dump["sim_rewards"], dump["plan_final_rewards"]
    A = epdms(sa[[at_a[t] for t in tok]], cols)[:, :-2]                         # [N, 256] anchor truth
    R = epdms(sr[[at_r[t] for t in tok]], cols)[:, :-2]                         # [N, 256] refined truth
    sim = dsim[[at_d[t] for t in tok]].astype(np.float32)
    fin = dfin[[at_d[t] for t in tok]].astype(np.float32)
    n = np.arange(len(tok))
    pick = fin.argmax(1)                                                        # the deployed choice
    print(f"navtest {len(tok)} 장면\n")

    print("1. 고르기와 쓰기의 분해 (실제 EPDMS)")
    rows = [("v2 가 고른 앵커, 그대로", A[n, pick]),
            ("v2 가 고른 앵커, offset 적용 = 실제 출력", R[n, pick]),
            ("앵커 정답으로 고르고 그대로", A.max(1)),
            ("앵커 정답으로 고르고 offset 적용", R[n, A.argmax(1)]),
            ("refined 정답으로 고름 (offset 포함 상한)", R.max(1)),
            ("v2 점수로 고르되 점수를 잘 맞춘 경우(=앵커 상한) 대비 손실", A.max(1) - A[n, pick])]
    for k, v in rows:
        print(f"   {k:44s} {100 * v.mean():6.2f}")

    print("\n2. offset 이 후보를 얼마나 바꾸나 (전 후보 평균)")
    d = R - A
    print(f"   전체 256 후보 평균 변화      {100 * d.mean():+6.2f}")
    print(f"   v2 가 고른 후보              {100 * d[n, pick].mean():+6.2f}")
    print(f"   앵커 기준 상위 10 후보       {100 * np.take_along_axis(d, np.argsort(-A, 1)[:, :10], 1).mean():+6.2f}")
    print(f"   후보가 나빠진 비율           {100 * (d < -0.01).mean():5.1f}%   좋아진 비율 {100 * (d > 0.01).mean():5.1f}%")

    print("\n3. 채점 대상과 주행 대상의 불일치")
    best_a, best_r = A.argmax(1), R.argmax(1)
    print(f"   앵커 기준 최선 == refined 기준 최선      {100 * (best_a == best_r).mean():5.1f}% 의 장면")
    print(f"   앵커 최선을 골라 refine 하면 refined 상한의 {100 * R[n, best_a].mean() / R.max(1).mean():5.1f}%")
    top = np.argsort(-pdms_of(sim), 1)[:, :3]
    print(f"   v2 상위 3 중 refined 기준 최선          {100 * np.take_along_axis(R, top, 1).max(1).mean():6.2f}"
          f"   (실제 {100 * R[n, pick].mean():.2f})")
    print(f"   v2 상위 3 중 anchor 기준 최선           {100 * np.take_along_axis(A, top, 1).max(1).mean():6.2f}")

    print("\n4. 점수 head 가 무엇을 맞히고 있나 (장면별 순위 상관)")
    def spearman(x, y):
        rx = np.argsort(np.argsort(x, 1), 1).astype(np.float32)
        ry = np.argsort(np.argsort(y, 1), 1).astype(np.float32)
        rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
        return float(np.mean((rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1) + 1e-9)))
    p = pdms_of(sim)
    print(f"   v2 점수 ~ 앵커 실제        {spearman(p, A):.3f}")
    print(f"   v2 점수 ~ refined 실제     {spearman(p, R):.3f}")
    print(f"   앵커 실제 ~ refined 실제   {spearman(A, R):.3f}")


if __name__ == "__main__":
    main()
