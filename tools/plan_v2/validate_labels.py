"""Check the anchor labels against NAVSIM's own evaluation before any of them are tabulated.

The model's chosen candidate was already scored by the official evaluation (run_pdm_score writes a
per-scene csv). The same candidate appears in our labels, so the two must agree scene by scene. They
agree exactly once two things are matched: ego_progress must be normalised against NAVSIM's PDM
planner (the scorer normalises by the largest progress among the proposals scored together, and our
labels score 256 at once), and two_frame_extended_comfort must be excluded on both sides -- it needs
the previous frame's CHOSEN trajectory, so it is not defined per candidate.

    python tools/plan_v2/validate_labels.py --labels <labels.npz> --csv <official.csv> \
        --dump work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz [--refined]
"""
import argparse

import numpy as np
import pandas as pd

from clean_compare import MULT, WEIGHTED, epdms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--csv", required=True, help="official per-scene eval of the SAME candidate")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tol", type=float, default=0.05, help="max allowed difference in EPDMS points")
    args = ap.parse_args()

    z = np.load(args.labels, allow_pickle=True)
    cols = [str(c) for c in z["columns"]]
    at = {str(t): i for i, t in enumerate(z["tokens"])}
    scores = z["scores"]          # materialise once: indexing an NpzFile decompresses the whole array
    dump = np.load(args.dump)
    pick = {str(t): int(dump["plan_topk_index"][i, 0]) for i, t in enumerate(dump["tokens"])}
    df = pd.read_csv(args.csv).set_index("token")
    mine, theirs = [], []
    for t, i in at.items():
        if t not in df.index or t not in pick:
            continue
        r = df.loc[t]
        m = float(np.prod([float(r[c]) for c in MULT]))
        theirs.append(m * sum(w * float(r[c]) for c, w in WEIGHTED.items()) / sum(WEIGHTED.values()))
        mine.append(float(epdms(scores[i][None], cols)[0, pick[t]]))
    d = 100 * (np.asarray(mine) - np.asarray(theirs))
    print(f"라벨 검증: {len(d)} 장면, 최대 차이 {np.abs(d).max():.4f}, 평균 {d.mean():+.4f} "
          f"(내 라벨 {100 * np.mean(mine):.2f}, 공식 EC 제외 {100 * np.mean(theirs):.2f}, "
          f"공식 원본 {100 * df.loc[list(at), 'score'].mean():.2f})")
    ok = np.abs(d).max() <= args.tol
    print("판정: 일치" if ok else f"판정: 불일치 — {args.tol} 점을 넘는다. 이 라벨로 표를 만들지 말 것")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
