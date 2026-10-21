"""Can an anchor's simulator label be MOVED onto the trajectory the model actually emits?

The scoring heads are supervised on the 256 anchors (``plan_score_targets.py`` feeds WoTE's
per-anchor PDM scores), but the model emits anchor + offset and selects among those. Feeding the
heads their own target perfectly is worth 90.31 EPDMS; feeding them the label of what they emit is
worth 94.24. The 3.93 points in between are pure label/output mismatch.

iPad (2505.15111) and CLOVER (2605.15120) close that gap by simulating the emitted trajectories
online during training -- CLOVER reports 96 Ray threads on 4xA100 for 4-5 days, and GTRS declines
the same thing as "a deliberate design choice" over cost. So the question that decides whether a
cheap alternative exists is:

    given only the precomputed labels of the 256 anchors and the offsets, how well can the label of
    anchor_k + offset_k be reconstructed?

navtest has both label sets already (``navtest_anchor_scores.npz`` / ``navtest_refined_scores.npz``),
so this is answerable with no simulator, no GPU and no training. Schemes tested:

    own        the anchor's own label                       (= what we supervise on today)
    nearest    the label of the anchor closest to the REFINED trajectory
    knn        softmax(-d/T) weighted average of anchor labels around the refined trajectory
    oracle     the true refined label                        (upper bound)

Each is judged two ways: how close the transferred label is to the truth, and -- what actually
matters -- the true EPDMS of the candidate the deployed rule picks when fed that label.

    python tools/plan_v2/label_transfer_probe.py [--chunk 512] [--temps 0.5 1.0 2.0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "readout"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: F401,E402
from clean_compare import MULT, WEIGHTED, epdms  # noqa: E402

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
W = [0.1, 0.5, 0.5, 1.0]
KEYS = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
        "time_to_collision_within_bound", "history_comfort"]
SHORT = ["NC", "DAC", "EP", "TTC", "HC"]


def load_labels(path):
    z = np.load(path, allow_pickle=True)
    return ({str(t): i for i, t in enumerate(z["tokens"])}, np.asarray(z["scores"]),
            [str(c) for c in z["columns"]])


def official(raw, cols):
    """[N, 258, C] -> per-candidate sub-scores with the official EP and the human filter."""
    ci = {c: k for k, c in enumerate(cols)}
    x = raw.astype(np.float32).copy()
    m_raw = x[..., ci["progress_raw_m"]]
    mult = np.ones(x.shape[:-1], np.float32)
    for c in MULT:
        mult = mult * x[..., ci[c]]
    norm = np.maximum((m_raw * mult)[..., -1:], m_raw * mult)
    x[..., ci["ego_progress"]] = np.where(norm > 5.0, np.clip(m_raw / np.maximum(norm, 1e-6), 0, 1), 1.0)
    for c in MULT + tuple(WEIGHTED):
        j = ci[c]
        x[..., j] = np.where(x[..., -2, j][..., None] == 0.0, 1.0, x[..., j])
    return np.stack([x[:, :256, ci[c]] for c in KEYS], axis=-1).astype(np.float32)   # [N, 256, 5]


def rule(S, im):
    """S [.., 256, 5] sub-scores in KEYS order -> the deployed weighted reward."""
    e = 1e-6
    nc, dac, ep, ttc, hc = (S[..., i] for i in range(5))
    return (W[0] * np.log(im + e) + W[1] * np.log(nc + e) + W[2] * np.log(dac + e)
            + W[3] * np.log(5 * ttc + 2 * hc + 5 * ep + e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--temps", type=float, nargs="*", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--anchors", default=str(ROOT / "data/planning_vb/trajectory_anchors_256.npy"))
    args = ap.parse_args()

    a_at, a_raw_all, cols = load_labels(ROOT / "data/planning_vb/navtest_anchor_scores.npz")
    r_at, r_raw_all, _ = load_labels(ROOT / "data/planning_vb/navtest_refined_scores.npz")
    d = np.load(ROOT / "work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz")
    d_at = {str(t): i for i, t in enumerate(d["tokens"])}
    toks = sorted(set(a_at) & set(r_at) & set(d_at))
    ai = np.array([a_at[t] for t in toks]); ri = np.array([r_at[t] for t in toks])
    di = np.array([d_at[t] for t in toks])

    anchors = np.load(args.anchors).astype(np.float32)                    # [256, 8, 3]
    flat_anchor = anchors.reshape(256, -1)                                # [256, 24]
    A = official(a_raw_all[ai], cols)                                     # [N, 256, 5] anchor 진실
    R = official(r_raw_all[ri], cols)                                     # [N, 256, 5] refined 진실
    true = epdms(r_raw_all[ri], cols)[:, :256].astype(np.float64)         # 평가가 매기는 점수
    im = np.asarray(d["im_rewards"][di], np.float64)
    offs_all = d["trajectory_offset"]
    del a_raw_all, r_raw_all

    N = len(toks)
    names = ["own", "nearest"] + [f"knn T={t}" for t in args.temps]
    picks = {n: np.empty(N, np.int64) for n in names}
    err = {n: np.zeros(5, np.float64) for n in names}
    exact = {n: np.zeros(5, np.float64) for n in names}
    moved = 0.0

    for s in range(0, N, args.chunk):
        e = min(s + args.chunk, N)
        off = np.asarray(offs_all[di[s:e]], np.float32)                   # [B, 256, 8, 3]
        refined = (anchors[None] + off).reshape(len(off), 256, -1)        # [B, 256, 24]
        # 각 refined 후보에서 모든 앵커까지의 거리
        dist = np.linalg.norm(refined[:, :, None, :] - flat_anchor[None, None], axis=-1)  # [B,256,256]
        nn = dist.argmin(-1)                                              # [B, 256]
        moved += float((nn != np.arange(256)[None]).sum())
        Ab, Rb, imb = A[s:e], R[s:e], im[s:e]
        cand = {"own": Ab, "nearest": np.take_along_axis(Ab, nn[..., None], axis=1)}
        for t in args.temps:
            w = np.exp(-(dist - dist.min(-1, keepdims=True)) / t)
            w /= w.sum(-1, keepdims=True)
            cand[f"knn T={t}"] = np.einsum("bkj,bjm->bkm", w.astype(np.float32), Ab)
        for n, S in cand.items():
            picks[n][s:e] = rule(S.astype(np.float64), imb).argmax(-1)
            err[n] += np.abs(S - Rb).sum(axis=(0, 1))
            exact[n] += (np.abs(S - Rb) < 1e-6).sum(axis=(0, 1))
        if (e // args.chunk) % 4 == 0:
            print(f"  {e}/{N}", flush=True)

    ar = np.arange(N)
    denom = N * 256
    base_own = 100 * true[ar, picks["own"]].mean()
    oracle_lab = 100 * true[ar, rule(R.astype(np.float64), im).argmax(-1)].mean()
    print(f"\nnavtest {N} 장면 x 256 후보.  offset 때문에 가장 가까운 앵커가 바뀌는 후보 "
          f"{100 * moved / denom:.1f}%\n")
    print(f"{'라벨 이전 방식':14s} {'고른 후보의 실제 EPDMS':>22s} {'회복률':>8s}   "
          + " ".join(f"{k:>13s}" for k in SHORT))
    for n in names:
        got = 100 * true[ar, picks[n]].mean()
        rec = 100 * (got - base_own) / max(oracle_lab - base_own, 1e-9)
        cells = [f"{err[n][i] / denom:.3f}/{100 * exact[n][i] / denom:.0f}%" for i in range(5)]
        print(f"{n:14s} {got:22.2f} {rec:7.1f}%   " + " ".join(f"{c:>13s}" for c in cells))
    print(f"{'oracle (refined 라벨)':14s} {oracle_lab:22.2f} {100.0:7.1f}%")
    print("\n(지표 칸은 '평균 |오차| / 정확히 일치한 비율'. 이전 방식이 refined 진실을 얼마나 맞히는가)")
    print(f"현행 감독 = own = {base_own:.2f},  라벨을 출력에 맞추면 {oracle_lab:.2f}")


if __name__ == "__main__":
    main()
