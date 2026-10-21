"""If we shortlist by ANCHOR score, do the good REFINED candidates survive the cut?

SparseDriveV2 keeps 262k vocabulary elements affordable by scoring only a top-k shortlist
(1024x256 -> 128x64 -> 20x10). That works there because the vocabulary element IS the output. We add
a per-anchor offset, so the shortlist is drawn on one ranking and paid out on another.

Measured here, on the 12,146 navtest scenes where both label sets exist:
  * how much the two rankings agree at all (within-scene Spearman, and agreement at the top)
  * recall@k -- is the best refined candidate inside the top-k drawn by anchor score?
  * the oracle ceiling INSIDE the shortlist, which is what a coarse-to-fine design could ever reach

Three shortlist rules: the true anchor EPDMS (an upper bound on any anchor-side scorer), the model's
own deployed score, and the model's predicted sub-scores fed the anchor way.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "readout"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: F401,E402
from clean_compare import epdms  # noqa: E402

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
KS = [1, 2, 4, 8, 16, 32, 64, 128]


def load(p):
    z = np.load(p, allow_pickle=True)
    return ({str(t): i for i, t in enumerate(z["tokens"])}, np.asarray(z["scores"]),
            [str(c) for c in z["columns"]])


a_at, a_raw, cols = load(ROOT / "data/planning_vb/navtest_anchor_scores.npz")
r_at, r_raw, _ = load(ROOT / "data/planning_vb/navtest_refined_scores.npz")
d = np.load(ROOT / "work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz")
d_at = {str(t): i for i, t in enumerate(d["tokens"])}
toks = sorted(set(a_at) & set(r_at) & set(d_at))
di = np.array([d_at[t] for t in toks])
Ea = epdms(a_raw[[a_at[t] for t in toks]], cols)[:, :256].astype(np.float64)   # 앵커만 냈을 때 진짜 점수
Er = epdms(r_raw[[r_at[t] for t in toks]], cols)[:, :256].astype(np.float64)   # 앵커+offset 진짜 점수
fin = np.asarray(d["plan_final_rewards"][di], np.float64)                      # 모델 배포 점수
del a_raw, r_raw
N = len(toks)
ar = np.arange(N)

print(f"navtest {N} 장면 x 후보 256\n")
sub = np.arange(0, N, 8)
rs = np.array([spearmanr(Ea[i], Er[i]).statistic for i in sub], float)
rs = rs[np.isfinite(rs)]
print(f"앵커 점수 순위 vs 앵커+offset 점수 순위, 장면 내 Spearman  평균 {np.nanmean(rs):+.3f}  "
      f"중앙값 {np.nanmedian(rs):+.3f}")
print(f"  r < 0.3 인 장면 {100 * (rs < 0.3).mean():.1f}%,  r < 0 인 장면 {100 * (rs < 0).mean():.1f}%")

best_r = Er.argmax(1)
print(f"\n앵커 1위가 offset 후에도 1위인 장면 : {100 * (Ea.argmax(1) == best_r).mean():.1f}%")
print(f"앵커 기준 최선 후보의, offset 후 순위 중앙값 : "
      f"{np.median((Er > Er[ar, Ea.argmax(1)][:, None]).sum(1) + 1):.0f}")

rules = {"앵커 진실 점수": Ea, "모델 배포 점수": fin}
print(f"\n{'shortlist 규칙':16s} {'k':>5s} {'최선 후보 포함률':>15s} {'shortlist 내 상한':>18s}")
glob = 100 * Er.max(1).mean()
for name, key in rules.items():
    order = np.argsort(-key, axis=1)
    for k in KS:
        top = order[:, :k]
        hit = 100 * (top == best_r[:, None]).any(1).mean()
        ceil = 100 * np.take_along_axis(Er, top, axis=1).max(1).mean()
        print(f"{name if k == KS[0] else '':16s} {k:5d} {hit:14.1f}% {ceil:17.2f}")
    print()
print(f"전체 256 상한 {glob:.2f},  배포 선택 {100 * Er[ar, fin.argmax(1)].mean():.2f}")
