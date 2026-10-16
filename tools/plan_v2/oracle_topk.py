"""Selection ceiling on the OFFICIAL scale, as a function of how far the pick may move.

An oracle that re-picks freely at every frame destroys two-frame extended comfort (EC 83.3 -> 25.3
measured), because nothing ties consecutive frames together.  That collapse is a property of the
oracle, not of the trajectories, and it makes the free oracle's total a lower bound rather than a
ceiling.  Restricting the oracle to the top-k of the deployed ranking bounds how far it can jump,
so EC survives, and k traces the curve from "re-rank a shortlist" to "free choice".
"""
import numpy as np
from pathlib import Path

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
OUT = ROOT / "work_dirs/eval_epdms/oracle_topk"
KS = [1, 3, 5, 10, 32]

OUT.mkdir(parents=True, exist_ok=True)
d = np.load(ROOT / "work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz")
toks = [str(t) for t in d["tokens"]]
off = d["trajectory_offset"].astype(np.float32)
fin = d["plan_final_rewards"].astype(np.float32)
anchors = np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy").astype(np.float32)

z = np.load(ROOT / "data/planning_vb/navtest_refined_scores.npz", allow_pickle=True)
ci = [str(c) for c in z["columns"]].index("pdm_score")
at = {str(t): i for i, t in enumerate(z["tokens"])}
keep = [i for i, t in enumerate(toks) if t in at]
k_idx = np.array(keep)
truth = np.asarray(z["scores"])[[at[toks[i]] for i in keep]][:, :256, ci]
rank = np.argsort(-fin[k_idx], axis=1)                     # deployed order
rows = np.arange(len(k_idx))
names = np.array([toks[i] for i in keep])
print(f"{len(keep)} 장면")

for K in KS:
    cand = rank[:, :K]                                      # [N, K] candidate ids
    best = cand[rows, truth[rows[:, None], cand].argmax(1)]  # best-of-top-K by true score
    traj = (anchors[best] + off[k_idx][rows, best]).astype(np.float32)
    f = OUT / f"top{K}.npz"
    np.savez_compressed(f, tokens=names, trajectory=traj,
                        ego_fut_preds=np.zeros((len(traj), 4, 8, 3), np.float32),
                        command=np.zeros((len(traj), 4), np.float32),
                        missing=np.zeros((0,), np.float64), checkpoint=np.array(f"top{K}"))
    print(f"  top{K:<3d} -> {f}")
