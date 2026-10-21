"""The anchor-oracle claim, on the OFFICIAL scale.

The number that says "improving anchor selection buys nothing" has only ever existed on an
internal scale (EPDMS with two-frame extended comfort dropped), which must not appear in a report.
This rebuilds the same comparison as trajectory files, so the official NAVSIM v2.2 scorer -- EC
included -- produces every number.

Four selections over v2_r34's own 256 candidates:

    deployed        argmax of plan_final_rewards, anchor + offset   (must reproduce 87.80)
    oracle-anchor   argmax of the ANCHOR's own PDM score, then anchor + offset
    oracle-anchor0  the same anchor driven VERBATIM, no offset
    oracle-refined  argmax of the REFINED trajectory's PDM score    (the true selection ceiling)

`deployed` is the control: if it does not land on 87.80 the pipeline is wrong and nothing else
here can be believed.

    python tools/plan_v2/oracle_official.py            # writes four npz, then score each
    bash data/logs/epdms_from_npz.sh <name> <npz>
"""
import numpy as np
from pathlib import Path

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
OUT = ROOT / "work_dirs/eval_epdms/oracle_official"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / "work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz")
    toks = [str(t) for t in d["tokens"]]
    off = d["trajectory_offset"].astype(np.float32)            # [N, 256, 8, 3]
    fin = d["plan_final_rewards"].astype(np.float32)           # [N, 256]
    anchors = np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy").astype(np.float32)

    def load(name):
        z = np.load(ROOT / f"data/planning_vb/{name}.npz", allow_pickle=True)
        ci = [str(c) for c in z["columns"]].index("pdm_score")
        at = {str(t): i for i, t in enumerate(z["tokens"])}
        return np.asarray(z["scores"])[:, :256, ci], at

    a_score, a_at = load("navtest_anchor_scores")              # the anchors' own scores
    r_score, r_at = load("navtest_refined_scores")             # anchor + offset scores

    keep = [i for i, t in enumerate(toks) if t in a_at and t in r_at]
    print(f"{len(keep)} / {len(toks)} 장면에 앵커·정제 라벨이 모두 있음")
    ai = np.array([a_at[toks[i]] for i in keep])
    ri = np.array([r_at[toks[i]] for i in keep])
    k = np.array(keep)

    pick = {
        "deployed":       fin[k].argmax(1),
        "oracle_anchor":  a_score[ai].argmax(1),
        "oracle_refined": r_score[ri].argmax(1),
    }
    rows = np.arange(len(k))
    out = {}
    for name, p in pick.items():
        out[name] = (anchors[p] + off[k][rows, p]).astype(np.float32)
    # the same oracle anchor, driven verbatim -- isolates what the offset does to a perfect pick
    out["oracle_anchor0"] = anchors[pick["oracle_anchor"]].astype(np.float32)

    names = np.array([toks[i] for i in k])
    for name, traj in out.items():
        f = OUT / f"{name}.npz"
        np.savez_compressed(f, tokens=names, trajectory=traj,
                            ego_fut_preds=np.zeros((len(traj), 4, 8, 3), np.float32),
                            command=np.zeros((len(traj), 4), np.float32),
                            missing=np.zeros((0,), np.float64),
                            checkpoint=np.array("oracle_official"))
        print(f"  {name:16s} -> {f}")

    print("\n채점 명령:")
    for name in out:
        print(f"  bash data/logs/epdms_from_npz.sh oracle_{name} {OUT}/{name}.npz")


if __name__ == "__main__":
    main()
