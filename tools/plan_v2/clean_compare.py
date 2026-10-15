"""The clean comparison: v2 and the critic choose an anchor on navtest, scenes NEITHER has trained on.

The navtrain comparison is biased in v2's favour -- v2 trained on every navtrain scene, including the
critic's held-out split, so its 79.15 there is a training score while the critic's is a test score.
scripts/evaluation/label_anchors.py (in the navsim v2 worktree) labels the 256 anchors on navtest with
the official simulator and scorer, which removes the bias: navtest is held out for both.

v2's own scores come from the evaluation dump, so the sensor model is not re-run; the critics are fed
the GT entities and the entities v2 itself predicted for the same navtest scenes.

Reported per scorer: the true EPDMS of the anchor it picks, with NAVSIM's human_penalty_filter applied
exactly as run_pdm_score does, next to the oracle, a random pick, the human and NAVSIM's PDM planner.

    python tools/plan_v2/clean_compare.py --labels-npz data/planning_vb/navtest_anchor_scores.npz \
        --dump work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz \
        --cache data/critic_cache_navtest --pred-cache data/critic_cache_pred_navtest \
        --critic gt=work_dirs/entity_critic_big/critic.ckpt --critic pred=work_dirs/entity_critic_pred/critic.ckpt
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "readout"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: F401,E402

from navsim.agents.para_ssr.modules.entity_critic import EntityCritic  # noqa: E402

ENT = ("agents", "agent_fut", "agent_fut_mask", "agent_valid", "map_pts", "map_valid")
INT = ("map_labels",)
EXTRA = ("ego_status", "command")
MULT = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
        "traffic_light_compliance")
WEIGHTED = {"ego_progress": 5.0, "time_to_collision_within_bound": 5.0, "lane_keeping": 2.0,
            "history_comfort": 2.0}


def epdms(d, cols, human_filter=True):
    """[..., rows, C] sub-scores -> the EPDMS the official evaluation would give each row.

    Row -2 is the human trajectory (for the human_penalty_filter) and row -1 is NAVSIM's PDM planner.
    ego_progress needs the PDM planner: the scorer normalises progress by the largest progress among
    the proposals scored TOGETHER, and the official evaluation scores [pdm_planner, trajectory], so
    the value stored from a 258-proposal batch has the wrong denominator. Redo it from the raw metres.
    """
    i = {c: k for k, c in enumerate(cols)}
    x = d.astype(np.float32).copy()
    if "progress_raw_m" in i:
        raw = x[..., i["progress_raw_m"]]
        mult = np.ones(x.shape[:-1], np.float32)
        for c in MULT:
            mult = mult * x[..., i[c]]
        masked_pdm = (raw * mult)[..., -1:]                      # NAVSIM's PDM planner, as in the eval
        norm = np.maximum(masked_pdm, raw * mult)
        x[..., i["ego_progress"]] = np.where(norm > 5.0, np.clip(raw / np.maximum(norm, 1e-6), 0.0, 1.0), 1.0)
    if human_filter:
        for c in MULT + tuple(WEIGHTED):
            j = i[c]
            x[..., j] = np.where(x[..., -2, j][..., None] == 0.0, 1.0, x[..., j])
    mult = np.ones(x.shape[:-1], np.float32)
    for c in MULT:
        mult = mult * x[..., i[c]]
    num = sum(w * x[..., i[c]] for c, w in WEIGHTED.items())
    return mult * num / sum(WEIGHTED.values())


def pdms_of(pred):
    """5 predicted sub-scores (SIM_KEYS order) -> the PDMS a scorer ranks candidates by."""
    return pred[:, 0] * pred[:, 1] * (5 * pred[:, 3] + 2 * pred[:, 4] + 5 * pred[:, 2]) / 12


def load_cache(path, keys):
    data, toks = {}, []
    for f in sorted(glob.glob(f"{path}/shard_*.npz")):
        z = np.load(f)
        if len(z["tokens"]) == 0:
            continue
        toks += [str(t) for t in z["tokens"]]
        for k in keys:
            data.setdefault(k, []).append(z[k])
    return {t: i for i, t in enumerate(toks)}, {k: np.concatenate(v) for k, v in data.items()}


@torch.no_grad()
def critic_scores(model, at, data, tokens, device, batch=64):
    """[N, 5, 256] predicted sub-scores for the given tokens."""
    idx = np.array([at[t] for t in tokens])
    out = []
    for s in range(0, len(idx), batch):
        j = idx[s: s + batch]
        b = {k: torch.as_tensor(np.asarray(data[k][j])).to(device) for k in data}
        b["map_labels"] = b["map_labels"].long()
        out.append(model(b).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-npz", required=True)
    ap.add_argument("--dump", required=True, help="v2 evaluation dump (sim_rewards, plan_final_rewards)")
    ap.add_argument("--cache", required=True, help="GT token cache for the same split")
    ap.add_argument("--pred-cache", default="", help="the entities v2 itself predicts on the same split")
    ap.add_argument("--critic", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--anchors", default="data/planning_vb/trajectory_anchors_256.npy")
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out", default="data/perception_use/clean_compare.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(args.labels_npz, allow_pickle=True)
    cols = [str(c) for c in z["columns"]]
    label_at = {str(t): i for i, t in enumerate(z["tokens"])}
    dump = np.load(args.dump)
    dump_at = {str(t): i for i, t in enumerate(dump["tokens"])}
    gt_at, gt = load_cache(args.cache, ENT + INT + EXTRA)
    pred_at, pred = (load_cache(args.pred_cache, ENT + INT) if args.pred_cache else ({}, {}))

    tokens = sorted(set(label_at) & set(gt_at) & set(dump_at) & (set(pred_at) if pred_at else set(gt_at)))
    print(f"장면 {len(tokens)}  (라벨 {len(label_at)}, GT 캐시 {len(gt_at)}, 예측 인지 {len(pred_at)})", flush=True)
    if pred_at:                     # predicted entities; ego status and command come from the GT cache
        pred_sub = {k: pred[k][[pred_at[t] for t in tokens]] for k in ENT + INT}
        pred_sub.update({k: gt[k][[gt_at[t] for t in tokens]] for k in EXTRA})
        pred_at_sub = {t: i for i, t in enumerate(tokens)}

    # materialise each array once: indexing an NpzFile decompresses the WHOLE array every time,
    # so a per-token comprehension reads ~190 MB per scene (141 GB of churn over navtest)
    scores, sim, fin = z["scores"], dump["sim_rewards"], dump["plan_final_rewards"]
    li = [label_at[t] for t in tokens]
    di = [dump_at[t] for t in tokens]
    res = {"token": np.asarray(tokens), "label": scores[li], "v2": sim[di], "v2_final": fin[di]}
    keys = ["v2", "v2_final"]
    for spec in args.critic:
        name, path = spec.split("=", 1)
        m = EntityCritic(args.anchors, embed_dims=args.embed, num_layers=args.layers).to(device).eval()
        m.load_state_dict(torch.load(path, map_location=device)["model"])
        res[f"{name}@gt"] = critic_scores(m, gt_at, gt, tokens, device)
        keys.append(f"{name}@gt")
        if pred_at:
            res[f"{name}@pred"] = critic_scores(m, pred_at_sub, pred_sub, tokens, device)
            keys.append(f"{name}@pred")
        print(f"  critic {name}: {path}", flush=True)
    res["columns"] = np.asarray(cols)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **res)
    report(res, cols, keys)


def report(res, cols, keys):
    true = epdms(res["label"], cols)                             # [N, 258]
    anchors, human, pdm = true[:, :-2], true[:, -2], true[:, -1]
    rng = np.random.default_rng(0)
    idx = np.arange(len(anchors))
    print(f"\n고른 anchor 의 실제 EPDMS — navtest {len(idx)} 장면 (v2 도 critic 도 학습에 쓰지 않음)")
    # v2_final is the rule the car actually runs (WoTE's weighted log reward): it reproduces the
    # deployed choice in 99.5% of scenes, while ranking by the five sub-scores agrees only 14.8%.
    name = {"v2_final": "v2 (배포 선택식)", "v2": "v2 (sub-score 조합)"}
    rows = {"무작위 선택": anchors[idx, rng.integers(0, anchors.shape[1], len(idx))]}
    for k in keys:
        s = res[k].astype(np.float32)
        rows[name.get(k, k)] = anchors[idx, (s.argmax(1) if s.ndim == 2 else pdms_of(s).argmax(1))]
    rows["정답을 아는 선택 (상한)"] = anchors.max(1)
    for k, v in rows.items():
        print(f"   {k:22s} {100 * v.mean():6.2f}")
    print(f"   {'사람 궤적':22s} {100 * human.mean():6.2f}")
    print(f"   {'NAVSIM PDM planner':22s} {100 * pdm.mean():6.2f}")


if __name__ == "__main__":
    main()
