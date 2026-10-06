"""Seeing or reading?  Split the score-head gap between perception error and how the scores use it.

On the SAME held-out navtrain scenes (the critic's validation split, which has WoTE's PDM labels
for all 256 anchors) three scorers are compared:

  v2            the sensor model's own score head (sees images / BEV)
  critic(pred)  the entity-only critic fed the v2 model's OWN detections and map vectors
  critic(GT)    the entity-only critic fed the GT entities

  critic(GT) - critic(pred)  = what imperfect perception costs
  critic(pred) - v2          = what the sensor model loses by not using the perception it has

    python tools/plan_v2/critic_vs_student.py --critic work_dirs/entity_critic/critic.ckpt \
        --v2-ckpt <v2 ckpt> --cache data/critic_cache --labels data/planning_vb/pdm_score_256 \
        [--bb 34] [--limit 0] <agent.config overrides...>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "readout"))
import _env  # noqa: F401,E402
from _env import scene_filter, split_dirs  # noqa: E402
from cache_student_bev import FeatureData, build_agent, collate  # noqa: E402

from navsim.agents.para_ssr.modules.anchor_planner import SIM_KEYS  # noqa: E402
from navsim.agents.para_ssr.modules.entity_critic import EntityCritic  # noqa: E402
from train_critic import Cache, auc  # noqa: E402

MAX_AGENTS, MAX_MAP = 32, 48


def predicted_entities(det_out, map_out, det_thr, map_thr, device):
    """v2's own detections / map vectors in the critic's token format (batch of 1)."""
    from navsim.agents.para_ssr.modules.losses import denormalize_bbox

    cls = det_out["all_cls_scores"][-1][0].sigmoid().amax(-1)
    boxes = denormalize_bbox(det_out["all_bbox_preds"][-1][0].float())
    traj = det_out["traj_preds"][0].float()                        # [Q, modes, T, 2]
    mode = det_out["traj_cls_preds"][0].float().argmax(-1)         # [Q]
    keep = torch.nonzero(cls >= det_thr).squeeze(-1)
    if len(keep):
        keep = keep[torch.argsort(boxes[keep, :2].norm(dim=-1))][:MAX_AGENTS]
    a = torch.zeros(MAX_AGENTS, 9, device=device)
    fut = torch.zeros(MAX_AGENTS, traj.shape[2], 2, device=device)
    mask = torch.zeros(MAX_AGENTS, traj.shape[2], device=device)
    valid = torch.zeros(MAX_AGENTS, device=device)
    if len(keep):
        a[: len(keep)] = boxes[keep, :9]
        fut[: len(keep)] = traj[keep, mode[keep]]
        mask[: len(keep)] = 1
        valid[: len(keep)] = 1
    mcls = map_out["all_map_cls_scores"][-1][0].sigmoid()           # [V, classes]
    mpts = map_out["all_map_pts_preds"][-1][0].float()              # [V, P, 2] normalized
    score, label = mcls.max(dim=-1)
    mk = torch.nonzero(score >= map_thr).squeeze(-1)
    if len(mk):
        # same rule as the GT cache (cache_critic_tokens.py): keep the vectors nearest the ego,
        # so the GT and the predicted token sets are selected the same way
        mk = mk[torch.argsort((mpts[mk] - 0.5).abs().sum(dim=(1, 2)))][:MAX_MAP]
    pts = torch.zeros(MAX_MAP, mpts.shape[1], 2, device=device)
    lab = torch.zeros(MAX_MAP, dtype=torch.long, device=device)
    mv = torch.zeros(MAX_MAP, device=device)
    if len(mk):
        pts[: len(mk)] = mpts[mk]; lab[: len(mk)] = label[mk]; mv[: len(mk)] = 1
    return dict(agents=a[None], agent_fut=fut[None], agent_fut_mask=mask[None], agent_valid=valid[None],
                map_pts=pts[None], map_labels=lab[None], map_valid=mv[None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic", required=True)
    ap.add_argument("--v2-ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--anchors", default="data/planning_vb/trajectory_anchors_256.npy")
    ap.add_argument("--bb", default="34")
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--det-thr", type=float, default=0.3)
    ap.add_argument("--map-thr", type=float, default=0.3)
    ap.add_argument("--out", default="data/perception_use/critic_vs_student.npz")
    args, overrides = ap.parse_known_args()

    from navsim.common.dataloader import SceneLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = Cache(args.cache, args.labels)                              # same split as train_critic.py
    n_val = max(int(0.05 * len(ds)), 1)
    g = torch.Generator().manual_seed(0)
    _, val = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=g)
    val_tokens = [str(ds.tokens[i]) for i in val.indices]
    if args.limit:
        val_tokens = val_tokens[: args.limit]
    gt = {str(ds.tokens[i]): i for i in val.indices}
    print(f"held-out 장면 {len(val_tokens)}", flush=True)

    critic = EntityCritic(args.anchors, embed_dims=args.embed, num_layers=args.layers).to(device).eval()
    critic.load_state_dict(torch.load(args.critic, map_location=device)["model"])

    ov = overrides + [f"agent.config.image_architecture=resnet{args.bb}.tv_in1k",
                      "agent.config.use_task_interaction=true", "agent.config.use_det_motion_head=true",
                      "agent.config.use_map_head=true", "agent.config.plan_anchor=true",
                      "agent.config.plan_heading_from_xy=true",
                      f"agent.config.plan_anchor_file={Path(args.anchors).resolve()}"]
    agent = build_agent(args.v2_ckpt, ov).to(device)
    model = agent.para_ssr_model
    head = model.pts_bbox_head
    captured, orig_plan = {}, head.plan_from_bev

    def capture(*a, **k):
        captured["args"] = a
        return orig_plan(*a, **k)

    head.plan_from_bev = capture

    logs_dir, blobs_dir = split_dirs("trainval")
    loader = SceneLoader(logs_dir, blobs_dir, scene_filter("navtrain", tokens=val_tokens),
                         sensor_config=agent.get_sensor_config())
    tokens = [t for t in val_tokens if t in set(loader.tokens)]
    dl = DataLoader(FeatureData(tokens, loader, agent.get_feature_builders()), batch_size=1, num_workers=3,
                    collate_fn=collate)
    out = {k: [] for k in ("token", "v2", "v2_final", "critic_pred", "critic_gt", "label")}
    for n, (toks, feats, _) in enumerate(dl):
        if feats is None:
            continue
        feats = {k: v.to(device) for k, v in feats.items()}
        with torch.no_grad():
            pred = model(feats, run_aux=False)
            det_out, map_out = captured["args"][2], captured["args"][3]
            ent = predicted_entities(det_out, map_out, args.det_thr, args.map_thr, device)
            i = gt[toks[0]]
            base = {k: torch.as_tensor(np.asarray(ds.data[k][i]))[None].to(device) for k in
                    ("agents", "agent_fut", "agent_fut_mask", "agent_valid", "map_pts", "map_labels",
                     "map_valid", "ego_status", "command")}
            base["map_labels"] = base["map_labels"].long()
            ent.update({k: base[k] for k in ("ego_status", "command")})
            s_pred = critic(ent)
            s_gt = critic(base)
        out["token"].append(toks[0])
        out["v2"].append(pred["sim_rewards"][0].float().cpu().numpy().astype(np.float16))
        out["v2_final"].append(pred["plan_final_rewards"][0].float().cpu().numpy().astype(np.float16))
        out["critic_pred"].append(s_pred[0].float().cpu().numpy().astype(np.float16))
        out["critic_gt"].append(s_gt[0].float().cpu().numpy().astype(np.float16))
        out["label"].append(ds.labels[i].astype(np.float16))
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(tokens)}", flush=True)

    head.plan_from_bev = orig_plan
    res = {k: (np.asarray(v) if k == "token" else np.stack(v)) for k, v in out.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **res)
    report(res)


def report(res):
    """Pooled AUC hides the scene; what matters is the choice inside each scene."""
    lab = res["label"].astype(np.float32)                       # [N, 5, K]
    pdms = lambda x: x[:, 0] * x[:, 1] * (5 * x[:, 3] + 2 * x[:, 4] + 5 * x[:, 2]) / 12
    true = pdms(lab)                                            # [N, K] true PDMS of every anchor
    rng = np.random.default_rng(0)
    print(f"\n장면 {len(res['token'])}")
    print("주의: WoTE 학습 라벨 기준이다. 그 라벨의 ego_progress 는 256개 후보 중 최대 진행도로 정규화되어")
    print("      공식 EPDMS 와 척도가 다르다. 학습 진단용으로만 읽을 것. 평가는 navtest 공식 라벨로 한다")
    print("      (tools/plan_v2/clean_compare.py).")
    print("1. 채점기가 고른 anchor 의 점수 (WoTE 라벨 기준, 공식 EPDMS 아님)")
    rows = {"무작위 선택": true[np.arange(len(true)), rng.integers(0, true.shape[1], len(true))],
            "정답을 아는 선택": true.max(1)}
    for name in ("v2", "critic_pred", "critic_gt"):
        pick = pdms(res[name].astype(np.float32)).argmax(1)
        rows[name] = true[np.arange(len(true)), pick]
    if "v2_final" in res:                                   # v2's deployed rule (WoTE weighted log reward)
        pick = res["v2_final"].astype(np.float32).argmax(1)
        rows["v2 (자체 선택식)"] = true[np.arange(len(true)), pick]
    for k, v in rows.items():
        print(f"   {k:14s} {100 * v.mean():6.2f}")
    print("2. 장면 안에서의 anchor 순위 (장면별 AUC 평균)")
    print("               " + " ".join(f"{k[:3].upper():>6s}" for k in SIM_KEYS))
    for name in ("v2", "critic_pred", "critic_gt"):
        p = res[name].astype(np.float32)
        line = []
        for j in range(len(SIM_KEYS)):
            per = [auc(p[i, j][lab[i, j] > 0.5], p[i, j][lab[i, j] <= 0.5]) for i in range(len(p))]
            per = [x for x in per if np.isfinite(x)]
            line.append(f"{np.mean(per):6.3f}")
        print(f"  {name:12s} " + " ".join(line))


if __name__ == "__main__":
    main()
