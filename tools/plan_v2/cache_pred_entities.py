"""Cache the entities the v2 model itself predicts, in the critic's token format.

The critic is trained on GT entities, so it breaks when fed predicted ones (selection 79 -> 69).
These shards let it be trained on what a sensor model actually produces.

    python tools/plan_v2/cache_pred_entities.py --ckpt <v2 ckpt> --cache data/critic_cache \
        --out data/critic_cache_pred --shard 0 --shards 3 [--bb 34]
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "readout"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: F401,E402
from _env import scene_filter, split_dirs  # noqa: E402
from cache_student_bev import FeatureData, build_agent, collate  # noqa: E402
from critic_vs_student import predicted_entities  # noqa: E402

KEYS = ("agents", "agent_fut", "agent_fut_mask", "agent_valid", "map_pts", "map_labels", "map_valid")
PRED = ("sim_rewards", "im_rewards", "plan_final_rewards")   # the model's own scores, for the re-ranking probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True, help="GT token cache: its tokens define the scenes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="navtrain")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--bb", default="34")
    ap.add_argument("--anchors", default="data/planning_vb/trajectory_anchors_256.npy")
    ap.add_argument("--det-thr", type=float, default=0.3)
    ap.add_argument("--map-thr", type=float, default=0.3)
    args, overrides = ap.parse_known_args()

    from navsim.common.dataloader import SceneLoader

    tokens = []
    for f in sorted(glob.glob(f"{args.cache}/shard_*.npz")):
        tokens += list(np.load(f)["tokens"])
    tokens = sorted(str(t) for t in tokens)
    tokens = list(np.array_split(np.asarray(tokens), args.shards)[args.shard])
    print(f"shard {args.shard}/{args.shards}: {len(tokens)} scenes", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ov = overrides + [f"agent.config.image_architecture=resnet{args.bb}.tv_in1k",
                      "agent.config.use_task_interaction=true", "agent.config.use_det_motion_head=true",
                      "agent.config.use_map_head=true", "agent.config.plan_anchor=true",
                      "agent.config.plan_heading_from_xy=true",
                      f"agent.config.plan_anchor_file={Path(args.anchors).resolve()}"]
    agent = build_agent(args.ckpt, ov).to(device)
    model = agent.para_ssr_model
    head = model.pts_bbox_head
    captured, orig_plan = {}, head.plan_from_bev

    def capture(*a, **k):
        captured["args"] = a
        return orig_plan(*a, **k)

    head.plan_from_bev = capture
    logs_dir, blobs_dir = split_dirs("trainval" if args.split == "navtrain" else "test")
    loader = SceneLoader(logs_dir, blobs_dir, scene_filter(args.split, tokens=tokens),
                         sensor_config=agent.get_sensor_config())
    tokens = [t for t in tokens if t in set(loader.tokens)]
    dl = DataLoader(FeatureData(tokens, loader, agent.get_feature_builders()), batch_size=1, num_workers=3,
                    collate_fn=collate)
    data, names = {k: [] for k in KEYS + PRED}, []
    t0 = time.time()
    for n, (toks, feats, _) in enumerate(dl):
        if feats is None:
            continue
        feats = {k: v.to(device) for k, v in feats.items()}
        with torch.no_grad():
            out = model(feats, run_aux=False)
            ent = predicted_entities(captured["args"][2], captured["args"][3], args.det_thr, args.map_thr, device)
        names.append(toks[0])
        for k in KEYS:
            v = ent[k][0].cpu().numpy()
            data[k].append(v.astype(np.int64) if k == "map_labels" else v.astype(np.float32))
        for k in PRED:
            data[k].append(out[k][0].float().cpu().numpy().astype(np.float16))
        if (n + 1) % 2000 == 0:
            print(f"  {n + 1}/{len(tokens)}  {(time.time() - t0) / (n + 1):.3f} s/scene", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"shard_{args.shard:03d}.npz", tokens=np.asarray(names),
                        **{k: np.stack(v) for k, v in data.items()})
    print(f"saved {len(names)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
