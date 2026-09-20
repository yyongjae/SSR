"""WoTE (public checkpoint) on navtest: planned trajectory + its own predicted road mask per token.

For the "does the planner ignore its own perception" check on a public model: WoTE
predicts a BEV semantic map (128 x 256 at 0.25 m: 32 m ahead, +-32 m; class 1 = road =
LANE u INTERSECTION, the same definition as PARA-SSR's 'road').  The first call of
``bev_semantic_head`` in a forward pass is the current-frame map (later calls are the
world model's future maps).

    WOTE_EXTRA=data/planning_vb CUDA_VISIBLE_DEVICES=2 python tools/plan_v2/dump_wote.py \
        --ckpt data/wote/wote_epoch29.ckpt --out data/wote/navtest_dump.npz

Saved: tokens, trajectory [N, 8, 3], road [N, 128*256/8] packed bits of
(argmax in ON_ROAD), sem_classes note.  ON_ROAD = road, centerline, vehicles, ego box:
classes drawn ON the drivable surface and overwriting 'road' in the label raster.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "readout"))
import _env  # noqa: F401,E402
from _env import scene_filter, split_dirs  # noqa: E402
from cache_student_bev import FeatureData, collate  # noqa: E402

ON_ROAD = (1, 3, 5, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.WoTE.WoTE_agent import WoTEAgent
    from navsim.agents.WoTE.configs.default import WoTEConfig
    from navsim.common.dataloader import SceneLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = WoTEAgent(config=WoTEConfig(), lr=1e-4, checkpoint_path=args.ckpt,
                      trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5))
    agent.initialize()
    agent = agent.to(device).eval()
    model = agent.WoTE_model
    first = {}
    model.bev_semantic_head.register_forward_hook(lambda m, i, o: first.setdefault("map", o.detach()))

    logs_dir, blobs_dir = split_dirs("test")
    loader = SceneLoader(logs_dir, blobs_dir, scene_filter("navtest"), sensor_config=agent.get_sensor_config())
    tokens = sorted(loader.tokens)
    if args.limit:
        tokens = tokens[: args.limit]
    dl = DataLoader(FeatureData(tokens, loader, agent.get_feature_builders()), batch_size=args.batch_size,
                    num_workers=args.workers, collate_fn=collate)
    names, traj, road, missing = [], [], [], []
    # WoTE_model.forward_test computes these inside; recomputed here so the diagnostics can use
    # the per-candidate sub-scores (sim rewards) and the ranked candidates, not only the winner.
    extra = {k: [] for k in ("final_rewards", "im_rewards", "sim_rewards", "topk_index", "topk_trajectory")}
    on_road = torch.tensor(ON_ROAD, device=device)
    t0 = time.time()
    with torch.no_grad():
        for toks, feats, failed in dl:
            missing.extend(t for t, _ in failed)
            if feats is None:
                continue
            feats = {k: v.to(device) for k, v in feats.items()}
            first.clear()
            model.is_eval = True
            enc = model.process_trajectory_and_reward(feats)               # forward_test without its selection
            feat = enc["reward_feature"]
            im = torch.softmax(model.reward_head(feat).squeeze(-1).float(), dim=-1)
            sim = torch.cat([h(feat) for h in model.sim_reward_heads], dim=-1).permute(0, 2, 1).float().sigmoid()
            final = model.weighted_reward_calculation(im, [sim[:, i] for i in range(sim.shape[1])])
            anch = model.trajectory_anchors.unsqueeze(0).float()            # [1, K, T, 3]
            off = enc["trajectory_offset"].reshape(anch.shape[0] or 1, *anch.shape[1:]) if enc["trajectory_offset"].dim() != 4 \
                else enc["trajectory_offset"].float()
            all_traj = anch + off                                          # [B, K, T, 3]
            top = final.topk(3, dim=-1).indices
            gather = top[:, :, None, None].expand(-1, -1, all_traj.shape[2], all_traj.shape[3])
            top_traj = all_traj.gather(1, gather)
            sem = first["map"].argmax(dim=1)                                   # [B, 128, 256]
            mask = torch.isin(sem, on_road).cpu().numpy()
            names.extend(toks)
            traj.append(top_traj[:, 0].float().cpu().numpy().reshape(len(toks), 8, 3))
            road.append(np.packbits(mask.reshape(len(toks), -1), axis=1))
            extra["final_rewards"].append(final.float().cpu().numpy().astype(np.float16))
            extra["im_rewards"].append(im.float().cpu().numpy().astype(np.float16))
            extra["sim_rewards"].append(sim.float().cpu().numpy().astype(np.float16))
            extra["topk_index"].append(top.cpu().numpy())
            extra["topk_trajectory"].append(top_traj.float().cpu().numpy().astype(np.float32))
            if len(names) % 2000 < args.batch_size:
                print(f"{len(names)}/{len(tokens)}  {(time.time() - t0) / len(names):.3f} s/frame", flush=True)
    np.savez_compressed(args.out, tokens=np.asarray(names), trajectory=np.concatenate(traj),
                        road=np.concatenate(road), shape=np.asarray(mask.shape[1:]), missing=np.asarray(missing),
                        **{k: np.concatenate(v) for k, v in extra.items()})
    print(f"saved {len(names)} ({len(missing)} missing), map {mask.shape[1:]} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
