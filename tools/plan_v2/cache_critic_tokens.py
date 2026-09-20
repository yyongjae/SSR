"""Cache the GT perception tokens a token-only critic needs (no images), for navtrain.

Per scene: the GT agents (box, velocity, 4 s future at 0.5 s) and the GT map polylines that
ParaSSRTargetBuilder already produces, plus the ego status and the human trajectory.  The
labels are WoTE's PDM scores of the 256 anchors, read separately (plan_score_targets).

    python tools/plan_v2/cache_critic_tokens.py --out data/critic_cache [--split navtrain] [--workers 16]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "readout"))
import _env  # noqa: F401,E402
from _env import scene_filter, split_dirs  # noqa: E402

MAX_AGENTS, MAX_MAP, MAP_PTS = 32, 48, 20


def scene_tokens(scene, builder):
    t = builder.compute_targets(scene)
    boxes, valid = t["gt_boxes"].numpy(), t["gt_valid"].numpy()
    fut, fmask = t["gt_fut_trajs"].numpy(), t["gt_fut_masks"].numpy()
    keep = np.flatnonzero(valid)
    if len(keep):                                    # nearest agents first
        keep = keep[np.argsort(np.hypot(boxes[keep, 0], boxes[keep, 1]))][:MAX_AGENTS]
    a = np.zeros((MAX_AGENTS, 9), np.float32); af = np.zeros((MAX_AGENTS, fut.shape[1], 2), np.float32)
    am = np.zeros((MAX_AGENTS, fut.shape[1]), np.float32); av = np.zeros(MAX_AGENTS, np.float32)
    a[: len(keep)] = boxes[keep]; af[: len(keep)] = fut[keep]; am[: len(keep)] = fmask[keep]; av[: len(keep)] = 1
    pts, lab, mvalid = t["gt_map_pts"].numpy(), t["gt_map_labels"].numpy(), t["gt_map_valid"].numpy()
    if pts.ndim == 4:                                # [V, orders, P, 2] -> first ordering
        pts = pts[:, 0]
    mk = np.flatnonzero(mvalid)
    if len(mk):
        mk = mk[np.argsort(np.abs(pts[mk] - 0.5).sum(axis=(1, 2)))][:MAX_MAP]     # near the ego first
    m = np.zeros((MAX_MAP, MAP_PTS, 2), np.float32); ml = np.zeros(MAX_MAP, np.int64); mv = np.zeros(MAX_MAP, np.float32)
    if len(mk):
        p = pts[mk]
        if p.shape[1] != MAP_PTS:                    # resample to a fixed point count
            i = np.linspace(0, p.shape[1] - 1, MAP_PTS)
            p = np.stack([np.stack([np.interp(i, np.arange(p.shape[1]), q[:, d]) for d in (0, 1)], -1) for q in p])
        m[: len(mk)] = p; ml[: len(mk)] = lab[mk]; mv[: len(mk)] = 1
    es = scene.frames[scene.scene_metadata.num_history_frames - 1].ego_status
    return dict(agents=a, agent_fut=af, agent_fut_mask=am, agent_valid=av,
                map_pts=m, map_labels=ml, map_valid=mv,
                ego_status=np.concatenate([np.asarray(es.ego_velocity, np.float32),
                                           np.asarray(es.ego_acceleration, np.float32)]),
                command=t["command"].numpy().astype(np.float32),
                trajectory=t["trajectory"].numpy().astype(np.float32))


def run(shard, tokens, out, split):
    from dataclasses import replace
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    from navsim.agents.para_ssr.para_ssr_targets import ParaSSRTargetBuilder

    cfg = replace(ParaSSRConfig(), use_map_head=True, use_det_motion_head=True)
    builder = ParaSSRTargetBuilder(cfg, TrajectorySampling(time_horizon=4, interval_length=0.5))
    logs_dir, blobs_dir = split_dirs("trainval" if split == "navtrain" else "test")
    loader = SceneLoader(logs_dir, blobs_dir, scene_filter(split, tokens=tokens),
                         sensor_config=SensorConfig.build_no_sensors())
    keys, data = None, {}
    ok = []
    for tok in tokens:
        try:
            d = scene_tokens(loader.get_scene_from_token(tok), builder)
        except Exception as exc:  # a scene without a route or map coverage
            print(f"  skip {tok}: {type(exc).__name__}", flush=True)
            continue
        if keys is None:
            keys = list(d)
            data = {k: [] for k in keys}
        for k in keys:
            data[k].append(d[k])
        ok.append(tok)
    np.savez_compressed(out / f"shard_{shard:03d}.npz", tokens=np.asarray(ok),
                        **{k: np.stack(v).astype(np.float32 if v[0].dtype != np.int64 else np.int64) for k, v in data.items()})
    return shard, len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="navtrain")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subsample", type=int, default=0)
    args = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logs_dir, blobs_dir = split_dirs("trainval" if args.split == "navtrain" else "test")
    tokens = sorted(SceneLoader(logs_dir, blobs_dir, scene_filter(args.split),
                                sensor_config=SensorConfig.build_no_sensors()).tokens)
    if args.limit:
        tokens = tokens[: args.limit]
    if args.subsample and args.subsample < len(tokens):
        tokens = sorted(np.random.default_rng(0).choice(tokens, args.subsample, replace=False))
    shards = np.array_split(np.asarray(tokens), max(args.workers * 4, 1))
    print(f"{len(tokens)} tokens -> {len(shards)} shards", flush=True)
    done = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for shard, n in ex.map(run, range(len(shards)), [list(s) for s in shards],
                               [out] * len(shards), [args.split] * len(shards)):
            done += n
            print(f"shard {shard} done ({n}), total {done}/{len(tokens)}", flush=True)


if __name__ == "__main__":
    main()
