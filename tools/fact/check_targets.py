"""Sanity + timing of FactTargetBuilder on real navtrain scenes (CPU).

usage: PYTHONPATH=<repo> python tools/fact/check_targets.py --n 20
Prints per-key shapes, route success rate, label statistics of the 256 WoTE anchors
under the proxy labels vs the exact labels (agreement), and the builder's time per scene.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from navsim.agents.para_ssr.configs.default import ParaSSRConfig
from navsim.agents.para_ssr.fact.proxy import proxy_labels
from navsim.agents.para_ssr.fact.targets import FactTargetBuilder
from navsim.agents.para_ssr.plan_map import DrivableAreaTargetBuilder
from navsim.agents.para_ssr.plan_score_targets import AnchorScoreTargetBuilder
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--repo", default=os.environ.get("NAVSIM_DEVKIT_ROOT", "."))
a = ap.parse_args()
repo = Path(a.repo)
root = Path(os.environ["OPENSCENE_DATA_ROOT"])

cfg = ParaSSRConfig(
    fact_enable=True, input_target=True,
    fact_path_file=str(repo / "data/fact_vocab/paths_1024.npy"),
    fact_speed_file=str(repo / "data/fact_vocab/speeds_256.npy"),
    plan_anchor_file=str(repo / "data/planning_vb/trajectory_anchors_256.npy"),
    plan_score_file=str(repo / "data/planning_vb/pdm_score_256"),
)
sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                 max_scenes=a.n, log_names=None)
loader = SceneLoader(root / "navsim_logs/trainval", root / "sensor_blobs/trainval", sf, SensorConfig.build_no_sensors())
fb, db, sb = FactTargetBuilder(cfg), DrivableAreaTargetBuilder(cfg), AnchorScoreTargetBuilder(cfg)
anchors = torch.tensor(np.load(cfg.plan_anchor_file), dtype=torch.float32)       # [256, 8, 3]

times, ok, agree = [], [], {k: [] for k in ("nc", "dac", "ttc", "ep", "comfort")}
for i, token in enumerate(loader.tokens[: a.n]):
    scene = loader.get_scene_from_token(token)
    t0 = time.time()
    t = fb.compute_targets(scene)
    t.update(db.compute_targets(scene))
    times.append(time.time() - t0)
    s = sb.compute_targets(scene)
    ok.append(float(t["fact_route_ok"]))
    if i == 0:
        for k, v in t.items():
            print(f"{k:18s} {tuple(v.shape)} {v.dtype}")
    v0 = torch.tensor([float(scene.frames[3].ego_status.ego_velocity[0])])
    tg = {k: v.unsqueeze(0) for k, v in t.items()}
    lab = proxy_labels(anchors.unsqueeze(0), tg, tuple(cfg.plan_map_extent), v0)
    if float(s["sim_reward_valid"]) > 0:
        ex = s["sim_reward"]                                                       # [5, 256] NC DAC EP TTC C
        for j, k in enumerate(("nc", "dac", "ep", "ttc", "comfort")):
            if k == "ep":
                agree[k].append(float(torch.corrcoef(torch.stack([lab[k][0], ex[j]]))[0, 1]))
            else:
                agree[k].append(float(((lab[k][0] > 0.5) == (ex[j] > 0.5)).float().mean()))
print(f"scenes {len(times)}  builder {np.mean(times):.3f} s/scene (max {np.max(times):.3f})  route_ok {np.mean(ok):.2f}")
for k, v in agree.items():
    if v:
        print(f"proxy vs exact {k:8s} {'corr' if k == 'ep' else 'agreement'} {np.mean(v):.3f}")
