"""Create navsim_v2/navsim/planning/script/run_pdm_score_from_trajectories.py from run_pdm_score_one_stage.py.

Only the trajectory source changes: a pickle {"trajectories": {token: [N,3]}, "meta": {num_poses, interval_length}}
written by SSR's tools/dump_navtest_trajectories.py.  Simulator, scorer, traffic agents, two-frame extended comfort
and compute_final_scores are the unmodified v2 code.  Hydra: +trajectory_pkl=<path>.
"""
import sys
from pathlib import Path

src = Path(sys.argv[1]) / "navsim/planning/script/run_pdm_score_one_stage.py"
dst = Path(sys.argv[1]) / "navsim/planning/script/run_pdm_score_from_trajectories.py"
s = src.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)


rep("import os\n", "import os\nimport pickle\n")
rep("from navsim.common.dataclasses import PDMResults, SensorConfig\n",
    "from navsim.common.dataclasses import PDMResults, SensorConfig, Trajectory\n")
rep("""    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
""", """    with open(cfg.trajectory_pkl, "rb") as f:
        _dump = pickle.load(f)
    trajectories, meta = _dump["trajectories"], _dump["meta"]
    trajectory_sampling = TrajectorySampling(
        num_poses=int(meta["num_poses"]), interval_length=float(meta["interval_length"])
    )
""")
rep("""        sensor_config=agent.get_sensor_config(),
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
""", """        sensor_config=SensorConfig.build_no_sensors(),
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    missing = [t for t in tokens_to_evaluate if t not in trajectories]
    if missing:
        raise RuntimeError(f"{len(missing)} tokens have no dumped trajectory, e.g. {missing[:3]}")
""")
rep("""            agent_input = scene_loader.get_agent_input_from_token(token)
            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)
""", """            trajectory = Trajectory(np.asarray(trajectories[token], dtype=np.float32), trajectory_sampling)
""")
# main(): check every scene token has a trajectory before scoring
rep("""    logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")
""", """    with open(cfg.trajectory_pkl, "rb") as f:
        _dumped = set(pickle.load(f)["trajectories"])
    _missing = [t for t in tokens_to_evaluate if t not in _dumped]
    if _missing:
        raise RuntimeError(f"{len(_missing)} tokens have no dumped trajectory, e.g. {_missing[:3]}")
    logger.info(f"Trajectories from {cfg.trajectory_pkl}")
    logger.info(f"Starting pdm scoring of {len(tokens_to_evaluate)} scenarios...")
""")
s = s.replace('"""\nimport', '"""\nimport', 1)
dst.write_text("# Generated from run_pdm_score_one_stage.py: trajectories come from a pickle instead of an agent.\n" + s)
print("wrote", dst)
