#!/usr/bin/env python
"""Stage E: E0 parity harness (PARA-SSR loss / logs / gradients bit-identical with the refiner options off).

  python tools/refiner/stageE_parity.py dump  [--out <golden.pt>]   # run ONCE on the unmodified code (before any edit)
  python tools/refiner/stageE_parity.py check [--golden <golden.pt>]  # after the edits: refiner_mode=off must match

The golden file holds the collated CPU batch of 2 real navtrain scenes (features + targets), the training-mode total
loss, every value of agent.latest_logs, every parameter gradient, the eval-mode predictions, the optimiser group layout
and the global torch RNG state after agent construction.  `check` rebuilds the agent with the current code (options
off), recomputes features / targets from the same scenes (must be equal to the golden batch), runs the same seeded step
on the GOLDEN batch and requires exact equality (torch.equal) of everything.

Helpers here are also used by stageE_smoke.py / tests (load_scenes, build_agent, collate, train_step).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(REPO / "data/dataset/maps"))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(REPO / "data/dataset"))

import torch  # noqa: E402

GOLDEN = Path("/home/external-user/ssd/yongjae_refiner/stageE/parity_golden.pt")
# two navtrain train_logs tokens with every GT store present (objects, sdf, metric cache)
PARITY_TOKENS: Tuple[Tuple[str, str], ...] = (
    ("1aa44d46e4ab5bc7", "2021.05.12.19.36.12_veh-35_00005_00204"),
    ("9cc09b76c2c957a3", "2021.05.12.19.36.12_veh-35_00568_01168"),
)
LOGS = REPO / "data/dataset/navsim_logs/trainval"
BLOBS = REPO / "data/dataset/sensor_blobs/trainval"
STEP_SEED = 1234


def build_agent(**overrides):
    """ParaSSRAgent with the reference (interaction_final) config, no pretrained download, built under seed 0 like
    run_training.py (pl.seed_everything(0) before instantiate)."""
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent

    kw = dict(backbone_pretrained=False, use_task_interaction=True, use_det_motion_head=True, use_map_head=True)
    kw.update(overrides)
    cfg = ParaSSRConfig(**kw)
    torch.manual_seed(0)
    agent = ParaSSRAgent(cfg, TrajectorySampling(time_horizon=4, interval_length=0.5), lr=1e-4)
    return agent


def load_scenes(tokens: Sequence[Tuple[str, str]], sensor_config, with_sensors: bool = True):
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader

    scenes = []
    for tok, log in tokens:
        sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                         log_names=[log], tokens=[tok])
        sl = SceneLoader(data_path=LOGS, sensor_blobs_path=BLOBS if with_sensors else None, scene_filter=sf,
                         sensor_config=sensor_config if with_sensors else SensorConfig.build_no_sensors())
        scenes.append(sl.get_scene_from_token(tok))
    return scenes


def collate(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def make_batch(agent, scenes) -> Tuple[Dict, Dict]:
    feats, targs = [], []
    for sc in scenes:
        f, t = {}, {}
        ai = sc.get_agent_input(False)
        for b in agent.get_feature_builders():
            f.update(b.compute_features(ai))
        for b in agent.get_target_builders():
            t.update(b.compute_targets(sc))
        feats.append(f)
        targs.append(t)
    return collate(feats), collate(targs)


def train_step(agent, features, targets, seed: int = STEP_SEED):
    """One seeded training-mode forward + compute_loss + backward -> (loss, logs, grads, predictions)."""
    agent.train()
    agent.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    preds = agent.forward(features)
    loss = agent.compute_loss(features, targets, preds)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in agent.named_parameters() if p.grad is not None}
    logs = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else torch.tensor(float(v)))
            for k, v in agent.latest_logs.items()}
    return loss.detach().clone(), logs, grads, preds


@torch.no_grad()
def eval_predictions(agent, features, seed: int = STEP_SEED) -> Dict[str, torch.Tensor]:
    agent.eval()
    torch.manual_seed(seed)
    out = agent.forward(features)
    return {k: v.detach().clone() for k, v in out.items() if isinstance(v, torch.Tensor)}


def opt_layout(agent):
    o = agent.get_optimizers()["optimizer"]
    return [dict(n=len(g["params"]), numel=int(sum(p.numel() for p in g["params"])), lr_scale=g.get("lr_scale"),
                 weight_decay=g.get("weight_decay")) for g in o.param_groups]


def rng_digest() -> str:
    return hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()


def _src_hashes():
    d = REPO / "navsim/agents/para_ssr"
    fs = ["para_ssr_agent.py", "para_ssr_loss.py", "para_ssr_targets.py", "para_ssr_model.py", "para_ssr_features.py",
          "configs/default.py"]
    return {f: hashlib.sha256((d / f).read_bytes()).hexdigest()[:16] for f in fs}


GOLDEN_THREADS = 8        # the golden was taken with 8 intra-op threads; float reductions depend on it


def dump(out: Path) -> None:
    torch.set_num_threads(GOLDEN_THREADS)
    agent = build_agent()
    rng_after_build = rng_digest()
    scenes = load_scenes(PARITY_TOKENS, agent.get_sensor_config())
    features, targets = make_batch(agent, scenes)
    loss, logs, grads, _ = train_step(agent, features, targets)
    preds = eval_predictions(agent, features)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(tokens=PARITY_TOKENS, features=features, targets=targets, loss=loss, logs=logs, grads=grads,
                    eval_preds=preds, opt_layout=opt_layout(agent), rng_after_build=rng_after_build,
                    src=_src_hashes(), torch=torch.__version__, num_threads=torch.get_num_threads()), out)
    print(f"golden -> {out}: loss {float(loss):.8f}, {len(logs)} logs, {len(grads)} grads, "
          f"{len(preds)} eval outputs, rng {rng_after_build[:12]}")


def _cmp(name: str, a, b, bad: List[str]) -> None:
    if isinstance(a, torch.Tensor):
        if not (isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)):
            bad.append(name)
    elif a != b:
        bad.append(name)


def check(golden: Path, overrides=None, recompute_inputs: bool = True) -> List[str]:
    g = torch.load(golden, map_location="cpu", weights_only=False)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(int(g.get("num_threads", GOLDEN_THREADS)))
    try:
        return _check(g, golden, overrides, recompute_inputs)
    finally:
        torch.set_num_threads(n_threads)


def _check(g, golden: Path, overrides=None, recompute_inputs: bool = True) -> List[str]:
    agent = build_agent(**(overrides or {}))
    bad: List[str] = []
    _cmp("rng_after_build", g["rng_after_build"], rng_digest(), bad)
    _cmp("opt_layout", g["opt_layout"], opt_layout(agent), bad)
    if recompute_inputs:
        scenes = load_scenes(g["tokens"], agent.get_sensor_config())
        f, t = make_batch(agent, scenes)
        _cmp("feature_keys", sorted(g["features"]), sorted(f), bad)
        _cmp("target_keys", sorted(g["targets"]), sorted(t), bad)
        for k in g["features"]:
            _cmp(f"features/{k}", g["features"][k], f.get(k), bad)
        for k in g["targets"]:
            _cmp(f"targets/{k}", g["targets"][k], t.get(k), bad)
    loss, logs, grads, _ = train_step(agent, g["features"], g["targets"])
    _cmp("loss", g["loss"], loss, bad)
    _cmp("log_keys", sorted(g["logs"]), sorted(logs), bad)
    for k in g["logs"]:
        _cmp(f"logs/{k}", g["logs"][k], logs.get(k), bad)
    _cmp("grad_keys", sorted(g["grads"]), sorted(grads), bad)
    for k in g["grads"]:
        _cmp(f"grads/{k}", g["grads"][k], grads.get(k), bad)
    preds = eval_predictions(agent, g["features"])
    _cmp("eval_keys", sorted(g["eval_preds"]), sorted(preds), bad)
    for k in g["eval_preds"]:
        _cmp(f"eval/{k}", g["eval_preds"][k], preds.get(k), bad)
    print(f"parity check vs {golden}: loss golden {float(g['loss']):.8f} now {float(loss):.8f}; "
          f"{len(g['grads'])} grads, {len(g['logs'])} logs, {len(g['eval_preds'])} eval outputs; "
          f"{'IDENTICAL' if not bad else 'MISMATCH: ' + ', '.join(bad[:20])}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "check"])
    ap.add_argument("--out", default=str(GOLDEN))
    ap.add_argument("--golden", default=str(GOLDEN))
    ap.add_argument("--no-recompute", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    if a.cmd == "dump":
        if Path(a.out).exists():
            raise SystemExit(f"{a.out} exists; the golden must come from the unmodified code (delete it deliberately)")
        dump(Path(a.out))
    else:
        bad = check(Path(a.golden), recompute_inputs=not a.no_recompute)
        raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
