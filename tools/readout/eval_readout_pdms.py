"""PDMS of a readout on cached BEVs (navtest), with the official scorer.

Only the trajectory source differs from run_pdm_score_gpu.py: poses come from
``h`` applied to a BEV cache instead of a camera model, then go through the
same ``pdm_score`` with the same metric cache and scoring parameters.

    # S_own      : teacher-trained h on the teacher navtest cache
    python tools/readout/eval_readout_pdms.py --readout runs/teacher_h1_s0/readout.pt \
        --bev-cache /data3/kyungmin/kd_teacher_resmap/navtest \
        --targets plan_targets_navtest.npz --out runs/teacher_h1_s0/pdms_teacher.csv
    # S_transfer : the SAME checkpoint on the student navtest cache (no training)
    ...  --bev-cache /data3/kyungmin/student_bev/navtest --out .../pdms_transfer.csv
    # S_ego      : a --no-bev readout.  Keep --bev-cache pointing at the cache the
    #              other arms use, so the token set is identical; "none" skips it.

A token without a BEV or target is an error unless --allow-missing: a PDMS over
a different token set is not comparable to anything.
"""
import argparse
import json
import lzma
import multiprocessing as mp
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import _env  # noqa: F401
from _env import REPO
from navsim.agents.para_ssr.readout.bev_cache import BevCache
from navsim.agents.para_ssr.readout.plan_targets import PlanTargetStore
from navsim.agents.para_ssr.readout.readout import load_readout

SCORING_YAML = REPO / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"

_SIM = _SCORER = _MC = None


def _init_worker(metric_cache):
    global _SIM, _SCORER, _MC
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import MetricCacheLoader

    cfg = OmegaConf.load(SCORING_YAML)
    _SIM = instantiate(cfg.simulator)
    _SCORER = instantiate(cfg.scorer)
    _MC = MetricCacheLoader(Path(metric_cache)).metric_cache_paths


def _score(item):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score

    token, poses = item
    row = {"token": token, "valid": True}
    try:
        with lzma.open(_MC[token], "rb") as f:
            mc = pickle.load(f)
        res = pdm_score(mc, Trajectory(poses.astype(np.float32)), _SIM.proposal_sampling, _SIM, _SCORER)
        row.update({k: float(v) for k, v in asdict(res).items()})
    except Exception as exc:  # scored as invalid, like run_pdm_score_gpu
        row["valid"] = False
        row["error"] = repr(exc)
    return row


@torch.no_grad()
def predict(model, adapter, tokens, store, cache, device, batch_size):
    out = {}
    for i in range(0, len(tokens), batch_size):
        toks = tokens[i : i + batch_size]
        t = [store.get(k) for k in toks]
        cmd = torch.from_numpy(np.stack([x["command"] for x in t])).to(device)
        ego = torch.from_numpy(np.stack([x["ego"] for x in t])).to(device)
        bev = None
        if model.cfg.use_bev:
            bev = torch.from_numpy(np.stack([cache.bev(k) for k in toks])).to(device).float()
            if adapter is not None:
                bev = adapter(bev)
        traj = model(bev, cmd, ego)["trajectory"].float().cpu().numpy()
        out.update(zip(toks, traj))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readout", required=True)
    ap.add_argument("--bev-cache", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--metric-cache", default=str(_env.DATA.parent / "exp/metric_cache"))
    ap.add_argument("--out", required=True, help="per-token CSV; a .json summary is written next to it")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=0, help="smoke test only")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from navsim.common.dataloader import MetricCacheLoader
    import pandas as pd

    device = torch.device(args.device)
    ckpt = torch.load(args.readout, map_location="cpu")
    model = load_readout(args.readout).to(device).eval()
    adapter = None
    if "adapter" in ckpt:
        from train_readout import Adapter

        adapter = Adapter(model.cfg.in_channels)
        adapter.load_state_dict(ckpt["adapter"])
        adapter.to(device).eval()

    store = PlanTargetStore(Path(args.targets))
    if args.bev_cache == "none":
        if model.cfg.use_bev:
            raise SystemExit("--bev-cache none is only valid for a --no-bev (h_ego) readout")
        cache = None
        have = set(store.tokens)
    else:
        cache = BevCache(args.bev_cache)
        have = set(store.tokens) & set(cache.tokens())
    mc_tokens = set(MetricCacheLoader(Path(args.metric_cache)).tokens)
    tokens = sorted(mc_tokens & have)
    missing = len(mc_tokens - have)
    if missing and not (args.allow_missing or args.max_tokens):
        raise SystemExit(
            f"{missing} of {len(mc_tokens)} metric-cache tokens lack a BEV or target; "
            "build the navtest caches first or pass --allow-missing"
        )
    if args.max_tokens:
        tokens = tokens[: args.max_tokens]
    if not tokens:
        raise SystemExit("no token has a metric cache, a target and a BEV")
    print(f"scoring {len(tokens)} tokens ({missing} metric-cache tokens missing)", flush=True)

    poses = predict(model, adapter, tokens, store, cache, device, args.batch_size)
    with mp.get_context("spawn").Pool(args.workers, _init_worker, (args.metric_cache,)) as pool:
        rows = list(pool.imap_unordered(_score, sorted(poses.items()), chunksize=8))

    df = pd.DataFrame(rows).sort_values("token")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    valid = df[df["valid"]]
    metrics = [c for c in df.columns if c not in ("token", "valid", "error")]
    summary = {
        "readout": str(args.readout),
        "bev_cache": str(args.bev_cache),
        "num_tokens": len(df),
        "num_invalid": int((~df["valid"]).sum()),
        "num_missing_from_metric_cache": missing,
        **{m: float(valid[m].mean()) for m in metrics},
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
