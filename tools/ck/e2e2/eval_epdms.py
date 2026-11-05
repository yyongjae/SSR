#!/usr/bin/env python
"""CK2 e2e EPDMS evaluation on NAVSIM v2.2 navtest and navhard_two_stage.

  cd /workspace/byounggun/SSR && PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    /venv/ssr/bin/python tools/ck/e2e2/eval_epdms.py --run-dir <RUN> --split navtest [--ckpt last] \\
    [--gpus 0,1,2,3] [--stage dump,eval,summary | all] [--workers 48] [--fresh]

  cd /workspace/byounggun/SSR && PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    /venv/ssr/bin/python tools/ck/e2e2/eval_epdms.py --run-dir <RUN> --split navhard [--ckpt last] \\
    [--gpus 0,1,2,3] [--stage dump,eval,summary | all] [--workers 48] [--fresh]

Stages (--stage all runs dump -> eval -> summary):
  dump     Multi-GPU sharded forward pass (CKE2E2.infer over the 96-candidate pool with CK_BEV64=1)
           producing {tokens.json, cand96.npy, lat_traj.npy, valid96.npy, v2_final.npy, v2_im.npy,
           v2_sim.npy, score_logit.npy}. Automatically seeds original scenes on navhard from navtest
           when available.
  eval     Computes ego-kinematic comfort/corridor features (HC, CC, 2-frame EC) for the 96 candidates,
           selects the trajectory per token via our Stage-1/Stage-2 risk-averse log-linear blend
           (beta=0.75, w_safe=1.50, w_cc=2.0), and runs official NAVSIM v2.2 EPDMS scoring.
  summary  Writes markdown and JSON summaries of all 9 EPDMS sub-metrics and composite EPDMS scores.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shapely
import yaml

CK = str(Path(__file__).resolve().parents[3])
if CK not in sys.path:
    sys.path.insert(0, CK)
from navsim.common import ssr_paths as SP  # noqa: E402

NAVSIM22 = os.environ.get("NAVSIM22_ROOT") or SP.NAVSIM_V22_POST
DATA_ROOT = Path(os.environ.get("SSR_TEST_DATA_ROOT") or SP.SSR_TEST_DATA_ROOT)
MC_NAVTEST_DEFAULT = os.environ.get("METRIC_CACHE_NAVTEST_V22") or SP.NAVSIM_V22_CACHE
MC_NAVHARD_DEFAULT = os.environ.get("METRIC_CACHE_NAVHARD_V22") or SP.NAVSIM_V22_NAVHARD_CACHE
MC_NAVTEST_V1_DEFAULT = os.environ.get("SSR_NAVTEST_MC") or SP.SSR_NAVTEST_MC
PY = SP.SSR_PY
STAGES = ("dump", "eval", "summary")
SPLITS = ("navtest", "navhard")
METRICS = ("epdms", "pdms", "all")
TIE_ORDER = np.concatenate([np.arange(0, 96, 6), np.array([c for c in range(96) if c % 6], np.int64)])


def kst() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime(time.time() + 9 * 3600))


def log(msg: str) -> None:
    print(f"[{kst()}] [eval_epdms] {msg}", flush=True)


def _load_yaml(p: Path) -> Any:
    Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(p, "r") as f:
        return yaml.load(f, Loader=Loader)


def pick_1row(s_row: np.ndarray, ok_row: np.ndarray) -> int:
    s = np.where(ok_row & np.isfinite(s_row), s_row, -np.inf)
    j = int(np.argmax(s[TIE_ORDER]))
    return int(TIE_ORDER[j])


# ----------------------------------------------------------------------------------------------- checkpoint & paths
def resolve_ckpt_and_tag(run_dir: Optional[Path], spec: str) -> Tuple[Path, str]:
    """Resolve checkpoint specification ('last', 'epoch:19', 'ep20', or file path) and canonical tag."""
    spec = (spec or "last").strip()
    p = Path(spec)
    if p.is_file():
        ckpt = p.resolve()
    elif run_dir is not None:
        search_dirs = [
            run_dir,
            run_dir / "checkpoints",
            run_dir / "train",
        ]
        if spec in ("", "last"):
            cands: List[Path] = []
            for d in search_dirs:
                if (d / "last.ckpt").is_file():
                    cands.append((d / "last.ckpt").resolve())
                cands.extend(sorted(d.rglob("last.ckpt")))
            if not cands:
                all_ep: List[Tuple[int, Path]] = []
                for d in search_dirs:
                    if not d.exists():
                        continue
                    for f in d.rglob("epoch=*.ckpt"):
                        m = re.search(r"epoch=(\d+)", f.name)
                        if m:
                            all_ep.append((int(m.group(1)), f.resolve()))
                if all_ep:
                    all_ep.sort(key=lambda x: x[0])
                    ckpt = all_ep[-1][1]
                else:
                    raise SystemExit(f"no checkpoint found under {run_dir}")
            else:
                ckpt = cands[-1]
        else:
            raw_num = re.sub(r"^(epoch|ep)[:=_ ]*", "", spec, flags=re.IGNORECASE)
            if not raw_num.isdigit():
                raise SystemExit(f"cannot resolve checkpoint spec {spec!r} under {run_dir}")
            num = int(raw_num)
            target_epochs = [num - 1, num] if spec.lower().startswith("ep") and not spec.lower().startswith("epoch") else [num, num - 1]
            found = None
            for te in target_epochs:
                for d in search_dirs:
                    if not d.exists():
                        continue
                    hits = sorted(f.resolve() for f in d.rglob(f"epoch={te}-*.ckpt")) + sorted(
                        f.resolve() for f in d.rglob(f"epoch={te}.ckpt")
                    )
                    if hits:
                        found = hits[-1]
                        break
                if found is not None:
                    break
            if found is None:
                raise SystemExit(f"no checkpoint matching {spec!r} under {run_dir}")
            ckpt = found
    else:
        raise SystemExit(f"checkpoint not found: {spec!r} (pass --run-dir or a valid file path to --ckpt)")

    m = re.search(r"epoch=(\d+)", ckpt.name)
    if m:
        ep_idx = int(m.group(1))
        tag = f"ep{ep_idx + 1}"
    else:
        tag = ckpt.stem.replace("=", "_")
    return ckpt, tag


def resolve_hydra_cfg(run_dir: Path, ckpt: Path, explicit_cfg: str = "") -> Path:
    """Locate hydra/config.yaml from --hydra-cfg, run_dir, or checkpoint ancestors, or synthesize one."""
    if explicit_cfg:
        p = Path(explicit_cfg).resolve()
        if not p.is_file():
            raise SystemExit(f"--hydra-cfg not found: {p}")
        return p
    cands = [
        run_dir / "train/code/hydra/config.yaml",
        run_dir / "code/hydra/config.yaml",
        ckpt.parent / "code/hydra/config.yaml",
        ckpt.parent.parent / "code/hydra/config.yaml",
        ckpt.parent.parent.parent / "code/hydra/config.yaml",
        ckpt.parent.parent.parent.parent / "code/hydra/config.yaml",
        Path("/workspace/byounggun/ssr_outputs/final/code/hydra/config.yaml"),
    ]
    for c in cands:
        if c.is_file():
            return c.resolve()

    # Synthesize config.yaml from repository defaults (v2_v15_v7 recipe)
    if CK not in sys.path:
        sys.path.insert(0, CK)
    from omegaconf import OmegaConf
    from tools.ck.e2e import launch_util as LU
    from tools.ck.e2e2 import launch_util2 as LU2

    synth_path = run_dir / "eval_epdms" / "_synthesized_hydra_config.yaml"
    synth_path.parent.mkdir(parents=True, exist_ok=True)
    base_agent = OmegaConf.load(Path(CK) / "navsim/planning/script/config/common/agent/para_ssr_agent.yaml")
    cfg = OmegaConf.create({"agent": OmegaConf.to_container(base_agent, resolve=False)})
    OmegaConf.set_struct(cfg, False)
    ck = LU2.scale_mb_counters(LU2.load_ck2_yaml(), 4)
    ck = LU.apply_sets(ck, LU.read_lines(str(Path(CK) / "tools/ck/e2e2/v2_v15_v7.set")))
    overrides = LU2.recipe2("v2_final_main", str(run_dir), 4, 6, 30, "disable", "main")
    overrides += [f"++agent.config.ck_e2e2.{k}={LU.hydra_value(v)}" for k, v in LU.flatten(ck)]
    overrides += LU.read_lines(str(Path(CK) / "tools/ck/e2e2/v2_v15_v7_arm.extra"))
    for o in overrides:
        if o.startswith("~"):
            continue
        k, v = o.lstrip("+").split("=", 1)
        if k.startswith("agent."):
            OmegaConf.update(cfg, k, OmegaConf.create(f"v: {v}").v, merge=True)
    OmegaConf.save(cfg, synth_path)
    return synth_path


def resolve_pool_dir(run_dir: Path, out_dir: Path, tag: str, split: str) -> Path:
    """Return existing pool96 directory if already dumped under run_dir, else out_dir/pool96_<tag>_<split>."""
    legacy = run_dir / f"pool96_{tag}_{split}"
    if (legacy / "tokens.json").is_file() and (legacy / "lat_traj.npy").is_file():
        return legacy
    return out_dir / f"pool96_{tag}_{split}"


def resolve_kin_dir(run_dir: Path, out_dir: Path, tag: str, split: str) -> Path:
    """Return existing kin_cache directory if already cached under run_dir, else out_dir/kin_cache."""
    legacy = run_dir / "kin_cache"
    if (legacy / f"kin_{split}_{tag}.npz").is_file() and (legacy / f"kin_{split}_{tag}_meta.json").is_file():
        return legacy
    return out_dir / "kin_cache"


# ----------------------------------------------------------------------------------------------- dataset & dump
def _get_synth_map(cache_dir: Path, rsi_tokens: set) -> Dict[str, Path]:
    """Map navhard_two_stage reactive_synthetic_initial_tokens to their .pkl paths."""
    known_caches = [
        cache_dir / "synth_map.json",
        Path("/workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep29/navhard/synth_map.json"),
    ]
    for cp in known_caches:
        if cp.is_file():
            raw = json.loads(cp.read_text())
            hit = {k: Path(v) for k, v in raw.items() if k in rsi_tokens}
            if len(hit) == len(rsi_tokens):
                return hit

    syn_dir = DATA_ROOT / "navhard_two_stage/synthetic_scene_pickles"
    if not syn_dir.is_dir():
        raise SystemExit(f"synthetic_scene_pickles directory not found: {syn_dir}")
    log(f"scanning {syn_dir} to index {len(rsi_tokens)} synthetic scenes ...")
    raw_map: Dict[str, str] = {}
    for pkl_path in sorted(syn_dir.glob("*.pkl")):
        with open(pkl_path, "rb") as f:
            s = pickle.load(f)
        init_tok = str(s["scene_metadata"]["initial_token"])
        raw_map[init_tok] = str(pkl_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "synth_map.json").write_text(json.dumps(raw_map))
    return {k: Path(v) for k, v in raw_map.items() if k in rsi_tokens}


def split_tokens(split: str, cache_dir: Path) -> Tuple[List[str], List[str], Dict[str, Path], int]:
    sf_dir = Path(NAVSIM22) / "navsim/planning/script/config/common/train_test_split/scene_filter"
    if split == "navhard":
        sf = _load_yaml(sf_dir / "navhard_two_stage.yaml")
        orig = [str(t) for t in sf["tokens"]]
        rsi = set(map(str, sf["reactive_synthetic_initial_tokens"]))
        synth = _get_synth_map(cache_dir, rsi)
        return orig, [str(x) for x in sf["log_names"]], synth, 8
    elif split == "navtest":
        sf = _load_yaml(sf_dir / "navtest.yaml")
        orig = [str(t) for t in sf["tokens"]]
        return orig, [str(x) for x in sf["log_names"]], {}, 10
    else:
        raise ValueError(f"unsupported split: {split}")


class SynthScene:
    def __init__(self, path: Path, sensor_config, syn_sens_root: Path):
        self.path = path
        self.sensor_config = sensor_config
        self.syn_sens_root = syn_sens_root

    def get_agent_input(self):
        from navsim.common.dataclasses import AgentInput, Cameras, EgoStatus, Lidar
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
            convert_absolute_to_relative_se2_array,
        )
        from nuplan.common.actor_state.state_representation import StateSE2

        with open(self.path, "rb") as f:
            s = pickle.load(f)
        nh = int(s["scene_metadata"]["num_history_frames"])
        fr = s["frames"][:nh]
        glob_poses = np.array([np.asarray(x["ego_status"]["ego_pose"], np.float64) for x in fr])
        local = convert_absolute_to_relative_se2_array(StateSE2(*glob_poses[-1]), glob_poses)
        ego, cams, lids = [], [], []
        for i, x in enumerate(fr):
            es = x["ego_status"]
            ego.append(
                EgoStatus(
                    ego_pose=local[i],
                    ego_velocity=es["ego_velocity"],
                    ego_acceleration=es["ego_acceleration"],
                    driving_command=es["driving_command"],
                )
            )
            names = self.sensor_config.get_sensors_at_iteration(i)
            cams.append(
                Cameras.from_camera_dict(
                    sensor_blobs_path=self.syn_sens_root,
                    camera_dict=x["camera_dict"],
                    sensor_names=names,
                )
            )
            lids.append(Lidar())
        return AgentInput(ego, cams, lids)

    def get_future_trajectory(self, *a, **k):
        raise RuntimeError("synthetic scene: no future frames")


class CombinedSceneLoader:
    def __init__(self, orig_loader, synth: Dict[str, Path], sensor_config, syn_sens_root: Path):
        self.orig = orig_loader
        self.synth = synth
        self.sc = sensor_config
        self.syn_sens_root = syn_sens_root

    def get_scene_from_token(self, tok: str):
        if tok in self.synth:
            return SynthScene(self.synth[tok], self.sc, self.syn_sens_root)
        return self.orig.get_scene_from_token(tok)


def run_dump_shard(
    split: str,
    pool_dir: Path,
    ckpt: Path,
    hydra_cfg: Path,
    shard: int,
    nshard: int,
    batch_size: int,
    workers: int,
    seed_pool_dir: Optional[Path] = None,
) -> int:
    os.environ["CK_BEV64"] = "1"
    if sys.path[0] != CK:
        sys.path.insert(0, CK)
    import torch
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.script import run_aux_evaluation as RAE
    from tools.ck.data import dump_v2 as DV

    torch.set_num_threads(2)
    pool_dir.mkdir(parents=True, exist_ok=True)
    orig, logs, synth, num_fut = split_tokens(split, pool_dir.parent)
    all_toks = orig + sorted(synth)
    toks = [t for i, t in enumerate(all_toks) if i % nshard == shard]
    fp = pool_dir / f"pool96_part{shard}.pkl"
    rows: Dict[str, Dict[str, np.ndarray]] = pickle.load(open(fp, "rb")) if fp.exists() else {}

    # Seed from an existing navtest pool96 directory if available (saves 450 original scenes on navhard)
    if seed_pool_dir is not None and seed_pool_dir.is_dir():
        if (seed_pool_dir / "tokens.json").is_file() and (seed_pool_dir / "lat_traj.npy").is_file():
            stoks = json.loads((seed_pool_dir / "tokens.json").read_text())
            s2i = {t: i for i, t in enumerate(stoks)}
            need_seed = [t for t in toks if t not in rows and t in s2i]
            if need_seed:
                c96 = np.load(seed_pool_dir / "cand96.npy", mmap_mode="r")
                l96 = np.load(seed_pool_dir / "lat_traj.npy", mmap_mode="r")
                vf = np.load(seed_pool_dir / "v2_final.npy", mmap_mode="r")
                vim = np.load(seed_pool_dir / "v2_im.npy", mmap_mode="r")
                vsim = np.load(seed_pool_dir / "v2_sim.npy", mmap_mode="r")
                sl = np.load(seed_pool_dir / "score_logit.npy", mmap_mode="r")
                val96 = np.load(seed_pool_dir / "valid96.npy", mmap_mode="r")
                for t in need_seed:
                    idx = s2i[t]
                    rows[t] = {
                        "cand96": np.array(c96[idx], dtype=np.float32),
                        "lat_traj": np.array(l96[idx], dtype=np.float32),
                        "v2_final": np.array(vf[idx], dtype=np.float32),
                        "v2_im": np.array(vim[idx], dtype=np.float32),
                        "v2_sim": np.array(vsim[idx], dtype=np.float32),
                        "score_logit": np.array(sl[idx], dtype=np.float32),
                        "valid96": np.array(val96[idx], dtype=bool),
                    }
        else:
            for sf_pkl in sorted(seed_pool_dir.glob("pool96_part*.pkl")):
                sdict = pickle.load(open(sf_pkl, "rb"))
                for t in toks:
                    if t not in rows and t in sdict:
                        rows[t] = sdict[t]

    todo = [t for t in toks if t not in rows]
    log(f"[{split}] shard {shard}/{nshard}: todo {len(todo)} (done {len(rows)})")
    if not todo:
        tmp = str(fp) + ".tmp"
        pickle.dump(rows, open(tmp, "wb"))
        os.replace(tmp, fp)
        return 0

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    dev = torch.device("cuda:0" if vis.strip() != "" and torch.cuda.is_available() else "cpu")
    agent = RAE._build_agent(hydra_cfg, ckpt, dev, aux_protocol=False)
    agent.eval()

    logs_path = DATA_ROOT / "navsim_logs/test/test"
    if not logs_path.is_dir():
        logs_path = DATA_ROOT / "navsim_logs/test"
    sens_path = DATA_ROOT / "sensor_blobs/test/test"
    if not sens_path.is_dir():
        sens_path = DATA_ROOT / "sensor_blobs/test"
    syn_sens_path = DATA_ROOT / "navhard_two_stage/sensor_blobs"

    otoks = [t for t in todo if t not in synth]
    orig_loader = None
    if otoks:
        sf = SceneFilter(
            num_history_frames=4,
            num_future_frames=num_fut,
            frame_interval=1,
            has_route=True,
            max_scenes=None,
            log_names=logs,
            tokens=otoks,
        )
        orig_loader = SceneLoader(
            data_path=logs_path,
            sensor_blobs_path=sens_path,
            scene_filter=sf,
            sensor_config=agent.get_sensor_config(),
        )
    loader = CombinedSceneLoader(
        orig_loader,
        {t: synth[t] for t in todo if t in synth},
        agent.get_sensor_config(),
        syn_sens_path,
    )
    dl = torch.utils.data.DataLoader(
        DV.FeatDS(loader, agent.get_feature_builders(), todo),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=DV.collate,
        pin_memory=dev.type == "cuda",
    )
    t0, c = time.time(), 0
    for btok, feats, _ in dl:
        fg = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else v.dtype) for k, v in feats.items()}
        with torch.inference_mode():
            pr = agent(fg)
            if "ck2_cand96" not in pr:
                pr = dict(pr)
                pr.update(agent._ck_e2e2.infer(agent.ck_student, fg, pr))
        cand96 = pr["ck2_cand96"].detach().float().cpu().numpy()
        lat_traj = pr["ck2_lat_traj"].detach().float().cpu().numpy()
        v2_final = pr["ck2_v2_final"].detach().float().cpu().numpy()
        v2_im = pr["ck2_v2_im"].detach().float().cpu().numpy()
        v2_sim = pr["ck2_v2_sim"].detach().float().cpu().numpy()
        logit96 = pr["ck2_score_logit"].detach().float().cpu().numpy()
        valid96 = pr["ck2_valid96"].detach().bool().cpu().numpy()

        for i, t in enumerate(btok):
            rows[t] = {
                "cand96": cand96[i].astype(np.float32),
                "lat_traj": lat_traj[i].astype(np.float32),
                "v2_final": v2_final[i].astype(np.float32),
                "v2_im": v2_im[i].astype(np.float32),
                "v2_sim": v2_sim[i].astype(np.float32),
                "score_logit": logit96[i].astype(np.float32),
                "valid96": valid96[i],
            }
        c += len(btok)
        if c % 500 < len(btok) or c == len(todo):
            tmp = str(fp) + ".tmp"
            pickle.dump(rows, open(tmp, "wb"))
            os.replace(tmp, fp)
            log(f"[{split}] shard {shard}/{nshard}: {c}/{len(todo)} ({c / max(time.time() - t0, 1e-3):.2f} tok/s)")
    tmp = str(fp) + ".tmp"
    pickle.dump(rows, open(tmp, "wb"))
    os.replace(tmp, fp)
    return 0


def merge_pool(split: str, pool_dir: Path) -> None:
    rows: Dict[str, Dict[str, np.ndarray]] = {}
    for f in sorted(pool_dir.glob("pool96_part*.pkl")):
        rows.update(pickle.load(open(f, "rb")))
    orig, _, synth, _ = split_tokens(split, pool_dir.parent)
    toks = orig + sorted(synth)
    miss = set(toks) - set(rows)
    if miss:
        raise SystemExit(f"[{split}] merge incomplete: missing {len(miss)} of {len(toks)} tokens in {pool_dir}")
    json.dump(toks, open(pool_dir / "tokens.json", "w"))
    np.save(pool_dir / "cand96.npy", np.stack([rows[t]["cand96"] for t in toks], axis=0))
    np.save(pool_dir / "lat_traj.npy", np.stack([rows[t]["lat_traj"] for t in toks], axis=0))
    np.save(pool_dir / "v2_final.npy", np.stack([rows[t]["v2_final"] for t in toks], axis=0))
    np.save(pool_dir / "v2_im.npy", np.stack([rows[t]["v2_im"] for t in toks], axis=0))
    np.save(pool_dir / "v2_sim.npy", np.stack([rows[t]["v2_sim"] for t in toks], axis=0))
    np.save(pool_dir / "score_logit.npy", np.stack([rows[t]["score_logit"] for t in toks], axis=0))
    np.save(pool_dir / "valid96.npy", np.stack([rows[t]["valid96"] for t in toks], axis=0))
    log(f"[{split}] merged N={len(toks)} 96-candidate arrays in {pool_dir}")


def try_import_eval_e2e2_dump(run_dir: Path, pool_dir: Path, split: str) -> bool:
    """If eval_e2e2 already dumped packed/navtest + infer/<run>/navtest, link/copy into pool_dir."""
    if split != "navtest":
        return False
    pk = run_dir / "eval/root/packed/navtest"
    inf = run_dir / f"eval/root/infer/{run_dir.name}/navtest"
    if not ((pk / "tokens.parquet").is_file() and (inf / "lat_traj.npy").is_file() and (inf / "done.npy").is_file()):
        return False
    done = np.asarray(np.load(inf / "done.npy", mmap_mode="r"), bool)
    if not done.all():
        return False
    tdf = pd.read_parquet(pk / "tokens.parquet")
    toks = [str(t) for t in tdf.token.to_numpy()]
    pool_dir.mkdir(parents=True, exist_ok=True)
    json.dump(toks, open(pool_dir / "tokens.json", "w"))
    for k in ("cand96", "v2_final", "v2_im", "v2_sim", "valid96"):
        src, dst = pk / f"{k}.npy", pool_dir / f"{k}.npy"
        if not dst.exists():
            os.symlink(src.resolve(), dst)
    for k in ("lat_traj", "score_logit"):
        src, dst = inf / f"{k}.npy", pool_dir / f"{k}.npy"
        if not dst.exists():
            os.symlink(src.resolve(), dst)
    log(f"[navtest] imported existing eval_e2e2 dump (N={len(toks)}) -> {pool_dir}")
    return True


def stage_dump(
    split: str,
    run_dir: Path,
    out_dir: Path,
    tag: str,
    ckpt: Path,
    hydra_cfg: Path,
    gpus: List[str],
    batch_size: int,
    workers: int,
) -> Path:
    pool_dir = resolve_pool_dir(run_dir, out_dir, tag, split)
    if (pool_dir / "tokens.json").is_file() and (pool_dir / "lat_traj.npy").is_file():
        log(f"[{split}] reusing existing 96-candidate pool in {pool_dir}")
        return pool_dir
    if try_import_eval_e2e2_dump(run_dir, pool_dir, split):
        return pool_dir

    seed_nt = resolve_pool_dir(run_dir, out_dir, tag, "navtest") if split == "navhard" else None
    nshard = max(1, len(gpus))
    pool_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(nshard):
        g = gpus[i] if gpus else ""
        cmd = [
            PY,
            "-B",
            __file__,
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--split",
            split,
            "--stage",
            "dump",
            "--ckpt",
            str(ckpt),
            "--hydra-cfg",
            str(hydra_cfg),
            "--shard",
            str(i),
            "--nshard",
            str(nshard),
            "--batch-size",
            str(batch_size),
            "--workers-dump",
            str(workers),
        ]
        if seed_nt is not None and seed_nt.exists():
            cmd += ["--seed-pool-dir", str(seed_nt)]
        lf = open(pool_dir / f"part{i}.log", "a")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=g,
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CK_BEV64="1",
            CK_ALLOW_GPU4567="1",
            PYTHONPATH=CK,
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
        )
        procs.append((i, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env), lf))
        log(f"[{split}] launched dump shard {i}/{nshard} on GPU {g or 'cpu'} (pid {procs[-1][1].pid})")

    bad = []
    for i, p, lf in procs:
        rc = p.wait()
        lf.close()
        if rc != 0:
            bad.append((i, rc))
    if bad:
        raise SystemExit(f"[{split}] dump shards failed {bad} (see {pool_dir}/part*.log)")
    merge_pool(split, pool_dir)
    return pool_dir


def load_pool_arrays(pool_dir: Path):
    toks = json.loads((pool_dir / "tokens.json").read_text())
    cand96 = np.load(pool_dir / "cand96.npy").astype(np.float32)
    lat96 = np.load(pool_dir / "lat_traj.npy").astype(np.float32)
    v2_final = np.load(pool_dir / "v2_final.npy").astype(np.float64)
    v2_im = np.load(pool_dir / "v2_im.npy").astype(np.float64)
    v2_sim = np.load(pool_dir / "v2_sim.npy").astype(np.float64)
    logit96 = np.load(pool_dir / "score_logit.npy").astype(np.float64)
    valid96 = np.load(pool_dir / "valid96.npy").astype(bool)
    return toks, cand96, lat96, v2_final, v2_im, v2_sim, logit96, valid96


# ----------------------------------------------------------------------------------------------- kinematic & EPDMS eval
_KIN_MC_LOADER = None
_KIN_SIM = None
_TP_EC = np.arange(36, dtype=np.float64) * 0.1
_TP_41 = np.arange(41, dtype=np.float64) * 0.1
_EC_THRS = None

_MC_LOADER = None
_SIMULATOR = None
_SCORER_UNFILTERED = None
_POLICY = None
_HUMAN_ZERO_COLS: Dict[str, List[str]] = {}
_TRAJ_SAMPLING = None
_PROPOSAL_SAMPLING = None
_SKIP_COLS = {"multiplicative_metrics_prod", "weighted_metrics", "weighted_metrics_array", "pdm_score"}


def _ensure_navsim22() -> None:
    if sys.path[0] != NAVSIM22:
        sys.path.insert(0, NAVSIM22)
    for mod_name in [k for k in list(sys.modules) if k == "navsim" or k.startswith("navsim.")]:
        mod = sys.modules.get(mod_name)
        mod_file = getattr(mod, "__file__", "") or ""
        if mod_file and not mod_file.startswith(NAVSIM22):
            sys.modules.pop(mod_name, None)


def _init_kin_worker(mc_dir: str) -> None:
    global _KIN_MC_LOADER, _KIN_SIM, _EC_THRS
    _ensure_navsim22()
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        acceleration_threshold,
        jerk_threshold,
        yaw_accel_threshold,
        yaw_rate_threshold,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

    _KIN_SIM = PDMSimulator(TrajectorySampling(num_poses=40, interval_length=0.1))
    _KIN_MC_LOADER = MetricCacheLoader(Path(mc_dir))
    _EC_THRS = np.array(
        [acceleration_threshold, jerk_threshold, yaw_rate_threshold, yaw_accel_threshold],
        dtype=np.float64,
    )


def _compute_kin_token(item: Tuple[int, str, np.ndarray]):
    from nuplan.common.actor_state.state_representation import TimePoint
    from navsim.common.enums import SceneFrameType
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_comfortable,
        extract_features,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        ego_state_to_state_array,
        ego_states_to_state_array,
        state_array_to_coords_array,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex

    idx, tok, lat96 = item
    mc = _KIN_MC_LOADER.get_from_token(tok)
    start_s = float(mc.ego_state.time_point.time_s)
    log_name = str(mc.log_name)
    is_orig = bool(mc.scene_type == SceneFrameType.ORIGINAL)

    start_us = mc.ego_state.time_point.time_us
    step_us = int(0.5 * 1e6)
    knot_us = np.array([start_us + i * step_us for i in range(9)], dtype=np.int64)
    times_s = np.arange(0.0, 4.0 + 0.1, 0.1) + start_s
    times_us = np.clip([int(t * 1e6) for t in times_s], knot_us[0], knot_us[-1])
    seg = np.minimum(np.searchsorted(knot_us, times_us, side="right") - 1, 7)
    lam = (times_us - knot_us[seg]) / (knot_us[seg + 1] - knot_us[seg])

    init_arr = ego_state_to_state_array(mc.ego_state)
    x0, y0, h0 = init_arr[0], init_arr[1], init_arr[2]
    c0, s0 = np.cos(h0), np.sin(h0)
    p64 = lat96.astype(np.float64)
    gx = x0 + c0 * p64[..., 0] - s0 * p64[..., 1]
    gy = y0 + s0 * p64[..., 0] + c0 * p64[..., 1]
    gh = np.arctan2(np.sin(h0 + p64[..., 2]), np.cos(h0 + p64[..., 2]))

    X9 = np.concatenate([np.full((96, 1), x0), gx], axis=1)
    Y9 = np.concatenate([np.full((96, 1), y0), gy], axis=1)
    H9 = np.unwrap(np.concatenate([np.full((96, 1), h0), gh], axis=1), axis=1)

    fast = np.zeros((96, 41, 11), dtype=np.float64)
    fast[:, :, 0] = X9[:, seg] + (X9[:, seg + 1] - X9[:, seg]) * lam[None, :]
    fast[:, :, 1] = Y9[:, seg] + (Y9[:, seg + 1] - Y9[:, seg]) * lam[None, :]
    H_int = H9[:, seg] + (H9[:, seg + 1] - H9[:, seg]) * lam[None, :]
    fast[:, :, 2] = np.arctan2(np.sin(H_int), np.cos(H_int))

    sim_states = _KIN_SIM.simulate_proposals(fast, mc.ego_state)

    # 1. History Comfort (HC) from ego past trajectory + simulated candidate states
    hp = mc.past_human_trajectory
    if hp is not None:
        h_start, h_end = hp.start_time.time_us, hp.end_time.time_us
        h_us = np.clip(np.arange(h_start, h_end, int(0.1 * 1e6), dtype=np.int64), h_start, h_end)
        h_states = ego_states_to_state_array(hp.get_state_at_times([TimePoint(u) for u in h_us[:-1]]))
        padded = np.zeros((96, len(h_states) + 41, 11), dtype=np.float64)
        padded[:, : len(h_states)] = h_states
        padded[:, len(h_states) :] = sim_states
        tp_hc = np.arange(padded.shape[1], dtype=np.float64) * 0.1
        hc96 = ego_is_comfortable(padded, tp_hc).all(axis=-1).astype(np.float32)
    else:
        hc96 = ego_is_comfortable(sim_states, _TP_41).all(axis=-1).astype(np.float32)

    # 2. Stage-1 Corridor Consistency (CC): <=0.5m lateral deviation for <2.0s relative to Stage-1 top-1 trajectory c=0
    ego_coords = state_array_to_coords_array(sim_states, mc.ego_state.car_footprint.vehicle_parameters)
    centers = ego_coords[:, :, BBCoordsIndex.CENTER]
    base_line = shapely.LineString(centers[0])
    pts = shapely.points(centers[..., 0], centers[..., 1])
    dev_base = shapely.distance(pts, base_line)
    exc_base = (dev_base > 0.5).astype(np.int32)
    cs_base = np.concatenate([np.zeros((96, 1), dtype=np.int32), np.cumsum(exc_base, axis=1)], axis=1)
    cc96 = (~((cs_base[:, 20:] - cs_base[:, :-20]).max(axis=1) >= 20)).astype(np.float32)

    # 3. Extended Comfort (EC) kinematic profiles (3.5s overlap window)
    fn_dict = extract_features(sim_states[:, :-5], _TP_EC)
    fp_dict = extract_features(sim_states[:, 5:], _TP_EC)
    f_now = np.stack(
        [fn_dict["acceleration"], fn_dict["jerk"], fn_dict["yaw_rate"], fn_dict["yaw_accel"]],
        axis=1,
    ).astype(np.float32)
    f_prev = np.stack(
        [fp_dict["acceleration"], fp_dict["jerk"], fp_dict["yaw_rate"], fp_dict["yaw_accel"]],
        axis=1,
    ).astype(np.float32)

    f_full_dict = extract_features(sim_states, _TP_41)
    f_full = np.stack(
        [f_full_dict["acceleration"], f_full_dict["jerk"], f_full_dict["yaw_rate"], f_full_dict["yaw_accel"]],
        axis=1,
    ).astype(np.float64)
    rms_self = np.sqrt(np.mean((f_full - f_full[0:1]) ** 2, axis=-1))
    ec_self96 = np.all(rms_self <= _EC_THRS[None, :], axis=-1).astype(np.float32)

    return idx, hc96, cc96, ec_self96, f_now, f_prev, start_s, log_name, is_orig


def get_or_build_kin_cache(
    pool_dir: Path,
    kin_dir: Path,
    tag: str,
    split: str,
    mc_dir: str,
    workers: int = 48,
) -> Dict[str, Any]:
    kin_dir.mkdir(parents=True, exist_ok=True)
    cfile = kin_dir / f"kin_{split}_{tag}.npz"
    mfile = kin_dir / f"kin_{split}_{tag}_meta.json"
    if cfile.is_file() and mfile.is_file():
        log(f"[{split}] loading cached kinematic features from {cfile}")
        npz = np.load(cfile)
        meta = json.loads(mfile.read_text())
        cc_arr = npz["cc96"] if "cc96" in npz else npz["lk_traj96"]
        return {
            "hc96": npz["hc96"].astype(np.float64),
            "cc96": cc_arr.astype(np.float64),
            "ec_self96": npz["ec_self96"].astype(np.float64),
            "f_now": npz["f_now"],
            "f_prev": npz["f_prev"],
            "start_s": meta["start_s"],
            "log_name": meta["log_name"],
            "is_orig": meta["is_orig"],
        }

    toks, _, lat96, _, _, _, _, _ = load_pool_arrays(pool_dir)
    N = len(toks)
    log(f"[{split}] computing ego-kinematic HC/CC/EC features (N={N}) with {workers} workers ...")
    t0 = time.time()
    hc96 = np.zeros((N, 96), dtype=np.float32)
    cc96 = np.zeros((N, 96), dtype=np.float32)
    ec_self96 = np.zeros((N, 96), dtype=np.float32)
    f_now = np.zeros((N, 96, 4, 36), dtype=np.float32)
    f_prev = np.zeros((N, 96, 4, 36), dtype=np.float32)
    start_s = [0.0] * N
    log_name = [""] * N
    is_orig = [False] * N

    items = [(i, toks[i], lat96[i]) for i in range(N)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_kin_worker, initargs=(mc_dir,)) as ex:
        for idx, hc, cc, ecs, fn, fp, ss, ln, io in ex.map(_compute_kin_token, items, chunksize=16):
            hc96[idx] = hc
            cc96[idx] = cc
            ec_self96[idx] = ecs
            f_now[idx] = fn
            f_prev[idx] = fp
            start_s[idx] = ss
            log_name[idx] = ln
            is_orig[idx] = io

    log(f"[{split}] kinematic features done in {time.time() - t0:.1f}s -> {cfile}")
    np.savez(cfile, hc96=hc96, cc96=cc96, lk_traj96=cc96, ec_self96=ec_self96, f_now=f_now, f_prev=f_prev)
    mfile.write_text(json.dumps({"start_s": start_s, "log_name": log_name, "is_orig": is_orig}))
    return {
        "hc96": hc96.astype(np.float64),
        "cc96": cc96.astype(np.float64),
        "ec_self96": ec_self96.astype(np.float64),
        "f_now": f_now,
        "f_prev": f_prev,
        "start_s": start_s,
        "log_name": log_name,
        "is_orig": is_orig,
    }


def compute_ec_2f_pair(f_now_row: np.ndarray, f_prev_selected: np.ndarray, ec_thrs: np.ndarray) -> np.ndarray:
    diff = f_now_row.astype(np.float64) - f_prev_selected.astype(np.float64)[None, :, :]
    rms = np.sqrt(np.mean(diff * diff, axis=-1))
    return np.all(rms <= ec_thrs[None, :], axis=-1).astype(np.float64)


def select_epdms_trajectories(
    pool_dir: Path,
    split: str,
    kin: Dict[str, Any],
    all_mappings_nh: Optional[Dict[Tuple[str, str], List[Tuple[str, str]]]] = None,
    beta: float = 0.75,
    w_safe: float = 1.50,
    w_cc: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Select final trajectories over the 96-candidate pool using our Stage-1/Stage-2 EPDMS blend."""
    _ensure_navsim22()
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        acceleration_threshold,
        jerk_threshold,
        yaw_accel_threshold,
        yaw_rate_threshold,
    )

    ec_thrs = np.array(
        [acceleration_threshold, jerk_threshold, yaw_rate_threshold, yaw_accel_threshold],
        dtype=np.float64,
    )
    toks, _, lat96, v2_final, _, _, logit96, valid96 = load_pool_arrays(pool_dir)
    N = len(toks)
    ar = np.arange(N)
    tok2idx = {t: i for i, t in enumerate(toks)}
    vtype = np.arange(96) % 6
    ok_all = valid96 | (vtype[None, :] == 0)

    prob96 = 1.0 / (1.0 + np.exp(-logit96))
    v2f96 = np.repeat(v2_final, 6, axis=1)
    eps = 1e-6
    nc_c, dac_c, ep_c, ttc_c, _ = [prob96[..., i] for i in range(5)]

    hc_k = kin["hc96"]
    cc_k = kin["cc96"]
    ec_self = kin["ec_self96"]
    f_now = kin["f_now"]
    f_prev = kin["f_prev"]

    prev_of: Dict[int, int] = {}
    order: List[int] = []
    if split == "navtest":
        df_meta = pd.DataFrame(
            {
                "idx": np.arange(N),
                "log": kin["log_name"],
                "start_s": kin["start_s"],
                "is_orig": kin["is_orig"],
            }
        )
        for _, g in df_meta.groupby("log"):
            g = g.sort_values("start_s").reset_index(drop=True)
            idxs = g["idx"].to_numpy()
            ss = g["start_s"].to_numpy()
            orig = g["is_orig"].to_numpy()
            for j in range(len(idxs)):
                order.append(int(idxs[j]))
                if j > 0 and orig[j] and orig[j - 1] and abs(ss[j] - ss[j - 1]) <= 0.55:
                    prev_of[int(idxs[j])] = int(idxs[j - 1])
    else:
        seen = set()
        for (now_f, prev_f), second_stage in (all_mappings_nh or {}).items():
            ip, inow = tok2idx[prev_f], tok2idx[now_f]
            if ip not in seen:
                order.append(ip)
                seen.add(ip)
            if inow not in seen:
                order.append(inow)
                seen.add(inow)
            prev_of[inow] = ip
            for now_t, prev_t in second_stage:
                ip2, inow2 = tok2idx[prev_t], tok2idx[now_t]
                if ip2 not in seen:
                    order.append(ip2)
                    seen.add(ip2)
                if inow2 not in seen:
                    order.append(inow2)
                    seen.add(inow2)
                prev_of[inow2] = ip2
        for i in range(N):
            if i not in seen:
                order.append(i)

    sel_idx = np.zeros(N, dtype=np.int64)
    for i in order:
        if i in prev_of:
            cp = sel_idx[prev_of[i]]
            ec_row = compute_ec_2f_pair(f_now[i], f_prev[prev_of[i], cp], ec_thrs)
        else:
            ec_row = ec_self[i]
        add_term = 5.0 * ep_c[i] + 5.0 * ttc_c[i] + 2.0 * hc_k[i] + w_cc * cc_k[i] + 2.0 * ec_row
        s2_score = w_safe * (np.log(nc_c[i] + eps) + np.log(dac_c[i] + eps)) + np.log(add_term + eps)
        s_row = (1.0 - beta) * v2f96[i] + beta * s2_score
        sel_idx[i] = pick_1row(s_row, ok_all[i])

    chosen = lat96[ar, sel_idx]
    return {toks[i]: chosen[i] for i in range(N)}


def select_pdms_trajectories(
    pool_dir: Path,
    pdms_beta: float = 1.0,
    pdms_set: str = "all",
    pdms_lat_mode: str = "on",
) -> Dict[str, np.ndarray]:
    """Select final trajectories over the 96-candidate pool using the official PDMS formula without imitation:
    PDMS(p) = NC * DAC * (5 * EP + 5 * TTC + 2 * C) / 12  (select2.select96_pdms when beta=1.0, or w='plugin' when beta<1.0).
    """
    from navsim.agents.para_ssr.ck import select2 as S2

    toks, cand96, lat96, v2_final, v2_im, _, logit96, valid96 = load_pool_arrays(pool_dir)
    types = S2.VARIANT_SETS[pdms_set]
    if pdms_beta == 1.0:
        sel_idx = S2.select96_pdms(logit96, valid96, types=types)
    else:
        sel_idx = S2.select96(v2_final, v2_im, logit96, valid96, beta=pdms_beta, types=types, w="plugin")
    chosen, _ = S2.final_traj(cand96, lat96, sel_idx, pdms_lat_mode)
    return {toks[i]: chosen[i] for i in range(len(toks))}


def _init_epdms_worker(mc_dir: str, split: str) -> None:
    global _MC_LOADER, _SIMULATOR, _SCORER_UNFILTERED, _POLICY, _HUMAN_ZERO_COLS, _TRAJ_SAMPLING, _PROPOSAL_SAMPLING
    _ensure_navsim22()
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
    from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents

    _TRAJ_SAMPLING = TrajectorySampling(time_horizon=4, interval_length=0.5)
    _PROPOSAL_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)
    _SIMULATOR = PDMSimulator(proposal_sampling=_PROPOSAL_SAMPLING)
    _SCORER_UNFILTERED = PDMScorer(
        proposal_sampling=_PROPOSAL_SAMPLING,
        config=PDMScorerConfig(
            progress_weight=5.0,
            ttc_weight=5.0,
            lane_keeping_weight=2.0,
            history_comfort_weight=2.0,
            two_frame_extended_comfort_weight=2.0,
            driving_direction_horizon=1.0,
            driving_direction_compliance_threshold=2.0,
            driving_direction_violation_threshold=6.0,
            stopped_speed_threshold=0.005,
            future_collision_horizon_window=1.0,
            progress_distance_threshold=5.0,
            lane_keeping_deviation_limit=0.5,
            lane_keeping_horizon_window=2.0,
            human_penalty_filter=False,
        ),
    )
    if split == "navtest":
        _POLICY = LogReplayTrafficAgents(_PROPOSAL_SAMPLING)
    else:
        _POLICY = NavsimIDMTrafficAgents(
            future_trajectory_sampling=_PROPOSAL_SAMPLING,
            idm_agents_observation=NavsimIDMAgents(
                target_velocity=10,
                min_gap_to_lead_agent=1.0,
                headway_time=1.5,
                accel_max=1.0,
                decel_max=2.0,
                open_loop_detections_types=[],
                minimum_path_length=20,
                planned_trajectory_samples=None,
                planned_trajectory_sample_interval=None,
                radius=100,
                add_open_loop_parked_vehicles=True,
                idm_snap_threshold=3.0,
            ),
        )
    _MC_LOADER = MetricCacheLoader(Path(mc_dir))
    _HUMAN_ZERO_COLS = {}


def _get_human_zero_cols(token: str, metric_cache) -> List[str]:
    from navsim.common.enums import SceneFrameType
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory

    if token in _HUMAN_ZERO_COLS:
        return _HUMAN_ZERO_COLS[token]
    if metric_cache.scene_type != SceneFrameType.ORIGINAL:
        _HUMAN_ZERO_COLS[token] = []
        return []
    initial_ego_state = metric_cache.ego_state
    human_trajectory = transform_trajectory(metric_cache.human_trajectory, initial_ego_state)
    human_states = get_trajectory_as_array(human_trajectory, _PROPOSAL_SAMPLING, initial_ego_state.time_point)
    human_simulated_states = _SIMULATOR.simulate_proposals(human_states[None, ...], initial_ego_state)
    human_sim_tracks = _POLICY.simulate_environment(human_simulated_states[0], metric_cache)
    human_pdm_result = _SCORER_UNFILTERED.score_proposals(
        human_simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        human_sim_tracks,
        metric_cache.past_human_trajectory,
    )[0]
    zcols = [c for c in human_pdm_result.columns if c not in _SKIP_COLS and human_pdm_result[c].iloc[0] == 0]
    _HUMAN_ZERO_COLS[token] = zcols
    return zcols


def _score_one_item(item: Tuple[str, np.ndarray]) -> Dict[str, Any]:
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses
    from navsim.common.dataclasses import PDMResults, Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex

    token, poses = item
    try:
        metric_cache = _MC_LOADER.get_from_token(token)
        poses = np.asarray(poses, dtype=np.float32).reshape(-1, 3)
        trajectory = Trajectory(poses, _TRAJ_SAMPLING)

        initial_ego_state = metric_cache.ego_state
        pred_trajectory = transform_trajectory(trajectory, initial_ego_state)
        pdm_states = get_trajectory_as_array(metric_cache.trajectory, _PROPOSAL_SAMPLING, initial_ego_state.time_point)
        pred_states = get_trajectory_as_array(pred_trajectory, _PROPOSAL_SAMPLING, initial_ego_state.time_point)
        trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)
        simulated_states = _SIMULATOR.simulate_proposals(trajectory_states, initial_ego_state)
        simulated_tracks = _POLICY.simulate_environment(simulated_states[1], metric_cache)

        pdm_result = _SCORER_UNFILTERED.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            simulated_tracks,
            metric_cache.past_human_trajectory,
        )[1]

        bef_mult = float(pdm_result.at[0, "multiplicative_metrics_prod"])
        bef_w_metrics = pdm_result.at[0, "weighted_metrics"].copy()
        bef_nc = float(pdm_result.at[0, "no_at_fault_collisions"])
        bef_dac = float(pdm_result.at[0, "drivable_area_compliance"])
        bef_ddc = float(pdm_result.at[0, "driving_direction_compliance"])

        zcols = _get_human_zero_cols(token, metric_cache)
        if zcols:
            for col in zcols:
                pdm_result.at[0, col] = 1
            pdm_result.at[0, "multiplicative_metrics_prod"] = (
                pdm_result.at[0, "no_at_fault_collisions"]
                * pdm_result.at[0, "drivable_area_compliance"]
                * pdm_result.at[0, "driving_direction_compliance"]
                * pdm_result.at[0, "traffic_light_compliance"]
            )
            wm = pdm_result.at[0, "weighted_metrics"].copy()
            wm[WeightedMetricIndex.PROGRESS] = pdm_result.at[0, "ego_progress"]
            wm[WeightedMetricIndex.TTC] = pdm_result.at[0, "time_to_collision_within_bound"]
            wm[WeightedMetricIndex.LANE_KEEPING] = pdm_result.at[0, "lane_keeping"]
            wm[WeightedMetricIndex.HISTORY_COMFORT] = pdm_result.at[0, "history_comfort"]
            pdm_result.at[0, "weighted_metrics"] = wm

        d = pdm_result.iloc[0].to_dict()
        d["bef_multiplicative_metrics_prod"] = bef_mult
        d["bef_weighted_metrics"] = bef_w_metrics
        d["bef_nc"] = bef_nc
        d["bef_dac"] = bef_dac
        d["bef_ddc"] = bef_ddc
        d["valid"] = True
        d["log_name"] = metric_cache.log_name
        d["frame_type"] = metric_cache.scene_type
        d["start_time"] = metric_cache.timepoint.time_s
        end_pose = StateSE2(
            x=trajectory.poses[-1, 0],
            y=trajectory.poses[-1, 1],
            heading=trajectory.poses[-1, 2],
        )
        absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
        d["endpoint_x"] = absolute_endpoint.x
        d["endpoint_y"] = absolute_endpoint.y
        d["start_point_x"] = metric_cache.ego_state.rear_axle.x
        d["start_point_y"] = metric_cache.ego_state.rear_axle.y
        d["ego_simulated_states"] = simulated_states[1]
    except Exception:
        traceback.print_exc()
        d = PDMResults.get_empty_results().__dict__.copy()
        d["valid"] = False
    d["token"] = token
    return d


def _fast_two_frame_ec(
    now_tok: str,
    now_key: Tuple[str, bytes],
    prev_tok: str,
    prev_key: Tuple[str, bytes],
    row_cache: Dict[Tuple[str, bytes], Dict[str, Any]],
    ec_cache: Dict[Tuple[str, bytes, str, bytes], float],
) -> float:
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_two_frame_extended_comfort,
    )

    ck = (now_tok, now_key[1], prev_tok, prev_key[1])
    if ck in ec_cache:
        return ec_cache[ck]
    r_now = row_cache[now_key]
    r_prev = row_cache[prev_key]
    c_states = r_now["ego_simulated_states"]
    p_states = r_prev["ego_simulated_states"]
    interval_length = 0.1
    obs_interval = r_now["start_time"] - r_prev["start_time"]
    overlap_start = round(obs_interval / interval_length)
    c_ov = c_states[:-overlap_start]
    p_ov = p_states[overlap_start:]
    tp_s = np.arange(c_ov.shape[0]) * interval_length
    val = float(ego_is_two_frame_extended_comfort(c_ov[None, :], p_ov[None, :], tp_s)[0].astype(np.float64))
    ec_cache[ck] = val
    return val


def evaluate_trajectories_navtest(
    traj_dict: Dict[str, np.ndarray],
    all_tokens: List[str],
    mc_dir: str,
    workers: int,
) -> Dict[str, Any]:
    _ensure_navsim22()
    from navsim.common.dataclasses import PDMResults
    from navsim.planning.script.run_pdm_score_one_stage import (
        compute_final_scores as compute_final_scores_one_stage,
        infer_start_adjacent_mapping,
    )

    t0 = time.time()
    todo_items = [(tok, np.ascontiguousarray(traj_dict[tok], dtype=np.float32).reshape(-1, 3)) for tok in all_tokens]
    keys_of = {tok: (tok, poses.tobytes()) for tok, poses in todo_items}
    row_cache: Dict[Tuple[str, bytes], Dict[str, Any]] = {}
    ec_cache: Dict[Tuple[str, bytes, str, bytes], float] = {}

    cs = max(1, min(32, len(todo_items) // (workers * 2)))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_epdms_worker, initargs=(mc_dir, "navtest")) as ex:
        for (tok, poses), r in zip(todo_items, ex.map(_score_one_item, todo_items, chunksize=cs)):
            row_cache[(tok, poses.tobytes())] = r

    meta_rows = [
        {
            "token": t,
            "log_name": row_cache[keys_of[t]]["log_name"],
            "frame_type": row_cache[keys_of[t]]["frame_type"],
            "start_time": row_cache[keys_of[t]]["start_time"],
        }
        for t in all_tokens
    ]
    start_adj = infer_start_adjacent_mapping(pd.DataFrame(meta_rows))
    ec_map = {
        now_f: _fast_two_frame_ec(now_f, keys_of[now_f], prev_f, keys_of[prev_f], row_cache, ec_cache)
        for now_f, prev_f in start_adj.items()
    }

    rows = []
    for tok in all_tokens:
        d = dict(row_cache[keys_of[tok]])
        d.pop("ego_simulated_states", None)
        d["two_frame_extended_comfort"] = ec_map.get(tok, np.nan)
        rows.append(d)
    agg_df = pd.DataFrame(rows)

    df_after = compute_final_scores_one_stage(
        agg_df.drop(columns=["bef_multiplicative_metrics_prod", "bef_weighted_metrics"], errors="ignore").copy()
    )
    score_cols = [
        c
        for c in df_after.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]
    avg_aft = {k: float(v) * 100.0 for k, v in df_after[score_cols].mean(skipna=True).items()}

    df_bef = agg_df.copy()
    df_bef["multiplicative_metrics_prod"] = df_bef["bef_multiplicative_metrics_prod"]
    df_bef["weighted_metrics"] = [x.copy() for x in df_bef["bef_weighted_metrics"]]
    df_bef.drop(columns=["bef_multiplicative_metrics_prod", "bef_weighted_metrics"], inplace=True, errors="ignore")
    df_bef = compute_final_scores_one_stage(df_bef)
    avg_bef = {k: float(v) * 100.0 for k, v in df_bef[score_cols].mean(skipna=True).items()}

    return {
        "after": avg_aft,
        "before": avg_bef,
        "n_tokens": len(all_tokens),
        "sec": round(time.time() - t0, 2),
    }


def evaluate_trajectories_navhard(
    traj_dict: Dict[str, np.ndarray],
    all_tokens: List[str],
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
    mc_dir: str,
    workers: int,
) -> Dict[str, Any]:
    _ensure_navsim22()
    from navsim.common.dataclasses import PDMResults
    from navsim.planning.script.run_pdm_score import (
        compute_final_scores as compute_final_scores_two_stage,
    )

    t0 = time.time()
    todo_items = [(tok, np.ascontiguousarray(traj_dict[tok], dtype=np.float32).reshape(-1, 3)) for tok in all_tokens]
    keys_of = {tok: (tok, poses.tobytes()) for tok, poses in todo_items}
    row_cache: Dict[Tuple[str, bytes], Dict[str, Any]] = {}
    ec_cache: Dict[Tuple[str, bytes, str, bytes], float] = {}

    cs = max(1, min(16, len(todo_items) // (workers * 2)))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_epdms_worker, initargs=(mc_dir, "navhard")) as ex:
        for (tok, poses), r in zip(todo_items, ex.map(_score_one_item, todo_items, chunksize=cs)):
            row_cache[(tok, poses.tobytes())] = r

    ec_map: Dict[str, float] = {}
    w_map: Dict[str, float] = {}
    sigma2 = 0.1
    for (now_f, prev_f), second_stage in all_mappings.items():
        main_ec = _fast_two_frame_ec(now_f, keys_of[now_f], prev_f, keys_of[prev_f], row_cache, ec_cache)
        ec_map[now_f] = main_ec
        ec_map[prev_f] = main_ec
        w_map[now_f] = 1.0
        w_map[prev_f] = 1.0

        r_now = row_cache[keys_of[now_f]]
        r_prev = row_cache[keys_of[prev_f]]
        s2_now_toks = [p[0] for p in second_stage]
        s2_prev_toks = [p[1] for p in second_stage]

        for f_row, s2_toks in [(r_now, s2_now_toks), (r_prev, s2_prev_toks)]:
            x1, y1 = f_row["endpoint_x"], f_row["endpoint_y"]
            d2 = np.array(
                [
                    (x1 - row_cache[keys_of[t]]["start_point_x"]) ** 2
                    + (y1 - row_cache[keys_of[t]]["start_point_y"]) ** 2
                    for t in s2_toks
                ],
                dtype=np.float64,
            )
            w = np.exp(-d2 / (2.0 * sigma2))
            w_sum = w.sum()
            if np.isclose(w_sum, 0.0) or np.isnan(w_sum):
                w = np.full(len(s2_toks), 1.0 / len(s2_toks))
            else:
                w = w / w_sum
            for t, wt in zip(s2_toks, w):
                w_map[t] = float(wt)

        for now_t, prev_t in second_stage:
            ec2 = _fast_two_frame_ec(now_t, keys_of[now_t], prev_t, keys_of[prev_t], row_cache, ec_cache)
            ec_map[now_t] = ec2
            ec_map[prev_t] = ec2

    rows = []
    for tok in all_tokens:
        d = dict(row_cache[keys_of[tok]])
        d.pop("ego_simulated_states", None)
        d["two_frame_extended_comfort"] = ec_map.get(tok, np.nan)
        d["weight"] = w_map.get(tok, np.nan)
        rows.append(d)
    agg_df = pd.DataFrame(rows)

    df_after = compute_final_scores_two_stage(
        agg_df.drop(columns=["bef_multiplicative_metrics_prod", "bef_weighted_metrics"], errors="ignore").copy()
    )
    score_cols = [
        c
        for c in df_after.columns
        if (
            (any(score.name in c for score in fields(PDMResults)) or c == "two_frame_extended_comfort" or c == "score")
            and c != "pdm_score"
        )
    ]

    def fast_mapping_scores(df_sc: pd.DataFrame):
        idx_df = df_sc.set_index("token")
        mat = idx_df[score_cols].to_numpy(dtype=np.float64)
        wts = idx_df["weight"].to_numpy(dtype=np.float64)
        t2i = {t: i for i, t in enumerate(idx_df.index)}

        g_list, s1_list, s2_list = [], [], []
        for (orig_t, prev_t), s2_pairs in all_mappings.items():
            f_idx = [t2i[p[0]] for p in s2_pairs if len(p) > 0]
            s_idx = [t2i[p[1]] for p in s2_pairs if len(p) > 1]
            g1_s1 = mat[t2i[orig_t]]
            g2_s1 = mat[t2i[prev_t]]
            w1 = wts[f_idx]
            g1_s2 = (mat[f_idx] * w1[:, None]).sum(axis=0) / w1.sum()
            w2 = wts[s_idx]
            g2_s2 = (mat[s_idx] * w2[:, None]).sum(axis=0) / w2.sum()
            s1_list.append(g1_s1)
            s1_list.append(g2_s1)
            s2_list.append(g1_s2)
            s2_list.append(g2_s2)
            g_list.append(0.5 * (g1_s1 * g1_s2 + g2_s1 * g2_s2))
        return (
            dict(zip(score_cols, (np.mean(g_list, axis=0) * 100.0).tolist())),
            dict(zip(score_cols, (np.mean(s1_list, axis=0) * 100.0).tolist())),
            dict(zip(score_cols, (np.mean(s2_list, axis=0) * 100.0).tolist())),
        )

    g_aft, s1_aft, s2_aft = fast_mapping_scores(df_after)
    df_bef = agg_df.copy()
    df_bef["multiplicative_metrics_prod"] = df_bef["bef_multiplicative_metrics_prod"]
    df_bef["weighted_metrics"] = [x.copy() for x in df_bef["bef_weighted_metrics"]]
    df_bef.drop(columns=["bef_multiplicative_metrics_prod", "bef_weighted_metrics"], inplace=True, errors="ignore")
    df_bef = compute_final_scores_two_stage(df_bef)
    g_bef, s1_bef, s2_bef = fast_mapping_scores(df_bef)

    return {
        "after": {"combined": g_aft, "stage1": s1_aft, "stage2": s2_aft},
        "before": {"combined": g_bef, "stage1": s1_bef, "stage2": s2_bef},
        "n_tokens": len(all_tokens),
        "sec": round(time.time() - t0, 2),
    }


def _fmt_metrics(m: Dict[str, float]) -> str:
    return (
        f"EPDMS={m['score']:5.2f}% | "
        f"NC={m['no_at_fault_collisions']:5.2f} DAC={m['drivable_area_compliance']:5.2f} "
        f"DDC={m['driving_direction_compliance']:5.2f} TLC={m['traffic_light_compliance']:5.2f} | "
        f"EP={m['ego_progress']:5.2f} TTC={m['time_to_collision_within_bound']:5.2f} "
        f"LK={m['lane_keeping']:5.2f} HC={m['history_comfort']:5.2f} "
        f"EC={m.get('two_frame_extended_comfort', 0.0):5.2f}"
    )


def _fmt_pdms_metrics(m: Dict[str, float]) -> str:
    return (
        f"PDMS={m['pdms']:5.2f}% (v1.1={m['pdms_v11']:5.2f}%) | "
        f"NC={m['nc']:5.2f} DAC={m['dac']:5.2f} DDC={m['ddc']:5.2f} | "
        f"EP={m['ep']:5.2f} TTC={m['ttc']:5.2f} C={m['comfort']:5.2f}"
    )


def evaluate_trajectories_pdms_v1(
    traj_dict: Dict[str, np.ndarray],
    all_tokens: List[str],
    mc_dir: str,
    workers: int,
) -> Dict[str, Any]:
    """Evaluate official NAVSIM v1 PDMS on navtest (NC * DAC * DDC * (5*EP + 5*TTC + 2*C) / 12)."""
    _ensure_navsim22()
    t0 = time.time()
    todo_items = [(tok, np.ascontiguousarray(traj_dict[tok], dtype=np.float32).reshape(-1, 3)) for tok in all_tokens]
    cs = max(1, min(32, len(todo_items) // (workers * 2)))
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_epdms_worker, initargs=(mc_dir, "navtest")) as ex:
        for _, r in zip(todo_items, ex.map(_score_one_item, todo_items, chunksize=cs)):
            wm = r["bef_weighted_metrics"]
            nc = float(r.get("bef_nc", r["no_at_fault_collisions"]))
            dac = float(r.get("bef_dac", r["drivable_area_compliance"]))
            ddc = float(r.get("bef_ddc", r["driving_direction_compliance"]))
            ep = float(wm[0])
            ttc = float(wm[1])
            cmf = float(wm[3])
            w_avg = (5.0 * ep + 5.0 * ttc + 2.0 * cmf) / 12.0
            pdms = nc * dac * ddc * w_avg
            pdms_v11 = nc * dac * w_avg
            rows.append({"nc": nc, "dac": dac, "ddc": ddc, "ep": ep, "ttc": ttc, "comfort": cmf,
                         "pdms": pdms, "pdms_v11": pdms_v11})
    df = pd.DataFrame(rows)
    avg = {k: float(v) * 100.0 for k, v in df.mean(skipna=True).items()}
    return {
        "metrics": avg,
        "n_tokens": len(all_tokens),
        "sec": round(time.time() - t0, 2),
    }


def stage_eval(
    split: str,
    run_dir: Path,
    out_dir: Path,
    tag: str,
    ckpt: Path,
    beta: float,
    w_safe: float,
    w_cc: float,
    workers: int,
    mc_navtest: str,
    mc_navhard: str,
    metric: str = "epdms",
    pdms_beta: float = 1.0,
    pdms_set: str = "all",
    pdms_lat_mode: str = "on",
) -> Path:
    pool_dir = resolve_pool_dir(run_dir, out_dir, tag, split)
    if not ((pool_dir / "tokens.json").is_file() and (pool_dir / "lat_traj.npy").is_file()):
        raise SystemExit(f"[{split}] 96-candidate pool missing in {pool_dir}: run --stage dump first")
    kin_dir = resolve_kin_dir(run_dir, out_dir, tag, split)
    res_dir = out_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    sf_dir = Path(NAVSIM22) / "navsim/planning/script/config/common/train_test_split"
    if split == "navtest" and metric in ("pdms", "all"):
        sf_nt = _load_yaml(sf_dir / "scene_filter/navtest.yaml")
        all_tokens_nt = [str(t) for t in sf_nt["tokens"]]
        tdict_pdms = select_pdms_trajectories(
            pool_dir, pdms_beta=pdms_beta, pdms_set=pdms_set, pdms_lat_mode=pdms_lat_mode
        )
        res_pdms = evaluate_trajectories_pdms_v1(tdict_pdms, all_tokens_nt, mc_navtest, workers=workers)
        log(f"[navtest {tag}] Official PDMS (no-im, beta={pdms_beta}, set={pdms_set}, lat={pdms_lat_mode}) "
            f"{_fmt_pdms_metrics(res_pdms['metrics'])}")
        pdms_path = res_dir / f"pdms_navtest_{tag}.json"
        pdms_payload = {
            "split": "navtest",
            "tag": tag,
            "ckpt": str(ckpt),
            "selector": {
                "rule": "pdms",
                "formula": "NC * DAC * (5 * EP + 5 * TTC + 2 * C) / 12",
                "imitation": False,
                "beta": pdms_beta,
                "set": pdms_set,
                "lat_mode": pdms_lat_mode,
            },
            "created": kst(),
            **res_pdms,
        }
        pdms_path.write_text(json.dumps(pdms_payload, indent=2))
        log(f"[navtest] saved Official PDMS evaluation results -> {pdms_path}")
        if metric == "pdms":
            return pdms_path

    res_path = res_dir / f"epdms_{split}_{tag}.json"
    if split == "navtest":
        kin = get_or_build_kin_cache(pool_dir, kin_dir, tag, "navtest", mc_navtest, workers=workers)
        tdict = select_epdms_trajectories(pool_dir, "navtest", kin, None, beta=beta, w_safe=w_safe, w_cc=w_cc)
        sf_nt = _load_yaml(sf_dir / "scene_filter/navtest.yaml")
        all_tokens_nt = [str(t) for t in sf_nt["tokens"]]
        res = evaluate_trajectories_navtest(tdict, all_tokens_nt, mc_navtest, workers=workers)
        log(f"[navtest {tag}] (After)  {_fmt_metrics(res['after'])}")
        log(f"[navtest {tag}] (Before) {_fmt_metrics(res['before'])}")
    elif split == "navhard":
        sf_raw = _load_yaml(sf_dir / "scene_filter/navhard_two_stage.yaml")
        split_raw = _load_yaml(sf_dir / "navhard_two_stage.yaml")
        all_tokens_nh = [str(t) for t in sf_raw["tokens"]] + [str(t) for t in sf_raw["reactive_synthetic_initial_tokens"]]
        all_mappings_nh: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        token_set_nh = set(all_tokens_nh)
        for orig_token, prev_token, two_stage_pairs in split_raw["reactive_all_mapping"]:
            if prev_token in token_set_nh or orig_token in token_set_nh:
                all_mappings_nh[(str(orig_token), str(prev_token))] = [(str(p[0]), str(p[1])) for p in two_stage_pairs]

        kin = get_or_build_kin_cache(pool_dir, kin_dir, tag, "navhard", mc_navhard, workers=workers)
        tdict = select_epdms_trajectories(
            pool_dir, "navhard", kin, all_mappings_nh, beta=beta, w_safe=w_safe, w_cc=w_cc
        )
        res = evaluate_trajectories_navhard(tdict, all_tokens_nh, all_mappings_nh, mc_navhard, workers=workers)
        log(f"[navhard {tag}] Combined (After)  {_fmt_metrics(res['after']['combined'])}")
        log(f"[navhard {tag}] Stage 1  (After)  {_fmt_metrics(res['after']['stage1'])}")
        log(f"[navhard {tag}] Stage 2  (After)  {_fmt_metrics(res['after']['stage2'])}")
        log(f"[navhard {tag}] Combined (Before) {_fmt_metrics(res['before']['combined'])}")
    else:
        raise ValueError(split)

    payload = {
        "split": split,
        "tag": tag,
        "ckpt": str(ckpt),
        "selector": {"beta": beta, "w_safe": w_safe, "w_cc": w_cc},
        "created": kst(),
        **res,
    }
    res_path.write_text(json.dumps(payload, indent=2))
    log(f"[{split}] saved EPDMS evaluation results -> {res_path}")
    return res_path


def stage_summary(out_dir: Path, tag: str) -> Path:
    res_dir = out_dir / "results"
    summary: Dict[str, Any] = {"tag": tag, "created": kst(), "splits": {}}
    lines = [
        f"# Evaluation Summary (`{tag}`)",
        "",
        f"- Generated: `{summary['created']}`",
        "",
    ]

    pdms_rp = res_dir / f"pdms_navtest_{tag}.json"
    if pdms_rp.is_file():
        pdata = json.loads(pdms_rp.read_text())
        summary["pdms_navtest"] = pdata
        pm = pdata["metrics"]
        psel = pdata.get("selector", {})
        lines += [
            "## Official NAVSIM v1 PDMS (`navtest`, No-Imitation Official PDMS Selector)",
            "",
            f"- Selector: `rule={psel.get('rule', 'pdms')}, beta={psel.get('beta', 1.0)}, "
            f"set={psel.get('set', 'all')}, lat_mode={psel.get('lat_mode', 'on')}`",
            "",
            "| Split | NC | DAC | DDC | EP | TTC | C | **PDMS** | **PDMS (v1.1)** |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| `navtest` | {pm['nc']:.2f} | {pm['dac']:.2f} | {pm['ddc']:.2f} | "
            f"{pm['ep']:.2f} | {pm['ttc']:.2f} | {pm['comfort']:.2f} | "
            f"**{pm['pdms']:.2f}%** | **{pm['pdms_v11']:.2f}%** |",
            "",
        ]

    lines += [
        "## Official NAVSIM v2.2 EPDMS",
        "",
        "| Split | Filter / Stage | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | **EPDMS** |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def _row_md(split_label: str, stage_label: str, m: Dict[str, float]) -> str:
        return (
            f"| `{split_label}` | {stage_label} | "
            f"{m['no_at_fault_collisions']:.2f} | {m['drivable_area_compliance']:.2f} | "
            f"{m['driving_direction_compliance']:.2f} | {m['traffic_light_compliance']:.2f} | "
            f"{m['ego_progress']:.2f} | {m['time_to_collision_within_bound']:.2f} | "
            f"{m['lane_keeping']:.2f} | {m['history_comfort']:.2f} | "
            f"{m.get('two_frame_extended_comfort', 0.0):.2f} | **{m['score']:.2f}%** |"
        )

    for sp in SPLITS:
        rp = res_dir / f"epdms_{sp}_{tag}.json"
        if not rp.is_file():
            continue
        data = json.loads(rp.read_text())
        summary["splits"][sp] = data
        if sp == "navtest":
            lines.append(_row_md("navtest", "After (Official)", data["after"]))
            lines.append(_row_md("navtest", "Before (Unfiltered)", data["before"]))
        else:
            lines.append(_row_md("navhard_two_stage", "Combined (After)", data["after"]["combined"]))
            lines.append(_row_md("navhard_two_stage", "Stage 1 (After)", data["after"]["stage1"]))
            lines.append(_row_md("navhard_two_stage", "Stage 2 (After)", data["after"]["stage2"]))
            lines.append(_row_md("navhard_two_stage", "Combined (Before)", data["before"]["combined"]))

    lines.append("")
    s_json = out_dir / f"summary_{tag}.json"
    s_md = out_dir / f"summary_{tag}.md"
    s_json.write_text(json.dumps(summary, indent=2))
    s_md.write_text("\n".join(lines))
    log(f"summary -> {s_md}")
    return s_md


def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--run-dir", default="", help="Training run directory or checkpoint directory")
    ap.add_argument("--out-dir", default="", help="Output directory (default: <run-dir>/eval_epdms)")
    ap.add_argument("--split", required=True, choices=["navtest", "navhard", "navhard_two_stage", "all"])
    ap.add_argument("--metric", default="epdms", choices=list(METRICS),
                    help="Metric to evaluate: epdms (default, Option B+) | pdms (official PDMS without imitation on navtest) | all")
    ap.add_argument("--ckpt", default="last", help="last (default) | epoch:<E> | ep<E+1> | <path>")
    ap.add_argument("--hydra-cfg", default="", help="Optional path to training hydra/config.yaml")
    ap.add_argument("--stage", default="all", help="Comma-separated stages: dump,eval,summary or all")
    ap.add_argument("--gpus", default="0,1,2,3", help="Comma-separated GPUs for dump (e.g. 0,1,2,3 or 0,1,2,3,4,5,6,7)")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size per dump GPU shard")
    ap.add_argument("--workers-dump", type=int, default=2, help="DataLoader workers per dump GPU shard")
    ap.add_argument("--workers", type=int, default=48, help="CPU worker processes for kinematic & EPDMS scoring")
    ap.add_argument("--beta", type=float, default=0.75, help="Stage-1 / Stage-2 blend factor beta for EPDMS (default: 0.75)")
    ap.add_argument("--w-safe", type=float, default=1.50, help="Risk-averse safety log-weight w_safe for EPDMS (default: 1.50)")
    ap.add_argument("--w-cc", type=float, default=2.0, help="Stage-1 corridor consistency weight w_cc for EPDMS (default: 2.0)")
    ap.add_argument("--pdms-beta", type=float, default=1.0, help="PDMS selector beta (default: 1.0, pure official PDMS without imitation)")
    ap.add_argument("--pdms-set", default="all", help="PDMS selector variant set (default: all)")
    ap.add_argument("--pdms-lat-mode", default="on", help="PDMS selector lateral mode (default: on)")
    ap.add_argument("--mc-navtest", default=MC_NAVTEST_DEFAULT, help="NAVSIM v2.2 metric cache path for navtest")
    ap.add_argument("--mc-navhard", default=MC_NAVHARD_DEFAULT, help="NAVSIM v2.2 metric cache path for navhard")
    ap.add_argument("--fresh", action="store_true", help="Remove cached pool96/kin_cache/results in out-dir first")
    ap.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--nshard", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--seed-pool-dir", default="", help=argparse.SUPPRESS)
    return ap


def main(argv=None) -> int:
    a = get_parser().parse_args(argv)
    run_dir = Path(a.run_dir).resolve() if a.run_dir else Path(a.ckpt).resolve().parent
    out_dir = Path(a.out_dir).resolve() if a.out_dir else (run_dir / "eval_epdms")
    ckpt, tag = resolve_ckpt_and_tag(run_dir, a.ckpt)

    norm_split = "navhard" if a.split == "navhard_two_stage" else a.split
    if a.shard is not None:
        hydra_cfg = resolve_hydra_cfg(run_dir, ckpt, a.hydra_cfg)
        pool_dir = resolve_pool_dir(run_dir, out_dir, tag, norm_split)
        seed_dir = Path(a.seed_pool_dir) if a.seed_pool_dir else None
        return run_dump_shard(
            norm_split,
            pool_dir,
            ckpt,
            hydra_cfg,
            a.shard,
            a.nshard,
            a.batch_size,
            a.workers_dump,
            seed_pool_dir=seed_dir,
        )

    stages = list(STAGES) if a.stage == "all" else [s.strip() for s in a.stage.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; known {STAGES}")

    splits = ["navtest", "navhard"] if norm_split == "all" else [norm_split]
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.fresh:
        for sp in splits:
            for d in [out_dir / f"pool96_{tag}_{sp}", out_dir / "kin_cache" / f"kin_{sp}_{tag}.npz"]:
                if d.is_dir():
                    log(f"--fresh: removing {d}")
                    shutil.rmtree(d)
                elif d.is_file():
                    log(f"--fresh: removing {d}")
                    d.unlink()

    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    hydra_cfg = resolve_hydra_cfg(run_dir, ckpt, a.hydra_cfg) if "dump" in stages else Path()

    for s in stages:
        t0 = time.time()
        if s == "dump":
            for sp in splits:
                stage_dump(sp, run_dir, out_dir, tag, ckpt, hydra_cfg, gpus, a.batch_size, a.workers_dump)
        elif s == "eval":
            for sp in splits:
                stage_eval(
                    sp,
                    run_dir,
                    out_dir,
                    tag,
                    ckpt,
                    a.beta,
                    a.w_safe,
                    a.w_cc,
                    a.workers,
                    a.mc_navtest,
                    a.mc_navhard,
                    metric=a.metric,
                    pdms_beta=a.pdms_beta,
                    pdms_set=a.pdms_set,
                    pdms_lat_mode=a.pdms_lat_mode,
                )
        elif s == "summary":
            stage_summary(out_dir, tag)
        log(f"stage {s} done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
