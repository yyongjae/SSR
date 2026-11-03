#!/usr/bin/env python
"""Dump full 96-candidate pool tensors (cand96, lat_traj, valid96, v2_final, v2_im, v2_sim, score_logit) for v2_v15_v7_main (64m BEV)."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["CK_BEV64"] = "1"

import numpy as np
import yaml

REPO = Path("/workspace/yongjae/SSR-v7")
RUN = Path("/workspace/yongjae/ssd/yongjae_refiner/v7/e2e/v2_v15_v7_main")
DATA = Path("/workspace/navsim_workspace/dataset")
NAVSIM22 = Path("/workspace/navsim_workspace/navsim")
SF_NAVHARD = NAVSIM22 / "navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"
SF_NAVTEST = NAVSIM22 / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
LOGS = DATA / "navsim_logs/test/test"
SENS = DATA / "sensor_blobs/test/test"
SYN_SENS = DATA / "navhard_two_stage/sensor_blobs"
SYN_MAP_CACHE = Path("/workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep29/navhard/synth_map.json")
TOKENS_PARQUET = Path("/workspace/yongjae/ssd/yongjae_refiner/v7/eval/v2_v15_v7_main_ep30/eval/root/packed/navtest/tokens.parquet")


def _load_yaml(p: Path):
    Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(p, "r") as f:
        return yaml.load(f, Loader=Loader)


class SynthScene:
    def __init__(self, path: Path, sensor_config):
        self.path, self.sensor_config = path, sensor_config

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
                    sensor_blobs_path=SYN_SENS,
                    camera_dict=x["camera_dict"],
                    sensor_names=names,
                )
            )
            lids.append(Lidar())
        return AgentInput(ego, cams, lids)

    def get_future_trajectory(self, *a, **k):
        raise RuntimeError("synthetic scene: no future frames")


class Loader:
    def __init__(self, orig_loader, synth: dict, sensor_config):
        self.orig, self.synth, self.sc = orig_loader, synth, sensor_config

    def get_scene_from_token(self, tok):
        if tok in self.synth:
            return SynthScene(self.synth[tok], self.sc)
        return self.orig.get_scene_from_token(tok)


def split_tokens(split: str):
    if split == "navhard":
        sf = _load_yaml(SF_NAVHARD)
        orig = [str(t) for t in sf["tokens"]]
        rsi = set(map(str, sf["reactive_synthetic_initial_tokens"]))
        raw_map = json.load(open(SYN_MAP_CACHE))
        synth = {k: Path(v) for k, v in raw_map.items() if k in rsi}
        return orig, [str(x) for x in sf["log_names"]], synth, 8
    elif split == "navtest":
        import pandas as pd
        sf = _load_yaml(SF_NAVTEST)
        tdf = pd.read_parquet(TOKENS_PARQUET)
        orig = [str(t) for t in tdf.token.to_numpy()]
        return orig, [str(x) for x in sf["log_names"]], {}, 10
    else:
        raise ValueError(split)


def cmd_dump(a) -> int:
    if REPO.as_posix() not in sys.path:
        sys.path.insert(0, REPO.as_posix())
    import torch
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.script import run_aux_evaluation as RAE
    from tools.ck.data import dump_v2 as DV

    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    orig, logs, synth, num_fut = split_tokens(a.split)
    toks = orig + sorted(synth)
    pi, pn = map(int, a.part.split("/"))
    toks = [t for i, t in enumerate(toks) if i % pn == pi]
    fp = out / f"pool96_part{pi}.pkl"
    rows = pickle.load(open(fp, "rb")) if fp.exists() else {}
    if a.seed_dir:
        seed_path = Path(a.seed_dir)
        if seed_path.is_dir():
            seed_rows = {}
            for sf in sorted(seed_path.glob("pool96_part*.pkl")):
                seed_rows.update(pickle.load(open(sf, "rb")))
            for t in toks:
                if t not in rows and t in seed_rows:
                    rows[t] = seed_rows[t]
    if a.seed_eval_root:
        import pandas as pd
        er = Path(a.seed_eval_root)
        pk = er / "packed/navtest"
        inf = er / f"infer/{a.seed_run_name}/navtest"
        if (pk / "tokens.parquet").is_file() and (inf / "score_logit.npy").is_file():
            tdf = pd.read_parquet(pk / "tokens.parquet")
            tok_to_idx = {str(t): i for i, t in enumerate(tdf.token.to_numpy())}
            c96 = np.load(pk / "cand96.npy", mmap_mode="r")
            l96 = np.load(inf / "lat_traj.npy", mmap_mode="r")
            vf = np.load(pk / "v2_final.npy", mmap_mode="r")
            vim = np.load(pk / "v2_im.npy", mmap_mode="r")
            vsim = np.load(pk / "v2_sim.npy", mmap_mode="r")
            sl = np.load(inf / "score_logit.npy", mmap_mode="r")
            val96 = np.load(pk / "valid96.npy", mmap_mode="r")
            for t in toks:
                if t not in rows and t in tok_to_idx:
                    idx = tok_to_idx[t]
                    rows[t] = {
                        "cand96": np.array(c96[idx], dtype=np.float32),
                        "lat_traj": np.array(l96[idx], dtype=np.float32),
                        "v2_final": np.array(vf[idx], dtype=np.float32),
                        "v2_im": np.array(vim[idx], dtype=np.float32),
                        "v2_sim": np.array(vsim[idx], dtype=np.float32),
                        "score_logit": np.array(sl[idx], dtype=np.float32),
                        "valid96": np.array(val96[idx], dtype=bool),
                    }
    toks = [t for t in toks if t not in rows]
    print(f"[{a.split}] part {a.part}: todo {len(toks)} (done {len(rows)})", flush=True)
    if not toks:
        pickle.dump(rows, open(str(fp) + ".tmp", "wb"))
        os.replace(str(fp) + ".tmp", fp)
        return 0

    dev = torch.device(a.device if a.device != "cuda" else "cuda:0")
    cfgp = RUN / "train/code/hydra/config.yaml"
    agent = RAE._build_agent(cfgp, Path(a.ckpt), dev, aux_protocol=False)
    agent.eval()

    otoks = [t for t in toks if t not in synth]
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
            data_path=LOGS,
            sensor_blobs_path=SENS,
            scene_filter=sf,
            sensor_config=agent.get_sensor_config(),
        )
    loader = Loader(orig_loader, {t: synth[t] for t in toks if t in synth}, agent.get_sensor_config())
    dl = torch.utils.data.DataLoader(
        DV.FeatDS(loader, agent.get_feature_builders(), toks),
        batch_size=a.bs,
        num_workers=a.workers,
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
        if c % a.save_every < len(btok) or c == len(toks):
            pickle.dump(rows, open(str(fp) + ".tmp", "wb"))
            os.replace(str(fp) + ".tmp", fp)
            print(f"[{a.split}] part {a.part}: {c}/{len(toks)} ({c / max(time.time() - t0, 1e-3):.2f} tok/s)", flush=True)
    pickle.dump(rows, open(str(fp) + ".tmp", "wb"))
    os.replace(str(fp) + ".tmp", fp)
    return 0


def cmd_merge(a) -> int:
    out = Path(a.out)
    rows = {}
    for f in sorted(out.glob("pool96_part*.pkl")):
        rows.update(pickle.load(open(f, "rb")))
    orig, _, synth, _ = split_tokens(a.split)
    toks = orig + sorted(synth)
    miss = set(toks) - set(rows)
    print(f"[{a.split}] {len(rows)} rows, missing {len(miss)} of {len(toks)}", flush=True)
    if miss:
        return 1
    N = len(toks)
    json.dump(toks, open(out / "tokens.json", "w"))
    np.save(out / "cand96.npy", np.stack([rows[t]["cand96"] for t in toks], axis=0))
    np.save(out / "lat_traj.npy", np.stack([rows[t]["lat_traj"] for t in toks], axis=0))
    np.save(out / "v2_final.npy", np.stack([rows[t]["v2_final"] for t in toks], axis=0))
    np.save(out / "v2_im.npy", np.stack([rows[t]["v2_im"] for t in toks], axis=0))
    np.save(out / "v2_sim.npy", np.stack([rows[t]["v2_sim"] for t in toks], axis=0))
    np.save(out / "score_logit.npy", np.stack([rows[t]["score_logit"] for t in toks], axis=0))
    np.save(out / "valid96.npy", np.stack([rows[t]["valid96"] for t in toks], axis=0))
    print(f"[{a.split}] Saved N={N} 96-candidate arrays in {out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dump")
    d.add_argument("--split", required=True, choices=("navhard", "navtest"))
    d.add_argument("--out", required=True)
    d.add_argument("--seed-dir", default="")
    d.add_argument("--seed-eval-root", default="")
    d.add_argument("--seed-run-name", default="v2_v15_v7_main_ep30")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--part", default="0/1")
    d.add_argument("--device", default="cuda")
    d.add_argument("--bs", type=int, default=4)
    d.add_argument("--workers", type=int, default=2)
    d.add_argument("--threads", type=int, default=2)
    d.add_argument("--save-every", type=int, default=500)
    m = sp.add_parser("merge")
    m.add_argument("--split", required=True, choices=("navhard", "navtest"))
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    return cmd_dump(a) if a.cmd == "dump" else cmd_merge(a)


if __name__ == "__main__":
    sys.exit(main())
