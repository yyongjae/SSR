#!/usr/bin/env python3
"""Stage-E diagnostics: dump, per navtest token, the E2 checkpoint's planner draft tau0, the student-refined tau_final,
the student refiner's raw outputs (z_lon, w_lat, gate_logit / p_g), its decoded controls (c_lon, e_lat [8]; the KD
controls are c_lon[2:], e_lat[2:]) and flags, and the ego inputs it read (v0, a0, eds, cmd, status_feature).

Inference path = the official navtest eval (run_pdm_score_gpu.py -> AbstractAgent.compute_trajectory_gpu):
  agent built from the archived EVAL hydra config (work_dirs/eval/stageE_E2/code/hydra/config.yaml, agent section),
  strict load of the checkpoint, eval mode, fp32, batch size 1 by default (official is 1 token per forward),
  forward = agent._loss.apply_aux_scales + para_ssr_model(features) (run_aux from cfg, as in agent.forward), then the
  StageE.infer computation re-done inline so the student outputs can be recorded; StageE.infer itself is also run and
  its tau_final / tau0 are asserted BITWISE equal to the recorded ones.
Outputs (--out-dir): e2_navtest_dump.npz (sorted tokens), e2_navtest_dump.meta.json,
  e2_tau0_navtest_trajectories.pkl / e2_final_navtest_trajectories.pkl ({'trajectories': {tok: [8,3]}, 'meta'}),
  e2_score_drafts.npz (tokens, drafts [N, 2, 8, 3] = [tau0, tau_final]) for tools/refiner/score_trajectories.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from dump_navtest_trajectories import _TokenFeatures, _collate, _worker_init  # noqa: E402

RUN = REPO / "work_dirs/stageE_E2_30ep_ema_r1.0_s0_rr5_wmax100_hmix_hwu5_bg1.0"
EVAL_CFG = REPO / "work_dirs/eval/stageE_E2/code/hydra/config.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=RUN / "lightning_logs/version_0/checkpoints/last.ckpt")
    ap.add_argument("--eval-cfg", type=Path, default=EVAL_CFG)
    ap.add_argument("--download", type=Path, default=Path("/home/external-user/navsim/download"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("/home/external-user/ssd/yongjae_refiner/stageE_diag"))
    a = ap.parse_args()

    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(REPO / "data/dataset/maps"))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(REPO / "data/dataset"))
    torch.set_num_threads(1)

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    from navsim.agents.para_ssr.refiner.decoder import lon_live

    cfg = OmegaConf.load(a.eval_cfg)
    agent_cfg = OmegaConf.create(OmegaConf.to_container(cfg.agent, resolve=False))
    OmegaConf.set_struct(agent_cfg, False)
    agent_cfg.checkpoint_path = str(a.checkpoint)
    assert agent_cfg.config.refiner_mode == "E2" and agent_cfg.config.ref_eval_traj == "final"
    dev = torch.device(a.device)
    agent = instantiate(agent_cfg).to(dev)
    agent.initialize()
    agent.is_eval = True
    agent.eval()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    ck_epoch, ck_step = int(ck["epoch"]), int(ck["global_step"])
    del ck
    st, student, model = agent._stage_e, agent.ref_student, agent.para_ssr_model
    assert st is not None and not agent.training and not student.training

    scene_filter = instantiate(cfg.scene_filter)
    loader = SceneLoader(sensor_blobs_path=a.download / "test_sensor_blobs/test",
                         data_path=a.download / "test_navsim_logs/test", scene_filter=scene_filter,
                         sensor_config=agent.get_sensor_config())
    tokens = sorted(loader.tokens)
    def _log(t):
        try:
            return str(loader.scene_frames_dicts[t][0]["log_name"])
        except Exception:
            return ""
    logs = {t: _log(t) for t in tokens}
    if a.max_tokens:
        tokens = tokens[: a.max_tokens]
    print(f"E2 epoch {ck_epoch} step {ck_step} | {len(tokens)} tokens | {a.checkpoint}", flush=True)
    data = DataLoader(_TokenFeatures(loader, agent.get_feature_builders(), tokens), batch_size=a.batch_size,
                      num_workers=a.workers, collate_fn=_collate, worker_init_fn=_worker_init,
                      persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)

    keys = ("tau0", "tau_final", "z_lon", "w_lat", "gate_logit", "p_g", "c_lon", "e_lat", "d", "alpha", "beta",
            "lat_on", "lon_live", "v0", "a0", "eds", "cmd", "status_feature", "command", "ego_fut_preds")
    rec = {k: [] for k in keys}
    toks, t0 = [], time.time()
    with torch.no_grad():
        for batch_tokens, feats in data:
            feats = {k: v.to(dev, non_blocking=True) for k, v in feats.items()}
            agent._loss.apply_aux_scales(model)
            pred = model(feats)
            tau0 = pred["trajectory"].detach()
            ego = ego_inputs(feats["status_feature"])
            out = st._run(student, pred["bev_embed"], tau0.float(), ego)
            dec = st._decode(tau0, out, ego[0], 0.0)
            tau_f = dec["traj"].to(tau0.dtype)
            chk = st.infer(student, feats, pred)                        # the official inference helper
            if not (torch.equal(chk["trajectory"], tau_f) and torch.equal(chk["tau0"], tau0)):
                raise RuntimeError(f"infer mismatch at {batch_tokens[0]}")
            if not (torch.isfinite(tau_f).all() and torch.isfinite(tau0).all()):
                raise RuntimeError(f"non-finite at {batch_tokens[0]}")
            v0, a0, eds, cmd = ego
            f = dec["flags"]
            vals = {"tau0": tau0, "tau_final": tau_f, "z_lon": out["z_lon"][:, 0], "w_lat": out["w_lat"][:, 0],
                    "gate_logit": out["gate_logit"][:, 0], "p_g": torch.sigmoid(out["gate_logit"][:, 0].float()),
                    "c_lon": dec["c_lon"], "e_lat": dec["e_lat"], "d": dec["d"], "alpha": f["alpha"],
                    "beta": f["beta"], "lat_on": f["lat_on"], "lon_live": lon_live(out["z_lon"][:, 0].float()),
                    "v0": v0, "a0": a0, "eds": eds, "cmd": cmd, "status_feature": feats["status_feature"],
                    "command": feats["command"], "ego_fut_preds": pred["ego_fut_preds"]}
            for k, v in vals.items():
                rec[k].append(v.detach().cpu().numpy())
            toks += list(batch_tokens)
            if len(toks) % 1000 < a.batch_size:
                print(f"  {len(toks)}/{len(tokens)} {len(toks) / (time.time() - t0):.1f} tok/s", flush=True)
    R = {k: np.concatenate(v) for k, v in rec.items()}
    R["tau0"] = R["tau0"].astype(np.float32)
    R["tau_final"] = R["tau_final"].astype(np.float32)
    tokens = np.array(toks).astype(str)
    assert list(tokens) == sorted(tokens)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(a.out_dir / "e2_navtest_dump.npz", tokens=tokens,
             log=np.array([logs.get(t, "") for t in tokens]).astype(str), **R)
    np.savez(a.out_dir / "e2_score_drafts.npz", tokens=tokens,
             drafts=np.stack([R["tau0"], R["tau_final"]], 1).astype(np.float32))
    meta = dict(checkpoint=str(a.checkpoint), checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                checkpoint_epoch=ck_epoch, global_step=ck_step, eval_cfg=str(a.eval_cfg), batch_size=a.batch_size,
                device=a.device, dtype="fp32 (no autocast, as compute_trajectory_gpu)", n_tokens=int(len(tokens)),
                token_sha256=hashlib.sha256("".join(f"{t}\n" for t in tokens).encode()).hexdigest(),
                tau_final_eq_tau0_frac=float((R["tau_final"] == R["tau0"]).all((1, 2)).mean()),
                decode="mode A, lon_st_slope 0 (StageE.infer), no gate (p_g recorded only; gate head untrained in "
                       "stage E: loss adds 0 * gate_logit)",
                shapes={k: list(v.shape) for k, v in R.items()},
                sec=round(time.time() - t0, 1), created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    for name, key in (("e2_tau0", "tau0"), ("e2_final", "tau_final")):
        with open(a.out_dir / f"{name}_navtest_trajectories.pkl", "wb") as fh:
            pickle.dump({"trajectories": {t: R[key][i] for i, t in enumerate(tokens)},
                         "meta": dict(meta, trajectory=key)}, fh)
    (a.out_dir / "e2_navtest_dump.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
