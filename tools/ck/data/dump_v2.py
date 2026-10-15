#!/usr/bin/env python
"""CK Phase 1: dump the frozen v2 r34 student (plain v2: no refiner, no CK) per token of a split.

Port of ck/data-map/scripts/dump_v2.py (verified CPU path) with --split {navtrain_train, navtrain_val, navtest},
navtest test logs/sensors, gt_traj, per-shard done list and meta.  The v2 code is not modified: one
ParaSSRModel.forward(run_aux=True) per batch (= ParaSSRAgent.forward without aux-scale bookkeeping, refiner off).
Model build = run_aux_evaluation._build_agent (strict initialize(), plan_score_file=None, backbone_pretrained=False).

Asserted: ckpt sha256[:16], bev 50x100x256, plan_anchor, plan_heading_from_xy, no heading_from_path /
kinematic_projection / plan_kinematic / plan_rescore_refined, refiner_mode off, use_lidar off,
cand[0] == predictions['trajectory'] (top-1 of plan_final_rewards is the submitted trajectory).

Per token (resumable: tokens whose bev + plan files exist are skipped; atomic tmp + os.replace):
  v2dump/<split>/bev/<tok[:2]>/<tok>.npy   f16 [256, 50, 100] S grid (bev_embed [5000, 256], index row*100+col ->
                                           bev_embed.T.reshape(256, 50, 100); row -> x fwd, col 0 = +32 m left)
  v2dump/<split>/plan/<tok[:2]>/<tok>.npz  cand_top16 f32 [16,8,3], cand_top16_idx i16 [16], offset f16 [256,8,3],
      final f32 [256], im f32 [256], sim f32 [5,256] (CK_KEYS), model_topk_idx i16 [6], trajectory f32 [8,3],
      status f32 [8], gt_traj f32 [8,3] (scene.get_future_trajectory(8).poses, N frame; NaN if unavailable),
      det_cls f16 [300,7], det_box f16 [300,10], det_traj f16 [300,6,8,2], det_traj_cls f16 [300,6],
      map_cls f16 [100,4], map_pts f16 [100,20,2]
  v2dump/<split>/stats_shard<i>.jsonl  per batch fp16 rel err, absmax, top-1 err, seconds
  v2dump/<split>/done_shard<i>.txt     tokens finished by this shard (appended per batch)
  v2dump/<split>/meta_shard<i>.json    ckpt sha, config asserts, anchors sha, args, git HEAD, finished, tok/s

Usage (one process per GPU 0-3; dataloader workers summed over processes <= 48):
  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES=g PYTHONPATH=/workspace/yongjae/SSR-ck2 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 /venv/ssr/bin/python tools/ck/data/dump_v2.py --split S [--shard i --nshard n] [--limit N] \
    [--claim] [--out-root CK_DATA/v2dump] [--tokens <parquet>] [--batch-size 8] [--workers 8] [--k 16] [--no-bev]
  --split may be a comma list (model loaded once).  --claim: shards of every split are claimed dynamically
  (v2dump/<split>/claims/shard<i>of<n>.{json,done}); a GPU process can join or restart at any time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root of this worktree
from tools.ck.data import common as CM  # noqa: E402  (pins navsim)

DATASET = Path("/home/external-user/yongjae/SSR/data/dataset")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(DATASET / "maps"))
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(DATASET))
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", CM.CK)
os.environ.setdefault("NAVSIM_EXP_ROOT", str(CM.CK_DATA / "_exp"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import navsim  # noqa: E402

BEV_H, BEV_W, C = 50, 100, 256
SIM_KEYS_EXPECTED = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                     "time_to_collision_within_bound", "comfort")


def bev_to_sgrid(bev: torch.Tensor) -> torch.Tensor:
    """v2 bev_embed [B, 5000, 256] (query index = row * 100 + col) -> S grid [B, 256, 50, 100]
    (= bev_embed.transpose(1, 2).reshape(B, 256, 50, 100); row -> x forward, col 0 = +32 m left)."""
    assert bev.shape[1:] == (BEV_H * BEV_W, C), tuple(bev.shape)
    return bev.transpose(1, 2).reshape(-1, C, BEV_H, BEV_W).contiguous()


class FeatDS(torch.utils.data.Dataset):
    def __init__(self, loader, builders, tokens, gt_frames: int = 8):
        self.loader, self.builders, self.tokens, self.gt_frames = loader, list(builders), list(tokens), gt_frames

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        tok = self.tokens[i]
        scene = self.loader.get_scene_from_token(tok)
        ai = scene.get_agent_input()
        f = {}
        for b in self.builders:
            f.update(b.compute_features(ai))
        try:
            gt = np.asarray(scene.get_future_trajectory(self.gt_frames).poses, np.float32)
            if gt.shape != (self.gt_frames, 3):
                gt = np.full((self.gt_frames, 3), np.nan, np.float32)
        except Exception:
            gt = np.full((self.gt_frames, 3), np.nan, np.float32)
        return tok, f, torch.from_numpy(gt)


def collate(batch):
    return ([b[0] for b in batch], {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]},
            torch.stack([b[2] for b in batch]))


def build(device):
    from navsim.agents.para_ssr.modules.anchor_planner import SIM_KEYS
    from navsim.planning.script import run_aux_evaluation as RAE
    assert tuple(SIM_KEYS) == SIM_KEYS_EXPECTED, SIM_KEYS
    assert len(CM.CK_KEYS) == 5
    t0 = time.time()
    sha = CM.sha256_file(CM.V2_CKPT)
    print(f"ckpt sha256 {sha} ({time.time() - t0:.1f}s; expected [:16] {CM.V2_CKPT_SHA16})", flush=True)
    assert sha[:16] == CM.V2_CKPT_SHA16, "v2 checkpoint changed since the data-map check; re-verify before dumping"
    agent = RAE._build_agent(CM.V2_TRAIN_CFG, CM.V2_CKPT, device)
    cfg = agent.config
    chk = dict(bev=(int(cfg.bev_h), int(cfg.bev_w), int(cfg.embed_dims)), plan_anchor=bool(cfg.plan_anchor),
               plan_heading_from_xy=bool(cfg.plan_heading_from_xy), use_lidar=bool(cfg.use_lidar),
               heading_from_path=bool(getattr(cfg, "heading_from_path", False)),
               kinematic_projection=bool(getattr(cfg, "kinematic_projection", False)),
               plan_kinematic=bool(getattr(cfg, "plan_kinematic", False)),
               plan_rescore_refined=bool(getattr(cfg, "plan_rescore_refined", False)),
               refiner_mode=str(getattr(cfg, "refiner_mode", "off")), plan_topk=int(getattr(cfg, "plan_topk", 0)),
               plan_reward_weights=[float(x) for x in getattr(cfg, "plan_reward_weights", [])])
    assert chk["bev"] == (BEV_H, BEV_W, C), chk
    assert chk["plan_anchor"] and chk["plan_heading_from_xy"] and not chk["use_lidar"], chk
    assert not (chk["heading_from_path"] or chk["kinematic_projection"] or chk["plan_kinematic"]
                or chk["plan_rescore_refined"]), chk
    assert chk["refiner_mode"] == "off", chk
    return agent, sha, chk


_SF_CACHE: dict = {}


def shard_tokens(tdf: pd.DataFrame, shard: int, nshard: int) -> list:
    """Tokens of shard i: whole logs, greedily balanced by token count (largest log first, ties by name), so a
    shard opens only its own logs.  Token order inside the shard = tdf order.  nshard 1 -> all tokens."""
    if nshard <= 1:
        return list(tdf.token)
    cnt = tdf.groupby("log").size()
    logs = sorted(cnt.index, key=lambda lg: (-int(cnt[lg]), lg))
    load = [0] * nshard
    own = {}
    for lg in logs:
        j = min(range(nshard), key=lambda i: (load[i], i))
        own[lg] = j
        load[j] += int(cnt[lg])
    return [t for t, lg in zip(tdf.token, tdf.log) if own[lg] == shard]


def claim_shard(out: Path, i: int, n: int, gpu: str) -> bool:
    """Atomically claim shard i of n (claims/shard<i>of<n>.json, O_EXCL).  A claim whose pid is dead and that has no
    .done marker is taken over.  Returns True if this process owns the shard now."""
    d = out / "claims"
    d.mkdir(parents=True, exist_ok=True)
    c, done = d / f"shard{i}of{n}.json", d / f"shard{i}of{n}.done"
    if done.exists():
        return False
    rec = json.dumps(dict(pid=os.getpid(), gpu=gpu, host=os.uname().nodename, t=time.strftime("%F %T")))
    try:
        fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, rec.encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            old = json.loads(c.read_text())
            alive = Path(f"/proc/{int(old['pid'])}").exists() and old.get("host") == os.uname().nodename
        except Exception:
            alive = True    # unreadable claim (being written): leave it
        if alive:
            return False
        tmp = c.with_name(c.name + f".tmp{os.getpid()}")
        tmp.write_text(rec)
        os.replace(tmp, c)
        return True


def run_shard(agent, model, anchors_t, sha, chk, a, split_name: str, shard: int, nshard: int, dev, tdf=None) -> dict:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader

    sp = CM.split(split_name)
    out = (Path(a.out_root) if a.out_root else CM.root_dir() / "v2dump") / split_name
    out.mkdir(parents=True, exist_ok=True)
    if tdf is None:
        tdf = pd.read_parquet(a.tokens) if a.tokens else CM.split_tokens(split_name)
        tdf["token"] = tdf.token.astype(str)
    toks = shard_tokens(tdf, shard, nshard)
    if a.limit:
        toks = toks[: a.limit]
    done = lambda t: CM.plan_path(out, t).is_file() and (a.no_bev or CM.bev_path(out, t).is_file())  # noqa: E731
    todo = [t for t in toks if not done(t)]
    print(f"[{time.strftime('%F %T')}] split {split_name} shard {shard}/{nshard} tokens {len(toks)} todo {len(todo)} "
          f"-> {out}", flush=True)
    meta = dict(split=split_name, shard=shard, nshard=nshard, n_tokens=len(toks), n_todo=len(todo),
                ckpt=str(CM.V2_CKPT), ckpt_sha256=sha, config_checks=chk,
                anchors_sha16=CM.sha16_array(anchors_t.cpu().numpy()), k=a.k, args=vars(a), git_head=CM.git_head(),
                navsim=navsim.__file__, gpu=os.environ.get("CUDA_VISIBLE_DEVICES", ""), started=time.strftime("%F %T"))
    if not todo:
        meta.update(n_done=0, finished=time.strftime("%F %T"))
        CM.atomic_write_json(out / f"meta_shard{shard}.json", meta)
        return meta
    if split_name not in _SF_CACHE:     # the navtrain filter yaml lists 103k tokens: instantiate once per process
        _SF_CACHE[split_name] = instantiate(OmegaConf.load(CM.resolve_path(sp["scene_filter"])))
    sf = replace(_SF_CACHE[split_name], tokens=list(todo), max_scenes=None)
    need = set(tdf.set_index("token").loc[todo, "log"])
    sf = replace(sf, log_names=sorted(set(sf.log_names or need) & need))
    t0 = time.time()
    loader = SceneLoader(data_path=Path(sp["navsim_logs"]), sensor_blobs_path=Path(sp["sensor_blobs"]), scene_filter=sf,
                         sensor_config=agent.get_sensor_config())
    miss = set(todo) - set(loader.tokens)
    assert not miss, f"scene loader misses {len(miss)} tokens e.g. {sorted(miss)[:3]}"
    print(f"scene loader {len(loader.tokens)} tokens in {time.time() - t0:.1f}s", flush=True)
    ds = FeatDS(loader, agent.get_feature_builders(), todo)
    if a.cpu_check:
        rep = {}
        for i in range(min(a.cpu_check, len(todo))):
            t1 = time.time()
            tok, f, gt = ds[i]
            rep[tok] = {k: [list(v.shape), str(v.dtype)] for k, v in f.items()}
            rep[tok]["gt_traj_finite"] = bool(torch.isfinite(gt).all())
            print(tok, f"{time.time() - t1:.2f}s", rep[tok], flush=True)
        CM.atomic_write_json(out / f"cpu_check_shard{shard}.json", dict(meta, features=rep))
        return meta
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers, collate_fn=collate,
                                     pin_memory=True, prefetch_factor=2 if a.workers else None,
                                     persistent_workers=False)
    st = open(out / f"stats_shard{shard}.jsonl", "a")
    dn = open(out / f"done_shard{shard}.txt", "a")
    n, t_start = 0, time.time()
    max_rel = max_top1 = 0.0
    anchors_dev = anchors_t.to(dev)
    for btok, feats, gt in dl:
        tb = time.time()
        fg = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else v.dtype, non_blocking=True)
              for k, v in feats.items()}
        with torch.inference_mode():
            pr = model(fg, run_aux=True)
        bev = pr["bev_embed"]
        assert bev.shape[1:] == (BEV_H * BEV_W, C), tuple(bev.shape)
        final = pr["plan_final_rewards"].float()                                         # [B, 256]
        top = final.topk(a.k, dim=-1).indices                                             # [B, K] descending
        off = pr["trajectory_offset"].float()                                             # [B, 256, 8, 3]
        refined = anchors_dev.unsqueeze(0) + off
        cand = refined.gather(1, top[:, :, None, None].expand(-1, -1, 8, 3))             # [B, K, 8, 3]
        err1 = float((cand[:, 0] - pr["trajectory"].float()).abs().max())
        assert err1 < 1e-4, f"top-1 candidate != submitted trajectory ({err1})"
        g = lambda k: pr[k].detach().float().cpu().numpy()  # noqa: E731
        x = bev_to_sgrid(bev.float())
        x16 = x.half()
        rel = float((x16.float() - x).norm() / x.norm().clamp_min(1e-12))
        fin16 = bool(torch.isfinite(x16).all())
        assert fin16, "non-finite fp16 BEV"
        arr = x16.cpu().numpy()
        P = dict(cand=cand.cpu().numpy(), idx=top.cpu().numpy().astype(np.int16), offset=g("trajectory_offset"),
                 final=g("plan_final_rewards"), im=g("im_rewards"), sim=g("sim_rewards"),
                 mtop=pr["plan_topk_index"].cpu().numpy().astype(np.int16), traj=g("trajectory"),
                 status=fg["status_feature"].float().cpu().numpy(), gt=gt.numpy().astype(np.float32),
                 det_cls=g("all_cls_scores")[-1], det_box=g("all_bbox_preds")[-1], det_traj=g("traj_preds"),
                 det_traj_cls=g("traj_cls_preds"), map_cls=g("all_map_cls_scores")[-1],
                 map_pts=g("all_map_pts_preds")[-1])
        for b, tok in enumerate(btok):
            if not a.no_bev:
                CM.atomic_save_npy(CM.bev_path(out, tok), arr[b])
            rec = {f"cand_top{a.k}": P["cand"][b].astype(np.float32), f"cand_top{a.k}_idx": P["idx"][b],
                   "offset": P["offset"][b].astype(np.float16), "final": P["final"][b].astype(np.float32),
                   "im": P["im"][b].astype(np.float32), "sim": P["sim"][b].astype(np.float32),
                   "model_topk_idx": P["mtop"][b], "trajectory": P["traj"][b].astype(np.float32),
                   "status": P["status"][b].astype(np.float32), "gt_traj": P["gt"][b],
                   "det_cls": P["det_cls"][b].astype(np.float16), "det_box": P["det_box"][b].astype(np.float16),
                   "det_traj": P["det_traj"][b].astype(np.float16),
                   "det_traj_cls": P["det_traj_cls"][b].astype(np.float16),
                   "map_cls": P["map_cls"][b].astype(np.float16), "map_pts": P["map_pts"][b].astype(np.float16)}
            CM.atomic_savez(CM.plan_path(out, tok), **rec)
        dn.write("\n".join(btok) + "\n")
        dn.flush()
        n += len(btok)
        max_rel, max_top1 = max(max_rel, rel), max(max_top1, err1)
        st.write(json.dumps(dict(n=n, absmax=float(x.abs().max()), fp16_rel_err=rel, top1_err=err1, finite=fin16,
                                 gt_finite=int(np.isfinite(P["gt"]).all(axis=(1, 2)).sum()),
                                 sec_gpu=round(time.time() - tb, 3), t=round(time.time() - t_start, 1))) + "\n")
        st.flush()
        if n % 400 < len(btok) or n == len(todo):
            el = time.time() - t_start
            print(f"[{time.strftime('%T')}] {split_name} s{shard} {n}/{len(todo)} {el:.0f}s {n / el:.2f} tok/s "
                  f"rel {rel:.2e} top1_err {err1:.1e} ETA {(len(todo) - n) / max(n / el, 1e-9) / 60:.1f} min",
                  flush=True)
    st.close()
    dn.close()
    el = time.time() - t_start
    meta.update(n_done=n, seconds=round(el, 1), tok_per_s=round(n / max(el, 1e-9), 3), max_fp16_rel_err=max_rel,
                max_top1_err=max_top1, finished=time.strftime("%F %T"),
                gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == "cuda" else 0.0)
    CM.atomic_write_json(out / f"meta_shard{shard}.json", meta)
    print(f"done {split_name} shard {shard}: {n} in {el:.0f}s ({n / max(el, 1e-9):.2f} tok/s)", flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, help="split name or comma list processed in order (model loaded once)")
    ap.add_argument("--out-root", default="", help="default <CK_DATA or $CK_DATA_ROOT>/v2dump")
    ap.add_argument("--tokens", default="", help="override token parquet (token, log); single split only")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--claim", action="store_true",
                    help="dynamic: loop over all --nshard shards of every split, claiming free ones (any number of GPU "
                         "processes can join or restart; claims in v2dump/<split>/claims/)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=CM.K_CAND)
    ap.add_argument("--no-bev", action="store_true", help="plan/det/map only (no 2.56 MB BEV)")
    ap.add_argument("--cpu-check", type=int, default=0, help="CPU: build + load + features of N tokens, no forward")
    a = ap.parse_args(argv)
    print("navsim", navsim.__file__, flush=True)
    assert navsim.__file__.startswith(CM.CK), navsim.__file__
    assert a.workers <= 48
    splits = [s.strip() for s in a.split.split(",") if s.strip()]
    for s in splits:
        CM.split(s)
    assert not (a.tokens and len(splits) > 1), "--tokens needs a single split"
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not a.cpu_check:
        assert gpu in ("0", "1", "2", "3"), f"CUDA_VISIBLE_DEVICES={gpu!r}: one of GPU 0-3 only"
    dev = torch.device("cpu" if a.cpu_check else "cuda:0")
    t0 = time.time()
    agent, sha, chk = build(dev)
    model = agent.para_ssr_model.eval()
    anchors_t = model.pts_bbox_head.anchor_planner.trajectory_anchors.detach().float()
    print(f"agent built + strictly loaded on {dev} in {time.time() - t0:.1f}s; anchors {tuple(anchors_t.shape)} "
          f"sha16 {CM.sha16_array(anchors_t.cpu().numpy())}", flush=True)
    for s in splits:
        if not a.claim:
            run_shard(agent, model, anchors_t, sha, chk, a, s, a.shard, a.nshard, dev)
            continue
        out = (Path(a.out_root) if a.out_root else CM.root_dir() / "v2dump") / s
        tdf = pd.read_parquet(a.tokens) if a.tokens else CM.split_tokens(s)
        tdf["token"] = tdf.token.astype(str)
        order = [(a.shard + j) % a.nshard for j in range(a.nshard)]     # start at --shard to spread processes
        for i in order:
            if not claim_shard(out, i, a.nshard, gpu):
                continue
            run_shard(agent, model, anchors_t, sha, chk, a, s, i, a.nshard, dev, tdf=tdf)
            (out / "claims" / f"shard{i}of{a.nshard}.done").write_text(time.strftime("%F %T"))
    print(f"[{time.strftime('%F %T')}] all requested splits finished", flush=True)


if __name__ == "__main__":
    main()
