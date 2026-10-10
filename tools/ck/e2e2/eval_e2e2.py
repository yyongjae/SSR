#!/usr/bin/env python
"""CK2 e2e evaluation of a trained v2 + CK2 checkpoint (SPEC ck2e2e s5-3).

  cd /workspace/yongjae/SSR-ck2 && PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /venv/ssr/bin/python tools/ck/e2e2/eval_e2e2.py --run-dir <RUN> --split navtrain_val [--ckpt last] [--limit N] \
    [--gpus 0,1] [--stage extract,dump,label,infer_check,eval,summary | all] [--workers-label 48] \
    [--fix-from <val metrics.json>] [--calib-offset none] [--choose-lat on|any] [--sel-w default|noim|plugin] [--fresh]

Stages (each resumable; --stage all = this order; layout = tools/ck/e2e/eval_e2e.Paths):
  extract      Lightning ckpt 'agent.ck_student.*' -> <RUN>/eval/ck_run/{config.json, ckpt_last.pt} (load_ck format, arm S,
               score_hidden 256, lead_aux 0, e2e2 true, lon head zero) + strict load_ck round trip (eval_e2e.extract_ck).
  dump         agent from <RUN>/train/code/hydra/config.yaml + ckpt (run_aux_evaluation._build_agent, strict; the BEV-KD
               arm module is rebuilt too), eval forward (CKE2E2.infer: the 96-column pool) on the split's tokens ->
               eval root <RUN>/eval/root:
                 packed/<split>/{tokens.parquet, cand96 [N,96,8,3], valid96 [N,96], cand_idx, v2_final, v2_im [N,16],
                                 v2_sim [N,16,5], status, gt_traj, sub_traj, ok, ok_rows, meta.json}
                 infer/<RUN>/<split>/{score_logit [N,96,5], w_lat, e_lat [N,96,6], lat_traj [N,96,8,3], done,
                                 check_bev.npz, meta.json}
               asserts cand96[:, ::6] == gather(anchors + offset, topk16(final)) (1e-5), cand_idx == topk and
               cand96[:, 0] == trajectory (1e-4).  Several GPUs: one child per GPU (whole logs per shard).
  label        label_cands.run (official score_token, LQR; K = 96 from the file shape) on packed/cand96 -> labels/<split>/
               pool96_<RUN> and on infer/lat_traj -> lat96_<RUN> (rows = packed ok; workers <= 48).
  infer_check  the extracted ck_run (load_ck, fp32) on 8 stored BEV grids x 96 columns reproduces score_logit (3e-2) and
               lat_traj (2e-3).
  eval         eval_ck2.run_eval: navtrain_val = beta x variant set x lateral-mode grid, choice stored as 'best'
               (beta / variant set; lateral mode fixed 'on' = user decision unless --choose-lat any);
               navtest only with --fix-from (default the val metrics.json of this run): the val choice is reported once.
               --sel-w (select.SEL_W_SETS, default 'default' = SEL_W): optional comparison weight set, e.g. 'plugin'
               = (0, 1, 1, 1); a non-default set writes <split>__selw_<name> next to the default results, a
               calibration offset <split>[...]__cal_<sha8> (an offset ablation never overwrites the val result).
               The val result holds ONE frozen eval spec (eval_ck2: choice, sel_w values, calibration offset values +
               source sha, ep_target, ckpt sha16, ...); navtest restores it -- an unset --sel-w / --calib-offset is
               inherited, a given value must equal it, and the navtest dump must be from the same ckpt sha16.
  summary      <RUN>/eval/summary.{json,md} (times in KST) of the (--sel-w, --calib-offset) result dirs; each split's
               checkpoint comes from its own metrics.json and a val / navtest checkpoint mismatch is refused.

Checkpoint pin and cache identity (report 48 F1):
  --ckpt X resolves X ('last' | epoch:<E> | <path>), hashes the FILE and pins it in <RUN>/eval/ckpt.json (written only
  after the dump-cache check below passed, so a refused attempt leaves the pin unchanged).  Without
  --ckpt the documented default 'last' is resolved again and must equal the pin (same path, same sha16 of the file);
  otherwise SystemExit (pass --ckpt explicitly) -- never a silent reuse of an older pin or of a last.ckpt rewritten in
  place.  --fresh removes the split's cached outputs AND the pin.
  The dump cache packed/<split>/cache_key.json = {ckpt, ckpt_sha16 of the file, hydra_sha16 of the training config,
  code_sha16 over the dump-path sources (DUMP_CODE, per-file sha16 kept)}: done rows are reused only under an equal
  key; a different key with done rows, or done rows without a key, is a SystemExit (use --fresh).  Labels are reused
  only when their meta.json traj_sha16 equals the current cand96 / lat_traj; infer_check refuses an extracted ck_run of
  a different checkpoint than the dump.

Standard NAVSIM submissions (report 48 section 5): a regular agent evaluation (run_pdm_score / the agent's
compute_trajectory) submits predictions['trajectory'], which stays v2's trajectory in the CK2 e2e agent (NU30;
ck_e2e2.infer_select only adds ck2_traj / ck2_sel_idx and cannot carry --sel-w / a calibration offset).  CK2 numbers
(selection over the 96 pool + lateral) come ONLY from this tool (eval_ck2); a standard NAVSIM result of a CK2 e2e
checkpoint is the v2 trajectory of that checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)
import navsim  # noqa: E402

assert navsim.__file__.startswith(CK), navsim.__file__

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402
from tools.ck.e2e import eval_e2e as E1  # noqa: E402   (Paths, resolve / pin ckpt, extract_ck, split_table, shards)
from tools.ck.e2e2 import eval_ck2 as EV  # noqa: E402

PY = "/venv/ssr/bin/python"
K16, G_K = 16, 96
STAGES = ("extract", "dump", "label", "infer_check", "eval", "summary")
N_CHECK_BEV = 8
Paths = E1.Paths
PACKED_SPECS = {          # name -> (per-row shape, dtype, fill)
    "cand96": ((G_K, 8, 3), np.float32, np.nan), "valid96": ((G_K,), np.bool_, False),
    "cand_idx": ((K16,), np.int16, -1), "v2_final": ((K16,), np.float32, np.nan),
    "v2_im": ((K16,), np.float32, np.nan), "v2_sim": ((K16, 5), np.float32, np.nan),
    "status": ((8,), np.float32, np.nan), "gt_traj": ((8, 3), np.float32, np.nan),
    "sub_traj": ((8, 3), np.float32, np.nan), "ok": ((), np.bool_, False),
}
INFER_SPECS = {
    "score_logit": ((G_K, 5), np.float32, np.nan), "w_lat": ((G_K, 6), np.float32, np.nan),
    "e_lat": ((G_K, 6), np.float32, np.nan), "lat_traj": ((G_K, 8, 3), np.float32, np.nan),
    "done": ((), np.bool_, False),
}
PRED_TO_PACKED = {"ck2_cand96": "cand96", "ck2_valid96": "valid96", "ck2_cand_idx": "cand_idx",
                  "ck2_v2_final": "v2_final", "ck2_v2_im": "v2_im", "ck2_v2_sim": "v2_sim"}
PRED_TO_INFER = {"ck2_score_logit": "score_logit", "ck2_w_lat": "w_lat", "ck2_e_lat": "e_lat",
                 "ck2_lat_traj": "lat_traj"}


def log(msg: str) -> None:
    print(f"[{EV.kst()}] [eval_e2e2] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- ckpt pin / cache key
# sources whose change can change a dump (v2 forward + CKE2E2.infer + candidates / variants + this dump code); a
# directory = every *.py below it except tests / __pycache__
DUMP_CODE = ("navsim/agents/para_ssr", "navsim/planning/script/run_aux_evaluation.py", "tools/ck/e2e2/eval_e2e2.py",
             "tools/ck/e2e/eval_e2e.py", "tools/ck/data/dump_v2.py")
CACHE_KEY_FIELDS = ("ckpt_sha16", "hydra_sha16", "code_sha16")


def dump_code_files(root: str = CK) -> List[Path]:
    out: List[Path] = []
    for rel in DUMP_CODE:
        p = Path(root) / rel
        if p.is_dir():
            out += sorted(q for q in p.rglob("*.py") if "__pycache__" not in q.parts and not q.name.startswith("test_"))
        elif p.is_file():
            out.append(p)
    return out


def code_manifest(files: Optional[List[Path]] = None) -> Dict[str, str]:
    """{path (relative to the repo when inside it): sha16} of the dump-path sources."""
    out = {}
    for f in (dump_code_files() if files is None else files):
        f = Path(f)
        try:
            k = str(f.resolve().relative_to(Path(CK).resolve()))
        except ValueError:
            k = str(f)
        out[k] = U.sha256_file(f)
    return out


def _sha16_obj(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:16]


def resolve_pin(P: Paths, spec: Optional[str]) -> Tuple[Path, str, Optional[Dict[str, Any]]]:
    """-> (checkpoint, sha16 of the file, new pin record or None).  --ckpt given (or no pin yet): resolve (eval_e2e.
    resolve_ckpt) + hash the file; the caller writes the returned record to <RUN>/eval/ckpt.json (eval_e2e.pinned_ckpt
    format) only after the cache check passed.  No --ckpt and a pin: the pinned file must still have the pinned sha16
    AND be the current 'last' (the documented default) -- else SystemExit: an older pin (an epoch:E comparison,
    version_0 after a resume) or a last.ckpt rewritten in place is never used silently."""
    pin = U.read_json(P.pin)
    if spec or not pin:
        p = E1.resolve_ckpt(P.run_dir, spec or "last")
        if pin and pin.get("path") != str(p):
            log(f"ckpt changed: pinned {pin.get('path')} -> {p}")
        sha = U.sha256_file(p)
        return p, sha, {"path": str(p), "spec": spec or "last", "sha16": sha, "size": p.stat().st_size,
                        "pinned_utc": time.strftime("%F %T", time.gmtime())}
    pp = Path(pin["path"])
    if not pp.is_file():
        raise SystemExit(f"pinned checkpoint {pp} ({P.pin}) no longer exists; pass --ckpt explicitly")
    sha = U.sha256_file(pp)
    if sha != pin.get("sha16"):
        raise SystemExit(f"pinned checkpoint {pp} changed on disk since it was pinned (sha16 {pin.get('sha16')} -> "
                         f"{sha}, e.g. last.ckpt rewritten by a resumed run); pass --ckpt explicitly (and --fresh if "
                         f"outputs were dumped from the old bytes)")
    try:
        cur = E1.resolve_ckpt(P.run_dir, "last")
    except (SystemExit, Exception) as e:      # noqa: BLE001  (LaunchError: no lightning_logs/version_*/last.ckpt)
        raise SystemExit(f"no --ckpt: cannot resolve 'last' under {P.run_dir} to check the pin {pp} "
                         f"({getattr(e, 'msg', e)}); pass --ckpt explicitly") from None
    if cur != pp.resolve():
        raise SystemExit(f"no --ckpt: the pinned checkpoint {pp} (spec {pin.get('spec')!r}, sha16 {sha}) is not the "
                         f"current 'last' {cur}; pass --ckpt explicitly (--ckpt last re-pins; --fresh for a new "
                         f"checkpoint's outputs)")
    return pp, sha, None


def pinned_ckpt2(P: Paths, spec: Optional[str]) -> Tuple[Path, str]:
    """resolve_pin + write the pin -> (checkpoint, sha16 of the file)."""
    p, sha, rec = resolve_pin(P, spec)
    if rec is not None:
        U.write_json(P.pin, rec)
    return p, sha


def cache_key(P: Paths, ckpt: Path, ckpt_sha16: str, limit: int = 0) -> Dict[str, Any]:
    man = code_manifest()
    return {"ckpt": str(ckpt), "ckpt_sha16": ckpt_sha16,
            "hydra_sha16": U.sha256_file(P.hydra_cfg) if P.hydra_cfg.is_file() else None,
            "code_sha16": _sha16_obj(man), "code_files": man, "limit": int(limit or 0), "created": EV.kst()}


def n_done_rows(P: Paths) -> int:
    f = P.infer / "done.npy"
    return int(np.asarray(np.load(f, mmap_mode="r"), bool).sum()) if f.is_file() else 0


def check_cache_key(P: Paths, key: Dict[str, Any]) -> Dict[str, Any]:
    """Refuse reusing done dump rows of another checkpoint / training config / dump code (SystemExit, use --fresh);
    a cache with done rows but no key (written before the key existed, or by another tool) is refused too.  Writes
    the key when there is nothing to reuse.  -> the stored key."""
    kp = P.packed / "cache_key.json"
    old = U.read_json(kp) if kp.is_file() else None
    nd = n_done_rows(P)
    if old is None:
        if nd:
            raise SystemExit(f"{P.packed}: {nd} done dump rows but no cache_key.json (unkeyed cache: its checkpoint / "
                             f"config / code cannot be verified); rerun with --fresh")
    else:
        diff = [k for k in CACHE_KEY_FIELDS if old.get(k) != key[k]]
        if not diff:
            if old.get("ckpt") != key["ckpt"]:     # same bytes under another path (Lightning's last.ckpt == the final
                old = dict(old, ckpt=key["ckpt"])  # epoch=E file): the dump shards check the path -> record this one
                U.write_json(kp, old)
            return old
        if nd:
            what = []
            for k in diff:
                if k == "code_sha16":
                    a, b = old.get("code_files") or {}, key["code_files"]
                    ch = sorted(f for f in set(a) | set(b) if a.get(f) != b.get(f))
                    what.append(f"dump code ({len(ch)} files: {', '.join(ch[:5])}{' ...' if len(ch) > 5 else ''})")
                elif k == "ckpt_sha16":
                    what.append(f"checkpoint {old.get('ckpt')} ({old.get('ckpt_sha16')}) -> {key['ckpt']} "
                                f"({key['ckpt_sha16']})")
                else:
                    what.append(f"training config sha16 {old.get(k)} -> {key[k]}")
            raise SystemExit(f"{P.packed}: {nd} done dump rows were made with a different " + "; ".join(what)
                             + ": reusing them would mix models / code in one eval; rerun with --fresh")
    P.packed.mkdir(parents=True, exist_ok=True)
    U.write_json(kp, key)
    return key


# ----------------------------------------------------------------------------------------------- extract
def hydra_ck2_cfg(P: Paths) -> Dict[str, Any]:
    if not P.hydra_cfg.is_file():
        return {}
    from omegaconf import OmegaConf
    c = OmegaConf.load(P.hydra_cfg)
    try:
        ck = c.agent.config.get("ck_e2e2")
    except Exception:       # noqa: BLE001
        ck = None
    return OmegaConf.to_container(ck, resolve=False) if ck is not None else {}


def extract_ck2(ckpt: Path, out_dir: Path, run_name: str, ck_cfg: Dict[str, Any]) -> Dict:
    """eval_e2e.extract_ck (agent.ck_student.* -> load_ck run, strict round trip) + CK2 fields in config.json; the lon
    head of the extracted student must be all zero (z_lon = 0)."""
    from navsim.agents.para_ssr.ck.model import load_ck
    if not ck_cfg or not str(ck_cfg.get("enabled", "")).lower() in ("true", "1"):
        log(f"warning: training config has no enabled ck_e2e2 ({ck_cfg and ck_cfg.get('enabled')})")
    cfg = E1.extract_ck(ckpt, out_dir, run_name, {"seed": int((ck_cfg or {}).get("seed", 0) or 0)})
    net, _ = load_ck(out_dir, "last", "cpu")
    lw = net.trunk.lon_head[-1]
    lon_max = max(float(lw.weight.abs().max()), float(lw.bias.abs().max()))
    if lon_max != 0.0:
        raise SystemExit(f"extracted student lon head max |w| {lon_max} != 0 (not a CK2 e2e checkpoint)")
    cfg.update(e2e2=True, ck_e2e2=ck_cfg, lon_head="removed (last layer zero, frozen; decode z_lon = 0)",
               k=G_K, pool="c = k*6 + v (v2 top-16 x id, a-1.0, a-0.5, a+0.5, l-0.5, l+0.5)")
    cfg.pop("ck_e2e", None)
    U.write_json(out_dir / "config.json", cfg)
    load_ck(out_dir, "last", "cpu")          # config.json still loads
    return cfg


# ----------------------------------------------------------------------------------------------- writer
class EvalRootWriter2(E1.EvalRootWriter):
    """eval_e2e.EvalRootWriter with the 96-column specs."""

    def __init__(self, P: Paths, tdf, ck_run: Optional[Path] = None):
        import pandas as pd
        self.P = P
        n = len(tdf)
        self.n = n
        P.packed.mkdir(parents=True, exist_ok=True)
        P.infer.mkdir(parents=True, exist_ok=True)
        tdf = tdf[["token", "log", "city"]].reset_index(drop=True).copy()
        tdf["row"] = np.arange(n, dtype=np.int64)
        with U.FileLock(P.packed / ".lock"):
            for d in (P.packed, P.infer):
                tp = d / "tokens.parquet"
                if tp.is_file():
                    old = pd.read_parquet(tp)
                    if len(old) != n or not (old.token.astype(str).to_numpy() == tdf.token.to_numpy()).all():
                        raise SystemExit(f"{tp} holds a different token list (n {len(old)} vs {n}); use --fresh")
                else:
                    tmp = d / f".tokens.tmp{os.getpid()}.parquet"
                    tdf.to_parquet(tmp, index=False)
                    os.replace(tmp, tp)
            self.A = {k: U.open_memmap(P.packed / f"{k}.npy", (n,) + s, d, fill=f) for k, (s, d, f) in
                      PACKED_SPECS.items()}
            self.I = {k: U.open_memmap(P.infer / f"{k}.npy", (n,) + s, d, fill=f) for k, (s, d, f) in
                      INFER_SPECS.items()}
        self.ck_run = ck_run

    def write(self, rows, pred, status, gt, traj):
        rows = np.asarray(rows, np.int64)
        B = len(rows)
        ok = np.isfinite(pred["ck2_cand96"]).reshape(B, -1).all(1)
        ok &= np.isfinite(pred["ck2_score_logit"]).reshape(B, -1).all(1)
        ok &= np.isfinite(pred["ck2_lat_traj"]).reshape(B, -1).all(1)
        for src, dst in PRED_TO_PACKED.items():
            self.A[dst][rows] = np.asarray(pred[src]).astype(self.A[dst].dtype)
        self.A["status"][rows] = status
        self.A["gt_traj"][rows] = gt
        self.A["sub_traj"][rows] = traj
        for src, dst in PRED_TO_INFER.items():
            self.I[dst][rows] = np.asarray(pred[src]).astype(self.I[dst].dtype)
        self.A["ok"][rows] = ok
        self.I["done"][rows] = ok
        return ok

    def finalize_meta(self, extra: Dict[str, Any]) -> None:
        self.flush()
        ok = np.asarray(self.A["ok"], bool)
        U.write_json(self.P.packed / "meta.json", dict(
            split=self.P.split, n=self.n, k=G_K, n_ok=int(ok.sum()), source="eval_e2e2 in-process dump (CKE2E2.infer)",
            layout="c = k*6 + v (v2 top-16 x id, a-1.0, a-0.5, a+0.5, l-0.5, l+0.5)", created=EV.kst(), **extra))
        np.save(self.P.packed / "ok_rows.npy", np.flatnonzero(ok).astype(np.int64))
        meta = U.read_json(self.P.infer / "meta.json", {}) or {}
        meta.update(run=str(self.ck_run or ""), which="last", arm="S", split=self.P.split, n=self.n, k=G_K, slope=0.0,
                    source="eval_e2e2 in-process (CKE2E2.infer, fp32, z_lon = 0)", **extra)
        U.write_json(self.P.infer / "meta.json", meta)


# ----------------------------------------------------------------------------------------------- dump
def check_pool(pr: Dict[str, Any], anchors=None) -> float:
    """cand96[:, ::6] == gather(anchors + offset, topk16(final)), cand_idx == topk, cand96[:, 0] == trajectory."""
    import torch
    final = pr["plan_final_rewards"].float()
    top = final.topk(K16, dim=-1).indices
    if anchors is None and "trajectory_anchors" in pr:
        anchors = pr["trajectory_anchors"]
    cand = pr["ck2_cand96"].float()
    if anchors is not None:
        anchors = anchors.float().to(final.device)
        ref = (anchors.reshape(1, -1, 8, 3) + pr["trajectory_offset"].float()).gather(
            1, top[:, :, None, None].expand(-1, -1, 8, 3))
        e = float((ref - cand[:, ::6]).abs().max())
        assert e < 1e-5, f"ck2_cand96[:, ::6] != gather(anchors + offset, topk) ({e})"
    assert torch.equal(pr["ck2_cand_idx"].long(), top), "ck2_cand_idx != topk(plan_final_rewards)"
    err1 = float((cand[:, 0] - pr["trajectory"].float()).abs().max())
    assert err1 < 1e-4, f"pool column 0 != submitted trajectory ({err1})"
    return err1


def run_dump_shard(P: Paths, ckpt: Path, tdf, shard: int, nshard: int, batch_size: int, workers: int) -> Dict:
    import torch
    from dataclasses import replace
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.script import run_aux_evaluation as RAE
    from tools.ck.data import common as CM
    from tools.ck.data import dump_v2 as DV

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis.strip() == "":
        dev = torch.device("cpu")
        log("no CUDA_VISIBLE_DEVICES: CPU dump (tests / tiny smoke only)")
    else:
        U.gpu_guard("cuda")
        dev = torch.device("cuda:0")
    key = U.read_json(P.packed / "cache_key.json") if (P.packed / "cache_key.json").is_file() else None
    if not key or key.get("ckpt") != str(ckpt):
        raise SystemExit(f"dump shard {shard}: {P.packed}/cache_key.json is missing or names another checkpoint "
                         f"({(key or {}).get('ckpt')} vs {ckpt}); shards run under stage_dump only")
    W = EvalRootWriter2(P, tdf, P.ck_run)
    pos = E1.shard_positions(tdf, shard, nshard)
    pos = pos[~W.done_rows()[pos]]
    log(f"dump {P.split} shard {shard}/{nshard}: todo {len(pos)} of {len(tdf)} on {dev} (ckpt {ckpt})")
    if len(pos) == 0:
        return {"shard": shard, "n_done": 0}
    t0 = time.time()
    agent = RAE._build_agent(P.hydra_cfg, ckpt, dev)
    if getattr(agent, "_ck_e2e2", None) is None:
        raise SystemExit("agent has no CK2 module: training config without ck_e2e2.enabled")
    sp = CM.split(P.split)
    todo_toks = [str(t) for t in tdf.token.to_numpy()[pos]]
    sf = instantiate(OmegaConf.load(CM.resolve_path(sp["scene_filter"])))
    need = set(tdf.log.to_numpy()[pos])
    sf = replace(sf, tokens=list(todo_toks), max_scenes=None)
    sf = replace(sf, log_names=sorted(set(sf.log_names or need) & need))
    loader = SceneLoader(data_path=Path(sp["navsim_logs"]), sensor_blobs_path=Path(sp["sensor_blobs"]),
                         scene_filter=sf, sensor_config=agent.get_sensor_config())
    miss = set(todo_toks) - set(loader.tokens)
    assert not miss, f"scene loader misses {len(miss)} tokens e.g. {sorted(miss)[:3]}"
    log(f"agent + scene loader ready in {time.time() - t0:.0f}s")
    try:
        anchors = agent.para_ssr_model.pts_bbox_head.anchor_planner.trajectory_anchors.detach().float().to(dev)
    except AttributeError:
        anchors = None
    row_of = {t: int(p) for t, p in zip(todo_toks, pos)}
    ds = DV.FeatDS(loader, agent.get_feature_builders(), todo_toks)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=workers, collate_fn=DV.collate,
                                     pin_memory=dev.type == "cuda", prefetch_factor=2 if workers else None)
    chk_path = P.infer / "check_bev.npz"
    chk_bev, chk_rows = [], []
    n, n_bad, max_top1, t1 = 0, 0, 0.0, time.time()
    for btok, feats, gt in dl:
        fg = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else v.dtype, non_blocking=True)
              for k, v in feats.items()}
        with torch.inference_mode():
            pr = agent(fg)
            if "ck2_cand96" not in pr:       # infer_outputs off in the stored config: call the hook directly
                pr = dict(pr)
                pr.update(agent._ck_e2e2.infer(agent.ck_student, fg, pr))
        max_top1 = max(max_top1, check_pool(pr, anchors))
        rows = np.array([row_of[t] for t in btok], np.int64)
        npd = {k: pr[k].detach().float().cpu().numpy() if pr[k].is_floating_point() else pr[k].cpu().numpy()
               for k in list(PRED_TO_PACKED) + list(PRED_TO_INFER)}
        ok = W.write(rows, npd, fg["status_feature"].float().cpu().numpy(), gt.numpy().astype(np.float32),
                     pr["trajectory"].float().cpu().numpy())
        n_bad += int((~ok).sum())
        if shard == 0 and not chk_path.is_file() and len(chk_rows) < N_CHECK_BEV:
            from navsim.agents.para_ssr.ck.online import bev_sgrid
            g = bev_sgrid(pr["bev_embed"].float()).cpu().numpy()
            take = min(N_CHECK_BEV - len(chk_rows), len(rows))
            chk_bev.append(g[:take])
            chk_rows += rows[:take].tolist()
            if len(chk_rows) >= min(N_CHECK_BEV, len(pos)):
                tmp = chk_path.with_name(f".check_bev.tmp{os.getpid()}.npz")
                with open(tmp, "wb") as f:
                    np.savez(f, rows=np.asarray(chk_rows, np.int64), bev=np.concatenate(chk_bev, 0).astype(np.float32))
                os.replace(tmp, chk_path)
        n += len(rows)
        if n % 400 < len(rows) or n == len(pos):
            W.flush()
            el = time.time() - t1
            log(f"{P.split} s{shard} {n}/{len(pos)} {n / max(el, 1e-6):.2f} tok/s top1_err {max_top1:.1e} not-ok {n_bad}")
    W.flush()
    meta = {"shard": shard, "nshard": nshard, "n_done": n, "n_not_ok": n_bad, "max_top1_err": max_top1,
            "seconds": round(time.time() - t0, 1), "gpu": vis, "ckpt": str(ckpt), "finished": EV.kst()}
    U.write_json(P.packed / f"meta_shard{shard}.json", meta)
    return meta


def stage_dump(P: Paths, ckpt: Path, a, ckpt_sha16: Optional[str] = None) -> None:
    tdf = E1.split_table(P.split, a.limit)
    gpus = [g for g in a.gpus.split(",") if g.strip()] if a.gpus else []
    for g in gpus:
        if g not in U.ALLOWED_GPUS:
            raise SystemExit(f"GPU {g} not allowed (0-3 only)")
    if a.shard is not None:
        run_dump_shard(P, ckpt, tdf, a.shard, a.nshard, a.batch_size, a.workers)
        return
    key = check_cache_key(P, cache_key(P, ckpt, ckpt_sha16 or U.sha256_file(ckpt), a.limit))
    if len(gpus) <= 1:
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        run_dump_shard(P, ckpt, tdf, 0, 1, a.batch_size, a.workers)
    else:
        EvalRootWriter2(P, tdf, P.ck_run)
        procs = []
        for i, g in enumerate(gpus):
            cmd = [PY, __file__, "--run-dir", str(P.run_dir), "--split", P.split, "--stage", "dump",
                   "--ckpt", str(ckpt), "--shard", str(i), "--nshard", str(len(gpus)), "--limit", str(a.limit),
                   "--batch-size", str(a.batch_size), "--workers", str(a.workers)]
            lf = open(P.eval / f"dump_{P.split}_shard{i}.log", "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONPATH=CK,
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            procs.append((i, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env), lf))
            log(f"dump shard {i}/{len(gpus)} on GPU {g} pid {procs[-1][1].pid}")
        bad = []
        for i, p, lf in procs:
            rc = p.wait()
            lf.close()
            if rc != 0:
                bad.append((i, rc))
        if bad:
            raise SystemExit(f"dump shards failed {bad} (see {P.eval}/dump_{P.split}_shard*.log)")
    W = EvalRootWriter2(P, tdf, P.ck_run)
    n_done = int(W.done_rows().sum())
    W.finalize_meta({"ckpt": str(ckpt), "ckpt_sha16": key["ckpt_sha16"], "hydra_sha16": key["hydra_sha16"],
                     "code_sha16": key["code_sha16"], "limit": a.limit, "n_done": n_done})
    log(f"dump {P.split}: {n_done}/{len(tdf)} rows done -> {P.root}")
    if n_done < len(tdf):
        raise SystemExit(f"dump incomplete: {n_done}/{len(tdf)}")


# ----------------------------------------------------------------------------------------------- label
def stage_label(P: Paths, a) -> None:
    if not 1 <= a.workers_label <= 48:
        raise SystemExit("--workers-label must be in [1, 48]")
    rows = P.packed / "ok_rows.npy"
    if not rows.is_file():
        raise SystemExit(f"{rows} missing: run --stage dump first")
    env = dict(os.environ, CK_DATA_ROOT=str(P.root), PYTHONPATH=CK, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    lab = [PY, f"{CK}/tools/ck/data/label_cands.py", "--split", P.split, "--root", str(P.root), "--workers",
           str(a.workers_label), "--rows", str(rows)]
    jobs = [(f"pool96_{P.name}", lab + ["--name", f"pool96_{P.name}", "--traj", str(P.packed / "cand96.npy")],
             P.packed / "cand96.npy"),
            (f"lat96_{P.name}", lab + ["--name", f"lat96_{P.name}", "--traj", str(P.infer / "lat_traj.npy")],
             P.infer / "lat_traj.npy")]
    for name, cmd, traj in jobs:
        if (P.labels / name / "labels.npy").is_file() and (P.labels / name / "meta.json").is_file():
            from tools.ck.data import common as CM
            have = (U.read_json(P.labels / name / "meta.json") or {}).get("traj_sha16")
            cur = CM.sha16_array(np.load(traj, mmap_mode="r"))
            if have != cur:
                raise SystemExit(f"labels {P.split}/{name} were scored on trajectories sha16 {have}, the current "
                                 f"{traj.name} has {cur} (stale labels of another dump); rerun with --fresh")
            log(f"labels {P.split}/{name} present (traj sha16 {cur} = current {traj.name}): skipped")
            continue
        log(f"label {P.split}/{name}: {' '.join(cmd[2:])}")
        with open(P.eval / f"label_{P.split}_{name}.log", "a") as lf:
            rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not (P.labels / name / "labels.npy").is_file():
            raise SystemExit(f"label_cands {name} failed rc {rc} (see {P.eval}/label_{P.split}_{name}.log)")


# ----------------------------------------------------------------------------------------------- infer_check
def stage_infer_check(P: Paths, a, tol_logit: float = 3e-2, tol_traj: float = 2e-3) -> Dict:
    import torch
    from navsim.agents.para_ssr.ck.model import load_ck
    from navsim.agents.para_ssr.ck.online2 import lateral_decode, student_forward2
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    f = P.infer / "check_bev.npz"
    if not f.is_file():
        raise SystemExit(f"{f} missing (dump shard 0 stores it)")
    src = (U.read_json(P.ck_run / "config.json") or {}).get("source_ckpt_sha16")
    dumped = (U.read_json(P.infer / "meta.json") or {}).get("ckpt_sha16")
    if src and dumped and src != dumped:
        raise SystemExit(f"infer_check: the extracted ck_run is from checkpoint sha16 {src}, the {P.split} dump from "
                         f"{dumped}; rerun --stage extract with the dump's --ckpt (or --fresh)")
    z = np.load(f)
    rows, bev = z["rows"], z["bev"]
    gpus = [g for g in (a.gpus or "").split(",") if g.strip()]
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        U.gpu_guard("cuda")
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")
    net, cfg = load_ck(P.ck_run, "last", str(dev))
    cand = torch.from_numpy(np.load(P.packed / "cand96.npy", mmap_mode="r")[rows].astype(np.float32)).to(dev)
    status = torch.from_numpy(np.load(P.packed / "status.npy", mmap_mode="r")[rows].astype(np.float32)).to(dev)
    with torch.no_grad():
        o = student_forward2(net, torch.from_numpy(bev).to(dev), cand, status)
        lat = lateral_decode(cand, o["w_lat"], ego_inputs(status)[0], 0.0)
    got = {"score_logit": o["score_logit"], "w_lat": o["w_lat"], "e_lat": lat["e_lat"][..., 2:],
           "lat_traj": lat["traj"]}
    err = {}
    for k, v in got.items():
        ref = np.load(P.infer / f"{k}.npy", mmap_mode="r")[rows].astype(np.float32)
        err[k] = float(np.nanmax(np.abs(v.float().cpu().numpy() - ref)))
    tol = {k: (tol_logit if "logit" in k else tol_traj) for k in err}
    res = {"rows": rows.tolist(), "max_abs_err": err, "tol": tol, "device": str(dev),
           "pass": all(err[k] <= tol[k] for k in err), "ck_run": str(P.ck_run), "checked": EV.kst()}
    U.write_json(P.infer / "infer_check.json", res)
    log(f"infer_check {P.split}: pass={res['pass']} {err}")
    if not res["pass"]:
        raise SystemExit("infer_check failed: extracted CK2 run does not reproduce the in-process outputs")
    return res


# ----------------------------------------------------------------------------------------------- eval / summary
def stage_eval(P: Paths, a) -> Path:
    """navtrain_val: choice + frozen spec under out_name(split, sel_w, offset).  navtest: --fix-from (default: the val
    result of the same --sel-w / --calib-offset, i.e. 'default' / OFF when unset) restores the frozen spec; the out
    dir follows the RESTORED weights / offset (eval_ck2.run_eval)."""
    fix = a.fix_from
    sel_w = getattr(a, "sel_w", None) or None
    cal = getattr(a, "calib_offset", "") or ""
    if P.split == "navtest" and not fix:
        cand = P.root / "eval" / P.name / EV.out_name("navtrain_val", sel_w or "default", EV.read_offset(cal)) / \
            "metrics.json"
        if not cand.is_file():
            raise SystemExit(f"navtest needs --fix-from (beta / variant set / lateral mode are chosen on navtrain_val "
                             f"only); run the navtrain_val eval first (looked for {cand})")
        fix = str(cand)
    res = EV.run_eval(P.root, P.name, P.split, None, fix, a.bootstrap, cal or None,
                      choose_lat=getattr(a, "choose_lat", "on"), sel_w=sel_w)
    return Path(res["out_dir"]) / "metrics.json"


def _pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.2f}"


def stage_summary(run_dir: Path, sel_w=None, offset=None) -> Dict:
    """summary of the out_name(split, sel_w, offset) result dirs (default: 'default' weights, no calibration).  Each
    split's checkpoint is taken from its own metrics.json ('model'); val and navtest of different checkpoints are
    refused (SystemExit: a stale navtest result after a new val, or the reverse)."""
    name = Path(run_dir).name
    ev = Path(run_dir) / "eval"
    out: Dict[str, Any] = {"run": name, "created": EV.kst(), "ckpt_pin": U.read_json(ev / "ckpt.json"), "splits": {},
                           "sel_w": EV.S2.sel_w_name(sel_w or "default"), "calib_offset_values": offset}
    ck_by: Dict[str, Any] = {}
    for split in ("navtrain_val", "navtest"):
        mp = ev / "root" / "eval" / name / EV.out_name(split, sel_w or "default", offset) / "metrics.json"
        if not mp.is_file():
            continue
        m = json.loads(mp.read_text())
        ck_by[split] = {k: (m.get("model") or {}).get(k) for k in ("ckpt", "ckpt_sha16", "ep_target", "code_sha16")}
        keep = {"v2", "v2_lat", "oracle16", "oracle96", "oracle96_lat", m.get("representative"),
                "sel b=1 set=all lat=on"}
        rows = {r["key"]: r for r in m["rows"] if r["key"] in keep}
        out["splits"][split] = {"n_eval": m["n_eval"], "n_total": m["n_total"], "n_logs": m["n_logs"],
                                "choice": m.get("best") or m.get("fixed_from_val"),
                                "representative": m.get("representative"), "rows": rows,
                                "infer_check": U.read_json(ev / "root/infer" / name / split / "infer_check.json"),
                                "model": ck_by[split],
                                "eval_spec_sha16": (m.get("eval_spec") or {}).get("spec_sha16")}
    out["ckpt_by_split"] = ck_by
    if len(ck_by) == 2:
        a_, b_ = ck_by["navtrain_val"], ck_by["navtest"]
        if a_.get("ckpt_sha16") != b_.get("ckpt_sha16"):
            raise SystemExit(f"summary: navtrain_val was evaluated on ckpt {a_.get('ckpt')} ({a_.get('ckpt_sha16')}), "
                             f"navtest on {b_.get('ckpt')} ({b_.get('ckpt_sha16')}); rerun the stale split (--fresh)")
        out["code_sha16_same"] = a_.get("code_sha16") == b_.get("code_sha16")
        if not out["code_sha16_same"]:
            log(f"summary: warning: val / navtest dumps from different dump-code sha16 ({a_.get('code_sha16')} vs "
                f"{b_.get('code_sha16')})")
    out["ckpt"] = next(iter(ck_by.values()), {}) if ck_by else None
    U.write_json(ev / "summary.json", out)
    L = [f"# CK2 e2e 평가: {name}", "", f"생성 {out['created']}. ckpt {(out.get('ckpt') or {}).get('ckpt')} (sha16 "
         f"{(out.get('ckpt') or {}).get('ckpt_sha16')}, 두 split 같음 확인). 선택 가중치 {out['sel_w']}, 보정 offset "
         f"{out['calib_offset_values'] or 'OFF'}. 공식 채점(score_token, "
         "LQR) [실측]. β·변형 집합·측방 모드는 navtrain_val에서만 골랐다. ★ = 대표 행. oracle·β=1 행은 서술용. 단위 0-100.", ""]
    for split, sp in out["splits"].items():
        L += [f"## {split}", "", f"토큰 {sp['n_eval']} / {sp['n_total']}, log {sp['n_logs']}. 선택: {sp['choice']}", "",
              "| 행 | PDMS | ΔPDMS vs v2 [95% CI] | NC | DAC | EP | TTC | C | 앞차 감속 NC/TTC 실패 (n) | 바뀐 비율 | 측방 적용 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, r in sp["rows"].items():
            mark = "★ " if k == sp["representative"] else ""
            lead = f"{_pct(r.get('lead_fail_nc_ttc'))} ({r.get('lead_n')})" if r.get("lead_n") else "-"
            L.append(f"| {mark}{k} | {_pct(r['pdms'])} | {EV.fmt_ci(r.get('d_pdms'))} | {_pct(r['nc'])} | "
                     f"{_pct(r['dac'])} | {_pct(r['ep'])} | {_pct(r['ttc'])} | {_pct(r['comfort'])} | {lead} | "
                     f"{100 * r['frac_changed']:.1f} | {100 * r['frac_lat']:.1f} |")
        L.append("")
    (ev / "summary.md").write_text("\n".join(L) + "\n")
    log(f"summary -> {ev / 'summary.md'}")
    return out


# ----------------------------------------------------------------------------------------------- main
def fresh(P: Paths) -> None:
    """remove this split's cached outputs (packed incl. cache_key.json, infer, labels, eval out dirs) and the ckpt pin."""
    outs = sorted(P.out.parent.glob(f"{P.split}*")) if P.out.parent.is_dir() else []
    for d in [P.packed, P.infer, P.labels] + [d for d in outs if d.name == P.split or d.name.startswith(P.split + "__")]:
        if d.exists():
            log(f"--fresh: removing {d}")
            shutil.rmtree(d)
    if P.pin.is_file():
        log(f"--fresh: removing the checkpoint pin {P.pin} ({(U.read_json(P.pin) or {}).get('path')})")
        P.pin.unlink()


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", required=True, choices=["navtrain_val", "navtest"])
    ap.add_argument("--ckpt", default="", help="last (default) | epoch:<E> | <path>; pinned in <RUN>/eval/ckpt.json "
                    "(no --ckpt: the pin must be the current 'last', same file sha16)")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--limit", type=int, default=0, help="first N split tokens (smoke)")
    ap.add_argument("--gpus", default="0", help="dump GPUs (0-3), one process each; '' = CPU")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6, help="dataloader workers per dump process")
    ap.add_argument("--workers-label", type=int, default=48)
    ap.add_argument("--fix-from", default="")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--calib-offset", default="",
                    help="'' (default): OFF on navtrain_val (user), inherited from the frozen val spec on navtest | "
                         "none | <json with 5 logit offsets>")
    ap.add_argument("--choose-lat", default="on", choices=["on", "any"],
                    help="navtrain_val choice: on (default, user: lateral applied after selection) | any")
    ap.add_argument("--sel-w", default=None, choices=tuple(EV.S2.SEL_W_SETS),
                    help="selection weight set: default (= SEL_W) | noim | plugin (0, 1, 1, 1); comparison only; "
                         "unset: 'default' on navtrain_val, inherited from the frozen val spec on navtest")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--nshard", type=int, default=1, help=argparse.SUPPRESS)
    return ap


def short_tmpdir(limit: int = 75) -> None:
    """DataLoader workers pass tensors through AF_UNIX sockets under TMPDIR (pymp-*/listener-*, +32 chars, limit 107):
    a long TMPDIR makes the bind fail in a worker thread and the dump hangs.  Such a TMPDIR is replaced by /tmp (the
    dump shards inherit os.environ)."""
    t = os.environ.get("TMPDIR", "")
    if len(t) > limit:
        import tempfile
        log(f"TMPDIR {t!r} has {len(t)} chars (> {limit}): using /tmp (AF_UNIX socket path limit)")
        os.environ["TMPDIR"] = "/tmp"
        tempfile.tempdir = None


def main(argv=None) -> int:
    a = get_parser().parse_args(argv)
    short_tmpdir()
    P = Paths(a.run_dir, a.split)
    stages = list(STAGES) if a.stage == "all" else [s.strip() for s in a.stage.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; known {STAGES}")
    if a.fresh and a.shard is None:
        fresh(P)
    P.eval.mkdir(parents=True, exist_ok=True)
    ckpt, ckpt_sha, pin_rec = None, None, None
    if any(s in stages for s in ("extract", "dump")):
        if a.shard is not None:
            ckpt = Path(a.ckpt)
        else:
            ckpt, ckpt_sha, pin_rec = resolve_pin(P, a.ckpt or None)
    elif a.ckpt:
        log(f"warning: --ckpt {a.ckpt} ignored (stages {stages} use the existing dump; its checkpoint is checked by "
            f"infer_check / the eval spec)")
    if "dump" in stages and a.shard is None:      # before 'extract' rewrites ck_run / the pin: refuse a foreign cache
        check_cache_key(P, cache_key(P, ckpt, ckpt_sha, a.limit))
    if pin_rec is not None:
        U.write_json(P.pin, pin_rec)
    for s in stages:
        t0 = time.time()
        if s == "extract":
            extract_ck2(ckpt, P.ck_run, P.name, hydra_ck2_cfg(P))
        elif s == "dump":
            stage_dump(P, ckpt, a, ckpt_sha)
        elif s == "label":
            stage_label(P, a)
        elif s == "infer_check":
            stage_infer_check(P, a)
        elif s == "eval":
            stage_eval(P, a)
        elif s == "summary":
            stage_summary(P.run_dir, getattr(a, "sel_w", None), EV.read_offset(a.calib_offset or ""))
        log(f"stage {s} done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
