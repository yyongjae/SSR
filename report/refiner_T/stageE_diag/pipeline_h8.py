#!/usr/bin/env python3
"""H8 (evaluation / pipeline differences between E0 + R_* and E2) -- inference side.

  ext    : tools/refiner/refine_external_drafts.predict (the E0 + R_* path: packed ego, stage-T loader) on a drafts pkl,
           optionally with autocast OFF (--fp32; the stored e0_teacher_refine arms ran autocast fp16 as stage T).
  stagee : the frozen teacher refiners run through the STAGE-E code path (navsim/agents/para_ssr/refiner/e2e.py:
           StageE.teachers-style load (train_refiner.load_run_model 'best', .float().eval()), StageE._run,
           StageE._decode(slope 0), ego = e2e.ego_inputs(status_feature from the E2 dump), BEV = cache.load_bev(token,
           s_grid=True) as float16 -> .float() as GTLoader / StageE.loss, fp32, no autocast) on a drafts pkl.  Same
           function the KD targets were computed with during E2 training (navtest caches instead of navtrain).
  stack  : stack several pred.npz tau1 into one [N, K, 8, 3] drafts npz for score_trajectories.py.
Outputs under /home/external-user/ssd/yongjae_refiner/stageE_diag/pipeline/.  Nothing existing is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

REPO = Path("/home/external-user/yongjae/SSR")
for p in (str(REPO), str(REPO / "tools/refiner")):
    if p not in sys.path:
        sys.path.insert(0, p)

import refine_external_drafts as REX  # noqa: E402
import train_refiner as TR  # noqa: E402
from navsim.agents.para_ssr.refiner import data as RD  # noqa: E402
from navsim.agents.para_ssr.refiner import e2e as E2E  # noqa: E402
from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache  # noqa: E402

DIAG = Path("/home/external-user/ssd/yongjae_refiner/stageE_diag")
DUMP = DIAG / "e2_navtest_dump.npz"


def cmd_ext(a):
    if a.fp32:
        orig = TR.load_run_model

        def patched(run_dir, ckpt="best", device="cpu"):
            net, cfg = orig(run_dir, ckpt, device)
            cfg = dict(cfg)
            cfg["amp"] = 0
            return net, cfg
        TR.load_run_model = patched
        REX.TR.load_run_model = patched
    ns = SimpleNamespace(run=a.run, drafts_pkl=a.drafts_pkl, out=a.out, split="navtest", ckpt="best", gpu=a.gpu,
                         packed_root=str(RD.DATA_ROOT / "packed"), teacher_root=None, resmap_root=None,
                         tokens_per_batch=32, loader_workers=2, limit=None)
    REX.predict(ns)
    m = json.loads((Path(a.out) / "predict_meta.json").read_text())
    m["h8_fp32_forced"] = bool(a.fp32)
    (Path(a.out) / "predict_meta.json").write_text(json.dumps(m, indent=1))


@torch.no_grad()
def cmd_stagee(a):
    dev = torch.device(f"cuda:{a.gpu}")
    tr = E2E.train_refiner_module()
    net, cfg = tr.load_run_model(Path(a.run), "best", "cpu")
    net.requires_grad_(False)
    net = net.float().eval().to(dev)
    arm = cfg["arm"]
    cache = RD.TeacherCache.for_subset("navtest") if arm == "T" else ResmapCache.for_subset("navtest")
    drafts = REX.load_drafts_pkl(a.drafts_pkl)
    z = np.load(DUMP)
    toks = z["tokens"].astype(str)
    sf = torch.as_tensor(z["status_feature"]).float()
    sel = np.arange(len(toks))[:: max(1, int(a.every))]      # equivalence check on every n-th (sorted) token
    toks, sf = toks[sel], sf[sel]
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(4)
    helper = SimpleNamespace()
    B = 32
    rec = {k: [] for k in ("tau0", "tau1", "z_lon", "w_lat", "c_lon", "e_lat", "p_g")}
    t0 = time.time()
    for s in range(0, len(toks), B):
        tt = toks[s:s + B]
        tau = torch.as_tensor(np.stack([drafts[t] for t in tt])).float().to(dev)
        bev = torch.as_tensor(np.stack(list(pool.map(lambda t: np.asarray(cache.load_bev(t, s_grid=True), np.float16),
                                                           tt)))).to(dev)
        ego = E2E.ego_inputs(sf[s:s + B].to(dev))
        out = E2E.StageE._run(net, bev.float(), tau, ego)
        dec = E2E.StageE._decode(helper, tau, out, ego[0], 0.0)
        rec["tau0"].append(tau.cpu().numpy())
        rec["tau1"].append(dec["traj"].float().cpu().numpy())
        rec["z_lon"].append(out["z_lon"][:, 0].float().cpu().numpy())
        rec["w_lat"].append(out["w_lat"][:, 0].float().cpu().numpy())
        rec["c_lon"].append(dec["c_lon"].float().cpu().numpy())
        rec["e_lat"].append(dec["e_lat"].float().cpu().numpy())
        rec["p_g"].append(torch.sigmoid(out["gate_logit"].float()).cpu().numpy())
        if (s // B) % 20 == 0:
            print(f"[stagee] {s}/{len(toks)} {time.time() - t0:.0f}s", flush=True)
    R = {k: np.concatenate(v) for k, v in rec.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "pred.npz", tokens=toks, **R)
    meta = dict(kind="stage-E code path teacher", run=a.run, arm=arm, cache=str(cache.root), drafts_pkl=a.drafts_pkl,
                fp32=True, ego="e2e.ego_inputs(dump status_feature)", every=int(a.every), n=int(len(toks)), sec=round(time.time() - t0, 1),
                created=time.strftime("%Y-%m-%dT%H:%M:%S"), gpu=a.gpu)
    (out / "predict_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


def cmd_stack(a):
    """--preds name=path/pred.npz ... -> drafts npz [N, K, 8, 3] (order of --preds) + tokens.parquet + arms.json."""
    names, arrs, tok_ref = [], [], None
    for spec in a.preds:
        n, p = spec.split("=", 1)
        z = np.load(p)
        t = z["tokens"].astype(str)
        tau1 = z["tau1"]
        tau1 = tau1[:, 0] if tau1.ndim == 4 else tau1
        o = np.argsort(t)
        t, tau1 = t[o], tau1[o]
        if tok_ref is None:
            tok_ref = t
        assert np.array_equal(tok_ref, t), n
        names.append(n)
        arrs.append(tau1.astype(np.float32))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "drafts.npz", tokens=tok_ref, drafts=np.stack(arrs, 1))
    tp = pd.read_parquet(DIAG / "tokens.parquet")
    tp = tp.set_index("token").loc[tok_ref].reset_index()
    tp.to_parquet(out / "tokens.parquet", index=False)
    (out / "arms.json").write_text(json.dumps({"k": {i: n for i, n in enumerate(names)}, "preds": a.preds}, indent=1))
    print(names, len(tok_ref))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("ext")
    e.add_argument("--run", required=True)
    e.add_argument("--drafts-pkl", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--gpu", type=int, default=0)
    e.add_argument("--fp32", action="store_true")
    s = sp.add_parser("stagee")
    s.add_argument("--run", required=True)
    s.add_argument("--drafts-pkl", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--gpu", type=int, default=0)
    s.add_argument("--every", type=int, default=1)
    k = sp.add_parser("stack")
    k.add_argument("--preds", nargs="+", required=True)
    k.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    {"ext": cmd_ext, "stagee": cmd_stagee, "stack": cmd_stack}[a.cmd](a)


if __name__ == "__main__":
    main()
