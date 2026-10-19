#!/usr/bin/env python
"""CPU cost / memory benchmark of the refiner-v3 temporal branch vs the CK2 trunk (no GPU, no files written).

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/yongjae/SSR-v3 \
    nice -n 10 /venv/ssr/bin/python tools/ck/v3/bench_temporal_model.py [--tokens 2] [--k-obj 64] [--reps 2]

One CK2 train step layout per token (ck2T10 recipe): cand = 32 anchors + extra = 80 variants, i.e. two
trunk.candidates calls (Kc = 32, 80) sharing one scene; the temporal branch adds scene() once and features() per call
(through the SPEC 4.5 hook, tests/v3_temporal_testlib.candidates_hooked).  Reported per arm (off / a learned / b CA):
  fwd_s, bwd_s           wall time (1 thread) of forward and backward of sum(score) (+ forecast loss in arm a)
  saved_mb               bytes of the tensors autograd saves for backward (float32 on CPU; parameters excluded) ->
                         activation-memory proxy of the GPU step (the trunk / st_enc run fp16 under autocast on GPU,
                         the raster / forecaster fp32)
  branch_* breakdown     scene / raster+gap / st_enc times
Prints one JSON object.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import v3_temporal_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck import model as M  # noqa: E402
from navsim.agents.para_ssr.ck.temporal.branch import TemporalBranch  # noqa: E402
from navsim.agents.para_ssr.ck.temporal.config import TemporalCfg  # noqa: E402
from navsim.agents.para_ssr.ck.temporal.loss import forecast_loss  # noqa: E402
from navsim.agents.para_ssr.refiner import e2e as E  # noqa: E402
from navsim.agents.para_ssr.refiner.corridor import corridor_geometry  # noqa: E402

K_CAND, K_EXTRA = 32, 80


class Saved:
    def __init__(self, exclude):
        self.ex = exclude
        self.st = {}

    def pack(self, t):
        try:
            s = t.untyped_storage()
            p = s.data_ptr()
            if p not in self.ex:
                self.st[p] = s.nbytes()
        except Exception:
            pass
        return t

    @property
    def mb(self):
        return sum(self.st.values()) / 2 ** 20


def inputs(T, k_obj, seed=0):
    g = np.random.default_rng(seed)
    toks = [TL.rand_token(g, k_obj, n_distract=k_obj) for _ in range(T)]
    obj, mask, fv = TL.pack_tokens(toks)
    cand = TL.rand_cands(T, K_CAND, seed=seed).reshape(T, K_CAND, 8, 3)
    extra = TL.rand_cands(T, K_EXTRA, seed=seed + 1).reshape(T, K_EXTRA, 8, 3)
    sf = torch.zeros(T, 8)
    sf[:, 1] = 1.0
    sf[:, 4] = 8.0
    bev = torch.randn(T, 256, 50, 100, generator=torch.Generator().manual_seed(seed))
    return dict(obj=obj, mask=mask, fv=fv, cand=cand, extra=extra, status=sf, bev=bev)


def step(net, br, x, timers):
    T = x["cand"].shape[0]
    v0, a0, eds, cmd = E.ego_inputs(x["status"])
    feat, mem = net.trunk.scene(x["bev"], T)
    ctx = None
    if br is not None:
        t = time.perf_counter()
        ctx = br.scene({"obj_in": x["obj"], "obj_mask": x["mask"], "frame_valid": x["fv"]})
        timers["scene"] += time.perf_counter() - t
    loss = 0.0
    for c in (x["cand"], x["extra"]):
        K = c.shape[1]
        hook = None
        if br is not None:
            def hook(geom, flat, K=K):
                t = time.perf_counter()
                R, G = br.raster_and_gap(ctx, geom, flat, T, K)
                t1 = time.perf_counter()
                Hh = br.st_enc(R)
                st = br.st_proj(Hh.permute(0, 2, 1, 3).reshape(R.shape[0], 48, -1))
                ud = br.gap_proj(G)
                timers["raster_gap"] += t1 - t
                timers["st_enc"] += time.perf_counter() - t1
                return st, ud
        o = TL.candidates_hooked(net.trunk, feat, mem, c.float(), v0, a0, eds, cmd, hook)
        loss = loss + net.score(o, T, K).sum()
    if ctx is not None and ctx.fc_raw is not None:
        y = torch.randn(T, 64, 10, 3) * 5
        lf, _ = forecast_loss(ctx.fc_raw, y, torch.ones(T, 64, 10, dtype=torch.int8), x["mask"][:, 0])
        loss = loss + lf
    return loss, ctx


def run(arm, x, reps):
    rng = np.random.default_rng(3)
    norm = (rng.normal(0, 0.3, 256).astype(np.float32), rng.uniform(0.5, 2.0, 256).astype(np.float32))
    net = M.CKNet("T", seed=0, norm=norm).train()
    br = None
    if arm != "off":
        cfg = TemporalCfg(on=True, forecaster={"a": "learned", "b": "ca"}[arm], n_modes=6 if arm == "a" else 1)
        br = TemporalBranch(cfg, 192, 0, ca_b=None if arm == "a" else torch.full((2, 10), 0.5)).train()
        with torch.no_grad():                                    # non-zero projections: realistic backward
            for m in (br.st_proj, br.gap_proj):
                m.weight.normal_(0, 0.02)
    params = {p.data_ptr() for p in net.parameters()} | ({p.data_ptr() for p in br.parameters()} if br else set())
    res = []
    for r in range(reps + 1):
        timers = {"scene": 0.0, "raster_gap": 0.0, "st_enc": 0.0}
        sv = Saved(params)
        t0 = time.perf_counter()
        with torch.autograd.graph.saved_tensors_hooks(sv.pack, lambda t: t):
            loss, ctx = step(net, br, x, timers)
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        net.zero_grad(set_to_none=True)
        if br is not None:
            br.zero_grad(set_to_none=True)
        if r:                                                    # first run = warm-up
            res.append(dict(fwd_s=t1 - t0, bwd_s=t2 - t1, saved_mb=sv.mb, **{f"branch_{k}_s": v for k, v in timers.items()},
                            n_obj=None if ctx is None else ctx.n_obj))
    out = {k: float(np.mean([d[k] for d in res])) for k in res[0] if k != "n_obj"}
    out["n_obj"] = res[0]["n_obj"]
    if br is not None:
        out.update(br.param_counts())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2)
    ap.add_argument("--k-obj", type=int, default=64, help="t0 objects per token (past frames fill to 64)")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(1)
    x = inputs(a.tokens, a.k_obj)
    res = {"tokens": a.tokens, "cands_per_token": K_CAND + K_EXTRA, "k_obj": a.k_obj, "threads": 1}
    for arm in ("off", "a", "b"):
        res[arm] = run(arm, x, a.reps)
    for arm in ("a", "b"):
        o, r = res["off"], res[arm]
        r["rel_step_time"] = (r["fwd_s"] + r["bwd_s"]) / (o["fwd_s"] + o["bwd_s"])
        r["extra_saved_mb_per_token"] = (r["saved_mb"] - o["saved_mb"]) / a.tokens
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
