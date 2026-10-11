#!/usr/bin/env python
"""GPU step-time breakdown of the refiner-v3 temporal branch on ONE real ck2T10-layout batch (Integrate smoke tool;
short: about 1-2 min on one GPU; writes only the JSON given by --out).

  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/yongjae/SSR-v3 \
    nice -n 10 /venv/ssr/bin/python -B tools/ck/v3/bench_gpu_step.py --pack-root D3/smoke/objpack \
    --ca-scales D3/smoke/ca_scales_SMOKE_limit2048.json --out D3/smoke/bench_gpu_step.json

Batch: CK2Dataset(navtrain_train, bev T, n_var 80, gt True, first --tokens rows, objects aug on) collated once and
reused (8 tokens = one rank of the Step-1 recipe).  Per arm (off / a learned / b CA): mean wall time of the full
train-step compute (train_ck2.forward + lateral_decode + compute_loss + scaled backward, fp16 autocast, CUDA
synchronised) over --iters after --warmup, peak memory; for a / b the branch components (scene, raster + gap of the
32 anchors and of the 80 variants, st_enc + projections) timed separately; optional torch.profiler table (--profile).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = str(Path(__file__).resolve().parents[3])
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def sync():
    torch.cuda.synchronize()


def timed(fn, iters: int, warmup: int):
    for _ in range(warmup):
        fn()
    sync()
    t = time.time()
    for _ in range(iters):
        fn()
    sync()
    return (time.time() - t) / iters


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-root", required=True)
    ap.add_argument("--ca-scales", required=True)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--arms", default="off,a,b")
    ap.add_argument("--profile", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from navsim.agents.para_ssr.ck import constants as Cn
    from navsim.agents.para_ssr.ck.model import build_ck
    from navsim.agents.para_ssr.ck.temporal.config import TemporalCfg
    from navsim.agents.para_ssr.refiner.adapters import load_norm
    from tools.ck import train_ck as TC
    from tools.ck import train_ck2 as T2
    from tools.ck.data.ck2_dataset import CK2Dataset
    from tools.ck.data.ck_dataset import collate_ck
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    cfgs = {"off": TemporalCfg(),
            "a": TemporalCfg(on=True, forecaster="learned", n_modes=6, pack_root=a.pack_root, ca_scales=a.ca_scales),
            "b": TemporalCfg(on=True, forecaster="ca", n_modes=1, pack_root=a.pack_root, ca_scales=a.ca_scales)}
    ds = CK2Dataset("navtrain_train", bev="T", n_var=80, gt=True, limit=a.tokens,
                    var_name="var_separate_sampler16_accstraight",
                    objects={"cfg": cfgs["a"].to_dict(), "aug": True, "targets": True})
    batch = TC.to_dev(collate_ck([ds[i] for i in range(len(ds))]), dev)
    mean, std, _ = load_norm(Path("/home/external-user/ssd/yongjae_refiner/ck/ck2/train/ck2T10/norm.npz"))
    res = {"tokens": a.tokens, "n_obj_t0_valid": int(batch["obj_mask"][:, 0].sum()), "arms": {}}
    slope = Cn.LON_ST_SLOPE["train"]
    for arm in a.arms.split(","):
        tc = cfgs[arm]
        args = T2.resolve_args(T2.get_parser().parse_args(
            ["--arm", "T", "--run", "bench", "--sur-groups", "near_mid", "--score-keys", "nc,ttc,ep,comfort"]
            + (["--temporal", "1", "--t-forecaster", tc.forecaster, "--t-pack-root", a.pack_root, "--t-ca-scales",
                a.ca_scales] if tc.on else [])))
        torch.manual_seed(0)
        net = build_ck("T", seed=0, norm=(mean, std), **({"temporal": tc} if tc.on else {})).to(dev)
        T2.freeze_unused(net)
        net.train()
        scaler = torch.amp.GradScaler("cuda")

        def step():
            out = T2.forward(net, batch, True, dev, True, tc)
            corr = T2.lateral_decode(batch["cand"], out["w_lat"], out["ego"][0], slope)
            loss, _ = T2.compute_loss(out, corr, batch, args)
            scaler.scale(loss).backward()
            net.zero_grad(set_to_none=True)

        torch.cuda.reset_peak_memory_stats()
        r = {"step_s": timed(step, a.iters, a.warmup), "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
        if tc.on:
            br = net.temporal
            objs = {k: batch[k] for k in ("obj_in", "obj_mask", "frame_valid")}
            with torch.no_grad():
                ctx = br.scene(objs)
            r["n_obj"] = ctx.n_obj
            r["scene_fwd_s"] = timed(lambda: br.scene(objs), a.iters, 1)
            # geometry of the two candidate sets, as CKTrunk.candidates builds it
            from navsim.agents.para_ssr.refiner.corridor import corridor_geometry
            for name, tau in (("cand32", batch["cand"].float()), ("var80", batch["var_traj"].float())):
                T, K = tau.shape[:2]
                flat = tau.reshape(T * K, 8, 3)
                geom = corridor_geometry(flat)
                r[f"raster_gap_{name}_s"] = timed(lambda: br.raster_and_gap(ctx, geom, flat, T, K), a.iters, 1)
                R, G = br.raster_and_gap(ctx, geom, flat, T, K)

                def enc():
                    with torch.autocast("cuda", dtype=torch.float16):
                        H = br.st_enc(R)
                        st = H.permute(0, 2, 1, 3).reshape(R.shape[0], 48, -1)
                        return br.st_proj(st), br.gap_proj(G)
                r[f"st_enc_{name}_fwd_s"] = timed(enc, a.iters, 1)
            if a.profile and arm == "a":
                from torch.profiler import ProfilerActivity, profile
                step()
                sync()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    step()
                    sync()
                tbl = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25)
                r["profile_top_cuda"] = tbl
                tbl2 = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
                r["profile_top_cpu"] = tbl2
                ev = prof.key_averages()
                r["n_kernel_launches"] = int(sum(e.count for e in ev if e.key in ("cudaLaunchKernel",
                                                                               "cudaLaunchKernelExC")))
        res["arms"][arm] = r
        del net
        torch.cuda.empty_cache()
        print(json.dumps({arm: {k: v for k, v in r.items() if not k.startswith("profile")}}), flush=True)
    off = res["arms"].get("off", {}).get("step_s")
    if off:
        res["ratio_vs_off"] = {k: round(v["step_s"] / off, 3) for k, v in res["arms"].items()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res.get("ratio_vs_off")))


if __name__ == "__main__":
    main()
