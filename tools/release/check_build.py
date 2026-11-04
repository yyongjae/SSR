#!/usr/bin/env python
"""Build the real agent of a composed run on CPU, from its launcher overrides: Hydra compose + instantiate, the parent
rules of the launcher's dry run do not reach (grid, heads, teachers' manifests, norm files).  No GPU, no training, no
checkpoint: backbone_pretrained is false, CUDA is hidden.  Run with the environment of the composition (CK_BEV64 and the
teacher variables are read at import).

  check_build.py --overrides <run>/launch_dryrun/overrides.txt [--out F.json] [--reader-tokens N]

--reader-tokens N also reads N tokens through both teacher readers (the refiner's and the BEV KD's), requires them to
agree, and reports a sha per teacher.  Prints one summary, exit 3 with the reason when the agent refuses the config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[2]
sys.path[:] = [str(REPO)] + [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]


def _module_params(agent):
    out = {}
    for name, p in agent.named_parameters():
        d = out.setdefault(name.split(".")[0], [0, 0])
        d[0] += p.numel()
        d[1] += p.numel() if p.requires_grad else 0
    return out


def summarize(agent):
    cfg = agent._config
    mods = _module_params(agent)
    rec = {"params_total": sum(v[0] for v in mods.values()), "params_trainable": sum(v[1] for v in mods.values()),
           "params_by_module": mods, "builders": [type(b).__name__ for b in agent.get_target_builders()],
           "backbone": cfg.image_architecture, "bn_mode": agent.para_ssr_model._backbone_bn_mode,
           "frames": list(cfg.frame_indices), "bev": [cfg.bev_h, cfg.bev_w], "pc_range": list(cfg.pc_range),
           "interaction": bool(cfg.use_task_interaction), "det_head": bool(cfg.use_det_motion_head),
           "map_head": (cfg.map_head_type if cfg.use_map_head else "off"), "undistort": bool(cfg.undistort),
           "heading_from_xy": bool(cfg.plan_heading_from_xy)}
    d = getattr(agent, "_distill", None)
    rec["bev_kd"] = None if d is None else {
        "off_banks": list(getattr(cfg, "distill_off_banks", ()) or ()),
        "stores": {k: {"cache": s.teacher, "root": s.teacher_root()} for k, s in d.stores.items()}}
    ck = agent._ck_e2e2
    rec["refiner"] = None if ck is None else {
        "use_teachers": ck.cfg.use_teachers, "det_run": ck.cfg.teacher_det_run, "map_run": ck.cfg.teacher_map_run,
        "kd_sources": list(ck.cfg.kd_sources), "kd_score_keys": list(ck.cfg.kd_score_keys),
        "lambda_kd_score": ck.cfg.lambda_kd_score, "lat_kd_balance": ck.cfg.lat_kd["balance"],
        "lat_kd_fixed": ck.cfg.lat_kd["fixed"], "kd_calib_enabled": ck.cfg.kd_calib["enabled"],
        "kd_calib_mode": ck.cfg.kd_calib["mode"], "infer_select": ck.cfg.infer_select,
        "submit_selection": ck.cfg.submit_selection, "seed": ck.cfg.seed}
    return rec


def reader_shas(overrides, n):
    """sha16 of n teacher BEVs of each role through both readers; raises when the readers disagree."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd
    from navsim.agents.para_ssr.distill.teacher_store import DistillTeacherTargetBuilder
    from navsim.agents.para_ssr.refiner import data as RD
    from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache
    kv = {o.split("=", 1)[0].lstrip("+"): o.split("=", 1)[1] for o in overrides if "=" in o}
    packed = kv["agent.config.ck_e2e2.warmup.packed"]
    toks = sorted(pd.read_parquet(Path(packed) / "tokens.parquet").token.astype(str))[:n]
    branches = {"bevfusion": {"cache_name": kv.get("agent.config.distill_branches.bevfusion.cache_name", "bevfusion")},
                "resmap": {"cache_name": kv.get("agent.config.distill_branches.resmap.cache_name", "resmap")}}
    store = DistillTeacherTargetBuilder(SimpleNamespace(
        distill_feature_root=kv.get("agent.config.distill_feature_root", ""), embed_dims=256, bev_h=100, bev_w=100,
        distill_cache_size=(100, 100), distill_branches=branches))._open()
    refiner = [RD.TeacherCache.for_subset("navtrain"), ResmapCache.for_subset("navtrain")]
    out = {}
    for i, role in enumerate(("det", "map")):
        h = hashlib.sha256()
        for t in toks:
            x = np.asarray(refiner[i].load_bev(t, s_grid=True), np.float16)
            y = np.asarray(store[i].load_one_token(t), np.float16)
            if x.tobytes() != y.tobytes():
                raise ValueError(f"teacher {role}, token {t}: the refiner reader and the BEV KD reader differ")
            h.update(x.tobytes())
        out[role] = h.hexdigest()[:16]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--overrides", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reader-tokens", type=int, default=0)
    a = ap.parse_args(argv)
    overrides = [ln.strip() for ln in Path(a.overrides).read_text().splitlines() if ln.strip()]
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(REPO / "navsim/planning/script/config/training"), version_base="1.2"):
            cfg = compose(config_name="default_training", overrides=overrides + ["++agent.config.backbone_pretrained=false"])
        agent = instantiate(cfg.agent)
        rec = summarize(agent)
        rec["seed"] = cfg.seed
        rec["max_epochs"] = {"trainer": cfg.trainer.params.max_epochs, "agent": cfg.agent.config.max_epochs}
        if a.reader_tokens:
            rec["reader_sha16"] = reader_shas(overrides, a.reader_tokens)
    except Exception as e:      # noqa: BLE001  (the agent's refusals are ValueError, KeyError, NotImplementedError, ...)
        print(f"[check_build] REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"[check_build] params {rec['params_total'] / 1e6:.2f} M, trainable {rec['params_trainable'] / 1e6:.2f} M")
    print(f"[check_build] modules {({k: round(v[1] / 1e6, 2) for k, v in rec['params_by_module'].items()})} (M trainable)")
    print(f"[check_build] builders {rec['builders']}")
    print(f"[check_build] backbone {rec['backbone']}, bn {rec['bn_mode']}, frames {rec['frames']}, bev {rec['bev']}, "
          f"pc_range {rec['pc_range']}, seed {rec['seed']}, epochs {rec['max_epochs']}")
    print(f"[check_build] student interaction {rec['interaction']}, det head {rec['det_head']}, map head {rec['map_head']}, "
          f"undistort {rec['undistort']}, heading from xy {rec['heading_from_xy']}")
    print(f"[check_build] BEV KD {rec['bev_kd']}")
    print(f"[check_build] refiner {rec['refiner']}")
    if a.reader_tokens:
        print(f"[check_build] teacher readers agree on {a.reader_tokens} tokens, sha16 {rec['reader_sha16']}")
    print("[check_build] refusals: none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
