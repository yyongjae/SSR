#!/usr/bin/env python
"""refiner v5 student smoke on CPU (not a training run): the real v2 ParaSSRAgent with the fusion v13 BEV-selector
distillation + the CK2 e2e module with the v5 block (relation 'student', unknown evidence, corrector A_S, Z-KD to the v5
teachers), through pytorch_lightning on the 2-token CK2 fixture of tools/ck/e2e2/cpu_dryrun.py (+ the DET teacher's
object-pack rows, which that fixture predates), batch 1, accumulate 2: epoch 0 'warmup' (+ epoch 1 'warmup_record'),
then one eval-mode forward (ck2_* inference incl. ck2_v5_traj).  Teachers = the v5 runs' latest ckpt_last.pt
(teacher_require_done false: they may still be training).

  cd /workspace/yongjae/SSR-v5 && env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
    /venv/ssr/bin/python tools/ck/v5/smoke_student_v5.py [--zkd 0|1] [--epochs 2] [--tag T]

Checks: finite ck2/* and v5/* logs, no non-finite loss, v5 terms present (rel_a_bce, rel_h_bce, ref_loss, zkd_det /
zkd_map with --zkd 1), every v5 parameter moved, the inference returns ck2_v5_traj [B, 8, 3] finite and ck2_v5_src.
Output: CK/v5/smoke_student/<tag>/ (result.json, steps.jsonl).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from tools.ck.e2e2 import cpu_dryrun as CD  # noqa: E402

CK = Path("/home/external-user/ssd/yongjae_refiner/ck")
T_DET = CK / "v5/train/v5T_rel"
T_MAP = CK / "v5/train/v5M_unk"
V13 = dict(use_distill=True, distill_selector_v8=False, distill_selector_v9=False, distill_selector_v10=False,
           distill_selector_v11=False, distill_selector_v12=False, distill_selector_v13=True,
           distill_teacher_source="targets", distill_feature_root="/workspace/teacher_cache", distill_lambda_gt=0.45,
           distill_lambda_cov=0.20, distill_lambda_risk=0.35, distill_tau_gt=4.5, distill_tau_risk=8.0,
           distill_probe_warmup_steps=2, distill_probe_ramp_steps=2, distill_probe_loss_weight=0.5,
           distill_probe_lr_mult=3.0, distill_proj_lr_mult=2.0, distill_probe_use_ema=True,
           distill_probe_ema_decay=0.99)
V5_AGENT = dict(det_vel_code_weights=[0.41, 1.05], expose_det_outputs=True)


def v5_ck2(io_dir: Path, zkd: bool) -> dict:
    d = CD.ck2_dict(io_dir, "main")
    d.update(teacher_det_run=str(T_DET), teacher_map_run=str(T_MAP), teacher_require_done=False,
             kd_score_keys=["nc", "dac"])
    d["bev_kd"] = {"enabled": False, "teachers": ["det"]}
    d["v4"] = {"readout": "range", "exec_model": "lqr", "ev_tbins": 20, "ev_tbin_s": 0.25, "ep_evidence": True,
               "pref_head": True, "corridor_version": 2, "rg_read": "any4", "rg_stopped_aware": True, "edits": "off",
               "corrector": False, "kd_parent_only": True, "kd_score_scope": "all",
               "kd_constraint": {"enabled": False}, "rg_gt_aux": {"enabled": True, "lambda": 0.5, "max_parents": 8},
               "v5": {"enabled": True, "relation": "student", "unknown_ev": True, "corrector": True,
                      "z_kd": bool(zkd)}}
    return d


class FixtureV5(CD.FixtureDS):
    """the cpu_dryrun fixture + the v5 DET teacher's pack rows (e2e_data2.KDObjRows of its ck_row)."""

    def __init__(self):
        super().__init__()
        from navsim.agents.para_ssr.ck import e2e_data2 as D2
        from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
        from navsim.agents.para_ssr.ck import e2e_data as ED
        packed = CKE2E2Config().warmup["packed"]
        kd = D2.KDObjRows(D2.teacher_v5_rel_cfg(str(T_DET)), packed)
        self.kd = [kd.item(int(r)) for r in self.t["ck_row"]]
        toks = ED.RowMap(packed).tokens                 # the v13 distillation needs targets['scene_token']
        self.tok = [str(toks[int(r)]) for r in self.t["ck_row"]]

    def __getitem__(self, i):
        f, t = super().__getitem__(i)
        t = dict(t, **self.kd[i], scene_token=self.tok[i])
        return f, t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zkd", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--tag", default="cpu")
    a = ap.parse_args(argv)
    root = CK / "v5/smoke_student" / a.tag
    root.mkdir(parents=True, exist_ok=True)
    for p in (T_DET, T_MAP):
        if not (p / "ckpt_last.pt").is_file():
            print(f"teacher {p}: no ckpt_last.pt yet", flush=True)
            return 2
    io = root / "io"
    io.mkdir(exist_ok=True)
    ck2 = v5_ck2(io, bool(a.zkd))
    CD.V2_OVERRIDES.update(V13)
    CD.V2_OVERRIDES.update(V5_AGENT)
    CD.FixtureDS = FixtureV5                                   # fit() builds its DataLoader from CD.FixtureDS
    agent = CD.build_agent(ck2)
    student = agent.ck_student
    v5_mods = {n: p.detach().clone() for n, p in student.named_parameters()
               if n.startswith(("relation.", "corrector_v5.")) or n in ("v4_readout.rel_w", "v4_readout.resid.0.weight")}
    t0 = time.time()
    CD.fit(agent, root, a.epochs)
    steps = CD.read_jsonl(io / "steps_rank0.jsonl")
    res = {"tag": a.tag, "zkd": bool(a.zkd), "sec_fit": round(time.time() - t0, 1), "n_steps_logged": len(steps)}
    keys = sorted({k for s in steps for k in s if k.startswith(("ck2/", "v5/"))})
    res["v5_keys"] = [k for k in keys if k.startswith("v5/")]
    bad = sorted({k for s in steps for k, v in s.items() if k.startswith(("ck2/", "v5/")) and isinstance(v, float)
                  and not math.isfinite(v)})
    res["nonfinite_keys"] = bad
    res["nonfinite_loss"] = int(sum(float(s.get("ck2/loss_nonfinite", 0.0)) for s in steps))
    moved = {n: float((p.detach() - v5_mods[n]).abs().max()) for n, p in student.named_parameters() if n in v5_mods}
    res["v5_params_unmoved"] = sorted(n for n, d in moved.items() if d == 0.0)
    need = ["v5/rel_a_bce", "v5/rel_h_bce", "v5/ref_loss"] + (["v5/zkd_det", "v5/zkd_map"] if a.zkd else [])
    res["missing_terms"] = [k for k in need if k not in keys]
    # one eval-mode forward: the ck2_* inference with the v5 output
    ds = FixtureV5()
    f, t = ds[0]
    feats = {k: v[None] for k, v in f.items()}
    agent.eval()
    with torch.no_grad():
        pred = agent.forward(feats)
    tv = pred.get("ck2_v5_traj")
    res["infer_v5_traj"] = None if tv is None else {"shape": list(tv.shape), "finite": bool(torch.isfinite(tv).all()),
                                                     "src": int(pred["ck2_v5_src"][0]),
                                                     "mu_abs": float(pred["ck2_v5_mu"].abs().mean())}
    res["pass"] = (not bad and res["nonfinite_loss"] == 0 and not res["missing_terms"] and tv is not None
                   and bool(torch.isfinite(tv).all()) and tuple(tv.shape) == (1, 8, 3))
    (root / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
