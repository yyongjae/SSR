#!/usr/bin/env python
"""Summarise smoke_v3.sh runs: python -B tools/ck/v3/smoke_summary.py <D3/smoke/<mode>> -> one JSON line per run
(+ <dir>/summary.json): steps done, skipped steps, gpu_mem_gb max, median s/step and cand_per_s over the step-log
windows after the first (warm-up excluded), loss / fc_loss / grad norms of the last logged step, temporal config,
parameter counts, every val split's metrics finite (non-NaN numeric fields; AUCs of a key without failures are NaN by
construction and listed separately)."""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def summarize_pre(run: Path, cfg: dict) -> dict:
    """stage pre (tools/ck/v3/pretrain_forecaster.py): steps, s/step, memory, loss, val ADE learned vs CA."""
    log = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines()] \
        if (run / "train_log.jsonl").is_file() else []
    st = [r for r in log if r.get("kind") == "step"]
    out = {"run": run.name, "stage": "pre", "done": (run / "done.json").is_file(), "steps": st[-1]["step"] if st else 0,
           "param_count": cfg.get("param_count"), "batch": (cfg.get("args") or {}).get("batch")}
    if len(st) >= 2:
        ds = [(b["sec"] - a["sec"]) / max(b["step"] - a["step"], 1) for a, b in zip(st, st[1:]) if b["sec"] > a["sec"]]
        out["s_per_step_median"] = round(statistics.median(ds), 4) if ds else None
    if st:
        out["gpu_mem_gb"] = max((r.get("gpu_mem_gb", 0.0) for r in st), default=None)
        out["fc_loss_first_last"] = [st[0].get("fc_loss"), st[-1].get("fc_loss")]
        out["loss_finite"] = all(math.isfinite(float(r["fc_loss"])) for r in st if "fc_loss" in r)
    val = [json.loads(x) for x in (run / "val_metrics.jsonl").read_text().splitlines()] \
        if (run / "val_metrics.jsonl").is_file() else []
    out["val"] = {r["split"]: ({"ade_learned": (r.get("learned") or {}).get("fc_minade"),
                                "ade_ca": (r.get("ca") or {}).get("fc_minade"),
                                "ratio": r.get("ade_ratio_learned_over_ca")} if "learned" in r else
                               {"value": r.get("value"), "flag": r.get("flag")}) for r in val}
    out["val_finite"] = bool(out["val"]) and all(v.get("ade_learned") is None or math.isfinite(v["ade_learned"])
                                                 for v in out["val"].values())
    return out


def summarize(run: Path) -> dict:
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").is_file() else {}
    if str(cfg.get("version", "")).startswith("v3_fc_pretrain"):
        return summarize_pre(run, cfg)
    log = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines()] \
        if (run / "train_log.jsonl").is_file() else []
    st = [r for r in log if r.get("kind") == "step"]
    out = {"run": run.name, "done": (run / "done.json").is_file(), "steps": st[-1]["step"] if st else 0,
           "world": cfg.get("world_size"), "tokens_per_rank": cfg.get("tokens_per_rank"),
           "temporal": (cfg.get("temporal") or {}).get("forecaster") if (cfg.get("temporal") or {}).get("on") else "off",
           "privileged": cfg.get("privileged"), "param_count": cfg.get("param_count"),
           "param_count_trainable": cfg.get("param_count_trainable"),
           "param_count_temporal_shared": cfg.get("param_count_temporal_shared"),
           "param_count_forecaster": cfg.get("param_count_forecaster")}
    if st:
        last = st[-1]
        out["n_skip"] = [last.get("n_skip_loss"), last.get("n_skip_grad")]
        out["gpu_mem_gb"] = max((r.get("gpu_mem_gb", 0.0) for r in st), default=None)
        w = st[1:] if len(st) > 1 else st
        cps = [r["cand_per_s"] for r in w if r.get("cand_per_s")]
        out["cand_per_s_median"] = statistics.median(cps) if cps else None
        cand_per_step = None
        tpr, nv = cfg.get("tokens_per_rank"), cfg.get("n_var")
        if tpr and cfg.get("world_size"):
            cand_per_step = int(cfg["world_size"]) * int(tpr) * (32 + int(nv or 0))
        if cand_per_step and out["cand_per_s_median"]:
            out["s_per_step_median"] = round(cand_per_step / out["cand_per_s_median"], 4)
        for k in ("loss", "loss_ck", "score_bce", "sur", "fc_loss", "fc_nll", "fc_minade", "fc_n_obj", "t_n_obj",
                  "t_n_valid", "grad_norm", "grad_norm_fc", "n_obj_bad"):
            if k in last:
                out[k] = last[k]
        out["loss_finite"] = all(math.isfinite(float(r["loss"])) for r in st if "loss" in r)
    val = [json.loads(x) for x in (run / "val_metrics.jsonl").read_text().splitlines()] \
        if (run / "val_metrics.jsonl").is_file() else []
    vs = {}
    for r in val:
        nums = {k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        nan = sorted(k for k, v in nums.items() if not math.isfinite(float(v)))
        # a forecast group with no supervised object has NaN means by construction (e.g. no VRU in 8 tokens)
        nan = [k for k in nan if not (k.startswith(("fc_veh_", "fc_vru_"))
                                      and nums.get(k[:7] + "n_obj", 1) == 0)]
        vs[r["split"]] = {"n_tokens": r.get("n_tokens"), "n_nan": len(nan),
                          "nan_non_auc": [k for k in nan if "auc" not in k and "pair_acc" not in k],
                          "privileged": r.get("privileged"),
                          **{k: r[k] for k in ("pdms_ck_noim", "auc_fail_ttc", "auc_fail_nc", "fc_nll", "fc_minade",
                                               "fc_minfde", "n_obj_bad") if k in r}}
    out["val"] = vs
    out["val_finite"] = bool(vs) and all(not v["nan_non_auc"] for v in vs.values())
    return out


def main(d: str) -> None:
    root = Path(d)
    rows = [summarize(r) for r in sorted(root.iterdir()) if r.is_dir() and (r / "config.json").is_file()]
    for r in rows:
        print(json.dumps(r, default=str))
    (root / "summary.json").write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
