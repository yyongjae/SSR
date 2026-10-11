#!/usr/bin/env python
"""Independent re-checks of a built object pack (owner B1; read-only on the pack and every input).  CPU only.

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/yongjae/SSR-v3 \
    nice -n 10 /venv/ssr/bin/python tools/ck/v3/verify_obj_pack.py --split navtrain_val [--n 500] [--pack-root ...]

Checks on a seeded sample of rows (default 500):
  t0      pack obj[:, 0] == rows rebuilt from data.TeacherCache.load_det (the CK2 DET teacher's own reader; identity
          pose chain) for every slot; slot count == boxes above 0.3 (capped at K_OBJ).
  past    frame_token == CausalPastIndex.past(token, f) (pos - f only), dt == parquet timestamps, the npz exists in
          the cache named by frame_src, rows rebuilt from the raw npz == pack rows; frames marked invalid really are
          missing / out of range.
  targets match_targets recomputed from the pack's clean t0 rows + the GT file == fc_xyh / fc_state / fc_track
          bitwise; at OBS steps fc_xyh == gt_future.query of the matched track.
Whole split (no sampling): frame coverage and source split vs a2_availability.json; per-k GT-target coverage vs
f4_target_coverage_gt_val.json (val); CA (objects.ca_forecast_np on the clean pack) position error medians vs f4's
err_CA_median_by_k (val; f4 = a3 CA on uncapped boxes).
Writes <pack>/verify.json (a new file next to the pack; never touches the pack arrays).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[3]
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))

from navsim.agents.para_ssr.ck.temporal import config as TC  # noqa: E402
from navsim.agents.para_ssr.ck.temporal import objects as O  # noqa: E402
from navsim.agents.para_ssr.refiner import gt_future as GF  # noqa: E402
from navsim.agents.para_ssr.refiner.data import TeacherCache  # noqa: E402
from tools.ck.data import obj_pack as OP  # noqa: E402

KST = timezone(timedelta(hours=9))
A2 = Path("/workspace/yongjae/SSR/report/51_teacher_specialization_evidence/verify/A_ceiling/out/a2_availability.json")
F4 = Path("/workspace/yongjae/SSR/report/53_eval_and_plugin_evidence/A_full16/out/f4_target_coverage_gt_val.json")
SUBSET = {"navtrain_train": "navtrain", "navtrain_val": "navtrain", "navtest": "navtest"}
_G = {}


SFX = {"": ".npy"}


def _init(split, pack_dir, gt_dir):
    _G["src"] = O.BevfusionObjectSource(SUBSET[split])
    _G["tc"] = TeacherCache.for_subset(SUBSET[split])
    _G["P"] = Path(pack_dir)
    _G["gt"] = Path(gt_dir) if gt_dir else None
    _G["mm"] = {k: np.load(Path(pack_dir) / f"{k}{SFX['']}", mmap_mode="r") for k in
                list(OP.PACK_FILES) + (list(OP.TARGET_FILES) if gt_dir else [])}


def check_rows(rows):
    src, tc, mm = _G["src"], _G["tc"], _G["mm"]
    tok_all = _G["tokens"]
    r_ = dict(t0_max_abs=0.0, t0_count_bad=0, past_checked=0, past_token_bad=0, past_dt_max_err=0.0, past_rows_max_abs=0.0,
              past_src_bad=0, invalid_checked=0, invalid_bad=0, tgt_checked=0, tgt_bad=0, tgt_query_max_err=0.0)
    for r in rows:
        tok = str(tok_all[r])
        obj, mask = np.asarray(mm["obj"][r]), np.asarray(mm["obj_mask"][r])
        # t0 vs data.TeacherCache.load_det
        d = tc.load_det(tok, s_grid=False)
        ref = O.teacher_frame_rows(d["pred_boxes_3d"], d["pred_scores_3d"], d["pred_labels_3d"], (0.0, 0.0, 0.0),
                                   (0.0, 0.0, 0.0), 0.0, TC.SCORE_THR[0]).astype(np.float32)
        m0 = mask[0]
        if m0.sum() != len(ref) or len(ref) != min(TC.K_OBJ, int((d["pred_scores_3d"] >= TC.SCORE_THR[0]).sum())):
            r_["t0_count_bad"] += 1
        else:
            r_["t0_max_abs"] = max(r_["t0_max_abs"], float(np.abs(obj[0, m0] - ref).max()) if len(ref) else 0.0)
        # past frames
        _, _, ts0, pose0 = src.index.t0(tok)
        for f in (1, 2):
            pt, dt, why = src.index.past(tok, f)
            if mm["frame_valid"][r, f]:
                r_["past_checked"] += 1
                if str(mm["frame_token"][r, f]) != pt or why:
                    r_["past_token_bad"] += 1
                    continue
                r_["past_dt_max_err"] = max(r_["past_dt_max_err"], abs(float(mm["frame_dt"][r, f]) - dt))
                s = int(mm["frame_src"][r, f])
                cache = src.current if s == O.SRC_CURRENT else src.future
                if not cache.has(pt) or (s == O.SRC_FUTURE and src.current.has(pt)):
                    r_["past_src_bad"] += 1
                    continue
                raw = cache.load_boxes(pt)
                rows_p = O.teacher_frame_rows(raw["pred_boxes_3d"], raw["pred_scores_3d"], raw["pred_labels_3d"],
                                              src.index.pose(pt), pose0, dt, TC.SCORE_THR[f]).astype(np.float32)
                mf = mask[f]
                if mf.sum() != len(rows_p):
                    r_["past_token_bad"] += 1
                elif len(rows_p):
                    r_["past_rows_max_abs"] = max(r_["past_rows_max_abs"], float(np.abs(obj[f, mf] - rows_p).max()))
            else:
                r_["invalid_checked"] += 1
                really = (why != "") or (pt is not None and not (src.current.has(pt) or src.future.has(pt)))
                if not really or mask[f].any():
                    r_["invalid_bad"] += 1
        # targets
        if _G["gt"] is not None:
            gt = GF.load_objects(_G["gt"] / f"{tok}.npz")
            xyh, st, tr = O.match_targets(obj[0], m0, gt)
            r_["tgt_checked"] += 1
            if not (np.array_equal(xyh, mm["fc_xyh"][r]) and np.array_equal(st, mm["fc_state"][r])
                    and np.array_equal(tr, mm["fc_track"][r])):
                r_["tgt_bad"] += 1
            sel = np.flatnonzero(tr >= 0)
            if sel.size:
                bx, sq = GF.query(gt, np.asarray(TC.FC_T), dtype=np.float64)
                obs = (st[sel] == TC.FC_OBS) | (st[sel] == TC.FC_SINGLE)          # FC_SINGLE = official hold
                if (st[sel] == TC.FC_SINGLE).any():
                    single = np.asarray(gt["meta"])[tr[sel], GF.MT_SINGLE].astype(bool)
                    if not np.array_equal((st[sel] == TC.FC_SINGLE).any(1), single & obs.any(1)):
                        r_["tgt_bad"] += 1
                e = np.abs(np.asarray(mm["fc_xyh"][r])[sel, :, :2] - bx[tr[sel], :, :2])[obs]
                if e.size:
                    r_["tgt_query_max_err"] = max(r_["tgt_query_max_err"], float(e.max()))
                if not np.array_equal(obs, sq[tr[sel]] == GF.OBS):
                    r_["tgt_bad"] += 1
    return r_


def ca_errors(rows):
    """CA - GT euclidean error per k over matched agent slots at OBS steps (f4 definition, clean pack)."""
    mm = _G["mm"]
    E = [[] for _ in range(TC.FC_STEPS)]
    for r in rows:
        obj = np.asarray(mm["obj"][r], np.float64)
        mask = np.asarray(mm["obj_mask"][r])
        st = np.asarray(mm["fc_state"][r])
        if not mask[0].any():
            continue
        ca = O.ca_forecast_np(obj, mask, np.asarray(mm["frame_valid"][r]))
        gt = np.asarray(mm["fc_xyh"][r], np.float64)[..., :2]
        e = np.hypot(*(ca["xy"] - gt).transpose(2, 0, 1))
        ok = (st == TC.FC_OBS) & ca["agent"][:, None]
        for k in range(TC.FC_STEPS):
            E[k].append(e[ok[:, k], k])
    return [np.concatenate(x) if x else np.zeros(0) for x in E]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SUBSET))
    ap.add_argument("--pack-root", default=str(TC.PACK_ROOT))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--incomplete", action="store_true",
                    help="check the *.npy.tmp arrays of a pack whose build.json status is failed_* (writes "
                         "verify_incomplete.json); the pack stays unusable for ObjPack")
    a = ap.parse_args(argv)
    assert 1 <= a.workers <= 16
    t0 = time.time()
    if a.incomplete:
        import pandas as pd
        from types import SimpleNamespace
        P = OP.pack_dir("bevfusion", a.split, a.pack_root)
        build = json.loads((P / "build.json").read_text())
        if build.get("status") == "complete":
            raise SystemExit(f"{P}: complete pack: run without --incomplete")
        SFX[""] = ".npy.tmp"
        toks = pd.read_parquet(P / "tokens.parquet.tmp").token.astype(str).to_numpy()
        if not np.array_equal(toks, OP.packed_tokens(a.split)[: len(toks)]):
            raise SystemExit(f"{P}: token order differs from the packed order")
        pk = SimpleNamespace(dir=P, build=build, n=len(toks), tokens=toks, has_targets=bool(build.get("targets")))
    else:
        pk = OP.ObjPack(a.split, root=a.pack_root, targets=False)
    P = pk.dir
    targets = pk.has_targets
    gt_dir = pk.build["provenance"].get("gt", {}).get("dir") if targets else None
    n = pk.n
    _G["tokens"] = pk.tokens
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(n, min(a.n, n), replace=False))
    parts = np.array_split(sample, a.workers)
    with get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.split, str(P), gt_dir)) as pool:
        R = pool.map(check_rows, [list(p) for p in parts if len(p)])
        res = {k: (max(r[k] for r in R) if ("max" in k) else int(sum(r[k] for r in R))) for k in R[0]}
        out = dict(split=a.split, pack=str(P), n_rows=int(n), n_sample=int(len(sample)), sample_seed=0, checks=res)
        fv = np.load(P / f"frame_valid{SFX['']}")
        src = np.load(P / f"frame_src{SFX['']}")
        cov = {"t0_valid": float(fv[:, 0].mean()), "t-0.5_valid": float(fv[:, 1].mean()),
               "t-1.0_valid": float(fv[:, 2].mean()), "both_past": float((fv[:, 1] & fv[:, 2]).mean()),
               "no_past": float((~fv[:, 1] & ~fv[:, 2]).mean()),
               "t-0.5_src_current": int((src[:, 1] == 1).sum()), "t-0.5_src_future": int((src[:, 1] == 2).sum()),
               "t-1.0_src_current": int((src[:, 2] == 1).sum()), "t-1.0_src_future": int((src[:, 2] == 2).sum())}
        out["coverage"] = cov
        if A2.is_file():
            a2 = json.loads(A2.read_text()).get(a.split)
            if a2:
                out["a2_reference"] = {"t-0.5_cached": a2["past_t-0.5"]["frac_cached"], "t-1.0_cached": a2["past_t-1.0"]["frac_cached"],
                                       "both": a2["past_both_t-0.5_and_t-1.0"], "none": a2["past_none"],
                                       "t-0.5_src_current": a2["past_t-0.5"]["src_current"], "t-0.5_src_future": a2["past_t-0.5"]["src_future"],
                                       "t-1.0_src_current": a2["past_t-1.0"]["src_current"], "t-1.0_src_future": a2["past_t-1.0"]["src_future"],
                                       "note": "a2 counts cached frames without the dt_range rule"}
        if targets:
            E = pool.map(ca_errors, [list(c) for c in np.array_split(np.arange(n), a.workers * 4)])
            Ek = [np.concatenate([e[k] for e in E]) for k in range(TC.FC_STEPS)]
            out["ca_err_median_by_k"] = [round(float(np.median(e)), 3) for e in Ek]
            out["ca_err_mean_by_k"] = [round(float(np.mean(e)), 3) for e in Ek]
            out["ca_err_n_by_k"] = [int(len(e)) for e in Ek]
            if a.split == "navtrain_val" and F4.is_file():
                f4 = json.loads(F4.read_text())
                out["f4_reference"] = dict(gt_matched_at_t0=f4["gt_matched_at_t0"], gt_target_by_k=f4["all"]["gt_target_by_k"],
                                           err_CA_median_by_k=f4["all"]["err_CA_median_by_k"],
                                           err_CA_mean_by_k=f4["all"]["err_CA_mean_by_k"],
                                           note="f4: a3 CA on uncapped score>=.3 boxes, GT target = matched & OBS")
            out["pack_target_steps"] = pk.build["counts"].get("target_steps")
            out["pack_match_rate"] = pk.build["counts"]["targets"]["match_rate"]
    out["kst"] = datetime.now(KST).isoformat(timespec="seconds")
    out["wall_s"] = round(time.time() - t0, 1)
    bad = res["t0_count_bad"] + res["past_token_bad"] + res["past_src_bad"] + res["invalid_bad"] + res["tgt_bad"]
    out["ok"] = bool(bad == 0 and res["t0_max_abs"] < 1e-4 and res["past_rows_max_abs"] < 1e-4 and res["past_dt_max_err"] < 1e-5
                     and res["tgt_query_max_err"] < 1e-4)
    name = "verify_incomplete.json" if a.incomplete else "verify.json"
    out["pack_status"] = pk.build.get("status")
    tmp = P / f"{name}.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, P / name)
    print(json.dumps(out, indent=1))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
