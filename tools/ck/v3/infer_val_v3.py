#!/usr/bin/env python
"""refiner-v3 Step-1 inference on navtrain_val r34 top-16 + 80 variants (SPEC section 6; builder LOSS+EVAL,
2026-10-11 KST).  Port of ck2/analysis/lead_remedy/scripts/infer_val_ck2T.py stage infer (+ check), for temporal
runs (arms a / b / d) and temporal-off runs (arm c = ck2/train/ck2T10).

  cd /workspace/yongjae/SSR-v3 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/yongjae/SSR-v3 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 10 /venv/ssr/bin/python tools/ck/v3/infer_val_v3.py --run <run dir> \
      [--which last] [--out D3/infer/<run name>] [--allow-privileged] [--device cuda] [--batch 8] [--workers 6]
  CPU check (no GPU): CUDA_VISIBLE_DEVICES= ... --device cpu --limit 64 --out /tmp/...

GPU: only GPU 0 or 1 (CUDA_VISIBLE_DEVICES must be exactly "0" or "1"); GPUs 2-7 are refused.  Run by the main
session after training (~4 min per arm [추정 from ck2T10's 207 s]); builders run it on CPU only.

Inputs: every packed navtrain_val row (18,179), CKDataset(bev 'T', k 16, labels None, gt False, objects =
{cfg, aug False, targets True} when the run's temporal cfg is on; NO objects kwarg when off, so the CK2 path is
untouched), extra = lead_remedy/infer/variants/traj80.npy (the variants of labels96.npz), model = load_ck(run,
which, allow_privileged) (refuses privileged runs unless --allow-privileged), forward = CKNet.forward(bev, cand,
status, extra = variants, decode = False[, objs = OBJ_INPUT_KEYS (+ PRIV_INPUT_KEYS for oracle_gt)]) under fp16
autocast on cuda (as train_ck2 eval), fp32 on cpu.  Nothing GT-derived reaches the model except the PRIV keys of a
privileged oracle run.

Outputs (OUT, default D3/infer/<run name>; resumable via done.npy): score_logit16 [N,16,5], prob16, score_logit80
[N,80,5], prob80 (sigmoid, CK_KEYS nc dac ep ttc comfort), w_lat16 [N,16,6], done [N], bev_ok [N], obj_ok [N]
(True for temporal-off runs), tokens.parquet (token, log, row), infer_meta.json (run, which, ckpt sha16, temporal
cfg, privileged, cand_sha16 of packed cand.npy, variants path / sha16, code sha256 of this file and the model
files, KST times, check).  Temporal runs with targets also write fc_metrics.json: the arm's own forecast (learned:
out['fc']; ca / oracle: TemporalBranch.scene(objs).fc) and the analytic CA reference (fixed scales), both through
metrics.ForecastMetrics on the CLEAN val pack (no augmentation).
Check (stage check, also run at the end of a full inference; review fix D6: REPORTED in infer_meta.json['check'],
never an exit; decide_step1.py requires check.ok):
  temporal runs  the trainer saved the r34 logits of its last navtrain_val eval (eval_r34_navtrain_val_last.npz,
                 same weights as ckpt_last).  Per token and logit: |dlogit| = |infer - train|; picks = argmax
                 ck_final(sigmoid, im = 1, noim weights) per token.  ok iff the 99.9 % quantile of |dlogit| <= 2e-4
                 and at most 0.1 % of the picks flip (CUDA scatter_add order, another batch composition and fp16
                 make bit equality impossible; measured scale 1e-7..1e-4 [실측 review_correctness E1]).
  temporal-off   (arm c) no saved logits: pdms_ck_noim recomputed from prob16 + cand labels vs the run's last
                 navtrain_val r34 val_metrics row; ok iff |diff| <= 1e-4 and n_sel equal ('exact' = |diff| <= 1e-6).
Rows whose object pack row has obj_ok False are scored like any other (outputs are finite for empty object sets)
and only recorded in obj_ok.npy (review correctness finding 4).
infer_meta.json also carries provenance: the run's training provenance (config.json 'provenance') and this
inference's own (tools/ck/v3/provenance.py: code, val pack, ca_scales), which decide_step1.py compares.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)
import navsim  # noqa: E402

assert navsim.__file__.startswith(REPO), navsim.__file__

import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Dict, Optional  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CK = Path("/home/external-user/ssd/yongjae_refiner/ck")
D3 = CK / "v3"
LR = CK / "ck2" / "analysis" / "lead_remedy"
SPLIT = "navtrain_val"
VARIANTS = LR / "infer" / "variants" / "traj80.npy"
K16, NVAR = 16, 5
K80 = K16 * NVAR
KST = 9 * 3600
CHECK_TOL = 1e-6                 # 'exact' (reported)
CHECK_PDMS_TOL = 1e-4            # temporal-off tolerance
CHECK_DLOGIT = 2e-4              # temporal runs: 99.9 % quantile of |dlogit|
CHECK_DLOGIT_Q = 0.999
CHECK_FLIP_MAX = 0.001           # at most 0.1 % flipped picks
MODEL_FILES = ("navsim/agents/para_ssr/ck/model.py", "navsim/agents/para_ssr/ck/temporal/branch.py",
               "navsim/agents/para_ssr/ck/temporal/forecast.py", "navsim/agents/para_ssr/ck/temporal/raster.py",
               "navsim/agents/para_ssr/ck/temporal/config.py", "tools/ck/data/ck_dataset.py",
               "tools/ck/data/obj_pack.py")


def kst(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime((time.time() if t is None else t) + KST))


def log(msg: str) -> None:
    print(f"[{kst()}] {msg}", flush=True)


def sha(p, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:n]


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))).astype(np.float32)


def save_npy(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def device_guard(device: str) -> None:
    if device.startswith("cuda"):
        cv = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cv not in ("0", "1"):
            raise SystemExit(f"cuda inference only on GPU 0 or 1 (CUDA_VISIBLE_DEVICES='0' or '1'), got {cv!r}")
    elif device != "cpu":
        raise SystemExit(f"--device {device!r}: cuda | cpu")


def temporal_of(cfg: Dict):
    """TemporalCfg of a run config (old configs without 'temporal' -> off); None if the temporal package is absent
    and the config has no temporal key."""
    t = cfg.get("temporal")
    try:
        from navsim.agents.para_ssr.ck.temporal.config import TemporalCfg
    except ImportError:
        if t and t.get("on"):
            raise
        return None
    return TemporalCfg.from_dict(t)


def load_model(run: Path, which: str, device: str, allow_privileged: bool):
    from navsim.agents.para_ssr.ck.model import load_ck
    params = inspect.signature(load_ck).parameters
    cfg0 = json.load(open(run / "config.json"))
    tc = temporal_of(cfg0)
    if "allow_privileged" in params:
        net, cfg = load_ck(run, which=which, device=device, allow_privileged=allow_privileged)
    else:                                   # pre-v3 model.py: only temporal-off runs can be loaded at all
        if tc is not None and tc.on:
            raise SystemExit("model.load_ck has no allow_privileged: temporal model code missing")
        net, cfg = load_ck(run, which=which, device=device)
    return net, cfg, tc


def infer_provenance(tc) -> Dict:
    """this inference's own provenance (code, the val pack it reads, ca_scales): tools/ck/v3/provenance.py."""
    from tools.ck.v3 import provenance as PV
    if tc is None or not tc.on:
        return {"code": {"train": {k: v for k, v in PV.train_code().items() if k != "files"},
                         "worktree": PV.worktree()}, "data": {}}
    return PV.run_provenance(tc, splits=[SPLIT])


def objs_of(batch: Dict, tc, dev):
    if tc is None or not tc.on:
        return None
    from navsim.agents.para_ssr.ck.temporal.config import OBJ_INPUT_KEYS, PRIV_INPUT_KEYS
    keys = OBJ_INPUT_KEYS + (PRIV_INPUT_KEYS if tc.forecaster == "oracle_gt" else ())
    miss = [k for k in keys if k not in batch]
    if miss:
        raise KeyError(f"batch lacks object keys {miss} (CKDataset objects= kwarg?)")
    return {k: batch[k].to(dev) for k in keys}


def logit_check(run: Path, logit16: np.ndarray, done: np.ndarray, ref_step: Optional[int]) -> Optional[Dict]:
    """Per-token comparison with the trainer's saved r34 logits (temporal runs; None if the file is absent)."""
    from navsim.agents.para_ssr.ck.select import ck_final
    f = run / f"eval_r34_{SPLIT}_last.npz"
    if not f.is_file():
        return None
    z = np.load(f)
    rows, tl = z["rows"].astype(np.int64), np.asarray(z["logit"], np.float64)
    out = {"file": str(f), "train_step": int(z["step"]), "train_epoch": int(z["epoch"]), "ref_step": ref_step}
    keep = (rows >= 0) & np.asarray(done, bool)[np.clip(rows, 0, len(done) - 1)]
    rows, tl = rows[keep], tl[keep]
    il = np.asarray(logit16, np.float64)[rows]
    fin = np.isfinite(il).all((1, 2)) & np.isfinite(tl).all((1, 2))
    d = np.abs(il[fin] - tl[fin]).reshape(-1)
    lab_ok = np.load(CK / "labels" / SPLIT / "cand" / "ok.npy")[:, :K16].astype(bool)[rows[fin]].all(1)
    pick = lambda L: np.asarray(ck_final(sigmoid(L).astype(np.float64), np.ones(L.shape[:2])), np.float64).argmax(1)  # noqa: E731
    pi, pt = pick(il[fin][lab_ok]), pick(tl[fin][lab_ok])
    out.update(n_rows=int(fin.sum()), n_logits=int(d.size), max_abs_dlogit=float(d.max()) if d.size else None,
               q999_abs_dlogit=float(np.quantile(d, CHECK_DLOGIT_Q)) if d.size else None,
               frac_dlogit_gt_tol=float((d > CHECK_DLOGIT).mean()) if d.size else None,
               n_picks=int(lab_ok.sum()), n_flipped=int((pi != pt).sum()),
               frac_flipped=float((pi != pt).mean()) if lab_ok.any() else None,
               tol={"dlogit": CHECK_DLOGIT, "quantile": CHECK_DLOGIT_Q, "flip_max": CHECK_FLIP_MAX})
    out["ok"] = bool(d.size and out["q999_abs_dlogit"] <= CHECK_DLOGIT and out["frac_flipped"] is not None
                     and out["frac_flipped"] <= CHECK_FLIP_MAX
                     and (ref_step is None or int(z["step"]) == int(ref_step)))
    return out


def reproduce_check(run: Path, prob16: np.ndarray, done: np.ndarray, logit16: Optional[np.ndarray] = None) -> Dict:
    """pdms_ck_noim from prob16 + official cand labels on the 16-labels-ok rows vs the run's last navtrain_val r34
    val_metrics row (as lead_remedy stage check) + the per-token logit check of temporal runs (module docstring)."""
    from navsim.agents.para_ssr.ck import constants as Cn
    from navsim.agents.para_ssr.ck.select import ck_final
    lab = np.load(CK / "labels" / SPLIT / "cand" / "labels.npy", mmap_mode="r")[:, :K16].astype(np.float32)
    lok = np.load(CK / "labels" / SPLIT / "cand" / "ok.npy")[:, :K16].astype(bool)
    r = np.flatnonzero(done)
    sel = lok[r].all(1)
    p = np.asarray(prob16, np.float64)[r][sel]
    s = np.asarray(ck_final(p, np.ones(p.shape[:2])), np.float64)
    ch = s.argmax(1)
    PD = lab[r][sel][..., Cn.LBL["pdms"]]
    ours = float(PD[np.arange(len(ch)), ch].mean()) if len(ch) else float("nan")
    out = {"n_rows": int(len(r)), "n_sel": int(sel.sum()), "pdms_ck_noim": ours, "frac_changed_noim":
           float((ch != 0).mean()) if len(ch) else float("nan")}
    vm = run / "val_metrics.jsonl"
    if vm.is_file():
        ref = [json.loads(x) for x in open(vm) if x.strip()]
        ref = [d for d in ref if d.get("split") == SPLIT and d.get("kind") == "r34"]
        if ref:
            ref = max(ref, key=lambda d: d["step"])
            out.update(ref_epoch=ref["epoch"], ref_step=ref["step"], ref_pdms_ck_noim=ref.get("pdms_ck_noim"),
                       ref_n_sel=ref.get("n_sel"))
            if ref.get("pdms_ck_noim") is not None:
                out["absdiff"] = abs(ours - float(ref["pdms_ck_noim"]))
                out["exact"] = bool(out["absdiff"] <= CHECK_TOL and int(ref.get("n_sel", -1)) == out["n_sel"])
                out["pdms_ok"] = bool(out["absdiff"] <= CHECK_PDMS_TOL and int(ref.get("n_sel", -1)) == out["n_sel"])
    lc = logit_check(run, logit16, done, out.get("ref_step")) if logit16 is not None else None
    if lc is not None:
        out["logit_check"] = lc
        out["rule"] = "per-token logits (temporal run): q99.9 |dlogit| <= 2e-4 and <= 0.1 % flipped picks"
        out["ok"] = bool(lc["ok"])
    elif "pdms_ok" in out:
        out["rule"] = "pdms_ck_noim |diff| <= 1e-4 and n_sel equal (no saved logits)"
        out["ok"] = bool(out["pdms_ok"])
    else:
        out["ok"] = False
        out["rule"] = "no reference (no val_metrics row, no saved logits)"
    return out


def infer(a) -> Dict:
    import torch
    from torch.utils.data import DataLoader

    from tools.ck.data.ck_dataset import CKDataset, collate_ck
    t0 = time.time()
    device_guard(a.device)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    run = Path(a.run)
    dev = torch.device("cuda:0" if a.device.startswith("cuda") else "cpu")
    net, cfg, tc = load_model(run, a.which, str(dev), a.allow_privileged)
    net.eval()
    if cfg.get("arm") != "T":
        raise SystemExit(f"{run}: arm {cfg.get('arm')!r} (Step 1 evaluates DET teachers, arm T)")
    on = bool(tc is not None and tc.on)
    priv = bool(on and tc.privileged)
    if priv and not a.allow_privileged:
        raise SystemExit(f"{run} is PRIVILEGED: pass --allow-privileged (analysis only)")
    name = cfg.get("run") or run.name
    out = Path(a.out) if a.out else D3 / "infer" / name
    ro = str(out.resolve())
    # D3 lives behind a symlink (/home/external-user/ssd -> /workspace/yongjae/ssd): compare resolved paths
    if not (ro.startswith(str(D3.resolve()) + "/") or ro.startswith("/tmp/")):
        raise SystemExit(f"--out must be under {D3} (or /tmp for checks), got {out}")
    if priv and not out.name.endswith("_PRIV"):
        raise SystemExit(f"privileged run: output dir name must end in '_PRIV', got {out}")
    pdir = CK / "packed" / SPLIT
    tdf = pd.read_parquet(pdir / "tokens.parquet")
    pok = np.load(pdir / "ok.npy").astype(bool)
    N = len(tdf)
    var_all = np.load(VARIANTS, mmap_mode="r")
    assert var_all.shape == (N, K80, 8, 3), var_all.shape
    out.mkdir(parents=True, exist_ok=True)
    from tools.ck import ckutil as U
    specs = {"done": ((N,), np.bool_, False), "bev_ok": ((N,), np.bool_, False), "obj_ok": ((N,), np.bool_, False),
             "score_logit16": ((N, K16, 5), np.float32, np.nan), "score_logit80": ((N, K80, 5), np.float32, np.nan),
             "w_lat16": ((N, K16, 6), np.float32, np.nan)}
    A = {k: U.open_memmap(out / f"{k}.npy", s, d, fill=f) for k, (s, d, f) in specs.items()}
    if not (out / "tokens.parquet").is_file():
        sub = tdf[[c for c in ("token", "log") if c in tdf.columns]].copy()
        sub["row"] = np.arange(N)
        sub.to_parquet(out / "tokens.parquet", index=False)
    rows = np.flatnonzero(pok)
    if a.limit:
        rows = rows[: a.limit]
    done = A["done"]
    rr = rows[~np.asarray(done[rows], bool)]
    ck_file = run / ("ckpt_last.pt" if a.which == "last" else f"ckpt_{a.which}.pt")
    meta = dict(teacher=name, run=str(run), which=a.which, arm=cfg.get("arm"), ckpt=str(ck_file),
                ckpt_sha16=sha(ck_file) if ck_file.is_file() else None, split=SPLIT, n=N, limit=int(a.limit),
                device=str(dev), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), batch=a.batch,
                workers=a.workers, amp="fp16 autocast" if dev.type == "cuda" else "fp32 (cpu)",
                temporal=(tc.to_dict() if tc is not None else {"on": False}), privileged=priv,
                cand_sha16=sha(pdir / "cand.npy"), variants=str(VARIANTS), variants_sha16=sha(VARIANTS),
                code_sha256={"tools/ck/v3/infer_val_v3.py": sha(__file__, 64),
                             **{f: sha(Path(REPO) / f, 64) for f in MODEL_FILES if (Path(REPO) / f).is_file()}},
                git_head_run=cfg.get("git_head"), started=kst(t0),
                provenance={"train": cfg.get("provenance"), "infer": infer_provenance(tc)})
    log(f"[infer] {name} (temporal {'on ' + tc.forecaster if on else 'off'}, privileged {priv}) -> {out}: "
        f"todo {len(rr)} of {len(rows)}")
    fcm = None
    if on:
        from navsim.agents.para_ssr.ck.temporal.metrics import ForecastMetrics
        fcm = {"arm": ForecastMetrics(), "ca_reference": ForecastMetrics()}
    n, n_bad, n_obj_bad, t1 = 0, 0, 0, time.time()
    if len(rr):
        kw = dict(bev="T", k=K16, labels=None, gt=False, rows=rr)
        if on:
            kw["objects"] = {"cfg": tc.to_dict(), "aug": False, "targets": True}
        ds = CKDataset(SPLIT, **kw)
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, collate_fn=collate_ck, num_workers=a.workers,
                        pin_memory=dev.type == "cuda", prefetch_factor=4 if a.workers > 0 else None)
        ca_b = None
        if on:
            from navsim.agents.para_ssr.ck.temporal import branch as TB
            if getattr(net.temporal, "ca_b", None) is not None:
                ca_b = net.temporal.ca_b.detach().to(dev)
            elif Path(tc.ca_scales).is_file():
                ca_b = TB.load_ca_b(tc.ca_scales).to(dev)
        with torch.no_grad():
            for b in dl:
                prow = b["rows"].numpy().astype(np.int64)
                bok = b["bev_ok"].numpy().astype(bool).reshape(-1)
                ook = b["obj_ok"].numpy().astype(bool).reshape(-1) if on else np.ones_like(bok)
                ok = bok                    # object-free rows are scored too (obj_ok only recorded)
                n_bad += int((~bok).sum())
                n_obj_bad += int((~ook).sum())
                x = {"bev": b["bev"].to(dev, non_blocking=True), "cand": b["cand"].to(dev).float(),
                     "status": b["status"].to(dev).float()}
                var = torch.from_numpy(np.asarray(var_all[prow], np.float32)).to(dev)
                objs = objs_of(b, tc, dev)
                kw_f = {"objs": objs} if objs is not None else {}
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    o = net(x["bev"], x["cand"], x["status"], extra=var, decode=False, **kw_f)
                g = lambda t: t.float().cpu().numpy()[ok]  # noqa: E731
                q = prow[ok]
                A["score_logit16"][q] = g(o["score_logit"])
                A["score_logit80"][q] = g(o["extra_score_logit"])
                A["w_lat16"][q] = g(o["w_lat"])
                A["bev_ok"][prow] = bok
                A["obj_ok"][prow] = ook
                done[q] = True
                if fcm is not None and "fc_xyh" in b:
                    tgt = (b["fc_xyh"].to(dev), b["fc_state"].to(dev), objs["obj_mask"][:, 0])
                    fc = o.get("fc")
                    if fc is None:
                        fc = net.temporal.scene(objs).fc
                    fcm["arm"].update(fc, *tgt)
                    if ca_b is not None:
                        from navsim.agents.para_ssr.ck.temporal.forecast import ca_forecast, trim_objects
                        oi, om, fv = trim_objects(objs["obj_in"].float(), objs["obj_mask"], objs["frame_valid"])
                        fcm["ca_reference"].update(ca_forecast(oi, om, fv, ca_b), *tgt)
                n += len(prow)
                if n % (a.batch * 200) < a.batch or n == len(rr):
                    el = time.time() - t1
                    log(f"[infer] {n}/{len(rr)} {el:.0f}s {n / max(el, 1e-6):.1f} tok/s bev_bad {n_bad} "
                        f"obj_bad {n_obj_bad}")
        for m in A.values():
            m.flush()
    D = np.asarray(done, bool)
    save_npy(out / "prob16.npy", sigmoid(np.asarray(A["score_logit16"])))
    save_npy(out / "prob80.npy", sigmoid(np.asarray(A["score_logit80"])))
    if fcm is not None and fcm["arm"].sums["n_obj"] > 0:
        old = U.read_json(out / "fc_metrics.json", {}) or {}
        if old and len(rr) < len(rows):
            log("[infer] fc_metrics.json kept from the earlier session (resumed run: metrics cover the first session)")
        else:
            U.write_json(out / "fc_metrics.json", {
                "forecaster": tc.forecaster, "privileged": priv, "split": SPLIT, "clean_pack": True,
                "arm": fcm["arm"].result(), "ca_reference": fcm["ca_reference"].result()
                if fcm["ca_reference"].sums["n_obj"] > 0 else None, "n_rows": int(len(rr)), "created": kst()})
    meta.update(n_rows=int(len(rows)), n_done=int(D.sum()), n_bev_bad_this_call=n_bad, n_obj_bad_this_call=n_obj_bad,
                gpu_loop_sec=round(time.time() - t1, 1), sec=round(time.time() - t0, 1), finished=kst())
    # --limit: the PDMS comparison is meaningless on a subset, the per-token logit check (temporal runs) is not
    meta["check"] = reproduce_check(run, np.asarray(np.load(out / "prob16.npy")), D,
                                    np.asarray(A["score_logit16"]) if on else None) if (on or not a.limit) else \
        {"skipped": "--limit"}
    if a.limit and isinstance(meta["check"], dict):
        meta["check"]["limit"] = int(a.limit)
    old = U.read_json(out / "infer_meta.json", {}) or {}
    meta["sessions"] = (old.get("sessions") or []) + [dict(started=meta["started"], finished=kst(), n=n,
                                                           sec=round(time.time() - t0, 1))]
    U.write_json(out / "infer_meta.json", meta)
    chk = meta["check"]
    log(f"[infer] done {n} rows ({n_bad} without BEV, {n_obj_bad} without objects) in {meta['sec']} s; check {chk}")
    if isinstance(chk, dict) and chk.get("ok") is False:
        log(f"[infer] WARNING reproduce check not ok (reported in infer_meta.json; decide_step1 requires it): {chk}")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--which", default="last")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-privileged", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N packed rows only (CPU checks); skips the check")
    a = ap.parse_args(argv)
    if a.workers > 16:
        raise SystemExit("--workers <= 16")
    infer(a)


if __name__ == "__main__":
    main()
