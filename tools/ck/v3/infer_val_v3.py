#!/usr/bin/env python
"""refiner-v3 Step-1 inference on navtrain_val r34 top-16 + 80 variants (SPEC section 6; builder LOSS+EVAL,
2026-10-11 KST).  Port of ck2/analysis/lead_remedy/scripts/infer_val_ck2T.py stage infer (+ check), for temporal
runs (arms a / b / d) and temporal-off runs (arm c = ck2/train/ck2T10).

  cd /workspace/yongjae/SSR-v3 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/yongjae/SSR-v3 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 10 /venv/ssr/bin/python tools/ck/v3/infer_val_v3.py --run <run dir> \
      [--which last] [--out D3/infer/<run name>] [--allow-privileged] [--device cuda] [--batch 8] [--workers 6]
  CPU check (no GPU): CUDA_VISIBLE_DEVICES= ... --device cpu --limit 64 --out /tmp/...

GPU: only GPU 0, 1, 4 or 5 (CUDA_VISIBLE_DEVICES exactly one of them; 4 / 5 user 2026-10-12, short jobs, idle only).  Run by the main
session after training (~4 min per arm [추정 from ck2T10's 207 s]); builders run it on CPU only.

Inputs: every packed navtrain_val row (18,179), CKDataset(bev 'T', k 16, labels None, gt False, objects =
{cfg, aug False, targets True} when the run's temporal cfg is on; NO objects kwarg when off, so the CK2 path is
untouched), extra = lead_remedy/infer/variants/traj80.npy (the variants of labels96.npz), model = load_ck(run,
which, allow_privileged) (refuses privileged runs unless --allow-privileged), forward = CKNet.forward(bev, cand,
status, extra = variants, decode = False[, objs = OBJ_INPUT_KEYS (+ PRIV_INPUT_KEYS for oracle_gt)]) under fp16
autocast on cuda (as train_ck2 eval), fp32 on cpu.  Nothing GT-derived reaches the model except the PRIV keys of a
privileged oracle run.

Outputs (OUT, default D3/infer/<run name> for --which last and D3/infer/<run name>_<which> otherwise; resumable via
done.npy): score_logit16 [N,16,5], prob16, score_logit80
[N,80,5], prob80 (sigmoid, CK_KEYS nc dac ep ttc comfort), w_lat16 [N,16,6], done [N], bev_ok [N], obj_ok [N]
(True for temporal-off runs), tokens.parquet (token, log, row), cache_key.json, infer_meta.json (run, which, ckpt
path / sha16, temporal cfg, privileged, cand_sha16 of packed cand.npy, variants path / sha16, code sha256 of this
file and the model files, KST times, check, cache_key, sessions).
Cache key (tooling fix F2, 2026-10-11): rows already done in OUT are reused only if they were computed under the same
key = ckpt sha256 + which + run dir, the packed tokens / ok / cand / status shas, the variants sha, the temporal
cfg + privileged flag, the precision (fp16 autocast on cuda / fp32 on cpu), this inference's data provenance (val
object pack build.json sha, ca_scales sha, fc_init sha) and the forward-code sha (the static import closure of this
file inside navsim/ + tools/: model, temporal branch, refiner, CKDataset, obj_pack, ckutil, this file; ~40 files).
The key is written to OUT/cache_key.json BEFORE any row is computed; if OUT has done rows and its cache_key.json is
missing (pre-F2 dump) or differs, the run is refused (SystemExit naming the differing fields): use a new --out (e.g.
<run>_ep7 for --which ep7), or remove OUT deliberately.  infer_meta.json records the key; its sessions list is kept
only across sessions with the same key.
The BEV feature files themselves are not hashed (the shared teacher dump, read-only).
Review 2 (2026-10-12): --which last needs <run>/done.json (a finished run; --allow-unfinished for checks only, decide
refuses such a dump); the checkpoint sha taken before the load must equal the one load_ck takes after reading it
(else SystemExit: the file was replaced while it was loaded).
Temporal runs with targets also write fc_metrics.json: the arm's own forecast (learned:
out['fc']; ca / oracle: TemporalBranch.scene(objs).fc) and the analytic CA reference (fixed scales), both through
metrics.ForecastMetrics on the CLEAN val pack (no augmentation).  The file keeps the additive sums; a same-key resume
merges them, so n_rows counts every row computed under the key (decide_step1 notes n_rows != n_done).
Check (stage check, also run at the end of a full inference; review fix D6: REPORTED in infer_meta.json['check'],
never an exit; decide_step1.py requires check.ok):
  temporal runs  the trainer saved the r34 logits of its last navtrain_val eval (eval_r34_navtrain_val_last.npz,
                 same weights as ckpt_last).  Per token and logit: |dlogit| = |infer - train|; picks = argmax
                 ck_final(sigmoid, im = 1, noim weights) per token.  ok iff the 99.9 % quantile of |dlogit| <= 2^-5,
                 at most 0.1 % of the picks flip and pdms_ck_noim reproduces the val_metrics row (|diff| <= 1e-4,
                 n_sel equal).  CUDA scatter_add order, another batch composition and fp16 make bit equality
                 impossible.  Tolerance (fix 2026-10-12 KST, user-approved after Step 1): the logits are fp16 autocast
                 outputs (every saved value is fp16-representable, |logit| < 32, fp16 spacing 2^-6 there), so the
                 tolerance is 2 fp16 ULP of that range = 2^-5 = 0.03125.  The first tolerance 2e-4 came from fp32-scale
                 CPU checks [review_correctness E1] and refused arm a (q99.9 0.0039 = 2^-8, max 0.0156, 2 / 18,179
                 flipped picks, pdms exact) [실측 2026-10-12]; arm b max 0.027.  Different weights (arm a infer vs
                 arm b training logits) differ by median 0.535, q99.9 11.6 [실측], so 2^-5 still separates them.
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
CHECK_DLOGIT = 2.0 ** -5         # temporal runs: 99.9 % quantile of |dlogit| (2 fp16 ULP at |logit| < 32; docstring)
CHECK_DLOGIT_V1 = 2e-4           # first tolerance (fp32 scale), still reported as frac_dlogit_gt_v1
CHECK_DLOGIT_Q = 0.999
CHECK_FLIP_MAX = 0.001           # at most 0.1 % flipped picks
MODEL_FILES = ("navsim/agents/para_ssr/ck/model.py", "navsim/agents/para_ssr/ck/temporal/branch.py",
               "navsim/agents/para_ssr/ck/temporal/forecast.py", "navsim/agents/para_ssr/ck/temporal/raster.py",
               "navsim/agents/para_ssr/ck/temporal/config.py", "tools/ck/data/ck_dataset.py",
               "tools/ck/data/obj_pack.py")
CACHE_KEY_VERSION = "v3_infer_cache_1"         # decide_step1.CACHE_KEY_PREFIX
# the code the scores depend on = the static import closure of this script inside the repo (navsim/, tools/):
# every module it, the model (CKNet -> ..refiner, temporal) and the readers (CKDataset, obj_pack) import, at module
# level or inside functions, plus their parent packages.  Static (ast), so the key does not depend on what else the
# process happened to import; unrelated modules (e.g. the e2e student code) are not part of it.
FORWARD_CODE_ENTRY = "tools/ck/v3/infer_val_v3.py"
FORWARD_CODE_ROOTS = ("navsim", "tools")
PACKED_KEY_FILES = ("tokens.parquet", "ok.npy", "cand.npy", "status.npy")


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
        # 0-3 are ours (2 / 3 once the e2e runs there are done); 4 / 5: user 2026-10-12, short jobs, idle only
        if cv not in ("0", "1", "2", "3", "4", "5"):
            raise SystemExit(f"cuda inference only on one of GPUs 0-5 (CUDA_VISIBLE_DEVICES one of them), got {cv!r}")
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


def _mod_file(repo: Path, name: str) -> Optional[Path]:
    p = repo / Path(*name.split("."))
    for c in (p.with_suffix(".py"), p / "__init__.py"):
        if c.is_file():
            return c
    return None


def import_closure(entry: str = FORWARD_CODE_ENTRY, repo: Path = Path(REPO)) -> list:
    """repo-relative paths of the static import closure of `entry` within FORWARD_CODE_ROOTS (ast: module-level and
    function-level `import` / `from .. import`, relative imports resolved, parent packages included)."""
    import ast
    seen, todo = set(), [repo / entry]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        rel = f.relative_to(repo).with_suffix("")
        is_pkg = rel.name == "__init__"
        parts = list(rel.parts[:-1] if is_pkg else rel.parts)
        for i in range(1, len(parts)):
            pf = _mod_file(repo, ".".join(parts[:i]))
            if pf is not None:
                todo.append(pf)
        for node in ast.walk(ast.parse(f.read_text(), str(f))):
            names = []
            if isinstance(node, ast.Import):
                names = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = parts if is_pkg else parts[:-1]
                    base = base[:len(base) - (node.level - 1)]
                    mod = ".".join(base + ([node.module] if node.module else []))
                else:
                    mod = node.module or ""
                names = [mod] + [f"{mod}.{x.name}" for x in node.names]
            for nm in names:
                if nm.split(".")[0] in FORWARD_CODE_ROOTS:
                    mf = _mod_file(repo, nm)
                    if mf is not None:
                        todo.append(mf)
    return sorted(str(p.relative_to(repo)) for p in seen)


def forward_code(repo: Path = Path(REPO)) -> Dict:
    """sha256 over the sorted (path, sha256) of the forward-code files (import_closure) + per-file sha16."""
    rows = sorted((f, sha(repo / f, 64)) for f in import_closure(repo=repo))
    return {"sha256": hashlib.sha256("\n".join(f"{a} {b}" for a, b in rows).encode()).hexdigest(),
            "n_files": len(rows), "files": {a: b[:16] for a, b in rows}}


def cache_key(run: Path, which: str, ck_file: Path, tc, priv: bool, amp: str, data_prov: Optional[Dict],
              pdir: Path, ckpt_sha256: Optional[str] = None) -> Dict:
    """Identity of the inputs every computed row depends on (tooling fix F2; module docstring).  JSON-normalised."""
    key = dict(version=CACHE_KEY_VERSION, split=SPLIT, run=str(Path(run).resolve()), which=which,
               ckpt_name=Path(ck_file).name, ckpt_sha256=ckpt_sha256 or sha(ck_file, 64),
               packed={f: sha(pdir / f, 64) for f in PACKED_KEY_FILES},
               variants_sha256=sha(VARIANTS, 64),
               temporal=(tc.to_dict() if tc is not None else {"on": False}), privileged=bool(priv), amp=amp,
               data=data_prov or {}, code=forward_code())
    return json.loads(json.dumps(key))


def key_diff(old, new, path: str = "") -> list:
    """dotted paths where two cache keys differ (refusal message)."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(set(old) | set(new)):
            out += key_diff(old.get(k), new.get(k), f"{path}.{k}" if path else str(k))
        return out
    return [] if old == new else [path or "<root>"]


def check_cache(out: Path, key: Dict) -> bool:
    """Refuse (SystemExit) when OUT already has done rows computed under another or an unknown key (F2).
    -> True if OUT's existing rows / sessions belong to this key."""
    dn = out / "done.npy"
    n_old = int(np.asarray(np.load(dn, mmap_mode="r"), bool).sum()) if dn.is_file() else 0
    kf = out / "cache_key.json"
    old = json.loads(kf.read_text()) if kf.is_file() else None
    if n_old:
        if old is None:
            raise SystemExit(f"{out}: {n_old} rows are already done but there is no cache_key.json (a dump written "
                             f"before the cache-key fix): their checkpoint / inputs cannot be verified, refusing to "
                             f"reuse them.  Use a new --out (or remove {out} deliberately).")
        diff = key_diff(old, key)
        if diff:
            raise SystemExit(f"{out}: {n_old} rows were computed under another cache key (differs in: "
                             f"{', '.join(diff[:12])}{' ...' if len(diff) > 12 else ''}).  Refusing to mix them with "
                             f"this run: use a new --out (e.g. <run>_<which>) or remove {out} deliberately.")
    return old is not None and not key_diff(old, key)


def default_out_name(name: str, which: str) -> str:
    """default infer dir name under D3/infer: <name> for ckpt_last, <name>_<which> otherwise."""
    return name if which == "last" else f"{name}_{which}"


def fc_payload(fcm: Dict, n_rows: int) -> Dict:
    """result + additive sums of the arm / CA ForecastMetrics (the sums let a resumed session merge, finding 5)."""
    ca = fcm["ca_reference"]
    return {"arm": fcm["arm"].result(), "ca_reference": ca.result() if ca.sums["n_obj"] > 0 else None,
            "sums": {"arm": dict(fcm["arm"].sums), "ca_reference": dict(ca.sums)}, "n_rows": int(n_rows)}


def merge_fc(old: Optional[Dict], fcm: Dict, n_new: int):
    """-> (fcm merged with an earlier session's sums, total rows, merged?) .  old = the fc_metrics.json of an
    earlier session with the SAME cache key (None / without sums: nothing to merge)."""
    from navsim.agents.para_ssr.ck.temporal.metrics import ForecastMetrics
    sums = (old or {}).get("sums")
    if not sums or set(sums) != {"arm", "ca_reference"}:
        return fcm, int(n_new), False
    out = {}
    for k in ("arm", "ca_reference"):
        prev = ForecastMetrics()
        if set(sums[k]) != set(prev.KEYS):
            return fcm, int(n_new), False
        prev.sums = {kk: float(v) for kk, v in sums[k].items()}
        out[k] = prev.merge(fcm[k])
    return out, int(old.get("n_rows", 0)) + int(n_new), True


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
               frac_dlogit_gt_v1=float((d > CHECK_DLOGIT_V1).mean()) if d.size else None,
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
        out["rule"] = ("per-token logits (temporal run): q99.9 |dlogit| <= 2^-5 (fp16), <= 0.1 % flipped picks and "
                       "pdms_ck_noim |diff| <= 1e-4 with n_sel equal")
        out["ok"] = bool(lc["ok"] and out.get("pdms_ok") is True)
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
    run = Path(os.path.abspath(a.run))              # absolute: decide_step1 re-hashes meta['ckpt'] from anywhere
    dev = torch.device("cuda:0" if a.device.startswith("cuda") else "cpu")
    ck_file = run / ("ckpt_last.pt" if a.which == "last" else f"ckpt_{a.which}.pt")
    if not ck_file.is_file():
        raise SystemExit(f"{ck_file} missing (the cache key and decide_step1 need the checkpoint file)")
    run_done = (run / "done.json").is_file()
    if a.which == "last" and not run_done and not getattr(a, "allow_unfinished", False):
        raise SystemExit(f"{run} has no done.json: training is not finished, ckpt_last.pt is an intermediate "
                         f"checkpoint (and may be replaced while it is read).  Wait for done.json "
                         f"(--allow-unfinished for a check only; decide_step1 refuses such a dump).")
    ck_sha256 = sha(ck_file, 64)                    # hashed before loading: the key names the weights scored
    net, cfg, tc = load_model(run, a.which, str(dev), a.allow_privileged)
    net.eval()
    # review 2 finding 6: load_ck hashes the file again after reading it; a different sha means the checkpoint was
    # replaced between the key hash and the load (e.g. a still-running trainer saving ckpt_last.pt)
    got = (getattr(net, "init_report", None) or {}).get("ckpt_sha16") or sha(ck_file)
    if got != ck_sha256[:16]:
        raise SystemExit(f"{ck_file} changed while it was loaded (sha16 {ck_sha256[:16]} before, {got} after): "
                         f"the weights scored are unknown; rerun when the checkpoint is no longer being written")
    if cfg.get("arm") != "T":
        raise SystemExit(f"{run}: arm {cfg.get('arm')!r} (Step 1 evaluates DET teachers, arm T)")
    on = bool(tc is not None and tc.on)
    priv = bool(on and tc.privileged)
    if priv and not a.allow_privileged:
        raise SystemExit(f"{run} is PRIVILEGED: pass --allow-privileged (analysis only)")
    name = cfg.get("run") or run.name
    # review 2 finding 4: the default dir of a non-last checkpoint is <name>_<which>, so a descriptive ep7 / ep8
    # infer never takes (and then blocks) the decision dump's default dir D3/infer/<name>
    out = Path(a.out) if a.out else D3 / "infer" / default_out_name(name, a.which)
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
    amp = "fp16 autocast" if dev.type == "cuda" else "fp32 (cpu)"
    prov_infer = infer_provenance(tc)
    # F2: the key is checked against OUT before anything is written and saved before any row is computed
    key = cache_key(run, a.which, ck_file, tc, priv, amp, prov_infer.get("data"), pdir, ck_sha256)
    same_key = check_cache(out, key)
    out.mkdir(parents=True, exist_ok=True)
    from tools.ck import ckutil as U
    U.write_json(out / "cache_key.json", key)
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
    meta = dict(teacher=name, run=str(run), which=a.which, arm=cfg.get("arm"), ckpt=str(ck_file),
                ckpt_sha16=key["ckpt_sha256"][:16], split=SPLIT, n=N, limit=int(a.limit),
                device=str(dev), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), batch=a.batch,
                workers=a.workers, amp=amp,
                temporal=(tc.to_dict() if tc is not None else {"on": False}), privileged=priv,
                cand_sha16=key["packed"]["cand.npy"][:16], variants=str(VARIANTS),
                variants_sha16=key["variants_sha256"][:16],
                code_sha256={"tools/ck/v3/infer_val_v3.py": sha(__file__, 64),
                             **{f: sha(Path(REPO) / f, 64) for f in MODEL_FILES if (Path(REPO) / f).is_file()}},
                git_head_run=cfg.get("git_head"), run_done=run_done, started=kst(t0),
                provenance={"train": cfg.get("provenance"), "infer": prov_infer}, cache_key=key)
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
        # review 2 finding 5: a same-key resume MERGES the earlier session's sums, so the metrics cover every row
        # computed under this key (n_rows); decide_step1 notes fc n_rows != n_done.  A different-key dir with done
        # rows was refused above, so an old file here is either same-key or belongs to rows that are all re-done.
        old = (U.read_json(out / "fc_metrics.json", {}) or {}) if same_key else {}
        fcm_all, n_fc, merged = merge_fc(old, fcm, len(rr))
        if old and not merged:
            log("[infer] fc_metrics.json of the earlier session has no sums: replaced by this session's metrics "
                f"({len(rr)} rows)")
        U.write_json(out / "fc_metrics.json", {
            "forecaster": tc.forecaster, "privileged": priv, "split": SPLIT, "clean_pack": True,
            **fc_payload(fcm_all, n_fc), "n_sessions_merged": int(old.get("n_sessions_merged", 1)) + 1 if merged else 1,
            "created": kst()})
    meta.update(n_rows=int(len(rows)), n_done=int(D.sum()), n_bev_bad_this_call=n_bad, n_obj_bad_this_call=n_obj_bad,
                gpu_loop_sec=round(time.time() - t1, 1), sec=round(time.time() - t0, 1), finished=kst())
    # --limit: the PDMS comparison is meaningless on a subset, the per-token logit check (temporal runs) is not
    meta["check"] = reproduce_check(run, np.asarray(np.load(out / "prob16.npy")), D,
                                    np.asarray(A["score_logit16"]) if on else None) if (on or not a.limit) else \
        {"skipped": "--limit"}
    if a.limit and isinstance(meta["check"], dict):
        meta["check"]["limit"] = int(a.limit)
    old = U.read_json(out / "infer_meta.json", {}) or {}
    # sessions of an earlier key (whose rows were all not-done, else check_cache refused) are not carried over
    prev = (old.get("sessions") or []) if (same_key and old.get("cache_key") == key) else []
    meta["sessions"] = prev + [dict(started=meta["started"], finished=kst(), n=n, sec=round(time.time() - t0, 1))]
    U.write_json(out / "infer_meta.json", meta)
    chk = meta["check"]
    log(f"[infer] done {n} rows ({n_bad} without BEV, {n_obj_bad} without objects) in {meta['sec']} s; check {chk}")
    if isinstance(chk, dict) and chk.get("ok") is False:
        log(f"[infer] WARNING reproduce check not ok (reported in infer_meta.json; decide_step1 requires it): {chk}")
    return meta


def recheck(a) -> Dict:
    """Recompute the reproduce check of a finished dump (--out, required) with the current tolerance, from its saved
    prob16 / score_logit16 / done arrays; the run's checkpoint must still hash to the dump's cache key (the check is
    about those weights).  Rewrites infer_meta.json: check = new, check_history += [old check, KST, rule]."""
    from tools.ck import ckutil as U
    if not a.out:
        raise SystemExit("--recheck needs --out <dump dir>")
    out = Path(a.out)
    meta = U.read_json(out / "infer_meta.json", None)
    if not meta:
        raise SystemExit(f"{out}/infer_meta.json missing")
    run = Path(meta["run"])
    if Path(os.path.abspath(a.run)) != run:
        raise SystemExit(f"--run {a.run} != the dump's run {run}")
    key = meta.get("cache_key") or {}
    ck_file = Path(meta["ckpt"])
    if not key.get("ckpt_sha256") or sha(ck_file, 64) != key["ckpt_sha256"]:
        raise SystemExit(f"{ck_file} no longer hashes to the dump's cache key: the check would compare other weights")
    if int(meta.get("limit") or 0):
        raise SystemExit("dump made with --limit: no check to recompute")
    on = bool((meta.get("temporal") or {}).get("on"))
    D = np.load(out / "done.npy").astype(bool)
    new = reproduce_check(run, np.asarray(np.load(out / "prob16.npy")), D,
                          np.asarray(np.load(out / "score_logit16.npy")) if on else None)
    new["recheck"] = {"at": kst(), "infer_val_v3_sha16": sha(__file__)}
    meta["check_history"] = (meta.get("check_history") or []) + [
        {"check": meta.get("check"), "replaced": kst(), "by_rule": new.get("rule")}]
    meta["check"] = new
    U.write_json(out / "infer_meta.json", meta)
    log(f"[recheck] {out}: ok {new.get('ok')} ({new.get('rule')}); previous ok "
        f"{(meta['check_history'][-1]['check'] or {}).get('ok')}")
    return new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--which", default="last")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-privileged", action="store_true")
    ap.add_argument("--allow-unfinished", action="store_true",
                    help="checks only: allow --which last of a run without done.json (decide refuses the dump)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N packed rows only (CPU checks); skips the check")
    ap.add_argument("--recheck", action="store_true",
                    help="recompute infer_meta.json['check'] from the saved arrays of --out (no model, no GPU); the "
                         "old check is kept in check_history")
    a = ap.parse_args(argv)
    if a.workers > 16:
        raise SystemExit("--workers <= 16")
    if a.recheck:
        recheck(a)
        return
    infer(a)


if __name__ == "__main__":
    main()
