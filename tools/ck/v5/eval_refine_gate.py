#!/usr/bin/env python
"""refiner v5 Stage-1 (3) gate (SSR/report/97_refiner_v5_design.md section 5-2): the corrector A of one or more v5 nets
on FIXED drafts, every trajectory scored by the official scorer, paired per token.  GPU only for stage infer with
--device cuda (approval + an idle GPU of 0-3); the default CPU route needs no GPU.

  cd /workspace/yongjae/SSR-v5 && CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      nice -n 10 /venv/ssr/bin/python tools/ck/v5/eval_refine_gate.py --run T=/home/external-user/ssd/yongjae_refiner/ck/v5/train/v5T_rel \
      --which ep2 --split navtrain_val --device cpu --label-workers 32
  (--split navtrain_val defaults to --tokens heldout12179 = the registered gate set; --limit 4 = a CPU smoke into
   <out>/<split>/smoke4_<tokens>; the MAP teacher: --run v5M_unk --split navtest, navtest defaults to --tokens all.)

Arms (--run [ROLE=]DIR, repeatable; DIR = a run dir or a name under CK/v5/train | CK/v4/train | CK/ck2/train)
  ROLE T = the teacher-input refiner A_T (the DET teacher on val; the MAP teacher's gate on navtest uses T too), S =
  the student-input A_S, GT = the GT-input A_GT (each at most once), or no role (report only, labelled by the run dir
  name).  A run must be a v5 net with the corrector A (config.json v5
  corrector) on the range readout, arm T (DET, BEVFusion t0 BEV) or M (MAP, ReSMap BEV).  Refused: arm M on
  navtrain_val (no ReSMap cache for val: constants.SPLITS map_teacher False; 0 of the 12,179 held-out tokens are in the
  navtrain ReSMap index -- the MAP gate runs on navtest only), relation 'student' (Stage 1b needs the student's det rows,
  not wired here), arm S, a privileged (v4.rg_priv / temporal oracle) run, a temporal-branch run.
Drafts (--drafts): r34top1 (the only one) = packed/<split>/cand.npy[:, 0] = the v2 r34 top-1 (= sub_traj, v2_final
  argmax) of the Phase-1 packed set; its stored official labels are CK/labels/<split>/cand[:, 0].
Tokens (--tokens): heldout12179 (navtrain_val only) = the registered gate set = eval_range_teacher.p1_registered_rows
  (HO/prep/ho_rows.npy: the 12,179 packed navtrain_val rows outside the F1 6,000, the v4_3 registered P1 token set);
  all = every packed ok row; <file> = .npy of packed rows (int) or tokens (str), or a text file of tokens (one per line).

Stages (--stage comma list, default infer,label,report; each resumable)
  infer   per arm: the checkpoint ckpt_<--which>.pt (last | ep<e>) is COPIED to <arm>/ckpt_<which>.pt first (a running
          training replaces ckpt_last.pt atomically; the copy is what is evaluated, its sha16 is the identity; a later
          change of the run's file is reported, never picked up: --fresh re-copies); load_ck(copy) (strict) on --device;
          CKDataset(split, bev = the arm, k = --score-k, rows, objects = the refiner-v3 object pack rows of the tokens
          (e2e_data2.teacher_v5_rel_cfg: the run's t_pack_root, no augmentation, no targets) when v5.relation is
          'teacher'); trunk.scene -> scene_pref -> score_set(the --score-k first candidates, each its own parent,
          return_ctx, rel_objs) -> refine_drafts(cols = 0, fp32) = the ops of CKNet._forward_v4(refine_cols) ->
          <arm>/traj3.npy f32 [N, 3, 8, 3] (column 0 tau0 = the draft (checked bitwise = packed cand[:, 0]), 1 tau_base =
          mu 0 = the lateral head, 2 tau1), mu.npy f32 [N, 12], done / bev_ok / obj_ok [N] (N = packed rows; NaN / False
          outside the tokens), infer_meta.json.  CUDA: score_set under fp16 autocast (as eval_range_teacher), refine fp32;
          CPU: fp32.  Resume only into outputs of the same identity (INFER_ID_KEYS: run, which, checkpoint sha16, arm,
          split, drafts, score_k, knockout, amp, v4 / v5 config; the run dir compared resolved).  A token without BEV stays not done (counted).
          --knockout objects (report only, design section 2-8 'object off'): obj_mask / frame_valid False -> every
          a_i = 0, H = 0; outputs in <run>@<which>@objoff.  ('dense off' has no code hook: refused.)
  label   tools/ck/data/label_cands.py (score_trajectories.score_token = the official pairwise PDM score per
          trajectory, LQR; a subprocess with CK_DATA_ROOT unset, --label-workers <= 48) on traj3 for the done rows whose
          three trajectories are finite (a non-finite one is counted, never scored) -> <arm>/labels/{labels [N, 3, 9],
          ok [N, 3], meta.json, shards/}; reused only when label_key.json (traj3 sha16 + rows sha16) is unchanged, else
          rescored from scratch.  Check (label_check.json): column 0 must reproduce the stored draft labels
          CK/labels/<split>/cand[:, 0] (ok equal, |diff| <= LABEL_TOL on every finite column), else SystemExit.
  report  token set = the selected rows that every arm has done with all three columns labelled ok; the registered
          set's completeness per input.  Per arm (x 100 units in the deltas): PDMS / NC / DAC / EP / TTC / C / DDC
          means, fail rates and pass rates of draft, tau_base, tau1; paired log-cluster bootstrap (eval_ck.log_bootstrap,
          --bootstrap 2,000, seed 0, ratio over logs) of tau1 - tau_base, tau1 - draft, tau_base - draft; break / fix of
          tau1 (and tau_base) against both references under both rules -- pdms0: P(new PDMS == 0 | ref PDMS > 0), fix
          P(new PDMS > 0 | ref PDMS == 0); fails3 (eval_range_teacher.fails3, the v4_3 rule): P(new fails NC / TTC / DAC
          | ref passes all three); per-key (nc, dac, ttc, ddc, comfort) fail / break / fix counts, EP loss; |mu|
          (L_inf over the 12 controls) mean, share > MU_THR (0.05), per-control mean |mu|; share of tokens whose tau1
          moves (> 1 mm) from the draft / from tau_base.  Between arms: tau1_a - tau1_b paired.
          -> <root>/gate_<arms>[_<tag>].json / .md / _per_token.parquet (+ the printed table).

Gate (registered before any v5 result; report 97 section 5-2, the four clauses):
  (a) PDMS(tau1^T) - PDMS(tau1^S) >= +0.30 with the paired CI lower bound > 0;   (needs T and S)
  (b) that difference >= 0.5 x (PDMS(tau1^GT) - PDMS(tau1^S)) (point estimates);  (needs T, S and GT)
  (c) break(tau1^T vs no edit) <= 1.3 % (--break-rule, default pdms0);
  (d) tau1^T - no edit >= +0.30 with CI lower bound > 0.
  'no edit' = --noedit: draft (default: tau0, the unmodified r34 top-1, = report 97's "pass -> fail of tokens the draft
  passed") | base (tau_base, mu = 0 = the arm's lateral head alone).  Both references are always reported.
  Verdict PASS iff all four hold, FAIL iff an evaluable clause fails, else INCOMPLETE (a clause lacks its arm: one T run
  alone can only FAIL or stay INCOMPLETE).  It DECIDES only on navtrain_val --tokens heldout12179 without --limit,
  --drafts r34top1, no knock-out, with all 12,179 registered tokens in the token set; every other run is 'report only'
  (with the reason).  Also reported: (GT - T) = "what no teacher gives", the student floor (report 97: tau1^S vs no edit
  >= 0 with CI lower bound > -0.10, break <= 1.3 %).  Not here: the tier table (T1/T2/T3/DAC) of section 5-1.
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
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402
from tools.ck.v5.refine_stats import (BREAK_RULES, KEYS5, MOVE_M, boot100, break_fix, key_counts,  # noqa: E402
                                      moved, mu_stats)

CK = Path("/home/external-user/ssd/yongjae_refiner/ck")
OUT_ROOT = CK / "v5" / "eval_gate"
RUN_ROOTS = (CK / "v5" / "train", CK / "v4" / "train", CK / "ck2" / "train")
SPLITS = ("navtrain_val", "navtest")
TOKEN_SETS = ("heldout12179", "all")
REG_TOKENS, REG_SPLIT, N_REGISTERED = "heldout12179", "navtrain_val", 12179
DRAFTS = ("r34top1",)
COLS = ("draft", "base", "tau1")            # traj3 / labels column order: tau0, tau_base (mu = 0), tau1
ROLES = ("T", "S", "GT")
NOEDIT = ("draft", "base")
KNOCKOUTS = ("none", "objects")
N_BOOT, BOOT_SEED = 2000, 0
GATE_D_MIN, GATE_BREAK_MAX, GATE_GT_FRAC = 0.30, 0.013, 0.5      # report 97 section 5-2
FLOOR_D_MIN, FLOOR_LO_MIN = 0.0, -0.10                           # the student-channel floor (report only here)
LABEL_TOL = 1e-6                                                 # draft column vs the stored float32 labels
MAX_LABEL_WORKERS = 48
INFER_ID_KEYS = ("run_real", "which", "ckpt_sha16", "arm", "split", "drafts", "score_k", "knockout", "amp", "v4", "v5",
                 "corridor_version")                    # run_real = the resolved run dir (symlinked paths are equal)
STAGES = ("infer", "label", "report")
GATE_RULE = ("report 97 section 5-2 (3) gate, registered before any v5 result: PASS iff (a) PDMS(tau1^T) - PDMS(tau1^S) "
             ">= +0.30 with the paired log-bootstrap CI lower bound > 0, (b) that difference >= 0.5 x (PDMS(tau1^GT) - "
             "PDMS(tau1^S)), (c) break(tau1^T vs no edit) <= 1.3 %, (d) tau1^T - no edit >= +0.30 with CI lower bound > "
             "0; on navtrain_val held-out 12,179 tokens, fixed r34 top-1 drafts, official scorer, x 100 units.")


def kst(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime((time.time() if t is None else t) + 9 * 3600))


def log(msg: str) -> None:
    print(f"[{kst()}] [v5gate] {msg}", flush=True)


def _sha16_rows(r: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(r, np.int64)).tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- inputs
def packed(split: str):
    """CK/packed/<split> (the Phase-1 v2 r34 dump: cand [N, 16, 8, 3], tokens.parquet, ok [N])."""
    d = CK / "packed" / split
    return d, pd.read_parquet(d / "tokens.parquet"), np.load(d / "ok.npy").astype(bool)


def stored_draft_labels(split: str, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """official labels of the r34 top-1 draft: CK/labels/<split>/cand[:, 0] -> (labels [n, 9] f32, ok [n])."""
    d = CK / "labels" / split / "cand"
    lab, ok = np.load(d / "labels.npy", mmap_mode="r"), np.load(d / "ok.npy", mmap_mode="r")
    if lab.shape[0] != n or ok.shape[0] != n:
        raise SystemExit(f"{d}: labels {lab.shape} / ok {ok.shape} do not have the {n} packed rows")
    return np.asarray(lab[:, 0], np.float32), np.asarray(ok[:, 0], bool)


def resolve_run(name: str) -> Path:
    p = Path(name)
    if p.is_dir():
        return p.absolute()
    for q in RUN_ROOTS:
        if (q / name).is_dir():
            return q / name
    raise SystemExit(f"--run {name}: no such run dir (looked under {', '.join(str(q) for q in RUN_ROOTS)})")


def parse_runs(specs: Sequence[str]) -> List[Dict[str, Any]]:
    """--run [ROLE=]DIR values -> [{role (T / S / GT / None), run (Path), label}] (label = role or the run dir name);
    a role given twice, the same run twice, or two equal labels -> SystemExit."""
    out: List[Dict[str, Any]] = []
    for s in specs:
        role, _, name = s.partition("=") if "=" in s else ("", "", s)
        role = role.strip() or None
        if role is not None and role not in ROLES:
            raise SystemExit(f"--run {s!r}: role {role!r} not in {ROLES} (or omit 'ROLE=')")
        if not name.strip():
            raise SystemExit(f"--run {s!r}: no run")
        run = resolve_run(name.strip())
        out.append({"role": role, "run": run, "label": role or run.name})
    roles = [r["role"] for r in out if r["role"]]
    if len(roles) != len(set(roles)):
        raise SystemExit(f"--run: role given twice ({roles})")
    runs = [str(r["run"]) for r in out]
    if len(runs) != len(set(runs)):
        raise SystemExit(f"--run: the same run given twice ({runs})")
    labels = [r["label"] for r in out]
    if len(labels) != len(set(labels)):
        raise SystemExit(f"--run: two arms with the label {labels} (give roles)")
    return out


def check_which(which: str) -> str:
    if which != "last" and not re.fullmatch(r"ep\d+", which):
        raise SystemExit(f"--which {which!r}: last | ep<e>")
    return which


def default_tokens(split: str, tokens: Optional[str]) -> str:
    return tokens or (REG_TOKENS if split == REG_SPLIT else "all")


def select_rows(split: str, tokens: str, tdf: pd.DataFrame, pok: np.ndarray) -> Tuple[np.ndarray, str, Dict]:
    """--tokens -> (sorted unique packed rows, tag, info).  heldout12179: the registered set (navtrain_val only; every
    row must be a packed ok row); all: the packed ok rows; a file of rows / tokens (unknown tokens / rows out of range
    -> SystemExit; rows that are not packed ok are dropped and counted)."""
    n = len(tdf)
    info: Dict[str, Any] = {"tokens": tokens}
    if tokens == REG_TOKENS:
        if split != REG_SPLIT:
            raise SystemExit(f"--tokens {REG_TOKENS}: the registered set is navtrain_val's (this is --split {split})")
        from tools.ck.v4 import eval_range_teacher as E
        reg, why = E.p1_registered_rows()
        if reg is None:
            raise SystemExit(f"--tokens {REG_TOKENS}: {why}")
        if len(reg) != N_REGISTERED or int(reg.max()) >= n:
            raise SystemExit(f"--tokens {REG_TOKENS}: {len(reg)} registered rows (max {int(reg.max())}) vs the "
                             f"expected {N_REGISTERED} of {n} packed rows")
        if not pok[reg].all():
            raise SystemExit(f"--tokens {REG_TOKENS}: {int((~pok[reg]).sum())} registered rows are not packed ok")
        info.update(source=str(E.HO_DIR / "prep" / "ho_rows.npy"), n_registered=int(len(reg)))
        return reg.astype(np.int64), REG_TOKENS, info
    if tokens == "all":
        r = np.flatnonzero(pok).astype(np.int64)
        info.update(source="packed ok")
        return r, "all", info
    p = Path(tokens)
    if not p.is_file():
        raise SystemExit(f"--tokens {tokens!r}: heldout12179 | all | an existing .npy / text file")
    if p.suffix == ".npy":
        x = np.load(p, allow_pickle=False)
    else:
        x = np.array([ln.strip() for ln in p.read_text().splitlines() if ln.strip()])
    if x.dtype.kind in "iu":
        r = np.unique(x.astype(np.int64))
        if len(r) and (int(r.min()) < 0 or int(r.max()) >= n):
            raise SystemExit(f"--tokens {p}: packed rows out of range [0, {n})")
    elif x.dtype.kind in "US":
        row_of = pd.Series(np.arange(n), index=tdf.token.astype(str).to_numpy())
        xs = x.astype(str)
        miss = ~pd.Index(xs).isin(row_of.index)
        if miss.any():
            raise SystemExit(f"--tokens {p}: {int(miss.sum())} tokens are not in packed/{split} (e.g. {xs[miss][:3]})")
        r = np.unique(row_of.reindex(xs).to_numpy().astype(np.int64))
    else:
        raise SystemExit(f"--tokens {p}: dtype {x.dtype} (packed rows int or tokens str)")
    bad = ~pok[r]
    info.update(source=str(p.resolve()), n_not_packed_ok=int(bad.sum()))
    r = r[~bad]
    return r, f"file-{p.stem}-{_sha16_rows(r)[:8]}", info


def gate_root(out: str, split: str, tag: str, limit: int) -> Path:
    return Path(out) / split / (f"smoke{limit}_{tag}" if limit else tag)


def arm_dir(root: Path, run: Path, which: str, knockout: str) -> Path:
    return root / f"{run.name}@{which}{'@objoff' if knockout == 'objects' else ''}"


def run_config(run: Path) -> Dict:
    p = run / "config.json"
    if not p.is_file():
        raise SystemExit(f"{run}: no config.json")
    return json.loads(p.read_text())


def check_run_cfg(run: Path, cfg: Dict, split: str, knockout: str = "none") -> Dict[str, Any]:
    """the refusals that need config.json only (before any checkpoint is read) -> {arm, v5 (V5Cfg)}."""
    from navsim.agents.para_ssr.ck.temporal.config import TemporalCfg
    from navsim.agents.para_ssr.ck.v5.config import V5Cfg
    arm = cfg.get("arm")
    try:
        v5 = V5Cfg.from_dict(cfg.get("v5"))
    except ValueError as e:
        raise SystemExit(f"{run}: config.json v5 {cfg.get('v5')!r}: {e}") from None
    if not (v5.on and v5.corrector):
        raise SystemExit(f"{run}: not a refiner v5 net with the corrector A (config.json v5 {cfg.get('v5')!r}): nothing "
                         f"to gate")
    if arm not in ("T", "M"):
        raise SystemExit(f"{run}: arm {arm!r}: only the DET (T) / MAP (M) teacher-trunk arms are wired (Stage 1b A_S on "
                         f"the student's features needs its own inputs)")
    if arm == "M" and split == "navtrain_val":
        raise SystemExit(f"{run}: arm M (MAP teacher, ReSMap BEV): navtrain_val has no ReSMap cache (constants.SPLITS "
                         f"map_teacher False; 0 of the 12,179 held-out tokens are in the navtrain ReSMap index) -- the "
                         f"MAP gate is navtest only: --split navtest")
    if v5.relation == "student":
        raise SystemExit(f"{run}: v5.relation 'student' reads the e2e student's det rows (Stage 1b): no such input here")
    if knockout == "objects" and not v5.rel:
        raise SystemExit(f"--knockout objects: {run} has no relation channel (v5.relation {v5.relation!r})")
    if str((cfg.get("v4") or {}).get("rg_priv", "off")) != "off":
        raise SystemExit(f"{run}: PRIVILEGED range teacher (v4.rg_priv): its GT-object input is not built here")
    if TemporalCfg.from_dict(cfg.get("temporal")).on:
        raise SystemExit(f"{run}: temporal branch on: not a v5 range net input this tool builds")
    return {"arm": arm, "v5": v5}


# ----------------------------------------------------------------------------------------------- stage infer
def freeze_ckpt(run: Path, which: str, ad: Path) -> Dict[str, Any]:
    """<ad>/ckpt_<which>.pt = a copy of <run>/ckpt_<which>.pt made once (the evaluated file; ckpt_last.pt of a running
    training is replaced atomically, so the copy is one consistent version) -> {copy, ckpt_sha16 (of the copy), source,
    source_sha16_now, copied, source_changed}."""
    src = run / f"ckpt_{which}.pt"
    if not src.is_file():
        raise SystemExit(f"{src} missing (--which {which}: ckpt_{which}.pt of the run)")
    cp = ad / f"ckpt_{which}.pt"
    copied = False
    if not cp.is_file():
        ad.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_name(f".{cp.name}.tmp{os.getpid()}")
        shutil.copyfile(src, tmp)
        os.replace(tmp, cp)
        copied = True
    sha, now = U.sha256_file(cp), U.sha256_file(src)
    return {"copy": str(cp), "ckpt_sha16": sha, "source": str(src), "source_sha16_now": now, "copied": copied,
            "source_changed": bool(now != sha)}


def infer_resume_check(ad: Path, meta: Dict) -> Dict:
    """outputs in ad are resumed only when written under the same identity (INFER_ID_KEYS) -> the old meta ({} fresh)."""
    old = U.read_json(ad / "infer_meta.json", None)
    dp = ad / "done.npy"
    has_done = bool(dp.is_file() and np.load(dp, mmap_mode="r").any())
    if old is None:
        if has_done:
            raise SystemExit(f"{ad}: done rows but no infer_meta.json: resume refused (--fresh)")
        return {}
    js = lambda x: json.loads(json.dumps(x, sort_keys=True, default=str))      # noqa: E731  (tuples vs JSON lists)
    diff = [f"{k} {old.get(k)!r} -> {meta.get(k)!r}" for k in INFER_ID_KEYS if js(old.get(k)) != js(meta.get(k))]
    if diff and has_done:
        raise SystemExit(f"{ad}: resume refused -- the outputs there were written under another identity "
                         f"({'; '.join(diff)}); --fresh or another --out")
    return old if not diff else {}


def refine_forward(net, bev, cand, status, rel_objs=None) -> Dict:
    """the gate's forward (= CKNet._forward_v4(..., refine_cols = 0) without decode): trunk.scene, scene_pref,
    score_set(cand [T, K, 8, 3], every candidate its own parent, return_ctx, rel_objs), refine_drafts(column 0) in fp32
    -> tau0 / tau_base / tau1 [T, 8, 3] f32, mu [T, 12] f32."""
    import torch
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    T = cand.shape[0]
    ego = ego_inputs(status)
    feat, mem = net.trunk.scene(bev, T)
    pref = net.scene_pref(mem, ego)
    kw = {"rel_objs": rel_objs} if rel_objs is not None else {}
    o = net.score_set(feat, mem, ego, cand.float(), pref=pref, return_ctx=True, **kw)
    cols = torch.zeros(T, 1, dtype=torch.long, device=cand.device)
    with torch.autocast(device_type=cand.device.type, enabled=False):
        rf = net.refine_drafts(o, cand.float(), cols, ego)
    return {k: rf[k][:, 0].float() for k in ("tau0", "tau_base", "tau1", "mu")}


def stage_infer(a, R: Dict, rows: np.ndarray) -> Dict:
    import torch
    from torch.utils.data import DataLoader

    from navsim.agents.para_ssr.ck.e2e_data2 import teacher_v5_rel_cfg
    from navsim.agents.para_ssr.ck.model import load_ck
    from navsim.agents.para_ssr.ck.online2 import is_range_net
    from navsim.agents.para_ssr.ck.temporal.config import OBJ_INPUT_KEYS
    from tools.ck.data.ck2_dataset import OBJ_BAD_MAX
    from tools.ck.data.ck_dataset import CKDataset, collate_ck
    t0 = time.time()
    run, ad = R["run"], R["dir"]
    cfg = run_config(run)
    chk = check_run_cfg(run, cfg, a.split, a.knockout)
    if a.device.startswith("cuda"):
        from tools.ck.v4 import eval_range_teacher as E
        E.cuda_guard(a)                                 # approval + CUDA_VISIBLE_DEVICES = one idle GPU of 0-3
        dev = torch.device("cuda:0")
    elif a.device == "cpu":
        dev = torch.device("cpu")
    else:
        raise SystemExit(f"--device {a.device!r}: cuda | cpu")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    ck = freeze_ckpt(run, a.which, ad)
    if ck["source_changed"]:
        log(f"[infer] {R['label']}: {ck['source']} changed since the copy (sha16 {ck['source_sha16_now']} vs the copy "
            f"{ck['ckpt_sha16']}): evaluating the COPY; --fresh re-copies")
    try:
        net, _ = load_ck(run, which=ck["copy"], device=str(dev), allow_privileged=False)
    except PermissionError as e:
        raise SystemExit(f"{run}: {e}") from None
    net.eval()
    if not is_range_net(net) or net.corrector_v5 is None:
        raise SystemExit(f"{run}: the loaded net has no range readout / corrector A")
    rel_on = bool(net.v5.rel)
    spec = None
    if rel_on:
        tc = teacher_v5_rel_cfg(run)
        if tc is None:
            raise SystemExit(f"{run}: relation {net.v5.relation!r} but e2e_data2.teacher_v5_rel_cfg found no pack")
        spec = {"cfg": tc.to_dict(), "aug": False, "targets": False}
    _, tdf, _ = packed(a.split)
    N = len(tdf)
    amp = "fp16 autocast (score_set), refine fp32" if dev.type == "cuda" else "fp32"
    meta = dict(label=R["label"], role=R["role"], run=str(run), run_real=str(run.resolve()), which=a.which,
                source_ckpt=ck["source"],
                ckpt=ck["copy"], ckpt_sha16=ck["ckpt_sha16"], source_sha16_at_start=ck["source_sha16_now"],
                arm=chk["arm"], split=a.split, drafts=a.drafts, score_k=int(a.score_k), knockout=a.knockout, amp=amp,
                v4=net.v4.to_dict(), v5=net.v5.to_dict(), corridor_version=int(net.corridor_version),
                objects=(spec or {}).get("cfg"), device=str(dev), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                batch=a.batch, n=N, tokens=a.tokens, limit=a.limit, started=kst(t0),
                columns=list(COLS), traj3="tau0 (the draft), tau_base (mu = 0: the lateral head), tau1")
    old = infer_resume_check(ad, meta)
    specs = {"traj3": ((N, 3, 8, 3), np.float32, np.nan), "mu": ((N, 12), np.float32, np.nan),
             "done": ((N,), np.bool_, False), "bev_ok": ((N,), np.bool_, False), "obj_ok": ((N,), np.bool_, False)}
    A = {k: U.open_memmap(ad / f"{k}.npy", s, d, fill=f) for k, (s, d, f) in specs.items()}
    todo = rows[~np.asarray(A["done"][rows], bool)]
    U.write_json(ad / "infer_meta.json", dict(meta, status="running", sessions=old.get("sessions") or []))
    log(f"[infer] {R['label']} ({run.name} ckpt_{a.which}, sha16 {ck['ckpt_sha16']}, relation {net.v5.relation}, "
        f"knockout {a.knockout}, {dev}) -> {ad}: todo {len(todo)} of {len(rows)}")
    n = n_bad = 0
    obj_desc = None
    if len(todo):
        ds = CKDataset(a.split, bev=chk["arm"], k=int(a.score_k), labels=None, gt=False, rows=todo, objects=spec)
        if ds.objects is not None:
            obj_desc = ds.objects.describe()
            if ds.objects.frac_obj_bad > OBJ_BAD_MAX:      # the training rule (train_ck2.check_obj_bad)
                raise SystemExit(f"{ds.objects.n_obj_bad} of {len(ds)} tokens have no object rows (obj_ok False) in "
                                 f"{ds.objects.pack.dir}: more than {100 * OBJ_BAD_MAX:.1f} %")
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, collate_fn=collate_ck, num_workers=a.workers_io,
                        pin_memory=dev.type == "cuda", prefetch_factor=4 if a.workers_io > 0 else None)
        t1 = time.time()
        with torch.no_grad():
            for b in dl:
                prow = b["rows"].numpy().astype(np.int64)
                bok = b["bev_ok"].numpy().astype(bool).reshape(-1)
                n_bad += int((~bok).sum())
                cand = b["cand"].to(dev).float()
                rel = None
                if rel_on:
                    rel = {k: b[k].to(dev) for k in OBJ_INPUT_KEYS}
                    if a.knockout == "objects":       # object off: no valid row -> every a_i = 0, H = 0
                        rel["obj_mask"] = torch.zeros_like(rel["obj_mask"])
                        rel["frame_valid"] = torch.zeros_like(rel["frame_valid"])
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    o = refine_forward(net, b["bev"].to(dev, non_blocking=True), cand, b["status"].to(dev).float(), rel)
                g = {k: v.cpu().numpy() for k, v in o.items()}
                if not np.array_equal(g["tau0"], b["cand"].numpy()[:, 0].astype(np.float32)):
                    raise SystemExit("refine_drafts tau0 != packed cand[:, 0] (the draft) bitwise: refusing")
                q = prow[bok]
                A["traj3"][q] = np.stack([g["tau0"], g["tau_base"], g["tau1"]], 1)[bok]
                A["mu"][q] = g["mu"][bok]
                A["bev_ok"][prow] = bok
                if "obj_ok" in b:
                    A["obj_ok"][prow] = b["obj_ok"].numpy().astype(bool).reshape(-1)
                A["done"][q] = True
                n += len(prow)
                if n % (a.batch * 50) < a.batch or n == len(todo):
                    el = time.time() - t1
                    log(f"[infer] {R['label']} {n}/{len(todo)} {el:.0f}s {n / max(el, 1e-6):.1f} tok/s")
        for m in A.values():
            m.flush()
    done = np.asarray(A["done"][rows], bool)
    meta.update(n_rows=int(len(rows)), n_done=int(done.sum()), n_bev_bad_this_call=n_bad, sec=round(time.time() - t0, 1),
                finished=kst(), status="finished", objects_desc=obj_desc or old.get("objects_desc"),
                sessions=(old.get("sessions") or []) + [dict(started=meta["started"], finished=kst(), n=n,
                                                             ckpt_sha16=ck["ckpt_sha16"])])
    U.write_json(ad / "infer_meta.json", meta)
    log(f"[infer] {R['label']}: {n} rows this call ({n_bad} without BEV), {int(done.sum())}/{len(rows)} done, "
        f"{meta['sec']} s")
    return meta


# ----------------------------------------------------------------------------------------------- stage label
def check_draft_labels(lab_new: np.ndarray, ok_new: np.ndarray, lab_ref: np.ndarray, ok_ref: np.ndarray,
                       tol: float = LABEL_TOL) -> Dict[str, Any]:
    """the re-scored draft column [n, 9] / ok [n] vs the stored draft labels: pass iff ok is equal on every row and the
    labels agree within tol where both are finite (NaN on the same entries)."""
    lab_new, lab_ref = np.asarray(lab_new, np.float64), np.asarray(lab_ref, np.float64)
    ok_new, ok_ref = np.asarray(ok_new, bool), np.asarray(ok_ref, bool)
    both = ok_new & ok_ref
    fn, fr = np.isfinite(lab_new[both]), np.isfinite(lab_ref[both])
    nan_mismatch = int((fn != fr).sum())
    d = np.abs(np.where(fn & fr, lab_new[both] - lab_ref[both], 0.0))
    mx = float(d.max()) if d.size else 0.0
    n_ok_mismatch = int((ok_new != ok_ref).sum())
    return {"n": int(len(ok_new)), "n_both_ok": int(both.sum()), "n_ok_mismatch": n_ok_mismatch,
            "n_nan_mismatch": nan_mismatch, "max_abs_diff": mx, "tol": float(tol),
            "pass": bool(n_ok_mismatch == 0 and nan_mismatch == 0 and mx <= tol)}


def label_rows(ad: Path, rows: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """the rows the label stage scores: done & all three trajectories finite (counts of the rest)."""
    done = np.asarray(np.load(ad / "done.npy", mmap_mode="r")[rows], bool)
    traj = np.asarray(np.load(ad / "traj3.npy", mmap_mode="r")[rows])
    fin = np.isfinite(traj).reshape(len(rows), -1).all(1)
    return rows[done & fin], {"n_not_done": int((~done).sum()), "n_nonfinite": int((done & ~fin).sum())}


def stage_label(a, R: Dict, rows: np.ndarray) -> Dict:
    from tools.ck.data import common as CM
    ad = R["dir"]
    tp = ad / "traj3.npy"
    if not (tp.is_file() and (ad / "done.npy").is_file()):
        raise SystemExit(f"{ad}: no traj3.npy / done.npy: run --stage infer first")
    lrows, cnt = label_rows(ad, rows)
    if cnt["n_nonfinite"]:
        log(f"[label] {R['label']}: {cnt['n_nonfinite']} done rows with a non-finite tau_base / tau1: not scored "
            f"(missing in the report)")
    if not len(lrows):
        raise SystemExit(f"[label] {R['label']}: no done row to score")
    key = {"traj_sha16": CM.sha16_array(np.load(tp, mmap_mode="r")), "rows_sha16": _sha16_rows(lrows),
           "n_rows": int(len(lrows))}
    ld, kp = ad / "labels", ad / "label_key.json"
    old = U.read_json(kp, None)
    if ld.exists() and old != key:              # label_cands' shard plan does not pin the row set: never resume across
        log(f"[label] {R['label']}: label inputs changed ({old} -> {key}): rescoring from scratch")
        shutil.rmtree(ld)
    have = U.read_json(ld / "meta.json", None) if (ld / "labels.npy").is_file() else None
    if have is not None and old == key and have.get("traj_sha16") == key["traj_sha16"]:
        log(f"[label] {R['label']}: labels present (traj3 sha16 {key['traj_sha16']}, {key['n_rows']} rows): skipped")
    else:
        if not 1 <= int(a.label_workers) <= MAX_LABEL_WORKERS:
            raise SystemExit(f"--label-workers must be in [1, {MAX_LABEL_WORKERS}]")
        U.write_json(kp, key)
        rp = ad / "label_rows.npy"
        np.save(rp, lrows)
        cmd = [sys.executable, f"{REPO}/tools/ck/data/label_cands.py", "--split", a.split, "--name",
               f"v5gate_{ad.name}", "--traj", str(tp), "--rows", str(rp), "--workers", str(int(a.label_workers)),
               "--out", str(ld)]
        env = {k: v for k, v in os.environ.items() if k != "CK_DATA_ROOT"}
        env.update(PYTHONPATH=REPO, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
        log(f"[label] {R['label']}: {len(lrows)} rows x 3, workers {a.label_workers} -> {ld} (log {ad / 'label.log'})")
        t0 = time.time()
        with open(ad / "label.log", "a") as lf:
            rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        have = U.read_json(ld / "meta.json", None)
        if rc != 0 or not (ld / "labels.npy").is_file() or have is None:
            raise SystemExit(f"[label] label_cands failed for {R['label']} (rc {rc}; see {ad / 'label.log'})")
        if have.get("traj_sha16") != key["traj_sha16"]:
            raise SystemExit(f"[label] {ld}/meta.json traj_sha16 {have.get('traj_sha16')} != traj3 {key['traj_sha16']}")
        log(f"[label] {R['label']}: done in {time.time() - t0:.0f} s ({have.get('n_scored_tokens')} tokens, "
            f"{have.get('n_err_tokens')} errors)")
    lab = np.load(ld / "labels.npy", mmap_mode="r")
    ok = np.load(ld / "ok.npy", mmap_mode="r")
    _, tdf, _ = packed(a.split)
    ref, ref_ok = stored_draft_labels(a.split, len(tdf))
    chk = check_draft_labels(np.asarray(lab[lrows, 0]), np.asarray(ok[lrows, 0]), ref[lrows], ref_ok[lrows])
    chk.update(stored=str(CK / "labels" / a.split / "cand"), checked=kst(), **cnt)
    U.write_json(ad / "label_check.json", chk)
    if not chk["pass"]:
        raise SystemExit(f"[label] {R['label']}: the re-scored draft column does not reproduce the stored draft labels "
                         f"({chk}): the scorer differs from the one behind CK/labels/{a.split}/cand")
    log(f"[label] {R['label']}: draft column = stored labels (max |diff| {chk['max_abs_diff']:.1e} on "
        f"{chk['n_both_ok']} rows)")
    return {"key": key, "check": chk, "meta": have}


# ----------------------------------------------------------------------------------------------- metrics (pure)
def arm_block(lab3: np.ndarray, logs: np.ndarray, mu: Optional[np.ndarray] = None, traj3: Optional[np.ndarray] = None,
              noedit: str = "draft", break_rule: str = "pdms0", n_boot: int = N_BOOT) -> Dict[str, Any]:
    """one arm on the token set: lab3 [n, 3, 9] (COLS order), logs [n], mu [n, 12], traj3 [n, 3, 8, 3] ->
    {rows: metrics + pass rates per column, d: paired deltas x 100, brk: tau1 / base vs each reference under each rule,
    keys: per-key counts of tau1 vs each reference, mu, moved, vs_noedit, break_gate}."""
    from navsim.agents.para_ssr.ck.constants import LBL
    from tools.ck.eval_ck import row_metrics
    if noedit not in NOEDIT or break_rule not in BREAK_RULES:
        raise ValueError(f"noedit {noedit!r} / break_rule {break_rule!r}")
    lab3 = np.asarray(lab3, np.float64)
    P = LBL["pdms"]
    L = {c: lab3[:, i] for i, c in enumerate(COLS)}
    rows = {}
    for c in COLS:
        m = row_metrics(L[c], None)
        for k in KEYS5:
            m[f"pass_{k}"] = float(np.mean(L[c][:, LBL[k]] == 1)) if len(L[c]) else float("nan")
        m["pdms_pos"] = float(np.mean(L[c][:, P] > 0)) if len(L[c]) else float("nan")
        rows[c] = m
    d = {"tau1-base": boot100(L["tau1"][:, P], L["base"][:, P], logs, n_boot),
         "tau1-draft": boot100(L["tau1"][:, P], L["draft"][:, P], logs, n_boot),
         "base-draft": boot100(L["base"][:, P], L["draft"][:, P], logs, n_boot)}
    brk = {f"{new}_vs_{ref}": {r: break_fix(L[ref], L[new], r) for r in BREAK_RULES}
           for new, ref in (("tau1", "draft"), ("tau1", "base"), ("base", "draft"))}
    keys = {f"tau1_vs_{ref}": key_counts(L[ref], L["tau1"]) for ref in ("draft", "base")}
    keys["base_vs_draft"] = key_counts(L["draft"], L["base"])
    out = {"n": int(len(lab3)), "rows": rows, "d": d, "brk": brk, "keys": keys, "noedit": noedit,
           "break_rule": break_rule, "vs_noedit": d[f"tau1-{noedit}"],
           "break_gate": brk[f"tau1_vs_{noedit}"][break_rule]}
    if mu is not None:
        out["mu"] = mu_stats(mu)
    if traj3 is not None:
        t = np.asarray(traj3)
        out["moved"] = {"tau1_vs_draft": float(moved(t[:, 2], t[:, 0]).mean()) if len(t) else float("nan"),
                        "tau1_vs_base": float(moved(t[:, 2], t[:, 1]).mean()) if len(t) else float("nan"),
                        "base_vs_draft": float(moved(t[:, 1], t[:, 0]).mean()) if len(t) else float("nan"),
                        "tol_m": MOVE_M}
    return out


def _clause(ok: Optional[bool], value, bar: str, why: str = "") -> Dict[str, Any]:
    return {"pass": ok, "evaluable": ok is not None, "value": value, "bar": bar, **({"why": why} if why else {})}


def gate_decide(blocks: Dict[str, Dict], roles: Dict[str, Optional[str]], pairs: Dict[str, Dict],
                deciding_set: bool, context: str = "") -> Dict[str, Any]:
    """the four registered clauses (module docstring 'Gate') from the arm blocks (label -> arm_block), roles (label ->
    T / S / GT / None) and the paired tau1 differences pairs['A-B'] (boot100 of tau1_A - tau1_B) -> {clauses, verdict
    PASS / FAIL / INCOMPLETE, deciding (deciding_set and every clause evaluable), context, gt_minus_t, student_floor}."""
    by = {r: lb for lb, r in roles.items() if r}
    T, S, G = by.get("T"), by.get("S"), by.get("GT")
    pair = lambda x, y: pairs.get(f"{x}-{y}")                                    # noqa: E731
    cl: Dict[str, Dict] = {}
    if T and S:
        ts = pair(T, S)
        cl["a_t_minus_s"] = _clause(bool(ts["mean"] >= GATE_D_MIN and ts["lo"] > 0), ts,
                                    f"mean >= +{GATE_D_MIN:.2f} and CI lower bound > 0")
    else:
        cl["a_t_minus_s"] = _clause(None, None, f"mean >= +{GATE_D_MIN:.2f} and CI lower bound > 0",
                                    "needs a T and an S arm")
    if T and S and G:
        ts, gs = pair(T, S), pair(G, S)
        cl["b_t_minus_s_vs_half_gt"] = _clause(bool(ts["mean"] >= GATE_GT_FRAC * gs["mean"]),
                                               {"t_minus_s": ts["mean"], "gt_minus_s": gs["mean"]},
                                               f"T - S >= {GATE_GT_FRAC} x (GT - S) (point estimates)")
    else:
        cl["b_t_minus_s_vs_half_gt"] = _clause(None, None, f"T - S >= {GATE_GT_FRAC} x (GT - S)",
                                               "needs T, S and GT arms")
    if T:
        bt = blocks[T]["break_gate"]
        cl["c_break_t"] = _clause(bool(np.isfinite(bt["rate"]) and bt["rate"] <= GATE_BREAK_MAX), bt,
                                  f"break ({bt['rule']}, tau1^T vs {blocks[T]['noedit']}) <= {100 * GATE_BREAK_MAX:.1f} %")
        vt = blocks[T]["vs_noedit"]
        cl["d_t_vs_noedit"] = _clause(bool(vt["mean"] >= GATE_D_MIN and vt["lo"] > 0), vt,
                                      f"tau1^T - {blocks[T]['noedit']} >= +{GATE_D_MIN:.2f} and CI lower bound > 0")
    else:
        cl["c_break_t"] = _clause(None, None, f"break <= {100 * GATE_BREAK_MAX:.1f} %", "needs a T arm")
        cl["d_t_vs_noedit"] = _clause(None, None, f"tau1^T - no edit >= +{GATE_D_MIN:.2f}", "needs a T arm")
    ev = [c["pass"] for c in cl.values() if c["evaluable"]]
    if len(ev) == len(cl):
        verdict = "PASS" if all(ev) else "FAIL"
    else:
        verdict = "FAIL" if any(x is False for x in ev) else "INCOMPLETE"
    deciding = bool(deciding_set and len(ev) == len(cl))
    res = {"rule": GATE_RULE, "clauses": cl, "verdict": verdict, "deciding": deciding,
           "context": ("decision" if deciding else "report only") + (f" ({context})" if context else "")}
    if T and G:
        res["gt_minus_t"] = pair(G, T)
    if S:
        fs, bs = blocks[S]["vs_noedit"], blocks[S]["break_gate"]
        res["student_floor"] = {
            "d": fs, "break": bs,
            "pass": bool(fs["mean"] >= FLOOR_D_MIN and fs["lo"] > FLOOR_LO_MIN and np.isfinite(bs["rate"])
                         and bs["rate"] <= GATE_BREAK_MAX),
            "bar": f"tau1^S - no edit >= {FLOOR_D_MIN:g} with CI lower bound > {FLOOR_LO_MIN:g}, break <= "
                   f"{100 * GATE_BREAK_MAX:.1f} %"}
    return res


def token_gate(sel: np.ndarray, keep: np.ndarray, filters: Dict[str, np.ndarray], registered: bool) -> Dict[str, Any]:
    """completeness of the token set: sel = the selected rows, keep [N] bool = the evaluated rows, filters = per input a
    [N] bool mask (True = available) -> {n, n_selected, n_missing, missing_by_input (overlapping counts), complete,
    registered (sel is the registered set)}."""
    sel = np.asarray(sel, np.int64)
    k = np.asarray(keep, bool)[sel]
    by = {name: int((~np.asarray(m, bool)[sel]).sum()) for name, m in filters.items()}
    miss = sel[~k]
    return {"n": int(k.sum()), "n_selected": int(len(sel)), "n_missing": int(len(miss)),
            "missing_by_input": {name: v for name, v in by.items() if v},
            "missing_example": [int(x) for x in miss[:5]], "complete": bool(len(miss) == 0),
            "registered": bool(registered)}


# ----------------------------------------------------------------------------------------------- stage report
def _arm_arrays(ad: Path, N: int) -> Dict[str, Any]:
    need = ["done.npy", "traj3.npy", "mu.npy", "infer_meta.json", "labels/labels.npy", "labels/ok.npy",
            "label_check.json"]
    miss = [x for x in need if not (ad / x).is_file()]
    if miss:
        raise SystemExit(f"{ad}: missing {miss}: run --stage infer,label first")
    lab = np.load(ad / "labels/labels.npy", mmap_mode="r")
    ok = np.load(ad / "labels/ok.npy", mmap_mode="r")
    if lab.shape != (N, 3, 9) or ok.shape != (N, 3):
        raise SystemExit(f"{ad}/labels: labels {lab.shape} / ok {ok.shape} != [{N}, 3, 9] / [{N}, 3]")
    chk = U.read_json(ad / "label_check.json")
    if not chk.get("pass"):
        raise SystemExit(f"{ad}/label_check.json: the draft column check did not pass ({chk})")
    return {"lab": lab, "ok": np.asarray(ok, bool), "done": np.load(ad / "done.npy").astype(bool),
            "traj3": np.load(ad / "traj3.npy", mmap_mode="r"), "mu": np.load(ad / "mu.npy", mmap_mode="r"),
            "meta": U.read_json(ad / "infer_meta.json"), "check": chk,
            "label_meta": U.read_json(ad / "labels" / "meta.json", {})}


def _f(x, nd: int = 2) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _ci(d: Optional[Dict]) -> str:
    if not d or not d.get("n"):
        return "-"
    return f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"


def _bf(b: Dict) -> str:
    return f"{100 * b['rate']:.2f} % ({b['n_break']}/{b['den']}); fix {b['n_fix']}/{b['fix_den']}"


def report_md(res: Dict) -> str:
    g, ts = res["gate"], res["token_set"]
    L = [f"# refiner v5 (3) gate: {' + '.join(res['arms'])} / {res['split']} {res['tokens']}", "",
         f"Created {res['created']}. Drafts {res['drafts']} (packed cand[:, 0]). Tokens {ts['n']} of {ts['n_selected']} "
         f"selected (missing {ts['n_missing']}: {ts['missing_by_input'] or 'none'}), logs {res['n_logs']}. Official "
         f"scorer (label_cands / score_token, LQR). Bootstrap {res['n_boot']} (log clusters, seed {BOOT_SEED}). Units "
         f"x 100. No edit = {res['noedit']}; break rule {res['break_rule']}.", ""]
    for lb, A in res["arm_info"].items():
        L.append(f"- arm {lb}: role {A['role']}, run {A['run']}, --which {A['which']}, ckpt copy sha16 {A['ckpt_sha16']}"
                 + (f" (the run's file CHANGED since: {A['source_sha16_now']})" if A["source_changed"] else
                    " (the run's file unchanged)") + f", relation {A['relation']}, knockout {A['knockout']}, "
                 f"{A['amp']}, score_k {A['score_k']}.")
    L += ["", f"**Gate: {g['verdict']}** -- {g['context']}.", "", f"Rule: {g['rule']}", "",
          "| clause | value | bar | pass |", "|---|---|---|---|"]
    for k, c in g["clauses"].items():
        v = c["value"]
        vs = (_ci(v) if isinstance(v, dict) and "mean" in v else
              (f"{100 * v['rate']:.2f} % ({v['n_break']}/{v['den']})" if isinstance(v, dict) and "rate" in v else
               (json.dumps(v) if v is not None else "-")))
        L.append(f"| {k} | {vs} | {c['bar']} | {c['pass'] if c['evaluable'] else 'not evaluable: ' + c.get('why', '')} |")
    if g.get("gt_minus_t"):
        L += ["", f"GT - T (what no teacher gives): {_ci(g['gt_minus_t'])}."]
    if g.get("student_floor"):
        sf = g["student_floor"]
        L += ["", f"Student floor (report only): tau1^S - no edit {_ci(sf['d'])}, break {_bf(sf['break'])} -> "
                  f"{'pass' if sf['pass'] else 'FAIL'} ({sf['bar']})."]
    L += ["", "## Per arm", "",
          "| arm | column | PDMS | NC | DAC | EP | TTC | C | DDC | fail NC | fail DAC | fail TTC | PDMS > 0 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lb, B in res["blocks"].items():
        for c in COLS:
            r = B["rows"][c]
            L.append(f"| {lb} | {c} | {_f(100 * r['pdms'])} | {_f(100 * r['nc'])} | {_f(100 * r['dac'])} | "
                     f"{_f(100 * r['ep'])} | {_f(100 * r['ttc'])} | {_f(100 * r['comfort'])} | {_f(100 * r['ddc'])} | "
                     f"{_f(100 * r['fail_nc'])} | {_f(100 * r['fail_dac'])} | {_f(100 * r['fail_ttc'])} | "
                     f"{_f(100 * r['pdms_pos'])} |")
    L += ["", "| arm | tau1 - base | tau1 - draft | base - draft | break tau1 vs draft (pdms0) | break tau1 vs draft "
              "(fails3) | break tau1 vs base (pdms0) | break tau1 vs base (fails3) | |mu| mean | |mu| > 0.05 | moved "
              "vs draft |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lb, B in res["blocks"].items():
        mu, mv = B.get("mu") or {}, B.get("moved") or {}
        L.append(f"| {lb} | {_ci(B['d']['tau1-base'])} | {_ci(B['d']['tau1-draft'])} | {_ci(B['d']['base-draft'])} | "
                 f"{_bf(B['brk']['tau1_vs_draft']['pdms0'])} | {_bf(B['brk']['tau1_vs_draft']['fails3'])} | "
                 f"{_bf(B['brk']['tau1_vs_base']['pdms0'])} | {_bf(B['brk']['tau1_vs_base']['fails3'])} | "
                 f"{_f(mu.get('linf_mean'), 3)} | {_f(100 * mu['frac_gt_thr'] if mu else None, 1)} % | "
                 f"{_f(100 * mv['tau1_vs_draft'] if mv else None, 1)} % |")
    L += ["", "Per-key counts, tau1 vs the no-edit reference (fail = value < 1; break = ref pass -> tau1 fail):", "",
          "| arm | key | fail ref | fail tau1 | break | fix |", "|---|---|---|---|---|---|"]
    for lb, B in res["blocks"].items():
        kc = B["keys"][f"tau1_vs_{res['noedit']}"]
        for k in KEYS5 + ("pdms0",):
            x = kc[k]
            L.append(f"| {lb} | {k} | {x['fail_ref']} | {x['fail_new']} | {x['break']} | {x['fix']} |")
        L.append(f"| {lb} | EP mean ref / tau1 / loss | {_f(100 * kc['ep']['mean_ref'])} | {_f(100 * kc['ep']['mean_new'])} "
                 f"| {_f(100 * kc['ep']['loss'])} | |")
    if res["pairs"]:
        L += ["", "Between arms (tau1_A - tau1_B, paired): " + "; ".join(f"{k} {_ci(v)}" for k, v in res["pairs"].items())
              + "."]
    if res["caveats"]:
        L += ["", "Caveats:"] + [f"- {c}" for c in res["caveats"]]
    return "\n".join(L) + "\n"


def stage_report(a, arms: List[Dict], rows: np.ndarray, root: Path, tok_info: Dict) -> Dict:
    from navsim.agents.para_ssr.ck.constants import LBL
    _, tdf, pok = packed(a.split)
    N = len(tdf)
    sel = np.zeros(N, bool)
    sel[rows] = True
    keep = sel.copy()
    filters: Dict[str, np.ndarray] = {}
    D: Dict[str, Dict] = {}
    for R in arms:
        X = _arm_arrays(R["dir"], N)
        lab = np.asarray(X["lab"])
        lab_ok = X["ok"].all(1) & np.isfinite(lab[..., :7]).reshape(N, -1).all(1)      # the 7 score columns
        filters[f"{R['label']}: infer done (BEV)"] = X["done"]
        filters[f"{R['label']}: labels ok (3 columns)"] = lab_ok | ~X["done"]
        keep &= X["done"] & lab_ok
        D[R["label"]] = X
    r = np.flatnonzero(keep)
    if not len(r):
        miss = {k: int((~m[rows]).sum()) for k, m in filters.items()}
        raise SystemExit(f"[report] no selected token is done and labelled by every arm (missing per input: {miss})")
    logs = tdf.log.astype(str).to_numpy()[r]
    registered = bool(a.split == REG_SPLIT and a.tokens == REG_TOKENS and not a.limit)
    ts = token_gate(rows, keep, filters, registered)
    blocks, roles = {}, {}
    for R in arms:
        X = D[R["label"]]
        blocks[R["label"]] = arm_block(np.asarray(X["lab"][r]), logs, np.asarray(X["mu"][r]),
                                       np.asarray(X["traj3"][r]), a.noedit, a.break_rule, a.bootstrap)
        roles[R["label"]] = R["role"]
    P = LBL["pdms"]
    pairs = {}
    labels = [R["label"] for R in arms]
    for i, x in enumerate(labels):
        for y in labels[i + 1:]:
            for p, q in ((x, y), (y, x)):
                pairs[f"{p}-{q}"] = boot100(np.asarray(D[p]["lab"][r, 2, P]), np.asarray(D[q]["lab"][r, 2, P]), logs,
                                            a.bootstrap)
    reasons = []
    if not registered:
        reasons.append(f"not the registered token set ({a.split} --tokens {a.tokens}{', --limit' if a.limit else ''}; "
                       f"the gate decides on {REG_SPLIT} {REG_TOKENS})")
    if registered and not ts["complete"]:
        reasons.append(f"token set incomplete ({ts['n']} of {ts['n_selected']}: {ts['missing_by_input']})")
    if a.drafts != "r34top1":
        reasons.append(f"drafts {a.drafts}")
    if a.knockout != "none":
        reasons.append(f"knock-out {a.knockout} (report only)")
    deciding_set = registered and ts["complete"] and a.drafts == "r34top1" and a.knockout == "none"
    ctx = "; ".join(reasons) if reasons else f"{REG_SPLIT} {REG_TOKENS}, tokens {ts['n']} / {ts['n_selected']}"
    gate = gate_decide(blocks, roles, pairs, deciding_set, ctx)
    arm_info = {}
    for R in arms:
        m = D[R["label"]]["meta"]
        now = U.sha256_file(m["source_ckpt"]) if Path(m["source_ckpt"]).is_file() else None
        arm_info[R["label"]] = dict(role=R["role"], run=m["run"], which=m["which"], ckpt=m["ckpt"],
                                    ckpt_sha16=m["ckpt_sha16"], source_ckpt=m["source_ckpt"], source_sha16_now=now,
                                    source_changed=bool(now != m["ckpt_sha16"]), relation=(m.get("v5") or {}).get(
                                        "relation"), knockout=m.get("knockout"), amp=m.get("amp"),
                                    score_k=m.get("score_k"), arm=m.get("arm"), label_check=D[R["label"]]["check"],
                                    label_meta={k: D[R["label"]]["label_meta"].get(k) for k in (
                                        "traj_sha16", "n_scored_tokens", "n_err_tokens", "wall_s", "scorer_yaml")})
    caveats = ["The DET BEV backbone (BEVFusion) and the object-pack detections are in-sample on navtrain: val decides, "
               "navtest reports." if a.split == "navtrain_val" else "navtest: report only (the gate decides on val)."]
    if any("fp16" in str(v["amp"]) for v in arm_info.values()):
        caveats.append("CUDA fp16 autocast inference: numbers differ slightly from a CPU fp32 run of the same ckpt.")
    if any(v["source_changed"] for v in arm_info.values()):
        caveats.append("a run's checkpoint file changed after the copy was taken: the numbers are about the COPY.")
    res = {"split": a.split, "tokens": a.tokens, "token_info": tok_info, "drafts": a.drafts, "limit": a.limit,
           "arms": labels, "roles": roles, "arm_info": arm_info, "noedit": a.noedit, "break_rule": a.break_rule,
           "n_boot": int(a.bootstrap), "token_set": ts, "n_tokens": int(len(r)), "n_logs": int(len(set(logs))),
           "blocks": blocks, "pairs": pairs, "gate": gate, "caveats": caveats, "created": kst(), "root": str(root)}
    tag = "+".join(labels) + (f"_{a.tag}" if a.tag else "")
    U.write_json(root / f"gate_{tag}.json", res)
    md = report_md(res)
    (root / f"gate_{tag}.md").write_text(md)
    pt = pd.DataFrame({"row": r, "token": tdf.token.astype(str).to_numpy()[r], "log": logs})
    for lb in labels:
        X = D[lb]
        for i, c in enumerate(COLS):
            pt[f"{lb}:pdms_{c}"] = np.asarray(X["lab"][r, i, P], np.float32)
        pt[f"{lb}:mu_linf"] = np.abs(np.asarray(X["mu"][r], np.float32)).max(1)
    pt.to_parquet(root / f"gate_{tag}_per_token.parquet", index=False)
    print(md, flush=True)
    log(f"[report] -> {root / f'gate_{tag}.json'} / .md / _per_token.parquet")
    return res


# ----------------------------------------------------------------------------------------------- main
def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="[ROLE=]DIR (ROLE T | S | GT); repeatable")
    ap.add_argument("--which", default="last", help="last | ep<e> (ckpt_<which>.pt, copied before use)")
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--tokens", default=None, help=f"{REG_TOKENS} (default on navtrain_val) | all (default on navtest) "
                                                   f"| <.npy rows / tokens, or a text file of tokens>")
    ap.add_argument("--limit", type=int, default=0, help="first N selected tokens (smoke) into <out>/<split>/smoke<N>_*")
    ap.add_argument("--drafts", choices=DRAFTS, default="r34top1")
    ap.add_argument("--device", default="cuda", help="cuda (approval + one idle GPU of 0-3) | cpu")
    ap.add_argument("--approved", action="store_true", help="--device cuda: the user approved this GPU run (or "
                                                            "V4_USER_APPROVED=1)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers-io", type=int, default=4, help="DataLoader workers (stage infer)")
    ap.add_argument("--score-k", type=int, choices=(1, 16), default=1,
                    help="candidates scored with the draft (1: the draft alone; 16: the packed top-16, column 0 "
                         "refined; equal up to ~2e-7: rows independent, no dedupe)")
    ap.add_argument("--label-workers", type=int, default=32, help=f"label_cands workers (<= {MAX_LABEL_WORKERS})")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--stage", default=",".join(STAGES), help=f"comma list of {STAGES}")
    ap.add_argument("--noedit", choices=NOEDIT, default="draft",
                    help="the gate's no-edit reference: draft (tau0) | base (tau_base, mu = 0); both are reported")
    ap.add_argument("--break-rule", choices=BREAK_RULES, default="pdms0",
                    help="the gate's break: pdms0 (PDMS > 0 -> 0) | fails3 (NC / TTC / DAC pass -> fail); both reported")
    ap.add_argument("--knockout", choices=KNOCKOUTS, default="none",
                    help="objects: the relation rows masked (report only; outputs <run>@<which>@objoff)")
    ap.add_argument("--bootstrap", type=int, default=N_BOOT)
    ap.add_argument("--tag", default="", help="report file suffix gate_<arms>_<tag>.*")
    ap.add_argument("--fresh", action="store_true", help="remove the given arms' outputs (ckpt copy, infer, labels) "
                                                         "in this token root first")
    return ap


def main(argv=None) -> Dict:
    a = get_parser().parse_args(argv)
    if os.environ.get("CK_DATA_ROOT"):
        raise SystemExit(f"CK_DATA_ROOT={os.environ['CK_DATA_ROOT']} is set: the gate reads CK/packed and CK/labels "
                         f"only (unset it)")
    stages = [s.strip() for s in a.stage.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"--stage {bad} not in {STAGES}")
    if a.tag and not all(ch.isalnum() or ch in "-_." for ch in a.tag):
        raise SystemExit(f"--tag {a.tag!r}: letters, digits, '-', '_', '.' only")
    if a.limit < 0 or a.batch < 1 or a.workers_io < 0 or a.bootstrap < 10:
        raise SystemExit("--limit >= 0, --batch >= 1, --workers-io >= 0, --bootstrap >= 10")
    if not 1 <= int(a.label_workers) <= MAX_LABEL_WORKERS:
        raise SystemExit(f"--label-workers must be in [1, {MAX_LABEL_WORKERS}]")
    check_which(a.which)
    arms = parse_runs(a.run)
    for R in arms:                                      # every config refusal before any file is written
        check_run_cfg(R["run"], run_config(R["run"]), a.split, a.knockout)
    a.tokens = default_tokens(a.split, a.tokens)
    _, tdf, pok = packed(a.split)
    rows, ttag, tok_info = select_rows(a.split, a.tokens, tdf, pok)
    if a.limit:
        rows = rows[: a.limit]
    root = gate_root(a.out, a.split, ttag, a.limit)
    for R in arms:
        R["dir"] = arm_dir(root, R["run"], a.which, a.knockout)
        if a.fresh and R["dir"].exists():
            log(f"--fresh: removing {R['dir']}")
            shutil.rmtree(R["dir"])
    root.mkdir(parents=True, exist_ok=True)
    log(f"{a.split} tokens {a.tokens} ({len(rows)} rows{f', --limit {a.limit}' if a.limit else ''}) -> {root}; arms "
        f"{[(R['label'], R['run'].name) for R in arms]}")
    out: Dict[str, Any] = {"root": str(root), "n_rows": int(len(rows))}
    if "infer" in stages:
        out["infer"] = {R["label"]: stage_infer(a, R, rows) for R in arms}
    if "label" in stages:
        out["label"] = {R["label"]: stage_label(a, R, rows) for R in arms}
    if "report" in stages:
        out["report"] = stage_report(a, arms, rows, root, tok_info)
    return out


if __name__ == "__main__":
    main()
