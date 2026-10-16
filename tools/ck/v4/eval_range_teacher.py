#!/usr/bin/env python
"""refiner v4_3 (c) -- range teachers as SELECTION evidence (docs/refiner_v4/V4_3_SPEC.md 'Phase 0b': "S passes -> (c):
retrain the DET / MAP range teachers with R2 targets and measure how much of the GT selection gain their learned fields
recover (fixed-pool critic-only, val decides, navtest reported)").  Nothing here trains; GPU only for stage infer.

Pool (fixed, the same for every critic): the v2 r34 top-16 + the 80 rule variants (a-1.0, a-0.5, a+0.5, l-0.5, l+0.5)
of tools/ck/e2e2/eval_teacher_navtest_r34 (select2 layout c = k * 6 + v; invalid lateral variants excluded = 'the 96').
Official labels: CK_DATA/labels/<split>/cand (16) + <variants>/labels (80); '_a' picks (the pool trajectory as is, no
lateral head), so nothing is re-scored.

Splits
  navtrain_val  DET only (no ReSMap cache for val: a MAP teacher is refused); tokens = the 'heldout' subset by default
                (the navtrain_val rows outside the Phase 0 F1 6,000, = the Phase 0b tokens; --subset all for every row).
                Caveat: the DET BEV backbone (BEVFusion) is in-sample on navtrain; val decides, navtest reports.
                Variants / baselines: CK/ck2/analysis/refiner_dm_traits/infer/navtrain_val (r34_variants, ck2T10/r34).
  navtest       DET + MAP (or one); variants / baselines: CK/v4/eval/navtest_r34 (r34_variants, ck2T10 / ck2M10 /
                v4T10_haz / v4M10_haz /r34 prob16 / prob80 / done of eval_teacher_navtest_r34).

Stages (--stage comma list; resumable per row through done.npy)
  infer   each given range teacher (--det, --map: run dir or name under CK/ck2/train | CK/v4/train; config.json v4
          readout 'range'; load_ck(which)) on the pool (--device cuda needs the user's approval, V4_USER_APPROVED=1 or
          --approved, and an idle GPU: CUDA_VISIBLE_DEVICES = one of 0-3 using < 1024 MiB by nvidia-smi, else
          refused): the ops of eval_teacher_navtest_r34.teacher_forward (trunk.scene,
          score_set of the 16 (own parents) and of the 80 (their parents = the 16), P_ref query, fp16 autocast on CUDA)
          -> <root>/<teacher>/: score_logit16 / 80 [N, K, 5], slack16 / 80 [N, K, 3] (the readout's thresholded-field
          slacks nc [s], ttc [s], dac [m], range_readout.hard_evidence with the teacher's own rg_read), lp16 / 80
          [N, K, 3] (log P_pass of the soft evidence), done / bev_ok [N], infer_meta.json.  The LEARNED tube flag of a
          key = slack < 0 (outside), NC / TTC from the DET teacher, DAC from the MAP teacher.
  gttube  (CPU, --workers <= 28) the Phase 0b R2 GT tube on the same pool: GT fields on the 16 parents' grids
          (corridor version 2, heading 'parent', range_targets.blocked_field_np at 0.1 s with the STOPPED-AWARE B
          (stopped_mask_np) + lon_max, pooled to 0.25 s: OR for B, max for LON; D = the official SDF at the grid points,
          unreadable -> +SDF_CLIP) read at the executed states of ck/v4/lqr.simulate (float64, from status v0 = [4],
          a0 = [6]) by range_targets.executed_slack(read='any4', lon_field=LON, field=D) -> <root>/gt_tube_r2/:
          hit16 / hit80 [N, K, 3] bool (nc, ttc, dac outside), ok [N] (GT objects + SDF present), done [N],
          meta.json.  This is oracle_common's 'grid stopped-aware NC + read any4' representation (= gt_tube_flags,
          tests/test_v4_eval_range_teacher.py: bitwise equal to oracle_common.grid_eval on the same states).
          + the READOUT-RULE perfect-field tube rule16 / rule80 [N, K, 3]: range_readout.field_hits (the R2 teacher's
          own thresholded rule: NC on the stopped-aware B, TTC on Bp = LON >= -1.3 m below the 2 m switch else U, DAC
          on D, read any4) on the same GT fields with the teacher's float32 LQR states / parent frames = what a teacher
          whose fields equal their targets would flag (the ceiling of the learned tube; it differs from the R2 tube by
          the 2 m TTC switch and float32: 0.59 % of the rows on NC | TTC, 1,500 F1 tokens).  gttube writes only its
          own <root>/gt_tube_r2 (--gt-tube is refused here).  stage eval reads --gt-tube DIR instead (another run in
          this format: meta.json with this split and representation_id 'R2', arrays in packed-row order [N_packed, ..],
          opened read-only; rule16 / rule80 optional).
  eval    on the tokens with labels, teacher outputs, baselines, the blend critic (and the GT tube when present):
          (i)  fidelity of the learned tube vs the official labels (criterion-1 metrics, R2 read): per key and joint
               (over the keys present), fail recall = P(outside | official fail), pass recall = P(inside | pass) on
               every valid row of the 96, log-clustered bootstrap CIs (ratio estimator, 2,000, seed 0); the GT R2 tube
               on the same rows, and the learned / GT agreement;
          (ii) criterion S with the LEARNED tube: switch = the v2 top-1 if it is inside the tube, else the best
               deployment-blend candidate inside the tube (none inside -> the blend pick); blend = select2 pool96_old_b1
               (beta 1, SEL_W, v2 im inherited by the variants) on the --blend-critic probabilities (default: navtrain_val
               ck2T10 = the Phase 0b rule; navtest the deployed pair ck2T10,ck2M10 (critic_prob: each key from its own
               teacher, EP mean), + the ck2T10-only rows labelled 'Phase 0b convention'); argmax ties in the pool order
               (parent, then its variants: heldout_analyze.amax); dPDMS vs the blend over the 96 without the tube (x 100,
               paired log bootstrap), break = P(the switch pick fails NC / TTC / DAC | the blend pick passes all
               three); the same with the GT tube on the same keys (and on all three keys) and with the readout-rule
               tube, and the fraction of the GT-tube gain the learned tube recovers = sum(dPDMS learned) / sum(dPDMS GT)
               (bootstrap of the ratio over the same log resamples), also against the readout-rule ceiling.  On the
               Phase 0b token set (navtrain_val heldout, no --limit) the ck2T10 GT-tube row on all keys must reproduce
               HO/results.json criterion S (mean dPDMS, break count): checked, printed, a mismatch exits non-zero
               after the outputs are written;
          (iii) fixed-pool critic-only selection of the teacher pair (each key from its own teacher: DET nc / ttc /
               comfort, MAP dac, EP mean; DET only: dac neutral = 1, EP the DET's; MAP only: nc / ttc / comfort
               neutral) vs each baseline pair (--baselines, the same key rule), all modes of
               eval_teacher_navtest_r34.select_modes (pool96_ck_noim = the critic-only headline), PDMS b - a paired
               log bootstrap.
          -> <root>/eval_<tag>.json / .md (+ eval_<tag>_picks.npz).  PDMS only (no EPDMS: not approved for val).
          The md states that no (c) decision rule is registered in V4_3_SPEC.md 'Phase 0b' (DECISION_RULE None): the
          numbers are descriptive until the user fixes the rule (before the val infer).
  gtcheck (CPU, no teacher) the GT-only rows of (i) / (ii) (R2 tube and readout-rule tube, all keys, each blend critic)
          and the Phase 0b reproduction check -> <root>/gtcheck.json / .md: run it after gttube, before training.

  cd /workspace/yongjae/SSR-v4 && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
      /venv/ssr/bin/python tools/ck/v4/eval_range_teacher.py --split navtest --det v4T10_rng2 --map v4M10_rng2 \
      --stage infer --device cuda
  CUDA_VISIBLE_DEVICES= ... eval_range_teacher.py --split navtest --stage gttube --workers 24
  CUDA_VISIBLE_DEVICES= ... eval_range_teacher.py --split navtest --stage gtcheck
  CUDA_VISIBLE_DEVICES= ... eval_range_teacher.py --split navtest --det v4T10_rng2 --map v4M10_rng2 --stage eval
  (stage infer: V4_USER_APPROVED=1 and CUDA_VISIBLE_DEVICES = one idle GPU of 0-3.)
  (navtrain_val: --split navtrain_val --det v4T10_rng2, no --map.)  --limit N: the first N tokens of the subset into
  <out-root>/smoke<N>/<split> (CPU smoke).  Output root default CK/v4/eval/range_teacher.
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
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402

CK = Path("/home/external-user/ssd/yongjae_refiner/ck")
DATA = CK.parent
OUT_ROOT = CK / "v4" / "eval" / "range_teacher"
NAVTEST_ROOT = CK / "v4" / "eval" / "navtest_r34"
VAL_INFER = CK / "ck2" / "analysis" / "refiner_dm_traits" / "infer" / "navtrain_val"
F1_ROWS = CK / "v4" / "analysis" / "range_oracle" / "prep" / "f1_rows.npy"
K16, NV, NVAR = 16, 6, 5
K80 = K16 * NVAR
KEYS = ("nc", "ttc", "dac")
KI = {k: i for i, k in enumerate(KEYS)}
TBIN = 0.25                       # Phase 0b R2: 0.25 s x 20 bins
CORRIDOR_VERSION = 2
N_BOOT, BOOT_SEED = 2000, 0
BREAK_MAX, S_MIN = 0.013, 1.0     # criterion S bars (Phase 0b; reference rows only here)
MAX_WORKERS = 28
HO_DIR = CK / "v4" / "analysis" / "range_oracle" / "heldout_r2"    # Phase 0b held-out run (results.json, prep/)
PHASE0B_BLEND = "ck2T10"          # Phase 0b's deployment-blend critic (criterion S)
REP_ID = "R2"                     # gt tube meta.json representation_id
GT_TUBE_FORMAT = 2                # 2: + rule16 / rule80 (the readout-rule perfect-field tube)
GPU_MEM_MAX = 1024                # [MiB] stage infer refuses a GPU using this much or more (as run_teachers.sh)
# (c) decision rule of Phase 0b ('measure how much of the GT selection gain the learned fields recover, val decides'):
# NOT registered in docs/refiner_v4/V4_3_SPEC.md -- the user fixes it before the val infer; until then None and the
# eval md says the numbers are descriptive.
DECISION_RULE: Optional[str] = None
SPLITS = {
    "navtest": dict(variants=NAVTEST_ROOT / "r34_variants", base_root=NAVTEST_ROOT, arms=("T", "M"),
                    baselines=("ck2T10,ck2M10", "v4T10_haz,v4M10_haz"), subset="all", blend="ck2T10,ck2M10",
                    gt_obj=CK / "v3" / "gt_objects" / "navtest", sdf=DATA / "sdf" / "navtest"),
    "navtrain_val": dict(variants=VAL_INFER / "r34_variants", base_root=VAL_INFER, arms=("T",),
                         baselines=("ck2T10",), subset="heldout", blend=PHASE0B_BLEND,
                         gt_obj=CK / "v3" / "gt_objects" / "navtrain_val", sdf=DATA / "sdf" / "navtrain_val"),
}
STAGES = ("infer", "gttube", "gtcheck", "eval")


def kst(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime((time.time() if t is None else t) + 9 * 3600))


def log(msg: str) -> None:
    print(f"[{kst()}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- inputs
def packed(split: str):
    d = CK / "packed" / split
    return d, pd.read_parquet(d / "tokens.parquet"), np.load(d / "ok.npy")


def split_root(a) -> Path:
    return Path(a.out_root) / (f"smoke{a.limit}" if a.limit else "") / a.split


def subset_mask(split: str, subset: str, n: int) -> np.ndarray:
    """[n] bool rows of the token subset: 'all' | 'heldout' (navtrain_val rows outside the Phase 0 F1 sample) | 'f1'."""
    if subset == "all":
        return np.ones(n, bool)
    if split != "navtrain_val":
        raise SystemExit(f"--subset {subset}: only navtrain_val has the F1 sample")
    if not F1_ROWS.is_file():
        raise SystemExit(f"--subset {subset}: {F1_ROWS} missing (tools/ck/v4/oracle/prep.py)")
    f1 = np.zeros(n, bool)
    f1[np.load(F1_ROWS)] = True
    return ~f1 if subset == "heldout" else f1


def base_rows(a) -> np.ndarray:
    """packed rows of this run: packed ok & variants built & subset, first --limit."""
    _, tdf, pok = packed(a.split)
    n = len(tdf)
    vd = SPLITS[a.split]["variants"]
    built = np.load(vd / "built.npy") if (vd / "built.npy").is_file() else np.ones(n, bool)
    r = np.flatnonzero(pok & built & subset_mask(a.split, a.subset, n))
    return r[: a.limit] if a.limit else r


def resolve_run(name: str) -> Path:
    p = Path(name)
    if p.is_dir():
        return p
    for q in (CK / "ck2" / "train" / name, CK / "v4" / "train" / name):
        if q.is_dir():
            return q
    raise SystemExit(f"teacher run not found: {name}")


def tname(name: Optional[str]) -> Optional[str]:
    return None if not name else Path(name).name


def to_pool(x16: np.ndarray, x80: np.ndarray) -> np.ndarray:
    """[n, 16, ...] + [n, 80, ...] (column j = 5 k + v - 1) -> [n, 96, ...] select2 layout c = 6 k + v."""
    from tools.ck.e2e2.eval_teacher_navtest_r34 import to_pool as tp
    return tp(x16, x80)


def load_labels(split: str, vd: Path, n: int):
    from tools.ck.e2e2.eval_teacher_navtest_r34 import load_lab
    labc, okc = load_lab(CK / "labels" / split / "cand", n, K16)
    labv, okv = load_lab(vd / "labels", n, K80)
    if labc is None or labv is None:
        raise SystemExit(f"official labels missing: {CK / 'labels' / split / 'cand'} / {vd / 'labels'}")
    return labc, okc, labv, okv


# ----------------------------------------------------------------------------------------------- stage infer
def range_forward(net, bev, cand, status, var):
    """eval_teacher_navtest_r34.teacher_forward for a range teacher + its haz slacks / log P_pass:
    -> s16 [T, 16, 5], s80 [T, 80, 5], slack16 / slack80 [T, K, 3] (nc s, ttc s, dac m), lp16 / lp80 [T, K, 3]."""
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    T = cand.shape[0]
    ego = ego_inputs(status)
    feat, mem = net.trunk.scene(bev, T)
    pref = net.scene_pref(mem, ego)
    o16 = net.score_set(feat, mem, ego, cand.float(), pref=pref)
    o80 = net.score_set(feat, mem, ego, var.float(), cand.float().repeat_interleave(NVAR, 1), pref=pref)
    import torch
    st = lambda o, ks: torch.stack([o["haz"][k].float() for k in ks], -1)
    return {"s16": o16["score_logit"], "s80": o80["score_logit"],
            "slack16": st(o16, ("slack_nc", "slack_ttc", "slack_dac")),
            "slack80": st(o80, ("slack_nc", "slack_ttc", "slack_dac")),
            "lp16": st(o16, ("lp_nc", "lp_ttc", "lp_dac")), "lp80": st(o80, ("lp_nc", "lp_ttc", "lp_dac"))}


def cuda_guard(a) -> None:
    """stage infer on CUDA: the user's approval (V4_USER_APPROVED=1 or --approved), CUDA_VISIBLE_DEVICES = exactly one
    allowed GPU (ckutil.gpu_guard) and that GPU idle by nvidia-smi (memory.used < GPU_MEM_MAX MiB; a failed query
    refuses; V4_NVIDIA_SMI overrides the binary) -- checked before anything touches CUDA."""
    import subprocess
    if os.environ.get("V4_USER_APPROVED") != "1" and not getattr(a, "approved", False):
        raise SystemExit("stage infer on CUDA needs the user's approval (V4_USER_APPROVED=1 or --approved); GPUs 0-3 "
                         "only, idle ones")
    U.gpu_guard(a.device)
    g = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    smi = os.environ.get("V4_NVIDIA_SMI", "nvidia-smi")
    try:
        r = subprocess.run([smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", g],
                           capture_output=True, text=True, timeout=60)
        used = float(r.stdout.strip().splitlines()[0]) if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        used = None
    if used is None:
        raise SystemExit(f"stage infer: nvidia-smi query of GPU {g} failed: refusing (cannot tell that it is idle)")
    if used >= GPU_MEM_MAX:
        raise SystemExit(f"stage infer: GPU {g} uses {used:.0f} MiB (needs < {GPU_MEM_MAX}): busy, refusing")
    log(f"[infer] GPU {g}: {used:.0f} MiB used (< {GPU_MEM_MAX}), approved")


def stage_infer(a, role: str, name: str) -> Dict:
    import torch
    from torch.utils.data import DataLoader

    from navsim.agents.para_ssr.ck.model import load_ck
    from navsim.agents.para_ssr.ck.online2 import is_range_net
    from tools.ck.data.ck_dataset import CKDataset, collate_ck
    t0 = time.time()
    run = resolve_run(name)
    want = {"det": "T", "map": "M"}[role]
    if want not in SPLITS[a.split]["arms"]:
        raise SystemExit(f"{a.split}: no {role} teacher (no ReSMap cache for navtrain_val: DET only)")
    if a.device.startswith("cuda"):
        cuda_guard(a)
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    net, cfg = load_ck(run, which=a.which, device=str(dev))
    net.eval()
    if cfg["arm"] != want:
        raise SystemExit(f"--{role} {run}: arm {cfg['arm']!r}, expected {want!r}")
    if not is_range_net(net):
        raise SystemExit(f"--{role} {run}: not a range teacher (config.json v4.readout {net.v4.readout!r})")
    pdir, tdf, _ = packed(a.split)
    N = len(tdf)
    rows = base_rows(a)
    vd = SPLITS[a.split]["variants"]
    var_all = np.load(vd / "traj80.npy", mmap_mode="r")
    td = split_root(a) / tname(name)
    specs = {"done": ((N,), np.bool_, False), "bev_ok": ((N,), np.bool_, False),
             "score_logit16": ((N, K16, 5), np.float32, np.nan), "score_logit80": ((N, K80, 5), np.float32, np.nan),
             "slack16": ((N, K16, 3), np.float32, np.nan), "slack80": ((N, K80, 3), np.float32, np.nan),
             "lp16": ((N, K16, 3), np.float32, np.nan), "lp80": ((N, K80, 3), np.float32, np.nan)}
    A = {k: U.open_memmap(td / f"{k}.npy", s, d, fill=f) for k, (s, d, f) in specs.items()}
    rr = rows[~np.asarray(A["done"][rows], bool)]
    ck_file = run / f"ckpt_{a.which}.pt"
    meta = dict(role=role, teacher=tname(name), run=str(run), which=a.which, arm=cfg["arm"], ckpt=str(ck_file),
                ckpt_sha16=U.sha256_file(ck_file) if ck_file.is_file() else None, v4=net.v4.to_dict(),
                rg_read=net.v4.rg_read, rg_stopped_aware=bool(net.v4.rg_stopped_aware),
                corridor_version=int(net.corridor_version), split=a.split, subset=a.subset, n=N, limit=a.limit,
                device=str(dev), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), batch=a.batch,
                amp="fp16 autocast on CUDA (as eval_teacher_navtest_r34)", variants=str(vd / "traj80.npy"),
                started=kst(t0))
    log(f"[infer] {role} {tname(name)} ({ck_file.name}, rg_read {net.v4.rg_read}) -> {td}: todo {len(rr)} of "
        f"{len(rows)}")
    n = n_bad = 0
    if len(rr):
        ds = CKDataset(a.split, bev=want, k=K16, labels=None, gt=False, rows=rr)
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, collate_fn=collate_ck, num_workers=a.workers_io,
                        pin_memory=dev.type == "cuda", prefetch_factor=4 if a.workers_io > 0 else None)
        use_amp = dev.type == "cuda"
        t1 = time.time()
        with torch.no_grad():
            for b in dl:
                prow = b["rows"].numpy().astype(np.int64)
                bok = b["bev_ok"].numpy().astype(bool).reshape(-1)
                n_bad += int((~bok).sum())
                cand = b["cand"].to(dev).float()
                var = torch.from_numpy(np.asarray(var_all[prow], np.float32)).to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                    o = range_forward(net, b["bev"].to(dev, non_blocking=True), cand, b["status"].to(dev).float(), var)
                q = prow[bok]
                g = lambda t: t.float().cpu().numpy()[bok]
                A["score_logit16"][q], A["score_logit80"][q] = g(o["s16"]), g(o["s80"])
                for k in ("slack16", "slack80", "lp16", "lp80"):
                    A[k][q] = g(o[k])
                A["bev_ok"][prow] = bok
                A["done"][q] = True
                n += len(prow)
                if n % (a.batch * 100) < a.batch or n == len(rr):
                    el = time.time() - t1
                    log(f"[infer] {n}/{len(rr)} {el:.0f}s {n / max(el, 1e-6):.1f} tok/s")
        for m in A.values():
            m.flush()
    old = U.read_json(td / "infer_meta.json", {}) or {}
    meta.update(n_rows=int(len(rows)), n_done=int(np.asarray(A["done"][rows]).sum()), n_bev_bad_this_call=n_bad,
                sec=round(time.time() - t0, 1), finished=kst(),
                sessions=(old.get("sessions") or []) + [dict(started=meta["started"], finished=kst(), n=n)])
    U.write_json(td / "infer_meta.json", meta)
    log(f"[infer] done {n} rows ({n_bad} without BEV) in {meta['sec']} s")
    return meta


# ----------------------------------------------------------------------------------------------- stage gttube
def load_gt_store(gt_dir: Path, token: str) -> Optional[dict]:
    """GT objects of a token on the 0.1 s grid 0 .. 4.9 s (= oracle_common.load_gt on another store): boxes [A, 50, 5]
    float64, obs [A, 50], is_agent [A], vfirst [A]; None without a file."""
    from navsim.agents.para_ssr.ck.v4 import range_targets as RT
    from navsim.agents.para_ssr.refiner import gt_future as GF
    p = gt_dir / f"{token}.npz"
    if not p.is_file():
        return None
    obj = GF.load_objects(p)
    boxes, st = GF.query(obj, RT.fine_times(), dtype=np.float64)
    meta, first = np.asarray(obj["meta"]), np.asarray(obj["first"], np.float64)
    return dict(boxes=boxes, obs=st == GF.OBS, is_agent=meta[:, GF.MT_AGENT] > 0,
                vfirst=np.hypot(first[:, GF.FI_VX], first[:, GF.FI_VY]))


def r2_fields(par: np.ndarray, gt: Optional[dict], sdf: np.ndarray) -> dict:
    """Phase 0b R2 GT fields of P parents [P, 8, 3] (float64): fr (torch frame), B [P, 48, 17, 20] (stopped-aware,
    0.1 s -> 0.25 s OR), LON [P, 48, 17, 20] (0.1 s -> 0.25 s max), D [P, 48, 17] (SDF at the grid points,
    unreadable -> +SDF_CLIP)."""
    import torch

    from navsim.agents.para_ssr.ck.v4 import range_targets as RT
    from navsim.agents.para_ssr.refiner.corridor import N_LAT, N_ST, corridor_geometry
    from navsim.agents.para_ssr.refiner.sdf import SDF_CLIP, sample_sdf_np
    g = corridor_geometry(torch.as_tensor(par, dtype=torch.float64), version=CORRIDOR_VERSION)
    fr = RT.parent_frame(g, "parent")
    frn = RT.frame_np(fr)
    P = par.shape[0]
    if gt is None or gt["boxes"].shape[0] == 0:
        B01 = np.zeros((P, N_ST, N_LAT, RT.N_FINE), bool)
        L01 = np.full((P, N_ST, N_LAT, RT.N_FINE), -np.inf)
    else:
        stp = RT.stopped_mask_np(gt["is_agent"], gt["vfirst"])
        B01, L01 = RT.blocked_field_np(frn, gt["boxes"], gt["obs"], tbin_s=0.1, lon_max=True, stopped_aware=True,
                                       stopped=stp)
    mt = RT.time_bins(RT.fine_times(), TBIN)
    M = RT.n_bins_of(TBIN)
    B = np.stack([B01[..., mt == m].any(-1) for m in range(M)], -1)
    LON = np.stack([L01[..., mt == m].max(-1) for m in range(M)], -1)
    Dv, Dok = sample_sdf_np(sdf, frn["pts"])
    return dict(fr=fr, B=B, LON=LON, D=np.where(Dok, Dv, SDF_CLIP))


def gt_tube_flags(X: np.ndarray, F: dict, idx: np.ndarray) -> np.ndarray:
    """R2 GT tube of executed states X [C, 41, 6] (float64) of candidates whose parent is row idx [C] of the fields F
    (r2_fields) -> outside flags [C, 3] bool (nc, ttc, dac) = range_targets.executed_slack(read 'any4', TTC on LON
    from the current ego, DAC on D)."""
    import torch

    from navsim.agents.para_ssr.ck.v4 import range_targets as RT
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x))
    o = RT.executed_slack(t(X).double(), F["fr"], t(F["B"]), t(F["D"]).double(), index=torch.as_tensor(idx).long(),
                          tbin_s=TBIN, read="any4", lon_field=t(F["LON"]).double())
    return np.stack([o["hit_nc"].numpy(), o["hit_ttc"].numpy(), o["hit_dac"].numpy()], -1)


def executed_states(traj: np.ndarray, status: np.ndarray) -> np.ndarray:
    """ck/v4/lqr.simulate float64 of float32 trajectories [C, 8, 3] from the ego (v0, a0) = (status[4], status[6]) ->
    [C, 41, 6] (Phase 0b; = oracle_common.simulate with steer0 = 0)."""
    import torch

    from navsim.agents.para_ssr.ck.v4 import lqr as LQ
    tr = torch.as_tensor(np.asarray(traj, np.float32).astype(np.float64))
    s = np.asarray(status, np.float64)
    ego = torch.tensor([[np.nan_to_num(s[4]), np.nan_to_num(s[6])]], dtype=torch.float64)
    return LQ.simulate(tr, ego, dtype=torch.float64).numpy()


def rule_tube_flags(trajs: np.ndarray, status: np.ndarray, par32: np.ndarray, F: dict, idx: np.ndarray,
                    read: str = "any4") -> np.ndarray:
    """READOUT-RULE perfect-field tube (module docstring 'gttube'): range_readout.field_hits on the GT R2 fields of F
    (r2_fields) as an R2 teacher's three groups [B (stopped-aware), U = LON > -inf, Bp = LON >= BEHIND_M] + D, read at
    the teacher's own states (range_readout.candidate_states 'lqr', float32, ego (status[4], status[6])) on the parents'
    float32 frames (corridor version 2, as CKNet.score_set) -> outside flags [C, 3] bool (nc, ttc, dac)."""
    import torch

    from navsim.agents.para_ssr.ck.v4 import range_readout as RR
    from navsim.agents.para_ssr.ck.v4 import range_targets as RT
    from navsim.agents.para_ssr.refiner.corridor import corridor_geometry
    C = trajs.shape[0]
    s = torch.as_tensor(np.asarray(status, np.float32)).reshape(1, -1).expand(C, -1)
    st, tq = RR.candidate_states(torch.as_tensor(np.asarray(trajs, np.float32)), s[:, 4], s[:, 6], "lqr")
    fr = RT.parent_frame(corridor_geometry(torch.as_tensor(np.array(par32, np.float32)), version=CORRIDOR_VERSION))
    LON = torch.as_tensor(np.ascontiguousarray(F["LON"]))
    blk = torch.cat([torch.as_tensor(np.ascontiguousarray(F["B"])), LON > -np.inf, LON >= RT.BEHIND_M], -1)
    h = RR.field_hits(blk, torch.as_tensor(np.ascontiguousarray(F["D"])).float(), fr, torch.as_tensor(idx).long(),
                      st, tq, RT.n_bins_of(TBIN), TBIN, read=read)
    return np.stack([h["hit_nc"].numpy(), h["hit_ttc"].numpy(), h["hit_dac"].numpy()], -1)


def token_tubes(cand16: np.ndarray, traj80: np.ndarray, status: np.ndarray, gt: Optional[dict],
                sdf: np.ndarray) -> Dict[str, np.ndarray]:
    """-> {hit16 [16, 3], hit80 [80, 3] (R2 GT tube), rule16, rule80 (readout-rule tube)} of one token (variant j's
    parent = j // 5)."""
    F = r2_fields(np.asarray(cand16, np.float64), gt, sdf)
    trajs = np.concatenate([np.asarray(cand16, np.float32), np.asarray(traj80, np.float32)], 0)
    X = executed_states(trajs, status)
    idx = np.concatenate([np.arange(K16), np.arange(K80) // NVAR])
    h = gt_tube_flags(X, F, idx)
    q = rule_tube_flags(trajs, status, np.asarray(cand16, np.float32), F, idx)
    return {"hit16": h[:K16], "hit80": h[K16:], "rule16": q[:K16], "rule80": q[K16:]}


def token_tube(cand16: np.ndarray, traj80: np.ndarray, status: np.ndarray, gt: Optional[dict],
               sdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """-> (hit16 [16, 3], hit80 [80, 3]) of one token (R2 GT tube; variant j's parent = j // 5)."""
    F = r2_fields(np.asarray(cand16, np.float64), gt, sdf)
    trajs = np.concatenate([np.asarray(cand16, np.float32), np.asarray(traj80, np.float32)], 0)
    X = executed_states(trajs, status)
    idx = np.concatenate([np.arange(K16), np.arange(K80) // NVAR])
    h = gt_tube_flags(X, F, idx)
    return h[:K16], h[K16:]


_W: Dict = {}


def _w_init(split: str):
    import torch
    torch.set_num_threads(1)
    pdir, tdf, _ = packed(split)
    _W.update(split=split, tokens=tdf.token.astype(str).to_numpy(), cand=np.load(pdir / "cand.npy", mmap_mode="r"),
              status=np.load(pdir / "status.npy", mmap_mode="r"),
              traj80=np.load(SPLITS[split]["variants"] / "traj80.npy", mmap_mode="r"))


TUBE_ARRAYS = {"hit16": K16, "hit80": K80, "rule16": K16, "rule80": K80}


def _w_rows(rows: np.ndarray):
    from navsim.agents.para_ssr.refiner.sdf import load_sdf
    sp = SPLITS[_W["split"]]
    H = {k: np.zeros((len(rows), n, 3), bool) for k, n in TUBE_ARRAYS.items()}
    ok = np.zeros(len(rows), bool)
    for q, r in enumerate(rows):
        tok = str(_W["tokens"][r])
        ps = sp["sdf"] / f"{tok}.npz"
        gt = load_gt_store(sp["gt_obj"], tok)
        if gt is None or not ps.is_file():
            continue
        sdf = load_sdf(ps).astype(np.float64)
        o = token_tubes(np.asarray(_W["cand"][r, :K16]), np.asarray(_W["traj80"][r]), np.asarray(_W["status"][r]),
                        gt, sdf)
        for k in TUBE_ARRAYS:
            H[k][q] = o[k]
        ok[q] = True
    return rows, H, ok


def stage_gttube(a) -> Dict:
    import multiprocessing as mp
    t0 = time.time()
    if a.gt_tube:
        raise SystemExit("gttube writes only its own <root>/gt_tube_r2 (--gt-tube is for stage eval / gtcheck: another "
                         "run's tube, read-only)")
    _, tdf, _ = packed(a.split)
    N = len(tdf)
    rows = base_rows(a)
    gd = split_root(a) / "gt_tube_r2"
    if (gd / "done.npy").is_file() and not (gd / "rule16.npy").is_file() and bool(np.load(gd / "done.npy").any()):
        raise SystemExit(f"{gd}: an older GT-tube format without the readout-rule tube (rule16 / rule80): remove it "
                         f"(or use another --out-root) and rerun gttube")
    specs = {"done": ((N,), np.bool_, False), "ok": ((N,), np.bool_, False),
             **{k: ((N, n, 3), np.bool_, False) for k, n in TUBE_ARRAYS.items()}}
    A = {k: U.open_memmap(gd / f"{k}.npy", s, d, fill=f) for k, (s, d, f) in specs.items()}
    rr = rows[~np.asarray(A["done"][rows], bool)]
    log(f"[gttube] {a.split} -> {gd}: todo {len(rr)} of {len(rows)} ({a.workers} workers)")
    chunks = [rr[i:i + 16] for i in range(0, len(rr), 16)]
    n = 0
    workers = max(1, min(int(a.workers), MAX_WORKERS))
    ctx = mp.get_context("fork")
    pool = ctx.Pool(workers, initializer=_w_init, initargs=(a.split,)) if workers > 1 and len(chunks) > 1 else None
    it = pool.imap_unordered(_w_rows, chunks) if pool is not None else (_w_init(a.split) or map(_w_rows, chunks))
    try:
        for r, H, ok in it:
            for k in TUBE_ARRAYS:
                A[k][r] = H[k]
            A["ok"][r] = ok
            A["done"][r] = True
            n += len(r)
            if (n // 16) % 50 == 0 or n == len(rr):
                el = time.time() - t0
                log(f"[gttube] {n}/{len(rr)} {el:.0f}s")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    for m in A.values():
        m.flush()
    meta = dict(split=a.split, subset=a.subset, limit=a.limit, n_rows=int(len(rows)),
                n_done=int(np.asarray(A["done"][rows]).sum()), n_ok=int(np.asarray(A["ok"][rows]).sum()),
                representation_id=REP_ID, format=GT_TUBE_FORMAT, n_packed=int(N),
                row_order=f"packed rows of CK/packed/{a.split} (arrays [N_packed, ...])",
                representation="Phase 0b R2: corridor v2 parent grid, heading parent, 0.25 s x 20 bins (pooled from "
                               "0.1 s), box target, stopped-aware B (range_targets.stopped_mask_np), read any4, TTC on "
                               "LON (not-behind from the current ego), DAC on D (SDF at the grid points, bilinear)",
                rule="rule16 / rule80: range_readout.field_hits on the same GT fields as an R2 teacher's [B_sa, U, Bp] "
                     "+ D (NC on B_sa; TTC on Bp below the 2 m switch, U above; DAC on D; any4), float32 LQR states "
                     "(candidate_states) and float32 parent frames = the readout-rule perfect-field tube",
                states="ck/v4/lqr.simulate float64 from status (v0 = [4], a0 = [6], steer0 = 0)",
                gt_objects=str(SPLITS[a.split]["gt_obj"]), sdf=str(SPLITS[a.split]["sdf"]),
                columns="hit[..., (nc, ttc, dac)] = outside the GT tube", sec=round(time.time() - t0, 1),
                finished=kst())
    U.write_json(gd / "meta.json", meta)
    log(f"[gttube] done {n} rows in {meta['sec']} s ({meta['n_ok']} with GT)")
    return meta


# ----------------------------------------------------------------------------------------------- metrics
class LogBoot:
    """log-clustered bootstrap weights (tools/ck/eval_ck.log_bootstrap: multinomial log counts, seed 0) for ratio
    estimators over row masks of tokens (logs_tok [n])."""

    def __init__(self, logs_tok: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
        ul, self.inv = np.unique(np.asarray(logs_tok).astype(str), return_inverse=True)
        self.L = len(ul)
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(self.L, np.full(self.L, 1.0 / self.L), size=n_boot).astype(np.float64)

    def sums(self, x: np.ndarray) -> np.ndarray:
        """per-token values x [n] (or [n, C], summed over C) -> resampled sums [B]."""
        x = np.asarray(x, np.float64)
        x = x.reshape(len(self.inv), -1).sum(1)
        return self.W @ np.bincount(self.inv, weights=x, minlength=self.L)

    def ratio(self, num: np.ndarray, den: np.ndarray, alpha: float = 0.05) -> List[float]:
        n, d = self.sums(num), self.sums(den)
        est = np.where(d != 0, n / np.where(d != 0, d, 1.0), np.nan)
        if not np.isfinite(est).any():
            return [float("nan"), float("nan")]
        lo, hi = np.nanquantile(est, [alpha / 2, 1 - alpha / 2])
        return [float(lo), float(hi)]


def recall(out: np.ndarray, fail: np.ndarray, valid: np.ndarray, lb: Optional[LogBoot]) -> Dict:
    """outside flags / official fail [n, C] bool on valid [n, C] rows -> fail recall P(out | fail), pass recall
    P(in | pass) (+ bootstrap CIs of the ratio estimators), counts."""
    f, p = fail & valid, ~fail & valid
    nf, npass = int(f.sum()), int(p.sum())
    fr = float((out & f).sum() / nf) if nf else float("nan")
    pr = float((~out & p).sum() / npass) if npass else float("nan")
    res = {"fail_recall": fr, "pass_recall": pr, "n_fail": nf, "n_pass": npass}
    if lb is not None:
        res["fail_recall_ci"] = lb.ratio(out & f, f) if nf else [float("nan")] * 2
        res["pass_recall_ci"] = lb.ratio(~out & p, p) if npass else [float("nan")] * 2
    return res


def fidelity(out: np.ndarray, fail: np.ndarray, valid: np.ndarray, keys: Sequence[str],
             lb: Optional[LogBoot]) -> Dict:
    """criterion-1 metrics per key of `keys` and joint (any of them) -- out / fail [n, 96, 3] (KEYS order)."""
    res = {k: recall(out[..., KI[k]], fail[..., KI[k]], valid, lb) for k in keys}
    ki = [KI[k] for k in keys]
    res["joint"] = recall(out[..., ki].any(-1), fail[..., ki].any(-1), valid, lb)
    res["keys"] = list(keys)
    return res


def argmax_tie(s: np.ndarray) -> np.ndarray:
    """select2.select96's argmax (first maximum in TIE_ORDER: identity columns first, then the variants) -- the
    deployment rule of eval_teacher_navtest_r34 (stage eval (iii) select_modes); NOT used by criterion S."""
    from navsim.agents.para_ssr.ck.select2 import TIE_ORDER
    return TIE_ORDER[np.argmax(s[:, TIE_ORDER], 1)].astype(np.int64)


def argmax_pool(s: np.ndarray) -> np.ndarray:
    """Phase 0b criterion S's argmax (tools/ck/v4/oracle/heldout_analyze.amax): the first maximum in the stored pool96
    order, parent k then its variants -- in the select2 layout c = 6 k + v that is the column order itself."""
    return np.argmax(s, 1).astype(np.int64)


def tube_switch(score: np.ndarray, allowed: np.ndarray, inside: np.ndarray) -> Dict[str, np.ndarray]:
    """Phase 0b criterion S picks over the 96: score [n, 96] (deployment blend), allowed [n, 96] (valid columns),
    inside [n, 96] (tube) -> blend (no tube), filter (best inside, none inside -> the blend pick), switch (column 0 if
    inside, else filter); ties in the pool order (argmax_pool, as heldout_analyze)."""
    s = np.where(allowed & np.isfinite(score), score, -np.inf)
    blend = argmax_pool(s)
    sf = np.where(inside & allowed, s, -np.inf)
    filt = np.where(np.isfinite(sf).any(1), argmax_pool(sf), blend)
    return {"blend": blend, "filter": filt, "switch": np.where(inside[:, 0], 0, filt)}


def fails3(lab: np.ndarray) -> np.ndarray:
    """[..., 9] LABEL_COLS -> official NC or TTC or DAC fail."""
    from navsim.agents.para_ssr.ck.constants import LBL
    return (lab[..., LBL["nc"]] < 1) | (lab[..., LBL["ttc"]] < 1) | (lab[..., LBL["dac"]] < 1)


def criterion_s(lab96: np.ndarray, score: np.ndarray, allowed: np.ndarray, inside: np.ndarray, logs: np.ndarray
                ) -> Tuple[Dict, np.ndarray, Dict[str, np.ndarray]]:
    """-> (row dict, per-token dPDMS x 100 of switch - blend [n], picks)."""
    from navsim.agents.para_ssr.ck.constants import LBL
    from tools.ck.eval_ck import log_bootstrap
    pk = tube_switch(score, allowed, inside)
    ar = np.arange(len(lab96))
    Lb, Ls = lab96[ar, pk["blend"]], lab96[ar, pk["switch"]]
    pdm = lambda L: 100.0 * L[:, LBL["pdms"]]
    d = pdm(Ls) - pdm(Lb)
    ok_b = ~fails3(Lb)
    n_brk, den = int(fails3(Ls)[ok_b].sum()), int(ok_b.sum())
    brk = n_brk / den if den else float("nan")
    boot = log_bootstrap(d, logs, N_BOOT)
    row = dict(pdms_switch=float(pdm(Ls).mean()), pdms_blend=float(pdm(Lb).mean()), d_vs_blend=boot, break_=brk,
               break_n=n_brk, break_den=den, top1_outside=float((~inside[:, 0]).mean()),
               changed_vs_top1=float((pk["switch"] != 0).mean()),
               no_inside=float((~(inside & allowed).any(1)).mean()), fail3_switch=float(fails3(Ls).mean()),
               bars_pass=bool(boot["mean"] >= S_MIN and boot["lo"] > 0 and brk <= BREAK_MAX))
    return row, d, pk


def critic_prob(pd_: Optional[np.ndarray], pm: Optional[np.ndarray]) -> np.ndarray:
    """[..., 5] CK_KEYS probabilities of a DET / MAP pair (each key from its own teacher, EP mean; DET only: dac
    neutral (1), EP the DET's; MAP only: nc / ttc / comfort neutral)."""
    from navsim.agents.para_ssr.ck.constants import CK_KEYS
    from tools.ck.v4.eval_teacher_pair_navtest import pair_prob
    if pd_ is not None and pm is not None:
        return pair_prob(pd_, pm)
    src = pd_ if pd_ is not None else pm
    out = np.array(src, np.float64, copy=True)
    neutral = ("dac",) if pd_ is not None else ("nc", "ttc", "comfort")
    for k in neutral:
        out[..., CK_KEYS.index(k)] = 1.0
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


# ----------------------------------------------------------------------------------------------- stage eval
def _teacher_arrays(a, name: Optional[str], N: int) -> Optional[Dict[str, np.ndarray]]:
    if not name:
        return None
    td = split_root(a) / tname(name)
    if not (td / "done.npy").is_file():
        raise SystemExit(f"{td}: no infer outputs (run --stage infer)")
    out = {k: np.load(td / f"{k}.npy", mmap_mode="r") for k in ("done", "bev_ok", "score_logit16", "score_logit80",
                                                                 "slack16", "slack80")}
    out["meta"] = U.read_json(td / "infer_meta.json", {}) or {}
    return out


def _base_probs(root: Path, name: str):
    td = root / name / "r34"
    need = [td / f for f in ("prob16.npy", "prob80.npy", "done.npy")]
    if not all(p.is_file() for p in need):
        return None
    return {"prob16": np.load(need[0], mmap_mode="r"), "prob80": np.load(need[1], mmap_mode="r"),
            "done": np.load(need[2])}


def blend_critic(sp: Dict, name: str) -> Dict:
    """Deployment-blend critic of criterion S: 'X' = X's own probabilities as stored (Phase 0b: ck2T10 alone, all five
    heads as trained) | 'X,Y' = the DET / MAP pair (critic_prob: each key from its own teacher, EP mean) -> {name,
    parts, done [N] bool, prob(r) -> (p16 [n, 16, 5], p80 [n, 80, 5])}."""
    parts = [x for x in name.split(",") if x]
    if not 1 <= len(parts) <= 2:
        raise SystemExit(f"--blend-critic {name!r}: one critic or a det,map pair")
    P = [_base_probs(sp["base_root"], x) for x in parts]
    if any(x is None for x in P):
        raise SystemExit(f"blend critic {name}: no prob16 / prob80 / done under {sp['base_root']}")
    done = np.logical_and.reduce([np.asarray(x["done"], bool) for x in P])

    def prob(r):
        g = lambda x, k: np.asarray(x[k][r], np.float64)
        if len(P) == 1:
            return g(P[0], "prob16"), g(P[0], "prob80")
        return critic_prob(g(P[0], "prob16"), g(P[1], "prob16")), critic_prob(g(P[0], "prob80"), g(P[1], "prob80"))
    return {"name": name, "parts": parts, "done": done, "prob": prob}


def load_gt_tube(a, N: int) -> Optional[Dict[str, np.ndarray]]:
    """The R2 GT tube of this split, read-only: <root>/gt_tube_r2 (stage gttube) or --gt-tube DIR (another run in the
    same format: meta.json with this split and representation_id 'R2', packed-row order) -> arrays (done, ok, hit16 /
    hit80 [N, K, 3], rule16 / rule80 when present) + meta; None when the own dir has no tube (an explicit --gt-tube
    that does not hold one is an error)."""
    ext = bool(a.gt_tube)
    gd = Path(a.gt_tube) if ext else split_root(a) / "gt_tube_r2"
    if not (gd / "done.npy").is_file():
        if ext:
            raise SystemExit(f"--gt-tube {gd}: no done.npy")
        return None
    meta = U.read_json(gd / "meta.json", None)
    if ext:
        rep = (meta or {}).get("representation_id") or ("R2" if str((meta or {}).get("representation", "")).startswith(
            "Phase 0b R2") else None)
        if meta is None or meta.get("split") != a.split or rep != REP_ID:
            raise SystemExit(f"--gt-tube {gd}: meta.json must name split {a.split!r} and representation_id {REP_ID!r} "
                             f"(got split {(meta or {}).get('split')!r}, representation {rep!r})")
    elif meta is not None and meta.get("split") not in (None, a.split):
        raise SystemExit(f"{gd}: meta.json split {meta.get('split')!r} != {a.split!r}")
    want = {"done": (N,), "ok": (N,), "hit16": (N, K16, 3), "hit80": (N, K80, 3), "rule16": (N, K16, 3),
            "rule80": (N, K80, 3)}
    out: Dict = {"dir": str(gd), "meta": meta}
    for k, shp in want.items():
        p = gd / f"{k}.npy"
        if not p.is_file():
            if k.startswith("rule"):
                continue
            raise SystemExit(f"GT tube {gd}: {k}.npy missing")
        x = np.load(p, mmap_mode="r")
        if tuple(x.shape) != shp or x.dtype != np.bool_:
            raise SystemExit(f"GT tube {gd}: {k}.npy is {x.dtype} {tuple(x.shape)}, expected bool {shp} (packed-row "
                             f"order of CK/packed/{a.split})")
        out[k] = x
    if ("rule16" in out) != ("rule80" in out):
        raise SystemExit(f"GT tube {gd}: rule16 / rule80 incomplete")
    return out


def s_block(LP: np.ndarray, score: np.ndarray, allowed: np.ndarray, logs: np.ndarray, lb: LogBoot,
            tubes: Dict[str, np.ndarray]) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """criterion S rows of every tube (inside [n, 96]) with one blend score, + the gain fractions of the learned tube
    (vs gt_r2_same_keys = 'gain_fraction', vs gt_rule_same_keys = 'gain_fraction_vs_rule') -> (block, picks)."""
    from tools.ck.eval_ck import log_bootstrap
    blk: Dict = {}
    d: Dict[str, np.ndarray] = {}
    picks: Dict[str, np.ndarray] = {}
    for nm, inside in tubes.items():
        row, dd, pk = criterion_s(LP, score, allowed, inside & allowed, logs)
        blk[nm], d[nm] = row, dd
        picks.update({f"S_{nm}_{k}": v for k, v in pk.items()})
    for tag, ref in (("gain_fraction", "gt_r2_same_keys"), ("gain_fraction_vs_rule", "gt_rule_same_keys")):
        if "learned" in d and ref in d:
            den = float(d[ref].sum())
            blk[tag] = {"vs": ref, "estimate": float(d["learned"].sum() / den) if den != 0 else float("nan"),
                        "ci": lb.ratio(d["learned"], d[ref]),
                        "d_learned_minus_ref": log_bootstrap(d["learned"] - d[ref], logs, N_BOOT)}
    return blk, picks


def phase0b_check(a, r: np.ndarray, row: Optional[Dict]) -> Dict:
    """The tool's criterion S of the R2 GT tube on all keys with the Phase 0b blend (ck2T10) vs the pre-registered
    held-out run (HO/results.json selection.criterion_S): comparable only on its token set (navtrain_val heldout, no
    --limit, the same packed rows) -> {comparable, match, ours, phase0b, reason}."""
    out: Dict = {"comparable": False, "match": None, "results": str(HO_DIR / "results.json")}
    if a.split != "navtrain_val" or a.subset != "heldout" or a.limit:
        out["reason"] = "not the Phase 0b token set (needs --split navtrain_val --subset heldout, no --limit)"
        return out
    if row is None:
        out["reason"] = "no R2 GT-tube row (no GT tube)"
        return out
    rp, hp = HO_DIR / "results.json", HO_DIR / "prep" / "ho_rows.npy"
    if not (rp.is_file() and hp.is_file()):
        out["reason"] = f"no Phase 0b outputs ({rp} / {hp})"
        return out
    R = json.loads(rp.read_text())
    ho = np.load(hp)
    if int(R.get("n_tokens_used", -1)) != len(ho):
        out["reason"] = f"Phase 0b used {R.get('n_tokens_used')} of its {len(ho)} rows (its token mask is not stored)"
        return out
    if not np.array_equal(np.sort(ho), np.sort(r)):
        out["reason"] = (f"token sets differ: {len(r)} here vs {len(ho)} Phase 0b ({len(np.setdiff1d(ho, r))} Phase 0b "
                         f"rows missing here, e.g. teacher outputs / labels / BEV)")
        return out
    cs = R["selection"]["criterion_S"]
    ours = {"mean": row["d_vs_blend"]["mean"], "lo": row["d_vs_blend"]["lo"], "hi": row["d_vs_blend"]["hi"],
            "break_n": row["break_n"], "break_den": row["break_den"]}
    theirs = {"mean": cs["d_vs_blend"]["mean"], "lo": cs["d_vs_blend"]["lo"], "hi": cs["d_vs_blend"]["hi"],
              "break_n": cs["break_"]["n"], "break_den": cs["break_"]["den"], "verdict": cs["verdict"]}
    out.update(comparable=True, ours=ours, phase0b=theirs,
               match=bool(abs(ours["mean"] - theirs["mean"]) <= 1e-9 and ours["break_n"] == theirs["break_n"]
                          and ours["break_den"] == theirs["break_den"]),
               ci_abs_diff=max(abs(ours["lo"] - theirs["lo"]), abs(ours["hi"] - theirs["hi"])))
    return out


def _common(a, need_teachers: bool) -> Dict:
    """Shared inputs of stages eval / gtcheck: rows, labels, pool, blend critics, GT tube, teachers, token set."""
    from navsim.agents.para_ssr.ck.constants import LBL
    from navsim.agents.para_ssr.ck.select2 import allowed_mask
    sp = SPLITS[a.split]
    pdir, tdf, _ = packed(a.split)
    N = len(tdf)
    rows = base_rows(a)
    vd = sp["variants"]
    labc, okc, labv, okv = load_labels(a.split, vd, N)
    valid80 = np.load(vd / "valid80.npy")
    tch = {"det": _teacher_arrays(a, a.det, N), "map": _teacher_arrays(a, a.map, N)} if need_teachers else \
        {"det": None, "map": None}
    if need_teachers and tch["det"] is None and tch["map"] is None:
        raise SystemExit("--stage eval needs --det and / or --map (stage gtcheck: the GT rows without teachers)")
    primary = a.blend_critic or sp["blend"]
    names = [primary] + ([PHASE0B_BLEND] if primary != PHASE0B_BLEND else [])
    blends = [blend_critic(sp, nm) for nm in names]
    gt = load_gt_tube(a, N)
    keep = np.zeros(N, bool)
    keep[rows] = True
    keep &= okc.all(1) & (okv | ~valid80).all(1)
    for b in blends:
        keep &= b["done"]
    for t in tch.values():
        if t is not None:
            keep &= np.asarray(t["done"], bool) & np.asarray(t["bev_ok"], bool)
    if gt is not None:
        keep &= np.asarray(gt["done"], bool) & np.asarray(gt["ok"], bool)
    return dict(sp=sp, pdir=pdir, tdf=tdf, N=N, rows=rows, labc=labc, labv=labv, valid80=valid80, tch=tch,
                blends=blends, gt=gt, keep=keep, LBL=LBL, allowed_mask=allowed_mask)


def _gt_flags(gt: Dict, r: np.ndarray) -> Dict[str, np.ndarray]:
    out = {"gt_r2": to_pool(np.asarray(gt["hit16"][r]), np.asarray(gt["hit80"][r]))}
    if "rule16" in gt:
        out["gt_rule"] = to_pool(np.asarray(gt["rule16"][r]), np.asarray(gt["rule80"][r]))
    return out


def _s_all(C: Dict, r: np.ndarray, LP, allowed, logs, lb, tube_sets: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
    """criterion S blocks for every blend critic: res['criterion_S'] = the primary's, 'criterion_S_phase0b_convention'
    = ck2T10 alone when the primary is another critic (navtest)."""
    from navsim.agents.para_ssr.ck.select2 import pool_scores
    v2f = np.asarray(np.load(C["pdir"] / "v2_final.npy", mmap_mode="r")[r], np.float64)
    v2i = np.asarray(np.load(C["pdir"] / "v2_im.npy", mmap_mode="r")[r], np.float64)
    out, picks = {}, {}
    for q, b in enumerate(C["blends"]):
        p16, p80 = b["prob"](r)
        score = pool_scores(v2f, v2i, to_pool(p16, p80), beta=1.0)
        blk, pk = s_block(LP, score, allowed, logs, lb, tube_sets)
        blk["blend_critic"] = b["name"]
        key = "criterion_S" if q == 0 else "criterion_S_phase0b_convention"
        blk["label"] = ("primary" if q == 0 else "Phase 0b convention (ck2T10 alone as the deployment blend)")
        out[key] = blk
        picks.update({(f"{k}" if q == 0 else f"conv_{k}"): v for k, v in pk.items()})
    return out, picks


def stage_gtcheck(a) -> Dict:
    """GT-only rows (no teacher): (i) fidelity of the R2 / readout-rule GT tubes, (ii) criterion S with them on all
    three keys for each blend critic, and the Phase 0b reproduction check -> <root>/gtcheck.json / .md."""
    t0 = time.time()
    C = _common(a, need_teachers=False)
    if C["gt"] is None:
        raise SystemExit(f"stage gtcheck: no GT tube under {split_root(a) / 'gt_tube_r2'} (run --stage gttube)")
    r = np.flatnonzero(C["keep"])
    if len(r) == 0:
        raise SystemExit("[gtcheck] no token has labels, blend critics and the GT tube")
    LBL = C["LBL"]
    logs = C["tdf"].log.astype(str).to_numpy()[r]
    lb = LogBoot(logs)
    LP = to_pool(C["labc"][r], C["labv"][r])
    allowed = C["allowed_mask"](to_pool(np.ones((len(r), K16), bool), C["valid80"][r]))
    fail = np.stack([LP[..., LBL[k]] < 1 for k in KEYS], -1)
    G = _gt_flags(C["gt"], r)
    res: Dict = {"stage": "gtcheck", "split": a.split, "subset": a.subset, "limit": a.limit, "n_tokens": int(len(r)),
                 "n_logs": int(lb.L), "gt_tube": C["gt"]["dir"], "fidelity": {}, "caveats": []}
    for nm, h in G.items():
        res["fidelity"][f"{nm}_all_keys"] = fidelity(h, fail, allowed, KEYS, lb)
    if "gt_rule" in G:
        res["fidelity"]["agreement_rule_vs_r2"] = {
            **{k: float((G["gt_rule"][..., KI[k]] == G["gt_r2"][..., KI[k]])[allowed].mean()) for k in KEYS},
            "joint": float((G["gt_rule"].any(-1) == G["gt_r2"].any(-1))[allowed].mean())}
    tubes = {f"{nm}_all_keys": ~h.any(-1) for nm, h in G.items()}
    S, _ = _s_all(C, r, LP, allowed, logs, lb, tubes)
    res.update(S)
    conv = res.get("criterion_S_phase0b_convention", res["criterion_S"])
    res["phase0b_check"] = phase0b_check(a, r, conv.get("gt_r2_all_keys"))
    res["sec"], res["finished"] = round(time.time() - t0, 1), kst()
    root = split_root(a)
    root.mkdir(parents=True, exist_ok=True)
    (root / "gtcheck.json").write_text(json.dumps(res, indent=1, default=float))
    md = report_md(res)
    (root / "gtcheck.md").write_text(md)
    print(md, flush=True)
    _raise_on_mismatch(res)
    return res


def _raise_on_mismatch(res: Dict) -> None:
    c = res.get("phase0b_check") or {}
    if c.get("comparable") and not c.get("match"):
        raise SystemExit(f"Phase 0b reproduction FAILED (outputs written): ours {c['ours']} vs Phase 0b {c['phase0b']}")


def stage_eval(a) -> Dict:
    from tools.ck.e2e2.eval_teacher_navtest_r34 import select_modes
    from tools.ck.eval_ck import log_bootstrap
    t0 = time.time()
    C = _common(a, need_teachers=True)
    sp, N, tch, gt, LBL = C["sp"], C["N"], C["tch"], C["gt"], C["LBL"]
    rows, labc, labv, valid80 = C["rows"], C["labc"], C["labv"], C["valid80"]
    keys = tuple(k for k in KEYS if (k in ("nc", "ttc") and tch["det"] is not None)
                 or (k == "dac" and tch["map"] is not None))
    base_names = [b for b in (a.baselines.split(";") if a.baselines else sp["baselines"]) if b]
    bases = {}
    for b in base_names:
        det_b, map_b = (b.split(",") + [None])[:2]
        P = {"det": _base_probs(sp["base_root"], det_b) if det_b and tch["det"] is not None else None,
             "map": _base_probs(sp["base_root"], map_b) if map_b and tch["map"] is not None else None}
        if (tch["det"] is not None and P["det"] is None) or (tch["map"] is not None and P["map"] is None):
            log(f"[eval] baseline {b}: probabilities missing under {sp['base_root']}, skipped")
            continue
        bases[b] = P
    # token set (the same for (i) - (iii))
    keep = C["keep"].copy()
    for P in bases.values():
        for v in P.values():
            if v is not None:
                keep &= np.asarray(v["done"], bool)
    r = np.flatnonzero(keep)
    n = len(r)
    if n == 0:
        raise SystemExit("[eval] no token has every input (labels, teacher outputs, baselines, blend critic, GT tube)")
    logs = C["tdf"].log.astype(str).to_numpy()[r]
    lb = LogBoot(logs)
    LP = to_pool(labc[r], labv[r])                                                   # [n, 96, 9]
    allowed = C["allowed_mask"](to_pool(np.ones((n, K16), bool), valid80[r]))
    fail = np.stack([LP[..., LBL[k]] < 1 for k in KEYS], -1)                         # [n, 96, 3]
    learned = np.zeros((n, 6 * K16, 3), bool)
    if tch["det"] is not None:
        s = to_pool(np.asarray(tch["det"]["slack16"][r]), np.asarray(tch["det"]["slack80"][r]))
        learned[..., 0], learned[..., 1] = s[..., 0] < 0, s[..., 1] < 0
    if tch["map"] is not None:
        s = to_pool(np.asarray(tch["map"]["slack16"][r]), np.asarray(tch["map"]["slack80"][r]))
        learned[..., 2] = s[..., 2] < 0
    res: Dict = {"stage": "eval", "split": a.split, "subset": a.subset, "limit": a.limit, "n_tokens": int(n),
                 "n_rows_candidate": int(len(rows)), "n_logs": int(lb.L), "keys": list(keys), "det": tname(a.det),
                 "map": tname(a.map), "teacher_meta": {k: (v["meta"] if v is not None else None) for k, v in tch.items()},
                 "blend_critic": C["blends"][0]["name"], "gt_tube": gt["dir"] if gt is not None else None,
                 "baselines": list(bases), "caveats": [], "decision_rule": DECISION_RULE}
    if a.split == "navtrain_val":
        res["caveats"].append("navtrain_val: DET only (no ReSMap cache); the DET BEV backbone is in-sample on navtrain")
    if gt is None:
        res["caveats"].append("no GT tube: (i) GT rows, (ii) GT rows and the gain fractions skipped")
    elif "rule16" not in gt:
        res["caveats"].append("the GT tube has no readout-rule tube (rule16 / rule80): rule rows skipped")
    # (i) fidelity
    res["fidelity"] = {"learned": fidelity(learned, fail, allowed, keys, lb)}
    G = _gt_flags(gt, r) if gt is not None else {}
    ki = [KI[k] for k in keys]
    for nm, h in G.items():
        res["fidelity"][nm] = fidelity(h, fail, allowed, keys, lb)
        res["fidelity"][f"{nm}_all_keys"] = fidelity(h, fail, allowed, KEYS, lb)
        res["fidelity"][f"agreement_learned_{nm}"] = {
            **{k: float((learned[..., KI[k]] == h[..., KI[k]])[allowed].mean()) for k in keys},
            "joint": float((learned[..., ki].any(-1) == h[..., ki].any(-1))[allowed].mean())}
    # (ii) criterion S with the learned / GT / readout-rule tubes, for each blend critic
    tubes = {"learned": ~learned[..., ki].any(-1)}
    for nm, h in G.items():
        tubes[f"{nm}_same_keys"] = ~h[..., ki].any(-1)
        tubes[f"{nm}_all_keys"] = ~h.any(-1)
    S, picks_out = _s_all(C, r, LP, allowed, logs, lb, tubes)
    for blk in S.values():
        blk["keys"] = list(keys)
    res.update(S)
    picks_out["rows"] = r
    conv = res.get("criterion_S_phase0b_convention", res["criterion_S"])
    res["phase0b_check"] = phase0b_check(a, r, conv.get("gt_r2_all_keys"))
    # (iii) fixed-pool critic-only selection: the range pair vs each baseline pair
    v2f = np.asarray(np.load(C["pdir"] / "v2_final.npy", mmap_mode="r")[r], np.float64)
    v2i = np.asarray(np.load(C["pdir"] / "v2_im.npy", mmap_mode="r")[r], np.float64)
    p16 = critic_prob(sigmoid(tch["det"]["score_logit16"][r]) if tch["det"] is not None else None,
                      sigmoid(tch["map"]["score_logit16"][r]) if tch["map"] is not None else None)
    p80 = critic_prob(sigmoid(tch["det"]["score_logit80"][r]) if tch["det"] is not None else None,
                      sigmoid(tch["map"]["score_logit80"][r]) if tch["map"] is not None else None)
    ar = np.arange(n)
    LC = labc[r]
    pdm = lambda L: 100.0 * L[:, LBL["pdms"]]

    def modes(q16, q80):
        out = {}
        for mode, idx in select_modes(q16, q80, valid80[r], v2f, v2i).items():
            out[mode] = (idx, (LC if mode.startswith("r34") else LP)[ar, idx])
        return out

    mine = modes(p16, p80)
    sel = {"v2_top1_pdms": float(pdm(LC[:, 0]).mean()), "range": {}, "vs": {}}
    for mode, (idx, L) in mine.items():
        sel["range"][mode] = {"pdms": float(pdm(L).mean()), "changed": float((idx != 0).mean()),
                              "fail3": float(fails3(L).mean())}
        picks_out[f"sel_range_{mode}"] = idx
    for b, P in bases.items():
        q16 = critic_prob(np.asarray(P["det"]["prob16"][r], np.float64) if P["det"] is not None else None,
                          np.asarray(P["map"]["prob16"][r], np.float64) if P["map"] is not None else None)
        q80 = critic_prob(np.asarray(P["det"]["prob80"][r], np.float64) if P["det"] is not None else None,
                          np.asarray(P["map"]["prob80"][r], np.float64) if P["map"] is not None else None)
        theirs = modes(q16, q80)
        sel["vs"][b] = {}
        for mode, (idx, L) in theirs.items():
            d = pdm(mine[mode][1]) - pdm(L)
            sel["vs"][b][mode] = {"pdms_base": float(pdm(L).mean()), "pdms_range": float(pdm(mine[mode][1]).mean()),
                                  "range_minus_base": log_bootstrap(d, logs, N_BOOT),
                                  "better": float(np.mean(d > 1e-9)), "worse": float(np.mean(d < -1e-9))}
            picks_out[f"sel_{b.replace(',', '+')}_{mode}"] = idx
    sel["rule"] = ("pair: DET nc / ttc / comfort, MAP dac, EP mean" if tch["det"] is not None and tch["map"] is not None
                   else "DET only: dac neutral (1), EP = DET" if tch["det"] is not None
                   else "MAP only: nc / ttc / comfort neutral (1), EP = MAP")
    res["selection"] = sel
    res["sec"] = round(time.time() - t0, 1)
    res["finished"] = kst()
    tag = "+".join(x for x in (tname(a.det), tname(a.map)) if x)
    root = split_root(a)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"eval_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez(root / f"eval_{tag}_picks.npz", **picks_out)
    md = report_md(res)
    (root / f"eval_{tag}.md").write_text(md)
    print(md, flush=True)
    _raise_on_mismatch(res)
    return res


# ----------------------------------------------------------------------------------------------- report
def _ci(b) -> str:
    return f"{b['mean']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}]"


def _rc(x) -> str:
    if not np.isfinite(x["fail_recall"]) and not np.isfinite(x["pass_recall"]):
        return "n/a"
    f = lambda v: "n/a" if not np.isfinite(v) else f"{v:.3f}"
    s = f"{f(x['fail_recall'])} / {f(x['pass_recall'])}"
    if "fail_recall_ci" in x:
        s += f" ([{f(x['fail_recall_ci'][0])}, {f(x['fail_recall_ci'][1])}] / [{f(x['pass_recall_ci'][0])}, " \
             f"{f(x['pass_recall_ci'][1])}])"
    return s + f" (fail {x['n_fail']})"


def _s_md(S: Dict) -> List[str]:
    L = [f"Blend critic: {S['blend_critic']} ({S['label']}).", "",
         "| tube | PDMS switch | PDMS blend | switch - blend | break % (n / den) | top-1 outside % | no inside % | bars |",
         "|---|---|---|---|---|---|---|---|"]
    for nm, x in S.items():
        if not isinstance(x, dict) or "pdms_switch" not in x:
            continue
        L.append(f"| {nm} | {x['pdms_switch']:.2f} | {x['pdms_blend']:.2f} | {_ci(x['d_vs_blend'])} | "
                 f"{100 * x['break_']:.2f} ({x['break_n']} / {x['break_den']}) | {100 * x['top1_outside']:.1f} | "
                 f"{100 * x['no_inside']:.2f} | {'pass' if x['bars_pass'] else 'fail'} |")
    for tag in ("gain_fraction", "gain_fraction_vs_rule"):
        if tag in S:
            g = S[tag]
            L += ["", f"Fraction of the {g['vs']} gain (same keys, same tokens) recovered by the learned tube: "
                      f"{g['estimate']:.3f} [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}]; learned - ref "
                      f"{_ci(g['d_learned_minus_ref'])}."]
    return L


def report_md(res: Dict) -> str:
    keys = res.get("keys", list(KEYS))
    gtc = res.get("stage") == "gtcheck"
    title = "GT-tube check (no teacher)" if gtc else "range teacher as selection evidence"
    L = [f"# {title} -- {res['split']} ({res['subset']})", ""]
    if gtc:
        L.append(f"Generated {res['finished']}. Tokens {res['n_tokens']} ({res['n_logs']} logs); GT tube "
                 f"{res['gt_tube']}. PDMS x 100, official labels of the fixed r34 96 pool. CIs: log-clustered bootstrap "
                 "(2,000).")
    else:
        L.append(f"Generated {res['finished']}. Teachers: DET {res['det']}, MAP {res['map']}; keys {', '.join(keys)}; "
                 f"tokens {res['n_tokens']} ({res['n_logs']} logs); blend critic {res['blend_critic']}; GT tube "
                 f"{res['gt_tube']}. PDMS x 100, official labels of the fixed r34 96 pool ('_a' picks). CIs: "
                 "log-clustered bootstrap (2,000).")
        L += ["", ("**(c) decision rule: not registered** in docs/refiner_v4/V4_3_SPEC.md 'Phase 0b' -- the numbers below "
                   "are descriptive until the user fixes the rule (before the val infer)." if not res.get("decision_rule")
                   else f"(c) decision rule (registered): {res['decision_rule']}")]
    L.append("")
    for c in res["caveats"]:
        L.append(f"- caveat: {c}")
    c = res.get("phase0b_check") or {}
    if c.get("comparable"):
        o, t = c["ours"], c["phase0b"]
        L.append(f"- Phase 0b reproduction (R2 GT tube, all keys, ck2T10 blend vs HO/results.json criterion S): "
                 f"{'MATCH' if c['match'] else 'MISMATCH'} -- here {o['mean']:+.4f} [{o['lo']:+.2f}, {o['hi']:+.2f}], "
                 f"break {o['break_n']} / {o['break_den']}; Phase 0b {t['mean']:+.4f} [{t['lo']:+.2f}, {t['hi']:+.2f}], "
                 f"break {t['break_n']} / {t['break_den']} (CI bound diff {c['ci_abs_diff']:.2e}).")
    else:
        L.append(f"- Phase 0b reproduction: not comparable ({c.get('reason', 'n/a')}).")
    L += ["", "## (i) tube fidelity vs official labels (fail recall / pass recall)", "",
          "| tube | " + " | ".join(KEYS) + " | joint |", "|---|" + "---|" * (len(KEYS) + 1)]
    for nm, f in res["fidelity"].items():
        if nm.startswith("agreement"):
            continue
        ks = f["keys"]
        L.append(f"| {nm} ({','.join(ks)}) | " + " | ".join(_rc(f[k]) if k in f else "--" for k in KEYS)
                 + f" | {_rc(f['joint'])} |")
    for nm, ag in res["fidelity"].items():
        if nm.startswith("agreement"):
            L += ["", f"{nm} (flag agreement on the valid rows): " + ", ".join(f"{k} {v:.4f}" for k, v in ag.items())]
    L += ["", "gt_r2 = the Phase 0b R2 GT tube; gt_rule = the readout-rule perfect-field tube (the R2 teacher's own rule "
              "on the GT fields: the ceiling of the learned tube; differs from gt_r2 by the 2 m TTC switch and "
              "float32)."]
    L += ["", "## (ii) criterion S: switch selection with the tube vs the deployment blend over the 96", ""]
    for key in ("criterion_S", "criterion_S_phase0b_convention"):
        if key in res:
            L += _s_md(res[key]) + [""]
    if gtc:
        return "\n".join(L) + "\n"
    sel = res["selection"]
    L += [f"## (iii) fixed-pool selection of the range critic ({sel['rule']}) vs baselines", "",
          f"v2 top-1 PDMS {sel['v2_top1_pdms']:.2f}.", "",
          "| mode | range | " + " | ".join(f"{b} | range - {b}" for b in sel["vs"]) + " |",
          "|---|---|" + "---|---|" * len(sel["vs"])]
    for mode, x in sel["range"].items():
        L.append(f"| {mode} | {x['pdms']:.2f} | " + " | ".join(
            f"{sel['vs'][b][mode]['pdms_base']:.2f} | {_ci(sel['vs'][b][mode]['range_minus_base'])}"
            for b in sel["vs"]) + " |")
    L += ["", "pool96_ck_noim = the critic-only headline (no v2 im); r34_* modes pick among the 16 only."]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------- main
def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=tuple(SPLITS), required=True)
    ap.add_argument("--stage", default="infer,gttube,eval", help=f"comma list of {STAGES}")
    ap.add_argument("--det", default=None, help="DET range teacher run (dir or name)")
    ap.add_argument("--map", default=None, help="MAP range teacher run (navtest only)")
    ap.add_argument("--which", default="last")
    ap.add_argument("--subset", choices=("all", "heldout", "f1"), default=None,
                    help="token subset (default: navtrain_val heldout, navtest all)")
    ap.add_argument("--limit", type=int, default=0, help="first N tokens of the subset into <out-root>/smoke<N>")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--gt-tube", default="", help="stages eval / gtcheck: another run's GT tube dir, read-only (meta.json "
                                                  "split + representation_id 'R2', packed-row arrays); default "
                                                  "<root>/gt_tube_r2 (gttube refuses this flag)")
    ap.add_argument("--baselines", default="", help="';'-separated det[,map] baseline names under the split's eval root "
                                                    "(default navtest ck2T10,ck2M10;v4T10_haz,v4M10_haz, val ck2T10)")
    ap.add_argument("--blend-critic", default=None,
                    help="critic of the deployment blend in criterion S: 'X' or a det,map pair 'X,Y' (default: "
                         "navtrain_val ck2T10 = Phase 0b; navtest ck2T10,ck2M10 + the ck2T10-only Phase 0b-convention rows)")
    ap.add_argument("--device", default="cuda", help="cuda (CUDA_VISIBLE_DEVICES = one idle allowed GPU) | cpu")
    ap.add_argument("--approved", action="store_true",
                    help="stage infer on CUDA: the user approved this GPU run (or V4_USER_APPROVED=1)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers-io", type=int, default=4, help="DataLoader workers (stage infer)")
    ap.add_argument("--workers", type=int, default=24, help=f"CPU workers (stage gttube, <= {MAX_WORKERS})")
    return ap


def main(argv=None):
    a = get_parser().parse_args(argv)
    if a.subset is None:
        a.subset = SPLITS[a.split]["subset"]
    stages = [s.strip() for s in a.stage.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"--stage {bad} not in {STAGES}")
    if a.map and a.split == "navtrain_val":
        raise SystemExit("navtrain_val has no ReSMap cache: DET only (--map refused)")
    if a.workers > MAX_WORKERS:
        raise SystemExit(f"--workers <= {MAX_WORKERS}")
    out = {}
    if "infer" in stages:
        for role in ("det", "map"):
            nm = getattr(a, role)
            if nm:
                out[f"infer_{role}"] = stage_infer(a, role, nm)
    if "gttube" in stages:
        out["gttube"] = stage_gttube(a)
    if "gtcheck" in stages:
        out["gtcheck"] = stage_gtcheck(a)
    if "eval" in stages:
        out["eval"] = stage_eval(a)
    return out


if __name__ == "__main__":
    main()
