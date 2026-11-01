#!/usr/bin/env python
"""Wrapper around eval_teacher_navtest_r34.py for a fixed-epoch teacher checkpoint (user 2026-10-08: "10epoch짜리로
navtest 다시 진행해봐 그럼 둘다").  The old script is imported unchanged; this file only
  * changes two defaults: --which ep9 (ckpt_ep9.pt = 10 epochs done) and --out-root CK_DATA/ck2/eval/navtest_ep9,
  * adds stage 'linkvar': reuse the previous run's official var80 labels when the rebuilt variant trajectories are
    identical (array sha16 == the label meta's traj_sha16 AND bitwise equal to the old traj80 / valid80); the link is
    <out-root>/r34_variants/labels -> <old labels dir>, a record goes to <out-root>/r34_variants/reuse_var_labels.json,
  * after stage eval, fixes the table title "(ckpt_last)" -> "(ckpt_<which>)" (the old script hard-codes it).

  cd /workspace/yongjae/SSR-ck2 && PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
    /venv/ssr/bin/python tools/ck/e2e2/eval_teacher_navtest_r34_ep.py --stage variants,linkvar
  CUDA_VISIBLE_DEVICES=2 ... eval_teacher_navtest_r34_ep.py --stage infer,select --teacher ck2T
  CUDA_VISIBLE_DEVICES=3 ... eval_teacher_navtest_r34_ep.py --stage infer,select --teacher ck2M
  ... eval_teacher_navtest_r34_ep.py --stage eval --teacher ck2T        (after official scoring of score_stack.npy)
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import argparse  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402
from tools.ck.e2e2 import eval_teacher_navtest_r34 as R  # noqa: E402

OUT_ROOT_EP = R.CK_DATA / "ck2" / "eval" / "navtest_ep9"
OLD_ROOT = R.CK_DATA / "ck2" / "eval" / "navtest"
OLD_VAR_LABELS = OLD_ROOT / "labels" / "ck2eval_r34_var80"


def link_var(out_root: Path, limit: int, old_vd: Path, old_labels: Path) -> dict:
    from tools.ck.data import common as CM
    vd = out_root / (f"smoke{limit}" if limit else "") / "r34_variants"
    new_t = np.load(vd / "traj80.npy", mmap_mode="r")
    new_v = np.load(vd / "valid80.npy")
    lm = U.read_json(old_labels / "meta.json")
    rec = dict(new=str(vd), old_variants=str(old_vd), old_labels=str(old_labels), created=R.kst(),
               label_meta_traj_path=lm.get("traj_path"), label_meta_traj_sha16=lm.get("traj_sha16"),
               label_meta_n_err_tokens=lm.get("n_err_tokens"), label_meta_n_rec_not_ok=lm.get("n_rec_not_ok"))
    rec["new_traj80_sha16_array"] = CM.sha16_array(new_t)
    old_t = np.load(old_vd / "traj80.npy", mmap_mode="r")
    old_v = np.load(old_vd / "valid80.npy")
    rec["shape_equal"] = bool(new_t.shape == old_t.shape and new_v.shape == old_v.shape)
    rec["traj80_bitwise_equal"] = bool(rec["shape_equal"] and np.array_equal(np.asarray(new_t).view(np.uint32),
                                                                              np.asarray(old_t).view(np.uint32)))
    rec["valid80_equal"] = bool(rec["shape_equal"] and np.array_equal(new_v, old_v))
    for k in ("v0.npy", "built.npy"):
        a, b = np.load(vd / k), np.load(old_vd / k)
        rec[f"{k[:-4]}_bitwise_equal"] = bool(a.shape == b.shape and a.tobytes() == b.tobytes())
    rec["sha_matches_label_meta"] = rec["new_traj80_sha16_array"] == lm.get("traj_sha16")
    ok = (rec["traj80_bitwise_equal"] and rec["valid80_equal"] and rec["sha_matches_label_meta"]
          and not lm.get("n_err_tokens") and int(lm.get("n", -1)) == new_t.shape[0] and int(lm.get("k", -1)) == 80)
    rec["reused"] = bool(ok)
    dst = vd / "labels"
    if ok:
        if dst.is_symlink() or dst.exists():
            if not (dst.is_symlink() and Path(os.readlink(dst)) == old_labels):
                raise SystemExit(f"{dst} exists and is not the expected link")
        else:
            dst.symlink_to(old_labels)
    U.write_json(vd / "reuse_var_labels.json", rec)
    print(f"[linkvar] {rec}", flush=True)
    if not ok:
        raise SystemExit("variant trajectories differ from the scored ones: score traj80.npy officially instead")
    return rec


def fix_title(out_root: Path, limit: int, teacher: str, which: str) -> None:
    td = out_root / (f"smoke{limit}" if limit else "") / teacher / "r34"
    p = td / "table.md"
    if p.is_file():
        s = p.read_text()
        lines = s.split("\n")
        if lines and "(ckpt_last)" in lines[0]:
            lines[0] = lines[0].replace("(ckpt_last)", f"(ckpt_{which})")
            p.write_text("\n".join(lines))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--stage", default="infer,select")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--which", default="ep9")
    ap.add_argument("--out-root", default=str(OUT_ROOT_EP))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--old-variants", default=str(OLD_ROOT / "r34_variants"))
    ap.add_argument("--old-var-labels", default=str(OLD_VAR_LABELS))
    a, rest = ap.parse_known_args(argv)
    stages = [s.strip() for s in a.stage.split(",") if s.strip()]
    inner = [s for s in stages if s != "linkvar"]
    fwd = ["--which", a.which, "--out-root", a.out_root, "--limit", str(a.limit)] + rest
    if a.teacher:
        fwd += ["--teacher", a.teacher]
    # variants first, then linkvar, then the remaining stages (the old script runs its stages in STAGES order)
    if "variants" in inner:
        R.main(["--stage", "variants"] + fwd)
    if "linkvar" in stages:
        print(f"[{R.kst()}] stage linkvar start", flush=True)
        link_var(Path(a.out_root), a.limit, Path(a.old_variants), Path(a.old_var_labels))
    rem = [s for s in inner if s != "variants"]
    if rem:
        R.main(["--stage", ",".join(rem)] + fwd)
        if "eval" in rem and a.teacher:
            fix_title(Path(a.out_root), a.limit, a.teacher, a.which)


if __name__ == "__main__":
    main()
