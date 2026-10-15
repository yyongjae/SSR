#!/usr/bin/env python
"""Wrapper around eval_teacher_navtest_gtfree.py (imported, NOT modified) for a non-default checkpoint and eval root.

  cd /workspace/yongjae/SSR-ck2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nice -n 10 /venv/ssr/bin/python tools/ck/e2e2/eval_teacher_navtest_gtfree_ep.py --run ck2T --which ep9 \
      --eval-root /home/external-user/ssd/yongjae_refiner/ck/ck2/eval/navtest_ep9 [--limit 50 --sanity --out ...]

What it changes (nothing else):
  * the base module's EVAL_ROOT -> --eval-root (so the base --out guard checks the new root), and the default out dir
    -> <eval-root>/<run>/gtfree;
  * --which is required (no silent fallback to ckpt_last) and ckpt_<which>.pt must exist;
  * refuses an --eval-root / --out inside the base default root (CK_DATA/ck2/eval/navtest), so the earlier
    ckpt_last run can never be overwritten (path-component check, not a string prefix: navtest_ep9 != navtest);
  * meta.json gets a "wrapper" entry (script, eval_root, which, ckpt sha16) before the base main runs.
Inference, selection (ck_final without im), variants, final modes, output layout and resumability are the base
script's: every other argument is forwarded unchanged to eval_teacher_navtest_gtfree.main.
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import argparse  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402
from tools.ck.e2e2 import eval_teacher_navtest_gtfree as G  # noqa: E402

BASE_ROOT = G.EVAL_ROOT          # .../ck2/eval/navtest (the earlier ckpt_last run) - never written by this wrapper


def _inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--run", required=True, help="ck2T | ck2M | run dir")
    ap.add_argument("--which", required=True, help="checkpoint tag, e.g. ep9 -> ckpt_ep9.pt")
    ap.add_argument("--eval-root", required=True, help="replaces the base EVAL_ROOT")
    ap.add_argument("--out", default="", help="default <eval-root>/<run>/gtfree")
    a, rest = ap.parse_known_args(argv)

    root = Path(a.eval_root)
    if _inside(root, BASE_ROOT):
        raise SystemExit(f"--eval-root {root} is inside the base root {BASE_ROOT} (earlier run); pick another root")
    run = Path(a.run) if "/" in a.run else G.TRAIN_ROOT / a.run
    ck_file = run / f"ckpt_{a.which}.pt"
    if not ck_file.is_file():
        raise SystemExit(f"missing checkpoint {ck_file}")
    out = Path(a.out) if a.out else root / run.name / "gtfree"
    if not _inside(out, root) or _inside(out, BASE_ROOT):
        raise SystemExit(f"--out {out} must be under {root} and outside {BASE_ROOT}")

    G.EVAL_ROOT = root
    out.mkdir(parents=True, exist_ok=True)
    meta_p = out / "meta.json"
    meta = U.read_json(meta_p, {}) or {}
    meta["wrapper"] = dict(script=str(Path(__file__).resolve()), base_script=str(Path(G.__file__).resolve()),
                           eval_root=str(root), which=a.which, ckpt=str(ck_file),
                           ckpt_sha16=U.sha256_file(ck_file, 16), forwarded=list(rest))
    U.write_json(meta_p, meta)
    return G.main(["--run", a.run, "--which", a.which, "--out", str(out), *rest])


if __name__ == "__main__":
    sys.exit(main())
