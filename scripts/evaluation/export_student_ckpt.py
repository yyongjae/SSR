#!/usr/bin/env python3
"""Export a student-only PARA-SSR checkpoint from a Stage-2 distill Lightning file.

PDM evaluation instantiates ``agent=para_ssr_agent`` (``use_distill=false``) and
strict-loads the agent state. Stage-2 checkpoints also store frozen teacher
adapters under ``agent._distill.*``; those keys are training-only and must be
dropped before eval.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

DISTILL_PREFIXES = ("agent._distill.", "_distill.")


def _is_distill_key(key: str) -> bool:
    return key.startswith(DISTILL_PREFIXES)


def export_student_checkpoint(src: Path, dst: Path) -> Dict[str, Any]:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        raise RuntimeError(f"{src} is not a Lightning checkpoint with state_dict")
    raw_state = checkpoint["state_dict"]
    if not isinstance(raw_state, Mapping):
        raise RuntimeError(f"{src}: state_dict is not a mapping")

    kept = {key: value for key, value in raw_state.items() if not _is_distill_key(key)}
    dropped = len(raw_state) - len(kept)
    if not kept:
        raise RuntimeError(f"{src} has no student parameters after dropping distill keys")

    student = {
        "state_dict": kept,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "pytorch-lightning_version": checkpoint.get("pytorch-lightning_version"),
        "hyper_parameters": checkpoint.get("hyper_parameters"),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student, dst)
    return {
        "src": str(src),
        "dst": str(dst),
        "epoch": student["epoch"],
        "global_step": student["global_step"],
        "n_kept": len(kept),
        "n_dropped": dropped,
        "bytes": dst.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Stage-2 Lightning checkpoint")
    parser.add_argument("dst", type=Path, help="Student-only checkpoint for PDM eval")
    args = parser.parse_args()
    if not args.src.is_file():
        print(f"missing checkpoint: {args.src}", file=sys.stderr)
        return 2
    info = export_student_checkpoint(args.src, args.dst)
    print(
        "exported student ckpt "
        f"epoch={info['epoch']} step={info['global_step']} "
        f"kept={info['n_kept']} dropped_distill={info['n_dropped']} "
        f"-> {info['dst']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
