#!/usr/bin/env python
"""Test-time ablation of the planner's readouts, dumped as navtest trajectories.

Wraps tools/dump_navtest_trajectories.py.  ABLATE (env or --ablate) is a comma list of
  bev   zero the BEV cross-attention update in every planner layer
  det   zero the det/motion-memory cross-attention update
  map   zero the map-memory cross-attention update
The attention modules are swapped for a zero-returning stub only inside the layer's forward,
so the strict checkpoint load is untouched.  Perception heads still run on the BEV, so with
`bev` the planner sees the scene only through the det/map memories.

    python tools/analysis/dump_with_ablation.py --ablate det,map -- --experiment <exp> --checkpoint <ckpt> \
        --device cuda:2 --batch-size 1 --workers 8 --out work_dirs/eval/<name>.pkl
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.para_ssr.modules import planner_head as ph  # noqa: E402

_ATTN = {"bev": "bev_cross_attn", "det": "plan_det_cross_attn", "map": "plan_map_cross_attn"}


class _ZeroAttention(torch.nn.Module):
    def forward(self, query, key, value, need_weights=False, **kwargs):
        return torch.zeros_like(query), None


_ZERO = _ZeroAttention()
_ORIG_FORWARD = ph.PlanTaskMemoryLayer.forward


def _install(ablate: set) -> None:
    names = [_ATTN[k] for k in sorted(ablate)]

    def forward(self, *args, **kwargs):
        saved = {}
        for name in names:
            if name in self._modules:
                saved[name] = self._modules[name]
                self._modules[name] = _ZERO
        try:
            return _ORIG_FORWARD(self, *args, **kwargs)
        finally:
            for name, module in saved.items():
                self._modules[name] = module

    ph.PlanTaskMemoryLayer.forward = forward


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablate", default=os.environ.get("ABLATE", ""), help="comma list of bev,det,map")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="arguments for dump_navtest_trajectories.py (after --)")
    a = ap.parse_args()
    ablate = {k for k in a.ablate.split(",") if k}
    unknown = ablate - set(_ATTN)
    if unknown:
        ap.error(f"unknown ablation {sorted(unknown)}; choose from {sorted(_ATTN)}")
    _install(ablate)
    print(f"[ablation] zeroed readouts: {sorted(ablate) or 'none'}", flush=True)
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    sys.argv = [str(REPO_ROOT / "tools/dump_navtest_trajectories.py")] + rest
    runpy.run_path(sys.argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
