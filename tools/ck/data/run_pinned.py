#!/usr/bin/env python
"""Run a v1 data tool (tools/refiner/<name>.py or any module path) with navsim pinned to the SSR-ck worktree.

The v1 builders (build_metric_cache, score_trajectories, extract_lead ...) prepend the MAIN checkout to sys.path at
import time.  Importing navsim from SSR-ck first keeps every navsim.* submodule on the SSR-ck tree (submodules resolve
through navsim.__path__), so the CK rule navsim.__file__ under SSR-ck holds for the whole process and its forked
workers.  The tool is imported as a module (picklable worker functions) and its main() is called with the remaining
argv.

  python tools/ck/data/run_pinned.py build_metric_cache build --splits val_mc_part1 --splits-dir DIR --workers 3 ...
"""
import os
import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])   # repo root of this worktree


def main():
    if CK not in sys.path:
        sys.path.insert(0, CK)
    import navsim
    assert navsim.__file__.startswith(CK), navsim.__file__
    name = sys.argv[1]
    if os.path.sep in name or name.endswith(".py"):
        d, f = os.path.split(os.path.abspath(name))
        sys.path.insert(0, d)
        name = f[:-3] if f.endswith(".py") else f
    else:
        sys.path.insert(0, CK + "/tools/refiner")
    mod = __import__(name)
    import navsim as nv2
    assert nv2.__file__.startswith(CK), nv2.__file__
    print(f"[run_pinned] navsim {navsim.__file__} | {name} {mod.__file__}", flush=True)
    sys.argv = [mod.__file__] + sys.argv[2:]
    mod.main()


if __name__ == "__main__":
    main()
