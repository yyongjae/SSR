#!/usr/bin/env python
"""Shell export lines for the refiner teacher trainer, from tools/release/registry.yaml.

  teacher_env.py DET_TEACHER MAP_TEACHER

Prints SSR_DET_CACHE_TRAIN, SSR_DET_CACHE_VAL, SSR_DET_SHA_HEAD, SSR_MAP_CACHE_ROOT, SSR_MAP_SHA256 (the variables
refiner/data.py and refiner/resmap_cache.py read), SSR_MAP_REQUIRE_SEG=0 when the entry says so, DET_RUN and MAP_RUN (the entries'
refiner_run).  A teacher that is not ready: exit 3, with what it needs.
"""
from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[3]      # repo root of this worktree
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/release"))

from navsim.common import ssr_paths as SP  # noqa: E402


def export_lines(registry: dict, det: str, mapt: str) -> List[str]:
    vals: Dict[str, str] = {}
    for role, name in (("det", det), ("map", mapt)):
        e = (registry.get("teachers", {}).get(role) or {}).get(name)
        if e is None:
            raise SystemExit(f"{role} teacher {name!r} is not in the registry")
        if e.get("status") != "ready":
            raise SystemExit(f"{role} teacher {name!r} has status {e.get('status')!r}, missing: {e.get('needs')}")
        if role == "det":
            vals.update(SSR_DET_CACHE_TRAIN=e["cache_train"], SSR_DET_CACHE_VAL=e["cache_val"],
                        SSR_DET_SHA_HEAD=e["sha_head"], DET_RUN=e["refiner_run"])
        else:
            vals.update(SSR_MAP_CACHE_ROOT=e["cache_root"], SSR_MAP_SHA256=e["sha256"], MAP_RUN=e["refiner_run"])
            if e.get("require_seg") is False:
                vals["SSR_MAP_REQUIRE_SEG"] = "0"
    return [f"export {k}={shlex.quote(str(v))}" for k, v in vals.items()]


def main(argv=None) -> int:
    import verify_assets as VA
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: teacher_env.py DET_TEACHER MAP_TEACHER", file=sys.stderr)
        return 2
    reg = VA.load_registry(Path(os.environ.get("SSR_RELEASE_REGISTRY") or ROOT / VA.REGISTRY), SP.resolve())
    if reg is None:
        print(f"{VA.REGISTRY} not found", file=sys.stderr)
        return 3
    try:
        print("\n".join(export_lines(reg, argv[0], argv[1])))
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
