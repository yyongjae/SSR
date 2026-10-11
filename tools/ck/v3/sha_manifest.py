#!/usr/bin/env python
"""Per-file sha256 manifest of a data directory (refiner-v3 provenance; CPU only, read-only on --dir).

  PYTHONPATH=/workspace/yongjae/SSR-v3 /venv/ssr/bin/python -B tools/ck/v3/sha_manifest.py --dir <dir> --out <json>

Output json: {dir, resolved, n_files, total_bytes, files: {name: [sha256, bytes]}, manifest_sha256 (sha256 of the
sorted "name sha256" lines, the single value runs record), kst}.  Never overwrites --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context
from pathlib import Path

KST = timezone(timedelta(hours=9))


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _one(p: str):
    return os.path.basename(p), sha256_file(p), os.path.getsize(p)


def manifest(d: Path, workers: int = 16) -> dict:
    d = Path(d)
    files = sorted(str(p) for p in d.iterdir() if p.is_file())
    with get_context("fork").Pool(max(1, min(workers, 16))) as pool:
        rows = pool.map(_one, files, chunksize=256)
    lines = "\n".join(f"{n} {s}" for n, s, _ in sorted(rows)).encode()
    return dict(dir=str(d), resolved=os.path.realpath(str(d)), n_files=len(rows),
                total_bytes=int(sum(b for _, _, b in rows)), manifest_sha256=hashlib.sha256(lines).hexdigest(),
                files={n: [s, int(b)] for n, s, b in sorted(rows)},
                kst=datetime.now(KST).isoformat(timespec="seconds"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    m = manifest(Path(a.dir), a.workers)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m))
    os.replace(tmp, out)
    print(json.dumps({k: v for k, v in m.items() if k != "files"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
