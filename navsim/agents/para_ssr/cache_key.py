"""Cache names that change when the cached tensor changes.

navsim finds a cached sample by ``builder.get_unique_name() + ".gz"`` and
nothing else (``navsim/planning/training/dataset.py``), so a constant name is a
silent-staleness bug rather than a naming detail.  When the map GT moved from
divider/ped_crossing/boundary to NAVSIM's own four layers, an existing
``para_ssr_target.gz`` would still have been loaded and the run would have
trained for days against the old labels without one error -- the failure mode
that leaves you comparing a model to numbers it never saw.

Appending a digest of the config values that actually reach the tensor fixes it
by construction: a changed geometry or class set changes the name, the old file
is simply not found, and the cache is rebuilt.  Old caches keep their old names,
so nothing is destroyed and a revert re-finds them.

Only the fields that alter the bytes belong in the digest.  Adding an unrelated
one (a loss weight, a learning rate) is not harmless -- it silently invalidates
terabytes of correct cache.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence, Tuple


def _canonical(value: Any) -> str:
    """Stable across processes and python versions, unlike ``hash()``."""
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, float):
        return format(value, ".6g")
    if isinstance(value, (list, tuple)):
        return "(" + ",".join(_canonical(v) for v in value) + ")"
    return str(value)


def cache_key(prefix: str, fields: Sequence[Tuple[str, Any]]) -> str:
    """``prefix`` plus an 8-hex digest of the given ``(name, value)`` pairs."""
    blob = ";".join(f"{name}={_canonical(value)}" for name, value in fields)
    return f"{prefix}_{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:8]}"
