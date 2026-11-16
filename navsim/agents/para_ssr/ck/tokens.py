"""Token arrays and their provenance hash: ONE implementation for every token set of the repo, real or synthetic.

A real NAVSIM token is 16 hex characters.  A SimScale scene token (navsim/common/simscale.py) is '<16 hex>-<rrr>', 20
characters, and its 16 hex part is the real token at the same time stamp.  np.array(tokens, "U16") -- what every
token_sha16 was computed with before SimScale -- cuts a SimScale token down to that real token, so two token orders
that differ only in synthetic rows (or a synthetic row and its real twin) hashed alike and a provenance guard could not
tell them apart.  token_array never truncates; for token sets of at most 16 characters per token (every split before
SimScale) it is that same U16 array byte for byte, so every pinned token_sha16 keeps its value.

  token_array(tokens) -> np.ndarray     U16 when no token is longer than 16 characters, else U<longest>
  token_sha16(tokens) -> str            sha256 of the token_array bytes, first 16 hex characters: the token_sha16 of
                                        packed/meta.json, the variant build.json, raw-label shards/_plan.json, the
                                        warm-up provenance guard (e2e_data2._WUArrays.provenance) and the setup pins

No torch / navsim imports (data workers and setup tools import this module).
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

TOKEN_LEN = 16          # a real NAVSIM token; the narrowest dtype token_array uses (= the historical "U16")


def token_array(tokens: Iterable) -> np.ndarray:
    """tokens as a numpy unicode array, never truncated: dtype U16 when no token has more than 16 characters (shorter
    ones are zero-padded, as np.array(tokens, "U16") always did), else U<length of the longest token>."""
    toks = [str(t) for t in tokens]
    n = max((len(t) for t in toks), default=0)
    return np.array(toks, f"U{max(n, TOKEN_LEN)}")


def token_sha16(tokens: Iterable) -> str:
    """sha256(token_array(tokens) bytes)[:16]: order-sensitive identity of a token list (a row order)."""
    return hashlib.sha256(np.ascontiguousarray(token_array(tokens)).tobytes()).hexdigest()[:16]
