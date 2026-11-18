"""CK2 e2e selection over the 16 x V candidate pool (SPEC s1-5 / s5-2; user: old ck_final blended with v2 by beta, no
gamma, calibration offset OFF by default, all pool candidates, lateral applied AFTER selection).

Pool layout (= CKE2E2.infer, e2e_data2; variants.VariantLayout): column c = k * nv + v, k = v2 top-16 rank
(plan_final_rewards desc), v = layout.names; identity columns c = k * nv are the v2 top-16, column 0 = v2's trajectory.
The base layout (BASE_LAYOUT, the main recipe) is nv = 6, 96 columns: id, a-1.0, a-0.5, a+0.5, l-0.5, l+0.5.  Every
function takes a keyword-only `layout` (default BASE_LAYOUT; a variant-set ablation arm passes
VariantLayout.from_cfg(ck_e2e2.variants)) and refuses arrays whose column count is not 16 * layout.nv.

  pool_scores(v2_final16, v2_im16, prob96, beta)  score_c = (1 - beta) v2_final[c // nv]
                                                            + beta ck_final(p_c, im[c // nv])
                                                  (variants inherit the parent's v2_final / im, NU26; ck_final / blend
                                                  = ck/select.py, SEL_W / SEL_EPS); optional per-key logit `offset`
                                                  (calibration; default None = OFF); w = 4 weights or a SEL_W_SETS
                                                  name ('default' = SEL_W, 'noim', 'plugin' = (0, 1, 1, 1)).
  select96(..., valid96, beta, types)             argmax over the allowed columns (identity always allowed; variant
                                                  columns: valid96 & type in `types`; non-finite score never chosen);
                                                  ties -> identity columns first (k ascending), then variants in c order.
                                                  types: a variant-set NAME (SET_NAMES, resolved by meaning through
                                                  layout.sets: decel = a < 0, speed = a != 0, lat = d != 0, ...), None
                                                  (= 'all') or type ids OF THAT LAYOUT.
  final_traj(cand96, lat_traj96, idx, lat_mode)   'on': the selected column's laterally corrected trajectory (b-style,
                                                  user); 'on_except_latvar': uncorrected when the selected column is a
                                                  lateral variant (layout.lat_types, d != 0; base: l-0.5 / l+0.5), else
                                                  corrected (NU27); 'off': the selected pool trajectory.
numpy in / numpy out for the scores and indices; final_traj keeps the type of cand96 (torch or numpy).
The module constants K16, NV, G_K, VNAMES, VARIANT_SETS, LAT_TYPES and TIE_ORDER are the BASE layout's (checked against
BASE_LAYOUT at import); code that may see another layout uses the layout's own fields instead.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .constants import SEL_W, SEL_W_SETS  # noqa: F401  (SEL_W_SETS re-exported for the eval tools)
from .select import blend, ck_final, resolve_w, sel_w_name  # noqa: F401
from .variants import BASE_LAYOUT, VariantLayout  # noqa: F401  (re-exported for the eval tools)

K16, NV, G_K = 16, 6, 96                                                            # base layout aliases
VNAMES = ("id", "a-1.0", "a-0.5", "a+0.5", "l-0.5", "l+0.5")
VARIANT_SETS: Dict[str, Tuple[int, ...]] = {
    "none": (0,), "decel": (0, 1, 2), "speed": (0, 1, 2, 3), "lat": (0, 4, 5),
    "all_noaccel": (0, 1, 2, 4, 5), "all": (0, 1, 2, 3, 4, 5)}                       # NU28 (type ids = VNAMES index)
SET_NAMES: Tuple[str, ...] = tuple(VARIANT_SETS)                                    # = VariantLayout.SET_NAMES
LAT_MODES = ("on", "on_except_latvar", "off")                                       # NU27
LAT_TYPES = (4, 5)
# identity columns first (k ascending), then the variant columns in c order (tie-break order of select96)
TIE_ORDER = np.concatenate([np.arange(0, G_K, NV), np.array([c for c in range(G_K) if c % NV], np.int64)])


def _check_base_aliases() -> None:
    """the historical base constants above == BASE_LAYOUT (variants.VariantLayout of the base table)."""
    B = BASE_LAYOUT
    same = (B.nv == NV and B.g(K16) == G_K and B.names == VNAMES and tuple(B.sets) == SET_NAMES
            and {k: tuple(v) for k, v in B.sets.items()} == VARIANT_SETS and tuple(B.lat_types) == LAT_TYPES
            and tuple(VariantLayout.SET_NAMES) == SET_NAMES and np.array_equal(B.tie_order(K16), TIE_ORDER))
    if not same:
        raise RuntimeError(f"select2 base constants != variants.BASE_LAYOUT ({B!r}, sets {B.sets}, lat_types "
                           f"{B.lat_types}): the base selection would change")


_check_base_aliases()


def _np(x, dtype=None) -> np.ndarray:
    a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    return a if dtype is None else a.astype(dtype)


def _cols(x) -> int:
    return int(x.shape[1])


def check_cols(what: str, n_cols: int, layout: VariantLayout = BASE_LAYOUT, k: int = K16) -> None:
    """ValueError unless n_cols == k * layout.nv (a pool of another variant set, or the wrong layout passed)."""
    if int(n_cols) != layout.g(k):
        extra = "" if layout.is_base else f" (variant layout {list(layout.names)})"
        raise ValueError(f"{what}: {n_cols} pool columns, the variant layout needs {k} x {layout.nv} = "
                         f"{layout.g(k)}{extra}; pass the layout of the pool (VariantLayout.from_cfg / from_meta)")


@lru_cache(maxsize=32)
def _tie_order_cached(layout: VariantLayout) -> np.ndarray:
    t = layout.tie_order(K16).astype(np.int64)
    t.flags.writeable = False
    return t


def tie_order(layout: VariantLayout = BASE_LAYOUT) -> np.ndarray:
    """identity columns first (k ascending), then the variant columns in c order; base = TIE_ORDER."""
    return TIE_ORDER if layout == BASE_LAYOUT else _tie_order_cached(layout)


def set_types(types=None, *, layout: VariantLayout = BASE_LAYOUT) -> Tuple[int, ...]:
    """variant type ids of a selection set: None -> layout.sets['all']; a set name (SET_NAMES) -> layout.sets[name]
    (by meaning, so a name selects the same kind of variant in every layout); a sequence -> type ids of THIS layout
    (0 .. nv-1).  The base VARIANT_SETS tuples themselves are refused with a non-base layout when that set means other
    type ids there (base type ids would select the wrong variants: pass the set name)."""
    if types is None:
        return tuple(layout.sets["all"])
    if isinstance(types, str):
        if types not in layout.sets:
            raise ValueError(f"unknown variant set {types!r}; one of {list(layout.sets)}")
        return tuple(layout.sets[types])
    if not layout.is_base:
        for base_sets in (VARIANT_SETS, BASE_LAYOUT.sets):
            for k, v in base_sets.items():
                if types is v and tuple(v) != tuple(layout.sets[k]):
                    raise ValueError(f"base type ids {tuple(types)} of the set {k!r} (select2.VARIANT_SETS) with the "
                                     f"variant layout {list(layout.names)}, where {k!r} is {layout.sets[k]}: pass "
                                     f"the variant-set name instead")
    t = tuple(int(x) for x in types)
    bad = [x for x in t if not 0 <= x < layout.nv]
    if bad:
        raise ValueError(f"variant type ids {bad} outside 0..{layout.nv - 1} of the variant layout "
                         f"{list(layout.names)}")
    return t


def parent_of(x16: np.ndarray, *, layout: VariantLayout = BASE_LAYOUT) -> np.ndarray:
    """[N, 16, ...] -> [N, 16 * nv, ...] (column c takes row c // nv)."""
    return np.repeat(np.asarray(x16), layout.nv, axis=1)


def decode_cols(idx, *, layout: VariantLayout = BASE_LAYOUT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pool columns [N] -> (k = parent rank c // nv, v = type c % nv, name = layout.names[v]) (per-token exports)."""
    i = _np(idx).astype(np.int64)
    if i.size and (i.min() < 0 or i.max() >= layout.g(K16)):
        raise ValueError(f"pool columns outside 0..{layout.g(K16) - 1} of the variant layout {list(layout.names)}")
    v = i % layout.nv
    return i // layout.nv, v, np.asarray(layout.names, dtype=object)[v]


def apply_offset(prob: np.ndarray, offset: Optional[Sequence[float]]) -> np.ndarray:
    """sigmoid(logit(p) + offset) per CK key (calibration; None = unchanged)."""
    if offset is None:
        return prob
    off = np.asarray(offset, np.float64).reshape((1,) * (prob.ndim - 1) + (-1,))
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return 1.0 / (1.0 + np.exp(-(np.log(p) - np.log1p(-p) + off)))


def pool_scores(v2_final16, v2_im16, prob96, beta: float, w=SEL_W, offset=None, *,
                layout: VariantLayout = BASE_LAYOUT) -> np.ndarray:
    """v2_final16 / v2_im16 [N, 16], prob96 [N, 16 nv, 5] (CK2 sigmoid, CK_KEYS order) -> blended score [N, 16 nv]
    f64."""
    v2f = parent_of(_np(v2_final16, np.float64), layout=layout)
    im = parent_of(_np(v2_im16, np.float64), layout=layout)
    p = apply_offset(_np(prob96, np.float64), offset)
    g = layout.g(K16)
    if p.shape[1] != g or v2f.shape[1] != g:
        extra = "" if layout.is_base else f" (variant layout {list(layout.names)})"
        raise ValueError(f"pool_scores: expected 16 parents and {g} columns, got {v2f.shape} / {p.shape}{extra}")
    return blend(v2f, ck_final(p, im, w), beta)


def allowed_mask(valid96, types=None, *, layout: VariantLayout = BASE_LAYOUT) -> np.ndarray:
    """[N, 16 nv] bool: identity columns always; variant columns valid & type in `types` (set_types)."""
    v = _np(valid96).astype(bool)
    check_cols("allowed_mask: valid96", v.shape[-1], layout)
    t = layout.vtype(K16)
    in_set = np.isin(t, np.asarray(set_types(types, layout=layout), np.int64)) | (t == 0)
    return (v | (t == 0)[None]) & in_set[None]


def select96(v2_final16, v2_im16, prob96, valid96, beta: float, types=None, offset=None, w=SEL_W, *,
             layout: VariantLayout = BASE_LAYOUT) -> np.ndarray:
    """-> selected column [N] int64 (0 .. 16 nv - 1).  Non-finite / disallowed columns get -inf; all -inf -> column 0.
    types: set name / None (= 'all') / type ids (set_types)."""
    s = pool_scores(v2_final16, v2_im16, prob96, beta, w, offset, layout=layout)
    ok = allowed_mask(valid96, types, layout=layout) & np.isfinite(s)
    s = np.where(ok, s, -np.inf)
    tie = tie_order(layout)
    j = np.argmax(s[:, tie], axis=1)                       # first maximum in tie order
    return tie[j].astype(np.int64)


def exp_pdms96(prob96) -> np.ndarray:
    """prob [.., C, 5] (CK_KEYS order) -> expected PDMS E[NC] P(DAC) (5 EP + 5 TTC + 2 C) / 12, f64 (the official
    5/5/2 weights; the critic has no DDC output and the NAVSIM v1.1 score has no DDC term).  Any column count."""
    p = _np(prob96, np.float64)
    nc, dac, ep, ttc, cmf = (p[..., i] for i in range(5))
    return nc * dac * (5 * ep + 5 * ttc + 2 * cmf) / 12.0


def select96_pdms(score_logit96, valid96, types=None, *, layout: VariantLayout = BASE_LAYOUT) -> np.ndarray:
    """Plain expected-PDMS rule (user 2026-11-03): argmax over the allowed columns of exp_pdms96(sigmoid(logit)), the
    refiner critic alone (no im, no v2 score, nothing tuned).  Sigmoid in f64 from the logits (= eval_ck2.sigmoid);
    allowed columns and ties as select96.  -> selected column [N] int64."""
    z = _np(score_logit96, np.float64)
    check_cols("select96_pdms: score_logit96", _cols(z), layout)
    s = exp_pdms96(1.0 / (1.0 + np.exp(-z)))
    ok = allowed_mask(valid96, types, layout=layout) & np.isfinite(s)
    s = np.where(ok, s, -np.inf)
    tie = tie_order(layout)
    j = np.argmax(s[:, tie], axis=1)
    return tie[j].astype(np.int64)


def use_lat(idx, lat_mode: str, *, layout: VariantLayout = BASE_LAYOUT) -> np.ndarray:
    """[N] bool: whether the submitted trajectory is the laterally corrected one (pool columns of `layout`)."""
    if lat_mode not in LAT_MODES:
        raise ValueError(f"lat_mode must be one of {LAT_MODES}, got {lat_mode!r}")
    i = _np(idx).astype(np.int64)
    if i.size and (i.min() < 0 or i.max() >= layout.g(K16)):
        raise ValueError(f"use_lat: pool columns outside 0..{layout.g(K16) - 1} (16 x {layout.nv}); pass the layout "
                         f"of the pool")
    if lat_mode == "on":
        return np.ones(i.shape, bool)
    if lat_mode == "off":
        return np.zeros(i.shape, bool)
    return ~np.isin(i % layout.nv, np.asarray(layout.lat_types, np.int64))


def final_traj(cand96, lat_traj96, idx, lat_mode: str, *, layout: VariantLayout = BASE_LAYOUT):
    """-> (traj [N, 8, 3] (type of cand96), src [N] str 'pool' | 'lat')."""
    check_cols("final_traj: cand96", _cols(cand96), layout)
    check_cols("final_traj: lat_traj96", _cols(lat_traj96), layout)
    lat = use_lat(idx, lat_mode, layout=layout)
    src = np.where(lat, "lat", "pool")
    if isinstance(cand96, torch.Tensor):
        ii = torch.as_tensor(_np(idx).astype(np.int64), device=cand96.device)
        ar = torch.arange(cand96.shape[0], device=cand96.device)
        a, b = cand96[ar, ii], torch.as_tensor(lat_traj96, device=cand96.device)[ar, ii]
        m = torch.as_tensor(lat, device=cand96.device)[:, None, None]
        return torch.where(m, b.to(a.dtype), a), src
    c, lt = np.asarray(cand96), np.asarray(lat_traj96)
    ii = _np(idx).astype(np.int64)
    ar = np.arange(len(ii))
    return np.where(lat[:, None, None], lt[ar, ii], c[ar, ii]), src


def chosen_labels(lab_pool: np.ndarray, lab_lat: np.ndarray, idx, lat_mode: str, *,
                  layout: VariantLayout = BASE_LAYOUT) -> np.ndarray:
    """official labels of the submitted trajectory: [N, 16 nv, C] pool / lateral-corrected labels -> [N, C]."""
    check_cols("chosen_labels: lab_pool", _cols(lab_pool), layout)
    check_cols("chosen_labels: lab_lat", _cols(lab_lat), layout)
    ii = _np(idx).astype(np.int64)
    ar = np.arange(len(ii))
    lat = use_lat(ii, lat_mode, layout=layout)
    return np.where(lat[:, None], lab_lat[ar, ii], lab_pool[ar, ii])


# ----------------------------------------------------------------------------------------------- layout sources
def _float_list(v):
    if v is None:
        return None
    if isinstance(v, str):                       # hydra override written as a string, e.g. '[-1.0, -0.5]' / '[]'
        v = [s.strip() for s in v.strip().strip("[]()").split(",") if s.strip()]
    return [float(x) for x in v]


def layout_from_cfg(vcfg) -> VariantLayout:
    """the layout of a ck_e2e2.variants dict (a training / hydra config: lists or '[..]' strings; None or missing
    keys = base, as the CKE2E2Config defaults); a VariantLayout is returned as is."""
    if isinstance(vcfg, VariantLayout):
        return vcfg
    v = dict(vcfg or {})
    try:
        return VariantLayout.from_cfg({"speeds": _float_list(v.get("speeds")), "lats": _float_list(v.get("lats")),
                                       "combine": v.get("combine"), "cross_speeds": _float_list(v.get("cross_speeds")),
                                       "cross_lats": _float_list(v.get("cross_lats"))})
    except (TypeError, ValueError) as e:
        raise ValueError(f"ck_e2e2.variants {v}: not a variant table ({e})") from None


def layout_from_meta(meta, what: str = "dump") -> VariantLayout:
    """the layout recorded as 'variant_layout' (VariantLayout.meta()); None / no keys = base (written before layouts
    were recorded).  The stored names / nv must equal the rebuilt table's (ValueError otherwise)."""
    lay = VariantLayout.from_meta(meta)
    if meta:
        if "names" in meta and tuple(meta["names"]) != lay.names:
            raise ValueError(f"{what}: variant_layout names {list(meta['names'])} != the table of its speeds / lats "
                             f"{list(lay.names)}")
        if "nv" in meta and int(meta["nv"]) != lay.nv:
            raise ValueError(f"{what}: variant_layout nv {meta['nv']} != {lay.nv}")
    return lay
