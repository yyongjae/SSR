#!/usr/bin/env python
"""CK2 e2e evaluation over the 16 x V candidate pool with official labels (SPEC ck2e2e s5-2 / s5-3 'eval').

  cd /workspace/yongjae/SSR-ck2 && PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /venv/ssr/bin/python tools/ck/e2e2/eval_ck2.py --root <RUN>/eval/root --run <RUN name> --split navtrain_val
    [--fix-from <val metrics.json>] [--bootstrap 2000] [--calib-offset none|<json>] [--sel-w default|noim|plugin]
    [--out DIR] [--tokens FILE [--tokens-allow-missing]]

Inputs (eval root written by eval_e2e2.py dump / label; G = 16 x nv pool columns, 96 for the base variant set):
  packed/<split>/{tokens.parquet, valid96 [N,G], v2_final [N,16], v2_im [N,16], ok [N], meta.json}
  infer/<run>/<split>/{score_logit [N,G,5], done [N]}
  labels/<split>/pool96_<run>/{labels [N,G,9], ok [N,G]}     official labels of every pool column (c = k*nv + v)
  labels/<split>/lat96_<run>/{labels, ok}                     the same columns with the learned lateral applied
Variant layout (variants.VariantLayout): packed/<split>/meta.json 'variant_layout' (no key = base, a dump written before
  layouts were recorded), cross-checked with infer/<run>/<split>/meta.json and <root>/../ck_run/config.json
  ck_e2e2.variants; arrays whose column count != 16 x nv are refused (the three 80-column arms v4 / spd4 / lat4 are told
  apart by the recorded layout, never by the shape).  Variant-set names (none / decel / speed / lat / all_noaccel / all)
  resolve by meaning through layout.sets; metrics.json records 'variant_layout' (+ the resolved 'variant_sets' for a
  non-base layout).
Token set (NU43): packed ok & inference done & every column the selection can submit is labelled ok (identity columns
  and valid variant columns, pool and lateral) -- every row below on the same tokens.  The rule depends on the pool
  (more columns, more chances to miss a label), so arms differ in n_eval: --tokens FILE (one token per line, a JSON list
  / {"tokens": [...]}, or a parquet with a 'token' column, e.g. the intersection of the arms' per_token.parquet)
  restricts the evaluation to those tokens; every listed token must be in the split and evaluable here
  (--tokens-allow-missing: evaluate the evaluable subset), the out dir gets __tok_<sha8 of the sorted list> and
  metrics.json 'tokens' records the file / sha16 / counts.  n_eval is always reported.
Rows
  v2            column 0 = v2's submitted trajectory (pool label)                         [reference of every delta]
  v2_lat        column 0 with the learned lateral applied (descriptive: lateral head alone)
  oracle16 / oracle96 / oracle96_lat   label-best of the 16 identities / the valid pool / pool U lateral (upper bounds)
  sel           select2.select96: score_c = (1 - beta) v2_final[c // nv] + beta ck_final(p_c, im[c // nv]) over the
                allowed columns of the variant set, then final trajectory by lat_mode (on = laterally corrected,
                on_except_latvar, off); beta x variant-set names x LAT_MODES (5 x 6 x 3 = 90) on navtrain_val (every
                layout; a set that resolves to the same type ids as another, e.g. spd4 'lat' = 'none', gives an
                identical row and ties are broken by SET_PREF).
Choice (navtrain_val only): the 'sel' row with the highest mean PDMS among lat_mode 'on' rows (--choose-lat on, default:
  the user's "lateral applied after selection"; the other lateral modes are descriptive) -- or among all rows with
  --choose-lat any; ties -> set order all, all_noaccel, speed, decel, lat, none, then larger beta, then lat_mode on,
  on_except_latvar, off (NU28 / NU46).  navtest: only with --fix-from
  (the val metrics.json): v2, v2_lat, oracles, the val-chosen row (representative) and beta 1 / all / on (descriptive).
Metrics (eval_ck.row_metrics): PDMS, NC / DAC / EP / TTC / C / DDC means, fail rates, lead-decel NC|TTC failure (n),
  Delta vs v2 with the paired log-cluster bootstrap 95% CI (eval_ck.log_bootstrap), changed fraction, lateral-applied
  fraction, selection share per variant type.  Calibration offset OFF by default (user).
Selection weights --sel-w (select.SEL_W_SETS): 'plugin' = (0, 1, 1, 1) (official PDMS without imitation, default);
  'default' = SEL_W (0.1, 0.5, 0.5, 1.0) / 'noim' are optional comparison sets.  A non-default set writes to
  <split>__selw_<name> (default out dir) and is recorded in metrics.json ('sel_w', 'sel_w_values').
Frozen eval spec (report 48 F1-c / F3): the navtrain_val run stores ONE spec in metrics.json['eval_spec'] (+ eval_spec.json):
  the chosen beta / variant set (name + type ids) / lateral mode, choose_lat, the selection weights (name + 4 values),
  the calibration offset VALUES applied (+ source path and sha16) and the model it was chosen on (run, ckpt path +
  sha16, effective ep_target of the extracted CK run, code / hydra sha16 and --limit of the dump; read from the eval root's
  infer / packed meta.json and <root>/../ck_run/config.json), with spec_sha16; a non-base variant layout and a --tokens
  restriction are part of the spec (keys 'variant_layout' / 'tokens', absent for the base layout / all tokens).
  navtest --fix-from restores it: an unset --sel-w / --calib-offset inherits the val values, an explicitly given value
  must equal them; the run, the checkpoint sha16 and the ep_target of the navtest dump must equal the spec's, a --limit
  val choice cannot fix a full navtest, and the navtest dump must have the spec's variant layout -- every mismatch is
  a SystemExit.  A calibrated run writes <split>[__selw_<name>]__cal_<sha8 of the offset values>, so an offset
  ablation never overwrites the uncalibrated val result.
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)
import navsim  # noqa: E402

assert navsim.__file__.startswith(CK), navsim.__file__

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from navsim.agents.para_ssr.ck import select2 as S2  # noqa: E402
from navsim.agents.para_ssr.ck.variants import BASE_LAYOUT, VariantLayout  # noqa: E402
from tools.ck import ckutil as U  # noqa: E402
from tools.ck.eval_ck import log_bootstrap, row_metrics  # noqa: E402

CK_DATA = Path("/home/external-user/ssd/yongjae_refiner/ck")
SET_PREF = ("all", "all_noaccel", "speed", "decel", "lat", "none")            # tie-break order (NU46)
LAT_PREF = ("on", "on_except_latvar", "off")
DEFAULT_SEL_W = "plugin"                                                      # official PDMS (0, 1, 1, 1), no imitation
KST = 9 * 3600


def kst(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime((time.time() if t is None else t) + KST))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


# ----------------------------------------------------------------------------------------------- core (arrays)
def combos_grid(betas: Sequence[float] = Cn.BETA_GRID, sets: Sequence[str] = tuple(S2.VARIANT_SETS),
                lat_modes: Sequence[str] = S2.LAT_MODES) -> List[Tuple[float, str, str]]:
    return [(float(b), s, m) for b in betas for s in sets for m in lat_modes]


def token_mask(packed_ok, done, valid96, ok_pool, ok_lat, *, layout: VariantLayout = BASE_LAYOUT) -> np.ndarray:
    """packed ok & done & every submittable column (identity or valid variant) labelled ok in pool and lateral."""
    v = np.asarray(valid96, bool).copy()
    S2.check_cols("token_mask: valid96", v.shape[1], layout)
    v[:, ::layout.nv] = True
    need_pool = (np.asarray(ok_pool, bool) | ~v).all(1)
    need_lat = (np.asarray(ok_lat, bool) | ~v).all(1)
    return np.asarray(packed_ok, bool) & np.asarray(done, bool) & need_pool & need_lat


def _key(r: Dict[str, Any]) -> str:
    if r["variant"] != "sel":
        return r["variant"]
    return f"sel b={r['beta']:g} set={r['set']} lat={r['lat_mode']}"


def evaluate_pool(lab_pool: np.ndarray, lab_lat: np.ndarray, prob96: np.ndarray, valid96: np.ndarray,
                  v2_final16: np.ndarray, v2_im16: np.ndarray, logs: np.ndarray, lead_mask: Optional[np.ndarray],
                  combos: Sequence[Tuple[float, str, str]], n_boot: int = 2000, seed: int = 0,
                  offset=None, extra_rows: bool = True, sel_w=DEFAULT_SEL_W, *,
                  layout: VariantLayout = BASE_LAYOUT) -> Dict[str, Any]:
    """Arrays already restricted to the evaluation tokens.  lab_* [n, G, 9] (LABEL_COLS), prob96 [n, G, 5],
    valid96 [n, G], v2_final16 / v2_im16 [n, 16] (G = 16 x layout.nv); sel_w = select2.select96 weights (SEL_W_SETS
    name or 4 floats, default 'plugin' = official PDMS without imitation); combos name the variant sets (resolved
    through layout.sets).
    -> {'rows': [...], 'idx': {key: [n]}}."""
    w_sel = S2.resolve_w(DEFAULT_SEL_W if sel_w is None else sel_w)
    nv = layout.nv
    for what, arr in (("lab_pool", lab_pool), ("lab_lat", lab_lat), ("prob96", prob96), ("valid96", valid96)):
        S2.check_cols(f"evaluate_pool: {what}", np.shape(arr)[1], layout)
    L = Cn.LBL
    P = L["pdms"]
    n = lab_pool.shape[0]
    ar = np.arange(n)
    base = lab_pool[:, 0]
    fb = ((base[:, L["nc"]] < 1) | (base[:, L["ttc"]] < 1)).astype(np.float64)
    rows: List[Dict[str, Any]] = []
    idx_of: Dict[str, np.ndarray] = {}

    def add(row: Dict[str, Any], idx: np.ndarray, chosen: np.ndarray, lat: np.ndarray):
        m = row_metrics(chosen, lead_mask)
        m.update(row)
        m["frac_changed"] = float(np.mean((idx != 0) | lat)) if n else float("nan")
        m["frac_lat"] = float(np.mean(lat)) if n else float("nan")
        m["sel_share"] = {layout.names[t]: float(np.mean(idx % nv == t)) if n else float("nan")
                          for t in range(nv)}
        if row["variant"] != "v2":
            m["d_pdms"] = log_bootstrap(chosen[:, P] - base[:, P], logs, n_boot, seed)
            if lead_mask is not None and lead_mask.any():
                fv = ((chosen[:, L["nc"]] < 1) | (chosen[:, L["ttc"]] < 1)).astype(np.float64)
                m["d_lead_fail"] = log_bootstrap((fv - fb)[lead_mask], logs[lead_mask], n_boot, seed)
        m["key"] = _key(m)
        rows.append(m)
        idx_of[m["key"]] = idx

    zero, f0, t0 = np.zeros(n, np.int64), np.zeros(n, bool), np.ones(n, bool)
    add({"variant": "v2", "beta": None, "set": None, "lat_mode": None}, zero, base, f0)
    if extra_rows:
        add({"variant": "v2_lat", "beta": None, "set": None, "lat_mode": "on"}, zero, lab_lat[:, 0], t0)
        allow = S2.allowed_mask(valid96, "all", layout=layout)
        o16 = np.nan_to_num(lab_pool[:, ::nv, P], nan=-1.0).argmax(1) * nv
        add({"variant": "oracle16", "beta": None, "set": "none", "lat_mode": "off"}, o16, lab_pool[ar, o16], f0)
        sp = np.where(allow, np.nan_to_num(lab_pool[..., P], nan=-1.0), -np.inf)
        o96 = sp.argmax(1)
        add({"variant": "oracle96", "beta": None, "set": "all", "lat_mode": "off"}, o96, lab_pool[ar, o96], f0)
        sl = np.where(allow, np.nan_to_num(lab_lat[..., P], nan=-1.0), -np.inf)
        both = np.concatenate([sp, sl], 1)
        ob = both.argmax(1)
        G = lab_pool.shape[1]
        lat_b = ob >= G
        ib = ob % G
        chosen = np.where(lat_b[:, None], lab_lat[ar, ib], lab_pool[ar, ib])
        add({"variant": "oracle96_lat", "beta": None, "set": "all", "lat_mode": "best"}, ib, chosen, lat_b)
        p_cal = S2.apply_offset(prob96, offset)
        s_pdms = S2.exp_pdms96(p_cal)
        s_pdms = np.where(allow & np.isfinite(s_pdms), s_pdms, -np.inf)
        tie = S2.tie_order(layout)
        idx_pdms = tie[np.argmax(s_pdms[:, tie], axis=1)].astype(np.int64)
        add({"variant": "pdms", "beta": 1.0, "set": "all", "lat_mode": "on"}, idx_pdms,
            S2.chosen_labels(lab_pool, lab_lat, idx_pdms, "on", layout=layout), t0)
    cache: Dict[Tuple[float, str], np.ndarray] = {}
    for beta, sname, lat_mode in combos:
        k = (float(beta), sname)
        if k not in cache:
            cache[k] = S2.select96(v2_final16, v2_im16, prob96, valid96, beta, S2.set_types(sname, layout=layout),
                                   offset, w=w_sel, layout=layout)
        idx = cache[k]
        lat = S2.use_lat(idx, lat_mode, layout=layout)
        chosen = S2.chosen_labels(lab_pool, lab_lat, idx, lat_mode, layout=layout)
        add({"variant": "sel", "beta": float(beta), "set": sname, "lat_mode": lat_mode}, idx, chosen, lat)
    return {"rows": rows, "idx": idx_of}


CHOOSE_LAT = {"on": ("on",), "any": LAT_PREF}


def choose_best(rows: Sequence[Dict[str, Any]], choose_lat: str = "on") -> Optional[Dict[str, Any]]:
    """max mean PDMS over the 'sel' rows whose lat_mode is allowed by choose_lat; ties -> SET_PREF order, larger beta,
    LAT_PREF order.  choose_lat 'on' (default) = the user's decision (the learned lateral is applied AFTER selection to
    the selected trajectory): only beta and the variant set are chosen on navtrain_val, the other lateral modes stay
    descriptive rows; 'any' also lets navtrain_val choose the lateral mode (on / on_except_latvar / off)."""
    if choose_lat not in CHOOSE_LAT:
        raise ValueError(f"choose_lat must be one of {tuple(CHOOSE_LAT)}, got {choose_lat!r}")
    lat_ok = CHOOSE_LAT[choose_lat]
    sel = [r for r in rows if r["variant"] == "sel" and r["lat_mode"] in lat_ok and r.get("pdms") is not None
           and np.isfinite(r["pdms"])]
    if not sel:
        return None
    best = max(sel, key=lambda r: (r["pdms"], -SET_PREF.index(r["set"]), r["beta"], -LAT_PREF.index(r["lat_mode"])))
    lat_rule = ("lateral fixed 'on' (user: applied after selection)" if choose_lat == "on" else
                "lateral chosen too, ties on > on_except_latvar > off")
    return {"beta": best["beta"], "set": best["set"], "lat_mode": best["lat_mode"], "pdms": best["pdms"],
            "key": best["key"], "choose_lat": choose_lat,
            "rule": f"max mean PDMS over beta x variant set on navtrain_val, {lat_rule}; ties -> set all > all_noaccel "
                    f"> speed > decel > lat > none, larger beta"}


# ----------------------------------------------------------------------------------------------- io
def load_labels(d: Path, n: int, g: int = S2.G_K) -> Tuple[np.ndarray, np.ndarray]:
    """labels [n, g, 9] / ok [n, g] of a label dir (g = 16 x nv of the dump's variant layout)."""
    lab = np.load(d / "labels.npy", mmap_mode="r")
    ok = np.load(d / "ok.npy", mmap_mode="r")
    if lab.shape[0] != n or lab.shape[1] != g:
        raise SystemExit(f"{d}: labels {lab.shape} != [{n}, {g}, 9]")
    if tuple(ok.shape) != (n, g):
        raise SystemExit(f"{d}: ok {ok.shape} != [{n}, {g}]")
    return np.asarray(lab, np.float64), np.asarray(ok, bool)


# ----------------------------------------------------------------------------------------------- variant layout
def layout_of_cfg(vcfg) -> VariantLayout:
    """layout of a ck_e2e2.variants dict (training / hydra config; None or missing keys = base)."""
    try:
        return S2.layout_from_cfg(vcfg)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def layout_of_root(root, run: str, split: str) -> VariantLayout:
    """the variant layout of an eval root split: packed/<split>/meta.json 'variant_layout' (no key = base: a dump
    written before layouts were recorded); infer/<run>/<split>/meta.json and <root>/../ck_run/config.json
    ck_e2e2.variants must agree when they record one (SystemExit otherwise)."""
    root = Path(root)
    srcs = [("packed", root / "packed" / split / "meta.json"), ("infer", root / "infer" / run / split / "meta.json")]
    lays: Dict[str, VariantLayout] = {}
    for what, p in srcs:
        m = U.read_json(p) or {}
        if what == "packed" or "variant_layout" in m:
            try:
                lays[what] = S2.layout_from_meta(m.get("variant_layout"), str(p))
            except ValueError as e:
                raise SystemExit(str(e)) from None
    ck = U.read_json(root.parent / "ck_run" / "config.json") or {}
    cke = ck.get("ck_e2e2")
    if isinstance(cke, dict) and cke.get("variants") is not None:
        lays["ck_run"] = layout_of_cfg(cke["variants"])
    lay = lays["packed"]
    bad = {k: list(v.names) for k, v in lays.items() if v != lay}
    if bad:
        raise SystemExit(f"{root} / {split}: variant layouts disagree: packed {list(lay.names)} vs {bad} (the eval "
                         f"root "
                         f"mixes dumps / configs of different variant sets); rerun eval_e2e2 with --fresh")
    return lay


def layout_info(layout: VariantLayout) -> Dict[str, Any]:
    """metrics.json fields of the variant layout: 'variant_layout' (VariantLayout.meta()), + the resolved sets
    'variant_sets' for a non-base layout only (the base sets are select2.VARIANT_SETS; base metrics.json = the v2_final
    keys + 'variant_layout')."""
    d: Dict[str, Any] = {"variant_layout": layout.meta()}
    if not layout.is_base:
        d["variant_sets"] = {k: list(v) for k, v in layout.sets.items()}
    return d


def per_token_cols(idx, layout: VariantLayout) -> Dict[str, np.ndarray]:
    """sel_k (parent rank) / sel_v (type id) / sel_name of pool columns (per_token.parquet; select2.decode_cols)."""
    k, v, nm = S2.decode_cols(idx, layout=layout)
    return {"sel_k": k.astype(np.int64), "sel_v": v.astype(np.int64), "sel_name": nm.astype(str)}


# ----------------------------------------------------------------------------------------------- token restriction
def read_tokens(spec) -> List[str]:
    """--tokens FILE: one token per line ('#' comments), a JSON list / {'tokens': [...]}, or a parquet 'token'
    column."""
    p = Path(spec)
    if not p.is_file():
        raise SystemExit(f"--tokens {spec}: no such file")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
        if "token" not in df.columns:
            raise SystemExit(f"--tokens {spec}: parquet without a 'token' column")
        toks = df.token.astype(str).tolist()
    elif p.suffix == ".json":
        d = json.loads(p.read_text())
        toks = d.get("tokens") if isinstance(d, dict) else d
        if not isinstance(toks, list):
            raise SystemExit(f"--tokens {spec}: JSON must be a list or {{'tokens': [...]}}")
        toks = [str(t) for t in toks]
    else:
        toks = [s.strip() for s in p.read_text().splitlines() if s.strip() and not s.strip().startswith("#")]
    toks = sorted(set(toks))
    if not toks:
        raise SystemExit(f"--tokens {spec}: no tokens")
    return toks


def tokens_sha16(toks: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(str(t) for t in toks))).encode()).hexdigest()[:16]


def tokens_tag(toks: Optional[Sequence[str]]) -> str:
    """'' without a restriction; else '__tok_<sha8 of the sorted token list>' (output dir suffix)."""
    return "" if not toks else "__tok_" + tokens_sha16(toks)[:8]


def lead_mask_of(split: str, tokens: Sequence[str], lead_root: Path = CK_DATA / "lead"):
    p = lead_root / f"{split}.parquet"
    if not p.is_file():
        return None, None
    df = pd.read_parquet(p).drop_duplicates("token").set_index("token")
    t = pd.Index(tokens)
    has = df["has_lead"].reindex(t).to_numpy(np.float64)
    d1 = df["D_1"].reindex(t).to_numpy(np.float64)
    m = (has == 1) & (d1 == 1)
    if "censored" in df.columns:
        m &= ~df["censored"].reindex(t).fillna(True).astype(bool).to_numpy()
    return m, str(p)


def read_offset(spec: str):
    if not spec or spec.lower() == "none":
        return None
    d = json.loads(Path(spec).read_text())
    off = d.get("offset") if isinstance(d, dict) else d
    off = [float(x) for x in off]
    if len(off) != len(Cn.CK_KEYS):
        raise SystemExit(f"--calib-offset {spec}: need {len(Cn.CK_KEYS)} values (CK_KEYS order)")
    return off


def offset_source(spec: str) -> Optional[Dict[str, Any]]:
    """{'path', 'sha16'} of a --calib-offset file (None when OFF)."""
    if not spec or spec.lower() == "none":
        return None
    return {"path": str(Path(spec).resolve()), "sha16": U.sha256_file(spec)}


def offset_tag(offset) -> str:
    """'' when OFF; else '__cal_<sha8 of the canonical offset values>' (output dir suffix of a calibrated run)."""
    if offset is None:
        return ""
    vals = json.dumps([float(x) for x in offset])
    return "__cal_" + hashlib.sha256(vals.encode()).hexdigest()[:8]


def same_offset(a, b) -> bool:
    """None-aware exact equality of two offset vectors."""
    if a is None or b is None:
        return a is None and b is None
    return [float(x) for x in a] == [float(x) for x in b]


def model_meta(root: Path, run: str, split: str) -> Dict[str, Any]:
    """The model behind an eval root split: ckpt / ckpt_sha16 / code_sha16 / hydra_sha16 / limit of the dump
    (infer/<run>/<split>/meta.json, cross-checked with packed/<split>/meta.json) and ep_target of the extracted CK run
    (<root>/../ck_run/config.json ck_e2e2.ep_target; None when not recorded).  Missing files -> None fields."""
    inf = U.read_json(root / "infer" / run / split / "meta.json") or {}
    pk = U.read_json(root / "packed" / split / "meta.json") or {}
    for k in ("ckpt_sha16", "code_sha16", "hydra_sha16"):
        if inf.get(k) is not None and pk.get(k) is not None and inf[k] != pk[k]:
            raise SystemExit(f"{root}: packed/{split} and infer/{run}/{split} meta.json disagree on {k} "
                             f"({pk[k]} vs {inf[k]}): the eval root mixes two dumps; rerun eval_e2e2 with --fresh")
    ck = U.read_json(root.parent / "ck_run" / "config.json") or {}
    ept = None
    if "ck_e2e2" in ck:               # the effective training value (a config without the key = the CKE2E2 default)
        from navsim.agents.para_ssr.ck.e2e_data2 import ep_target_of
        ept = ep_target_of(ck.get("ck_e2e2") or {})
    return {"ckpt": inf.get("ckpt", pk.get("ckpt")), "ckpt_sha16": inf.get("ckpt_sha16", pk.get("ckpt_sha16")),
            "code_sha16": inf.get("code_sha16", pk.get("code_sha16")),
            "hydra_sha16": inf.get("hydra_sha16", pk.get("hydra_sha16")),
            "limit": inf.get("limit", pk.get("limit")), "ep_target": ept,
            "ck_run_source_sha16": ck.get("source_ckpt_sha16")}


def spec_sha16(spec: Dict[str, Any]) -> str:
    body = {k: v for k, v in spec.items() if k not in ("spec_sha16", "created")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


def frozen_spec(src: Dict[str, Any], fix_from: str) -> Dict[str, Any]:
    """the val eval_spec of a --fix-from metrics.json (SystemExit when absent / without a choice / tampered)."""
    spec = src.get("eval_spec")
    if not spec:
        raise SystemExit(f"{fix_from}: no frozen 'eval_spec' (a val result written before report 48's fix); rerun the "
                         f"navtrain_val eval")
    if spec.get("split") != "navtrain_val":
        raise SystemExit(f"{fix_from}: eval_spec of split {spec.get('split')!r}, need a navtrain_val choice")
    if not spec.get("best"):
        raise SystemExit(f"{fix_from}: no 'best' choice")
    if spec.get("spec_sha16") != spec_sha16(spec):
        raise SystemExit(f"{fix_from}: eval_spec does not match its spec_sha16 (edited after the val run)")
    return spec


def resolve_navtest(spec: Dict[str, Any], run: str, sel_w=None, calib_offset=None, cur: Optional[Dict] = None, *,
                    layout: VariantLayout = BASE_LAYOUT):
    """navtest restores the frozen val spec: -> (weights, offset).  sel_w / calib_offset None or '' = inherit; a given
    value must equal the spec's (SystemExit otherwise).  cur: model_meta of the navtest split (run, ckpt sha16,
    ep_target and limit must match the spec); layout: the navtest dump's variant layout (must equal the spec's, which
    is base when the spec has no 'variant_layout')."""
    best = spec["best"]
    if spec.get("run") != run:
        raise SystemExit(f"--fix-from: the val choice was made for run {spec.get('run')!r}, this is {run!r} (every run "
                         f"chooses on its own navtrain_val)")
    spec_w = tuple(float(x) for x in spec["sel_w_values"])
    if sel_w is None or sel_w == "":
        w_sel = spec_w
    else:
        w_sel = S2.resolve_w(sel_w)
        if tuple(float(x) for x in w_sel) != spec_w:
            raise SystemExit(f"--fix-from: chosen with selection weights {spec.get('sel_w')!r} {list(spec_w)}, this run "
                             f"uses {S2.sel_w_name(w_sel)!r} {list(w_sel)} (--sel-w)")
    spec_off = spec.get("calib_offset_values")
    if calib_offset is None or calib_offset == "":
        offset = None if spec_off is None else [float(x) for x in spec_off]
    else:
        offset = read_offset(calib_offset)
        if not same_offset(offset, spec_off):
            raise SystemExit(f"--fix-from: navtest calibration offset {offset} differs from the val choice {spec_off} "
                             f"(--calib-offset; omit it to inherit the val values)")
    try:
        spec_lay = S2.layout_from_meta(spec.get("variant_layout"), "--fix-from eval_spec")
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if spec_lay != layout:
        raise SystemExit(f"--fix-from: chosen on val with the variant layout {list(spec_lay.names)}, this navtest dump "
                         f"has {list(layout.names)}")
    if tuple(layout.sets.get(best["set"], ())) != tuple(spec.get("set_types") or ()):
        raise SystemExit(f"--fix-from: variant set {best['set']!r} is {layout.sets.get(best['set'])} now, "
                         f"{spec.get('set_types')} when chosen on val")
    cur = cur or {}
    m = spec.get("model") or {}
    if cur.get("ckpt_sha16") != m.get("ckpt_sha16"):
        raise SystemExit(f"--fix-from: val chose on ckpt {m.get('ckpt')} (sha16 {m.get('ckpt_sha16')}), this navtest "
                         f"dump is from {cur.get('ckpt')} (sha16 {cur.get('ckpt_sha16')}); evaluate both splits on "
                         f"the same checkpoint (eval_e2e2 --ckpt / --fresh)")
    if cur.get("ep_target") != m.get("ep_target"):
        raise SystemExit(f"--fix-from: ep_target {m.get('ep_target')!r} on val vs {cur.get('ep_target')!r} here")
    if int(m.get("limit") or 0) and not int(cur.get("limit") or 0):
        raise SystemExit(f"--fix-from: the val choice used --limit {m.get('limit')} (smoke subset); a full navtest "
                         f"needs a full navtrain_val choice")
    return w_sel, offset


def fmt_ci(d) -> str:
    if not isinstance(d, dict) or not d.get("n"):
        return ""
    return f"{100 * d['mean']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}]"


def table_md(res: Dict[str, Any]) -> str:
    L = [f"# CK2 e2e eval: {res['run']} / {res['split']}", "",
         f"토큰 {res['n_eval']} / 전체 {res['n_total']} (packed ok, 추론 완료, 제출 가능한 열(식별열 + 유효 변형) 라벨 모두 ok, "
         f"pool·측방 둘 다). log {res['n_logs']}. bootstrap {res['n_boot']}회(log 군집). PDMS 등은 0-100 단위. "
         f"calibration offset {res['calib_offset']} {res.get('calib_offset_values') or ''}. 선택 가중치 "
         f"{res.get('sel_w', 'default')} {tuple(res.get('sel_w_values') or ())}. ckpt sha16 "
         f"{(res.get('model') or {}).get('ckpt_sha16')}, ep_target {(res.get('model') or {}).get('ep_target')}, "
         f"eval spec {(res.get('eval_spec') or {}).get('spec_sha16')}. 생성 {res['created']}.", ""]
    if res.get("best"):
        L += [f"navtrain_val 선택: β={res['best']['beta']:g}, 변형 집합 {res['best']['set']}, 측방 {res['best']['lat_mode']} "
              f"(PDMS {100 * res['best']['pdms']:.2f})", ""]
    if res.get("fixed_from_val"):
        f = res["fixed_from_val"]
        L += [f"val에서 고정: β={f['beta']:g}, {f['set']}, 측방 {f['lat_mode']} (출처 {res['fix_from']})", ""]
    vl = res.get("variant_layout")
    if vl and not S2.layout_from_meta(vl).is_base:
        L += [f"변형 배치(variant layout): nv {vl['nv']}, {', '.join(vl['names'])} (열 c = k*{vl['nv']} + v); 집합 "
              + "; ".join(f"{k}={tuple(v)}" for k, v in (res.get("variant_sets") or {}).items()), ""]
    if res.get("tokens"):
        t = res["tokens"]
        L += [f"토큰 제한 --tokens {t['path']} (sha16 {t['sha16']}): 목록 {t['n_listed']}, 평가 가능 {t['n_evaluable']}, "
              f"제외 {t['n_not_evaluable']}", ""]
    L += ["| 행 | PDMS | ΔPDMS [95% CI] | NC | DAC | EP | TTC | C | NC 실패 | DAC 실패 | TTC 실패 | 앞차 감속 NC/TTC 실패 (n) | "
          "Δ앞차 실패 [95% CI] | 바뀐 비율 | 측방 적용 | 변형 선택 비율 |", "|" + "---|" * 16]
    for r in res["rows"]:
        lead = f"{100 * r['lead_fail_nc_ttc']:.2f} ({r['lead_n']})" if r.get("lead_n") else "-"
        share = " ".join(f"{k}:{100 * v:.0f}" for k, v in r["sel_share"].items() if k != "id" and v > 0)
        mark = "★ " if res.get("representative") == r["key"] else ""
        L.append(f"| {mark}{r['key']} | {100 * r['pdms']:.2f} | {fmt_ci(r.get('d_pdms'))} | {100 * r['nc']:.2f} | "
                 f"{100 * r['dac']:.2f} | {100 * r['ep']:.2f} | {100 * r['ttc']:.2f} | {100 * r['comfort']:.2f} | "
                 f"{100 * r['fail_nc']:.2f} | {100 * r['fail_dac']:.2f} | {100 * r['fail_ttc']:.2f} | {lead} | "
                 f"{fmt_ci(r.get('d_lead_fail'))} | {100 * r['frac_changed']:.1f} | {100 * r['frac_lat']:.1f} | "
                 f"{share or '-'} |")
    return "\n".join(L) + "\n"


def out_name(split: str, sel_w=DEFAULT_SEL_W, offset=None, tokens: Optional[Sequence[str]] = None) -> str:
    """default out dir name: <split> for the default weights ('plugin'), <split>__selw_<name> otherwise, + __cal_<sha8>
    when a calibration offset is applied (offset_tag), + __tok_<sha8> with a --tokens restriction (tokens_tag)."""
    n = S2.sel_w_name(DEFAULT_SEL_W if sel_w is None else sel_w)
    base = split if n == DEFAULT_SEL_W else f"{split}__selw_{n.replace(',', '_')}"
    return base + offset_tag(offset) + tokens_tag(tokens)


def run_eval(root, run: str, split: str, out=None, fix_from: str = "", n_boot: int = 2000,
             calib_offset: Optional[str] = None, lead_root: Path = CK_DATA / "lead", choose_lat: str = "on",
             sel_w=None, tokens: Optional[str] = None, tokens_allow_missing: bool = False) -> Dict[str, Any]:
    """sel_w / calib_offset None (or ''): navtrain_val = 'plugin' (official PDMS without imitation) / OFF; navtest =
    inherited from the frozen val spec (--fix-from), an explicit value must equal it.  tokens: --tokens FILE (None /
    '' = every evaluable token); the variant layout comes from the eval root (layout_of_root)."""
    root = Path(root)
    cur = model_meta(root, run, split)
    layout = layout_of_root(root, run, split)
    tok_list = read_tokens(tokens) if tokens else None
    fixed = spec_in = None
    if split == "navtest":
        if not fix_from:
            raise SystemExit("navtest needs --fix-from <navtrain_val metrics.json> (beta / variant set / lateral mode "
                             "are chosen on navtrain_val only)")
        src = json.loads(Path(fix_from).read_text())
        spec_in = frozen_spec(src, fix_from)
        w_sel, offset = resolve_navtest(spec_in, run, sel_w, calib_offset, cur, layout=layout)
        fixed = dict(spec_in["best"])
        cal_src = spec_in.get("calib_offset_src") if (calib_offset in (None, "")) else offset_source(calib_offset)
        cal_label = ("none (OFF)" if offset is None else
                     (f"inherited from val: {(cal_src or {}).get('path')}" if calib_offset in (None, "")
                      else calib_offset))
    else:
        w_sel = S2.resolve_w(sel_w or DEFAULT_SEL_W)
        cal = calib_offset or "none"
        offset = read_offset(cal)
        cal_src = offset_source(cal)
        cal_label = cal if offset is not None else "none (OFF)"
    w_name = S2.sel_w_name(w_sel)
    out = Path(out) if out else root / "eval" / run / out_name(split, w_sel, offset, tok_list)
    pk = root / "packed" / split
    inf = root / "infer" / run / split
    lb = root / "labels" / split
    tdf = pd.read_parquet(pk / "tokens.parquet")
    n = len(tdf)
    valid96 = np.load(pk / "valid96.npy")
    G = layout.g(S2.K16)
    if valid96.ndim != 2 or valid96.shape[1] != G:
        raise SystemExit(f"{pk}/valid96.npy {valid96.shape}: the eval root's variant layout {list(layout.names)} needs "
                         f"[{n}, 16 x {layout.nv} = {G}] (a dump without a recorded 'variant_layout' is read as base)")
    v2f = np.load(pk / "v2_final.npy").astype(np.float64)
    v2im = np.load(pk / "v2_im.npy").astype(np.float64)
    pok = np.load(pk / "ok.npy")
    logit = np.load(inf / "score_logit.npy").astype(np.float64)
    if logit.shape[:2] != (n, G):
        raise SystemExit(f"{inf}/score_logit.npy {logit.shape} != [{n}, {G}, 5] (variant layout {list(layout.names)})")
    done = np.load(inf / "done.npy")
    lab_p, ok_p = load_labels(lb / f"pool96_{run}", n, G)
    lab_l, ok_l = load_labels(lb / f"lat96_{run}", n, G)
    m = token_mask(pok, done, valid96, ok_p, ok_l, layout=layout)
    tok_info = None
    if tok_list is not None:
        split_toks = tdf.token.astype(str).to_numpy()
        want = set(tok_list)
        listed = np.fromiter((t in want for t in split_toks), bool, len(split_toks))
        unknown = sorted(set(tok_list) - set(split_toks.tolist()))
        if unknown:
            raise SystemExit(f"--tokens {tokens}: {len(unknown)} of {len(tok_list)} tokens are not in {split} "
                             f"(e.g. {unknown[:3]})")
        not_eval = int((listed & ~m).sum())
        if not_eval and not tokens_allow_missing:
            ex = split_toks[listed & ~m][:3].tolist()
            raise SystemExit(f"--tokens {tokens}: {not_eval} of {len(tok_list)} listed tokens are not evaluable in "
                             f"{run}/{split} (packed / done / labels not ok, e.g. {ex}); the comparison would not be "
                             f"on the common set (--tokens-allow-missing evaluates the evaluable subset)")
        m = m & listed
        tok_info = {"path": str(Path(tokens).resolve()), "sha16": tokens_sha16(tok_list), "n_listed": len(tok_list),
                    "n_evaluable": int(m.sum()), "n_not_evaluable": not_eval,
                    "allow_missing": bool(tokens_allow_missing)}
    sel = np.flatnonzero(m)
    toks = tdf.token.astype(str).to_numpy()[sel]
    logs = tdf.log.astype(str).to_numpy()[sel]
    lead, lead_src = lead_mask_of(split, toks, lead_root)
    if split == "navtest":
        combos = [(float(fixed["beta"]), fixed["set"], fixed["lat_mode"])]
        if (1.0, "all", "on") not in combos:
            combos.append((1.0, "all", "on"))
    else:
        combos = combos_grid()
        if fix_from:
            src = json.loads(Path(fix_from).read_text())
            fixed = src.get("best")
    res = evaluate_pool(lab_p[sel], lab_l[sel], sigmoid(logit[sel]), valid96[sel], v2f[sel], v2im[sel], logs, lead,
                        combos, n_boot, 0, offset, sel_w=w_sel, layout=layout)
    rows = res["rows"]
    out_d: Dict[str, Any] = {"run": run, "split": split, "root": str(root), "n_total": int(n), "n_eval": int(len(sel)),
                             "n_logs": int(len(set(logs))), "n_boot": int(n_boot), "lead_src": lead_src,
                             "calib_offset": cal_label,
                             "calib_offset_values": None if offset is None else [float(x) for x in offset],
                             "calib_offset_src": cal_src,
                             "sel_w": w_name, "sel_w_values": [float(x) for x in w_sel], "model": cur,
                             "created": kst(), "fix_from": fix_from or None, "rows": rows,
                             "token_rule": "packed ok & done & identity + valid variant columns labelled ok (pool, lat)",
                             "out_dir": str(out)}
    if split == "navtest":
        out_d["fixed_from_val"] = fixed
        out_d["eval_spec"] = spec_in                     # the frozen val spec this navtest was evaluated with
        out_d["representative"] = _key({"variant": "sel", **{k: fixed[k] for k in ("beta", "set", "lat_mode")}})
    else:
        out_d["best"] = choose_best(rows, choose_lat)
        out_d["choose_lat"] = choose_lat
        out_d["representative"] = (out_d["best"] or {}).get("key")
        if fixed:
            out_d["fixed_from_val"] = fixed
        b = out_d["best"]
        spec = {"split": split, "run": run, "best": b, "set_types": list(layout.sets[b["set"]]) if b else None,
                "choose_lat": choose_lat, "sel_w": w_name, "sel_w_values": [float(x) for x in w_sel],
                "calib_offset_values": out_d["calib_offset_values"], "calib_offset_src": cal_src, "model": cur,
                "n_eval": int(len(sel)), "n_total": int(n), "token_rule": out_d["token_rule"], "created": kst()}
        if not layout.is_base:                 # base: no key, so the base spec (and its sha16) stays as before
            spec["variant_layout"] = layout.meta()
        if tok_info is not None:
            spec["tokens"] = tok_info
        spec["spec_sha16"] = spec_sha16(spec)
        out_d["eval_spec"] = spec
        U.write_json(out / "eval_spec.json", spec)
    out_d.update(layout_info(layout))          # added keys (after the historical ones)
    if tok_info is not None:
        out_d["tokens"] = tok_info
    U.write_json(out / "metrics.json", out_d)
    (out / "table.md").write_text(table_md(out_d))
    rep = out_d.get("representative")
    pt = pd.DataFrame({"token": toks, "log": logs})
    P = Cn.LBL["pdms"]
    pt["pdms_v2"] = lab_p[sel][:, 0, P]
    if rep in res["idx"]:
        r = next(x for x in rows if x["key"] == rep)
        idx = res["idx"][rep]
        pt["sel_col"] = idx
        if not layout.is_base:                 # sel_col = k * nv + v of THIS layout (base: c // 6, c % 6 as before)
            for k, v in per_token_cols(idx, layout).items():
                pt[k] = v
        pt["sel_lat"] = S2.use_lat(idx, r["lat_mode"], layout=layout)
        pt["pdms_sel"] = S2.chosen_labels(lab_p[sel], lab_l[sel], idx, r["lat_mode"], layout=layout)[:, P]
    pt.to_parquet(out / "per_token.parquet", index=False)
    if layout.is_base and tok_info is None:     # the v2_final line
        print(f"[{kst()}] eval_ck2 {run}/{split}: {len(sel)}/{n} tokens, {len(rows)} rows -> {out}", flush=True)
    else:
        print(f"[{kst()}] eval_ck2 {run}/{split}: n_eval {len(sel)}/{n} tokens, {len(rows)} rows, variant layout nv "
              f"{layout.nv} -> {out}", flush=True)
    return out_d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True, choices=["navtrain_val", "navtest"])
    ap.add_argument("--out", default="")
    ap.add_argument("--fix-from", default="")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--calib-offset", default="",
                    help="'' (default): OFF on navtrain_val, inherited from the val spec on navtest | none | <json>")
    ap.add_argument("--choose-lat", default="on", choices=tuple(CHOOSE_LAT),
                    help="navtrain_val choice: 'on' (default, user: lateral applied after selection; beta / variant "
                         "set chosen) | 'any' (the lateral mode is chosen too)")
    ap.add_argument("--sel-w", default=None, choices=tuple(S2.SEL_W_SETS),
                    help="selection weight set (select.SEL_W_SETS): plugin (0, 1, 1, 1, official PDMS without imitation, "
                         "default) | default (= SEL_W) | noim; unset: 'plugin' on navtrain_val, inherited from the val "
                         "spec on navtest")
    ap.add_argument("--tokens", default="",
                    help="restrict the evaluation to these tokens (text: one per line | JSON list | parquet 'token' "
                         "column), e.g. the tokens every arm evaluates; out dir <split>[...]__tok_<sha8>")
    ap.add_argument("--tokens-allow-missing", action="store_true",
                    help="with --tokens: evaluate the evaluable subset instead of refusing listed tokens that are not "
                         "evaluable here")
    a = ap.parse_args(argv)
    run_eval(a.root, a.run, a.split, a.out or None, a.fix_from, a.bootstrap, a.calib_offset, choose_lat=a.choose_lat,
             sel_w=a.sel_w, tokens=a.tokens or None, tokens_allow_missing=a.tokens_allow_missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
