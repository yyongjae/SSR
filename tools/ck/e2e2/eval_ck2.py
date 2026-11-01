#!/usr/bin/env python
"""CK2 e2e evaluation over the 96-candidate pool with official labels (SPEC ck2e2e s5-2 / s5-3 'eval').

  cd /workspace/yongjae/SSR-ck2 && PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /venv/ssr/bin/python tools/ck/e2e2/eval_ck2.py --root <RUN>/eval/root --run <RUN name> --split navtrain_val
    [--fix-from <val metrics.json>] [--bootstrap 2000] [--calib-offset none|<json>] [--sel-w default|noim|plugin]
    [--out DIR]

Inputs (eval root written by eval_e2e2.py dump / label):
  packed/<split>/{tokens.parquet, valid96 [N,96], v2_final [N,16], v2_im [N,16], ok [N]}
  infer/<run>/<split>/{score_logit [N,96,5], done [N]}
  labels/<split>/pool96_<run>/{labels [N,96,9], ok [N,96]}   official labels of every pool column (c = k*6 + v)
  labels/<split>/lat96_<run>/{labels, ok}                     the same columns with the learned lateral applied
Token set (NU43): packed ok & inference done & every column the selection can submit is labelled ok (identity columns
  and valid variant columns, pool and lateral) -- every row below on the same tokens.
Rows
  v2            column 0 = v2's submitted trajectory (pool label)                         [reference of every delta]
  v2_lat        column 0 with the learned lateral applied (descriptive: lateral head alone)
  oracle16 / oracle96 / oracle96_lat   label-best of the 16 identities / the valid pool / pool U lateral (upper bounds)
  sel           select2.select96: score_c = (1 - beta) v2_final[c // 6] + beta ck_final(p_c, im[c // 6]) over the
                allowed columns of the variant set, then final trajectory by lat_mode (on = laterally corrected,
                on_except_latvar, off); beta x VARIANT_SETS x LAT_MODES (5 x 6 x 3 = 90) on navtrain_val.
Choice (navtrain_val only): the 'sel' row with the highest mean PDMS among lat_mode 'on' rows (--choose-lat on, default:
  the user's "lateral applied after selection"; the other lateral modes are descriptive) -- or among all rows with
  --choose-lat any; ties -> set order all, all_noaccel, speed, decel, lat, none, then larger beta, then lat_mode on,
  on_except_latvar, off (NU28 / NU46).  navtest: only with --fix-from
  (the val metrics.json): v2, v2_lat, oracles, the val-chosen row (representative) and beta 1 / all / on (descriptive).
Metrics (eval_ck.row_metrics): PDMS, NC / DAC / EP / TTC / C / DDC means, fail rates, lead-decel NC|TTC failure (n),
  Delta vs v2 with the paired log-cluster bootstrap 95% CI (eval_ck.log_bootstrap), changed fraction, lateral-applied
  fraction, selection share per variant type.  Calibration offset OFF by default (user).
Selection weights --sel-w (select.SEL_W_SETS): 'default' = SEL_W (0.1, 0.5, 0.5, 1.0) unchanged; 'plugin' = (0, 1, 1, 1)
  (w_im 0, NC 1, DAC 1, rest 1) / 'noim' are optional comparison sets (navtrain_val first).  A non-default set writes to
  <split>__selw_<name> (default out dir) and is recorded in metrics.json ('sel_w', 'sel_w_values').
Frozen eval spec (report 48 F1-c / F3): the navtrain_val run stores ONE spec in metrics.json['eval_spec'] (+ eval_spec.json):
  the chosen beta / variant set (name + type ids) / lateral mode, choose_lat, the selection weights (name + 4 values),
  the calibration offset VALUES applied (+ source path and sha16) and the model it was chosen on (run, ckpt path +
  sha16, effective ep_target of the extracted CK run, code / hydra sha16 and --limit of the dump; read from the eval root's
  infer / packed meta.json and <root>/../ck_run/config.json), with spec_sha16.  navtest --fix-from restores it: an
  unset --sel-w / --calib-offset inherits the val values, an explicitly given value must equal them; the run, the
  checkpoint sha16 and the ep_target of the navtest dump must equal the spec's, and a --limit val choice cannot fix a
  full navtest -- every mismatch is a SystemExit.  A calibrated run writes <split>[__selw_<name>]__cal_<sha8 of the
  offset values>, so an offset ablation never overwrites the uncalibrated val result.
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
from tools.ck import ckutil as U  # noqa: E402
from tools.ck.eval_ck import log_bootstrap, row_metrics  # noqa: E402

CK_DATA = Path("/home/external-user/ssd/yongjae_refiner/ck")
SET_PREF = ("all", "all_noaccel", "speed", "decel", "lat", "none")            # tie-break order (NU46)
LAT_PREF = ("on", "on_except_latvar", "off")
KST = 9 * 3600


def kst(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime((time.time() if t is None else t) + KST))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


# ----------------------------------------------------------------------------------------------- core (arrays)
def combos_grid(betas: Sequence[float] = Cn.BETA_GRID, sets: Sequence[str] = tuple(S2.VARIANT_SETS),
                lat_modes: Sequence[str] = S2.LAT_MODES) -> List[Tuple[float, str, str]]:
    return [(float(b), s, m) for b in betas for s in sets for m in lat_modes]


def token_mask(packed_ok, done, valid96, ok_pool, ok_lat) -> np.ndarray:
    """packed ok & done & every submittable column (identity or valid variant) labelled ok in pool and lateral."""
    v = np.asarray(valid96, bool).copy()
    v[:, ::S2.NV] = True
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
                  offset=None, extra_rows: bool = True, sel_w="default") -> Dict[str, Any]:
    """Arrays already restricted to the evaluation tokens.  lab_* [n, 96, 9] (LABEL_COLS), prob96 [n, 96, 5],
    valid96 [n, 96], v2_final16 / v2_im16 [n, 16]; sel_w = select2.select96 weights (SEL_W_SETS name or 4 floats).
    -> {'rows': [...], 'idx': {key: [n]}}."""
    w_sel = S2.resolve_w(sel_w)
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
        m["sel_share"] = {S2.VNAMES[t]: float(np.mean(idx % S2.NV == t)) if n else float("nan")
                          for t in range(S2.NV)}
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
        allow = S2.allowed_mask(valid96, S2.VARIANT_SETS["all"])
        o16 = np.nan_to_num(lab_pool[:, ::S2.NV, P], nan=-1.0).argmax(1) * S2.NV
        add({"variant": "oracle16", "beta": None, "set": "none", "lat_mode": "off"}, o16, lab_pool[ar, o16], f0)
        sp = np.where(allow, np.nan_to_num(lab_pool[..., P], nan=-1.0), -np.inf)
        o96 = sp.argmax(1)
        add({"variant": "oracle96", "beta": None, "set": "all", "lat_mode": "off"}, o96, lab_pool[ar, o96], f0)
        sl = np.where(allow, np.nan_to_num(lab_lat[..., P], nan=-1.0), -np.inf)
        both = np.concatenate([sp, sl], 1)
        ob = both.argmax(1)
        lat_b = ob >= S2.G_K
        ib = ob % S2.G_K
        chosen = np.where(lat_b[:, None], lab_lat[ar, ib], lab_pool[ar, ib])
        add({"variant": "oracle96_lat", "beta": None, "set": "all", "lat_mode": "best"}, ib, chosen, lat_b)
    cache: Dict[Tuple[float, str], np.ndarray] = {}
    for beta, sname, lat_mode in combos:
        k = (float(beta), sname)
        if k not in cache:
            cache[k] = S2.select96(v2_final16, v2_im16, prob96, valid96, beta, S2.VARIANT_SETS[sname], offset,
                                   w=w_sel)
        idx = cache[k]
        lat = S2.use_lat(idx, lat_mode)
        chosen = S2.chosen_labels(lab_pool, lab_lat, idx, lat_mode)
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
def load_labels(d: Path, n: int) -> Tuple[np.ndarray, np.ndarray]:
    lab = np.load(d / "labels.npy", mmap_mode="r")
    ok = np.load(d / "ok.npy", mmap_mode="r")
    if lab.shape[0] != n or lab.shape[1] != S2.G_K:
        raise SystemExit(f"{d}: labels {lab.shape} != [{n}, {S2.G_K}, 9]")
    return np.asarray(lab, np.float64), np.asarray(ok, bool)


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


def resolve_navtest(spec: Dict[str, Any], run: str, sel_w=None, calib_offset=None, cur: Optional[Dict] = None):
    """navtest restores the frozen val spec: -> (weights, offset).  sel_w / calib_offset None or '' = inherit; a given
    value must equal the spec's (SystemExit otherwise).  cur: model_meta of the navtest split (run, ckpt sha16,
    ep_target and limit must match the spec)."""
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
    if tuple(S2.VARIANT_SETS.get(best["set"], ())) != tuple(spec.get("set_types") or ()):
        raise SystemExit(f"--fix-from: variant set {best['set']!r} is {S2.VARIANT_SETS.get(best['set'])} now, "
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


def out_name(split: str, sel_w="default", offset=None) -> str:
    """default out dir name: <split> for the default weights, <split>__selw_<name> otherwise, + __cal_<sha8> when a
    calibration offset is applied (offset_tag)."""
    n = S2.sel_w_name(sel_w)
    base = split if n == "default" else f"{split}__selw_{n.replace(',', '_')}"
    return base + offset_tag(offset)


def run_eval(root, run: str, split: str, out=None, fix_from: str = "", n_boot: int = 2000,
             calib_offset: Optional[str] = None, lead_root: Path = CK_DATA / "lead", choose_lat: str = "on",
             sel_w=None) -> Dict[str, Any]:
    """sel_w / calib_offset None (or ''): navtrain_val = 'default' / OFF; navtest = inherited from the frozen val spec
    (--fix-from), an explicit value must equal it."""
    root = Path(root)
    cur = model_meta(root, run, split)
    fixed = spec_in = None
    if split == "navtest":
        if not fix_from:
            raise SystemExit("navtest needs --fix-from <navtrain_val metrics.json> (beta / variant set / lateral mode "
                             "are chosen on navtrain_val only)")
        src = json.loads(Path(fix_from).read_text())
        spec_in = frozen_spec(src, fix_from)
        w_sel, offset = resolve_navtest(spec_in, run, sel_w, calib_offset, cur)
        fixed = dict(spec_in["best"])
        cal_src = spec_in.get("calib_offset_src") if (calib_offset in (None, "")) else offset_source(calib_offset)
        cal_label = ("none (OFF)" if offset is None else
                     (f"inherited from val: {(cal_src or {}).get('path')}" if calib_offset in (None, "")
                      else calib_offset))
    else:
        w_sel = S2.resolve_w(sel_w or "default")
        cal = calib_offset or "none"
        offset = read_offset(cal)
        cal_src = offset_source(cal)
        cal_label = cal if offset is not None else "none (OFF)"
    w_name = S2.sel_w_name(w_sel)
    out = Path(out) if out else root / "eval" / run / out_name(split, w_sel, offset)
    pk = root / "packed" / split
    inf = root / "infer" / run / split
    lb = root / "labels" / split
    tdf = pd.read_parquet(pk / "tokens.parquet")
    n = len(tdf)
    valid96 = np.load(pk / "valid96.npy")
    v2f = np.load(pk / "v2_final.npy").astype(np.float64)
    v2im = np.load(pk / "v2_im.npy").astype(np.float64)
    pok = np.load(pk / "ok.npy")
    logit = np.load(inf / "score_logit.npy").astype(np.float64)
    done = np.load(inf / "done.npy")
    lab_p, ok_p = load_labels(lb / f"pool96_{run}", n)
    lab_l, ok_l = load_labels(lb / f"lat96_{run}", n)
    m = token_mask(pok, done, valid96, ok_p, ok_l)
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
                        combos, n_boot, 0, offset, sel_w=w_sel)
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
        spec = {"split": split, "run": run, "best": b, "set_types": list(S2.VARIANT_SETS[b["set"]]) if b else None,
                "choose_lat": choose_lat, "sel_w": w_name, "sel_w_values": [float(x) for x in w_sel],
                "calib_offset_values": out_d["calib_offset_values"], "calib_offset_src": cal_src, "model": cur,
                "n_eval": int(len(sel)), "n_total": int(n), "token_rule": out_d["token_rule"], "created": kst()}
        spec["spec_sha16"] = spec_sha16(spec)
        out_d["eval_spec"] = spec
        U.write_json(out / "eval_spec.json", spec)
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
        pt["sel_lat"] = S2.use_lat(idx, r["lat_mode"])
        pt["pdms_sel"] = S2.chosen_labels(lab_p[sel], lab_l[sel], idx, r["lat_mode"])[:, P]
    pt.to_parquet(out / "per_token.parquet", index=False)
    print(f"[{kst()}] eval_ck2 {run}/{split}: {len(sel)}/{n} tokens, {len(rows)} rows -> {out}", flush=True)
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
                    help="selection weight set (select.SEL_W_SETS): default = SEL_W | noim | plugin (0, 1, 1, 1); "
                         "unset: 'default' on navtrain_val, inherited from the val spec on navtest")
    a = ap.parse_args(argv)
    run_eval(a.root, a.run, a.split, a.out or None, a.fix_from, a.bootstrap, a.calib_offset, choose_lat=a.choose_lat,
             sel_w=a.sel_w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
