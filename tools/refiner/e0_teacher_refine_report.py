#!/usr/bin/env python
"""E0 (PARA-SSR interaction_final) navtest trajectories refined by frozen stage-T refiners -> report (json + md).

STAGE E — E2 EVALUATION SPEC, side experiment (descriptive): "is the teacher refiner reliable on real planner drafts?"
Inputs: <base>/<arm>/{pred.npz, tokens.parquet, scores.parquet} written by refine_external_drafts.py + score_trajectories.py
for arm E0 (original) and the refined arms.  Everything is paired on the tokens scored in ALL arms.

  python tools/refiner/e0_teacher_refine_report.py [--base <dir>] [--out-dir report/refiner_T/e0_teacher_refine]

Definitions (same as eval_refiner.mix_metrics, one draft per token):
  fail_any = nc < 1 | dac < 1 | ddc < 1 | ttc < 1;  fixed = E0 fail_any and arm passes;  new_fail = E0 passes nc, dac,
  ddc, ttc, comfort and the arm fails any of them;  helped / harmed = token PDMS higher / lower than E0.
  CI: log-cluster paired bootstrap (stageE_compare.boot: resample logs, 10,000 draws, seed 0, percentile 95 %).
Modification statistics (from pred.npz; ungated = what is scored, theta 0):
  lon_live = any mode-A control point c_lon < 0 (decoder.lon_live; the refiner brakes), short_m = arc(tau0) - arc(tau1)
  at 4 s (liveness.arc_len), lat_m = max_k |e_lat_k| (decoded lateral offset, m), disp_m = max_k ||xy1_k - xy0_k||,
  unchanged = tau1 == tau0 bitwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stageE_compare import boot  # noqa: E402
from liveness import arc_len  # noqa: E402

BASE = Path("/home/external-user/ssd/yongjae_refiner/e0_teacher_refine")
OUT = Path("/home/external-user/yongjae/SSR/report/refiner_T/e0_teacher_refine")
SPLITS = Path("/home/external-user/ssd/yongjae_refiner/splits/navtest.parquet")
CSV = Path("/home/external-user/yongjae/SSR/work_dirs/eval/para_ssr_interaction_final/2026.09.17.00.09.41.csv")
ARMS = ("E0", "R_T4", "R_M4", "R_none4", "R_T3")
LABEL = {"E0": "E0 (PARA-SSR interaction_final, unrefined)", "R_T4": "E0 + run-4 R_T (BEVFusion)",
         "R_M4": "E0 + run-4 R_M (ReSMap)", "R_none4": "E0 + run-4 R_none (no BEV)", "R_T3": "E0 + run-3 R_T (reference)"}
CONTRASTS = (("R_T4", "E0"), ("R_M4", "E0"), ("R_none4", "E0"), ("R_T3", "E0"), ("R_T4", "R_none4"),
             ("R_M4", "R_none4"), ("R_T4", "R_M4"))
M = {"PDMS": "pdms", "NC": "nc", "DAC": "dac", "DDC": "ddc", "TTC": "ttc", "EP": "ep", "C": "comfort"}
CSV_COLS = dict(nc="no_at_fault_collisions", dac="drivable_area_compliance", ddc="driving_direction_compliance",
                ep="ego_progress", ttc="time_to_collision_within_bound", comfort="comfort", pdms="score")
CORE = ("nc", "dac", "ddc", "ttc")


def load_scores(base: Path, arm: str) -> pd.DataFrame:
    s = pd.read_parquet(base / arm / "scores.parquet")
    err = s[(s.k < 0) | (s.get("error", "").fillna("").astype(str) != "")]
    s = s[(s.k == 0)]
    s = s[s["error"].fillna("").astype(str) == ""] if "error" in s.columns else s
    return s.set_index("token")[list(M.values())].astype(float), sorted(set(err.token))


def fails(d: pd.DataFrame, keys) -> np.ndarray:
    return np.any(np.stack([d[k].values < 1 for k in keys], -1), -1)


def mod_stats(P, sel: np.ndarray, e0_fail: np.ndarray) -> dict:
    tau0 = P["tau0"][sel, 0].astype(np.float64)
    tau1 = P["tau1"][sel, 0].astype(np.float64)
    live = P["lon_live"][sel, 0].astype(bool)
    short = arc_len(tau0) - arc_len(tau1)
    lat = np.abs(P["e_lat"][sel, 0]).max(-1)
    disp = np.linalg.norm(tau1[..., :2] - tau0[..., :2], axis=-1).max(-1)
    unchanged = (P["tau1"][sel, 0] == P["tau0"][sel, 0]).all((-1, -2))
    pg = P["p_g"][sel, 0]

    def block(m):
        n = int(m.sum())
        if n == 0:
            return dict(n=0)
        return dict(n=n, unchanged=float(unchanged[m].mean()), brake_live=float(live[m].mean()),
                    short_gt_0p5m=float((short[m] > 0.5).mean()), short_gt_2m=float((short[m] > 2).mean()),
                    short_mean_m=float(short[m].mean()), short_p90_m=float(np.percentile(short[m], 90)),
                    short_max_m=float(short[m].max()),
                    short_mean_when_live_m=float(short[m & live].mean()) if (m & live).any() else None,
                    lat_gt_0p1m=float((lat[m] > 0.1).mean()), lat_gt_0p5m=float((lat[m] > 0.5).mean()),
                    lat_mean_m=float(lat[m].mean()), lat_p90_m=float(np.percentile(lat[m], 90)),
                    lat_max_m=float(lat[m].max()), disp_gt_0p5m=float((disp[m] > 0.5).mean()),
                    disp_mean_m=float(disp[m].mean()), p_g_mean=float(pg[m].mean()),
                    p_g_ge_0p5=float((pg[m] >= 0.5).mean()))
    allm = np.ones(len(short), bool)
    return {"all": block(allm), "E0_fail_any": block(e0_fail), "E0_pass": block(~e0_fail)}


def contrast(A: pd.DataFrame, B: pd.DataFrame, logs: np.ndarray, n_boot: int) -> dict:
    r = {}
    for name, c in M.items():
        d = A[c].values - B[c].values
        lo, hi = boot(d, logs, n_boot)
        r[name] = dict(diff=float(d.mean()), ci95=[lo, hi])
    fa, fb = fails(A, CORE), fails(B, CORE)
    pa, pb = fails(A, CORE + ("comfort",)), fails(B, CORE + ("comfort",))
    r["fixed"] = int((fb & ~fa).sum())
    r["new_fail"] = int((~pb & pa).sum())
    r["fail_any_A"], r["fail_any_B"] = int(fa.sum()), int(fb.sum())
    for k in ("nc", "dac", "ddc", "ttc", "comfort"):
        r[f"fixed_{k}"] = int(((B[k].values < 1) & (A[k].values >= 1)).sum())
        r[f"new_{k}"] = int(((B[k].values >= 1) & (A[k].values < 1)).sum())
    r["helped"] = int((A.pdms.values > B.pdms.values).sum())
    r["harmed"] = int((A.pdms.values < B.pdms.values).sum())
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args(argv)
    base, out = Path(a.base), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    S, errs = {}, {}
    for arm in ARMS:
        S[arm], errs[arm] = load_scores(base, arm)
    toks0 = pd.read_parquet(base / "E0" / "tokens.parquet")
    tok = pd.Index(toks0.token.astype(str))
    for arm in ARMS:
        tok = tok.intersection(S[arm].index)
    tok = pd.Index(sorted(tok))
    S = {k: v.loc[tok] for k, v in S.items()}
    sp = pd.read_parquet(SPLITS).drop_duplicates("token").set_index("token")
    logs = sp.log.reindex(tok).values
    city = sp.map_location.reindex(tok).values
    fg = toks0.set_index("token").frame_gap.reindex(tok).values.astype(bool)
    pkl_n = json.loads((base / "E0" / "predict_meta.json").read_text())["n_pkl"]

    res = dict(n_pkl_tokens=pkl_n, n_packed_rows=int(len(toks0)), n_tokens=int(len(tok)), n_logs=int(len(set(logs))),
               n_frame_gap_included=int(fg.sum()), score_errors={k: v for k, v in errs.items()},
               excluded=dict(not_in_pack=int(pkl_n - len(toks0)), not_scored_in_all_arms=int(len(toks0) - len(tok))),
               theta=0.0, bootstrap="log-cluster paired, 10,000 draws, seed 0, percentile 95 %",
               predict_meta={arm: json.loads((base / arm / "predict_meta.json").read_text()) for arm in ARMS})
    res["arms"] = {arm: dict(label=LABEL[arm], n=int(len(S[arm])), **{m: float(S[arm][c].mean()) for m, c in M.items()},
                             fail_any=int(fails(S[arm], CORE).sum()),
                             **{f"fail_{k}": int((S[arm][k] < 1).sum()) for k in ("nc", "dac", "ddc", "ttc", "comfort")})
                   for arm in ARMS}
    res["contrasts"] = {f"{x}-{y}": contrast(S[x], S[y], logs, a.n_boot) for x, y in CONTRASTS}
    # per city
    res["per_city"] = {}
    for c in sorted(set(city)):
        m = city == c
        res["per_city"][c] = dict(n=int(m.sum()), **{arm: float(S[arm].pdms.values[m].mean()) for arm in ARMS})
        for x, y in CONTRASTS[:5]:
            d = S[x].pdms.values[m] - S[y].pdms.values[m]
            lo, hi = boot(d, logs[m], a.n_boot)
            res["per_city"][c][f"{x}-{y}"] = dict(diff=float(d.mean()), ci95=[lo, hi],
                                                  fixed=int((fails(S[y][m], CORE) & ~fails(S[x][m], CORE)).sum()),
                                                  new_fail=int((~fails(S[y][m], CORE + ("comfort",)) &
                                                                fails(S[x][m], CORE + ("comfort",))).sum()))
    # modification statistics
    e0_fail = fails(S["E0"], CORE)
    res["modification"] = {}
    for arm in ARMS[1:]:
        P = np.load(base / arm / "pred.npz")
        pos = {str(t): i for i, t in enumerate(P["tokens"])}
        sel = np.array([pos[t] for t in tok])
        ms = mod_stats(P, sel, e0_fail)
        # outcome among modified tokens: helped / harmed split by E0 fail status
        d = S[arm].pdms.values - S["E0"].pdms.values
        ms["d_pdms_points_E0_fail_any"] = float(100 * d[e0_fail].mean())
        ms["d_pdms_points_E0_pass"] = float(100 * d[~e0_fail].mean())
        ms["d_ep_points_E0_pass"] = float(100 * (S[arm].ep.values - S["E0"].ep.values)[~e0_fail].mean())
        res["modification"][arm] = ms
    # stage-T convention sensitivity: frame_gap tokens excluded
    m = ~fg
    res["no_frame_gap"] = dict(n=int(m.sum()), **{arm: float(S[arm].pdms.values[m].mean()) for arm in ARMS})
    for x, y in CONTRASTS:
        d = S[x].pdms.values[m] - S[y].pdms.values[m]
        lo, hi = boot(d, logs[m], a.n_boot)
        res["no_frame_gap"][f"{x}-{y}"] = dict(diff=float(d.mean()), ci95=[lo, hi])
    # E0 batched scores vs the official E0 csv
    off = pd.read_csv(CSV, index_col=0)
    off = off[off.token != "average"].set_index("token")
    common = tok.intersection(off.index)
    e = S["E0"].loc[common]
    o = off.loc[common]
    res["E0_vs_official_csv"] = dict(
        n=int(len(common)), csv_pdms=float(o.score.mean()), batched_pdms=float(e.pdms.mean()),
        mismatch_exact={m: int((e[m].values != o[c].values.astype(float)).sum()) for m, c in CSV_COLS.items()},
        mismatch_gt_1e6={m: int((np.abs(e[m].values - o[c].values.astype(float)) > 1e-6).sum()) for m, c in CSV_COLS.items()},
        maxabs={m: float(np.abs(e[m].values - o[c].values.astype(float)).max()) for m, c in CSV_COLS.items()},
        csv_valid_false=int((off.valid.astype(str).str.lower() != "true").sum()),
        csv_pdms_all_rows=float(off.score.astype(float).mean()), csv_n_rows=int(len(off)))
    res["notes"] = NOTES
    (out / "e0_teacher_refine.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "e0_teacher_refine.md").write_text(render_md(res))
    print((out / "e0_teacher_refine.md").read_text())


NOTES = [
    "Inference path: tools/refiner/refine_external_drafts.py (stage-T net + decoder + teacher cache selection of "
    "eval_refiner.predict; the 13-draft bank replaced by E0's single trajectory; ego state v0/a0/eds/cmd from the packed "
    "navtest split; ckpt_best; autocast fp16 on GPU 0 as in stage T). The refiner has no cross-draft interaction: "
    "tools/refiner/tests/test_refine_external_drafts.py (CPU fp32, arms none / T) shows the single-draft path equals the "
    "bank path for the identity (human) slot and a perturbed slot (atol 1e-5). Real-data check (run-3 R_T, 640 navtest "
    "tokens, bank slot 0 fed alone on GPU vs the stored eval_navtest bank prediction): max |d tau1| 3.0 mm, max |d p_g| "
    "1.1e-3 (fp16 kernels depend on the batch composition).",
    "E0 check: the pkl trajectories scored with the batched scorer give PDMS 84.875 vs 84.868 in the official csv; the "
    "same metric caches are used, and the batched scorer is bitwise equal to pdm_score (scorer_equivalence.json), so the "
    "differences come from the pkl poses themselves (EP differs by <= 8.8e-4, median 5e-6, on 7,142 tokens as in "
    "scorer_equivalence.json 'model_vs_official_csv'; two tokens flip a multiplier: 14d53eb06a7d582a csv DAC 0 -> 1, "
    "72be63ed04f15f97 csv NC 1 -> 0). All arms start from the same pkl, so the contrasts are unaffected.",
    "theta 0 = correction always applied; p_g is recorded only (p_g >= 0.5 on ~10 % of E0 drafts for R_T4 / R_M4).",
    "Teachers: BEVFusion cache_val_50x100 (navtest is out-of-sample for BEVFusion, DECISION_NAVTEST.md); ReSMap navtest "
    "cache (its training data were not re-checked here). One seed, one fold-0 refiner per arm; descriptive.",
]


def _ci(r, scale=100.0, nd=2):
    return f"{scale * r['diff']:+.{nd}f} [{scale * r['ci95'][0]:+.{nd}f}, {scale * r['ci95'][1]:+.{nd}f}]"


def render_md(r) -> str:
    L = []
    L.append("# E0 navtest trajectories + frozen stage-T refiners (theta 0, official batched scorer)\n")
    L.append("Side experiment of STAGE E — E2 EVALUATION SPEC (descriptive): E0 = PARA-SSR interaction_final (epoch 29), "
             "one draft per token = E0's navtest trajectory, corrected by the frozen refiners (ckpt_best), correction "
             "always applied (gate ignored), scored with tools/refiner/score_trajectories.py (bitwise == pdm_score).\n")
    L.append(f"Tokens: {r['n_tokens']} of {r['n_pkl_tokens']} E0 navtest tokens ({r['n_logs']} logs); excluded: "
             f"{r['excluded']}; the {r['n_frame_gap_included']} frame_gap tokens (irregular FUTURE frames; stage T "
             "dropped them because the human bank / labels need the future) are included (ego state at t0 is finite "
             "for all). Scoring errors: " + ", ".join(f"{k} {len(v)}" for k, v in r["score_errors"].items()) + ".\n")
    L.append("## Per arm (means over tokens, x100)\n")
    L.append("| arm | PDMS | NC | DAC | DDC | TTC | EP | C | fail_any |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, v in r["arms"].items():
        L.append(f"| {k} | {100 * v['PDMS']:.2f} | {100 * v['NC']:.2f} | {100 * v['DAC']:.2f} | {100 * v['DDC']:.2f} | "
                 f"{100 * v['TTC']:.2f} | {100 * v['EP']:.2f} | {100 * v['C']:.2f} | {v['fail_any']} |")
    L.append("\n## Paired contrasts (points, log-cluster bootstrap 95 % CI)\n")
    L.append("| contrast | PDMS | NC | DAC | DDC | TTC | EP | C | fixed | new_fail | helped / harmed |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in r["contrasts"].items():
        L.append(f"| {k} | " + " | ".join(_ci(v[m]) for m in M) +
                 f" | {v['fixed']} | {v['new_fail']} | {v['helped']} / {v['harmed']} |")
    L.append("\nfixed = B fails nc/dac/ddc/ttc and A passes all four; new_fail = B passes nc/dac/ddc/ttc/comfort and A "
             "fails one. Per-metric fixed / new counts (A vs B):\n")
    L.append("| contrast | " + " | ".join(f"{k} fixed/new" for k in ("nc", "dac", "ddc", "ttc", "comfort")) + " |")
    L.append("|---|---|---|---|---|---|")
    for k, v in r["contrasts"].items():
        L.append(f"| {k} | " + " | ".join(f"{v['fixed_' + m]} / {v['new_' + m]}" for m in
                                          ("nc", "dac", "ddc", "ttc", "comfort")) + " |")
    L.append("\n## Per city (PDMS x100; delta vs E0 with CI)\n")
    L.append("| city | n | E0 | R_T4 - E0 | R_M4 - E0 | R_none4 - E0 | R_T3 - E0 | R_T4 - R_none4 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c, v in r["per_city"].items():
        L.append(f"| {c} | {v['n']} | {100 * v['E0']:.2f} | {_ci(v['R_T4-E0'])} | {_ci(v['R_M4-E0'])} | "
                 f"{_ci(v['R_none4-E0'])} | {_ci(v['R_T3-E0'])} | {_ci(v['R_T4-R_none4'])} |")
    L.append("\n## Modification statistics (what is scored; theta 0)\n")
    L.append("| arm | subset | n | unchanged | brake live | short>0.5 m | short>2 m | mean short m | p90 short m | "
             "max short m | lat>0.1 m | lat>0.5 m | mean lat m | max lat m | disp>0.5 m | p_g>=0.5 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, ms in r["modification"].items():
        for sub in ("all", "E0_fail_any", "E0_pass"):
            b = ms[sub]
            if not b.get("n"):
                continue
            L.append(f"| {arm} | {sub} | {b['n']} | {b['unchanged']:.3f} | {b['brake_live']:.3f} | "
                     f"{b['short_gt_0p5m']:.3f} | {b['short_gt_2m']:.3f} | {b['short_mean_m']:.2f} | "
                     f"{b['short_p90_m']:.2f} | {b['short_max_m']:.1f} | {b['lat_gt_0p1m']:.3f} | "
                     f"{b['lat_gt_0p5m']:.3f} | {b['lat_mean_m']:.3f} | {b['lat_max_m']:.2f} | "
                     f"{b['disp_gt_0p5m']:.3f} | {b['p_g_ge_0p5']:.3f} |")
    L.append("\n| arm | d PDMS pts on E0 fail_any tokens | d PDMS pts on E0 passing tokens | d EP pts on E0 passing |")
    L.append("|---|---|---|---|")
    for arm, ms in r["modification"].items():
        L.append(f"| {arm} | {ms['d_pdms_points_E0_fail_any']:+.2f} | {ms['d_pdms_points_E0_pass']:+.2f} | "
                 f"{ms['d_ep_points_E0_pass']:+.2f} |")
    nf = r["no_frame_gap"]
    L.append(f"\n## Sensitivity: stage-T token convention (frame_gap excluded, n = {nf['n']})\n")
    L.append("| " + " | ".join(ARMS) + " | " + " | ".join(f"{x}-{y}" for x, y in CONTRASTS) + " |")
    L.append("|" + "---|" * (len(ARMS) + len(CONTRASTS)))
    L.append("| " + " | ".join(f"{100 * nf[a]:.2f}" for a in ARMS) + " | " +
             " | ".join(_ci(nf[f"{x}-{y}"]) for x, y in CONTRASTS) + " |")
    e = r["E0_vs_official_csv"]
    L.append("\n## Check: E0 batched scores vs the official E0 csv\n")
    L.append(f"n {e['n']}; PDMS csv {100 * e['csv_pdms']:.4f} vs batched {100 * e['batched_pdms']:.4f} "
             f"(csv all rows {100 * e['csv_pdms_all_rows']:.4f}, {e['csv_n_rows']} rows, valid=False {e['csv_valid_false']}); "
             f"exact mismatches {e['mismatch_exact']}; > 1e-6 {e['mismatch_gt_1e6']}; max abs {e['maxabs']}.\n")
    L.append("## Notes\n")
    L += [f"- {n}" for n in r.get("notes", [])]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
