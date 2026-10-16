"""Every arm's official PDMS and EPDMS, in one table, read from the scorers' own output files.

Assembling this by hand from logs is how a sub-score from one scorer ends up next to a total from
another -- report/22 already has that bug.  So nothing here is typed in: PDMS comes from
work_dirs/eval/<name>/*.csv (navtest v1 scorer, `average` row) and EPDMS from
work_dirs/eval_epdms/_exp/<name>/*/*.csv (NAVSIM v2.2 scorer, `average_all_frames` row), and a
column that only one of the two scorers has is shown under that scorer alone.

    python tools/plan_v2/collect_results.py [--md report/28_results.md]
"""
import argparse
import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path("/home/external-user/kyungmin/SSR-v2")

# v1 PDMS sub-scores (navtest, the scorer scripts/evaluation/eval_para_ssr.sh runs)
PDMS_COLS = [("no_at_fault_collisions", "NC"), ("drivable_area_compliance", "DAC"),
             ("driving_direction_compliance", "DDC"), ("ego_progress", "EP"),
             ("time_to_collision_within_bound", "TTC"), ("comfort", "C")]
# v2.2 EPDMS sub-scores
EPDMS_COLS = [("no_at_fault_collisions", "NC"), ("drivable_area_compliance", "DAC"),
              ("driving_direction_compliance", "DDC"), ("traffic_light_compliance", "TLC"),
              ("ego_progress", "EP"), ("time_to_collision_within_bound", "TTC"),
              ("lane_keeping", "LK"), ("history_comfort", "HC"),
              ("two_frame_extended_comfort", "EC")]

# what each name is, so the table is readable without the launcher open
WHAT = {
    "para_ssr_v2_r34": "published baseline (WoTE heads, no corridor)",
    "para_ssr_v2_r34_feas_gt": "corridor, split field, GT field",
    "para_ssr_v2_r34_feas_pred": "corridor, split field, predicted field",
    "para_ssr_v2_r34_feas_gt_ch": "corridor MERGED + heads collapsed, GT field",
    "para_ssr_v2_r34_feas_pred_ch": "corridor MERGED + heads collapsed, predicted field",
    "para_ssr_v2_r34_feas_pred_gradfield": "ablation: cost gradient into the field (collapses)",
    "para_ssr_v2_r34_feas_pred_seg_ch": "corridor MERGED + heads collapsed + seg map, predicted field",
    "para_ssr_v2_r34_seg": "baseline + seg map (WoTE heads, no corridor)",
}

# PDMS runs whose work_dirs/eval directory is not the experiment name.  The first PDMS attempt
# on these died on a relative checkpoint path under hydra's cwd change, so the re-run got a new
# name; the alias keeps the number in the table instead of next to a blank PDMS column.
PDMS_ALIAS = {
    "para_ssr_v2_r34_feas_gt": "feas_gt_pdms_fix",
}


def latest(pattern):
    hits = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime)
    return hits[-1] if hits else None


def row_of(csv, token):
    try:
        d = pd.read_csv(csv)
    except Exception:
        return None
    if "token" not in d.columns or "score" not in d.columns:
        return None                      # some eval dirs hold other csvs (per-scene dumps)
    m = d[d.token == token]
    return m.iloc[0] if len(m) else None


def collect():
    out = {}
    unalias = {v: k for k, v in PDMS_ALIAS.items()}
    for c in glob.glob(str(ROOT / "work_dirs/eval/*/*.csv")):
        name = unalias.get(Path(c).parent.name, Path(c).parent.name)
        r = row_of(c, "average")
        if r is not None:
            prev = out.setdefault(name, {})
            if prev.get("_pdms_mtime", 0) < Path(c).stat().st_mtime:
                prev["_pdms_mtime"] = Path(c).stat().st_mtime
                prev["PDMS"] = 100 * float(r["score"])
                for col, short in PDMS_COLS:
                    if col in r:
                        prev[f"p_{short}"] = 100 * float(r[col])
    for c in glob.glob(str(ROOT / "work_dirs/eval_epdms/_exp/*/*/*.csv")):
        name = Path(c).parents[1].name
        r = row_of(c, "average_all_frames")
        if r is not None:
            prev = out.setdefault(name, {})
            if prev.get("_ep_mtime", 0) < Path(c).stat().st_mtime:
                prev["_ep_mtime"] = Path(c).stat().st_mtime
                prev["EPDMS"] = 100 * float(r["score"])
                for col, short in EPDMS_COLS:
                    if col in r:
                        prev[f"e_{short}"] = 100 * float(r[col])
    return out


# Inference field gate rows: gate_<arm>_deployed | gate_<arm>_s<static>_d<dyn>.  These are the SAME
# checkpoint as their arm with only the selection rule changed, so the arm has to be readable in the
# row or the comparison is lost.  `deployed` is each arm's control and must match its own published
# score; a gate row is only meaningful as a delta from it.
GATE = re.compile(r"^gate_(?P<arm>.+?)_(?:deployed|s(?P<ms>[\d.]+|off)_d(?P<md>[\d.]+|off))$")


def describe(name):
    g = GATE.match(name)
    if g:
        arm = WHAT.get(f"para_ssr_v2_r34_{g['arm']}", g["arm"])
        if g["ms"] is None:
            return f"  \u2514 {arm[:34]}", "게이트없음"
        which = ("정적+동적" if "off" not in (g["ms"], g["md"]) else
                 "정적만" if g["md"] == "off" else "동적만")
        m = g["ms"] if g["md"] == "off" else (g["md"] if g["ms"] == "off" else f"{g['ms']}/{g['md']}")
        return f"  \u2514 {arm[:26]} · 장게이트 {which}", f"m{m}"
    base = re.sub(r"_(ep\d+|last)$", "", name)
    tag = name[len(base) + 1:] if name != base else ""
    return WHAT.get(base, base), tag or "last"


def sortkey(name):
    """A field-gate row is the same checkpoint as its arm with one rule changed, so it belongs next
    to that arm, not in its own alphabetical clump."""
    g = GATE.match(name)
    arm = f"para_ssr_v2_r34_{g['arm']}" if g else re.sub(r"_(ep\d+|last)$", "", name)
    return (WHAT.get(arm, arm), 1 if g else 0, 0 if (g and g["ms"] is None) else 1, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="", help="also write a markdown table here")
    args = ap.parse_args()
    got = collect()
    if not got:
        print("평가 결과 없음")
        return

    lines = []
    lines.append(f"{'실험':46s}{'ckpt':>6s}{'PDMS':>8s}{'EPDMS':>8s}   "
                 + "".join(f"{s:>6s}" for _, s in EPDMS_COLS))
    lines.append("-" * (46 + 6 + 16 + 3 + 6 * len(EPDMS_COLS)))
    for name in sorted(got, key=sortkey):
        d = got[name]
        what, tag = describe(name)
        p = f"{d['PDMS']:8.2f}" if "PDMS" in d else f"{'-':>8s}"
        e = f"{d['EPDMS']:8.2f}" if "EPDMS" in d else f"{'-':>8s}"
        sub = "".join(f"{d[f'e_{s}']:6.2f}" if f"e_{s}" in d else f"{'-':>6s}"
                      for _, s in EPDMS_COLS)
        lines.append(f"{what[:45]:46s}{tag:>6s}{p}{e}   {sub}")
    lines.append("")
    lines.append("PDMS = navtest v1 scorer, EPDMS = NAVSIM v2.2 (human_penalty_filter on, i.e. FIXED).")
    lines.append("하위 점수는 EPDMS 쪽. 빈칸은 그 채점기를 아직 안 돌린 것.")
    text = "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text("# 결과 표 (자동 수집)\n\n```\n" + text + "\n```\n")
        print(f"\n-> {args.md}")


if __name__ == "__main__":
    main()
