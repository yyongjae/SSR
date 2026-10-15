#!/usr/bin/env python
"""Wrapper around summarize_teacher_navtest.py (imported, NOT modified) for a fixed-epoch eval root, plus a paired
comparison against the earlier ckpt_last run (user 2026-10-08: "10epoch짜리로 navtest 다시 진행해봐 그럼 둘다").

  cd /workspace/yongjae/SSR-ck2 && PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
    nice -n 10 /venv/ssr/bin/python tools/ck/e2e2/summarize_teacher_navtest_ep.py
  then: tools/ck/e2e2/check_teacher_navtest_headline_ep.py (independent re-check, appends section 6 of table.md)

What it changes (nothing else):
  * the base module's eval root D -> --eval-root (default CK_DATA/ck2/eval/navtest_ep9); the base main / render run
    unchanged on it (same tables, same token set, same bootstrap) and write <root>/{metrics.json, table.md,
    per_token.parquet}; refuses a root inside the earlier ckpt_last root (CK_DATA/ck2/eval/navtest);
  * checks that every teacher output under the root came from ckpt_<which>.pt (gtfree meta + r34 infer_meta, same
    sha16) before summarising, and records each teacher run's ep_target (the base summary scores the CK EP head against
    ck_targets(labels, that ep_target); official PDMS / sub-scores and the paired section 0 stay official);
  * table title "(..., ckpt_last)" -> "(..., ckpt_<which>)", the section-5 pointer to the ep check script;
  * adds section 0 / metrics.json["ep_vs_ref"]: <which> - ckpt_last per mode, paired on the same tokens (the per-token
    labels of both runs' per_token.parquet, joined on token): ΔPDMS with the log-clustered bootstrap
    (tools/ck/eval_ck.log_bootstrap, 2000, seed 0), better / worse token fractions, Δ sub-score means, lead-decel
    slice NC|TTC failure rate of both runs and its paired Δ (same bootstrap on the slice), fraction of tokens whose pick
    changed.  Modes that do not depend on the teacher checkpoint (v2, oracle256, refA oracle16 / oracle96, old-teacher
    rows) must come out bitwise identical and are listed as a check, not in the table.
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import argparse  # noqa: E402
import time  # noqa: E402
from typing import Dict, List  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools.ck import ckutil as U  # noqa: E402
from tools.ck.e2e2 import summarize_teacher_navtest as S  # noqa: E402

BASE_ROOT = S.D                                   # .../ck2/eval/navtest (ckpt_last) - read only here
EP_ROOT = S.CK_DATA / "ck2" / "eval" / "navtest_ep9"
METRIC_COLS = ("pdms", "nc", "dac", "ep", "ttc", "comfort")
# per_token (section, mode) whose labels do not depend on the CK2 checkpoint -> must be identical in both runs
INDEP = {("gtfree", "v2"), ("gtfree", "oracle256"), ("refA", "oracle16"), ("refA", "oracle96"),
         ("old", "a(1)"), ("old", "b(1)"), ("old", "ck_noim_a")}
ORDER = (("gtfree", ("P256", "P256_lat", "P96", "P96_lat_always", "P96_lat_skiplatvar", "oracle_top16", "oracle96")),
         ("refA", tuple(f"{m}_{x}" for m in S.R34_MODES for x in ("a", "b")) + ("v2_lat",)))


def _inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def check_ckpts(root: Path, which: str) -> Dict:
    out = {}
    for t in S.TEACHERS:
        gm = U.read_json(root / t / "gtfree" / "meta.json") or {}
        im = U.read_json(root / t / "r34" / "infer_meta.json") or {}
        want = S.CK_DATA / "ck2" / "train" / t / f"ckpt_{which}.pt"
        sha = U.sha256_file(want, 16)
        rec = dict(ckpt=str(want), sha16=sha, gtfree_ckpt=gm.get("ckpt"), gtfree_sha16=gm.get("ckpt_sha16"),
                   r34_ckpt=im.get("ckpt"), r34_sha16=(im.get("ckpt_sha16") or "")[:16], r34_which=im.get("which"))
        rec["ep_target"] = S.teacher_ep_target(want)      # EP of the score-quality metrics (base summary, ck_targets)
        rec["ok"] = (Path(gm.get("ckpt", "")).name == want.name and gm.get("ckpt_sha16") == sha
                     and Path(im.get("ckpt", "")).name == want.name and rec["r34_sha16"] == sha
                     and im.get("which") == which)
        if not rec["ok"]:
            raise SystemExit(f"{t}: outputs under {root} are not all from {want}: {rec}")
        out[t] = rec
    return out


def ep_vs_ref(root: Path, ref: Path, n_boot: int) -> Dict:
    """Paired <root> - <ref> per (teacher, section, mode) on the same tokens."""
    from navsim.agents.para_ssr.ck import constants as Cn
    pn = pd.read_parquet(root / "per_token.parquet")
    po = pd.read_parquet(ref / "per_token.parquet")
    Mn, Mo = U.read_json(root / "metrics.json"), U.read_json(ref / "metrics.json")
    tok = pn[(pn.teacher == "ck2T") & (pn.section == "gtfree") & (pn["mode"] == "v2")].token.to_numpy()
    tok_o = po[(po.teacher == "ck2T") & (po.section == "gtfree") & (po["mode"] == "v2")].token.to_numpy()
    same_set = bool(len(tok) == len(tok_o) and set(tok) == set(tok_o))
    if not same_set:
        raise SystemExit(f"token sets differ: {len(tok)} vs {len(tok_o)}")
    logs = pn[(pn.teacher == "ck2T") & (pn.section == "gtfree") & (pn["mode"] == "v2")].log.to_numpy()
    lead, lead_src = S.lead_flags(S.SPLIT, tok)
    idx = pd.Index(tok)

    def get(df, t, s, m):
        q = df[(df.teacher == t) & (df.section == s) & (df["mode"] == m)]
        if q.empty:
            return None
        q = q.set_index("token").reindex(idx)
        assert q["pdms"].notna().all(), (t, s, m)
        return q

    fail = lambda q: ((q["nc"].to_numpy() < 1) | (q["ttc"].to_numpy() < 1)).astype(np.float64)  # noqa: E731
    res: Dict = dict(ref_root=str(ref), root=str(root), n=int(len(tok)), n_logs=int(len(np.unique(logs))),
                     lead_n=None if lead is None else int(lead.sum()), lead_src=lead_src, n_boot=n_boot,
                     same_token_set=same_set, ref_created=Mo.get("created"),
                     ref_ckpt={t: Mo["teachers"][t]["ckpt"] for t in S.TEACHERS},
                     ref_ckpt_sha16={t: Mo["teachers"][t]["ckpt_sha16"] for t in S.TEACHERS},
                     rows={}, independent={}, headline_consistency={})
    hmap_old = {t: Mo["headline_pdms"][t] for t in S.TEACHERS}
    hmap_new = {t: Mn["headline_pdms"][t] for t in S.TEACHERS}
    keys = sorted(set(map(tuple, pn[["teacher", "section", "mode"]].drop_duplicates().to_numpy().tolist())))
    mx = 0.0
    for t, s, m in keys:
        a, b = get(pn, t, s, m), get(po, t, s, m)
        if b is None:
            continue
        la = {c: a[c].to_numpy(np.float64) for c in Cn.LABEL_COLS}
        lb = {c: b[c].to_numpy(np.float64) for c in Cn.LABEL_COLS}
        # the per-token rows must reproduce each run's own headline numbers (same key naming as the base summary)
        hk = f"{s}/{m}"
        for H, lab, tag in ((hmap_new, la, "new"), (hmap_old, lb, "ref")):
            if hk in H[t]:
                d_ = abs(H[t][hk] - float(lab["pdms"].mean()))
                mx = max(mx, d_)
                res["headline_consistency"].setdefault(f"{t}:{hk}", {})[tag] = d_
        if (s, m) in INDEP:
            same = all(np.array_equal(la[c], lb[c]) for c in Cn.LABEL_COLS)
            res["independent"][f"{t}:{s}/{m}"] = {"identical": bool(same),
                                                  "max_abs": float(max(np.abs(la[c] - lb[c]).max()
                                                                       for c in Cn.LABEL_COLS))}
            continue
        d = la["pdms"] - lb["pdms"]
        fa, fb = fail(a), fail(b)
        r = dict(teacher=t, section=s, mode=m,
                 pdms_new=float(la["pdms"].mean()), pdms_ref=float(lb["pdms"].mean()),
                 d_pdms=S.log_bootstrap(d, logs, n_boot, 0),
                 new_better=float((d > 1e-9).mean()), new_worse=float((d < -1e-9).mean()),
                 d_sub={c: float((la[c] - lb[c]).mean()) for c in METRIC_COLS[1:]},
                 fail_nc_new=float((la["nc"] < 1).mean()), fail_nc_ref=float((lb["nc"] < 1).mean()),
                 fail_ttc_new=float((la["ttc"] < 1).mean()), fail_ttc_ref=float((lb["ttc"] < 1).mean()),
                 pick_changed=(None if m == "v2_lat" else float((a["chosen"].to_numpy() != b["chosen"].to_numpy()).mean())),
                 traj_changed_note=("pick index only; _b / lateral rows can differ with the same index" if
                                    m.endswith("_b") or "lat" in m else ""))
        if lead is not None and lead.any():
            r.update(lead_fail_new=float(fa[lead].mean()), lead_fail_ref=float(fb[lead].mean()),
                     lead_pdms_new=float(la["pdms"][lead].mean()), lead_pdms_ref=float(lb["pdms"][lead].mean()),
                     d_lead_fail=S.log_bootstrap((fa - fb)[lead], logs[lead], n_boot, 0),
                     d_lead_pdms=S.log_bootstrap(d[lead], logs[lead], n_boot, 0))
        res["rows"][f"{t}:{s}/{m}"] = r
    res["headline_consistency_maxabs"] = mx
    res["all_independent_identical"] = bool(all(v["identical"] for v in res["independent"].values()))
    return res


def render_ep(E: Dict, which: str, ck: Dict) -> List[str]:
    f2, ci = S.f2, S.ci
    Lm = [f"## 0. {which} − ckpt_last (같은 teacher, 같은 토큰, paired)", "",
          f"이전 실행 {E['ref_root']} (ckpt_last, summary {E['ref_created']}) 의 per_token.parquet 와 토큰으로 맞춰 비교. "
          f"토큰 {E['n']}, log {E['n_logs']}, 앞차 감속 slice n {E['lead_n']}. Δ = {which} − last. bootstrap "
          f"{E['n_boot']}회 (log 군집, paired, seed 0). 모드마다 따로 낸 95% CI이며 다중 비교 보정은 하지 않음. [실측]", "",
          "- ckpt: " + "; ".join(f"{t} {which} {ck[t]['sha16']} vs last {E['ref_ckpt_sha16'][t]}" for t in ck),
          "- 고른 후보 바뀐 % = 저장된 선택 index가 다른 토큰 비율 (_b / lateral 행은 index가 같아도 lateral 교정 궤적이 "
          "달라질 수 있음). v2_lat는 index가 항상 0이라 '-'.", "",
          f"| teacher | 모드 | PDMS {which} | PDMS last | ΔPDMS [95% CI] | {which} 나음/나쁨 % | ΔNC | ΔDAC | ΔEP | ΔTTC | ΔC | "
          f"NC 실패 {which}/last | TTC 실패 {which}/last | 앞차 NC/TTC 실패 {which}/last | Δ앞차 실패 [95% CI] | "
          f"앞차 PDMS {which}/last | 고른 후보 바뀐 % |",
          "|" + "---|" * 17]
    for t in S.TEACHERS:
        for s, modes in ORDER:
            for m in modes:
                r = E["rows"].get(f"{t}:{s}/{m}")
                if r is None:
                    continue
                lab = S.GTFREE_LABEL.get(m, m) if s == "gtfree" else m
                ds = r["d_sub"]
                pc = "-" if r["pick_changed"] is None else f"{100 * r['pick_changed']:.1f}"
                Lm.append(
                    f"| {t} | {s}/{lab} | {f2(r['pdms_new'])} | {f2(r['pdms_ref'])} | {ci(r['d_pdms'])} | "
                    f"{100 * r['new_better']:.1f} / {100 * r['new_worse']:.1f} | "
                    + " | ".join(f"{100 * ds[c]:+.2f}" for c in ("nc", "dac", "ep", "ttc", "comfort")) + " | "
                    f"{f2(r['fail_nc_new'])} / {f2(r['fail_nc_ref'])} | {f2(r['fail_ttc_new'])} / {f2(r['fail_ttc_ref'])} | "
                    f"{f2(r.get('lead_fail_new'))} / {f2(r.get('lead_fail_ref'))} | {ci(r.get('d_lead_fail'))} | "
                    f"{f2(r.get('lead_pdms_new'))} / {f2(r.get('lead_pdms_ref'))} | {pc} |")
    ind = E["independent"]
    Lm += ["", f"- checkpoint와 무관한 행 {len(ind)}개 (v2, oracle256, 참고 A oracle16 / oracle96, 이전 teacher 행) 는 두 "
               f"실행에서 라벨이 비트 단위로 같음: {E['all_independent_identical']} (최대 차 "
               f"{max(v['max_abs'] for v in ind.values()):.1e}).",
           f"- 두 실행의 per_token 평균 = 각자 metrics.json headline: 최대 차 {E['headline_consistency_maxabs']:.1e}.", ""]
    return Lm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--eval-root", default=str(EP_ROOT))
    ap.add_argument("--which", default="ep9")
    ap.add_argument("--ref-root", default=str(BASE_ROOT), help="earlier run to compare against (ckpt_last)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args(argv)
    root, ref = Path(a.eval_root), Path(a.ref_root)
    if _inside(root, BASE_ROOT):
        raise SystemExit(f"--eval-root {root} is inside the earlier run root {BASE_ROOT}")
    t0 = time.time()
    ck = check_ckpts(root, a.which)
    print(f"[{S.kst()}] ckpts ok: " + ", ".join(f"{t} {v['sha16']}" for t, v in ck.items()), flush=True)

    S.D = root                                   # the base main / render read the module global at call time
    rc = S.main(["--bootstrap", str(a.bootstrap), "--out", str(root)])
    if rc:
        return rc

    E = ep_vs_ref(root, ref, a.bootstrap)
    M = U.read_json(root / "metrics.json")
    M["wrapper"] = dict(script=str(Path(__file__).resolve()), base_script=str(Path(S.__file__).resolve()),
                        eval_root=str(root), which=a.which, ref_root=str(ref), ckpts=ck, created=S.kst())
    M["ep_vs_ref"] = E
    U.write_json(root / "metrics.json", M)

    txt = (root / "table.md").read_text()
    lines = txt.split("\n")
    assert lines[0].endswith("ckpt_last)"), lines[0]
    lines[0] = lines[0].replace("ckpt_last)", f"ckpt_{a.which})")
    txt = "\n".join(lines)
    txt = txt.replace("- 독립 재계산: tools/ck/e2e2/check_teacher_navtest_headline.py ->",
                      "- 독립 재계산: tools/ck/e2e2/check_teacher_navtest_headline_ep.py (기존 check 스크립트 복사본, 경로와 "
                      "재사용 라벨의 시각 검사만 바뀜) ->")
    head, sep, rest = txt.partition("\n## 1. ")
    assert sep, "section 1 not found"
    txt = head.rstrip("\n") + "\n\n" + "\n".join(render_ep(E, a.which, ck)) + "\n## 1. " + rest
    (root / "table.md").write_text(txt)
    bad = [k for k, v in E["independent"].items() if not v["identical"]]
    print("\n".join(render_ep(E, a.which, ck)), flush=True)
    print(f"[{S.kst()}] ep_vs_ref: {len(E['rows'])} rows, independent identical {E['all_independent_identical']} "
          f"{bad}, headline consistency {E['headline_consistency_maxabs']:.1e} ({time.time() - t0:.1f} s)", flush=True)
    return 0 if not bad and E["headline_consistency_maxabs"] < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
