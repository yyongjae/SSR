#!/usr/bin/env python
"""Independent second check of the CK2-teacher navtest headline numbers (does not import summarize_teacher_navtest.py,
eval_ck, select / select2 or any repo module; numpy + pandas + ast only).

  cd /workspace/yongjae/SSR-ck2 && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 /venv/ssr/bin/python \
    tools/ck/e2e2/check_teacher_navtest_headline.py

  1 picks: every saved choice is re-derived from the saved model outputs alone (probabilities, v2 im / final, variant
    validity) with an own implementation of the selection rules -> must equal the saved indices.
  2 trajectories: the submitted trajectories equal the chosen candidates (bitwise) and every label file was computed
    on the trajectory file now on disk (label meta traj_sha16 == sha256 of the array).
  3 headline PDMS: recomputed from the official labels at the chosen indices -> must equal <D>/metrics.json.
  4 no GT / labels in the choice: (1) + an AST scan of the inference / selection code (file reads, dataset flags, batch
    keys) + file times (picks written before the labels of their own trajectories existed).
-> <D>/check_headline.json; exit 1 if any assertion fails.
"""
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CK = Path("/home/external-user/ssd/yongjae_refiner/ck")
D = CK / "ck2" / "eval" / "navtest"
REPO = Path(__file__).resolve().parents[3]
ANCH = Path("/home/external-user/kyungmin/SSR-v2/data/planning_vb/trajectory_anchors_256.npy")
COLS = ("nc", "dac", "ep", "ttc", "comfort", "ddc", "pdms", "raw_progress", "pdm_progress_eff")
PD = COLS.index("pdms")
EPS = 1e-6
FAILS = []


def kst(t=None):
    return time.strftime("%Y-%m-%d %H:%M:%S KST", time.gmtime((time.time() if t is None else t) + 9 * 3600))


def need(ok, what):
    if not ok:
        FAILS.append(what)
    return bool(ok)


def lab(d, k):
    x = np.load(Path(d) / "labels.npy").astype(np.float64)
    assert x.shape[1:] == (k, len(COLS)) and np.load(Path(d) / "ok.npy").all(), d
    return x


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def score(p, im=None, beta=None, v2f=None):
    """spec: w_im log(im + eps) + 0.5 log(nc + eps) + 0.5 log(dac + eps) + log(5 ttc + 2 c + 5 ep + eps) (keys nc dac ep ttc
    comfort); CK-only = w_im 0; old formula = w_im 0.1, blended (1 - beta) v2_final + beta score."""
    p = np.asarray(p, np.float64)
    wim = 0.0 if im is None else 0.1
    im = np.ones(p.shape[:-1]) if im is None else im
    s = (wim * np.log(im + EPS) + 0.5 * np.log(p[..., 0] + EPS) + 0.5 * np.log(p[..., 1] + EPS)
         + 1.0 * np.log(5 * p[..., 3] + 2 * p[..., 4] + 5 * p[..., 2] + EPS))
    if beta is not None:
        s = (1.0 - beta) * v2f + beta * s
    return np.where(np.isfinite(s), s, -np.inf)


def pick96(s96, valid96):
    """identity columns (c % 6 == 0) always allowed, variants only if valid; ties: identities (k asc) then c order."""
    c = np.arange(96)
    s = np.where(valid96 | (c % 6 == 0)[None], s96, -np.inf)
    order = np.r_[c[c % 6 == 0], c[c % 6 != 0]]
    return order[np.argmax(s[:, order], 1)]


def pool(x16, x80):
    n = x16.shape[0]
    return np.concatenate([x16[:, :, None], x80.reshape((n, 16, 5) + x16.shape[2:])], 2).reshape((n, 96) + x16.shape[2:])


def sha16(path):
    return hashlib.sha256(np.ascontiguousarray(np.load(path, mmap_mode="r")).tobytes()).hexdigest()[:16]


def main():
    t0 = time.time()
    M = json.loads((D / "metrics.json").read_text())
    H = M["headline_pdms"]
    pk = CK / "packed" / "navtest"
    tdf = pd.read_parquet(pk / "tokens.parquet")
    N = len(tdf)
    ar = np.arange(N)
    out = {"created": kst(), "script": str(Path(__file__).resolve()), "n": N, "picks": {}, "traj": {}, "headline": {},
           "label_traj_sha": {}, "static": {}, "times": {}}
    need(M["n_eval"] == N, "summary token set != full split")
    v2f = np.load(pk / "v2_final.npy").astype(np.float64)
    v2im = np.load(pk / "v2_im.npy").astype(np.float64)
    cand = np.load(pk / "cand.npy", mmap_mode="r")
    LC = lab(CK / "labels/navtest/cand", 16)
    L256 = lab(CK / "labels/navtest/raw256", 256)
    LVR = lab(D / "labels/ck2eval_r34_var80", 80)
    valid80 = np.load(D / "r34_variants/valid80.npy")
    traj80 = np.load(D / "r34_variants/traj80.npy", mmap_mode="r")
    anchors = np.load(ANCH).astype(np.float32)
    rawt = Path(json.loads((CK / "labels/navtest/raw256/meta.json").read_text())["traj_path"])
    need(bool((np.load(rawt) == anchors[None]).all()), "raw256 trajectories != the 256 anchors")
    lsrc = {"raw256": (CK / "labels/navtest/raw256", rawt), "cand": (CK / "labels/navtest/cand", pk / "cand.npy"),
            "r34_var80": (D / "labels/ck2eval_r34_var80", D / "r34_variants/traj80.npy")}
    hl = {}

    def head(t, key, x):
        ref = H[t].get(key)
        v = float(x[:, PD].mean())
        hl[f"{t}:{key}"] = {"recomputed": v, "summary": ref, "absdiff": None if ref is None else abs(v - ref)}
        need(ref is not None and abs(v - ref) < 1e-9, f"headline {t}:{key} {v} vs {ref}")

    for t, old in (("ck2T", "ckT_p1"), ("ck2M", "ckM_p1")):
        g, rd = D / t / "gtfree", D / t / "r34"
        P = {}
        # ---------------- 1 picks from model outputs only
        g1 = np.load(g / "g1_prob.npy")
        s256 = score(g1)
        p256 = np.load(g / "p256_idx.npy")
        top16 = np.load(g / "g2_top16.npy")
        P["P256"] = int((np.argmax(s256, 1) == p256).sum())
        P["top16"] = int((np.argsort(-s256, 1, kind="stable")[:, :16] == top16).all(1).sum())
        del g1
        v96 = np.load(g / "g2_valid96.npy")
        col = np.load(g / "p96_col.npy")
        P["P96"] = int((pick96(score(np.load(g / "g2_prob96.npy")), v96) == col).sum())
        pr16 = sig(np.load(rd / "score_logit16.npy"))
        pr80 = sig(np.load(rd / "score_logit80.npy"))
        pz = np.load(rd / "picks.npz")
        need((pz["rows"] == ar).all(), f"{t} picks rows")
        im96, f96 = np.repeat(v2im, 6, 1), np.repeat(v2f, 6, 1)
        pr96, val96r = pool(pr16, pr80), pool(np.ones((N, 16), bool), valid80)
        mine = {"r34_ck_noim": np.argmax(score(pr16), 1),
                "r34_old_b1": np.argmax(score(pr16, v2im, 1.0, v2f), 1),
                "r34_old_b0.5": np.argmax(score(pr16, v2im, 0.5, v2f), 1),
                "pool96_ck_noim": pick96(score(pr96), val96r),
                "pool96_old_b1": pick96(score(pr96, im96, 1.0, f96), val96r)}
        for m, idx in mine.items():
            P[m] = int((idx == pz[m]).sum())
        opr = sig(np.load(CK / "infer" / old / "navtest/score_logit.npy").astype(np.float32))
        opt = pd.read_parquet(CK / "eval" / old / "navtest/per_token.parquet")
        oa = opt[(opt.variant == "a") & (opt.beta == 1.0)].set_index("token").reindex(tdf.token)
        ob = opt[(opt.variant == "b") & (opt.beta == 1.0)].set_index("token").reindex(tdf.token)
        oa_mine = np.argmax(score(opr, v2im, 1.0, v2f), 1)
        P["old_a1"] = int((oa_mine == oa.chosen.to_numpy()).sum())
        need((ob.chosen.to_numpy() == oa.chosen.to_numpy()).all(), f"{old} b(1) idx != a(1) idx")
        for k, v in P.items():
            need(v == N, f"{t} pick {k}: {v}/{N} equal")
        out["picks"][t] = P
        # ---------------- 2 trajectories
        ft = np.load(g / "final_traj.npy")
        g1l = np.load(g / "g1_lat_traj.npy", mmap_mode="r")
        var = np.load(g / "g2_var80_traj.npy")
        poolt = pool(anchors[top16], var)
        latv = np.isin(col % 6, (4, 5))
        T = {"P256=anchor": np.array_equal(ft[:, 0], anchors[p256]),
             "P256_lat=g1_lat": np.array_equal(ft[:, 1], np.asarray(g1l[ar, p256])),
             "P96_off=pool": np.array_equal(ft[:, 2], poolt[ar, col]),
             "P96_onexl=rule": np.array_equal(ft[:, 4], np.where(latv[:, None, None], ft[:, 2], ft[:, 3]))}
        sel = json.loads((rd / "select.json").read_text())
        rf = np.load(rd / "final_traj.npy")
        rs = np.load(rd / "score_stack.npy")
        cp = pool(np.asarray(cand[:, :16], np.float32), np.asarray(traj80))
        lp = pool(np.load(rd / "lat_traj16.npy"), np.load(rd / "lat_traj80.npy"))
        for m in mine:
            idx = pz[m]
            isr = m.startswith("r34")
            a_ = np.asarray(cand[ar, idx], np.float32) if isr else cp[ar, idx]
            b_ = np.load(rd / "lat_traj16.npy")[ar, idx] if isr else lp[ar, idx]
            T[f"{m}_a"] = np.array_equal(rf[:, sel["final_names"].index(f"{m}_a")], a_)
            T[f"{m}_b"] = np.array_equal(rf[:, sel["final_names"].index(f"{m}_b")], b_)
            T[f"{m}_b=stack"] = np.array_equal(rs[:, sel["stack_names"].index(f"{m}_b")], b_)
        for k, v in T.items():
            need(v, f"{t} traj {k}")
        out["traj"][t] = {k: bool(v) for k, v in T.items()}
        lsrc[f"{t}_var80"] = (D / f"labels/ck2eval_gtfree_{t}_var80", g / "g2_var80_traj.npy")
        lsrc[f"{t}_final5"] = (D / f"labels/ck2eval_gtfree_{t}_final5", g / "final_traj.npy")
        lsrc[f"{t}_stack"] = (D / f"labels/ck2eval_{t}_r34_stack", rd / "score_stack.npy")
        # ---------------- 3 headline PDMS from labels at the chosen indices
        LV = lab(lsrc[f"{t}_var80"][0], 80)
        LF = lab(lsrc[f"{t}_final5"][0], 5)
        LS = lab(lsrc[f"{t}_stack"][0], len(sel["stack_names"]))
        LP = pool(L256[ar[:, None], top16], LV)
        allowed = v96 | (np.arange(96) % 6 == 0)[None]
        head(t, "gtfree/v2", LC[:, 0])
        head(t, "gtfree/P256", L256[ar, p256])
        head(t, "gtfree/P256_lat", LF[:, 1])
        head(t, "gtfree/P96", LP[ar, col])
        head(t, "gtfree/P96_lat_always", LF[:, 3])
        head(t, "gtfree/P96_lat_skiplatvar", np.where(latv[:, None], LP[ar, col], LF[:, 3]))
        head(t, "gtfree/oracle256", L256[ar, L256[..., PD].argmax(1)])
        head(t, "gtfree/oracle96", LP[ar, np.where(allowed, LP[..., PD], -np.inf).argmax(1)])
        o16 = L256[ar[:, None], top16]
        head(t, "gtfree/oracle_top16", o16[ar, o16[..., PD].argmax(1)])
        LPR = pool(LC, LVR)
        head(t, "refA/oracle16", LC[ar, LC[..., PD].argmax(1)])
        head(t, "refA/oracle96", LPR[ar, np.where(val96r, LPR[..., PD], -np.inf).argmax(1)])
        for m in mine:
            head(t, f"refA/{m}_a", (LC if m.startswith("r34") else LPR)[ar, pz[m]])
            head(t, f"refA/{m}_b", LS[:, sel["stack_names"].index(f"{m}_b")])
        head(t, "refA/v2_lat", LS[:, sel["stack_names"].index("v2_lat")])
        lcorr = lab(CK / f"labels/navtest/corr_{old}", 16)
        head(t, f"old/{old} a(1)", LC[ar, oa_mine])
        head(t, f"old/{old} b(1)", lcorr[ar, oa_mine])
        head(t, f"old/{old} CK-only a (context)", LC[ar, np.argmax(score(opr), 1)])
        # ---------------- 4c file times: picks written before the labels of their own trajectories
        tp = max(os.path.getmtime(g / f) for f in ("p256_idx.npy", "p96_col.npy", "final_traj.npy", "g2_var80_traj.npy"))
        tr = max(os.path.getmtime(rd / f) for f in ("picks.npz", "final_traj.npy", "score_stack.npy"))
        tl = {k: os.path.getmtime(lsrc[k][0] / "labels.npy") for k in (f"{t}_var80", f"{t}_final5", f"{t}_stack")}
        tl["r34_var80"] = os.path.getmtime(lsrc["r34_var80"][0] / "labels.npy")
        out["times"][t] = {"gtfree_picks_written": kst(tp), "r34_picks_written": kst(tr),
                           **{f"labels_{k}": kst(v) for k, v in tl.items()},
                           "preexisting_labels_note": "raw256 (14:19:18) and cand (Oct 6) existed before the picks; "
                                                      "covered by the pick re-derivation + static scan"}
        need(tp < min(tl[f"{t}_var80"], tl[f"{t}_final5"]), f"{t} gtfree picks not older than their labels")
        need(tr < min(tl[f"{t}_stack"], tl["r34_var80"]), f"{t} r34 picks not older than their labels")
    out["headline"] = hl
    for k, (ld, tf) in lsrc.items():
        meta = json.loads((ld / "meta.json").read_text())
        rec = {"label_meta_traj_sha16": meta.get("traj_sha16"), "traj_path": meta.get("traj_path")}
        if tf is not None:
            rec["file_sha16"] = sha16(tf)
            need(rec["file_sha16"] == rec["label_meta_traj_sha16"], f"labels {k} not computed on {tf}")
        out["label_traj_sha"][k] = rec

    # ---------------- 4b static scan of the inference / selection code
    FORBID = ("label", "gt_traj", "raw256", "pdms", "gtloader")
    READS = ("load", "read_parquet", "open", "read_json", "load_bev", "for_subset", "open_memmap", "read_csv", "load_ck")

    def scan(path, funcs=None):
        src = path.read_text()
        tree = ast.parse(src)
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and (funcs is None or n.name in funcs)]
        reads, bad, keys, dsets = [], [], set(), []
        for fn in nodes:
            for n in ast.walk(fn):
                if isinstance(n, ast.Call):
                    f = n.func
                    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                    seg = ast.get_source_segment(src, n) or ""
                    if name in READS:
                        reads.append(f"{fn.name}: {seg[:120]}")
                        if any(w in seg.lower() for w in FORBID):
                            bad.append(seg[:160])
                    if name == "CKDataset":
                        kw = {k.arg: ast.get_source_segment(src, k.value) for k in n.keywords}
                        dsets.append(kw)
                if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "b":
                    sl = n.slice.value if isinstance(n.slice, ast.Index) else n.slice      # py3.8 wraps in Index
                    keys.add(sl.value if isinstance(sl, ast.Constant) else f"<expr {ast.dump(sl)[:40]}>")
        return {"functions": sorted(fn.name for fn in nodes), "reads": reads, "forbidden_reads": bad,
                "batch_keys": sorted(keys), "ckdataset_kwargs": dsets}

    sg = scan(REPO / "tools/ck/e2e2/eval_teacher_navtest_gtfree.py")
    gsrc = (REPO / "tools/ck/e2e2/eval_teacher_navtest_gtfree.py").read_text()
    gi = next(n for n in ast.walk(ast.parse(gsrc)) if isinstance(n, ast.FunctionDef) and n.name == "__getitem__")
    rets = [n.value for n in ast.walk(gi) if isinstance(n, ast.Return)]
    sg["dataset_item_keys"] = sorted({k.value for r_ in rets if isinstance(r_, ast.Dict) for k in r_.keys})
    need(not sg["forbidden_reads"], "gtfree inference reads labels / GT")
    need(set(sg["dataset_item_keys"]) == {"row", "bev", "bev_ok", "status"}, "gtfree dataset item keys")
    need(set(sg["batch_keys"]) <= {"row", "bev", "bev_ok", "status"}, "gtfree batch keys")
    sr = scan(REPO / "tools/ck/e2e2/eval_teacher_navtest_r34.py",
              {"stage_variants", "teacher_forward", "lat_decode", "stage_infer", "select_modes", "stage_select",
               "to_pool", "packed", "roots", "todo_rows", "resolve_run", "sanity_checks"})
    need(not sr["forbidden_reads"], "r34 inference / selection reads labels / GT")
    need(set(sr["batch_keys"]) <= {"rows", "bev_ok", "bev", "cand", "status"}, "r34 batch keys")
    need(all(d.get("labels") == "None" and d.get("gt") == "False" for d in sr["ckdataset_kwargs"])
         and sr["ckdataset_kwargs"], "r34 CKDataset labels=None gt=False")
    sr["note"] = ("CKDataset items always carry the packed gt_traj array; stage_infer reads only the batch keys "
                  "listed in batch_keys (gt_traj never reaches the model or the selection)")
    out["static"] = {"gtfree": sg, "r34": sr}
    out["fails"] = FAILS
    out["ok"] = not FAILS
    out["sec"] = round(time.time() - t0, 1)
    (D / "check_headline.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    # section 6 of <D>/table.md (replaced on every run)
    tbl = D / "table.md"
    if tbl.is_file():
        head_txt = tbl.read_text().split("\n## 6. 독립 재계산")[0].rstrip("\n")
        mx = max(v["absdiff"] for v in hl.values())
        sec = ["", "## 6. 독립 재계산 (tools/ck/e2e2/check_teacher_navtest_headline.py)", "",
               f"{kst()} 실행, 결과 **{'OK' if not FAILS else 'FAILED'}**, 실패 항목 {FAILS}. repo 모듈 import 없이 "
               "numpy/pandas/ast로 따로 구현. [실측]", "",
               "- 선택 재유도 (저장된 모델 출력만 사용: 확률, v2 im/final, 변형 유효성; 선택 규칙은 따로 구현) = 저장된 index: "
               + "; ".join(f"{t} " + ", ".join(f"{k} {v}/{N}" for k, v in P.items()) for t, P in out["picks"].items()),
               f"- 제출 궤적 = 고른 후보 (bitwise): {all(all(v.values()) for v in out['traj'].values())} "
               f"({sum(len(v) for v in out['traj'].values())}개 항목). 라벨 meta traj_sha16 = 현재 궤적 파일 sha: "
               f"{sum(v.get('file_sha16') == v['label_meta_traj_sha16'] for v in out['label_traj_sha'].values())}/"
               f"{len(out['label_traj_sha'])}. raw256 궤적 = 256 anchor (전 토큰).",
               f"- headline PDMS {len(hl)}개를 공식 라벨 + 고른 index로 다시 계산: metrics.json과 최대 차 {mx:.1e}.",
               "- GT/라벨 미사용: (1) 위 재유도 (라벨을 입력으로 받지 않는 규칙으로 같은 index가 나옴); (2) AST 검사: gtfree "
               "추론 코드의 파일 읽기에 label / gt_traj / raw256 / pdms 없음, dataset item 키 "
               f"{out['static']['gtfree']['dataset_item_keys']}; r34 variants/infer/select 코드의 읽기에도 없음, "
               f"CKDataset {out['static']['r34']['ckdataset_kwargs']}, batch에서 읽는 키 "
               f"{out['static']['r34']['batch_keys']} (CKDataset item에는 packed gt_traj가 항상 들어 있지만 읽지 않음); "
               "(3) 파일 시각: picks 기록 (ck2T gtfree "
               f"{out['times']['ck2T']['gtfree_picks_written'][11:19]}, ck2M gtfree "
               f"{out['times']['ck2M']['gtfree_picks_written'][11:19]}, r34 {out['times']['ck2T']['r34_picks_written'][11:19]}) "
               "< 그 궤적들의 라벨 생성 (14:24-14:35 KST). raw256 / cand 라벨은 그 전부터 있었으므로 (1)(2)로만 확인.", "",
               "| teacher | 항목 | 재계산 PDMS | metrics.json | 차 |", "|---|---|---|---|---|"]
        for k, v in hl.items():
            t, key = k.split(":", 1)
            sec.append(f"| {t} | {key} | {100 * v['recomputed']:.3f} | {100 * v['summary']:.3f} | {v['absdiff']:.1e} |")
        tbl.write_text(head_txt + "\n" + "\n".join(sec) + "\n")
    print(json.dumps({"picks": out["picks"], "traj_all_equal": {t: all(v.values()) for t, v in out["traj"].items()},
                      "headline_max_absdiff": max(v["absdiff"] for v in hl.values()), "n_headline": len(hl),
                      "times": out["times"], "fails": FAILS}, indent=1, ensure_ascii=False))
    print(f"[{kst()}] {'OK' if not FAILS else 'FAILED'} -> {D / 'check_headline.json'} ({out['sec']} s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
