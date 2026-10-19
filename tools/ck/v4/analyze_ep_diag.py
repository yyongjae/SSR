#!/usr/bin/env python
"""Why is the refiner-v4 teachers' EP worse than CK2's?  Analysis of tools/ck/v4/diag_val_teachers.py dumps (CPU).

  cd /workspace/yongjae/SSR-v4 && PYTHONPATH=$PWD OMP_NUM_THREADS=4 /venv/ssr/bin/python tools/ck/v4/analyze_ep_diag.py \
      --root /home/external-user/ssd/yongjae_refiner/ck/v4/analysis/ep_diag
Facts used: official EP = EP_dec x NC x DAC x DDC (EP_dec = the progress ratio, ep_target.decoupled_ep), so a failing
candidate (NC, DAC or DDC < 1) has EP ~ 0.
  A  EP quality per teacher: BCE vs the official EP target, all / pass (mult = 1) / fail (mult < 1) candidates, the
     AUC of p_EP for 'fails' vs 'passes', BCE vs EP_dec on passing candidates, per variant family.
  B  v4 P_ref head: error of the estimate; the v4 EP BCE with the TRUE P_ref fed to the EP head (no retraining).
  C  probes (frozen features of the first n_feat tokens, split by log 70 / 30, MLP + BCE, 5 seeds averaged):
     v4 EP-head input (h 96 = the trained size, h 256), v4 EP-head input + the teacher pair's NC / DAC / TTC logits
     (detached hazard evidence), CK2 score-head input h (h 256): held-out EP BCE / AUC(fail).
  D  safety keys of the pair (each key from its own teacher): AUC per family (NC / TTC DET, DAC MAP).
Output: <root>/<split>/analysis.json (+ printed tables).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

VN = ("id", "a-1.0", "a-0.5", "a+0.5", "l-0.5", "l+0.5")


def bce(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))) if len(y) else float("nan")


def auc(score, pos):
    from sklearn.metrics import roc_auc_score
    pos = np.asarray(pos, bool)
    return float(roc_auc_score(pos, score)) if 0 < pos.sum() < len(pos) else float("nan")


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def probe(X, y, logs, hidden, seeds=5, epochs=40):
    """MLP probe (Linear-GELU-Linear, BCE on soft targets), logs split 70 / 30 -> held-out BCE, AUC(fail), mean."""
    import torch
    ul = np.unique(logs)
    out = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        test_logs = set(rng.choice(ul, size=max(1, int(0.3 * len(ul))), replace=False))
        te = np.array([l in test_logs for l in logs])
        torch.manual_seed(s)
        mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-4
        Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        tr = np.flatnonzero(~te)
        for ep in range(epochs):
            rng.shuffle(tr)
            for i in range(0, len(tr), 2048):
                b = tr[i:i + 2048]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b])[:, 0], yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            p = torch.sigmoid(net(Xt[te])[:, 0]).numpy()
        out.append((bce(p, y[te]), auc(-p, y[te] < 0.05)))
    a = np.asarray(out)
    return {"bce": float(a[:, 0].mean()), "bce_sd": float(a[:, 0].std()), "auc_fail": float(a[:, 1].mean())}


def main(argv=None):
    from navsim.agents.para_ssr.ck import constants as Cn
    from navsim.agents.para_ssr.ck.ep_target import decoupled_ep
    from tools.ck.e2e2 import eval_teacher_navtest_r34 as R
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--pairs", default="ck2T10,ck2M10;v4T10_haz,v4M10_haz")
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root) / a.split
    z = np.load(root / "labels96.npz")
    lab, ok = z["lab96"].astype(np.float64), z["ok96"].astype(bool)
    N = lab.shape[0]
    pdir, tdf, _ = R.packed() if a.split == "navtest" else (None, None, None)
    if a.split == "navtest":
        logs_tok = tdf.log.to_numpy()[z["rows"]]
    else:
        import pandas as pd
        logs_tok = pd.read_parquet(R.CK_DATA / "packed" / a.split / "tokens.parquet").log.to_numpy()[z["rows"]]
    L = Cn.LBL
    ok = ok & np.isfinite(lab).all(-1)
    ep = lab[..., L["ep"]]
    dec = decoupled_ep(lab)
    mult = lab[..., L["nc"]] * lab[..., L["dac"]] * lab[..., L["ddc"]]
    fail = mult < 1
    vt = np.broadcast_to(np.arange(96) % 6, (N, 96))
    I = {k: Cn.CK_KEYS.index(k) for k in Cn.CK_KEYS}
    res = {"split": a.split, "n_tok": int(N), "n_cand": int(ok.sum()), "fail_frac": float(fail[ok].mean()),
           "teachers": {}}
    teachers = sorted({t for p in a.pairs.split(";") for t in p.split(",")})
    P = {}
    for t in teachers:
        d = root / t
        if not (d / "logit96.npy").is_file():
            continue
        lg = np.load(d / "logit96.npy").astype(np.float64)
        P[t] = sig(lg)
        pe = P[t][..., I["ep"]]
        m = ok
        r = {"bce_all": bce(pe[m], ep[m]), "bce_pass": bce(pe[m & ~fail], ep[m & ~fail]),
             "bce_fail": bce(pe[m & fail], ep[m & fail]), "auc_fail_from_ep": auc(-pe[m], fail[m]),
             "mean_p_ep_fail": float(pe[m & fail].mean()), "mean_p_ep_pass": float(pe[m & ~fail].mean()),
             "corr_pass": float(np.corrcoef(pe[m & ~fail], dec[m & ~fail])[0, 1]),
             "by_family": {}}
        for v, nm in enumerate(VN):
            mv = m & (vt == v)
            r["by_family"][nm] = {"bce": bce(pe[mv], ep[mv]), "bce_pass": bce(pe[mv & ~fail], ep[mv & ~fail]),
                                  "bce_fail": bce(pe[mv & fail], ep[mv & fail]),
                                  "fail_frac": float(fail[mv].mean())}
        if (d / "ep_logit_oracle_pref.npy").is_file():
            po = sig(np.load(d / "ep_logit_oracle_pref.npy"))
            pl = np.load(d / "pref_logp.npy").astype(np.float64)
            pt = np.log1p(np.clip(lab[:, 0, L["pdm_progress_eff"]], 0, None))
            fin = np.isfinite(pl)
            r["pref"] = {"mae_log1p": float(np.abs(pl - pt)[fin].mean()),
                         "median_rel_err": float(np.median(np.abs(np.expm1(pl) - np.expm1(pt))[fin] /
                                                           np.maximum(np.expm1(pt)[fin], 1.0))),
                         "corr_log1p": float(np.corrcoef(pl[fin], pt[fin])[0, 1])}
            r["oracle_pref"] = {"bce_all": bce(po[m], ep[m]), "bce_pass": bce(po[m & ~fail], ep[m & ~fail]),
                                "bce_fail": bce(po[m & fail], ep[m & fail])}
        res["teachers"][t] = r
    # pairs: each key from its own teacher, EP mean (= the KD sources)
    res["pairs"] = {}
    for pr in a.pairs.split(";"):
        dt, mt = pr.split(",")
        if dt not in P or mt not in P:
            continue
        pd_, pm = P[dt], P[mt]
        pe = 0.5 * (pd_[..., I["ep"]] + pm[..., I["ep"]])
        m = ok
        q = {"ep_bce_all": bce(pe[m], ep[m]), "ep_bce_pass": bce(pe[m & ~fail], ep[m & ~fail]),
             "ep_bce_fail": bce(pe[m & fail], ep[m & fail]), "ep_auc_fail": auc(-pe[m], fail[m]),
             "auc": {}}
        for key, src in (("nc", pd_), ("ttc", pd_), ("dac", pm)):
            y = lab[..., L[key]] < 1
            q["auc"][key] = {"all": auc(-src[..., I[key]][m], y[m])}
            for v, nm in enumerate(VN):
                mv = m & (vt == v)
                q["auc"][key][nm] = auc(-src[..., I[key]][mv], y[mv])
        res["pairs"][pr] = q
    # probes
    if not a.no_probe:
        res["probes"] = {}
        for t in teachers:
            d = root / t
            if not (d / "feat.npy").is_file():
                continue
            F = np.load(d / "feat.npy").astype(np.float32)
            nf = F.shape[0]
            m = ok[:nf].reshape(-1)
            X = F.reshape(nf * 96, -1)[m]
            y = ep[:nf].reshape(-1)[m]
            logs = np.repeat(logs_tok[:nf], 96)[m]
            meta = json.loads((d / "meta.json").read_text())
            pr = {}
            hid = 96 if meta["v4"] else 256
            pr[f"own_input_h{hid}"] = probe(X, y, logs, hid)
            if meta["v4"]:
                pr["own_input_h256"] = probe(X, y, logs, 256)
                # + the detached hazard evidence of the pair this teacher belongs to (NC, TTC from the DET teacher,
                # DAC from the MAP teacher: what an EP head could be given at no parameter cost)
                for p_ in a.pairs.split(";"):
                    dt, mt = p_.split(",")
                    if t in (dt, mt) and dt in P and mt in P:
                        lg = lambda x: np.log(np.clip(x, 1e-6, 1)) - np.log(np.clip(1 - x, 1e-6, 1))
                        hz = np.stack([lg(P[dt][..., I["nc"]]), lg(P[dt][..., I["ttc"]]), lg(P[mt][..., I["dac"]])],
                                      -1)[:nf].reshape(nf * 96, 3)[m]
                        pr["own_input_plus_pair_hazards_h96"] = probe(np.concatenate([X, hz], 1), y, logs, 96)
                        own = [i for i, k in ((0, "nc"), (1, "ttc"), (2, "dac"))
                               if (k in ("nc", "ttc")) == (t == dt)]
                        pr["own_input_plus_own_hazards_h96"] = probe(np.concatenate([X, hz[:, own]], 1), y, logs, 96)
            res["probes"][t] = pr
    out = root / "analysis.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
