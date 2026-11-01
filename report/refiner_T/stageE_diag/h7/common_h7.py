"""H7 shared loaders: per-token scores of the arms on E2's tau0 drafts, log-cluster bootstrap."""
from pathlib import Path
import numpy as np
import pandas as pd

D = Path("/home/external-user/ssd/yongjae_refiner/stageE_diag")
H7 = D / "h7"
E0R = Path("/home/external-user/ssd/yongjae_refiner/e0_teacher_refine")
MET = ["nc", "dac", "ddc", "ttc", "ep", "comfort", "pdms"]


def load_arms(with_none=True):
    s = pd.read_parquet(D / "scores.parquet")
    arms = {"A": s[s.k == 0], "S": s[s.k == 1]}
    for a, name in (("T", "R_T4"), ("M", "R_M4")) + ((("N", "R_none4"),) if with_none else ()):
        p = H7 / "e2tau0_refine" / name / "scores.parquet"
        if p.exists():
            arms[a] = pd.read_parquet(p)
    for a, name in (("E0", "E0"), ("E0T", "R_T4"), ("E0M", "R_M4"), ("E0N", "R_none4")):
        arms[a] = pd.read_parquet(E0R / name / "scores.parquet")
    out = None
    for a, df in arms.items():
        df = df.set_index("token")[MET + ["log", "nc_track", "ttc_track"]].add_prefix(f"{a}.")
        out = df if out is None else out.join(df, how="inner")
    out["log"] = out["A.log"]
    return out.reset_index()


class Boot:
    def __init__(self, logs, B=2000, seed=0):
        self.u, self.idx = np.unique(np.asarray(logs), return_inverse=True)
        rng = np.random.default_rng(seed)
        L = len(self.u)
        self.W = np.stack([np.bincount(rng.integers(0, L, L), minlength=L) for _ in range(B)]).astype(np.float64)

    def mean(self, v, mask=None):
        v = np.asarray(v, float)
        m = np.ones_like(v, bool) if mask is None else np.asarray(mask, bool)
        num = np.bincount(self.idx[m], v[m], minlength=len(self.u))
        den = np.bincount(self.idx[m], minlength=len(self.u)).astype(float)
        pt = num.sum() / max(den.sum(), 1)
        bs = (self.W @ num) / np.maximum(self.W @ den, 1e-9)
        return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    def ratio(self, num_v, den_v):
        """sum(num)/sum(den) with cluster bootstrap"""
        n = np.bincount(self.idx, np.asarray(num_v, float), minlength=len(self.u))
        d = np.bincount(self.idx, np.asarray(den_v, float), minlength=len(self.u))
        pt = n.sum() / max(d.sum(), 1e-9)
        bs = (self.W @ n) / np.maximum(self.W @ d, 1e-9)
        return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
