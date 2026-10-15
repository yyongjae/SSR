import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,"/home/external-user/yongjae/SSR")
from navsim.agents.para_ssr.refiner.decoder import FAMILY_NAME
D = Path("/home/external-user/ssd/yongjae_refiner")
out = {}
fam_rows = []
for s in ("dev", "navtest"):
    sc = pd.read_parquet(D / f"scores/{s}.parquet")
    sp = pd.read_parquet(D / f"splits/{s}.parquet")
    city = dict(zip(sp.token, sp.map_location))
    # validity + family from the packed drafts (same source eval uses)
    idx = pd.read_parquet(D / f"packed/{s}/index.parquet")
    dv = np.load(D / f"packed/{s}/draft_valid.npy", mmap_mode="r")
    fam = np.load(D / f"packed/{s}/family.npy", mmap_mode="r")
    done = np.load(D / f"packed/{s}/done.npy")
    rows = np.flatnonzero(done[:, 1] == 1)
    V = pd.DataFrame({"token": np.repeat(idx.token.values[rows], 13), "k": np.tile(np.arange(13), len(rows)),
                      "valid": np.asarray(dv[rows]).ravel()})
    m = sc[sc.k >= 0].merge(V, on=["token", "k"], how="inner")
    assert len(m) == len(V), (s, len(m), len(V))
    m["city"] = m.token.map(city)
    m["family"] = m.family.map(lambda x: FAMILY_NAME.get(int(x), str(x)) if isinstance(FAMILY_NAME, dict) else FAMILY_NAME[int(x)])
    for c in ("nc", "dac", "ddc", "ttc", "comfort"):
        m[c + "_f"] = m[c] < 1
    m["fail"] = m.nc_f | m.dac_f | m.ddc_f
    m["fail_ttc"] = m.fail | m.ttc_f
    v = m[m.valid]
    agg = lambda g: dict(n=int(len(g)), fail=round(100 * g.fail.mean(), 2), fail_ttc=round(100 * g.fail_ttc.mean(), 2),
                         nc=round(100 * g.nc_f.mean(), 2), dac=round(100 * g.dac_f.mean(), 2), ttc=round(100 * g.ttc_f.mean(), 2),
                         ddc=round(100 * g.ddc_f.mean(), 2), pdms=round(float(g.pdms.mean()), 4))
    hum = m[m.k == 0]
    out[s] = dict(n_split_tokens=len(sp), n_drafted_tokens=int(len(rows)), n_scored_tokens=int(sc.token.nunique()),
                  score_error_rows=int((sc.k < 0).sum()), valid_rate=round(100 * m.valid.mean(), 2),
                  bank_valid=agg(v), human_k0=dict(n=int(len(hum)), pdms=round(float(hum.pdms.mean()), 4),
                  nc_fail=round(100 * hum.nc_f.mean(), 2), dac_fail=round(100 * hum.dac_f.mean(), 2),
                  ttc_fail=round(100 * hum.ttc_f.mean(), 2), ep=round(float(hum.ep.mean()), 4)),
                  tokens_per_city=sp.map_location.value_counts().to_dict(),
                  city_frac=(sp.map_location.value_counts(normalize=True) * 100).round(1).to_dict(),
                  by_family={f: agg(g) for f, g in v.groupby("family")},
                  by_city={c: agg(g) for c, g in v.groupby("city")},
                  human_pdms_by_city={c: round(float(g.pdms.mean()), 4) for c, g in hum.groupby("city")})
print(json.dumps(out, indent=1))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
