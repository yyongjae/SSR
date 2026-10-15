"""Does the CURRENT code reproduce the dev artifacts (built 2026-09-28 with earlier file versions)?
drafts (decoder.py changed 17:27 after the dev bank), sdf (sdf.py changed 06:53 after the navtest-550 / at navtrain
start), pack centerline (surrogate.py changed 09-29 00:15 after the dev pack), + the 550 pre-existing navtest SDFs."""
import json, os, sys, tempfile, lzma, pickle
from pathlib import Path
import numpy as np, pandas as pd
R = "/home/external-user/yongjae/SSR"
sys.path[:0] = [R, R + "/tools/refiner"]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import make_draft_bank as MB
import score_trajectories as ST
from navsim.agents.para_ssr.refiner import sdf as SDF, data as RD
D = Path("/home/external-user/ssd/yongjae_refiner")
tmp = Path(sys.argv[1]); tmp.mkdir(parents=True, exist_ok=True)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
rng = np.random.default_rng(0)
out = {}
# --- drafts (dev)
H = MB.load_human("dev")
cfg = MB.BankConfig(lconst_ap=MB.TUNED_AP)
have = sorted(p.stem for p in (D / "drafts/dev").glob("*.npz"))
pick = [have[i] for i in rng.choice(len(have), n, replace=False)]
roots = [Path(r) for r in ST.MC_ROOTS]
mism = []
keys = ("drafts", "family", "params", "valid", "slot", "z_lon", "w_lat", "path_src")
for t in pick:
    i = H["index"][t]; log = str(H["logs"][i])
    st, mcp = MB.mc_ready(t, log, roots, 0)
    op = tmp / "drafts" / f"{t}.npz"; op.parent.mkdir(exist_ok=True, parents=True)
    r = MB._gen_one((t, log, "dev", H["traj"][i], H["path"][i], int(H["n_reg"][i]), float(H["v0"][i]), float(H["a0"][i]),
                     mcp, str(op), cfg.to_json()))
    a, b = np.load(op), np.load(D / "drafts/dev" / f"{t}.npz")
    bad = [k for k in keys if a[k].tobytes() != b[k].tobytes()]
    if bad or str(a["cfg_hash"]) != str(b["cfg_hash"]):
        mism.append((t, bad))
out["drafts_dev"] = dict(n=len(pick), mismatch=len(mism), examples=mism[:5], cfg_hash=cfg.hash)
# --- sdf (dev, navtrain subset) and navtest pre-existing 550
def sdf_cmp(toks, logs, subset, roots):
    m = []
    for t, lg in zip(toks, logs):
        p = RD.locate_metric_cache(t, lg, roots)
        with lzma.open(p, "rb") as f:
            mc = pickle.load(f)
        field, info = SDF.build_sdf_from_metric_cache(mc)
        op = tmp / "sdf" / subset / f"{t}.npz"; op.parent.mkdir(parents=True, exist_ok=True)
        SDF.save_sdf(op, field, token=t, ego_xyh=info["ego_xyh"], info=info)
        a, b = SDF.load_sdf(op), SDF.load_sdf(D / "sdf" / subset / f"{t}.npz")
        if a.tobytes() != b.tobytes():
            m.append((t, float(np.nanmax(np.abs(a.astype(np.float32) - b.astype(np.float32))))))
    return m
dev = pd.read_parquet(D / "splits/dev.parquet")
sp = dev.sample(min(n // 3, len(dev)), random_state=1)
m = sdf_cmp(sp.token, sp.log, "navtrain", RD.MC_ROOTS)
out["sdf_dev"] = dict(n=len(sp), mismatch=len(m), examples=m[:5])
nt = pd.read_parquet(D / "splits/navtest.parquet")
old = json.loads("[" + ",".join(l for l in open(D / "sdf/navtest/_build/stats.jsonl").read().splitlines()[:550]) + "]")
old_t = [r["token"] for r in old if r["status"] == "built"]
lg = dict(zip(nt.token, nt.log))
pk = [old_t[i] for i in rng.choice(len(old_t), min(n // 3, len(old_t)), replace=False)]
m = sdf_cmp(pk, [lg[t] for t in pk], "navtest", RD.MC_ROOTS)
out["sdf_navtest_preexisting550"] = dict(n=len(pk), mismatch=len(m), examples=m[:5])
# --- centerline (dev pack)
P = RD.PackedSplit("dev")
rows = rng.choice(np.flatnonzero(np.asarray(P.done[:, RD.PART_ID["centerline"]]) == 1), n, replace=False)
m = []
for r in rows:
    t, lg_ = str(P.index.token[r]), str(P.index.log[r])
    c = RD.centerline_samples(RD.load_metric_cache(RD.locate_metric_cache(t, lg_)))
    if not (c["cl_xy"].tobytes() == np.asarray(P.arrays["cl_xy"][r]).tobytes() and int(c["cl_n"]) == int(P.arrays["cl_n"][r])):
        m.append(t)
out["centerline_dev_pack"] = dict(n=len(rows), mismatch=len(m), examples=m[:5])
print(json.dumps(out, indent=1))
json.dump(out, open(tmp / "repro_check.json", "w"), indent=1)
