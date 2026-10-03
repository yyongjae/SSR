"""(a) e2e_side of the 100 sampled new tokens re-computed from the metric cache (build_e2e_side.compute_side) -> bit
equality; (b) e2e.GTLoader.load (the E2E target builder's GT path) over ALL 85,109 list tokens -> ref_gt_ok counts."""
import json, os, sys, time
from multiprocessing import get_context
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools/refiner"))
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = Path("/home/external-user/ssd/yongjae_refiner")
_L = {}


def side_one(a):
    import build_e2e_side as BS
    from navsim.agents.para_ssr.refiner import data as RD
    tok, log = a
    d = BS.compute_side(RD.load_metric_cache(RD.locate_metric_cache(tok, log)))
    with np.load(BS.side_path(tok)) as z:
        st = {k: z[k] for k in z.files}
    eq = all(np.array_equal(np.asarray(d[k]), st[k]) for k in d) and set(d) == set(st)
    return tok, eq, float(st["p_pdm"]), int(st["cl_n"])


def gt_chunk(toks):
    from navsim.agents.para_ssr.refiner.e2e import GTLoader
    if "L" not in _L:
        _L["L"] = GTLoader(DATA)
    L = _L["L"]
    out = []
    for t in toks:
        o = L.load(t)
        out.append((t, bool(o["ref_gt_ok"]), int(o["ref_obj_n"]), float(o["ref_p_pdm"]), int(o["ref_cl_n"]),
                    float(o["ref_sdf"].float().abs().max())))
    return out


if __name__ == "__main__":
    ctx = get_context("fork")
    s = pd.read_parquet(HERE / "sample100.parquet")
    t0 = time.time()
    with ctx.Pool(4) as p:
        side = list(p.imap_unordered(side_one, list(zip(s.token, s.log)), chunksize=4))
    sd = pd.DataFrame(side, columns=["token", "side_equal", "p_pdm", "cl_n"])
    print("side recompute", sd.side_equal.value_counts().to_dict(), "p_pdm range", sd.p_pdm.min(), sd.p_pdm.max(),
          "cl_n min", sd.cl_n.min(), f"{time.time()-t0:.0f}s", flush=True)
    e = pd.read_parquet(DATA / "splits/e2e_train_trainlogs.parquet")
    toks = list(e.token)
    chunks = [toks[i:i + 500] for i in range(0, len(toks), 500)]
    rows = []
    t0 = time.time()
    with ctx.Pool(6) as p:
        for k, r in enumerate(p.imap_unordered(gt_chunk, chunks), 1):
            rows += r
            if k % 20 == 0:
                print(k, len(chunks), f"{time.time()-t0:.0f}s", flush=True)
    g = pd.DataFrame(rows, columns=["token", "ok", "obj_n", "p_pdm", "cl_n", "sdf_absmax"])
    g.to_parquet(HERE / "gtloader_all.parquet", index=False)
    sm = dict(side_recompute_equal=int(sd.side_equal.sum()), side_n=len(sd),
              gt_n=len(g), gt_ok=int(g.ok.sum()), gt_not_ok=g[~g.ok].token.tolist()[:20],
              obj_n_eq_Amax=int((g.obj_n >= 800).sum()), obj_n_max=int(g.obj_n.max()), obj_n_zero=int((g.obj_n == 0).sum()),
              p_pdm_nonfinite=int((~np.isfinite(g.p_pdm)).sum()), p_pdm_min=float(g.p_pdm.min()), p_pdm_max=float(g.p_pdm.max()),
              p_pdm_zero=int((g.p_pdm == 0).sum()), cl_n_lt2=int((g.cl_n < 2).sum()), sdf_absmax_max=float(g.sdf_absmax.max()),
              sdf_absmax_zero=int((g.sdf_absmax == 0).sum()), wall_s=time.time() - t0)
    json.dump(sm, open(HERE / "side_gt_summary.json", "w"), indent=1)
    print(json.dumps(sm, indent=1))
