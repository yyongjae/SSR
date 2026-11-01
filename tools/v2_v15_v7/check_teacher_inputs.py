#!/usr/bin/env python
"""v2_v15_v7 setup check (report 132 G1-4): the refiner teachers and the v15 BEV KD read the SAME teacher BEVs.

With the refiner on, kd_bev_0 / kd_bev_1 come from CK2E2ETargetBuilder (refiner.e2e.GTLoader: refiner.data.TeacherCache
'T' and refiner.resmap_cache.ResmapCache 'M', load_bev(s_grid=True)); km's 64 m port reads them through
distill.teacher_store.DistillTeacherTargetBuilder (TeacherFeatureStore bevfusion flip_w=True / resmap flip_w=False).
Both must give the same [256, 100, 100] S grid (row = forward 0..64 m, col 0 = 32 m left) on every token, else the
v15 MAP / DET streams would be read in another orientation than km's v15 run.  CPU only.

  CK_BEV64=1 PYTHONPATH=$PWD /venv/ssr/bin/python tools/v2_v15_v7/check_teacher_inputs.py --n 24
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--split", default="navtrain_train", choices=["navtrain_train", "navtrain_val"])
    a = ap.parse_args()
    assert os.environ.get("CK_BEV64") == "1", "run with CK_BEV64=1 (the v2_v15_v7 refiner grid)"
    from navsim.agents.para_ssr.refiner import data as RD
    from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache
    from navsim.agents.para_ssr.distill.teacher_store import DistillTeacherTargetBuilder

    toks = pd.read_parquet(Path(REPO) / "resources/km_corridor/packed" / a.split / "tokens.parquet").token.astype(str)
    toks = toks.to_numpy()[np.linspace(0, len(toks) - 1, a.n).astype(int)]
    ref = [RD.TeacherCache.for_subset("navtrain"), ResmapCache.for_subset("navtrain")]
    km = DistillTeacherTargetBuilder(SimpleNamespace(distill_feature_root="/workspace/teacher_cache", embed_dims=256,
                                                     bev_h=100, bev_w=100, distill_cache_size=(100, 100)))
    stores = km._open()
    worst = [0.0, 0.0]
    for t in toks:
        for i in range(2):
            x = np.asarray(ref[i].load_bev(t, s_grid=True), np.float32)
            y = np.asarray(stores[i].load_one_token(t), np.float32)
            assert x.shape == y.shape == (256, 100, 100), (t, i, x.shape, y.shape)
            worst[i] = max(worst[i], float(np.abs(x - y).max()))
    print(f"{len(toks)} {a.split} tokens: max |refiner reader - km v15 reader| DET {worst[0]:.3g}, MAP {worst[1]:.3g}")
    if max(worst) > 0:
        raise SystemExit("MISMATCH: the refiner teachers and v15 would read different teacher BEVs")
    print("OK: identical teacher BEVs")


if __name__ == "__main__":
    main()
