"""Independent GT extraction for the ReSMap axis verification (verifier script; does NOT use resmap_cache.py, sdf.py or
packed/*).  Reads the NAVSIM metric cache pickle directly and rasterises, in the N frame (rear axle at t0, x forward,
y left), on a padded 0.32 m grid in S orientation:
  rows i: x = X0 + (i + .5) * 0.32, X0 = -3.84  (124 rows: 12 pad cells behind 0 m and beyond 32 m)
  cols j: y = Y0 - (j + .5) * 0.32, Y0 = +33.92 (212 cols: 6 pad cells each side), col 0 = LEFT
  dac  : union of ROADBLOCK / INTERSECTION / DRIVABLE_AREA / CARPARK_AREA polygons (official DAC layers)
  lane : union of LANE / LANE_CONNECTOR polygons
  isec : INTERSECTION polygons
  cl   : route centerline (metric_cache.centerline) in N, densified to 0.2 m
Output: one npz per token in OUT.
"""
import lzma
import math
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/home/external-user/yongjae/SSR")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

OUT = Path(sys.argv[1])
X0, Y0, C, NI, NJ = -3.84, 33.92, 0.32, 124, 212
xs = X0 + (np.arange(NI) + 0.5) * C
ys = Y0 - (np.arange(NJ) + 0.5) * C
XX, YY = np.meshgrid(xs, ys, indexing="ij")


def work(args):
    token, path, subset = args
    out = OUT / f"{token}.npz"
    if out.exists():
        return token
    mc = pickle.load(lzma.open(path, "rb"))
    ra = mc.ego_state.rear_axle
    x0, y0, h0 = float(ra.x), float(ra.y), float(ra.heading)
    c, s = math.cos(h0), math.sin(h0)
    GX = x0 + c * XX - s * YY
    GY = y0 + s * XX + c * YY
    d = mc.drivable_area_map
    types = [str(t).split(".")[-1] for t in d._map_types]
    roi = shapely.box(-60, -60, 60, 60)
    roi_g = shapely.affinity.affine_transform(roi, [c, -s, s, c, x0, y0])

    def union(names):
        g = [d._geometries[i] for i, t in enumerate(types) if t in names]
        g = [q for q in g if q.intersects(roi_g)]
        return unary_union(g) if g else shapely.Polygon()

    dac = shapely.contains_xy(union({"ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA", "CARPARK_AREA"}), GX, GY)
    lane = shapely.contains_xy(union({"LANE", "LANE_CONNECTOR"}), GX, GY)
    isec = shapely.contains_xy(union({"INTERSECTION"}), GX, GY)
    arr = np.asarray(mc.centerline._states_se2_array[:, :2], np.float64)
    ls = shapely.LineString(arr)
    n = max(2, int(ls.length / 0.2))
    pts = np.asarray([ls.interpolate(t).coords[0] for t in np.linspace(0, ls.length, n)])
    dx, dy = pts[:, 0] - x0, pts[:, 1] - y0
    cl = np.stack([c * dx + s * dy, -s * dx + c * dy], 1)
    keep = (cl[:, 0] > -5) & (cl[:, 0] < 40) & (np.abs(cl[:, 1]) < 36)
    v = mc.ego_state.dynamic_car_state.rear_axle_velocity_2d
    np.savez_compressed(out, dac=np.packbits(dac), lane=np.packbits(lane), isec=np.packbits(isec),
                        cl=cl[keep].astype(np.float32), speed=np.float32(math.hypot(v.x, v.y)),
                        subset=subset, ego=np.array([x0, y0, h0]))
    return token


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260929)
    man = pd.read_parquet("/home/external-user/ssd/yongjae_refiner/metric_cache/manifest.parquet")
    man = man[man.ok].set_index("token")
    tr = pd.concat([pd.read_parquet(f"/home/external-user/ssd/yongjae_refiner/splits/{s}_trainlogs.parquet")
                    for s in ("train", "dev")])
    tr = tr[tr.token.isin(man.index)]
    pick_tr = rng.choice(tr.token.values, int(sys.argv[2]), replace=False)
    nt = [l.strip() for l in open("/home/external-user/yongjae/SSR/data/exp/metric_cache/metadata/"
                                  "metric_cache_metadata_node_0.csv").readlines()[1:]]
    nt = {Path(p).parent.name: p for p in nt}
    pick_nt = rng.choice(sorted(nt), int(sys.argv[3]), replace=False)
    jobs = [(t, man.loc[t, "path"], "navtrain") for t in pick_tr] + [(t, nt[t], "navtest") for t in pick_nt]
    with Pool(4) as p:
        for i, _ in enumerate(p.imap_unordered(work, jobs, chunksize=4)):
            if i % 100 == 0:
                print(i, flush=True)
    pd.DataFrame(jobs, columns=["token", "path", "subset"]).to_parquet(OUT / "_jobs.parquet")


if __name__ == "__main__":
    main()
