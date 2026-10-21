"""If the vector map looks right, why is DAC 96.07?

Two readings fit "the map prediction looks fine but DAC is low". Either the ego leaves the road by a
lot where the map is wrong, or it leaves it by a little everywhere because nothing in the model says
"my footprint must stay inside". The two call for completely different work, and the penetration
depth separates them: a metre-scale tail says perception, a decimetre-scale one says margin.

Measured over every navtest scene the official evaluation scored DAC = 0, against the scorer's own
``drivable_area_map`` (the polygons ``pdm_scorer.py:318`` queries):

    depth          how far the worst ego-footprint corner is outside the drivable polygon
    first step     when it first leaves
    corners        how many of the four are outside at that step

    python tools/plan_v2/dac_depth.py [--limit 0]
"""
import argparse

import numpy as np

import viz_seen_but_hit as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import pandas as pd
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union

    df = pd.read_csv(V.CSV)
    df = df[df.token != "average"]
    bad = df[df.drivable_area_compliance == 0].token.tolist()
    if args.limit:
        bad = bad[: args.limit]
    d = np.load(V.DUMP)
    at = {str(t): i for i, t in enumerate(d["tokens"])}
    anchors = np.load(V.ANCHORS).astype(np.float32)

    rows = []
    for n, tok in enumerate(bad):
        if tok not in at:
            continue
        _, polys, _ = V.load_cache(tok)
        if not polys:
            continue
        area = unary_union([Polygon(p) for p in polys if len(p) >= 4])
        i = at[tok]
        k = int(np.asarray(d["plan_final_rewards"][i], np.float32).argmax())
        tr = (anchors[k] + np.asarray(d["trajectory_offset"][i][k], np.float32)).astype(np.float64)
        ego = V.ego_corners(tr[:, 0], tr[:, 1], tr[:, 2])                  # [8, 4, 2]
        depth = np.zeros((8, 4))
        for t in range(8):
            for c in range(4):
                pt = Point(ego[t, c])
                if not area.contains(pt):
                    depth[t, c] = pt.distance(area)
        out = depth.max(axis=1)
        if not (out > 0).any():
            continue
        first = int(np.argmax(out > 0))
        rows.append(dict(token=tok, depth=float(out.max()), first_step=first,
                         corners=int((depth[first] > 0).sum()),
                         depth_at_first=float(out[first])))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(bad)}", flush=True)

    r = pd.DataFrame(rows)
    print(f"\nDAC = 0 장면 {len(bad)} 중 실제로 폴리곤 밖으로 나간 것 {len(r)}")
    q = r.depth.quantile([.25, .5, .75, .9, .95]).round(2)
    print(f"\n최대 침범 깊이 (m):  25% {q[.25]}   중앙값 {q[.5]}   75% {q[.75]}   90% {q[.9]}   95% {q[.95]}")
    for lo, hi, name in [(0, 0.25, "0.25 m 이하"), (0.25, 0.5, "0.25~0.5 m"), (0.5, 1.0, "0.5~1 m"),
                         (1.0, 2.0, "1~2 m"), (2.0, 1e9, "2 m 초과")]:
        m = (r.depth > lo) & (r.depth <= hi)
        print(f"  {name:12s} {m.sum():4d} 장면 ({100 * m.mean():5.1f}%)")
    print(f"\n처음 벗어나는 시점 중앙값 {0.5 * (r.first_step.median() + 1):.1f} s")
    print(f"그 시점에 밖으로 나간 모서리 개수: " +
          "  ".join(f"{c}개 {100 * (r.corners == c).mean():.0f}%" for c in (1, 2, 3, 4)))
    r.to_csv("data/analysis/dac_depth.csv", index=False)
    print("\n-> data/analysis/dac_depth.csv")


if __name__ == "__main__":
    main()
