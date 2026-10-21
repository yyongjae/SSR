"""In a collision, did the model's OWN motion prediction say the way was clear?

The detection is right at t=0 in these scenes, so the tempting reading is "it saw and ignored". But
the planner does not act on boxes at t=0, it acts on where it thinks they will be. This measures
that directly, over every navtest scene the official evaluation scored NC = 0:

    predicted travel   how far our motion head said the struck agent would move over 4 s
    actual travel      how far it actually moved, from the metric cache observation
    predicted gap      distance between our predicted position of that agent at the failure step
                       and our own ego footprint there

A large predicted gap means the planner had a self-consistent reason: its own prediction cleared the
way. A near-zero one means the planner drove into a collision it had already predicted.

    python tools/plan_v2/why_it_hit.py [--limit 0]
"""
import argparse

import numpy as np

import viz_seen_but_hit as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--det-thr", type=float, default=0.3)
    args = ap.parse_args()
    import pandas as pd

    df = pd.read_csv(V.CSV)
    df = df[df.token != "average"]
    bad = df[df.no_at_fault_collisions == 0].token.tolist()
    if args.limit:
        bad = bad[: args.limit]
    d = np.load(V.DUMP)
    at = {str(t): i for i, t in enumerate(d["tokens"])}
    anchors = np.load(V.ANCHORS).astype(np.float32)
    rows = []
    for n, tok in enumerate(bad):
        f = V.REC / f"{tok}.npz"
        mz = V.MOTION / f"{tok}.npz"
        if not (f.exists() and mz.exists() and tok in at):
            continue
        z = np.load(f, allow_pickle=True)
        gt = V.ssr_box_to_nav(z["det_gt_boxes"]) if len(z["det_gt_boxes"]) else None
        if gt is None or not len(gt):
            continue
        gtc = V.box_corners(gt)
        i = at[tok]
        k = int(np.asarray(d["plan_final_rewards"][i], np.float32).argmax())
        tr = (anchors[k] + np.asarray(d["trajectory_offset"][i][k], np.float32)).astype(np.float64)
        ego = V.ego_corners(tr[:, 0], tr[:, 1], tr[:, 2])
        gaps = np.array([[V.poly_gap(e, g) for g in gtc] for e in ego])
        j = int(gaps.min(0).argmin())
        if gaps[:, j].min() > 0.5:
            continue
        t = int(gaps[:, j].argmin())
        m0 = np.load(mz)
        mb = m0["boxes"].astype(np.float64)
        if not len(mb):
            continue
        pxy = V.ssr_xy_to_nav(mb[:, :2])
        q = int(np.linalg.norm(pxy - gt[j, :2], axis=1).argmin())
        seen = float(np.linalg.norm(pxy[q] - gt[j, :2]))
        if seen > 2.0:                      # the struck agent was not detected: a perception miss
            rows.append(dict(token=tok, t=t, detected=False))
            continue
        pf = V.ssr_xy_to_nav(mb[:, None, None, :2] + np.cumsum(m0["traj"].astype(np.float64), axis=2))
        b = int(m0["mode"][q])
        pbox = gt[j].copy()
        pbox[:2] = pf[q, b, t]
        _, _, fut = V.load_cache(tok)
        real = np.nan
        if fut and fut[0]:
            cen0 = np.array([p.mean(0) for p in fut[0]])
            g = int(np.linalg.norm(cen0 - gt[j, :2], axis=1).argmin())
            if g < len(fut[t]):
                real = float(np.linalg.norm(np.asarray(fut[t][g]).mean(0) - cen0[g]))
        rows.append(dict(token=tok, t=t, detected=True,
                         pred_travel=float(np.linalg.norm(pf[q, b, -1] - pxy[q])),
                         real_travel=real,
                         pred_gap=V.poly_gap(ego[t], V.box_corners(pbox[None])[0]),
                         true_gap=float(gaps[t, j])))
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(bad)}", flush=True)

    r = pd.DataFrame(rows)
    det = r[r.detected]
    print(f"\nNC = 0 장면 {len(bad)} 중 분석 가능 {len(r)}, 충돌 대상을 검출한 경우 {len(det)} "
          f"({100 * len(det) / max(len(r), 1):.0f}%)")
    if not len(det):
        return
    print(f"\n검출한 {len(det)} 장면에서, 우리 모션 예측 기준으로 그 시점에")
    for lo, hi, name in [(-1, 0.001, "이미 충돌이라고 예측"), (0.001, 1.0, "1 m 이내"),
                         (1.0, 3.0, "1~3 m 여유"), (3.0, 1e9, "3 m 이상 여유")]:
        m = (det.pred_gap > lo) & (det.pred_gap <= hi)
        print(f"  {name:22s} {m.sum():4d} 장면 ({100 * m.mean():5.1f}%)")
    print(f"\n예측 이동거리 중앙값 {det.pred_travel.median():.2f} m,  실제 {det.real_travel.median():.2f} m")
    over = det.pred_travel - det.real_travel
    print(f"  예측이 실제보다 큰(비켜갈 거라 본) 장면 {100 * (over > 1.0).mean():.1f}%,  "
          f"작은 장면 {100 * (over < -1.0).mean():.1f}%")
    print(f"  |예측 - 실제| 중앙값 {over.abs().median():.2f} m")
    det.to_csv("data/analysis/why_it_hit.csv", index=False)
    print("\n-> data/analysis/why_it_hit.csv")


if __name__ == "__main__":
    main()
