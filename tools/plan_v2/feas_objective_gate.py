"""Is the corridor objective degenerate in the field, and does a listwise one fix it?

The `pred` arm collapsed: over 849 optimizer steps its cost under its OWN field fell 0.199 -> 0.100
while the cost of the SAME trajectories under the ground-truth field ROSE 10.1 -> 15.9, and
field_mae_near rose 1.01 -> 1.42 m.  The cause is structural, not a hyperparameter:
`feasibility.py:291` uses the predicted field undetached, so

    c_k = sum_{t,c} softplus((margin - Phi(corner)) / beta) * beta

is monotone decreasing in Phi.  Raising the field everywhere lowers the cost without moving a single
waypoint, so gradient descent takes that route.

This script measures it, on real geometry, through the REAL cost functions (imported, not
reimplemented), and tests the proposed replacement.  Two perturbations of the ground-truth field,
each standing for one way a head can cheat:

    lift      Phi + delta                       say everything is further inside than it is
    flatten   (1-a) Phi + a mean(Phi)           lose the boundary, keep the average

and two objectives:

    absolute   mean_k c_k                       what the arm trains now
    listwise   CE(softmax(-c/tau) || p*)        proposed; p* uniform over the DAC-feasible anchors

The absolute objective must fall monotonically under `lift` -- that is the degeneracy, stated as a
number.  The listwise one must have an interior minimum: as delta -> +inf every c_k -> 0, softmax
goes uniform and CE -> log K; as delta -> -inf the costs diverge, softmax concentrates on one
candidate and the other feasible candidates' log q -> -inf.  Both ends blow up.  Whether the minimum
actually sits near delta = 0 on real scenes is what has to be checked rather than argued.

    PYTHONPATH=~/kyungmin/navsim_v2 python tools/plan_v2/extract_drivable_polys.py --out p.npz
    python tools/plan_v2/feas_objective_gate.py --polys p.npz [--scenes 60] [--tau 0.5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/plan_v2"))

PC_RANGE = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)
BEV_H, BEV_W = 50, 100          # the student's grid, deliberately: this is where the cheating happens
MARGIN, BETA, CLIP = 0.4, 0.2, 8.0


def scene_field(polys):
    """Ground-truth drivable SDF on the student grid, from the scorer's own polygons."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from navsim.agents.para_ssr.feasibility import rasterize_bev_sdf

    good = []
    for p in polys:
        try:
            g = Polygon(p)
            if g.is_valid and not g.is_empty:
                good.append(g)
        except Exception:
            continue
    if not good:
        return None
    return rasterize_bev_sdf(unary_union(good), PC_RANGE, BEV_H, BEV_W, CLIP)


def costs(field, traj):
    """[H, W] field, [K, T, 3] anchors -> [K] cost, through the model's own functions."""
    from navsim.agents.para_ssr.feasibility import corner_sdf, cost_from_sdf

    f = torch.as_tensor(field, dtype=torch.float32)[None]
    sdf = corner_sdf(f, torch.as_tensor(traj, dtype=torch.float32)[None], PC_RANGE)
    return cost_from_sdf(sdf, MARGIN, BETA)[0]


def objectives(c, dac, tau):
    """All candidate field objectives at once, on one scene.  None if the scene cannot rank
    (no feasible or no infeasible anchor)."""
    n = dac.sum()
    if n == 0 or n == len(dac):
        return None
    d = torch.as_tensor(dac, dtype=torch.float32)
    feas, infeas = d > 0.5, d <= 0.5
    out = {"absolute": float(c.mean())}

    # (1) listwise CE against p* uniform over the feasible set
    out["listwise"] = float(-((d / float(n)) * torch.log_softmax(-c / tau, dim=-1)).sum())

    # (2) pairwise logistic on the RAW cost: infeasible must cost more than feasible.  Shift-
    # invariant in the cost but not scale-invariant, so pushing the field DOWN (inventing
    # obstacles everywhere) amplifies every gap -- the mirror degeneracy, and the table should
    # show it.
    gap = c[infeas][:, None] - c[feas][None, :]              # want > 0
    out["pair_raw"] = float(F_softplus(-gap / tau).mean())

    # (3) the same pairwise loss on the within-scene standardised cost.  Both a global lift and a
    # global scaling of the field are removed by construction, so the only way to lower it is to
    # order the candidates correctly.
    ch = (c - c.mean()) / c.std().clamp_min(1e-3)
    gap_n = ch[infeas][:, None] - ch[feas][None, :]
    out["pair_norm"] = float(F_softplus(-gap_n / 1.0).mean())

    # (4) separation actually achieved, as a scale-free number a reader can check: AUC of
    # "cost ranks infeasible above feasible".
    out["auc"] = float((gap > 0).float().mean() + 0.5 * (gap == 0).float().mean())
    return out


def F_softplus(x):
    return torch.nn.functional.softplus(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polys", required=True, help="output of extract_drivable_polys.py")
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()

    z = np.load(args.polys, allow_pickle=True)
    tokens = [str(t) for t in z["__tokens__"]]
    dac_all = np.asarray(z["__dac__"])
    polys_of = {t: [] for t in tokens}
    for k in z.files:
        if k.startswith("__"):
            continue
        tok, _ = k.split("|")
        polys_of[tok].append(z[k])
    anchors = np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy").astype(np.float32)

    order = range(len(tokens))

    lifts = [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    flats = [0.0, 0.25, 0.5, 0.75, 1.0]
    scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    NAMES = ["absolute", "listwise", "pair_raw", "pair_norm", "auc"]
    acc = {"lift": {d: [] for d in lifts},
           "flat": {a: [] for a in flats},
           "scale": {s_: [] for s_ in scales}}
    used = 0

    for i in order:
        if used >= args.scenes:
            break
        field = scene_field(polys_of[tokens[i]])
        if field is None:
            continue
        dac = dac_all[i]
        if dac.sum() == 0 or dac.sum() == len(dac):
            continue
        base = float(field.mean())
        ok = False
        for knob, vals, make in (("lift", lifts, lambda d: field + d),
                                 ("flat", flats, lambda a: (1 - a) * field + a * base),
                                 ("scale", scales, lambda s_: field * s_)):
            for v in vals:
                o = objectives(costs(make(v), anchors), dac, args.tau)
                if o is not None:
                    acc[knob][v].append(o)
                    ok = True
        if ok:
            used += 1
            if used % 20 == 0:
                print(f"  {used}/{args.scenes}", flush=True)

    K = anchors.shape[0]
    print(f"\n{used} 장면 x {K} 앵커.  tau={args.tau}, margin={MARGIN}, "
          f"격자 {BEV_H}x{BEV_W} (student).  균일 분포 상한 log K = {np.log(K):.3f}")

    def table(title, knob, unit):
        print(f"\n  {title}")
        print(f"    {unit:>9s}  " + "".join(f"{n:>12s}" for n in NAMES))
        rows = {}
        for v, lst in acc[knob].items():
            if not lst:
                continue
            rows[v] = {n: float(np.mean([o[n] for o in lst])) for n in NAMES}
            print(f"    {v:9.2f}  " + "".join(f"{rows[v][n]:12.4f}" for n in NAMES))
        return rows

    lr = table("전역 상승: Phi + delta  (장을 들어올림 = 치팅 방향)", "lift", "delta[m]")
    table("평탄화: (1-a)Phi + a*mean  (경계를 잃음)", "flat", "alpha")
    sr = table("전역 배율: Phi * s  (기울기만 바꿈)", "scale", "s")

    print("\n  === 판정 ===")
    for n in NAMES:
        if n == "auc":
            continue
        seq = [lr[v][n] for v in sorted(lr)]
        mono = all(b <= a + 1e-9 for a, b in zip(seq[:-1], seq[1:]))
        best = min(sorted(lr), key=lambda v: lr[v][n])
        span = max(seq) - min(seq)
        worst_at_far = lr[max(lr)][n]
        print(f"    {n:10s} delta 단조감소 {'예(퇴화)' if mono else '아니오':8s}"
              f"  최소 delta {best:+5.2f}  최소값 {lr[best][n]:8.4f}"
              f"  delta=+8 {worst_at_far:8.4f}  분별폭 {worst_at_far - lr[best][n]:8.4f}")
    print(f"    scale 불변성(s=0.25 vs 4): "
          + "  ".join(f"{n} {sr[0.25][n]:.3f}->{sr[4.0][n]:.3f}" for n in NAMES))


if __name__ == "__main__":
    main()
