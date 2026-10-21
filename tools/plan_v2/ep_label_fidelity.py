"""How well does each ego-progress label predict the v1 scorer's EP of the EMITTED trajectory?

The EP head is trained on the ANCHOR's precomputed EP while the model emits anchor + offset -- the
same mismatch the TTC label had.  A geometric EP label needs the scorer's quantity:

    EP_k = raw_k * m_k / max(raw_ref * m_ref, raw_k * m_k)        (pair scoring, pdm_scorer.py)
    raw  = project(end centre) - project(start centre) onto the ROUTE centerline, clipped at 0

where m is the multiplicative gate (NC * DAC * DDC).  At training time the PDM-Closed reference
trajectory does not exist, so its progress must be replaced; this measures the candidates for that
replacement, on navtest where the true reference IS available (metric cache), against the anchor's
own label:

    anchor       v1 EP of the anchor driven verbatim (tools/plan_v2/pdms_all_candidates.py on a
                 zero-offset dump) -- the scorer that produced the training labels
    geo/pdm      raw / max(raw_pdm, raw)      privileged reference -- the ceiling of this design
    geo/human    raw / max(raw_human, raw)    the logged human trajectory as the reference
    geo/max      raw / max_k raw_k            the scene's best candidate as the reference

Scored two ways: against the true EP of every candidate (the gate included), and against the
true EP of the candidates the gate passes (progress fidelity alone).  Per-scene rank correlation is
the number that matters for selection.

    python tools/plan_v2/ep_label_fidelity.py --workers 10
"""
import argparse
import lzma
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
E = ROOT / "work_dirs/eval_epdms"
MODELS = {"v2_r34": ("diag_para_ssr_v2_r34/dump.npz", "candidate_pdms_para_ssr_v2_r34.npz"),
          "feas_gt": ("field_para_ssr_v2_r34_feas_gt.npz", "candidate_pdms_para_ssr_v2_r34_feas_gt.npz"),
          "B3_ep23": ("field_para_ssr_v2_r34_feas_gt_select_ttcband_hold_ep23.npz", "candidate_pdms_b3_ep23.npz")}
REAR_TO_CENTER = (4.049 - 1.127) / 2          # pacifica: rear axle -> bbox centre (pdm_scorer uses CENTER)
_CTX = {}


def _init(traj_path, human_path, cache_root):
    from navsim.common.dataloader import MetricCacheLoader
    _CTX["traj"] = np.load(traj_path, mmap_mode="r")             # [M, N, 256, 8, 3] ego-frame poses
    _CTX["human"] = np.load(human_path, mmap_mode="r")           # [N, 8, 3]
    _CTX["loader"] = MetricCacheLoader(Path(cache_root))


def _to_global_centres(poses, ego):
    """[..., 3] ego-frame rear-axle poses -> [..., 2] global bbox centres."""
    x0, y0, h0 = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
    c, s = np.cos(h0), np.sin(h0)
    gx = x0 + c * poses[..., 0] - s * poses[..., 1]
    gy = y0 + s * poses[..., 0] + c * poses[..., 1]
    gh = h0 + poses[..., 2]
    return np.stack([gx + REAR_TO_CENTER * np.cos(gh), gy + REAR_TO_CENTER * np.sin(gh)], -1)


def _one(item):
    i, token = item
    from shapely.geometry import Point
    try:
        with lzma.open(_CTX["loader"].metric_cache_paths[token], "rb") as f:
            cache = pickle.load(f)
        ego = cache.ego_state
        line = cache.centerline
        start = _to_global_centres(np.zeros(3), ego)
        s0 = line.project([Point(*start)])[0]

        def raw(poses):                                           # [..., 8, 3] -> [...]
            ends = _to_global_centres(poses[..., -1, :], ego).reshape(-1, 2)
            prog = np.asarray(line.project([Point(*p) for p in ends])) - s0
            return np.clip(prog, 0, None).reshape(poses.shape[:-2])

        cand = np.stack([raw(np.asarray(_CTX["traj"][m, i], dtype=np.float64))
                         for m in range(_CTX["traj"].shape[0])])  # [M, 256]
        pdm_states = cache.trajectory.get_sampled_trajectory()
        pend = pdm_states[-1].center
        pdm_raw = max(line.project([Point(pend.x, pend.y)])[0] - s0, 0.0)
        hum = raw(np.asarray(_CTX["human"][i], dtype=np.float64)[None])[0]
        return i, cand, pdm_raw, hum
    except Exception as exc:
        return i, repr(exc)[:160], None, None


def spearman_rows(a, b, mask=None):
    """mean over rows of the Spearman correlation between a[r] and b[r] (rows with variance only)."""
    out = []
    for r in range(len(a)):
        x, y = a[r], b[r]
        if mask is not None:
            x, y = x[mask[r]], y[mask[r]]
        if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        out.append(np.corrcoef(rx, ry)[0, 1])
    return float(np.mean(out)), len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    sel = np.load(E / "anchors0_3000.npz")
    toks = [str(t) for t in sel["tokens"]]
    anc = np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy").astype(np.float32)
    trajs, truth = [], {}
    for m, (dump, cand) in MODELS.items():
        d = np.load(E / dump); dat = {str(t): j for j, t in enumerate(d["tokens"])}
        rows = np.array([dat[t] for t in toks])
        trajs.append(anc[None] + d["trajectory_offset"][rows].astype(np.float32))
        truth[m] = np.load(E / cand)["scores"][rows]               # [N, 256, 7] NC DAC DDC EP TTC C PDMS
    tp = E / "_ep_fid_traj.npy"; np.save(tp, np.stack(trajs))
    pt = np.load("/home/external-user/kyungmin/SSR/data/readout/plan_targets_navtest.npz", allow_pickle=True)
    pat = {str(t): j for j, t in enumerate(pt["tokens"])}
    off = pt["offsets"][[pat[t] for t in toks]].astype(np.float64)   # per-step displacement
    human = off.copy(); human[..., :2] = np.cumsum(off[..., :2], axis=1); human[..., 2] = np.cumsum(off[..., 2], axis=1)
    hp = E / "_ep_fid_human.npy"; np.save(hp, human)
    print(f"{len(toks)} 장면.  사람 궤적 끝점 거리 median {np.median(np.linalg.norm(human[:, -1, :2], axis=-1)):.1f} m "
          f"(변위 누적; 이상하면 offsets 정의를 다시 볼 것)", flush=True)

    M = len(MODELS); N = len(toks)
    raw = np.full((M, N, 256), np.nan); pdm = np.full(N, np.nan); hum = np.full(N, np.nan); bad = 0
    with ProcessPoolExecutor(a.workers, initializer=_init,
                             initargs=(str(tp), str(hp), str(ROOT / "data/exp/metric_cache"))) as ex:
        for i, c, p, h in ex.map(_one, list(enumerate(toks)), chunksize=8):
            if isinstance(c, str):
                bad += 1; continue
            raw[:, i], pdm[i], hum[i] = c, p, h
    tp.unlink(); hp.unlink()
    ok = ~np.isnan(pdm)
    print(f"실패 {bad}.  PDM 기준 진행량 median {np.nanmedian(pdm):.1f} m,  사람 median {np.nanmedian(hum):.1f} m,  "
          f"상관 {np.corrcoef(pdm[ok], hum[ok])[0, 1]:.3f}\n", flush=True)

    anchor = np.load(E / "candidate_pdms_anchors0_3000.npz")["scores"][:, :, 3]
    np.savez_compressed(E / "ep_fidelity_arrays.npz", tokens=np.array(toks), ok=ok, raw=raw, pdm=pdm, hum=hum,
                        anchor=anchor, models=np.array(list(MODELS)))

    # the gate a training step CAN compute: the merged corridor on the ground-truth static + agent
    # fields at margin 0 (feasibility.corridor_clearance) -- what the feasible head's target uses
    import torch
    from navsim.agents.para_ssr.feasibility import corridor_clearance

    class _C:
        map_pc_range = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)
        feas_dyn_weight = 1.0
    g = np.load(E / "gt_field_navtest.npz"); gat = {str(t): j for j, t in enumerate(g["tokens"])}
    gi = np.array([gat[t] for t in toks])
    gate_geo = {}
    for mi, m in enumerate(MODELS):
        tr = np.stack(trajs)[mi]
        out = []
        for s0 in range(0, N, 100):
            e0 = min(s0 + 100, N)
            p = {"trajectory_anchors": torch.zeros(256, 8, 3),
                 "trajectory_offset": torch.from_numpy(tr[s0:e0]).float(),
                 "drivable_sdf_pred": torch.from_numpy(g["drivable_sdf_gt"][gi[s0:e0]].astype(np.float32)),
                 "agent_sdf_pred": torch.from_numpy(g["agent_sdf_gt"][gi[s0:e0]].astype(np.float32))}
            out.append(corridor_clearance(p, _C, 0.0, 0.0).numpy() >= 0)
        gate_geo[m] = np.concatenate(out)
    np.savez_compressed(E / "ep_fidelity_gate.npz", **{m: v for m, v in gate_geo.items()})
    for mi, m in enumerate(MODELS):
        T = truth[m][ok]; ep = T[:, :, 3]; gate = (T[:, :, 0] * T[:, :, 1] * T[:, :, 2]) >= 1 - 1e-9
        r = raw[mi][ok]
        gh = r / np.maximum(np.maximum(hum[ok][:, None], r), 1e-6)
        labels = {"anchor (v1)": anchor[ok],
                  "geo/pdm (상한)": r / np.maximum(np.maximum(pdm[ok][:, None], r), 1e-6),
                  "geo/human": gh,
                  "geo/max": r / np.maximum(r.max(1, keepdims=True), 1e-6),
                  "geo/human x 진짜판정": gh * gate,
                  "geo/human x 장판정": gh * gate_geo[m][ok]}
        print(f"=== {m} 후보 ({int(ok.sum())} 장면 x 256)")
        print(f"  {'라벨':18s}{'MAE 전체':>10s}{'MAE 통과후보':>13s}{'장면내 순위상관 전체':>20s}{'통과후보':>10s}")
        for name, lab in labels.items():
            mae_all = np.abs(lab - ep).mean(); mae_g = np.abs(lab - ep)[gate].mean()
            sp_all, _ = spearman_rows(lab, ep); sp_g, _ = spearman_rows(lab, ep, gate)
            print(f"  {name:18s}{mae_all:10.3f}{mae_g:13.3f}{sp_all:20.3f}{sp_g:10.3f}")
        print()


if __name__ == "__main__":
    main()
