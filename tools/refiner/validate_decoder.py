#!/usr/bin/env python
"""Validation of the M4 decoder, DraftPath geometry and the draft-bank perturbation sampler (IMPL_SPEC §3.1, §3.2, §3.6).

Writes one JSON with every number quoted in the module docstrings / final report:
  identity      decode(z = 0, w = 0) on ALL navtest student (interaction_final) and human trajectories: max |error|,
                bitwise equality; float32 / float64; modes A / B.  (navtest used only for this exactness check.)
  lon_bounds    max |da| and jerk vs analytic bounds on random z (4001-point grid).
  path          DraftPath statistics on the human sample: merged knots, spline-vs-polyline deviation, constant-curvature
                extension error vs the real human path at +1 s / +2 s.
  projection    share of drafts whose OWN curvature exceeds kappa_lim somewhere (where the literal spec formula
                alpha = kappa_lim / max|kappa_new| would shrink any lateral correction), residual kappa ratio.
  bank          sample_bank on the human sample: valid rate per slot / family, rejection reasons, path sources,
                realised A_p / D_p, t0-continuity.
  roundtrip     perturbation -> mode-A decoder fitted back to the human (best of analytic-inverse / zero start):
                share within 0.1 m per family; lateral-only also in mode B.
  timing        decode forward+backward for a training batch (104 drafts, float32, 1 thread); sample_bank per token.

Frames/units: N frame, metres, seconds, radians (see geometry.py).  CPU only.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 python tools/refiner/validate_decoder.py \\
      --human /home/external-user/ssd/yongjae_refiner/human/train.npz --n 1000 \\
      --out report/refiner_T/decoder_validation.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from navsim.agents.para_ssr.refiner import decoder as D  # noqa: E402
from navsim.agents.para_ssr.refiner.geometry import DraftPath, kappa_limit, safe_norm  # noqa: E402

STUDENT_PKL = os.path.join(REPO, "work_dirs", "eval", "para_ssr_interaction_final_navtest_trajectories.pkl")
HUMAN_PKL = os.path.join(REPO, "report", "navsim_version_audit", "adversarial_verify", "human_navtest_trajectories.pkl")


def q(a, ps=(50, 90, 99, 100)):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return {f"p{p}": float(np.percentile(a, p)) for p in ps} if len(a) else {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_pkl(p):
    d = pickle.load(open(p, "rb"))["trajectories"]
    return np.stack([d[k] for k in sorted(d)]).astype(np.float32)


# ------------------------------------------------------------------------------------------------------------ sections
def sec_identity():
    out = {}
    for name, p in (("student_navtest", STUDENT_PKL), ("human_navtest", HUMAN_PKL)):
        T = load_pkl(p)
        for dt in (torch.float32, torch.float64):
            tau = torch.as_tensor(T).to(dt)
            z = torch.zeros(len(T), 6, dtype=dt)
            for mode in ("A", "B"):
                o = D.decode(tau, z, z, None, mode)
                out[f"{name}/{str(dt)[6:]}/{mode}"] = {"n": len(T), "max_abs_err": float((o["traj"] - tau).abs().max()),
                                                       "bitwise": bool(torch.equal(o["traj"], tau))}
    return out


def sec_lon_bounds():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(20000, 6, generator=g, dtype=torch.float64) * 3.0
    t = torch.linspace(0, 4, 4001, dtype=torch.float64)
    res = {}
    for mode in ("free", "A"):
        c = D.lon_c_from_q(D.lon_q_from_z(z))
        if mode == "A":
            c = torch.clamp(c, max=0.0)
        pr = D.lon_profile(c, t)
        res[mode] = {"da_min": float(pr["da"].min()), "da_max": float(pr["da"].max()),
                     "bound": [-D.A_DEC, D.A_UP], "jerk_abs_max": float(pr["jerk"].abs().max()),
                     "jerk_le_ctrl_bound": bool((pr["jerk"].abs().max(1).values <= pr["r"].abs().max(1).values + 1e-9).all())}
    # the draft's original increment rule (critic_impl issue 1) for reference
    qq = -D.A_DEC * torch.ones(1, 6, dtype=torch.float64)
    c_bad = torch.cat([torch.zeros(1, 2, dtype=torch.float64), torch.cumsum(0.8 * qq, 1)], 1)
    res["draft_v0_rule_da_max"] = float(D.lon_profile(c_bad, t)["da"].abs().max())
    return res


def load_human(path, n, seed):
    z = np.load(path, allow_pickle=True)
    ok = ~z["frame_gap"] & (z["n_reg"] >= 8)
    idx = np.random.default_rng(seed).choice(np.nonzero(ok)[0], min(n, int(ok.sum())), replace=False)
    idx.sort()
    return {"token": z["tokens"][idx], "tau_h": z["traj"][idx], "path_long": z["path"][idx], "n_valid": z["n_reg"][idx],
            "v0": z["v0"][idx].astype(np.float64), "a0": z["a0"][idx].astype(np.float64),
            "log": z["logs"][idx], "n_skipped_gap": int((~ok).sum()), "n_pool": int(len(ok))}


def sec_path(H):
    tau = torch.as_tensor(H["tau_h"].astype(np.float64))
    p = DraftPath(tau)
    moving = (p.knots()[:, -1] > 5).numpy()
    s = p.S[:, -1:] * torch.linspace(0, 1, 200, dtype=torch.float64)[None]
    xy = p.eval(s).xy.numpy()
    V = p.V.numpy()

    def segd(P, poly):
        a, b = poly[:-1], poly[1:]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(-1), 1e-12)
        t = np.clip(((P[:, None] - a[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
        return np.linalg.norm(P[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min(1)

    dev = np.array([segd(xy[i], V[i]).max() for i in range(len(V))])
    # extension vs the real human path (+1 s = pose 10, +2 s = pose 12), where available
    pl = H["path_long"].astype(np.float64)
    av = (H["n_valid"] >= 12) & moving
    full = DraftPath(torch.as_tensor(pl[av, :12]))
    ext = {}
    for bc in ("not_a_knot", "natural"):
        for pol in ("const_curv", "chord"):
            q_ = DraftPath(torch.as_tensor(pl[av, :8]), end_bc=bc).extend(10.0, pol)
            e = q_.eval(full.S[:, [10, 12]])
            err = safe_norm(e.xy - torch.as_tensor(pl[av][:, [9, 11], :2])).numpy()
            ext[f"{bc}/{pol}"] = {"+1s": q(err[:, 0]), "+2s": q(err[:, 1])}
    return {"n": int(len(tau)), "merged_knot_rows": float((~p.keep.all(1)).float().mean()),
            "smooth_residual_max_m": q(p.smooth_residual_max().numpy()),
            "spline_vs_polyline_max_dev_m": q(dev), "extension_error_m": ext, "n_extension_eval": int(av.sum())}


def sec_projection(H):
    out = {}
    sets = {"student_navtest": load_pkl(STUDENT_PKL), "human_navtest": load_pkl(HUMAN_PKL),
            "human_sample": H["tau_h"]}
    for name, T in sets.items():
        tau = torch.as_tensor(T.astype(np.float64))
        o = D.decode(tau, torch.zeros(len(T), 6, dtype=torch.float64), torch.zeros(len(T), 6, dtype=torch.float64))
        p = o["path"]
        pe = p.eval(o["s"])
        klim = kappa_limit(o["v"])
        over = (pe.kappa.abs() > klim).any(1)
        lat_on = o["flags"]["lat_on"]
        out[name] = {"n": len(T), "lat_on": float(lat_on.float().mean()),
                     "draft_exceeds_kappa_lim_somewhere": float(over[lat_on].float().mean())}
    # residual ratio after projection, random offsets on the human sample
    tau = torch.as_tensor(H["tau_h"].astype(np.float64))
    w = torch.as_tensor(np.random.default_rng(1).normal(0, 1, (len(tau), 6)))
    for n in (1, 2, 4, 6):
        o = D.decode(tau, torch.zeros_like(w), w, torch.as_tensor(H["v0"]), "A", n_proj=n)
        on = o["flags"]["lat_on"]
        out[f"kappa_ratio_after_{n}_passes"] = q(o["flags"]["kappa_ratio"][on].numpy())
    out["alpha_mean_random_w"] = float(o["flags"]["alpha"][on].mean())
    return out


def sec_bank(H):
    t0 = time.time()
    banks = []
    for i in range(len(H["token"])):
        banks.append(D.sample_bank(H["tau_h"][i], str(H["token"][i]), path_long=H["path_long"][i],
                                   n_valid=int(H["n_valid"][i]), v0=float(H["v0"][i]), a0=float(H["a0"][i])))
    dt = (time.time() - t0) / len(banks)
    fam = np.stack([b["family"] for b in banks])
    val = np.stack([b["valid"] for b in banks])
    par = np.stack([b["params"] for b in banks])
    out = {"n_tokens": len(banks), "sec_per_token": dt, "valid_rate": float(val.mean()),
           "valid_rate_per_slot": val.mean(0).round(4).tolist(),
           "family_count": {D.FAMILY_NAME[int(k)]: int(v) for k, v in zip(*np.unique(fam, return_counts=True))},
           "valid_rate_per_family": {D.FAMILY_NAME[int(f)]: float(val[fam == f].mean()) for f in np.unique(fam)},
           "reasons": dict(Counter(r for b in banks for r in b["reason"] if r).most_common(20)),
           "path_src": dict(Counter(s for b in banks for s in b["path_src"])),
           "tokens_with_all_lconst_invalid": float(np.mean([(~b["valid"][4:7]).all() for b in banks]))}
    for f in ("small", "lconst", "combined", "lat", "creep", "ignore_brake", "hdrift"):
        m = (fam == D.FAMILY[f]) & val
        if m.any():
            out[f"params/{f}"] = {c: q(par[m][:, j], (5, 50, 95)) for j, c in enumerate(D.PARAM_COLS)}
    # t0 continuity of valid perturbed drafts (first-segment speed - v0)
    fdv = []
    for i, b in enumerate(banks):
        for k in range(13):
            if b["valid"][k] and b["family"][k] not in (D.FAMILY["cv"],):
                fdv.append(np.hypot(*b["drafts"][k][0, :2]) / 0.5 - H["v0"][i])
    out["first_seg_dv"] = q(fdv, (1, 5, 50, 95, 99))
    return out, banks


def sec_roundtrip(H, banks, n_max):
    rows = []
    for i, b in enumerate(banks):
        ctx = D.HumanContext(H["tau_h"][i], H["path_long"][i], int(H["n_valid"][i]), float(H["v0"][i]))
        for k in range(13):
            f = D.FAMILY_NAME[int(b["family"][k])]
            if not b["valid"][k] or f in ("identity", "cv", "hdrift"):
                continue
            rows.append((i, k, f))
    rng = np.random.default_rng(2)
    if len(rows) > n_max:
        rows = [rows[j] for j in sorted(rng.choice(len(rows), n_max, replace=False))]
    # effective controls of each perturbation (re-decoded on its own path)
    tau, tgt, q0, e0, fam = [], [], [], [], []
    for i, k, f in rows:
        ctx = D.HumanContext(H["tau_h"][i], H["path_long"][i], int(H["n_valid"][i]), float(H["v0"][i]))
        b = banks[i]
        src = str(b["path_src"][k])
        if src == "human_long":
            path = ctx.long_path
        else:  # creep with extrapolation / centerline: rebuild as the sampler did
            need = max(0.0, float(b["params"][k][5])) + 2.0
            path = DraftPath(torch.as_tensor(ctx.tau_h.astype(np.float64))[None]).extend(need + 8.0, "straight")
        o = D.decode(torch.as_tensor(ctx.tau_h.astype(np.float64))[None], torch.as_tensor(b["z_lon"][k].astype(np.float64))[None],
                     torch.as_tensor(b["w_lat"][k].astype(np.float64))[None], ctx.v0, "P", path=path, lat_len=ctx.S8)
        tau.append(b["drafts"][k].astype(np.float64))
        tgt.append(H["tau_h"][i].astype(np.float64))
        q0.append(-o["q_lon"][0, 1:].numpy())
        e0.append(-o["e_lat"][0, 2:].numpy())
        fam.append(f)
    tau, tgt = torch.as_tensor(np.stack(tau)), torch.as_tensor(np.stack(tgt))
    fam = np.array(fam)
    z0 = D.lon_z_from_q(torch.as_tensor(np.stack(q0)).clamp(-0.999 * D.A_DEC, 0.999 * D.A_UP))
    w0 = torch.atanh((torch.as_tensor(np.stack(e0)) / D.D_MAX).clamp(-0.999, 0.999))
    t0 = time.time()
    r1 = D.fit_controls(tau, tgt[..., :2], None, "A", iters=20, target_h=tgt[..., 2], z_init=z0, w_init=w0)
    r2 = D.fit_controls(tau, tgt[..., :2], None, "A", iters=20, target_h=tgt[..., 2])
    e1, e2 = r1["err"].numpy(), r2["err"].numpy()
    err = np.minimum(e1, e2)
    herr = np.degrees(np.where(e1 <= e2, r1["herr"].numpy(), r2["herr"].numpy()))
    base = safe_norm(tau[..., :2] - tgt[..., :2]).max(1).values.numpy()
    out = {"n": int(len(fam)), "fit_sec": time.time() - t0, "per_family": {}}
    for f in sorted(set(fam)):
        m = fam == f
        out["per_family"][f] = {"n": int(m.sum()), "perturbation_max_dx_m": q(base[m], (50, 90)),
                                "within_0.1m": float(np.mean(err[m] < 0.1)), "err_m": q(err[m]),
                                "within_0.1m_zero_start_only": float(np.mean(e2[m] < 0.1)),
                                "heading_err_deg": q(herr[m], (50, 90, 99))}
    m = fam == "lat"
    if m.any():
        rB = D.fit_controls(tau[m], tgt[m][..., :2], None, "B", iters=20, target_h=tgt[m][..., 2], z_init=z0[m],
                            w_init=w0[m])
        eB = rB["err"].numpy()
        out["lat_mode_B"] = {"within_0.1m": float(np.mean(eB < 0.1)), "err_m": q(eB),
                             "max_dv_used_mps": q(rB["out"]["dv"].max(1).values.numpy(), (50, 90, 100))}
    out["all_within_0.1m"] = float(np.mean(err < 0.1))
    return out


def sec_timing():
    T = load_pkl(STUDENT_PKL)[:104]
    tau = torch.as_tensor(T)
    torch.set_num_threads(1)
    z = torch.zeros(104, 6, requires_grad=True)
    w = torch.zeros(104, 6, requires_grad=True)
    for _ in range(3):
        o = D.decode(tau, z, w)
        o["traj"].sum().backward()
    t0 = time.time()
    n = 20
    for _ in range(n):
        o = D.decode(tau, z + 0.1, w + 0.1)
        (o["traj"].sum() + o["kappa"].sum()).backward()
    return {"decode_fwd_bwd_ms_batch104_f32_1thread": (time.time() - t0) / n * 1000}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", default="/home/external-user/ssd/yongjae_refiner/human/train.npz")
    ap.add_argument("--n", type=int, default=1000, help="tokens sampled from --human")
    ap.add_argument("--n_rt", type=int, default=4000, help="max perturbed drafts in the round-trip fit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "report", "refiner_T", "decoder_validation.json"))
    ap.add_argument("--sections", default="identity,lon_bounds,path,projection,bank,roundtrip,timing")
    a = ap.parse_args()
    torch.set_num_threads(1)
    secs = a.sections.split(",")
    res = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "human": a.human, "n": a.n, "seed": a.seed,
           "constants": {"A_DEC": D.A_DEC, "A_UP": D.A_UP, "D_MAX": D.D_MAX, "S_LAT_MIN": D.S_LAT_MIN,
                         "DV_ACC": D.DV_ACC, "L_EXT_MAX": D.L_EXT_MAX, "KAPPA_SLACK": D.KAPPA_SLACK}}
    H = load_human(a.human, a.n, a.seed)
    res["human_sample"] = {"n": int(len(H["token"])), "n_pool": H["n_pool"], "n_skipped_frame_gap_or_short": H["n_skipped_gap"],
                           "n_logs": int(len(np.unique(H["log"])))}
    banks = None
    for s in secs:
        log("section", s)
        if s == "identity":
            res[s] = sec_identity()
        elif s == "lon_bounds":
            res[s] = sec_lon_bounds()
        elif s == "path":
            res[s] = sec_path(H)
        elif s == "projection":
            res[s] = sec_projection(H)
        elif s == "bank":
            res[s], banks = sec_bank(H)
        elif s == "roundtrip":
            if banks is None:
                res["bank"], banks = sec_bank(H)
            res[s] = sec_roundtrip(H, banks, a.n_rt)
        elif s == "timing":
            res[s] = sec_timing()
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        log("wrote", a.out)
    log("done")


if __name__ == "__main__":
    main()
