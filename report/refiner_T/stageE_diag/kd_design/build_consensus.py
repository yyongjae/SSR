#!/usr/bin/env python
"""H5: decode KD-consensus corrections of the two teachers on E2's navtest tau0 drafts.
Inputs: fp32 teacher outputs (<B>/R_T4_fp32, <B>/R_M4_fp32; decoded c_lon [8] after the mode-A clamp and e_lat [8]
after the curvature projection = exactly the E2 KD targets) and the fp16 outputs (as e0_teacher_refine).
Controls -> raw: z = lon_z_from_q(lon_q_from_c(c)[1:]) (c0 = c1 = 0, c <= 0), w = atanh(e / D_MAX); decode mode A
(same v0 as the teachers).  Variants (k):
  0 R_T4 fp16 (tool output)   1 R_M4 fp16 (tool output)
  2 R_T4 fp32 roundtrip (its own c/e re-encoded, sanity: == R_T4 fp32 tau1)   3 R_M4 fp32 roundtrip
  4 MID  = element-wise mean of the two teachers' decoded controls (the L1-KD midpoint)
  5 WEAK = element-wise KD-optimal point with the SMALLEST correction: lon c = max(c_T, c_M) (less braking);
           lat e = the smaller |e| if same sign, else 0 (every point of [e_T, e_M] minimises the 2-teacher L1).
Writes <B>/consensus/refined.npz (drafts [N, 6, 8, 3]), tokens.parquet, consensus_meta.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/external-user/yongjae/SSR")
from navsim.agents.para_ssr.refiner.decoder import decode, lon_q_from_c, lon_z_from_q, A_DEC, A_UP, D_MAX
B = Path("/home/external-user/ssd/yongjae_refiner/stageE_diag/kd_design")

def L(n):
    return dict(np.load(B / n / "pred.npz"))

T16, M16, T32, M32 = L("R_T4_fp16"), L("R_M4_fp16"), L("R_T4_fp32"), L("R_M4_fp32")
tok = T32["tokens"]
for d in (T16, M16, M32):
    assert np.array_equal(d["tokens"], tok)
assert np.array_equal(T32["v0"], M32["v0"])
tau0 = torch.tensor(T32["tau0"][:, 0], dtype=torch.float64)
v0 = torch.tensor(T32["v0"], dtype=torch.float64)

def enc_dec(c, e):
    c = torch.tensor(c, dtype=torch.float64); e = torch.tensor(e, dtype=torch.float64)
    q = lon_q_from_c(c)[..., 1:]
    q = torch.clamp(q, -A_DEC * (1 - 1e-9), A_UP * (1 - 1e-9))
    z = lon_z_from_q(q)
    w = torch.atanh(torch.clamp(e[..., 2:] / D_MAX, -1 + 1e-12, 1 - 1e-12))
    out = []
    for i in range(0, len(c), 2048):
        o = decode(tau0[i:i + 2048], z[i:i + 2048], w[i:i + 2048], v0=v0[i:i + 2048], mode="A")
        out.append((o["traj"].numpy(), o["c_lon"].numpy(), o["e_lat"].numpy(), o["flags"]["alpha"].numpy()))
    return [np.concatenate(x) for x in zip(*out)]

cT, cM, eT, eM = T32["c_lon"][:, 0].astype(np.float64), M32["c_lon"][:, 0].astype(np.float64), \
    T32["e_lat"][:, 0].astype(np.float64), M32["e_lat"][:, 0].astype(np.float64)
rt_T = enc_dec(cT, eT); rt_M = enc_dec(cM, eM)
mid = enc_dec((cT + cM) / 2, (eT + eM) / 2)
same = np.sign(eT) == np.sign(eM)
e_weak = np.where(same, np.where(np.abs(eT) < np.abs(eM), eT, eM), 0.0)
weak = enc_dec(np.maximum(cT, cM), e_weak)
meta = {"roundtrip_T_max_abs_vs_tau1_fp32": float(np.abs(rt_T[0] - T32["tau1"][:, 0]).max()),
        "roundtrip_M_max_abs_vs_tau1_fp32": float(np.abs(rt_M[0] - M32["tau1"][:, 0]).max()),
        "roundtrip_T_p99_abs": float(np.percentile(np.abs(rt_T[0] - T32["tau1"][:, 0]).max((1, 2)), 99)),
        "roundtrip_M_p99_abs": float(np.percentile(np.abs(rt_M[0] - M32["tau1"][:, 0]).max((1, 2)), 99)),
        "mid_ctrl_reproduced_max_abs": float(max(np.abs(mid[1][:, 2:] - (cT + cM)[:, 2:] / 2).max(), np.abs(mid[2][:, 2:] - (eT + eM)[:, 2:] / 2).max())),
        "mid_alpha_lt1_frac": float((mid[3] < 1 - 1e-6).mean()), "weak_alpha_lt1_frac": float((weak[3] < 1 - 1e-6).mean()),
        "fp16_vs_fp32_T_traj_max_abs": float(np.abs(T16["tau1"] - T32["tau1"]).max()),
        "fp16_vs_fp32_T_traj_p99": float(np.percentile(np.abs(T16["tau1"] - T32["tau1"]).max((1, 2, 3)), 99)),
        "fp16_vs_fp32_M_traj_max_abs": float(np.abs(M16["tau1"] - M32["tau1"]).max()),
        "fp16_vs_fp32_M_traj_p99": float(np.percentile(np.abs(M16["tau1"] - M32["tau1"]).max((1, 2, 3)), 99)),
        "variants": ["R_T4_fp16", "R_M4_fp16", "R_T4_fp32_roundtrip", "R_M4_fp32_roundtrip", "MID", "WEAK"]}
drafts = np.stack([T16["tau1"][:, 0], M16["tau1"][:, 0], rt_T[0], rt_M[0], mid[0], weak[0]], 1).astype(np.float32)
o = B / "consensus"; o.mkdir(exist_ok=True)
np.savez(o / "refined.npz", tokens=tok, drafts=drafts)
np.savez(o / "consensus_controls.npz", tokens=tok, c_mid=mid[1], e_mid=mid[2], c_weak=weak[1], e_weak=weak[2])
pd.read_parquet(B / "R_T4_fp32" / "tokens.parquet").to_parquet(o / "tokens.parquet", index=False)
(o / "consensus_meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
