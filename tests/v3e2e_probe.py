"""Probe of the CK2 e2e teacher / data path for tests/test_v3_e2e_teacher.py (bit-identity of the non-temporal path
against the HEAD code).  Not a test module.  Run in a FRESH process with the code root to probe as argv[1]:

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -B tests/v3e2e_probe.py <code root> <out.pt> <io dir>

Imports navsim from <code root> only (asserted) and touches nothing but the read-only CK data and <io dir>.  Saves a
dict of tensors: TeacherPair2.run outputs (CK2 smoke teachers uncalibrated; the ck2e2e_main teachers ck2T10 / ck2M10
with their KD calibration files), CK2E2ETargetBuilder.compute_for_token of the 2 fixture tokens + an unknown token
(ck2e2e_main config = the yaml teachers), and one warm-up CKE2E2.loss step (loss, logs without timings, BEV and
student gradients).  Uses only APIs that exist in both the HEAD and the current code.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(sys.argv[1]).resolve())
sys.path.insert(0, ROOT)
import navsim  # noqa: E402

assert str(Path(navsim.__file__).resolve()).startswith(ROOT + "/"), (navsim.__file__, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402

D = Path("/home/external-user/ssd/yongjae_refiner/ck")
FIX2 = D / "ck2/e2e_smoke_cpu/fixtures/real_b2_ck2.pt"
SMOKE_T, SMOKE_M = D / "ck2/smoke/smoke_ck2T_ddp2", D / "ck2/smoke/smoke_ck2M_ddp2"
MAIN_T, MAIN_M = D / "ck2/train/ck2T10", D / "ck2/train/ck2M10"
CAL = {"det": str(D / "ck2/kd_calib/ck2T10__last.json"), "map": str(D / "ck2/kd_calib/ck2M10__last.json")}
ANCHORS = "/home/external-user/kyungmin/SSR-v2/data/planning_vb/trajectory_anchors_256.npy"
KD_KEYS = ["nc", "dac", "ep", "ttc"]


def fake_predictions(B: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    anchors = torch.from_numpy(np.load(ANCHORS)).float()
    off = 0.2 * torch.randn(B, 256, 8, 3, generator=g)
    final = torch.randn(B, 256, generator=g)
    im = torch.softmax(torch.randn(B, 256, generator=g), -1)
    sim = torch.sigmoid(torch.randn(B, 5, 256, generator=g))
    bev = torch.randn(B, 5000, 256, generator=g)
    bev = ((bev - bev.mean(-1, keepdim=True)) / bev.std(-1, keepdim=True)).requires_grad_(True)
    top = final.argmax(-1)
    traj = (anchors[None] + off)[torch.arange(B), top]
    return {"bev_embed": bev, "plan_final_rewards": final, "trajectory_offset": off, "trajectory_anchors": anchors,
            "im_rewards": im, "sim_rewards": sim, "trajectory": traj}


def main() -> None:
    out, io = sys.argv[2], sys.argv[3]
    torch.set_num_threads(1)
    d = torch.load(FIX2, map_location="cpu", weights_only=False)
    f, t = d["features"], d["targets"]
    res = {}
    g = torch.Generator().manual_seed(11)
    a = torch.from_numpy(np.load(ANCHORS)).float()
    cand = a[torch.randint(0, 256, (2, 24), generator=g)] + 0.05 * torch.randn(2, 24, 8, 3, generator=g)
    cand_ok = torch.rand(2, 24, generator=g) > 0.2
    status = f["status_feature"].float()
    for tag, det, mp, calib in (("smoke", SMOKE_T, SMOKE_M, None), ("main", MAIN_T, MAIN_M, CAL)):
        if tag == "main" and not all(Path(p).is_file() for p in CAL.values()):
            continue
        tp = O.TeacherPair2(str(det), str(mp), "last", "cpu", amp=False, require_done=True,
                            ep_target=None if tag == "smoke" else "official", calib=calib, kd_keys=KD_KEYS)
        R = tp.run(t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], cand, status, cand_ok=cand_ok)
        res.update({f"pair/{tag}/{k}": v for k, v in R.items()})
    b = D2.CK2E2ETargetBuilder(ck_cfg={"teacher_det_run": str(MAIN_T), "teacher_map_run": str(MAIN_M)})
    for tok in list(d["tokens"]) + ["ffffffffffffffff"]:
        res.update({f"builder/{tok}/{k}": v for k, v in b.compute_for_token(tok).items()})
    cfg = O.CKE2E2Config.from_any({"enabled": True, "io_dir": io, "teacher_det_run": str(SMOKE_T),
                                   "teacher_map_run": str(SMOKE_M), "teacher_amp": False, "score_prior": "none",
                                   "grad_share_every": 0, "strict_rows": 0.0, "teacher_ep_check": False,
                                   "kd_calib": {"enabled": False}})
    ck = O.CKE2E2(cfg, max_epochs=30)
    student = O.build_student_ck2(cfg, apply_prior=False)
    pred = fake_predictions(2, seed=3)
    loss, logs = ck.loss(student, f, t, pred)
    loss.backward()
    res["loss/loss"] = loss.detach()
    res.update({f"loss/logs/{k}": v for k, v in logs.items() if not k.endswith("_ms")})
    res["loss/grad_bev"] = pred["bev_embed"].grad.detach().clone()
    res["loss/grad_student"] = torch.cat([p.grad.reshape(-1) for p in student.parameters() if p.grad is not None])
    torch.save(res, out)
    print(f"probe {ROOT}: {len(res)} tensors -> {out}")


if __name__ == "__main__":
    main()
