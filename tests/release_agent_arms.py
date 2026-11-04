"""Helper of tests/test_release_core1.py, run in its own process because CK_BEV64 is read at import time.
One training step on CPU (real student, real v15, the real 64 m refiner teachers when a run needs them) on the CK2 2-token
fixture with synthetic 100 x 100 teacher BEVs, for each KD arm.  Prints one JSON object {arm: record}:
  no_grad     trainable parameters without a gradient (plain DDP would raise on any of them)
  builders    target builder class names
  frozen      names of the requires_grad False parameters under _distill.
  loss_keys   the distill loss terms of the step
  init_sha    sha256 of the initial weights outside the refiner and the BEV KD (the arms must agree on it)
With the arm name 'reference': the v2_v15_v7_main overrides (kd_calib without a mode key, teacher paths set) composed and
built, then that run's checkpoint loaded with strict keys.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CK_BEV64"] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from test_ck_e2e_model_agent import sampling, v2_config  # noqa: E402

D = Path("/home/external-user/ssd/yongjae_refiner")
FIX2 = D / "ck/ck2/e2e_smoke_cpu/fixtures/real_b2_ck2.pt"
_RT = Path("/home/external-user/datasets/teacher_cache/refiner")
T_DET, T_MAP = _RT / "det", _RT / "map"
ANCHORS = REPO / "resources/planning_vb/trajectory_anchors_256.npy"
GRID = dict(bev_h=100, bev_w=100, pc_range=[-32.0, 0.0, -1.0, 32.0, 64.0, 3.0], map_pc_range=[-32.0, 0.0, -1.0, 32.0, 64.0, 3.0],
            distill_cache_size=[100, 100], plan_anchor_file=str(ANCHORS))
V15 = dict(use_distill=True, distill_selector_v15=True, distill_teacher_source="targets",
           distill_probe_warmup_steps=0, distill_probe_ramp_steps=1, distill_feature_root="/workspace/teacher_cache")
TEACHERS = dict(teacher_det_run=str(T_DET), teacher_map_run=str(T_MAP), teacher_amp=False)
NOKD = dict(use_teachers=False, teacher_det_run=None, teacher_map_run=None, lambda_kd_score=0.0,
            lat_kd={"balance": "fixed", "fixed": 0.0, "start_mb": 0}, kd_calib={"enabled": False})
DAC_EP, NC_EP_TTC = ["dac", "ep"], ["nc", "ep", "ttc"]
# arm: (extra agent config, ck_e2e2 config or None = refiner off)
ARMS = {
    "main": (V15, TEACHERS),
    "nokd_bev_on": (V15, NOKD),
    "det_off": (dict(V15, distill_off_banks=["det"]), dict(TEACHERS, kd_sources=["map"], kd_score_keys=DAC_EP)),
    "map_off": (dict(V15, distill_off_banks=["map"]), dict(TEACHERS, kd_sources=["det"], kd_score_keys=NC_EP_TTC)),
    "det_off_bev": (dict(V15, distill_off_banks=["det"]), TEACHERS),
    "det_off_ref": (V15, dict(TEACHERS, kd_sources=["map"], kd_score_keys=DAC_EP)),
    "noref_map_off": (dict(V15, distill_off_banks=["map"]), None),
    "noref_nobev": (dict(use_distill=False), None),
}


def init_sha(agent):
    h = hashlib.sha256()
    for k, v in sorted(agent.state_dict().items()):
        if torch.is_tensor(v) and not k.startswith(("ck_student.", "_distill.")):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run_reference():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
    ov = [ln.strip() for ln in (D / "v7/v2_v15_v7_main_overrides_reference.txt").read_text().splitlines() if ln.strip()]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "navsim/planning/script/config/training"), version_base="1.2"):
        cfg = compose(config_name="default_training", overrides=ov + ["++agent.config.backbone_pretrained=false"])
    GlobalHydra.instance().clear()
    agent = ParaSSRAgent(instantiate(cfg.agent.config), sampling(), lr=1e-4)
    ck = torch.load(sorted((D / "v7/e2e/v2_v15_v7_main/train/lightning_logs").glob("version_*/checkpoints/last.ckpt"))[-1],
                    map_location="cpu", weights_only=False)
    sd = {k[len("agent."):]: v for k, v in ck["state_dict"].items()}
    res = agent.load_state_dict(sd, strict=True)
    return {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys), "n_keys": len(sd),
            "kd_calib_mode": agent._ck_e2e2.cfg.kd_calib["mode"], "bn_mode": agent.para_ssr_model._backbone_bn_mode,
            "calib_online": agent._ck_e2e2.calib_online is not None}


def run_arm(name, fix, tmp):
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
    if name == "reference":
        return run_reference()
    over, ck = ARMS[name]
    cfg = dict(GRID, **over)
    if ck is not None:
        cfg["ck_e2e2"] = dict({"enabled": True, "io_dir": str(Path(tmp) / name), "score_prior_rows": 200}, **ck)
    torch.manual_seed(0)
    agent = ParaSSRAgent(v2_config(**cfg), sampling(), lr=1e-4)
    sha0 = init_sha(agent)
    builders = [type(b).__name__ for b in agent.get_target_builders()]
    agent.train()
    f = {k: v.clone() for k, v in fix["features"].items()}
    t = {k: v.clone() for k, v in fix["targets"].items() if not k.startswith(("kd_bev_", "kd_ok_"))}
    t["scene_token"] = ["fixture_0", "fixture_1"]
    g = torch.Generator().manual_seed(1)
    for i in (0, 1):
        t[f"kd_bev_{i}"] = torch.randn(2, 256, 100, 100, generator=g).to(torch.float16)
        t[f"kd_ok_{i}"] = torch.ones(2, dtype=torch.bool)
    pred = agent.forward(f)
    loss = agent.compute_loss(f, t, pred)
    loss.backward()
    d = getattr(agent, "_distill", None)
    logs = agent.latest_logs
    ck2 = agent._ck_e2e2
    return {
        "loss": float(loss.detach()), "builders": builders, "init_sha": sha0,
        "no_grad": [n for n, p in agent.named_parameters() if p.requires_grad and p.grad is None],
        "frozen": [] if d is None else sorted({".".join(n.split(".")[:3]) for n, p in d.named_parameters()
                                               if not p.requires_grad and n.startswith(("anchor_distill.probes",
                                                                                        "anchor_distill.student_proj"))}),
        "loss_keys": sorted(k for k in logs if k.startswith(("loss_distill_", "loss_probe_"))),
        "teacher_ok_frac": float(logs["distill/teacher_ok_frac"]) if "distill/teacher_ok_frac" in logs else None,
        "kd_ok_frac": None if ck2 is None else float(logs["ck2/kd_ok_frac"]),
        "kd_score": None if ck2 is None else float(logs["ck2/kd_score"]),
        "teachers_built": None if ck2 is None else ck2._teachers is not None,
        "calib_online": None if ck2 is None else ck2.calib_online is not None,
        "n_keys_state": len(agent.state_dict()),
    }


def main():
    tmp, arms = sys.argv[1], sys.argv[2].split(",")
    fix = torch.load(FIX2, map_location="cpu", weights_only=False)
    print("\n" + json.dumps({a: run_arm(a, fix, tmp) for a in arms}))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
