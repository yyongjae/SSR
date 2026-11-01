"""CK2 e2e T2 (model side): CKE2E2Config defaults (USER DECISIONS + SPEC s7-1), nested merge, unknown-key errors,
validate rules, and the student build (arm S, lon head zeroed + frozen, gate frozen, bev_grad_scale, score prior).
CPU only:
  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_config.py \
    --basetemp=/tmp/claude-0/-workspace/f9196902-3a88-4418-a505-cd713226ed4a/scratchpad/ck2e2e/pytest/config
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402

pytestmark = pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")


def parent(**over):
    d = dict(refiner_mode="off", ck_e2e={}, plan_anchor=True, bev_h=50, bev_w=100,
             pc_range=[-32.0, 0.0, -2.0, 32.0, 32.0, 2.0], embed_dims=256)
    d.update(over)
    return SimpleNamespace(**d)


def test_defaults_are_the_user_decisions():
    c = O.CKE2E2Config.from_any(None)
    assert not c.enabled and c.topk == 16 and c.n_var_step == 32
    # doubled per-mb constants (2 GPUs x accumulate 16 vs the old 4 x 8)
    assert c.lat_kd["start_mb"] == 1000 and c.label_refresh_every_mb == 1000 and c.grad_share_every == 100
    assert c.lambda_score == 1.0 and c.lambda_kd_score == 0.5 and c.teacher_amp and not c.amp
    assert c.record_from_epoch == 4 and c.onpolicy_from_epoch == 5 and c.label_lag == 1
    assert c.variants == {"speeds": [-1.0, -0.5, 0.5], "lats": [-0.5, 0.5], "combine": "separate", "s_on_frac": 0.2,
                          "ext": "straight", "compute_dtype": "float64"}
    # user 2026-10-09 00:30 KST: official-EP teachers and student label targets
    assert c.teacher_det_run.endswith("ck2/train/ck2T10") and c.teacher_map_run.endswith("ck2/train/ck2M10")
    assert c.ep_target == "official" and c.teacher_which == "last"
    assert c.warmup["var_dir"].endswith("var_separate_sampler16_accstraight")
    assert c.warmup["raw_labels"].endswith("labels/navtrain_train/raw256")
    assert c.onpolicy_label == {"n_orig": 16, "n_var": 32, "fallback": "warmup"}
    assert c.bev_kd["enabled"] is False and c.bev_kd["teachers"] == ["det"] and c.bev_kd["ratio"] == 0.1 \
        and c.bev_kd["cap"] == 0.25 and c.bev_kd["init"] == "zero" and c.bev_kd["distance"] == "mse"
    assert all(c.bev_kd[k] is None for k in ("ratio_det", "ratio_map", "cap_det", "cap_map"))   # shared by default
    assert c.infer_select is None and c.kd_ramp_epochs == 0.0 and c.bev_grad_scale == 1.0
    assert O.CKE2E2Config.from_any(c.to_dict()) == c
    assert O.CKE2E2Config.from_any(c) == c


def test_nested_merge_dictconfig_and_unknown_keys(tmp_path):
    from omegaconf import OmegaConf
    d = OmegaConf.create({"enabled": True, "io_dir": str(tmp_path), "lat_kd": {"start_mb": 7},
                          "bev_kd": {"enabled": "true", "teachers": "[det,map]", "w_max": "null",
                                     "ratio_map": "0.2", "cap_det": "null"},
                          "infer_select": {"beta": 0.5, "set": "all", "lat_mode": "on"}})
    c = O.CKE2E2Config.from_any(d)
    assert c.lat_kd["start_mb"] == 7 and c.lat_kd["cap"] == 10.0 and c.lat_kd["balance"] == "ema"
    assert c.bev_kd["enabled"] is True and c.bev_kd["teachers"] == ["det", "map"] and c.bev_kd["w_max"] is None
    assert c.bev_kd["ratio_map"] == 0.2 and c.bev_kd["cap_det"] is None and c.bev_kd["ratio_det"] is None
    assert c.infer_select == {"beta": 0.5, "set": "all", "lat_mode": "on"}
    for bad in ({"enabld": True}, {"lat_kd": {"mm": 1}}, {"warmup": {"x": 1}}, {"bev_kd": {"share": 0.1}},
                {"infer_select": {"beta": 1.0}}, {"variants": 3}):
        with pytest.raises(ValueError):
            O.CKE2E2Config.from_any(bad)


def test_validate_rules(tmp_path):
    O.CKE2E2Config.from_any({"enabled": False, "io_dir": "rel"}).validate()        # off: nothing checked
    base = {"enabled": True, "io_dir": str(tmp_path / "io"), "teacher_det_run": str(TL.SMOKE_T),
            "teacher_map_run": str(TL.SMOKE_M)}
    O.CKE2E2Config.from_any(base).validate(parent())
    bad = [({"io_dir": "rel/path"}, "absolute"), ({"io_dir": "/a=b"}, "absolute"),
           ({"record_from_epoch": 6}, "record_from_epoch"), ({"label_lag": 0}, "label_lag"),
           ({"topk": 8}, "topk"), ({"score_prior": "phase1"}, "score_prior"),
           ({"n_var_step": 81}, "n_var_step"), ({"warmup": {"n_var": -1}}, "warmup.n_var"),
           ({"onpolicy_label": {"n_orig": 8}}, "n_orig"), ({"onpolicy_label": {"fallback": "r34"}}, "fallback"),
           ({"var_sampling": "x"}, "var_sampling"), ({"warmup": {"sur_groups": "far"}}, "sur_groups"),
           ({"variants": {"combine": "cross"}}, "variants"), ({"variants": {"ext": "centerline_gt"}}, "variants"),
           ({"variants": {"speeds": [-1.0, 0.5]}}, "variants"), ({"variants": {"ext": "const_curv"}}, "label files"),
           ({"lat_kd": {"balance": "grad"}}, "balance"), ({"lat_kd": {"m": 1.0}}, "lat_kd"),
           ({"teacher_map_run": ""}, "teacher_map_run"), ({"teacher_det_run": str(TL.SMOKE_M)}, "arm"),
           ({"bev_kd": {"enabled": True, "teachers": ["bev"]}}, "teachers"),
           ({"bev_kd": {"enabled": True, "cap": 0.0}}, "bev_kd"),
           ({"bev_kd": {"enabled": True, "adapter_weight": 0.0}}, "bev_kd"),
           ({"bev_kd": {"enabled": True, "teachers": ["det", "map"], "ratio_map": 0.0}}, "ratio_map"),
           ({"bev_kd": {"enabled": True, "teachers": ["det"], "cap_map": 0.5}}, "cap_map"),
           ({"infer_select": {"beta": 1.5, "set": "all", "lat_mode": "on"}}, "beta")]
    for over, msg in bad:
        kw = json.loads(json.dumps(base))
        for k, v in over.items():
            if isinstance(v, dict):
                kw.setdefault(k, {}).update(v)
            else:
                kw[k] = v
        with pytest.raises(ValueError, match=msg):
            O.CKE2E2Config.from_any(kw).validate(parent())
    c = O.CKE2E2Config.from_any(base)
    for p, msg in ((parent(refiner_mode="E2"), "refiner_mode"), (parent(ck_e2e={"enabled": True}), "both"),
                   (parent(plan_anchor=False), "plan_anchor"), (parent(bev_h=100), "S grid"),
                   (parent(embed_dims=128), "embed_dims")):
        with pytest.raises(ValueError, match=msg):
            c.validate(p)


def test_build_student_lon_gate_frozen_and_rng_untouched(tmp_path):
    c = TL.cfg2(tmp_path, bev_grad_scale=0.5)
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    net = O.build_student_ck2(c)
    assert torch.equal(before, torch.get_rng_state())                  # no global RNG consumed (arms stay aligned)
    assert net.arm == "S" and net.lead_head is None and net.trunk.adapter.bev_grad_scale == 0.5
    lh = net.trunk.lon_head[-1]
    assert float(lh.weight.abs().max()) == 0.0 and float(lh.bias.abs().max()) == 0.0
    frozen = {n for n, p in net.named_parameters() if not p.requires_grad}
    assert frozen == {f"trunk.{m}.{x}" for m in ("lon_head", "gate_head") for x in ("0.weight", "0.bias",
                                                                                       "2.weight", "2.bias")}
    assert set(net.init_report["frozen"]) == frozen
    trainable = O.ck2_parameters(SimpleNamespace(ck_student=net))
    assert len(trainable) == sum(1 for p in net.parameters() if p.requires_grad) and len(trainable) > 0
    # same init as a plain CKNet('S', seed) apart from the zeroed lon head
    from navsim.agents.para_ssr.ck.model import CKNet
    ref = CKNet("S", 0, score_hidden=256, lead_aux=False)
    sd, sr = net.state_dict(), ref.state_dict()
    assert set(sd) == set(sr)
    for k in sd:
        if not k.startswith("trunk.lon_head.2."):
            assert torch.equal(sd[k], sr[k]), k


def test_score_prior_sources(tmp_path):
    c = TL.cfg2(tmp_path, score_prior="teacher")
    net = O.build_student_ck2(c)
    want = json.loads((TL.SMOKE_T / "config.json").read_text())["init_report"]["score_prior"]
    b = net.score_head[-1].bias.detach().double()
    np.testing.assert_allclose(torch.sigmoid(b).numpy(), np.clip(want, 1e-3, 1 - 1e-3), rtol=0, atol=1e-6)
    assert net.init_report["score_prior_source"].startswith("teacher:")
    net0 = O.build_student_ck2(TL.cfg2(tmp_path, score_prior="none"))
    net1 = O.build_student_ck2(TL.cfg2(tmp_path, score_prior="teacher"), apply_prior=False)
    assert torch.equal(net0.score_head[-1].bias, net1.score_head[-1].bias)       # apply_prior False: untouched
    assert net1.init_report["score_prior"] is None


@pytest.mark.skipif(not Path(O._CK, "ck2/labels/navtrain_train/var_separate_sampler16_accstraight/index.npz").is_file(),
                    reason="variant label files absent")
def test_score_prior_ck2_warmup_small(tmp_path):
    """'ck2_warmup' = e2e_data2.ck2_warmup_prior(cfg, score_prior_rows) (small n for speed)."""
    from navsim.agents.para_ssr.ck import e2e_data2 as D2
    c = TL.cfg2(tmp_path, score_prior="ck2_warmup", score_prior_rows=20)
    net = O.build_student_ck2(c)
    want = D2.ck2_warmup_prior(c, n_max=20)
    got = torch.sigmoid(net.score_head[-1].bias.detach().double()).numpy()
    np.testing.assert_allclose(got, np.clip(want, 1e-4, 1 - 1e-4), atol=1e-6)
    assert net.init_report["score_prior_source"] == "ck2_warmup"
