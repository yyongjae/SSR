"""v2_nokd: the refiner (ck_e2e2) without any teacher (ck_e2e2.use_teachers false, tools/ck/e2e2/v2_nokd.set).  CPU only:
  nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_nokd.py

Covers: the config refuses use_teachers false with any teacher / KD knob left on; no teacher is ever built; the loss and EVERY gradient (student CK params, bev_embed) of the no-teacher run are
bit-identical to the teacher run with lambda_kd_score 0 and lateral-KD weight 0, in the warm-up and the on-policy
phase, while the no-teacher targets carry no kd_bev / kd_ok tensor; the target builder opens no teacher cache; the
launcher records teachers.json mode 'none', arm 'nokd' and differs from v2_v15_v7 only by the KD / v15 lines; the agent
trains a step without teacher targets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
import ck2e2e_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402

pytestmark = pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")
E2 = REPO / "tools/ck/e2e2"
NOKD = {"use_teachers": False, "teacher_det_run": None, "teacher_map_run": None, "lambda_kd_score": 0.0,
        "lat_kd": {"balance": "fixed", "fixed": 0.0, "start_mb": 0}, "kd_calib": {"enabled": False}}
# the teacher run it must equal: same everything, KD weights 0 (= the old ck2e2e_nokd recipe, report 58)
ZERO_KD = {"lambda_kd_score": 0.0, "lat_kd": {"balance": "fixed", "fixed": 0.0, "start_mb": 0}}
KD_KEYS = ("kd_bev_", "kd_ok_")


def _no_kd_targets(t):
    return {k: v for k, v in t.items() if not k.startswith(KD_KEYS)}


def _student(c, seed=0):
    torch.manual_seed(seed)
    return O.build_student_ck2(c)


# ----------------------------------------------------------------------------------------------- config
def test_config_rules(tmp_path):
    c = TL.cfg2(tmp_path, **NOKD)
    c.validate(None)
    assert c.use_teachers is False and c.teacher_det_run == "" and c.teacher_map_run == ""
    with pytest.raises(RuntimeError, match="no refiner teachers"):
        O.CKE2E2(c).teachers("cpu")
    for over, msg in ((dict(teacher_det_run=str(TL.SMOKE_T)), "teacher_det_run / teacher_map_run must be null"),
                      (dict(teacher_map_run=str(TL.SMOKE_M)), "teacher_det_run / teacher_map_run must be null"),
                      (dict(lambda_kd_score=0.5), "lambda_kd_score must be 0"),
                      (dict(lat_kd={"balance": "ema", "fixed": 0.0}), "lat_kd must be balance 'fixed'"),
                      (dict(lat_kd={"balance": "fixed", "fixed": 1.0}), "lat_kd must be balance 'fixed'"),
                      (dict(kd_calib={"enabled": True, "det": "/x/calib.json"}), "kd_calib.enabled must be false"),
                      (dict(bev_kd={"enabled": True, "teachers": ["det"]}), "bev_kd.enabled must be false"),
                      (dict(score_prior="teacher"), "score_prior 'teacher'")):
        d = {**NOKD, **over}
        with pytest.raises(ValueError, match="use_teachers false"):
            TL.cfg2(tmp_path, **d).validate(None)
        with pytest.raises(ValueError, match=msg.replace("(", r"\(").replace(")", r"\)")):
            TL.cfg2(tmp_path, **d).validate(None)
    # the teacher default is unchanged: both runs still required
    with pytest.raises(ValueError, match="teacher_det_run is required"):
        TL.cfg2(tmp_path, teacher_det_run=None).validate(None)


# ----------------------------------------------------------------------------------------------- loss
def _pair(tmp_path, phase):
    """(teacher run with zero KD weights, no-teacher run): configs, helpers, identical students."""
    out = []
    for name, over in (("t", ZERO_KD), ("n", NOKD)):
        c = TL.cfg2(tmp_path / name, rec_chunk_tokens=1, **over)
        ck = O.CKE2E2(c, max_epochs=30)
        g = None
        if phase == "onpolicy":
            ck._labels = D2.LabelStore2(c.io_dir, n_rows=16, packed=c.warmup["packed"], var_sampling=c.var_sampling,
                                        ep_target=c.ep_target)
            g = TL.make_gen(c.io_dir, 4, [3], 16)
            ck.epoch, ck.epoch_frac, ck.gstep = 5, 5.5, 3
        out.append((c, ck, _student(c), g))
    (_, _, st_t, g_t), (_, _, st_n, g_n) = out
    sd_t, sd_n = st_t.state_dict(), st_n.state_dict()
    assert list(sd_t) == list(sd_n) and all(torch.equal(sd_t[k], sd_n[k]) for k in sd_t)
    if g_t is not None:
        assert all(np.array_equal(a, b) for a, b in zip(g_t, g_n))
    return out


@pytest.mark.parametrize("phase", ["warmup", "onpolicy"])
def test_loss_and_gradients_equal_the_zero_kd_teacher_run(tmp_path, phase):
    res = {}
    for (c, ck, st, _), name in zip(_pair(tmp_path, phase), ("t", "n")):
        rows = [3, 7] if phase == "onpolicy" else None          # row 3 has generation 4, row 7 -> fallback
        f, t = TL.fresh(2, rows=rows)
        if name == "n":
            t = _no_kd_targets(t)                               # the no-teacher target builder emits none
        pred = TL.fake_predictions(2, seed=3)
        v2 = (pred["bev_embed"] ** 2).mean()
        loss, logs = ck.loss(st, f, t, pred, v2_loss=v2)
        loss.backward()
        res[name] = (loss.detach(), logs, {n: p.grad.clone() for n, p in st.named_parameters() if p.grad is not None},
                     pred["bev_embed"].grad.clone(), ck)
    (l_t, lg_t, g_t, b_t, ck_t), (l_n, lg_n, g_n, b_n, ck_n) = res["t"], res["n"]
    assert ck_t._teachers is not None and ck_n._teachers is None          # the no-teacher run never built one
    assert torch.isfinite(l_n) and float(l_n) > 0
    assert torch.equal(l_t, l_n), (float(l_t), float(l_n))
    assert list(g_t) == list(g_n) and g_n
    for k in g_t:
        assert torch.equal(g_t[k], g_n[k]), k
    assert torch.equal(b_t, b_n) and float(b_n.abs().sum()) > 0
    # the teacher run computed a (zero-weighted) score KD; the no-teacher run has none
    assert float(lg_t["ck2/kd_score"]) > 0 and float(lg_t["ck2/wt_kd_score"]) == 0.0
    for k in ("ck2/kd_score", "ck2/wt_kd_score", "ck2/lat_kd", "ck2/wt_lat_kd", "ck2/w_lat", "ck2/kd_ok_frac"):
        assert float(lg_n[k]) == 0.0, k
    assert not any(k in lg_n for k in ("ck2/e_abs_T", "ck2/e_abs_det", "ck2/e_abs_map", "ck2/lat_disagree"))
    same = [k for k in lg_t if k.startswith(("ck2/bce", "ck2/sur", "ck2/t_", "ck2/G_", "ck2/n_", "ck2/wt_bce",
                                             "ck2/wt_sur", "ck2/loss", "ck2/phase", "ck2/ratio_v2"))]
    assert "ck2/bce" in same and "ck2/sur" in same
    for k in same:
        assert torch.equal(lg_t[k], lg_n[k]), k
    for k, v in lg_n.items():
        assert torch.isfinite(v).all(), k
    if phase == "onpolicy":
        assert float(lg_n["ck2/G_src_prev"]) == 0.5 and float(lg_n["ck2/G_src_fallback"]) == 0.5


# ----------------------------------------------------------------------------------------------- target builder
def test_target_builder_reads_no_teacher_bev(tmp_path):
    d = TL.batch()
    c = TL.cfg2(tmp_path, **NOKD)
    b = D2.CK2E2ETargetBuilder(ck_cfg=c)
    assert b.use_teachers is False and b.teacher_runs == ()
    for i, tok in enumerate(d["tokens"]):
        out = b.compute_for_token(tok)
        assert not any(k.startswith(KD_KEYS) for k in out), sorted(out)
        assert int(out["ck_row"]) == int(d["rows"][i])
        fix = {k: v[i] for k, v in d["targets"].items() if k.startswith(("ck_row", "ck2_", "ref_"))}
        assert sorted(out) == sorted(fix)
        for k in fix:
            assert torch.equal(out[k], fix[k]), k                 # = the teacher builder's non-KD targets
    assert b._loader()._caches is None                            # no teacher cache was ever opened
    # the teacher default still needs the DET run and reads both teacher BEVs
    tb = D2.CK2E2ETargetBuilder(ck_cfg=TL.cfg2(tmp_path))
    assert tb.use_teachers is True and tb.teacher_runs == (str(TL.SMOKE_T), str(TL.SMOKE_M))


# ----------------------------------------------------------------------------------------------- launcher
def _flat(ov):
    return {o.split("=", 1)[0].lstrip("+"): o for o in ov}


def test_launch_overrides_and_diff_to_v2_v15_v7(tmp_path):
    from tools.ck.e2e2 import launch_util2 as L2
    rd = lambda p: L2.LU.read_lines(str(p))  # noqa: E731
    ov, ck, teachers = L2.build_overrides2(str(tmp_path / "v2_nokd_test"), ["0", "1", "2", "3"], wandb="offline",
                                           ck_sets=rd(E2 / "v2_nokd.set"), extra=rd(E2 / "v2_nokd_arm.extra"))
    f = _flat(ov)
    assert teachers["mode"] == "none" and teachers["use_teachers"] is False and "det" not in teachers \
        and "map" not in teachers and teachers["warmup_sources"]["n_rows"] == 85109
    assert L2.arm_of(ck) == "nokd" and f["wandb.tags"] == "wandb.tags=[refiner,nokd,gpu4]"
    for k, v in (("use_teachers", "false"), ("teacher_det_run", "null"), ("teacher_map_run", "null"),
                 ("lambda_kd_score", "0.0"), ("lat_kd.balance", "fixed"), ("lat_kd.fixed", "0.0"),
                 ("kd_calib.enabled", "false"), ("bev_kd.enabled", "false"), ("lambda_score", "1.0"),
                 ("lambda_sur", "1.0")):
        assert f[f"agent.config.ck_e2e2.{k}"] == f"++agent.config.ck_e2e2.{k}={v}", k
    assert f["agent.config.use_distill"] == "++agent.config.use_distill=false"
    assert not any(k.startswith("agent.config.distill_") for k in f)
    assert f["trainer.params.accumulate_grad_batches"].endswith("=8") and f["trainer.params.devices"].endswith("=4")
    # against the v2_v15_v7 recipe on the same 4 GPUs: only the KD / v15 lines and the names differ
    ov7, _, _ = L2.build_overrides2(str(tmp_path / "v2_nokd_test"), ["0", "1", "2", "3"], wandb="offline",
                                    ck_sets=rd(E2 / "v2_v15_v7.set"), extra=rd(E2 / "v2_v15_v7_arm.extra"),
                                    teacher_deep=False)
    f7 = _flat(ov7)
    diff = sorted(k for k in set(f) | set(f7) if f.get(k) != f7.get(k))
    allowed = {"wandb.group", "wandb.tags", "agent.config.use_distill", "agent.config.ck_e2e2.use_teachers",
               "agent.config.ck_e2e2.teacher_det_run", "agent.config.ck_e2e2.teacher_map_run",
               "agent.config.ck_e2e2.lambda_kd_score", "agent.config.ck_e2e2.lat_kd.balance",
               "agent.config.ck_e2e2.lat_kd.fixed", "agent.config.ck_e2e2.kd_calib.enabled",
               "agent.config.ck_e2e2.kd_calib.det", "agent.config.ck_e2e2.kd_calib.map",
               "agent.config.ck_e2e2.kd_calib.mode"}
    extra_ok = [k for k in diff if k not in allowed]
    assert all(k.startswith("agent.config.distill_") and k not in f for k in extra_ok), extra_ok
    assert f7["agent.config.ck_e2e2.use_teachers"].endswith("=true")
    # a leftover teacher in the no-teacher set is refused before anything starts
    with pytest.raises(L2.LaunchError, match="use_teachers false"):
        L2.build_overrides2(str(tmp_path / "v2_nokd_bad"), ["0", "1", "2", "3"], wandb="offline",
                            ck_sets=rd(E2 / "v2_nokd.set") + [f"teacher_det_run={TL.SMOKE_T}"])


# ----------------------------------------------------------------------------------------------- agent
def test_agent_trains_without_teacher_targets(tmp_path):
    from test_ck2_e2e_agent import _fresh, build
    ck = {"enabled": True, "io_dir": str(tmp_path / "ck_e2e2"), "score_prior_rows": 200, **NOKD}
    agent = build(0, ck_e2e2=ck)
    tb = agent.get_target_builders()[-1]
    assert type(tb).__name__ == "CK2E2ETargetBuilder" and tb.teacher_runs == ()
    assert getattr(agent, "_distill", None) is None and getattr(agent, "ck_bev_kd", None) is None
    assert not any("teacher" in k for k in agent.state_dict())
    agent.train()
    f, t = _fresh(TL.batch())
    t = _no_kd_targets(t)
    torch.manual_seed(7)
    pred = agent.forward(f)
    loss = agent.compute_loss(f, t, pred)
    loss.backward()
    lg = agent.latest_logs
    assert torch.isfinite(loss) and float(lg["ck2/loss"]) > 0 and float(lg["ck2/kd_score"]) == 0.0
    assert agent._ck_e2e2._teachers is None
    st = agent.ck_student
    assert not [n for n, p in st.named_parameters() if p.requires_grad and p.grad is None]
