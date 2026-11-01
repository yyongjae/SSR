"""CK Phase 2 model unit tests (contract model.tests 1-9, CPU): config parsing / validation, student_topk, bev_sgrid +
gradient scale g, replay loss == tools/ck/train_ck.compute_loss, on-policy loss (online teachers, LabelStore G) finite
with gradients on every student parameter, recording -> e2e_data.read_rec_chunk, callback state / schedule / step skip
/ CK clip, log_sync_dist, gloo 2-process DDP (find_unused_parameters False) in replay and on-policy.
Real-agent tests: test_ck_e2e_model_agent.py; GPU teacher parity: test_ck_e2e_model_gpu.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.ck import online as O
from navsim.agents.para_ssr.ck.constants import SUR_TERMS

REPO = Path(__file__).resolve().parents[1]
ANCHORS = "/home/external-user/kyungmin/SSR-v2/data/planning_vb/trajectory_anchors_256.npy"
FIX = Path("/workspace/yongjae/ssd/yongjae_refiner/ck/phase2/impl-model/fixtures/real_b2.pt")


# ----------------------------------------------------------------------------------------------- helpers
def real_batch():
    if not FIX.is_file():
        sys.path.insert(0, str(REPO / "tests"))
        from test_ck_e2e_model_agent import make_fixture
        make_fixture(FIX)
    return torch.load(FIX, map_location="cpu", weights_only=False)


def fake_predictions(B: int, seed: int = 0, bev=None, requires_grad: bool = True):
    g = torch.Generator().manual_seed(seed)
    anchors = torch.from_numpy(np.load(ANCHORS)).float()
    off = 0.2 * torch.randn(B, 256, 8, 3, generator=g)
    final = torch.randn(B, 256, generator=g)
    im = torch.softmax(torch.randn(B, 256, generator=g), -1)
    sim = torch.sigmoid(torch.randn(B, 5, 256, generator=g))
    if bev is None:
        bev = torch.randn(B, 5000, 256, generator=g)
        bev = (bev - bev.mean(-1, keepdim=True)) / bev.std(-1, keepdim=True)
        bev.requires_grad_(requires_grad)
    top = final.argmax(-1)
    traj = (anchors[None] + off)[torch.arange(B), top]
    return {"bev_embed": bev, "plan_final_rewards": final, "trajectory_offset": off, "trajectory_anchors": anchors,
            "im_rewards": im, "sim_rewards": sim, "trajectory": traj}


def cfg(tmp: Path, **over) -> O.CKE2EConfig:
    d = {"enabled": True, "io_dir": str(tmp / "ck_e2e"), "grad_share_every": 0}
    d.update(over)
    return O.CKE2EConfig.from_any(d)


def make_teacher_runs(tmp: Path, seed: int = 0):
    """random CK teachers (arms T / M) saved in load_ck run-dir format, control heads perturbed (non-identity)."""
    from navsim.agents.para_ssr.ck.model import CKNet, load_arm_norm, save_ck
    runs = []
    real = {"T": "/home/external-user/ssd/yongjae_refiner/ck/train/ckT_p1",
            "M": "/home/external-user/ssd/yongjae_refiner/ck/train/ckM_p1"}
    for arm, s in (("T", seed), ("M", seed + 1)):
        torch.manual_seed(100 + s)
        net = CKNet(arm, s, norm=load_arm_norm(real[arm], arm))      # the real teachers' BEV z-score
        with torch.no_grad():
            for head in (net.trunk.lon_head, net.trunk.lat_head):
                for p in head.parameters():
                    p.add_(0.05 * torch.randn_like(p))
        d = tmp / f"teacher_{arm}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps({"arm": arm, "seed": s}))
        save_ck(d / "ckpt_last.pt", net, {"arm": arm})
        runs.append(str(d))
    return runs


class FakeLabels:
    """LabelStore stand-in: G = (16 perturbed anchors, 16 more), labels in [0, 1], some rows / candidates not ok."""

    def __init__(self, seed=0):
        self.g = torch.Generator().manual_seed(seed)
        self.refreshed = []
        self.anchors = torch.from_numpy(np.load(ANCHORS)).float()

    def refresh(self, max_epoch):
        self.refreshed.append(int(max_epoch))
        return {"n_prev": 1, "n_older": 0, "n_phase1": 0, "gens": [max_epoch]}

    def lookup(self, rows, device=None):
        B = len(rows)
        traj = self.anchors[:32][None].repeat(B, 1, 1, 1) + 0.1 * torch.randn(B, 32, 8, 3, generator=self.g)
        y = torch.rand(B, 32, 5, generator=self.g).round()
        ok = torch.ones(B, 32, dtype=torch.bool)
        ok[0, 3] = False
        traj[0, 3] = 0.0                      # LabelStore zeroes non-finite / not-ok rows
        src = torch.tensor([1] + [2] * (B - 1), dtype=torch.int8)
        rr = torch.as_tensor(rows).reshape(-1)
        ok[rr < 0] = False
        return {"G_traj": traj, "G_y": y, "G_ok": ok, "G_src": src, "G_epoch": torch.full((B,), 4, dtype=torch.int16)}


def replay_targets(B: int = 2):
    d = real_batch()
    t = {k: v[:B].clone() for k, v in d["targets"].items()}
    f = {k: v[:B].clone() for k, v in d["features"].items()}
    return f, t


# ----------------------------------------------------------------------------------------------- (0) config
def test_config_parse_and_validate(tmp_path):
    from omegaconf import OmegaConf
    c = O.CKE2EConfig.from_any(None)
    assert not c.enabled and c.topk == 16 and c.bev_grad_scale == 1.0 and c.lr_mult == 3.0
    assert len(c.to_dict()) == 36
    d = OmegaConf.create({"enabled": True, "io_dir": str(tmp_path), "kd_ema": {"start_mb": 7},
                          "phase1": {"packed": "/x"}})
    c = O.CKE2EConfig.from_any(d)
    assert c.kd_ema["start_mb"] == 7 and c.kd_ema["cap"] == 10.0 and c.phase1["packed"] == "/x"
    assert c.phase1["labels_cand"].endswith("labels/navtrain_train/cand")
    assert O.CKE2EConfig.from_any(c.to_dict()) == c
    with pytest.raises(ValueError, match="unknown"):
        O.CKE2EConfig.from_any({"enabld": True})
    with pytest.raises(ValueError, match="unknown"):
        O.CKE2EConfig.from_any({"kd_ema": {"mm": 1}})
    O.CKE2EConfig.from_any({"enabled": False, "io_dir": "rel"}).validate()       # off: nothing checked
    bad = [({"io_dir": "rel/path"}, "absolute"), ({"io_dir": "/a=b"}, "absolute"),
           ({"record_from_epoch": 6}, "record_from_epoch"), ({"label_lag": 0}, "label_lag"),
           ({"topk": 0}, "topk"), ({"score_prior": "x"}, "score_prior")]
    for over, msg in bad:
        kw = {"enabled": True, "io_dir": str(tmp_path)}
        kw.update(over)
        with pytest.raises(ValueError, match=msg):
            O.CKE2EConfig.from_any(kw).validate()
    runs = make_teacher_runs(tmp_path)
    with pytest.raises(ValueError, match="arm"):
        O.CKE2EConfig.from_any({"enabled": True, "io_dir": str(tmp_path), "teacher_det_run": runs[1]}).validate()
    O.CKE2EConfig.from_any({"enabled": True, "io_dir": str(tmp_path), "teacher_det_run": runs[0],
                            "teacher_map_run": runs[1]}).validate()
    parent = SimpleNamespace(refiner_mode="E2", plan_anchor=True, bev_h=50, bev_w=100,
                             pc_range=(-32.0, 0.0, -2.0, 32.0, 32.0, 2.0), embed_dims=256)
    with pytest.raises(ValueError, match="refiner_mode"):
        cfg(tmp_path).validate(parent)
    parent.refiner_mode, parent.plan_anchor = "off", False
    with pytest.raises(ValueError, match="plan_anchor"):
        cfg(tmp_path).validate(parent)


def test_schedule_phase_r_recording(tmp_path):
    ck = O.CKE2E(cfg(tmp_path), max_epochs=30)
    assert [ck.phase(e) for e in (0, 3, 4, 5, 29)] == ["replay", "replay", "replay_record", "onpolicy", "onpolicy"]
    assert ck.r(2.5) == 1.0 and ck.r(4.9) == 1.0 and ck.r(5.0) == 0.0
    assert abs(ck.r(6.5) - 0.5) < 1e-12 and ck.r(8.0) == 1.0 and ck.r(20.0) == 1.0
    assert [ck.recording(e) for e in (3, 4, 28, 29)] == [False, True, True, False]
    ck2 = O.CKE2E(cfg(tmp_path, record_from_epoch=1, onpolicy_from_epoch=2, kd_ramp_epochs=0.5,
                      record_until_epoch=2), max_epochs=4)
    assert [ck2.phase(e) for e in range(4)] == ["replay", "replay_record", "onpolicy", "onpolicy"]
    assert ck2.r(2.25) == 0.5 and [ck2.recording(e) for e in range(4)] == [False, True, True, False]


# ----------------------------------------------------------------------------------------------- (3) topk / sgrid
def test_student_topk_matches_dump_v2_and_planner():
    from navsim.agents.para_ssr.modules.anchor_planner import AnchorPlanner
    torch.manual_seed(0)
    m = AnchorPlanner(ANCHORS, embed_dims=32, topk=6, heading_from_xy=True).eval()
    with torch.no_grad():
        out = m(m.queries(torch.randn(3, 1, 32)))
    top = O.student_topk(out, 16)
    final = out["plan_final_rewards"].float()
    idx = final.topk(16, dim=-1).indices
    refined = out["trajectory_anchors"].unsqueeze(0) + out["trajectory_offset"].float()
    cand = refined.gather(1, idx[:, :, None, None].expand(-1, -1, 8, 3))           # dump_v2.py:236-240
    assert torch.equal(top["idx"], idx) and torch.equal(top["cand"], cand)
    assert float((top["cand"][:, 0] - out["trajectory"]).abs().max()) < 1e-4
    assert torch.equal(top["idx"][:, :6], out["plan_topk_index"])
    assert torch.equal(top["final"], final.gather(1, idx)) and torch.equal(top["im"], out["im_rewards"].gather(1, idx))
    assert torch.equal(top["sim"], out["sim_rewards"].gather(2, idx[:, None].expand(-1, 5, -1)).transpose(1, 2))
    assert top["cand"].dtype == torch.float32 and not top["cand"].requires_grad


def test_bev_sgrid_layout():
    x = torch.randn(2, 5000, 256)
    s = O.bev_sgrid(x)
    assert s.shape == (2, 256, 50, 100)
    assert torch.equal(s[1, :, 7, 33], x[1, 7 * 100 + 33])
    assert torch.equal(s.flatten(2).transpose(1, 2), x)


# ----------------------------------------------------------------------------------------------- (2) replay parity
def _train_ck_args():
    from navsim.agents.para_ssr.ck.constants import LOSS_DEFAULTS as LD
    return SimpleNamespace(lambda_score=LD["lambda_score"], corr_aug="on", lambda_corr_score=LD["lambda_corr_score"],
                           lambda_sur=LD["lambda_sur"], kd="on", lambda_kd_score=LD["lambda_kd_score"],
                           kd_ctrl_balance="ema", kd_ctrl_weight=1.0, lead_aux=0, lambda_lead=0.0)


def test_replay_loss_equals_train_ck(tmp_path):
    """Same student, same BEV tensor, same Phase 1 batch (2 real tokens): CKE2E replay loss == train_ck.compute_loss
    over 3 micro-batches with the EMA KD weight on from step 1 (1e-5)."""
    sys.path.insert(0, str(REPO))
    from tools.ck import train_ck as TC
    from navsim.agents.para_ssr.ck.losses import EmaBalancer

    f, t = replay_targets(2)
    assert bool(t["ck_p1_ok"].all()) and bool(t["ref_gt_ok"].any())
    c = cfg(tmp_path, kd_ema={"start_mb": 1})
    ck = O.CKE2E(c, max_epochs=30)
    torch.manual_seed(0)
    student = O.build_student_ck(c)
    bal = EmaBalancer(ratio=1.0, m=0.99, floor=1e-4, cap=10.0, start_step=1)
    a = _train_ck_args()
    for step in range(3):
        pred = fake_predictions(2, seed=step)
        loss_e, logs = ck.loss(student, f, t, pred)
        p1 = t["ck_p1_ok"].bool()
        batch = {"cand": t["ck_p1_cand"], "corr_traj": t["ck_p1_kdc"], "status": f["status_feature"],
                 "y": t["ck_p1_y"], "y_ok": t["ck_p1_y_ok"] & p1[:, None],
                 "y_corr": t["ck_p1_y_kdc"], "y_corr_ok": t["ck_p1_y_kdc_ok"] & p1[:, None],
                 "kd_score_prob": t["ck_p1_kd_prob"], "kd_score_prob_corr": t["ck_p1_kd_prob_corr"],
                 "kd_c_lon": t["ck_p1_kd_c_lon"], "kd_e_lat": t["ck_p1_kd_e_lat"], "kd_ok": t["ck_p1_kd_ok"] & p1[:, None],
                 "gt_traj": t["trajectory"], **{k: v for k, v in t.items() if k.startswith("ref_")}}
        out = TC.forward(student, {**batch, "bev": O.bev_sgrid(pred["bev_embed"])}, False, torch.device("cpu"),
                         0.1, with_extra=True)
        loss_t, st = TC.compute_loss(student, out, batch, a, bal, step)
        assert abs(float(loss_e) - float(loss_t)) < 1e-5 * max(1.0, abs(float(loss_t))), (step, float(loss_e),
                                                                                         float(loss_t))
        assert abs(float(logs["ck/w_ema"]) - st["kd_w"]) < 1e-6 * max(1.0, st["kd_w"])   # log = f32 tensor
        assert abs(float(logs["ck/sur"]) - st.get("sur", 0.0)) < 1e-6
        assert abs(float(logs["ck/kd_score"]) - st["kd_score"]) < 1e-6
    assert ck.mb == 3 and ck.ema.n == 3 and float(logs["ck/w_ema"]) > 0


# ----------------------------------------------------------------------------------------------- (4) gradient scale g
def test_bev_gradient_proportional_to_g(tmp_path):
    f, t = replay_targets(2)
    grads = []
    for g in (1.0, 0.25):
        c = cfg(tmp_path, bev_grad_scale=g)
        ck = O.CKE2E(c)
        student = O.build_student_ck(c)
        assert student.trunk.adapter.bev_grad_scale == g
        pred = fake_predictions(2, seed=0)
        loss, _ = ck.loss(student, f, t, pred)
        loss.backward()
        grads.append(pred["bev_embed"].grad.clone())
    assert grads[0].abs().sum() > 0
    torch.testing.assert_close(grads[1], 0.25 * grads[0], rtol=1e-5, atol=1e-10)


# ----------------------------------------------------------------------------------------------- on-policy / recording
def _onpolicy_setup(tmp_path, **over):
    runs = make_teacher_runs(tmp_path)
    c = cfg(tmp_path, teacher_det_run=runs[0], teacher_map_run=runs[1], teacher_amp=False, **over)
    ck = O.CKE2E(c, max_epochs=30)
    ck._labels = FakeLabels()
    return c, ck


@pytest.mark.parametrize("rescore", [True, False])
def test_onpolicy_loss_finite_backward(tmp_path, rescore):
    f, t = replay_targets(2)
    t["ck_row"] = torch.tensor([5, -1])
    c, ck = _onpolicy_setup(tmp_path, teacher_rescore_kd=rescore, kd_ema={"start_mb": 0}, strict_rows=0.0)
    ck.epoch, ck.epoch_frac = 7, 7.5
    torch.manual_seed(0)
    student = O.build_student_ck(c)
    pred = fake_predictions(2, seed=3)
    v2 = (pred["bev_embed"] ** 2).mean()
    loss, logs = ck.loss(student, f, t, pred, v2_loss=v2)
    assert ck._labels.refreshed == [6]                       # epoch 7, lag 1
    assert torch.isfinite(loss) and float(logs["ck/loss"]) > 0
    assert float(logs["ck/phase"]) == 2.0 and abs(float(logs["ck/r"]) - 2.5 / 3) < 1e-6
    assert float(logs["ck/G_src_prev"]) == 0.5 and float(logs["ck/G_src_older"]) == 0.5
    assert float(logs["ck/G_ok_frac"]) < 0.5 + 1e-6           # row -1 -> no G
    assert float(logs["ck/kd_score"]) > 0 and float(logs["ck/kd_ctrl"]) > 0 and float(logs["ck/w_ema"]) > 0
    for k in ("ck/raw_fail_nc", "ck/raw_fail_dac", "ck/raw_fail_ttc", "ck/teacher_ms", "ck/student_ms"):
        assert k in logs
    for k in SUR_TERMS:
        assert f"ck/t_{k}" in logs
    for k, v in logs.items():
        assert torch.isfinite(v).all(), k
    (v2 + loss).backward()
    missing = [n for n, p in student.named_parameters() if p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in student.parameters())
    assert pred["bev_embed"].grad is not None and torch.isfinite(pred["bev_embed"].grad).all()


def test_onpolicy_kd_terms_scale_with_r(tmp_path):
    """r = 0 at the start of on-policy: score / control KD weighted terms vanish, the rest is unchanged."""
    f, t = replay_targets(2)
    t["ck_row"] = torch.tensor([5, 6])
    c, ck = _onpolicy_setup(tmp_path, kd_ema={"start_mb": 0})
    student = O.build_student_ck(c)
    ck.epoch, ck.epoch_frac = 5, 5.0
    pred = fake_predictions(2, seed=1)
    ck._labels = FakeLabels(0)
    l0, g0 = ck.loss(student, f, t, pred)
    assert float(g0["ck/r"]) == 0.0 and float(g0["ck/wt_kd_score"]) == 0.0 and float(g0["ck/wt_kd_ctrl"]) == 0.0
    expect = sum(float(g0[k]) for k in ("ck/wt_bce_cand", "ck/wt_bce_tkd", "ck/wt_sur"))
    assert abs(float(l0) - expect) < 1e-5


def test_grad_share_logged(tmp_path):
    f, t = replay_targets(2)
    c = cfg(tmp_path, grad_share_every=2)
    ck = O.CKE2E(c)
    student = O.build_student_ck(c)
    for i in range(3):
        pred = fake_predictions(2, seed=i)
        v2 = (pred["bev_embed"] ** 2).mean()
        _, logs = ck.loss(student, f, t, pred, v2_loss=v2)
        has = "ck/gshare_ck" in logs
        assert has == (i % 2 == 0)
        if has:
            assert 0 < float(logs["ck/gshare_ck"]) < 1 and float(logs["gnorm/bev_v2"]) > 0


def test_replay_record_writes_chunks(tmp_path):
    """replay_record epoch: loss = replay, the online teacher runs on the student's top-16 and (top-16, tau'_KD,
    kd_ok) are recorded; CandRecorder -> e2e_data.read_rec_chunk round trip; DONE lists the chunks."""
    from navsim.agents.para_ssr.ck import e2e_data as D
    f, t = replay_targets(2)
    t["ck_row"] = torch.tensor([11, -1])
    c, ck = _onpolicy_setup(tmp_path, rec_chunk_tokens=2)
    student = O.build_student_ck(c)
    ck.epoch, ck.epoch_frac, ck.gstep = 4, 4.0, 17
    ck.recorder = O.CandRecorder(c.io_dir, 0, 2, 4, c.rec_chunk_tokens)
    seen = []
    for i in range(3):
        pred = fake_predictions(2, seed=10 + i)
        loss, logs = ck.loss(student, f, t, pred)
        assert float(logs["ck/phase"]) == 1.0 and float(logs["ck/rec_tokens"]) == 1.0
        top = O.student_topk(pred, 16)
        T = ck.teachers(torch.device("cpu")).run(t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"],
                                                 top["cand"], f["status_feature"], rescore=False)
        seen.append((top["cand"][0].numpy(), top["idx"][0].numpy(), T["kd_corr"][0].numpy(), T["kd_ok"][0].numpy()))
    rec = ck.recorder
    assert rec.chunks == [f"c_{rec.attempt}_000000.npz"] and rec.n_tokens == 2
    rec.close_epoch()
    assert len(rec.chunks) == 2 and rec.n_tokens == 3 and rec.seq == 2
    files = sorted(Path(D.rec_rank_dir(c.io_dir, 4, 0)).glob("c_*.npz"))
    assert len(files) == 2
    arrs = [D.read_rec_chunk(p) for p in files]
    rows = np.concatenate([a["row"] for a in arrs])
    assert rows.tolist() == [11, 11, 11]
    cand = np.concatenate([a["cand"] for a in arrs])
    kdc = np.concatenate([a["kd_corr"] for a in arrs])
    for j, (cd, ix, kc, ko) in enumerate(seen):
        np.testing.assert_allclose(cand[j], cd, atol=0)
        np.testing.assert_allclose(kc, kdc[j], atol=1e-6)
    assert all(int(a["epoch"]) == 4 and int(a["rank"]) == 0 for a in arrs)
    assert np.concatenate([a["gstep"] for a in arrs]).tolist() == [17, 17, 17]
    done = json.loads((Path(D.rec_rank_dir(c.io_dir, 4, 0)) / f"DONE_{rec.attempt}.json").read_text())
    assert done["n_tokens"] == 3 and done["world_size"] == 2 and len(done["chunks"]) == 2


def test_recorder_roundtrip_and_flush(tmp_path):
    from navsim.agents.para_ssr.ck import e2e_data as D
    r = O.CandRecorder(str(tmp_path), 1, 4, 9, chunk_tokens=3)
    g = np.random.default_rng(0)
    added = []
    for step in range(3):
        rows = np.array([step * 10, -1, step * 10 + 1])
        cand = g.normal(size=(3, 16, 8, 3)).astype(np.float32)
        idx = g.integers(0, 256, size=(3, 16))
        kdc = g.normal(size=(3, 16, 8, 3)).astype(np.float32)
        ok = g.random((3, 16)) > 0.2
        r.add(torch.from_numpy(rows), torch.from_numpy(cand), torch.from_numpy(idx), torch.from_numpy(kdc),
              torch.from_numpy(ok), step)
        for i in (0, 2):
            added.append((rows[i], cand[i], idx[i], kdc[i], ok[i], step))
    r.flush()                              # partial chunk, no DONE (interrupted epoch)
    d = Path(D.rec_rank_dir(str(tmp_path), 9, 1))
    assert not list(d.glob("DONE_*")) and len(list(d.glob("c_*.npz"))) == 2
    arrs = [D.read_rec_chunk(p) for p in sorted(d.glob("c_*.npz"))]
    cat = {k: np.concatenate([a[k] for a in arrs]) for k in ("row", "cand", "cand_idx", "kd_corr", "kd_ok", "gstep")}
    assert cat["row"].tolist() == [x[0] for x in added]
    for j, (_, cd, ix, kc, ko, st) in enumerate(added):
        assert np.array_equal(cat["cand"][j], cd) and np.array_equal(cat["cand_idx"][j], ix.astype(np.int16))
        assert np.array_equal(cat["kd_corr"][j], kc) and np.array_equal(cat["kd_ok"][j], ko)
        assert cat["gstep"][j] == st
    r.close_epoch()
    assert len(list(d.glob("DONE_*.json"))) == 1
    with pytest.raises(RuntimeError):
        r.add(np.array([1]), np.zeros((1, 16, 8, 3)), np.zeros((1, 16)), np.zeros((1, 16, 8, 3)),
              np.ones((1, 16), bool), 0)


# ----------------------------------------------------------------------------------------------- (7) callback
class _Trainer(SimpleNamespace):
    pass


def _stub_agent(tmp_path, ck):
    student = O.build_student_ck(ck.cfg)
    return SimpleNamespace(_ck_e2e=ck, ck_student=student, latest_logs={}, config=SimpleNamespace())


def test_callback_state_roundtrip_and_flow(tmp_path):
    c, ck = _onpolicy_setup(tmp_path, record_from_epoch=1, onpolicy_from_epoch=2, rec_chunk_tokens=2,
                            label_refresh_every_mb=2)
    agent = _stub_agent(tmp_path, ck)
    cb = O.make_ck_callback(agent)
    ck.ema.update(0.3, 0.1)
    ck.ema.update(0.2, 0.2)
    ck.mb, ck.skipped_steps, ck.cum["tokens"] = 1234, 2, 99
    sd = json.loads(json.dumps(cb.state_dict()))            # checkpoint-serialisable
    ck2 = O.CKE2E(c)
    cb2 = O.make_ck_callback(SimpleNamespace(_ck_e2e=ck2))
    cb2.load_state_dict(sd)
    assert ck2.ema.state_dict() == ck.ema.state_dict() and ck2.mb == 1234 and ck2.skipped_steps == 2
    assert ck2.cum["tokens"] == 99 and cb.state_key == "CKE2ECallback"
    # epoch flow: epoch 1 = replay_record (recorder), epoch 2 = onpolicy (labels refreshed with max_epoch 1)
    f, t = replay_targets(2)
    t["ck_row"] = torch.tensor([3, 4])
    tr = _Trainer(current_epoch=1, global_rank=0, world_size=1, is_global_zero=True, global_step=0,
                  num_training_batches=2)
    cb.on_train_epoch_start(tr, None)
    assert ck.recorder is not None and ck.phase(ck.epoch) == "replay_record"
    for bi in range(2):
        cb.on_train_batch_start(tr, None, None, bi)
        assert ck.epoch_frac == 1 + bi / 2
        _, logs = ck.loss(agent.ck_student, f, t, fake_predictions(2, seed=bi))
        agent.latest_logs = logs
        cb.on_train_batch_end(tr, None, None, None, bi)
    att = ck.recorder.attempt
    cb.on_train_epoch_end(tr, None)
    assert ck.recorder is None
    from navsim.agents.para_ssr.ck import e2e_data as D
    rd = Path(D.rec_rank_dir(c.io_dir, 1, 0))
    assert (rd / f"DONE_{att}.json").is_file() and len(list(rd.glob("c_*.npz"))) == 2
    tr.current_epoch = 2
    cb.on_train_epoch_start(tr, None)
    assert ck._labels.refreshed[-1] == 1 and ck.recorder is not None
    cb.on_train_batch_start(tr, None, None, 0)
    _, logs = ck.loss(agent.ck_student, f, t, fake_predictions(2, seed=5))
    agent.latest_logs = logs
    cb.on_train_batch_end(tr, None, None, None, 0)
    cb.on_exception(tr, None, RuntimeError("x"))             # interrupted: chunks flushed, no DONE
    rd2 = Path(D.rec_rank_dir(c.io_dir, 2, 0))
    assert not list(rd2.glob("DONE_*")) and len(list(rd2.glob("c_*.npz"))) == 1
    steps = [json.loads(l) for l in (Path(c.io_dir) / "steps_rank0.jsonl").read_text().splitlines()]
    assert len(steps) == 3 and all("ck/loss" in s and "mb" in s for s in steps)
    ev = [json.loads(l) for l in (Path(c.io_dir) / "epochs.jsonl").read_text().splitlines()]
    assert [e["event"] for e in ev] == ["epoch_start", "epoch_end", "epoch_start"]
    assert ev[2]["labels"]["max_epoch"] == 1
    lw = ev[2]["label_warning"]                               # 4 rows recorded in epoch 1 (rec DONE n_tokens)
    assert lw["src_epoch"] == 1 and lw["n_rec"] == 4 and "warn" in lw
    assert ev[0]["label_warning"] is None                     # not an on-policy epoch
    cb.on_train_end(tr, None)
    assert (Path(c.io_dir) / "TRAIN_DONE").is_file()


def test_label_supply_warning(tmp_path):
    """Dead / slow labeler: rows recorded in the label-source epoch (rec DONE over ranks) vs rows already labelled
    from that generation (refresh n_prev).  Logging only."""
    from navsim.agents.para_ssr.ck import e2e_data as D
    c = cfg(tmp_path, record_from_epoch=4, onpolicy_from_epoch=5)
    ck = O.CKE2E(c, max_epochs=30)
    ck.epoch = 6
    st = {"max_epoch": 5, "n_prev": 900}
    w = O.label_supply_warning(ck, st)
    assert w["warn"] and w["n_rec"] == 0                      # nothing recorded in epoch 5: warn
    for r in range(4):
        D.write_rec_done(c.io_dir, 5, r, 4, "0000000a", [], 250)
    D.write_rec_done(c.io_dir, 5, 0, 4, "0000000b", [], 240)   # an earlier / other attempt: max per rank counts
    w = O.label_supply_warning(ck, st)
    assert w["n_rec"] == 1000 and w["frac_labeled"] == pytest.approx(0.9) and not w["warn"]
    w = O.label_supply_warning(ck, dict(st, n_prev=300))
    assert w["warn"] and w["frac_labeled"] == pytest.approx(0.3) and "labeler" in w["msg"]
    ck.epoch = 4                                              # source epoch 3 < record_from: no expectation
    assert not O.label_supply_warning(ck, {"max_epoch": 3, "n_prev": 0})["warn"]
    assert O.label_supply_warning(ck, None) is None


def test_callback_skip_nonfinite_and_ck_clip(tmp_path):
    c = cfg(tmp_path, clip=1.0)
    ck = O.CKE2E(c)
    agent = _stub_agent(tmp_path, ck)
    other = torch.nn.Linear(3, 3)
    mod = torch.nn.ModuleDict({"ck": agent.ck_student, "v2": other})
    cb = O.make_ck_callback(agent)
    tr = _Trainer(is_global_zero=True, global_step=5)
    for p in mod.parameters():
        p.grad = torch.full_like(p, 10.0)
    cb.on_before_optimizer_step(tr, mod, None)
    gn = torch.norm(torch.stack([p.grad.norm() for p in agent.ck_student.parameters()]))
    assert abs(float(gn) - 1.0) < 1e-4 and ck.last_clip_norm > 1.0
    assert torch.all(other.weight.grad == 10.0)                 # v2 grads untouched (Lightning clips them later)
    other.weight.grad[0, 0] = float("nan")
    cb.on_before_optimizer_step(tr, mod, None)
    assert ck.skipped_steps == 1 and all(p.grad is None for p in mod.parameters())


def test_nonfinite_ck_loss_is_dropped(tmp_path):
    f, t = replay_targets(2)
    c = cfg(tmp_path, kd_ema={"start_mb": 0})
    ck = O.CKE2E(c)
    student = O.build_student_ck(c)
    pred = fake_predictions(2, requires_grad=False)
    bev = pred["bev_embed"].clone()
    bev[0, 10, 3] = float("inf")
    pred["bev_embed"] = bev.requires_grad_(True)
    loss, logs = ck.loss(student, f, t, pred)
    assert float(loss) == 0.0 and float(logs["ck/loss_nonfinite"]) == 1.0 and ck.cum["nonfinite_loss"] == 1
    loss.backward()
    assert all(p.grad is not None for p in student.parameters())


# ----------------------------------------------------------------------------------------------- (9) log_sync_dist
@pytest.mark.parametrize("flag", [True, False, None])
def test_log_sync_dist(flag):
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRLoggingCallback
    from navsim.planning.training.agent_lightning_module import AgentLightningModule

    conf = SimpleNamespace() if flag is None else SimpleNamespace(log_sync_dist=flag)
    want = True if flag is None else flag

    class _Agent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = conf
            self.latest_logs = {"a": torch.tensor(1.0), "loss": torch.tensor(2.0)}

        def forward(self, f):
            return {}

        def compute_loss(self, f, t, p):
            return torch.tensor(3.0)

    calls = []
    m = AgentLightningModule(_Agent())
    m.log = lambda name, value, **kw: calls.append((name, kw))
    m._step(({}, {}), "train")
    ParaSSRLoggingCallback()._log(m, "train")
    assert [c[0] for c in calls] == ["train/traj_loss", "train/a"]
    assert all(c[1]["sync_dist"] is want for c in calls)


# ----------------------------------------------------------------------------------------------- (5) DDP gloo
def _ddp_worker(rank, world, port, tmp, runs, out_q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        f, t = replay_targets(2)
        f = {k: v[rank:rank + 1] for k, v in f.items()}
        t = {k: v[rank:rank + 1] for k, v in t.items()}
        t["ck_row"] = torch.tensor([7 + rank])
        res = {}
        for phase_epoch in (0, 6):
            c = O.CKE2EConfig.from_any({"enabled": True, "io_dir": f"{tmp}/ck_e2e", "teacher_det_run": runs[0],
                                        "teacher_map_run": runs[1], "teacher_amp": False, "grad_share_every": 0,
                                        "kd_ema": {"start_mb": 0}, "strict_rows": 0.0})
            ck = O.CKE2E(c)
            ck._labels = FakeLabels(seed=rank)
            ck.epoch, ck.epoch_frac, ck.rank = phase_epoch, phase_epoch + 0.5, rank

            class Wrap(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    torch.manual_seed(0)
                    self.bev = torch.nn.Parameter(0.1 * torch.randn(1, 5000, 256))
                    self.ck_student = O.build_student_ck(c)

                def forward(self, f, t):
                    pred = fake_predictions(1, seed=rank, bev=self.bev * 1.0)
                    v2 = (pred["bev_embed"] ** 2).mean()
                    loss, logs = ck.loss(self.ck_student, f, t, pred, v2_loss=v2)
                    return v2 + loss, logs

            model = DDP(Wrap(), find_unused_parameters=False)
            loss, logs = model(f, t)
            loss.backward()
            g = torch.cat([p.grad.reshape(-1) for p in model.module.ck_student.parameters()])
            gl = [torch.zeros_like(g) for _ in range(world)]
            dist.all_gather(gl, g)
            res[phase_epoch] = {"finite": bool(torch.isfinite(g).all()), "same": bool(torch.equal(gl[0], gl[1])),
                                "phase": float(logs["ck/phase"]), "nonzero": bool(g.abs().sum() > 0)}
        out_q.put((rank, res))
    except Exception as e:  # pragma: no cover
        import traceback
        out_q.put((rank, {"error": traceback.format_exc() + repr(e)}))
    finally:
        dist.destroy_process_group()


def test_ddp_gloo_two_process(tmp_path):
    import socket
    import torch.multiprocessing as mp
    real_batch()
    runs = make_teacher_runs(tmp_path)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_ddp_worker, args=(r, 2, port, str(tmp_path), runs, q)) for r in range(2)]
    for p in ps:
        p.start()
    res = dict(q.get(timeout=900) for _ in ps)
    for p in ps:
        p.join(timeout=120)
    for r in (0, 1):
        assert "error" not in res[r], res[r].get("error")
        for ep, phase in ((0, 0.0), (6, 2.0)):
            v = res[r][ep]
            assert v["finite"] and v["same"] and v["nonzero"] and v["phase"] == phase
