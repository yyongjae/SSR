"""report 129 P5: training integration (CPU unless noted).

  * F25 (A30 / A42): DistillScheduleCallback round-trips the v15 distillation schedule (micro_step, ratio controllers,
    probe ema_step + ema__* shadows) through Lightning's own callback-state glue; the next forward after a restore logs
    the same ramp / lam_* as the uninterrupted agent; an old checkpoint falls back to loss.iteration; the model
    state_dict keys are unchanged and an agent without distillation registers no such callback; a toy module
    resumes through a real PL 2.2.1 Trainer (fit(ckpt_path)) with the state and without it
  * F27 (A47): ParaSSRLoggingCallback.on_before_optimizer_step logs clip_grad_norm_'s norm, never touches .grad
  * F28 (A28, GPU): torch_tracker.track is eager under inference_mode (no graph key), a later training-shape call works
  * F29 (A51): v6c needs corridor_select_rule=cand_official; v6c off builds with any rule
  * F30 / F31 (A48 / A32): train.sh DRYRUN: v6c warm-up 26600 at 4 GPUs / 13300 at 8, cf_seg_v15 byte-identical to the
    pre-report-129 train.sh (R129_BASE_REV, default 2e674b4), the v6c config differs from it only in the warm-up
The agent tests need the dataset, the GT caches and the teacher BEV caches of this host (skipped when absent).
"""
from __future__ import annotations

import io
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_refiner_v6c_agent import FIX, REPO, TEACH, _fresh, build_agent, have_inputs  # noqa: E402

# train.sh's V15_DISTILL (cf_seg_v15) with a short probe warm-up / ramp so a few micro-batches cover both
V15 = dict(use_distill=True, distill_feature_root=str(TEACH), distill_teacher_source="targets",
           distill_cache_size=[100, 100], distill_selector_v15=True, distill_anchor_mode="cross_attn",
           distill_anchor_proj=True, distill_anchor_loss_type="hybrid", distill_anchor_loss_scale=10.0,
           distill_attn_loss_scale=5.0, distill_resmap_scale=3.0, distill_lambda_gt=0.45, distill_lambda_cov=0.20,
           distill_lambda_risk=0.35, distill_tau_gt=4.5, distill_tau_risk=8.0, distill_loss_weight=1.0,
           distill_probe_seed=0, distill_proj_lr_mult=3.0, distill_probe_lr_mult=3.0, distill_probe_warmup_steps=1,
           distill_probe_ramp_steps=4, distill_probe_loss_weight=0.5, distill_probe_use_ema=True,
           distill_probe_ema_decay=0.99)
BASE_REV = os.environ.get("R129_BASE_REV", "2e674b4")      # train.sh before report 129


@pytest.fixture(scope="module")
def batch1():
    """the 2-token v6c fixture cut to its first token (B = 1) + scene_token (the distillation's token list)"""
    if not have_inputs():
        pytest.skip("dataset / GT caches / teacher caches not on this host")
    if not FIX.is_file():
        from test_refiner_v6c_agent import make_fixture
        make_fixture(FIX)
    d = torch.load(FIX, map_location="cpu", weights_only=False)
    return {"tokens": d["tokens"][:1], "features": {k: v[:1] for k, v in d["features"].items()},
            "targets": {k: v[:1] for k, v in d["targets"].items()}}


def _step(agent, batch, seed):
    """one training micro-batch (forward + compute_loss, no optimizer step) under a fixed RNG -> its logs"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    f, t = _fresh(batch)
    t["scene_token"] = list(batch["tokens"])
    agent.compute_loss(f, t, agent.forward(f, t))
    return {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in agent.latest_logs.items()}


def _stub_trainer(cb):
    from pytorch_lightning.profilers import PassThroughProfiler
    return SimpleNamespace(callbacks=[cb], lightning_module=None, profiler=PassThroughProfiler())


def _sched_cb(agent):
    from navsim.agents.para_ssr.para_ssr_agent import DistillScheduleCallback
    cbs = [c for c in agent.get_training_callbacks() if isinstance(c, DistillScheduleCallback)]
    assert len(cbs) <= 1
    return cbs[0] if cbs else None


def _sched(ad):
    """everything the callback carries, as plain comparable values"""
    ctrl = lambda c: None if c is None else c.state_dict()  # noqa: E731
    return (int(ad.micro_step), ctrl(ad.ratio_ctrl), ctrl(ad.probe_ratio_ctrl),
            [(int(p.ema_step), {n: b.clone() for n, b in p.named_buffers(recurse=False) if n.startswith("ema__")})
             for p in ad.probes])


def _same_sched(x, y):
    if x[:3] != y[:3] or len(x[3]) != len(y[3]):
        return False
    return all(sa == sb and ea.keys() == eb.keys() and all(torch.equal(ea[k], eb[k]) for k in ea)
               for (sa, ea), (sb, eb) in zip(x[3], y[3]))


# ---------------------------------------------------------------------------------------------------------------- F25
def test_distill_schedule_round_trip(batch1):
    import pytorch_lightning as pl
    from pytorch_lightning.trainer import call
    a = build_agent(0, **V15)
    a.train()
    ad = a._distill.anchor_distill
    for i in range(3):
        _step(a, batch1, seed=10 + i)
        with torch.no_grad():                         # stands in for the optimizer: live probe weights move off the EMA
            for p in ad.probes.parameters():
                p.add_(0.01 * torch.randn_like(p))
    assert int(ad.micro_step) == a._loss.iteration == 3           # one tick per training micro-batch on a fresh run
    assert [int(p.ema_step) for p in ad.probes] == [3, 3]
    assert ad.ratio_ctrl["det"].n > 0 and ad.probe_ratio_ctrl["map"].n > 0   # measured on the gnorm/plan micro-batch
    ad.ratio_ctrl.update({"det": 0.5, "map": 0.25}, 1.0)
    ad.probe_ratio_ctrl.update({"det": 0.125, "map": 0.0625}, 1.0)
    want = _sched(ad)
    assert all(not torch.equal(p.ema__q_proj__weight, p.q_proj.weight) for p in ad.probes)
    # save through Lightning's callback-state glue and a torch.save round trip
    cb_a = _sched_cb(a)
    assert cb_a is not None and cb_a.state_key == "DistillScheduleCallback"
    buf = io.BytesIO()
    torch.save({"state_dict": {"agent." + k: v for k, v in a.state_dict().items()},
                "callbacks": call._call_callbacks_state_dict(_stub_trainer(cb_a)),
                "pytorch-lightning_version": pl.__version__}, buf)
    buf.seek(0)
    ck = torch.load(buf, map_location="cpu", weights_only=False)
    assert "DistillScheduleCallback" in ck["callbacks"]
    # restore in PL 2.2.1's order: model (loss.iteration via its extra state), then on_load_checkpoint, load_state_dict
    b = build_agent(0, **V15)
    b.train()
    bd = b._distill.anchor_distill
    assert not _same_sched(_sched(bd), want)                     # a fresh agent restarts the schedule
    b.load_state_dict({k[len("agent."):]: v for k, v in ck["state_dict"].items()}, strict=True)
    assert b._loss.iteration == 3 and int(bd.micro_step) == 0       # the model state alone does not carry it
    stub = _stub_trainer(_sched_cb(b))
    call._call_callbacks_on_load_checkpoint(stub, ck)
    call._call_callbacks_load_state_dict(stub, ck)
    assert _same_sched(_sched(bd), want)
    # the next micro-batch: same ramp / lam_* (and every other log) as the uninterrupted agent
    la, lb = _step(a, batch1, seed=99), _step(b, batch1, seed=99)
    assert float(la["distill/ramp"]) == 0.75 and float(la["distill/lam_det"]) > 0
    assert la.keys() == lb.keys()
    for k in la:
        assert torch.equal(torch.as_tensor(la[k]), torch.as_tensor(lb[k])), k
    assert _same_sched(_sched(ad), _sched(bd))


def test_distill_schedule_old_checkpoint_fallback(batch1):
    from pytorch_lightning.trainer import call
    c = build_agent(0, **V15)
    cd = c._distill.anchor_distill
    cb = _sched_cb(c)
    with torch.no_grad():
        for p in cd.probes.parameters():
            p.add_(0.01 * torch.randn_like(p))
    fresh = (cd.ratio_ctrl.state_dict(), cd.probe_ratio_ctrl.state_dict())
    c._loss.iteration = 1234                                         # as restored from the model's extra state
    old = {"callbacks": {"ModelCheckpoint{'monitor': None}": {"k": 1}}, "pytorch-lightning_version": "2.2.1"}
    stub = _stub_trainer(cb)
    call._call_callbacks_on_load_checkpoint(stub, old)
    call._call_callbacks_load_state_dict(stub, old)
    assert int(cd.micro_step) == 1234
    for p in cd.probes:
        assert int(p.ema_step) == 1234
        live = dict(p.named_parameters())
        for n, buf in p.named_buffers(recurse=False):
            if n.startswith("ema__"):
                assert torch.equal(buf, live[n[len("ema__"):].replace("__", ".")]), n
    assert (cd.ratio_ctrl.state_dict(), cd.probe_ratio_ctrl.state_dict()) == fresh    # controllers start fresh
    # a checkpoint that has the state: on_load_checkpoint leaves it to load_state_dict
    st = cb.state_dict()
    cd.micro_step.fill_(7)
    cb.on_load_checkpoint(None, None, {"callbacks": {cb.state_key: st}})
    assert int(cd.micro_step) == 7
    cb.load_state_dict(st)
    assert int(cd.micro_step) == 1234


def test_distill_schedule_callback_registration_and_state_dict_keys():
    a = build_agent(0, **V15)
    keys = list(a.state_dict().keys())
    assert _sched_cb(a) is not None
    assert list(a.state_dict().keys()) == keys                         # nothing registered on the model
    off = build_agent(0)                                               # cf_seg_v15 without distillation
    assert _sched_cb(off) is None
    assert not any(k.startswith("_distill.") for k in off.state_dict())
    from navsim.agents.para_ssr.para_ssr_agent import DistillScheduleCallback
    assert DistillScheduleCallback(off).state_dict() == {}              # Lightning would store nothing


class _ToyProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)
        self.register_buffer("ema_step", torch.zeros((), dtype=torch.long), persistent=False)
        for n, p in self.named_parameters():
            self.register_buffer("ema__" + n.replace(".", "__"), p.detach().clone(), persistent=False)


class _ToyAD(torch.nn.Module):
    """the schedule state of TrajectoryAnchorDistillation v15: micro_step, two probes, two TeacherRatioControllers"""

    def __init__(self):
        super().__init__()
        from navsim.agents.para_ssr.ck.bev_kd_arm import TeacherRatioControllers
        self.register_buffer("micro_step", torch.zeros((), dtype=torch.long), persistent=False)
        self.probes = torch.nn.ModuleList([_ToyProbe(), _ToyProbe()])
        self.ratio_ctrl = TeacherRatioControllers(("det", "map"), start_mb=1)
        self.probe_ratio_ctrl = TeacherRatioControllers(("det", "map"), start_mb=1)


class _ToyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.iteration = 0

    def get_extra_state(self):
        return {"iteration": int(self.iteration)}

    def set_extra_state(self, state):
        self.iteration = int(state["iteration"])


def _toy_module():
    import pytorch_lightning as pl

    class Agent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._distill = torch.nn.Module()
            self._distill.anchor_distill = _ToyAD()
            self._loss = _ToyLoss()
            self.latest_logs = {}

    class Module(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.agent = Agent()

        def training_step(self, batch, batch_idx):
            ad, (x,) = self.agent._distill.anchor_distill, batch
            ad.micro_step.add_(1)
            self.agent._loss.iteration += 1
            ad.ratio_ctrl.update({"det": 0.5, "map": 0.25}, 1.0 + batch_idx)
            loss = 0.0
            for p in ad.probes:
                with torch.no_grad():
                    for n, q in p.named_parameters():
                        getattr(p, "ema__" + n.replace(".", "__")).mul_(0.5).add_(q.detach(), alpha=0.5)
                    p.ema_step.add_(1)
                loss = loss + 100.0 * p.lin(x).pow(2).mean()
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    return Module()


def test_distill_schedule_survives_a_real_lightning_resume(tmp_path):
    """PL 2.2.1 end to end (CPU, toy module): ModelCheckpoint-style save, fit(ckpt_path) resume; the logging callback's
    optimizer-step hook runs inside a real Trainer (clip + accumulation)"""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    from navsim.agents.para_ssr.para_ssr_agent import DistillScheduleCallback, ParaSSRLoggingCallback

    data = DataLoader(TensorDataset(torch.randn(8, 3)), batch_size=2)          # 4 micro-batches per epoch

    def trainer(epochs, m):
        return pl.Trainer(accelerator="cpu", devices=1, max_epochs=epochs, logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                          gradient_clip_val=0.5, accumulate_grad_batches=2, default_root_dir=str(tmp_path),
                          callbacks=[ParaSSRLoggingCallback(), DistillScheduleCallback(m.agent)])

    torch.manual_seed(0)
    m1 = _toy_module()
    t1 = trainer(1, m1)
    t1.fit(m1, data)
    ad1 = m1.agent._distill.anchor_distill
    assert int(ad1.micro_step) == m1.agent._loss.iteration == 4
    assert {"train/grad_norm_preclip_step", "train/grad_clip_active_step", "lr/pg0"} <= set(t1.logged_metrics)
    ck = tmp_path / "e0.ckpt"
    t1.save_checkpoint(ck)
    saved = torch.load(ck, map_location="cpu", weights_only=False)
    assert "DistillScheduleCallback" in saved["callbacks"]
    want = _sched(ad1)
    for old in (False, True):
        if old:                                                            # a checkpoint from before the callback
            saved["callbacks"].pop("DistillScheduleCallback")
            ck = tmp_path / "e0_old.ckpt"
            torch.save(saved, ck)
        seen = {}

        class Peek(pl.Callback):
            def on_train_start(self, trainer, pl_module):
                ad = pl_module.agent._distill.anchor_distill
                seen["sched"] = _sched(ad)
                seen["live"] = [{n: q.detach().clone() for n, q in p.named_parameters()} for p in ad.probes]

        torch.manual_seed(1)
        m2 = _toy_module()
        t2 = trainer(2, m2)
        t2.callbacks.append(Peek())
        t2.fit(m2, data, ckpt_path=str(ck))
        got = seen["sched"]
        if not old:
            assert _same_sched(got, want)
        else:
            assert got[0] == 4 and [s for s, _ in got[3]] == [4, 4]       # micro_step / ema_step = loss.iteration
            assert got[1] == got[2] == m2.agent._distill.anchor_distill.probe_ratio_ctrl.state_dict()  # fresh (n 0)
            for (_, ema), live in zip(got[3], seen["live"]):
                assert all(torch.equal(ema["ema__" + n.replace(".", "__")], q) for n, q in live.items())
        ad2 = m2.agent._distill.anchor_distill
        assert int(ad2.micro_step) == m2.agent._loss.iteration == 8


# ---------------------------------------------------------------------------------------------------------------- F27
def test_grad_norm_and_lr_logging_is_read_only():
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRLoggingCallback
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.Linear(8, 4), torch.nn.Linear(4, 2))
    opt = torch.optim.AdamW([{"params": list(m[0].parameters()) + list(m[2].parameters()), "lr": 1e-3},
                             {"params": m[1].parameters(), "lr": 5e-4}], weight_decay=1e-4)
    for p in m.parameters():
        p.grad = torch.randn_like(p) * 20.0
    m[2].bias.grad = None                                              # a parameter without a gradient is skipped
    before = {n: (None if p.grad is None else p.grad.clone()) for n, p in m.named_parameters()}
    in_opt = [p for g in opt.param_groups for p in g["params"]]        # Lightning clips the optimizer's params
    ref_params = [torch.nn.Parameter(p.detach().clone()) for p in in_opt]
    for r, p in zip(ref_params, in_opt):
        r.grad = None if p.grad is None else p.grad.clone()
    ref = float(torch.nn.utils.clip_grad_norm_(ref_params, 35.0))
    for clip, active in ((35.0, 1.0), (1e9, 0.0), (0.0, 0.0), (None, 0.0)):
        logged = {}
        stub = SimpleNamespace(log=lambda name, value, **kw: logged.__setitem__(name, (value, kw)))
        ParaSSRLoggingCallback().on_before_optimizer_step(SimpleNamespace(gradient_clip_val=clip), stub, opt)
        assert logged["train/grad_norm_preclip"][0] == ref
        assert logged["train/grad_clip_active"][0] == active
        assert logged["lr/pg0"][0] == 1e-3 and logged["lr/pg1"][0] == 5e-4 and "lr/pg2" not in logged
        for name, (_, kw) in logged.items():
            assert kw["rank_zero_only"] and not kw["sync_dist"] and kw["batch_size"] == 1 and kw["on_step"], name
        assert logged["train/grad_norm_preclip"][1]["on_epoch"] and not logged["lr/pg0"][1]["on_epoch"]
        for n, p in m.named_parameters():
            assert (p.grad is None) == (before[n] is None) and (p.grad is None or torch.equal(p.grad, before[n])), n
    assert ref > 35.0


# ---------------------------------------------------------------------------------------------------------------- F28
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (CUDA_VISIBLE_DEVICES=0)")
def test_track_eager_under_inference_mode(monkeypatch):
    from navsim.agents.para_ssr.modules import torch_tracker as TT
    monkeypatch.setenv("KM_TRACK_GRAPH", "1")
    monkeypatch.setattr(TT, "_GRAPHS", {})                              # keys of other tests in this process
    monkeypatch.setattr(TT, "_CONSTS", {})                              # first _consts / q_lat fill inside inference_mode
    torch.manual_seed(0)
    anc = torch.tensor(np.load(REPO / "resources/km_corridor/trajectory_anchors_256.npy"), dtype=torch.float32)[:16]
    poses = (anc + torch.randn(16, 8, 3) * torch.tensor([0.3, 0.2, 0.03])).cuda()
    st = torch.cat([torch.rand(16, 1) * 15, torch.randn(16, 1) * 0.2, torch.randn(16, 2)], -1)
    init = torch.cat([st, st.new_zeros(16, 3)], -1).cuda()
    with torch.inference_mode():
        a = TT.track(poses, init, full=True)
        ref = TT._track(poses, init, full=True)
    assert len(TT._GRAPHS) == 0                                        # nothing captured inside inference_mode
    assert torch.equal(a, ref)
    # the same shape afterwards, as a training step would call it: no_grad (model forward) and with grad (slot 1)
    with torch.no_grad():
        b = TT.track(poses, init, full=True)
        rb = TT._track(poses, init, full=True)
    assert len(TT._GRAPHS) == 1                                        # the graph path outside it
    assert torch.allclose(b, rb, rtol=0.0, atol=1e-5)
    w = torch.randn(16, 41, 7, device="cuda")
    pa, pb = poses.clone().requires_grad_(True), poses.clone().requires_grad_(True)
    oa = TT.track(pa, init, full=True, slot=1)
    (oa * w).sum().backward()
    ob = TT._track(pb, init, full=True)
    (ob * w).sum().backward()
    assert torch.isfinite(pa.grad).all()
    assert torch.allclose(oa, ob, rtol=0.0, atol=1e-5) and torch.allclose(pa.grad, pb.grad, rtol=0.0, atol=1e-5)


# ---------------------------------------------------------------------------------------------------------------- F29
def test_v6c_needs_cand_official_rule():
    with pytest.raises(ValueError, match="corridor_select_rule"):
        build_agent(0, v6c=True, corridor_select_rule="v2")
    a = build_agent(0, v6c=False, corridor_select_rule="v2")
    assert a.para_ssr_model.v6c is None


# ---------------------------------------------------------------------------------------------------------- F30 / F31
@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    """DRYRUN configs of the current train.sh and of BASE_REV's (same env, same run root), composed in parallel"""
    tmp = tmp_path_factory.mktemp("dry")
    old = tmp / "base" / "train.sh"
    old.parent.mkdir()
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE_REV}:scripts/km_corridor/train.sh"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"{BASE_REV} not in this checkout")
    old.write_text(r.stdout)
    env_sh = REPO / "scripts/km_corridor/env.sh"                       # the env train.sh itself would source
    src = env_sh if env_sh.is_file() else REPO / "scripts/km_corridor/env.example.sh"
    (old.parent / "env.sh").write_text(f"source {src}\n")
    env = dict(os.environ, DRYRUN="1", WANDB="false", KM_RUN_ROOT=str(tmp / "runs"))
    env.pop("V6C_WARMUP", None)
    cur = REPO / "scripts/km_corridor/train.sh"
    jobs = {"v6c4": (cur, "cf_seg_v15_v6c", "0,1,2,3"), "v6c8": (cur, "cf_seg_v15_v6c", "0,1,2,3,4,5,6,7"),
            "v15": (cur, "cf_seg_v15", "0,1,2,3"), "v15_base": (old, "cf_seg_v15", "0,1,2,3"),
            "v6c4_base": (old, "cf_seg_v15_v6c", "0,1,2,3")}

    def run(job):
        script, var, gpus = job
        p = subprocess.run(["bash", str(script), var, "dry_r129", gpus], cwd=REPO, env=env, capture_output=True,
                           text=True, timeout=900)
        assert p.returncode == 0, p.stderr[-2000:]
        return p.stdout

    with ThreadPoolExecutor(len(jobs)) as ex:
        out = dict(zip(jobs, ex.map(run, jobs.values())))
    return out


def test_train_sh_syntax():
    assert subprocess.run(["bash", "-n", str(REPO / "scripts/km_corridor/train.sh")]).returncode == 0


def test_v6c_warmup_is_exactly_five_epochs(dry):
    assert "    v6c_warmup_steps: 26600\n" in dry["v6c4"]
    assert "    v6c_warmup_steps: 13300\n" in dry["v6c8"]
    ceil = lambda a, b: -(-a // b)  # noqa: E731
    for ng, steps in ((4, 26600), (8, 13300)):
        per_epoch = ceil(ceil(85109, ng), 4)        # DistributedSampler pads to ceil(N / NG); batch 4, no drop_last
        assert 5 * per_epoch == steps and per_epoch % (32 // ng) == 0     # ends on an accumulation boundary


def test_base_config_unchanged(dry):
    assert dry["v15"] == dry["v15_base"]                              # cf_seg_v15: byte-identical composed config
    a, b = dry["v6c4"].splitlines(), dry["v6c4_base"].splitlines()
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert diff == [("    v6c_warmup_steps: 26600", "    v6c_warmup_steps: 26596")]
