"""refiner v6c on CUDA (report 125 §7): CUDA-graph tracker identities (T11), a real-batch training step on the GPU with
the online teachers, and the Lightning order 'val forward under inference_mode, then a training step' (the CUDA-graph
inference-tensor hazard).  Skipped without a GPU or without the fixture of test_refiner_v6c_agent.py."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
TEACH = os.environ.get("V6C_TEACHER_SMOKE", "")


def _dev():
    return torch.device("cuda:0")


def test_graph_track_detached_equals_nograd_track():
    """S_v6_d = S_v6_g.detach() replaces a separate no-grad LQR (report 124 §4-2a, T11 graph part)"""
    from navsim.agents.para_ssr.modules.torch_tracker import track
    from navsim.agents.para_ssr.refiner_v6c import nets as N
    dev = _dev()
    tt = torch.arange(1, 9, dtype=torch.float32, device=dev) * 0.5
    tau = torch.stack([10 * tt, 0.05 * tt ** 2, 0.01 * tt], -1)[None].repeat(4, 1, 1)
    init = torch.cat([torch.full((4, 1), 10.0, device=dev), torch.zeros(4, 6, device=dev)], -1)
    ctrl = torch.zeros(4, 12, device=dev)
    ctrl[:, :6] = -0.4
    ctrl[:, 6:] = 0.3
    ctrl.requires_grad_()
    tau_v6 = N.decode_v6(tau, ctrl, init[:, 0])["traj"]
    S_g = track(tau_v6, init, full=True)
    with torch.no_grad():
        S_n = track(tau_v6.detach(), init, full=True)
    err = (S_g.detach() - S_n).abs().max().item()
    assert err <= 1e-5, err
    S_g.sum().backward()
    assert torch.isfinite(ctrl.grad).all() and ctrl.grad.abs().sum() > 0
    # Delta = 0 on the GPU: decoder identity and the same states as the B x 256 graph row
    z = torch.zeros(4, 12, device=dev)
    assert torch.equal(N.decode_v6(tau, z, init[:, 0])["traj"], tau)
    with torch.no_grad():
        big = track(tau.repeat(64, 1, 1), init.repeat(64, 1), full=True)[:4]
        one = track(tau, init, full=True)
    assert (big - one).abs().max().item() <= 1e-5


def _agent_and_batch(**over):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_refiner_v6c_agent import FIX, build_agent, _fresh, have_inputs
    if not have_inputs() or not FIX.is_file():
        pytest.skip("fixture missing (run test_refiner_v6c_agent.py once)")
    d = torch.load(FIX, map_location="cpu", weights_only=False)
    a = build_agent(0, v6c=True, **over).to(_dev())
    f, t = _fresh(d)
    mv = lambda x: {k: (v.to(_dev()) if torch.is_tensor(v) else v) for k, v in x.items()}  # noqa: E731
    return a, mv(f), mv(t)


@pytest.mark.parametrize("stage", ["D", "P_like"])
def test_gpu_training_step_val_then_train(stage):
    over = {"v6c_stage": "D"}
    if TEACH:
        over["v6c_teacher_ckpt"] = TEACH
    a, f, t = _agent_and_batch(**over)
    if stage == "P_like":                        # stage P needs a C0 checkpoint; freeze in place for the GPU path
        for n, p in a.para_ssr_model.named_parameters():
            if not n.startswith("v6c."):
                p.requires_grad_(False)
    # Lightning order: training first (the baseline's no-grad candidate track is captured there; a graph FIRST
    # captured under inference_mode cannot be replayed in training: a baseline property, never hit because val runs
    # after training), then a val step under inference_mode (v6c's own tracks run eagerly there), then training again
    a.train()
    p0 = a.forward({k: v.clone() for k, v in f.items()}, t)
    a.compute_loss(f, t, p0).backward()
    a.zero_grad(set_to_none=True)
    a.eval()
    with torch.inference_mode():
        pe = a.forward({k: v.clone() for k, v in f.items()})
        le = a.compute_loss(f, t, pe)
    assert torch.isfinite(le)
    a.train()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        p = a.forward({k: v.clone() for k, v in f.items()}, t)
        loss = a.compute_loss(f, t, p)
        loss.backward()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
        a.zero_grad(set_to_none=True)
    assert torch.isfinite(loss)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"\n[v6c gpu {stage}] B=2 step s {['%.2f' % x for x in ts]} peak {peak:.2f} GiB, "
          f"v6c/loss {float(a.latest_logs['v6c/loss']):.3f}, teacher {'on' if TEACH else 'off'}")
    with torch.inference_mode():
        a.eval()
        pe2 = a.forward({k: v.clone() for k, v in f.items()})
    assert torch.equal(pe2["trajectory"], pe2["v6c_out"].to(pe2["trajectory"].dtype))
