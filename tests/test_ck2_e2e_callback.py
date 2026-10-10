"""CK2 e2e T13: make_ck2_callback (CKE2E2Callback) without a Lightning trainer (stub trainer / agent).  CPU only:
  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_ck2_e2e_callback.py \
    --basetemp=/tmp/claude-0/-workspace/f9196902-3a88-4418-a505-cd713226ed4a/scratchpad/ck2e2e/pytest/callback

State round trip (lateral-KD EMA, BEV-KD controller, mb, counters; JSON-serialisable), recorder open / close per phase
(warmup none, warmup_record and on-policy until record_until, none after), LabelStore2 refresh at the on-policy epoch
start and every label_refresh_every_mb, the label-supply warning, steps_rank0.jsonl key filter, TRAIN_DONE, the
CK-only gradient clip (BEV-KD adapter and v2 untouched), the non-finite optimiser-step skip, and (2-process gloo DDP)
the lateral-KD EMA identical on both ranks with the rank-0 checkpoint state restoring every rank exactly (report 48 F4).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ck2e2e_testlib as TL  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data as ED  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402

pytestmark = pytest.mark.skipif(not TL.have_inputs(), reason="CK2 smoke teachers / fixture absent")


class _Trainer(SimpleNamespace):
    pass


def _agent(ck, bev_kd=False):
    st = O.build_student_ck2(ck.cfg)
    a = SimpleNamespace(_ck_e2e2=ck, ck_student=st, latest_logs={}, config=SimpleNamespace())
    if bev_kd:
        from navsim.agents.para_ssr.ck.bev_kd_arm import build_bev_kd_arm
        a.ck_bev_kd = build_bev_kd_arm(ck.cfg)
    return a


def _rec_add(ck, rows, seed=0):
    g = np.random.default_rng(seed)
    n = len(rows)
    return ck.recorder.add(np.asarray(rows), g.normal(size=(n, 96, 8, 3)).astype(np.float32), g.random((n, 96)) > 0.1,
                           g.integers(0, 256, (n, 16)), g.normal(size=(n, 16)).astype(np.float32),
                           g.random(n).astype(np.float32), ck.gstep)


def test_state_roundtrip(tmp_path):
    c = TL.cfg2(tmp_path, bev_kd={"enabled": True})
    ck = O.CKE2E2(c)
    cb = O.make_ck2_callback(_agent(ck))
    assert cb.state_key == "CKE2E2Callback"
    ck.ema.update(0.3, 0.1)
    ck.ema.update(0.2, 0.2)
    ck.bev_ctrl.update({"det": 2.0}, 1.0)
    ck.mb, ck.skipped_steps, ck.cum["tokens"], ck.cum["G_src_prev"] = 1234, 2, 99, 5
    sd = json.loads(json.dumps(cb.state_dict()))
    ck2 = O.CKE2E2(c)
    O.make_ck2_callback(SimpleNamespace(_ck_e2e2=ck2)).load_state_dict(sd)
    assert ck2.ema.state_dict() == ck.ema.state_dict() and ck2.bev_ctrl.state_dict() == ck.bev_ctrl.state_dict()
    assert sd["bev_ctrl"]["det"]["n"] == 1 and ck2.bev_ctrl.weights(10 ** 6) == ck.bev_ctrl.weights(10 ** 6)
    assert ck2.mb == 1234 and ck2.skipped_steps == 2 and ck2.cum["tokens"] == 99 and ck2.cum["G_src_prev"] == 5
    ck3 = O.CKE2E2(TL.cfg2(tmp_path))                      # arm off: the state still loads (bev_ctrl ignored)
    ck3.load_state_dict(sd)
    assert ck3.bev_ctrl is None and ck3.mb == 1234


def test_epoch_flow_recorder_refresh_warning_and_logs(tmp_path):
    c = TL.cfg2(tmp_path, record_from_epoch=1, onpolicy_from_epoch=2, rec_chunk_tokens=2, label_refresh_every_mb=2)
    ck = O.CKE2E2(c, max_epochs=5)
    assert ck.record_until == 3
    ck._labels = D2.LabelStore2(c.io_dir, n_rows=16, packed=c.warmup["packed"])
    refreshed = []
    real = ck._labels.refresh
    ck._labels.refresh = lambda e: (refreshed.append(int(e)), real(e))[1]
    agent = _agent(ck)
    cb = O.make_ck2_callback(agent)
    tr = _Trainer(current_epoch=0, global_rank=0, world_size=1, is_global_zero=True, global_step=0,
                  num_training_batches=4)

    def run_epoch(e, n_mb=2, rows=(3, 4)):
        tr.current_epoch = e
        cb.on_train_epoch_start(tr, None)
        for bi in range(n_mb):
            cb.on_train_batch_start(tr, None, None, bi)
            assert ck.epoch_frac == e + bi / 4
            if ck.recorder is not None:
                _rec_add(ck, list(rows), seed=e * 10 + bi)
            ck.mb += 1                                       # (what CKE2E2.loss does)
            agent.latest_logs = {"ck2/loss": torch.tensor(1.5), "bevkd/term": torch.tensor(0.1),
                                 "gnorm/plan": torch.tensor(2.0), "loss": torch.tensor(3.0),
                                 "plan/other": torch.tensor(9.0)}
            cb.on_train_batch_end(tr, None, None, None, bi)

    run_epoch(0)
    assert ck.recorder is None and ck.phase(0) == "warmup"
    cb.on_train_epoch_end(tr, None)
    run_epoch(1)                                             # warmup_record: recorder open
    att = ck.recorder.attempt
    cb.on_train_epoch_end(tr, None)
    assert ck.recorder is None
    rd = Path(ED.rec_rank_dir(c.io_dir, 1, 0))
    done = json.loads((rd / f"DONE_{att}.json").read_text())
    assert done["n_tokens"] == 4 and len(list(rd.glob("c_*.npz"))) == 2
    a = D2.read_rec2_chunk(sorted(rd.glob("c_*.npz"))[0])
    assert a["traj"].shape == (2, 96, 8, 3) and int(a["epoch"]) == 1
    # the labeler wrote generation 1 for 3 of the 4 recorded rows
    TL.make_gen(c.io_dir, 1, [3, 4, 5], 16)
    run_epoch(2, n_mb=3)                                     # on-policy: refresh at start (max_epoch 1) + every 2 mb
    assert refreshed[0] == 1 and len(refreshed) == 2 and ck.recorder is not None
    cb.on_exception(tr, None, RuntimeError("x"))             # interrupted: chunks flushed, no DONE
    rd2 = Path(ED.rec_rank_dir(c.io_dir, 2, 0))
    assert not list(rd2.glob("DONE_*")) and len(list(rd2.glob("c_*.npz"))) == 3 and ck.recorder is None
    run_epoch(4, n_mb=1)                                     # > record_until: on-policy without recording
    assert ck.recorder is None and ck.phase(4) == "onpolicy"
    cb.on_train_end(tr, None)
    assert (Path(c.io_dir) / "TRAIN_DONE").is_file()
    steps = [json.loads(x) for x in (Path(c.io_dir) / "steps_rank0.jsonl").read_text().splitlines()]
    assert len(steps) == 8
    assert all("ck2/loss" in s and "bevkd/term" in s and "gnorm/plan" in s and "loss" in s and "plan/other" not in s
               for s in steps)
    ev = [json.loads(x) for x in (Path(c.io_dir) / "epochs.jsonl").read_text().splitlines()]
    assert [e["event"] for e in ev] == ["epoch_start", "epoch_end", "epoch_start", "epoch_end", "epoch_start",
                                        "epoch_start"]
    assert ev[0]["label_warning"] is None and not ev[0]["recording"] and ev[2]["recording"]
    lw = ev[4]["label_warning"]                              # source epoch 1: 4 rows recorded, 3 labelled
    assert ev[4]["labels"]["max_epoch"] == 1 and lw["src_epoch"] == 1 and lw["n_rec"] == 4
    assert lw["n_labeled"] == 3 and not lw["warn"]


def test_ck_clip_and_nonfinite_skip(tmp_path):
    c = TL.cfg2(tmp_path, clip=1.0, bev_kd={"enabled": True})
    ck = O.CKE2E2(c)
    agent = _agent(ck, bev_kd=True)
    other = torch.nn.Linear(3, 3)
    mod = torch.nn.ModuleDict({"ck": agent.ck_student, "kd": agent.ck_bev_kd, "v2": other})
    cb = O.make_ck2_callback(agent)
    tr = _Trainer(is_global_zero=True, global_step=5)
    for p in mod.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, 10.0)
    cb.on_before_optimizer_step(tr, mod, None)
    gn = torch.norm(torch.stack([p.grad.norm() for p in O.ck2_parameters(agent)]))
    assert abs(float(gn) - 1.0) < 1e-4 and ck.last_clip_norm > 1.0
    assert torch.all(other.weight.grad == 10.0)                                  # v2 untouched (Lightning clips)
    assert all(torch.all(p.grad == 10.0) for p in agent.ck_bev_kd.parameters())  # arm not in the CK clip (NU34)
    assert all(p.grad is None for n, p in agent.ck_student.named_parameters() if not p.requires_grad)
    other.weight.grad[0, 0] = float("nan")
    cb.on_before_optimizer_step(tr, mod, None)
    assert ck.skipped_steps == 1 and all(p.grad is None for p in mod.parameters())


# ----------------------------------------------------------------------------------------------- DDP EMA state (F4)
def _ema_ddp_worker(rank, world, port, tmp, out_q):
    """real CKE2E2.loss on DIFFERENT fixture tokens per rank (warm-up, lat_kd.start_mb 0, BEV-KD arm [det] on with
    'gnorm/plan' on mbs 0 and 2, i.e. the EMA all-reduce interleaved with the BEV-KD collective and the DDP gradient
    all-reduce); after mb 2 the rank-0 callback state (what Lightning writes) is loaded on every rank and mb 3 is
    repeated (no optimiser step: same parameters)."""
    import os
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from navsim.agents.para_ssr.ck import bev_kd_arm as A
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        f, t = TL.fresh(2)
        f = {k: v[rank:rank + 1] for k, v in f.items()}
        t = {k: v[rank:rank + 1] for k, v in t.items()}
        c = TL.cfg2(tmp, lat_kd={"start_mb": 0}, bev_kd={"enabled": True, "start_mb": 0, "teachers": ["det"]},
                    io_dir=f"{tmp}/io")

        def new_ck():
            ck = O.CKE2E2(c)
            ck.epoch, ck.epoch_frac, ck.rank, ck.world_size = 0, 0.5, rank, world
            return ck

        ck = new_ck()

        class Wrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                self.bev = torch.nn.Parameter(0.1 * torch.randn(1, 5000, 256))
                self.ck_student = O.build_student_ck2(c)
                self.ck_bev_kd = A.build_bev_kd_arm(c)
                with torch.no_grad():
                    self.ck_bev_kd.kd.adapters["det"].proj.weight.add_(0.05 * torch.randn(256, 256))

            def forward(self, helper, v2_logs):
                pred = TL.fake_predictions(1, seed=rank, bev=self.bev * 1.0)
                v2 = (pred["bev_embed"] ** 2).mean()
                loss, logs = helper.loss(self.ck_student, f, t, pred, v2_loss=v2, v2_logs=v2_logs,
                                         bev_kd=self.ck_bev_kd)
                return v2 + loss, logs

        model = DDP(Wrap(), find_unused_parameters=False)
        plan = {0: {"gnorm/plan": torch.tensor(0.7)}, 2: {"gnorm/plan": torch.tensor(0.6)}}
        res = {"w": [], "ema": [], "lat_kd": [], "sur": []}
        snap = None
        for mb in range(4):
            if mb == 3:
                cb = O.make_ck2_callback(SimpleNamespace(_ck_e2e2=ck))
                box = [json.loads(json.dumps(cb.state_dict())) if rank == 0 else None]
                dist.broadcast_object_list(box, src=0)
                snap = box[0]
            model.zero_grad(set_to_none=True)
            loss, logs = model(ck, plan.get(mb))
            loss.backward()
            res["w"].append(float(logs["ck2/w_lat"]))
            res["ema"].append(dict(ck.ema.state_dict()))
            res["lat_kd"].append(float(logs["ck2/lat_kd"]))
            res["sur"].append(float(logs["ck2/sur"]))
        ck_r = new_ck()
        O.make_ck2_callback(SimpleNamespace(_ck_e2e2=ck_r)).load_state_dict(snap)
        model.zero_grad(set_to_none=True)
        loss, logs = model(ck_r, plan.get(3))
        loss.backward()
        res.update(w_res=float(logs["ck2/w_lat"]), ema_res=dict(ck_r.ema.state_dict()), mb_res=ck_r.mb,
                   ctrl=ck.bev_ctrl.state_dict(), ctrl_res=ck_r.bev_ctrl.state_dict())
        out_q.put((rank, res))
    except Exception as e:  # pragma: no cover
        import traceback
        out_q.put((rank, {"error": traceback.format_exc() + repr(e)}))
    finally:
        dist.destroy_process_group()


def test_ddp_gloo_lat_kd_ema_identical_and_resume_exact(tmp_path):
    """report 48 F4: 2-process gloo DDP with different inputs per rank -> the lateral-KD EMA is fed all-rank means, so
    its state and w_lat are identical on both ranks at every micro-batch although the rank-local losses differ, and
    loading the rank-0 checkpoint state on every rank continues exactly like the uninterrupted run (both ranks)."""
    import socket

    import torch.multiprocessing as mp
    TL.batch()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_ema_ddp_worker, args=(r, 2, port, str(tmp_path), q)) for r in range(2)]
    for p in ps:
        p.start()
    res = dict(q.get(timeout=900) for _ in ps)
    for p in ps:
        p.join(timeout=60)
    for r in (0, 1):
        assert "error" not in res[r], res[r].get("error")
    a, b = res[0], res[1]
    assert a["lat_kd"] != b["lat_kd"] and a["sur"] != b["sur"]            # rank-local inputs really differ
    assert a["ema"] == b["ema"] and a["w"] == b["w"]                      # ... the EMA / weight do not
    assert all(math.isfinite(w) for w in a["w"]) and a["w"][3] > 0 and a["ema"][3]["n"] == 4
    for x in (a, b):                                                      # rank-0 state restores EVERY rank exactly
        assert x["mb_res"] == 4 and x["w_res"] == x["w"][3] and x["ema_res"] == x["ema"][3]
        assert x["ctrl_res"] == x["ctrl"]
    assert a["ctrl"] == b["ctrl"]
    # the EMA saw the mean of the two ranks' values (bias-corrected hat after mb 0 = the mean itself)
    m0 = O.CKE2E2(TL.cfg2(tmp_path / "ref", lat_kd={"start_mb": 0})).cfg.lambda_sur
    want_sur = 0.5 * (m0 * a["sur"][0] + m0 * b["sur"][0])
    assert a["ema"][0]["sur"] == pytest.approx((1 - 0.99) * want_sur, rel=1e-9)
