#!/usr/bin/env python
"""Stage E: E1 / E2(kd_balance fixed) parity with the options added after the E2 pilot left at their defaults.

  python tools/refiner/stageE_parity_e12.py check [/home/external-user/ssd/yongjae_refiner/stageE/parity_golden_e12.pt]

The golden (write-once, taken on 2026-09-29 with the code BEFORE kd_balance / kd_draft_source / grad_share_every
existed) holds, for E1 and for E2 (both run-4 teacher snapshots, decoded, lambda 19.14, no ramp), two seeded CPU
training steps on the parity batch (stageE_parity.PARITY_TOKENS, per-arm targets): total loss, every log value that
existed then (time/* excluded) and every parameter gradient.  `check` requires torch.equal on all of them; new log
keys are listed, not compared.  The student lon / lat heads are set non-zero so that the refiner losses reach bev_embed.
"""
import sys, os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE))
GOLDEN = "/home/external-user/ssd/yongjae_refiner/stageE/parity_golden_e12.pt"
import torch
import stageE_parity as P
T = "/home/external-user/ssd/yongjae_refiner/stageE/teachers"
ARMS = {
    "E1": dict(refiner_mode="E1"),
    "E2": dict(refiner_mode="E2", kd_teacher_runs=(f"{T}/stageT4_T_fold0_seed0", f"{T}/stageT4_M_fold0_seed0"),
               kd_lambda=19.14, kd_ramp=(0.0, 0.0), kd_space="decoded"),
}

def run(arm, batch=None):
    torch.set_num_threads(8)
    agent = P.build_agent(**ARMS[arm])
    if batch is None:
        scenes = P.load_scenes(P.PARITY_TOKENS, agent.get_sensor_config())
        batch = P.make_batch(agent, scenes)
    f, t = batch
    # make the student heads non-trivial so the surrogate / KD reach bev (zero-init heads give zero BEV grads)
    with torch.no_grad():
        for h in (agent.ref_student.lon_head, agent.ref_student.lat_head):
            h[-1].weight.normal_(0, 0.05, generator=torch.Generator().manual_seed(1))
            h[-1].bias.fill_(-0.3)
    out = []
    for it in range(2):   # two steps: iteration 0 (grad logs) and 1
        loss, logs, grads, _ = P.train_step(agent, f, t)
        out.append(dict(loss=loss, logs=logs, grads=grads))
    return out, batch

def check(path=GOLDEN):
    g = torch.load(path, weights_only=False)
    allbad = []
    for arm in ARMS:
        out, _ = run(arm, g["batch"][arm])
        for i, (a, b) in enumerate(zip(g["res"][arm], out)):
            bad = []
            if not torch.equal(a["loss"], b["loss"]): bad.append("loss")
            for k in a["logs"]:
                if k.startswith("time/"): continue
                if k not in b["logs"] or not torch.equal(a["logs"][k].float(), b["logs"][k].float()): bad.append("log:" + k)
            if sorted(a["grads"]) != sorted(b["grads"]): bad.append("gradkeys")
            for k in a["grads"]:
                if not torch.equal(a["grads"][k], b["grads"].get(k, torch.empty(0))): bad.append("grad:" + k)
            print(arm, "step", i, "loss", float(a["loss"]), float(b["loss"]), "IDENTICAL" if not bad else bad[:10],
                  "new log keys:", sorted(set(b["logs"]) - set(a["logs"])))
            allbad += [f"{arm}/{i}/{x}" for x in bad]
    return allbad


if __name__ == "__main__":
    cmd, path = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else GOLDEN)
    if cmd == "dump" and Path(path).exists():
        raise SystemExit(f"{path} exists (write-once golden)")
    if cmd == "dump":
        res, batches = {}, {}
        for arm in ARMS:
            res[arm], batches[arm] = run(arm)
            print(arm, [float(s["loss"]) for s in res[arm]], {k: float(v) for k, v in res[arm][0]["logs"].items() if k.startswith(("ref/L_sur", "kd/", "gnorm/ref", "loss_e0"))})
        torch.save(dict(res=res, batch=batches), path)
    else:
        sys.exit(1 if check(path) else 0)
