#!/usr/bin/env python
"""CK2 e2e CPU dry run: the real v2 ParaSSRAgent + CK2 module through pytorch_lightning on CPU (no GPU, no DDP), a few
micro-batches per phase, with the real background-labeler round trip and a resume.  NOT a training run: 2 fixture tokens
(packed navtrain_train rows 0 / 40000), batch 1, accumulate 2, compressed schedule.

  cd /workspace/yongjae/SSR-ck2 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python tools/ck/e2e2/cpu_dryrun.py [--arm main|bevkd] [--tag T] [--eval]

Steps (output D/ck2/e2e_smoke_cpu/dryrun_<tag>_<arm>/):
  A  Trainer.fit epochs 0-1: epoch 0 'warmup' (64 official-label candidates / token, online teacher KD), epoch 1
     'warmup_record' (+ the 96-column record of the student's top-16 x 6) -> rec/ep001 + DONE
  L1 labeler2 --scorer official --workers 2 --once on rec/ep001 -> lab/ep001 (gen 1, 96 columns, official labels)
  B  resume from last.ckpt, epoch 2 'onpolicy' (current top-16 + 32 variants KD, BCE on gen 1, recorded again)
  L2 labeler2 again -> lab/ep002
  checks: phases 0 / 1 / 2 in steps_rank0.jsonl, finite ck2/* logs, no skipped step, mb / EMA continuity across the
     resume, G_src_prev = 1 in epoch 2 (labels from gen 1), every labelled row of gen 1 / 2 = the recorded trajectories
     bitwise, lateral-KD weight > 0 after lat_kd.start_mb, (bevkd, teachers det + map as bevkd_arm.set) every
     teacher's lam > 0 after bev_kd.start_mb, every adapter moved, per-teacher measurement, per-teacher controller
     state in both checkpoints and continued across the resume (n + n_skip of every teacher = mb).
  --eval: writes train/code/hydra/config.yaml and runs eval_e2e2.py stages extract,dump,label,infer_check,eval,summary
     on CPU for navtrain_val --limit 2 then navtest --limit 2 (--fix-from the val choice), label workers 2.
Schedule overrides (dry run only): record_from 1, onpolicy_from 2, record_until 2, label_refresh_every_mb 2,
  grad_share_every 1, lat_kd.start_mb 2, score_prior_rows 200, teacher = CK2 smoke teachers (smoke_ck2{T,M}_ddp2),
  teacher_amp off (CPU); bevkd arm: bev_kd.teachers [det, map], bev_kd.start_mb 2; v2 grad_norm_log_interval 1
  (BEV-KD measurement every mb).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

PY = "/venv/ssr/bin/python"
D = Path("/home/external-user/ssd/yongjae_refiner/ck")
FIX2 = D / "ck2/e2e_smoke_cpu/fixtures/real_b2_ck2.pt"
SMOKE_T = D / "ck2/smoke/smoke_ck2T_ddp2"
SMOKE_M = D / "ck2/smoke/smoke_ck2M_ddp2"
ANCHORS = "/home/external-user/kyungmin/SSR-v2/data/planning_vb/trajectory_anchors_256.npy"
SCORES = "/home/external-user/kyungmin/SSR-v2/data/planning_vb/pdm_score_256"
V2_OVERRIDES = dict(  # = tools/ck/e2e/launch_util.recipe (r34 source) agent fields; pretrained backbone off (CPU)
    max_epochs=30, use_task_interaction=True, use_det_motion_head=True, use_map_head=True,
    grad_balance_target={"plan": 0.4, "det": 0.3, "map": 0.3}, plan_anchor=True, plan_anchor_file=ANCHORS,
    plan_score_file=SCORES, image_architecture="resnet34.tv_in1k", plan_heading_from_xy=True,
    backbone_pretrained=False, grad_balance_warmup_iters=10600, grad_balance_interval=200, grad_norm_log_interval=1,
    log_sync_dist=False,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T', time.gmtime(time.time() + 9 * 3600))} KST] [cpu_dryrun] {msg}", flush=True)


def ck2_dict(io_dir: Path, arm: str) -> dict:
    d = {"enabled": True, "io_dir": str(io_dir), "teacher_det_run": str(SMOKE_T), "teacher_map_run": str(SMOKE_M),
         "teacher_amp": False, "score_prior_rows": 200, "record_from_epoch": 1, "onpolicy_from_epoch": 2,
         "record_until_epoch": 2, "label_refresh_every_mb": 2, "grad_share_every": 1, "lat_kd": {"start_mb": 2},
         "strict_rows": 0.0,
         "teacher_ep_check": False,       # the smoke teachers predate --ep-target ('official'); kept off so either student ep_target runs
         "kd_calib": {"enabled": False}}  # no KD calibration files for the smoke teachers
    if arm == "bevkd":
        d["bev_kd"] = {"enabled": True, "teachers": ["det", "map"], "start_mb": 2}
    return d


def agent_yaml(ck2: dict):
    from omegaconf import OmegaConf
    y = OmegaConf.load(Path(REPO) / "navsim/planning/script/config/common/agent/para_ssr_agent.yaml")
    OmegaConf.set_struct(y, False)
    for k, v in V2_OVERRIDES.items():
        y.config[k] = v
    y.config["ck_e2e2"] = ck2
    y.lr = 1e-4
    return y


def build_agent(ck2: dict):
    from hydra.utils import instantiate
    torch.manual_seed(0)
    return instantiate(agent_yaml(ck2))


class FixtureDS(torch.utils.data.Dataset):
    def __init__(self):
        d = torch.load(FIX2, map_location="cpu", weights_only=False)
        self.f, self.t = d["features"], d["targets"]
        self.n = int(d["targets"]["ck_row"].shape[0])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.f.items()}, {k: v[i] for k, v in self.t.items()}


def fit(agent, root: Path, max_epochs: int, ckpt=None):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from navsim.planning.training.agent_lightning_module import AgentLightningModule
    pl.seed_everything(0, workers=True)
    dl = torch.utils.data.DataLoader(FixtureDS(), batch_size=1, shuffle=False, num_workers=0)
    cbs = list(agent.get_training_callbacks()) + [ModelCheckpoint(every_n_epochs=1, save_top_k=-1, save_last=True,
                                                                  save_on_train_epoch_end=True)]
    tr = pl.Trainer(accelerator="cpu", devices=1, max_epochs=max_epochs, accumulate_grad_batches=2,
                    gradient_clip_val=35.0, gradient_clip_algorithm="norm", callbacks=cbs, logger=False,
                    enable_progress_bar=False, num_sanity_val_steps=0, limit_val_batches=0,
                    default_root_dir=str(root / "train"), enable_model_summary=False)
    t0 = time.time()
    tr.fit(AgentLightningModule(agent), train_dataloaders=dl, ckpt_path=None if ckpt is None else str(ckpt))
    log(f"fit to epoch {max_epochs} in {time.time() - t0:.0f}s (global_step {tr.global_step})")
    return tr


def last_ckpt(root: Path) -> Path:
    c = sorted((root / "train").rglob("last.ckpt"), key=lambda p: p.stat().st_mtime)
    assert c, f"no last.ckpt under {root / 'train'}"
    return c[-1]


def labeler(io: Path, max_epoch: int) -> int:
    cmd = [PY, f"{REPO}/tools/ck/e2e2/labeler2.py", "--io-dir", str(io), "--world-size", "1", "--workers", "2",
           "--poll", "1", "--status-every", "2", "--max-epoch", str(max_epoch), "--once"]
    env = dict(os.environ, PYTHONPATH=REPO, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    with open(io.parent / f"labeler_ep{max_epoch}.log", "a") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"labeler2 --once max-epoch {max_epoch}: rc {rc} in {time.time() - t0:.0f}s")
    return rc


def read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def gen_roundtrip(io: Path, epoch: int) -> dict:
    from navsim.agents.para_ssr.ck import e2e_data as ED
    from navsim.agents.para_ssr.ck import e2e_data2 as D2
    meta = json.loads((ED.gen_dir(str(io), epoch) / "meta.json").read_text())
    g = D2.open_generation2(str(io), epoch, int(meta["n_rows"]), mode="r")
    rs = np.asarray(g["row_state"])
    done = np.flatnonzero(rs == 1)
    rec = {}
    for c in ED.list_rec_chunks(str(io), epoch):
        d = D2.read_rec2_chunk(c["path"])
        for i, r in enumerate(d["row"]):
            rec.setdefault(int(r), (d["traj"][i], d["valid"][i]))
    bad = sum(1 for r in done if not (np.array_equal(np.nan_to_num(np.asarray(g["traj"][r]), nan=-9.0),
                                                     np.nan_to_num(rec[int(r)][0], nan=-9.0))
                                      and np.array_equal(np.asarray(g["valid"][r]), rec[int(r)][1])))
    lab = np.asarray(g["labels"])[done]
    return {"format": meta.get("format"), "rows_labelled": int(len(done)), "rows_recorded": len(rec),
            "mismatch": int(bad), "cand_ok_frac": float(np.asarray(g["cand_ok"])[done].mean()) if len(done) else None,
            "label_finite_frac": float(np.isfinite(lab).mean()) if len(done) else None,
            "pdms_mean": float(np.nanmean(lab[..., 6])) if len(done) else None}


def ckpt_state(p: Path) -> dict:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return dict((ck.get("callbacks") or {}).get("CKE2E2Callback") or {})


def write_hydra_cfg(root: Path, ck2: dict) -> Path:
    from omegaconf import OmegaConf
    p = root / "train/code/hydra/config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"agent": OmegaConf.to_container(agent_yaml(ck2), resolve=False)}), p)
    return p


def run_eval(root: Path) -> dict:
    env = dict(os.environ, PYTHONPATH=REPO, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    out = {}
    for split, extra in (("navtrain_val", []), ("navtest", [])):
        stages = "extract,dump,label,infer_check,eval,summary" if split == "navtrain_val" else \
            "dump,label,infer_check,eval,summary"
        cmd = [PY, f"{REPO}/tools/ck/e2e2/eval_e2e2.py", "--run-dir", str(root), "--split", split, "--limit", "2",
               "--gpus", "", "--batch-size", "1", "--workers", "0", "--workers-label", "2", "--bootstrap", "50",
               "--stage", stages] + extra
        # no lightning_logs/version_* here (logger off): pin the ckpt explicitly for BOTH splits (without --ckpt
        # eval_e2e2 re-resolves 'last' to check the pin, which needs lightning_logs)
        cmd += ["--ckpt", str(last_ckpt(root))]
        t0 = time.time()
        with open(root / f"eval_{split}.log", "a") as f:
            rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        m = root / "eval/root/eval" / root.name / split / "metrics.json"
        ic = root / "eval/root/infer" / root.name / split / "infer_check.json"
        out[split] = {"rc": rc, "sec": round(time.time() - t0, 1), "metrics": m.is_file(),
                      "infer_check": json.loads(ic.read_text()).get("pass") if ic.is_file() else None}
        if m.is_file():
            mm = json.loads(m.read_text())
            out[split].update(n_eval=mm["n_eval"], representative=mm.get("representative"),
                              best=mm.get("best") or mm.get("fixed_from_val"))
        log(f"eval_e2e2 {split}: {out[split]}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="main", choices=["main", "bevkd"])
    ap.add_argument("--tag", default=time.strftime("%m%d_%H%M", time.gmtime(time.time() + 9 * 3600)))
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-only", default="", help="existing dry-run root: run only the --eval part on it")
    a = ap.parse_args(argv)
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only: run with CUDA_VISIBLE_DEVICES="
    if a.eval_only:
        root = Path(a.eval_only)
        rp = root / "dryrun_result.json"
        res = json.loads(rp.read_text())
        write_hydra_cfg(root, res["ck_e2e2"])
        res["eval"] = run_eval(root)
        chk = res["checks"]
        chk["eval_val_ok"] = res["eval"]["navtrain_val"]["rc"] == 0 and bool(res["eval"]["navtrain_val"]["infer_check"])
        chk["eval_navtest_ok"] = res["eval"]["navtest"]["rc"] == 0
        res["pass"] = all(bool(v) for v in chk.values())
        rp.write_text(json.dumps(res, indent=1, default=str))
        log(f"eval-only checks {json.dumps(chk)} -> pass={res['pass']}")
        return 0 if res["pass"] else 4
    root = D / "ck2/e2e_smoke_cpu" / f"dryrun_{a.tag}_{a.arm}"
    assert not root.exists(), f"{root} exists (new --tag)"
    io = root / "ck_e2e2"
    root.mkdir(parents=True)
    ck2 = ck2_dict(io, a.arm)
    res = {"root": str(root), "arm": a.arm, "started": time.strftime("%F %T KST", time.gmtime(time.time() + 9 * 3600)),
           "ck_e2e2": ck2}
    # ---- A: epochs 0 (warmup) + 1 (warmup_record)
    agent = build_agent(ck2)
    adapter0 = None if getattr(agent, "ck_bev_kd", None) is None else \
        {t: agent.ck_bev_kd.kd.adapters[t].proj.weight.detach().clone() for t in agent.ck_bev_kd.teachers}
    fit(agent, root, 2)
    ck_a = last_ckpt(root)
    st_a = ckpt_state(ck_a)
    res["A"] = {"ckpt": str(ck_a), "state": {k: st_a.get(k) for k in ("mb", "skipped_steps")},
                "ema_n": (st_a.get("ema") or {}).get("n"), "bev_ctrl": st_a.get("bev_ctrl")}
    res["L1_rc"] = labeler(io, 1)
    res["gen1"] = gen_roundtrip(io, 1)
    # ---- B: resume -> epoch 2 (onpolicy, gen 1 labels)
    del agent
    agent = build_agent(ck2)
    fit(agent, root, 3, ckpt=ck_a)
    ck_b = last_ckpt(root)
    st_b = ckpt_state(ck_b)
    res["B"] = {"ckpt": str(ck_b), "state": {k: st_b.get(k) for k in ("mb", "skipped_steps")},
                "ema_n": (st_b.get("ema") or {}).get("n"), "bev_ctrl": st_b.get("bev_ctrl")}
    res["L2_rc"] = labeler(io, 2)
    res["gen2"] = gen_roundtrip(io, 2)
    # ---- checks
    recs = read_jsonl(io / "steps_rank0.jsonl")
    by_phase = {}
    for r in recs:
        by_phase.setdefault(int(r.get("ck2/phase", -1)), []).append(r)
    nonfin = [(r["mb"], k) for r in recs for k, v in r.items() if k.startswith(("ck2/", "bevkd/", "gnorm/"))
              and isinstance(v, float) and not math.isfinite(v) and not k.endswith("ratio_now")]
    onp = by_phase.get(2, [])
    chk = {
        "phases_0_1_2": sorted(by_phase) == [0, 1, 2] and all(len(v) == 2 for v in by_phase.values()),
        "nonfinite_logs_0": not nonfin,
        "skipped_steps_0": max(int(r.get("skipped_steps") or 0) for r in recs) == 0 if recs else False,
        "record_ep1_ep2": all((io / f"rec/ep{e:03d}").is_dir() for e in (1, 2)),
        "gen1_bitexact": res["gen1"]["rows_labelled"] == 2 and res["gen1"]["mismatch"] == 0,
        "gen2_bitexact": res["gen2"]["rows_labelled"] == 2 and res["gen2"]["mismatch"] == 0,
        "gen_format_ck2_96": res["gen1"]["format"] == "ck2_96",
        "onpolicy_G_from_gen1": bool(onp) and all(float(r.get("ck2/G_src_prev", 0)) == 1.0 for r in onp),
        "onpolicy_96_cands": bool(onp) and all(float(r.get("ck2/n_cand", 0)) == 96.0 for r in onp),
        "resume_mb_continuous": st_a.get("mb") == 4 and st_b.get("mb") == 6,
        "lat_kd_weight_on_after_start": any(float(r.get("ck2/w_lat", 0)) > 0 for r in recs if r["mb"] >= 3),
        "gshare_logged": all("ck2/gshare" in r for r in recs),
        "labelers_rc_0": res["L1_rc"] == 0 and res["L2_rc"] == 0,
    }
    if a.arm == "bevkd":
        teachers = list(ck2["bev_kd"]["teachers"])
        chk["bevkd_lambda_on_after_start"] = all(any(float(r.get(f"bevkd/{t}/lam", 0)) > 0 for r in recs
                                                     if r["mb"] >= 3) for t in teachers)
        sd = torch.load(ck_b, map_location="cpu", weights_only=False)["state_dict"]
        chk["adapter_moved"] = bool(adapter0 is not None and all(
            float((sd[f"agent.ck_bev_kd.kd.adapters.{t}.proj.weight"] - adapter0[t]).abs().max()) > 0
            for t in teachers))
        chk["bevkd_measured"] = all(any(f"bevkd/{t}/g_kd_unit" in r for r in recs) for t in teachers)
        ca, cb = res["A"].get("bev_ctrl") or {}, res["B"].get("bev_ctrl") or {}
        # one measurement per mb (grad_norm_log_interval 1): every teacher's n + n_skip = mb in both checkpoints, so
        # B continued A's controller state through the resume (a fresh controller would count only B's mbs)
        chk["bevkd_ctrl_saved_and_resumed"] = all(
            t in ca and t in cb and ca[t]["n"] + ca[t]["n_skip"] == st_a.get("mb")
            and cb[t]["n"] + cb[t]["n_skip"] == st_b.get("mb") for t in teachers)
    res["checks"] = chk
    res["steps"] = [{k: r.get(k) for k in ("epoch", "mb", "ck2/phase", "ck2/loss", "ck2/bce", "ck2/kd_score",
                                           "ck2/sur", "ck2/lat_kd", "ck2/w_lat", "ck2/G_src_prev",
                                           "ck2/G_src_fallback", "ck2/n_cand", "ck2/rec_tokens", "ck2/gshare",
                                           "bevkd/det/lam", "bevkd/map/lam", "bevkd/det/g_kd_unit",
                                           "bevkd/map/g_kd_unit", "bevkd/term", "loss_v2", "loss",
                                           "sec_step", "ck2/total_ms", "ck2/teacher_ms")} for r in recs]
    if a.eval:
        write_hydra_cfg(root, ck2)
        res["eval"] = run_eval(root)
        chk["eval_val_ok"] = res["eval"]["navtrain_val"]["rc"] == 0 and bool(res["eval"]["navtrain_val"]["infer_check"])
        chk["eval_navtest_ok"] = res["eval"]["navtest"]["rc"] == 0
    res["pass"] = all(bool(v) for v in chk.values())
    res["finished"] = time.strftime("%F %T KST", time.gmtime(time.time() + 9 * 3600))
    (root / "dryrun_result.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"checks {json.dumps(chk)} -> pass={res['pass']} ({root / 'dryrun_result.json'})")
    return 0 if res["pass"] else 4


if __name__ == "__main__":
    sys.exit(main())
