#!/usr/bin/env python3
"""Automated Training Monitor & NAVSIM Evaluator for tmux 19 & 21.

Monitors:
1. tmux 21 (para_ssr_plan_only, GPUs 4,5):
   - When finished (epoch 29 / process exits):
   - Triggers NAVSIM PDM evaluation on GPUs 4,5
2. tmux 19 (paradrive_distill_bev_selector_v6, GPUs 6,7):
   - When finished (epoch 29 / process exits):
   - Triggers NAVSIM PDM evaluation on GPUs 6,7

Saves aggregated results to /workspace/byounggun/SSR/work_dirs/AUTO_EVAL_REPORT.txt
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path("/workspace/byounggun/SSR").resolve()
WORK_DIRS = REPO / "work_dirs"
LOG_FILE = WORK_DIRS / "auto_monitor_eval.log"
REPORT_FILE = WORK_DIRS / "AUTO_EVAL_REPORT.txt"


def log(msg: str) -> None:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{now_str}] {msg}"
    print(formatted, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_matching_pids(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", pattern],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return [int(p.strip()) for p in out.strip().split() if p.strip()]
    except Exception:
        return []


def check_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Find epoch 29 checkpoint or last.ckpt with epoch >= 29."""
    if not ckpt_dir.is_dir():
        return None
    # 1. Direct match for epoch=29
    epoch29_files = sorted(ckpt_dir.glob("epoch=29-step=*.ckpt"))
    if epoch29_files:
        return epoch29_files[-1]

    # 2. Check last.ckpt
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.is_file():
        try:
            import torch
            ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            ep = ckpt.get("epoch", -1)
            if ep >= 29:
                return last_ckpt
        except Exception as e:
            log(f"Warning checking {last_ckpt}: {e}")
    return None


def get_latest_epoch(ckpt_dir: Path) -> Optional[int]:
    if not ckpt_dir.is_dir():
        return None
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.is_file():
        try:
            import torch
            ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            return ckpt.get("epoch")
        except Exception:
            pass
    ckpts = sorted(ckpt_dir.glob("epoch=*-step=*.ckpt"))
    if ckpts:
        fname = ckpts[-1].name
        # epoch=14-step=9975.ckpt
        try:
            part = fname.split("-")[0]
            return int(part.split("=")[1])
        except Exception:
            pass
    return None


def run_evaluation(
    job_name: str,
    ckpt_path: Path,
    gpus: str,
    arch: str,
    extra_hydra_args: list[str],
) -> dict[str, Any]:
    log(f"[{job_name}] Starting NAVSIM PDM Evaluation on GPUs {gpus}...")
    log(f"[{job_name}] Checkpoint: {ckpt_path}")

    snapshot_dir = WORK_DIRS / "eval_snapshots" / job_name
    exp_name = f"eval/{job_name}"
    eval_script = REPO / "scripts/evaluation/eval_para_ssr_distill_epoch29_gpus4567.sh"

    env = os.environ.copy()
    env["GPU_LIST"] = gpus
    env["CKPT_SRC"] = str(ckpt_path)
    env["SNAPSHOT_DIR"] = str(snapshot_dir)
    env["EXPERIMENT_NAME"] = exp_name
    env["SKIP_AUX"] = "1"
    env["IMAGE_ARCHITECTURE"] = arch
    env["NAVSIM_EXP_ROOT_OVERRIDE"] = str(WORK_DIRS)
    env["PYTHON"] = "/venv/ssr/bin/python"
    env["PATH"] = f"/venv/ssr/bin:{env.get('PATH', '')}"

    cmd = [
        "bash",
        str(eval_script),
        f"agent.config.image_architecture={arch}",
    ] + extra_hydra_args

    log(f"[{job_name}] Command: {' '.join(cmd)}")
    log(f"[{job_name}] Environment: GPU_LIST={gpus} SKIP_AUX=1")

    start_t = time.time()
    res = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True)
    elapsed = time.time() - start_t
    log(f"[{job_name}] Evaluation exited with code {res.returncode} in {elapsed:.1f}s")

    summary_file = WORK_DIRS / exp_name / "evaluation_summary.txt"
    merged_csv = WORK_DIRS / exp_name / "merged.csv"
    summary_text = ""
    if summary_file.is_file():
        summary_text = summary_file.read_text()
    elif res.stdout:
        summary_text = res.stdout[-2000:]

    if res.returncode != 0:
        log(f"[{job_name}] ERROR: Evaluation failed! Stderr tail:")
        log(res.stderr[-1000:])

    return {
        "job_name": job_name,
        "gpus": gpus,
        "ckpt_path": str(ckpt_path),
        "returncode": res.returncode,
        "elapsed_sec": elapsed,
        "summary_text": summary_text,
        "merged_csv": str(merged_csv) if merged_csv.is_file() else None,
        "summary_file": str(summary_file) if summary_file.is_file() else None,
    }


def update_report_file(eval_results: dict[str, dict[str, Any]]) -> None:
    lines = [
        "=" * 72,
        f" NAVSIM AUTOMATED EVALUATION FINAL REPORT",
        f" Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
    ]
    for job_name, res in eval_results.items():
        lines.append(f"--- [Job: {job_name}] ---")
        lines.append(f"GPUs: {res.get('gpus')}")
        lines.append(f"Checkpoint: {res.get('ckpt_path')}")
        lines.append(f"Status: {'SUCCESS' if res.get('returncode') == 0 else 'FAILED'}")
        lines.append(f"Elapsed: {res.get('elapsed_sec', 0):.1f}s")
        if res.get("merged_csv"):
            lines.append(f"Merged CSV: {res.get('merged_csv')}")
        lines.append("")
        if res.get("summary_text"):
            lines.append(res.get("summary_text"))
        lines.append("\n" + "=" * 72 + "\n")

    REPORT_FILE.write_text("\n".join(lines))
    log(f"Report updated at: {REPORT_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    args = parser.parse_args()

    jobs = {
        "tmux21_plan_only": {
            "name": "para_ssr_plan_only",
            "pattern": "run_training.py.*para_ssr_plan_only",
            "initial_pids": [840256, 840650],
            "ckpt_dir": WORK_DIRS / "para_ssr_plan_only/lightning_logs/version_0/checkpoints",
            "gpus": "4,5",
            "arch": "resnet50.tv_in1k",
            "extra_args": [
                "agent.config.use_task_interaction=false",
                "agent.config.use_det_motion_head=false",
                "agent.config.use_map_head=false",
                "agent.config.plan_anchor=false",
                "agent.config.plan_anchor_file=null",
                "agent.config.plan_score_file=null",
            ],
            "status": "training",
            "result": None,
        },
        "tmux19_bev_selector_v6": {
            "name": "paradrive_distill_bev_selector_v6",
            "pattern": "run_training.py.*paradrive_distill_bev_selector_v6",
            "initial_pids": [893377, 894151],
            "ckpt_dir": WORK_DIRS / "paradrive_distill_bev_selector_v6/lightning_logs/version_0/checkpoints",
            "gpus": "6,7",
            "arch": "resnet34.tv_in1k",
            "extra_args": [
                "agent.config.use_task_interaction=false",
                "agent.config.use_det_motion_head=false",
                "agent.config.use_map_head=false",
                "agent.config.plan_anchor=true",
                "agent.config.plan_anchor_file=/workspace/byounggun/SSR/data/planning_vb/trajectory_anchors_256.npy",
                "agent.config.plan_score_file=/workspace/byounggun/SSR/data/planning_vb/pdm_score_256",
            ],
            "status": "training",
            "result": None,
        },
    }

    log("=" * 72)
    log("Auto-monitor & Eval started.")
    log(f"Monitoring tmux 21 (para_ssr_plan_only) -> will eval on GPUs 4,5")
    log(f"Monitoring tmux 19 (paradrive_distill_bev_selector_v6) -> will eval on GPUs 6,7")
    log(f"Report destination: {REPORT_FILE}")
    log("=" * 72)

    eval_results: dict[str, dict[str, Any]] = {}
    loop_count = 0

    while True:
        all_done = True

        for key, job in jobs.items():
            if job["status"] == "done":
                continue

            all_done = False
            ckpt_dir = job["ckpt_dir"]
            final_ckpt = check_checkpoint(ckpt_dir)

            # Check running processes
            alive_pids = [pid for pid in job["initial_pids"] if is_pid_alive(pid)]
            pattern_pids = find_matching_pids(job["pattern"])
            is_alive = bool(alive_pids or pattern_pids)

            # Log heartbeat every 10 iterations (~10 mins)
            if loop_count % 10 == 0:
                cur_ep = get_latest_epoch(ckpt_dir)
                log(f"[Status {job['name']}] alive={is_alive} (pids={alive_pids or pattern_pids}), latest_checkpoint_epoch={cur_ep}, target=29")

            if not is_alive:
                # Give a grace period for filesystem sync
                time.sleep(10)
                final_ckpt = check_checkpoint(ckpt_dir)

                if final_ckpt:
                    log(f"[{job['name']}] Training completed! Found final checkpoint: {final_ckpt}")
                    job["status"] = "evaluating"
                    result = run_evaluation(
                        job_name=job["name"],
                        ckpt_path=final_ckpt,
                        gpus=job["gpus"],
                        arch=job["arch"],
                        extra_hydra_args=job["extra_args"],
                    )
                    job["result"] = result
                    job["status"] = "done"
                    eval_results[job["name"]] = result
                    update_report_file(eval_results)
                else:
                    exit_retries = job.get("exit_retries", 0) + 1
                    job["exit_retries"] = exit_retries
                    cur_ep = get_latest_epoch(ckpt_dir)
                    if exit_retries < 6:
                        log(f"[{job['name']}] Process exited, waiting for final checkpoint sync (attempt {exit_retries}/5)...")
                        time.sleep(15)
                    else:
                        log(f"[{job['name']}] ALERT: Process exited but final checkpoint (epoch 29) was not found after retries! Latest saved epoch: {cur_ep}")
                        job["status"] = "process_exited_no_final_ckpt"
            else:
                # Process is still running. Check if final ckpt appeared early (e.g. during last epoch flush)
                if final_ckpt and not is_alive:
                    log(f"[{job['name']}] Final checkpoint detected: {final_ckpt}")

        if all_done:
            log("All monitored jobs have completed evaluation!")
            break

        loop_count += 1
        time.sleep(args.interval)

    log("Auto-monitor & Eval finished cleanly.")


if __name__ == "__main__":
    main()
