#!/usr/bin/env python3
"""Automated Training Monitor & NAVSIM Evaluator for tmux 21 (Student) & tmux 19 (Distill v8).

Monitors:
1. tmux 21 (para_ssr_v2_r34_student, GPUs 4,5):
   - Pure student baseline (ResNet-34 + Aux heads + 256 Trajectory Anchors)
   - When finished (epoch 29 / process exits):
   - Automatically triggers NAVSIM PDM evaluation on GPUs 4,5
2. tmux 19 (paradrive_distill_bev_selector_v8, GPUs 6,7):
   - Distill v8 experiment (Hybrid Anchor Weighting & Multi-Scale Spatial Distance Bias)
   - When finished (epoch 29 / process exits):
   - Automatically triggers NAVSIM PDM evaluation on GPUs 6,7

Logs and Reports:
- Log: /workspace/byounggun/SSR/work_dirs/auto_monitor_eval.log
- Status / Progress: /workspace/byounggun/SSR/work_dirs/AUTO_EVAL_STATUS.json
- Final Summary Report: /workspace/byounggun/SSR/work_dirs/AUTO_EVAL_REPORT.txt
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path("/workspace/byounggun/SSR").resolve()
WORK_DIRS = REPO / "work_dirs"
LOG_FILE = WORK_DIRS / "auto_monitor_eval.log"
REPORT_FILE = WORK_DIRS / "AUTO_EVAL_REPORT.txt"
STATUS_FILE = WORK_DIRS / "AUTO_EVAL_STATUS.json"


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


def get_tmux_pane_tail(target: str, n_lines: int = 15) -> str:
    """Read recent lines from a tmux target window/pane (e.g. '21:0')."""
    try:
        out = subprocess.check_output(
            ["tmux", "capture-pane", "-pt", target, "-S", f"-{n_lines}"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return ""


def get_latest_ckpt_dir(exp_dir: Path) -> Path:
    version_dirs = sorted(
        exp_dir.glob("lightning_logs/version_*"),
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0,
    )
    if version_dirs:
        return version_dirs[-1] / "checkpoints"
    return exp_dir / "lightning_logs/version_0/checkpoints"


def get_latest_epoch_from_filenames(ckpt_dir: Path) -> Optional[int]:
    if not ckpt_dir.is_dir():
        return None
    epochs = []
    for f in ckpt_dir.glob("epoch=*-step=*.ckpt"):
        try:
            ep = int(f.name.split("-")[0].split("=")[1])
            epochs.append(ep)
        except Exception:
            pass
    return max(epochs) if epochs else None


def check_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Find epoch 29 checkpoint or last.ckpt."""
    if not ckpt_dir.is_dir():
        return None

    # 1. Match for epoch=29
    epoch29_files = list(ckpt_dir.glob("epoch=29-step=*.ckpt"))
    if epoch29_files:
        epoch29_files.sort(
            key=lambda p: int(p.name.split("-")[1].split("=")[1].split(".")[0])
            if "step=" in p.name
            else 0
        )
        return epoch29_files[-1]

    # 2. Check last.ckpt if present
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.is_file():
        try:
            import torch
            ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            ep = ckpt.get("epoch", -1)
            if ep >= 29:
                return last_ckpt
        except Exception as e:
            log(f"Warning reading {last_ckpt}: {e}")

    return None


def wait_file_stable(path: Path, wait_sec: int = 10, retries: int = 3) -> bool:
    """Verify that file exists and its size is stable across intervals."""
    if not path.is_file():
        return False
    prev_size = path.stat().st_size
    for _ in range(retries):
        time.sleep(wait_sec)
        if not path.is_file():
            return False
        cur_size = path.stat().st_size
        if cur_size == prev_size and cur_size > 0:
            return True
        prev_size = cur_size
    return True


def run_evaluation(
    job_name: str,
    ckpt_path: Path,
    gpus: str,
    arch: str,
    extra_hydra_args: list[str],
) -> dict[str, Any]:
    log("=" * 72)
    log(f"[{job_name}] Starting NAVSIM PDM Evaluation on GPUs {gpus}...")
    log(f"[{job_name}] Checkpoint: {ckpt_path}")
    log("=" * 72)

    snapshot_dir = WORK_DIRS / "eval_snapshots" / f"{job_name}_epoch29"
    exp_name = f"eval/{job_name}_epoch29"
    eval_script = REPO / "scripts/evaluation/eval_para_ssr_distill_epoch29_gpus4567.sh"

    env = os.environ.copy()
    env["GPU_LIST"] = gpus
    env["CKPT_SRC"] = str(ckpt_path)
    env["SNAPSHOT_DIR"] = str(snapshot_dir)
    env["EXPERIMENT_NAME"] = exp_name
    env["SKIP_AUX"] = "1"  # Focus on primary planning score (PDMS)
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

    eval_out_dir = WORK_DIRS / exp_name
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    eval_run_log = eval_out_dir / "eval_run.log"

    start_t = time.time()
    with open(eval_run_log, "w") as out_f:
        res = subprocess.run(
            cmd,
            env=env,
            cwd=str(REPO),
            stdout=out_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - start_t
    log(f"[{job_name}] Evaluation exited with code {res.returncode} in {elapsed:.1f}s")

    summary_file = eval_out_dir / "evaluation_summary.txt"
    merged_csv = eval_out_dir / "merged.csv"
    summary_text = ""
    if summary_file.is_file():
        summary_text = summary_file.read_text()
    elif eval_run_log.is_file():
        log_content = eval_run_log.read_text()
        summary_text = log_content[-2000:]

    parsed_scores: dict[str, float] = {}
    if summary_file.is_file():
        for line in summary_text.splitlines():
            m = re.search(r"PDM Score \(PDMS\)\s*:\s*([0-9.]+)", line)
            if m:
                parsed_scores["PDMS"] = float(m.group(1))
            m = re.search(r"No Collisions \(NC\)\s*:\s*([0-9.]+)", line)
            if m:
                parsed_scores["NC"] = float(m.group(1))
            m = re.search(r"Drivable Area \(DAC\)\s*:\s*([0-9.]+)", line)
            if m:
                parsed_scores["DAC"] = float(m.group(1))
            m = re.search(r"Ego Progress \(EP\)\s*:\s*([0-9.]+)", line)
            if m:
                parsed_scores["EP"] = float(m.group(1))
            m = re.search(r"Time-to-Coll \(TTC\)\s*:\s*([0-9.]+)", line)
            if m:
                parsed_scores["TTC"] = float(m.group(1))
            m = re.search(r"Comfort\s*:\s*([0-9.]+)", line)
            if m:
                parsed_scores["Comfort"] = float(m.group(1))

    return {
        "job_name": job_name,
        "gpus": gpus,
        "ckpt_path": str(ckpt_path),
        "returncode": res.returncode,
        "elapsed_sec": elapsed,
        "summary_text": summary_text,
        "parsed_scores": parsed_scores,
        "merged_csv": str(merged_csv) if merged_csv.is_file() else None,
        "summary_file": str(summary_file) if summary_file.is_file() else None,
        "eval_run_log": str(eval_run_log),
    }


def update_report_file(eval_results: dict[str, dict[str, Any]]) -> None:
    lines = [
        "=" * 72,
        f" NAVSIM AUTOMATED EVALUATION FINAL REPORT",
        f" Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
    ]
    for job_name, res in eval_results.items():
        lines.append(f"--- [Job: {job_name}] ---")
        lines.append(f"GPUs: {res.get('gpus')}")
        lines.append(f"Checkpoint: {res.get('ckpt_path')}")
        lines.append(f"Status: {'SUCCESS' if res.get('returncode') == 0 else 'FAILED'}")
        lines.append(f"Elapsed: {res.get('elapsed_sec', 0):.1f}s")
        if res.get("parsed_scores"):
            lines.append("Parsed Key Scores:")
            for k, v in res["parsed_scores"].items():
                lines.append(f"  - {k}: {v:.4f}")
        if res.get("merged_csv"):
            lines.append(f"Merged CSV: {res.get('merged_csv')}")
        lines.append("")
        if res.get("summary_text"):
            lines.append(res.get("summary_text"))
        lines.append("\n" + "=" * 72 + "\n")

    REPORT_FILE.write_text("\n".join(lines))
    log(f"Report updated at: {REPORT_FILE}")


def update_status_file(jobs: dict[str, dict[str, Any]], eval_results: dict[str, Any]) -> None:
    status_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "jobs": {},
    }
    for k, j in jobs.items():
        status_data["jobs"][k] = {
            "name": j["name"],
            "status": j["status"],
            "gpus": j["gpus"],
            "pids": j.get("last_pids", []),
            "latest_ckpt_epoch": j.get("latest_ckpt_epoch"),
            "progress_info": j.get("progress_info", ""),
            "has_result": k in eval_results,
            "scores": eval_results.get(k, {}).get("parsed_scores", {}),
        }
    STATUS_FILE.write_text(json.dumps(status_data, indent=2))


def extract_progress_from_tmux(pane_str: str) -> str:
    """Extract progress bar line like 'Epoch 28: 70%|...' from tmux output."""
    for line in reversed(pane_str.splitlines()):
        if "Epoch" in line and "%" in line:
            return line.strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    args = parser.parse_args()

    jobs: dict[str, dict[str, Any]] = {
        "student_tmux21": {
            "name": "para_ssr_v2_r34_student",
            "pattern": "run_training.py.*para_ssr_v2_r34_student",
            "exp_dir": WORK_DIRS / "para_ssr_v2_r34_student",
            "gpus": "4,5",
            "arch": "resnet34.tv_in1k",
            "tmux_target": "21:0",
            "extra_args": [
                "agent.config.image_architecture=resnet34.tv_in1k",
                "agent.config.use_task_interaction=true",
                "agent.config.use_det_motion_head=true",
                "agent.config.use_map_head=true",
                "agent.config.plan_anchor=true",
                "agent.config.plan_anchor_file=/workspace/byounggun/SSR/data/planning_vb/trajectory_anchors_256.npy",
                "agent.config.plan_score_file=/workspace/byounggun/SSR/data/planning_vb/pdm_score_256",
            ],
            "status": "training",
            "exit_confirm_count": 0,
            "latest_ckpt_epoch": None,
            "last_pids": [],
            "progress_info": "",
        },
        "distill_tmux19": {
            "name": "paradrive_distill_bev_selector_v8",
            "pattern": "run_training.py.*paradrive_distill_bev_selector_v8",
            "exp_dir": WORK_DIRS / "paradrive_distill_bev_selector_v8",
            "gpus": "6,7",
            "arch": "resnet34.tv_in1k",
            "tmux_target": "19:0",
            "extra_args": [
                "agent.config.image_architecture=resnet34.tv_in1k",
                "agent.config.use_task_interaction=true",
                "agent.config.use_det_motion_head=true",
                "agent.config.use_map_head=true",
                "agent.config.plan_anchor=true",
                "agent.config.plan_anchor_file=/workspace/byounggun/SSR/data/planning_vb/trajectory_anchors_256.npy",
                "agent.config.plan_score_file=/workspace/byounggun/SSR/data/planning_vb/pdm_score_256",
            ],
            "status": "training",
            "exit_confirm_count": 0,
            "latest_ckpt_epoch": None,
            "last_pids": [],
            "progress_info": "",
        },
    }

    log("=" * 72)
    log("Auto-monitor & Evaluator started.")
    log("Monitoring:")
    log("  1) tmux 21 (para_ssr_v2_r34_student) -> GPU 4,5 eval upon completion")
    log("  2) tmux 19 (paradrive_distill_bev_selector_v8) -> GPU 6,7 eval upon completion")
    log(f"Status file: {STATUS_FILE}")
    log(f"Report file: {REPORT_FILE}")
    log("=" * 72)

    eval_results: dict[str, dict[str, Any]] = {}
    loop_count = 0

    while True:
        all_done = True

        for key, job in jobs.items():
            if job["status"] in ("done", "failed"):
                continue

            all_done = False
            exp_dir = job["exp_dir"]
            ckpt_dir = get_latest_ckpt_dir(exp_dir)

            # Check running process
            pattern_pids = find_matching_pids(job["pattern"])
            is_alive = bool(pattern_pids)
            job["last_pids"] = pattern_pids

            cur_ep = get_latest_epoch_from_filenames(ckpt_dir)
            job["latest_ckpt_epoch"] = cur_ep

            # Parse progress bar from tmux pane
            pane_tail = get_tmux_pane_tail(job["tmux_target"])
            prog = extract_progress_from_tmux(pane_tail)
            if prog:
                job["progress_info"] = prog

            # Heartbeat logging every 5 iterations (~5 mins)
            if loop_count % 5 == 0:
                log(
                    f"[{job['name']}] alive={is_alive} (pids={pattern_pids}) | "
                    f"latest_saved_epoch={cur_ep}/29 | progress: {prog or 'N/A'}"
                )

            if not is_alive:
                job["exit_confirm_count"] = job.get("exit_confirm_count", 0) + 1
                log(f"[{job['name']}] Process not alive (check {job['exit_confirm_count']}/3)...")

                if job["exit_confirm_count"] >= 3:
                    # Confirmed exit! Check for checkpoint
                    log(f"[{job['name']}] Confirmed process exit. Locating final checkpoint in {ckpt_dir}...")
                    final_ckpt = check_checkpoint(ckpt_dir)

                    if final_ckpt and wait_file_stable(final_ckpt, wait_sec=5, retries=2):
                        log(f"[{job['name']}] Training completed successfully! Checkpoint verified: {final_ckpt}")
                        job["status"] = "evaluating"
                        update_status_file(jobs, eval_results)

                        result = run_evaluation(
                            job_name=job["name"],
                            ckpt_path=final_ckpt,
                            gpus=job["gpus"],
                            arch=job["arch"],
                            extra_hydra_args=job["extra_args"],
                        )
                        eval_results[key] = result
                        job["status"] = "done" if result["returncode"] == 0 else "failed"
                        update_report_file(eval_results)
                        update_status_file(jobs, eval_results)
                    else:
                        log(f"[{job['name']}] ERROR: Process exited but valid final checkpoint was not found! Latest saved epoch was: {cur_ep}")
                        job["status"] = "failed"
                        update_status_file(jobs, eval_results)
            else:
                # Reset confirm count if process is alive
                job["exit_confirm_count"] = 0

        update_status_file(jobs, eval_results)

        if all_done:
            log("All monitored training jobs have completed their evaluations!")
            break

        loop_count += 1
        time.sleep(args.interval)

    log("Auto-monitor & Evaluator finished cleanly.")


if __name__ == "__main__":
    main()
