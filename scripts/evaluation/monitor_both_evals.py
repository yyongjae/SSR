#!/usr/bin/env python3
"""Monitor both evaluations (v8 ep20 and student ep29) running on GPUs 6 and 7.
Logs progress and writes comparison summary when finished.
"""
import time
import datetime
from pathlib import Path

REPO = Path("/workspace/byounggun/SSR")
EVAL_DIR = REPO / "work_dirs/eval"
V8_SUMMARY = EVAL_DIR / "paradrive_distill_bev_selector_v8_ep20/evaluation_summary.txt"
STUDENT_SUMMARY = EVAL_DIR / "para_ssr_v2_r34_student_epoch29/evaluation_summary.txt"
COMPARE_FILE = REPO / "work_dirs/EVAL_COMPARE_SUMMARY.txt"
LOG_FILE = REPO / "work_dirs/monitor_both_evals.log"

def log(msg: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log("Starting monitor for both evaluations on GPUs 6 and 7.")

while True:
    v8_done = V8_SUMMARY.is_file()
    student_done = STUDENT_SUMMARY.is_file()
    
    if v8_done and student_done:
        log("Both evaluations have completed! Creating comparison summary...")
        lines = [
            "=" * 72,
            f" NAVSIM EVALUATION COMPARISON REPORT (GPUs 6,7)",
            f" Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 72,
            "",
            "--- [1] Student Baseline (Epoch 29) ---",
            STUDENT_SUMMARY.read_text(),
            "",
            "--- [2] Distill v8 (Flipped ReSMap, Epoch 20) ---",
            V8_SUMMARY.read_text(),
            "=" * 72,
        ]
        COMPARE_FILE.write_text("\n".join(lines))
        log(f"Comparison report saved to {COMPARE_FILE}")
        break
        
    time.sleep(30)

log("Monitor finished.")
