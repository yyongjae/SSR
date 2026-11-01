#!/usr/bin/env python3
"""
Continuous Autonomous Pipeline:
Monitors current training (v4 in version_1) -> Evaluates on GPUs 6,7 ->
Analyzes metrics -> Writes docs/bev_selector_v{N+1}.md -> Launches next version -> Loops continuously!
"""

import os
import sys
import time
import subprocess
import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][auto_pipeline][%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/workspace/byounggun/ssr_outputs/auto_pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger("auto_pipeline")

REPO = Path("/workspace/byounggun/SSR")
EXP_ROOT = Path("/workspace/byounggun/ssr_outputs")
LIGHTNING_LOGS = EXP_ROOT / "paradrive_distill_bev_selector/lightning_logs"
PYTHON = "/root/miniconda3/envs/ssr/bin/python"


def is_training_active() -> bool:
    res = subprocess.run(["pgrep", "-f", "run_training.py"], capture_output=True, text=True)
    return bool(res.stdout.strip())


def get_latest_version_dir() -> Path:
    versions = sorted(
        [d for d in LIGHTNING_LOGS.glob("version_*") if d.is_dir()],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else -1,
    )
    if not versions:
        raise FileNotFoundError(f"No version_* found in {LIGHTNING_LOGS}")
    return versions[-1]


def get_active_tmux_session() -> str:
    res = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
    lines = res.stdout.strip().splitlines()
    for name in ["bg_gpu67", "bg_v4", "bg"]:
        if any(line.startswith(f"{name}:") for line in lines):
            return name
    return "bg_v4"


def is_training_active_for_dir(version_dir: Path) -> bool:
    # Check if a python process with this experiment's run directory is running
    res = subprocess.run(["pgrep", "-f", "run_training.py.*paradrive_distill_bev_selector"], capture_output=True, text=True)
    # Exclude v5 process
    pids = res.stdout.strip().split()
    for pid in pids:
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_text()
            if "paradrive_distill_bev_selector_v5" not in cmd and "paradrive_distill_bev_selector" in cmd:
                return True
        except Exception:
            continue
    return False


def wait_for_training(version_dir: Path, poll_interval: int = 60) -> Path:
    ckpt_dir = version_dir / "checkpoints"
    v_num = version_dir.name
    logger.info("Monitoring %s (checkpoints in %s)...", v_num, ckpt_dir)

    while True:
        epoch29_ckpts = list(ckpt_dir.glob("epoch=29*.ckpt"))
        training_active = is_training_active_for_dir(version_dir)

        if epoch29_ckpts:
            ckpt = epoch29_ckpts[0]
            logger.info("[%s] Found epoch 29 checkpoint: %s", v_num, ckpt)
            if not training_active:
                logger.info("[%s] Training process has finished.", v_num)
                return ckpt
            else:
                logger.info("[%s] Epoch 29 checkpoint saved, waiting 45s for trainer shutdown...", v_num)
                time.sleep(45)
                return ckpt

        if not training_active:
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists():
                logger.warning("[%s] Training process stopped. Using last.ckpt: %s", v_num, last_ckpt)
                return last_ckpt
            else:
                logger.error("[%s] Training process stopped but no checkpoint found in %s", v_num, ckpt_dir)
                raise RuntimeError(f"Training stopped unexpectedly in {version_dir}")

        # Log current progress periodically
        sess = get_active_tmux_session()
        try:
            tmux_out = subprocess.run(
                ["tmux", "capture-pane", "-pt", f"{sess}:0"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            epoch_lines = [line.strip() for line in tmux_out.splitlines() if "Epoch " in line and "%" in line]
            if epoch_lines:
                logger.info("[%s] Current progress: %s", v_num, epoch_lines[-1])
        except Exception:
            pass

        time.sleep(poll_interval)


def run_evaluation(ckpt_path: Path, version_tag: str) -> dict:
    logger.info("Starting NAVSIM evaluation on GPUs 6, 7 for %s: %s", version_tag, ckpt_path)
    snapshot_dir = EXP_ROOT / f"eval_snapshots/paradrive_distill_bev_selector_{version_tag}_epoch29"
    experiment_name = f"eval/paradrive_distill_bev_selector_{version_tag}_epoch29"

    env = os.environ.copy()
    env["PYTHON"] = PYTHON
    env["GPU0"] = "6"
    env["GPU1"] = "7"
    env["CKPT_SRC"] = str(ckpt_path)
    env["SNAPSHOT_DIR"] = str(snapshot_dir)
    env["EXPERIMENT_NAME"] = experiment_name
    env["IMAGE_ARCHITECTURE"] = "resnet34.tv_in1k"
    env["NAVSIM_EXP_ROOT_OVERRIDE"] = str(EXP_ROOT)
    env["NAVTEST_LOGS"] = str(REPO / "data/dataset/navsim_logs/test")
    env["NAVTEST_BLOBS"] = str(REPO / "data/dataset/sensor_blobs/test")

    eval_script = REPO / "scripts/evaluation/eval_para_ssr_distill_epoch29_gpus45.sh"
    eval_log_path = EXP_ROOT / f"eval/eval_{version_tag}.log"
    eval_log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(eval_log_path, "w") as log_f:
        p = subprocess.Popen(
            ["bash", str(eval_script)],
            cwd=str(REPO),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        logger.info("[%s] Evaluation process started (PID %d). Log: %s", version_tag, p.pid, eval_log_path)
        p.wait()
        if p.returncode != 0:
            logger.error("[%s] Evaluation failed with return code %d. See %s", version_tag, p.returncode, eval_log_path)
            raise RuntimeError(f"Evaluation failed with return code {p.returncode}")

    logger.info("[%s] Evaluation completed successfully!", version_tag)
    return parse_results(experiment_name)


def parse_results(experiment_name: str) -> dict:
    merged_csv_path = EXP_ROOT / experiment_name / "merged.csv"
    aux_json_path = EXP_ROOT / f"{experiment_name}_aux" / "aux_metrics.json"

    if not merged_csv_path.exists():
        raise FileNotFoundError(f"merged.csv not found at {merged_csv_path}")

    df = pd.read_csv(merged_csv_path)
    total_scenes = len(df[df["token"] != "average"])
    avg_row = df[df["token"] == "average"].iloc[0].to_dict()

    scenes_df = df[df["token"] != "average"].copy()
    dac_zero = int((scenes_df["drivable_area_compliance"] == 0).sum())
    ttc_zero = int((scenes_df["time_to_collision_within_bound"] == 0).sum())
    nc_less1 = int((scenes_df["no_at_fault_collisions"] < 1.0).sum())
    score_zero = int((scenes_df["score"] == 0).sum())

    results = {
        "valid": float(avg_row.get("valid", 1.0)),
        "no_at_fault_collisions": float(avg_row.get("no_at_fault_collisions", 0.0)),
        "drivable_area_compliance": float(avg_row.get("drivable_area_compliance", 0.0)),
        "driving_direction_compliance": float(avg_row.get("driving_direction_compliance", 0.0)),
        "ego_progress": float(avg_row.get("ego_progress", 0.0)),
        "time_to_collision_within_bound": float(avg_row.get("time_to_collision_within_bound", 0.0)),
        "comfort": float(avg_row.get("comfort", 0.0)),
        "score": float(avg_row.get("score", 0.0)),
        "dac_zero": dac_zero,
        "ttc_zero": ttc_zero,
        "nc_less1": nc_less1,
        "score_zero": score_zero,
        "total_scenes": total_scenes,
        "det_map": None,
        "map_map": None,
    }

    if aux_json_path.exists():
        try:
            aux_data = json.loads(aux_json_path.read_text())["metrics"]
            results["det_map"] = float(aux_data["detection"]["mAP"])
            results["map_map"] = float(aux_data["map"]["mAP"])
        except Exception as e:
            logger.warning("Could not read aux metrics: %s", e)

    return results


def write_v5_document(res_v4: dict, next_config: dict) -> Path:
    doc_path = REPO / "docs/bev_selector_v5.md"
    logger.info("Writing evaluation results and v5 plan to %s", doc_path)

    content = f"""# BEV Selector Distillation v5
## v4 결과 분석 및 v5 개선 계획

**기준 문서**: [`bev_selector_v4.md`](bev_selector_v4.md)  
**소속**: `/workspace/byounggun/SSR`  
**작성일**: 2026-10-06  
**평가 체크포인트**: `ssr_outputs/paradrive_distill_bev_selector/lightning_logs/version_1/checkpoints/epoch=29-step=19950.ckpt`  
**평가 데이터셋**: navtest {res_v4['total_scenes']} tokens  

---

## 1. 점수 비교 (v4 baseline vs v1 vs v3 vs v4)

| 항목 | v4 baseline | v1 selector | v3 selector | v4 (GradBalancer) |
|---|---:|---:|---:|---:|
| valid | 1.0000 | 1.0000 | 1.0000 | {res_v4['valid']:.4f} |
| no_at_fault_collisions | 0.9810 | 0.9797 | 0.9816 | {res_v4['no_at_fault_collisions']:.4f} |
| drivable_area_compliance | 0.9371 | 0.9152 | 0.9199 | {res_v4['drivable_area_compliance']:.4f} |
| driving_direction_compliance | 1.0000 | 1.0000 | 1.0000 | {res_v4['driving_direction_compliance']:.4f} |
| ego_progress | 0.8020 | 0.7836 | 0.7879 | {res_v4['ego_progress']:.4f} |
| time_to_collision_within_bound | 0.9439 | 0.9373 | 0.9407 | {res_v4['time_to_collision_within_bound']:.4f} |
| comfort | 0.9999 | 1.0000 | 0.9999 | {res_v4['comfort']:.4f} |
| **score** | **0.8563** | **0.8343** | **0.8405** | **{res_v4['score']:.4f}** |

### 실패 장면 수
- **DAC 실패 (도로 이탈)**: v4 base 764 -> v1 1030 -> v3 973 -> **v4 {res_v4['dac_zero']}**
- **TTC 실패**: v4 base 681 -> v1 761 -> v3 720 -> **v4 {res_v4['ttc_zero']}**
- **NC < 1 (충돌)**: v4 base 246 -> v1 268 -> v3 244 -> **v4 {res_v4['nc_less1']}**
- **Score 0 장면**: v4 base 969 -> v1 1237 -> v3 1162 -> **v4 {res_v4['score_zero']}**

{"### Auxiliary Detection & Map mAP" if res_v4['det_map'] is not None else ""}
{f"- det mAP: {res_v4['det_map']:.4f}" if res_v4['det_map'] is not None else ""}
{f"- map mAP: {res_v4['map_map']:.4f}" if res_v4['det_map'] is not None else ""}

---

## 2. v4 결과 진단 및 v5 개선 방향

v4에서 GradBalancer (plan 0.4 / det 0.3 / map 0.3)를 켜서 planner 그래디언트의 영향력을 회복시켰다.

### 진단:
1. **DAC 이탈 실패 분석**: DAC 실패수가 {res_v4['dac_zero']}장으로 측정됨.
2. **개선 전략 (v5)**:
{next_config['rationale']}

---

## 3. v5 실행 파라미터

- GradBalancer: `{next_config.get('grad_balance_target', '{plan:0.4,det:0.3,map:0.3}')}`
- `distill_selector_struct_mix`: `{next_config.get('struct_mix', 0.5)}`
- `distill_selector_plan_tau`: `{next_config.get('plan_tau', 0.3)}`
- Backbone: `{next_config.get('image_architecture', 'resnet34.tv_in1k')}`

---

## 4. 실행 명령 (GPU 6, 7)

```bash
cd /workspace/byounggun/SSR
source env.vast.sh
CUDA_VISIBLE_DEVICES=6,7 nohup bash scripts/training/run_bev_selector_distill.sh \\
  agent.config.grad_balance_target='{next_config.get('grad_balance_target', '{plan:0.4,det:0.3,map:0.3}')}' \\
  agent.config.distill_selector_struct_mix={next_config.get('struct_mix', 0.5)} \\
  agent.config.distill_selector_plan_tau={next_config.get('plan_tau', 0.3)} \\
  wandb.tags='[bev-selector,r34,v5]' \\
  > /workspace/byounggun/ssr_outputs/v5_train.log 2>&1 &
```
"""
    doc_path.write_text(content.strip() + "\n")
    return doc_path


def plan_v5_configuration(res_v4: dict) -> dict:
    # Based on docs/bev_selector_v2.md and v3.md:
    # If DAC is still the main bottleneck (> 800), increase map bank struct mix (beta)
    # to 0.7 so slots more aggressively attend to road boundary rings, or sharpen planner tau to 0.2
    if res_v4["dac_zero"] > 800:
        rationale = (
            "- DAC 이탈이 여전히 주된 감점 요인(800건 초과)이므로, map bank의 구조적 타깃 가중치(struct_mix, beta)를 "
            "0.5에서 0.7로 상향하여 road boundary 링에 선택기가 더 밀착하도록 유도.\n"
            "- 또한 planner cover target의 softmax temperature(tau)를 0.3에서 0.25로 낮추어 "
            "불필요하게 넓게 퍼지는 커버를 좁히고 도로 경계 추종 집중도를 극대화함."
        )
        return {
            "grad_balance_target": "{plan:0.4,det:0.3,map:0.3}",
            "struct_mix": 0.7,
            "plan_tau": 0.25,
            "image_architecture": "resnet34.tv_in1k",
            "rationale": rationale,
        }
    else:
        rationale = (
            "- DAC 이탈이 유의미하게 억제되었으므로, ResNet-34에서 ResNet-50 backbone으로 스케일업하여 "
            "v4 baseline (0.8563)을 넘어서는 종합 고득점 달성을 목표로 함."
        )
        return {
            "grad_balance_target": "{plan:0.4,det:0.3,map:0.3}",
            "struct_mix": 0.5,
            "plan_tau": 0.3,
            "image_architecture": "resnet50.tv_in1k",
            "rationale": rationale,
        }


def launch_next_iteration_gpus67(config: dict, next_tag: str):
    logger.info("Preparing and launching %s training on GPUs 6, 7...", next_tag)
    log_file = EXP_ROOT / f"gpu67_{next_tag}_train.log"
    sess_name = "bg_gpu67"

    cmd = (
        f"cd {REPO} && source env.vast.sh && "
        f"CUDA_VISIBLE_DEVICES=6,7 bash scripts/training/run_bev_selector_distill.sh "
        f"agent.config.grad_balance_target='{config.get('grad_balance_target')}' "
        f"agent.config.distill_selector_struct_mix={config.get('struct_mix')} "
        f"agent.config.distill_selector_plan_tau={config.get('plan_tau')} "
        f"agent.config.image_architecture={config.get('image_architecture')} "
        f"wandb.tags='[bev-selector,r34,{next_tag}]' "
        f"> {log_file} 2>&1"
    )

    subprocess.run(["tmux", "kill-session", "-t", sess_name], capture_output=True)
    res = subprocess.run(["tmux", "new-session", "-d", "-s", sess_name, "bash", "-c", cmd])
    if res.returncode != 0:
        logger.error("Failed to create tmux session %s: %s", sess_name, res.stderr)
        raise RuntimeError(f"tmux {sess_name} launch failed")

    logger.info("%s training successfully launched in tmux session '%s'! Log: %s", next_tag, sess_name, log_file)


def run_continuous_pipeline():
    logger.info("=== CONTINUOUS AUTO PIPELINE ACTIVE ===")
    while True:
        current_version_dir = get_latest_version_dir()
        v_idx = int(current_version_dir.name.split("_")[1])
        # v3 was version_0, v4 is version_1, next will be v4_iter2...
        v_tag = f"v{v_idx + 3}"
        next_tag = f"v{v_idx + 4}"

        logger.info("--- Watching %s (%s) ---", current_version_dir.name, v_tag)
        ckpt_path = wait_for_training(current_version_dir, poll_interval=60)
        logger.info("[%s] Finished training! Checkpoint: %s", v_tag, ckpt_path)

        # 1. Run evaluation
        results = run_evaluation(ckpt_path, v_tag)
        logger.info("[%s] Evaluation results: %s", v_tag, results)

        # 2. Plan next iteration
        next_config = plan_v5_configuration(results)

        # 3. Document
        doc_path = write_v5_document(results, next_config)
        logger.info("Documented to %s", doc_path)

        # 4. Launch next iteration
        launch_next_iteration_gpus67(next_config, next_tag)
        logger.info("Waiting 120s for new training process to initialize...")
        time.sleep(120)


if __name__ == "__main__":
    run_continuous_pipeline()
