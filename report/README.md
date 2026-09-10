# PARA-SSR NAVSIM 학습·평가 실행 가이드

이 문서는 다음 세 실행을 구분한다.

Metric을 학습하는 multi-candidate planner는 별도 `agent=para_ssr_metric_agent`로 제공한다.
Train-only anchor 생성, metric world cache 준비, smoke 및 학습/평가 명령은
[`11_para_ssr_metric_planner.md`](11_para_ssr_metric_planner.md)에 정리했다.

| 실행 | 목적 | GPU 방식 | 대표 산출물 |
|---|---|---|---|
| 학습 | PARA-SSR 30 epoch 학습 | DDP (reference: 2 GPU) | Lightning checkpoint |
| NAVSIM planning 평가 | trajectory의 PDM score | 단일 GPU | scenario별 CSV |
| Detection/map auxiliary 평가 | perception head 상대 비교 | multi-GPU 추론 + CPU 집계 | JSON/CSV |

Detection/map auxiliary metric은 NAVSIM 공식 leaderboard metric이 아니다. 같은 protocol과
split을 사용한 PARA-SSR 계열 모델끼리 비교할 때만 사용한다.

현재 front-only ROI 모델의 **PDM 평가**는 지원하지만, 아래 auxiliary mAP 실행기는 아직
이전 전후방 detection ROI/V2 protocol에 고정되어 새 기본 모델을 거부한다. 새 front-only
baseline/metric 모델에 auxiliary 명령을 그대로 실행하면 안 된다. 기존 결과의 재현에는
당시 code revision/config를 함께 사용하고, 새 ROI의 mAP는 별도 protocol 이관이 필요하다.

---

## 1. 실행 전 확인

아래 명령은 clone한 저장소 내부 어느 위치에서든 repository root로 이동한다. Conda 환경 이름은
`environment.yml`의 기본값인 `ssr-navsim`이며, 다른 이름으로 만든 경우 해당 환경을
활성화하면 된다.

```bash
cd "$(git rev-parse --show-toplevel)"
conda activate ssr-navsim

python -c "import torch, navsim, nuplan; print(torch.__version__, torch.cuda.is_available())"
```

NAVSIM 데이터의 물리 저장 위치는 사용자마다 달라도 된다. 코드가 기대하는 repository 상대
구조만 맞추면 된다. `data`가 아직 없을 때 한 번만 다음처럼 연결한다.

```bash
# /path/to/navsim은 dataset/과 exp/를 포함하는 각 사용자의 NAVSIM root로 교체한다.
ln -s /path/to/navsim ./data

# 결과를 repository 안에 둘 경우
mkdir -p ./work_dirs

# 또는 용량이 큰 별도 디스크를 사용할 경우, 위 mkdir 대신 다음처럼 연결한다.
# ln -s /path/to/experiment-storage ./work_dirs
```

이미 `data`나 `work_dirs`가 존재하면 삭제하거나 덮어쓰지 말고 올바른 위치인지 먼저 확인한다.
필수 구조와 확인 명령은 다음과 같다.

```bash
test -d data/dataset/maps
test -d data/dataset/navsim_logs
test -d data/dataset/sensor_blobs
test -d data/exp/metric_cache  # planning/PDM 평가에 필요

readlink -f data
readlink -f work_dirs
```

마지막 두 명령의 출력은 고정값이 아니라 각 사용자가 선택한 데이터와 결과 root여야 한다.

아래 예시는 GPU ID를 직접 박아두지 않고 사용자가 한 번 선택한 변수를 재사용한다.

```bash
export TRAIN_GPUS=0,1  # DDP 학습에 사용할 GPU 두 장
export EVAL_GPU=0      # 순차 PDM 평가에 사용할 GPU 한 장
export AUX_GPUS=0,1    # auxiliary shard 추론에 사용할 GPU들
```

제공된 train/eval wrapper가 `NUPLAN_MAPS_ROOT`, `OPENSCENE_DATA_ROOT`,
`NAVSIM_DEVKIT_ROOT`, `NAVSIM_EXP_ROOT`, `PYTHONPATH`를 자동으로 설정한다. 수동 Python
entry point를 실행할 때만 직접 설정하면 된다.

---

## 2. 학습 전 smoke test

실제 NAVSIM 8개 scene으로 train forward/loss/backward/optimizer step과 validation 한 batch를
fp32로 확인한다.

```bash
CUDA_VISIBLE_DEVICES="$EVAL_GPU" bash scripts/training/smoke_para_ssr.sh
```

이 smoke는 본 학습과 같은 전방 3개 카메라를 사용하지만 BEV와 이미지 해상도를 줄이므로
모델 성능이나 full-model throughput을
측정하는 용도가 아니다.

---

## 3. 본 학습

### 3.1 현재 reference recipe

| 항목 | 값 |
|---|---:|
| 카메라 입력 | 전방 3개: `cam_f0`, `cam_l0`, `cam_r0` |
| 이미지 history | `[2, 3]`: 과거 1 + 현재 1 frame |
| GPU | 2장 |
| microbatch | 4/GPU |
| gradient accumulation | 16 |
| effective global batch | `2 × 4 × 16 = 128` |
| epoch | 30 |
| optimizer / LR | AdamW / `1e-4` |
| precision | fp32 |
| gradient clip | norm `35.0` |
| validation | 5 epoch마다 |
| dataset | `trainval` + `navtrain` SceneFilter, online target 생성 |

전방 카메라 집합은 WoTE와 동일하다. PARA-SSR은 세 이미지를 각각 `768 × 416`으로
전처리하고, 각 카메라의 `lidar2img`로 BEVFormer에 투영한다. 배치 입력은
`[B, 2, 3, 3, 416, 768]` (`batch, time, camera, RGB, height, width`)이며,
나머지 5개 카메라와 LiDAR는 로드하지 않는다. 카메라 임베딩도 `[3, 256]`이다.
WoTE의 단일-frame panorama 전처리와 LiDAR fusion까지 복제한 설정은 아니다.

세 task는 하나의 전방 ROI를 공유한다. SSR 좌표로 shared BEV, detection/motion,
vector map 모두 `x_right ∈ [-32, 32] m`, `y_forward ∈ [0, 32] m`이다.
detection/motion GT는 이 사각형에 더해 전방축 기준 **±80°** box-center FOV를
적용하며, 학습 target과 auxiliary mAP GT가 같은 필터 함수를 사용한다. HD-map GT에는
각도 필터를 적용하지 않고 공통 사각 ROI만 적용한다.

기존 8-camera checkpoint와는 카메라 임베딩 크기가 다르므로 **새 학습**으로 시작한다.
또한 ROI와 detection GT 모집단도 달라졌으므로 기존 3-camera/full-BEV checkpoint 역시
새 기본 설정과 같은 실험으로 비교하면 안 된다. 과거의 서로 다른 ROI checkpoint를
재평가하려면 보관된 config만 덮는 것으로는 부족하며, 당시 code revision과 config를 함께
사용해야 한다. 현재 agent는 mismatched/rear ROI를 의도적으로 거부한다.
당시 카메라 순서는 `[cam_f0, cam_l0, cam_l1, cam_l2, cam_r0, cam_r1, cam_r2, cam_b0]`였다.
`camera_names`는 feature cache key에도 포함되어 3-camera와 8-camera 캐시가 분리된다.

### 3.2 새 학습 시작

`EXPERIMENT`는 기존 결과와 겹치지 않는 고유한 이름을 사용한다.

```bash
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
EXPERIMENT=para_ssr_front3_30ep \
WANDB=0 \
bash scripts/training/train_para_ssr.sh
```

W&B를 사용할 때는 `WANDB=0`을 빼고 로그인한다. 네트워크 없이 기록하려면 다음처럼 한다.

```bash
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
EXPERIMENT=para_ssr_front3_30ep \
WANDB_MODE=offline \
bash scripts/training/train_para_ssr.sh
```

주요 환경변수:

| 변수 | 기본값 | 설명 |
|---|---:|---|
| `EXPERIMENT` | `para_ssr_front3` | `work_dirs/` 아래 실행 이름 |
| `BATCH_SIZE` | 4 | GPU당 microbatch |
| `ACCUMULATE` | 16 | gradient accumulation |
| `MAX_EPOCHS` | 30 | 전체 목표 epoch |
| `WORKERS` | 8 | dataloader worker/process |
| `LR` | `1e-4` | optimizer learning rate |
| `WANDB` | 1 | `0`이면 W&B 비활성화; TensorBoard는 계속 기록 |

GPU 수나 batch를 바꾸면 다음 값을 직접 확인한다.

```text
global batch = GPU 수 × BATCH_SIZE × ACCUMULATE
```

현재 LR recipe와 동일하게 비교하려면 global batch 128을 유지한다. `BATCH_SIZE=4`일 때 예시는
다음과 같다.

| GPU 수 | `ACCUMULATE` | global batch |
|---:|---:|---:|
| 1 | 32 | 128 |
| 2 | 16 | 128 |
| 4 | 8 | 128 |

GPU 메모리가 부족하면 `BATCH_SIZE`를 낮추고 `ACCUMULATE`를 높인다. 반대로 GPU 수나
microbatch를 늘릴 때도 global batch와 LR을 함께 검토한다. `WORKERS` 역시 각 DDP process마다
적용되므로 host CPU/RAM에 맞게 조정한다.

Hydra override는 명령 마지막에 추가할 수 있다.

```bash
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
EXPERIMENT=para_ssr_debug \
WANDB=0 \
MAX_EPOCHS=2 \
bash scripts/training/train_para_ssr.sh \
  trainer.params.check_val_every_n_epoch=1
```

### 3.3 학습 산출물

```text
work_dirs/<EXPERIMENT>/code/hydra/config.yaml
work_dirs/<EXPERIMENT>/run_training.log
work_dirs/<EXPERIMENT>/lightning_logs/version_*/events.out.tfevents.*
work_dirs/<EXPERIMENT>/lightning_logs/version_*/checkpoints/epoch=*.ckpt
work_dirs/<EXPERIMENT>/lightning_logs/version_*/checkpoints/last.ckpt
```

매 epoch checkpoint와 resume용 `last.ckpt`를 저장한다. TensorBoard 확인:

```bash
tensorboard --logdir work_dirs/para_ssr_front3_30ep/lightning_logs --port 6006
```

### 3.4 중단된 학습 resume

```bash
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
EXPERIMENT=para_ssr_front3_30ep \
WANDB=0 \
MAX_EPOCHS=30 \
RESUME_CHECKPOINT=/absolute/path/to/last.ckpt \
bash scripts/training/train_para_ssr.sh
```

`RESUME_CHECKPOINT`는 model weight뿐 아니라 optimizer, scheduler, 현재 epoch/global step과
GradBalancer state까지 복원한다. `MAX_EPOCHS=30`은 추가 30 epoch가 아니라 **resume 전후를
포함한 전체 목표 epoch**다. 평가용 `agent.checkpoint_path`는 weights-only load이므로 resume에
사용하지 않는다.

완료된 30-epoch run을 단순히 `MAX_EPOCHS=60`으로 resume해 연장하는 용도로는 사용하지
않는다. Resume은 checkpoint에 저장된 기존 30-epoch scheduler state도 복원하므로 LR schedule이
자동으로 60-epoch용으로 재설계되지 않는다. 학습 기간 연장은 scheduler와 resume policy를
별도로 정한 뒤 새 실험으로 취급한다.

---

## 4. NAVSIM planning/PDM 평가

이 평가는 모델이 출력한 ego trajectory를 NAVSIM PDM scorer로 채점한다. Detection/map head의
mAP를 계산하는 평가가 아니다.

### 4.1 전체 navtest 평가

```bash
CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
bash scripts/evaluation/eval_para_ssr.sh \
  /absolute/path/to/model.ckpt \
  experiment_name=eval/my_model_pdm
```

- 입력 checkpoint는 Lightning `state_dict`가 들어 있는 `.ckpt`여야 한다.
- `navtest` 12,146개 token과 `data/exp/metric_cache`를 사용한다.
- 이 runner는 **단일 GPU 순차 평가**다. 여러 GPU ID를 주어도 분산되지 않으므로
  `EVAL_GPU`에 GPU 하나만 지정한다.
- 모델마다 고유한 `experiment_name`을 사용한다.
- 기본 입력은 전방 3개 카메라다. 8-camera checkpoint의 평가에는 학습 때의
  `agent.config.camera_names` 순서와 나머지 architecture/ROI 설정을 함께 복원한다.
- checkpoint를 만든 agent config가 현재 config와 다르면 필요한 `agent.config.*` override를
  함께 전달해야 한다. Parameter key/shape는 strict load로 검사하지만 모든 비-tensor 설정을
  checkpoint가 보존하는 것은 아니다.

짧은 pipeline smoke만 필요하면 다음처럼 제한할 수 있다. `max_scenes` subset은 filesystem
순서에 의존하므로 성능 비교에는 사용하지 않는다.

```bash
CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
bash scripts/evaluation/eval_para_ssr.sh \
  /absolute/path/to/model.ckpt \
  experiment_name=eval/my_model_pdm_smoke \
  scene_filter.max_scenes=8
```

### 4.2 산출물과 완료 확인

```text
work_dirs/eval/my_model_pdm/<YYYY.MM.DD.HH.MM.SS>.csv
work_dirs/eval/my_model_pdm/run_pdm_score_gpu.log
work_dirs/eval/my_model_pdm/code/hydra/config.yaml
```

CSV에는 token별 다음 지표와 최종 `average` 행이 들어간다.

```text
no_at_fault_collisions, drivable_area_compliance,
driving_direction_compliance, ego_progress,
time_to_collision_within_bound, comfort, score
```

CSV 파일 존재만으로 full 평가 완료를 판단하면 안 된다. 로그에서 아래 두 조건과 마지막
`average` 행을 함께 확인한다.

```text
Number of successful scenarios: 12146
Number of failed scenarios: 0
```

---

## 5. Detection/map auxiliary 평가

PARA-SSR의 parallel perception head를 별도로 평가한다.

| metric | 평가 내용 | 평가하지 않는 것 |
|---|---|---|
| `NAVSIMAuxDet/center_mAP` | 7-class 2-D center AP @ 0.5/1/2/4 m | size, yaw, velocity, TP error, NDS |
| `NAVSIMAuxMap/chamfer_mAP` | divider/crosswalk/boundary Chamfer AP @ 0.5/1/1.5 m | polygon 면적, topology, raster IoU |

둘 다 **비공식 auxiliary metric**이며 동일 evaluator protocol의 모델 상대 비교용이다.

### 5.1 Checkpoint 평가

checkpoint와 **그 학습 run에 저장된 Hydra config**를 반드시 짝지어 사용한다. Checkpoint,
config, token 집합, evaluator/wrapper source 또는 protocol이 달라지면 기존 output directory의
manifest identity 검사가 거부한다. 서로 다른 실행에는 고유한 `AUX_EXPERIMENT`를 지정한다.

```bash
GPU_IDS="$AUX_GPUS" \
AUX_BATCH_SIZE=4 \
AUX_EXPERIMENT=eval/my_model_aux \
AUX_TRAINING_CONFIG=/absolute/path/to/run/code/hydra/config.yaml \
bash scripts/evaluation/eval_para_ssr_aux.sh \
  /absolute/path/to/model.ckpt
```

이 평가는 PDM 평가와 달리 지정한 GPU마다 deterministic token shard를 처리한 뒤 CPU 한
process가 전체 AP를 집계한다. 같은 명령을 다시 실행하면 완료된 token record를 검증하고
재사용하므로 중단된 실행을 안전하게 이어갈 수 있다. `AUX_BATCH_SIZE=4`가 GPU 메모리에 맞지
않으면 낮춰서 실행한다.

Wrapper는 현재 활성화된 environment의 `python`을 사용한다. 다른 interpreter를 명시해야 할
때만 `SSR_NAVSIM_PYTHON=/absolute/path/to/python`을 추가한다.

### 5.2 산출물과 완료 확인

```text
work_dirs/eval/my_model_aux/manifest.json
work_dirs/eval/my_model_aux/records/<token>.npz
work_dirs/eval/my_model_aux/aux_metrics.csv
work_dirs/eval/my_model_aux/aux_metrics.json
```

`aux_metrics.json`이 최종 completion marker다. 정상 full run은 JSON 최상위의
`num_tokens == 12146`이며 Detection과 map 결과가 모두 들어 있다. `records/`는 resume
용이므로 Git에 push하지 않는다.

개발 보고서에 기록된 epoch-30 reference 수치는 다음과 같다. Checkpoint와 대용량 record가
Git repository에 포함된다는 뜻은 아니며, evaluator 재현 확인용 기준값이다.

| 평가 | 결과 |
|---|---:|
| NAVSIM PDM score | 0.846130 |
| Detection center mAP | 0.362148 |
| Vector-map Chamfer mAP | 0.228846 |

---

## 6. 코드 및 문서 검증

```bash
bash -n scripts/training/smoke_para_ssr.sh
bash -n scripts/training/train_para_ssr.sh
bash -n scripts/evaluation/eval_para_ssr.sh
bash -n scripts/evaluation/eval_para_ssr_aux.sh

python -m pytest -q \
  tests/test_para_ssr_targets.py \
  tests/test_para_ssr_model_invariants.py \
  tests/test_para_ssr_aux_metrics.py \
  tests/test_aux_evaluation_runner.py \
  navsim/agents/para_ssr/test_loss_parity.py
```

관련 상세 문서:

- [`09_navsim_port_progress.md`](09_navsim_port_progress.md): 포팅·학습·PDM/aux 전체 실행 기록
- [`10_para_ssr_navsim_vector_mapping_reference.md`](10_para_ssr_navsim_vector_mapping_reference.md):
  vector-map GT 좌표·task·loss·평가 protocol
- [`../docs/PARA_SSR_NAVSIM.md`](../docs/PARA_SSR_NAVSIM.md): 환경 설치와 짧은 사용법
