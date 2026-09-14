# PARA-SSR NAVSIM 학습·평가 실행 가이드

이 문서는 다음 세 실행을 구분한다.

| 실행 | 목적 | GPU 방식 | 대표 산출물 |
|---|---|---|---|
| 학습 | PARA-SSR 30 epoch 학습 | DDP (reference: 2 GPU) | Lightning checkpoint |
| NAVSIM planning 평가 | trajectory의 PDM score (EPDMS는 §4.3) | 단일 GPU | scenario별 CSV |
| Detection/map auxiliary 평가 | perception head 상대 비교 | multi-GPU 추론 + CPU 집계 | JSON/CSV |

Detection/map auxiliary metric은 NAVSIM 공식 leaderboard metric이 아니다. 같은 protocol과
split을 사용한 PARA-SSR 계열 모델끼리 비교할 때만 사용한다.

Auxiliary mAP 실행기는 metric protocol **V3**를 사용한다. shared BEV와 같은 전방 ROI
(`x_right ∈ [-32, 32] m`, `y_forward ∈ [0, 32] m`)에 detection GT만 ±80° FOV를 더 적용하고,
checkpoint에 존재하는 head의 task만 채점한다(§5). 이전 전후방 ROI의 V2 결과와는 GT 모집단이
달라 수치를 비교하지 않으며, V2 결과의 재현에는 당시 code revision/config를 사용한다.

관련 문서:

- [`12_lidar_bev_encoder_50x100.md`](12_lidar_bev_encoder_50x100.md): LiDAR 입력 branch와
  50×100 BEV grid의 설계·검증
- [`13_head_ablation_evaluation.md`](13_head_ablation_evaluation.md): 4-arm head ablation의
  PDMS·EPDMS·auxiliary mAP 결과와 BEVFusion teacher 비교

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
| LiDAR 입력 | merged point cloud, 전방 ROI clip, frame당 최대 65,536점 |
| LiDAR encoder | SafeDrive `SpMiddleResNetFHD` (spconv, 0.08 m voxel, stride 8); `pillar` fallback |
| sensor history | `[2, 3]`: 과거 1 + 현재 1 frame (카메라·LiDAR 동일) |
| BEV grid | `50 × 100`, 0.64 m 정사각 셀 (BEVFusion teacher와 동일) |
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
전처리하고, 각 카메라의 `lidar2img`로 BEVFormer에 투영한다. 카메라 배치 입력은
`[B, 2, 3, 3, 416, 768]` (`batch, time, camera, RGB, height, width`)이며,
나머지 5개 카메라는 로드하지 않는다. 카메라 임베딩은 `[3, 256]`이다.

LiDAR는 같은 두 frame의 merged point cloud를 전방 ROI와 `lidar_z_range`로 잘라
SSR 축으로 회전한 뒤 `lidar_max_points`행으로 zero-pad한 `lidar_points`
`[B, 2, 65536, 5]`와 실제 점 수 `lidar_num_points` `[B, 2]`로 들어간다. 모델은
이를 SafeDrive의 `SpMiddleResNetFHD`(spconv)로 `[256, 50, 100]` LiDAR BEV로 만들어
SafeDrive와 같이 BEV query 초기값으로 쓰고, encoder 각 층의 gated `lidar_cross_attn`이
읽는다 ([`12_lidar_bev_encoder_50x100.md`](12_lidar_bev_encoder_50x100.md)). spconv는
`pip install spconv-cu126==2.3.8`으로 설치한다 (RTX 5090 / torch 2.8+cu128 검증).
`lidar_encoder: pillar`는 spconv 없는 대체 encoder, `use_lidar: false`는 camera-only
arm이며 각각 feature cache 이름이 분리된다.

세 task는 하나의 전방 ROI를 공유한다. SSR 좌표로 shared BEV, detection/motion,
vector map 모두 `x_right ∈ [-32, 32] m`, `y_forward ∈ [0, 32] m`이다. BEV grid는
`bev_h=50`(전방 32 m) × `bev_w=100`(측방 64 m)로 셀이 0.64 m 정사각형이며,
BEVFusion teacher의 50×100 cache(`teacher_cache/*_50x100`)와 셀 단위로 대응한다
(측방 축 flip만 필요).
detection/motion GT는 이 사각형에 더해 전방축 기준 **±80°** box-center FOV를
적용하며, 학습 target과 auxiliary mAP GT가 같은 필터 함수를 사용한다. HD-map GT에는
각도 필터를 적용하지 않고 공통 사각 ROI만 적용한다.

기존 8-camera checkpoint와는 카메라 임베딩 크기가 다르고, 100×100 checkpoint와는
BEV grid와 LiDAR branch 유무가 다르므로 **새 학습**으로 시작한다.
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

Head ablation 두 가지는 wrapper로 돌린다. head flag를 끄면 target 생성·loss·GradBalancer가
없는 head를 건너뛰고 해당 parameter가 생성되지 않으며(DDP 미사용-parameter 없음), wrapper는
shared-BEV gradient target을 **plan : 남은 task = 1 : 1**로 준다.

```bash
# DET + MOTION + PLAN (map head 없음)          -> work_dirs/para_ssr_det_motion_plan
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" bash scripts/training/train_para_ssr_det_motion_plan.sh

# MAP + PLAN (detection/motion head 없음)      -> work_dirs/para_ssr_map_plan
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" bash scripts/training/train_para_ssr_map_plan.sh

# PLAN only (aux head 둘 다 없음)              -> work_dirs/para_ssr_plan_only
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" bash scripts/training/train_para_ssr_plan_only.sh
```

네 arm은 encoder(3 camera + LiDAR, BEV 50×100)와 recipe가 동일하고 aux head 유무만 다르다.
plan-only는 `grad_balance_target=null`로 GradBalancer를 끈다 — task가 하나뿐이라 조정할
대상이 없다. `gshare/plan`은 계속 기록되며 1.0이면 다른 task가 섞이지 않았다는 확인이 된다.
map target(nuPlan map query)과 detection target이 모두 빠지므로 dataloader가 가장 가볍고,
네 arm 중 가장 빠르다.

세 wrapper는 `train_para_ssr.sh`에 `agent.config.use_map_head=false` /
`use_det_motion_head=false`와 `agent.config.grad_balance_target={plan:0.5,det:0.5}` /
`{plan:0.5,map:0.5}`를 넘길 뿐이라 `EXPERIMENT`, `WANDB=0`, `RESUME_CHECKPOINT`, 추가 Hydra
override를 그대로 쓸 수 있다. Hydra의 dict override는 yaml의 `{plan,det,map}`에 **병합**되어
key를 지우지 못하므로, 꺼진 head의 valve는 loss 모듈이 자동으로 제거한다.

평가는 checkpoint에 없는 head를 agent가 만들지 않도록 **같은 flag를 넘긴다** (strict load).
출력 디렉터리는 `EVAL_EXPERIMENT`로 arm마다 분리한다:

```bash
EVAL_EXPERIMENT=eval/para_ssr_det_motion_plan \
  bash scripts/evaluation/eval_para_ssr.sh /abs/ckpt agent.config.use_map_head=false
EVAL_EXPERIMENT=eval/para_ssr_map_plan \
  bash scripts/evaluation/eval_para_ssr.sh /abs/ckpt agent.config.use_det_motion_head=false
EVAL_EXPERIMENT=eval/para_ssr_plan_only \
  bash scripts/evaluation/eval_para_ssr.sh /abs/ckpt \
  agent.config.use_det_motion_head=false agent.config.use_map_head=false
```

head를 남긴 채 objective만 끄는 대안(`agent.config.task_loss_weight.map=0`, det는
`task_loss_weight.det=0 task_loss_weight.motion=0`)도 동작한다. parameter 수·clip-norm
분모·target 생성이 full 모델과 같아지는 대신, 쓰지 않는 head의 연산과 (map의 경우) nuPlan map
query가 그대로 남아 dataloading이 느리다. 또 auxiliary mAP 평가기는 checkpoint에 존재하는
head를 모두 채점하므로, 이 방식에서는 학습되지 않은 head의 의미 없는 mAP가 함께 나온다.
보고서의 ablation 결과는 head flag 방식으로 학습했다.

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
work_dirs/<EXPERIMENT>/train_time.json
```

매 epoch checkpoint와 resume용 `last.ckpt`를 저장한다. 학습 wall-clock은 TensorBoard/W&B의
`time/elapsed_hours`(step), `time/eta_hours`(step), `time/epoch_hours`, `time/val_hours`로
기록되고, 세션이 끝나면(정상 종료·중단 모두) `train_time.json`에 세션별 시작/종료 시각,
소요 시간, epoch·step 범위가 append된다. resume하면 세션이 추가되므로 `total_hours`가 전체
학습 시간이다. `run_training.log`에도 epoch마다 소요 시간과 ETA가 남는다. TensorBoard 확인:

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
EVAL_EXPERIMENT=eval/my_model_pdm \
bash scripts/evaluation/eval_para_ssr.sh \
  /absolute/path/to/model.ckpt
```

- 입력 checkpoint는 Lightning `state_dict`가 들어 있는 `.ckpt`여야 한다.
- `navtest` 12,146개 token과 `data/exp/metric_cache`(`METRIC_CACHE_PATH`로 변경)를 사용한다.
  cache가 없으면 먼저 만든다. Ray 없이 thread pool로 동작하며 `WORKERS` 기본값은 6이다.

  ```bash
  bash scripts/evaluation/cache_metric_navtest.sh
  ```

- navtest log/sensor는 기본적으로 `data/dataset/{navsim_logs,sensor_blobs}/test`에서 읽는다.
  test split을 다른 곳에 풀어두었다면 `NAVSIM_DOWNLOAD=/path/to/download`(하위에
  `test_navsim_logs/test`, `test_sensor_blobs/test`)를 지정한다. PDM 평가, metric cache,
  auxiliary 평가 wrapper가 모두 이 변수를 경로 override로 바꾼다.
- 이 runner는 **단일 GPU 순차 평가**다. 여러 GPU ID를 주어도 분산되지 않으므로
  `EVAL_GPU`에 GPU 하나만 지정한다.
- 모델마다 고유한 `EVAL_EXPERIMENT`(기본 `eval/para_ssr_front3`)를 사용한다. wrapper가
  `experiment_name`을 직접 설정하므로 `experiment_name=`을 인자로 다시 넘기면 Hydra가
  중복 override로 거부한다.
- 기본 입력은 전방 3개 카메라다. 8-camera checkpoint의 평가에는 학습 때의
  `agent.config.camera_names` 순서와 나머지 architecture/ROI 설정을 함께 복원한다.
- checkpoint를 만든 agent config가 현재 config와 다르면 필요한 `agent.config.*` override를
  함께 전달해야 한다. Parameter key/shape는 strict load로 검사하지만 모든 비-tensor 설정을
  checkpoint가 보존하는 것은 아니다.

짧은 pipeline smoke만 필요하면 다음처럼 제한할 수 있다. `max_scenes` subset은 filesystem
순서에 의존하므로 성능 비교에는 사용하지 않는다.

```bash
CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
EVAL_EXPERIMENT=eval/my_model_pdm_smoke \
bash scripts/evaluation/eval_para_ssr.sh \
  /absolute/path/to/model.ckpt \
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

### 4.3 NAVSIM v2 EPDMS (navtest)

EPDMS(NC·DAC·DDC·TLC 곱, EP·TTC·LK·HC·EC 가중합)는 NAVSIM v2 devkit으로만 채점한다.
이 저장소는 NAVSIM v1 fork이고 v2도 같은 `navsim` package 이름을 쓰므로, v2는 **별도
checkout**에서 `PYTHONPATH`로 격리해 실행한다. v2의 one-stage runner는 CPU worker 안에서
agent를 만들기 때문에 spconv LiDAR encoder(CUDA 전용)를 쓸 수 없어서, 두 단계로 나눈다.

1. **이 저장소, GPU**: PDM 평가와 같은 checkpoint·feature builder·보관된 학습 config로
   navtest trajectory를 저장한다. batch 추론은 PDM 경로와 최대 2.5e-3 m 차이가 나므로
   `--batch-size 1`을 쓴다.

   ```bash
   CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
   python tools/dump_navtest_trajectories.py --arm ssr --batch-size 1 --workers 4
   # -> work_dirs/eval/<experiment>_navtest_trajectories.pkl
   #    --arm: ssr | nodet | nomap | plan_only (4-arm ablation 실험 디렉터리에 대응)
   ```

2. **NAVSIM v2 checkout, CPU**: v2 metric cache를 만든 뒤, 저장한 trajectory를 수정하지 않은
   v2 simulator·scorer·two-frame extended comfort 집계로 채점한다. 사용한 v2 쪽 스크립트
   (`run_pdm_score_from_trajectories.py`, `run_epdms_from_trajectories.sh` 등)와 검증 결과는
   [`13_head_ablation_evaluation.md`](13_head_ablation_evaluation.md) §3에 정리했다.

`navhard_two_stage`는 synthetic scene에 LiDAR가 제공되지 않아 LiDAR 입력 모델로는 평가할 수
없다(같은 문서 §3.4).

---

## 5. Detection/map auxiliary 평가

PARA-SSR의 parallel perception head를 별도로 평가한다.

| metric | 평가 내용 | 평가하지 않는 것 |
|---|---|---|
| `NAVSIMAuxDet/center_mAP` | 7-class 2-D center AP @ 0.5/1/2/4 m | size, yaw, velocity, TP error, NDS |
| `NAVSIMAuxMap/chamfer_mAP` | road/walkway/centerline/crosswalk Chamfer AP @ 0.5/1/1.5 m | polygon 면적, topology, raster IoU |

둘 다 **비공식 auxiliary metric**이며 동일 evaluator protocol의 모델 상대 비교용이다.
Protocol V3의 GT는 shared 전방 ROI 안의 것만 쓰고, detection GT에는 학습 target과 같은
±80° box-center FOV 필터를 적용한다.

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

Head ablation checkpoint도 같은 명령을 쓴다. 실행기는 `AUX_TRAINING_CONFIG`의
`use_det_motion_head`/`use_map_head`로 채점할 task를 정하고 결과의 `metrics/tasks`에 기록한다.
head가 하나만 있는 arm은 그 task만 채점하고, 둘 다 없는 plan-only arm은 거부한다.
navtest 데이터 경로는 §4.1과 같이 `NAVSIM_DOWNLOAD`로 바꿀 수 있다.

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
`num_tokens == 12146`이며 `metrics/protocol_version == 3`, 그리고 `metrics/tasks`에 적힌
task의 결과만 들어 있다. `records/`는 resume 용이므로 Git에 push하지 않는다.

현재 기본 모델(3 camera + LiDAR, BEV 50×100, 30 epoch)의 reference 수치는 다음과 같다.
Checkpoint와 대용량 record가 Git repository에 포함된다는 뜻은 아니며, evaluator 재현 확인용
기준값이다. Ablation arm을 포함한 전체 결과는
[`13_head_ablation_evaluation.md`](13_head_ablation_evaluation.md)에 있다.

| 평가 | 결과 |
|---|---:|
| NAVSIM PDM score (v1) | 0.8556 |
| NAVSIM EPDMS (v2, navtest) | 0.8589 |
| Detection center mAP (V3) | 0.5938 |
| Vector-map Chamfer mAP (V3) | 0.3585 |

이전 8-camera·전후방 ROI 모델의 V2 수치(PDM 0.846130, detection 0.362148, map 0.228846)는
ROI·GT 모집단·입력이 달라 위 값과 비교하지 않는다.

---

## 6. 코드 및 문서 검증

```bash
bash -n scripts/training/smoke_para_ssr.sh
bash -n scripts/training/train_para_ssr.sh
bash -n scripts/evaluation/eval_para_ssr.sh
bash -n scripts/evaluation/eval_para_ssr_aux.sh
bash -n scripts/evaluation/cache_metric_navtest.sh

python -m pytest -q tests/
```

`tests/`에는 target·feature·설정 검증, model invariant, auxiliary metric/runner 테스트가
포함된다. 실제 scene에서 확인한 LiDAR·카메라 좌표와 스케일, loss 감소 검증 결과는
[`12_lidar_bev_encoder_50x100.md`](12_lidar_bev_encoder_50x100.md) §5에 정리했다.

관련 상세 문서:

- [`09_navsim_port_progress.md`](09_navsim_port_progress.md): 포팅·학습·PDM/aux 전체 실행 기록
- [`12_lidar_bev_encoder_50x100.md`](12_lidar_bev_encoder_50x100.md): LiDAR branch·50×100 BEV 설계와 검증
- [`13_head_ablation_evaluation.md`](13_head_ablation_evaluation.md): head ablation PDMS/EPDMS/aux 결과
- [`10_para_ssr_navsim_vector_mapping_reference.md`](10_para_ssr_navsim_vector_mapping_reference.md):
  vector-map GT 좌표·task·loss·평가 protocol
- [`../docs/PARA_SSR_NAVSIM.md`](../docs/PARA_SSR_NAVSIM.md): 환경 설치와 짧은 사용법
