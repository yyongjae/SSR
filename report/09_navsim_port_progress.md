# 보고서 #09 — PARA-SSR의 navsim 이식

작성일 2026-08-25 · 대상: `SSR@para-navsim` 워크트리 (`/home/yongjae/e2e/SSR-para-navsim`)

> 작업 단계마다 갱신한다. 최신 상태는 §2 체크리스트를 본다.

---

## 0. 지금 상태 한 줄 요약

**돌아간다.** 실제 navsim 데이터로 scene 로딩 → feature → target → forward → loss →
backward → optimizer step 전 구간과 2-GPU DDP가 검증됐다. **2 GPU × 4/GPU ×
accumulate 16 = global batch 128**로 실행한다. 원본의 Q×M motion decoder까지 복원한 뒤
기본 8-camera/BEV 100×100/batch 4 단일-GPU 스모크도 통과했다. 현재 구조의
checkpoint 로딩 → GPU 추론 → PDM scorer도 1-scene 스모크를 통과했다.

```bash
./scripts/training/smoke_para_ssr.sh      # 1 train batch + 1 val batch
./scripts/training/train_para_ssr.sh      # 2-GPU 본 학습
```

---

## 1. 왜 "이식"이 아니라 "재작성"인가

| | `ssr` env (현재) | navsim 요구사항 |
|---|---|---|
| python | 3.8.20 | 3.9 |
| torch | 1.9.1+cu111 | 2.0.1+cu118 |
| numpy | 1.19.5 | 1.23.4 |
| 프레임워크 | mmcv 1.4.0 / mmdet 2.14.0 / mmdet3d 0.17.1 | pytorch-lightning 2.2.1 + nuplan-devkit |

한 env에 공존 불가. mmcv 1.4.0은 torch 1.9용 CUDA 확장이라 torch 2.0에서 깨지고,
numpy 1.19에서는 nuplan-devkit이 돌지 않는다.

결정적 확인: **참조 구현이 전부 mmcv-free다.** WoTE/navsim 0건, SeerDrive/navsim 0건,
GTRS/navsim 2건(주석 수준). 주신 레시피의 `use_coslr_opt`, `opt_paramwise_cfg.image_encoder`,
`WarmupCosLR(warmup_epochs=3)`는 정확히 WoTE agent 구조이므로 레시피 작성자도
navsim devkit fork 안에 새로 구현한 것이 맞다.

따라서 `projects/mmdet3d_plugin/SSR/` 약 4,000줄이 재작성 대상이었다.

---

## 2. 진행 체크리스트

### ✅ 완료 — 전 항목 검증됨

| # | 항목 | 산출물 | 검증 방법 |
|---|---|---|---|
| 0 | 워크트리 | `SSR-para-navsim` @ `para-navsim` | 메인 `SSR`은 `para` 유지 확인 |
| 1 | 데이터 링크 | `data -> /data/navsim` | 실제 train intersection 85,109 샘플 확인 |
| 2 | devkit | `navsim/` (WoTE fork, 105 py) | — |
| 3 | 의존성 분리 | `requirements_navsim.txt` | SSR 원본 `requirements.txt` 복원 |
| 4 | deformable attn | `modules/ms_deform_attn.py` | fwd/bwd, 32/32 param grad |
| 5 | transformer bricks | `modules/transformer_blocks.py` | decoder 출력 shape |
| 6 | BEVFormer encoder | `modules/bevformer.py` | temporal 유/무 finite, bwd OK |
| 7 | TokenLearner | `modules/tokenlearner.py` | `[B,16,512]` |
| 8 | GradBalancer | `modules/grad_balance.py` | DDP 평균·결합 det/motion gradient·checkpoint 상태 검증 |
| 9 | 손실/매칭 | `modules/losses.py` | focal/L1/ptsL1/dirCos + Hungarian |
| 10 | planner head | `modules/planner_head.py` | `ego_fut_preds [B,4,8,3]` |
| 11 | det/motion head | `modules/det_motion_head.py` | 8 loss 유한, bev grad 도달 |
| 12 | vector map head | `modules/map_head.py` | 9 loss 유한, bev grad 도달 |
| 13 | feature builder | `para_ssr_features.py` | 실데이터 7개 텐서 생성 |
| 14 | target builder | `para_ssr_targets.py` | 실데이터 map 3클래스 전부 생성 |
| 15 | model 통합 | `para_ssr_model.py` | 38,417,195 param, fan-out 확인 |
| 16 | loss 통합 | `para_ssr_loss.py` | gnorm/gshare 로깅 동작 |
| 17 | agent | `para_ssr_agent.py` | AdamW + WarmupCosLR + lr_mult |
| 18 | hydra config | `agent/para_ssr_agent.yaml` | 오버라이드 가능 노브 노출 |
| 19 | 실행 스크립트 | `scripts/{training,evaluation}/` | smoke 스크립트 통과 |
| 20 | E2E 검증 | — | 67 pytest + tiny/full GPU + 2-GPU DDP train/val + 1-scene PDM 통과 |
| 21 | 평가 cache | `data/exp/metric_cache` | 12,146 token, `MetricCache` 역직렬화 호환 확인 |
| 22 | 전용 conda env | `ssr-navsim` | clean 생성, 의존성·CUDA·1/2-GPU smoke 검증 완료 (§15) |
| 23 | lr/effective batch 확정 | `train_para_ssr.sh` | 2 GPU × 4/GPU × accumulate 16 = 128, lr 1e-4 |

### ⬜ 남은 작업 (사용자 결정 또는 실행 필요)

| # | 항목 | 왜 남았나 |
|---|---|---|
| 24 | **grad_balance 재측정/활성화 결정** | 결합 det+motion valve와 원본 head 복원 뒤 종전 share 수치는 폐기해야 함 (§6) |
| 25 | 본 학습 + 최종 PDM score | 파이프라인 스모크는 통과. 학습된 checkpoint의 전체 navtest 평가는 본 학습 후 실행 |
| 26 | **map divider semantic 확정** | nuPlan lane edge 전부를 VAD divider로 보는 현재 근사는 별도 검증 필요 |
| 27 | photometric augmentation | 원본 train-only distortion을 NAVSIM의 공용 train/val builder에 안전하게 분리 이식할지 결정 필요 |

---

## 3. 이식하면서 바뀐 설계

### 3.1 좌표계

SSR의 `pc_range = [-15,-30,-2, 15,30,2]`는 VAD 관례로 **x=측방(+우), y=종방(+전방)**.
navsim/nuPlan lidar는 **x=전방, y=좌측**. -90° 회전을 `lidar2img`에 접어 넣어
모델 내부는 전부 SSR 좌표계를 쓴다.

```
T_lidar<-ssr = [[0,1,0,0], [-1,0,0,0], [0,0,1,0], [0,0,0,1]]
lidar2img    = K_scaled @ [R|t]_{cam<-lidar} @ T_lidar<-ssr
```

GT 박스도 같은 변환을 받는다. NAVSIM
`[x_fwd,y_left,z,L,W,H,heading]`에서 SSR physical box
`[x_right,y_fwd,z,W,L,H,SECOND_yaw,vx_right,vy_fwd]`로 바꾼다:

```
sx=-y, sy=x, geometric_r=heading+π/2
SECOND_yaw=-geometric_r-π/2=-heading-π
svx=-vy, svy=vx
```

즉 기하학적 heading과 detector가 회귀하는 SECOND yaw를 혼동하면 안 된다.

### 3.2 CAN bus 제거

nuScenes 경로는 18차원 CAN bus에서 `can_bus[0],[1],[-2]`로 BEV shift를 역산한다.
navsim `AgentInput`은 과거 ego pose를 **이미 현재 프레임 상대좌표로** 준다
(`convert_absolute_to_relative_se2_array`). 따라서 직접 계산한다:

```
이전 pose가 현재 프레임에서 (px,py,ph)일 때
d = (0,0) - (px,py) = (-px,-py)      # (전방, 좌측)
shift_x(SSR 우측) = +py / grid_x / bev_w
shift_y(SSR 전방) = -px / grid_y / bev_h
```

query conditioning 경로(`can_bus_mlp`)는 `ego_motion_mlp`로 유지했다(18차원).

`rotate_prev_bev`는 원본 config대로 **False**다. 프레임 간 ego 회전을 prev_bev에
반영하지 않는다는 뜻이고, 원본의 선택을 그대로 따랐다.

### 3.3 플래닝 출력

| | nuScenes SSR | navsim PARA-SSR |
|---|---|---|
| horizon | 6 × 0.5s = 3s | **8 × 0.5s = 4s** |
| 출력 차원 | 2 (x,y) | **3 (x,y,heading)** — navsim `Trajectory` 요구 |
| command 분기 | 3 | **4** (`[left, straight, right, unknown]`) |

heading은 라디안이고 x/y는 미터라 `heading_weight=0.5`로 별도 가중한다.
실데이터 확인: `driving_command: [0 1 0 0]`.

### 3.4 vector map GT — 만들 수 있다 (초기 보고 정정)

처음에 "navsim에 vector map GT가 없다"고 한 것은 **틀렸다.** nuplan map API에 다 있고,
navsim이 target builder를 제공하지 않을 뿐이다. transfuser도 같은 API에서 뽑은
geometry를 래스터화만 해서 쓴다(`transfuser_features.py:197-270`).

단, **`SemanticMapLayer.BOUNDARIES`는 쓸 수 없다.** `get_proximal_map_objects`가
지원하는 레이어는 LANE, LANE_CONNECTOR, ROADBLOCK, ROADBLOCK_CONNECTOR, STOP_LINE,
CROSSWALK, INTERSECTION, WALKWAYS, CARPARK_AREA뿐이다(`nuplan_map.py:50-58`).
lane boundary는 lane 객체를 통해 접근해야 한다.

| VAD 클래스 | 소스 | 형태 |
|---|---|---|
| divider | `Lane`/`LaneConnector`의 `left_boundary`, `right_boundary`.linestring | open |
| ped_crossing | `CROSSWALK` polygon exterior | closed |
| boundary | `ROADBLOCK` polygon exterior | closed |

**실측한 클래스 불균형과 그 처리** — 20개 씬 기준:

| 레이어 | min | median | max |
|---|---:|---:|---:|
| LANE | 9 | 14 | 28 |
| LANE_CONNECTOR | 7 | 20 | 43 |
| CROSSWALK | 0 | 1 | 5 |
| ROADBLOCK | 3 | 4 | 10 |

divider는 (14+20)×2 ≈ 68개, crosswalk는 1개다. 처음 구현은 클래스 순서대로 slot을
채워서 **divider만 GT에 들어가고 ped_crossing/boundary가 한 개도 안 들어갔다**
(`map labels present: [0]`). 클래스 라운드로빈으로 고쳤고, 지금은 `[0, 1, 2]` 전부 들어간다.

**등가 순서(equivalent orders)** — VAD는 같은 geometry를 나타내는 모든 점 순서 중
가장 싼 것으로 매칭한다. closed contour는 원본 VAD v2와 같이 **reverse 없이 cyclic
shift만** 만들고, open line은 정/역방향을 쓴다. clipping으로 닫힌 contour가 잘리면
실제 clipped piece를 open으로 판정한다. 남는 slot은 결정론적으로 반복해 batch shape를
고정한다.

### 3.5 detection / motion

- 클래스: nuScenes 10 → nuPlan 7 (vehicle, pedestrian, bicycle, traffic_cone,
  barrier, czone_sign, generic_object)
- motion GT: `track_tokens`를 미래 프레임과 매칭해 각 미래 박스 중심을 **현재 ego 프레임**으로
  옮긴 뒤 step offset으로 만든다. 실측 `gt_fut_masks` 평균 0.77 — 즉 agent의 77%가 8 step
  전 구간 추적된다.
- `max_agents`: 초기 60은 20개 씬 중 7개에서 잘렸다(in-range agent median 53, max 80).
  **100으로 올렸다** (>100은 0/20).
- 박스 GT는 위 SECOND code로 한 번만 encode하며, decoder prediction은 이미
  `[x,y,logW,logL,z,logH,sin,cos,vx,vy]`이므로 다시 encode하지 않는다.
- motion은 원본처럼 detection query 300 × mode 6 = 1,800개의 mode token과 detached
  normalized box-center PE를 쓴다. 포팅 초안의 Q-token+대형 MLP 및 det/map별 2.56M
  BEV table은 원본과 다른 모델이어서 제거했다.

### 3.6 fp32 강제

`multi_scale_deformable_attn_pytorch` 주변을 `autocast(enabled=False)`로 감쌌다.
레시피 표에서 유일하게 참조 구현(16-mixed)과 다른 항목이고, 근거는 원본과 같다 —
`num_points × num_levels` bilinear tap의 다항 누적이 fp16에서 불안정하다.

### 3.7 아직 의식적으로 남긴 차이

- 원본 encoder는 `shift_ref_2d = ref_2d; shift_ref_2d += shift` alias 때문에 current
  reference까지 이동하는 알려진 구현 버그가 있다. 포트는 `clone()`으로 history reference만
  shift하는 올바른 BEVFormer 의미를 유지한다. exact bug-for-bug 재현은 아니다.
- 원본 train pipeline의 photometric distortion은 아직 없다. NAVSIM agent가 train/val에 같은
  feature builder를 쓰므로 그대로 넣으면 validation까지 랜덤 변형된다. split-aware builder로
  분리하기 전에는 넣지 않는다.
- planning weighted-L1는 full-tensor mean을 보존하지만 branch/dimension이
  `(3,2) -> (4,3)`이 되어 같은 commanded x/y 오차의 평균 계수가 원본 대비 0.5배다.
  `task_loss_weight.plan=2.0`으로 이 **x/y 계수만** 복원했다. 새 heading 항과 6→8 step
  horizon까지 포함한 전체 objective가 원본과 같아졌다는 뜻은 아니다.
- nuPlan lane boundary에는 road outline/virtual edge도 섞일 수 있다. 현재는 중복 ID를 제거한
  모든 lane edge를 divider로 쓰며, `boundary_type_fid` 의미를 검증하기 전까지 근사로 표기한다.

---

## 4. 레시피 반영 결과

| 항목 | 레시피 | 구현 | 비고 |
|---|---|---|---|
| optimizer | AdamW | ✅ | |
| lr | 1e-4 | ✅ 기본값 | §5.2 참조 |
| weight_decay | 1e-4 | ✅ | |
| scheduler | WarmupCosLR, warmup 3ep, min_lr 1e-6 | ✅ | |
| epochs | 30 | ✅ | |
| backbone lr_mult | 0.1 | ✅ `image_encoder` 파라미터 그룹 | |
| grad clip | 35.0 norm | ✅ trainer 설정 | |
| **global batch** | **128 (WoTE 기준)** | **4/GPU × 2 GPU × accumulate 16 = 128** | **§5.1** |
| dataset cache | 없음(온라인) | ✅ | |
| val 주기 | 5 epoch | ✅ | |
| precision | fp32 | ✅ | |

---

## 5. 실행 설정과 메모리 (A6000 48GB × 2)

### 5.1 2-GPU 채택값

최종 원본-parity 모델(8 cam, queue 2, 416×768, BEV 100×100, encoder 3층,
Q×M motion token 1,800개)은 **38,417,195 parameters**다. A6000 한 장에서
batch 4의 실제 scene → target → forward → loss → backward → optimizer → validation
스모크가 통과했다. 최종 temporal-attention 수정 후에도 축소 모델과 실제 NAVSIM
8개 샘플로 **2-GPU NCCL/DDP train/backward/optimizer/validation** 전 경로가 exit 0으로
통과했다.

초안에서 기록했던 26.3/26.8 GiB와 4.2 samples/s는 Q-token motion MLP와 원본에 없는
BEV tables를 쓰던 **수정 전 다른 모델의 측정값**이므로 최종 모델 수치로 재사용하지 않는다.
원본 Q×M self-attention의 score tensor만 B4 fp32에서 약 0.39 GiB이며 backward 저장량이
추가된다. 따라서 검증한 B4보다 batch를 높이지 않는다.

**채택한 해법**: GPU당 batch 4 + gradient accumulation 16.

```
4 (per GPU) × 2 (GPU) × 16 (accumulate) = 128
```

accumulation은 optimizer update 주기만 바꾸며 sample당 forward/backward 계산량은 같다.
다만 최종 Q×M 모델의 steady-state throughput은 아직 측정하지 않았다. 폐기된 pre-parity
모델의 `4.2 samples/s/GPU`로 epoch 시간이나 30-epoch ETA를 계산하지 않는다. 본 학습 전
최종 설정(2 GPU, B4/GPU, queue 2, 8 camera, BEV 100×100, encoder 3층)으로 warm-up 뒤
최소 50~100 microbatch를 측정해야 한다.

### 5.2 lr — global 128로 정합 완료

WoTE는 **16/GPU × 8 GPU = global 128**에서 lr 1e-4를 사용한다. 이 포트는 메모리상
16/GPU를 그대로 쓸 수 없으므로 **4/GPU × 2 GPU × accumulate 16 = global 128**로
동일한 effective batch를 재현한다. 따라서 기본 lr은 **1e-4로 확정**했고, 더 이상
`7.07e-5` 보정 대상이 아니다. `LR` 환경변수는 실험용 override로만 남긴다.

---

## 6. 실측: shared BEV gradient 분배 — grad_balance가 필요하다

초안 모델의 실제 navsim 배치에서 기록했던 `gshare/*`는 다음과 같았다:

| task | share |
|---|---:|
| det | **98.2%** |
| map | 1.4% |
| plan | **0.37%** |
| (motion) | 0.17% (det에 포함) |

하지만 이 측정은 detection과 motion의 norm을 따로 잰 뒤 같은 valve처럼 취급한 결함이
있었고, head도 최종 원본 구조가 아니었다. 현재 코드는 실제 valve와 동일하게
`grad(L_det + L_motion)`을 측정하고 controller state/iteration/scale을 checkpoint에 저장한다.
따라서 위 숫자는 역사 기록일 뿐 **활성화 근거로 쓰지 않는다**. 최종 모델에서 먼저
unscaled share를 재측정한 뒤 아래 설정 여부를 정한다:

```bash
# 문법 예시일 뿐이며, 아래 비율은 추천값이 아님
./scripts/training/train_para_ssr.sh \
  agent.config.grad_balance_target='{plan:0.4,det:0.3,map:0.3}'
```

---

## 7. 알아둘 동작

- **`val/traj_loss`는 planning loss만이다.** `test_aux_heads=False`라 eval 모드에서
  aux head가 꺼지고, 그래서 `train/traj_loss`(전체 합)와 직접 비교하면 안 된다.
  이건 의도된 동작이고 오히려 감시하기 좋은 지표다.
  둘을 같은 단위로 보고 싶으면 `agent.config.test_aux_heads=true`.
- **로그 경로**: `compute_loss`는 스칼라를 반환한다. `AgentLightningModule`이 dict를
  받으면 값을 전부 더해버리는데, 세부 항목에는 total도 들어있어 이중 계산이 된다.
  세부 breakdown은 `ParaSSRLoggingCallback`이 별도로 `train/*`, `val/*`에 올린다.
- 파라미터 수: 기본 설정 **38,417,195 total / 38,141,675 trainable**.
- 수정 전 checkpoint는 motion branch shape와 제거된 BEV table이 달라 호환되지 않는다.
  현재 구조는 fresh training을 기준으로 한다. 평가 로더도 키 누락/초과를
  `strict=True`로 검증하여 부분 로드된 랜덤 head로 PDM score가 실행되지 않게 했다.

---

## 8. 환경

- 초기 스모크/벤치마크는 `/data1/sungoh/envs/wote`를 읽기 전용으로 빌려 수행했다.
- 이후 사용자 소유 clean 환경 `/home/yongjae/miniconda3/envs/ssr-navsim`을 생성했고,
  의존성·CUDA·pytest·1/2-GPU smoke를 모두 재검증했다 (§15).
- 재현 절차는 `docs/PARA_SSR_NAVSIM.md`. 원본 nuScenes용 `ssr` 환경은 변경하지 않았다.
- 출력: `work_dirs -> /data1/yongjae/SSR/navsim` (여유 4.1T)

---

## 9. 저장소 상태 메모

- `.gitignore`의 `mmdetection3d/`, `data/`는 트레일링 슬래시 때문에 **심볼릭 링크를
  무시하지 못한다.** 추적 파일을 건드리지 않으려고
  `/home/yongjae/e2e/SSR/.git/info/exclude`(워크트리 공용, 로컬 전용)에
  슬래시 없이 `mmdetection3d`, `data`를 추가했다.
- `projects/`(mmdet3d SSR 원본)는 **삭제하지 않았다.** 이식 원본이자 참조.
- `navsim/agents/WoTE/`도 남겨뒀다. 레시피 출처라 대조용.

---

## 10. 2026-08-25 코드 검수 결과와 확정 수정 내역

이 절은 초안을 원본 SSR, NAVSIM/WoTE/SeerDrive, 실제 `/data/navsim`
샘플과 대조하며 발견한 문제를 정리한다. 단순한 튜닝 선택은 자동으로 바꾸지 않았고,
좌표/타깃 오염, 원본 구조 불일치, 배치/DDP 오류, 평가 오염처럼 결과를 확실히
잘못되게 만드는 항목은 즉시 수정했다.

### 10.1 GT convention 검수

| 항목 | 초안의 위험 | 확정한 처리 | 검증 |
|---|---|---|---|
| detection 중심/크기 | NAVSIM의 `(x_fwd,y_left,L,W)`를 SSR 축에 그대로 넣으면 위치와 장·단축이 바뀜 | `(x_right,y_fwd,W,L)=(-y,x,W,L)`로 변환 | 단위 테스트 + 실제 박스 range 프로브 |
| detection yaw | NAVSIM 기하 heading과 구형 mmdet3d `LiDARInstance3DBoxes`/SECOND yaw는 동일하지 않음 | `SECOND_yaw=-heading-π`로 변환하고 `[-π,π]` wrap. box `z`는 center를 유지 | cardinal heading/size/velocity 변환 테스트 |
| velocity | NAVSIM `(v_fwd,v_left)`를 그대로 회귀하면 detector 축과 불일치 | `(v_right,v_fwd)=(-v_left,v_fwd)` | 좌표 단위 테스트 |
| motion GT | 미래 frame의 local box를 현재 frame처럼 쓰면 ego motion이 agent motion에 섞임 | future-local → global → current-ego → SSR 축 순서로 변환한 후 step offset 생성 | 이동/회전 ego 합성 테스트 + 실데이터 mask 프로브 |
| planning GT | SSR 내부 BEV 축을 NAVSIM 출력에도 적용하면 PDM convention과 불일치 | planning은 NAVSIM의 `(x_fwd,y_left,heading)` 8×0.5s를 유지하고, model은 step offset을 회귀한 후 cumsum해 `Trajectory` 반환 | target/prediction shape·cumsum·4-command 테스트 + PDM 1 scene |
| vector map | 클래스 순서대로 100 slot을 채우면 divider가 모두 소진하고 crosswalk/boundary가 사라짐 | class round-robin, roadblock+connector polygon union의 exterior/interior, clip 후 open/closed 재판정 | Singapore/Boston/Vegas/Pittsburgh 실제 map probe, 3 class 존재 확인 |

박스는 head 입력에서 physical 9D를 canonical 10D
`[x,y,logW,logL,z,logH,sin(yaw),cos(yaw),vx,vy]`로 **한 번만** encode한다.
decoder 출력은 이미 canonical 10D이므로 재-encode하지 않도록 고쳤다.

### 10.2 모델/어텐션 parity 수정

| 발견 문제 | 영향 | 수정 |
|---|---|---|
| temporal queue의 `B>1` packing 순서 오류 | sample과 history/current BEV가 섞여 다른 장면을 attention | batch/query/queue 순서를 원본과 같게 재배열하고 B=2 exact test 추가 |
| temporal attention softmax에 queue 축까지 포함 | queue 2일 때 각 branch의 규모가 잘못 줄어들고, 후단 mean과 중복 평균 | `[B,Q,H,queue,levels×points]`로 reshape해 마지막 축에만 softmax. constant-value scale 회귀 테스트 추가 |
| det/map에 원본에 없는 전체 BEV positional table | head당 약 2.56M parameter 증가, 다른 model을 학습 | table 제거, 원본 decoder query/reference 경로 복원 |
| motion이 Q-token + 대형 MLP로 단순화됨 | 원본 SSR의 query×mode 상호작용과 checkpoint shape 소실 | 300 query × 6 mode = 1,800 token, q-major/mode-minor, detached normalized box-center PE, per-mode branch로 복원 |
| 일괄 Xavier가 deformable-attention 특수 init을 덮음 | sampling offset/attention weight의 초기 기하 prior 소실 | decoder Xavier 후 deformable module의 radial/specialized init 재적용 |
| backbone recipe/freeze 차이 | timm A1 weight, 움직이는 BN affine/statistics로 원본과 다른 초기화 | `resnet50.tv_in1k`(torchvision 0676ba61), stem+layer1 freeze, 전체 BN eval + affine freeze |
| GridMask/dropout/ego-motion MLP 세부 차이 | 같은 이름이지만 원본과 다른 정규화 | GridMask mode/crop, MHA attn/proj/residual dropout, ego-motion final LayerNorm을 원본 설정에 맞춤 |

최종 기본 모델은 **38,417,195 total / 38,141,675 trainable**이다.

### 10.3 loss, DDP, 학습/평가 경로 수정

- motion classification은 모든 `B×Q` logit을 사용한다. unmatched query는 background,
  matched이지만 future annotation이 전혀 없는 query만 classification weight 0으로 처리했다.
  regression/classification denominator는 원본의 detection-positive 기준을 복원하고 DDP global
  normalization을 맞췄다. empty-GT에서도 background classifier gradient가 생긴다.
- planning heading residual은 `atan2(sin Δθ, cos Δθ)`로 주기 wrap한다. 기존 in-place view
  대입은 diagnostic `autograd.grad`와 실제 backward가 같이 있을 때 version-counter error를
  냈고, out-of-place tensor 구성으로 고쳤다.
- prediction/loss의 NaN/Inf는 head 경계에서 sanitize하고, invalid GT는 matching/loss에
  들어가지 않게 했다.
- GradBalancer는 `nn.Module` extra state로 iteration/scale/seen을 checkpoint에 저장한다.
  validation은 controller clock을 증가시키지 않고, 하나의 valve를 공유하는 det+motion은
  두 loss를 먼저 더한 결합 gradient로 측정한다. 2-GPU에서는 rank별 측정치를 all-reduce해
  동일한 scale을 쓴다.
- config override가 GT convention을 조용히 깨지 않도록 frame index, 4-command,
  8×0.5s trajectory, 3D pose, 7 detection class, 10D code, 3 map class, camera/feature-level
  조합을 agent 생성 시 fail-fast 검사한다.
- training resume은 weight-only initialize가 아니라 Lightning `ckpt_path`를 사용하여
  optimizer/scheduler/epoch/GradBalancer state까지 복원한다.
- 평가 checkpoint는 CPU로 portable load한 후 `agent.` **접두사만** 제거하고
  `strict=True`로 전체 키/shape를 검증한다. 수정 전 checkpoint와 타 모델이 부분
  로드된 랜덤 head로 점수를 만드는 경로를 차단했다.
- feature builder는 resize rounding 후의 실제 `new_w/w`, `new_h/h`로 intrinsic을
  스케일한다. GPU scorer는 model이 있는 실제 device를 따르고, scene/cache token을
  정렬·교집합하며 cache schema를 먼저 역직렬화해 fail-fast한다.

### 10.4 실행 검증 결과

| 검증 | 설정 | 결과 |
|---|---|---|
| unit/integration | `PYTHONPATH=. .../python -m pytest -q` | **67 passed**, 14개 warning은 matplotlib/pyparsing 외부 deprecation |
| 정적 검사 | PARA-SSR Python compile + 전체 shell `bash -n` | exit 0 |
| 1-GPU full-model | 8 camera, queue 2, 416×768, BEV 100×100, encoder 3, batch/GPU 4 | real NAVSIM train/backward/optimizer/validation 통과 |
| 2-GPU DDP 최종 회귀 | A6000 × 2, NCCL, real NAVSIM 8 samples, 축소 BEV/camera, 1 train + 1 val batch | rank 0/1 등록, backward/optimizer/validation, **exit 0** |
| PDM 평가 경로 | 현재 구조의 임시 random checkpoint, navtest 1 scene, GPU | checkpoint strict load → 8-camera inference → `Trajectory` → PDM score, **1 success / 0 failed** |

PDM 스모크의 `0.530026...`는 **random planning head의 파이프라인 테스트 점수**라서
모델 품질 결과로 해석하면 안 된다. 목적은 평가 전용 코드가 현재 checkpoint/schema/출력
convention으로 끝까지 실행되는지 확인하는 것이었다. 결과 CSV는
`work_dirs/smoke_eval_para_ssr/2026.08.25.02.22.15.csv`에 있다.

### 10.5 코드 버그가 아니라 본 학습 전 결정/실측할 항목

1. 2 GPU에서 `4/GPU × accumulate 16 = global 128`로 WoTE 기준을 맞췄고,
   기본 `lr=1e-4`를 사용한다.
2. `task_loss_weight.plan=2.0`을 적용한 최종 head에서 unscaled `gshare/*`를 재측정한
   후에만 GradBalancer target을 정한다. 설정의 `{plan:0.4,det:0.3,map:0.3}`은 문법
   예시이지 추천값이 아니다.
3. nuPlan의 모든 lane edge를 VAD divider로 보는 현재 근사는
   `boundary_type_fid` 시맨틱을 확정하기 전까지 남은 데이터 의미 리스크다.
4. 원본의 photometric distortion은 train/val 공용 builder에 바로 넣지 않는다.
   split-aware augmentation으로 분리한 후 적용해야 validation이 랜덤하게 변하지 않는다.
5. 현재 B4 메모리 통과는 확인했지만 최종 Q×M 모델의 steady-state throughput,
   30-epoch 수렴, 학습된 checkpoint의 전체 navtest PDM score는 실제 장기 run으로만 확정할 수 있다.
6. clean 전용 conda env는 `ssr-navsim` 이름으로 생성·검증 완료했다 (§15).

---

## 11. 2026-08-25 독립 검증 (§10 수정본에 대한 교차 확인)

§10의 수정 내역을 **코드를 믿지 않고 실데이터·합성 프로브로 다시 measure**한 결과다.
테스트가 코드와 같은 가정을 공유하면 둘 다 틀려도 통과하므로, 가능한 항목은 테스트를
거치지 않고 물리량으로 직접 확인했다.

### 11.1 확인된 항목 (§10 주장 타당)

| 검증 | 방법 | 결과 |
|---|---|---|
| temporal attention softmax 축 | 상수 value=5.0, 64×64 맵에서 partition-of-unity | 출력 **정확히 5.0000** |
| temporal attention B>1 packing | batch-major `[prev0,cur0,prev1,cur1]`, 샘플별 상수 1.0/9.0 | **정확히 1.0 / 9.0**, 혼선 없음 |
| history/current 슬롯 분리 | prev만 4.0, cur은 0 | **정확히 2.0** = (4+0)/2 |
| 원본 대조 | `projects/.../temporal_self_attention.py:196-214` | 포트가 **사용자 원본 수정본과 동일**. 초기 포팅이 두 버그를 넣은 것이 맞음 |
| box encode↔decode | 실 GT 79개 round-trip | max abs err **8.9e-16** |
| yaw 물리 정합성 | `SECOND_yaw=3.134` → heading≈0.008 rad, 속도 `vy=+15.25` 전진 | **일치** |
| planning target | 25개 실 씬에서 `cumsum(offsets)` vs `trajectory` | 오차 0 |
| `select_trajectory` | 임의 branch 선택 후 cumsum 비교 | 오차 0 |
| det+motion 결합 valve | `para_ssr_loss.py:207-214` | `grad(L_det+L_motion)`로 올바르게 측정 |
| 파라미터 수 | 기본 config 인스턴스화 | **38,417,195** 정확히 일치 |
| metric_cache | `<log>/unknown/<token>/` 3-depth 집계 | `exp/metric_cache` 12,211 token, **navtest 12,146 100% 커버** |
| 평가 경로 | 랜덤 checkpoint로 navtest 2 씬 실행 | strict 로드 → 8-cam GPU 추론 → PDM → CSV, **2 성공 / 0 실패** |
| 테스트 스위트 | `pytest -q` | **67 passed** (기하 독립 회귀 2개 추가 후) |
| planning loss 0.5배 주장 | `(3×2)/(4×3)` | **0.5 맞음** (mode 3→4, D 2→3; T는 분자·분모에서 상쇄) |
| `_equivalent_orders` closed 처리 | 코드 검토 | 닫힌 contour의 **중복 끝점 roll 버그를 올바르게 회피**. 초기 포팅에 있던 실제 결함 |

### 11.2 §10에 없던 신규 확인 — `lidar2img` 기하 검증

이식 전체에서 가장 위험한 항목인데 §10에 실측 근거가 없었다. 다음처럼 코드와 실데이터를
독립적으로 교차 검증했다.

- `build_lidar2img`의 `R.T`, `-R.T @ t`가 NAVSIM 공식 lidar→camera 식과 대수적으로
  같음을 non-identity 회전·이동으로 확인했다.
- `T_LIDAR_FROM_SSR`는 literal basis로 `SSR forward → lidar forward`,
  `SSR right → lidar -left`를 확인했다. expected에 구현 상수를 재사용하지 않는다.
- 실제 pkl calibration에서 반경 20m, 높이 0m 방위각 링을 0.05° 간격으로 검사해
  8-camera union coverage **100%**를 얻었다. 서로 다른 100개 로그를 0.1° 간격으로
  반복해도 전부 **100%**였다.

아래 표는 한 calibration을 10° 간격으로 요약한 가독성용 표이지, 그 자체가 빈틈 없음의
증명은 아니다.

```
CAM_F0    -20 ..   20     광축 SSR az =   -0.2
CAM_L0    -80 ..  -30     광축 SSR az =  -55.2
CAM_L1   -140 ..  -80     광축 SSR az = -112.2
CAM_L2   -170 .. -110     광축 SSR az = -140.9
CAM_R0     30 ..   80     광축 SSR az =   55.5
CAM_R1     80 ..  140     광축 SSR az =  112.0
CAM_R2    120 ..  170     광축 SSR az =  141.3
CAM_B0   [-180,-150] U [150,170]  광축 SSR az = -179.7
```

CAM_B0는 연속 `-180..170°` 구간이 아니라 ±180° 경계를 감싸는 wrap 구간이다. 같은
calibration의 dense probe에서는 대략 `[-180°, -149.2°] ∪ [148.75°, 179.95°]`였다.
8개 카메라는 서로 겹치면서 360°를 cover한다. **`lidar2img`와 SSR↔navsim 축 변환은
정확하다.** trainval 200개 로그에서 `lidar2ego`가 identity이고, 한 로그 944 frame에서
camera `R/t/K`가 고정인 것도 확인해 현재 calibration 재사용 가정까지 점검했다.

> 주의: GT 박스만으로 이 검증을 하면 오판한다. 실제로 한 씬에서 CAM_F0에 투영된 박스의
> 방위각이 +19°~+30°뿐이었는데, 이는 그 씬에 정면 객체가 없었던 데이터 편향이지 버그가
> 아니다. 합성 프로브가 필요한 이유다.

### 11.3 신규 확인 — `navtrain.yaml`의 `tokens` 리스트가 학습 가능 여부를 결정한다

`sensor_blobs/trainval`은 로그당 전 프레임이 아니라 **연속 구간에만** 존재한다
(예: 500 프레임 중 136개, 인덱스 18-28 / 89-96 / 149-156 ...).

`navtrain.yaml`에는 `log_names` 1,192개 **외에 `tokens` 103,288개**가 있고,
`filter_scenes`가 이 토큰으로 거른다. 이 필터의 유무가 결정적이다:

| SceneFilter 구성 | 정상 scene / 전체 | 정상 비율 | 예상 FileNotFoundError |
|---|---:|---:|---:|
| `log_names`만, frame 3 | 151,778 / 651,526 | 23.296% | 76.704% |
| `log_names`만, frame 2,3 | 138,018 / 651,526 | 21.184% | 78.816% |
| `log_names`만, frame 0,1,2,3 | 110,399 / 651,526 | 16.945% | 83.055% |
| **`tokens` 포함, frame 0~3** | **103,288 / 103,288** | **100%** | **0%** |

- `scene_filter=navtrain`을 hydra로 쓰면 `tokens`가 자동 포함되므로 **현재 스크립트는 안전하다.**
- 반대로 `SceneFilter(...)`를 손으로 만들면서 `tokens`를 빠뜨리면 기본 queue 2의 전체
  로그 기준 약 **78.816%**가 `FileNotFoundError` 후보가 된다. 실제 train split 기준은
  **79.590%**다. 디버그 스크립트를 짤 때의 함정이다.
- 부수 소득: **frame 0~3이 전부 100% 커버**되므로 `frame_indices=(0,1,2,3)`으로
  queue를 4까지 늘려도 데이터 파일 결손은 없다. 메모리·I/O·target 생성·학습시간은
  별도로 검증해야 한다.

`log_names` 1,192개는 모두 `sensor_blobs` 디렉터리가 존재한다. `navsim_logs`에만 있는
118개 pkl은 전부 navtrain 밖이다. Hydra는 `tokens`를 보존하며 실제 split은
train 85,109 + val 18,179 = 103,288이다.

### 11.4 반영 완료 — planning x/y 계수 복원 후 gradient share 측정

두 항목이 §10에서 **각각 따로** 열려 있는데, 실제로는 하나의 결정이다.

- 원본은 `[B,3,T,2]`, NAVSIM 포트는 `[B,4,T,3]` 전체 element mean이라 같은
  commanded x/y 오차의 평균 계수가 **0.5배**가 됐다.
- `task_loss_weight.plan=2.0`은 이 `3×2 → 4×3` 희석을 정확히 보정한다. 다만 새 heading
  항과 horizon 6→8의 의미 차이까지 없애는 **전체 objective parity는 아니다.**

따라서 구현·측정 순서는 다음으로 고정했다:

1. `task_loss_weight.plan=2.0`으로 commanded x/y 계수를 먼저 복원한다.
2. 그 상태에서 unscaled `gshare/*`를 재측정한다.
3. 그 수치를 보고 `grad_balance_target`을 정한다.

초안의 det 98% 수치는 pre-parity head와 잘못된 det/motion 분리 측정에서 나온 값이라
근거로 재사용하지 않는다. `loss_plan_reg`는 호환성을 위해 weight 적용 전 raw loss로
유지하고, 실제 total loss 기여를 바로 볼 수 있도록 `loss_plan_reg_weighted`를 추가했다.
balancer가 켜져도 planner head 자체 gradient와 shared-BEV의 plan numeraire gradient는
이 보정을 받으며, clip 빈도도 다시 관찰해야 한다.

### 11.5 실행 전 정정/결정할 것

1. **GPU 2장 확정**: 사용자 결정대로 기본 `CUDA_VISIBLE_DEVICES=0,1`을 유지한다.
   `4/GPU × 2 × accumulate 16 = global 128`이다. 4-GPU 전환은 현재 작업 범위가 아니며,
   online map/target/I/O 병목 때문에 GPU 수만으로 벽시계 시간이 정확히 절반이 된다고
   가정할 수도 없다.
2. **throughput 미측정**: 폐기된 pre-parity 수치로 ETA를 만들지 않는다. 최종 Q×M 모델의
   2-GPU/B4/queue2/8-camera/BEV100/encoder3 설정에서 warm-up 후 최소 50~100
   microbatch를 재측정한 뒤 30-epoch 일정을 잡는다.
3. **divider semantic** (§10.5-3)에 동의한다. 남은 데이터 의미 리스크 중 가장 크다.

### 11.6 초안(§0~§9)에서 내가 틀렸던 수치 — 정정

| 초안 기술 | 실제 |
|---|---|
| navtrain 47,950 샘플 | 잘못됨. `frame_interval`·`tokens` 누락. navtrain 토큰 103,288, train 교집합 85,109 |
| B=4 26.3 GiB / 4.2 samples/s | pre-parity 모델 측정. 최종 모델에 재사용 불가 (§10.2 지적 타당) |
| "navsim에 vector map GT 없음" | 잘못됨. §3.4에서 이미 정정 |

---

## 12. 2026-08-25 즉시 반영한 수정

독립 검증 결과에서 코드·실행 설정에 바로 반영해야 하는 항목을 다음과 같이 수정했다.

| 항목 | 변경 | 상태 |
|---|---|---|
| effective batch | 2 GPU, B4/GPU, `ACCUMULATE=16` → **global 128** | 학습 스크립트 기본값·경고·주석 반영 |
| learning rate | WoTE global 128과 같으므로 **1e-4 유지** | 7.07e-5 선택지 폐기 |
| planning scale | `task_loss_weight.plan: 1.0 → 2.0` | Python default와 Hydra YAML 모두 반영 |
| planning 로그 | raw `loss_plan_reg` + 실제 기여 `loss_plan_reg_weighted` | 대시보드에서 ×2 가중치 누락 방지 |
| gradient share | plan 2.0 적용 전의 기존 측정 폐기 | 최종 objective에서 재측정 대기 |
| `lidar2img` 회귀 | literal basis 및 non-identity `R/t` 직접 투영 테스트 추가 | 순환 expected 문제 제거 |
| 데이터 필터 문서 | no-token 정상률/FNF를 실제 `filter_scenes` 결과로 교정 | queue 2 train FNF 79.590% 명시 |
| throughput | pre-parity 4.2 samples/s 기반 ETA 제거 | 최종 설정 50~100 microbatch 측정 필요 |

검증 결과:

```text
pytest -q: 67 passed, 14 external deprecation warnings
python -m compileall: exit 0
all scripts/*.sh bash -n: exit 0
Hydra compose: plan=2.0, navtrain tokens=103,288
1-GPU real NAVSIM tiny smoke (plan=2.0): train/backward/optimizer/val, exit 0
```

이번 변경은 설정·손실 가중치·회귀 테스트와 문서 정정이다. 변경 후 tiny GPU smoke를 다시
통과했고, 모델 architecture 자체는 건드리지 않았으므로 앞서 통과한 full-model GPU/DDP/PDM
smoke 결과도 그대로 유효하다. 다만 본 학습 전에 `plan=2.0` 상태의 unscaled `gshare/*`와
최종 throughput은 새로 측정해야 한다.

---

## 13. 2026-08-25 §12 변경 검증 + 미해결 항목 실측

### 13.1 §12 변경 확인 — 전부 타당

| 변경 | 확인 방법 | 결과 |
|---|---|---|
| `ACCUMULATE=16` → global 128 | 4 × 2 × 16, 가드 조건도 128 | ✅ |
| lr=1e-4 확정 | global 128은 **WoTE/SeerDrive와 동일**한 배치 | ✅ √2 보정 문제 자체가 소멸. 가장 깔끔한 해소 |
| `plan=2.0` | `default.py:144` + `yaml:32` 일관 | ✅ |
| plan=2.0 산술 | 원본 `12e/36B=e/3B` vs 포트 `16e/96B=e/6B` = 0.5배 | ✅ 2.0이 정확한 역수 |
| `loss_plan_reg_weighted` | 실배치에서 비율 측정 | ✅ 정확히 **2.00** |
| lidar2img 회귀 테스트 | 구현 재사용 없이 literal convention에서 독립 재유도 | ✅ 제대로 된 회귀 테스트 |
| 방위각 링 프로브 | 재실행 | ✅ CAM_F0 −20..20 / 광축 −0.2°, 8-cam 360° 타일링 유지 |
| 테스트·문법 | `pytest -q`, `bash -n` | ✅ **67 passed**, shell 3개 OK |

> plan=2.0은 **x/y 목적함수 크기**를 복원하는 것이지, heading이라는 새 목적을 없애지 않는다.
> heading은 포트 내부에서 x/y의 0.5배 가중을 유지한다. 주석 문구와 실제 동작이 일치한다.

### 13.2 미해결 #1 해소 — 최종 모델 throughput/메모리 실측

실제 NAVSIM navtrain 배치, A6000 1장, 전체 config(8 cam, queue 2, 416×768, BEV 100×100,
encoder 3층, Q×M motion 1,800 token), 파라미터 38,417,195:

| batch/GPU | peak memory | ms/step | samples/s/GPU |
|---:|---:|---:|---:|
| **4** | **23.3 GiB** | 983 | **4.07** |
| 6 | 33.8 GiB | 1542 | 3.89 |
| 8 | OOM | — | — |

- **B=4가 실제 최적이다.** B=6은 메모리를 45% 더 쓰면서 샘플당 오히려 느리고, B=8은 OOM.
  현재 스크립트 기본값이 맞다.
- Q×M motion decoder 복원 비용은 미미하다 (폐기했던 4.2 samples/s와 4.07은 사실상 동일).
  §5.1에서 "재측정 전까지 쓰지 말라"고 한 판단은 타당했고, 재측정 결과 값은 유지된다.
- **dataloader는 병목이 아니다**: worker 8개로 프로세스당 **6.93 samples/s** (연산 4.07의 1.7배).
  머신은 64 core / 503 GB이므로 4 GPU × 8 worker = 32 worker도 여유가 있다.

**30 epoch 소요 (85,109 samples, 연산 기준)**

| GPU | 분/epoch | 30 epoch |
|---:|---:|---:|
| 2 | 174 | **87.2 시간** |
| 4 | 87 | **43.6 시간** |

### 13.3 미해결 #2 해소 — 최종 모델 unscaled gradient share

§6이 요구한 "최종 head + plan=2.0 상태의 재측정"이다. 실 NAVSIM 배치:

| task | gnorm | gshare |
|---|---:|---:|
| det | 1.45e-2 | **73.5%** |
| map | 5.18e-3 | 26.2% |
| **plan** | **5.0e-5** | **0.25%** |

두 가지를 동시에 말한다.

1. **parity 수정이 실제로 효과가 있었다.** 이식 초안의 det 98.2% / map 1.4%가
   det 73.5% / map 26.2%로 바뀌었다. head별 2.56M BEV table 제거와 decoder 복원의 결과다.
2. **그러나 plan은 여전히 0.25%다.** det 대비 gradient가 **약 290배** 작다.
   `plan=2.0`은 loss 스케일 교정이지 gradient share 교정이 아니므로, 이것만으로는
   §11.4에서 지적한 구조적 불균형이 해결되지 않는다. 보고서 #01이 nuScenes에서
   PARA-SSR 열세의 원인으로 지목한 바로 그 상태다.

### 13.4 GradBalancer 폐루프 실검증

`target={plan:0.4, det:0.3, map:0.3}`, warmup 0 / interval 1로 실 배치 10 step:

```
 it  gshare/plan  gshare/det  gshare/map   scale det   scale map
  0      0.00274     0.75540     0.24186    1.00e+00    1.00e+00
  3      0.27409     0.39407     0.33183    1.34e-03    5.94e-03
  6      0.36446     0.32303     0.31251    1.18e-03    5.35e-03
  9      0.39527     0.27708     0.32765    1.16e-03    5.38e-03
```

- **목표에 수렴한다.** plan 0.27% → 약 40%.
- 정착 scale은 det ≈ 1.2e-3, map ≈ 5.4e-3로 **clamp `(1e-5, 1.0)`에 걸리지 않는다.**
  즉 목표가 도달 가능 범위 안에 있다.
- 운영 설정(warmup 500, interval 200)에서는 500 + 7×200 ≈ 1,900 micro-batch 후 정착한다.
  2 GPU 기준 epoch당 약 10,600 micro-batch이므로 **epoch 1의 18% 지점에서 수렴**한다.

컨트롤러는 동작하고 목표는 도달 가능하다. 켤지 말지는 이제 코드 문제가 아니라 실험 설계 결정이다.

### 13.5 남은 권고

1. **GPU 4장 사용**: 현재 4장 모두 유휴다(`nvidia-smi` 확인). global batch를 128로 유지한 채
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 ACCUMULATE=8 ./scripts/training/train_para_ssr.sh
   ```
   4 × 4 × 8 = 128로 동일하고, 소요는 87시간 → **43.6시간**이 된다. 스크립트의 가드도 그대로 통과한다.
2. **batch는 4를 유지한다.** 실측상 최적이다 (§13.2).
3. `grad_balance_target` 결정은 §13.3의 0.25%와 §13.4의 도달 가능성을 근거로 판단하면 된다.
4. `divider` semantic(모든 lane edge를 VAD divider로 간주)은 여전히 최대의 데이터 의미 리스크다.

---

## 14. 2026-08-25 GradBalancer 기본 활성화 + 최종 코드 검토

### 14.1 기본값 변경

밸런서를 **기본 활성화**했다. 오버라이드 없이 `./scripts/training/train_para_ssr.sh`로 켜진다.

| 항목 | 값 | 근거 |
|---|---|---|
| `grad_balance_target` | `{plan:0.4, det:0.3, map:0.3}` | nuScenes 60ep arm(`PARA_SSR_e2e_60ep.py:114`)과 동일 |
| `grad_balance_warmup_iters` | **10600** | §14.2 |
| `grad_balance_interval` | 200 | 원본과 동일 |
| `momentum` / `clamp` | 0.9 / (1e-5, 1.0) | 원본과 동일 |

`grad_balance_warmup_iters`와 `grad_balance_interval`은 yaml에 없어 `+` 접두사가 필요했다.
실수를 유발하므로 `para_ssr_agent.yaml`에 노출했다.

### 14.2 warmup을 500 → 10600으로 바꾼 이유

**`iteration`은 optimizer step이 아니라 micro-batch를 센다** (`compute_loss` 호출당 1,
`model.training`일 때만). `accumulate_grad_batches=16`이라 단위가 16배 어긋난다.

| | 원본 (mmcv) | 포트 |
|---|---|---|
| iteration 단위 | optimizer step | **micro-batch** |
| warmup 500 | 500 optimizer step | **31 optimizer step** = epoch의 4.7% |
| LR warmup 종료 | 500 iter — 밸런서 시작과 일치 | 3 epoch = **31,916 micro-batch** |

원본은 밸런서가 LR warm-up 종료 시점에 시작하도록 설계돼 있다. 500을 그대로 쓰면 LR이
최고치의 1.6%인 지점에서 **첫 측정을 EMA 없이 그대로 채택**한다. 10600(= 2 GPU × batch 4
기준 1 epoch, LR이 최고치의 2/3)을 절충값으로 택했다. 엄격 재현은 31900이다.

기준 수치: rank당 epoch = 10,639 micro-batch = 665 optimizer step.

### 14.3 실행 검증

| 검증 | 결과 |
|---|---|
| hydra 기본 구성 | `target={plan:0.4,det:0.3,map:0.3} warmup=10600 interval=200`, `task_loss_weight.plan=2.0` |
| `pytest -q` | **67 passed** |
| `bash -n` (shell 3개) | OK |
| 표준 스모크 | train + validation, exit 0 |
| **밸런서 발화 스모크** (Lightning 경로, warmup 0 / interval 1, 10 batch) | `gshare/plan` 0.003 → epoch 평균 **0.293**, `gscale/det` **1.68e-3**, `gscale/map` **1.05e-2** |

독립 측정(§13.4)의 det 1.2e-3 / map 5.4e-3와 같은 자릿수다. 컨트롤러가 Lightning 경로와
`ParaSSRLoggingCallback`을 통해 정상 동작한다.

### 14.4 최종 코드 검토 — 기하/규약 재검증

| 검증 | 방법 | 결과 |
|---|---|---|
| **박스 규약 (축·크기·yaw 결합)** | 실 GT 46개(길쭉한 차량 포함)의 **코너를 양쪽에서 독립 재구성**해 비교 | 최대 오차 **1.1e-06 m** (float32 정밀도). 축 교환·크기 교환(L↔W)·SECOND yaw가 상호 정합 |
| `bev_shift` 부호/크기 | 25개 샘플에서 `v_fwd × 0.5s`와 대조 | 상관 **0.997**, 평균절대차 0.074 m (유한차분 vs 순간속도 차이 수준) |
| `bev_shift` 측방 | `-v_left × 0.5s`와 대조 | 평균절대차 0.037 m. 0.5초 측방 변위가 워낙 작아 상관계수는 의미 없음 |
| `lidar2img` | 방위각 링 프로브 재실행 | CAM_F0 −20..20 / 광축 −0.2°, 8-cam 360° 타일링 |
| **split 누수** | log 목록 교집합 | train/val/test 상호 **disjoint**. navtrain 1192 = train 978 + val 214. **navtest ∩ navtrain = 0** |

### 14.5 발견된 사소한 문제 (전부 무해, 수정하지 않음)

본 학습 직전이라 shape를 바꾸는 변경은 하지 않았다. 전부 성능·정확도에 영향이 없다.

1. **`status_feature`가 만들어지지만 아무도 안 쓴다.** 모델·손실 참조 0회. 내용(속도,
   가속도, command)은 이미 `ego_motion[5:9]`와 `command`에 있다. float 8개라 비용은 무시 가능.
   읽는 사람이 "모델이 ego status를 직접 쓴다"고 오해할 수 있다는 게 유일한 문제다.
2. **`ego_motion_dims=18`인데 인덱스 14~17은 항상 0이다.** 실제로 채우는 건 0~13.
   `ego_motion_mlp` 입력 4차원이 죽어 있다. 무해하지만 의도된 여백은 아니다.
3. **`trajectory`(절대 pose) 타깃을 손실이 쓰지 않는다.** 손실은 `trajectory_offsets`만 쓴다.
   설계상 맞고 디버깅에 유용하므로 남겨도 되지만, 주석이 없어 오해 소지가 있다.
4. **밸런서 진단이 TensorBoard에 `_epoch` 집계로만 남는다.** 콜백이 `on_train_batch_end`에서
   기록하기 때문이다. `gshare/*`는 측정 iteration에만 `latest_logs`에 존재하므로 epoch 평균은
   **측정 batch만 평균**한다 — 값 자체는 올바르다. 다만 step 단위 곡선은 볼 수 없다.

### 14.6 결론

이번 패스에서 **결과를 잘못되게 만드는 문제는 발견되지 않았다.** 기하 규약(lidar2img,
박스 축/크기/yaw, bev_shift), 데이터 split, 밸런서 동작이 모두 실측으로 확인됐다.
남은 리스크는 코드가 아니라 §10.5의 데이터 의미(`divider` semantic)와 학습 결과 자체다.

---

## 15. 2026-08-25 전용 `ssr-navsim` 환경 생성

원본 nuScenes SSR의 `ssr` 환경(Python 3.8 / Torch 1.9 / mmcv)은 그대로 보존하고,
NAVSIM 포트 전용 환경을 다음 경로에 clean 생성했다.

```text
/home/yongjae/miniconda3/envs/ssr-navsim
```

설치된 핵심 버전:

| 항목 | 버전 |
|---|---|
| Python | 3.9.23 |
| PyTorch | 2.0.1+cu118 |
| torchvision | 0.15.2+cu118 |
| NumPy | 1.23.4 |
| Hydra | 1.2.0 |
| PyTorch Lightning | 2.2.1 |
| timm | 1.0.28 |
| nuPlan devkit | 1.2.0 |
| PARA-SSR NAVSIM | editable `para-ssr-navsim==1.0.0` |

환경 이름은 `environment.yml`과 `docs/PARA_SSR_NAVSIM.md`에도 `ssr-navsim`으로 통일했다.
setup 배포명 `para-ssr-navsim`은 conda 환경 이름과 독립이므로 그대로 유지한다.

검증 결과:

```text
pip check: No broken requirements found
pytest -q: 67 passed, 14 external deprecation warnings
compileall: exit 0
all shell bash -n: exit 0
CUDA: torch 2.0.1+cu118, runtime 11.8, 4 x NVIDIA RTX A6000 인식
1-GPU real NAVSIM smoke: train/backward/optimizer/validation, exit 0
2-GPU NCCL/DDP real NAVSIM smoke: rank 0/1 + train/backward/optimizer/validation, exit 0
```

본 학습 실행:

```bash
conda activate ssr-navsim
cd /home/yongjae/e2e/SSR-para-navsim
./scripts/training/train_para_ssr.sh
```

---

## 15. 2026-08-25 W&B 로깅 추가

### 15.1 배경 — 이식에서 빠져 있던 부분

원본 `PARA_SSR_e2e_60ep.py:117-131`은 `TextLoggerHook` + `TensorboardLoggerHook` +
`SSRWandbLoggerHook` 세 갈래로 로깅했다. navsim devkit(WoTE/SeerDrive/GTRS 포함)에는
wandb가 **0건**이고, `pl.Trainer`에 `logger=`를 넘기지 않아 Lightning 기본 TensorBoard만
붙어 있었다. 즉 **navsim 기준으로는 정상, 원본 SSR 기준으로는 누락**이었다.

### 15.2 구현

새 모듈 `navsim/planning/training/wandb_logging.py`. 원본 훅의 설계 의도를 그대로 옮겼다.

| 원칙 | 구현 |
|---|---|
| TensorBoard **대신**이 아니라 **함께** | `build_loggers()`가 항상 `TensorBoardLogger`를 먼저 넣고, 활성화 시 W&B를 추가 |
| **fail-open** — telemetry가 rank를 죽이면 안 됨 | `SafeWandbLogger`가 `log_metrics`/`log_hyperparams`/`finalize` 예외를 잡아 경고 후 프로세스 잔여 구간 비활성화. `non_fatal=false`로 끌 수 있음 |
| 미설치/로그인 실패도 치명적이지 않음 | `build_wandb_logger()`가 `None`을 반환하고 경고만 남김 |
| 태그 제외 | `exclude` 지원. **기본값은 비어 있다** — 원본이 제외하던 `map.loss_map_{bbox,iou}` 6종은 이 포트에 아예 없다(가중치 0으로 두지 않고 제거했음). 메커니즘만 유지 |

`run_training.py`는 `pl.Trainer(..., logger=build_loggers(cfg))` 한 줄만 바뀌었다.
`default_training.yaml`에 `wandb:` 블록(기본 `enable: false`), `requirements_navsim.txt`에
`wandb` 추가. `train_para_ssr.sh`는 **기본 활성화**이며 `WANDB=0`으로 끈다.

```bash
./scripts/training/train_para_ssr.sh                 # W&B on (기본)
WANDB=0 ./scripts/training/train_para_ssr.sh         # W&B off
WANDB_MODE=offline ./scripts/training/train_para_ssr.sh   # 오프라인, 나중에 sync
WANDB_PROJECT=my-proj WANDB_GROUP=ablation ./scripts/training/train_para_ssr.sh
```

스크립트가 붙이는 태그: `[para-ssr, navsim, no-ffp, aux, global<GLOBAL_BATCH>, <EPOCHS>ep]`
(원본의 태그 구성을 따름).

### 15.3 검증

| 검증 | 결과 |
|---|---|
| hydra가 `wandb.tags=[para-ssr,navsim,...]` 파싱 | ✅ 하이픈 포함 리스트 정상 |
| **미설치 상태에서 `wandb.enable=true`** | ✅ 경고 1줄 후 학습 정상 완료. fail-open 동작 확인 |
| TensorBoard 경로 회귀 | ✅ `work_dirs/<exp>/lightning_logs/version_0/` — 패치 전과 **동일** |
| 지표 손실 여부 | ✅ scalar tag 59개, `gshare/*`·`gscale/*` 그대로 기록 |
| `pytest -q` / `bash -n` | ✅ 67 passed / shell 3개 OK |

### 15.4 확인하지 못한 것

**실제 W&B 업로드는 검증하지 못했다.** 검증에 빌려 쓰는 `/data1/sungoh/envs/wote`에
`wandb`가 설치돼 있지 않고, 타인 소유 env라 설치하지 않았다. 따라서 다음은 미검증이다:

- 실제 run 생성 / 지표 업로드
- DDP 2-rank에서 rank-0만 업로드하는지 (Lightning `WandbLogger`가 `@rank_zero_experiment`로
  보장하지만 이 조합은 직접 돌려보지 않았다)

전용 env에서 `pip install wandb && wandb login` 후 짧은 run으로 한 번 확인하는 것을 권한다.
미설치 상태에서도 학습이 죽지 않는 것은 확인했으므로, 최악의 경우에도 TensorBoard는 남는다.
