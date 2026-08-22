# PARA-SSR 구조와 로그 해석 가이드

이 문서는 이 저장소에서 새로 구성한 **PARA-SSR**의 구조, 학습·추론 코드 흐름, 로그 지표의 의미를 정리한 문서다. 대화에서 사용한 “SSR-Para”와 코드의 `ParaSSR`/`PARA_SSR`는 같은 모델을 뜻하며, 이 문서에서는 코드 이름에 맞춰 **PARA-SSR**로 표기한다.

문서의 기준 구현은 다음 파일들이다.

- 기본 모델/config: [`projects/configs/SSR/PARA_SSR_e2e.py`](projects/configs/SSR/PARA_SSR_e2e.py)
- 현재 장기 학습 config: [`projects/configs/SSR/PARA_SSR_e2e_60ep.py`](projects/configs/SSR/PARA_SSR_e2e_60ep.py)
- 전체 모델 흐름: [`projects/mmdet3d_plugin/SSR/para_ssr.py`](projects/mmdet3d_plugin/SSR/para_ssr.py)
- sparse planner: [`projects/mmdet3d_plugin/SSR/para_ssr_head.py`](projects/mmdet3d_plugin/SSR/para_ssr_head.py)
- gradient balancer: [`projects/mmdet3d_plugin/SSR/utils/grad_balance.py`](projects/mmdet3d_plugin/SSR/utils/grad_balance.py)
- 통합 실행기: [`run.sh`](run.sh)

현재 기본 구성은 **planning + detection/motion + vector map**이며, occupancy head는 비활성화되어 있다.

---

## 1. 새롭게 구성한 PARA-SSR의 구조와 코드 흐름

### 1.1 설계 목적

기존 SSR의 navigation-guided sparse planning 경로는 유지하면서, perception 보조 과제를 PARA-Drive 방식으로 **공유 BEV 위에 병렬 배치**했다.

핵심 원칙은 다음과 같다.

1. planning, detection/motion, map은 동일한 `bev_embed`를 읽는다.
2. 한 auxiliary head의 출력이 다른 head나 planner의 입력으로 들어가지 않는다.
3. 과제 사이의 정보 공유는 오직 shared BEV를 공동 학습하는 방식으로 일어난다.
4. auxiliary head는 추론 시 끌 수 있다.
5. auxiliary head 자체의 학습 강도와 shared BEV에 미치는 영향은 `_ScaleGrad`와 `GradBalancer`로 분리한다.

### 1.2 전체 구조

```text
NuScenes multi-view image queue (length=3)
    ├─ history frames ── ResNet-50 + FPN ── BEVFormerEncoder ── prev_bev
    └─ current frame  ── ResNet-50 + FPN ── BEVFormerEncoder(prev_bev)
                                                  │
                                                  ▼
                              bev_embed [B, 100×100, 256]
                                                  │
                 ┌────────────────┬───────────────┴───────────────┐
                 │                │                               │
                 ▼                ▼                               ▼
        Sparse planning     Detection + motion             Vector map
        ParaSSRHead         ParaDetMotionHead              ParaMapHead
        3 commands          300 object queries             100 polylines
        × 6 steps           6 motion modes                 × 20 points
                 │                │                               │
                 └────────────────┴──────── loss only ────────────┘

Optional: ParaOccHead (현재는 config와 pipeline에서 모두 비활성화)
```

중요한 점은 위의 세 head가 **순차 연결이 아니라 fan-out 구조**라는 것이다. 예를 들어 map prediction을 motion decoder가 읽거나 detection 결과를 planner가 읽는 경로는 없다.

### 1.3 기존 SSR에서 제거된 부분

PARA-SSR은 기존 SSR에서 다음 FFP/future world model 경로를 제거했다.

- `latent_world_model`
- `TokenFuser`
- `loss_bev`
- `obtain_next_bev`
- temporal queue의 future frame

따라서 temporal queue는 과거 프레임과 현재 프레임만 포함한다. `queue[-1]`이 실제 current frame이므로 current image와 box/map/planning GT가 같은 시점에 정렬된다.

### 1.4 한 번의 학습 iteration에서 일어나는 일

실제 진입점은 `ParaSSR.forward_train()`이다.

1. 입력 queue를 history와 current로 나눈다.
   - `prev_img = img[:, :-1]`
   - `img = img[:, -1]`
2. history frame은 `obtain_history_bev()`에서 순차 처리한다.
   - `eval()`과 `torch.no_grad()`를 사용하므로 history BEV 생성에는 gradient를 저장하지 않는다.
   - 각 history frame의 BEV가 다음 frame의 `prev_bev`가 된다.
3. current image를 ResNet-50과 FPN으로 처리한다.
4. `ParaSSRHead` 안의 BEVFormer encoder가 current `bev_embed`를 만든다.
5. planner가 같은 forward에서 planning prediction을 만든다.
6. `bev_embed`를 detection/motion head와 map head에 병렬로 전달한다.
7. 각 head의 loss에 `task_loss_weight`를 적용한다.
8. auxiliary head에서 shared BEV로 역전파되는 경로에만 현재 `gscale`을 적용한다.
9. 지정된 주기에는 task별 BEV gradient, encoder gradient, 표현 품질을 추가 측정한다.
10. 모든 활성 loss를 합산해 backward하고, global gradient clipping 후 AdamW step과 EMA update를 수행한다.

### 1.5 Shared BEV encoder

| 항목 | 현재 값 |
|---|---:|
| Image backbone | ResNet-50, stage 4 output |
| Neck | single-level FPN, 256 channels |
| BEV 크기 | `100 × 100` |
| BEV embedding 차원 | `256` |
| 관심 영역 | x `[-15, 15] m`, y `[-30, 30] m` |
| Temporal queue | 3 frames |
| BEVFormer encoder | 3 layers |
| Attention | temporal self-attention + spatial cross-attention |

`pts_bbox_head`라는 이름과 달리, 현재 `ParaSSRHead`는 box head가 아니라 **BEV encoder와 planner를 함께 소유**한다. 로그에서 parameter group을 나눌 때도 `pts_bbox_head.transformer`, `bev_embedding`, `positional_encoding`은 `bev`로, 나머지 planning 모듈은 `plan`으로 분리한다.

### 1.6 Sparse planning 경로

Planning 경로는 다음 순서로 동작한다.

```text
bev_embed
  + navigation command embedding
  └─ SE channel gating (`navi_se`)
       + learned BEV positional encoding
       └─ TokenLearnerV11: 10,000 BEV cells → 16 scene tokens
            └─ 3-layer latent self-attention decoder
                 └─ 18 waypoint queries
                      ├─ right:    6 future offsets
                      ├─ left:     6 future offsets
                      └─ straight: 6 future offsets
                           └─ cross-attention + MLP
                                → [B, 3, 6, 2]
```

각 sample은 one-hot navigation command에 해당하는 branch 하나만 supervision을 받는다. `loss_plan_reg`는 예측·GT의 **0.5초 간격 offset**에 적용하는 L1 loss다. 평가 시에는 offset을 누적합하여 position trajectory로 바꾼 뒤 planning L2와 collision을 계산한다.

### 1.7 Detection + motion head

`ParaDetMotionHead` 한 개가 detection과 motion을 함께 만든다.

| 항목 | 현재 값 |
|---|---:|
| Object query | 300 |
| Detection decoder | 3 layers |
| Detection class | nuScenes 10 classes |
| Motion horizon | 6 steps, 3 seconds |
| Motion mode | 6 modes |
| Motion decoder | 1 layer |

Detection decoder의 각 layer에서 classification과 box regression loss가 발생한다. Motion loss는 마지막 detection layer의 object representation을 사용하며, trajectory regression과 mode classification으로 나뉜다.

Detection과 motion은 loss weight는 별도로 조정할 수 있지만, 동일한 head forward와 동일한 BEV 입력을 사용하므로 shared-BEV gradient valve는 `det` 하나를 공유한다. `gshare/motion`은 detection gradient 안에 포함되는 부분을 따로 관찰하는 진단값이지 독립된 share가 아니다.

### 1.8 Vector map head

현재 map head는 dense semantic map이 아니라 VAD 계열의 vectorized map head다.

| 항목 | 현재 값 |
|---|---:|
| Map class | divider, ped_crossing, boundary |
| Polyline query | 100 instances |
| Points per polyline | 20 |
| Decoder | 3 layers |
| Matching | Hungarian assignment |
| 주요 supervision | class, point L1, direction cosine |

`loss_map_bbox`와 `loss_map_iou`는 코드는 실행되지만 config의 weight가 `0.0`이다. 따라서 이 값들이 항상 0인 것은 정상이며, 현재 map geometry의 실질적인 학습 신호는 `loss_map_pts`와 `loss_map_dir`이다.

### 1.9 Occupancy head의 현재 상태

`ParaOccHead` 구현은 남아 있지만 현재 base config에서 `occ_head=None`이다. train/test pipeline의 `GenerateSSROccLabels`와 collect key도 주석 처리되어 있어 occupancy GT rasterization 역시 현재는 실행되지 않는다.

다시 활성화하려면 다음 항목을 함께 복구해야 한다.

1. `occ_head=_occ_head`
2. `task_loss_weight`의 `occ`
3. `grad_balance.target`의 `occ`
4. train/test pipeline의 `GenerateSSROccLabels`
5. `CustomCollect3D`의 `gt_occ_seg`, `gt_occ_valid`

head만 켜고 GT pipeline을 복구하지 않으면 `forward_train()`에서 명시적으로 실패하도록 되어 있다.

### 1.10 Gradient valve와 GradBalancer

일반적인 loss weight는 다음 두 경로를 동시에 줄인다.

```text
loss ──→ auxiliary head parameters
   └──→ shared BEV
```

PARA-SSR의 `_ScaleGrad`는 forward에서는 identity이고 backward에서만 gradient를 곱한다.

```text
loss ───────────────→ auxiliary head parameters  (full gradient)
   └─ _ScaleGrad ───→ shared BEV                 (scaled gradient)
```

따라서 auxiliary head는 teacher로서 정상적으로 학습하면서, shared BEV를 과도하게 바꾸는 영향만 제한할 수 있다.

현재 60-epoch config의 목표는 다음과 같다.

```python
target = dict(plan=0.4, det=0.3, map=0.3)
interval = 200
momentum = 0.9
clamp = (1e-5, 1.0)
warmup_iters = 500
```

Planning은 기준 task이므로 scale이 항상 1이다. 매 200 iteration마다 현재 gradient norm에서 기존 scale을 역산하고, 목표 비율이 되도록 detection과 map scale을 다시 구한다. DDP에서는 rank별 norm을 평균낸 뒤 모든 rank가 같은 scale을 사용한다.

처음 500 iteration은 모든 scale을 1로 유지하며, `it > 500`이면서 200의 배수인 첫 시점인 iteration 600에서 최초 보정이 적용된다. 첫 측정값은 바로 채택하고, 이후 scale은 momentum 0.9의 geometric EMA로 완만하게 갱신한다.

이 target이 의미하는 것은 정확히 **shared BEV activation에서 측정한 task별 gradient norm의 비율**이다. 다음을 의미하지는 않는다.

- 전체 parameter update의 40%가 planning이라는 뜻
- AdamW step의 40%가 planning이라는 뜻
- gradient 방향과 충돌까지 반영한 영향도가 40%라는 뜻
- auxiliary supervision이 planning에 도움이 된다는 뜻

방향은 `gcos/*`, optimizer가 만든 실제 상대 변화량은 `uwr/*`, 최종 유용성은 동일 조건의 plan-only control과 validation 결과로 판단해야 한다.

### 1.11 추론과 평가 흐름

기본값 `test_aux_heads=False`에서는 planner만 실행한다. `PARA_SSR_e2e_60ep.py`는 학습 중 validation에서 auxiliary metric을 얻기 위해 `test_aux_heads=True`로 덮어쓴다.

| 설정 | 추론 결과 |
|---|---|
| `test_aux_heads=False` | planning trajectory만 생성 |
| `test_aux_heads=True` | planning + detection/motion + map 생성 |
| `occ_head`도 활성 | 위 결과 + occupancy score 생성 |

Video test mode에서는 scene이 이어지는 동안 이전 sample의 BEV를 다음 sample에 전달한다. scene이 바뀌면 `prev_bev`를 초기화한다.

### 1.12 주요 코드 위치

```text
projects/mmdet3d_plugin/SSR/
├── para_ssr.py                    전체 train/test orchestration
├── para_ssr_head.py               BEV encoder + sparse planner
├── tokenlearner.py                10,000 cells → 16 tokens
├── dense_heads/
│   ├── para_det_motion_head.py    detection + multimodal motion
│   ├── para_map_head.py           vector map
│   └── para_occ_head.py           optional occupancy
├── utils/
│   └── grad_balance.py            shared-BEV gradient controller
├── hooks/
│   ├── clip_monitor.py            clipping, pnorm, uwr
│   ├── anomaly.py                 이상 징후 기록
│   ├── ema.py                     EMA checkpoint
│   └── wandb_logger.py            W&B metric filtering
└── planner/metric_stp3.py         planning evaluation
```

### 1.13 Config 계층

| Config | 용도 | 핵심 차이 |
|---|---|---|
| `PARA_SSR_e2e.py` | 공통 base | 모델·dataset·hook 정의, occupancy off |
| `PARA_SSR_e2e_12ep.py` | 12 epoch 단일 학습 | global batch 8, LR `2e-4` |
| `PARA_SSR_e2e_60ep.py` | 현재 주 장기 학습 | 60 epochs, target `0.4/0.3/0.3`, 6 epoch마다 평가 |
| `PARA_SSR_e2e_60ep_planonly.py` | control arm | auxiliary head는 학습하지만 aux→BEV gradient는 0 |
| `PARA_SSR_stage1_detmap.py` | staged stage 1 | 48 epochs, planning off, det/motion/map만 학습 |
| `PARA_SSR_stage2_all.py` | staged stage 2 | stage 1 weight에서 12 epochs, 전 task 활성 |

`PARA_SSR_w_equal.py`, `PARA_SSR_w_gradscale.py`, `PARA_SSR_w_plan.py`는 이전 weight/scale sweep용 config다. 주석 일부가 현재 occupancy-off 구조보다 오래된 가정을 포함하므로 현재 표준 실행에는 위 6개 config를 우선 사용한다.

---

## 2. 로깅 중인 수치의 설명 및 구분

### 2.1 로그가 저장되는 위치

각 run의 `work_dirs/<run-name>/` 아래에 다음 산출물이 생긴다.

- `<timestamp>.log`: 사람이 읽는 text log
- `<timestamp>.log.json`: iteration/epoch별 JSON line log
- TensorBoard event file
- W&B run
- `epoch_N.pth`: raw model checkpoint
- `epoch_N_ema.pth`: EMA checkpoint
- `latest.pth`: 최신 raw checkpoint 링크 또는 사본
- `anomalies.jsonl`: anomaly hook이 감지한 사건

### 2.2 로그 주기와 평균 방식

현재 일반 logger interval은 100 iteration이고, 무거운 진단값은 200 iteration마다 계산한다.

| 종류 | 계산 주기 |
|---|---:|
| loss, time, data_time, grad_norm, clip | 매 iteration |
| `gnorm`, `gshare`, `gcos`, `gscale` | 200 iterations |
| `enc_gnorm`, `enc_gshare`, `enc_gcos` | 200 iterations |
| `pnorm`, `uwr` | 200 iterations |
| `tok_*`, `bev_std`, per-command, map quality | 200 iterations |
| text/JSON/W&B 기록 | 100 iterations |
| 60ep validation | 6 epochs |

MMCV `LogBuffer`는 logger가 호출될 때 history를 평균한다. 따라서 200 iteration마다만 생성되는 `gshare/*` 같은 값은 JSON의 모든 100-step row를 독립 측정값으로 보면 안 된다. 이전 측정값이 반복되거나 epoch 내 누적 평균처럼 보일 수 있다.

정확한 instantaneous `gshare`를 복원할 때는 다음 도구를 사용한다.

```bash
python tools/analyze_gshare.py work_dirs/para_ssr_60ep
```

### 2.3 기본 실행·성능 지표

| Key | 의미 | 해석 |
|---|---|---|
| `epoch`, `iter` | 현재 epoch와 epoch 내부 iteration | 재시작 여부와 진행 위치 확인 |
| `lr` | 현재 learning rate | warmup과 cosine schedule 확인 |
| `time` | logger window의 평균 wall time/iteration | 작을수록 학습 throughput이 높음 |
| `data_time` | batch를 기다린 평균 시간/iteration | `time`에서 DataLoader가 차지하는 병목 판단 |
| `memory` | 현재 process의 peak GPU allocated memory, MB 단위 | OOM 여유 확인 |
| `loss` | 이름에 `loss`가 포함된 활성 loss의 최종 합 | finite 여부와 큰 spike 확인. task 영향도 지표는 아님 |

Global batch가 8로 고정된 현재 실행에서 대략적인 학습 throughput은 `8 / time` samples/s로 비교할 수 있다. Worker 8과 16을 비교할 때는 같은 GPU 수·같은 config에서 초기 warmup row를 제외하고 `time`과 `data_time`의 안정 구간 평균을 비교해야 한다.

`data_time / time`이 이미 작다면 worker를 늘려도 전체 `time`은 거의 줄지 않는다. 반대로 GPU가 batch를 자주 기다리고 `data_time` 변동이 크다면 worker 증가가 유효할 수 있다.

### 2.4 Planning loss와 command별 진단

| Key | 의미 |
|---|---|
| `loss_plan_reg` | 선택된 navigation command branch의 6개 future offset에 대한 L1 loss |
| `plan_err_sum/right` | right sample들의 absolute offset error 합 |
| `plan_err_sum/left` | left sample들의 absolute offset error 합 |
| `plan_err_sum/straight` | straight sample들의 absolute offset error 합 |
| `plan_n/right`, `left`, `straight` | 위 합의 valid element 수 |

Command별 평균 offset error는 같은 window에서 다음처럼 계산한다.

```text
command MAE = sum(plan_err_sum/command) / sum(plan_n/command)
```

`plan_n/left=0`일 때 `plan_err_sum/left=0`인 것은 완벽한 예측이라는 뜻이 아니라 해당 window에 left supervision이 없었다는 뜻이다. Training split은 straight 비율이 매우 높으므로 total planning loss만 보면 turn branch 문제를 놓칠 수 있다.

또한 `loss_plan_reg`/command MAE는 **step offset error**이고, validation의 `plan_L2_*`는 offset 누적 후의 **position displacement error**다. 숫자의 크기를 직접 비교하면 안 된다.

### 2.5 Detection과 motion loss

| Key pattern | 의미 |
|---|---|
| `det.d0.loss_cls`, `det.d1.loss_cls` | detection decoder 중간 layer classification loss |
| `det.loss_cls` | 마지막 detection layer classification loss |
| `det.d0.loss_bbox`, `det.d1.loss_bbox` | 중간 layer box regression loss |
| `det.loss_bbox` | 마지막 layer box regression loss |
| `motion.loss_traj` | matched agent의 multimodal trajectory regression loss |
| `motion.loss_traj_cls` | trajectory mode classification loss |

중간 decoder loss도 `loss`라는 이름을 가지므로 total optimized loss에 포함된다. Detection과 motion loss의 크기가 planning보다 크다고 해서 shared BEV를 그 비율만큼 지배한다는 뜻은 아니다. shared BEV 영향은 `gshare`, head parameter update는 `uwr`로 확인한다.

### 2.6 Map loss

각 key는 final layer와 `d0`, `d1` 중간 decoder layer에 존재한다.

| Key suffix | 의미 | 현재 상태 |
|---|---|---|
| `loss_map_cls` | map class focal loss | 활성 |
| `loss_map_pts` | polyline point L1 loss | 활성, 주 geometry loss |
| `loss_map_dir` | 인접 point 방향 cosine loss | 활성 |
| `loss_map_bbox` | polyline bounding box L1 loss | weight `0.0`, 0이 정상 |
| `loss_map_iou` | polyline bounding box GIoU loss | weight `0.0`, 0이 정상 |

예: `map.d0.loss_map_pts`, `map.d1.loss_map_pts`, `map.loss_map_pts`.

W&B에서는 `loss_map_bbox`와 `loss_map_iou` 6개 key를 noise로 보고 숨긴다. Text log와 TensorBoard에는 남아 있으며, 이 필터는 학습 objective를 바꾸지 않는다.

### 2.7 Occupancy loss와 진단값

현재는 occupancy가 꺼져 있어 아래 key가 나타나지 않는 것이 정상이다. Head를 다시 활성화했을 때 사용한다.

| Key | 의미 |
|---|---|
| `occ.loss_occ_mask` | top-k binary segmentation loss |
| `occ.loss_occ_dice` | Dice loss |
| `occ_iou0.1/cls*`, `occ_iou0.5/cls*` | 학습 batch에서 threshold별 class IoU |
| `occ_iou*/mean` | class 평균 IoU |
| `occ_sep/p_on` | GT positive cell의 평균 예측 확률 |
| `occ_sep/p_off` | GT negative cell의 평균 예측 확률 |
| `occ_sep/ratio` | `p_on / p_off`; 1에 가까우면 공간 구분을 못 함 |
| `occ_sep/valid_frac` | 진단에 포함된 valid future-frame 비율 |

### 2.8 Global gradient clipping 지표

현재 optimizer는 모든 parameter를 합친 L2 norm에 `max_norm=35`를 적용한다.

| Key | 의미 | 해석 |
|---|---|---|
| `grad_norm` | clipping 전 전체 parameter gradient norm | 35 초과 시 clip 발동 |
| `clip/rate` | logger window에서 clip이 발동한 iteration 비율 | 0은 미발동, 1은 매 step 발동 |
| `clip/factor` | 실제 적용된 평균 `min(1, 35/grad_norm)` | 1이면 변화 없음, 작을수록 강한 축소 |

Global clipping은 auxiliary head만 줄이는 것이 아니라 planner를 포함한 모든 parameter gradient를 같은 비율로 줄인다. 따라서 aux→BEV `gscale`을 작게 해도 auxiliary head 자체의 parameter gradient가 크면 `grad_norm`과 clipping은 여전히 클 수 있다.

### 2.9 Parameter group gradient와 update ratio

200 iteration마다 parameter를 다음 그룹으로 나눈다.

| 그룹 | 포함 모듈 |
|---|---|
| `trunk` | image backbone + FPN |
| `bev` | BEV transformer + BEV embedding + positional encoding |
| `plan` | `ParaSSRHead`에서 BEV 부분을 제외한 planner |
| `aux` | detection/motion + map + optional occupancy |
| `other` | 위 prefix에 매칭되지 않은 parameter |

`pnorm/*`는 clipping 전 각 그룹의 parameter-gradient L2 norm이다.

- `pnorm/total² ≈ trunk² + bev² + plan² + aux² + other²`여야 한다.
- `pnorm/total`은 같은 시점의 `grad_norm`과 일치해야 한다.
- `pnorm/other`는 현재 구조에서 0이어야 한다. 0이 아니면 분류되지 않은 새 parameter가 있다는 뜻이다.

`uwr/*`는 optimizer step 전후의 상대 parameter 변화량이다.

```text
uwr/group = ||ΔW_group|| / ||W_group||
```

코드에 적힌 경험적 읽기 기준은 다음과 같다.

- 약 `1e-3`: 일반적인 건강 구간
- `1e-5` 미만: 사실상 거의 움직이지 않음
- `1e-2` 초과: 불안정할 만큼 큰 update 가능성

이 값은 절대 판정 기준이라기보다 동일 run의 그룹 간 비교와 시간 추세에 사용한다. AdamW에서는 gradient norm과 update 크기가 정비례하지 않는다.

### 2.10 Shared BEV task gradient 지표

| Key | 의미 |
|---|---|
| `gnorm/plan`, `det`, `map` | 현재 valve를 통과한 `||dL_task / d bev_embed||` |
| `gshare/plan`, `det`, `map` | 위 norm을 활성 task 합으로 정규화한 비율 |
| `gnorm_raw/*` | 현재 `gscale`을 역산한 valve 이전 norm |
| `gscale/det`, `gscale/map` | GradBalancer가 현재 적용하는 backward scale |
| `gcos/plan-det` 등 | 두 task의 BEV gradient cosine similarity |

60ep run에서는 정상적으로 `gshare/plan + gshare/det + gshare/map ≈ 1`이다. Controller가 안정화된 뒤 목표인 `0.4/0.3/0.3` 근처인지 확인한다.

`motion`은 예외다.

- `gnorm/motion`은 detection/motion head loss 중 motion 부분만 분리한 값이다.
- `gshare/motion`은 `det` 안에 중복 포함되므로 전체 합에 더하지 않는다.
- 확률 share처럼 해석하지 말고 detection gradient 구성 진단으로만 사용한다.

`gnorm_raw`는 scale이 0보다 클 때만 역산 의미가 있다. Plan-only config처럼 `gscale=0`이면 BEV에 도달한 gradient 자체가 0이므로 로그에서 원래 unscaled norm을 복원할 수 없다.

### 2.11 Gradient 방향: `gcos/*`

Cosine similarity는 magnitude가 아니라 방향을 나타낸다.

| 범위 | 의미 |
|---|---|
| `+1`에 가까움 | 두 task가 shared BEV를 비슷한 방향으로 업데이트하려 함 |
| `0` 근처 | 거의 직교, 서로 다른 표현 축을 사용 |
| `-1`에 가까움 | 서로 반대 방향, gradient conflict 가능성 |

예를 들어 `gshare/plan=0.4`, `gshare/map=0.3`이어도 `gcos/plan-map<0`이면 두 gradient가 실제 update에서 일부 상쇄될 수 있다. GradBalancer는 norm만 맞추며 conflict를 해결하지 않는다.

### 2.12 Encoder parameter 공간의 gradient

| Key | 의미 |
|---|---|
| `enc_gnorm/*` | task별 `||dL_task / dW_BEV_encoder||` |
| `enc_gshare/*` | encoder parameter gradient norm share |
| `enc_gcos/*` | encoder parameter 공간에서 task 간 cosine |

`gshare`는 BEV **출력 feature**를 누가 얼마나 미는지 보고, `enc_gshare`는 실제 BEV encoder **weight**를 누가 얼마나 바꾸려 하는지 본다. 동일한 BEV gradient norm이라도 공간적으로 집중된 gradient와 넓게 분산된 gradient는 encoder parameter에 다른 영향을 줄 수 있어 두 수치는 다를 수 있다.

실제 encoder update conflict를 볼 때는 `enc_gcos`가 더 직접적이지만 계산 비용도 더 크다.

### 2.13 Shared representation 품질

| Key | 의미 | 위험 신호 |
|---|---|---|
| `tok_sim` | 16 scene token 사이 off-diagonal 평균 cosine | 1에 가까우면 token들이 복사본처럼 collapse |
| `bev_std` | channel별 BEV spatial std의 평균 | 0에 가까우면 위치별 표현이 거의 동일 |
| `tok_cover` | token 하나의 attention entropy가 나타내는 effective BEV coverage 비율 | 1에 가까우면 모든 token이 global average처럼 될 수 있음 |
| `tok_union` | 적어도 한 token이 uniform보다 높은 attention을 준 BEV cell 비율 | 너무 작으면 scene의 일부만 사용 |

`tok_cover`의 이론적 범위는 대략 `1/10000`에서 `1`이다. 한 cell만 선택해도 좋다는 뜻은 아니고, 1이 나쁘다는 뜻도 항상 아니다. `tok_sim`, `tok_union`, planning 성능의 추세를 함께 봐야 한다.

Stage 1은 planning loss가 꺼져 있으므로 planner와 TokenLearner 관련 지표를 성능 판단에 사용하면 안 된다. `tok_sim≈1`이어도 stage 1의 목적상 이상으로 단정할 수 없다.

### 2.14 Map head의 학습 품질 지표

| Key | 의미 | 해석 |
|---|---|---|
| `map_cls_acc` | Hungarian matched query의 class accuracy | 높을수록 좋지만 이것만으로 FP를 판단할 수 없음 |
| `map_conf_pos` | matched query의 평균 최대 confidence | 학습되며 올라가는지 확인 |
| `map_conf_neg` | unmatched query의 평균 최대 confidence | 낮아지는 것이 바람직 |
| `map_spread_m` | query별 polyline center의 spatial std, metre | 0에 가까우면 모든 query가 같은 선으로 collapse |
| `map_pts_err_m` | matched point의 평균 Euclidean error, metre | 낮을수록 좋음 |
| `map_n_gt` | image당 matched GT instance 수 | 모델 지표가 아니라 data/pipeline sanity check |

`map_conf_pos - map_conf_neg`의 gap을 함께 본다. Anomaly hook은 epoch 2부터 gap이 `0.02` 미만이면 discrimination 부족을 경고하고, `map_spread_m < 0.5 m`이면 query collapse를 critical event로 기록한다.

`map_pts_err_m`는 point index가 정렬된 error이며 chamfer mAP와 같은 수치가 아니다. 학습 중 추세 파악에는 유용하지만 최종 map 품질은 validation의 `NuscMap_chamfer/*`로 판단한다.

### 2.15 Anomaly 지표

| Key | 의미 |
|---|---|
| `anomaly/new` | 이번 iteration에서 새로 기록된 anomaly 수 |
| `anomaly/total` | run 시작 이후 누적 anomaly 수 |

상세 내용은 `work_dirs/<run>/anomalies.jsonl`에 저장된다. 현재 hook이 검사하는 주요 항목은 다음과 같다.

- loss 또는 gradient의 NaN/Inf
- 최근 200개 median의 5배를 넘는 total loss spike
- 3 epochs 동안 상대 변화가 0.5% 미만인 loss stall
- map query collapse
- map positive/negative confidence 미분리
- 모든 iteration에서 clipping 발생
- `gscale`의 clamp saturation
- 이전 validation 대비 2%를 넘는 regression
- 두 번의 평가 이후에도 map/detection mAP가 정확히 0

Anomaly hook은 경고만 기록하며 학습을 중단하지 않는다. 또한 모든 가능한 문제를 감시하지 않는다. 특히 현재 `tok_sim` collapse는 자동 anomaly 조건에 포함되지 않으므로 직접 봐야 한다.

### 2.16 Validation 지표

#### Planning

| Key | 의미 | 방향 |
|---|---|---|
| `plan_L2_1s`, `2s`, `3s` | 각 horizon까지의 trajectory L2 | 낮을수록 좋음 |
| `plan_L2_avg` | 1/2/3초 값의 평균 | 낮을수록 좋음 |
| `plan_L2_stp3_*` | ST-P3 방식 L2 | 낮을수록 좋음 |
| `plan_obj_col_*` | ego point collision rate | 낮을수록 좋음 |
| `plan_obj_box_col_*` | ego footprint box collision rate | 낮을수록 좋음 |
| `*_stp3_*` | 해당 collision의 ST-P3 집계 방식 | 낮을수록 좋음 |
| `plan_num_valid` | 유효한 3초 future를 가진 validation sample 수 | dataset sanity check |

`plan_L2`와 `plan_L2_stp3`는 서로 다른 집계 규칙이므로 같은 이름의 변형끼리 비교한다.

#### Detection

| Key | 의미 | 방향 |
|---|---|---|
| `pts_bbox_NuScenes/mAP` | nuScenes detection mAP | 높을수록 좋음 |
| `pts_bbox_NuScenes/NDS` | nuScenes Detection Score | 높을수록 좋음 |
| `pts_bbox_NuScenes/<class>_AP_dist_*` | class·거리 threshold별 AP | 높을수록 좋음 |
| `pts_bbox_NuScenes/mATE`, `mASE`, `mAOE`, `mAVE`, `mAAE` | nuScenes true-positive error | 낮을수록 좋음 |

현재 dataset의 기본 detection score floor는 `0.0`이다. 다른 실험이나 논문이 `0.2` floor를 사용했다면 같은 protocol인지 확인해야 한다.

#### Motion

| Key | 의미 | 방향 |
|---|---|---|
| `motion_EPA/car`, `motion_EPA/pedestrian` | false positive penalty를 포함한 EPA | 높을수록 좋음 |
| `motion_ADE/*` | matched agent의 평균 displacement error | 낮을수록 좋음 |
| `motion_FDE/*` | 최종 displacement error | 낮을수록 좋음 |
| `motion_MR/*` | miss rate | 낮을수록 좋음 |
| `motion_cnt_<class>/*` | GT, matched, hit, FP raw count | metric 신뢰도와 NaN 원인 확인 |

Motion 평가는 detection score `0.6`을 넘는 prediction만 사용한다. Match가 하나도 없으면 ADE/FDE를 0으로 속이지 않고 NaN으로 반환한다.

#### Vector map

| Key | 의미 | 방향 |
|---|---|---|
| `NuscMap_chamfer/mAP` | 0.5/1.0/1.5 m threshold와 3 class의 평균 AP | 높을수록 좋음 |
| `NuscMap_chamfer/<class>_AP` | class별 threshold 평균 AP | 높을수록 좋음 |
| `NuscMap_chamfer/<class>_AP_thr_*` | class·Chamfer threshold별 AP | 높을수록 좋음 |

#### Occupancy, 활성화한 경우

| Key | 의미 | 방향 |
|---|---|---|
| `occ_iou_thr0.1/mean`, `0.3`, `0.5` | 전체 validation의 threshold별 IoU | 높을수록 좋음 |
| `occ_valid_frame_frac` | scene 끝을 넘지 않은 future frame 비율 | dataset sanity check |

---

## 3. 추가로 알아야 할 내용

### 3.1 현재 60-epoch 표준 실험 설정

| 항목 | 값 |
|---|---:|
| Optimizer | AdamW |
| Base LR | `2e-4` |
| Backbone LR multiplier | `0.1` |
| Weight decay | `0.01` |
| LR schedule | CosineAnnealing |
| Warmup | 500 iterations, linear, ratio `1/3` |
| Global batch | 8 |
| Gradient clip | L2 max norm 35 |
| Epochs | 60 |
| Validation | 6 epochs마다 |
| Raw checkpoint | 매 epoch |
| EMA checkpoint | 매 epoch, decay `0.9990` |
| Active heads | planning, detection/motion, map |
| BEV gradient target | plan 0.4, det 0.3, map 0.3 |

### 3.2 GPU 수, batch, worker 수

`run.sh`은 training GPU 수에 따라 per-GPU batch를 자동 조정하여 global batch 8을 유지한다.

| GPU 수 | samples/GPU | Global batch | workers/GPU 기본값 | 총 DataLoader worker |
|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 8 | 8 |
| 2 | 4 | 8 | 8 | 16 |
| 4 | 2 | 8 | 8 | 32 |
| 8 | 1 | 8 | 8 | 64 |

현재 4-GPU 실행은 자동으로 `4 × batch 2 = global batch 8`, `4 × worker 8 = 총 worker 32`가 된다.

CPU가 64 logical core라고 해서 한 training job에 worker 64가 자동으로 최적은 아니다. `workers_per_gpu`는 DDP process마다 생성되고, image decode·augmentation library가 내부 thread를 추가로 사용할 수 있으며, 여러 job이 동시에 돌면 CPU·RAM·storage bandwidth를 공유한다. Worker는 모델의 실험 정의가 아니라 throughput knob이므로 8과 16의 steady-state `time`/`data_time`을 직접 비교해 결정한다.

다른 worker 수는 다음처럼 지정한다.

```bash
SSR_WORKERS_PER_GPU=16 ./run.sh 60ep 0,1,2,3 \
  projects/configs/SSR/PARA_SSR_e2e_60ep.py
```

GPU 수가 달라도 global batch는 이미 고정되므로 `--autoscale-lr`를 추가하지 않는다.

### 3.3 실행 명령

```bash
# 12 epochs
./run.sh 12ep 0,1 projects/configs/SSR/PARA_SSR_e2e_12ep.py

# 현재 60-epoch 표준 run
./run.sh 60ep 0,1,2,3 projects/configs/SSR/PARA_SSR_e2e_60ep.py

# plan-only control
./run.sh planonly 0,1 projects/configs/SSR/PARA_SSR_e2e_60ep_planonly.py

# 48ep perception-first + 12ep all-task
./run.sh staged 0,1 \
  projects/configs/SSR/PARA_SSR_stage1_detmap.py \
  projects/configs/SSR/PARA_SSR_stage2_all.py

# validation 경로 smoke test
./run.sh smoke 0,1 projects/configs/SSR/PARA_SSR_e2e_60ep.py

# 200-iteration 속도 측정
./run.sh calib 0,1,2,3 projects/configs/SSR/PARA_SSR_e2e_60ep.py
```

Config path는 실행 명령의 인자로 전달할 수 있으므로 GPU 이름이 들어간 별도 config를 복제할 필요가 없다.

### 3.4 Validation 전에 확인할 것

Vector map 평가는 prediction과 GT cache의 sample token을 대조한다. Cache가 없으면 만들거나 evaluator가 생성하도록 해야 한다.

```bash
python tools/build_map_gt_cache.py \
  projects/configs/SSR/PARA_SSR_e2e_60ep.py
```

장기 run 전에 `smoke`로 distributed result ordering, map cache, auxiliary output, evaluator 경로를 확인한다.

### 3.5 Raw checkpoint와 EMA checkpoint

학습 중 EvalHook은 **raw model**을 평가한다. `MEGVIIEMAHook`은 별도 `epoch_N_ema.pth`를 저장하지만 EvalHook에 자동으로 연결되지 않는다.

따라서 학습 중 6-epoch validation은 추세 확인용이고, 최종 보고 수치는 EMA checkpoint를 1 GPU로 다시 평가하는 것이 기준이다.

```bash
./run.sh eval \
  work_dirs/para_ssr_60ep/epoch_60_ema.pth \
  0 \
  projects/configs/SSR/PARA_SSR_e2e_60ep.py
```

Distributed validation은 scene 중간에서 시작하는 rank의 첫 sample이 `prev_bev`를 잃을 수 있어 sequential 1-GPU 평가와 아주 조금 다를 수 있다.

`--resume-from`은 raw model과 optimizer 상태를 복원하지만 현재 EMA hook의 별도 EMA 상태까지 자동 복원하지 않는다. 완전한 EMA continuation이 필요하면 hook의 `resume`도 해당 `epoch_N_ema.pth`로 지정해야 한다.

### 3.6 Plan-only control의 정확한 의미

`PARA_SSR_e2e_60ep_planonly.py`는 auxiliary head를 제거하지 않는다. Detection/motion과 map head는 자신의 loss로 계속 학습하지만 `_ScaleGrad(scale=0)` 때문에 shared BEV를 바꾸지 못한다.

이 방식은 다음을 동일하게 유지한다.

- parameter 수
- auxiliary head forward/backward 비용
- optimizer가 보는 auxiliary parameter gradient
- global clipping에 auxiliary head가 주는 영향

따라서 full PARA-SSR과 비교할 때 주로 바뀌는 것은 **auxiliary supervision이 shared BEV를 공동 학습하는가**다. Auxiliary head가 아예 없는 기존 SSR과 비교하는 것보다 더 직접적인 control이다.

### 3.7 Staged 학습을 해석할 때의 주의점

Stage 1과 2를 합치면 60 epochs지만, 단일 60-epoch run과 “staging 여부 하나만 다른 통제 실험”은 아니다.

- LR cosine이 48ep와 12ep 두 번 시작된다.
- Stage 2는 optimizer moment를 새로 시작한다.
- EMA와 RNG stream도 새로 시작한다.
- Planner는 staged recipe에서 12 epochs만 학습한다.
- Stage 2는 perception weight를 불러오지만 optimizer continuation이 아니라 `load_from`이다.

따라서 두 결과는 같은 compute budget의 서로 다른 **training recipe**로 비교해야 한다.

현재 `_scale(weight=0)`은 해당 loss dict를 아예 제거하므로 stage 1의 planner parameter에는 zero tensor gradient조차 생성되지 않는다. AdamW도 `grad is None`인 parameter는 건너뛰므로 planner가 weight decay로 조용히 줄어드는 경로는 현재 구현에는 없다.

### 3.8 Loss 크기와 task 영향도를 혼동하지 않기

다음 값들은 서로 다른 질문에 답한다.

| 질문 | 봐야 할 지표 |
|---|---|
| Objective 값이 내려가는가? | task별 loss |
| Shared BEV feature를 누가 크게 미는가? | `gshare/*` |
| Task gradient 방향이 충돌하는가? | `gcos/*`, `enc_gcos/*` |
| BEV encoder weight를 누가 크게 미는가? | `enc_gshare/*` |
| 어느 module의 parameter gradient가 clipping을 지배하는가? | `pnorm/*` |
| Optimizer가 실제로 parameter를 얼마나 움직였는가? | `uwr/*` |
| Head가 의미 있는 출력을 학습했는가? | map quality, validation mAP/EPA |
| Planner에 최종적으로 도움이 되었는가? | 동일 조건 control 대비 planning validation |

큰 loss가 반드시 큰 gradient를 만들지 않고, 큰 gradient가 AdamW에서 반드시 큰 update를 만들지 않으며, 큰 update가 반드시 최종 성능 향상을 뜻하지도 않는다.

### 3.9 현재 코드에서 확인된 주의사항

아래 내용은 문서 작성 시점의 실제 resolved 구조를 읽으며 확인한 사항이다.

1. **12ep target에 비활성 occupancy가 남아 있다.**

   `PARA_SSR_e2e_12ep.py`는 target을 `plan=.4, det=.2, map=.2, occ=.2`로 선언하지만 base에서 `occ_head=None`이다. 존재하는 세 task만 보면 목표 비율은 결과적으로 약 `plan=.5, det=.25, map=.25`가 된다. 현재 주 config인 60ep는 `_delete_=True`로 이를 `0.4/0.3/0.3`으로 교체하므로 이 문제를 피한다.

2. **60ep config의 occupancy pipeline 설명 일부는 현재 base보다 오래됐다.**

   60ep 주석에는 occupancy GT rasterization이 남는다고 적힌 부분이 있지만, 현재 base의 train/test pipeline에서는 `GenerateSSROccLabels`와 collect key가 실제로 주석 처리되어 있다. 실행되는 코드는 occupancy rasterization 없음이 맞다.

3. **Motion validation regression은 anomaly hook이 현재 자동 감시하지 못한다.**

   evaluator key는 `motion_EPA/car`, `motion_EPA/pedestrian`인데 anomaly hook의 기본 direction table에는 `EPA_car`, `EPA_pedestrian`이 들어 있다. Motion 수치는 W&B/JSON에서 직접 확인해야 한다.

4. **`tok_sim`은 anomaly hook의 자동 중단·경고 조건이 아니다.**

   `tok_sim→1` 추세는 dashboard에서 직접 감시한다.

5. **W&B metadata보다 resolved config와 실제 log key를 우선한다.**

   일부 오래된 sweep 주석·tag에는 occupancy가 활성인 시절의 표현이 남아 있다. 실제 head 존재 여부는 resolved config의 `model.occ_head`와 로그 key로 판단한다.

### 3.10 권장 모니터링 순서

장기 run 중에는 다음 순서로 보면 빠르게 상태를 판단할 수 있다.

1. `loss`, `grad_norm`에 NaN/Inf가 없는지 확인한다.
2. `time`, `data_time`으로 GPU가 DataLoader를 기다리는지 본다.
3. `clip/rate`, `clip/factor`, `pnorm/*`로 clipping 강도와 원인을 본다.
4. warmup 이후 `gscale/*`이 clamp에 붙지 않는지 확인한다.
5. `gshare/*`가 목표 `0.4/0.3/0.3` 부근으로 가는지 확인한다.
6. `gcos/*`, `enc_gcos/*`로 planning과 auxiliary task의 conflict 추세를 본다.
7. `map_conf_pos-neg`, `map_spread_m`, `map_pts_err_m`으로 map head가 실제로 학습되는지 본다.
8. `tok_sim`, `tok_cover`, `tok_union`, `bev_std`로 shared representation collapse를 본다.
9. 6-epoch validation에서 planning L2/collision, map mAP, detection mAP/NDS, motion EPA를 함께 확인한다.
10. `anomalies.jsonl`의 상세 사건을 확인하되, 경고 하나만으로 run을 중단하지 말고 원본 지표의 추세를 재확인한다.

최종 결론은 single iteration이나 한 번의 validation 값보다 여러 measurement의 추세, 동일 schedule의 plan-only control, 1-GPU EMA evaluation을 함께 사용해 내린다.
