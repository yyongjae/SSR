# 실험 보고서 #08 — SSR-noFFP에서 현 PARA-SSR까지: 구조 해설과 최종 코드 검수

**검수 대상:** `/home/yongjae/e2e/SSR`의 2026-08-18 최신 작업 트리  
**주 실험 config:** `projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py`  
**비교 기준:** `/home/yongjae/e2e/SSR-orig/projects/configs/SSR/SSR_noffp_e2e.py`  
**검수 방식:** 코드 경로 추적, SSR/VAD/UniAD 구현 대조, config resolve, CPU model build, 정적 검사와 제공된 회귀 테스트 재실행. 이 검수에서는 모델 코드를 수정하지 않았다.

---

## 0. 전반적인 요약

### 0.1 한 문장 결론

현 구조의 **핵심 forward graph, 멀티배치 처리, aux head 초기화, map/occupancy 평가 배선은 대체로 정상**이다. 다만 장기 본실험을 지금 그대로 시작하는 것은 권하지 않는다. 특히 아래 세 항목은 학습 결과의 재현성과 해석을 직접 훼손하므로 먼저 고쳐야 한다.

1. **GradBalancer 상태가 checkpoint에 저장되지 않는다.** Resume하면 scale과 iteration이 초기화되어 aux→BEV gradient가 다시 600 iteration 동안 scale 1(unscaled)로 들어간다.
2. **map 품질 metric이 NumPy RNG를 소비해 실제 map loss의 GT shift와 다음 iteration의 GridMask augmentation까지 바꾼다.** 즉 계측 on/off가 학습 자체를 바꾼다.
3. **현재 소스가 commit으로 고정되지 않았다.** HEAD `9010b77` 위에 34개 status 항목이 있고 핵심 config/controller/hook/test가 untracked 또는 dirty 상태라 checkpoint만으로 실험을 재현할 수 없다.

그 다음 우선순위는 controller 첫 600-step 정책, partial-horizon motion mode 선택 오류, GT를 요구하는 inference API, full detection evaluator의 silent skip이다.

### 0.2 최종 판정표

| 영역 | 판정 | 핵심 근거 |
|---|---:|---|
| Fresh train forward | **PASS** | target config resolve/model build 성공, 기존 bs=4 및 2-rank smoke 근거 존재 |
| 멀티배치 correctness | **PASS** | command, camera visibility, image shape, per-sample occ mask 수정 반영 |
| Deformable attention 초기화 | **PASS** | det 6 + map 3 layer의 offset/attention weight와 radial bias 재검증 정상 |
| Aux loss/valid mask | **대체로 PASS** | occ invalid frame gradient 차단, map/occ 단위검사 통과 |
| EvalHook 배선 | **조건부 PASS** | dict unwrap/order/valid mask 정상. 다만 full det metric은 미검증이고 오류를 숨길 수 있음 |
| 정확한 resume | **FAIL** | GradBalancer scale, `_seen`, `_train_iter`가 state dict에 없음; EMA도 별도 resume 필요 |
| 계측의 비개입성 | **FAIL** | `map.train_metrics()`가 loss보다 먼저 별도 random shift를 materialize |
| 순수 deployment inference | **FAIL** | `forward_test`와 `simple_test_pts`가 GT를 필수로 요구 |
| PARA-Drive 구현 재현 | **아님** | 병렬 dependency graph만 차용; head, planner, occupancy, metric protocol은 별도 조합 |
| 장기 본실험 착수 | **HOLD** | 아래 Blocker 3건 해결 및 source snapshot 후 권장 |

### 0.3 먼저 처리할 일

**본실험 전 필수**

1. GradBalancer의 scale, `_seen`, global train iteration을 checkpoint에 저장하고 resume round-trip test를 추가한다.
2. map GT shifts를 한 번만 생성해 metric과 loss가 공유하도록 바꾼다. 최소한 metric 호출 전후 NumPy RNG를 완전히 보존해야 한다.
3. controller가 iteration 1~600에 어떤 scale을 쓸지 명시적으로 결정한다. 지금은 target 40/20/20/20이 아니라 사실상 raw aux 99.93% 구간이다.
4. 현재 소스를 commit/tag하거나, 최소한 tracked diff와 untracked source를 함께 immutable snapshot으로 저장한다.
5. 존재하지 않는 `--resume-from` 경로는 즉시 실패하게 하고 raw/EMA/controller 복원 조합을 실제로 시험한다.

**첫 장기 run의 metric을 믿기 전 필수**

6. full 6019-sample detection eval을 한 번 끝까지 통과시키고 mAP/NDS key 존재를 assert한다.
7. motion partial mask의 last-step mode 선택을 수정한다.
8. 최종 보고용 metric은 단일 GPU에서 `epoch_N_ema.pth`로 다시 산출한다.

---

## 1. 검수 시점의 정확한 실험 정의

Target config를 실제로 resolve한 값은 다음과 같다.

| 항목 | 값 |
|---|---:|
| 모델 | `ParaSSR` |
| GPU / batch | 2 GPU × 4 samples = global batch 8 |
| train set / val set | 28,130 / 6,019 samples |
| iterations per epoch | 3,517 |
| epochs / 총 optimizer steps | 60 / 약 211,020 |
| optimizer | AdamW, base LR `2e-4`, weight decay `0.01` |
| image backbone LR | `2e-5` (`lr_mult=0.1`) |
| LR schedule | 500-iter linear warmup + 60-epoch cosine |
| gradient clip | 모든 parameter를 한 번에 global L2 norm 35 |
| GradBalancer target | plan 0.4 / det+motion 0.2 / map 0.2 / occ 0.2 |
| controller | interval 200, momentum 0.9, clamp `[1e-5, 1]`, warmup 500 |
| aux quality / grad logging | 200 iterations마다 생성 |
| validation | 5 epochs마다, raw model, aux heads on |
| checkpoint | raw 매 epoch, 최대 60개 + EMA 매 epoch |
| final intended evaluation | single-GPU EMA checkpoint |

CPU에서 현재 모델을 직접 build해 센 parameter 수는 다음과 같다.

| 모듈 | parameters |
|---|---:|
| ResNet-50 image backbone | 23,508,032 |
| FPN | 1,114,624 |
| shared BEV encoder + planner (`pts_bbox_head`) | 7,068,759 |
| detection + motion | 6,670,536 |
| vector map | 2,910,001 |
| occupancy | 406,471 |
| **현재 전체** | **41,678,423** |
| **현재 trainable** | **41,402,903** |
| SSR-noFFP 전체 / trainable | 33,793,931 / 33,518,397 |

현재 slim shared path는 약 31.69M이고 aux 3개 합은 약 9.99M이다. SSR-noFFP의 `SSRHead`에는 실제 planning forward에서 쓰지 않는 det/map/motion branch와 FFP action plumbing이 남아 있어, 단순히 두 전체 parameter 수를 빼서 “aux 크기”로 해석하면 안 된다.

---

## 2. 구조를 한눈에 보기

### 2.1 시작점: SSR-noFFP

SSR-noFFP는 이름 그대로 SSR에서 Future Feature Prediction(FFP) loss만 끈 버전이다.

```text
6-camera temporal images
  ├─ history frame 2개 ── no-grad recurrent BEV
  ├─ current frame ────── ResNet-50 C5 → FPN → BEVFormer encoder ×3
  └─ future frame ─────── noFFP에서는 encode하지 않지만 로딩/샘플 필터에는 유지
                                      │
                                      ▼
                         BEV [B, 10000, 256]
                                      │
                     navigation command SE channel gate
                                      │
                       TokenLearner: 10000 → 16 tokens
                                      │
                        latent self-attention decoder ×3
                                      │
                   18 waypoint queries = 3 commands × 6 steps
                                      │
                         cross-attention decoder ×1
                                      │
                       relative waypoint offsets [B,3,6,2]
                                      │
                      commanded branch masked L1 planning loss
```

여기서 중요한 점은 noFFP config가 future frame을 여전히 queue에 둔다는 것이다. 실제 tensor queue는 **history 2 + current 1 + future 1**이다. `latent_world_model=None`이면 future BEV는 계산하지 않지만, future sample에도 box/map GT가 있어야 해당 training sample을 유지하는 원 SSR ablation 조건은 보존한다.

`SSRHead` 안에는 detection/map/motion branch가 선언돼 있지만 noFFP의 실제 loss는 planning L1만 반환한다. 즉 이 parameter들은 planning path의 일부가 아니라 원 VAD 계보에서 남은 dead allocation이다.

### 2.2 현재 구조: shared BEV 위의 네 갈래

현재 모델은 FFP와 future queue frame을 완전히 제거하고, current BEV 하나를 네 모듈이 읽는다.

```text
history 2 + current 1, six cameras
              │
       ResNet-50 C5 + FPN
              │
  BEVFormer encoder ×3 + prev_bev
              │
     shared bev_embed [B,10000,256]
              │
       ┌──────┼────────────┬──────────────┐
       │      │            │              │
       ▼      ▼            ▼              ▼
   planner  det+motion   vector map     occupancy
     SSR      VAD형        VAD형          custom conv
   sparse   300 queries  100×20 pts    [B,7,1,100,100]
    path
       │      │            │              │
       │   own losses   own losses      own losses
       │      └──── _ScaleGrad per task ──┘
       │                    │
       └──────────── shared encoder gradient 합산
```

이 모델에서 “parallel”은 **data dependency가 병렬**이라는 뜻이다. det output이 map으로 들어가거나 map output이 motion/planner로 들어가지 않는다. 하지만 실제 Python/CUDA 호출은 `det → map → occ` 순서이며 별도 CUDA stream 병렬 실행은 없다 (`para_ssr.py:313-336`, test `694-724`). Aux-on latency가 빨라지는 구조는 아니고, 배포 시 `test_aux_heads=False`로 aux 계산을 통째로 생략하기 때문에 빨라진다.

### 2.3 두 구조의 핵심 차이

| 항목 | SSR-noFFP | 현 PARA-SSR target |
|---|---|---|
| temporal train queue | history 2 + current + unused future | history 2 + current |
| shared representation | BEVFormer BEV | 동일 계열 BEVFormer BEV |
| planner | SSR sparse-token planner | slim `ParaSSRHead`, retained graph 동일 |
| auxiliary training | 없음 | det+motion, vector map, vehicle occupancy |
| aux→planner output edge | 없음 | 없음 |
| interaction | planning만 BEV를 학습 | 네 task gradient가 shared BEV에서 만남 |
| FFP | off, plumbing/dead modules 일부 잔존 | 관련 plumbing 제거 |
| LR / duration | 5e-5 / 12 epochs | 2e-4 / 60 epochs |
| 평가 | planning 중심 | planning + det + motion + map + occ |
| strict paired comparison | 기준 | 아님: dataset population, LR, duration, aux/clip 모두 다름 |

`ParaSSRHead`의 retained forward 연산은 같은 weight를 이식하면 SSR planning graph와 동등하다. 그러나 새로 같은 seed로 두 모델을 build해도 초기 weight가 paired되지는 않는다. 원 `SSRHead`가 dead branch들을 먼저 생성하며 RNG를 소비하기 때문이다. 엄밀한 paired ablation이면 shared/planner state를 명시적으로 이식하거나 multi-seed 비교가 필요하다.

---

## 3. 현 구조 상세 설명

### 3.1 데이터와 temporal queue

Train dataset의 `queue_length=3`은 최종적으로 **과거 2장 + 현재 1장**을 만든다.

1. 현재 index 이전 3개 후보를 만든다.
2. 그중 하나를 무작위로 버리고 나머지 2개를 시간순으로 정렬한다.
3. scene 시작처럼 과거 frame이 없으면 직전에 준비한 frame을 deepcopy해 queue 길이를 맞춘다.
4. `union2one()`은 image, command, ego future trajectory/mask만 temporal stack한다.
5. box/map/occupancy GT는 `queue[-1]`, 즉 현재 frame의 값을 유지한다.
6. `can_bus` pose는 인접 frame 간 delta로 바꾼다.

관련 경로는 `nuscenes_vad_dataset.py:1134-1252`, model 쪽은 `para_ssr.py:213-229,267-285`다. History image는 backbone/BEV encoder를 통과하지만 `torch.no_grad()`이며, current frame만 planning/aux loss graph를 만든다.

SSR-noFFP의 unused future frame을 제거한 것은 current aux GT 정렬 관점에서는 옳다. 다만 future frame GT 존재 여부로 sample을 거르던 조건까지 없어지므로 noFFP와 **학습 sample population이 엄밀히 동일하지는 않다.**

### 3.2 image encoder와 shared BEV

- 입력은 nuScenes 6-camera image다.
- 원본 1600×900을 0.4배 scale하고 pad하므로 통상 network 입력은 640×384다.
- ResNet-50의 마지막 C5 feature만 사용한다. stage 1과 BN은 freeze되어 있다.
- 단일-level FPN이 channel을 256으로 맞춘다.
- BEVFormer encoder 3개 layer가 temporal self-attention과 camera spatial deformable cross-attention을 수행한다.
- BEV 범위는 x `[-15,15] m`, y `[-30,30] m`, grid는 100×100이다.
- 따라서 한 cell은 lateral 0.3 m × longitudinal 0.6 m다.
- 최종 `bev_embed` shape은 `[B, 10000, 256]`이다.

History BEV와 ego-motion `can_bus` delta를 이용하는 recurrent BEV 구조는 유지된다. Multibatch에서 sample 0의 metadata/visibility를 다른 sample에 broadcast하던 문제는 현재 batch-major indexing으로 수정돼 있다.

### 3.3 planning head

Planner는 다음 순서로 동작한다.

1. 현재 sample의 3-way navigation command를 embedding한다.
2. SE gate로 BEV channel을 command-conditioned하게 조절한다.
3. BEV positional encoding을 붙여 cell당 512-d vector를 만든다.
4. TokenLearner가 10,000 cells를 16개의 scene token으로 압축한다.
5. 3-layer latent decoder가 16 token 사이 self-attention을 수행한다.
6. `3 commands × 6 future steps = 18` waypoint query가 token을 1-layer cross-attend한다.
7. MLP가 각 step의 `(dx, dy)`를 예측한다.
8. 학습은 실제 command branch에만 future mask를 곱한 L1을 적용한다.
9. 평가는 6개 relative offset을 `cumsum`해 0.5/1.0/.../3.0 s absolute trajectory로 바꾼다.

`ego_his_trajs`와 `ego_lcf_feat`는 현재 planner가 직접 사용하지 않는다. Ego motion은 BEV encoder의 `can_bus`를 통해 간접 반영된다.

### 3.4 detection + motion head

이 head는 300개의 agent query를 사용한다.

- 6-layer deformable decoder가 shared BEV를 cross-attend한다.
- 각 layer에서 10-class nuScenes classification과 3D box regression을 낸다.
- 마지막 layer agent feature에 6개 motion-mode embedding을 더한다.
- agent-to-agent self-attention 1회 후 mode별 6-step relative trajectory를 낸다.
- Detection은 focal classification + L1 box loss다.
- Motion은 winner-take-all trajectory L1 + mode focal loss다.
- Map query와의 cross-attention은 의도적으로 없다.

이것은 코드상 **tracking head가 아니다.** Persistent track query, identity association, track ID, AMOTA가 없다. Temporal context는 shared `prev_bev`뿐이다. 따라서 정확한 이름은 “frame-wise detection + detection-conditioned motion forecasting”이며, PARA-Drive의 tracking-prediction module과 동일하다고 쓰면 안 된다.

평가 때 motion은 detection result를 deepcopy하고 score `>0.6`만 남긴 뒤 VAD ViP3D 방식으로 EPA/ADE/FDE/MR counter를 만든다. 이전의 in-place label 훼손 문제는 수정돼 있다.

### 3.5 vector map head

현재 map head는 과거 dense 3-channel segmentation head가 아니라 VAD식 vector map이다.

- 100개의 polyline instance query
- polyline당 20 points
- class: divider / pedestrian crossing / boundary
- 3-layer map deformable decoder
- Hungarian matching
- focal class loss
- ordered point L1 + direction cosine loss
- box/GIoU loss weight는 0이며 matching 보조 표현만 남는다.
- 평가에서는 score top 50을 decode하고 chamfer threshold 0.5/1.0/1.5 m AP를 계산한다.

이 선택은 SSR/VAD 계보와 잘 맞고 현재 synthetic map positive-control은 통과한다. 하지만 PARA-Drive 최종 모델의 dense 4-channel Panoptic SegFormer와 같은 module이나 같은 metric은 아니다.

### 3.6 occupancy head

Target config의 occupancy는 vehicle 1개 channel만 사용한다.

- shared BEV를 `[B,256,100,100]`으로 reshape
- 3×3 convolution trunk `256→128→64→64`
- 1×1 conv로 present + 6 future = 7 frame logits 동시 출력
- output: `[B,7,1,100,100]`
- top-k BCE(`ratio=0.25`, future discount 0.95, weight 5) + Dice(weight 1)
- 평가 threshold: 0.1/0.3/0.5

GT는 **현재 frame에 존재하는 agent의 현재 box와 저장된 future trajectory/yaw**를 current LiDAR frame에서 이동시켜 rasterize한다. Range filter 전에 만들기 때문에 현재 annotation에 존재하지만 지금 BEV 밖에 있고 미래에 들어오는 agent는 포함한다. 반면 현재 frame에 아직 존재하지 않는 future entrant는 포함할 수 없다. 이는 UniAD가 실제 미래 frame annotation을 다시 읽는 protocol과 다른 surrogate target이다.

Scene 끝의 future frame은 `gt_occ_valid`로 BCE/Dice와 evaluator에서 제외한다. 이 valid-mask 배선 자체는 현재 정상이다.

### 3.7 task gradient valve와 GradBalancer

Aux loss weight를 줄이면 aux head 자체도 약하게 학습된다. 현재 코드는 이를 피하려고 `_ScaleGrad`를 aux head 입력 BEV에 둔다.

```text
forward:  x 그대로 전달
backward: dL/dx에 task scale만 곱함
```

따라서 aux head parameter gradient는 full strength이고, aux가 shared BEV로 보내는 gradient만 조절된다.

GradBalancer는 planning을 scale 1인 기준 task로 두고, 매 측정에서 다음을 푼다.

```text
raw_g(k) = measured_g(k) / current_scale(k)
new_scale(k) = target(k)/target(plan) × raw_g(plan)/raw_g(k)
```

Target은 40/20/20/20이다. 여기서 “share”는 각 task의 `||dL/d bev_embed||`를 더한 값에 대한 비율이다. 이는 **실제 model parameter update 지분이 아니다.** Shared encoder Jacobian, task 간 방향 상쇄, rank 간 DDP vector averaging을 통과하기 전의 activation-gradient norm proxy다.

또한 global clip 35는 aux head 자체 parameter gradient까지 포함해 모든 parameter에 한 번에 적용된다. BEV valve를 작게 해도 aux head parameter norm이 크면 planner와 encoder도 같은 clip factor를 받는다.

### 3.8 train, validation, deployment의 차이

- 학습: planner와 aux 3개를 모두 계산한다.
- 60ep in-training validation: `test_aux_heads=True`, raw weight, 2-GPU temporal split이다.
- 최종 metric: single-GPU EMA checkpoint가 필요하다.
- 배포 의도: aux off, planner만 계산한다.

현재 코드는 마지막 “배포 의도”를 완전한 API로 구현하지는 않았다. Aux를 꺼도 `forward_test`가 GT box/label/ego future를 요구하고 planning metric을 즉시 계산하기 때문이다. 상세는 finding F-06에 적었다.

---

## 4. 상세 코드 검수 결과

### Blocker F-01 — GradBalancer와 iteration이 checkpoint에 없다

**위치:** `SSR/utils/grad_balance.py:44,79-106`, `SSR/para_ssr.py:137-153,344`  
**영향:** resume한 장기 run의 objective가 조용히 바뀜

`GradBalancer`는 `nn.Module`이 아닌 일반 Python 객체이고 `scale`은 dict, `_seen`은 set이다. `_train_iter`도 일반 int다. 실제 target model의 `state_dict()`에서 balance/scale/train_iter 관련 key는 0개였다.

직접 controller scale과 iteration을 바꿔 state dict를 새 model에 load하면:

```text
before save: scales 0.00123 / 0.00456 / 0.00789, train_iter 12345
after load : scales 1.0 / 1.0 / 1.0, train_iter 0
```

Resume 후 controller는 다시 warmup으로 들어간다. Iteration 600의 forward까지 aux scale 1이 적용되고, iteration 601부터 새 scale이 적용된다. 기존 geometric EMA history와 `_seen`도 사라진다.

**권장 수정**

- scale, seen 여부, controller update count와 model train iteration을 buffer/extra state 또는 checkpoint hook state로 저장한다.
- `runner.iter`와 model 내부 iteration 중 하나만 source of truth로 둔다.
- save→새 process load→다음 iteration의 valve와 update 여부가 원 run과 bitwise 동일한 resume test를 만든다.
- raw checkpoint와 EMA checkpoint 복원은 별개다. `MEGVIIEMAHook.resume`도 matching `epoch_N_ema.pth`를 받아야 한다.

### Blocker F-02 — map 관측 metric이 실제 학습 RNG를 바꾼다

**위치:** `para_map_head.py:457-469,497-503,551-560`, `nuscenes_vad_dataset.py:253-308`  
**영향:** `aux_metric_log_interval` 변경만으로 loss target과 augmentation이 달라짐

`forward_train()`은 다음 순서다.

```text
preds = forward(...)
train_metrics(preds, GT)   # 먼저
loss(preds, GT)            # 다음
```

Metric과 loss가 각각 `shift_fixed_num_sampled_points_v2` property를 호출한다. Closed polyline의 가능한 shift가 19개보다 많으면 이 property는 `np.random.choice`로 subset을 새로 뽑는다. 따라서 metric matcher와 loss matcher가 서로 다른 GT shift set을 볼 수 있다.

현재 val cache 기준으로 이 random branch를 타는 closed polyline은 263개, sample은 257/6019(4.27%)다. 연속 호출 결과가 실제로 달랐고 point 차이는 수 m 이상이었다.

더 큰 문제는 NumPy global RNG를 소비한다는 점이다. 다음 iteration의 model-side GridMask도 NumPy RNG를 사용하므로 metric on/off가 map target뿐 아니라 image augmentation sequence까지 바꾼다. 이 metric은 관측기가 아니라 학습 개입이다.

**권장 수정**

- 가장 좋은 방법은 GT shift tensor를 **한 번만 materialize**하고 metric과 모든 decoder-layer loss가 공유하게 하는 것이다.
- 임시 방편으로 metric 전후 NumPy RNG state를 save/restore할 수 있지만, metric과 loss가 같은 target을 봐야 한다는 요구까지 해결하려면 공유가 낫다.
- regression test는 metric on/off 두 run에서 loss, 다음 GridMask, NumPy state가 동일함을 확인해야 한다.

### Blocker F-03 — 현재 소스는 실험 재현 가능한 snapshot이 아니다

**위치:** repository 상태 전체  
**영향:** 5–7일 후 checkpoint가 있어도 같은 코드를 복원하지 못함

검수 시점은 branch `para`, HEAD `9010b77`이며 `git status`에 34개 변경/신규 항목이 있다. Target configs, GradBalancer, clip hook, launcher, 검증 scripts, 보고서 다수가 untracked다. MMCV가 merged config는 work dir에 저장하지만 Python source diff와 untracked source를 저장하지 않는다.

**권장 조치**

- 학습 전 commit/tag를 만든다.
- commit이 곤란하면 `git diff`, `git diff --cached`, untracked source archive, resolved config, environment, exact launch command를 같은 immutable run manifest에 둔다.
- checkpoint meta와 W&B config에 source SHA와 dirty 여부를 기록한다.

### P1 F-04 — controller의 첫 600 iteration은 target과 정반대다

**위치:** `grad_balance.py:90-112`, `PARA_SSR_e2e_2gpu_b4.py:40-66`  
**영향:** epoch 1 초기에 irreversible한 aux-dominated BEV 형성

Scale은 모두 1로 시작한다. 조건이 `it > 500 and it % 200 == 0`이므로 첫 solve는 iteration 600 forward가 끝난 뒤다. 즉 iteration 1~600은 unscaled다. 600/3517은 epoch 1의 약 17.1%다.

코드에 기록된 raw split은 대략 plan 0.07%, det 57%, map 27%, occ 15%다. 따라서 이 구간은 40/20/20/20이 아니라 aux가 norm 합의 약 99.93%를 차지한다. 반면 plan-only arm은 zero target scale을 constructor에서 바로 0으로 만들어 iteration 1부터 aux→BEV가 없다.

“gradient가 안정될 때까지 기다린다”는 의도는 이해되지만, 기다리는 동안 가장 위험한 raw objective를 사용한다. 다음 중 하나를 명시적으로 고르는 편이 낫다.

- 이전 probe에서 얻은 safe initial scales로 시작하고 600에서 controller에 넘긴다.
- warmup 동안 aux→BEV를 0 또는 보수적 scale로 둔다.
- 훨씬 이른 여러 batch를 평균해 초기 scale을 정한다.
- 현재 정책을 유지한다면 실험 이름과 해석에 “600-step unbalanced burst”를 포함한다.

Planning gradient 자체를 500배 키우는 구현은 아니다. Planning은 항상 scale 1이고 aux를 줄여 **상대 share**를 높인다.

### P1 F-05 — partial-horizon motion target은 항상 mode 0을 고른다

**위치:** `para_det_motion_head.py:416-439`  
**영향:** partial trajectory의 WTA regression과 mode classification supervision 오류

현재 코드는 per-step distance에 future mask를 곱한 뒤 무조건 마지막 index를 읽는다.

```python
dist = dist * gt_fut_masks[:, None, :]
dist = dist[..., -1]
```

마지막 future step이 invalid이면 모든 mode의 마지막 distance가 0이므로 `argmin`은 mode 0이다. Mode 1이 유효 구간을 정확히 맞추고 mode 0이 크게 틀려도 mode 0이 선택된다. 유효 future가 0개인 positive query도 trajectory classifier는 mode 0 target을 받는다.

Range-filtered agent를 샘플링한 측정에서 1~5 step만 유효한 agent가 약 22%, zero-step도 수 %라 영향이 작지 않다. 이 코드는 VAD에도 있지만 논리 오류를 그대로 상속한 것이다.

**권장 수정**

- 각 instance의 last-valid index를 gather해 FDE winner를 정한다. 또는 valid ADE로 winner를 정한다.
- zero-valid positive는 trajectory regression과 mode classification weight를 0으로 둔다.
- partial/zero/full mask 각각의 synthetic counterexample을 test한다.

### P1 F-06 — prediction-only inference API가 없다

**위치:** `para_ssr.py:421-468,475-506,725-752`  
**영향:** unlabeled image deployment, export, `forward_dummy` 불가

`forward_test` positional arguments에 `gt_bboxes_3d`와 `gt_labels_3d`가 필수다. `simple_test_pts`도 항상 GT box/map/attribute/future flag를 index하고 planning metric을 계산한다. `test_aux_heads=False`여도 동일하다. `forward_dummy()` 역시 필수 GT를 채우지 않아 현재 signature로는 실패한다.

따라서 “deployment에서는 aux를 끄면 planner만 빠르게 실행한다”는 구조적 의도와 실제 public API가 다르다.

**권장 수정**

- prediction과 metric 계산을 분리한다.
- prediction-only path는 image, metadata/can_bus, navigation command만 요구한다.
- GT가 있을 때만 별도 evaluator가 planning/collision metric을 계산하게 한다.
- unlabeled batch와 `forward_dummy` smoke test를 추가한다.

### P1 F-07 — full detection evaluation은 아직 검증되지 않았고 실패를 숨긴다

**위치:** `nuscenes_vad_dataset.py:1964-2034`, `tools/verify_dist_eval.py:114-122`  
**영향:** epoch validation이 끝나도 det mAP/NDS가 조용히 빠질 수 있음

Subset distributed test는 nuScenes full split assertion 때문에 detection을 원래 계산할 수 없다. 그런데 verifier는 metric family를 출력만 하고 detection key를 assert하지 않은 채 `DIST EVAL PATH OK`를 출력한다.

Production evaluator도 `NuScenesEval`의 **모든 `AssertionError`**를 catch해 `None`으로 바꾼다. Full split에서 schema/order/class bug가 나도 map/occ/planning만 남고 det metric은 조용히 사라진다.

**권장 수정**

- partial length는 `evaluate_det()`의 사전 check로만 skip한다.
- full-length evaluation의 assertion은 re-raise한다.
- full-val smoke에서 `pts_bbox_NuScenes/mAP`와 `NDS` 존재를 hard assert한다.
- Target val/test에는 `det_score_thresh`가 없어서 현재 default 0.0만 평가한다. VAD의 0.2 protocol과 비교하려면 `[0.0, 0.2]`를 명시하고 둘을 별도 protocol로 보고한다.

참고로 “threshold를 낮추면 AP에 항상 유리하다”는 현재 docstring도 보장되지 않는다. nuScenes interpolation에서는 마지막 TP 뒤 low-score FP가 특정 precision bin을 낮출 수 있다.

### P1 F-08 — occupancy target과 metric에는 네 가지 protocol 주의점이 있다

#### (a) Future entrant 누락

현재 GT는 current cohort의 future track만 이동시킨다. 미래 frame에 새로 등장하는 agent는 없다. Loss와 자체 evaluator는 서로 일관되지만 UniAD/PARA occupancy protocol과 동등하지 않다.

이 차이는 이론적인 예외에 그치지 않는다. nuScenes val 전체에서 future frame의 vehicle annotation 중 current frame에 동일 `instance_token`이 없고, target BEV 안으로 들어오는 box를 세면 horizon 1~6별로 `181 / 330 / 467 / 592 / 726 / 844`개, 합계 3,140 sample-horizon box였다. 3초 horizon에서는 673/6,019 samples(11.2%)가 최소 한 개의 누락 future entrant를 갖는다. 일반 scene occupancy forecasting이 목표라면 future-frame annotation 기반 target이 필요하고, 현 정의를 유지한다면 “current-agent-conditioned occupancy”라고 명명하는 편이 정확하다.

#### (b) Train/val annotation filter 불일치

Train dataset은 `use_valid_flag=True`, val/test는 이를 지정하지 않아 `num_lidar_pts > 0` 기준을 쓴다 (`nuscenes_vad_dataset.py:1268-1273`). Camera-only 모델인데 val은 lidar point가 0인 box를 제외한다.

Val raw 측정에서는 `valid_flag=True`지만 lidar point가 0인 box가 8,854개/3,271 frames였고, 그중 occupancy vehicle class가 7,329개였다. 이는 eligible vehicle의 약 7.9%다. VAD convention을 따르려는 것인지, train/eval target 일관성을 우선할지 명시해야 한다.

#### (c) Masked BCE의 denominator

Invalid frame loss를 0으로 만든 뒤 전체 `B×C×T×top-k`에 `mean()`한다 (`occ_loss.py:58-75`). 따라서 current frame만 valid인 sample은 같은 valid pixel loss여도 all-valid sample의 약 1/7 weight를 갖는다. Invalid supervision leakage는 아니지만 “invalid를 제외한 평균”도 아니다.

Fixed per-valid-cell weighting이 의도라면 valid frame과 temporal discount weight 합으로 다시 나눠야 한다. Scene-end sample을 의도적으로 약하게 볼 목적이면 문서화해야 한다.

#### (d) Horizon별 품질이 보이지 않음

Evaluator는 present+future 7개 frame의 intersection/union을 모두 합쳐 global IoU 하나만 낸다. 쉬운 present frame이 먼 future degradation을 가릴 수 있다. `t=0`, future-only, 0.5/1/2/3 s per-horizon IoU를 함께 내는 편이 forecasting head 진단에 맞다.

### P1 F-09 — `map_spread_m`은 decoded-query collapse detector가 아니다

**위치:** `para_map_head.py:581-587`  
**영향:** healthy로 보이는 false negative 가능

현재 metric은 score와 관계없이 100개 query centre의 std를 잰다. Evaluator는 top 50만 decode한다. 따라서 high-confidence top 50이 같은 line에 완전히 collapse하고 low-confidence bottom 50만 넓게 흩어져 있어도 현재 spread는 크게 나온다. 실제 synthetic 반례에서 current spread는 9.418 m인데 decoded top-50 spread는 0이었다.

또 값은 `(std_x_m + std_y_m)/2`라 Euclidean/RMS spread가 아니며 방향에 따라 달라진다. Top-k 또는 confidence-weighted, class-aware diversity/duplicate metric으로 바꾸고 `map_spread_m`을 과도하게 해석하지 않는 것이 좋다.

### P1 F-10 — 2-GPU temporal validation bias는 한 sample이 아니다

**위치:** `DistributedSampler`, `collect_results_cpu`, `para_ssr.py:421-473`, 60ep config 주석  
**영향:** in-training trend metric의 작은 temporal bias

Contiguous sampler와 rank-part concatenate는 result **순서**를 올바르게 복구한다. 문제는 rank 1이 scene 중간에서 시작해 `prev_bev=None`이라는 점이다. Target val dataset을 실제 build했을 때 rank 1 시작 index 3010은 scene frame 14이고, scene end index는 3035였다. 잘못된 recurrent state는 첫 sample 하나가 아니라 같은 scene 끝까지 최대 26 samples(0.432%)에 전파될 수 있다.

2-GPU raw validation은 trend로는 쓸 수 있지만 최종 숫자는 single-GPU sequential EMA여야 한다. Config의 “1/6019 sample” 주석은 고쳐야 한다.

### P1 F-11 — global clipping과 control-arm 해석

Plan-only config는 aux head를 제거하지 않고 aux→BEV scale만 0으로 둔다. Parameter 수, optimizer group, aux head own-gradient가 유지되므로 SSR-orig보다 훨씬 좋은 matched control이다.

다만 “clip norm까지 동일하고 오직 한 변수만 바뀐다”는 표현은 강하다. BEV가 달라지면 aux head prediction과 own-parameter gradient도 달라지고, global clip factor도 달라진다. 그 factor가 planner까지 공통 적용되므로 valve의 효과는 direct BEV gradient뿐 아니라 clip 경로도 포함한다. 두 arm의 `clip/rate`, `clip/factor`, group norm을 함께 보고해야 한다.

`ClipMonitor`의 이름도 주의해야 한다.

- `pnorm/trunk` = image backbone + neck
- `pnorm/plan` = **BEVFormer encoder를 포함한 전체 `pts_bbox_head`**
- `pnorm/aux` = det/map/occ heads

따라서 `pnorm/plan`을 planner-only norm이라고 해석하면 안 된다.

### P1 F-12 — 평가 artifact 경로가 의도한 `/data1`이 아니다

**위치:** `SSR/apis/mmdet_train.py:170-174`, launcher의 relative `WORK_DIR`  
**영향:** root volume 사용, epoch별 artifact 덮어쓰기

Raw/EMA checkpoint 60개 총 약 37.6 GB는 `/data1`의 1.8 TB 여유 공간에 충분하다. 그러나 EvalHook의 `jsonfile_prefix`는 다음과 같이 생성된다.

```python
os.path.join('val', cfg.work_dir, startup_timestamp)
```

기본 `WORK_DIR=work_dirs/...`는 상대경로라 evaluation artifact는 work_dirs symlink 아래가 아니라 repo의 `val/work_dirs/...`에 기록된다. Repo가 있는 root volume은 검수 시 97% 사용, 약 68 GB 여유였다. Timestamp도 hook 등록 때 한 번만 정해져 매 eval이 같은 formatted artifact를 덮어쓴다.

절대경로 `WORK_DIR=/data1/yong/SSR/work_dirs/<unique-run>`를 쓰고 eval epoch별 suffix를 두는 것이 안전하다.

### P2 F-13 — inherited temporal-reference alias bug

**위치:** `SSR/modules/encoder.py:197-210`

```python
shift_ref_2d = ref_2d
shift_ref_2d += shift
```

두 이름이 같은 tensor를 가리키므로 intended previous reference뿐 아니라 current reference도 shift된다. 코드 주석대로 BEVFormer 의도는 `ref_2d.clone()`이다. SSR paper result 재현을 위해 일부러 유지한 inherited bug이므로, 지금 본실험 중 조용히 고치면 baseline compatibility가 깨진다.

권장은 현재 실험에는 명시적으로 고정하고, 별도의 `clone()` ablation에서만 수정해 두 결과를 분리하는 것이다.

### P2 F-14 — tooling과 문서의 잔존 불일치

- `tools/test.py` distributed 경로는 이미 dict인 result를 다시 `{'bbox_results': ...}`로 감싸 double-wrap한다.
- `SSR/apis/test.py:158-159`의 `collect_results_gpu()`는 CPU collector 호출 결과를 return하지 않는다. `--gpu-collect`는 `None`이 된다. Target 기본 CPU collect는 안전하다.
- `verify_grad_balance.py --skip-model`은 real model loop를 건너뛰어도 `ALL PASS`를 출력한다.
- `verify_grad_scale.py`, 일부 batch-equivalence script는 실패 시 non-zero exit를 보장하지 않는다.
- `verify_dist_eval.py`는 metric family를 assert하지 않는다.
- Text/TensorBoard와 W&B의 `by_epoch` 설정이 달라 sparse metric cadence가 깔끔하게 정렬되지 않는다.
- `gshare`는 update 전 scale의 측정인데 같은 row의 `gscale`은 update 후, 다음 iteration용 값이다.
- `analyze_gshare.py`의 “aux quality는 매 iteration raw” 설명은 실제 200-iter sparse logging과 맞지 않는다.
- `PARA_SSR_e2e_2gpu_b4.py` 상단은 5e-5/8-GPU comparable이라고 쓰지만 실제는 2e-4 + GradBalancer다.
- 60ep launcher는 eval/10이라고 쓰지만 실제는 eval/5다.
- `PARA_SSR_w_gradscale.py`는 2-GPU config가 fixed 0.01이라고 쓰지만 현재는 adaptive per-task controller다.
- `run_experiments.sh`는 현 60ep target이 아니라 old 8-GPU base, 12ep, 5e-5, unbalanced, validation-off arm을 실행한다.
- 60ep config의 epoch 12 LR 4.5e-5 설명은 옛 base 기준이다. 현재 약 1.81e-4다.
- W&B metadata에는 LR, grad target, warmup/clamp, clip norm, source revision이 빠져 있다.

---

## 5. PARA-Drive/VAD/UniAD와의 관계를 정확히 표현하기

현 코드는 좋은 의미에서 **새 조합**이다. 다만 논문 재현이라고 부르면 metric과 module 의미가 섞인다.

| 구성요소 | 출처/성격 | 동일성 판단 |
|---|---|---|
| dependency graph | PARA-Drive Fig. 5식 shared-BEV parallel graph | 개념적으로 유사 |
| planner | SSR command-gated sparse 16-token planner | PARA planner가 아님 |
| det + motion | VAD detection/motion에서 map interaction 제거 | VAD 유사, tracking 아님 |
| map | VAD 3-class vector map | PARA 최종 dense map이 아님 |
| occupancy | query-free single-shot vehicle conv | UniAD/PARA query occupancy가 아님 |
| aux deactivation | test 시 aux 생략 | PARA의 deployment 철학과 유사 |
| planning metric | SSR/VAD ST-P3 계열 0.5 m grid | PARA standardized protocol과 동일하지 않음 |
| gradient balancer | 이 repo의 새 adaptive BEV valve | 논문 기본 구성 아님 |

따라서 가장 정확한 설명은 다음과 같다.

> “SSR의 sparse planner와 temporal BEV를 유지하고, VAD형 detection/motion·vector map 및 custom vehicle occupancy를 shared BEV 위에 output-independent하게 배치한 PARA-style architecture. Aux→BEV activation-gradient norm은 별도 adaptive controller로 조절한다.”

`set head=None` 실험도 PARA Table 5의 **analogous ablation**이지 동일 module/protocol의 재현이 아니다. 코드와 config의 “Table 5 reproduce” 표현은 고치는 편이 낫다.

---

## 6. 이번 검수에서 확인된 정상 항목

다음은 현재 snapshot에서 긍정적으로 확인했다.

1. Target config가 60 epochs, LR 2e-4, batch 8, eval/5, aux-on, checkpoint 60으로 정확히 resolve된다.
2. CPU에서 target model build에 성공했다.
3. 현 ParaSSR parameter 수와 module 구성이 config와 일치한다.
4. Command는 sample별로 reshape/argmax하며 sample 0 broadcast가 없다.
5. Temporal/spatial attention의 multibatch metadata indexing은 batch-major 구조와 맞는다.
6. Det 6 + map 3 deformable attention module은 init 후 zero weight/radial bias를 보존한다.
7. Motion evaluator는 detection result를 deepcopy하고 VAD score `>0.6` filter를 적용한다.
8. Occupancy loss/eval의 per-sample future-valid mask 배선은 정상이다.
9. Map output schema, token reorder, cache coverage check, Shapely 2 chamfer path는 현재 범위에서 정상이다.
10. Contiguous distributed sampler와 rank-part concatenate의 result ordering은 서로 맞는다.
11. Launcher execute permission과 shell syntax가 정상이다.
12. `git diff --check`가 통과한다.

재실행해 통과한 명령/검사는 다음과 같다.

```text
verify_aux_metrics.py
  - occ valid-set separation/IoU
  - map Euclidean point error
  - current spread/confidence synthetic checks

verify_multibatch_and_metrics.py
  - distributed result dict unwrap
  - motion label preservation + score filter
  - occupancy valid-aware train/eval metric
  - 12ep/60ep config 분리

verify_grad_balance.py --skip-model
  - solver algebra, target shares, clamp/warmup gates

verify_grad_scale.py
  - BEV gradient만 정확히 scale, aux head own gradient/loss 불변

git diff --check, bash -n, Python compile/build
```

이 세션에는 CUDA runtime이 노출되지 않아 real-batch GradBalancer loop와 2-rank distributed eval을 이번 최종 점검에서 다시 돌리지는 못했다. 과거 smoke 근거가 있더라도 위 CPU test의 `ALL PASS`를 CUDA/DDP 최종 gate로 확대 해석하면 안 된다.

---

## 7. 권장 수정·검증 순서

### Phase A — 코드 의미를 고정

1. Map shift를 한 번 생성해 metric/loss에 공유한다.
2. GradBalancer state와 global iteration을 checkpointable하게 만든다.
3. Warmup 초기 valve 정책을 결정한다.
4. Motion last-valid/zero-valid target을 수정한다.
5. Prediction-only inference path를 분리한다.
6. Full-length detection assertion을 숨기지 않게 한다.

### Phase B — 회귀 test를 진짜 gate로 만들기

1. 모든 verifier가 failure에서 `sys.exit(1)`하도록 한다.
2. CUDA가 없어서 model test를 skip하면 PASS가 아니라 SKIP을 명확히 반환한다.
3. Resume round-trip test를 추가한다.
4. Metric on/off RNG invariance test를 추가한다.
5. Motion partial-mask counterexample test를 추가한다.
6. Unlabeled inference smoke test를 추가한다.
7. Full det eval에서 mAP/NDS key를 assert한다.

### Phase C — 실험 운영을 고정

1. source commit/tag + resolved config + environment manifest를 만든다.
2. 절대경로의 새로운 work dir를 사용한다.
3. W&B에 source SHA, LR, target, clamp, warmup, clip norm, control-arm 이름을 기록한다.
4. Resume 명령은 raw, matching EMA, controller state 세 가지가 모두 복원되는지 dry-run한다.
5. Epoch별 eval artifact path를 분리한다.

### Phase D — 결과 해석

Balanced와 plan-only 양쪽에서 다음을 함께 본다.

- 다음 measurement의 `gshare/*`와 직전 row의 `gscale/*`
- clamp pinning 여부
- `clip/rate`, `clip/factor`, `pnorm/*`
- planning L2/collision
- det mAP/NDS와 motion EPA/ADE/FDE/MR
- map chamfer mAP와 confidence/point error
- occ present/future/per-horizon IoU

최종 headline metric은 반드시 **동일 checkpoint epoch의 single-GPU EMA**끼리 비교한다. SSR-noFFP 5e-5/12ep 숫자는 현 2e-4/60ep 실험의 control이 아니다. 현 `...60ep_planonly.py`가 더 가까운 control이지만, global clipping의 간접 coupling까지 포함한 “valve 정책의 총 효과”로 해석해야 한다.

---

## 8. 처음 보는 사람을 위한 파일 지도

| 파일 | 역할 |
|---|---|
| `projects/configs/SSR/PARA_SSR_e2e.py` | 기본 architecture, dataset, losses, optimizer |
| `projects/configs/SSR/PARA_SSR_e2e_2gpu_b4.py` | 2×4 batch, adaptive target, LR 2e-4 |
| `projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py` | 60ep/eval5/checkpoint60 target |
| `projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep_planonly.py` | aux reader는 유지하고 aux→BEV만 0인 control |
| `projects/mmdet3d_plugin/SSR/para_ssr.py` | 전체 train/test orchestration, four-way fan-out, metrics |
| `projects/mmdet3d_plugin/SSR/para_ssr_head.py` | BEV encoder + SSR sparse planner |
| `.../dense_heads/para_det_motion_head.py` | detection + multimodal motion |
| `.../dense_heads/para_map_head.py` | VAD-style vector map |
| `.../dense_heads/para_occ_head.py` | vehicle occupancy decoder |
| `.../utils/grad_balance.py` | adaptive per-task BEV gradient valve |
| `.../hooks/clip_monitor.py` | global clipping/group norm telemetry |
| `.../pipelines/occ_label.py` | current-cohort occupancy GT rasterizer |
| `.../datasets/nuscenes_vad_dataset.py` | temporal queue, vector GT, all evaluators |
| `.../SSR/apis/test.py` | multi-GPU result collection |
| `train_para_2gpu_b4_60ep.sh` | target launcher |

---

## 9. 최종 권고

현 구조는 “SSR planner를 유지한 shared-BEV multi-task 실험”으로는 설계가 명확해졌고, 초기 버전의 치명적 multibatch/init/evaluator 오류도 상당 부분 제거됐다. 문제는 이제 forward가 돌아가느냐가 아니라 **그 run이 같은 objective를 끝까지 유지하고, 중단 후 정확히 이어지며, metric이 학습을 건드리지 않고, 결과 이름이 실제 protocol과 일치하느냐**에 있다.

따라서 다음 세 조건을 만족한 뒤 본실험을 시작하는 것이 가장 안전하다.

1. controller resume round-trip 통과,
2. map metric RNG invariance 통과,
3. source snapshot 고정.

그 뒤 early valve 정책과 motion partial-mask를 정리하고 full det smoke까지 통과하면, 현재 60ep balanced arm과 plan-only arm은 충분히 의미 있는 비교가 된다. 반대로 이 상태로 바로 시작하면 uninterrupted run 자체는 진행될 가능성이 높지만, resume 순간 objective가 바뀌고 계측 cadence가 augmentation을 바꾸므로 최종 차이를 architecture 효과라고 깔끔하게 설명하기 어렵다.
