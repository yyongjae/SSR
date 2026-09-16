# Planning readout: ReSMap teacher → PARA-SSR 설계와 실험 프로토콜

2026-09-16. 브랜치 `km/planning-readout`. `planning-readout-design.html`(연구 설계 v3)을
현재 PARA-SSR 코드에 맞춰 구체화하고, 논의에서 확정한 결정과 구현된 실험 도구를 정리한다.
실험 결과는 아직 없다. 이 문서는 **무엇을, 왜, 어떤 순서로 돌리는지**와 그 도구를 설명한다.

---

## 0. 목적

> Module teacher(ReSMap)가 가진 성질 중 **planning에 도움이 되는 성분만** 골라
> e2e 모델(PARA-SSR)의 BEV로 옮기고, 그 결과로 planning 성능을 올린다.

**모듈 성능(map mAP) 향상은 부산물이지 메커니즘이 아니다.** 근거는 다음과 같다.

- 모듈 점수는 라벨 재현을 재므로, planning에 쓰이지 않는 성분(설계 문서 그림 1의 "과잉")까지 포함한다.
- 현재 데이터에서 "모듈 감독 → planning 향상"이라는 연결은 오히려 역방향이다.

| PARA-SSR arm | PDMS | map mAP |
|---|---|---|
| plan only | **86.45** | – |
| map + plan (GT 라벨) | 85.26 | 39.02 |
| ReSMap teacher (참고) | – | 77.43 |

그래서 모든 arm에서 **PDMS와 map mAP를 나란히 보고하고 둘의 상관을 직접 잰다.**
"mAP는 거의 안 올랐는데 PDMS가 올랐다"는 결과는 그 자체로 설계 문서 그림 1을 뒷받침한다.

## 1. 적용 범위

BEV 수준 증류는 **planner가 BEV를 직접 읽는 모델**에서만 의미가 있다.

| 모델군 | planner 입력 | 이 설계 |
|---|---|---|
| PARA-SSR, TransFuser/LTF, PARA-Drive, Hydra 계열 | BEV를 직접 읽음 | 측정과 증류 모두 적용 |
| UniAD, VAD | agent/map query만 읽음 | 측정만 가능. BEV에 증류해도 planner까지 닿는다는 보장이 없다 |
| SparseDrive 계열 | BEV 없음 | 적용 대상 아님 |

query 기반 planner에서는 query를 tap하는 방식이 대응물이다. 같은 아이디어의 확장일 뿐,
이 문서의 설계가 그대로 커버하는 영역은 아니다.

## 2. 정의와 측정량

### 2.1 `h = h_dec ∘ h_enc`

```
h_enc(F, cmd) = z       BEV를 읽어 planning 요약 z를 만든다   z: [B, Nq, 256], 기본 Nq = 1
h_dec(z, ego) = τ       z에 ego를 더해 궤적을 낸다              τ: [B, 8, 3] (NAVSIM ego 좌표 offset)
```

- **ego(vx, vy, ax, ay)는 z 뒤에서 더한다** (`ego_inject="late"`).
  - z에는 ego가 없다. z_T와 z_S가 같은 ego를 공유해서 증류 loss가 저절로 줄어드는 일이 생기지 않는다.
  - BEV attention을 거치지 않으면 z가 상수이므로, ego만으로 푸는 지름길이 막힌다.
  - PARA-SSR planner는 ego를 attention 전 query에 더한다([planner_head.py](../navsim/agents/para_ssr/modules/planner_head.py) `plan_from_bev`). 이 위치(`"early"`)는 ablation으로 둔다.
- **command는 query에 넣는다** (`cmd_inject="query"`).
  - command는 BEV의 어느 쪽을 읽을지 정한다. 즉 읽기 기능의 일부다.
  - ego 속도와 달리 궤적의 크기를 정량적으로 결정하지 않으므로 지름길 위험이 작다.
  - 대가: z_T와 z_S가 같은 cmd를 공유해서 매칭의 일부가 자명해진다. `"late"`를 ablation으로 둔다.
- **`L_plan`은 ego를 더한 뒤(τ)에 건다. 증류는 ego를 더하기 전(z)에서 뽑는다.** tap이 서로 다른 두 지점에 있다.

### 2.2 측정량

| 기호 | 정의 | 학습 | loss에 쓰이나 |
|---|---|---|---|
| `S_ego` | BEV를 읽지 않는 h (`use_bev=False`)의 PDMS | ego + cmd로 학습 | 아니오. 바닥선 |
| `S_own` | teacher BEV에서 학습한 `h_T`를 teacher BEV에 적용 | `F_T`에서 학습 | 아니오 |
| `S_student` | student BEV에서 학습한 `h_S`를 student BEV에 적용 | `F_S`에서 학습 | 아니오 |
| `S_transfer` | `h_T`를 **학습 없이** student BEV에 적용 | 없음 | 아니오 |
| `S_transfer+A` | `h_T`는 고정하고, 1×1 adapter만 `F_S`에서 학습 | adapter만 | 아니오 |
| `S_shuffled` | 궤적 라벨을 섞어서 `F_T`에서 학습 | 섞인 라벨 | 아니오. `S_ego` 이하여야 정상 |

모든 측정량은 **`S − S_ego`, 즉 BEV가 ego 위에 얹어준 몫**으로 읽는다.
NAVSIM에서는 BEV 없이 ego만으로도 PDMS가 상당히 나오므로, 절대값을 비교하면 ego가 공짜로 준 점수가 비교를 흐린다.
`S_own ≈ S_ego`라면 h가 BEV를 무시하는 버그이거나 teacher에 planning 성분이 없다는 뜻이다.
학습 로그의 `val/l2_4s_bev_shuffled`(배치 안에서 BEV를 한 칸씩 밀어 넣은 L2)가 `val/l2_4s`와 같으면 전자다.

**이 값들은 증류 loss에 들어가지 않는다.** loss에 들어가는 것은 `F_T`에서 학습해 동결한 `h_enc`뿐이다.

### 2.3 해석: `S_student`가 있어야 `S_transfer`를 읽을 수 있다

`S_own`과 `S_transfer`만 비교하면 격차의 원인이 둘로 섞인다.
student BEV에 정보가 실제로 없는 경우와, 정보는 있는데 `h_T`가 그 표현 방식을 못 읽는 경우다.

| 관측 | 해석 | 처방 |
|---|---|---|
| `S_student ≈ S_own`, `S_transfer` 낮음 | 정렬 문제 | 내용 증류는 불필요하다. `S_transfer+A`로 정렬이 풀리는지 확인한다 |
| `S_student ≪ S_own` | 내용 격차 | z 공간 증류가 타당하다 |
| `S_transfer ≈ S_ego`, `S_transfer+A`도 낮음 | BEV 수준 호환 불가 | 설계 문서 §05 (3): query 레벨 매칭으로 전환한다 |
| `S_student` ≫ student의 실제 PDMS | 표현은 충분하지만 planner가 못 읽는다 | 증류로는 안 오른다. planner 쪽 문제다 |

마지막 행은 "가벼운 h로 planning이 제대로 뽑히냐"는 리뷰어 태클에 대한 방어이기도 하다.
`S_student`가 같은 체크포인트의 실제 PDMS에 근접하면, 가벼운 h도 실제 planner만큼 읽는다는 뜻이다.

## 3. `h`의 구조

### 3.1 PnP 원칙: 고정되는 것은 `h`, 모델별로 다른 것은 adapter

```
F_X ──[adapter_X]──> 정규화된 BEV ──[ h: 구조·용량 고정 ]──> z ──> τ
     1×1 conv, ROI crop                  모든 모델에서 동일
```

- BEV를 **미터 좌표를 가진 토큰 집합**으로 읽는다. 위치 임베딩은 칸 인덱스 표가 아니라 칸 중심 (x, y) 미터 값의 MLP다. 그래서 같은 ROI면 격자가 달라도 같은 h가 붙는다.
- 쿼리가 1개면 dense attention 비용이 칸 수에 선형이다. 200×200 = 4만 토큰도 가볍다.
- 모델의 query나 head 출력을 쓰지 않는다. BEV만 있으면 정의된다.
- 각 모델의 BEV→plan 모듈 **구조를 베낀 h**는 주 도구가 될 수 없다. UniAD/VAD에는 그런 모듈이 없고, teacher(ReSMap)에는 planner가 없다. 대신 "당신들이 만든 reader가 모델 X에게 불리한 것 아니냐"는 반론에 대한 **fairness check**로 쓴다. PARA-SSR에서는 `use_task_interaction=false`인 `plan_from_bev`가 그대로 해당 구조다.

### 3.2 용량 사다리 (ablation 축)

설계 문서 E0의 "linear → MLP → 소형 transformer"에 맞춘다. 뼈대는 같고 용량만 바뀐다.

| preset | 구조 | 파라미터 | 역할 |
|---|---|---|---|
| `h0` | attention pooling 1층, FFN 없음, linear head | 0.47M | "linear probe" 자리. BEV에서는 pooling 없이 진짜 linear가 불가능하고, 쿼리 1개짜리 attention pooling이 가장 작은 readout이다 |
| `h1` | cross-attn 1층 + FFN, 2-layer MLP head | 0.80M | **기본값. feasibility check를 먼저 돌린다** |
| `h2` | cross-attn 3층 + FFN, 2-layer MLP head | 1.86M | PARA-SSR planner 규모 |

**`h1`을 먼저 돌리는 이유**

- `h0`만으로는 결과가 안 나왔을 때 결론을 낼 수 없다.
- `h2`는 비용이 크다.
- `h1`은 PARA-SSR planner의 `bev_cross_attn`과 같은 종류이므로, 안 되면 우리 모델에 대한 정보가 된다.

**검토 후 뺀 구조**

- conv → flatten → MLP: flatten 때문에 격자 크기에 묶인다. 다른 모델에 붙이려면 resampling이 필요한데, 설계 문서가 resampling을 교란 요인으로 막아두었다.
- 궤적 앵커 샘플링: planning prior를 주입하므로 중립적인 측정기가 아니다.

**추가 ablation 축** (모두 `train_readout.py` 플래그로 제공)

- `--num-queries 1/4/8`: 교차로·다수 agent 장면에서 256차원 벡터 하나가 정보를 뭉개는지 본다. 장면 유형별 PDMS를 함께 보고한다.
- `--ego-inject late/early`
- `--cmd-inject query/late`

**용량 통제:** 비교 arm 사이에서는 구조, 파라미터 수, optimizer, LR, epoch, batch, seed 수, split을 모두 같게 둔다.
h 자체의 결론은 h0/h1/h2에서 **teacher 순서와 증류 효과의 부호가 유지되는지**로 판단한다.

## 4. 학습 프로토콜

### Stage 1 — teacher BEV 위에서 `h`를 학습한다. 여기가 관문이다

```
입력: teacher 캐시 F_T (동결), ego, cmd, 궤적 target     이미지 backbone 없음
학습: h만.  L = compute_plan_loss(...)                    PARA-SSR 함수를 그대로 사용
      aux 없음, GradBalancer 없음
산출: S_own, S_ego, S_shuffled (h1 × seed 3)
```

- **plan loss는 PARA-SSR의 [`compute_plan_loss`](../navsim/agents/para_ssr/para_ssr_loss.py)를 그대로 쓴다.** commanded branch만 L1로 계산하고, heading은 wrap한 뒤 0.5배, `err.mean()`으로 정규화한다. 그래야 `S_*`와 student PDMS가 같은 자로 잰 값이 된다.
- 백본이 없으므로 가볍다. fp16 BEV는 샘플당 2.56 MB이고, batch 64도 164 MB다.
- **`S_own − S_ego`가 seed 편차보다 뚜렷하게 크지 않으면 여기서 멈춘다.**

### Stage 2 — student BEV를 캐싱하고 `S_student`, `S_transfer`를 잰다

- 학습된 PARA-SSR 체크포인트를 동결하고 BEV를 teacher 캐시와 같은 형식으로 저장한다(navtrain 103,288 token × 2.56 MB ≈ 264 GB, navtest 12,146 token ≈ 31 GB).
- `S_student`: Stage 1과 **같은 스크립트**로 student 캐시에서 h를 학습한다.
- `S_transfer`: Stage 1 체크포인트를 student navtest 캐시에서 평가만 한다. 파인튜닝하면 정렬 측정이 오염된다.
- `S_transfer+A`: h는 고정하고 1×1 adapter(identity 초기화, 65,792 파라미터)만 학습한다.
- 체크포인트는 **현재 구조**(BEV 50×100, 전방 ROI, `use_stl=false`)여야 teacher와 칸이 1:1로 맞는다.

### Stage 3 — 증류

```
h_enc ← Stage 1 가중치 복사 후 동결 (agent의 submodule이 아니다. 저장/로드되지 않는다)
z_T   = h_enc(F_T, cmd)                 no_grad
z_S   = h_enc(A(F_S), cmd)              gradient가 student 백본까지 흐른다
L     = L_plan + λ · ramp(t) · d(z_S, z_T)
d     = 1 − cos (쿼리별, 기본)  |  mse
```

- **`h_enc`는 student의 살아있는 planner와 절대 묶지 않는다.** 묶으면 planner가 F_S에 맞춰 적응해서 loss는 줄지만 F_S는 움직이지 않는다.
- `ker(h_enc)` 방향의 F_S 성분에는 gradient가 0이다. teacher의 과잉을 복제하도록 강요하지 않는다.
- **λ는 GradBalancer에 맡길 수 있다.** distill의 gradient는 BEV를 통해서만 모델에 닿으므로(h_enc 동결), 기존 BEV 수준 보정이 정확하게 적용된다. 예: `KD_BALANCE="plan:0.7,distill:0.3"`. 이 경우 FP32가 필요하다. 쓰지 않으면 고정 λ를 쓴다. cosine은 [0, 2], plan loss는 O(0.1~1)이므로 λ = 1부터 시작한다.
- **λ warmup:** 학습 초반 F_S는 의미 없는 값이라 z_S 방향을 믿을 수 없다. `kd_warmup_iters` 동안 λ = 0으로 두고 `kd_ramp_iters`에 걸쳐 선형으로 올린다. 카운터는 **증류 첫 step부터** 센다. fine-tune은 base 런의 전역 iteration 카운터를 복원하므로, 전역 카운터로 세면 warmup이 통째로 건너뛰어진다. 시작 지점은 checkpoint의 extra state에 저장한다.
- missing token(teacher 캐시에 없는 frame)은 `teacher_valid = 0`으로 마스킹한다.

#### arm 구성

| # | ARM | head | teacher → student BEV 경로 |
|---|---|---|---|
| ① | `plan_only` | 없음 | 없음 (86.45, 있음) |
| ② | `map_gt` | map | GT map 라벨 (85.26, 있음) |
| ③ | `map_teacher` | map | **teacher vector를 map 라벨로** (라벨 경로) |
| ④ | `kd_readout` | 없음 | **d(h_enc(F_S), h_enc(F_T))** (readout 경로) |
| 대조 | `kd_random` | 없음 | 고정 무작위 256차원 projection에서의 d |
| 대조 | `kd_feature` | 없음 | BEV 전체 MSE (과잉 포함) |

- **③ vs ④가 논지를 가르는 쌍이다.** 정보원이 같은 teacher이고, 주입 지점만 다르다. ReSMap의 추가 prior(scene 전체 temporal memory, satellite 입력)는 ③과 ④에 똑같이 들어가서 비교에서 상쇄된다. ②와 ④를 직접 비교하면 "teacher가 더 많이 알아서"라는 반론을 막을 수 없다.
- **② vs ③**은 GT 라벨과 teacher 출력의 차이, 즉 teacher prior의 효과를 따로 분리해서 보여준다.
- **`kd_random`**은 "planning-relevant라서가 아니라 256차원으로 줄여서 좋아진 것"이라는 반론을 막는다.
  - 학습 안 된 attention reader는 대조군으로 쓸 수 없다. 무작위 가중치의 attention pooling은 수천 칸을 평균해서 거의 상수인 벡터를 낸다. 무관한 BEV끼리의 cosine 거리가 약 1e-5다. 그러면 대조군이 차원만 맞춘 게 아니라 그냥 약해진다.
  - 그래서 채널 16 × 공간 16 방향의 분리형 orthonormal projection을 쓴다.
- **`kd_feature`**는 "과잉을 강요하지 않는 것이 이득"임을 직접 보인다. ④가 이걸 이겨야 한다.
- 모든 arm에서 **PDMS와 map mAP를 함께** 기록한다. map head가 없는 arm(①, ④, 대조군)은 mAP를 잴 수 없으므로
  **detached map probe**(`MAP_PROBE=1`)를 붙인다. map head를 켜되 GradBalancer target을 `map: 0`으로 둔다.
  head는 자기 파라미터로 학습되지만 BEV로 가는 gradient는 제거되므로(스모크에서 `gshare/map = 0.0` 확인),
  BEV를 바꾸지 않고 BEV에 담긴 map 정보만 잰다. 비교하는 arm에는 전부 붙이거나 전부 떼야 한다.
  기존 ①(86.45)은 probe가 없는 런이므로 probe를 켠 ①을 다시 돌려야 같은 조건이 된다.
  평가는 `agent.config.test_aux_heads=true`로 한다.

#### 처음부터 학습할까, fine-tuning할까

두 방식은 답하는 질문이 다르다.

| | 처음부터 (joint) | 학습된 student + fine-tuning |
|---|---|---|
| 질문 | 증류가 표현이 형성되는 과정을 바꾸는가 | 이미 학습된 모델을 증류로 더 올릴 수 있는가 |
| 비용 | arm마다 전체 학습 | 짧음 |
| 혼동 요인 | 적음 | **더 오래 학습한 효과**가 섞인다 |

1. **먼저 fine-tuning으로 feasibility를 본다.** plan-only 체크포인트에서 시작하고(`S_transfer`를 잰 그 체크포인트), `INIT_CKPT`, 낮은 LR, `KD_WARMUP=0`, 짧은 ramp를 쓴다. **대조군(`ARM=plan_only`에 같은 INIT_CKPT, 같은 step, 같은 LR)이 필수다.**
2. **본 실험은 처음부터 학습한다.** ①~④와 대조군을 같은 스케줄로 돌리고, λ warmup을 켠다. PlanKD 계열 비교 대상도 이 설정이다.

## 5. 예상되는 리뷰어 태클

| 태클 | 대응 |
|---|---|
| 학습된 projection을 끼운 feature KD 아니냐 (novelty) | projection을 KD가 아니라 planning objective로 학습했다는 점이 차이다. `kd_random`과 `kd_feature` 대조군으로 수치로 보인다 |
| 이렇게 가벼운 h로 planning이 뽑히냐 | h0→h1→h2 용량 곡선이 포화하는지 보인다. `S_student`를 student 실제 PDMS와 맞춰본다 |
| 256차원 벡터 하나로 복잡한 장면을 담을 수 있냐 | 쿼리 수 ablation과 장면 유형별 PDMS |
| 결론이 h 선택에 달린 것 아니냐 | h0/h1/h2와 native 구조 복사본에서 순서와 부호가 유지되는지 |
| student BEV는 h_enc에게 처음 보는 입력 아니냐 | `S_transfer`, `S_transfer+A`를 먼저 보고한다. 포기 기준(§2.3)을 미리 정해둔다 |
| 고용량 h가 없는 정보를 만든다 | `S_shuffled`가 `S_ego` 이하로 떨어져야 한다 |
| PDMS는 non-reactive라 제한적이다 | 한계로 인정한다. 가능하면 EPDMS(navhard_two_stage)로 한 번 더 확인한다 |
| teacher가 더 많이 알아서 오른 것이다 | ③ vs ④ 비교 (§4) |

## 6. 데이터와 정렬 (실측)

### 6.1 ReSMap 캐시

- 위치: `/data3/kyungmin/kd_teacher_resmap` (turing). 공개 사본은 HF `rudals/resmap-navsim-teacher-kd`.
- 형식: `index.json`(token → [shard, row]), `meta.json`, `<field>/<shard>.npy`. fp16이며 BEV는 301 GB다.
- teacher 입력: 전방 카메라 3대 + 전방 crop 위성 타일. scene 전체 temporal memory를 켠 상태로 scene 순서대로 생성했다.
- student 입력: 전방 카메라 3대, 2 frame(`frame_indices=(2,3)`).

### 6.2 커버리지

**teacher 캐시는 navtrain의 `train_logs`만 담고 있다.** teacher는 NAVSIM의 train/val log 분할(978/214)을 따라
`log_split='train'`의 센서 있는 frame 전부(126,032)로 학습·캐싱했고, `log_split='val'`(26,463 frame)로 검증했다.
8개 log 표본에서 train_logs의 navtrain token은 1966/1966개 있었고, val_logs token은 0/303개였다.

- PARA-SSR validation(val_logs)에서는 `teacher_valid = 0`이 되어 증류 항이 마스킹된다.
- readout의 open-loop validation은 train_logs 중 log 단위 5%를 hold-out해서 쓴다.
- **navtest teacher 캐시**는 HF repo의 `navtest/`에 있다(§8.1). navtest info에는 test log의 frame 71,460개가 있고, teacher의 memory bank를 위해 전부 통과시키되 allow-list token 12,146개만 저장한다(`--only-split-tokens`, 약 31 GB). 모든 navtest token에 위성 타일이 있음을 확인했다.
- teacher와 student 모두 train_logs로 학습했다. 따라서 train 캐시는 "본 데이터"의 feature이고, 평가는 navtest feature로 한다. 이 train/test 품질 차이는 양쪽에 대칭으로 존재한다.

### 6.3 축 정렬

코드를 읽고 가정하지 않고, **실측으로 확정했다.** `tools/readout/verify_teacher_alignment.py`를 60 frame에 대해 돌린 결과다.

| teacher BEV → student 격자 | seg logit과 GT raster의 상관 |
|---|---|
| **transpose만** | **0.365** |
| transpose + 좌우 flip | 0.051 |
| transpose + 전후 flip | 0.142 |
| transpose + 둘 다 | 0.016 |

| teacher vector → student 정규화 좌표 | GT와의 chamfer |
|---|---|
| **`(x_right, y_fwd) = (1 − v, u)`** | **0.43 m** |
| 나머지 7개 swap/flip 중 최선 | 5.43 m |

- teacher의 `(256, 100, 50)` = (C, 좌측부터 lateral, forward)는 student의 `(256, 50, 100)` = (C, forward, 좌→우)와 **transpose 한 번으로 칸이 1:1 대응한다.** flip도 resample도 필요 없다.
- `BevCache.bev()`가 이 변환을 적용하고, 이후 모든 코드는 student layout만 본다.
- teacher의 vector는 nuPlan ego 축 기준으로 `u = x_forward / 32`, `v = (y_left + 32) / 64`다.

## 7. 구현

| 파일 | 역할 |
|---|---|
| `navsim/agents/para_ssr/readout/bev_cache.py` | 분할 캐시 reader (memmap, pid별 핸들), teacher layout 변환, teacher vector → PARA-SSR map target |
| `navsim/agents/para_ssr/readout/readout.py` | `PlanningReadout` (`encode`=h_enc, `decode`=h_dec), preset h0/h1/h2, 체크포인트 저장/로드 |
| `navsim/agents/para_ssr/readout/plan_targets.py` | token별 cmd / ego / 궤적 offset 저장소. PARA-SSR builder와 같은 계산 |
| `navsim/agents/para_ssr/readout/teacher_targets.py` | `ResMapTeacherTargetBuilder`: `teacher_bev`, `teacher_valid`, `teacher_map_*` |
| `navsim/agents/para_ssr/readout/distill.py` | `ReadoutDistiller`: readout / random / feature 모드, adapter, λ 스케줄 |
| `para_ssr_loss.py` | distill 항 추가, GradBalancer에 `distill` 키 허용, `map_label_source=teacher` 처리 |
| `para_ssr_agent.py` | teacher builder 조건부 등록, config 검증, adapter 파라미터를 optimizer에 추가, 체크포인트 로드 시 KD adapter 키만 허용 |
| `configs/default.py`, `para_ssr_agent.yaml` | `kd_*`, `map_label_source`, `map_pseudo_score_thr` (기본값에서는 전부 꺼짐) |
| `tools/readout/build_plan_targets.py` | Stage 0: split별 plan target 추출 (CPU) |
| `tools/readout/train_readout.py` | Stage 1/2: h 학습 (`--no-bev`, `--shuffle-labels`, `--adapter-only`, ablation 플래그) |
| `tools/readout/eval_readout_pdms.py` | 캐시된 BEV + h → 공식 `pdm_score` |
| `tools/readout/cache_student_bev.py` | Stage 2: 동결된 student BEV 캐싱 (rank 분할, resume, 누락 sensor 건너뛰기, `--merge`) |
| `tools/readout/verify_teacher_alignment.py` | §6.3 측정 |
| `tools/readout/run_stage1.sh`, `run_stage2.sh`, `collect_results.py` | 단계 실행과 결과 표 (평균 ± 표준편차, `S − S_ego`) |
| `scripts/training/train_para_ssr_kd.sh` | Stage 3 arm 런처 (fine-tune 포함) |
| `tools/readout/resmap/` | teacher 캐시 생성기 사본(resmap env + maptracker repo에서 실행; navtest용 `--split none --only-split-tokens --ann-file` 추가), HF 업로드/다운로드 |
| `tests/test_para_ssr_readout.py` | 25개 테스트 |

**기본 설정에서 기존 동작은 바뀌지 않는다.** `kd_mode=none`, `map_label_source=gt`이면 teacher builder도, distiller도 생성되지 않는다.
KD로 학습한 체크포인트는 KD 설정 없이 평가된다.

## 8. 진행 로드맵 (turing → 5090)

순서를 정하는 기준은 두 가지다. teacher는 5090(sm_120)에서 돌지 않으므로 **teacher가 필요한 작업은 turing에서 먼저 끝낸다.**
그리고 가장 오래 걸리는 **plan-only student 학습을 가장 먼저 건다.**

### 8.1 turing에서 끝낸 것과 5090에서 받을 것

teacher가 필요한 작업은 turing에서 끝냈고, 결과는 HF `rudals/resmap-navsim-teacher-kd` 한 곳에 있다.

| 데이터 | 상태 | 5090에서 |
|---|---|---|
| train teacher 캐시 (repo 루트, 301 GB) | 완료 | `download_teacher_cache.py --subsets train` |
| navtest teacher 캐시 (`navtest/`, 약 31 GB) | turing에서 캐싱 진행 중, HF 업로드는 아직 (2026-09-16 시작, 로그 `/data3/kyungmin/logs/navtest_teacher.log`) | `download_teacher_cache.py --subsets navtest --fields bev` |
| plan target (navtrain, navtest) | 5090에서 만든다 (CPU, navsim log만 필요) | `build_plan_targets.py` (§9 0-c) |
| navtest metric cache | 5090 서버에 없으면 만들거나 turing의 `/data/navsim/exp/metric_cache`를 복사 | – |

navtest teacher 캐시가 필요한 이유는 **평가**다. 학습(h 학습, Stage 3 증류)에는 train 캐시만 쓴다.
h는 추론할 때도 teacher BEV를 입력으로 받으므로, navtest 장면을 PDMS로 채점하려면 그 장면들의 teacher BEV가 있어야 한다.
train_logs 일부를 떼어 평가하면 teacher가 학습한 장면이라 feature가 실제보다 좋게 나와 `S_own`이 부풀려진다.
val_logs로 평가하려면 teacher val 캐시와 navtrain metric cache가 따로 필요한데, 후자는 turing에서 미완성이다.

### 8.2 5090 환경 확인 (반나절)

1. `pytest tests/`
2. `train_readout.py --max-train 256 --epochs 1`, `eval_readout_pdms.py --max-tokens 40`
3. `scripts/training/smoke_para_ssr.sh`로 PARA-SSR 학습 스모크. 두 가지를 확인한다.
   - torch 2.7 이상에서 navsim/nuplan-devkit이 도는지
   - FP32, GPU당 B=4가 32 GB에 들어가는지 (안 들어가면 batch를 줄이고 accumulate를 늘려 global 128 유지)

### 8.3 바로 병렬로 시작할 두 가지

**A. plan-only student 학습** — 가장 오래 걸리므로 먼저 건다.

```bash
MAP_PROBE=1 ARM=plan_only bash scripts/training/train_para_ssr_kd.sh
```

이 체크포인트 하나를 세 군데에 쓴다.
- arm ① 기준선 (probe를 켠 조건. 기존 86.45 런은 probe가 없다)
- Stage 2의 student BEV 캐시
- Stage 3 fine-tune의 시작점

**B. Stage 1 관문** — readout만 학습하므로 가볍다. GPU 한 장에 여러 개를 같이 돌려도 된다.

```bash
PRESETS="h1" SEEDS="0 1 2" bash tools/readout/run_stage1.sh
```

**통과 기준** (셋 다 만족해야 한다)
- `S_own − S_ego`가 seed 편차보다 뚜렷하게 크다.
- `S_shuffled ≤ S_ego`
- readout 학습 로그에서 `val/l2_4s_bev_shuffled`가 `val/l2_4s`보다 확실히 나쁘다. 비슷하면 h가 BEV를 보지 않는다.

**통과하지 못하면** Stage 2, 3으로 가지 않는다. teacher BEV에 planning 성분이 없는지, readout 버그인지(설계 문서 §08의 "probe가 작동하지 않음": ego 지름길, 미수렴, BEV 불일치)부터 확인한다.
A는 그대로 모델 작업의 기준선으로 쓸 수 있으므로 버려지지 않는다.

### 8.4 통과한 뒤

1. **용량 곡선:** h0, h2 (`PRESETS="h0 h2"`). A가 학습되는 동안 진행한다.
2. **Stage 2:** A가 끝나면 student BEV를 캐싱하고(navtrain, navtest) `run_stage2.sh`로 `S_student`, `S_transfer`, `S_transfer+A`를 잰다. §2.3 표로 해석한다.
3. **Stage 3 fine-tune (feasibility):** `kd_readout`과 대조군 `plan_only`를 같은 `INIT_CKPT`, 같은 LR, 같은 step으로 돌린다.
4. **Stage 3 본 실험:** ①~④와 대조군 `kd_random`, `kd_feature`를 처음부터, 같은 스케줄로, `MAP_PROBE=1`로 돌린다. PDMS와 map mAP를 함께 보고한다.
5. **ablation:** `--num-queries`, `--ego-inject`, `--cmd-inject`, λ(`KD_WEIGHT`, `KD_BALANCE`), warmup 길이.

## 9. 실행 명령

경로는 예시다. `PY`는 navsim env의 python이다.

```bash
# 0-a) teacher 캐시 (5090 서버라면 HF에서 받는다; bev + map arm용 vector, navtest는 bev만)
python tools/readout/resmap/download_teacher_cache.py --out $DATA/kd_teacher_resmap \
    --subsets train --fields bev vectors scores labels
python tools/readout/resmap/download_teacher_cache.py --out $DATA/kd_teacher_resmap \
    --subsets navtest --fields bev        # -> $DATA/kd_teacher_resmap/navtest

# 0-b) (완료, 재생성할 때만) navtest teacher 캐시: turing(A6000)의 resmap env, maptracker repo에서
#      (teacher stack은 sm_120에서 돌지 않는다). 71,460 frame 통과
cd maptracker && torchrun --nproc_per_node=4 tools/cache_teacher_kd.py \
    --cfg work_dirs/resmap_nav_rideflux_stage3/resmap_nav_stage3.py \
    --ckpt work_dirs/resmap_nav_rideflux_stage3/iter_63024.pth \
    --split none --only-split-tokens \
    --ann-file /data2/kyungmin/navsim/infos/navsim_map_infos_navtest.pkl \
    --out /data3/kyungmin/kd_teacher_resmap/navtest
python tools/readout/verify_teacher_alignment.py --cache /data3/kyungmin/kd_teacher_resmap

# 0-c) plan target (CPU)
$PY tools/readout/build_plan_targets.py --filter navtrain --split trainval --out $R/plan_targets_navtrain.npz
$PY tools/readout/build_plan_targets.py --filter navtest  --split test     --out $R/plan_targets_navtest.npz

# 1) Stage 1 — 관문
TEACHER_CACHE=$DATA/kd_teacher_resmap TEACHER_CACHE_TEST=$DATA/kd_teacher_resmap/navtest \
TARGETS_TRAIN=$R/plan_targets_navtrain.npz TARGETS_TEST=$R/plan_targets_navtest.npz \
PRESETS="h1" SEEDS="0 1 2" RUNS=$R/runs PYTHON=$PY bash tools/readout/run_stage1.sh
#   통과하면 PRESETS="h0 h2", --num-queries / --ego-inject / --cmd-inject ablation
#   (READOUT_TRAIN_ARGS로 전달)

# 2) Stage 2 — 현재 구조의 plan-only 체크포인트 필요
for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r $PY tools/readout/cache_student_bev.py \
    --ckpt $CKPT --filter navtrain --split trainval --out $DATA/student_bev/navtrain --rank $r --world 4 \
    agent.config.use_task_interaction=false agent.config.use_det_motion_head=false \
    agent.config.use_map_head=false agent.config.grad_balance_target=null & done; wait
$PY tools/readout/cache_student_bev.py --merge --out $DATA/student_bev/navtrain
#   navtest도 같은 방식 (--filter navtest --split test)
STUDENT_CACHE=$DATA/student_bev/navtrain STUDENT_CACHE_TEST=$DATA/student_bev/navtest \
TARGETS_TRAIN=... TARGETS_TEST=... RUNS=$R/runs PYTHON=$PY bash tools/readout/run_stage2.sh

# 3) Stage 3 — fine-tune으로 feasibility, 그 다음 처음부터
INIT_CKPT=$CKPT MAX_EPOCHS=5 LR=2e-5 ARM=plan_only bash scripts/training/train_para_ssr_kd.sh   # 대조군
INIT_CKPT=$CKPT MAX_EPOCHS=5 LR=2e-5 KD_WARMUP=0 KD_RAMP=2000 \
  TEACHER_CACHE=$DATA/kd_teacher_resmap READOUT_CKPT=$R/runs/teacher_h1_s0/readout.pt \
  ARM=kd_readout bash scripts/training/train_para_ssr_kd.sh
#   본 실험: ARM ∈ {plan_only, map_gt, map_teacher, kd_readout, kd_random, kd_feature},
#   INIT_CKPT 없이, 같은 스케줄, MAP_PROBE=1 (map head 없는 arm의 mAP 측정)
#   DRY_RUN=1 을 붙이면 실행하지 않고 override만 출력한다
```

학습 로그에서 볼 지표는 다음과 같다.

- `kd/raw`: λ와 무관한 z 거리. warmup 중에도 기록되므로, KD가 켜지기 전의 정렬 정도를 참고할 수 있다.
- `kd/coef`: 현재 적용 중인 λ × ramp
- `kd/valid_frac`: 배치에서 teacher 캐시에 있는 frame 비율
- `kd/pseudo_frac`: map_teacher arm에서 teacher 라벨로 대체된 비율
- `gshare/distill`, `gscale/distill`: GradBalancer를 쓸 때의 distill gradient 몫과 scale
- readout 쪽 `val/z_cos_spread`: z가 샘플마다 얼마나 다른지. 0에 가까우면 cosine 증류가 당길 것이 없다.

## 10. 5090 서버 메모

- **teacher stack(torch 1.12 / cu116 / mmcv-full 1.6)은 sm_120(RTX 5090)에서 돌지 않는다.** teacher가 필요한 작업(navtest 캐시, 필요하면 val_logs 캐시)은 옮기기 전에 turing에서 끝내야 한다. 그 뒤로는 캐시만 있으면 된다.
- Stage 1/2 readout 학습은 캐시와 navsim 코드만 필요하다. mmcv가 필요 없다.
- 5090에는 torch ≥ 2.7(cu128)이 필요하다. 이 코드는 torch 2.0.1(e2e env)에서만 검증했다. `WarmupCosLR`는 `verbose` 인자 호환 처리가 되어 있지만, navsim/nuplan-devkit 전체가 새 torch에서 도는지는 **검증하지 않았다.**
- GradBalancer와 `KD_BALANCE`는 FP32를 요구한다. 기존 레시피는 GPU당 B=4 FP32이고, 32 GB에 들어가는지는 확인하지 않았다. 안 들어가면 batch를 줄이고 accumulate를 늘려 global 128을 유지한다. KD arm은 GPU당 teacher BEV(fp16 2.56 MB)와 동결 h만 추가된다.
- 저장공간: teacher BEV 301 GB, student BEV 약 264 GB, navtest 캐시 각 약 31 GB.

## 11. 검증한 것과 하지 않은 것

**검증함** (turing, CPU. GPU는 다른 사용자가 점유 중이었다)

- 기존 테스트 98개와 신규 25개, 총 123개 통과
- `train_readout.py`: 실제 teacher 캐시로 h0/h1/h2, `--no-bev`, `--shuffle-labels`, `--adapter-only`, `--num-queries 4 --ego-inject early --cmd-inject late` 각각 1 epoch 스모크
- `eval_readout_pdms.py`: navtest metric cache로 스모크. 같은 경로로 **GT 궤적을 채점하면 PDMS 0.969**가 나온다. 궤적 좌표계와 scorer 경로가 맞다.
- `cache_student_bev.py`: 무작위 가중치 student로 캐싱 → merge → `BevCache` 로드
- teacher navtest 캐싱 경로: A6000 1장으로 60 frame을 통과시켜 allow-list token 11개가 저장되고 `BevCache`로 `(256, 50, 100)`이 읽히는 것까지 확인했다.
- 실제 PARA-SSR 모델 + 실제 scene feature + teacher target builder로 CPU 학습 2 step. `kd_readout`(GradBalancer로 distill 몫 제어, `gscale/distill` 0.40), `kd_readout` + map probe(`gshare/map` 0.0), `map_teacher`(pseudo 라벨 100% 대체)를 확인했다.
- `train_para_ssr_kd.sh`: `DRY_RUN=1`로 arm별 Hydra override 조합을 확인했다(실제 학습 런처 실행은 하지 않았다).

**하지 않음**

- 실제 실험. Stage 1~3 결과는 아직 없다. 진행 순서는 §8.
- turing에는 현재 구조의 PARA-SSR 체크포인트가 없다(학습은 다른 서버에서 했다). Stage 2 전에 가져와야 한다.
- turing의 `sensor_blobs/trainval`에는 일부 navtrain frame의 이미지가 없다. `cache_student_bev.py`는 이런 frame을 건너뛰고 `missing_r*.txt`에 기록한다. 실제 학습 서버의 데이터는 확인하지 않았다.

## 12. 미결 사항

- PARA-SSR 구조 자체가 아직 확정되지 않았다. 구조가 바뀌면 Stage 2 캐시와 Stage 3을 다시 돌려야 한다. Stage 1은 teacher에만 의존하므로 재사용된다.
- teacher 선정 기준을 `S − S_ego`로 수치화하는 것은 보류한다. 단일 점수로 teacher를 고르는 것은 너무 나이브하다.
- h를 자르는 깊이(z를 어느 층 뒤에서 뽑을지)와 여러 h를 동시에 매칭하는 앙상블은 Stage 1 결과를 본 뒤 결정한다.
- 성분 단위 분석(설계 문서 E2, `g` 축, 잔차화)은 이 브랜치에 구현하지 않았다.
- 같은 레포의 `origin/aux_distill` 브랜치(comflife)는 nuScenes/mmdet3d 경로에서 teacher BEV → adapter → SSR planner(stage 1) → 동결 adapter로 feature MSE(stage 2)를 구현했다. 이 문서는 NAVSIM 경로이고, 차이는 두 가지다. h를 공통 readout으로 고정했고, 매칭 지점이 BEV 전체가 아니라 z다.
