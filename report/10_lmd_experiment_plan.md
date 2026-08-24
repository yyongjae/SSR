# 실험 계획 #10 — LMD 차용 BEV 분해 실험

**방법론 근거:** [report #09](09_lmd_planning_centric_bev.md)
**코드:** [`tools/lmd/`](../tools/lmd/) — 브랜치 `lmd-bev-analysis`
**한 줄 목표:** *"aux를 붙여서 BEV가 달라진 부분 중, planning을 도운 곳과 방해한 곳을 100×100 지도로 분리한다."*

---

## 1. 지금 있는 자산

`/data2/byounggun/rideflux` — rsync 완료 (`EXIT:0`), 아래 3개가 전부다.

| run | epochs | task_loss_weight | 체크포인트 | 우리 조건으로 | 상태 |
|---|---:|---|---|---|---|
| `para_ssr_60ep` | 60 | plan 1 / det 1 / motion 1 / map 1 + GradBalancer | `epoch_60_ema.pth` | **둘 다** | ✅ |
| `para_ssr_stage1` | 48 | **plan 0** / det 1 / motion 1 / map 1 | `epoch_48_ema.pth` | **aux만** | ✅ |
| `para_ssr_stage2` | 12 | plan 1 / det 1 / motion 1 / map 1 + GradBalancer | `epoch_12_ema.pth` | stage1에서 fork | ✅ |

세 run 모두 `occ_head=None`이므로 **aux = detection + motion + vector map** 이다. occupancy는 이번 분석 대상이 아니다.

**중요한 두 가지**

1. **`stage2`는 `load_from = work_dirs/para_ssr_stage1/latest.pth`로 stage1 epoch 48에서 갈라져 나왔다.** 즉 이 쌍은 **정의상 같은 basin**이고, report #09 §7.1이 최상위 리스크로 지목한 문제가 여기서는 **구조적으로 없다.**
2. **`para_ssr_60ep_planonly`는 전송되지 않았다.** 사용자가 물은 "planning만" 조건의 깨끗한 대조군이 지금 없다. → E4에서 처리.

---

## 2. 비교쌍과 각각이 답하는 질문

| # | 쌍 | 답하는 질문 | basin | 등급 |
|---|---|---|---|---|
| **P1** | `60ep` 단독 (모델 내부) | 한 모델 안에서 planner와 aux head가 BEV의 **같은 곳을 보는가** | 무관 | **A** — 지금 가능 |
| **P2** | `stage1(48)` → `stage2(12)` | **planning supervision을 추가**하면 aux-shaped BEV의 어디가 바뀌고, 그게 도움이 되는가 | fork라 안전 | **A** — 지금 가능 |
| **P3** | `planonly` ↔ `60ep` | **aux를 추가**하면 plan-shaped BEV의 어디가 바뀌고, 그게 도움이 되는가 | 60ep 독립학습 → **검증 필요** | **C** — 체크포인트 없음 |
| P4 | `stage1` ↔ `60ep` | — | 서로 다른 recipe, 교란변수 5개 (report #08) | **참고용만** |

P2가 P3의 **부호 반대 버전**이라는 점이 중요하다. P3(aux 추가)를 못 하는 동안 P2(planning 추가)는 같은 기계장치로 지금 돌아가고, "planning supervision이 BEV의 어디를 자기 쪽으로 끌어오는가"를 답한다. 논문 서사로도 이게 오히려 직접적이다.

---

## 3. 실험

### E0 · 실제 모듈에서 선형화 재검증 — **게이트**

| | |
|---|---|
| 목적 | `tools/lmd/`의 검증은 op 스택 **복제본**에서 통과했다. 실제 mmcv 모듈에서도 성립하는지 확인 |
| 입력 | `ssr` conda env (이 머신엔 mmcv-full 없음 → 학습 머신에서) |
| 방법 | `verify_lmd_linearisation.py`를 실제 `SSRPerceptionTransformer` / `ParaSSRHead` import 버전으로 포팅 후 실행 |
| 산출 | `sum_cells C_i + b == trajectory` 상대오차 |
| **판정** | **< 1e-5 (fp32 기준)이면 통과.** 실패하면 어느 op가 새는지 이진탐색 — 아래 전부 중단 |
| 소요 | 반나절 |

### E1 · 한 모델 안에서 planner와 aux가 보는 곳 (P1) — **지금 가능**

| | |
|---|---|
| 목적 | basin 가정 없이 오늘 결과가 나오는 경로 |
| 입력 | `para_ssr_60ep/epoch_60_ema.pth`, nuScenes val |
| 방법 | 얼린 그래프에서 adjoint(`grad × input`)로 ① planner 궤적 → BEV cell 분해 `π_i`, ② det head / map head 출력 → BEV cell 분해 `ω_i`. `ρ`도 같이. |
| 산출 | per-sample `π, ω, ρ` npz + 집계 |
| **판정** | 각 분해가 `Σ = 출력 − bias`를 만족(정확성 회귀). 그 다음 `cos(π, ω)`, 4분면 질량 비율 |
| 읽는 법 | `ω`만 높은 칸 = **planner가 안 보는데 aux가 쓰는 BEV 용량**. `π`만 높은 칸 = planner 전용(ego shortcut 의심 지대) |
| 소요 | 후킹 1~2일 + val 500샘플 ~30분 |
| 의존 | E0 |

### E2 · hybrid forward 러너 구현

| | |
|---|---|
| 목적 | 체크포인트 두 개를 동시에 올리고 weight group을 섞어 forward |
| 방법 | `replica.hybrid`와 같은 구조를 실제 모델에. group 경계 8개: `embed · enc0 · enc1 · enc2 · tokenl · latent · way · mlp` |
| 필요한 후킹 | `MSDeformableAttention3D` / `TemporalSelfAttention`에 `sampling_locations`·`attention_weights` 캐시-재사용 (`bev_mask`·`indexes`·`count`는 `point_sampling` 유래 기하학 전용이라 안전) |
| **판정** | `assign = 전부 A` 일 때 원 모델 출력과 bit-exact 재현 |
| 소요 | 2~3일 |
| 의존 | E0 |

### E3 · paired delta: planning supervision의 효과 (P2) — **본 실험**

| | |
|---|---|
| 입력 | `stage1/epoch_48_ema.pth` (= P), `stage2/epoch_12_ema.pth` (= A) |
| 방법 | ① telescoping 양방향 → `δ_g`, ② 대표 샘플 수십 개에 exact Shapley(256 coalition) → `φ_g`, ③ **BUILD/READ 분리**: reader를 A에 고정하고 builder만 P→A → `Δτ_bev`, ④ cell 분해 후 GT 오차 방향에 사영 → `σ^Δ_i` |
| 산출 | 100×100 **부호 있는 지도** (도움/방해), group별 Shapley 기여, `H` = 방해 비율 |
| **판정 (정확성)** | `Σ_g δ_g = out_A − out_P`, `Σ_i σ^Δ_i = ⟨Δτ_bev, ê⟩` — 둘 다 fp32에서 < 1e-5 |
| **판정 (해석)** | 두 순서 스프레드가 `|Δτ|`의 몇 %인지 함께 보고. group 숫자는 **Shapley만 인용** |
| 소요 | telescoping 9 forward/샘플 → val 전체 ~1시간. Shapley는 256 forward/샘플이라 대표 샘플 30개만 |
| 의존 | E2 |

### E4 · aux 추가의 효과 (P3) — **체크포인트 확보 필요**

세 갈래 중 하나를 고른다.

| 안 | 내용 | 비용 | 비고 |
|---|---|---|---|
| **4a** | 학습 머신에서 `para_ssr_60ep_planonly` 전송 | rsync만 | **먼저 있는지 확인.** 있으면 이게 최선 |
| **4b** | `planonly` config로 60 epoch 재학습 | 60 epoch × 8GPU | 비쌈 |
| **4c** | **paired fork**: `60ep`의 `epoch_40.pth`에서 aux-off 갈래를 20 epoch 학습 | 20 epoch | **basin 문제 원천 차단**, 4b보다 1/3 비용. 방법론적으로 4a보다 낫다 |

4a를 쓸 경우 **basin 사전점검이 필수**: `θ(α) = (1−α)θ_planonly + α θ_60ep`를 α ∈ {0, .1, …, 1}로 훑으며 planning L2를 잰다. barrier가 낮으면 진행, 높으면 4c로 전환. 4c를 쓰면 이 점검이 필요 없다.

### E5 · 세 조건 종합

| | |
|---|---|
| 방법 | E1/E3/E4 결과를 하나의 표로. **`stage1`은 planner가 학습되지 않았으므로**(plan=0으로 48 epoch, weight decay만 받아 초기 norm의 ~85%) 자기 planner로 `π`/`ρ`/`σ`를 읽으면 안 된다 → BEV를 얼리고 **동일한 planner probe**를 세 BEV 위에 각각 학습해 "이 BEV가 planning 정보를 얼마나 담고 있나"를 공통 척도로 잰다 |
| 산출 | 세 조건 × {`π` 지지집합, `ρ`, PPA, `H`, `cos(π,ω)`} 표 + 지도 3장 |
| 의존 | E1, E3, E4 |

---

## 4. 지표

| 기호 | 정의 | 읽는 법 |
|---|---|---|
| `C_i` | `J_i · bev_i` — cell i가 궤적에 기여한 12차 벡터 | `Σ_i C_i + b = τ` (잔차 0) |
| `π_i` | `‖C_i‖₁ / Σ‖C_j‖₁` | planning 기여 지도. **planning-centric BEV := π의 유효 지지집합**, participation ratio `(Σπ)²/Σπ²` |
| `b`, `ρ` | `b` = BEV를 0으로 둔 frozen forward, `ρ = ‖b‖/(‖b‖+‖Σ C_i‖)` | **장면과 무관하게 나오는 궤적 비율.** ablation 없이 ego-shortcut 측정 |
| `ω_i` | det/map head 출력의 BEV cell 분해 | aux가 읽는 곳 |
| `σ^Δ_i` | `⟨J^A_i · Δbev_i, ê⟩`, `ê = (τ_P−τ_gt)/‖·‖` | **음수 = 도움, 양수 = 방해되는 BEV.** `Σ_i σ^Δ_i = ⟨Δτ_bev, ê⟩` 정확 |
| `H` | `Σ_{σ^Δ>0} σ^Δ_i / |Σ_i σ^Δ_i|` | 변화분 중 방해 비율 |
| `φ_g` | group g의 Shapley 값 | 변화가 **어느 모듈에서** 만들어졌나 |
| PPA | `(Σ_{i∈GT} π_i) / (|GT|/10000)` | GT 박스·drivable에 planning 질량이 균등 대비 몇 배 |

---

## 5. 산출물 (논문 그림 후보)

1. **Fig A** — `σ^Δ` 부호 지도 100×100, GT 박스·차선 오버레이. 빨강 = aux/plan supervision이 망친 곳, 파랑 = 도운 곳. **핵심 그림.**
2. **Fig B** — `π` vs `ω` 4분면 산점도 + 각 사분면 대표 장면 (E1).
3. **Fig C** — group별 Shapley 막대 (변화가 encoder 어느 층에서 만들어졌나).
4. **Table 1** — 세 조건 × 지표 (E5).
5. **Table 2** — 정확성 회귀: 각 분해의 `Σ` vs 출력 상대오차. **이게 있어야 나머지 숫자가 신뢰된다.**

---

## 6. 리스크와 게이트

| 리스크 | 영향 | 대응 |
|---|---|---|
| 실제 모듈에서 선형화가 안 깨끗함 | 전부 중단 | **E0가 게이트.** 실패 시 op 이진탐색 |
| `planonly` 부재 | P3 불가 | E4a 확인 → 없으면 **4c(fork)**. P2는 그동안 그대로 진행 |
| 두 체크포인트가 다른 basin | P3 해석 불가 (P2는 무관) | α-interpolation barrier 측정을 E4의 사전조건으로 |
| 경로 의존성 (측정치: `|Δτ|`의 25%) | group 숫자 왜곡 | group은 **Shapley로만** 보고, telescoping 단일 순서 인용 금지 |
| 얼린 routing — value 경로만 귀속 | 해석 범위 축소 | 결과 문장을 *"거기를 보기로 한 상태에서 내용을 누가 바꿨나"*로 한정. 별도로 두 모델의 frozen `sampling_locations` 차이를 "시선이 얼마나 옮겨졌나" 지표로 보고 |
| `stage1` planner 미학습 | 자기 planner readout 무효 | E5의 공통 probe planner |
| EMA/raw 혼용 | 비교 무효 | **전부 `*_ema.pth`로 통일** (최종 eval이 EMA) |

---

## 7. 순서

```
E0 (게이트: 실제 모듈 선형화)
 ├─> E1  P1: 60ep 내부 π vs ω          [basin 무관, 가장 빨리 결과]
 └─> E2  hybrid runner
          └─> E3  P2: stage1 -> stage2  [basin 안전, 본 실험]
E4  planonly 확보(4a 확인 → 없으면 4c fork)
          └─> P3: aux 추가 효과
E1 + E3 + E4 ─> E5  세 조건 종합 (+ probe planner)
```

**다음 액션 두 개**

1. 학습 머신에 `para_ssr_60ep_planonly` work_dir이 남아 있는지 확인 → E4 분기 결정.
2. `ssr` env에서 E0 실행.

E1과 E3는 서로 독립이므로 E0만 통과하면 병렬로 갈 수 있다.
