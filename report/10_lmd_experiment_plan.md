# 실험 계획 #10 — LMD 차용 BEV 분해 실험 (inference-only)

**방법론 근거:** [report #09](09_lmd_planning_centric_bev.md)
**코드:** [`tools/lmd/`](../tools/lmd/) — 브랜치 `lmd-bev-analysis`
**한 줄 목표:** *"BEV의 어느 부분이 planning에 실제로 쓰이고, 어느 부분이 오히려 방해하는지를 100×100 지도로 분리한다."*

> **학습은 없다.** 아래 전부 `/data2/byounggun/rideflux`의 기존 체크포인트로 nuScenes **val set inference**만 돌린다. 새 weight를 만들지 않는다. 그 제약이 무엇을 못 하게 만드는지는 §3에 명시했다.

---

## 1. 가진 자산

rsync 완료(`EXIT:0`). 아래 3개가 **전부**다.

| run | epochs | task_loss_weight | `load_from` | 우리 조건 |
|---|---:|---|---|---|
| `para_ssr_60ep` | 60 | plan 1 / det 1 / motion 1 / map 1 + GradBalancer | `None` → **독립** | **둘 다** |
| `para_ssr_stage1` | 48 | **plan 0** / det 1 / motion 1 / map 1 | `None` → **독립** | **aux만** |
| `para_ssr_stage2` | 12 | plan 1 / det 1 / motion 1 / map 1 + GradBalancer | `stage1/latest.pth` → **fork** | stage1 + planning |

세 run 모두 `occ_head=None` → **aux = detection + motion + vector map**. 전부 `*_ema.pth` 사용.

**여기서 나오는 두 개의 사실이 계획 전체를 결정한다.**

1. **`stage1` → `stage2`는 fork다.** 같은 basin이 구조적으로 보장되므로 두 모델의 weight를 섞는 분석(report #09 §4.2)이 **이 쌍에서만 유효**하다.
2. **`60ep`은 둘 다와 독립이다.** `60ep`을 stage1/stage2와 weight 수준에서 섞으면 안 된다. 비교는 공간 지도와 스칼라로만.

---

## 2. 무엇을 어디에 먹일 수 있는가

`stage1`의 planner는 **학습되지 않았다** (plan=0으로 48 epoch, weight decay만 받아 초기 norm의 ~85%). 그래서 "어떤 BEV를 어떤 planner로 읽을 것인가"를 먼저 정해야 한다.

| BEV 출처 | 읽는 planner | 유효 | 답하는 것 |
|---|---|:---:|---|
| `60ep` | `60ep` | ✅ | "둘 다" 조건의 planning-centric BEV |
| `stage2` | `stage2` | ✅ | fork 후 조건 |
| `stage1` | `stage1` | ❌ | planner 미학습 |
| **`stage1`** | **`stage2`** | ✅ | **aux만으로 만든 BEV를 학습된 planner가 읽으면 무엇이 보이나** ← fork라서 가능 |
| `stage1` | `60ep` | ❌ | 다른 basin |

**4번째 줄이 핵심이다.** report #09에서 "planner probe를 따로 학습시켜야 한다"고 썼는데, **fork 덕분에 그럴 필요가 없다.** `stage2`의 planner가 `stage1`의 BEV에 대한 정당한 학습된 reader다. 학습 0으로 해결된다.

---

## 3. 이 제약으로 못 하는 것 — 먼저 명시

**`planonly` 체크포인트가 없으므로 "aux 유무" 직접 대조(report #09 §4.2의 원래 목표)는 지금 불가능하다.** 그건 `plan-shaped BEV`와 `plan+aux-shaped BEV`를 같은 basin에서 비교해야 하는데, 그런 쌍이 없고 만들려면 학습이 필요하다.

**대신 부호가 반대인 같은 질문을 한다.** `stage1`→`stage2`는 *aux로 만든 BEV에 planning supervision을 얹었을 때* BEV가 어디로 움직이는가다. 이게 오히려 정의를 직접 준다:

> **planning-centric BEV := planning supervision이 실제로 바꿔놓은 BEV 영역**
> **방해되는 BEV := aux가 써넣었는데 planning supervision이 되돌리려 한 영역** (= `σ^Δ`가 그 방향을 가리키는 cell)

"aux를 붙였더니 어떻게 됐나"는 답 못 하지만, **"planning이 aux가 만든 BEV의 무엇을 고치려 했나"**는 답한다. 논문 서사로는 이쪽이 더 직접적이다.

---

## 4. 실험

### E0 · 실제 모듈에서 선형화 재검증 — **게이트**

| | |
|---|---|
| 목적 | `tools/lmd/`의 검증은 op 스택 **복제본**에서 통과했다. 실제 mmcv 모듈에서도 성립하는지 |
| 방법 | `verify_lmd_linearisation.py`를 실제 `SSRPerceptionTransformer` / `ParaSSRHead` import 버전으로 포팅해 `ssr` env에서 실행 |
| **판정** | `sum_cells C_i + b == trajectory` 상대오차 **< 1e-5 (fp32)**. 실패하면 어느 op가 새는지 이진탐색 — 아래 전부 중단 |
| 소요 | 반나절 |
| 학습 | 없음 |

### Tier A — 모델별 독립 readout (basin 무관, 체크포인트 3개 전부)

#### E1 · planning-centric BEV 지도

| | |
|---|---|
| 입력 | `60ep/epoch_60_ema` (자기 planner), `stage2/epoch_12_ema` (자기 planner), `stage1/epoch_48_ema` (**stage2 planner로 읽음**) |
| 방법 | 얼린 그래프에서 adjoint(`grad × input`) 12회/샘플 → `C_i` → `π_i`. `b`는 BEV=0 frozen forward 1회 → `ρ` |
| 산출 | 조건별 `π` 지도(100×100), `ρ`, participation ratio, PPA |
| **판정** | 정확성 회귀 `Σ_i C_i + b = τ` < 1e-5. 그 다음 지도/스칼라 비교 |
| 비교 규칙 | 세 조건은 **공간 지도와 스칼라로만** 비교. 채널 단위 비교 금지 (독립 학습이라 채널 기저가 다름) |
| 소요 | val 500샘플 × 3조건 ≈ 1.5시간 (+ 후킹 1~2일) |

#### E2 · aux가 읽는 곳 `ω`

| | |
|---|---|
| 입력 | `60ep`, `stage1`, `stage2` — 각자의 det/map head |
| 방법 | aux head 출력은 개수가 많아 cell별 adjoint가 비싸다. **스칼라 요약**의 adjoint를 쓴다: det는 confident query의 max-class logit 합, map은 classification logit 합 → backward 1~2회/샘플 |
| 산출 | `ω` 지도, `cos(π, ω)`, 4분면 질량 비율 |
| 읽는 법 | `ω`만 높은 칸 = **planner가 안 보는데 aux가 쓰는 BEV 용량**. `π`만 높은 칸 = planner 전용 (ego shortcut 의심 지대) |

### Tier B — fork 쌍 weight-delta (`stage1` ↔ `stage2`만)

#### E3 · hybrid forward 러너 구현

| | |
|---|---|
| 방법 | `replica.hybrid` 구조를 실제 모델에. group 8개: `embed · enc0 · enc1 · enc2 · tokenl · latent · way · mlp` |
| 필요한 후킹 | `MSDeformableAttention3D` / `TemporalSelfAttention`에 `sampling_locations`·`attention_weights` 캐시-재사용 (`bev_mask`·`indexes`·`count`는 `point_sampling` 유래 기하학 전용이라 query를 0으로 만들어도 안전) |
| **판정** | `assign = 전부 A`일 때 원 모델 출력 bit-exact 재현 |
| 소요 | 2~3일 |

#### E4 · 부호 있는 지도 — **본 실험**

| | |
|---|---|
| 입력 | `stage1/epoch_48_ema` (= P), `stage2/epoch_12_ema` (= A) |
| 방법 | ① telescoping 양방향 → `δ_g` ② 대표 샘플 ~30개에 exact Shapley(256 coalition) → `φ_g` ③ **BUILD/READ 분리**: reader를 A에 고정하고 builder만 P→A → `Δτ_bev` ④ cell 분해 후 GT 오차 방향 사영 → `σ^Δ_i` |
| 산출 | **100×100 부호 있는 지도** (도움/방해), group별 Shapley, `H` = 방해 비율 |
| **판정 (정확성)** | `Σ_g δ_g = out_A − out_P`, `Σ_i σ^Δ_i = ⟨Δτ_bev, ê⟩` — fp32에서 < 1e-5 |
| **판정 (해석)** | 두 순서 스프레드를 `|Δτ|` 대비 %로 함께 보고. group 숫자는 **Shapley만 인용** |
| 소요 | telescoping 9 forward/샘플 → val 전체 ~1시간. Shapley는 대표 샘플만 |

---

## 5. 지표

| 기호 | 정의 | 읽는 법 |
|---|---|---|
| `C_i` | `J_i · bev_i` — cell i가 궤적에 기여한 12차 벡터 | `Σ_i C_i + b = τ` (잔차 0) |
| `π_i` | `‖C_i‖₁ / Σ‖C_j‖₁` | planning 기여 지도. **planning-centric BEV := π의 유효 지지집합**, participation ratio `(Σπ)²/Σπ²` |
| `b`, `ρ` | `b` = BEV를 0으로 둔 frozen forward, `ρ = ‖b‖/(‖b‖+‖Σ C_i‖)` | **장면과 무관하게 나오는 궤적 비율.** ablation 없이 ego-shortcut 측정 |
| `ω_i` | det/map head 스칼라 요약의 BEV cell 분해 | aux가 읽는 곳 |
| `σ^Δ_i` | `⟨J^A_i · Δbev_i, ê⟩`, `ê = (τ_P−τ_gt)/‖·‖` | **음수 = 도움, 양수 = 방해.** `Σ_i σ^Δ_i = ⟨Δτ_bev, ê⟩` 정확 |
| `H` | `Σ_{σ^Δ>0} σ^Δ_i / \|Σ_i σ^Δ_i\|` | 변화분 중 방해 비율 |
| `φ_g` | group g의 Shapley 값 | 변화가 **어느 모듈에서** 만들어졌나 |
| PPA | `(Σ_{i∈GT} π_i) / (\|GT\|/10000)` | GT 박스·drivable에 planning 질량이 균등 대비 몇 배 |

---

## 6. 산출물 (논문 그림 후보)

1. **Fig A** — `σ^Δ` 부호 지도 100×100, GT 박스·차선 오버레이. 빨강 = planning supervision이 되돌리려 한 aux 흔적, 파랑 = 강화한 곳. **핵심 그림.**
2. **Fig B** — `π` vs `ω` 4분면 산점도 + 사분면별 대표 장면 (E1+E2).
3. **Fig C** — group별 Shapley 막대 (변화가 encoder 어느 층에서 만들어졌나).
4. **Table 1** — 세 조건 × {`π` 지지집합, `ρ`, PPA, `cos(π,ω)`} — 공간·스칼라 비교만.
5. **Table 2** — 정확성 회귀: 각 분해의 `Σ` vs 출력 상대오차. **이게 있어야 나머지 숫자가 신뢰된다.**

---

## 7. 리스크

| 리스크 | 영향 | 대응 |
|---|---|---|
| 실제 모듈에서 선형화가 안 깨끗함 | 전부 중단 | **E0가 게이트** |
| `planonly` 부재 | "aux 유무" 직접 대조 불가 | §3대로 질문을 뒤집는다. 학습이 가능해지면 `60ep`의 `epoch_40`에서 aux-off 갈래를 20 epoch fork하는 게 최선(basin 원천 차단) — **지금은 범위 밖** |
| `60ep`을 fork 쌍과 섞고 싶은 유혹 | 무의미한 숫자 | 비교는 공간 지도·스칼라로만. weight-delta는 fork 쌍 전용 |
| 경로 의존성 (실측: `\|Δτ\|`의 25%) | group 숫자 왜곡 | group은 **Shapley로만** 보고 |
| 얼린 routing — value 경로만 귀속 | 해석 범위 축소 | 결과 문장을 *"거기를 보기로 한 상태에서 내용을 누가 바꿨나"*로 한정. 별도로 두 모델의 frozen `sampling_locations` 차이를 "시선이 얼마나 옮겨졌나" 지표로 보고 |
| stage1/stage2의 짧은 fork (12 epoch) | `Δbev`가 작아 SNR 낮을 수 있음 | `‖Δbev‖/‖bev‖`를 먼저 측정. 너무 작으면 stage2의 중간 epoch(예: 4, 8)도 함께 써서 궤적을 본다 — **체크포인트가 이미 다 있다** |
| EMA/raw 혼용 | 비교 무효 | 전부 `*_ema.pth` |

---

## 8. 순서

```
E0 (게이트: 실제 모듈 선형화)
 ├─> E1  π, ρ  (3조건)          ┐
 ├─> E2  ω, cos(π,ω)            ┴─ Tier A, basin 무관, 먼저 결과 나옴
 └─> E3  hybrid runner
          └─> E4  σ^Δ 부호 지도  ── Tier B, fork 쌍, 본 실험
```

**다음 액션:** `ssr` env에서 E0. 통과하면 Tier A와 E3를 병렬로.

**보너스로 이미 확보된 것:** stage2는 12 epoch 전 구간의 체크포인트가 다 있다(`epoch_1` ~ `epoch_12`). fork 이후 BEV가 **epoch별로 어떻게 이동했는지** 궤적으로 볼 수 있다 — 한 시점의 스냅샷보다 훨씬 강한 증거이고, 추가 학습 없이 공짜다.
