# 실험 계획 #10 — LMD 차용 BEV 분해 실험 (inference-only)

**방법론 근거:** [report #09](09_lmd_planning_centric_bev.md)
**코드:** [`tools/lmd/`](../tools/lmd/) — 브랜치 `lmd-bev-analysis`
**한 줄 목표:** *"BEV의 어느 부분이 planning에 실제로 쓰이고, 어느 부분이 오히려 방해하는지를 100×100 지도로 분리한다."*

> **학습은 없다.** 전부 `/data2/byounggun/rideflux`의 기존 체크포인트로 nuScenes **val set inference**만 돌린다.

---

## 1. 가진 자산

| run | model class | epochs | lr | task_loss_weight | `load_from` | 조건 |
|---|---|---:|---:|---|---|---|
| `ssr_noffp_2gpu_b4` | `SSR` / `SSRHead` | 12 | 5e-5 | planning만 (aux head 자체가 없음), FFP off | `None` | **planning만** |
| `para_ssr_stage1` | `ParaSSR` / `ParaSSRHead` | 48 | 2e-4 | **plan 0** / det 1 / motion 1 / map 1 | `None` | **aux만** |
| `para_ssr_stage2` | `ParaSSR` / `ParaSSRHead` | 12 | 2e-4 | plan 1 / det 1 / motion 1 / map 1 | **`stage1/latest.pth`** | stage1 + planning |
| `para_ssr_60ep` | `ParaSSR` / `ParaSSRHead` | 60 | 2e-4 | plan 1 / det 1 / motion 1 / map 1 | `None` | **둘 다** |

**네 run 모두 epoch 1부터 끝까지 `*_ema.pth`가 전부 있다.** (12 / 48 / 12 / 60개) 이게 §4의 E5를 공짜로 만든다.
`occ_head=None`이므로 **aux = detection + motion + vector map**.

### 1.1 `ssr_noffp`가 무엇이고 무엇이 아닌가

이건 report #08의 **SSR-noFFP 베이스라인**이지, `PARA_SSR_e2e_60ep_planonly.py`가 **아니다.** 차이가 결과 해석을 좌우한다.

| | `ssr_noffp` (가진 것) | `planonly` config (없는 것) |
|---|---|---|
| model class | `SSRHead` — 안 쓰이는 det/map/motion branch + FFP plumbing 잔존 | `ParaSSRHead` — slim |
| aux head | **아예 없음** (31.4M) | 존재하되 BEV로 gradient 차단 (41.4M) |
| lr / epochs | 5e-5 / 12 | 2e-4 / 60 |
| temporal queue | history 2 + current + **미사용 future 1** | history 2 + current |
| 학습 sample 모집단 | future frame GT 존재 조건으로 필터됨 | 필터 없음 |

report #08이 `planonly` config를 따로 만든 이유가 정확히 이것이다 — **SSR-noFFP와 비교하면 한 번에 네 가지가 같이 움직인다.** 그래서:

- ✅ **기능적 대조는 유효**: `π` / `ρ` / PPA 같은 **공간 지도와 스칼라** 비교. basin 가정이 필요 없다.
- ❌ **weight-delta 분해는 불가**: 모델 클래스가 다르고 독립 학습이라 같은 basin이 아니다. `σ^Δ` 부호 지도를 이 쌍에 쓸 수 없다.

### 1.2 fork 쌍

`stage2`는 `stage1` epoch 48에서 갈라져 나왔다 → **같은 basin이 구조적으로 보장**. weight를 섞는 분석(report #09 §4.2)이 **이 쌍에서만** 유효하다. `60ep`과 `ssr_noffp`는 둘 다 독립이므로 weight 수준에서 섞지 않는다.

---

## 2. 어떤 BEV를 어떤 planner로 읽는가

`stage1`의 planner는 **학습되지 않았다** (plan=0으로 48 epoch, weight decay만 받아 초기 norm의 ~85%).

| BEV 출처 | 읽는 planner | 유효 | 답하는 것 |
|---|---|:---:|---|
| `ssr_noffp` | `ssr_noffp` | ✅ | **planning만** 조건 |
| `60ep` | `60ep` | ✅ | **둘 다** 조건 |
| `stage2` | `stage2` | ✅ | fork 후 조건 |
| `stage1` | `stage1` | ❌ | planner 미학습 |
| **`stage1`** | **`stage2`** | ✅ | **aux만으로 만든 BEV를 학습된 planner가 읽으면** ← fork라서 가능 |
| 서로 다른 독립 run 간 교차 | | ❌ | 다른 basin |

5번째 줄 덕분에 **planner probe를 따로 학습시킬 필요가 없다.** 학습 0으로 "aux만" 조건도 읽힌다.

읽기 코드는 `SSRHead`와 `ParaSSRHead` **둘 다** 지원해야 한다. report #08이 확인했듯 planning path 연산은 두 클래스가 동등하다 (`bev_embedding → navi_se → tokenlearner → latent_decoder → way_decoder → ego_fut_decoder`). 차이는 `SSRHead`에 남은 dead branch뿐이고 forward에 안 탄다.

---

## 3. 되는 것과 안 되는 것

| 질문 | 방법 | 가능? |
|---|---|:---:|
| 세 조건의 BEV가 planning 중심성에서 어떻게 다른가 | Tier A — 조건별 `π`/`ρ`/PPA 지도·스칼라 | ✅ |
| aux를 붙이면 **어느 cell이** 도움/방해가 되는가 | Tier B weight-delta — 같은 basin 쌍 필요 | ❌ 쌍 없음 |
| planning supervision이 aux-shaped BEV의 **어느 cell을** 바꿨고 도움됐나 | Tier B, fork 쌍 | ✅ |
| 한 모델 안에서 planner와 aux가 같은 곳을 보나 | Tier A — `cos(π, ω)` | ✅ |

**"aux 유무의 cell 단위 부호 지도"는 못 만든다.** 대신 fork 쌍이 부호가 반대인 같은 질문에 답한다:

> **planning-centric BEV := planning supervision이 실제로 바꿔놓은 BEV 영역**
> **방해되는 BEV := aux가 써넣었는데 planning supervision이 되돌리려 한 영역**

---

## 4. 실험

### E0 · 실제 모듈에서 선형화 재검증 — **게이트**

`tools/lmd/`의 검증은 op 스택 **복제본**에서 통과했다. 실제 mmcv 모듈에서 재확인한다.

| | |
|---|---|
| 방법 | `verify_lmd_linearisation.py`를 실제 `SSRPerceptionTransformer` / `ParaSSRHead` / `SSRHead` import 버전으로 포팅해 `ssr` env에서 실행 |
| **판정** | `Σ_i C_i + b == τ` 상대오차 **< 1e-5 (fp32)**. 실패 시 어느 op가 새는지 이진탐색 — 아래 전부 중단 |
| 소요 | 반나절 |

### Tier A — 조건별 독립 readout (basin 무관, 4개 체크포인트 전부)

#### E1 · planning-centric BEV 지도 `π`, ego-shortcut `ρ`

| | |
|---|---|
| 입력 | `ssr_noffp/epoch_12_ema` · `stage1/epoch_48_ema`(stage2 planner로) · `stage2/epoch_12_ema` · `60ep/epoch_60_ema` |
| 방법 | 얼린 그래프에서 adjoint 12회/샘플 → `C_i` → `π_i`. `b`는 BEV=0 frozen forward 1회 → `ρ` |
| 산출 | 조건별 `π` 지도(100×100), `ρ`, participation ratio, PPA |
| **판정** | 정확성 회귀 `Σ_i C_i + b = τ` < 1e-5 |
| **헤드라인** | **`ρ` 비교 — aux를 붙이면 ego-shortcut이 줄어드는가.** ablation 없이 직접 측정되는 값 |
| 소요 | val 500샘플 × 4조건 ≈ 2시간 (+ 후킹 1~2일) |

#### E1b · epoch-matched 대조 — **교란변수 하나 제거**

`ssr_noffp`는 12 epoch, `60ep`은 60 epoch이라 §1.1의 차이 중 "5배 더 학습" 하나가 결과를 설명해버릴 수 있다. **`60ep/epoch_12_ema`가 있으므로** epoch 수를 맞춘 대조를 같이 낸다. lr(5e-5 vs 2e-4)과 모델 클래스는 여전히 다르다 — 완전한 통제가 아니라 confound 하나를 뺀 것.

#### E2 · aux가 읽는 곳 `ω`

| | |
|---|---|
| 입력 | `stage1`, `stage2`, `60ep` (각자의 det/map head). `ssr_noffp`는 aux head가 없어 `ω` 정의 안 됨 |
| 방법 | aux head 출력은 개수가 많아 cell별 adjoint가 비싸다. **스칼라 요약**의 adjoint를 쓴다 — det는 confident query의 max-class logit 합, map은 classification logit 합 → backward 1~2회/샘플 |
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
| 산출 | **100×100 부호 있는 지도**, group별 Shapley, `H` = 방해 비율 |
| **판정 (정확성)** | `Σ_g δ_g = out_A − out_P`, `Σ_i σ^Δ_i = ⟨Δτ_bev, ê⟩` — fp32에서 < 1e-5 |
| **판정 (해석)** | 두 순서 스프레드를 `\|Δτ\|` 대비 %로 함께 보고. group 숫자는 **Shapley만 인용** |
| 사전 확인 | fork가 12 epoch뿐이라 `‖Δbev‖/‖bev‖`부터 잰다. 너무 작으면 E5로 보완 |

### E5 · epoch 궤적 — **추가 학습 없이 공짜, 그리고 confound를 녹인다**

| | |
|---|---|
| 근거 | 네 run 모두 epoch 1~끝까지 EMA 체크포인트가 다 있다 |
| 방법 | E1의 `ρ`, participation ratio, PPA를 **epoch의 함수로** 측정 (예: 매 4 epoch) |
| 왜 중요한가 | 스냅샷 한 장이 아니라 곡선이 나온다. `ssr_noffp`와 `60ep`의 `ρ` 곡선이 **초반부터 갈라져 끝까지 유지되면** "5배 더 학습해서 그렇다"는 설명이 배제된다. §1.1의 confound를 부분적으로 해소하는 유일한 수단 |
| 추가로 | fork 이후 `stage2`가 epoch 1→12로 가며 BEV가 어디로 이동하는지 궤적 (E4의 시계열 버전) |
| 소요 | 조건당 epoch 12~15포인트 × 샘플 200 → 반나절씩 |

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

## 6. 산출물

1. **Fig A** — `σ^Δ` 부호 지도 100×100, GT 박스·차선 오버레이 (E4). **핵심 그림.**
2. **Fig B** — `π` vs `ω` 4분면 산점도 + 사분면별 대표 장면 (E1+E2).
3. **Fig C** — `ρ` / PPA의 **epoch 곡선**, 조건별 (E5). confound 해소의 근거.
4. **Fig D** — group별 Shapley 막대 (E4).
5. **Table 1** — 네 조건 × {`π` 지지집합, `ρ`, PPA, `cos(π,ω)`}. epoch-matched 행 포함 (E1+E1b).
6. **Table 2** — 정확성 회귀: 각 분해의 `Σ` vs 출력 상대오차. **이게 있어야 나머지 숫자가 신뢰된다.**

---

## 7. 리스크

| 리스크 | 영향 | 대응 |
|---|---|---|
| 실제 모듈에서 선형화가 안 깨끗함 | 전부 중단 | **E0가 게이트** |
| `ssr_noffp` vs PARA 비교에 confound 4개 (모델 클래스·lr·epoch·sample 모집단) | Table 1 해석 약화 | **E1b**(epoch 맞춤)와 **E5**(epoch 곡선)로 두 개를 줄인다. 나머지 둘은 못 없애므로 **결론 문장에 명시**하고, 차이가 작으면 aux 효과라고 주장하지 않는다 |
| `planonly` 부재 | aux 유무의 cell 단위 부호 지도 불가 | §3대로 질문을 뒤집는다. 학습이 가능해지면 `60ep`의 `epoch_40`에서 aux-off 갈래 20 epoch fork가 최선 — **지금은 범위 밖** |
| 독립 run을 weight 수준에서 섞고 싶은 유혹 | 무의미한 숫자 | weight-delta는 **fork 쌍 전용**. 나머지는 공간 지도·스칼라만 |
| 경로 의존성 (실측: `\|Δτ\|`의 25%) | group 숫자 왜곡 | group은 **Shapley로만** 보고 |
| 얼린 routing — value 경로만 귀속 | 해석 범위 축소 | 결과 문장을 *"거기를 보기로 한 상태에서 내용을 누가 바꿨나"*로 한정. 별도로 frozen `sampling_locations` 차이를 "시선이 얼마나 옮겨졌나" 지표로 보고 |
| fork가 12 epoch뿐 → `Δbev` 작을 수 있음 | E4 SNR | E4 전에 `‖Δbev‖/‖bev‖` 측정. 작으면 E5 시계열로 보완 |
| EMA/raw 혼용 | 비교 무효 | 전부 `*_ema.pth` |

---

## 8. 순서

```
E0 (게이트: 실제 모듈 선형화)
 ├─> E1  π, ρ  (4조건)  + E1b epoch 맞춤   ┐
 ├─> E2  ω, cos(π,ω)                       ┴ Tier A, basin 무관, 먼저 결과
 ├─> E5  epoch 곡선  (E1 코드 재사용)
 └─> E3  hybrid runner
          └─> E4  σ^Δ 부호 지도             ── Tier B, fork 쌍, 본 실험
```

**다음 액션:** `ssr` env에서 E0. 통과하면 Tier A와 E3를 병렬로.
E1의 코드가 그대로 E1b/E5가 되므로, E1을 먼저 끝내면 곡선까지 거의 공짜로 따라온다.
