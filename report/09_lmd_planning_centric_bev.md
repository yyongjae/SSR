# 실험 보고서 #09 — LMD로 "planning-centric BEV"를 정의할 수 있는가

**차용 논문:** Layer-Wise Modality Decomposition for Interpretable Multimodal Sensor Fusion, NeurIPS 2025 ([arXiv:2511.00859](https://arxiv.org/pdf/2511.00859)) — `craft_좋은bevfeature란.pdf`의 첫 번째 논문
**대상:** `origin/para` 기준 PARA-SSR (`para_ssr.py`, `para_ssr_head.py`)
**질문:** aux만 / planning만 / 둘 다 붙였을 때의 BEV를 LMD 방식으로 분해해서, planning에 실제로 쓰이는 BEV와 방해되는 BEV를 정의할 수 있는가
**결론:** **가능하다.** 세 단계 모두 float64 기계정밀도에서 잔차 0으로 성립하는 것을 수치로 확인했다 (`tools/lmd/verify_lmd_linearisation.py`). 다만 LMD를 그대로 쓰는 게 아니라 **분해축을 바꿔야** 하고, 세 모델 비교에는 아래 §6의 제약이 붙는다.

---

## 1. LMD를 그대로 쓸 수 없는 이유, 그리고 무엇을 바꾸는가

LMD의 분해축은 **입력 모달리티**(cam/lidar/radar)다. PARA-SSR은 카메라 전용이라 그 축이 자명하게 붕괴한다.

하지만 LMD의 실제 기여는 "모달리티"가 아니라 **덧셈으로 들어오는 임의의 source 집합에 대한 잔차 없는 정확 귀속**이다. 그 성질만 가져오면 축은 우리가 정할 수 있고, 이 코드베이스에는 쓸 만한 축이 두 개 있다.

| | LMD 원본 | 여기서 |
|---|---|---|
| 분해 대상 | fusion BEV | `bev_embed [B,10000,256]` 및 `ego_fut_preds` |
| 분해축 (상류) | cam / lidar / radar | **6개 카메라 · prev_bev · BEV query prior · can_bus · bev_pos** |
| 분해축 (하류) | 없음 | **10,000개 BEV cell → 궤적** |
| 읽는 방향 | forward (2차 pass) | 상류는 forward, 하류는 adjoint |

상류 축이 이 모델에서 특히 잘 맞는 이유: `SSR_transformer.py:270`에서 `bev_queries = bev_embedding.weight + can_bus_mlp(can_bus)`로 ego 신호가 **문자 그대로 덧셈 source**로 들어온다. 즉 BEV-Planner가 제기한 "Is Ego Status All You Need?" 질문이 여기서는 ablation이 아니라 **닫힌 형식의 지분 계산**이 된다.

---

## 2. 이 코드베이스가 LMD-선형화 가능한가 — 경로별 확인

LMD가 요구하는 건 "경로상 모든 연산이 (a) 이미 선형이거나, (b) 분해 대상에 대해 상수거나, (c) LMD가 얼려내는 비선형(ReLU/GELU, LN/GN, softmax) 중 하나"라는 것이다. 전부 확인했다.

### 2.1 하류: `bev_embed` → `ego_fut_preds` (`para_ssr_head.py:186-232`)

| 연산 | 위치 | 처리 |
|---|---|---|
| `navi_se` SE gate | `para_ssr_head.py:36-52` | **얼릴 필요 없음.** gate가 nav embedding만으로 만들어져 BEV에 대해 상수 → 이미 정확히 선형 |
| `cat(bev, pos_embd)` | `:205` | 상수 concat |
| TokenLearner `GroupNorm(1,C)` | `tokenlearner.py:31` | μ,σ 고정 → affine |
| TokenLearner `GELU` | `tokenlearner.py:19` | exact GELU는 `x·Φ(x)`라서 Φ 고정 = **정확, bias 0** |
| TokenLearner `softmax` | `tokenlearner.py:49` | `selected` 고정 → `einsum`이 BEV에 선형 |
| latent_decoder ×3 | `SSR_e2e.py:80-91` | MHA softmax 고정, LN μ/σ 고정, FFN ReLU mask 고정 |
| way_decoder ×1 | `SSR_e2e.py:92-105` | query가 학습 embedding(상수), key/value만 BEV 의존 → softmax 고정 후 value에 선형 |
| `ego_fut_decoder` | `:139-143` | Linear-ReLU-Linear-ReLU-Linear, mask 고정 |

**전 구간 선형화 가능. 근사 없음.**

### 2.2 상류: 이미지 → `bev_embed` (`encoder.py` / `spatial_cross_attention.py` / `temporal_self_attention.py`)

| 연산 | 위치 | 처리 |
|---|---|---|
| `sampling_offsets`, `attention_weights` | `spatial_cross_attention.py:345-350`, `temporal_self_attention.py:211-216` | query에서 나오므로 **얼린다**. 고정 후 bilinear 샘플 + 가중합은 `value`(=이미지 feature)에 선형 |
| per-camera scatter | `spatial_cross_attention.py:175-177` | `slots[j, idx] += queries[j,i]` — **카메라별 덧셈 누적**이라 카메라 귀속이 구조적으로 정확 |
| `/count` | `:179-182` | `bev_mask` 기반 = 기하학만의 함수, query 무관 → 상수 |
| residual | `:183`, `temporal_self_attention.py:277` | `output + identity` — 순수 덧셈이라 source 분해가 그대로 관통 |
| `prev_bev` 스택 | `encoder.py:207-208` | `stack([prev_bev, bev_query])` — 두 개가 각각 독립 source |
| `bev_pos` | `encoder.py:366-367` | self_attn의 q/k에만 덧셈으로 진입 (cross_attn의 `query_pos`는 None) |
| LN ×3/layer, FFN ReLU | `encoder.py:378-405` | 표준 LMD 처리 |

**전 구간 선형화 가능.**

### 2.3 수치 검증 결과

`tools/lmd/verify_lmd_linearisation.py` — 위 op 스택을 실제 배선 그대로 복제해 float64로 검증 (mmcv-full이 이 머신에 없어 실제 모듈 import 대신 복제; 각 모듈에 출처 파일/줄을 주석으로 달아두었다).

```
DOWN  bev_embed -> ego_fut_preds, split over 10,000 BEV cells
  PASS  frozen MHA == nn.MultiheadAttention                  rel 1.43e-15
  PASS  frozen forward reproduces the real output            rel 0.00e+00
  PASS  frozen map is exactly affine in bev                  rel 2.57e-15
  PASS  sum_cells C_i + b == trajectory  (residual-free)     rel 1.19e-15

UP    image features -> bev_embed, split over 10 additive sources
  PASS  sum_sources h^s + b == bev_embed  (residual-free)    rel 8.28e-16

COMPOSE  source -> bev -> trajectory
  PASS  sum_sources (routed to plan) + b == trajectory       rel 4.83e-15
```

**한 가지 함정이 실제로 있었다.** 1차 pass에서 캐싱한 상태값을 `detach()` 하지 않으면 μ/σ와 softmax를 통해 gradient가 새고, 분해가 출력과 맞지 않는다 — 같은 스크립트가 detach 없이는 **상대오차 4.0e-1로 실패**하고, detach를 넣으면 1.2e-15로 통과한다. LMD 논문의 "고정된 선형 스위치"는 구현상 이 한 줄이다 (`tools/lmd/lmd_core.py:31-40`).

### 2.4 읽는 방향은 단계마다 반대다

같은 선형사상을 두 방향으로 읽어야 한다. 이건 구현상 중요한 포인트다.

- **상류**: source 10개 → 출력 `[10000,256]`. **forward (LMD 2차 pass)**가 싸다. 인코더 forward 11회/샘플.
- **하류**: source 10,000개 → 출력 12개. **adjoint**가 싸다. 얼린 그래프 위에서의 `grad × input`인데, 얼린 그래프에서는 이게 1차 근사가 아니라 **정확한 기여값**이다. backward 12회/샘플.

---

## 3. 정의: planning-centric BEV, 그리고 방해되는 BEV

하류 분해가 주는 것:

$$\tau \;=\; \sum_{i=1}^{10000} C_i \;+\; b, \qquad C_i = J_i\,\mathrm{bev}_i \in \mathbb{R}^{12}\ (6\text{ step} \times 2)$$

잔차가 0이므로 아래 정의들이 "대략"이 아니라 **항등식**으로 성립한다.

**M1 · planning 기여 지도** `π_i = ‖C_i‖₁ / Σ_j ‖C_j‖₁` → 100×100 지도. gradient saliency와 달리 합이 정확히 1이고 출력을 재구성한다.
→ **planning-centric BEV := π의 유효 지지집합** (participation ratio `(Σπ)²/Σπ²`, 또는 상위 q% 질량).

**M2 · scene-independence 비율** `ρ = ‖b‖ / (‖b‖ + ‖Σ_i C_i‖)`. `b`는 BEV를 0으로 두고 스위치를 고정한 forward, 즉 **장면과 무관하게 나오는 궤적 성분**. ρ가 높으면 planner가 BEV를 안 읽고 있는 것 — ablation 없이 직접 측정된다.

**M3 · source 지분** (§2.2의 10개 축, 그리고 §2.4의 COMPOSE로 궤적까지 라우팅한 지분)
`α_cam = Σ_c ‖routed cam_c‖ / Σ_all`, `α_ego = ‖routed can_bus‖ / Σ_all`.
→ "이 프레임의 planning 결정에 CAM_FRONT가 41%, can_bus가 33%, prev_bev가 12%, 학습된 BEV prior가 14% 기여했다" — LMD의 결과 형식 그대로다. **이게 헤드라인 숫자다.**

**M4 · Planning–Perception Alignment** `PPA = (Σ_{i∈GT} π_i) / (|GT|/10000)`. GT 박스·drivable area 안에 planning 질량이 균등분포 대비 몇 배 몰려 있는가. 1보다 크면 planner가 실제 객체/도로를 본다.

**M5 · 방해되는 BEV.** `e = τ_cmd − τ_gt`, `ê = e/‖e‖` 로 두고
$$\sigma_i = \langle C_i, \hat e\rangle, \qquad \sum_i \sigma_i + \langle b,\hat e\rangle = \|e\| \ \text{(정확)}$$
- `σ_i > 0` → 그 cell이 궤적을 GT에서 **밀어내고** 있다 = **방해 BEV**
- `σ_i < 0` → 교정에 기여
- 지표: 방해 질량 비율 `H = Σ_{σ>0} σ_i / ‖e‖`, 그리고 그 공간 분포

부호 있는 분해가 가능한 건 오로지 잔차가 0이기 때문이다. attention rollout이나 gradient saliency로는 이 정의를 못 만든다 — 합이 오차에 맞지 않으니 "방해"의 기준선이 없다.

**M6 · aux와의 겹침 (세 조건 비교의 핵심).** aux head가 있는 모델에서는 det/map/occ 출력도 같은 adjoint로 BEV에 분해해 aux 관련도 `ω_i`를 얻는다. 그러면 cell을 아래처럼 나눌 수 있다:

| cell 유형 | 조건 | 해석 |
|---|---|---|
| **공유-유용** | `π` 높음 · `ω` 높음 · `σ<0` | aux가 써넣은 내용이 planning에 실제로 도움 |
| **aux-방해** | `π` 높음 · `ω` 높음 · `σ>0` | aux가 써넣은 내용을 planner가 잘못 씀 |
| **aux-무관** | `π` 낮음 · `ω` 높음 | BEV 용량을 planning이 안 쓰는 데 소모 |
| **plan 전용** | `π` 높음 · `ω` 낮음 | ego shortcut이 여기 몰린다 |

지표: `overlap = cos(π, ω)`, `aux_attributable_interference = Σ_{i: ω_i > q90} σ_i^+ / Σ_i σ_i^+`.
**"방해되는 BEV"의 구체적 답은 aux-방해 칸이다.**

**M7 · 공짜 counterfactual.** 스위치 고정 하에서 사상이 affine이므로 cell 집합 `S`를 0으로 만든 궤적은 재-forward 없이 `τ_S = τ − Σ_{i∈S} C_i`. 다만 마스킹은 동작점을 옮기므로 **실제 재-forward와 반드시 대조**하고, 그 차이(선형화 갭)를 함께 보고해야 한다. 갭 자체가 "국소 선형 모델이 얼마나 멀리까지 유효한가"의 측정치다.

---

## 4. 세 조건은 이미 config로 존재한다

| 조건 | config | 설정 |
|---|---|---|
| **planning만** | `PARA_SSR_e2e_60ep_planonly.py` | `grad_balance target = plan 1.0 / det 0 / map 0` — aux head는 존재하되 BEV로 gradient가 안 간다 |
| **aux만** | `PARA_SSR_stage1_detmap.py` | `plan 0.0 / det 1.0 / motion 1.0 / map 1.0` |
| **둘 다** | `PARA_SSR_e2e_60ep.py` | GradBalancer plan 0.4 / det 0.2 / map 0.2 / occ 0.2 |

`planonly`가 좋은 대조군인 이유는 report #08과 그 config docstring에 이미 정리돼 있다 — 코드 경로·파라미터 수·clip norm이 동일하고 **aux→BEV gradient 하나만** 다르다.

---

## 5. 실행 계획

1. `tools/lmd/lmd_core.py` — 완료. 선형화 primitive + forward/adjoint 리더.
2. `tools/lmd/verify_lmd_linearisation.py` — 완료. **`ssr` env에서 실제 모듈로 한 번 더 돌린 뒤** 체크포인트 숫자를 믿을 것.
3. 실제 모듈 후킹 (미구현):
   - `MSDeformableAttention3D.forward` / `TemporalSelfAttention.forward`에 `sampling_locations`·`attention_weights` 캐시-재사용 훅. `bev_mask`·`indexes`·`count`는 기하학 전용이라 query를 0으로 만들어도 안전하다 (`point_sampling` 유래).
   - head 경로의 LN/GN/ReLU/GELU/softmax는 §2.3의 detach 방식으로 교체.
   - `get_bev_features`에 source override 인자 추가 (`bev_prior`/`can_bus`를 개별로 0으로 만들 수 있어야 함).
4. 드라이버: val 샘플 순회 → per-sample `π, σ, ω, α, ρ` npz 덤프 → 집계/플롯.

**비용.** 샘플당 인코더 forward 11회 + backward 12회. A6000에서 대략 2–3 s/샘플, val 500샘플이면 ~25분/모델. `parts`는 `[1,10000,256]` fp32 × 11 = 112 MB로 메모리 여유 있음. `obtain_history_bev`의 `no_grad`는 **건드릴 필요 없다** — 상류는 forward 분해라 gradient가 필요 없다.

---

## 6. 반드시 같이 보고해야 할 한계

1. **얼린 routing 문제 (가장 중요).** LMD는 **value 경로로만** 귀속한다. "어디를 볼지" 정하는 query 경로는 source가 아니라 스위치로 청구된다. 그래서 결과의 정확한 문장은 *"모델이 거기를 보기로 한 상태에서, 그 내용을 누가 공급했는가"*이지 *"누가 거기를 보게 만들었는가"*가 아니다. LMD 원본의 fusion 세팅보다 여기서 더 큰 문제인데, deformable attention의 sampling offset이 실제 정보를 꽤 지고 있기 때문이다. 반드시 명시할 것.
2. **모델 간 채널 비교는 무효.** 따로 학습된 세 모델의 BEV 채널 공간은 정렬돼 있지 않다. 비교 가능한 것은 **공간 지도(100×100, 물리적으로 정렬됨)**, **source 지분(의미적으로 정렬됨)**, **스칼라 요약**뿐이다. 채널별 분해를 모델 간에 겹쳐 읽으면 안 된다.
3. **`stage1_detmap`의 planner는 학습되지 않았다.** 해당 config docstring이 직접 밝히듯 planner 전용 모듈 2.42M은 gradient 0에 weight decay만 받아 초기 norm의 ~85%로 끝난다. 따라서 이 모델에서는 M1/M2/M5를 자기 planner로 읽을 수 없다. 두 가지 프로토콜을 구분해야 한다:
   - **(a) 자기 readout** — 각 모델의 자기 planner로. "이 모델의 planner가 무엇을 쓰는가."
   - **(b) 공통 probe** — BEV를 얼리고 동일한 planner probe를 각 BEV 위에 학습. "이 BEV가 planning에 쓸 정보를 얼마나 담고 있는가." `stage1_detmap`에는 (b)만 유효하다.
4. **`stage1_detmap` vs 나머지는 통제된 비교가 아니다.** report #08의 "budget match, not a controlled experiment"에 LR schedule·optimizer state·EMA·RNG·planner 업데이트 횟수(1/5) 다섯 개 교란변수가 이미 정리돼 있다. 깨끗한 쌍은 `planonly` vs `60ep`뿐이다.
5. **EMA/raw 통일.** 최종 eval이 `epoch_N_ema.pth`로 나가므로 분석도 EMA로 통일할 것.
6. **기존 진단과의 관계.** `para_ssr.py`에는 이미 `_bev_grad_norms`(task별 BEV gradient norm + task 간 cosine)와 `_representation_metrics`(`tok_sim`, `bev_std`, `tok_cover`)가 있다. 이들은 **loss가 BEV를 어디로 밀려 하는지**를 재는 스칼라다. LMD가 더하는 것은 (i) 잔차 0의 완전 분해, (ii) **forward 귀속** — BEV가 실제로 출력에 무엇을 기여했는지, (iii) 부호 있는 유용/방해 분리, (iv) 스칼라가 아닌 공간 지도. 대체가 아니라 직교하는 축이다.
