# 실험 보고서 #09 — aux 유무로 BEV를 분해할 수 있는가 (LMD 차용)

**차용 논문:** Layer-Wise Modality Decomposition for Interpretable Multimodal Sensor Fusion, NeurIPS 2025 ([arXiv:2511.00859](https://arxiv.org/pdf/2511.00859)) — `craft_좋은bevfeature란.pdf`의 첫 번째 논문
**대상:** `origin/para` 기준 PARA-SSR (`para_ssr.py`, `para_ssr_head.py`)
**질문:** aux만 / planning만 / 둘 다 학습했을 때의 BEV를 LMD 방식으로 분해해, planning에 실제로 쓰이는 BEV와 방해되는 BEV를 정의할 수 있는가
**결론:** **가능하다.** 단, LMD를 그대로 쓰는 게 아니라 **분해 대상을 입력에서 weight로 옮겨야** 한다. 그 옮김이 성립한다는 것을 float64에서 잔차 0으로 확인했다 (`tools/lmd/verify_model_delta.py`). 최상위 리스크는 방법론이 아니라 **두 체크포인트가 같은 basin에 있는가**다 (§7.1).

---

## 1. LMD를 그대로는 못 쓴다 — 무엇을 바꾸는가

LMD는 **한 forward 안에서 덧셈으로 들어오는 입력들**(cam/lidar/radar)을 분해한다. 그런데 aux 유무는 입력 차이가 아니라 **학습된 weight의 차이**다. 같은 입력, 같은 아키텍처, 다른 θ. LMD의 분해 대상 자체가 어긋난다.

옮길 수 있는 건 LMD의 축이 아니라 **LMD를 성립시키는 성질**이다:

> 모든 비선형을 동작점에서 얼리면 망은 **정확히 affine**이 된다.

이게 있으면 weight 차이도 정확히 분해된다. 각 weight group `g`가 고정된 affine 연산자 `A_g`(aux 모델) 또는 `P_g`(planning-only 모델)가 되므로, 두 모델의 차이가 **telescoping**으로 쪼개진다:

$$A_L\cdots A_1 - P_L\cdots P_1 \;=\; \sum_g A_L\cdots A_{g+1}\,(A_g - P_g)\,P_{g-1}\cdots P_1$$

`G_g` := "group 1..g는 P, g+1..L은 A" 인 hybrid forward라 두면

$$\delta_g = G_{g-1} - G_g, \qquad \sum_g \delta_g \;=\; \text{out}_A - \text{out}_P \quad(\text{잔차 } 0)$$

`δ_g`는 **"group g의 weight만 planning-only에서 aux-trained로 바꿨을 때 BEV(혹은 궤적)가 얼마나 달라지는가"**다. LMD와 같은 기계장치, 축만 modality → supervision으로 바꾼 것. 이름을 붙이면 **Layer-wise *Model* Decomposition**.

---

## 2. 경로가 선형화 가능한가 — 확인 완료

| 연산 | 위치 | 처리 |
|---|---|---|
| `navi_se` SE gate | `para_ssr_head.py:36-52` | **얼릴 필요 없음.** gate가 nav embedding만으로 만들어져 BEV에 대해 상수 → 이미 정확히 선형 |
| TokenLearner `GroupNorm(1,C)` | `tokenlearner.py:31` | μ,σ 고정 → affine |
| TokenLearner `GELU` | `tokenlearner.py:19` | exact GELU는 `x·Φ(x)`라서 Φ 고정 = **정확, bias 0** |
| TokenLearner `softmax` | `tokenlearner.py:49` | `selected` 고정 → `einsum`이 BEV에 선형 |
| latent/way decoder | `SSR_e2e.py:80-105` | MHA softmax 고정, LN μ/σ 고정, FFN ReLU mask 고정 |
| `ego_fut_decoder` | `para_ssr_head.py:139-143` | ReLU mask 고정 |
| deformable `sampling_offsets`·`attention_weights` | `spatial_cross_attention.py:345-350`, `temporal_self_attention.py:211-216` | query에서 나오므로 **얼린다**. 고정 후 bilinear 샘플 + 가중합은 `value`에 선형 |
| per-camera scatter, `/count`, residual | `spatial_cross_attention.py:175-183` | 덧셈 누적 · 기하학 전용 상수 · 순수 덧셈 → 그대로 관통 |

**전 구간 선형화 가능. 근사 없음.**

**함정 하나 실제로 밟았다.** 1차 pass에서 캐싱한 상태값을 `detach()` 하지 않으면 μ/σ와 softmax를 통해 gradient가 새고 분해가 출력과 안 맞는다 — 같은 스크립트가 detach 없이는 **상대오차 4.0e-1로 실패**, 넣으면 1e-15로 통과. 논문의 "고정된 선형 스위치"는 구현상 이 한 줄이다 (`tools/lmd/lmd_core.py:31-40`).

---

## 3. 수치 검증

### 3.1 하류 선형화 (`verify_lmd_linearisation.py`)

```
  PASS  frozen MHA == nn.MultiheadAttention                  rel 1.32e-15
  PASS  frozen forward reproduces the real output            rel 0.00e+00
  PASS  frozen map is exactly affine in bev                  rel 3.67e-15
  PASS  sum_cells C_i + b == trajectory (residual-free)      rel 9.72e-16
```

### 3.2 aux 유무 분해 (`verify_model_delta.py`)

group은 실제 모듈 경계를 따라 8개: `embed · enc0 · enc1 · enc2 · tokenl · latent · way · mlp`.

```
  PASS  frozen forward reproduces model P                    rel 0.00e+00
  PASS  frozen forward reproduces model A                    rel 0.00e+00
  PASS  sum_g delta_g == bev_A - bev_P                       rel 1.83e-16
  PASS  sum_g delta_g == traj_A - traj_P                     rel 0.00e+00
  PASS  reverse order also sums exactly                      rel 0.00e+00
  PASS  sum_g shapley_g == traj_A - traj_P (256 coalitions)  rel 2.49e-16

  PASS  builder groups alone == BEV-change effect on the plan  rel 0.00e+00
  PASS  sum_cells J_A,i . d_bev_i == that same change          rel 1.04e-13
```

**경로 의존성은 실재한다.** telescoping은 어떤 순서로도 정확히 합해지지만 group별 값은 순서에 따라 달라진다 — 두 순서 간 최대 차이가 `|Δτ|`의 **25%**. 그래서 group 단위 숫자를 보고할 때는 **exact Shapley**를 쓴다. 스위치가 얼린 상태에서 출력은 group 연산자들에 대해 multilinear이므로, group 8개면 coalition 256개로 정확히 계산되고 efficiency 공리(합 = Δτ)가 진짜 검증 대상이 된다 — 위에서 2.5e-16.

---

## 4. 정의: planning-centric BEV, 그리고 방해되는 BEV

### 4.1 group을 "BEV를 만드는 쪽"과 "BEV를 읽는 쪽"으로 가른다

```
BUILD = embed, enc0, enc1, enc2      (shared BEV encoder)
READ  = tokenl, latent, way, mlp     (SSR planner)
```

**reader를 A에 고정한 채 builder만 P→A로 바꾼다.** 그러면 그 궤적 변화가 정확히 *"aux가 BEV를 바꿔서 생긴 planning 변화"*이고, reader가 얼린 affine 사상이므로 **BEV cell 단위로 정확히 쪼개진다**:

$$\Delta\tau_{\text{bev}} \;=\; \sum_{i=1}^{10000} J^{A}_i \,\Delta\mathrm{bev}_i, \qquad \Delta\mathrm{bev} = \mathrm{bev}_A - \mathrm{bev}_P$$

(검증됨: 위 마지막 두 줄. `builder groups alone == ...` 이 BUILD/READ 분리가 정확함을, `sum_cells J_A,i . d_bev_i == ...` 이 cell 분해가 정확함을 보인다.)

### 4.2 부호를 붙이면 답이 나온다

model P의 planning 오차 방향 `ê = (τ_P − τ_gt)/‖·‖`에 사영한다:

$$\sigma^{\Delta}_i \;=\; \big\langle J^{A}_i \Delta\mathrm{bev}_i,\ \hat e \big\rangle, \qquad \sum_i \sigma^{\Delta}_i = \langle \Delta\tau_{\text{bev}}, \hat e\rangle \ \ (\text{정확})$$

- **`σ^Δ_i < 0`** → aux가 그 cell을 바꿔서 궤적이 GT 쪽으로 왔다 = **planning-centric한 aux 효과**
- **`σ^Δ_i > 0`** → aux가 그 cell을 바꿔서 궤적이 GT에서 멀어졌다 = **방해되는 BEV**

100×100 **부호 있는 지도**가 나온다. 검증 스크립트의 출력 형태:

```
      helped     1449 cells   -0.00426
      HURT       2151 cells   +0.00791   <- "방해되는 BEV"
      net +0.00365
```

부호 있는 분해가 가능한 건 오로지 **잔차가 0이기 때문**이다. attention rollout이나 gradient saliency로는 이 정의를 못 만든다 — 합이 오차에 맞지 않으니 "방해"의 기준선 자체가 없다.

### 4.3 같이 볼 스칼라

| 기호 | 정의 | 읽는 법 |
|---|---|---|
| `π_i` | `‖C_i‖₁ / Σ‖C_j‖₁`, `C_i = J_i·bev_i` | planning 기여 지도. **planning-centric BEV := π의 유효 지지집합** (participation ratio `(Σπ)²/Σπ²`) |
| `ρ` | `‖b‖ / (‖b‖+‖Σ_i C_i‖)`, `b` = BEV를 0으로 둔 frozen forward | **장면과 무관하게 나오는 궤적 성분의 비율.** ablation 없이 ego-shortcut을 직접 측정 |
| `H` | `Σ_{σ^Δ>0} σ^Δ_i / |Σ_i σ^Δ_i|` | aux 효과 중 방해 비율 |
| `φ_g` | group g의 Shapley 값 | aux 효과가 **어느 모듈에서 만들어졌는지** |
| PPA | `(Σ_{i∈GT} π_i)/(|GT|/10000)` | GT 박스·drivable area에 planning 질량이 균등 대비 몇 배 몰렸나 |

### 4.4 체크포인트 하나로도 지금 되는 것 (basin 문제 없음)

두 모델 비교와 **직교하는** 축이 하나 더 있고, 이건 `para_ssr_60ep` 하나만으로 지금 가능하다. full 모델 안에서 det/map/occ head의 출력도 같은 adjoint로 BEV에 분해해 aux 관련도 `ω_i`를 얻으면:

| cell 유형 | 조건 | 해석 |
|---|---|---|
| **공유-유용** | `π` 높음 · `ω` 높음 · `σ<0` | aux가 써넣은 내용이 planning에 도움 |
| **aux-방해** | `π` 높음 · `ω` 높음 · `σ>0` | aux가 써넣은 내용을 planner가 잘못 씀 |
| **aux-무관** | `π` 낮음 · `ω` 높음 | BEV 용량을 planning이 안 쓰는 데 소모 |
| **plan 전용** | `π` 높음 · `ω` 낮음 | ego shortcut이 여기 몰린다 |

`overlap = cos(π, ω)`. §4.2가 *"aux를 붙였더니 BEV가 어떻게 달라졌나"*라면, 이건 *"한 모델 안에서 aux와 planner가 BEV의 같은 곳을 보나"*다. 둘 다 있어야 그림이 닫힌다.

---

## 5. 세 조건은 이미 config로 존재한다

| 조건 | config | 설정 |
|---|---|---|
| **planning만** (= model P) | `PARA_SSR_e2e_60ep_planonly.py` | `grad_balance target = plan 1.0 / det 0 / map 0` — aux head는 존재하되 BEV로 gradient가 안 간다 |
| **aux만** | `PARA_SSR_stage1_detmap.py` | `plan 0.0 / det 1.0 / motion 1.0 / map 1.0` |
| **둘 다** (= model A) | `PARA_SSR_e2e_60ep.py` | GradBalancer plan 0.4 / det 0.2 / map 0.2 / occ 0.2 |

`planonly`가 좋은 대조군인 이유는 report #08과 그 config docstring에 이미 있다 — 코드 경로·파라미터 수·clip norm이 동일하고 **aux→BEV gradient 하나만** 다르다. §4.2의 (A, P) 쌍은 이 둘이다.

---

## 6. 실행 계획

0. **basin 사전점검 (§7.1). 이걸 통과 못 하면 아래는 전부 무의미하다.**
1. `tools/lmd/lmd_core.py` — 완료. 선형화 primitive.
2. `tools/lmd/replica.py` — 완료. 실제 op 스택을 weight group 체인으로 노출.
3. `tools/lmd/verify_lmd_linearisation.py`, `verify_model_delta.py` — 완료. **`ssr` env에서 실제 모듈로 한 번 더 돌린 뒤** 체크포인트 숫자를 믿을 것.
4. 실제 모듈 후킹 (미구현):
   - `MSDeformableAttention3D.forward` / `TemporalSelfAttention.forward`에 `sampling_locations`·`attention_weights` 캐시-재사용 훅. `bev_mask`·`indexes`·`count`는 `point_sampling` 유래의 기하학 전용이라 안전하다.
   - head 경로의 LN/GN/ReLU/GELU/softmax는 §2의 detach 방식으로 교체.
   - 두 체크포인트를 동시에 올린 hybrid forward 러너 (`replica.hybrid`와 같은 구조).
5. 드라이버: val 샘플 순회 → per-sample `σ^Δ, π, ω, ρ, φ_g` npz 덤프 → 집계/플롯.

**비용.** telescoping은 group 8개 + 1 = 9회 hybrid forward/샘플, Shapley는 256회. Shapley는 대표 샘플 수십 개에만 쓰고 전체 val은 telescoping 양방향으로 돌린 뒤 두 순서의 스프레드를 불확실성으로 함께 보고하는 게 현실적이다. cell 단위 `σ^Δ`는 backward 12회/샘플로 싸다. 모델 두 개를 동시에 올려야 하므로 GPU 메모리는 2배.

---

## 7. 반드시 같이 보고해야 할 한계

### 7.1 basin 문제 — 이게 최상위 리스크

telescoping은 **어떤 두 weight에 대해서도 산술적으로 정확**하다. 하지만 *"group g의 weight 변화"*가 **해석 가능한 양인지**는 별개다. 두 모델이 서로 다른 basin으로 갔다면 (permutation symmetry 등) `A_g − P_g`는 정확하지만 무의미한 숫자다.

`planonly`와 `60ep`는 같은 init에서 출발하지만 **60 epoch 동안 독립적으로 학습됐다.** 반드시 먼저 확인할 것:

- **linear mode connectivity**: `θ(α) = (1−α)θ_P + α θ_A` 를 α∈[0,1]로 훑으며 planning L2를 잰다. barrier가 낮으면 delta가 해석 가능하다. 높으면 §4.2를 그대로 쓰면 안 된다.
- **막혔을 때의 정공법 — paired branch 학습**: 한 run을 epoch N까지 돌린 뒤 거기서 aux on / aux off 두 갈래로 fork해 나머지를 학습한다. 그러면 weight delta가 정의상 작고 의미 있다. 60ep 예산에서 epoch 40 fork + 20 epoch면 비교적 싸다. **§4.2를 제대로 하려면 이 실험 설계가 맞다.**
- **막힌 채로도 되는 것**: §4.4는 모델 하나만 쓰므로 basin과 무관하다.

### 7.2 얼린 routing

LMD는 **value 경로로만** 귀속한다. deformable attention의 sampling offset이 "어디를 볼지" 정하는데 그건 source가 아니라 스위치로 청구된다. 결과의 정확한 문장은 *"거기를 보기로 한 상태에서 내용을 누가 공급/변경했나"*이지 *"누가 거기를 보게 만들었나"*가 아니다. LMD 원본의 fusion 세팅보다 여기서 더 큰 제약이다. 부분적 보정: model A와 P의 frozen `sampling_locations`를 직접 비교해 "aux가 시선 자체를 얼마나 옮겼는가"를 별도 지표로 보고할 것.

### 7.3 경로 의존성

§3.2에서 측정된 대로 group별 값은 순서에 따라 `|Δτ|`의 25%까지 달라진다. group 단위 숫자는 **Shapley로 보고**하고, telescoping 단일 순서 값을 그대로 인용하지 말 것.

### 7.4 `stage1_detmap`은 planner가 학습되지 않았다

해당 config docstring이 직접 밝히듯 planner 전용 2.42M은 gradient 0에 weight decay만 받아 초기 norm의 ~85%로 끝난다. 이 모델에서는 §4.2/4.3을 자기 planner로 읽을 수 없다. "이 BEV가 planning에 쓸 정보를 담고 있나"를 물으려면 BEV를 얼리고 **동일한 planner probe**를 각 BEV 위에 따로 학습시켜야 한다. 그리고 report #08의 "budget match, not a controlled experiment"대로 `stage1_detmap` vs 나머지는 교란변수 5개짜리 recipe 비교다. 깨끗한 쌍은 `planonly` vs `60ep`뿐이다.

### 7.5 기타

- **EMA/raw 통일.** 최종 eval이 `epoch_N_ema.pth`로 나가므로 분석도 EMA로 통일.
- **기존 진단과의 관계.** `para_ssr.py`의 `_bev_grad_norms`(task별 BEV gradient norm + cosine)와 `_representation_metrics`(`tok_sim`, `bev_std`, `tok_cover`)는 **loss가 BEV를 어디로 밀려 하는지**를 재는 스칼라다. 여기서 더하는 건 (i) 잔차 0의 완전 분해, (ii) **forward 귀속** — BEV가 실제로 출력에 무엇을 기여했는지, (iii) 부호 있는 유용/방해 분리, (iv) 스칼라가 아닌 공간 지도. 대체가 아니라 직교하는 축이다.

---

## 부록 — 쓰지 않기로 한 축

LMD의 원래 축(입력 source)도 이 모델에서 **정확히 성립한다**. `SSR_transformer.py:270`의 `bev_queries = bev_embedding.weight + can_bus_mlp(can_bus)` 덕분에 BEV를 `6개 카메라 + prev_bev + BEV query prior + can_bus + bev_pos + bias`로 잔차 없이 쪼갤 수 있고, 초기 검증에서 8.3e-16으로 통과했다 (`git show 8ca9b20`). "Is Ego Status All You Need?"를 ablation 없이 닫힌 형식 지분으로 답하는 축이다. 지금 질문의 축이 아니라 접어두지만, aux 유무 분석에서 ego-shortcut 이야기가 나오면 그때 꺼내 쓰면 된다.
