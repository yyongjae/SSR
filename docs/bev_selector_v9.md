# BEV Selector Distillation v9
## Planning-Grounded Probe (PGP): 도메인 특화 Planning Loss로 학습하는 증류용 Cross-Attention Probe

**기준 문서**: [`bev_selector_v8.md`](bev_selector_v8.md), [`bev_selector_v7.md`](bev_selector_v7.md)  
**상태**: 설계 제안 및 심층 분석 반영본 (미구현, `v8_fix` 완주 후 진행 예정). 아래 효과는 §6.2 사전 검증 및 §8의 ablation으로 확인한다.  
**목표**: Aux Teacher(BEVFusion, ReSMap)의 BEV 지식을 Student BEV로 증류하되, 증류 attention이 고정된 거리 커널에 머물지 않고 **각 Teacher의 전문 도메인에서 planning에 실제로 중요한 영역(위험 객체, 차선/연석 경계)**을 보도록 만든다.

---

## 1. 동기: `v8` / `v8_fix`의 기대 성능과 구조적 천장(Ceiling)

### 1.1 `v8` 실측 결과 및 현재 진행 중인 `v8_fix` 전망
* **직전 `v8` (Epoch 20 조기 중단) 실측**:
  - ① `plan_heading_from_xy=False`(24채널 직접 회귀), ② ReSMap 좌우 반전(`flip_w=True`, Road IoU `0.360`), ③ 20 에포크 중단이라는 3중 페널티 상태에서도 동일 조건 Student(Ep29, **0.8711**) 대비 **0.8782 (+0.71%p)**를 기록하며 증류 구조의 유효성을 입증했다.
* **현재 `tmux 19`에서 학습 중인 `v8_fix` 기대치**:
  - 원본 `km/para-ssr-v2` 사양인 `plan_heading_from_xy=True`(16채널 + 3차 베지에 접선 헤딩 유도, 단독 베이스라인 **0.8814**)와 ReSMap 정방향(`flip_w=False`, Road IoU **0.965**)을 모두 복원했다.
  - 30 에포크 완주 시 DAC(도로 이탈 방지)와 EP 이득이 온전히 결합되어 **PDMS `0.886 ~ 0.892` 구간(Student 대비 `+0.5 ~ +1.0%p`)** 달성이 유력하다.

### 1.2 그럼에도 `v8`이 가진 구조적 한계 (왜 `v9`가 필요한가?)
`v8`은 collapse를 막기 위해 증류용 cross-attention의 `anchor_mlp`, `cmd_mlp`, `status_mlp`, `query_norm`, `q/k/v/out_proj`를 모두 `requires_grad_(False)`로 **랜덤 초기화 상태에서 동결**했다 ([`para_ssr_agent.py`](../navsim/agents/para_ssr/para_ssr_agent.py) `get_optimizers`). 코드를 엄밀히 분석하면 다음 한계가 존재한다.

| 문제 | 코드 및 수식 기반 상세 분석 |
|---|---|
| **Attention이 의미를 학습하지 않음** | 무작위 투영에서 내적 $\frac{Q K^T}{\sqrt{d}}$은 평균 $0$, 분산 $\approx 1$인 가우시안 노이즈에 가깝다. 반면 공간 편향 $M_{h,k} = -\frac{d^2}{2\sigma_h^2}$는 거리에 따라 $0 \sim -50$까지 변하므로, 실질적인 어텐션 분포의 90% 이상을 **고정된 거리 가우시안 커널(`spatial_bias`)**이 지배한다. |
| **KD 대상이 불분명** | `A_t`는 Teacher 자신의 attention이 아니라 "랜덤 probe가 Teacher BEV를 읽은 분포"다. |
| **사실상 Dense Feature KD화** | $N_{\text{eff}} \approx 141$개로 평탄화된 앵커 가중치와 $\sigma_4 = 16\text{m}$ 반경이 결합되어, 특정 위험/중요 영역을 선택(Select)하기보다 **전방 BEV 격자 전체($50 \times 100$)를 부드럽게 풀링해 정렬**하는 것과 유사하게 작동한다. |
| **단순히 열면 Collapse 발생** | 일치도(KL, L2)만 최소화하는 loss에서 probe를 열면 `q_proj→0`, `v_proj→0` 같은 상수 해가 최적해가 되어 어텐션이 균일 분포로 붕괴한다. |
| **문서-코드 간 KL 방향 불일치** | `v8` 문서는 $\text{KL}(A_s \parallel A_t)$로 기술했으나, 실제 코드([`anchor_distill.py:397`](../navsim/agents/para_ssr/distill/anchor_distill.py#L397))의 `F.kl_div(attn_s.log(), attn_t)`는 **Forward KL인 $\text{KL}(A_t \parallel A_s) = \sum A_t \log \frac{A_t}{A_s}$**를 계산한다. (단, Teacher가 양수인 영역을 Student가 놓치지 않게 강제하는 Mean-seeking 성질 덕분에 KD 목적에는 오히려 $\text{KL}(A_t \parallel A_s)$가 적합하므로 `v9`에서도 이 방향을 유지하고 수식을 명확히 통일한다.) |

**근본 원인**: 증류 loss에 "probe가 정보를 보존/선택하라"는 task 신호가 없다. 일반 cross-attention이 의미를 배우는 이유는 출력이 task loss로 이어지기 때문이다. 따라서 `v9`는 probe를 **각 Teacher의 도메인에 맞는 planning loss에 직접 연결**한다.

### 1.3 `v8` 학습 로그에서 관찰한 점 (W&B run `vaga21vb`, `r34_sel_v8_2`)

| 지표 | 실측 값 | 의미 및 진단 |
|---|---|---|
| `distill/effective_anchors` | **137~150** (고정) | 목표(20~60)보다 훨씬 평평하다. 바닥에 깔린 균일 분포(`λ_cov=0.2`)와 넓은 `τ_risk=8.0` 정규화 때문에 앵커 선택성이 희석됨 (§3.4에서 해결). |
| `distill_cos/*` | 0.97~0.98 (빠르게 포화) | 선형 투영 정렬은 초반에 빠르게 포화되고 이후 변별력 있는 신호가 약해진다. |
| `loss_attn_*` (KL) | 0.2~0.5 → 0.13~0.2 | `v7`(0.013)보다 큰 값에서 시작해 학습 신호는 존재하나, 랜덤 투영 내적을 맞추는 수준에 머문다. |
| `gnorm/distill` vs `gnorm/plan` | `0.012` vs `0.091` (~13%) | 증류 그래디언트가 주 플래너 그래디언트의 약 13% 수준으로, `GradBalancer` 외부에서 더해질 때 이 정도 스케일이 안전 상한선임을 보여준다. |

---

## 2. `v9` 핵심 아이디어: 비대칭 Stop-Grad와 도메인 특화 Probe

```
                   ┌────────────────────────────────────────────────────────┐
                   │  Domain-Specific Planning-Grounded Probe P_b           │
                   │  b ∈ {det (BEVFusion), map (ReSMap)}                   │
                   │  256 anchor queries → L=1 Cross-Attn(BEV)              │
                   │  (Head 1,2: Local Bias σ=[2,4]m / Head 3,4: Global)    │
                   │  heads: im_reward /offset / domain-specific sim_reward │
                   └───────────────────────────┬────────────────────────────┘
                                               │ 100% 동일한 Query Q 및 파라미터 P_b 공유
         ┌─────────────────────────────────────┴─────────────────────────────────────┐
         ▼ (Teacher BEV: 캐시, BEV grad 없음)                                        ▼ (Student BEV → student_proj_b)
  [P_b, 파라미터 학습 O]                                                      [P_b, 파라미터 stop-grad (functional_call)]
    A_t, h_t 생성                                                               A_s, h_s 생성
    L_probe,b = 도메인 특화 planning loss                                       gradient는 Student BEV / student_proj 로만 흐름
      • P_det: NC, TTC, EP + im/offset (동적 충돌 위험 집중)                      │
      • P_map: DAC, EP + im/offset (차선/도로 경계 집중)                          │
         │                                                                      │
         └────────────── Attention KD: KL(A_t ‖ A_s), Feature KD ───────────────┤
                                                                                └─ (선택적 Ablation) L_task^S:
                                                                                   기본값 λ_t = 0 (GradBalancer 교란 방지)
```

1. **Teacher 쪽 probe `P_b`를 도메인 특화 planning loss(`L_probe,b`)로 학습한다.**
   - Teacher BEV는 캐시된 텐서이므로 BEV로는 gradient가 흐르지 않고, probe 파라미터만 `L_probe,b`로 업데이트된다.
   - `P_det`는 충돌(`NC`, `TTC`), `P_map`은 도로 준수(`DAC`) 예측에 도움이 되는 BEV 셀을 찾아내도록 attention이 학습된다.
2. **Student는 같은 probe를 stop-grad 파라미터(`functional_call` detach)로 통과한다.**
   - 증류 loss의 gradient는 probe 파라미터에 닿지 않고 오직 Student BEV와 `student_proj`로만 흐른다.
3. **`L_task^S`(Student BEV 직접 planning loss)는 기본값에서 비활성화(`λ_t = 0`)한다.**
   - 초기 구상에서는 Student BEV에도 `L_task^S`를 직접 거는 방안을 고려했으나, 심층 분석 결과 **GradBalancer 교란 위험**이 확인되어 기본 설정에서는 순수 `Attention KD + Feature KD`만 사용한다 (상세 이유는 §2.2 참조).

### 2.1 Collapse가 원천 차단되는 이유
- Probe 파라미터(`q/k/v/out_proj`, MLP)를 움직이는 유일한 손실은 Teacher BEV 위에서 계산되는 `L_probe,b`(planning loss)뿐이다. `q_proj→0`이나 `v_proj→0`으로 가면 Teacher의 reward/offset 예측이 망가지므로 `L_probe,b`가 급증한다.
- 증류 손실(`KL`, `L2/Cosine`)은 probe 파라미터로 역전파되지 않으므로, Student와 Teacher의 차이를 쉽게 줄이기 위해 probe가 스스로 망가지는 지름길(Shortcut)이 존재하지 않는다.

### 2.2 왜 `L_task^S`를 기본값 `0`으로 두는가? (`GradBalancer`와의 상호작용 분석)
- 현재 Student 모델은 본체 플래너(`AnchorPlanner`)와 `Det/Motion`, `Map` 보조 헤드를 가지고 있으며, [`para_ssr_loss.py`](../navsim/agents/para_ssr/para_ssr_loss.py)의 `GradBalancer`가 Student BEV에 유입되는 그래디언트 비율을 **`plan(0.4) : det(0.3) : map(0.3)`**으로 정밀하게 맞추고 있다.
- 그런데 증류 손실은 `GradBalancer` 계산이 끝난 뒤 바깥([`para_ssr_agent.py:536`](../navsim/agents/para_ssr/para_ssr_agent.py#L536))에서 `loss = loss + weight * distill_loss`로 더해진다.
- 만약 여기에 본체 planning loss와 맞먹는 크기의 `L_task^S` (Det probe + Map probe 2개 분량의 planning loss)를 그대로 더하면, Student BEV가 총 3개의 서로 다른 플래너 헤드(본체 + `P_det` + `P_map`)로부터 거대한 planning gradient를 받게 되어 **`plan:det:map` 그래디언트 밸런스가 붕괴**하고 보조 과업 학습이 저해된다.
- 따라서 주력 모델(Main)에서는 **`λ_t = 0`으로 두어 Probe를 순수하게 "어디를 보고 무엇을 뽑을지 알려주는 학습된 렌즈(Lens)"로만 사용**하고, `L_task^S`는 작은 가중치로 §8의 Ablation(D)에서만 별도 검증한다.

---

## 3. 핵심 설계 및 수식 보강 (`v8` 병목 해결)

### 3.1 도메인 특화 Single-Layer Probe `P_b` (`L=1` & 채널 분리)

1. **왜 `L=2`가 아니라 `L=1` (Single-Layer Cross-Attention)인가?**
   - **동일 Query $Q$ 보장**: `L=1`이면 Query $Q$는 오직 `(256 Anchor, Command, Ego Status)`로만 결정되므로 **Teacher와 Student가 100% 동일한 Query $Q$를 공유**한다. 반면 `L=2`층에서는 1층 출력 피처가 2층 Query로 들어가므로 Teacher Query $Q_t^{(2)}$와 Student Query $Q_s^{(2)}$ 자체가 달라져, 2층 Attention 차이($A_t^{(2)}$ vs $A_s^{(2)}$)가 BEV Key 때문인지 Query 때문인지 교란된다.
   - **메모리 및 속도 최적화**: `[B=4, H=4, K=256, HW=5000]` 어텐션 행렬은 텐서 1개당 약 `82 MB`다. `L=1`로 구성하면 연산량과 활성화 메모리를 `v8`과 거의 동일한 수준(+수백 MB 이내)으로 유지할 수 있다.
   - **구조**: 256개 앵커 임베딩(`anchor_mlp + cmd_mlp + status_mlp → LayerNorm`) → **1층 Multi-Head Cross-Attention + FFN(SwiGLU 또는 ReLU MLP)** → 경량 Prediction Heads (`im_reward`, `sim_reward`, `offset_head`).

2. **도메인 특화 Loss 분리 (필수 설계로 승격)**
   - **문제점**: ReSMap(지도 Teacher) BEV에는 차량·보행자 같은 **동적 객체 정보가 전혀 없다.** ReSMap BEV를 보고 충돌(`NC`)이나 `TTC`를 맞추라고 학습시키면, `P_map`은 도로 형태만으로 앞차 유무를 추측하는 가짜 상관관계(Spurious correlation)에 과적합된다. 반대로 BEVFusion에는 정밀 차선/연석 폴리곤(`DAC`) 정보가 부족하다.
   - **해결책 (`SIM_KEYS = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort"]`)**:
     - **`P_det` (BEVFusion Probe)**: 동적 장애물 인지 및 속도 제어에 직결되는 **`NC`(ch 0), `EP`(ch 2), `TTC`(ch 3), `Comfort`(ch 4)** BCE + `im_reward` + `offset`으로만 학습한다 (**`DAC` 가중치 = 0**).
     - **`P_map` (ReSMap Probe)**: 도로 기하학 및 차선 유지에 직결되는 **`DAC`(ch 1), `EP`(ch 2), `Comfort`(ch 4)** BCE + `im_reward` + `offset`으로만 학습한다 (**`NC`, `TTC` 가중치 = 0**).

3. **Hybrid Local-Global Head Bias (§7.1 "편향만 쓰는 현상" 해결)**
   - **문제점 (Lazy Anchor Bias)**: 4개 헤드 모두에 강한 가우시안 거리 편향 $M_{h,k}$를 걸어두면, Probe가 `Q·Kᵀ` 콘텐츠 내적을 학습하지 않고도 고정된 $M_{h,k}$가 긁어오는 궤적 위 평균 피처만으로 `L_probe`를 어느 정도 풀 수 있어 `q_proj, k_proj` 학습이 둔화된다.
   - **해결책 (역할 분담 헤드)**:
     - **Head 1, 2 (Local Corridor Heads, $\sigma = [2.0\text{m}, 4.0\text{m}]$)**: 궤적 중심선 및 차선 폭 내부의 정밀 기하학 피처를 추출하되, 학습 가능한 스케일 게이트 $\gamma_h$ ($\text{Logits}_h = \frac{Q_h K_h^T}{\sqrt{d}} + \gamma_h M_{h,k}$)를 둔다.
     - **Head 3, 4 (Global Context / Interaction Heads, $\sigma = [12.0\text{m}, \infty(\text{Unbiased})]$)**: 거리 편향을 아주 약하게 주거나 아예 제거($M_{4,k} = 0$)하여, 멀리서 교차로로 진입하는 차량이나 끼어드는 차량을 **순수하게 `Q·Kᵀ` 콘텐츠 유사도만으로 탐지**하도록 강제한다.

### 3.2 Student 쪽 증류 신호
`student_proj_b(student_bev)`를 `P_b`에 통과시킨다. 이때 `torch.func.functional_call(P_b, {k: v.detach() ...}, ...)`을 사용하여 Probe 파라미터로의 그래디언트를 완전히 차단한다.

| 신호 | 수식 정의 | gradient 대상 |
|---|---|---|
| **Attention KD** | $\mathcal{L}_{\text{attnKD}, b} = \sum_{k=1}^K w_k \cdot \frac{1}{H} \sum_{h=1}^H \text{KL}\left(A_{t, h, k}^{\text{detach}} \parallel A_{s, h, k}\right)$ | Student BEV, `student_proj_b` |
| **Feature KD** | $\mathcal{L}_{\text{featKD}, b} = \sum_{k=1}^K w_k \left[ 0.5 \|\tilde{h}_{s,k} - \tilde{h}_{t,k}^{\text{detach}}\|_2^2 + 0.5 (1 - \cos(\tilde{h}_{s,k}, \tilde{h}_{t,k}^{\text{detach}})) \right]$ | Student BEV, `student_proj_b` |
| **Task Grounding (Ablation 전용)** | $\mathcal{L}_{\text{task}}^{S, b} = \mathcal{L}_{\text{probe}, b}(P_b^{\text{detach}}(\text{student\_bev}))$ (기본값 $\lambda_t = 0.0$) | Student BEV, `student_proj_b` |

### 3.3 학습 스케줄
1. **Warm-up** (첫 1 epoch, 약 665 step): $\text{ramp}(t) = 0$. `L_probe,det + L_probe,map`만으로 Teacher BEV 위에서 두 Probe를 먼저 수렴시킨다.
2. **Ramp** (다음 1~2 epoch): 증류 가중치 $\text{ramp}(t)$를 $0 \to 1$로 선형 증가시킨다.
3. **본 학습 (EMA Probe 옵션)**: Probe가 계속 학습되면서 Teacher 타깃 분포 $A_t, h_t$가 흔들리는 것을 방지하기 위해, Teacher/Student 증류 경로에는 Probe의 **EMA 사본**(decay `0.999`)을 사용하는 옵션을 둔다.

### 3.4 앵커 가중치 $w_k$ 재보정 (`v8`의 $N_{\text{eff}} \approx 141$ 평탄화 원인 규명 및 해결)
* **왜 `v8`에서 $N_{\text{eff}} \approx 141$이 나왔는가? (수학적 원인 2가지)**:
  1. **바닥 균일 분포(`λ_cov = 0.2`)의 엔트로피 하한 효과**: 256개 앵커에 0.2의 균일 확률을 깔면, 나머지 0.8이 단 1개 앵커에 100% 집중(Dirac delta)되더라도 섀넌 엔트로피가 최소 $H \approx 1.65 \implies N_{\text{eff}} \ge 5.2$에서 시작하며, 조금만 퍼져도 $N_{\text{eff}}$가 급증한다.
  2. **위험 탐색 항(`r_k`, [`anchor_distill.py:219-221`](../navsim/agents/para_ssr/distill/anchor_distill.py#L219-L221))의 과도한 평탄화**: `risk_proximity = softmax(-dist / 8.0)`으로 계산한 뒤 안전하지 않은 모든 앵커($1 - \text{NC}\cdot\text{DAC} > 0$)에 곱해 다시 합이 1이 되도록 정규화(`r_k / r_k.sum()`)한다. $\tau_{\text{risk}} = 8.0$은 8-waypoint L2 거리 기준 너무 넓어서 사실상 80~120개의 실패 앵커에 고르게 퍼진 두 번째 균일 분포 역할을 했다. 즉 `w_cov(0.2) + r_k(0.2) = 40%`가 광역 분포였기 때문에 $N_{\text{eff}}$가 140 아래로 내려갈 수 없었다.
* **`v9` 보정 해법 (목표 $N_{\text{eff}} \in [25, 50]$)**:
  - **`λ_cov`: `0.2 → 0.02`** (어차피 Multi-scale head의 $\sigma=12\text{m}, \infty$가 공간 전체의 커버리지를 보장하므로, 앵커 가중치에서까지 균일 분포를 크게 줄 필요가 없다.)
  - **`λ_gt`: `0.6 → 0.68`, `λ_risk`: `0.2 → 0.30`**
  - **`τ_gt`: `5.0 → 2.0`, `τ_risk`: `8.0 → 2.5`**: 정답 궤적에 정말 가까운 후보군과, 정답 궤적 바로 옆에서 아슬아슬하게 충돌/이탈하는 **Near-miss Hard Negative 앵커 15~30개**에만 가중치가 날카롭게 모이도록 조인다.

---

## 4. 구현 위치 (예정)

| 파일 | 변경 내용 |
|---|---|
| `navsim/agents/para_ssr/distill/planning_probe.py` (신규) | `PlanningGroundedProbe` (`L=1` Cross-Attn + FFN + 도메인별 마스크된 Reward/Offset Head), `functional_call` 기반 stop-grad 호출, EMA 사본 관리 |
| [`distill/anchor_distill.py`](../navsim/agents/para_ssr/distill/anchor_distill.py) | `version="v9"` 분기: `P_det`, `P_map` 호출, `L_probe_det`/`L_probe_map` 및 증류 손실 반환, Warm-up/Ramp 스케줄러, 보정된 $w_k$ 수식 |
| [`para_ssr_agent.py`](../navsim/agents/para_ssr/para_ssr_agent.py) | `get_optimizers`에서 `version == "v9"`일 때는 `P_det`, `P_map` 및 `student_proj`를 옵티마이저에 등록 (`v8`의 probe 동결 블록과 분기). 시작 시 학습 가능 파라미터 수 assert 검증 |
| [`distill/distillation.py`](../navsim/agents/para_ssr/distill/distillation.py) | `gt_trajectory`, `sim_reward`, `sim_reward_valid`를 probe loss까지 전달 |
| `configs/default.py`, `configs/agent/para_ssr_selector_v9_agent.yaml` (신규) | `v9` 하이퍼파라미터 (`plan_heading_from_xy: true`, `resmap flip_w: false` 기본 탑재) |
| `scripts/training/run_bev_selector_v9_distill.sh` (신규) | 분산 학습 런처 |

---

## 5. 손실 함수 요약

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{student\_balanced}}(\text{plan}, \text{det}, \text{map})}_{\text{GradBalancer (0.4 : 0.3 : 0.3)}} + \underbrace{\mathcal{L}_{\text{probe}}^{\text{det}}(T_{\text{bev}}) + \mathcal{L}_{\text{probe}}^{\text{map}}(T_{\text{map}})}_{\text{Teacher BEV 위에서 Probe만 학습 (Student BEV와 무관)}} + \text{ramp}(t) \cdot \frac{1}{2} \sum_{b \in \{\text{det}, \text{map}\}} s_b \left( \lambda_f \mathcal{L}_{\text{featKD}, b} + \lambda_a \mathcal{L}_{\text{attnKD}, b} + \lambda_t \mathcal{L}_{\text{task}}^{S, b} \right)$$

- 기본값: $\lambda_t = 0.0$ (`GradBalancer` 보호), $\lambda_f = 10.0$, $\lambda_a = 5.0$, $s_{\text{det}} = 1.0$, $s_{\text{map}} = 3.0$.
- $\mathcal{L}_{\text{probe}}^{\text{det}}$와 $\mathcal{L}_{\text{probe}}^{\text{map}}$은 Student 그래프와 완전히 분리되어 있으므로 Student BEV의 `gnorm`에 전혀 영향을 주지 않는다.
- `v8`에서 고친 중복 합산 버그를 반복하지 않도록, 텔레메트리용 항(`loss_distill_tok`, `loss_attn_*`, `probe_auc/*`)은 `metrics`로만 기록하고 `losses` 딕셔너리에는 넣지 않는다.

---

## 6. 진단 지표 (W&B) 및 본 학습 전 10분 사전 검증 프로토콜

### 6.1 실시간 학습 진단 지표 (W&B)
| 지표 | 확인할 것 |
|---|---|
| `probe/det/{nc_auc, ttc_auc, offset_l1}` (Teacher BEV) | `P_det`가 BEVFusion BEV에서 충돌/TTC 위험을 실제로 분류해내는지 (AUC > 0.85 목표). |
| `probe/map/{dac_auc, ep_corr, offset_l1}` (Teacher BEV) | `P_map`이 ReSMap BEV에서 도로 이탈(DAC) 여부를 정확히 분류해내는지 (AUC > 0.90 목표). |
| `probe/{det,map}/student_gap` | 동일 Probe에 Student BEV를 넣었을 때 Teacher BEV 대비 reward 예측 오차가 줄어드는지. |
| `probe/head_entropy/{h0,h1,h2,h3}` | Unbiased Global Head(`h3`)가 균일 분포에 빠지지 않고 의미 있는 피크(낮은 엔트로피)를 형성하는지. |
| `distill/effective_anchors` | §3.4 보정 후 목표 구간(`25 ~ 50`) 내에 안착하는지. |
| `gnorm/distill` vs `gnorm/plan` | 증류 그래디언트 노름이 `gnorm/plan`의 `10% ~ 25%` 범위를 유지하여 밸런서를 해치지 않는지. |

### 6.2 [핵심] 30시간 본 학습 전 "10분 Standalone Probe 검증" (Step 0)
Student 전체 모델을 끼고 30 에포크를 돌리기 전에, **캐시된 Teacher BEV(`BEVFusion`, `ReSMap`)와 `pdm_score_256` 라벨만 로드하는 초경량 스크립트**로 다음 두 가지를 10분 만에 먼저 검증한다:
1. **`N_eff` 격자 탐색 (CPU, 1분)**: 실제 train 토큰 500개에서 $(\tau_{\text{gt}}, \tau_{\text{risk}}, \lambda_{\text{cov}})$ 조합별 $N_{\text{eff}}$ 분포를 계산하여 정확히 평균 $35 \sim 45$가 나오는 하이퍼파라미터를 고정한다.
2. **Teacher Probe 단독 학습 및 Attention 시각화 (GPU 1장, 5~10분)**:
   - `P_det`와 `P_map`만 단독으로 3 에포크 학습시켜 `NC/TTC` 및 `DAC` 분류 AUC를 측정한다.
   - 학습된 `A_t` 히트맵을 샘플 시나리오 10개에 대해 플롯하여, **`P_det`의 어텐션이 실제로 전방/교차로 차량 위에 맺히는지, `P_map`의 어텐션이 차선/연석 경계선 위에 맺히는지** 눈으로 확인한다. (이 그림은 그대로 논문의 핵심 Qualitative Figure로 사용할 수 있다.)

---

## 7. 위험 요인과 설계 반영된 해결책 요약

| 위험 요인 | `v9` 설계에 반영된 해결책 |
|---|---|
| **1. Probe가 공간 편향($M_{h,k}$)만 쓰고 콘텐츠($Q\cdot K^T$)를 안 배울 위험** | 4개 헤드 중 Head 1, 2만 Local Bias($\sigma=2, 4\text{m}$)를 주고, **Head 3, 4는 약한 편향/무편향($\sigma=12\text{m}, \infty$)으로 분리**하여 콘텐츠 유사도 학습 강제 (§3.1). |
| **2. Teacher 도메인 불일치로 인한 가짜 상관관계 과적합** | ReSMap에서 충돌(NC/TTC)을 배우거나 BEVFusion에서 DAC를 배우지 않도록 **도메인별 Loss 마스킹** 적용 (§3.1). |
| **3. `L_task^S`가 Student의 `GradBalancer`를 무너뜨릴 위험** | 기본 설정에서 **`λ_t = 0`으로 비활성화**하고 순수 KD 신호만 전달하여 `plan:det:map` 비율 보존 (§2.2). |
| **4. 다층 Probe(`L=2`)로 인한 Query 불일치 및 VRAM 폭증** | **`L=1` Single-Layer Cross-Attention**으로 Teacher-Student 간 100% 동일한 앵커 Query $Q$ 보장 및 메모리 절약 (§3.1). |
| **5. 동시 학습 시 움직이는 타깃 불안정** | 1 Epoch Warm-up + 1 Epoch Linear Ramp + EMA Probe(0.999) 적용 (§3.3). |

---

## 8. 실험 계획 (Ablation)

| ID | 설정 | 목적 |
|---|---|---|
| **A** | **`v8_fix` (현재 `tmux 19` 학습 중)** | 클린 베이스라인 (`plan_heading_from_xy=True` + ReSMap `flip_w=False`) |
| **Step 0** | Standalone Teacher Probe (`P_det`, `P_map`) + $N_{\text{eff}}$ 튜닝 | 본 학습 전 10분 사전 검증 (AUC 및 Attention 히트맵 확인) |
| **B** | `v8_fix` + 보정된 $w_k$ ($N_{\text{eff}} \approx 35$) | 앵커 선택성(Sharpening)만의 단독 효과 분리 |
| **C (v9 Main)** | **B + 도메인 특화 PGP (`L=1`, EMA, $\lambda_t=0$)** | **Planning-Grounded Probe의 핵심 효과 검증** |
| **D** | C + 약한 `L_task^S` ($\lambda_t = 0.05$) | Student 직접 task grounding의 추가 이득 유무 확인 |
| **E** | (대조) C에서 Attention KD 제거, Feature KD만 사용 | 학습된 Attention Map 정렬 자체의 기여도 분리 |

---

## 9. 오픈 이슈
- Step 0 Standalone Probe 실험에서 Global Unbiased Head(`Head 4`)의 엔트로피가 너무 높게 나올 경우의 temperature scaling 계수
- `resmap_scale`(현재 3.0)을 도메인 특화 Probe 도입 후에도 3.0으로 유지할지, 아니면 `gnorm` 비율(1:1 또는 1:2)로 자동 조절할지 여부
- 증류 시 고정된 256 raw 앵커 대신 Student 플래너가 예측한 refined 궤적(`anchor + offset.detach()`)을 Query로 쓰는 Dynamic Corridor PGP 확장 가능성
