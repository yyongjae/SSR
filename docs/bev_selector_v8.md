# BEV Selector Distillation v8
## Hybrid Anchor Weighting & Multi-Scale Spatial Distance Bias

**기준 문서**: [`bev_selector_v7.md`](bev_selector_v7.md), [`bev_selector_v7_2.md`](bev_selector_v7_2.md), [`bev_selector_v7_2_future_expected_issue.md`](bev_selector_v7_2_future_expected_issue.md)  
**소속**: `/workspace/byounggun/SSR`  
**작성일**: 2026-10-12  
**실행 대상**: GPU 6, 7  
**학생 백본**: ResNet-34 (`resnet34.tv_in1k`)  
**학생 아키텍처**: Pure Plan-Only (No Det/Map aux heads, No task interaction, 256 Anchor Planner)  

---

## 1. 배경 및 긴급 결함 진단 (Post-Mortem of v7 / v7_2)

v7 및 v7_2 체크포인트의 내부 가중치 분석 과정에서 발견된 **2가지 중대한 버그**와 이를 해결하기 위한 개선점입니다.

### 1.1 🔴 결함 1: 증류 모듈 파라미터가 옵티마이저에서 누락됨 (Frozen Identity)
* **원인**: 
  - `get_optimizers()`가 `self.para_ssr_model`의 파라미터만 옵티마이저에 등록하고 있었습니다. 이는 v1~v5 단계에서 사전학습된 어댑터를 동결(`frozen`)하기 위해 의도되었던 설계였으나, v6/v7의 `TrajectoryAnchorDistillation` 모듈 내 학습 대상 파라미터(`student_proj`, `q_proj`, `k_proj`, `v_proj`, `out_proj`)까지 전부 옵티마이저에서 빠지게 되었습니다.
* **실측 확인**:
  - v7 epoch 29 체크포인트(`epoch=29-step=19950.ckpt`) 및 진행 중이던 v7_2 체크포인트 검증 결과:
    - `student_proj`의 가중치는 완벽한 단위행렬($\max |W - I| = 0.0$)로 전혀 학습되지 않음.
    - `q_proj`, `k_proj`, `v_proj`, `out_proj`의 Frobenius norm이 초기화 이론값($9.24$)과 완벽히 동일.
* **영향 및 통찰**:
  - v7이 실제로 수행한 것은 **"초기화된 랜덤 투영을 거친 GT 궤적 주변 BEV 피처의 L2/Cosine 증류"**였습니다.
  - Teacher-Student가 동일한 미학습 랜덤 $Q, K$를 공유했기 때문에 Attention 행렬이 양쪽 모두 등방성 가우시안 공간 분포로 수렴했고, 이로 인해 Attention KL이 $\approx 0.013$ nats로 비정상적으로 작게 나왔습니다.
* **v8 수정 및 Collapse 방지 설계**:
  - `self._distill.student_proj` (Det/Map 각 1x1 conv)를 `get_optimizers()`의 `other_params`에 명시적으로 등록하여 정상 학습되도록 수정했습니다.
  - **Shared Q/K/V/out_proj의 동결 (`requires_grad_(False)`)**:
    - 만약 Teacher와 공유하는 $Q, K, V$ 투영기를 Student의 어텐션 KL 손실로 역전파하면, $q\_proj \to 0$으로 수렴하여 어텐션이 균일 분포가 되면서 손실이 $0$으로 붕괴하는 **자명해(Degenerate Collapse)**가 발생합니다.
    - 따라서 교차 어텐션 모듈 전체는 **불변의 기하학적 공간 탐침(Invariant Spatial Probe)**으로 동결하고, 학생의 1x1 선형 프로젝터(`student_proj`)만 역전파로 최적화하여 학생 BEV 공간이 Teacher BEV 공간으로 정확히 정렬되도록 격리했습니다.

### 1.2 🟠 결함 2: 증류 손실의 3배 중복 합산 (Loss Double Counting)
* **원인**:
  - 개별 브랜치 손실인 `loss_distill_bevfusion`과 `loss_distill_resmap` 외에, 토큰 평균 손실인 `loss_distill_tok`이 `losses` 딕셔너리에 추가로 담겨 최종 손실에 합산되었습니다.
  - 이로 인해 실제 증류 가중치는 문서에 명시된 $0.5 \times (\text{det} + \text{map})$이 아니라 **$1.5 \times (\text{det} + \text{map})$ (3배 과적합)**으로 주입되었습니다.
* **v8 수정**:
  - 각 브랜치 손실을 `0.5 * bank_loss`로 정규화했습니다.
  - `loss_distill_tok` 및 `loss_attn_*`은 `losses`에서 제외하고 `metrics`로 이동시켜 순수 W&B 텔레메트리 모니터링용으로만 분리했습니다.

---

## 2. v8 핵심 방법론 (Architectural Innovations)

```
             [256 Physical Trajectory Anchors]
                             │
     ┌───────────────────────┴───────────────────────┐
     ▼ (Multi-Scale Geometry)                        ▼ (Hybrid Importance)
[Per-Head Spatial Distance Bias]          [Hybrid Anchor Weighting w_k]
  Head 1: σ = 2m  (Corridor Center)         • Relaxed GT: τ_gt = 5.0 (λ_gt = 0.6)
  Head 2: σ = 4m  (Lane Boundaries)         • Uniform Cov: 1/256     (λ_cov = 0.2)
  Head 3: σ = 8m  (Adjacent Cut-ins)        • Risk Explore: r_k      (λ_risk = 0.2)
  Head 4: σ = 16m (Global Context)                   │
     │                                               ▼
     │                                    [N_eff Monitoring: 20~60]
     │                                               │
     ▼                                               │
[Teacher Attention A_t] ◄── [Attention KD] ──► [Student Attention A_s]
  (LiDAR + Map Guided)      KL(A_s || A_t)       (Trainable student_proj)
     │                                               │
     ▼                                               ▼
[Teacher Attended Feat] ◄─── [Feature KD] ───► [Student Attended Feat]
                           Weighted Hybrid
```

### 2.1 하이브리드 앵커 가중치 설계 (Hybrid Anchor Weighting)
기존 v7은 $\tau=1.0, \alpha_{\text{winner}}=2.0$으로 인해 상위 2~3개 앵커에 가중치의 90% 이상이 집중되는 **국소성 편향(Locality Blindness)**이 발생했습니다. v8은 전체 공간 커버리지와 안전성(Hard Negatives)을 고루 반영하는 하이브리드 수식을 도입합니다:

$$w_k = \lambda_{\text{gt}} w_{\text{gt}, k} + \lambda_{\text{cov}} w_{\text{cov}, k} + \lambda_{\text{risk}} r_k, \quad \sum_{k=1}^K w_k = 1$$

1. **완화된 정답 항 ($w_{\text{gt}}$)**:
   $$w_{\text{gt}, k} = \text{Softmax}\left(-\frac{\text{dist}(\text{anchor}_k, \text{gt\_traj})}{\tau_{\text{gt}}}\right), \quad \tau_{\text{gt}} = 5.0$$
   - 온도를 $5.0$으로 완화하여 주행 경로 주변 $20 \sim 50$개의 유효 주행 후보군으로 학습 시야를 대폭 확장합니다.
2. **균일 커버리지 항 ($w_{\text{cov}}$)**:
   $$w_{\text{cov}, k} = \frac{1}{K} = \frac{1}{256}$$
   - 256개 전체 앵커에 기본 가중치를 분배하여, Teacher가 감지한 전방위 도로 환경 지식이 한 군데도 누락되지 않도록 보장합니다.
3. **위험 탐색 항 ($r_k$, Hard Negatives)**:
   $$r_k = \frac{\text{Softmax}\left(-\frac{d_k}{\tau_{\text{risk}}}\right) \cdot (1 - \text{Safety}_k)}{\sum_j \text{Softmax}\left(-\frac{d_j}{\tau_{\text{risk}}}\right) \cdot (1 - \text{Safety}_j)}, \quad \tau_{\text{risk}} = 8.0$$
   - PDM 시뮬레이션 보상 지표(`sim_reward`, NC 및 DAC)를 결합하여 $\text{Safety}_k = \text{NC}_k \times \text{DAC}_k$를 산출합니다.
   - 충돌 위험이 있거나 도로 연석을 아슬아슬하게 침범하는 앵커에 선택적으로 집중하여, 위험 궤적이 **"왜 위험한지"**에 대한 Teacher의 인지 표현을 학생에게 전수합니다.
4. **가중치 하이퍼파라미터**:
   - $\lambda_{\text{gt}} = 0.6, \quad \lambda_{\text{cov}} = 0.2, \quad \lambda_{\text{risk}} = 0.2$
5. **실시간 모니터링 ($N_{\text{eff}}$)**:
   $$N_{\text{eff}} = \exp\left(-\sum_{k=1}^K w_k \log w_k\right)$$
   - W&B 지표 `distill/effective_anchors`로 기록되며, 목표 범위인 **$20 \sim 60$개**가 안정적으로 유지됩니다.

---

### 2.2 멀티스케일 공간 거리 편향 (Multi-Scale Spatial Distance Bias)
단일 $\sigma=4.0\text{m}$ 편향 대신, Multi-Head Cross-Attention의 각 헤드(4개 헤드)마다 공간적 탐색 반경을 차등화합니다:

$$\mathbf{\sigma} = [2.0\text{ m}, \ 4.0\text{ m}, \ 8.0\text{ m}, \ 16.0\text{ m}]$$

$$M_{h, k}(x, y) = -\frac{\text{dist}(k, (x, y))^2}{2 \sigma_h^2}$$

$$\text{Logits}_{h, k, (x, y)} = \frac{Q_{h, k} K_{h, (x, y)}^T}{\sqrt{d}} + M_{h, k}(x, y)$$

* **Head 1 ($\sigma = 2\text{m}$)**: 정밀 차로 중앙 및 차간 거리 유지.
* **Head 2 ($\sigma = 4\text{m}$)**: 차선 경계선 및 연석(Curb) 회피 (DAC 직접 방어).
* **Head 3 ($\sigma = 8\text{m}$)**: 인접 차선 및 끼어드는 차량 포착.
* **Head 4 ($\sigma = 16\text{m}$)**: 교차로 형태, 신호등, 대형 시나리오의 전역적 맥락 파악.
* **효과**: 
  - 256개 앵커에 대해 `[4, 256, 5000]` 텐서로 1회 사전 계산되어 **런타임 오버헤드 0, 추가 파라미터 0**!

---

## 3. 버전별 비교 요약 (v6 ~ v8)

| 항목 | v6 | v7 | **v8 (현재)** |
|---|---|---|---|
| **Student 백본** | ResNet-34 Plan-Only | ResNet-34 Plan-Only | **ResNet-34 Plan-Only** |
| **Student Projection** | 미등록 (미학습 단위행렬) | 미등록 (미학습 단위행렬) | **옵티마이저 등록 (정상 학습)** |
| **QKV Probe 상태** | 미학습 양쪽 최적화 (Shortcut) | 미학습 양쪽 최적화 (동결 누락) | **Invariant Frozen Probe (Collapse 방지)** |
| **손실 합산 버그** | 3배 중복 합산 | 3배 중복 합산 | **완전 해결 (0.5x 정규화)** |
| **공간 편향 ($\sigma$)** | 없음 | 단일 $\sigma=4.0\text{m}$ | **Multi-Scale $[2, 4, 8, 16]\text{m}$** |
| **앵커 가중치 수식** | Softmax ($\tau=2.0$) | Sharpened ($\tau=1.0$, boost=2.0) | **하이브리드 (GT 0.6 + Cov 0.2 + Risk 0.2)** |
| **유효 앵커 수 ($N_{\text{eff}}$)** | ~5개 | ~2개 | **$20 \sim 60$개 (광범위 균형)** |
| **W&B 모니터링** | Cosine 포화 (0.983) | Cosine (0.978), KL (0.013) | **실효 앵커 수, Attention KL, Cosine** |

---

## 4. 소스 코드 수정 위치

1. **옵티마이저 및 프로브 동결**:
   - [`navsim/agents/para_ssr/para_ssr_agent.py`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/para_ssr_agent.py): `student_proj` 파라미터 등록, 공유 어텐션 및 쿼리 생성기 `requires_grad_(False)` 설정.
2. **증류 손실 계산 및 가중치**:
   - [`navsim/agents/para_ssr/distill/anchor_distill.py`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/distill/anchor_distill.py):
     - `_compute_anchor_weights`: 하이브리드 수식 및 $N_{\text{eff}}$ 산출.
     - `_build_spatial_bias`: 멀티스케일 `[H, K, HW]` 바이어스 생성.
     - `forward`: 손실 0.5x 정규화 및 `loss_distill_tok` 텔레메트리 분리.
3. **설정 및 인터페이스**:
   - [`navsim/agents/para_ssr/configs/default.py`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/configs/default.py): v8 파라미터 추가.
   - [`navsim/planning/script/config/common/agent/para_ssr_selector_v8_agent.yaml`](file:///workspace/byounggun/SSR/navsim/planning/script/config/common/agent/para_ssr_selector_v8_agent.yaml): v8 Hydra 에이전트 설정.
4. **실행 스크립트**:
   - [`scripts/training/run_bev_selector_v8_distill.sh`](file:///workspace/byounggun/SSR/scripts/training/run_bev_selector_v8_distill.sh): GPU 6, 7 분산 학습 런처.

---

## 5. 실행 가이드 (GPU 6, 7)

### 5.1 기존 v7_2 프로세스 정리
```bash
# tmux 19 세션 정리 또는 PID 1650517, 1651532 종료
kill -9 1650517 1651532 2>/dev/null || true
```

### 5.2 v8 학습 시작
```bash
cd /workspace/byounggun/SSR

# GPU 6, 7 백그라운드 학습 시작 (W&B 로깅 활성화)
CUDA_VISIBLE_DEVICES=6,7 nohup bash scripts/training/run_bev_selector_v8_distill.sh \
  > /workspace/byounggun/SSR/work_dirs/v8_train.log 2>&1 &
```

* 실시간 학습 로그 모니터링:
```bash
tail -f /workspace/byounggun/SSR/work_dirs/v8_train.log
```
