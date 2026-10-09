# BEV Selector Distillation v7
## Trajectory-Anchor Spatial Grounding & Teacher Attention Alignment

**기준 문서**: [`bev_selector_v6.md`](bev_selector_v6.md), [`bev_selector_v5_distill_upgrade.md`](bev_selector_v5_distill_upgrade.md)  
**소속**: `/workspace/byounggun/SSR`  
**작성일**: 2026-10-09  
**실행 대상**: GPU 6, 7  
**학생 백본**: ResNet-34 (`resnet34.tv_in1k`)  
**학생 아키텍처**: Pure Plan-Only (No Det/Map aux heads, No task interaction, 256 Anchor Planner)  

---

## 1. 배경 및 문제의식: v6의 0.865 병목과 W&B 분석

### 1.1 v6의 성과와 한계
- **성과**: 순수 Plan-Only ResNet-50 베이스라인(PDMS **0.8485**) 대비, 더 가벼운 **ResNet-34** 백본으로 **0.8647 (+1.62%p)**을 달성했습니다.
- **한계**: 반면 3D Detection 및 Map 보조 헤드를 붙여 학습한 풀 모델(PDMS **0.8773 ~ 0.88**)에는 약 1.3%p 미치지 못했습니다.

### 1.2 12,146개 전체 시나리오 세부 지표 분석
| 지표 | Baseline (R50 Plan-Only) | **v6 Distill (R34 Plan-Only)** | **Aux Head 모델 (R50+Aux+DPD)** | 격차 분석 |
| :--- | :---: | :---: | :---: | :---: |
| **PDMS** | 0.8485 | **0.8647** | **0.8773** | -1.26%p |
| **NC (충돌 방지)** | 0.9777 | **0.9856** | **0.9874** | **-0.18%p (거의 동등 수준 도달)** |
| **TTC (안전 여유)** | 0.9378 | **0.9498** | **0.9548** | -0.50%p |
| **DAC (차선/도로 준수)** | 0.9334 | **0.9466** | **0.9558** | **-0.92%p (주요 병목)** |
| **EP (주행 진행도)** | 0.7937 | **0.8016** | **0.8150** | **-1.34%p (주요 병목)** |

> **핵심 관찰**: 장애물 회피(NC)는 0.9856으로 Aux 모델에 근접했으나, **도로 경계 준수(DAC)**와 **진행도(EP)**가 전체 점수의 발목을 잡았습니다.

### 1.3 W&B 학습 로그에서 드러난 3가지 구조적 결함
1. **비정상적 Cosine 포화(0.983)와 Attention 지름길(Shortcut)**:
   - v6의 Cross-Attention(`self.attn_det`, `self.attn_map`)은 사전학습 없이 distill 손실로만 동시에 최적화되었습니다.
   - Student와 Teacher가 동일한 미학습 어텐션 레이어를 통과하다 보니, 어텐션 레이어가 5,000칸을 균일하게 뭉개는(Global Blurring) **지름길(Shortcut)**을 찾아버려 W&B 상 Cosine은 98.3%로 완벽해 보이지만 실제 고해상도 도로 경계선(Boundary) 특징은 전혀 전달되지 못했습니다.
2. **BEV 2D Positional Encoding의 부재 (Spatial Agnosia)**:
   - BEV 토큰 5,000칸에 위치 인코딩(`bev_pos`)이 더해지지 않아, Cross-Attention이 좌표를 모르는 순서 없는 집합(Bag of Tokens)으로 취급했습니다. 좌회전 궤적이 전방 좌측 차선을 특정해서 볼 수 없었습니다.
3. **ReSMap(지도) 손실의 그래디언트 기아 (Gradient Starvation)**:
   - `loss_distill_bevfusion` (0.762) 대비 `loss_distill_resmap` (0.257)이 1/3로 급락하여, 역전파 그래디언트를 BEVFusion이 독점하고 ReSMap(차선 준수) 신호가 소멸했습니다.

---

## 2. v7 아키텍처 및 핵심 방법론

v7은 Cross-Attention의 동적이고 우아한 설계를 100% 보존하면서, 기하학적 정렬과 지름길 차단을 달성합니다.

```
       [256 Trajectory Anchors (x, y)]
                      │
       ┌──────────────┴────────────────────────┐
       ▼ (Geometry)                            ▼ (Intent Query)
[2D Gaussian Spatial Bias M_k]       [256 Anchor Queries Q_k [B, 256, C]]
       │                                       │
       │     ┌─────────────────────────────────┤
       │     ▼                                 ▼
       │  [Teacher Cross-Attention]       [Student Cross-Attention]
       │  K_t = proj(t_bev + bev_pos)     K_s = proj(s_bev + bev_pos)
       │  V_t = proj(t_bev)               V_s = proj(s_bev)
       │     │                                 │
       └───► ├─────────────────────────────────┤ ◄───┘
             ▼                                 ▼
       Logits_t = Q·K_t^T + M_k          Logits_s = Q·K_s^T + M_k
             │                                 │
             ▼                                 ▼
       [Teacher Attn A_t]               [Student Attn A_s]
             │                                 │
             ├──────── [Attention KD] ─────────┤
             │     KL(A_s || A_t.detach())     │
             ▼                                 ▼
       feat_t = A_t · V_t                feat_s = A_s · V_s
             │                                 │
             └───────── [Feature KD] ──────────┘
                 Hybrid L2 + Cosine Similarity
```

### 2.1 BEV 2D Positional Encoding 주입 (공간 좌표 인지)
- 플래너 디코더와 동일한 `LearnedPositionalEncoding`을 Cross-Attention Key에 결합:
  $$K_t = \text{Linear}(t\_bev + \text{bev\_pos}), \quad K_s = \text{Linear}(s\_bev + \text{bev\_pos})$$
- 256개 앵커 쿼리가 5,000개 BEV 토큰의 정확한 2차원 공간 좌표 $(x, y)$를 인지하여 공간적 탐색을 수행합니다.

### 2.2 Soft Relative Spatial Bias (구조 파괴 없는 기하 정렬)
- 하드코딩된 영역 자르기(`grid_sample`) 대신, Attention 로짓에 궤적 앵커 물리 좌표와의 거리에 따른 부드러운 **Gaussian Spatial Distance Bias**를 주입:
  $$\text{dist}(k, (x, y)) = \min_{t=0..7} \sqrt{(x_{\text{grid}} - x_{\text{anchor}, k, t})^2 + (y_{\text{grid}} - y_{\text{anchor}, k, t})^2}$$
  $$M_k(x, y) = -\frac{\text{dist}(k, (x, y))^2}{2 \sigma_{\text{spatial}}^2} \quad (\sigma_{\text{spatial}} = 4.0\text{ m})$$
  $$\text{Logits}_{k, (x, y)} = \frac{Q_k K_{(x, y)}^T}{\sqrt{d}} + M_k(x, y)$$
- **효과**:
  - `TrajectoryAnchorDistillation` 초기화 시 256개 고정 앵커에 대해 `[256, 5000]` 크기로 1회 사전 연산(5.1MB)되어 런타임 지연 0!
  - 궤적 선상($\pm 4\text{m}$)을 우선적으로 탐색하되, 끼어드는 차량이나 교차로 회전 차선이 나타나면 Attention 가중치로 자유롭게 포착(Dynamic Cross-Attention 장점 100% 유지).

### 2.3 Teacher-Guided Attention Map 증류 (Shortcut 원천 차단)
- Teacher의 고해상도 LiDAR 및 HD Map이 주목한 Attention 분포 $\mathbf{A}_t$를 정답 라벨로 활용:
  $$\mathbf{A}_t = \text{Softmax}(\text{Logits}_t), \quad \mathbf{A}_s = \text{Softmax}(\text{Logits}_s)$$
  $$\mathcal{L}_{\text{attn}} = \sum_{k=1}^{256} w_k \cdot \text{KL}(\mathbf{A}_{s, k} \,\|\, \mathbf{A}_{t, k}.\text{detach}())$$
- **효과**: 미학습 어텐션 레이어가 양쪽을 뭉개는 지름길을 봉쇄하고, "교차로에서 어떤 도로 경계와 차량을 주목해야 하는지" Teacher의 시선을 학생이 직접 학습.

### 2.4 ReSMap 3.0x 도메인 가중치 리밸런싱 (DAC 회복)
- BEVFusion 대비 1/3로 작았던 ReSMap 손실에 $3.0\times$ 배수를 부여하여 그래디언트 균형 회복:
  $$\mathcal{L}_{\text{det}} = 10.0 \cdot \mathcal{L}_{\text{feat, det}} + 5.0 \cdot \mathcal{L}_{\text{attn, det}}$$
  $$\mathcal{L}_{\text{map}} = \mathbf{3.0} \cdot \left( 10.0 \cdot \mathcal{L}_{\text{feat, map}} + 5.0 \cdot \mathcal{L}_{\text{attn, map}} \right)$$
  $$\mathcal{L}_{\text{distill}} = \frac{\mathcal{L}_{\text{det}} + \mathcal{L}_{\text{map}}}{2}$$

### 2.5 Sharpened Winner & Softmax Importance Weighting
- GT 궤적과의 거리 기반 가중치 집중도를 상향:
  $$\tau = 1.0 \quad (\text{기존 } 2.0 \to \text{유력 후보 집중}), \quad \alpha_{\text{winner}} = 2.0 \quad (\text{기존 } 1.0)$$

---

## 3. 버전별 발전 요약 (v1 ~ v7)

| 항목 | v4 | v5 | v6 | **v7 (본 방법론)** |
|---|---|---|---|---|
| **Query 주체** | 32 Learnable Registers | 32 Learnable Registers | 256 Physical Anchors | **256 Physical Anchors** |
| **선택 메커니즘** | Cross-Attn + Gauss Anchor | Cross-Attn + Gauss Anchor | Unconstrained Cross-Attn | **Spatial-Biased Cross-Attention** |
| **공간 좌표 (Pos)** | 없음 | 없음 | 없음 | **2D Learned Positional Encoding** |
| **Attention 지도** | 없음 | 없음 | 없음 (양쪽 동시 최적화) | **Teacher Attention Map KD (KL)** |
| **ReSMap 가중치** | 1.0 (기아 발생) | 1.0 (기아 발생) | 1.0 (기아 발생) | **3.0x (DAC 그래디언트 복원)** |
| **Student Task** | Det + Map + Plan | Det + Map + Plan | Pure Plan-Only | **Pure Plan-Only (R34)** |
| **추론 오버헤드** | 0 (탈착) | 0 (탈착) | 0 (탈착) | **0 (탈착, 가장 가볍고 빠름)** |

---

## 4. 소스 코드 구현 위치

1. **증류 모듈**: [`navsim/agents/para_ssr/distill/anchor_distill.py`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/distill/anchor_distill.py)
   - `TrajectoryAnchorDistillation`: 2D `LearnedPositionalEncoding`, `spatial_bias` 버퍼, Multi-Head Projections, Attention KL loss, ReSMap 3.0x 스케일링.
2. **파이프라인 연동**: [`navsim/agents/para_ssr/distill/distillation.py`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/distill/distillation.py)
   - `selector_v7` 구성, `bev_pos` 전달.
3. **플래너 헤드**: [`navsim/agents/para_ssr/modules/planner_head.py`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/planner_head.py)
   - `_anchor_outputs`에서 `bev_pos` 온전 보존.
4. **Hydra 설정**: [`navsim/planning/script/config/common/agent/para_ssr_selector_v7_agent.yaml`](file:///workspace/byounggun/SSR/navsim/planning/script/config/common/agent/para_ssr_selector_v7_agent.yaml)
5. **실행 스크립트**: [`scripts/training/run_bev_selector_v7_distill.sh`](file:///workspace/byounggun/SSR/scripts/training/run_bev_selector_v7_distill.sh)

---

## 5. 실행 명령 (GPU 6, 7)

```bash
cd /workspace/byounggun/SSR

# GPU 6, 7 백그라운드 학습 시작 (W&B 로깅 활성화)
CUDA_VISIBLE_DEVICES=6,7 nohup bash scripts/training/run_bev_selector_v7_distill.sh \
  > /workspace/byounggun/SSR/work_dirs/v7_train.log 2>&1 &
```

* 실시간 로그 확인:
```bash
tail -f /workspace/byounggun/SSR/work_dirs/v7_train.log
```
