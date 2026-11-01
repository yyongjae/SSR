# BEV Selector Distillation v6
## Trajectory-Anchor-Guided Distillation with Pure Plan-Only Student

**기준 문서**: [`bev_selector_v5_distill_upgrade.md`](bev_selector_v5_distill_upgrade.md), [`bev_selector_v4.md`](bev_selector_v4.md)  
**소속**: `/workspace/byounggun/SSR`  
**작성일**: 2026-10-08  
**실행 대상**: GPU 6, 7  

---

## 1. 배경 및 문제의식: 복잡성의 악순환 탈피

### 1.1 이전 버전(v1 ~ v5)의 구조적 한계와 복잡성
- **추상적 32개 Register의 방황**:
  - v1~v5의 BEV Selector는 주행 궤적과 직접적인 기하학적 연관성이 없는 32개의 추상적인 학습 파라미터(Learnable Tokens)를 Query로 사용했습니다.
  - 이 점들이 한 곳으로 뭉치는 붕괴(Collapse)를 막기 위해 `Diversity Loss`를 도입했고, 플래너 시선을 쫓아가도록 유도하기 위해 `Coverage Loss`를 강제해야 했습니다.
- **Auxiliary 헤드(3D Det, Map)와의 극심한 충돌**:
  - Student 모델이 거대한 보조 태스크(900개 Detection 쿼리, 100개 Map 쿼리)를 함께 학습하면서 발생하는 그래디언트($0.03 \sim 0.05$)가 증류 그래디언트($0.0005$)를 압도했습니다.
  - 이를 인위적으로 억제하기 위해 Closed-loop `GradBalancer`(`{plan:0.4, det:0.3, map:0.3}`)를 도입했으나, 이는 근본적인 태스크 간 간섭(Negative Transfer)을 해결하지 못했습니다.

### 1.2 핵심 전환: "Trajectory-Centric Distillation + Plan-Only Student"
- **DistillDrive (arXiv:2508.05402)의 핵심 통찰**:
  - 복잡한 임의의 레지스터 대신, **"실제 주행할 궤적(Trajectory)"**이 Query가 되어 Teacher에게서 내 경로에 필요한 물리적 정보만 직접 요청하는 방식이 가장 정답에 가깝습니다.
- **v6의 대담한 결정**:
  1. **Auxiliary Task (Det/Map) 100% 제거**: Student는 오직 주행 계획(Planning)만 수행합니다.
  2. **256 Trajectory Anchor 기반 직접 증류**: Student의 256개 K-Means 앵커 물리 좌표를 따라 Teacher의 고해상도 특징을 직접 캐와서 증류합니다.
  3. **보조 손실 & GradBalancer 완전 제거**: Diversity Loss, Coverage Loss, Closed-loop GradBalancer를 모두 걷어내어 순수한 Planning + Distillation 단일 목표로 수렴시킵니다.

---

## 2. v6 아키텍처 및 핵심 방법론

```
[256 Trajectory Anchors (물리 좌표)] + Command + Ego Status
                    │
                    ▼  (MLP Encoding)
       [256 Anchor Queries Q_k [B, 256, C]]
         │                             │
         ├──────────────────────┐      │
         ▼                      ▼      ▼
[Cross-Attention: BEVFusion]  [Cross-Attention: ReSMap]  [Cross-Attention: Student BEV]
  - 3D Obstacle Tokens          - Road Boundary Tokens     - Camera-only BEV Tokens
  - 충돌 상호작용 셀렉션        - 차선 준수 상호작용 셀렉션
         │                             │                           │
         ▼                             ▼                           ▼
[Teacher Det Feat F_t,det]    [Teacher Map Feat F_t,map]   [Student Feat F_s,det & F_s,map]
  (B, 256, C)                   (B, 256, C)                  (B, 256, C)
         └─────────────────────────────┼───────────────────────────┘
                                       ▼
                          [Winner & Softmax Weighting]
                          w_k = Softmax(-d_k / τ) + α·1[k=k*]
                                       ▼
                        [Hybrid Anchor Distillation Loss]
                        L_distill = 10.0 · (L_bevfusion + L_resmap)
```

### 2.1 Pure Plan-Only Student (Aux Heads 100% 절제)
- `use_det_motion_head = false`, `use_map_head = false`, `use_task_interaction = false`
- Student는 3D Detection 헤드, Map 세그멘테이션 헤드, 헝가리안 매칭, Chamfer 손실을 완전히 배제합니다.
- 대신 256개 K-means Trajectory Anchors가 3계층 Transformer Decoder를 통해 Student BEV와 직접 Cross-Attention하여 궤적 오프셋 및 5대 PDM 시뮬레이션 보상(NC, DAC, EP, TTC, Comfort)을 직접 예측합니다.
- **효과**: 태스크 간 충돌 0, 학습 속도 2배 향상, 추론 속도 대폭 개선!

### 2.2 Dynamic Cross-Attention Interaction Selection (방법 2: 핵심 메커니즘)
단순히 경로 선 위의 점(Centerline)만 긁어오는 것은 궤적 주변의 끼어드는 차량이나 차선 곡률을 놓칠 위험이 있습니다.
따라서 v6는 **"256개 앵커 쿼리(Query)가 Teacher BEV 전체를 동적으로 훑어서(Cross-Attention) 상호작용하는 핵심 셀들을 스스로 선택"**하도록 설계되었습니다.

1. **256 앵커 의도 쿼리 생성**:
   - 256개 물리 궤적 $[256, 8, 3]$을 MLP로 투영하고, 주행 명령(Command) 및 Ego 상태(속도/가속도)를 주입:
     $$Q_k = \text{MLP}(\text{Anchor}_k) + \text{MLP}(\text{Command}) + \text{MLP}(\text{Ego\_Status}) \quad \in \mathbb{R}^{B \times 256 \times C}$$
   - *이 쿼리는 "30km/h로 좌회전하려는 내 주행 계획 입장에서 주의해야 할 환경이 무엇인가?"라는 질문을 담고 있습니다.*

2. **Teacher별 상호작용 셀렉션 (Cross-Attention)**:
   - **Teacher 1 (BEVFusion)**: 3D 장애물 전문가 BEV에 $Q_k$가 Attention을 걸어 **내 궤적의 진행을 가로막거나 충돌할 위험이 있는 차량/보행자 셀**을 집중 선택!
     $$F_{t, \text{det}, k} = \text{CrossAttn}(Q_k, \text{BEVFusion\_BEV}, \text{BEVFusion\_BEV})$$
   - **Teacher 2 (ReSMap)**: HD Map 도로 경계 전문가 BEV에 $Q_k$가 Attention을 걸어 **내 궤적이 준수해야 할 차선 및 도로 경계선 셀**을 집중 선택!
     $$F_{t, \text{map}, k} = \text{CrossAttn}(Q_k, \text{ReSMap\_BEV}, \text{ReSMap\_BEV})$$

3. **Student 특징 추출 및 1:1 정렬**:
   - Student BEV에도 동일한 어텐션 레이어를 통해 상호작용 특징을 추출:
     $$F_{s, \text{det}, k} = \text{CrossAttn}(Q_k, \text{Proj}_{\text{det}}(\text{Student\_BEV}), \text{Proj}_{\text{det}}(\text{Student\_BEV}))$$
     $$F_{s, \text{map}, k} = \text{CrossAttn}(Q_k, \text{Proj}_{\text{map}}(\text{Student\_BEV}), \text{Proj}_{\text{map}}(\text{Student\_BEV}))$$
   - Student는 Teacher가 해당 궤적에 대해 반응한 상호작용 특징($F_t$)을 완벽하게 재현하도록 학습됩니다.

### 2.3 Winner & Softmax Importance Weighting
- 모든 256개 앵커를 동일한 비중으로 증류하면 비현실적인 궤적(역주행, 인도 돌진 등)을 맞추느라 용량을 낭비합니다.
- GT 궤적과의 $L_2$ 거리 $d_k$를 기반으로 중요도 가중치 $w_k$를 부여:
  $$d_k = \| P_k - P_{\text{gt}} \|_2, \quad k^* = \arg\min_k d_k \quad (\text{Winner 앵커})$$
  $$p_k = \text{Softmax}\left(-\frac{d_k}{\tau}\right) \quad (\tau = 2.0)$$
  $$w_k = \frac{p_k + \alpha \cdot \mathbf{1}[k = k^*]}{\sum_{j} (p_j + \alpha \cdot \mathbf{1}[j = k^*])} \quad (\alpha = 1.0)$$
- **효과**: Winner 앵커와 유력 후보 궤적에 증류 역량이 집중되어 주행 판단 정밀도가 극대화됩니다.

### 2.4 Multi-Teacher Domain Decomposition & 손실 함수
- **Teacher 1 (BEVFusion)**: 충돌 방지 보상(**NC, No Collision**) 학습 견인.
- **Teacher 2 (ReSMap)**: 주행 가능 구역 준수 보상(**DAC, Drivable Area Compliance**) 학습 견인.
- **Hybrid (LayerNorm L2 + Cosine Distance) 손실**:
  $$\ell(s, t) = 0.5 \cdot \|\text{LN}(s) - \text{LN}(t)\|^2 + 0.5 \cdot (1 - \cos(\text{LN}(s), \text{LN}(t)))$$
  $$\mathcal{L}_{\text{distill}} = 10.0 \cdot \left( \sum_{k=1}^{256} w_k \ell(F_{s,\text{det},k}, F_{t,\text{det},k}) + \sum_{k=1}^{256} w_k \ell(F_{s,\text{map},k}, F_{t,\text{map},k}) \right)$$

---

## 3. 버전별 비교 요약

| 항목 | v1 ~ v3 | v4 | v5 | **v6 (본 방법론)** |
|---|---|---|---|---|
| **Query 주체** | 32 Learnable Registers | 32 Learnable Registers | 32 Learnable Registers | **256 Physical Trajectory Anchors** |
| **선택 메커니즘** | Cross-Attn + Gauss Anchor | Cross-Attn + Gauss Anchor | Cross-Attn + Gauss Anchor | **Grid Sample Trajectory Corridors** |
| **Student Task** | Det + Map + Plan (다중) | Det + Map + Plan (다중) | Det + Map + Plan (다중) | **Pure Plan-Only (Det/Map OFF)** |
| **보조 손실** | Coverage + Diversity Loss | Coverage + Diversity Loss | Coverage + Diversity Loss | **없음 (0개)** |
| **GradBalancer** | 꺼짐 | 0.4 / 0.3 / 0.3 강제 밸브 | 꺼짐 (Tok scale 25.0) | **완전 불필요 (`null`)** |
| **가중치 방식** | 균등 / Attention 합 | 균등 / Attention 합 | 공간 마스크 부스트 | **Winner + Softmax 중요도 가중치** |
| **추론 오버헤드** | 0 (레지스터 탈착) | 0 (레지스터 탈착) | 0 (레지스터 탈착) | **0 (Aux 헤드도 없어 훨씬 가볍고 빠름)** |

---

## 4. 소스 코드 구현 위치

1. **증류 모듈**: [`navsim/agents/para_ssr/distill/anchor_distill.py`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/distill/anchor_distill.py)
   - `TrajectoryAnchorDistillation` 클래스: 좌표계 변환, Grid-sample 수집, Winner 가중치, Hybrid 손실 계산.
2. **증류 파이프라인 연동**: [`navsim/agents/para_ssr/distill/distillation.py`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/distill/distillation.py)
   - `_anchor_forward`: Teacher BEV 로딩 및 `TrajectoryAnchorDistillation` 호출.
3. **Pure Plan-Only 256 앵커 플래너 브릿지**:
   - [`navsim/agents/para_ssr/modules/planner_head.py`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/planner_head.py): `use_task_interaction=False` 상태에서 256 앵커 플래너 직접 구동 지원.
   - [`navsim/agents/para_ssr/para_ssr_model.py`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/para_ssr_model.py): 보조 헤드 없는 플랜 출력에서 앵커 예측 딕셔너리 온전 보존.
4. **Hydra Agent 설정**: [`navsim/planning/script/config/common/agent/para_ssr_selector_v6_agent.yaml`](file:///workspace/byounggun/SSR/navsim/planning/script/config/common/agent/para_ssr_selector_v6_agent.yaml)
5. **실행 스크립트**: [`scripts/training/run_bev_selector_v6_distill.sh`](file:///workspace/byounggun/SSR/scripts/training/run_bev_selector_v6_distill.sh)

---

## 5. 실행 명령 (GPU 6, 7)

```bash
cd /workspace/byounggun/SSR

# GPU 6, 7 백그라운드 학습 시작 (W&B 로깅 활성화)
CUDA_VISIBLE_DEVICES=6,7 nohup bash scripts/training/run_bev_selector_v6_distill.sh \
  > /workspace/byounggun/SSR/work_dirs/v6_train.log 2>&1 &
```

* 실시간 로그 확인:
```bash
tail -f /workspace/byounggun/SSR/work_dirs/v6_train.log
```
