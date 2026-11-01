# BEV Selector Distillation v7_2
## PARA-SSR v2 Student (Aux Task Heads + Task Interaction) with v7 Trajectory-Anchor Distillation

**기준 문서**: [`bev_selector_v7.md`](bev_selector_v7.md), [`bev_selector_v6.md`](bev_selector_v6.md)  
**소속**: `/workspace/byounggun/SSR`  
**작성일**: 2026-10-11  
**실행 대상**: GPU 6, 7  
**학생 백본**: ResNet-34 (`resnet34.tv_in1k`)  
**학생 아키텍처**: `km/para-ssr-v2` 풀 구조 (Det+Motion Head, Map Head, Task Interaction Layer, 256 Trajectory Anchor Planner)  
**증류 방법론**: v7 Trajectory-Anchor Guided Spatial Grounding & Attention Alignment  

---

## 1. 배경 및 동기 (Why v7_2?)

### 1.1 이전 실험 성과 요약
1. **v7 (ResNet-34 Plan-Only + v7 Distillation)**:
   - **PDMS 0.8671** 달성 (미증류 베이스라인 0.8485 대비 **+1.86%p** 향상).
   - Soft Gaussian Spatial Bias($\sigma=4.0\text{ m}$)와 ReSMap 3.0x 스케일링을 통해 **DAC를 0.9466에서 0.9508로 대폭 개선**.
   - 단, 순수 Plan-Only 구조의 특성상 인지 보조 헤드가 없어 복잡한 동적 인터랙션 상황에서 한계 존재.
2. **`km/para-ssr-v2` (ResNet-34 with Aux Heads & Task Interaction, 증류 없음)**:
   - 3D Detection, Motion, Vectorized Map 헤드 및 Task Interaction Layer 장착.
   - 256개 앵커 플래너 + PDM 점수 보상 학습으로 단독 **PDMS ~0.88** (25 에폭 기준 88.23, EPDMS 87.80) 기록.

### 1.2 핵심 가설: 최강 Student + 최강 Distillation의 결합
- 이미 강력한 표현력을 검증받은 `km/para-ssr-v2` 학생 아키텍처에, 검증된 **v7 증류 방법론(공간 편향 + 어텐션 KD + ReSMap 3.0x)**을 동시에 적용하면:
  1. 학생은 GT 바운딩 박스/차선 라벨로부터 **명시적 기하 구조(Explicit Geometry)**를 학습하고,
  2. Teacher(BEVFusion + ReSMap)로부터 **멀티모달 고해상도 BEV 컨텍스트(Teacher Context)**를 증류받으며,
  3. 플래닝 헤드는 Task Interaction을 통해 **인지 메모리와 증류된 BEV를 동시에 참조**하여 주행 결정을 내립니다.
- **목표 성능**: **PDMS 0.885 ~ 0.895+ 돌파**

---

## 2. v7_2 아키텍처 개요

```
                   [Camera Images (Front 3-Views)]
                                 │
                     [ResNet-34 Backbone + FPN]
                                 │
                    [BEVFormer Encoder (50x100)]
                                 │
                        [Student BEV Embed S]
                     ┌───────────┼───────────┐
                     ▼           ▼           ▼
               [Det+Motion]   [Map Head]  [v7 Distill Branch]
               (3D Boxes/Traj) (Vectors)   │ (BEVFusion & ReSMap)
                     │           │         │ - 2D Spatial Pos Encoding
                     └─────┬─────┘         │ - Soft Gaussian Spatial Bias (M_k)
                           ▼               │ - Teacher Attention Map KD (KL)
                  [Task Memories]          │ - Attended Feature KD (Hybrid)
                           │               │ - ReSMap 3.0x Scale
                           ▼               ▼
                 [Task Interaction Layer (Cross-Attn)]
                           │
                 [256 Trajectory Anchor Planner]
                  (WTA Offset + PDM Simulation Rewards)
                           │
                 [Final Trajectory & PDM Scores]
```

### 2.1 학생 네트워크 구성 (`km/para-ssr-v2`)
- **백본**: ResNet-34 (`resnet34.tv_in1k`)
- **BEV 인코더**: 3 Layers, $50 \times 100$ 그리드, 256 채널
- **보조 헤드 (Aux Heads)**:
  - `use_det_motion_head: true` (300 쿼리, 3D Bounding Box + Future Motion)
  - `use_map_head: true` (100 벡터, 도로 경계/차선/횡단보도)
- **상호작용 플래너**:
  - `use_task_interaction: true`
  - 플래너 디코더 레이어에서 Det/Motion 및 Map 헤드의 디코더 은닉 상태(`memories`)를 Cross-Attention으로 상호 참조.
- **앵커 플래너**:
  - `plan_anchor: true`
  - 256개 K-Means 물리 궤적 앵커 + Winner-Takes-All L1 오프셋 + 모방 Soft-CE + PDM 시뮬레이션 보상(NC, DAC, EP, TTC, C) BCE 손실.
- **그래디언트 밸런서 (GradBalancer)**:
  - 학생 태스크 간 BEV 그래디언트 균형 유지: `plan: 0.4, det: 0.3, map: 0.3`
  - Warmup: 10,600 iters (1 에폭), Interval: 200 iters

### 2.2 v7 증류 손실 구성
- **Key 좌표 인지**: BEV Key에 플래너와 동일한 2D Positional Encoding 결합 ($K_s = \text{proj}(s\_bev + bev\_pos)$)
- **Soft Gaussian Spatial Distance Bias**: 256개 앵커 선상($\sigma=4.0\text{ m}$) 거리에 기반한 기하학적 로짓 바이어스 주입 ($M_k$)
- **Teacher Attention Map KD**: Teacher Cross-Attention 가중치 $A_t$를 모방하는 $\text{KL}(A_s \,\|\, A_t.\text{detach}())$
- **Attended Feature KD**: $feat_s = A_s V_s$ 와 $feat_t = A_t V_t$ 간의 Hybrid L2 + Cosine 손실
- **ReSMap 손실 가중치**: $3.0\times$ (DAC 도로 경계 준수 강화)

---

## 3. 손실 함수 (Total Loss Formulation)

전체 학습 손실은 **학생 멀티태스크 밸런싱 손실**과 **v7 궤적 앵커 증류 손실**의 결합으로 구성됩니다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{student\_balanced}} + \lambda_{\text{distill}} \cdot \mathcal{L}_{\text{distill\_v7}}$$

1. **학생 멀티태스크 손실 ($\mathcal{L}_{\text{student\_balanced}}$)**:
   - Closed-loop GradBalancer에 의해 BEV 역전파 그래디언트 노름이 `plan: 0.4`, `det: 0.3`, `map: 0.3` 비율로 안정화됩니다.
2. **v7 증류 손실 ($\mathcal{L}_{\text{distill\_v7}}$)**:
   $$\mathcal{L}_{\text{distill\_v7}} = \frac{\mathcal{L}_{\text{det\_kd}} + 3.0 \cdot \mathcal{L}_{\text{map\_kd}}}{2}$$
   $$\mathcal{L}_{\text{kd}} = 10.0 \cdot \mathcal{L}_{\text{feat}} + 5.0 \cdot \mathcal{L}_{\text{attn}}$$

---

## 4. 파일 변경 및 설정 매니페스트

1. **Hydra 에이전트 설정**:
   - [`navsim/planning/script/config/common/agent/para_ssr_selector_v7_2_agent.yaml`](file:///workspace/byounggun/SSR/navsim/planning/script/config/common/agent/para_ssr_selector_v7_2_agent.yaml)
2. **모델 코드 보완**:
   - [`navsim/agents/para_ssr/para_ssr_model.py`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/para_ssr_model.py):
     - `use_task_interaction=True` 환경에서 `distill_selector_v7` 활성화 시 `return_bev_attn` 및 `bev_pos`가 누락 없이 전달되도록 수정 완료.
3. **학습 실행 스크립트**:
   - [`scripts/training/run_bev_selector_v7_2_distill.sh`](file:///workspace/byounggun/SSR/scripts/training/run_bev_selector_v7_2_distill.sh)

---

## 5. 실행 명령 (GPU 6, 7)

```bash
cd /workspace/byounggun/SSR

# GPU 6, 7에서 백그라운드 학습 시작 (W&B 로깅 활성화)
CUDA_VISIBLE_DEVICES=6,7 nohup bash scripts/training/run_bev_selector_v7_2_distill.sh \
  > /workspace/byounggun/SSR/work_dirs/v7_2_train.log 2>&1 &
```

* 실시간 학습 로그 확인:
```bash
tail -f /workspace/byounggun/SSR/work_dirs/v7_2_train.log
```

* W&B 모니터링:
  - Project: `v1_distill`
  - Group: `bev-selector-v7_2`
  - Run Prefix: `r34_sel_v7_2`
