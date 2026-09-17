# Dual-Teacher Planning-Aware BEV Distillation (DPD)
## 탑티어(CVPR / ECCV / NeurIPS) 지향 아키텍처 및 연구 계획서

**소속/작업공간**: `/home/external-user/byounggun/SSR`  
**작성일**: 2026-09-16  
**연구 주제**: 동적 3D 객체(BEVFusion)와 정적 HD-Map(ReSMap) 교사의 지식을 주행 복도(Corridor) 마스크 기반으로 카메라 학생 모델에 전이하는 계획 중심 종단간 자율주행 프레임워크

---

## 1. 연구 배경 및 모티베이션 (Motivation)

### 1.1 기존 연구의 한계점
1. **균일한 공간 증류 (Uniform Spatial Distillation)의 비효율성**:
   - 기존의 BEV Feature Distillation은 $50 \times 100$ 전방 격자의 모든 셀(빈 허공, 먼 보도블록, 주행과 무관한 배경)에 대해 균일하게 $L_2$ 손실을 계산한다.
   - 이로 인해 자차가 향후 4초 동안 주행할 궤적(Driving Corridor) 상의 핵심 장애물 및 차선 중심선 정보의 그래디언트가 배경 노이즈에 희석된다.
2. **단일 모달리티 교사의 정보 불균형**:
   - 3D Detection 교사(BEVFusion)는 동적 차량/보행자의 3D 바운딩 박스와 LiDAR 깊이감은 탁월하지만, 도로의 차선 위상(Topology)이나 횡단보도 경계선 정보를 온전히 담지 못한다.
   - 반면 Online HD-Map 교사(ReSMap)는 정적 차선/도로 경계에는 완벽하지만, 움직이는 객체의 3D 물리량(속도, 자세)을 제공하지 못한다.
   - 기존 파이프라인은 이 중 하나(BEVFusion)만 단독으로 증류하여 절반의 특권 정보만 활용하고 있었다.
3. **플래너의 동역학 인지 부재 (Kinematics-Agnostic Planning)**:
   - 기존 SSR 플래너는 단일 쿼리에 단순 명령(Command)만 더해져 있어, 차량의 현재 속도/가속도($v_x, v_y, a_x, a_y$)를 인지하지 못해 물리적 관성을 무시한 궤적이 생성될 위험이 존재했다.

---

## 2. 제안 아키텍처 핵심 구성요소 (Architecture Overview)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          [ Teacher Cache Store ]                            │
│   1. BEVFusion (3D Dynamic Agent Teacher, mAP 0.8236)                       │
│   2. ReSMap    (Online HD-Map Static Teacher, mAP 0.7743)                   │
└──────────────────────┬───────────────────────────────┬──────────────────────┘
                       │                               │
                       ▼                               ▼
       ┌───────────────────────────────┐┌───────────────────────────────┐
       │ Stage-1 Frozen Det Adapter    ││ Stage-1 Frozen Map Adapter    │
       │ (Trained via Trajectory Loss) ││ (Trained via Trajectory Loss) │
       └───────────────┬───────────────┘└───────────────┬───────────────┘
                       │                               │
                       └───────────────┬───────────────┘
                                       │ Dual Teacher Privileged Representation
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               [ Planning-Aware Corridor Spatial Masking ]                   │
│   - Ego Planned Trajectory τ_ego = {(x_t, y_t)}_{t=1}^8                     │
│   - Gaussian Driving Corridor Mask: W(x, y) = max_t exp(-||p - τ_t||^2/2σ²) │
│   - Loss Formulation: L_distill = (1 / ΣW) Σ W(x,y) * ||S(x,y) - T(x,y)||^2 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▲ Feature Distillation Gradient
                                       │
┌──────────────────────────────────────┴──────────────────────────────────────┐
│                     [ Student Autonomous Vehicle Model ]                    │
│   1. Front 3 Cameras (CAM_F0, CAM_L0, CAM_R0) -> ResNet50 + FPN            │
│   2. Temporal Aligned BEVFormer Encoder -> Student Dense BEV [50 x 100]     │
│   3. Kinematics-Conditioned Dense BEV Planner:                              │
│      - Ego Status MLP([vx, vy, ax, ay]) + Command Embedding                 │
│      - Dense BEV Cross-Attention (TokenLearner 배제)                        │
│      - 8-Step Future Trajectory Head (x, y, heading)                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 수식 및 알고리즘

### 3.1 Planning-Aware Driving Corridor Mask ($W_{\text{corridor}}$)
미래 $T=8$ 시점의 궤적 좌표 $\mathbf{p}_t = (x_t, y_t)$에 대해, BEV 격자 좌표 $(x, y)$에서의 주행 복도 가중치 맵은 다음과 같이 정의된다:

$$W(x, y) = \epsilon + (1 - \epsilon) \cdot \max_{t \in \{1, \dots, T\}} \exp\left( -\frac{(x - x_t)^2 + (y - y_t)^2}{2\sigma_t^2} \right)$$

* $\sigma_t = \sigma_0 + \alpha \cdot t$: 미래 시점으로 갈수록 주행 영역의 불확실성을 반영해 점진적으로 반경을 넓힘.
* $\epsilon = 0.1$: 주행 복도 외곽 영역에도 최소한의 정규화 그래디언트를 전달하는 Base Weight.

### 3.2 Dual-Teacher Multi-Objective Loss
$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{plan}} + \lambda_{\text{det}} \mathcal{L}_{\text{distill}}^{\text{det}}(W) + \lambda_{\text{map}} \mathcal{L}_{\text{distill}}^{\text{map}}(W) $$

* $\mathcal{L}_{\text{distill}}^{\text{det}}(W) = \frac{\sum_{x,y} W(x,y) \cdot \| \text{Adapter}_{\text{det}}(S)_{x,y} - \text{Adapter}_{\text{det}}(T_{\text{det}})_{x,y} \|^2}{\sum_{x,y} W(x,y)}$
* $\mathcal{L}_{\text{distill}}^{\text{map}}(W) = \frac{\sum_{x,y} W(x,y) \cdot \| \text{Adapter}_{\text{map}}(S)_{x,y} - \text{Adapter}_{\text{map}}(T_{\text{map}})_{x,y} \|^2}{\sum_{x,y} W(x,y)}$

### 3.3 Kinematics Planning Conditioning
$$ \mathbf{q}_0 = \text{PlanQuery} + \text{Emb}(\text{Command}) + \text{MLP}_{\text{kinematics}}([v_x, v_y, a_x, a_y]) $$
$$ \mathbf{q}_{\text{out}} = \text{CrossAttention}(\mathbf{q}_0, \text{Key}=\text{BEV}_{\text{dense}}, \text{Value}=\text{BEV}_{\text{dense}}) $$
$$ \tau_{\text{pred}} = \text{MLP}_{\text{traj}}(\mathbf{q}_{\text{out}}) \in \mathbb{R}^{8 \times 3} $$

---

## 4. 파이프라인 및 실행 가이드 (Execution Guide)

### 4.1 올인원 원클릭 자동화 실행 (Recommended)
Stage 1A(BEVFusion 어댑터) $\rightarrow$ Stage 1B(ReSMap 어댑터) $\rightarrow$ Stage 2(학생 모델 주행회랑 증류)를 한 번의 명령어로 순차 실행하며, 생성된 어댑터 체크포인트를 Stage 2에 자동으로 연결합니다.

* **기본 할당 GPU**: **GPU 2, 3번** (`CUDA_VISIBLE_DEVICES=2,3`, RTX 5090 32GB × 2)
* **스크립트 경로**: [`scripts/training/run_all_stages_distill.sh`](file:///home/external-user/byounggun/SSR/scripts/training/run_all_stages_distill.sh)

```bash
# [단일 실행 명령어] 백그라운드 실행 및 로그 저장
nohup ./scripts/training/run_all_stages_distill.sh > run_distill.log 2>&1 &

# 실시간 훈련 로그 모니터링
tail -f run_distill.log
```

#### 올인원 파이프라인 자동화 흐름
1. **Stage 1A (BEVFusion Adapter, 20 Epochs)**:
   - 교사 캐시: `/home/external-user/datasets/teacher_cache/bevfusion/cache_{train,val}_50x100`
   - 센서 로딩 없이 경량 어댑터와 플래너만 훈련 (`batch_size=16`, `accumulate=4` $\rightarrow$ 글로벌 배치 128).
   - 훈련 완료 시 `work_dirs/paradrive_distill_stage1_bevfusion/`에서 체크포인트를 자동 탐색해 `BEVFUSION_CKPT`로 설정.
2. **Stage 1B (ReSMap Adapter, 20 Epochs)**:
   - 교사 캐시: `/home/external-user/datasets/teacher_cache/resmap/index.json` (sharded memmap)
   - 온라인 HD-map 교사 특성으로 어댑터 훈련 (`batch_size=16`, `accumulate=4` $\rightarrow$ 글로벌 배치 128).
   - 훈련 완료 시 `work_dirs/paradrive_distill_stage1_resmap/`에서 체크포인트를 자동 탐색해 `RESMAP_CKPT`로 설정.
3. **Stage 2 (Dual-Teacher Distillation Student Training, 30 Epochs)**:
   - 앞선 1A, 1B의 동결된 어댑터 체크포인트를 자동으로 로드.
   - 전방 카메라 3대(`cam_f0, cam_l0, cam_r0`) 기반 학생 모델(Dense BEV Cross-Attention + Kinematics conditioning)에 가우시안 주행 회랑 가중 손실 적용 (`batch_size=4`, `accumulate=16` $\rightarrow$ 글로벌 배치 128).

#### 유연한 실행 옵션 (환경변수 오버라이드)
```bash
# 1. 다른 GPU 번호로 실행하고 싶을 때 (예: GPU 0, 1번)
CUDA_VISIBLE_DEVICES=0,1 ./scripts/training/run_all_stages_distill.sh

# 2. Stage 1은 이미 완료되어 체크포인트가 있고, Stage 2만 실행하고 싶을 때
BEVFUSION_ADAPTER_CKPT=/path/to/stage1_bevfusion.ckpt \
RESMAP_ADAPTER_CKPT=/path/to/stage1_resmap.ckpt \
ONLY_STAGE=stage2 ./scripts/training/run_all_stages_distill.sh

# 3. 특정 1단계 어댑터만 단독 실행하고 싶을 때
ONLY_STAGE=stage1a ./scripts/training/run_all_stages_distill.sh  # BEVFusion만
ONLY_STAGE=stage1b ./scripts/training/run_all_stages_distill.sh  # ReSMap만

# 4. W&B 로깅을 끄고 순수 TensorBoard만 사용할 때
WANDB=0 ./scripts/training/run_all_stages_distill.sh
```

---

### 4.2 개별 단계별 독립 실행 스크립트

필요에 따라 각 단계를 완전히 분리된 스크립트로 직접 실행할 수도 있습니다:

1. **Stage 1A (BEVFusion Adapter)**:
   ```bash
   bash scripts/training/train_stage1_adapter_bevfusion.sh
   ```
2. **Stage 1B (ReSMap Adapter)**:
   ```bash
   bash scripts/training/train_stage1_adapter_resmap.sh
   ```
3. **Stage 2 (Dual-Teacher 본 학습)**:
   ```bash
   BEVFUSION_ADAPTER_CKPT=/path/to/stage1_bevfusion.ckpt \
   RESMAP_ADAPTER_CKPT=/path/to/stage1_resmap.ckpt \
   bash scripts/training/train_stage2_distill_dual.sh
   ```

### 4.3 Stage 3: 추론 및 NAVSIM PDMS 채점
```bash
bash scripts/evaluation/eval_para_ssr.sh /path/to/stage2_checkpoint.ckpt
```

---

## 5. 논문 작성 시 핵심 기여점 (Target Contributions)

1. **최초의 이종 모달리티 다중 교사 증류 (Dual-Teacher Synergistic Distillation)**:
   LiDAR 기반 3DOD 교사와 위성/카메라 기반 HD-Map 교사를 하나의 통일된 주행 공간으로 합성 전이.
2. **주행 의도 기반 공간 집중 손실 (Planning-Aware Corridor Distillation)**:
   단순 Feature Matching의 고질적 한계인 공간 노이즈 희석 문제를 가우시안 주행 복도 마스크로 극복.
3. **추론 비용 0 (Zero Inference Overhead)**:
   학습 시에만 교사와 어댑터를 사용하고, 배포 시에는 순수 전방 카메라 3개와 경량 Dense BEV 플래너만으로 구동되어 완벽한 실시간성 확보.
