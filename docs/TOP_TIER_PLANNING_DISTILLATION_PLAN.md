# Dual-Teacher Planning-Aware BEV Distillation (DPD)
## 탑티어(CVPR / ECCV / NeurIPS) 지향 아키텍처 및 연구 계획서

**소속/작업공간**: `/home/external-user/byounggun/SSR`  
**작성일**: 2026-09-16  
**개정일**: 2026-09-18  
**연구 주제**: 동적 3D 객체(BEVFusion)와 정적 HD-Map(ReSMap) 교사의 지식을, GT 주행 복도(Corridor) 마스크로 걸러 카메라 학생에 전이하는 계획 중심 종단간 자율주행 프레임워크

이 문서는 NAVSIM PARA-SSR 구현(`aux_distill`)과 동일한 학습 계약을 적는다. 그림·수식·실행 기본값은 코드와 일치해야 한다.

---

## 1. 연구 배경 및 모티베이션 (Motivation)

### 1.1 기존 연구의 한계점

1. **균일한 공간 증류 (Uniform Spatial Distillation)의 비효율성**
   - 기존 BEV Feature Distillation은 전방 격자 전체(빈 허공, 먼 보도, 주행과 무관한 배경)에 대해 균일 \(L_2\)를 건다.
   - 현재 학생 격자는 \(100 \times 100\) (좌우 32 m, 전방 0–32 m). 셀이 1만 개이므로, 향후 4초 궤적 위의 장애물·차선 그래디언트가 배경에 희석된다.
   - \(50 \times 100\) 정사각 셀(0.64 m)은 후속 ablation으로 둔다. 첫 통제 실험은 원래 1만 토큰 플래너를 유지한다.

2. **단일 모달리티 교사의 정보 불균형**
   - 3D Detection 교사(BEVFusion)는 동적 차량/보행자의 3D 박스와 LiDAR 깊이는 강하지만, 차선 위상과 횡단보도 경계를 온전히 담지 못한다.
   - Online HD-Map 교사(ReSMap)는 정적 차선/도로 경계는 강하지만, 움직이는 객체의 3D 물리량(속도, 자세)을 주지 못한다.
   - 기존 파이프라인은 BEVFusion만 증류해 특권 정보의 절반만 썼다.

3. **플래너의 동역학 인지 부재 (Kinematics-Agnostic Planning)**
   - 기존 SSR/TokenLearner 경로는 내비게이션 명령만 쿼리에 더하고, BEV를 SE 게이트로 조절한다. 현재 속도/가속도 \((v_x, v_y, a_x, a_y)\)가 쿼리에 없다.
   - 명령은 “어디로 가라”이고, 동역학은 “지금 얼마나 빠르게/세게 움직이고 있는지”다. 정차 직후와 고속 직진은 같은 직진 명령이어도 4초 궤적이 달라야 한다. 속도 없이 장면만 보면 관성을 무시한 급격한 꺾임이 나온다.

4. **공유 BEV가 보조 헤드에 먹히는 문제**
   - 학생은 플래너 외에 검출/모션·벡터 맵 헤드를 같은 `bev_embed`에 붙인다. 밸브를 열지 않으면 공유 BEV 그래디언트는 대략 계획 0.25% / 검출 73.5% / 맵 26.2%다. 계획이 특징을 거의 못 민다.

---

## 2. 제안 아키텍처 핵심 구성요소 (Architecture Overview)

학습 때와 배포 때를 나눈다. 교사 캐시, 어댑터, 학생 보조 헤드는 **학습 전용**이다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     [ Teacher Cache Store, train-only ]                     │
│   1. BEVFusion  — 동적 3D 에이전트 교사 (캐시 BEV, 네트워크 미로드)          │
│   2. ReSMap     — 정적 Online HD-Map 교사 (sharded memmap)                  │
└──────────────────────┬───────────────────────────────┬──────────────────────┘
                       │                               │
                       ▼                               ▼
       ┌───────────────────────────────┐┌───────────────────────────────┐
       │ Stage-1 BEVFusion Adapter     ││ Stage-1 ReSMap Adapter        │
       │ PlanningBEVAdapter + Planner  ││ PlanningBEVAdapter + Planner  │
       │ 궤적 L1만, 센서 없음           ││ 궤적 L1만, 센서 없음           │
       └───────────────┬───────────────┘└───────────────┬───────────────┘
                       │  Stage 2에서 freeze            │
                       └───────────────┬───────────────┘
                                       │ 같은 어댑터 인스턴스: Adapter(S), Adapter(T)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│          [ Planning-Aware Corridor Mask, distill only, GT trajectory ]      │
│   τ_gt = {(x_t, y_t)}_{t=1}^{8}     (예측 궤적이 아님)                       │
│   W(x,y) = ε + (1-ε) · max_t exp(-||p - τ_t||² / 2σ_t²)                    │
│   L_distill = Σ W ||Adapter(S)-Adapter(T)||²  /  (Σ W · C)                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▲ 그래디언트는 얼린 어댑터를 지나 학생 인코더로
                                       │
┌──────────────────────────────────────┴──────────────────────────────────────┐
│              [ Student PARA-SSR, cameras + GT + distill ]                   │
│   1. Front 3 cameras (cam_f0, cam_l0, cam_r0) → ResNet50.tv_in1k + 1-level  │
│      FPN + BEVFormer 3 layers → dense BEV 100×100, C=256                    │
│   2. Shared bev_embed → planner / det+motion / vector-map                   │
│   3. Dense BEV planner (use_stl=false, TokenLearner 배제):                  │
│      PlanQuery ⊕ Command ⊕ MLP([vx, vy, ax, ay]) → CrossAttn(dense BEV)    │
│      → 8-step (x, y, heading), 0.5 s, 4 s horizon                           │
│   4. GradBalancer: 보조 헤드가 bev_embed로 넣는 그래디언트만 조절            │
└─────────────────────────────────────────────────────────────────────────────┘
```

배포 그래프는 `camera → BEVFormer → planner`만 남긴다 (`test_aux_heads=false`). 어댑터·교사 캐시·검출/맵 헤드는 추론에 없다.

---

## 3. 기하, 캐시, 어댑터 계약

### 3.1 학생 BEV ROI

SSR 축: \(x\) 오른쪽, \(y\) 전방. 전방 전용 ROI를 인코더·검출·맵이 공유한다.

| 항목 | 값 |
|---|---|
| `pc_range` / `map_pc_range` | `(-32, 0, -2, 32, 32, 2)` m |
| 격자 | `bev_h=100`, `bev_w=100` (10,000 토큰) |
| 셀 크기 | 가로 0.64 m × 세로 0.32 m |
| 카메라 | `cam_f0, cam_l0, cam_r0` (후방 `cam_b0` 없음) |
| 이미지 | 1920×1080 → ×0.4 → 상단 16 px 크롭 → 768×416 |
| 시간 큐 | `frame_indices=(2, 3)` |
| 검출 FOV | 전방축 기준 반각 80° |
| 궤적 | 8 step × 0.5 s = 4 s, `(x, y, heading)` |

`pc_range[1] < 0`(후방 반평면)은 거부한다. 카메라·격자·GT 모두 앞쪽만 본다.

### 3.2 교사 캐시

루트: `DISTILL_FEATURE_ROOT=/home/external-user/datasets/teacher_cache`.

| 교사 | 레이아웃 | 대표 격자 | 비고 |
|---|---|---|---|
| BEVFusion | `cache_{train,val}_{50x100\|100x100}/samples/<token>.npz` | 증류 정렬은 보통 50×100 | `bev_feature` CHW |
| ReSMap | `index.json` + sharded memmap | 50×100 | train_logs 토큰만. `val_logs`는 인덱스에 없음 |

좌표 변환은 매니페스트 계약 `student_bev = teacher_bev[:, :, ::-1]`이다. 교사 격자가 학생과 다르면 Stage 2가 학생 BEV를 교사 H×W로 bilinear 리샘플한 뒤 증류한다. Stage 1 플래너 격자는 학생 `bev_h/bev_w`가 아니라 `TeacherFeatureStore.spatial_size()`다.

학습 시작 전 `filter_datasets_to_teacher_cache`가 캐시에 있는 토큰만 남긴다. train이 비면 에러, val이 비면 validation을 건너뛴다. ReSMap이 train_logs만 있으면 val은 0이 되는 것이 현재 데이터 계약이다. 부분 rsync 캐시로 학습하지 않는다.

### 3.3 Frozen-adapter 계약

어댑터는 셀마다 같은 residual MLP (`PlanningBEVAdapter`)다.

```
LayerNorm → Linear(C→C) → GELU → Dropout → Linear(C→C) → +residual → LayerNorm
```

마지막 Linear는 0 초기화라 학습 0스텝에서 거의 항등 사상이다.

- Stage 1: 교사 BEV → 어댑터 → **그대로인** `ParaSSRPlannerHead` → 궤적 L1. 카메라/BEV 인코더 없음.
- Stage 2: **같은 어댑터 인스턴스**를 교사·학생에 쓰고 `requires_grad_(False)`, `eval()`. 어댑터가 손실을 숨기도록 회전하지 못한다. 그래디언트는 학생 인코더로만 들어간다.
- 추론: 어댑터 없음.

교사 네트워크 자체는 학습하지 않는다. 이름이 Det/Map Adapter인 것은 구 초안이다. 실제로는 교사별 **계획 공간 투영기**다.

---

## 4. 핵심 수식 및 알고리즘

### 4.1 Planning-Aware Driving Corridor Mask (\(W_{\text{corridor}}\))

마스크의 궤적은 **GT 미래 xy**다. 학습 초반의 틀린 예측으로 복도가 흔들리지 않게 한다. Heading은 쓰지 않는다. Stage 1은 이 마스크를 쓰지 않는다.

미래 \(T=8\) 웨이포인트 \(\mathbf{p}_t=(x_t,y_t)\)에 대해, 교사 격자 셀 중심 \((x,y)\)에서

$$
W(x,y)=\varepsilon+(1-\varepsilon)\cdot\max_{t\in\{1,\ldots,T\}}
\exp\left(-\frac{(x-x_t)^2+(y-y_t)^2}{2\sigma_t^2}\right)
$$

| 기호 | 코드 기본값 | 의미 |
|---|---|---|
| \(\sigma_t\) | \(2.5 + 0.1\cdot t\) m | 미래로 갈수록 복도를 넓힘 |
| \(\varepsilon\) | \(0.1\) | 복도 밖에도 약한 정규화 그래디언트 |
| max over \(t\) | `w.max(dim=1)` | 겹치는 웨이포인트가 가중치를 폭주시키지 않음 |

셀 중심은 `pc_range`와 `(bev_h, bev_w)`에서 잡는다. 두 교사가 격자가 달라도 같은 GT 궤적으로 각자 H×W 마스크를 만든다. 마스크는 **증류 MSE에만** 곱한다. 궤적 L1, 검출, 맵 손실에는 안 곱는다.

### 4.2 Dual-Teacher + 학생 과제 손실

Stage 2 총손실은 계획서 초안의 \(\mathcal{L}_{\text{plan}}+\mathcal{L}_{\text{distill}}\)가 아니다. GT 과제 손실과 교사 BEV 증류를 더한다.

$$
\begin{aligned}
\mathcal{L}
&= 2.0\,\mathcal{L}_{\text{plan}}
 + 1.0\,\mathcal{L}_{\text{det}}
 + 1.0\,\mathcal{L}_{\text{motion}}
 + 1.0\,\mathcal{L}_{\text{map}} \\
&\quad + \lambda_{\text{kd}}
   \big(\lambda_{\text{bev}}\,\mathcal{L}_{\text{distill}}^{\text{bevfusion}}(W)
      + \lambda_{\text{map}}\,\mathcal{L}_{\text{distill}}^{\text{resmap}}(W)\big)
\end{aligned}
$$

기본값: \(\lambda_{\text{kd}}=\texttt{distill\_loss\_weight}=1.0\), 가지별 `loss_weight=1.0`.

가지별 증류 (같은 얼린 어댑터, 학생을 교사 격자에 맞춘 뒤):

$$
\mathcal{L}_{\text{distill}}^{k}(W)
=\frac{\sum_{x,y} W(x,y)\,
\big\|\mathrm{Adapter}_k(S)_{x,y}-\mathrm{Adapter}_k(T_k)_{x,y}\big\|^2}
{\big(\sum_{x,y} W(x,y)\big)\cdot C}
$$

분모의 \(C=256\)은 채널 평균이다. 복도가 좁아져도 스케일이 폭주하지 않는다.

역할 분담:

| 항 | 감독 | 역할 |
|---|---|---|
| \(\mathcal{L}_{\text{plan}}\) | GT 궤적, 명령 모드만 L1, heading wrap × 0.5 | 최종 출력 |
| \(\mathcal{L}_{\text{det}}, \mathcal{L}_{\text{motion}}\) | NAVSIM 에이전트 GT | 동적 객체를 학생 BEV에 심음. BEVFusion 증류와 같은 방향 |
| \(\mathcal{L}_{\text{map}}\) | 벡터 맵 GT | 정적 구조를 심음. ReSMap 증류와 같은 방향 |
| \(\mathcal{L}_{\text{distill}}\) | 캐시 교사 BEV | 복도에서만 특권 표현을 맞춤 |

검출/맵 헤드는 **학생 보조 헤드**이지 교사 네트워크가 아니다. 증류 신호는 캐시 BEV뿐이다.

\(\mathcal{L}_{\text{plan/det/motion/map}}\)는 `ParaSSRLoss` 안에서 `task_loss_weight`를 곱한다. 증류는 `ParaSSRAgent.compute_loss`가 그 바깥에서 더한다. 그래서 GradBalancer는 증류 vs 계획을 맞추지 않는다.

Stage 1 손실은 궤적 L1만이다.

### 4.3 Kinematics-Conditioned Dense BEV Planner

TokenLearner(`use_stl`)를 끈다. 단일 PlanQuery가 dense BEV 1만 셀에 cross-attention한다. 이 쿼리에 **명령**과 **현재 동역학**을 같이 넣는다.

#### 왜 명령만으로는 부족한가

내비게이션 명령은 4-way one-hot (`left / straight / right / unknown`)이다. 같은 “직진”이어도

- 정차 상태 (\(v\approx 0\))에서 4초 궤적은 거의 제자리여야 하고,
- 고속 (\(v_x\) 큼)에서는 전방으로 길게 나가야 하며,
- 감속 중 (\(a\)가 진행 방향과 반대)이면 웨이포인트 간격이 줄어들어야 한다.

명령만 있으면 플래너가 이 세 경우를 BEV 외형으로만 구분해야 한다. BEV는 주변이지 자차 관성 상태가 아니다. 그래서 현재 프레임의 속도/가속도를 쿼리에 명시적으로 더한다. TransFuser/WoTE의 `status_feature`와 같은 NAVSIM 신호를, dense BEV 쿼리 조건으로 쓴 것이다.

이 조건은 **배포 그래프에도 남는다**. 교사·어댑터·보조 헤드와 달리 추론 비용 0 대상이 아니다. 스칼라 4개를 MLP에 넣는 비용이다.

#### 입력: 현재 프레임 EgoStatus

학생 피처 빌더(`para_ssr_features`)가 현재 프레임(`ego_statuses[-1]`)에서 8차원 `status_feature`를 만든다.

$$
\texttt{status\_feature}
=\big[\underbrace{c_{\text{left}},c_{\text{straight}},c_{\text{right}},c_{\text{unk}}}_{4\text{-way command}}
,\; v_x,\, v_y,\, a_x,\, a_y\big]\in\mathbb{R}^{8}
$$

\((v_x,v_y)\)와 \((a_x,a_y)\)는 NAVSIM `EgoStatus`의 `ego_velocity`, `ego_acceleration`이다. 로그의 `ego_dynamic_state`에서 오며, TransFuser/WoTE와 동일한 4슬롯이다. 플래너는 명령 4차원을 빼고 동역학만 받는다.

$$
\mathbf{s}
=\texttt{status\_feature}[:,\; \texttt{num\_navi\_cmd}:]
=[v_x,v_y,a_x,a_y]\in\mathbb{R}^{4}
$$

코드: `para_ssr_model.py`가 `features["status_feature"][:, cfg.num_navi_cmd:]`를 `ego_status`로 넘긴다. `status_feature`가 없거나 길이가 4 이하면 `features.get("ego_status")`로 떨어지고, 그것도 없으면 플래너가 \(\mathbf{0}_4\)를 넣는다.

#### 주입 위치: 명령 융합 뒤, BEV cross-attention 앞

dense 경로 (`use_stl=false`)만 `ego_status_encoder`를 갖는다. TokenLearner 경로는 명령으로 BEV를 SE 게이트할 뿐, 이 MLP가 없다.

$$
\begin{aligned}
\mathbf{q}_{\text{cmd}}
&= \mathrm{Fuser}\big(\mathrm{PlanQuery},\; \mathrm{Emb}(\mathrm{Command})\big) \\
\mathbf{q}_0
&= \mathbf{q}_{\text{cmd}} + \mathrm{MLP}_{\text{kin}}(\mathbf{s}) \\
\mathbf{q}_{\text{out}}
&= \mathrm{CrossAttn}_{L=3}\big(\mathbf{q}_0,\; \mathrm{Key}=\mathrm{BEV}_{\text{dense}},\; \mathrm{Value}=\mathrm{BEV}_{\text{dense}}\big) \\
\tau_{\text{pred}}
&= \mathrm{MLP}_{\text{traj}}(\mathbf{q}_{\text{out}})\in\mathbb{R}^{8\times 3}
\end{aligned}
$$

- `Fuser`: `concat(PlanQuery, navi_emb) → Linear(2C→C) → LayerNorm → ReLU`
- `MLP_kin` (`ego_status_encoder`): `Linear(4→C) → ReLU → Linear(C→C)`, \(C=256\)
- 동역학은 concat이 아니라 **residual add**다. 명령으로 만든 쿼리 방향을 유지한 채, 관성 상태를 같은 공간에 더한다.
- 그 다음 쿼리가 1만 BEV 셀을 본다. 속도가 크면 먼 셀, 정차면 근거리 셀을 보게 만드는 것이 이 순서의 이유다.
- 궤적 헤드는 8 step × `(x, y, heading)`을 한 번에 회귀한다. 동역학 적분기를 따로 두지 않고, 쿼리 조건으로만 물리 상태를 넣는다.

기존 SSR TokenLearner와 대비:

| | TokenLearner (`use_stl=true`) | Dense + kinematics (현재) |
|---|---|---|
| 장면 토큰 | 16개, TokenLearner가 BEV에서 선택 | BEV 10,000셀 전부 |
| 명령 | `navi_embedding` + SE로 **BEV 채널 게이트** | PlanQuery와 concat 후 Fuser |
| 동역학 \((v,a)\) | 없음 | 쿼리에 MLP residual |
| 배포 | 명령 + 선택된 토큰 | 명령 + \((v,a)\) + dense BEV |

#### `ego_motion` / `bev_shift`와는 다른 신호

둘 다 자차 움직임이지만 **소비자가 다르다**. 섞어 쓰지 않는다.

| 텐서 | 모양 | 누가 읽나 | 하는 일 |
|---|---|---|---|
| `status_feature` → \(\mathbf{s}\) | `[B, 4]` 현재 프레임 | **플래너 쿼리** `ego_status_encoder` | 궤적이 현재 관성에서 시작하게 함 |
| `ego_motion` | `[B, T, 18]` 큐의 매 스텝 | **BEVFormer** `ego_motion_mlp` | BEV 쿼리에 더해 시간 특징을 조건 |
| `bev_shift` | `[B, T, 2]` | **BEVFormer** `use_shift` | 이전 BEV를 현재 격자로 이동 |

`ego_motion` 18차원은 플래너 쿼리가 아니라 인코더용이다. 스텝 \(k\)에서 대략

\[
[d_{\text{fwd}}, d_{\text{left}}, d_{\text{yaw}}, \sin d_{\text{yaw}}, \cos d_{\text{yaw}}, v_x, v_y, a_x, a_y, \|(d_{\text{fwd}},d_{\text{left}})\|, \text{command}_{1:4}, \ldots]
\]

`bev_shift`는 연속 프레임 사이 자차 변위를 셀 단위로 나눈 값이다. 히스토리 BEV를 현재 시점으로 정렬한다. 궤적 헤드의 \((v,a)\) 조건이 아니다.

#### Stage 1에서는 일부러 끈다

Stage 1 피처 빌더(`TeacherAdapterFeatureBuilder`)는 **명령만** 만든다. `status_feature`/`ego_status`가 없어 플래너는 \(\mathbf{s}=\mathbf{0}\)을 넣는다.

이유: Stage 1의 목적은 어댑터가 교사 BEV에서 **계획에 필요한 장면 정보**를 통과시키게 하는 것이다. 여기서 \((v,a)\)를 주면 플래너가 교사 BEV를 거의 안 보고, 속도 적분만으로 무난한 궤적을 낼 수 있다. 그러면 어댑터는 계획 관련 특징을 안 실어도 궤적 손실이 줄어든다. freeze-adapter 계약이 깨진다.

Stage 2와 추론의 카메라 학생만 현재 프레임 \((v_x,v_y,a_x,a_y)\)를 쿼리에 더한다. 어댑터는 이미 얼었으므로, 동역학은 학생 플래너가 장면을 어떻게 읽을지만 바꾼다.

### 4.4 GradBalancer (공유 BEV 밸브)

보조 경로가 `bev_embed`로 넣는 그래디언트만 `_ScaleGrad`로 줄인다. 헤드 파라미터 그래디언트는 그대로다.

| 항목 | 값 |
|---|---|
| 목표 점유율 | plan 40% / det 30% / map 30% |
| 계획 경로 | \(s_{\text{plan}}=1\) (기준, 스케일 안 함) |
| 검출+모션 | 밸브 하나 \(s_{\text{det}}\). 측정 전 손실을 합산 |
| 맵 | \(s_{\text{map}}\) |
| clamp | \((10^{-5}, 1)\) — **줄이기만 하고 키우지 않음** |
| 측정 | 200 마이크로배치마다 \(\|dL/d\,\texttt{bev\_embed}\|\) |
| warmup | 10600 마이크로배치 (2 GPU × batch 4, 1 epoch), 그동안 \(s=1\) |
| EMA | 0.9 |
| 증류 | 이 루프 밖 |

점유율은 BEV 활성화 그래디언트 노름 비율이다. Adam step 비율이 아니다. DDP는 rank별 노름의 평균을 all-reduce한다. 상태(`iteration`, scales)는 체크포인트에 저장한다.

복도 마스크와 밸런서는 축이 다르다. 마스크는 **공간의 어디**에 증류할지, 밸런서는 **어느 학생 헤드**가 공유 BEV를 밀지다.

---

## 5. 2단계 학습 절차

### Stage 1A / 1B — 교사별 어댑터 (20 epoch)

교사마다 따로 학습한다. 한 어댑터가 동적 객체와 맵을 동시에 통과시키면 한쪽이 다른 쪽을 지울 수 있다.

1. 캐시에서 교사 BEV를 읽는다. 센서 없음.
2. `PlanningBEVAdapter` + `ParaSSRPlannerHead`만 학습한다 (`lr=2e-4`).
3. 손실은 명령 모드 궤적 L1만. det/map 헤드는 끈다 (`use_det_motion_head=false`).
4. 동역학 입력은 넣지 않는다. \((v,a)\)가 있으면 플래너가 교사 BEV를 우회할 수 있다.
5. `batch=16`, `accumulate=4` → 글로벌 배치 128 (2 GPU).
6. 완료 조건은 `epoch=(max_epochs-1)-*.ckpt`다. `last.ckpt`만으로는 끝난 단계로 치지 않는다.

산출: Stage 2가 freeze할 어댑터 체크포인트.

### Stage 2 — 카메라 학생 + dual distill (30 epoch)

1. Stage 1A/1B 어댑터를 로드하고 freeze.
2. 전방 3 카메라로 학생 BEV를 만든다 (`lr=1e-4`, backbone 0.1×).
3. 현재 프레임 \((v_x,v_y,a_x,a_y)\)를 dense 플래너 쿼리에 더한다. 이 조건은 배포에도 남는다.
4. GT로 plan / det / motion / map을 학습하고, 같은 얼린 어댑터로 두 교사 BEV에 복도 가중 MSE를 건다.
5. GradBalancer가 공유 BEV 점유율을 맞춘다.
6. `batch=4`, `accumulate=16` → 글로벌 배치 128. warmup 3 epoch, AdamW, grad clip 35.

### 추론 / 평가

Stage-2 Lightning 파일에는 `agent._distill.*` 키가 있다. PDM 평가는 `agent=para_ssr_agent` (`use_distill=false`)로 strict load하므로, `scripts/evaluation/export_student_ckpt.py`로 그 키를 벗겨 학생 전용 체크포인트를 만든다.

---

## 6. 파이프라인 및 실행 가이드

### 6.1 올인원 (권장)

Stage 1A → 1B → 2를 한 명령으로 돌리고, 끝난 어댑터를 Stage 2에 연결한다.

- **기본 GPU**: **4, 5번** (`CUDA_VISIBLE_DEVICES=4,5`, RTX 5090 32GB × 2). 0–3은 다른 작업이 쓰는 경우가 많다.
- 스크립트: [`scripts/training/run_all_stages_distill.sh`](../scripts/training/run_all_stages_distill.sh)
- Python: `/home/external-user/miniconda3/envs/ssr/bin/python` (있으면)
- 실험 접두어: `EXP_PREFIX=paradrive_distill`
- 산출: `work_dirs/paradrive_distill_stage1_bevfusion`, `_stage1_resmap`, `_stage2_dual_distill`

```bash
nohup ./scripts/training/run_all_stages_distill.sh > run_distill.log 2>&1 &
tail -f run_distill.log
```

올인원 스크립트는 `.env`의 `WANDB_API_KEY`로 개인 엔티티 W&B를 켠다. 단계별 단독 스크립트는 `WANDB=0`이면 TensorBoard만 쓴다.

#### 자동화 흐름

1. **Stage 1A (BEVFusion, 20 epoch)**  
   캐시: `/home/external-user/datasets/teacher_cache/bevfusion/cache_{train,val}_*`  
   센서 없이 어댑터+플래너. 끝나면 `BEVFUSION_CKPT` 설정.

2. **Stage 1B (ReSMap, 20 epoch)**  
   캐시: `/home/external-user/datasets/teacher_cache/resmap/index.json`  
   동일 레시피. 끝나면 `RESMAP_CKPT` 설정.

3. **Stage 2 (dual distill, 30 epoch)**  
   두 어댑터 freeze. `use_corridor_mask=true`, `use_stl=false`, `plan_num_layers=3`.  
   `checkpoint.save_top_k=1`이라 live `last.ckpt`는 epoch마다 덮인다. 평가용은 스냅샷을 따로 복사한다.

끝난 단계는 `epoch=(E-1)-*.ckpt`가 있으면 건너뛴다. 다시 돌리려면 `FORCE_RETRAIN=1` 또는 `FORCE_RETRAIN=stage1b`.

```bash
# GPU 변경
CUDA_VISIBLE_DEVICES=0,1 ./scripts/training/run_all_stages_distill.sh

# Stage 2만 (어댑터 체크포인트 지정)
BEVFUSION_ADAPTER_CKPT=/path/to/stage1_bevfusion.ckpt \
RESMAP_ADAPTER_CKPT=/path/to/stage1_resmap.ckpt \
ONLY_STAGE=stage2 ./scripts/training/run_all_stages_distill.sh

# 어댑터만
ONLY_STAGE=stage1a ./scripts/training/run_all_stages_distill.sh
ONLY_STAGE=stage1b ./scripts/training/run_all_stages_distill.sh

# 끝난 단계 재학습
FORCE_RETRAIN=stage2 ONLY_STAGE=stage2 ./scripts/training/run_all_stages_distill.sh
```

### 6.2 단계별 단독 스크립트

```bash
bash scripts/training/train_stage1_adapter_bevfusion.sh
bash scripts/training/train_stage1_adapter_resmap.sh

BEVFUSION_ADAPTER_CKPT=/path/to/stage1_bevfusion.ckpt \
RESMAP_ADAPTER_CKPT=/path/to/stage1_resmap.ckpt \
bash scripts/training/train_stage2_distill_dual.sh
```

단독 Stage 2 기본 실험명은 `stage2_dual_distill_corridor`이고, 올인원은 `paradrive_distill_stage2_dual_distill`이다.

### 6.3 평가 (NAVSIM PDMS)

navtest 로그/블롭은 `/home/external-user/navsim/download/` 아래 공식 트리다. 이 레포 `data/dataset/navsim_logs`는 trainval 심링크다.

```bash
# 증류 키를 벗긴 학생 체크포인트로 평가
python scripts/evaluation/export_student_ckpt.py \
  /path/to/stage2.ckpt /path/to/last_student.ckpt
bash scripts/evaluation/eval_para_ssr.sh /path/to/last_student.ckpt

# live last.ckpt를 얼린 뒤 GPU 1에서 PDM (기본: epoch-18 스냅샷 경로)
bash scripts/evaluation/eval_para_ssr_distill_snapshot.sh
SMOKE=1 bash scripts/evaluation/eval_para_ssr_distill_snapshot.sh
```

Hydra 평가 에이전트는 `para_ssr_agent`, `use_stl=false`, `plan_num_layers=3`, `scene_filter=navtest`, `split=test`다.

---

## 7. 논문 작성 시 핵심 기여점 (Target Contributions)

1. **이종 모달리티 이중 교사 증류 (Dual-Teacher Synergistic Distillation)**  
   LiDAR 3DOD 교사와 카메라/맵 HD-Map 교사를, 궤적 손실로 검증된 얼린 어댑터 공간에서 하나의 주행 BEV로 합쳐 카메라 학생에 옮긴다.

2. **계획 의도 기반 공간 집중 손실 (Planning-Aware Corridor Distillation)**  
   GT 4초 궤적 위의 가우시안 복도로 증류 MSE를 가중해, 균일 feature matching의 배경 희석을 줄인다.

3. **추론 비용 0 (Zero Inference Overhead)**  
   교사·어댑터·보조 헤드는 학습 그래프에만 있다. 배포는 전방 카메라 3장과, 현재 속도/가속도를 쿼리에 더하는 dense BEV 플래너만 쓴다. 동역학 MLP는 스칼라 4개라 실시간 비용이 사실상 없다.

4. **공유 BEV 과제 균형 (Planning-Preserving Auxiliary Supervision)**  
   학생 det/map GT는 표현을 풍부하게 하되, GradBalancer가 공유 BEV에서 계획 점유율을 지켜 보조 헤드가 플래너를 밀어내지 못하게 한다.

---

## 8. 구현 대조표 (초안 → 현재)

계획서 초안과 코드가 달랐던 지점. 현재 문서·코드는 오른쪽을 따른다.

| 항목 | 구 초안 | 현재 구현 |
|---|---|---|
| 학생 BEV | 50×100 | **100×100**, 50×100은 ablation |
| 복도 궤적 | \(\tau_{\text{ego}}\) (모호) | **GT xy**, 예측 아님 |
| \(\sigma_t\) | \(\sigma_0+\alpha t\) (미기입) | **2.5 + 0.1 t** m |
| 증류 분모 | \(\sum W\) | \(\sum W \cdot C\) |
| Stage 2 손실 | plan + distill | plan + det + motion + map **+** distill |
| 어댑터 이름 | Frozen Det/Map Adapter | `PlanningBEVAdapter` (교사별) |
| 동역학 | 전 단계에 있는 것처럼 그림 | **플래너 쿼리**에 \((v,a)\) residual. Stage 1은 0 (어댑터 우회 방지). `ego_motion`은 BEV 시간 정렬용으로 별개 |
| GradBalancer | 없음 | plan 0.4 / det 0.3 / map 0.3 |
| 기본 GPU | 2, 3 | **4, 5** |
| 평가 | `eval_para_ssr.sh`에 raw ckpt | distill 키를 벗긴 학생 ckpt |
| 완료 판정 | `last.ckpt` | `epoch=(E-1)-*.ckpt` |
