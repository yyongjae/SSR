# BEV Selector v7_2 잠재적 병목 및 향후 발전 과제
## Expected Issues & Architectural Roadmap toward 90+ PDMS

**기준 문서**: [`bev_selector_v7_2.md`](bev_selector_v7_2.md), [`bev_selector_v7.md`](bev_selector_v7.md)  
**참조 보고서**: [`report/22`](../report/22_planning_readout_results.md), [`report/23`](../report/23_v2_directions.md), [`report/24`](../report/24_hypothesis_ledger.md)  
**소속**: `/workspace/byounggun/SSR`  
**작성일**: 2026-10-12  
**대상 모델**: `paradrive_distill_bev_selector_v7_2` (ResNet-34 + Aux Heads + Task Interaction + v7 Distillation)  

---

## 1. 문제의식: "왜 v7_2 증류는 90점을 바로 돌파하기 어려운가?"

현재 실행 중인 `v7_2` 모델은 **NAVSIM 검증 최고 성능의 Student 아키텍처 (`km/para-ssr-v2`, 단독 PDMS ~0.88)**에 **최신 궤적 앵커 공간 편향 증류 (v7 KD, DAC 0.9508 달성)**를 결합한 현 시점 최강의 구조입니다.

그럼에도 불구하고 목표 점수를 90점이 아닌 **88.5 ~ 89.5+**로 보수적으로 산정하는 이유는, 단순한 특징 표현력 부족이 아니라 **NAVSIM 평가 체계의 수학적 특성과 플래너의 구조적 병목 4가지**가 가로막고 있기 때문입니다.

---

## 2. 핵심 예상 병목 4가지 (Structural Bottlenecks)

### 2.1 PDM Scorer의 곱셈 페널티와 "0점 탈락"의 잔혹성
NAVSIM 공식 PDM 점수 공식은 가혹한 곱셈 연산으로 정의됩니다:
$$\text{PDMS} = \frac{1}{N} \sum_{i=1}^N \Big[ \mathbf{NC}_i \times \mathbf{DAC}_i \times \mathbf{DDC}_i \times \frac{\mathbf{EP}_i + \mathbf{TTC}_i + \mathbf{Comfort}_i}{3} \Big]$$

* **12,146개 시나리오 중 1%만 실수해도 -1.0점 감점**:
  * 4초 주행 중 단 한 프레임이라도 타이어가 도로 경계(Curb)를 10cm 스치면 $\text{DAC}=0$이 되어 해당 시나리오 전체 점수는 **0.0000점**으로 폐기됩니다.
  * 현재 88점 모델도 약 600~700개 시나리오에서 0점을 기록 중입니다.
* **90점 돌파의 절대 조건**:
  * $\text{DAC} \ge 0.965$, $\text{NC} \ge 0.990$ 수준으로 올라서서 0점 탈락 시나리오를 300개 이하로 줄여야만 산술적으로 90.0에 도달할 수 있습니다.

---

### 2.2 "고르기(Selection)"와 "쓰기(Offset)"의 괴리 ([`report/24 H3`](../report/24_hypothesis_ledger.md))
연구팀의 가설 장부([`report/24`](../report/24_hypothesis_ledger.md)) 실증 결과:
* **256개 앵커의 이론적 상한 (Oracle Selection)**: **93.64점**
* **상위 3개 앵커 기준 상한**: **91.07점**

즉, 256개 앵커 세트 자체는 93점까지 달릴 수 있는 능력이 완벽히 갖추어져 있습니다. 문제는 플래너가 256개 중 **최종 1등을 고르는 Softmax 분류 단계에서 미세한 순위 뒤바뀜**이 발생한다는 점입니다.
* 1위 앵커와 2위 앵커의 예측 점수 차이가 $0.01$에 불과한데, 1위가 연석을 아슬아슬하게 스치는 앵커를 택해버리면 $1.0$점이 될 시나리오가 $0.0$점으로 곤두박질칩니다.
* **현재 증류의 한계**: BEV 피처의 품질을 높여줄 뿐, 최종 분류기 헤드의 미세한 Top-1 결정 오류를 100% 보장하지 못합니다.

---

### 2.3 "위험 앵커(Hard Negatives)"에 대한 안전 증류 부재
* **현재 v7_2 증류 가중치**:
  $$w_k = \text{Softmax}\left(-\frac{\text{dist}(\text{anchor}_k, \text{gt\_traj})}{\tau}\right), \quad \tau=1.0, \ \alpha_{\text{winner}}=2.0$$
* **문제점 (Safety-Blindness)**:
  * 정답 궤적 주변의 앵커는 집중적으로 배우지만, **살짝 연석을 넘어가거나 앞차와 부딪히는 위험 앵커는 $w_k \approx 0$으로 증류 손실에서 완전 배제**됩니다.
  * 따라서 학생 모델은 위험 궤적이 "왜 위험한지(앞차의 위치, 연석의 존재)"에 대해 Teacher의 깊이 있는 인지 표현을 전달받지 못합니다.
  * 결과적으로 테스트 시 위험 앵커의 예측 점수를 단호하게 $0$점으로 떨어뜨리지 못하고 오선택하는 원인이 됩니다.

---

### 2.4 "자기 인지 무시" 현상 ([`report/22 §13`](../report/22_planning_readout_results.md))
연구팀의 실증 분석 결과:
> *"DAC(도로 이탈) 실패의 80%에서 모델의 Map Head는 이미 그곳을 도로 밖(0.6m 이탈)으로 정확히 보고 있었다. **즉, 인지를 못 해서가 아니라 플래너가 인지 결과를 따르지 않고 무시해서 이탈한다.**"*

* **구조적 불일치**:
  * Student는 BEV뿐만 아니라 `det_hidden`, `map_point_hidden`이라는 인지 메모리를 `TaskInteractionLayer`로 참조합니다.
  * 하지만 현재 `v7_2`의 증류 손실은 **오직 Student의 Raw BEV에만 걸려 있으며**, 앵커 쿼리가 인지 메모리를 참조하는 `TaskInteraction` 어텐션 경로에는 증류 신호가 전혀 흐르지 않습니다.
  * 인지 메모리를 읽는 방법이 교정되지 않으므로, 플래너의 고질적인 "인지 무시 버릇"이 잔존하게 됩니다.

---

### 2.5 기하학적 L2 거리 의존성 vs 실제 PDM 시뮬레이션 지표
* 현재 가중치 $w_k$는 GT 궤적과의 단순 유클리드 좌표 거리에만 의존합니다.
* 그러나 실제 주행에서는:
  * GT와 0.3m 떨어져 있어도 급커브에서 바퀴가 연석을 스치면 탈락입니다.
   * 반대로 GT와 1.2m 떨어져 있어도 차선 중앙을 유지하며 완벽히 안전한 앵커가 존재합니다.
* 이미 사전 계산된 256개 앵커의 실제 시뮬레이션 결과([`pdm_score_256`](../data/planning_vb/pdm_score_256))가 증류 가중치에 연동되지 않고 있습니다.

---

### 2.6 증류 앵커(Distillation Probes)와 플래닝 앵커의 목적 불일치 (The Teacher-Probe Dilemma)
현재 `v7` 및 `v7_2`는 학생의 플래닝용 256개 K-Means 정적 앵커(`trajectory_anchors_256.npy`)를 증류 모듈의 쿼리로 그대로 재사용하고 있습니다. 그러나 **"플래닝을 위한 앵커"**와 **"증류를 위한 앵커"**는 본질적인 목적이 다릅니다:

1. **플래닝 앵커의 목적**: 자차가 실제로 추종할 수 있는 **"물리적으로 실행 가능한 소수의 후보 궤적"**이어야 함.
2. **증류 앵커의 목적**: Teacher의 BEV 공간(장애물, 도로 경계, 보행자 등)을 구석구석 찔러보고 학생에게 전수하는 **"공간 탐침 센서(Spatial Query Probes)"**여야 함.

* **현 구조의 심각한 맹점**:
  * **GT 국소성 편향(Locality Blindness)**: `tau=1.0, winner_boost=2.0`으로 인해 증류 가중치의 90% 이상이 오직 인간 정답 궤적 주변 2~3개 앵커에만 독점됩니다.
  * **장면 사각지대(Spatial Blind Spots)**: 자차가 직진 중일 때, 좌측에서 급격히 끼어드는 차량, 우측 횡단보도를 건너려는 보행자, 25m 전방의 도로 연석 등은 **GT 궤적 바깥에 위치**합니다. 
  * 결과적으로 정적 앵커 세트는 이 영역들을 쿼리하지 못하므로, **Teacher가 아무리 고해상도 LiDAR와 HD 지도로 주변 위험 요소를 완벽히 보고 있어도 학생은 그 특권 지식을 전혀 증류받지 못하는 치명적인 정보 누수**가 발생합니다.

---

## 3. v8 아키텍처 해결 방안 (Roadmap to 90+ PDMS)

이상의 병목들을 극복하고 90점 고지를 넘기 위해 향후 v8에서 적용해야 할 구체적인 개선안입니다:

```
[Student BEV & Task Memories]               [Teacher BEVFusion & ReSMap]
              │                                           │
  ┌───────────┴───────────┐                   ┌───────────┴───────────┐
  ▼                       ▼                   ▼                       ▼
[Safe Anchors (R_k ≈ 1)] [Risky Anchors (DAC=0)] [Safe Anchors] [Risky Anchors]
  │                       │                   │               │
  ├─ Imitation Feature KD ┤                   ├─ Teacher Attn ┤
  │  (Hybrid L2 + Cosine) │                   │  (KL Divergence)
  ▼                       ▼                   ▼               ▼
[Safety-Contrastive Penalty] ◄────────────────[Obstacle Focus Alignment]
("저 앞의 연석/차량 때문에 위험함"을 Teacher로부터 직접 학습)
```

### 방안 1: Safety-Contrastive Anchor Distillation (대조적 안전 증류)
* **목표**: 256개 앵커 중 실패하는 앵커들에 대한 분별력 부여.
* **메커니즘**:
  1. 사전 계산된 PDM 점수에서 $\text{DAC}=0$ 또는 $\text{NC}=0$인 앵커들을 **Hard Negative Anchors**로 지정.
  2. 위험 앵커에 대해서도 Teacher의 어텐션을 모방하게 하여, 학생이 "이 궤적이 왜 장애물/연석에 가로막히는지" 원인을 확실하게 주시하도록 강제.
  3. 위험 앵커의 예측 PDM 보상 로짓에 마진 기반 대조 손실(Contrastive Margin Loss) 부과.

### 방안 2: Task Interaction 메모리 증류 (Privileged Attention KD)
* **목표**: 플래너가 인지 결과를 무시하고 위험 궤적을 고르는 현상 제거.
* **메커니즘**:
  * 플래너 내부의 `TaskInteractionLayer`에서 앵커 쿼리가 `map_point_hidden`(도로 경계) 및 `det_hidden`(차량)을 바라보는 Cross-Attention 가중치를 Teacher/GT Hungarian 매칭 기반으로 직접 감독.

### 방안 3: PDM 시뮬레이션 보상 기반 가중치 연동
* **목표**: 단순 L2 거리 대신 실제 안전도 기반으로 증류 중요도 재편.
* **메커니즘**:
  $$w_k = \text{Softmax}\left( \frac{\text{Reward}_k}{\tau_{\text{reward}}} - \frac{\text{dist}_k}{\tau_{\text{geom}}} \right)$$
  * 실제 PDM 시뮬레이션 점수 $\text{Reward}_k = \text{NC}_k \times \text{DAC}_k \times \dots$가 만점인 안전 앵커에 높은 가중치를 부여.

### 방안 4: Test-Time Safety Critic / Refined Rescoring 도입
* **목표**: 분류기(Classifier)의 미세 순위 오차로 인한 0점 탈락 방어.
* **메커니즘**:
  * 추론 시 Top-1 앵커 하나만 맹신하지 않고, Top-3 후보를 추출한 뒤 가벼운 Safety Critic 모듈(또는 Refined 궤적 재채점)을 통해 가장 안전한 궤적을 최종 선택.

### 방안 5: 증류 전용 동적 앵커 (Distillation-Specific Adaptive Probes) 설계
* **목표**: 플래닝 앵커의 한계를 벗어나, Teacher BEV의 핵심 지식을 사각지대 없이 100% 흡수.
* **핵심 전략 (추론 비용 0 - 학습 전용 증류 프로브)**:
  1. **횡방향 섭동 프로브 (Lateral Perturbation Probes)**:
     - 인간 정답 궤적($\tau_{\text{gt}}$)을 좌우로 $\pm 0.3\text{m}, \pm 0.6\text{m}, \pm 1.0\text{m}$ 미세 오프셋시킨 섭동 궤적들을 증류 프로브로 추가 주입.
     - 학생에게 "차선 중앙에서 벗어날수록 Teacher의 연석/장애물 어텐션과 위험도 반응이 어떻게 변하는지(Spatial Safety Gradient)"를 전수하여 복원력 강화.
  2. **장면 적응형 위험 탐침 프로브 (Scene-Adaptive Hazard Rays)**:
     - 정적 256개 앵커에 얽매이지 않고, Teacher(BEVFusion)가 높은 신뢰도로 검출한 핵심 차량/보행자를 향하는 가상 궤적 레이(Ray)를 증류 쿼리에 포함.
     - 직진 상황에서도 주변의 끼어들기 차량이나 도로 경계선 정보를 온전히 학생 BEV로 전이.
  3. **속도 적응형 파라메트릭 궤적 (Velocity-Adaptive Clothoid/Bezier Probes)**:
     - 고속 주행 시에는 전방 30~50m까지 뻗는 고속 프로브를, 교차로 회전 시에는 회전 반경에 촘촘한 곡선 프로브를 동적으로 생성하여 탐침 해상도 극대화.

---

## 4. 진단 프로토콜 (v7_2 완료 시 체크포인트)

현재 구동 중인 `v7_2` (GPU 6, 7) 학습이 완료되면, 본 문서의 가설을 검증하기 위해 아래 항목을 순차적으로 확인합니다:

1. **DAC 준수율 확인**:
   * v7(0.9508) 대비 v7_2에서 **0.955+**를 달성했는가? (보조 태스크 헤드와의 시너지 확인)
2. **NC 충돌 방지 확인**:
   * v7(0.9826) 대비 Detection 헤드 탑재 효과로 **0.988+**로 반등했는가?
3. **EP 진행도 확인**:
   * Task Interaction 효과로 **0.810+**에 도달했는가?
4. **결과 분석 및 판정**:
   * 만약 **PDMS 88.7 ~ 89.4** 범위에서 정체된다면, 본 문서에서 지적한 **"위험 앵커 오선택(2.3)"**, **"인지 무시(2.4)"**, 그리고 **"증류 앵커의 사각지대(2.6)"**가 마지막 0.8점의 주원인임을 확정하고 위의 v8 아키텍처로 즉시 전환합니다.

