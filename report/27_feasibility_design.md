# 후보 집합 위의 미분 가능 feasibility — 설계서

2026-09-23. 실험만 돌리면 되는 수준을 목표로 쓴다. 모든 수치에 **출처**와 **직접 확인 여부**를 단다.
확인 못 한 것은 근거로 쓰지 않는다.

---

## 1. 한 줄 주장

> 예측된 주행가능 장(field)에서 **후보 앵커 각각의 footprint 코너**로 계산한 **하나의 미분 가능 비용**을,
> **선택 확률로 가중**해 **offset head · 점수 head · 장 예측기 세 곳에 동시에** 흘린다.

선행연구 19편 대조 결과, 이 중 문헌적으로 **완전히 비어 있는 좌표는 두 개**다.

| | 내용 | 19편 중 |
|---|---|---|
| **C4** | 후보별 비용 × **선택 확률 가중** | **0편** |
| **C5** | 하나의 비용이 **offset head와 점수 head로 동시에** | **0편** |

나머지 구성요소는 전부 선점되어 있으므로 단독으로 내세우지 않는다(§2).

---

## 2. 선행연구 좌표 — 무엇이 겹치고 무엇이 비었나

구성요소 정의:

- **C1** 주행가능 장을 **모델이 예측**(GT 조회가 아님)
- **C2** 앵커+offset 후보의 ego footprint **4코너 × 전 타임스텝**을 그 장에서 질의
- **C3** 비용을 후보 **K개 전부**에
- **C4** 후보별 비용에 **선택 확률 가중**
- **C5** 같은 비용이 **offset head와 점수 head 양쪽**으로
- **C6** 비용이 **장 예측기로 역전파**되어 인지를 학습
- **C7** **WTA**(GT 최근접 1개만 offset 감독)를 대체·보완

### 2.1 완전히 선점된 것 — 단독 주장 불가

| 요소 | 선점자 | 근거 |
|---|---|---|
| **C1** | DriveReferee, SafeDrive, FlowDrive, READ | DriveReferee: *"a convolutional encoder–decoder predicts a static drivable-area map M̂_drv"* |
| **C2 기하 프리미티브** | **SV-WAM** | *"we compute the four corner points c_ij of the vehicle footprint and sample their signed distances from the signed distance field"* (Eq.14). **문자 그대로 동일** |
| **C3** | DriveReferee, WoTE, DiffusionDriveV2 | DriveReferee: *"the referee evaluates all K plans on the same predicted map M̂"* |
| **C6 원리** | **VAD (2023)** | `plan_loss.py:31` `perception_detach=False` 기본값, 공식 config 3개에 오버라이드 **0건** (직접 확인) |
| **C7 문제의식** | DiffusionDriveV2 | WTA가 negative mode를 무감독으로 남긴다는 지적이 core 기여 |

**"field를 예측한다", "코너를 질의한다", "전 후보에 건다"는 전부 이미 있다.** 이걸 새롭다고 쓰면 즉시 반박당한다.

### 2.2 부분 겹침 — 결합에서만 살아남는 것

**C2 전체** (K개 고정 앵커 + per-anchor offset **구조 위에** 걸린 4코너 질의) — 개별 부품은 선점됐으나 이 결합은 0/19.
SV-WAM·CorDriver·DriveReferee는 앵커 구조가 없고, SafeDrive는 앵커(K=256)가 있으나 질의 결과가 **점수 경로 전용**이다:
> SafeDrive §3.5는 `L_plan`(궤적 refinement + imitation)과 `L_TwDAC`(안전 점수 추정)를 **명시적으로 분리**한다.

**C6 전체** — VAD가 **벡터 맵 · 단일 궤적 · waypoint 1점**에서 선점. "후보 집합 × footprint 코너 × 밀집 장" 조합은 없다.
따라서 "최초"가 아니라 **"VAD식 제약을 후보 집합·footprint·밀집 장으로 일반화"**로만 주장한다.

**C7 전체** — DiffusionDriveV2는 같은 문제를 **미분 불가능한 RL 보상**으로 풀었다.
우리는 **미분 가능한 기하 비용**으로 푼다. 수단의 차이로만 주장한다.

### 2.3 비어 있는 것

**C4 — 0/19.** 19편 전문에 `weighted by the.*probab`, `probability-weighted`, `score-weighted`, `expected cost`,
`softmax.*cost`, `per-candidate weight` 등 정규식 전수 검색 → 유의미 hit 0.
전 후보에 거는 사례들은 전부 **균등 가중**이다:
- DiffusionDriveV2 Eq.(7): `1/N_anchor · 1/G`
- Versatile Samples: *"We set every λ_j = 1"*
- SparseDriveV2 Eq.(29): 단순 합

혼동하기 쉬운 셋은 명시적으로 배제한다:
- SV-WAM의 log-mean-exp — 가중 대상이 **footprint 점**이지 후보가 아님
- WoTE의 softmax — reward head의 **타깃 라벨**이지 비용 가중치가 아님
- FeasibleFlow의 pair 가중 — **정확한 평가값 차이**로 쌍을 가중, 예측 선택확률이 아님

**C5 — 0/19.** 그리고 여러 논문이 **의도적으로 분리**해 두었다:
- WoTE §4.2: *"We then feed trajectory anchors into the trajectory evaluation module during training.
  During testing, the trajectory evaluation module evaluates the refined trajectory instead of the anchor trajectories."*
- PLUTO `pluto_trainer.py:223-238`: 기하 비용은 **WTA 승자 1개**에, **GT cost map**에서, 점수 head와 **무관하게**
  (`cls_loss`의 타깃은 `.detach()`)

---

## 3. 이 방향은 한 번 시도되어 기각되었다 — 그리고 그게 기회다

**CorDriver가 정확히 이 경로를 시험하고 버렸다.** (§V-D, Table III, 직접 확인)

> *"The gradient of ℒ_imi is selectively detached to isolate its effects on the weight matrices,
> the reference trajectory ξ̂ and corridor constraints Ĉ."*
> *"training only the weight matrices yields the best overall performance, while introducing learnable
> trajectory and corridor components leads to comparable or even degraded performance."*

공개 코드도 그 결론을 따른다:
```python
# projects/mmdet3d_plugin/VAD/VAD_head.py:933  (활성 경로)
qp_traj = self.qp(..., outputs_ego_trajs_mpc.detach(), outputs_ego_cors_mpc.detach(), train=False)
# utils/corridor_loss.py:70-71, 277-280
lane_preds = lane_preds.detach();  agent_preds = agent_preds.detach()   # 인지로 가는 경로 전부 차단
```

**그들이 진단한 실패 원인이 우리 설계가 직접 겨냥하는 것이다:**
> 기울기 진동이 크다 — *"their influence on optimization results varies significantly across different driving scenarios"*

**C3(후보 256개) + C4(확률 가중)가 바로 그 분산 문제의 해법이다.** 장면당 1개 궤적에서 오는 기울기는
장면마다 크게 흔들리지만, 256개 후보에 확률 가중으로 평균하면 같은 신호가 안정된다.

추가로 **CorDriver의 corridor는 타임스텝별 직사각형 5개 파라미터** `[c_x, c_y, θ, l, w]_t` 이고
기울기가 **QP의 KKT 시스템**을 통과한다. 직사각형을 회전·확대해 제약을 만족시키는 문제는 조건수가 나쁘고,
KKT 역전파는 불안정하기로 알려져 있다. **밀집 장에 대한 직접 hinge는 그 두 가지가 모두 없다.**

그리고 DriveReferee가 남긴 문장이 우리 개입 지점을 그대로 가리킨다:
> *"This suggests that the remaining deployment error mainly comes from **map prediction** rather than the geometric verdict."* (§IV-E)
> *"errors in the self-predicted maps can lead to wrong verdicts; **better scene prediction** ... may reduce these errors without changing the analytic rule."* (§V)

**즉 "왜 이전에는 실패했고 우리는 왜 되는가"를 실험으로 보이는 것이 기여의 실질이다.**

---

## 4. 설계 근거가 되는 우리 측정

전부 navtest 12,146 장면, NAVSIM 공식 채점기. 후보 단위 분석은 EC를 후보별로 정의할 수 없어
**EPDMS(EC 제외)** 척도이며 그 척도에서 배포가 88.81이다. 공식은 **PDMS 88.14 / EPDMS 87.80**.

| # | 측정 | 값 | 스크립트 | 신뢰도 |
|---|---|---|---|---|
| M1 | 하드 제약(진짜 DAC·NC=0 제거)의 값어치 | 88.81 → **93.26**, 같은 수 무작위 제거 통제군 **85.30** | `hard_constraint_ceiling.py` | 견고 (통제군 보유) |
| M2 | 제약 하에서는 모방만 > 배포 규칙 | 93.58 vs 93.26 | 동일 | 견고 |
| M3 | 예측 DAC로 필터 | 86.4개 제거해도 **−0.00** | 동일 | 견고 |
| M4 | DAC 실패 477 중 **offset이 깬** 비율 | **59.7%** (앵커는 도로 안이었음) | 인라인 | 견고 |
| M5 | 깬 경우 offset 크기 / 통과 장면 | **1.19 m** / 0.58 m | 동일 | 견고 |
| M6 | 침범 깊이 | 중앙값 **0.32 m**, 모서리 **1개가 93%** | `dac_depth.py` | 견고 (채점기 폴리곤 대조) |
| M7 | DAC·NC를 완벽히 고칠 때 EPDMS | +3.31 / +0.77 | 인라인 | 견고 |
| M8 | 앵커 라벨 감독의 천장 vs 출력 일치 라벨 | 90.31 vs **94.24** | `label_transfer_probe.py` | 견고 |
| M9 | im이 장면 정보를 쓴다 | +13.5 DAC | 인라인 | **약함 — 추정기 잡음, 재측정 필요** |

**M1·M4·M6이 설계를 결정한다:**
- 제약은 큰 레버다(M1) → 걸 가치가 있다
- 실패의 대부분을 **offset이 만든다**(M4) → 선택이 아니라 **생성**에 걸어야 한다
- 침범이 **0.32 m, 모서리 1개**(M6) → 마진은 **0.3~0.5 m**. 크게 잡으면 소심해져 EP를 잃는다

---

## 5. 방법

### 5.1 왜 토큰이 아니라 장인가

비용은 임의의 (x, y)에서 조회되어야 한다 — 후보 256 × 스텝 8 × 코너 4 = **8,192점**.
토큰 집합은 기하 질의가 불가능하다. 따라서 corridor는 **BEV 격자 위의 SDF** `[B, 50, 100]`, 0.64 m/칸이다.

### 5.2 무엇으로부터 만드는가 — C6의 실체

`bev_embed`만으로 디코딩하면 기울기가 BEV로만 간다. det/map head는 배우지 못한다.
**det/map의 hidden state에서 만들어야** 인지가 planning 필요로 학습된다.

```
det_hidden [B,100,C] · motion_hidden [B,100,6,C] · map_point_hidden [B,100,20,C]
   │  예측 위치로 BEV 격자에 scatter          ← 좌표가 처음으로 기하적으로 쓰임
   │    det: all_bbox_preds[-1][..., :2]
   │    map: all_map_pts_preds[-1]  (20점 전부)
   └─ concat(bev_embed, scattered) → conv 디코더 → SDF Φ [B, 50, 100]
```

**이것이 현 구조의 결함을 정면으로 고친다.** 지금 `prepare_task_memories`
(`planner_head.py:311-317`)는 위치를 attention **key에만** 넣고 value는 latent뿐이라 좌표가 planner로 흐르지 않는다.
scatter는 위치를 **격자 인덱스**로 쓰므로 좌표가 구조적으로 반영된다.

### 5.3 비용과 기울기 경로

```
c_k = Σ_t Σ_{corner} softplus( (margin − Φ(corner of (anchor_k + offset_k) at t)) / β ) · β
      k = 1..256,  Φ 는 grid_sample 로 미분 가능
```

```
c_k ──┬─► (i) offset head :  L_off = Σ_k w_k · c_k ,  w_k = softmax(선택점수)_k , detach   [C4, C5]
      ├─► (ii) 점수 head  :  DAC head 타깃을 exp(−c_k) 로 교체                              [C5]
      └─► (iii) Φ 자신    :  (i)(ii) 의 기울기가 디코더 → det/map hidden 으로 역류           [C6]
                            + GT seg 접지 손실(낮은 가중치, 붕괴 방지)
```

**(ii)가 라벨 문제를 부수적으로 해결한다.** 지금 DAC head는 **앵커의** PDM 점수를 타깃으로 배우는데
모델은 **앵커+offset**을 내보낸다 — M8의 3.93점 불일치의 원인이다.
`c_k`는 **refine된 궤적에서 계산되므로 타깃과 출력이 일치한다.**

### 5.4 기존 구조에서 바꾸거나 빼는 것

| 대상 | 조치 | 근거 |
|---|---|---|
| `sim_reward_heads`의 DAC 타깃 | **`exp(−c_k)`로 교체** | M3(예측 DAC 무용), M8(라벨 불일치) |
| `weighted_reward`의 `0.5·log(DAC)` | **제거 arm 포함** | M2(제약 하에선 모방만이 더 나음) |
| WTA offset 감독 | **유지 + 병행** | offset은 DAC에 순이득(+9.06). 죽이면 안 됨 |
| `plan_det_cross_attn` / `plan_map_cross_attn` | **유지, 절제 평가 후 결정** | 기여 여부 **미측정** |
| `plan_map_loss` (v1 경로) | 미사용 | 궤적 1개 × GT SDF. 새 손실이 대체 |

### 5.5 비용

SDF 디코더 ~0.5 M 파라미터 / `grid_sample` 8,192점 무시 가능 / GT seg 캐시 약 600 MB /
학습 시간 증가 5% 미만 예상.

---

## 6. 실험 설계

**A0**는 보유(공식 PDMS 88.14 / EPDMS 87.80). 나머지는 GPU 2장 × 25시간.

| arm | 장 | 대상 | 가중 | 경로 | 증명하는 것 |
|---|---|---|---|---|---|
| **A1** | **GT** | 배포 궤적 1개 | — | offset | **SV-WAM 재현**. 90.1 근처가 나와야 정상 |
| **A2** | GT | 후보 256 | **확률** | offset | **C3+C4의 값어치** = A2 − A1 |
| **A3** | GT | 후보 256 | 확률 | offset **+ 점수** | **C5의 값어치** = A3 − A2 |
| **A4** | **예측** | 후보 256 | 확률 | offset + 점수 + **장** | **C6의 값어치** = A4 − A3 |
| A2′ | GT | 후보 256 | **균등** | offset | C4가 가중 때문인지 확인 (통제군) |

**실패 조건을 미리 못 박는다:**
- **A2 ≈ A1** → C3+C4에 값이 없다. 기여의 절반이 사라진다
- **A3 ≈ A2** → C5에 값이 없다. 나머지 절반도 사라진다
- **A4 ≈ A3** → C6에 값이 없다. "GT 맵을 쓰면 되는데 왜 예측하나"에 답 못 함
- **DAC ↑ 인데 EP ↓** → 소심해진 것. margin을 줄여 재시도
- 위 넷 중 둘 이상이면 **이 방향을 접는다**

**모든 arm에서 공식 PDMS·EPDMS를 항목별로 낸다.** 예측 sub-score는 지표로 쓰지 않는다.

**학습 중 진단 로그(추가 실행 없이 얻음):**
- `c_k`(예측 장) **와** GT SDF로 계산한 같은 비용을 detach해서 병행 기록
  - 둘 다 ↓ → 제대로 배움
  - 예측만 ↓, GT는 그대로 → **모델이 장을 "전부 주행 가능"으로 붕괴시켜 비용 회피** (최빈 실패)
  - 둘 다 그대로 → 손실이 안 먹힘
- 항목별 EPDMS를 매 validation마다 — 특히 **EP**

**선행 필수(GPU 나오면 학습보다 먼저, 4시간):**
인지 경로 절제 평가 — `det_update=0` / `map_update=0` / 둘 다 / memory 셔플(통제군).
`plan_det_cross_attn`·`plan_map_cross_attn`을 뺄지 말지를 결정한다.

---

## 7. 예상 반박과 답변

**Q. SV-WAM이 이미 4코너 SDF hinge를 했다.**
A. 맞다. C2의 기하 프리미티브는 선점됐고 우리는 그걸 새롭다고 주장하지 않는다.
SV-WAM은 **GT 맵**·**궤적 1개**·앵커 구조 없음이다(논문 전체에 vocabulary/앵커/점수 head 없음, grep 확인).
우리 주장은 그 비용을 **후보 집합에 확률 가중으로 걸어 선택과 생성을 동시에 학습시키는 배선**이다.

**Q. CorDriver가 corridor를 이미 했고, 학습 가능한 corridor는 오히려 나빴다(Table III).**
A. 그것이 우리 출발점이다. 그들이 지목한 실패 원인이 *"장면마다 영향이 크게 달라 기울기 진동이 크다"*인데,
**C3(256 후보)+C4(확률 가중)가 정확히 그 분산을 줄인다.** 또한 그들의 corridor는 타임스텝별 직사각형 5파라미터이고
기울기가 **QP의 KKT**를 통과한다. 우리는 밀집 장에 대한 직접 hinge라 두 불안정 요인이 모두 없다.
이 대비 자체를 논문에서 실험으로 보인다(A4 vs 직사각형 파라미터화 arm을 추가할 수 있다).

**Q. VAD가 이미 예측 맵으로 역전파했다.**
A. 맞다. `perception_detach=False`가 기본값임을 코드로 확인했다.
우리는 "최초"를 주장하지 않고 **"벡터 맵·단일 궤적·waypoint 1점에서 → 밀집 장·후보 집합·footprint 코너로 일반화"**로 쓴다.

**Q. DriveReferee가 "규칙은 학습할 필요 없다"고 통제 실험으로 보였다(−0.01 EPDMS).**
A. 그들이 기각한 것은 **verdict를 학습으로 대체하는 것**이다. 우리는 verdict를 학습하지 않는다 —
**해석적 비용을 미분 가능하게 만들어 장 예측기와 플래너를 학습시킨다.** 그리고 그들 스스로
*"remaining deployment error mainly comes from map prediction"*이라고 썼다. 우리 개입 지점이 거기다.

**Q. 앵커를 늘리면 되지 않나(SparseDriveV2 262K).**
A. 우리 구조에서는 안 된다. 앵커 순위와 refine 후 순위의 장면 내 Spearman이 0.624이고
top-1 일치가 37.4%, 앵커 점수 상위 32개에 최선 후보가 41.5%만 들어온다(`rank_disagreement.py`).
coarse-to-fine이 성립하지 않는다. **그리고 앵커 증량은 C4를 하이퍼파라미터로 격하시키므로 피한다.**

**Q. 진단이 기여인가.**
A. 아니다. 기여는 §1의 배선이다. §4의 측정은 **설계 선택의 근거**로만 쓴다.

---

## 8. 미확인 · 위험

| 항목 | 내용 | 영향 |
|---|---|---|
| **SafeDrive `L_plan` 수식** | 논문에 미기재. Eq.4의 `M_Driv` 샘플링에 detach가 걸렸는지 불명 | **C6 주장의 유일한 실질 리스크.** 코드 공개 시 최우선 확인 |
| M9 (`im`의 장면 정보) | 추정기 잡음(B<A) | 재측정 필요. 현재 논거로 쓰지 않음 |
| 절제 평가 미실시 | `plan_det/map_cross_attn` 기여 불명 | 5.4의 "유지" 결정이 잠정 |
| DiffusionDriveV2 `R(τ)` | 본문에 정의 없음 | C7 대비 서술 시 조심 |
| GTRS BEV seg 클래스 구성 | drivable 포함 여부 미기재 | C1 선점 판정에 영향 없음(다른 셋으로 충분) |

---

## 9. 일정

| | 종료 | 자원 |
|---|---|---|
| GPU 0–3 (jaeha UniAD, 20ep 중 9ep) | 9/26~27 (12ep에서 끊으면 **9/24 오후 5시**) | — |
| GPU 4–5 (byounggun, 30ep 중 8ep) | **9/24 낮** | — |

**9/24 저녁 기준**: 절제 평가 4시간 → A1·A2·A3 동시(2장씩) 25시간 → A4·A2′.
