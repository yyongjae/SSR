# Refiner 초안 생성과 출력 파라미터화: 사실 조사와 설계 선택지 (a–e)

## 한눈에

- **사람 경로 길이:** navtrain/navtest의 모든 token에 사람 경로가 5초까지 있습니다. 6초까지는 raw log에서 약 98% 구할 수 있습니다.
- **균일 시간 재표본화는 초안으로 부적합합니다.** 모든 교란이 t=0 속도 불연속을 만듭니다. ×1.4는 31% token에서 속도가 2.4 m/s 넘게 튑니다. 실제 student 초안은 사람만큼 연속적입니다.
- **실제 오차는 작습니다.** student 초안과 사람의 차이는 가속도 오프셋 환산 ±0.4 m/s²(p5–p95), 4초 횡편차 ±1 m 수준입니다. ×0.6은 전체 0.5%로 실제 분포 밖이고, ×1.4는 NC 실패에서 p75–p90 구간(전체 2.7%)입니다.
- **NC 실패의 속도 방향:** NC 실패에서 student는 대부분 사람보다 빠릅니다. 4초 거리 비 중앙값 1.16, 4초 위치 +2.1 m입니다. 감속만으로 방향이 맞는 경우가 다수입니다.
- **가속을 허용하면 경로 연장이 필수입니다.** 초안보다 조금이라도 빠르게 하려면 초안 경로 끝(4초) 너머의 경로가 필요합니다. 추론 때는 사람 경로가 없으므로 외삽밖에 없습니다. 등곡률 외삽은 +1초까지 p90 0.28 m 오차로 쓸 만합니다.
- **EP 이득은 작습니다.** EP는 PDM-Closed 대비로 정규화됩니다. 사람 궤적 자체의 EP 평균이 0.87이고, 가속으로 EP를 올릴 여지는 제한적입니다.
- **디코딩:** 초안 폴리라인 위에서 시간을 다시 매겨도 모서리 오차는 p99 0.1 m 미만입니다. heading 규약은 기준선(baseline)과 반드시 같이 정해야 합니다(아래 b-5, e-4).

---

## (a) GT(사람) 궤적의 범위와 t=0 연속성

1. **5초까지는 항상 있습니다.** navtrain/navtest 설정은 `num_history_frames: 4`, `num_future_frames: 10`, `frame_interval: 1`, `has_route: true`입니다(`navsim/planning/script/config/common/scene_filter/navtrain.yaml:1-7`, navtest 동일). `filter_scenes`가 14프레임이 안 되는 창을 버리므로(`navsim/common/dataloader.py:43-44`) 모든 token에 미래 10프레임(5초)이 있습니다.
2. **5초 너머는 raw log에만 있습니다.** Scene 객체에는 14프레임만 들어 있습니다. 5초 너머가 필요하면 둘 중 하나를 써야 합니다.
   - log pkl을 직접 읽는다.
   - `num_future_frames`를 늘린 SceneFilter를 쓴다. 이 경우 log 끝 근처 token 약 1–3%가 빠집니다(아래 c 표).
3. **원점 규약:** `get_future_trajectory`는 현재 프레임(`num_history_frames-1`)을 원점으로 둡니다. 반환하는 것은 t=0.5..5.0 상대 pose이고 t=0 pose는 포함하지 않습니다(`navsim/common/dataclasses.py:297-321`, 특히 305, 312).
   - 지역 좌표계는 x 전방, y 좌측, heading 0입니다.
   - 채점기는 `metric_cache.ego_state.rear_axle`을 원점으로 씁니다.
   - navtest 무작위 40 token에서 metric cache의 rear_axle이 scene ego pose와 정확히 같았습니다(위치·heading 차 0.0). vx 차는 5e-7 m/s입니다. 즉 원점은 rear axle, heading 0으로 일치합니다.
4. **학습 target은 8 pose입니다.** PARA-SSR은 `get_future_trajectory(num_trajectory_frames=8)`만 씁니다(`navsim/agents/para_ssr/para_ssr_targets.py:262`). `Trajectory` 클래스도 8 pose(4초, 0.5초 간격)를 assert합니다(`dataclasses.py:248-262`).
5. **이상치: 프레임 누락 창**
   - navtrain 표본의 1.34%(282/21,065 token, 250개 중 30개 log)는 5초 창 안에 1.0초 간격이 있습니다. 프레임이 빠져 "0.5초 pose"의 시각이 틀립니다. navtest는 0.37%입니다.
   - 시간 기반 교란과 속도 target에서는 이 token에 flag를 달거나 빼는 것이 안전합니다.

## (b) 공식 채점기가 궤적을 소비하는 방식

1. **8 pose → 0.1초 참조 궤적**
   - `transform_trajectory`는 8 pose를 rear_axle 기준 절대좌표로 바꿉니다. 앞에 실제 `initial_ego_state`(실제 속도·가속도·조향)를 붙여 `InterpolatedTrajectory`를 만듭니다. 8 pose의 속도와 가속도는 0으로 들어가고 무시됩니다(`navsim/evaluate/pdm_score.py:41-73`).
   - 이후 0..4.0초를 0.1초 간격 41개 상태로 샘플합니다. 4초 끝에서 clip되므로 4초 너머 pose는 채점에 쓰이지 않습니다(`pdm_score.py:76-101`, `config/pdm_scoring/default_scoring_parameters.yaml:1-5`).
   - 보간 방식: x, y는 시간에 대해 선형(scipy `interp1d`)입니다. heading은 `np.unwrap` 후 선형입니다(nuplan `interpolated_trajectory.py:47-48`, `common/geometry/compute.py:159-161`).
2. **LQR + 자전거 모델 추종**
   - `PDMSimulator`가 41단계를 추종합니다(`pdm_simulator.py:39-90`).
   - 참조 속도와 곡률은 pose에서 최소제곱으로 추정합니다.
     - 속도: xy 변위를 **제출된 heading**의 (cos, sin)에 투영합니다(`batch_lqr_utils.py:81-156`, 특히 113-114, 252).
     - 곡률: **heading 변위**로 추정합니다(`batch_lqr_utils.py:158-220, 267`).
   - 종방향 제어는 1초 앞(10단계)의 참조 속도를 추종합니다. 횡방향 Q는 diag(lateral 1, heading 10, steer 0)입니다. 참조 속도와 현재 속도가 모두 0.2 m/s 이하이면 P 정지 제어로 바뀝니다(`batch_lqr.py:72-85, 191-219`).
   - 자전거 모델: 최대 조향 π/3, 가속 시상수 0.2초, 조향 시상수 0.05초(`batch_kinematic_bicycle.py:39-56`).
   - 차량(Pacifica): 길이 5.176 m, 폭 2.297 m, 축거 3.089 m, rear axle→중심 1.461 m.
3. **Comfort(PDMS v1)**는 추종된 41개 상태로 판정합니다(`pdm_comfort_metrics.py:13-30`).

   | 항목 | 한계 |
   |---|---|
   | 종가속 | −4.05 … +2.40 m/s² |
   | 횡가속 | 4.89 m/s² |
   | 전체 jerk | 8.37 m/s³ |
   | 종 jerk | 4.13 m/s³ |
   | yaw rate | 0.95 rad/s |
   | yaw 가속 | 1.93 rad/s² |

   - EC(extended comfort)는 v2 EPDMS에만 있습니다. 연속 두 프레임 궤적의 RMS 차이를 봅니다: 가속 0.7 m/s², jerk 0.5 m/s³, yaw rate 0.1 rad/s, yaw 가속 0.1 rad/s²(`/home/external-user/yongjae/navsim_v2/.../pdm_comfort_metrics.py:35-39, 430-464`). PDMS v1에는 없습니다.
4. **EP 정규화**
   - 진행량은 중심선 투영 길이로, 음수는 0으로 자릅니다(`pdm_scorer.py:442-456`).
   - 정규화 분모는 max(PDM-Closed, 예측)입니다. 곱셈 지표가 0이면 진행량도 0이 됩니다. 임계값은 yaml에서 5.0 m입니다(코드 기본값 0.1, `pdm_scorer.py:61, 167-185`).
   - 따라서 PDM-Closed보다 빨라져도 EP 이득은 0입니다.
   - TTC는 등속으로 0/0.3/0.6/0.9초 앞을 투영합니다(`pdm_scorer.py:469-497`). 빨라질수록 TTC 위험이 커집니다.
5. **heading 불일치의 영향과 기존 측정**
   - heading이 경로 방향과 어긋나면 추정 속도가 cos만큼 줄고, 곡률도 틀리고, heading 오차 가중 10이 차를 선 밖으로 밉니다.
   - student(v1)는 step별 (dx, dy, dh)를 누적합합니다. 그래서 heading 오차가 쌓입니다. 이번에 재측정한 값(navtest 무작위 3,000개 중 유효 2,774개)은 다음과 같습니다.

     | | step 1 | step 8 |
     |---|---|---|
     | 중앙값 | 0.29° | 0.97° |
     | p90 | 0.93° | 4.0° |

     KM report 22 §14의 수치(0.28→0.98°, p90 0.83→3.91°)와 일치합니다. 사람 GT는 step과 무관하게 중앙값 0.26°, p90 약 0.95°, p99 약 2.4°입니다.
   - 교정 코드: `heading_from_path`(`/home/external-user/kyungmin/SSR/navsim/agents/para_ssr/para_ssr_model.py:36-54`, 중앙차분, 0.5 m 미만 이동이면 원래 heading 유지).
   - KM report 22 §14에 따르면 평가 때만 적용해 interaction_final이 84.87 → 85.93이 되었습니다. DAC 실패 223개가 고쳐지고 138개가 새로 생겼습니다.

## (c) navtrain에서 잰 사람 경로 길이와 속도

**표본:** navtrain log 1,192개 중 250개를 무작위(seed 0)로 뽑았습니다. filter_scenes 규칙을 재현해 21,065 token을 얻었습니다. 요청하신 약 2,000개보다 큽니다. 스크립트: `/tmp/claude-1001/-home-external-user/9262a8d2-fa20-46d7-ba9a-92d7d13b8ceb/scratchpad/c_navtrain_paths.py`, `c_analyze.py`. 원자료: 같은 폴더의 `navtrain_paths.npz`.

**기본 분포**

| 항목 | 값 |
|---|---|
| v0 | 중앙값 4.37 m/s (p95 10.5) |
| 4초 경로 s4 | 중앙값 16.3 m (p5 3.3, p95 39.5) |
| 5초 경로 s5 | 중앙값 20.4 m |
| 4→5초 여분 경로 | 중앙값 4.26 m (p5 0.02, p95 10.2) |
| 거의 정지 | s4 < 0.5 m: 0.4%, s4 < 2 m: 3.1% |

**0.5초 간격이 끊기지 않는 경로가 있는 비율**

| 시간 | 비율 |
|---|---|
| 5초 | 98.7% |
| 6초 | 98.0% |
| 7초 | 97.4% |
| 8초 | 96.7% |

navtest는 5초 99.6%, 6초 99.1%입니다.

**사람 GT 자체의 t=0 연속성**
- x1 − (v0·0.5 + ½a0·0.25): 중앙값 −0.06 m (p5 −0.18, p95 +0.06).
- 첫 구간 속도 − v0: 중앙값 −0.14 m/s.

**사람 운동 범위** (곡률 한계 설정용, p99 / p99.9)

| 항목 | p99 | p99.9 |
|---|---|---|
| max yaw rate | 0.49 rad/s | 0.62 rad/s |
| max 곡률 | 0.145 /m | 0.213 /m |
| max 횡가속 | 2.6 m/s² | 3.8 m/s² |

**가속 초안을 만드는 방법별 요구조건**

| 방법 | 필요 경로 | scene(5초)만으로 | log까지 쓰면 | 연속성 |
|---|---|---|---|---|
| 균일 time-warp ×1.1 | 4.4초 | 가능 | 98.8% | t=0 속도 점프 중앙값 0.44 m/s |
| 균일 ×1.25 | 5.0초 | 가능 | 98.7% | 점프 중앙값 1.09 m/s. 7.9%는 2.4 m/s 초과 |
| 균일 ×1.4 | 5.6초 | 불가 | 98.0% | 점프 중앙값 1.75 m/s. 31%는 2.4 m/s 초과 |
| ramp warp τ'(t)=1+(f−1)·min(t/Tr,1), f=1.25, Tr=2초 | 4.75초 | 가능 | – | 추가 가속 중앙값 0.51, p90 1.07 m/s² |
| ramp f=1.4, Tr=2초 | 5.2초 | 불가 | 98.4% | 추가 가속 중앙값 0.82, p90 1.71. 4.9%는 2.0 초과 |
| 가산 가속 오프셋 δa(0초부터 일정, v(0) 유지) +0.5 m/s² | s4+4 m | 53.5% | 78.9% | 연속 |
| δa +1.0 | s4+8 m | 14.7% | 69.3% | 연속 |
| δa +2.0 | s4+16 m | 0.01% | 51.5% | 연속 |

- **균일 warp:** 초안 초기속도가 f·v0가 됩니다. 채점기는 실제 v0에서 출발하므로 LQR이 약 1초 안에 속도를 맞추려 급가감속합니다. 감속 warp ×0.6도 같은 크기의 불연속을 만듭니다(−0.4·v0, 31%에서 2.4 m/s 초과).
- **경로 외삽 오차:** 사람 4초 경로 끝에서 실제 5초·6초 위치로 외삽해 보았습니다(n 약 17.6k).

  | 외삽 | 등곡률(마지막 두 구간) 오차 p50 / p90 / p99 | 직선 오차 p90 / p99 |
  |---|---|---|
  | +1초 (중앙값 4.8 m) | 0.055 / 0.28 / 0.62 m | 0.73 / 1.63 m |
  | +2초 (중앙값 9.6 m) | 0.17 / 1.09 / 3.2 m | 2.3 / 5.3 m |

## (d) 실제 student 초안(navtest)과 사람 비교

**자료:**
- student 궤적: `report/planner_vs_perception_tests/dump/all_dets.npz`의 `ego_traj`. interaction_final seed 0이고 원 pkl과 비트 단위로 같습니다.
- 사람 궤적: test log에서 5–8초까지 추출했습니다. `human_navtest_trajectories.pkl`과의 최대 차 1.2e-7로 확인했습니다.
- 채점 결과: `report/perception_reliability/pdm_attr/table.parquet`의 csv_* 열.
- Frenet 좌표는 사람 경로 기준입니다.
- 스크립트: `scratchpad/d_analyze.py`. 결과: `scratchpad/d_student_vs_human.parquet`.

**전체 (사람 s4 > 2 m, 11,723개)**
- 4초 거리 비 Ss4/Sh4: p5 0.82, p10 0.88, p50 1.00, p90 1.16, p95 1.27.
  - 1.25 초과 5.7%, 1.4 초과 2.7%, 0.9 미만 12.7%, 0.8 미만 4.1%, 0.6 미만 0.5%.
- 4초 거리 차: p5 −2.8 m, p95 +3.1 m. 등가 가속 오프셋 δa_eq = 2Δs/16은 p1 −0.59, p5 −0.36, p95 +0.39, p99 +0.65 m/s²입니다. 99.7%가 ±1 m/s² 안에 있습니다.
- 횡편차:
  - 4초 시점 d: p5 −1.02, p95 +1.01 m. |d4| ≤ 2 m가 97.2%입니다.
  - 4초 동안 최대 |d|: p50 0.25, p90 1.02, p99 3.15 m.
  - 시점별: |d(0.5초)| p99 0.08 m, |d(2초)| p90 0.30 m.
- 첫 구간 속도 − v0: student 중앙값 −0.21 (p5 −0.57, p95 +0.38), 사람 중앙값 −0.18 (−0.55, +0.37). **실제 초안은 t=0에서 사람만큼 연속적입니다.**
- student의 4초 경로가 사람 5초 경로보다 긴 경우가 12.3%, 추출한 log 경로 전체(최대 8초)보다 긴 경우가 7.9%입니다.

**채점 결과별 분포**

| 그룹 | n | 거리 비 p10/p50/p90 | 비 >1.1 | δa_eq p50 | max\|d\| p50/p90 | \|d4\|>1 m |
|---|---|---|---|---|---|---|
| 통과 (score=1) | 3,657 | 0.88/1.00/1.21 | 21% | +0.01 | 0.19/0.88 | 8% |
| NC<1 | 278 | 0.93/1.16/2.03 | – | +0.26 (p90 +0.80) | 1.0/3.7 | – |
| NC=0 | 242 | 0.93/1.19/2.18 | 59% | +0.30 | 1.03/3.81 | 51% |
| NC<1, 움직이는 물체와 충돌 | 130 | Δs4 중앙값 +3.5 m, 비 1.23 | – | +0.43 | 0.70/3.3 | – |
| NC<1, 정지 물체와 충돌 | 148 | Δs4 +0.8 m, 비 1.05 | – | +0.10 | 1.1/4.2 | – |
| DAC 실패 | 813 | 0.92/1.00/1.12 | 14% | +0.01 | 0.61/1.66 | 24% |
| TTC 실패 (NC 통과) | 545 | 0.93/1.08/1.27 | 42% | +0.15 | 0.62/2.06 | 31% |
| EP<0.5 (NC·DAC 통과) | 73 | 0.50/0.70/0.91 | 0% | −0.20 | 0.07/0.40 | 0% |

- **NC 실패의 특징**
  - NC<1 중 54%는 student 4초 경로가 사람 5초 경로보다 깁니다.
  - NC 충돌 시각은 중앙값 3.5초(IQR 3.1–3.8)로, 알려진 수치와 같습니다.
  - NC 꼬리의 횡편차는 큽니다: d4 p1/p99 ±6.5 m. 다른 차선이나 다른 기동을 택한 경우를 포함하므로, 사람 대비 편차 전체를 "오차"로 볼 수는 없습니다.
- **사람이 거의 정지한 423 token:** student 4초 경로 중앙값 1.1 m. 이 중 NC<1인 19개에서는 student가 중앙값 5.1 m를 전진했습니다.
- **EP 여지**
  - 곱셈 지표 통과 11,082개의 EP 평균은 0.873입니다. 모두 EP=1이 된다고 쳐도 상한은 PDMS +4.84입니다.
  - EP<0.9인 통과 token 중 사람보다 느린(비 <0.95) 경우는 31%뿐이고, 중앙값 비는 0.99입니다.
  - 사람 궤적 자체도 navtest에서 EP 평균 0.870입니다. NC·DAC·TTC 실패는 0건, comfort 실패는 0.1%입니다(`v11_scores_human.csv`).
  - 즉 EP 손실은 대부분 "사람 속도 < PDM-Closed"입니다. 사람을 기준으로 한 progress loss로는 회수되지 않습니다.
- **감속 교정과 comfort:** 31번 안전 필터(A_MAX 4.0 m/s², jerk 제한 없는 greedy)를 적용해 수정된 token에서 comfort 실패가 0.75–3.4%였습니다(variant별). student 원래 궤적의 comfort 실패는 12,146개 중 1개입니다(`safety_filter/variant_table.parquet`).
- **참고(실험 E, 출처 그대로 인용):** 학습에 쓴 log에서의 student 초안은 외워진 상태입니다. 사람까지 평균 거리 0.33 m, 안 본 log는 0.61 m입니다. navtrain 충돌은 "사람과 같은 속도로 옆 약 0.6 m"형이고, navtest 충돌은 "4초에 +2.3 m 앞섬"형입니다(`E_train_split_feasibility/REPORT.md` 한눈에 1, 3). 이번 측정(Δs4 중앙값 +2.1 m)과 일치합니다.

## (e) Refiner 출력 파라미터화 제안 (설계안, 결정은 사용자 몫)

**e-1. 기준 경로 Γ**
- 원점과 초안 8점을 잇는 폴리라인입니다. 호장 S_k, 시간 대응 S(t_k)를 둡니다. 0.5초 구간 안에서는 등속으로 봅니다. 채점기의 선형 보간과 같은 방식이고, sf_common `Path`(`report/planner_vs_perception_tests/safety_filter/sf_common.py:57-112`)와도 같습니다.
- 수치 확인(student 초안 2,774개): δa −1 / −2 m/s²로 시간을 다시 매기면, 새 8점 폴리라인이 원 폴리라인 모서리를 깎는 양이 p99 0.095 / 0.079 m, 최대 0.12 m입니다.
- 시작 기울기를 (1,0)으로 고정한 cubic spline은 항등 재현 오차 0, 폴리라인과의 차 p99 0.075 m, 깎임 p99 0.069 m입니다.
- 둘 다 0.1 m 이하입니다. 참고로 KM `feasibility.py` docstring에 적힌 0.64 m 격자 SDF bilinear 오차 중앙값은 0.142 m입니다. 항등 재현이 정확한 폴리라인이 단순합니다.

**e-2. 종방향: 가산 가속 오프셋 δa(t)** (곱셈 time-warp 대신)
- 정의: v_new(t) = softplus 또는 max(0, v_d(t) + ∫₀ᵗ δa), s_new(t) = ∫ v_new.
- 연속성: δa(0) = 0(jerk ramp)으로 δv(0) = δs(0) = 0이 되고, t=0 속도가 보존됩니다.
- 표현: 0.1초 격자에서 B-spline 4–6개 계수를 tanh로 bound합니다.
- 제약 후보:
  - δa ∈ [−A_DEC, +A_ACC]. sf_common은 A_MAX = 4.0, A_ACC = 2.0이고(`sf_common.py:42-43`), 거기서 A_ACC는 "원래 속도 이하로의 재가속"에만 쓰입니다.
  - 합성 가속 a_d + δa ∈ [−4.05, +2.40].
  - |dδa/dt| ≤ 약 4 m/s³ (종 jerk 한계 4.13).
- 곱셈 warp보다 나은 근거:
  - 실제 오차 척도가 가산형입니다(±0.4 m/s²).
  - 정지한 초안도 움직일 수 있습니다.
  - 한계값이 comfort 기준과 바로 대응합니다.
- 교란 범위 근거: 실제 δa_eq는 99.7%가 ±1 m/s² 안이고, NC 실패 p95는 +0.91, p99는 +1.27입니다.

**e-3. 가속할 때의 경로 연장**
- s_new(4) > S_8이면 초안 끝 너머 경로가 필요합니다. **순가속은 항상 연장이 필요합니다.**
- 추론 시에는 등곡률 외삽을 씁니다. κ_end는 마지막 두 구간의 heading 변화로 잡고, |κ| ≤ min(0.95/v, 4.89/v²)로 자릅니다.
- 토큰별로 `ext_m = max(0, s_new(4) − S_8)`과 `ext_src ∈ {scene, log, extrap}`를 기록합니다.
- 연장 길이 상한 후보: 약 5 m 또는 v_end × 1초. 등곡률 외삽 오차가 +1초에서 p90 0.28 m, +2초에서 1.09 m이기 때문입니다.
- 학습 초안은 GT 경로로 연장할 수 있지만 refiner 자신의 연장은 외삽입니다. 학습과 추론을 맞추려면 refiner 출력의 연장은 항상 외삽으로 하고, GT 연장은 진단용으로만 비교하는 편이 일관됩니다.

**e-4. 횡방향: Γ 기준 Frenet 오프셋 d(s)** (시간이 아니라 호장의 함수)
- 호장의 함수로 두면 정지 시 곡률 발산을 피할 수 있습니다.
- d(0) = 0, d'(0) = 0입니다. 예: d(s) = s²·p(s), 또는 첫 두 제어점을 0으로 둔 B-spline.
- |d| ≤ D_MAX. 근거: 통과 그룹 max|d| p90 0.88 m, 전체 p95 1.5 m, NC 실패 p75 2.1 m.
- 곡률: κ_new ≈ (κ_Γ + d'')/(1 − κ_Γ d). |κ_new| ≤ min(0.95/v, 4.89/v²)로 제한하거나, 사람 p99 수준(0.145 /m, 횡가속 2.6 m/s²)의 soft 제약을 둡니다.
- 위치: p_k = Γ(s_new(t_k)) + d·n_Γ.
- heading 두 가지:
  - **H1 (path 모드):** 초안 heading을 호장으로 보간하고 atan(d'/(1−κd))를 더합니다. 보정량이 0이면 초안이 정확히 재현됩니다. 단 누적합 heading 결함도 그대로 남습니다.
  - **H2 (tangent 모드):** 출력 점의 중앙차분 heading. 0.5 m 미만 이동이면 원 heading을 유지합니다(`heading_from_path`).
  - H2는 그 자체로 PDMS를 약 +1.06 올리고 실패 집합도 바꿉니다(KM 22 §14). 그래서 기준선은 **"같은 디코더에 보정 0을 넣은 초안"**이어야 합니다. H1/H2 중 하나로 고정한 뒤 실패 층을 정의해야 합니다(`report/kd_ideation/05_reading_kd_question_design.md` C2와 같은 경고).

**e-5. 손실 샘플링**
- 채점기는 8점 사이를 선형 보간한 0.1초 참조를 추종합니다. 따라서 surrogate(충돌·SDF)는 t=0 실제 ego 상태를 포함한 41점 선형 보간 위에서 계산하는 편이 맞습니다.
- 32-B 원 근사 penalty는 8개 knot에서만 계산했습니다. 반지름은 1.436 m, 여유는 0.3 m, ego 원 3개입니다(`report/cause_and_correction_tests/B_read_vs_generate/b1_train.py:14-17, 114-139`). 10 m/s이면 knot 간격이 5 m입니다.

**e-6. 대안(참고): 제어 공간 출력**
- KM `modules/kinematics.py`는 (a, ω)를 출력하고 자전거 모델로 적분합니다. TOAD 한계값은 +2.40/−4.05/0.95입니다.
- KM 22 §14에서 v1 초안에 TOAD 투영을 사후 적용했을 때 84.87 → 82.41로 떨어졌습니다. 그 보고서는 원인을 "틀린 heading으로 적분했기 때문"이라고 해석합니다.
- 이 결과는 사후 투영에 대한 것입니다. 학습된 제어 출력 헤드의 성능을 말해 주지는 않습니다.

## 피드백 1에 대한 초안 생성 선택지 (판정하지 않음)

| 선택지 | 내용 | 비용·위험 |
|---|---|---|
| A. 감속만 허용 | 느린 교란 초안(δa < 0)은 교정 대상에서 빼거나 target을 "변화 없음"으로 둔다 | 실제 초안 중 사람보다 느린 경우(비 <0.9, 12.7%)를 다룰 수 없다. EP 여지는 원래 작다 |
| B. 제한 가속 허용 (+A_ACC 약 0.5–1.0, 연장 ≤ 5 m 외삽 + flag) | 느린 초안 복구 가능 | 연장은 늘 외삽이다. TTC 위험이 늘고, PDM-Closed보다 빠르면 EP 이득이 0이다 |
| C. 비대칭 | 교란 범위 δa ∈ [−0.7, +1.3] (실제 p1–p99 포괄), refiner는 −4…+1 | 설계가 복잡해진다 |

**공통 규칙 후보**
- 교란은 refiner와 **같은 디코더**로 만듭니다: 초안 = decode(GT 경로, δa_pert, d_pert). 그러면 모든 교란이 표현 가능하고 역변환도 가능합니다.
- t=0 연속성을 유지합니다: δa(0) = 0, d(0) = d'(0) = 0. 균일 time-warp는 쓰지 않습니다.
- 빠른 교란 초안의 경로는 GT 5초(scene) 또는 log 6초(98%)로 채웁니다. 모자라면 외삽하고 `ext_src` flag를 답니다. 프레임 누락 token(navtrain 1.34%)은 뺍니다.
- 안전한 초안을 충분히 넣습니다.
  - 항등 초안(=GT)을 포함합니다. navtest에서는 사람 궤적의 NC/DAC/TTC 실패가 0건입니다. navtrain에서는 미검증입니다.
  - 교정 임계 미만의 작은 교란도 넣고 target을 "변화 없음"으로 둡니다.
  - 실제 student 초안도 대부분 안전합니다(NC 실패 navtest 2.3%, 학습 log 0.7%, 안 본 log 1.5%).
- 실제 student 초안으로 학습할 때는 외워진 학습 log 초안을 피합니다. 대신 쓸 수 있는 것:
  - held-out log: val_logs 214개, 18,179장면.
  - 이른 checkpoint dump: `E_train_split_feasibility/dump_ep2`, `dump_ep9`, `dump_ep19`.

## 피드백 2와 관련해 이번에 코드로 확인한 것 (간단히)

- `_compute_agent_targets`는 **현재 프레임**에서 다음 조건을 만족하는 객체만 대상으로 합니다(`para_ssr_targets.py:297-334`, 설정 `configs/default.py:70, 167, 183, 188`).
  - ROI: y_forward 0..32 m, |x_right| ≤ 32 m.
  - FOV ±80°.
  - 가까운 순 최대 100개.
- `_track_future`는 track이 처음 사라지는 프레임에서 멈추고, 이후 mask는 0입니다(`para_ssr_targets.py:401-433`).
- 따라서 나중에 진입하는 객체, 뒤쪽이나 FOV 밖 객체, 잠시 가려진 객체의 나머지 구간은 이 target에 없습니다. 충돌 surrogate는 이 target이 아니라 metric cache의 observation(미래 전 시점 전체 객체)에서 만드는 것이 안전합니다. metric cache에 들어 있는 필드: `centerline`, `drivable_area_map`, `ego_state`, `observation`, `route_lane_ids`, `trajectory`.

## 스크립트와 결과 파일

모든 작업은 읽기 전용이었습니다. repo 파일은 수정하지 않았습니다. 모두 `/tmp/claude-1001/-home-external-user/9262a8d2-fa20-46d7-ba9a-92d7d13b8ceb/scratchpad/` 아래에 있습니다.
- `c_navtrain_paths.py`, `c_analyze.py`, `navtrain_paths.npz`
- `d_navtest_paths.py`, `d_analyze.py`, `navtest_paths.npz`, `d_student_vs_human.parquet`
- `e_decode_check.py`