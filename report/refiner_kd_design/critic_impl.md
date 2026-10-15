# 구현 검토: refiner 기반 teacher KD 아키텍처 초안

읽기 전용으로만 확인했습니다. 수정한 파일은 없고, 계산은 CPU(nice 10, 실행당 10분 미만)로만 돌렸습니다. 검증 스크립트는 `/tmp/claude-1001/-home-external-user/9262a8d2-fa20-46d7-ba9a-92d7d13b8ceb/scratchpad/review/`에 있습니다: `t_score.py`, `t_batch_eq2.py`, `bspline_chk.py`, `cmf_chk.py`, `probe_map.py`, `t_sdf.py`, `t_io.py`.

## 요약

- **그대로 구현하면 틀리는 곳이 5개 있습니다.**
  - 종방향 spline의 가속 한계가 실제로는 최대 3배(12 m/s²)까지 나갑니다.
  - softplus 때문에 교정이 0이어도 초안이 그대로 나오지 않습니다.
  - comfort 손실을 0.1 s 선형보간 궤적에 걸면 사람 궤적도 71%(가속)~99.9%(jerk)가 위반으로 잡힙니다.
  - 이 v1 fork는 DDC를 곱셈 지표로 씁니다. 초안은 DDC를 빠뜨렸습니다.
  - 초안 여러 개를 한 번에 채점할 때 EP와 DDC가 공식값과 달라집니다.
- **이 5개는 모두 정확한 수정안이 있습니다.** 배치 채점은 수정하면 공식 채점과 140/140 일치합니다.
- **설계 수준에서 사용자가 결정할 문제가 5개 있습니다.** teacher BEV의 지도 정보가 약한데 주 지표에 DAC가 섞여 있는 점, heading drift를 다루지 않는 교란 초안 분포, TRAIN-OUT과 navtest의 도시·실패율 차이, E2E에서 KD 신호가 사라질 위험, arm × seed × 가중 격자 계산량입니다.
- **초안이 "미측정/미확인"으로 둔 4가지는 이번에 확인했습니다.**
  - 채점 속도: token당 약 0.07 s. 초안 은행 236k개도 CPU로 수십 분입니다.
  - red-light 객체: 공식 NC와 TTC 모두 건너뜁니다.
  - metric cache의 `trajectory`: PDM-Closed 궤적이 맞습니다.
  - 확장 격자 SDF 생성: token당 0.73 s.
- **GT 미래 물체 빌더와 SDF는 1주 안에 충분히 가능합니다.** 작업량은 약 2일입니다.
- **단계 F 전체는 1인 기준 작업일 약 13일입니다.** 초안 일정(V1–V9 8–10일)에는 M2/M3/M5/M7 코딩과 채점 래퍼 작업이 빠져 있습니다. arm 범위를 줄이면 약 11일입니다.

## 확인한 환경과 비용

| 항목 | 측정값 |
|---|---|
| 디스크 | `/` 735 GB 여유(79% 사용, 공유). `/home/external-user/ssd`는 1.6 TB 여유이고 쓰기 가능 → 캐시는 SSD에 둡니다 |
| RAM / CPU | 251 GB 중 가용 46 GB. 32코어에 load average 약 28 → 실제로 쓸 수 있는 CPU는 약 4코어입니다 |
| GPU | 6개 모두 사용 중(8–15 GB / 32 GB, 사용률 38–100%). 가정한 2장 × 약 20 GB는 성립하지만 SM을 공유하므로 느려집니다 |
| teacher npz 읽기 | 4.4 ms/파일, 단일 스레드 586 MB/s → I/O는 병목이 아닙니다 |
| metric cache 로드 / 공식 채점 | 0.042 s / 궤적 1개 0.030 s / 14개 배치 0.073 s |
| 확장 격자 SDF [320,256] | metric cache 폴리곤으로 만들 때 중앙값 0.73 s, 최대 1.4 s. 30k token에 약 6 CPU-h, worker 4개로 약 1.5 h |
| raw log | trainval 1,310개, 합계 14 GB(개당 약 10 MB) → M6 빌더 실행은 몇 분이면 끝납니다 |
| 단계 F 저장량 | 약 95 GB(teacher 복사 없음) ~ 175 GB(teacher를 memmap으로 복사). 초안의 추정과 같습니다 |

## 이슈

### A. 명세 오류 (그대로 구현하면 버그)

**1. [High] M4 종방향 B-spline의 가속·jerk 한계가 성립하지 않습니다**
- 매듭이 [0,0,0,0, .8,1.6,2.4,3.2, 4,4,4,4]인 clamped spline에서 증분을 `c_i = c_{i−1} + 0.8·u_i`로 두면, 도함수 제어점이 `3Δc/(t_{i+4} − t_{i+1})`가 됩니다. 앞쪽 계수는 1.5u, 뒤쪽 끝은 3u입니다.
- 실측(`bspline_chk.py`): A = 4일 때 최대 |δa| = **12.0 m/s²**(t = 4 s), |jerk| 최대 44.7 m/s³. "hard 제약"이라는 설명이 틀립니다.
- **수정:**
  - 증분을 `c_i = c_{i−1} + ((t_{i+3} − t_i)/3)·u_i`로 바꿉니다 (i = 2..7). 이러면 |δa| ≤ A가 정확히 성립합니다. 실측 max 4.00, δa(0) = 0.
  - jerk는 |u_{i+1} − u_i|를 제한해서 묶습니다.
  - V3 단위 시험에 해석적 도함수 검사를 넣습니다.

**2. [High] softplus(β = 0.3) 때문에 항등성과 모드 A의 s1 ≤ s0가 깨집니다**
- softplus(v) − v 값: v = 0에서 +0.208 m/s(4 s 동안 약 0.83 m 전진), v = 0.5에서 +0.052.
- 정지·저속 초안이 바로 L-creep류 NC 사례입니다. 사람이 거의 정지한 423 token 중 NC 19건이 여기에 해당합니다.
- **수정:** 학습과 추론 모두 `v1 = relu(v0 + Δv)`를 씁니다. softplus를 없애면 학습과 평가도 일치합니다.

**3. [High] C_cmf를 41점 선형보간 참조에서 계산하면 안 됩니다**
- 공식 comfort는 LQR로 추종한 상태에서 판정합니다(`pdm_comfort_metrics.py`).
- 사람 navtest 궤적을 0.1 s 선형보간하면(`cmf_chk.py`):
  - 0.9 × 한계를 넘는 경우: 가속 **71%**, jerk **99.9%**.
  - max|a| 중앙값 4.57 m/s², p90 8.31. max|jerk| 중앙값 45.7.
- 같은 궤적을 keyframe(0.5 s) 단위로 보면 위반은 1.2%입니다.
- **수정:**
  - 종방향 comfort는 (a) t0 실제 상태(v0, a0)를 포함한 0.5 s 차분으로 계산하고, (b) Δ 성분은 spline의 해석적 δa, jerk, v²·d″로 계산합니다.
  - 합격 기준: 사람 궤적 위반율 ≤ 1%. V3에서 확인합니다.

**4. [High] DDC가 빠져 있습니다**
- 이 fork에서 DDC는 곱셈 지표입니다. `pdm_scorer.py` L158에서 계산하고 L174/L195에서 곱해집니다.
- 공식 csv에서 DDC<1은 1건이지만, 2 m 횡이동은 반대 차선 진입으로 새 곱셈 실패를 만들 수 있습니다.
- **수정:**
  - 실패 집합, 새 실패 수, 전체 실패 수, gate 라벨(NC∪DAC∪DDC), M8 층화에 DDC를 넣습니다.
  - 필요하면 on-route lane 폴리곤으로 surrogate를 추가합니다.

**5. [High] 배치 채점에 함정이 두 개 있습니다 (수정하면 정확히 일치)**
- 기존 `navsim/evaluate/pdm_score.py` L165의 `pdm_score_multi_trajs`는 PDM-Closed를 빼고 EP를 초안끼리 정규화합니다.
- 공식 DDC 코드(`pdm_scorer.py` L398-439, 특히 L421-428)는 시간축이 아니라 **proposal 축**으로 합산합니다. 그래서 공식 DDC_model = thr(max_t(op_pdm[t] + op_model[t]))가 됩니다.
- 실측(navtest 20 token × 초안 7개, `t_batch_eq2.py`):
  - 순진한 배치: EP 불일치 55/140, DDC 4/140.
  - 수정 후 NC, DAC, DDC, TTC, comfort, EP, PDMS 모두 **0/140 불일치**.
- 속도: 초안 은행 18,179 × 13은 약 36 CPU-분입니다. 초안의 "채점 속도 미측정, 느리면 K=8"은 필요 없습니다.
- **수정:** `tools/refiner/score_trajectories.py`를 이렇게 만듭니다.
  - token마다 [PDM-Closed, 초안 1..K]를 한 번에 `simulate_proposals`와 `score_proposals`로 돌립니다.
  - `DDC_i = thr(max_t(op[0] + op[i]))`, `mult_i`를 다시 계산합니다.
  - `EP_i = raw_i·mult_i / max(raw_0·mult_0, raw_i·mult_i)`. 단 이 max가 5.0 m 이하이면 1(mult = 0이면 0).
  - PDMS = mult·(5EP + 5TTC + 2C)/12.
  - pdm_score 단일 호출과 500 token에서 동등성 시험을 합니다.

**6. [Medium] L_prog와 P_pdm의 정의를 공식에 맞춰야 합니다**
- 공식 EP 규칙(`pdm_scorer.py` L171-185, yaml `progress_distance_threshold: 5.0`):
  - max(raw·mult)가 5 m 이하이면 EP = 1(mult = 0이면 0).
  - 초안 식 `min(1, P/max(P_pdm, 5 m))`는 이 경우를 틀리게 계산합니다.
- 진행량은 ego **중심** 좌표를 t = 0에서 4 s까지 중심선에 투영한 값입니다(L442-456).
- `metric_cache.trajectory`는 PDM-Closed가 맞습니다(`pdm_score.py` pdm_score 안 `pdm_trajectory = metric_cache.trajectory`, `metric_cache_processor.py` L255-263). [가정] 표시를 지웁니다.
- **수정:** 5번 채점기에서 token마다 `P_pdm_eff = _progress_raw[0]·mult[0]`(LQR 추종, DDC 포함)을 저장해 L_prog에 씁니다.

**7. [Medium] M6 세부 규칙을 metric cache와 정확히 맞춰야 합니다**
- 시간은 실제 timestamp가 아니라 **index 기반**입니다(k·0.5, `metric_cache_processor.py` L127-137). 1 s 간격이 비는 token도 같은 규칙을 따르고 flag를 답니다. 기존 `rebuild_check.py`의 `d[i:i+11]`도 같은 방식입니다.
- 정적 물체의 속도는 0입니다. velocity는 AGENT_TYPES만 채웁니다(L150-156).
- 속도와 heading은 각 프레임의 ego 좌표에서 t0 좌표로 **회전**해야 합니다. `_track_future`는 중심만 변환하므로 그대로 쓰면 안 됩니다.
- red-light: 공식 NC(L343)와 TTC(L516)가 모두 건너뜁니다. 빼는 것이 맞고, [미확인]을 지웁니다.
- is_agent: nuplan `AGENT_TYPES`는 vehicle/pedestrian/bicycle(+ego)입니다. 초안이 맞습니다.

**8. [Medium] 꺾은선 Γ로는 κ_Γ와 법선이 정의되지 않습니다**
- H1의 heading 식 `atan2(d′, 1 − κ_Γ·d)`와 κ_new에 필요한 κ_Γ가, 꺾은선에서는 꼭짓점의 델타 함수입니다.
- **수정:**
  - Γ를 원점과 8점을 지나는 C2 interpolating spline으로 둡니다. 시작 접선은 (1,0), 매개변수는 spline 호장입니다. 사실 조사 e-1에 따르면 항등 오차 0, 꺾은선과의 차이 p99 0.075 m입니다.
  - s0(t)는 spline 호장의 knot을 잇는 꺾은선으로 둡니다. knot에서 항등이 정확히 유지됩니다.
  - M3 통로도 같은 Γ를 씁니다.

**9. [Medium] 횡방향 곡률 제약이 학습과 추론에서 다릅니다**
- 추론 때만 e를 축소하므로 학습과 평가가 어긋납니다.
- S_8이 3–10 m인 짧은 경로에서 D_MAX = 2 m는 곡률 한계(0.213 /m)를 크게 넘습니다.
- **수정:** decoder 안에서 학습과 추론 모두 `α = min(1, κ_lim/max|κ_new|)` 투영을 미분 가능하게 적용합니다. κ_lim = min(0.95/v, 4.89/v², 0.213).

**10. [Medium] "모든 교란은 M4 디코더로"와 family 정의가 모순됩니다**
- L-ignore-brake는 δa(0) = −α·a_h(0) ≠ 0입니다. L-const의 t_on은 knot 격자에 있지 않습니다.
- M4의 입력 `decode(τ0 [8,3], …)`로는 빠른 교란이 쓰는 5–8 s log 경로를 넣을 수 없습니다.
- **수정:**
  - API: `decode(path_xyh[M,3], s_knots[9], u_lon[6], w_lat[6], mode, ext_policy)`. τ0 입력은 그 특수한 경우로 둡니다.
  - family는 u 공간에서 직접 샘플링하거나, 제약된 basis에 최소제곱 투영한 결과를 초안으로 씁니다. A_p는 파생 통계로만 기록합니다.

**11. [Medium] UNKNOWN 상태의 처리가 정의되지 않았고, 실제로는 거의 비어 있습니다**
- 공식 판정 구간은 최대 4.9 s < 5.0 s이고, 초안 ego는 GT ego에서 수십 m 안에 있습니다. 75 m 밖 조건은 사실상 발생하지 않습니다.
- 실제 "안 보이는 것"은 등장 이전(ABSENT_OFFICIAL)입니다.
- **수정:**
  - V2에서 UNKNOWN 비율을 잽니다. 0.1% 미만이면 해당 초안을 학습에서 뺍니다.
  - pre_entry/post_exit flag는 층화 분석에만 씁니다.
  - M6 반경은 max(80, S_avail + 25) m로 둡니다.

**12. [Low] `grid_sample` 좌표를 명시해야 합니다** (align_corners=False, grid[...,0] = 열, grid[...,1] = 행)
- S 격자: `gx = −y_left/32`, `gy = x/16 − 1`.
- E 격자: `gx = −y_left/32`, `gy = (x − 32)/40`.
- 단위 시험: teacher heatmap의 박스 중심(뒤집은 뒤)과 student 검출 박스 중심을 찍어 확인합니다.

**13. [Low] 박스 변환 규칙을 적어 둡니다**
- student 박스:
  - heading_navsim = −yaw_ssr − π (`report/collision_counterfactual/counterfactual/cf_common.py` L12-15, 검증된 규약)
  - (x_f, y_l) = (y_fwd, −x_right), (vx_f, vy_l) = (vy_fwd, −vx_right)
- teacher 박스: `tools/rescore_teacher_detection.py` L49-58을 씁니다. manifest의 `z_gravity_centre`와 이 도구의 bottom-z 해석이 충돌하지만, BEV에는 영향이 없습니다.

**14. [Low] M5 세부 사항**
- u_d는 실제로 69차원입니다(72 아님).
- gate head가 trunk에 기울기를 보내는지 명시해야 합니다. 권장은 `sg(trunk)`입니다.
- gate가 꺼졌을 때는 decode(τ0, 0) 대신 **τ0 원본 bytes**를 출력합니다. float 왕복 오차가 LQR 판정에 끼는 것을 막기 위해서입니다.
- ego status는 추론 때 쓰는 `status_feature`에서 가져와 D1에 함께 dump합니다.

**15. [Low] M9 수치가 사실 조사와 맞지 않습니다**
- 층화 수치 149/119/11은 사실 조사와 다릅니다. 사실 조사는 움직임 기준 130/148, 충돌 유형 기준 153/121/5, t0 부재 11입니다. `pdm_attr/table.parquet`에서 정의를 명시해 다시 셉니다.
- 기준선 NC 수: csv는 NC<1이 278건(그중 0.5가 36건), pdm_attr은 279건입니다.
- **수정:** 기준선 τ0를 5번 채점기로 다시 채점하고, 같은 파이프라인 안에서만 비교합니다.

### B. 설계 수준 문제 (선택지 제시, 결정은 사용자)

**16. [High] 주 지표 NC∪DAC가 teacher에 구조적으로 불리합니다**
- teacher는 검출 전용 BEVFusion입니다.
- linear probe(navtest 400 token으로 학습, 100 token으로 평가, `probe_map.py`) 결과:

  | 클래스 | teacher BEV IoU | 위치 prior IoU |
  |---|---|---|
  | road | 0.646 | 0.426 |
  | walkway | 0.116 | 0 |
  | centerline | 0.071 | – |
  | crosswalk | 0.049 | 0 |

- 실패 수는 DAC 813이 NC 279보다 훨씬 많습니다. 이 때문에 R_T와 R_S의 gate 비교가 DAC 쪽에서 R_T에게 불리하게 기울 수 있습니다.
- 선택지:
  - (a) 주 지표를 NC(+TTC) 순해소(같은 진행)로 두고, DAC와 DDC는 비열등성 guard로 둡니다.
  - (b) R_{T⊕S}(teacher + student BEV concat) 대 R_S arm을 추가합니다. "student에 teacher가 더하는 몫"을 직접 재는 비교이고 KD와 가장 가깝습니다. 단 E2E에서는 고정된 R_{T⊕S}가 계속 변하는 student feature를 읽게 됩니다.
  - (c) 모든 arm에 같은 지도 채널을 줍니다.

**17. [High] 교란 초안에 heading drift가 없습니다 (H1 기준)**
- 실제 student 초안의 step 8 heading 오차는 중앙값 0.97°, p90 4.0°입니다. 사람 기반 교란은 사람 heading(p90 약 0.95°)을 그대로 씁니다.
- 따라서 heading 때문에 생기는 DAC 실패는 학습 분포 밖에 있습니다.
- **수정 후보:**
  - 위치는 그대로 두고 누적 heading 잡음을 student 분포에 맞춘 "H-drift" family를 추가합니다.
  - 또는 H1과 H2를 섞는 학습 스칼라를 자유도로 추가합니다. 기준선은 여전히 정확한 항등으로 유지됩니다.

**18. [Medium] TRAIN-OUT과 navtest의 분포가 다르고, inner-val이 작습니다**
- TRAIN-OUT token의 도시 구성: SG 36.4%, LV 26.4%, Boston 20.8%, Pitt 16.4%.
- navtest log 구성(147개 파일 기준): Boston 62, LV 50, Pitt 22, SG 13.
- 실패율 비교:

  | 지표 | TRAIN-OUT (E, 3,000) | navtest |
  |---|---|---|
  | NC | 1.5% | 2.3% |
  | DAC | 9.2% | 6.7% |
  | TTC | 5.3% | 6.4% |
  | PDMS | 0.828 | 0.849 |

- inner-val 20%에는 NC 실패가 약 55건밖에 없습니다.
- **수정:**
  - 도시별로 층화한 log 단위 5-fold cross-fitting으로 θ와 가중치를 고릅니다. 모든 token에 out-of-fold 예측이 생깁니다.
  - inner-val은 navtest 도시 비율로 재가중하고, 도시별 결과도 보고합니다.

**19. [Medium] E2E에서 KD 신호가 사라질 위험이 있습니다**
- TRAIN-IN의 student 초안은 외워진 상태입니다(사람과의 평균 거리 0.33 m, NC 0.7%). 학습 후반에는 R_T target이 대부분 0에 가까워집니다.
- **수정:** E2E에서 sample의 일부(예: 50%)는 sg(τ0)에 M1 교란(u 공간)을 걸어 R_S와 R_T 모두에 넣고 KD합니다. 교란 없는 KD와 교란 KD를 분리해서 보고합니다.

**20. [Medium] navtest에서 θ를 맞추는 것(D2)은 누수 규칙(0-2의 7번)과 충돌합니다**
- **수정:**
  - ΔEP 예산(예: −0.25 / −0.5점)을 사전 등록합니다. arm별 θ는 cross-fit으로 정합니다. navtest에서는 실현된 ΔEP와 전체 곡선을 보고합니다.
  - 곡선에 재채점은 필요 없습니다. PDMS는 token별로 독립이므로 τ1을 한 번만 채점하면, 어떤 θ든 token별로 τ0 점수와 τ1 점수를 섞어서 계산할 수 있습니다.

**21. [Medium] 계산량이 급격히 늘어납니다**
- arm 11개 × seed 3 = 33 run. 여기에 가중 격자 12개 × 3개 이상 arm이면 약 70 run입니다.
- CPU가 부족해서 dataloader worker가 실제 병목입니다.
- **수정:**
  - 가중치는 비교하지 않는 arm(R_none, R_GT-cur)에서 한 번만 보정하고 모든 arm에 고정합니다.
  - arm을 세 단계로 나눕니다.
    - Tier 1: R_S, R_T, R_none, R_GT-cur × seed 3.
    - Tier 2: R_GT-fut, R_T+box, R_S+h × seed 1.
    - Tier 3(여유가 있으면): R_GT-cur-obj, R_T-box, R_S-box, R_GT-fut-ext.
  - 모든 입력을 split별 memmap으로 미리 묶어 두면 loader는 slicing만 하므로 run당 worker 1개면 됩니다. feature [N,256,50,100] f16, objects [N,512,11,6], SDF [N,320,256], drafts [N,13,8,3].
  - GPU당 run을 3개씩 동시에 돌립니다.

**22. [Low] kyungmin 코드는 복사해서 고정해야 합니다**
- `SSR-v2/.../feasibility.py`는 2026-09-27 01:36에 수정됐습니다(아직 작업 중). import하지 말고 hash를 기록한 사본을 둡니다.
- `SSR-v2/tools/readout/cache_student_bev.py`는 `_env.REPO`(kyungmin repo)와 그쪽 data root를 씁니다. 실행하지 말고 shard 형식만 참고합니다.
- 우리 dumper는 `E_train_split_feasibility/e1_dump.py`에 두 가지를 더해 만듭니다.
  - `bev_embed` → `.view(b,50,100,256).permute(0,3,1,2)`.
  - `pts_bbox_head.final_norm` hook으로 h_final을 얻습니다(`modules/planner_head.py` L197/L350). `outs["scene_query"]`는 predictions에 포함되지 않습니다(`para_ssr_model.py` L383-397).

**23. [Low] E2E 비용의 근거와 GPU 메모리**
- "report 34 추정 26 h"는 report 34에서 찾지 못했습니다.
- checkpoint 시각으로 잰 epoch당 시간: version_2 약 49분, version_1 약 76분. 설정은 GPU 2장, batch 4, accum 16, fp32, GridMask입니다(hydra config L120427-120560).
- 따라서 30 epoch는 약 25–38 h이고, 8 run을 순차로 돌리면 8.5–13일입니다.
- interaction_final은 seed 0만 있습니다.
- 다른 작업이 GPU당 8–15 GB를 쓰고 있으므로, 시작 전에 peak 메모리를 재야 합니다. 부족하면 batch 2 × accum 32로 바꿉니다.
- 권장: E0(기존 seed 0) + E1/E2 × 2 seed를 먼저 돌리고, E3는 선택으로 둡니다.

**24. [Low] §6 서술을 조금 더 조심스럽게 고칩니다**
- kyungmin의 S_student 82.32 ≥ S_own 81.49는 **ReSMap(지도) teacher**에서 나온 결과입니다. 그 결론도 나중에 "z로 쟀을 때만 성립"으로 고쳐졌습니다(`report/collaborator_audit/audit_km-ssr.md` L47-49). "직접적인 경고"를 "다른 teacher에서 나온 유사 위험 신호"로 바꿉니다.
- byounggun의 1 epoch adapter는 현재 Stage 2 run에서만 확인됐습니다. version_0–4는 가능성이 높지만 미확인입니다(`audit_bg.md` L70-75).
- 나머지 수치는 audit과 일치합니다: −0.362, +0.007, +0.02, 0.8301–0.8563, ±0.003, 88.13 대 87.80.

## 구현 순서 (1인 기준, 공유 GPU 2장, 가용 CPU 약 4코어)

| 일 | 작업 | 산출물과 합격 기준 |
|---|---|---|
| 0.5 | V1 확인. D5(TRAIN-OUT metric cache 15,179개, 백그라운드 약 2 h) 시작. D1 dumper 작성 후 실행(30,325 token, GPU 1장 45–60분) | dump와 E dump(TRAIN-OUT 3,000개), navtest pkl이 bit 단위로 같음 |
| 1–2 | `geometry.py`, `decoder.py` (이슈 1, 2, 8, 9, 10 반영) | 항등 오차 0, 역교정 < 0.1 m, 한계 위반 0, 41점이 채점기 참조와 일치 |
| 2 | 배치 공식 채점기 (이슈 4, 5, 6) | pdm_score와 500 token 전부 일치, P_pdm_eff 저장 |
| 3 | M6 빌더 (이슈 7, 11) + V2. SDF 빌더(백그라운드 약 1.5 h) | 모서리 오차 < 1e-3 m, UNKNOWN 비율 측정 |
| 4 | M1 초안 은행 (u 공간 샘플링, H-drift family) + pilot 500 × 13 채점 → 전체 은행 채점(수십 분) | 실패율 20–35%, t0 연속성 확인 |
| 5–6 | M7 torch 구현 (box-box smoothmax, keyframe comfort) + M8 검증 | 사전 합격 기준 판정. 불합격이면 M7b |
| 5 (병행) | 현재·미래 GT raster (walkway/crosswalk/centerline용 map API) | 약 1 d 코딩 + 1–2 h 실행 |
| 7–8 | memmap 패킹, M2/M3/M5, 학습·평가 루프, 파라미터 수 동일성 시험 | – |
| 9 | V7: R_GT-cur 대 R_none (seed 1) + 디버깅 | 0단계 기준 |
| 10–12 | Tier 1 × seed 3과 가중 보정(R_none/R_GT-cur), Tier 2 × seed 1 | run당 약 1–2 h (미측정, V7에서 잼), 동시 실행 3개/GPU |
| 13 | navtest: run마다 τ1을 1회 채점(각 약 14 CPU-분), 분석, 사전 판정 | – |

- 합계 작업일 약 13일입니다. Tier 3와 모드 B arm을 빼면 약 11일입니다.
- GT 미래 물체 빌더와 SDF(3일차)는 1주 목표 안에 여유 있게 들어갑니다.