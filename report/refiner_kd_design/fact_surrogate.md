# 미분 가능한 surrogate loss와 공식 채점 기준 검증: 사실 조사와 pilot 결과

## 핵심 요약

1. **계획 궤적만 보는 surrogate는 공식 실패의 약 1/4~1/3을 원리적으로 놓칩니다.** 공식 NC/DAC는 LQR로 추종한 궤적을 0.1 s 간격 41시점에서 판정합니다. 계획 궤적(raw plan)과 실제 추종 궤적의 최대 편차는 중앙값 0.59 m, p90 1.01 m입니다. 원래 궤적의 NC 실패 279건 중 32.3%는 raw 궤적에서 GT 물체와 겹침이 전혀 없습니다. DAC 실패도 약 26%는 raw 궤적의 모서리가 도로 안에 있습니다.
   - safety filter의 GT 변형 결과도 같은 방향입니다. raw 궤적만으로 회피를 검증한 `gt_m3_ref`는 279건 중 170건만 해소했고, 추종 궤적까지 확인한 `gt_m3`는 266건을 해소했습니다.
2. **"없음"을 "장애물 없음"으로 처리하는 경로가 두 곳 있습니다.**
   - `_compute_agent_targets`의 ROI 제한: 원래 NC 실패의 원인 물체 가운데 최소 17.2%(48/279)가 빠집니다.
   - kyungmin `feasibility.py`의 border padding: 필드가 전방 0~32 m만 덮는데, 원래 궤적의 25.5%가 그 밖으로 나갑니다.
3. **32-B circle penalty는 원래 궤적에서는 잘 맞지만, 궤적이 크게 바뀌면 급격히 나빠집니다.** 원래 궤적에서 AUC 0.977이고, D2e_0.1에서는 0.593입니다. 물체 집합을 원래 경로와 사람 경로 기준으로 골랐기 때문일 가능성이 높습니다(사례별 검증은 안 함).
4. **DAC 층 차이는 실질적으로 무시해도 됩니다.** kyungmin의 영역 합집합은 공식 DAC 영역보다 넓지만, 판정이 달라진 궤적은 5,658개 중 1개였습니다.

## (a) kyungmin의 SDF / feasibility 코드

`/home/external-user/kyungmin/SSR/navsim/agents/para_ssr/plan_map.py`와 SSR-v2의 같은 파일은 내용이 동일합니다(diff 없음). `feasibility.py`는 SSR-v2에만 있습니다. `/home/external-user/yongjae/SSR` 쪽에는 두 파일 모두 없습니다.

### plan_map.py (정적 SDF)

- **Footprint**: Pacifica 기준, 폭 2.297 m, 뒤축에서 앞 4.049 m / 뒤 1.127 m, 모서리 4개(:39-44). 모서리 계산은 `footprint_corners`(:130-136)입니다.
- **영역 질의**: `drivable_polygon`(:98-120)이 현재 ego 자세를 중심으로 `map_api.get_proximal_map_objects`를 호출합니다.
  - 반경은 extent 모서리까지 최대 거리의 1.05배입니다.
  - 층: ROADBLOCK 폴리곤, ROADBLOCK·ROADBLOCK_CONNECTOR의 interior_edges(lane, lane connector), INTERSECTION, CARPARK_AREA를 `unary_union`한 뒤 ego 좌표로 변환합니다.
- **Raster**: `rasterize_sdf`(:52-77)
  - 셀 중심이 폴리곤 안이면 주행 가능으로 봅니다(`shapely.contains_xy`).
  - 안쪽/바깥쪽 `cv2.distanceTransform`(DIST_L2, PRECISE)에서 반 셀을 빼고 해상도를 곱한 뒤 ±clip으로 자릅니다.
- **캐시**: `DrivableAreaTargetBuilder` → `drivable_sdf` fp16. cache key는 extent/res/clip/"pdm_drivable_v1"입니다(:86-96).
- **기본 설정**(`configs/default.py:311-317`): weight 0, margin 0, extent (-8, 72, -32, 32), res 0.25 m, clip 10 m → 320×256 격자.
- **Loss**: `plan_map_loss`(:151-181)는 commanded branch의 8 step × 4 모서리에 `relu(margin − sdf)`를 겁니다.
  - 샘플링은 bilinear `grid_sample`(border padding)이고 격자 안 여부 mask를 같이 씁니다(:139-148).
  - 다음 조건을 모두 만족할 때만 부과합니다(:170): 예측이 격자 안, GT가 격자 안, 그리고 GT(사람) 모서리가 도로 안(`s_gt >= 0`).
  - 즉 격자 밖 모서리는 건너뜁니다.

### 공식 DAC와의 차이

- 공식 DAC는 ROADBLOCK/INTERSECTION/DRIVABLE_AREA/CARPARK 유형만 봅니다(`pdm_scorer.py:271-278, 306-314`). 모서리 하나라도 41개 추종 시점 중 한 번이라도 밖에 있으면 실패입니다.
- LANE과 LANE_CONNECTOR는 drivable map에 들어 있지만(`pdm_occupancy_map.py:170-190`) DAC 판정에는 쓰이지 않습니다.
- 따라서 kyungmin의 합집합은 공식 영역보다 넓습니다. pilot에서 영향은 무시할 수준이었습니다(아래 참고).

### feasibility.py (정적 + 동적 필드)

- **Target 격자**: `target_shape`(:81-84)는 bev 50×100에 upsample 4를 적용해 200×400(0.16 m)입니다.
  - 범위는 `map_pc_range`로, **전방 0~32 m, 좌우 ±32 m**입니다(`default.py:77`).
  - 축은 SSR 기준입니다(행 = 전방, 열 = 오른쪽).
  - 문서에 적힌 bilinear 오차 중앙값은 0.64 m 격자에서 0.142 m, 0.16 m 격자에서 0.035 m입니다(:39-48).
- **정적 필드**: `rasterize_bev_sdf`(:89-116). 폴리곤은 plan_map의 `drivable_polygon`을 재사용합니다(:188-192).
- **동적 필드**: `agent_free_sdf`(:119-167)
  - step t마다 `scene.frames[cur+1+t].annotations`의 모든 box를 씁니다. 클래스·ROI 필터가 없고, 나중에 등장한 물체도 포함됩니다.
  - log를 벗어난 물체는 그 시점부터 빠집니다. 프레임이 없으면 이전 step 값을 복사합니다(:141).
  - `fillConvexPoly`에 반올림한 픽셀 좌표를 넣으므로(:161) 셀 절반 정도의 양자화가 생깁니다.
  - 출력은 `agent_sdf_bev` [8, 200, 400] fp16입니다(:197-199).
- **저장 크기(계산값)**: 정적 약 160 KB, 동적 약 1.28 MB / 샘플.
- **샘플링**: `sample_field`(:239-256)는 `padding_mode="border"`입니다. 격자 밖 점은 경계 셀 값을 그대로 받습니다. 격자 밖을 표시하는 mask가 없습니다.
- **Cost**: `softplus((margin − d)/β)·β`를 T × 4 모서리에 대해 합합니다(:264-267).
  - 정적 margin 0.4, 동적 margin 0.2, β 0.2입니다.
  - 동적 필드도 **ego 모서리 4개만** 조회합니다(:330-333).
  - 256개 후보에 softmax(detach된 rank) 가중을 줍니다(:354-367).
  - `feas_cost_grad_to_field`는 기본 False입니다. 켜면 필드 전체를 올려 cost를 낮추는 붕괴가 측정됐습니다(:293-306).

### 실행 시간과 재사용성

- **시간(pilot, CPU 1코어)**: 정적 union과 0.25 m raster 320×256이 token당 중앙값 약 0.54 s입니다(12개 token 측정). 동적 8면 distance transform은 약 6 ms입니다.
- **재사용**: `footprint_corners`, `sample_field`, `cost_from_sdf`, `rasterize_*`는 torch/numpy/cv2/shapely만 쓰는 순수 함수입니다.
  - target builder는 NAVSIM `Scene.map_api`, kyungmin의 `cache_key`, `para_ssr_targets._geometry_local_coords`에 의존합니다.
  - 우리 metric cache에도 같은 폴리곤(`drivable_area_map`, 반경 100 m)과 0.1 s observation이 들어 있습니다. 따라서 Scene 없이 metric cache로 같은 필드를 만들 수 있습니다.

### kyungmin 실험 결과 (csv 값을 그대로 읽음, 단일 seed, 재검증 안 함)

| run | 채점 | 점수 | DAC | NC |
|---|---|---|---|---|
| v2_r34 기준 | EPDMS | 87.80 | 96.07 | — |
| feas_gt ep29 | EPDMS | 88.13 | 96.25 | — |
| feas_gt ep19 | EPDMS | 87.71 | — | — |
| feas_gt ep9 | EPDMS | 85.47 | — | — |
| feas_pred_last | EPDMS | 87.52 | — | — |
| v2_r34 기준 | v1 PDMS | 88.14 | 96.08 | 98.70 |
| feas_pred_last | v1 PDMS | 88.00 | 95.80 | 98.80 |

추가 사실:
- `data/logs/dac_depth.log`: DAC 실패의 침범 깊이 중앙값 0.32 m, 43%가 0.25 m 이하, 첫 이탈 중앙값 3.5 s, 93%가 모서리 1개만 밖입니다. surrogate 오차가 문제 현상과 같은 크기라는 뜻입니다.
- `raw_vs_tracked.py`와 `dac_reachability.py`는 있지만 결과 로그는 찾지 못했습니다.

## (b) 32-B circle collision penalty

- **Ego**: 원 3개, 반지름 `EGO_R = hypot(1.1485, 2.588/3) = 1.436 m`, 중심은 box 중심에서 heading 방향 −1.725 / 0 / +1.725 m입니다(`common.py:32-35`). MARGIN 0.3 m(:36).
- **물체**: `box_to_circles`(`common.py:49-69`)로 n = ceil(L/W) ≤ 5개의 원을 긴 축을 따라 놓고, r = hypot(W/2, L/(2n))입니다.
- **GT 물체 선택**(`b0_gt.py:38-76`)
  - `sf_common.gt_objects(mc)`, 즉 metric-cache occupancy(모든 유형, 나중에 등장한 물체 포함, red-light pseudo-object 제외)를 8 knot(0.5~4.0 s)에서 샘플링합니다.
  - 8 knot 중 한 번이라도 {원점, E0 pose_t, 사람 pose_t} 중 하나와 40 m 이내인 물체만 남깁니다. 즉 **궤적에 따라 달라지는 집합**입니다.
  - 가까운 순으로 최대 A_MAX개를 남깁니다. 코드 값은 64이고 docstring에는 48로 적혀 있습니다.
  - token의 22.0%가 이 상한에 걸립니다. 평균 80.0개 중 35.4개를 남깁니다.
- **Penalty**(`b1_train.py:114-138`): `Σ relu(r_e + r_a + 0.3 − d)²`를 8 knot의 raw 계획 궤적에서 계산합니다.
  - 물체 중심이 ego 중심보다 −1.294 m 이상 앞에 있을 때만 셉니다(heading은 detach).
  - 사람 궤적도 위반한 (물체, 시점) 쌍은 제외합니다(:163-167). λ는 10 epoch에 걸쳐 올립니다(:40, 220-222).
- **기존 점검값**(`data/losscheck.json`): 원래 궤적에서 AUC 0.977, 사람 위반 제외 시 0.960. pilot에서 같은 값을 재현했습니다.

## (c) safety filter 기하 (`sf_common.py`)

- **Path**(:57-112): 원점과 8개 pose를 잇는 polyline입니다. `ref_poses_time`(:108-112)이 채점기와 동일하게 시간에 선형 보간해 0.1 s × 41점을 만들고 heading은 unwrap합니다.
- **Footprint**: `ego_corners`(:115-121)는 Pacifica box에 margin을 길이·폭 모두에 더합니다.
- **겹침 판정**: `sat_overlap`(:138-146)은 사각형 두 개의 4개 축 SAT이고, 접촉도 겹침으로 봅니다. `overlap_timeline`(:175-188)은 중심 거리로 먼저 거른 뒤 SAT를 돌립니다.
- **blocked_grid**(:191-212): 시간 0.1 s × 호 길이 0.1 m 격자이며, 피할 수 없는 쌍은 무시합니다.
- **GT 물체**: `gt_objects`(:408-431)는 metric cache 51개 occupancy map(0~5 s) 중 41개를 씁니다.
- **Closed-loop 변형**: 공식 LQR + bicycle `simulate`(`run_filter.py:165-170`)와 팽창 재시도(EXTRAS, :266)를 씁니다.
- boolean 판정이라 미분할 수 없습니다.

## GT 범위 사실 (사용자 피드백 2번 관련)

### `_compute_agent_targets` (`/home/external-user/yongjae/SSR/navsim/agents/para_ssr/para_ssr_targets.py:289-351`)

- 현재 프레임 annotation만 씁니다.
- 클래스는 DET_CLASS_NAMES 7종입니다(:71-80).
- ROI는 `x_right ∈ [−32, 32]`, `y_forward ∈ [0, 32]`, FOV ±80°입니다(:128-, `default.py:70, 188`).
- 가까운 순으로 max_agents 100개까지 남깁니다.
- 미래 정보는 `_track_future`(:401-448)가 주며, **중심 offset만** 있습니다. 미래 heading이나 크기는 없고, 트랙이 한 번 끊기면 이후 mask는 0입니다(:431-432).
- 나중에 등장하는 물체, 뒤쪽 물체, 32 m 너머 물체는 들어오지 않습니다.

### 원래 NC 실패 279건의 원인 물체 (`perception_reliability/pdm_attr/table.parquet`)

| 구분 | 건수 | 비율 |
|---|---|---|
| t0에 존재하지 않음 | 11 | 3.9% |
| t0에 있으나 ROI 밖 | 37 | 13.3% |
| └ 전방 32 m 초과 | 29 | |
| └ ego 뒤쪽 | 8 | |
| ROI 안 | 231 | 82.8% |

- 따라서 `_compute_agent_targets` 기준 GT는 **원인 물체의 최소 17.2%를 놓칩니다.**
- 충돌 시점에 ego가 32 m보다 앞에 있던 경우가 21건입니다.
- 정적 물체 충돌이 40건이며, 공식 NC에서 0.5로 계산됩니다.
- 충돌 유형: STOPPED_TRACK 153, ACTIVE_FRONT 121, LATERAL 5.

### Metric cache observation (`metric_cache_processor.py:115-238`)

- 모든 tracked object를 2 Hz로 읽어 10 Hz로 보간합니다. 나중에 등장한 물체도 포함됩니다.
- 한 번만 관측된 물체는 전 시점에 복제됩니다(:192-193).
- `update_detections_tracks`에는 반경 필터가 없습니다(`pdm_observation.py:255-279`).

### 공식 NC 규칙 (`pdm_scorer.py:323-387`, `pdm_scorer_utils.py:17-`)

- 추종된 41개 상태에서 intersects로 충돌을 찾습니다.
- 과실 충돌(at-fault)로 치는 경우: ACTIVE_FRONT, STOPPED_TRACK, 그리고 ego가 여러 차선에 걸치거나 도로 밖일 때의 LATERAL.
- ego 정지(속도 ≤ 0.05 m/s) 충돌과 뒤쪽 충돌은 과실이 아닙니다. 한 번 과실 아님으로 판정된 트랙은 이후 무시됩니다.

### 사람 궤적과 LQR 편차

- 사람 궤적의 공식 NC/DAC 실패는 E 실험 navtrain 무작위 9,000개에서 0건입니다(`E_train_split_feasibility/token_table.parquet`). kyungmin의 v2 라벨에서도 human 행은 NC=DAC=1이지만, EP도 전부 1이라 따로 확인은 못 했습니다.
- LQR 편차: 원래 궤적 `lqr_max_dev_m` 중앙값 0.589 m, p90 1.011, p99 1.45 m. safety filter `lqr_dev`는 중앙값 0.61, p99 3.17 m입니다.

## (d) 검증 프로토콜 제안

### 추가 채점 없이 쓸 수 있는 풀

| 풀 | 규모 | 위치 | 비고 |
|---|---|---|---|
| **P0** 실제 student 초안 | 12,146 | `pdm_attr/table.parquet` | NC/DAC, `nc_time_idx`, `nc_track`, `nc_ref_overlap_idx`, `lqr_max_dev_m` 포함 |
| **P1** 32-B 학습 수정 궤적 | 24 set × 12,146 | `B_read_vs_generate/results/*.parquet`, `sets/*.npz` | inner-val(`iv_*`)과 `*_sub` set 별도 |
| **P1** D_teacher_info_filter | — | `D_teacher_info_filter/variant_table.parquet`, `modified_trajectories.npz` | 내용은 보지 않음 |
| **P2** 규칙 기반 속도 수정 | 수정 궤적 29,288 (23 variant) | `safety_filter/variant_table.parquet`, `modified_trajectories.npz` | |
| **P3** 스트레스 풀 | 12,146 × 256 = 3.1M | `kyungmin/SSR-v2/data/planning_vb/navtest_refined_scores.npz`, 궤적은 `navtest_refined_r34.npz` | NAVSIM v2.2 채점기·별도 metric cache. v1 수치와 섞지 말 것. kyungmin 자체 검증에서 최대 차이 0.159로 "불일치" 판정이 한 번 있음 |
| navtrain | 9,000 (student, 사람, 교정 5종) | E 실험 | metric cache도 함께 있음 |

### 평가할 surrogate 조합

- 기하: circle chain / 정확한 SAT(참고용) / 모서리 SDF / 사각형-사각형 signed distance
- 물체 집합: metric cache 전체 / 32-B 식 40 m·64개 / `_compute_agent_targets` ROI / 모델이 인지한 물체
- 시간: 8 knot / 0.1 s 보간
- 궤적: raw / 공식 simulate(천장값) / 미분 가능한 추종 근사
- 과실 판정: 없음 / 뒤쪽 필터 / 전체 규칙 복제
- margin: 0 / 0.2 / 0.3 / 0.5 m

### 지표

1. **궤적 단위**: ROC-AUC, PR-AUC, 학습에 쓸 margin에서의 recall/precision/FPR, FPR 1%·5%에서의 recall.
   - 층화 기준: 풀, 실패 유형(agent 0 vs 정적 0.5, 충돌 유형, DAC), 원인 물체 출처(ROI 안 / ROI 밖 / 나중 등장), 시점(0~2 s / 2~4 s), ego x > 32 m 여부, LQR 편차 구간.
2. **쌍 단위(교정기에 가장 중요)**: 같은 token의 (초안, 수정) 쌍에서 다음을 봅니다.
   - surrogate가 "해소"라고 할 때 공식도 해소일 확률
   - surrogate가 "안전"이라고 할 때 공식은 새 실패인 비율
   - Δsurrogate와 Δ공식의 일치도
3. **시점과 물체 일치**: 첫 위반 시점 차이 |Δt| ≤ 0.5 s, 그리고 위반 물체가 `nc_track`과 같은지.
4. **사람 궤적 오경보율**: 공식으로는 안전한 사람 궤적을 surrogate가 얼마나 위반으로 잡는지.
5. **신뢰구간**: log 단위 bootstrap(136 log), 건수는 Clopper-Pearson. 판정 기준은 결과를 보기 전에 정해 둡니다.

## Pilot 결과 (CPU, nice 10, 약 5.5분, 읽기 전용)

### Part A: 32-B circle penalty, 12,146 token 전체, 공식 NC<1 기준 (73 s)

| set | AUC | recall(pen>0) | precision | FPR | 새 실패 중 탐지 | 해소 token 중 pen=0 | 시점 \|Δt\|≤0.5 s |
|---|---|---|---|---|---|---|---|
| orig | 0.977 (제외 0.960) | 0.971 (제외 0.932) | 0.19 (제외 0.27) | 0.096 (제외 0.060) | – | – | 0.79 |
| D1h_0 | 0.940 | 0.909 | 0.23 | 0.096 | 0.82 / 186 | 0.39 / 91 | 0.80 |
| D2h_0.1 | 0.932 | 0.901 | 0.22 | 0.091 | 0.83 / 181 | 0.42 / 128 | 0.72 |
| D2hp_0.1 | 0.902 | 0.844 | 0.23 | 0.087 | 0.74 / 207 | 0.44 | 0.74 |
| D2e_0.01 | 0.925 | 0.892 | 0.44 | 0.115 | 0.88 / 969 | 0.62 | 0.72 |
| **D2e_0.1** | **0.593** | **0.316** | 0.37 | 0.143 | **0.30** / 2375 | 0.66 | 0.51 |
| D2hg_0.1 | 0.766 | 0.628 | 0.28 | 0.128 | 0.58 / 768 | 0.49 | 0.53 |

("제외"는 사람 궤적도 위반한 쌍을 뺀 값입니다.)

- 원래 경로에서 크게 벗어난 set(D2e_0.1, D2hg_0.1)에서 recall이 무너집니다.
- 같은 D2e_0.1을 metric cache 전체 물체로 SAT 판정하면 recall이 0.675입니다(Part B). 물체 집합이 원래·사람 경로 기준으로 잘려 있는 것과 맞는 결과이지만, 사례별로 확인하지는 않았습니다.

### Part B: 600 token(무작위 300 + 실패 포함 300), 5,658 궤적, 모든 metric-cache 물체 (244 s, 4 worker)

**NC**
- raw 0.1 s SAT(과실 판정 없음): orig recall 0.75, precision 0.83. 제가 구한 첫 겹침 시점은 표의 `nc_ref_overlap_idx`와 20건 모두 일치했습니다. 279건 전체로 보면 raw 겹침이 있는 경우는 67.7%입니다.
- 32-B decoder 궤적: recall 0.62, precision 0.64, FPR 0.031.
- 8 knot SAT: recall 0.41로 떨어집니다.
- 모서리만 보는 방식(kyungmin 동적 필드와 같음): recall 0.32이고, 0.2 m margin을 줘도 0.41입니다.
- sf `gt_m3_ref`: 공식 실패 5건 중 raw SAT가 잡은 것은 0건입니다.
- sf `gt_m3`: 공식 실패 0건인데 raw SAT가 17.5%를 위반으로 잡습니다. 과실이 아닌 접촉 때문입니다.
- 시점 일치(n=210, 물체 일치는 확인 안 함): 중앙값 −0.2 s, |Δt| ≤ 0.5 s가 54%입니다.
- 참고로 제가 만든 `sat01_f` 과실 근사는 마지막 시점(k=40)에서 속도를 0으로 두는 버그가 있어 과소 추정입니다. 결과 해석에는 `sat01`을 기준으로 삼았습니다.

**DAC**
- 공식 유형으로 raw 0.1 s 모서리 판정: orig recall 0.744, precision 0.762. decoder 궤적 recall 0.727, precision 0.710, FPR 0.082. D2e_0.1의 FPR은 0.405입니다.
- 8 knot 판정 0.715, 0.25 m SDF(margin 0) 0.696으로 recall이 약간 줄어듭니다.
- 공식 유형과 전체 합집합의 판정이 다른 궤적은 **1/5,658**입니다. 50 m 이내에서 추가 영역이 1 m²를 넘는 token은 4.2%(중앙값 3.6 m²)입니다.
- 사람 궤적(공식적으로 안전)에 대한 오경보: raw 판정 1.3%, SDF 2.0%. NC SAT는 0%.

**격자 범위**
- 원래 궤적의 25.5%는 모서리가 x > 32 m까지 갑니다. 72 m를 넘는 경우는 없습니다.

## 조심스럽게 적어야 할 기존 작업과의 관계

- **kyungmin**: 모서리 SDF라는 기하 요소는 같습니다. 다만 그의 실험은 256 anchor 후보 전체에 cost를 거는 anchor planner 학습이고, EPDMS 차이는 ep29에서 +0.33이며 단일 seed입니다. 사후 교정기의 근거로 옮겨 쓸 수는 없습니다.
- **32-B**: 해소 128/279, 새 충돌 181, 전체 충돌 332로 원래 279보다 많습니다. surrogate 기준으로 궤적을 바꾸면 surrogate가 보지 못하는 새 실패가 생길 수 있다는 사례입니다. 쌍 단위 검증이 필요한 이유입니다.
- byounggun Stage 1/2와 kyungmin readout 코드는 이번에 보지 않았습니다.

## 파일

모두 scratchpad(`/tmp/claude-1001/-home-external-user/9262a8d2-fa20-46d7-ba9a-92d7d13b8ceb/scratchpad/`)에 있습니다. 공유 파일은 수정하지 않았습니다.
- pilot_surrogate.py
- pilot_analyze.py
- pilot_partA.csv
- pilot_partB.parquet
- pilot_partB_labeled.parquet