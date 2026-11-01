## 미래 객체 GT 조사 보고서 (읽기 전용: 파일 수정 없음, CPU 전용)

### 요약
- **`_compute_agent_targets`는 현재 프레임(t=0)의 객체만 추적합니다.** 앞쪽 ROI(전방 0–32 m, 좌우 ±32 m, ±80°) 안에 중심이 있는 객체만 남기고, 미래는 중심 좌표만 4 s까지 기록합니다. navtrain 50 token에서 60 m 안에 들어오는 track의 21.2%만 남았습니다. [확인]
- **공식 metric cache의 객체도 같은 raw log annotation에서 만들어집니다.** 범위는 t=0–5 s의 모든 track, 모든 class, 360°입니다. log keyframe만으로 metric cache의 10 Hz polygon을 **오차 0**으로 다시 만들 수 있었습니다. [확인]
- **navtest NC 원인 객체 279개 기준:**
  - 현재-ROI tracker는 18.6%(52개)를 놓칩니다.
  - 360°·0–5 s 전체 track builder는 반경 60 m 이상에서 0개를 놓칩니다.
  - TTC 원인 781개는 현재-ROI tracker로 37.6%만 덮입니다. `fut_ts=8`(4 s)이 TTC가 보는 구간(최대 4.9 s)보다 짧기 때문입니다. [확인]
- **"없음"의 의미는 공식 채점 규칙과 맞춰야 합니다.** 특이 사례로, 한 번만 annotation된 track(singleton)은 metric cache가 0–5 s 전체에 고정 배치합니다. NC 원인의 2.5%가 이 경우입니다. [확인]

---

### (a) `_compute_agent_targets` / `_track_future` / `detection_box_in_roi`
파일: `/home/external-user/yongjae/SSR/navsim/agents/para_ssr/para_ssr_targets.py`

**객체 선택 기준**
- **현재 프레임만 봅니다.** `scene.frames[cur_idx].annotations`를 쓰고, `cur_idx = num_history_frames-1 = 3`입니다 (L294, L272).
- **class 필터는 실제로 아무것도 거르지 않습니다.** DET 7 class(L71-79, L314)는 log에 나오는 이름 7종과 같습니다. 50개 log에서 `ego`라는 이름은 한 번도 나오지 않았습니다.
- **ROI 판정은 박스 중심 하나로만 합니다** (L128-147).
  - 범위: `pc_range` x_right ±32, y_forward 0..32 (`configs/default.py` L70), |bearing| ≤ 80° (default.py L188, yaml L68).
  - z와 박스 모서리는 보지 않습니다. 따라서 박스는 ROI에 걸쳐 있어도 중심이 밖이면 제외됩니다. 예: ego 바로 옆 차선 차량(bearing이 80°를 넘는 경우), 뒤쪽 차량.
- **가까운 순으로 최대 100개까지 남깁니다** (L321-329, `max_agents=100` default.py L183). 표본 50 token에서는 이 상한에 한 번도 걸리지 않았습니다.

**저장되는 미래 정보**
- t=0 박스 9개 값만 저장합니다: x_r, y_f, z, w, l, h, yaw, vx, vy (L333-337).
- 미래는 **중심 좌표만** 저장합니다. 현재 ego 프레임 기준의 step별 offset이며, frame 3+1 … 3+8, 즉 0.5 s 간격으로 **4 s까지**입니다 (L401-448, `fut_ts=8` default.py L167).
- 미래 시점의 heading, 크기, 속도는 저장하지 않습니다.

**`_track_future`의 중단 규칙**
- track이 처음 빠지는 프레임에서 기록을 멈춥니다 (L431-432).
- 실측으로는 annotation track 안에 끊긴 구간이 없었습니다(0/5,847). 따라서 기록이 짧아지는 경우는 track이 끝났을 때뿐입니다.
- 남긴 track 852개 중 750개(88%)는 8 step을 다 채웠고, 13개는 0 step이었습니다.

**빠지는 것**
- t=0에 ROI 밖에 있던 객체: 뒤, 옆, 32 m 밖, 시야각 밖.
- t=0 이후에 처음 나타나는 객체 전부.
- 4 s 이후의 미래.
- 미래의 박스 방향과 크기.

### (b) raw NAVSIM log (`trainval_navsim_logs/trainval/*.pkl`, key `anns`)

**필드**

| 필드 | shape / 의미 |
|---|---|
| `gt_boxes` | (N, 7) = (x_fwd, y_left, z, L, W, H, yaw). 각 프레임의 ego rear-axle 기준 (`lidar2ego`는 항등) |
| `gt_names` | (N,) |
| `gt_velocity_3d` | (N, 3). 각 프레임의 ego 프레임 기준 |
| `track_tokens` | 프레임이 바뀌어도 유지되는 식별자 |
| `instance_tokens` | 프레임마다 바뀜. 연속한 두 프레임의 공통 track 61개에서 일치 0개 |

- 로딩 코드: `navsim/common/dataclasses.py` L382-392.
- scene 구성: 4+10 프레임, 0.5 s 간격 (`scene_filter/navtrain.yaml` L3-5).

**navtrain 50 token 측정** (log를 seed 0으로 섞은 뒤 log당 1 token)

*annotation 범위*
- 프레임 간격은 0.499–0.501 s입니다. 미래 프레임 500개 중 빈 프레임은 0개입니다.
- 프레임당 박스 수: 중앙값 65, p5 7, 최대 362.
- 프레임당 가장 먼 박스까지의 거리: 중앙값 77.3 m, p95 80.4 m, 최대 83.1 m.
- 박스의 47.9%가 ego 뒤쪽입니다. 즉 annotation은 **360°, 반경 약 80 m**를 덮습니다.

*track 구성*
- 끊긴 구간이 있는 track: 0개.
- singleton track: 544/5,847 (9.3%). 창의 첫 프레임에만 있는 것 133개, 마지막 프레임에만 있는 것 217개, 중간 한 프레임에만 있는 것 194개.

*t..t+4 s 동안 60 m 안에 들어오는 서로 다른 track* (중심 기준, t0 ego 프레임)
- 합계 4,023개 (token당 평균 80.5, 중앙값 62.5, 최대 344).
- `_compute_agent_targets`가 남긴 것: **852개 (21.2%)**, token당 평균 17.0.
- t=0에 있었지만 제외된 것: 뒤 1,577, 옆 298, 32 m 밖 452, 시야각 밖 56.
- **t=0 이후에만 나타나는 것: 788개 (19.6%).**
  - 처음 나타날 때의 거리: p5/p50/p95 = 14.9/39.5/59.3 m. 51%는 40 m 안에서 처음 나타납니다.
  - 처음 나타날 때의 위치: ROI 안 196, 뒤 258, 32 m 밖 200, 옆 126, 시야각 밖 8.
  - class: generic 300, vehicle 206, pedestrian 178, cone 82 등.
  - ID switch(같은 객체의 track 번호가 바뀐 경우)로 설명되는 것은 939개 중 16개(2 m 이내)~27개(4 m 이내)뿐입니다. 대부분은 실제로 새로 annotation되기 시작한 객체입니다.

*경로 근처 track*
- 기준(대리 지표): 같은 시각 GT ego와 10 m 이내이거나, GT 경로에서 4 m 이내.
- 해당 track 392개 중 **169개(43%)가 제외되었습니다.** 사유: 뒤 118, 나중에 등장 23, 32 m 밖 14, 시야각 밖 14.
- 4 m 경로 corridor만 보면 84개 중 38개가 제외되었습니다.

*token당 track 수* (0–5 s 중 어느 시점이든 t0 원점에서 R 이내)
- R = 60 m: 평균 84.4, p95 179, 최대 373.
- R = 80 m: 평균 107.6, p95 247, 최대 447.
- GT ego의 4 s 이동거리: p95 42.1 m, 최대 48.2 m.

### (c) metric cache의 미래 객체와 공식 NC/TTC

**metric cache가 객체를 만드는 방식**

1. **출처는 같은 log `anns`입니다.** `NavSimScenario.get_tracked_objects_at_iteration`이 `scene.frames[3+k].annotations`를 읽습니다 (`navsim_scenario.py` L195-212, `navsim_scenario_utils.py` L47-118).
2. **2 Hz keyframe 11개(0..5 s)를 10 Hz 51장으로 선형 보간합니다** (`metric_cache_processor.py` L115-238).
   - x, y, heading(unwrap), vx, vy를 보간합니다 (`metric_caching_utils.py`).
   - 박스 L/W는 **처음 등장했을 때의 값으로 고정**합니다 (L201-207).
   - 등장 이전과 사라진 이후에는 객체가 없습니다 (`interpolate` → None).
   - **singleton track은 51장 전부에 같은 자세로 배치됩니다** (L192-193).
3. **반경 필터가 없습니다.** `update_detections_tracks`(`pdm_observation.py` L255-279)가 모든 class를 그대로 넣습니다. obs[0]에서 가장 먼 객체는 132 m였습니다.

**검증** (`mc_check.py`, `rebuild_check.py`)
- navtest 8 token:
  - `unique_objects`는 t0..t+5 s log track의 합집합과 정확히 같았습니다.
  - 여러 프레임에 걸친 track 621/621개가 정확히 [5·first_k, 5·last_k] 구간에만 있었습니다.
  - singleton 48/48개가 51장 전부에 있었습니다.
- 별도 navtest 10 token에서 log keyframe으로 다시 만든 결과:
  - (track, 시각) 40,976쌍의 **모서리 최대 오차 0.0 m**.
  - 객체가 없어야 하는 16,195쌍도 모두 일치.
- track별 L/W는 프레임마다 조금씩 다릅니다: 중앙값 0.05 m, p95 0.39 m, 최대 1.28 m.

**공식 NC** (`pdm_scorer.py` L323-384)
- 대상: LQR로 추종한 ego footprint 41개(0.1 s, 0..4 s). Pacifica 차체 5.176 × 2.297 m, rear axle에서 중심까지 1.461 m.
- 각 시각의 occupancy map과 교차 여부를 봅니다. 객체는 log를 그대로 재생하며 ego에 반응하지 않습니다.
- at-fault로 치는 경우: ACTIVE_FRONT, STOPPED_TRACK, 그리고 ego가 여러 차선이나 비주행영역에 있을 때의 lateral.
- 충돌 분류에는 `unique_objects[token]`, 즉 **처음 등장했을 때의** heading과 속도를 씁니다 (`pdm_scorer_utils.py` L17-).
- 감점: agent와 충돌하면 0, 정적 객체와 충돌하면 0.5.

**공식 TTC** (L458-548)
- ego를 등속으로 +0/0.3/0.6/0.9 s 투영해 map[time_idx+Δ]와 비교합니다. 따라서 **최대 4.9 s**까지의 객체가 필요합니다.

**시간 해상도**
- NC 실패 279개 중 276개(98.9%)는 0.5 s 배수 시각(idx 5..40)에도 at-fault 접촉이 있습니다.
- 3개는 keyframe 사이에서만 접촉합니다.
- 접촉 지속 시간 p10/p25/p50 = 1/3/6 step(0.1 s 단위).

### (d) navtest 원인 객체 커버리지 (`cause_cov.py`, 원인 track을 log에서 직접 추적)

| | NC (N=279) | TTC (N=781) |
|---|---|---|
| 현재-ROI tracker가 t0에 남김 | 229 (82.1%) | 571 (73.1%) |
| 남기고 미래 중심이 사건 keyframe까지 있음 | **227 (81.4%)** | **294 (37.6%)** |
| 놓침 | **52 (18.6%)**: 32 m 밖 26, t0에 없음 15, 뒤 8, 옆 1, t0에만 있는 singleton 2 | 487: 제외 210 (32 m 밖 120, 새 객체 51, 뒤 31, 시야각 밖 4, 옆 4), 4 s 이후 사건 272, singleton 7 |
| 360°·0–5 s builder 놓침, R = 32 / 40 / 50 / 60 / 80 m | 23 / 5 / 1 / **0** / 0 | 120 / 47 / 13 / **0** / 0 |
| t0 원점에서 원인 객체까지 최소거리: p50 / p95 / 최대 | 18.1 / 36.2 / 50.8 m | 20.8 / 41.9 / 57.0 m |

- **t0에 없던 NC 원인 15개**가 처음 나타난 keyframe: 1,1,2,2,3,5,5,6,6,6,7,7,8,8,10. 처음 나타날 때 ROI 안 10개, 32 m 밖 5개.
- **NC 원인 중 singleton 7개(2.5%):**
  - t0에만 있는 vehicle 3개.
  - t=3.0–5.0 s에 한 번만 annotation된 객체 4개(generic ×2, barrier, bicycle). metric cache가 이들을 0–5 s 전체에 배치합니다.
- **report 31(`pdm_attr`) 분류와의 관계:**
  - `nc_vis=roi_no_ann`(token `bf0a29ccead65750`)은 annotation이 빠진 경우가 아닙니다. k=8에만 있는 singleton이 t0에 배치된 것입니다.
  - 표의 `new` 11개와 log 기준 15개의 차이 4개도 이 규칙으로 설명됩니다. 따라서 이 두 분류의 해석은 조심스럽게 고쳐 적는 편이 맞습니다.
- **사건 시점 원인 객체가 전방 32 m BEV 밖에 있는 NC: 41/279.**
  - navtest expert 4 s 경로가 32 m를 넘는 비율은 17.3%입니다.
  - 따라서 50×100 raster 기반 SDF만 쓰면 이 부분은 반영되지 않습니다.

### 제안: 'all future objects' builder (신규 target. `_compute_agent_targets`는 수정하지 않음)

1. **출처:** raw log `anns` 프레임 cur..cur+10 (0..5 s, 0.5 s keyframe 11개).
   - 모든 class, 360°, 식별자는 `track_tokens`.
   - 좌표는 `_track_future`와 같은 방식으로 t0 rear-axle 기준 ego 프레임으로 변환합니다.
   - 공식 metric cache와 정의가 같습니다 (위에서 오차 0 확인).
2. **반경:** 어느 keyframe에서든 중심이 t0 원점에서 **80 m** 이내인 track.
   - 60 m가 NC/TTC 원인을 전부 덮는 최소값입니다. 80 m는 ×1.4 초안 여유분입니다.
   - token당 평균 108개, 최대 447개입니다. ragged로 저장하거나 A_max=512로 padding합니다.
3. **token당 저장 항목:**
   - keyframe `[A, 11]`: x, y, heading(t0 프레임, unwrap), vx, vy, present.
   - 첫 등장 시점의 L, W, heading, 속도 (공식 크기·충돌 분류와 맞추기 위함).
   - class, is_agent(NC 감점 0) / static(0.5), first_k, last_k, singleton 여부.
4. **시간 격자:**
   - 0.1 s(51 step) 값은 keyframe을 선형 보간해 즉석에서 만듭니다. metric cache와 같은 규칙입니다.
   - loss에서는 초안도 0.1 s로 보간해 비교하는 것을 권합니다. keyframe 사이에서만 접촉하는 NC가 3/279 있고, 접촉의 10% 이상이 0.1 s 한 step짜리이기 때문입니다.
   - 범위는 0..5.0 s로 둡니다(TTC 4.9 s 포함).
5. **validity 의미 (3단계):**
   - (i) **observed.**
   - (ii) **공식적으로 없음:** 등장 이전이나 사라진 이후. 공식 채점은 이를 '없음'으로 봅니다. 다만 등장 이전 객체의 51%가 40 m 이내에서 처음 나타나므로 물리적으로는 '미확인'입니다. 별도 플래그로 구분합니다.
   - (iii) **unknown:** 각 시각 GT ego에서 약 75 m 밖(annotation 한계 77–83 m), 그리고 5.0 s 이후.
   - singleton 처리는 두 가지 중 하나를 골라야 합니다.
     - A: 공식 규칙대로 51 step 전체에 배치 (채점과 일치).
     - B: 해당 keyframe에서만 유효하고 나머지 시각은 unknown (물리적 해석).
6. **검증:**
   - navtest metric cache와 polygon을 비교합니다 (지금 10 token에서 오차 0 확인).
   - navtrain은 실험 E의 metric cache로 같은 검사를 합니다: `/home/external-user/yongjae/SSR/report/cause_and_correction_tests/E_train_split_feasibility/metric_cache` (9,000 token, `e2_cache.py`로 repo 코드를 써서 생성).

### 기존 작업과의 관계 (신중한 기술)
- **32-B (`B_read_vs_generate/b1_train.py` L12-17):**
  - metric cache의 GT 미래 박스를 원 여러 개로 근사해 썼습니다. 조건은 40 m 이내, ego 뒤쪽이 아닌 쌍, 사람 궤적도 겹치는 쌍 제외입니다.
  - navtest CV 안에서만 사용했습니다.
  - 40 m 기준의 누락률은 제 기준(t0 원점에서의 거리)으로 NC 1.8%, TTC 6.0%입니다. b0_gt는 원점·E0·human 중 가장 가까운 거리로 판정하므로 실제 누락은 이보다 적을 수 있습니다. 직접 측정하지는 않았습니다 [추측].
- **kyungmin의 `SSR-v2/.../feasibility.py` `agent_free_sdf` (L119-167):**
  - 각 미래 keyframe의 박스를 **전부** 씁니다. 따라서 나중에 등장하는 객체와 360° 객체도 포함됩니다.
  - 다만 전방 32 m raster 안에서만 그립니다. raster 밖은 `grid_sample`의 border padding으로 경계 값이 복사됩니다 (L239-256).
  - 위의 "사건 시점 32 m 밖 41건"은 이 방식에서 반영되지 않는 범위와 겹칠 수 있습니다 [추측].

### 사용한 스크립트 (scratchpad, 읽기 전용)
`/tmp/claude-1001/-home-external-user/9262a8d2-fa20-46d7-ba9a-92d7d13b8ceb/scratchpad/`:
- `fut_obj_audit2.py`
- `frag_check.py`
- `count_r.py`
- `mc_check.py`
- `rebuild_check.py`
- `cause_cov.py` → 결과 `cause_cov.csv`