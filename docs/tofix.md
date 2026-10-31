# To-Fix: `exact2` (NAVSIM v2.2 EPDMS 학습 정답 테이블) 내 v1 데이터 혼입 문제 및 조치 계획

작성일: 2026-10-28

---

## 1. 문제 요약 (Executive Summary)

현재 서버의 EPDMS 학습 정답용 기하 테이블 캐시인 [`/workspace/navsim_workspace/corridor_cache/exact2/{navtrain_train,navtrain_val}`](file:///workspace/navsim_workspace/corridor_cache/exact2/navtrain_train)는 순수 NAVSIM v2.2 `metric_cache`에서 생성된 것이 아니라, [`scripts/km_corridor/build_gt.sh`](file:///workspace/byounggun/SSR/scripts/km_corridor/build_gt.sh#L16-L20)가 호출한 우회 스크립트 [`tools/corridor/build_exact2_from_v1.py`](file:///workspace/byounggun/SSR/tools/corridor/build_exact2_from_v1.py#L1-L36)로 생성되었습니다.

이 과정에서 전체 **12개 테이블 중 8개(`SHARED_TABLES`)가 v1 `metric_cache` 기반 [`exact`](file:///workspace/navsim_workspace/corridor_cache/exact/navtrain_train) 디렉토리로 심볼릭 링크(symlink)** 처리되었으며, 그중 동적 객체 테이블(`agent_rows`)과 PDM-Closed 기준 궤적 테이블(`pdm`), 그리고 자체 재구성한 적색 신호등 테이블(`red_seg`, `red_t`)이 실제 NAVSIM v2.2 `metric_cache`와 불일치합니다.

---

## 2. 원인 상세 분석

### 2.1 원래 설계 vs 실제 생성 경로
- **원래 설계** ([`exact_score.py:1-4`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/exact_score.py#L1-L4), [`build_exact.py:35-38`](file:///workspace/byounggun/SSR/tools/corridor/build_exact.py#L35-L38)):
  - `exact` (v1): v1 `metric_cache` + v1 `navsim` 패키지로 생성
  - `exact2` (v2.2): **v2.2 `metric_cache` + `navsim_v2` 패키지(`--v2 --mc-root <v2_cache> --cl-full`)**로 생성
  - 코드 주석에도 *"v1 tables from the v1 metric cache, v2 tables from the v2 metric cache — **the two caches differ in observation and PDM-Closed**"*라고 명시되어 있음.
- **실제 생성 경로** ([`build_exact2_from_v1.py:35-36, 194-200`](file:///workspace/byounggun/SSR/tools/corridor/build_exact2_from_v1.py#L35-L36)):
  - `navtrain` 전체에 대한 v2.2 `metric_cache`가 디스크에 준비되어 있지 않아, v1 `exact` 테이블 8개를 그대로 심볼릭 링크하고 나머지 4개(`hum`, `past`, `red_seg`, `red_t`)만 `navsim_logs`와 v1 `drivable_area_map`으로부터 재구성함:
    ```python
    SHARED_TABLES = ("agent_rows", "drv_seg", "lane_seg", "int_seg", "route_seg", "cl", "pdm", "ego")
    ```

---

## 3. 테이블별 실제 NAVSIM v2.2 `metric_cache` 대조 결과 (50개 토큰 실측)

v1 [`build_exact.py`](file:///workspace/byounggun/SSR/tools/corridor/build_exact.py) 출력(현재 `exact2`가 참조하는 값)과 실제 NAVSIM v2.2 `metric_cache`([`/workspace/yongjae/perception_eval/epdms/metric_cache_v22_navtest`](file:///workspace/yongjae/perception_eval/epdms/metric_cache_v22_navtest))에서 추출한 값을 테이블별로 대조한 결과입니다.

| `exact2` 테이블 | 현재 디스크 상태 | 실제 v2.2와 불일치율 | 불일치 원인 및 세부 내용 |
| :--- | :---: | :---: | :--- |
| **`agent_rows`** | **v1 Symlink** | **96.0% (48/50 토큰)** | v1은 미래 `5.0s` (`num_future_frames=10`, 50 step) 관측이고 v2.2는 `4.0s` (`num_future_frames=8`, 40 step) 관측이며, `observation.unique_objects` 구성 자체가 다름 |
| **`pdm`** | **v1 Symlink** | **26.0% (13/50 토큰)** | 관측 객체 및 신호등 차이가 IDM/PDM-Closed 시뮬레이션에 반영되어 PDM-Closed 궤적 끝점이 최대 **5~7m 이상** 달라짐 |
| **`red_seg` / `red_t`** | v1+log 재구성 | **32.0% (16/50 토큰)** | [`build_exact2_from_v1.py:97-126`](file:///workspace/byounggun/SSR/tools/corridor/build_exact2_from_v1.py#L97-L126)의 자체 재구성 로직이 실제 v2.2 `observation` 내 `red_light` 폴리곤과 32% 토큰에서 불일치 |
| `drv_seg` | v1 Symlink | **0.0% (완전 일치)** | 도로(`ROADBLOCK`, `INTERSECTION`, `DRIVABLE_AREA`, `CARPARK_AREA`) 폴리곤은 v1과 v2.2가 동일 |
| `lane_seg` | v1 Symlink | **0.0% (완전 일치)** | 차선(`LANE`, `LANE_CONNECTOR`) 폴리곤 동일 |
| `int_seg` | v1 Symlink | **0.0% (완전 일치)** | 교차로(`INTERSECTION`) 폴리곤 동일 |
| `route_seg` | v1 Symlink | **0.0% (완전 일치)** | 주행 경로(`route_lane_ids`) 폴리곤 동일 |
| `cl` | v1 Symlink | **0.0% (완전 일치)** | `exact` 생성 시 `--cl-full` 옵션이 적용되어 전체 중심선이 v2.2와 동일하게 저장됨 |
| `ego` | v1 Symlink | **0.0% (완전 일치)** | 초기 자차 상태(`vx, vy, ax, ay, steer, ...`) 동일 |
| `hum`, `past` | log 재구성 | **0.0% (완전 일치)** | `NavSimScenario`에서 추출한 미래/과거 자차 궤적은 v2.2와 동일 |

---

## 4. Candidate Scorer의 EPDMS 학습 정답(`official_scores`)에 미치는 영향

50개 씬 $\times$ 256개 앵커 후보($12,800$개 후보)에 대해 [`exact_score.official_scores`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/exact_score.py#L152-L222)가 산출하는 EPDMS 라벨을 비교한 결과입니다.

| EPDMS 정답 항목 | 후보 단위 불일치 (`12,800`개 중) | 씬(토큰) 단위 영향 비율 | 영향받는 학습 Head 및 추론 경로 |
| :--- | :---: | :---: | :--- |
| **`eff2`** (PDM-Closed 유효 진행도) | **26.00%** (`3,328 / 12,800`) | **26.0%** (`13 / 50` 씬) | `cand_eff2` Head가 사실상 v1 PDM-Closed 진행도 기반 값을 학습함 |
| **`ep2`** (v2 Ego Progress) | **17.21%** (`2,203 / 12,800`) | **20.0%** (`10 / 50` 씬) | 분모인 `pdm` 진행도(`eff2`)가 달라져 `ep2` 정규화 값이 달라짐 |
| **`epdms2`** (종합 EPDMS 점수) | **6.85%** (`877 / 12,800`) | **22.0%** (`11 / 50` 씬) | 전체 씬의 **약 22%**에서 후보별 `epdms2` 타깃 값이 달라짐 |
| **`nc2`** (v2 No At-Fault Collision) | **0.17%** (`22 / 12,800`) | **6.0%** (`3 / 50` 씬) | [`exact_score.py:179`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/exact_score.py#L179)가 `t = 1..40`만 검사하므로 99.83% 후보는 일치하나, 일부 객체/인간 페널티 차이로 0.17% 불일치 |
| **`ttc2`** (v2 Time-to-Collision) | **0.18%** (`23 / 12,800`) | **6.0%** (`3 / 50` 씬) | [`exact_score.py:179`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/exact_score.py#L179)에서 `ttc_last = 31`이 적용되어 99.82% 일치하나 0.18% 불일치 |
| **`dac2`, `ddc2`, `lk2`, `hc2`** | **0.00%** (`0 / 12,800`) | **0.0%** (`0 / 50` 씬) | **100% 정확** (정적 맵 및 과거/미래 자차 궤적 테이블이 v2.2와 완전 일치) |

---

## 5. 기존 평가 및 현재 진행 중인 학습(`corr30_cf_v15`)에 대한 판단

1. **`navtest` 공식 PDMS / EPDMS 평가 결과: 재평가 불필요 (100% 유효)**
   - `navtest` 평가 파이프라인([`run_eval_ep10.sh`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15/eval_ep10/run_eval_ep10.sh), [`run_eval_ep15.sh`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15/eval_ep15/run_eval_ep15.sh))은 `exact` / `exact2` 캐시를 전혀 읽지 않습니다.
   - 추론은 카메라 이미지와 `status_feature`만 사용하며, 채점은 공식 NAVSIM v1 `metric_cache` 및 공식 NAVSIM v2.2 [`metric_cache_v22_navtest`](file:///workspace/yongjae/perception_eval/epdms/metric_cache_v22_navtest) + `run_pdm_score_one_stage.py`로 수행되었습니다.

2. **현재 진행 중인 `corr30_cf_v15` 30에폭 학습: 중단 없이 완주**
   - `v2` 플래너 손실, `corridor_loss`, `corridor_plan_loss`(가중치 20.0), `cf_planners`, **`BEV Selector v15` Distillation**, 그리고 `CandidateScorer`의 **v1 헤드(`nc1, dac1, ddc1, ttc1, prog, eff1`)**는 `exact2`와 무관하게 100% 정상 학습 중입니다.
   - `CandidateScorer`의 v2 보조 헤드 중에서도 핵심 도로/차선/승차감 항목(`dac2, ddc2, lk2, hc2`)은 오차 0.0%이며, `nc2 / ttc2`도 99.82% 이상 일치합니다.
   - 실제로 15에폭 중간 평가에서 이미 **`PDMS 90.94% / EPDMS 89.43%`**로 30에폭 Baseline(`90.82% / 89.32%`)을 모두 추월했으므로, 현재 학습은 그대로 30에폭까지 완주시킵니다.

---

## 6. 향후 수정 방법 (Action Items)

### 방법 A: `navtrain`용 NAVSIM v2.2 `metric_cache` 생성 후 정식 [`build_exact.py --v2`](file:///workspace/byounggun/SSR/tools/corridor/build_exact.py#L35-L38) 실행 (권장)
1. `/workspace/navsim_workspace/navsim` (NAVSIM v2.2) 환경에서 `navtrain` (`navtrain_train` + `navtrain_val`)의 v2.2 `metric_cache`를 생성합니다 (`navtrain_val` 18,179개는 이미 생성된 캐시가 있으면 경로 연결).
2. 정식 빌더인 [`tools/corridor/build_exact.py`](file:///workspace/byounggun/SSR/tools/corridor/build_exact.py)로 `exact2`를 생성합니다:
   ```bash
   for s in navtrain_val navtrain_train; do
     nice $PY tools/corridor/build_exact.py \
       --split $s \
       --v2 --cl-full \
       --mc-root <PATH_TO_V22_METRIC_CACHE> \
       --out $KM_ROOT/exact2 \
       --workers 64 \
       --root $(dirname $TOKENS_ROOT)
   done
   ```

### 방법 B: `metric_cache` 전체 저장 없이 [`build_exact2_from_v1.py`](file:///workspace/byounggun/SSR/tools/corridor/build_exact2_from_v1.py)에서 v2.2 `observation` & `PDM-Closed` 직접 시뮬레이션
- 디스크 용량 문제로 `navtrain` 10.3만 개 토큰의 v2.2 `metric_cache.pkl`을 모두 저장하기 어렵다면, [`tools/corridor/build_exact2_from_v1.py`](file:///workspace/byounggun/SSR/tools/corridor/build_exact2_from_v1.py)를 수정하여:
  1. `SHARED_TABLES`에서 **`agent_rows`와 `pdm`을 제외** (`SHARED_TABLES = ("drv_seg", "lane_seg", "int_seg", "route_seg", "cl", "ego")`만 symlink 유지).
  2. 워커 내부에서 `/workspace/navsim_workspace/navsim` (NAVSIM v2.2)의 `NavSimScenario(scene, num_future_frames=8)`로부터 v2.2 공식 `PDMObservation`과 `PDM-Closed` 궤적을 메모리 상에서 생성한 뒤, [`build_exact.py`](file:///workspace/byounggun/SSR/tools/corridor/build_exact.py#L132-L175)와 동일한 로직으로 **`agent_rows`, `red_seg`, `red_t`, `pdm`을 직접 추출**하여 저장합니다.
  3. 생성 후 `navtest` (또는 `navtrain_val`) 50개 토큰에 대해 `build_exact.py --v2` 대비 **12개 전 테이블 0.0% 오차(bit-exact)**를 검증합니다.

---

## 7. 추가 이슈: 경로 필드 판정(`corridor_head._sample`)에서 자차 뒤쪽(`x < 0`) 범위 검사 누락

### 7.1 원인 코드 ([`corridor_head.py:267-277`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/corridor_head.py#L267-L277))

```python
def _sample(field: torch.Tensor, pts: torch.Tensor) -> tuple:
    """field [B, P, H, 200] (row fwd 0..H*0.32 m, col SSR x right -32..32 m); pts [B, P, N, 2] NAVSIM ego (x fwd,
    y left) -> (bilinear values [B, P, N], in-range mask (x <= H*0.32 m, |y| <= 32 m))."""
    fwd = field.shape[-2] * 0.32                                   # 32 m (100 rows) or 64 m (200 rows)
    u = (-pts[..., 1] + 32.0) / 64.0 * 2 - 1
    v = pts[..., 0] / fwd * 2 - 1
    b, p, n = pts.shape[:3]
    g = torch.stack([u, v], -1).reshape(b * p, n, 1, 2)
    val = F.grid_sample(field.reshape(b * p, 1, *field.shape[2:]).float(), g.float(), mode="bilinear",
                        padding_mode="border", align_corners=False).reshape(b, p, n)
    return val, (v <= 1) & (u.abs() <= 1)
```

1. **하한(`v >= -1`, 즉 `x >= 0`) 검사 누락**:
   - Corridor 격자(`corridor_sdf`, `corridor_occ`, `corridor_occ_dist`)는 자차 후륜(`x = 0`) 전방 `0 .. 64m` (`v in [-1, 1]`)만 표현하며, 후륜 뒤쪽(`x < 0`, `v < -1`)은 격자 밖입니다.
   - 그러나 `_sample`의 반환 마스크는 `(v <= 1) & (u.abs() <= 1)`로 작성되어 있어, **전방 상한(`x <= 64m`, `v <= 1`)과 좌우(`|y| <= 32m`, `|u| <= 1`)만 검사하고 자차 뒤쪽 하한(`x >= 0`, `v >= -1`)을 검사하지 않습니다.**
2. **`padding_mode="border"`와의 결합 동작**:
   - `x < 0` (`v < -1`)인 샘플링 포인트는 범위 밖(`inr = False`)으로 마스킹되는 대신 **`inr = True`로 판정**되며, `F.grid_sample(..., padding_mode="border")`에 의해 **격자의 첫 번째 행(`row 0`, `x = +0.16m`, 후륜 위치) 값을 그대로 복사**해 읽습니다.
   - 반면 BEV 특징을 뽑는 [`candidate_scorer._bev_sample`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/candidate_scorer.py#L46-L55)은 `padding_mode="zeros"`를 사용하여 `x < 0`에서 `0`을 반환하므로, 같은 포인트에 대해 BEV 샘플링과 필드 샘플링의 경계 처리가 서로 다릅니다.

### 7.2 어디서 `x < 0` 포인트가 발생하며 실제 영향은 무엇인가?

자차(Pacifica) 좌표계 원점 `(0, 0)`은 **후륜 축(Rear Axle)**이고, 차량 뒷범퍼는 후륜 뒤쪽으로 **`REAR = 1.127m` (`x = -1.127m`)** 돌출되어 있습니다. 따라서 출발 직후($t = 0.0 \sim 0.4\text{s}$, 차량이 `1.127m` 전진하기 전)나 정지/저속 후보 궤적에서는 차량 뒷부분 포인트들이 항상 `x < 0` 영역에 놓입니다:

1. **`corridor_plan_loss` ([`corridor_head.py:320-352`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/corridor_head.py#L320-L352))**:
   - **동적 충돌(`v_occ`) 및 TTC(`v_ttc`)**: 차량 **앞 절반(`fx = linspace(0.0, FRONT, 4)`) 12개 점만 샘플링**하므로 `fx >= 0`이어서 `x < 0` 문제가 거의 발생하지 않습니다.
   - **정적 도로 경계(`v_static`)**: 4개 모서리(`cx = [FRONT, FRONT, -REAR, -REAR]`)를 샘플링하므로 초기 스텝에서 뒷모서리 2개(`x = -1.127m`)가 `x = 0` 행의 도로 SDF를 읽습니다. 직선/완만한 도로에서는 `x = 0`과 `x = -1.1m`의 도로 폭이 거의 같아 근사치로 작동하지만, 급회전 시 뒷모서리가 `x < 0`으로 돌아가면 `x = 0` 위치의 SDF를 참조하게 됩니다.
2. **`CandidateScorer.field_readout` ([`candidate_scorer.py:57-89`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/candidate_scorer.py#L57-L89))**:
   - 4개 모서리(`sdf_min`)와 전체 차량 박스 15개 점(`fx = linspace(-REAR, FRONT, 5)`, `occ1` / `ttc1`)을 샘플링합니다.
   - 초기 스텝이나 정지 궤적에서 뒷줄 포인트(`fx = -1.127m`)가 범위 밖(`CLIP` / `0`)으로 제외되지 않고 `row 0` (`x = +0.16m`)의 SDF/점유율을 중복해서 읽습니다.

### 7.3 수정 방법 (To-Fix)

- **신규 학습 또는 Refiner 모듈([`candidate_scorer.py:field_readout`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/candidate_scorer.py#L57-L89))**:
  - `rear_mask: bool` 옵션을 두어 `pts[..., 0] >= 0` (`v >= -1`) 조건을 `inr` / `ti` 마스크에 추가합니다 (`inr = inr & (pts[..., 0] >= 0)`).
  - 단, **이미 `rear_mask=False` 규칙으로 학습된 체크포인트(`corr30_cf_v15` 등)를 평가할 때**는 학습 시와 동일한 분포를 유지하기 위해 기존 `CandidateScorer`에는 `rear_mask=False`를 유지하고, 재학습 실험부터 `v.abs() <= 1` (`0 <= x <= fwd`)을 적용합니다.

---

## 8. BEVFormer 카메라 Pillar 높이 기준점(`z`) 1개가 땅속(`-1.5m`)에 위치하는 문제

### 8.1 문제 위치 및 계산 과정
- **설정 파일**: [`default.py:70`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/configs/default.py#L70), [`default.py:98`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/configs/default.py#L98)
  ```python
  pc_range: Tuple[float, ...] = (
      -32.0,
      0.0,
      -2.0,
      32.0,
      32.0,
      2.0,
  )  # 64m 설정도 z는 (-2.0, 2.0) 동일
  encoder_num_points_in_pillar: int = 4
```
- **높이 기준점 생성 및 복원**: [`bevformer.py:369-374`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/bevformer.py#L369-L374) (`get_reference_points`), [`bevformer.py:429-431`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/bevformer.py#L429-L431) (`point_sampling`)
  ```python
  # get_reference_points (Z = pc_range[5] - pc_range[2] = 4.0, num_points_in_pillar = 4)
  zs = (
      torch.linspace(0.5, Z - 0.5, num_points_in_pillar) / Z
  )  # [0.5, 1.5, 2.5, 3.5] / 4.0 = [0.125, 0.375, 0.625, 0.875]

  # point_sampling (미터 단위 복원)
  reference_points[..., 2:3] = (
      reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
  )
  # -> [0.125, 0.375, 0.625, 0.875] * 4.0 + (-2.0) = [-1.5m, -0.5m, +0.5m, +1.5m]
```

### 8.2 원인: nuScenes 좌표계와 nuPlan/NAVSIM 좌표계의 원점(`z=0`) 차이
1. **nuScenes (원본 BEVFormer / VAD / SSR)**:
   - 좌표계 원점 `(0, 0, 0)`이 차량 지붕 위 **LiDAR 센서 위치(지면으로부터 약 `+1.84m`)**이므로 지면이 `z ≈ -1.8m`에 위치합니다.
2. **nuPlan / NAVSIM (현재 코드베이스)**:
   - 모든 카메라 외부 파라미터(`sensor2lidar_translation`)와 포인트클라우드가 **자차 후륜 차축(`rear_axle`) 기준 좌표계**로 정의되어 있으며, 원점 `z = 0.0m`는 **지면으로부터 약 `0.3m ~ 0.5m` 높이**에 있습니다 ([`default.py:129`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/configs/default.py#L129) 주석: `ego frame (rear axle ~0.3-0.5 m above ground)`).
   - 따라서 NAVSIM 좌표계에서 실제 지면(`ground`)의 높이는 **`z ≈ -0.3m ~ -0.5m`**입니다.
3. **4개 Pillar 기준점의 실제 물리적 위치**:
   - **Point 0 (`z = -1.5m`)**: **지면(`-0.4m`)보다 약 `1.0m ~ 1.2m` 아래인 땅속(지하)**
   - **Point 1 (`z = -0.5m`)**: 실제 도로 노면(지면) 높이
   - **Point 2 (`z = +0.5m`)**: 차량 범퍼/보닛 높이 (지면 위 약 `0.9m`)
   - **Point 3 (`z = +1.5m`)**: 승용차 지붕 / 보행자 상체 높이 (지면 위 약 `1.9m`)

### 8.3 영향 분석 및 수정 시 주의사항
- **기하학적 영향 (Camera Spatial Cross-Attention)**:
  - 각 BEV 셀 `(x, y)`의 4개 3D 기준점 중 첫 번째 점 `(x, y, -1.5m)`을 루프 카메라(`z ≈ +1.5m`)에서 투영하면, 실제 노면(`z ≈ -0.4m`)을 뚫고 지하 `1.1m` 지점을 바라보게 됩니다.
  - 이로 인해 근거리 BEV 셀에서는 투영점이 이미지 하단 밖(`v > 1.0`, `bev_mask = False`)으로 잘려 나가거나, 중·원거리 BEV 셀에서는 시선(ray)이 노면과 먼저 교차하는 **더 가까운 거리($d_{\text{road}} < d_{\text{cell}}$)의 노면 픽셀**을 샘플링하는 시차(parallax) 오차가 발생합니다.
- **기존 학습 모델 평가 시 주의사항**:
  - `SpatialCrossAttention`의 Deformable Attention은 4개 pillar 투영점을 기준으로 2D 오프셋($\Delta u, \Delta v$)과 attention 가중치를 학습하므로, **이미 `pc_range z=(-2.0, 2.0)`으로 학습된 체크포인트(`corr30_cf_v15` 등)의 추론 시점에는 `pc_range`를 변경하면 안 됩니다.**
  - **신규 학습 시 개선 방향**: `pc_range`의 `z` 범위를 지면 기준에 맞춘 **`(-1.0, 3.0)`** (기준점 `[-0.5m, +0.5m, +1.5m, +2.5m]`) 등으로 조정하면 4개 기준점이 모두 노면(`-0.5m`)부터 대형 차량/신호등(`+2.5m`)까지의 유효 높이에 배치됩니다. (참고: LiDAR 브랜치는 [`default.py:132`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/configs/default.py#L132)의 `lidar_z_range = (-3.0, 5.0)`을 별도로 사용하므로 이 문제의 영향을 받지 않습니다.)

---

## 9. 박스/Yaw 규칙 불일치 및 32m 격자 하드코딩 (현재 `corr30_cf_v15` 설정에서는 미사용 코드)

### 9.1 박스 / Yaw 규칙 불일치 (`distill/masks.py:rasterize_agent_mask` & `navsim_box_to_ssr`)
- **문제 위치**:
  1. [`para_ssr_targets.py:110-120`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/para_ssr_targets.py#L110-L120) ([`navsim_box_to_ssr`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/para_ssr_targets.py#L90-L125)):
     - NAVSIM 원본 박스 `(x_forward, y_left, z, length, width, height, heading)`를 SSR 좌표계 `(x_right, y_forward, z, width, length, height, syaw, vx, vy)`로 변환할 때 `syaw = wrap(-heading - π)`로 저장합니다.
     - 즉, 정면(`+y_forward`, `heading = 0`)을 바라보는 차량은 `syaw = -π` (`cos = -1, sin = 0`)가 되며, SSR 축에서 폭(`width = boxes[:, 3]`)은 `x_right`(`dx`) 방향, 길이(`length = boxes[:, 4]`)는 `y_forward`(`dy`) 방향에 놓입니다.
  2. [`distill/masks.py:158-167`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/distill/masks.py#L158-L167) ([`rasterize_agent_mask`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/distill/masks.py#L127-L189)):
     ```python
     half_w = (boxes[:, 3] * 0.5 * inflate)[:, None, None]
     half_l = (boxes[:, 4] * 0.5 * inflate)[:, None, None]
     yaw = boxes[:, 6][:, None, None]
     cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
     local_x = cos_y * dx + sin_y * dy
     local_y = -sin_y * dx + cos_y * dy
     inside = (local_x.abs() <= half_l) & (local_y.abs() <= half_w)
```
     - `yaw = 0`일 때 `dx`(`x_right`) 방향이 차량의 길이(`half_l`), `dy`(`y_forward`) 방향이 차량의 폭(`half_w`)이라는 일반 데카르트 회전 규칙을 가정하고 있어, `syaw = -heading - π` 규칙과 충돌합니다.
     - 그 결과 `heading = 0` (`syaw = -π`)인 차량에서 횡방향 거리(`|dx|`)를 차량 길이(`half_l`)와 비교하고 종방향 거리(`|dy|`)를 차량 폭(`half_w`)과 비교하여 **박스가 90° 회전된 상태로 래스터화**되며, 회전 방향(`±heading`)도 반대로 적용됩니다.
- **현재 설정(`corr30_cf_v15`)에서의 영향**:
  - `rasterize_agent_mask`는 구버전(`v1 ~ v5`) BEV Distillation의 Role Mask 경로에서만 호출되며, **`v6 ~ v16` (`TrajectoryAnchorDistillation`) 및 `corr30_cf_v15`에서는 아예 호출되지 않습니다.**
  - 또한 Detection Head([`det_motion_head.py`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/det_motion_head.py))는 학습과 평가([`aux_metrics.py`](file:///workspace/byounggun/SSR/navsim/evaluate/aux_metrics.py)) 모두 동일한 `navsim_box_to_ssr` 타깃 코드(`sin(syaw), cos(syaw)`)와 중심점 거리 매칭만 사용하므로 현재 학습·평가에는 영향이 없습니다.

### 9.2 32m 격자 하드코딩 (`corridor_head.py:AgentCrossAttn`)
- **문제 위치**: [`corridor_head.py:52-56`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/corridor_head.py#L52-L56) ([`AgentCrossAttn.__init__`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/corridor_head.py#L38-L65))
  ```python
  cell = 32.0 / bev_h  # 0.64 m (bev_h=50 가정 하드코딩)
  fwd = (torch.arange(bev_h) + 0.5) * cell
  ssr_x = -32.0 + (torch.arange(bev_w) + 0.5) * (64.0 / bev_w)
  gy, gx = torch.meshgrid(fwd, ssr_x, indexing="ij")
  self.register_buffer(
      "grid",
      torch.stack([gx, gy], -1).reshape(-1, 2) / 32.0,
      persistent=False,
  )
```
  - 64m 설정(`bev_h = 100`)에서 `cell = 32.0 / 100 = 0.32m`가 되어, 실제 `0 ~ 64m`인 BEV 픽셀의 종방향 위치 인코딩(`fwd`)이 **`0 ~ 32m`로 절반 압축**됩니다(반면 Agent 토큰 위치 `c0 = box[..., :2] / 32.0`는 실제 미터 좌표를 `32m` 단위로 정규화하므로 픽셀과 에이전트 간 종방향 스케일이 2배 어긋남).
  - *(추가로 [`distill/anchor_distill.py:308`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/distill/anchor_distill.py#L308)의 `_build_distance_weight_map` 내 `version == "v12"` 분기에도 `y / 32.0`이 하드코딩되어 있으나, `v15`에서는 실행되지 않습니다.)*
- **현재 설정(`corr30_cf_v15`)에서의 영향**:
  - `AgentCrossAttn`은 `corridor_agent_attn=True`일 때만 생성되며([`corridor_head.py:103`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/corridor_head.py#L103)), 기본값([`default.py:376`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/configs/default.py#L376)) 및 `corr30_cf_v15` 학습 스크립트([`scripts/km_corridor/train.sh`](file:///workspace/byounggun/SSR/scripts/km_corridor/train.sh))에서 **`corridor_agent_attn=False`로 비활성화(`self.agent = None`)**되어 있어 현재 실험에는 전혀 사용되지 않습니다.

### 9.3 미사용 코드 수정 여부에 대한 결론
- **현재 학습/평가 재수행 필요 여부**: **불필요** (`corr30_cf_v15`에서 두 코드 경로 모두 전혀 실행되지 않으므로 현재 모델의 가중치나 점수에 영향을 주지 않음).
- **코드 수정 권장 여부**: **수정 권장 (기존 체크포인트 호환성 100% 유지)**.
  - 현재 비활성화된 경로이므로 코드를 수정하더라도 기존 체크포인트(`corr30_cf_v15` 등)의 `state_dict`나 추론 결과는 1비트도 바뀌지 않습니다.
  - 반면 추후 `corridor_agent_attn=True` 어블레이션을 64m에서 켜거나 박스를 시각화/래스터화할 때 조용히(silently) 버그가 재발하는 것을 방지할 수 있으므로, 코드베이스 차원에서는 수정해 두는 것이 안전합니다.

---

## 10. 카메라 렌즈 왜곡 보정(`Undistortion`) 미적용 문제 (vs. SafeDrive)

### 10.1 문제 위치 및 SafeDrive와의 비교
- **NAVSIM 원본 카메라 데이터 ([`navsim/common/dataclasses.py:67-73`](file:///workspace/byounggun/SSR/navsim/common/dataclasses.py#L67-L73))**:
  - 각 카메라(`cam_f0`, `cam_l0`, `cam_r0` 등) 객체에는 왜곡이 포함된 원본 `1920×1080` 이미지(`cam.image`), 핀홀 내부 파라미터 행렬 `cam.intrinsics` ($K$), 그리고 5개 렌즈 왜곡 계수 `cam.distortion` ($D = [k_1, k_2, p_1, p_2, k_3]$)이 함께 저장되어 있습니다.
- **SafeDrive의 처리 방식 ([`/workspace/yongjae/SafeDrive/navsim/agents/safedrive/safedrive_features.py:92-116`](file:///workspace/yongjae/SafeDrive/navsim/agents/safedrive/safedrive_features.py#L92-L116))**:
  ```python
  def undistort_image(image, intrinsics, distortion, img_size):
    map1, map2 = cv2.initUndistortRectifyMap(
        intrinsics, distortion, None, intrinsics, img_size, cv2.CV_32FC1
    )
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)


  def process_camera(cam, crop_offset=(28, 0), resize_shape=(512, 256)):
    undist_img = undistort_image(
        cam.image, cam.intrinsics, cam.distortion, img_size=[1920, 1080]
    )
    cropped = undist_img[crop_offset[0] : -crop_offset[0]]
    resized = cv2.resize(cropped, resize_shape)
```
  - SafeDrive는 `cv2.initUndistortRectifyMap` + `cv2.remap`으로 **렌즈 왜곡을 먼저 보정(`undistort`)한 뒤** crop 및 resize를 수행하여, `lidar2img`가 가정하는 이상적인 핀홀(pinhole) 투영 기하와 실제 이미지 픽셀 위치를 일치시킵니다.
- **현재 `PARA-SSR`의 처리 방식 ([`para_ssr_features.py:240-259`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/para_ssr_features.py#L240-L259))**:
  ```python
  img = cam.image
  h, w = img.shape[:2]
  new_w = int(round(w * cfg.image_scale))
  new_h = int(round(h * cfg.image_scale))
  img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
  if cfg.crop_top:
    img = img[cfg.crop_top :]
  ...
  mats.append(
      build_lidar2img(
          cam.intrinsics,
          cam.sensor2lidar_rotation,
          cam.sensor2lidar_translation,
          (new_w / w, new_h / h),
          cfg.crop_top,
      )
  )
```
  - **`cam.distortion`을 전혀 사용하지 않고**, 왜곡된 원본 `cam.image`를 그대로 리사이즈한 뒤 선형 핀홀 행렬 `build_lidar2img(cam.intrinsics, ...)`로 BEV 3D 기준점을 투영합니다.

### 10.2 정량적 왜곡 크기 실측 (NAVSIM 카메라 캘리브레이션 기준)
NAVSIM 로그의 실제 카메라 왜곡 계수 `D = [-0.3561, 0.1725, -0.0021, 0.0005, -0.0523]` (`k1 = -0.3561`의 강한 배럴 왜곡)로 픽셀 변위량을 실측한 결과는 다음과 같습니다:

| 해상도 기준 | 평균 픽셀 오차 (Mean) | 중앙값 (p50) | 상위 10% (p90, 외곽부) | 최대 오차 (Max, 코너) | ResNet C5 (`stride 32`, `24×13`) 기준 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **원본 (`1920 × 1080`)** | `40.0 px` | `27.3 px` | `98.3 px` | `159.9 px` | — |
| **모델 입력 (`768 × 432`, `0.4×`)** | **`16.0 px`** | **`10.9 px`** | **`39.3 px`** | **`64.0 px`** | **평균 `0.5칸`, 외곽 `1.2칸` 어긋남** |

- 이미지 중심부에서는 오차가 작지만, **`cam_f0`와 좌우 카메라(`cam_l0`, `cam_r0`)가 이어지는 이미지 좌우 외곽 영역(p90)에서는 모델 입력 해상도 기준으로도 약 `39px` (C5 feature map에서 `1.2칸` 이상)**의 기하학적 투영 불일치가 발생합니다.

### 10.3 수정 방향 및 주의사항
- **기존 체크포인트 평가 시 주의사항**:
  - 이미 왜곡된 이미지 분포에 맞춰 Deformable Attention 오프셋과 백본이 학습된 체크포인트(`corr30_cf_v15` 등)를 평가할 때는 왜곡 보정을 켜면 안 됩니다.
- **신규 재학습 시 적용 방법 (속도 최적화 포함)**:
  - [`para_ssr_features.py:undistort_image`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/para_ssr_features.py#L48-L65)에서 `(w, h, intrinsics, distortion)` 바이트 키로 `cv2.initUndistortRectifyMap`의 `(map1, map2)`를 `@functools.lru_cache(maxsize=64)`로 캐싱하고 `cv2.remap`을 수행하여 SafeDrive와 100% 동일한 왜곡 보정을 CPU 병목 없이 적용했습니다.

---

## 11. 조치 완료 현황 (2026-10-28)

`corr30_cf_v15` 20에폭(`epoch=19-step=13300.ckpt`) 평가 완료 후, 위 5가지 항목을 모두 수정·검증하고 신규 학습(`corr30_cf_v15_fix`, GPU `4,5,6,7`)에 반영 완료했습니다:

1. **Issue 1 (`exact2` v1 데이터 혼입 해결)**: [`tools/corridor/build_exact2_v22.py`](file:///workspace/byounggun/SSR/tools/corridor/build_exact2_v22.py)로 `navtrain_val`(`18,179/18,179`) 및 `navtrain_train`(`85,109/85,109`) 전체를 순수 NAVSIM v2.2 기준으로 재생성하여 `/workspace/navsim_workspace/corridor_cache/exact2`에 적용 완료 (기존 캐시는 `exact2_v1_backup`으로 백업).
2. **Issue 2 (`corridor_head.py:_sample` 후방 범위 검사)**: [`corridor_head.py:277`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/corridor_head.py#L277)을 `(v.abs() <= 1) & (u.abs() <= 1)`로 수정하여 `x < 0` 후방 좌표가 `border` 클램핑으로 오판정되는 문제 해결.
3. **Issue 3 (BEVFormer 기둥 높이 기준점 지면화)**: [`default.py:70,77`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/configs/default.py#L70-L77) 및 [`train.sh:74,82`](file:///workspace/byounggun/SSR/scripts/km_corridor/train.sh#L74-L82)의 `pc_range` / `map_pc_range` $z$ 범위를 `[-1.0, 3.0]`으로 수정하여 4개 기둥 기준점이 `[-0.5m(노면), +0.5m, +1.5m, +2.5m]`에 놓이도록 보정.
4. **Issue 4 (미사용 코드 버그 수정)**: [`corridor_head.py:52`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/modules/corridor_head.py#L52)의 `AgentCrossAttn` 셀 크기를 `cell = 64.0 / bev_w`(`0.64m`)로 수정하고, [`distill/masks.py:165-167`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/distill/masks.py#L165-L167)의 `rasterize_agent_mask` 회전/폭·길이 축을 `navsim_box_to_ssr` 규약과 일치하도록 수정.
5. **Issue 5 (카메라 왜곡 보정 적용)**: [`para_ssr_features.py:48-65,262-265`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/para_ssr_features.py#L48-L65)에 SafeDrive와 동일한 `undistort_image`(`cv2.initUndistortRectifyMap` LRU 캐싱 + `cv2.remap`)를 적용.




