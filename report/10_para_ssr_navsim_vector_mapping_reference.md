# PARA-SSR NAVSIM vector mapping 구현·GT convention·포팅 참고서

작성일: 2026-09-01  
대상 저장소: `/home/yongjae/e2e/SSR-para-navsim`  
목적: 다른 online mapping 모델을 NAVSIM으로 포팅할 때 현재 PARA-SSR의 map GT,
좌표계, tensor contract, loss, 평가기를 재사용하거나 비교하기 위한 기준 문서

이 문서는 **현재 동작하는 코드를 기준으로 작성했다.** 이전 보고서의 설계안이 아니라
`para_ssr_targets.py`, `map_head.py`, `losses.py`, auxiliary mAP 산출물과 실제
`navtest` 12,146 token을 대조한 결과다.

---

## 0. 가장 먼저 알아야 할 결론

1. NAVSIM/nuPlan HD map의 원천은 polygon/linestring 기반 **vector geometry**다.
2. NAVSIM bundled TransFuser는 이 geometry를 학습 시점에 **raster semantic map**으로
   변환한다.
3. 현재 PARA-SSR 포트는 원본 SSR/VAD 구조를 유지하기 위해 geometry를 직접 추출한
   **3-class vector polyline task**를 사용한다.
4. 현재 map head는 planner 입력이 아니다. 학습 중 공유 BEV encoder에 흘리는 gradient를
   통해서만 planning에 영향을 준다.
5. 현재 Chamfer mAP는 PARA-SSR ablation용 **비공식 auxiliary metric**이다. NAVSIM
   leaderboard metric이 아니다.

현재 contract를 한 표로 줄이면 다음과 같다.

| 항목 | 현재 PARA-SSR NAVSIM 값 |
|---|---|
| 입력 | 8 surround cameras, history index `[2, 3]` |
| map 기준 시점 | current history frame, 기본 index `3` |
| 모델 좌표 | `x=right`, `y=forward` |
| ROI | `x_right ∈ [-15,15] m`, `y_forward ∈ [-30,30] m` |
| 공유 BEV | `100×100×256`, 10,000 tokens |
| map 표현 | vector polyline instance |
| 클래스 | `divider(0)`, `ped_crossing(1)`, `boundary(2)` |
| prediction | 100 vectors × 20 points × 2D, 3 decoder layers |
| training GT | 최대 100 vectors × 20 equivalent orders × 20 points × 2D |
| map head 크기 | 2,910,001 parameters |
| matching | focal class cost + ordered normalized-point L1 Hungarian |
| loss | focal cls + point L1 + metric direction cosine, 3-layer deep supervision |
| auxiliary 평가 | symmetric Chamfer AP @ 0.5/1.0/1.5 m |

---

## 1. NAVSIM TransFuser raster와 PARA-SSR vector는 다른 task다

NAVSIM 데이터가 raster GT를 파일로 저장해 두는 것은 아니다. `Scene.map_api`가 nuPlan
vector map을 제공하고, 각 모델의 target builder가 필요한 표현으로 변환한다.

| 항목 | NAVSIM bundled TransFuser | 현재 PARA-SSR |
|---|---|---|
| 표현 | raster semantic segmentation | vector polyline detection |
| target/output shape | GT `[128,256]`, logit `[B,7,128,256]` | class `[L,B,100,3]`, point `[L,B,100,20,2]` |
| 좌표 | NAVSIM `x_forward,y_left` | SSR/VAD `x_right,y_forward` |
| ROI | 전방 `0~32 m`, 좌우 `±32 m` | 앞뒤 `±30 m`, 좌우 `±15 m` |
| 해상도 | `0.25 m/pixel` | 연속 좌표; 20-point polyline |
| 클래스 | background/road/walkway/centerline/static/vehicle/pedestrian | divider/crosswalk/boundary |
| loss | pixel cross entropy, weight 10 | query focal + point L1 + direction |
| 제공 평가 | 이 checkout에 map mIoU 집계 없음 | 비공식 Chamfer mAP 추가 |

특히 TransFuser의 `centerline`은 lane/lane-connector의 `baseline_path`이고,
PARA-SSR의 `divider`는 lane 객체의 좌우 edge다. 이름만 바꿔 같은 class로 취급하면 안 된다.
TransFuser raster에는 vehicle/pedestrian도 들어가므로 순수한 static HD-map segmentation도
아니다.

관련 코드:

- `navsim/agents/transfuser/transfuser_config.py:82-111`
- `navsim/agents/transfuser/transfuser_features.py:197-269`
- `navsim/agents/transfuser/transfuser_model.py:34-57`
- `navsim/agents/transfuser/transfuser_loss.py:21-30`

---

## 2. End-to-end map 데이터 흐름

```text
NAVSIM log + nuPlan map DB
          │
          ▼
Scene.map_api에서 global vector geometry 조회
          │
          ▼
current ego pose 기준 NAVSIM local frame (x_forward, y_left)
          │
          ▼
SSR frame으로 축 변환 (x_right=-y_left, y_forward=x_forward)
          │
          ▼
30 m × 60 m ROI clip → 1 m 미만 fragment 제거
          │
          ▼
20-point arc-length resample + equivalent orders 생성
          │
          ▼
[0,1] 좌표 정규화 → class round-robin → train cap 100
          │
          ▼
100×100 shared BEV → 100 vector queries × 20 point tokens
          │
          ▼
Hungarian matching + 3-layer deep supervision
          │
          ├── training: total loss와 shared-BEV gradient
          └── eval: metric 좌표 복원 → 100-point Chamfer mAP
```

중요하게도 map API는 **GT를 만드는 target builder에서만** 사용된다. `ParaSSRModel`의
map prediction 입력으로 HD map이 들어가는 것은 아니다. 모델은 camera로 만든 BEV에서
현재 local map을 예측한다.

---

## 3. 좌표계와 ROI convention

### 3.1 세 좌표계를 분리해서 생각한다

#### A. nuPlan global map frame

Map polygon/linestring과 `ego_pose=(X,Y,heading)`가 있는 world frame이다.

#### B. NAVSIM current-ego local frame

```text
x_nav = forward
y_nav = left
```

global point `(X,Y)`와 current ego `(X_e,Y_e,θ_e)`에 대해:

```text
dx = X - X_e
dy = Y - Y_e

x_forward =  cos(θ_e)·dx + sin(θ_e)·dy
y_left    = -sin(θ_e)·dx + cos(θ_e)·dy
```

`_geometry_local_coords()`가 이 변환을 Shapely affine transform으로 수행한다.

여기서는 반드시 `Scene.frames[cur_idx].ego_status.ego_pose`의 **global pose**를 써야 한다.
`AgentInput.ego_statuses[-1].ego_pose`는 feature 입력용으로 current ego 기준 상대좌표화된
pose이므로, 이것으로 global map API를 조회하면 엉뚱한 지역의 지도를 가져온다.
Builder는 pose가 global인지 나타내는 flag를 직접 검사하지 않고 `Scene` contract를 신뢰한다.
따라서 local pose를 잘못 넘겨도 예외 없이 그럴듯하지만 틀린 GT가 생길 수 있어 좌표 회귀
테스트가 필요하다.

#### C. PARA-SSR/VAD BEV frame

```text
x_ssr = right
y_ssr = forward

x_ssr = -y_left
y_ssr =  x_forward
```

현재 구현의 핵심 한 줄은 다음과 같다.

```python
arr = np.stack([-arr[:, 1], arr[:, 0]], axis=-1)
```

좌표 sanity check:

| NAVSIM local point | 의미 | SSR point |
|---|---|---|
| `(10, 0)` | 10 m 전방 | `(0, 10)` |
| `(0, 5)` | 5 m 좌측 | `(-5, 0)` |
| `(0, -5)` | 5 m 우측 | `(5, 0)` |
| `(-10, 0)` | 10 m 후방 | `(0, -10)` |

새 모델이 일반적인 `x_forward,y_left` BEV를 사용한다면 위 마지막 축 변환을 하지 말아야
한다. 현재 PARA-SSR target을 native NAVSIM 축으로 되돌리는 식은 다음과 같다.

```text
x_forward =  y_ssr
y_left    = -x_ssr
```

### 3.2 현재 물리 범위

```python
pc_range = (-15.0, -30.0, -2.0, 15.0, 30.0, 2.0)
#            x_min  y_min  z_min  x_max y_max z_max
```

Map은 z를 사용하지 않고 x/y rectangle만 사용한다.

```text
SSR frame:
  x_right   ∈ [-15, 15] m   # 좌우 30 m
  y_forward ∈ [-30, 30] m   # 앞뒤 60 m

NAVSIM frame으로 같은 영역을 표현하면:
  x_forward ∈ [-30, 30] m
  y_left    ∈ [-15, 15] m
```

BEV는 `H=100`, `W=100`이므로 dense feature의 물리 cell 크기는:

```text
longitudinal: 60 / 100 = 0.6 m/cell
lateral:      30 / 100 = 0.3 m/cell
```

Vector point는 cell center로 quantize되지 않으며 연속 좌표를 회귀한다.

현재 dense BEV tensor는 우연히 `100×100` 정사각형이라 H/W를 바꾸거나 lateral/longitudinal
축을 뒤집은 버그가 shape만으로 드러나지 않는다. 비정사각 ROI/feature로 포팅할 때는
`x_right → W`, `y_forward → H` 관계를 좌표 probe로 따로 검증해야 한다.

### 3.3 정규화와 역정규화

GT와 head point prediction은 ROI 기준 `[0,1]` 좌표다.

```text
u = (x_right   - (-15)) / 30 = (x_right + 15) / 30
v = (y_forward - (-30)) / 60 = (y_forward + 30) / 60

x_right   = 30·u - 15
y_forward = 60·v - 30
```

모델 출력에 sigmoid를 쓰므로 prediction은 구조적으로 ROI 안에 있다. 새 모델이 meter
좌표를 직접 회귀한다면 이 정규화를 억지로 적용할 필요는 없지만, 기존 matcher/loss와
비교하려면 동일한 공간에서 cost를 계산해야 한다.

관련 코드:

- `navsim/agents/para_ssr/configs/default.py:58-62,189-195`
- `navsim/agents/para_ssr/para_ssr_targets.py:112-117,562-620`
- `navsim/evaluate/aux_metrics.py:106-188`

---

## 4. Map task taxonomy와 GT source

현재 class order는 코드 contract다.

```python
MAP_CLASS_NAMES = ("divider", "ped_crossing", "boundary")
```

| label | class | nuPlan source | 현재 표현 |
|---:|---|---|---|
| 0 | `divider` | `LANE`, `LANE_CONNECTOR`의 `left_boundary`와 `right_boundary` | open line |
| 1 | `ped_crossing` | `CROSSWALK` polygon의 exterior | 원천은 closed ring; ROI clipping 후 open 가능 |
| 2 | `boundary` | roadblock/connector polygon union의 exterior와 hole | 원천 closed ring; clip 후 open 가능 |

### 4.1 divider

- Lane과 lane-connector 양쪽 edge를 모두 수집한다.
- boundary object에 `id`가 있으면 같은 ID를 중복 제거한다.
- 같은 geometry가 서로 다른 ID를 가지면 geometry-level 중복 제거는 하지 않는다.
- `SemanticMapLayer.BOUNDARIES`를 직접 요청하지 않는다. 현재 nuPlan
  `get_proximal_map_objects()` 경로가 이 layer를 제공하지 않기 때문에 lane object를 통해
  접근한다.
- lane centerline/baseline path가 아니다.

가장 큰 의미 리스크는 **모든 lane edge를 VAD divider로 간주한다는 점**이다. nuPlan lane
edge에는 road outline이나 virtual edge가 섞일 수 있는데, 현재 `boundary_type_fid`를
분류해 걸러내지 않는다. 새 모델이 이 GT를 그대로 쓰면 PARA-SSR과 비교는 가능하지만,
일반적인 lane divider semantic과 완전히 같다고 주장하면 안 된다.

실제 네 지도 GPKG를 확인하면 참조되는 boundary type도 하나가 아니다. Lane boundary에는
type 0/2, lane-connector boundary에는 type 3이 관찰됐다. 공개 map object 경로에서 이
semantic을 안정적으로 노출·검증하지 않았으므로 현재 구현은 의식적인 근사다.

### 4.2 ped_crossing

- crosswalk polygon의 채워진 면적이 아니라 **외곽 contour 하나**를 학습한다.
- polygon interior/hole은 현재 추가하지 않는다.
- 인접하거나 겹치는 crosswalk polygon끼리 union하지 않고 object별 instance로 둔다.
- clipping되지 않고 완전히 ROI 안에 있으면 closed vector, ROI에서 잘리면 open fragment가
  될 수 있다.

Raster crosswalk segmentation을 원하는 모델이라면 contour를 채운 mask로 바꾸어야 한다.
현재 vector AP와 raster IoU는 서로 다른 task다.

### 4.3 boundary

- 각 roadblock exterior를 그대로 넣지 않는다.
- `ROADBLOCK`과 `ROADBLOCK_CONNECTOR` polygon을 먼저 `unary_union`한다.
- union 결과의 exterior와 interior ring을 vector로 넣는다.
- 이 처리는 인접 roadblock 사이의 내부 seam을 boundary로 잘못 학습하는 문제를 줄이고,
  intersection connector 형상을 보존한다.

### 4.4 현재 task에 없는 것

다음은 현재 3-class map target에 없다.

- lane centerline / baseline path
- stop line
- walkway area
- drivable-area mask
- intersection area class
- traffic-light state/vector
- lane topology, predecessor/successor, connectivity
- map element persistent ID 또는 temporal tracking ID

새 online mapping 모델이 topology나 lane graph를 출력한다면 geometry만 재사용하고 graph
relation GT를 별도로 만들어야 한다.

관련 코드:

- `navsim/agents/para_ssr/para_ssr_targets.py:10-33,71`
- `navsim/agents/para_ssr/para_ssr_targets.py:418-560`

---

## 5. GT geometry 생성 절차

### 5.1 기준 시점과 map query

```python
cur_idx = scene.scene_metadata.num_history_frames - 1
pose = scene.frames[cur_idx].ego_status.ego_pose
```

현재 `navtrain`/`navtest`는 history 4장이므로 current index는 `3`이다. Builder의 일반
contract는 `cur_idx = num_history_frames - 1`이므로 history 길이를 바꾸면 map 기준 index도
함께 바뀐다. 모델 camera queue `[2,3]`은 현재 설정에서 두 장을 쓰는 별도 contract다.

Map query radius는 현재 ROI 좌표 절댓값 최댓값 30 m에 1.5를 곱한 **45 m**다. ROI 가장
먼 corner까지 거리는 약 33.54 m이므로 rectangle을 덮는다. nuPlan의 proximal query는
이 radius를 global axis-aligned pre-query 영역에 사용하고, 실제 rotated local rectangle은
후단 Shapely clip에서 확정한다.

요청 layer가 빠지거나 map query/extraction이 실패하면 empty GT로 조용히 진행하지 않고
token/map context를 포함한 `RuntimeError`를 낸다. 새 포트에서도 schema mismatch를
background-only 학습으로 숨기지 않는 편이 안전하다. 다만 정상 query 결과가 비었거나 모든
geometry가 clipping/min-length filter에서 탈락한 것은 오류가 아니다. 이 경우 정상적으로
all-padding/background target을 만든다.

다만 **개별 geometry clipping 예외는 현재 warning 후 그 instance만 버린다.** 새 모델에서는
누락 counter를 기록하거나 strict mode에서 fail-fast하도록 개선하는 것이 좋다.

### 5.2 global → local → SSR → clip

각 geometry는 다음 순서로 처리한다.

1. global map geometry를 current ego local NAVSIM frame으로 변환
2. `(x_forward,y_left)`를 `(x_right,y_forward)`로 회전
3. `[-15,15]×[-30,30]` Shapely polygon과 intersection
4. `LineString`, `MultiLineString`, `GeometryCollection`의 유효 line piece를 분리
5. 점이 2개 미만이거나 Shapely 길이가 `1.0 m` 미만이면 제거

Clipping이 하나의 원본 line을 여러 instance로 나눌 수 있다. closed source contour라도
ROI에 의해 잘린 piece는 좌표의 첫/마지막 점을 다시 비교해 open으로 처리한다.

### 5.3 20-point arc-length resampling

각 piece는 선분 누적 길이 기준으로 균일한 20개 point를 생성한다. 원본 vertex index를
균일 샘플링하는 방식이 아니다. 길이가 사실상 0이면 같은 점을 반복하는 fallback이 있지만,
앞 단계의 1 m filter 때문에 정상 line에서는 거의 쓰이지 않는다.

Closed contour는 첫 point와 마지막 point가 같으므로 tensor는 20점이지만 실질적인 고유
sample 위치는 최대 19개다.

### 5.4 equivalent ordering 20개

Prediction과 동일 geometry의 점 순서 차이를 벌하지 않기 위해 GT마다 20개 ordering을
저장한다.

| geometry | ordering 생성 |
|---|---|
| open line | 정방향과 역방향을 번갈아 반복해 20 slot |
| closed contour | unique cycle의 시작점을 바꾼 cyclic shift 20개; reverse는 만들지 않음 |

Closed input의 중복 endpoint를 포함한 채 roll하면 contour 내부를 가로지르는 jump가 생기므로,
중복 endpoint를 제거한 cycle을 shift하고 다시 닫은 뒤 20점으로 resample한다. 원본 vertex가
20개보다 많으면 시작점을 대략 균일하게 고르고, 적으면 반복한다.

이 구현은 VAD v2의 **의미**를 따르지만 exact code 복제는 아니다. 원 VAD v2는 최대
`fixed_num-1` ordering을 random subset하고 남는 slot에 sentinel을 둘 수 있지만, 현재 포트는
20개 slot을 모두 결정론적인 유효 ordering으로 채운다. 새 모델이 원 VAD checkpoint/code와
exact parity를 요구하면 이 차이도 별도 검증해야 한다.

#### 현재 closed-direction convention의 한계

코드 주석의 `canonical direction`은 전 dataset에 걸쳐 방향을 정규화한다는 뜻이 아니다.
**source geometry의 방향을 유지한다**는 뜻이며, 실제 GT에는 clockwise/counter-clockwise가
혼재한다.

전체 `navtest`의 완전 closed contour를 실측한 signed-area 방향:

| class | negative | positive | near-zero |
|---|---:|---:|---:|
| ped_crossing | 1,624 | 8,311 | 0 |
| boundary | 2,824 | 5,557 | 2,287 |

따라서 새 모델에는 두 선택지가 있다.

- **PARA-SSR checkpoint parity가 목적**: 현재 source direction + cyclic-only convention 유지
- **새 target v2를 정의**: exterior/interior orientation을 정규화하거나 closed reverse까지
  equivalent order에 포함

두 방식을 같은 실험표에서 비교하려면 target protocol 버전을 반드시 분리해야 한다.

#### zero-area/backtracking closed boundary

현재 filter는 `LineString.length >= 1 m`만 확인하므로 면적이 0에 가까운 왕복형 closed
contour도 통과한다. 전체 `navtest`에서 이런 boundary가 2,287개이고, 1,912/12,146
token(15.7%)에 최소 하나가 있다. 확인된 예시는 token `00c893a01244562c`의 6.80 m line으로,
A→B를 따라갔다가 같은 길로 B→A를 되돌아온다.

새 target v2에서는 다음 중 하나를 명시적으로 결정하는 것이 좋다.

- closed ring에 최소 `abs(signed_area)`와 valid-polygon 검사를 적용
- backtracking contour를 open line으로 정리
- 현 protocol 호환을 위해 그대로 두되 품질 flag를 함께 저장

기존 checkpoint를 같은 GT로 평가하는 동안에는 사후에 몰래 제거하면 안 된다.

### 5.5 class round-robin과 train cap

원천 geometry 수는 divider에 치우친다. 소스 순서대로 100 slot을 채우면 divider가 전부
차지해 crosswalk/boundary가 사라질 수 있어 다음처럼 class round-robin으로 채운다.

```text
divider[0] → ped_crossing[0] → boundary[0]
divider[1] → ped_crossing[1] → boundary[1]
...
한 class가 소진되면 남은 class를 계속 채움
```

Class 내부 순서는 map API/geometry 처리 순서를 유지한다. 거리나 길이 순 정렬은 아니다.
특히 union geometry 순서는 GEOS/Shapely 결과에도 의존하므로 map DB나 라이브러리 버전이
바뀌면 tail에서 잘리는 divider가 달라질 수 있다. Training은 100개에서 자르지만 evaluation
GT는 절대 자르지 않는다.

현재 agent validation은 `map_max_vec <= map_num_vec`, 양수 `map_min_length`, 유효한
`pc_range` 조합을 모두 강하게 검사하지 않는다. 현재 값 `100<=100`, `1.0 m`, 양의 extent는
안전하지만 새 config에서는 이 조건들을 fail-fast 검사로 추가하는 것이 좋다.

관련 코드:

- `navsim/agents/para_ssr/para_ssr_targets.py:400-448`
- `navsim/agents/para_ssr/para_ssr_targets.py:562-656`
- `navsim/agents/para_ssr/para_ssr_targets.py:658-674`

---

## 6. 실제 GT 분포: query/cap 설계 참고값

완료된 `navtest` 12,146 token의 uncapped auxiliary record를 독립 집계한 값이다.

### 6.1 scene당 vector 수

| class | mean | median | p95 | p99 | max | 존재 scene |
|---|---:|---:|---:|---:|---:|---:|
| divider | 28.95 | 29 | 54 | 66 | 111 | 100.0% |
| ped_crossing | 1.77 | 1 | 6 | 8 | 14 | 54.0% |
| boundary | 4.90 | 5 | 9 | 11 | 17 | 100.0% |
| **all** | **35.62** | **35** | **66** | **79** | **120** | **100.0%** |

전체 GT 수:

| class | GT 수 |
|---|---:|
| divider | 351,619 |
| ped_crossing | 21,511 |
| boundary | 59,534 |
| **total** | **432,664** |

100개를 넘는 scene은 5개다. 최대 120개 token은 `94d209006f485164`다. 현재
round-robin train cap은 이 5개 scene의 tail에서 divider 80개만 버렸고 boundary는 모두
보존했다. 다만 이 5개 scene에는 `ped_crossing` GT가 원래 0개였으므로 이 결과를
crosswalk 보존 효과의 실증으로 해석하면 안 된다.

따라서 현재 split에서는 100 queries가 대부분을 덮지만 완전 coverage는 아니다. 다른
ROI나 taxonomy를 쓰면 이 분포를 다시 측정해야 하며, 현재 숫자로 query 수를 정하면 안 된다.

---

## 7. Training target tensor contract

Target builder의 sample 단위 tensor와 batch 후 shape는 다음과 같다.

| key | sample shape | batch shape | dtype | 의미 |
|---|---|---|---|---|
| `gt_map_pts` | `[100,20,20,2]` | `[B,100,20,20,2]` | `float32` | vector, equivalent order, point, `(u,v)` |
| `gt_map_labels` | `[100]` | `[B,100]` | `int64` | `0/1/2`; padding 값 자체는 0 |
| `gt_map_valid` | `[100]` | `[B,100]` | `bool` | 실제 GT slot 여부 |

차원 순서는 반드시 다음처럼 읽는다.

```text
[vector_instance, equivalent_order, sampled_point, xy]
```

`gt_map_labels`의 padding도 숫자만 보면 divider `0`이다. 반드시 `gt_map_valid`로 먼저
필터링해야 한다. Label 값만 보고 padding을 학습에 넣으면 divider GT가 대량으로 생긴다.

Uncapped evaluation target은 variable length다.

```text
gt_map_pts:    [G,20,20,2]
gt_map_labels: [G]
valid mask 없음; 전부 valid
```

Auxiliary record로 저장할 때는 ordering 0만 metric 좌표로 복원한다. Chamfer는 배열 순서와
동일 sampled sequence의 역방향에는 불변이므로 평가에는 equivalent-order 차원이 필요 없다.
다만 closed curve의 cyclic start가 바뀐 상태에서 유한 개수로 다시 resample하면 sampling
phase가 달라질 수 있으므로 parameterization까지 수학적으로 완전 불변이라고 일반화하면 안
된다.

---

## 8. Map head 구조와 prediction contract

### 8.1 입력과 query

Map head 입력은 공유 BEV:

```text
bev_embed: [B, 100×100, 256] = [B,10000,256]
```

Map query는 instance와 point embedding을 factorize한다.

```text
instance_embedding: [100, 512]
point_embedding:    [20, 512]

query[i,j] = instance_embedding[i] + point_embedding[j]
100 × 20 = 2,000 decoder tokens
```

512차원을 `query_pos[256]`와 content `query[256]`로 나눈다. 별도의 map 전용
`100×100` BEV positional table은 없다.

### 8.2 decoder

3개 decoder layer 각각:

1. 2,000 point token self-attention
2. single-level shared BEV에 deformable cross-attention
3. FFN
4. normalized 2-D reference point iterative refinement

다음 layer로 넘기는 refined reference는 `detach()`된다. 각 vector의 20 point feature를
평균해 instance classification을 하고, point token별로 2-D 좌표를 회귀한다.

Map head parameter 수는 **2,910,001**이다.

### 8.3 prediction shape

```text
all_map_cls_scores: [3,B,100,3]       # raw logits; 분류 시 sigmoid
all_map_pts_preds:  [3,B,100,20,2]    # sigmoid-normalized [0,1]
```

3개 class channel과 별도로 background channel은 없다. Training에서 unmatched query의
target label은 `3 == num_classes`이고, focal one-hot을 만들 때 세 class가 모두 0인
background로 처리한다.

### 8.4 planning과의 관계

Map output은 planner, detection, motion decoder로 들어가지 않는다.

```text
shared BEV ──► planner
          ├──► detection/motion
          └──► map head
```

따라서 mapping이 planning을 바꾸는 경로는 **map loss가 공유 backbone/BEV encoder에 주는
gradient**뿐이다. Head 출력 품질이 좋아졌다고 planner가 그 polyline을 직접 읽는 구조는
아니다.

더 정확히는 map loss가 current image encoder/neck, shared BEV query·positional embedding과
BEV transformer까지는 업데이트하지만, `navi_se`, TokenLearner, latent/waypoint/trajectory
decoder 같은 post-BEV planner module은 업데이트하지 않는다.

현재 task는 current-frame local vector prediction이다. Persistent map ID, map tracking,
temporal association 또는 temporal consistency loss는 없다. History camera/BEV는 encoder가
사용하지만 과거 BEV 생성은 `no_grad` 경로다.

관련 코드:

- `navsim/agents/para_ssr/modules/map_head.py:1-14,143-268`
- `navsim/agents/para_ssr/para_ssr_model.py:200-223,298-336`

---

## 9. Hungarian matching과 loss

### 9.1 Matching cost

각 decoder layer, 각 batch sample마다 SciPy Hungarian matching을 별도로 수행한다.

```text
cost(q,g) = 2.0 × FocalLossCost(class_q, class_g)
          + 1.0 × min_order Σ |points_q - points_g,order|
```

- Point cost는 `[0,1]` normalized coordinate 40개(`20 points×2`)의 **합**이다.
- 20 equivalent ordering 중 가장 작은 cost와 그 order index를 선택한다.
- Hungarian 결과는 query와 GT의 1:1 assignment다.
- bbox, IoU, direction matching cost는 현재 없다.
- 선택된 order를 실제 point/direction loss에도 사용한다.

#### 숨은 config 결합

Matcher weight `assigner_cls_weight=2.0`, `assigner_pts_weight=1.0`은 현재 dataclass YAML에
노출되지 않고 `ParaMapHead` 생성자 기본값으로 들어간다. `loss_map_cls_weight` 또는
`loss_map_pts_weight`만 바꾸면 matcher objective와 training loss weight가 조용히 달라진다.

새 모델에서는 matcher weight를 config에 명시하고, 동일하게 유지할 의도라면 fail-fast
equality 검사를 넣는 편이 안전하다.

### 9.2 Loss term

| term | 정의 | weight |
|---|---|---:|
| `loss_map_cls` | sigmoid focal, `gamma=2`, `alpha=0.25` | 2.0 |
| `loss_map_pts` | matched normalized point-coordinate L1 합 / positive vector 수 | 1.0 |
| `loss_map_dir` | metric-space 인접 direction의 `1-cosine` 합 / positive vector 수 | 0.005 |

Direction interval은 1이므로 20-point vector에서 인접한 19개 direction을 쓴다. Direction
loss만 `(30,60)` metric extent를 곱해 anisotropic ROI를 보정한 뒤 각도를 계산한다.

Classification은 `100 queries × 3 binary class` focal 항을 모두 더해 positive-vector 수로
나눈다. Unmatched/background query도 numerator에 들어가며 별도 background channel/weight는
없다. DDP에서는 `sync_cls_avg_factor=True`로 rank-mean positive count를 denominator로 쓴다.

### 9.3 Deep supervision과 실제 최종 map loss

3개 decoder layer 모두 같은 weight로 supervision한다.

```text
layer 0: loss_map_cls_d0, loss_map_pts_d0, loss_map_dir_d0
layer 1: loss_map_cls_d1, loss_map_pts_d1, loss_map_dir_d1
layer 2: loss_map_cls,    loss_map_pts,    loss_map_dir
```

Suffix가 없는 `loss_map_*`는 **마지막 decoder layer 값**일 뿐 전체 map task loss가 아니다.
현재 별도 `loss_map` aggregate log는 없다.

```python
map_task_loss = sum(all_nine_map_terms) * task_loss_weight["map"]
```

각 term의 실제 scalar multiplier stack은 다음과 같다.

```text
internal term weight × head loss_weight(1.0) × task_loss_weight.map(1.0)
```

`gscale/map`은 이 scalar stack이 아니라 shared-BEV 경계 gradient에만 적용된다.

다른 모델과 total map loss를 비교할 때는 같은 수의 decoder layer/point/query와 같은
reduction인지 먼저 확인해야 한다.

### 9.4 Normalized-coordinate loss의 비대칭

Point matcher와 point L1은 meter가 아니라 normalized 좌표에서 계산된다.

```text
1 m lateral error      = 1/30
1 m longitudinal error = 1/60
```

따라서 현재 cost/loss는 같은 1 m라도 lateral error를 longitudinal error보다 2배 크게 본다.
이는 현재 VAD parity convention이며 direction loss만 metric scale을 쓴다.

새 모델에서는 다음 중 하나를 명시해야 한다.

- 기존 PARA-SSR 비교: normalized L1 유지
- 물리적으로 등방성인 새 protocol: matcher와 point loss 모두 meter 좌표에서 계산

ROI를 바꾸면서 loss weight를 그대로 쓰면 같은 meter error의 loss scale도 바뀐다.

### 9.5 구조 크기를 바꾸면 loss scale도 바뀐다

- point 20개를 늘리면 point cost/L1 합이 거의 point 수에 비례한다.
- direction term 수는 `P-1`에 비례한다.
- decoder layer를 늘리면 deep-supervision 합도 늘어난다.
- query를 늘리면 positive 수로만 정규화되는 background focal 합이 증가한다.

Query/point/layer/ROI 변경과 loss-weight 재조정은 하나의 실험으로 취급해야 한다.

Training head 경계에서 prediction의 NaN/+Inf/-Inf는 명시적으로 0으로 바꾼다. 최종 loss에도
`torch.nan_to_num`을 적용하지만 기본 동작상 NaN은 0, ±Inf는 dtype의 최대/최소 finite 값이
된다. Run을 즉시 죽이지 않는 장점이 있지만 instability를 숨길 수 있으므로, 새 모델에서는
별도의 NaN/Inf counter와 strict debug mode를 두는 것이 좋다.

관련 코드:

- `navsim/agents/para_ssr/modules/losses.py:36-64,98-133,178-217`
- `navsim/agents/para_ssr/modules/map_head.py:270-387`
- `navsim/agents/para_ssr/para_ssr_loss.py:183-192`

---

## 10. 현재 training wiring

### 10.1 epoch-30 run은 single-stage다

현재 완료된 NAVSIM epoch-30 모델은 staging이 아니다. 30 epoch 처음부터
plan/detection/motion/map을 모두 함께 학습했다.

```yaml
use_map_head: true

task_loss_weight:
  plan: 2.0
  det: 1.0
  motion: 1.0
  map: 1.0

grad_balance_target:
  plan: 0.4
  det: 0.3
  map: 0.3
```

`task_loss_weight.map`과 GradBalancer의 `gscale/map`은 역할이 다르다.

| 설정 | map head parameter gradient | shared BEV gradient |
|---|---|---|
| `task_loss_weight.map` | scaling함 | scaling함 |
| `gscale/map` | scaling하지 않음 | `_ScaleGrad`로 scaling함 |

즉 head 자체는 full-strength로 학습하면서 공유 perception/planning representation에 미치는
영향만 조절할 수 있다. Epoch-30 checkpoint의 마지막 controller state는
`gscale/map≈0.1955`였고 마지막 epoch 평균 `gshare/map≈0.2984`로 target 0.3 부근이었다.
이 값은 학습 중 변하는 controller state이지 새 모델의 추천 고정 weight가 아니다.

### 10.2 map을 0 weight로 두는 것과 freeze는 다르다

현재 경로는 map loss를 계산한 뒤 task weight를 곱한다. `task_loss_weight.map=0`만 쓰면:

- map forward와 Hungarian 계산 비용이 남는다.
- map parameter에 zero gradient tensor가 생길 수 있다.
- AdamW weight decay가 계속 적용될 수 있다.

완전히 끄려면 `use_map_head=false`, `requires_grad=False` 및 optimizer 제외 같은 명시적
처리가 필요하다. Head는 학습하되 shared BEV 영향만 끄는 실험은 별도 gradient valve로
구성해야 한다. 현재 GradBalancer에서는 target의 `map: 0`이 정확히 이 역할을 하며 map
head parameter는 계속 학습한다. 다만 `use_map_head=false`는 parameter schema가 달라져 기존
epoch-30 checkpoint의 `strict=True` load가 실패하므로 checkpoint ablation에서는 주의한다.

### 10.3 기본 validation에는 map이 없다

```python
run_aux = self.training or cfg.test_aux_heads
```

현재 archived config는 `test_aux_heads=false`다. 따라서 일반 Lightning validation에서는
map/detection head를 실행하지 않고 planning만 validation한다. TensorBoard에도 기본적으로
`val/loss_map_*`가 없다.

Map branch output과 validation loss가 필요하면 반드시 다음 중 하나가 필요하다.

- model call에 `run_aux=True`
- evaluation config에서 `test_aux_heads=true`
- 현재 auxiliary runner처럼 전용 inference path 사용

이 설정만으로 mAP가 자동 계산되는 것은 아니다. Full mAP에는 token별 prediction/uncapped GT
record와 전체 split AP aggregation이 추가로 필요하다.

### 10.4 archive YAML만으로 loss를 완전히 재현할 수 없다

Archived Hydra config에 `use_map_head=true`, `map_num_vec=100`은 명시되지만 여러 값은 당시
Python dataclass default에서 채워졌다.

- `map_max_vec=100`
- point/order 수 20/20
- class 3개
- decoder 3 layers
- `pc_range`, `map_min_length`, `map_dir_interval`
- loss와 matcher weight
- `sync_cls_avg_factor=True`, head-level `loss_weight=1.0`
- decoder attention/FFN dropout 0.1
- GradBalancer momentum/clamp

Checkpoint shape로 구조는 확인할 수 있지만 loss weight는 checkpoint tensor에 저장되지
않는다. 재현 실험에는 source revision과 dataclass default도 함께 보존해야 한다.

관련 코드:

- `navsim/agents/para_ssr/configs/default.py:130-165`
- `navsim/planning/script/config/common/agent/para_ssr_agent.yaml:27-58`
- `navsim/agents/para_ssr/para_ssr_loss.py:1-12,126-235`
- `navsim/agents/para_ssr/para_ssr_model.py:298-336`

---

## 11. Auxiliary vector-map mAP protocol

Metric 이름은 `NAVSIMAuxMap/chamfer_mAP`, protocol version은 1이다.

### 11.1 Prediction decode

1. 마지막 decoder layer만 사용
2. 100 query × 3 class logit에 sigmoid
3. query×class 300개 score를 flatten
4. deterministic stable confidence top-100 선택
5. score threshold 없음
6. normalized point를 SSR metric 좌표로 역정규화

Flattened query×class 방식이므로 **같은 polyline query가 서로 다른 class label로 두 번 이상
선택될 수 있다.** 새 모델이 per-query argmax를 사용하면 같은 이름의 mAP로 섞지 말아야 한다.

### 11.2 Distance

Prediction과 GT의 20-point line을 각각 arc-length 기준 100점으로 다시 resample한다.

```text
Chamfer(P,G) = 0.5 × (
    mean_{p∈P} min_{g∈G} ||p-g||₂
  + mean_{g∈G} min_{p∈P} ||g-p||₂
)
```

이는 symmetric mean Chamfer다. Hausdorff, polygon IoU, topology score가 아니다. 방향을
뒤집은 line은 같은 거리를 갖고 polygon 내부 면적도 평가하지 않는다.

평가 GT도 raw 고해상도 map geometry를 직접 쓰는 것이 아니라 **training representation인
20-point canonical line을 100점으로 재보간**한다. 따라서 원천 geometry의 20점 변환에서
이미 손실된 세부 곡률은 metric에서 복구되지 않는다.

### 11.3 Matching과 AP

- class별, scene별 confidence 내림차순 처리
- 각 prediction의 absolute-nearest GT를 먼저 고정
- 가장 가까운 GT가 이미 matched면 두 번째 unmatched GT로 fallback하지 않고 FP
- threshold `0.5`, `1.0`, `1.5 m`
- 경계 포함: `distance <= threshold`
- 전체 token을 confidence로 다시 정렬해 precision-envelope PR area AP 계산
- 최종 mAP는 전체 navtest에서 `3 classes × 3 thresholds` AP 평균

작은 debug subset에 어떤 class GT가 하나도 없으면 그 class AP가 평균에서 빠진다. 그런
subset mAP를 전체 navtest mAP와 직접 비교하면 안 된다.

### 11.4 Training GT cap과 evaluation GT

Training GT는 100개에서 cap하지만 evaluation은 `compute_map_evaluation_targets()`로 모든
vector를 사용한다. 평가에도 cap을 적용하면 cap에서 빠진 실제 GT에 대한 올바른 prediction이
FP가 되어 metric이 오염된다.

### 11.5 epoch-30 sanity baseline

동일 epoch-30 checkpoint, 전체 `navtest` 12,146 token 결과:

| metric | AP |
|---|---:|
| **Map Chamfer mAP** | **0.228846** |
| divider | 0.305325 |
| ped_crossing | 0.120198 |
| boundary | 0.261014 |

이 수치는 새 adapter/evaluator의 회귀 기준으로 사용할 수 있지만 NAVSIM 공식 map score로
보고하면 안 된다.

결과:

- `work_dirs/eval/para_ssr_ep30_aux/aux_metrics.json`
- `work_dirs/eval/para_ssr_ep30_aux/aux_metrics.csv`

관련 코드:

- `navsim/evaluate/aux_metrics.py:31-33,106-230,255-348,381-520,524-697`
- `navsim/planning/script/run_aux_evaluation.py`
- `scripts/evaluation/eval_para_ssr_aux.sh`

---

## 12. 새 모델에 auxiliary evaluator를 붙일 때

현재 runner는 generic map runner가 아니다. 다음을 하드코딩한다.

- `ParaSSRAgent`
- `ParaSSRTargetBuilder`
- archived PARA-SSR Hydra config
- detection과 map output을 함께 갖는 record schema
- 현재 모델의 100 query와 protocol V1의 decoded top-100/20 raw point/3 class/고정 ROI

따라서 새 map-only 또는 다른 output schema 모델에 `eval_para_ssr_aux.sh`를 그대로 실행하면
안 된다.

현재 PARA-SSR checkpoint를 동일 protocol로 재실행하는 명령은 다음과 같다.

```bash
cd /home/yongjae/e2e/SSR-para-navsim
conda activate ssr-navsim
GPU_IDS=2,3 AUX_EXPERIMENT=eval/para_ssr_ep30_aux \
  AUX_TRAINING_CONFIG=/path/to/code/hydra/config.yaml \
  scripts/evaluation/eval_para_ssr_aux.sh /path/to/model.ckpt
```

산출물은 `work_dirs/$AUX_EXPERIMENT/` 아래의 `manifest.json`, `records/<token>.npz`,
`aux_metrics.csv`, `aux_metrics.json`이다. 마지막 JSON이 정상 완료 marker다. Resume 시 manifest가
checkpoint/config/token/source/runtime/batch identity 불일치를 거부하므로 다른 checkpoint나
config에는 새 `AUX_EXPERIMENT` directory를 써야 한다.

### 권장 adapter 경계

현재 `evaluate_auxiliary_records()`도 detection+map field를 함께 요구하므로 map-only 모델에
그대로 호출할 수 없다. 가장 안전한 방법은 그 안의 map accumulator를
`evaluate_map_records()` 같은 순수 map 함수로 분리하고, 새 모델의 inference 결과를 다음
metric-space record로 바꾸는 adapter를 만드는 것이다. 새 모델도 detection head를 가진다면
기존 full record schema 전체를 맞춰 `evaluate_auxiliary_records()`를 재사용할 수 있다.

순수 accumulator는 단위 테스트 편의를 위해 `N_pred <= 100`, `P >= 1`도 허용하지만,
production protocol V1 runner가 기록을 만들 때는 **정확히 100개의 decoded query-class
prediction entry, raw 20 points, token 오름차순 및 중복 없는 record**를 요구한다. Flatten
decode라 하나의 vector query가 여러 class entry로 선택될 수 있으며, runner가 요구하는 것은
원 head query 수 100이 아니라 `Q×3 >= 100`이다. 동일 V1 수치를 비교하려면 이 stricter
runner contract까지 고정해야 한다.

```text
map_pred_points: [N_pred,P,2] float32, metric SSR frame
map_pred_scores: [N_pred]     float32, [0,1]
map_pred_labels: [N_pred]     int64, 0/1/2
map_gt_points:   [N_gt,P,2]   float32, metric SSR frame, uncapped
map_gt_labels:   [N_gt]       int64, 0/1/2
token:           unique string
```

새 모델이 NAVSIM native frame을 유지한다면 evaluator 직전에 SSR frame으로 바꾸거나,
GT와 prediction을 모두 native frame으로 둔 generic metric을 만들어야 한다. 한쪽만 회전하는
실수가 가장 위험하다.

Raster 모델은 이 vector record와 Chamfer AP를 사용하지 말고 class별 IoU/mIoU evaluator를
별도로 정의한다.

---

## 13. SceneFilter와 dataset contract

현재 checkout은 `data -> /data/navsim` symlink를 사용하며 기본 경로는 다음과 같다.

```text
OPENSCENE_DATA_ROOT=$REPO/data/dataset
NUPLAN_MAPS_ROOT=$REPO/data/dataset/maps
map version=nuplan-maps-v1.0
```

두 root 환경변수는 `navsim.common.dataclasses` import 시점에 module constant로 잡힌다. Python
process가 해당 module을 import하기 **전에** 설정해야 하며, 실행 중 뒤늦게 바꾸면 반영되지
않는다.

현재 비교 가능한 SceneFilter pool contract:

| 용도 | 물리 data split | SceneFilter | logs | token allow-list | history/future | interval |
|---|---|---|---:|---:|---:|---:|
| 학습/검증 candidate | `trainval` | `navtrain` | 1,192 | 103,288 | 4 / 10 | 1 |
| auxiliary test | `test` | `navtest` | 136 | 12,146 | 4 / 10 | 1 |

`navtrain`은 그 자체가 물리 dataset split 이름이 아니다. 기본 training은
`split=trainval`로 scene을 읽은 뒤 다음처럼 공식 train/val log 집합과 교집합한다.

```text
train loader: log ∈ (navtrain.log_names ∩ cfg.train_logs), current token ∈ navtrain.tokens
val loader:   log ∈ (navtrain.log_names ∩ cfg.val_logs),   current token ∈ navtrain.tokens
```

현재 결과는 **train 85,109 token / val 18,179 token**이며 합이 navtrain allow-list 103,288과
같다. SceneFilter의 `tokens`는 window 시작 token이 아니라
`frame_list[num_history_frames - 1]["token"]`, 즉 current history frame token과 비교된다.
새 dataloader adapter에서 시작 frame token을 검사하면 조용히 다른 sample을 만들게 된다.

### 반드시 지킬 점

1. `log_names`만 복사하지 말고 YAML의 `tokens` allow-list를 함께 사용한다. Sensor blob은
   log의 모든 frame에 존재하지 않는다.
2. `frame_interval=1`을 명시한다. 생략하면 `history+future=14`가 기본 interval이 되어
   14 frame마다 한 scene만 만든다.
3. `has_route=true`를 바꾸면 기존 training/test split과 달라진다.
4. `max_scenes`는 정렬되지 않은 filesystem log 순서에서 조기 종료하므로 reproducible
   debug subset의 보장이 약하다. 작은 explicit token subset과 `max_scenes=None`을 쓰거나
   loader log를 먼저 정렬하는 편이 낫다.
5. Map GT 자체는 future sensor를 사용하지 않지만 exact token/split parity를 위해 4/10 scene
   contract를 유지한다.
6. 일반 loader는 YAML에 요청한 token이 실제 loader에서 빠져도 자동 fail하지 않는다.
   `configured token set == actual token set` inventory 검사를 별도로 둔다.
7. Camera/history/lidar SensorConfig를 바꾸면 기존 `navtrain.tokens`만 믿지 말고 새로 요구되는
   sensor blob coverage를 전체 token에서 다시 측정한다.

관련 코드:

- `navsim/planning/script/config/common/scene_filter/navtrain.yaml`
- `navsim/planning/script/config/common/scene_filter/navtest.yaml`
- `navsim/planning/script/run_training.py:24-40`
- `navsim/common/dataclasses.py:470-497`
- `navsim/common/dataloader.py:14-70`

---

## 14. Cache 주의사항

현재 PARA-SSR training은 `cache_path: ''`로 online target generation을 사용한다. Target
builder unique name은 현재 상수 `para_ssr_target`이다.

기존 NAVSIM Dataset cache는 builder의 `get_unique_name()` 파일 존재 여부를 중심으로
validity를 판단한다. 다음 identity를 확인하지 않는다.

- target source hash
- class taxonomy
- coordinate convention
- ROI
- point/order 수
- map DB version/content
- camera indices, image scale/crop/input size와 SensorConfig
- feature-builder source hash

GT convention을 수정한 뒤 이전 cache를 그대로 두면 stale target을 조용히 읽을 수 있다.
cache directory key에 쓰이는 `SceneMetadata.initial_token`은 이름과 달리 현재 history frame의
current token이다.

두 cache 경로의 split 동작도 다르다.

- 일반 `Dataset + cache`는 계속 `SceneLoader.tokens`로 index하므로 SceneFilter split은
  유지한다. 다만 cache identity/config hash를 검증하지 않는다.
- `use_cache_without_dataset=true`의 `CacheOnlyDataset`은 `SceneLoader`를 만들지 않고
  `cfg.train_logs`/`cfg.val_logs` 아래에서 필요한 builder 파일이 모두 존재하는 cache token을
  전부 사용한다. 따라서 `navtrain`/`navtest` token allow-list, `has_route`, `frame_interval`,
  `max_scenes`를 적용하지 않는다.

새 포트에서 cache를 쓴다면 최소한 다음을 cache identity에 넣는다.

```text
target_protocol_version
target/feature source SHA
pc_range / coordinate frame
class names and sources
num_points / num_orders / min_length / cap policy
map version
camera indices / image transform / SensorConfig
sorted token-set hash
```

Auxiliary `records/` resume cache는 위 training Dataset cache와 별개이며 manifest로 실행 identity를
보호한다. 다만 dataset 실제 byte/content hash까지 넣지는 않으므로 같은 경로의 데이터가
교체되면 감지하지 못한다.

관련 코드:

- `navsim/planning/script/config/training/default_training.yaml:50-53`
- `navsim/planning/training/dataset.py:32-85,123-174`

---

## 15. 새 online mapping 모델 포팅 권장안

### 선택 A: PARA-SSR와 동일한 vector GT로 공정 비교

다음은 그대로 고정한다.

- SSR axis와 ROI
- 3-class source semantics
- clipping/min-length
- current equivalent-order convention
- train class round-robin cap과 eval uncapped GT
- 동일 Chamfer decode/matching/AP

Model head와 encoder만 교체하면 GT/metric 차이를 제거한 architecture 비교가 된다.

### 선택 B: 새 모델의 native vector convention 유지

예를 들어 `x_forward,y_left`, front-only ROI, lane centerline class를 쓰는 모델이라면 억지로
PARA-SSR target을 끼우지 않는다. 대신:

1. 원천 map geometry 조회 부분만 재사용
2. 모델 좌표/ROI/taxonomy로 target 생성
3. metric도 같은 좌표/클래스로 별도 versioning
4. 가능하면 공통 subset/class에 대한 adapter 평가를 추가

### 선택 C: NAVSIM TransFuser식 raster task

- `TransfuserTargetBuilder._compute_bev_semantic_map()`을 기준으로 rasterize한다.
- `[128,256]`, 0.25 m/pixel, nominal front `[0,32) m`, lateral `[-32,32) m` contract를
  유지한다.
- Class loop가 road→walkway→centerline→static→vehicle→pedestrian 순서로 mask를 덮어쓰므로
  동일 pixel의 **뒤 class 우선순위**도 GT protocol 일부로 고정한다.
- OpenCV mask에 적용하는 `rot90(...)[::-1]`과 pixel origin `[0,width/2]`까지 재현하고,
  전/후/좌/우 cardinal point가 기대 pixel에 가는지 독립 테스트한다.
- Pixel CE 학습과 per-class IoU/mIoU 평가를 구현한다.
- PARA-SSR vector Chamfer mAP와 같은 열에 직접 비교하지 않는다.

### 재사용 권장/비권장 경계

| 항목 | 권장 |
|---|---|
| nuPlan map layer 조회와 global→ego transform | 재사용 가치 높음 |
| union road boundary와 ROI clipping | 원하는 semantic이면 재사용 |
| SSR 축 회전 | 새 모델 축이 같을 때만 재사용 |
| class taxonomy | 연구 질문에 맞게 명시적으로 결정 |
| 100 cap/20 points/20 orders | 모델 구조와 GT 분포에 맞게 재측정 |
| `evaluate_auxiliary_records()` | full detection+map schema와 정렬 token까지 같을 때만 재사용 |
| map-only metric | 내부 map accumulator를 순수 함수로 분리해 재사용 |
| 전체 aux runner | PARA-SSR hardcoding을 분리한 뒤 사용 |
| TransFuser raster metric과 vector mAP 혼합 | 사용 금지 |

---

## 16. 포팅 체크리스트

### 16.1 GT 정의 전

- [ ] 모델 내부 BEV axis를 `x/y` 이름이 아니라 `forward/left/right` 의미로 문서화
- [ ] current ego 기준인지 global 기준인지 결정
- [ ] front-only인지 후방 포함인지 결정
- [ ] ROI 경계 포함/clip 규칙 결정
- [ ] divider가 lane edge인지 centerline인지 결정
- [ ] crosswalk가 contour인지 area인지 결정
- [ ] road boundary가 개별 polygon exterior인지 union contour인지 결정
- [ ] topology/ID/traffic-control GT 필요 여부 결정
- [ ] closed direction 정규화/reverse-equivalence 여부 결정
- [ ] degenerate ring 제거 protocol 결정

### 16.2 Tensor/loss

- [ ] GT/prediction point 수가 다를 때 interpolation 위치 결정
- [ ] padding label과 valid mask 분리
- [ ] train cap과 eval uncapped 경로 분리
- [ ] query/class decode가 flattened top-k인지 per-query argmax인지 명시
- [ ] point matcher cost가 sum인지 mean인지 확인
- [ ] normalized loss와 metric loss 중 하나를 의도적으로 선택
- [ ] matcher weight와 loss weight를 함께 config에 노출
- [ ] decoder deep-supervision layer 수 변화에 loss scale 보정
- [ ] aggregate `loss_map_total` 로그 추가 권장

### 16.3 Split/cache/eval

- [ ] dataset/map root 환경변수를 Python import 전에 설정
- [ ] `navtrain`/`navtest` token allow-list 사용
- [ ] `frame_interval=1` 확인
- [ ] configured/actual token inventory와 새 SensorConfig blob coverage 비교
- [ ] cache protocol/source/config hash 확인
- [ ] validation에서 aux head가 실제 실행되는지 확인
- [ ] full split에서 모든 class GT가 존재하는지 확인
- [ ] score threshold, top-k, duplicate query, threshold `<=/<` 고정
- [ ] metric 이름에 protocol version과 비공식 여부 기록
- [ ] raster이면 class overwrite priority와 cardinal-axis pixel test 고정

---

## 17. 최소 회귀 테스트 권장 세트

새 포트가 학습을 시작하기 전에 다음은 자동 테스트로 통과시키는 것이 좋다.

1. **Axis cardinal test**
   - NAVSIM 전방 `(10,0)` → 모델의 전방 축
   - NAVSIM 좌측 `(0,5)` → 모델의 좌측 축
2. **ROI corner round-trip**
   - metric → normalized → metric 오차가 float tolerance 이내
3. **Global pose test**
   - translation+rotation ego에서 global geometry를 current local로 독립 재유도
4. **Open order test**
   - forward/reverse line geometry가 동일
5. **Closed order test**
   - cyclic shift가 contour 내부 jump를 만들지 않음
6. **Clip test**
   - closed polygon이 ROI에서 잘리면 open으로 재판정
7. **Class presence test**
   - 실제 여러 도시 scene에서 0/1/2 class 생성 확인
8. **Cap test**
   - 100개 초과 GT에서 rare class가 round-robin으로 보존
9. **Padding test**
   - invalid label 0이 divider positive로 들어가지 않음
10. **Matcher oracle test**
    - exact prediction과 reversed open line이 올바른 order로 match
11. **Metric oracle test**
    - perfect prediction AP=1
12. **Threshold edge test**
    - Chamfer가 정확히 0.5 m일 때 `<=`로 TP
13. **Duplicate test**
    - nearest GT duplicate가 second-nearest로 fallback하지 않음
14. **Uncapped eval test**
    - token `648b875dc34259c2`: training 100, evaluation 112 vectors
15. **Full inventory test**
    - navtest 12,146 unique token, missing/extra 0

현재 관련 테스트:

- `tests/test_para_ssr_targets.py`
- `tests/test_para_ssr_model_invariants.py`
- `navsim/agents/para_ssr/test_loss_parity.py`
- `tests/test_para_ssr_aux_metrics.py`
- `tests/test_aux_evaluation_runner.py`

2026-09-01 기준 위 다섯 묶음을 `ssr-navsim` 환경에서 함께 실행해 **43 passed**를
확인했다. 출력된 14건은 Matplotlib/PyParsing 외부 deprecation warning이다.

---

## 18. 설정 치트시트

```yaml
# shared BEV / coordinate contract
pc_range: [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
bev_h: 100
bev_w: 100
embed_dims: 256

# vector-map contract
use_map_head: true
map_num_vec: 100
map_max_vec: 100
map_num_pts_per_vec: 20
map_num_orders: 20
map_num_classes: 3
map_num_decoder_layers: 3
map_dir_interval: 1
map_min_length: 1.0

# head loss
loss_map_cls_weight: 2.0
loss_map_pts_weight: 1.0
loss_map_dir_weight: 0.005

# 현재는 MapHead 내부 default; 새 포트에서는 config 노출 권장
assigner_cls_weight: 2.0
assigner_pts_weight: 1.0

# multi-task wiring
task_loss_weight:
  map: 1.0
grad_balance_target:
  map: 0.3
```

실제 Python dataclass 필드가 YAML에 명시되지 않으면 default로 채워지는 값이 있으므로,
위 블록을 그대로 Hydra override로 쓸 수 있다는 의미는 아니다. 새 모델 config schema에 맞춰
명시적으로 옮겨야 한다.

---

## 19. 코드 위치 색인

| 내용 | 파일 |
|---|---|
| 전체 map config | `navsim/agents/para_ssr/configs/default.py` |
| Hydra agent config | `navsim/planning/script/config/common/agent/para_ssr_agent.yaml` |
| GT geometry/좌표/cap | `navsim/agents/para_ssr/para_ssr_targets.py` |
| vector map head | `navsim/agents/para_ssr/modules/map_head.py` |
| matcher/loss primitives | `navsim/agents/para_ssr/modules/losses.py` |
| shared gradient/task 합산 | `navsim/agents/para_ssr/para_ssr_loss.py` |
| model branch wiring | `navsim/agents/para_ssr/para_ssr_model.py` |
| auxiliary metric | `navsim/evaluate/aux_metrics.py` |
| auxiliary runner | `navsim/planning/script/run_aux_evaluation.py` |
| evaluation launcher | `scripts/evaluation/eval_para_ssr_aux.sh` |
| NAVSIM raster 비교 | `navsim/agents/transfuser/transfuser_{config,features,model,loss}.py` |
| split contract | `navsim/planning/script/config/common/scene_filter/{navtrain,navtest}.yaml` |
| dataset/cache | `navsim/planning/training/dataset.py` |
| full mAP 결과 | `report/09_navsim_port_progress.md` §17 |

---

## 20. 최종 권고

다른 online mapping 모델을 NAVSIM으로 옮길 때 가장 안전한 비교 순서는 다음과 같다.

1. **첫 실험은 현재 PARA-SSR GT/ROI/class/metric을 그대로 사용**해 model architecture만
   바꾼다.
2. 다음 실험부터 모델 고유 taxonomy나 native coordinate를 적용하되 protocol 이름을
   분리한다.
3. `divider` semantic, closed orientation, degenerate boundary는 현재 구현의 알려진 GT
   품질 이슈로 기록한다.
4. Training loss가 내려가는 것만 보지 말고 uncapped full-split Chamfer AP와 class별 recall을
   함께 본다.
5. Mapping이 planning에 미치는 영향을 연구한다면 map head 성능뿐 아니라 shared-BEV
   `gshare/map`, gradient valve, planning PDM score를 별도로 측정한다. 현재 구조에서는 map
   output이 planner에 직접 들어가지 않기 때문이다.
