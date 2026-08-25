# SSR·PARA-SSR·외부 teacher BEV 표현의 통합 stitching 분석

이 문서는 기존 `REPORT.md`, `REPORT_alignment.md`에서 수행한
SSR↔PARA-SSR 표현 분석과, 후속으로 수행한 BEVFusion·MapTRv2 teacher
분석을 하나의 흐름으로 통합한다.

전체 연구 질문은 두 단계다.

1. planning-only SSR과 auxiliary task를 함께 학습한 PARA-SSR의 BEV는
   planning 관점에서 어떤 관계인가?
2. detection teacher BEVFusion과 mapping teacher MapTRv2의 BEV도 같은
   방식으로 planning-only SSR의 frozen planner가 읽을 수 있는가?

## 전체 결론

1. **Raw BEV swap은 표현 품질을 직접 재지 않는다.** 정보가 100% 보존된
   채널 순열 BEV조차 frozen P planner에서는 loss가 8.7배 나빠진다.
2. **SSR↔PARA는 일반 선형 adapter를 넣으면 대부분 연결된다.** Final
   X1/X2/X3에서 raw frozen-planner gap의 `86.6~95.7%`가 제거된다.
3. **하지만 단순 채널 순열이나 정확한 gauge symmetry 때문은 아니다.**
   permutation+scale은 `2.3~34.9%`만 회수한다. 관측된 차이는 여러 채널의
   일반 선형 재조합을 필요로 한다.
4. **외부 teacher는 전혀 다른 경우다.** BEVFusion·MapTRv2의 held-out
   feature R²는 `0.0975/0.1126`, planner gap recovery는 약 26%뿐이다.
5. **그 26%도 전부 같은 셀의 유용한 정보가 아니다.** Mean-only가 이미
   12.5~17.0%를 회수하며, BEVFusion은 잘못 spatial roll한 control이
   올바른 정렬보다 오히려 조금 낫다.
6. 따라서 teacher BEV 전체를 planning BEV로 직접 바꾸려 하지 말고,
   planning core를 유지한 별도 projected view에서 detection/map 정보를
   task-specific하게 증류하는 것이 현재 증거에 맞는 방향이다.

---

## 0. 이전 분석: SSR-noFFP와 PARA-SSR

### 0.1 출발점

최종 목적은 BEVFusion의 detection 정보와 MapTRv2의 mapping 정보를 SSR에
증류하는 것이다. 그 전에 다음 선행 질문을 해결해야 했다.

> Detection·motion·map auxiliary task가 함께 학습한 BEV는 planning-only
> SSR의 BEV보다 planning에 더 좋은가? 다르다면 정보가 없어진 것인가,
> 아니면 planner가 읽는 표현 방식이 달라진 것인가?

모델을 다음처럼 분리했다.

\[
B=E(x)\in\mathbb{R}^{10000\times256},\qquad
\hat{T}=R(B)
\]

- `E`: 카메라 sequence에서 BEV를 만드는 encoder
- `B`: 100×100 셀, 셀마다 256채널인 BEV 표현
- `R`: `navi_se → TokenLearner → decoder → trajectory` planning readout

### 0.2 비교 모델

| 이름 | checkpoint | 학습 역할 |
|---|---|---|
| P | SSR-noFFP 2GPU e12 | planning-only 기준, frozen readout 소유자 |
| Q | SSR-noFFP 8GPU e12 | P와 config·seed·batch·epoch가 같은 통제 run |
| X1 | PARA-SSR non-staging e60 | planning+det+motion+map 공동 학습 |
| X2 | PARA-SSR stage 1 e48 | det+motion+map, planning gradient 0 |
| X3 | PARA-SSR stage 2 e12 | stage 1 이후 전체 task 공동 학습 |

P와 Q는 model config와 seed가 같고 데이터 분배 순서만 다르다. 따라서
Q는 정상적인 동일 레시피 재실행에서 BEV/readout 호환성이 어느 정도
유지되는지 보기 위한 null control이다. 다만 seed까지 같아 진짜
run-to-run 분산을 과소평가한다.

### 0.3 분석 파이프라인 위생 검사

Frozen swap 전에 다음을 검증했다.

1. 모든 checkpoint의 BEV shape가 `[B,10000,256]`이다.
2. BEV 범위는 `x=[-15,15]m`, `y=[-30,30]m`로 같다.
3. Planning readout 68개 tensor의 key와 shape가 호환된다.
4. P encoder에서 뽑은 BEV를 분리한 P readout에 넣으면 공식 평가값
   `L2@1/2/3s = 0.2385/0.6454/1.3737`, `n=5119`를 소수점 4자리까지
   재현한다.
5. Scene을 처음부터 chronological replay하고 모델별 `prev_bev` chain과
   `can_bus` delta를 독립적으로 유지했다.

즉 이후 swap 열화가 잘못 자른 readout이나 temporal state 오염 때문에
생긴 것은 아니다.

### 0.4 Raw frozen-readout swap

각 모델 encoder의 BEV를 P의 얼린 readout에 그대로 입력했다. Feature
통계는 held-out 4,818 frame, L2는 full 3초 미래가 있는 held-out 4,098
frame에서 계산했다.

| BEV source | RMS | raw cos | raw CKA | swap L1 | L2@3s |
|---|---:|---:|---:|---:|---:|
| P | 0.9895 | 1.0000 | 1.0000 | 0.1438 | 1.360 |
| Q | 0.9895 | 0.7787 | 0.9544 | 0.1579 | 1.467 |
| X1 non-staging | 0.5611 | 0.0062 | 0.2300 | 0.7231 | 7.554 |
| X2 stage 1 | 0.6132 | -0.0029 | 0.1382 | 1.0025 | 11.164 |
| X3 stage 2 | 0.5594 | 0.0014 | 0.3500 | 0.9381 | 10.379 |

Raw 표만 보면 PARA BEV는 P와 거의 직교하고 planning이 완전히 망가진 것처럼
보인다. 그러나 X1은 자기 planner로 평가하면 P보다 약 11~12%만 나쁘다.
`+11~12%`인 모델이 남의 planner에서는 5배 이상 나빠지는 모순이 발생한다.

이 모순은 raw swap이 다음 두 항을 섞어 재기 때문이었다.

\[
\Delta L_{raw}
=\text{정보 차이}+\text{frozen readout 비호환}
\]

### 0.5 정보량을 아는 control

Raw swap이 무엇을 재는지 확인하기 위해 P BEV로 두 극단을 만들었다.

| Control | 실제 planning 정보 | raw swap L1 | 선형 정렬 후 | held-out R² |
|---|---:|---:|---:|---:|
| `perm(B_P)` | 100% | 1.004 | **0.116** | 1.000 |
| channel-stat matched noise | 0% | 0.864 | **0.864** | -0.105 |

P의 256채널 순서만 바꾼 `perm(B_P)`는 정보를 한 비트도 잃지 않았지만
raw planner loss가 native `0.116`에서 `1.004`로 8.7배 나빠졌다. 반면
선형 adapter는 정확히 원래 성능으로 복원했다.

Noise는 같은 수의 adapter parameter를 사용해도 전혀 복원되지 않았다.
따라서 다음이 실측으로 확인됐다.

> Raw frozen-readout swap이 나쁘다는 사실은 BEV 정보 손실의 증거가 아니다.
> Adapter가 held-out에서도 복구하는지를 함께 봐야 한다.

### 0.6 셀 단위 선형 stitching

동일 sample·동일 BEV 셀의 X feature에서 P feature를 예측하는 ridge map을
calibration 30 scene에서 구하고, 별도 120 scene에서 평가했다.

\[
\phi(B_X)_{i,:}=(B_X(i,:)-\mu_X)W+\mu_P,
\quad W\in\mathbb{R}^{256\times256}
\]

모든 10,000개 셀에 같은 `W`를 적용하므로 다른 공간 셀의 정보를 옮길 수
없다. `W`는 feature MSE만 보고 planning GT나 planning loss를 보지 않는다.

| source | held-out R² | raw ΔL1 | aligned ΔL1 | raw gap 제거 | aligned L2@3s |
|---|---:|---:|---:|---:|---:|
| Q | 0.7872 | +0.0141 | -0.0039 | 127.5% | 1.336 |
| X1 | 0.5031 | +0.5899 | +0.0338 | **94.3%** | 1.736 |
| X2 | 0.4466 | +0.8757 | +0.1176 | **86.6%** | 2.756 |
| X3 | 0.5066 | +0.8114 | +0.0351 | **95.7%** | 1.742 |

정렬 후 cos/CKA도 크게 회복됐다.

| source | cos raw→aligned | CKA raw→aligned |
|---|---:|---:|
| Q | 0.7787 → 0.9121 | 0.9544 → 0.9684 |
| X1 | 0.0062 → 0.7798 | 0.2300 → 0.8428 |
| X2 | -0.0029 → 0.7509 | 0.1382 → 0.7576 |
| X3 | 0.0014 → 0.7815 | 0.3500 → 0.8552 |

이 결과는 PARA BEV 안에서 P planner가 쓰는 신호가 상당 부분 선형적으로
접근 가능하다는 뜻이다. 하지만 여기서 곧바로 “94%가 순수 좌표계
차이”라고 부르면 안 된다. 일반 선형 map은 네트워크의 정확한 대칭군보다
강하기 때문이다.

### 0.7 최초 해석의 정정: 단순 순열/gauge가 아니다

초기 보고서는 “aux 학습이 BEV 채널을 순열·회전했고 gauge 자유도 때문에
생겼다”고 설명했다. 후속 nested-family 실험이 이 설명을 반증했다.

| source | diagonal | permutation+scale | full linear | 최적 1:1 평균 `|corr|` |
|---|---:|---:|---:|---:|
| Q | -334.4% | -334.4% | 127.5% | 0.735 |
| X1 | -7.6% | **2.3%** | **94.3%** | 0.166 |
| X2 | 26.0% | **29.3%** | **86.6%** | 0.156 |
| X3 | 22.6% | **34.9%** | **95.7%** | 0.191 |

BEV 채널 순열과 이에 대응하는 readout weight 순열이 정확한 대칭이라는
사실 자체는 `gauge_demo.py`로 확인했다. 그러나 실제 X↔P 차이는 그
대칭군 안에 있지 않다. 순열+scale은 X1에서 2.3%밖에 회수하지 못하고,
일반 선형으로 넘어갈 때 94.3%가 회수된다.

따라서 최신의 안전한 표현은 다음이다.

> X BEV에서 P BEV의 planning-relevant 부분을 동일 셀 일반 선형 map으로
> 상당 부분 예측할 수 있다. 이 map은 단순 채널 대응이나 네트워크의
> 정확한 gauge symmetry보다 큰 함수족이다.

다음 주장은 폐기한다.

- “Aux가 BEV 채널 순서를 바꿨다.”
- “관측된 차이는 permutation symmetry 때문이다.”
- “94%가 순수 좌표계 차이다.”

### 0.8 자기 planner 및 auxiliary 성능

X1이 자기 planner를 사용할 때 최종 planning 차이는 frozen raw swap보다
훨씬 작다.

| protocol | P | X1 final | 상대 차이 |
|---|---:|---:|---:|
| VAD L2 avg | 0.3856 | 0.4332 | +12.3% |
| UniAD L2 avg | 0.7526 | 0.8374 | +11.3% |

즉 다음 세 문장이 동시에 참이다.

1. X1 BEV는 P planner가 그대로 읽으면 5배 이상 나쁘다.
2. 셀 단위 full-linear adapter를 넣으면 격차 대부분이 사라진다.
3. X1의 실제 학습된 planner를 쓰면 P보다 약 11~12% 나쁘다.

초기 e38 checkpoint 평가에서는 planning L2가 나빠지는 동시에 box
collision이 `0.371% → 0.182%`로 절반 이하가 됐다. 이 값은 final e60
수치가 아니라는 점에 주의해야 하지만, auxiliary task를 단순히
“쓸모없다”고 부를 수 없고 trajectory L2와 collision 사이의 trade-off가
있음을 보여준다.

Stage 1 X2의 planner는 학습되지 않았으므로 X2 자체 planning 출력은 모델
품질 지표로 사용하면 안 된다. X2는 planning gradient 없이 학습한 BEV에
P 정보가 얼마나 남는지를 보는 representation control이다.

### 0.9 Integrated Gradients 공간 분석

Raw 차이와 linear alignment 후 남은 잔차를 P planner loss에 대해
Integrated Gradients로 분해했다. 영역은 서로 겹치지 않게
`obj_near > obj_far > route > map > background` 우선순위를 적용했다.

| 영역 | 평균 셀 | 면적 비율 |
|---|---:|---:|
| 가까운 object | 137 | 1.37% |
| 먼 object | 179 | 1.79% |
| ego 3초 route | 184 | 1.84% |
| vector map 주변 | 1,854 | 18.54% |
| background | 7,646 | 76.46% |

핵심 결과는 다음과 같다.

1. X1/X2/X3의 attribution 비율은 거의 영역 면적에 비례했다.
2. Object 전체는 약 3.2%의 attribution을 차지했고 면적도 약 3.16%였다.
3. Map은 약 16~19%, background는 약 75~81%로 null control Q와 유사했다.
4. X의 모든 영역에서 부호 있는 합은 대체로 양수, 즉 P planner loss를
   증가시키는 방향이었다. “object에서는 도움, background에서 손해”라는
   가설은 지지되지 않았다.
5. 정렬 후 attribution 절댓값의 약 70~75%가 상위 5% 셀에 집중됐지만,
   그 셀은 정의한 object/map/route 영역과 일치하지 않았다.

따라서 이 IG 결과는 object 중심 spatial KD 가중치를 정당화하지 않는다.
이는 object 정보가 planning에 중요하지 않다는 뜻이 아니라, 현재 정의와
protocol에서 aux residual이 GT object/map 영역에 특별히 몰린다는 증거를
찾지 못했다는 뜻이다.

### 0.10 이전 분석이 KD 설계에 준 결론

1. Teacher/student raw feature L2는 표현 좌표계와 readout 호환성을
   불필요하게 강제할 수 있다.
2. Learned projection adapter 또는 CKA·correlation·relational loss처럼
   좌표계에 덜 민감한 목적함수가 필요하다.
3. Student planning BEV 자체를 teacher 좌표계로 직접 미는 대신 별도
   projected view를 사용하는 편이 안전하다.
4. Spatial object weighting은 현재 IG 결과로 정당화되지 않는다.
5. Planning L2만 보지 말고 collision과 task-specific teacher metric도 함께
   봐야 한다.

### 0.11 아직 남아 있는 한계

1. P와 X1/X2/X3는 aux 유무뿐 아니라 epoch·learning rate·schedule도 다르다.
   따라서 표현 변화가 **aux 때문에 발생했다는 인과관계는 이 보고서만으로
   증명되지 않았다.**
2. 각 arm은 사실상 seed 1개이고 P/Q는 같은 seed라 run 분산을
   과소평가한다.
3. SSR+nuScenes 단일 architecture/dataset 결과다.
4. Full-linear map의 recovery는 선택한 함수족으로 제거 가능한 몫이며,
   정확한 gauge component의 상한이다.
5. IG가 집중된 상위 셀의 의미는 아직 규명하지 못했다.

### 0.12 외부 teacher 분석으로 이어지는 이유

이전 분석은 같은 SSR 계열 안에서는 raw channel이 거의 직교하더라도
P planning-relevant 표현이 일반 선형으로 상당 부분 복구될 수 있음을
보였다. 다음 질문은 외부 teacher도 같은가였다.

| source | held-out feature R² | full-linear planner gap recovery |
|---|---:|---:|
| X1 PARA non-staging | 0.5031 | 94.3% |
| X2 PARA stage 1 | 0.4466 | 86.6% |
| X3 PARA stage 2 | 0.5066 | 95.7% |
| **BEVFusion teacher** | **0.0975** | **26.1%** |
| **MapTRv2 teacher** | **0.1126** | **25.7%** |

이 차이가 아래 teacher 분석의 핵심이다. 외부 teacher는 단순히 “SSR보다
조금 더 다른 기저”가 아니라, 현재 tap과 동일 셀 선형 함수족에서는 P의
temporal planning 표현과 훨씬 약하게 연결된다.

---

## 1. 외부 teacher 분석 대상

### Target P

- checkpoint: `work_dirs/ssr_noffp_2gpu_b4/epoch_12.pth`
- 입력: 카메라 temporal sequence
- target feature: BEVFormer encoder의 `bev_embed [10000,256]`
- readout: P의 frozen `navi_se → TokenLearner → decoder → trajectory`

### Detection teacher

- BEVFusion camera+LiDAR nuScenes detection model
- cache: `/data1/yong/teacher_cache/bevfusion_cache/cache_val_100x100`
- tap: fuser output, decoder backbone 이전
- native ROI: 약 `[-54,54]m × [-54,54]m`
- SSR ROI로 crop한 뒤 `100×100` bilinear resize

### Mapping teacher

- MapTRv2 camera-only nuScenes mapping model
- cache: `/data1/yong/teacher_cache/maptrv2_cache/cache_val_100x100`
- tap: `pts_bbox_head`의 BEVFormer encoder `bev_embed`
- ROI: SSR과 동일한 `x=[-15,15]m`, `y=[-30,30]m`
- cached axis를 SSR `[y,x,C]` 순서로 transpose

두 cache 모두 val 6,019개 sample 전체를 포함하고 stored feature shape는
`[10000,256]`이다.

---

## 2. 실험 설계

### 2.1 Scene 분리

기존 SSR↔PARA stitching 분석과 동일하게 scene 단위로 분리했다.

| split | scene | frame | 용도 |
|---|---:|---:|---|
| calibration | 30 | 1,201 | 축 후보 선택, 선형 map 적합 |
| held-out | 120 | 4,818 | cosine, CKA, feature R² |
| held-out planning subset | 120 | 818 | full 3초 미래가 있고 cache stride에 포함된 frame |

동일 scene의 인접 frame이 calibration과 held-out 양쪽에 들어가지 않는다.
P encoder는 각 scene을 처음부터 순서대로 replay하여 `prev_bev`를 보존했다.

### 2.2 선형 통역기

동일 sample, 동일 BEV 셀의 teacher feature `x∈R²⁵⁶`과 P feature
`y∈R²⁵⁶`를 짝지어 다음 ridge 문제를 풀었다.

\[
\phi(x)=(x-\mu_x)W+\mu_y
\]

\[
W=(S_{xx}+\lambda I)^{-1}S_{xy},\qquad
\lambda=10^{-4}\,\mathrm{mean}(\mathrm{diag}(S_{xx}))
\]

- 모든 10,000개 셀에 동일한 `W [256,256]` 적용
- 다른 셀의 정보를 가져오는 spatial 연산 없음
- planning loss와 trajectory GT를 적합에 사용하지 않음
- calibration의 teacher/P feature MSE만 사용
- 충분통계는 전량 frame에서 FP64로 누적

### 2.3 Nested family와 control

| family | 설명 | parameter |
|---|---|---:|
| identity | teacher BEV 그대로 P planner에 입력 | 0 |
| mean-only | 입력을 무시하고 calibration P 채널 평균을 모든 셀에 출력 | 256 |
| diagonal | 같은 번호 채널의 scale+bias만 적합 | 512 |
| permutation+scale | 최적 1:1 채널 대응과 scale+bias | 512 |
| full linear | teacher 256채널 전체로 각 P 채널 예측 | 65,792 |
| roll-control full | teacher grid를 `(17,23)`셀 roll한 뒤 별도 full map 적합 | 65,792 |

`mean-only`는 독립 noise를 무한히 많이 사용해 P feature를 회귀했을 때의
극한과 같다. 입력과 P의 cross-covariance가 0이면 `W→0`이고 target 평균만
남기 때문이다.

`roll-control`은 feature의 채널 통계와 scene 내용은 그대로 보존하지만
동일 셀 대응을 고의로 파괴한다. 올바른 정렬이 이 control보다 좋아야
위치별 의미 관계가 있다고 해석할 수 있다.

---

## 3. 공간축 정렬 audit

축 후보는 **calibration R²만으로 선택**했다. Held-out R²는 선택 후의
독립 검증값이다.

| source | alignment candidate | calibration R² | held-out R² | raw cos | raw CKA |
|---|---|---:|---:|---:|---:|
| BEVFusion | **기존 visualizer 정렬** | **0.1218** | **0.0975** | 0.0049 | 0.0128 |
| BEVFusion | visualizer + lateral flip | 0.1202 | 0.0958 | 0.0054 | 0.0122 |
| BEVFusion | native manifest 물리축 해석 | 0.0828 | 0.0603 | 0.0077 | 0.0162 |
| BEVFusion | manifest, lateral flip 없음 | 0.0834 | 0.0605 | 0.0080 | 0.0165 |
| BEVFusion | roll-control | 0.1151 | 0.0909 | 0.0042 | 0.0301 |
| MapTRv2 | **transpose, flip 없음** | **0.1497** | **0.1126** | 0.0028 | 0.0376 |
| MapTRv2 | lateral flip | 0.1425 | 0.1046 | 0.0012 | 0.0355 |
| MapTRv2 | longitudinal flip | 0.1336 | 0.0976 | 0.0044 | 0.0385 |
| MapTRv2 | 양축 flip | 0.1332 | 0.0970 | 0.0047 | 0.0391 |
| MapTRv2 | roll-control | 0.1167 | 0.0813 | 0.0074 | 0.0435 |

두 source 모두 기존 시각화에서 사용한 정렬이 calibration과 held-out에서
가장 높은 R²를 보였다. 따라서 후속 planner 표에는 이 정렬을 사용했다.

그러나 절대 R²는 BEVFusion `0.0975`, MapTRv2 `0.1126`으로 낮다. P feature
분산의 약 90%는 동일 셀 일반 선형 map으로도 설명하지 못한다.

또한 raw CKA가 BEVFusion `0.0128`, MapTRv2 `0.0376`으로 매우 낮다.
이전 SSR↔PARA 분석의 raw CKA `0.14~0.23`, full-linear gap recovery
`86.6~95.7%`와 비교해도 teacher 표현은 P에서 훨씬 멀다.

---

## 4. Frozen P planner 결과

모든 행은 동일한 held-out 818 frame에서 평가했다. `L1`은 commanded mode의
mask-weighted displacement L1이고, `L2@3 MAX`는 3초 endpoint error,
`L2avg VAD`는 VAD식 horizon-average이다.

| source | family | L1 | P 대비 ΔL1 | raw gap 제거 | L2@3 MAX | L2avg VAD |
|---|---|---:|---:|---:|---:|---:|
| **P** | native | **0.1561** | — | — | **1.364** | **0.380** |
| BEVFusion | identity | 0.9143 | +0.7582 | 0.0% | 10.156 | 4.298 |
| BEVFusion | mean-only | 0.8199 | +0.6638 | 12.5% | 9.258 | 3.843 |
| BEVFusion | diagonal | 0.8185 | +0.6624 | 12.6% | 9.241 | 3.835 |
| BEVFusion | permutation+scale | 0.8162 | +0.6601 | 12.9% | 9.212 | 3.820 |
| BEVFusion | **full linear** | **0.7162** | **+0.5601** | **26.1%** | **7.995** | **3.287** |
| BEVFusion | roll-control full | 0.7133 | +0.5572 | 26.5% | 7.957 | 3.284 |
| MapTRv2 | identity | 0.9558 | +0.7996 | 0.0% | 10.245 | 4.128 |
| MapTRv2 | mean-only | 0.8199 | +0.6638 | 17.0% | 9.258 | 3.843 |
| MapTRv2 | diagonal | 0.8192 | +0.6631 | 17.1% | 9.251 | 3.838 |
| MapTRv2 | permutation+scale | 0.8171 | +0.6610 | 17.3% | 9.226 | 3.829 |
| MapTRv2 | **full linear** | **0.7499** | **+0.5937** | **25.7%** | **8.397** | **3.500** |
| MapTRv2 | roll-control full | 0.7711 | +0.6150 | 23.1% | 8.655 | 3.593 |

P native 수치는 기존 분리 readout 재현값과 일관된다. 표본 차이 때문에
전체 held-out의 값과 소폭 다르지만, P readout 연결이 깨진 신호는 없다.

---

## 5. 해석

### 5.1 BEVFusion

1. Raw P planner는 사실상 동작하지 않는다. L1 `0.9143`, L2@3 `10.156m`다.
2. Full map도 L1을 `0.7162`까지만 낮춘다. P native의 **4.59배**다.
3. L2@3도 `7.995m`로 P의 **5.86배**다.
4. Mean-only가 이미 raw gap의 `12.5%`를 제거한다. Full의 순증분은
   mean-only 대비 `13.7%p`다.
5. 하지만 roll-control full이 `26.5%`로 올바른 정렬 `26.1%`보다 오히려
   약간 높다. 이 차이는 동일 셀 의미 관계를 지지하지 않는다.
6. 최적 1:1 채널 대응의 평균 `|corr|`도 `0.086`에 불과하다.

**판정:** BEVFusion과 P 사이에 일부 scene/global 통계의 선형 예측성은
있지만, 올바른 공간 셀끼리 공유하는 planning 표현의 증거는 매우 약하다.
현재 tap과 `1×1` adapter로 BEVFusion BEV를 P BEV 대신 사용할 수 없다.

### 5.2 MapTRv2

1. Raw P planner L1은 `0.9558`, L2@3는 `10.245m`다.
2. Full map 후 L1 `0.7499`, L2@3 `8.397m`로 여전히 P의 각각
   **4.80배**, **6.15배**다.
3. Mean-only가 raw gap의 `17.0%`를 제거하고 full은 `25.7%`를 제거한다.
   실제 input-dependent 순증분은 `8.8%p`다.
4. 올바른 정렬은 roll-control보다 feature R²가 `3.13%p`, planner gap
   recovery가 `2.66%p` 높다. 위치별 관계 신호가 아예 0은 아니지만 작다.
5. 최적 1:1 채널 대응 평균 `|corr|`은 `0.115`다.

**판정:** MapTRv2가 BEVFusion보다 동일 셀의 P feature와 조금 더 관계가
있지만, P temporal planning 표현을 직접 복원할 수준은 아니다.

### 5.3 Mean-only가 중요한 이유

두 source의 mean-only 출력은 동일한 P calibration 평균이므로 planner
결과도 정확히 같다(`L1=0.8199`). 그런데 raw teacher 입력이 워낙 P 분포와
달라서, teacher 정보를 전혀 사용하지 않는 평균 출력만으로도 표면상
12.5~17.0%가 "회복"된다.

따라서 full-linear `25.7~26.1%`를 전부 teacher의 유용한 정보로 세면 안
된다. 최소한 mean-only와 spatial control을 빼고 읽어야 한다.

---

## 6. KD 설계에 대한 결론

### 이 결과가 반대하는 설계

1. `teacher BEV → P planning head` 직접 연결
2. raw `||B_teacher-B_student||²`
3. teacher→P 셀별 `1×1` adapter만 학습하면 planning 표현이 대부분
   호환될 것이라는 가정

### 여전히 가능한 설계

1. **Planning BEV를 유지한 projected view KD**
   - student core BEV 자체를 teacher 좌표계로 강제하지 않음
   - teacher별 별도 projection branch에서 distillation
   - core로 전달되는 KD gradient는 valve/scale로 통제
2. **Task-specific distillation**
   - BEVFusion: objectness, box/query, occupancy 또는 detection relation
   - MapTRv2: vector-map query, map logits/points, map spatial relation
   - teacher BEV 전체를 P BEV로 번역하기보다 teacher가 잘하는 정보를 전달
3. **공간 adapter**
   - `1×1`보다 `3×3 conv`, deformable alignment 또는 multi-layer adapter
   - 특히 BEVFusion의 crop/resize와 서로 다른 native resolution을 보정
4. **좌표계 무관 손실**
   - relational KD, attention transfer, local correlation/CKA 계열

이 대안들은 이번 실험으로 성능 향상이 증명된 것은 아니다. 이번 결과는
직접 feature 교체와 per-cell linear translation이 충분하지 않다는 음성
증거다.

---

## 7. 한계

1. BEVFusion은 native ±54m feature를 SSR ROI로 crop·resize한다. 원본
   resolution과 두 번의 resampling이 관계성을 낮출 수 있다.
2. feature tap의 추상화 단계가 다르다. BEVFusion은 fuser 직후,
   MapTRv2/P는 BEVFormer encoder 출력이다.
3. P는 temporal BEV이고 teacher cache는 해당 sample의 task feature다.
   동일 frame이라도 정보의 시간 범위가 다르다.
4. adapter는 동일 셀 `1×1` 선형 map에 한정된다. 비선형·공간 adapter의
   가능성을 부정하지 않는다.
5. feature 통계는 held-out 4,818 frame 전량이지만 planner 결과는 stride
   cache 중 full 3초 미래가 있는 818 frame이다.
6. calibration은 val의 분리된 30 scene을 썼다. 논문 최종 수치는 train에
   fit하고 val 전체에서 test하는 것이 더 깨끗하다.
7. MapTRv2의 correct-vs-roll 차이 `2.66%p`에는 paired bootstrap 신뢰구간을
   아직 계산하지 않았다. 작은 양성 신호를 유의한 효과로 단정하지 않는다.

---

## 8. 산출물과 재현

### 결과

- `/data1/yong/SSR/work_dirs/analysis/teacher_stitch/results.json`
- `/data1/yong/SSR/work_dirs/analysis/teacher_stitch/summary.pt`
- `/data1/yong/SSR/work_dirs/analysis/teacher_stitch/shard_{0..5}.pt`
- `/data1/yong/SSR/work_dirs/analysis/teacher_stitch/cache_{0..5}.pt`

### 코드

- `tools/teacher_stitch_worker.py`
- `tools/teacher_stitch_reduce.py`
- `tools/run_teacher_stitch_shards.sh`

### 실행 방식

```bash
# GPU 5 lane: shards 0,2,4
tools/run_teacher_stitch_shards.sh 5 0

# GPU 7 lane: shards 1,3,5
tools/run_teacher_stitch_shards.sh 7 1

CUDA_VISIBLE_DEVICES=5 NUMBA_CPU_NAME=generic \
  /home/yongjae/miniconda3/envs/ssr/bin/python \
  tools/teacher_stitch_reduce.py \
  --dir /data1/yong/SSR/work_dirs/analysis/teacher_stitch
```

Teacher cache manifest SHA256:

- BEVFusion: `2d0d15fb884ee7586667b90a77f0097a953c92ea1d99117d62363d4e1b3659a7`
- MapTRv2: `d02f932352f123b53515a5773bff46fc68dfa85ce7d14a43f36ec0c7d3a5c77d`
