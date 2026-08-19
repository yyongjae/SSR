# 보고서 #07 — 2차 코드 리뷰 검토 및 수정

**날짜** 2026-08-18
**대상** 60 epoch 본 실험 착수 전 PARA-SSR ver2
**결론** 리뷰 8개 항목 중 실질적 지적 **8개 전부 유효**. 3개는 영향 크기가 과장.
전부 수정하고 회귀 테스트를 붙였다.

리뷰 원문은 진단 지표의 **의미**를 문제 삼았다. 코드가 죽는 종류의 버그가 아니라
"이 숫자를 보고 head가 살아있다고 판단하면 틀린다"는 종류다. 60 epoch = 5~6일이
걸리는 실험을 이 지표들로 감시할 참이었으므로 착수 전에 고치는 게 맞다.

---

## 요약표

| # | 지적 | 판정 | 수정 |
|---|---|---|---|
| 1 | `aux_grad_scale` 문서/config 불일치 | ✅ | 문서 정정 |
| 2 | map 진단 지표 3종 해석 오류 | ✅ | 지표 재설계 + 회귀 테스트 |
| 3 | `occ_sep`가 invalid cell을 분모에 남김 | ✅ | 마스킹 수정 |
| 4 | `analyze_gshare.py` 재구성 로직이 본 설정에서 깨짐 | ✅ | 중복행 제거 + epoch 리셋 |
| 5 | global clipping이 aux/planner를 결합 | ✅ / ⚠️ 과장 | 계측 훅 추가 (값은 유지) |
| 6 | 회귀 테스트가 통합 테스트가 아님 | ✅ | exit code + 실제 2-rank 통합 테스트 신규 |
| 7 | 논문 관계 서술 부정확 | ✅ / ⚠️ 일부 | 주석·문서 정정 |
| 8 | det threshold / occ 메모리 / EMA resume / 60ep 비교 / 2-GPU eval | ✅ (2-GPU만 ⚠️) | 전부 반영 |

---

## 1. map 진단 지표 — 재설계

### 1.1 `map_pos_frac`은 모델 지표가 아니라 데이터 통계였다

`MapHungarianAssigner3D`에는 cost 임계값이 없다:

```python
# core/bbox/assigners/map_hungarian_assigner_3d.py:157-159
assigned_gt_inds[:] = 0
assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
```

`linear_sum_assignment`은 언제나 `min(Q, N_gt)`쌍을 반환하고, 그중 버려지는 쌍은
없다. 따라서

```
map_pos_frac  ≡  num_total_pos / (B·Q)  ≡  ΣN_gt / (B·100)
```

**예측이 무엇이든 값이 같다.** 완벽한 예측과 완전히 붕괴한 예측이 동일한 숫자를
낸다. "쿼리 붕괴를 잡는다"는 초판 서술은 성립하지 않는다.

**조치:** 폐기. 같은 값을 `map_n_gt`(이미지당 GT 개수)로 이름만 정직하게 남겼다 —
파이프라인 sanity check로는 여전히 쓸모가 있다.

### 1.2 붕괴 감지를 실제로 하는 지표: `map_spread_m`

쿼리 100개가 예측한 폴리라인 **중심점의 표준편차**를 미터로 잰다. 모든 쿼리가 한
선으로 붕괴하면 0으로 떨어지고, 다른 어떤 지표도 이걸 보지 못한다.

검증 (`tools/verify_aux_metrics.py`):

```
spread: 정상 5.420 m   붕괴 0.000 m
```

### 1.3 `map_pts_err_m`은 축별 MAE였다

```python
err = (pts_flat - target).abs() * scale * pts_weights   # |dx|, |dy| 각각 평균
```

이건 유클리드 거리가 아니다. 측정:

| 실제 변위 | 구 지표 | 실제 | 과소보고 |
|---|---:|---:|---:|
| x축 0.75 m | 0.375 | 0.750 | **2.00×** |
| y축 0.75 m | 0.375 | 0.750 | **2.00×** |
| 45° 0.75 m | 0.530 | 0.750 | **1.41×** |

`delta.norm(dim=-1)`로 수정. 수정 후 세 방향 모두 0.7500을 보고한다.

> 이건 **지난번 map/occ shift 대조군에서 내가 저지른 것과 같은 실수**다. 그때
> 고친 코드 바로 옆에 같은 버그를 남겨뒀다. ([#05](05_ver2_changes.md) §4 참조)

**남는 한계 — 리뷰어가 짚은 것보다 하나 더 있다.** 유클리드로 고쳐도 chamfer와
같지 않다. 이건 매칭이 고른 shift 순열 기준 **index 정렬** 거리이고 chamfer는
최근접점 거리다. 따라서 `map_pts_err_m`은 **chamfer의 상한**이며, "1.5 m 미만이면
chamfer 임계값 통과"라는 판정에는 쓸 수 없다. 추세 지표로만 읽어야 한다.

### 1.4 `map_cls_acc`의 사각지대

매칭된 positive 쿼리만 본다. background에 confident FP가 쏟아져도, 전체 score가
가라앉아도 정확도는 유지될 수 있다. `map_conf_pos` / `map_conf_neg`를 추가했다 —
매칭/미매칭 쿼리의 평균 최대 확신도. 건강한 head는 벌어지고, 죽은 head는 붙는다.

### 1.5 "Hungarian 재사용" 서술은 틀렸다

`train_metrics()`는 `map_get_targets`를 **새로** 호출한다
([para_map_head.py:563](../projects/mmdet3d_plugin/SSR/dense_heads/para_map_head.py#L563)).
`loss()`가 decoder 3층에 대해 또 호출하므로 총 4회다.
**추가 network forward가 없다는 것만 맞다.** docstring과 보고서 #06을 정정했다.
`aux_metric_log_interval=200` 게이팅은 이 비용 때문에 유지한다.

### 최종 지표 세트

| 지표 | 잡는 실패 모드 |
|---|---|
| `map_cls_acc` | 점은 맞는데 라벨이 틀림 |
| `map_conf_pos` / `map_conf_neg` | 확신도 붕괴, background FP 홍수 |
| `map_spread_m` | **쿼리 붕괴** |
| `map_pts_err_m` (유클리드) | 점 회귀 정체 |
| `map_n_gt` | 데이터 파이프라인 이상 |

---

## 2. `occ_sep` — invalid cell이 분모에 남아 있었다

```python
# 수정 전
probs = probs * v[...]          # invalid → 0
gt    = gt & v[...]             # invalid → False
pos, neg = gt, ~gt              # ← invalid가 neg에 들어감
p_off = (probs*neg).sum() / neg.sum()
```

invalid cell은 분자에서 0이지만 **분모에는 그대로 남는다.** p_off가 아래로
희석되고 separation ratio가 부풀려진다.

합성 검증 (상수 출력 head, 참값 p_on=0.80 / p_off=0.30 / ratio=2.667):

```
수정 전 : p_on 0.8000  p_off 0.1500  ratio 5.3333   ← 2배 부풀림
수정 후 : p_on 0.8000  p_off 0.3000  ratio 2.6667
```

IoU는 무사했다 — 0으로 눌린 cell은 intersection·union 양쪽에서 빠지므로.
`if v.any()` 로 마스킹을 통째로 건너뛰던 분기도 제거했다.

**함의.** ver2 probe에서 `occ_sep/ratio` 0.9994 → 1.0079를 보고 "여전히 상수
출력"이라고 판단했는데, 이 지표가 **위쪽으로** 편향돼 있었으므로 실제 분리도는
1.0079보다 **더 나쁘다.** 결론 방향은 그대로이고 근거만 강해졌다.

`occ_sep/valid_frac`도 추가했다 — 지표가 배치의 몇 %에서 계산됐는지.

---

## 3. `analyze_gshare.py` — 본 설정에서 재구성이 깨진다

두 가지가 겹쳤다.

1. **로깅이 측정보다 잦다.** `grad_norm_log_interval=200`,
   `log_config.interval=100`. iter 300·500에는 직전 누적평균이 그대로 다시
   찍히는데, 스크립트는 이걸 새 측정으로 세서 `k`가 실제보다 앞서 나간다.
2. **epoch 리셋 없음.** `log_buffer.clear()`가 epoch마다 누적평균을 초기화하는데
   `k`는 계속 증가한다.

합성 로그(측정 200 / 로깅 100 / 2 epoch)로 재현:

```
구 스크립트 :   5.10   5.10   6.60   5.60   8.10   0.70   5.20   9.20   5.70  10.70
정답        :   5.10   5.10   6.10   6.10   7.10   5.20   5.20   6.20   6.20   7.20
```

epoch 경계에서 5.20 → **0.70**, 끝에서 7.20 → **10.70**. 리뷰어가 본 음수값과
같은 현상이다.

**수정:** 행 단위 중복 제거(모든 키를 함께 보므로 시리즈 정렬이 유지된다) +
epoch별 `k` 리셋 + 재구성값이 [0,1]을 벗어나면 경고. 수정 후 같은 합성 로그에서
정답과 자릿수까지 일치한다.

**기존 측정은 유효하다.** ver2 probe는 측정·로깅 간격이 둘 다 25였고 단일
epoch이었다 — 두 버그가 전부 무해해지는 조건이다. `plan 0.83% / det 42% /
map 55% / occ 1.8%`는 그대로 쓸 수 있다.

덤으로 `--epoch` 옵션을 넣었다. epoch을 섞어 평균내면 움직이는 표적을 재게 된다
(v1의 planning share는 epoch 3에 정점을 찍고 절반으로 떨어졌다).

---

## 4. Global gradient clipping — 사실이고, 크기는 작다

`optimizer_config = dict(grad_clip=dict(max_norm=35))`는 **전체 파라미터**에
한꺼번에 걸린다. 완료된 두 run의 로그에서 실제 발생률을 측정했다:

| run | grad_norm 평균 | >35 비율 |
|---|---:|---:|
| `ssr_noffp_2gpu_b4` (plan only) | 0.93 | **0.0%** |
| `para_ssr_8gpu` (v1, aux 포함) | 34.07 | **56.2%** |

epoch별:

| ep | 1 | 3 | 5 | 6 | 8 | 10 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 평균 norm | 21.7 | 30.3 | 34.1 | 35.7 | 37.7 | 38.6 | 38.5 |
| >35 비율 | 0% | 0% | 20% | 77% | 97% | **100%** | **100%** |

**리뷰어 말이 맞다.** aux head를 붙이면 norm이 0.93 → 38.5로 뛰고 epoch 10부터는
매 iteration clipping이 걸린다. 나는 이걸 측정한 적이 없었다.

**다만 실질 영향은 작다.** 세 가지 이유:

1. clip 계수가 `35/38.5 ≈ 0.91` — 9% 축소이지 자릿수 왜곡이 아니다.
2. 균일 스케일이라 모든 파라미터의 **업데이트 방향은 불변**이다.
3. optimizer가 AdamW다. `m/√v`는 gradient의 상수배에 근사 불변이므로 균일 축소는
   대부분 흡수된다. 남는 건 clip 계수가 iteration마다 달라지는 부분과 decoupled
   weight decay가 함께 축소되지 않는 부분뿐이다. SGD였다면 리뷰어 서술이 그대로
   맞다.

**리뷰어가 놓친, 더 중요한 점.** `aux_grad_scale`은 이 문제를 **전혀** 완화하지
못한다. `_ScaleGrad`는 `bev_embed`로 흐르는 경로만 줄이고 aux head **파라미터**
gradient는 손대지 않는데, norm 38을 만드는 주범이 바로 그 파라미터들이다. 0.01을
넣어도 clipping 발생률은 그대로다.

**조치:** `max_norm`은 **바꾸지 않았다.** SSR/VAD의 값이고, 바꾸면 plan-only
baseline과의 비교가 깨진다. 대신 계측을 붙였다
([`hooks/clip_monitor.py`](../projects/mmdet3d_plugin/SSR/hooks/clip_monitor.py)):

| 키 | 내용 |
|---|---|
| `clip/rate` | 로깅 구간에서 clipping이 걸린 iteration 비율 |
| `clip/factor` | 실제 적용된 평균 `min(1, max_norm/norm)` |
| `pnorm/{trunk,plan,aux,other}` | clip 이전 모듈군별 gradient norm (200 iter마다) |

`clip/rate`는 **매 iteration** 갱신되므로 구간 평균이 곧 참값이다 —
`gshare/*`와 달리 §3의 재구성이 필요 없다.

### 4.1 첫 측정 — `aux_grad_scale`이 무력한 이유가 숫자로 나왔다

2-rank 학습 smoke run (`aux_grad_scale=0.01` 적용 상태, iteration 10):

```
grad_norm 23.76   clip/rate 0.0000   clip/factor 1.0000
pnorm/trunk  0.0846
pnorm/plan   0.4131
pnorm/aux   24.7853      <-- 전체 norm의 사실상 전부
pnorm/other  0.0000
```

**aux head 파라미터 gradient가 planner의 60배, trunk의 293배다.** `max_norm=35`가
재는 norm은 거의 전적으로 aux head의 것이고, `_ScaleGrad`는 여기에 손을 대지
않는다. `aux_grad_scale`을 아무리 낮춰도 clipping 발생률은 변하지 않는다는
위 주장이 그대로 확인된다.

warm-up 첫 iteration(5)에서는 `grad_norm 58.70 / clip/rate 1.0000 /
clip/factor 0.6503` — 35% 축소가 걸렸다. 즉 clipping은 학습 초반과 후반
(v1 기준 epoch 10+) 양쪽에서 발동한다.

**함의:** `max_norm`을 planner 기준(≈0.4)이 아니라 aux 기준(≈25)으로 다시
생각해야 한다. 선택지는 (a) `max_norm` 상향, (b) param group별 clip, (c) 현행
유지 + 계측. 지금은 (c)이고, 1 epoch 뒤 `pnorm/*` 추이를 보고 결정한다.

**그리고 gshare 자체의 한계.** gradient **크기**의 비율일 뿐 방향·상쇄·충돌은
보지 못한다. `aux_grad_scale=0.01`은 논문 재현값이 아니라 별도 ablation이
필요한 실험값이다 — config 주석에 명시했다.

---

## 5. 논문과의 관계 — 리뷰어 구분을 채택

| 서술 | 판정 |
|---|---|
| PARA-Drive의 **병렬 그래프**를 SSR/VAD head로 재구성했다 | ✅ |
| PARA-Drive **구현 또는 평가 프로토콜**을 재현했다 | ❌ |
| VAD **vector-map / motion 평가 일부**를 재사용했다 | ✅ |
| VAD **planner 구조**를 재현했다 | ❌ |

PARA-Drive는 UniAD 모듈(query 기반 occupancy, track query 기반 motion,
Panoptic-SegFormer map)을 그대로 쓴다. 현재 구현은 VAD식 3-class vector map,
query-free vehicle-only occupancy, frame별 detection query다.

**가장 중요한 지적:** SSR 논문의 핵심 주장은 **supervised perception task를
제거해도 된다**는 것이다. 여기서 aux head를 붙이는 건 SSR 재현이 아니라 **그
주장의 반대 방향 실험**이다. 이건 의도한 것이지만 — KD 실험을 위한 깨끗한 병렬
구조가 목적이므로 — 보고서와 주석이 이를 명시해야 한다.

**"distillation teacher" 문구:** 리뷰어는 부정확하다고 했지만, config 주석은
`"the aux heads are meant to be distillation teachers"` — **의도 서술**이지
구현 주장이 아니다. 다만 오해 소지는 인정해서 "오늘 시점에서는 KD가 구현돼 있지
않고 그냥 auxiliary supervised head"라고 명시했다.

또한 현재 planner는 SSR의 command-SE → 16 sparse tokens → waypoint decoder이며
**aux 출력이 planner에 들어가지 않는다.** 병렬 설계로는 의도한 바이고, VAD가
agent/map query를 planner 입력과 vector constraint에 직접 쓰는 것과는 다르다.

---

## 6. 회귀 테스트 — exit code + 실제 통합 테스트

`verify_multibatch_and_metrics.py`가 실패 목록을 출력만 하고 exit 0으로
끝났다. CI에 걸면 항상 통과한다. `sys.exit(1 if fails else 0)` 추가.

문구도 "ALL P0 FIXES VERIFIED" → "ALL P0 UNIT CHECKS PASS"로 낮췄다. 이건
단위 검증이다 — 분산 eval은 source 문자열을 grep했고, motion·occ는 production
경로 밖에서 재현했다. docstring에 범위를 명시했다.

**그리고 진짜 통합 테스트를 새로 만들었다:** `tools/verify_dist_eval.sh`.
무작위 초기화 가중치로 N개 샘플에 대해 **실제** 경로를 전부 태운다 — 실제
sampler, 실제 `custom_multi_gpu_test`, 실제 `collect_results_cpu`, 실제
`dataset.evaluate()`. 숫자는 의미 없고, **60 epoch 중 epoch 10에서 죽지 않는지**를
5분 만에 확인하는 게 목적이다.

```bash
tools/verify_dist_eval.sh 8 projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py 0,1
```

이걸 돌릴 수 있게 하려고 `evaluate_map`의 GT 정합성 검사를 **위치 일치**에서
**토큰 매칭 후 재정렬**로 바꿨다. subset 평가가 가능해지면서 보장은 오히려
강해졌다(예측된 토큰이 캐시에 없으면 여전히 raise).

---

## 7. 결과 순서 검증 — 새로 추가

리뷰어의 2-GPU 지적을 확인하다가, 이 저장소가 **정합성이 깨져도 조용히 틀린
점수를 내는** 구조라는 걸 발견했다. 모든 formatter가 `results[i]`를
`data_infos[i]['token']`과 **위치로** 짝짓는다. 이게 성립하려면
`distributed_sampler.py`의 연속 블록 분할과 `apis/test.py`의 수집 순서가
일치해야 하는데, 두 파일에 걸친 불변식을 아무도 검사하지 않았다.

확인 결과 현재는 일치한다:

```python
# datasets/samplers/distributed_sampler.py:38  — 연속 블록 (interleave는 주석 처리)
indices = indices[self.rank*per_replicas:(self.rank+1)*per_replicas]

# SSR/apis/test.py:149                          — 같은 가정
#for res in zip(*part_list):
for res in part_list:
    ordered_results.extend(list(res))
```

한쪽만 바뀌면 6019개 예측이 전부 엉뚱한 scene에 채점된다. `ParaSSR`이 결과에
`sample_token`을 찍고, `evaluate()`가 `_assert_token_alignment()`로 검사하도록
했다. 어긋나면 첫 불일치 인덱스와 함께 raise한다.

---

## 8. 개별 항목

### 8.1 detection score threshold

| | SSR (이 저장소) | VAD |
|---|---|---|
| `_format_bbox` / `_format_det_results` | **0.0** | **0.2** |

nuScenes AP는 PR 곡선 적분이라 threshold가 **낮을수록 유리**하다. VAD의 0.2는
json 크기 절감용이고 그 결과 논문 수치도 0.2 기준이다. 0.0이 정직한 숫자이고
0.2가 VAD 대비 like-for-like다.

**조치:** `det_score_thresh`를 dataset 인자로 노출. float 또는 리스트를 받는다.
`det_score_thresh=[0.0, 0.2]`로 두면 두 번 평가해 `.../mAP`와 `.../mAP@0.2`를
모두 보고한다. 기본값은 0.0 (기존 동작 유지).

### 8.2 occupancy 결과 메모리

리뷰어 지적대로 내 주석의 "0.8 GiB"는 **바로 다음 줄에서 내가 직접 넣는 GT를
빼먹은** 수치였다. `formating.py:60`이 GT를 float32로 캐스팅한다.

| | 수정 전 | 수정 후 |
|---|---:|---:|
| 예측 (fp16) | 0.79 GiB | 0.79 GiB |
| GT | 1.57 GiB (float32) | **0.39 GiB** (uint8) |
| 합계 | **2.35 GiB** | **1.18 GiB** |

`gt_occ_valid`도 bool로. `evaluate_occ`는 이미 `.bool()`을 호출하므로 무영향.

### 8.3 EMA resume

`ema.py:93`의 `if self.resume is not None`은 config 값이다. `--resume-from`으로
raw 체크포인트를 복구해도 EMA는 `init_updates`로 리셋된다. 60ep config 주석에
명시했다.

### 8.4 60 epoch은 12 epoch의 연장이 아니다

CosineAnnealing은 `by_epoch`이라 epoch 12 시점의 LR이 두 run에서 완전히 다르다.
"60 epoch의 효과"를 인과적으로 말하려면 **동일한 60 epoch schedule의 baseline**이
필요하다. 60ep config 주석에 명시했다.

### 8.5 2-GPU 평가 — 기전은 맞고 규모는 작다

rank 1이 validation index 3010(scene 중간)에서 시작하는 건 사실이다. 다만
영향받는 건 **6019개 중 rank 1의 첫 1개**(0.017%)다. 그리고 **보고된
0.7526은 이미 1-GPU 평가 결과**다.

epoch 10 in-training eval에만 해당하는 주의사항이며, 60ep config 주석에
"추세용, 최종 수치는 1-GPU EMA"라고 적었다.

---

## 9. 검증

```
$ python tools/verify_aux_metrics.py
=== occ_sep must ignore unsupervised frames on BOTH sides ===
  p_on 0.8000 (want 0.80)   p_off 0.3000 (want 0.30)   ratio 2.6667 (want 2.667)
  constant head ratio 1.0000 (want 1.0000)
  IoU@0.5 1.0000 (want 1.0000)   valid_frac 0.4286 (want 0.4286)

=== map_pts_err_m must be Euclidean, and collapse must be visible ===
  exact match          : err 0.0000 m (want 0.0000)
  shifted x by 0.75 m  : err 0.7500 m (want 0.7500) ok
  shifted y by 0.75 m  : err 0.7500 m (want 0.7500) ok
  shifted 45deg 0.75 m : err 0.7500 m (want 0.7500) ok

=== map_spread_m must detect query collapse (map_pos_frac cannot) ===
  spread: healthy 5.420 m  collapsed 0.000 m
  map_n_gt 4.00 (want 4.00)

=== map_conf_* must separate matched from unmatched queries ===
  conf_pos 0.9997   conf_neg 0.0003

ALL AUX-METRIC CHECKS PASS

$ python tools/verify_multibatch_and_metrics.py
ALL P0 UNIT CHECKS PASS      (exit 0)
```

2-rank 학습 경로 (신규 지표 키가 DDP `all_reduce`를 통과하는지 — 키 집합이 rank마다
다르면 데드락이다):

```
Epoch [1][10/3517]  loss: 57.2266  grad_norm: 23.7626
  clip/rate: 0.0000   clip/factor: 1.0000
  map_cls_acc: 0.4042   map_conf_pos: 0.0250   map_conf_neg: 0.0250
  map_spread_m: 10.1478   map_pts_err_m: 14.2142   map_n_gt: 8.3750
  occ_iou0.5/mean: 0.0349
  occ_sep/p_on: 0.4768  p_off: 0.4767  ratio: 1.0001  valid_frac: 0.8750
  gshare/plan: 0.0410  det: 0.7090  map: 0.1611  occ: 0.0888
  pnorm/trunk: 0.0846  plan: 0.4131  aux: 24.7853
```

초기화 직후 상태로 전부 말이 된다 — `occ_sep/ratio` 1.0001(상수 출력),
`map_conf_pos == map_conf_neg`(미학습), `map_spread_m` 10.1 m(붕괴 아님),
`valid_frac` 0.875. `map_pts_err_m` 14.2 m는 구 지표였다면 대략 절반으로
찍혔을 값이다.

2-rank 분산 평가 경로 (`tools/verify_dist_eval.sh 8 ... 0,1`):

```
   result order matches dataset order
metric families present: ['ADE', 'EPA', 'FDE', 'MR', 'NuscMap', 'motion', 'occ', 'plan']
DIST EVAL PATH OK
```

---

## 9.5 후속: `grad_balance` 및 VAD 학습률 채택 (2026-08-18, 같은 날 결정)

리뷰 대응과는 별개로, 60 epoch 본 실험 설정을 두 군데 바꿨다.

### `aux_grad_scale` → `grad_balance`

`aux_grad_scale`은 **노브**를 정하고 결과 지분은 나오는 대로 두는 방식이었다.
실측하니 0.01에서 planning 지분이 4.1%였다 — 이름이 시사하는 "planning 보호"를
하지 못했다. 그래서 **결과**를 지정하는 방식으로 교체했다.

`projects/mmdet3d_plugin/SSR/utils/grad_balance.py`. `interval`마다
`‖∂L_k/∂bev_embed‖`를 재고, 현재 밸브를 역산한 뒤 다시 푼다. planning이 기준이라
스케일되지 않는다:

```
g_k^raw = g_k^meas / s_k
s_k     = (t_k / t_plan) · (g_plan^raw / g_k^raw)
```

DDP에서는 solve 이전에 all_reduce로 norm을 평균낸다. 이게 없으면 rank마다 다른
스케일을 적용해 **서로 다른 목적함수를 최적화**하게 된다.

실제 배치 검증 (`tools/verify_grad_balance.py`):

| iter | plan | det | map | occ |
|---:|---:|---:|---:|---:|
| 0 (보정 전) | 0.0007 | 0.5745 | 0.2701 | 0.1546 |
| 7 | 0.3929 | 0.2064 | 0.2003 | 0.2004 |

목표 `plan=0.4, det/map/occ=0.2` 대비 최대 오차 0.0071. 풀려 나온 밸브는
**det 5.5e-4 / map 1.2e-3 / occ 2.1e-3** — 서로 4배씩 다르다. 단일 상수로는
원리적으로 균형이 불가능하고, `0.01`은 det에 필요한 값의 18배였다.

### EMA가 산술이면 컨트롤러가 4 epoch 동안 아무것도 안 한다

2-rank 실제 학습을 붙여 보고서야 나온 버그다. 단위 테스트는 `momentum=0.0`으로
돌아서 못 잡았다.

```
   iter |     s_det
     10 |  9.00e-01
     20 |  8.55e-01     <- 목표는 ~5e-4
```

밸브가 초기값 1.0에서 목표로 내려가는데 산술 EMA는 `0.9ⁿ`을 따른다. 5e-4에
닿으려면 **약 70회 갱신 = interval 200 기준 14,000 iteration ≈ 4 epoch**이다.
그 4 epoch이 정확히 컨트롤러가 막으려던 불균형 구간이다.

**수정 두 가지:**

* **기하 EMA** — `s ← s^m · s_new^(1-m)`. 밸브가 4자릿수를 넘나드는 양이라
  산술 평균이 맞지 않는다.
* **첫 측정은 그대로 채택** — 1.0은 추정값이 아니라 임의의 초기값이라 평균낼
  대상이 없다. warm-up 이후 첫 측정에서 바로 해로 점프하고, 이후 드리프트만
  EMA로 평활한다.

수정 후 실제 iter-0 분포(plan 0.07% / det 57% / map 27% / occ 15%)에서
`momentum=0.9`로 **1회 갱신 만에** 목표에 도달한다. 이 케이스를 회귀 테스트에
추가했다.

**해석 주의:** `plan=0.4`는 자연 지분(0.07~4.6%) 대비 **6~500배 증폭**이다.
SSR baseline과 다른 체제이므로, planning이 좋아져도 "aux 덕분"과 "planning
gradient를 키워서"가 동등하게 그럴듯한 설명이다.

또한 det과 motion은 **밸브를 공유한다** — `det_motion_head`를 한 번 forward하므로
BEV 입력이 하나뿐이다. `gshare/motion`은 정보용으로 따로 찍히며 det과 겹치므로
합계 1에서 제외된다.

### 학습률 5e-5 → 2e-4 (VAD 값)

VAD_base_e2e.py / VAD_tiny_e2e.py와 한 줄씩 대조한 결과:

| 항목 | VAD | SSR (기존) |
|---|---|---|
| AdamW, `weight_decay=0.01` | ✓ | **동일** |
| `img_backbone` `lr_mult=0.1` | ✓ | **동일** |
| `grad_clip max_norm=35, norm_type=2` | ✓ | **동일** |
| CosineAnnealing, warmup linear 500 / ratio ⅓ | ✓ | **동일** |
| `min_lr_ratio=1e-3` | ✓ | **동일** |
| **`lr`** | **2e-4** | **5e-5** |

optimizer 설정 전체가 이미 VAD와 같았고 **base LR만 4배 낮았다.** aux head가
VAD의 것이고 VAD가 60 epoch에 수렴시킨 rate가 2e-4이므로, 1/4로 돌리면
"60 epoch에 aux가 수렴하지 않았다"가 구조 탓인지 LR 탓인지 구분되지 않는다.
그게 정확히 이 장기 실험이 답하려는 질문이므로 2e-4로 맞췄다.

> **§4 정정.** `max_norm=35`가 VAD와 동일하다는 것은, 내가 §4에서 "clipping이
> epoch 10부터 100% 발동한다"고 경고한 체제가 **VAD가 논문 수치를 낸 정상
> 동작 영역**이라는 뜻이다. 계측(`clip/rate`, `pnorm/*`)은 유지하되 경고 수위는
> 낮춘다. `max_norm` 상향은 근거가 약해졌다.

**대가:** planning이 SSR 프로토콜을 벗어난다. 재현했던 L2 MAX avg 0.7526은
5e-5 / 12 epoch 값이므로 이 실험의 참조점이 될 수 없다. "aux가 planning에
얼마를 치르게 하는가"를 말하려면 **동일 스케줄(60ep cosine, lr 2e-4)의
plan-only baseline**이 따로 필요하다.

---

## 9.6 착수 전 최종 검수 (2026-08-18)

`PARA_SSR_e2e_2gpu_b4_60ep.py`로 학습·평가 경로를 실제로 돌려 점검했다. 버그
셋을 찾았고, 셋 다 정적 읽기로는 안 보이고 **실행해야 보이는 종류**였다.

### 발견 1 — occupancy GT가 계속 수집되고 있었다

`occ_head=None`인데 `simple_test`의 조건이 `test_aux_heads`만 봤다. 파이프라인은
여전히 occ GT를 만들므로, **예측이 존재하지 않는 대상의 정답을 6019샘플 전부**
`collect_results_cpu`로 실어 나르고 있었다 — 0.39 GiB. `self.occ_head is not
None`을 추가했다.

### 발견 2 — 평가 산출물이 잘못된 파티션에 15 GB 쌓인다

`mmdet_train.py`가 `jsonfile_prefix`를 `osp.join('val', cfg.work_dir, ...)`로
**덮어쓰고** 있었다. `work_dirs`는 `/data1`(1.8 TB) 심볼릭인데
`val/work_dirs/...`는 그 심볼릭을 타지 않고 저장소 밑, 즉 루트 파티션(67 GB
여유, 97% 사용, 4인 공유)에 새로 생긴다.

측정: detection 제출 json이 8샘플에 1.6 MB → 6019샘플이면 **1회당 ~1.2 GB**.
타임스탬프 디렉터리라 덮어쓰지도 않으므로 12회 = **~15 GB**. 7일 실행 후반에
디스크가 찼을 것이다. `setdefault` + 기본 경로를 `work_dir/val/`로 바꿨다.

### 발견 3 — `analyze_gshare.py`가 occ 없는 로그에서 죽는다

`TASKS`가 하드코딩이라 `KeyError: 'gshare/occ'`. 로그에서 task를 발견하도록
고쳤다. 어떤 head가 존재하는지는 config 선택이므로 고정 목록은 애초에 틀렸다.

### 실행 검증

평가 (실제 2-rank, 실제 sampler/수집/`evaluate()`):

```
result order matches dataset order
metric families present: ['ADE','EPA','FDE','MR','NuscMap','motion','plan']
DIST EVAL PATH OK
```

학습 (실제 2-rank):

```
lr 7.17e-05 (warmup->2e-4)  loss 42.66  grad_norm 17.68  clip/rate 0.0000
gscale/det 0.0005  gscale/map 0.0021
map_cls_acc 0.5586  map_conf_pos 0.1104  map_conf_neg 0.1103
map_spread_m 9.8216  map_pts_err_m 11.9996  map_n_gt 10.06
pnorm/trunk 2.2553  plan 2.0281  aux 18.1940
```

occ 관련 키는 하나도 없다. 지분 복원: `plan 0.05% -> 25.8%`(목표 40, 배치
노이즈 범위 내).

---

## 9.7 det 디코더 6층 -> 3층

파라미터를 task별로 집계하다 드러난 불일치다. v1은 det만 VAD-Base의 6층이고
나머지(map, BEV 크기, backbone, 쿼리 수)는 전부 VAD-Tiny였다.

| task | 6층 | 3층 | aux 내 |
|---|---:|---:|---|
| shared trunk | 28,998,917 | 28,998,917 | - |
| plan | 2,416,978 | 2,416,978 | - |
| **det** | **5,873,595** | **3,013,983** | 61.3% -> **44.8%** |
| motion | 796,941 | 796,941 | 8.3% -> 11.9% |
| map | 2,910,001 | 2,910,001 | 30.4% -> **43.3%** |
| TOTAL | 40,996,432 | **38,136,820** | |

6층 디코더 하나가 head 6.67M 중 4.26M이었고, 그 결과 detection이 전체 aux
파라미터의 70%와 (실측) 공유 BEV에 도달하는 gradient의 82%를 가져갔다. 3층에서는
det과 map이 45:43으로 거의 대등해지고, `grad_balance`가 적용해야 할 보정 폭도
그만큼 줄어든다.

branch 복제도 층수를 따라간다(det cls/reg 각 6 -> 3개). v1의 6층 정의는
`work_dirs/para_ssr_8gpu/PARA_SSR_e2e.py`에 보존돼 있다.

### task별 파라미터 세부 (3층 기준)

```
SHARED  backbone + FPN            24,347,136   63.8%
SHARED  BEV encoder                4,651,781   12.2%   transformer 2.07M
                                                        + bev_embedding 2.56M
PLAN    navi-SE + TokenLearner       167,248    0.4%   16 sparse tokens
PLAN    latent decoder x3          1,581,312    4.1%
PLAN    waypoint decoder             668,418    1.8%   18 query (3cmd x 6step)
DET     agent decoder x3           2,205,987    5.8%   300 query
DET     cls + reg branches           807,996    2.1%
MOTION  motion decoder               529,408    1.4%
MOTION  traj branches                267,533    0.7%
MAP     map decoder x3             2,113,570    5.5%   100 vec x 20 pts
MAP     cls + reg branches           796,431    2.1%
```

공유 trunk가 76%다. 네 task가 다투는 대상을 만드는 부분이 압도적이고, task 고유
파라미터는 다 합쳐도 24%다 -- `grad_balance`가 조절하는 것이 정확히 그 76%로
되돌아가는 gradient다.

---

## 10. 변경 파일

| 파일 | 변경 |
|---|---|
| `SSR/dense_heads/para_occ_head.py` | `occ_sep` valid 마스킹, `valid_frac` 추가 |
| `SSR/dense_heads/para_map_head.py` | `map_pts_err_m` 유클리드화, `map_pos_frac`→`map_n_gt`, `map_spread_m`·`map_conf_*` 신규 |
| `SSR/para_ssr.py` | occ GT uint8, `sample_token` 스탬프 |
| `SSR/hooks/clip_monitor.py` | **신규** — clip 발생률/계수/모듈군 norm |
| `SSR/hooks/__init__.py` | 훅 등록 |
| `datasets/nuscenes_vad_dataset.py` | `det_score_thresh` 인자, `_assert_token_alignment`, map GT 토큰 재정렬 |
| `configs/SSR/PARA_SSR_e2e.py` | `optimizer_config`에 계측 훅 |
| `configs/SSR/PARA_SSR_e2e_2gpu_b4.py` | grad scale 주석 정정 |
| `configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py` | EMA resume / 1-GPU 평가 / schedule 비교 주의 |
| `configs/SSR/PARA_SSR_w_gradscale.py` | sweep arm 정리 |
| `tools/analyze_gshare.py` | 중복행 제거 + epoch 리셋 + `--epoch` |
| `tools/verify_aux_metrics.py` | **신규** — 지표 의미 회귀 테스트 |
| `tools/verify_dist_eval.{py,sh}` | **신규** — 실제 2-rank 평가 통합 테스트 |
| `tools/verify_multibatch_and_metrics.py` | exit code, 범위 명시 |
| `report/02, 05, 06` | 정정 |

---

## 11. 남은 것

| 우선순위 | 항목 |
|---|---|
| P1 | 1 epoch 후 `gshare/*` + `pnorm/*`로 `aux_grad_scale` / `max_norm` 재보정 |
| P1 | 60 epoch schedule-matched SSR-orig baseline (§8.4) |
| P2 | 보고서 #03 (occupancy teacher 설계) — 미작성 |
| P2 | occ teacher / map KD 인터페이스 설계 |
