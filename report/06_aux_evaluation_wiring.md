# 보고서 #06 — Aux head 평가 배선 (map mAP / occ IoU)

작성일 2026-08-17 · 브랜치 `para`

---

## 0. 왜 필요했나

1차 실험에서 **14.7시간 학습이 끝난 뒤에야 map head가 죽어 있었다는 걸 알았다.** 학습 로그에는 map 손실이 있었고 감소도 했지만(BCE −18%), 실제 head는 전 픽셀에 ≈0.29를 출력하는 상수 함수였다. 손실 곡선만으로는 구분이 안 된다.

그리고 map을 VAD 벡터로 바꾼 지금, **그게 학습되는지 확인할 수단이 아예 없었다.**

| | 이전 | 이후 |
|---|---|---|
| **det mAP / NDS (val)** | ❌ 불가 | ✅ `pts_bbox_NuScenes/mAP`, `/NDS` |
| **motion EPA/ADE/FDE/MR (val)** | ❌ 불가 | ✅ `EPA_{car,pedestrian}` 등 |
| map mAP (val) | ❌ 불가 | ✅ `NuscMap_chamfer/*` |
| occ IoU (val) | ❌ 불가 | ✅ `occ_iou_thr*/…` |
| 학습 중 aux 품질 | ❌ 없음 | ✅ `occ_iou*/…`, `occ_sep/…` (200 iter마다) |

**4개 head 전부 채점 가능해졌다.** det/motion은 초판에서 "범위 밖"으로 미뤘다가 같은 이유(학습 여부를 확인할 수단이 없다)로 §7에 추가했다.

---

## 1. 왜 기존 경로를 못 썼나

SSR의 `_evaluate_single()`에는 map 평가 기계(`format_res_gt_by_classes` + `eval_map`)가 **이미 있었다.** 못 쓴 이유는 두 가지다.

1. **`_format_bbox()`가 planning 전용으로 축소되어 있었다.** `map_pred_annos = {}`가 선언만 되고 채워지지 않는다. VAD 원본에는 있는 코드가 SSR에서 제거됐다.
2. **`_evaluate_single()`은 nuScenes detection 평가를 먼저 돌린다.** `NuScenesEval_custom`이 detection 결과를 요구하는데, PARA-SSR은 `test_aux_heads=False`가 기본이라 det를 내보내지 않는다. map만 켜고 싶어도 이 경로를 타면 죽는다.

그래서 **detection에 의존하지 않는 독립 map 평가 경로**를 새로 만들었다.

---

## 2. 변경 내용

### 2.1 모델 출력 형식 (`para_ssr.py`)

평가 함수 `output_to_vecs`는 `detection['map_boxes_3d']` 등 **최상위 키 4개**를 읽는데, 기존 코드는 `bbox_results[0]['map']`에 중첩 저장하고 있었다.

```python
@staticmethod
def map_pred2result(bboxes, scores, labels, pts):
    return dict(map_boxes_3d=bboxes.to('cpu'), map_scores_3d=scores.cpu(),
                map_labels_3d=labels.cpu(), map_pts_3d=pts.to('cpu'))
```

VAD의 `VAD.map_pred2result`와 동일하다. 디코딩된 벡터 head는 flat 키로, dense head(dict 반환)는 기존대로 `['map']`에 남긴다 — 두 표현을 config로 오갈 수 있어야 하므로 분기했다.

occupancy는 `occ_seg`(이진)에 더해 `occ_scores`(확률)도 내보낸다. threshold sweep이 필요하기 때문이다 (§4).

### 2.2 Occupancy GT 전달 (`PARA_SSR_e2e.py`, `para_ssr.py`)

occ IoU를 재려면 val 시점에 GT가 필요하다. **학습 파이프라인과 동일한 transform**을 test 파이프라인에도 넣고 결과 dict로 흘려보낸다.

```python
# test_pipeline, range filter 앞 (train과 같은 위치)
dict(type='GenerateSSROccLabels', pc_range=..., bev_size=(bev_h_, bev_w_),
     n_future=occ_n_future, class_labels=occ_class_labels),
...
keys=[..., 'gt_occ_seg', 'gt_occ_valid']
```

`evaluate()` 안에서 GT를 다시 만드는 방법도 있었지만 택하지 않았다 — 클래스 그룹핑과 horizon이 두 곳에 중복되면 언젠가 어긋난다. **손실이 쓴 바로 그 타깃으로 채점**하는 것이 요점이다.

### 2.3 독립 평가 경로 (`nuscenes_vad_dataset.py`)

```python
def _format_map_results(self, results, out_dir)   # 예측 -> eval_map 입력 형식
def evaluate_map(self, results, ...)              # chamfer / iou mAP
def evaluate_occ(self, results, thresholds=(0.1, 0.3, 0.5))
```

`evaluate()`는 모델이 실제로 내보낸 것만 채점한다:

```python
first = results[0].get('pts_bbox', {})
if 'map_pts_3d' in first:  results_dict.update(self.evaluate_map(...))
if 'occ_seg'    in first:  results_dict.update(self.evaluate_occ(...))
```

**주의점 하나.** `format_res_gt_by_classes`는 `zip(gen_results.values(), annotations)`로 예측 dict과 GT **리스트**를 위치로 짝짓는다. 즉 예측은 데이터셋 순서로 삽입되어야 하고 샘플이 하나도 빠지면 안 된다. 조용히 어긋나면 mAP가 의미 없어지므로 명시적 assert를 넣었다:

```python
assert len(map_pred_annos) == len(self), \
    'chamfer mAP pairs predictions with GT by position'
```

### 2.4 shapely 2.x 비호환 수정 (`map_utils/tpfp_chamfer.py`)

배선을 끝내고 돌리니 터졌다:

```
AttributeError: 'numpy.int64' object has no attribute 'intersects'
```

**코드베이스의 기존 버그다.** shapely 1.x의 `STRtree.query()`는 geometry 객체를 돌려주지만 **2.0부터는 정수 인덱스**를 돌려준다. 이 환경은 shapely 2.0.1이라 map 평가는 애초에 동작한 적이 없었다.

같은 패턴이 5곳에 있어 버전 호환 래퍼로 일괄 교체했다:

```python
class GeomIndex:
    """STRtree that yields (index, geometry) on shapely 1.x and 2.x."""
    def __init__(self, geoms):
        self.geoms = geoms
        self.tree = STRtree(geoms)
        self._by_id = None if _SHAPELY2 else {id(g): i for i, g in enumerate(geoms)}

    def query(self, geom):
        for item in self.tree.query(geom):
            if self._by_id is None:
                idx = int(item); yield idx, self.geoms[idx]
            else:
                yield self._by_id[id(item)], item
```

1.x 경로도 남겨서 다른 환경에서 깨지지 않게 했다.

### 2.5 학습 중 품질 로깅 (`para_ssr.py`, `para_occ_head.py`)

`grad_norm_log_interval`과 같은 방식으로 `aux_metric_log_interval=200`을 추가했다. 200 iter마다 `ParaOccHead.train_metrics()`가 현재 배치에서 계산한다:

| 키 | 의미 |
|---|---|
| `occ_iou0.1/mean`, `occ_iou0.5/mean` | threshold별 IoU |
| `occ_sep/p_on` | GT 양성 셀의 평균 예측 확률 |
| `occ_sep/p_off` | GT 음성 셀의 평균 예측 확률 |
| **`occ_sep/ratio`** | **p_on / p_off — 공간 분별력** |

**`occ_sep/ratio`가 핵심이다.** 상수 출력 head는 손실이 어떻든 이 값이 **1.0에 붙어 있다.** v1의 dense map head는 boundary에서 1.06×였고 그게 "죽었다"의 정확한 정의였다. 정상 학습된 occ vehicle head는 7.8×였다.

비용은 이미 메모리에 있는 텐서에 sigmoid 한 번 + reduction 몇 개다. 키에 `loss` 문자열이 없어 `_parse_losses`가 로깅만 하고 최적화 대상에 넣지 않는다.

---

## 3. 검증

### 3.1 전체 체인 (미학습 가중치, val 12샘플)

```
per-sample result keys: [ego_fut_cmd, ego_fut_preds, gt_occ_seg,
                         map_boxes_3d, map_labels_3d, map_pts_3d,
                         map_scores_3d, occ_scores, occ_seg]
  map_pts_3d    (50, 20, 2)   torch.float32
  occ_scores    (1, 7, 1, 100, 100)
  gt_occ_seg    (1, 7, 1, 100, 100)

NuscMap_chamfer/mAP        0.0000
occ_iou_thr0.1/mean        0.0238
(+ 25 planning metrics)
VERDICT: wiring OK
```

### 3.2 Positive control — 여기가 중요하다

미학습 mAP 0.0000은 **정상인지 조용히 망가진 건지 구분이 안 된다.** GT를 그대로 예측으로 넣어 확인했다.

> **정정 (2026-08-18).** 초판의 control은 `pts += shift`로 **x와 y에 같은 값을 동시에 더했다.** "0.75 m 이동"의 실제 유클리드 거리는 0.75·√2 = **1.061 m**였고, occupancy는 `torch.roll`이라 밀려난 셀이 **반대편으로 되감겼다.** 따라서 "chamfer 임계값이 미터 단위로 동작함을 증명했다"는 결론은 그 실험으로는 성립하지 않았다. 단일 축 이동 + 제로 패딩으로 다시 측정했다 (`tools/verify_aux_eval.py`).

**Map — x축으로만 이동** (BEV 셀 = x 0.30 m, y 0.60 m):

| x 이동 | mAP | divider AP @ thr 0.5 / 1.0 / 1.5 |
|---:|---:|---|
| **0.00 m** | **1.0000** | 1.000 / 1.000 / 1.000 |
| 0.75 m | 0.7778 | **0.000** / 1.000 / 1.000 |
| 1.25 m | 0.5523 | 0.000 / **0.000** / 1.000 |
| 2.50 m | 0.2435 | 0.000 / 0.037 / 0.131 |

**이제 임계값이 미터 단위로 정확히 동작함이 실제로 증명된다** — 이동량 s가 임계값을 넘는 순간 그 임계값의 AP가 0이 되고, 넘지 않은 임계값은 1.0을 유지한다.

**Occupancy — 행(세로) 방향 평행이동, 되감기 없음:**

| 이동 | 거리 | IoU@0.5 |
|---:|---:|---:|
| 0 셀 | 0.00 m | **1.0000** |
| 1 셀 | 0.60 m | 0.8512 |
| 3 셀 | 1.80 m | 0.6237 |

**valid-frame 마스킹:**

| | IoU@0.5 |
|---|---:|
| `gt_occ_valid` 적용 (유효 프레임만) | **1.0000** |
| 미적용 (전 프레임) | 0.4693 |

미학습 모델의 IoU 0.0238이 이 12개 val 샘플의 **GT 양성률 2.38%와 정확히 일치**하는 것도 앞뒤가 맞는다 — 미학습 head는 전 셀을 양성으로 예측하므로 IoU = 양성률이다.

> 주의: 이 12샘플의 양성률 2.38%는 train 120샘플에서 잰 15.5%와 다르다. 표본이 작고 씬이 달라서다. positive control이 정렬을 보증하므로 버그는 아니지만, **12샘플 수치를 성능으로 읽으면 안 된다.**

### 3.3 학습 중 로깅

```
occ_iou0.1/cls0    0.169438      gnorm/plan    0.000053
occ_iou0.5/cls0    0.131367      gshare/plan   0.000661
occ_sep/p_on       0.472450      gshare/det    0.576165
occ_sep/p_off      0.475468      gshare/map    0.274826
occ_sep/ratio      0.993652      gshare/occ    0.148348

total optimised loss          : 63.622913
sum of keys containing "loss" : 63.622918
difference                    : 5.11e-06
VERDICT: training-time aux metrics wired and non-optimised
```

초기화 직후 **`occ_sep/ratio = 0.9937`** — 상수 출력 상태의 서명이 그대로 잡힌다. 잡아내려던 신호가 정확히 이것이다.

### 3.4 회귀 — aux off 기본 경로

`test_aux_heads=False`(기본값)에서:
```
map mAP produced : False
occ IoU produced : False
VERDICT: planning-only path OK
```
planning 25개 지표만 나오고 map/occ 경로는 건너뛴다. 기존 동작이 유지된다.

### 3.5 config 회귀

`tools/verify_multibatch_and_metrics.py`가 확인한다:

```
2gpu_b4        : 12 ep, eval/1,  aux=False     <- 기존 실험 정의 유지
2gpu_b4_60ep   : 60 ep, eval/10, aux=True      <- 장기 실행 (별도 파일)
```

> **정정.** 초판 작성 시점에는 60-epoch 설정을 `PARA_SSR_e2e_2gpu_b4.py`에 직접 넣었는데, 그러면 **기존 `train_para_2gpu_b4.sh`가 조용히 60 epoch + `--no-validate`로 바뀐다.** 12-epoch 정의를 되돌리고 장기 실행을 `PARA_SSR_e2e_2gpu_b4_60ep.py`로 분리했다.

---

## 4. 사용법

### 학습 중
자동이다. 200 iter마다 로그와 wandb에 찍힌다. **1 epoch 안에 봐야 할 것:**

| 지표 | 죽은 head | 학습 중 |
|---|---|---|
| `occ_sep/ratio` | ≈ 1.0 | 상승 |
| `occ_iou0.5/mean` | ≈ GT 양성률에 고정 | 상승 |
| `map_pts_err_m` | 정체 | 하락 |
| `map_cls_acc` | ≈ 0 또는 무작위 | 상승 |
| `map_conf_pos` / `map_conf_neg` | 둘 다 0으로 수렴, 또는 둘 다 높음 | pos↑ / neg↓ 로 벌어짐 |
| `map_spread_m` | **→ 0** (모든 쿼리가 한 선에 붕괴) | GT 배치 폭 수준 유지 |

> **정정 (2026-08-18).** 초판은 "`loss_map_pts`가 매칭된 점의 L1이라 학습 신호로 충분하다"고 썼는데 **과한 결론이었다.** 점 회귀가 좁혀지는 동안에도 (a) classification이 계속 틀리거나, (b) 모든 쿼리가 같은 폴리라인으로 붕괴하거나, (c) recall이 바닥이면 chamfer mAP는 0일 수 있다. dense map의 교훈("손실 감소 ≠ 품질")이 벡터 head에도 그대로 적용된다. 그래서 `ParaMapHead.train_metrics()`를 추가했다.

> **재정정 (2026-08-18, 2차 리뷰).** 위 문단에서 추가했던 지표 세 개 중 **두 개의 해석이 틀렸고, 매칭 재사용 서술도 틀렸다.** 상세는 [보고서 #07](07_review_round2_fixes.md) §1–2. 요약:
>
> | 초판 서술 | 실제 | 조치 |
> |---|---|---|
> | "손실의 Hungarian을 **재사용**" | `train_metrics()`가 `map_get_targets`를 **새로** 호출하고, `loss()`가 decoder 층마다 또 호출한다 (총 4회). **추가 forward가 없다는 것만 맞다.** | 문구 정정, `aux_metric_log_interval` 게이팅 유지 |
> | "`map_pos_frac` — 쿼리 붕괴를 잡는다" | `MapHungarianAssigner3D`에 cost 임계값이 없어 **항상** `min(Q, N_gt)`개를 매칭한다. 즉 `ΣN_gt / (B·Q)`로, **예측과 무관한 데이터 통계**다. | **폐기.** `map_n_gt`(이미지당 GT 개수)로 이름만 정직하게 남김. 붕괴 감지는 신규 `map_spread_m`이 담당 |
> | "`map_pts_err_m` — chamfer 임계값과 직접 비교 가능" | `|dx|`와 `|dy|`의 평균이었다. 단일 축 오차를 **2배**, 45° 오차를 **1.41배** 과소보고한다. | 유클리드 노름으로 수정. 다만 chamfer는 최근접점 거리이고 이건 index 정렬 거리라 **여전히 chamfer의 상한**이며, 임계값 통과 판정에는 쓸 수 없다 |
>
> 그래도 **최종 판단은 10 epoch마다 나오는 validation mAP로 해야 한다.** 위 지표들은 조기 경보일 뿐이다. 회귀 테스트는 `tools/verify_aux_metrics.py`.

### 평가
`test_aux_heads`는 추론 비용 때문에 기본이 `False`다(PARA-Drive의 런타임 이점). aux를 채점하려면 켠다:

```bash
cd /home/yongjae/e2e/SSR
export PATH=/home/yongjae/miniconda3/envs/ssr/bin:$PATH NUMBA_CPU_NAME=generic

# 한 번만: map GT 캐시 (~30분, CPU)
python tools/build_map_gt_cache.py

# 최종 비교용 -> EMA 가중치를 쓸 것 (학습 중 validation은 raw를 평가한다)
python tools/test.py projects/configs/SSR/PARA_SSR_e2e.py \
    work_dirs/<run>/epoch_60_ema.pth --eval bbox \
    --cfg-options model.test_aux_heads=True
```

> **정정:** 초판의 명령은 raw `epoch_12.pth`를 썼다. 학습 중 validation이 이미 raw를 평가하므로, 최종 수치는 `epoch_N_ema.pth`로 재야 논문 프로토콜과 맞는다.

**지표 검증 스크립트** (repo 안에 있으므로 재현 가능):

| 스크립트 | 확인 내용 |
|---|---|
| `tools/verify_aux_eval.py` | map chamfer / occ IoU positive·negative control |
| `tools/verify_multibatch_and_metrics.py` | P0 회귀 4건 (dict 언래핑, label 보존, valid mask, config 분리) |
| `tools/verify_batch_equivalence.py` | bs=4 전체 학습 스텝 + 샘플 간 누수 개입 테스트 |
| `tools/verify_grad_scale.py` | `aux_grad_scale`이 BEV gradient만 줄이는지 |
| `tools/analyze_gshare.py` | 로그에서 순간 gshare 복원 + `aux_grad_scale` 계산 |
| `tools/build_map_gt_cache.py` | map GT 캐시 사전 생성 |

출력 지표:
```
pts_bbox_NuScenes/mAP, /NDS         detection
pts_bbox_NuScenes/{cls}_AP_dist_*   클래스별 AP
EPA_{car,pedestrian}                motion (ViP3D)
ADE_/FDE_/MR_{car,pedestrian}
motion_cnt_{cls}/{gt,matched,hit,fp}  위 값 해석용 카운트
NuscMap_chamfer/mAP                 map
NuscMap_chamfer/{divider,ped_crossing,boundary}_AP
NuscMap_chamfer/{cls}_AP_thr_{0.5,1.0,1.5}
occ_iou_thr{0.1,0.3,0.5}/cls0       vehicle occupancy IoU
occ_iou_thr{0.1,0.3,0.5}/mean
plan_L2_*, plan_obj_box_col_*       planning
```

> **`ADE`가 `nan`이면 성능이 아니라 매칭 실패다.** `motion_cnt_{cls}/matched`를 함께 볼 것.

> **threshold를 여러 개 내는 이유**: 유용한 동작점은 클래스 밀도에 따라 다르다. UniAD는 occupancy를 `test_seg_thresh=0.1`로 평가한다. 0.5 한 점만 보고 "죽었다"고 결론냈던 것이 보고서 #01의 오진단이었다.

### 기준선

| | 값 | 출처 |
|---|---:|---|
| VAD-Tiny map mAP (chamfer) | ~0.35–0.40 | VAD 논문 |
| VAD-Tiny detection mAP | ~0.27 | VAD 논문 |
| UniAD vehicle occ IoU | ~0.6 (근거리) | UniAD 논문 |
| PARA-SSR v1 occ vehicle IoU | 0.574 | 보고서 #01 §6.1 |
| PARA-SSR v1 dense map IoU | 0.117 (최적 thr) | 보고서 #01 §6.2 |

**판단 기준**

| head | 학습됨 | 죽음 → 중단 |
|---|---|---|
| map (벡터) | mAP > 0.2 | mAP < 0.05 |
| det | mAP > 0.15 | mAP < 0.03 |
| motion | `matched` > 0, ADE < 2 m | ADE `nan` |
| occ | IoU@0.5 > 0.4 | `occ_sep/ratio` ≈ 1 |

> PARA-SSR v1에서 det는 BEV gradient의 65%를 가져갔고 `det_cls` 손실이 −67%였다. **그런데 mAP는 한 번도 재본 적이 없다.** ver2 실행에서 처음으로 숫자가 나온다.
>
> **정정:** 초판은 이를 근거로 det를 "가장 잘 학습된 head"라고 단정했는데, gradient 점유율과 손실 감소만으로는 말할 수 없다. 그건 이 보고서의 전제("손실 감소 ≠ 품질")와 정면으로 모순된다 — dense map도 손실이 18% 내려가면서 상수 함수였다. det의 품질은 mAP가 나오기 전까지는 미지수다.

---

## 5. 변경 파일

```
M  projects/mmdet3d_plugin/SSR/para_ssr.py
     map_pred2result(), det_pred2result(),
     assign_pred_to_gt_vip3d(), compute_motion_metric_vip3d(),
     occ_scores/gt_occ_seg 전달, aux_metric_log_interval
M  projects/mmdet3d_plugin/SSR/dense_heads/para_occ_head.py
     train_metrics() — IoU + 양성/음성 분리도
M  projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py
     _format_det_results(), evaluate_det(), evaluate_motion(),
     _format_map_results(), evaluate_map(), evaluate_occ(),
     _load_nuscenes() — lidarseg 우회,
     evaluate() 분기 + planning 집계에서 motion 카운터 분리
M  projects/mmdet3d_plugin/datasets/map_utils/tpfp_chamfer.py
     GeomIndex — shapely 2.x 호환 (기존 버그 수정, 5곳)
M  projects/mmdet3d_plugin/SSR/dense_heads/para_map_head.py
     train_metrics() — cls_acc / pos_frac / pts_err_m
M  projects/configs/SSR/PARA_SSR_e2e.py
     test_pipeline에 GenerateSSROccLabels + collect keys,
     aux_metric_log_interval, aux_grad_scale=0.01
A  projects/configs/SSR/PARA_SSR_e2e_2gpu_b4_60ep.py   60 epoch 장기 실행
M  projects/configs/SSR/PARA_SSR_w_gradscale.py        0.02 -> 0.005 (약한 쪽 arm)
A  tools/verify_aux_eval.py, verify_multibatch_and_metrics.py,
   verify_batch_equivalence.py, verify_grad_scale.py,
   analyze_gshare.py, build_map_gt_cache.py
A  train_para_2gpu_b4_60ep.sh
```

### 리뷰 대응으로 고친 P0 4건 (2026-08-18)

| # | 문제 | 결과 |
|---|---|---|
| 1 | `custom_multi_gpu_test`는 dict를 반환하는데 `evaluate()`는 list를 전제 | **분산 validation이 epoch 10에서 죽었을 것.** VAD에 있는 언래핑이 SSR 축소판에서 누락 |
| 2 | 60-epoch 설정을 공유 config에 직접 넣음 | 기존 12-epoch 실행 정의가 조용히 변경됨 → 별도 파일로 분리 |
| 3 | motion 지표가 detection label을 in-place 변형, score 필터 누락 | truck/bus/trailer AP 손상 + EPA의 FP 항이 VAD와 비교 불가 |
| 4 | occupancy 지표가 `gt_occ_valid`를 무시 | 손실과 다른 타깃으로 채점 (val 미래 프레임의 8.7%가 invalid) |

### 고친 기존 버그 (내가 만든 것이 아닌 것)

| 버그 | 영향 |
|---|---|
| shapely 2.x `STRtree.query` 반환형 변경 미대응 | **map 평가가 이 환경에서 한 번도 동작한 적 없음** |
| `_format_bbox`에서 detection 변환 루프 누락 | det 평가 불가 |
| `evaluate()`가 `metric_results` 전 키를 평균 | motion 카운터가 planning 지표로 둔갑 |
| devkit이 라벨 없는 `lidarseg.json`에서 assert | detection 평가 진입 불가 |
| ADE/FDE 분모를 `max(n,1)`로 클램프 (VAD 원본 포함) | **매칭 0인 죽은 head가 ADE 0.0000 = 만점으로 보임** |

---

## 7. Detection + Motion 평가 (추가분)

초판에서 "지금 필요한 건 map/occ이라 범위에서 뺐다"고 미뤘던 부분이다. **미룰 이유가 없었다** — det head도 학습되는지 확인할 수단이 없기는 마찬가지였고, 그게 이 작업 전체의 목적이다.

### 7.1 Detection mAP / NDS

SSR의 `_format_bbox()`는 planning 전용으로 축소되면서 **detection 변환 루프가 통째로 제거**돼 있었다(VAD 원본에는 있다). 복구해서 `_format_det_results()`로 분리하고, 독립 `evaluate_det()`를 추가했다.

```python
@staticmethod
def det_pred2result(bboxes, scores, labels, trajs):
    return dict(boxes_3d=bboxes.to('cpu'), scores_3d=scores.cpu(),
                labels_3d=labels.cpu(), trajs_3d=trajs.cpu())
```

`output_to_nusc_box` / `lidar_nusc_box_to_global` / `NuScenesEval_custom`은 SSR에 이미 다 있었다. 없던 건 이들을 잇는 배선뿐이다.

### 7.2 Motion 지표 (ViP3D 프로토콜)

SSR에는 아예 없어서 VAD에서 포팅했다.

| 함수 | 위치 | 역할 |
|---|---|---|
| `assign_pred_to_gt_vip3d` | `ParaSSR` | BEV 중심거리 헝가리안 매칭 (2 m 임계) |
| `compute_motion_metric_vip3d` | `ParaSSR` | 샘플별 카운터 (gt/hit/fp/ADE/FDE/MR) |
| `evaluate_motion` | dataset | split 전체 집계 → EPA/ADE/FDE/MR |

매칭 규칙은 VAD와 동일하다. 규칙이 다르면 논문 수치와 비교 자체가 안 된다. 정적 클래스는 양쪽에서 제외되어 주차된 barrier가 매칭을 흡수하지 못한다.

EPA는 프레임별로 정의되지 않으므로(`(hit − 0.5·fp) / gt`) 모델은 **원시 카운트**만 내보내고 나눗셈은 데이터셋이 split 전체에 대해 한 번 한다.

> **정정 (2026-08-18) — 초판은 VAD와 동일하지 않았다.** 두 가지가 빠져 있었고, 그 상태에서 "논문 수치와 비교 가능"이라고 쓴 것은 사실이 아니었다.
>
> 1. **결과 dict 변형.** `compute_motion_metric_vip3d`는 vehicle 클래스를 `car`로 접는데 이걸 **live 결과 dict에 in-place로** 했다. truck(1)/bus(3)/trailer(4)가 전부 car(0)로 바뀐 채 detection evaluator로 넘어가 truck/bus/trailer AP가 망가지고 car AP가 오염된다. VAD는 `copy.deepcopy` 후 계산한다(`VAD.py:397`). → 사본에서 계산하도록 수정, 회귀 테스트 추가.
> 2. **score 필터 누락.** VAD는 motion 평가 전 `scores_3d > 0.6`을 적용한다. 없으면 디코딩된 저신뢰 쿼리 300개가 전부 매칭과 FP 계산에 들어가 EPA의 FP 항이 논문과 비교 불가능해진다. → `motion_score_thresh=0.6` 추가.

**지표 이름 주의:** `motion_cnt_{cls}/matched`는 실제로 `cnt_ade` — **미래가 한 스텝이라도 유효한** 매칭 수다. 반면 `gt`는 **6 스텝 전부 유효한** GT 수다. 기준이 다르므로 `matched > gt`가 나올 수 있다 (실제로 positive control에서 pedestrian이 gt 51, matched 64였다). 버그가 아니다.

### 7.3 부수적으로 고친 버그 3개

**(a) planning 집계 오염** — 기존 `evaluate()`는 `metric_results`의 **모든 키**를 평균내고 있었다. motion 카운터를 넣는 순간 `gt_car` 같은 값이 planning 지표로 둔갑한다. 게다가 `fut_valid_flag`가 True인 샘플만 세므로 motion 입장에서는 잘못된 분모다.

```python
def _is_plan_key(k):
    return k.startswith('plan_') or k == 'fut_valid_flag'
```

**(b) nuScenes devkit 초기화 실패** — 이 데이터 루트에는 `v1.0-trainval/lidarseg.json`이 있는데 그것이 색인하는 `lidarseg/` 라벨 파일이 없다. devkit이 `__init__`에서 assert로 죽는다. **detection 평가는 lidarseg를 쓰지도 않는데** 막힌다. SSR은 detection 평가를 돌린 적이 없어서 드러나지 않았던 잠복 문제다.

사용자 데이터를 건드리지 않고 `_load_nuscenes()`가 심볼릭 링크 shadow root를 만들어 그 json 하나만 빼고 재시도한다. (심볼릭 링크 타깃은 링크 위치 기준으로 해석되므로 절대 경로여야 한다 — 처음에 상대 경로로 만들었다가 전부 깨졌다.)

**(c) 죽은 head가 만점으로 보이는 지표** ★

첫 통합 실행에서 미학습 모델이 이렇게 나왔다:

```
car   EPA -0.6667  ADE 0.0000  FDE 0.0000  MR 0.0000
```

`ADE 0.0000`은 "완벽"이 아니라 **매칭이 하나도 없어서 분모가 0**이었던 것이다. 내 초안이 분모를 `max(n, 1)`로 클램프하고 있었다. **죽은 head를 잡으려고 만든 지표가 죽은 head에게 만점을 주는** 셈이다.

> **정정.** 초판은 "VAD 원본도 `max(n,1)`로 클램프한다"고 적었는데 **틀렸다.** VAD는 그냥 직접 나눈다(`ADE / cnt_ade`). 즉 VAD는 분모가 0이면 `ZeroDivisionError`로 죽거나 NaN이 나오는, 다르지만 역시 안전하지 않은 구현이다. 클램프는 순전히 내가 넣은 것이고 그게 더 나빴다.

```python
def _div(num, den):
    return float(num) / float(den) if den > 0 else float('nan')
```

카운트도 함께 출력해 NaN이나 이상값을 해석할 수 있게 했다:

```
car  EPA -4.0000  ADE nan  FDE nan  MR nan   [gt 15, matched 0, hit 0, fp 120]
   ^ no car matched: ADE/FDE are undefined, not zero
```

### 7.4 검증

**A. 실제 모델 출력**
```
boxes_3d (100,)  scores_3d (100,)  labels_3d (100,)  trajs_3d (100, 6, 12)
motion counters present: 16
```

**B. 제출 json ↔ 평가기 역직렬화** — detection 평가는 전체 val 커버리지를 요구하는 assert(`set(pred_tokens) == set(gt_tokens)`)가 있어 짧은 런으로는 호출할 수 없다. 그래서 평가기가 실제로 쓰는 클래스로 직접 검증했다:
```
EvalBoxes.deserialize(data['results'], DetectionBox)
  -> 10 samples / 1000 boxes OK
  max_boxes_per_sample=500, samples over the cap: 0
```
스키마가 틀렸다면 여기서 걸린다.

> ⚠️ **`NuScenesEval_custom.main()`은 아직 전체 6,019 샘플로 실행한 적이 없다.** 검증된 것은 제출 형식이 평가기의 역직렬화를 통과한다는 것까지다. 정확한 표현은 **"det mAP/NDS 배선 완료"가 아니라 "스키마 확인 완료, full-val 미검증"**이다. 첫 60-epoch 실행의 epoch-10 validation이 이것의 첫 실전 통과가 된다.

**C. Motion positive control** — GT 박스 + GT 궤적을 예측으로 투입:
```
car         EPA 1.0000  ADE 0.0000  FDE 0.0000  MR 0.0000  [gt 21, matched 21, hit 21, fp 0]
pedestrian  EPA 1.0000  ADE 0.0000  FDE 0.0000  MR 0.0000  [gt 51, matched 64, hit 51, fp 0]
```

**D. 죽은 head 대조군** — 예측은 있는데 매칭이 0인 경우:
```
car  EPA -4.0000  ADE nan  FDE nan  MR nan  [gt 15, matched 0, hit 0, fp 120]
```
**완벽(ADE 0.0000)과 죽음(ADE nan)이 구분된다.** (c) 수정의 요점이다.

**E. 4개 head 동시 실행 회귀** (12샘플, 미학습)
```
-------------- Planning --------------
-------------- Detection --------------
  detection evaluation skipped: Samples in split doesn't match samples in predictions.
-------------- Motion Prediction --------------
-------------- Map (vectorised) --------------  map (chamfer): 0.0
-------------- Occupancy --------------          thr 0.1: cls0 0.0238
VERDICT: wiring OK
```
detection이 커버리지 부족으로 건너뛰어도 **나머지 지표가 살아남는다.** `evaluate_det`이 `AssertionError`를 잡아 `{}`를 반환하도록 한 결과다 — det 하나 때문에 map/occ/planning을 통째로 잃으면 안 된다.

---

## 8. 지표 해석 시 주의 (측정 방식의 한계)

**학습 중 `occ_iou*` / `map_*` / `gshare*`는 진단용이지 정확한 전역 지표가 아니다.**

1. **LogBuffer 누적 평균.** 계산은 `aux_metric_log_interval`(200)마다인데 로깅은 `log_config.interval`(100)마다다. `log_buffer.average(100)`이 `val_history[-100:]`를 평균할 때 이 키에는 epoch 시작 이후의 측정값만 들어 있으므로, **iter 300에는 iter 200 값이 그대로 다시 찍히고 iter 400에는 (200,400)의 평균이 찍힌다.** 각 epoch의 마지막 항목이 그 epoch의 참 평균이다. 순간값은 `tools/analyze_gshare.py`가 복원한다.
2. **DDP는 rank별 비율을 평균한다.** IoU는 global intersection/union이 아니라 **rank별 IoU의 산술평균**이고, `gshare`도 rank별 비율의 평균이다 (`mean_r‖g_r‖ ≠ ‖mean_r g_r‖`).
3. **`gshare`는 `aux_grad_scale` 적용 후 값이다.** `_bev_grad_norms`는 "실제로 trunk에 도달하는" gradient를 재므로, `aux_grad_scale=0.01`이면 aux gnorm이 이미 100배 줄어든 상태로 찍힌다. 목표 점유율을 역산할 때는 **scale을 끈 상태의 측정값**을 써야 한다.

정확한 숫자가 필요하면 10 epoch마다 도는 validation(`occ_iou_thr*`, `NuscMap_chamfer/*`)을 볼 것.

---

## 9. 알려진 제약

- **`NuScenesEval_custom.main()` full-val 미검증** (§7.4-B). 첫 epoch-10 validation이 실전 통과 시점이다.
- **detection mAP는 전체 val 커버리지가 필요하다.** 부분 실행은 명확한 메시지와 함께 건너뛴다(§7.4-E). motion 지표에는 이 제약이 없다.
- **결과 텐서 메모리.** occupancy 결과를 6019 샘플 전부 모으면 `[1,7,C,100,100]` float32 + int64 `occ_seg`로 **~6.3 GiB**였다. 평가기는 확률만 쓰므로 `occ_seg`를 빼고 `occ_scores`를 float16으로 저장해 **~0.8 GiB**로 줄였다.
- **map GT 캐시 최초 생성 30분.** `map_ann_file` 경로에 한 번만 생기고 재사용된다.
- **`evaluate_map`도 전체 val을 요구한다** (위치 기반 짝짓기). 부분집합 평가는 `data_infos` 절단 + `map_ann_file` 별도 경로가 필요하다(검증 스크립트가 쓰는 방식).
- **`lidarseg.json` 우회는 증상 대응이다.** 근본 해결은 `data/nuscenes/v1.0-trainval/lidarseg.json`을 지우거나 실제 lidarseg 라벨을 받는 것이다. 다른 프로젝트가 그 파일에 의존할 수 있어 건드리지 않았다.
