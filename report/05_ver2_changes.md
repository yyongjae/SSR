# 보고서 #05 — PARA-SSR ver2 수정 내역

작성일 2026-08-17 · 브랜치 `para` · 대상: 1차 실험(#01) 진단 결과 반영 + multibatch 지원

---

## 0. 요약

| # | 항목 | 상태 |
|---|---|---|
| 1 | occupancy에서 pedestrian 채널 제거 | ✅ 완료 · 검증됨 |
| 2 | `aux_grad_scale`이 BEV gradient만 줄이는지 확인 | ✅ **원래부터 그렇게 설계됨** · 수치 증명 |
| 3 | map head를 dense raster → VAD 벡터로 전환 | ✅ 완료 · 설계 논의 §3 |
| 4 | multibatch 수정(`SSR-orig@8fe89d5`)을 para에 이식 | ✅ **위상** 등가성 검증 · 최종 metric 등가성은 §4.8 |
| 5 | 2 GPU × 4 samples config | ✅ `PARA_SSR_e2e_2gpu_b4.py` |
| 6 | 보고서 #01·#02 오진단 정정 | ✅ 완료 · §6 |
| 7 | 코드 리뷰 대응 (deformable init 파괴 등) | ✅ §10 |

---

## 1. Occupancy — pedestrian 제거

### 변경

`projects/configs/SSR/PARA_SSR_e2e.py`

```python
occ_class_labels = (
    (0, 1, 2, 3, 4, 6, 7),  # vehicle only
)
occ_num_classes = 1          # was 2
```

`occ_class_labels`가 GT 파이프라인(`GenerateSSROccLabels`)과 헤드(`ParaOccHead.num_classes`) 양쪽의 단일 출처라 이 한 곳만 바꾸면 전체가 따라온다.

### 근거 (실측)

| | vehicle | pedestrian |
|---|---:|---:|
| GT 양성률 (BEV 셀) | 15.50% | **0.162%** |
| 프레임당 인스턴스 | 11.2개 | 1.78개 |
| 인스턴스당 블롭 크기 | 평균 124 px / 중앙값 78 | **평균 7.2 px / 중앙값 7** |
| IoU @0.5 | 0.574 | 0.021 |
| IoU @최적 threshold | 0.574 @0.5 | 0.104 @0.05 |
| **GT 양성 픽셀 recall @0.05** | **97.9%** | **20.2%** |
| top-k BCE 균형점 `n_pos/2500` | 0.620 | **0.0064** |

세 가지가 겹쳤다:

1. **해상도 한계.** 입력은 1600×900을 0.4× 축소한 640×384, FPN은 C5(stride 32) 단일 레벨. 20 m 거리의 보행자는 입력에서 세로 ~12 px, C5에서는 **0.4 px** — feature map 셀 하나도 채우지 못한다.
2. **BEV 격자 한계.** 0.3 m × 0.6 m 셀에서 보행자 발자국은 중앙값 7 px. 한 셀만 어긋나도 IoU가 급락한다.
3. **손실 함수가 동작 범위를 좁힌다.** top-k BCE에서 **분별력이 없는** 픽셀이 안착하는 확률은 `n_pos/k = 16/2500 = 0.0064`다. vehicle은 이 값이 0.62라 threshold 0.5를 자연스럽게 넘지만 보행자는 78배를 올려야 한다.

> **정정.** 초판은 이를 근거로 "학습을 아무리 오래 해도 0.5에 도달 불가"라고 썼는데 **과도한 단정이었다.** `n_pos/k`는 **모든 위치에 같은 logit을 내는 붕괴 상태의 균형점**이지 출력의 수학적 상한이 아니다. 실제로 보행자의 실측 확률 0.052는 이미 균형점의 8배이고 vehicle도 0.692 > 0.620이다. 분별력이 있으면 균형점을 넘어설 수 있다.
>
> 정확한 서술: **현재 해상도·표현에서 실험적으로 유용한 동작점을 얻지 못했고, 손실 함수가 그 동작 범위를 매우 좁게 만든다.**

> **정정 2.** "recall 20%라 나머지 80%는 어떤 threshold로도 못 잡는다"도 문자 그대로는 틀렸다. 20.2%는 **thr 0.05에서의** recall이고, 더 낮추면 recall은 올라간다(0.05 미만은 측정하지 않았다). 정확한 서술: **쓸 만한 precision/IoU를 유지하면서 recall을 끌어올리는 threshold가 없었다** — IoU가 최대가 되는 지점(0.05)에서조차 recall이 20%였다.

### 선행 연구와의 정합

| 논문 | occupancy 대상 |
|---|---|
| UniAD | `only_vehicle=True` — `['car','bus','construction_vehicle','bicycle','motorcycle','truck','trailer']` |
| FIERY | vehicle |
| ST-P3 | vehicle |
| PARA-Drive | UniAD 방식 계승 |

UniAD 코드에 `plan_classes = vehicle_classes + ['pedestrian']`가 정의돼 있긴 하지만 릴리스 config는 train/test 파이프라인 모두 `only_vehicle=True`다. **카메라 기반 BEV occupancy에서 보행자를 별도 채널로 예측하는 주요 E2E 연구는 없다.**

### 검증

```
occ_class_labels = ((0, 1, 2, 3, 4, 6, 7),)
occ_num_classes  = 1
gt_occ_seg       (1, 7, 1, 100, 100)   양성 px/프레임: 1754 → 1766
occ logits       (1, 7, 1, 100, 100)
losses           loss_occ_mask 4.093 / loss_occ_dice 0.742
params with gradient: 11/11
```

---

## 2. `aux_grad_scale` — BEV gradient만 줄이는가?

### 답: 원래부터 정확히 그렇게 설계되어 있다

질문하신 "task head 말고 BEV gradient만"이 `_ScaleGrad`의 존재 이유다. 구현은 `para_ssr.py`:

```python
class _ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x                      # 순전파는 항등 — head는 같은 값을 봄

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None # 역전파에서 BEV로 흘러가는 것만 축소
```

사용:
```python
aux_bev = _ScaleGrad.apply(bev_embed, self.aux_grad_scale)
det_losses = self.det_motion_head.forward_train(aux_bev, ...)
```

### 왜 head 파라미터는 영향을 받지 않는가

head 파라미터의 gradient는 `∂L/∂W`이고, 이 계산 경로는 `_ScaleGrad` 노드를 **통과하지 않는다.** `_ScaleGrad`는 head **입력** 위치의 노드이므로, head 내부에서 손실까지 올라간 gradient가 head 파라미터로 갈라져 나갈 때는 이미 분기가 끝난 뒤다. scale은 그 분기점을 **더 지나서** BEV 쪽으로 계속 내려가는 갈래에만 곱해진다.

```
      loss
       │  ∂L/∂(head 중간활성)
       ▼
   [head layers] ──분기──► head 파라미터 grad   (scale 영향 없음)
       │
       ▼  ∂L/∂aux_bev
   [_ScaleGrad]  × scale        ← 여기서만 축소
       │
       ▼
   bev_embed → encoder → FPN → backbone   (전부 scale 적용됨)
```

### 수치 증명

동일 weight, 동일 입력, scale만 1.0 → 0.1:

```
loss value            5.283404  ->  5.283404    ratio 1.0000
||grad @ bev_embed||  1.185e-02 ->  1.185e-03   ratio 0.1000   ← 정확히 0.1배

head parameter gradients (전 11개):
  pred_head.weight    3.247644e+00 -> 3.247644e+00   ratio 1.0000
  trunk.0.weight      1.000503e+00 -> 1.000503e+00   ratio 1.0000
  ... (모두 ratio 1.0000)

BEV gradient scaled by exactly 0.1 : True
head gradients unchanged (ratio 1) : True
loss value unchanged               : True
VERDICT: BEV-only scaling CONFIRMED
```

### `task_loss_weight`와의 차이 (중요)

두 손잡이가 있고 용도가 다르다:

| | `task_loss_weight` | `aux_grad_scale` |
|---|---|---|
| 손실 값 | 바뀜 | **안 바뀜** |
| head 파라미터 gradient | 축소됨 | **원본 유지** |
| BEV로 가는 gradient | 축소됨 | 축소됨 |
| head 학습 속도 | **느려짐** | 유지 |

**KD 실험에는 `aux_grad_scale`이 맞다.** aux head를 teacher로 쓰려면 head 자체는 최대한 잘 학습돼야 하는데, `task_loss_weight`를 낮추면 teacher 품질까지 같이 떨어진다.

### 권장값 (#01 측정 기반)

epoch 12의 plan BEV gradient 점유율 **4.62%**. 목표 점유율 X에 필요한 scale:

```
s = 0.0462 · (1/X − 1) / 0.9538

  X = 0.30  →  s ≈ 0.113
  X = 0.50  →  s ≈ 0.049
```

`PARA_SSR_w_gradscale.py`의 **s=0.02는 plan 점유율 ~70%를 겨냥하는 과교정**이다. `0.11` / `0.05`로 sweep을 권한다.

### `gshare/*` 로그의 정확도 한계 (중요)

이 값을 정밀한 전역 점유율로 읽으면 안 된다. 두 가지 이유가 있다.

**(1) DDP가 rank별 *비율*을 평균한다.** `_bev_grad_norms`는 각 rank에서 `‖∂L_task/∂bev‖`을 구해 비율을 낸 뒤, `_parse_losses`의 `all_reduce`가 그 스칼라를 평균한다. 일반적으로

```
mean_r ‖g_r‖  ≠  ‖mean_r g_r‖
```

**(2) LogBuffer가 epoch 내 누적 평균을 낸다.** 측정은 200 iter마다, 로깅은 100 iter마다이므로 `log_buffer.average(100)`이 `val_history[-100:]`를 평균할 때 그 키에는 epoch 시작 이후의 모든 측정값만 들어 있다. 결과적으로 **iter 300에는 iter 200 값이 그대로 다시 찍히고, iter 400에는 (200, 400)의 평균이 찍힌다.**

따라서 **각 epoch의 마지막 로그 항목이 그 epoch의 참 평균**이다. 그 값으로 다시 계산했다:

| ep | 로그값의 단순 평균 (초판) | **마지막 항목 = 참 평균** | det | map | occ |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.9% | **3.2%** | 92.7 | 0.9 | 3.2 |
| 3 | 8.2% | **8.0%** | 74.7 | 8.1 | 9.2 |
| 6 | 5.1% | **4.9%** | 68.9 | 14.3 | 11.9 |
| 12 | 4.7% | **4.62%** | 66.3 | 17.3 | 11.8 |

차이는 작고 §1의 결론(ep3 정점 후 map에 밀려남, 최종 5% 미만)은 그대로다. 다만 **0.113 / 0.049는 sweep의 출발점이지 "정확히 30%/50%를 만드는 값"이 아니다.**

> 주의: map을 VAD 벡터로 바꾸면서 파라미터가 7배(0.41M → 2.91M), 손실 항이 5개×3층으로 늘었다. gradient 크기가 달라지므로 **1 epoch 학습 후 `gshare/*`로 재보정**해야 한다. `grad_norm_log_interval=200`이 켜져 있으니 초반 로그만 봐도 판단할 수 있다. 정밀한 값이 필요하면 로깅 간격을 맞추거나(`grad_norm_log_interval == log_config.interval`) 각 epoch 마지막 항목만 읽을 것.

---

## 3. Map head — 지금 구조와 설계 판단

### 3.1 용어 정리

"dense map"과 "raster map"은 같은 것을 가리킨다. 실제 축은 이것이다:

| 축 | 의미 |
|---|---|
| **raster / dense** | BEV 격자에 클래스별 마스크를 그림. 출력 `[B, C, 100, 100]` |
| **vector** | 폴리라인 인스턴스를 점 좌표열로 회귀. 출력 `[B, 100, 20, 2]` + 클래스 |

### 3.2 현재 상태

**어제까지**: `ParaMapSegHead` — **raster/dense**, conv 3층 0.41M
**지금**: `ParaMapHead` — **VAD 벡터**, 100 인스턴스 × 20 점, 디코더 3층 2.91M

바꾼 이유는 §6과 보고서 #02에 있다. 요약하면 dense 버전은 **boundary 채널이 전 픽셀에 ≈0.29를 출력하는 상수 함수로 수렴**했고(분별비 1.06×), 그러면서 BEV gradient의 17.7%를 소비했다.

### 3.3 "단순히 VAD를 가져오는 게 정답인가"

**정답은 아니다. 다만 지금 단계에서는 맞는 선택이다.** 이유를 나눠서:

**VAD 벡터가 지금 맞는 이유**
- 우선 확인해야 할 것은 "aux head가 학습이 되긴 하는가"다. **죽은 teacher는 KD 대상이 될 수 없다.**
- VAD-Tiny에서 학습성이 검증된 구성이고, SSR이 바로 그 VAD-Tiny 기반이다. 데이터 파이프라인(`VectorizedLocalMap`, `MapHungarianAssigner3D`, `PtsL1Loss`, `PtsDirCosLoss`)이 이미 전부 있다.
- dense가 실패한 근본 원인이 **형상 사전지식의 부재**인데, 벡터 쿼리 임베딩은 "차선은 길고 매끄러운 곡선"이라는 사전지식을 내장한다. 0.4× 축소된 C5 feature처럼 근거가 약할 때 이 차이가 크다.

**VAD 벡터가 KD 목적에는 불리한 이유**
- Hungarian 매칭 때문에 **teacher의 쿼리 k와 student의 쿼리 k가 같은 대상을 가리킨다는 보장이 없다.** teacher 쿼리 37번과 student 쿼리 12번이 같은 차선일 수 있다.
- 이걸 우회하려면 매칭 결과를 teacher→student로 전달하거나 별도 정렬이 필요한데, 그 순간 "깔끔한 구조"라는 원래 목적이 훼손된다.

**그래서 제 권고 — KD 인터페이스를 head 출력이 아닌 곳에 두는 것**

| 레벨 | KD 대상 | 표현 의존성 | 평가 |
|---|---|---|---|
| **L1. feature KD** | teacher BEV → student BEV | **없음** | ⭐ 가장 견고. head가 벡터든 dense든 무관 |
| L2. dense logit KD | 픽셀별 로짓 | dense head 필요 | occ는 이미 dense라 즉시 가능 |
| L3. query KD | 쿼리 임베딩 | 매칭 필요 | 비권장 |

PARA-Drive의 전제 자체가 "aux task가 BEV를 형성한다"이므로, **"aux task가 BEV에 가르친 것"을 증류하려면 BEV를 증류하는 게 논리적으로도 가장 직접적**이다. 이렇게 하면 map head 표현은 KD 설계와 분리되어 순수하게 "가장 잘 학습되는 것"으로 고를 수 있다.

**중기 선택지 (필요해지면)**

- **(a) 하이브리드**: 벡터로 학습하고, KD 시점에만 폴리라인을 미분가능 래스터화해 dense 로짓으로 변환. 학습성과 픽셀 정렬을 동시에 얻는다. 구현 비용 중간.
- **(b) dense 재시도**: 다중 스케일 FPN + 더 큰 디코더 + focal/dice + thickness ≥3. PARA-Drive에 가장 충실하지만, §6에서 본 근본 원인(C5 단일 레벨에서 노면 표시 소실)은 백본을 안 건드리면 남는다.
- **(c) map 제거**: PARA-Drive Table 5는 map 모듈 제거 시 L2/CR이 거의 안 변하고 map-compliance만 나빠진다고 보고. SSR Appendix E Table 6도 map aux 추가 시 planning이 **나빠졌다**고 보고(L2 0.75 → 0.79~0.86). 실험 축을 줄이려면 유효한 선택.

**단계별 권고**: 지금은 VAD 벡터로 "학습되는가"를 확인 → BEV feature KD로 실험 설계 → map 표현은 나중에 ablation 축으로.

### 3.4 왜 drivable_area가 없고 3개 클래스인가

코드를 확인했다. `nuscenes_vad_dataset.py`:

```python
class VectorizedLocalMap:
    CLASS2LABEL = {'road_divider': 0, 'lane_divider': 0,
                   'ped_crossing': 1, 'contours': 2, 'others': -1}

    map_classes = ['divider', 'ped_crossing', 'boundary']
    contour_classes = ['road_segment', 'lane']       # → 'boundary'
```

`gen_vectorized_samples()`는 이 3개만 분기 처리하고 그 외에는 `ValueError`를 던진다.

**이유는 표현 때문이다.** VAD의 벡터는 **열린 폴리라인**을 인코딩한다. `drivable_area`는 폴리라인이 아니라 **면(폴리곤)**이라 이 형식에 담기지 않는다. 굳이 벡터화하면 그 **경계선**이 되는데, 그건 이미 `boundary`(= `road_segment` + `lane`의 컨투어)가 하고 있다. **벡터 표현에서 drivable_area는 boundary와 중복이다.**

반면 UniAD/PARA-Drive는 dense 세그멘테이션이라 drivable_area를 "채워진 영역"으로 자연스럽게 4번째 채널에 넣을 수 있다. **클래스 개수 차이는 표현 선택의 결과이지 누락이 아니다.**

> **정정.** 초판은 "dense head로 돌아가면 drivable_area 추가는 **파이프라인 수정 없이 가능**"이라고 썼는데 **틀렸다.** 두 곳을 고쳐야 한다:
> 1. `gen_vectorized_samples()`는 세 클래스 외에는 `raise ValueError(f'WRONG vec_class: {vec_class}')`로 막는다 (`nuscenes_vad_dataset.py:553`). 데이터셋이 drivable_area 지오메트리를 **내보내지 않는다.**
> 2. dense head의 `rasterize_gt()`는 `fixed_num_sampled_points` + `cv2.polylines`로 **선만 그린다** (`para_map_seg_head.py:107`). 면을 채우려면 `cv2.fillPoly` 경로가 필요하다.
>
> 맞는 부분은 `get_map_geom()`이 `layer_name == 'drivable_area'` 분기를 갖고 폴리곤을 반환한다는 것뿐이다(`nuscenes_vad_dataset.py:783`). **저수준 지오메트리 접근은 있지만 데이터셋 출력과 rasterizer 양쪽을 고쳐야 한다.**

---

## 4. Multibatch 이식

### 4.1 배경

`SSR-orig` 커밋 `8fe89d5` "Support topology-equivalent multibatch training"이 원본 SSR의 `samples_per_gpu > 1` 버그 3개를 고쳤다. `para` 브랜치는 이 커밋의 후손이 아니라 같은 뿌리(`a483198`)에서 갈라졌으므로 별도 이식이 필요했다.

### 4.2 공유 모듈 — main 버전 그대로 반입

먼저 para 브랜치에서 해당 파일들이 **전혀 수정되지 않았음**을 md5로 확인한 뒤 `git checkout 8fe89d5 -- <paths>`로 가져왔다 (충돌 위험 0).

| 파일 | 고친 버그 |
|---|---|
| `modules/encoder.py` | `img_metas[0]['img_shape'][0]`으로 **모든 샘플·카메라**를 정규화. 샘플별 `[B, num_cam, 2]` 텐서로 교체 |
| `modules/spatial_cross_attention.py` | `bev_mask[cam][0]` — **샘플 0의 카메라 가시성**을 배치 전체에 적용. 유효 쿼리를 버리거나 무효 쿼리를 모았음. `(sample, camera)`별 인덱스로 교체 |
| `modules/temporal_self_attention.py` | `value[:bs]` — 인코더는 `[prev0, cur0, prev1, cur1, ...]` 순으로 flatten하므로 B>1에서 **샘플이 섞였음**. queue 축 복원 후 선택 |
| `SSR_head.py` | 원본 SSR planning head (para에서는 `SSR` detector가 사용) |

### 4.3 `para_ssr_head.py` — 동일 로직 수동 이식

`ParaSSRHead`는 `SSRHead`의 planning-only 복제본이라 같은 버그가 있었다.

```python
# before
cmd = cmd[0, 0, 0]                                    # ← 샘플 0의 명령을
cmd_idx = torch.nonzero(cmd)[0, 0]                    #   배치 전체에 적용
navi_embed = self.navi_embedding.weight[cmd_idx][None, None]

# after
if cmd is None:
    raise ValueError('cmd is required when only_bev=False')
num_commands = self.navi_embedding.num_embeddings
if cmd.size(0) != bs or cmd.numel() != bs * num_commands:
    raise ValueError(...)                             # 형상이 틀리면 조용히 넘어가지 않고 실패
cmd = cmd.reshape(bs, num_commands)
cmd_idx = cmd.argmax(dim=-1)                          # [B]
navi_embed = self.navi_embedding(cmd_idx).unsqueeze(1)  # [B, 1, C]
```

원본 `SSRHead`와 달리 `ParaSSRHead`에는 FFP의 `action_mln` / `wp_vector` 경로가 없어서 이 한 곳만 고치면 된다.

### 4.4 추가 발견 — occupancy frame mask (para 고유 버그)

`ParaOccHead.loss()`가 배치 전체에 **하나의 `[T]` 마스크**를 쓰고 있었다:

```python
frame_mask = gt_occ_valid[:, :T].bool().any(dim=0)   # ← B>1에서 부정확
```

`any(dim=0)`이므로 **한 샘플이라도 유효하면 그 프레임은 전 샘플에 대해 유효 처리**된다. 씬 끝을 넘어간 샘플의 미래 프레임이 "전부 비어 있음"으로 학습된다. bs=1에서는 정확했지만 bs=4에서는 실제 오류다.

수정:

```python
# para_occ_head.py — 인스턴스 축(batch-major)에 맞춰 샘플별 마스크
frame_mask = gt_occ_valid[:, :self.num_frames].bool()
frame_mask = frame_mask.repeat_interleave(c, dim=0)   # [B*C, T]
```

```python
# occ_loss.py — [T] 와 [n, T] 를 모두 받도록 일반화
def _expand_frame_mask(frame_mask, n_gt, s):
    mask = frame_mask.float()
    if mask.dim() == 1:
        assert mask.size(0) == s
        return mask.view(1, s)          # 브로드캐스트 (기존 동작 보존)
    assert tuple(mask.shape) == (n_gt, s)
    return mask
```

`OccBinarySegmentationLoss`와 `occ_dice_loss` 양쪽에 적용. `[T]` 입력은 이전과 동일하게 동작하므로 하위 호환된다.

### 4.5 검증 1 — bs=2 학습 + 샘플별 명령 라우팅

```
--- 1. bs=2 forward_train ---
  32 loss terms, all finite. loss_plan_reg=0.030388
  pts_bbox_head        155/159 params with gradient
  det_motion_head      262/262 params with gradient
  map_head             118/118 params with gradient
  occ_head              11/11  params with gradient

--- 2. per-sample navigation command (예전 bs>1 버그) ---
  sample 1 command 2 -> 0
  |delta| sample 0 (명령 그대로): 0.000e+00   expect 0    ✅
  |delta| sample 1 (명령 변경  ): 1.541e-02   expect > 0  ✅
  per-sample command routing: OK
```

**수정 전에는 sample 1의 delta가 정확히 `0.000e+00`이었다** — 자기 명령을 무시하고 sample 0의 명령으로 주행하고 있었다는 뜻이다. 이제 자기 명령에 반응하고, 동시에 sample 0은 전혀 영향받지 않는다.

### 4.6 검증 2 — bs=4가 bs=1과 등가인가

**같은 샘플**을 batch 4의 원소로 한 번, 단독으로 한 번 넣어 비교했다 (dataloader 차이를 배제).

| sample | max&#124;Δ&#124; plan | rel plan | max&#124;Δ&#124; bev | rel bev |
|---:|---:|---:|---:|---:|
| 0 | 9.161e-05 | 6.21e-04 | 2.122e-03 | 4.66e-04 |
| 1 | 1.509e-04 | 1.04e-03 | 2.113e-03 | 4.71e-04 |
| 2 | 1.061e-04 | 7.55e-04 | 2.699e-03 | 6.18e-04 |
| 3 | 1.221e-04 | 7.67e-04 | 2.213e-03 | 4.61e-04 |

상대오차 ~1e-3은 float32 노이즈(~1e-6)보다 크므로 원인을 특정했다:

```
=== A. ResNet50+FPN 단독, 이미지 24장 vs 6장 ===
  relative diff on sample 0 features: 6.049e-04
=== B. 동일 + cudnn.deterministic=True, benchmark=False ===
  relative diff on sample 0 features: 5.471e-04        ← 거의 안 줄어듦
=== C. 완전히 동일한 이미지를 4번 타일링해서 투입 ===
  backbone rel diff : 5.471e-04
  bev_embed rel diff: 5.347e-04     ← BEV 인코더가 증폭하지 않음
  plan      rel diff: 8.435e-04
  한 배치 안 4개 동일 복사본 사이의 편차: 0.000e+00   ★
```

**결론:**
1. 차이는 **ResNet50/FPN에서 발생**한다. 배치 크기가 바뀌면 cuDNN이 다른 convolution 알고리즘/타일링을 선택하고, 누적 순서가 달라진다. `cudnn.deterministic=True`로도 사라지지 않는다(알고리즘 선택 자체가 배치 형상에 의존하므로). **프레임워크·모델 무관하게 배치 크기를 바꾸면 항상 생기는 현상이다.**
2. BEV 인코더는 이를 **증폭하지 않는다** (5.47e-04 → 5.35e-04). SCA의 `max_len` 패딩이 실제 쿼리를 오염시키지 않는다는 뜻이다.
3. 한 배치 안의 동일 입력 4개가 비트 단위로 같은 출력을 낸다(0.000e+00) — 배치 내 **위치** 효과가 없다는 뜻이다.

planning의 상대오차(8.4e-04)가 BEV(5.3e-04)보다 약간 큰 것은 16-token TokenLearner 병목이 작은 변화를 증폭하기 때문이다 — 보고서 #01 §7에서 관측한 SSR의 민감성과 일관된다.

> **정정.** 초판은 위 3번(동일 입력 4복사본이 같은 출력)을 "샘플 간 누수가 없다는 **결정적** 증거"라고 썼는데 **과장이었다.** 입력이 모두 같으면 sample-0 broadcast 버그도 이 테스트를 통과한다. 배치 내 위치 효과만 배제할 뿐이다. 제대로 된 개입 테스트를 §4.7에 추가했다.

### 4.7 누수 개입 테스트 (제대로 된 버전)

샘플 하나의 **이미지를 난수로 파괴**하고 나머지 샘플의 출력이 비트 단위로 불변인지 확인한다. cmd만 뒤집는 §4.5와 달리 backbone / `lidar2img` / 카메라 가시성 / `prev_bev` 경로를 전부 지나간다.

| 파괴한 샘플 | sample 0 Δ | sample 1 Δ | sample 2 Δ | sample 3 Δ |
|---:|---:|---:|---:|---:|
| 0 | **5.301e-02** | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 1 | 0.000e+00 | **5.082e-02** | 0.000e+00 | 0.000e+00 |
| 2 | 0.000e+00 | 0.000e+00 | **5.198e-02** | 0.000e+00 |
| 3 | 0.000e+00 | 0.000e+00 | 0.000e+00 | **6.091e-02** |

대각선만 0이 아니고 비대각 12칸이 **전부 정확히 0**이다. 각 샘플은 자기 입력에만 반응한다.

### 4.8 bs=4 전체 학습 경로 · DDP

forward 비교만으로는 부족하다는 지적을 받아 실제 학습 경로를 검증했다.

**단일 GPU, bs=4, 전 head 활성, forward_train → backward → grad clip → AdamW step:**

```
loss terms: 32   total loss = 63.5407
non-finite loss terms: none
grad norm before clip: 46.710          (config의 max_norm=35로 클리핑됨)
params with non-finite grad: 0
parameters updated by the step: 592/713
non-finite parameters after step: 0
PEAK GPU MEMORY: 16.60 GiB             (49 GiB 카드)
```

업데이트되지 않은 121개는 동결 파라미터다 — `frozen_stages=1`의 ResNet stem/layer1과 `norm_cfg=dict(requires_grad=False)`의 모든 BN.

**2-rank DDP (GPU 5,7), bs=4:**

```
Epoch [1][5/3517]  time: 7.536, data_time: 5.367, memory: 17755
  loss_plan_reg: 0.4321
  det.*     (6 layers x cls/bbox + traj + traj_cls)
  map.*     (3 layers x cls/pts/dir, bbox·iou는 weight 0이라 0.0000)
  occ.loss_occ_mask: 3.5750, occ.loss_occ_dice: 0.9278
  loss: 60.3959, grad_norm: 60.2359
```

hang 없음, unused-parameter 오류 없음(`find_unused_parameters = True`가 config:602에 이미 있음), 32개 손실 전부 유한.

### 4.9 검증되지 않은 것 — 최종 metric 등가성

**8×1과 2×4가 같은 최종 성능을 낸다는 것은 검증하지 않았고, 원리적으로 사전 검증이 불가능하다.** §4.6에서 측정한 ~1e-3의 cuDNN 차이는 12 epoch 학습을 거치며 다른 국소해로 갈라진다.

검증된 것과 아닌 것을 명확히 구분하면:

| 항목 | 상태 |
|---|---|
| 샘플별 명령 라우팅 | ✅ 검증 |
| 샘플 간 누수 없음 (이미지 개입) | ✅ 검증 |
| 배치 내 위치 효과 없음 | ✅ 검증 |
| bs=4 forward + backward + optimizer step | ✅ 검증 |
| 2-rank DDP 정상 동작 | ✅ 검증 |
| epoch당 optimizer step 수 · LR 스케줄 동일 | ✅ 3,517 step 확인 |
| **최종 metric 등가성 (L2 / CR)** | ⏳ **진행 중인 SSR-orig 2×4 실행으로 확인 예정** |

지금 도는 `ssr_noffp_2gpu_b4`가 8-GPU 실행의 L2 0.737 / CR 0.221%에 근접하면 실증된다.

### 4.7 새 config / 스크립트

`projects/configs/SSR/PARA_SSR_e2e_2gpu_b4.py`

```python
_base_ = ['./PARA_SSR_e2e.py']
data = dict(samples_per_gpu=4, workers_per_gpu=16)
log_config = ...   # wandb name='para_ssr_2gpu_b4'
```

`train_para_2gpu_b4.sh` — `--no-validate`, epoch마다 체크포인트, `--autoscale-lr` 없음(global batch가 이미 8).

**`workers_per_gpu` 4 → 16**: 실험 의미와 무관한 dataloader 병렬도 값이다. 이 값을 안 올리면 GPU가 65% 유휴가 된다 (측정: FFP 실행에서 4.83 s/iter 중 data_time 3.12 s vs 8-GPU의 1.22 s / 0.12 s). 샘플당 3프레임 × 6카메라 JPEG를 디코딩하므로 GPU를 4분의 1로 줄여도 CPU 부하는 그대로다. 워커 총 32개(2×16)로 8-GPU 실행(8×4)과 같아진다. 머신은 64코어.

실측 효과 (SSR-orig no-FFP 실행):

| | workers=4 | workers=16 |
|---|---:|---:|
| time/iter | 4.83 s | **1.33 s** |
| data_time | 3.12 s (65%) | **0.35 s (23%)** |
| ETA (12 ep) | 4일 2시간 | **16시간 20분** |

---

## 5. 현재 진행 중인 학습

`SSR-orig` no-FFP 2GPU×b4 — GPU 1,2

```
Epoch [1][2500/3517]  lr: 5.000e-05, eta: 16:19:35, time: 1.522,
                      data_time: 0.354, memory: 9594,
                      loss_plan_reg: 0.0559, loss: 0.0559, grad_norm: 1.0273
```

`loss_bev` 없음 = FFP 꺼짐 확인. 완료 예상 08-18 오전.

> **경위**: 처음 띄운 것은 `SSR_e2e_2gpu_b4.py`였는데 이 config의 `_base_`가 `SSR_e2e.py`(FFP 포함)라 `loss_bev`가 살아 있었다. 7시간 학습 후 발견해 중단하고 `SSR_noffp_e2e_2gpu_b4.py`로 재시작했다.

비교 기준: 8-GPU no-FFP 실행 **L2 MAX avg 0.737 / CR 0.221%** (논문 Table 3: 0.75 / 0.20). 이번 실행이 여기 근접하면 multibatch 경로도 함께 검증된다.

---

## 6. 보고서 #01 · #02 오진단 정정

1차 실험 진단을 threshold 0.5 한 점에서만 재고 작성했는데, UniAD는 BEV occupancy를 `test_seg_thresh=0.1`로 평가한다. 전 threshold를 다시 재고 GT 양성률도 실측했다.

| 초판 주장 | 실제 | 정정 |
|---|---|---|
| map GT 양성 픽셀 ~1% | divider 8.18%, ped_crossing 1.82%, boundary 9.95% | **틀림** |
| 극단적 불균형 → 전부 배경으로 붕괴 | top-k 25% 적용 시 유효 비율 1:2~1:3, 불균형은 주범 아님 | **틀림** |
| map IoU 0.003 (완전 실패) | 최적 threshold(0.1)에서 mean 0.117 | 수치 정정 |
| ped-occ IoU 0.005 (완전 실패) | 최적 threshold(0.05)에서 0.104 | 수치 정정 |
| map과 ped-occ의 실패 형태가 동일 | **원인이 서로 다름** | **틀림** |

**새로 확정된 원인:**

| 채널 | GT 양성률 | p(GT양성) | p(GT음성) | 분별비 |
|---|---:|---:|---:|---:|
| occ vehicle | 15.50% | 0.6918 | 0.0886 | 7.8× |
| occ pedestrian | 0.162% | 0.0519 | 0.00042 | **124×** |
| map divider | 8.18% | 0.1154 | 0.0702 | **1.6×** |
| map ped_crossing | 1.82% | 0.0548 | 0.0128 | 4.3× |
| map boundary | 9.95% | 0.3024 | 0.2862 | **1.06×** |

- **map**: 위치를 못 찾는다. boundary는 전 픽셀 ≈0.29의 **상수 함수**이고, 그때 IoU 0.0995는 GT 양성률과 정확히 같다("전부 양성"과 동일한 답).
- **ped-occ**: 위치는 찾는다(124×, vehicle보다도 높음). 확률 수준이 낮고 recall이 20%인 게 문제.

**확률 수준을 결정하는 메커니즘도 특정했다.** top-k BCE(k=2500)의 균형점 `p* = n_pos/k`가 6채널 중 5개의 실측 확률을 예측한다:

| 채널 | 예측 p* | 실측 |
|---|---:|---:|
| occ vehicle | 0.620 | 0.692 ✓ |
| occ pedestrian | 0.0064 | 0.052 ✓ |
| map boundary | 0.398 | 0.302 ✓ |
| map ped_crossing | 0.073 | 0.055 ✓ |

각 채널의 출력 확률 수준은 학습 품질이 아니라 **GT 밀도**가 결정한다. 그래서 보행자는 threshold 0.5에 원리적으로 도달할 수 없다.

방향(map 교체, ped-occ 제거)은 초판과 같지만, **근거가 달라졌으므로 후속 조치도 달라진다** — dense map을 되살릴 때 고칠 것은 "클래스 가중치"가 아니라 **수용 영역·해상도·형상 사전지식**이다.

---

## 7. 변경 파일 목록

```
설정 / 실행
M  projects/configs/SSR/PARA_SSR_e2e.py                     occ 1채널, map → VAD 벡터,
                                                            wandb 메타 갱신, 주석 정정
A  projects/configs/SSR/PARA_SSR_e2e_2gpu_b4.py             신규
A  train_para_2gpu_b4.sh                                    신규 (mode 0755)

PARA-SSR 코드
M  projects/mmdet3d_plugin/SSR/para_ssr.py                  aux_grad_scale / _ScaleGrad,
                                                            task_loss_weight, gnorm·gshare 로깅
M  projects/mmdet3d_plugin/SSR/para_ssr_head.py             샘플별 navigation command
M  projects/mmdet3d_plugin/SSR/dense_heads/para_occ_head.py 샘플별 frame mask
M  projects/mmdet3d_plugin/SSR/dense_heads/para_map_head.py deformable init 복구
M  projects/mmdet3d_plugin/SSR/dense_heads/para_det_motion_head.py  deformable init 복구
M  projects/mmdet3d_plugin/SSR/utils/occ_loss.py            [T] / [n,T] mask 지원

공유 모듈 (8fe89d5에서 반입)
M  projects/mmdet3d_plugin/SSR/modules/encoder.py
M  projects/mmdet3d_plugin/SSR/modules/spatial_cross_attention.py
M  projects/mmdet3d_plugin/SSR/modules/temporal_self_attention.py
M  projects/mmdet3d_plugin/SSR/SSR_head.py
```

> **`para_ssr.py`는 반드시 함께 반영해야 한다.** 이 파일 없이 새 config만 적용하면
> `grad_norm_log_interval` / `aux_grad_scale` / `task_loss_weight`에서
> `unexpected keyword argument` 오류가 난다. 초판 목록에서 빠져 있었다.
>
> 커밋 시에도 이 단위를 함께 묶을 것. 현재 index에는 공유 모듈 4개만 staged 되어 있고
> PARA 핵심 수정은 unstaged, b4 config/launcher는 untracked 상태다. 지금 index만
> 커밋하면 동작하지 않는 중간 상태가 된다. launcher는 executable bit(`git add --chmod=+x`)
> 까지 포함해야 한다.

`SSR-orig` 쪽 (main 브랜치, 미커밋):
```
A  projects/configs/SSR/SSR_noffp_e2e_2gpu_b4.py
A  train_noffp_2gpu_b4.sh
```

---

## 8. 실행

```bash
# PARA-SSR ver2, 2 GPU × 4 samples
cd /home/yongjae/e2e/SSR
CUDA_VISIBLE_DEVICES=<a>,<b> ./train_para_2gpu_b4.sh

# 8 GPU 그대로 쓰려면 (config는 PARA_SSR_e2e.py)
./train_para_8gpu.sh
```

**주의**: GPU 4는 드라이버가 깨져 있다. `nvidia-smi`에는 보이지만 `CUDA driver initialization failed`가 난다. 관리자 리셋이 필요하다.

---

## 9. 코드 리뷰 대응 (2026-08-17)

외부 리뷰 지적을 하나씩 실증 확인했다. 대부분 타당했고 3건은 부정확했다.

### 9.1 수정한 것

**(P0) deformable attention 초기화가 파괴되고 있었다 — 실제 학습에 영향**

`ParaMapHead.init_weights()` / `ParaDetMotionHead.init_weights()`가 디코더의 모든 2-D 파라미터를 Xavier로 덮으면서, `CustomMSDeformableAttention`이 생성자에서 설정한 값을 지웠다. 두 헤드 모두 `init_weights`를 오버라이드하면서 `super().init_weights()`를 부르지 않아 자식의 재초기화도 일어나지 않았다.

실측:
```
CLOBBERED parameters by module: {'det_motion_head': 12, 'map_head': 6}
  map_decoder.layers.*.attentions.1.sampling_offsets.weight    0.0000 -> 0.1369
  map_decoder.layers.*.attentions.1.attention_weights.weight   0.0000 -> 0.1443
```

`sampling_offsets.weight = 0`은 "step 0에서는 `sampling_offsets.bias`에 담긴 방사형 격자를 **정확히** 샘플링한다"는 뜻이고, `attention_weights.weight = 0`은 "초기 attention은 균일"이라는 뜻이다. 이게 깨지면 학습 초반의 샘플링 위치가 무작위가 된다.

VAD(`VAD_transformer.py:192`)와 `SSRPerceptionTransformer.init_weights`가 쓰는 방식(Xavier 후 deformable 모듈의 `init_weight()` 재호출)을 두 헤드에 그대로 적용했다. 재검증: `deformable-attention init survives init_weights()`.

> **리뷰의 세부 하나는 틀렸다.** 리뷰는 방사형 `sampling_offsets.bias`도 깨진다고 했지만, `if p.dim() > 1` 필터가 1-D bias를 건너뛰므로 **bias는 보존된다**(측정값 4.0000 유지, 전후 동일). 깨진 것은 `.weight` 2종뿐이다.
>
> 참고로 `pts_bbox_head`(BEV 인코더)는 원래부터 무사했다 — `SSRPerceptionTransformer.init_weights`가 이미 재초기화를 하고 있었다.

**(P0) launcher 실행 권한** — `train_para_2gpu_b4.sh`가 0664였다. `chmod +x` 완료 (0755).

**설정 주석이 정정된 진단과 모순** — `map_head` 주석의 "~1% 양성률 / 불균형 / all-background 붕괴"와 dense 대안 주석의 "class-balanced BCE로 고쳐라"를 §6의 실측(8~10% 양성률, 공간 분별력 부재)에 맞게 다시 썼다. dense 재시도 시 고칠 것은 **수용 영역·다중 스케일 feature·디코더 용량**이지 클래스 가중치가 아니다.

`occ_class_labels` 주석의 "아무리 학습해도 도달 불가"도 실험적 근거로 한정했다.

**wandb 메타데이터가 stale** — base config가 `tags=[...'2gpu'], gpus=2, batch_per_gpu=1, aux='det+map_seg+occ'`로 되어 있어 8-GPU 실행이 잘못 기록되고 있었다. 8×1 기준으로 고치고 `map_vec` / `occ_vehicle`로 갱신했다. 파생 config는 이 블록을 통째로 덮어쓴다.

**보고서 정정 4건** — §1(상한 아님), §1(recall 서술), §3.4(drivable_area), §4.6(누수 증거의 강도).

### 9.2 검증을 추가한 것

**bs=4 전체 학습 경로 + 2-rank DDP** — §4.8. forward 비교만 했다는 지적이 맞았다.

**제대로 된 누수 개입 테스트** — §4.7. 이미지 파괴 방식으로 backbone/기하/가시성 경로까지 커버.

### 9.3 타당하나 아직 미해결 (알려진 제약)

| 항목 | 상태 | 비고 |
|---|---|---|
| **map mAP 평가 미연결** | 미해결 | `para_ssr.py:458`은 `bbox_results[0]['map']`에 중첩 저장하는데 `output_to_vecs`는 `detection['map_boxes_3d']` / `map_scores_3d` / `map_labels_3d` / `map_pts_3d`를 기대한다(`nuscenes_vad_dataset.py:1873`). `test_aux_heads=True`로 raw 출력은 볼 수 있으나 mAP는 못 낸다. **"VAD map이 실제로 학습되는가"를 확인하려면 이게 먼저다** — §10 P0 |
| **`gshare/*`의 정확도** | 문서화함 | §2의 한계 절 참조. sweep 시작점으로만 사용 |
| **EMA resume** | 미해결 | `MEGVIIEMAHook(..., resume=None)`이고 config에 설정이 없다. `--resume-from`으로 raw 모델/optimizer를 복구해도 EMA history는 초기화되어 무중단 실행과 달라진다. 중단 재개 시 EMA 가중치를 쓸 계획이면 hook에 `resume=<ckpt>`를 넘겨야 한다 |
| **8×1 vs 2×4 최종 metric** | 진행 중 | §4.9 |

---

## 10. 다음 단계 제안

| 우선순위 | 항목 | 이유 |
|---|---|---|
| **P0** | **map mAP 평가 배선** (`map_boxes_3d` 등으로 출력 형식 변환) | 이게 없으면 "VAD map이 학습되는가"를 확인할 수 없다. 1차 실험에서 14.7시간 학습이 끝난 뒤에야 map이 죽어 있음을 알았던 실수의 재발 방지 |
| **P0** | 학습 중 aux 품질(map mAP / occ IoU) 로깅 | 위와 같은 이유 |
| **P0** | PARA-SSR ver2 12 epoch 학습 | occ 1채널 + VAD map + deformable init 복구의 효과 측정 |
| P1 | 1 epoch 후 `gshare/*`로 `aux_grad_scale` 재보정 | map 파라미터가 7배로 늘어 기존 계산이 무효. **`tools/analyze_gshare.py`를 쓸 것 — 2026-08-18에 재구성 로직 버그를 고쳤다** ([#07](07_review_round2_fixes.md) §3) |
| P1 | `aux_grad_scale` sweep | 현재 값: 2GPU config **0.01**, `PARA_SSR_w_gradscale.py` **0.02**, `PARA_SSR_w_equal.py` 1.0(대조군). 초판이 적었던 0.11 / 0.05는 v1 dense map 시절 계산이라 폐기 |
| P2 | BEV feature KD 실험 설계 | §3.3의 L1 인터페이스 |
| P2 | TokenLearner 다양성 정규화 | 보고서 #01 §7 (토큰 유사도 +63%) |
| P3 | EMA resume 설정 | 장기 실행이 중단될 경우를 대비 (§9.3) |
| P3 | map 제거 ablation | PARA-Drive Table 5 / SSR Appendix E Table 6 재현 |
