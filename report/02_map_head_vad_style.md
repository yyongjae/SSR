# 보고서 #02 — Map head를 VAD 스타일(벡터)로 전환

작성일 2026-08-16 · 변경 파일: `projects/configs/SSR/PARA_SSR_e2e.py`

---

## 0. 먼저, "지금 map 코드 구조는 레퍼런스 삼아서 한게 아닌건가?"

**절반은 레퍼런스 기반이었고, 절반은 아니었다.** 정확히 나누면:

| 층위 | 기존 `ParaMapSegHead` | 레퍼런스 여부 |
|---|---|---|
| **표현(representation)** | dense BEV semantic segmentation | ✅ **PARA-Drive**를 따름. PARA-Drive의 mapping 모듈은 실제로 dense 4채널 semantic BEV map이다 |
| **손실 함수** | BCE(top-k) + Dice | △ **UniAD/FIERY**의 occupancy loss를 그대로 재사용. map용으로 설계된 게 아님 |
| **디코더 아키텍처** | Conv 3층 (256→128→64→3), 0.41M | ❌ **레퍼런스 없음. 내가 최소 구성으로 직접 작성** |
| **GT 래스터화** | 폴리라인 thickness=2 | ❌ **레퍼런스 없음. 직접 작성** |

즉 "PARA-Drive가 dense map을 쓰니까 dense로 간다"는 **선택은 논문 근거가 있었지만**, 그 dense를 **어떻게 실현할지**는 참조 구현 없이 만든 것이다. PARA-Drive는 mapping에 **Panoptic SegFormer**(query 기반 마스크 디코더, 수천만 파라미터급)를 쓰는데, 나는 그 자리를 conv 3층으로 대체했다. 이것이 격차의 핵심이다.

당시 dense를 고른 이유는 **KD 실험에 유리하다**는 판단이었다 — per-pixel logit은 teacher/student가 픽셀 단위로 정렬되지만, Hungarian 매칭 기반 query 출력은 teacher의 query k와 student의 query k가 같은 대상을 가리킨다는 보장이 없다. 이 논리 자체는 여전히 유효하지만, **애초에 학습이 안 되는 teacher는 KD 대상이 될 수 없으므로** 우선순위가 뒤집혔다.

한편 `ParaMapHead`(VAD 벡터 버전)는 **처음부터 VAD 코드를 그대로 포팅해 만들어 두었고** 설정 블록에 주석 처리된 채로 있었다. 이번 변경은 새로 만드는 게 아니라 **스위치를 미리 만들어둔 쪽으로 넘기는 것**이다.

---

## 1. 왜 바꾸는가 — 측정된 실패

보고서 #01 §6의 수치:

| 항목 | 값 |
|---|---|
| divider IoU @0.5 | 0.003 |
| ped_crossing IoU @0.5 | 0.000 |
| boundary IoU @0.5 | 0.005 |
| divider 평균 예측 확률 | 0.056 (최대 0.60) |
| p>0.5 픽셀 / 프레임 | **0.2개** (GT는 855개) |
| corr(prob, GT) | +0.232 |
| 12 epoch BCE 감소율 | **−18%** (det_cls는 −67%) |
| 최종 BEV gradient 점유율 | **17.7%** |

> **정정 (2026-08-17).** 이 절의 초판은 "양성 픽셀 ~1%의 극단적 불균형 → 전부 배경으로
> 붕괴"라고 썼는데 **틀렸다.** GT 양성률을 실측하니 divider 8.18%, ped_crossing 1.82%,
> boundary 9.95%다. top-k 25%(k=2500)를 적용하면 유효 비율이 1:2~1:3까지 완화되므로
> 불균형은 주범이 아니다. 아래는 재측정에 근거한 설명이다.

상관계수가 +0.23이므로 **좌표계 버그는 아니다.** 실제 문제는 **공간 분별력의 부재**다:

| 채널 | GT 양성률 | mean p (GT 양성) | mean p (GT 음성) | 분별비 |
|---|---:|---:|---:|---:|
| divider | 8.18% | 0.1154 | 0.0702 | **1.6×** |
| ped_crossing | 1.82% | 0.0548 | 0.0128 | 4.3× |
| boundary | 9.95% | 0.3024 | 0.2862 | **1.06×** |

**boundary는 전 픽셀에 ≈0.29를 출력하는 상수 함수로 수렴했다.** thr ≤ 0.2에서 10,000픽셀을 전부 양성으로 예측하고, 그때 IoU 0.0995는 GT 양성률과 정확히 같다 — "전부 양성"과 같은 답이다. 비교로 occ vehicle의 분별비는 7.8×다.

원인 4가지:

1. **BEV feature에 차선 정보가 거의 없다.** 입력은 0.4× 축소된 640×384이고 FPN은 C5(stride 32) 단일 레벨뿐이다. 노면 표시는 이 해상도에서 사실상 소실된다. **벡터 head가 유리한 근본 이유가 여기 있다** — 약한 근거만 있어도 "차선은 길고 매끄러운 곡선"이라는 형상 사전지식이 쿼리 임베딩에 들어 있어 그럴듯한 기하를 출력할 수 있지만, per-pixel conv에는 그런 사전지식이 없다.
2. **수용 영역 부족.** conv 3×3 3층 ≈ 7×7 셀 = 2.1 m × 4.2 m. 차선은 수십 미터짜리 구조다.
3. **디코더 용량 부족.** 0.41M. PARA-Drive는 Panoptic SegFormer를 쓴다.
4. **top-k BCE의 균형점이 확률 수준을 고정한다.** `p* = n_pos/k`가 실측 확률을 6채널 중 5개에서 예측한다 (boundary 995/2500 = 0.398 vs 실측 0.302). 출력 확률은 학습 품질이 아니라 GT 밀도가 결정한다. 유도는 보고서 #01 §6.4.

그러면서도 **BEV gradient의 17.7%를 가져갔다.** 상수 함수를 학습하는 head의 gradient는 공유 표현 입장에서 노이즈다.

---

## 2. VAD 스타일 map head는 어떻게 작동하는가

한 문장: **DETR을 폴리라인에 적용한 것.** 클래스당 마스크를 그리는 대신, "지도에 있는 선 100개"를 각각 20개 점의 좌표열로 직접 회귀하고, 예측-GT를 Hungarian 매칭으로 짝지어 손실을 건다.

### 2.1 쿼리 구성 — instance × point 분해

```python
map_instance_embedding = nn.Embedding(100, 512)   # 폴리라인 1개당 1개
map_pts_embedding      = nn.Embedding(20,  512)   # 점 위치 1개당 1개

map_query_embed = pts_embeds[None] + instance_embeds[:, None]   # [100, 20, 512]
map_query_embed = map_query_embed.flatten(0, 1)                 # [2000, 512]
map_query_pos, map_query = split(map_query_embed, 256)          # 각 [2000, 256]
```

**핵심 아이디어**: 2000개 쿼리를 독립적으로 두지 않고 `instance[i] + point[j]`로 **분해(factorize)**한다.
- 같은 폴리라인의 20개 점은 `instance[i]`를 공유 → "우리는 같은 선이다"
- 서로 다른 폴리라인의 j번째 점은 `point[j]`를 공유 → "우리는 선의 j번째 지점이다"

파라미터는 `100×512 + 20×512 = 61K`뿐이다 (독립 쿼리라면 `2000×512 = 1M`). 그리고 각 폴리라인이 **점 순서에 대한 일관된 사전지식**을 갖게 된다.

### 2.2 디코더 — BEV에서 좌표를 반복 정제

```python
reference_points = Linear(256, 2)(query_pos).sigmoid()   # [2000, 2] 초기 위치 ∈ (0,1)²

for layer in decoder(3 layers):
    query = self_attn(query)                       # 2000개 점끼리 상호작용
    query = deform_attn(query, bev_embed,          # ★ BEV를 참조점 주변에서 샘플링
                        reference_points)
    query = ffn(query)
    reference_points = sigmoid(inverse_sigmoid(reference_points)
                               + reg_branch(query))  # 좌표 갱신 (iterative refinement)
```

- **deformable attention**이므로 BEV 10,000셀 전체가 아니라 **각 점의 현재 추정 위치 주변만** 샘플링한다. dense 방식이 모든 셀에 대해 배경/전경을 판정해야 하는 것과 대조적이다.
- `reference_points`가 층마다 갱신되므로 **점이 BEV 위를 걸어다니며 차선을 찾아간다.** 이것이 dense와의 가장 큰 차이다 — dense는 "이 픽셀이 차선인가?"를 묻고, 벡터는 "차선이 어디 있지?"를 묻는다.
- 3층 = VAD-Tiny 설정 (SSR이 기반으로 삼은 것). VAD-Base는 6층.

### 2.3 출력

층 `l`마다:
```python
cls[l]  = cls_branch[l]( query.view(100, 20, 256).mean(dim=1) )   # [100, 3] 폴리라인 단위 클래스
pts[l]  = reference_points.view(100, 20, 2)                       # [100, 20, 2] 정규화 좌표
bbox[l] = minmax(pts) → (cx, cy, w, h)                            # 매칭 보조용
```

클래스 점수는 **20개 점 임베딩의 평균**으로 낸다 — 폴리라인은 하나의 인스턴스이므로 라벨도 하나다.

### 2.4 Hungarian 매칭 ★ (불균형이 사라지는 지점)

`MapHungarianAssigner3D`가 예측 100개와 GT N개 사이의 비용 행렬을 만들고 최적 일대일 매칭을 구한다:

```
cost = 2.0 · FocalLossCost(cls)          # 클래스 일치
     + 1.0 · OrderedPtsL1Cost(pts)       # 점 좌표 L1 거리
     + 0.0 · BBoxL1Cost / IoUCost        # 사용 안 함 (weight 0)
```

**`OrderedPtsL1Cost`의 "Ordered"가 중요하다.** 폴리라인은 방향이 정해져 있지 않다 — A→B로 그리든 B→A로 그리든 같은 차선이다. VAD는 GT의 **모든 순환/역순 배열(shift permutation)**을 미리 만들어두고 (`shift_fixed_num_sampled_points_v2`), 그중 예측과 가장 가까운 배열을 매칭 시점에 고른다. 반환값 `order_index`가 그 선택이며, 손실 계산에서 그 배열을 타깃으로 쓴다:

```python
pts_targets[pos_inds] = gt_shifts_pts[pos_assigned_gt_inds, assigned_shift, :, :]
```

이렇게 하면 **모델이 임의로 정한 점 순서 때문에 벌을 받는 일이 없다.** dense 방식에는 이 문제 자체가 없지만, 대신 픽셀 정렬 오차를 그대로 맞는다.

매칭 결과:
- **매칭된 쿼리(보통 10~30개)만** 좌표 손실을 받는다.
- 나머지는 "배경" 클래스로 분류 손실만 받는다.

→ **10,000픽셀 중 100개 양성이라는 25:1 불균형이, 100쿼리 중 20개 양성이라는 4:1로 바뀐다.** 게다가 배경 쿼리는 Focal Loss(α=0.25, γ=2)로 이미 다운웨이팅된다. 이것이 dense가 실패한 자리에서 벡터가 학습되는 근본 이유다.

### 2.5 손실

| 손실 | 타입 | weight | 역할 |
|---|---|---|---|
| `loss_map_cls` | FocalLoss(γ=2, α=0.25) | 2.0 | 폴리라인 클래스 |
| `loss_map_pts` | `PtsL1Loss` | 1.0 | **점 좌표 L1 (주력)** |
| `loss_map_dir` | `PtsDirCosLoss` | 0.005 | 인접 점 방향 벡터의 코사인 |
| `loss_map_bbox` | L1 | **0.0** | 비활성 (매칭 보조용 박스) |
| `loss_map_iou` | GIoU | **0.0** | 비활성 |

`PtsDirCosLoss`는 연속한 두 점의 차분 벡터가 GT와 같은 방향을 향하도록 강제한다. 가중치가 0.005로 작지만 **폴리라인이 지그재그로 꼬이는 것을 막는 역할**이라 없으면 눈에 띄게 나빠진다.

손실은 **3개 디코더 층 전부**에 걸린다 (deep supervision). 로그에 `map.loss_map_*`(마지막 층)와 `map.d0.*`, `map.d1.*`로 찍힌다.

### 2.6 dense vs vector 요약

| | dense `ParaMapSegHead` | vector `ParaMapHead` (VAD) |
|---|---|---|
| 출력 | `[B, 3, 100, 100]` 픽셀 로짓 | `[B, 100, 20, 2]` 좌표 + `[B, 100, 3]` 클래스 |
| 감독 | 전 픽셀 BCE + Dice | 매칭된 쿼리만 L1 + Focal |
| 불균형 | **25:1 (치명적)** | 4:1 + Focal |
| 순서 문제 | 없음 | shift permutation으로 해결 |
| 파라미터 | 0.41 M | **2.91 M** |
| BEV 접근 | 전체 셀 conv | 참조점 주변 deformable 샘플링 |
| 해상도 한계 | 0.3×0.6 m 셀에 갇힘 | **연속 좌표 (셀보다 정밀)** |
| 평가 | IoU | chamfer-distance mAP |
| KD 인터페이스 | 픽셀 정렬 (쉬움) | 쿼리 대응 모호 (어려움) |
| 실측 결과 | **IoU 0.003 (실패)** | VAD에서 검증됨 |

---

## 3. 변경 내용

`projects/configs/SSR/PARA_SSR_e2e.py`의 `map_head` 블록만 교체했다. **모델 코드는 한 줄도 바뀌지 않았다** — `ParaMapHead`는 이미 구현되어 있었고 `forward_train(bev_embed, img_metas, map_gt_bboxes_3d, map_gt_labels_3d)` 시그니처가 동일하다.

```python
map_head=dict(
    type='ParaMapHead',                    # was: 'ParaMapSegHead'
    map_num_vec=100,
    map_num_pts_per_vec=20,
    map_num_pts_per_gt_vec=20,
    map_num_classes=3,
    map_gt_shift_pts_pattern='v2',
    transformer_decoder=dict(
        type='MapDetectionTransformerDecoder',
        num_layers=3,                      # VAD-Tiny (SSR의 기반). VAD-Base는 6
        ...),
    bbox_coder=dict(type='MapNMSFreeCoder', max_num=50, ...),
    loss_map_cls=FocalLoss(w=2.0),
    loss_map_pts=PtsL1Loss(w=1.0),
    loss_map_dir=PtsDirCosLoss(w=0.005),
    loss_map_bbox=L1Loss(w=0.0),
    loss_map_iou=GIoULoss(w=0.0)),
```

dense 버전은 주석 블록으로 보존했다 (표현 ablation + KD 인터페이스용). 재활성화 시 고칠 점 2가지를 주석에 남겼다: `line_thickness≥3`, class-balanced loss.

### 3.1 부수 효과

| | 이전 | 이후 |
|---|---|---|
| map head 파라미터 | 0.41 M | 2.91 M (+2.5 M) |
| 모델 총 파라미터 | 39.2 M | ~41.7 M |
| loss 키 | `map.loss_map_mask`, `map.loss_map_dice` | `map.loss_map_{cls,pts,dir,bbox,iou}` + `map.d{0,1}.*` |
| 필요한 GT | `map_gt_bboxes_3d` (동일) | `map_gt_bboxes_3d` (동일) |
| 필요한 assigner | 없음 | `train_cfg.pts.map_assigner` — **이미 config에 있음** |
| 파이프라인 변경 | — | **없음** |

> `train_cfg`는 `_build_aux`가 head `__init__` 시그니처를 검사해 자동 주입한다. `ParaMapHead`는 `train_cfg`를 받으므로 `map_assigner`가 자동으로 연결된다.

### 3.2 검증 (실행 완료)

`smoke_test.py`, GPU 3, 실제 dataloader 1 iteration:

```
map.loss_map_cls    +2.354      map.d0.loss_map_cls   +2.178
map.loss_map_pts    +8.284      map.d0.loss_map_pts   +8.210
map.loss_map_dir    +0.102      map.d0.loss_map_dir   +0.102
map.loss_map_bbox   +0.000      (weight 0, 의도된 값)
map.loss_map_iou    +0.000      (weight 0, 의도된 값)

per-head gradient coverage:
  pts_bbox_head     155/159     (미사용 4개는 원본 SSR의 잔재)
  det_motion_head   262/262
  map_head          118/118  ✅  전 파라미터가 gradient 수신
  occ_head           11/11

parallelism check (동일 가중치, head 토글):
  all heads on    loss_plan_reg = 0.0439840890
  no occ          loss_plan_reg = 0.0439840890
  no det/map      loss_plan_reg = 0.0439840890
  planner only    loss_plan_reg = 0.0439840890   ✅ 비트 단위 동일
```

**병렬성 보존 확인**: aux head를 켜든 끄든 planning 출력이 비트 단위로 같다. map head가 planner에 어떤 경로로도 정보를 주지 않는다는 증명이며, PARA-Drive Fig.4의 edge (1)/(5) 제거가 유지된다.

---

## 4. 예상 효과와 남은 리스크

### 기대
- map 손실이 실제로 내려갈 것 (VAD-Tiny 기준 12 epoch에 map mAP 0.3~0.4 수준).
- 학습되는 신호가 되면 BEV에 주입되는 gradient가 **노이즈에서 정보로** 바뀐다.

### 리스크 (정직하게)
1. **gradient 점유율은 오히려 늘 수 있다.** 파라미터가 7배, 손실 항이 5개(×3층). 학습이 되든 안 되든 `gshare/map`은 커질 수 있으므로 **§5의 `aux_grad_scale` 조정과 반드시 함께** 가야 한다.
2. **SSR 저자 결과와 충돌.** SSR Appendix E Table 6은 저자들이 이미 map aux를 붙였을 때 planning L2가 0.75→0.79~0.86으로 나빠졌다고 보고한다 (CCR@2s만 개선). 즉 **map을 잘 학습시키는 것과 planning이 좋아지는 것은 별개 문제**다. 이번 변경은 "map이 죽어서 노이즈였다"는 원인 하나를 제거할 뿐, planning 개선을 보장하지 않는다.
3. **메모리/속도.** 2000 쿼리 × 3층 deformable attention. 이전 dense 대비 activation이 늘어난다 (기존 측정으로 ~3 GB 추가 예상). 4.1 GB → 7 GB 내외 예상이며 48 GB GPU에서는 문제없다.
4. **KD 인터페이스 상실.** 벡터 head는 Hungarian 매칭 때문에 teacher/student query 대응이 모호하다. KD를 map에 적용하려면 (a) 매칭 결과를 teacher에서 student로 전달하거나, (b) teacher 폴리라인을 래스터화해 dense student를 가르치거나, (c) map은 KD 대상에서 제외해야 한다. **→ 보고서 #03 §4에서 occ teacher와 함께 다룸.**

### 권고 실험 순서
| ID | 구성 | 목적 |
|---|---|---|
| E3a | VAD map + `aux_grad_scale=1.0` (`PARA_SSR_w_equal.py`) | 학습성만 먼저 확인 (map mAP > 0.2면 성공) |
| E3b | VAD map + `aux_grad_scale=0.01` (`PARA_SSR_e2e_2gpu_b4.py`) | planning share 상향. 0.02 arm은 `PARA_SSR_w_gradscale.py`. 초판의 0.1은 v1 dense map 시절 계산이라 폐기 |
| E3c | map 제거 | PARA-Drive Table 5 재현 — map이 정말 필요한지 |

`grad_norm_log_interval=200`이 켜져 있으므로 `gshare/map`을 학습 초반 1 epoch만 봐도 (1)의 리스크를 즉시 판단할 수 있다.

---

## 5. 실행

```bash
cd /home/yongjae/e2e/SSR
bash train_para_8gpu.sh          # config는 이미 VAD map head로 전환됨
```

되돌리려면 `PARA_SSR_e2e.py`의 `map_head` 블록에서 주석을 반대로 옮기면 된다.
