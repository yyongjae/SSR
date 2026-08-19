# 참조 문서 #04 — BEV encoder / BEV map shape·구조

PARA-SSR의 모든 aux head가 붙는 지점(`bev_embed`)의 정확한 정의. 새 head를 만들 때 이 문서의 좌표 규약을 그대로 따를 것.

---

## 1. 한눈에 보기

```
6 × camera image                                    [B, 6, 3, 384, 640]
        │
        ▼  ResNet-50 (frozen stage1, BN eval), out_indices=(3,) → C5, stride 32
    C5 feature                                      [B, 6, 2048, 12, 20]
        │
        ▼  FPN (in 2048 → out 256, num_outs=1)
    mlvl_feats[0]                                   [B, 6, 256, 12, 20]
        │
        │        bev_queries  nn.Embedding(10000, 256)      ← 학습 파라미터
        │        bev_pos      LearnedPositionalEncoding     ← 학습 파라미터
        │        can_bus MLP  (18 → 256)  + ego shift
        ▼
    BEVFormerEncoder ×3 layers
      layer = TSA(prev_bev) → norm → SCA(image feats) → norm → FFN(512) → norm
        │
        ▼
  ★ bev_embed                                       [B, 10000, 256]
        │                                            = [B, 100×100, 256]
        │
        ├──────────────┬──────────────┬──────────────┐
        │              │              │              │
     planning       det+motion       map            occ      ← 4개 병렬 소비자
   (SE gate →      (300 query)   (conv/vector)   (conv)
    TokenLearner)
```

`bev_embed`는 **이 지점에서 네 갈래로 복제**되며, aux head들은 planner와 서로를 전혀 보지 않는다 (PARA-Drive Fig.5의 병렬 배선).

---

## 2. 입력 이미지

| 항목 | 값 |
|---|---|
| 원본 | nuScenes 6-cam, 1600 × 900 |
| `RandomScaleImageMultiViewImage` | scale 0.4 → 640 × 360 |
| `PadMultiViewImage` | `size_divisor=32` → **640 × 384** |
| normalize | ImageNet mean/std, to_rgb |

> 논문은 640×360이라고 적었지만 실제 네트워크에 들어가는 것은 패딩 후 **640×384**다. (`lidar2img`는 스케일에 맞춰 갱신되므로 기하학적으로는 일관됨)

**이 해상도가 §5의 작은 객체 문제의 근원이다.** 0.4× 축소 후 20 m 거리의 보행자는 세로 약 12 px, C5(stride 32)에서는 **0.4 px** — 즉 feature map 상에서 하나의 셀도 채우지 못한다.

---

## 3. BEV 격자 정의 ★

```python
point_cloud_range = [-15.0, -30.0, -2.0,  15.0, 30.0, 2.0]
#                   [x_min,  y_min,  z_min, x_max, y_max, z_max]
bev_h_ = bev_w_ = 100
```

| 축 | 범위 | 셀 수 | **셀 크기** | 텐서 차원 |
|---|---|---|---|---|
| **x (lateral, 좌우)** | −15 ~ +15 m (30 m) | 100 | **0.30 m** | **W = column** |
| **y (longitudinal, 전후)** | −30 ~ +30 m (60 m) | 100 | **0.60 m** | **H = row** |
| z | −2 ~ +2 m (4 m) | — | pillar 4점 샘플 | — |

### 3.1 반드시 기억할 두 가지

**(1) 격자가 비등방(anisotropic)이다.** 세로 셀이 가로 셀의 **2배 크다**. 정사각형 100×100 텐서지만 실제로 표현하는 영역은 30 m × 60 m의 세로로 긴 직사각형이다. 원형 물체는 BEV 상에서 세로로 눌린 타원으로 보인다.

**(2) row = y, column = x.** BEVFormer 관례이며 `encoder.py:get_reference_points`에서 확정된다:
```python
xs = linspace(0.5, W-0.5, W).view(1, 1, W).expand(D, H, W) / W   # x는 W축
ys = linspace(0.5, H-0.5, H).view(1, H, 1).expand(D, H, W) / H   # y는 H축
```

### 3.2 인덱스 ↔ 실좌표 변환 (복붙용)

```python
dx = (pc_range[3] - pc_range[0]) / bev_w   # 0.3
dy = (pc_range[4] - pc_range[1]) / bev_h   # 0.6

# 셀 중심 → LiDAR 좌표
x = pc_range[0] + (col + 0.5) * dx
y = pc_range[1] + (row + 0.5) * dy

# LiDAR 좌표 → 셀
col = (x - pc_range[0]) / dx
row = (y - pc_range[1]) / dy

# flatten 인덱스 (bev_embed의 dim 1)
idx = row * bev_w + col
```

**자차(ego)는 (row=50, col=50), 즉 텐서 정중앙**이고 전방은 row 증가 방향이다.

### 3.3 nuScenes box → BEV 폴리곤 (occ GT와 collision metric이 공유하는 규약)

```python
width, length = boxes[:, 3], boxes[:, 4]      # nuScenes는 (w, l, h) 순
yaw = -1.0 * (boxes[:, 6] + np.pi / 2)        # SECOND format → BEV yaw
```
`occ_label.py`와 `planner/metric_stp3.py:PlanningMetric`이 **동일한 변환**을 쓴다. 새 head를 만들 때 반드시 이 두 줄을 그대로 복사할 것 — 부호나 순서를 바꾸면 GT가 조용히 90° 돌아간다.

---

## 4. BEV encoder 내부

### 4.1 구성

```python
encoder = BEVFormerEncoder(num_layers=3, num_points_in_pillar=4)
  layer = BEVFormerLayer(
      operation_order = ('self_attn','norm','cross_attn','norm','ffn','norm'),
      self_attn  = TemporalSelfAttention(embed_dims=256, num_levels=1),
      cross_attn = SpatialCrossAttention(
                      deformable_attention=MSDeformableAttention3D(
                          embed_dims=256, num_points=8, num_levels=1)),
      feedforward_channels = 512,   # FFN 256→512→256
      ffn_dropout = 0.1)
```

VAD-Tiny 설정 그대로다 (BEVFormer-Base는 6층).

### 4.2 TSA — Temporal Self-Attention

- BEV query가 **이전 프레임의 BEV**를 deformable attention으로 참조.
- `use_shift=True`: `can_bus[0:2]`의 ego 변위를 격자 단위로 환산해 reference point를 미리 이동 → 자차 운동을 보정한 뒤 참조.
- `rotate_prev_bev=False` (SSR 설정): 회전 보정은 하지 않음.
- `prev_bev=None`(시퀀스 첫 프레임)이면 `ref_2d`를 두 번 쌓아 self-attention으로 퇴화.

> `encoder.py:187`에 원저자가 명시한 알려진 버그가 있다:
> ```python
> shift_ref_2d = ref_2d  # .clone()   ← in-place로 ref_2d까지 오염됨
> shift_ref_2d += shift[:, None, None, :]
> ```
> 논문 수치 재현을 위해 **의도적으로 유지**한다. 건드리지 말 것.

### 4.3 SCA — Spatial Cross-Attention

- 각 BEV 셀 위에 **높이 방향 4점 pillar**를 세운다. `Z = z_max − z_min = 4 m`, 정규화 z = 0.125/0.375/0.625/0.875 → 실좌표 **−1.5, −0.5, +0.5, +1.5 m**.
- 각 3D 점을 `lidar2img`로 6개 카메라에 투영. 이미지 안에 들어가고 depth > 0인 카메라만 `bev_mask`로 선택.
- 선택된 카메라의 feature map에서 deformable attention (`num_points=8`).
- 어떤 카메라에도 보이지 않는 셀은 업데이트되지 않는다.

### 4.4 사전 조건 주입

BEV query는 encoder에 들어가기 전에 두 가지를 흡수한다:
```python
bev_queries = bev_embedding.weight + can_bus_mlp(can_bus)   # ego 상태 (속도/가속/각속도 등 18-d)
# + bev_pos (LearnedPositionalEncoding, row/col 각각 100개 embedding, num_feats=128 → 총 256)
```

---

## 5. `bev_embed`의 네 소비자

| 소비자 | 입력 처리 | 출력 |
|---|---|---|
| **planning** (`ParaSSRHead`) | navigation SE gating → `cat(bev, pos_embd)` 512-d → TokenLearner 16 token | `ego_fut_preds [B,3,6,2]` |
| **det+motion** (`ParaDetMotionHead`) | 300 query × DeformAttn decoder ×6, + motion self-attn ×1 | box + 6-mode trajectory |
| **map** (`ParaMapSegHead`) | `view(B,100,100,256).permute(0,3,1,2)` → conv trunk | `[B, 3, 100, 100]` |
| **occ** (`ParaOccHead`) | 동일 reshape → conv trunk → 1×1 conv | `[B, 7, 2, 100, 100]` |

### 5.1 planning 경로 상세 (SSR 고유 병목)

```
bev_embed [B, 10000, 256]
   │
   ▼  ① navigation SE gating
   navi_embed = navi_embedding.weight[cmd_idx]        # cmd ∈ {left, right, straight}
   bev = bev_embed * sigmoid(MLP(navi_embed))         # 채널별 gating
   │
   ▼  ② positional concat
   bev_query = cat(bev, bev_pos)                      [B, 10000, 512]
   │
   ▼  ③ TokenLearner V11  ★ 병목
   selected = softmax_HW( MLP_64( GroupNorm(bev_query) ) )   [B, 16, 10000]
   tokens   = einsum('bsi,bic->bsc', selected, bev_query)    [B, 16, 512]
   │        └─ 10000개 셀 → 16개 토큰. 각 토큰은 BEV 전체에 대한
   │           softmax 가중합 = "학습된 soft ROI pooling"
   ▼  ④ split 512 → (query 256, pos 256)
   ▼  ⑤ latent_decoder: self-attn ×3 (16 token 끼리)
   ▼  ⑥ way_decoder: cross-attn ×1
        query  = way_point.weight  (18 = 3 cmd × 6 step, 512-d → query/pos로 split)
        key/val= 16 scene tokens
   ▼  ⑦ ego_fut_decoder: MLP(256→256→256→2)
   ego_fut_preds [B, 3, 6, 2]   → cmd로 하나 선택
```

**여기가 PARA-SSR 실패의 구조적 급소다.** planner가 보는 것은 `bev_embed` 전체가 아니라 **16개 토큰(=16×256)** 뿐이다. dense BEV를 직접 cross-attend하는 PARA-Drive/VAD의 planner와 달리, SSR planner는 이 압축을 거치므로 BEV 표현의 변화에 훨씬 민감하다. 실제로 PARA-SSR에서 토큰 간 유사도가 +63% 상승했다 (보고서 #01 §7).

### 5.2 aux head가 `bev_embed`를 받는 방식

`para_ssr.py`에서 gradient scaling을 거쳐 분배한다:

```python
aux_bev = _ScaleGrad.apply(bev_embed, self.aux_grad_scale)   # forward는 identity
```
`_ScaleGrad`는 forward에서 아무것도 하지 않고 backward에서만 gradient에 `scale`을 곱한다. aux head **내부** 파라미터의 학습률에는 영향이 없고, 공유 BEV로 흘러드는 양만 조절한다.

---

## 6. 텐서 shape 치트시트

| 이름 | shape | 비고 |
|---|---|---|
| `img` | `[B, 6, 3, 384, 640]` | 6 cam |
| `mlvl_feats[0]` | `[B, 6, 256, 12, 20]` | FPN 단일 레벨 |
| `bev_queries` | `[10000, 256]` | `nn.Embedding` |
| `bev_pos` | `[B, 256, 100, 100]` | flatten 후 `[B,10000,256]` |
| **`bev_embed`** | **`[B, 10000, 256]`** | **모든 head의 입력** |
| BEV 2D view | `[B, 256, 100, 100]` | `view(B,100,100,256).permute(0,3,1,2)` |
| `token_attn` (`selected`) | `[B, 16, 10000]` | softmax over HW |
| `scene_query` | `[16, B, 256]` | seq-first |
| `ego_fut_preds` | `[B, 3, 6, 2]` | (cmd, step, xy) |
| `gt_occ_seg` | `[B, 7, 2, 100, 100]` | (T, class) |
| map seg logits | `[B, 3, 100, 100]` | divider/ped/boundary |

---

## 7. 배치 크기 제약 (중요)

**`samples_per_gpu`는 반드시 1이어야 한다.** 이것은 원본 SSR의 제약이며 개조로 생긴 것이 아니다.

- 원본 SSR: bs=2에서 **hard crash** — `action_mln`이 `B*12` 차원의 `wp_vector`를 받는데 `Linear(12, 256)`이다.
- PARA-SSR: crash는 안 하지만 `para_ssr_head.py:194`의
  ```python
  cmd = cmd[0, 0, 0]
  ```
  때문에 **샘플 0의 navigation command가 배치 전체에 적용**된다. 검증됨 — 샘플 1의 cmd를 뒤집어도 출력 변화가 `0.000e+00`.

배치를 키우려면 gradient accumulation을 쓸 것.

---

## 8. 새 aux head를 붙일 때 체크리스트

1. 입력은 `[B, 10000, 256]`. 2D로 쓰려면 `view(B, 100, 100, 256).permute(0, 3, 1, 2)`.
2. GT 래스터화 시 **row=y, col=x**, `dx=0.3`, `dy=0.6`. cv2는 `(x, y) = (col, row)` 순서를 받는다.
3. box → 폴리곤은 §3.3의 두 줄을 그대로 복사.
4. 비등방 격자를 고려할 것 — 세로 방향 1 셀 = 0.6 m다. 얇은 구조(차선)를 그릴 때 thickness를 축별로 다르게 볼 필요가 있다.
5. `para_ssr.py:_build_aux`에 등록. head의 `__init__` 시그니처에 `train_cfg`가 없으면 주입하지 않는다 (자동 판별됨).
6. `aux_grad_scale`이 적용되는 `aux_bev`를 받도록 배선.
7. **학습 중 품질 지표(IoU/mAP)를 로깅할 것.** 보고서 #01의 교훈.
