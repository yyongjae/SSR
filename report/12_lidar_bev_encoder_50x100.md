# PARA-SSR LiDAR 입력 BEV encoder와 50×100 BEV grid

두 가지를 함께 바꿨다.

1. **BEV grid `100×100` → `50×100`.** 전방 32 m를 50행, 측방 64 m를 100열로 나눠 셀이
   0.64 m 정사각형이 된다. BEVFusion teacher가 같은 grid로 재학습·재캐시되어 있으므로
   (`bevfusion/configs/navsim/default.yaml`: voxel 0.08 m, sparse_shape `[400, 800, 41]`,
   out_size_factor 8; `teacher_cache/bevfusion/cache_{train,val}_50x100`) teacher BEV와
   student `bev_embed`가 셀 단위로 대응한다.
2. **LiDAR 입력.** SafeDrive의 wiring을 따라 LiDAR BEV가 BEVFormer의 BEV query 초기값이
   되고, encoder 각 층의 `lidar_cross_attn`이 이를 읽는다. `use_lidar: false`면 기존
   camera-only 구조가 그대로 남는다.

## 1. 구조

```text
frame t ∈ {2, 3}
  merged point cloud (x_fwd, y_left, z, intensity, ring, id)
    → 전방 ROI(x_right ∈ [-32,32), y_forward ∈ [0,32)) · lidar_z_range [-3,5) clip
    → SSR 축 회전 (x_right = -y_left, y_forward = x_fwd), intensity/255, ring/40
    → lidar_max_points=65536 행으로 zero-pad          ┐ feature builder
                                                    ┘
    → voxel mean-pool (0.08 m, 400×800×40) → SpMiddleResNetFHD (spconv, ×8↓, z→채널)
    → LiDAR BEV [B, 256, 50, 100]                                        LidarSparseEncoder
    → (a) BEV query 초기값  (학습형 bev_embedding 대체)
    → (b) 각 encoder 층: temporal self-attn → norm → lidar_cross_attn → camera cross_attn
          → gate·q_lidar + (1-gate)·q_camera + identity → norm → ffn → norm
  현재 frame의 bev_embed → planner / det·motion / map (기존과 동일)
```

과거 frame(2)에도 같은 경로가 적용된다. 과거 BEV는 temporal self-attention의 정렬 대상이므로
현재 BEV와 같은 종류의 feature여야 하고, 따라서 LiDAR도 `frame_indices` 전부에서 로드한다
(`get_sensor_config().lidar_pc == [2, 3]`).

## 2. SafeDrive와의 대응

| 항목 | SafeDrive | 이 구현 |
|---|---|---|
| LiDAR encoder | voxel 0.125 m mean-pool → `SpMiddleResNetFHD` (spconv, stride 8, z→채널 128×2=256) | **동일** (`LidarSparseEncoder`): voxel 0.08 m → 같은 `SpMiddleResNetFHD` 포팅, 41→2 depth, 256 채널 = `embed_dims`라 projection 없음 |
| BEV query 초기값 | `lidar_feats.flatten(2)` | 동일 (`get_bev_features(lidar_bev=...)`) |
| 층별 LiDAR attention | `MSDeformableAttention3D`, 2D ref, 같은 LiDAR BEV를 value로 | `CustomMSDeformableAttention`(2D deformable), 동일 value |
| camera cross-attn | `no_residual=True` | `add_residual=False` |
| gate | `nn.Embedding(1, C)` N(0,1) 초기화 | `nn.Parameter(zeros(C))` → 초기 0.5/0.5 |
| residual | mmcv post-norm 순서상 self-attn **norm 이전** 출력 | self-attn **norm 이후** 출력 (camera-only 경로와 동일한 residual) |
| BEV grid | 32×64, 1.0 m | 50×100, 0.64 m |
| z 범위 | `[-5, 3]` | `[-3, 5]` (teacher와 동일; 실측 ROI 점의 99%가 5.1 m 이하, -3 m 이하 없음) |
| 점 전달 | 경로 → 모델에서 로드 (가변 길이 list) | ROI clip 후 고정 길이 padding (기본 collate·cache와 호환) |

### SafeDrive_Backbone 전체와의 대응 — 무엇을 가져오고 무엇을 남겼나

`SafeDrive_Backbone`은 LiDAR encoder만이 아니라 자체 BEV pipeline 전체다. 이 포팅은
**LiDAR encoder와 그 주입 지점(query 초기화·`lidar_cross_attn`·gate)** 만 가져오고, 나머지는
PARA-SSR(SSR BEVFormer → TokenLearner) 것을 유지했다. camera-only PARA-SSR arm과 같은
encoder 위에서 LiDAR 유무만 비교하기 위해서다.

| 구성요소 | SafeDrive | PARA-SSR (+LiDAR) | 상태 |
|---|---|---|---|
| 이미지 backbone / neck | resnet34 + 4-level SECONDFPN(64×4→256), 입력 512×256 | resnet50.tv_in1k C5 + SingleLevelFPN, 768×416 | SSR 유지 |
| camera reference pillar z | `[-3, 5]` m (4점: -2, 0, 2, 4 m) | `[-2, 2]` m (4점: -1.5, -0.5, 0.5, 1.5 m) | SSR/VAD 유지 |
| BEV query 초기값 | LiDAR BEV | LiDAR BEV | **동일** |
| 층별 `lidar_cross_attn` + gate | 있음 | 있음 (gate zero-init) | **동일** |
| ego-motion query conditioning | `use_can_bus=False` | `ego_motion_mlp` 출력을 query에 더함 | SSR 유지 |
| temporal | frame별 BEV 독립 계산(`prev_bev=None`) → ego pose로 `grid_sample` warp → 3 frame 채널 concat → 3×3 conv | BEVFormer temporal self-attention (`prev_bev`), 2 frame | SSR 유지 |
| BEV 후처리 | `CustomResNet(256→[256,512,1024])` + SECONDFPN + 3×3 conv 후 head | `bev_embed`를 곧바로 TokenLearner/det/map head에 | SSR 유지 |
| positional encoding | `LearnedPositionalEncoding3D` (row/col 학습 임베딩; 2D와 동일 구조) | `LearnedPositionalEncoding` | 실질 동일 |
| LiDAR voxel / grid | 0.125 m, 32×64 (1.0 m 셀), z `[-5, 3]` | 0.08 m, 50×100 (0.64 m 셀), z `[-3, 5]` | grid에 종속 |
| LiDAR 입력 frame 수 | 3 | 2 (`frame_indices`) | SSR 유지 |

이 중 결과에 영향이 클 만한 것은 temporal 방식과 BEV 후처리(`bev_backbone`/`bev_neck`)다.
둘 다 PARA-SSR encoder를 SafeDrive로 바꾸는 일이라 이번 범위 밖으로 두었고, 필요하면 별도
arm으로 붙인다.

### spconv 설치

PyPI에 `spconv-cu128`은 없지만 **`spconv-cu126==2.3.8`(cumm-cu126 0.7.11) prebuilt wheel이
torch 2.8.0+cu128 / RTX 5090(sm_120)에서 그대로 동작**한다. 이 machine에서 `ssr` env 위에
격리 venv(`--system-site-packages`)를 만들어 확인했다: SubM/stride-2 sparse conv
forward·backward 유한, SafeDrive 크기 stack(`[41, 400, 800]`, 50k voxel, B=4)
forward+backward **28 ms/iter, 1.45 GiB**.

```bash
conda activate ssr
pip install spconv-cu126==2.3.8      # requirements_navsim.txt에 추가됨
python -c "import spconv.pytorch as s; print(s.__name__)"
```

SafeDrive 자체 환경은 torch 2.1 + CUDA 11.8(`spconv-cu118`)이지만 API(`spconv.pytorch`,
`replace_feature`)는 2.x 계열이 같다. CUDA toolkit 12.8과 nvcc가 `/usr/local/cuda-12.8`에
있으므로 필요 시 `CUMM_CUDA_ARCH_LIST="12.0"`로 source build도 가능하나, prebuilt로 충분하다.

### 두 encoder

| `lidar_encoder` | 구현 | 의존성 |
|---|---|---|
| `sparse` (기본) | SafeDrive `voxelization` + `SpMiddleResNetFHD` 포팅 (`LidarSparseEncoder`) | spconv |
| `pillar` | dynamic pillar 0.16 m → SECOND 2D backbone → SECONDFPN (`LidarPillarEncoder`) | 없음 |

둘 다 `forward(points [B,N,5], num_points [B]) → [B, embed_dims, bev_h, bev_w]`로 계약이
같아 encoder 배선(query 초기화·cross-attention·gate)은 공유한다. spconv가 없는 환경에서
`sparse`를 고르면 설치 명령을 담은 ImportError로 즉시 실패하며 조용히 pillar로 대체하지
않는다. `sparse`의 voxel은 stride 8이 고정이라 `cell / 8 = 0.08 m`여야 하고 config 검증이
이를 강제한다 (smoke는 `bev 10×20` → voxel 0.4 m).

## 3. Feature와 cache

새 feature 두 개가 추가된다.

| key | shape | dtype |
|---|---|---|
| `lidar_points` | `[T, lidar_max_points, 5]` = (x_right, y_forward, z, intensity, ring) | float32 |
| `lidar_num_points` | `[T]` | int64 |

trainval 실측(12 frame): frame당 약 91k점 중 전방 ROI 안은 평균 49.8k, 최대 53k. 65,536행
padding이면 잘리지 않는다. 초과 시 scan 순서로 균등 thinning하며 RNG를 쓰지 않아 cache가
결정적이다. Cache key(`para_ssr_feature_*`)에는 `use_lidar`, layout, `lidar_max_points`,
`lidar_z_range`가 들어가고 pillar/backbone 크기는 모델 쪽이라 들어가지 않는다.

`bev_h`/`bev_w`는 원래부터 feature key에 포함돼 있다 (`bev_shift` 정규화에 쓰이므로).
따라서 100×100 feature cache는 이름이 달라 자동으로 분리되고, **새 학습은 feature cache를
다시 만든다**. Target cache는 grid에 의존하지 않아 그대로 재사용된다.

## 4. Grid 정렬과 distillation

Teacher `bev_feature`는 mmdet3d LiDAR 축 `(C, H=x_forward, W=y_left)`, student는
`(C, bev_h=y_forward, bev_w=x_right)`다. 50×100끼리는 측방 flip만 필요하다.

```python
student_bev = teacher_bev[:, :, ::-1]   # transpose 없음, resample 없음
```

주의할 실측이 하나 있다. `bevfusion/runs/`의 같은 조건(20 epoch, `camera_fov: false`,
sweeps 2) 로그에서 teacher detection은 100×100 → 50×100에서 떨어진다: LiDAR-only mAP
0.879 → 0.811, camera+LiDAR 0.904 → 0.824. 이 변경은 head grid와 voxel 크기(0.04 → 0.08)를
동시에 바꿨으므로 grid 자체의 효과는 분리되지 않았다. 50×100은 teacher-student 정렬을
우선한 선택이며, voxel 0.04를 유지하고 out_size_factor만 16으로 올리는 조합은 검증되지 않았다.

## 5. 검증

- `pytest -q tests navsim/agents/para_ssr`: **149 passed** (`ssr` env에 `spconv-cu126 2.3.8`이
  설치된 상태; spconv가 없으면 sparse 케이스 10개가 설치 안내와 함께 skip). LiDAR 회귀
  테스트: 축 회전·ROI clip·half-open 경계·padding·결정적 thinning, cache 분리,
  pillar→BEV 셀 대응, sparse voxel 좌표 `(b, z, y, x)`·mean-pool·grid 검증, gate·layer
  입력 검증, 두 encoder 모두 전체 모델 forward/backward 유한성, DDP를 위한 미사용
  parameter 없음, 과거 frame LiDAR가 `prev_bev`에 반영됨, history pass BN 모드, 빈/1-voxel
  배치 생존.
- `scripts/training/smoke_para_ssr.sh` (실 navtrain 8 scene, LiDAR 포함, fp32, GPU 1장):
  train forward/loss/backward/optimizer step + validation 통과 — pillar, sparse(spconv)
  각각. Smoke는 `bev_h=10, bev_w=20, lidar_voxel_size=[0.4,0.4,0.2],
  lidar_pillar_size=[0.8,0.8]`로 줄인 모델을 쓴다.
- 실 navtrain 3 scene을 agent의 sensor config로 로드한 feature 검사: LiDAR는 frame 2, 3에서만
  로드되고 (`[False, False, True, True]`), frame당 raw 97k점 → ROI clip 후 46~48k점,
  `x_right ∈ [-32, 32)`, `y_forward ∈ [0, 32)`, `z ∈ [-2.7, 5.0)`, intensity ≤ 1, ring ≤ 0.975.
- 본 모델 크기 비용 (B=4, 768×416×3 cam×2 frame, 50k점/frame, fp32, RTX 5090 1장,
  forward+backward):

| | parameter | ms/iter | peak memory |
|---|---:|---:|---:|
| camera-only (`use_lidar: false`) | 37.1M | 167 | 11.8 GiB |
| camera + LiDAR, `lidar_encoder: pillar` | 41.5M (+5.1M) | 196 (+17%) | 13.5 GiB |
| camera + LiDAR, **`lidar_encoder: sparse`** (기본) | 39.1M (+2.7M) | 271 (+62%) | 15.2 GiB |

`bev_embed`는 세 경우 모두 `[4, 5000, 256]`이다 (100×100 대비 query 절반). sparse
stack 단독은 28 ms인데 모델 안에서 +104 ms인 이유는 voxelization(`torch.unique`,
frame당 ~50k점 × 2 frame × B=4)과 spconv indice 생성이 매 forward에 들어가기 때문이다;
2-GPU × B=4 × accumulate 16 recipe 기준 epoch당 약 1.6배로, 학습 시간 예산에 반영해야 한다.

## 5-1. 실데이터 계측 검증

축소 smoke는 "한 step이 돈다"만 보므로, **본 모델 크기**로 실 navtrain 8 scene을 돌리며
LiDAR 경로가 조용히 틀어질 수 있는 지점을 전부 수치로 확인했다
(**PASS 125 / WARN 0 / FAIL 0**).

| 구간 | 확인한 것 | 실측 |
|---|---|---|
| A 특징 생성 | frame별 raw/ROI/z drop/kept, half-open 경계, padding=0, 정규화, thinning, 시간 | raw 94~97k → kept 44~56k (padding의 67~85%), thinning 0회, pcd decode ≈1 ms/frame, 특징 생성 17 ms/sample (camera-only와 동일) |
| B voxel화 | sample당 voxel 수, grid 점유율, 좌표 범위·유일성 | 27~30k voxel/sample, 800×400×40의 0.23%, 좌표 모두 sparse_shape 내부 |
| C LiDAR BEV | train/eval BN 모드별 통계, 0 셀 비율 | sparse train-BN: std 0.45 (active 셀 max 13.9), eval-BN(미학습 running stat): std 0.0025 — **~180배 차이**; 0 셀 50.8% (raw 점 점유 셀 43.7%와 정합) |
| D 카메라 가시성 | 3 카메라가 보는 BEV 셀 비율 | 92.1% (나머지 8%는 LiDAR·temporal만) |
| D2 history BN 모드 | 과거 frame pass에서 LiDAR encoder의 BN 모드 | train→train, eval→eval (아래 수정 후) |
| E 학습 3 step | loss/모든 log 항 유한, 그룹별 grad norm, gate, 미사용 parameter, 시간/메모리 | lidar_encoder grad 9.6~13.5 (image 8~25), gate grad>0, 미사용 0, 141~201 ms/step(B=2), 7.2 GiB |
| F 추론 | `run_aux=False` forward, `compute_trajectory_gpu` 실 AgentInput | trajectory [B,8,3] 유한 |
| G checkpoint | Lightning prefix 저장 → strict load → 같은 입력 재현 | max diff 2.2e-7, LiDAR encoder tensor 142개 |
| H 경계 | batch 내 빈 sample / 전부 빈 / 1 voxel / 2 voxel, train·eval | 전부 통과 (아래 guard 후) |
| I 정밀도 | fp16 autocast 하 sparse encoder | 동작(출력 fp32); recipe는 fp32 |
| J 회귀 | pillar, camera-only arm 1 step | 통과 |

계측이 드러낸 문제와 수정 두 가지:

1. **빈/1-voxel 배치에서 spconv·BN 크래시.** spconv는 voxel 0개에서 cumm kernel assert로,
   BatchNorm1d는 train 모드 voxel 1개에서 예외로 죽는다. 실데이터에서는 드물지만 손상된 pcd
   하나가 며칠짜리 학습을 세운다. `LidarSparseEncoder.forward`가 voxel 0개면 0 BEV를 내고
   (parameter 합×0 graph edge로 DDP 미사용-parameter 오류 방지), voxel <2면 BN을 running stat으로
   한 번 돌린다. H 구간과 단위 테스트가 이를 고정한다.
2. **history pass의 BN 모드 불일치.** `obtain_history_bev()`는 SSR대로 `self.eval()`로 과거
   frame을 돌린다. 이미지 backbone은 BN이 얼어 있어 무해하지만, 처음부터 학습되는 LiDAR
   encoder의 BN(momentum 0.01)은 running stat이 따라올 때까지 과거 BEV를 현재 BEV와 전혀 다른
   scale(위 C의 180배)로 만들어 temporal self-attention에 넘긴다. SafeDrive는 모든 frame을 같은
   모드로 처리한다. `obtain_history_bev()`가 `was_training`이면 `lidar_encoder.train()`으로
   되돌리도록 고쳤고(D2, 단위 테스트), 이미지 backbone의 `norm_eval` 정책은 그대로다.

참고: `ego_motion_mlp` 출력(LayerNorm, std≈1)이 모든 query에 더해지는 SSR 설계는 유지했다.
SafeDrive는 `use_can_bus=False`라 LiDAR query에 아무것도 더하지 않는다.

## 5-2. 데이터셋부터의 기하·스케일·loss 검증

카메라·LiDAR·annotation·map·궤적이 **같은 좌표계와 스케일**에 있는지를 raw 데이터와 builder
출력만으로 교차 검증하고, loss가 실제로 입력에 반응하는지 확인했다. 실 navtrain 16 scene,
본 모델 크기.

| 구간 | 확인 | 실측 |
|---|---|---|
| A1 extrinsics | 카메라 광축 yaw | f0 0.0°, l0 +55.0°, r0 −54.5° (좌/우 대칭), 초점 1529 px @1920×1080 |
| A2 LiDAR→이미지 | 모델의 `lidar2img`로 전방 ROI 점 투영 | 이미지 안 25/32/35% (f0/l0/r0); 안에 들어온 점의 mean x_right: l0 −12.1 m, f0 +0.2 m, r0 +10.4 m → LiDAR 회전 부호가 카메라 extrinsics와 일치 |
| A3 BEV 셀↔픽셀 | 점의 픽셀 vs 그 점이 속한 BEV 셀 중심의 픽셀 | 중앙값 6.8 px, 100%가 투영 셀 크기의 1.5배 이내; 전치(x↔y) 가설은 346 px → encoder reference point가 x→W, y→H로 셀을 정확히 가리킴 |
| A4 GT box ↔ LiDAR | box 안 LiDAR 점 수 | vehicle 36개 100%가 ≥5점(중앙값 728), 같은 크기 임의 box 대비 ×6.1; yaw 규약(길이축 = (−sin ψ, −cos ψ)) 축 교환보다 2배 많은 점 |
| A5 map ↔ ego | ego 원점에서 가장 가까운 centerline | 중앙값 0.11 m (최대 0.29) → map target이 ego frame |
| A6 궤적 스케일 | 첫 미래 pose vs \|v\|·0.5 s | 상대 오차 중앙값 6%, 100%가 전방 종료 |
| A7 `bev_shift` | shift × BEV 범위 vs ego 변위 | 오차 0.0 m (64 × 32 m 정규화 정확), \|v\|·0.5 s 대비 3% |
| A8 카메라 가시성 | encoder pillar가 보는 셀 | f0 28%, l0 36%(mean x −19 m), r0 36%(+19 m), 합집합 92% |
| A9 overlay | 3 카메라 LiDAR/box 투영 + BEV | 육안 확인: 점이 차량·건물에 얹히고 box가 차량을 감쌈 |
| B 50×100 addressing | 공용 deformable sampler | 2×4 격자에서 (x=col 3, y=row 1) → key 7; head는 `spatial_shapes=[[50,100]]`, PE 표 50/100 |
| D 수정 기본값 | 3 카메라·grid·FOV | images `[B,2,3,3,416,768]`, lidar2img `[B,3,4,4]`, cams_embeds `[3,256]`, 0.64 m 셀 5,000 token; 전방 사각형 51 box 중 ±80°가 49(96%) 유지, scene당 3.1 box |

### loss가 실제로 반응하는가 (C)

실 2 sample, 본 모델, lr 2e-4, 200 step overfit: total 55 → 3.5, `loss_cls` 2.37 → 0.009,
`loss_map_cls` 2.19 → 0.002, `loss_map_pts` 7.3 → 0.45, `loss_plan_reg` 0.14 → 0.002,
planner ADE 0.12 m. 입력 gradient도 두 modality 모두 0이 아니다.

그 뒤 modality ablation(aux head 포함 full loss, eval): **LiDAR 제거 +204%** (`loss_cls`
0.00 → 2.5), **이미지 제거 −0%**, 둘 다 제거 +204%. `bev_embed` 변화율: LiDAR 제거 35%,
이미지 제거 1%. 즉 이 시점의 fused 모델은 2 sample을 LiDAR만으로 구분한다.

이것이 카메라 경로의 결함인지 확인하기 위해 (C2) **camera-only arm을 ego-motion conditioning
없이**(`use_ego_motion=False`; 이게 켜져 있으면 sample별 ego 벡터만으로도 2 sample을 외울 수
있다) 같은 batch에 200 step overfit했다: `loss_cls` 2.24 → 0.038, `loss_map_pts` 7.5 → 0.81로
떨어지고, 이미지를 지우면 loss +38%, `bev_embed` 변화 34%. **카메라 경로는 end-to-end로
동작한다.** fused 모델의 마지막 step에서 두 branch의 attention 출력 norm은 camera 2.1~2.6k vs
LiDAR 0.8~1.0k로 카메라 쪽이 오히려 크다 — 크기가 작아서가 아니라, LiDAR가 이미 과제를 풀어
카메라 출력이 입력 내용에 둔감한 채로 남은 것이다(2 sample overfit의 특성).

전체 학습에서 카메라 branch가 내용을 학습하는지는 이 검증으로 단정할 수 없으므로 학습 log에
`lidar_gate/layer{i}`(gate sigmoid 평균)와 `attn_norm/{lidar,camera}_layer{i}`(두 branch
attention 출력 크기)를 추가했다. gate가 1로 수렴하거나 camera norm이 LiDAR 대비 두 자릿수
아래로 내려가면 starvation이며, 그때는 camera-only arm과의 비교로 판단한다.


- 기존 checkpoint는 grid·LiDAR branch·`bev_embedding` 유무가 달라 strict load되지 않는다.
  새 학습으로 시작한다.
- Dataloading: frame당 pcd 2개를 추가로 읽는다. IO가 병목이면 `WORKERS`를 먼저 올린다.
- 전방 ROI와 detection ±80° FOV는 카메라 시야 근거로 정해졌다. LiDAR는 360°를 보므로 ROI를
  다시 검토할 여지가 있으나, 이번 변경은 ROI를 건드리지 않았다.
- spconv를 빌드하면 `LidarPillarEncoder`를 SafeDrive의 sparse SECOND로 교체한 ablation을
  같은 wiring 위에서 돌릴 수 있다.
