# PARA-SSR task-memory planner 구현·검증

2026-09-15. 적용 대상은 현재 NAVSIM 학습·평가에서 사용하는
`navsim/agents/para_ssr`이다. `projects/mmdet3d_plugin/SSR`의 이전 nuScenes 구현은
수정하지 않았다.

이 문서의 테스트 결과와 미검증 목록은 각 변경 시점의 누적 기록이다.
후속 전체 코드 검토에서 발견한 AMP·평가 재개 문제의 수정과 **최신 검증 범위**는
[16_code_audit.md](16_code_audit.md)에 정리했다. 현재 GradBalancer 활성 학습은 FP32 전용이다.
현재 구조를 그림과 함께 설명한 문서는 [17_architecture_guide.md](17_architecture_guide.md)에 있다.
최신 변경은 이 문서 마지막의 **Plan-det / plan-map 병렬 분기** 절이며, 그 이전 smoke와
성능 기록은 각 당시 구조의 결과다.

## 구현 구조

`use_stl=false`, `plan_num_layers=3`을 유지한다. 기본 interaction 모드에서는
`use_task_interaction=true`이고 det/motion 및 map head 활성화를 요구한다.
아래 상세 memory/metadata 구조는 interaction on 기준이다.
Scene TokenLearner, scene-token decoder, planner gate는 생성하지 않는다.
기존 BEV encoder, det/motion·map decoder와 command embedding을 재사용한다.
후속 요청으로 속도·가속도는 BEV conditioning에서 planning query로 옮겼다.
과거 frame BEV는 기존대로 `no_grad`에서 계산한다. 현재는 `bev_shift`와 캐시의 상대 yaw로
이동·회전 정렬한 history feature를 TSA에 전달한다. 세 모드 모두 같은 정렬을 사용한다.

세 실험 모드를 지원한다. 실행 명령은 [학습 가이드](README.md#3-본-학습)에 있다.

| 모드 | `use_task_interaction` | `use_det_motion_head` / `use_map_head` | Planner | 학습 loss |
|---|---|---|---|---|
| Interaction | true | true / true | `[BEV → (det ∥ map) → 합산 → FFN] × 3` | plan+det+motion+map |
| Parallel | false | true / true | `[BEV → FFN] × 3` | plan+det+motion+map |
| Planning only | false | false / false | `[BEV → FFN] × 3` | plan |

Off에서는 interaction attention·norm·projection·metadata MLP를 생성하지 않는다.
Parallel은 공유 BEV에서 supervision만 병렬로 전달하며 planning loss가 private head로
흐르지 않는다. 기본 eval에서는 head를 생략하고 `run_aux=True`/`test_aux_heads=True`일 때만
auxiliary prediction을 계산한다. Planning only는 auxiliary head와 target 생성까지 생략하며
GradBalancer도 자동으로 끈다. Off 두 모드의 planning 구조는 동일하다. 이는 현재 planner에서
interaction을 제거한 비교이며, PARA-Drive 논문의 전체 recipe를 그대로 재현한 모델은 아니다.

```text
camera + history(SE(2) feature warp); use_lidar=false → shared BEV
                                     ├→ det decoder → motion decoder
                                     ├→ map decoder
                                     └→ planner BEV attention
command + learned planning query + MLP(ego velocity, acceleration)
    → [BEV → (det ∥ map) → 합산 → FFN] × 3 → final LN → trajectory head
```

세 planner layer는 서로 다른 파라미터를 가진다. BEV attention 후의 같은 hidden에
각 분기의 Pre-LN을 적용해 plan-det과 plan-map을 독립적으로 계산한다. 두 결과를
`h_bev + det_update + map_update`로 합산한 뒤 Pre-LN FFN residual을 적용한다.
같은 층의 map attention은 det attention 결과를 입력으로 받지 않는다.
이 병렬성은 계산 그래프의 분기 독립성을 뜻하며 CUDA kernel 동시 실행을 보장하지 않는다.
Det와 map의 private decoder 사이에는 새 forward dependency가 없다.

Memory는 한 번 구성하고 세 층에서 재사용한다.

| Memory | 실제 마지막 latent | 구성 |
|---|---|---|
| Det/motion | det `[B,Q,C_det]`, motion `[B,Q,M,C_motion]` | `LN(det_proj(det) + motion_proj(mean(motion, mode)))` |
| Map | `[B,V,P,C_map]` | `LN(map_proj(mean(points, point)))` |

기본값은 `Q=300`, `M=6`, `V=100`, `P=20`, `C=256`이다. 구현의 reshape는 기존
head 설정을 사용하고, planner attention과 projection은 query 수를 고정하지 않는다.
Planner의 별도 latent 차원 projection도 테스트했다. Top-k, threshold, mode 선택,
학습형 pooling은 추가하지 않았다.

위치와 confidence는 latent와 별도로 처리한다.

- Det의 `all_bbox_preds[-1, ..., :2]`는 metric XY이므로 shared BEV range로 한 번 정규화한다.
- Map의 `all_map_pts_preds[-1]`는 이미 `[0,1]` 좌표다. 그대로 coordinate MLP에 넣은 뒤
  point 방향으로 평균한다. BEV와 map의 `pc_range` 일치도 검사한다.
- 두 head 모두 background logit 없는 sigmoid focal 분류다. 기존 foreground confidence인
  `sigmoid(logits).amax(-1)`를 사용한다.
- 좌표와 score는 MLP 전에 detach한다. Metadata MLP의 파라미터와 latent는 학습된다.
- `Q=LN(h)+plan_pos`, `K=LN(memory)+position+confidence`, `V=LN(memory)`이다.
  위치와 confidence는 key에만 각각 한 번 더한다.

`ParaSSRModel.forward`는 BEV 생성 → det/motion head 한 번 → map head 한 번 → planning
순서다. Head는 `return_hidden=True`로 기존 prediction branch와 마지막 latent를 함께 반환한다.
Interaction on에서 `run_aux=False`와 `test_aux_heads=False`도 decoder 실행을 생략하지 못한다. 학습에서는 모든
supervision용 출력을 반환하고, eval에서는 기존처럼 auxiliary 출력 노출만 이 flag로 결정한다.
GT나 matching은 forward에 필요하지 않고, matcher는 명시적인 loss 계산 경로에만 있다.

기존 planning query 1개, command 4개, trajectory MLP, 8-step/4-second horizon,
`ego_fut_preds [B,4,8,3]`, `trajectory [B,8,3]`와 evaluation API를 유지한다.
한 command-conditioned trajectory를 기존 branch 차원에 expand하고, 기존 누적 offset
변환과 loss를 사용한다. Task loss weight는 `plan=2, det=1, motion=1, map=1`로 유지한다.

## GradBalancer 처리

Decoder BEV 입력의 `_ScaleGrad`를 제거했다. 이 위치에서 gradient를 줄이면 decoder를
통과하는 planning loss까지 det/map task로 잘못 줄어든다.

새 `balance_shared_gradients`는 loss 출처별로 BEV의 VJP를 계산한다. Total loss의 forward
값과 private parameter gradient는 그대로 두고, shared BEV backward에만
`(s_task-1) * dL_task/dBEV`를 보정한다. 최종 shared gradient는 다음과 같다.

```text
g_BEV = dL_plan/dBEV
      + s_det * d(L_det + L_motion)/dBEV
      + s_map * dL_map/dBEV
```

`dL_plan/dBEV`에는 직접 BEV와 det/motion·map 경유 경로가 모두 포함된다.
Det와 motion loss는 더한 뒤 미분하므로 두 gradient 사이의 상쇄도 반영한다.
측정된 task norm에는 현재 계수를 적용해서 기존 GradBalancer update 계약을 유지한다.
Controller의 새 계수는 다음 loss graph부터 적용한다.

보정은 각 loss graph에 속하는 autograd Function에서 수행하며 upstream loss 배율도 함께
곱한다. FP32 graph에서 gradient accumulation, scalar loss scaling, 여러 미완료 graph를
지원한다. 이것이 FP16/BF16 decoder 연산에서의 수치 정확성을 보장하지는 않는다.
후속 검토에서 AMP의 보정 상쇄 오차를 확인했으므로, GradBalancer가 활성화된 학습은
warm-up부터 FP32를 요구한다. AMP inference와 balancing을 끈 AMP 학습은 별도다.
Private decoder는 supervision과 planning의 원래 gradient를 모두 받는다.

## 완료 조건별 검증

| 조건 | 결과·근거 |
|---|---|
| 1. STL 미실행 | 생성 모듈 검사 및 TokenLearner forward를 실패 처리한 train/eval·CUDA 테스트 통과 |
| 2. 3층 attention 분기 | Hook으로 BEV 후 동일 hidden에서 det/map 분기, 두 결과 합산 후 FFN을 검증; 3층×양방향 memory 교란 및 gradient 독립성 검증 |
| 3. Train/eval decoder 한 번 | `run_aux=None/False/True`, B=2와 history frame 포함 실제 모델에서 det/motion/map 각 1회 |
| 4. GT 없는 inference, matcher 미호출 | Matcher를 실패 처리한 forward 및 NAVSIM `compute_trajectory`/`compute_trajectory_gpu` 통과 |
| 5. Planning-only backward | 실제 decoder·BEV encoder·image backbone에 유한한 nonzero gradient; sparse/pillar LiDAR 회귀 테스트 포함 |
| 6. Latent 유지·metadata detach | Exported latent gradient 및 coordinate/confidence MLP 입력 검사; prediction branch로의 planning gradient 없음 |
| 7. Private decoder 새 dependency 없음 | Det latent를 map parameter로, map latent를 det parameter로 미분하면 모두 unused |
| 8. Trajectory/evaluation 계약 | Shape, command 조건화, offset 누적, 양쪽 평가 API, checkpoint strict round trip 통과 |
| 9. 전체 planning gradient 처리 | 실제 decoder의 `g_plan = g_direct + g_det_route + g_map_route` 수치 검증; 계수 0 포함 synthetic gradient oracle 및 2-rank Gloo 검증 |
| 10. 파일·결과·한계 보고 | 아래 목록 및 범위 참조 |

검증 환경: Python 3.9, PyTorch 2.8.0+cu128, RTX 5090. 실행 interpreter는
`/home/external-user/miniconda3/envs/ssr/bin/python`이다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q tests --disable-warnings
python tools/verify_task_memory_cuda.py --device cuda:0
```

Interaction on 최초 구현의 pytest: **196 passed, 99 warnings, 18.79초**.
[원본 결과](../work_dirs/task_memory_validation/pytest.log). Test에는 full supervision 2회
optimizer step, GradBalancer state checkpoint 복구, 두 rank의 연속 2회 DDP backward 및
controller 동기화도 포함된다. 기존 STL/head-removal 테스트의 기대값은 새 구조의 명시적
거부로 변경했고, LiDAR logging 테스트도 두 head supervision을 사용하도록 바꿨다.
이후 추가된 on/off 설정에서는 interaction off에 한해 head 제거를 다시 지원한다.

기본 크기 CUDA probe는 `BEV [1,5000,256]`, det 300개, motion 6 mode, map 100×20,
실제 decoder로 수행했다. **FP32/BF16 각각 GT-free eval과 planning-only backward 모두 통과**했다.
FP32 peak allocated memory는 eval 321.2 MiB / backward 1595.1 MiB, BF16은
393.3 MiB / 1395.1 MiB였다. 이는 random BEV 경계부터 실행한 head/planner probe이며,
image·BEV encoder와 GradBalancer 비용을 제외한 값이다. 전체 모델 메모리나 throughput으로
해석하지 않는다.
[CUDA probe 원본 JSON](../work_dirs/task_memory_validation/cuda_probe.jsonl)에 각 stage의
shape, attention 순서, gradient norm을 보관했다.

**실제 NAVSIM smoke도 통과(exit 0)**했다. GPU 1, sparse LiDAR, 기본 det/map query 수,
축소한 `10×20` BEV와 encoder 1층에서 8 train/8 validation scene을 로드하고,
실제 train 1 batch loss→backward→optimizer step 및 validation 1 batch를 마쳤다.
`fast_dev_run`으로 장기 학습 없이 종료했다. 설정은 실행 산출물의 Hydra config에 저장되어 있다.

- [실데이터 smoke 로그](../work_dirs/task_memory_validation/real_smoke.log)
- [실행 기록](../work_dirs/task_memory_validation/smoke_1789401595092/train_time.json)
- [실제 실행 config](../work_dirs/task_memory_validation/smoke_1789401595092/code/hydra/config.yaml)

## 변경 파일

| 구분 | 파일 |
|---|---|
| Model 실행 순서 | `navsim/agents/para_ssr/para_ssr_model.py` |
| BEV ego conditioning 제거 | `navsim/agents/para_ssr/modules/bevformer.py` |
| Planner 및 memory/metadata | `navsim/agents/para_ssr/modules/planner_head.py` |
| Latent 반환 | `navsim/agents/para_ssr/modules/det_motion_head.py`, `map_head.py` |
| Loss 출처별 gradient | `navsim/agents/para_ssr/para_ssr_loss.py`, `modules/grad_balance.py` |
| 기본 설정 | `navsim/agents/para_ssr/configs/default.py`, `navsim/planning/script/config/common/agent/para_ssr_agent.yaml` |
| 신규 테스트 | `tests/test_para_ssr_decoder_hidden.py`, `test_para_ssr_task_memory_planner.py`, `test_para_ssr_task_memory_integration.py`, `test_para_ssr_planning_gradients.py` |
| 기존 기대값 갱신 | `tests/test_para_ssr_para_drive_planner.py`, `test_para_ssr_head_ablation.py`, `test_para_ssr_lidar.py` |
| CUDA probe | `tools/verify_task_memory_cuda.py` |
| 문서 | 이 문서, `report/README.md`, `docs/PARA_SSR_NAVSIM.md` |

## 설계 피드백과 검증 한계

요청 구조를 막는 dependency나 tensor-layout 문제는 발견하지 않았다. 다음 항목은 학습
실험에서 확인할 필요가 있다.

1. **추론 비용 증가:** eval에서도 1,800개 motion token과 2,000개 map point의 기존 decoder를
   실행한다. 과거 train-only auxiliary head의 latency 이점은 유지되지 않는다.
2. **Gradient balancing 비용:** 계수가 1이 아닌 auxiliary task마다 microbatch당 BEV까지
   추가 VJP가 필요하다. Decoder forward는 계속 한 번이지만 backward traversal은 늘어난다.
   보정은 ordinary first-order 학습을 대상으로 한다.
3. **Private parameter 충돌:** BEV GradBalancer는 private decoder에서 planning과 supervision
   gradient가 충돌하는 문제까지 해결하지 않는다. Planning/auxiliary 성능과 gradient norm을
   함께 관찰해야 한다. 현재 loss weight와 balancing target은 요구대로 유지했다.
4. **평균 pooling의 정보 손실:** motion mode 평균은 mode 간 차이를, map point 평균은 point
   순서 정보를 압축한다. 좌표 MLP 평균이 일부 공간 정보를 제공하지만 순서 표현 자체는
   permutation invariant다. 이는 요청한 단순 pooling의 tradeoff이며 추가 interaction은 넣지 않았다.
5. **Checkpoint 호환성:** 과거 STL/post-LN BEV-only planner checkpoint는 현재 planner와
   파라미터가 달라 strict load되지 않는다. 현재 on/off checkpoint도 동일 모드 flag로 복원해야 한다.
   과거 실험 재현에는 당시 code revision이 필요하고,
   새 구조는 새 학습 또는 별도로 검토한 부분 weight 초기화가 필요하다.

전체 해상도 sensor 입력의 장기 학습·수렴·PDMS/EPDMS/auxiliary mAP, 실제 모델의 다중 GPU
NCCL 학습, FP16/GradScaler를 포함한 전체 학습, production batch 크기의 peak memory와
latency는 검증하지 않았다. BF16 검증은 위의 head/planner CUDA probe 범위다.
기존 라이브러리 deprecation/spconv warning은 남아 있다.

## Precision과 추가 개선안

기존 학습·smoke wrapper는 `trainer.params.precision=32`였다. 반면 공통
`default_training.yaml`은 `16-mixed`여서 wrapper 없이 `agent=para_ssr_agent`로 직접
학습하면 FP16 mixed precision을 상속했다. 현재는 PARA-SSR의 `training_precision=32`를
공통 config가 읽어 두 실행 모두 FP32로 통일한다. 다른 agent의 기존 기본값과 명시적인
`trainer.params.precision` override는 유지한다. BF16 CUDA probe는 별도 검증이다.

Final LN, 직접 ego 입력, confidence bias, motion pooling, map flatten, yaw/velocity 제안의
검토는 [baseline 설계 피드백](15_baseline_design_review.md)에 있다. Toggle 구현 당시에는
여섯 제안을 추가하지 않았다. 이후 사용자 요청과 reference 코드 확인에 따라 final LN을
세 모드 공통으로 추가했다. 이어 ego 직접 입력도 세 모드에 공통 적용하고 기존 BEV
ego conditioning은 제거했다. 나머지 네 제안은 적용하지 않았다.

## Interaction toggle 추가 검증

전체 pytest는 **231 passed, 164 warnings, 25.45초**였다. 추가된
`tests/test_para_ssr_training_modes.py`의 31개 test는 세 모드의 decoder 실행 조건,
planning gradient 분리, 모드별 2회 optimizer step, 모든 학습 파라미터의 gradient,
checkpoint/evaluation API, plan-only target 생략, Hydra precision 선택을 확인한다.
Planner 자체의 off 테스트도 추가했다. On의 변경 전후 state_dict, 초기 값, 출력 및
backward gradient는 직접 비교에서 정확히 같았다.

세 wrapper를 `WANDB=0 ... --cfg job --resolve`로 실제 실행해 각각의 모드 flag,
experiment 이름, `trainer.params.precision=32`를 확인했다. 수정·추가한 다섯 training
wrapper는 `bash -n`을 통과했고, `git diff --check`도 통과했다.

Parallel과 planning-only의 실제 NAVSIM FP32 GPU smoke도 각각 **exit 0**이었다.
두 실행 모두 sparse LiDAR, BEV `10×20`, encoder 1층으로 train 1 batch의
loss→backward→optimizer step(`global_step 0→1`)과 validation 1 batch를 완료했다.
Parallel은 기존 기본 query 수를 사용했고, planning-only는 auxiliary head가 없다.
Planning-only는 기본 multi-task target config를 남겨도 GradBalancer가 자동으로 꺼졌다.

- [Parallel smoke 로그](../work_dirs/interaction_modes_validation/parallel_smoke.log)
- [Parallel 실행 기록](../work_dirs/interaction_modes_validation/parallel_1789403891/train_time.json)
- [Planning-only smoke 로그](../work_dirs/interaction_modes_validation/plan_only_smoke.log)
- [Planning-only 실행 기록](../work_dirs/interaction_modes_validation/plan_only_1789403945/train_time.json)
- [실제 scene의 planning-only target 검증](../work_dirs/interaction_modes_validation/plan_only_real_targets.json):
  두 auxiliary target 함수를 실패 처리해도 trajectory·offset·mask·command만 정상 생성.

이번 toggle 관련 변경은 model/planner/config/loss, 공통 `default_training.yaml`의
precision 선택, `train_para_ssr_interaction.sh`·`train_para_ssr_parallel.sh` 신설 및 기존
세 head-ablation wrapper의 mode flag 추가, 관련 테스트·문서다. 장기 학습과 성능 비교는
실행하지 않았다. 앞 절의 full-resolution·NCCL·FP16/GradScaler 검증 한계는 그대로다.

## Final LayerNorm 추가

로컬 SafeDrive와 WoTE는 Post-LN decoder의 마지막 LN을 통과한 hidden을 trajectory
regression head에 전달한다. 별도의 stack-final LN이 있는 것은 아니지만, 회귀 입력은
정규화되어 있다. [reference 코드 근거](15_baseline_design_review.md#final-layernorm과-ego-status)를
확인한 뒤 현재 Pre-LN planner도 같은 역할의 출력 정규화를 갖도록 변경했다.

```python
for layer in self.planner_layers:
    h = layer(...)
h = self.final_norm(h)  # LayerNorm(C), 모든 모드에서 한 번
plan = self.ego_fut_decoder(h[:, 0])
```

Interaction on / parallel / planning-only 모두 같은 final LN을 사용한다. 각 block 내부의
Pre-LN residual, query 수, trajectory head, 출력 shape 및 loss weight는 유지한다.
새 learnable parameter는 `final_norm.weight/bias`, 합계 `2C`개다. 기본 C=256에서는 512개이며,
이 key가 없는 이전 checkpoint는 현재 strict loader에 그대로 복원되지 않는다.

검증: **233 passed, 164 warnings, 25.30초**, `git diff --check` 통과.
추가 검사는 마지막 layer → final LN 한 번 → trajectory head의 실행 순서, feature 차원의
정규화, LN parameter gradient를 interaction on/off에서 확인했다. 기존 세 모드의
optimizer step·checkpoint round trip·전체 gradient 경로 검사도 통과했다.
이 final LN 추가 뒤에는 실데이터 장기 학습이나 성능 비교를 실행하지 않았다.

이번 변경 파일은 `modules/planner_head.py`, `tests/test_para_ssr_task_memory_planner.py`,
이 문서, `report/15_baseline_design_review.md`, `report/README.md`다. 이 단계에서는
ego status 경로의 수정 범위만 검토했으며, 실제 변경은 다음 절과 같다.

## Ego status를 planning으로 이동

Interaction / parallel / planning-only에 같은 입력 경로를 적용했다.

```python
ego_status = features["status_feature"][:, config.num_navi_cmd:]  # [B,4]
# NAVSIM 순서: vx_forward, vy_left (m/s), ax_forward, ay_left (m/s²)
h = plan_fuser(cat(learned_plan_query, command_embedding))
h = h + ego_status_encoder(ego_status).unsqueeze(1)  # Linear(4,C) → ReLU → Linear(C,C)
for layer in planner_layers:
    h = layer(h, ...)
trajectory = trajectory_head(final_norm(h))
```

Ego embedding은 첫 attention 전에 한 번 더한다. Command는 기존 별도 embedding을
유지하며 ego MLP에 중복 입력하지 않는다. 속도·가속도 원값을 BEV 좌표로 회전하거나
`pc_range`로 정규화하지 않고, 네 원소에 LayerNorm을 적용하지 않는다. Latent dimension은
기존 `embed_dims`를 사용한다. Planner에 status가 없거나 `[B,4]`가 아니면 오류를 내며,
history/current frame의 `only_bev=True` 실행에는 status를 요구하지 않는다.

`use_ego_motion=false`를 Python/Hydra 기본값으로 두고 PARA-SSR model에서 true를 거부한다.
기존 BEV `ego_motion_mlp`는 생성하지 않으며 model은 encoder에 `ego_motion=None`을 넘긴다.
따라서 속도·가속도·command가 이 MLP를 통해 perception query에 들어가는 경로와
미사용 학습 파라미터가 사라진다. 최초 ego 경로 변경에서는 과거 BEV의 평행 이동 정렬만
유지하고 yaw는 사용하지 않았다. 후속 수정에서는 아래 절처럼 상대 yaw를 geometry로
다시 전달해 이동·회전 정렬을 구현했다. Learned BEV conditioning은 다시 추가하지 않는다.

Feature builder와 cache 형식은 그대로다. Planner는 이미 저장되는 `status_feature`의
현재 속도·가속도를 사용한다. 후속 BEV 정렬은 `ego_motion` 중 상대 yaw 원소만 읽는다. Planner query 수,
command branch, final LN, trajectory shape/horizon, loss weight와 evaluation API는 유지한다.
Planning loss가 ego MLP를 학습하며 BEV 및 interaction memory를 통한 기존 gradient도 유지한다.
Ego MLP는 planning private parameter이므로 shared-BEV GradBalancer의 task 비율 측정 대상에
추가하지 않는다. BEV에 도달하는 planning gradient는 기존처럼 전체 경로로 측정한다.

BEV ego MLP 제거와 planner ego MLP 추가로 checkpoint key가 달라졌다. 이전 checkpoint를
현재 strict loader에 그대로 넣을 수 없으며, 세 모드는 동일한 새 입력 경로로 새 학습한다.
장기 학습의 성능 향상은 아직 확인하지 않았다.

이번 경로 변경 후 검증:

- **254 passed, 182 warnings, 26.28초**. [전체 pytest 로그](../work_dirs/ego_planner_validation/pytest.log).
  새 21개 case는 세 모드의 ego 입력 반응, 첫 layer 전 한 번 주입, sample 간 분리,
  shape 검증, status/ego MLP의 planning gradient, BEV·auxiliary gradient 독립성,
  legacy BEV MLP 제거와 `bev_shift` 동작을 확인했다. 기존 GradBalancer 전체 planning
  경로 분해·두 번 optimizer step·checkpoint/evaluation API 검사도 통과했다.
- 기본 크기 `BEV [1,5000,256]`, det 300, motion 6 mode, map 100×20의 CUDA
  **FP32/BF16 GT-free eval 및 planning-loss-only backward 통과**.
  Ego MLP, det/motion/map decoder, BEV boundary 모두 유한한 nonzero gradient다.
  [CUDA probe JSON](../work_dirs/ego_planner_validation/cuda_probe.jsonl).
  이 probe는 image/BEV encoder와 GradBalancer를 우회한다.
- 실제 NAVSIM FP32 interaction smoke **exit 0**: sparse LiDAR, BEV `10×20`, encoder 1층,
  기본 det/map query 수에서 train 1 batch loss→backward→optimizer step 및 validation 1 batch.
  [로그](../work_dirs/ego_planner_validation/interaction_smoke.log),
  [global_step 0→1 기록](../work_dirs/ego_planner_validation/interaction_smoke/train_time.json),
  [실행 config](../work_dirs/ego_planner_validation/interaction_smoke/code/hydra/config.yaml).
  이번 변경 뒤 off 두 모드의 실데이터 smoke는 다시 실행하지 않았으며 전체 pytest에서
  두 모드의 실제 작은 모델 forward/backward·optimizer step을 검증했다.
- `git diff --check` 통과. 장기 학습·성능 비교·전체 모델 NCCL 및 FP16 학습은 미검증이다.

이번 변경 파일은 model, `modules/planner_head.py`, `modules/bevformer.py`, Python/Hydra
agent 설정, `para_ssr_features.py`의 설명 주석, `tests/test_para_ssr_ego_routing.py`,
기존 planner/integration test, CUDA probe, 이 보고서와 baseline review·학습 가이드다.

## 과거 BEV의 이동·회전 정렬

상대 pose를 이용해 과거 BEV feature 전체를 현재 frame의 grid로 한 번 resampling한 뒤
TSA에 전달한다. [SafeDrive backbone](../../SafeDrive/navsim/agents/safedrive/safedrive_backbone.py)의
`pose_to_3x3_transform` → `shift_feature`와 같은 SE(2) feature warp 원리다.
구현은 [temporal_alignment.py](../navsim/agents/para_ssr/modules/temporal_alignment.py)에 있다.
SafeDrive는 정렬한 history/current feature를 concat하고
convolution으로 융합하지만, PARA-SSR은 기존 temporal self-attention으로 융합한다.
Task interaction on/off, planning-only 모두 이 공통 BEV 경로를 사용한다.

```text
cached bev_shift [B,2] + ego_motion[:,t,2] (relative yaw, radians)
    → current grid의 각 cell을 previous frame의 sampling 좌표로 변환
    → grid_sample(previous BEV), bilinear / zero padding
    → aligned history BEV + current BEV query → TSA

current velocity / acceleration → ego MLP → planning query
```

Feature builder의 이동량은 현재 frame 축으로 표현된다. SSR 축은 `x=right, y=forward`다.
따라서 sampling 좌표의 변환식은 다음과 같다.

```text
t_current = [bev_shift.x * x_span, bev_shift.y * y_span] = [-d_left, d_forward]
p_previous = R(+d_yaw) @ (p_current + t_current)
```

기존 shift를 회전 뒤 그대로 더하면 translation의 좌표계가 맞지 않는다. Helper는
`pc_range`에서 metric cell center를 만들고 위 변환을 적용한 뒤 normalized sampling grid로
바꾼다. 전방 ROI `[-32,32] × [0,32]`에서 ego 원점은 normalized `(0.5,0)`이며,
grid 중앙이나 normalized XY의 동일 길이 가정을 사용하지 않는다. 임의 BEV 크기와
비정사각 cell 크기도 설정에서 계산한다.

PARA-SSR의 기존 reference/grid는 cell center 규칙이므로 warp도 `align_corners=False`다.
SafeDrive의 `align_corners=True`와 pixel-index 변환식을 그대로 복사하지 않았다.
Pose/grid 계산과 저정밀도 입력의 resampling은 FP32로 수행하고 feature dtype으로 돌려준다.
Feature gradient는 유지하며 shift/yaw 입력은 detach한다. 범위 밖 feature는 zero padding하며
경계를 clamp해서 다른 위치의 feature를 복제하지 않는다.

Warp는 이전 BEV가 있는 frame당 한 번, encoder layer loop 전에 실행한다. History가 없는
첫 frame에서는 생략한다. History/current reference point는 모두 current grid를 사용하며
기존 `shift_ref_2d + shift`를 제거해 이중 이동 보정을 막았다. TSA offset/weight predictor도
정렬된 history feature를 읽는다. 새로운 학습 파라미터나 task 간 연결은 없다.

기존 flag 이름 `use_shift`를 유지하되, 현재는 **이동·회전 feature 정렬 전체의 on/off**다.
기본값은 true다. False면 두 정렬을 모두 생략한다. 정렬을 켜고 history가 있는데 yaw를
전달하지 않으면 오류를 내며 zero yaw로 조용히 대체하지 않는다. `bev_shift`는 계속 2차원이고
`bev_yaw`를 별도 인자로 전달하므로 cache 생성·이름·tensor layout은 바뀌지 않는다.

Warp 자체에는 state_dict parameter/buffer가 없어 직전 ego-planner checkpoint와 key가 같다.
따라서 strict load는 가능하지만 시간 융합 동작이 바뀌어 같은 실험 재현으로 볼 수 없다.
과거의 shift-only 실험 재현에는 당시 code revision을 사용한다. 세 모드 비교 학습은
모두 이번 정렬을 적용한 동일한 baseline에서 수행한다.

이번 정렬 변경 후 검증:

- **289 passed, 215 warnings, 26.83초**, skip 없음.
  [전체 pytest 로그](../work_dirs/temporal_alignment_validation/pytest.log).
- 새 geometry test 21개: zero pose identity, ±방향 한 cell 이동, ego 원점 기준 ±회전,
  비대칭 ROI·비정사각 cell, 이동+회전 순서, batch별 다른 pose, 범위 밖 zero padding.
  실제 feature builder의 cache shift/yaw를 독립적인 world-pose 변환과 대조했고
  `+π/-π` heading 경계도 포함했다. FP32 cache 오차를 포함해 interior 좌표는 `2e-6 m`
  tolerance 이내다. CPU/CUDA dtype 복원과 feature gradient·metadata detach를 확인했다.
- 새 integration test 14개: 세 모드 train/eval에서 yaw가 BEV에 반영됨, batch 2·frame 3개·
  encoder 2층에서 정확히 2회 warp, 정렬된 history가 TSA offset/weight predictor에 도달함,
  양쪽 temporal reference가 canonical grid라 shift를 중복 적용하지 않음, history 없음과
  `use_shift=False`에서 warp 생략, 기존 planning gradient 유지.
  기존 ego-routing test는 geometric yaw만 고정하고 나머지 legacy 값의 비사용을 확인한다.
- 실제 NAVSIM sparse-LiDAR FP32 interaction smoke **exit 0**. 기본 det/map query 수,
  BEV `10×20`, encoder 1층에서 train 1 batch loss→backward→optimizer step(`0→1`) 및
  validation 1 batch를 완료했다.
  [로그](../work_dirs/temporal_alignment_validation/interaction_smoke.log),
  [실행 기록](../work_dirs/temporal_alignment_validation/interaction_smoke/train_time.json),
  [실행 config](../work_dirs/temporal_alignment_validation/interaction_smoke/code/hydra/config.yaml).
- `git diff --check` 통과. 장기 학습·PDMS/EPDMS·auxiliary mAP 및 전체 모델 NCCL/FP16
  학습은 미검증이다. 이번 FP16/BF16 확인 범위는 warp의 CUDA 단위 테스트이며 학습 기본값은 FP32다.

이번 변경 파일: `modules/temporal_alignment.py` 신설, `modules/bevformer.py`,
`modules/planner_head.py`, `para_ssr_model.py`, Python/Hydra agent 설정,
`para_ssr_features.py`의 주석, `tests/test_para_ssr_temporal_alignment.py`,
`tests/test_para_ssr_temporal_integration.py`, `tests/test_para_ssr_ego_routing.py`,
이 보고서와 `15_baseline_design_review.md`·`README.md`·`docs/PARA_SSR_NAVSIM.md`.


## 후속 변경: Plan-det / plan-map 병렬 분기

2026-09-15. 각 planner layer의 task attention을 다음처럼 변경했다.

```python
h = h + bev_attention(h, bev)              # Pre-LN attention, residual
det_update = plan_det_attention(h, det_memory)
map_update = plan_map_attention(h, map_memory)
h = h + det_update + map_update
h = h + ffn(ffn_norm(h))
```

위 의사 코드에서 det/map은 같은 post-BEV hidden을 읽고 각자 query LayerNorm을 적용한다.
같은 층의 det 결과가 map query에 들어가는 의존성을 제거했다. 두 update를 합산한 뒤
FFN으로 넘기며, 다음 층은 앞 층의 두 task 정보가 합쳐진 hidden을 읽는다.
Memory/metadata 구성, 3층의 독립 파라미터, final LN, ego 입력, latent gradient,
trajectory 출력과 세 학습 모드 설정은 유지된다. CUDA stream 병렬화를 추가한 것은 아니다.
기존 순차 구조와 state_dict key/shape는 같지만 forward 식이 달라졌으므로 기존 결과
재현에는 당시 코드가 필요하다.

검증 결과:

- 회귀 테스트를 먼저 실행해 기존 순차 구현에서 **4 failed / 3 passed**를 확인했다.
  실패는 공통 post-BEV query 검사와 세 층의 det→map gradient 의존성 검사다.
- 수정 후 planner·통합·planning gradient·학습 모드 관련 테스트 **106 passed**.
- 전체 `tests/` **379 passed, 259 warnings, 74.88초**.
- 세 층 각각에서 det/map 양방향 memory 교란 시 반대 분기의 Q와 update가 동일하며,
  cross-gradient는 unused이고 합산된 output은 양쪽 memory로 유한한 gradient를 전달한다.
- 전체 suite는 train/eval decoder 호출 횟수, GT-free inference, metadata detach,
  BEV와 private decoder의 planning gradient, GradBalancer/2-rank DDP 및 출력 API를 포함한다.
- Camera-only 기본값을 검증하도록 Hydra 테스트 기대값을 갱신하고, 선택적 LiDAR 전용
  fixture에는 `use_lidar=True`를 명시했다. 제품 코드의 `use_lidar=false` 기본값은 유지했다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/external-user/miniconda3/envs/ssr/bin/python -m pytest -q tests --disable-warnings
```

이번 변경 파일:

- `navsim/agents/para_ssr/modules/planner_head.py`: 병렬 attention 분기와 residual 합산.
- `tests/test_para_ssr_task_memory_planner.py`: 공통 hidden·분기 독립성·합산 검증.
- `tests/test_para_ssr_lidar.py`, `tests/test_para_ssr_head_ablation.py`,
  `tests/test_para_ssr_para_drive_planner.py`: 실제 camera-only 기본값과 LiDAR 전용 fixture 명시.
- `scripts/training/train_para_ssr_interaction.sh`: 구조 설명 주석.
- 이 문서, `17_architecture_guide.md`, `README.md`, `docs/PARA_SSR_NAVSIM.md`:
  병렬 분기·현재 camera-only 설정 설명.
- `report/figures/render_task_memory_architecture.py` 및 두 구조도의 PNG/SVG:
  공통 hidden에서 두 attention으로 갈라져 합산하는 흐름 표시.

미검증: 이번 병렬 구조로 실제 데이터 학습·장기 수렴·PDMS/EPDMS·throughput은 측정하지 않았다.
이전 순차 구조의 실데이터 smoke나 CUDA 수치를 새 구조의 성능 결과로 해석하지 않는다.
