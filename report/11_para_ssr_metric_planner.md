# PARA-SSR candidate planner와 NAVSIM metric supervision

새 agent `para_ssr_metric_agent`는 전방 3-camera PARA-SSR에 train-derived trajectory
anchor refinement, imitation classification, learned PDM metric critic을 추가한다.
기존 `para_ssr_agent`는 single-trajectory baseline으로 유지된다.

## 1. 구조와 좌표

```text
3 front cameras + ego status / command
    -> shared front BEV (측방 ±32 m, 전방 0~32 m)
    -> navigation gating / scene tokens
    -> 4 commands × K candidates × 8 future poses
    -> 현재 command의 K개 trajectory
    -> BEV를 읽는 metric critic + imitation logits
    -> 최종 trajectory 1개 [B,8,3]
```

기본 `K=16`은 CPU rollout 비용을 제한하기 위한 시작점이며 최적 후보 수라는 주장은 아니다.
네 command는 left/straight/right/unknown으로 유지한다. 각 branch는 같은 K개 anchor bank를
사용하되 서로 다른 learned waypoint query로 residual을 예측한다. 처음에는 residual output을
0으로 초기화하므로 K개 후보가 서로 다른 실제 train trajectory에서 시작한다.

- BEV/detection/map: SSR `(x_right, y_forward)`.
- planning anchor, residual, 후보, PDM 입력: NAVSIM `(x_forward, y_left, heading)`.
- anchor: 현재 ego 기준 절대 future poses, 시각 0.5, 1.0, ..., 4.0 s. 원점 pose 미포함.
- archive metadata의 NAVSIM 좌표계·절대 pose 표현·현재 pose 미포함 convention이 다르거나
  누락되면 로드 단계에서 거부한다. nuScenes/SSR offsets를 anchor로 잘못 읽지 않도록 한다.
- regression: per-step offsets. heading 차이와 최종 heading은 원형 각도로 처리한다.
- BEV ROI와 detection FOV는 metric world의 범위를 자르지 않는다. 평가용 전체 map과 주변
  객체를 사용하며, 후보 trajectory도 32 m에서 clip하지 않는다.

SafeDrive의 bridge, pair-wise collision head, time-wise DAC head는 추가하지 않았다.

## 2. 학습 정답과 loss

WoTE에서 참조한 부분은 고정 anchor에 대한 imitation assignment와 score prediction이다.
SafeDrive에서 참조한 부분은 현재 모델이 refine한 실제 후보를 simulator로 채점하는 방식이다.

각 GT trajectory에 XY가 가장 가까운 **고정 anchor**를 지정하고 해당 후보의 offsets를 회귀한다.
동일 anchor index를 imitation classification 정답으로 사용한다. 후보가 이동할 때 assignment가
한 후보로 몰리는 것을 줄이기 위해 nearest-refined-candidate 방식은 사용하지 않는다.

Metric critic은 candidate pose encoding, 현재 ego status, shared BEV와 BEV position으로
NC/DAC/DDC/EP/TTC/comfort/aggregate-score의 logit 7개를 예측한다. Candidate pose encoding은
`detach()`하며, metric BCE가 trajectory coordinate generator로 흘러가지는 않는다. Critic의
BEV 입력은 기본적으로 gradient를 허용하고 `metric_detach_bev=true`로 차단할 수 있다.

학습과 Lightning validation의 `compute_loss()`에서만 token을 사용해 world cache를 읽고
현재 후보를 PDM simulator에 넣는다. Validation도 자신의 held-out world로 metric loss를
계산하지만 optimizer update는 하지 않는다. Model `forward(features)`와 public trajectory
inference에는 token, 미래 GT, world cache가 필요 없다.

사용하는 metric은 **이 checkout의 NAVSIM v1 PDM**이다. SafeDrive의 custom EPDMS/TLC/LK를
가져오지 않았으므로 현재 PARA-SSR의 공식 평가와 같은 정의로 학습 정답을 생성한다.

```text
L_plan = task_loss_weight.plan × (
    L_winner_reg
    + candidate_cls_loss_weight × L_candidate_CE
    + metric_loss_weight × Σ_j metric_loss_weights[j] × BCE_j
)
L_total = L_plan + L_det + L_motion + L_map
```

기본 `task_loss_weight.plan=2.0`, candidate CE/metric loss weight는 각각1.0이며 metric별
weight는 순서대로 `[3,3,1,2,4,1,1]`이다. 기존 plan2.0의 parity 근거는 **regression 항만**
해당한다. 새 CE/BCE가 추가된 전체 planning loss와 gradient 규모가 이전과 같다는 뜻은 아니다.
Winner regression은 K로 추가 평균하지 않으므로 기존 reg scale이 유지된다.
GradBalancer는 새 CE/BCE를 포함한 전체 planning gradient를 plan task로 측정한다.

`loss_plan_reg_weighted`, `loss_plan_cls_weighted`, `loss_plan_metric_weighted`의 합이
`loss_plan_total`이다. 추가로 metric별 BCE/GT 평균, score MAE, candidate oracle/selected ADE,
candidate spread와 rollout seconds를 기록한다. 이 값의 예시 초기값이나 스모크 결과를
학습된 성능으로 해석하면 안 된다.

## 3. 정확한 PDM label을 위한 두 가지 처리

공식 evaluator는 cached PDM reference와 최종 후보 **둘**을 같이 평가하고 progress를
정규화한다. 모든 K개 후보를 한 번에 scorer에 넣으면 다른 후보의 progress 때문에 같은
trajectory의 정답이 달라진다. 따라서 supervisor는 매 후보를 `[reference, candidate]` 쌍으로
독립 평가한다. 기존 `pdm_score_multi_trajs()`는 이 목적에 사용하지 않는다.

또한 `PDMScorerConfig` 클래스의 progress threshold 기본값은0.1 m지만 공식 평가 YAML은
5.0 m로 override한다. Supervisor는 공식 YAML과 같은5.0 m 및40 poses×0.1 s를 사용한다.
NC/DDC=0.5도 원래 soft label로 보존한다.

World metadata는 모든 CSV shard를 읽고, 이동된 cache 경로를 처리하며, train/validation
전체 token의 cache 유무를 본 학습 전에 검사한다. 누락/손상/non-finite 입력을 0점 정답으로
대체하지 않고 오류로 중단한다. Decompressed world는 process당8개까지 LRU로 보관한다.

## 4. 준비 및 실행

Repository root에서 해당 프로젝트 dependency 환경을 활성화한다. 아래 경로는 예시이며,
대용량 결과 디렉터리는 사용자가 선택한 storage에 연결할 수 있다.

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NUPLAN_MAPS_ROOT="$PWD/data/dataset/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export OPENSCENE_DATA_ROOT="$PWD/data/dataset"
export NAVSIM_DEVKIT_ROOT="$PWD"
export NAVSIM_EXP_ROOT="$PWD/work_dirs"
export PARA_SSR_PLAN_ANCHORS="$PWD/work_dirs/planning_anchors/train_k16.npz"
export PARA_SSR_METRIC_CACHE="$PWD/work_dirs/metric_cache_navtrain"

# Train logs만으로 vocabulary 생성. Navtrain token allowlist도 적용한다.
python tools/build_para_ssr_anchors.py \
  --data-root "$OPENSCENE_DATA_ROOT" \
  --output "$PARA_SSR_PLAN_ANCHORS" --num-candidates 16 --max-scenes 4096

# 학습과 내부 validation을 합친 navtrain world cache 생성.
# Navtest 평가 cache는 여기에 대체 사용할 수 없다.
python navsim/planning/script/run_metric_caching.py \
  scene_filter=navtrain split=trainval worker=sequential \
  cache.cache_path="$PARA_SSR_METRIC_CACHE"

# 기존 2-GPU/global128 recipe를 사용한다.
CUDA_VISIBLE_DEVICES=0,1 WANDB=0 \
  bash scripts/training/train_para_ssr_metric.sh
```

World cache 전체 생성은 별도 장시간 작업이다. 먼저 아래 smoke를 실행하면4개 실제 world만
만들어 구현을 검증할 수 있다. Anchor vocabulary의 seed, source token/log, hash는 NPZ metadata에
저장한다. `--max-scenes 256` 등 작은 vocabulary는 smoke용으로 사용할 수 있다.

```bash
python tools/smoke_para_ssr_metric.py \
  --anchor-path "$PARA_SSR_PLAN_ANCHORS" --device cpu

# 실제 본 모델 크기에서 batch1 확인; --device cuda도 지원한다.
python tools/smoke_para_ssr_metric.py \
  --anchor-path "$PARA_SSR_PLAN_ANCHORS" --device cpu --full-model
```

Smoke는 train2/validation2 scene을 사용하고 Lightning train2 optimizer steps + val2 batches,
finite gradients/parameter update, 공식 PDM 점수 일치, strict checkpoint 복원 및 target-free
public inference를 확인한다. `summary.json`, `config.yaml`, `smoke.ckpt`, metric world cache가
output directory에 남는다. Smoke checkpoint는 학습 완료 가중치가 아니다.

학습 완료 checkpoint의 PDM 평가:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluation/eval_para_ssr.sh /path/to/model.ckpt \
  agent=para_ssr_metric_agent \
  agent.config.plan_anchor_path='' agent.config.metric_cache_path='' \
  experiment_name=eval/para_ssr_front3_metric_k16
```

Anchor bank는 checkpoint buffer로 저장된다. Checkpoint로 agent를 생성하면 훈련 때의 anchor
파일을 다시 읽거나 backbone 사전학습 가중치를 다운로드하지 않는다.
K, BEV 크기 등 architecture override는 학습 당시 값과 맞춰야 한다.
기존 single-trajectory checkpoint를 새 metric architecture에 strict load할 수는 없다.

`train_para_ssr_metric.sh`는 **새 metric-supervised 학습**용 준비물 검사 wrapper다.
아래 imitation-only 실험이나 checkpoint resume은 기본 wrapper에 metric agent를 명시한다.

```bash
# 같은 K의 imitation-only: metric world cache 없이 학습.
EXPERIMENT=para_ssr_front3_k16_imitation WANDB=0 \
  bash scripts/training/train_para_ssr.sh agent=para_ssr_metric_agent \
  agent.config.metric_loss_weight=0 agent.config.metric_score_weight=0 \
  agent.config.metric_cache_path=''

# 완전한 Lightning training checkpoint로 optimizer/epoch까지 resume.
# smoke.ckpt는 이 용도가 아니다. 원래 학습의 architecture/K 설정을 유지한다.
# Metric-supervised resume에는 world cache가 계속 필요하지만 anchor 원본은 불필요하다.
RESUME_CHECKPOINT=/path/to/training.ckpt EXPERIMENT=para_ssr_front3_metric_k16 WANDB=0 \
  bash scripts/training/train_para_ssr.sh agent=para_ssr_metric_agent \
  agent.config.plan_anchor_path='' agent.config.backbone_pretrained=false
```

**Auxiliary mAP runner는 별도 미해결 항목이다.** 현재 `run_aux_evaluation.py`의 V2 protocol은
이전 detection 전후방 ±32 m ROI와 그 GT 수에 고정되어 있어 새 front-only checkpoint를
거부한다. 해당 runner의 ROI/FOV protocol과 GT reference를 별도 검증하여 이관하기 전에는
새 모델의 detection/map mAP 실행을 지원한다고 볼 수 없다. 위 평가 명령은 이 runner가
아닌 공식 **PDM trajectory 평가** 경로다.

## 5. 권장 비교 실험과 현재 한계

같은 입력, ROI, 학습 split, optimizer recipe에서 다음을 비교할 수 있다.

| 실험 | 설정 |
|---|---|
| 기존 single trajectory | `agent=para_ssr_agent` |
| K-candidate imitation-only | metric agent + `metric_loss_weight=0`, `metric_score_weight=0` |
| metric critic, BEV 차단 | metric agent + `metric_detach_bev=true` |
| metric critic, BEV 학습 | metric agent 기본값 |

Override는 `agent.config.<name>=...`로 준다. Imitation-only는 simulator labels/cache를 사용하지
않으며 같은 K로 candidate 개수 증가와 metric supervision의 효과를 구분한다. Metric BCE를
끄고 무학습 metric head로 ranking하는 조합은 config validation에서 거부한다.

Inference ranking은 `0.1 × log_softmax(imitation) + 1.0 × logsigmoid(predicted aggregate score)`다.
이 inference weight는 training BCE weight와 다른 역할이며 validation으로 정해야 한다.
Shared BEV detach 실험에서도 imitation loss는 기존 planner를 계속 학습한다.

CPU rollout을 microbatch마다 수행하므로 기존 single-trajectory throughput을 그대로 적용할 수
없다. 후보 커버리지, K/weight 최적값, 학습 안정성 및 최종 PDM 향상은 본 학습 실험이 필요하다.
검증은 실행·gradient 경로·평가 정의의 일치에 관한 것이며 장기 학습 품질 보장은 아니다.

관련 구현: `modules/candidate_planner.py`, `modules/planner_head.py`, `para_ssr_model.py`,
`para_ssr_loss.py`, `para_ssr_agent.py`, `metric_supervision.py`.

## 6. 실제 실행 검증 (2026-09-08)

- `ssr-navsim` 환경에서 전체 회귀 테스트 및 원본 loss parity 테스트를 실행했다.
  **160 passed**이며 실제 metric cache를 사용한 공식 PDM parity 테스트도 skip하지 않고
  포함했다.
- Train-only 256-scene 표본으로 K=16 smoke vocabulary를 생성했다. 본 학습용 vocabulary
  크기나 성능을 검증한 것은 아니다.
- 축소 모델과 본 모델 크기 각각 실제 train 2개/validation 2개 scene으로 Lightning
  optimizer 2회 + validation 2회를 통과했다.
- 본 모델: 39,070,262 parameters, BEV `100×100`, 3-camera, fp32, 전체 aux head 활성화.
  후보 `[1,16,8,3]`, metric label `[1,16,7]`; 모든 loss/gradient finite,
  metric critic·candidate classifier·refinement gradient와 실제 parameter update 확인.
- 공식 평가기와 refined candidate metric 최대 차이 `2.48e-8`.
- Strict checkpoint 재로딩과 anchor/cache를 제거한 public inference 통과.
- 현재 실행 중인 다른 학습을 방해하지 않도록 CPU를 사용했다. GPU/DDP, batch4 메모리,
  장기 수렴, 전체 navtest PDM 점수는 이번 스모크의 검증 범위가 아니다.

최종 코드로 재실행한 본 모델 결과는 repository의
`work_dirs/smoke_metric_k16_full_final_20260908/summary.json`, `config.yaml`, `smoke.ckpt`에
저장했다. 실행 시간은 world 4개 생성 21.88초, Lightning fit 전체 23.42초이며
**CPU batch1 smoke 시간이지 본 학습 throughput이 아니다**.
사용한 vocabulary는 `work_dirs/smoke_metric_k16_full_20260908/anchors_smoke_k16.npz`다.
