# Dual-Teacher Planning Distillation v3
## 셀 에너지 증류를 빼고, 궤적의 최악 지점을 계획 손실에 넣는다

**기준 문서**: [`TOP_TIER_PLANNING_DISTILLATION_PLAN_V2.md`](TOP_TIER_PLANNING_DISTILLATION_PLAN_V2.md) (v2 계약과 `version_2` 측정. 그 파일은 그대로 둔다)  
**소속/작업공간**: `/home/external-user/byounggun/SSR`  
**작성일**: 2026-09-23  
**측정일**: 2026-09-24 (`version_3` epoch 29, navtest PDMS **0.8301**)  
**코드 기본값**: 이 문서의 5절은 `version_3`를 학습한 값이다. 측정 뒤의 기본값은 [`TOP_TIER_PLANNING_DISTILLATION_PLAN_V4.md`](TOP_TIER_PLANNING_DISTILLATION_PLAN_V4.md). `version_4` epoch 29 navtest PDMS는 **0.8563**이고 기록은 그 문서 8절이다.

배포 그래프는 그대로다. `camera → BEVFormer → dense planner` (`use_stl=false`, `plan_num_layers=3`). 디코더를 후보 생성기로 바꾸지 않는다. Stage 1 어댑터도 다시 학습하지 않는다.

---

## 1. 측정이 가른 것

같은 12146장, epoch 29, distill 키를 벗긴 navtest.

| 런 | 한 일 | score | DAC 실패 |
|---|---|---:|---:|
| v1 `version_0` | 등방 복도 MSE | 0.8539 | 792 |
| 첫 v2 `version_1` | 좁은 복도, 보도 suppress, CWD, adaptive, map-head KD | 0.8433 | 891 |
| v2 `version_2` | 넓은 이방 복도, 역할 마스크, look prior | 0.8534 | 785 |

`version_2`는 기준점보다 **0.0006** 낮다. 첫 v2에서 빠졌던 0.010은 돌아왔다. DAC 실패는 7장 줄었지만 새로 나간 373장과 고친 380장이 점수를 서로 지운다. 둘 다 도로 안에 있는 장면의 EP는 0.8588에서 0.8563으로 조금 낮아졌다.

look prior는 실패한 손실이 아니다. KL이 0.284에서 0.069로 내려갔고 epoch 19 이후에는 거의 안 움직인다. 배운 뒤에도 PDMS가 기준점에 붙어 있다. epoch 29에 그 항은 0.035이고, feature MSE는 BEVFusion 1.28, ReSMap 0.78이다. 코사인은 0.37 / 0.61로 v1과 같은 자리에서 멈춘다.

남는 병목은 문서가 처음부터 적은 그대로다. NC 0.981, comfort 1.000, DDC 1.000은 천장이다. 점수를 움직이는 것은 DAC 꼬리 약 6.5%와, 이미 합법인 장면의 EP 0.856이다. NAVSIM v1은 DAC가 0이면 EP를 0으로 두고 그 항을 곱한다.

---

## 2. 왜 BEV 증류를 더 세게 하면 안 되나

플래너는 쿼리 하나가 `bev_embed` 100×100 전부와 cross-attention을 한다. 키는 256차원 벡터와 위치 임베딩이다 (`planner_head.py`의 dense 경로). look prior가 맞춘 것은 채널 절댓값의 평균이 어느 셀에 몰리는지다. 에너지가 같고 방향이 다르면 KL은 맞고 궤적은 다르다. 방향은 어댑터를 거친 MSE가 맞추는데, 그 코사인이 이미 v1의 천장에 있다.

첫 v2는 그 축을 더 세게 눌렀다. 횡방향 σ를 1.25 m로 좁히고, 보도를 누르고, CWD·relation·attention과 inverse-EMA를 올렸다. BEVFusion 코사인이 0.37에서 0.15로 떨어졌고 DAC 실패가 99장 늘었다. 같은 마스크를 다시 조이거나 look 가중만 올리는 것은 이미 포화된 축이거나, 한 번 점수를 깎은 축이다.

공식 PDM 폴리곤으로 매 스텝을 채점하는 길은 지금 데이터에 없다. `data/exp/metric_cache`는 navtest 12146 토큰이다. 학습 85,109장에는 없다. 후보 플래너(`use_metric_planner`)는 디코더를 바꾼다. 이번 개정은 그 스위치를 켜지 않는다.

맵 GT의 `road` 클래스는 차선+교차로 폴리곤의 **외곽선**이다. 내부를 채운 마스크가 아니다. 그 스플랫이 높은 셀을 “도로 안”으로 쓰면 차를 경계선 위로 당긴다. 그래서 도로 스플랫으로 DAC 손실을 만들지 않는다.

---

## 3. 빼는 것

| 항 | v2 `version_2` | v3 기본값 | 이유 |
|---|---:|---:|---|
| `distill_plan_look_weight` | 0.5 | **0** | KL은 수렴했고 PDMS는 안 움직였다. 코드 경로는 남긴다. |
| CWD / relation / attention | 0 | 0 | 첫 v2에서 좁은 마스크와 함께 점수를 깎았다. |
| adaptive branch | false | false | ReSMap을 키우고 BEVFusion을 죽였다. |
| head KD | 0 | 0 | det는 발산, map은 계획 공간이 아니다. |
| walkway suppress | 0 | 0 | 연석을 같이 눌렀다. |

feature MSE, 이방 복도(σ 4.0 / 2.5 m, 바닥 0.1), 역할 마스크, GradBalancer, GT 계획/검출/맵 가중, ResNet-50, Stage 1 어댑터는 유지한다. 이 조합이 0.8539를 붙들고 있다. `version_0`과 `version_2` 체크포인트는 지우지 않는다. `version_2`를 이어서 학습하지 않는다.

---

## 4. 넣는 것: 궤적 꼬리 손실

dense planner가 이미 내는 8개 웨이포인트에만 건다. 명령 분기 하나만 본다. 예측이 GT와 같으면 두 항 모두 0이다. 목표 궤적을 바꾸지 않고, 평균 L1이 묻어 버리는 한 스텝에 기울기를 더 준다.

계획 L1은 8스텝 × 좌표를 평균한다. 한 스텝이 1 m 옆으로 벗어나도 나머지는 맞으면 평균은 작다. 닫힌 루프 DAC는 그 한 스텝의 차체 모서리가 도로 밖이면 장면 전체를 0으로 만든다. 평균과 꼬리가 다른 이유다.

절대 좌표는 오프셋의 누적합이다. GT는 캐시의 `trajectory` (현재 자아 프레임, x 전방, y 왼쪽)를 쓴다.

### 4.1 최악 횡오차

\[
\mathcal{L}_{\text{lat}}=\operatorname{mean}_b\max_t\big(|y^{\text{pred}}_{b,t}-y^{\text{GT}}_{b,t}|\cdot m_{b,t}\big)
\]

`m`은 유효 스텝 마스크다. 최댓값이라 기울기는 가장 많이 벗어난 스텝으로 간다. 가중 `plan_tail_lat_weight=0.1`.

이것은 공식 DAC가 아니다. 차체 폴리곤도, nuPlan drivable polygon도 보지 않는다. GT보다 옆으로 더 나간 스텝을 평균에 묻지 않게 하는 항이다. GT 자체가 경계에 붙어 있으면 10 cm도 공식 채점에서는 실패할 수 있고, 이 항은 그 10 cm를 특별 취급하지 않는다. 그 한계를 적어 두는 이유다.

### 4.2 진행 거리 부족

원점부터의 폴리라인 길이만 본다.

\[
\mathcal{L}_{\text{prog}}=\operatorname{mean}_b\operatorname{relu}\big(\ell(\text{GT}_b)-\ell(\text{pred}_b)\big)
\]

GT보다 짧은 경로만 벌한다. 더 긴 경로는 0이다. 코너를 잘라 길이를 늘리라고 밀지 않는다. 길이를 늘리려고 도로 밖으로 도는 기울기도 없다. 옆으로 크게 나간 경우는 4.1이 따로 당긴다. 가중 `plan_tail_progress_weight=0.05`.

EP는 DAC가 깨지면 0이 되므로, 진행 항만 키우고 횡오차를 두면 점수가 내려갈 수 있다. 횡오차 가중을 진행 가중보다 크게 둔 이유다.

### 4.3 어디에 더해지나

`ParaSSRLoss`의 dense 경로에서 계획 L1에 더한 뒤 `task_loss_weight.plan=2.0`을 곱한다. GradBalancer는 그 합을 plan 기울기로 본다. 검출/맵 가중은 그대로다. 후보 플래너 경로는 이 항을 쓰지 않는다.

epoch 0 근처에서 최악 횡오차가 수 미터여도, 가중 0.1이면 총 손실(당시 약 19) 옆에서 보조 항이다. epoch 29의 계획 L1이 0.028인 운전점에서는, 이미 GT에 붙은 샘플에서 이 항이 거의 0이라 기존 MSE 증류를 밀어내지 않는다.

---

## 5. 코드 기본값

| 키 | 값 |
|---|---|
| `image_architecture` | `resnet50.tv_in1k` |
| `distill_plan_look_weight` | **0** |
| `distill_cwd_weight` / `relation` / `attn` | 0 |
| `distill_adaptive_branch` | false |
| `distill_head_kd_weight` | 0 |
| `distill_walkway_suppress` | 0 |
| `corridor_sigma_along` / `cross` | 4.0 / 2.5 |
| `corridor_base_weight` | 0.1 |
| `distill_use_role_masks` | true |
| `distill_mse_weight` | 1.0 |
| `plan_tail_lat_weight` | **0.1** |
| `plan_tail_progress_weight` | **0.05** |
| `use_stl` / `use_metric_planner` | false / false |
| `task_loss_weight.plan` | 2.0 |
| GradBalancer | plan 0.4 / det 0.3 / map 0.3 |
| W&B | 기본 꺼짐 |

Stage 2만 처음부터. Stage 1 어댑터는 `version_0` BEVFusion과 기존 ReSMap 체크포인트를 그대로 쓴다.

```bash
FORCE_RETRAIN=stage2 ONLY_STAGE=stage2 bash ./scripts/training/run_all_stages_distill.sh
```

끝나면 distill 키를 벗기고 GPU 4, 5에서 navtest. 스냅샷 디렉터리는 `version_2` 것과 겹치지 않게 새로 둔다. 비교 기준은 `version_0`의 0.8539이고, 직전 측정은 `version_2`의 0.8534다.

---

## 6. 맞으면 어떤 숫자이고, 아니면 어떤 숫자인가

맞다에 해당하는 결과:

- score가 0.8539보다 높다.
- DAC 실패가 785장보다 줄고, 그 감소가 서로 맞바꾼 장면의 상쇄가 아니다. 새로 실패한 장수보다 고친 장수가 분명히 많다.
- DAC를 둘 다 통과한 장면의 EP가 0.856 아래로 떨어지지 않는다.

아니다에 해당하는 결과:

- score가 0.8534 근처에서 다시 멈춘다. 그러면 꼬리 손실도 평균 L1과 같은 개루프 목표라 닫힌 루프 꼬리를 못 본 것이다. 가중을 더 올리는 재학습은 하지 않는다.
- DAC 실패가 늘거나, 합법 장면의 EP가 떨어진다. 진행 항이 횡오차를 이긴 신호다. 그 체크포인트는 기준점으로 쓰지 않는다.

학습 중에 볼 것: `train/loss_plan_tail_lat`, `train/loss_plan_tail_progress`. look 항은 로그에 없어야 한다. BEVFusion/ReSMap 코사인이 0.37 / 0.61에서 크게 무너지면 꼬리 항이 인코더를 계획 쪽으로만 끌어간 것이니, 그 런은 점수가 올라도 검출이 망가졌는지 따로 본다. 코사인은 메트릭일 뿐이고 기본 가중은 바꾸지 않았다.

---

## 7. 측정 (2026-09-24): 이 계약은 점수를 깎았다

`version_3`는 이 문서대로 학습됐다. TensorBoard에 `loss_plan_tail_lat`, `loss_plan_tail_progress`가 있고 `loss_distill_plan_look`은 없다. 체크포인트 `work_dirs/paradrive_distill_stage2_dual_distill/lightning_logs/version_3/checkpoints/epoch=29-step=19950.ckpt`. navtest 12146장, GPU 4+5, distill 키를 벗긴 병합 점수. 병합 CSV `work_dirs/eval/paradrive_distill_stage2_version3_epoch29/merged.csv`.

| 항 | v1 `version_0` | v2 `version_2` | v3 `version_3` |
|---|---:|---:|---:|
| valid | 1.0000 | 1.0000 | 1.0000 |
| no_at_fault_collisions | 0.9815 | 0.9806 | **0.9664** |
| drivable_area_compliance | 0.9348 | 0.9354 | **0.9231** |
| driving_direction_compliance | 1.0000 | 1.0000 | 1.0000 |
| ego_progress | 0.8022 | 0.8003 | 0.8017 |
| time_to_collision_within_bound | 0.9395 | 0.9413 | **0.9015** |
| comfort | 0.9998 | 0.9999 | 0.9995 |
| **score** | **0.8539** | **0.8534** | **0.8301** |

실패 장수 (평균 행 제외, 12146):

| | v1 | v2 | v3 |
|---|---:|---:|---:|
| NC < 1 | 239 | 247 | 436 |
| 그중 NC = 0 | 211 | 225 | 381 |
| 그중 NC = 0.5 | 28 | 22 | 55 |
| DAC = 0 | 792 | 785 | 934 |
| TTC = 0 | 735 | 713 | 1196 |
| score = 0 | 983 | 999 | 1295 |

6절의 “맞다”에 해당하는 숫자는 하나도 없다. score는 0.8539보다 0.0238 낮고, DAC 실패는 785보다 149장 늘었다. 합법 장면의 EP는 떨어지지 않았다. `version_3` 혼자 도로 안에 있고 충돌이 없는 10825장의 EP는 0.898이다. v2의 같은 조건 11132장은 0.872였다. 진행은 올랐고, 그 대가로 NC와 TTC와 DAC가 같이 무너졌다. 이 체크포인트는 기준점으로 쓰지 않는다. `version_2`를 이어서 학습하지 않고, `version_3`도 이어서 학습하지 않는다.

장면별 원인과 다음 계약은 [`TOP_TIER_PLANNING_DISTILLATION_PLAN_V4.md`](TOP_TIER_PLANNING_DISTILLATION_PLAN_V4.md).
