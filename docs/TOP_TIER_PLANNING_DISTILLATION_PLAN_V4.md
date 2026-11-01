# Dual-Teacher Planning Distillation v4
## 최악 횡오차는 빼고, 진행은 인간 길이의 90%에서 멈춘다

**기준 문서**: [`TOP_TIER_PLANNING_DISTILLATION_PLAN_V3.md`](TOP_TIER_PLANNING_DISTILLATION_PLAN_V3.md) (`version_3` 측정. 그 파일은 그대로 둔다)  
**소속/작업공간**: `/home/external-user/byounggun/SSR`  
**작성일**: 2026-09-24  
**측정일**: 2026-09-26 (`version_4` epoch 29, navtest PDMS **0.8563**)  
**코드 기본값**: `ParaSSRConfig`, `para_ssr_agent.yaml`. `para_ssr_distill_agent.yaml`은 그 계획 가중을 상속한다.

배포 그래프는 그대로다. `camera → BEVFormer → dense planner` (`use_stl=false`, `plan_num_layers=3`). 디코더를 후보 생성기로 바꾸지 않는다. Stage 1 어댑터도 다시 학습하지 않는다. `version_3`를 이어서 학습하지 않는다. 측정은 8절이다. epoch 29 navtest score는 **0.8563**이다.

---

## 1. version_3가 낸 점수

tmux 39에서 끝난 Stage-2 `version_3`, epoch 29. navtest 12146장, 유효 12146, 실패 0. GPU 4가 shard 0, GPU 5가 shard 1. distill 키를 벗긴 학생으로 채점했다.

병합 CSV: `work_dirs/eval/paradrive_distill_stage2_version3_epoch29/merged.csv`

샤드:

- `work_dirs/eval/paradrive_distill_stage2_version3_epoch29_shard0of2/2026.09.24.20.32.04.csv`
- `work_dirs/eval/paradrive_distill_stage2_version3_epoch29_shard1of2/2026.09.24.20.33.15.csv`

체크포인트: `work_dirs/paradrive_distill_stage2_dual_distill/lightning_logs/version_3/checkpoints/epoch=29-step=19950.ckpt`

| 항 | v1 `version_0` | v2 `version_2` | v3 `version_3` |
|---|---:|---:|---:|
| valid | 1.0000 | 1.0000 | 1.0000 |
| no_at_fault_collisions | 0.9815 | 0.9806 | 0.9664 |
| drivable_area_compliance | 0.9348 | 0.9354 | 0.9231 |
| driving_direction_compliance | 1.0000 | 1.0000 | 1.0000 |
| ego_progress | 0.8022 | 0.8003 | 0.8017 |
| time_to_collision_within_bound | 0.9395 | 0.9413 | 0.9015 |
| comfort | 0.9998 | 0.9999 | 0.9995 |
| **score** | **0.8539** | **0.8534** | **0.8301** |

v3 계약은 그대로 실행됐다. `train/loss_plan_tail_lat`와 `train/loss_plan_tail_progress`가 있고, look 태그는 없다. score는 기준점 0.8539보다 **0.0238**, 직전 0.8534보다 **0.0233** 낮다. DDC는 1이고 comfort 실패는 6장이다. 점수를 옮긴 항은 NC, DAC, TTC다.

장면 점수 식은 장면마다

\[
\mathrm{NC}\times\mathrm{DAC}\times\mathrm{DDC}\times\frac{5\,\mathrm{EP}+5\,\mathrm{TTC}+2\,\mathrm{comfort}}{12}
\]

이다. NC, DAC, DDC 중 하나라도 0이면 그 장면의 EP도 0으로 기록되고 score도 0이다. 평균끼리 곱하면 0.8301이 나오지 않는다. 아래 기여는 장면 점수의 차이다.

---

## 2. 어디서 0.023이 빠졌나

v2 대비 v3. 같은 12146 토큰. “신규”는 v2에서는 통과하고 v3에서 실패한 장면이다.

| 실패 | v2 | v3 | 신규 | 고친 장면 | 순증 |
|---|---:|---:|---:|---:|---:|
| NC < 1 | 247 | 436 | 277 | 88 | +189 |
| NC = 0 (차량·보행자) | 225 | 381 | 240 | 84 | +156 |
| NC = 0.5 (그 외 물체) | 22 | 55 | | | |
| DAC = 0 | 785 | 934 | 575 | 426 | +149 |
| TTC = 0 | 713 | 1196 | 684 | 201 | +483 |

신규 NC 277장 중 240장은 점수 0인 전방·정지 물체 충돌이고 37장은 0.5다. 정적 긁힘이 주원인이 아니다.

v2→v3 평균 score −0.0233을 서로 겹치지 않게 나누면 다음과 같다. 기여는 (그 장면들의 score 합의 차) / 12146 이다.

| 묶음 | 장수 | 평균 score 기여 | 그 묶음의 EP | 그 묶음의 TTC |
|---|---:|---:|---|---|
| 새로 NC < 1 | 277 | −0.0167 | 0.880 → 0.028 | 0.574 → 0.007 |
| NC를 고침 | 88 | +0.0059 | 0.021 → 0.873 | 0.000 → 0.761 |
| NC는 1인데 DAC가 새로 0 | 550 | −0.0420 | 0.856 → 0.000 | 0.971 → 0.915 |
| NC는 1인데 DAC를 고침 | 416 | +0.0315 | 0.000 → 0.845 | 0.974 → 0.966 |
| 둘 다 합법, TTC가 새로 0 | 470 | −0.0155 | 0.926 → 0.966 | 1 → 0 |
| 둘 다 합법, TTC를 고침 | 118 | +0.0041 | 0.931 → 0.940 | 0 → 1 |
| 둘 다 합법이고 TTC도 1 | 9552 | **+0.0093** | 0.866 → 0.894 | 1 → 1 |
| 둘 다 합법, TTC는 원래 0 | 190 | +0.0001 | 0.955 → 0.967 | 0 → 0 |

세 구멍이 각각 약 0.01이다.

- NC 순기여 −0.0108. 고친 88장(+0.0059)이 새 충돌 277장(−0.0167)을 못 메운다.
- DAC 순기여 −0.0105. 고친 416장(+0.0315)과 새로 나간 550장(−0.0420)이 크게 맞바뀌고, 맞바뀜의 차이가 남는다.
- 합법인 채 TTC만 깨진 순기여 −0.0114. 470장의 EP는 오히려 0.926에서 0.966으로 올랐다.

마지막 줄이 꼬리 손실이 한 일의 직접 증거다. 도로 안에 있고 충돌도 없는 장면이 **더 멀리 갔고**, 그 속도의 1초 전방 투영이 에이전트에 닿아 TTC가 0이 됐다. TTC가 0이 되면 가중 평균에서 5/12가 빠진다. EP가 같이 올라도 장면 score는 약 0.40 떨어진다. 이 470장만으로 전체 평균이 0.0155 깎였다.

반대로, 합법과 TTC를 둘 다 지킨 9552장에서는 EP가 0.866에서 0.894로 올랐고 전체 평균이 0.0093 올랐다. 꼬리 손실은 진행을 움직이는 손실이 맞다. 그 진행이 TTC·NC·DAC 경계를 넘어서 순기여가 음수다.

둘 다 합법인 장면만 모아도 같다. 10330장에서 EP는 0.871 → 0.900, TTC는 0.970 → 0.936, 그 교집합의 score는 0.934 → 0.931이다. 진행이 0.029 올라도 TTC가 더 빨리 빠져 교집합 score는 내려간다.

v0 대비도 같은 방향이다. 평균 score −0.0239. 신규 NC 274 (−0.0167) , 신규 DAC 521 (−0.0397) , 합법 신규 TTC 449 (−0.0149), 둘 다 깨끗한 9601장은 +0.0093.

---

## 3. 학습 신호가 그 점수와 맞다

epoch 29. 계획 L1은 `loss_plan_reg` (모드 4 × 8 × 3 으로 나눈 평균)이다. `loss_plan_total`은 여기에 꼬리를 더하고 `task_loss_weight.plan=2`를 곱한 값이다.

| 신호 | v2 | v3 |
|---|---:|---:|
| 계획 L1 | 0.0140 | 0.0191 |
| 횡꼬리 원값 / 가중 0.1 | — | 0.231 / **0.0231** |
| 진행꼬리 원값 / 가중 0.05 | — | 0.072 / 0.0036 |
| `loss_plan_total` | 0.0280 | **0.0916** |
| `gnorm/plan` | 0.00456 | **0.0191 (4.2×)** |
| `gscale/det` | 0.056 | 0.233 |
| `gscale/map` | 0.104 | 0.512 |
| BEVFusion cosine | 0.374 | 0.338 |
| ReSMap cosine | 0.611 | 0.557 |
| distill MSE bf / rm | 1.276 / 0.779 | 1.348 / 0.888 |
| look | 0.035 | 없음 |
| `traj_loss` | 8.85 | 8.59 |

횡꼬리 가중이 계획 L1보다 크다. 0.0231 대 0.0191. epoch 1부터 그랬고 epoch 29까지 유지됐다. 계획 L1 자체는 v2의 0.014보다 나빠졌다. 꼬리가 평균 L1을 도와 준 것이 아니다. 평균을 밀어냈다.

기울기가 이렇게 커진 이유는 스칼라를 총손실 19와 비교해서 고른 가중 0.1이, 실제 기울기와 다르기 때문이다.

- 계획 L1의 한 좌표 기울기는 \(1/(B \times 4 \times 8 \times 3) = 1/(96B)\) 이다. 명령이 아닌 3개 분기는 분모에만 들어간다.
- 횡꼬리는 배치 평균이라 최악 스텝의 누적 \(y\) 기울기가 \(0.1/B\) 이다. 계획 L1 한 좌표의 약 **9.6배**이고, 누적합이라 그 앞 스텝의 횡오프셋까지 같이 꺾는다.
- 진행 항은 짧을 때 매 스텝 길이를 늘리는 기울기다. 가중 0.05면 한 스텝이 계획 L1 한 좌표의 약 **4.8배**다. epoch 29의 스칼라는 0.0036으로 작아 보이지만, 힌지가 닫히기 전에는 크기가 아니라 방향이 남는다.

`gnorm/plan`은 epoch 4부터 29까지 v2의 약 4배였다. GradBalancer는 몫을 0.40 / 0.30 / 0.30으로 유지하려고 det·map 밸브를 4배 열었다. 증류 MSE는 이 루프 밖이라 같이 커지지 않았다. 코사인은 v1·v2가 머물던 0.37 / 0.61에 도달하지 못하고 0.338 / 0.557에서 멈췄다. 동적 교사 정합이 나빠진 상태에서 궤적만 더 세게 당긴 것이다. v3 문서가 예고한 “코사인이 크게 무너지면 인코더가 계획 쪽으로 끌려간 것”이 이 숫자다. 첫 v2의 0.15까지 무너지진 않았다.

look를 뺀 것은 이 하락의 원인이 아니다. look가 없던 v1이 0.8539다. v2에서 look의 가중 손실은 0.035였고 score는 0.0005 안에서 붙었다.

---

## 4. 왜 그 두 항이 그 구멍을 냈나

**진행 항**은 GT 폴리라인보다 짧으면 길이를 늘린다. 닫힌 루프 TTC는 현재 속도로 앞 범퍼를 최대 1초 투영해 에이전트와 겹치면 0이다. 로그 재생 에이전트는 양보하지 않는다. 열린 루프에서 인간 길이까지 당기면, 이미 EP 0.91이던 장면이 0.96까지 가고 그 1초 안에 들어온다. 479장은 v3에서도 NC=1, DAC=1 인데 TTC만 새로 깨졌고 EP는 0.909 → 0.963 이다.

**횡 항**은 공식 DAC가 아니다. 차체 폴리곤을 보지 않고, GT의 최악 \(|y|\) 한 점으로 기울기가 모인다. GT가 경계에 붙어 있으면 그 점을 쫓다 모서리가 도로 밖으로 나간다. 누적 \(y\)를 고치려고 앞 스텝이 꺾이면 추적 제어가 그 꺽임을 더 나간다. DAC 실패 785 → 934 가 그 결과다. v3 문서의 성공 조건 “DAC 실패가 785보다 줄고, 맞바뀜이 아닐 것”은 실패했다. 신규 575, 회복 426 이다.

깨끗한 9552장에서 전체 평균이 0.0093 오른 것은 남겨 둘 신호가 있다는 뜻이다. 그 장면의 EP 평균은 0.866이었다. TTC가 깨진 합법 장면은 이미 0.91이었고 0.96까지 밀리다 떨어졌다. 인간 길이의 **마지막 10%를 당기지 않으면** 짧은 경로는 아직 올라오고, TTC로 넘어진 구간은 보상이 0이다. 이 구분은 폴리라인 길이 비율이지 EP 임계값이 아니다. EP와 길이는 같은 수가 아니다. 그래서 캡을 공식 채점으로 착각하지 않는다.

횡 항을 가중만 줄여 다시 켜는 실험은 하지 않는다. 최악 한 스텝으로 모이는 구조가 DAC를 늘렸고, 작게 줄여도 그 구조는 남는다. v3가 이미 “가중을 더 올리는 재학습은 하지 않는다”고 적어 두었고, 이번 측정은 그 항을 내리는 쪽도 같은 가족으로 본다.

---

## 5. 빼는 것, 바꾸는 것

| 항 | v3 `version_3` | v4 기본값 | 이유 |
|---|---:|---:|---|
| `plan_tail_lat_weight` | 0.1 | **0** | 계획 L1보다 큰 기울기, DAC +149, 코사인 이탈 |
| `plan_tail_progress_weight` | 0.05 | **0.003** | 길이 기울기를 L1 한 좌표의 약 0.3배로 |
| `plan_tail_progress_cap` | 1 (캡 없음) | **0.9** | GT 길이의 90%에 도달하면 진행 손실 0 |
| `distill_plan_look_weight` | 0 | 0 | 이미 포화 |
| CWD / relation / attention / adaptive / head KD / walkway suppress | 0 | 0 | 첫 v2에서 점수를 깎음 |
| 복도 σ along / cross, 바닥 | 4.0 / 2.5, 0.1 | 유지 | v2가 0.8534를 붙든 마스크 |
| GradBalancer 측정 | `plan_total` (꼬리 포함) | **계획 L1만** | 꼬리가 det/map 밸브를 열지 못하게 |

진행 손실은 이렇게 바뀐다. \(\ell\) 은 원점부터의 폴리라인 길이이고, \(c=0.9\) 이다.

\[
\mathcal{L}_{\text{prog}}=\operatorname{mean}_b\operatorname{relu}\big(c\cdot\ell(\text{GT}_b)-\ell(\text{pred}_b)\big)
\]

예측이 GT와 같으면 0이다. GT보다 길어도 0이다. \(c\cdot\ell(\text{GT})\) 보다 길면 0이다. 그 길이보다 짧은 경로만 벌한다. 코너를 잘라 100%까지 맞추라고 밀지 않는다.

가중 0.003은 스칼라가 아니라 기울기로 고른 값이다. 길이 힌지의 한 스텝 기울기는 가중 × 96 만큼 계획 L1 한 좌표와 비교된다. 0.05 × 96 = 4.8 이었고, 0.003 × 96 = 0.29 이다. 여덟 스텝에 같은 방향(더 길게)으로 걸리므로 짧은 장면의 속도에는 아직 편향이 있다. v3처럼 계획 기울기 노름을 4배로 만들 크기는 아니다.

GradBalancer가 보는 계획 손실은 명령 분기 L1이다. 진행 항은 총손실에는 들어가 인코더와 플래너로 흐르고, det/map 밸브를 키우는 측정에는 들어가지 않는다. 후보 플래너(`use_metric_planner`)는 이 경로를 타지 않으며, 그 분기에서는 지금처럼 cls·metric까지 묶어 측정한다. 이번 학습은 `use_metric_planner=false` 이다.

복도, 역할 마스크, feature MSE, GradBalancer 목표 몫 0.4 / 0.3 / 0.3, GT 검출·맵 가중, ResNet-50, Stage 1 어댑터는 유지한다. `version_0`과 `version_2` 체크포인트는 지우지 않는다.

---

## 6. 코드 기본값

| 키 | 값 |
|---|---|
| `image_architecture` | `resnet50.tv_in1k` |
| `distill_plan_look_weight` | 0 |
| `distill_cwd_weight` / `relation` / `attn` | 0 |
| `distill_adaptive_branch` | false |
| `distill_head_kd_weight` | 0 |
| `distill_walkway_suppress` | 0 |
| `corridor_sigma_along` / `cross` | 4.0 / 2.5 |
| `corridor_base_weight` | 0.1 |
| `distill_use_role_masks` | true |
| `distill_mse_weight` | 1.0 |
| `plan_tail_lat_weight` | **0** |
| `plan_tail_progress_weight` | **0.003** |
| `plan_tail_progress_cap` | **0.9** |
| `use_stl` / `use_metric_planner` | false / false |
| `task_loss_weight.plan` | 2.0 |
| GradBalancer | plan 0.4 / det 0.3 / map 0.3. 측정은 계획 L1 |
| W&B | 기본 꺼짐 |

Stage 2만 처음부터. Stage 1 어댑터는 기존 BEVFusion `version_0`, ReSMap 체크포인트를 그대로 쓴다. Lightning이 `version_4`를 새로 만든다.

```bash
cd /home/external-user/byounggun/SSR
FORCE_RETRAIN=stage2 ONLY_STAGE=stage2 bash ./scripts/training/run_all_stages_distill.sh
```

GPU 기본값은 4, 5다. 학습이 끝나면 distill 키를 벗기고 같은 GPU에서 navtest. 스냅샷 이름은 `version_3`와 겹치지 않게 둔다.

```bash
cd /home/external-user/byounggun/SSR
CKPT_SRC=/home/external-user/byounggun/SSR/work_dirs/paradrive_distill_stage2_dual_distill/lightning_logs/version_4/checkpoints/epoch=29-step=19950.ckpt \
SNAPSHOT_DIR=/home/external-user/byounggun/SSR/work_dirs/eval_snapshots/paradrive_distill_stage2_version4_epoch29 \
EXPERIMENT_NAME=eval/paradrive_distill_stage2_version4_epoch29 \
GPU0=4 GPU1=5 \
bash scripts/evaluation/eval_para_ssr_distill_epoch29_gpus45.sh
```

비교 기준은 `version_0`의 0.8539, 직전 성공에 가까운 측정은 `version_2`의 0.8534, 이번 실패한 측정은 `version_3`의 0.8301이다.

학습 중에 볼 것:

- `train/loss_plan_tail_lat`는 남아 있어도 가중이 0이라 총손실에 더해지지 않는다. 원값이 로그에 보여도 된다.
- `train/loss_plan_tail_progress`는 0.9 캡 대비 부족 길이(미터)다. epoch 29에 가중을 곱한 값이 계획 L1 0.02 안팎을 넘으면 캡이 안 닫힌 것이다.
- `gnorm/plan`이 v2의 0.0046에서 2배를 넘게 머물면 진행 항이 다시 플래너를 잡아먹은 것이다.
- BEVFusion / ReSMap 코사인이 0.36 / 0.59 아래로 머물면 증류 운전점을 잃은 것이다. 그 체크포인트는 점수가 올라도 쓰지 않는다.

---

## 7. 맞으면 어떤 숫자이고, 아니면 어떤 숫자인가

이 런의 기대는 v3의 깨끗한 장면이 만들던 +0.009를 그대로 회수하는 것이 아니다. 그 +0.009는 TTC를 깨뜨린 같은 밀기에서 나왔다. 캡과 0.003 가중은 그 밀기의 마지막 구간을 잘라 낸 것이다. 올라간다면 천 단위 소수에서, 잘리면 0.853 근처에 다시 붙는다.

맞다에 해당하는 결과:

- score가 0.8539보다 높다.
- TTC 실패가 713장(v2) 이하이고, 735장(v0)을 넘지 않는다.
- DAC 실패가 785장 이하이다. 신규보다 고친 장수가 많다.
- epoch 29 코사인이 BEVFusion 0.36 이상, ReSMap 0.59 이상이다.
- `gnorm/plan`이 0.009를 넘지 않는다. v3의 0.019는 실패한 운전점이다.
- 합법 장면 EP가 0.872 아래로 떨어지지 않는다.

아니다에 해당하는 결과:

- score가 0.8534 이하에서 멈춘다. 길이 캡도 열린 루프 모방이라 닫힌 루프 여유를 못 만든 것이다. 가중을 0.003 위로 올리지 않는다. 횡꼬리를 다시 켜지 않는다.
- TTC 실패가 800장을 넘거나 DAC 실패가 850장을 넘는다. 길이 항 자체가 PDMS에 반대다. 그 체크포인트는 기준점으로 쓰지 않고, 다음 계약은 `plan_tail_progress_weight=0` 으로 이 계열을 끝낸다.
- 코사인이 0.35 아래로 떨어진다. 밸브 분리로도 증류가 밀린 것이다. 그 체크포인트를 이어서 학습하지 않는다.

---

## 8. 측정 (2026-09-26): score 0.8563

`version_4`는 이 문서의 기본값으로 Stage 2를 처음부터 학습했다. 체크포인트 `work_dirs/paradrive_distill_stage2_dual_distill/lightning_logs/version_4/checkpoints/epoch=29-step=19950.ckpt` (2026-09-26 20:54). navtest 12146장, 유효 12146, 실패 0. GPU 4가 shard 0, GPU 5가 shard 1. distill 키를 벗긴 학생으로 채점했다.

병합 CSV: `work_dirs/eval/paradrive_distill_stage2_version4_epoch29/merged.csv`

샤드:

- `work_dirs/eval/paradrive_distill_stage2_version4_epoch29_shard0of2/2026.09.26.21.23.19.csv`
- `work_dirs/eval/paradrive_distill_stage2_version4_epoch29_shard1of2/2026.09.26.21.24.52.csv`

| 항 | v1 `version_0` | v2 `version_2` | v3 `version_3` | v4 `version_4` |
|---|---:|---:|---:|---:|
| valid | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| no_at_fault_collisions | 0.9815 | 0.9806 | 0.9664 | **0.9810** |
| drivable_area_compliance | 0.9348 | 0.9354 | 0.9231 | **0.9371** |
| driving_direction_compliance | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ego_progress | 0.8022 | 0.8003 | 0.8017 | 0.8020 |
| time_to_collision_within_bound | 0.9395 | 0.9413 | 0.9015 | **0.9439** |
| comfort | 0.9998 | 0.9999 | 0.9995 | 0.9999 |
| **score** | **0.8539** | **0.8534** | **0.8301** | **0.8563** |

score는 기준점 0.8539보다 **0.0024**, `version_2`의 0.8534보다 **0.0029**, `version_3`의 0.8301보다 **0.0262** 높다. DDC는 1이고 comfort 실패는 1장이다. 평균을 올린 항은 DAC와 TTC다. 전역 EP는 0.8020으로 v1의 0.8022 옆에 있다.

실패 장수 (평균 행 제외, 12146):

| | v1 | v2 | v3 | v4 |
|---|---:|---:|---:|---:|
| NC < 1 | 239 | 247 | 436 | 246 |
| 그중 NC = 0 | 211 | 225 | 381 | 216 |
| 그중 NC = 0.5 | 28 | 22 | 55 | 30 |
| DAC = 0 | 792 | 785 | 934 | **764** |
| TTC = 0 | 735 | 713 | 1196 | **681** |
| score = 0 | 983 | 999 | 1295 | 969 |

7절의 판정:

- score 0.8563은 0.8539보다 높다.
- TTC 실패 681은 v2의 713 이하이고, v0의 735 이하이다.
- DAC 실패 764는 785 이하이다. v2 대비 신규 333, 고친 장면 354이다. v0 대비는 신규 358, 고친 장면 386이다.
- epoch 29 코사인은 BEVFusion 0.378, ReSMap 0.607이다.
- `gnorm/plan`은 0.00488이다.
- 합법 장면(NC=1, DAC=1) 11158장의 EP는 0.8719다. v2의 같은 조건 11132장은 0.8724이고, 7절에 적힌 선은 0.872다. 이 항목만 그 선보다 0.0005 낮다.

중단 조건에는 들어가지 않는다. score는 0.8534 위이고, TTC 실패는 800 아래, DAC 실패는 850 아래, 코사인은 0.35 위다. `plan_tail_progress_weight=0`으로 이 계열을 끝내는 분기는 열리지 않는다.

v2 대비 평균 +0.0029를 장면 점수로 나누면 다음과 같다. 기여는 (그 장면들의 score 합의 차) / 12146 이다.

| 묶음 | 장수 | 평균 score 기여 | 그 묶음의 EP |
|---|---:|---:|---|
| 새로 NC < 1 | 88 | −0.0050 | 0.848 → 0.061 |
| NC를 고침 | 89 | +0.0052 | 0.036 → 0.886 |
| NC는 1인데 DAC가 새로 0 | 327 | −0.0250 | 0.849 → 0.000 |
| NC는 1인데 DAC를 고침 | 349 | +0.0265 | 0.000 → 0.840 |
| 둘 다 합법, TTC가 새로 0 | 120 | −0.0040 | 0.926 → 0.944 |
| 둘 다 합법, TTC를 고침 | 160 | +0.0054 | 0.944 → 0.935 |
| 둘 다 합법이고 TTC도 1 | 10218 | −0.0001 | 0.869 → 0.869 |

세 순기여로 모으면 NC +0.0002, DAC +0.0015, 합법인 채의 TTC +0.0014다. 합법이고 TTC도 1인 10218장의 EP는 0.869에 머문다. v3의 깨끗한 9552장이 만들던 +0.0093은 이 런에 없다. 캡과 가중 0.003은 그 밀기를 재현하지 않았고, 올라간 0.0029는 실패 장면이 조금 더 많이 돌아온 차이다.

v0 대비 평균은 +0.0024다. NC 신규 113 / 회복 106, DAC 신규 358 / 회복 386, TTC 신규 234 / 회복 288. 합법이고 TTC도 1인 10140장의 EP는 0.869에서 0.868이다.

epoch 29 학습 신호. 계획 L1은 `loss_plan_reg`다.

| 신호 | v2 | v4 |
|---|---:|---:|
| 계획 L1 | 0.0140 | 0.0139 |
| 횡꼬리 원값 (가중 0) | — | 0.310 |
| 진행꼬리 원값 / 가중 0.003 | — | 0.0588 / **0.00018** |
| `loss_plan_total` | 0.0280 | 0.0282 |
| `gnorm/plan` | 0.00456 | 0.00488 |
| `gscale/det`, `gscale/map` | 0.056, 0.104 | 0.061, 0.121 |
| BEVFusion / ReSMap cosine | 0.374 / 0.611 | 0.378 / 0.607 |
| distill MSE bf / rm | 1.276 / 0.779 | 1.266 / 0.788 |
| `traj_loss` | 8.85 | 8.77 |

6절에서 보기로 한 것과 맞다. 횡꼬리 원값은 로그에 있고 총손실에는 더해지지 않는다. `loss_plan_total` 0.02815는 (계획 L1 0.01390 + 0.003 × 0.05875) × 2 와 같다. 진행 가중이 계획 L1 0.02를 넘지 않는다. `gnorm/plan`은 v2의 0.00456 옆이고 0.009를 넘지 않는다. 코사인은 0.36 / 0.59 위에 있다.

`version_0`과 `version_2` 체크포인트는 그대로 둔다. 지금까지 측정한 epoch 29 navtest score 중 가장 높은 값은 `version_4`의 0.8563이다. `version_3`는 기준점으로 쓰지 않는다. 그 측정은 100×100 student다. 다음 학습은 9절이다. v5 범퍼 손실은 쓰지 않는다.

---

## 9. 다음 학습: 같은 전방 박스를 50×100으로

8절의 0.8563은 전방 32 m를 100칸(0.32 m)으로 나눈 student다. 보는 박스는 그대로 좌우 `[-32, 32]` m, 전방 `[0, 32]` m이다. 칸만 interaction 브랜치와 teacher 캐시에 맞춘다. 전방 50칸, 좌우 100칸, 둘 다 0.64 m.

손실은 이 문서의 v4다. 횡꼬리 0, 진행 0.003, 캡 0.9. 범퍼 항 `plan_ttc_proxy_weight`는 0이다. `version_4` 가중치는 이어 받지 않는다. 플래너가 바뀌어 있다. BEV cross-attention 뒤에 det/motion과 map 디코더 상태를 같은 residual에서 병렬로 보고 더한다. 18차원 ego motion은 BEV 쿼리에 더하지 않고, 명령과 `[vx, vy, ax, ay]`는 플래너에 들어간다. 과거 BEV는 temporal attention에 넣기 전에 이동과 상대 yaw로 한 번 `grid_sample`한다. yaw와 이동량은 detach되고, feature gradient만 남는다. 그 뒤 reference point에 이동을 다시 더하지 않는다.

캐시. 범위는 teacher도 전방 32 m, 좌우 64 m이다.

| teacher | 읽는 것 | 로드 후 격자 | student 50×100과의 관계 |
|---|---|---|---|
| BEVFusion | `cache_{train,val}_50x100`. 체크포인트 `runs/navsim-fusion-50x100/epoch_20.pth`. 칸 0.64 m | `(C, 50, 100)`, 축은 전방·좌. 폭을 뒤집어 SSR의 오른쪽이 된다 | 크기가 같다. 세로 50→100 리샘플은 하지 않는다 |
| BEVFusion 100×100 | `cache_*_100x100`, `runs/navsim-fusion/epoch_20.pth`, 전방 칸 0.32 m | 이번 런에서 읽지 않는다 | 8절 Stage 1A 어댑터는 이쪽에 맞춰져 있다. 재사용하지 않는다 |
| ReSMap | sharded `bev` `[256, 100, 50]`, 축 `(C, lateral, forward)` | `(C, lateral, forward)`를 `(C, forward, lateral)`로 전치한 뒤 폭을 뒤집으면 `(C, 50, 100)` | 기존 Stage 1B 어댑터는 이 50×100에 학습됐다. 그대로 쓴다 |

`distill_cache_size`는 `(50, 100)`이다. Stage 2는 `cache_train_50x100`이 있으면 그 디렉터리만 연다. 격자가 같으면 MSE 앞의 bilinear는 항등이다.

Stage 1A만 다시 학습한다. Stage 1B는 건너뛴다. 그 다음 Stage 2를 처음부터 돌린다. Lightning 디렉터리는 `version_6`이다. `version_5`는 100×100에 범퍼 항을 켠 런이라 잇지 않는다.

가중치는 5에폭마다 남긴다. Lightning 파일 이름은 0부터라 `epoch=4`, `epoch=9`, `epoch=14`, `epoch=19`, `epoch=24`, `epoch=29`이다. `last.ckpt`도 남는다. navtest 채점은 PDMS와 함께 det mAP, map mAP를 낸다. mAP의 박스는 전방 ROI와 80도 시야다.

```bash
cd /home/external-user/byounggun/SSR
FORCE_RETRAIN=stage1a,stage2 bash ./scripts/training/run_all_stages_distill.sh
```

GPU 기본값은 4, 5다. tmux 39의 `version_5`를 끈 뒤에 실행한다.
