# BEV Selector Distillation v3
## 앵커를 확률로 섞지 않는다

**기준 문서**: [`bev_selector_v2.md`](bev_selector_v2.md) (v2. 칸별 MSE와 cover 타깃은 유지하고, 앵커 혼합만 바꾼다)  
**앞 기록**: [`bev_selector.md`](bev_selector.md) (v1, PDMS 0.8343)  
**소속**: `/home/external-user/byounggun/SSR`  
**작성일**: 2026-10-04  
**상태**: 코드가 이 문서다. 이 레시피의 navtest 점수는 아직 없다.  
**비교**: 같은 navtest 12146 토큰, distill 키를 벗긴 학생.

Student는 바꾸지 않는다. 카메라, ResNet-34, BEV 50×100, dense planner, task interaction, plan / det / motion / map 손실, 진행 꼬리 0.003, GradBalancer 꺼짐은 v1, v2와 같다. 배포 그래프도 같다. Register는 학습 중에만 있고 `agent._distill.*`로 빠진다.

`version_2`는 잇지 않는다. 그 가중치는 epoch 5에 슬롯이 한 맵으로 잠긴 상태다.

---

## 1. v2가 낸 점수

평가한 파일은 `work_dirs/paradrive_distill_bev_selector/lightning_logs/version_2/checkpoints/last.ckpt`다. W&B `r34_sel_trial_3` (`td7bioqq`). epoch 29 체크포인트는 없다. 학습은 epoch 27, micro-step 28에서 끊겼다. 지표는 epoch 5부터 고정이라, 이 점수는 덜 배운 런이 아니라 잠긴 v2의 점수다.

병합 CSV: `work_dirs/eval/paradrive_distill_bev_selector_r34_trial3_last/merged.csv`

| 항 | v4 `version_4` | v1 selector | v2 selector |
|---|---:|---:|---:|
| valid | 1.0000 | 1.0000 | 1.0000 |
| no_at_fault_collisions | 0.9810 | 0.9797 | 0.9821 |
| drivable_area_compliance | 0.9371 | 0.9152 | 0.9251 |
| driving_direction_compliance | 1.0000 | 1.0000 | 1.0000 |
| ego_progress | 0.8020 | 0.7836 | 0.7951 |
| time_to_collision_within_bound | 0.9439 | 0.9373 | 0.9392 |
| comfort | 0.9999 | 1.0000 | 1.0000 |
| **score** | **0.8563** | **0.8343** | **0.8457** |

실패 장수. DAC는 1030 → **910** → v4는 764. TTC는 761 → 738 → v4는 681. NC < 1은 268 → **243** → v4는 246. score 0인 장면은 1237 → 1091 → v4는 969.

v2는 v1보다 **+0.0114**다. v4보다는 **−0.0105**다. NC 평균은 v4보다 높다. 표의 EP 0.7951은 DAC 또는 NC가 0인 장면의 EP가 0으로 적힌 결과다.

합법이고 TTC도 1인 장면의 EP:

| 런 | 장수 | EP | 장면 score |
|---|---:|---:|---:|
| v1 | 10449 | 0.8693 | 0.9456 |
| v2 | 10588 | 0.8706 | 0.9460 |
| v4 | 10759 | 0.8688 | 0.9453 |

세 런이 동시에 합법이고 TTC도 1인 9450장에서 EP는 v1 0.8686, v2 0.8699, v4 0.8693이다. 열린 루프 계획 L1은 epoch 26에 0.0148이다. v1 epoch 29는 0.0143, v4는 0.0139였다. 도로 안에 있는 궤적의 진행은 이미 v4 옆이다.

v4 대비 평균 −0.0105를 장면 점수로 나누면 다음과 같다. 기여는 (그 장면들의 score 합의 차) / 12146 이다. 음수는 v2가 더 낮은 쪽이다.

| 묶음 | 장수 | score 기여 | 그 묶음의 EP |
|---|---:|---:|---|
| NC는 1인데 DAC가 새로 0 | 506 | −0.0382 | 0.828 → 0 |
| NC는 1인데 DAC를 고침 | 376 | +0.0286 | 0 → 0.849 |
| 새로 NC < 1 | 132 | −0.0078 | 0.854 → 0.065 |
| NC를 고침 | 135 | +0.0084 | 0.039 → 0.866 |
| 둘 다 합법, TTC가 새로 0 | 225 | −0.0076 | 0.931 → 0.949 |
| 둘 다 합법, TTC를 고침 | 170 | +0.0057 | 0.939 → 0.921 |
| 둘 다 합법이고 TTC도 1 | 9970 | +0.0003 | 0.869 → 0.870 |

순기여는 DAC **−0.0096**, TTC −0.0019, NC +0.0006, 합법이고 TTC가 1인 장면 +0.0003이다. 새로 도로를 벗어난 506장은 v4에서 평균 score 0.917, EP 0.828이던 장면이다.

v1 대비 +0.0114의 순기여는 DAC +0.0086 (462장을 고치고 350장을 새로 놓침), NC +0.0019, TTC +0.0006, 합법이고 TTC가 1인 장면 +0.0004다.

---

## 2. 왜 이렇게 나왔나

v2의 칸별 MSE는 동작했다. 선택기는 동작하지 않았다.

`version_2` epoch 평균:

| epoch | div | cover | entropy | nearest | anchor mix | 칸 cosine bf / rm | `gnorm/distill` | `gnorm/det` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.117 | 0.246 | 7.17 | 4.67 m | 0.91 | 0.02 / 0.10 | 0.0001 | 0.027 |
| 4 | 1.097 | 0.607 | 8.33 | 0.96 m | 0.15 | 0.20 / 0.53 | 0.0005 | 0.046 |
| 5 | 1.769 | 0.665 | 8.45 | 0.339 m | 0.05 | 0.21 / 0.54 | 0.0005 | 0.051 |
| 26 | 1.770 | 0.809 | 8.44 | 0.339 m | 0.05 | 0.25 / 0.57 | 0.0005 | 0.053 |

epoch 26의 `bev_embed` 기울기 비중은 plan 0.051 / det 0.608 / map 0.342다. `gnorm/plan`은 0.0043이다. `distill_sel_mass_frac`는 0.041이라, 합친 attention이 0.01을 넘는 칸은 약 200개다. 5000칸 위의 학습된 선택이 아니라 σ 2 m 앵커 잔류다.

v2 식은 확률 혼합이었다.

\[
A=m\,G_{\sigma}+(1-m)\,\mathrm{softmax}(qK^{\top}/\sqrt{d})
\]

\[
m=0.05+(1-0.05)(1-\lambda),\qquad \sigma=6+\lambda(2-6)
\]

λ가 1이 되면 \(m=0.05\)다. 학습된 softmax가 5000칸 위에서 균일해지면, 슬롯 32개의 차이는 2 m 가우시안의 5%뿐이다. 그 상태를 같은 격자로 다시 만들면 diversity는 1.757, 최근접 평균은 0.339 m, entropy는 8.49다. 로그의 1.770 / 0.339 m / 8.44와 같다. 이 지점에서 diversity를 로짓으로 미분하면 절대값 평균이 \(1\times 10^{-5}\)다. 기울기가 없어서 epoch 5 이후에 최근접이 소수점 아래까지 움직이지 않았다.

Cover는 그 균일 맵이 뾰족한 타깃을 못 따라가서 0.25에서 0.81로 올랐다. 도로 경계와 차량 박스는 타깃에만 있고, 칸 가중에는 들어오지 않았다.

점수가 0.8343에서 0.8457로 오른 이유는 슬롯이 경계를 찾아서가 아니다. Attention이 거의 균일해진 뒤에도 칸별 LayerNorm MSE는 전 칸에 걸려, BEVFusion과 ReSMap 특징이 조금씩 맞춰졌다. 칸 cosine은 붕괴 이후에도 0.21 / 0.54에서 0.25 / 0.57로 천천히 올랐다. 그 약한 전역 맞춤이 DAC 실패를 1030에서 910으로 줄였다. 경계만 세게 맞춘 것은 아니다. `gnorm/distill` 0.0005는 `gnorm/det` 0.053의 약 100분의 1이다. 칸 가중이 \(1/5000\)이라 셀마다 기울기가 얇다.

그래서 남은 −0.0105는 진행 부족이 아니다. DAC 실패 910과 764의 차다. 진행 가중 0.003을 올리는 수정은 이 표에 없다.

---

## 3. 그대로 두는 것

| 항목 | 값 |
|---|---|
| backbone | `resnet34.tv_in1k` |
| BEV | 50×100, `pc_range` (−32, 0, −2, 32, 32, 2) m |
| planner | dense, `use_stl=false`, `plan_num_layers=3`, `use_task_interaction=true`, `use_ego_motion=false` |
| 태스크 가중 | plan 2, det 1, motion 1, map 1 |
| 진행 꼬리 | 가중 0.003, 캡 0.9. 횡꼬리 0 |
| GradBalancer | 꺼짐 |
| register | bank당 16, 합 32. Object bank는 BEVFusion, map bank는 ReSMap |
| 칸별 MSE | v2와 같다. Attention은 detach. LayerNorm은 칸마다, affine 없이 |
| cover 타깃 | v2와 같다. τ 0.3, β 0.5. 물체는 현재 박스, 지도는 road boundary |
| diversity | σ 4 m. 공간 거리와 attention cosine |
| λ | epoch 0에 0, epoch 5에 1. `distill_selector_tok_warmup_steps=53195` |
| 추론 | register 없음. planner는 5000칸을 본다 |

Stage 1 adapter, corridor MSE, role-mask MSE, CWD, relation, attention KD, head KD, look KL은 가중 0이다. GradBalancer를 이 런에서 켜지 않는다. 격자와 backbone을 이 런에서 바꾸지 않는다. 붕괴와 밸브를 한 번에 바꾸면 어느 쪽이 DAC를 움직였는지 남지 않는다.

---

## 4. 앵커는 같은 softmax 안에 둔다

확률 혼합을 뺀다. σ는 그대로 6 m에서 2 m로 줄어든다.

\[
A=\mathrm{softmax}\big(\lambda\, qK^{\top}/\sqrt{d}+\log G_{\sigma}\big)
\]

\[
\sigma=6+\lambda(2-6)
\]

λ=0이면 로짓은 꺼지고 \(A=G_{6}\)이다. 시작 순간의 무작위 projection이 격자를 지우지 않는다. λ=1이면 로짓과 \(\log G_{2}\)가 한 softmax다.

상수 로짓은 softmax를 바꾸지 못한다. 슬롯 32개의 로짓이 같아도 맵은 격자 가우시안 32개로 남는다. 50×100에서 σ 2 m인 그 가우시안은 entropy 5.05, 최근접 최소 4.18 m, 최근접 평균 6.77 m다. v2가 잠긴 0.34 m / entropy 8.44와 다른 점이다.

슬롯이 격자 밖에 있는 경계로 가는 경로는 열려 있다. 앵커에서 8 m 떨어진 칸의 로짓을 40 올리면 그 슬롯의 평균은 8.0 m 움직인다. Cover가 경계와 차량으로 당길 때 필요한 이동이다. 혼합의 5% 잔류에는 그 기울기가 없었다.

칸 가중, 두 bank, detach, cover의 후반 타깃은 v2 식을 유지한다. 선택기가 경계를 따라가면 그 칸의 LayerNorm MSE만 진해진다. v2 후반처럼 5000칸을 같은 무게로 맞추지 않는다.

`distill_selector_anchor_floor`와 `distill_sel_anchor_mix`는 뺀다. 명령 쐐기 prior 안의 0.05는 바닥 질량이고, 이번에 뺀 앵커 혼합과 다른 수다.

---

## 5. 로그와 판정

W&B 그룹 `bev-selector`. run 이름은 `r34_sel_trial_N`. 태그는 `bev-selector`, `r34`, `v3`다.

| 신호 | v2 epoch 26 | 이 런 |
|---|---:|---|
| `distill_sel_entropy` | 8.44 | 5 근처. 7.5를 넘고 최근접이 1 m 아래면 로짓이 가우시안을 지운 것이다 |
| `distill_sel_nearest_m` | 0.339 m | 2 m보다 큼. σ 2 m 격자의 최소는 4.2 m다. 0.34 m면 v2 잔류다 |
| `loss_distill_div` | 1.77 | 1.5 아래를 유지. 1.7에서 멈추면 중단한다 |
| `loss_distill_cover` | 0.809 | epoch 5 이후에 내려갈 것. 0.81에 머무르면 타깃을 못 따르는 것이다 |
| `distill_sel_anchor_sigma` | epoch 5 이후 2 | epoch 5 이후 2 |
| `gnorm/distill` | 0.0005 | 칸이 모이면 0.0005보다 커진다. `gshare` 분모에는 넣지 않는다 |
| 합법이고 TTC가 1인 장면 EP | 0.871 | 0.87 유지 |
| DAC 실패 | 910 | 910보다 적을 것. v4는 764 |

epoch 10에 entropy가 7.5보다 크거나 최근접이 1 m보다 작으면 그 체크포인트를 잇지 않는다. 진행 가중은 올리지 않는다.

기하가 맞은 뒤에도 `gnorm/distill`이 0.001 아래이고 `gnorm/det`가 0.05 근처이며 DAC가 910보다 줄었는데 764에 못 미치면, 그 체크포인트를 잇지 않는다. 다음 런은 이 문서의 distill을 유지한 채 GradBalancer만 0.40 / 0.30 / 0.30으로 켠다. 그 런은 이번 런이 아니다.

---

## 6. 코드

| 자리 | 역할 |
|---|---|
| `navsim/agents/para_ssr/distill/selector.py` | `anchor_log_attention`. 칸별 MSE와 sharpened cover는 v2 |
| `navsim/agents/para_ssr/distill/distillation.py` | 앵커 바닥 인자를 넘기지 않음 |
| `navsim/agents/para_ssr/configs/default.py` | σ 6→2. `distill_selector_anchor_floor` 없음 |
| `navsim/planning/script/config/common/agent/para_ssr_selector_agent.yaml` | 이 값. Student와 GradBalancer는 v2 그대로 |
| `scripts/training/run_bev_selector_distill.sh` | 태그 `v3` |
| `tests/test_bev_selector.py` | 끝난 뒤는 σ 2 m 가우시안. v2 잔류의 diversity 기울기는 \(10^{-4}\) 아래. 로짓 봉우리는 앵커를 떠날 수 있음 |

v4 yaml의 `distill_selector`는 false다. `scripts/training/run_all_stages_distill.sh`는 이 변경을 타지 않는다.

---

## 7. 실행

`version_2`를 resume하지 않는다. 같은 실험 디렉터리에 Lightning이 다음 version을 만든다. 지금 디렉터리는 `version_0`, `version_1`, `version_2`까지 있다.

```bash
cd /home/external-user/byounggun/SSR
bash ./scripts/training/run_bev_selector_distill.sh
```

GPU 기본값은 4, 5다. Epoch 30, batch 4, accumulate 16, lr \(10^{-4}\). 체크포인트는 5 epoch마다와 `last.ckpt`.

학습이 끝나면 distill 키를 벗기고 navtest. `IMAGE_ARCHITECTURE`는 `resnet34.tv_in1k`다. 아래는 다음 디렉터리가 `version_3`일 때의 epoch 29다. 디렉터리 번호가 다르면 `CKPT_SRC`만 그 epoch 29 파일로 바꾼다.

```bash
cd /home/external-user/byounggun/SSR
CKPT_SRC=/home/external-user/byounggun/SSR/work_dirs/paradrive_distill_bev_selector/lightning_logs/version_3/checkpoints/epoch=29-step=19950.ckpt \
SNAPSHOT_DIR=/home/external-user/byounggun/SSR/work_dirs/eval_snapshots/paradrive_distill_bev_selector_v3_epoch29 \
EXPERIMENT_NAME=eval/paradrive_distill_bev_selector_v3_epoch29 \
IMAGE_ARCHITECTURE=resnet34.tv_in1k \
GPU0=4 GPU1=5 \
bash scripts/evaluation/eval_para_ssr_distill_epoch29_gpus45.sh
```
