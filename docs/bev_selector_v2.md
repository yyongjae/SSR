# BEV Selector Distillation v2
## 평균 토큰 대신, 고른 칸을 맞춘다

**기준 문서**: [`bev_selector.md`](bev_selector.md) (v1. 그 파일의 측정만 여기로 가져오고, 레시피는 바꾸지 않는다)  
**소속**: `/home/external-user/byounggun/SSR`  
**작성일**: 2026-09-30  
**상태**: 코드가 이 문서다. 학습 점수는 아직 없다.  
**비교**: 같은 navtest 12146 토큰, distill 키를 벗긴 epoch 29.

Student는 바꾸지 않는다. 카메라, ResNet-34, BEV 50×100, dense planner, task interaction, plan / det / motion / map 손실, 진행 꼬리 0.003, GradBalancer 꺼짐은 v1과 같다. 배포 그래프도 같다. Register는 학습 중에만 있고 `agent._distill.*`로 빠진다.

---

## 1. v1이 낸 점수

`r34_sel_trial_2`, Lightning `version_1`, 체크포인트 `work_dirs/paradrive_distill_bev_selector/lightning_logs/version_1/checkpoints/epoch=29-step=19950.ckpt`. 유효 12146, 실패 0.

병합 CSV: `work_dirs/eval/paradrive_distill_bev_selector_r34_trial2_epoch29/merged.csv`

| 항 | v4 `version_4` | v1 selector |
|---|---:|---:|
| valid | 1.0000 | 1.0000 |
| no_at_fault_collisions | 0.9810 | 0.9797 |
| drivable_area_compliance | 0.9371 | 0.9152 |
| driving_direction_compliance | 1.0000 | 1.0000 |
| ego_progress | 0.8020 | 0.7836 |
| time_to_collision_within_bound | 0.9439 | 0.9373 |
| comfort | 0.9999 | 1.0000 |
| **score** | **0.8563** | **0.8343** |

실패 장수: DAC 764 → **1030** (+266), TTC 681 → 761, NC < 1은 246 → 268. score 0인 장면은 969 → 1237.

v4 대비 평균 −0.0220을 장면 점수로 나누면 다음과 같다. 기여는 (그 장면들의 score 합의 차) / 12146 이다.

| 묶음 | 장수 | score 기여 | 그 묶음의 EP |
|---|---:|---:|---|
| NC는 1인데 DAC가 새로 0 | 626 | −0.0474 | 0.839 → 0 |
| NC는 1인데 DAC를 고침 | 382 | +0.0293 | 0 → 0.858 |
| 새로 NC < 1 | 143 | −0.0089 | 0.863 → 0.061 |
| NC를 고침 | 121 | +0.0074 | 0.064 → 0.853 |
| 둘 다 합법, TTC가 새로 0 | 228 | −0.0077 | 0.924 → 0.943 |
| 둘 다 합법, TTC를 고침 | 164 | +0.0055 | 0.935 → 0.915 |
| 둘 다 합법이고 TTC도 1 | 9835 | −0.0002 | 0.869 → 0.869 |

순기여는 DAC **−0.0181**, TTC −0.0022, NC −0.0015다. 새로 도로를 벗어난 626장은 v4에서 평균 score 0.920이던 장면이다.

합법이고 TTC도 1인 9835장의 EP는 0.869에서 움직이지 않았다. 표의 EP 0.7836은 DAC가 0인 장면의 EP가 0으로 적힌 결과다. 열린 루프 계획 L1은 epoch 29에 0.0143으로 v4의 0.0139 옆이다. 궤적 평균은 배웠고, 경계에서 잘렸다.

v1 학습 신호. entropy 7.70은 슬롯 하나의 유효 칸이 약 2200개, 반경 약 7 m라는 뜻이다. cover는 epoch 0의 0.20에서, 타깃이 planner attention으로 바뀐 뒤 0.48에 고정됐다. 최근접 거리는 4.94 m에서 3.10 m로 모였다. `bev_embed` 기울기 비중은 plan 0.05 / det 0.62 / map 0.33이었다. 토큰 cosine 0.588 / 0.726은 그 넓은 attention으로 평균 낸 벡터의 cosine이다.

Register가 한 점에 죽은 런은 아니다. diversity는 0.37이고 최근접은 3.1 m다. 진행 꼬리 가중은 v4와 같은 0.003이고, 총손실에 들어간 양은 0.00018이다.

---

## 2. 그대로 두는 것

| 항목 | 값 |
|---|---|
| backbone | `resnet34.tv_in1k` |
| BEV | 50×100, `pc_range` (−32, 0, −2, 32, 32, 2) m |
| planner | dense, `use_stl=false`, `plan_num_layers=3`, `use_task_interaction=true`, `use_ego_motion=false` |
| 태스크 가중 | plan 2, det 1, motion 1, map 1 |
| 진행 꼬리 | 가중 0.003, 캡 0.9. 횡꼬리 0 |
| GradBalancer | 꺼짐 |
| register | bank당 16, 합 32. Object bank는 BEVFusion, map bank는 ReSMap |
| attention | detach. 맞추는 손실이 슬롯을 움직이지 않는다 |
| diversity | σ 4 m. 공간 거리와 attention cosine |
| 초반 cover 타깃 | command 쐐기. GT 궤적 Gaussian은 넣지 않는다 |
| λ | epoch 0에 0, epoch 5에 1. `distill_selector_tok_warmup_steps=53195` |
| 추론 | register 없음. planner는 5000칸을 본다 |

Stage 1 adapter, corridor MSE, role-mask MSE, CWD, relation, attention KD, head KD, look KL은 가중 0이다. `distill_use_role_masks`는 false다. 아래 경계 ring은 cover 타깃으로만 쓰고, MSE 가중으로 쓰지 않는다.

`version_1` 가중치는 잇지 않는다. 손실이 바뀌어 있다.

---

## 3. 칸마다 맞춘다

v1의 \(\mathcal{L}_{\mathrm{tok}}\)는 register attention으로 5000칸을 평균한 뒤 LayerNorm 벡터 하나를 teacher와 비교했다. 경계와 옆 칸이 한 벡터로 섞였다.

v2는 그 attention을 detach한 뒤 칸 가중으로 쓴다. Object bank의 합은 BEVFusion에만, map bank의 합은 ReSMap에만 곱한다. LayerNorm은 칸마다, affine 없이. 학습되는 adapter는 없다.

\[
\bar A^{(b)}=\mathrm{normalize}\Big(\sum_k A^{(b)}_k.\mathrm{detach}\Big)
\]

\[
\mathcal{L}_{\mathrm{tok}}
=\sum_{b\in\{\mathrm{bf},\mathrm{rm}\}}
\sum_{ij}\bar A^{(b)}_{ij}\;
\mathrm{mean}_c\big(\mathrm{LN}(s_{ij})-\mathrm{LN}(t^{(b)}_{ij})\big)^2
\]

제곱은 채널 평균이다. 256을 그대로 더하면 태스크 손실 스케일을 넘는다. \(\mathcal{L}_{\mathrm{tok}}\)는 지금처럼 λ를 곱해 epoch 0에 0, epoch 5에 1이다.

로그의 `train/distill_sel_cos/{bevfusion,resmap}`는 이 칸 가중의 cosine이다. v1의 0.588 / 0.726과 다른 수다. 평균 벡터 cosine은 `train/distill_sel_cos_pooled/*`에 따로 남긴다. 초반에 칸 cosine이 0.59보다 낮아도 경계가 평균에서 빠져나온 것이다.

---

## 4. Cover의 후반 타깃

v1은 λ 이후에 planner attention \(P\) 전체를 따라갔다. \(P\)가 넓어서 슬롯이 겹치고 softmax는 더 넓어졌다.

후반 타깃만 바꾼다. \(P\)는 detach한 뒤 temperature \(\tau=0.3\)으로 뾰족하게 만든다. 확률에 대한 연산은 \(P^{1/\tau}\)를 다시 정규화한 것이다.

\[
P_{\sharp}=\mathrm{normalize}\big(P^{1/\tau}\big)
\]

그 다음 bank별 구조와 \(\beta=0.5\)로 섞는다.

\[
T_{\mathrm{late}}=(1-\beta)\,P_{\sharp}+\beta\,S,\qquad
T=(1-\lambda)\,T_{\mathrm{cmd}}+\lambda\,T_{\mathrm{late}}
\]

\(S\)는 행마다 정규화한다. 그 행의 합이 0이면 그 샘플은 \(P_{\sharp}\)만 쓴다.

| bank | \(S\) |
|---|---|
| object (BEVFusion) | 현재 GT 차량 박스. `rasterize_agent_mask`, inflate 1.5. 미래 궤적 splat은 넣지 않는다 |
| map (ReSMap) | road 클래스 splat의 morphological boundary. `distill_map_sigma=1.25`, kernel 3 |

도로 내부, 차선, 횡단보도, GT 궤적 Gaussian은 \(S\)에 넣지 않는다. 궤적 Gaussian은 v4 corridor가 되고, v3에서 진행을 밀다 TTC를 깨던 계열과 이어진다. 경계 ring은 차체가 나가면 DAC가 0이 되는 칸이다.

---

## 5. Anchor는 확률로 섞는다

격자 위치는 v1과 같다. 4×4, 좌우 [−24, 24] m, 전방 [4, 28] m. Map bank는 반 칸 어긋난다. 시작 분포는 \(N(0, 10^{-6})\)이다.

폭과 세기는 λ와 같이 움직인다.

\[
\sigma=6+\lambda(2-6),\qquad m=0.05+(1-0.05)(1-\lambda)
\]

\[
A=m\,G_{\sigma}+(1-m)\,\mathrm{softmax}(qK^{\top}/\sqrt{d})
\]

\(G_{\sigma}\)는 그 폭의 정규화된 격자 gaussian이다. λ=0이면 \(m=1\), σ=6 m라서 슬롯이 처음부터 갈라진다. λ=1이면 \(m=0.05\), σ=2 m이다. 바닥 0.05는 남은 자리 편향이고, 슬롯이 한 점에 붙는 것을 막는 힘은 diversity다.

이 \(m\)은 logit에 곱하지 않는다. σ 6 m 로짓을 0.05배로 곱하면 softmax 폭이 \(6/\sqrt{0.05}\approx 27\) m로 넓어진다. 폭을 2 m로 유지하려고 확률에서 섞는다.

---

## 6. 로그와 판정

W&B 그룹 `bev-selector`. run 이름은 `r34_sel_trial_N`. 태그는 `bev-selector`, `r34`, `v2`다.

볼 것:

| 신호 | v1 epoch 29 | 이 런이 맞은 범위 |
|---|---:|---|
| `distill_sel_entropy` | 7.70 | 5.5 아래. σ 2 m면 약 5.1 |
| `distill_sel_nearest_m` | 3.10 m | 2–4 m. 0이면 슬롯이 다시 한 점이다 |
| `loss_distill_cover` | 0.484 | 0.25 아래 |
| `distill_sel_anchor_sigma` | 6에 고정이었음 | epoch 5 이후 2 |
| `distill_sel_anchor_mix` | — | epoch 5 이후 0.05 |
| `gnorm/distill` | 없음 | `gnorm/det` 옆에 읽는다. `gshare` 분모에는 넣지 않는다 |
| DAC 실패 | 1030 | 1030보다 적을 것. v4는 764 |
| 합법이고 TTC가 1인 장면 EP | 0.869 | 0.869 유지 |

`gnorm/plan`, `gnorm/det`, `gnorm/map`의 비중 정의는 v1과 같다. plan / det / map만으로 1이 된다.

entropy가 5.5 아래로 내려가는데 DAC가 1030 근처면, 폭은 좁아졌고 장소가 아직 경계가 아니다. 그때 map bank의 \(\beta\)를 올린다. entropy가 7 근처에 남으면 cover 타깃이 다시 넓어진 것이다. \(\tau\)를 0.3보다 낮춘다. 진행 가중은 올리지 않는다.

`gnorm/distill`이 `gnorm/det`보다 훨씬 작고, DAC는 1030보다 줄었는데 764에 못 미치면, 그 체크포인트를 잇지 않는다. 다음 런은 이 문서의 distill을 유지한 채 GradBalancer만 0.40 / 0.30 / 0.30으로 켠다. Backbone과 격자는 그 다음이다.

---

## 7. 코드

| 자리 | 역할 |
|---|---|
| `navsim/agents/para_ssr/distill/selector.py` | 칸별 MSE, sharpened cover, anchor 확률 혼합 |
| `navsim/agents/para_ssr/distill/distillation.py` | 차량 박스와 road boundary를 bank 타깃으로 넘김 |
| `navsim/agents/para_ssr/configs/default.py` | τ 0.3, σ 6→2, 바닥 0.05, β 0.5 |
| `navsim/planning/script/config/common/agent/para_ssr_selector_agent.yaml` | 이 값. Student와 GradBalancer는 v1 그대로 |
| `navsim/agents/para_ssr/para_ssr_agent.py` | `gnorm/distill` |
| `tests/test_bev_selector.py` | 칸 밖 오차는 손실이 0, step 0은 σ 6 m anchor, 끝난 뒤 잔류는 σ 2 m의 5% |

v4 yaml의 `distill_selector`는 false다. `scripts/training/run_all_stages_distill.sh`는 이 변경을 타지 않는다.

---

## 8. 실행

v1 `version_1`을 resume하지 않는다. 같은 실험 디렉터리에 Lightning이 다음 version을 만든다.

```bash
cd /home/external-user/byounggun/SSR
bash ./scripts/training/run_bev_selector_distill.sh
```

GPU 기본값은 4, 5다. Epoch 30, batch 4, accumulate 16, lr \(10^{-4}\). 체크포인트는 5 epoch마다와 `last.ckpt`.

학습이 끝나면 distill 키를 벗기고 navtest. `IMAGE_ARCHITECTURE`는 `resnet34.tv_in1k`다. version 번호는 디렉터리를 보고 넣는다.

```bash
cd /home/external-user/byounggun/SSR
CKPT_SRC=/home/external-user/byounggun/SSR/work_dirs/paradrive_distill_bev_selector/lightning_logs/version_2/checkpoints/epoch=29-step=19950.ckpt \
SNAPSHOT_DIR=/home/external-user/byounggun/SSR/work_dirs/eval_snapshots/paradrive_distill_bev_selector_v2_epoch29 \
EXPERIMENT_NAME=eval/paradrive_distill_bev_selector_v2_epoch29 \
IMAGE_ARCHITECTURE=resnet34.tv_in1k \
GPU0=4 GPU1=5 \
bash scripts/evaluation/eval_para_ssr_distill_epoch29_gpus45.sh
```

version 디렉터리가 `version_2`가 아니면 `CKPT_SRC`만 그 epoch 29 파일로 바꾼다.
