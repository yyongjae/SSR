# Dual-Teacher Planning-Aware BEV Distillation (DPD) v2
## Aux GT를 플래너 BEV로 가져오는 Stage-2 개정 (디코더 변경 없음)

**기준 문서**: [`TOP_TIER_PLANNING_DISTILLATION_PLAN.md`](TOP_TIER_PLANNING_DISTILLATION_PLAN.md) (v1 학습 계약, 이 파일이 원본이며 그대로 둔다)  
**소속/작업공간**: `/home/external-user/byounggun/SSR`  
**작성일**: 2026-09-20  
**개정일**: 2026-09-21  
**코드 기본값**: `ParaSSRConfig`, `para_ssr_distill_agent.yaml`, `navsim/agents/para_ssr/distill/`

v1은 frozen-adapter dual-teacher + 등방 복도 MSE + GT plan/det/map + GradBalancer다. v2는 **플래너/디코더를 바꾸지 않고** Stage-2 증류만 바꾼다. 배포 그래프는 v1과 같다: `camera → BEVFormer → planner`.

첫 개정은 마스크를 좁히고 헤드 KD·CWD를 얹었다가 navtest에서 PDMS가 떨어졌다. 이번 개정은 **aux 헤드 응답을 맞추지 않고**, det/map GT가 가리키는 셀을 플래너가 읽는 `bev_embed`로 옮긴다.

---

## 1. 왜 v1 위에 더 올렸나

v1 Stage-2 ResNet-50, 30 epoch, navtest PDM (aux-distill epoch 29):

| 항 | 값 | 비고 |
|---|---:|---|
| NC | 0.9815 | 거의 천장 |
| DAC | 0.9348 | NAVSIM v1 PDMS의 곱항. 가장 비싼 병목 |
| DDC | 1.0000 | |
| EP | 0.8022 | 그다음 병목 |
| TTC | 0.9395 | EP와 같이 움직임 |
| comfort | 0.9998 | |
| **score** | **0.8539** | published PARA-Drive camera 84.0보다 위 |

목표는 DiffusionDrive / Hydra-MDP++ 쪽 점수다. 남은 갭은 DAC, 그다음 EP/TTC다. 디코더 교체는 하지 않기로 했다.

등방 복도 MSE의 한계는 그대로다.

1. NAVSIM 궤적은 `(x_forward, y_left, heading)`인데 학생 BEV는 SSR `(x_right, y_forward)`다. 변환 없이 뿌리면 복도가 옆 축에 깔린다.
2. 두 교사에 같은 마스크를 쓰면 BEVFusion의 동적 객체와 ReSMap의 차선이 같은 셀에서 싸운다.
3. Aux 헤드(det/map)는 공유 BEV를 보지만, 그 정보가 플래너 cross-attn이 쓰는 셀로 정리되지 않는다. GradBalancer는 aux가 BEV를 잡아먹지 못하게만 한다. **aux가 아는 “어디가 위험한지 / 어디가 도로인지”를 플래너 쪽으로 옮기는 증류는 v1에 없다.**

---

## 2. v1에서 안 바꾸는 것

| 항목 | 유지 |
|---|---|
| Stage 1A/1B | 교사별 `PlanningBEVAdapter` + 궤적 L1, 센서 없음, \((v,a)=0\) |
| Stage 2 어댑터 | 같은 인스턴스, freeze, `eval()` |
| 학생 | 전방 3캠, ResNet-50, BEVFormer 3층, dense planner (`use_stl=false`, `plan_num_layers=3`) |
| 플래너 동역학 | 현재 프레임 \((v_x,v_y,a_x,a_y)\) residual. 배포에도 남음 |
| GradBalancer | plan 0.4 / det 0.3 / map 0.3. 증류는 이 루프 밖 |
| GT 손실 | \(2.0\mathcal{L}_{\text{plan}}+1.0\mathcal{L}_{\text{det}}+1.0\mathcal{L}_{\text{motion}}+1.0\mathcal{L}_{\text{map}}\) |
| 추론 | 어댑터·교사 캐시·보조 헤드 없음. distill 키는 export 때 제거 |

비교 기준 체크포인트 `work_dirs/paradrive_distill_stage2_dual_distill/lightning_logs/version_0/checkpoints/epoch=29-step=19950.ckpt` (PDMS 0.8539)는 지우지 않는다.

---

## 3. 첫 v2가 깎은 것 (navtest로 확인됨)

첫 개정 기본값: 이방 복도 \(\sigma_\perp=1.25\), 바닥가중 0.05, 보도 suppress 0.8, CWD/relation/attention, inverse-EMA adaptive scale, map-head KD 0.3. ResNet-50 Stage-2 `lightning_logs/version_1` epoch 29:

| 항 | v1 | 첫 v2 | 차이 |
|---|---:|---:|---:|
| NC | 0.9815 | 0.9799 | −0.0016 |
| DAC | 0.9348 | 0.9266 | −0.0082 |
| EP | 0.8022 | 0.7920 | −0.0102 |
| TTC | 0.9395 | 0.9348 | −0.0047 |
| **score** | **0.8539** | **0.8433** | **−0.0106** |

같은 12146 토큰. DAC 실패 792 → 891 (신규 443, 회복 344). NAVSIM은 DAC가 깨지면 EP를 0으로 찍는다. 점수 하락의 약 0.008은 그 추가 DAC 실패고, DAC를 둘 다 통과한 장면의 EP는 0.858 → 0.855다. v2가 노리던 병목이 나빠졌다.

학습 신호:

| 신호 (ep29) | v1 | 첫 v2 |
|---|---:|---:|
| BEVFusion cosine | 0.37 | 0.15 |
| ReSMap cosine | 0.60 | 0.78 |
| unscaled MSE bf / rm | (가지 KD만) | 1.74 / 0.43 |
| scaled `loss_distill_*` | 1.29 / 0.81 | 둘 다 1.16 |
| plan L1 | 0.0275 | 0.030 |

원인:

1. **횡방향 복도가 차선 폭보다 좁다.** \(\sigma_\perp=1.25\,\mathrm{m}\)이면 도로 경계(~1.7–2.5 m)가 배경이 된다. DAC가 필요한 셀이 증류에서 빠진다.
2. **보도 suppress가 도로 경계를 같이 누른다.** 보도 가우시안 \(\sigma=1.25\)가 연석과 겹치면 ReSMap이 올려야 할 경계 가중치가 0.8배로 깎인다.
3. **inverse-EMA adaptive scale이 쉬운 교사를 키운다.** ReSMap MSE가 작아지니 \(s_{\text{resmap}}\approx 2.5\), BEVFusion은 \(s\approx 0.62\). 객체 교사(NC/TTC)가 약해지고 맵 교사만 세진다.
4. **CWD / relation / teacher-attention은 같은 좁은 마스크 위에 한 번 더 눌렀다.** attention 항은 \(\sim 10^{-5}\)라 사실상 0이다.
5. **map-head KD는 aux를 플래너로 가져오지 않는다.** Stage-1 어댑터는 계획 공간 투영기다. 맵 헤드를 그 토큰에 다시 태우면 헤드가 두 언어를 배우고, 플래너가 읽는 `bev_embed`는 그대로다. det-head KD와 같은 종류고, det는 이미 발산해서 뺐다.

CSV: `work_dirs/eval/paradrive_distill_stage2_v1_epoch29/merged.csv`.

---

## 4. Aux에서 플래너로 가져올 것

Aux 헤드 출력(cls/bbox/polyline)을 어댑터 토큰에 맞추는 대신, **aux GT가 표시하는 계획-전경 셀**을 공유 BEV에 심는다. 플래너는 dense cross-attn으로 그 BEV를 읽는다. 디코더는 그대로다.

| Aux가 아는 것 | PDMS | 플래너 BEV에 남기는 방법 |
|---|---|---|
| 현재·미래 에이전트 (det GT) | NC, TTC | BEVFusion 가지 마스크에 에이전트 splat을 **더한다**. 학생 `bev_embed` 공간 에너지가 그 셀을 보도록 look prior에 넣는다. |
| 도로 경계 / 센터라인 / 횡단보도 (map GT) | DAC, EP | ReSMap 가지 마스크에 도로·센터라인·**형태학적 경계 링**을 더한다. 보도는 누르지 않는다. look prior에도 도로·경계를 넣는다. |
| GT 궤적 복도 | EP | heading-aware 복도. 횡방향은 차선+연석이 들어가게 넓힌다. |

가져오지 않는 것:

- 어댑터 토큰 위 det/map 헤드 응답 KD (계획 공간이 헤드 교사가 아님)
- 보도 셀을 눌러서 연석 대비를 지우는 것
- 쉬운 교사 손실을 키우는 inverse-EMA
- 교사 공간 softmax를 학생에 복사하는 generic attention KD (첫 v2에서 항이 0)

GradBalancer는 그대로 둔다. Aux 헤드 파라미터는 GT로 학습하고, BEV로 역류하는 양만 줄인다. 증류는 그 루프 밖에서 **셀 선택**을 담당한다.

$$
\begin{aligned}
\mathcal{L}
&= \mathcal{L}_{\text{task}}
 + \lambda_{\text{kd}}\sum_{k\in\{\text{bevfusion},\text{resmap}\}}
      \mathcal{L}_{\text{MSE}}^{k}(W_k)
 + \lambda_{\text{kd}}\,\mathcal{L}_{\text{look}}(W_{\text{plan}})
\end{aligned}
$$

\(W_k\)는 교사별 전경, \(W_{\text{plan}}\)은 두 aux를 합친 전경이다. \(s_k\) adaptive는 끈다.

---

## 5. Stage-2에 실제로 넣는 것

기본값은 `para_ssr_distill_agent.yaml`.

### 5.1 NAVSIM → SSR 궤적 변환 (유지)

`distill_trajectory_frame=navsim`. 복도를 뿌리기 전에

\[
(x_{\text{right}}, y_{\text{forward}}, \psi_{\text{SSR}})
= (-y_{\text{left}},\; x_{\text{forward}},\; \psi_{\text{NAVSIM}}+\pi/2)
\]

코드: `navsim_trajectory_to_ssr`. heading은 이방 복도 축에만 쓴다.

### 5.2 이방 복도, 횡방향은 차선+연석

진행 방향으로는 길고, 옆으로는 v1 등방 \(\sigma=2.5\,\mathrm{m}\)와 같게 연다. 바닥 가중은 v1과 같은 0.1이다.

| 기호 | 값 | 의미 |
|---|---|---|
| \(\sigma_{\parallel}\) | 4.0 m, 성장 0.15 | 궤적 따라 앞뒤 |
| \(\sigma_{\perp}\) | 2.5 m, 성장 0.10 | 차선 + 인접 경계 |
| \(\varepsilon\) | 0.1 | 복도 밖 바닥. 배경 BEV가 죽지 않게 |

\[
W_{\text{corr}}=\varepsilon+(1-\varepsilon)\cdot\max_t
\exp\!\Big(-\tfrac{1}{2}\big((d_\parallel/\sigma_{\parallel,t})^2+(d_\perp/\sigma_{\perp,t})^2\big)\Big)
\]

### 5.3 교사별 역할 마스크 (더하기만)

`distill_use_role_masks=true`. 복도 위에 aux 전경을 **max**로 올린다. 보도 곱 억제는 끈다 (`distill_walkway_suppress=0`).

| 교사 | \(W_k\)에 올리는 셀 |
|---|---|
| BEVFusion | GT 박스 1.5× inflate + 미래 에이전트 splat \(\sigma=1.5\) |
| ReSMap | 도로 / 센터라인 / 횡단보도 splat \(\sigma=1.25\) + 도로의 형태학적 경계 (dilate−erode, kernel 3) |

마스크는 증류 feature MSE에만 곱한다. plan/det/map GT에는 안 곱는다.

### 5.4 Feature KD: 마스크 MSE만

어댑터를 통과한 학생/교사에 같은 \(W_k\)로 정규화 MSE. CWD / relation / teacher-attention 가중은 0. 가지 `loss_weight=1` 고정 (adaptive 꺼짐). 코사인은 메트릭만.

### 5.5 Planning-look prior (aux → `bev_embed`)

DistillBEV의 “어디를 보는지”를 교사 에너지가 아니라 **aux GT 전경**에 건다. 학생 격자 위 raw `bev_embed` (어댑터 전, 플래너가 읽는 텐서)에

\[
W_{\text{plan}}=\max(W_{\text{corr}},\,W_{\text{agent}},\,W_{\text{road}},\,W_{\text{center}},\,W_{\text{cross}},\,W_{\text{boundary}})
\]

\[
\mathcal{L}_{\text{look}}
=\mathrm{KL}\Big(\mathrm{softmax}(\overline{|S|}/\tau)\;\Big\|\;
\mathrm{normalize}(W_{\text{plan}})\Big)
\]

\(\overline{|S|}\)는 채널 평균 절댓값, \(\tau=0.5\), 가중 `distill_plan_look_weight=0.5`. 그래디언트는 얼린 어댑터를 거치지 않고 인코더로 바로 간다. 추론 그래프는 그대로다.

### 5.6 Head KD (둘 다 끔)

`distill_head_kd_weight=0`, `distill_head_kd_det=false`. 맵/디텍 헤드는 GT만 본다. Aux 전경은 §5.3–5.5가 플래너 BEV로 옮긴다.

---

## 6. 넣었다가 뺀 것

### 6.1 det-head response KD (발산)

ResNet-34 Stage-2, 2026-09-19. `loss_distill_head_det` ep0 0.54 → ep5 449. Stage-1 어댑터는 계획 투영기이고, DETR를 Hungarian 없이 인덱스 비교했으며 `traj_preds` L1은 상한이 없다.

### 6.2 map-head response KD (점수 하락과 함께 제거)

항은 작았지만 (0.02–0.04) 같은 잘못된 공간이다. 맵 헤드는 GT polyline으로 학습하고, 도로 셀은 ReSMap MSE와 look prior가 가져간다.

### 6.3 좁은 복도 · 보도 suppress · adaptive · extra KD

§3. 코드 경로는 남긴다. 가중 0 / suppress 0 / adaptive false가 기본이다.

---

## 7. 현재 코드 기본값

| 키 | 값 |
|---|---|
| `image_architecture` | `resnet50.tv_in1k` (v1 점수 런과 동일) |
| `distill_trajectory_frame` | `navsim` |
| `corridor_sigma_along` / `cross` | 4.0 / **2.5** |
| `corridor_base_weight` | **0.1** |
| `distill_use_role_masks` | true |
| `distill_walkway_suppress` | **0.0** |
| `distill_boundary_kernel` | 3 |
| `distill_mse_weight` | 1.0 |
| `distill_cwd_weight` | **0.0** |
| `distill_relation_weight` | **0.0** |
| `distill_attn_weight` | **0.0** |
| `distill_adaptive_branch` | **false** |
| `distill_plan_look_weight` | **0.5** |
| `distill_plan_look_tau` | 0.5 |
| `distill_head_kd_weight` | **0.0** |
| `distill_head_kd_det` | false |
| W&B | 기본 꺼짐. TensorBoard `lightning_logs/` |

Stage 2는 처음부터. 첫 v2 `version_1` epoch-29를 이어서 쓰지 않는다. v1 `version_0` epoch-29는 비교 기준이니 지우지 않는다.

```bash
FORCE_RETRAIN=stage2 ONLY_STAGE=stage2 bash ./scripts/training/run_all_stages_distill.sh
```

---

## 8. 성능에 대해 지금 말할 수 있는 것

맞다: 첫 v2 조합은 navtest에서 PDMS 0.8539 → 0.8433이고, 깎인 항은 주로 DAC다. 그 레시피는 기본값에서 뺐다.

아니다: 그래서 이번 개정이 0.8539를 넘는다는 보장은 없다. look prior와 경계 링은 설계상 DAC/TTC 셀을 플래너 BEV에 남기려는 것이고, 숫자는 30 epoch 뒤 distill 키를 벗긴 navtest PDM으로만 안다.

비교 기준은 `work_dirs/paradrive_distill_stage2_dual_distill/lightning_logs/version_0`의 epoch-29, PDMS 0.8539다.
