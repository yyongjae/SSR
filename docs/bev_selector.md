# BEV Selector Distillation

학습 중에만 도는 BEV register가 planner가 읽는 칸을 고르고, 그 칸을 모은 토큰에만 teacher를 맞춘다. Student 배포 그래프는 그대로다. 이 실험은 GradBalancer를 쓰지 않는다. Stage 1 adapter와 GT corridor도 쓰지 않는다.

**소속**: `/home/external-user/byounggun/SSR`  
**작성일**: 2026-09-28  
**상태**: v1 기록. 코드 기본값은 [`bev_selector_v2.md`](bev_selector_v2.md)다.  
**측정**: `r34_sel_trial_2` epoch 29, navtest PDMS **0.8343** (DAC fail 1030). v4는 0.8563.  
**기준 점수**: 같은 navtest 12146 토큰, distill 키를 벗긴 epoch 29. version_0 PDMS 0.8539, version_4 PDMS 0.8563 (DAC fail 764, TTC fail 681).

---

## 1. 한 줄

Selector는 추론용 새 backbone이 아니다. `bev_embed` 위에 얹는 학습 전용 register다. Bank마다 16개, 합 32개다. Object bank는 BEVFusion, map bank는 ReSMap을 읽는다. Attention은 planner가 실제로 보는 칸을 덮도록 학습한다. 학생의 plan / det / motion / map 손실은 그대로 두고, 그 뒤에 selector distill만 더한다. Distill은 그 attention으로 모은 토큰에만 걸리고, register를 움직이는 기울기와 feature를 맞추는 기울기는 분리한다. 그래서 칸마다 MLP를 얼리는 stage 1이 필요 없다.

---

## 2. Student는 그대로

배포와 학습의 student forward는 현재 `aux_distill` 기본과 같다.

- 카메라 `cam_f0`, `cam_l0`, `cam_r0`. 이 실험의 backbone은 `resnet34.tv_in1k`다. C5는 FPN에서 256차원으로 맞춘다. v4 명령의 student는 `resnet50.tv_in1k` 그대로다.
- BEVFormer 3층. `bev_embed`는 \(50\times 100\), 5000칸, `pc_range` \((-32, 0, -2, 32, 32, 2)\) m. 칸 한 변 0.64 m.
- `use_stl=false`. Dense planner, `plan_num_layers=3`. `use_task_interaction=true`. `use_ego_motion=false`.
- Planner는 command one-hot과 \([v_x, v_y, a_x, a_y]\)를 받고, 8스텝 \((x, y, heading)\)을 낸다. BEV cross-attention 뒤에 det와 map cross-attention이 같은 residual에서 갈라져 더해진다.
- Det, map, motion head와 plan / det / motion / map GT 손실은 유지한다. 손실 가중은 지금 코드 기본이다. plan 2.0, det 1.0, motion 1.0, map 1.0. 이 실험이 더하는 것은 \(\mathcal{L}_{\mathrm{cover}}\), \(\mathcal{L}_{\mathrm{div}}\), \(\mathcal{L}_{\mathrm{tok}}\) 뿐이다. 학생 손실을 distill로 갈아끼우지 않는다.
- Teacher는 지금과 같은 전방 캐시다. BEVFusion `cache_*_50x100`, ReSMap. 경로는 `/home/external-user/datasets/teacher_cache`. Teacher forward는 `no_grad`.

전방만 쓰는 것은 캐시를 학습 때 잘라서가 아니다. 학생과 teacher 캐시가 처음부터 같은 전방 박스다.

- 카메라는 `cam_f0`, `cam_l0`, `cam_r0`이다. `cam_b0`은 없다. 중앙 카메라 한 대로 줄인 것이 아니다. 좌우 전방 카메라는 남긴다.
- `pc_range`는 \((-32, 0, -2, 32, 32, 2)\) m이다. \(x\)는 우측 \([-32, 32]\), \(y\)는 전방 \([0, 32]\). 뒤쪽 반평면은 격자 밖에 있고, `y` 하한이 음수면 에이전트가 거부한다.
- 박스는 전방 32 m, 좌우 64 m, \(50\times 100\), 칸 0.64 m이다. BEVFusion manifest의 `point_cloud_range`는 mmdet3d 순서로 같은 박스를 적는다. `TeacherFeatureStore.validate_manifest`가 세로 \((0, 32)\), 가로 \((-32, 32)\)를 학생과 비교하고, 다르면 학습을 멈춘다. ReSMap sharded `meta.json`도 같은 검사다. 저장은 transpose와 좌우 뒤집기 뒤 \(50\times 100\)이다.
- Detection GT의 `det_fov_half_angle_deg=80`은 박스 라벨의 전방 콘이다. Teacher 텐서를 중앙 카메라 시야로 한 번 더 자르는 값이 아니다. Selector가 읽는 teacher도 이 전방 사각형 전체다.

뒤를 teacher에서 가져오지 않는 이유는 학생이 뒤를 보지 않기 때문이다. 뒤 칸을 맞추면 카메라가 설명할 수 없는 타깃이 `bev_embed`에 들어간다. 그 칸은 격자에도 없다.
- Selector와 distill 파라미터는 `agent._distill.*`에 두고, `scripts/evaluation/export_student_ckpt.py`가 지금 adapter 키를 벗기듯 벗긴다. 추론 때 planner는 5000칸을 모두 본다.

version_0부터 version_4까지의 체크포인트는 이 실험의 초기값이 아니다. 100×100이거나 interaction planner가 꺼져 있다. 처음부터 학습한다.

---

## 3. 이 실험에서 하지 않는 것

- **GradBalancer를 끈다.** `grad_balance_target=null`. `bev_embed`로 들어가는 plan / det / map 기울기를 0.4 / 0.3 / 0.3으로 다시 맞추지 않는다. `para_ssr_agent.yaml`에 적힌 측정은, 이 밸브를 끄면 실제 navsim 배치에서 기울기 비중이 plan 0.25%, det 73.5%, map 26.2%였다는 것이다. 그 비중을 이 실험은 그대로 둔다. 로그에는 태스크별 `bev_embed` 기울기 노름을 남기되, 스케일을 곱하지 않는다. Aux가 plan보다 크면 그 사실로 남긴다. 대응은 고정된 손실 가중이지 GradBalancer가 아니다.
- Stage 1A / 1B adapter를 학습하지 않고, 얼린 `PlanningBEVAdapter`를 불러오지도 않는다.
- GT 궤적 Gaussian corridor, role mask, walkway suppress를 distill 가중으로 쓰지 않는다.
- CWD, relation, attention KD, adaptive scale, head KD, planning-look KL, plan tail, TTC proxy는 가중 0이다.
- Distill 항은 GradBalancer 안에도, 태스크 손실의 재가중 루프 안에도 넣지 않는다. 태스크 손실 합 뒤에 더한다.
- 추론에서 토큰을 버리거나 planner를 16개 토큰만 보게 바꾸지 않는다. ViT, DINOv2, LoRA, 4번째 카메라, trajectory vocabulary, PDMS sub-score head를 가져오지 않는다.

---

## 4. 세 논문이 설계에 주는 것

### 4.1 SSR, Navigation-Guided Sparse Scene Representation

Li and Cui, ICLR 2025. arXiv:2409.18341.

Scene TokenLearner는 navigation command를 Squeeze-and-Excitation으로 BEV에 섞고, 칸마다 attention을 만들어 16개 토큰으로 모은다.

\[
B^{\mathrm{navi}} = \mathrm{SE}(B, \mathrm{cmd}), \qquad
s_k = \rho\big(B^{\mathrm{navi}} \odot \varpi_k(B^{\mathrm{navi}})\big)
\]

Planner는 그 16개만 cross-attention한다. 미래 BEV를 복원하는 분기는 학습 중에만 있고 추론에서는 빠진다.

가져오는 것: command가 선택을 조건화한다. 선택은 BEV 칸의 attention이고, 모은 토큰이 teacher와 비교할 단위다.

두지 않는 것: perception head를 지우는 선택. 우리 student의 det / map head와 task interaction은 유지한다. 미래 BEV 복원 손실은 이 실험의 기본 가중을 0으로 둔다.

### 4.2 Prune2Drive

Xiong et al., arXiv:2508.13305.

점수만으로 토큰을 고르면 한 물체와 위치 bias로 몰린다. T-FPS는 이미 고른 집합과 cosine 거리가 가장 먼 토큰을 반복해 집어, 공간과 의미의 coverage를 유지한다. 뷰별 유지 비율은 하류 task 점수로 정한다.

가져오는 것: selector가 planner attention의 최댓값 한 점에 겹치지 않게 하는 differentiable repulsion. \(\sigma\)는 3–4 m.

두지 않는 것: 학습 없는 추론 pruning, VLM 시각 토큰, 뷰별 비율 탐색. 우리 입력은 전방 3카메라가 이미 만든 BEV다.

### 4.3 Driving on Registers (DrivoR) — selector에 쓰는 부분

Kirby et al., CVPR 2026. [Driving on Registers](https://openaccess.thecvf.com/content/CVPR2026/papers/Kirby_Driving_on_Registers_CVPR_2026_paper.pdf).

판단: **register 감각은 selector 설계에 맞다.** 다만 DrivoR의 시스템 전체를 가져오지는 않는다. 그들은 BEV를 없애고, 카메라마다 register를 ViT 안에 넣어 추론 표현 자체를 \(N \times R\) 토큰으로 줄인다. 우리는 BEVFormer와 dense planner를 유지하고, register를 `bev_embed` 위의 학습 전용 슬롯으로만 쓴다.

DrivoR에서 확인된 사실:

- Register는 ViT patch와 self-attention으로 섞인 뒤, patch는 버리고 register만 scene token으로 남긴다. Perceiver처럼 바깥 cross-attention으로 압축하는 구조와 다르다고 논문이 명시한다. 압축 토큰이 encoder 안에서 입력을 읽는다.
- 카메라마다 register를 따로 둔다. 기본 16개. 16개와 32개 사이에서 navval PDMS가 정체해서 16개를 골랐다. 토큰을 더 넣는다고 계속 오르지 않는다.
- DINOv2가 원래 가진 4개 register는 attention sink라서 버리고, 주행용 register는 \(N(0, 10^{-6})\)으로 새로 초기화한다. Sink로  pretrained된 register를 쓰면 더 나빴다. 이미 특화된 슬롯은 주행 슬롯의 초기값으로 나쁘다.
- navval에서 register끼리 cosine을 재면 전방 카메라 register는 서로 거의 상관없고, 후방 카메라는 한 토큰을 빼고 붕괴한다. 마지막 layer attention에서 전방 register는 신호등, 앞차, 도로 경계처럼 서로 다른 영역을 본다.
- 궤적 decoder와 score decoder가 같은 토큰 경로를 쓰면 navval PDMS 84.7이다. Decoder를 나누고 score 기울기를 궤적 decoder에서 끊으면 90.0이다. 한 토큰 경로에 두 목적을 섞으면 둘 다 약해진다.
- Frozen backbone에서 register만 학습하는 쪽은 LoRA finetune보다 낮다. Register는 읽기 전용 feature 위의 얇은 머리로도 동작은 하지만, 논문의 최고점은 backbone이 같이 움직일 때다. 우리 설정에서는 student encoder가 학습되므로 이쪽에 가깝다. Register만 학습하고 `bev_embed`를 얼리는 ablation은 하지 않는다.

Selector에 반영하는 네 가지:

1. **슬롯은 지속되는 register다.** 매 forward마다 BEV에서 attention map을 새로 그리는 TokenLearner만 쓰면, 배치 사이에서 "3번 슬롯"의 정체가 없다. DrivoR은 슬롯 정체가 있어야 특화가 측정된다. 우리 register는 학습되는 임베딩이고, 입력에 따라 BEV 칸을 cross-attention한다. 정체는 파라미터가 잡고, 어디를 볼지는 입력이 정한다.
2. **Sink나 GT corridor로 초기화하지 않는다.** DrivoR이 DINOv2 sink register를 버린 것과 같다. \(N(0, 10^{-6})\)에서 시작하고, command modulation이 초반의 보는 방향을 만든다.
3. **특화는 손실이면서 로그다.** 전방 register가 서로 다른 영역을 본 것이 논문의 정성 결과다. 우리는 attention map끼리의 cosine과 공간 평균 \(\mu_k\)의 거리를 손실로 두고, register 출력끼리의 cosine은 로그만 한다. 출력 cosine을 손실로 두면 teacher를 맞추는 손실과 싸운다.
4. **두 teacher는 두 register bank다.** DrivoR에서 궤적과 score가 서로 다른 카메라를 본다. 한 bank로 BEVFusion과 ReSMap을 동시에 맞추면, 물체 채널과 지도 채널을 한 벡터에 넣게 된다. Bank를 나누는 것은 그 분리의 BEV 버전이다.

---

## 5. 모듈

Train-only `BEVRegisterSelector`. 입력은 planner가 이미 받는 것과 같다.

- Student `bev_embed` \([B, 5000, 256]\).
- Command one-hot.
- Ego status \([v_x, v_y, a_x, a_y]\).

Teacher BEV는 고르는 데 쓰지 않는다. Teacher는 모을 때의 값으로만 들어온다.

Register는 bank당 16개, 합 32개다. Object bank 16개는 BEVFusion 캐시를 읽고, map bank 16개는 ReSMap 캐시를 읽는다. 학습 임베딩의 초기 분포는 \(N(0, 10^{-6})\)이다. 그것만 두면 32개 attention이 같은 맵이 되고 diversity 기울기가 0이다. 그래서 각 슬롯 로짓에 전방 격자 좌표를 더한다. 4×4, 좌우 \([-24, 24]\) m, 전방 \([4, 28]\) m, 폭 6 m. 지도 bank는 물체 bank보다 격자 간격의 절반만큼 어긋난다. GT 궤적이나 GT 박스는 시작 위치가 아니다. 이 좌표 항의 세기는 step 0에서 1이고, token ramp와 같이 줄어 0.3에서 멈춘다. 0까지 내리지 않아서 슬롯이 다시 한 점에 붙지 않는다.

16은 칸 수가 아니다. Register 하나가 5000칸 위의 attention이다. SSR과 DrivoR의 16은 planner가 읽는 장면 토큰 전체였다. 우리 planner는 5000칸을 그대로 보고, register는 distill을 담는 통이다. 통이 너무 적으면 앞 32 m × 좌우 64 m 안의 차가 한 토큰에 섞인다. Bank당 8개는 그 쪽에 걸친다. DrivoR은 카메라당 16과 32 사이에서 점수가 멈췄다. 32를 넘기면 diversity가 빈 구석으로 슬롯을 밀고, \(w_k\)가 그 슬롯을 꺼서 손실만 늘어난다. 그래서 기본은 bank당 16이고, ablation은 bank당 8, 16, 32이다. 추론 비용은 어느 쪽이든 0이다.

Command와 ego status는 register에 더해지는 조건이다. SSR의 SE가 BEV 전체에 command를 섞는 것과 같은 역할이고, 자리는 register 쪽이다. 직진과 좌회전이 같은 슬롯이라도 다른 칸을 보게 하려는 것이다. DrivoR Fig. 5에서 score 경로가 기동에 따라 다른 카메라를 보는 것에 대응한다. 우리는 후방 카메라가 없고 BEV가 전방 32 m이므로, 기동은 BEV 안의 좌측과 중앙을 옮긴다.

각 register가 `bev_embed`를 cross-attention한다.

\[
A_k = \mathrm{softmax}\big(q_k K^\top / \sqrt{d}\big), \qquad
z_k = A_k V
\]

\(q_k\)는 조건이 섞인 register, \(K, V\)는 `bev_embed`의 선형 투영이다. \(A_k\)는 \([B, 5000]\)이다. 같은 \(A_k\)로 student `bev_embed`와 align된 teacher BEV를 모은다. BEVFusion bank는 BEVFusion 캐시만, ReSMap bank는 ReSMap 캐시만 모은다. 그리드가 이미 \(50\times 100\)이면 resample은 없다.

Selector 파라미터와 이 attention은 추론 그래프에 없다.

---

## 6. 기울기를 나누는 방식

Stage 1이 있던 이유는 구역을 몰라서가 아니다. Teacher 칸과 student 칸의 256차원이 같은 뜻이 아니고, 칸마다 MLP를 student와 같이 학습하면 그 MLP가 차이를 흡수하기 때문이다. Adapter를 stage 2에서 얼리던 일이, 여기서는 기울기 분리로 바뀐다.

### 6.1 Register가 배우는 것

\(A\)와 register 임베딩은 distill MSE의 기울기를 받지 않는다. \(A\)는 detach된 뒤 토큰을 모은다.

Register의 손실은 두 개다.

**Cover.** Dense planner 마지막 layer에서 plan query가 5000칸에 주는 attention \(P\)를 detach한다. \(A\)들의 합이 \(P\)를 따르게 한다. 타깃은 초반에 command prior와 섞는다.

\[
T = (1-\lambda)\, T_{\mathrm{cmd}} + \lambda\, P
\]

\(\lambda\)는 epoch 0에서 0, epoch 5에서 1이다. \(T_{\mathrm{cmd}}\)는 GT 궤적이 아니다. Command 방향의 넓은 전방 쐐기다. 직진은 전방 사각형, 좌회전은 좌전방이다. 바닥은 얇게 둔다. GT 궤적 Gaussian을 \(T\)에 넣는 순간 이 모듈은 지금 corridor의 학습판이 된다.

**Diversity.** Prune2Drive의 farthest-point를 미분 가능하게 둔다. \(A_k\)의 공간 평균을 \(\mu_k\)라 하면

\[
\mathcal{L}_{\mathrm{div}}
= \sum_{k < k'}
\exp\big(-\lVert \mu_k - \mu_{k'} \rVert^2 / 2\sigma^2\big),
\quad \sigma = 4\,\mathrm{m}.
\]

합은 두 bank를 합친 32개 전체에 걸린다. \(P\)가 앞차 한 점에 몰려도 register 하나가 그 봉우리를 맡고, 나머지는 \(P\)의 가장자리로 퍼진다. \(P\) 밖 배경으로 밀리지는 않는다. Coverage의 기준이 \(T\)이기 때문이다. Attention map끼리의 cosine도 같은 이유로 작게 유지한다.

Object bank와 map bank의 \(\mu\)가 약 1 m 안에서 겹치는 것은 허용한다. 차 한 대가 차선 위에 있으면 두 teacher가 같은 칸을 볼 수 있다. 손실은 같은 칸에 여러 register가 겹치는 경우만 민다.

### 6.2 Feature가 배우는 것

\(A_k.\mathrm{detach()}\)로 student 토큰과 teacher 토큰을 모으고, 각각 LayerNorm한 뒤 MSE를 건다.

\[
\mathcal{L}_{\mathrm{tok}}
= \sum_k w_k
\big\lVert \mathrm{LN}(z_k^{s}) - \mathrm{LN}(z_k^{t}) \big\rVert^2
\]

\[
w_k = \langle A_k, P \rangle
\]

\(w_k\)도 detach한다. Planner가 읽지 않는 register는 distill 기울기를 거의 받지 않는다. 기울기는 student `bev_embed`의 선택된 칸으로만 간다. Teacher, register, \(A\), LayerNorm의 affine은 학습하지 않는다. LayerNorm은 채널 스케일을 지우기 위한 고정 정규화다. 칸마다 다른 Linear는 없다.

이 경로에서 손실이 주는 방향은 하나다. 선택된 칸의 student feature가 teacher와 비슷해진다. Register가 맞추기 쉬운 배경으로 옮겨 손실을 줄이는 경로와, MLP가 정답을 가리는 경로는 닫혀 있다. DrivoR이 score 기울기를 궤적 decoder에서 끊은 것과 같은 종류의 절단이다. 목적은 다르다. 그들은 생성과 채점을 나누고, 우리는 고르기와 맞추기를 나눈다.

선택되지 않은 칸의 distill 기울기는 0이다. Plan, det, motion, map 손실은 전 칸에 그대로 흐른다. GradBalancer가 없으므로 그 비중은 손실 값과 head 구조가 정한 그대로다.

### 6.3 LayerNorm이 부족할 때

처음 1000 step에서 pooled token cosine이 우연 수준이면, 채널 기저가 LayerNorm만으로 안 맞는 것이다. 그때의 보완은 stage 1을 되살리는 것이 아니다. Teacher 캐시 일부로 한 번 고정한 공용 Linear 256→256을 양쪽에 곱한다. 그 행렬은 학습하지 않는다. 이 보완은 cosine이 안 움직일 때만 켠다. 기본 실험은 LayerNorm만 쓴다.

---

## 7. 학습 한 단계

Teacher 캐시를 읽고 student와 selector를 한 번의 학습으로 돌린다. Stage 1 스크립트는 이 실험의 진입점이 아니다.

| 항 | 기울기 | 가중 스케줄 |
| --- | --- | --- |
| plan / det / motion / map | 지금 head와 `bev_embed`. 스케일 1 | 고정. plan 2, det 1, motion 1, map 1 |
| \(\mathcal{L}_{\mathrm{cover}}\) | register와 \(A\)를 만드는 투영 | epoch 0부터. 타깃은 \(\lambda\)로 prior에서 \(P\)로 이동 |
| \(\mathcal{L}_{\mathrm{div}}\) | 같은 selector 경로 | epoch 0부터. \(\sigma=4\) m |
| \(\mathcal{L}_{\mathrm{tok}}\) | student `bev_embed`만 | epoch 0에 0, epoch 5에 1.0 |

Epoch 수는 30, batch 4, accumulate 16, lr \(10^{-4}\). 지금 stage 2와 같은 길이로 두고, 점수가 오르기 전에 epoch를 늘리지 않는다. 체크포인트는 5 epoch마다와 `last.ckpt`.

미래 BEV 복원(SSR의 train-only predictor)은 구현해 두더라도 가중 0이다. 기본 실험이 version_4의 DAC / TTC fail을 줄인 뒤에만 별도 ablation으로 켠다.

---

## 8. 로그

W&B와 TensorBoard에 태스크 손실과 함께 다음을 남긴다.

- \(\mathcal{L}_{\mathrm{tok}}\), \(\mathcal{L}_{\mathrm{cover}}\), \(\mathcal{L}_{\mathrm{div}}\). Bank별 token cosine.
- \(A_k\) 질량이 0.01 이상인 칸의 비율, \(A_k\) entropy, \(\mu_k\) 사이 최근접 거리.
- 두 bank 각각의 register 출력 cosine 행렬. DrivoR Fig. 3과 같은 그림이다. 전방이 붕괴하면 selector가 한 슬롯으로 죽은 것이다.
- Plan query attention \(P\)와 \(\sum_k A_k\)의 cosine.
- `bev_embed`에 대한 plan, det, map, distill 기울기 노름. 재가중은 하지 않는다. 비중을 나중에 읽기 위한 기록이다.
- 몇 장면에 대해 \(A_k\)를 BEV 위에 그린 그림. DAC가 깨진 장면에서 중심선이 아니라 그 실패의 agent와 road edge에 질량이 있는지를 본다.

---

## 9. 실험

채점은 지금 navtest 절차다. 12146 토큰, distill 키 제거, epoch 29, GPU 4와 5로 샤드를 나누는 기존 스크립트. 에이전트는 학습을 시작하지 않는다. 돌릴 명령은 구현 뒤에 따로 적는다.

성공은 version_4 epoch 29의 PDMS 0.8563을 넘는 것이다. 볼 항은 DAC fail과 TTC fail이다. 합법이고 TTC가 1인 장면의 EP는 version_4에서 이미 0.869로 version_2와 같았다. 전체 BEV cosine이 올라도 DAC fail이 그대로면 이 selector는 점수를 옮기지 못한 것이다.

| ID | 설정 | 확인하는 것 |
| --- | --- | --- |
| S0 | 전 칸 MSE. Register 없음. Adapter 없음. GradBalancer 없음 | 선택 없이 맞출 때의 바닥 |
| S1 | 지금 recipe. Stage-1 adapter, GT corridor, GradBalancer 켜짐 | 이미 측정된 0.8563. 다시 학습하지 않는다 |
| S2 | 이 문서의 기본. Register 16+16, 학생 손실 유지, LN 토큰 MSE, GradBalancer 없음, corridor 없음 | 본 실험 |
| S3 | S2에서 \(\mathcal{L}_{\mathrm{div}}\) 제거 | peak가 한 칸으로 모이고 DAC / TTC가 나빠져야 한다 |
| S4 | S2에서 distill 기울기를 \(A\)에 허용 | register가 맞추기 쉬운 칸으로 이동하는지 |
| S5 | bank를 합쳐 32개가 두 teacher를 둘 다 맞춤 | 두 teacher를 한 슬롯에 넣는 비용 |
| S6 | bank당 \(K \in \{8, 16, 32\}\) | 8이 차를 섞는지, 32가 빈 슬롯을 만드는지 |
| S7 | \(T_{\mathrm{cmd}}\) 대신 GT 궤적 Gaussian | corridor를 정답으로 주면 S2와 얼마나 같아지는지 |
| S8 | S2와 같고 GradBalancer만 0.4 / 0.3 / 0.3 | 밸브를 끈 효과와 selector 효과를 분리. S2가 본 실험이고 S8은 그 다음이다 |

S2가 0.8563 근처에서 멈추고, 토큰 cosine만 오르고 DAC fail이 안 줄면, 남은 오차는 selector가 고를 BEV 내용 밖에 있다. 그 결론을 숫자로 남기고 recipe를 더 쌓지 않는다.

---

## 10. 구현할 때 손대는 자리

이 문서는 계획이다. 아래는 구현 시 위치만 고정한다.

- 새 모듈은 `navsim/agents/para_ssr/distill/selector.py`다. `distill_selector=false`가 기본이라 v4 경로의 adapter와 corridor는 그대로다.
- Agent yaml은 `navsim/planning/script/config/common/agent/para_ssr_selector_agent.yaml`이다. 실행은 `scripts/training/run_bev_selector_distill.sh`다.
- `PlanningDistillation`의 align → frozen adapter → corridor MSE 경로를, 이 실험에서 register pool → LN → token MSE로 바꾼다. 기존 경로는 플래그로 남기고 기본값을 이 실험으로 바꾸지는 않는다. 지금 돌아가는 학습과 version_4 재현을 이 변경이 바꾸면 안 된다.
- `grad_balance_target`은 이 실험 yaml에서 `null`이다. 코드 기본값 0.4 / 0.3 / 0.3은 다른 실험용으로 둔다.
- Export는 `agent._distill.*` 제거 규칙을 그대로 쓴다. Register 키는 `agent._distill.selector.*`다.

## 11. 실행

v4 recipe는 이 실험과 체크포인트 디렉터리를 공유하지 않는다. 나중에 같은 명령으로 다시 돌린다.

```bash
cd /home/external-user/byounggun/SSR
FORCE_RETRAIN=stage1a,stage2 bash ./scripts/training/run_all_stages_distill.sh
```

Selector 실험은 한 단계다. Backbone은 `resnet34.tv_in1k`다. Stage 1 adapter를 읽지 않고, GradBalancer가 꺼져 있으며, 출력은 `work_dirs/paradrive_distill_bev_selector/`다. W&B run 이름은 `r34_sel_trial_숫자`다.

```bash
cd /home/external-user/byounggun/SSR
bash ./scripts/training/run_bev_selector_distill.sh
```

`tests/test_bev_selector.py`는 token 손실이 register로 역전파되지 않는 것, v4 yaml의 GradBalancer와 corridor가 그대로인 것을 본다.
