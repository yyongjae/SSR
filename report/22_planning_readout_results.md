# 22. Planning readout 실험 결과: ReSMap teacher → PARA-SSR

2026-09-16 ~ 09-19, 5090 서버(blackwell64). 브랜치 `km/planning-readout` (v2는 worktree `~/kyungmin/SSR-v2`, 브랜치 `km/para-ssr-v2`).
설계와 실험 프로토콜은 [report/19](19_planning_readout.md)에 있다. 이 문서는 **실제로 돌린 것, 결과, 결과가 뒤집은 가정, 그래서 바꾼 방법**을 처음부터 정리한다.
모든 수치는 navtest(12,146 장면)이고, 따로 적지 않으면 학습 seed 0 하나다.

---

## 0. 요약

1. **teacher BEV에는 planning에 쓰이는 정보가 있다.** 같은 readout으로 읽으면 BEV를 안 읽는 기준선보다 PDMS가 +16.9 높다 (h1: `S_own` 81.49, `S_ego` 64.59).
2. **z 증류는 의도대로 동작했지만 PDMS가 오르지 않았다.** student의 장면별 z가 teacher 쪽으로 8배 정렬됐는데(cos 0.057 → 0.443), PDMS는 대조군과 같았다(84.74 vs 84.74).
3. **원인은 z라는 통로다.** BEV 칸 단위로 재면 teacher BEV가 map 정보를 훨씬 많이 가진다(road/walkway/centerline/crosswalk IoU 92/66/42/62 vs student 81/39/23/27). 그런데 z로 압축하면 teacher와 student가 거의 같아진다(76/37/23/28 vs 74/34/22/26). z에 실리지 않는 차이는 z 증류로 옮길 수 없다.
4. **query 수를 늘려도(Nq 4~32) 통로가 열리지 않는다.** z에서의 teacher-student 격차는 2~6점에 머물고 `S_own`도 오르지 않는다.
5. **그래서 전달 방식을 바꿨다.** readout이 "어디를 보는지"(attention)로 칸을 고르고, BEV를 칸 단위로 맞추는 `attn_feature`를 구현했다. attention은 약 900칸을 덮고 road·centerline·crosswalk에 1.7배 집중되며 command 방향을 따른다.
6. **BEV feature 증류도 map 정보를 옮기지 못했다.** `attn_feature` 84.76, 전체 BEV `kd_feature` 84.50 (대조군 84.74). 증류 후 student BEV의 map probe도 teacher 쪽으로 전혀 오르지 않았다(§9).
7. **PDMS 실패의 대부분은 인지가 아니라 planning 쪽이다.** DAC 실패 중 map head가 도로로 잘못 본 경우는 3.6%, NC/TTC 실패 중 det가 놓친 경우는 3.5%/13%다. DAC 실패는 경계를 0.14 m 스치는 수준이고, 이탈 지점의 80%에서 모델의 map head는 경계를 이미 0.6 m 안쪽으로 보고 있었다(§13).
8. **v1은 heading 출력에 설계 결함이 있다.** step별 heading 변화량을 누적해 heading이 경로 방향에서 벗어나고, PDM의 LQR 추종이 이를 따라간다. 평가 때 heading을 경로 방향으로 다시 계산하면 학습 없이 PDMS가 오른다: interaction_final 84.87 → 85.93, 대조군 84.74 → 85.61. TOAD 식 운동 모델 투영은 반대로 82.23으로 떨어진다(§14). 논문 기여는 아니고 기준선 보정이다.
9. **PARA-SSR v2를 새로 학습 중이다(§15).** WoTE의 anchor 256개 + PDM 점수 loss + DiffusionDriveV2의 Bézier heading, ResNet-34/50 두 벌, interaction_final과 같은 레시피로 처음부터 30 epoch.

---

## 1. 무엇을 하려는가 (report/19 요약)

> map module teacher(ReSMap)의 BEV에서 **planning에 쓰이는 성분만** 골라 e2e 모델(PARA-SSR)의 BEV로 옮기고, 그 결과로 planning 성능을 올린다. map mAP 향상은 부산물이다.

- **readout `h = h_dec ∘ h_enc`**: BEV를 읽어 planning 요약 z를 만들고(`h_enc`), ego를 더해 4초 궤적을 낸다(`h_dec`). teacher BEV로 학습한 `h_enc`를 얼려 두고 증류에 쓴다.
- **구조** (`navsim/agents/para_ssr/readout/readout.py`)

```
 BEV 256 x 50 x 100 ─ in_proj(1x1) + pos_mlp(칸 중심 m) ─► 토큰 5,000개
 query(Nq개) + cmd_embed ─► [cross-attention (+FFN)] x L ─► z (Nq x 256)
 z + ego_mlp ─► LayerNorm ─► head ─► 궤적 8 x 3
```

| preset | L | FFN | head | 파라미터 |
|---|---|---|---|---|
| h0 | 1 | 없음 | Linear | 0.47M |
| h1 | 1 | 있음 | MLP 2층 | 0.80M |
| h2 | 3 | 있음 | MLP 2층 | 1.86M |

- **측정량**: `S_own`(teacher BEV로 학습, teacher BEV로 평가), `S_ego`(BEV 없음), `S_student`(student BEV로 학습·평가), `S_transfer`(teacher h를 학습 없이 student BEV에), `S_transfer+A`(teacher h 고정, 1×1 adapter만 학습).

---

## 2. 환경과 데이터

| 항목 | 위치 / 내용 |
|---|---|
| 서버 | blackwell64, RTX 5090 × 6 (우리 작업은 GPU 0~3), `ssr` env torch 2.8.0+cu128 |
| 데이터 경로 | `data/` 아래 symlink (`dataset/{maps,navsim_logs,sensor_blobs}/{trainval,test}`, `exp/metric_cache`) |
| teacher 캐시 | `~/datasets/teacher_cache/resmap` (navtrain train_logs 126,032 frame, 301 GB) + `resmap/navtest` (12,146) |
| student 캐시 | `~/datasets/teacher_cache/student_interaction_final` (navtrain 103,288 + navtest) 외 fine-tune 체크포인트별 navtest |
| plan target | `data/readout/plan_targets_{navtrain,navtest}.npz` |
| GT map raster | `data/readout/map_rasters_navtest.npz` (4클래스, student BEV 격자, 측정 전용) |
| navtrain 커버리지 | navtrain 103,288 중 train_logs 85,109 = teacher 캐시 token과 정확히 일치. 나머지 18,179는 val_logs |

운영 중 겪은 것:
- navtest teacher 캐시 다운로드가 느렸다. 같은 계정의 다른 다운로드가 회선(약 100 Mbps)을 연결 32개로 쓰고 있었다. 병렬 range 요청(연결 32개)으로 다시 받고 sha256으로 검증했다.
- **OOM**: 같은 계정의 모든 프로세스가 cgroup 메모리 한도를 함께 쓴다. KD 학습이 두 번 OOM killer에 죽었다(9/17 15:34 재개로 복구, 21:18 `kd_readout_adapter_ft`는 epoch 1에서 중단). 학습 위에 캐싱을 얹거나 BEV 전체를 메모리에 올리는 스크립트가 원인이었다.

---

## 3. Stage 1: teacher BEV에 planning 정보가 있는가

navtrain train_logs 85,109 frame(log 단위 5% hold-out, 학습 80,507 / 검증 4,602), 10 epoch, seed 3개.

| preset | `S_own` | `S_ego` | `S_own − S_ego` | `S_transfer` | 검증 L2@4s teacher / ego (m) | BEV를 다른 장면 것으로 바꿨을 때 |
|---|---|---|---|---|---|---|
| h0 | 76.96 ± 0.49 | 63.28 ± 0.14 | +13.68 | 31.51 ± 2.11 | 1.538 / 2.188 | 3.77 |
| h1 | 81.49 ± 0.21 | 64.59 ± 0.19 | **+16.90** | 43.73 ± 1.99 | 1.364 / 2.024 | 4.08 |
| h2 | 82.06 ± 0.12 | 64.68 ± 0.31 | +17.38 | 42.58 ± 3.69 | 1.333 / 2.024 | 4.35 |
| h1 cmd-late | 80.82 ± 0.07 | 64.59 | +16.23 | – | – | – |

- 관문(report/19 §8.3)을 통과한다. `S_own − S_ego`가 seed 편차(±0.2)보다 수십 배 크고, BEV를 섞으면 오차가 ego보다도 나빠진다.
- 용량은 h1에서 거의 포화한다(h1 → h2 +0.57).
- **`S_shuffled`(라벨을 섞어 학습)는 버렸다.** 정답을 섞어 학습한 모델은 테스트에서 당연히 낮고(약 44), 높게 나오면 파이프라인 버그라는 뜻일 뿐이다. BEV 정보량에 대한 정보가 없다. 결과는 `data/readout/runs_discarded/`로 옮겼다.
- seed 편차가 작아서(`S_own` ±0.21, `S_student` ±0.04) 이후 실험은 seed 0 하나로 진행한다.

---

## 4. Stage 2: student BEV와 비교 (h1)

student는 `para_ssr_interaction_final`(navtest PDMS 84.87)의 BEV다.

| 측정 | NC | DAC | EP | TTC | PDMS |
|---|---|---|---|---|---|
| `S_ego` | 92.92 | 76.54 | 61.80 | 83.17 | 64.59 ± 0.19 |
| `S_own` | 96.66 | 91.54 | 76.62 | 90.93 | 81.49 ± 0.21 |
| `S_student` | 97.26 | 91.41 | 77.55 | 92.06 | **82.32 ± 0.04** |
| `S_transfer` | 77.77 | 67.05 | 44.58 | 63.24 | 43.73 ± 1.99 |
| `S_transfer+A` | 96.84 | 88.64 | 74.99 | 90.96 | 79.29 ± 0.46 |

- report/19 §2.3 표의 **"`S_student` ≈ `S_own`, `S_transfer` 낮음 → 정렬 문제"** 행에 해당한다. 1×1 adapter 하나로 43.7 → 79.3까지 회복된다. 두 BEV는 채널 배치가 다를 뿐이다.
- 이 시점의 해석은 "student BEV에 teacher만 가진 planning 정보가 없다"였다. **이 해석은 §7에서 z 기준 측정의 한계로 밝혀졌다.**
- 장면 단위로는 두 BEV가 실패하는 장면이 다르다: student 실패 & teacher 성공 578, 그 반대 838, 둘 다 실패 518.

---

## 5. Stage 3: z 증류 fine-tune

### 5.1 설정
- **student base를 `interaction_final`로 바꿨다.** 현재 구조(전방 카메라 3대, LiDAR 없음)에서는 full 모델이 가장 좋다(plan only 83.98, parallel 84.03, interaction 84.87). report/19 초판의 plan-only 기준은 이전 LiDAR 구조 수치(86.45)에 근거한 것이었다.
- 5 epoch, LR 2e-5(base 스케줄의 epoch 22 수준), warmup 0, 매 epoch 검증, GPU 2장, global batch 128.
- 증류 강도는 GradBalancer로 정한다. base target `plan:0.4, det:0.3, map:0.3`에 `distill:0.1`(`KD_SHARE`)만 더해, det/map scale은 대조군과 같고 distill gradient 크기는 plan의 25%로 맞춘다. 고정 λ=1이면 distill 몫이 91%까지 커지는 것을 스모크에서 확인했다.
- 대조군은 같은 조건에서 증류만 뺀 fine-tune이다. LR을 다시 올리고 더 학습한 효과를 뺀다.

### 5.2 결과

| run | 설정 | NC | DAC | EP | TTC | **PDMS** | det mAP | map mAP |
|---|---|---|---|---|---|---|---|---|
| base | 30 epoch | 97.86 | 93.31 | 79.77 | 93.56 | 84.87 | 27.07 | 27.23 |
| 대조군 | 증류 없음 | 97.80 | 93.18 | 79.59 | 93.55 | **84.74** | 27.44 | 27.74 |
| KD | adapter 없음 | 97.78 | 92.85 | 79.09 | 93.66 | **84.38** | 27.41 | 27.70 |
| KD + adapter + B | adapter, `KD_CENTER=command` | 97.76 | 93.20 | 79.58 | 93.68 | **84.74** | 27.43 | 27.69 |
| KD + adapter | – | epoch 1에서 OOM으로 중단 | | | | | | |

대조군과 같은 장면끼리 짝지은 차이(bootstrap 2,000회, 평가 장면 표본의 흔들림만 반영):

| run | 차이 | 95% CI | 나은 장면 / 나쁜 장면 | DAC 위반 해결 / 새로 위반 |
|---|---|---|---|---|
| KD | −0.362 | [−0.560, −0.145] | 2,614 / 4,778 | 57 / 97 |
| KD + adapter + B | +0.007 | [−0.179, +0.188] | 4,115 / 3,204 | 62 / 60 |

- adapter 없이 증류하면 student BEV의 채널 배치를 teacher 쪽으로 밀어서 손해를 본다(−0.36).
- adapter와 B(평균 제거)를 켜면 손해는 사라지지만 이득도 없다. DAC 위반은 해결과 악화가 62 대 60으로 방향성이 없다.
- teacher readout이 student보다 강한 상위 20% 장면(2,429개)만 봐도 PDMS 71.74 vs 71.66, DAC 84.77 vs 84.64다.

### 5.3 B(평균 제거)가 필요했던 이유: z의 구성

| h1 z 성분 (navtest 1,000 장면) | 에너지 비중 |
|---|---|
| 모든 장면 공통 방향 | 79.3% |
| command별 추가 성분 | 15.8% |
| 장면별 성분 | **4.9%** |

- z는 command를 query에 넣기 때문에 대부분 command로 정해진다. 다른 장면이라도 command가 같으면 cos 0.949로, 같은 장면의 student-teacher(0.898)보다 가깝다.
- 그대로 cosine 거리를 쓰면 증류 loss의 약 60%가 command별 평균을 맞추는 데 쓰인다. `KD_CENTER=command`는 각자의 command별 이동평균을 빼고 장면별 성분만 비교한다.
- cmd-late(A, command를 z 뒤로)는 command 성분을 15.8% → 5.2%로 줄였지만, 공통 방향이 85.6%로 커지고 `S_own`이 0.7점 떨어졌다. adapter 없이 student z와 teacher z가 장면을 구분하지 못해서(cos 0.835 vs 0.837) 이번 증류에는 쓰지 않았다.

### 5.4 증류가 의도대로 동작했는가

navtest 512 장면에서 fine-tune된 체크포인트의 BEV를 직접 비교했다.

| | base 대비 BEV 변화 | teacher z와 cos (원본) | teacher z와 cos (command 평균 제거) |
|---|---|---|---|
| base | – | 0.897 | 0.059 |
| 대조군 | 5.6% | 0.897 | 0.057 |
| KD + adapter + B | **17.1%** | 0.927 | **0.443** |

**증류는 설계대로 동작했다.** BEV가 움직였고 장면별 z가 teacher 쪽으로 8배 정렬됐다. 그런데도 PDMS와 map mAP가 그대로였다.

---

## 6. 무엇이 문제였나: z로 재면 보이지 않던 map 정보 차이

z를 거치지 않고, BEV 칸 단위로 GT map raster(nuPlan map API, 4클래스, student BEV 격자)를 얼마나 복원할 수 있는지 probe로 쟀다(`data/readout/probe_map_full.py`).
navtest 12,146 장면을 log 단위로 8:2 분리했고(학습 9,475 / 검증 2,671), BEV는 얼려 두고 probe(3×3 conv 2층)만 학습했다.

| 입력 | road | walkway | centerline | crosswalk |
|---|---|---|---|---|
| **teacher BEV** | **92.4** | **65.8** | **42.3** | **62.2** |
| student base BEV | 81.4 | 38.7 | 22.5 | 27.2 |
| 대조군 BEV | 81.6 | 38.7 | 22.5 | 26.6 |
| KD BEV | 81.6 | 37.9 | 22.6 | 26.6 |
| KD + adapter + B BEV | 81.6 | 39.2 | 22.5 | 27.0 |
| teacher z (h1, Nq=1) | 75.9 | 36.6 | 23.3 | 27.5 |
| student base z (같은 h1) | 74.1 | 33.6 | 22.2 | 26.2 |

- **teacher BEV는 map 정보가 훨씬 많다.** road +11, walkway·centerline·crosswalk는 약 2배다.
- **그 차이는 z를 지나면서 거의 사라진다.** teacher z와 student z의 격차는 +1.8 / +3.0 / +1.1 / +1.3뿐이다. §4에서 `S_own` ≈ `S_student`로 보인 이유가 이것이다. z는 4초 궤적을 맞추도록 학습돼서, 궤적을 바꾸지 않는 map 정보는 버린다.
- **z 증류는 student BEV의 map 정보를 전혀 바꾸지 못했다.** KD 두 run 모두 대조군과 IoU가 같다.
- 결론: §4의 "student에 옮길 정보가 없다"는 판단은 z로 쟀을 때만 성립한다. **BEV 수준의 map 차이는 분명히 있고, z가 그 차이를 전달하지 못했다.**
- 참고: 우리 DAC(93.3)는 SparseDriveV2(98.1)보다 4.8 낮다(§10). map 정보 격차와 같은 방향이다.

---

## 7. query 수를 늘리면 통로가 열리는가 (A 방식 검증)

h1에서 query 수만 바꿔 teacher BEV로 학습했다(10 epoch, seed 0). report/19 §3.2의 `--num-queries` ablation이 여기서 처음 실제로 돌았다.

| Nq | `S_own` | `S_ego` | teacher z IoU (road/walkway/centerline/crosswalk) | student z IoU | 격차 |
|---|---|---|---|---|---|
| 1 | 81.57 | 64.76 | 75.9 / 36.6 / 23.3 / 27.5 | 74.1 / 33.6 / 22.2 / 26.2 | +1.8 / +3.0 / +1.1 / +1.3 |
| 4 | 81.19 | – | 77.4 / 37.6 / 23.4 / 29.5 | 75.0 / 34.0 / 21.3 / 23.1 | +2.4 / +3.7 / +2.1 / +6.5 |
| 8 | 80.77 | 64.50 | 76.3 / 36.4 / 22.4 / 26.8 | 74.0 / 32.1 / 20.7 / 21.0 | +2.3 / +4.3 / +1.7 / +5.8 |
| 16 | 81.39 | 64.64 | (probe 실패) | | |
| 32 | 80.36 | – | (probe 실패) | | |
| 참고: BEV | – | – | 92.4 / 65.8 / 42.3 / 62.2 | 81.4 / 38.7 / 22.5 / 27.2 | +11.0 / +27.1 / +19.8 / +35.0 |

- query를 늘려도 `S_own`은 오르지 않고, z에서의 격차도 BEV 수준(11~35)에 한참 못 미친다.
- Nq ≥ 16에서는 z-probe(MLP, 입력 Nq × 256)의 학습이 실패했다. Nq=32에서 teacher와 student 값이 소수점까지 같다(probe가 입력을 무시). 이 두 값은 해석하지 않는다.
- **결론: query 수는 통로 폭을 넓히지만 무엇을 담을지는 궤적 loss가 정하고, 궤적 loss는 map 정보를 z에 담게 만들지 않는다.** A(multi-query z 증류)는 채택하지 않는다.

---

## 8. readout attention은 어디를 보는가 (B 방식 검증)

teacher_h1_s0 readout을 teacher BEV에 적용한 attention(navtest 12,146 장면 전체, `data/readout/attention_map_analysis.py`).

| 항목 | 값 |
|---|---|
| 유효 칸 수 exp(entropy) | 중앙값 **904** / 5,000 (p10 661, p90 1,196) |
| attention 50% / 90%를 덮는 칸 | 162 / 1,202 |
| 전방 0~8 / 8~15 / 15~23 / 23~31 m | 17.6 / 17.6 / 22.9 / **32.3** % |
| 좌우 ±4 m 이내 | 36.1% |

| map 클래스 | attention 비율 | 면적 비율 | 농축도 |
|---|---|---|---|
| road | 64.2% | 37.7% | **×1.70** |
| centerline | 13.1% | 7.6% | **×1.73** |
| crosswalk | 4.4% | 2.7% | **×1.66** |
| walkway | 8.9% | 11.2% | ×0.79 |

| command | 장면 | 좌우 무게중심 |
|---|---|---|
| left | 2,501 | −7.5 m |
| straight | 8,070 | +0.3 m |
| right | 1,575 | +5.4 m |

- attention은 몇 칸에 몰리지 않고 약 900칸을 덮는다. 칸 단위로 맞추면 공간 배치가 넘어갈 수 있는 넓이다.
- 주행에 쓰이는 클래스(road, centerline, crosswalk)는 면적보다 1.7배 보고, walkway는 덜 본다. "planning에 쓰이는 map 성분"이라는 선택 기준과 맞다.
- command 방향(가려는 쪽)을 따라가고, 4초 궤적 끝인 전방 23~31 m를 가장 많이 본다.

---

## 9. 새 방법: attention 가중 칸별 feature 증류 (`kd_mode=attn_feature`)

```
 teacher BEV ─► h_enc (고정) ─► attention a(u), u = BEV 칸      (query 평균, 합 1로 정규화, no grad)
 L_distill = Σ_u a(u) · d( A(F_S)(u), F_T(u) )                  d = 채널 방향 1 − cos,  A = 1×1 adapter
```

- **선택은 readout(planning 기준)이 하고, 전달은 BEV 원래 해상도로 한다.** z(1 × 256)로 압축하지 않는다.
- 비교 설계: 가중치만 다른 대조군 `kd_feature`(모든 칸 균일). 거리(cosine), adapter, `KD_SHARE=0.1`, 시작 체크포인트, 스케줄이 같다. 대조군 fine-tune(84.74)은 기존 것을 쓴다.
- 판단 기준: PDMS와 DAC가 대조군보다 오르는가, `kd_feature`보다 나은가, 그리고 증류 후 student BEV의 map probe IoU(§6의 81.4 / 38.7 / 22.5 / 27.2)가 teacher 쪽으로 올라가는가.
### 9.1 결과

interaction_final에서 5 epoch fine-tune(LR 2e-5, distill 비율 0.1, adapter, cosine). 대조군은 같은 조건에서 증류만 뺀 것.

| run | NC | DAC | TTC | EP | PDMS | 대조군 대비 PDMS (95% CI) |
|---|---|---|---|---|---|---|
| 대조군 (control) | 97.80 | 93.18 | 93.55 | 79.59 | 84.74 | — |
| z 증류 (adapter 없음) | 97.78 | 92.85 | 93.66 | 79.09 | 84.38 | −0.36 |
| z 증류 + adapter + center | 97.76 | 93.20 | 93.68 | 79.58 | 84.74 | +0.00 |
| **kd_attn** (attention 가중 칸별) | 97.77 | 93.10 | 93.47 | 79.81 | 84.76 | +0.02 [−0.34, +0.38] |
| kd_feature (전체 칸 균일) | 97.70 | 92.83 | 93.40 | 79.59 | 84.50 | −0.23 [−0.59, +0.16] |
| kd_sens (§9.2) | 97.84 | 93.13 | 93.70 | 79.51 | 84.72 | −0.02 [−0.18, +0.15] |

증류 후 student BEV의 map probe (2층 probe, navtest log 단위 8:2 분할, IoU):

| BEV | road | walkway | centerline | crosswalk |
|---|---|---|---|---|
| teacher | 92.4 | 65.8 | 42.3 | 62.2 |
| 대조군 | 81.6 | 38.7 | 22.5 | 26.6 |
| kd_attn | 81.3 | 35.3 | 21.8 | 24.2 |
| kd_feature | 81.0 | 36.3 | 21.7 | 25.5 |
| kd_sens | 81.6 | 38.4 | 22.3 | 26.4 |

- **map 정보가 student BEV로 옮겨지지 않았다.** 두 run 모두 teacher 쪽으로 오르지 않았고, 대조군과의 1~3점 차이는 probe가 epoch마다 ±2점 흔들리는 범위와 겹친다.
- 학습 중 `kd/raw`(가중 cosine 거리)는 0.80 → 0.35로 줄었지만, 같은 배치에서 det/map/plan loss가 대조군보다 5~25% 높았다. BEV가 teacher를 map과 무관한 방향으로 흉내 냈고, head들이 그 비용을 치른 것으로 본다.
- probe 25 epoch은 과했다(약 10 epoch에서 수렴). 다음부터는 짧게 돌리고 마지막 몇 번의 평가를 평균한다.

### 9.2 planning 민감도 가중 증류 (`kd_mode=sens_feature`)

attention은 "어디를 보는지"만 말한다. 칸의 차이 중 **궤적을 바꾸는 성분**만 줄이도록 했다.

```
 L = Σ_u ||G_u (v_S(u) − v_T(u))||² / (2 Σ_u ||G_u||_F²)
     v(u) = LN(W F(u) + b)          h1이 실제로 읽는 정규화된 값 (student 칸은 teacher 칸 크기로 맞춘 뒤)
     G_u  = d traj / d v_T(u)        teacher BEV에서 잰 h1의 Jacobian (random probe 4개로 추정)
```

- 처음 설계(원래 BEV 공간의 Jacobian)는 두 문제가 있었다. (1) student BEV는 LayerNorm 출력이라 칸 크기가 15.7로 일정하고 teacher는 약 8이라 크기 차이가 loss를 지배했다. (2) h1이 LayerNorm 뒤에서 읽어서 teacher 방향 자체가 Jacobian의 null space에 있어, 반대 방향 feature가 벌점 0이었다. 둘 다 고쳤다.
- 첫 run은 `KD_RAMP=2000`과 GradBalancer가 맞물려 distill 비율이 0.40까지 치솟아(목표 0.1) 버리고 `KD_RAMP=0`으로 다시 돌렸다(0.05~0.06에서 시작).
- 결과: PDMS 84.72(대조군과 같음), map probe도 대조군과 같다. 대조군 대비 차이의 신뢰구간이 다른 증류 run의 절반(±0.17)이라, 이 loss는 모델을 거의 바꾸지 않았다. 궤적을 바꾸는 성분만 맞추라고 하면 맞출 것이 거의 없다는 뜻으로, map 정보가 v1 planning의 병목이 아니라는 §13과 맞는다.

---

## 10. 평가 참고

### 10.1 navtest 비교 (SparseDriveV2 논문, arXiv 2603.29163)

| 방법 | backbone | NC | DAC | EP | TTC | PDMS |
|---|---|---|---|---|---|---|
| SparseDriveV2 | ResNet-34 | 98.5 | 98.4 | 88.6 | 95.0 | 92.0 |
| DiffusionDrive | ResNet-34 | 98.2 | 96.2 | 82.2 | 94.7 | 88.1 |
| Hydra-MDP | ResNet-34 | 98.3 | 96.0 | 78.7 | 94.6 | 86.5 |
| **interaction_final (우리)** | ResNet-50 | 97.86 | 93.31 | 79.77 | 93.56 | **84.87** |
| Transfuser | ResNet-34 | 97.7 | 92.8 | 79.2 | 92.8 | 84.0 |

| 방법 | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---|---|---|---|---|---|---|---|---|---|
| SparseDriveV2 | 98.1 | 98.1 | 99.6 | 99.8 | 91.1 | 97.3 | 96.9 | 98.2 | 78.4 | 90.1 |
| DiffusionDriveV2 | 97.7 | 96.6 | 99.2 | 99.8 | 88.9 | 97.2 | 96.0 | 97.8 | 91.0 | 87.5 |
| **interaction_final (우리)** | 97.84 | 93.31 | 99.27 | 99.84 | 87.53 | 97.00 | 96.03 | 98.34 | 88.16 | **85.07** |

- 우리 입력은 전방 카메라 3대, 0.4배 축소, 2 frame이고 궤적을 직접 회귀한다. 상위 방법들은 궤적 후보에 점수를 매겨 고른다. 백본 차이(우리 ResNet-50)는 우리에게 유리한 쪽이다.
- **격차는 DAC(−4.8)와 EP(−3.6)에 몰려 있다.** NC·TTC는 0.3 차이, DDC·TLC는 99% 이상이다.
- map과 직접 연결된 항목: DAC(주행 가능 영역), LK(차선 유지), DDC(역주행), TLC(신호), 그리고 경로 중심선 기준인 EP. 이 중 올릴 여지가 큰 것은 DAC와 EP다.

### 10.2 navhard two-stage EPDMS가 낮은 이유

| arm | 1단계 (실제 452 frame) | 2단계 (합성 5,464) | 공식 2-stage |
|---|---|---|---|
| interaction_final | 68.11 | 42.73 | 25.84 |
| parallel_final | 66.48 | 41.92 | 23.85 |
| plan_only_final | 69.16 | 41.59 | 27.33 |

- 공식 점수는 1단계 × (가중) 2단계의 곱이라 구조적으로 낮다(0.68 × 0.39 ≈ 0.27). navtest EPDMS(85)와 비교할 값이 아니다.
- 2단계는 교란된 시작 상태에서 복귀를 잰다. 곱셈 항 하나라도 0인 장면이 45~47%(DAC 0이 28~30%)이고 LK가 93~96 → 46으로 떨어진다.
- navtest의 v2 지표 경로는 사람 궤적 94.51, 등속 31.68로 정상이다. navhard 전용 사람·등속 기준선은 아직 없다. 순위 역전 분석은 yongjae의 report 21에 있다.

---

## 11. 코드 변경

| 파일 | 변경 |
|---|---|
| `navsim/agents/para_ssr/readout/distill.py` | `kd_center`(none/global/command, 갱신 전 이동평균으로 제거), `kd_mode=attn_feature` |
| `navsim/agents/para_ssr/para_ssr_loss.py` | `kd/raw_uncentered` 로그 |
| `navsim/agents/para_ssr/para_ssr_agent.py` | `attn_feature`도 readout 체크포인트 필수 |
| `navsim/agents/para_ssr/configs/default.py`, `para_ssr_agent.yaml` | `kd_center`, `kd_center_momentum` |
| `scripts/training/train_para_ssr_kd.sh` | `HEADS=off|parallel|interaction`, `ARM=control|kd_attn`, `KD_SHARE`, `KD_CENTER`, W&B 프로젝트 `para-ssr-readout`. **Hydra dict 병합 버그 수정**: `grad_balance_target={...}`는 덮어쓰지 않고 병합돼서 `distill` key가 거부되고 빠진 key가 남았다. `~key` 후 `+key=`로 다시 넣는다 |
| `scripts/training/train_para_ssr.sh` | `VAL_EVERY` (기본 5) |
| `navsim/planning/script/run_aux_evaluation.py` | KD run 평가 시 증류 설정을 끈다(teacher target builder가 붙어 단일 builder 검사에서 실패하던 문제) |
| `tools/readout/_env.py` | `<repo>/data/dataset` 우선 |
| `tests/test_para_ssr_readout.py` | `kd_center` 4개, `attn_feature` 4개, `sens_feature` 9개 추가 |
| `navsim/agents/para_ssr/readout/distill.py` (추가) | `kd_mode=sens_feature` (§9.2) |
| `navsim/agents/para_ssr/plan_map.py` | planning 쪽 map 일관성 제약: DAC와 같은 주행 가능 영역의 SDF, 차량 footprint hinge, 여유폭 옵션. `PLAN_MAP=<가중치>` (아직 학습 안 함) |
| `navsim/agents/para_ssr/para_ssr_model.py` | 평가 전용 `heading_from_path`, `kinematic_projection` (§14) |
| `navsim/agents/para_ssr/modules/kinematics.py` | TOAD 운동 모델 층, DiffusionDriveV2 Bézier heading |
| `tools/readout/dump_plan_predictions.py` | 장면별 궤적 추출 |
| `tests/test_para_ssr_plan_map.py` | 7개. v1 저장소 전체 146개 통과 |

측정 스크립트(`data/`, gitignore): `build_map_rasters.py`, `probe_map_full.py`, `attention_map_analysis.py`, `run_multiquery*.sh`, `stage3/summarize_results.py`, `analysis/failure_attribution.py`, `analysis/boundary_precision.py`. v2 쪽 변경은 §15.

---

## 12. v1 실험의 결론 (2026-09-19 마감)

- **가설**("map teacher의 planning 관련 BEV 정보를 student BEV에 증류하면 PDMS/DAC가 오른다")은 v1에서 **기각**됐다. 다섯 가지 증류(z, z+adapter+center, attention 가중 feature, 전체 feature, planning 민감도) 모두 PDMS가 대조군과 같았고, student BEV의 map probe도 teacher 쪽으로 오르지 않았다(§9.1).
- 막힌 곳은 두 군데다. (1) **전달**: 구조가 다른 teacher의 BEV feature를 흉내 내게 해도 map 정보가 옮겨지지 않는다. (2) **필요**: 옮겨졌더라도 v1 planning의 병목이 map 인지가 아니다(DAC 실패의 3.6%만 map 오인식, 이탈 지점의 80%에서 모델은 이미 경계를 봤다, §13).
- 부산물: v1의 heading 설계 결함과 그 보정(§14), PDMS 실패 원인 분석 도구(§13), planning 쪽 map 제약 loss(`plan_map.py`, 학습은 안 함).
- 정리: 증류 run 5개의 체크포인트는 지웠다(평가 CSV와 분석 결과는 `work_dirs/eval`, `data/analysis`에 남김). 대조군 체크포인트와 readout 체크포인트(`data/readout/runs`)는 남겼다. 이후 작업은 v2(§15) 위에서 하고, 방향 후보는 [report/23](23_v2_directions.md)에 정리했다.

## 12.1 남은 일

- v2 결과가 나오면 v2에서 실패 원인 분석, map probe, teacher와의 차이를 다시 잰다(§15). 증류는 v2 위에서 다시 설계한다(report/23).
- v1에서 map 정보가 왜 BEV로 옮겨지지 않았는지 진단한다.
- report/19의 plan-only 기준 서술과 `S_shuffled` 관문 서술을 이 문서 기준으로 고친다.
- 변경 사항은 아직 커밋하지 않았다.

---

## 13. PDMS 실패 원인 분석 (대조군, navtest)

대조군 궤적을 다시 뽑아(`tools/readout/dump_plan_predictions.py`) aux 평가가 저장한 장면별 det/map 예측과 함께 재생했다(`data/analysis/failure_attribution.py`). 실패 장면 1,596개.

| 항목 | 원인 | 개수 (비율) |
|---|---|---|
| DAC 828 | 궤적 그대로는 재현 안 됨 (PDM 추종 시뮬레이션에서만 이탈) | 349 (42%) |
| | GT road 경계가 이탈 지점을 덮음 (20점 근사 경계 오차 범위의 스침) | 216 (26%) |
| | map head는 경계를 봤는데 궤적이 넘음 | 152 (18%) |
| | map 범위(앞 32 m) 밖 | 81 (10%) |
| | **map head가 도로로 잘못 봄** | **30 (3.6%)** |
| NC 286 | 원인 차량 못 찾음 (target의 전방 ROI 밖 등) | 121 (42%) |
| | motion 오차 (det 박스 속도로 등속 가정한 근사) | 82 (29%) |
| | det와 motion이 맞았는데 부딪힘 | 73 (26%) |
| | **det가 놓침** | **10 (3.5%)** |
| TTC 784 | det와 motion이 맞았는데 근접 | 419 (53%) |
| | motion 오차 (근사) | 256 (33%) |
| | **det가 놓침** | **101 (13%)** |
| EP 1,129 (EP < 0.5) | 사람 경로보다 느리지 않음 | 1,083 (96%) |

- DAC 이탈 깊이는 중앙값 0.14 m, 이탈 시점은 중앙값 3.0 s다.
- **경계 정밀도 (`boundary_precision.py`, map API의 원본 road polygon 기준):** 경로 주변 경계 위치 오차는 DAC 통과 장면 0.44 m, 실패 장면 0.50 m로 거의 같다. 이탈 지점 400개에서 실제 경계 기준 여유는 −0.10 m인데 모델 map head 기준 여유는 −0.66 m다. **80%의 이탈 지점에서 모델은 이미 그곳을 도로 밖으로 봤다.** planner가 자기 map 예측을 따르지 않는다.
- 해석: 인지 오류(det 놓침, map 오인식)의 몫은 작다. map 증류로 DAC를 올릴 수 있는 폭도 작다.

---

## 14. heading 설계 결함 (v1)

- v1 planner는 step별 (x, y, heading) 변화량을 예측하고 누적합한다. heading이 경로 방향과 어긋나고, 오차가 step마다 쌓인다: 중앙값 0.28° → 0.98°, 상위 10% 0.83° → 3.91° (step 1 → 8).
- NAVSIM의 PDM은 궤적을 LQR + bicycle 모델로 추종해 채점하고, 제출된 heading을 속도 추정과 곡률 추정에 쓴다(`batch_lqr_utils.py`). heading이 어긋나면 추종된 차가 선에서 밀려나고 추정 속도도 낮아진다.
- 직진 명령 장면에서 heading 차이가 2°를 넘으면 DAC 실패율이 11.9%, 아니면 4.7%다.

평가 때만 궤적을 바꾼 결과 (학습 없음):

| 모델 | 원래 | heading을 경로 방향으로 (`heading_from_path`) | TOAD 운동 모델 투영 (`kinematic_projection`) |
|---|---|---|---|
| interaction_final | 84.87 | **85.93** (DAC +0.91, EP +0.91) | 82.41 |
| 대조군 | 84.74 | **85.61** (DAC +0.70, EP +0.75) | 82.23 |

- 모든 지표 차이가 95% 신뢰구간에서 유의하다. DAC 실패 223개가 고쳐졌고(그중 183개가 추종 시뮬레이션에서만 드러나던 것) 138개가 새로 생겼다.
- TOAD 투영은 회전 속도를 예측된 heading에서 계산해 위치를 다시 적분한다. v1은 heading 쪽이 틀려 있어서 궤적이 틀린 쪽으로 휘었다. **틀린 쪽은 heading이고 위치는 맞다**는 것이 두 결과로 확인된다.
- 이 보정은 논문 기여가 아니라 기준선에 빠져 있던 처리다. 이후 v1과 v2를 비교할 때는 v1에 보정을 켠다(실질 기준선 85.93).
- 공개 코드 조사(요약): TransFuser/LTF, DiffusionDrive, WoTE, DrivoR, iPad는 heading을 따로 회귀한다. DiffusionDriveV2는 xy로 Bézier 곡선을 만들어 heading을 계산한다. GTRS/Hydra 계열과 SparseDriveV2는 일관된 vocabulary를 그대로 낸다. 운동 모델 적분 출력은 TOAD의 추론 시점 CEM뿐이다. VAD/UniAD의 NAVSIM 포팅은 공개 코드가 없다.

---

## 15. PARA-SSR v2

worktree `~/kyungmin/SSR-v2`. 공개 코드를 따라 구현했다.

| 부분 | 출처 | 내용 |
|---|---|---|
| anchor 256개, PDM 점수 loss, 선택식 | WoTE (`navsim/agents/WoTE`, 공개 extra data) | anchor query → PARA-SSR planner 층(BEV + det/motion + map memory) → offset, 모방 점수, NC/DAC/EP/TTC/C 점수. loss는 WTA offset L1 + 모방 soft CE + PDM 점수 BCE×5. world model은 뺐다 |
| PDM 점수 라벨 | WoTE 공개 `formatted_pdm_score_256.npy` | navtrain 103,288 토큰 전부. 264 MB로 압축(`tools/plan_v2/pack_pdm_scores.py`) |
| heading | DiffusionDriveV2 `bezier_xyyaw` | offset head는 x, y만 낸다. heading = 원점과 8점으로 만든 8차 Bézier의 접선. 정지 장면은 anchor heading(atan2 기울기 NaN 방지) |
| 운동 모델 층 (`plan_kinematic`, 꺼 둠) | TOAD `_bicycle_rollout` 등 | 현재 속도에서 anchor를 재현하면 마지막 위치가 중앙값 3.1 m 어긋나 WoTE 라벨과 충돌해 쓰지 않는다 |

- heading을 따로 회귀하던 첫 v2는 2 epoch째 heading이 경로와 중앙값 약 3° 어긋나 있어 지우고 Bézier heading으로 다시 시작했다.
- 학습: interaction_final과 같은 레시피(처음부터 30 epoch, LR 1e-4, warmup 3, batch 4×2×16, 5 epoch마다 validation, grad clip 35, GradBalancer plan 0.4 / det 0.3 / map 0.3).
  - `para_ssr_v2_r34`: ResNet-34, GPU 0,1, 2026-09-18 23:55 시작. W&B `para-ssr-v2` / `resnet-34`.
  - `para_ssr_v2_r50`: ResNet-50, GPU 2,3, kd_sens 측정 뒤 시작. W&B `para-ssr-v2` / `resnet-50`. interaction_final과 planner만 다르다.
- 중간 평가(`SSR-v2/data/logs/mid_eval.sh`): 5·10·15·20·25 epoch마다 v1(heading 보정) / v2_r34 / v2_r50 PDMS.
- 테스트: `tests/test_para_ssr_anchor_planner.py` 10개(WoTE 원본 loss와 값 일치, 쓰이지 않는 파라미터 없음, Bézier heading 원본 일치 등). 전체 156개 통과.
