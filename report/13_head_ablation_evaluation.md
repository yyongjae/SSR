# PARA-SSR head ablation 평가: PDMS · EPDMS · auxiliary mAP

3 camera + LiDAR, BEV 50×100 encoder([`12_lidar_bev_encoder_50x100.md`](12_lidar_bev_encoder_50x100.md))
위에서 parallel aux head의 유무만 바꾼 네 모델을 navtest에서 평가했다.

## 1. 실험 구성

| arm | 실험 디렉터리 | det+motion head | map head | `grad_balance_target` |
|---|---|:---:|:---:|---|
| ssr | `para_ssr_front3_lidar_30ep` | ✓ | ✓ | `plan 0.4 / det 0.3 / map 0.3` (yaml 기본값) |
| nodet | `para_ssr_map_plan` | – | ✓ | `plan 0.5 / map 0.5` |
| nomap | `para_ssr_det_motion_plan` | ✓ | – | `plan 0.5 / det 0.5` |
| plan_only | `para_ssr_plan_only` | – | – | `null` (단일 task) |

- 공통 recipe: 2 GPU × microbatch 4 × accumulate 16 = global batch 128, AdamW 1e-4, fp32, 30 epoch
  ([`README.md`](README.md) §3). 평가는 모두 마지막 checkpoint(`epoch=29`, 30번째 epoch)다.
- head를 끈 arm은 head parameter·target·loss가 생성되지 않는다(loss weight 0 방식 아님).
- **주의**: ssr의 planner gradient 비중은 0.4로 나머지 arm(0.5)과 다르다. ssr과 ablation arm의
  차이에는 head 유무와 이 비중 차이가 섞여 있다.
- 각 arm은 seed 1개다. 아래 CI는 scene 샘플링의 불확실성만 반영하고 학습 seed 편차는 포함하지 않는다.

## 2. NAVSIM v1 PDMS (navtest)

`scripts/evaluation/eval_para_ssr.sh`, metric cache `data/exp/metric_cache`, 12,146 scene 전부 valid.

| arm | NC | DAC | DDC | EP | TTC | C | **PDMS** |
|---|---:|---:|---:|---:|---:|---:|---:|
| ssr | 98.18 | 93.75 | 100.00 | 80.11 | 94.09 | 99.99 | **85.56** |
| nodet | 97.79 | 93.94 | 99.99 | 80.37 | 93.06 | 99.99 | **85.26** |
| nomap | 98.05 | 93.05 | 100.00 | 79.84 | 93.74 | 99.98 | **84.92** |
| plan_only | 98.19 | 94.66 | 100.00 | 81.08 | 94.01 | 99.98 | **86.45** |

## 3. NAVSIM v2 EPDMS (navtest, one-stage)

| arm | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | **EPDMS** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ssr | 98.18 | 93.75 | 99.27 | 99.84 | 87.12 | 97.39 | 96.32 | 98.34 | 88.65 | **85.89** |
| nodet | 97.79 | 93.94 | 99.29 | 99.82 | 87.61 | 96.90 | 96.76 | 98.35 | 88.59 | **85.85** |
| nomap | 98.04 | 93.05 | 99.09 | 99.81 | 87.49 | 97.18 | 96.00 | 98.35 | 88.59 | **85.21** |
| plan_only | 98.18 | 94.66 | 99.46 | 99.79 | 87.30 | 97.40 | 96.60 | 98.34 | 88.53 | **86.79** |

EC는 v2 two-frame extended comfort이다. 모든 arm에서 12,146 scene 전부 valid.

### 3.1 plan_only 대비 paired 차이

같은 12,146 token에서 scene 단위 차이의 평균과 bootstrap 95% CI (10,000 resample).

| arm | ΔEPDMS | ΔPDMS |
|---|---|---|
| ssr | −0.90 [−1.34, −0.46] | −0.88 [−1.35, −0.44] |
| nodet | −0.94 [−1.40, −0.49] | −1.19 [−1.66, −0.73] |
| nomap | −1.58 [−2.03, −1.13] | −1.52 [−1.98, −1.07] |

### 3.2 해석

- 두 지표에서 순위가 같다: plan_only > ssr ≳ nodet > nomap. aux head를 가진 세 arm 모두
  plan_only보다 유의하게 낮다. 이 설정에서 aux head는 planning 점수를 올리지 않는다.
- 차이는 곱셈 항인 **DAC**가 가장 크게 만든다(plan_only 94.66 → ssr 93.75, nomap 93.05).
  DDC도 같은 방향이다.
- nodet은 NC(97.79)·TTC(96.90)가 가장 낮지만 EP·LK가 높아 EPDMS에서는 ssr과 같은 수준이 된다.
- v2에서 추가된 TLC·HC·EC는 arm 간 차이가 0.1 이내라 순위에 영향이 없다.

### 3.3 채점 방법과 검증

NAVSIM v2는 SSR(v1 fork)과 같은 `navsim` package 이름을 쓰므로 v2 checkout을 따로 두고
`PYTHONPATH`로 격리했다(v2.2 `main`, `0a380a9`). v2 one-stage runner는 CPU worker 안에서
agent를 만들어 CUDA 전용 spconv encoder를 쓸 수 없으므로 두 단계로 나눴다.

1. **SSR, GPU** — `tools/dump_navtest_trajectories.py --arm <arm> --batch-size 1`: PDM 평가와 같은
   checkpoint·feature builder·보관된 학습 config(`work_dirs/<exp>/code/hydra/config.yaml`)로
   token별 `[8, 3]` trajectory를 저장한다. batch 추론은 PDM 경로(batch 1)와 최대 2.5e-3 m
   달랐고 batch 1은 ~2e-6 m로 일치해 batch 1을 썼다. navtest scene filter는
   v1과 v2가 동일하다(공백 제외).
2. **NAVSIM v2, CPU** — v2 checkout에 추가한 스크립트:

   | 파일 | 역할 |
   |---|---|
   | `scripts/evaluation/para_ssr_env.sh` | v2 경로를 `PYTHONPATH` 앞에 두고 `navsim`이 v2로 import되는지 assert |
   | `scripts/evaluation/cache_metric_navtest_v2.sh` | v2 metric cache (process pool) |
   | `navsim/planning/script/run_pdm_score_from_trajectories.py` | 저장한 trajectory 채점 |
   | `scripts/evaluation/run_epdms_from_trajectories.sh` | arm 하나 채점 wrapper |
   | `scripts/evaluation/run_para_ssr_epdms_all.sh`, `summarize_para_ssr_epdms.py` | 4 arm 일괄 실행·요약·paired CI |

   `run_pdm_score_from_trajectories.py`는 v2 `run_pdm_score_one_stage.py`와 trajectory 출처만
   다르다. simulator·scorer·traffic agent policy·`pdm_score`와 score row 필드, 인접 frame
   mapping(`infer_start_adjacent_mapping`), two-frame EC(`create_scene_aggregators`),
   최종 점수(`compute_final_scores`)는 v2 함수를 import해 그대로 쓴다. cache나 trajectory에
   빠진 token이 있으면 조용히 건너뛰지 않고 실패한다.

**동일성 검증.** 이미 cache가 끝난 3개 log(709 scene)에서 v2 원본 runner(agent를 worker 안에서
실행)와 새 runner(같은 agent의 저장된 trajectory)의 token별 CSV 전 열을 비교했다.

| reference agent | row | valid | max \|diff\| | NaN 불일치 | 평균 score |
|---|---:|---:|---:|---:|---|
| constant velocity | 710 | 710/710 | 0 | 0 | 0.316751 = 0.316751 |
| human (GT trajectory) | 710 | 710/710 | 5.7e-8 | 0 | 0.945117 = 0.945117 |

human의 차이는 trajectory를 float32로 저장한 반올림이다.

### 3.4 navhard two-stage를 하지 않은 이유

`download_navhard_two_stage.sh`의 synthetic scene pickle(5,462개)은 frame마다
`MergedPointCloud/*.pcd` 경로를 갖지만, 현재 frame sensor archive
(`navsim_v2.2_navhard_two_stage_curr_sensors.tar.gz`, 12.8 GB) 전체 목록 44,574개는 8개 카메라
이미지(카메라당 5,562장)뿐이고 point cloud는 0개다. history archive 앞 3 GB도 카메라만 있었다.
synthetic scene에 LiDAR가 없으므로 LiDAR 입력으로 학습한 네 모델은 stage 2를 학습 조건대로
평가할 수 없다. LiDAR를 0으로 채우면 결과가 aux head 효과가 아니라 입력 결손의 영향이 된다.

## 4. Auxiliary mAP (protocol V3)

`scripts/evaluation/eval_para_ssr_aux.sh`, 12,146 token. 비공식 metric이며 같은 protocol의 arm끼리만
비교한다. plan_only는 aux head가 없어 대상이 아니다.

| arm | tasks | detection center mAP | map Chamfer mAP |
|---|---|---:|---:|
| ssr | detection, map | 59.38 | 35.85 |
| nodet | map | – | **39.02** |
| nomap | detection | **60.91** | – |

한 task만 남긴 arm이 그 task에서 더 높다(detection +1.53, map +3.17). 다만 그 arm에서는 해당
task의 shared-BEV gradient 비중도 0.3에서 0.5로 커지므로, 다른 aux task가 빠진 효과와 비중 증가의
효과가 분리되지 않는다.

Detection class별 AP (GT 수는 세 arm 공통):

| class | GT | ssr | nomap |
|---|---:|---:|---:|
| vehicle | 49,705 | 92.4 | 93.1 |
| pedestrian | 24,918 | 74.0 | 75.1 |
| bicycle | 545 | 41.9 | 45.1 |
| traffic_cone | 16,974 | 64.9 | 65.4 |
| barrier | 5,646 | 45.9 | 48.1 |
| czone_sign | 503 | 13.9 | 17.3 |
| generic_object | 43,282 | 82.7 | 82.3 |

Map class별 AP:

| class | GT | ssr | nodet |
|---|---:|---:|---:|
| road | 46,007 | 52.6 | 55.8 |
| walkway | 47,115 | 21.5 | 24.6 |
| centerline | 187,039 | 49.7 | 53.7 |
| crosswalk | 17,904 | 19.6 | 22.1 |

## 5. BEVFusion teacher와의 detection 비교

BEVFusion(camera+LiDAR fusion, 50×100 재학습) teacher는 자체 nuScenes-style evaluator에서
mAP 82.36 / NDS 81.65이다. 이 값은 evaluator와 GT 정의가 달라 위 표와 직접 비교할 수 없으므로,
teacher cache의 detection을 PARA-SSR과 **같은 V3 evaluator**로 다시 채점했다
(navtest 12,146 token, top-100).

| class | teacher | ssr | nomap |
|---|---:|---:|---:|
| vehicle | 95.5 | 92.4 | 93.1 |
| pedestrian | 88.0 | 74.0 | 75.1 |
| bicycle | 72.6 | 41.9 | 45.1 |
| traffic_cone | 84.2 | 64.9 | 65.4 |
| barrier | 73.9 | 45.9 | 48.1 |
| czone_sign | 53.7 | 13.9 | 17.3 |
| generic_object | 89.5 | 82.7 | 82.3 |
| **mAP** | **79.63** | **59.38** | **60.91** |

ssr 기준으로 vehicle·generic_object는 3–7 point, pedestrian은 14 point 차이지만
traffic_cone·barrier·bicycle·czone_sign에서는 19–40 point 벌어진다. GT가 적거나 작은 class에서 student detection head가 teacher에 크게 못 미친다.

## 6. 산출물 위치

| 내용 | 경로 |
|---|---|
| PDMS token별 CSV | `work_dirs/eval/<experiment>/<timestamp>.csv` |
| aux mAP | `work_dirs/eval/<experiment>_aux/aux_metrics.{json,csv}` |
| teacher 재채점 | `work_dirs/eval/teacher_vs_student_detection.json` |
| 저장한 trajectory | `work_dirs/eval/<experiment>_navtest_trajectories.pkl` |
| EPDMS token별 CSV | `$NAVSIM_EXP_ROOT/eval_epdms/<arm>/final/<timestamp>.csv` (v2 checkout 쪽) |
| EPDMS 요약·paired CI | `$NAVSIM_EXP_ROOT/eval_epdms/para_ssr_epdms_summary.json` |
