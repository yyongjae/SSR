# 42. r41_v2b (motion mode tokens + map 4 segments) 결과와 1단계 진단

- 작성: 2026-10-19. 학습은 Poseidon, 평가·분석은 Turing.
- 대상 run: `r41_v2b_motionmap_modetok_s4` (+ `_resume`). 코드 `99e920a`(브랜치 `chan`).
- 구성: PARA-SSR v2 recipe(WoTE식 256 anchor + PDM score reward, Bézier heading, ResNet-34, 30 epoch,
  global batch 128 = 4 GPU × 4 × accumulate 8, LR 1e-4, GradBalancer plan/det/map 0.4/0.3/0.3,
  grad_balance_warmup 5300 / interval 100) 위에 report 41의 두 memory 변경을 모두 켠 것:
  `plan_det_memory_mode=mode_tokens`, `plan_map_memory_segments=4`. 단일 seed, 단일 run.
- 비교 대상 v2 baseline(`mean`, S=1)의 수치는 사용자 제공값이며 scene별 CSV와 checkpoint는 이 저장소에 없다.
  따라서 **paired 비교는 아직 못 했다.**

## 1. 학습 이력

| 시각 | 사건 |
|---|---|
| 10-16 19:43 | 시작 (queue_v2/queue.sh). 같은 run을 Turing에서도 걸었다가 중복이라 10-16 19:54 중단 |
| 10-17 19:20 | epoch 25 도중 exit 137 (rank 1 자식 프로세스 SIGKILL). 원인: validation 후 남는 idle val worker 32개가 train worker와 겹쳐 188 GB 호스트 OOM (dmesg 10-17 19:18) |
| 10-18 00:07 | `last.ckpt`(epoch 24, step 16625)에서 resume. 이때 `SceneFrameStore`(dataloader.py)와 `release_val_workers.py`를 넣었다. **epoch 0~24와 25~29의 데이터 로더 코드가 다르다** (모델·loss는 동일) |
| 10-18 03:43 | epoch 29 완료, exit 0. 총 약 26.7 h |

epoch 1~6이 66~103분으로 이후(41~43분)의 2배였다. GPU 공유 간섭.

val traj loss (5 epoch마다): 7.48 (ep4) → 7.37 (ep9) → 7.38 (ep14) → 7.51 (ep19) → 7.59 (ep24) → 7.71 (ep29).
train loss는 13.7 → 12.8로 계속 내려간다. epoch 9 이후 val loss는 오르지만 PDMS는 epoch 19까지 오른 뒤 정체한다
(§2). **val loss로 checkpoint를 고르면 안 된다.**

## 2. navtest PDMS (v1 scorer, `run_pdm_score_gpu`, 12,146 scene, 실패 0)

| ckpt | PDMS | NC | DAC | DDC | EP | TTC | C | score=0 |
|---|---|---|---|---|---|---|---|---|
| ep09 | 87.33 | 98.45 | 95.29 | 99.99 | 81.88 | 94.66 | 99.97 | 609 |
| ep19 | 88.32 | 98.69 | 96.18 | 100.00 | 81.98 | 95.69 | 99.99 | 609 |
| ep29 | 88.37 | 98.72 | 96.25 | 100.00 | 82.02 | 95.67 | 99.98 | 596 |

scene별 paired bootstrap (2,000회):
ep09→ep19 +0.99 [0.64, 1.35], ep19→ep29 +0.05 [−0.20, 0.30], ep09→ep29 +1.03 [0.66, 1.39].
ep19→ep29에서 좋아진 scene 3,975 vs 나빠진 scene 4,031. **수렴은 epoch 19에서 끝났다.**
세 epoch 모두 0점인 scene 342개(2.8%).

개선은 안전 지표에서 나왔다: DAC=0 572→455, NC=0 180→146, TTC=0 649→526 (ep09→ep29).
EP는 81.9→82.0으로 정체이고 score>0인 scene만 보면 EP 평균이 87.2→86.3으로 오히려 내려갔다.

Turing에서 v1 metric cache를 새로 만들고 dump한 궤적으로 재채점한 PDMS가 Poseidon 공식 CSV와
scene별로 일치했다(최대 차이 9e-8, DAC·NC 불일치 0). 이후 분석은 이 기반 위에서 했다.

## 3. v2 baseline과 비교 (ep29)

| 지표 | v2 baseline (사용자 제공) | r41 ep29 | Δ |
|---|---|---|---|
| PDMS | 88.14 | 88.37 | +0.23 |
| NC | 98.70 | 98.72 | +0.02 |
| DAC | 96.08 | 96.25 | +0.17 |
| DDC | 99.99 | 100.00 | +0.01 |
| EP | 81.74 | 82.02 | +0.28 |
| TTC | 95.54 | 95.67 | +0.13 |
| C | 99.98 | 99.98 | 0 |
| Det center mAP (protocol V3) | 0.255 | 0.2534 | −0.002 |
| Map Chamfer mAP (protocol V3) | 0.221 | 0.2193 | −0.002 |
| EPDMS | 37.1 (split·scorer 미확인, navhard two-stage로 추정) | 87.84 (navtest one-stage) | 비교 불가 |

판정: **효과를 주장할 수 없다.** +0.23은 같은 run의 ep19↔ep29 차이(CI 폭 ±0.25)와 같은 크기이고, run 간 seed noise는
아직 재지 않았다. report 41의 가설(motion token → NC·TTC, map token → DAC)과 맞춰 보면 NC +0.02, TTC +0.13은 변화가 없고
DAC +0.17은 방향만 맞다. 가장 큰 항은 가설에 없던 EP +0.28이다. 인지 head(mAP)는 변하지 않았다.
baseline CSV를 받으면 paired CI를 낸다. CI가 0을 포함하면 기본값(`mean`, S=1)으로 되돌린다.
r41은 det token 300→1,800, map token 100→400으로 추론 비용이 늘었고 그 대가가 없다.

Det 세부(AP): vehicle 0.51, generic_object 0.52, barrier 0.22, traffic_cone 0.24, pedestrian 0.23, bicycle 0.05, czone_sign 0.00.
Map 세부: road 0.34, centerline 0.31, walkway 0.11, crosswalk 0.11. 전체 파일 `42_results/r41_v2b_ep29_aux_metrics.json`.

## 4. navtest EPDMS (NAVSIM v2.2 one-stage, non-reactive, ep29)

| EPDMS | EPDMS (EC 제외) | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
|---|---|---|---|---|---|---|---|---|---|---|
| 87.84 | 88.89 | 98.72 | 96.25 | 99.36 | 99.82 | 86.48 | 98.33 | 96.21 | 98.31 | 83.16 |

score=0 scene 644. EC는 인접 frame이 없는 2,106 scene에서 NaN(가중치 0 처리). 절차는 `scripts/evaluation/epdms/README.md`.
report 24에 기록된 9월 v2 run의 EC 제외 EPDMS 86.49와 비교하면 +2.4이지만, 그 run이 §3의 baseline과 같은 checkpoint인지
확인되지 않았다.

## 5. DAC: raw 궤적 vs LQR 추종 궤적 (ep29, `tools/analysis/dac_raw_vs_tracked.py`)

| | scene |
|---|---|
| 추종 후 DAC 실패 | 455 |
| 그중 raw 궤적도 도로 밖 (계획 단계 이탈) | 370 (81%) |
| 그중 추종 때문에 생긴 이탈 | 85 (19%) |
| raw는 이탈인데 추종이 구한 경우 | 248 |

v1에서 42%였던 추종 유발 비율이 19%다. 추종 편차 중앙값 0.63 m(p90 1.05, p99 1.47)로 실패/통과 그룹 간 차이가 없다.
**tracking-aware planning(report 23 §C)의 근거는 v2에서 약하다.**

주의: 같은 방법으로 raw NC를 계산하면 전부 1이 나온다. PDM scorer는 자차 속도가 0이면 충돌을 과실 없음으로 처리하는데
raw 배열에는 속도가 없다. raw는 DAC·DDC만 유효하다.

## 6. 간이 실패 원인 분해 (ep29, `tools/analysis/failure_attribution_lite.py`)

aux record의 scene별 det/map 예측·GT를 PDMS 실패와 결합한 proxy다. 인과가 아니라 연관이다.
proxy: 15 m 이내 GT 객체를 score≥0.3 예측이 중심 2 m 이내로 맞췄는가(det), GT road polyline과 예측 road polyline의
Chamfer 거리(map, 1.5 m 초과를 bad-map으로 둠).

| 그룹 | n | road Chamfer 중앙값 | bad-map | 근거리 객체 놓침 |
|---|---|---|---|---|
| DAC 통과 | 11,691 | 1.74 m | 55.6% | 24.7% |
| DAC 실패 | 455 | 2.89 m | 69.7% | 20.7% |
| NC 통과 | 11,981 | 1.76 m | 56.0% | 24.3% |
| NC 실패 | 165 | 2.17 m | 67.9% | 37.6% |
| TTC 실패 | 526 | 1.98 m | 61.2% | 33.8% |
| EP<0.5 (score>0) | 134 | 2.79 m | 82.1% | 33.6% |

- NC 실패의 62%, TTC 실패의 66%는 근거리 GT 객체를 모두 검출한 상태에서 났다.
- DAC 실패와 bad-map의 연관은 있지만 약하다(70% vs 56%). map head 품질 자체가 낮아(mAP 0.22) proxy 변별력이 떨어진다.
- EP가 낮은 scene은 bad-map 비율이 82%로 가장 높다.

## 7. 방향 D 전제 검사: GT 기하 proxy vs PDM anchor 라벨 (`tools/analysis/anchor_proxy_check.py`)

navtrain 1,000 token(seed 0) × anchor 256 = 256,000개. metric cache를 즉석에서 만들어 공식 scorer로 재채점하고,
GT drivable 경계까지의 signed margin과 GT 객체 폴리곤까지의 최소 거리를 계산했다.

- 공식 재채점 = WoTE 라벨: NC·DAC·TTC·C 일치 100%, EP 상관 1.0000. 라벨과 파이프라인 모두 확인.
- DAC: 추종 없는 raw anchor의 DAC가 공식과 94.2% 일치(추종이 바꾸는 anchor 약 6%, 양방향 대칭).
  GT 경계 signed margin으로 공식 DAC 실패 예측 AUC 0.975. 32 m BEV 범위 밖에서만 이탈하는 경우는 DAC 실패의 1.8%뿐이고
  범위 제한 margin의 AUC도 0.975. DAC 통과 anchor 중 경계 0.3 m 이내 14.5%, 0.5 m 이내 22.7%; 실패 중 0.3 m 미만의 얕은 이탈 11.8%.
  **이진 라벨이 숨기는 연속 정보가 상당하다. DAC proxy는 학습 target으로 쓸 수 있다.**
- NC: GT 객체 최소 거리로 공식 NC 실패 예측 AUC 0.908, 거리≤0과의 일치 88.1%. 겹치지만 과실 아님 9.2%, 겹침 없이 과실 2.7%.
  **과실 규칙(자차 정지, 후방 추돌 등) 없이는 12%가 어긋난다.** 같은 거리가 TTC 실패도 AUC 0.908로 예측한다.
- token당 NC=DAC=1인 안전 anchor 중앙값 82, p10 26, 0개인 token 0.1% (report 24 H2와 일치).

## 8. 결론과 다음

1. memory 해상도(r41)는 lever가 아니다. report 24(H1·H5, GT critic 천장 +1.5)와 합치면 인지→점수 연결이 끊긴 것이 문제이고,
   hidden token을 더 주는 것으로는 풀리지 않는다.
2. 학습 없이 할 진단: (a) ep29에서 det/map memory·BEV readout을 각각 0으로 막는 test-time ablation, (b) 예측 road polyline으로
   만든 DAC margin의 공식 DAC 예측 AUC(GT는 0.975).
3. 학습 실험 후보: 기하 feature(예측 경계 margin, 예측 객체 거리, route 진행도)를 점수 head에 명시 입력 + GT margin 연속 target(방향 D와 통합),
   readout dropout, map head 개선. 순서는 진단 결과로 정한다.
4. 후속 run은 20 epoch로 줄이고, run 간 noise를 재기 전까지 ±0.3 수준의 차이는 판정하지 않는다.

## 9. 산출물과 코드

| 내용 | 위치 |
|---|---|
| PDMS scene별 CSV (ep09/19/29) | Poseidon `work_dirs/eval/r41_v2b_motionmap_modetok_s4_ep*/`, 사본 Turing `work_dirs/eval/poseidon_csv/`, ep29는 `report/42_results/` |
| checkpoint ep29 | Poseidon `work_dirs/r41_v2b_motionmap_modetok_s4_resume/lightning_logs/version_0/checkpoints/epoch=29-step=19950.ckpt` (sha256 `cbbf404f…`), Turing `work_dirs/eval/ckpts/r41_v2b_ep29.ckpt` |
| aux mAP | Turing `work_dirs/eval/r41_v2b_motionmap_modetok_s4_ep29_aux/aux_metrics.{json,csv}`, json은 `report/42_results/` |
| 궤적 dump | Turing `work_dirs/eval/r41_v2b_ep29_navtest_trajectories.pkl` |
| EPDMS CSV | Turing `~/navsim_v2_exp/exp/epdms_r41_v2b_ep29/…/*.csv`, 사본 `report/42_results/` |
| raw vs tracked, attribution, anchor proxy CSV | Turing `work_dirs/eval/r41_v2b_ep29_raw_vs_tracked.csv`(사본 `report/42_results/`), `r41_v2b_ep29_attribution.csv`, `anchor_proxy_check_1000.csv` |
| metric cache | Turing v1 `data/exp/metric_cache`(navtest 12,146), v2 `~/navsim_v2_exp/exp/metric_cache_navtest`, navtrain 샘플 `work_dirs/eval/navtrain_cache_sample` |
| 학습 코드 변경 (resume부터 적용) | `navsim/common/dataloader.py`(SceneFrameStore), `navsim/planning/training/release_val_workers.py`, `run_training.py`, `dataset.py` |
| 평가·분석 도구 | `tools/dump_navtest_trajectories.py --experiment`, `tools/analysis/*.py`, `scripts/evaluation/epdms/` |

Turing의 archived config(`work_dirs/r41_v2b_motionmap_modetok_s4_resume/code/hydra/config.yaml`)는 Poseidon 절대경로
`/data4/chanyoung/data/ssr/SSR`을 `/home/chanyoung/SSR`로 바꾼 것이다(원본 `config.yaml.poseidon`).
