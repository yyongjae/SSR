# Stage T (teacher refiner) 구현 상태

작성: 2026-09-28 09:30 KST (통합 단계). 설계 계약은 `report/refiner_T/IMPL_SPEC.md`, 판정 규칙은 `report/refiner_T/PRESTATED_DECISION_RULE.txt`를 따른다.

## 0. 요약

- 파이프라인 전체를 구현했다. 코드 29개 파일, CPU 테스트 23개 파일에서 **테스트 185개가 모두 통과**했다(전체 한 번에 약 2분 20초).
- 실제 데이터 20개 토큰(train 12, dev 8)으로 처음부터 끝까지 CPU에서 돌렸다. 경로는 데이터 → refiner(R_T, R_none) → 대리 손실 → 역전파 → 최적화 → `eval_refiner` → 공식 채점이다. 모든 단계가 돌았다.
  - "수정하지 않은" 초안은 원래 공식 점수를 **정확히**(float64 비트 단위) 돌려준다.
- 백그라운드에서 데이터를 만들고 있다. 전부 끝나는 시각은 약 **10:45–11:15**이다. 이 작업들은 metric cache, SDF, 초안 은행, packing이다.
  - 물체(M6) 전체 빌드는 이번에 실행해서 끝냈다.
- **GPU 학습을 시작하기 전에 정할 것**이 넷 있다(§4의 D1–D4).
  1. M8 기준 2개가 미달이다.
  2. 초안 은행 실패율이 목표보다 낮다.
  3. loss weight 후보를 정해야 한다.
  4. GPU 예산을 정해야 한다.

## 1. 만들어진 것

### 1.1 코드

패키지 `navsim/agents/para_ssr/refiner/`:

| 파일 | 역할 |
|---|---|
| `geometry.py` | 초안 경로(C2 스플라인), arc length, 연장 |
| `decoder.py` | M4 교정 디코더, 초안 교란 생성기 |
| `gt_future.py` | M6 미래 물체(GT) 조회 |
| `sdf.py` | 주행 가능 영역 SDF |
| `surrogate.py` | M7 대리 손실(충돌, DAC, 진행, 편안함, 수정량, gate) |
| `adapters.py` | teacher BEV → S grid, z-score, 1×1 어댑터 |
| `corridor.py` | 통로 샘플링과 전역 token |
| `refiner_net.py` | M5 refiner |
| `data.py` | teacher cache 접근, split packing(memmap), loader |

도구 `tools/refiner/`:

| 파일 | 역할 |
|---|---|
| `make_splits.py`, `build_metric_cache.py`, `extract_human.py` | split 생성, metric cache, 사람 궤적 |
| `build_future_objects.py`, `build_sdf.py`, `validate_sdf.py` | 물체, SDF |
| `score_trajectories.py`, `check_scorer_equivalence.py` | 공식 채점(batch) |
| `make_draft_bank.py`, `validate_decoder.py`, `validate_surrogate.py` | 초안 은행, 디코더 검증, M8 |
| `train_refiner.py`, `eval_refiner.py` | 학습, 평가 |
| `smoke_refiner.py` | 이전 smoke(초안 은행이 생기기 전 버전) |
| **이번에 추가** | |
| `integrate_smoke.py` | 실제 데이터 end-to-end 검사(§1.3) |
| `pack_follow.py` | 상류 작업이 끝나는 대로 packing을 이어서 하는 작업 |
| `stageT_decision.py` | 교차 적합 θ 선택 + dev 판정(사전 규칙 구현) |
| `stageT_gpu_plan.sh` | GPU 여러 장에 run 30개/6개를 분배 |
| `stageT_gpu_commands.sh` | run 1개: 학습 후 평가 |

테스트는 `tools/refiner/tests/test_*.py` 23개 파일이다. 이번에 `test_pack_follow.py`(3개), `test_stageT_decision.py`(5개), `test_integrate_smoke.py`(2개)를 추가했다.

### 1.2 검증 수치(모듈별, 모두 CPU)

- **공식 채점기 동일성**
  - batch 채점(PDM-Closed와 최대 13개 초안을 함께 채점)이 단독 `pdm_score`와 **불일치 0건**이다. 1,300 토큰, 8,500 궤적에서 NC, DAC, DDC, EP, TTC, comfort, PDMS 7개 값을 비교했다.
  - 구성: navtest K=6 1,000토큰, K=13 100토큰, E navtrain 200토큰.
  - 그냥 한꺼번에 채점하면 틀린다. EP는 2,163건, DDC는 76건이 달라진다(K=6 세트).
  - progress 임계값은 5.0 m이다(공식 yaml).
- **M6 미래 물체 재구성**
  - metric cache 다각형 대비 모서리 최대 오차는 **7.6e-6 m**이다.
  - (track, 시각) 쌍 약 600만 개에서 존재 여부 불일치는 **0건**이다.
  - 대상은 1,279토큰이다(navtest 무작위 500 + NC 실패 279 + navtrain E 500).
  - UNKNOWN 비율은 0이다(기준 궤적 3종).
- **SDF와 공식 DAC 일치**
  - LQR로 추적한 41개 footprint에서 "모서리 SDF < 0"이 공식 DAC와 **548/550** 일치한다. 놓침 2건, 오경보 0건, AUC 0.9999다.
  - 놓친 2건은 폭 10 cm 미만의 지도 틈 때문이다. 0.25 m grid로는 표현할 수 없다.
  - 추적 전 원시 41점 기준으로는 무작위 토큰에서 97.3% 일치하고 recall은 0.684다. 이 차이는 SDF가 아니라 LQR 추적 때문이다.
- **디코더(M4)**
  - 교정량 0이면 원래 초안이 **비트 단위로 그대로** 나온다. navtest 초안 12,146개와 사람 궤적 12,146개, float32/64, mode A/B에서 확인했다.
  - |δa|는 [-4, +2] 안에 있다.
  - 왕복 재현(교란 초안 → 사람 궤적 복원)은 98.3%가 0.1 m 이내다.
- **metric cache와 사람 궤적**
  - navtest 20토큰을 다시 캐시해서 120번 채점했다. 결과가 완전히 같다.
  - 사람 궤적 오차는 1.8e-6 m다(float32 변환 오차뿐).
- **M8 대리 손실 사전 기준**(dev 800토큰, 212개 log, 공식 채점 궤적 32,269개)

  | 기준 | 결과 | 95% CI | 판정 |
  |---|---|---|---|
  | A1 NC recall ≥ 0.70 | 0.948 (471/497) | 0.924–0.966 | **통과** |
  | A2 사람 궤적 오경보 ≤ 2% | 2.75% (22/800) | 1.7–4.1% | **미달** |
  | A3 P(공식으로 고쳐짐 \| 대리 손실이 고쳤다고 판단) ≥ 0.6 | 0.510 (349/684) | 0.472–0.548 | **미달** |

  - A2: 걸린 사람 궤적 22개는 모두 물체와 0.16–0.35 m 떨어져 지나갔다. margin 0에서는 오경보가 0%다.
  - A3: 실패 원인은 교정이 틀려서가 아니라 0.3 m margin에서 판단이 부정확하기 때문이다.
    - 대리 손실이 "고쳤다"고 한 684건 중 실제 NC 실패는 379건이었고, 그중 92%(349건)가 공식으로도 고쳐졌다.
    - 나머지 305건은 공식 NC가 원래 통과하던 근접 통과였다.
- **초안 은행 공식 실패율**(1차 pass 결과, 유효 초안만. 실패 = NC<1 또는 DAC<1 또는 DDC<1)

  | | 토큰 | 유효 초안 비율 | 실패율 | NC<1 | DAC<1 | TTC<1 | 실패율(+TTC) |
  |---|---|---|---|---|---|---|---|
  | train | 14,844 | 94.5% | **12.1%** | 4.8% | 8.5% | 10.8% | 17.6% |
  | dev | 5,837 | 94.7% | **12.1%** | 5.2% | 8.1% | 11.3% | 17.7% |

  - 계열별(train): identity 0%, small 2.3%, L-const 2.7%, ignore-brake 3.4%, creep 80%(455개), lateral 21%, combined 25%, CV 66%, heading-drift 20%(10개).
  - 목표 20–35%에 못 미친다(§4 D2).
  - 채점 오류는 0건이다.

### 1.3 통합 smoke(`tools/refiner/integrate_smoke.py`)

- 대상: 실제 초안 은행, 라벨, 물체, SDF, centerline, teacher, train 12 + dev 8토큰. 각 토큰은 서로 다른 log에서 뽑았다.
- 결과 파일: `report/refiner_T/integration_smoke.json`. 실행 시간은 170 s다.

| 검사 | 내용 | 결과 |
|---|---|---|
| A 패킹 | 6개 part를 실제 출처로 pack | 20/20행 완료. 초안은 npz와, 라벨은 채점 행과 **완전히 같음**. centerline 최대 1,148 vertex(잘림 0). 물체 최대 306(버림 0) |
| B1 라벨 재현 | pack의 원래 초안을 `score_token`으로 다시 채점 | 260궤적에서 **불일치 0**(8개 값 + pdm_progress_eff) |
| B2 수정 없음 = 원래 점수 | 학습 안 된 net(두 arm) → 디코딩 → `eval_refiner` → 채점 CLI | τ1 바이트 == τ0 바이트. θ=0(전부 "수정")에서도 7개 공식 값이 은행 라벨과 **정확히 같음**(arm마다 104개). PDMS 0.801211…로 동일 |
| C 대리 손실 연결 | 학습 루프의 SceneBatch vs 대리 손실 자체 builder | col, dac, cmf, mod 차이 0. prog 차이 ≤ 7.2e-7(centerline float32 저장). UNKNOWN 판정 동일(비율 0) |
| D 기울기 | 156개 초안 batch, 두 arm | 모든 기울기 유한 |
| E CLI 학습 | `train_refiner.py` 두 arm, 4 step | loss와 grad norm 유한. 실제 대리 손실 사용 |
| F CLI 평가 | predict → score → report. θ = p_g 중앙값이라 두 gate 분기가 모두 나옴 | gate 적용 궤적을 직접 채점한 값과 섞은 값이 **104/104 일치**(arm마다) |

D 기울기 검사의 세부:

- step 0에서는 head(8/119 파라미터)에만 기울기가 흐른다. 교정 head의 마지막 층을 0으로 초기화했기 때문이고, 의도한 동작이다. step 1부터는 119/119에 흐른다.
- 8 step 동안 안정적이었다.
- step 0의 correction loss는 두 arm에서 같다(0.04913). 파라미터 수는 trunk가 두 arm 모두 3,375,181이고, adapter만 41,152(T)와 0(none)으로 다르다.

과적합 탐침(arm none, gate 제외, lr 3e-4, 같은 batch로 60 step):

- correction loss가 0.0491에서 0.0436으로 **11% 줄었다**. 충돌 항은 23%, DAC 항은 32% 줄었다.
- 처음 약 40 step 동안은 모든 초안에 거의 같은 작은 교정을 낸다. 그 뒤부터 초안마다 다른 교정(최대 1.2 m)을 배운다.
- 결과는 `integration_smoke.json`의 `D2_overfit_train`에 있다.
- 같은 탐침을 lr 1e-3으로 200 step 돌렸다(`_integrate/logs/overfit_1e-3.log`).
  - 균일 교정 구간은 약 60 step으로 더 길었다.
  - 200 step에서 correction loss는 0.0491에서 0.0286으로 **42% 줄었다**. 충돌 항은 39%, DAC 항은 56% 줄었다.
  - 교정 크기는 초안 중앙값이 0.03 m, 최대가 1.8 m다. 대부분은 거의 건드리지 않고 일부만 크게 고치는 형태로 수렴한다.

### 1.4 통합 단계에서 고친 것

1. **centerline padding** `CL_MAX`: 1024 → 1536(`data.py`).
   - `surrogate.py`가 centerline crop을 +150 m에서 +250 m로 늘렸다(08:59). 그래서 실제 vertex 수가 최대 1,153개가 되어 pack에서 잘리고 있었다. `test_data.py`가 이것으로 실패했다.
2. **물체 padding** `A_MAX`: 640 → 800.
   - 전체 물체 빌드에서 train 최대 781개가 나왔다(640 초과 14토큰). dev 최대는 592다. 이제 train/dev에서 버려지는 물체가 없다.
   - pack 형식 버전은 `refiner_pack_v2`다. 토큰당 409 KiB이므로 train은 약 10.1 GB, dev는 약 3.4 GB다.
3. **학습 기본값**: `train_refiner.py --surrogate` 기본값을 `auto`에서 `real`로 바꿨다. 대리 손실이 없거나 깨지면 stub으로 조용히 넘어가지 않고 오류가 난다. stub은 `--surrogate stub`으로만 쓸 수 있다.
4. **GPU 스크립트**
   - 동시에 도는 run의 평가를 `flock`으로 한 줄로 세웠다. 평가마다 채점 worker 4개를 쓰기 때문이다.
   - 로그를 덮어쓰지 않고 이어서 쓰게 했다.
5. **`stageT_decision.py`의 버그**: 입력 DataFrame을 제자리에서 바꾸던 버그를 테스트로 찾아 고쳤다.

## 2. 백그라운드 작업(09:29 기준)

| 작업 | PID | 진행 | 예상 완료 | 로그 | 중지 |
|---|---|---|---|---|---|
| metric cache (worker 4) | 2558232 (pgid 2558230) | 새 토큰 19,210/22,526, 오류 0, 약 2.0 tok/s | 약 09:57(+ manifest 5분) | `metric_cache/logs/build.log` | `kill -TERM -- -2558230` |
| SDF (worker 2, 15분마다 추적) | 2573988 (pgid) | 이번 pass 대상 14,226개 중 13,561개 처리(6,475개 새로 생성). metric cache를 뒤따름 | 약 10:15–10:30 | `sdf/logs/build_navtrain.log` | `kill -TERM -- -2573988` |
| 초안 은행 생성·채점 (worker 2) | 2949694 (pgid) | 1차 pass 완료(train 14,844, dev 5,837, 오류 0). 2차 pass에서 train 5,317개 생성 후 채점 중(오류 0). 15분마다 다음 pass | 약 10:30–10:45 | `drafts/logs/run.log` | `kill -TERM -- -2949694` |
| **packing (이번에 실행, worker 2)** | 3163541 (pgid 3163536) | 1차 pass: 학습에 쓸 수 있는 행이 dev 4,586/7,930, train 14,243/23,820. 15분마다 다시 pass | 상류 완료 후 약 10:45–11:15 | `packed/logs/follow.log`, `packed/<split>/follow_status.json` | `kill -TERM -- -3163536` |
| 물체 전체 빌드 (이번에 실행, worker 2) | 3127507 (종료) | **완료**: train 24,000, dev 8,000, 오류 0, 약 4분. 물체 수 최대 781(train) / 592(dev) | – | `objects/logs/build_full.log` | – |

- 로그 경로의 앞부분은 모두 `/home/external-user/ssd/yongjae_refiner/`이다.
- 모든 작업은 같은 명령으로 다시 실행하면 이어서 진행한다.
  - packing: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nohup nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/pack_follow.py --splits dev,train --workers 2 --interval 900 --max-h 12 > /home/external-user/ssd/yongjae_refiner/packed/logs/follow.log 2>&1 &`
  - packing은 모든 part가 채워지면 스스로 멈춘다. 상류 작업이 모두 끝났는데 새로 채울 것이 없을 때도 멈춘다.
- 참고: frame gap 토큰(train 180개, dev 70개)은 초안을 만들지 않으므로 학습에서 빠진다.

## 3. IMPL_SPEC과 다른 점

모두 해당 모듈 docstring 맨 위에 적혀 있다. 인터페이스는 바꾸지 않았고, 선택 인자만 추가했다.

- **geometry/decoder**
  - 경로 매개변수는 채점기의 polyline arc length다. 0.2 m 이내 정지 떨림 점은 knot에서 뺐다.
  - 곡률 투영은 점별 방식이다(K = max(κ_lim, 1.1·|κ_draft|), 6회 반복). spec 식을 그대로 쓰면 원래 경로가 한계를 넘는 5.6–7% 초안에서 횡 교정이 모두 막힌다.
  - 끝 조건은 not-a-knot이다. 곡률은 1/m, g는 0.2에서 잘랐다(둘 다 flag를 남김).
  - 교란 전용 mode 'P'를 추가했다. CV 무효 규칙과 heading-drift 크기는 직접 정했다.
- **채점기**: 출력 key를 추가했다(mult, ttc_track 등). `--tokens`는 선택 사항이다.
- **M6**
  - UNKNOWN 정의에 "원점에서 R − 10 m 밖"을 추가했다.
  - 도달 거리는 GT ego 뒤차축에서 잰다.
  - 정적 물체의 속도는 0이다(metric cache와 같음).
  - `frame_idx`는 frame dict의 필드 값이다.
- **SDF**
  - distance transform 대신 셀 중심의 정확한 부호 거리를 쓴다(DT는 오차 0.07–0.11 m, 격자 끝 오류가 있었다).
  - 유효 영역은 가장자리에서 0.125 m씩 줄었다. 값은 ±10 m로 잘랐다. 저장은 subset별(navtrain/navtest)이다.
- **split/cache**
  - 모든 log를 한쪽(train 또는 dev)에 배정한 뒤 log 안에서 같은 비율로 토큰을 솎았다. 층화 기준은 (city, part)다.
  - E의 cache 9,000개 중 8,997개를 재사용했다(복사).
  - a0는 부호 있는 종가속도다.
- **초안 은행**
  - 실패율 목표 20–35%는 A_p 범위만으로 닿을 수 없다. 가장 넓은 범위에서도 14%라서 spec 범위 U[0.2, 1.3]을 유지했다.
  - 기각된 계열의 초안은 사람 궤적 바이트에 `valid=False`를 붙여 둔다.
  - 운동학 검사는 0.5 s keyframe에서 사람 궤적 대비로 한다.
- **대리 손실**
  - keyframe 편안함 기준은 한계의 1.0배다. spec은 0.9배지만, 0.9배로는 사람 궤적 위반이 1.6–2.1%라 spec 자체의 수용 기준(≤1%)을 넘는다.
  - t0 jerk 항은 뺐다.
  - 사람-물체 겹침 mask는 정확한 SAT 검사로 한다.
  - 비용은 n = 1..40의 평균이다.
  - centerline은 [-30, +250] m로 잘랐다.
  - TTC와 DDC 대리 손실은 없다.
- **모델/학습**
  - u_d 69차원에 ego (vx, vy, ax, ay)를 넣었다. spec 목록은 65개다.
  - 두 번째 flag는 "연장 곡률이 잘렸는지"다.
  - 마지막에 LayerNorm을 넣었다. grad clip은 1.0이다. gate BCE는 유효 초안만으로 계산한다.
  - teacher z-score 통계는 run이 실제로 학습하는 토큰(최대 2,048개)으로 계산한다. 그래서 OOF 평가 토큰이 통계에 들어가지 않는다.
  - 물체 padding은 800, centerline padding은 1536이다.
  - loss weight는 placeholder다(col 1, dac 1, prog 2, cmf 0.1, mod 0.1, gate 0.5).
- **판정 스크립트**(`stageT_decision.py`, 사전 규칙을 해석한 부분)
  - P2는 "NC+TTC 실패 감소량 = none − T"(pp)로 정의했다. 규칙의 "하한 > 0"과 방향을 맞추기 위해서다.
  - 규칙의 PASS 줄에는 DAC/DDC 비열등이 없다. 그래서 판정에 섞지 않고 옆에 따로 보고한다.
  - θ는 OOF 예산 안에서 가장 작은 θ다. 이것이 가장 많이 교정하는 gate다.
  - seed는 초안별로 평균한다.

## 4. 남은 문제와 위험

### GPU 학습 전에 결정할 것

- **D1. M8 기준 A2, A3 미달.** [2026-09-28 결정] 사용자 선택 'D1-b'는 채팅에서 제시한 순서 기준으로 '여유(margin)를 줄여 train 쪽에서 재확인'(아래 목록의 (c))이다. 아래 목록의 (b) M7b가 아니다. 결과: m8_recheck/, PRESTATED_DECISION_RULE.txt 수정 2 (m_col 0.15, m_dac 0.05). M7b는 구현하지 않았다.
  - (원래 목록) spec은 이때 "보고 후 M7b(미분 가능한 추적 근사)를 고려"하라고 한다. 선택지는 다음과 같다.
  - (a) 현재 대리 손실로 학습을 진행한다. A1은 통과했고, A3 미달은 margin 0.3 m의 정밀도 문제이며, 실제 NC 실패 수정률은 92%다.
  - (b) M7b를 구현한다. 교정 후 남은 NC 실패 82건 중 37건은 원시 기준 궤적으로는 보이지 않는다. 그중 36건은 LQR 추적 상태에서 잡힌다.
  - (c) m_col이나 사람 mask 규칙을 바꾼다.
  - 대리 손실을 바꾸면 학습을 다시 해야 하므로 먼저 정해야 한다.
  - 참고로 DAC margin 0.2 m에서 사람 오경보는 6.1%다(margin 0이면 0.4%).
- **D2. 초안 은행 실패율 12%**(목표 20–35%). gate의 양성 가중치는 약 8–9다. 목표에 가까이 가는 방법은 다음과 같다.
  - TTC를 실패 라벨에 넣는다(17.6%).
  - 가속 초안이 centerline을 쓰게 한다(pilot 실패율 14.2%, 유효 비율 99.5%).
  - 횡 오프셋을 넓힌다.
  - slot 구성을 바꾼다(13개 중 4개가 identity/small).
  - 초안 은행을 바꾸면 pack의 drafts와 labels part를 다시 만들어야 한다(`--redo drafts,labels`).
- **D3. loss weight 후보.** 사전 규칙은 "θ와 weight를 교차 적합으로 고른다"고 한다. weight 후보 하나당 교차 적합 run이 30개 필요하다. 후보 목록(예: col 2배, mod 0.3배)은 정해야 한다.
- **D4. GPU 예산.**
  - GPU 속도는 아직 재지 않았다. fp16을 GPU에서 돌려 본 적도 없다(CPU autocast만 확인).
  - 추정은 이렇다. 한 epoch는 약 2,100–2,700 step이고 40 epoch면 약 90k–110k step이다. arm T는 teacher 파일 읽기만으로 step당 약 0.085 s가 든다(loader worker 2). 그러면 run 하나에 약 2–5시간이고, 36개 run(교차 적합 30 + 최종 6)이면 약 70–180 GPU시간이다.
  - early stopping(patience 6)으로 줄어들 수 있다. §5의 시간 측정 pilot을 먼저 돌리기를 권한다.

### 알려진 한계와 위험

- **mode A 불감대**: 제어점이 0 위로 밀리면 기울기가 0이다. z = 0에서 제동 쪽 기울기 크기는 절반이다(부호는 맞음). leaky 변형은 없다.
- **학습 초기의 균일 교정 구간**: 과적합 탐침에서 처음 수십 step 동안 모든 초안에 같은 교정을 낸다. 실제 학습 곡선은 GPU에서 확인해야 한다.
- **"small" 초안의 2.3%가 실패**한다(주로 도로 가장자리 DAC). gate는 이것을 양성으로 본다. CV 초안(66% 실패)이 은행 실패의 큰 몫을 차지한다.
- **pack은 part 단위로 한 번 채우면 끝**이다. 상류 산출물(초안 은행 config, 대리 손실의 centerline crop 상수)이 바뀌면 해당 part를 `--redo`로 다시 만들어야 한다.
- **navtest는 아직 준비하지 않았다**(navtest human npz, pack). 최종 보고 전에 필요하다.
- **예전 smoke 데이터**: `smoke_refiner.py`의 `_smoke/packed`는 예전 layout(640/1024)이다. 다시 돌리려면 그 디렉터리를 지워야 한다(pack이 layout 불일치를 거부함). 이제는 `integrate_smoke.py`가 그 역할을 한다.
- **채점기와 공식 eval CSV의 작은 차이**: 저장된 모델 궤적을 다시 채점하면 EP가 최대 4.2e-4 다르다(batch 문제는 아님). 저장된 pkl을 만든 GPU 추론과 eval 때 추론이 달랐던 것으로 보인다.
- **frame gap 정의 차이**: 0.75–0.88%와 설계 문서의 1.34%를 맞춰 보지 않았다. train 쪽 log 6개는 반올림 때문에 토큰을 하나도 받지 못했다.
- **shapely 경고**: centerline에 길이 0인 선분이 있어서 `line_locate_point` RuntimeWarning이 난다. 투영 결과는 유한하고 공식 채점기와 같은 linestring을 쓰므로 무해하다(200토큰 확인).
- **해석상의 한계**(blocker 아님): teacher가 navtrain 전체로 학습되었으므로 train과 dev 모두 in-sample이다.

## 5. GPU를 받으면 실행할 명령

모두 `cd /home/external-user/yongjae/SSR`에서 실행한다. `<GPUS>`에는 GPU 번호를 공백으로 구분해 넣는다(예: `"2 3"`, 프로세스에 보이는 CUDA index).

```bash
# 0. 준비 확인: 두 split 모두 "complete": true 여야 한다 (plan 스크립트도 확인하며, FORCE=1이면 건너뜀)
grep -E '"complete"|"ready"|"n_usable"' /home/external-user/ssd/yongjae_refiner/packed/{train,dev}/follow_status.json

# 1. (권장) 시간 측정 pilot: 300 step, 로그의 sec로 step 시간 확인
OMP_NUM_THREADS=2 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/train_refiner.py \
  --arm T --fold 0 --seed 0 --gpu <GPU> --workers 2 --tag gpupilot --max-steps 300 --epochs 1 --log-every 50 --max-val-batches 20

# 2. 교차 적합: arm {T, none} x fold {0..4} x seed {0,1,2} = run 30개, 각 run은 자기 fold를 OOF로 평가
nohup tools/refiner/stageT_gpu_plan.sh crossfit "<GPUS>" > /home/external-user/ssd/yongjae_refiner/runs/logs/plan_crossfit.log 2>&1 &
#    loss weight 후보를 더 돌릴 때(D3):
#    TAG=w_col2 nohup tools/refiner/stageT_gpu_plan.sh crossfit "<GPUS>" --w col=2,dac=1,prog=2,cmf=0.1,mod=0.1 > .../plan_crossfit_w_col2.log 2>&1 &

# 3. arm별 θ(와 weight) 선택 (train OOF만 사용, dev는 읽지 않음)
CUDA_VISIBLE_DEVICES="" /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/stageT_decision.py select \
  --wtags stageT --seeds 0,1,2 --folds 0,1,2,3,4 --budget-ep 0.5 --out report/refiner_T/crossfit_selection.json

# 4. 최종 학습 (전체 fold) + dev 평가: arm {T, none} x seed {0,1,2} = run 6개
#    (weight 후보가 여럿이면 선택된 TAG와 --w로 실행)
nohup tools/refiner/stageT_gpu_plan.sh final "<GPUS>" > /home/external-user/ssd/yongjae_refiner/runs/logs/plan_final.log 2>&1 &

# 5. 사전 규칙 판정 (seed 평균, log-cluster paired bootstrap 10,000)
CUDA_VISIBLE_DEVICES="" /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/stageT_decision.py compare \
  --selection report/refiner_T/crossfit_selection.json --seeds 0,1,2 --n-boot 10000 --out report/refiner_T/decision_dev.json
```

- run 하나만 돌릴 때는 `tools/refiner/stageT_gpu_commands.sh <GPU> <T|none> <fold: -1|0..4> <seed> [--w ...]`를 쓴다. 학습 후 평가까지 한다.
  - 로그: `/home/external-user/ssd/yongjae_refiner/runs/logs/<run>.{train,eval}.log`
  - run 디렉터리: `runs/<TAG>_<arm>_fold<k>_seed<s>/`
- dev 평가만 따로 할 때:

  ```bash
  OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/eval_refiner.py all \
    --run /home/external-user/ssd/yongjae_refiner/runs/stageT_T_fold-1_seed0 --split dev --gpu <GPU> --workers 4 \
    --theta 0.5 --sweep --budget-ep 0.5
  ```

- 중단 후 재개: 같은 명령을 다시 실행한다. 학습은 `ckpt_last.pt`에서 이어가고, 끝난 run(DONE)은 건너뛴다.
- 사전 규칙에서 [default]로 표시된 값(예산 0.5점, margin 0.5 pp, 동등 범위 ±0.2점, seed 3개)은 **첫 dev 평가 전까지만** 바꿀 수 있다. `stageT_decision.py`의 인자로 바꾼다.

## 6. Run 2 changes (2026-09-28, PRESTATED_DECISION_RULE AMENDMENT 3 구현)

- 목적: run 1의 mode-A 불감대(z_lon > 0 → c 전부 clamp, 감속 0, 기울기 0) 수정(B)과 EP 예산 재정의(B+).
- **기본값은 모두 꺼져 있다.** `--lon-st-slope 0`, `--w-zdead 0`, `--budget-def all`이면 run 1과 같은 계산이다.
  - forward는 lam과 무관하게 `torch.clamp(c, max=0.0)`과 비트 단위로 같다(-0.0 포함). 그래서 `eval_refiner`는 옵션 없이 그대로 쓴다.
- 새 CLI
  - `train_refiner.py --lon-st-slope <lam>`: mode A clamp의 c > 0 구간 역전파 기울기(학습 loss 전용). mode B와 함께 쓰면 거부한다.
  - `train_refiner.py --w-zdead <w>`: `w * mean_valid(mean_i relu(z_lon_i)^2)` hinge.
  - `liveness.py --run <run> --eval <eval subdir> [--min-liveness 0.01] [--out <json>]`: `--eval` 필수(기본값 없음), `--out`이 없으면 아무 파일도 쓰지 않는다(run 1 디렉터리에 읽기 전용으로 쓸 수 있게).
  - `stageT_decision.py --budget-def {all,passing}`: select 기본 all. compare는 selection json의 `budget_def`를 따르고, 다른 값을 명시하면 거부한다.

| 파일 | 변경 |
|---|---|
| `navsim/agents/para_ssr/refiner/decoder.py` | `lon_clamp_a`(custom autograd Function: forward = torch.clamp, backward 1 / lam), `lon_live(z)`, `decode(..., lon_st_slope=0.0)`(mode A만, B/P는 무시), docstring(Known limitations에 옵션 명시) |
| `tools/refiner/train_refiner.py` | `--lon-st-slope`, `--w-zdead`; `decode_batch`/`compute_loss`에 전달; `zdead_penalty`; stats `zdead`, `live`, `loss_ex_zd`; epoch 기록 `val_loss`, `val_liveness`, `train_liveness`; config.json 기록, 값이 다르면 resume 거부(키가 없는 run 1 config는 0/0으로 간주); 인자 검사; `check_resume_config`(m_col/m_dac/lon_st_slope/w_zdead 비교)를 DONE 조기 종료 **앞**에서 실행 |
| `tools/refiner/stageT_gpu_commands.sh` | `--lon-st-slope`/`--w-zdead`를 TAG=stageT(기본값, run 1 태그)와 함께 주면 아무것도 실행·기록하지 않고 exit 2 |
| `tools/refiner/liveness.py` (새 파일) | pred.npz → liveness(유효 초안 중 pre-clamp c < 0이 하나라도 있는 비율), z_lon 전부 양수 비율, 4 s 호 길이 0.5 m 초과 단축 비율, 초안 유형별, `gate_pass` (≥ 1 %) |
| `tools/refiner/stageT_decision.py` | `--budget-def`; `final_outcomes`에 `pass_both`; `ep_loss_points(f, budget_def)`; sweep에 `ep_loss_points_all/_passing`, `n_passing`; select/compare json에 `budget_def` |
| `tools/refiner/tests/test_dead_zone.py` (새 파일) | forward 비트 동일성(float64/32/16, bfloat16, autocast, ±0), 기울기(1 / lam / 경계 / NaN), decode identity 바이트, lam과 무관한 출력, B/P 무시, 불감대 기울기, `lon_live`, net 1 step 충돌 loss 기울기(lam 0 → 0, lam 1 → ≠ 0, z를 낮추는 방향), hinge 값·기울기, config 거부(DONE인 run 포함), GPU 스크립트 TAG 가드(sandbox 복사본), liveness.py |
| `tools/refiner/tests/test_stageT_budget_def.py` (새 파일) | passing 정의(전후 DAC/NC/DDC 실패 초안 제외), all 불변, passing 예산이 구속력을 갖는 경우, select/compare/CLI |

- 결정한 것(사양에 없던 부분)
  - ST는 제안된 `c - relu(c) + lam (relu(c) - relu(c).detach())` 대신 custom autograd Function으로 구현했다. 제안 식은 c = -0.0을 +0.0으로 바꿔 clamp와 비트가 달라진다(테스트 `test_naive_relu_form_is_not_bitwise`). 기울기는 c ≤ 0에서 1(torch.clamp와 같은 경계), c > 0에서 lam, NaN에서 0(torch.clamp와 같음)이다.
  - 검증 `loss`(조기 종료와 variant 선택에 쓰는 값)에는 hinge가 포함된다. hinge를 뺀 값은 `loss_ex_zd`로 따로 기록한다. V3(w_zd 0.01)와 V1/V2를 비교할 때 어느 쪽을 쓸지는 주 세션이 정한다.
  - decode는 학습·평가 모두 autocast 밖에서 float32로 돈다(`compute_loss`가 autocast 블록 밖). 그래도 fp16/bf16/autocast forward 동일성을 테스트했다.
  - `eval_refiner.py`의 `theta_at_budget`(report.json, 참고용)은 여전히 'all' 정의다. 판정용 θ는 `stageT_decision.py select`가 정한다.
- run 1 확인(train OOF fold 0만 읽음, 파일 쓰기 없음): `liveness.py` 결과 두 arm 모두 liveness 0 %, z_lon 전부 양수 100 %, `gate_pass` false로 DECISION_RUN1.md와 같다. `select --seeds 0 --folds 0 --budget-def passing`은 두 arm 모두 θ = 0이다(passing EP 손실 T +0.0007점, none −0.0038점, n_passing 약 50.4k/57.7k). 감속이 없었으므로 예상대로다. 'all'은 −3.43 / −2.48점으로 run 1 값을 재현한다.
- 리뷰 수정(major): 이전에는 DONE이 있는 run이 config 비교 전에 exit 0으로 끝났다. 그래서 TAG를 빠뜨린 run-2 실행이 run 1 디렉터리(`stageT_*_fold0_seed0`, 둘 다 DONE, tag stageT)로 들어가 그대로 `eval_refiner.py all`로 넘어가고, run 1의 `eval_train_fold0` 출력을 덮어쓸 수 있었다. 이제 비교가 DONE 확인보다 먼저 돌아서 불일치면 non-zero로 끝나고(`set -e`로 스크립트가 멈춘다), 셸 스크립트도 새 플래그 + TAG=stageT를 거부한다. 같은 옵션으로 다시 부르면 여전히 DONE에서 exit 0이고 eval을 다시 돈다(기존 동작).
- 테스트: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python -m pytest -q tools/refiner/tests -x` → **257 passed, 19 warnings in 147.04s**. 기존 테스트는 모두 그대로 통과했고, 새 테스트 파일 2개(dead zone 44개, budget_def 4개)가 추가됐다.

## 7. Run 3 changes (2026-09-28, PRESTATED_DECISION_RULE AMENDMENT 4 (1) 구현; m_ttc 선택 (2)는 아직 안 함)

- 목적: P2(NC+TTC)의 TTC 부분에 대응하는 학습 surrogate 항 C_ttc 추가.
- **기본값은 꺼져 있다.** `--w`에 ttc가 없으면 weight 0, `--m-ttc`가 없으면 SurrogateConfig().m_ttc = 0.0. 실제 packed train
  batch 3개 x {기본값, run 2 설정(m_col 0.15 / m_dac 0.05 / lam 0.1)}에서 변경 전 코드와 loss, 모든 stats, 모든 파라미터
  기울기가 비트 단위로 같다(추가된 것은 로그 키 `t_ttc` 하나).
- C_ttc: n = 1..40, delta in (0.3, 0.6, 0.9) s. ego box를 v_n delta만큼 h_n 방향으로 평행이동(v_n = ref_speed, 전방 차분),
  물체는 dense 시각 n + 10 delta(0..5 s = 51개 시각, `SceneBatch.boxes_ttc/obs_ttc`). 저장 범위 밖이거나 OBS가 아닌 쌍은
  **제외**(clamp 안 함). 같은 smooth g, 같은 물체 weight, not-behind는 **투영 전** ego 중심/heading(n) vs 물체(n + k),
  human mask는 human dense reference의 **투영된** box가 물체(n + k)와 겹치는지(exact SAT). 속도 0이면 투영 0(delta 0 = C_col,
  테스트로 확인). 공식 규칙 중 정지 ego 제외(speed < 5e-3)는 AMENDMENT 4 식에 없으므로 기본 off(`ttc_min_speed` 옵션).
- 새 CLI: `train_refiner.py --m-ttc <m>`, `--w ...,ttc=<w>`(모르는 term 이름은 이제 거부). config.json에 `m_ttc`,
  `weights.ttc`, `surrogate_cfg.{m_ttc,ttc_deltas,ttc_min_speed}` 기록, 값이 다르면 resume 거부(DONE 확인 **앞**, 키 없는
  run 1/2 config = unset/0). stub + TTC 옵션은 거부. `stageT_gpu_commands.sh`는 `--m-ttc`/`ttc=`를 TAG stageT/stageT2와 함께
  주면 exit 2 (run 3는 예: `TAG=stageT3 ... --m-col 0.15 --m-dac 0.05 --lon-st-slope 0.1 --w col=1,dac=1,prog=2,cmf=0.1,mod=0.1,ttc=1 --m-ttc <선택값>`).
- `tools/refiner/ttc_flag_check.py`(새 파일): D1-b sweep pool 일부에서 flag vs 공식 TTC 라벨 confusion (선택용 아님).
  100 train token / 1,218 bank draft, m_ttc 0, raw reference: proj flag TP 81 / FP 20 / FN 8 / TN 1,109 (recall 0.91, FA 0.018);
  col@0.15 OR proj@0도 같은 값; human k=0 100개는 공식 TTC 실패 0, flag 0.
- 비용: 실제 batch(8 token, A 221)에서 ttc_cost forward가 collision_cost의 약 3.5배(CPU).
- 테스트: `tools/refiner/tests/test_ttc.py`(27개). `test_train_refiner_margins.py`의 기본 config 비교 한 줄은 JSON 왕복으로
  바꿨다(ttc_deltas tuple -> list). 전체: 284 passed.

### 7b. m_ttc 선택 (AMENDMENT 4 (2), 2026-09-29, train D1-b sweep pool만 사용)
- 규칙과 구현 세부: `report/refiner_T/ttc_select/SELECTION_RULE.txt`(숫자를 보기 전에 작성). 스크립트 `tools/refiner/ttc_select.py`,
  테스트 `tools/refiner/tests/test_ttc_select.py`(6개). 결과는 `ttc_select/{ttc_select.json, ttc_select.csv}`, draft별 행은
  `<data>/ttc_select/rows.parquet`.
- 800 token / 800 log, bank draft 9,805개(human 800개), 공식 TTC 실패 1,036개(모두 perturbed, human 0). 오류 0.
  sweep shard와 대조: TTC 라벨 불일치 0, col_g와 g_s_sat 차이 최대 0.0.
- T1 / T2(점추정, log bootstrap 95 %): 0.00 -> 0.856 / 0.0037, 0.05 -> 0.864 / 0.0037, 0.10 -> 0.872 / 0.0050,
  0.15 -> 0.880 [0.842, 0.916] / 0.0063 [0.0013, 0.0125]. col@0.15 단독: 0.437 / 0.0025.
- 네 값 모두 feasible, 규칙대로 **m_ttc = 0.15**(grid 최댓값). run 3: `--m-ttc 0.15`.

### 7c. 리뷰 수정 (2026-09-29): TTC 가중치 0이면 autograd 없이 계산
- 문제: 가중치 0(run 1/2 방식)에서도 `ttc_cost`가 autograd 그래프를 만들어 CPU 12 batch 기준 peak RSS가 +725 MB 늘었음(loss 값은 동일).
- 수정: `surrogate.surrogate_terms(..., ttc_grad=True)` 인자 추가. False면 TTC 항만 `no_grad`로 계산(값 동일, 그래프 없음).
  `surrogate_loss`는 `ttc_grad=bool(w_ttc)`, `train_refiner.compute_loss`는 `terms_fn(..., ttc_grad=bool(w_ttc))`로 넘김
  (`surrogate_terms_batch`, `stub_terms` 모두 인자 받음). `t_ttc` 로그는 그대로.
- 확인(real train batch 12개, run 2 설정, CPU): 수정 전 코드(bak/)와 loss, stat, 파라미터 gradient 1,380개 모두 bit 단위 동일(`t_ttc`만 추가).
  peak RSS 증가 +1366 MB(수정 전 코드) / +1620 MB(수정 후, 가중치 0) / +2082 MB(가중치 1, m_ttc 0.15). 남은 차이는 로그용 TTC forward의
  일시 메모리. step 시간은 여전히 약 +15%(로그용 forward 계산).
- TTC 값과 flag는 바뀌지 않으므로 m_ttc 선택(7b)은 다시 하지 않음: m_ttc = 0.15 유지.
- 테스트: `test_ttc.py`에 `test_ttc_no_graph_at_weight_zero` 추가, `test_compute_loss_ttc_weight`에 ttc_grad 전달 확인 추가. 전체 291 passed.

## 8. Run 4 changes (2026-09-29, PRESTATED_DECISION_RULE AMENDMENT 6 + addendum 구현)

- 목적: arm 4개 {none, T, M, TM}을 train_logs 토큰만으로 처음부터 학습(tag `stageT4`). 비교 C1 T−none, C2 M−none, C3 TM−T,
  C4 TM−M, 각각 기존 P1/P2/P2-ni/P3 규칙, Bonferroni 98.75 % CI. 손실 항은 run 3와 같다(addendum). navtest에서는 EPDMS도 보고(판정 아님).
- **기본값은 모두 꺼져 있다.** 새 옵션 없이 부르면 arm T / none은 이전 코드와 비트 단위로 같다. 확인: 실제 packed train
  24 토큰 x 3 step(run 3 설정) + dev 8 토큰 predict를 변경 전 코드(백업 사본)와 새 코드로 각각 돌려 `ckpt_best/last.pt`의 모든
  텐서, `log.jsonl`(sec 제외), `norm.npz`의 mean/std, `pred.npz`의 모든 배열이 같다. config.json만 `code` 해시와 argparse가 넣는 null 키
  3개(`token_subset`, `dev_token_subset`, `resmap_root`)가 다르다. `stageT_decision.py select`/`compare`(옛 두-arm CLI, budget-def all/passing)는
  합성 행에서 출력 JSON이 바이트 단위로 같다.

### 8.1 새 CLI

| 도구 | 옵션 | 뜻 |
|---|---|---|
| `train_refiner.py` | `--arm M` / `--arm TM` | ReSMap 지도 teacher(M), BEVFusion + ReSMap(TM) |
| | `--token-subset <parquet>` | packed 행을 이 토큰으로 제한(inner-val 선택 전). config.json `token_subset` = path, sha256, n_rows, n_tokens, n_packed_rows, n_train_rows, n_ival_rows. M/TM은 필수이고 모든 토큰이 ReSMap cache에 있어야 한다 |
| | `--dev-token-subset <parquet>` | 기록만 함(config.json `dev_token_subset`). eval_refiner `--split dev`의 기본 subset |
| | `--resmap-root <dir>` | ReSMap cache 경로 교체(테스트용, meta sha256 검사) |
| `eval_refiner.py` | `--token-subset <parquet>` | train: run config의 subset을 자동으로 쓰고 sha256을 다시 확인(명시하면 같은 파일이어야 함). dev: 기본값 = run의 `dev_token_subset`. navtest: 거부 |
| | `--shuffle-which {det,map,both}` | `--shuffle-teacher-seed`와 함께. 기본값 = arm의 teacher(T det, M map, TM both). both는 derangement 하나를 두 branch에 같이 씀 |
| | `--drop-branch {det,map}` | TM만. 그 branch의 **정규화된** 입력을 0(= 채널별 학습 평균 feature)으로 |
| | `--eval-name NAME` | 출력 = `<run>/NAME`. shuffle / branch-drop은 `--eval-name`(또는 `--out`)이 필수이고 기본 이름(eval_dev, eval_navtest, eval_train_fold*)은 거부 |
| | `--resmap-root <dir>` | 테스트용 |
| `stageT_decision.py` | `select --arms T,M,TM,none` | arm별 θ(원래도 arm 이름에 무관, 그대로) |
| | `compare --pairs "T:none,M:none,TM:T,TM:M"` | A:B = A − B. 쌍마다 기존 규칙(A가 R_T 역할). 원소는 `ARM@EVAL`도 가능(예: `TM@eval_dev_drop_det:TM`, 대조군 비교) |
| | `compare --ci-level L` | 결과 규칙에 쓰이는 모든 CI(P1, P2, P2-ni DAC/DDC, P3, sanity)를 양측 L 구간으로. 기본 0.95, run 4는 0.9875 |
| `stageT_gpu_commands.sh` | arm M / TM, `--token-subset`, `--dev-token-subset` | TAG가 stageT/stageT2/stageT3이면 exit 2(`--resmap-root`, 모르는 arm도 거부). OOF 평가에 train subset을, `DEV_EVAL=1`이면 dev 평가에 dev subset을 넘긴다(기본 off) |
| `epdms_navtest.py` (새 파일) | `bench` / `score` / `report` | navsim v2 EPDMS(§8.5) |

### 8.2 파일 변경

| 파일 | 변경 |
|---|---|
| `navsim/agents/para_ssr/refiner/adapters.py` | `AdapterM`(AdapterT 형태, 256→128 GELU 128→64), `AdapterTM`(det/map branch, 각자 ChannelZScore, concat 128 → 1x1 128→64, `set_drop_branch`), `build_adapter(..., map_mean, map_std, seed)`, `MAP_SEED_OFFSET`, `FUSE_SEED_OFFSET`, `ARMS` |
| `navsim/agents/para_ssr/refiner/refiner_net.py` | `RefinerNet(..., map_norm_mean, map_norm_std)`; T / none는 이전과 같은 호출 |
| `navsim/agents/para_ssr/refiner/data.py` | `ConcatTeacher`(det 0..255, map 256..511, f16 [512,50,100]), `arm_teachers(arm, split)`, `load_token_subset`, `restrict_rows` |
| `tools/refiner/train_refiner.py` | 위 옵션, `select_rows(..., subset_tokens)`, `load_run_norms`, `norm_map.npz`, config 기록(`token_subset`, `dev_token_subset`, `resmap_root`, `resmap_sha_head`, `norm_files`), resume 시 subset sha256 비교(DONE 앞) |
| `tools/refiner/eval_refiner.py` | 위 옵션, `check_run4_eval_args`, `resolve_token_subset`, `run4_teacher`(sha head 확인, ReSMap coverage 확인), predict_meta에 `token_subset` / `drop_branch` / `eval_name` / `teacher_shuffle.which`(새 옵션이 켜졌을 때만) |
| `tools/refiner/stageT_decision.py` | `ci_percentiles`, `cluster_boot(level)`, `parse_pairs`, `arm_rows`, `pair_result`, `compare_pairs`; 옛 `compare`는 `ci_level`을 명시할 때만 키 하나 추가 |
| `tools/refiner/stageT_gpu_commands.sh` | arm / TAG 가드, subset 전달 |
| `tools/refiner/epdms_navtest.py` (새 파일) | §8.5 |
| `tools/refiner/tests/refiner_synth.py` | `fake_resmap_bev`, `make_fake_resmap`(meta가 유효한 가짜 ReSMap cache, 물리적 순열 옵션) |
| `tools/refiner/tests/test_stageT4.py` (새 파일, 29개) | param count / trunk 동일성, TM branch 초기값 = T / M adapter, branch drop 의미, ConcatTeacher, subset 선택, 학습(norm 파일, config, 거부), 평가(subset, shuffle-which = 물리적으로 순열한 cache, drop-branch, 거부 10종), CI level, 4-arm select/compare, 옛 compare와 T:none 일치, `ARM@EVAL`, GPU 스크립트 가드, EPDMS 도구(토큰 subset, load_sets, report, refined 오류 draft 제외, 중복 arm 거부) |
| `report/refiner_T/run4/epdms_bench.json` (새 파일) | EPDMS 처리량 측정 결과 |

### 8.3 결정한 것 (사양에 없던 부분)

- **초기화 seed.** M adapter는 `seed + 7919 + 104729`로 만든다. 그래서 M의 초기값은 T와 다르다. TM은 det branch를 `seed + 7919`(= T adapter와 파라미터 단위로 같음), map branch를 `seed + 7919 + 104729`(= M adapter와 같음), fuse를 `seed + 7919 + 1299709`로 만든다. T / none은 이전 코드 경로 그대로다. trunk는 네 arm이 같다(테스트).
- **TM fuse.** 사양 문구대로 concat 뒤에 활성화 없이 1x1 128→64만 둔다. 각 branch 마지막 1x1(128→64)과 fuse가 모두 선형이므로, 함수 공간은 "두 branch hidden(128)의 선형 결합"과 같다. 파라미터 수만 늘어난다(90,560).
- **param counts.** none 0, T 41,152, M 41,152, TM 90,560. trunk 3,375,181(공통).
- **branch drop**은 adapter 안에서 정규화 입력을 0으로 둔다(fp16 입력으로는 평균을 정확히 만들 수 없음). 평균 feature를 넣은 결과와 1e-5 안에서 같고, drop한 입력을 바꿔도 출력이 비트 단위로 같다. 체크포인트에 저장되지 않는 속성이다.
- **norm.** TM의 `norm.npz`는 T와 똑같이 계산한다(같은 subset·fold면 같은 파일 내용, 테스트). 각 norm 파일의 meta에 cache root와 sha head, `branch`, `token_subset_sha256`이 들어간다. config.json `norm_files`에도 파일별 sha head를 적는다.
- **subset 적용 순서.** fold 필터와 토큰 필터는 교환 가능하다. inner-val log는 제한된 log 집합에서 기존 hash 규칙으로 다시 고른다. 결정적이다.
- **dev 기본 subset.** "run config의 dev subset"이 있으려면 학습 때 기록해야 하므로 `--dev-token-subset`(기록 전용)을 추가했다.
- **train OOF 평가**는 run config의 subset을 자동으로 쓴다. 명시한 파일이 다르면(sha256) 거부한다. navtest는 언제나 subset 없이 평가한다.
- **shuffle.** TM의 both는 derangement 하나를 두 branch에 같이 쓴다(토큰이 다른 토큰의 입력 전체를 받는다). arm T에서 `--shuffle-which`를 주지 않으면 predict_meta 형식이 run 3와 같다.
- **compare --pairs.**
  - sanity는 쌍마다 그 쌍의 두 arm으로 판정한다. 모든 arm 기준 값은 `sanity_any_arm`으로 따로 보고한다.
  - draft는 쌍마다 (token, k) inner join으로 맞춘다. 짝이 없는 수를 보고한다.
  - 대조군 비교를 위해 `ARM@EVAL` 원소를 추가했다.
  - 옛 두-arm 경로는 코드도 출력도 그대로다.
- **GPU 스크립트.** `--resmap-root`도 새 TAG를 요구한다. 모르는 arm은 거부한다.
- **EPDMS**는 §8.5에 따로 적었다.

### 8.4 CPU smoke (2026-09-29, 실제 데이터, `CUDA_VISIBLE_DEVICES=""`, OMP 2, loader workers 2, nice 10; 출력은 scratch)

- **학습.** 조건은 run 3 설정 + `--token-subset train_trainlogs.parquet --dev-token-subset dev_trainlogs.parquet --limit-tokens 160 --max-steps 20`, fold 0.
  - config 기록: train 160 / inner-val 40 토큰, subset n_rows 19,732, 학습 가능한 packed 행 19,588(전체 23,820).
  - ReSMap sha head `0eaeda793402a804`, BEVFusion `cddf943ffec8d6a8`.

  | arm | adapter params | loss step 1 / 5 / 10 / 15 / 20 | val loss | s/step (median, 1-20) | peak RSS |
  |---|---:|---|---:|---:|---:|
  | M | 41,152 | 0.667 / 0.729 / 0.745 / 0.725 / 0.710 | 0.753 | 1.50 | 3.27 GB |
  | TM | 90,560 | 0.670 / 0.737 / 0.745 / 0.728 / 0.712 | 0.753 | 1.50 | 3.34 GB |
  | T (참고) | 41,152 | 0.667 / 0.728 / 0.749 / 0.729 / 0.712 | – | 1.40 | – |
  | none (참고) | 0 | 0.668 / 0.727 / 0.747 / 0.732 / 0.715 | – | 1.30 | – |

  - 20 step은 배치가 매번 다르므로 loss 추세에는 의미가 없다. 확인한 것은 값이 유한하고, 모든 경로가 돈다는 점이다.
- **평가 predict**(dev 8 토큰, dev subset은 config에서 자동, 토큰당 약 0.07 s).
  - M: true / shuffle(map)
  - TM: true / shuffle det / map / both / drop det / drop map
  - 모두 정상. p_g가 true에서 달라진 최대 폭: M shuffle 0.013, TM shuffle det 0.002 / map 0.013 / both 0.015, drop det 0.003 / drop map 0.020. 20 step 모델이라 크기에는 의미가 없고, 입력이 네트워크에 도달한다는 것만 확인한 값이다.
  - navtest 8 토큰(M, TM)도 정상.
  - 거부 확인: eval-name 없는 drop, M에 drop, navtest에 subset → exit 1, 아무것도 쓰지 않음.
- **판정**(합성 4-arm 행, CLI `select --arms T,M,TM,none --budget-def passing` → `compare --pairs ... --ci-level 0.9875`, 10,000 resample).
  - 4쌍 모두 결과가 나온다. CI 백분위는 [0.625, 99.375].

### 8.5 EPDMS 도구 (`tools/refiner/epdms_navtest.py`, 보고용)

- **채점기**(report 29).
  - `/home/external-user/yongjae/navsim_v2` HEAD `0a380a9`(#151 수정 포함), navtest metric cache `navsim_v2_exp/exp/metric_cache_navtest`
  - 설정 `default_run_pdm_score`: non-reactive, human filter on, 가중치 EP 5 / TTC 5 / LK 2 / HC 2 / EC 2
  - v2 `pdm_score`와 `compute_final_scores`를 **수정 없이** 호출한다.
  - SSR의 `navsim`과 이름이 겹친다. 그래서 `bench`/`score`는 스스로 v2 환경(para_ssr_env.sh 변수)으로 다시 실행하고 `navsim.__file__`을 확인한다. `report`는 SSR 환경에서 돈다.
- **EC(two-frame extended comfort)는 계산하지 않는다.**
  - 이유: EC는 한 planner의 인접 프레임 궤적 쌍이 필요하다. 초안 은행은 토큰마다 서로 무관한 교란 13개라서 쌍이 성립하지 않는다.
  - 방법: `two_frame_extended_comfort = NaN`으로 둔다. 공식 코드가 이전 프레임이 없는 토큰에 쓰는 경로이며, EC weight가 0이 된다.
  - 원본과 모든 arm에 똑같이 적용한다.
- **원본 초안은 한 번만 채점한다**(arm과 무관). run들의 tau0 / tokens / draft_valid가 같은지 확인한다.
- tau1이 tau0과 바이트 단위로 같으면 원본 행을 복사한다(`copied_from_orig`). 무효 초안은 채점하지 않는다.
- **토큰 subset**: 정렬된 토큰 목록에서 `default_rng(seed).choice(n)`으로 뽑아 다시 정렬한다. 13개 초안을 모두 쓴다. 목록과 sha256은 meta.json과 tokens.txt에 남긴다.
- 채점은 400 토큰 단위 shard로 재개할 수 있다.
- `report`: final = arm의 선택 θ에서 p_g ≥ θ이면 tau1, 아니면 원본. arm별 EPDMS, 원본 대비 차이, 부분점수 평균, 쌍 차이(log-cluster bootstrap, 기본 98.75 %)를 낸다.
  - 채점 오류 처리(리뷰 반영): 원본 행이 오류면 그 draft는 빠진다. p_g ≥ θ인데 refined 행이 오류/비유한이면 그 draft도 빠지고 `n_refined_errors`로 센다(원본 점수로 대체하지 않음; PDMS 쪽 final_outcomes와 같은 처리). arm별 `n_orig_errors`도 보고.
  - 한 arm에 run이 둘 이상이면(seed 여러 개) `score`/`report`가 거부한다(seed 평균 없음; run 4는 arm당 seed 0 하나).
- **처리량**(`report/refiner_T/run4/epdms_bench.json`; stageT3_T의 eval_navtest에서 seed 0 토큰 50개 x 13 초안, 유효 605개, 오류 0).
  - worker 1: 16.0 traj/s. 궤적당 0.050 s, cache load는 토큰당 0.115 s.
  - worker 4: 64.7 traj/s(효율 0.93).
  - **추정**(토큰 12,101 x 13 초안 x 세트 5개 = 원본 + arm 4개, cache load는 토큰당 1회): worker 4에서 2.9 h, worker 8에서 **1.6 h**. worker 8 효율은 측정한 worker 4 효율 x 0.9로 가정했다.
  - 2 h 안에 들어가므로 **권장은 navtest 전체**(n = 12,101)다. 머신이 붐벼서 시간이 모자라면 대안은 `--n-tokens 6000 --subset-seed 0`(약 0.8 h)이다. 어느 쪽인지는 채점 전에 정한다.

### 8.6 Run 4 GPU 명령 (GPU 0: T, M / GPU 1: TM, none; run마다 학습 뒤 OOF fold 0 평가까지, dev는 따로)

```bash
cd /home/external-user/yongjae/SSR
TAG=stageT4 nohup tools/refiner/stageT_gpu_commands.sh 0 T 0 0 --token-subset /home/external-user/ssd/yongjae_refiner/splits/train_trainlogs.parquet --dev-token-subset /home/external-user/ssd/yongjae_refiner/splits/dev_trainlogs.parquet --gate-ttc 1 --m-col 0.15 --m-dac 0.05 --lon-st-slope 0.1 --w col=1,dac=1,prog=2,cmf=0.1,mod=0.1,ttc=1 --m-ttc 0.15 > /home/external-user/ssd/yongjae_refiner/runs/logs/plan_stageT4_T.log 2>&1 &
TAG=stageT4 nohup tools/refiner/stageT_gpu_commands.sh 0 M 0 0 --token-subset /home/external-user/ssd/yongjae_refiner/splits/train_trainlogs.parquet --dev-token-subset /home/external-user/ssd/yongjae_refiner/splits/dev_trainlogs.parquet --gate-ttc 1 --m-col 0.15 --m-dac 0.05 --lon-st-slope 0.1 --w col=1,dac=1,prog=2,cmf=0.1,mod=0.1,ttc=1 --m-ttc 0.15 > /home/external-user/ssd/yongjae_refiner/runs/logs/plan_stageT4_M.log 2>&1 &
TAG=stageT4 nohup tools/refiner/stageT_gpu_commands.sh 1 TM 0 0 --token-subset /home/external-user/ssd/yongjae_refiner/splits/train_trainlogs.parquet --dev-token-subset /home/external-user/ssd/yongjae_refiner/splits/dev_trainlogs.parquet --gate-ttc 1 --m-col 0.15 --m-dac 0.05 --lon-st-slope 0.1 --w col=1,dac=1,prog=2,cmf=0.1,mod=0.1,ttc=1 --m-ttc 0.15 > /home/external-user/ssd/yongjae_refiner/runs/logs/plan_stageT4_TM.log 2>&1 &
TAG=stageT4 nohup tools/refiner/stageT_gpu_commands.sh 1 none 0 0 --token-subset /home/external-user/ssd/yongjae_refiner/splits/train_trainlogs.parquet --dev-token-subset /home/external-user/ssd/yongjae_refiner/splits/dev_trainlogs.parquet --gate-ttc 1 --m-col 0.15 --m-dac 0.05 --lon-st-slope 0.1 --w col=1,dac=1,prog=2,cmf=0.1,mod=0.1,ttc=1 --m-ttc 0.15 > /home/external-user/ssd/yongjae_refiner/runs/logs/plan_stageT4_none.log 2>&1 &
```

- `DEV_EVAL`는 설정하지 않는다(기본 off). run 디렉터리는 `runs/stageT4_{T,M,TM,none}_fold0_seed0/`, 로그는 `runs/logs/stageT4_*.{train,eval}.log`.
- OOF 평가는 `flock`으로 차례대로 돈다(채점 4 CPU workers).

### 8.7 학습 뒤 순서 (사전 규칙 순서; dev는 1번을 통과한 뒤에만)

```bash
cd /home/external-user/yongjae/SSR
PY=/home/external-user/miniconda3/envs/ssr/bin/python; RUNS=/home/external-user/ssd/yongjae_refiner/runs
SP=/home/external-user/ssd/yongjae_refiner/splits; OUT=report/refiner_T/run4
ls $RUNS/stageT4_{T,M,TM,none}_fold0_seed0/DONE $RUNS/stageT4_{T,M,TM,none}_fold0_seed0/eval_train_fold0/report.json

# 1. liveness gate (train OOF fold 0; 네 arm 모두 liveness >= 1 %여야 dev로 간다. 하나라도 false면 멈추고 보고)
for A in T M TM none; do CUDA_VISIBLE_DEVICES= $PY tools/refiner/liveness.py --run $RUNS/stageT4_${A}_fold0_seed0 \
  --eval eval_train_fold0 --out $OUT/liveness_$A.json | grep -E '"liveness"|gate_pass'; done

# 2. dev 평가 (dev_trainlogs, 차례대로; GPU 하나면 충분)
for A in T M TM none; do OMP_NUM_THREADS=1 nice -n 10 $PY tools/refiner/eval_refiner.py all --run $RUNS/stageT4_${A}_fold0_seed0 \
  --split dev --token-subset $SP/dev_trainlogs.parquet --gpu 0 --workers 4 --theta 0.5 --sweep --budget-ep 0.5 \
  >> $RUNS/logs/stageT4_${A}_fold0_seed0.eval.log 2>&1; done

# 3. theta 선택 (train OOF fold 0만 읽음)
CUDA_VISIBLE_DEVICES= $PY tools/refiner/stageT_decision.py select --wtags stageT4 --arms T,M,TM,none --seeds 0 --folds 0 \
  --budget-ep 0.5 --budget-def passing --out $OUT/selection.json

# 4. dev 판정 C1-C4 (98.75 %)
CUDA_VISIBLE_DEVICES= $PY tools/refiner/stageT_decision.py compare --wtags stageT4 --selection $OUT/selection.json --seeds 0 \
  --final-fold 0 --n-boot 10000 --pairs "T:none,M:none,TM:T,TM:M" --ci-level 0.9875 --out $OUT/decision_dev.json

# 5. 대조군 (dev, 기술용): teacher shuffle (T, M, TM x det/map/both), branch drop (TM)
E="OMP_NUM_THREADS=1 nice -n 10 $PY tools/refiner/eval_refiner.py all --split dev --token-subset $SP/dev_trainlogs.parquet --gpu 0 --workers 4 --theta 0.5 --sweep --budget-ep 0.5"
eval $E --run $RUNS/stageT4_T_fold0_seed0 --shuffle-teacher-seed 0 --eval-name eval_dev_shuffle
eval $E --run $RUNS/stageT4_M_fold0_seed0 --shuffle-teacher-seed 0 --eval-name eval_dev_shuffle
for W in det map both; do eval $E --run $RUNS/stageT4_TM_fold0_seed0 --shuffle-teacher-seed 0 --shuffle-which $W --eval-name eval_dev_shuffle_$W; done
for B in det map; do eval $E --run $RUNS/stageT4_TM_fold0_seed0 --drop-branch $B --eval-name eval_dev_drop_$B; done
CUDA_VISIBLE_DEVICES= $PY tools/refiner/stageT_decision.py compare --wtags stageT4 --selection $OUT/selection.json --seeds 0 \
  --final-fold 0 --n-boot 10000 --ci-level 0.9875 --out $OUT/controls_dev.json --pairs \
  "T:T@eval_dev_shuffle,M:M@eval_dev_shuffle,TM:TM@eval_dev_shuffle_det,TM:TM@eval_dev_shuffle_map,TM:TM@eval_dev_shuffle_both,TM:TM@eval_dev_drop_det,TM:TM@eval_dev_drop_map,T@eval_dev_shuffle:none,M@eval_dev_shuffle:none"
#    (outcome 문자열은 대조군에서는 판정이 아니다. true − 대조 차이와 CI만 읽는다.)

# 6. navtest (한 번, 고정 모델과 선택 θ, subset 없음) + 판정
for A in T M TM none; do OMP_NUM_THREADS=1 nice -n 10 $PY tools/refiner/eval_refiner.py all --run $RUNS/stageT4_${A}_fold0_seed0 \
  --split navtest --gpu 0 --workers 4 --theta 0.5 --sweep --budget-ep 0.5 >> $RUNS/logs/stageT4_${A}_fold0_seed0.eval.log 2>&1; done
CUDA_VISIBLE_DEVICES= $PY tools/refiner/stageT_decision.py compare --wtags stageT4 --selection $OUT/selection.json --seeds 0 \
  --final-fold 0 --n-boot 10000 --pairs "T:none,M:none,TM:T,TM:M" --ci-level 0.9875 --eval-name eval_navtest \
  --out $OUT/decision_navtest.json

# 7. EPDMS (navtest, 보고용; 전체 약 1.6 h @ 8 workers. subset을 쓰면 --n-tokens N --subset-seed 0을 붙인다)
nice -n 10 $PY tools/refiner/epdms_navtest.py score --runs $RUNS/stageT4_{T,M,TM,none}_fold0_seed0 --eval eval_navtest \
  --workers 8 --out /home/external-user/ssd/yongjae_refiner/epdms/stageT4_navtest
$PY tools/refiner/epdms_navtest.py report --scores /home/external-user/ssd/yongjae_refiner/epdms/stageT4_navtest \
  --selection $OUT/selection.json --pairs "T:none,M:none,TM:T,TM:M" --ci-level 0.9875 --out $OUT/epdms_navtest.json
```

- 5번 쌍은 A − B이다. `T:T@eval_dev_shuffle`의 P1이 양수면 진짜 BEV가 섞은 BEV보다 낫다는 뜻이다.
- 테스트: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python -m pytest -q tools/refiner/tests -x` → §8.8.

### 8.8 테스트

- `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python -m pytest -q tools/refiner/tests -x` → **330 passed, 19 warnings in 154.09s** (리뷰 반영 뒤 재실행; EPDMS report 회귀 테스트 1개 추가). M/TM smoke도 재실행해 §8.4 수치와 같았다.
  - 기존 테스트는 수정 없이 모두 통과했다.
  - 새로 `test_stageT4.py` 28개가 추가됐다.
- `runs/stageT_*`, `stageT2_*`, `stageT3_*`, `gpupilot*`, `pilot2V*`, packed / splits 아래에서는 이번 작업 시각 이후 바뀐 파일이 없다(`find -newermt` 확인). run-3 eval_navtest는 읽기만 했다(EPDMS bench).
