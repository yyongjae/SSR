# 공정성: 잠근 것, 밝힐 것, 믿으면 안 되는 숫자

**현재 상태:** `experiment.sh eval`과 평가 규칙 고정은 아직 커밋에 들어 있지 않습니다. 이 문서의 평가 설명은 그 작업이 끝난 뒤의 기준입니다.

논문이나 발표에 숫자를 쓰기 전에 읽는다. 준비 순서는 [SETUP.md](SETUP.md), 실험 실행은 [EXPERIMENTS.md](EXPERIMENTS.md)에 있다.

**한 줄 요약.** 테스트 데이터가 학습에 들어간 흔적은 없다. 그러나 지금까지의 navtest 숫자는 대부분 설정을 고정하기 전에 나온 것이고, 서로 BN, GPU 수, 채점기가 다르다. **이 저장소의 `experiment.sh eval`로 새로 낸 숫자만 표에 함께 쓴다.**

## 1. 잠근 것

평가에 쓰는 값은 옵션이 아니라 상수다. 한곳에서만 바꿀 수 있다: `navsim/agents/para_ssr/ck/protocol.py`.

| 항목 | 값 |
|---|---|
| 선택 식 | `rule=pdms`, `set=all`, `lat_mode=on`, `beta=1.0`. 제출하는 궤적은 refiner가 고른 것 |
| refiner off | v2 head를 제출. `selection: none`으로 기록 |
| epoch | 기본은 마지막 epoch. 다른 epoch은 `EPOCH=<n>`으로만 가능하고 `epoch_rule: manual`로 기록된다. 표에도 표시된다. navtest를 볼 때마다 `ledger.jsonl`에 한 줄이 쌓인다 |
| PDMS 채점 | 원본 NAVSIM v1.1 (`0811876`), 그 코드로 새로 만든 navtest cache, 추가 override 없음 |
| EPDMS 채점 | v2.2 `359c7f7`(수정 후)와 `8021f91`(수정 전) 둘 다 기록 |
| 추론 | batch 1, 토큰 정렬, model invariant 확인, hydra override 거부 |
| 코드 상태 | `train`, `eval`은 dirty tree를 거부. `ALLOW_DIRTY=1`이면 `git_dirty: true`와 diff를 run 폴더에 남긴다 |
| 표 | `table`은 GPU 수, 커밋, 프로토콜 버전이 섞이면 거부(`ALLOW_MIXED=1`), 차이가 CI보다 작으면 경고 |

`online2.validate`는 `infer_select`가 비어 있거나 `RELEASE_SELECT`와 같아야 통과시킨다. 옛 평가 도구(`eval_e2e2.py`, `eval_ck2.py`, `run_pdm_score*.py` 등)는 `SSR_RESEARCH_TOOLS=1` 없이는 실행되지 않는다.

**이 선택 식 자체가 사후에 정해졌다.** 2-7을 본다.

## 2. 논문에 밝힐 것

### 2-1. 백본 BatchNorm이 run마다 다르다

코드에는 `train()` 정의가 두 개 있고, 뒤의 것이 앞의 것을 덮는다. 그래서 v7 계열에서만 ResNet BN 통계가 학습 중에 움직였다. 가중치는 얼어 있고 통계만 움직인다. 체크포인트의 `image_encoder.bn1.num_batches_tracked`로 직접 확인했다.

| Run | 전역 step | 카운터 | BN | GPU |
|---|---:|---:|---|---:|
| `v2_v15_v7_main` ep15 | 9,975 | 39,900 (step × 4) | **움직임** | 8 |
| `v2_nokd_main` ep15 | 9,975 | 79,800 (step × 8) | **움직임** | 4 |
| v2 r34 기준선 | 19,950 | 0 | 얼어 있음 | 2 |
| `ck2e2e_bevkd` | 19,950 | 0 | 얼어 있음 | 2 |
| `fusion_v15_main` | 19,950 | 0 | 얼어 있음 | 4 |
| `fusion_v15_noref_main` | 19,950 | 0 | 얼어 있음 | 4 |
| `v6cD_r129` | 19,950 | 0 | 얼어 있음 | 4 |

- 이제 `backbone_bn_mode: moving | frozen` 설정으로 드러나 있고, 값이 run 이름에 들어간다(`bnm`, `bnf`). 기본값은 `moving`이다. v7 숫자를 만든 실제 동작이기 때문이다.
- **기본값을 무엇으로 둘지는 아직 결정되지 않았다.**
- 비교할 때 모든 행의 BN을 같게 둔다. v7과 옛 모델 비교에는 아무도 고르지 않은 BN 차이가 섞여 있다. 그 크기는 모른다.

### 2-2. GPU 수가 학습 내용을 바꾼다

전역 batch 128과 카운터는 맞춰 주지만 다음은 맞춰지지 않는다.

| 항목 | 2 GPU | 4 GPU | 8 GPU |
|---|---:|---:|---:|
| optimizer step당 BN 갱신 (`moving`) | 16 | 8 | 4 |
| optimizer step당 GridMask 추첨 | 16 | 8 | 4 |
| optimizer step당 EMA 갱신 (`lat_kd`, probe, ratio) | 16 | 8 | 4 |

- 모든 rank가 같은 RNG 시드를 쓴다. 그래서 GridMask는 rank 사이에 같다.
- 크기를 잰 적은 없다. 그래서 GPU 수를 이름(`g<N>`)에 넣고, 비교 표에서 섞지 못하게 했다. 기준은 8이다.

### 2-3. 학습 라벨은 v1.0 꼴이다

- 학습 라벨, warm-up 라벨, on-policy 라벨은 이 저장소의 NAVSIM fork와 그 fork로 만든 navtrain metric cache에서 나온다. 결정 D6으로 유지했다.
- 보고 숫자는 원본 v1.1로 채점한다.

| 차이 | fork | 원본 v1.1 |
|---|---|---|
| PDM-Closed 안의 `progress_distance_threshold` | 0.1 | 5.0 |
| DDC | 곱한다 (값이 거의 1) | 가중치 0 |
| 결과 | 같은 cache로 채점하면 NC, DAC, TTC, C, EP는 토큰마다 같고 DDC 처리만 다르다. cache가 다르면 EP가 run마다 326-377개 토큰에서 다르다 | 모든 행이 약 0.1 낮음 |

쓸 문구: "training labels use the v1.0-style scorer on a fork-built metric cache; reported numbers use pristine NAVSIM v1.1".

### 2-4. 학습에만 쓰는 특권 신호

표에 "테스트 입력"과 "학습 전용 신호"를 따로 적는다.

- **추론:** 카메라 f0, l0, r0, 프레임 2·3. teacher, KD 모듈, 라벨, 저장된 cache를 읽지 않는다.
- **학습:**
  - LiDAR + 카메라 teacher (BEVFusion, nuImages 사전학습)
  - 카메라 + 위성 영상 + 지도 teacher (ReSMap)
  - HD map seg 정답
  - PDM 세부 점수 라벨
  - ImageNet ResNet-34
- **파라미터:** 학습 41.57 M(그중 학습 전용 v15 probe 1.50 M), 추론 40.07 M(critic 3.56 M). 그 밖에 고정 refiner teacher 7.11 M과 BEVFusion, ReSMap이 있다. 기본 설정을 바꾸면 다시 잰다.

### 2-5. teacher와 입력에 관한 사실

| 입력 | 사실 |
|---|---|
| BEVFusion 64 m | navtrain 103,288 프레임 전체로 학습: **navtrain_val이 in-sample이다.** 학습 중 navtest로 5 epoch마다 모니터링, 마지막 epoch 사용(선택 없음). 체크포인트와 cache는 공개되지 않았다 |
| ReSMap front64 | 제3자 모델, cc-by-nc-4.0, 18 epoch 계획 중 9 epoch. 입력에 위성 영상. val 로그 포함, **학습 split은 확인하지 못했다.** 설정 경로 문자열에 `rideflux`가 있어 비공개 데이터 사용 여부를 소유자에게 물어야 한다 |
| refiner teacher | 데이터셋, 고정 앵커, 규칙 변형만 사용. 기존 teacher는 모니터링에 r34 후보를 썼다(가중치에는 안 들어감). 새 레시피는 `--r34-evals 0` |
| 앵커 | WoTE k-means, navtrain 전체(val 포함) |
| `pdm_score_256` | WoTE에서 온 값. 학습된 가중치는 쓰지 않음. EP 열에 알려진 batch artifact가 있다. val 토큰도 들어 있으나 학습은 train만 읽는다 |
| online KD 보정 | run 자신의 값. labeler 시점에 영향받는다 |
| 설계 상수 | 변형 종류, 가속 확장, `sur_groups`는 r34 후보 통계를 보고 정했다. 수치 입력은 아니다 |

- BEVFusion은 val에서 in-sample이므로, DET teacher 출력으로 val에서 결정하지 않는다.
- 값의 출처는 `tools/release/verify_assets.py`의 pin으로 고정했다.

### 2-6. navtest를 보고 정한 것

테스트 라벨이 학습에 들어간 것은 아니다. 하지만 아래는 navtest 결과를 보고 정했다. navtest 점수는 낙관적으로 치우칠 수 있다.

| 결정 | 본 것 | 영향 |
|---|---|---|
| BEV KD v15 상수 (λ, τ, row gain, 비율) | 이전 버전의 navtest 실패 수 | 있음. 재학습 없이 val로 다시 확인할 수 없음 |
| refiner teacher 키 특화 유지, 10 epoch | navtest keep/drop 규칙 | 작음 |
| Platt 보정 켜기 | navtest MAP DAC 실패 예측 8.6 % 대 실제 4.9 %. val에서는 차이 없음 | 작음 |
| `MERGE_EPS` 0.2 m | navtest 사람 궤적 12,146개 | 매우 작음 |
| 측방 보정 항상 켬, 점수 KD 0.5, comfort 제외, 공식 EP | navtest 일부 참조 (대부분 val로도 뒷받침) | 작음 |
| base 설계 (xy에서 heading 계산, seg head) | navtest 측정 | 기준선과 공통 |
| 진행 판단 기준 S1-S5, teacher 점검, SDF 방식 검증 | navtest | 개발 판단 |
| 선택 식 P, ep15 | 2-7 | 직접 |

선택 전 설계 문서(`docs/bev_selector_v*.md`)에 navtest 기반 반복 개발이 기록돼 있다. 공개 저장소를 낼 때 이 파일들과 navtest 도구를 뺄지 정해야 한다.

### 2-7. 선택 식은 사후 채택이다

- **P 식**은 96개 후보 중 E[NC]·P(DAC)·(5EP+5TTC+2C)/12가 가장 큰 후보를 고른다. 모방 점수가 없다.
- navtest에서 여러 번 좋게 나온 뒤에 채택됐다. 학습 전에 적은 사전 규칙은 이 행을 대표 행에서 제외했다.
- P의 이득은 대부분 EP다. 모든 epoch에서 기존 식보다 NC, TTC가 낮다.
- **P는 역주행을 보지 못한다.** 학습 라벨의 DDC는 거의 0이 되지 않고, critic에 DDC 출력이 없다. v1.1은 DDC 가중치가 0이라 PDMS에 드러나지 않는다. v1.1 DDC 열로 보면 P 95.73, 기존 식 98.01, v2 98.18이다.
- EPDMS에서 불리할 수 있다. **P의 공식 EPDMS는 아직 잰 적이 없다.** `experiment.sh eval`이 처음으로 잰다.

논문에는 고정 규칙의 결과, 마지막 epoch, 사전 규칙 행(기존 식 ep30, 92.02)을 함께 싣고, P가 사후 선택임을 적는다.

### 2-8. 분산과 시드

- 지금까지 seed 1개로 한 번 돌렸다. **같은 설정을 다시 돌렸을 때의 차이를 잰 적이 없다.**
- `grid_sample` 역전파, AMP teacher, labeler 시점이 비결정적이다.
- navtest 구간 136개는 주행 44개에서 나왔다. 구간 단위 CI는 너무 좁다. 주행 단위로 다시 뽑으면 넓어진다.
- 그래서 **0.6보다 작은 차이는 주장하지 않는다.** 이전 분석의 최소 검출 효과는 0.40이었다.

## 3. 믿으면 안 되는 숫자

아래 숫자를 새 표에 옮기지 않는다. 이 저장소로 다시 낸 숫자로 바꾼다.

| 숫자 | 문제 |
|---|---|
| **93.37, 93.35** (ep15, P 식) | fork cache. ep15와 P 모두 navtest를 본 뒤 고름. val에서 ep15와 ep20은 0.02 차이인데 navtest에서는 0.55 차이다 |
| **93.28** (같은 run, 원본 v1.1) | 채점은 맞다. 위의 사후 선택, r34로 맞춘 KD 보정, 움직이는 BN, 정렬 안 된 데이터 순서를 그대로 갖고 있다 |
| **EPDMS 90.58 / 88.52**, ep30 90.75 / 88.18 | 공식 러너가 아닌 빠른 채점기다. HC가 달라 공식 값과 다르다. 공식 값은 기존 식 ep15 **90.386(수정 후), 86.863(수정 전)**, ep30 90.562(수정 후)이다. "P의 EPDMS 88.52"를 공식처럼 쓰면 안 된다 |
| **navhard 43.37 / 41.21** | 같은 비공식 채점기다. 이 저장소의 파이프라인으로 잰 적이 없다 |
| **93.75-93.88, EPDMS 91.4, navhard 48.3-48.6** | 선택 식을 navtest, navhard에서 **직접 최적화**한 결과다. 쓰면 안 된다. 다른 모델의 navtest에 맞춘 식을 그대로 옮긴 행도 같다 |
| **`v2_v15_v7_main`의 모든 숫자** | r34로 맞춘 KD 보정(file 모드), 움직이는 BN, 파일 시스템 순서. v2_final과 다른 실험이다 |
| **v7 대 기준선, bevkd, fusion_v15 차이** | BN이 다르다(2-1). GPU 수도 다르다. "P−v2 r34"에도 8가지 이상의 차이가 섞여 있다 |
| **v2 r34 88.15** | fork cache, 2 GPU, 다른 레시피. v1.1로 88.03이다. 같은 레시피의 64 m 기준선(`base`)이 아직 없다 |
| **`v2_nokd_main`** | 4 GPU, 합치기 전 코드, epoch 15에서 죽음. KD ablation 행이 아니다 |
| **E1 91.85 대 v7 93.37** | 선택 식과 epoch 규칙이 다르다. 둘의 차이 중 약 +1.1은 선택 식 효과다. 같은 규칙으로 맞추면 v7은 이전 32 m 모델과 비슷하다 |
| **이전 문서의 "ResNet BN은 고정"** | v7 계열에서 틀렸다(2-1) |
| **`eval_e2e2.py` navtest 값** | fork cache 내부 값이다. 내부 비교에만 쓴다 |
| **"카메라만" 비교표에 섞인 C+L 방법** | 입력이 다르다. 열을 나눈다 |

## 4. 아직 답이 없는 질문

| 질문 | 누가 | 막는 것 |
|---|---|---|
| BN 기본값 (`moving` 또는 `frozen`) | 프로젝트 결정 | 최종 학습 전 |
| ReSMap 학습 split, `rideflux` 경로, 비상업 라이선스가 증류된 가중치에 미치는 영향 | ReSMap 소유자 | 공개 |
| 위성 영상 라이선스 문구 | ReSMap 소유자, Mapbox 약관 | 공개 |
| BEVFusion 체크포인트와 cache 공개 (545 GB) | 프로젝트 결정 | 외부 협업자의 `full` 프로필 |
| 같은 설정 재실행 분산 | GPU 20시간 | 작은 차이 해석 |

라이선스 목록은 [THIRD_PARTY.md](../THIRD_PARTY.md)에 있다.
