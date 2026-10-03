# 40. 미래 데이터 방법론: 정리와 새 서버 세팅 계획

작성 2026-10-03. 읽기 전용 조사로 만든 문서다. 이 문서를 쓰는 동안 실행한 잡, GPU 사용, 다운로드는 없다.

검증 2026-10-03: 별도 검증 단계에서 수치와 절차를 원자료(로그 pkl, 캐시, 코드, 33–39)와 다시 대조했다. 검증에서 새로 계산한 값에는 "검증 단계"라고 적었다. 검증도 읽기 전용 CPU 계산만 했고 잡·GPU·다운로드는 없다.

**표기**
- [실측]: 파일이나 로그에서 직접 확인한 값, 또는 기존 파일을 CPU로 전수 계산한 값
- [재구성]: 2026-09-27 세션 기록에서 되살린 명령이나 값
- [추정]: 측정값에서 외삽한 값
- [추론]: 근거는 있지만 검증하지 않은 판단
- [제안]: 아직 실행하지 않은 절차
- [새 코드]: 지금 저장소에 없어서 새로 써야 하는 코드

**읽는 순서**
- 처음 보는 세션: §0 → §1 → §3 → §7
- 세팅을 할 때: §2 → §4 → §6

**전제:** 새 서버에는 NAVSIM 다운로드, 현재 프레임 teacher 캐시(`cache_{train,val}_50x100`, ReSMap root + navtest)가 있다. 나머지 캐시는 `report/39_new_server_setup_runbook.md`(이하 39)대로 다시 만든다. 경로는 옛 서버와 같은 절대경로라고 가정한다(39 §3.2).

---

## 0. 요약

**무엇인가**
- 학습 때만 **미래 정보**(t+0.5 … t+4 s)를 써서 target을 만든다. 추론 때 학생은 지금처럼 현재·과거 카메라만 본다(LUPI).
- 대표 형태는 "시간 이동 teacher"다 [33 §8-1]. BEVFusion teacher는 구조와 가중치를 그대로 두고, 미래 프레임에 돌린다. 출력은 t0 좌표로 옮겨 KD target으로 쓴다.

**왜 필요한가**
- 학생 NC 실패 279장면 중 119장면은 **움직이는 물체**와 부딪힌 경우다 [33 §3].
  - 현재 프레임 teacher 박스에 등속 가정을 더한 필터로는 17/119만 피한다.
  - 정답의 실제 미래를 쓰면 115/119를 피한다. 33 §3은 이를 "상한"이라 했으나 34 §0에서 "정답 미래를 넣은 이 필터의 참고값이지 엄밀한 상한이 아니다"로 고쳤다(다른 차는 ego에 반응하지 않는 로그 기록이다).
- 충돌 시각 중앙값은 3.5 s다. 그래서 **4 s horizon은 줄일 수 없다** [34 §1-1].

**지금 상태**
- 34의 O(t,s) 계획은 35에서 대체됐다. 이유는 네 가지다: GT로 대체할 수 있음, KD 위치가 애매함, finetuning 구조임, 데이터가 16.5%뿐임.
- 현재 주 라인(37/38)에서 미래 정보는 **GT 벌점**(GT 미래 물체 0–5 s)으로만 들어간다. teacher 교정은 t=0 판단이다.
- 미래 프레임 teacher 캐싱은 2026-09-27에 **중단**됐다. 원인은 공유 GPU에서 프로세스당 18.5 GB를 쓴 것이다 [35 §9]. 캐시를 만드는 파이프라인 자체는 24/24 재현으로 검증을 마쳤다(§6-2).
- 학습 로더 `TeacherCache`는 경로에 `_future`가 들어 있으면 일부러 거부한다(`data.py:146-169`).

**무엇이 필요한가 (4 s 기준, [실측])**

| 대상 | BEVFusion을 새로 돌릴 미래 프레임 | 센서를 새로 받아야 하는 프레임 |
|---|---:|---:|
| navtest | 12,900 | **0** (전부 디스크에 있음) |
| stage-T(train+dev trainlogs) | 62,629 | 52,528 |
| E2E 학습 집합(85,109 token) | 93,606 | 78,543 |
| navtrain 전체 | 112,152 | 93,670 |

**비용을 결정하는 것**
1. **센서 다운로드가 가장 크다.** 빠진 미래 프레임은 navtrain 패키지(current+history)에 구조적으로 없다. 유일한 출처는 OpenScene v1.1 trainval 센서 전체(camera tgz 200개 + lidar tgz 200개)다.
   - 크기는 약 1.6–2.25 TB [추정]이다. 1.6 TB는 35 §0-2의 "압축 기준" 값이고(출처 미기재), 2.25 TB는 trainval 1,310 log 전체 723,019 frame × 3.116 MB/frame(§2-4)로 계산한 추출 후 크기다. archive와 log의 대응표가 없어서 필요한 log만 받을 수 있는지 모른다.
   - 시간은 관측 실효 속도 10 MB/s 기준 약 62 h, 50 MB/s 가정 시 약 12.5 h다 [추정]. 10 MB/s는 옛 서버 navtrain 다운로드의 `.done_navtrain_*` 마커 64개 시각(첫 마커부터 마지막까지 12.3 h)과 `trainval_sensor_blobs` 446 GB에서 나온 근사값이다.
2. **디스크:** 스트리밍 추출로 필요한 파일만 남기면 센서는 106–190 GB다 [추정]. teacher npz는 `--drop-bev`면 약 10 GB, BEV까지 저장하면 약 300 GB다 [추정].
3. **GPU 시간은 작다:** navtrain 4 s 전체가 전용 GPU 1장으로 약 2.7 h다 [추정, 11.6 frame/s]. 11.6은 전용 GPU 실측이 아니다. 공유 GPU 두 프로세스의 실측 5.7–5.8 task/s(`logs/cache_gpu{0,1}.log`)를 합한 값과 같다. 다만 GPU를 공유하면 18.5 GB 메모리 문제가 다시 생긴다.
4. **ReSMap 미래**(F5 계열에만 필요): 공동연구자 환경에 의존한다. torch 1.12 env, checkpoint, 위성 타일, infos가 kyungmin 서버에 있고, sm_120에서는 돌지 않는다.

**가장 싼 길 두 개 [추론]**
- 다운로드 없이 할 수 있는 것은 두 가지다.
  - **GT 미래 teacher**(R_GT-fut): 센서가 필요 없다. GT objects는 85,109 token에 이미 있다(39 §5.5).
  - **navtest 진단**(34 단계 A): BEVFusion 12,900프레임, GPU 약 20–40분.
- teacher-미래를 학습 신호로 쓰려면 다운로드를 피할 수 없다.

---

## 1. 방법론 정리

### 1-1. 문제와 숫자 [33 §2–§7, 34 §1-1]

| t=0 GT 속도 기준 충돌 대상 | 장면 | 학생 박스+학생 미래 | 정답 박스+등속 | teacher 박스+등속 | 정답의 실제 미래 |
|---|---:|---:|---:|---:|---:|
| 정지 | 149 | 91 | 115 / 129 | 106 | 140 |
| **움직임** | **119** | **44** | **14 / 24** | **17** | **115** |
| t=0에 없던 물체 | 11 | 1 | 2 / 5 | 1 | 11 |

- 움직이는 대상의 114/119는 차량이다. t=0 속도 중앙값은 3.0 m/s이고, 115/119가 앞범퍼 충돌이다.
- 충돌 시각 중앙값은 3.5 s다. 움직임 119장면 중 92장면이 3 s 이후에 충돌하고, 2 s 이내 충돌은 1장면이다.
- 학생 미래 예측이 약한 구조적 이유 [33 §4–5]:
  - 입력이 2프레임뿐이다.
  - 미래 예측에 지도 attention이 없다.
  - 모드 점수가 거의 균등하다(1순위 확률 중앙값 0.215, 균등값은 0.167).
  - 박스 속도의 옆 성분이 붕괴했다(표준편차 0.06, GT는 1.37).
  - planner는 6개 모드 latent의 평균만 읽는다.
- 현재 프레임 teacher의 속도는 정확하다(기울기 0.996, R² 0.985). 그래도 등속 가정으로는 17/119만 피한다. **t=0 정보만으로는 이 문제를 풀 수 없다** [33 §7].

### 1-2. 변형들

| 이름 | 미래 정보원 | 무엇을 증류하나 | 어디에 거나 | 출처 | 상태 |
|---|---|---|---|---|---|
| 시간 이동 teacher | BEVFusion을 t+k 프레임에 돌림 → t0 좌표 | 아래 F1–F5와 O_T+의 재료 | – | 33 §8-1 | 제안 |
| F1 경로 충돌 프로파일 | teacher 미래 | 경로 위 시각별 "앞이 비어 있는 정도" | h_final 보조 출력 | 33 §8-4 | 제안 |
| F2 미래를 반영한 교정 행동 | teacher 미래 | 필터/교정기가 고친 궤적 | 궤적 출력 | 33 §8-4 | 제안. teacher ≈ GT라 teacher만의 가치는 낮음 |
| F3 미래 BEV feature | teacher t+k BEV를 t0로 warp | 학생 BEV → 미래 BEV 예측 | 공유 BEV 또는 경로 통로 | 33 §8-4 | 제안. GT에 없는 dense 정보 |
| F4 사후 의도 라벨 | teacher 미래 | 급감속·출발·끼어들기 | 검출 query, h_final | 33 §8-4 | 제안, 가치 작음 |
| F5 사후 지도 | ReSMap을 미래 프레임에 돌림 | 32 m 너머·가려진 차선 | 지도 head, h_final | 33 §8-4 | 34 이후 빠짐 |
| O(t,s) 점유 격자 | O_T+ (teacher 미래 박스) | 경로 위 시공간 점유 | 예측기 Ô(B) → refiner(C) | 34 §2–§7 | 35에서 대체됨 |
| R_GT-fut | GT 미래 물체 raster | 미래를 아는 교정기의 교정량 | student 교정기(KD) | 36 §1, fact_features | 선택 arm, 실행 안 함 |
| GT 벌점 (현재) | GT 미래 물체 0–5 s, 0.1 s 간격, 360° | 벌점(loss) | student 교정기와 BEV | 37 §5 | **진행 중** |

**O(t,s) 정의** [34 §2]
- O(t,s) = 1[시각 t에 기준 경로 위 거리 s에 있는 ego 차체(+0.3 m 여유)가 어떤 물체 차체와 겹친다].
- 학습 target 해상도는 0.5 s × 0.5 m, 0–4 s(8시점)이고 장면당 약 1 KB다.
- 정보원만 바꾼 같은 형식의 격자가 다섯 가지 있다: O_GT, **O_T+**, O_T,cv, O_S, Ô.
- teacher 출력은 그 시각의 **로그 ego 위치** 기준 앞 32 m만 덮는다. 덮지 못하는 칸은 "모름"으로 두고 loss에서 뺀다.

### 1-3. Horizon과 간격 [34 §1-1]
- NAVSIM 로그는 2 Hz다. token당 미래 프레임은 8장(t+0.5 … t+4.0 s)이다.
- **4 s가 필수다.** 2 s 파일럿은 철회됐다.
- 프레임 간격을 넓혀도 비용이 거의 줄지 않는다. navtest 기준 0.5 s 간격이 12,900프레임, 1 s가 12,062, 2 s가 10,517이다. 비용은 **장면 샘플링**으로 줄인다.
- 시각은 index 기준(k × 0.5 s)이다. (token, k) 쌍 중 0.35%는 간격이 0.5 s가 아니다 [실측; 검증 단계 재계산: 허용 오차 ±50 ms에서 navtrain 3,087/826,304 = 0.37%, navtest 185/97,168 = 0.19%, 합 0.35%]. `gt_future.py`도 index 기준을 쓰고 `dt_max`로 표시한다.

### 1-4. 지금까지의 결정과 그 이유
1. 4 s 필수, 2 s 파일럿 철회 [34 §1-1].
2. 간격을 줄이는 서브샘플링 대신 장면 샘플링으로 비용을 줄인다 [34 §1-1].
3. teacher의 미래 사각지대(전방 32 m)는 "모름" 칸으로 처리한다 [34 §2-3].
4. 34를 35로 대체했다 [35 §0]. 사용자 요구는 다섯이다(번호는 35 §0-1 그대로, 이 문서의 "요구 n"은 이 번호를 가리킨다).
   1. teacher가 **GT 대비** 필요함을 보일 것
   2. KD 위치가 분명할 것
   3. **E2E를 학습하는 동안** KD할 것(학습된 E2E finetune 금지)
   4. 전 단계(34의 A)가 학습된 E2E 모델을 전제로 하지 않을 것
   5. 논문이 될 수 있는 방향이고 선례가 있을 것
5. 캐싱 중단: GPU 메모리 위험 때문이다. 계속할지는 사용자 결정으로 남아 있다 [35 §9].
6. 현재 라인: 미래는 GT 벌점이 맡는다. teacher 교정은 t=0 판단이라 "따라 배우기 현실적"이다 [37 §12-4].

**이 결정들에서 나오는 제약 [추론]**
- 미래 방법을 다시 열려면 요구 3·4를 만족해야 한다. 즉 E2E 학습 중 KD여야 하고, 학습된 E2E 궤적을 전제로 하는 사전 단계가 없어야 한다.
- 34의 A/B/C 단계 구조(학습된 E2E 위 finetuning)는 그대로 쓸 수 없다. §5는 현재 E2 틀(교정 KD)에 붙이는 형태로만 제시한다.

---

## 2. 필요한 데이터 매트릭스

**열 설명**
- 필요: 서로 다른 미래 프레임 수(+0.5 s부터 그 horizon까지의 합집합)
- 있음: 해당 teacher 캐시에 이미 있음. 다른 token의 현재 프레임이라서 들어 있는 경우다.
- 실행: 새로 추론할 프레임
- DL: 그중 센서를 새로 받아야 하는 프레임

출처: 이번 조사의 CPU 전수 계산 [실측]. 기존 `counts.json`(112,152 / 12,900), `train_disk_availability.json`(18,482)과 일치한다. 새 서버는 같은 NAVSIM 패키지와 같은 현재 프레임 캐시를 쓴다고 가정했다. 옛 서버의 부분 캐시 `cache_val_50x100_future`(1,422개)는 새 서버에 없다고 보았다.

### 2-1. BEVFusion (검출 teacher; O_T+, F1–F4, R_T+의 재료)

| 집합 (token / log) | h | 필요 | 있음 | **실행** | 그중 디스크에 있음 | **DL** | 4 s 전부 BEV 캐시에 있는 token |
|---|---|---:|---:|---:|---:|---:|---:|
| navtest (12,146 / 136) | 2 s | 17,872 | 10,525 | 7,347 | 7,347 | 0 | 46.9% |
| | **4 s** | 23,771 | 10,871 | **12,900** | 12,900 | **0** | 22.6% |
| stage-T `train_trainlogs`+`dev_trainlogs` (19,732+6,634 / 합집합 972 log) | 2 s | 75,527 (56,608+18,919) | 47,106 | 28,421 | 5,181 | 23,240 | 44.3% |
| | **4 s** | 118,396 (88,642+29,754) | 55,767 | **62,629** | 10,101 | **52,528** | 19.1% |
| E2E `e2e_train_trainlogs` (85,109 / 978) | 2 s | 126,032 | 73,385 | 52,647 | 9,431 | 43,216 | 44.5% (2 s) |
| | **4 s** | 169,248 | 75,642 | **93,606** | 15,063 | **78,543** | 19.0% |
| navtrain 전체 (103,288 / 1,192) | 2 s | 152,495 | 89,256 | 63,239 | 11,570 | 51,669 | 45.1% (2 s) |
| | **4 s** | 204,164 | 92,012 | **112,152** | 18,482 | **93,670** | 19.7% |

- 검증 단계에서 같은 정의(디스크 = F0/L0/R0 `.jpg`와 lidar `.pcd`가 모두 있음, 있음 = 해당 현재 프레임 캐시에 npz가 있음)로 모든 칸을 다시 계산해 일치를 확인했다. stage-T 행의 "있음"과 navtest·stage-T의 마지막 열은 검증 단계에서 채운 값이다 [실측]. E2E 2 s 행의 "필요" 126,032는 ReSMap root 프레임 수 126,032와 같은 숫자지만 서로 다른 집합이다. 혼동하지 않는다.
- **sweep 조건:** BEVFusion은 lidar sweep 2개(f−1, f−2)를 쓴다.
  - 받을 집합 전체를 받으면, sweep 때문에 추가로 받을 프레임은 0개다 [실측, stage-T / E2E / navtrain 4 s 각각 확인]. 장면 샘플링(§3 E)처럼 집합을 임의로 줄이면 다시 확인해야 한다.
  - "그중 디스크에 있음" 열은 sweep을 보지 않은 값이다. sweep `.pcd`가 없는 프레임은 navtrain 4 s 18,482개 중 **8,075개**, E2E 4 s 15,063개 중 6,610개, stage-T 4 s 10,101개 중 4,164개, navtest 0개다 [실측]. 그래서 다운로드 없이 바로 돌릴 수 있는 navtrain 미래 프레임은 약 10,407개다.

### 2-2. ReSMap (지도 teacher; F5나 R_M+에만 필요)

| 집합 | h | 필요 | ReSMap 캐시에 있음 | 실행 | 비고 |
|---|---|---:|---:|---:|---|
| navtest | 4 s | 23,771 | 10,871 (allow-list token이라서) | 12,900 | 센서는 있음. 71,460프레임을 scene 순서로 다시 돌리고, 저장 목록만 넓힘 [제안] |
| E2E | 2 s / 4 s | 126,032 / 169,248 | 82,816 / 90,705 | 43,216 / **78,543** | ReSMap root = "train_logs에서 센서가 디스크에 있는 프레임 전부" [실측]. 그래서 실행 = DL |
| stage-T(train+dev) | 4 s | 118,396 | 65,868 | 52,528 [실측, 검증 단계] | 실행 = DL(52,528)이 실제로 성립 |
| navtrain 전체 | 4 s | 204,164 | 90,705 | 113,459 | val_logs token 18,179개는 현재 프레임도 없음 |

- **ReSMap은 temporal memory 모델이다.** 미래 프레임을 끼워 넣으면 log 안의 연속 구간이 바뀐다. 구간 수는 정의에 따라 다르다. 작성 단계 값 14,032 → 9,339는 검증 단계에서 재현하지 못했다. 검증 단계 재계산: "log 순서상 연속 run"으로 세면 11,724 → 7,875(E2E 4 s 추가), "run + scene_token 경계"로 세면 14,607 → 12,827이다 [실측]. 실제 memory 리셋 기준은 kyungmin map infos의 `local_idx == 0`이라 어느 쪽과도 같다는 보장이 없다. 어느 정의든 구간이 합쳐지므로 결론은 같다. **이미 캐시된 프레임의 feature도 달라진다.** 기존 shard에 덧붙이면 안 되고 새 폴더에 log 전체를 다시 돌려야 한다.

### 2-3. GT 미래 (센서 불필요)

| 집합 | 필요한 것 | 새 서버 상태 |
|---|---|---|
| E2E 85,109 (stage-T trainlogs 포함) | `objects/<sub>/<token>.npz` (`gt_future_v1`, 0–5 s, 11 keyframe, 360°) | 39 §5.5로 만든다(CPU 약 10–15분). 이미 GT 벌점에 쓰는 파일이다 |
| navtest 12,146 | 같은 형식 | 지금 학습 파이프라인은 만들지 않는다. raw log만 있으면 `build_future_objects.py`로 만들 수 있다 [추론, 미확인] |

### 2-4. 프레임당 크기 [실측]과 합계 [추정]
- **센서**
  - 프레임 전체(8캠 + lidar): 3.116 MB
  - BEVFusion 최소 입력(F0/L0/R0 + lidar): 2.023 MB
  - ReSMap 입력(전방 3캠): 0.626 MB
- **BEVFusion npz**
  - `bev_feature` 포함: 2,648,014 B
  - `--drop-bev`: 약 0.088 MB [추정, npz 멤버 크기에서 계산]
- **ReSMap feature:** 약 2.55–2.73 MB/frame

| 항목 (decimal GB) | navtest 4 s | stage-T 4 s | E2E 4 s | navtrain 4 s | navtrain 2 s |
|---|---:|---:|---:|---:|---:|
| 받을 센서, F0/L0/R0 + lidar만 | 0 | 106 | 159 | 190 | 105 |
| 받을 센서, 9개 파일 전부 | 0 | 164 | 245 | 292 | 161 |
| BEVFusion npz, bev 포함 | 34 | 166 | 248 | 297 | 167 |
| BEVFusion npz, `--drop-bev` | 1.1 | 5.5 | 8.2 | 9.9 | 5.6 |
| ReSMap (필요할 때만, 2.55–2.73 MB × 실행 프레임) | 약 33–35 | 약 134–143 | 약 200–214 | 약 289–310 | 약 178–190 |

---

## 3. 선택지

모두 [추정]이다. 다운로드 속도는 옛 서버에서 navtrain을 받을 때의 실효값(약 10 MB/s, 단일 스트림, 추출 포함)을 기준으로 했다. 새 서버의 대역폭과 HF 속도 제한은 모른다.

| | 내용 | 전송량 | 새로 쓰는 디스크 | 시간 | GPU | 주요 위험 | 과학적 의미 |
|---|---|---|---|---|---|---|---|
| **A. OpenScene 전체 다운로드 + 전체 추출** | `download_trainval.sh` 그대로 | 약 1.6–2.25 TB | 약 2.25 TB(센서) + npz 10–300 GB | 10 MB/s 기준 약 62 h. 50 MB/s면 약 12.5 h | navtrain 4 s 2.7 GPU-h (전용) | 디스크 부족. 기존 `trainval_sensor_blobs`와 섞여 덮어쓸 위험 | 무엇이든 할 수 있다(F1–F5, 모든 split) |
| **B. 전부 받되 필요한 파일만 스트리밍 추출** | `wget -O- … \| tar -xz -T needed.txt` | A와 같음(사실상 400개 archive 전부) | E2E 4 s 159 GB, stage-T만 106 GB (+ npz) | A와 같음 | A와 같음 | 파일 목록 생성기가 필요함 [새 코드]. tar가 "Not found in archive"로 exit 2를 내므로 무시 처리가 필요함 | A와 같음. 다만 나중에 범위를 넓히면 다시 받아야 함 |
| **B'. 필요한 log가 든 archive만 받기** | archive와 log의 대응을 먼저 확인 | E2E 978 log면 최대 약 1.79 TB, navtrain이면 약 2.09 TB | B와 같음 | 최대 약 20% 절약 | 같음 | 대응이 log 단위가 아니면 이득이 0. 확인에 archive 1개(약 5–6 GB [추정])가 필요함 | B와 같음 |
| **C1. navtest만 (34 단계 A식 진단)** | 다운로드 없음 | 0 | npz 1.1 GB(`--drop-bev`) / 34 GB | 준비 반나절 [추정] | 12,900프레임, 전용 약 19분, 공유 약 37분 | 학습이 없다. 학습된 E2E 궤적(E0 navtest 궤적)을 전제로 하므로 사용자 요구 4와 충돌 [35 §0-1] | H1: "미래 프레임 teacher 출력이 교정 정보로 쓸 만한가"만 답한다. teacher-미래 vs GT-미래 vs 등속 비교 |
| **C2. navtrain 중 이미 디스크에 있는 것만** | sweep까지 있는 약 10,407프레임 | 0 | 약 1 GB / 28 GB | – | 약 15분 | 4 s 전부가 있는 token은 BEV∪디스크 기준 30.1%가 상한이고, sweep 누락으로 더 줄어든다 [미계산]. 위치 편향 가능성(history가 겹치는 위치에 몰림) [추론] | 파일럿 정도만 된다. KD를 그 token에만 걸 수는 있다(§5-4). 그러나 편향된 부분집합이다 |
| **D. GT 미래만 (R_GT-fut)** | GT objects로 미래 raster를 만들어 stage-T 교정기를 학습한 뒤 E2에 KD | 0 | raster 약 7 GB(16ch uint8 × 85,109) [추정] | raster 빌더 + stage T 경로 B(39 §6.3, CPU 약 5–6 h [추정]) | stage-T 교정기 1개 약 1.5 h(39 §6.3 ⑧: run-4 T 89분 [실측]) + E2 결합 학습 약 20–21 h × 4 GPU(37 §11 표) | aux teacher가 아니다(GT 특권). 35의 "teacher 필요성" 요구와 충돌한다. GT 벌점과 정보원이 같아서 E2−E1과 구분하기 어렵다 [37 §12-4] | "미래를 아는 교정기의 KD가 E2E로 전달되는가"의 천장 |
| **E. 장면 샘플링 + B** | 34 §1-1: navtrain 2–3만 장면 | B와 같음(센서가 archive에 섞여 있으므로) | stage-T 집합이면 106 GB(§2-4). 2–3만 장면을 stage-T처럼 흩어 뽑으면 약 80–120 GB [추정: 프레임 × DL 비율 0.84 × 2.023 MB] | B와 같음 | stage-T 집합이면 62,629프레임, 전용 약 1.5 GPU-h. 2–3만 장면이면 약 4.8–7.1만 프레임 [추정] | 전송량은 줄지 않는다. 34 §1-1의 "2.5–3.5만 프레임"은 navtrain 전체 값(11.6만)을 장면 수에 비례해 줄인 것이라 과소다. 표본에서는 이웃 token끼리 미래 프레임을 덜 공유한다(stage-T: token당 2.38 프레임, navtrain 전체: 1.09). stage-T 집합(26,366 token)을 그대로 쓰면 자연스러운 표본이 된다 | stage-T 교정기 학습에는 충분하다. E2 KD는 85,109 중 일부에만 걸린다 |

**정리 [추론]**
- 다운로드가 필요한 선택지(A/B/B'/E)는 전송량이 거의 같다. 다른 것은 디스크 사용량뿐이다. 다운로드를 하기로 하면 **B를 E2E 4 s 집합 기준으로 한 번에 하는 것**이 범위를 나중에 넓힐 필요가 적다.
- 다운로드 없이 할 수 있는 것은 C1(진단), C2(편향된 파일럿), D(GT 특권)뿐이다.
- C1과 D는 서로 독립이라 병행할 수 있다.
- teacher-미래가 GT-미래보다 나은지(요구 1)를 직접 보려면 **D와 C1 또는 A/B를 같은 raster 형식으로 맞춰 비교**해야 한다(§5-2).

---

## 4. 새 서버 세팅 절차

**실행 순서 (검증 단계에서 고침):** 0 → 1 → 3 → 4 → (2, 다운로드할 때만) → 5 → 6 → 7 → 8 → (9) → 10.
- 단계 2-2의 파일 목록 생성기는 단계 3이 만드는 `future_*.yaml`(또는 걸러 낸 yaml)이 필요하다. 그래서 단계 3이 단계 2보다 먼저다.
- 단계 4(재현 검증)는 bevfusion `data/navsim` symlink와 `infos/validate_test.pkl`이 필요하다. 둘 다 단계 4 안에서 만든다.
- 다운로드하지 않는 길(C1 navtest, P1)은 단계 2를 건너뛴다. P1만 할 때는 단계 1–7도 필요 없다(단계 8의 GT source와 단계 10만). 대신 stage T 학습 데이터(39 §6.3 경로 B: drafts, labels, pack, train/dev objects)가 있어야 한다.

변수는 39 §0.5를 따른다. 추가 변수는 아래와 같다.
```bash
REPO=/home/external-user/yongjae/SSR; D=/home/external-user/ssd/yongjae_refiner
PY=/home/external-user/miniconda3/envs/ssr/bin/python
DL=/home/external-user/navsim/download; TC=/home/external-user/datasets/teacher_cache
BF=/home/external-user/yongjae/bevfusion; W=$REPO/tools/future_teacher_cache
BPY=/home/external-user/miniconda3/envs/bevfusion/bin/python
```

### 단계 0. 옛 서버에서 가져올 것 (git으로 따라오지 않음)

| 항목 | 이유 |
|---|---|
| `$REPO/tools/future_teacher_cache/` 중 `make_token_lists.py`, `cache_teacher_future.py`, `run_cache.sh`, `*.yaml`, `*.json` | **untracked**다. `infos/`는 새로 만든다. `cfg_cf3e02a/`(100×100 config)와 `old_100x100/`은 가져오지 않는다 |
| `$BF` working tree 전체(`runs`, `data` 제외, `.git`·`build/` 포함 약 918 MB) 또는 `git diff` + untracked 2개(`tools/cache_teacher_bev.py`, `chain_stage2.sh`) | 캐시를 만든 코드는 커밋 `cf3e02a`가 아니라 그 위의 **dirty tree**(수정 10개 파일: config 4개, `navsim_dataset.py`, `transforms_3d.py`, `fov_utils.py`, `hungarian_assigner.py`, `setup_fix.md`, `docs/NAVSIM.md`)다. clone만 하면 config가 100×100(`voxel_size [0.04,0.08,0.2]`)이 된다. working tree를 통째로 옮겨도 `build/`와 `.so`는 단계 1에서 다시 빌드한다. `.git`이 없으면 manifest의 `git_commit`이 null이 된다 |
| `$BF/data/infos/navsim_infos_val_navtest.pkl` (156,409,747 B) | 단계 4 infos 재현의 비교 기준. 위 "data 제외"에 걸리므로 따로 옮긴다 |
| `$BF/runs/navsim-fusion-50x100/epoch_20.pth` (481,739,184 B) + `configs.yaml` | teacher 가중치와 학습 당시의 resolved config |
| (선택) `$TC/bevfusion/cache_val_50x100_future/` (1,422 npz, 3.6 GB) | navtest 실행을 11,478개로 줄인다. manifest가 없으므로 다시 돌려도 손해는 작다 |

```bash
# 새 서버에서 확인
sha256sum $BF/runs/navsim-fusion-50x100/epoch_20.pth
#  기대: 6ad75eca1a7df5257bc21e77931189dff83b8a360f5923e6267745428fc4a904
head -c 1048576 $BF/runs/navsim-fusion-50x100/epoch_20.pth | sha256sum | cut -c1-16   # 기대: cddf943ffec8d6a8
```

### 단계 1. bevfusion env
- `$BF/setup_fix.md` 순서를 따른다. 이 파일도 dirty tree의 수정본이므로 단계 0에서 옮긴 것을 읽는다.
  0. 전제: 시스템 CUDA toolkit 12.8(`/usr/local/cuda-12.8`)과 `gcc-13`/`g++-13`(`CC`/`CXX`로 지정). conda 툴체인은 쓰지 않는다(setup_fix.md 표 5행)
  1. conda-forge python 3.9 + openmpi 4.1 + ninja
  2. torch 2.8.0+cu128
  3. setuptools 59.5.0
  4. mmcv-full 1.7.2 소스 빌드
  5. mmdet 2.28.2 등
  6. 버전 재고정
  7. `tools/patch_mmcv_for_torch2.py`
  8. `python setup.py develop`
- **GPU arch:** 새 서버 GPU가 RTX 5090(sm_120)이 아니면 `TORCH_CUDA_ARCH_LIST`를 그 GPU에 맞춰 mmcv와 repo ops를 다시 빌드한다.
- 확인 항목:
  - `mmdet3d/ops/*/*.so` 13개
  - `torchpack dist-run -np 1 python -c "import mmdet3d"` 성공
  - `NCCL_SOCKET_IFNAME=lo`가 필요할 수 있다(옛 컨테이너에서는 필요했다)
- config 확인 [제안]: 캐시에 쓸 config를 resolve한 결과가 `runs/navsim-fusion-50x100/configs.yaml`과 같은지, `voxel_size [0.08,0.08,0.2]`인지 본다.

### 단계 2. 센서 다운로드 (선택지 A/B/B'/E일 때만)
1. **archive 대응 확인 [제안]:** archive 하나의 내용과 크기를 본다.
   ```bash
   U=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1
   wget -qO- $U/openscene_sensor_trainval_lidar/openscene_sensor_trainval_lidar_0.tgz | tar -tz | cut -d/ -f1-4 | sort -u | head
   # 크기: HF API (예: https://huggingface.co/api/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1/openscene_sensor_trainval_lidar)
   ```
   - archive가 log 단위로 묶여 있으면 B'(필요한 log가 든 archive만)가 가능하다. 아니면 400개를 모두 받는다.
2. **필요 파일 목록 [새 코드, 작음]:** `make_needed_files.py`를 만든다.
   - 입력: `future_train.yaml`의 token과 log pkl의 `cams[*].data_path` / `lidar_path`
   - 출력: archive 안 경로(`openscene-v1.1/sensor_blobs/trainval/<log>/CAM_F0/<tok>.jpg` 형식, 실제 prefix는 위 1에서 확인)의 목록
   - 대상 범위: F0/L0/R0 + `MergedPointCloud` `.pcd`. ReSMap까지 할 계획이면 이것으로 충분하다(ReSMap은 3캠만 쓴다). 8캠이 모두 필요하면 범위를 넓힌다.
   - 집합(E2E / stage-T / navtrain)은 §7의 결정을 따른다. 입력 yaml은 단계 3에서 만든다(단계 3을 먼저 실행).
   - 각 프레임의 sweep(f−1, f−2) lidar도 목록에 넣는다. §2-1 집합에서는 추가분이 0이지만, 생성기가 직접 포함하면 집합을 바꿔도 안전하다.
3. **스트리밍 추출 [제안]:**
   ```bash
   cd $DL; mkdir -p logs
   for i in $(seq 0 199); do for m in camera lidar; do
     wget -qO- $U/openscene_sensor_trainval_${m}/openscene_sensor_trainval_${m}_${i}.tgz \
       | tar -xz --skip-old-files --strip-components=2 -C $DL/trainval_sensor_blobs -T needed_${m}.txt 2> logs/x_${m}_${i}.err
     echo "$m $i ${PIPESTATUS[*]}" >> logs/dl_status.txt     # tar exit 2 ("Not found in archive")는 정상일 수 있다
   done; done
   ```
   - `--strip-components=2` 뒤의 경로가 기존 구조 `trainval_sensor_blobs/trainval/<log>/...`와 맞는지 archive 1개로 먼저 확인한다.
   - 이미 있는 파일을 덮어쓰지 않도록 `--skip-old-files`를 쓴다(위 명령에 넣었다).
   - `download_trainval.sh`를 그대로 쓰면 마지막에 `mv openscene-v1.1/meta_datas trainval_navsim_logs`, `mv openscene-v1.1/sensor_blobs trainval_sensor_blobs`를 한다. 대상 디렉터리가 이미 있으면 그 안에 `meta_datas/`, `sensor_blobs/`로 중첩돼 들어가 구조가 어긋난다. 기존 `$DL`에서 그대로 실행하지 않는다(선택지 A의 위험).
   - 병렬로 받으려면 archive 번호로 나눠 여러 개를 띄운다.
4. **lidar `.bin` 변환은 하지 않는다.** 기존 캐시와 infos는 `.pcd`를 직접 읽었다. 형식을 바꾸면 §6-2 재현 검증을 다시 통과해야 한다.

### 단계 3. 미래 token 목록 (ssr env, CPU, 수 분)
```bash
CUDA_VISIBLE_DEVICES="" $PY $W/make_token_lists.py
cat $W/counts.json   # 기대: train future_frames_to_run 112152, logs 1192 / test 12900, logs 136
```
- 절대경로가 하드코딩돼 있다(`$TC/bevfusion/cache_{train,val}_50x100`, `$DL/*_navsim_logs`). 같은 경로를 쓰면 고칠 필요가 없다.
- 이 스크립트는 단계 0에서 옮긴 `future_*.yaml`, `future_index_*.json`, `validate_test.yaml`, `counts.json`을 **덮어쓴다**. 옮긴 사본을 다른 이름으로 두고 덮어쓴 결과와 `cmp`해서 같으면 통과로 본다. 다르면 현재 프레임 캐시나 log pkl이 옛 서버와 다르다는 뜻이다.
- 결과가 navtrain 전체 기준이다. E2E나 stage-T로 줄이려면 `future_index_train.json`과 `$D/splits/*.parquet`의 token으로 걸러 새 yaml을 쓴다 [새 코드, 수십 줄].

### 단계 4. 재현 검증 먼저 (§6-2)
```bash
# (0) bevfusion data/ symlink (data/는 git에 없고 단계 0에서 옮기지 않았다). infos의 경로는 data/navsim 기준 상대경로다
mkdir -p $BF/data/navsim/sensor_blobs $BF/data/navsim/lidar $BF/data/infos
for s in trainval test; do
  ln -sfn $DL/${s}_sensor_blobs/$s $BF/data/navsim/sensor_blobs/$s
  ln -sfn $DL/${s}_sensor_blobs/$s $BF/data/navsim/lidar/$s
done
ln -sfn $DL/maps $BF/data/navsim/maps      # 옛 서버와 같은 형태(검출에는 필요 없을 수 있음)
# (1) infos 재현: validate_test.yaml(이미 캐시된 navtest 24 token) → infos 생성 → 기존 navsim_infos_val_navtest.pkl과 24/24 일치
cd $BF
CUDA_VISIBLE_DEVICES="" $BPY tools/data_converter/navsim_converter.py --navsim-logs $DL/test_navsim_logs/test \
  --sensor-root $DL/test_sensor_blobs --split test --scene-filter $W/validate_test.yaml --keyframes-only --check-files \
  --workers 8 --lidar-prefix lidar --lidar-ext .pcd --max-sweeps 2 --out $W/infos/validate_test.pkl
#    기대 로그: "24 frames (24 in official token set)", "mean 2.00 sweeps/frame"
# (2) teacher 재현: 24개를 50x100 teacher로 돌려 cache_val_50x100과 비교
export PATH=/home/external-user/miniconda3/envs/bevfusion/bin:$PATH NCCL_SOCKET_IFNAME=lo
torchpack dist-run -np 1 python $W/cache_teacher_future.py \
  configs/navsim/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml runs/navsim-fusion-50x100/epoch_20.pth \
  --cache-dir $W/validate_out --split val --ann-file $W/infos/validate_test.pkl
```
- (1)의 converter 옵션은 옛 `infos/validate_test.pkl`과 기존 `navsim_infos_val_navtest.pkl`의 형식(lidar prefix `lidar/test/…pcd`, sweep 2개, 8캠)에서 거꾸로 맞춘 것이다 [재구성]. 옛 명령 원문은 남아 있지 않다.
- 비교 스크립트는 옛 세션에서 즉석으로 쓴 것이다. 파일로 남아 있지 않다 [새 코드, 작음; §6-2 기준]. 검증 단계에서 같은 계산(BEV 상대 차이, score≥0.3 박스의 BEV 중심 Hungarian 매칭 0.5 m)을 약 30줄(`scipy.optimize.linear_sum_assignment`)로 다시 써서 옛 `validate_out/`에 돌려 §6-2 수치를 재현했다.

### 단계 5. future infos 생성 (bevfusion env, CPU)
```bash
cd $BF   # data/navsim/{sensor_blobs,lidar}/{trainval,test} symlink(단계 4 (0))가 $DL을 가리켜야 한다(infos 경로는 상대경로)
# test: L=$DL/test_navsim_logs/test R=$DL/test_sensor_blobs SP=test S=test
# train: L=$DL/trainval_navsim_logs/trainval R=$DL/trainval_sensor_blobs SP=trainval S=train
CUDA_VISIBLE_DEVICES="" nice -n 10 $BPY tools/data_converter/navsim_converter.py --navsim-logs $L --sensor-root $R \
  --split $SP --scene-filter $W/future_$S.yaml --keyframes-only --check-files --workers 8 \
  --lidar-prefix lidar --lidar-ext .pcd --max-sweeps 2 --out $W/infos/future_$S.pkl
```
- 전방 3캠만 추출했다면 **`--cameras CAM_F0 CAM_L0 CAM_R0`을 반드시 준다.** 기본값은 8캠이라 모든 프레임이 drop된다.
  - teacher 입력은 그대로다. 데이터셋은 config의 `camera_names`(F0/L0/R0)만 읽는다(`navsim_dataset.py:146, 243`, `configs/navsim/default.yaml:86-89`). 다만 옛 infos는 8캠으로 만들었으므로, 3캠 infos로 단계 4 (2)의 24개를 한 번 더 돌려 같은지 본다 [제안].
- **`--check-files`는 카메라만 검사한다.** lidar와 sweep `.pcd`는 검사하지 않는다(`navsim_converter.py:259-290`).
  - 옛 서버의 `future_train.pkl` 18,482개 중 8,075개가 sweep 누락이었다. 이대로 돌리면 `loading.py:140`에서 FileNotFoundError로 죽는다.
  - 사전 검사가 필요하다 [새 코드, 작음]. infos의 `lidar_path`와 `sweeps[*].data_path`가 모두 있는지 확인하고, 없는 프레임은 빼거나 데이터를 받는다. **sweep 수를 바꿔 우회하지 않는다**(teacher 입력이 달라진다).
- 기대값: test 12,900 frame, 715,399 box, drop 0 [실측, 옛 서버]. train은 받은 범위에 따라 다르다.
- **part 분할은 하지 않는다.** 아래 단계 6처럼 split당 한 번의 `-np N` 실행으로 돌린다.

### 단계 6. BEVFusion 미래 프레임 추론 (GPU)
```bash
cd $BF; export PATH=/home/external-user/miniconda3/envs/bevfusion/bin:$PATH NCCL_SOCKET_IFNAME=lo OMP_NUM_THREADS=2
torchpack dist-run -np $NGPU python $W/cache_teacher_future.py \
  configs/navsim/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml runs/navsim-fusion-50x100/epoch_20.pth \
  --cache-dir $TC/bevfusion/cache_val_50x100_future --split val --ann-file $W/infos/future_test.pkl \
  --skip-existing [--drop-bev]
# train도 같은 형식: --cache-dir $TC/bevfusion/cache_train_50x100_future --split train --ann-file $W/infos/future_train.pkl
```
- **`run_cache.sh`를 그대로 쓰지 않는 이유** [실측, 코드]:
  - part별로 단일 프로세스를 띄우면 각 프로세스가 자기 part의 `ann_file`과 `num_samples_expected`로 manifest를 덮어쓴다. 결국 마지막 part의 manifest만 남는다.
  - 9/11의 현재 프레임 캐시처럼 split당 `-np N` 한 번으로 돌린다. 분할은 DistributedSampler가 하고, rank별 tmp 파일과 `os.replace`로 저장한다.
- **`--drop-bev`:** O_T+나 박스 raster(§5-2)만 쓴다면 켠다(약 10 GB). F3(미래 BEV feature)나 H5를 할 계획이면 끈다(약 300 GB). 사용자가 정한다.
- **`--mem-frac`는 빼도 된다.** allocator 밖에서 약 17 GB를 써서 효과가 없었다.
  - 전용 GPU(VRAM 24 GB 이상)면 그대로 돌린다.
  - 공유 GPU면 먼저 진단한다. 단계마다 `torch.cuda.mem_get_info()`를 찍고, DDP 없이 / cudnn off / `CUDA_MODULE_LOADING=LAZY`로 각각 돌려 본다 [제안].
- **처리량 [추정]:** 전용 GPU 약 11.6 frame/s/GPU, 공유 GPU 약 5.8. 공유 값만 실측이다(`logs/cache_gpu{0,1}.log` 5.7–5.8 task/s, `memtest.log` 5.6–5.7). 11.6은 그 두 배로 둔 가정이고 전용 GPU 로그는 없다. CPU 디코딩이 상한이 될 수 있다(옛 서버 16코어에서 전체 약 69 frame/s [재구성, 근거 로그를 찾지 못함]).

| 대상 | 프레임 | GPU 1장 (전용 / 공유) | GPU 4장 (전용) |
|---|---:|---:|---:|
| navtest 4 s | 12,900 | 19분 / 37분 | 5분 |
| stage-T 4 s | 62,629 | 1.5 h / 3.0 h | 23분 |
| E2E 4 s | 93,606 | 2.2 h / 4.5 h | 34분 |
| navtrain 4 s | 112,152 | 2.7 h / 5.4 h | 40분 |

- **manifest의 z 규약 [수정 필요]:** manifest의 박스 필드 설명(`cache_teacher_future.py:340`, "`[x, y, z_gravity_centre, …]`")과 모듈 docstring(`:28`)에는 z가 중력중심이라고 적히지만, 실제로 저장하는 것은 bottom-z(`boxes.tensor`, `:260`)다(fact_features.md). 2D 점유에는 영향이 없다. `:340` 문자열을 "z_bottom"으로 고친다 [새 코드, 1–2줄]. 이미 있는 `cache_{train,val}_50x100` manifest도 같은 오기다.

### 단계 7. 사후 검사 (§6-3)
- npz 개수가 yaml token 수와 같은가
- token 집합의 차집합이 ∅인가
- manifest가 있고 `checkpoint_sha256_head == cddf943ffec8d6a8`인가
- `.tmp.npz`가 0개인가

### 단계 8. 미래 target 로더와 빌더 [새 코드]
- **`TeacherCache`는 고치지 않는다.** `_future` 거부는 안전장치다. 출력 폴더 이름의 `_future`도 그대로 둔다.
- `navsim/agents/para_ssr/refiner/future_teacher.py` (신규, 약 150–250줄 [추정])
  - `FutureTeacherCache`:
    - sha·layout 검사는 `TeacherCache`와 같다.
    - `bev_feature_stored=False`를 처리한다(지금 `load_bev`는 KeyError).
    - token 조회 순서: future 캐시 → 기존 현재 프레임 캐시. 미래 프레임 92,012개(train)와 10,871개(test)는 다른 token의 현재 프레임이라 기존 캐시에 있다.
  - `future_boxes(t0_token)` → [8, 200, 9]:
    - `future_index_*.json`으로 미래 token 8개를 찾는다.
    - pose는 log pkl의 `ego2global`로 t0 N frame에 옮긴다. `gt_future.frame_pose`와 `_to_n`을 재사용한다(lidar2ego = identity).
    - yaw와 속도 벡터도 회전한다.
    - score 임계는 0.3을 기본으로 한다(fact_features의 박스 raster와 같음). 임계는 결정 사항이다.
    - 미래 프레임이 없으면 `valid[k]=False`로 둔다.
- `tools/refiner/build_future_raster.py` (신규, CPU, worker 4 관례)
  - token마다 `[C, 50, 100]` uint8 raster를 S grid에 쓴다. C = k 1..8 × {agent, static} = 16. 정보원은 둘이다.
    - `--source gt`: `objects/` npz, `gt_future.query`, 0.5 s keyframe
    - `--source teacher`: `FutureTeacherCache.future_boxes`
  - 두 source가 **같은 채널과 같은 격자**를 쓰게 한다. 이것이 §5-2 비교의 전제다.
  - S grid 축 규약은 `adapters.teacher_to_s_grid`와 `s_grid_cell`을 따른다.
  - 출력: `$D/future_raster/{gt,teacher}/<tok[:2]>/<tok>.npz` + `manifest.json`(source, sha, 임계, 버전). 폴더 이름에 `_future`를 넣지 않아도 되지만, 현재 프레임 캐시와는 경로를 분리한다.
  - 크기: 16 × 50 × 100 B = 80 KB/token, E2E 85,109면 약 6.8 GB [추정]. npz 압축을 쓰면 더 작다.
  - **사각지대 처리:** teacher source에서는 시각 t의 로그 ego 기준 앞 32 m 밖을 "모름" 채널(또는 mask)로 따로 저장한다. GT source에서는 `gt_future.unknown_space`를 쓴다.

### 단계 9. (필요할 때만) ReSMap 미래
- **조건:** F5나 R_M+를 하기로 했을 때만 한다. O_T+, 박스 raster, R_GT-fut에는 필요 없다.
- **필요한 것 (모두 kyungmin 서버에 있음):**
  - maptracker repo의 `plugin/`
  - resmap env (torch 1.12 / mmcv-full 1.6 / mmdet3d 1.0.0rc6)
  - config `resmap_nav_stage3.py`, checkpoint `iter_63024.pth` (sha256 `0eaeda79…6f1a`)
  - `navsim_map_infos_*.pkl`
  - 위성 타일
- sm_120에서는 돌지 않는다(`kyungmin/SSR/report/19_planning_readout.md` §8).
- **navtest (가장 쉬움):** 기존 navtest 실행(71,460 frame, scene 순서)을 다시 하되 저장 조건을 "allow-list ∪ 미래 12,900 token"으로 넓힌다.
  - 생성기에 `--save-tokens` 같은 인자가 필요하다 [새 코드, 공동연구자 코드].
  - 용량은 약 +33 GB다 [추정].
  - 12,900개는 71,460 안에 있다고 본다 [실측, 간접]. navtest 136 log에서 CAM_F0가 디스크에 있는 프레임이 정확히 71,460개이고(검증 단계 계산), 12,900개는 모두 그 log의 디스크에 있다. kyungmin의 `navsim_map_infos_navtest.pkl` 자체는 열어 보지 못했다.
- **navtrain:** 센서(§단계 2), 그 frame의 위성 타일, 새 map infos가 필요하다.
  - log 전체를 새 폴더에 다시 돌려야 한다. 기존 root와 섞지 않는다.
  - 약 113k(navtrain) 또는 78.5k(E2E) frame, 약 214–310 GB다 [추정].
  - ReSMap은 train_logs로 학습했으므로, train 쪽 feature는 이미 본 데이터의 feature다(`resmap_cache.py` docstring).

### 단계 10. refiner/KD 통합 지점 (§5에서 고른 설계에 따라)

| 파일 | 지금 | 바꿀 것 [새 코드] |
|---|---|---|
| `navsim/agents/para_ssr/refiner/adapters.py` | `AdapterT/M`이 256ch 고정(`IN_CH_T`) | `build_adapter`에 arm `G`(GT raster), `F`(teacher raster) 추가. `in_ch=16`(또는 C)인 `AdapterT` 인스턴스. z-score 통계는 `compute_norm`을 재사용 |
| `navsim/agents/para_ssr/refiner/data.py` | `arm_teachers`가 none/T/M/TM만 처리. `TeacherCache`가 `_future` 거부 | `arm_teachers`에 G/F를 추가하고 `FutureRasterCache.load_bev(token)` → `[16,50,100]`. `TeacherCache`는 그대로 |
| `tools/refiner/train_refiner.py` | `--arm`, `--teacher-root`, norm 계산, `ARM_BRANCHES` / `load_run_norms`(`:401`) | `--arm G|F`, `--raster-root`, `ARM_BRANCHES`에 G/F와 norm 파일 이름 추가(E2가 `load_run_model`로 teacher를 읽을 때 필요) |
| `navsim/agents/para_ssr/refiner/e2e.py` | `teacher_arm()`이 T/M 외에는 raise. `GTLoader._teacher_caches()`가 arm T면 BEVFusion, **그 밖의 arm은 모두 `ResmapCache`**를 연다(`:266-273`). `GTLoader.load`의 실패 시 zeros가 `(256,50,100)`으로 하드코딩 | arm G/F 허용. `_teacher_caches()`에 G/F → `FutureRasterCache` 분기 추가(없으면 G/F teacher에 ReSMap BEV가 들어간다). zeros shape를 cache의 채널 수로. `KD_OK_KEY`로 token별 유무 전달(이미 있음) |
| `e2e.kd_loss` | teacher별 ok 마스크 평균, teacher 간 단순 평균 | 부분 커버리지(§5-4)일 때 ok가 0인 teacher도 분모에 들어가 KD 크기가 batch마다 달라진다. 유지할지, ok가 있는 teacher만 평균할지 정한다 |

---

## 5. 현재 refiner 틀과 연결하는 방법

**현재 틀** [37 §3, §8]
- 교정기는 초안 하나와 ego 상태, 그리고 BEV `[C,50,100]`(S grid)를 받는다. 출력은 12개 교정량이다(감속 6 + 옆 이동 6).
- 장면 읽기는 두 경로다. 통로 읽기는 48지점 × 17칸이고, 전체 읽기는 5×5 평균으로 만든 200 token이다.
- stage T에서 teacher 교정기(R_T, R_M)를 GT 벌점으로 학습한다. stage E(E2)에서 student 교정기는 GT 벌점과 KD(디코딩된 교정량 L1, ½·L_T + ½·L_M)로 학습한다.
- teacher 교정기는 **BEV 형태의 입력만 바꾸면** 같은 몸통과 같은 학습 코드를 쓸 수 있다. 미래 정보를 붙이는 가장 작은 변경점이 이것이다 [추론].

### 5-1. 설계 후보 (추천이 아니라 선택지)

| 후보 | teacher 교정기 입력 | 학습 (stage T) | E2 KD | 데이터 | 코드 변경 | 이 결과로 말할 수 있는 것 | 약점 |
|---|---|---|---|---|---|---|---|
| **P1. R_G (GT 미래 raster)** | `[16,50,100]` GT 미래 점유 (k=1..8, agent/static) | 기존 stage-T 파이프라인(39 §6.3 경로 B), arm G | R_T, R_M에 R_G를 더하거나 바꿈 | 센서 불필요. objects만 | 빌더 + arm 추가 (§단계 10) | 미래 특권 교정 KD의 천장. 그 정보가 카메라 student로 전달되는가 | aux teacher가 아님(35 요구 1과 충돌). GT 벌점과 정보원이 같음 |
| **P2. R_F (teacher 미래 raster)** | P1과 **같은 채널**, 정보원만 BEVFusion 미래 박스 | arm F. stage-T 집합의 미래 프레임 62,629개 필요 | 같음 | 센서 다운로드 필요 | P1 + `FutureTeacherCache` | P1 대비 손실 = teacher가 GT를 대신할 수 있는 정도 | 학습 장면에서 teacher ≈ GT(2.5 cm)라 P1과 거의 같을 가능성이 큼 [추론]. 그러면 "teacher가 필요하다"는 논거가 되지 않음 |
| **P3. R_T ⊕ G (t0 teacher BEV + GT 미래 raster)** | `[256+16,50,100]` | `AdapterTM`처럼 두 branch | R_T 자리 | P1과 같음 | P1 + concat adapter | t=0 장면 이해(teacher)와 미래(GT)의 분담 | teacher 몫을 떼기 어려움 |
| **P4. R_T+ (미래 BEV feature stack)** | t+k BEV를 t0 grid로 warp해 채널 축소(예: k=4, 8만, 각 256→32) | 새 adapter | 같음 | 센서 + bev 저장(약 166–297 GB) | warp(`grid_sample`) + 채널 축소 + 저장 형식 | 33 F3: GT에 없는 dense 미래 정보 | warp하면 전방 32 m 밖이 비고, 비용이 가장 큼. 30번의 "표현만 좋아지고 궤적은 그대로" 위험 |
| **P5. student aux loss (교정 KD 아님)** | – | – | student BEV에 작은 head를 달아 O(t,s)나 16ch raster를 예측. BCE, "모름" 칸 제외 | GT면 P1, teacher면 P2 | `StageE.loss`에 항 하나 + head | 33 F1/F3의 E2E 판 | KD 위치가 "표현"이라 35 요구 2(KD 위치가 분명할 것)와 거리가 있음 |

### 5-2. P1과 P2를 같은 격자로 맞추는 이유
- 사용자 요구 1(teacher가 GT 대비 필요함)을 미래 축에서 직접 물을 수 있는 유일한 형태다. 비교는 **R_F 대 R_G를 stage T에서, 같은 초안·같은 벌점·같은 몸통으로** 한다. 37 §6-0 판정 규칙과 같은 틀이다.
- 결과 해석은 셋 중 하나다 [추론].
  - R_F ≈ R_G: teacher 미래가 GT를 대신할 수 있다. 그러나 "teacher가 GT보다 필요하다"는 논거는 아니다.
  - R_F < R_G: 차이가 사각지대(32 m)나 검출 오차에서 오는지 A3식 분석이 필요하다.
  - R_F > R_G: 라벨 없는 장애물 등이 이유일 수 있다. 확인이 필요하다.
- **out-of-sample 비대칭 [추론, 미검증]:** BEVFusion은 navtrain token 프레임으로 학습했다. 미래 프레임의 약 55%(112,152/204,164)는 token이 아닌 프레임이다. 그래서 "teacher ≈ GT"가 미래 프레임에서는 덜 성립할 수 있다. 이 점이 P2를 P1과 다르게 만들 수 있는 유일한 경로다. 먼저 §6-4로 잰다.

### 5-3. E2에 붙이는 방식
- **추가:** KD = ⅓·(L_T + L_M + L_G or L_F). 기존 E2와 비교가 깨지므로 새 arm(E3계)이 된다.
- **교체:** R_T 자리에 R_T⊕G(P3). teacher 수가 같아 KD 비중이 유지된다.
- KD 비중 자동 맞춤(EMA 1:1)과 ramp는 그대로 쓴다. teacher 교정기는 고정·eval·checkpoint 미저장이며 `kd_teacher_runs`에 run dir을 추가하면 된다(`StageE.teachers`).
- **누수 규칙:** 미래 정보는 teacher 입력(dataloader target)으로만 들어간다. student 교정기와 PARA-SSR forward에는 들어가지 않는다. 추론 경로(`StageE.infer`)는 teacher를 부르지 않으므로 지금 구조가 이미 이를 보장한다 [실측, 코드 구조].

### 5-4. 부분 커버리지 (선택지 C2/E)
- `GTLoader.load`가 teacher BEV를 읽지 못하면(예외) `kd_ok=False`가 되고, `kd_loss`는 그 sample을 뺀다. 그래서 **미래가 있는 token에만 KD를 거는 것은 마스크 쪽 코드 변경 없이 가능하다** [실측, `e2e.py:326-333, 378-386`]. 전제는 새 `FutureRasterCache.load_bev`가 없는 token에서 0 raster를 돌려주지 말고 예외를 내는 것이다(§단계 10의 G/F 분기는 어차피 필요).
- 주의할 점 두 가지 [추론]:
  - ok가 없는 batch에서 그 teacher 항이 0인데도 분모(teacher 수)에 들어간다. 그래서 KD 크기가 batch마다 흔들린다.
  - 커버리지가 위치 편향을 가지면 KD가 특정 장면 유형에만 걸린다. 커버리지 비율과 편향을 학습 로그와 보고에 남긴다.

---

## 6. 검증 기준

합격 기준 수치는 예시이고 근거가 없다. 사용자가 정한다.

### 6-1. 환경과 데이터
- checkpoint sha256과 head가 §단계 0 값과 같아야 한다.
- resolved config가 학습 당시 `configs.yaml`과 같아야 한다(`voxel_size [0.08,0.08,0.2]`).
- `counts.json`: train 112,152 / test 12,900, log 1,192 / 136.
- 다운로드 뒤: 대상 미래 프레임 전부에 대해 F0/L0/R0 `.jpg`, lidar `.pcd`, sweep 2개 `.pcd`가 있어야 한다(누락 0).
- 센서 크기 표본 검사: 프레임당 약 2.0 MB(3캠 + lidar).

### 6-2. teacher 재현 (9/27에 한 것을 새 서버에서 다시) [재구성; 아래 옛 서버 수치는 검증 단계에서 `validate_out/` 24개로 다시 계산해 확인함]
- infos: 이미 캐시된 navtest 24 token으로 만든 infos가 기존 `navsim_infos_val_navtest.pkl`과 24/24 같아야 한다(재귀 비교).
- teacher 출력: 같은 24개를 돌려 `cache_val_50x100`과 비교한다.
  - **옛 서버 결과:** BEV 평균 상대 차이 1.46e-4(최대 3.42e-4), heatmap 1.10e-4. score≥0.3 박스 232/232가 Hungarian 매칭(BEV 중심, 0.5 m 안)으로 짝지어졌다. 위치 차이 중앙값 0.06 mm, 최대 0.39 mm. score 차이 최대 0.001, 속도 차이 최대 0.0039 m/s, label 불일치 0.
  - **새 서버 기준 예시:** 매칭 100%, 위치 최대 < 5 cm, BEV 상대 차이 < 1e-2. GPU나 드라이버가 다르면 차이가 더 커질 수 있다.
  - yaw 차이 최대 0.084 rad는 옛 서버에서도 원인을 보지 않았다. 대칭 물체(콘 등)의 yaw 모호성일 수 있다 [추론]. 점유 raster에는 영향이 작지만 확인 항목으로 둔다.
- 추가 [제안]: 미래 목록 중 기존 캐시와 겹치는 token 수백 개를 future infos 경로로 돌려 같은 비교를 한다.

### 6-3. 캐시 완결성
- npz 수가 yaml token 수와 같고, 집합 차집합이 ∅이어야 한다.
- manifest가 있고 sha head가 맞고, `.tmp.npz`가 0개여야 한다.
- `bev_feature_stored`가 의도와 맞아야 한다.

### 6-4. 좌표 변환과 teacher 미래 품질 [제안]
- **좌표:** 학습 token 중 표본(예: 1,000개)에서 `future_boxes(t0)[k]`와 GT objects의 `query(t=0.5k)`를 Hungarian 매칭(차량, score≥0.3, 0–32 m 전방)한다.
  - in-sample 프레임(미래 프레임이 다른 token의 현재 프레임인 경우): 중심 오차 중앙값이 cm 수준이어야 한다(현재 프레임 2.5 cm 기준).
  - 수 m 단위 오차가 나오면 pose 변환 버그다.
- **out-of-sample 품질:** 같은 비교를 token이 아닌 미래 프레임에서 해서 재현율, 정밀도, 중심 오차를 in-sample과 나란히 보고한다. P2를 할지 판단할 근거다.
- **사각지대 비율**(34 A3): 칸 중 "모름" 비율을 k별로 잰다.

### 6-5. raster
- GT source raster를 `objects/` 기반 GT 벌점의 충돌 판정과 대조한다. 충돌 장면에서 그 시각 그 위치가 점유로 찍혀야 한다.
- 축 검사: `validate_resmap_axes.py`처럼 S grid 방향(col 0 = 왼쪽)을 표본 시각화로 확인한다.
- 양자화 손실(34 A2): 0.64 m급 격자에서 충돌 판정이 바뀌는 비율을 GT만으로 잰다.

### 6-6. 학습 (stage T, E)
- stage T는 기존 판정 규칙(37 §6-0)과 같은 형식을 쓴다. R_G/R_F 대 R_none, 섞기 대조, navtest 1회 확인.
- E2E는 E1과 같은 설정에 KD teacher만 바꾼다. 커버리지(ok 비율)를 로그로 남긴다(`kd/ok_frac_i` 같은 스칼라 [새 코드]).
- **누수 검사:** 추론 때 future 캐시와 raster 경로를 읽지 않는지 확인한다. 예: 평가 실행 중 future 경로 파일 접근이 0인지 확인하거나, 평가 시 그 경로를 숨기고 돌린다.

---

## 7. 리스크와 열린 결정

### 7-1. 리스크

| 리스크 | 근거 | 대응 |
|---|---|---|
| 다운로드 크기와 시간을 모름 (1.6 vs 2.25 TB, 10 MB/s) | 35 §0-2(1.6 TB, 압축 기준, 출처 미기재)와 frame 수 × 프레임당 크기 계산(2.25 TB)이 다름. 브리프의 2.1 TB는 navtrain 1,192 log 분량(669,588 frame × 3.116 MB ≈ 2.09 TB)과 맞는다. archive 크기 실측 없음 | 단계 2-1에서 archive 1개와 HF 크기 API로 먼저 잰다 |
| 필요한 log만 받을 수 있는지 모름 | archive와 log의 대응표 없음 | 단계 2-1 |
| 공유 GPU에서 프로세스당 18.5 GB | 9/27 실측. 원인 미상 | 전용 GPU에서 돌리거나 진단 먼저 |
| sweep 누락으로 실행 중 사망 | `--check-files`가 lidar를 검사하지 않음. 8,075/18,482 | 단계 5 사전 검사 |
| manifest 덮어쓰기 | `run_cache.sh` part 구조 | split당 `-np N` 한 번 |
| dirty bevfusion tree를 clone으로 재현할 수 없음 | `git status` | working tree나 diff 이관 |
| teacher 미래 품질이 out-of-sample | token이 아닌 프레임 약 55% | §6-4 |
| P2 ≈ P1 → teacher 필요성 논거가 안 생김 | 학습 장면에서 teacher ≈ GT | 결과를 미리 해석 규칙으로 등록 |
| 부분 커버리지 편향 | 4 s 전부가 있는 token이 history 겹침 위치에 몰릴 수 있음 | 편향 분석 후 사용 |
| ReSMap 공동연구자 의존 | env, checkpoint, 타일이 kyungmin 서버에 있음 | ReSMap 미래는 필요할 때만 |
| GT 벌점과 R_G의 정보원 중복 | 37 §12-4 | E2−E1 해석 범위를 미리 한정 |
| 디스크 | 새 서버 여유 공간 모름 | `--drop-bev` + 3캠·lidar만 추출이면 약 170 GB(E2E) [추정] |

### 7-2. 사용자가 정할 것
1. **미래 정보원:** GT 미래만(P1/P3/P5-GT, 다운로드 없음) / teacher 미래(P2/P4, 다운로드) / 둘 다(P1 대 P2 비교).
2. **다운로드 여부와 범위:** 안 함(C1/C2/D) / stage-T 집합만(약 106 GB 디스크) / E2E 4 s(약 159 GB) / navtrain 전체. 8캠을 모두 받을지(F3·ReSMap 확장 대비) 3캠만 받을지.
3. **`--drop-bev`:** 박스만 쓸지(약 10 GB), BEV까지 저장할지(약 300 GB). P4나 H5를 할 계획이 있는지에 달려 있다.
4. **teacher 필요성 논거를 어디서 세울지:** out-of-sample 미래 품질 / 라벨 없는 장애물 / dense 미래 feature(P4) / 포기(R_G는 oracle 참고값으로만).
5. **E2 결합 방식:** 추가(KD 3-teacher) / 교체(R_T⊕G) / KD가 아닌 aux loss(P5).
6. **부분 커버리지 KD 허용 여부**와, 허용한다면 `kd_loss` 분모 규칙(§5-4).
7. **34 단계 A식 navtest 진단(C1)을 E2E와 무관한 진단으로 허용할지.** 사용자 요구 3·4와의 관계를 정해야 한다.
8. score 임계(0.3), 사각지대 처리(모름 vs 비어 있음), raster 해상도(S grid 50×100 대 경로 좌표 O(t,s)).
9. 현재 주 라인(E1 필요, 38 §8)과의 우선순위. 미래 작업은 E1/E2 결과 뒤인지, 병행인지.

**의존성상 가능한 순서 하나 [추론, 결정 아님]**
1. 다운로드 없이: 단계 1·3·4(재현) → 단계 5–7을 test split만(navtest 미래 추론 C1, 약 20–40분) → §6-4 품질 측정 → GT raster 빌더와 R_G(P1). P1은 bevfusion 쪽(단계 1–7)과 독립이라 먼저 해도 된다.
2. 그 결과를 보고 다운로드(B, E2E 4 s)를 할지 정한다 → R_F(P2) → E2 결합

---

## 8. 참고 파일

**보고서**
- `report/33_future_motion_problem_and_kd_design.md`: 문제 수치, 시간 이동 teacher, F1–F5 (§2–§9)
- `report/34_future_kd_experiment_design.md`: O(t,s), 단계 A–D, 4 s 근거 (§0–§7)
- `report/35_teacher_planner_reading_kd_design.md`: 34 대체 이유와 사용자 요구 5개, 데이터 16.5%, 캐싱 중단 (§0-1, §0-2, §9)
- `report/36_refiner_kd_architecture.md`: R_GT-fut arm, M6 (§1, §2)
- `report/37_teacher_refiner_experiment_overview.md`: 교정기, GT 벌점, E2 KD (§3, §5, §8, §12-4)
- `report/38_stageE_E2_results_analysis.md`: 현재 다음 단계 (§8)
- `report/39_new_server_setup_runbook.md`: 새 서버 전제, `_future` 거부 (§0, §3.2, §5.5, §5.9)
- `report/refiner_kd_design/fact_features.md`: npz 크기, z 규약, 미래 GT raster 16ch
- `report/future_motion_analysis/{cache_coverage.py,cache_coverage.json,future_frame_inventory.json}`
  - 주의: `cache_coverage.py`는 파일 존재가 아니라 메타데이터 필드만 확인했다. 그래서 "모든 프레임에 센서가 있다"는 33/34의 문장은 틀렸다.

**미래 캐시 도구** (`tools/future_teacher_cache/`, untracked)
- `make_token_lists.py`, `cache_teacher_future.py`, `run_cache.sh`
- `future_index_{train,test}.json`, `future_{train,test}.yaml`, `validate_test.yaml`
- `counts.json`, `train_disk_availability.json`
- `logs/{infos_test,infos_train,validate,validate50,memtest,cache_gpu0,cache_gpu1}.log`
- `validate_out/`, `memtest_out/manifest.json`
- 부분 캐시: `/home/external-user/datasets/teacher_cache/bevfusion/cache_val_50x100_future` (1,422 npz, manifest 없음)

**bevfusion**
- `tools/cache_teacher_bev.py` (untracked)
- `tools/data_converter/navsim_converter.py`: `--check-files`는 카메라만, `--cameras` 기본값 8, `:259-290`
- `tools/data_converter/navsim_lidar_to_bin.py`
- `mmdet3d/datasets/pipelines/loading.py:140`
- `configs/navsim/default.yaml`: `sweeps_num: 2`, F0/L0/R0
- `setup_fix.md`, `docs/NAVSIM.md` §3
- `runs/navsim-fusion-50x100/{epoch_20.pth,configs.yaml}`

**SSR refiner 코드**
- `navsim/agents/para_ssr/refiner/data.py`: `TeacherCache` `:146-169`, `arm_teachers` `:232`, `TEACHER_ROOTS` `:92`
- `navsim/agents/para_ssr/refiner/e2e.py`: `GTLoader` `:247-335`, `teacher_arm` `:337`, `kd_loss` `:378`, `StageE.teachers` `:425`
- `navsim/agents/para_ssr/refiner/adapters.py`: `IN_CH_T`, `build_adapter`, `teacher_to_s_grid`
- `navsim/agents/para_ssr/refiner/gt_future.py`: `frame_pose`, `_to_n`, `query`, `unknown_space`
- `tools/refiner/{train_refiner.py,build_future_objects.py}`

**ReSMap**
- `/home/external-user/datasets/teacher_cache/resmap/{README.md,meta.json,index.json,navtest/}`
- `kyungmin/SSR(-v2)/tools/readout/resmap/cache_teacher_kd.py`: memory 리셋 `local_idx==0`, `:27-31`
- `kyungmin/SSR/report/19_planning_readout.md` §6, §8

**다운로드**
- `/home/external-user/navsim/download/{download_trainval.sh,download_navtrain_hf.sh,download_test.sh}`

**확인하지 못한 것** (이 문서의 수치에 영향을 줄 수 있음)
- OpenScene archive 크기와 log 대응
- 새 서버의 대역폭, 디스크, GPU arch
- 미래 프레임에서의 teacher 정확도
- "4 s 전부 확보 가능" 부분집합에서 sweep 누락을 뺀 실제 비율과 그 편향
- ReSMap 처리량, ReSMap memory 리셋 구간의 정확한 정의(§2-2의 구간 수)
- 전용 GPU에서의 BEVFusion 처리량(11.6 frame/s는 가정)과 CPU 디코딩 상한 69 frame/s의 근거
- 18.5 GB GPU 메모리의 원인
- `--drop-bev` npz 실제 크기 (멤버 크기에서 계산함)
