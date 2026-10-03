# 40. 미래 데이터 방법론: 정리와 새 서버 세팅 계획

작성 2026-10-03. 읽기 전용 조사로 만든 문서다. 이 문서를 쓰는 동안 실행한 잡, GPU 사용, 다운로드는 없다.

검증 2026-10-03: 별도 검증 단계에서 수치와 절차를 원자료(로그 pkl, 캐시, 코드, 33–39)와 다시 대조했다. 검증에서 새로 계산한 값에는 "검증 단계"라고 적었다. 검증도 읽기 전용 CPU 계산만 했고 잡·GPU·다운로드는 없다.

갱신 2026-10-03 (§9 추가): "받고 → 캐싱 → 원본 지우기"가 가능한지 보려고 OpenScene archive 구조를 쟀다. HF에서 archive 0쌍 전체와 400개 archive의 앞 2 MB를 스트리밍으로 읽었다(네트워크 약 10.3 GB, 디스크에 저장한 archive 바이트 없음). 이 측정으로 바뀐 §0·§3·§7-2의 수치에는 "(§9에서 갱신)"을 붙였다. GPU와 잡은 쓰지 않았다.

갱신 2026-10-03 (§10 추가): 사용자 결정(아래 상자)에 맞춰 운영안을 §10으로 확정했다. §9-8에서 [새 코드]였던 목록 생성기와 스트리밍 추출기를 구현하고 시험했다(합성 서버 시험 28/28, HF 실제 archive 쌍 #146 시험 통과). 시험은 scratch 폴더에만 풀었고 `/home/external-user/{navsim,datasets,ssd}`의 기존 데이터에는 쓰지 않았다. GPU는 쓰지 않았다. 바뀐 §0·§3·§4 단계 2·§7-2·§9-0·§9-8에는 "(§10에서 확정)"을 붙였고, 이전 분석은 지우지 않았다.

검토 2026-10-03 (§10): §10의 명령을 스크립트 인자·동작과 대조하고(`--help`, 목록 생성기 재실행, TEST A 재실행 28/28) 수치를 `summary.json`과 시험 로그에 다시 맞췄다. 고친 것: §10-3에 `TOK`/`HS`/`NGPU` 변수(7번이 navtrain 5 s 입력으로 고정돼 있던 문제), 4번의 "8개 스트림" 표현, 6번 `kept_files` 식, §4 단계 0 이관 목록에 `xfer/` 추가, `stream_extract.sh`의 반영 실패 시 상태 JSON 처리(§10-5 5번), §7-1·§8의 해소된 항목 표시.

갱신 2026-10-03 (§10-7 ReSMap): 사용자 요청("ReSMap도 적용 가능하게 데이터 구성")에 맞춰 ReSMap 미래 추론에 필요한 데이터를 §10-7로 정리했다. `make_needed_files.py`에 B 방식(log 전 프레임 3캠) 옵션 `--resmap-full-logs`를 더했고, 옵션 없이 돌린 목록은 이전과 바이트 단위로 같음을 확인했다(E2E 4 s, navtrain 5 s). §10-0, §10-1 3번, §10-3 1번을 함께 고쳤다.

**표기**
- [실측]: 파일이나 로그에서 직접 확인한 값, 또는 기존 파일을 CPU로 전수 계산한 값
- [재구성]: 2026-09-27 세션 기록에서 되살린 명령이나 값
- [추정]: 측정값에서 외삽한 값
- [추론]: 근거는 있지만 검증하지 않은 판단
- [제안]: 아직 실행하지 않은 절차
- [새 코드]: 지금 저장소에 없어서 새로 써야 하는 코드
- [구현·시험됨]: 이전에 [새 코드]였고 지금은 구현해 시험까지 한 코드. 시험 범위는 §10-5에 적었다

**읽는 순서**
- 처음 보는 세션: §0 → §1 → §3 → §7
- 세팅을 할 때: §2 → §4 → §6
- **센서를 받아 캐싱할 때: §10** (확정 운영안. §4 단계 2와 §9-8보다 우선)
- 전송량·archive 구조의 근거와 "원본 삭제" 분석: §9

> **결정 (2026-10-03, 사용자)**
> - OpenScene v1.1 trainval 센서 원본(400개 archive, 2,124 GB)은 **저장하지 않는다.**
> - archive는 스트리밍으로 받는다. 받는 동안 **캐싱에 필요한 파일만 풀어 보관**하고, 나머지는 스트림에서 바로 버린다. archive 바이트는 디스크에 쓰지 않는다.
> - 풀어 둔 파일은 캐싱 뒤에도 남기는 것이 기본값이다. 그래서 teacher·config·BEV 저장 여부는 다시 받지 않고 바꿀 수 있다. 되돌릴 수 없는 것은 버린 부분(나머지 5캠, 고른 범위 밖 프레임)뿐이다.
> - **ReSMap도 적용할 수 있게 구성한다**(사용자 요청). ReSMap 입력은 BEVFusion과 같은 앞카메라 3장(+위성)이라, 같은 추출본으로 충분하다(A 방식, 추가 0 GB). 자세한 것은 §10-7.
> - 운영안·명령·검증 기준·시험 결과·남은 결정은 **§10**에 있다. 구현된 도구(`SSR/tools/future_teacher_cache/xfer/`): `make_needed_files.py`, `stream_extract.sh`, `rstream.py`.

**전제:** 새 서버에는 NAVSIM 다운로드, 현재 프레임 teacher 캐시(`cache_{train,val}_50x100`, ReSMap root + navtest)가 있다. 나머지 캐시는 `report/39_new_server_setup_runbook.md`(이하 39)대로 다시 만든다. 경로는 옛 서버와 같은 절대경로라고 가정한다(39 §3.2).

---

## 0. 요약

**확정된 것 (2026-10-03, §10)**
- 다운로드 방식: 선택지 B(전부 스트리밍하되 필요한 파일만 추출)로 한다. 원본 archive는 저장하지 않고, 풀어 낸 필요 파일은 보관한다.
- 전송량은 어느 범위든 2,124 GB다. 보관하는 센서는 E2E 4 s 약 159 GB, navtrain 5 s 약 227 GB다(S1, F0/L0/R0 + lidar + sweep) [추정, 목록 생성기 실측 파일 수 × archive 0 평균 크기].
- 도구는 구현·시험했다(§10-5). 남은 결정은 범위(E2E 4 s / navtrain 5 s), 카메라(3 / 8캠), ReSMap 연속 구간 필요 여부다(§10-1).

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
   - **전송량은 2,124 GB(camera 1,242 + lidar 882, tgz 기준)다 [실측, HF API] (§9에서 갱신, 이전 값 "약 1.6–2.25 TB [추정]").** 35 §0-2의 1.6 TB는 틀렸다. 2.25 TB는 전부 풀었을 때의 크기로 맞다(실측 프레임당 크기로 약 2.25 TB).
   - archive와 log의 공식 대응표가 있고(`openscene_sensor_trainval_0-199.json`), archive는 log 단위로 정렬돼 있다. **그러나 stage-T·E2E·navtrain 어느 집합이든 400개 archive가 모두 필요하다.** 필요한 log만 골라 받아도 줄지 않는다 [실측] (§9에서 갱신).
   - 시간은 10 MB/s면 약 59 h, 50 MB/s면 약 11.8 h, 100 MB/s면 약 5.9 h다 [추정] (§9에서 갱신). 이 서버에서 HF로 잰 값은 스트림당 1.2–2.75 MB/s, 합산 약 4–5.6 MB/s였다(약 148 h). 새 서버 값은 모른다.
   - (§10-5에서 추가) 구현한 도구로 archive 쌍 #146을 받았을 때는 camera 1.35, lidar 0.67 MB/s, 2스트림 합산 1.64 MB/s였다. 이 속도면 전체가 약 15일이다. 그래서 새 서버에서 대역폭과 병렬 수를 먼저 잰다(§10-3 3–4번).
2. **디스크:** 스트리밍 추출로 필요한 파일만 남기면 센서는 106–190 GB다 [추정]. teacher npz는 `--drop-bev`면 약 10 GB, BEV까지 저장하면 약 300 GB다 [추정]. **원본은 캐싱이 끝나면 지울 수 있다.** archive 쌍 단위로 받기 → 추론 → 지우기를 돌리면 원본이 동시에 디스크에 있는 양은 수–수십 GB다 (§9에서 갱신).
   - (§10에서 확정) 원본 archive는 디스크에 쓰지 않는다. 풀어 낸 필요 파일(106–227 GB)은 캐싱 뒤에도 남긴다. 그래서 `--drop-bev`가 안전한 기본값이 된다(bev가 필요해지면 남긴 파일로 다시 추론). 새 서버 합계는 `--drop-bev` 기준 E2E 4 s 약 1.67 TB, navtrain 5 s 약 1.75 TB다(§10-2).
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
| **A. OpenScene 전체 다운로드 + 전체 추출** | `download_trainval.sh` 그대로 | **2,124 GB** [실측] (§9에서 갱신) | 약 2.25 TB(센서) + npz 10–300 GB | 10 MB/s 기준 약 59 h. 50 MB/s면 약 11.8 h (§9에서 갱신) | navtrain 4 s 2.7 GPU-h (전용) | 디스크 부족. 기존 `trainval_sensor_blobs`와 섞여 덮어쓸 위험 | 무엇이든 할 수 있다(F1–F5, 모든 split) |
| **B. 전부 받되 필요한 파일만 스트리밍 추출 [선택됨, §10에서 확정]** | Range 재개 스트리머 `\| tar -xz -T needed.txt` (§9-8에서 갱신, 이전: `wget -O- … \| tar`) | A와 같음(400개 archive 전부, 2,124 GB [실측]) | E2E 4 s 159 GB, stage-T만 106 GB (+ npz). archive 쌍 단위로 처리하고 지우면 동시 최대 수–수십 GB (§9에서 갱신) | A와 같음 | A와 같음 | 파일 목록 생성기가 필요함 [새 코드]. tar가 "Not found in archive"로 exit 2를 내므로 무시 처리가 필요함(archive별 목록이면 exit 2는 진짜 누락이다, §9-8). 이어받기 없는 `curl | tar`는 끊겼다(HTTP/2 exit 92 실측, §9-1). wget은 기본으로 Range 재시도를 한다(로컬 확인, HF 미시험) | A와 같음. 다만 원본을 지우면 범위를 넓힐 때 다시 받아야 함(§9-5) |
| **B'. 필요한 log가 든 archive만 받기** | archive와 log의 대응을 먼저 확인 | **절약 0: 어느 집합이든 400개 전부 필요** [실측] (§9에서 갱신) | B와 같음 | B와 같음 | 같음 | archive는 log 단위가 맞지만, 필요한 log가 200쌍 전부에 퍼져 있다(§9-1). B'는 B와 같아진다 | B와 같음 |
| **C1. navtest만 (34 단계 A식 진단)** | 다운로드 없음 | 0 | npz 1.1 GB(`--drop-bev`) / 34 GB | 준비 반나절 [추정] | 12,900프레임, 전용 약 19분, 공유 약 37분 | 학습이 없다. 학습된 E2E 궤적(E0 navtest 궤적)을 전제로 하므로 사용자 요구 4와 충돌 [35 §0-1] | H1: "미래 프레임 teacher 출력이 교정 정보로 쓸 만한가"만 답한다. teacher-미래 vs GT-미래 vs 등속 비교 |
| **C2. navtrain 중 이미 디스크에 있는 것만** | sweep까지 있는 약 10,407프레임 | 0 | 약 1 GB / 28 GB | – | 약 15분 | 4 s 전부가 있는 token은 BEV∪디스크 기준 30.1%가 상한이고, sweep 누락으로 더 줄어든다 [미계산]. 위치 편향 가능성(history가 겹치는 위치에 몰림) [추론] | 파일럿 정도만 된다. KD를 그 token에만 걸 수는 있다(§5-4). 그러나 편향된 부분집합이다 |
| **D. GT 미래만 (R_GT-fut)** | GT objects로 미래 raster를 만들어 stage-T 교정기를 학습한 뒤 E2에 KD | 0 | raster 약 7 GB(16ch uint8 × 85,109) [추정] | raster 빌더 + stage T 경로 B(39 §6.3, CPU 약 5–6 h [추정]) | stage-T 교정기 1개 약 1.5 h(39 §6.3 ⑧: run-4 T 89분 [실측]) + E2 결합 학습 약 20–21 h × 4 GPU(37 §11 표) | aux teacher가 아니다(GT 특권). 35의 "teacher 필요성" 요구와 충돌한다. GT 벌점과 정보원이 같아서 E2−E1과 구분하기 어렵다 [37 §12-4] | "미래를 아는 교정기의 KD가 E2E로 전달되는가"의 천장 |
| **E. 장면 샘플링 + B** | 34 §1-1: navtrain 2–3만 장면 | B와 같음(센서가 archive에 섞여 있으므로) | stage-T 집합이면 106 GB(§2-4). 2–3만 장면을 stage-T처럼 흩어 뽑으면 약 80–120 GB [추정: 프레임 × DL 비율 0.84 × 2.023 MB] | B와 같음 | stage-T 집합이면 62,629프레임, 전용 약 1.5 GPU-h. 2–3만 장면이면 약 4.8–7.1만 프레임 [추정] | 전송량은 줄지 않는다. 34 §1-1의 "2.5–3.5만 프레임"은 navtrain 전체 값(11.6만)을 장면 수에 비례해 줄인 것이라 과소다. 표본에서는 이웃 token끼리 미래 프레임을 덜 공유한다(stage-T: token당 2.38 프레임, navtrain 전체: 1.09). stage-T 집합(26,366 token)을 그대로 쓰면 자연스러운 표본이 된다 | stage-T 교정기 학습에는 충분하다. E2 KD는 85,109 중 일부에만 걸린다 |

**정리 [추론]**
- **(§10에서 확정)** 사용자가 B를 골랐다. 원본은 저장하지 않고, 풀어 낸 필요 파일은 보관한다. 파일 목록 생성기와 스트리밍 추출기는 구현·시험됐다(§10-5). "tar exit 2 무시 처리"는 필요 없어졌다: 목록이 archive별이라 exit 2와 "Not found"는 실패로 처리한다. 범위(E2E 4 s / navtrain 5 s)는 §10-1에서 정한다. 아래 문장들은 결정 전 분석이다.
- 다운로드가 필요한 선택지(A/B/B'/E)는 전송량이 **같다(2,124 GB, §9에서 갱신)**. 다른 것은 디스크 사용량뿐이다. 원본을 계속 보관할 필요는 없다. 받기 → 캐싱 → 지우기 운영안은 §9에 있다. 다운로드를 하기로 하면 **B를 E2E 4 s 집합 기준으로 한 번에 하는 것**이 범위를 나중에 넓힐 필요가 적다.
- 다운로드 없이 할 수 있는 것은 C1(진단), C2(편향된 파일럿), D(GT 특권)뿐이다.
- C1과 D는 서로 독립이라 병행할 수 있다.
- teacher-미래가 GT-미래보다 나은지(요구 1)를 직접 보려면 **D와 C1 또는 A/B를 같은 raster 형식으로 맞춰 비교**해야 한다(§5-2).

---

## 4. 새 서버 세팅 절차

**실행 순서 (검증 단계에서 고침):** 0 → 1 → 3 → 4 → (2, 다운로드할 때만) → 5 → 6 → 7 → 8 → (9) → 10.
- 단계 2-2의 파일 목록 생성기는 단계 3이 만드는 `future_*.yaml`(또는 걸러 낸 yaml)이 필요하다. 그래서 단계 3이 단계 2보다 먼저다.
- 단계 4(재현 검증)는 bevfusion `data/navsim` symlink와 `infos/validate_test.pkl`이 필요하다. 둘 다 단계 4 안에서 만든다.
- (§10에서 확정) 다운로드하는 경우 단계 2는 §10-3을 따른다. 전송이 수일 걸릴 수 있으므로 단계 0 → 3 다음에 바로 전송을 시작하고, 단계 1·4는 전송하는 동안 병행한다. 단계 5–7은 전송과 전역 완결성 검사(§10-3 7번)가 끝난 뒤에 한다.
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
| `$REPO/tools/future_teacher_cache/` 중 `make_token_lists.py`, `cache_teacher_future.py`, `run_cache.sh`, `*.yaml`, `*.json`, `xfer/` | **2026-10-03부터 git에 포함**(branch `exp-refine`). clone하면 따라오므로 따로 옮기지 않는다. `infos/`, `logs/`는 gitignore 대상이라 새로 만든다. `memtest_out/`, `validate_out/`도 커밋하지 않았다. `cfg_cf3e02a/`(100×100 config)와 `old_100x100/`은 가져오지 않는다 |
| `$BF` working tree 전체(`runs`, `data` 제외, `.git`·`build/` 포함 약 918 MB) 또는 `git diff` + untracked 2개(`tools/cache_teacher_bev.py`, `chain_stage2.sh`) | 캐시를 만든 코드는 커밋 `cf3e02a`가 아니라 그 위의 **dirty tree**(수정 10개 파일: config 4개, `navsim_dataset.py`, `transforms_3d.py`, `fov_utils.py`, `hungarian_assigner.py`, `setup_fix.md`, `docs/NAVSIM.md`)다. clone만 하면 config가 100×100(`voxel_size [0.04,0.08,0.2]`)이 된다. working tree를 통째로 옮겨도 `build/`와 `.so`는 단계 1에서 다시 빌드한다. `.git`이 없으면 manifest의 `git_commit`이 null이 된다 |
| `$BF/data/infos/navsim_infos_val_navtest.pkl` (156,409,747 B) | 단계 4 infos 재현의 비교 기준. 위 "data 제외"에 걸리므로 따로 옮긴다 |
| `$BF/runs/navsim-fusion-50x100/epoch_20.pth` (481,739,184 B) + `configs.yaml` | teacher 가중치와 학습 당시의 resolved config |
| `$REPO/tools/future_teacher_cache/xfer/` 전체 (약 6 MB; `run_*/`가 생긴 뒤면 그 폴더도) | (§10에서 추가) 다운로드 도구와 출처 고정 파일. `hf/tree_*.json`(크기·`lfs.oid`), `hf/map_trainval.json`, `results/stream_test_146/extra_*_146.txt`(§10-3 3번 provenance 대조)가 여기에만 있다 |
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

**(§10에서 확정) 이 단계는 §10-3의 절차로 실행한다.** 아래 1–3은 결정 전 기록이다. 1은 §9-1에서 해소됐고, 2는 `make_needed_files.py`로 구현됐고, 3의 명령 스케치는 `stream_extract.sh`로 대체됐다. 4는 그대로 유효하다.

1. **archive 대응 확인 [제안 → §9-1에서 해소]:** archive 하나의 내용과 크기를 본다. 결과: log 단위로 정렬돼 있지만 어느 집합이든 400개가 모두 필요하다. 명령 원문은 생략한다(`main` 대신 revision sha로 고정한 URL은 §9-8, §10-3).
2. **필요 파일 목록 [새 코드 → 구현·시험됨: `$W/xfer/make_needed_files.py`, §10-3 1번]:** 아래는 원래 요구 사항이다. 구현은 이를 모두 만족한다(입력은 parquet·yaml·json·txt token 목록, sweep 포함, 디스크에 있는 파일 제외, archive별 목록).
   - 입력: `future_train.yaml`의 token과 log pkl의 `cams[*].data_path` / `lidar_path`
   - 출력: archive 안 경로(`openscene-v1.1/sensor_blobs/trainval/<log>/CAM_F0/<tok>.jpg` 형식, 실제 prefix는 위 1에서 확인)의 목록
   - 대상 범위: F0/L0/R0 + `MergedPointCloud` `.pcd`. ReSMap까지 할 계획이면 이것으로 충분하다(ReSMap은 3캠만 쓴다). 8캠이 모두 필요하면 범위를 넓힌다.
   - 집합(E2E / stage-T / navtrain)은 §7의 결정을 따른다. 입력 yaml은 단계 3에서 만든다(단계 3을 먼저 실행).
   - 각 프레임의 sweep(f−1, f−2) lidar도 목록에 넣는다. §2-1 집합에서는 추가분이 0이지만, 생성기가 직접 포함하면 집합을 바꿔도 안전하다.
3. **스트리밍 추출 [제안 → 구현·시험됨: `$W/xfer/stream_extract.sh`, §10-3 3–6번]:** 이전의 `wget -qO- … | tar -xz --skip-old-files … -T needed_${m}.txt` 루프 스케치는 지웠다. 바뀐 점은 다음과 같다.
   - 목록이 archive별이라 tar exit 2 / "Not found in archive"는 정상이 아니라 실패다.
   - `--skip-old-files`로 최종 폴더에 바로 풀지 않는다. 끊긴 스트림이 잘린 파일을 남기고 재시도 때 `--skip-old-files`가 그것을 지키기 때문이다. archive별 staging 폴더에 풀고, 검사를 통과한 뒤 hard link로 옮긴다(덮어쓰기 없음, §10-5).
   - `--strip-components=2` 뒤의 경로가 기존 구조와 맞는 것은 §9-1(archive 0)과 §10-5(#146)에서 확인했다.
   - `download_trainval.sh`를 그대로 쓰면 마지막에 `mv openscene-v1.1/meta_datas trainval_navsim_logs`, `mv openscene-v1.1/sensor_blobs trainval_sensor_blobs`를 한다. 대상 디렉터리가 이미 있으면 그 안에 `meta_datas/`, `sensor_blobs/`로 중첩돼 들어가 구조가 어긋난다. 기존 `$DL`에서 그대로 실행하지 않는다(선택지 A의 위험).
   - 병렬은 `stream_extract.sh -P N`으로 한다(§10-3 4번).
4. **lidar `.bin` 변환은 하지 않는다.** 기존 캐시와 infos는 `.pcd`를 직접 읽었다. 형식을 바꾸면 §6-2 재현 검증을 다시 통과해야 한다.

### 단계 3. 미래 token 목록 (ssr env, CPU, 수 분)
```bash
CUDA_VISIBLE_DEVICES="" $PY $W/make_token_lists.py
cat $W/counts.json   # 기대: train future_frames_to_run 112152, logs 1192 / test 12900, logs 136
```
- 절대경로가 하드코딩돼 있다(`$TC/bevfusion/cache_{train,val}_50x100`, `$DL/*_navsim_logs`). 같은 경로를 쓰면 고칠 필요가 없다.
- 이 스크립트는 git에 들어 있는 `future_*.yaml`, `future_index_*.json`, `validate_test.yaml`, `counts.json`을 **덮어쓴다**. 실행 뒤 `git status --short $W`에 이 파일들이 바뀐 것으로 나오지 않으면(= 옛 서버 결과와 같으면) 통과로 본다. 다르면 현재 프레임 캐시나 log pkl이 옛 서버와 다르다는 뜻이다.
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
  - (§10에서 확정) 다운로드 경로에서는 `make_needed_files.py` 재실행이 이 사전 검사를 대신한다(§10-3 7번). 이 스크립트는 sweep을 `navsim_converter.build_sweeps`와 같은 규칙으로 세고, 실행 프레임 중 sweep이 빠진 수(`run_frames_on_disk_missing_some_sweep`)를 낸다 [제안, 추출 후 재실행은 미시험].
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
| 다운로드 크기와 시간을 모름 (1.6 vs 2.25 TB, 10 MB/s) | 35 §0-2(1.6 TB, 압축 기준, 출처 미기재)와 frame 수 × 프레임당 크기 계산(2.25 TB)이 다름. 브리프의 2.1 TB는 navtrain 1,192 log 분량(669,588 frame × 3.116 MB ≈ 2.09 TB)과 맞는다. archive 크기 실측 없음 | 단계 2-1에서 archive 1개와 HF 크기 API로 먼저 잰다. **(§9-1에서 해소: 2,124 GB. 시간은 새 서버 대역폭에 달림, §10-3 3–4번에서 잰다)** |
| 필요한 log만 받을 수 있는지 모름 | archive와 log의 대응표 없음 | 단계 2-1. **(§9-1에서 해소: 대응표는 있으나 어느 집합이든 400개 전부 필요)** |
| 공유 GPU에서 프로세스당 18.5 GB | 9/27 실측. 원인 미상 | 전용 GPU에서 돌리거나 진단 먼저 |
| sweep 누락으로 실행 중 사망 | `--check-files`가 lidar를 검사하지 않음. 8,075/18,482 | 단계 5 사전 검사 |
| manifest 덮어쓰기 | `run_cache.sh` part 구조 | split당 `-np N` 한 번 |
| dirty bevfusion tree를 clone으로 재현할 수 없음 | `git status` | working tree나 diff 이관 |
| teacher 미래 품질이 out-of-sample | token이 아닌 프레임 약 55% | §6-4 |
| P2 ≈ P1 → teacher 필요성 논거가 안 생김 | 학습 장면에서 teacher ≈ GT | 결과를 미리 해석 규칙으로 등록 |
| 부분 커버리지 편향 | 4 s 전부가 있는 token이 history 겹침 위치에 몰릴 수 있음 | 편향 분석 후 사용 |
| ReSMap 공동연구자 의존 | env, checkpoint, 타일이 kyungmin 서버에 있음 | ReSMap 미래는 필요할 때만 |
| GT 벌점과 R_G의 정보원 중복 | 37 §12-4 | E2−E1 해석 범위를 미리 한정 |
| 디스크 | 새 서버 여유 공간 모름 | `--drop-bev` + 3캠·lidar만 추출이면 약 170 GB(E2E) [추정]. (§10-2에서 갱신) 보관 센서 + npz는 E2E 4 s 약 167 GB, navtrain 5 s 약 239 GB, 새 서버 합계 1.67 / 1.75 TB, 권장 약 2 TB |

### 7-2. 사용자가 정할 것
1. **미래 정보원:** GT 미래만(P1/P3/P5-GT, 다운로드 없음) / teacher 미래(P2/P4, 다운로드) / 둘 다(P1 대 P2 비교).
2. **(방식은 §10에서 확정, 범위는 §10-1에 남음)** 방식: B, 원본 저장 안 함, 풀어 낸 필요 파일 보관. 범위 기본값은 navtrain 5 s(보관 227 GB) [제안], 다른 후보는 E2E 4 s(159 GB). 카메라 기본값은 3캠 [제안]. 아래는 결정 전 문장이다.
   **다운로드 여부와 범위:** 안 함(C1/C2/D) / stage-T 집합만(약 106 GB 디스크) / E2E 4 s(약 159 GB) / navtrain 전체. 8캠을 모두 받을지(F3·ReSMap 확장 대비) 3캠만 받을지. **전송량은 어느 범위든 2,124 GB로 같다.** 그래서 범위를 넓히는 비용은 일시 디스크와 GPU 시간뿐이다(navtrain 5 s, trainval 전 프레임까지 §9-3) (§9에서 갱신).
3. **`--drop-bev`:** 박스만 쓸지(약 10 GB), BEV까지 저장할지(약 300 GB). P4나 H5를 할 계획이 있는지에 달려 있다.
   - (§10에서 확정) 기본값은 `--drop-bev`다. 풀어 낸 입력을 보관하므로 이 선택은 되돌릴 수 있다. bev가 필요해지면 보관 파일로 다시 추론한다(전용 GPU 1장 약 2.2–3.2 h).
4. **teacher 필요성 논거를 어디서 세울지:** out-of-sample 미래 품질 / 라벨 없는 장애물 / dense 미래 feature(P4) / 포기(R_G는 oracle 참고값으로만).
5. **E2 결합 방식:** 추가(KD 3-teacher) / 교체(R_T⊕G) / KD가 아닌 aux loss(P5).
6. **부분 커버리지 KD 허용 여부**와, 허용한다면 `kd_loss` 분모 규칙(§5-4).
7. **34 단계 A식 navtest 진단(C1)을 E2E와 무관한 진단으로 허용할지.** 사용자 요구 3·4와의 관계를 정해야 한다.
8. score 임계(0.3), 사각지대 처리(모름 vs 비어 있음), raster 해상도(S grid 50×100 대 경로 좌표 O(t,s)).
9. 현재 주 라인(E1 필요, 38 §8)과의 우선순위. 미래 작업은 E1/E2 결과 뒤인지, 병행인지.
10. **(§10에서 확정)** 원본 archive는 저장하지 않는다. 풀어 낸 필요 파일은 캐싱 뒤에도 남긴다(기본값. 지우려면 §10-1 4번). 그래서 아래의 "재DL" 경우 중 teacher·config 변경과 bev 미저장은 재DL 없이 처리된다. 8캠, 범위 밖 프레임, ReSMap 연속 구간만 남는다. 아래는 결정 전 문장이다.
    **원본 센서를 캐싱 후 지울지, 지운다면 무엇을 남길지** (§9에서 추가). 원본을 지운 뒤 다시 필요해지면(§9-5: teacher·config 변경, bev 미저장 상태의 F3/P4, ReSMap 미래, 8캠) 2,124 GB를 다시 받아야 한다. 그래서 삭제 전에 확정한다(§9-4).
    - 남길 것: bev feature(E2E 4 s 248 GB), 전방 3캠(ReSMap 미래·학생 미래용 49 GB–), golden 샘플(약 3 GB)
    - 파이프라인: P-stream-all 또는 P-chunked(§9-2)

**의존성상 가능한 순서 하나 [추론, 결정 아님]**
1. 다운로드 없이: 단계 1·3·4(재현) → 단계 5–7을 test split만(navtest 미래 추론 C1, 약 20–40분) → §6-4 품질 측정 → GT raster 빌더와 R_G(P1). P1은 bevfusion 쪽(단계 1–7)과 독립이라 먼저 해도 된다.
2. 그 결과를 보고 다운로드(B, E2E 4 s)를 할지 정한다 → R_F(P2) → E2 결합. (§10에서 갱신) 방식은 B로 확정, 범위 기본값은 navtrain 5 s(§10-1)

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

**미래 캐시 도구** (`tools/future_teacher_cache/`, 2026-10-03부터 git에 포함. `infos/`, `logs/`, `memtest_out/`, `validate_out/`, `cfg_cf3e02a/`, `old_100x100/`은 제외)
- `make_token_lists.py`, `cache_teacher_future.py`, `run_cache.sh`
- `future_index_{train,test}.json`, `future_{train,test}.yaml`, `validate_test.yaml`
- `counts.json`, `train_disk_availability.json`
- `logs/{infos_test,infos_train,validate,validate50,memtest,cache_gpu0,cache_gpu1}.log`
- `validate_out/`, `memtest_out/manifest.json`
- 부분 캐시: `/home/external-user/datasets/teacher_cache/bevfusion/cache_val_50x100_future` (1,422 npz, manifest 없음)
- 다운로드 도구 `xfer/` (§9, §10): `make_needed_files.py`, `stream_extract.sh`, `rstream.py`, `hf/`(tree·대응표), `tests/{flaky_server.py,test_a.sh,test_b_verify.py}`, `results/{needed_summaries/,stream_test_146/,stream_test_A.out}`

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
- ~~OpenScene archive 크기와 log 대응~~ (§9-1에서 해소)
- 새 서버의 대역폭, 디스크, GPU arch (대역폭·병렬 수는 §10-3 3–4번에서 잰다)
- 미래 프레임에서의 teacher 정확도
- "4 s 전부 확보 가능" 부분집합에서 sweep 누락을 뺀 실제 비율과 그 편향
- ReSMap 처리량, ReSMap memory 리셋 구간의 정확한 정의(§2-2의 구간 수)
- 전용 GPU에서의 BEVFusion 처리량(11.6 frame/s는 가정)과 CPU 디코딩 상한 69 frame/s의 근거
- 18.5 GB GPU 메모리의 원인
- `--drop-bev` npz 실제 크기 (멤버 크기에서 계산함. §9-8 메모: 기존 npz를 bev 없이 다시 써 보면 87,760 B, 차이 0.1%)

---

## 9. 받고 → 캐싱 → 지우기 운영안 (원본을 계속 보관하지 않는 경우)

작성 2026-10-03. 질문: "OpenScene을 받아 teacher 캐싱을 하고 원본을 지워도 되는가. 1 TB급을 계속 들고 있기는 너무 크다."

**근거**
- HF API tree(파일별 크기와 `lfs.oid`)와 공식 archive→log 대응표
- archive 0쌍(camera_0, lidar_0) 전체 member 목록
- 400개 archive 각각의 앞 2 MB를 zlib으로 디코드한 결과
- 옛 서버 npz 33,568개 전수 확인
- tar/gzip CPU 벤치
- 네트워크는 약 10.3 GB를 썼다. archive 바이트는 디스크에 저장하지 않았다. GPU와 잡은 쓰지 않았다.
- 검증 단계(같은 날): HF tree를 다시 받아 크기 합계를 재현했다. 집합별 run/DL/sweep, 추출 합, g별 최대치, 5 s 값, 전 프레임 값은 log pkl에서 독립 스크립트(`verify/v1.py`–`v3.py`)로 다시 계산해 모두 일치했다. 400/400 첫 member도 재확인했다. 고친 것은 버리는 비율, wget 재개 여부, P-full 합계, 8캠 크기, 반올림 몇 개다.

스크립트와 결과는 세션 scratchpad에서 만들었고, 2026-10-03에 `$REPO/tools/future_teacher_cache/xfer/`(= `$W/xfer/`, 약 7 MB)로 옮겼다. 이 폴더는 git에 포함돼 있다.
- `hf/{map_trainval.json,sizes.json,tree_*.json,info.json}`, `heads/`, `full/list_{camera,lidar}_0.txt`
- `rstream.py`(xfer 바로 아래), `scripts/{headparse,needed,perarc,listing,nodisk,b_keep}.py`, `scripts/taskC/`, `scripts/verify/`, `results/{needed_result,perarc_result,listing_result_0,archives_*_4s}.json`, `results/full/`
- `$W`의 스크립트·목록·`xfer/`는 git에 포함돼 있다(2026-10-03). 새 서버에서는 clone으로 받는다(§4 단계 0).

**시나리오**
- S1: 새 서버에 navtrain current+history 센서 패키지가 있다(39의 전제). 별도 표기가 없으면 S1 기준이다.
- S2: 새 서버에 trainval 센서가 하나도 없다.

### 9-0. 결론
- **(§10에서 확정)** 사용자는 "원본 2 TB는 저장하지 않고, 받으면서 캐싱에 필요한 파일만 풀어 보관하고 나머지는 버린다"로 정했다. 이 절의 P-stream-all에서 "삭제" 단계를 뺀 형태다. 그래서 아래의 "지운 뒤 재DL" 위험은 보관하지 않은 부분(5캠, 범위 밖 프레임, ReSMap 연속 구간)에만 해당한다. 보관 파일까지 지우기로 할 때만 §9-4 전체가 다시 삭제 조건이 된다. 아래 분석은 근거로 남긴다.
- **받고 → 필요한 파일만 풀고 → 캐싱 → 원본 삭제는 가능하다** [실측 근거 + 추론].
  - archive 400개가 log 단위로 정렬돼 있다. camera_i와 lidar_i에는 같은 log 6–7개가 들어 있고, log는 archive 경계를 넘지 않는다.
  - BEVFusion의 sweep(f−1, f−2)도 같은 log 안에 있다. 그래서 archive 쌍 하나만 있으면 그 log들의 추론이 끝난다.
  - 결과적으로 1 TB급 원본을 계속 들고 있을 필요가 없다. 원본이 동시에 디스크에 있는 양은 두 가지 방식에 따라 다르다.
    - 전부 푼 뒤 한 번에 캐싱(P-stream-all): E2E 4 s 159 GB
    - archive 묶음 단위로 처리하고 지움(P-chunked): 수–수십 GB
  - 캐싱 뒤 남는 것은 npz뿐이다. `--drop-bev`면 E2E 4 s 8.2 GB, navtrain 5 s 11.7 GB다.
- **전송량은 줄일 수 없다.** 어느 집합(stage-T, E2E, navtrain, 2–5 s)이든 400개 archive가 모두 필요하다. 2,124 GB이고, 받은 바이트의 약 90–95%는 곧바로 버린다(S1, 압축 기준: stage-T 4 s 95%, E2E 4 s 93%, navtrain 4 s 92%, navtrain 5 s 90%) [검증 단계 재계산].
- **지운 뒤에 원본이 다시 필요해지면 2,124 GB를 다시 받아야 한다.** 해당하는 경우는 다음과 같다.
  - teacher나 config를 바꿀 때
  - bev feature를 저장하지 않았는데 F3/P4가 필요해질 때
  - ReSMap 미래나 학생 쪽 미래 카메라를 쓸 때
  - 8캠이 필요할 때
  - 그래서 삭제 전에 범위, horizon, bev 저장, 보존할 카메라를 확정한다(§9-4).

### 9-1. archive 구성 [실측]
- **크기 (HF API, decimal)**
  - camera 200개 1,242.21 GB (min 1.77 / median 5.30 / max 19.30 GB)
  - lidar 200개 882.14 GB (min 1.23 / median 3.79 / max 14.46 GB)
  - **합계 2,124.35 GB**
  - archive 크기는 그 archive에 든 log의 프레임 수에 비례한다(상관 0.997 / 0.9995). 압축 기준으로 camera(8캠) 1.718 MB/frame, lidar 1.220 MB/frame이다.
  - archive별 "tgz 크기 ÷ 대응표상 프레임 수"는 중앙값 대비 camera 0.90–1.11, lidar 0.95–1.04 안에 있다 [검증 단계 재계산]. archive 평균 프레임은 약 3,600, log 평균은 약 550이라 log 하나가 통째로 다른 archive에 들어 있으면 lidar 비율이 약 ±15% 벗어나야 한다. 그래서 "대응표 = 실제 내용"을 뒷받침한다. 다만 약 150프레임 미만의 짧은 log나 log 일부가 넘어가는 경우는 이 방법으로 배제되지 않는다.
  - archive 쌍 하나는 평균 10.6 GB, 최대 33.8 GB(#124)다.
- **revision 고정**
  - dataset sha는 `a76f840b65e972bc45e56c2adced897498e9a026`(lastModified 2025-04-28)이다.
  - URL은 `/resolve/main/` 대신 `/resolve/<sha>/`를 쓴다. 이번 측정도 이 URL로 했다.
- **archive → log 대응표:** `openscene-v1.1/openscene_sensor_trainval_0-199.json`(68,352 B)
  - 키 200개, log 1,310개, 중복 없음. archive당 log 수는 6개가 90개, 7개가 110개다.
  - 이 log 집합은 로컬 `trainval_navsim_logs/trainval/*.pkl` 1,310개와 정확히 같다.
- **member 경로:** `openscene-v1.1/sensor_blobs/trainval/<log>/{CAM_F0..CAM_B0|MergedPointCloud}/<token>.{jpg|pcd}`
  - `--strip-components=2 -C $DL/trainval_sensor_blobs`로 풀면 기존 구조와 맞는다.
  - archive 0에서 로컬에 이미 있는 파일(camera 5,680개, lidar 710개)과 크기를 대조했고 모두 같았다.
- **log 정렬**
  - archive 0쌍 전체: camera 22,360 = 2,795 frame × 8캠, lidar 2,795개다. 대응표의 7 log와 같고, log pkl과 대조한 결과 missing 0, extra 0이다.
  - 400/400개 archive의 첫 member가 대응표의 첫 log와 같았다.
  - "log가 archive 경계를 넘지 않는다"는 판단은 대응표와 archive 0 전체 목록에 근거한다. 나머지 399개는 끝까지 보지 않았다 [추론].
- **풀었을 때 크기 (archive 0 member 평균)**
  - CAM_F0 220,578 B, L0 201,296 B, R0 203,852 B, pcd 1,403,088 B
  - 프레임당: F0/L0/R0+lidar **2.029 MB**, 8캠+lidar 3.091 MB, 3캠만 0.626 MB. archive 0의 7 log 값이다.
  - 검증 단계에서 로컬 navtrain 센서 80 log 표본(10,594 frame)으로 다시 재면 2.006 / 3.086 / 0.613 MB다. §2-4의 2.023 / 3.116 MB와 함께 어느 값이든 1–2% 안이다. §9의 합계는 2.029 / 0.626 MB 기준이며 ±2% 오차로 본다.
  - 풀 때 늘어나는 비율(archive 0): camera 1.003, lidar 1.134. trainval 전체를 풀면 약 2.24–2.25 TB다 [추정].

**집합별 필요 archive와 전송량** (§2-1 정의를 그대로 재계산해 §2-1 값을 재현했다)

| 집합 | 실행 | DL 프레임 | 필요 archive (camera / lidar) | 전송 | DL / 전체 프레임 | archive 하나에서 나오는 최대 DL 프레임 |
|---|---:|---:|---|---:|---:|---:|
| stage-T 4 s | 62,629 | 52,528 | 200 / 200 | 2,124 GB | 7.3% | 939 |
| E2E 4 s | 93,606 | 78,543 | 200 / 200 | 2,124 GB | 10.9% (archive별 median 10.7%, 1.2–19.2%) | 1,431 |
| navtrain 4 s | 112,152 | 93,670 | 200 / 200 | 2,124 GB | 13.0% | 1,588 |
| navtrain 5 s | 133,113 | 112,007 | 200 / 200 | 2,124 GB | – | – |
| 2 s (모든 집합) | – | – | 200 / 200 | 2,124 GB | – | – |

- stage-T 4 s도 log 972개가 200개 archive 전부에 퍼져 있다. 필요 프레임이 0인 archive는 없다. **그래서 B′(필요한 log가 든 archive만 받기)는 절약이 0이다.**
- sweep 때문에 실행 집합 밖에서 추가로 받을 프레임은 S1에서 0이다(4 s, 5 s 모두).
- S2에서는 sweep용 lidar를 추가로 받아야 한다. stage-T 23,008 / E2E 28,623 / navtrain 34,447 프레임이다.

**전송 안정성 (이 서버에서 잼, 새 서버 값 아님)**
- curl 단일 HTTP/2 스트림은 중간에 끊겼다(exit 92). camera는 330 MB/297 s 뒤, lidar는 570 MB/943 s 뒤였다.
  - 그래서 `curl | tar`처럼 이어받기가 없는 파이프는 2 TB 규모에서 거의 확실히 끊긴다 [추론].
  - `wget -O- | tar`는 HF에서 시험하지 않았다. wget은 같은 실행 안에서 연결이 끊기면 `Range`로 이어받는 것이 기본 동작이다(기본 `--tries=20`). 검증 단계에서 7 MB마다 연결을 끊는 로컬 HTTP 서버로 `wget -O-`를 돌렸고, `bytes=7000000-`, `bytes=14000000-`로 다시 붙어 sha256이 원본과 같았다 [실측, 로컬]. 그러니 §4 단계 2의 `wget` 루프가 "이어받기가 없어 반드시 끊긴다"고 볼 근거는 없다. 쓴다면 `--tries=0 --read-timeout=60 --waitretry=30`을 준다. rstream의 장점은 재시도 로그와 HTTP/1.1 경로가 이번에 HF에서 끝까지 돌았다는 점이다. 다만 아래 측정은 재시도 0이라 rstream의 재개 경로 자체는 HF에서 한 번도 쓰이지 않았다. HF CDN이 206을 준다는 것만 확인했다.
- `rstream.py`(HTTP/1.1, 끊기면 `Range: bytes=<offset>-`로 다시 붙어 같은 stdin으로 계속 보냄)로 받은 결과:
  - camera_0: 4.70 GB / 1,710 s = 2.75 MB/s
  - lidar_0: 3.46 GB / 2,821 s = 1.23 MB/s
  - 재시도 0, tar exit 0. 두 스트림을 동시에 받아 합산 약 4 MB/s였다.
- resolve URL은 302로 `us.aws.cdn.hf.co/xet-bridge-us/...`로 넘어가고, 거기서 206(Range)을 지원한다.
- 2 TB 규모에서 재개 방식을 검증하지는 않았다.

### 9-2. 파이프라인 선택지와 디스크·시간

**정의**
- **P-full:** `download_trainval.sh`를 그대로 실행해 전부 받고 전부 푼다.
- **P-stream-all:** 400개를 스트리밍하면서 필요한 파일만 푼다. 다 푼 뒤 단계 5–7을 한 번에 돌리고, 검증한 다음 지운다. 지금 §4 단계 2–7과 같은 구조다.
- **P-chunked(g):** archive 번호를 g쌍씩 묶는다. 묶음마다 받기 → 풀기 → infos → 추론 → 검증 → 지우기를 한다.
  - 직렬: 묶음을 차례로 처리한다.
  - 겹침: 묶음 k+1을 받는 동안 묶음 k를 추론한다. 디스크 최대치는 연속한 두 묶음의 합이다.

**원시 센서가 동시에 디스크에 있는 양 (decimal GB, npz 제외)** [추정 = 실측 프레임 수 × 실측 파일 크기, archive별 tgz 크기로 보정]

| 집합 | 시나리오 | P-stream-all | g=1 직렬 / 겹침 | g=10 직렬 / 겹침 | g=40 직렬 / 겹침 |
|---|---|---:|---:|---:|---:|
| stage-T 4 s | S1 | 106.6 | 1.9 / 3.0 | 7.6 / 13.6 | 24.1 / 46.1 |
| | S2 | 159.3 | 2.8 / 4.3 | 11.0 / 20.1 | 35.7 / 68.3 |
| E2E 4 s | S1 | 159.3 | 2.9 / 4.5 | 11.0 / 20.2 | 35.7 / 68.1 |
| | S2 | 230.1 | 4.2 / 6.3 | 15.7 / 29.2 | 51.1 / 97.7 |
| navtrain 4 s | S1 | 190.0 | 3.2 / 4.6 | 12.7 / 22.4 | 39.8 / 78.5 |
| | S2 | 275.9 | 4.7 / 6.6 | 18.3 / 32.6 | 57.2 / 113.3 |
| navtrain 5 s | S1 | 227.2 | 3.9 / 5.5 | 15.2 / 26.8 | 47.7 / 94.0 |
| trainval 전 프레임 | S1 | 1,157.5 | 18.5 / 30.9 | 72.6 / 139.5 | 241.7 / 475.1 |

- P-full은 전부 푼 2,246 GB에 tgz 하나(최대 19.3 GB)가 더해진다. 기존 `trainval_sensor_blobs`에 `--skip-old-files`로 합쳐도 약 1.78 TB가 새로 늘어난다.

**비교 (E2E 4 s, S1, 전용 GPU 1장 11.6 frame/s 가정)**

| 파이프라인 | 원시 센서 최대 | 삭제 후 남는 것 | 새 서버 합계 (기준선 1.51 TB, §9-7) | wall-clock 10 / 50 / 100 MB/s | 새 코드 | 주된 위험 |
|---|---:|---|---:|---|---|---|
| P-full | 약 2,265 GB | (원본을 지우면) npz | 약 3.3 TB(기존 navtrain 센서에 합침) – 3.8 TB(따로 풂) | 61 / 14 / 8 h | 없음 | 디스크. `mv` 중첩(§4 단계 2-3). 필요 없는 약 2.1 TB를 풂 |
| **P-stream-all** | 159 GB (S2 230) | npz 8.2 GB (bev 포함 248) | 약 1.68 TB (bev 포함 1.91) | 61 / 14 / 8 h (직렬) | archive별 needed 목록, rstream, lidar·sweep 존재 검사 [작음] | 전부 받을 때까지 추론을 시작할 수 없음. 중간 실패 시 재시작 관리 |
| **P-chunked g=10 겹침** | 20.2 GB (S2 29.2) | 같음 | 약 1.54 TB (1.78) | 59 / 12 / 6 h | 위에 더해 묶음 루프, 묶음별 infos·manifest, 병합, backpressure [중간] | 코드 버그가 묶음마다 반복됨. manifest 덮어쓰기(§4 단계 6) |
| P-chunked g=1 겹침 | 4.5 GB (S2 6.3) | 같음 | 약 1.52 TB | 59 / 12 / 6 h (+ torchpack 시작 약 10 s × 200) | 같음 | 같음. 묶음이 200개라 관리가 번거로움 |

**시간** [추정. 다운로드는 실측 크기 ÷ 가정 대역폭]
- **다운로드 2,124 GB:** 10 MB/s 59 h, 50 MB/s 11.8 h, 100 MB/s 5.9 h, 300 MB/s 2.0 h. 이 서버의 약 4 MB/s면 148 h다.
- **추론 (전용 1장 / 공유 1장 / 전용 4장):**
  - E2E 4 s: 2.24 / 4.48 / 0.56 h
  - navtrain 5 s: 3.19 / 6.38 / 0.80 h
  - trainval 전 프레임: 14.8 / 29.7 / 3.7 h
- **풀기 CPU** [실측 벤치]: 압축 해제는 병목이 아니다.
  - `gzip -dc` 216 MB/s/core(압축 기준), `pigz -dc` 474 MB/s, python `tarfile` 스트림 + set 조회 474 MB/s
  - `tar -T` 이름 6,000개 208 MB/s, 300,000개 94 MB/s. 그래서 이름 목록은 archive별로 나눈다.
  - 전체 2.73 core-h다.
- **병목이 다운로드에서 추론으로 바뀌는 대역폭:**
  - E2E 4 s: 전용 1장 263 MB/s, 공유 1장 132 MB/s, 전용 4장 1,053 MB/s
  - trainval 전 프레임: 전용 1장 40 MB/s, 전용 4장 159 MB/s
  - 이보다 느린 대역폭에서는 겹침 방식의 wall-clock이 다운로드 시간과 거의 같다.

### 9-3. 한 번 받을 때 어디까지 뽑을까 (전송은 모두 2,124 GB로 같음)

| 범위 | 실행 frame | GPU-h (전용 1 / 공유 1 / 전용 4) | npz drop / bev (GB) | S1 추출 합 (GB) | S1 g=10 겹침 최대 (GB) | 현실적인 파이프라인 |
|---|---:|---|---:|---:|---:|---|
| E2E 4 s | 93,606 | 2.24 / 4.48 / 0.56 | 8.2 / 248 | 159 | 20 | stream-all 또는 chunked |
| navtrain 4 s | 112,152 | 2.69 / 5.37 / 0.67 | 9.9 / 297 | 190 | 22 | stream-all 또는 chunked |
| navtrain 5 s | 133,113 | 3.19 / 6.38 / 0.80 | 11.7 / 352 | 227 | 27 | stream-all 또는 chunked |
| navtrain 1,192 log 전 프레임 | 566,300 | 13.6 / 27.1 / 3.4 | 49.8 / 1,500 | 1,049 | 124 | 2 TB 디스크면 chunked만 (stream-all은 약 2.6 TB 필요) |
| trainval 1,310 log 전 프레임 | 619,731 | 14.8 / 29.7 / 3.7 | 54.5 / 1,641 | 1,158 | 140 | 2 TB 디스크면 chunked만 (stream-all은 약 2.7 TB 필요) |

- 전 프레임 행의 "실행 frame"은 이미 캐시된 navtrain 현재 프레임(103,288)을 뺀 값이다. log의 전체 프레임은 navtrain log 669,588, trainval 723,019다. 추출 합에는 sweep용으로 캐시된 프레임의 lidar도 들어 있다.

**E2E 4 s에서 navtrain 5 s로 넓힐 때의 추가 비용** [추론]
- 비용: 일시 디스크 +68 GB, npz +3.5 GB, GPU 약 +1.0 h(0.95 h, 전용 1장)
- 얻는 것:
  - dev(val_logs) token의 R_F 평가
  - GT 벌점과 같은 0–5 s 범위

**trainval 전 프레임을 `--drop-bev`로 캐싱할 때** [추론]
- 얻는 것: BEVFusion 박스, score, 속도를 어떤 horizon(log 끝까지), 어떤 간격, 어떤 token 부분집합, val log 118개에 대해서도 다시 받지 않고 쓸 수 있다.
- 비용: npz +46 GB, GPU 전용 1장 기준 약 15 h
  - 대역폭이 40 MB/s 이하면 GPU가 어차피 다운로드를 기다리므로 wall-clock은 거의 늘지 않는다.
  - P-chunked 코드가 필요하다.
  - 전 프레임 infos는 약 10 GB pkl로 추정된다. 그래서 묶음별로 나눈다. rank별 RAM은 재지 않았다.
- 전 프레임에 bev까지 저장하면 1.64 TB라 비현실적이다. bev는 E2E 4 s(248 GB)나 navtrain 5 s(352 GB)에만 저장하는 선택지가 있다.

### 9-4. 지우기 전에 반드시 확정·검증할 것 (순서대로)
괄호 안은 건너뛰었을 때의 비용이다. "재DL"은 2,124 GB를 다시 받는 것이다. camera만 필요하면 1,242 GB, lidar만이면 882 GB다. archive 쌍 하나만 다시 받으면 되는 경우는 평균 약 10.6 GB, 최대 33.8 GB다(g=10 묶음이면 약 106 GB).

1. **결정을 확정한다** (§7-2 2·3·10): 집합(E2E / navtrain / 전 프레임), horizon(4 / 5 s), 보존할 카메라(3캠 / 8캠), `--drop-bev`, 돌릴 teacher·config 목록, ReSMap 미래 여부, 학생 쪽 미래 사용 여부. (잘못 정하면 재DL)
2. **출처를 고정한다.**
   - URL을 `/resolve/<sha>/`로 고정한다.
   - tree JSON 2개와 대응표를 보존한다.
   - archive마다 스트림 sha256이 `lfs.oid`와 같은지 확인한다.
   - member 목록과 wget/rstream·gzip·tar의 종료 코드를 남긴다.
   - (빠뜨리면 같은 데이터를 다시 받을 수 있다는 보장이 없다. 하는 비용은 0이다)
3. **출처가 같은지 확인한다.** 이미 디스크에 있는 navtrain current/history 파일 수백 개를 다운로드 쪽에서 별도 폴더로 풀어 `cmp`한다 [제안, 실행 안 함]. (지운 뒤에 불일치를 발견하면 재DL)
4. **풀어 낸 부분집합을 검사한다.**
   - 대상 프레임 전부에 F0/L0/R0 `.jpg`, `.pcd`, sweep 2개 `.pcd`가 있어야 한다(누락 0).
   - 프레임당 크기가 약 2.03 MB여야 한다.
   - archive별 tar err에 "Not found in archive"가 0개여야 한다.
   - (누락이 있으면 그 archive 쌍만 다시 받으면 된다)
5. **infos를 만든다.**
   - 단계 5를 `--check-files`로 돌리고, 3캠만 풀었으면 `--cameras CAM_F0 CAM_L0 CAM_R0`을 준다.
   - lidar·sweep 존재를 사전 검사한다 [새 코드].
   - infos의 sha256을 남긴다.
   - (infos가 없으면 추론이 불가능하다. lidar·sweep 경로는 디스크를 보지 않고 만들어지므로 경로 오류는 삭제 전에 잡아야 한다)
6. **§6-2 재현을 통과시킨다.** navtest로 하므로 삭제와 무관하지만, 단계 6보다 먼저 한다. (env 오류를 늦게 발견하면 재DL과 재추론)
7. **추론을 돌린다.**
   - 원본이 있는 동안 필요한 teacher와 config 변형을 **모두** 돌린다.
   - teacher마다 cache 폴더를 따로 쓴다. `--skip-existing`은 경로만 보기 때문이다.
   - 묶음마다 manifest 사본을 남긴다.
   - (빠뜨리면 재DL)
8. **단계 7 완결성을 확인한다.**
   - npz 수가 infos 수와 같고, 차집합이 ∅이고, `.tmp.npz`가 0이어야 한다.
   - sha head `cddf943ffec8d6a8`과 `bev_feature_stored`를 확인한다.
   - **모든 npz를 `np.load`로 연다.**
   - (구멍이 있으면 해당 archive 쌍만 다시 받으면 된다)
9. **§6-4 표본 검사와 눈으로 하는 QA를 한다.**
   - 좌표: in-sample 오차가 cm 수준이어야 한다. 수 m면 infos나 sweep 버그다.
   - out-of-sample 재현율
   - 수십 프레임에서 박스를 영상에 투영해 본다.
   - (infos 버그를 늦게 발견하면 재DL)
10. **golden 샘플을 떼어 둔다** [제안]: 64 token × 미래 8 = 512 frame, 8캠+lidar와 bev를 넣은 기준 npz, 약 3 GB. 서로 다른 log·도시에서 뽑고, 다운로드 프레임과 token이 아닌 프레임을 포함한다. (빠뜨리면 새 env나 새 teacher의 회귀 확인을 할 수 없다)
11. **보존하기로 한 카메라를 옮긴다**(§9-5). 예: ReSMap 미래나 학생 미래용 3캠. (빠뜨리면 camera 재DL 1,242 GB)
12. **provenance 묶음(§9-6)을 저장한 뒤에 지운다.**

### 9-5. 지운 뒤 불가능해지는 것과 대비책

**npz에 남는 것** [실측, `cache_teacher_future.py:256-268`, npz 33,568개 전수 확인]
- `pred_boxes_3d` (200, 9): 그 프레임 ego 좌표, z는 bottom, yaw와 vx·vy 포함
- `pred_scores_3d`, `pred_labels_3d`
- `dense_heatmap` (7, 50, 100) f16: `--drop-bev`여도 저장된다
- `raw_*` (fp16)
- `bev_feature` (256, 50, 100) f16: `--drop-bev`면 빠진다
- 크기는 `--drop-bev` 87,886 B, bev 포함 2,648,014 B다.
- 저장할 때 score 임계와 NMS는 걸지 않았다(`score_threshold 0.0`은 falsy, `nms_type null`).
- ego pose와 timestamp는 npz에 없다. log pkl과 `future_index_*.json`에서 가져오며, 둘 다 센서가 아니므로 계속 남는다.

**지운 뒤에도 캐시만으로 바꿀 수 있는 것** [실측과 추론]
- score 임계, 사후 circle NMS, 클래스 부분집합과 클래스별 임계
- t0 변환(yaw·속도 회전 포함), raster 해상도와 채널, O(t,s)
- "모름" mask, horizon 축소(≤ 캐싱한 horizon)와 k 부분집합
- heatmap 기반 F1(0.64 m 해상도)

**캐싱 시점에 동결되는 것**
- checkpoint와 dirty-tree config
- ROI(x 0..32 m, y ±32 m)와 그에 따른 사각지대
- 카메라 F0/L0/R0, `sweeps_num 2`, `reduce_beams 32`
- 영상 256×704, top-200 query, bev 저장 여부
- **실행한 프레임 집합**

| 용도 | 캐시로 충분한가 | 남겨 둘 최소 파일 | 크기: 4 s E2E / navtrain (5 s E2E / navtrain) | 지운 뒤 다시 하려면 |
|---|---|---|---|---|
| 박스 계열: O_T+, F1, F2, F4, P2 R_F, P5-teacher, §6-4, §6-5 | 예 | npz `--drop-bev` | 8.2 / 9.9 (9.8 / 11.7) GB | – |
| BEV feature 계열: F3, P4 R_T+, H5 | bev를 저장했을 때만 | npz bev 포함 | 248 / 297 (295 / 352) GB | 재DL 2,124 GB |
| (위의 대안) 원본 최소 입력 보존 | – | F0/L0/R0 + pcd | 159 / 190 (191 / 227) GB | – |
| ReSMap 미래 (F5, R_M+) | 아니오 | 다운로드 프레임의 전방 3캠 | 49.2 / 58.6 (58.8 / 70.1) GB | camera 재DL 1,242 GB |
| (같음, log 전체를 빈틈없이 돌리는 경우) | 아니오 | 디스크 밖 모든 프레임의 3캠 | navtrain log 324, trainval 357 GB | camera 재DL |
| 다른 BEVFusion (재학습, 100×100, lidar-only, sweep 수 변경) | 아니오 | F0/L0/R0 + pcd | 159 / 190 GB | 재DL 2,124 GB |
| 8캠·360° teacher | 아니오 | 8캠 + pcd | 243 / 290 GB (3.091 MB 기준) | 재DL |
| 학생 쪽 자기 미래 latent (devkit `use_fut_frames`) | 아니오(학생 가중치가 바뀜) | 3캠, k=1이면 디스크 밖 프레임만 | k=1 7.3 / 8.8 GB, 10프레임이면 58.8 / 70.1 GB | camera 재DL |
| 회귀 확인과 눈으로 하는 QA | – | golden 샘플 | 약 3 GB | 해당 log의 archive 쌍(약 10.6 GB씩) |

- **BEV 계열을 할 거라면 bev npz 대신 원본 최소 입력을 남기는 편이 낫다** [추론]. bev를 넣은 npz(E2E 4 s 248 GB)가 원본 최소 입력(F0/L0/R0 + pcd, 159 GB)보다 크다. 최소 입력을 남기면 더 작고, bev는 필요할 때 다시 추론하면 되며(전용 1장 약 2.2 h), 다른 teacher를 돌릴 선택지도 남는다. 나머지 원본(5캠 등 약 2 TB 분량)은 그래도 지운다.
- **ReSMap 보존 크기는 두 값이 있다** [추론, 미확인]. 49 GB와 324–357 GB이고, 어느 쪽인지는 memory 구간 정의에 달려 있다.
  - 49 GB: 기존 ReSMap root처럼 "센서가 있는 프레임"만 이어 돌리는 경우다. 이미 디스크에 있는 프레임에 다운로드 프레임 3캠을 더하면 된다.
  - 324–357 GB: log를 2 Hz로 빈틈없이 돌려야 하는 경우다.
  - 실제 리셋 기준은 kyungmin map infos의 `local_idx == 0`이다(§2-2). 이 기준이 어느 쪽인지는 확인하지 않았다. 삭제 전에 정한다.

### 9-6. 재다운로드 재현성을 위해 기록할 것
- **출처**
  - HF revision sha `a76f840b65e972bc45e56c2adced897498e9a026`
  - `tree_{camera,lidar}.json`(크기와 `lfs.oid`)
  - `openscene_sensor_trainval_0-199.json`
- **archive별 기록**
  - 스트림 sha256(= `lfs.oid` 확인 결과)
  - `tar -v` member 목록. 실제로 푼 파일 목록이 되며 삭제 목록으로도 쓴다.
  - rstream 재시도 로그, PIPESTATUS, tar err
- **입력**
  - `needed_{camera,lidar}_<i>.txt`
  - 풀어 낸 파일의 크기 목록. golden 샘플은 sha256까지 남긴다.
  - log pkl sha256
  - `future_index_*.json`, `future_*.yaml`(걸러 낸 yaml과 묶음별 yaml 포함), `counts.json`
  - `infos/future_*.pkl`과 그 sha256
- **teacher**
  - checkpoint 전체 sha256 `6ad75eca…a904`. manifest에는 1 MiB head만 들어간다.
  - bevfusion `cf3e02a` + dirty tree의 `git diff` patch와 untracked 파일의 sha. manifest의 `git_commit`은 dirty 여부를 기록하지 않는다.
  - resolved config dump, torch/mmcv/CUDA/GPU/driver, cudnn 플래그, world_size, 명령줄
- **결과물**
  - 묶음별 manifest 사본과 병합 manifest
  - npz 목록과 크기
- **기존 manifest 오기 3건** (데이터에는 영향 없음, 고칠 때 함께 기록)
  - z는 "gravity centre"로 적혀 있지만 실제로는 bottom이다(`:340`).
  - raw_center 설명의 "add query position"(`:338`)은 틀렸다. raw_center는 이미 절대 cell 좌표다.
  - `student_bev_shape_para_ssr: [100,100]`(`:329`)은 지금 50×100이다.

### 9-7. 새 서버 디스크 크기
**기준선** (39의 전제, 이 서버 `du -B1` 할당 바이트) [실측]

| 항목 | GB |
|---|---:|
| NAVSIM (maps 1.4, trainval logs 14.3, trainval sensor 477.9, test logs 1.0, test sensor 234.7) | 729.3 |
| teacher cache (bevfusion train 273.7 + val 32.2, resmap 377.0) | 683.0 |
| yongjae_refiner 72.5, navtest metric cache 3.3, env·repo 약 19.5 | 95.3 |
| **합계** | **약 1,507** |

- `bevfusion.tar.zst`(155.6 GB)로 teacher cache를 옮기면 그동안 +156 GB가 일시적으로 더 필요하다.
- ReSMap 캐시(377 GB)를 새 서버에 두지 않으면 그만큼 줄어든다.

**권장 범위** [추론, 결정은 사용자]

| 운영 | 최소 | 권장 |
|---|---:|---:|
| P-chunked + `--drop-bev` (E2E·navtrain·전 프레임 모두) | 1.54–1.70 TB | **약 2 TB** |
| P-stream-all + `--drop-bev` (E2E 4 s / navtrain 5 s) | 1.68 / 1.75 TB | **약 2 TB** |
| E2E 4 s bev 포함 | 1.78–1.91 TB | 2–2.5 TB |
| navtrain 5 s bev 포함, stream-all | 약 2.09 TB | 2.5 TB |
| P-full (원본 보관) | 약 3.3 TB(기존 navtrain 센서에 합침) – 3.8 TB(따로 풂) | 4 TB 이상 |

- 위 값은 일시 공간(`bevfusion.tar.zst` 156 GB)을 뺀 것이다. 넣으면 각 행에 더한다.
- **S2(새 서버에 navtrain 센서 패키지를 아직 받지 않은 경우)** [추론]
  - 같은 OpenScene 스트림에서 navtrain current/history 파일까지 풀어 영구 보관한다.
  - 그러면 navtrain 패키지(449 GB tgz)를 따로 받지 않아도 된다. 전송이 449 GB 줄고, 남는 센서 약 471 GB는 같다.
  - 근거: archive 0에서 겹치는 파일의 크기가 모두 같았다. 내용 일치는 9-4 3번으로 확인한다.

### 9-8. 명령 스케치 [제안; `make_needed_files.py`, `check_group.py`, `validate_group.py`는 새 코드]

**(§10에서 확정) 실제 실행은 §10-3을 따른다.** 이 스케치는 기록용으로 남긴다. 상태는 다음과 같다.
- (1) `make_needed_files.py`: 구현·시험됨. 인자는 아래 스케치(`--set`)와 다르다(`--tokens … --horizon-s …`, §10-3).
- (2) `fetch()`: `stream_extract.sh`로 대체됐다. 바뀐 점: staging에 풀고 검사 뒤 hard link로 옮김(`--skip-old-files` 대신), sha256은 `rstream.py`가 직접 계산(`tee >(sha256sum)` 대신), archive별 상태 JSON과 재실행 시 완료분 건너뛰기.
- tar 함정 하나를 구현 중에 확인했다: `-C`가 `-T`보다 뒤에 오면 tar가 오류 없이 현재 폴더에 푼다. 아래 스케치와 구현은 모두 `-C`를 앞에 둔다.
- (3a)의 `check_group.py --all`: archive별 검사는 `stream_extract.sh`에 들어갔고, 전역 검사는 `make_needed_files.py` 재실행으로 한다(§10-3 7번).
- (3b)·(5) P-chunked, (4) `delete_group`, `validate_group.py`: 원본을 지우지 않기로 했으므로 구현하지 않았다. 보관 파일을 나중에 지우기로 하면 그때 쓴다 [새 코드].

`rstream.py`는 `$W/xfer/rstream.py`에 있다. 변수는 §4를 따른다.

```bash
REV=a76f840b65e972bc45e56c2adced897498e9a026
U=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/$REV/openscene-v1.1
API=https://huggingface.co/api/datasets/OpenDriveLab/OpenScene/tree/$REV/openscene-v1.1
X=$W/xfer; mkdir -p $X/{lists,sha,members,logs,manifests,yaml}; touch $X/keep.txt   # keep.txt: 보존할 파일의 절대경로
SB=$DL/trainval_sensor_blobs

# (0) 출처 고정: tree(크기, lfs.oid=sha256)와 공식 대응표
for m in camera lidar; do curl -s $API/openscene_sensor_trainval_$m > $X/tree_$m.json; done
curl -sL $U/openscene_sensor_trainval_0-199.json > $X/map_trainval.json
find $SB/trainval -type f | sort > $X/preexisting.txt          # 삭제 가드: 원래 있던 파일 스냅샷

# (1) archive별 필요 파일 목록 [새 코드]
#   입력: 집합 yaml(단계 3에서 걸러 낸 것), log pkl, map_trainval.json, preexisting.txt
#   출력: $X/lists/{camera,lidar}_$i.txt (archive member 경로, 디스크에 없는 것만, sweep f-1/f-2 포함)
#         $X/yaml/g$g.yaml (묶음 g의 token만, chunked일 때)
$PY $W/make_needed_files.py --set e2e_4s --cams CAM_F0 CAM_L0 CAM_R0 --map $X/map_trainval.json --out $X

# (2) archive 하나를 스트리밍해 필요한 파일만 푼다 (archive 바이트는 디스크에 쓰지 않음)
fetch() {  # $1=camera|lidar  $2=i
  local f=openscene_sensor_trainval_$1_$2.tgz
  local n=$(jq -r --arg f "$f" '.[]|select(.path|endswith($f)).size' $X/tree_$1.json)
  local oid=$(jq -r --arg f "$f" '.[]|select(.path|endswith($f)).lfs.oid' $X/tree_$1.json)
  $PY $X/rstream.py $U/openscene_sensor_trainval_$1/$f $n 2> $X/logs/rs_$1_$2.err \
    | tee >(sha256sum | cut -d' ' -f1 > $X/sha/$1_$2.sha) \
    | tar -xzv --skip-old-files --strip-components=2 -C $SB -T $X/lists/$1_$2.txt \
        > $X/members/$1_$2.txt 2> $X/logs/tar_$1_$2.err
  echo "$1 $2 ${PIPESTATUS[*]}" >> $X/logs/status.txt
  wait $!   # bash ≥ 4.4: 마지막 process substitution(sha256sum)이 끝나기를 기다림
  [ "$(cat $X/sha/$1_$2.sha)" = "$oid" ] || echo "SHA_MISMATCH $1 $2" >> $X/logs/status.txt
}
# - GNU tar는 --occurrence가 없으면 목록을 다 찾아도 archive 끝까지 읽는다. 그래서 sha가 전체 스트림 기준이 된다.
#   tar가 먼저 죽으면 tee가 SIGPIPE를 받고 sha가 틀어진다. 이것도 실패로 잡힌다.
# - 목록이 archive별이므로 tar exit 2 / "Not found in archive"는 정상이 아니라 진짜 누락이다.
# - tar -v 출력이 strip 전 member 이름인지, 출력 위치가 stdout인지 archive 0으로 먼저 확인한다.

# (3a) P-stream-all: 전부 받은 뒤 §4 단계 5-7을 한 번에 실행
for i in $(seq 0 199); do echo $i; done | xargs -P 4 -I{} bash -c 'fetch camera {}; fetch lidar {}'   # 함수는 export -f 해 둔다
$PY $W/check_group.py --all      # 누락 0(F0/L0/R0, pcd, sweep 2), 약 2.03 MB/frame, SHA_MISMATCH 0, Not found 0, PIPESTATUS 0

# (3b) P-chunked(g, 겹침): 함수 정의(run_group, delete_group) 뒤의 (5) 루프로 돌린다
G=10

run_group() {  # $1=g
  $PY $W/check_group.py --group $1 || return 1
  ( cd $BF && CUDA_VISIBLE_DEVICES="" $BPY tools/data_converter/navsim_converter.py \
      --navsim-logs $DL/trainval_navsim_logs/trainval --sensor-root $SB --split trainval \
      --scene-filter $X/yaml/g$1.yaml --keyframes-only --check-files --cameras CAM_F0 CAM_L0 CAM_R0 \
      --workers 8 --lidar-prefix lidar --lidar-ext .pcd --max-sweeps 2 --out $W/infos/future_train_g$1.pkl ) || return 1
  ( cd $BF && torchpack dist-run -np $NGPU python $W/cache_teacher_future.py \
      configs/navsim/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml runs/navsim-fusion-50x100/epoch_20.pth \
      --cache-dir $TC/bevfusion/cache_train_50x100_future --split train \
      --ann-file $W/infos/future_train_g$1.pkl --skip-existing --drop-bev ) || return 1
  cp $TC/bevfusion/cache_train_50x100_future/manifest.json $X/manifests/g$1.json   # 매 실행이 manifest를 덮어씀
  $PY $W/validate_group.py --group $1 && delete_group $1     # npz 수 = infos 수, 차집합 ∅, .tmp 0, np.load 전수, sha head
}

# (4) 삭제: 검증을 통과한 묶음만, 실제로 푼 파일만, 원래 있던 파일과 보존 목록은 제외
delete_group() {  # $1=g  (P-stream-all이면 모든 i를 한 번에)
  for i in $(seq $1 $(($1+G-1))); do cat $X/members/{camera,lidar}_$i.txt; done \
    | sed "s#^openscene-v1.1/sensor_blobs/#$SB/#" | sort -u > $X/del_g$1.txt
  if [ -n "$(comm -12 $X/del_g$1.txt $X/preexisting.txt | head -1)" ]; then echo "ABORT g$1: preexisting"; return 1; fi
  grep -vxFf $X/keep.txt $X/del_g$1.txt > $X/del_g$1.final || true   # keep.txt: golden, 보존할 3캠 등
  [ -s $X/del_g$1.final ] && xargs -d '\n' rm -f -- < $X/del_g$1.final
  echo "deleted g$1 $(wc -l < $X/del_g$1.final)" >> $X/logs/delete.txt
}

# (5) P-chunked 루프 (겹침): 묶음 g+1을 받는 동안 묶음 g를 추론한다 (backpressure: 풀어 둔 묶음은 최대 2개)
PREV=
for g in $(seq 0 $G 199); do
  ( for i in $(seq $g $((g+G-1))); do fetch camera $i & fetch lidar $i & done; wait ) &   # 다음 묶음 받기
  FETCH=$!
  [ -n "$PREV" ] && { run_group $PREV || echo "FAIL g$PREV" >> $X/logs/status.txt; }   # 실패한 묶음은 지우지 않고 남김
  wait $FETCH; PREV=$g
done; run_group $PREV
```

- **병렬:** archive 번호별로 스트림을 여러 개 띄운다. HF xet의 스트림당·IP당 제한은 모른다. 새 서버에서 archive 한 쌍과 앞 2 MB 읽기 몇 개로 대역폭을 먼저 재고 병렬 수를 정한다 [제안].
- **manifest 병합:** 묶음 manifest의 `ann_file`과 `num_samples`를 합친 병합 manifest를 끝에 쓴다 [새 코드]. `FutureTeacherCache`(§4 단계 8)는 병합본의 sha head를 검사한다.
- **삭제 순서:** 묶음 단위 삭제는 그 묶음의 9-4 4·5·8번만 보장한다. 9-4 6·9번(§6-2 재현, §6-4 품질)은 **첫 묶음에서 통과시킨 뒤** 나머지 묶음을 지운다. 첫 묶음은 지우지 않고 남겨 두었다가 전체 9-4를 통과한 뒤에 지운다 [제안].

**다른 절 갱신 메모** (이번에는 §0·§3·§7-2만 고쳤다)
- §2-4의 프레임당 크기 2.023 / 3.116 MB는 archive 0 값 2.029 / 3.091 MB, 로컬 80 log 표본 2.006 / 3.086 MB와 1–2% 안에서 같다. 고칠 필요는 크지 않다.
- §4 단계 2의 `wget -O- | tar` 루프는 wget 기본 Range 재시도로 이어받는다(로컬 끊김 서버로 확인, HF 미시험). 그대로 쓰려면 `--tries=0 --read-timeout=60`을 넣고 `/resolve/<sha>/`로 고정한다. sha256 대조와 member 목록 기록이 필요하므로 9-8의 `fetch`(rstream 또는 같은 자리에 wget)를 권한다.
- §7-1의 "다운로드 크기와 시간을 모름"과 "필요한 log만 받을 수 있는지 모름" 두 행은 9-1로 해소됐다. 2,124 GB이고, log만 골라 받아도 절약이 0이다.
- §8 "확인하지 못한 것"의 "OpenScene archive 크기와 log 대응"과 "`--drop-bev` npz 실제 크기"는 해소됐다. 87,886 B는 2,648,014 B에서 bev 멤버를 뺀 계산값이다. 검증 단계에서 기존 npz 3개를 bev 없이 `np.savez`로 다시 쓰면 87,760 B였다(차이 0.1%).
- 9-1의 "log가 archive 경계를 넘지 않는다"는 archive 0과 대응표에 근거한 추론이다. P-chunked의 첫 묶음 검사(누락 0)가 이를 실제로 확인한다.

---

## 10. 확정 운영안: 스트리밍하며 필요한 파일만 풀어 보관

작성 2026-10-03. 사용자 결정: "모든 원본 2TB 저장은 안 할 거고 캐싱에 필요한 건 저장하는 식으로 할게. 데이터 전송받으면서 필요한 데이터만 풀고 안 쓰는 건 다 버리는 식으로."

- 이 절이 다운로드·추출의 실행 기준이다. §4 단계 2와 §9-8의 스케치보다 우선한다. §9-1–§9-3의 측정값은 근거로 그대로 쓴다.
- §9-8에서 [새 코드]였던 두 도구를 구현해 이 서버에서 시험했다(§10-5). 시험은 scratch 폴더에만 풀었다. `/home/external-user/{navsim,datasets,ssd}`의 기존 데이터에는 쓰지 않았다.
- 수치는 이 서버에서 `make_needed_files.py`로 다시 센 값이다. 새 서버도 같은 navtrain 센서 패키지와 같은 현재 프레임 캐시를 가진다(S1)고 가정한다. 새 서버에서 목록을 다시 만들면 같은 값이 나와야 한다(§10-3 1번).

도구 위치: `$W/xfer/` (= `$REPO/tools/future_teacher_cache/xfer/`, git에 포함. 새 서버에서는 clone으로 받는다)

| 파일 | 하는 일 | 상태 |
|---|---|---|
| `make_needed_files.py` | token 목록과 horizon을 받아, 실행할 미래 프레임과 archive별 필요 파일 목록(`needed_{camera,lidar}_<i>.txt`, 200개씩)을 만든다. sweep(f−1, f−2) 포함. 디스크에 이미 있는 파일은 뺀다. `run_frames.txt`, `run_frames.yaml`(converter `--scene-filter` 형식), `summary.json`도 쓴다 | [구현·시험됨] 전체 집합 약 4–8 s |
| `stream_extract.sh` | archive마다 `rstream.py` → `tar -xzvv -C <staging> --strip-components=2 -T <목록>` 파이프. 검사를 통과한 archive만 hard link로 센서 폴더에 옮긴다. archive별 상태 JSON, 재실행 시 완료분 건너뛰기, `-P` 병렬 | [구현·시험됨] TEST A 28/28, TEST B 통과 |
| `rstream.py` | HTTP/1.1 스트림을 stdout으로. 끊기면 `Range: bytes=<offset>-`로 다시 붙는다. 스트림의 sha256을 직접 계산하고, TOTAL보다 많이 쓰지 않고, Content-Range를 확인한다. 옛 사용법(`rstream.py URL TOTAL > out`)도 된다 | [구현·시험됨] 종료 코드 0 정상, 3 재시도 소진(기본 200회), 4 읽는 쪽이 닫힘, 5 서버가 너무 많이 보냄 |
| `tests/flaky_server.py`, `tests/test_a.sh`, `tests/test_b_verify.py` | 합성 서버 시험, 실제 archive 시험 검증 | §10-5 |

### 10-0. 한 줄 결론과 남기는 것·버리는 것

**400개 archive(2,124 GB)를 한 번 스트리밍하면서, 실행할 미래 프레임의 F0/L0/R0 `.jpg`·`MergedPointCloud` `.pcd`와 그 sweep `.pcd` 중 디스크에 없는 것만 기존 센서 폴더에 풀어 보관한다. 나머지는 스트림에서 바로 버리고, archive 바이트는 디스크에 쓰지 않는다.**

| 구분 | 무엇 | 양 (S1) |
|---|---|---|
| **보관** | 실행 프레임의 3캠 `.jpg` + `.pcd`, 그 프레임의 sweep `.pcd`. 모두 디스크에 없던 것만. `$DL/trainval_sensor_blobs/trainval/<log>/<CAM_xx\|MergedPointCloud>/`에 기존 파일과 같은 구조로 들어간다 | E2E 4 s: camera 235,629 + lidar 78,543 파일, 약 159 GB / navtrain 5 s: 336,021 + 112,007 파일, 약 227 GB [추정: 실측 파일 수 × archive 0 평균 크기, ±2%] |
| **보관 (기록)** | archive별 상태 JSON(sha256, 바이트, 재시도, tar 종료 코드, 푼 파일 수·크기), `members/`(tar가 푼 파일 목록 = 나중에 지울 때의 목록), 목록·summary | 수십 MB |
| **보관 (결과)** | teacher npz (`--drop-bev`) | E2E 4 s 8.2 GB / navtrain 5 s 11.7 GB [추정] |
| **버림** | 같은 archive 안의 나머지 member: 다른 5개 카메라(CAM_L1, L2, R1, R2, B0) 전부, 범위 밖 프레임의 3캠·lidar, 이미 디스크에 있는 파일(덮어쓰지 않음) | 압축 기준 받은 바이트의 약 93%(E2E 4 s) / 89%(navtrain 5 s) |
| **저장 안 함** | archive(tgz) 바이트 자체. `rstream.py` → `tar` 파이프로만 흐른다. 디스크에 생기는 것은 archive별 staging 폴더에 풀린 "목록에 있는 파일"뿐이고, 검사 뒤 hard link로 옮기고 staging은 지운다 | 0 (staging은 동시에 최대 P × archive당 필요분, 대부분 1 GB 미만, 최대 2.7 GB) |

**보관하기 때문에 생기는 결과 [추론]**
- **되돌릴 수 있는 것** (다시 받지 않고 보관 파일로 다시 추론하면 된다. 전용 GPU 1장 E2E 4 s 약 2.2 h, navtrain 5 s 약 3.2 h):
  - BEV feature 저장 여부. 그래서 **`--drop-bev`를 안전한 기본값**으로 둔다. bev를 넣은 npz(E2E 4 s 248 GB, navtrain 5 s 352 GB)는 보관 센서(159 / 227 GB)보다 크다. F3/P4가 필요해지면 그때 bev를 다시 추론한다(§9-5의 권고와 같은 결론).
  - teacher checkpoint 교체(재학습 teacher), config 변경(100×100 격자, lidar-only, ROI, score 임계, query 수), infos 재생성. 단 F0/L0/R0와 sweep ≤ 2 안에서만이다.
  - §6-2 재현이나 §6-4 품질 검사에서 늦게 버그가 나와도 재전송이 필요 없다. §9-4의 삭제 전 관문 대부분이 "재DL 방지" 조건에서 "다시 추론하면 되는" 조건으로 바뀐다.
- **되돌릴 수 없는 것** (바꾸려면 2,124 GB를 다시 받는다. camera만이면 1,242 GB, lidar만이면 882 GB):
  - 나머지 5개 카메라: 8캠·360° teacher, 학생 쪽 다른 카메라
  - 고른 범위·horizon 밖의 프레임: 예를 들어 E2E 4 s로 정한 뒤의 dev(val_logs) token, 5 s, 다른 token 집합, trainval 전 프레임(§9-3)
  - ReSMap 미래를 log 안에서 빈틈없이(2 Hz 연속) 돌려야 할 경우의 연속 구간 프레임(B 방식). 기존 ReSMap 캐시처럼 "센서가 있는 프레임만 장면 순서대로" 돌리면(A 방식) 보관한 3캠으로 충분하다. B 방식의 추가분은 `--resmap-full-logs`로 미리 풀 수 있다(§10-7, +231 GB E2E 4 s / +254 GB navtrain 5 s)
  - sweep을 3개 이상 쓰는 teacher(f−3 `.pcd`가 없음)
- 그래서 **스트리밍을 시작하기 전에** 범위, 카메라, ReSMap 연속 구간 필요 여부를 정한다(§10-1). 시작한 뒤에 범위를 넓히면 그만큼 다시 받아야 한다. 범위가 어디든 400개 archive가 모두 필요하므로 사실상 전체 재전송이다.

### 10-1. 남은 결정과 기본값

기본값은 [제안]이다. 사용자가 정한다. 디스크는 S1, decimal GB, 보관 센서 기준이다.

| # | 결정 | 선택지와 디스크 | 기본값 [제안] | 근거와 되돌릴 수 있는지 |
|---|---|---|---|---|
| 1 | **범위·horizon** | E2E 4 s: 실행 93,606 프레임, 보관 159 GB, npz 8.2 GB, GPU 2.24 h / navtrain 5 s: 133,113, 227 GB, 11.7 GB, 3.19 h. 참고: stage-T 4 s 107 GB, navtrain 4 s 190 GB, E2E 5 s 191 GB | **navtrain 5 s** | 전송량은 같다(2,124 GB). E2E 4 s의 실행 프레임은 navtrain 5 s에 모두 들어 있다(run_frames 비교, 차집합 0) [실측]. 차이 +68 GB, npz +3.5 GB, GPU +1.0 h(전용 1장)로 dev token 평가와 GT 벌점과 같은 0–5 s 범위를 얻는다(§9-3). 범위 밖은 되돌릴 수 없다. 5 s를 쓰려면 `future_index_train.json`(지금 k=1..8)을 k=1..10으로 다시 만든다(`make_token_lists.py:30`의 `range(1, 9)` → `range(1, 11)`, 그리고 `:35-36, :46`의 출력 파일 이름을 바꿔 4 s 파일을 덮어쓰지 않게 한다) [새 코드, 몇 줄]. 이 index는 §4 단계 8 로더에만 필요하고 스트리밍·추론(§10-3 1–11번)에는 필요 없다 |
| 2 | **카메라** | 3캠(F0/L0/R0) / 8캠: E2E 4 s +83 GB(합 242), navtrain 5 s +119 GB(합 346) [추정: DL 프레임 × 다른 5캠 평균 1.062 MB] | **3캠** | BEVFusion과 ReSMap 모두 3캠만 쓴다. 8캠은 360° teacher나 학생 쪽 다른 카메라를 쓸 때만 필요하다. 되돌릴 수 없다. 8캠으로 하려면 `make_needed_files.py --cams CAM_F0 CAM_L0 CAM_R0 CAM_L1 CAM_L2 CAM_R1 CAM_R2 CAM_B0`만 바꾼다(단계 5의 `--cameras`는 빼도 된다) |
| 3 | **ReSMap 미래 구간 방식** (§10-7) | A. 센서가 있는 프레임만 장면 순서대로(기존 ReSMap 캐시 방식): **추가 0 GB** / B. log를 2 Hz로 빈틈없이: 3캠 추가 E2E 4 s 368,453 프레임 약 231 GB(합 약 390 GB), navtrain 5 s 405,086 프레임 약 254 GB(합 약 481 GB) [실측 파일 수 × 평균 크기] | **A** | 사용자 요청으로 ReSMap 적용 가능성을 남긴다. A는 기존 ReSMap 캐시와 R_M teacher가 만들어진 조건과 같아 비교가 깨지지 않는다. B를 나중에 원하면 camera archive 1,242 GB를 다시 받아야 하므로, B가 필요하면 시작 전에 `--resmap-full-logs`를 켠다 [구현·시험됨]. §9-5의 "324 GB"는 디스크 밖 모든 프레임의 3캠(A에서 이미 푸는 실행 프레임 포함)이었다 |
| 4 | **캐싱 뒤 보관 파일 삭제** | 보관: 위 표 그대로 / 삭제: −159 GB(E2E 4 s) 또는 −227 GB(navtrain 5 s) | **보관** | 사용자 결정의 "캐싱에 필요한 건 저장"과 같다. 지우면 §10-0의 "되돌릴 수 있는 것"이 모두 되돌릴 수 없게 된다. 지우기로 하면 §9-4 관문을 모두 통과한 뒤 `state/members/*.txt`와 `preexisting.txt`(§10-3 2번)로 지운다. 삭제 도구는 아직 없다 [새 코드] |
| 5 | `--drop-bev` | 켬: npz 8.2 / 11.7 GB / 끔: 248 / 352 GB | **켬** | 보관 파일로 다시 추론할 수 있어 되돌릴 수 있다(§10-0) |

- **시나리오 확인:** 위 수치는 새 서버에 navtrain current+history 센서 패키지가 있다는 전제(S1)다. 없다면(S2) E2E 4 s 보관량은 230 GB가 된다(실행 93,606 프레임 전부 + sweep용 lidar 28,623 프레임) [추정]. 다만 S2에서는 학생 학습에 쓰는 navtrain 현재·과거 센서도 없으므로, 그것을 같은 스트림에서 함께 풀지(§9-7) 따로 받을지를 먼저 정해야 한다. 지금 `make_needed_files.py`는 실행 프레임과 sweep만 목록에 넣는다. navtrain 현재·과거까지 넣으려면 목록 확장이 필요하다 [새 코드, 작음].

### 10-2. 용량·시간 예산

**전송** [실측, HF API]: 어느 범위든 400개 archive, 2,124 GB(camera 1,242 + lidar 882). archive 쌍 평균 10.6 GB, 최소 3.0 GB(#146), 최대 33.8 GB(#124).

**보관하는 센서** [추정: `make_needed_files.py` 파일 수 × archive 0 평균 크기]

| 범위 | 실행 프레임 | DL 프레임 | 보관 (S1) | 보관 (S2) | 버리는 비율 (S1, 압축 기준) | npz `--drop-bev` / bev 포함 |
|---|---:|---:|---:|---:|---:|---:|
| stage-T 4 s | 62,629 | 52,528 | 106.6 GB | 159.3 GB | 95% | 5.5 / 166 GB |
| E2E 4 s | 93,606 | 78,543 | 159.3 GB | 230.1 GB | 93% | 8.2 / 248 GB |
| navtrain 4 s | 112,152 | 93,670 | 190.0 GB | 275.9 GB | 91% | 9.9 / 297 GB |
| E2E 5 s | 111,236 | 93,994 | 190.7 GB | – | 91% | 9.8 / 295 GB |
| **navtrain 5 s** | 133,113 | 112,007 | **227.2 GB** | – | 89% | 11.7 / 352 GB |

- S2 값은 §9-2와 E2E 4 s 재계산(`e2e_4s_S2`)에서 가져왔다. "–"는 세지 않았다.

**새 서버 디스크 합계** (기준선 약 1,507 GB, §9-7) [추정]

| 운영 | 합계 | 비고 |
|---|---:|---|
| E2E 4 s, 3캠, `--drop-bev` | 약 1.67 TB | 1,507 + 159 + 8 |
| E2E 4 s, 3캠, bev 포함 | 약 1.91 TB | + 248 |
| **navtrain 5 s, 3캠, `--drop-bev` (기본값)** | **약 1.75 TB** | 1,507 + 227 + 12 |
| navtrain 5 s, 3캠, bev 포함 | 약 2.09 TB | + 352 |
| 8캠으로 할 때 | 각 행 + 83 GB(E2E 4 s) / + 119 GB(navtrain 5 s) | |

- 일시 공간: staging은 동시에 P × archive당 필요분이다. 가장 큰 것은 navtrain 5 s의 lidar_124 약 2.7 GB이고, P=4면 많아야 약 11 GB다. `bevfusion.tar.zst`(156 GB)로 teacher cache를 옮긴다면 그동안 +156 GB가 더 든다.
- ReSMap 캐시(377 GB)를 새 서버에 두지 않으면 그만큼 준다.
- 권장 디스크는 **약 2 TB**다(기본값 1.75 TB + 일시 공간). bev까지 저장하면 2.5 TB [추론].

**다운로드 시간** [추정: 2,124 GB ÷ 대역폭]

| 합산 대역폭 | 시간 | 근거 |
|---:|---:|---|
| 1.64 MB/s | 약 360 h (15일) | 이 서버, 쌍 #146, 2스트림(§10-5) |
| 4 MB/s | 약 148 h (6.2일) | 이 서버, 쌍 0, 2스트림(§9-1) |
| 10 MB/s | 약 59 h | 가정 |
| 50 MB/s | 약 11.8 h | 가정 |
| 100 MB/s | 약 5.9 h | 가정 |

- 스트림당 실측은 camera 1.35–2.75 MB/s, lidar 0.67–1.23 MB/s다. 병렬 3개 이상은 HF에서 재지 않았다. 새 서버 대역폭과 HF의 스트림당·IP당 제한은 모른다.
- 가장 큰 archive 하나가 걸리는 시간: lidar_124(14.46 GB)는 0.67 MB/s면 약 6.0 h, camera 최대(19.30 GB)는 1.35 MB/s면 약 4.0 h다. 큰 archive부터 받으면 마지막에 긴 꼬리가 남지 않는다(§10-3 5번).
- 풀기 CPU는 병목이 아니다(전체 약 2.7 core-h, §9-2).

**GPU 시간** [추정, 전용 11.6 frame/s는 가정이고 공유 5.8만 실측]

| 범위 | 전용 1장 | 공유 1장 | 전용 4장 |
|---|---:|---:|---:|
| E2E 4 s (93,606) | 2.24 h | 4.48 h | 0.56 h |
| navtrain 5 s (133,113) | 3.19 h | 6.38 h | 0.80 h |

- 추론은 전송이 끝난 뒤 한 번에 돌린다(§9-2의 P-stream-all). 전송이 GPU보다 수십 배 길어서, 묶음 단위 겹침(P-chunked)으로 아낄 수 있는 시간은 GPU 시간 정도(수 h)뿐이다 [추론]. 그래서 P-chunked는 구현하지 않았다.

### 10-3. 실행 절차 (새 서버)

변수는 §4와 같다. 아래를 더한다.
```bash
REPO=/home/external-user/yongjae/SSR; D=/home/external-user/ssd/yongjae_refiner
PY=/home/external-user/miniconda3/envs/ssr/bin/python; BPY=/home/external-user/miniconda3/envs/bevfusion/bin/python
DL=/home/external-user/navsim/download; TC=/home/external-user/datasets/teacher_cache
BF=/home/external-user/yongjae/bevfusion; W=$REPO/tools/future_teacher_cache; X=$W/xfer
SB=$DL/trainval_sensor_blobs; NGPU=1   # NGPU: 추론에 쓸 전용 GPU 수
SET=navtrain_5s; TOK=$W/future_index_train.json; HS=5          # 기본값 navtrain 5 s
# E2E 4 s로 정했다면 대신: SET=e2e_4s; TOK=$D/splits/e2e_train_trainlogs.parquet; HS=4   (39 S5 §5.2가 만든 parquet)
R=$X/run_$SET       # 이 폴더에 lists/, state/, 로그가 쌓인다(수십 MB)
```

**0. 준비 (§4 단계 0, 3)** — `$W`와 `$W/xfer/`는 git으로 받는다(clone, branch `exp-refine`). 단계 0의 나머지 항목(bevfusion working tree, 체크포인트, 비교 기준 infos)만 옮기고, 단계 3(`make_token_lists.py`)으로 `future_index_train.json`을 만든다. E2E 4 s면 39 S5(§5.2)의 `$D/splits/e2e_train_trainlogs.parquet`가 먼저 있어야 한다.
- 5 s의 k=1..10 index(§10-1 1번)는 이 절의 1–11번에는 필요 없다. `make_needed_files.py`가 `--horizon-s`로 log pkl에서 미래 프레임을 직접 센다. k=1..10 index는 §4 단계 8 로더용이므로 그 전에 만들면 된다.
```bash
ls $X/{make_needed_files.py,stream_extract.sh,rstream.py} $X/hf/{map_trainval.json,tree_openscene_sensor_trainval_camera.json,tree_openscene_sensor_trainval_lidar.json}
jq --version; flock -V | head -1; tar --version | head -1    # jq, flock, GNU tar 필요 (이 서버: GNU tar 1.35)
ls $SB/trainval | wc -l     # S1 확인. 이 서버 값 1192 (navtrain log 폴더)
df -h $SB                   # §10-2 예산과 비교
```

**1. 필요 파일 목록** (CPU, 수 초–수 분)
```bash
$PY $X/make_needed_files.py --tokens $TOK --horizon-s $HS --out $R/lists      # 약 5–20 s
jq '{run_frames, run_frames_on_disk, dl_frames, run_frames_on_disk_missing_some_sweep, sweep_only_lidar_frames_outside_run,
     camera_files:.totals.camera_files, lidar_files:.totals.lidar_files, est_extracted_GB, archives_with_nonempty_list}' $R/lists/summary.json
```
- 기본값 `--logs-dir`, `--sensor-root`, `--cache-dir`(`cache_train_50x100`), `--map xfer/hf/map_trainval.json`, `--cams CAM_F0 CAM_L0 CAM_R0`은 옛 서버와 같은 절대경로를 가정한다.
- ReSMap B 방식(§10-1 3번, §10-7)으로 정했다면 위 명령에 `--resmap-full-logs`를 더한다. A 방식(기본)이면 그대로 둔다. 어느 쪽이든 `jq .resmap $R/lists/summary.json`으로 ReSMap 프레임 수를 본다(아래 §10-7 표와 같아야 한다).
- 기대값(이 서버, S1):

| 키 | navtrain 5 s | E2E 4 s |
|---|---:|---:|
| `run_frames` | 133,113 | 93,606 |
| `run_frames_on_disk` | 21,106 | 15,063 |
| `dl_frames` | 112,007 | 78,543 |
| `run_frames_on_disk_missing_some_sweep` | 9,727 | 6,610 |
| `sweep_only_lidar_frames_outside_run` | 0 | 0 |
| `camera_files` / `lidar_files` | 336,021 / 112,007 | 235,629 / 78,543 |
| `est_extracted_GB` | 227.2 | 159.3 |
| `archives_with_nonempty_list` | camera 200, lidar 200 | 같음 |

- 값이 다르면 새 서버의 navtrain 센서 패키지나 현재 프레임 캐시가 옛 서버와 다르다는 뜻이다. 스트리밍 전에 원인을 본다.

**2. 기록 남기기** (출처 고정, §9-6)
```bash
mkdir -p $R/prov
find $SB/trainval -type f | sort > $R/prov/preexisting.txt          # 스트리밍 전 파일 스냅샷 (나중에 지울 때의 가드)
sha256sum $X/hf/*.json > $R/prov/hf.sha256
sha256sum $DL/trainval_navsim_logs/trainval/*.pkl > $R/prov/logs.sha256
cp $R/lists/summary.json $R/prov/summary_before.json
```
- URL은 `stream_extract.sh` 안에서 revision `a76f840b65e972bc45e56c2adced897498e9a026`로 고정돼 있다. archive별 sha256 대조는 도구가 한다.

**3. 대역폭 시험 + 출처 대조** (가장 작은 쌍 #146, 3.00 GB, scratch 폴더에 푼다. 본 실행 전에 한다)
```bash
T=$R/probe146
bash $X/stream_extract.sh -l $R/lists -r $T/root -s $T/state -P 2 --extra-lists $X/results/stream_test_146 146
$PY $X/tests/test_b_verify.py $R/lists $X/results/stream_test_146 $T/state $T/root $SB 146
rm -rf $T/root                  # 시험 추출물. 본 실행에서 #146을 다시 받는다(3.0 GB)
```
- `--extra-lists`는 이미 디스크에 있는 파일 50개(camera 30, 8캠 전부 포함 / lidar 20)를 같은 스트림에서 함께 풀어, `test_b_verify.py`가 디스크의 원본과 `cmp`한다(§9-4 3번). 이 50개는 navtrain 패키지 파일이므로 S1이면 새 서버에도 있다 [추론].
- 기대 출력(이 서버 값): `summary {'done': 2} … aggregate`, 마지막 줄 `TEST B: PASS`. 필요 파일은 navtrain 5 s 목록이면 camera 459 + lidar 153개, E2E 4 s 목록이면 261 + 87개다. 여기에 provenance 50개가 더해진다. `provenance_cmp_identical`이 30/20, `needed_present_on_real_disk`가 0이어야 한다.
- 이 서버에서는 1,835 s, 합산 1.64 MB/s였다. 이 값으로 전체 시간을 다시 어림한다(§10-2).
- 이 시험은 2번(스냅샷) 뒤, 본 실행(4·5번) 전에 한다. 본 실행이 #146을 이미 받은 뒤에는 `needed_present_on_real_disk`가 0이 아니라 FAIL이 된다. provenance 파일이 새 서버 디스크에 없으면 `test_b_verify.py`가 FileNotFoundError로 멈추는데, 이는 S1 전제가 다르다는 뜻이다.

**4. 병렬 수 정하기** (본 실행의 일부. 받은 것은 그대로 쓴다)
```bash
bash $X/stream_extract.sh -l $R/lists -r $SB -s $R/state -P 4 144 143 126 58      # 다음으로 작은 4쌍 = archive 8개, 동시 스트림 4개
# 마지막 요약 줄의 "aggregate MB/s"를 3번(-P 2)과 비교한다. 거의 2배면 -P 8로 다음 몇 쌍(128 …)을 더 재 본다
```
- `-P`는 동시 스트림 수다. camera_i와 lidar_i는 따로 센다.
- 합산 속도가 더 오르지 않는 지점의 P를 쓴다. HF가 429나 연결 거부를 내기 시작하면 줄인다 [제안. 3개 이상은 미측정].

**5. 본 실행** (tmux 또는 screen 안에서. 멈췄다가 다시 시작해도 같은 명령)
```bash
$PY -c "
import json,re;s={}
for m in ('camera','lidar'):
  for f in json.load(open('$X/hf/tree_openscene_sensor_trainval_'+m+'.json')):
    g=re.search(r'_(\d+)\.tgz$',f['path'])
    if g: s[int(g[1])]=s.get(int(g[1]),0)+f['size']
print(*sorted(s,key=lambda i:-s[i]))" > $R/order_desc.txt       # 큰 쌍부터: 124 116 191 153 51 …
P=4      # 4번에서 정한 값
bash $X/stream_extract.sh -l $R/lists -r $SB -s $R/state -P $P -f $R/order_desc.txt 2>&1 | tee -a $R/main.out
# 멈추기: 그 창에서 Ctrl-C (프로세스 그룹 전체가 멈춘다). 다시 시작: 같은 명령. done인 archive는 건너뛴다
```
- 결과 위치: `$SB/trainval/<log>/{CAM_F0,CAM_L0,CAM_R0,MergedPointCloud}/<token>.{jpg,pcd}`. 기존 navtrain 파일과 같은 폴더에 hard link로 들어가고, 기존 파일은 덮어쓰지 않는다.
- staging은 `$SB/.xfer_staging/<mod>_<i>/`(같은 파일시스템이어야 한다. 다르면 시작할 때 거부한다).
- archive마다 기대 로그: `<mod>_<i> done in …s (<bytes> B, sha ok, {"committed_new":N,"already_present_same_size":0,"conflict_different_size":0,…,"missing_after_commit":0})`
- 끝났을 때 기대 요약: `summary {'done': 400}`. 3·4번에서 받은 것은 건너뛰므로 `streamed_this_run`은 그만큼 적다. 하나라도 done이 아니면 `NOT DONE: …`을 찍고 exit 1이다.

**6. 상태 확인** (실행 중 아무 때나)
```bash
jq -r .state $R/state/status/*.json | sort | uniq -c                    # 끝나면 400 done
jq -r 'select(.state!="done" and .state!="skipped_empty") | "\(.key) \(.why // .error)"' $R/state/status/*.json
jq -s 'map(select(.state=="done")) | {n:length, tgz_GB:(map(.bytes)|add/1e9), retries:(map(.rstream_retries)|add),
        kept_files:(map(.commit.committed_new + .commit.already_present_same_size)|add), kept_GB:(map(.verify.extracted_bytes)|add/1e9),
        not_found:(map(.not_found)|add), sha_bad:(map(select(.sha_ok|not))|length)}' $R/state/status/*.json
tail -n 5 $R/state/status.log; du -sh $SB/.xfer_staging
```
- 이 서버의 #146 시험에서 같은 명령의 결과는 `n 2, tgz_GB 3.00, retries 2, kept_files 398, not_found 0, sha_bad 0`이었다(398 = 필요 348 + provenance 50).
- `kept_files`에 `already_present_same_size`를 더하는 이유: 반영(hard link) 도중에 실행이 죽으면 다음 실행에서 그 파일들은 "이미 있음, 같은 크기"로 세진다.
- 끝났을 때 `kept_files`는 `camera_files + lidar_files`(navtrain 5 s 448,028 / E2E 4 s 314,172)와 같아야 하고, `kept_GB`는 `est_extracted_GB`와 약 ±2% 안이어야 한다. `.xfer_staging`은 비어 있어야 한다(끝나면 `rmdir`).

**7. 전역 완결성** (전송이 끝난 뒤)
```bash
$PY $X/make_needed_files.py --tokens $TOK --horizon-s $HS --out $R/lists_after    # 1번과 같은 입력
jq '{run_frames, run_frames_on_disk, dl_frames, run_frames_on_disk_missing_some_sweep,
     camera_files:.totals.camera_files, lidar_files:.totals.lidar_files, archives_with_nonempty_list}' $R/lists_after/summary.json
```
- 기대: `run_frames`는 1번과 같고, `run_frames_on_disk` = `run_frames`, `dl_frames` 0, `run_frames_on_disk_missing_some_sweep` 0, 파일 0, `archives_with_nonempty_list` camera 0 / lidar 0 [제안, 추출 후 재실행은 미시험. 정의상 성립].
- 이것이 §4 단계 5의 lidar·sweep 사전 검사를 대신한다.

**8. 전송 중에 병행할 것:** §4 단계 1(bevfusion env)과 단계 4(§6-2 재현, navtest 24 token)는 센서 다운로드와 무관하다. 전송하는 며칠 동안 끝내 둔다.

**9. infos** (§4 단계 5, bevfusion env, CPU)
```bash
cd $BF
CUDA_VISIBLE_DEVICES="" nice -n 10 $BPY tools/data_converter/navsim_converter.py \
  --navsim-logs $DL/trainval_navsim_logs/trainval --sensor-root $SB --split trainval \
  --scene-filter $R/lists/run_frames.yaml --keyframes-only --check-files --cameras CAM_F0 CAM_L0 CAM_R0 \
  --workers 8 --lidar-prefix lidar --lidar-ext .pcd --max-sweeps 2 --out $W/infos/future_train_$SET.pkl
sha256sum $W/infos/future_train_$SET.pkl > $R/prov/infos.sha256
```
- `run_frames.yaml`은 `{log_names, tokens}` 형식이라 converter의 `--scene-filter`에 그대로 들어간다(`load_scene_filter`가 두 키를 읽음) [실측, 코드].
- 기대: 프레임 수 = `run_frames`(133,113 / 93,606), drop 0. 3캠만 풀었으므로 `--cameras`는 반드시 준다.

**10. BEVFusion 추론** (§4 단계 6, GPU)
```bash
cd $BF; export PATH=/home/external-user/miniconda3/envs/bevfusion/bin:$PATH NCCL_SOCKET_IFNAME=lo OMP_NUM_THREADS=2
torchpack dist-run -np $NGPU python $W/cache_teacher_future.py \
  configs/navsim/det/transfusion/secfpn/camera+lidar/swint_convfuser.yaml runs/navsim-fusion-50x100/epoch_20.pth \
  --cache-dir $TC/bevfusion/cache_train_50x100_future --split train --ann-file $W/infos/future_train_$SET.pkl \
  --skip-existing --drop-bev
```
- split당 `-np N` 한 번으로 돌린다(manifest 덮어쓰기 방지, §4 단계 6).

**11. 사후 검사** (§4 단계 7, §6-3)
```bash
C=$TC/bevfusion/cache_train_50x100_future
jq '{checkpoint_sha256_head, bev_feature_stored, num_samples_expected, num_samples_written}' $C/manifest.json
# 기대: "cddf943ffec8d6a8", false, 둘 다 run_frames
$PY - $R/lists/run_frames.txt $C <<'EOF'
import sys, os, glob, numpy as np
run = [l.split('\t')[0] for l in open(sys.argv[1]) if l.strip()]
S = sys.argv[2] + '/samples'
miss = [t for t in run if not os.path.isfile(f'{S}/{t[:2]}/{t}.npz')]
bad = 0
for t in run:
    try: np.load(f'{S}/{t[:2]}/{t}.npz')['pred_boxes_3d']
    except Exception: bad += 1
print(len(run), 'missing', len(miss), 'tmp', len(glob.glob(f'{S}/*/*.tmp.npz')), 'unreadable', bad)
EOF
# 기대: <run_frames> missing 0 tmp 0 unreadable 0
cp $C/manifest.json $R/prov/manifest_$SET.json
```

**12. 품질 검사와 그 뒤:** §6-4(좌표 in-sample cm 수준, out-of-sample 재현율), 박스 영상 투영 QA를 한다. 그다음 §4 단계 8(로더·raster 빌더)로 간다. 보관 파일이 남아 있으므로 여기서 버그가 나와도 다시 추론하면 된다.

### 10-4. 검증 기준

| 수준 | 항목 | 합격 기준 | 어디서 확인 | 상태 |
|---|---|---|---|---|
| archive | 받은 바이트 | `bytes == size`(HF tree) | `stream_extract.sh` 상태 JSON | [구현·시험됨] |
| archive | 출처 | 스트림 sha256 == `lfs.oid` | 같음(`sha_ok`) | [구현·시험됨] TEST A의 잘못된 oid는 실패로 잡힘 |
| archive | 종료 코드 | rstream rc 0, tar rc 0 | 같음 | [구현·시험됨] |
| archive | 누락 | "Not found in archive" 0 | `logs/<key>.tar.err` | [구현·시험됨] TEST A의 없는 이름은 실패로 잡힘 |
| archive | 목록 일치 | 목록의 파일이 모두 있고 크기 > 0, tar가 보고한 크기와 같음, 목록 밖 파일 0 | 같음(`verify`) | [구현·시험됨] |
| archive | 반영 | 덮어쓰기 0(`conflict_different_size` 0), 반영 뒤 누락 0(`missing_after_commit` 0) | 같음(`commit`) | [구현·시험됨] 크기가 다른 기존 파일은 건드리지 않고 실패 처리 |
| 출처 | 기존 파일과 내용 일치 | provenance 50개 `cmp` 100% | §10-3 3번 `test_b_verify.py` | 이 서버에서 통과. 새 서버에서 다시 |
| 전체 | 실행 프레임 완결 | `make_needed_files.py` 재실행에서 DL 0, sweep 누락 0, 필요 파일 0 | §10-3 7번 | [제안, 미시험] |
| 전체 | 크기 | 보관 바이트가 `est_extracted_GB`와 ±2% 안 | §10-3 6번 | #146에서 175.4 vs 174.6 MB(+0.5%) |
| infos | 프레임 수 | `run_frames`와 같음, drop 0 | §10-3 9번 | [제안] |
| teacher | 재현 | §6-2 기준(24/24, 매칭 100%) | §4 단계 4 | [제안] |
| npz | 완결 | npz 수 = `run_frames`, 차집합 ∅, `.tmp.npz` 0, 전수 `np.load`, sha head `cddf943ffec8d6a8`, `bev_feature_stored` false | §10-3 11번 | [제안] |
| npz | 품질 | §6-4 좌표 오차 cm 수준(in-sample) | §6-4 | [제안] |

### 10-5. 시험 결과 (이 서버, 2026-10-03)

**목록 생성기 수치** (`results/needed_summaries/*.json`, 각 약 4–8 s)

| 집합 | 실행 | 디스크에 있음 | DL | 있지만 sweep 누락 | 실행 밖 sweep 파일 | 보관 GB (archive 비율 보정) | 버리는 비율 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2E 4 s | 93,606 | 15,063 | 78,543 | 6,610 | 0 | 159.3 (158.9) | 93% |
| stage-T 4 s | 62,629 | 10,101 | 52,528 | 4,164 | 0 | 106.6 | 95% |
| navtrain 4 s | 112,152 | 18,482 | 93,670 | 8,075 | 0 | 190.0 | 91% |
| navtrain 5 s | 133,113 | 21,106 | 112,007 | 9,727 | 0 | 227.2 | 89% |
| E2E 5 s | 111,236 | 17,242 | 93,994 | 7,997 | 0 | 190.7 | 91% |
| E2E 2 s | 52,647 | 9,431 | 43,216 | 3,024 | 0 | 87.7 | 96% |
| E2E 4 s, S2(디스크에 센서 없음) | 93,606 | 0 | 93,606 | 0 | 28,623 | 230.1 | 89% |

- 모든 집합에서 camera 200개, lidar 200개 archive의 목록이 비어 있지 않다. E2E 4 s는 camera 235,629개, lidar 78,543개 파일이다.
- §2-1, §9-1, §9-2의 값과 모두 같다. 차이는 버리는 비율의 반올림뿐이다: navtrain 4 s 91%(§9-0은 92%), navtrain 5 s 89%(90%). 보관 바이트를 압축 크기로 되돌릴 때 archive 0의 확장 비율(camera 1.003, lidar 1.134)을 써서 생긴 차이다.
- E2E 4 s의 실행 프레임은 navtrain 5 s에 모두 들어 있다(차집합 0). E2E 5 s도 마찬가지다 [실측].

**TEST A: 합성 서버, 네트워크 없음** (`tests/test_a.sh`, 결과 `results/stream_test_A.out`): **28/28 통과**
- 가짜 archive 4쌍(총 33 MB). 서버는 Range, 302 한 번, 연결 끊김을 흉내 낸다. 각 파일의 처음 두 요청은 150 kB 뒤에 끊었다.
- 재개: rstream이 Range로 이어받았고(archive마다 재시도 2회 이상), 완료된 archive의 sha256이 모두 맞았다.
- 실패 처리:
  - 목록이 빈 archive는 받지 않고 `skipped_empty`
  - archive에 없는 이름: tar exit 2, "Not found" 1개, 센서 폴더에 아무것도 쓰지 않음
  - 틀린 `lfs.oid`: sha 검사 실패, 아무것도 쓰지 않음
  - 크기가 다른 기존 파일: 그 파일은 그대로 두고 실패 처리. 검사를 통과한 다른 파일은 반영
- 추출: 센서 폴더에는 목록의 파일만 있고(71개, 목록 밖 0, 누락 0), 원본과 바이트 단위로 같다.
- 재실행: 완료된 archive는 다시 요청하지 않는다. 실패한 것은 다시 시도한다. 입력을 고친 뒤 재실행하면 모두 완료된다.
- 강제 종료 후 재개: `-P 2` 실행을 받는 도중 프로세스 그룹째 SIGKILL하고 `-P 3`으로 다시 돌렸다. 끝까지 갔고, 결과가 중단 없는 실행과 같았고, staging이 남지 않았다.

**TEST B: 실제 HF, archive 쌍 #146(3.00 GB, 가장 작은 쌍), E2E 4 s 목록** (결과 `results/stream_test_146/`): **통과**
- 무결성: sha256 == `lfs.oid`(camera `86a20b3e…99a9`, lidar `c796de6d…8b28`), tar exit 0, "Not found" 0.
- 필요 파일 camera 261 + lidar 87개가 모두 크기 > 0으로 풀렸다. 목록 밖 0, 누락 0(provenance 포함 398개). 필요 파일 중 실제 디스크에 이미 있던 것은 0개였다.
- 출처: 이미 디스크에 있던 camera 30개(8캠 전부 포함)와 lidar 20개를 같은 스트림에서 풀어 `cmp`했다. 50/50 같다.
- 크기: 필요 파일 175.4 MB, 추정 174.6 MB(+0.5%).
- **처음으로 HF에서 재개 경로가 쓰였다.** 두 archive 모두 한 번씩 끊겼다(camera는 1,087,930,077 B에서 "premature end of body"). Range로 이어받았고 sha256이 맞았다.
- 속도: camera 1.35 MB/s(1,317 s), lidar 0.67 MB/s(1,834 s), 합산 약 1.64 MB/s. §9-1의 약 4 MB/s보다 느리다. 이 속도면 전체가 약 15일이다.
- 재실행하면 두 archive 모두 네트워크 없이 건너뛴다.
- 시험 추출물은 지웠다. 로그와 상태는 `results/stream_test_146/state/`에 있다.

**구현 중 고친 문제**
1. rstream: 서버가 오류 없이 연결을 일찍 닫으면 빈 read가 나와 조용히 다시 붙고 재시도 수가 0으로 남았다. 지금은 재시도로 세고 로그를 남긴다.
2. tar: `-C`가 `-T`보다 뒤에 오면 오류 없이 현재 폴더에 푼다. 도구는 `-C`를 앞에 둔다(§9-8 스케치도 순서가 맞다).
3. 요약이 이전 실행에서 끝난 archive까지 바이트에 셌다. 지금은 run ID로 이번 실행분만 센다.
4. TEST A의 강제 종료 단계가 처음에는 worker를 남겼다. 고친 것은 시험(프로세스 그룹 전체를 죽임)이고 도구는 아니다.
5. (검토 단계) `stream_extract.sh`: 반영(hard link) 파이썬이 예외로 죽으면 상태 JSON이 깨진 채(`"commit":` 뒤가 빈 값) 남았다. 지금은 `failed`/`commit script failed`로 기록한다. 경로를 파일로 막은 합성 시험으로 확인했고, 수정 뒤 TEST A를 다시 돌려 28/28이다. `--help`가 코드 줄까지 찍던 것도 고쳤다.

**남은 한계**
- rstream이 포기하면(한 실행에서 재시도 200회 초과) 그 archive는 다음 실행에서 처음부터 다시 받는다. 실행 사이의 바이트 단위 재개는 없다.
- 요약의 바이트는 archive별 마지막 시도만 센다.
- 같은 이름·같은 크기의 기존 파일은 내용 비교 없이 받아들인다(출처 대조는 3번의 50개 표본으로 한다).
- 크기 추정은 archive 0 평균 파일 크기다. 실제 크기는 상태 JSON(`verify.extracted_bytes`)에 남는다.
- 병렬 3개 이상은 HF에서 재지 않았다.
- 보관 파일 삭제 도구(`members/` 기반)는 없다. 지우지 않는 것이 기본값이라 만들지 않았다.

### 10-6. 실패·재시작 대응

| 상황 | 도구의 동작 | 할 일 |
|---|---|---|
| 받는 도중 연결이 끊김 | rstream이 `Range`로 같은 위치부터 다시 붙는다(한 실행에서 최대 200회, `--max-retries`) | 없음. 상태 JSON의 `rstream_retries`로 본다. #146에서는 archive당 1회였다 |
| rstream 재시도 소진(rc 3), sha 불일치, tar 오류, 바이트 부족 | 그 시도의 staging을 지우고 같은 실행 안에서 다시 시도한다(`-R`, 기본 3회, 대기 10 s × 시도 번호). 센서 폴더는 바뀌지 않는다 | 같은 명령을 다시 실행한다. 같은 archive가 계속 sha 불일치면 HF tree·revision이 바뀌었는지 본다(`$R/prov/hf.sha256`) |
| "Not found in archive" > 0 | 실패. 센서 폴더는 바뀌지 않는다 | 목록과 실제 archive가 다르다는 뜻이다(대응표 불일치나 log가 archive 경계를 넘음, §9-1의 [추론]). `logs/<key>.tar.err`의 이름을 보고, 그 log가 어느 archive에 있는지 확인한다 |
| 반영 충돌(같은 이름, 다른 크기) | 그 파일은 건드리지 않고 archive를 실패로 둔다. 다른 파일은 반영된다. 같은 실행에서 다시 시도하지 않는다 | 사람이 본다. 기존 파일이 잘린 파일인지(옛 다운로드의 흔적), archive 쪽이 다른지 확인한다. 정리한 뒤 다시 실행한다 |
| Ctrl-C, 강제 종료, 재부팅 | 끝난 archive는 done으로 남는다. 진행 중이던 archive는 상태가 done이 아니다 | 같은 명령을 다시 실행한다. done은 건너뛰고, 진행 중이던 것은 처음부터 받는다. 남은 staging은 다음 시도가 지운다(TEST A 4번) |
| 반영 단계 자체가 실패(hard link 거부, 권한, 경로가 파일로 막힘 등) | 상태 JSON이 `failed`, `why: commit`, `commit.error: "commit script failed"`가 된다. 다시 시도하지 않는다(검토 단계에서 추가, 합성 시험으로 확인) | `logs/<key>.log`와 센서 폴더 권한·파일시스템을 본다. 고친 뒤 같은 명령. 이미 들어간 파일은 다음 실행에서 `already_present_same_size`로 세진다 |
| 디스크 부족 | tar가 실패해 그 archive가 실패한다. 센서 폴더는 바뀌지 않는다 | 공간을 확보하고 다시 실행한다. §10-2 예산 + staging(P × 최대 2.7 GB)을 미리 확보한다 |
| 같은 state로 두 번 동시에 실행 | archive별 `flock`이 잡혀 있으면 그 archive를 건너뛴다 | 피한다. 한 번에 한 실행만 둔다 |
| HF 속도 제한, 429, 연결 거부 [추론, 미관측] | rstream이 재시도한다. 계속되면 rc 3으로 실패한다 | `-P`를 줄이고 다시 실행한다 |
| 전송 뒤 7번 완결성 검사에서 남은 파일이 있음 | – | `lists_after/needed_*_<i>.txt`가 비어 있지 않은 archive만 다시 받는다: 그 목록으로 `bash $X/stream_extract.sh -l $R/lists_after -r $SB -s $R/state_fix -P $P <i …>`. `$R/state`에는 그 archive가 done으로 남아 있으므로 새 state 폴더를 쓴다 |
| infos·추론·QA 단계의 버그 | – | 보관 파일로 다시 돌린다. 재전송은 필요 없다(§10-0) |

### 10-7. ReSMap 적용 (같은 추출본으로)

작성 2026-10-03. 사용자 요청: "ReSMap도 적용 가능하게끔 데이터 구성". 이 절은 데이터 구성만 다룬다. ReSMap을 미래 프레임에 실제로 돌리는 것은 공동연구자(kyungmin) 자원이 있어야 한다.

**ReSMap 입력과 캐시 방식** [실측, `$TC/resmap/README.md`, `meta.json`, `navtest/meta.json`]
- 입력은 **앞카메라 3장(CAM_L0/F0/R0) + 앞쪽으로 자른 위성 타일**이다. LiDAR와 뒤쪽 관측은 쓰지 않는다. ROI는 전방 0..32 m, 좌우 ±32 m다.
- temporal memory 모델이다. 기존 캐시는 메모리를 켠 채 **장면 순서대로** 만들었다.
  - root: navtrain `train_logs`에서 센서가 있는 126,032프레임
  - navtest: 센서가 있는 71,460프레임을 모두 장면 순서대로 돌리고 allow-list token 12,146개만 저장했다(`only_split_tokens: true`)
- 그래서 **BEVFusion용 추출본(3캠 + LiDAR)에 ReSMap 카메라 입력이 이미 들어 있다.** 위성 타일은 OpenScene에 없으므로 따로 필요하다.

**A/B 방식** [실측: 이 서버에서 `make_needed_files.py` 재실행, `summary.json`의 `resmap`. 크기는 파일 수 × archive 0 평균 크기, 괄호는 archive별 보정]

| | E2E 4 s (978 log) | navtrain 5 s (1,192 log) |
|---|---:|---:|
| 실행 log의 전체 프레임 | 573,028 | 669,588 |
| **A**: 추출 뒤 3캠이 모두 있는 프레임 (ReSMap이 장면 순서대로 돌 프레임) | 204,575 | 264,502 |
| **A**: 추가로 풀 파일 | 0 | 0 |
| **B**: 추가로 풀 프레임 / 카메라 파일 | 368,453 / 1,105,359 | 405,086 / 1,215,258 |
| **B**: 추가 디스크 | 약 231 GB (235) | 약 254 GB (259) |
| B를 포함한 보관 합계 | 약 390–394 GB | 약 481–485 GB |

- 전송량은 A든 B든 같다(2,124 GB). B는 디스크만 더 쓴다.
- B를 나중에 원하게 되면 camera archive(1,242 GB)를 다시 받아야 한다. 그래서 **스트리밍 전에 A/B를 정한다**(§10-1 3번). 기본값은 A다. 기존 ReSMap 캐시와 R_M teacher가 만들어진 조건과 같아 비교가 깨지지 않는다.
- A가 맞는지는 ReSMap의 memory 리셋 기준에 달려 있다. 기준은 kyungmin map infos의 `local_idx == 0`이고, 센서가 없는 프레임으로 생긴 간격에서 리셋하는지는 확인하지 않았다(아래 질문 1).
- B 목록: `make_needed_files.py ... --resmap-full-logs`. 옵션 없이 만든 목록은 이전 결과와 바이트 단위로 같다(E2E 4 s, navtrain 5 s에서 `diff -rq` 확인) [구현·시험됨].

**ReSMap 미래 캐시는 새 폴더에 따로 만든다** [추론, 모델 구조]
- 기존 ReSMap 캐시에 미래 프레임을 덧붙이면 안 된다. 미래 프레임이 끼면 메모리가 바뀌어, 같은 log에서 이미 캐시된 프레임의 feature도 달라진다(§2-2). 실행 log 전체를 장면 순서대로 다시 돌린다(A: 위 표 204,575 / 264,502프레임).
- navtrain 5 s를 고르면 `val_logs` token의 현재 프레임도 새 실행에 들어간다. ReSMap root에는 `train_logs`만 있기 때문이다. 이 프레임 수는 위 표의 A에 이미 포함돼 있다.
- 저장은 navtest처럼 필요한 프레임만 한다. 시간 이동 teacher target이 되는 미래 프레임 합집합은 E2E 4 s 169,248, navtrain 5 s 226,176프레임이다.

| 저장 필드 | 프레임당 | E2E 4 s | navtrain 5 s |
|---|---:|---:|---:|
| 전 필드 (`bev` 256×100×50 f16 = 2.56 MB 포함) | 약 2.73 MB | 약 462 GB | 약 617 GB |
| `bev` 제외 (`seg` 4×200×100 + `vectors` + `scores` + `labels` + `props`) | 약 0.17 MB | 약 28.5 GB | 약 38.1 GB |

- 지도 KD를 query·logit 수준(벡터, 점수, seg logit)으로 한다면 `bev` 없이도 된다 [추론]. BEVFusion의 `--drop-bev`와 같은 판단이다. 보관한 3캠이 있으므로 나중에 `bev`가 필요해지면 다시 돌리면 된다(재전송 불필요).

**kyungmin 쪽에서 받아야 하는 것** (이 서버에 없음을 확인했다. 경로는 `meta.json` 기준)

| 항목 | 위치 |
|---|---|
| 설정 | `/home/kyungmin/min_ws/rideflux/maptracker/work_dirs/resmap_nav_rideflux_stage3/resmap_nav_stage3.py` |
| checkpoint | 같은 폴더 `iter_63024.pth` (sha256 `0eaeda793402a804…`) |
| 지도 infos | `/data2/kyungmin/navsim/infos/navsim_map_infos_*.pkl`. 미래 프레임을 포함한 infos를 새로 만들어야 한다. converter도 kyungmin 쪽에 있다 |
| 위성 타일 | 출처가 문서화되어 있지 않다. 미래 프레임 위치까지 덮는지 확인한다 |
| 생성 코드 | maptracker repo의 `tools/cache_teacher_kd.py` |
| 실행 환경 | torch 1.12 env. RTX 5090(sm_120)에서는 돌지 않는다(§2-2). 새 서버 GPU에서 돌 수 있는지 확인한다 |

**kyungmin에게 물어볼 것**
1. memory 리셋 기준(`local_idx == 0`): 센서가 없는 프레임으로 간격이 생기면 리셋하는가? → A로 충분한지 판단한다. 이 답은 **스트리밍 전에** 필요하다.
2. 미래 프레임을 포함한 map infos를 만들 수 있는가?
3. 위성 타일이 navtrain 전 구간을 덮는가?
4. 실행 위치: 새 서버에서 env를 돌릴 수 있는가, 아니면 kyungmin 서버에서 돌리고 결과만 받는가? 후자라면 새로 푼 프레임의 3캠만 넘기면 된다. E2E 4 s 78,543프레임 약 49 GB, navtrain 5 s 112,007프레임 약 70 GB다. navtrain 현재·과거 센서는 kyungmin 쪽에도 있다고 가정한다(기존 캐시를 만들었으므로).

**순서**
- 스트리밍 전에는 A/B(§10-1 3번)만 정하면 된다. 질문 1의 답이 그 근거다.
- ReSMap 실행은 추출이 끝난 뒤 언제든 할 수 있다. BEVFusion 추론(§10-3)과 독립이다.
