# ego_progress 학습 라벨의 정규화 차이 — 측정과 코드 조사

> **2026-09-21 두 번 정정했다. 읽기 전에 이 상자를 먼저 볼 것.**
>
> **초판 주장(전부 철회)**: "EP head 만 아무것도 배우지 못했다", "순위 정보가 생기지 않는다",
> "정규화 관례 차이라는 설명은 기각된다", 그리고 이것을 WoTE 의 **결함**으로 규정한 것.
>
> **1차 정정**: 상관계수를 후보 256개 **전체**로 계산한 것이 원인이었다. 곱셈지표를 통과한
> 후보(장면당 약 79개) 안에서만 재면 EP head 의 Spearman 은 **+0.502** 로 정상이다
> (전체로 재면 −0.114). 실제로 고를 후보들 안에서는 잘 맞힌다.
>
> **2차 정정**: 남은 "정규화 차이"조차 결함이라 부르기 어렵다. WoTE 는 **분자에도 곱셈지표를 곱한다**
> (`pdm_scorer.py:177`), 그래서 위험한 후보의 EP 라벨이 정확히 0 이 된다 — 실제 라벨 파일의 66% 가 0 인
> 이유다. 이는 의도적 설계로 보이며, **통과 후보 구간에서 WoTE 라벨과 공식 정의의 차이는 0.017** 에
> 불과하다. 필자가 앞서 낸 0.194 / 0.718 은 **재구성을 공식 공식으로 해서 나온 잘못된 값**이다(철회).
>
> **남는 결론**: 고쳐서 버는 값의 천장은 +0.24~0.51 (EP head 출력을 정답으로 바꿔 넣고 가중치까지
> 재튜닝한 측정). 비교로 DAC head 를 정답으로 바꾸면 +3.52. **EP 라벨은 주력 과제가 아니다.**
> 아래 본문 1~3 절은 초판 그대로이므로 위 정정을 적용해 읽을 것.

2026-09-21. 대상 모델 PARA-SSR v2 (`km_para_ssr_v2_r34`), 평가 navtest 12,146 장면 공식 EPDMS.
아래 수치는 모두 공식 채점기 결과이거나 공식 평가 csv 와 대조 검증된 라벨에서 나온 것이다.

---

## 1. 무엇을 발견했나

NAVSIM 의 `PDMScorer` 는 `ego_progress` 를 **같은 호출에서 함께 채점한 후보들 중 최대 진행도**로 나눈다:

```python
# navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py
masked_progress = self._progress_raw * multiplicate_metric_scores
norm_constant_progress = np.max(masked_progress)
normalized_progress = np.clip(self._progress_raw / norm_constant_progress, 0.0, 1.0)
```

공식 평가는 `[PDM planner 궤적, 에이전트 궤적]` **둘만** 채점하므로 분모가 사실상 PDM planner 다
(`navsim/evaluate/pdm_score.py`). 반면 학습 라벨을 만들 때 궤적 사전(vocabulary) 전체를 **한 번에**
채점하면 분모가 "그 사전에서 가장 멀리 가는 안전한 궤적"이 되어, 라벨 값이 통째로 눌리고 순위도 바뀐다.

**우리 v2 는 WoTE 가 공개한 navtrain 라벨을 그대로 학습에 썼고, 그 라벨이 이 방식으로 만들어졌다.**

## 2. 그 결과 — EP head 만 아무것도 배우지 못했다

같은 계산으로 다섯 채점 head 를 모두 비교했다(장면 안 256 후보에 대한 상관의 평균):

| head | 앵커 라벨과의 상관 | refined 라벨과의 상관 |
|---|---|---|
| NC | **+0.603** | +0.508 |
| DAC | **+0.889** | +0.645 |
| **ego_progress** | **−0.028** | −0.039 |
| TTC | **+0.569** | +0.459 |
| comfort | **+0.720** | +0.746 |

- 네 head 는 0.5~0.89 로 학습됐는데 **EP 만 0** 이다. 배치 정규화 라벨로 비교해도 −0.086 으로 같다
  (즉 "정규화 관례가 달라서 상관이 낮다"는 설명은 **기각**된다).
- 모델의 EP 예측은 도달거리와 **−0.172** 로 약한 역상관이다. 실제 EP 는 도달거리와 +0.678 이다.

분포도 일치한다:

| | 평균 | 장면 내 표준편차 |
|---|---|---|
| 모델의 EP 예측 | **0.119** | 0.152 |
| 앵커의 실제 EP(공식 정의) | **0.776** | 0.170 |
| **navtrain 학습 라벨의 EP 타깃** | **0.130** | – |

학습 타깃 분포: **정확히 0 이 66.1%**, 0.9 이상이 1.1%(장면당 중앙값 2개/256). 대부분이 0 인 타깃을
BCE 로 학습하니 "전부 0 근처"가 손실 최소해가 되고, 순위 정보가 생기지 않는다.
**모델은 잘못된 타깃을 충실히 재현했다.**

## 2-1. 정정 — 어디까지가 사실인가

구간을 나누면 그림이 달라진다.

| 구간 | 학습 라벨 | 공식 EP | 헤드가 배운 것 |
|---|---|---|---|
| 곱셈지표 **통과** (장면당 약 81개) | raw/max | raw/max — **같은 양** (평균 차이 0.017) | **맞는 신호, Spearman 0.51** |
| 곱셈지표 **실패** (나머지 약 175개) | 평균 0.020 | 평균 0.720 | 틀린 신호 |

- **실제로 고를 후보들(통과 구간) 안에서는 두 정의가 같은 값**이고, 헤드는 거기서 0.51 로 맞힌다.
  NC·TTC 헤드와 같은 수준이다. 초판의 "EP 만 0" 은 통과/실패를 섞어 계산한 탓이다.
- 틀린 신호는 **실패 후보** 쪽에 몰려 있는데, 그 후보들은 선택식의 NC·DAC 항에서 이미 탈락한다.
  **틀린 부분이 선택 결과에 닿지 않는 자리에 있다.**
- 예외: 거의 정지한 장면(최대 진행 < 5 m)은 임계값 규칙이 달라 두 정의가 크게 갈린다.
  전체 평균 차이 0.017 의 대부분이 여기서 나온다.

**고쳐서 버는 값의 천장** (EP 헤드 출력을 정답으로 바꿔 넣고 EP 항 가중치까지 재튜닝):

| EP 신호 | 최선 EPDMS 변화 | 필요한 가중치 |
|---|---|---|
| 현행 예측 | +0.05 | 20~40 |
| 현행 + 순위정규화 | +0.06 | 20 |
| **정답 EP** | **+0.51** | 160 (현행의 32배) |
| 정답 EP + 순위정규화 | +0.08 | 20 |

현행 가중치 그대로면 **+0.24**. 비교로 **DAC 헤드를 정답으로 바꾸면 +3.52** 다.

따라서 **"EP 라벨을 고쳐 재학습"을 주력으로 삼을 근거가 없다.** 고칠 가치는 있으나 상한이 0.5 점이다.
아래 3 절의 "+4.15" 와 갈리는 이유는, 그 값이 *주행 궤적 자체의 EP 가 100 이 되는* 가정이기 때문이다.
256 후보 안에서 EP 가 높은 것을 실제로 고르면 NC·DAC·LK 를 그만큼 내주므로 순증이 남지 않는다.

## 3. 이게 왜 중요한가 — EP 는 navtest 손실 1위 항목

공식 csv 산수로 EPDMS 87.80 에서 빠진 12.2 점을 분해하면(공식값을 정확히 재현):

| 지표 | 완벽해지면 얻는 점수 | 현재 평균 |
|---|---|---|
| **ego_progress** | **+4.15** | 86.29 |
| drivable_area_compliance | +3.31 | 96.07 |
| two_frame_extended_comfort | +1.57 | 83.25 |
| no_at_fault_collisions | +0.77 | 98.70 |
| 나머지 (DDC/LK/HC/TLC/TTC) | 각 +0.42 이하 | – |

그리고 이 손실은 **후보가 없어서가 아니다**: 곱셈지표를 모두 통과하는 후보 중 EP 최대는 **98.75**,
그런 후보가 없는 장면은 0.1% 뿐이다. v2 가 DAC 를 깬 477 장면에서도 **100%** 가 DAC 를 지키면서
EP 가 더 높은(95.9 vs 83.4) 후보를 갖고 있었다. 고르지 못했을 뿐이다.

선택식의 EP 가중치를 8~20 으로 키워도 +0.11 점에 그치는데(A 에서 고르고 B 에서 확인), **순위 정보가 없는
신호를 증폭했기 때문**이라는 것이 위 상관값으로 설명된다.

## 4. 다른 공개 구현은 어떻게 하고 있나 (코드 직접 확인, 2026-09-21 클론)

> **2차 정정 반영.** 아래 표는 **분모**만 본 것이라 불완전했다. **분자** 처리를 직접 확인한 결과:
>
> | 저장소 | 분자 | 분모 |
> |---|---|---|
> | autonomousvision/navsim (공식) | 원시 진행거리 | 배치 최대 |
> | **liyingyanUCAS/WoTE** | **곱셈지표로 마스킹 (이 저장소만)** | 사전 내 최대 |
> | NVlabs/GTRS | 원시 | 후보별 max(자기, PDM) |
> | William-Yao-2000/DriveSuprim | 원시 | 배치 전역 최대 |
> | swc-17/SparseDriveV2 | 원시 | 후보별 max(자기, PDM) |
>
> WoTE 의 분자 마스킹은 "위험한 후보에는 진행 점수를 주지 않는다"는 의도로 읽히며, 통과 후보 구간에서
> 공식 정의와의 차이는 0.017 이다. 따라서 아래 표의 "결함" 이라는 단어는 **과하다**. 정확히는
> "공식 평가와 다른 정규화 관례"이고, 그 영향은 이미 탈락할 후보에 몰려 있다.

| 저장소 | 라벨 생성 배치 | 정규화 분모 | 결함 |
|---|---|---|---|
| autonomousvision/navsim (공식 평가) | [PDM, 궤적 1개] | PDM planner | 해당 없음 (라벨 생성 스크립트 자체가 없음) |
| **liyingyanUCAS/WoTE** | **사전 128개만, PDM 미포함** | 사전 내 최대 | **있음 (가장 심한 형태)** |
| **William-Yao-2000/DriveSuprim** | [PDM]+16384 | 배치 전역 최대 | **있음 (아래 참고)** |
| NVlabs/GTRS | [PDM]+16384 | 후보별 max(자기, PDM) | 없음 — `PROGRESS_MODE=gen_gt` |
| stepankonev/Hydra-MDP | [PDM]+1024~8192 | 후보별 max(자기, PDM) | 없음 — `_ego_progress_compare_to_zeroth` |
| swc-17/SparseDriveV2 | [PDM]+현재 후보(~20), 온라인 | 후보별 max(자기, PDM) | 없음 — `train_mode` |
| hustvl/DiffusionDriveV2 | GTRS 라벨 재사용 / RL 은 직접 재계산 | 〃 | 없음 |
| hustvl/DiffusionDrive v1 | PDM 점수 타깃 없음 | – | 해당 없음 |

**직접 확인한 근거** (로컬 클론에서 grep):

- WoTE `navsim/evaluate/pdm_score.py:191` — `trajectory_states = np.concatenate(pred_states_list, axis=0)`
  이고, PDM 궤적을 포함하는 줄은 **192~193 행에 주석 처리**되어 있다. 채점기도 패치가 없다(참조 0건).
- GTRS `pdm_scorer.py:240` — `if os.environ.get('PROGRESS_MODE', 'eval') == 'gen_gt':`
- stepankonev/Hydra-MDP `pdm_scorer.py:115,238` — `_ego_progress_compare_to_zeroth`
- SparseDriveV2 `pdm_scorer.py:84,115` — `train_mode`
- **DriveSuprim**: `docs/score_generation.md:36,50` 과 `scripts/.../gen_training_full_score_aug.sh:8`
  (`export PROGRESS_MODE='gen_gt'  # IMPORTANT!!!`) 이 설정을 지시하지만, **`.py` 파일에서 그 환경변수를
  읽는 코드가 0건**이다. GTRS 파이프라인과 문서를 가져오면서 채점기 패치만 빠진 것으로 보인다.

즉 **보편적인 문제도, WoTE 만의 문제도 아니다.** 대부분의 구현이 이 문제를 알고 명시적으로 패치했고
(플래그 이름이 `gen_gt`, `compare_to_zeroth`, `train_mode` 로 의도가 분명하다), **WoTE 와 DriveSuprim 만
결함을 안은 채 공개되어 있다.**

## 5. 확인되지 않은 것 (과장 방지)

- **고치면 점수가 얼마나 오르는지는 아직 모른다.** 라벨을 올바른 정규화로 다시 만들고 재학습해야 한다.
  EP 헤드가 순위를 배우지 못한 것은 측정됐지만, 그것이 4.15 점 중 몇 점으로 환산되는지는 미측정이다.
- WoTE 자신의 EP 가 특별히 낮은지도 결론이 아니다. 우리 측정으로 WoTE 86.08 / v2_r34 86.29 /
  v1(앵커 채점 없는 회귀 모델) 87.50 이고, 공개 표 기준 Hydra-MDP++ 83.1, DiffusionDrive 87.4,
  SafeDrive 88.6 이다. EP 는 모델 전반에서 가장 약한 가중 지표이지, WoTE 계열만의 현상으로 보기 어렵다.
- DriveSuprim 의 결함은 **공개 저장소 코드 기준**이다. 논문 수치를 낼 때 다른 내부 코드를 썼을 가능성은
  배제할 수 없다.

## 6. 다음에 할 일 (2-1 절 반영해 수정)

1. **EP 라벨 재생성은 주력이 아니다.** 상한이 +0.24~0.51 이다. 다만 라벨 생성기가 PDM planner 행과
   `progress_raw_m` 을 함께 저장하므로 **재채점 없이** 정규화만 바꿔 만들 수 있다 — 비용이 낮으니
   다른 작업에 곁들여 한다.
2. **DAC 가 우선이다.** 헤드를 정답으로 바꿀 때 **+3.52** 로 EP 의 7 배다.
3. WoTE·DriveSuprim 의 라벨 결함 보고(4 절)는 그대로 유효하다. 다만 "그래서 점수가 크게 오른다"는
   주장은 **하지 않는다** — 우리 모델에서 측정한 천장이 0.5 점이다.

## 7. 재현

```bash
# 라벨 채널 확인 (채널 2 만 연속값 = ego_progress)
python - <<'PY'
import numpy as np
x = np.asarray(np.load("data/planning_vb/pdm_score_256.npy", mmap_mode="r")[:8000], np.float32)
for k in range(5):
    v = x[:, k]; print(k, v.mean(), (( v<0.01)|(v>0.99)).mean())
PY

# 예측-라벨 상관 (head 별)  : tools/plan_v2 의 clean_compare.epdms 와 navtest 라벨 사용
# 손실 분해                  : work_dirs/eval_epdms/_exp/para_ssr_v2_r34/*/*.csv 산수
```

**파일**
- 학습 라벨: `/home/external-user/kyungmin/SSR-v2/data/planning_vb/pdm_score_256.npy` (WoTE 공개)
- navtest 공식 라벨: `.../navtest_anchor_scores.npz`, `.../navtest_refined_scores.npz` (검증 완료)
- v2 예측 덤프: `.../work_dirs/eval_epdms/diag_para_ssr_v2_r34/dump.npz`
- 공식 평가 csv: `.../work_dirs/eval_epdms/_exp/para_ssr_v2_r34/*/*.csv`
