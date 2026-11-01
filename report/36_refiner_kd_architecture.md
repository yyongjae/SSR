# 36. 교정기(refiner) KD 아키텍처 설계 (v2, 구현 전)

> **2026-09-29 참고:** 이 문서의 단계 F는 "학습된 E2E 모델 없이" 바뀐 형태(단계 T: R_T vs R_none, 사람 궤적 교란 초안만)로 실행됐다. 실제 실행 내용, 경과, 결과는 `report/37_teacher_refiner_experiment_overview.md`를 본다.

작성 2026-09-28. v1에 대한 사용자 피드백 5개와 추가 의견 2개를 반영한 판이다.

**근거 자료** (`report/refiner_kd_design/`)
- 사실 조사 4건: `fact_*.md`
- 상세 초안: `architecture_draft_v0.md`
- 비판 검토: `critic_logic.md`, `critic_impl.md`

세부 수식과 파라미터는 상세 초안의 해당 절을 따른다. 이 문서와 다르면 **이 문서가 우선**이다.

**v1에서 바뀐 점**

| 항목 | v1 | v2 |
|---|---|---|
| 동등 결과의 결론 | "teacher 불필요" | "현재 고정 feature·교정기 조건에서 teacher의 교정 우위가 없음". E2E 공동학습의 이득(예: 초기 수렴)은 부정하지 않음 |
| E2E 진행 조건 | 고정 feature KD가 GT-current KD 이상이어야 함 | 필수 조건에서 뺌. GT-current는 비교 기준으로만 둠. 고정 feature 전이 결과와 E2E 공동학습 결과를 따로 해석 |
| 통로 밖 정보 | 통로 샘플링만 적음 | 전체 BEV를 읽는 전역 경로를 명시 (§2 M3) |
| τ0 stopgrad | "τ0 → refiner 입력 sg" | teacher 입력, student 입력, 대리 손실 경로별로 명시 (§2 M10) |
| navtest 사용 | F→E 판정에 사용 | 진행 판단과 설정 선택은 **내부 dev(val_logs 교차 적합)**에서만. navtest는 고정된 설정의 최종 보고 1회 |
| teacher 비대칭·독립 student 부재 | blocker | **해석상의 한계** (§5). dropout 완화책은 뺌. dropout은 사전학습 노출 차이를 없애지 못함 |
| 단계 F 범위 | arm 11개, 약 13일 | **핵심 3 arm (R_none, R_S, R_T)**, 약 8일. E2E 구현을 병행해 E 검증을 앞당김. 나머지 arm은 첫 결과가 남긴 질문에 맞춰 추가 |

---

## 0. 전체 구조

```
[단계 F: 고정 feature에서 교정기 비교 — E2E 학습 없음]
 M1 초안 은행: 사람 궤적 교란 + student 실제 초안(val_logs) + 공식 채점 라벨
 입력 feature ─M2 어댑터─► 공통 [64,50,100] ─┬─ M3a 통로 샘플(초안 경로 주변, 도착 시각 포함)
      (teacher BEV / student BEV / 없음)      └─ M3b 전역 경로(BEV 전체를 3.2 m 칸 200 token으로)
                                        └──► M5 refiner ─► gate + 종·횡 교정량 ─M4 디코더─► τ1
 학습: M7 대리 손실(M6 미래 물체 전체 · 도로 SDF · 진행 · 편안함 · 수정량 · DDC) + gate BCE
 검증: M8 대리 손실 ↔ 공식 판정 (dev에서 먼저)
 판정: M9 dev(val_logs 교차 적합) 실제 초안 → 공식 채점 → 진행 여부

[단계 E: E2E 공동학습]
 카메라 → PARA-SSR(처음부터) → 초안 τ0 ─┬─► student refiner R_S^E ─► τ_final
                                          └─ sg ─► R_T(고정, teacher BEV) ─► Δτ_T (KD target)
[최종 보고] 설정을 고정한 뒤 navtest 1회
[추론] PARA-SSR + student refiner (teacher·GT 불필요)
```

---

## 1. 단계 F 구성 (핵심 3 arm)

| arm | 입력 | 질문 |
|---|---|---|
| **R_none** | 초안 + ego 상태만 (장면 feature 없음) | 장면을 보지 않는 학습형 교정의 몫 |
| **R_S** | student BEV (초안을 만든 모델, 고정) | student 자신의 표현으로 할 수 있는 교정 |
| **R_T** | teacher BEV (BEVFusion 50×100, 고정) | teacher 표현이 교정에 주는 추가 몫 |

- 세 arm은 **어댑터 첫 층만 다르고** 나머지 코드, 하이퍼파라미터, 초안 은행, seed 목록이 같다.
- seed는 arm당 3개다. 부족하면 2개로 줄인다.
- **추가 arm은 첫 결과를 본 뒤에 정한다.** 첫 결과가 남긴 질문에 맞춰 고른다.

| 후보 arm | 붙이는 경우 |
|---|---|
| R_GT-cur (현재 GT raster) | 이득의 상한이나 인지 격차를 묻고 싶을 때 |
| R_GT-fut (미래 GT, oracle) | 미래 정보의 천장을 묻고 싶을 때 |
| R_S←KD(R_T) (고정 feature KD) | 고정 feature에서도 전이되는지 묻고 싶을 때 |
| R_S self-KD | KD 자체의 정규화 효과를 분리하고 싶을 때 |
| R_S′ (다른 seed student) | "다른 모델이라서"를 분리하고 싶을 때 |
| R_{T⊕S} (teacher와 student feature 결합) | teacher가 student에 더하는 몫을 직접 재고 싶을 때 |

---

## 2. 모듈 명세 (v1과 달라진 곳 위주)

### M1 초안 생성기
- **교란 방식:** 가산 가속 오프셋 family를 쓴다. 균일한 ×배율은 쓰지 않는다. 초안은 **M4 디코더와 같은 기저 공간에서 직접 샘플**한다.
- **추가 family:** heading drift. 실제 초안의 8번째 점 heading 오차는 p90 4.0°로, 사람 궤적 기반 교란(약 1°)보다 크다.
- **실제 초안 출처:** val_logs(student가 학습에 쓰지 않은 기록)에서 student가 낸 궤적이다. 학습과 dev 평가 모두에 쓰고, 교차 적합 fold로 나눈다.
- **교란 범위 수치:** navtest 분석에서 나온 값이다. 동결 전에 val_logs 실제 초안으로 다시 계산한다.

### M2 입력 어댑터
- **정규화:** 채널별 z-score를 모든 arm에 동일하게 적용한다. 통계는 학습 fold에서 구한다.
- **구조:** 1×1 conv 두 층으로 [C → 128 → 64]. R_none은 0 입력이다.

### M3 장면 읽기 경로 — 두 개 (③ 반영)
- **M3a 통로 경로 (세밀)**
  - 초안 경로(C2 spline)를 따라 48개 지점을 둔다. 지점 간격은 max(1 m, 전방 길이/48)이다.
  - 전방 길이는 초안 끝 + max(8 m, 끝 속도 × 1 s)다. TTC 투영 구간까지 포함하기 위해서다.
  - 좌우는 ±4.8 m를 0.6 m 간격 17칸으로 샘플한다.
  - 기하 채널 6개를 붙인다: 격자 안 여부, 경로 안 여부, 호장, 옆 거리, **초안의 도착 시각**, 도착 속도.
- **M3b 전역 경로 (전체 BEV, 거침)**
  - 50×100 BEV **전체**를 5×5 평균 pool해 3.2 m 칸 200개 token을 만든다. 위치 임베딩을 더한다.
  - refiner가 매 층에서 cross-attention으로 읽는다.
  - 통로 밖에 있다가 나중에 들어올 물체(옆 차선 끼어들기, 교차로 진입 등)는 이 경로로 **현재 시점의 모습**을 본다.
- **모든 arm이 똑같이 못 보는 것**
  - BEV 격자(전방 0–32 m, 좌우 ±32 m) 밖의 물체
  - t=0에 아직 나타나지 않은 물체
  - 이 경우는 결과를 층화해 보고한다(v1 표: 사건 시점 원인 물체가 32 m 밖인 NC 41건, t0에 없던 원인 물체 11건).
- **v1 초안 대비:** 전역 경로는 v0 초안 M3에 있었지만 v1 요약에서 빠졌다. v2에서 명시한다.

### M4 교정 디코더
- **종방향:** 가속도 제어점 6개를 A·tanh(z)로 직접 제한하고 적분한다. |δa| ≤ A가 정확히 성립한다.
- **횡방향:** Frenet 옆 거리 d(s)이며, d(0) = d′(0) = 0이다. 곡률 투영은 학습과 추론에 같게 적용한다.
- **항등 보장:** v ≥ 0은 relu로 맞춘다. gate가 꺼지면 τ0 원본을 그대로 낸다.
- **모드:** 기본은 감속 전용(s1 ≤ s0)이다. 제한 가속은 분석용 부 arm이다.

### M5 refiner
- **입력:** 초안 token, 통로 정거장 token 48개, 전역 token 200개(M3b)
- **구조:** transformer 4층, d = 192. 각 층은 [초안 + 정거장] 자기 attention → 전역 cross-attention → FFN이다. 약 3.4M 파라미터다.
- **출력:** gate p_g, 종방향 제어점 6개, 횡방향 제어점 6개

### M6 미래 물체 빌더
- **출처:** 로그 주석, 0–5 s, 360°. 반경은 max(80 m, 경로 길이 + 25 m)다.
- **검증:** metric cache와 모서리 오차 0으로 재현됨을 확인했다.
- **UNKNOWN 처리:** 비율을 먼저 잰다. 해당 초안은 진행 보상에 상한을 두고, gate 음성 라벨의 가중치를 낮추고, 결과를 층화한다.

### M7 대리 손실
- 충돌(차체 사각형 사이 분리 거리), 도로 SDF, **DDC**, 진행, 편안함, 수정량, gate BCE로 구성한다.
- **진행:** 공식 규칙에 맞춘다. EP는 PDM-Closed 기준으로 정규화하고, 최대 진행이 5 m 이하면 EP = 1이다.
- **편안함:** 0.5 s keyframe과 해석적 성분으로 계산한다.

### M8 대리 손실 검증 (학습 전)
- 설정은 **dev 풀**(val_logs와 E navtrain 풀)에서만 고른다.
- 합격 기준: NC recall ≥ 0.70, 사람 궤적 오경보 ≤ 2%, 쌍 단위 P(공식 해소 | 대리 해소) ≥ 0.6
- 미달이면 미분 가능한 추종 근사(M7b)를 넣는다.

### M9 평가 (⑤ 반영)
- **dev 평가 (진행 판단, 설정 선택)**
  - val_logs 18,179장면을 도시별로 층화한 log 단위 5-fold 교차 적합으로 나눈다. fold마다 refiner를 학습하고 나머지 fold의 **실제 student 초안**에 적용한다.
  - 그 결과를 공식 채점한다. 먼저 val_logs 채점 캐시를 만들어야 한다.
  - dev의 NC 실패는 약 270건으로, navtest(279)와 규모가 비슷하다. [navtrain E 실측 1.5% 기준]
- **최종 보고 (navtest)**
  - 설정을 모두 고정한 뒤 1회 평가한다. 어떤 선택에도 쓰지 않는다.
- **채점기:** [PDM-Closed, 초안 1..K]를 한 번에 추종·채점한다. EP와 DDC는 공식식으로 다시 계산하며, 140/140 일치를 확인했다.

### M10 E2E 통합: 기울기 경로 명시 (④ 반영)

| 경로 | 설정 | 결과 |
|---|---|---|
| τ0 → teacher refiner R_T 입력 | **sg** (R_T는 고정) | KD target이 student를 따라 움직이지 않음 |
| τ0 → student refiner R_S^E 입력 | **기본: sg** | KD와 대리 손실의 기울기는 R_S^E와, R_S^E가 읽는 student BEV를 거쳐 encoder에만 간다. **기존 planner 출력층에는 직접 가지 않는다** |
| student BEV → R_S^E | 기울기 배율 γ_bev = 0.1 (1도 비교) | teacher의 교정 지식이 student 표현으로 들어가는 통로 |
| 대리 손실(τ_final) → τ0 | 기본: 막음 | planner는 기존 모방 손실로만 학습 |
| τ0 ← planner의 기존 손실 | 그대로 | 기존 PARA-SSR 학습 유지 |

**τ0를 sg로 막을 때의 의미와 대안**
- **기본값(sg)의 의미:** 교정 지식은 **refiner와 encoder**에 담긴다. planner 출력층은 모방만 배운다. 추론 때 refiner가 항상 붙어 있으므로 성능 목적에는 충분하다. 단 "planner 출력층 자체가 교정 지식을 배웠다"는 주장은 할 수 없다.
- **대안 1 (접어 넣기, fold-back):** L_fold = ‖τ0 − sg(τ_final)‖를 작은 가중으로 추가한다. planner 출력층이 교정 결과를 따라가도록 해서 교정 지식을 planner에 직접 넣는다. refiner 없이도 추론할 수 있는지 볼 수 있다.
- **대안 2 (sg 해제):** τ0를 통해서도 기울기를 흘린다. planner가 "교정하기 쉬운 초안"을 내도록 refiner와 서로 맞춰질 위험이 있어 분석용으로만 둔다.
- **추천 순서:** 기본(sg) → 대안 1 → 대안 2

**KD 손실과 일정**
- KD 손실은 제어점 공간에서 계산한다. 클리핑 전 값 기준으로 L1이다.
- 신호 소멸 방지: 학습 후반에는 student 초안이 좋아져 교정 target이 0에 가까워진다. 샘플 일부(예: 50%)는 sg(τ0)에 M1 교란을 걸어 R_S^E와 R_T에 함께 넣는다.
- λ_KD 일정: epoch 0–4는 0, 5–9는 선형 증가, 이후 고정. 초기 초안이 R_T 학습 분포 밖이기 때문이다.

**E2E arm**

| arm | 구성 | 가르는 것 |
|---|---|---|
| E0 | 기존 PARA-SSR | 기준. seed 0은 이미 있음 |
| E1 | + R_S^E, 대리 손실만 | refiner와 GT 대리 손실의 효과 |
| E2 | + R_S^E, 대리 손실 + KD(R_T) | teacher 교정 KD의 효과 (E1 대비) |
| (선택) E3 | + KD(R_GT-cur) | 비교 기준. 진행 조건이 아님 |

---

## 3. 판정 규칙 (① ② ⑤ 반영)

결과를 보기 전에 `PRESTATED_DECISION_RULE.txt`로 고정한다.

**F → E 진행 판단 (dev, 교차 적합 out-of-fold, seed 평균, log 단위 짝지은 bootstrap)**
- 주 지표 (R_T − R_S, 미리 정한 진행 손실 예산에서):
  - 공식 PDMS
  - 전체 실패: NC(+TTC)가 주 지표이고, DAC·DDC는 비열등성 조건이다. teacher BEV는 지도 정보가 약하기 때문이다(centerline probe IoU 0.07).
  - 새 실패
- 기준선 점검: R_S와 R_T가 R_none보다 나아야 장면 정보가 쓰인다고 말할 수 있다.

| dev 결과 | 결론 (범위를 정확히) | 다음 |
|---|---|---|
| R_T > R_S | 고정 feature·교정기 조건에서 teacher 표현이 교정에 우위가 있음 | 단계 E |
| 동등 (구간이 ±Δ_eq 안) | **현재 고정 feature·교정기 조건에서 teacher의 교정 우위가 없음.** E2E 공동학습의 이득(초기 수렴, 표현 형성)은 이 결과로 부정되지 않음 | 단계 E를 할지는 사용자가 판단. 한다면 E1 대 E2가 그 질문에 답함 |
| 불확정 | 검정력 부족 | seed 추가 |
| R_T < R_S | 고정 조건에서 teacher 표현이 교정에 더 불리 | 원인 분석(어댑터, 정규화, 지도 정보) 후 판단 |

- **GT-current 결과는 진행 조건이 아니다.** 붙였다면 격차 해석의 참고로만 쓴다.
- **고정 feature KD arm 결과도 진행 조건이 아니다.** 붙였다면 "고정 feature에서의 전이"로 보고하고, E2E 공동학습 결과와 섞지 않는다.
- **navtest는 진행 판단에 쓰지 않는다.** 단계 F·E의 최종 설정을 고정한 뒤 1회 보고한다.
- **검정력:** 32-D 기준 표준오차는 약 8건이다. 80% 검정력의 최소 검출 효과는 약 23건이다. dev도 규모가 비슷해 같은 한계가 있다.

---

## 4. 구현 비버그 수정 사항 (v1 §2 유지)

| 항목 | 수정 |
|---|---|
| spline 가속 한계 | 초안 명세는 최대 12 m/s²까지 나갔다. 가속도 제어점을 직접 제한해 실측 최대 4.00 |
| softplus | 4초간 0.83 m 밀림 → relu |
| 편안함 | 0.1 s 보간에서는 사람 궤적도 jerk 99.9% 위반 → 0.5 s keyframe과 해석적 계산 |
| DDC | 곱셈 지표에 포함 |
| 배치 채점 | 기존 함수는 EP·DDC가 틀림 → 새 채점기, 140/140 일치 |
| EP 임계값 | 5 m (`default_scoring_parameters.yaml`) |
| 정규화 | 채널별 z-score 동일 적용 |
| 캐시 | `/home/external-user/ssd/yongjae_refiner/` (1.6 TB 여유). teacher 로더는 manifest checkpoint sha로 고정 |
| kyungmin 코드 | import하지 않고 hash를 기록한 사본 사용 |

---

## 5. 해석상의 한계 (blocker 아님)

- **teacher 사전학습 노출**
  - BEVFusion teacher는 navtrain 전체로 학습했다. 단계 F 학습·dev 풀(val_logs)에서 teacher feature는 본 데이터, student feature는 처음 보는 데이터다.
  - 보고할 것: 분할별 teacher·student 검출 재현율, dev와 navtest의 성능 차.
  - 완화책은 두지 않는다. dropout이나 잡음으로는 노출 차이가 사라지지 않는다.
- **독립 student 부재**
  - R_S는 초안을 만든 모델의 feature를 읽는다. 그래서 R_T > R_S에는 "다른 모델이라서"의 몫이 섞일 수 있다.
  - 주장은 "초안을 만든 student 표현 대비"로 적는다. R_S′는 필요할 때 추가한다.
- **navtest 유래 설계값:** 교란 범위 등은 동결 전에 val_logs로 다시 계산한다. 남는 값은 [navtest 유래]로 표시한다.

---

## 6. 일정 (핵심 3 arm, E2E 병행)

| 일 | 단계 F | 단계 E 병행 |
|---|---|---|
| 0.5 | 좌표 확인, val_logs 채점 캐시 시작(CPU 약 2 h), student dump(val_logs + navtest) | – |
| 1–2 | `geometry.py`, `decoder.py`, 단위 시험 | – |
| 2 | 배치 공식 채점기 | – |
| 3 | M6 빌더, 도로 SDF | – |
| 4 | 초안 은행과 채점 | – |
| 5 | M7 대리 손실, M8 검증 | – |
| 6–7 | M2/M3a/M3b/M5, 학습·평가 루프 | M10 구현 시작 (flag 뒤, 꺼지면 기존과 동일함을 시험) |
| 8 | R_none / R_S / R_T × seed 3 학습, dev 판정 | E1·E2 스모크 |
| 9– | 필요한 추가 arm | E1·E2 × seed 2 학습 (25–38 h × GPU 2장/run) |

- E 학습의 시작 시점은 GPU 확보에 달려 있다. 지금은 6장이 모두 사용 중이다.

---

## 7. 기존 작업과의 관계 (v1 §6 유지)

- **kyungmin feasibility:** GT SDF 비용을 anchor 후보에 거는 학습이다. E1(대리 손실만)과 관련된 구성요소다.
- **kyungmin readout KD:** 지도 teacher에서 나온 유사한 위험 신호다.
- **byounggun Stage 1·2:** 같은 teacher 캐시를 읽는 planner와 feature KD다. PDM 채점과 같은 설정의 대조군은 찾지 못했다.
- **32-B:** 해소 128, 새 충돌 181로 순감소가 아니었다. 그래서 새 실패를 주 지표에 넣었다.
- **ThinkTwice·DistillDrive:** 구조 자체는 선행연구에 있다. 기여는 교정 출력의 teacher/student 비교와 그 증류로 한정한다. 새로움은 별도 조사가 필요하다.
