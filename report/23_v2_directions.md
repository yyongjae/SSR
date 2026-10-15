# 23. v2 이후 방향 후보

2026-09-19 작성. v1 증류 실험의 결론은 [report/22 §12](22_planning_readout_results.md)에 있다. 이 문서는 v2(`para_ssr_v2_r34`, `para_ssr_v2_r50`, 학습 중) 결과가 나온 뒤 무엇을 볼지, 무엇을 시도할 만한지를 정리한다. 아직 아무것도 결정하지 않았다.

---

## 0. v1에서 배운 것 (방향을 고를 때의 제약)

1. **NAVSIM은 학습 때 GT 감독이 매우 풍부하다.** 사람 궤적이 있고, 어떤 궤적이든 PDM scorer로 채점할 수 있고, GT map/박스가 있다. teacher가 "GT보다 더 주는 것"이 없으면 증류는 억지다.
2. **구조가 다른 teacher의 BEV feature를 흉내 내는 증류는 정보를 옮기지 못했다**(다섯 번 모두 map probe 불변).
3. **v1 PDMS의 병목은 인지가 아니라 planning이었다.** map 오인식은 DAC 실패의 3.6%, det 놓침은 NC의 3.5%와 TTC의 13%.
4. **planner가 자기 인지 결과를 따르지 않는다.** DAC 이탈 지점의 80%에서 모델의 map head는 이미 그곳을 도로 밖으로 봤다.
5. **출력 설계가 점수를 좌우한다.** heading 결함 하나가 PDMS 1점이었다. PDM은 궤적을 LQR로 추종해 채점하므로, 추종 가능한 궤적을 내는 것이 중요하다.

## 1. v2 결과가 나오면 먼저 볼 것 (도구는 준비됨)

| 확인 | 방법 | 무엇을 결정하나 |
|---|---|---|
| PDMS / EPDMS, v1 대비 | 자동 대기열 (`run_v2b.sh`, `epdms_after.sh`), v1은 heading 보정 85.93 기준 | v2가 기준 모델로 쓸 만한가. r50 vs interaction_final = planner 효과 |
| 실패 원인 분해 | `data/analysis/failure_attribution.py` (v2 궤적 추출 + aux 평가 필요) | 남은 실패가 인지 / motion / planning 중 어디인가 |
| LK 실패 원인 분해 | `SSR-v2/tools/plan_v2/lk_attribution.py` (EPDMS csv + 궤적 dump). control: 461건 중 차선 안 치우침 52%, 추종 뒤에만 실패 37%, 다른 차선 11% | LK 실패가 map(centerline) 문제인가, 횡방향 위치·추종 문제인가 |
| "자기 인지 무시"가 남았나 | `boundary_precision.py` | §2-A를 할 가치가 있는가 |
| 점수 head의 정확도 | anchor별 예측 DAC/NC/EP vs 실제 PDM 점수 (navtest는 라벨이 없으므로 선택된 궤적과 상위 k개만 채점) | 선택 오류가 점수 예측 오류에서 오는가, offset에서 오는가 |
| v2 BEV의 map probe | `probe_map_full.py` | DAC 점수 학습이 BEV의 map 정보를 바꿨는가 |

30 epoch 최종 체크포인트에서만 본다(중간 체크포인트의 진단은 불확실하다).

## 2. 방향 후보

각 후보에 **왜 될 수 있는지 / 왜 안 될 수 있는지 / 새로움 / 먼저 할 싼 확인**을 적었다.

### A. 인지를 "읽는 방식"을 증류한다 (privileged planner teacher)

- 내용: GT 박스, GT 미래 궤적, GT map 벡터를 student의 det/map token과 같은 형태로 넣은 planner(v2와 같은 구조, 이미지 없음)를 teacher로 학습한다. student planner에는 (1) anchor query가 어떤 인지 요소를 보는지(attention, GT↔예측 Hungarian 매칭으로 대응), (2) planner 층을 지난 anchor query 표현, (3) 256개 anchor의 연속 점수를 증류한다.
- 될 수 있는 이유: GT 라벨은 정답 궤적만 주고 "인지를 어떻게 써야 하는지"는 주지 않는다. §0-4가 v2에도 남아 있으면 이 신호는 GT로 대체되지 않는다. teacher와 student의 planner 구조와 anchor가 같아 1:1로 대응된다(§0-2의 실패 원인이 없다).
- 안 될 수 있는 이유: v2의 PDM 점수 학습이 §0-4를 이미 줄였을 수 있다. teacher는 깨끗한 인지만 봐서 잡음 섞인 student 인지에 그대로 맞지 않을 수 있다(DistillDrive가 겪은 문제, teacher 입력에 잡음을 넣어 완화).
- 새로움: 구조는 새롭지 않다(LBC, Roach, DistillDrive, Hydra-MDP). 새로움은 "planner가 자기 인지를 무시한다"는 정량 진단 + 그 지표를 직접 겨냥한 증류 + 여러 모델에서의 일반성에서만 나올 수 있다. 진단이 v2나 다른 모델에서 성립하지 않으면 접는다.
- 먼저 할 확인: (i) v2에서 §1의 "자기 인지 무시" 지표. (ii) teacher만 먼저 학습해 **인지가 완벽할 때 이 planner가 몇 점인지**(상한선)를 본다. 이미지 backbone이 없어 수 시간이면 된다. 상한이 student와 비슷하면 접는다.

### B. 증류가 본질적으로 유리한 설정으로 문제를 옮긴다

- 내용: GT 감독이 부족하거나 입력이 제한된 조건에서 teacher를 쓴다. 후보: (1) navhard / 합성·교란 장면에서의 강건성(v1은 stage 2에서 DAC 0점 장면이 28~30%), (2) 카메라 수·해상도·frame을 줄인 student, (3) 라벨 일부만 쓰는 semi-supervised 설정.
- 될 수 있는 이유: 이런 조건에서는 "GT가 있는데 왜 teacher를 쓰나"가 성립하지 않는다. ResMap은 위성 prior와 긴 시간 memory를 써서 가려짐과 분포 이동에 강하다.
- 안 될 수 있는 이유: navhard two-stage는 v2 planner를 용재의 navsim_v2 평가 저장소에 옮겨야 하고, stage 2 점수는 구조적으로 낮고 분산이 크다. 문제 설정을 바꾸는 것이라 연구 목표 자체의 결정이 필요하다.
- 새로움: 설정에 따라 중간. "E2E planner의 분포 이동 강건성을 인지 전문가 증류로 올린다"는 주장은 NAVSIM v2에서 아직 드물다(문헌 확인 필요).
- 먼저 할 확인: v2의 navtest EPDMS와 navhard stage 1/2 점수, stage 2 실패의 원인 분해(인지 붕괴인가 planning인가). 인지 붕괴가 크면 이 방향의 근거가 된다.

### C. 추종을 고려한 planning (tracking-aware)

- 내용: PDM은 LQR + bicycle로 궤적을 추종해 채점한다. 출력 궤적을 미분 가능한 추종 시뮬레이션에 통과시켜, **추종된 궤적**에 DAC 여유폭 벌점(`plan_map.py`, margin 0.3 m)과 모방 loss를 건다. v2에서는 offset이 반영된 최종 궤적에 적용한다.
- 될 수 있는 이유: v1 DAC 실패의 42%가 추종 시뮬레이션에서만 드러났고, 이탈 깊이 중앙값이 0.14 m다. v2의 DAC 라벨은 고정 anchor에 대한 이진값이라 offset 뒤의 스침을 구분하지 못한다.
- 안 될 수 있는 이유: heading을 경로에서 계산하는 v2에서는 추종 이탈이 이미 크게 줄었을 수 있다. 미분 가능한 LQR 구현이 필요하다.
- 새로움: 중간. 증류가 아니다.
- 먼저 할 확인: v2의 DAC 실패 중 "궤적 그대로는 재현 안 됨"의 비율(§1 실패 원인 분해에서 바로 나온다).

### D. 임의 궤적에 대한 연속 metric 점수 (critic 확장)

- 내용: WoTE/Hydra는 고정 vocabulary의 이진 PDM 라벨만 쓴다. GT map의 SDF와 GT 차량 미래 박스로, offset이 반영된 궤적과 교란한 궤적에도 **경계까지의 여유, 최소 차간 거리, 진행 거리** 같은 연속값을 학습 중에 바로 계산해 점수 head를 학습시킨다.
- 될 수 있는 이유: 점수 head가 실제 출력 궤적을 평가하게 되고, 경계 스침을 구분한다. TOAD 식 추론 시점 탐색(§E)의 scorer가 anchor 밖에서도 믿을 만해진다.
- 안 될 수 있는 이유: proxy 점수와 실제 PDM 점수의 차이(추종 시뮬레이션, 면제 규칙). 증류가 아니다.
- 새로움: 중간.
- 먼저 할 확인: navtrain에서 proxy DAC/NC와 WoTE의 PDM 라벨의 일치율(anchor 256개에 대해 바로 잴 수 있다).

### E. 추론 시점 탐색 (TOAD 식 CEM)

- 내용: v2의 점수 head를 scorer로 써서 제어 입력 공간에서 CEM으로 궤적을 다듬는다. 코드는 `modules/kinematics.py`에 일부 있다.
- 기대치: TOAD 논문 기준 NAVSIM v1 +0.3. v1에 투영만 적용했을 때는 −2.5였다(heading이 틀려서). v2는 heading이 경로와 일치해 조건이 다르다. 후순위.

### F. det/map 성능 자체를 올리는 증류

- 내용: planning과 별개로 map mAP(27.7)와 det mAP(27.4)를 teacher로 올린다. v1에서 map 벡터를 pseudo-label로 쓰는 `map_teacher` arm은 준비만 하고 돌리지 않았다.
- 주의: §0-3 때문에 PDMS로 이어진다는 보장이 없다. 목표가 "det/map도 올린다"일 때만 의미가 있다.

## 3. 추천 순서

1. v2 최종 결과가 나오면 §1을 전부 돌린다(하루 이내, GPU 1~2장).
2. 그 결과로 A와 C 중 어디에 실패가 몰려 있는지 본다. A의 teacher 상한선 실험은 결과와 무관하게 "인지가 완벽하면 몇 점인가"에 답하므로 먼저 해 둘 가치가 있다.
3. B는 연구 목표를 바꾸는 결정이라 v2의 EPDMS/navhard 결과를 보고 따로 논의한다.
4. 어떤 방법이든 v2 하나에서만 보이지 말고 공개 기준 모델(저장소에 있는 WoTE/TransFuser) 하나에 같이 적용해 일반성을 본다. WoTE 공개 체크포인트의 평가와 궤적·map 예측 추출은 `SSR-v2/data/wote`에 준비돼 있다.
