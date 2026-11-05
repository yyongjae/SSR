# 최종 EPDMS 궤적 선택(Trajectory Selection) 방법론 및 성적 정리 (`epdms_score_final.md`)

---

## 1. 최종 방법론 요약 (Option B+)

- **주행 안전성 및 효율 예측 (NC, DAC, EP, TTC)**: 2단계 Refiner가 각 후보 궤적 주변의 BEV 특징을 바탕으로 예측한 충돌 회피($\widehat{\text{NC}}$), 도로 내 주행($\widehat{\text{DAC}}$), 진행 거리($\widehat{\text{EP}}$), 충돌 여유 시간($\widehat{\text{TTC}}$) 점수를 반영합니다. **(DrivoR, GTRS, DriveSuprim, Hydra-MDP++)**

- **자차 동역학 기반 승차감 및 시간적 연속성 (HC, EC)**: 후보 궤적 좌표로부터 가속도·저크·요레이트 기준 만족 여부($\text{HC}$)와 직전 프레임($t-1$) 선택 궤적과의 연속성($\text{EC}$)을 직접 계산해 반영합니다. **(DriveSuprim, GTRS)**

- **1단계 대표 경로와의 일관성 (CC, Path Consistency)**: 2단계에서 변형된 후보 궤적이 1단계 플래너의 1등 궤적($\tau_{0,0}$)에서 2.0초 이상 연속으로 $0.5\text{m}$ 넘게 벗어나지 않는지($\text{CC} \in \{0, 1\}$)를 평가해 경로 안정성을 유지합니다. **(Ours)**

- **핵심 안전 지표 가중치 강화 ($w_{\text{safe}}$)**: 위반 시 전체 점수가 0점이 되는 충돌·도로이탈 지표($\widehat{\text{NC}}, \widehat{\text{DAC}}$)의 가중치를 $w_{\text{safe}} = 1.50$으로 높이고, 1단계와 2단계 점수를 $\beta = 0.75$ 비율로 결합합니다. **(DrivoR, GTRS, Hydra-MDP++)**

---

## 2. 최종 수식

후보 궤적 $\tau_{k,v}$ ($k \in \{0,\dots,15\}$: 1단계 상위 16개 앵커, $v \in \{0,\dots,5\}$: 2단계 6종 변형, 총 96개 후보)에 대한 최종 선택 점수 $S(k, v)$:

$$S(k, v) = (1 - \beta)\,S_{\text{stage1}}(k) + \beta\Big(w_{\text{safe}}\big(\log \widehat{\text{NC}}_{k,v} + \log \widehat{\text{DAC}}_{k,v}\big) + \log\big(5\,\widehat{\text{EP}}_{k,v} + 5\,\widehat{\text{TTC}}_{k,v} + 2\,\text{HC}(\tau_{k,v}) + 2\,\text{CC}(\tau_{k,v}, \tau_{0,0}) + 2\,\text{EC}(\tau_{k,v}^{(t)}, \tau^{*(t-1)})\big)\Big)$$

$$\text{CC}(\tau_{k,v}, \tau_{0,0}) = \mathbb{I}\!\left[\max_{\Delta t \text{ window of } 2.0\text{s}} \min_{t \in \Delta t} \text{dist}\big(\tau_{k,v}(t), \tau_{0,0}\big) \le 0.5\text{m}\right]$$

$$\tau^{*(t)} = \tau_{k^*, v^*}, \qquad (k^*, v^*) = \arg\max_{k, v} S(k, v)$$

---



## 3. 평가 코드 위치

### 3.1 코드 경로
- **EPDMS 평가 파이프라인 메인 스크립트 (Python)**: [`tools/ck/e2e2/eval_epdms.py`](file:///workspace/byounggun/SSR/tools/ck/e2e2/eval_epdms.py)
  - 기본 파라미터: `--beta 0.75 --w-safe 1.50 --w-cc 2.0` (`Option B+` 최종 설정 내장)
  - 96개 후보 덤프(`dump_pool96.py`) $\rightarrow$ 자차 동역학(`HC, CC, EC`) 기반 궤적 선택 $\rightarrow$ 공식 `navsim_v2.2` EPDMS 채점기(`eval_ckpt_epdms.py`) 호출 $\rightarrow$ `summary.json` 저장까지 원스톱 수행
- **EPDMS 평가 실행 쉘 래퍼 (Bash)**: [`scripts/evaluation/eval_epdms.sh`](file:///workspace/byounggun/SSR/scripts/evaluation/eval_epdms.sh)
  - [`tools/release/load_paths.sh`](file:///workspace/byounggun/SSR/tools/release/load_paths.sh) 로드 및 `CK_BEV64=1`, `CK_ALLOW_GPU4567=1`, `PYTHONPATH` 자동 설정
- **통합 실험 스크립트 `eval` 엔트리포인트**: [`tools/release/eval.sh`](file:///workspace/byounggun/SSR/tools/release/eval.sh)
  - `bash scripts/experiment.sh eval RUN=<이름> GPUS=...` (`EPDMS=1`, 기본값) 실행 시 자동으로 `eval_epdms.sh` (`Option B+`: $\beta=0.75, w_{\text{safe}}=1.50, w_{\text{cc}}=2.0$) 호출
  - `bash scripts/experiment.sh eval RUN=<이름> GPUS=... EPDMS=0` 실행 시 모방 항 없는 공식 PDMS 수식($\widehat{\text{NC}}\cdot\widehat{\text{DAC}}\cdot\frac{5\widehat{\text{EP}}+5\widehat{\text{TTC}}+2\widehat{\text{C}}}{12}$)으로 `navtest` v1 PDMS 평가 수행
  - `bash scripts/experiment.sh eval RUN=<이름> GPUS=... EPDMS=all` 실행 시 v1 PDMS와 v2.2 EPDMS(`navtest` + `navhard_two_stage`) 동시 평가 수행
