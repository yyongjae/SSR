# arm d (oracle forecaster, PRIVILEGED) — 결과 전 판정 규칙 사전 등록

작성: 2026-10-12 09:25 KST. arm d는 학습 중이다(08:02 KST 시작, GPU 4·5, step 14,700 / 53,190). val 결과는 아직 아무도 보지 않았다.
사용자 승인: "진단 d 해보셈" (10-12 ~07:55 KST). 분석 전용이다. KD teacher나 plug-in 결과로는 절대 쓰지 않는다(PRIVILEGED: GT 미래 입력).

## 무엇을 묻나

b − c ≈ 0 진단(`D3/diag_bc/`, synthesis + 반박 검증 2개)의 결론은 다음과 같다.
- CA·learned 예보가 c(ck2T10)에 더해 줄 수 있는 NC/TTC 정보는 원래 작다. 같은 20 gap scalar에 train-fit 결합기를 쓰면 +0.13..+0.20이다(clean, stackview).
- 같은 emulation에서 oracle 예보는 +0.71 [+0.52, +0.92]다(d − c, ncttc_only; `D3/diag_bc/synthesis/out/s6_oracle_emul.json`).

질문: **이 학습 파이프라인(raster/gap → CNN → zero-init 주입 → critic)이 강한 예보 신호를 실제 선택 이득으로 바꾸는가?**

## 지표

**주 지표: `pdms_ncttc_only` d − b** (decide_step1, paired log bootstrap)
- NC/TTC는 각 arm 것을 쓴다. DAC/EP/C는 ck2T 고정이다.
- 그래서 양쪽 모두 남의 EP/C와 짝지어진다. 반박 검증이 찾은 일반적인 "자기 head 결합 벌점"(D_own −0.09)이 상쇄된다.

**판정**
- d − b ≥ +0.30이면 파이프라인은 강한 신호를 회수한다. 다음 레버는 forecaster 품질이다(hazard 관련 물체의 단일 궤적 정확도).
- d − b ≤ +0.15이면 파이프라인이 병목이다. branch 주입, raster station-loop, 학습 aug × CA를 먼저 고친다.
- 그 사이면 결론을 내지 않고 보조 지표로 해석한다.

**보조 지표** (보고만 하고 판정에는 쓰지 않음)
- ncttc_only d − c. 직접 계산하고, emulation 기대 +0.71과 비교한다.
- decision PDMS d − b / d − a
- lead-decel TTC AUC, NC/TTC fail AUC
- CPU path ablation(`D3/diag_bc/synthesis/scripts/s2_path_ablation.py`, `verify_mechanism/scripts/v4_objpath.py`)을 d에도 돌려, oracle 물체 정보가 raster 경로와 gap 경로 중 어디로 가는지 본다.

## 주의
- 단일 seed다. training-seed 잡음 ±0.1–0.2는 재지 않았다(사용자 결정: seed ablation 없음).
- d의 train-code sha는 a·b와 다르다. 바뀐 파일은 e2e 파일(online2.py, e2e_data2.py)과 ckutil.py의 GPU guard(CK_ALLOW_GPU45)뿐이다. teacher 학습 경로(model.py, temporal/**, train_ck2.py, data)는 aa7a7b2와 같다.
- infer_val_v3.py는 재현 검사 수정(157c8df)으로 바뀌었다. decide의 forward-code 검사가 d를 a·b와 다르다고 표시할 수 있다. 그래서 `decide/with_d/`에 공식 모드와 `--no-arm-check` 모드를 함께 남긴다.
- 체인: `D3/run_armd_chain.sh` (done.json 대기 → GPU 4 추론 → decide --d → `D3/decide/with_d/`). 공식 `D3/decide/step1.json`은 덮어쓰지 않는다.
