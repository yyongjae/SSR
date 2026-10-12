# refiner-v3 Step 1 결과 — temporal DET teacher arm a/b/c 판정

작성: 2026-10-12 02:55 KST. 갱신 06:55 KST (user 승인으로 재현 허용치 수정 → 공식 판정). 단일 seed 최종 판정
(user: seed ablation 하지 않음).
산출물: **`D3/decide/step1.json` + `D3/decide/step1_official_20261012.log` (공식, NO-GO)**; 이전 기록
`D3/decide/step1_official_refused.log` (첫 공식 실행, REFUSED), `D3/decide/step1_diag_no_arm_check_20261012`
(`--no-arm-check` 진단 JSON). D3 = `/home/external-user/ssd/yongjae_refiner/ck/v3`.

## 1. 결론

- **공식 decide: `NO-GO` (final, 단일 seed)** — G1·G2 미달. 06:51 KST, rc 0, 진단 판정과 수치 동일.
- 첫 공식 실행(02:34)은 성능이 아니라 arm a의 재현 무결성 검사에서 REFUSED → 허용치 수정 (§2) 후 재실행.
- 그래도 e2e KD 전체 학습(`ck2e2e_v3T`)은 user 지시("det, map 다 끝나면 알아서 e2e 모델에 KD까지")대로 GO 여부와 무관하게
  launch함 (STEP2_E2E.md §7).

## 2. 무결성 검사 실패 원인 (arm a reproduce)

- 첫 규칙: 재계산한 per-token logit과 학습 중 마지막 val eval logit의 차이 q99.9 |dlogit| ≤ 2e-4, flipped pick ≤ 0.1 %.
- 실측 (arm a): q99.9 |dlogit| 0.0039 (= 2^-8), max 0.0156 (= 2^-6), 차이 나는 logit 0.81 % / token 9.3 %,
  flipped pick 2 / 18,179 (0.011 %), PDMS 재현 0.90288 (absdiff 0). arm b: q99.9 0, max 0.027, 차이 token 14개.
- 원인: 저장된 logit은 모두 fp16 표현 가능한 값이다 (fp16 autocast 출력, |logit| < 32, 그 구간 fp16 간격 2^-6).
  두 GPU 실행 사이의 연산 순서·batch 구성 차이가 fp16 반올림을 바꾼 것. 최대 차이는 출력 구간의 1 ULP 수준이지만,
  값 자체 기준 ULP로 보면 q99.9 2 ULP (작은 logit에서는 더 큼) — 처음 쓴 "fp16 1-ULP"는 부정확했다.
  첫 허용치 2e-4는 fp32 규모의 CPU 검사에서 정한 값이라 fp16 GPU 출력에는 너무 엄격했다.
- 음성 대조: 다른 가중치 (arm a infer vs arm b 학습 logit)는 |dlogit| 중앙값 0.535, q99.9 11.6.
- **수정 (user 승인 2026-10-12 06:47 KST, `tools/ck/v3/infer_val_v3.py`)**: q99.9 |dlogit| ≤ 2^-5 (= 0.03125,
  출력 구간 fp16 2 ULP), flipped ≤ 0.1 %, 그리고 pdms_ck_noim 재현 (|diff| ≤ 1e-4, n_sel 같음)을 새로 필수로 추가.
  옛 허용치 초과 비율은 `frac_dlogit_gt_v1`로 계속 기록. 음성 대조와는 여전히 20배 이상 떨어져 있다.
- GPU 재추론 없이 `--recheck`로 저장 배열에서 검사만 다시 계산 (ckpt sha256이 cache key와 같은지 먼저 확인).
  a·b 모두 ok True. 이전 결과는 `infer_meta.json` `check_history`, 원본 사본 `infer_meta.pre_recheck_20261012.json`.
- 결과를 본 뒤 바꾼 기술 검사라는 점은 기록으로 남긴다. 성능 gate (G1–G4)는 손대지 않았다.

## 3. 수치 (navtrain val, n = 18,179 token / 214 log, paired bootstrap)

| arm | 내용 | PDMS | lead-decel TTC AUC | lead NC/TTC fail | a−1.0 hazard pref |
|---|---|---|---|---|---|
| a | learned forecaster (frozen) | 91.088 | .8296 | 6.85 % | .6841 |
| b | CA forecaster | 90.922 | .8444 | 6.73 % | .6677 |
| c | ck2T10 (temporal 없음, legacy dump) | 90.919 | .8369 | 6.17 % | .6526 |

| delta | dPDMS | 95 % CI |
|---|---|---|
| a − b | +0.166 | [−0.016, +0.343] |
| b − c | +0.003 | [−0.176, +0.197] |
| a − c | +0.169 | [−0.061, +0.376] |

| gate | 규칙 | 값 | 판정 |
|---|---|---|---|
| G1 | ΔPDMS(a−b) ≥ +0.20, CI lo > 0 | +0.166 [−0.016, +0.343] | fail (near_threshold) |
| G2 | lead TTC AUC(a) ≥ .855 | .8296 (a−b −0.0148 [−0.0298, −0.0004]) | fail |
| G3 | lead NC/TTC fail(a) ≤ 6.97 % | 6.85 % | pass |
| G4 | hazard pref 비열등 vs c | +0.0316 [−0.0025, +0.0651] | pass |

## 4. 해석

- learned forecaster(a)는 CA(b)보다 PDMS +0.17이지만 CI가 0을 포함하고, lead-decel TTC 판별력은 오히려 b보다 낮음 (G2).
- **b − c ≈ 0**: 분석 A에서 CA stack으로 기대했던 ~+0.38이 나오지 않음. forecaster 품질 이전에 temporal branch 자체
  (raster → CNN → zero-init projection 경로)가 정보를 충분히 쓰지 못하고 있을 가능성. 미확인.
- forecaster pretrain overfit_index 1.1035 (경고 > 1.10). 이 bias는 NO-GO 쪽 방향.

## 5. 후속

1. ~~무결성 허용치 수정~~ → 완료 (§2), 공식 NO-GO.
2. b − c ≈ 0 진단: user 승인 (06:47 KST), CPU 전용으로 진행 중 → `D3/diag_bc/`.
3. arm d (oracle forecaster, PRIV): 2번 결과를 보고 결정 (GPU 학습이라 user 승인 필요).

## 6. arm d (oracle forecaster, PRIVILEGED) 결과 — 2026-10-12 12:12 KST

판정 규칙은 결과 전에 등록했다(`ARM_D_PREREG.md`, 커밋 `88be625`, 09:19 KST). 결과는 12:12 KST에 나왔다.
- 학습: GPU 4·5, 08:02 → 12:08 KST (4.1 h).
- 추론: GPU 4, 207 s. 재현 검사 exact (pdms absdiff 0).
- 출력: `D3/decide/with_d/step1_with_d_no-arm-check.{json,log}`. d − c는 `D3/decide/with_d/dc_contrast/d_as_a.json`(d를 arm a 자리에 넣고 `--no-arm-check`).
- 공식 모드(`step1_with_d_check.log`)는 provenance D5에서 거부됐다. 예상한 대로다.
  - 원인: train-code sha 차이 (d 393ab2f8 vs a·b eb1bb78f)
  - 차이 나는 파일: online2.py, e2e_data2.py(e2e 전용), ckutil.py(GPU guard), pretrain_forecaster.py(d는 실행하지 않음)
  - teacher 학습 경로는 같다.

| | 판정 PDMS | ncttc_only | lead-decel TTC AUC | lead NC/TTC fail |
|---|---|---|---|---|
| d (oracle) | **91.600** | **91.528** | **.925** | **3.74 %** |
| a (learned) | 91.088 | 91.119 | .830 | 6.85 % |
| b (CA) | 90.922 | 91.000 | .844 | 6.73 % |
| c (ck2T10) | 90.919 | 90.888 | .837 | 6.17 % |

| Δ (paired log bootstrap) | ncttc_only | 판정 PDMS |
|---|---|---|
| **d − b (주 지표)** | **+0.529 [+0.340, +0.713]** | +0.678 [+0.435, +0.890] |
| d − a | +0.409 [+0.220, +0.614] | +0.512 [+0.288, +0.718] |
| d − c | +0.640 [+0.444, +0.853] | +0.681 [+0.415, +0.931] |

- lead-decel 기준 d − c: TTC AUC +0.088 [+0.060, +0.112], 실패율 −2.43 pp [−4.02, −0.86].

**사전 등록 판정: d − b = +0.529 ≥ +0.30 → 파이프라인은 강한 예보 신호를 회수한다. 다음 레버는 forecaster 품질이다.**
- d − c +0.640은 CPU emulation 기대 +0.71 [+0.52, +0.92]의 약 90%다. 즉 raster/gap → CNN → 주입 → critic 경로는 정보가 있으면 쓴다.
- b·a가 c에 비해 작게 오른 것은 파이프라인 문제가 아니다. CA와 learned 예보가 t0 BEV 위에 더해 주는 정보가 작기 때문이다(진단 `D3/diag_bc/`).
- d는 PRIVILEGED(GT 미래 입력)다. 분석 상한일 뿐이고, plug-in 결과나 KD teacher로는 쓰지 않는다.
- 단일 seed다.
