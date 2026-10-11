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
