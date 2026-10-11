# refiner-v3 Step 1 결과 — temporal DET teacher arm a/b/c 판정

작성: 2026-10-12 02:55 KST. 단일 seed 최종 판정 (user: seed ablation 하지 않음).
산출물: `D3/decide/step1_official_refused.log` (공식 decide, REFUSED), `D3/decide/step1_diag_no_arm_check_20261012`
(`--no-arm-check` 진단 JSON, v3_decide_4). D3 = `/home/external-user/ssd/yongjae_refiner/ck/v3`.

## 1. 결론

- **공식 decide: REFUSED (rc=1)** — 성능이 아니라 arm a의 inference reproduce 무결성 검사 실패 때문.
- **진단 decide (`--no-arm-check`): `INVALID (would be NO-GO)`** — G1·G2 미달. 수치상 temporal branch는 GO 기준을 넘지 못함.
- 그래도 e2e KD 전체 학습(`ck2e2e_v3T`)은 user 지시("det, map 다 끝나면 알아서 e2e 모델에 KD까지")대로 GO 여부와 무관하게
  launch함 (STEP2_E2E.md §7).

## 2. 무결성 검사 실패 원인 (arm a reproduce)

- 규칙: 재계산한 per-token logit과 dump의 차이 q99.9 |dlogit| ≤ 2e-4, 그리고 flipped pick ≤ 0.1 %.
- 실측: q99.9 |dlogit| 0.0039 (> 2e-4), flipped pick 2 / 18,179 (0.011 %, 통과), PDMS 재현 0.90288 (absdiff 0).
- 진단: fp16 autocast의 1-ULP 차이. fp32로 재계산하면 maxdiff 2.8e-5이고, fp16에서는 값의 0.06 %만 1 ULP 차이가 남.
  logit 허용치 2e-4가 fp16 출력 기준으로는 너무 엄격함. arm b는 통과.
- 허용치는 사후에 고치지 않음 (결과를 본 뒤 gate 변경 금지). 고칠지 여부는 user 결정 사항.

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

## 5. 남은 user 결정

1. 무결성 허용치를 fp16-aware로 바꿔 공식 decide를 다시 낼지 (현재 수치로는 NO-GO는 그대로).
2. b − c ≈ 0 진단 (temporal branch ablation / feature probe) 진행 여부.
3. arm d (oracle forecaster, PRIV) — temporal branch 상한 확인용.
