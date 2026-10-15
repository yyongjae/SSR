# refiner-fix B1 — 경로 연장 말림(station loop) 수정과 영향 측정

작성 2026-10-12 KST. 작업 위치 `/workspace/yongjae/SSR-cfix` (branch `refiner-fix`, 기준 `c1c66b7`).
CPU만 사용(`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, nice 10, ≤ 8 proc). 학습·GPU 없음. 분할: navtrain_val(val).
표기: [실측] 스크립트 출력으로 확인 / [추정] 판단.

## 1. 무엇을 고쳤나

`corridor_geometry(tau0, version)`에 version 인자를 추가했다. **기본값 1은 지금까지의 모든 run과 bit 단위로 같다**(§4 테스트).

version 2:
- (i) 원호 연장은 의도한 look-ahead(S_look = S8 + max(8 m, v_end·1 s))까지만 쓴다. 그 너머 station은 S_look의 접선 방향으로 **직선**으로 이어진다.
- (ii) v_end < 3 m/s인 draft는 연장 전체를 **직선**으로 한다.
- 그 밖(S_look, ds, station 위치 s, 채널 정의, flag)은 같다.

연결:
- `DraftPath.extend(arc_len=)` 추가. None이면 기존 코드를 그대로 탄다.
- refiner trunk: `CKNet(corridor_version=)`, `build_ck(..., corridor_version=)`. `load_ck`는 config.json `corridor_version`을 읽고, 없으면 1이다.
- `train_ck2.py --corridor-version {1,2}`. 기본은 1이고 config.json에 기록되며, resume할 때 같은지 검사한다.
- `RefinerNet.corridor_version` 클래스 속성(기본 1).
- v3 raster는 `geom.path`를 쓰므로 자동으로 따라간다.
- e2e student 경로(online2.py, e2e_data2.py)는 바꾸지 않았다. 기본 1로 동작한다.

**3 m/s 기준을 정한 근거** [실측 `out/m0_threshold.json`; val r34 후보 290,864개, S8 ≥ 2 m]

| v_end (m/s) | 연장 곡률과 draft 자체 곡률의 상관 | 부호 일치 | 연장 곡률 / draft 곡률 (중앙값) | 8 m look-ahead 회전 (중앙값) |
|---|---|---|---|---|
| 0–0.5 | −0.04 | 0.31 | 1.9× | 88° |
| 0.5–1 | −0.11 | 0.42 | 9.7× | 98° |
| 1–1.5 | −0.13 | 0.44 | 10.8× | 98° |
| 1.5–2 | −0.04 | 0.49 | 8.9× | 83° |
| 2–2.5 | +0.15 | 0.55 | 5.7× | 61° |
| 2.5–3 | +0.24 | 0.60 | 4.3× | 43° |
| 3–4 | +0.43 | 0.66 | 3.6× | 31° |
| 4–6 | +0.54 | 0.65 | 2.8× | 16° |

- 3 m/s 아래에서는 연장 곡률이 draft 자체의 회전과 거의 무관하다. 사실상 노이즈다.
- 그런데도 8 m look-ahead를 43–98° 꺾는다. draft의 마지막 구간 자체는 3–5°만 돈다.

## 2. a. 말림 통계 [실측 `out/m1_loop_impact.json`, val r34 후보 290,864개]

| 구간 | v1 회전 > π | v2 회전 > π | v1 회전 > π/2 | v2 회전 > π/2 |
|---|---|---|---|---|
| draft 안 (S8까지) | 1.0% | 1.0% | 2.5% | 2.5% |
| look-ahead (S_look까지) | 3.4% | 2.1% | 26.8% | 12.3% |
| station 48개 전체 | **34.0%** | **2.2%** | 48.6% | 12.8% |
| 느린 후보 (v_end < 3, 31.6%) 48개 전체 | **69.9%** | **6.0%** | 81.2% | 10.7% |

- v2에 남은 회전은 draft 자체의 회전(S8 안)이거나, 빠른 후보가 look-ahead 안에서 실제로 도는 경우다.

## 3. b. 라벨이 정해지는 구간에서의 변화 [실측 `out/m1_loop_impact.json`]

구간 = s ≤ S8 + 4 m + v_end·1 s. NC/TTC 라벨이 정해지는 ego 앞부분이다.

| 부분집합 | n | 중심선 이동 > 0.6 m | > 1.2 m | ±4.8 m 띠 이동 > 0.6 m | v2 띠에만 물체 | v1 띠에만 물체 |
|---|---|---|---|---|---|---|
| 전체 | 290,864 | 24.4% | 20.7% | 29.1% | 1.00% | 0.47% |
| 느린 후보 (v < 3) | 91,778 | 77.5% | 65.7% | 84.4% | 3.12% | 1.48% |
| NC 실패 | 7,420 | 44.8% | 38.8% | 49.7% | 3.07% | **5.53%** |
| TTC 실패 | 18,801 | 32.6% | 27.5% | 38.1% | 1.98% | 2.95% |
| lead-decel token | 25,680 | 33.4% | 26.2% | 38.9% | 1.39% | 0.93% |
| lead-decel의 NC/TTC 실패 | 2,645 | 27.0% | 20.4% | 31.6% | 0.91% | 1.70% |

- 물체는 val object pack의 BEVFusion t0 box 중심이다. "띠 안"은 구간 안 station 중심선과의 거리 ≤ 4.8 m이고, 투영점이 구간 안에 있는 경우다.
- 읽는 법 [추정]:
  - refiner trunk가 BEV를 읽는 **위치**는 느린 후보에서 크게 바뀐다(77%가 0.6 m 이상).
  - 그러나 띠 안에 **어떤 물체가 들어오는지**가 바뀌는 후보는 1.5% 정도이고, 방향도 양쪽이다.
  - NC 실패 후보에서는 오히려 v1의 말린 띠가 덮는 물체가 더 많다(5.5% vs 3.1%). 말린 띠가 ego 옆을 쓸고 지나가며 옆에서 오는 위험 물체를 우연히 덮는 경우로 보인다.
  - refiner trunk는 BEV 전체를 덮는 global token 200개도 함께 본다.
  - 따라서 trunk의 NC/TTC 판별에 대한 순효과는 부호도 크기도 이 측정만으로는 알 수 없다. 작을 가능성이 크다.

## 4. c. v3 raster 충돌 readout [실측 `out/m2_raster_v2.json`; diag_bc/raster와 같은 1,200 token / 19,200 후보, arm b(CA)의 예보]

- v1로 다시 계산하면 `diag_bc/raster/out/s1_main.npz`가 정확히 재현된다(모든 readout max |Δ| = 0).

| readout | v1 AUC | v2 AUC | Δ [95% CI, log bootstrap] |
|---|---|---|---|
| NC: r_nc | .761 | .819 | **+0.058 [+0.029, +0.088]** |
| NC: r_ttc_same | .763 | .821 | +0.058 [+0.030, +0.086] |
| TTC: r_ttc_same | .741 | .773 | **+0.032 [+0.019, +0.046]** |
| TTC: r_nc | .702 | .722 | +0.020 [+0.006, +0.034] |

- 참고로 loop station을 탐색에서 뺀 "cut Frenet" 진단은 +0.060 / +0.030이었다(`raster/out/s4_loop.json`). v2가 같은 이득을 낸다.
- v1에서는 충돌이 안 보이고(r_nc < 0.1) v2에서 보이는(≥ 0.5) NC 실패 후보는 6.4%(29 / 453)다.

## 5. 테스트 [실측]

- 새 테스트 `tests/test_refiner_fix_corridor.py` 9개 통과:
  - v1 == c1c66b7: 무작위·val draft에서 CorridorGeom 모든 필드와 path.eval 10개 필드가 bit 단위로 같다. `extend` 3 정책도 같다.
  - v2: S_look 너머 곡률 0, 회전 0. 느린 행은 직선이다. 빠른 행은 S_look까지 그대로다.
  - CKNet(corridor_version=1) forward == c1c66b7 corridor로 바꾼 forward (bit 단위, score/extra/w_lat/z_lon).
  - v2가 trunk까지 전달된다. load_ck가 config를 읽는다. 잘못된 version은 거부된다.
- 기존 refiner 테스트(corridor, geometry, refiner_net, decoder, ck_core): 83 통과 / 5 실패.
  - 실패 5개는 기준 tree(c1c66b7)에서도 똑같이 실패한다. 원인은 `sf_common` 모듈 없음, fixture 표본 크기다. 이번 수정과 무관하다.
- tests/ 중 ck2·v3·ck_ 묶음(`-k "ck2 or v3 or ck_"`): 586 통과 / 4 실패 / 2 skip (42 min) [실측 `logs/tests_ck_v3_all.log`].
  - 기존부터 있던 실패 3개:
    - `test_ck_data::test_label_real_navtest_matches_csv`(파일 없음)
    - `test_ck_e2e_launch::test_ck_yaml_matches_contract_and_config`
    - 위 둘은 기준 tree에서도 똑같이 실패한다.
    - `test_v3_e2e_teacher::test_non_temporal_path_bit_identical_to_head`: e184002 커밋 이후 HEAD 자체가 KDObjRows를 포함하므로 항상 실패한다. 이번 수정과 무관하다.
  - 이번 수정 때문에 생긴 1개: `test_v3_integration_off`.
    - 배열 1,000개 이상은 모두 같았다. config.json에 새 키 `corridor_version`이 생긴 것만 다르다(facts_diff train1/train3_config_keys).
    - `tools/ck/v3/off_equiv_compare.py`의 허용 새 키 목록에 `corridor_version`을 추가했다.
    - 재실행: `test_refiner_fix_corridor` + `test_v3_integration_off` + `test_ck_core` **33 통과 / 0 실패** [실측 `logs/tests_rerun.log`].

## 6. 권고 [추정]

- **v3 시간 branch**: v2의 이득이 분명하다(readout NC AUC +0.06). 다음 v3 반복(예보기 재설계)에서는 `--corridor-version 2`를 기본으로 쓰는 것이 맞다.
- **refiner v2 teacher(trunk)**: 띠 안 물체 구성의 변화가 1.5% 안팎이고 방향도 섞여 있다. 그래서 이득은 작거나 불확실하다.
  - GPU 4·5로 3–4 h짜리 확인(DET teacher 10 ep, ck2T10 recipe + `--corridor-version 2`, val에서 ck2T10과 비교)은 싼 편이라 해볼 만하다.
  - 단, e2e 재학습은 teacher가 val에서 확실히 나을 때만 한다(예: ncttc_only ≥ +0.2, CI > 0).
- **e2e에 v2 teacher를 쓰려면 먼저 연결 작업이 필요하다.**
  - student CK head의 corridor version은 아직 e2e 설정에 연결돼 있지 않다.
  - 그래서 v2 teacher를 KD에 쓰면 teacher는 v2, student는 v1 geometry가 된다.
- 지금 도는 학습(v3T, nokd)과 이미 낸 결과는 그대로 유효하다. 모두 v1이라 서로의 비교는 공정하다.

## 파일
- 스크립트: `scripts/m0_threshold.py`, `scripts/m1_loop_impact.py`, `scripts/m2_raster_v2.py`
- 출력: `out/m0_threshold.json`, `out/m1_loop_impact.json` (+ `m1_per_candidate.npz`), `out/m2_raster_v2.json` (+ `.npz`)
- 로그: `logs/`
