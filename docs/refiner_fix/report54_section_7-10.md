# report 54 §7-10 추가분 (SSR/report/54_refiner_v3_temporal_critic.md에 넣을 내용)

작성 2026-10-13 01:40 KST.
이 세션은 worktree `SSR-cfix`에 격리돼 있어 공유 checkout의 `SSR/report`를 직접 고칠 수 없다. 그래서 내용을 여기에 먼저 둔다.

### 7-10. 경로 연장 말림 버그(B1) 수정: refiner-fix (10-12 15:40 → 10-13 01:35 KST) [실측 `D3/diag_bc/corridor_fix/notes.md`, `out/m0–m2`; 코드 `/workspace/yongjae/SSR-cfix` branch `refiner-fix`]
- 사용자 지시: "버그 고쳐"(10-12), 이어서 "허락받지 말고 너가 알아서 처리"(10-12 밤). branch 이름은 사용자 요청으로 `corridor-fix`에서 `refiner-fix`로 바꿨다.

**무엇이 문제였나**
- 후보 경로는 4초(S8)에서 끝난다. 그 뒤는 끝 곡률을 그대로 쓴 원호로 연장한다.
- 느린 후보(v_end < 3 m/s)는 끝 곡률이 노이즈다. draft 자체의 회전과 상관이 −0.13..+0.24밖에 안 되는데도, 연장은 8 m look-ahead를 43–98° 꺾는다.
- station 간격이 최소 1 m라 station이 48 m까지 깔린다. 그래서 원호가 S_look 너머까지 이어져 출발점 근처로 되감긴다.
- 결과: station 48개 전체가 180° 넘게 도는 후보가 **34.0%**이고, 느린 후보에서는 69.9%다.

**수정 (version 2; 기본 version 1은 기존과 bit 단위로 같다)**
- 원호는 S_look까지만 쓰고, 그 너머는 직선으로 잇는다.
- v_end < 3 m/s이면 연장 전체를 직선으로 한다.
- 버전은 `train_ck2.py --corridor-version`으로 정하고 config.json에 기록한다. `load_ck`가 이를 읽고, v3 raster는 자동으로 따라간다.
- e2e student 경로는 바꾸지 않았다. version 2 teacher를 e2e에 쓰려면 student head 쪽도 연결해야 한다.
- 커밋: `98280e5`(수정 + 테스트 9개), `2217a73`(val 추론을 GPU 2,3에서도 허용).

**효과 [실측]**

| 측정 | version 1 | version 2 |
|---|---|---|
| 48 station 전체 회전 > π | 34.0% | 2.2% |
| 느린 후보 회전 > π | 69.9% | 6.0% |
| v3 raster 충돌 판독 NC AUC (1,200 token) | .761 | **.819** (+0.058 [+0.029, +0.088]) |
| v3 raster TTC 판독 AUC | .741 | **.773** (+0.032 [+0.019, +0.046]) |
| raster에서 충돌이 안 보이던 NC 실패 | 29 / 453 (6.4%) | 0 |

**trunk(v2 teacher)에 대한 영향은 아직 모른다 [실측 / 추정]**
- 라벨이 정해지는 구간(s ≤ S8 + 4 m + v_end·1 s)에서 중심선이 0.6 m 넘게 움직이는 후보는 24.4%다. 느린 후보만 보면 77.5%다.
- 그러나 ±4.8 m 띠 안에 드는 물체가 바뀌는 후보는 약 1.5%이고, 바뀌는 방향도 양쪽이다.
- NC 실패 후보에서는 v1 띠에만 있는 물체(5.5%)가 v2 띠에만 있는 물체(3.1%)보다 많다.
- 그래서 성능에 미치는 효과는 학습해 봐야 알 수 있다.

**기존 결과는 유효하다**
- 지금까지의 run은 모두 version 1이다: v2 teacher, main / bevkd / nokd e2e, v3 arm a–d, ck2e2e_v3T.
- 학습과 테스트가 같은 코드라 결과가 틀린 것은 아니다. 정보를 일부 덜 쓴 "성능 손실"이다.
- 모든 run이 같은 버그를 공유하므로 run끼리의 비교도 공정하다. 그래서 진행 중인 run은 멈추지 않았다.

**GPU 확인 실험** (자동 체인 `CK/refiner_fix/run_teacher_cv_chain.sh`, 로그 `CK/refiner_fix/chain.log`)
- ck2T10 레시피(10 epoch, GPU 2,3)로 두 arm을 학습한다.
  - `refT10_cv2`: version 2
  - `refT10_cv1`: 같은 코드에 version 1. 코드 차이와 run 간 편차를 통제하기 위한 것이다.
- val r34 추론 뒤 paired 비교를 3개 한다: cv2 − ck2T10, cv1 − ck2T10, cv2 − cv1. 결과는 `CK/refiner_fix/decide/`에 쓴다.
- 시작 시점: nokd 평가가 끝나 GPU 2,3이 비면 자동으로 시작한다(10-13 저녁 예상).
- CPU smoke(학습 3 step, 추론 16 row)는 통과했다.

**해석 기준 (실험 전에 기록) [추정]**
- cv2 − cv1 ncttc_only가 +0.2 이상이고 CI가 0보다 크면, e2e 적용을 검토한다. 이때 student head 연결이 필요하다.
- 0 근처면 trunk는 그대로 두고, 다음 v3(예측기 재설계)에만 version 2를 쓴다.
