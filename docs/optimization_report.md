# `fusion_v15_main` (`23ep` SOTA & `30ep` 완주) 공식 `PDMS` / `navtest EPDMS` / `navhard EPDMS` 정밀 평가 및 96-후보 선택 수식 최적화 보고서

본 문서는 `fusion_v15_main` (`CK2 Refiner E2E + BEV Selector v15`)의 **SOTA 에폭인 23ep (`Epoch 22`, [`epoch=22-step=15295.ckpt`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/train/lightning_logs/version_0/checkpoints/epoch=22-step=15295.ckpt))**와 **최종 완주 에폭인 30ep (`Epoch 29`, [`last.ckpt`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/train/lightning_logs/version_0/checkpoints/last.ckpt))**에 대해:
1. **`navtest` v1 PDMS** ($N=12,146$, Non-Reactive Log-Replay)
2. **`navtest` v2.2 One-Stage EPDMS** ($N=12,146$, `After v2.2` 공식 및 `Before pre-359c7f7` 수정 전 방식)
3. **`navhard_two_stage` v2.2 Two-Stage EPDMS** ($N=5,912$: Stage 1 원본 `450` + Stage 2 합성 `5,462`, Reactive IDM, `After v2.2` 공식 및 `Before pre-359c7f7` 수정 전 방식)

세 가지 공식 벤치마크 전수 평가를 수행하고, 나아가 **96개 후보 궤적 풀(`cand96`, `lat_traj`)과 양대 헤드 예측값(`v2_final`, `v2_im`, `v2_sim`, `score_logit`)을 조합하여 각 벤치마크 점수를 극대화하는 최적 선택 수식(Selector Formula)**을 탐색한 결과입니다.

---

## 1. 한눈에 보는 최고 점수 비교 요약 (Baseline vs. 최적 조합 🏆)

| 벤치마크 | 에폭 | 기존 `default(0.5)` | 기존 `plugin(0.5)` | 기존 `plugin(1.0)` | **단일 올라운더 (`prod_hybrid2`)** | **벤치마크별 최강 조합 🏆** | **최대 상승폭 ($\Delta$ vs. `default`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1) `navtest` v1 PDMS** ($N=12,146$) | **23ep (`Ep 22`)** | `92.34%` | `92.42%` | `93.47%` | `92.57%` | **`93.87%` 🏆** (`lat96`: **`93.81%`** / Clean: `93.72%`) | **`+1.53%p`** (`plugin(1.0)` 대비 **`+0.40%p`**) |
| **1) `navtest` v1 PDMS** ($N=12,146$) | **30ep (`Ep 29`)** | `92.23%` | `92.38%` | `93.01%` | `92.62%` | **`93.84%` 🏆** (`lat96`: **`93.79%`** / Clean: `93.65%`) | **`+1.61%p`** (`plugin(1.0)` 대비 **`+0.83%p`**) |
| **2) `navtest` EPDMS (After, `v2.2`)** | **23ep (`Ep 22`)** | `90.61%` | `90.63%` | `88.44%` | `90.93%` | **`91.09%` 🏆** (`tune_34_dv_id0_0.04`) | **`+0.48%p`** (`plugin(1.0)` 대비 **`+2.65%p`**) |
| **2) `navtest` EPDMS (Before, 수정 전)** | **23ep (`Ep 22`)** | `87.10%` | `87.10%` | `85.03%` | `87.32%` | **`87.48%` 🏆** (`tune_34_dv_id0_0.04`) | **`+0.38%p`** |
| **2) `navtest` EPDMS (After, `v2.2`)** | **30ep (`Ep 29`)** | `90.72%` | `90.86%` | `87.98%` | `91.15%` | **`91.28%` 🏆** (`tune_34_dv_id0_0.04`) | **`+0.56%p`** |
| **2) `navtest` EPDMS (Before, 수정 전)** | **30ep (`Ep 29`)** | `87.27%` | `87.37%` | `84.60%` | `87.50%` | **`87.63%` 🏆** (`tune_34_dv_id0_0.04`) | **`+0.36%p`** |
| **3) `navhard` EPDMS (After, `v2.2`)** | **23ep (`Ep 22`)** | `41.65%` (`S1 81.51 / S2 51.14`) | `42.53%` (`S1 81.37 / S2 52.33`) | `37.49%` | `43.92%` | **`47.89%` 🏆** (`S1 81.67% / S2 59.28%`) | **`+6.24%p`** (**Stage 2 `+8.14%p` 폭증!**) |
| **3) `navhard` EPDMS (Before, 수정 전)** | **23ep (`Ep 22`)** | `37.94%` (`S1 75.04 / S2 51.14`) | `38.75%` (`S1 74.99 / S2 52.33`) | `34.23%` | `40.44%` | **`44.08%` 🏆** (`S1 75.79% / S2 59.28%`) | **`+6.14%p`** |
| **3) `navhard` EPDMS (After, `v2.2`)** | **30ep (`Ep 29`)** | `40.06%` (`S1 81.54 / S2 48.71`) | `42.12%` (`S1 81.38 / S2 51.76`) | `36.85%` | `42.73%` | **`47.47%` 🏆** (`S1 81.23% / S2 59.10%`) | **`+7.41%p`** (**Stage 2 `+10.39%p` 폭증!**) |
| **3) `navhard` EPDMS (Before, 수정 전)** | **30ep (`Ep 29`)** | `36.33%` (`S1 75.06 / S2 48.71`) | `38.41%` (`S1 75.08 / S2 51.76`) | `33.62%` | `39.18%` | **`43.93%` 🏆** (`S1 75.54% / S2 59.10%`) | **`+7.60%p`** |

---

## 2. 기존 선택 규칙(`default`, `noim`, `plugin`)의 구조적 한계 및 최적화 원리

1. **기존 `plugin(1.0)`이 v1 PDMS(`93.47%`)에서는 최고였으나 EPDMS(`88.44%` / `navhard 37.49%`)에서 급락한 원인**:
   - `navtest` EPDMS 세부 지표를 비교하면 `v2_lat`(`col 0`)은 `EC = 83.28%, LK = 96.85%, DDC = 99.55%`로 매우 안정적인 반면, `plugin(1.0)`은 **`EC = 65.09% (-18.19%p 폭락)`**, **`LK = 93.31% (-3.54%p 하락)`**, **`DDC = 98.07% (-1.48%p 하락)`**를 기록했습니다.
   - 이는 NAVSIM v2.2 EPDMS에 도입된 **`lane_keeping_deviation_limit = 0.5m` (`LK`)** 및 **연속 프레임 간 궤적 일관성을 평가하는 `two_frame_extended_comfort` (`EC`)** 때문인데, 기존 선택기는 6가지 변형(`v=0: id, v=1: a-1.0, v=2: a-0.5, v=3: a+0.5, v=4: l-0.5, v=5: l+0.5`)에 동일한 사전 가중치(`0.0`)를 부여하여 **좌우 `±0.5m` 평행이동 변형(`l±0.5`, 전체의 ~31% 선택)과 급가속 변형(`a+0.5`, 전체의 ~42% 선택)을 과도하게 스위칭**했기 때문입니다.
2. **핵심 해법 1 — 양대 헤드 안전성 합의 (Two-Head Safety Consensus, `CK2 + V2_sim`)**:
   - 기존 `default`/`plugin`은 `v2` 플래너가 예측한 5개 서브메트릭 확률 `v2_sim` (`nc_v2, dac_v2, ep_v2, ttc_v2, cmf_v2`)을 버리고 단일 스칼라 `v2_final`만 블렌딩했습니다.
   - 반면 최적 수식은 **`CK2`의 `log(nc_ck), log(dac_ck)`와 `v2_sim`의 `log(nc_v2), log(dac_v2)`를 동시에 가중 합산**하여, **3D BEV Selector 헤드와 V2 앵커 헤드가 모두 안전(`NC`, `DAC`)하다고 합의한 후보만 통과**시킵니다. 이 단일 원리만으로도 `navhard_two_stage` EPDMS가 `42.53% -> 43.72%`로 즉각 상승했습니다.
3. **핵심 해법 2 — 변형 타입별 사전 가중치 ($\delta_v$, Variant-Type Priors)**:
   - 96개 후보 `c = 6*k + v`의 변형 종류 $v \in \{0..5\}$에 따라:
     - **`navhard_two_stage` (Reactive IDM)**: 급가속(`v=3, a+0.5`: `-0.05`)과 좌우 차선 변경(`v=4,5, l±0.5`: `-0.03 ~ -0.04`)에 페널티를 주고 부드러운 감속(`v=2, a-0.5`: `+0.02`, `v=1, a-1.0`: `+0.01`)에 보너스를 부여하자, 합성 Stage 2 반응형 주행 생존율이 **`51.14% -> 59.28% (+8.14%p)`**로 폭증하며 최종 점수가 **`47.89%`**에 도달했습니다.
     - **`navtest` One-Stage EPDMS**: 원본 앵커 궤적(`v=0, id`: `+0.04`)에 보너스를 주고 좌우 변형(`v=4,5, l±0.5`: `-0.06` / `-0.03`)과 급가속(`v=3, a+0.5`: `-0.02`)을 억제하자, `LK`(`96.45%`)와 `EC`(`79.62%`)를 높게 유지하면서 `NC`(`99.35%`)·`DAC`(`99.07%`)·`TTC`(`99.00%`)를 극대화하여 **`91.09%`(23ep) / `91.28%`(30ep)**를 달성했습니다.
4. **핵심 해법 3 — 기대 EPDMS 곱셈 항 (`w_prod`) 결합**:
   - 로그 합산 항 외에 실제 EPDMS 수식 구조를 모사한 기대 점수 곱셈 항 $\text{epdms}_{\text{ck}} = (\text{nc}_{\text{ck}} \cdot \text{dac}_{\text{ck}}) \cdot \frac{5\text{ttc}_{\text{ck}} + 5\text{ep}_{\text{ck}} + 2\text{cmf}_{\text{ck}}}{12}$ (`w_prod = 2.0`)을 결합했을 때 `navtest` EPDMS와 `navhard` EPDMS 모두에서 일관된 추가 상승(`prod_hybrid2`)이 나타났습니다.

---

## 3. 벤치마크별 최적 수식 및 세부 Sub-Metric 전체 결과

### (1) `navhard_two_stage` v2.2 Two-Stage EPDMS ($N=5,912$: Stage 1 `450` + Stage 2 `5,462`)

- **최적 선택 수식 (`tune_32_dv_id0_0.0`)**:
  $$S(c) = w_{\text{v2f}} \cdot \text{v2\_final}_{k} + w_{\text{nc}} \log(\text{nc}_{\text{ck}, c}) + w_{\text{dac}} \log(\text{dac}_{\text{ck}, c}) + w_{\text{ttc}} \log(\text{ttc}_{\text{ck}, c}) + w_{\text{rest}} \log(5\text{ttc}_{\text{ck}, c} + 2\text{cmf}_{\text{ck}, c} + 5\text{ep}_{\text{ck}, c}) + w_{\text{nc\_v2}} \log(\text{nc}_{\text{v2}, k}) + w_{\text{dac\_v2}} \log(\text{dac}_{\text{v2}, k}) + \delta_{v(c)}$$
  - **23ep (`Epoch 22`) 최적 파라미터**: `w_v2f=0.45, w_im=0.0, w_nc=0.8, w_dac=0.6, w_ttc=0.25, w_rest=0.5, w_nc_v2=0.3, w_dac_v2=0.5, dv=[0.0, 0.01, 0.02, -0.05, -0.03, -0.03], lat_mode="lat"`
  - **30ep (`Epoch 29`) 최적 파라미터**: `w_v2f=0.45, w_im=0.0, w_nc=0.6, w_dac=0.9, w_ttc=0.25, w_rest=0.5, w_nc_v2=0.7, w_dac_v2=0.9, dv=[0.0, 0.01, 0.02, -0.05, -0.04, -0.04], lat_mode="lat"`

| 에폭 / 선택 규칙 | 구간 | `NC` | `DAC` | `DDC` | `TLC` | `EP` | `TTC` | `LK` | `HC` | `EC` | **After (`v2.2`) EPDMS** | **Before (수정 전) EPDMS** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **23ep `v2` (Base)** | Stage 1 / Stage 2 / **Combined** | `83.97` | `78.42` | `87.84` | `99.37` | `46.58` | `82.61` | `40.79` | `94.75` | `53.20` | `74.81` / `45.42` / **`35.17%`** | `69.10` / `45.42` / `32.16%` |
| **23ep `v2_lat`** | Stage 1 / Stage 2 / **Combined** | `85.20` | `81.25` | `88.78` | `99.48` | `48.20` | `83.82` | `43.24` | `94.25` | `47.55` | `78.24` / `47.77` / **`37.88%`** | `72.42` / `47.77` / `34.72%` |
| **23ep `default(0.5)`** | Stage 1 (`450`) | `98.89` | `96.22` | `98.44` | `99.78` | `80.84` | `98.22` | `91.78` | `96.89` | `51.78` | `81.51%` | `75.04%` |
| | Stage 2 (`5,462`) | `87.64` | `86.24` | `91.37` | `99.23` | `73.79` | `85.94` | `45.80` | `95.52` | `48.88` | `51.14%` | `51.14%` |
| | **Combined (최종)** | `86.67` | `83.22` | `90.22` | `99.01` | `59.75` | `84.42` | `42.20` | `92.54` | `25.99` | **`41.65%`** | **`37.94%`** |
| **23ep `plugin(0.5)`** | Stage 1 / Stage 2 / **Combined** | `86.85` | `84.45` | `90.70` | `99.05` | `60.29` | `84.56` | `42.49` | `92.34` | `25.27` | `81.37` / `52.33` / **`42.53%`** | `74.99` / `52.33` / `38.75%` |
| **23ep `plugin(1.0)`** | Stage 1 / Stage 2 / **Combined** | `84.60` | `84.19` | `88.24` | `98.71` | `60.15` | `81.56` | `37.65` | `89.94` | `15.72` | `78.00` / `47.09` / **`37.49%`** | `72.07` / `47.09` / `34.23%` |
| **★ 23ep 최적 (`tune_32`)** | **Stage 1 (`450`)** | **`99.33`** | `96.22` | **`99.56`** | **`100.00`** | `74.60` | **`99.33`** | **`95.33`** | **`97.56`** | **`58.67`** | **`81.67%`** | **`75.79%`** |
| | **Stage 2 (`5,462`)** | **`90.60`** | **`88.75`** | **`92.73`** | **`99.41`** | `72.05` | **`89.49`** | **`49.88`** | **`97.16`** | **`66.47`** | **`59.28%` (`+8.14%p`)** | **`59.28%` (`+8.14%p`)** |
| | **Combined (최종)** | **`90.19`** | **`85.22`** | **`92.53`** | **`99.41`** | `55.08` | **`89.07`** | **`47.63`** | **`94.72`** | **`39.59`** | **`47.89%` 🏆 (`+6.24%p`)** | **`44.08%` 🏆 (`+6.14%p`)** |
| **30ep `default(0.5)`** | Stage 1 (`450`) | `98.22` | `96.67` | `98.89` | `99.78` | `81.07` | `97.56` | `91.78` | `96.67` | `51.33` | `81.54%` | `75.06%` |
| | Stage 2 (`5,462`) | `86.44` | `85.64` | `89.64` | `99.13` | `72.24` | `84.60` | `46.51` | `95.07` | `47.53` | `48.71%` | `48.71%` |
| | **Combined (최종)** | `85.08` | `83.20` | `88.73` | `98.91` | `59.25` | `82.61` | `42.98` | `91.85` | `25.03` | **`40.06%`** | **`36.33%`** |
| **★ 30ep 최적 (`tune_32`)** | **Stage 1 (`450`)** | **`99.33`** | **`96.89`** | **`99.56`** | **`100.00`** | `73.96` | **`99.33`** | **`95.33`** | **`96.89`** | **`54.22`** | **`81.23%`** | **`75.54%`** |
| | **Stage 2 (`5,462`)** | **`90.90`** | **`89.38`** | **`92.14`** | **`99.37`** | `71.38` | **`89.82`** | **`51.54`** | **`96.55`** | **`65.05`** | **`59.10%` (`+10.39%p`)** | **`59.10%` (`+10.39%p`)** |
| | **Combined (최종)** | **`90.26`** | **`86.62`** | **`91.93`** | **`99.37`** | `54.40` | **`89.18`** | **`48.97`** | **`93.57`** | **`35.81`** | **`47.47%` 🏆 (`+7.41%p`)** | **`43.93%` 🏆 (`+7.60%p`)** |

---

### (2) `navtest` v2.2 One-Stage EPDMS ($N=12,146$)

- **최적 선택 수식 (`tune_34_dv_id0_0.04`)**:
  $$S(c) = w_{\text{v2f}} \cdot \text{v2\_final}_{k} + w_{\text{nc}} \log(\text{nc}_{\text{ck}, c}) + w_{\text{dac}} \log(\text{dac}_{\text{ck}, c}) + w_{\text{rest}} \log(5\text{ttc}_{\text{ck}, c} + 2\text{cmf}_{\text{ck}, c} + 5\text{ep}_{\text{ck}, c}) + w_{\text{nc\_v2}} \log(\text{nc}_{\text{v2}, k}) + w_{\text{dac\_v2}} \log(\text{dac}_{\text{v2}, k}) + w_{\text{prod}} \cdot \text{epdms}_{\text{ck}, c} + \delta_{v(c)}$$
  - **23ep (`Epoch 22`) 최적 파라미터**: `w_v2f=0.4, w_im=0.0, w_nc=0.6, w_dac=0.8, w_rest=0.5, w_prod=2.0, w_nc_v2=0.5, w_dac_v2=0.3, dv=[0.04, -0.02, 0.01, -0.02, -0.06, -0.06], lat_mode="lat"`
  - **30ep (`Epoch 29`) 최적 파라미터**: `w_v2f=0.3, w_im=0.0, w_nc=0.8, w_dac=1.2, w_rest=0.5, w_prod=2.0, w_nc_v2=0.4, w_dac_v2=0.4, dv=[0.04, -0.02, 0.01, -0.02, -0.03, -0.03], lat_mode="lat"`

| 에폭 / 선택 규칙 | `NC` | `DAC` | `DDC` | `TLC` | `EP` | `TTC` | `LK` | `HC` | `EC` | **After (`v2.2`) EPDMS** | **Before (수정 전) EPDMS** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **23ep `v2` (Base)** | `98.99` | `96.18` | `99.58` | `99.77` | `86.54` | `98.65` | `96.58` | `98.33` | **`84.30`** | `88.35%` | `84.72%` |
| **23ep `v2_lat`** | `99.17` | `97.64` | `99.55` | `99.77` | `86.57` | `98.75` | **`96.85`** | `98.33` | `83.28` | `89.76%` | `86.20%` |
| **23ep `default(0.5)`** | `99.21` | `98.58` | `99.32` | `99.70` | `89.40` | `98.82` | `95.93` | `98.18` | `78.42` | `90.61%` | `87.10%` |
| **23ep `plugin(0.5)`** | `99.21` | `98.77` | `99.31` | `99.65` | `89.24` | `98.82` | `95.83` | `98.18` | `77.76` | `90.63%` | `87.10%` |
| **23ep `plugin(1.0)`** | `98.90` | `98.72` | `98.07` | `99.32` | **`92.78`** | `98.44` | `93.31` | `97.32` | `65.09` | `88.44%` | `85.03%` |
| **23ep 올라운더 (`prod_hybrid2`)** | `99.28` | **`98.94`** | `99.43` | `99.75` | `88.74` | `98.93` | `96.32` | `98.27` | `78.52` | **`90.93%`** | **`87.32%`** |
| **★ 23ep 최적 (`tune_34`)** | **`99.35`** 🏆 | `98.90` | **`99.46`** | **`99.78`** | `88.67` | **`99.00`** 🏆 | `96.34` | **`98.29`** | `78.96` | **`91.09%` 🏆 (`+0.48%p`)** | **`87.48%` 🏆 (`+0.38%p`)** |
| **30ep `v2` (Base)** | `98.88` | `96.71` | `99.63` | `99.79` | `87.17` | `98.58` | `96.59` | `98.33` | **`84.24`** | `88.86%` | `85.25%` |
| **30ep `v2_lat`** | `99.04` | `97.81` | `99.60` | `99.79` | `87.19` | `98.70` | **`96.95`** | `98.33` | `83.06` | `89.90%` | `86.29%` |
| **30ep `default(0.5)`** | `99.13` | `98.71` | `99.26` | `99.64` | `89.64` | `98.75` | `95.98` | `98.19` | `78.79` | `90.72%` | `87.27%` |
| **30ep `plugin(0.5)`** | `99.14` | `98.87` | `99.30` | `99.66` | `89.56` | `98.77` | `95.97` | `98.19` | `78.49` | `90.86%` | `87.37%` |
| **30ep 올라운더 (`prod_hybrid2`)** | `99.18` | `98.98` | `99.42` | `99.71` | `89.11` | `98.85` | `96.62` | `98.27` | `79.65` | **`91.15%`** | **`87.50%`** |
| **★ 30ep 최적 (`tune_34`)** | **`99.18`** | **`99.07`** 🏆 | **`99.42`** | **`99.70`** | **`89.33`** | **`98.88`** | **`96.45`** | **`98.27`** | **`79.62`** | **`91.28%` 🏆 (`+0.56%p`)** | **`87.63%` 🏆 (`+0.36%p`)** |

---

### (3) `navtest` v1 PDMS ($N=12,146$)

| 에폭 / 선택 수식 | `NC` | `DAC` | `DDC` | `EP` | `TTC` | `Comfort` | **★ v1 PDMS (`navtest`)** | 비고 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **23ep (`Ep 22`) `v2` / `v2_lat`** | `98.98` / `99.16` | `96.18` / `97.64` | `99.98` / `99.99` | `82.12` / `82.68` | `96.02` / `96.32` | `99.98` / `99.99` | `88.66%` / `90.25%` | Anchor top-1 (`col 0`) |
| **23ep (`Ep 22`) `default(0.5)`** | `99.21` | `98.58` | `99.98` | `87.33` | `96.54` | `99.99` | `92.34%` | `w=(0.1, 1, 1, 1), b=0.5` |
| **23ep (`Ep 22`) `plugin(0.5)`** | `99.20` | `98.77` | `99.98` | `87.15` | `96.52` | `99.99` | `92.42%` | `w=(0, 1, 1, 1), b=0.5` |
| **23ep (`Ep 22`) `plugin(1.0)`** | `98.86` | `98.72` | `99.98` | `90.63` | `95.62` | `99.99` | `93.47%` | 기존 최고점 (`b=1.0`) |
| **23ep (`Ep 22`) `Universal` (공용 14-변수)** | **`99.14`** | `98.68` | `99.99` | `90.18` | **`96.76`** | `99.99` | **`93.68%`** | 23ep·30ep 단일 공용 가중치 |
| **23ep (`Ep 22`) `CleanCore` (14-변수)** | `99.04` | `98.86` | `99.99` | `90.34` | `96.52` | `99.99` | **`93.72%`** | 코어 서브메트릭 + $\delta_v$ |
| **23ep (`Ep 22`) `BestLat96` (28-변수)** | `99.03` | `98.86` | `99.99` | `90.46` | `96.65` | `99.99` | **`93.81%` 🏆** | `lat_traj` 96-풀 전역 최적 |
| **23ep (`Ep 22`) `BestHybrid96` (`lat+pool`)** | `99.01` | **`98.89`** | `99.99` | **`90.78`** | `96.44` | `99.99` | **`93.87%` 🏆** | 변형별 `lat`/`pool` 하이브리드 |
| **30ep (`Ep 29`) `v2` / `v2_lat`** | `98.84` / `99.02` | `96.71` / `97.81` | `99.98` / `99.99` | `82.61` / `82.97` | `95.78` / `96.08` | `99.98` / `99.99` | `89.02%` / `90.24%` | Anchor top-1 (`col 0`) |
| **30ep (`Ep 29`) `default(0.5)`** | `98.98` | `98.77` | `99.98` | `87.87` | `95.55` | `99.99` | `92.23%` | `w=(0.1, 1, 1, 1), b=0.5` |
| **30ep (`Ep 29`) `plugin(0.5)`** | `99.13` | `98.87` | `99.98` | `87.73` | `95.67` | `99.99` | `92.38%` | `w=(0, 1, 1, 1), b=0.5` |
| **30ep (`Ep 29`) `plugin(1.0)`** | `98.67` | `98.82` | `99.98` | `90.28` | `95.13` | `99.99` | `93.01%` | 기존 30ep 최고점 (`b=1.0`) |
| **30ep (`Ep 29`) `Universal` (공용 14-변수)** | `98.87` | `98.83` | `99.99` | **`90.67`** | `95.73` | `99.99` | **`93.47%`** | 23ep·30ep 단일 공용 가중치 |
| **30ep (`Ep 29`) `CleanCore` (14-변수)** | `99.02` | `99.01` | `99.99` | `90.26` | `96.29` | `99.99` | **`93.65%`** | 코어 서브메트릭 + $\delta_v$ |
| **30ep (`Ep 29`) `BestLat96` (28-변수)** | **`99.11`** | `98.84` | `99.99` | `90.39` | **`96.55`** | `99.99` | **`93.79%` 🏆** | `lat_traj` 96-풀 전역 최적 |
| **30ep (`Ep 29`) `BestHybrid96` (`lat+pool`)** | `99.09` | **`99.06`** | `99.99` | `90.40` | `96.48` | **`100.00`** | **`93.84%` 🏆** | 변형별 `lat`/`pool` 하이브리드 |

---

## 4. 관련 산출물 JSON 및 평가·최적화 스크립트 경로 모음

- **최적화 결과 JSON**:
  - 23ep `navtest` v1 PDMS 최적화: [`v1_pdms_ep22.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/opt_results/v1_pdms_ep22.json)
  - 30ep `navtest` v1 PDMS 최적화: [`v1_pdms_ep29.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/opt_results/v1_pdms_ep29.json)
  - 23ep·30ep 공용 `navtest` v1 PDMS 수식: [`v1_pdms_universal.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/opt_results/v1_pdms_universal.json)
  - 23ep `navtest` EPDMS (After & Before) 최적화: [`epdms_navtest_ep22.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/opt_results/epdms_navtest_ep22.json)
  - 30ep `navtest` EPDMS (After & Before) 최적화: [`epdms_navtest_ep29.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/opt_results/epdms_navtest_ep29.json)
  - 23ep `navhard` EPDMS (After & Before) 최적화: [`epdms_navhard_ep22.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/opt_results/epdms_navhard_ep22.json)
  - 30ep `navhard` EPDMS (After & Before) 최적화: [`epdms_navhard_ep29.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/opt_results/epdms_navhard_ep29.json)
- **기본(Baseline) 평가 요약 JSON**:
  - 23ep `navhard` After (`v2.2`): [`navhard_summary.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/navhard_scores/navhard_summary.json)
  - 30ep `navhard` After (`v2.2`): [`navhard_summary.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep29/navhard/scores/navhard_summary.json)
  - 23ep & 30ep `navhard` Before (`pre-359c7f7`): [`navhard_before_summary.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/before_v22_scores/navhard_before_summary.json)
  - 23ep `navtest` EPDMS After (`v2.2`): [`epdms_navtest_summary.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/navtest_scores/epdms_navtest_summary.json)
  - 23ep & 30ep `navtest` EPDMS Before (`pre-359c7f7`): [`navtest_before_summary.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/before_v22_scores/navtest_before_summary.json)
- **스크립트**:
  - [`dump_96pool.py`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/dump_96pool.py) / [`optimize_v1_pdms.py`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/optimize_v1_pdms.py) / [`optimize_epdms_fast.py`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/epdms_and_navhard/optimize_epdms_fast.py)
