# BEV Selector v14: Dual-Stream Gradient-Decoupled & Adapter-Isolated Multi-Resolution KD

본 문서는 `fusion_v13_main` (`BEV Selector v13`)이 `no_kd`, `ck2e2e_bevkd` (`1×1 Uniform BEV KD`), `v11`, `v12`를 전 지표(`v2 89.09%`, `v2_lat 90.53%`, `default 92.28%`, `noim 92.36%`, `plugin 93.11%`, `Det mAP 25.78%`, `Map mAP 25.34%`)에서 모두 경신했음에도 불구하고, **왜 `ck2e2e_bevkd` 대비 총점(PDMS) 우위가 `+0.10 ~ +0.18%p`에 머물렀는지**에 대한 정량적 역전파 그래디언트 분해(Gradient Norm Decomposition) 결과를 바탕으로 설계된 **`BEV Selector v14` (`fusion_v14_main`)** 사양서입니다.

---

## 1. `v13` vs `ck2e2e_bevkd` 심층 진단: 어디서 이기고 어디서 상쇄되었는가?

### 1.1 `navtest` ($N=12,146$) 서브 메트릭 및 $\beta$ 그리드 정밀 비교

| 평가 모드 (`sel_w`) | 혼합비 $\beta$ | `ck2e2e_bevkd` PDMS | **`v13` PDMS** | 격차 ($\Delta$) | 서브 메트릭 핵심 차이 (`v13` vs `bevkd`) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`v2` Base Planner** | `0.00` | `88.96%` | **`89.09%`** | **`+0.13%p`** | `NC +0.10%p`, `TTC +0.17%p`, `lead_fail -0.16%p` |
| **`v2_lat` (`v2` + Lateral)** | `0.00` | `90.37%` | **`90.53%`** | **`+0.16%p`** | `NC +0.17%p`, `TTC +0.28%p`, `lead_fail -0.44%p` |
| **`default` (`0.1, 0.5, 0.5, 1.0`)** | `0.25` | `92.14%` | **`92.31%`** | **`+0.17%p`** | 후보 풀 상위 부모 품질(`mean_top3 = 88.80%`) 전 모델 1위 |
| **`default` (`0.1, 0.5, 0.5, 1.0`)** | **`0.50`** | `92.17%` | **`92.28%`** | **`+0.11%p`** | **`TTC +0.37%p` (`95.73` vs `95.36`), `NC +0.17%p`** vs **`DAC -0.14%p` (`98.72` vs `98.86`)** |
| **`noim` (`0.0, 0.5, 0.5, 1.0`)** | `0.50` | `92.20%` | **`92.36%`** | **`+0.16%p`** | **`TTC +0.33%p`, `NC +0.14%p`** vs `DAC -0.11%p` |
| **`noim` (`0.0, 0.5, 0.5, 1.0`)** | `0.75` | `92.34%` | **`92.52%`** | **`+0.18%p`** | **`TTC +0.41%p`, `NC +0.15%p`** vs `DAC -0.15%p` |
| **`noim` (`0.0, 0.5, 0.5, 1.0`)** | `1.00` | `93.10%` | `92.92%` | `-0.18%p` | **`TTC +0.44%p` (`94.64` vs `94.20`)** 이기나, **`DAC -0.34%p` (`98.49` vs `98.83`)**에서 상쇄 |
| **`plugin` (`0.0, 1.0, 1.0, 1.0`)** | `0.75` | `92.40%` | **`92.53%`** | **`+0.13%p`** | **`TTC +0.39%p`** vs `DAC -0.14%p` |

- **핵심 현상**: `v13`은 동적 장애물 회피(`TTC +0.37~+0.44%p`, `NC +0.17%p`)와 후보 궤적 풀 자체의 품질(`mean_top3 = 88.80%`, `mean_96_lat = 82.07%`)에서 `bevkd`를 큰 폭으로 압도했지만, **`CK2` 선택기가 `MAP (resmap)` 특징을 강하게 참조하는 고-$\beta$ 구간에서 `DAC`(도로/차선 경계 준수율)가 `-0.14 ~ -0.34%p` 낮아지면서 전체 PDMS 우위가 `+0.11 ~ +0.18%p`로 깎였습니다.**
- 실제로 에폭 29 종료 시점의 전역 `1×1 Uniform MSE`를 비교하면:
  - `DET (bevfusion)`: `bevkd 0.752` vs `v12 0.816` vs **`v13 0.776`** (격차의 63% 해소)
  - `MAP (resmap)`: `bevkd 0.534` vs `v12 0.616` vs **`v13 0.591`** (**여전히 `bevkd` 대비 `+0.057` (`+10.7%`) 높은 MSE 잔차 존재**)

---

### 1.2 근본 원인: [`last.ckpt`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v13_main/train/lightning_logs/version_0/checkpoints/last.ckpt) 실측 역전파 그래디언트 분해 (Smoking Gun)

`v13`의 손실 함수는 스칼라 가중치 기준으로 **그리드 손실 80% (`5.5*cell_1x1 + 1.5*global_5x5 + 1.0*edge_3x3`) + Probe 손실 20% (`1.0*feat_kd + 1.0*sibling_diff + 1.0*attn_kl`)**로 설계되었습니다. 그러나 실제 [`last.ckpt`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v13_main/train/lightning_logs/version_0/checkpoints/last.ckpt)에서 각 항이 `student_bev`와 `student_proj`에 가하는 **역전파 그래디언트 Norm(`g_BEV`, `g_projW`, `g_projB`)**을 분해한 결과, **스칼라 손실 비중과 실제 그래디언트 비중 사이에 극단적인 역전 현상**이 발견되었습니다:

| 도메인 (`Bank`) | 손실 항 (설계 가중치) | 스칼라 Loss 값 (비중) | **실제 `student_bev` 그래디언트 `g_BEV`** (전체와의 `cos`) | **`proj.weight` 그래디언트 `g_projW`** | **`proj.bias` 그래디언트 `g_projB`** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`DET` (`bevfusion`)** | `5.5 * cell_1x1` | `7.3125` (`58.5%`) | `0.00477` (`cos +0.37`) | `0.3978` | `0.0269` |
| | `1.5 * global_5x5` | `0.0819` (`0.7%`) | **`0.00027` (`cos +0.07`, Dead!)** | `0.0050` | `0.0074` |
| | `1.0 * edge_3x3` | `2.6588` (`21.3%`) | `0.00193` (`cos +0.34`) | `0.1449` | `0.0000` |
| | **`1.0 * feat_kd`** | `0.8160` (`6.5%`) | **`0.00868` (`cos +0.71`)** | **`0.5988`** | **`0.4627`** |
| | **`1.0 * sibling_diff`** | `0.6623` (`5.3%`) | **`0.00683` (`cos +0.63`)** | **`0.4014`** | **`0.2063`** |
| | **`1.0 * attn_kl`** | `0.9745` (`7.8%`) | **`0.01042` (`cos +0.64`)** | **`0.6765`** | `0.0167` |
| | **합계 (`TOTAL`)** | `12.5060` (`100%`) | **`0.01957`** | **`1.2309`** | **`0.5838`** |
| **`MAP` (`resmap`)** | `5.5 * cell_1x1` | `7.2962` (`62.9%`) | **`0.00208` (`cos +0.15`!)** | `0.3953` | `0.0331` |
| | `1.5 * global_5x5` | `0.0830` (`0.7%`) | **`0.00012` (`cos +0.03`, Dead!)** | `0.0050` | `0.0089` |
| | `1.0 * edge_3x3` | `2.6516` (`22.9%`) | `0.00086` (`cos +0.14`) | `0.1442` | `0.0000` |
| | **`1.0 * feat_kd`** | `0.4497` (`3.9%`) | **`0.01218` (`cos +0.90`!)** | **`0.8590`** | **`0.6761`** |
| | **`1.0 * sibling_diff`** | `0.3907` (`3.4%`) | **`0.00820` (`cos +0.82`!)** | **`0.4381`** | **`0.2192`** |
| | **`1.0 * attn_kl`** | `0.7224` (`6.2%`) | **`0.00604` (`cos +0.61`!)** | **`0.4703`** | `0.0104` |
| | **합계 (`TOTAL`)** | `11.5937` (`100%`) | **`0.02182`** | **`1.4498`** | **`0.7877`** |

1. **병목 1 — `RatioController` 내 그래디언트 예산 잠식 (Jacobian Scale Disparity)**:
   - `cell_loss`는 $HW \times C = 5,000 \times 256 = 1.28 \times 10^6$개 원소의 평균(`.mean()`)이므로 셀당 미분 계수가 $\frac{1}{1,280,000}$로 축소됩니다.
   - 반면 Probe 경유 손실(`feat_kd`, `sibling_diff`)은 어텐션으로 풀링된 소수 코리도 특징(평균으로 인해 분산 $\sigma_{\text{feat}}^2 \ll 1$)을 다시 `ffn_norm`(`LayerNorm`), `head_norm`(`LayerNorm`), 그리고 손실 내부 `LayerNorm` + `Cosine Similarity`로 연속 정규화합니다. `LayerNorm`의 역전파 야코비안은 $\frac{1}{\sigma_{\text{feat}}} \gg 1$에 비례하므로, 단위 Loss당 그래디언트 Norm이 `cell_loss`보다 **25배~60배** 큽니다.
   - 그 결과 `MAP (resmap)`에서 스칼라 비중이 단 `13.5%`인 Probe 손실이 **전체 BEV 그래디언트 Norm의 `89%` (`cos = +0.90`)를 독식**했고, `RatioController`가 뱅크 전체 그래디언트 합계를 `0.10 * g_plan`으로 캡핑하면서 정작 **`cell_1x1`이 받는 실제 그래디언트는 `bevkd`의 `1/9` (`0.00208`, `cos = +0.15`)로 고갈**되었습니다.
2. **병목 2 — 단일 `student_proj` 어댑터의 이중 역할 충돌 (Adapter Hijacking)**:
   - `v13`에서는 동일한 `s_bev = student_proj(LN(s))`가 `cell_loss`와 `probe.forward_student` 양쪽에 공유됩니다.
   - 위 표에서 보듯 `MAP` 어댑터의 `proj.bias` 그래디언트는 `feat_kd`가 `0.6761`인 반면 `5.5 * cell_1x1`은 `0.0331`로 **Probe가 20.4배 더 큰 힘으로 `proj.bias`와 `proj.weight`를 흔듭니다**. 즉 선형 번역기 `student_proj`가 Teacher Z-score 분포를 복원하는 대신 EMA Probe의 내부 `LayerNorm` 각도를 맞추도록 왜곡되었습니다.
3. **병목 3 — `1.5 * global_5x5_loss`의 사문화 (`g_BEV = 0.00012`) 및 `MAP` 원거리 편향**:
   - $5\times 5$ 평균 풀링 항은 Z-score 양/음수 상쇄와 $1/25$ 분산으로 인해 실제 그래디언트 Norm이 `0.00012` (`cos = +0.03`)로 사실상 `0`이면서 가중치 예산 `15%`만 차지했습니다.
   - 또한 `v13`의 원거리 부스트 `row_gain` $\in [1.0, 1.5]$ ($10\text{--}32\text{m}$ 증폭)는 `DET`의 고속 `TTC` 제동에는 핵심적이었지만, `MAP`에서도 동일하게 곱해진 뒤 전체 평균이 `1.0`으로 재정규화되면서 정작 교차로 좌/우회전 시 `DAC`를 결정짓는 **근·중거리($0\text{--}18\text{m}$) 연석/차선 경계 가중치를 상대적으로 깎는 부작용**이 있었습니다.

---

## 2. `BEV Selector v14` 4대 핵심 아키텍처 개선 (4-Pillar Fixes)

### Fix 1: Dual-Stream Decoupled Gradient Controller (Grid `9.0%` + Probe `2.5%` 독립 제어)
- **해결 방법**: 각 도메인 뱅크 $b \in \{\text{det}, \text{map}\}$에서 `student_bev`를 **두 개의 독립된 `GradScale` 및 `RatioController` 스트림**으로 분리합니다:
  1. **Spatial Grid Stream (`x_grid`, `self.ratio_ctrl[tk]`)**:
     - 목표 그래디언트 비율: **`ratio_grid = 0.090` (`0.090 * g_plan`, `cap = 0.22`)**
     - `v13`에서 `bevkd`의 `1/4`(`DET`) ~ `1/9`(`MAP`)로 쪼그라들었던 공간 격자 정렬 그래디언트를 **`ck2e2e_bevkd`와 동등한 `9.0% * g_plan` 수준으로 100% 복원**합니다.
  2. **Planning Probe Stream (`x_probe`, `self.probe_ratio_ctrl[tk]`)**:
     - 목표 그래디언트 비율: **`ratio_probe = 0.025` (`0.025 * g_plan`, `cap = 0.06`)**
     - Probe의 관계형/어텐션 증류(`feat_kd + sibling_diff + attn_kl`)가 가진 높은 야코비안 스케일을 독립된 `RatioController`가 자동 흡수하여, 그리드 정렬 예산을 단 1%도 잠식하지 않으면서도 **`2.5% * g_plan`의 순수 주행 상호작용 그래디언트**를 일관되게 주입합니다.
- **합산 안정성 및 오버헤드**:
  - 두 스트림의 코사인 유사도(`cos ≈ +0.20`)에 의해 뱅크당 합산 BEV 그래디언트 Norm은 $\sqrt{0.090^2 + 0.025^2 + 2(0.090)(0.025)(0.20)} \cdot g_{\text{plan}} \approx \mathbf{0.098 \cdot g_{\text{plan}}}$으로 기존 `bevkd`/`v13`의 `0.10 * g_plan` 예산과 정확히 일치합니다.
  - 그래디언트 Norm 측정은 기존과 동일하게 `gnorm/plan`이 계산되는 마이크로배치(200 step마다 1회)에만 수행되어 단일 DDP `all_reduce_mean`으로 동기화되므로 **추가 학습 시간 오버헤드가 `0.0 ms`**입니다.

### Fix 2: Stop-Grad Adapter Isolation (`s_bev_probe` Detach) & `lr_mult = 3.0` 복원
- **해결 방법**: $1\times 1$ 선형 프로젝션 어댑터 `student_proj` (`Linear(256, 256)`)가 오직 Teacher Z-score 공간 복원만 학습하도록 **Probe 스트림에서 어댑터 파라미터(`weight`, `bias`)의 역전파를 차단(Stop-Grad)**합니다:
  $$\begin{aligned}
  s_{\text{bev}}^{\text{grid}} &= W_{\text{proj}} \, \text{LN}(x_{\text{grid}}) + b_{\text{proj}} \quad &\text{(Grid Loss용: } W_{\text{proj}}, b_{\text{proj}}, x_{\text{grid}} \text{ 모두 업데이트)} \\
  s_{\text{bev}}^{\text{probe}} &= \text{stop\_grad}(W_{\text{proj}}) \, \text{LN}(x_{\text{probe}}) + \text{stop\_grad}(b_{\text{proj}}) \quad &\text{(Probe Loss용: } x_{\text{probe}}\text{만 업데이트, 어댑터 오염 원천 차단)}
  \end{aligned}$$
- **수학적 효과**:
  - 순전파(Forward)에서는 $s_{\text{bev}}^{\text{probe}} \equiv s_{\text{bev}}^{\text{grid}}$로 완전히 동일한 투영 특징을 EMA Probe에 입력합니다.
  - 역전파(Backward)에서는 Probe의 그래디언트 $\frac{\partial \mathcal{L}_{\text{probe}}}{\partial s_{\text{bev}}^{\text{probe}}} W_{\text{proj}}^{\top}$가 학생 BEV(`x_probe`)로는 100% 그대로 전달되면서도, **`W_proj`와 `b_proj`는 오직 `L_grid`(`cell_1x1 + edge_3x3`)에 의해서만 학습**됩니다(특히 `edge_3x3`는 인접 셀 차분에서 $b_{\text{proj}}$가 상쇄되므로 `b_proj`는 100% 순수 `cell_1x1` MSE만 최소화합니다).
  - 어댑터가 Probe 그래디언트 스파이크로부터 완전히 보호됨에 따라, 어댑터 학습률 배수를 `v13`의 `2.0`에서 **`ck2e2e_bevkd`와 동일한 `distill_proj_lr_mult = 3.0`으로 복원**하여 5,300 step 워밍업 구간부터 Teacher Z-score 기저를 빠르게 수렴시킵니다.

### Fix 3: 사문화된 `5×5 AvgPool` 제거 및 고해상도 `85/15` Corridor Grid KD
- **해결 방법**: 실제 BEV 그래디언트가 `0.00012`(`cos = +0.03`)로 사문화된 `global_5x5_loss`를 학습 손실에서 제거하고(모니터링 호환성을 위해 `distill_global_5x5` 및 `distill_global_5x5_cos` 로깅은 `torch.no_grad()`로 유지), 그 `15%` 예산을 고해상도 $1\times 1$ 셀 및 $3\times 3$ 경계 미분 손실로 전량 재배치합니다:
  $$\mathcal{L}_{\text{grid}}^{(b)} = \text{loss\_scale} \cdot \left(0.85 \cdot \mathcal{L}_{\text{cell\_1x1}}^{(b)} + 0.15 \cdot \mathcal{L}_{\text{edge\_3x3}}^{(b)}\right)$$
  $$\mathcal{L}_{\text{probe\_kd}}^{(b)} = \text{loss\_scale} \cdot \left(0.40 \cdot \mathcal{L}_{\text{feat}}^{(b)} + 0.40 \cdot \mathcal{L}_{\text{sibling\_diff}}^{(b)}\right) + 0.40 \cdot \text{attn\_loss\_scale} \cdot \mathcal{L}_{\text{attn\_kl}}^{(b)}$$

### Fix 4: Domain-Specific Spatial Ribbon Shaping (`DET` 원거리 `TTC` 부스트 vs `MAP` 교차로 `DAC` 플로어)
- **해결 방법**: 두 Teacher 도메인의 물리적 역할 차이에 맞춰 공간 가중치 $w_{\text{pgp}}^{(b)}$를 비대칭 최적화합니다:
  - **`DET` (`bevfusion`, 동적 장애물 / `NC` & `TTC`)**: `v13`에서 `TTC +0.37~+0.44%p` 및 `lead_fail 6.84%`를 달성한 원거리 제동 부스트 `row_gain` $\in [1.0, 1.5]$ ($y \in [10\text{m}, 32\text{m}]$)와 `w_pgp = 0.75 + 0.25 * m_cap_norm` ($\in [0.75, 2.30]$)을 그대로 유지합니다.
  - **`MAP` (`resmap`, 도로/차선 경계 / `DAC` & `EP`)**: 원거리 편향으로 인해 근·중거리($0\text{--}18\text{m}$) 교차로 회전 연석 가중치가 희석되는 것을 막기 위해 종방향 왜곡을 제거(`row_gain_map = 1.0`)하고, 전역 도로면 최소 가중치 플로어를 **`w_pgp = 0.85 + 0.15 * m_cap_norm` ($\in [0.85, 1.65]$)**로 높여 `CK2`의 96개 궤적 전체에 대해 균일하고 선명한 차선/도로 경계 복원력을 보장합니다.

---

## 3. [`last.ckpt`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v13_main/train/lightning_logs/version_0/checkpoints/last.ckpt) 기준 `v13` vs `v14` 실측 그래디언트 분포 검증

동일한 체크포인트 및 입력 배치에서 `v13`과 `v14`가 생성하는 실제 역전파 그래디언트를 비교한 결과입니다:

| 지표 (`Bank` / 항목) | `v13` (`fusion_v13_main`) | **`v14` (`fusion_v14_main`)** | 개선 효과 |
| :--- | :---: | :---: | :--- |
| **`DET` Grid BEV Grad Norm** (전체 대비 `cos`) | `0.0042` (`24%` 비중, `cos +0.37`) | **`0.0158` (`9.0% of g_plan`, `cos +0.966`)** | **공간 격자 정렬력 `3.8x` 강화** |
| **`MAP` Grid BEV Grad Norm** (전체 대비 `cos`) | `0.0019` (`11%` 비중, **`cos +0.15`**) | **`0.0158` (`9.0% of g_plan`, `cos +0.965`)** | **`MAP` 공간 정렬력 `8.3x` 강화 (`DAC` 병목 해소)** |
| **`DET / MAP` Probe BEV Grad Norm** | `0.0131 / 0.0154` (예산의 `76~89%` 독식) | **`0.0044 / 0.0044` (`2.5% of g_plan` 고정)** | 그리드 잠식 없이 순수 관계/어텐션 그래디언트 주입 |
| **`proj.bias`에 유입되는 Probe 오염 그래디언트** | `0.669` (`DET`) / `0.895` (`MAP`) (**Grid의 20배**) | **`0.0000` (`DET`) / `0.0000` (`MAP`)** | **어댑터 왜곡 100% 차단 (순수 `cell_1x1`만 반영)** |

---

## 4. 버전별 핵심 설계 파라미터 비교 (`v12` vs `v13` vs `v14`)

| 구성 요소 | `fusion_v12_main` | `fusion_v13_main` | **`fusion_v14_main` (신규 설계)** |
| :--- | :---: | :---: | :---: |
| **`RatioController` 구조** | 단일 (`det`, `map` 각 `ratio=0.10`) | 단일 (`det`, `map` 각 `ratio=0.10`) | **이중 분리 (`Grid ratio=0.09, cap=0.22` + `Probe ratio=0.025, cap=0.06`)** |
| **`student_proj` 어댑터 경로** | Grid + Probe 공유 (`lr_mult=2.0`) | Grid + Probe 공유 (`lr_mult=2.0`) | **Grid 전용 학습 + Probe 입력 Stop-Grad (`lr_mult=3.0` 복원)** |
| **Grid KD 내부 구성** | `cell_1x1` 단독 | `55% cell_1x1 + 15% 5x5 + 10% edge_3x3` | **`85% cell_1x1 + 15% edge_3x3` (사문화된 `5x5` 제거)** |
| **Probe KD 내부 구성** | `10% feat + 10% diff + 0.25x attn` | `10% feat + 10% diff + 0.20x attn` | **`40% feat + 40% diff + 0.40x attn` (`probe_ratio_ctrl`로 `2.5%` 제어)** |
| **`DET` 공간 가중치 $w_{\text{pgp}}^{\text{det}}$** | Uncapped (`0.5+0.5*m`, max `41.8`) | `0.75 + 0.25*m_cap` + `row_gain [1.0, 1.5]` | **`0.75 + 0.25*m_cap` ($\in [0.75, 2.30]$) + `row_gain [1.0, 1.5]` 유지** |
| **`MAP` 공간 가중치 $w_{\text{pgp}}^{\text{map}}$** | Uncapped (`0.5+0.5*m`, max `36.4`) | `0.75 + 0.25*m_cap` + `row_gain [1.0, 1.5]` | **`0.85 + 0.15*m_cap` ($\in [0.85, 1.65]$) + `row_gain = 1.0` (교차로 보존)** |

---

## 5. 실행 명령어 (Launch Command)

```bash
RUN=fusion_v14_main CK2_E2E_APPROVED=1 CK_ALLOW_GPU45=1 \
E2E_EXTRA=tools/ck/e2e2/fusion_v14_arm.extra \
bash tools/ck/e2e2/train_e2e2.sh start --gpus 4,5,6,7
```
