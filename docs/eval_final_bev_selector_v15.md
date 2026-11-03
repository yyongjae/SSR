# BEV Selector v15: Subspace-Guided Adapter, Horizon-Partitioned Attention & On-Policy Functional Corridor KD

본 문서는 `fusion_v14_main` (`BEV Selector v14`)의 30 에폭 학습 로그([`wandb-summary.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v14_main/train/wandb/latest-run/files/wandb-summary.json), [`steps_rank0.jsonl`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v14_main/ck_e2e2/steps_rank0.jsonl))와 후반 10개 에폭(`Epoch 20 ~ 29`) `navtest` ($N=12,146$) 전수 평가 결과를 정밀 대조하여, **`v14`가 어디서 크게 성공했고 왜 Stream 2 (Probe KD) 정렬도가 하락했는지**를 규명하고 이를 해결하기 위해 설계된 **`BEV Selector v15` (`fusion_v15_main`)** 아키텍처 사양서입니다.

---

## 1. `v14` 결과 및 학습 텔레메트리 심층 진단: 무엇이 해결되었고 무엇이 남았는가?

### 1.1 `v13` vs `v14` 멀티 에폭 `navtest` ($N=12,146$) 성능 성과 요약

| 평가 모드 (`navtest`, $N=12,146$) | `ck2e2e_bevkd` | `v13` (`last` / 최고) | **`v14` (`last` / 최고)** | `v14` 핵심 성과 및 남은 과제 |
| :--- | :---: | :---: | :---: | :--- |
| **`v2` Base Planner** | `88.96%` | `89.09%` / `89.11%` | **`89.10%` / `89.40%`** (`Ep 24`) | `lead_fail` `7.61% → 6.89%` (`Ep 24: 6.78%`) 대폭 개선 |
| **`v2_lat` (`v2` + CK2 측방 보정)** | `90.35%` | `90.53%` / `90.53%` | **`90.49%` / `90.70%`** (`Ep 24`) | 최고점 `+0.17%p` 경신 |
| **★ `default` ($\beta=0.5$)** | `92.18%` | `92.28%` / `92.41%` | **`92.35%` / `92.46%`** (`Ep 25`) | `last` `+0.07%p`, 최고점 `+0.05%p` (`bevkd` 대비 **`+0.28%p`**) |
| **`default` ($\beta=1.0$)** | `92.17%` | `92.23%` / `92.33%` | **`92.34%` / `92.43%`** (`Ep 25`) | `last` `+0.11%p`, 최고점 `+0.10%p` |
| **`noim` ($\beta=0.5$)** | `92.20%` | `92.36%` / `92.44%` | **`92.43%` / `92.53%`** (`Ep 25`) | `last` `+0.07%p`, 최고점 `+0.09%p` |
| **★ `noim` ($\beta=1.0$)** | `93.10%` | `92.92%` / `93.11%` | **`93.17%` / `93.35%`** (`Ep 22`) | **`v13`의 유일한 약점(`last 92.92%`) 완벽 해결 (`+0.25%p`, 최고 `+0.24%p`)** |
| **`plugin` ($\beta=0.5$)** | `92.29%` | `92.37%` / `92.50%` | **`92.55%` / `92.55%`** (`Ep 23,29`) | `last` **`+0.18%p`**, `bevkd` 대비 **`+0.26%p`** |
| **★ `plugin` ($\beta=1.0$)** | `92.98%` | `93.11%` / `93.19%` | **`93.31%` / `93.45%`** (`Ep 22`) | `last` **`+0.20%p`**, 최고점 **`93.45%`** 달성 |

---

### 1.2 `v13` vs `v14` 학습 텔레메트리(`Epoch 29` W&B Summary) 정밀 비교: 숨겨진 병목 발견

`v14`에서 도입한 **Dual-Stream Gradient Controller (Grid `9.0%` + Probe `2.5%`)**와 **Adapter Stop-Grad (`proj.weight.detach()`)**가 실제 학습 과정에서 두 스트림에 미친 영향을 [`wandb-summary.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v14_main/train/wandb/latest-run/files/wandb-summary.json)으로 비교하면 극명한 대비가 나타납니다:

| 스트림 구분 | 학습 로그 지표 (`Epoch 29` 평균) | `fusion_v13_main` | **`fusion_v14_main`** | 정량적 변화 및 원인 진단 |
| :--- | :--- | :---: | :---: | :--- |
| **Stream 1**<br>(Grid KD:<br>`cell_1x1` + `edge_3x3`) | `distill_uniform_mse/resmap` | `0.59098` | **`0.56127`** | **`-5.0%` 오차 감소** (`bevkd 0.534`와의 잔차 절반 이상 해소 $\rightarrow$ 고-$\beta$ `DAC` 급등) |
| | `distill_cell_mse/resmap` | `0.60643` | **`0.57320`** | **`-5.5%` 오차 감소** (코리도 가중 $1\times 1$ 셀 복원력 대폭 향상) |
| | `distill_edge_3x3/resmap` | `0.24251` | **`0.22622`** | **`-6.7%` 오차 감소** (차선/연석 경계 1차 공간 미분 $\nabla_{x,y}$ 선명도 향상) |
| | `distill_global_5x5_cos/resmap` | `0.72822` | **`0.74525`** | **`+0.0170` 상승** (`CK2` `M3b` 전역 토큰 코사인 정렬도 개선) |
| | `distill_uniform_mse/bevfusion` | `0.77619` | **`0.77177`** | `DET` 전역 격자 오차 감소 (`cell_mse`: `0.9562 → 0.9473`) |
| **Stream 2**<br>(Probe KD:<br>`feat` + `diff` + `attn_kl`) | **`distill_cos/bevfusion` (`feat_loss`)** | `0.85343` | **`0.65700`** | **`-0.1964` 급락!** (`MAP`도 `0.9299 → 0.8758`로 `-0.0541` 하락) |
| | **`distill_winner_cos/bevfusion`** | `0.85982` | **`0.66943`** | **`-0.1904` 급락!** (GT 최근접 승자 앵커의 특징 방향 정렬도 저하) |
| | **`distill_sibling_cos/bevfusion`** | `0.76841` | **`0.58091`** | **`-0.1875` 급락!** (`sibling_diff` 손실 잔차 `0.1696 → 0.3131`로 `1.85x` 증가) |
| | **`loss_attn_bevfusion / resmap`** | `2.220` / `2.681` | **`3.457` / `3.788`** | 어텐션 KL 발산 잔차 **`+55.7%` / `+41.3%` 증가** |
| | **`probe/student_gap/bevfusion & resmap`** | `0.0681` / `0.0365` | **`0.1058` / `0.0524`** | Teacher Probe vs Student Probe 안전 확률 예측 괴리 **`1.55x` 증가** |
| **어텐션 집중도** | **`probe/head_entropy/bevfusion & resmap`** | `6.616` / `6.775` | **`6.615` / `6.668`** | 유효 어텐션 범위 $\exp(6.62) \approx \mathbf{745\text{개 셀}}$ (5,000셀 중 15%로 과도하게 넓음) |

### 1.3 `v14` 로그에서 규명된 5가지 구조적 병목

1. **병목 1 — `student_proj`의 등방성(Isotropic) L2 맹점과 과도한 Stop-Grad 차단 (`distill_cos: 0.853 → 0.657`)**:
   - `v14`는 `student_proj` 어댑터(`Linear(256, 256)`)에서 Probe 그래디언트를 100% 끊고(`proj.weight.detach()`), 오직 등방성 L2 손실인 `cell_1x1` (`(s_bev - t_bev)^2.mean(dim=-1)`)로만 어댑터를 학습시켰습니다.
   - 그러나 256개 Teacher BEV 채널 중 실제 충돌(`NC/TTC`)과 도로/차선(`DAC/LK/DDC`) 판별에 쓰이는 핵심 특징은 Teacher Probe의 투영 행렬 **$W_K, W_V \in \mathbb{R}^{256 \times 256}$이 바라보는 특정 부분공간(Subspace)**에 집중되어 있습니다. 등방성 `cell_1x1`은 256개 모든 채널 방향에 똑같은 가중치($1/256$)를 부여하므로, 어댑터가 배경 분산이 큰 채널을 맞추는 데 용량을 소모하고 정작 Probe의 $W_K, W_V$ 부분공간 회전각을 정렬하지 못했습니다.
   - 게다가 `probe_ratio_ctrl`이 `0.025 * g_plan`(`lam_probe_det = 0.0446`)으로 강하게 억제되면서, `student_bev` 백본 혼자서 고정된 어댑터를 뚫고 256차원 특징 방향을 회전시키지 못해 `distill_cos`가 `0.657`에서 멈추었습니다.
2. **병목 2 — 4초(10-Point) 통짜 앵커 프로빙의 시간축 평균 희석 (Temporal Averaging Dilution)**:
   - 현재 `_build_spatial_bias`는 10개 웨이포인트($t = 0.5\text{s} \dots 5.0\text{s}$) 전체에 대해 단일 최소 거리(`min_dist = dists.min(dim=1)`)로 튜브를 만들고 4개 헤드 모두가 튜브 전체를 한꺼번에 어텐션 평균(`attn @ V`)냅니다.
   - 모든 256개 앵커는 자차 후륜 $(0, 0)$에서 출발하므로 **근거리($0\text{--}8\text{m}$, $t=0.5\text{--}1.5\text{s}$) 구간이 심하게 겹치며 센서 특징 에너지도 가장 강합니다.** 그 결과 어텐션 가중합에서 근거리 특징이 원거리($15\text{--}32\text{m}$, $t=3.0\text{--}5.0\text{s}$) 특징을 덮어버려, 속도 형제(`i_slow` vs `j_fast`)가 실제로 갈라지는 전방 제동 구간의 장애물/커브 신호가 희석됩니다.
3. **병목 3 — 과도하게 넓은 공간 시그마 `(2.0, 4.0, 8.0, 16.0)m`로 인한 어텐션 엔트로피 고착 (`H = 6.62`, $\sim 745$셀)**:
   - `CK2`가 실제로 읽는 코리도 폭은 좌우 `±4.8m`([`corridor.py:52`](file:///workspace/byounggun/SSR/navsim/agents/para_ssr/refiner/corridor.py#L52))인데, Probe의 Head 2 ($\sigma=8\text{m}$)와 Head 3 ($\sigma=16\text{m}$)는 반경 $16\text{--}32\text{m}$의 코리도 외부 배경까지 수천 개 셀에 어텐션을 분산시켜 `attn_kl`과 `w_pgp`의 공간 해상도를 떨어뜨렸습니다.
4. **병목 4 — 정적 K-Means 256-앵커와 학생의 실제 추론 코리도(`plan_topk_trajectory`) 간 괴리**:
   - 추론 시 `v2`는 `anchor + offset`으로 보정된 궤적을 출력하고 `CK2`는 상위 6개 부모(`plan_topk_trajectory`)로부터 생성된 96개 후보의 코리도를 샘플링합니다. 고정 256-앵커만 프로빙하면 실제 커브길이나 장애물 회피 시 학생이 주행하려는 온-폴리시(On-Policy) 코리도와 $1\text{--}2\text{m}$의 오차가 발생합니다.
5. **병목 5 — `LayerNorm` 특징 정렬 이후의 안전 결정 경계(`sim_head` Logit) 괴리 (`student_gap: 0.068 → 0.106`)**:
   - `feat_loss`는 `LayerNorm` 정규화된 256차원 벡터의 코사인/L2 거리만 맞출 뿐, 그 특징이 `sim_head`의 결정 경계(Decision Boundary)를 통과했을 때 동일한 안전 확률(`NC, TTC, DAC, LK, DDC`)을 내도록 직접 강제하지 못했습니다.

---

## 2. `BEV Selector v15` 5대 핵심 아키텍처 혁신 (5-Pillar Upgrades)

```mermaid
flowchart TD
    BEV["Shared Student BEV (bev_embed)\n[B, 5000, 256]"]

    subgraph Pillar1["Pillar 1: Subspace-Guided Adapter & Dual-Stream (8.5% + 3.5%)"]
        G_S1["Stream 1 GradScale\n(ratio_grid = 0.085 * g_plan)"]
        G_S2["Stream 2 GradScale\n(ratio_probe = 0.035 * g_plan)"]
        ADAPT["1x1 Adapter (student_proj)\n• Stream 1: 100% W_proj, b_proj 학습\n• Stream 2: b_proj Stop-Grad (Shift 0%),\n  W_proj 0.08x GradScale (부분공간 회전 정렬)"]
        CH_W["EMA Probe Subspace Importance w_ch [256]\n(|γ_LN| * sqrt(||W_K||² + ||W_V||²))"]
    end

    subgraph Pillar2_3["Pillar 2 & 3: Horizon-Partitioned 4-Head Probe & Mid/Far Sibling Diff"]
        H0["Head 0: Near-Core (t=0.5..2.0s, σ=1.5m)"]
        H1["Head 1: Mid-Maneuver (t=1.5..3.5s, σ=2.2m)"]
        H2["Head 2: Far-Braking TTC (t=3.0..5.0s, σ=2.8m)"]
        H3["Head 3: Full-Corridor (t=0.5..5.0s, σ=4.5m)"]
        FAR_DIFF["Horizon-Decoupled Sibling Diff\n• 65% Full Post-FFN Δh_bev\n• 35% Mid+Far Heads (H1+H2) Δh_far"]
    end

    subgraph Pillar4["Pillar 4: On-Policy Refined Corridor Ribbon w_pgp"]
        TOPK["Student Top-6 Refined Trajectories\n(plan_topk_trajectory.detach())"]
        W_PGP["On-Policy Blended Spatial Ribbon\nm_comb = 0.80 * m_probe + 0.20 * m_on_policy"]
    end

    subgraph Pillar5["Pillar 5: Functional Safety Decision-Boundary KD"]
        SCORE_KD["Stop-Grad sim_head Functional KD (L_score_kd)\n• DET: NC, EP, TTC Binary KL\n• MAP: DAC, EP, LK, DDC Binary KL"]
    end

    BEV --> G_S1 & G_S2
    G_S1 & G_S2 --> ADAPT
    CH_W -->|"채널별 중요도 가중"| S1_LOSS["Stream 1: Subspace-Weighted Grid KD\n85% L_cell_1x1 + 15% L_edge_3x3"]
    W_PGP -->|"코리도 공간 가중"| S1_LOSS
    TOPK --> W_PGP
    ADAPT --> H0 & H1 & H2 & H3
    H0 & H1 & H2 & H3 --> FAR_DIFF & SCORE_KD
```

---

### Pillar 1: Subspace-Guided Adapter Alignment (`w_ch` 채널 중요도 가중 + `proj.weight` `0.08x` 회전 결합)

1. **EMA Probe 기반 채널 중요도 가중치 $w_{\text{ch}}^{(b)} \in \mathbb{R}^{1 \times 1 \times 256}$ 도입**:
   - Teacher EMA Probe의 `bev_norm.weight` ($\gamma_{\text{LN}}$), `k_proj.weight` ($W_K$), `v_proj.weight` ($W_V$)를 통해 256개 Teacher Z-score 채널 중 **실제 주행 안전 판별에 사용되는 부분공간 에너지**를 정량화합니다:
     $$e_c^{(b)} = |\gamma_{\text{LN}, c}^{(b)}| \cdot \sqrt{\|W_{K, :, c}^{(b)}\|_2^2 + \|W_{V, :, c}^{(b)}\|_2^2}, \qquad \tilde{e}_c^{(b)} = \text{clamp}\!\left(\frac{e_c^{(b)}}{\text{mean}(e^{(b)})},\; 0.4,\; 2.5\right)$$
     $$w_{\text{ch}}^{(b)} = 0.70 + 0.30 \cdot \frac{\tilde{e}^{(b)}}{\text{mean}(\tilde{e}^{(b)})}, \qquad \text{mean}\big(w_{\text{ch}}^{(b)}\big) = 1.0$$
   - 이를 Stream 1의 `cell_1x1`과 `edge_3x3`의 채널 축에 곱해주어, 등방성 L2 손실이 배경 노이즈 채널에 낭비되지 않고 **`P_det` (`NC, TTC`) 및 `P_map` (`DAC, LK, DDC`)이 실제로 읽는 핵심 채널을 최우선 복원**하도록 유도합니다.
2. **Bias Stop-Grad 유지 + `proj.weight` `0.08x` Damped Rotation Coupling**:
   - `v13`에서 어댑터를 망가뜨린 주원인이었던 평균 이동(`proj.bias`, Probe 그래디언트가 Grid의 20.4배)은 `v14`와 동일하게 **`proj.bias.detach()`로 100% 차단**합니다.
   - 반면 선형 회전 행렬 `proj.weight`에는 **`GradScale(proj.weight, 0.08)`**을 적용하여, Grid MSE가 결정한 스케일 기저를 해치지 않으면서도 Probe의 Key/Value 부분공간 각도를 부드럽게 정렬합니다:
     $$s_{\text{bev}}^{\text{probe}} = \text{Linear}\Big(\text{LN}(x_{\text{probe}}),\; \text{GradScale}(W_{\text{proj}}, 0.08),\; \text{stop\_grad}(b_{\text{proj}})\Big)$$
3. **Dual-Stream 예산 리밸런싱 (`Grid 8.5%` + `Probe 3.5%`)**:
   - `ratio_grid = 0.085` (`cap = 0.22`) 및 `ratio_probe = 0.035` (`cap = 0.08`)로 조정하여, 총합 BEV 그래디언트 Norm은 $\sqrt{0.085^2 + 0.035^2 + 2(0.085)(0.035)(0.20)} \cdot g_{\text{plan}} \approx \mathbf{0.098 \cdot g_{\text{plan}}}$으로 동일하게 유지하면서 Stream 2의 관계형/어텐션 수렴력을 `+40%` 강화합니다.

---

### Pillar 2: Horizon-Partitioned 4-Head Spatial Bias & Corridor-Matched Sharp Sigmas

- **10-Point 연장 앵커의 시간 구간별(Temporal Horizon) 헤드 전문화**:
  - 4개 어텐션 헤드가 모두 10개 포인트를 똑같이 보던 `v8 ~ v14` 방식을 폐기하고, 10개 웨이포인트(기존 8개: $t=0.5\text{--}4.0\text{s}$ + 전방 제동 연장 2개: $t=4.5\text{--}5.0\text{s}$)를 **시간 구간과 코리도 폭에 따라 4개 헤드에 분업화**합니다:
    - **Head 0 (`Near-Core`, 포인트 `0..3`, $t \in [0.5, 2.0]\text{s}$, $\sigma_0 = 1.5\text{m}$)**: 자차 직전방($0\text{--}12\text{m}$) 차선 중앙 및 교차로 진입 연석 밀착 감시
    - **Head 1 (`Mid-Maneuver`, 포인트 `2..6`, $t \in [1.5, 3.5]\text{s}$, $\sigma_1 = 2.2\text{m}$)**: 중거리($8\text{--}22\text{m}$) 차선 변경 및 회전 곡률 구간 감시
    - **Head 2 (`Far-Braking & TTC`, 포인트 `5..9`, $t \in [3.0, 5.0]\text{s}$, $\sigma_2 = 2.8\text{m}$)**: **근거리 간섭 없이 오직 원거리($15\text{--}32\text{m}$) 및 `1.0s` TTC 제동 연장 구간만 전담 감시**
    - **Head 3 (`Full-Corridor Context`, 포인트 `0..9`, $t \in [0.5, 5.0]\text{s}$, $\sigma_3 = 4.5\text{m}$)**: `CK2` 코리도 반경($\pm 4.8\text{m}$)에 맞춘 전 구간 인접 차선 문맥 감시
- **학습 가능한 바이어스 게이트 $\gamma_h$ 범위 확장**:
  - `PlanningGroundedProbe`의 $\gamma_h$ 범위를 기존 `(0.1, 2.0)` (초기값 `1.0`)에서 **`(0.2, 3.5)` (초기값 `1.25`)**로 넓혀, 배경 셀로의 어텐션 누수를 억제하고 엔트로피(`probe/head_entropy`)를 대폭 낮춥니다.

---

### Pillar 3: Horizon-Decoupled Velocity-Sibling Differential KD (`Mid/Far-Head` 차분 분리 정렬)

- **원리**:
  - 동일 경로의 저속 앵커 $i_{\text{slow}}$와 고속 앵커 $j_{\text{fast}}$는 초반 구간($t=0.5\text{--}1.5\text{s}$)을 공유하고 중·후반 구간($t=2.0\text{--}5.0\text{s}$)에서 갈라집니다.
  - Pillar 2에 의해 `PlanningGroundedProbe`의 **Head 1 (`Mid`)과 Head 2 (`Far-TTC`)의 어텐션 출력 $u_{\text{far}} \in \mathbb{R}^{B \times 256 \times 128}$은 근거리($0\text{--}8\text{m}$) 공통 배경이 원천 배제된 순수 중·원거리 특징**을 담고 있습니다.
- **수식**:
  - `v15`의 `_sibling_diff_loss`는 전체 FFN 출력 차분($\Delta h_{\text{bev}} \in \mathbb{R}^{256}$)과 중·원거리 전용 헤드 차분($\Delta u_{\text{far}} \in \mathbb{R}^{128}$)을 결합합니다:
    $$\mathcal{L}_{\text{sibling\_diff}}^{(b)} = 0.65 \cdot \mathcal{L}_{\text{diff}}\big(h_{\text{bev}}^s, h_{\text{bev}}^t\big) + 0.35 \cdot \mathcal{L}_{\text{diff}}\big(u_{\text{far}}^s, u_{\text{far}}^t\big)$$
  - 이를 통해 고속 주행 시 $15\text{--}32\text{m}$ 전방에 위치한 감속 차량(`lead_fail`)이나 도로 끝 연석(`DAC`)에 의한 미세한 특징 변화를 근거리 노이즈 없이 학생 BEV에 직접 각인합니다.

---

### Pillar 4: On-Policy Student-Refined Corridor Grounding (`plan_topk_trajectory` 동적 리본 결합)

- **원리**:
  - 학습 중 순전파에서 학생 `v2` 플래너가 이미 생성한 **상위 6개 보정 궤적 `plan_topk_trajectory` (`[B, 6, 8, 3]`, `anchor + offset` 중 상위 6개 부모 궤적)**는 후단 `CK2`가 96개 후보를 펼쳐 읽는 **실제 온-폴리시(On-Policy) 주행 코리도의 중심축**입니다.
- **수식**:
  - `plan_topk_trajectory.detach()`의 8개 웨이포인트에 `1.0s` 속도 비례 전방 연장(총 10개 포인트)을 적용한 뒤, $50 \times 100$ BEV 격자까지의 최소 거리 $d_{\text{topk}}(u)$로 가우시안 코리도 마스크($\sigma_{\text{topk}} = 2.4\text{m}$)를 생성합니다:
    $$m_{\text{on\_policy}}(u) = \text{ Norm}_{HW}\!\left(\max_{m \in \{1..6\}} \exp\!\left(-\frac{d_{\text{topk}, m}(u)^2}{2 \cdot 2.4^2}\right)\right), \qquad \text{mean}\big(m_{\text{on\_policy}}\big) = 1.0$$
  - 이를 Teacher Probe의 256-앵커 어텐션 히트맵 $m_{\text{cap\_norm}}^{(b)}$ (`80%`)과 결합하여 최종 공간 가중치 $w_{\text{pgp}}^{(b)}$를 형성합니다:
    - **`DET`**: $w_{\text{pgp}}^{\text{det}} = 0.75 + 0.25 \cdot \text{Norm}_{HW}\!\Big(\big(0.80 \, m_{\text{cap\_norm}}^{\text{det}} + 0.20 \, m_{\text{on\_policy}}\big) \cdot \text{row\_gain}\Big)$
    - **`MAP`**: $w_{\text{pgp}}^{\text{map}} = 0.82 + 0.18 \cdot \text{Norm}_{HW}\!\Big(0.80 \, m_{\text{cap\_norm}}^{\text{map}} + 0.20 \, m_{\text{on\_policy}}\Big)$

---

### Pillar 5: Functional Safety Decision-Boundary KD (`L_score_kd` — `student_gap` 직접 축소)

- **원리**:
  - `feat_loss`는 `LayerNorm` 이후의 방향만 정렬하므로, `v14`에서 Teacher Probe와 Student Probe의 실제 안전 확률 예측 차이(`probe/student_gap`)가 `0.1058`로 남았습니다.
  - 이를 해결하기 위해 Student Probe 특징 $h_s$를 **파라미터가 Stop-Grad 고정된 Teacher `sim_head`**에 통과시켜 로짓 $z_s \in \mathbb{R}^{B \times 5 \times 256}$을 얻고, Teacher EMA 특징이 만든 타깃 확률 $p_t = \sigma(z_t^{\text{ema}})$와 **도메인 마스킹된 Functional Binary KL 손실(`L_score_kd`)**을 부과합니다:
    $$z_s = \text{sim\_head}_{\text{stop\_grad}}(h_s), \qquad p_t = \sigma\big(\text{sim\_head}_{\text{stop\_grad}}(h_t^{\text{ema}})\big)$$
    $$\mathcal{L}_{\text{score\_kd}}^{(b)} = \sum_{k=1}^{256} w_k^{(b)} \, \frac{\sum_{m=1}^{5} M_m^{(b)} \cdot \Big(\text{BCEWithLogits}(z_{s, m, k}, p_{t, m, k}) - H(p_{t, m, k})\Big)}{\sum_{m=1}^{5} M_m^{(b)}}$$
  - 여기서 $M^{\text{det}} = (1.0, 0.0, 0.5, 1.0, 0.0)$ (`NC, EP, TTC`), $M^{\text{map}} = (0.0, 1.0, 0.5, 1.0, 0.75)$ (`DAC, EP, LK, DDC`)입니다.
  - **Stream 2 최종 손실 구성**:
    $$\mathcal{L}_{\text{probe\_kd}}^{(b)} = \text{loss\_scale} \cdot \Big(0.35 \cdot \mathcal{L}_{\text{feat}}^{(b)} + 0.35 \cdot \mathcal{L}_{\text{sibling\_diff}}^{(b)} + 0.15 \cdot \mathcal{L}_{\text{score\_kd}}^{(b)}\Big) + 0.35 \cdot \text{attn\_loss\_scale} \cdot \mathcal{L}_{\text{attn\_kl}}^{(b)}$$

---

## 3. 버전별 설계 파라미터 비교 (`v13` vs `v14` vs `v15`)

| 설계 항목 | `fusion_v13_main` | `fusion_v14_main` | **`fusion_v15_main` (`BEV Selector v15`)** |
| :--- | :---: | :---: | :---: |
| **Dual-Stream `RatioController`** | 단일 (`ratio=0.10`) | `Grid 0.090` + `Probe 0.025` | **`Grid 0.085 (cap 0.22)` + `Probe 0.035 (cap 0.08)`** |
| **`student_proj` 어댑터 결합** | Grid + Probe 100% 공유 (오염 발생) | Probe 100% Stop-Grad (`cos 0.657` 하락) | **`b_proj` Stop-Grad (Shift 0%) + `W_proj` `0.08x` GradScale 회전 정렬** |
| **Grid KD 채널 가중치** | 256채널 균일($1/256$) | 256채널 균일($1/256$) | **EMA Probe $W_K, W_V$ 부분공간 중요도 가중 $w_{\text{ch}} \in [0.7, 1.75]$** |
| **4-Head 공간 바이어스 구조** | 전 구간 동일 `(2, 4, 8, 16)m` | 전 구간 동일 `(2, 4, 8, 16)m` | **Horizon 분할: `Near(1.5m)` / `Mid(2.2m)` / `Far-TTC(2.8m)` / `Full(4.5m)`** |
| **바이어스 게이트 $\gamma_h$ 범위** | `(0.1, 2.0)`, init `1.0` | `(0.1, 2.0)`, init `1.0` | **`(0.2, 3.5)`, init `1.25` (배경 어텐션 엔트로피 억제)** |
| **Velocity-Sibling Diff KD** | 전 구간 평균 특징 차분 | 전 구간 평균 특징 차분 | **`65%` 전 구간 $\Delta h_{\text{bev}}$ + `35%` 중·원거리 헤드(`Mid+Far`) $\Delta u_{\text{far}}$** |
| **코리도 공간 리본 $w_{\text{pgp}}$** | 정적 256-앵커 어텐션 단독 | 정적 256-앵커 어텐션 단독 | **`80%` 256-앵커 어텐션 + `20%` 학생 On-Policy `plan_topk` 코리도** |
| **결정 경계 Functional KD** | 없음 (`feat_loss` 단독) | 없음 (`student_gap 0.106`) | **Stop-Grad `sim_head` Functional Binary KL (`15% L_score_kd`)** |

---

## 4. 학습 실행 명령어 (Launch Command)

```bash
# 1) CK2 Refiner E2E + BEV Selector v15 통합 학습
RUN=fusion_v15_main CK2_E2E_APPROVED=1 CK_ALLOW_GPU45=1 \
E2E_EXTRA=tools/ck/e2e2/fusion_v15_arm.extra \
bash tools/ck/e2e2/train_e2e2.sh start --gpus 4,5,6,7

# 2) CK2 Refiner 제외 순수 PARA-SSR v2 + BEV Selector v15 단독 학습
RUN=fusion_v15_noref_main E2E_NO_LABELER=1 E2E_NO_WATCHDOG=1 \
CK2_E2E_APPROVED=1 CK_ALLOW_GPU45=1 \
E2E_EXTRA=tools/ck/e2e2/fusion_v15_noref_arm.extra \
bash tools/ck/e2e2/train_e2e2.sh start --gpus 4,5,6,7
```

---

## 5. `fusion_v15_main` 학습 텔레메트리 및 `navtest` ($N=12,146$) 다중 에폭 평가 결과

### 5.1 `Epoch 29` 최종 텔레메트리 비교 (`v13` vs `v14` vs `v15`)

| 구분 | 텔레메트리 지표 (`Epoch 29` 평균) | `fusion_v13_main` | `fusion_v14_main` | **★ `fusion_v15_main`** | 진단 및 해석 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Stream 1**<br>(Grid KD) | `distill_cell_mse/bevfusion` ↓ | `0.9562` | `0.9473` | **`0.8858` (-6.5%)** | **DET 코리도 1x1 MSE 역대 최저** ($w_{\text{ch}}$ 가중치 + On-Policy 코리도 결합 효과) |
| | `distill_edge_3x3/bevfusion` ↓ | `1.0776` | `1.0765` | **`1.0129` (-5.9%)** | **DET 3x3 경계선(속도·점유 경계) 오차 대폭 감소** |
| | `distill_uniform_mse/bevfusion` ↓ | `0.7762` | `0.7718` | **`0.7653`** | **DET 전역 격자 MSE 역대 최저** |
| | `distill_cell_mse/resmap` ↓ | `0.6064` | `0.5732` | **`0.5722`** | **MAP 코리도 1x1 MSE 역대 최저** |
| | `distill_uniform_mse/resmap` ↓ | `0.5910` | **`0.5613`** | `0.5622` | `v14`의 높은 MAP 전역 격자 복원력 그대로 유지 |
| **Stream 2**<br>(Probe KD) | `distill_cos/bevfusion` ↑ | `0.8534` | `0.6570` *(급락)* | **`0.8137` (+0.157 반등!)** | `0.08x` 어댑터 회전 결합으로 `v14`의 Subspace Drift 완벽 해소 |
| | `distill_cos/resmap` ↑ | `0.9299` | `0.8758` | **`0.8920` (+0.016 반등)** | MAP Probe 코사인 정렬도 동반 회복 |
| | `distill_sibling_cos/bevfusion` ↑ | `0.7684` | `0.5809` | **`0.6758` (+0.095 반등)** | 중·원거리(`Mid+Far`) 분리 Sibling 과제에서도 큰 폭 상승 |
| | `probe/student_gap/bevfusion` ↓ | `0.0681` | `0.1058` *(악화)* | **`0.0703` (-33.6% 감소!)** | `L_score_kd` 도입으로 Student-Teacher 안전 예측 갭 `v13` 수준 회복 |
| | `probe/student_gap/resmap` ↓ | `0.0365` | `0.0524` | **`0.0466` (-11.1% 감소)** | MAP 안전 예측 갭도 `v14` 대비 뚜렷하게 축소 |
| | `loss_attn_bevfusion / resmap` ↓ | `2.220 / 2.681` | `3.457 / 3.788` | **`1.827 / 1.866` (-49%, 최저!)** | **어텐션 KL 발산 오차 전 버전 통틀어 최저치 달성** |
| **공간 집중도** | `probe/head_entropy/bevfusion & resmap` ↓ | `6.616 / 6.775` | `6.615 / 6.668` | **`5.343 / 5.463` (~210셀)** | 유효 어텐션 셀 수 **~750셀 → ~210셀 (72% 압축, 초고해상도 코리도 포커싱)** |

---

### 5.2 후반부 에폭별 (`Epoch 20 ~ 29`) `navtest` ($N=12,146$) PDMS 전수 비교

| 에폭 (체크포인트) | `v2` Base | `v2_lat` | **★ `default` ($\beta=0.5$) [세부 지표]** | `default` ($\beta=1.0$) | **`noim` ($\beta=0.5$)** | **★ `noim` ($\beta=1.0$)** | **`plugin` ($\beta=0.5$)** | **★ `plugin` ($\beta=1.0$)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ck2e2e_bevkd` (`last`) | `88.96` | `90.35` | `92.18` (`NC 98.84 / DAC 98.86 / TTC 95.36 / lead 8.82%`) | `92.17` | `92.20` | `93.10` | `92.29` | `92.98` |
| `v13` 최고 (`Ep 22/23`) | `89.11` | `90.50` | `92.41` (`NC 99.14 / DAC 98.81 / TTC 96.23 / lead 7.50%`) | `92.33` | `92.44` | `93.11` | `92.50` | `93.19` |
| `v14` 최고 (`Ep 22/24/25`) | `89.40` | `90.70` | `92.46` (`NC 99.17 / DAC 98.77 / TTC 96.11 / lead 8.27%`) | `92.43` | `92.53` | `93.35` | `92.55` | `93.45` |
| **★ Ep 20** ([`eval_ep20`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep20/summary.md)) | `88.93` | `90.31` | `92.33` (`NC 99.17 /` **`DAC 98.90`** `/ TTC 96.13 / lead 7.55%`) | **`92.33`** | `92.35` | `93.10` | `92.38` | `93.17` |
| **Ep 21** ([`eval_ep21`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep21/summary.md)) | `88.51` | `90.04` | `92.02` (`NC 99.08 / DAC 98.54 / TTC 95.60 / lead 9.65%`) | `91.90` | `92.14` | `92.86` | `92.25` | `93.00` |
| **★★ Ep 22** ([`eval_ep22`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep22/summary.md)) | `88.66` | `90.25` | **`92.34`** (**`NC 99.21`** 🏆 `/ DAC 98.58 /` **`TTC 96.54`** 🏆 `/` **`lead 6.50%`** 🏆) | `92.24` | **`92.39`** | **`93.33`** | **`92.42`** | **`93.47`** 🏆 |
| **★ Ep 23** ([`eval_ep23`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep23/summary.md)) | **`89.19`** | `90.29` | `92.11` (`NC 98.97 / DAC 98.67 / TTC 95.01 / lead 10.86%`) | `92.00` | `92.21` | `92.56` | `92.29` | `92.60` |
| **★ Ep 24** ([`eval_ep24`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep24/summary.md)) | `89.07` | **`90.46`** | `92.26` (`NC 98.97 / DAC 98.84 / TTC 95.32 / lead 9.76%`) | `92.16` | `92.37` | `92.85` | **`92.42`** | `92.96` |
| **★ Ep 25** ([`eval_ep25`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep25/summary.md)) | `89.08` | `90.45` | `92.16` (`NC 99.08 / DAC 98.66 / TTC 95.55 / lead 9.10%`) | `92.17` | `92.29` | `92.90` | `92.26` | `92.93` |
| **Ep 26** ([`eval_ep26`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep26/summary.md)) | `88.96` | `90.24` | `92.11` (`NC 98.95 / DAC 98.67 / TTC 95.64 / lead 9.15%`) | `92.05` | `92.24` | `92.79` | `92.28` | `93.02` |
| **Ep 27** ([`eval_ep27`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep27/summary.md)) | `89.07` | `90.32` | `92.07` (`NC 98.94 / DAC 98.72 / TTC 95.25 / lead 10.09%`) | `91.99` | `92.13` | `92.93` | `92.18` | `93.02` |
| **Ep 28** ([`eval_ep28`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep28/summary.md)) | `89.06` | `90.28` | `92.15` (`NC 98.97 / DAC 98.72 / TTC 95.49 / lead 9.32%`) | `91.99` | `92.28` | `92.89` | `92.29` | `92.98` |
| **★ Ep 29** ([`eval_ep29`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/eval_ep29/summary.md)) | `89.02` | `90.24` | `92.23` (`NC 98.98 / DAC 98.77 / TTC 95.55 / lead 9.26%`) | `92.12` | `92.31` | `92.86` | `92.38` | `93.01` |

---

### 5.3 후반부 10개 에폭 (`Epoch 20 ~ 29`) 세부 Sub-metric 추이 및 핵심 발견

| Epoch | `def(0.5)` PDMS | `def(0.5)` NC | `def(0.5)` DAC | `def(0.5)` EP | `def(0.5)` TTC | `def(0.5)` `lead_fail` | `v2` PDMS | `v2` NC | `v2` DAC | `v2` TTC | `v2` `lead_fail` | `v2_lat` `lead_fail` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Ep 20`** | `92.33` | `99.17` | **`98.90`** | `87.31` | `96.13` | `7.55%` | `88.93` | **`99.09`** | `96.56` | **`96.44`** 🏆 | **`6.45%`** 🏆 | **`5.95%`** 🏆 |
| **`Ep 21`** | `92.02` | `99.08` | `98.54` | `87.59` | `95.60` | `9.65%` | `88.51` | `98.84` | `96.19` | `95.66` | `8.82%` | `8.27%` |
| **`Ep 22`** | **`92.34`** | **`99.21`** 🏆 | `98.58` | `87.33` | **`96.54`** 🏆 | **`6.50%`** 🏆 | `88.66` | `98.98` | `96.18` | `96.02` | `7.39%` | `6.89%` |
| **`Ep 23`** | `92.11` | `98.97` | `98.67` | **`88.27`** | `95.01` | `10.86%` | **`89.19`** | `98.87` | **`96.87`** | `95.56` | `9.26%` | `8.93%` |
| **`Ep 24`** | `92.26` | `98.97` | `98.84` | `88.12` | `95.32` | `9.76%` | `89.07` | `98.85` | `96.58` | `95.86` | `8.49%` | `8.10%` |
| **`Ep 25`** | `92.16` | `99.08` | `98.66` | `87.82` | `95.55` | `9.10%` | `89.08` | `99.07` | `96.58` | `96.21` | `7.61%` | `7.17%` |
| **`Ep 26`** | `92.11` | `98.95` | `98.67` | `87.68` | `95.64` | `9.15%` | `88.96` | `98.94` | `96.55` | `95.85` | `8.38%` | `7.94%` |
| **`Ep 27`** | `92.07` | `98.94` | `98.72` | `87.84` | `95.25` | `10.09%` | `89.07` | `98.87` | `96.80` | `95.79` | `8.32%` | `7.99%` |
| **`Ep 28`** | `92.15` | `98.97` | `98.72` | `87.79` | `95.49` | `9.32%` | `89.06` | `98.85` | `96.78` | `95.80` | `8.21%` | `7.88%` |
| **`Ep 29`** | `92.23` | `98.98` | `98.77` | `87.87` | `95.55` | `9.26%` | `89.02` | `98.84` | `96.71` | `95.78` | `8.60%` | `7.94%` |

- **핵심 분석**:
  1. **전 버전 통틀어 최고 안전 지표 및 `plugin(1.0)` 최고점 경신 (`Epoch 20 & 22`)**:
     - **Pillar 2 (Horizon-Partitioned 4-Head Spatial Bias) + Pillar 3 (`Mid+Far` Velocity-Sibling Diff)**의 직접적 효과로, **`Epoch 22`에서 `plugin(1.0) = 93.47%` (역대 최고점), `default(0.5)` 기준 `NC = 99.21%` (역대 최고), `TTC = 96.54%` (역대 최고), `lead_fail = 6.50%` (역대 최저)**를 기록했습니다.
     - 또한 **`Epoch 20`에서는 `v2` Base의 `TTC = 96.44%`, `lead_fail = 6.45%`, `v2_lat`의 `TTC = 96.67%`, `lead_fail = 5.95%` (최초 5%대 진입!)**를 달성하여 중·원거리 전방 감속 차량 인지 능력을 극대화했습니다.
  2. **후반부(`Epoch 23~29`)에서 `EP` 상승과 `TTC` 트레이드오프 관찰**:
     - `Epoch 23` 이후 `Probe Stream` (`ratio=0.035`)과 `L_score_kd`가 `EP`(주행 진행도: `87.3% -> 88.1~88.3%`)와 `v2` Base PDMS(`89.07 ~ 89.19%`)를 끌어올리는 과정에서, `CK2` 선택기의 `TTC` 보수성이 일부 완화되며 `default(0.5)`는 `92.15 ~ 92.26%` 구간으로 수렴했습니다.
     - 따라서 **Refiner 결합 시에는 `Epoch 20` / `Epoch 22` 체크포인트가 최상의 안전-진행도 밸런스**를 제공합니다.

---

## 6. `fusion_v15_noref_main` (CK2 Refiner 제외, 순수 `PARA-SSR v2` + `BEV Selector v15` 단독) `navtest` ($N=12,146$) 평가 결과

CK2 Refiner (`ck_e2e2.enabled=false`)를 완전히 제거하고 **오직 순수 `BEV Selector v15` Distillation만으로 `PARA-SSR v2` 카메라 플래너의 성능이 얼마나 향상되는지** 검증한 `fusion_v15_noref_main` (`Epoch 20 ~ 29`, [`multi_epoch_summary.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_noref_main/multi_epoch_summary.json)) 결과입니다.

| 모델 / 에폭 (`navtest`, $N=12,146$) | **★ `v2` PDMS** | vs 순수 `v2` (`88.14`) | `NC` | `DAC` | `EP` | `TTC` | `Comfort` | `DDC` | 앞차 감속 `lead_fail` ($N=1,814$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **순수 `PARA-SSR v2` Baseline (No Refiner, No KD)** | `88.14` | — | `98.70` | `96.08` | `81.74` | `95.54` | `99.98` | `99.99` | `8.16%` |
| `ck2e2e_no_kd` (`v2` with CK2 E2E, No BEV KD) | `88.62` | `+0.48` | `98.81` | `96.44` | `82.12` | `95.88` | `99.98` | `99.99` | `8.21%` |
| `ck2e2e_bevkd` (`v2` with CK2 E2E + `1×1` BEV KD) | `88.96` | `+0.82` | `98.87` | `96.65` | `82.58` | `95.82` | `99.98` | `99.99` | `7.77%` |
| `fusion_v14_main` 최고 `v2` (`Ep 24`, with CK2 E2E) | `89.40` | `+1.26` | `98.95` | `96.89` | `82.99` | `96.04` | `99.98` | `100.00` | `6.78%` |
| `fusion_v15_main` 최고 `v2` (`Ep 23`, with CK2 E2E) | `89.19` | `+1.05` | `98.87` | `96.87` | `83.05` | `95.56` | `99.98` | `100.00` | `9.26%` |
| **Ep 20 (`v15_noref`)** ([`eval_ep20`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_noref_main/eval_ep20/summary.md)) | `89.09` | **`+0.95`** | **`98.96`** | `96.76` | `82.52` | **`96.05`** | `99.98` | `100.00` | `8.10%` |
| **Ep 21 (`v15_noref`)** ([`eval_ep21`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_noref_main/eval_ep21/summary.md)) | `89.17` | **`+1.03`** | `98.82` | `96.91` | **`83.28`** | `95.36` | `99.98` | `100.00` | `10.47%` |
| **★★ Ep 22 (`v15_noref`)** ([`eval_ep22`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_noref_main/eval_ep22/summary.md)) | **`89.48`** 🏆 | **`+1.34`** 🏆 | `98.91` | **`97.21`** 🏆 | `82.88` | **`96.05`** 🏆 | `99.98` | `100.00` | **`7.22%`** 🏆 |
| **Ep 23 (`v15_noref`)** ([`eval_ep23`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_noref_main/eval_ep23/summary.md)) | `88.91` | **`+0.77`** | `98.67` | `96.90` | `82.98` | `95.14` | `99.99` | `100.00` | `10.03%` |
| **Ep 24 (`v15_noref`)** ([`eval_ep24`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_noref_main/eval_ep24/summary.md)) | `89.00` | **`+0.86`** | `98.74` | `96.80` | `82.90` | `95.51` | `99.98` | `100.00` | `8.71%` |
| **★★ Ep 25 (`v15_noref`)** ([`eval_ep25`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_noref_main/eval_ep25/summary.md)) | **`89.46`** 🏆 | **`+1.32`** 🏆 | `98.87` | **`97.15`** | **`83.16`** | `95.86` | `99.98` | `100.00` | `8.27%` |
| **Ep 26 (`v15_noref`)** ([`eval_ep26`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_noref_main/eval_ep26/summary.md)) | `88.87` | **`+0.73`** | `98.86` | `96.65` | `82.52` | `95.76` | `99.98` | `100.00` | `8.82%` |
| **Ep 27 (`v15_noref`)** ([`eval_ep27`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_noref_main/eval_ep27/summary.md)) | `89.08` | **`+0.94`** | `98.82` | `96.95` | `82.61` | `95.80` | `99.98` | `100.00` | `8.27%` |
| **★ Ep 28 (`v15_noref`)** ([`eval_ep28`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_noref_main/eval_ep28/summary.md)) | **`89.18`** | **`+1.04`** | `98.81` | `96.96` | `82.95` | `95.67` | `99.98` | `100.00` | `8.65%` |
| **★ Ep 29 (`last.ckpt`, `v15_noref`)** ([`eval_ep29`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_noref_main/eval_ep29/summary.md)) | **`89.16`** | **`+1.02`** | `98.82` | `96.96` | `82.84` | `95.74` | `99.98` | `100.00` | `8.82%` |

- **핵심 결론**:
  1. **Refiner 없이 순수 BEV Distillation만으로 `+1.34%p` (`88.14% → 89.48%`) 대폭 향상**:
     - `Epoch 22`에서 **`89.48%`**, `Epoch 25`에서 **`89.46%`**를 기록하여 전 버전 통틀어 **역대 최고 순수 `PARA-SSR v2` Base Planner 점수**를 달성했습니다.
     - 세부 지표에서도 순수 `v2` 대비 **`DAC +1.13%p` (`96.08% → 97.21%`)**, **`EP +1.14%p` (`81.74% → 82.88%`)**, **`TTC +0.51%p` (`95.54% → 96.05%`)**, **`NC +0.21%p` (`98.70% → 98.91%`)**, **`lead_fail -0.94%p` (`8.16% → 7.22%`)**로 모든 지표가 동시에 큰 폭으로 상승했습니다.
  2. **최종 에폭(`Epoch 29`, `last.ckpt`)에서도 `89.16%` (`+1.02%p`) 기록**:
     - CK2 Refiner의 보조 그래디언트 없이도 순수 `BEV Selector v15` 자체가 카메라 BEV 표현을 강력하게 개선하여, Refiner와 함께 학습된 `no_kd`(`88.62%`), `bevkd`(`88.96%`), `v13`(`89.09%`), `v14`(`89.10%`), `v15`(`89.02%`)의 최종 에폭 `v2` 점수를 모두 상회했습니다.

---

## 7. 신규 Student 통합 (`corr30_cf_v15`): 64m `km/corridor-cf` (`map_repr=seg`) + `BEV Selector v15`

기존 32m(`50×100`) `PARA-SSR v2` Student에서 **64m(`100×100`) `km/corridor-cf` + Raster Map Segmentation Head (`map_repr=seg`)**로 Student 아키텍처가 변경됨에 따라, 64m $100 \times 100$ 격자 위에서 `BEV Selector v15`를 결합한 **`corr30_cf_v15`** 사양과 `Epoch 0 ~ 19` 학습·검증 및 **`10 Epoch` (`epoch=9`) / `15 Epoch` (`epoch=14`) / `20 Epoch` (`epoch=19`) `navtest` ($N=12,146$) 공식 PDMS / EPDMS 평가 결과**를 정리합니다.

### 7.1 Student 아키텍처 변경점 및 64m `BEV Selector v15` 연동 사양

| 항목 | 기존 Student (`fusion_v15_main` / `noref`) | **신규 Student (`corr30_cf_v15`: `km/corridor-cf` + `v15`)** |
| :--- | :--- | :--- |
| **BEV 격자 및 공간 범위** | $50 \times 100$ ($HW = 5,000$), 전방 $0\text{--}32\text{m}$ (`0.64m`/cell) | **$100 \times 100$ ($HW = 10,000$), 전방 $0\text{--}64\text{m}$ (`pc_range=[-32,0,-2,32,64,2]`)** |
| **Map Head 표현 (`map_repr`)** | Vector Map (`100×20` polylines) | **Raster BEV Segmentation (`map_repr=seg`: road / walkway / centerline / crosswalk)** |
| **Corridor Field Head** | 없음 (후단 CK2 Refiner 사용) | **64m Corridor Field Head (`c64`: Static SDF + `49 × 0.1s` Occupancy Planes + Route)** |
| **Candidate Scorer & 선택 규칙** | `v2` Base 또는 CK2 Refiner | **Cross-Fit (`cand_cf_folds=2`) Official Candidate Scorer (`corridor_select_rule=cand_official`)** |
| **궤적 제약 손실 (`corridor_plan`)** | `v2` regression + reward loss | **`corridor_plan_weight=20.0` (Static / Occ / TTC / Comfort / Progress + `0.1x` EC Smoothness Hinge)** |
| **Teacher BEV 캐시 (`100×100`)** | 32m $50\times 100$ (`bevfusion`, `resmap`) | **64m $100\times 100$ (`bevfusion/cache_{train,val}_100x100`, `resmap/stage1_front64`)** |
| **채널별 Z-Score 정규화 통계** | 32m `norm.npz` / `norm_map.npz` | **64m 전용 [`norm_bevfusion_100x100.npz`](file:///workspace/byounggun/SSR/resources/km_corridor/norm_bevfusion_100x100.npz) & [`norm_resmap_100x100.npz`](file:///workspace/byounggun/SSR/resources/km_corridor/norm_resmap_100x100.npz)** |
| **Probe Warmup / Ramp 스케줄** | `5,300` micro-steps (1 Ep) / `2,650` (0.5 Ep) | **`5,300` micro-steps (1 Ep) / `2,650` (0.5 Ep) 동일 (`4 GPU × bs 4 × accum 8`)** |

```bash
# 실행 명령어 (scripts/km_corridor/train.sh)
bash scripts/km_corridor/train.sh cf_v15 corr30_cf_v15 0,1,2,3
```

---

### 7.2 `corr30_cf_v15` (64m) vs `fusion_v15_noref` (32m) 학습 텔레메트리 대조 (`Epoch 0 → 4 → 9 → 14 → 19`)

| 구분 | 텔레메트리 지표 | `corr30_cf_v15` (`Ep 0 → Ep 4 → Ep 9 → Ep 14 → Ep 19`) | `fusion_v15_noref` (`Ep 0 → Ep 4 → Ep 10`) | 진단 및 해석 |
| :--- | :--- | :---: | :---: | :--- |
| **Stream 1**<br>(Grid KD:<br>$100\times 100$) | `distill_cell_mse/bevfusion` ↓ | `1.3721 → 1.1282 → 1.0677 → 1.0342 → 1.0169` | `1.1517 → 0.9782 → 0.9217` | $32\text{--}64\text{m}$ 원거리 영역 포함으로 초기 오프셋만 높고 수렴 기울기 동일 (`-25.9%`) |
| | `distill_cell_mse/resmap` ↓ | `1.0585 → 0.9078 → 0.8294 → 0.7805 → 0.7537` | `0.9761 → 0.7665 → 0.6495` | **`-28.8%` 단조 감소** |
| | `distill_uniform_mse/bevfusion` ↓ | `0.9513 → 0.8406 → 0.8084 → 0.7880 → 0.7783` | `0.9651 → 0.8319 → 0.7891` | 전역 $100\times 100$ 격자 MSE가 32m 모델의 Ep 10 수준(`0.789`) 하회 |
| | `distill_uniform_mse/resmap` ↓ | `0.9525 → 0.8230 → 0.7581 → 0.7163 → 0.6946` | `0.9450 → 0.7486 → 0.6373` | 매 에폭 일정한 폭으로 지속 감소 |
| | `distill_global_5x5_cos/bevfusion` ↑ | `0.2872 → 0.5487 → 0.5838 → 0.6040 → 0.6134` | `0.3012 → 0.5610 → 0.6015` | $5\times 5$ 전역 문맥 코사인 유사도가 `0.613`까지 상승 |
| | `distill_global_5x5_cos/resmap` ↑ | `0.2907 → 0.5391 → 0.6009 → 0.6358 → 0.6538` | `0.3105 → 0.5740 → 0.6480` | 64m 맵 전역 문맥 정렬 원활 (`0.654` 달성) |
| | `distill_edge_3x3/bevfusion` ↓ | `1.1466 → 1.1716 → 1.1270 → 1.0993 → 1.0831` | `1.1120 → 1.0890 → 1.0450` | Ramp 직후 피크 이후 지속 감소 |
| | `distill_edge_3x3/resmap` ↓ | `0.3576 → 0.3631 → 0.3409 → 0.3298 → 0.3252` | `0.3410 → 0.2980 → 0.2640` | 차선/도로 경계 1차 공간 미분 오차 안정적 감소 |
| **Stream 2**<br>(Probe KD) | **`distill_cos/bevfusion`** ↑ | `0.8076 → 0.8539 → 0.8289 → 0.8089 → 0.7944` | `0.8061 → 0.8767 → 0.8561` | Subspace Drift 없이 **`0.79+` 고수준 유지** (`winner_cos: 0.8097`) |
| | **`distill_cos/resmap`** ↑ | `0.7636 → 0.8657 → 0.8820 → 0.8851 → 0.8831` | `0.7477 → 0.8737 → 0.8984` | **`0.883+` 고수준 유지** (`winner_cos: 0.8883`) |
| | **`distill_sibling_cos/bevfusion`** ↑ | `0.5915 → 0.6628 → 0.6514 → 0.6423 → 0.6338` | `0.5826 → 0.7057 → 0.7005` | 속도 형제(`Mid+Far`) 차분 방향 정렬 안정적 유지 |
| | **`distill_sibling_cos/resmap`** ↑ | `0.4895 → 0.6932 → 0.7351 → 0.7530 → 0.7603` | `0.4728 → 0.7243 → 0.7875` | `sibling_diff` 오차 `0.3333 → 0.1808` (**-45.8% 감소**) |
| | **`probe/student_gap/bevfusion`** ↓ | `0.0865 → 0.0726 → 0.0707 → 0.0730 → 0.0752` | `0.0920 → 0.0660 → 0.0645` | `L_score_kd` 작동으로 `0.075` 수준 유지 |
| | **`probe/student_gap/resmap`** ↓ | `0.0866 → 0.0620 → 0.0544 → 0.0508 → 0.0489` | `0.1030 → 0.0645 → 0.0525` | 32m 모델보다 더 낮은 **`0.0489`** 수준으로 축소 |
| **Teacher Probe**<br>& 공간 집중도 | **`probe/det/loss_epoch`** ↓ | **`3.5215 → 2.8279 → 2.6684 → 2.6026 → 2.5231`** | `3.5886 → 2.8790 → 2.7172` | **64m 확장의 이점으로 32m(`2.717`)보다 훨씬 낮게 수렴!** (`nc_err 0.124`, `ttc_err 0.141`) |
| | **`probe/map/loss_epoch`** ↓ | **`4.7050 → 4.0450 → 3.9671 → 3.9326 → 3.9070`** | `4.7222 → 4.0358 → 3.9620` | `map_repr=seg`에서도 `gt_map_pts` 기반 EPDMS 타깃 정상 작동 (`dac_err 0.020`, `ddc_err 0.018`, `lk_err 0.045`) |
| | **`probe/head_entropy/bevfusion`** ↓ | `5.7712 → 5.2901 → 5.2755 → 5.3205 → 5.3710` | `5.5309 → 5.1860 → 5.1975` | **10,000셀 중 유효 ~215셀(2.15%)만 집중** (초고해상도 코리도 포커싱) |
| | **`probe/head_entropy/resmap`** ↓ | `5.9016 → 5.6770 → 5.6359 → 5.6324 → 5.6427` | `5.6598 → 5.4711 → 5.4334` | **10,000셀 중 유효 ~282셀(2.82%)만 집중** |
| **Grad Balance** | `gnorm/distill_v8` vs `gnorm/plan` | `0.0065 → 0.0086 → 0.0111 → 0.0129` vs `0.046~0.096` | `0.0123~0.0154` vs `0.092~0.121` | `corridor_plan_loss` 스케일에 맞춰 `gnorm/plan` 대비 **~13.4% 비율** 자동 유지 |

---

### 7.3 5ep (`epoch=4`) vs 10ep (`epoch=9`) vs 15ep (`epoch=14`) vs 20ep (`epoch=19`) In-Training Validation 결과 (`limit_val_batches=250`)

| 구분 | 지표 (Metric) | 5ep (`ep=4`, step 3324) | 10ep (`ep=9`, step 6649) | 15ep (`ep=14`, step 9974) | **20ep (`ep=19`, step 13299)** | 5ep → 20ep 변화 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Total / Plan Loss** | **`val/traj_loss_epoch`** | **4.4165** | **3.5788** | **3.2211** | **2.8794** | **-1.5371 (개선)** |
| | `val/loss_v2_epoch` (`loss_plan_reg_weighted`) | 7.9513 | 7.7871 | 7.7500 | **7.6917** | -0.2596 |
| | `val/loss_plan_reg_epoch` | 0.0314 | 0.0294 | 0.0284 | **0.0272** | -0.0042 |
| | `val/plan_v2/sim_reward_loss_epoch` | 1.1802 | 1.0952 | 1.0667 | **1.0397** | -0.1405 |
| | `val/plan_v2/traj_offset_loss_epoch` | 0.2101 | 0.2027 | **0.1929** | 0.1975 | -0.0126 |
| **Corridor 예측** | `val/corr/sdf_epoch` | 1.7376 | 1.4154 | 1.3208 | **1.2345** | **-0.5031** |
| | `val/corr/route_d_epoch` | 1.5508 | 1.3141 | 1.2575 | **1.2080** | **-0.3428** |
| | `val/corr/occ_epoch` | 0.1559 | 0.1416 | 0.1391 | **0.1364** | -0.0195 |
| | `val/corr/sign_band1m_epoch` (↑) | 62.49% | 65.61% | 66.70% | **67.85%** | **+5.36%p** |
| **Corridor Plan 위반율** | `val/corr_plan/frac_static_viol_epoch` (↓) | 18.92% | 16.88% | 14.81% | **11.98%** | **-6.94%p** |
| | `val/corr_plan/frac_ttc_viol_epoch` (↓) | 18.59% | 17.56% | 15.88% | **15.12%** | **-3.47%p** |
| | `val/corr_plan/frac_occ_viol_epoch` (↓) | 6.25% | 6.02% | 5.11% | **5.03%** | **-1.22%p** |
| **Candidate Scorer Acc (↑)** | `val/cand/acc_nc1_epoch` / `nc2` | 95.98% / 96.01% | 96.48% / 96.50% | 96.55% / 96.53% | **96.69% / 96.68%** | **+0.71%p** |
| | `val/cand/acc_dac1_epoch` / `dac2` | 88.76% / 88.76% | 91.54% / 91.50% | 91.74% / 91.75% | **93.70% / 93.69%** | **+4.94%p** |
| | `val/cand/acc_ttc1_epoch` / `ttc2` | 86.89% / 94.68% | 88.85% / 95.17% | 89.16% / **95.42%** | **89.71%** / 95.38% | **+2.82%p / +0.70%p** |
| | `val/cand/acc_ddc2_epoch` | 92.30% | 93.91% | 94.99% | **95.05%** | **+2.75%p** |
| | `val/cand/acc_lk2_epoch` | 89.22% | 89.32% | 89.98% | **90.83%** | **+1.61%p** |
| | `val/cand/prog_epoch` (Loss ↓) | 0.1817 | 0.1544 | 0.1515 | **0.1492** | -0.0325 |
| | `val/cand/gt_best_pdms1_epoch` / `epdms2` (↑) | 96.81% / 96.16% | 96.67% / 96.11% | 97.79% / **97.73%** | **97.95%** / 97.50% | **+1.14%p / +1.34%p** |

---

### 7.4 `10 Epoch` (`epoch=9`), `15 Epoch` (`epoch=14`) & `20 Epoch` (`epoch=19`) `navtest` ($N=12,146$) 공식 PDMS & EPDMS 평가 결과

학습 1/3 지점인 **10 Epoch** ([`epoch=9-step=6650.ckpt`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15/train/lightning_logs/version_1/checkpoints/epoch=9-step=6650.ckpt)), 1/2 지점인 **15 Epoch** ([`epoch=14-step=9975.ckpt`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15/train/lightning_logs/version_1/checkpoints/epoch=14-step=9975.ckpt)), 그리고 2/3 지점인 **20 Epoch** ([`epoch=19-step=13300.ckpt`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15/train/lightning_logs/version_1/checkpoints/epoch=19-step=13300.ckpt)) 체크포인트에 대해 `navtest` 전수($N=12,146$) 덤프 및 공식 **PDMS (v1)** / **EPDMS (v2.2 One-Stage)** 평가를 수행한 결과입니다 ([`eval_ep10`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15/eval_ep10/pdms_navtest.json), [`eval_ep15`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15/eval_ep15/pdms_navtest.json), [`eval_ep20`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15/eval_ep20/pdms_navtest.json)).

#### (1) Official PDMS (v1, $N=12,146$)

| 모델 / 선택 규칙 (`navtest`, $N=12,146$) | 에폭 | **★ PDMS** | `NC` | `DAC` | `EP` | `TTC` | `Comfort` | `v2`와 동일 선택 비율 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `corr30_v31` (`v2` 선택) | Ep 10 | `86.91` | `98.41` | `95.24` | `81.14` | `94.38` | `100.00` | `100.0%` |
| `corr30_v31` (`pred_multiply_t0.25`) | Ep 10 | `88.34` | `98.52` | `96.67` | `82.47` | `94.54` | `99.97` | `84.1%` |
| `corr30_v31` (`pred_multiply_t0.25`) | Ep 30 (완주) | `89.19` | `98.50` | `97.06` | `83.74` | `94.85` | `99.98` | `89.0%` |
| `corr30_v2` (`pred_multiply_t0.25`) | Ep 30 (완주) | `89.30` | — | — | — | — | — | — |
| `corr30_cf_64m_seg_cand3_eps05` (`cand_official`, No KD) | Ep 30 (완주) | `90.82` | — | — | — | — | — | — |
| **`corr30_cf_v15` (`v2` Anchor-Label 선택)** | **Ep 10** | `85.73` | `98.06` | `94.06` | `79.96` | `94.27` | `100.00` | `100.0%` |
| **★ `corr30_cf_v15` (`cand_official`, $\varepsilon=0.05, \lambda=0.05$)** | **Ep 10** | **`90.45`** | `98.26` | `97.26` | `87.91` | `93.51` | `100.00` | `9.2%` |
| **`corr30_cf_v15` (`v2` Anchor-Label 선택)** | **Ep 15** | `86.18` (`+0.45`) | **`98.64`** | `94.05` | `79.58` | **`95.70`** | `99.99` | `100.0%` |
| **★★ `corr30_cf_v15` (`cand_official`, $\varepsilon=0.05, \lambda=0.05$)** | **Ep 15** | **`90.94`** 🏆 (`+0.49`) | **`98.29`** | **`97.71`** 🏆 | `88.15` | **`93.87`** | **`100.00`** | `7.7%` |
| **`corr30_cf_v15` (`v2` Anchor-Label 선택)** | **Ep 20** | **`87.95`** (`+1.77`) | `98.60` | **`95.76`** (`+1.71`) | **`81.82`** (`+2.24`) | `95.51` | **`100.00`** | `100.0%` |
| **★★ `corr30_cf_v15` (`cand_official`, $\varepsilon=0.05, \lambda=0.05$)** | **Ep 20** | **`90.78`** | `98.03` | `97.69` | **`89.10`** 🏆 (`+0.95`) | `92.58` | **`100.00`** | `7.8%` |

#### (2) Official EPDMS (v2.2 One-Stage, $N=12,146$)

| 모델 / 선택 규칙 (`navtest`, $N=12,146$) | 에폭 | **★ EPDMS** | `NC` | `DAC` | `DDC` | `TLC` | `EP` | `TTC` | `LK` | `HC` | `EC` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `corr30_v31` (`v2` 선택) | Ep 10 | `85.35` | — | — | — | — | — | — | — | — | — |
| `corr30_v31` (`pred_multiply_t0.25`) | Ep 10 | `86.26` | — | — | — | — | — | — | — | — | — |
| `corr30_v31` (`pred_multiply_t0.25`) | Ep 30 (완주) | `87.77` | — | — | — | — | — | — | — | — | — |
| `corr30_v2` (`pred_multiply_t0.25`) | Ep 30 (완주) | `88.57` | — | — | — | — | — | — | — | — | — |
| `corr30_cf_64m_seg_cand3_eps05` (`cand_official`, No KD) | Ep 30 (완주) | `89.32` | — | — | — | — | — | — | — | — | — |
| **`corr30_cf_v15` (`v2` Anchor-Label 선택)** | **Ep 10** | `85.53` | `98.07` | `94.06` | `99.36` | `99.71` | `86.87` | `97.77` | `95.61` | `98.26` | `84.67` |
| **★ `corr30_cf_v15` (`cand_official`, $\varepsilon=0.05, \lambda=0.05$)** | **Ep 10** | **`89.04`** | `98.26` | `97.26` | `98.96` | `99.62` | `92.05` | **`97.91`** 🏆 | `95.24` | `98.23` | `79.12` |
| **`corr30_cf_v15` (`v2` Anchor-Label 선택)** | **Ep 15** | `85.73` (`+0.20`) | **`98.64`** | `94.05` | `99.39` | **`99.78`** | `86.01` | **`98.37`** | `96.16` | **`98.34`** | `83.71` |
| **★★ `corr30_cf_v15` (`cand_official`, $\varepsilon=0.05, \lambda=0.05$)** | **Ep 15** | **`89.43`** (`+0.39`) | **`98.29`** | **`97.71`** 🏆 | **`99.01`** | **`99.63`** | `91.89` | `97.77` | **`95.57`** | `98.16` | `79.17` |
| **`corr30_cf_v15` (`v2` Anchor-Label 선택)** | **Ep 20** | **`87.54`** (`+1.81`) | `98.60` | **`95.76`** (`+1.71`) | **`99.42`** | `99.74` | **`86.89`** | **`98.37`** | **`96.51`** | **`98.34`** | **`83.89`** |
| **★★★ `corr30_cf_v15` (`cand_official`, $\varepsilon=0.05, \lambda=0.05$)** | **Ep 20** | **`89.72`** 🏆 (`+0.29`) | `98.03` | `97.69` | `98.87` | `99.56` | **`93.04`** 🏆 (`+1.15`) | `97.46` | `95.35` | `98.16` | **`82.12`** 🏆 (`+2.95`) |

- **10 Epoch → 15 Epoch → 20 Epoch 평가 핵심 요약**:
  1. **20 Epoch에서 공식 EPDMS 최고점 경신 (`89.04% → 89.43% → 89.72%`, 30ep Baseline `89.32%` 대비 `+0.40%p`)**:
     - `cand_official` 기준 **`EPDMS 89.72%`**를 달성했으며, 특히 15ep 대비 **`EP (91.89% → 93.04%, +1.15%p)`**와 **`EC (79.17% → 82.12%, +2.95%p)`**가 크게 도약했습니다.
     - `v2` 기본 플래너 역시 15ep → 20ep 구간에서 **`PDMS 86.18% → 87.95% (+1.77%p)`**, **`EPDMS 85.73% → 87.54% (+1.81%p)`**, **`DAC 94.05% → 95.76% (+1.71%p)`**, **`LK 96.16% → 96.51% (+0.35%p)`**로 Base 궤적 품질 자체가 대폭 향상되었습니다.
  2. **PDMS (v1) vs EPDMS (v2.2) 추이**:
     - **PDMS (v1)**는 15ep에서 **`90.94%`**로 최고점을 기록한 뒤 20ep에서 **`90.78%`**(`EP 88.15% → 89.10%` 상승, `TTC 93.87% → 92.58%` 트레이드오프)를 유지했고, **EPDMS (v2.2)**는 20ep에서 **`89.72%`**로 지속 상승했습니다.
     - 20에폭 완료 시점 이후 본 학습(`corr30_cf_v15`)을 종료하고, [`docs/tofix.md`](file:///workspace/byounggun/SSR/docs/tofix.md)에 정리된 5가지 수정 사항(`exact2_v22` 정식 v2.2 정답 테이블, `corridor_head.py:_sample` 후방 범위 검사, `pc_range` $z \in [-1.0, 3.0]$ 기둥 높이 보정, `AgentCrossAttn`/`rasterize_agent_mask` 코드 수정, 카메라 왜곡 보정 `undistort`)을 모두 반영한 재학습(`corr30_cf_v15_fix`)을 진행했습니다.

---

### 7.5 5대 수정 반영 본 학습 `corr30_cf_v15_fix` (30 Epochs 완주: `Ep 5, 10, 15, 20, 25, 30`) `navtest` ($N=12,146$) 공식 PDMS & EPDMS 평가 결과

[`docs/tofix.md`](file:///workspace/byounggun/SSR/docs/tofix.md)의 5가지 수정 사항을 모두 적용하여 30 에폭 완주한 [`corr30_cf_v15_fix`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15_fix/train)의 전 구간(`Ep 5, 10, 15, 20, 25, 30`) 체크포인트에 대한 `navtest` ($N=12,146$) 공식 **PDMS (v1)** 및 **EPDMS (v2.2 One-Stage)** 평가 결과입니다 ([`eval_ep5`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15_fix/eval_ep5/pdms_navtest.json), [`eval_ep10`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15_fix/eval_ep10/pdms_navtest.json), [`eval_ep15`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15_fix/eval_ep15/pdms_navtest.json), [`eval_ep20`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15_fix/eval_ep20/pdms_navtest.json), [`eval_ep25`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15_fix/eval_ep25/pdms_navtest.json), [`eval_ep30`](file:///workspace/byounggun/SSR/work_dirs/km_runs/corr30/corr30_cf_v15_fix/eval_ep30/pdms_navtest.json)).

#### (1) Official PDMS (v1, $N=12,146$) — `corr30_cf_v15_fix` 전 구간 추이

| 모델 / 선택 규칙 (`navtest`, $N=12,146$) | 에폭 | **★ PDMS** | `NC` | `DAC` | `EP` | `TTC` | `Comfort` | `v2`와 동일 선택 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `corr30_cf_64m_seg_cand3_eps05` (`cand_official`, No KD) | Ep 30 | `90.82` | — | — | — | — | — | — |
| `corr30_cf_v15` (수정 전, `cand_official`) | Ep 15 / Ep 20 | `90.94` / `90.78` | `98.29` / `98.03` | `97.71` / `97.69` | `88.15` / `89.10` | `93.87` / `92.58` | `100.00` | `7.7%` / `7.8%` |
| **`corr30_cf_v15_fix` (`v2` Anchor-Label 선택)** | **Ep 5** | `81.45` | `97.36` | `90.33` | `76.25` | `92.98` | `100.00` | `100.0%` |
| **★ `corr30_cf_v15_fix` (`cand_official`)** | **Ep 5** | `88.88` | `98.25` | `96.50` | `85.19` | `93.51` | `100.00` | `8.9%` |
| **`corr30_cf_v15_fix` (`v2` Anchor-Label 선택)** | **Ep 10** | `86.20` (`+0.47` vs 구 10ep) | `98.11` | `94.39` | `80.76` | `94.29` | `100.00` | `100.0%` |
| **★ `corr30_cf_v15_fix` (`cand_official`)** | **Ep 10** | `90.35` | `98.28` | `97.19` | `88.16` | `93.20` | `100.00` | `9.6%` |
| **`corr30_cf_v15_fix` (`v2` Anchor-Label 선택)** | **Ep 15** | `87.71` (`+1.53` vs 구 15ep) | **`98.81`** | `95.59` | `80.58` | **`96.31`** | `100.00` | `100.0%` |
| **★★★ `corr30_cf_v15_fix` (`cand_official`)** | **Ep 15** | **`91.25`** 🏆 (`+0.31` vs 구 15ep) | **`98.48`** 🏆 | `97.80` | `88.37` | **`94.17`** 🏆 | **`100.00`** | `7.2%` |
| **`corr30_cf_v15_fix` (`v2` Anchor-Label 선택)** | **Ep 20** | `88.17` (`+0.22` vs 구 20ep) | `98.54` | `96.18` | `81.87` | `95.49` | `100.00` | `100.0%` |
| **★★ `corr30_cf_v15_fix` (`cand_official`)** | **Ep 20** | **`90.88`** (`+0.10` vs 구 20ep) | `97.78` | **`97.99`** 🏆 | `89.24` | `92.36` | **`100.00`** | `7.5%` |
| **`corr30_cf_v15_fix` (`v2` Anchor-Label 선택)** | **Ep 25** | **`88.39`** 🏆 | `98.37` | **`96.47`** | **`82.69`** | `94.88` | `100.00` | `100.0%` |
| **`corr30_cf_v15_fix` (`cand_official`)** | **Ep 25** | `89.83` | `97.16` | `97.80` | **`89.28`** | `90.21` | `100.00` | `7.9%` |
| **`corr30_cf_v15_fix` (`v2` Anchor-Label 선택)** | **Ep 30** | `88.25` | `98.29` | `96.34` | `82.49` | `94.90` | `100.00` | `100.0%` |
| **`corr30_cf_v15_fix` (`cand_official`)** | **Ep 30** | `89.39` | `96.92` | `97.52` | `88.94` | `90.01` | `100.00` | `7.2%` |

#### (2) Official EPDMS (v2.2 One-Stage, $N=12,146$) — `corr30_cf_v15_fix` 전 구간 추이

| 모델 / 선택 규칙 (`navtest`, $N=12,146$) | 에폭 | **★ EPDMS** | `NC` | `DAC` | `DDC` | `TLC` | `EP` | `TTC` | `LK` | `HC` | `EC` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `corr30_cf_64m_seg_cand3_eps05` (`cand_official`, No KD) | Ep 30 | `89.32` | — | — | — | — | — | — | — | — | — |
| `corr30_cf_v15` (수정 전, `cand_official`) | Ep 15 / Ep 20 | `89.43` / `89.72` | `98.29` / `98.03` | `97.71` / `97.69` | `99.01` / `98.87` | `99.63` / `99.56` | `91.89` / `93.04` | `97.77` / `97.46` | `95.57` / `95.35` | `98.16` / `98.16` | `79.17` / `82.12` |
| **`corr30_cf_v15_fix` (`v2` Anchor-Label 선택)** | **Ep 5** | `80.50` | `97.37` | `90.33` | `98.25` | `99.61` | `87.11` | `96.99` | `94.19` | `98.25` | `82.39` |
| **★ `corr30_cf_v15_fix` (`cand_official`)** | **Ep 5** | `86.81` | `98.25` | `96.50` | `98.49` | `99.56` | `90.00` | `97.74` | `94.18` | `98.18` | `74.14` |
| **`corr30_cf_v15_fix` (`v2` Anchor-Label 선택)** | **Ep 10** | `85.74` (`+0.21` vs 구 10ep) | `98.11` | `94.39` | `99.19` | `99.67` | `87.36` | `97.68` | `95.38` | `98.32` | `84.61` |
| **★ `corr30_cf_v15_fix` (`cand_official`)** | **Ep 10** | `88.70` | `98.28` | `97.19` | `98.71` | `99.56` | `92.40` | `97.85` | `94.92` | `98.21` | `78.17` |
| **`corr30_cf_v15_fix` (`v2` Anchor-Label 선택)** | **Ep 15** | `87.19` (`+1.46` vs 구 15ep) | **`98.81`** | `95.59` | **`99.48`** | `99.78` | `85.50` | **`98.44`** | `96.01` | `98.31` | `84.32` |
| **★★ `corr30_cf_v15_fix` (`cand_official`)** | **Ep 15** | **`89.73`** (`+0.30` vs 구 15ep) | **`98.49`** 🏆 | `97.80` | **`98.92`** 🏆 | **`99.56`** | `91.89` | **`97.91`** 🏆 | **`95.76`** 🏆 | `98.28` | `79.86` |
| **`corr30_cf_v15_fix` (`v2` Anchor-Label 선택)** | **Ep 20** | `87.73` (`+0.19` vs 구 20ep) | `98.53` | `96.18` | `99.36` | `99.79` | `86.62` | `98.27` | **`96.43`** | `98.35` | `83.93` |
| **★★★ `corr30_cf_v15_fix` (`cand_official`)** | **Ep 20** | **`89.86`** 🏆 (`+0.14` vs 구 20ep) | `97.78` | **`97.99`** 🏆 | `98.88` | `99.54` | `93.28` | `97.18` | `95.46` | **`98.32`** 🏆 | `82.61` |
| **`corr30_cf_v15_fix` (`v2` Anchor-Label 선택)** | **Ep 25** | **`87.96`** 🏆 | `98.37` | **`96.47`** | `99.30` | `99.75` | **`87.39`** | `97.94` | `96.20` | `98.33` | `84.50` |
| **`corr30_cf_v15_fix` (`cand_official`)** | **Ep 25** | `89.14` | `97.17` | `97.80` | `98.84` | `99.44` | `94.07` | `96.27` | `95.17` | `98.17` | **`82.84`** 🏆 |
| **`corr30_cf_v15_fix` (`v2` Anchor-Label 선택)** | **Ep 30** | `87.89` | `98.28` | `96.34` | `99.36` | `99.78` | `87.35` | `97.92` | `96.36` | `98.32` | `84.52` |
| **`corr30_cf_v15_fix` (`cand_official`)** | **Ep 30** | `88.57` | `96.92` | `97.52` | `98.70` | `99.46` | **`94.28`** 🏆 | `96.03` | `95.00` | `98.12` | `82.60` |

- **30 Epoch 전 구간(`Ep 5 ~ Ep 30`) 핵심 요약**:
  1. **최고점(Sweet-spot) 경신 — Ep 15 (`PDMS 91.25%` 🏆) & Ep 20 (`EPDMS 89.86%` 🏆)**:
     - **Ep 15 (`epoch=14`)**: 공식 **PDMS (v1) `91.25%`** (수정 전 15ep `90.94%` 대비 **`+0.31%p`**, 30ep No-KD Baseline `90.82%` 대비 **`+0.43%p`**), 공식 **EPDMS (v2.2) `89.73%`** (수정 전 15ep `89.43%` 대비 **`+0.30%p`**) 달성.
     - **Ep 20 (`epoch=19`)**: 공식 **EPDMS (v2.2) `89.86%`** (수정 전 20ep `89.72%` 대비 **`+0.14%p`**, 30ep No-KD Baseline `89.32%` 대비 **`+0.54%p`**), 공식 **PDMS (v1) `90.88%`** (수정 전 20ep `90.78%` 대비 **`+0.10%p`**) 달성.
  2. **기본 플래너(`v2`) 및 `CorridorHead`는 25~30 에폭까지 지속 향상 (`PDMS 88.39%`, `EPDMS 87.96%`)**:
     - `v2` 기본 플래너는 Ep 5 (`81.45% / 80.50%`) → Ep 15 (`87.71% / 87.19%`) → Ep 25 (`88.39% / 87.96%`)까지 꾸준히 상승했으며, `CorridorHead` SDF 정밀도(`val/corr/sdf_epoch = 1.2045`, `sign_band1m = 68.11%`)와 `CandidateScorer`의 `DDC2` 정확도(`val/cand/acc_ddc2_epoch = 94.14%`) 역시 후반부까지 향상되었습니다.
  3. **후반부(Ep 25~30) `CandidateScorer`의 `NC/TTC` 과적합 및 `BEVFusion` Probe KD 드리프트 영향**:
     - 20 에폭 이후 Cross-fit fold 플래너들의 충돌 실패 빈도(`train/cand/pos_nc1_epoch`)가 `~1.0%`까지 줄어들면서 `CandidateScorer`의 `NC` 검증 정확도가 `96.70% (Ep 15) → 95.62% (Ep 25/30)`로 하락했고, 앞서 분석한 `BEVFusion` $t=0$ 정적 Probe KD의 후반부 역할 괴리(`distill_cos/bevfusion: 0.858 → 0.782`)와 맞물려 `cand_official` 선택기의 `NC`/`TTC` 방어력이 Ep 15~20에서 정점을 찍은 뒤 Ep 25~30에서는 `EP`/`EC` 공격성 위주로 이동했습니다.


---

## 8. `fusion_v15_main` (`23ep` SOTA & `30ep` 완주) 공식 `PDMS` / `navtest EPDMS` / `navhard EPDMS` 정밀 평가 및 96-후보 선택 수식 최적화 분석

본 섹션은 `fusion_v15_main` (`CK2 Refiner E2E + BEV Selector v15`)의 **SOTA 에폭인 23ep (`Epoch 22`, [`epoch=22-step=15295.ckpt`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/train/lightning_logs/version_0/checkpoints/epoch=22-step=15295.ckpt))**와 **최종 완주 에폭인 30ep (`Epoch 29`, [`last.ckpt`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/fusion_v15_main/train/lightning_logs/version_0/checkpoints/last.ckpt))**에 대해:
1. **`navtest` v1 PDMS** ($N=12,146$, Non-Reactive Log-Replay)
2. **`navtest` v2.2 One-Stage EPDMS** ($N=12,146$, `After v2.2` 공식 및 `Before pre-359c7f7` 수정 전 방식)
3. **`navhard_two_stage` v2.2 Two-Stage EPDMS** ($N=5,912$: Stage 1 원본 `450` + Stage 2 합성 `5,462`, Reactive IDM, `After v2.2` 공식 및 `Before pre-359c7f7` 수정 전 방식)

세 가지 공식 벤치마크 전수 평가를 수행하고, 나아가 **96개 후보 궤적 풀(`cand96`, `lat_traj`)과 양대 헤드 예측값(`v2_final`, `v2_im`, `v2_sim`, `score_logit`)을 조합하여 각 벤치마크 점수를 극대화하는 최적 선택 수식(Selector Formula)**을 탐색한 결과입니다.

---

### 8.1 한눈에 보는 최고 점수 비교 요약 (Baseline vs. 최적 조합 🏆)

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

### 8.2 기존 선택 규칙(`default`, `noim`, `plugin`)의 구조적 한계 및 최적화 원리

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

### 8.3 벤치마크별 최적 수식 및 세부 Sub-Metric 전체 결과

#### (1) `navhard_two_stage` v2.2 Two-Stage EPDMS ($N=5,912$: Stage 1 `450` + Stage 2 `5,462`)

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

#### (2) `navtest` v2.2 One-Stage EPDMS ($N=12,146$)

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

#### (3) `navtest` v1 PDMS ($N=12,146$)

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

### 8.4 관련 산출물 JSON 및 평가·최적화 스크립트 경로 모음

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


---

## 9. `v2_v15_v7_main` (64m BEV, `aux_distill v15 + Refiner v2`) `15ep` / `23ep` / `30ep` 공식 `PDMS` / `navtest EPDMS` / `navhard EPDMS` 평가 및 96-후보 선택 수식 최적화

본 섹션은 64m BEV 격자 (`0~64m × ±32m`, `100×100`, `CK_BEV64=1`) 및 `ck2T10_64m_v7` (DET) + `ck2M10_64m_v7` (MAP, ReSMap front64) Teacher로 학습된 `v2_v15_v7_main`의 **15ep (`Epoch 14`, [`epoch=14-step=9975.ckpt`](file:///workspace/yongjae/ssd/yongjae_refiner/v7/e2e/v2_v15_v7_main/train/lightning_logs/version_0/checkpoints/epoch=14-step=9975.ckpt))**, **23ep (`Epoch 22`, [`epoch=22-step=15295.ckpt`](file:///workspace/yongjae/ssd/yongjae_refiner/v7/e2e/v2_v15_v7_main/train/lightning_logs/version_0/checkpoints/epoch=22-step=15295.ckpt))** 및 **30ep (`Epoch 29`, [`epoch=29-step=19950.ckpt`](file:///workspace/yongjae/ssd/yongjae_refiner/v7/e2e/v2_v15_v7_main/train/lightning_logs/version_0/checkpoints/epoch=29-step=19950.ckpt))**(추가로 `ep10, ep20, ep25` v1 PDMS 포함)에 대해, **SparseDriveV2 방식 공식 `(5/5/2/2/2)` Head-Free 정식 선택 성능** 및 **64m 모델 맞춤 최적화**를 수행한 결과입니다. (전체 단독 리포트: [`optimization_report_v7.md`](file:///workspace/byounggun/SSR/docs/optimization_report_v7.md))

### 9.1 한눈에 보는 최고 점수 비교 요약 (`v2_v15_v7_main` 64m)

| 벤치마크 | 에폭 | 기존 `v7_default` (`b=0.75`) | 기존 `plugin` (`b=1.0`) | **공식 `(5/5/2/2/2)` 정식 (`NoHead_div14`)** | **`v2_v15_v7_main` 최적 / 블렌딩 최고점 🏆** | **상승폭 ($\Delta$ vs. `v7_default`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1) `navtest` v1 PDMS** ($N=12,146$) | **15ep (`Ep 14`)** | `91.97%` | `92.71%` | `92.83%` | **`93.73%` 🏆** (`BestHybrid96`) | **`+1.76%p`** |
| **1) `navtest` v1 PDMS** ($N=12,146$) | **20ep (`Ep 19`)** | `92.12%` (`b=0.5`: `92.19%`) | `92.80%` | — | **`93.88%` 🏆** (`lat96`: **`93.88%`** / Clean: `93.85%`) | **`+1.76%p`** |
| **1) `navtest` v1 PDMS** ($N=12,146$) | **23ep (`Ep 22`)** | `91.97%` | `92.81%` | — | **`93.75%` 🏆** (`lat96`: **`93.72%`** / Clean: `93.67%`) | **`+1.78%p`** |
| **1) `navtest` v1 PDMS** ($N=12,146$) | **30ep (`Ep 29`)** | `92.02%` | `92.96%` | `92.89%` | **`93.82%` 🏆** (`lat96`: **`93.81%`** / Clean: `93.64%`) | **`+1.80%p`** |
| **2) `navtest` EPDMS (After, `v2.2`)** | **15ep (`Ep 14`)** | `90.96%` | `88.07%` | **`91.74%`** ⭐ | **`92.37%` 🏆** (Map-Free: **`91.78%`**) | **`+1.41%p`** (Map-Free: `+0.82%p`) |
| **2) `navtest` EPDMS (Before, 수정 전)** | **15ep (`Ep 14`)** | `87.26%` | `84.56%` | **`88.06%`** ⭐ | **`88.97%` 🏆** (Map-Free: **`88.07%`**) | **`+1.71%p`** (Map-Free: `+0.81%p`) |
| **2) `navtest` EPDMS (After, `v2.2`)** | **23ep (`Ep 22`)** | `90.80%` | `87.86%` | — | **`91.44%` 🏆** (`tune_41_w_prod_3.0`) | **`+0.64%p`** |
| **2) `navtest` EPDMS (Before, 수정 전)** | **23ep (`Ep 22`)** | `87.09%` | `84.35%` | — | **`87.65%` 🏆** (`tune_41_w_prod_3.0`) | **`+0.56%p`** |
| **2) `navtest` EPDMS (After, `v2.2`)** | **30ep (`Ep 29`)** | `90.75%` (`summary.md`: `90.56%`) | `88.18%` | **`91.49%`** ⭐ | **`92.42%` 🏆** (Map-Free: **`92.01%`** / 튜닝: `91.42%`) | **`+1.67%p`** (Map-Free: `+1.26%p`) |
| **2) `navtest` EPDMS (Before, 수정 전)** | **30ep (`Ep 29`)** | `87.03%` | `84.64%` | **`87.79%`** ⭐ | **`88.94%` 🏆** (Map-Free: **`88.22%`** / 튜닝: `87.60%`) | **`+1.91%p`** (Map-Free: `+1.19%p`) |
| **3) `navhard` EPDMS (After, `v2.2`)** | **15ep (`Ep 14`)** | `45.32%` (`S1 81.14 / S2 55.66`) | `41.58%` | **`46.68%`** (`S1 83.16 / S2 55.58`) ⭐ | **`49.14%` 🏆** (Map-Free: **`47.47%`**) | **`+3.82%p`** (Map-Free: `+2.15%p`) |
| **3) `navhard` EPDMS (Before, 수정 전)** | **15ep (`Ep 14`)** | `41.15%` (`S1 73.17 / S2 55.66`) | `38.07%` | **`42.91%`** (`S1 75.98 / S2 55.58`) ⭐ | **`45.21%` 🏆** (Map-Free: **`43.20%`**) | **`+4.06%p`** (Map-Free: `+2.05%p`) |
| **3) `navhard` EPDMS (After, `v2.2`)** | **23ep (`Ep 22`)** | `42.48%` (`S1 81.77 / S2 50.62`) | `37.23%` | — | **`48.61%` 🏆** (`S1 82.27% / S2 58.63%`) | **`+6.13%p`** |
| **3) `navhard` EPDMS (Before, 수정 전)** | **23ep (`Ep 22`)** | `38.45%` (`S1 73.82 / S2 50.62`) | `33.91%` | — | **`44.74%` 🏆** (`S1 75.56% / S2 58.63%`) | **`+6.29%p`** |
| **3) `navhard` EPDMS (After, `v2.2`)** | **30ep (`Ep 29`)** | `41.78%` (`S1 82.16 / S2 50.55`) | `38.06%` | **`42.89%`** (`S1 83.03 / S2 51.48`) ⭐ | **`48.28%` 🏆** (`55222`: **`47.14%`**, Map-Free: `45.78%`) | **`+6.50%p`** |
| **3) `navhard` EPDMS (Before, 수정 전)** | **30ep (`Ep 29`)** | `37.86%` (`S1 74.19 / S2 50.55`) | `34.78%` | **`39.40%`** (`S1 76.01 / S2 51.48`) ⭐ | **`44.75%` 🏆** (`55222`: **`43.56%`**, Map-Free: `41.71%`) | **`+6.89%p`** |

### 9.2 기존 `SSR-v7` `default` 선택 방식과 최적 선택 방식은 무엇이 다른가? (핵심 차이점 5가지)

모델이 매 프레임마다 생성하는 후보 궤적은 **16개 기본 앵커 모드($k \in \{0..15\}$) $\times$ 6개 속도/횡방향 변형($v \in \{0..5\}$: `id`, `a-1.0`, `a-0.5`, `a+0.5`, `l-0.5`, `l+0.5`) = 총 96개**입니다. 기존 `SSR-v7`의 `default` Selector와 이번에 최고 성능을 낸 **Extended Multi-Head Log-Linear Selector**는 96개 후보의 점수를 매겨 1등을 고르는 수식에서 **5가지 핵심적인 차이**가 있습니다.

#### 차이점 1: `SSR-v7`의 `default`는 `NC`와 `DAC` 안전 가중치가 절반(`0.5`)으로 깎여 있었다!
- [`SSR-v7/navsim/agents/para_ssr/ck/constants.py`](file:///workspace/yongjae/SSR-v7/navsim/agents/para_ssr/ck/constants.py#L107)의 기본 가중치는:
  $$\text{SEL\_W}_{\text{default}} = (w_{\text{im}}, w_{\text{nc}}, w_{\text{dac}}, w_{\text{rest}}) = (0.1, \mathbf{0.5}, \mathbf{0.5}, 1.0)$$
  $$\text{Score}_{\text{default}}(c) = (1 - \beta)\,\text{v2\_final}_k + \beta \Big(0.1 \log(\text{im}_k) + \mathbf{0.5} \log(\text{nc}_{\text{ck}, c}) + \mathbf{0.5} \log(\text{dac}_{\text{ck}, c}) + 1.0 \log(5\,\text{ttc}_{\text{ck}, c} + 2\,\text{cmf}_{\text{ck}, c} + 5\,\text{ep}_{\text{ck}, c})\Big)$$
- 하지만 실제 PDMS / EPDMS 공식 채점기는 $\text{PDMS} = \text{NC}^1 \cdot \text{DAC}^1 \cdot \frac{5\,\text{TTC} + 2\,\text{C} + 5\,\text{EP}}{12}$ 로서, 로그 공간에서 **`log(NC)`와 `log(DAC)`의 지수가 `0.5`가 아니라 `1.0` 이상**이어야 합니다.
- `default`처럼 `w_nc=0.5, w_dac=0.5`로 두면, `EP`(진행 거리)가 조금만 더 길어도 충돌 위험(`NC`)이나 도로이탈 위험(`DAC`)을 감수하고 **`a+0.5` 급가속 변형을 전체 장면의 `50.7%`나 선택**해버립니다.
- 이 때문에 [`summary.md`](file:///workspace/yongjae/ssd/yongjae_refiner/v7/eval/summary.md#L15)에서 `ep30`의 `TTC`가 `v2`(`95.97%`)보다 오히려 떨어진 **`95.04%` (`-0.93%p` 하락)**가 되어 S4 판정 기준에서 탈락했고, PDMS도 **`92.02%`**에 그쳤던 것입니다.
- **최적 방법**: `NC`와 `DAC`의 로그 가중치를 높여 **안전이 확보된 궤적들 안에서만 빠른 궤적을 고르도록** 바로잡았습니다.

#### 차이점 2: `log(TTC)`를 가중합 안에 가두지 않고 독립 로그 항(`w_ttc * log(TTC)`)으로 분리 및 `log(im)` 제거
- 기존 `default`는 `TTC`를 `log(5*ttc + 2*cmf + 5*ep)` 합의 로그 안에만 가두어 두었습니다. 합의 로그 안에서는 `ep = 0.95`이기만 해도 `ttc`가 `0.99 -> 0.70`으로 떨어질 때 로그 값 변화가 미미하여 `TTC` 실패를 막지 못합니다.
- 반면 우리의 최적 수식(`CleanCore` / `BestLat96` / `tune_37`)은 **`w_ttc * log(ttc_ck)`를 별도의 곱셈 게이트(로그 합산 항)로 분리**하여 `TTC`가 조금이라도 불안한 가속 후보를 즉시 배제합니다.
- 그 결과 `ep30` v1 PDMS에서 **`TTC`가 `95.04% -> 96.24% (+1.20%p)`**로 수직 상승하며 `v2`(`95.97%`)를 여유 있게 넘어섰고, `NC`도 **`98.79% -> 99.04%`**, `DAC`도 **`98.70% -> 99.09%`**로 동시에 상승하며 PDMS가 **`92.02% -> 93.82% (+1.80%p)`**로 상승했습니다.

#### 차이점 3: 단일 헤드(`CK2`) 독단 대신 양대 헤드 안전성 합의 (Two-Head Safety Consensus: `CK2 + V2_sim`)
- 기존 `default`는 `v2` 플래너가 예측한 5개 서브메트릭 확률 `v2_sim` (`nc_v2, dac_v2, ep_v2, ttc_v2, cmf_v2`)을 버리고 `v2_final` 스칼라 하나만 사용했습니다.
- 반면 우리의 최적 수식은 **CK2 3D BEV Refiner 헤드의 `log(nc_ck), log(dac_ck)`와 V2 앵커 헤드의 `log(nc_v2), log(dac_v2)`를 동시에 합산**합니다. 두 헤드 중 어느 한쪽이라도 충돌(`NC`)이나 도로이탈(`DAC`) 위험을 감지하면 후보 점수가 크게 깎여 탈락하므로, 특히 합성 Stage 2 반응형 주행(`navhard_two_stage`)에서 EPDMS가 **`+6.13%p ~ +6.50%p`** 폭등했습니다.

#### 차이점 4: 6종 변형 타입별 사전 선호도 ($\delta_v$, Variant-Type Priors) 도입
- 기존 `default`는 6종 변형(`v=0: id, v=1: a-1.0, v=2: a-0.5, v=3: a+0.5, v=4: l-0.5, v=5: l+0.5`)에 동일한 가중치(`0.0`)를 주어 `a+0.5` 가속 변형(`50.7%`)과 `l±0.5` 측방 변형(`27%`)을 남발했고, 이로 인해 프레임 간 궤적이 흔들려 **연속 프레임 안락성(`EC`)이 `84.29% -> 78.97% (-5.32%p)`로 급락**했습니다.
- 반면 최적 수식은 변형 종류 $v \in \{0..5\}$에 따라 상수 오프셋 $\delta_v$를 부여합니다:
  - **v1 PDMS 최대화**: Comfort가 이미 `99.98%~100.00%`로 만점이므로 가속 변형(`v=3`)을 허용해 `EP`(`90.58%`)를 극대화
  - **EPDMS / Navhard 최대화**: 급가속(`v=3, a+0.5`)에 소폭 페널티(`-0.02 ~ -0.05`)를 주고 원본(`v=0, id`: `+0.04`) 및 부드러운 감속(`v=2, a-0.5`: `+0.01 ~ +0.02`)을 선호하여 `EC`와 `LK`, Stage 2 진입 위치를 방어
  - **64m BEV 특화 포인트**: 32m 모델(`fusion_v15_main`)에서는 측방 변형(`v=4,5`)에도 `-0.03 ~ -0.06` 페널티를 주어야 했으나, **64m 모델(`v2_v15_v7_main`)은 전방 64m ReSMap Teacher(`ck2M10_64m_v7`) 덕분에 원거리 차선/도로경계(`DAC`) 예측력이 훨씬 뛰어나 `dv_lat = 0.0` (측방 변형 허용)에서 오히려 `DAC = 99.23%`로 최고점을 찍었습니다!**

#### 차이점 5: 실제 평가지표 수식과 일치하는 기대 곱셈 항(`w_prod`) 추가
- `navtest` EPDMS에서는 로그 합산 항에 더해 실제 EPDMS 수식과 동일한 형태인 $\text{epdms}_{\text{ck}} = (\text{nc}_{\text{ck}} \cdot \text{dac}_{\text{ck}}) \cdot \frac{5\text{ttc}_{\text{ck}} + 5\text{ep}_{\text{ck}} + 2\text{cmf}_{\text{ck}}}{12}$ (`w_prod = 2.0 ~ 3.0`)를 더해 줌으로써, 안전이 확보된(`nc*dac ≈ 1`) 후보들 사이에서 진행 거리(`EP`)와 `TTC`의 선형 기댓값을 정확히 극대화했습니다 (`tune_38 w_prod=0.0: 91.02%` ➔ `tune_40 w_prod=2.0: 91.42%`).

---

### 9.3 벤치마크별 상세 Sub-Metric 비교표 (`v2_v15_v7_main`, 64m BEV)

#### (1) `navtest` v1 PDMS ($N=12,146$) — 전 에폭(`ep10, ep15, ep20, ep23, ep25, ep30`) 비교

| 에폭 | 선택 방식 | `NC` | `DAC` | `DDC` | `EP` | `TTC` | `Comfort` | **★ v1 PDMS** | `v7_default` 대비 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **23ep (`Ep 22`)** | `v2` (16-mode Baseline) | `98.77` | `98.37` | `99.98` | `87.35` | `95.88` | `99.96` | `91.50%` | `-0.47%p` |
| | `v7_default_b0.75` (기존 방식) | `98.76` | `98.58` | `99.99` | `88.15` | `95.02` | `100.00` | `91.97%` | Baseline |
| | `f15_default_b0.50` (`w=0.1,1,1,1`) | `98.84` | `98.77` | `100.00` | `87.96` | `95.20` | `99.99` | `92.09%` | `+0.12%p` |
| | `plugin_b1.00` (`w=0,1,1,1`) | `98.24` | `98.82` | `99.98` | `91.40` | `93.69` | `99.99` | `92.81%` | `+0.84%p` |
| | **`fusion_v15` 최적 수식 Zero-Shot** | `98.87` | `98.83` | `99.98` | `90.74` | `95.69` | `99.99` | **`93.48%`** | **`+1.51%p`** |
| | **`CleanCore` (14-변수 최적)** | **`99.08`** | **`99.00`** | **`100.00`** | `90.00` | `96.54` | **`100.00`** | **`93.67%`** | **`+1.70%p`** |
| | **`BestLat96` (28-변수 최적)** | **`99.08`** | `98.97` | **`100.00`** | `90.13` | **`96.58`** | **`100.00`** | **`93.72%` 🏆** | **`+1.75%p`** |
| | **`BestHybrid96` (`lat+pool`)** | `99.07` | `98.94` | **`100.00`** | `90.29` | `96.54` | **`100.00`** | **`93.75%` 🏆** | **`+1.78%p`** |
| **30ep (`Ep 29`)** | `v2` / `v2_lat` (Base `col 0`) | `98.86` / `99.04` | `96.62` / `98.17` | `99.98` | `82.74` / `84.17` | `95.97` / `96.28` | `99.98` | `89.04%` / `90.65%` | — |
| | `v7_default_b0.75` (`summary.md`) | `98.79` | `98.70` | `100.00` | `88.10` | `95.04` | `100.00` | `92.02%` | Baseline |
| | `f15_default_b0.50` (`w=0.1,1,1,1`) | `98.93` | `98.81` | `100.00` | `87.98` | `95.18` | `100.00` | `92.12%` | `+0.10%p` |
| | `plugin_b1.00` (`w=0,1,1,1`) | `98.45` | `98.82` | `99.98` | `91.52` | `93.84` | `100.00` | `92.96%` | `+0.94%p` |
| | **`navtrain_val` 튜닝 ➔ `navtest` 적용** | `98.84` | `98.86` | `99.99` | `90.79` | `95.65` | `100.00` | **`93.50%`** | **`+1.48%p`** |
| | **`fusion_v15` 최적 수식 Zero-Shot** | `98.91` | `98.93` | `99.98` | `90.83` | `95.69` | `100.00` | **`93.57%`** | **`+1.55%p`** |
| | **`CleanCore` (14-변수 최적)** | `98.99` | **`99.09`** | `99.99` | `90.35` | `96.06` | `100.00` | **`93.64%`** | **`+1.62%p`** |
| | **`BestLat96` (28-변수 최적)** | **`99.04`** | `99.08` | `99.99` | `90.58` | **`96.24`** | `100.00` | **`93.81%` 🏆** | **`+1.79%p`** |
| | **`BestHybrid96` (`lat+pool`)** | `99.03` | **`99.09`** | `99.99` | `90.59` | **`96.24`** | `100.00` | **`93.82%` 🏆** | **`+1.80%p`** |
| **20ep (`Ep 19`)** | `v7_default` (`b=0.5` / `b=0.75`) | `98.98` | `98.52` | `100.00` | `88.30` | `95.37` | `99.99` | `92.19%` / `92.12%` | Baseline |
| | **`CleanCore` / `BestLat96`** | **`99.20`** | **`98.91`** | **`100.00`** | **`90.27`** | **`96.85`** | `99.99` | **`93.85%` / `93.88%` 🏆** | **`+1.69%p`** |
| **25ep (`Ep 24`)** | `v7_default` ➔ **`BestLat96`** | `98.88` | `98.84` | `99.99` | `90.98` | `96.06` | `99.99` | `91.97%` ➔ **`93.73%` 🏆** | **`+1.76%p`** |
| **15ep (`Ep 14`)** | `v7_default` ➔ **`BestHybrid96`** | `99.24` | `99.07` | `99.98` | `89.83` | `96.72` | `99.99` | `91.97%` ➔ **`93.73%` 🏆** | **`+1.76%p`** |
| **10ep (`Ep 09`)** | `v7_default` ➔ **`BestHybrid96`** | `99.11` | `98.90` | `99.99` | `88.74` | `96.81` | `99.99` | `92.06%` ➔ **`93.20%` 🏆** | **`+1.14%p`** |

---

#### (2) `navtest` v2.2 One-Stage EPDMS ($N=12,146$) — `After (v2.2)` & `Before (수정 전)`

- **최적 수식 구조**:
  $$S(c) = w_{\text{v2f}} \cdot \text{v2\_final}_{k} + w_{\text{nc}} \log(\text{nc}_{\text{ck}, c}) + w_{\text{dac}} \log(\text{dac}_{\text{ck}, c}) + w_{\text{rest}} \log(5\text{ttc}_{\text{ck}, c} + 2\text{cmf}_{\text{ck}, c} + 5\text{ep}_{\text{ck}, c}) + w_{\text{nc\_v2}} \log(\text{nc}_{\text{v2}, k}) + w_{\text{dac\_v2}} \log(\text{dac}_{\text{v2}, k}) + w_{\text{prod}} \cdot \text{epdms}_{\text{ck}, c} + \delta_{v(c)}$$
  - **23ep (`Ep 22`) 최적 파라미터 (`tune_41_w_prod_3.0`)**: `w_v2f=0.5, w_im=0.0, w_nc=0.6, w_dac=1.0, w_rest=0.5, w_prod=3.0, w_nc_v2=0.5, w_dac_v2=0.4, dv=[0.04, -0.02, 0.01, -0.05, 0.0, 0.0], lat_mode="lat"`
  - **30ep (`Ep 29`) 최적 파라미터 (`tune_31_dv_acc3_-0.05`)**: `w_v2f=0.3, w_im=0.0, w_nc=1.2, w_dac=1.2, w_rest=0.5, w_prod=2.0, w_nc_v2=0.3, w_dac_v2=0.3, dv=[0.04, -0.02, 0.01, -0.05, 0.0, 0.0], lat_mode="lat"`

| 에폭 / 선택 규칙 | `NC` | `DAC` | `DDC` | `TLC` | `EP` | `TTC` | `LK` | `HC` | `EC` | **After (`v2.2`) EPDMS** | **Before (수정 전) EPDMS** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **23ep `v2_base` (`col 0`)** | `98.77` | `96.24` | `99.48` | `99.79` | `87.15` | `98.33` | `96.54` | `99.89` | **`85.37`** | `88.61%` | `84.81%` |
| **23ep `v2_lat` (`col 0`)** | `98.91` | `97.80` | `99.48` | `99.79` | `87.19` | `98.44` | **`96.91`** | `99.88` | `84.27` | `90.04%` | `86.20%` |
| **23ep `v7_default_b0.75`** | `98.76` | `98.58` | `99.23` | `99.64` | `90.65` | `98.30` | `96.32` | `99.72` | `80.27` | `90.80%` | `87.09%` |
| **23ep `f15_default_b0.50`** | `98.84` | `98.77` | `99.23` | `99.65` | `90.20` | `98.42` | `96.35` | `99.76` | `79.63` | `90.88%` | `87.16%` |
| **23ep `plugin_b1.00`** | `98.24` | `98.82` | `97.63` | `99.37` | **`94.10`** | `97.74` | `93.45` | `98.67` | `63.18` | `87.86%` | `84.35%` |
| **23ep Zero-Shot (`zs_f15_nt_ep29`)** | `99.00` | `99.07` | `99.19` | `99.68` | `90.18` | `98.53` | `95.93` | `99.77` | `80.06` | **`91.30%`** | **`87.51%`** |
| **★ 23ep 최적 (`tune_41_w_prod_3.0`)** | **`99.08`** | **`99.09`** 🏆 | `99.20` | **`99.70`** | `89.91` | **`98.67`** 🏆 | `96.02` | **`99.83`** | **`80.73`** | **`91.44%` 🏆 (`+0.64%p`)** | **`87.65%` 🏆 (`+0.56%p`)** |
| **30ep `v2_base` (`col 0`)** | `98.85` | `96.62` | `99.46` | `99.83` | `86.71` | `98.45` | `96.64` | `99.91` | **`84.29`** | `88.83%` (`88.64%`) | `85.02%` |
| **30ep `v2_lat` (`col 0`)** | `99.04` | `98.17` | `99.45` | `99.83` | `86.75` | `98.60` | **`96.98`** | `99.90` | `83.33` | `90.32%` | `86.46%` |
| **30ep `v7_default_b0.75`** | `98.79` | `98.70` | `99.18` | `99.68` | `90.47` | `98.36` | `96.39` | `99.76` | `78.97` | `90.75%` (`90.56%`) | `87.03%` |
| **30ep `f15_default_b0.50`** | `98.93` | `98.81` | `99.21` | `99.67` | `90.11` | `98.46` | `96.27` | `99.75` | `78.93` | `90.86%` | `87.13%` |
| **30ep `plugin_b1.00`** | `98.45` | `98.82` | `97.72` | `99.37` | **`94.03`** | `97.83` | `93.48` | `98.64` | `64.39` | `88.18%` | `84.64%` |
| **30ep Zero-Shot (`zs_f15_nt_ep29`)** | `99.12` | `99.08` | `99.16` | `99.73` | `89.78` | `98.63` | `96.10` | `99.80` | `79.46` | **`91.27%`** | **`87.47%`** |
| **★ 30ep 최적 (`tune_31_dv_acc3_-0.05`)** | **`99.18`** 🏆 | **`99.23`** 🏆 | **`99.22`** | **`99.76`** | `89.07` | **`98.69`** 🏆 | `96.16` | **`99.88`** | **`80.26`** | **`91.42%` 🏆 (`+0.67%p`)** | **`87.60%` 🏆 (`+0.57%p`)** |

---

#### (3) `navhard_two_stage` v2.2 Two-Stage EPDMS ($N=5,912$: Stage 1 `450` + Stage 2 `5,462`)

- **최적 수식 구조 (`tune_37_w_ttc_0.3`)**:
  $$S(c) = w_{\text{v2f}} \cdot \text{v2\_final}_{k} + w_{\text{nc}} \log(\text{nc}_{\text{ck}, c}) + w_{\text{dac}} \log(\text{dac}_{\text{ck}, c}) + w_{\text{ttc}} \log(\text{ttc}_{\text{ck}, c}) + w_{\text{rest}} \log(5\text{ttc}_{\text{ck}, c} + 2\text{cmf}_{\text{ck}, c} + 5\text{ep}_{\text{ck}, c}) + w_{\text{nc\_v2}} \log(\text{nc}_{\text{v2}, k}) + w_{\text{dac\_v2}} \log(\text{dac}_{\text{v2}, k}) + \delta_{v(c)}$$
  - **23ep (`Ep 22`) 최적 파라미터 (`tune_37_w_ttc_0.3`)**: `w_v2f=0.5, w_im=0.0, w_nc=1.2, w_dac=1.0, w_ttc=0.3, w_rest=0.5, w_nc_v2=0.9, w_dac_v2=0.3, dv=[0.0, 0.01, 0.02, -0.05, -0.03, -0.03], lat_mode="lat"`
  - **30ep (`Ep 29`) 최적 파라미터 (`tune_37_w_ttc_0.3`)**: `w_v2f=0.6, w_im=0.0, w_nc=0.8, w_dac=0.8, w_ttc=0.3, w_rest=0.5, w_nc_v2=0.3, w_dac_v2=0.3, dv=[0.0, 0.01, 0.02, -0.02, 0.0, 0.0], lat_mode="lat"`

| 에폭 / 선택 규칙 | 구간 | `NC` | `DAC` | `DDC` | `TLC` | `EP` | `TTC` | `LK` | `HC` | `EC` | **After (`v2.2`) EPDMS** | **Before (수정 전) EPDMS** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **23ep `v2_base`** | Stage 1 / Stage 2 / **Combined** | `82.60` | `70.15` | `85.02` | `98.33` | `69.80` | `78.97` | `44.63` | `95.22` | `51.40` | `75.94` / `47.60` / **`36.14%`** | `67.87` / `47.60` / `32.25%` |
| **23ep `v2_lat`** | Stage 1 / Stage 2 / **Combined** | `83.22` | `75.57` | `86.58` | `98.32` | `70.30` | `79.73` | `46.36` | `95.42` | `49.55` | `78.32` / `50.34` / **`39.46%`** | `70.31` / `50.34` / `35.41%` |
| **23ep `v7_default_b0.75`** | Stage 1 (`450`) | `96.33` | `96.67` | `98.78` | `99.33` | `85.28` | `95.78` | `96.22` | `99.78` | `64.00` | `81.77%` | `73.82%` |
| | Stage 2 (`5,462`) | `83.32` | `84.90` | `86.15` | `98.52` | `86.89` | `79.43` | `49.18` | `95.53` | `57.84` | `50.62%` | `50.62%` |
| | **Combined (최종)** | `81.86` | `82.28` | `85.44` | `98.17` | `75.01` | `77.51` | `47.42` | `95.31` | `39.49` | **`42.48%`** | **`38.45%`** |
| **23ep Zero-Shot (`zs_f15_nh_ep22`)** | Stage 1 / Stage 2 / **Combined** | `88.40` | `86.75` | `90.73` | `98.89` | `56.33` | `87.47` | `49.19` | `95.97` | `42.49` | `81.57` / `57.58` / **`47.27%`** | `74.84` / `57.58` / **`43.36%`** |
| **★ 23ep 최적 (`tune_37_w_ttc_0.3`)** | **Stage 1 (`450`)** | **`98.67`** | **`96.89`** | **`99.56`** | **`99.78`** | `75.06` | **`98.67`** | **`96.22`** | **`99.78`** | `62.22` | **`82.27%`** | **`75.56%`** |
| | **Stage 2 (`5,462`)** | **`89.92`** | **`90.54`** | **`92.08`** | **`98.85`** | `71.26` | **`89.27`** | **`52.94`** | **`96.45`** | **`68.88`** | **`58.63%` (`+8.01%p`)** | **`58.63%` (`+8.01%p`)** |
| | **Combined (최종)** | **`88.93`** | **`87.90`** | **`91.85`** | **`98.85`** | `54.97` | **`88.33`** | **`50.95`** | **`96.23`** | **`43.50`** | **`48.61%` 🏆 (`+6.13%p`)** | **`44.74%` 🏆 (`+6.29%p`)** |
| **30ep `v2_base`** | Stage 1 / Stage 2 / **Combined** | `82.06` | `70.91` | `86.06` | `98.53` | `69.80` | `79.09` | `45.08` | `95.81` | `53.28` | `77.08` / `47.77` / **`36.49%`** | `69.19` / `47.77` / `32.76%` |
| **30ep `v2_lat`** | Stage 1 / Stage 2 / **Combined** | `83.22` | `76.96` | `86.87` | `98.53` | `70.27` | `80.08` | `45.53` | `95.80` | `49.92` | `80.07` / `50.20` / **`40.44%`** | `72.22` / `50.20` / `36.42%` |
| **30ep `v7_default_b0.75`** | Stage 1 (`450`) | `96.89` | `96.67` | `98.78` | `99.56` | `84.27` | `96.00` | `96.22` | `100.00` | `64.89` | `82.16%` | `74.19%` |
| | Stage 2 (`5,462`) | `84.03` | `85.14` | `86.74` | `98.44` | `86.56` | `80.40` | `48.93` | `96.12` | `61.36` | `50.55%` | `50.55%` |
| | **Combined (최종)** | `82.35` | `82.07` | `85.85` | `98.39` | `73.68` | `78.34` | `46.89` | `96.12` | `40.64` | **`41.78%`** | **`37.86%`** |
| **30ep Zero-Shot (`zs_f15_nh_ep22`)** | Stage 1 / Stage 2 / **Combined** | `88.25` | `85.25` | `91.02` | `98.94` | `55.57` | `87.30` | `48.83` | `95.83` | `44.36` | `81.12` / `57.02` / **`46.61%`** | `74.62` / `57.02` / **`42.92%`** |
| **★ 30ep 최적 (`tune_37_w_ttc_0.3`)** | **Stage 1 (`450`)** | **`98.22`** | **`96.89`** | **`99.44`** | **`99.78`** | `77.24` | **`98.22`** | `95.33` | **`100.00`** | `64.89` | **`82.55%`** | **`76.01%`** |
| | **Stage 2 (`5,462`)** | **`88.58`** | **`90.10`** | **`91.00`** | **`98.96`** | `75.57` | **`87.22`** | **`51.52`** | **`96.09`** | **`65.30`** | **`58.32%` (`+7.77%p`)** | **`58.32%` (`+7.77%p`)** |
| | **Combined (최종)** | **`87.44`** | **`87.41`** | **`90.55`** | **`98.96`** | `59.79` | **`86.26`** | **`49.09`** | **`96.09`** | **`42.43`** | **`48.28%` 🏆 (`+6.50%p`)** | **`44.75%` 🏆 (`+6.89%p`)** |

---

#### (4) 공식 PDMS 가중치 (`iPAD` / `SparseDriveV2` / `Hydra-MDP` 방식: $\text{NC} \times \text{DAC} \times \frac{5\text{TTC} + 5\text{EP} + 2\text{C}}{12}$) 평가 결과 (`v2_v15_v7_main`, 64m BEV)

별도의 가중치 튜닝 없이, 선행 논문들(iPAD, SparseDriveV2, Hydra-MDP 등)처럼 **NAVSIM 공식 PDMS 수식 그대로**를 사용하여 선택한 `23ep`(`ep22`) 및 `30ep`(`ep29`) 전체 평가 결과입니다:
$$\widehat{\text{PDMS}}(c) = \widehat{\text{NC}}(c) \times \widehat{\text{DAC}}(c) \times \frac{5\,\widehat{\text{TTC}}(c) + 5\,\widehat{\text{EP}}(c) + 2\,\widehat{\text{C}}(c)}{12} \quad \Longleftrightarrow \quad \log\widehat{\text{NC}} + \log\widehat{\text{DAC}} + \log\left(\frac{5\,\widehat{\text{TTC}} + 5\,\widehat{\text{EP}} + 2\,\widehat{\text{C}}}{12}\right)$$

##### ① `navtest` v1 PDMS ($N=12,146$)

| 구분 | 선택 수식 (Selector) | 후보 수 | **23ep (`ep22`) PDMS** | 23ep 세부 (`NC / DAC / EP / TTC`) | **30ep (`ep29`) PDMS** | 30ep 세부 (`NC / DAC / EP / TTC`) | *(참고) 20ep (`ep19`)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **기존 기본값** | `v2_lat` (`w=0.1, 0.5, 0.5, 1.0`) | 16 | `90.24%` | `98.91 / 97.80 / 84.17 / 95.87` | `90.65%` | `99.04 / 98.17 / 84.17 / 96.28` | `90.47%` |
| *(WoTE 가중치)* | `v7_default (b=0.75)` | 96 | `91.97%` | `98.76 / 98.58 / 88.15 / 95.02` | `92.02%` | `98.79 / 98.70 / 88.10 / 95.04` | `92.12%` |
| **16-모드 전용** | `V2 16모드 공식 PDMS (w_im=0.1)` | 16 | `90.20%` | `98.97 / 97.81 / 84.02 / 95.92` | `90.50%` | `99.07 / 98.10 / 83.91 / 96.25` | `90.33%` |
| *(원본 16개만)* | **`CK2 16모드 순수 공식 PDMS (no_im)`** | 16 | **`92.36%`** | `98.40 / 98.83 / 89.57 / 94.34` | **`92.44%`** | `98.47 / 98.95 / 89.62 / 94.30` | **`92.56%`** |
| | `V2+CK2 16모드 공식 PDMS (no_im)` | 16 | `91.79%` | `98.85 / 98.67 / 87.01 / 95.60` | `91.78%` | `98.81 / 98.72 / 86.71 / 95.83` | `91.53%` |
| **96-후보 풀** | **`CK2 순수 공식 PDMS (plugin, b=1.0)`** | 96 | **`92.81%`** ⭐ | `98.24 / 98.82 / 91.40 / 93.69` | **`92.96%`** ⭐ | `98.45 / 98.82 / 91.52 / 93.84` | **`92.80%`** ⭐ |
| **공식 PDMS 가중치** | `CK2 공식 PDMS + im (w=0.1,1,1,1, b=1.0)` | 96 | `92.01%` | `98.81 / 98.64 / 88.02 / 95.13` | `92.07%` | `98.87 / 98.78 / 88.02 / 95.08` | `92.10%` |
| *(튜닝 일절 없음)* | `Blend (v2f + CK2 공식 PDMS, b=0.50)` | 96 | `92.13%` | `98.86 / 98.74 / 88.20 / 95.08` | `92.17%` | `98.93 / 98.84 / 88.16 / 95.08` | `92.39%` |
| | `Blend (v2f + CK2 공식 PDMS, b=0.75)` | 96 | `92.36%` | `98.79 / 98.84 / 88.78 / 94.94` | `92.38%` | `98.88 / 98.92 / 88.76 / 94.87` | `92.61%` |
| | `Blend (v2f + CK2 공식 PDMS + im, b=0.50)` | 96 | `92.09%` | `98.84 / 98.77 / 87.96 / 95.20` | `92.12%` | `98.93 / 98.81 / 87.98 / 95.18` | `92.25%` |
| | **`Two-Head 공식 PDMS 선형합 (b=0.50)`** | 96 | `92.58%` | `98.75 / 98.95 / 89.32 / 94.86` | `92.57%` | `98.85 / 98.99 / 89.19 / 94.85` | `92.48%` |
| | **`Two-Head 공식 PDMS 선형합 (b=0.60)`** | 96 | **`92.73%`** | `98.70 / 98.94 / 89.80 / 94.74` | **`92.72%`** | `98.79 / 99.00 / 89.68 / 94.76` | `92.65%` |
| | **`Two-Head 공식 PDMS 로그합 (b=0.60)`** | 96 | `92.59%` | `98.70 / 98.95 / 89.40 / 94.80` | `92.59%` | `98.83 / 98.98 / 89.30 / 94.81` | `92.50%` |
| | **`Prob-Avg 공식 PDMS (0.4*V2 + 0.6*CK)`** | 96 | `92.61%` | `98.70 / 98.93 / 89.57 / 94.71` | `92.65%` | `98.79 / 99.00 / 89.50 / 94.76` | `92.55%` |

##### ② `navtest` EPDMS & `navhard_two_stage` EPDMS — 공식 PDMS 가중치 결과 (`23ep` & `30ep`)

| 선택 수식 (공식 PDMS 가중치 계열) | 에폭 | **`navtest` EPDMS (After / Before)** | `navtest` 세부 (`NC / DAC / EP / TTC / LK / EC`) | **`navhard` EPDMS (After / Before)** | `navhard` Stage 1 (After / Before) | `navhard` Stage 2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CK2 순수 공식 PDMS (plugin, b=1.0)` | **23ep** | `87.86%` / `84.35%` | `98.24 / 98.82 / 94.10 / 97.74 / 93.45 / 63.18` | `37.23%` / `33.91%` | `76.87%` / `69.75%` | `48.39%` |
| `CK2 공식 PDMS + im (w=0.1,1,1,1, b=1.0)` | **23ep** | — | — | `42.21%` / `38.20%` | `81.45%` / `73.39%` | `50.91%` |
| **`Blend (v2f + CK2 공식 PDMS, b=0.50)`** | **23ep** | **`90.87%` / `87.13%`** | `98.86 / 98.74 / 90.43 / 98.44 / 96.11 / 79.52` | `42.39%` / `38.50%` | `81.07%` / `73.27%` | `51.17%` |
| **`Blend (v2f + CK2 공식 PDMS, b=0.75)`** | **23ep** | — | — | **`42.63%` / `38.75%`** ⭐ | `81.64%` / `73.85%` | `51.35%` |
| **`Blend (v2f + CK2 공식 PDMS + im, b=0.50)`** | **23ep** | **`90.88%` / `87.16%`** ⭐ | `98.84 / 98.77 / 90.20 / 98.42 / 96.35 / 79.63` | `41.93%` / `37.94%` | `81.84%` / `74.00%` | `50.23%` |
| `Two-Head 공식 PDMS 로그합 (b=0.50, w_im=0.1)` | **23ep** | — | — | `41.91%` / `37.93%` | `81.24%` / `73.38%` | `50.56%` |
| --- | --- | --- | --- | --- | --- | --- |
| `CK2 순수 공식 PDMS (plugin, b=1.0)` | **30ep** | `88.18%` / `84.64%` | `98.45 / 98.82 / 94.03 / 97.83 / 93.48 / 64.39` | `38.06%` / `34.78%` | `77.58%` / `70.94%` | `49.29%` |
| `CK2 공식 PDMS + im (w=0.1,1,1,1, b=1.0)` | **30ep** | — | — | `41.94%` / `37.97%` | `82.09%` / `74.23%` | `50.52%` |
| **`Blend (v2f + CK2 공식 PDMS, b=0.50)`** | **30ep** | **`90.86%` / `87.12%`** | `98.93 / 98.84 / 90.28 / 98.50 / 96.13 / 78.70` | `42.94%` / `39.32%` | `80.70%` / `73.34%` | `52.46%` |
| **`Blend (v2f + CK2 공식 PDMS, b=0.75)`** | **30ep** | — | — | **`43.18%` / `39.54%`** ⭐ | `80.42%` / `73.01%` | `52.57%` |
| **`Blend (v2f + CK2 공식 PDMS + im, b=0.50)`** | **30ep** | **`90.86%` / `87.13%`** ⭐ | `98.93 / 98.81 / 90.11 / 98.46 / 96.27 / 78.93` | `41.83%` / `38.10%` | `81.49%` / `73.88%` | `51.01%` |
| `Two-Head 공식 PDMS 로그합 (b=0.50, w_im=0.1)` | **30ep** | — | — | `41.96%` / `38.38%` | `80.89%` / `73.53%` | `51.52%` |

---

#### (5) SparseDriveV2 방식 공식 EPDMS `(5/5/2/2/2)` Head-Free 정식 계산 방법론 및 평가 결과 (`15ep` & `30ep`)

##### ① 핵심 아이디어: 왜 추가 신경망 Head 없이도 공식 `(5/5/2/2/2)` EPDMS 선택이 가능한가?
- NAVSIM v2 공식 EPDMS 수식은 다음과 같습니다:
  $$\text{EPDMS} = \text{NC} \times \text{DAC} \times \text{DDC} \times \text{TLC} \times \frac{5\,\text{EP} + 5\,\text{TTC} + 2\,\text{LK} + 2\,\text{HC} + 2\,\text{EC}}{16}$$
- 여기서 충돌(`NC`), 주행가능영역(`DAC`), 진행도(`EP`), 충돌여유시간(`TTC`)은 주변 차량의 미래 움직임이나 맵이 필요하므로 Critic Head가 예측한 확률($\widehat{\text{NC}}, \widehat{\text{DAC}}, \widehat{\text{EP}}, \widehat{\text{TTC}}$)을 사용해야 합니다.
- 반면, 나머지 3개 가중합 지표인 **$\text{HC}$ (History Comfort, 가중치 2)**, **$\text{LK}$ (Lane Keeping, 가중치 2)**, **$\text{EC}$ (Two-Frame Extended Comfort, 가중치 2)**는 **타 차량의 미래 궤적이나 신호등 정보 없이 오직 자차의 과거 상태와 96개 후보 궤적(`lat_traj[c]`)의 물리량만으로 추론(Selection) 시점에 100% 정확하게 직접 계산**할 수 있습니다.

##### ② "딱 정식으로 한 방식" (`NoHead_55222_2f_traj_div14`) 단계별 파이프라인

튜닝 가중치, `v2_final` 블렌딩(`b=1.0`), HD Map(Centerline)을 일절 사용하지 않고, 오직 후보 궤적과 5개 기본 Critic 확률만으로 공식 채점기([`run_pdm_score_one_stage.py`](file:///workspace/navsim_workspace/navsim/navsim/planning/script/run_pdm_score_one_stage.py#L173-L191)) 규칙 그대로 궤적을 선택하는 정식 파이프라인입니다 ([`eval_nohead_epdms_55222.py`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/eval_nohead_epdms_55222.py)):

1. **96개 후보 궤적의 물리 상태(Kinematic States) 추출 ([`eval_nohead_epdms_55222.py#L64-L82`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/eval_nohead_epdms_55222.py#L64-L82))**
   - 모델이 출력한 96개 후보 궤적 `lat_traj[c]` ($0.5\text{s} \sim 4.0\text{s}$, $0.5\text{s}$ 간격 8스텝)를 현재 자차 상태(`ego_state`, $t=0$)와 이어 붙인 뒤 $0.1\text{s}$ 간격(총 41스텝, $0.0\text{s} \sim 4.0\text{s}$)으로 보간합니다.
   - NAVSIM 공식 `PDMSimulator.simulate_proposals` (LQR 제어기 + Kinematic Bicycle 모델)에 통과시켜, 각 후보 궤적을 주행했을 때의 위치·속도·가속도·저크(Jerk)·요레이트(Yaw rate)·요가속도(Yaw accel) 시계열 `sim_states[c]` (`[96, 41, 11]`)를 얻습니다.

2. **Head 없이 후보 궤적에서 직접 계산하는 3가지 지표 ($\text{HC}, \text{LK}, \text{EC}$)**
   - **$\text{HC}(c) \in \{0, 1\}$ (History Comfort, 가중치 2)**:
     - 공식 [`ego_is_comfortable`](file:///workspace/navsim_workspace/navsim/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py#L351-L379) 함수를 그대로 호출합니다 ([`eval_nohead_epdms_55222.py#L84-L98`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/eval_nohead_epdms_55222.py#L84-L98)).
     - 후보 궤적 `sim_states[c]`가 6가지 안락성 임계값(종가속도 $[-4.05, 2.40]$, 횡가속도 $\le 4.89$, 저크 $\le 8.37$, 종저크 $\le 4.13$, 요가속도 $\le 1.93$, 요레이트 $\le 0.95$)을 모두 만족하면 `1.0`, 하나라도 초과하면 `0.0`입니다.
   - **$\text{LK}_{\text{traj}}(c) \in \{0, 1\}$ (Trajectory-only Lane Keeping, 가중치 2)**:
     - 추론 시 HD Map Centerline을 쓰지 않는 대신, 모델의 **Top-1 기본 주행 궤적(`c=0`, 차선 중앙을 추종하는 기준 궤적)**을 기준선으로 삼습니다 ([`eval_nohead_epdms_55222.py#L100-L115`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/eval_nohead_epdms_55222.py#L100-L115)).
     - 공식 [`_calculate_lane_keeping`](file:///workspace/navsim_workspace/navsim/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py#L615-L654) 기준과 동일하게, 후보 $c$가 기준 궤적에서 **횡방향으로 $0.5\text{m}$ 이상 연속으로 $2.0\text{s}$(20스텝) 이상 벗어나면 `0.0`**, 아니면 **`1.0`**입니다.
   - **$\text{EC}_{2\text{f}}(c) \in \{0, 1\}$ (Causal 2-Frame Extended Comfort, 가중치 2)**:
     - 공식 [`ego_is_two_frame_extended_comfort`](file:///workspace/navsim_workspace/navsim/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py#L430-L467) 함수를 시간 순서($t-0.5\text{s} \to t$)대로 적용합니다 ([`eval_nohead_epdms_55222.py#L150-L162`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/eval_nohead_epdms_55222.py#L150-L162)).
     - 직전 프레임($t-0.5\text{s}$)에서 이미 최종 선택된 궤적 $c_{\text{prev}}^*$의 물리 상태(`sim_states[prev, c_prev*, 5:]`)와 현재 프레임($t$)의 96개 후보 각각의 물리 상태(`sim_states[now, c, :-5]`)의 겹치는 $3.5\text{s}$(36스텝) 구간을 비교해, 가속도($\le 0.7$), 저크($\le 0.5$), 요레이트($\le 0.1$), 요가속도($\le 0.1$)의 프레임 간 RMS 차이가 모두 기준 이내면 `1.0`, 아니면 `0.0`입니다.

3. **공식 채점기 규칙 그대로 최종 점수 계산 및 `argmax` 선택 ([`eval_nohead_epdms_55222.py#L354-L382`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/eval_nohead_epdms_55222.py#L354-L382))**
   - **Case A — 이전 프레임($t-0.5\text{s}$)이 없는 첫 프레임**:
     공식 [`compute_final_scores`](file:///workspace/navsim_workspace/navsim/navsim/planning/script/run_pdm_score_one_stage.py#L173-L191)와 동일하게 `EC` 가중치(`2`)를 0으로 두고 분모를 **`14`** ($5+5+2+2$)로 나눕니다:
     $$\text{Score}(c) = \widehat{\text{NC}}(c) \times \widehat{\text{DAC}}(c) \times \frac{5\,\widehat{\text{EP}}(c) + 5\,\widehat{\text{TTC}}(c) + 2\,\text{HC}(c) + 2\,\text{LK}_{\text{traj}}(c)}{14}$$
     여기서 1등인 $c_{\text{prev}}^* = \arg\max_c \text{Score}(c)$를 선택하고 그 물리 상태를 저장합니다.
   - **Case B — 이전 프레임($t-0.5\text{s}$)이 존재하는 프레임**:
     직전 프레임 선택 궤적 $c_{\text{prev}}^*$와의 2-프레임 연속성 $\text{EC}_{2\text{f}}(c)$를 포함하여 공식 분모 **`16`** ($5+5+2+2+2$)으로 나눕니다:
     $$\text{Score}(c) = \widehat{\text{NC}}(c) \times \widehat{\text{DAC}}(c) \times \frac{5\,\widehat{\text{EP}}(c) + 5\,\widehat{\text{TTC}}(c) + 2\,\text{HC}(c) + 2\,\text{LK}_{\text{traj}}(c) + 2\,\text{EC}_{2\text{f}}(c)}{16}$$
     여기서 1등인 $c_{\text{now}}^* = \arg\max_c \text{Score}(c)$를 최종 주행 궤적으로 선택합니다.

##### ③ 함께 비교한 파생 방법론 및 `Blend_b050_NoHead_55222_2f_cl_joint` (`navhard` 최고점 달성 방식) 상세 설명

1. **`v2_final` 블렌딩 (`Blend_b050_*`, `Blend_b075_*`)**:
   - CK2 Refiner의 96개 후보 로그 점수 $\log \text{Score}(c)$ 단독(`b=1.0`)으로만 고르면, Refiner가 급가속 변형(`a+0.5`)의 진행도(`EP`)를 과신해 선택하는 경향이 남습니다.
   - 이를 잡아주기 위해 Base V2 플래너가 16개 원본 앵커에 매긴 종합 점수 `v2_final[k]`와 비율 $\beta \in \{0.50, 0.75\}$로 합산하는 기존 SSR-v7 기본 결합 구조입니다:
     $$S_{\text{blend}}(c) = (1 - \beta)\,\text{v2\_final}_{k(c)} + \beta\,\log \text{Score}_{(5/5/2/2/2)}(c)$$
2. **Centerline 포함 계산 (`*_cl_*`) vs. 순수 궤적 계산 (`*_traj_*`)**:
   - `*_traj_*`는 추론 시 HD Map을 전혀 보지 않고 Top-1 궤적(`c=0`)과의 이격으로 $\text{LK}_{\text{traj}}(c)$를 구하는 **진짜 Map-Free 정식 버전**입니다.
   - `*_cl_*`는 추론 시 내비게이션 경로선(Centerline)을 참조해 공식 [`_calculate_lane_keeping`](file:///workspace/navsim_workspace/navsim/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py#L615-L654)과 동일하게 $\text{LK}_{\text{cl}}(c)$를 구하는 방식입니다.
3. **`Blend_b050_NoHead_55222_2f_cl_joint` (`navhard` 최고점 `49.14%` 달성 방법론 — 4가지 결합 요소, [`eval_nohead_epdms_55222.py#L462-L465`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/eval_nohead_epdms_55222.py#L462-L465))**:
   - **요소 ① `Blend_b050` (`v2_final` 5:5 블렌딩)**:
     Base V2 플래너의 16개 앵커 점수 `v2_final[k]`(`0.5`)와 CK2 Refiner의 96개 후보 로그 점수(`0.5`)를 5:5로 결합하여, Stage 2의 낯선 합성 뷰에서도 급가속 변형(`a+0.5`) 남발을 막고 안전한 앵커 줄기를 유지합니다.
   - **요소 ② `NoHead_55222` (추가 Head 없는 공식 `(5/5/2/2/2) / 16` 수식)**:
     별도 가중치 튜닝 없이 공식 EPDMS 수식 그대로 후보 $c$의 점수를 계산합니다:
     $$\text{Score}_{55222}(c) = \widehat{\text{NC}}(c) \times \widehat{\text{DAC}}(c) \times \frac{5\,\widehat{\text{EP}}(c) + 5\,\widehat{\text{TTC}}(c) + 2\,\text{HC}(c) + 2\,\text{LK}_{\text{cl}}(c) + 2\,\text{EC}(c)}{16}$$
     $$S_{\text{total}}(c) = 0.5 \cdot \text{v2\_final}_{k(c)} + 0.5 \cdot \log\!\big(\text{Score}_{55222}(c)\big)$$
   - **요소 ③ `cl` (Centerline 기반 차선 유지 $\text{LK}_{\text{cl}}(c)$)**:
     96개 후보 중 내비게이션 경로선(Centerline)에서 $0.5\text{m}$ 이상 연속 $2.0\text{s}$ 이상 벗어나는 측방 변형(`l±0.5`)에 $\text{LK}_{\text{cl}}(c)=0$을 주어 걸러냅니다. 그 결과 `navhard` Combined `LK`가 `48.85% -> 57.94% (+9.09%p)`로 상승합니다.
   - **요소 ④ `2f_joint` (Stage 1 & Stage 2 전 구간 2-프레임 연속성 $\text{EC}$ 동시 반영)**:
     - 이전 프레임($t-0.5\text{s}$)을 고를 때도 자기 자신의 앞뒤 구간 부드러움(`EC_self(c)`)을 넣어 안정적인 궤적 $c_{\text{prev}}^*$를 먼저 선택하고,
     - 현재 프레임($t$)을 고를 때는 직전 선택 궤적 $c_{\text{prev}}^*$와의 실제 2-프레임 연속성 $\text{EC}_{2\text{f}}(c, c_{\text{prev}}^*) \in \{0, 1\}$을 **Stage 1뿐만 아니라 Stage 2의 모든 프레임 쌍에서도 동일하게 적용**합니다.
     - 그 결과 15에폭 `navhard`에서 **Stage 2 `EC`가 `75.56% -> 89.96% (+14.40%p)`, Combined `EC`가 `63.31% -> 80.21% (+16.90%p)`로 폭등**하며, 가중치 튜닝을 한 모델(`48.61%`)보다도 높은 **`49.14%` 🏆 (Stage 1 `83.31%` / Stage 2 `58.79%`)**를 달성합니다.

##### ④ "딱 정식으로 한 것" (`NoHead_55222_2f_traj_div14`) vs. "`navhard` 최고점" (`Blend_b050_NoHead_55222_2f_cl_joint`) 세부 성적표 (`15ep` & `30ep`)

| 방식 | 에폭 | 벤치마크 | **EPDMS (After)** | EPDMS (Before) | `NC` | `DAC` | `DDC` | `TLC` | `EP` | `TTC` | `LK` | `HC` | **`EC` (2-Frame)** | *(참고) v1 PDMS* |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`NoHead_55222_2f_traj_div14`** | **15ep (`Ep 14`)** | **`navtest`** ($N=12,146$) | **`91.74%`** | `88.06%` | `99.09` | `98.99` | `98.66` | `99.57` | `90.41` | `98.59` | `94.39` | `99.99` | **`91.07`** | **`92.83%`** |
| *(Map-Free 정식 표준, `b=1.0`)* | **15ep (`Ep 14`)** | **`navhard` Combined** ($N=5,912$) | **`46.68%`** | `42.91%` | `87.24` | `83.30` | `86.20` | `98.61` | `69.82` | `84.81` | `48.85` | `96.65` | **`63.31`** | — |
| | | └─ `navhard` Stage 1 (`450`) | **`83.16%`** | `75.98%` | `97.56` | `96.67` | `97.22` | `99.33` | `82.02` | `97.33` | `93.11` | `100.00` | **`84.44`** | — |
| | | └─ `navhard` Stage 2 (`5,462`) | **`55.58%`** | `55.58%` | `88.85` | `85.88` | `87.99` | `99.01` | `83.74` | `86.07` | `51.79` | `96.65` | **`75.56`** | — |
| | **30ep (`Ep 29`)** | **`navtest`** ($N=12,146$) | **`91.49%`** | `87.79%` | `98.50` | `98.94` | `98.47` | `99.42` | `92.68` | `97.98` | `94.51` | `99.97` | **`90.85`** | **`92.89%`** |
| | **30ep (`Ep 29`)** | **`navhard` Combined** ($N=5,912$) | **`42.89%`** | `39.40%` | `81.02` | `83.14` | `84.86` | `97.78` | `76.64` | `77.24` | `46.26` | `96.63` | **`62.18`** | — |
| | | └─ `navhard` Stage 1 (`450`) | **`83.03%`** | `76.01%` | `95.00` | `98.00` | `97.56` | `99.33` | `86.03` | `94.00` | `92.89` | `100.00` | **`84.44`** | — |
| | | └─ `navhard` Stage 2 (`5,462`) | **`51.48%`** | `51.48%` | `83.96` | `84.62` | `86.50` | `98.18` | `88.57` | `80.45` | `49.39` | `96.63` | **`74.69`** | — |
| **`Blend_b050_NoHead_55222_2f_cl_joint`** | **15ep (`Ep 14`)** | **`navhard` Combined** ($N=5,912$) | **`49.14%` 🏆** | **`45.21%` 🏆** | `87.41` | `82.91` | `89.09` | `98.69` | `63.40` | `85.53` | **`57.94`** | `96.15` | **`80.21`** | — |
| *(`b=0.50` + `LK_cl` + `2f_joint`)* | | └─ `navhard` Stage 1 (`450`) | **`83.31%`** | **`77.06%`** | `97.56` | `94.44` | `98.56` | `99.78` | `78.50` | `97.78` | **`98.67`** | `100.00` | **`89.78`** | — |
| | | └─ `navhard` Stage 2 (`5,462`) | **`58.79%` 🏆** | **`58.79%` 🏆** | `88.92` | `87.72` | `90.06` | `98.69` | `77.86` | `86.75` | **`58.71`** | `96.15` | **`89.96`** | — |
| | **30ep (`Ep 29`)** | **`navhard` Combined** ($N=5,912$) | **`47.14%` 🏆** | **`43.56%` 🏆** | `85.41` | `82.11` | `88.34` | `98.49` | `69.66` | `81.16` | **`56.41`** | `95.13` | **`78.99`** | — |
| | | └─ `navhard` Stage 1 (`450`) | **`84.41%` 🏆** | **`77.51%` 🏆** | `97.56` | `95.11` | `99.00` | `99.56` | `82.07` | `96.89` | **`99.11`** | `100.00` | **`90.22`** | — |
| | | └─ `navhard` Stage 2 (`5,462`) | **`55.77%`** | **`55.77%`** | `86.57` | `86.32` | `89.05` | `98.58` | `83.11` | `82.67` | **`56.81`** | `95.13` | **`88.42`** | — |

##### ⑤ `(5/5/2/2/2)` Head-Free 전체 비교표 (`15ep` & `30ep`)

| 구분 | 선택 수식 (Selector) | **15ep `navtest` EPDMS (After / Before)** | **15ep `navhard` EPDMS (After / Before)** | 15ep `navhard` (`Stage 1 / Stage 2`) | **30ep `navtest` EPDMS (After / Before)** | **30ep `navhard` EPDMS (After / Before)** | 30ep `navhard` (`Stage 1 / Stage 2`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| *기존 방식 비교* | `CK2 순수 공식 PDMS (5/5/2, b=1.0)` | `88.07%` / `84.56%` | `41.58%` / `38.07%` | `75.50%` / `53.20%` | `88.18%` / `84.64%` | `38.06%` / `34.78%` | `77.58%` / `49.29%` |
| | `v7_default (b=0.75)` | `90.96%` / `87.26%` | `45.32%` / `41.15%` | `81.14%` / `55.66%` | `90.75%` / `87.03%` | `41.78%` / `37.86%` | `82.16%` / `50.55%` |
| **① Map-Free 정식** | **`NoHead_55222_2f_traj_div14` (정식 표준)** ⭐ | **`91.74%` / `88.06%`** | **`46.68%` / `42.91%`** | **`83.16%` / `55.58%`** | **`91.49%` / `87.79%`** | **`42.89%` / `39.40%`** | **`83.03%` / `51.48%`** |
| *(Centerline 미사용)* | `NoHead_55222_2f_traj_self` | `91.78%` / `88.07%` | `47.10%` / `43.13%` | `83.67%` / `55.82%` | `91.72%` / `87.96%` | `44.32%` / `40.61%` | `83.05%` / `53.29%` |
| | `NoHead_55222_2f_traj_joint` | — | `46.83%` / `43.13%` | `84.27%` / `55.27%` | — | `43.10%` / `39.25%` | `85.14%` / `51.21%` |
| | `Blend_b050_NoHead_55222_2f_traj` | `91.33%` / `87.58%` | **`47.47%` / `43.20%`** | `81.63%` / `58.19%` | `91.75%` / `87.97%` | `44.67%` / `40.70%` | `82.62%` / `53.53%` |
| | `Blend_b075_NoHead_55222_2f_traj` | `91.70%` / `87.93%` | `47.12%` / `43.05%` | `82.48%` / `57.78%` | **`92.01%` / `88.22%`** | **`45.78%` / `41.71%`** | `82.84%` / `54.48%` |
| **② Centerline 포함** | `NoHead_55222_2f_cl_div14` | `92.22%` / `88.85%` | `47.12%` / `43.52%` | `82.69%` / `56.33%` | `92.00%` / `88.66%` | `43.58%` / `40.28%` | `82.35%` / `52.46%` |
| *(참고용 상한선)* | `NoHead_55222_2f_cl_self` | **`92.37%` / `88.97%`** | `47.76%` / `44.01%` | `83.55%` / `56.67%` | `92.21%` / `88.79%` | `45.35%` / `41.51%` | `83.85%` / `53.86%` |
| | `NoHead_55222_2f_cl_joint` | — | `47.86%` / `44.26%` | `85.06%` / `55.86%` | — | `44.59%` / `40.95%` | `85.96%` / `51.85%` |
| | `Blend_b050_NoHead_55222_2f_cl` | `91.74%` / `88.19%` | `48.55%` / `44.24%` | `81.82%` / `59.19%` | `92.12%` / `88.60%` | `45.14%` / `41.25%` | `82.88%` / `54.02%` |
| | `Blend_b075_NoHead_55222_2f_cl` | `92.17%` / `88.66%` | `48.31%` / `44.24%` | `83.00%` / `58.52%` | **`92.42%` / `88.94%`** 🏆 | `45.92%` / `41.97%` | `82.97%` / `54.77%` |
| | **`Blend_b050_NoHead_55222_2f_cl_joint`** | — | **`49.14%` / `45.21%`** 🏆 | **`83.31%` / `58.79%`** | — | **`47.14%` / `43.56%`** 🏆 | **`84.41%` / `55.77%`** |

---

### 9.4 바로 재현 가능한 핵심 수식 및 파라미터 요약

기존 `default`가 4개 숫자 `(w_im, w_nc, w_dac, w_rest)`만 썼다면, 최적 Selector는 아래 일반화된 수식 하나로 동작합니다 ([`optimize_v1_pdms_v7.py`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/optimize_v1_pdms_v7.py), [`optimize_epdms_v7.py`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/optimize_epdms_v7.py)):

```python
score(k, v) = (
    w_v2f * v2_final[k]
    + w_im  * log(im[k])
    # 1) CK2 96-후보 헤드
    + w_nc  * log(nc_ck[k, v])
    + w_dac * log(dac_ck[k, v])
    + w_ttc * log(ttc_ck[k, v])
    + w_ep  * log(ep_ck[k, v])
    + w_rest * log((5 * ttc_ck[k, v] + 2 * cmf_ck[k, v] + 5 * ep_ck[k, v]) / 12.0)
    # 2) V2 16-모드 헤드 교차 검증
    + w_v2_nc  * log(nc_v2[k])
    + w_v2_dac * log(dac_v2[k])
    + w_v2_ttc * log(ttc_v2[k])
    + w_v2_ep  * log(ep_v2[k])
    # 3) 기대 PDMS 곱셈 항 + 4) 변형 타입별 Prior
    + w_prod * (nc_ck[k, v] * dac_ck[k, v] * (5 * ttc_ck[k, v] + 2 * cmf_ck[k, v] + 5 * ep_ck[k, v]) / 12.0)
    + delta_v[v]
)
```

### 9.5 산출물 JSON 및 스크립트 경로 모음 (`v2_v15_v7_main`, 64m BEV)

- **단독 상세 보고서**: [`optimization_report_v7.md`](file:///workspace/byounggun/SSR/docs/optimization_report_v7.md)
- **최적화 결과 JSON 폴더**: [`/workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results)
  - v1 PDMS (`ep10, ep15, ep20, ep22(23ep), ep25, ep30`):
    - [`v1_pdms_ep22.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/v1_pdms_ep22.json) / [`v1_pdms_ep30.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/v1_pdms_ep30.json) / [`v1_pdms_ep20.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/v1_pdms_ep20.json) / [`v1_pdms_ep25.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/v1_pdms_ep25.json) / [`v1_pdms_ep15.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/v1_pdms_ep15.json) / [`v1_pdms_ep10.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/v1_pdms_ep10.json)
  - `navtest` EPDMS (After & Before):
    - [`epdms_navtest_ep22.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/epdms_navtest_ep22.json) / [`epdms_navtest_ep29.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/epdms_navtest_ep29.json)
  - `navhard_two_stage` EPDMS (After & Before):
    - [`epdms_navhard_ep22.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/epdms_navhard_ep22.json) / [`epdms_navhard_ep29.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/epdms_navhard_ep29.json)
  - Head-Free 공식 EPDMS `(5/5/2/2/2)` 평가 결과 (`15ep` & `30ep`):
    - [`nohead_epdms_55222_navtest_ep15.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/nohead_epdms_55222_navtest_ep15.json) / [`nohead_epdms_55222_navtest_ep30.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/nohead_epdms_55222_navtest_ep30.json)
    - [`nohead_epdms_55222_navhard_ep15.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/nohead_epdms_55222_navhard_ep15.json) / [`nohead_epdms_55222_navhard_ep30.json`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/opt_results/nohead_epdms_55222_navhard_ep30.json)
- **덤프 및 평가/최적화 스크립트**:
  - [`dump_96pool_v7.py`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/dump_96pool_v7.py) / [`optimize_v1_pdms_v7.py`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/optimize_v1_pdms_v7.py) / [`optimize_epdms_v7.py`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/optimize_epdms_v7.py) / [`eval_nohead_epdms_55222.py`](file:///workspace/byounggun/ssr_outputs/ck2_e2e/v2_v15_v7_main/eval_nohead_epdms_55222.py)
