# BEV Selector Distillation v9
## Planning-Grounded Probe (PGP): 증류용 Cross-Attention Probe를 Planning Loss로 학습

**기준 문서**: [`bev_selector_v8.md`](bev_selector_v8.md), [`bev_selector_v7.md`](bev_selector_v7.md)  
**상태**: 설계 제안 (미구현, 미검증). 아래 효과는 모두 가설이며 §8의 ablation으로 확인해야 한다.  
**목표**: Aux Teacher(BEVFusion, ReSMap)의 BEV 지식을 Student BEV로 증류하되, 증류 attention이 **planning에 실제로 중요한 곳**을 보도록 만든다.

---

## 1. 동기: v8 probe의 한계

v8은 collapse를 막기 위해 증류용 cross-attention의 `anchor_mlp`, `cmd_mlp`, `status_mlp`, `query_norm`, `q/k/v/out_proj`를 모두 `requires_grad_(False)`로 **랜덤 초기화 상태에서 동결**했다 ([`para_ssr_agent.py`](../navsim/agents/para_ssr/para_ssr_agent.py) `get_optimizers`). 이 선택의 결과는 다음과 같다.

| 문제 | 설명 |
|---|---|
| attention이 의미를 학습하지 않음 | `Q·Kᵀ`는 랜덤 투영 공간의 feature 유사도다. 실질적인 "중요 영역"은 공간 편향 `M_k`(궤적 근처)가 정한다. |
| KD 대상이 불분명 | `A_t`는 Teacher 자신의 attention이 아니라 "랜덤 probe가 Teacher BEV를 읽은 분포"다. |
| 사실상 feature distillation | 궤적 주변을 soft하게 모은 feature의 L2/cosine 정렬이 본체이고 attention KL은 보조다. |
| 열면 collapse | 일치도(KL, L2)만 최소화하는 loss에서 probe를 열면 `q_proj→0`, `v_proj→0` 같은 상수 해가 최적이다. |

**근본 원인**: 증류 loss에 "probe가 정보를 보존/선택하라"는 신호가 없다. 일반 cross-attention이 의미를 배우는 이유는 출력이 task loss로 이어지기 때문이다. 따라서 v9는 probe를 **planning loss에 연결**한다.

### 1.1 v8 학습 로그에서 관찰한 점 (W&B run `vaga21vb`, r34_sel_v8_2, 277 step 로그)

| 지표 | 값 | 의미 |
|---|---|---|
| `distill/effective_anchors` | 137~150 | 목표(20~60)보다 훨씬 평평하다. `w_k`가 거의 균일에 가깝다. |
| `distill_cos/*` | 0.97~0.98 (빠르게 포화) | 정렬은 빨리 포화되고 이후 신호가 약하다. |
| `loss_attn_*` (KL) | 0.2~0.5 → 0.13~0.2 | v7(0.013)보다 큰 값에서 시작해 학습 신호는 존재한다. |

---

## 2. v9 핵심 아이디어

```
                  ┌────────────────────────────────────────────────┐
                  │  Planning-Grounded Probe  P_b  (b ∈ {det, map}) │
                  │  256 anchor queries → L layers cross-attn(BEV)  │
                  │  (+ multi-scale spatial bias M_h,k)             │
                  │  heads: im_reward / sim_reward×5 / offset       │
                  └───────────────┬────────────────────────────────┘
                                  │ 같은 P_b
        ┌─────────────────────────┴─────────────────────────┐
        ▼ (Teacher BEV: 캐시, grad 없음)                    ▼ (Student BEV → student_proj_b)
 [P_b, 파라미터 학습 O]                              [P_b, 파라미터 stop-grad]
   A_t, h_t                                            A_s, h_s
   L_probe = planning loss (WTA L1 + reward BCE)       gradient는 BEV / student_proj 로만 흐름
        │                                                  │
        └────────── Attention KD, Feature KD ──────────────┤
                                                           ├─ L_task^S: 같은 planning loss를
                                                           │            Student BEV에 적용
```

1. **Teacher 쪽 probe `P_b`를 planning loss로 학습한다.** Teacher BEV는 캐시된 텐서이므로 BEV로는 gradient가 흐르지 않고, probe 파라미터만 `L_probe`로 업데이트된다. 그러면 attention은 reward/offset 예측에 도움이 되는 곳을 보도록 학습된다.
2. **Student는 같은 probe를 stop-grad 파라미터로 통과한다.** 증류 loss의 gradient는 probe 파라미터에 닿지 않고 Student BEV와 `student_proj`로만 흐른다.
3. **Student BEV에도 planning loss를 직접 건다 (`L_task^S`).** 학습된 probe가 Student BEV에서도 정답 reward/offset을 예측하도록 한다. 이 항은 일치도 loss와 달리 상수 해로 줄일 수 없다.

### 2.1 collapse가 막히는 이유 (가설)
- probe 파라미터를 움직이는 loss는 `L_probe`(planning)뿐이다. `q_proj→0`으로 가면 reward 예측이 나빠지므로 불리하다.
- 증류 loss(KL, L2)는 probe 파라미터를 업데이트하지 못하므로 "차이를 지우는" 방향으로 probe를 망가뜨릴 수 없다.
- 다만 이는 설계상의 논리이며, probe가 planning에서도 위치 정보를 쓰지 않고 편향만으로 풀 가능성은 남아 있다. §6 진단으로 확인한다.

---

## 3. 구성 요소

### 3.1 Planning-Grounded Probe `P_b`
- **입력 query**: v8과 동일하게 256개 앵커를 `anchor_mlp`로 임베딩하고 `cmd_mlp`, `status_mlp`를 더한 뒤 `query_norm`을 적용한다 (이번에는 **학습 가능**).
- **본체**: `L=2`층 cross-attention(+FFN). Key는 `bev + bev_pos`, Value는 `bev`. 공간 편향 `M_{h,k}`(σ=[2,4,8,16]m)는 **고정 prior**로 유지한다.
- **헤드**: [`anchor_planner.py`](../navsim/agents/para_ssr/modules/anchor_planner.py)의 `im_reward`(softmax), `sim_reward`(sigmoid×5), `offset_head`와 같은 구조를 가진 가벼운 사본.
- **뱅크별 분리**: `P_det`는 BEVFusion, `P_map`은 ReSMap을 읽는다. 파라미터는 공유하지 않는다.
- **학습 loss**: `L_probe = anchor_plan_losses(...)`를 그대로 재사용한다 (`traj_offset_loss + im_reward_loss + sim_reward_loss`). 타깃은 Student 플래너가 이미 쓰는 `gt_trajectory`, `sim_reward`, `sim_valid`다.
- **선택 사항(ablation)**: 뱅크 특화. `P_map`은 `drivable_area_compliance` 채널, `P_det`는 충돌 계열 채널의 BCE 가중치를 높인다.

### 3.2 Student 쪽 사용
`student_proj_b(student_bev)`를 `P_b`에 통과시킨다. 이때 `torch.func.functional_call`에 `{name: p.detach()}`를 넘겨 파라미터의 gradient를 끊는다.

| 신호 | 정의 | gradient 대상 |
|---|---|---|
| Attention KD | `Σ_k w_k · KL(A_s,k ‖ A_t,k)`. 마지막 층(필요시 전 층)의 헤드별 attention. | Student BEV, `student_proj` |
| Feature KD | anchor token `h_s`와 `h_t`의 hybrid(L2+cosine). 층별 token도 선택 가능. | Student BEV, `student_proj` |
| Task grounding | `L_task^S = anchor_plan_losses(P_b(student BEV))` | Student BEV, `student_proj` |

### 3.3 학습 스케줄
1. **Warm-up** (예: 첫 1 epoch): `L_probe`만으로 probe 학습. 증류 가중치는 0.
2. **Ramp** (예: 다음 1~2 epoch): 증류 및 `L_task^S` 가중치를 0에서 목표값까지 선형 증가.
3. **본 학습**: probe는 `L_probe`로 계속 학습한다. Teacher 타깃이 흔들리지 않게 probe의 **EMA 사본**(decay 0.999)을 Teacher/Student 양쪽의 증류에 쓰는 옵션을 둔다.

> [!NOTE]
> Warm-up 중에는 probe가 아직 부정확하므로, 이 단계의 Student 증류 신호는 신뢰할 수 없다. 그래서 가중치를 0에서 시작한다.

### 3.4 앵커 가중치 재보정 (v8 `N_eff` 문제)
v8 로그에서 `N_eff≈140`이었다. 하이브리드 `w_k`가 의도(20~60)보다 평평하다는 뜻이다.
- 후보: `τ_gt: 5.0→2.5`, `λ_cov: 0.2→0.1`, `λ_gt: 0.6→0.7`.
- 확정값은 코드로 정하지 말고, 학습 샘플 일부에서 `N_eff` 분포를 계산하는 스크립트로 격자 탐색해서 목표 구간(예: 30~80)에 맞춘다.
- 선택: GT 궤적을 257번째 query로 추가해 "GT가 어떤 앵커와도 멀 때" 생기는 불일치를 줄인다. (v8은 raw 앵커만 query로 쓴다.)

---

## 4. 구현 위치 (예정)

| 파일 | 변경 |
|---|---|
| `navsim/agents/para_ssr/distill/planning_probe.py` (신규) | `PlanningGroundedProbe` 모듈, `functional_call` 기반 stop-grad 호출, EMA 사본 |
| [`distill/anchor_distill.py`](../navsim/agents/para_ssr/distill/anchor_distill.py) | `version="v9"` 분기: 랜덤 probe 대신 `P_b` 사용, `L_probe`/`L_task^S` 반환, ramp 가중치 |
| [`para_ssr_agent.py`](../navsim/agents/para_ssr/para_ssr_agent.py) | `get_optimizers`에서 probe 파라미터를 옵티마이저에 등록하고, v8의 "probe 동결" 블록은 v9에서 제외 |
| [`distill/distillation.py`](../navsim/agents/para_ssr/distill/distillation.py) | `gt_trajectory`, `sim_reward`, `sim_valid`를 probe loss까지 전달 |
| `configs/default.py`, `configs/agent/para_ssr_selector_v9_agent.yaml` (신규) | v9 하이퍼파라미터 |
| `scripts/training/run_bev_selector_v9_distill.sh` (신규) | 런처 |

주의할 점:
- v8과 달리 증류 모듈 안에 **학습되는 probe 파라미터**가 생기므로, `P_b`는 옵티마이저 그룹에 반드시 포함시킨다 (v7의 "옵티마이저 누락" 버그 재발 방지). 학습 시작 직후 `requires_grad`와 옵티마이저 등록 파라미터 수를 로그로 출력해 확인한다.
- Teacher BEV는 캐시(`teacher_store`)에서 읽으므로 추가 Teacher 추론 비용은 없다.
- 연산량: `[B, 4, 256, HW]` attention을 뱅크 2개 × 층 2개 × (Teacher + Student)로 계산한다. v8 대비 증가하므로 메모리와 속도를 먼저 측정한다.

---

## 5. 손실 요약

```
L_total = L_plan(Student 본 플래너)
        + L_probe_det + L_probe_map                         # probe 학습 (Teacher BEV, planning)
        + ramp(t) · 0.5 · Σ_b s_b · [ λ_f · L_featKD,b
                                     + λ_a · L_attnKD,b
                                     + λ_t · L_task^S,b ]    # Student 증류 (probe stop-grad)
```
- `s_b`: 도메인 스케일 (v8의 `resmap_scale` 계승, 재검토 대상)
- v8에서 고친 중복 합산 버그를 반복하지 않도록, 텔레메트리용 항은 `metrics`로만 기록하고 `losses`에는 넣지 않는다.

---

## 6. 진단 지표 (W&B)

| 지표 | 확인할 것 |
|---|---|
| `probe/{det,map}/im_reward_loss`, `sim_reward_loss`, `offset_loss` (Teacher BEV) | probe가 Teacher BEV에서 planning을 풀 수 있는지. 못 풀면 Teacher KD는 의미가 없다. |
| `probe/{det,map}/student_*` (Student BEV, 동일 loss) | Student가 Teacher와의 격차를 줄이는지. |
| `probe/attn_entropy`, 헤드별 | 편향만으로 결정되는지(엔트로피가 편향 prior와 같음) 여부. |
| `distill/effective_anchors` | §3.4 보정 후 목표 구간. |
| attention 히트맵 (주기적 저장) | 차량, 차선, 연석 등 의미 영역에 걸리는지. |
| `distill_cos/*`, `loss_attn_*` | 포화 속도. |

---

## 7. 위험과 한계

1. **probe가 편향만 쓸 가능성**: planning 신호가 있어도 `M_{h,k}`만으로 풀 수 있으면 content attention이 학습되지 않는다. 엔트로피와 히트맵으로 확인하고, 필요하면 편향 σ를 키우거나 헤드 일부의 편향을 제거한다.
2. **probe 과적합**: 학습 로그에서만 probe를 학습하므로 Teacher BEV에 과적합할 수 있다. 현재 ReSMap 캐시가 train 로그에만 있어 **validation이 비어 있다** (v8 로그에서 확인). train 로그의 일부를 hold-out으로 분리해 probe와 Student 검증에 쓴다.
3. **동시 학습 불안정**: probe와 Student가 함께 학습되면 Teacher 타깃이 움직인다. warm-up, ramp, EMA로 완화한다.
4. **Student 플래너와의 역할 중복**: `L_task^S`는 Student BEV에 planning loss를 한 번 더 거는 것이므로 본 플래너 loss와 상관이 크다. 이 항이 이득인지 아니면 단순히 학습 신호를 중복하는지는 ablation으로 분리한다.
5. **비용**: 연산 및 메모리 증가.
6. **Teacher 품질 가정**: probe가 읽는 것은 Teacher의 BEV feature이지 Teacher의 의사결정이 아니다. Teacher BEV가 planning에 유용하다는 가정은 별도 확인이 필요하다 (§6 첫 행).

---

## 8. 실험 계획 (ablation)

| ID | 설정 | 목적 |
|---|---|---|
| A | v8 (현재) | 기준선 |
| B | v8 + `w_k` 재보정(§3.4)만 | 가중치 효과 분리 |
| C | probe를 `L_probe`로 학습, 증류는 Attention/Feature KD만 | PGP 자체의 효과 |
| D | C + `L_task^S` | task grounding 추가 효과 |
| E | D + EMA probe | 안정성 |
| F | (대조) probe를 열고 증류 loss만으로 학습 | collapse 재현 확인 |
| G | (대조) Attention KD 제거, Feature KD만 | attention KD 기여 분리 |

비교 지표: hold-out PDMS 및 세부 항목(NC, DAC, EP, TTC, Comf), `distill/*`, `probe/*`. 시드 간 분산을 보기 위해 최소 2시드를 권장한다.

---

## 9. 오픈 이슈
- probe 층 수 `L`과 뱅크별 특화 가중치(§3.1)의 기본값
- `resmap_scale`(v8: 3.0)을 probe 도입 후에도 유지할지
- GT 257번째 query 도입 여부
- 증류에서 refined 궤적(`anchor + offset`)을 query로 쓰는 방안: v8은 raw 앵커만 사용한다.
- 히트맵 시각화 스크립트 (현재 코드에 없음)
