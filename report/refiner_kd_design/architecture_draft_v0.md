# 교정기(refiner) 기반 teacher KD: 상세 아키텍처 설계 (구현 전)

작성 2026-09-28. 상태는 설계안이고 코드는 아직 없습니다. 근거는 이번 사실 조사 4건입니다: 미래 물체 GT, 대리 손실 검증 pilot, 입력 feature, 초안·출력 파라미터화. 여기서 확인되지 않은 수치나 선택은 **[가정]**으로 표시했습니다. 최종 선택은 §5 결정표에서 사용자가 정합니다.

---

## 용어 (처음 한 번만 정의)

| 용어 | 이 문서에서의 뜻 |
|---|---|
| 초안 τ0 | 교정 전 궤적. pose 8개(0.5 s 간격, 4 s), 각 pose는 (x, y, heading) |
| 교정 Δτ | refiner가 내는 수정량. 종방향(속도 프로필) 제어점 6개와 횡방향(옆 이동) 제어점 6개 |
| 교정 궤적 τ1 | τ1 = decode(τ0, Δτ). decode는 M4의 디코더 |
| refiner(교정기) | τ0와 장면 feature를 읽고 Δτ와 "고칠지 여부"를 내는 작은 네트워크 |
| R_T / R_S / R_GT | refiner 구조는 같고 입력 feature 출처만 다름: teacher / student / GT raster |
| 교란 초안 | 사람(GT) 궤적을 일부러 바꿔 만든 학습용 초안 |
| 실제 초안 | student(PARA-SSR interaction_final seed 0)가 실제로 낸 궤적 |
| 초안 은행 | token마다 미리 만들어 공식 채점까지 해 둔 고정 초안 묶음 |
| 기준 경로 Γ | 원점과 초안 8점을 잇는 꺾은선 |
| 호장 s | Γ를 따라 원점에서 잰 거리 |
| Frenet 옆 거리 d | Γ에 수직으로 잰 옆 거리. 왼쪽이 + |
| 통로(corridor) | Γ를 따라 앞으로, 좌우로 일정 폭을 가진 띠. refiner가 feature를 읽는 곳 |
| B-spline 제어점 | 곡선을 만드는 몇 개의 숫자. 곡선은 제어점들이 만드는 범위(볼록 껍질) 밖으로 나가지 않으므로, 제어점에 한계를 걸면 곡선 전체에 한계가 걸림 |
| 대리 손실(surrogate) | 공식 채점(NC·DAC·EP·TTC·comfort)을 흉내 내는 미분 가능한 학습용 손실. 채점 그 자체가 아님 |
| SDF | 부호 거리장. 도로 경계까지의 거리이며 도로 안이 +, 밖이 − |
| footprint | 차체 사각형. Pacifica 5.176 × 2.297 m, rear axle에서 중심까지 1.461 m |
| LQR 추종 | 공식 채점기가 8 pose를 0.1 s 참조로 보간하고 제어기로 따라가게 하는 과정. 판정은 추종된 궤적으로 함 |
| keyframe | log 주석이 있는 0.5 s 간격 프레임 |
| gate | "이 초안을 고칠 필요가 있나"를 나타내는 출력 확률 p_g |
| sg(·) | stopgrad. 역전파를 막는 연산 |
| TRAIN-IN / TRAIN-OUT | navtrain 중 student 학습에 쓴 log(train_logs, 85,109장면) / 쓰지 않은 log(val_logs 214개, 18,179장면) |
| 해소 / 새 실패 / 순해소 | 실패→통과 / 통과→실패 / (해소 − 새 실패) |
| 불필요한 수정 | 원래 NC·DAC·TTC를 모두 통과하던 초안을 바꿨는데 PDMS가 오르지 않은 경우 |
| 동일 진행 대조군 | 정보 없이 모든 궤적을 똑같이 느리게 해서 진행 손실만 맞춘 비교군(32-C) |

### 사용자 피드백 대응표

| 피드백 | 반영 위치 |
|---|---|
| 초안과 출력 자유도 일치 (감속 전용 refiner에 ×0.6 초안을 주는 문제) | M1 "자유도 일치 규칙", M4 모드 A/B, §5 D1–D2 |
| ×1.4에 4 s 밖 경로가 필요함: 유효 구간과 생성 방법 | M1 "생성 방법·유효 구간" (균일 재표본화 금지, 가산 속도 오프셋, log 경로 한도) |
| 현재 ego 위치·방향과의 연결 유지, 안전 초안 비율 | M1 연속성 조건, 안전 초안 비율 / M4 경계 조건 |
| 실제 초안 평가가 주, 교란 초안 평가는 분석용 | M9, §4 판정 규칙 (교란 초안 결과는 판정에 넣지 않음) |
| `_compute_agent_targets` 범위와 누락을 "없음"으로 처리하지 않기 | M6 (쓰지 않고 새 빌더를 만듦, validity 3단계) |
| 원 근사와 SDF는 근사이므로 footprint 반영 정도를 확인, 최종은 공식 채점 | M7 (box-box 거리, 0.1 s), M8 (쌍 단위 검증), M9 |
| R_GT 입력 구분 (현재 GT = 인지 비교, 미래 GT = oracle) | M2, §4 (R_GT-cur는 기준점, R_GT-fut는 별도 참고) |
| 기존 작업과의 관계를 조심스럽게 | §6 |

---

## 0. 목적과 범위

### 0-1. 두 단계

- **단계 F (feasibility, 먼저. E2E 학습 없음)**
  - 고정된 feature 위에서 refiner를 같은 구조·같은 데이터·같은 초안으로 학습합니다.
  - 질문은 하나입니다: **"teacher feature를 읽는 R_T가 student feature를 읽는 R_S보다, 실제 student 초안의 실패를 같은 진행 손실에서 더 많이 고치고 불필요한 수정은 더 적게 하는가?"**
  - R_GT-cur, R_GT-fut, 규칙 필터, 정보 없는 대조군은 해석용 기준점입니다.
  - 이 질문에 "예"가 나올 때만 단계 E로 갑니다.
- **단계 E (나중)**
  - PARA-SSR을 **처음부터** 학습하면서 student refiner를 함께 학습합니다.
  - 고정된 R_T가 student 자신의 초안에 대해 낸 Δτ_T를 KD target으로 씁니다.
- **추론:** student와 student refiner만 씁니다. teacher와 GT는 필요 없습니다.

### 0-2. 이 문서가 고정하는 것

1. 교란 초안과 refiner 출력은 **같은 디코더(M4)**를 공유합니다. 그래서 교란으로 만든 모든 변화는 refiner의 표현 범위 안에 있습니다.
2. 균일 시간 재표본화(×배율)는 쓰지 않습니다. 모든 초안과 교정은 t=0에서 위치·방향·속도·가속도가 이어집니다.
3. 충돌 대리 손실의 물체 출처는 **log 주석 전체**(360°, 0–5 s, 80 m)입니다. `_compute_agent_targets`는 쓰지 않습니다. 주석이 닿지 않는 곳("unknown")은 장애물 없음으로 처리하지 않습니다.
4. 주 평가는 **navtest 실제 초안 12,146개의 공식 채점**입니다. 교란 초안 평가는 교정 범위 분석에만 씁니다.
5. R_GT는 두 가지로 나눕니다. **현재 GT raster**(t0 스냅샷, 완벽한 현재 인지의 기준점)와 **미래 GT raster**(실제 미래를 본 oracle, 별도 참고)입니다.
6. R_T, R_S, R_GT, R_none은 입력 어댑터(M2)의 첫 층만 다르고 나머지는 코드와 하이퍼파라미터가 같습니다.
7. navtest는 어떤 선택(가중치, 임계값, 대리 손실 설정)에도 쓰지 않습니다.

### 0-3. 열어 두는 것

§5 결정표에 있습니다: 가속 허용 여부, 횡방향 범위, heading 규약, teacher 박스 입력, R_S의 h_final, singleton 처리, gate 라벨, loss 가중, 추종 근사, E2E의 stopgrad·KD 게이팅·일정.

---

## 1. 전체 구조도

### (a) 단계 F: feasibility

```
[오프라인 준비: 1회]
 raw log anns ──────M6──► 미래 물체 track (0–5 s, 360°, 80 m, validity 3단계)      [token당 약 30 KB]
 metric cache ──────────► DAC 영역 SDF (0.25 m, 전방 −8..72 m) · 경로 중심선 · PDM-Closed 진행량
 student ckpt ──D1 dump─► student BEV [256,50,100] f16 · h_final [256] · 실제 초안 τ0 [8,3]
 teacher cache ─────────► teacher BEV [256,50,100] f16 (좌우 뒤집어 student 배치로)
 GT ────────────────────► 현재-GT raster [12,50,100] · 미래-GT raster [19,50,100]
 사람 경로 + log ────M1──► 초안 은행 (token당 13개) ──공식 채점──► 초안별 NC/DAC/TTC/EP 라벨

[학습: arm마다 같은 코드, 입력 출처만 다름]
 출처 feature ─M2 어댑터─► [64,50,100] ─┬─M3 통로 샘플(τ0 따라)─► [70,48,17] ─┐
                                         └─전역 pool─► 200 token ─────────────┤
 τ0 + ego 상태 + command ─────────────────► 초안 token ─────────────────────┤
                                                                             ▼
                                                     M5 refiner (transformer 4층)
                                                       ├─ gate p_g
                                                       ├─ 종방향 제어점 6
                                                       └─ 횡방향 제어점 6
                                                                             ▼
                                                     M4 디코더 ─► τ1 (8 pose + 0.1 s 41점)
 손실 = M7 대리 손실(τ1; M6 물체, SDF, 진행, 편안함, 수정량) + gate BCE(공식 라벨)

[검증과 평가]
 M8: 대리 손실 ↔ 공식 판정 일치도 (학습 전에 먼저)
 M9: navtest 실제 초안 → 각 refiner → τ1 → 공식 PDM 채점 → 순해소·새 실패·PDMS·진행 손실
```

### (b) 단계 E: E2E

```
카메라 ─► PARA-SSR (처음부터 학습) ─► bev_embed [5000,256], h_final [256], τ0 [8,3]
                                                 │
     sg(τ0) ─┬─► R_S^E2E (student BEV (+h_final)) ─► p_S, c_S ─M4─► τ_final
             └─► R_T (고정, teacher BEV 캐시)     ─► p_T, c_T   (KD target)
손실 = 기존 PARA-SSR 손실 (τ0 모방 + det + motion + map, GradBalancer)
     + λ_KD(epoch) · m_KD · [ |c_S − sg(c_T)|₁ + BCE(p_S, sg(p_T)) ]
     + λ_sur · M7 대리 손실(τ_final)                         (선택)
기울기: R_S → student BEV (배율 γ_bev), τ0로는 막음(기본값)
```

### (c) 추론

```
카메라 ─► PARA-SSR ─► τ0, bev_embed (, h_final) ─► R_S^E2E ─► p_S ≥ θ ? decode(τ0, Δτ_S) : decode(τ0, 0) ─► 제출
```

teacher, GT, metric cache는 필요 없습니다. 추가 파라미터는 약 3.4M [가정]입니다.

---

## 2. 모듈별 명세

### 공통 규약

| 항목 | 정의 |
|---|---|
| **N 좌표** (NAVSIM ego, t0) | 원점은 rear axle, x 전방, y 왼쪽, heading 0 = 전방, 반시계 방향이 +. 초안, 물체, SDF, 디코더가 모두 이 좌표를 씀. metric cache의 rear_axle과 일치 확인 |
| **S 격자** (student BEV 배치) | [C, 50, 100]. 행 r은 x = (r+0.5)·0.64 m (0–32 m), 열 c는 y_left = 32 − (c+0.5)·0.64 m (열 0이 왼쪽 32 m). 평탄 index = r·100 + c |
| teacher 캐시 → S 격자 | 캐시는 (C, H=x, W=y_left)이고 열 j는 y_left = −32 + (j+0.5)·0.64. 변환은 `teacher[:, :, ::-1]` (heatmap으로 검증됨) |
| **E 격자** (확장 SDF) | 행은 x −8..72 m, 열은 y_left +32..−32, 0.25 m 간격 → [320, 256] |
| 시간 격자 | 초안 keyframe k=0..8 (0.5 s). 조밀 격자 n=0..40 (0.1 s, 채점기 참조와 같음). 물체 0..50 (0.1 s, 5 s) |

---

### M1. 초안 생성기

**역할:** 학습·분석용 초안 은행을 만듭니다. 출처는 두 가지입니다: (i) 사람 궤적 교란, (ii) 실제 student 초안.

**입력**
- 사람 경로: scene 미래 10프레임(5 s, 모든 token에 있음)과 raw log 경로(최대 8 s). 각 점은 (x, y, h), N 좌표, f64.
- ego 상태 v0, a0 (metric cache `ego_state`).
- 경로 연장 출처는 다음 순서로 씁니다: scene → log(6 s 98.0%, 8 s 96.7%, navtrain 표본 기준) → metric cache 경로 중심선 `centerline`.
- 실제 초안: student dump의 `trajectory` [8,3].

**출력 (token당)**

| 항목 | 형태 |
|---|---|
| `drafts` | [K, 8, 3] f32 |
| `family` | [K] int. identity / L-small / L-const / L-ignore-brake / L-creep / Lat-small / Lat / combined / student |
| `c_lon_pert`, `c_lat_pert` | [K, 6] f32 |
| `A_p`, `D_p`, `t_on` | [K] |
| flags | `ext_src` (scene/log/centerline), `ext_m`, `cont_ok`, `frame_gap` |

K = 13 [가정].

**생성 방법 (균일 시간 재표본화 금지)**

모든 교란은 **M4 디코더**로 만듭니다: τ_pert = decode(Γ_human, c_lon_pert, c_lat_pert). refiner와 같은 함수이고 경계 조건도 같습니다. 균일 ×f를 쓰지 않는 이유는 t=0에서 속도가 f·v0로 튀기 때문입니다. ×1.4에서는 token의 31%가 2.4 m/s 넘게 튀고, ×0.6도 같은 크기로 튑니다.

| family | 정의 | 범위 [가정, pilot으로 조정] |
|---|---|---|
| L-const | 가산 가속 오프셋 δa를 t_on부터 jerk ≤ 2 m/s³로 올려 A_p로 유지 | A_p ~ U[0.2, 1.3] m/s², t_on ∈ {0, 0.5, 1.0, 1.5, 2.0} s |
| L-ignore-brake | 사람이 한 감속의 일부를 무시: v_pert(t) = v_h(t) + α·max(0, v_h(0) − v_h(t)) | α ~ U[0.3, 1.0]. 사람 감속 ≥ 1 m/s인 token만 |
| L-creep | 사람이 거의 정지(s4 < 2 m)했는데 앞으로 나가는 초안. 경로 중심선을 따름 | δa ~ U[0.3, 1.0]. 실제 NC 중 이 형태 19건 |
| L-small | 고칠 필요가 없는 작은 교란 | 모드 A: A_p ~ U[0, 0.2]. 모드 B: U[−0.2, 0.2] |
| L-slow (**모드 B만**) | 사람보다 느린 초안 | A_p ~ U[−0.8, −0.2] |
| Lat | 끝 옆 거리 D_p까지 부드러운 계단. 시작은 s_on ~ U[0.2, 0.6]·S_end | \|D_p\| ~ U[0.3, 1.5] m, 좌우 대칭 |
| Lat-small | 작은 옆 이동 | \|D_p\| ≤ 0.3 m |
| combined | L-const와 Lat을 함께 | 위 범위 |

- 범위의 근거:
  - 실제 student–사람 차이를 등가 가속으로 환산하면 p1–p99가 −0.59..+0.65 m/s²이고 99.7%가 ±1 안입니다.
  - NC 실패의 p95는 +0.91, p99는 +1.27입니다.
  - |d4|는 97.2%가 2 m 이내입니다.
- ×배율과의 대응 (참고): 일정한 δa에서 Δs(4) ≈ 8·A_p m입니다.

  | 조건 | A_p = 0.5 | A_p = 1.0 | A_p = 1.3 |
  |---|---|---|---|
  | 중앙 속도 (s4 16.3 m) | ≈ ×1.25 | ≈ ×1.49 | — |
  | 고속 p95 (s4 39.5 m) | — | — | ≈ ×1.26 |

  따라서 "×1.4"는 중저속에서만 나오는 값입니다. 실제 오차가 가산형이므로 범위는 가산 단위로 정합니다.

**유효 구간 (이를 벗어나면 줄이거나 버림)**

1. **경로 가용:** s_pert(4) ≤ S_avail이어야 합니다. S_avail은 scene 5 s 또는 log 최대 8 s의 사람 경로 길이입니다.
   - 넘으면 A_p를 줄여 맞추고, 그래도 안 되면 버립니다.
   - 학습 초안에는 **외삽을 쓰지 않습니다**. L-creep만 경로 중심선을 씁니다.
   - ×1.4 상당(5.6 s 경로)은 약 98–99.6% token에서 가능합니다.
2. **t=0 연속:** 구조상 Δv(0)=0, δa(0)=0, d(0)=d′(0)=0입니다. 추가로 첫 구간 속도 − v0 ∈ [−0.8, +0.6] m/s를 검사합니다 [가정]. 실제 student는 p5–p95가 −0.57..+0.38입니다.
3. **운동 한계 (0.1 s 격자):**
   - 종가속 ∈ [−4.05, +2.40] m/s²
   - |κ| ≤ 0.213 /m (사람 p99.9)
   - 횡가속 ≤ 3.8 m/s² (사람 p99.9)
4. **프레임 누락 token**(5 s 창 안에 1.0 s 간격이 있는 token)은 교란하지 않습니다. navtrain 1.34%, navtest 0.37%입니다.

**자유도 일치 규칙 (M4와의 계약)**
- 교란의 범위는 **refiner 출력 범위 안에 있어야** 합니다. 즉 사람 궤적(안전함이 확인된 궤적)이 어떤 교란 초안에서든 refiner로 도달 가능해야 합니다.
  - 모드 A(감속 전용, 기본값): 빠르게 만드는 종방향 교란만 씁니다. refiner Δv ≤ 0으로 되돌릴 수 있습니다. 느린 교란은 만들지 않습니다.
  - 모드 B(제한 가속): L-slow를 허용하되, 최저값 −0.8 m/s²가 refiner의 가속 상한(ΔV_ACC)으로 되돌릴 수 있는 범위여야 합니다.
  - 횡방향: |D_p| ≤ 1.5 m < refiner D_MAX 2.0 m.
  - 횡방향 역변환은 Γ가 달라서 근사입니다. 오차는 V3에서 잽니다.
- 학습 target이 "사람으로 복귀"는 아닙니다. 대리 손실(M7)이 결정하므로 빠른 교란 초안도 안전하면 그대로 두는 것이 정답입니다.

**안전 초안 비율**
- token당 구성 [가정]: identity 1, 작은 교란 3, 큰 교란 8, 실제 student 1 = 13개.
- 배치 샘플링 가중 [가정]: identity와 작은 교란 30%, 큰 교란 45%, 실제 초안 25%.
- 은행 전체의 공식 실패율은 20–35%를 목표로 하고 pilot(V5)에서 A_p 범위로 조정합니다.
- identity(=사람)가 안전하다는 근거: navtest에서 사람 궤적의 NC/DAC/TTC 실패는 0, navtrain E 9,000개에서 NC/DAC 실패는 0입니다.

**구현 위치**
- `/home/external-user/yongjae/SSR/tools/refiner/make_draft_bank.py`
- 디코더는 `navsim/agents/para_ssr/refiner/decoder.py`를 가져다 씁니다.

**재사용**
- scratchpad의 `c_navtrain_paths.py`, `d_navtest_paths.py`(log 경로 추출), `e_decode_check.py`
- `E_train_split_feasibility/e1_dump.py`의 navtrain 경로
- `sf_common.py` `Path`

---

### M2. 입력 어댑터

**역할:** 출처별 입력을 공통 격자·공통 채널 공간 [64, 50, 100](S 격자)으로 옮깁니다. **arm 사이에 다른 것은 이 모듈의 입력 채널 수와 정규화뿐입니다.**

| 어댑터 | 입력 (S 격자로 정렬 후) | 정규화 | 층 | 파라미터 |
|---|---|---|---|---|
| A_T (teacher BEV) | npz `bev_feature` [256,50,100] f16 → `[:, :, ::-1]` | 칸별 LayerNorm(256, affine 없음). ReLU 출력이라 55.6%가 0 | 1×1 conv 256→128, GELU, 1×1 128→64 | 약 41k |
| A_S (student BEV) | `bev_embed` [5000,256] → [256,50,100] | 같은 칸별 LN (이미 LN 출력이지만 규약을 통일) | 같은 구조, 가중치는 별도 | 약 41k |
| A_T+box (선택) | A_T 입력 + teacher 박스 raster 7채널 → [263,…] | LN은 256 부분에만 | 1×1 263→128→64 | 약 42k |
| A_box (T/S/GT 공통, 선택) | 박스 raster 7채널: 점유(vehicle / VRU / static), (vx, vy)/10, (cos, sin) | 없음 | 1×1 7→128→64 | 약 9k |
| A_GTc (현재 GT) | [12,50,100]: 박스 raster 7채널 + DAC 영역 mask + SDF(±8 m, /8) + 차선 중심선 + 보도 + 횡단보도 | 없음 | 1×1 12→128→64 | 약 10k |
| A_GTf (미래 GT, oracle) | [19,50,100]: agent/static 점유 × k=1..8 (16) + DAC mask + SDF + 차선 중심선 | 없음 | 1×1 19→128→64 | 약 11k |
| A_none (정보 없음) | 없음. 출력은 0 | – | – | 0 |

**세부 규칙**
- **범위:** 모든 arm은 같은 전방 0–32 m, 좌우 ±32 m 격자를 씁니다. 현재·미래 GT raster도 이 범위로 잘라 공정하게 비교합니다. 확장 격자 R_GT-fut는 별도 참고 arm입니다(§4).
- **경로 중심선(route):** 어떤 arm에도 넣지 않습니다. student와 teacher는 route를 모르고 command만 압니다.
- **박스 raster 그리기:** footprint 다각형을 4배 세밀한 격자에 `fillPoly`로 그린 뒤 평균 내어 비율 점유를 만듭니다. 반 셀 바깥쪽 편향을 피하기 위해서입니다.
- **현재 GT 박스:** t0 log 주석 전체(모든 class)를 쓰고 격자 안만 남깁니다.
- **teacher 박스:** `pred_boxes_3d`에서 점수 ≥ 0.3 [가정]인 것을 `teacher_box_to_navsim`으로 변환합니다.
  - x, y, yaw는 검증됐고 z 규약은 미검증입니다.
  - 예전 기록에 이 도구의 90° yaw/크기 오류가 있으므로 V1에서 heading과 L/W를 다시 확인합니다 [확인 필요].
- **student 박스:** `all_bbox_preds[-1]`를 `denormalize_bbox`로 풀고 점수 ≥ 0.3 [가정]인 것을 씁니다.
- **미래 GT raster:** M6 track을 k=1..8 keyframe의 실제 자세로 그립니다. agent와 static을 나누는 이유는 NC 감점이 0과 0.5로 다르기 때문입니다.

**구현 위치**
- `navsim/agents/para_ssr/refiner/adapters.py`
- raster 생성: `tools/refiner/build_gt_rasters.py`

**재사용**
- kyungmin `plan_map.drivable_polygon`, `rasterize_sdf` (또는 metric cache `drivable_area_map`)
- `build_map_rasters.py`의 보도·횡단보도 층
- `transfuser_features.py`의 raster 규약 참고

---

### M3. 통로 샘플러

**역할:** 초안 경로를 따라 feature를 읽어 refiner가 "경로 위와 옆에 무엇이 있고 언제 도착하는지"를 보게 합니다.

**입력**
- 어댑터 출력 F [B, 64, 50, 100]
- τ0 [B, 8, 3]
- 초안 시간표 s0(t) (M4에서 계산)

**출력**
- 통로 텐서 X_c [B, 70, 48, 17]: 64 feature 채널 + 6 기하 채널
- 전역 token G [B, 200, 64]

**통로 점 만들기**
- **정거장:** Γ를 따라 N_s = 48개를 둡니다.
  - 간격 Δs = max(1.0 m, S_look/48).
  - S_look = S_8 + max(8 m, v_end·1 s). 끝 너머를 보는 이유는 TTC 투영(최대 +0.9 s) 때문입니다.
  - 끝 너머의 Γ는 등곡률로 외삽합니다. 거의 정지(S_8 < 2 m)이면 현재 heading 방향 직선으로 둡니다. 모두 flag를 답니다.
- **옆 방향:** d_i = −4.8..+4.8 m, 0.6 m 간격으로 17개 [가정]입니다. 교정 범위 ±2 m에 차체 반폭 1.15 m와 옆 차선을 더한 폭입니다.
- **점 위치:** q_ij = Γ(s_j) + d_i·n(s_j). n은 왼쪽 법선이며, 꼭짓점에서는 인접 두 구간 법선을 호장으로 선형 혼합해 불연속을 없앱니다.
- **격자 좌표 변환:** r = x/0.64 − 0.5, c = (32 − y)/0.64 − 0.5. `grid_sample`(bilinear, `padding_mode="zeros"`, `align_corners=False`)로 읽습니다.
- **격자 밖 점:** feature 0과 **mask 0**을 줍니다. kyungmin `sample_field`의 border padding처럼 경계값을 복사하지 않습니다. 격자 밖은 "없음"이 아니라 "안 보임"이기 때문입니다.

**기하 채널 6개**

| 채널 | 의미 |
|---|---|
| in_grid mask | 점이 50×100 격자 안인가 |
| path_valid mask | 정거장이 초안 끝(S_8) 안인가. 밖이면 외삽 구간 |
| s_j / 48 m | 정거장 호장 |
| d_i / 4.8 m | 옆 거리 |
| t_d(s_j) / 4 s | **초안이 그 정거장에 도착하는 시각**. 4 s 이후는 4로 자르고 path_valid로 구분 |
| v_d(s_j) / 15 m/s | 그 정거장에서의 초안 속도 |

**시간 정보**
- 현재 시점 feature(teacher, student, 현재 GT)에는 시간 축이 없습니다. 도착 시각 채널이 "언제 거기 있을지"를 알려 줍니다.
- 미래 GT arm은 8개 시각 평면이 채널로 접혀 들어가 있습니다. 네트워크가 도착 시각 채널과 비교해 씁니다. 그래서 구조를 바꿀 필요가 없습니다.

**전역 token:** F를 5×5 평균 pool해 [64, 10, 20] = 200 token(3.2 m 칸)을 만들고 학습되는 위치 임베딩을 더합니다. 옆에서 끼어드는 물체처럼 통로 밖에 있는 것을 보기 위해서입니다.

**구현 위치:** `navsim/agents/para_ssr/refiner/corridor.py`

**재사용:** `sf_common.Path`의 호장·보간 규약. 채점기와 같은 선형 보간입니다.

---

### M4. 교정 파라미터화와 디코더

**역할:** refiner 출력(제어점)과 초안 τ0를 8 pose 궤적 τ1으로 바꾸고, 대리 손실용 0.1 s 41점도 함께 냅니다. 구조상 제약을 보장하는 결정적 함수이며 학습 파라미터는 없습니다. M1도 같은 함수로 교란을 만듭니다.

**기준 경로와 초안 시간표**
- 꼭짓점은 P_0 = (0,0), P_k = τ0[k, :2]입니다.
- 구간 길이 ℓ_k, 누적 호장 S_k.
- 초안 시간표 s0(t)는 (t_k = 0.5k, S_k)를 잇는 꺾은선입니다. 구간 안에서는 등속이며, 채점기의 선형 보간과 같습니다.

**종방향: 경로 위 시간표 다시 매기기**

time-warp를 곱셈 배율이 아니라 **속도 오프셋**으로 표현합니다.

- Δv(t) = Σ_i c_i·B_i(t). 구간 [0, 4 s]의 clamped cubic B-spline이고 제어점 8개(0.8 s 간격 5구간)입니다.
- **c_0 = c_1 = 0** → Δv(0) = 0, δa(0) = 0. t=0에서 속도와 가속도가 초안과 이어집니다.
- 자유 제어점 6개는 누적 증분으로 만듭니다: c_i = clip(c_{i−1} + h·u_i, L_lo, L_hi), h = 0.8 s.
  - u_i = A_DEC·tanh(z_i) (z_i < 0일 때), A_UP·tanh(z_i) (z_i ≥ 0일 때).
  - 따라서 |δa| ≲ max(A_DEC, A_UP)입니다. clamped spline 끝부분 계수는 V3에서 확인합니다.
- v1(t) = max(0, v0(t) + Δv(t)). 학습 때는 softplus(β = 0.3) [가정]로 부드럽게 합니다. s1(t) = ∫v1을 0.1 s 사다리꼴 적분으로 구합니다.

| 모드 | L_lo | L_hi | A_DEC | A_UP | 보장되는 성질 |
|---|---|---|---|---|---|
| **A 감속 전용 (기본값 권장)** | −20 m/s | **0** | 4.0 | 2.0 (초안 속도까지 재가속만) | 모든 t에서 s1(t) ≤ s0(t). **경로 연장이 필요 없음** |
| B 제한 가속 | −20 m/s | +ΔV_ACC = 2.0 m/s [가정] | 4.0 | 2.0 | s1(4)가 S_8을 넘을 수 있음 → 아래 연장 규칙 |

**경로 연장 규칙 (모드 B만)**
- 마지막 두 구간의 heading 변화로 κ_end를 잡고, min(0.95/v, 4.89/v², 0.213)로 자릅니다.
- 등곡률로 외삽합니다. 오차는 +1 s에서 p90 0.28 m, +2 s에서 1.09 m입니다.
- 연장 상한은 L_ext = min(5 m, v_end·1 s)입니다. s1(4) > S_8 + L_ext이면 Δv의 양수 부분을 비례로 줄입니다(hard).
- refiner 출력의 연장은 **항상 외삽**입니다. 학습과 추론을 맞추기 위해서입니다. `ext_m`을 기록합니다.

**횡방향: 호장의 함수인 Frenet 옆 거리**
- d(s) = Σ_i e_i·B_i(s/S_L). clamped cubic, 제어점 8개. S_L은 S_8(모드 A) 또는 S_8 + L_ext(모드 B).
- **e_0 = e_1 = 0** → d(0) = 0, d′(0) = 0. 현재 위치와 방향이 유지됩니다.
- e_i = D_MAX·tanh(w_i), D_MAX = 2.0 m [가정]. 볼록 껍질 성질로 |d| ≤ D_MAX가 보장됩니다.
- S_8 < 3 m [가정]이면 d ≡ 0입니다. 정지 근처에서 옆으로 움직이면 곡률이 발산하기 때문입니다.
- 곡률 κ_new ≈ (κ_Γ + d″)/(1 − κ_Γ·d)는 M7의 편안함 손실로 누르고, 추론 때 |κ_new| > min(0.95/v, 4.89/v², 0.213)이면 e를 비례로 줄입니다.

**위치와 heading**
- 위치: p(t) = Γ(s1(t)) + d(s1(t))·n(s1(t)). τ1[k] = (p(t_k), h_k), k = 1..8.
- **H1 (path 모드, 기본값 권장):** h_k = h0(s1(t_k)) + atan2(d′, 1 − κ_Γ·d).
  - h0(s)는 초안 heading을 호장으로 보간한 값(unwrap)입니다.
  - 교정 0이면 **τ1 = τ0가 정확히** 나옵니다.
- H2 (tangent 모드): 출력 점의 중앙차분. 0.5 m 미만 이동이면 h0를 유지합니다(`heading_from_path`). §5 D5 참고.

**0.1 s 41점:** [원점(0,0,0); τ1]을 시간에 선형 보간합니다(heading은 unwrap 후 선형). **채점기가 참조를 만드는 방식과 똑같습니다.** M7은 이 41점으로 계산합니다.

**hard 제약 요약**
- 원점과 초기 heading 고정
- Δv(0) = δa(0) = 0, d(0) = d′(0) = 0
- v ≥ 0 (호장 단조 증가)
- |d| ≤ D_MAX
- 모드 A는 s1 ≤ s0, 모드 B는 s1(4) ≤ S_8 + L_ext
- 교정 0이면 항등(H1)

**입출력 형태**
- 입력: τ0 [B,8,3], c_lon_raw z [B,6], c_lat_raw w [B,6], mode
- 출력: τ1 [B,8,3], dense [B,41,3], s1 [B,41], v1 [B,41], d [B,41], κ_new [B,41], flags

**구현 위치:** `navsim/agents/para_ssr/refiner/decoder.py`, `geometry.py` (Pacifica 상수, 꺾은선, 호장, 법선)

**재사용:** `sf_common.ref_poses_time`(0.1 s 보간 규약), kyungmin `heading_from_path`(H2), `e_decode_check.py`(항등·깎임 오차 검사)

---

### M5. refiner 네트워크 (R_T, R_S, R_GT, R_none 공용)

**역할:** 통로 텐서, 전역 token, 초안 정보를 읽어 gate와 제어점을 냅니다.

**입력**
- X_c [B,70,48,17], G [B,200,64]
- 초안 벡터 u_d [B,72]:
  - v0, a0, (vx, vy, ax, ay)
  - command one-hot 4
  - τ0 8 × (x/40, y/10, cos h, sin h)
  - 구간별 속도, 가속, 곡률 각 8
  - S_8/40
  - flag 2 (거의 정지, 외삽 사용)
- R_S+h 변형만 h_final [B,256]을 추가 token으로 받습니다.

**내부 구조**

| 부분 | 층 | 출력 | 파라미터 |
|---|---|---|---|
| 초안 token | MLP 72→192→192 | [B,1,192] | 약 51k |
| 통로 encoder | Conv3×3 70→96, GN, GELU; 96→96; 96→128 | [B,128,48,17] | 약 254k |
| 정거장 token | 옆 방향 펼치기 17×128=2176 → Linear 192 (옆 구조 보존) | [B,48,192] | 약 418k |
| 전역 token | Linear 64→192 + 위치 임베딩 | [B,200,192] | 약 50k |
| transformer | 4층, d=192, head 6, FFN 768. 각 층: [초안 token + 정거장 48] 자기 attention → 전역 200 cross-attention → FFN | – | 약 2.36M |
| gate head | 초안 token 출력 → MLP 192→128→1 | p_g | 약 25k |
| 종방향 head | [초안 token ‖ 정거장 평균] 384→256→6 | z (M4로) | 약 100k |
| 횡방향 head | 같은 구조 384→256→6 | w (M4로) | 약 100k |
| **합계** | | | **약 3.35M** [가정]. 작은 변형(d=128, 2층)은 약 0.9M |

**학습과 추론 규칙**
- 교정 head는 **모든 초안**에 대해 대리 손실로 학습합니다. 안전한 초안에서는 수정량 손실 때문에 Δ ≈ 0이 됩니다.
- gate는 공식 라벨 BCE로 **따로** 학습합니다. 교정 head로는 기울기가 가지 않습니다.
- 추론에서는 p_g ≥ θ일 때만 교정을 적용하고, 아니면 decode(τ0, 0)을 냅니다. θ는 inner-val에서 정합니다.
- arm 사이의 동일성: M3 이후의 코드, 하이퍼파라미터, 초기화 seed가 같아야 합니다. 어댑터 밖 파라미터 수가 같은지 단위 시험으로 확인합니다.

**학습 설정 [가정]**
- AdamW, lr 3e-4 cosine, wd 0.01, 40 epoch, fp16 autocast.
- 배치는 **token 단위**입니다. 8 token × 그 token의 초안 13개. feature 파일은 token당 한 번만 읽습니다.
- early stop 기준은 inner-val 총손실입니다.
- arm마다 seed 3개입니다.

**구현 위치:** `navsim/agents/para_ssr/refiner/refiner_net.py`, 학습 스크립트 `report/refiner_feasibility/train_refiner.py`

**재사용:** `modules/transformer_blocks.py`(가능하면), `b1_train.py`의 CV·early-stop 골격

---

### M6. GT 미래 물체 빌더 (새 target. `_compute_agent_targets`는 건드리지 않음)

**왜 새로 만드나**
- `_compute_agent_targets`는 t0에 전방 0–32 m, ±80° 안에 중심이 있는 물체만 추적하고 미래는 중심만 4 s까지 기록합니다.
- 60 m 안 track 중 21.2%만 남습니다. navtest NC 원인 물체의 18.6%(52/279)를 놓치고, TTC 원인은 37.6%만 덮습니다.
- 나중에 등장하는 물체가 19.6%입니다.

**역할:** 공식 metric cache와 같은 정의의 미래 물체 집합을 validity와 함께 만듭니다.

**입력**
- raw log `anns`, 프레임 cur..cur+10 (cur = 3, 0..5 s keyframe 11개)
- 필드 `gt_boxes`, `gt_names`, `gt_velocity_3d`, `track_tokens`(프레임 간 유지되는 식별자)
- 각 프레임의 ego pose

**처리**
1. 모든 class, 360°에서, 어느 keyframe에서든 중심이 t0 원점에서 **80 m** 안인 track을 모읍니다.
   - 60 m가 NC/TTC 원인을 전부 덮는 최소값이고, 80 m는 여유분입니다.
   - token당 평균 108개, 최대 447개입니다.
2. 각 프레임 ego 좌표를 t0 N 좌표로 변환합니다(`_track_future`와 같은 방식).
3. 첫 등장 시의 L, W, heading, 속도를 따로 저장합니다. 공식 채점은 크기와 충돌 분류에 첫 등장 값을 씁니다.

**출력 (token당, npz)**

| 키 | 형태 | 내용 |
|---|---|---|
| `kf` | [A, 11, 6] f32 | x, y, heading(unwrap), vx, vy, present |
| `first` | [A, 6] f32 | L, W, heading, vx, vy, first_k |
| `meta` | [A, 5] i16 | class, is_agent(1: vehicle/pedestrian/bicycle, 0: static), first_k, last_k, singleton |
| `track` | [A] str | track_token |

ragged로 평균 약 31 KB, A_max = 512 padding 시 약 147 KB입니다.

**0.1 s 질의 함수:** `query(t) → boxes [A, 5], state [A]`. 51 step, keyframe 선형 보간이며 metric cache와 같은 규칙입니다.

**validity 3단계와 singleton 처리**

| 상태 | 정의 | 대리 손실에서 |
|---|---|---|
| OBS | first_k·0.5 ≤ t ≤ last_k·0.5 | 사용 |
| ABSENT_OFFICIAL | 등장 전이나 사라진 뒤 (공식 채점도 없음으로 봄) | 채점과 맞추기 위해 없음으로 처리. 다만 `pre_entry`/`post_exit` flag로 따로 집계. 등장 전 물체의 51%가 40 m 안에서 처음 나타나므로 물리적으로는 "미확인"이기 때문 |
| UNKNOWN | 그 시각 GT ego에서 75 m [가정] 밖(주석 한계 77–83 m), 또는 t > 5.0 s | **손실에서 제외하고 없음으로 치지 않음**. 해당 초안에 `has_unknown` flag |

- singleton(한 번만 주석된 track, NC 원인의 2.5%)은 기본값으로 **A: 공식 규칙대로 51 step 전체에 정지 배치**합니다. 대안 B는 §5 D8입니다.

**비용과 저장**
- 저장: TRAIN-OUT과 navtest 합계 약 1 GB.
- 시간: log 읽기가 대부분이고, CPU 4 worker로 1시간 미만 [가정].

**검증 (V2)**
- navtest metric cache와 polygon을 비교합니다. 목표는 모서리 오차 < 1e-3 m, 존재 여부 100% 일치입니다. 지금까지 10 token에서 오차 0이었습니다.
- navtrain은 E의 metric cache 9,000개 중 200개로 같은 검사를 합니다.
- red-light pseudo-object를 공식 NC가 세는지는 [미확인]입니다. log 주석에는 이것이 없으므로 metric cache와 비교해 확인합니다.

**구현 위치**
- `tools/refiner/build_future_objects.py`
- 질의 함수 `navsim/agents/para_ssr/refiner/gt_future.py`
- 저장 `/home/external-user/ssd/yongjae_refiner/objects/{trainout,navtest}/` [가정: 쓰기 권한 확인]

**재사용:** scratchpad `rebuild_check.py`(오차 0 재구성 코드), `mc_check.py`, `cause_cov.py`, `navsim/common/dataclasses.py` L382-392 로더

---

### M7. 대리 손실

모든 항은 M4의 **0.1 s 41점 참조 궤적**(n = 1..40)에서 계산합니다. 채점기는 이 참조를 LQR로 추종한 궤적으로 판정하므로, 이 참조만 보는 손실은 원리적으로 일부 실패를 놓칩니다. raw 궤적에서 겹침이 전혀 없는 NC 실패가 32.3%, DAC는 약 26%입니다. 그 정도는 M8에서 잽니다.

**(1) 충돌 C_col: 사각형-사각형 부호 분리 거리**
- ego 박스 B_e(n): 중심은 (x_n, y_n) + 1.461·(cos ψ_n, sin ψ_n), 반크기 (2.588, 1.1485).
- 물체 박스 B_j(n): M6 질의값.
- 축 u ∈ {두 박스의 변 방향 4개}마다
  - gap_u = |(c_j − c_e)·u| − r_e(u) − r_j(u)
  - r(u) = a₁|u·û₁| + a₂|u·û₂| (a는 반크기, û는 박스 축)
- g_j(n) = smoothmax_u gap_u (logsumexp, 온도 0.05 m [가정]).
  - g > 0이면 떨어져 있음 (실제 거리의 하한).
  - g ≤ 0이면 겹침이고, −g는 SAT 최소 관통 깊이.
- 식: **C_col = (1/40) Σ_n Σ_j w_j · m_j(n) · β·softplus((m_col − g_j(n))/β)**
  - m_col = 0.3 m, β = 0.1 m [가정]
  - w_j = 1.0 (agent), 0.5 (static). NC 감점 0과 0.5에 대응.
  - m_j(n) = 1[OBS] · 1[뒤쪽 아님: 물체 중심이 ego 중심보다 heading 방향으로 −1.3 m 이상 앞] · 1[사람도 같은 (j, n)에서 g < m_col이 아님] · 1[UNKNOWN 아님]
- 속도를 위해 ego 중심에서 20 m 밖 물체는 미리 뺍니다. 기울기와 무관합니다.
- 원 근사(32-B: ego 원 3개, r = 1.436 m)는 비교용 ablation으로만 둡니다.
  - 원래 궤적 AUC 0.977이 큰 수정(D2e_0.1)에서 0.593으로 떨어졌습니다. 물체 집합이 궤적에 따라 달라졌기 때문으로 보입니다.
  - 원 근사는 옆으로 0.29 m를 과하게 덮습니다.

**(2) 경계 C_dac: 모서리 SDF**
- **C_dac = (1/40) Σ_n Σ_{모서리 q=1..4} in_E(q) · β·softplus((m_dac − σ(q))/β)**
  - σ는 E 격자 SDF의 bilinear 값, m_dac = 0.2 m [가정], β = 0.1.
  - SDF는 metric cache `drivable_area_map`, 즉 공식 DAC 층(ROADBLOCK, INTERSECTION, DRIVABLE_AREA, CARPARK)으로 만듭니다.
  - kyungmin의 차선 포함 합집합과 판정이 달랐던 궤적은 5,658개 중 1개였습니다.
- E 격자 밖 모서리는 `n_oog`로 세고 손실에서 뺍니다(padding 복사 금지). 원래 궤적 중 72 m를 넘는 것은 0개였습니다.

**(3) TTC C_ttc (선택, 작은 가중)**
- Δ ∈ {0.3, 0.6, 0.9} s마다 ego 박스를 등속으로 v_n·Δ만큼 투영하고, 물체는 t_n + Δ(최대 4.9 s)의 값을 씁니다.
- 앞쪽 물체에만 margin 0으로 같은 식을 적용합니다.
- 공식 TTC 규칙의 근사이며 [가정]입니다.

**(4) 진행 L_prog: EP 정규화와 맞춤**
- P(τ)는 4 s 끝점을 경로 중심선에 투영한 진행량입니다(음수는 0). P_pdm은 metric cache PDM-Closed 궤적의 진행량입니다 [가정: `trajectory` 필드가 PDM-Closed].
- 공식 EP ≈ min(1, P/max(P_pdm, 5 m))이므로:
  - **모드 A:** L_prog = [min(P(τ0), P_pdm) − P(τ1)]₊ / max(P_pdm, 5 m). PDM-Closed보다 앞선 초안을 그 수준까지 늦추는 것은 EP 손실이 아니므로 벌하지 않습니다.
  - **모드 B:** L_prog = 1 − min(P(τ1), P_pdm)/max(P_pdm, 5 m). PDM-Closed를 넘는 가속에는 보상이 없습니다.
- P_pdm이 없는 경우(단계 E의 TRAIN-IN)는 P_pdm = ∞로 두어 초안 대비 손실로 대체합니다.

**(5) 편안함 C_cmf**
- 0.1 s 격자에서 계산합니다: 종가속, 종 jerk, 횡가속 v²κ, yaw rate vκ, yaw 가속.
- C_cmf = Σ_q mean_n relu(|x_q| − 0.9·lim_q)² / lim_q².
- 한계: 종가속 −4.05/+2.40, 종 jerk 4.13, 횡가속 4.89, yaw rate 0.95, yaw 가속 1.93.
- 참고: 규칙 필터(greedy 감속)는 수정 token의 0.75–3.4%에서 comfort 실패를 만들었습니다.

**(6) 수정량 C_mod**
- C_mod = mean_n |s1 − s0|/5 m + mean_n |d|/1 m.
- "필요 없으면 고치지 않음"을 학습시키는 항입니다.

**(7) gate L_gate**
- BCE(p_g, y_g), pos_weight = (1−π)/π, 최대 10.
- y_g = 1[초안이 공식 NC < 1 또는 DAC < 1]. 초안 은행의 공식 채점값입니다. TTC 포함 여부는 §5 D12.
- 공식 라벨이 없는 초안에는 대리 라벨 1[C_col + C_dac > ε]을 씁니다 [가정].

**총손실과 가중 보정**
- L = w_col·C_col + w_dac·C_dac + w_ttc·C_ttc + w_prog·L_prog + w_cmf·C_cmf + w_mod·C_mod + w_gate·L_gate
- 초기값 [가정]: 1.0 / 1.0 / 0.3 / 2.0 / 0.1 / 0.1 / 0.5.
- **보정 절차 (사전 고정):**
  - 격자: (w_prog, w_mod) ∈ {0.5, 1, 2, 4} × {0.03, 0.1, 0.3}.
  - 방법: TRAIN-OUT inner-val의 실제 초안과 은행 초안을 공식 채점합니다.
  - 목표: 순해소(NC∪DAC) 최대. 조건은 평균 EP 손실 ≤ 0.5점, 불필요한 수정 비율 ≤ 5% [가정].
  - **모든 arm이 같은 격자와 같은 예산**을 씁니다.

**구현 위치:** `navsim/agents/para_ssr/refiner/surrogate.py`

**재사용**
- `sf_common.ego_corners`, `sat_overlap` (판정용 SAT 참조 구현)
- kyungmin `feasibility.cost_from_sdf` (softplus 형태), `plan_map.footprint_corners`
- `b1_train.collision_penalty`, `common.box_to_circles` (원 근사 ablation)

---

### M8. 대리 손실 검증 모듈 (학습 전에 먼저 돌림)

**역할:** "대리 손실이 공식 판정을 얼마나 반영하나"를 궤적 단위와 **수정 쌍 단위**로 잽니다. refiner는 궤적을 바꾸는 모듈이므로 쌍 단위가 핵심입니다.

**입력 풀 (공식 점수가 이미 있음)**

| 풀 | 규모 | 용도 |
|---|---|---|
| E navtrain | 9,000 × (student, 사람, 교정 5종) | **설정 선택용** |
| 초안 은행 pilot | TRAIN-OUT 500 × 13 | **설정 선택용**, 큰 수정 포함 |
| P0 navtest 실제 초안 | 12,146 | 보고용. 선택에 쓰지 않음 (navtest 누수 방지) |
| P1 32-B 수정 궤적 | 24 set × 12,146 | 보고용 |
| P2 규칙 필터 | 29,288 | 보고용 |
| P3 kyungmin 스트레스 풀 | 3.1M | v2.2 채점기라 v1과 섞지 않음. 기술적 참고만 |

**비교할 설정**
- 기하: box-box / 원 / 모서리 SDF
- 물체 집합: M6 전체 / 40 m·64개 / 현재 ROI
- 시간: 0.1 s / 8 knot
- margin: 0 / 0.2 / 0.3 / 0.5
- 궤적: raw / 공식 simulate(천장값) / M7b 추종 근사(선택)

**지표**
1. **궤적 단위:** AUC, 학습 margin에서의 recall·precision·FPR, FPR 1%·5%에서의 recall.
   - 층화: 실패 유형(agent/static/DAC), 원인 물체 출처(ROI 안·밖·나중 등장·singleton), 시점(0–2 / 2–4 s), x > 32 m 여부, LQR 편차 구간.
2. **쌍 단위:** 같은 token의 (초안, 수정)에서 다음을 봅니다.
   - P(공식 해소 | 대리 손실이 해소라고 함)
   - P(공식 새 실패 | 대리 손실이 안전하다고 함)
   - Δ대리 손실과 Δ공식의 부호 일치율
3. 첫 위반 시점 차이 |Δt| ≤ 0.5 s 비율, 위반 물체가 `nc_track`과 같은지.
4. 사람 궤적 오경보율.
5. 136 log 단위 bootstrap, 건수는 Clopper-Pearson 구간.

**사전 합격 기준 [가정: 사용자 확정]** (E navtrain과 은행 pilot 기준)
- NC recall ≥ 0.70 (pilot raw 0.1 s SAT는 0.75)
- 사람 오경보 ≤ 2%
- 쌍 단위 P(공식 해소 | 대리 해소) ≥ 0.6
- DAC recall ≥ 0.70

**불합격이면**
- M7b를 넣습니다. M7b는 미분 가능한 추종 근사로, 1 s 앞 속도 추종과 0.2 s 가속 지연을 가진 자전거 모델을 41 step 풀어 쓰는 것입니다 [가정].
- 또는 margin을 다시 정합니다.
- 이 결정은 refiner 학습 전에 끝냅니다.

**구현 위치:** `report/refiner_feasibility/m8_validate_surrogate.py` (출력: `m8/*.parquet`, `m8/summary.json`)

**재사용:** scratchpad `pilot_surrogate.py`, `pilot_analyze.py`. `sat01_f`의 마지막 시점 속도 0 버그는 고쳐서 씁니다.

---

### M9. 평가 모듈

**주 평가: navtest 실제 초안**
- 입력: `work_dirs/eval/para_ssr_interaction_final_navtest_trajectories.pkl` (12,146 × [8,3]), 각 arm의 refiner ckpt(seed 3개), navtest feature(teacher val cache, student D1 dump, GT raster).
- 처리: τ1 = decode(τ0, Δτ)를 p_g ≥ θ일 때 적용하고, 아니면 decode(τ0, 0)을 냅니다.
  - θ는 **inner-val에서 미리 정한 값**을 씁니다.
  - 결과는 같은 pkl 형식으로 저장하고, 공식 PDM 채점(v1 fork, `SSR/data/exp/metric_cache`)을 돌립니다.
- **기준선:** decode(τ0, 0)입니다. H1이면 τ0와 같습니다. 기존 csv는 PDMS 0.8487, NC 실패 279(csv로는 278), DAC 813, TTC 782입니다.

**지표**

| 지표 | 정의 |
|---|---|
| 해소 / 새 실패 / 순해소 | NC, DAC, TTC별 및 NC∪DAC 합 |
| 전체 실패 수 | NC, DAC, TTC, NC∪DAC∪TTC (기준 1,524) |
| PDMS, ΔPDMS | 전체 및 원래 통과 장면만 |
| 진행 손실 | ΔEP 평균, 4 s 진행 거리 변화 |
| 수정률 | p_g ≥ θ이고 ‖Δ‖ > 0.1 m인 비율 |
| 불필요한 수정 | 원래 NC·DAC·TTC를 모두 통과했는데 수정했고 PDMS가 오르지 않은 수. 규칙 필터에서는 811 → 0 개선이었음 |
| comfort 실패 | 새로 생긴 수 |
| 층화 | NC 279를 원인별로: 정지 물체 149 / 움직이는 물체 119 / t0에 없음 11. 사건 시점 원인 물체가 32 m 격자 밖인 41건. teacher가 그 물체를 검출했는지 여부 |

**통계**
- seed 3개 평균
- 136 log 단위 짝지은 bootstrap 10,000회, 95% 구간

**부가 대조군**
- **동일 진행 대조군:** 각 arm의 평균 ΔEP와 같아지도록 전체 궤적을 똑같이 느리게 한 결과.
- **장면 선택 대조군:** 각 arm의 수정 모양을 같은 수의 무작위 token에 옮겨 붙인 결과. 32-C 방식입니다.
- **임계값 sweep 곡선:** θ를 바꾸며 (진행 손실, 순해소)와 (불필요한 수정, 순해소) 곡선을 그립니다. **같은 진행 손실에서 비교**하기 위해서입니다.

**분석 평가 (판정에 쓰지 않음): 교란 초안**
- navtest 사람 궤적으로 은행을 만듭니다. token 2,000개 × 8개, 고정 seed [가정]. 공식 채점합니다.
- 보는 것: family별·크기별 복원율(교란으로 생긴 실패가 해소되는 비율)과 과잉 수정. 즉 "어느 범위까지 고칠 수 있나"입니다.

**구현 위치**
- `report/refiner_feasibility/eval_refiner.py`, `analyze.py`
- 채점 래퍼 `tools/refiner/score_trajectories.py`

**재사용:** `collision_counterfactual`의 `cf_common.score`, `B_read_vs_generate/b2_score.py`와 `run_score_sub.sh`, 32-C 대조군 코드(`C_uninformed_and_gating/code`)

---

### M10. E2E 통합 (단계 E)

**배치 위치:** `para_ssr_model.py`의 forward에서 필요한 값을 모두 얻은 직후에 붙입니다.
- `planner_head.select_trajectory`가 τ0 [B,8,3]를 냅니다.
- `bev_embed` [B,5000,256]
- `final_norm` 출력 h[:,0] = h_final [B,256]

**구성 요소**
- **R_S^E2E:** M5와 같은 구조이고 A_S 어댑터를 씁니다. h_final token은 §5 D7에 따라 선택합니다. 처음부터 함께 학습합니다.
- **R_T:** 단계 F의 R_T를 고정해서 씁니다. 또는 TRAIN-IN+OUT 교란 초안과 이른 checkpoint 초안(`dump_ep2/ep9/ep19`)으로 다시 학습한 것을 씁니다(§5 D17).
  - 입력은 teacher `cache_train_50x100`과 sg(τ0)입니다.
  - A_T 출력 [64,50,100] f16을 미리 계산해 두면 token당 약 640 KB로 원본 2.65 MB의 1/4이 됩니다. 어댑터가 고정이라 가능합니다.

**손실**
- L_base: 기존 PARA-SSR 손실(τ0 모방, det, motion, map, GradBalancer 0.4/0.3/0.3). 바꾸지 않습니다.
- L_KD = m_KD · [ |c_S^lon − sg(c_T^lon)|₁/2 m/s + |c_S^lat − sg(c_T^lat)|₁/1 m + BCE(p_S, sg(p_T)) ]
  - **제어점 공간**에서 맞춥니다. 자유도가 같기 때문입니다.
  - 대안은 decode한 pose 공간입니다.
- L_sur: M7(τ_final). GT 물체는 M6를 TRAIN-IN까지 확장해서 씁니다. 약 2.6 GB입니다.

**기울기**

| 경로 | 기본값 | 이유 |
|---|---|---|
| τ0 → refiner 입력 | **sg** | KD가 planner를 "refiner가 고치기 쉬운 초안"으로 끄는 것을 막기 위해 |
| student BEV → R_S | 배율 γ_bev = 0.1 [가정] | teacher의 교정 지식이 student BEV로 들어가는 유일한 통로. γ=0이면 KD가 head만 학습 |
| L_sur → τ0 | 막음 (선택 arm에서 허용) | planner 자체 안전성 학습과 섞이지 않게 |

**KD 게이팅과 일정 [가정]**
- λ_KD는 epoch 0–4에 0, 5–9에 선형으로 올리고, 10–29에 고정합니다. 초기 초안은 R_T 학습 분포 밖이기 때문입니다.
- m_KD 기본값은 모든 초안에 1, 신뢰도 가중 |2p_T − 1|입니다.
- 변형은 "GT 확인" 게이트입니다: M7(decode(τ0, Δτ_T)) ≤ M7(τ0)일 때만 KD합니다. 32번에서 학습 중 teacher-agrees 게이트가 해악을 32 → 6으로 줄였습니다. 다만 GT를 쓰면 "왜 GT를 직접 쓰지 않나"가 되므로 아래 E3 arm과 반드시 짝으로 봅니다.

**E2E arm (처음부터 학습, seed ≥ 2)**

| arm | 구성 | 무엇을 가르나 |
|---|---|---|
| E0 | PARA-SSR (refiner 없음), 같은 설정으로 재학습 | 기준 |
| E1 | + R_S^E2E, L_sur만 (KD 없음) | GT 대리 손실 직접 지도의 효과 |
| **E2** | + R_S^E2E, L_sur + KD(R_T) | **aux teacher KD의 효과** (E1 대비) |
| E3 | + R_S^E2E, L_sur + KD(R_GT-cur) | aux teacher가 GT 특권 교정기보다 나은가 (필요성 시험) |

**비용 [가정]:** 30 epoch 학습 한 번에 2 GPU로 약 26 h입니다(report 34 추정). 4 arm × 2 seed면 약 400 GPU-h입니다. 현재 GPU 6개가 모두 사용 중입니다.

**구현 위치** (모두 flag 뒤에 둠. 꺼져 있으면 기존 동작과 같아야 하며 `test_loss_parity.py`로 확인)
- `navsim/agents/para_ssr/refiner/e2e_head.py`, `kd.py`, `targets.py` (`FutureObjectTargetBuilder`, `DrivableSDFTargetBuilder`)
- `para_ssr_model.py`, `para_ssr_loss.py`, `configs/default.py`, `para_ssr_features.py` (teacher feature loader)
- 새 experiment yaml

**재사용:** byounggun Stage 2의 teacher cache 로더(`distill_feature_root` 경로 규약), `dump_states.plan_capture`

---

## 3. 데이터 파이프라인과 저장

### 3-1. 분할과 쓰임

| 분할 | 규모 | 단계 F | 단계 E | 비고 |
|---|---|---|---|---|
| TRAIN-IN (train_logs) | 85,109 | **쓰지 않음** | student 학습, KD | student feature가 외워진 상태(인지 recall 76.8% vs 62.0%). teacher 박스 밀도도 높음(18.1 vs 9.0) |
| TRAIN-OUT (val_logs 214) | 18,179 | **학습 풀**. log 단위로 80/20 (약 171 / 43 log [가정]) → train / inner-val | (선택) R_T 재학습 | student에게는 안 본 log. teacher는 navtrain 전체를 학습해서 봤음(밀도는 navtest와 비슷) |
| navtest | 12,146 (136 log) | **최종 평가 1회** | 최종 평가 | 선택에 쓰지 않음 |

### 3-2. 누수 규칙

1. navtest의 점수와 라벨은 가중, θ, margin, 대리 손실 설정, 조기 종료 어디에도 쓰지 않습니다. M8 설정 선택도 navtrain 풀로 합니다.
2. 모든 분할은 log 단위입니다.
3. 32-B 산출물(navtest 5-fold CV로 학습)은 학습 데이터로 쓰지 않습니다.
4. refiner는 추론 입력으로 GT를 받지 않습니다. R_GT arm만 예외이며 이들은 기준점입니다.
5. 사람 궤적은 손실 안에서만 씁니다(사람 위반 쌍 제외, identity 초안).
6. arm 간 동일성: 같은 token, 같은 초안 은행, 같은 seed 목록.
7. teacher가 TRAIN-OUT을 학습 때 본 사실은 보고서에 한계로 적습니다.

### 3-3. 만들 것과 크기

| 항목 | 내용 | 규모와 크기 | 시간 | 위치 [가정] |
|---|---|---|---|---|
| D1 student dump | `bev_embed` f16 [256,50,100], `trajectory`, h_final(hook 또는 `plan_capture`), det/motion/map 출력 | (18,179 + 12,146) × 2.7–2.9 MB ≈ **85 GB** | GPU 1개 약 45 분 (11–12 tok/s) | `/home/external-user/ssd/yongjae_refiner/student/` (kyungmin `cache_student_bev` shard 형식에 우리 `build_agent`) |
| D2 teacher | 기존 캐시 그대로. 필요하면 TRAIN-OUT과 navtest만 SSD로 복사 | 30,325 × 2.65 MB ≈ 80 GB (복사 시) | 복사 수십 분 | 원본 `/home/external-user/datasets/teacher_cache/bevfusion/cache_{train,val}_50x100` |
| D3 미래 물체 (M6) | npz | 약 1 GB | CPU 1 h 미만 | `…/objects/` |
| D4 GT raster, SDF | 현재 [12,50,100] 약 70 KB, 미래 [19,50,100] 약 100 KB, E 격자 SDF [320,256] f16 160 KB | 약 0.33 MB × 30k ≈ **10 GB** | SDF 0.54 s/token → 8 worker로 약 35 분 | `…/rasters/` |
| D5 TRAIN-OUT metric cache | E의 3,000개 + 새로 15,179개 | 0.33 MB × 15k ≈ 5 GB | 4 worker로 약 1.85 h (E: 2.28 tok/s) | `E_train_split_feasibility/e2_cache.py` 사용 |
| D6 초안 은행 + 공식 점수 | TRAIN-OUT 18,179 × 13 ≈ **236k 궤적** (+ navtest 분석용 16k) | 1 GB 미만 | **채점 속도 미측정 [확인 필요]**. V5에서 재고 느리면 K=8 | `…/draft_bank/` |
| D7 기존 풀 | navtest csv, P0–P2, E | 있음 | – | 기존 경로 |
| 단계 E 추가 | TRAIN-IN M6 (약 2.6 GB), SDF (13.6 GB), A_T 출력 사전 계산 (85k × 640 KB ≈ 54 GB) | 약 70 GB | CPU 수 시간 | – |

- 합계: 단계 F 약 100–190 GB. `/home/external-user/ssd` 여유는 1.6 TB입니다.
- 학습 I/O: arm당 epoch마다 TRAIN-OUT-train 약 14.5k token × 2.6 MB ≈ 38 GB를 읽습니다. token 단위 배치이므로 초안 13개가 한 번의 읽기를 공유합니다.

---

## 4. 단계 F 실험 조건표와 사전 판정 규칙

### 4-1. 조건표

| arm | 입력 (M2) | 역할 | 판정 지위 |
|---|---|---|---|
| **R_S** | student BEV | 비교 기준 | **확인적 (주 비교)** |
| **R_T** | teacher BEV | 시험 대상 | **확인적 (주 비교)** |
| R_none | 없음 (초안 + ego 상태만) | 장면 정보 없이 학습형 교정이 할 수 있는 몫 | 타당성 점검 |
| R_GT-cur | 현재 GT raster (t0 박스 + 속도 + heading + 지도) | **완벽한 현재 인지의 기준점** | 기준점 |
| R_GT-cur-obj | 현재 GT 박스 raster 7채널만 | 물체 정보만의 기준점 | 기술적 |
| R_T-box / R_S-box | teacher / student 검출 박스 raster 7채널 (R_GT-cur-obj와 같은 채널) | feature 대 박스: 검출 품질 차이만 분리 | 기술적 |
| R_T+box | teacher BEV + 박스 raster | teacher 박스가 추가로 주는 몫 | 기술적 |
| R_S+h | student BEV + h_final | student 계획 상태가 주는 몫 | 기술적 |
| R_GT-fut | 미래 GT raster (실제 미래 점유) | **oracle 참고값**. 인지 비교가 아님 | 참고 |
| R_GT-fut-ext | 미래 GT, 전방 −8..72 m 확장 격자 | 32 m 격자 한계의 크기 | 참고 |
| 규칙 필터: 자기 인지 | 31번 (136 해소, PDMS +0.78) | 학습 없는 기존 기준 | 기준점 |
| 규칙 필터: teacher 정보 | 32-D (가짜 수정 739 → 383, 해소 +13 [−3, 30]) | teacher 정보의 규칙 기반 몫 | 기준점 |
| 규칙 필터: GT | 266 해소 | 규칙 필터 천장값 | 참고 |
| 동일 진행 대조군 | arm별 평균 ΔEP에 맞춘 일괄 감속 | "그냥 느려져서"를 제거 | 대조군 |
| 장면 선택 대조군 | arm의 수정을 무작위 token에 이식 | 장면 선택의 가치 | 대조군 |
| 기준선 | decode(τ0, 0) | 모든 비교의 0점 | – |

### 4-2. 사전 판정 규칙

결과를 보기 전에 `report/refiner_feasibility/PRESTATED_DECISION_RULE.txt`로 고정합니다.

**0단계. 측정 타당성 (inner-val, seed 1)**
- R_GT-cur의 순해소가 R_none보다 커야 합니다(95% 구간 하한 > 0).
- 아니면 refiner가 장면 정보를 쓰지 못하는 것입니다. teacher를 판정하지 않고 M3/M5를 다시 설계합니다.

**주 판정 (navtest, seed 3개 평균, 짝지은 log bootstrap)**

"R_T > R_S"는 다음을 **모두** 만족할 때입니다.
- (D1) 순해소(NC∪DAC)의 차이 R_T − R_S의 95% 구간 하한 > 0.
- (D2) 같은 진행 손실에서도 성립: θ sweep 곡선에서 두 arm 중 작은 쪽의 ΔEP에 맞춘 지점에서 R_T 순해소 ≥ R_S.
- (D3) 불필요한 수정: R_T ≤ R_S (점추정). 구간으로 볼 때 유의하게 나쁘지 않아야 합니다.
- (D4) R_T 순해소 > 동일 진행 대조군 (구간 하한 > 0).
- 최소 효과 크기: 순해소 차이 ≥ 10건 [가정: 사용자 확정]. 규칙 필터에서 teacher 정보의 이득이 +13 [−3, 30]이었으므로 검정력이 빠듯할 수 있습니다.

**결과별 결론**

| 결과 | 결론 |
|---|---|
| 통과 | 단계 E로 갑니다 |
| R_T ≈ R_S | 이 경로로는 teacher를 정당화할 수 없습니다. 중단하고 보고합니다 |
| R_T가 D3만 이김 (불필요한 수정만 감소) | "teacher가 과잉 수정을 줄인다"는 좁은 결론. 단계 E 진행 여부는 사용자가 판단합니다 |

**보조 보고 (판정에는 쓰지 않음)**
- teacher 격차 메움 비율 = (R_T − R_S)/(R_GT-cur − R_S)
- R_GT-fut와의 차이
- 교란 초안 복원율
- TTC, PDMS

---

## 5. 결정 사항 표

"권장"은 제안일 뿐이고 결정은 사용자가 합니다.

| # | 결정 | 선택지 | 권장 기본값 | 이유 |
|---|---|---|---|---|
| D1 | **가속 허용** | A 감속 전용 (s1 ≤ s0) / B 제한 가속 (+2 m/s, 연장 ≤ 5 m 외삽) / C 비대칭 | **단계 F 주 arm은 A, B는 사전 등록한 부 arm** | NC 실패는 대부분 사람보다 앞섭니다(비 중앙값 1.16, 움직이는 물체 충돌 Δs4 +3.5 m). 감속 교정으로는 통과 장면 점수가 오르지 않고(811 → 0), EP 여지는 작습니다(사람도 EP 0.87). 가속하면 추론 때 늘 외삽이 필요하고 TTC 위험이 늘며, "같은 진행" 비교가 흐려집니다. 대가: 사람보다 느린 실제 초안 12.7%와 EP < 0.5인 73건은 고칠 수 없습니다 |
| D2 | 느린 교란 초안 | A: 만들지 않음 / B: [−0.8, −0.2] m/s² 포함 | D1을 따름 | 자유도 일치 규칙 |
| D3 | 빠른 교란 범위 | A_p 상한 1.0 / **1.3** / 1.6 m/s² | 1.3 | NC p99 +1.27까지 덮습니다. log 경로 한도 안에서만 만듭니다 |
| D4 | 횡방향 범위 | 교정 D_MAX 1.0 / **2.0** / 3.0 m. 교란 ±1.5 m. 또는 횡방향 끄기 | 2.0 (교란 1.5) | 통과 그룹 max\|d\| p90 0.88, 전체 p95 1.5, NC p75 2.1 m. 진짜 차선 이탈은 11–14%뿐이라 횡방향 끄기 arm을 ablation으로 둡니다 |
| D5 | heading 규약 | **H1** (초안 heading + 보정) / H2 (접선) | H1 | 교정 0이 정확히 항등이라 기존 실패 집합(279/813/782)을 그대로 씁니다. H2 자체가 PDMS를 +1.06 바꾸므로(84.87 → 85.93) 그 효과가 refiner 몫으로 섞입니다. H2는 별도 점검용 |
| D6 | teacher 박스 입력 | 없음 / **raster 7채널** / token 집합 | 주 arm은 없음, 부 arm R_T+box는 raster | raster는 구조를 바꾸지 않고 R_GT-cur-obj, R_S-box와 같은 채널로 비교됩니다. 박스 변환 규약 재확인이 필요합니다 |
| D7 | R_S에 h_final 포함 | 주 arm 제외 / 포함 | **제외**. R_S+h는 부 arm | 포함하면 R_S만 입력이 하나 더 많아 주 비교가 불공정해집니다. 단계 E에서는 쓸 수 있습니다 |
| D8 | singleton 처리 | **A 공식대로 전 구간 정지** / B 해당 keyframe만, 나머지 unknown | A | 채점과 일치합니다. NC 원인의 2.5%입니다. B는 민감도 분석으로 |
| D9 | 충돌 기하 | **box-box 분리 거리** / 원 3개 / 모서리 SDF | box-box | 모서리만 보면 recall 0.32–0.41, 원은 옆을 과하게 덮습니다. M8 결과로 최종 결정 |
| D10 | 손실 시간 격자 | **0.1 s 41점** / 8 knot | 0.1 s | 8 knot SAT recall 0.41, 0.1 s 0.75. 10 m/s에서 knot 간격 5 m |
| D11 | 추종 반영 | raw 참조 / M7b 추종 근사 | raw로 시작하고 M8 결과로 결정 | raw에서 안 보이는 실패가 NC 32%, DAC 26% |
| D12 | gate 라벨 | **공식 NC∪DAC** / + TTC / 대리 라벨 | NC∪DAC | 곱셈 지표라 점수 영향이 가장 큽니다. TTC는 등속 투영 규칙이라 감속으로 오히려 늘 수 있습니다 |
| D13 | gate 적용 | **hard 임계값** / soft 곱 | hard, θ는 inner-val에서 | 불필요한 수정 억제가 판정 지표이기 때문 |
| D14 | 단계 F 학습 풀 | **TRAIN-OUT만** / + TRAIN-IN / 2-fold 교차 적합(약 13 h × 2) | TRAIN-OUT | TRAIN-IN의 student feature와 초안은 외워진 상태입니다. 데이터가 모자라면 교차 적합을 추가 |
| D15 | 사람 위반 쌍 제외 | **제외** / 포함 | 제외 | 사람 궤적도 원 penalty에 7.6% 걸립니다. 대부분 과실 아닌 접촉입니다 |
| D16 | 미래 GT 격자 | **50×100 (주)** / 확장 격자 (부) | 둘 다. 주는 50×100 | 같은 범위여야 인지 비교가 됩니다. 사건 시점 32 m 밖 NC가 41건이라 확장 격자는 참고값으로 |
| D17 | 단계 E의 R_T | 단계 F R_T 그대로 / TRAIN-IN+OUT과 이른 ckpt 초안으로 재학습 | 재학습 (구조와 손실은 같음) | 단계 E의 초안은 초기 student 분포이고 TRAIN-IN 위에 있습니다 |
| D18 | E2E BEV 기울기 γ_bev | 0 (detach) / **0.1** / 1 | 0.1 | γ = 0이면 KD가 student 인지 표현에 닿지 않습니다. 1이면 불안정할 위험 |
| D19 | E2E τ0 stopgrad | **sg** / 통과 | sg | planner가 "고치기 쉬운 초안"으로 끌려가는 것을 방지 |
| D20 | KD 공간 | **제어점** / pose | 제어점 | 자유도가 같고 척도가 명확합니다 |
| D21 | KD 게이팅 | **전부 + 신뢰도 가중** / GT 확인 게이트 | 전부. GT 확인은 변형 | GT 게이트는 "왜 GT가 아닌가" 문제를 키우므로 E3와 짝으로만 |
| D22 | KD 일정 | 0–4 epoch 0 → 5–9 ramp → 고정 / 처음부터 | 앞의 것 | 초기 초안은 R_T 학습 분포 밖 |
| D23 | 안전 초안 비율 | identity와 작은 교란 20 / **30** / 40% | 30%, pilot으로 조정 | 실제 실패율이 1.5–2.3%로 낮아 과잉 수정 위험이 큼 |
| D24 | 손실 가중 | 초기값 + 공통 격자 보정 | §M7 절차 | arm 간 공정성 |

---

## 6. 기존 작업과의 관계 (신중한 기술)

아래 서술은 감사 문서(`report/collaborator_audit/`)와 이번 사실 조사에서 확인한 범위만 담았습니다. "이미 했다"가 아니라 **"관련되거나 일부 겹치는 요소"**로 적었습니다.

| 기존 작업 | 확인된 범위 | 겹치는 요소 | 다른 점 | 이 설계에 주는 의미 |
|---|---|---|---|---|
| **kyungmin feasibility** (SSR-v2 report 27, `feas_gt`/`feas_pred`) | v2 anchor planner(후보 256개) 학습 중 후보별 footprint 모서리 SDF 비용을 offset·점수 head에 걸었습니다. 정적은 DAC 층 합집합, 동적은 미래 주석 박스 전체로 만든 8장 SDF(전방 0–32 m, 격자 밖 border padding)입니다. feas_gt ep29 EPDMS 88.13 vs 87.80, 단일 seed. feas_pred PDMS 88.00 vs 88.14 | GT 미래 물체 기반의 미분 가능한 기하 비용, 모서리 SDF, 도로 폴리곤 코드 | 우리는 초안 뒤에 붙는 교정 모듈과 gate이고, 목적은 입력 feature 출처 비교입니다. 물체는 box-box·0.1 s·80 m·validity mask로 다룹니다 | 그 결과는 anchor planner 학습 설정의 결과이므로 교정기 효과의 근거로 옮기지 않습니다. 코드는 재사용하되 padding 규칙은 바꿉니다 |
| **kyungmin readout KD** (report 19/22, v1) | ReSMap BEV 위 소형 readout planner. interaction_final에서 5 epoch fine-tune로 z, attention 가중 feature, 전체 feature, 민감도 KD를 했고 대조군 대비 −0.36 ~ +0.02로 효과가 없었습니다. S_student 82.32 ≥ S_own 81.49 | "teacher feature를 읽는 작은 모듈을 학습하고 그 결과를 student로 옮긴다"는 큰 틀 | teacher 종류(검출 BEVFusion 대 지도 ReSMap), 증류 대상(교정 출력 대 readout 잠재·feature), 학습(처음부터 E2E 대 fine-tune), 지도 신호(GT 대리 손실 대 모방) | **같은 reader로 읽었을 때 student BEV가 teacher BEV만큼 planning 정보를 가졌다는 관찰은 R_T vs R_S 게이트가 실패할 수 있다는 직접적인 경고입니다.** 그래서 게이트를 E2E 앞에 둡니다 |
| **kyungmin report 23 §2-A, entity critic** (24/25) | GT 특권 planner teacher KD를 제안했다가 보류했습니다. GT entity critic은 GT 인지에서 +1.5 EPDMS, 예측 인지에서 −1.35 | GT 특권 입력 모듈을 기준점으로 쓰는 발상 (R_GT-cur, E3) | 우리는 scorer가 아니라 교정기이고, teacher 대 GT를 같은 구조로 비교합니다 | GT 인지 상한이 작을 수 있습니다. R_GT-cur − R_S 폭이 작으면 teacher가 메울 여지도 작습니다 |
| **byounggun Stage 1** | PARA-SSR planner를 캐시된 BEVFusion(50×100 포함) 또는 ReSMap BEV와 command로 학습했습니다(ego 상태 0). 궤적 L1만 있고 PDMS는 없습니다. planner는 버리고 adapter만 썼습니다 | 같은 BEVFusion 50×100 캐시를 고정 teacher feature로 읽는 planning 모듈 | 우리는 전체 planner가 아니라 교정 모듈이고, 모방이 아니라 GT 대리 손실로 학습하며, student feature arm과 같은 구조로 비교합니다 | teacher feature의 planning 가치는 아직 closed-loop로 측정된 적이 없습니다. 이 설계의 게이트가 그 측정입니다 |
| **byounggun Stage 2** | 처음부터 E2E 학습하면서 고정 adapter 공간에서 corridor 가중 feature MSE를 걸었습니다. PDMS 0.830–0.856. 같은 설정의 KD 없는 대조군이 없고, ReSMap adapter가 1 epoch ckpt였습니다 | 처음부터 E2E 중 KD, 경로 주변 가중 | 증류 대상이 feature가 아니라 교정 출력(Δτ)이고, E0/E1 대조군을 둡니다 | feature 수준 KD의 효과는 분리되지 않았습니다(차이 ±0.003). 우리 E2E에는 같은 설정 대조군 E0/E1이 필수입니다 |
| **우리 32-B** | navtest 5-fold CV에서 h_final을 고정하고 decoder를 GT 원 충돌 penalty로 다시 학습했습니다(metric cache 물체, 40 m, 64개, 8 knot). 순효과 약 12% [6, 18]. 해소 128/279, 새 충돌 181 | GT 미래 물체 penalty로 궤적 출력층을 학습 | navtest를 학습에 쓰지 않고, 물체 집합 전체, 0.1 s, box-box, 교정 파라미터화, gate | 대리 손실만 보고 바꾸면 새 실패가 생깁니다(181) → M8 쌍 단위 검증과 gate가 필요합니다 |
| **우리 31 / 32-C / 32-D** (규칙 필터) | 자기 인지 필터 136 해소, 동일 진행 대조군 대비 +94 NC. teacher 정보는 가짜 수정을 절반으로 줄였고 해소는 +13 [−3, 30] | 같은 질문(teacher 정보가 교정에 주는 몫)의 규칙 기반 버전 | 학습형이고 횡방향이 있으며 gate가 있습니다 | R_T의 이득이 "해소 증가"보다 "불필요한 수정 감소"로 나타날 수 있습니다. 판정 규칙에 두 축을 모두 넣었습니다 |
| **우리 report 34 §6 / 35** | 학습형 속도 refiner(34, 미실행, T1 최소 수정 / T2 사람 잔차), teacher planner 읽기 KD(35) | 34 §6을 발전시킨 설계입니다 | 횡방향, gate, feature 출처 비교, 35의 "E2E 중 KD" 요구 | 새 아이디어가 아니라 34 §6의 구체화입니다 |
| **ThinkTwice** (CVPR 2023) [원문 세부는 이번에 재확인하지 않음] | coarse 궤적 주변 feature를 다시 읽어(look) 단계적으로 잔차를 고치는 decoder. 잔차 target은 전문가 궤적으로 알려져 있음 | "초안 주변 feature를 다시 읽어 잔차로 고친다"는 구조 (M3–M5) | target(사람 잔차 대 GT 대리 손실), gate, feature 출처 비교와 KD | 구조 자체는 새롭지 않습니다. 기여 주장은 "교정 출력의 teacher/student/GT 비교와 그 증류"로 한정해야 합니다 |
| **DistillDrive** (2025) [세부 미확인] | GT 구조화 인지를 입력으로 받는 같은 구조의 planner를 teacher로 두고 student로 증류하는 것으로 알려져 있음 | 특권 입력 planner → student 증류 | teacher 입력(aux 센서 모델 feature 대 GT), 증류 대상(교정 출력) | E3(R_GT-cur KD)가 사실상 이 비교축입니다. **E2가 E3보다 나아야 "aux teacher가 필요하다"를 말할 수 있습니다** |

---

## 7. 위험과 확인 순서

### 7-1. 주요 위험

| 위험 | 근거 | 대응 |
|---|---|---|
| 대리 손실의 사각지대 (LQR 추종) | raw에서 안 보이는 NC 32%, DAC 26%. 추종 편차 p90 1.01 m | M8에서 먼저 재고, 필요하면 M7b. 최종 판정은 공식 채점 |
| 32 m 격자 한계 (모든 arm 공통) | 궤적의 16–25%가 격자를 벗어나고, 사건 시점 32 m 밖 NC가 41/279 | 격자 밖은 mask. 층화 보고. 확장 격자는 참고 arm으로만 |
| 교란 초안과 실제 초안의 분포 차이 | 실제 오차는 작고(±0.4 m/s²) 실패는 드묾(1.5–2.3%) | 실제 초안 25% 가중, 주 평가는 실제 초안 |
| 낮은 실패율 때문에 gate 오경보 | 규칙 필터 해악의 80%가 예측 오류에서 나옴 (32-C) | 불필요한 수정을 판정 지표로, θ는 inner-val에서 |
| teacher가 TRAIN-OUT을 학습 때 봄 | teacher는 navtrain 전체로 학습 | 밀도로 보면 영향은 작음(9.0 vs navtest 10.3). 한계로 기록 |
| heading 규약 혼입 | H2 자체가 +1.06 | H1 기본값, 기준선은 decode(τ0, 0) |
| 검정력 부족 | 규칙 필터의 teacher 이득이 +13 [−3, 30] | seed 3개, 짝지은 bootstrap, 최소 효과 크기 사전 고정 |
| 채점 비용 | 초안 은행 236k 궤적, 속도 미측정 | V5에서 측정, 필요하면 K=8 |
| 일정 | 결과 동결 약 11-04. E2E 한 번에 약 26 h × 2 GPU, 현재 GPU 전부 사용 중 | 단계 F를 약 10-10까지 끝내야 E2E에 3주가 남음 [가정] |
| teacher 박스 규약 | z 미검증, 예전 90° 오류 기록 | V1에서 재확인 |

### 7-2. 확인 순서 (앞 단계 통과 후 다음으로)

| 순서 | 내용 | 합격 기준 | 예상 시간 [가정] |
|---|---|---|---|
| V1 | 좌표와 데이터 정합: teacher 좌우 뒤집기(heatmap 검사 재현), student dump 궤적이 pkl과 비트 단위로 같음, metric cache 원점, teacher 박스 heading·크기 | 전부 일치 | 반나절 |
| V2 | M6 빌더 대 metric cache: navtest 200, navtrain E 200 token. red-light pseudo-object 처리 확인 | 모서리 오차 < 1e-3 m, 존재 여부 100% 일치 | 반나절 |
| V3 | M4 디코더 단위 시험: 항등(H1), 교란 → 역교정 오차, 제약 위반 0, 41점이 채점기 참조와 일치, δa 실제 상한 | 항등 오차 0, 역교정 오차 < 0.1 m | 반나절 |
| V4 | M8 대리 손실 검증 (navtrain 풀로 선택, navtest는 보고) | §M8 사전 기준. 불합격이면 M7b 또는 margin 재설정 | 1–2일 |
| V5 | 초안 은행 pilot: TRAIN-OUT 500 × 13 공식 채점. family별 실패율, 연속성 분포, 채점 속도 | 실패율 20–35%, 첫 구간 속도 분포가 실제와 비슷 | 1일 |
| V6 | 전체 dump와 캐시: D1, D3–D6 | 파일 수와 크기 검사 | 약 1일 (wall) |
| V7 | 측정 타당성: R_GT-cur 대 R_none (inner-val, seed 1) | §4-2 0단계 | 반나절 |
| V8 | 전 arm × seed 3 학습, inner-val에서 가중·θ 선택, navtest 1회 평가 | – | 2–3일 (GPU 1개) |
| V9 | §4-2 판정, 보고서 작성 | – | 1일 |
| (이후) | 단계 E 구현 (flag 뒤, 동등성 시험) → E0–E3 | – | 약 3주 |

**구현할 파일 요약**

| 경로 | 파일 |
|---|---|
| `/home/external-user/yongjae/SSR/navsim/agents/para_ssr/refiner/` | `geometry.py`, `decoder.py`, `corridor.py`, `adapters.py`, `refiner_net.py`, `surrogate.py`, `gt_future.py`, `targets.py`, `e2e_head.py`, `kd.py` |
| `/home/external-user/yongjae/SSR/tools/refiner/` | `build_future_objects.py`, `build_gt_rasters.py`, `dump_student_states.py`, `make_draft_bank.py`, `score_trajectories.py` |
| `/home/external-user/yongjae/SSR/report/refiner_feasibility/` | `PRESTATED_DECISION_RULE.txt`, `m8_validate_surrogate.py`, `train_refiner.py`, `eval_refiner.py`, `analyze.py`, `configs/arms.yaml` |
| 캐시 [가정: 권한 확인] | `/home/external-user/ssd/yongjae_refiner/` |

**설계에서 참조한 기존 경로**

| 용도 | 경로 |
|---|---|
| student ckpt | `/home/external-user/yongjae/SSR/work_dirs/para_ssr_interaction_final/lightning_logs/version_2/checkpoints/last.ckpt` |
| navtest 초안 | `/home/external-user/yongjae/SSR/work_dirs/eval/para_ssr_interaction_final_navtest_trajectories.pkl` |
| teacher 캐시 | `/home/external-user/datasets/teacher_cache/bevfusion/cache_{train,val}_50x100` |
| navtest metric cache | `/home/external-user/yongjae/SSR/data/exp/metric_cache` |
| E 실험 | `/home/external-user/yongjae/SSR/report/cause_and_correction_tests/E_train_split_feasibility/` (`e2_cache.py`, `metric_cache/`, `tokens/`) |
| 32-B | `/home/external-user/yongjae/SSR/report/cause_and_correction_tests/B_read_vs_generate/` |
| 32-C | `/home/external-user/yongjae/SSR/report/cause_and_correction_tests/C_uninformed_and_gating/` |
| 32-D | `/home/external-user/yongjae/SSR/report/cause_and_correction_tests/D_teacher_info_filter/` |
| 규칙 필터 | `/home/external-user/yongjae/SSR/report/planner_vs_perception_tests/safety_filter/sf_common.py` |
| kyungmin 코드 | `/home/external-user/kyungmin/SSR-v2/navsim/agents/para_ssr/{plan_map.py, feasibility.py}`, `/home/external-user/kyungmin/SSR-v2/tools/readout/cache_student_bev.py` |
| 사실 조사 스크립트 | `/tmp/claude-1001/-home-external-user/9262a8d2-fa20-46d7-ba9a-92d7d13b8ceb/scratchpad/` (`rebuild_check.py`, `pilot_surrogate.py`, `c_navtrain_paths.py`, `d_analyze.py`, `e_decode_check.py`) |