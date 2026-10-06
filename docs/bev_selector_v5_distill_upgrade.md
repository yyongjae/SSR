# BEV Selector Distillation v5 (Advanced Distillation)
## GradBalancer 의존 탈피: 표현 공간 정렬, 스케일 복원, 상호 간섭 제거

**기준 문서**: [`bev_selector_v4.md`](bev_selector_v4.md), [`bev_selector_v3.md`](bev_selector_v3.md)  
**소속**: `/workspace/byounggun/SSR`  
**작성일**: 2026-10-05  
**실행 GPU**: GPU 4, 5  

---

## 1. 문제의식: 왜 GradBalancer 대신 Distill 자체를 개선해야 하는가?

v4에서 GradBalancer(0.4 / 0.3 / 0.3)를 도입한 것은 Det/Map 헤드 대비 Distill 그래디언트가 너무 약해(0.0005 vs 0.05) 인위적으로 밸브를 조인 것입니다. 하지만 이는 증상 완화제일 뿐, Distill 파트 자체에 존재하던 아래 4가지 병목을 해결하지 못했습니다.

1. **채널 기저 불일치 (Basis Mismatch)**:
   Student(ResNet-34 카메라)와 Teacher(BEVFusion 라이다+Swin-T, ReSMap)의 256차원 feature space는 직교 기저가 다릅니다. 프로젝션 없이 1:1 LN MSE를 강제하면 Student의 주행/검출 표현이 왜곡되어 태스크 헤드와 충돌합니다.
2. **Global Normalization으로 인한 5000배 희석**:
   `cell_w = _normalize_rows(attention.sum())`로 5000개 칸 합을 1.0으로 강제 나눔으로써, 활성화된 각 칸이 받는 그래디언트가 $10^{-5}$ 단위로 소멸했습니다.
3. **두 Teacher 간 Spatial Conflict**:
   동일한 `bev_embed`에 대해 도로 위에서 BEVFusion(3D 차량)과 ReSMap(도로 경계)이 상충하는 feature를 강요하여 그래디언트가 상쇄되었습니다.
4. **L2 단일 손실의 한계**:
   L2 손실만 쓰면 feature의 크기와 방향을 동시에 맞추려다 평균으로 뭉개집니다(blurring).

---

## 2. v5에서 적용된 핵심 개선

### 2.1 Bank별 경량 Linear Projection Adapter (`student_proj`)
- Object Bank(BEVFusion)와 Map Bank(ReSMap) 각각에 $256 \times 256$ Linear layer를 배치.
- 항등 행렬(Identity)로 초기화되어 학습 초기에 안정적으로 출발하며, Student feature를 Teacher 공간의 기저로 부드럽게 회전/투영.
- `agent._distill.*` 내부에 존재하므로, 추론 시에는 파라미터가 0개로 깔끔히 제거됨.

### 2.2 Loss Scale 복원 (`tok_scale=25.0`)
- 합이 1.0으로 정규화되면서 극도로 작아진 토큰 증류 손실을 $\times 25.0$ 스케일업.
- `gnorm/distill`이 기존 0.0005에서 **0.02 ~ 0.03** 수준으로 회복되어, Det/Map 헤드($0.03 \sim 0.05$)와 자연스러운 균형을 이룸.
- 인위적인 GradBalancer 억제 없이도 Student가 Teacher를 강력하게 학습함.

### 2.3 Hybrid (Normalized L2 + Cosine Similarity) 손실
\[
\mathcal{L}_{\mathrm{cell}} = 0.5 \cdot ||\mathrm{LN}(s) - \mathrm{LN}(t)||_2^2 + 0.5 \cdot (1 - \cos(\mathrm{LN}(s), \mathrm{LN}(t)))
\]
- 방향 정렬(Cosine)과 크기 정렬(L2)을 동시에 달성하여 표현의 변별력 유지.

### 2.4 공간적 가중치 부스팅 (`struct_mask_boost=2.0`)
- BEVFusion은 동적 차량 마스크 영역에 $+200\%$ 가중치 부여.
- ReSMap은 도로 경계 링(road boundary) 영역에 $+200\%$ 가중치 부여.
- 도로 위 겹치는 영역에서 두 Teacher가 충돌하지 않고 각자의 전문 영역에 집중하도록 유도.

---

## 3. 실행 파라미터 요약

| 항목 | v3 / v4 | v5 (이 런) |
|---|---|---|
| Student Projector | 없음 (직접 LN MSE) | **Bank별 Linear Projection (`student_proj`)** |
| Distill Token Scale | 1.0 (5000칸 희석) | **25.0 (자연스러운 gradient norm 확보)** |
| 손실 형태 | L2 MSE | **Hybrid (0.5 L2 + 0.5 Cosine)** |
| 공간 마스크 부스트 | 0.0 | **2.0 (차량=BEVFusion, 경계=ReSMap)** |
| GradBalancer | v3: 꺼짐, v4: 0.4/0.3/0.3 | **꺼짐 (`grad_balance_target: null`)** |
| GPU | GPU 6, 7 | **GPU 4, 5** |

---

## 4. 실행 명령

```bash
cd /workspace/byounggun/SSR
source env.vast.sh
CUDA_VISIBLE_DEVICES=4,5 nohup bash scripts/training/run_bev_selector_v5_distill.sh \
  > /workspace/byounggun/ssr_outputs/v5_train.log 2>&1 &
```
