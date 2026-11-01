# BEV Selector Distillation v4
## v3 결과 분석 및 GradBalancer 0.4/0.3/0.3 실험

**기준 문서**: [`bev_selector_v3.md`](bev_selector_v3.md)  
**소속**: `/workspace/byounggun/SSR`  
**작성일**: 2026-10-05  
**평가 체크포인트**: `ssr_outputs/paradrive_distill_bev_selector/lightning_logs/version_0/checkpoints/epoch=29-step=19950.ckpt`  
**평가 데이터셋**: navtest 12146 tokens  

---

## 1. 평가 점수 비교

| 항목 | v4 baseline | v1 selector | v3 selector (측정치) |
|---|---:|---:|---:|
| valid | 1.0000 | 1.0000 | 1.0000 |
| no_at_fault_collisions | 0.9810 | 0.9797 | 0.9816 |
| drivable_area_compliance | 0.9371 | 0.9152 | 0.9199 |
| driving_direction_compliance | 1.0000 | 1.0000 | 1.0000 |
| ego_progress | 0.8020 | 0.7836 | 0.7879 |
| time_to_collision_within_bound | 0.9439 | 0.9373 | 0.9407 |
| comfort | 0.9999 | 1.0000 | 0.9999 |
| **score** | **0.8563** | **0.8343** | **0.8405** |

### 실패 장면 수 비교
- **DAC 실패 (drivable area 이탈)**: v4 764 -> v1 1030 -> **v3 973**
- **TTC 실패**: v4 681 -> v1 761 -> **v3 720**
- **NC < 1 (충돌)**: v4 246 -> v1 268 -> **v3 244**
- **Score 0 장면**: v4 969 -> v1 1237 -> **v3 1162**

### Auxiliary Detection & Map mAP
- det mAP: 0.2918
- map mAP: 0.2637

---

## 2. v3 결과 분석 및 진단

v3에서 `anchor_log_attention`으로 확률 혼합 대신 로짓 덧셈 방식을 도입하여 anchor가 softmax 전체에 일관되게 작용하도록 개선하였다.

진단 기준 (`bev_selector_v3.md` §5):
1. **DAC 이탈 실패수**: 973장 (v1의 1030장 대비 개선됨).
2. **Score**: 0.8405 (v1 0.8343 대비 상승).
3. **다음 레시피 판단**:
   `bev_selector_v3.md` 163행 기준:
   "기하가 맞은 뒤에도 DAC가 910보다 줄었는데 764에 못 미치면, 다음 런은 이 문서의 distill을 유지한 채 GradBalancer만 0.40 / 0.30 / 0.30으로 켠다."

---

## 3. v4 개선 계획: GradBalancer (0.4 / 0.3 / 0.3) 활성화

### 변경 사항
- **GradBalancer 켜기**:
  - `agent.config.grad_balance_target='{"plan":0.4,"det":0.3,"map":0.3}'` 적용.
  - unscaled 상태에서는 det/map의 그래디언트가 shared BEV의 대부분을 차지하여, distill과 planner의 신호가 도로 경계와 궤적 조향에 온전히 전달되지 못함.
  - Closed-loop gradient balancing을 통해 plan 40% / det 30% / map 30%로 제어.
- **Student 및 Distill 유지**:
  - ResNet-34, BEV 50x100, 32 registers, per-cell LayerNorm MSE loss 유지.
  - Log-gaussian anchor 유지.

---

## 4. 실행 명령 (GPU 6, 7)

```bash
cd /workspace/byounggun/SSR
source env.vast.sh
CUDA_VISIBLE_DEVICES=6,7 nohup bash scripts/training/run_bev_selector_distill.sh \
  agent.config.grad_balance_target='{plan:0.4,det:0.3,map:0.3}' \
  wandb.tags='[bev-selector,r34,v4]' \
  > /workspace/byounggun/ssr_outputs/v4_train.log 2>&1 &
```
