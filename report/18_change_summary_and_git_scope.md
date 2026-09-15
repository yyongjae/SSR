# PARA-SSR 변경 요약과 학습용 Git 포함 범위

2026-09-15. 사용자 요청에 따라 `para_ssr_git_paths.txt`를 **학습 실행용 변경 파일
16개**로 축소했다. 현재 `/home/external-user/yongjae/SSR`의 기존 repository/HEAD 위에
누적 변경을 적용해 interaction / parallel / planning-only 세 모드를 실행하는 범위다.
이 16개 파일만 별도 빈 폴더로 옮겨 실행하는 standalone 배포 목록은 아니다.
기존 추적 중인 agent, target builder, training entrypoint, 라이브러리 및 데이터는 필요하다.

`tests/`는 단위·통합·회귀·스모크 검증용이다. 실제 학습 경로에서 해당 테스트를
import하거나 실행하지 않으므로 학습용 변경분 업로드에서 제외할 수 있다.
학습 중 validation loop는 `navsim/planning/training`의 런타임 코드에 있으며
`tests/`나 별도 auxiliary evaluation 실행기에 의존하지 않는다.

## Git에 포함할 파일 (16개)

현재 staging 목록에는 아래 모델·설정 12개와 공통 launcher + 세 모드 launcher 4개만 있다.

### 모델·loss·설정 (12개)

- [`navsim/agents/para_ssr/configs/default.py`](../navsim/agents/para_ssr/configs/default.py)
- [`navsim/agents/para_ssr/modules/bevformer.py`](../navsim/agents/para_ssr/modules/bevformer.py)
- [`navsim/agents/para_ssr/modules/det_motion_head.py`](../navsim/agents/para_ssr/modules/det_motion_head.py)
- [`navsim/agents/para_ssr/modules/grad_balance.py`](../navsim/agents/para_ssr/modules/grad_balance.py)
- [`navsim/agents/para_ssr/modules/map_head.py`](../navsim/agents/para_ssr/modules/map_head.py)
- [`navsim/agents/para_ssr/modules/planner_head.py`](../navsim/agents/para_ssr/modules/planner_head.py)
- [`navsim/agents/para_ssr/modules/temporal_alignment.py`](../navsim/agents/para_ssr/modules/temporal_alignment.py)
- [`navsim/agents/para_ssr/para_ssr_features.py`](../navsim/agents/para_ssr/para_ssr_features.py)
- [`navsim/agents/para_ssr/para_ssr_loss.py`](../navsim/agents/para_ssr/para_ssr_loss.py)
- [`navsim/agents/para_ssr/para_ssr_model.py`](../navsim/agents/para_ssr/para_ssr_model.py)
- [`navsim/planning/script/config/common/agent/para_ssr_agent.yaml`](../navsim/planning/script/config/common/agent/para_ssr_agent.yaml)
- [`navsim/planning/script/config/training/default_training.yaml`](../navsim/planning/script/config/training/default_training.yaml)

### 세 모드 학습 실행 스크립트 (4개)

- [`scripts/training/train_para_ssr.sh`](../scripts/training/train_para_ssr.sh)
- [`scripts/training/train_para_ssr_interaction.sh`](../scripts/training/train_para_ssr_interaction.sh)
- [`scripts/training/train_para_ssr_parallel.sh`](../scripts/training/train_para_ssr_parallel.sh)
- [`scripts/training/train_para_ssr_plan_only.sh`](../scripts/training/train_para_ssr_plan_only.sh)

## Stage 명령

```bash
cd /home/external-user/yongjae/SSR
git add --pathspec-from-file=report/para_ssr_git_paths.txt
git diff --cached --name-only
git diff --cached --stat
git diff --cached --check
```

목록 파일 자체도 학습 필수 파일이 아니므로 staging 목록에서 제외했다. 로컬에 있는
목록 파일을 `git add`의 입력으로 읽는 데에는 문제가 없다. Stage/commit/push는 수행하지 않았다.
이미 다른 파일을 stage한 상태라면 위 명령이 이를 자동으로 제거하지는 않는다.

## 이전 전체 목록에서 제외한 항목 (33개)

파일을 삭제하지 않고 공유 목록에서만 제외했다.

| 구분 | 수 | 제외 이유 |
|---|---:|---|
| `tests/` | 16 | 단위·통합·회귀·스모크 검증; 학습 runtime에서 사용하지 않음 |
| `tools/verify_task_memory_cuda.py` | 1 | 별도 CUDA 검증 probe |
| `navsim/planning/script/run_aux_evaluation.py` | 1 | 별도 detection/map 평가와 평가 재개 수정; 학습/학습 중 validation에는 불필요 |
| `train_para_ssr_det_motion_plan.sh`, `train_para_ssr_map_plan.sh` | 2 | 추가 single-head ablation; 현재 요청한 세 모드에 불필요 |
| `docs/`, `report/`, 구조도 및 목록 파일 | 13 | 설명·검증 기록·그림; 학습 runtime에서 사용하지 않음 |

기존 추적 중인 테스트/문서를 repository에서 지우는 명령은 아니다. 이번 변경분을 공유할 때
새 테스트와 테스트 변경을 포함하지 않는다는 뜻이다. 이후 전체 테스트까지 공유하려면
관련 테스트 변경도 함께 반영해야 현재 구조의 기대값과 일치한다.

## 별개의 변경 및 이전 도구


| 파일 | 이유 |
|---|---|
| `navsim/agents/WoTE/WoTE_agent.py` | WoTE scheduler의 PyTorch 버전 호환 수정. PARA-SSR 구조 변경과 별도 commit 권장. |
| `scripts/evaluation/eval_arm.sh` | 이전 LiDAR/4-arm 실험명과 head flag를 사용하는 wrapper. 현재 세 실험용으로 검증되지 않음. |
| `scripts/evaluation/eval_arm_aux.sh` | 이전 실험명 기반 auxiliary 평가 wrapper. 현재 실험 설정과 별도 검토 필요. |
| `tools/count_aux_gt.py` | Auxiliary GT 통계 도구. 이번 architecture 구현의 의존성이 아님. |
| `tools/rescore_teacher_detection.py` | Teacher detection 재채점 도구. 이번 architecture 구현의 의존성이 아님. |
| `tools/verify_geometry_and_loss.py` | 이전 LiDAR 포함 geometry/overfit probe. 기본 ParaSSRConfig()와 LiDAR 입력을 가정하는 경로가 있어 현재 camera-only 기본값에서 재검증 필요. |
| `tools/verify_lidar_pipeline.py` | 이전 LiDAR 전체 pipeline probe. 기본 ParaSSRConfig()를 사용하므로 LiDAR 활성화를 명시하고 현재 planner/precision과 함께 재검증 필요. |

제외는 삭제를 뜻하지 않는다. 현재 파일은 그대로 두고 별도 commit에서 다룬다.
`work_dirs/`, `data/`, checkpoint, feature/metric cache, TensorBoard/W&B 로그는 목록에 없다.
문서의 과거 검증 로그 링크가 가리키는 로컬 산출물도 이 Git 묶음에는 포함하지 않는다.


## 누적 주요 수정사항 (참고)


1. **Task-memory planner:** 기존 det/motion·map decoder의 마지막 latent를 planner에서
   읽는다. 각 층은 `BEV CA → (plan-det CA ∥ plan-map CA) → residual 합산 → FFN`이며,
   서로 다른 파라미터로 3층을 쌓는다. 두 task CA의 Q는 동일한 post-BEV hidden에서
   각자 query LN을 거쳐 만든다. Pre-LN residual과 최종 LayerNorm을 사용한다.
2. **Memory 구성:** Det는 300개 object slot의 det latent와 6-mode motion latent 평균을
   projection 후 결합한다. Map은 100개 instance마다 20개 point latent를 평균한다.
   모든 slot을 사용하며 memory를 forward당 한 번 만들어 세 층에서 재사용한다.
3. **Metadata와 gradient:** Box 중심·polyline 점·foreground score는 detach한 뒤 MLP로
   encoding해 key에만 한 번 넣는다. Latent는 detach하지 않는다. Planning loss는
   det/motion·map decoder까지 전달되고, private decoder 사이의 새 forward 연결은 없다.
4. **Planning 입력:** 기존 learned BEV ego-motion conditioning을 제거했다.
   Command one-hot 4차원은 별도의 `Embedding(4,256)`을 통해 planning query와 결합하고,
   `[vx,vy,ax,ay]` 4차원은 별도 MLP를 거쳐 더한다. 따라서 총 8개 입력 성분을 사용하지만
   8차원 전체를 하나의 MLP로 보내는 구조는 아니다. Command conditioning 경로는 재사용했다.
5. **Temporal BEV 정렬:** Frame 간 `(dx,dy,d_yaw)`로 과거 BEV 전체를 현재 frame에
   `grid_sample` 정렬한 뒤 TSA에 전달한다. 18차원 legacy vector의 상대 yaw 필드는
   이 기하학적 정렬에 사용한다. 18차원 전체 정보를 8차원으로 압축한 구조는 아니다.
6. **입력/토큰:** 기본 `use_lidar=false`, `use_stl=false`. 카메라와 learned BEV query를
   사용하고 Scene TokenLearner/scene-token 경로를 실행하지 않는다. Dense BEV token과
   object/map/planning query는 계속 사용한다.
7. **학습·추론·실험 모드:** Interaction on에서는 학습과 추론 모두 det/motion·map
   decoder 및 prediction branch를 한 번 실행한다. GT와 matching은 loss 경로에만
   필요하다. Interaction / interaction 없는 병렬 multi-task / planning-only를 지원한다.
8. **GradBalancer·수치 처리:** Planning의 direct-BEV 및 det/map 경유 gradient를 모두
   planning task로 측정한다. Auxiliary 계수는 loss 출처별로 shared BEV에 적용한다.
   GradBalancer 학습은 FP32를 요구한다. History autocast cache, AMP head loss와
   camera projection의 dtype, nonfinite controller norm, 평가 fingerprint도 수정했다.

Trajectory head의 8-step/4-second horizon, `[B,8,3]` 최종 출력, 기존 evaluation API와
loss weight는 유지한다. `[B,4,8,3]`은 command로 조건화한 한 trajectory를 기존 API의
branch 축에 expand한 것이다.

공유용 짧은 설명:

> 카메라 기반 공유 BEV 위에 기존 det/motion·map head를 유지하고, 각 head의 마지막
> latent를 memory로 사용하는 3층 Pre-LN planner를 구성했다. 각 층에서 BEV를 먼저
> 읽은 동일 planning hidden으로 det/map cross-attention을 병렬 계산하고 residual에
> 합산한 뒤 FFN을 적용한다. Command와 속도·가속도는 planning query를 조건화하고,
> ego pose 변화는 과거 BEV의 이동·회전 정렬에 사용한다. LiDAR와 Scene TokenLearner는
> 비활성화했다. 세 학습 모드를 지원하며 planning loss의 전체 gradient 경로에 맞춰
> GradBalancer를 수정했다.


## 확인 범위

- Runtime package와 학습 entrypoint의 Python import를 정적으로 확인했으며 `tests/`,
  `test_para_ssr*`, `pytest`에 대한 직접 의존성을 찾지 못했다.
- 새 runtime 의존성 `modules/temporal_alignment.py`를 목록에 포함했다.
- 파일 16개 모두 존재하고 중복·ignored 경로가 없음을 확인했다.
- 모델 실행 코드는 바꾸지 않았고, 학습/테스트를 새로 실행하지 않았다.
- 이전 병렬 planner의 전체 회귀 결과는 **379 passed**였으며, 이는 제외한 테스트 변경까지
  있는 로컬 상태에서 수행한 결과다. 테스트 변경을 뺀 Git 공유본의 별도 테스트 결과는 아니다.
