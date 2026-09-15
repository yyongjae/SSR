# PARA-SSR 누적 변경 코드 자체 검토

2026-09-15. Task-memory planner, 세 학습 모드, final LN, planning ego 입력,
temporal SE(2) 정렬까지의 누적 변경과 그 호출 경로를 다시 검토했다.
현재 NAVSIM 코드가 대상이며, 이전 nuScenes 구현이나 별개 WoTE 수정은 대상이 아니다.

이 문서는 det→map 순차 planner를 사용하던 감사 시점의 기록이다. 이후 task attention을
병렬 분기로 바꿨으며, 현재 구조는 [17번 문서](17_architecture_guide.md), 최신 변경 및
Git 포함 범위는 [18번 문서](18_change_summary_and_git_scope.md)를 따른다.

## 결론

기본 FP32에서 요청 구조를 위반하는 추가 dependency, tensor layout 오류, 회전 부호 오류는
발견하지 못했다. 다만 저정밀도 학습과 평가 결과 재사용 경로에 실제 문제가 있었고 수정했다.
그중 새 shared-gradient 보정의 AMP 수치 문제는 알고리즘을 재설계하지 않고 **FP32 전용으로
명시적으로 제한**했다. 기본 학습 설정은 이미 FP32이므로 세 baseline 실행 명령은 그대로다.

기존 테스트를 다시 실행하는 데 그치지 않고, 수치 미분, 독립 world-pose 계산,
실제 AMP loss/backward, 실제 모델 NCCL, 기본 해상도 학습으로 검증 범위를 넓혔다.

## 발견한 문제와 조치

### 1. 과거 frame의 AMP weight cache가 현재 encoder gradient를 끊음

`obtain_history_bev`는 `no_grad`에서 현재 frame과 같은 encoder를 호출한다. 이때 외부
autocast context가 캐시한 저정밀도 weight가 현재 frame에서도 재사용되면, 현재 frame의
graph까지 weight gradient가 끊겼다. 세 모드에서 재현했으며, 작은 planning-only 모델의
CPU BF16 실험에서는 학습 대상 backbone/FPN/BEV projection 파라미터 39개가 `grad=None`이었다.

- **영향:** AMP 학습. FP32 기본 학습에서는 재현되지 않았다. 기존 history 처리 문제다.
- **수정:** history 계산 중 autocast의 weight caching만 끈다. 외부 autocast의 dtype과 enabled
  상태는 유지한다. 성공·예외 모두 `finally`에서 기존 cache flag와 train/eval 상태를 복구한다.
- **검증:** 세 모드 × CPU BF16/CUDA FP16/CUDA BF16의 2 optimizer step에서 모든 학습 대상
  파라미터에 유한한 gradient가 존재했다. CUDA FP16은 GradScaler도 사용했다.
  이 AMP 검증에서는 아래 3번의 이유로 GradBalancer를 껐다.

### 2. Det/map의 실제 AMP supervision loss에서 dtype 충돌

저정밀도 prediction으로 만든 target buffer에 FP32 캐시 GT를 indexed assignment하면서
det loss가 실패했다. Map에도 같은 패턴이 있었다. 기존 CUDA probe는 GT 없는 inference와
planning backward만 수행하므로 이 경로를 검증하지 못했다.

- **영향:** FP16/BF16 auxiliary supervision. 기존 head loss 문제다.
- **수정:** matching, target buffer, loss reduction 입력을 최소 FP32로 승격한다.
  FP64는 보존하고 prediction cast는 autograd에 연결한다. Latent와 forward 출력은 그대로다.
- **검증:** empty/nonempty GT 및 FP16/BF16/FP64에서 같은 값을 가진 고정밀도 prediction과
  loss·gradient가 일치했다. 실제 agent의 matching→loss→backward→optimizer도 실행했다.

### 3. 새 GradBalancer 보정은 AMP에서 작은 planning gradient를 보존하지 못함

새 방식은 원래 backward에 `(scale - 1) * auxiliary_VJP`를 더한다. 수학적으로는 맞지만
저정밀도 decoder 내부에서 큰 auxiliary gradient와 작은 planning gradient가 먼저 합쳐질 때
정보를 잃는다. BEV boundary나 최종 loss만 FP32로 바꿔서는 회복되지 않는다.

기본 head 크기(C256, BEV5000, det300, map100×20)로, 각 task의 VJP를 독립적으로 구한
가중합과 실제 보정 gradient를 비교했다. 이 수치 실험의 image 입력은 64×96이고 LiDAR는 없다.
수정한 정상 map target fixture로도 다시 확인했다.

| 실행 | aux 계수 | 상대 L2 오차 | 방향 cosine |
|---|---:|---:|---:|
| FP32 | 0.001 | 0.0455% | 0.9999999 |
| FP32 | 0 | 0.1503% | 0.9999989 |
| BF16 | 0.001 | 483% | 0.186 |
| BF16 | 0 | 1490% | 0.075 |

- **영향:** 이번에 도입한 gradient correction의 AMP 사용. FP32에서도 유한정밀도 상쇄 오차는
  있으며 위 수치는 해당 probe 결과이지 모든 학습 상태의 오차 상한은 아니다.
- **조치:** GradBalancer가 켜진 학습은 warm-up부터 FP32를 요구한다. 수동 non-neutral
  auxiliary coefficient도 같은 제약을 적용한다. 원본 prediction dtype과 autocast 상태를
  검사하므로 loss만 FP32로 cast하는 우회도 정상 사용 경로에서 거부한다.
- **가능한 실행:** 기본 FP32 balancing, AMP inference, balancing을 끄고 수동 계수를 1로 둔
  AMP 학습. 마지막 경우는 `agent.config.grad_balance_target=null`로 설정한다.
- **한계:** AMP balancing을 고친 것은 아니다. 별도 설계와 수치 검증 전까지 지원하지 않는다.

[최신 수치 로그](../work_dirs/code_audit_validation/grad_precision_fixed_fixture.log),
[probe](../work_dirs/code_audit_validation/probe_grad_precision.py).
Probe는 잘못된 결과를 재현하기 위해 새 저정밀도 거부 검사만 실험 안에서 일시 우회한다.

### 4. 카메라 투영의 FP32 cast가 외부 autocast를 막지 못함

`point_sampling`의 행렬 입력이 FP32여도 `matmul`은 외부 autocast 때문에 FP16/BF16으로
실행됐다. 실제 calibration과 BEV 50×100에서 FP16은 비유한 투영 좌표 29,987개 및
visibility 판정 차이 8개, BF16은 판정 차이 68개와 최대 정규화 좌표 오차 0.04238을 보였다.
FP16 비유한 좌표는 이 표본에서 모두 visibility 필터 밖이었다.

- **영향:** AMP의 camera geometry. 기존 투영 문제다.
- **수정:** BEV reference grid를 처음부터 FP32로 만들고, 전체 투영·나눗셈을 autocast가
  꺼진 FP32 영역에서 실행한다.
- **검증:** 같은 calibration에서 두 저정밀도 모드 모두 비유한 좌표 0개, visibility 차이 0개,
  FP32 대비 투영 차이 0이었다. FP32 기존 결과는 bitwise 동일했다.

[실제 calibration 비교](../work_dirs/code_audit_validation/projection_precision.json).

### 5. NaN/Inf norm이 controller 상태를 영구 오염시킴

기존 `GradBalancer.update`는 NaN norm을 받은 뒤 scale이 NaN으로 남아, 다음 정상 측정에도
회복되지 않았다. 이제 nonfinite/음수 norm 또는 역산한 nonfinite norm이면 해당 update
전체를 건너뛰어 기존 scale과 첫 측정 상태를 보존한다. 다음 정상 update의 동작을 테스트했다.
이는 controller의 영구 오염을 막는 처리이며, nonfinite model gradient 자체를 복구하지는 않는다.

### 6. 평가 재개 시 새 model helper 변경을 감지하지 못함

Auxiliary evaluator의 고정 source 목록에 새 `temporal_alignment.py`와 기존 LiDAR helper 등이
빠져 있었다. 이 파일만 바꾸면 checkpoint/config/token이 같을 때 이전 token record를
잘못 재사용할 수 있었다.

`navsim/agents/para_ssr` 전체 Python source를 자동으로 fingerprint하도록 바꿨다.
Temporal/LiDAR/future helper 수정 각각에서 기존 manifest 재사용이 거부되는 것을 확인했다.
따라서 수정 전 코드의 평가 폴더는 새 실행에 재사용하지 말고 새 experiment 폴더를 사용한다.

추가로 통합 테스트의 synthetic map GT Y가 metric 값 `2..10`인 오류를 `[0,1]` 계약에 맞췄다.
실제 target builder는 이 문제의 원인이 아니었다. 학습 wrapper의 오래된 BEV query 수 주석도
현재 기본값 5,000으로 고쳤다.

## 구조와 흐름 재확인

| 항목 | 재검토 결과 |
|---|---|
| Planner | STL/gate 없음, 독립 파라미터 3층, 최신 hidden으로 BEV→det→map→FFN, final LN |
| Memory | 마지막 det/motion/map latent layout과 query/mode/instance/point 순서 일치 |
| Pooling/metadata | 요청한 mean 유지, 좌표 한 번 정규화, metadata 입력 detach, latent 연결, key에 한 번 주입 |
| 세 모드 | Interaction은 planning→private decoder gradient, parallel은 shared BEV만 공유, planning-only는 auxiliary 모듈/target/balancer 생략 |
| Forward/eval | 활성 head는 필요한 forward에서 한 번, GT 없이 trajectory 생성, matcher는 loss에서만 실행 |
| Ego status | 속도·가속도는 planning query, command는 기존 별도 embedding, pose는 temporal geometry |
| Temporal geometry | current→previous SE(2), ego 원점, row-major grid, metric/normalized 좌표, 중복 shift 제거 일치 |
| Gradient | Planning의 direct + det/motion + map 경로 모두 planning으로 측정, private supervision 계수 유지 |
| 외부 계약 | Trajectory shape/horizon/누적 offset, 두 evaluation API, 같은 구조 checkpoint strict round trip 유지 |

독립 planner probe에서 B=2 결과와 별도 B=1 결과가 일치했고, instance/mode/point 순열의
예상 불변성을 확인했다. BEV·det·motion·map latent 방향 미분은 FP64 수치 미분과 약
`1e-10` 절대 오차로 일치했다.
[결과](../work_dirs/code_audit_validation/planner_heads_probe.json).

회전 정렬은 사용 가능한 1,310개 trainval log 중 12개를 표본으로 선택해, 96개 연속 frame 쌍,
유효 셀 40,291개를 절대 world pose 행렬로 독립 계산했다. 최대 좌표 오차는 `2.373e-7 m`였다.
표본 log 내부 camera calibration 변화는 `1e-6`보다 작았다. 전체 데이터의 calibration
불변성을 증명한 것은 아니다.
[결과](../work_dirs/code_audit_validation/temporal_world_pose_oracle.json).

## 실행 검증

실행 환경은 Python 3.9, PyTorch 2.8.0+cu128, RTX 5090이다.
최종 전체 pytest는 **373 passed, 259 warnings, 28.96초**였다.
[전체 로그](../work_dirs/code_audit_validation/pytest.log).

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1 \
  /home/external-user/miniconda3/envs/ssr/bin/python -m pytest -q tests --disable-warnings
```

새 회귀 테스트에는 AMP loss/gradient/상태 복구 29개, camera projection 8개,
balancing precision/controller state 44개, 평가 재개 3개가 포함된다.

실제 작은 ParaSSRAgent의 **2 GPU NCCL 검증도 exit 0**이었다. 세 모드 각각 처음부터
non-neutral 계수(det=0.001, map=0.002)로 시작해, 2 microbatch accumulation/`no_sync`와
2 optimizer step을 실행했다. 모든 학습 대상 파라미터의 gradient가 유한했고, 매 step
양 rank의 파라미터는 bitwise 일치했다. Controller state도 microbatch마다 일치했다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NCCL_SOCKET_IFNAME=lo NCCL_DEBUG=INFO \
  /home/external-user/miniconda3/envs/ssr/bin/python \
  work_dirs/code_audit_validation/probe_grad_ddp.py
```

[최종 NCCL 로그](../work_dirs/code_audit_validation/grad_ddp_final_loopback.log),
[probe](../work_dirs/code_audit_validation/probe_grad_ddp.py).
재검증 중 기본 network interface에서는 NCCL bootstrap 연결이 거절되어 DDP 생성이
실패했고 forward/loss는 실행되지 않았다. 앞서 통과한 실행과 같은 단일 호스트 loopback
설정으로 재실행해 통과했다. 정확한 방화벽/라우팅 원인은 확인하지 않았으며, 이 loopback
설정은 여러 호스트에 그대로 적용할 수 없다.
[실패 로그도 보존](../work_dirs/code_audit_validation/grad_ddp_final.log)했다.

기본 해상도 실제 NAVSIM interaction FP32 학습은 **exit 0**이었다.
Front 3 camera 768×416, sparse LiDAR, BEV 50×100, encoder 3층, C256,
det300/motion6/map100×20, GPU당 batch 4, accumulation 1로 3 optimizer step과
validation 1 batch를 실행했다. Pretrained backbone weight 다운로드는 끄고 검증했다.
둘째 loss에서 산출한 `det=0.00057978`, `map=0.00854561`을 셋째 loss에 적용했으므로,
계수 1일 때뿐 아니라 추가 auxiliary VJP가 실행되는 구간도 확인했다.

[로그](../work_dirs/code_audit_validation/full_resolution_balanced.log),
[Hydra config](../work_dirs/code_audit_validation/full_resolution_balanced/code/hydra/config.yaml),
[계수와 실행 요약](../work_dirs/code_audit_validation/full_resolution_balanced/validation_summary.json).
이 짧은 실행의 시간이나 loss 감소를 throughput·수렴·성능 비교로 해석하지 않는다.
변경한 training wrapper의 `bash -n`과 `git diff --check`도 통과했다.

## 이번 검토에서 수정한 파일

- `navsim/agents/para_ssr/para_ssr_model.py`: history autocast cache 격리와 상태 복구.
- `navsim/agents/para_ssr/modules/bevformer.py`: FP32 camera geometry와 reference grid.
- `navsim/agents/para_ssr/modules/det_motion_head.py`, `map_head.py`: supervision loss dtype.
- `navsim/agents/para_ssr/modules/grad_balance.py`, `para_ssr_loss.py`: balancing precision 계약과 nonfinite controller 처리.
- `navsim/planning/script/run_aux_evaluation.py`: package source fingerprint.
- `tests/test_para_ssr_mixed_precision.py`, `test_para_ssr_projection_precision.py`: 신규 회귀 테스트.
- `tests/test_para_ssr_planning_gradients.py`, `test_para_ssr_grad_balance_state.py`,
  `test_para_ssr_task_memory_integration.py`, `test_aux_evaluation_runner.py`: guard/state/fixture/resume 회귀 테스트.
- `scripts/training/train_para_ssr.sh`: BEV query 수 주석 정정.
- 이 문서, `report/14_task_memory_planner.md`, `report/README.md`, `docs/PARA_SSR_NAVSIM.md`: 검증 범위와 precision 안내.

## 남아 있는 한계

- 장기 학습, 세 모드의 PDMS/EPDMS/auxiliary mAP 비교는 실행하지 않았다.
- 기본 크기 모델의 2 GPU 장기 학습, 기본 accumulation 16의 throughput/peak memory benchmark는
  수행하지 않았다. 기본 batch 4의 단일 GPU 실행과 작은 모델의 NCCL 검증은 구분해야 한다.
- AMP+GradBalancer는 지원하지 않는다. Balancing을 끈 AMP 검증은 작은 camera 모델이며,
  production 크기 sparse LiDAR AMP 학습의 안정성을 입증하지 않는다.
- 구 PyTorch autocast API fallback을 추가했지만 실제 PyTorch 2.0 runtime 재실행은 하지 않았다.
  기존 third-party deprecation/spconv warning도 남아 있다.
- SE(2) 정렬은 ego motion만 보상한다. 움직이는 객체의 잔여 운동은 TSA가 처리해야 하며,
  이전 ROI 밖에서는 zero padding 때문에 history가 없다. 긴 history는 반복 resampling으로
  흐려질 수 있다. History의 `no_grad` 정책도 유지하므로 과거 activation까지 학습하지 않는다.
- 평균 pooling의 정보 손실, private decoder에서 planning/auxiliary gradient 충돌,
  interaction 모드의 추론 비용은 기존 설계 tradeoff로 남는다.
