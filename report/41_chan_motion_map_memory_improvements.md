# chan: Motion·Map memory 개선 구현 명세

- 작성·수정일: 2026-10-14
- 작업 브랜치: `chan`
- 기준 커밋: `4f29592`
- 문서 상태: 구현을 다른 AI agent에 맡기기 위한 명세. 문서 작성 시점에는 모델 변경과 성능 실험을 수행하지 않았다.

## 1. 구현 agent에게 전달할 작업

이 문서를 읽고 아래 두 변경을 구현하고 검증한다. 필수 구현은 **Motion B: 객체 × 미래 mode별 token 유지**와 **Map A: 지도 요소당 4개 구간 token**이다. 각각 독립적으로 켜고 끌 수 있어야 한다. 두 옵션을 모두 켠 구성을 지원한다.

원래 의도는 기존 head가 만든 여러 미래 후보와 지도 세부 형상을 planner가 더 구체적으로 읽게 하는 것이다. Perception head를 새로 만들거나 teacher를 추가하는 작업이 아니다. 실제 궤적·지도 좌표를 평균내는 문제가 아니라, planner 입력 hidden feature의 압축 수준을 바꾼다.

모델의 기본 설정은 기존 결과를 재현할 수 있도록 기존 memory 구성을 유지한다. 사용자는 명시적인 config override로 개선을 활성화한다. 구현 agent의 완료 범위는 코드, 테스트, 실행 설정 예시와 검증 결과 기록이다. 이 문서만으로 장시간 학습·최종 test 평가·원격 push를 실행하지 않는다.

현재 branch와 작업 트리를 확인하고 기존 사용자 변경을 보존한다. 작성 시점에 `projects/configs/SSR/PARA_SSR_e2e_30ep_plan_det_mot.py`가 untracked였으며, 이 작업의 수정 대상이 아니다. 작업 위치는 NAVSIM PARA-SSR이다. `projects/mmdet3d_plugin/SSR`의 별도 구현을 대신 수정하지 않는다.

## 2. 현재 구조와 변경 파일

현재는 공유 BEV에서 Det+Motion과 Map head를 실행한 뒤, 그 마지막 decoder feature를 planner에 넘긴다. Planner 각 layer는 BEV를 먼저 읽고, 객체 memory와 지도 memory를 각각 attention으로 읽어 update를 합친다. 이 구조와 3개 planner layer는 유지한다.

| 파일 | 필요한 변경 |
|---|---|
| [configs/default.py](../navsim/agents/para_ssr/configs/default.py) | 아래 두 config 필드와 설명 추가 |
| [para_ssr_agent.yaml](../navsim/planning/script/config/common/agent/para_ssr_agent.yaml) | 같은 필드와 기본값을 Hydra config에 추가 |
| [para_ssr_model.py](../navsim/agents/para_ssr/para_ssr_model.py) | config를 `ParaSSRPlannerHead` constructor로 전달 |
| [modules/planner_head.py](../navsim/agents/para_ssr/modules/planner_head.py) | 옵션 저장·검증과 `prepare_task_memories()` 분기 구현 |
| [para_ssr_agent.py](../navsim/agents/para_ssr/para_ssr_agent.py) | `_validate_config()`에서 옵션 검증 추가 |
| [test_para_ssr_config.py](../tests/test_para_ssr_config.py) | config validation 테스트 확장 |
| `tests/test_para_ssr_task_memory.py` | memory 의미·순서·gradient·호환성 테스트 신규 작성 |

필요한 기존 출력은 [det_motion_head.py](../navsim/agents/para_ssr/modules/det_motion_head.py)와 [map_head.py](../navsim/agents/para_ssr/modules/map_head.py)에 이미 있다. 두 head의 forward 출력, perception loss, 학습 target 형식은 유지한다. `PlanTaskMemoryLayer`의 attention은 memory 길이가 달라도 동작하므로 attention 모듈을 교체할 필요가 없다.

## 3. 고정 config 계약

정확히 아래 이름을 사용한다.

```python
# ParaSSRConfig: planning-head 설정 옆에 추가
plan_det_memory_mode: str = "mean"    # mean | mode_tokens
plan_map_memory_segments: int = 1    # 1은 기존 동작, 첫 개선 실험은 4
```

```yaml
# para_ssr_agent.yaml의 config 아래 planner 설정에 추가
plan_det_memory_mode: mean
plan_map_memory_segments: 1
```

`ParaSSRPlannerHead.__init__()` 끝에 같은 이름·기본값의 keyword parameter를 추가한다. 기존 positional constructor 호출의 의미가 변하지 않게 한다. Model에서 다음 값을 전달한다.

```python
plan_det_memory_mode=getattr(cfg, "plan_det_memory_mode", "mean"),
plan_map_memory_segments=getattr(cfg, "plan_map_memory_segments", 1),
```

Agent validation에서도 같은 `getattr` fallback을 사용한다. 기존 테스트의 `SimpleNamespace`나 예전 config가 필드를 갖지 않아도 기본 동작을 한다.

검증 규칙:

- `plan_det_memory_mode`는 `mean`, `mode_tokens`만 허용한다. 잘못된 값은 `ValueError`이며 메시지에 필드 이름을 포함한다.
- `plan_map_memory_segments`는 bool이 아닌 양의 정수여야 한다. `True`, `False`, 0, 음수, 실수·문자열을 정수로 조용히 변환하지 않는다.
- Interaction이 켜졌을 때 config의 point 수 `P`에 대해 `1 <= S <= P`, `P % S == 0`을 검증한다. 실제 memory 구성 시에도 실제 tensor의 `P`로 재검증한다.
- Constructor에서도 mode와 segment 값 자체를 검증한다. Constructor에는 `P`가 없으므로 나눗셈 검증은 memory 구성 시 한다.
- `use_task_interaction=False`이면 두 설정의 형식은 검증하되 memory를 만들거나 요구하지 않는다. 유효한 설정값을 저장해도 forward는 기존 BEV-only 경로로 동작한다.

설정 조합은 다음 네 가지가 필수다.

| 조합 | `plan_det_memory_mode` | `plan_map_memory_segments` |
|---|---|---|
| Baseline | `mean` | 1 |
| Motion only | `mode_tokens` | 1 |
| Map only | `mean` | 4 |
| Motion + Map | `mode_tokens` | 4 |

## 4. 입력·출력 tensor 계약

`B`: batch, `Q`: 객체 query 수, `M`: motion mode 수, `V`: 지도 요소 수, `P`: 요소당 point 수, `S`: 구간 수, `L=P/S`: 구간당 point 수, `C`: planner embedding 차원.

기본 구성은 `Q=300`, `M=6`, `V=100`, `P=20`, `C=256`이다. 구현은 이 수를 하드코딩하지 않고 실제 tensor shape에서 읽는다. Detection·motion·map 입력 feature의 채널 수가 다른 경우에도 기존 projection이 각각 planner 차원으로 변환한다.

필수 입력:

| 입력 | 형태 |
|---|---|
| `det_out["det_hidden"]` | `[B,Q,C_det]` |
| `det_out["motion_hidden"]` | `[B,Q,M,C_motion]` |
| 마지막 `det_out["all_bbox_preds"]` | `[B,Q,D]`, 첫 두 값은 metric XY |
| 마지막 `det_out["all_cls_scores"]` | `[B,Q,K_det]` |
| `map_out["map_point_hidden"]` | `[B,V,P,C_map]` |
| 마지막 `map_out["all_map_pts_preds"]` | `[B,V,P,2]`, 이미 normalized XY |
| 마지막 `map_out["all_map_cls_scores"]` | `[B,V,K_map]` |

기존 rank·batch·객체 수 검증을 유지하고 metadata의 batch/query/point 수 불일치도 명확한 `ValueError`로 처리한다. `M`, `V`, `P`는 양수여야 한다. 정상적인 비연속 tensor도 지원하도록 `reshape`, `flatten` 등을 사용한다.

반환 dictionary 이름을 바꾸지 않는다.

| key | 객체 계열 형태 | 지도 계열 형태 |
|---|---|---|
| `det_memory` / `map_memory` | `[B,Q 또는 Q*M,C]` | `[B,V*S,C]` |
| `det_position` / `map_position` | memory와 동일 | memory와 동일 |
| `det_confidence` / `map_confidence` | memory와 동일 | memory와 동일 |

Memory·position·confidence의 token 순서가 반드시 같아야 한다. 첫 구현에는 별도 padding mask, top-k selection, hard confidence cutoff를 추가하지 않는다.

## 5. 개선 1의 정확한 연산

### 5.1 기존 `mean` 분기

기존 수식과 연산 순서를 유지한다.

```python
det_memory = self.det_memory_norm(
    self.det_projection(det_hidden)
    + self.motion_projection(motion_hidden.mean(dim=2))
)
```

기존 위치·confidence는 다음 수식이다.

```python
det_xy = det_out["all_bbox_preds"][-1, ..., :2].detach()
origin = det_xy.new_tensor(self.pc_range[:2])
extent = det_xy.new_tensor((self.real_w, self.real_h))
det_xy = (det_xy - origin) / extent
det_score = det_out["all_cls_scores"][-1].detach().sigmoid().amax(-1, keepdim=True)
det_position = self.det_position_encoder(det_xy)
det_confidence = self.det_confidence_encoder(det_score)
```

### 5.2 新 `mode_tokens` 분기

객체 feature를 각 mode에 더하고, **mode별로 LayerNorm을 적용한 뒤** flatten한다.

```python
det_content = self.det_projection(det_hidden).unsqueeze(2)  # [B,Q,1,C]
motion_content = self.motion_projection(motion_hidden)    # [B,Q,M,C]
det_memory = self.det_memory_norm(det_content + motion_content)
det_memory = det_memory.flatten(1, 2)                      # [B,Q*M,C]

# 위의 기존 정규화·detach 수식으로 만든 [B,Q,C] metadata를 확장한다.
det_position = det_position.unsqueeze(2).expand(-1, -1, M, -1).flatten(1, 2)
det_confidence = det_confidence.unsqueeze(2).expand(-1, -1, M, -1).flatten(1, 2)
```

정확한 token 인덱스는 `i = q*M + m`이다. 객체 0의 mode 0..M-1 다음에 객체 1의 mode 0..M-1이 온다. 같은 객체의 mode는 현재 위치·객체 confidence가 같고, content만 mode별로 다르다.

Mode 번호는 학습된 후보의 번호이며 직진·정지 같은 고정 semantic label이 아니다. Mode별 hidden feature를 그대로 전달하는 것이 이번 필수 변경이다. Mode probability embedding, weighted mean, 미래 XY·시간 embedding, 추가 mode embedding은 이번 구현에 넣지 않는다. 이를 섞으면 평균 제거 자체의 효과를 분리하기 어렵다.

## 6. 개선 2의 정확한 연산

### 6.1 기존 `S=1` 분기

기존 식을 명시적인 분기로 유지한다. 기본 설정의 수치 재현을 위해 일반 구간 코드로 불필요하게 재작성하지 않는다.

```python
map_memory = self.map_memory_norm(
    self.map_projection(map_point_hidden.mean(dim=2))
)
map_xy = map_out["all_map_pts_preds"][-1].detach()
map_score = map_out["all_map_cls_scores"][-1].detach().sigmoid().amax(-1, keepdim=True)
map_position = self.map_position_encoder(map_xy).mean(dim=2)
map_confidence = self.map_confidence_encoder(map_score)
```

### 6.2 구간 `S>1` 분기

연속된 point 인덱스를 겹치지 않는 같은 크기의 구간으로 나눈다. 기본 개선 구성은 `S=4`, `L=5`다.

```python
B, V, P, _ = map_point_hidden.shape
S = self.plan_map_memory_segments
L = P // S  # 앞서 P % S == 0 검증

segment_hidden = map_point_hidden.reshape(B, V, S, L, -1).mean(dim=3)
map_memory = self.map_memory_norm(self.map_projection(segment_hidden))
map_memory = map_memory.flatten(1, 2)  # [B,V*S,C]

# 먼저 각 point의 위치를 embedding하고, 그 다음 구간 안에서 평균낸다.
point_position = self.map_position_encoder(map_xy)  # [B,V,P,C]
map_position = point_position.reshape(B, V, S, L, -1).mean(dim=3).flatten(1, 2)

instance_confidence = self.map_confidence_encoder(map_score)  # [B,V,C]
map_confidence = instance_confidence.unsqueeze(2).expand(-1, -1, S, -1).flatten(1, 2)
```

`encoder(mean(XY))`로 바꾸지 않는다. 기존 위치 encoder는 비선형이므로 `mean(encoder(XY))`와 다르다. Map 좌표는 이미 normalized이므로 metric 변환이나 재정규화를 하지 않는다.

정확한 token 인덱스는 `j = v*S + s`이다. 구간 `s`는 point 인덱스 `[s*L, (s+1)*L)`를 포함한다. 기본 설정에서 구간은 0–4, 5–9, 10–14, 15–19다. 요소 0의 네 구간 다음에 요소 1의 네 구간이 온다.

추가 segment 번호·instance 번호 embedding, 방향 feature, 전체 요소 feature를 더하는 residual, point sorting, closed-polyline 재배열은 넣지 않는다. 현재 head의 point 순서를 그대로 사용한다. 따라서 구간은 같은 point 수를 가질 뿐 물리적으로 같은 길이라는 뜻이 아니다.

일반 config는 `S=P`도 허용한다. 이는 구간당 point 하나인 point-token 비교를 같은 코드로 지원하기 위한 것이며, 첫 필수 실험은 `S=4`다. 초안 경로 기반 attention은 후속 연구로 둔다.

## 7. Gradient와 checkpoint 계약

- `det_hidden`, `motion_hidden`, `map_point_hidden`에는 detach를 추가하지 않는다. Planning loss가 이 content 경로를 통해 두 perception decoder와 공유 BEV에 전달된다.
- 예측 box XY, map XY, classification score는 기존 위치에서 detach한다. Metadata encoder의 파라미터에는 gradient가 흐르지만 metadata의 원래 prediction tensor에는 흐르지 않는다.
- 반환 memory 전체를 `no_grad`로 만들거나 cache로 대체하지 않는다. Device·dtype는 기존 모듈과 tensor를 따른다. CPU 이동이나 강제 float32 변환을 새로 넣지 않는다.
- 기존 projection, LayerNorm, 위치·confidence encoder를 그대로 재사용한다. 이 변경은 **새 trainable parameter, persistent buffer, state-dict key를 추가하지 않는다.**
- 기존 checkpoint 로더의 strict 검사를 완화하지 않는다. 같은 기존 모델 옵션을 가진 checkpoint는 네 memory 조합 모두에서 parameter shape/key가 같아 strict loading이 가능해야 한다.
- Checkpoint tensor에는 memory 옵션이 저장되지 않는다. 학습·평가 시 같은 옵션을 명시하고 resolved config를 보관한다. 기존 checkpoint를 새 옵션으로 평가하면 출력은 달라질 수 있으며, 이는 로딩 호환성과 별개다.
- 이 문서의 첫 성능 비교는 같은 초기화·학습 레시피로 재학습하는 실험이다. 예전 checkpoint의 옵션만 바꿔 채점한 결과를 재학습 효과로 보고하지 않는다.

BEV 직접 attention, task attention의 병렬 합산, planner layer 수, v1 trajectory decoder, v2 anchor planner, trajectory 선택식, refiner, loss·grad-balancer, 카메라·BEV 범위는 유지한다. 두 memory 변경은 v1/v2 공통의 memory 생성 위치에 적용한다.

## 8. 필수 테스트와 검증 방법

새 테스트는 실제 `ParaSSRPlannerHead`를 작은 차원으로 만들어 `prepare_task_memories()`와 `plan_from_bev()`를 검증한다. `transformer=torch.nn.Identity()`, 작은 BEV와 `embed_dims=32`, `num_heads=4`를 사용할 수 있다. 큰 backbone·실제 데이터·teacher cache 없이 CPU에서 실행 가능해야 한다. Config import에 필요한 기존 NAVSIM dependency는 프로젝트 환경을 사용한다.

| 필수 항목 | 구체적인 판정 |
|---|---|
| 기본값 호환 | 기존 수식을 테스트에 독립적으로 계산해 여섯 반환 tensor를 비교한다. 기본값과 explicit `mean/1`도 같은 값이다. Dropout 없는 eval에서 같은 입력의 planner 출력이 기준선과 허용 오차 내에서 같다. 가능하면 수정 전 커밋의 고정-seed 출력도 별도 작업 디렉터리에서 비교한다. |
| 조합·shape | 네 조합에 대해 content·position·confidence shape가 같고 각각 Q 또는 Q*M, V 또는 V*S다. 기본 Q/M/V/P와 작은 가변 shape를 포함한다. |
| Motion 순서·정보 유지 | Q=2, M=3 등으로 mode별 서로 다른 feature를 넣고 각 output token이 해당 (q,m)의 독립 수식과 일치함을 확인한다. Metadata도 같은 q가 연속 반복된다. 전체 mode를 같은 값으로 두면 같은 객체의 mode memory는 같아야 한다. |
| Map 구간 의미 | 각 구간의 hidden과 XY를 다르게 설정해 구간별 pooling을 독립적으로 검증한다. 한 구간 입력 변경이 해당 memory token에만 반영됨을 확인한다. 비선형 위치 encoder로 embedding 후 평균 순서를 검증한다. |
| Gradient | 실제 plan output의 비대칭 scalar loss로 backward하고 세 hidden 입력과 BEV에 finite·nonzero gradient가 있는지 확인한다. Synthetic 입력에서는 metadata prediction을 hidden에서 독립적인 leaf로 만들어 XY·score leaf에 gradient가 없는지 확인한다. |
| 비연속 입력 | transpose/slice 등으로 만든 정상 shape의 비연속 hidden·metadata도 동작한다. |
| 오류 입력 | 잘못된 mode, bool/0/음수/float segment, P와 나누어떨어지지 않는 S, S>P, 불일치 metadata shape가 의미 있는 오류를 낸다. |
| Interaction off | det/map 출력 없이 기존 BEV-only forward가 동작하고 새 memory 옵션으로 task 모듈이 생기지 않는다. |
| Checkpoint | 네 옵션 간 `state_dict` key/shape가 동일하고 strict load가 성공한다. v1과 v2 각각 동일 architecture 안에서 검사한다. |
| v1/v2 smoke | v1 단일 query와 임시 anchor 파일을 사용한 v2 다중 query에서 결합 설정의 forward/backward가 유한하고 출력 API·shape가 유지된다. |
| Config 연결 | Dataclass/YAML 기본값 일치와 Hydra override compose를 확인하고, model이 constructor에 두 설정을 실제 전달하는지 검사한다. |

Gradient가 없는 metadata prediction이라도 실제 decoder가 content와 parameter를 공유하면 그 parameter는 content 경로로 업데이트될 수 있다. Test는 독립 leaf를 써서 경로 차단을 판정해야 한다.

기존 환경을 먼저 확인한다. 이 서버에는 `/home/chanyoung/conda-envs/ssr-navsim`이 있지만 dependency 가용성은 별도 확인한다. 아래는 해당 환경이 사용 가능한 경우의 테스트 명령이다.

```bash
/home/chanyoung/conda-envs/ssr-navsim/bin/python -m pytest -q tests/test_para_ssr_task_memory.py tests/test_para_ssr_config.py tests/test_para_ssr_anchor_planner.py
```

관련 테스트를 통과한 뒤 추가 코드 변경이 없으면 불필요하게 반복 실행하지 않는다. Dependency나 데이터가 없어 실행하지 못한 검증은 이유와 명령을 결과 기록에 남긴다. 테스트 통과와 EPDMS 향상을 구분한다.

## 9. 실행 설정과 성능 비교

아래는 구현 후 사용할 Hydra override 예시다. 신규 필드를 YAML에 정의하므로 `+` prefix 없이 사용한다.

```text
# Baseline
agent.config.plan_det_memory_mode=mean agent.config.plan_map_memory_segments=1
# Motion only
agent.config.plan_det_memory_mode=mode_tokens agent.config.plan_map_memory_segments=1
# Map only
agent.config.plan_det_memory_mode=mean agent.config.plan_map_memory_segments=4
# Motion + Map
agent.config.plan_det_memory_mode=mode_tokens agent.config.plan_map_memory_segments=4
```

기존 [train_para_ssr.sh](../scripts/training/train_para_ssr.sh)에 override를 전달할 수 있다. 예시는 기본 v1에 대한 시작점이며, 데이터 경로·GPU·epoch는 학습 요청 시 확정한다.

```bash
SSR_NAVSIM_PYTHON=/home/chanyoung/conda-envs/ssr-navsim/bin/python \
EXPERIMENT=chan_motion_map \
bash scripts/training/train_para_ssr.sh \
  agent.config.use_task_interaction=true \
  agent.config.use_det_motion_head=true \
  agent.config.use_map_head=true \
  agent.config.plan_det_memory_mode=mode_tokens \
  agent.config.plan_map_memory_segments=4
```

W&B 기록 (2026-10-14 추가): `train_para_ssr.sh`는 기본으로 W&B를 켠다 (`WANDB=0`으로 끔, 로그인은 `~/.netrc`, entity는 개인 공간 `chandsgdfkdsjvl`로 고정, project `para-ssr`; 팀 `Robust_Ex`에는 올라가지 않음). 각 run은 `wandb.name=${EXPERIMENT}`로 올라가고, run config에 `agent`·`dataloader`·`trainer` 설정이 그대로 들어가며, 태그에 `backbone:<arch>`·`det_mem:<mode>`·`map_seg:<S>`가 자동으로 붙는다. 네 조합은 대시보드에서 이 태그 또는 `agent.config.plan_det_memory_mode`/`plan_map_memory_segments`로 필터한다. 조합마다 `EXPERIMENT` 이름을 다르게 주고, 같은 비교군은 `WANDB_GROUP`을 공유한다. 구현은 `navsim/planning/training/wandb_logging.py`, 테스트는 `tests/test_wandb_logging.py`.

v2 실험은 기존 anchor config·파일·reward target을 그대로 사용해 네 조합을 비교한다. v1 PDMS 결과와 v2 EPDMS 결과를 같은 기준으로 섞지 않는다.

[eval_para_ssr.sh](../scripts/evaluation/eval_para_ssr.sh)는 현재 기본 PDM 평가 진입점이다. 이를 실행했다고 EPDMS를 계산한 것으로 간주하지 않는다. EPDMS 성능 실험에는 실제 사용하는 공식 평가 버전·진입점·metric cache·traffic policy·구성 지표를 확인하고 네 조합에 동일하게 적용한다. 평가 옵션이 checkpoint 학습 옵션과 일치해야 한다.

기준선과 각 개선을 독립적으로 비교한 뒤 결합 효과를 본다. Same seed, 데이터 분할, 초기화, 학습 step·유효 배치, LR schedule, loss·gradient balancing, 평가 버전을 맞춘다. 신규 teacher cache는 필요하지 않다.

| 변경 | 기대 효과 | 함께 확인할 부작용 |
|---|---|---|
| Motion mode별 token | 움직이는 객체 충돌·TTC 실패 감소 | 불필요한 감속·진행도 감소 |
| Map 구간별 token | 커브·경계의 도로 이탈·DAC 실패 감소 | 지도 오류 민감도·경로 흔들림·승차감 저하 |
| 결합 | 두 효과의 상호 보완 | 점수 상쇄·추론 비용 증가 |

선택은 log 단위 held-out validation에서 하고 최종 test는 확인에 사용한다. 유망한 결과는 여러 seed와 장면별 paired 비교·log 단위 불확실성으로 재확인한다. EPDMS 총점과 실제 평가 버전의 구성 지표, 움직이는 객체 충돌·도로 이탈 세부 결과, GPU memory와 latency를 함께 기록한다.

Motion token은 기본 300→1,800, Map token은 100→400이지만 전체 모델 비용이 그 비율만큼 증가한다는 뜻은 아니다. Benchmark를 할 경우 같은 batch·dtype·device, eval/inference mode와 warm-up을 사용하고 GPU synchronization을 맞춘다. Benchmark는 학습·평가 요청과 분리한다.

## 10. 후속 연구와 이번 작업의 경계

다음은 첫 두 변경의 효과를 본 뒤 검토할 후보이며 이번 구현의 필수 작업이 아니다.

- Motion score 가중 평균, mode별 score embedding, 미래 XY·시간 embedding.
- Map 전체 요소 residual, point-token(S=P) 성능 실험, 초안 궤적 주변 soft attention.
- 새로운 safety loss, KD, refiner, hard query selection 또는 카메라 확장.

현재 가설은 feature 압축이 planning 활용의 병목일 수 있다는 것이다. 이를 확인하지 않고 신규 loss나 다른 구조까지 동시에 넣지 않는다. 성능 향상이나 예상 상승폭은 보장하지 않는다.

## 11. 완료 조건과 agent 결과 보고

구현 완료 시 다음을 확인하고 사용자에게 보고한다.

- [ ] `chan`에서 작업했고 기존 사용자 파일을 보존했다.
- [ ] Dataclass·YAML·validation·model·head까지 두 설정을 연결했다.
- [ ] 기본 설정은 기존 memory 수식·출력·state-dict 계약을 유지한다.
- [ ] Motion mode token과 map 4구간 token의 순서·metadata가 정확하다.
- [ ] Hidden gradient와 metadata detach를 유지한다.
- [ ] 네 옵션 조합과 interaction-off, v1/v2 경로를 검증했다.
- [ ] 관련 테스트 결과와 실행하지 못한 검증을 기록했다.
- [ ] 새 파라미터 없이 strict checkpoint 호환성을 검증했다.
- [ ] 변경 파일·핵심 동작·검증 결과·실제 사용 override를 보고했다.
- [ ] 구현 검증만 했으면 EPDMS 성능은 미측정이라고 명시했다.

구현 후 이 문서 끝에 `구현 결과`를 추가해 변경 파일, 실제 테스트 명령·결과, 남은 제약을 기록한다. Commit이나 push는 별도 요청이 있을 때 수행한다.

## 12. 설계 근거

- [report 22](22_planning_readout_results.md): feature/readout KD와 planning 병목 진단. Feature KD가 해당 실험 조건에서 점수 향상으로 이어지지 않았다는 기록이다.
- [report 37](37_teacher_refiner_experiment_overview.md): 인지했지만 계획에 반영하지 못한 실패와 교정 KD 설계. Feature 평균화가 그 원인이라는 것은 아직 검증하지 않았다.
- [report 38](38_stageE_E2_results_analysis.md): 별도 교정기·GT 벌점·KD 실험의 PDMS 84.87→86.15 결과. 이 숫자는 본 두 변경의 효과나 EPDMS 상승폭의 근거가 아니다.

첫 변경은 teacher cache 없이 head-to-planner memory만 바꿔 검증한다. 구현 완료와 성능 개선 확인은 별도 단계다.

## 구현 결과

- 구현일: 2026-10-14, 브랜치 `chan`, 기준 커밋 `4f29592` 위 작업 트리 (미커밋). 기존 untracked 파일 `projects/configs/SSR/PARA_SSR_e2e_30ep_plan_det_mot.py`는 건드리지 않았다.
- EPDMS/PDMS 성능은 **미측정**이다. 아래는 구현 검증만의 결과이며 학습·평가는 실행하지 않았다.

### 변경 파일

| 파일 | 변경 |
|---|---|
| `navsim/agents/para_ssr/configs/default.py` | `plan_det_memory_mode="mean"`, `plan_map_memory_segments=1` 필드와 주석 추가 |
| `navsim/planning/script/config/common/agent/para_ssr_agent.yaml` | 같은 두 필드를 `plan_num_layers` 아래에 기본값으로 추가 (`+` prefix 없이 override 가능) |
| `navsim/agents/para_ssr/modules/planner_head.py` | `validate_plan_memory_options()` 추가, constructor keyword 2개 추가·검증·저장, `prepare_task_memories()`에 `mean`/`mode_tokens` 분기와 `S=1`/`S>1` 분기 구현, metadata shape 검증 추가 |
| `navsim/agents/para_ssr/para_ssr_model.py` | 두 설정을 `getattr` fallback으로 `ParaSSRPlannerHead`에 전달 |
| `navsim/agents/para_ssr/para_ssr_agent.py` | `_validate_config()`에서 형식 검증 + interaction on일 때 `map_num_pts_per_vec` 나눗셈 검증 |
| `tests/test_para_ssr_config.py` | 잘못된 mode/segment(bool, 0, 음수, float, 20%3≠0, S>P) 거부, 필드 없는 namespace·네 조합·S=P 허용 테스트 추가 |
| `tests/test_para_ssr_task_memory.py` | 신규. §8 표의 항목 전체 |

### 핵심 동작

- 기본값(`mean`/1)은 기존 수식을 그대로 두어 결과를 재현한다. 토큰 순서는 객체 `i = q*M + m`, 지도 `j = v*S + s`(구간 s = point `[s*L, (s+1)*L)`)이며 memory·position·confidence가 같은 순서다.
- `mode_tokens`는 mode별 LayerNorm 후 flatten, 지도 구간은 hidden을 구간 평균 → projection → LayerNorm, 위치는 point별 encoder 후 구간 평균, confidence는 instance 값을 구간에 복제한다.
- 새 파라미터·buffer·state-dict key가 없다. 옵션은 head의 일반 attribute로만 저장된다.
- `use_task_interaction=False`이면 옵션 형식만 검증하고 task 모듈을 만들지 않으며 BEV-only 경로 그대로다.

### 실행한 검증

```bash
cd ~/SSR
/home/chanyoung/conda-envs/ssr-navsim/bin/python -m pytest -q tests/test_para_ssr_task_memory.py tests/test_para_ssr_config.py tests/test_para_ssr_anchor_planner.py
# 155 passed in 14.08s
/home/chanyoung/conda-envs/ssr-navsim/bin/python -m pytest -q tests/test_para_ssr_model_invariants.py tests/test_para_ssr_plan_map.py tests/test_para_ssr_readout.py tests/test_para_ssr_checkpoint.py tests/test_para_ssr_front_cameras.py tests/test_para_ssr_features.py tests/test_para_ssr_targets.py tests/test_para_ssr_aux_metrics.py tests/test_para_ssr_grad_balance_state.py
# 98 passed in 21.88s
```

신규 테스트가 다루는 것: 기본값 vs 기존 수식의 여섯 tensor 완전 일치(atol=0)와 planner 출력 일치, 네 조합 × 세 shape(기본 Q=300/M=6/V=100/P=20 포함)의 shape, mode token 순서·정보 유지·동일 mode 동일 memory, 구간 pooling·encode-then-mean·한 구간 변경의 국소성, 네 조합 × v1/v2의 hidden/BEV gradient 존재와 XY·score leaf gradient 부재, 모든 (q,m)·(v,p)에 gradient 도달, 비연속 입력 일치, constructor/validation/실행 시 오류 입력, interaction off, v1/v2 head와 소형 `ParaSSRModel`의 state-dict key/shape 동일 및 strict load, v1/v2 결합 설정 forward/backward(v2 rescore 포함), dataclass·YAML 기본값 일치, Hydra compose override 네 조합, model→head 인자 전달.

추가로 수정 전 커밋 `4f29592`를 임시 worktree에 체크아웃해 같은 seed·같은 입력(Q=300, M=6, V=100, P=20, C=32)으로 여섯 memory tensor와 `ego_fut_preds`를 비교했다. 기본 옵션에서 모두 bit-identical(max |diff| = 0)이며 state-dict도 동일했다.

### 남은 제약

- 학습·EPDMS 평가·latency benchmark·원격 push·commit은 하지 않았다. §9의 override를 그대로 사용할 수 있다.
- 실제 데이터·GPU 경로는 테스트하지 않았다 (CPU 소형 head/model만). `projects/mmdet3d_plugin/SSR`는 수정하지 않았다.
