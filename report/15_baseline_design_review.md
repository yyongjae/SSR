# PARA-SSR baseline 설계 제안 검토

2026-09-15. 현재 NAVSIM 구현을 기준으로 사용자가 제안한 여섯 항목을 검토했다.
아래는 최초 설계 피드백과 후속 코드 확인 결과다. 사용자 요청에 따라 SafeDrive/WoTE의
회귀 입력 정규화를 확인한 뒤 **final LayerNorm을 세 모드 공통으로 적용했다**.
이어 ego 속도·가속도도 planning query로 직접 전달하고 기존 BEV ego conditioning을
제거했다. 나머지 네 제안은 아직 적용하지 않았다.

**여섯 항목 모두 기본 실험을 시작하기 위한 필수 조건은 아니다.** Final LayerNorm과
ego status 직접 입력은 공통 planner를 보강할 만한 후보이고, 나머지는 interaction
memory 설계의 후속 ablation에 가깝다. 우선 세 모드의 공통 planner·입력·학습 조건을
맞춰 비교해야 interaction 효과와 다른 설계 변경의 효과를 구분할 수 있다.

| 제안 | 판단 | 현재 코드에 근거한 이유 |
|---|---|---|
| 마지막 final LayerNorm | **세 모드 공통 적용 완료** | SafeDrive/WoTE 모두 마지막 Post-LN으로 정규화된 decoder 출력을 회귀 head에 넣는다. 현재 Pre-LN planner에는 stack 뒤 별도 LN을 두어 회귀 입력을 정규화했다. 동일한 decoder 구조를 복제했다거나 성능 향상이 입증됐다는 의미는 아니다. |
| Ego 속도·가속도를 plan query에 직접 입력 | **세 모드 공통 적용 완료** | 이미 있던 입력을 초기 planning query의 MLP embedding으로 옮겼다. 기존 BEV ego MLP는 제거했다. 별도 geometry 경로는 후속 요청으로 이동·회전을 모두 보정하는 history BEV warp로 확장했다. 새 센서 입력을 추가한 것은 아니며 성능 효과는 실험으로 확인해야 한다. |
| Det confidence를 `log(score)` attention bias로 사용 | **후속 ablation 권장** | 높은 confidence를 우선하는 명시적 prior는 된다. 하지만 현재 key embedding도 attention logit에 직접 영향을 주므로 무조건 약한 신호라는 주장은 성립하지 않는다. 초기 또는 오분류한 detector의 낮은 score를 강하게 믿게 되는 대가가 있다. |
| Motion mode를 `softmax(traj_cls.detach())`로 가중 평균 | **후속 ablation 권장** | 가능성 높은 mode를 강조할 수 있지만 현 mode 분류는 sigmoid focal이다. Softmax는 기존 출력의 학습 의미를 그대로 보존하는 확률 해석이 아니며, 배경 query에도 가중치 합 1을 부여한다. |
| Map 위치를 20점 flatten MLP로 변경 | **지금 baseline에는 비권장** | 순서를 표현할 수 있지만 현 GT와 matcher가 같은 선의 역순·같은 ring의 시작점 변경을 허용한다. Flatten만으로 차선의 주행 방향을 얻지 못하고 임의의 출력 순서에 민감해진다. 계산량·파라미터도 동일하지 않다. |
| Det metadata에 yaw·velocity 추가 | **합리적인 후속 ablation** | Planner에 방향·상대운동 단서를 명시적으로 전달하는 장점이 있다. 현재 det/motion latent에도 관련 supervision이 있으므로 필수 누락은 아니다. 예측 오차와 단위·좌표계 정규화를 함께 다뤄야 한다. |

## Final LayerNorm과 ego status

최초 [planner](../navsim/agents/para_ssr/modules/planner_head.py)는 마지막 residual hidden을
바로 `ego_fut_decoder`로 보냈다. 현재는 세 층을 마친 뒤 `h = self.final_norm(h)`를 한 번
적용하고 회귀 head에 넣는다. Xiong 등의 Pre-LN 분석에서는 prediction 전 final
LayerNorm을 포함한다. 따라서 “전형적인 Pre-LN 구성에 가깝게 만든다”는 설명은 타당하다.
다만 해당 논문의 NLP 실험 결과가 이 모델의 3층 회귀 planner 성능 향상을 입증하는 것은
아니다. 여기서의 채택 권고는 구조를 바탕으로 한 판단이다.
[원 논문, Figure 1 및 본문](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf).

로컬 reference 코드 확인 결과는 다음과 같다.

- [WoTE](../navsim/agents/WoTE/WoTE_model.py)의 `offset_tf_decoder`는 PyTorch
  `TransformerDecoderLayer` 기본값 `norm_first=False`를 사용한다. 마지막 block은
  `norm3(x + FFN(x))`로 끝나고 `_predict_offset`은 그 출력을 바로 regression head에 넣는다.
  별도 `TransformerDecoder.norm`은 지정하지 않아 `None`이다.
- [SafeDrive 설정](../../SafeDrive/navsim/planning/script/config/common/agent/SafeDrive_Phase3_Planner_FullTrain.yaml)의
  ProposalNet trajectory decoder와 SWNet은 `self_attn → norm → cross_attn → norm → ffn → norm`
  순서다. [Motion_Decoder_Layer](../../SafeDrive/navsim/agents/safedrive/modules/decoder_motion.py)의
  마지막 LN을 통과한 출력이 `plan_reg_branch`로 전달된다.

즉 두 reference는 **Post-LN 내부에서 회귀 입력을 정규화**한다. PARA-SSR의 Pre-LN stack 뒤
final LN은 그 역할을 맞춘 변경이며, reference에 별도의 stack-final LN이 있다고 주장하는
것은 아니다. 추가 파라미터는 `2C`개(기본 C=256에서 512개)다. 이 파라미터가 없는 이전
checkpoint는 현재 strict loader에 바로 복원되지 않는다.

현재 [feature builder](../navsim/agents/para_ssr/para_ssr_features.py)는 다음을 이미 만든다.

- `status_feature`: command 4 + velocity 2 + acceleration 2.
- `ego_motion`: velocity는 `[5:7]`, acceleration은 `[7:9]`에 저장.
- 과거 [BEV encoder](../navsim/agents/para_ssr/modules/bevformer.py)의 ego-motion MLP는
  이 정보를 BEV query에 더했다. 현재 PARA-SSR에서는 이 MLP를 생성하지 않는다.
- 현재는 `status_feature`의 속도·가속도 네 값을 planner MLP로 encoding한 뒤
  learned query와 command를 결합한 초기 planning hidden에 한 번 더한다.

공식 NAVSIM도 velocity·acceleration·command를 입력으로 제공하며, TransFuser는 이를
encoding한 status token을 decoder memory에 넣는다. 이것은 planner에 직접 접근 경로를
제공하는 근거이지, “반드시 query에 더해야 한다”는 근거는 아니다.
[NAVSIM 입력 문서](https://github.com/autonomousvision/navsim/blob/main/docs/agents.md),
[공식 TransFuser 구현](https://github.com/autonomousvision/navsim/blob/main/navsim/agents/transfuser/transfuser_model.py).

이미 command embedding이 있으므로 current velocity·acceleration 4차원만
`Linear(4,C) → ReLU → Linear(C,C)`로 encoding했다. 속도·가속도는 NAVSIM의
`x_forward/y_left` 순서와 metric 단위를 그대로 사용하며 BEV `pc_range`로 정규화하지 않는다.
기존 command fuser의 입력 크기와 feature builder·캐시·target·출력 API는 유지했다.

직접 연결만 추가하는 경우 model/planner 두 파일 정도면 충분하지만, 실제 후속 요청은
BEV 경로를 없애는 것이므로 BEV transformer와 Python/Hydra 설정도 수정했다.
세 모드 모두 `use_ego_motion=false`이고 BEV ego MLP를 생성하지 않으며, encoder에
legacy `ego_motion` 전체를 전달하지 않는다. 별도 `bev_shift`와 캐시의 상대 yaw만 geometry로
전달해 과거 BEV를 현재 frame으로 이동·회전 정렬한 뒤 TSA에 넣는다. 따라서 yaw를
planning MLP에 넣지 않아도 BEV 회전 정렬에는 명시적으로 사용한다.
Perception도 입력 조건이 달라지므로 이전 실험과 단순 동일시할 수는 없고,
세 모드를 새 공통 baseline으로 비교해야 한다. 실제 성능 개선은 아직 측정하지 않았다.
[구현·검증](14_task_memory_planner.md#ego-status를-planning으로-이동).

## Confidence와 motion mode

현재 confidence key embedding은 무력한 신호가 아니다. Query/key 선형 projection을
포함하면 confidence의 logit 기여는 대략 다음과 같다.

```text
confidence의 기여 = (Wq · query)ᵀ (Wk · confidence_embedding(score)) / sqrt(d)
```

즉 크기와 방향을 학습하는 query-dependent 신호다. 반면 아래 방식은 score가 높을수록
항상 유리해지는 prior를 강제한다. 이는 attention의 정의에서 직접 유도되는 차이다.
Float attention bias를 softmax 전 score에 더하는 동작은
[PyTorch 공식 설명](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)과 같다.

```text
attention_i = softmax(content_logit_i + log(max(score_i, eps)))
            ∝ exp(content_logit_i) × max(score_i, eps)
```

모든 score가 같으면 공통 bias가 softmax에서 상쇄되므로, 장면 전체의 낮은 신뢰도를
자동으로 “det를 덜 읽는다”로 바꾸지는 못한다. 낮은 confidence의 실제 위험 객체도
억제할 수 있다. 시도한다면 기존 confidence embedding과 동시에 켜서 이중 효과를
만들기보다, embedding 방식 대 logit-bias 방식의 별도 비교가 해석하기 쉽다.
`eps`는 log(0) 방지이며 query 제거 threshold와 다르다.

“장면당 실제 객체가 약 3개”라는 전제는 이번에 검증되지 않았다.
[기본 설정의 기존 기록](../navsim/agents/para_ssr/configs/default.py)은 과거 navtrain
20-scene 조사에서 in-range agent median 53, max 80이라고 적고 있다. 이 기록도 현재
split·ROI·FOV 필터의 전체 분포를 다시 측정한 값은 아니다. 먼저 현재 GT 분포와 실제
attention의 foreground/background 질량을 확인해야 배경 분산을 주요 병목으로 단정할 수 있다.

[Motion loss](../navsim/agents/para_ssr/modules/det_motion_head.py)는 matched object의
마지막 유효 시점 FDE가 가장 작은 mode를 분류 target으로 삼고, **sigmoid focal loss**를
쓴다. Unmatched query는 모든 mode의 negative이며, matched query에 future GT가 없으면
해당 mode 분류 loss를 제외한다.

따라서 `softmax(traj_cls)` 가중 평균은 가능한 설계지만 calibrated categorical posterior의
기대값이라고 부를 근거는 없다. 모든 mode가 낮은 배경도 합 1로 정규화된다. Normalized
sigmoid 역시 합 1로 만들면 이 문제를 자동으로 해결하지 않는다. 초기 학습에는 mode
score가 부정확할 수 있고, 어느 방식이든 latent를 하나로 압축하므로 multimodality가
완전히 보존되는 것도 아니다. 지금은 단순 mean을 기준으로 두고 나중에 비교하는 편이 낫다.
변경 시 weight만 detach하고 `motion_hidden`은 계속 gradient를 받아야 한다.

## Map 순서와 det metadata

현재 map 클래스는 road·walkway·centerline·crosswalk다.
[Target builder](../navsim/agents/para_ssr/para_ssr_targets.py)의 `_equivalent_orders`와
[matcher](../navsim/agents/para_ssr/modules/losses.py)의 `hungarian_assign_map`은 open line의
forward/reverse, closed contour의 cyclic 시작점 변경을 같은 geometry로 취급한다.
이는 MapTR 계열의 permutation-equivalent map 표현과 연관된 설계다.
[MapTR 원 논문](https://arxiv.org/abs/2208.14437).

`mean(MLP(point))`는 point 순서에 invariant하지만, 단순 좌표 평균과는 다르다.
비선형 MLP 뒤 평균이므로 point-set의 공간 분포를 표현할 수 있다. 전체 map latent에
모든 형상 정보가 없다는 뜻도 아니다. Flatten은 순서를 구분할 수 있으나, 현재 GT의
동등 순서 중 어느 것을 decoder가 출력했는지에 따라 다른 metadata를 만든다.
특히 centerline reverse도 정답으로 인정하므로 첫 점→끝 점이 법적 주행 방향이라는
해석은 성립하지 않는다.

비용도 같지 않다. 좌표 MLP를 현재 `2→C→C`에서 제안 `2P→C→C`로 바꾼다고 가정하면:

| 항목 | 점별 MLP 후 평균 | Flatten 후 MLP |
|---|---:|---:|
| 곱셈-누산 수/instance, bias·activation 제외 | `P × (2C + C²)` | `2PC + C²` |
| `P=20, C=256`일 때 | 1,320,960 | 75,776 |
| MLP 파라미터 수, bias 포함 | 66,560 | 76,288 |

이는 해당 작은 metadata MLP만의 이론 계산이다. Flatten은 이 가정에서 연산은 줄지만
파라미터가 늘고 point 수에 종속된다. 전체 모델 latency 개선량은 측정하지 않았다.
Geometry 표현을 바꾸려면 방향·시작점 convention과 함께 검토해야 한다.

Det raw prediction code는
`[x, y, logW, logL, z, logH, sin(yaw), cos(yaw), vx, vy]`다.
Yaw는 이미 `[6:8]`의 sin/cos 표현이므로 다시 sin/cos를 취하면 안 된다. 예측된 두 값이
정확한 단위원 위에 있다는 보장도 없다. 속도 `[8:10]`는 metric velocity이고 SSR 축은
`x=right, y=forward`이므로 NAVSIM의 `x=forward, y=left`와 혼동하면 안 된다.
추가한다면 적절한 velocity scale을 별도로 정하고, 기존 좌표·score와 같이 예측 metadata
입력을 detach해야 한다. 현재 det의 box·velocity supervision과 motion latent도 이 정보를
학습하도록 되어 있어, XY-only metadata가 즉시 baseline을 무효화하는 결함은 아니다.

## 세 모드 비교에 대한 권고

1. 먼저 현재 planner를 공통으로 두고 interaction on / 병렬 multi-task / planning-only를
   같은 split, sensor, BEV resolution, horizon, optimizer, effective batch, precision,
   update 수, seed 정책으로 비교한다. 병렬 multi-task와 planning-only의 planner는 같아야 한다.
2. Final LayerNorm·direct ego를 common baseline에 채택하려면 세 모드에 동일하게 적용한
   새 실험으로 묶는다. 한 모드에만 추가한 결과를 interaction 효과로 해석하지 않는다.
3. Confidence bias, weighted mode pooling, map flatten, 추가 det metadata는 interaction-on
   기준에서 하나씩 비교한다. 여러 변경을 한 번에 적용하면 어떤 변경이 도움이 됐는지 알기 어렵다.
4. Planning score와 함께 det/motion/map 지표 및 시간·메모리 비용을 본다. Interaction은
   학습 때 supervision뿐 아니라 추론 때 private decoder 실행 비용도 추가한다.

실제 최적 구조나 성능 개선 여부는 학습·평가 결과로 판단해야 한다. 이 검토는 현재 코드,
원 논문·공식 구현, 명시한 수식에 근거한 설계 판단이며 장기 학습 ablation 결과가 아니다.

## FP16 설정이 보인 이유

이전 설정에는 진입점 간 차이가 있었다. `train_para_ssr.sh`와 `smoke_para_ssr.sh`는
`trainer.params.precision=32`를 명시했지만, NAVSIM 공통
[training config](../navsim/planning/script/config/training/default_training.yaml)는
`16-mixed`가 기본이었다. 따라서 해당 shell script를 쓰면 FP32, 직접
`run_training.py agent=para_ssr_agent`만 실행하면 FP16 mixed precision을 상속했다.
이는 PARA-SSR interaction 구조에 맞춰 FP16을 선정했다는 의미가 아니다.

이번 변경은 PARA-SSR에 `training_precision=32`를 두고 공통 config가 agent별 값을
선택하도록 해 두 진입점의 기본값을 일치시킨다. Agent별 값이 없는 다른 모델에는 기존
공통 기본값을 유지한다. 세 실험은 동일한 precision으로 비교해야 한다.

이전 [task-memory 검증](14_task_memory_planner.md)의 저정밀도 probe는 **BF16**이며,
FP16 전체 학습 검증이 아니었다. BF16과 FP16을 같은 실행 형식으로 설명하면 안 된다.
보고서에 기록된 실제 NAVSIM smoke도 FP32였고, 전체 FP16/GradScaler 학습은
당시 미검증 범위였다.
