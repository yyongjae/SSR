# 실험 돌리는 법 (`scripts/experiment.sh`)

**현재 상태:** `eval`, `table`과 평가 규칙 고정은 아직 커밋에 들어 있지 않습니다. 이 문서의 평가 설명은 그 작업이 끝난 뒤의 기준입니다. GPU 학습도 아직 검증 전입니다.

학습, 스모크, 평가, 결과표를 이 스크립트 하나로 합니다. 설정은 `KEY=VALUE`로 넘기고, 폴더 이름은 설정에서 자동으로 만들어집니다.

```
bash scripts/experiment.sh <명령> [KEY=VALUE ...]
```

스크립트 맨 위에 설정 블록이 있고, 명령줄 값이 그 블록을 이깁니다. **명령줄로 넘기세요.** 블록을 고치면 git 트리가 dirty가 되고, `train`과 `eval`은 dirty 트리를 거부합니다.

서버 준비(데이터, 캐시, 환경)는 `docs/SETUP.md`에 있습니다. 경로는 `tools/release/paths.env`에 씁니다(`paths.env.example` 참고). 비워 둔 값은 기준 서버 경로를 씁니다.

## 1. 보통의 순서

```
bash scripts/experiment.sh check  DET_KD=off                # 30초, CPU. 설정이 말이 되는지 확인
bash scripts/experiment.sh smoke  DET_KD=off                # 수 분, GPU. 짧게 진짜로 돌려 봄. 메모리도 잼
bash scripts/experiment.sh train  DET_KD=off                # 약 20시간, GPU 8장
bash scripts/experiment.sh status DET_KD=off                # 진행 상황 (RUN=<이름> 으로도 됨)
bash scripts/experiment.sh eval   RUN=<이름> GPUS=0,1,2,3   # 약 15분 GPU + 25분 CPU 채점
bash scripts/experiment.sh table                            # 모든 run의 결과표
```

## 2. 명령

| 명령 | 하는 일 | 필요한 것 |
|---|---|---|
| `name` | 폴더 이름을 출력합니다. 아무것도 만들지 않습니다. | 없음 |
| `check` | 설정 조합 확인 → launcher 드라이런 → CPU에서 실제 agent 생성. 파라미터 수, target builder, 거부 사유를 출력합니다. 기본 레시피면 기준 run과 비교해 PASS/FAIL을 출력합니다. `<RUN_ROOT>/_check/` 아래에만 씁니다 (`CHECK_ROOT=`로 변경). | teacher run, 캐시의 메타 파일 |
| `smoke` | 같은 설정으로 2 epoch × 60 micro-batch, wandb 끔. `<RUN_ROOT>/_smoke/smoke_<이름>` 에 씁니다. 끝나면 `tools/release/smoke_check.py`로 확인합니다. `DRY=1`은 launcher 드라이런과 CPU 생성만 합니다. | GPU, 약 2.5 GiB |
| `train` | 학습을 시작합니다. 아래 거부 조건을 먼저 봅니다. `DRY=1`은 출력만 합니다. | GPU, 70 GiB 이상 |
| `resume` | `RUN=<이름>`. `last.ckpt`에서 이어갑니다. 설정 hash, GPU 수, commit이 같아야 합니다. 이전 `overrides.txt`는 보관합니다. | 그 run의 `settings.json` |
| `status`, `stop`, `wait` | 그 run의 상태 보기, 멈추기, 끝날 때까지 기다리기. | `RUN=<이름>` 또는 같은 설정 |
| `eval` | `RUN=<이름> GPUS=... [EPOCH=<n>] [EPDMS=0|1]`. 평가의 유일한 경로입니다. 다른 인자는 받지 않습니다. | GPU, 채점 트리, 캐시 |
| `table` | `<RUN_ROOT>/*/eval/*/summary.json`을 표로 출력합니다. | 평가 결과 |
| `assets` | 이 설정에 필요한 파일이 맞게 있는지 확인합니다 (`QUICK=1`은 빠른 확인). | 없음 |
| `teachers` | refiner teacher 두 개를 학습합니다 (`GPUS_DET=`, `GPUS_MAP=`, `WHICH=both|det|map`). 각 2.8시간, GPU 4장. | BEV 캐시 |
| `purge` | `RUN=<이름> CONFIRM=1`. `TRAIN_DONE`이 있는 run의 `ck_e2e2/{lab,rec}`만 지웁니다. 평가에 쓰지 않는 파일입니다. | 없음 |

`train`이 거부하는 경우:

- run 폴더가 이미 있음 → `resume`을 쓰세요.
- 커밋하지 않은 변경이 있음 → 커밋하거나 `ALLOW_DIRTY=1`. 이때 `git_dirty: true`가 기록되고 diff가 run 폴더에 저장됩니다.
- 디스크 여유가 `MIN_FREE_GB`(기본 70)보다 적음.
- 같은 `RUN_ROOT`에서 다른 학습이 돌고 있음 → `FORCE_CONCURRENT=1`. 메모리를 나눠 쓰게 됩니다.
- GPU가 1 GiB 이상 사용 중이거나, 학습 전 파일 확인(`verify_assets --quick`)이 실패함.

`check`, `smoke`, `name`은 dirty 트리에서도 돌아갑니다.

## 3. 설정

| 키 | 값 | 기본 | 이름에 들어가는 부분 | 뜻 |
|---|---|---|---|---|
| `REFINER` | on off | on | `ref1` `ref0` | refiner 전체를 켜고 끕니다. 끄면 labeler도 안 돌고 v2 head 궤적을 제출합니다. |
| `REFINER_KD` | on off score lat | on | `rkd1` `rkd0` `rkdS` `rkdL` | refiner로 가는 KD. `off`는 teacher를 아예 읽지 않습니다. `score`는 score KD만 남기고 lateral KD를 끕니다. `lat`는 lateral KD만 남기고 score KD를 끕니다. |
| `BEV_KD` | on off | on | `bkd1` `bkd0` | BEV KD v15 (student BEV로 가는 KD). |
| `DET_KD` | on off off_bev off_ref | on | `det1` `det0` `det0b` `det0r` | DET teacher에서 오는 KD. `off`는 두 경로 모두, `off_bev`는 BEV KD만, `off_ref`는 refiner KD만 끕니다. |
| `MAP_KD` | on off off_bev off_ref | on | `map1` `map0` `map0b` `map0r` | MAP teacher에서 오는 KD. 값의 뜻은 같습니다. |
| `BACKBONE` | r18 r34 r50 r101 r152 r50d convnext_tiny regnety_032 efficientnet_b0 | r34 | 맨 앞 | timm 이름과 필요한 플래그는 registry에 있습니다. swin, vit는 코드가 더 필요해서 거부합니다. |
| `DET_TEACHER` | registry 이름 | bevfusion | `<det>-<map>` | DET teacher. `bevformer`는 등록만 되어 있고 없어서, 무엇이 부족한지 출력하고 거부합니다. |
| `MAP_TEACHER` | registry 이름 | resmap | `<det>-<map>` | MAP teacher. `mask2map`도 같습니다. |
| `INTERACTION` | on off | on | `st` 안의 `noint` | task interaction. |
| `DET_HEAD` | on off | on | `nodet` | det head. off는 `INTERACTION=off`가 필요합니다. |
| `MAP_HEAD` | seg vector off | seg | `vecmap` `nomap` | map head. off는 `INTERACTION=off`가 필요합니다. |
| `FRAMES` | `2,3` `3` `0,1,2,3` | `2,3` | `f1` `f4` | 입력 프레임. |
| `UNDISTORT` | on off | on | `dist` | 렌즈 왜곡 보정. |
| `HEADING` | xy free | xy | `headfree` | heading을 경로에서 계산(xy)하거나 별도 head로 예측(free)합니다. |
| `PILLAR_Z` | `a,b` | `-1,3` | `z<a>_<b>` | 카메라 pillar 높이 범위(m). 소수점은 이름에서 `p`가 됩니다 (`z-1p5_2p5`). |
| `BEV_RANGE` | 64 32 | 64 | `r32` | 앞쪽 범위. 32는 `REFINER=off BEV_KD=off`일 때만 됩니다. 32 m teacher 캐시가 없기 때문입니다. |
| `BN` | moving frozen | moving | `bnm` `bnf` | backbone BatchNorm 통계. moving이 지금까지의 v7 숫자 뒤에 있는 동작입니다. 비교할 run들은 같은 값이어야 합니다. |
| `SEED` | 정수 | 0 | `s<N>` | 전역 seed, `ck_e2e2.seed`, `distill_probe_seed`. `warmup.sampler_seed`는 0 고정입니다 (미리 만든 label과 묶여 있음). |
| `EPOCHS` | 정수 | 30 | `e<N>` | trainer, `agent.max_epochs`, labeler 상한을 함께 바꿉니다. refiner를 켜면 6 이상이어야 합니다. |
| `GPUS` | 쉼표 목록 | `0,...,7` | `g<N>` | 2, 4, 8장. 번호 제한은 없고, 각 GPU 사용량이 1 GiB 미만이어야 합니다. 기준은 8장입니다. |
| `SAVE_EVERY` | 정수 | 1 | 없음 | N epoch마다 checkpoint. `EPOCHS`를 나누어떨어뜨려야 합니다 (평가는 마지막 epoch를 읽음). hash에도 안 들어갑니다. `resume`은 마지막으로 저장된 지점부터 이어갑니다. |
| `RUN_ROOT` | 경로 | `$SSR_RUN_ROOT` | 없음 | run 폴더 위치. |
| `TAG` | `[A-Za-z0-9.-]+` | 없음 | 접미사 | 같은 설정을 다시 돌릴 때 등. |
| `WANDB` | online offline disable | offline | 없음 | hash에 안 들어갑니다. 팀 계정은 `WANDB_ENTITY`로 지정합니다 (기본값 없음). |
| `EXTRA` | Hydra 줄, `;`로 구분 | 없음 | `_x` | 위 키에 없는 것을 바꿀 때. hash에 들어갑니다. |
| `DRY` | 0 1 | 0 | 없음 | 1이면 `train`, `smoke`가 시작하지 않고 출력만 합니다. |

명령별 옵션: `RUN`, `EPOCH`, `EPDMS`, `CONFIRM`, `QUICK`, `BASE`(표의 기준 run), `CHECK_ROOT`, `REGISTRY`(registry 파일 경로), `ALLOW_DIRTY`, `ALLOW_MIXED`, `ALLOW_COMMIT_CHANGE`, `FORCE_CONCURRENT`, `MIN_FREE_GB`, `MEM_LIMIT_GB`(기본 22.5), `SMOKE_KEEP`(0이면 통과 후 그 스모크의 ckpt, rec, lab 삭제), `GPUS_DET`, `GPUS_MAP`, `WHICH`. 그 외 키는 거부합니다.

`EXTRA`에서 거부하는 것:

- 고정된 값: `infer_select`, `submit_selection`, `kd_calib`, `teacher_`, `warmup.`, `distill_lambda_`, `distill_tau_`, `plan_anchor_file`, `dataloader.params.batch_size`, `io_dir`, `output_dir`, `seed`.
- 위 표의 키가 정하는 값 (`use_distill`, `ck_e2e2.enabled`, `image_architecture` 등). 이름과 실제가 달라지기 때문입니다. 해당 키를 쓰라고 안내합니다.

## 4. 폴더 이름

```
v2f_<bb>_<det>-<map>_ref<>_rkd<>_bkd<>_det<>_map<>_st<base|a.b>_bn<>_e<N>_g<N>_s<N>[_<TAG>][_x]_<hash6>
```

예: `v2f_r34_bevfusion-resmap_ref1_rkd1_bkd1_det1_map1_stbase_bnm_e30_g8_s0_c04aa7`

- 기본값도 생략하지 않습니다. 정렬과 grep이 그대로 되고, 기본값이 바뀌어도 이름의 뜻이 그대로입니다.
- `hash6`는 실제 Hydra override 전체(run 경로, wandb 줄 제외, 데이터 경로는 자리표시자로 바꿈)와 쓰는 teacher의 sha 고정값으로 계산합니다. commit은 넣지 않습니다. 그래서 같은 설정은 어느 서버에서든 같은 이름이고, 레시피 기본값이 코드에서 바뀌면 이름이 바뀝니다.
- 전체 기록은 `<run>/launch/settings.json`에 있습니다: 설정, override sha, commit, dirty 여부, GPU 목록, 호스트, python과 torch 버전, teacher 고정값, 파일 확인 결과. 이 파일의 설정으로 이름을 다시 만들 수 있습니다.

한 설정이 한 이름만 갖도록 다음을 정리합니다:

| 입력 | 결과 |
|---|---|
| `REFINER=off` | `rkd0`로 고정 |
| `REFINER_KD=off BEV_KD=off` | `det0 map0`로 고정 |
| `REFINER_KD=off DET_KD=off` (또는 `REFINER=off`) | `off`는 `off_bev`가 됩니다 (`det0b`). `off_ref`는 뜻이 없어 거부합니다. |
| `BEV_KD=off DET_KD=off` | `off`는 `off_ref`가 됩니다 (`det0r`). `off_bev`는 거부합니다. |
| `DET_KD=off MAP_KD=off` (KD가 하나라도 켜짐) | 거부합니다. `REFINER_KD=off BEV_KD=off`를 쓰세요. |
| 양쪽 BEV bank 또는 양쪽 refiner source를 모두 끔 | 거부합니다. 각각 `BEV_KD=off`, `REFINER_KD=off`를 쓰세요. |
| 아무도 읽지 않는 teacher | 이름에서 기본 teacher로 돌아갑니다. |

기본 레시피의 override는 `tools/release/reference/main_overrides.txt`(참조 run의 override, 경로만 자리표시자로 바꿈)와 다음 줄만 다를 수 있습니다. `check`가 이를 확인하고 PASS/FAIL을 출력합니다.

- run에서 나오는 값: `experiment_name`, `wandb.*`, `io_dir`, `output_dir`.
- KD calibration: `mode=online`, `det=null`, `map=null`, online 옵션 6개, `submit_selection=false`.
- `plan_anchor_file`: 저장소 안의 파일 (참조 run의 파일과 바이트가 같음).
- 새로 명시한 키의 기본값: `seed=0`, `backbone_bn_mode=moving`, `use_teachers=true`, `kd_sources=[det,map]`, `distill_off_banks=[]`, `distill_norm_files`.

## 5. 실험 목록

모든 줄 앞에 `bash scripts/experiment.sh train`을 붙이면 됩니다. 먼저 같은 인자로 `check`, 그다음 `smoke`를 돌리세요. 비교할 run들은 GPU 수와 `BN`을 같게 두세요.

`비용`의 기준: 8 GPU에서 30 epoch 약 20시간. 이 값은 참조 run 하나에서 측정했고, 나머지 줄은 그로부터의 추정입니다 (refiner를 끈 run의 시간을 측정한 적은 없습니다). 디스크는 run당 최대 약 60 GiB이고, `SAVE_EVERY=5`와 `purge`를 쓰면 끝난 뒤 약 3.5 GiB입니다. `필요한 것`은 `assets`가 확인하는 profile입니다 (표 아래 설명).

| 실험 | 인자 | 필요한 것 | 비용 |
|---|---|---|---|
| 기준 | (없음) | full | 20 h |
| refiner 끔 | `REFINER=off` | full | ≤ 20 h |
| refiner KD 끔 | `REFINER_KD=off` | full | 20 h 조금 아래 |
| score KD 끔 | `REFINER_KD=lat` | full | 20 h |
| lateral KD 끔 | `REFINER_KD=score` | full | 20 h |
| BEV KD 끔 | `BEV_KD=off` | full | 20 h |
| KD 전부 끔 | `REFINER_KD=off BEV_KD=off` | refiner_noteacher | 20 h 조금 아래 |
| refiner, KD 모두 끔 (baseline) | `REFINER=off BEV_KD=off` | base | ≤ 20 h |
| DET teacher 끔 (두 경로) | `DET_KD=off` | full | 20 h |
| MAP teacher 끔 (두 경로) | `MAP_KD=off` | full | 20 h |
| BEV KD에서만 DET 끔 | `DET_KD=off_bev` | full | 20 h |
| BEV KD에서만 MAP 끔 | `MAP_KD=off_bev` | full | 20 h |
| refiner KD에서만 DET 끔 | `DET_KD=off_ref` | full | 20 h |
| refiner KD에서만 MAP 끔 | `MAP_KD=off_ref` | full | 20 h |
| refiner 없이 BEV KD DET 끔 | `REFINER=off DET_KD=off` | full | ≤ 20 h |
| refiner 없이 BEV KD MAP 끔 | `REFINER=off MAP_KD=off` | full | ≤ 20 h |
| backbone r18 | `BACKBONE=r18` | full | 약 20 h |
| backbone r50 | `BACKBONE=r50` | full | 측정 안 됨. 메모리 smoke 필수 |
| backbone r101 | `BACKBONE=r101` | full | 측정 안 됨. 메모리 smoke 필수 |
| backbone r152 | `BACKBONE=r152` | full | 측정 안 됨. 메모리 smoke 필수 |
| backbone r50d | `BACKBONE=r50d` | full | 측정 안 됨 |
| backbone ConvNeXt | `BACKBONE=convnext_tiny` | full | 측정 안 됨. 단계 동결이 ResNet과 다름 |
| backbone RegNetY | `BACKBONE=regnety_032` | full | 측정 안 됨. 단계 동결이 ResNet과 다름 |
| backbone EfficientNet | `BACKBONE=efficientnet_b0` | full | 측정 안 됨. 단계 동결이 ResNet과 다름 |
| interaction 끔 | `INTERACTION=off` | full | 약 20 h |
| det head 끔 | `INTERACTION=off DET_HEAD=off` | full | 약 20 h |
| map head 끔 | `INTERACTION=off MAP_HEAD=off` | full | 약 20 h |
| head 둘 다 끔 | `INTERACTION=off DET_HEAD=off MAP_HEAD=off` | full | 약 20 h |
| vector map head | `MAP_HEAD=vector` | full | 약 20 h |
| 렌즈 보정 끔 | `UNDISTORT=off` | full | 20 h |
| pillar 높이 | `PILLAR_Z=-2,2` | full | 20 h |
| 프레임 1장 | `FRAMES=3` | full | 20 h 아래 |
| 프레임 4장 | `FRAMES=0,1,2,3` | full | 측정 안 됨. 프레임당 이미지 로드와 BEV 계산 추가 |
| heading 자유 | `HEADING=free` | full | 20 h |
| 32 m (KD, refiner 없음) | `BEV_RANGE=32 REFINER=off BEV_KD=off` | base | ≤ 20 h |
| BN 고정 | `BN=frozen` | full | 20 h |
| seed 반복 | `SEED=1` | full | 20 h |
| v1 planner | `REFINER=off BEV_KD=off EXTRA='agent.config.plan_anchor=false'` | base | ≤ 20 h |

| Profile | 무엇 | 디스크 |
|---|---|---|
| base | 데이터셋, 지도, anchor, `pdm_score_256`, 환경 | 약 693 GiB |
| refiner_noteacher | base + navtrain metric cache, GT 저장소, packed, warm-up label | 약 743 GiB |
| full | refiner_noteacher + BEV teacher 캐시 2개, refiner teacher run 2개 | 약 1.93 TiB |

거부되는 실험:

| 실험 | 이유 |
|---|---|
| `DET_TEACHER=bevformer`, `MAP_TEACHER=mask2map` | teacher 모델, 캐시(≥ 436 GB), refiner teacher가 없습니다. 아래 6절. |
| `BACKBONE=swin`, `BACKBONE=vit` | 코드 수정이 필요합니다 (입력 크기, 채널 순서, stride, 이미지 정규화). |
| `BEV_RANGE=32`와 KD 또는 refiner | 32 m teacher 캐시가 없습니다. |

## 6. teacher 추가

코드는 고치지 않습니다. 캐시와 `tools/release/registry.yaml`의 항목만 있으면 됩니다. registry의 항목이 경로, sha, norm, refiner teacher run을 정하고, `compose.py`가 이를 환경 변수와 override로 넘깁니다. 새 teacher는 다음이 필요합니다:

1. 같은 앞쪽 격자에서 NAVSIM으로 학습한 모델. 학습 token과 평가 token이 겹치면 안 됩니다.
2. 캐시. navtrain_train 85,109 token 이상, 256 채널, 100×100, float16, 현재 프레임 token을 key로. 없는 token은 그 token의 KD가 조용히 빠지므로, 시작 후 2000 token에서 비율이 0.999보다 낮으면 학습을 멈춥니다.
   - DET: `<dir>/manifest.json`, `<dir>/samples/<token[:2]>/<token>.npz` (`bev_feature`). 필수 항목은 `checkpoint_sha256_head`, `layout`, `target_bev_shape`, `bev_channels`, `point_cloud_range`, `to_student_transform`.
   - MAP: `<dir>/meta.json`, `index.json`, `bev/<shard>.npy` (+ seg). seg head가 없으면 registry에 `require_seg: false`.
   - 축은 측정해서 확인합니다. manifest 문구는 데이터와 대조되지 않습니다.
3. BEV KD용 norm 파일 (`mean`, `std`, 256 채널). 파일이 없으면 시작 때 거부합니다. 조용히 다른 teacher 통계를 쓰는 일은 없습니다.
4. 그 캐시로 학습한 refiner teacher run: `bash scripts/experiment.sh teachers DET_TEACHER=<이름> WHICH=det`.
5. registry 항목. 예시와 필드 설명은 `registry.yaml` 위쪽에 있습니다. sha는 따옴표로 묶으세요.

그다음 `check DET_TEACHER=<이름>`이 통과하면 됩니다. branch 이름 `bevfusion`, `resmap`은 코드에서 역할 이름이므로 그대로 둡니다.

## 7. 평가와 결과표

- 제출 규칙은 코드에 고정되어 있습니다 (`navsim/agents/para_ssr/ck/protocol.py`): `rule=pdms`, `set=all`, `lat_mode=on`, `beta=1.0`, refiner pick 제출. refiner를 끈 run은 v2 head를 제출하고 `selection: none`으로 기록합니다.
- 기본은 마지막 epoch입니다. `EPOCH=15`는 `epoch=14-*.ckpt`를 뜻하고, 표에 `*`로 표시되며 `epoch_rule: manual`로 남습니다.
- navtest를 볼 때마다 `<run>/eval/ledger.jsonl`에 한 줄이 쌓입니다.
- `table`은 GPU 수, commit, protocol 버전이 섞이면 거부합니다 (`ALLOW_MIXED=1`로 출력 가능). GPU 수가 다르면 optimizer step당 BatchNorm 갱신 횟수, GridMask, EMA가 달라져서 단일 요인 비교가 아닙니다.
- `vs base` 열은 기준 run(`BASE=<이름>`, 없으면 기본 레시피 run)과 같은 token에서 비교한 차이와 log 단위 bootstrap 95% 구간입니다. 구간이 0을 포함하면 표 아래에 경고가 나옵니다.

## 8. 아직 확인되지 않은 것

이 스크립트는 CPU에서만 검증했습니다. 다음은 GPU 없이 확인할 수 없어서 아직 돌려 보지 않았습니다:

- 실제 `smoke`와 `train` (DDP 포함). 온라인 calibration을 켠 학습은 이 서버에서 GPU로 돈 적이 없습니다.
- ResNet34 외 backbone의 GPU 메모리. r50 이상은 `smoke`로 `MEM_LIMIT_GB` 확인이 먼저입니다.
- DET 또는 MAP을 끈 arm의 DDP와 resume.
- 다른 서버에서의 전체 실행.
