# 새 서버 준비 순서

이 문서만 따라가면 새 서버에서 학습과 평가를 돌릴 수 있게 만드는 것이 목표다. 실험을 고르고 돌리는 방법은 [EXPERIMENTS.md](EXPERIMENTS.md), 숫자를 어떻게 믿고 무엇을 밝혀야 하는지는 [FAIRNESS.md](FAIRNESS.md)에 있다.

**주의: 이 순서를 실제 새 서버에서 처음부터 끝까지 돌려 본 적은 없다.** 스크립트는 `SETUP_DRY=1`로 명령만 찍어 확인했다. 실제로 실행해 확인한 것은 `repack_dataset.py`와 `verify_assets.py`뿐이다. 8절에 확인하지 못한 것을 적었다.

## 1. 무엇을 준비할지 먼저 정한다

| 프로필 | 돌릴 수 있는 실험 | 디스크 | 추가로 필요한 것 |
|---|---|---:|---|
| `base` | `REFINER=off BEV_KD=off` | 약 693 GiB | 데이터셋, 지도, 앵커, `pdm_score_256`, 평가용 원본 NAVSIM |
| `refiner_noteacher` | `REFINER=on REFINER_KD=off BEV_KD=off` | 약 743 GiB | 위 + navtrain metric cache, GT store, packed, warm-up 라벨 |
| `full` | 그 밖의 모든 실험 (기본 main 포함) | 약 1,979 GiB (1.93 TiB) | 위 + BEV teacher cache 2개, refiner teacher 2개 |

- 실행 한 번마다 추가로 최대 약 60 GiB가 필요하다(체크포인트 31개 15 GiB, `ck_e2e2` 45 GiB). `SAVE_EVERY=5`면 약 50 GiB다. `train`은 빈 공간이 70 GiB(`MIN_FREE_GB`)보다 적으면 거부한다.
- 선택 사항: navhard 32 GiB, `cache_val_100x100` 60 GiB(student에는 필요 없음), ReSMap `seg*` + `navtest` 약 190 GiB(읽는 코드 없음).
- 위 합계에는 EPDMS용 v2.2 navtest cache 3.6 GiB와 warm-up 입력 궤적 파일(`ck/raw256/traj_*.npy`) 2.4 GiB가 빠져 있다.

**기준 서버 사양:** RTX 4090 × 8 (24 GB), CPU 112코어 224스레드, RAM 503 GB. main 학습은 8 GPU와 labeler CPU 48 worker에 GPU당 dataloader 6 worker를 함께 쓴다. CPU가 적으면 labeler가 늦어져 학습 내용이 바뀐다(`INVALID_LABELS.json`, 6절). 다른 GPU 수는 [FAIRNESS.md](FAIRNESS.md)를 먼저 읽는다.

## 2. 순서

모든 스크립트는 `tools/release/setup/`에 있다. 위에서 아래로 실행한다. 경로는 전부 `tools/release/paths.env`에서 온다.

| # | 스크립트 | 만드는 것 | 프로필 | 디스크 | 시간 (기준 서버) | GPU | 앞서 끝나야 하는 것 |
|---|---|---|---|---:|---|---|---|
| 0 | `00_dirs_and_env.sh` | 빈 폴더, 경로 출력 | 전부 | - | 1초 | 아니오 | `paths.env` |
| 1 | `install_env.sh` | Python 환경 | 전부 | 9 GiB | 약 10분 | 아니오 | conda, 네트워크 |
| 2 | `10_download_dataset.sh` | 로그, 센서, 지도, 링크 | 전부 | 로그 14, trainval 센서 445, test 220, 지도 1.4 GiB | 약 7.1시간 [추정] | 아니오 | 네트워크 |
| 3 | `15_token_lists.sh` | 토큰 표 3개 (`derive_tokens.py`) | 전부 | 2 MB | 약 1분 | 아니오 | 2 |
| 4 | (복사) `pdm_score_256` | `SSR_PLAN_DATA/` | 전부 | 0.3 GiB | 복사 | 아니오 | 5절 참고 |
| 5 | `20_metric_cache_navtrain.sh` | navtrain metric cache | `refiner_noteacher`, `full` | 34 GiB | 41 + 9분 (CPU 64 + 36 worker) | 아니오 | 3 |
| 6 | `30_gt_stores.sh` | objects, sdf, e2e_side | `refiner_noteacher`, `full` | 12.5 GiB | 약 1시간 | 아니오 | 5 |
| 7 | `40_packed_from_dataset.sh` | `ck/packed/*` (`repack_dataset.py`) | `refiner_noteacher`, `full` | 16 MB | 1분 | 아니오 | 2 |
| 8 | `50_warmup_labels.sh` | raw256, 변형 라벨 (`make_raw256_traj.py`) | `refiner_noteacher`, `full` | 3.1 + 2.4 GiB | raw256 2.4시간, 변형 약 50분 (CPU 48 worker) | 아니오 | 4, 5, 7 |
| 9 | `60_teacher_bev.sh` | ReSMap, BEVFusion cache | `full` | 727 + 508 GiB | ReSMap 약 1시간, BEVFusion 복사 시간 미측정 | 아니오 (재생성은 예) | 없음 |
| 10 | `run_refiner_teachers.sh` | refiner teacher 2개 | `full` | 1 GiB | 2.77시간 | **4 + 4 GPU** | 6, 7, 8, 9 |
| 11 | `70_pristine_navsim.sh` | 원본 NAVSIM v1.1, v2.2, navtest cache | 평가 | 3 + 3.6 GiB | v1.1 cache 44분 | 아니오 | 2 |
| 12 | `verify_assets.py --profile <프로필>` | 전체 검사 | 전부 | - | 수 분 | 아니오 | 위 전부 |
| 13 | `scripts/experiment.sh check`, `smoke` | 조립 확인, 짧은 학습 | 전부 | 2.5 GiB | check 수 초, smoke 7분 | smoke만 **8 GPU** | 12 |
| 14 | `scripts/experiment.sh train` | 학습 | 전부 | 60 GiB | 19.9시간 | **8 GPU** | 13 |
| 15 | `scripts/experiment.sh eval` | 평가 | 전부 | 작음 | GPU 15분 + 채점 25분 | **4 GPU** | 11, 14 |

합계(앞뒤로 겹치지 않게 그대로 더한 추정): BEVFusion을 복사하면 약 38시간(복사 시간 제외), 다시 학습하면 약 78시간이다. CPU 단계는 동시에 돌릴 수 있다. 예전 서버에서는 metric cache와 sdf, side를 겹쳐 돌렸다.

**선택:** `20_metric_cache_navtest_fork.sh`는 연구 도구(`SSR_NAVTEST_MC`)용이다. release 경로에는 필요 없다.

## 3. 명령

```bash
git clone <repo> && cd <repo>
cp tools/release/paths.env.example tools/release/paths.env   # 경로를 고친다. git에 올라가지 않는다
S=tools/release/setup

bash $S/00_dirs_and_env.sh
CONDA=/path/to/conda bash $S/install_env.sh
bash $S/10_download_dataset.sh          # PARTS="logs navtrain maps test_logs test_cameras"가 기본
bash $S/15_token_lists.sh
# pdm_score_256.npy, pdm_score_256.tokens.json을 $SSR_PLAN_DATA로 복사 (5절)
K=16 bash $S/20_metric_cache_navtrain.sh
bash $S/30_gt_stores.sh
bash $S/40_packed_from_dataset.sh
bash $S/50_warmup_labels.sh
BEVFUSION_SRC=host:/path/cache_train_100x100/ bash $S/60_teacher_bev.sh
bash scripts/experiment.sh teachers     # run_refiner_teachers.sh. GPU 8장, 2.77시간
bash $S/70_pristine_navsim.sh

/venv/.../python tools/release/verify_assets.py --profile full     # 전체 검사. 실패하면 고칠 방법을 줄마다 알려 준다
bash scripts/experiment.sh check
bash scripts/experiment.sh smoke GPUS=0,1,2,3,4,5,6,7
bash scripts/experiment.sh train GPUS=0,1,2,3,4,5,6,7
bash scripts/experiment.sh eval RUN=<name> GPUS=0,1,2,3
```

공통 규칙:
- `SETUP_DRY=1`을 앞에 붙이면 무엇을 실행할지 찍기만 한다. 어떤 파일도 만들지 않는다.
- 스크립트는 **기존 결과를 덮어쓰지 않는다.** 있으면 멈춘다. 중간에 끊긴 단계는 `RESUME=1`로 이어 간다. `SPLITS=navtrain_train`을 주면 navtrain_val을 건너뛴다. navtrain_val은 refiner teacher를 다시 학습할 때만 쓴다.
- 오래 걸리는 단계는 foreground로 돈다. `tmux`나 `nohup`으로 띄운다. 로그는 `$SSR_DATA_ROOT/setup_logs/`와 `metric_cache/logs/`에 쌓인다.
- 경로 변수의 뜻은 `tools/release/paths.env.example`에 있다. `navsim/common/ssr_paths.py`가 한곳에서 읽는다.

**`verify_assets.py` 사용법**

```bash
python tools/release/verify_assets.py --profile {base,refiner_noteacher,full} [--quick] [--strict] [--only GROUP] [--json FILE]
```

- `--quick`: 수 초. manifest, 폴더 목록, index 안의 개수, 고정된 토큰 200개 표본만 본다. `experiment.sh train`이 시작할 때 부른다.
- 옵션 없음: 수 분. 85,109개 토큰 전부를 확인하고, 배열을 해시하고, 학습용 GT 로더로 전부 읽어 기대값과 비교한다.
- `FAIL`은 그 프로필의 학습이 멈추거나 입력 없이 조용히 돌아간다는 뜻이다. 종료 코드 1이다.
- `WARN`은 평가나 teacher 재학습에만 필요한 것, 또는 값에 영향이 없는 차이다. 종료 코드 0이다. `--strict`면 1이다.
- 어떤 검사도 학습된 planner 파일(`cand.npy`, `v2_final.npy`, `labels/*/cand`, `v2_source`)을 찾지 않는다.

## 4. Pin (이 값과 달라지면 다른 입력이다)

`sha16`은 배열 바이트의 sha256 앞 16자리, 파일이라고 적은 것은 파일 전체의 sha256 앞 16자리다. 전부 `verify_assets.py`에 들어 있다.

| 무엇 | 값 |
|---|---|
| 토큰 수 | navtrain_train 85,109 (log 978), navtrain_val 18,179 (log 214), navtest 12,146 (log 136) |
| 토큰 정렬 sha16 | train `c3fa309cbde971e8`, val `05b78ec43b95d954` |
| packed train | token `0b56b9aecbe1f094`, gt_traj `ad3654bd81bdb82f`, status `d0d6f908e8be4f3c`, ok `1f8f6dbc2f6c7496` |
| packed val | token `0fea22c41a4106b0`, gt_traj `adb06837415973fd`, status `2309a38c18a61c48`, ok `0bcf0f7a51703e00` |
| raw256 궤적 | train `72767c6c437f1132`, val `701c66d6ca564798` (앵커를 토큰 수만큼 반복한 것) |
| raw256 라벨 | train labels `b0f6e572d72692c3`, ok `d7bbcbf0fb3bb9e3`; val labels `dad2da663bebdb57`, ok `89612aae9df7937a` |
| 변형 train | labels `4d042a39ebd1fd8a`, ok `17efef5f358f3ac2`, traj `9969cf98fb7aa9ac`, anchor_idx `47744cd2f2076785`, valid `ceb5ebd777f6d4ef` |
| 변형 val | labels `3db198fbd7a00d07`, ok `c6ed18f987be86f3`, traj `9a9e488338450b24`, anchor_idx `c8e5c5a8893e3a99`, valid `ac4f8db652148258` |
| 앵커 | 배열 `6bdf549edc25a959`, 파일 `44f64a763473c3a8` (`resources/planning_vb/`) |
| `pdm_score_256` | npy 파일 `3b9e6c3e75bc698e`, tokens 파일 `b1278ec613611037`, shape [103288, 5, 256] |
| 저장된 metric cache 설정 | `resources/metric_cache_saved_config.yaml` sha256 `92d42853...` |
| GT 로더 기대값 | gt_n 85109, gt_ok 85109, p_pdm_nonfinite 0, cl_n_lt2 0, sdf_absmax_zero 0, sdf_absmax_max 10.0, obj_n_max 800, obj_n_zero 68, p_pdm_zero 4370, p_pdm_max 67.8193 |
| BEVFusion 64 m | manifest head `71ea94f672c89521`, 103,288 파일, 파일당 5,278,014 B |
| ReSMap front64 | meta sha256 `e67cc78d...`, HF revision `73db598669a8e6edb2b585fcc14ebe54b01b15f5` |
| refiner teacher | `$SSR_TEACHER_CACHE/refiner/det`, `$SSR_TEACHER_CACHE/refiner/map`. ckpt sha16은 `registry.yaml`의 `refiner_ckpt_sha16` |
| 환경 | Python 3.9.23, torch 2.8.0+cu128, torchvision 0.23.0+cu128, PL 2.2.1, timm 1.0.28, numpy 1.23.4, hydra 1.2.0, nuplan-devkit `ce3c323` |

전체 패키지 버전은 `requirements.lock.txt`와 `tools/release/setup/constraints.txt`에 있다. `requirements.lock.txt`는 `pip freeze`에서 `-e` 줄을 빼고 conda의 `file://` 줄을 `이름==버전`으로 바꾼 것이다(`tools/release/setup/make_lock.py`). 기존의 `requirements.txt`, `requirements_navsim.txt`, `environment.yml`, `docs/install.md`는 다른 스택 용이라 그대로 따라 하면 재현되지 않는다.

변형 라벨 폴더의 옛 `build.json`에 적힌 `traj_sha16` `f4b5a78117b06b0a`는 파일과 맞지 않는다. 가속 열을 합치기 전 값이다. 위 표의 `traj` 값이 실제 파일의 값이다.

## 5. 복사와 재생성, 이 서버에만 있는 것

| 자산 | 새 서버에서 | 공개 여부 | 비고 |
|---|---|---|---|
| 데이터셋, 지도 | 다운로드 | 공개 | `navtrain_*` 묶음은 카메라와 LiDAR가 섞여 있다. 필터 파일 3개는 NAVSIM v1.1과 바이트까지 같다 |
| 앵커, norm, 토큰 표, 저장된 cache 설정, `rescore_attr.py` | 저장소에 있음 | 저장소 | 복사 불필요 |
| 토큰 표 파일 | 재생성 (`15_token_lists.sh`) | 공식 필터에서 나옴 | 저장소 사본과 순서까지 비교한다 |
| navtrain metric cache | 재생성 | 아니오 | v1.0 꼴 fork로 만든다. pkl 안에 절대 경로가 있으니 경로를 나중에 옮기지 않는다 |
| objects, sdf, e2e_side | 재생성 | 아니오 | |
| packed | 재생성 (`repack_dataset.py`) | 아니오 | 이 서버 기존 폴더와 전체 행이 비트 단위로 같음을 확인했다 |
| raw256, 변형 라벨 | 재생성 | 아니오 | 변형은 `--ext straight` 한 번으로 만든다. 옛 빌드는 두 번 돌려 손으로 합쳤다 |
| ReSMap front64 | 다운로드 | HF 공개, cc-by-nc-4.0 | revision 고정. `bev/`와 json만 받으면 781 GB다 |
| **BEVFusion 64 m 체크포인트와 cache** | **복사** | **아니오** | 지금은 이 서버에만 있다. 545 GB |
| **`pdm_score_256`** | **복사** | **가공본은 비공개** | WoTE 저장소(Google Drive)의 `formatted_pdm_score_256.npy`를 `tools/plan_v2/pack_pdm_scores.py`로 묶어 다시 만들 수 있다. 학습된 가중치를 쓰지 않는다. 다시 만든 파일이 pin과 같은지는 확인하지 못했다 |
| **refiner teacher** | 복사 또는 재학습 | **아니오** | `$SSR_TEACHER_CACHE/refiner/{det,map}`에 만든다. 재학습하면 sha가 바뀌므로 `registry.yaml`을 고친다 |
| 평가 결과, 체크포인트 | 다시 돌림 | 아니오 | |

**BEVFusion을 다시 만들면** 비트 단위로 같아지지 않는다(`deterministic: false`). 그래서 `registry.yaml`의 `sha_head`, `norm_v15`, DET refiner teacher를 모두 새 값으로 바꿔야 한다. 저장소 `yyongjae/bevfusion` @ `b939770`, 4 GPU 39.8시간 + 8 GPU 0.5시간이다. 절차는 `60_teacher_bev.sh` 맨 위에 있다.

**새 teacher 추가:** 코드는 바꾸지 않는다. 같은 형식의 cache를 만들고 `tools/release/registry.yaml`에 항목을 넣는다. `bevformer`, `mask2map`은 `status: missing`으로 등록돼 있고, 고르면 부족한 것을 나열하며 거부한다. 형식은 [EXPERIMENTS.md](EXPERIMENTS.md)에 있다.

## 6. 조용한 실패와 잡는 장치

입력이 없어도 학습이 멈추지 않는 경우가 있다. 멈추지 않으면 다른 실험이 된다. 각 경우를 무엇이 잡는지 정리했다.

| 빠졌거나 틀린 것 | 그냥 두면 | 시작 전 검사 | 학습 중 검사 |
|---|---|---|---|
| objects, sdf, e2e_side | `ref_gt_ok=False`, 측방 손실이 토큰별로 꺼짐 | `verify_assets` `gt.*` | `online2._check_inputs`: 처음 2,000 토큰에서 `strict_rows` 아래면 중단 (rank 0 기준) |
| teacher BEV 토큰 일부 | 0으로 채우고 `kd_ok=False`, KD가 토큰별로 꺼짐 | `teacher.*.cache` | 같은 검사, `ck2/kd_ok_frac`. refiner off면 `distill_strict_ok` |
| `pdm_score_256` 토큰 일부 | `sim_reward_valid=0` | `pdm_score_256` (tokens 해시) | 없음 |
| metric cache 일부, labeler 느림 | on-policy 라벨 대신 warm-up 라벨 사용 | `metric_cache.*` | epoch 끝에 fallback > 1 %면 `<io_dir>/INVALID_LABELS.json` 기록, 멈추지는 않음. rank 0 기준 |
| warm-up 행 읽기 실패 | `ck2_wu_ok=False` | `labels.*` | `_check_rows`: 처음 2,000 토큰 |
| `meta.json`, `_plan.json`이 없음 | 출처 비교를 건너뜀 | `packed.*`, `labels.*` (pin으로 비교) | 없음 |
| `build.json`이 없음 (평가) | 변형 정의 비교를 건너뜀 | `labels.var.*` | 없음 |
| 앵커 파일 교체 | planner가 해시 없이 읽음 | `anchors` | 평가 preflight |
| 다른 작업 트리의 `navsim`을 import | 다른 코드로 학습 | `env.navsim_import` | `run_training.py`가 시작할 때 거부 |
| `run_training.py` 직접 실행 | ResNet-50, 32 m, KD 없음으로 학습 | - | `SSR_EXPERIMENT_SHA` 또는 `SSR_ALLOW_DIRECT=1`이 없으면 거부 |
| navtest 토큰이 학습에 들어감 | 누수 | - | `run_training.py`가 거부, `train_tokens.json` 기록 |
| 다른 teacher의 z-score | 조용히 옛 값 사용 | registry | `distill_norm_files`: 파일이 없으면 오류 |
| teacher를 다른 split에서 학습 | 그대로 사용 | `teacher.*.refiner_run` | `teacher_check`가 거부 |
| GPU 수가 다른 행 비교 | BN, GridMask, EMA가 다름 | 이름에 `g<N>` | `table`이 거부 (`ALLOW_MIXED=1` 필요) |
| 디스크 부족 | rank 0만 죽고 나머지 남음 | `train`이 70 GiB 미만이면 거부 | 없음 |
| 셸에 남은 `OPENSCENE_DATA_ROOT` (중첩 폴더) | 첫 장면을 읽을 때 실패 | `ssr_paths`가 기본 경로로 바꾸고 stderr에 한 줄 | - |

`--quick`은 표본만 본다. **cache를 복사했거나 다시 만들었으면 옵션 없는 전체 검사를 한 번 돌린다.**

## 7. "calib-kd 같은" 문제 목록

학습된 모델이나 테스트 데이터가 은근히 들어오는 경로를 모았다.

| 항목 | 들어오는 것 | 지금 상태 |
|---|---|---|
| KD 보정 | run 자신의 online 값만 | 해결. 기본값을 online으로 바꿨다. CLI에서 file 모드는 `SSR_ALLOW_FILE_CALIB=1` 없이 거부 |
| `packed/` | 옛 r34 dump 폴더였음. 읽는 값은 GT뿐 | 해결. `repack_dataset.py`로 데이터셋에서 만든다. 이 서버의 기존 폴더는 값이 같아 그대로 두고 `WARN` |
| teacher의 r34 평가 | 모니터링만, 없으면 멈췄음 | 해결. `--r34-evals 0`. 기존 teacher는 켠 채 학습됐다. 가중치에 영향이 없는지는 추론만 했고 비교하지 않았다 |
| 백본 BN | 없음. 하지만 v7 계열만 다르게 움직임 | 설정으로 드러냄 (`backbone_bn_mode`, 기본 `moving`). 기본값은 결정 대기 |
| GPU 수 | BN, GridMask, EMA를 바꿈 | 이름에 넣고 표에서 섞지 못하게 함. 기준은 8 |
| 데이터셋 순서 | 파일 시스템 순서 | 해결 (`sorted`) |
| labeler 시점 | 실행 시간에 따라 다른 라벨 | 일부 해결. fallback을 기록한다. 시점 자체는 그대로 비결정적이다 |
| 일부만 복사된 cache | KD, 측방 손실이 조용히 꺼짐 | 해결. 시작 전 검사 + 학습 중 중단 |
| BEVFusion이 navtrain_val도 학습 | val이 DET teacher에게 in-sample | 공개로 처리 |
| ReSMap | val 로그 포함, 학습 split 미확인, cc-by-nc | 공개로 처리. 소유자 확인 대기 |
| 앵커, `pdm_score_256` | navtrain(val 포함) k-means | 공개로 처리. WoTE 인용 |
| v15 상수, teacher 레시피, Platt, `MERGE_EPS` | navtest를 본 분석으로 정함 | 공개로 처리 ([FAIRNESS.md](FAIRNESS.md)) |
| 선택 β grid, `--sel-w`, `--calib-offset`, `--choose-lat` | val과 navtest를 봄 | 해결. `RELEASE_SELECT` 하나로 잠그고 옛 도구는 울타리를 침 |
| navtest, navhard에 직접 맞춘 선택 식 | 테스트에 직접 튜닝 | 이 저장소에 없음. 합치지 않는다 |
| `distill_lambda_*`, `distill_tau_*` | v15에는 효과 없는 키 | `EXTRA`에서 거부 |
| teacher 교체 시 옛 z-score | 조용히 사용 | 해결. registry와 `distill_norm_files` |

## 8. 확인하지 못한 것

- 새 서버에서 처음부터 끝까지 실행한 적이 없다. 다운로드, 환경 설치, metric cache, GT store, warm-up 라벨 스크립트는 명령 출력(`SETUP_DRY=1`)과 문법만 확인했다.
- `15_token_lists.sh`는 로그 1개로만 표와 같음을 확인했다. 전체 실행은 하지 않았다.
- 변형 라벨 한 번 빌드: `traj`와 `valid`만 1,500행에서 비트 단위로 같음을 확인했다. **라벨을 다시 채점해 비교하지는 않았다.** 다르면 `verify_assets`가 잡는다.
- metric cache: 3개 토큰을 다시 만들어 같음을 확인했다. 전체 재생성 결과의 pin 비교는 없다. 다른 서버에서 같은 라벨이 나오는지도 모른다.
- `--r34-evals 0`으로 재학습한 teacher가 기존 teacher와 비슷한지 비교하지 않았다. 비트 단위로는 같지 않을 것이다.
- test 카메라만 받은 상태로 평가가 도는지 실행하지 않았다(코드상 LiDAR를 읽지 않는다).
- `hf download`의 `--include` 두 번 지정은 도움말만 확인했다.
- 시간과 디스크는 기준 서버 기록에서 왔다. 합계는 단순 합이다. 전송 시간은 빠져 있다.
- `10_download_dataset.sh`의 test 부분은 공식 스크립트 순서를 옮긴 것으로, 실제로 내려받아 보지 않았다.
