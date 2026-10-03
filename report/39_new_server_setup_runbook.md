# 39. 새 서버 세팅 런북: refiner-KD (stage T teacher, stage E E1/E2, navtest 평가)

> 작성: 2026-10-03, 기준 서버 `blackwell64`. 모든 내용은 이 서버의 로그, 설정, 코드를 **읽기만 해서** 확인했다.
> 대상 독자는 두 부류다. 새 서버에서 새로 시작하는 Claude 세션, 그리고 사람.
> 이 문서만 보고 위에서부터 순서대로 실행하면 같은 상태가 되도록 썼다.
>
> **경로 규칙:** 모든 명령은 **이 서버와 같은 절대경로**를 쓴다. 코드에 절대경로가 100곳 넘게 하드코딩돼 있기 때문이다(§3).
> 새 서버의 실제 디스크 위치가 다르면 §3.2처럼 symlink로 이 경로들을 만들어 둔다. 그러면 코드를 한 줄도 고치지 않아도 된다.
>
> 표기: **[실측]** 이 서버에서 측정한 값, **[추정]** 측정값에서 외삽한 값, **[재구성]** 로그나 overrides에서 거꾸로 만든 명령(원래 셸 명령줄은 기록에 없음).

---

## 0. 목적과 전제

### 0.1 새 서버에 이미 있는 것 (사용자 전제)

| 항목 | 기대 위치 (이 서버와 같게) | 비고 |
|---|---|---|
| NAVSIM 다운로드 | `/home/external-user/navsim/download/{maps, trainval_navsim_logs/trainval, trainval_sensor_blobs/trainval, test_navsim_logs/test, test_sensor_blobs/test}` | 구조는 §3.4 |
| BEVFusion teacher cache | `/home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100`, `.../cache_val_50x100` | 103,288 / 12,146 npz. 검사는 §5.9 |
| ReSMap teacher cache | `/home/external-user/datasets/teacher_cache/resmap` (navtest는 `resmap/navtest`) | 126,032 / 12,146 frame |

### 0.2 새 서버에서 새로 만드는 것 (캐시 전부)

1. conda env `ssr` (§2)
2. `SSR/data/` symlink (§3.3)
3. navtest 공식 metric cache `SSR/data/exp/metric_cache` (§4)
4. stage-E 학습 GT: token list → metric cache → objects / SDF / e2e_side / human (§5)
5. (선택, 경로 B) stage T 전체: metric cache, human, objects, SDF, draft bank, labels, pack, teacher 학습 (§6.3)

### 0.3 옛 서버에서 반드시 가져와야 하는 작은 파일 (캐시가 아니거나, 다시 만들 수 없는 것)

"캐시는 전부 새로 만든다"는 방침이어도, 아래는 다시 만들 수 없거나(학습된 가중치, 무작위 split) git에 없다. 합계는 golden을 빼고 약 0.5 GB다. 묶는 명령은 §1.2에 있다.

| # | 파일 | 크기 | 왜 필요한가 | 필수 여부 |
|---|---|---|---|---|
| 1 | `/home/external-user/ssd/yongjae_refiner/splits/` (디렉터리 전체) | 3.4 MB | stage-T split은 git에 없는 pool과 E cache 때문에 **같게 다시 만들 수 없다**(§6.3). `navtest.parquet`는 `stageE_compare.py:21`(bootstrap log map)이 읽는다 | **필수** |
| 2 | `/home/external-user/ssd/yongjae_refiner/stageE/teachers/` | 27 MB | stage E E2가 쓰는 teacher 교정기 R_T, R_M. 다시 학습하면 비트 단위로 같지 않다 | E2에 **필수**(경로 A) |
| 3 | `/home/external-user/ssd/yongjae_refiner/stageE/liveness_stageT4_{T,M}_fold0_seed0.json` | 수 KB | gate 통과 기록. 경로 A에서는 다시 계산할 수 없다 | 기록용 |
| 4 | `/home/external-user/yongjae/SSR/report/perception_reliability/pdm_attr/rescore_attr.py` | 36 KB, sha256 `9dbd00cd…c521` | **untracked**다. `score_trajectories.py:102-106,147`가 import한다. e2e_side의 p_pdm, stage-T 라벨, 진단 채점에 필요하다 | **필수** |
| 5 | E0 기준 체크포인트 `work_dirs/para_ssr_interaction_final/lightning_logs/version_2/checkpoints/last.ckpt` | 462,496,505 B, sha256 `ee46147a…552c` | E0 PDMS 0.8487 재현과 E2−E0 비교 | 평가 비교에 **필수** |
| 6 | `work_dirs/eval/para_ssr_interaction_final/` (CSV `2026.09.17.00.09.41.csv` + `code/hydra/*` + log). `videos/`(380 MB)는 빼고 묶는다(§1.2 `--exclude`) | 약 6 MB | `stageE_gpu_commands.sh:172` compare가 이 CSV를 하드코딩한다. `tools/refiner/tests/test_scorer.py:65`가 `code/hydra/config.yaml`을 읽는다 | **필수** |
| 7 | `work_dirs/eval/para_ssr_interaction_final_navtest_trajectories.pkl` | 1.8 MB | E0 + frozen teacher 사후 실험(§8.4) | 선택 |
| 8 | `work_dirs/para_ssr_interaction_final/code/hydra/config.yaml` | 2.8 MB | 원래의 E2E token 도출 입력. 없으면 tracked 파일로 대체할 수 있다(§5.2) | 선택 |
| 9 | `git diff` (커밋 안 된 수정 4개 파일) | 수 KB | `tools/dump_navtest_trajectories.py`의 `ARMS`에 `interaction_final` 등 3줄. E0 궤적 dump(§8.4)와 `report/refiner_T/stageE_diag/dump_e2_navtest.py`에 필요하다 | 진단 시 필수 |
| 10 | `data/exp/metric_cache/metadata/code/hydra/config.yaml` | 261,784 B, sha256 `92d42853…4f84` | §4에서 다시 만들면 같은 파일이 나온다(경로가 같을 때, `--cfg job`으로 확인함). 경로가 달라 sha가 깨질 때 대신 쓸 백업 | 백업 |
| 11 | `$DATA/stageE/parity_golden.pt`(210 MB), `parity_golden_e12.pt`(775 MB) | 985 MB | parity 검사(§7.1). `parity_golden_e12.pt`는 git에 없는 중간 코드로 만든 것이라 **다시 만들 수 없다** | 선택 |
| 12 | `report/collision_counterfactual/counterfactual/cf_common.py`, `report/planner_vs_perception_tests/safety_filter/sf_common.py` | 작음 | untracked. `build_metric_cache.py verify-*`, `build_future_objects.py --mc-root` 검증에서만 쓴다 | 선택 |
| 13 | `docs/README.md`, `scripts/evaluation/eval_arm.sh`, `eval_arm_aux.sh` | 작음 | untracked 실행 가이드, 보조 평가 스크립트 | 선택 |
| 14 | E2 결과 체크포인트 `work_dirs/stageE_E2_30ep_ema_r1.0_s0_rr5_wmax100_hmix_hwu5_bg1.0/lightning_logs/version_0/checkpoints/last.ckpt` | 481 MB, sha256 `057515ad…23d1` | 다시 학습하지 않고 E2를 재평가하거나 진단할 때 | 선택 |

### 0.4 저장소

- repo: `https://github.com/yyongjae/SSR.git`, branch **`exp-refine`**, commit **`6568d4d5bde3de6d3c1c0a5906fe98a9df0825e5`**.
- **이미 origin에 push돼 있다.** 2026-10-03에 `git ls-remote origin exp-refine`의 결과가 `6568d4d5…`였다. 일부 하위 문서에는 "아직 push 안 됨"이라고 적혀 있지만, 그 기록은 지금 상태와 다르다. 새 서버에서는 그냥 clone하면 된다. 공개 저장소라 인증 없이 받을 수 있다(2026-10-03 `api.github.com/repos/yyongjae/SSR` → HTTP 200).
- clone으로 **따라오지 않는 것:** 커밋 안 된 수정 4개 파일(§0.3 #9), untracked 파일(§0.3 #4, 12, 13), `data/`, `work_dirs/`(`.gitignore` 대상).
- stage T/E 핵심 코드(`tools/refiner/*` 74개, `navsim/agents/para_ssr/refiner/*`, `report/37`, `38`, `report/refiner_T/*`, `scripts/training/train_para_ssr_interaction.sh`, `scripts/evaluation/cache_metric_navtest.sh`)는 tracked다.

### 0.5 공통 변수 (이 문서의 모든 명령이 가정함)

```bash
REPO=/home/external-user/yongjae/SSR
D=/home/external-user/ssd/yongjae_refiner          # DATA root
PY=/home/external-user/miniconda3/envs/ssr/bin/python
DL=/home/external-user/navsim/download
TC=/home/external-user/datasets/teacher_cache
```
명령 블록 안에서는 변수를 쓰지 않고 절대경로를 그대로 쓴 곳이 많다. 새 셸에서 블록 하나만 복사해도 동작하게 하려는 것이다.

CPU 잡 관례(`report/refiner_T/IMPL_SPEC.md:21`): `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 <PY> …`. 도구마다 **프로세스당 worker 상한**이 있다(build_metric_cache 4, build_sdf 4, build_future_objects 4, extract_human 2). 더 빨리 돌리려면 token을 겹치지 않는 part로 나누고 프로세스를 여러 개 띄운다.

---

## 1. 체크리스트 한 장

### 1.1 순서, 시간, 의존성

| # | 단계 | 시간 | 선행 | E1 | E2 | 평가만 | stage T 재구축 |
|---|---|---|---|---|---|---|---|
| S0 | 옛 서버에서 이관 묶음 만들기(§1.2) → 새 서버로 전송 | 분 단위 + 전송 | – | ✓ | ✓ | ✓ | ✓ |
| S1 | conda env `ssr` 설치 + 검증(§2) | 20–40분 [추정] | – | ✓ | ✓ | ✓ | ✓ |
| S2 | 경로 symlink, repo clone, patch, 이관 파일 배치, `data/` symlink(§3) | 10분 | S0, S1 | ✓ | ✓ | ✓ | ✓ |
| S3 | teacher cache 검사(§5.9) | 1분 | S2 | – | ✓ | – | ✓ |
| S4 | navtest metric cache(§4) | 약 60분 [추정](12 프로세스) | S2 | ✓(평가) | ✓ | ✓ | ✓ |
| S5 | E2E token list(§5.2) | 10초 | S2 | ✓ | ✓ | – | – |
| S6 | todo/part 파일(§5.3) → stage-E metric cache 85,109(§5.4) | 약 2.7 h(12 worker) ~ 4 h(8 worker) [추정] | S4(SAVED_CFG sha), S5 | ✓ | ✓ | – | – |
| S7 | objects(§5.5), human(§5.6) | 5–15분, 30초 | objects: §5.3이 만든 `objects/logs/e2e_objects_todo.parquet`. human: S5. (S6의 §5.4와 병렬) | ✓ | ✓ | – | – |
| S8 | SDF(§5.7), e2e_side(§5.8) | 약 50분, 약 35분 (서로 병렬) [추정] | S6 | ✓ | ✓ | – | – |
| S9 | GTLoader 전수 검사(§5.10) | 1분 | S7, S8 | ✓ | ✓ | – | – |
| S10 | teacher 교정기: 경로 A 복사(§6.2) **또는** 경로 B 재구축(§6.3) | A: 수 분 / B: 약 7–8 h | A: S0 / B: S4 | – | ✓ | – | ✓ |
| S11 | pytest, parity, code hash(§7.1) | 5–10분 | S9, S10 | ✓ | ✓ | – | – |
| S12 | (선택) pilot150(§7.4) | 약 5분 | S11 | – | ✓ | – | – |
| S13 | E2 학습(§7.2) / E1 학습(§7.3) | 각 약 20 h(4× RTX 5090) [실측 E2] | S11 | ✓ | ✓ | – | – |
| S14 | navtest 평가 final/tau0 + E0 재현 + compare(§8) | 평가 하나당 30–65분 | S4, S13 | ✓ | ✓ | ✓ | – |

- **"평가만"** 경로(E0나 이미 있는 E2 ckpt만 채점): S0–S2, S4, S14만 하면 된다.
- **E1만** 할 때: teacher cache와 teacher 교정기가 필요 없다(E1은 `kd_teacher_runs`가 비어 있음). 다만 `human/e2e_train_trainlogs.npz`는 필요하다(E1도 `KD_DRAFT_SOURCE=human_mix`, `REF_HUMAN_ONLY_UNTIL=5`로 돌리기 때문).
- **stage T 재구축(경로 B)** 을 할 거면 S6 **전에** 하는 편이 낫다. stage T의 train/dev 토큰 26,366개(19,732 + 6,634)가 E2E 목록과 겹치므로, S6에서 그만큼 덜 만든다. 이 서버에서도 그 순서였다. 순서를 바꿔도 결과는 맞다(모든 빌더가 이미 있는 파일은 건너뜀).
- 병렬화 요약: S4가 끝나야 S6가 시작할 수 있다(`build_metric_cache.py`가 S4가 쓴 config의 sha256을 assert함). S5, S7은 S4와 동시에 돌려도 된다.

### 1.2 S0: 옛 서버에서 이관 묶음 만들기 (옛 서버에서 실행)

```bash
# ── 옛 서버(blackwell64)에서 실행 ──
M=/home/external-user/ssd/ssr_migration; mkdir -p $M
cd /home/external-user/yongjae/SSR
git rev-parse HEAD > $M/git_head.txt                       # 6568d4d5bde3de6d3c1c0a5906fe98a9df0825e5
git diff > $M/uncommitted.patch                            # 4 files (WoTE_agent.py, 2 tests, dump_navtest_trajectories.py)
cp -p data/exp/metric_cache/metadata/code/hydra/config.yaml $M/navtest_saved_cfg_config.yaml   # sha256 92d42853…
SP=/tmp/claude-1001/-home-external-user/9262a8d2-fa20-46d7-ba9a-92d7d13b8ceb/scratchpad
cp -p $SP/derive_e2e_tokens.py $SP/e2e_manifest.py $SP/verify/v_side_gt.py $M/ 2>/dev/null   # /tmp는 지워질 수 있다. 내용은 부록 A에도 있다
cd / && tar cf $M/bundle.tar \
  --exclude='home/external-user/yongjae/SSR/work_dirs/eval/para_ssr_interaction_final/videos' \
  home/external-user/ssd/yongjae_refiner/splits \
  home/external-user/ssd/yongjae_refiner/stageE/teachers \
  home/external-user/ssd/yongjae_refiner/stageE/liveness_stageT4_T_fold0_seed0.json \
  home/external-user/ssd/yongjae_refiner/stageE/liveness_stageT4_M_fold0_seed0.json \
  home/external-user/yongjae/SSR/report/perception_reliability/pdm_attr/rescore_attr.py \
  home/external-user/yongjae/SSR/report/collision_counterfactual/counterfactual/cf_common.py \
  home/external-user/yongjae/SSR/report/planner_vs_perception_tests/safety_filter/sf_common.py \
  home/external-user/yongjae/SSR/work_dirs/para_ssr_interaction_final/lightning_logs/version_2/checkpoints/last.ckpt \
  home/external-user/yongjae/SSR/work_dirs/para_ssr_interaction_final/code/hydra \
  home/external-user/yongjae/SSR/work_dirs/eval/para_ssr_interaction_final \
  home/external-user/yongjae/SSR/work_dirs/eval/para_ssr_interaction_final_navtest_trajectories.pkl \
  home/external-user/yongjae/SSR/docs/README.md \
  home/external-user/yongjae/SSR/scripts/evaluation/eval_arm.sh \
  home/external-user/yongjae/SSR/scripts/evaluation/eval_arm_aux.sh
# (선택) parity golden 985 MB, E2 ckpt 481 MB:
# tar rf $M/bundle.tar home/external-user/ssd/yongjae_refiner/stageE/parity_golden.pt home/external-user/ssd/yongjae_refiner/stageE/parity_golden_e12.pt \
#   home/external-user/yongjae/SSR/work_dirs/stageE_E2_30ep_ema_r1.0_s0_rr5_wmax100_hmix_hwu5_bg1.0/lightning_logs/version_0/checkpoints/last.ckpt
sha256sum $M/bundle.tar > $M/bundle.tar.sha256
```
> **[실측] 이 묶음은 2026-10-03 13:17에 위 명령 그대로(선택 항목 제외) 이미 만들어 두었다:** `/home/external-user/ssd/ssr_migration/` (481 MB, `bundle.tar` 45개 항목, sha256은 `bundle.tar.sha256`, `git_head.txt` = `6568d4d5…`, `uncommitted.patch` 104줄, scratchpad 스크립트 3개 포함). 새 서버에서는 이 디렉터리를 통째로 받기만 하면 된다.

`tar`는 `/`를 기준으로 한 상대경로로 묶는다. 그래서 새 서버에서 `tar xf bundle.tar -C /`로 풀면 같은 자리에 들어간다(§3.2 symlink를 먼저 만들어야 함).
**새 서버로는 `$M` 디렉터리를 통째로 보낸다.** `bundle.tar` 말고도 `uncommitted.patch`, `navtest_saved_cfg_config.yaml`, `git_head.txt`, scratchpad 스크립트가 `$M`에 따로 있고, §3.3이 이 파일들을 `$M/`에서 읽는다(예: `rsync -a $M/ <새서버>:/home/external-user/ssd/ssr_migration/`).
`videos/`를 빼면 bundle은 약 0.5 GB다(E0 ckpt 462 MB가 대부분). 빼지 않으면 약 0.9 GB가 된다. [실측: 2026-10-03 `du`]

---

## 2. 환경

### 2.1 기준 서버 값 [실측]

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0-31-generic, glibc 2.39, gcc 13.3.0 |
| CPU / RAM | 32 threads(`nproc`), 251 GB |
| GPU | NVIDIA GeForce RTX 5090 ×6, 32,607 MiB, sm_120. driver 580.173.02 |
| CUDA toolkit | 쓰지 않는다. prebuilt wheel(torch cu128, spconv-cu126)만 쓴다. `nvcc`는 PATH에 없다 |
| 디스크 | refiner 데이터는 `/home/external-user/ssd`(nvme) 위에 있다 |
| `/dev/shm`, `ulimit -n` | 126 GB tmpfs, 1,048,576. 학습은 DataLoader 프로세스 24개(6/GPU × 4)를 띄운다. 컨테이너라면 `--shm-size`를 넉넉히 준다(새 서버에서 필요한 최소값은 측정하지 않음) |

새 서버 요구사항: torch 2.8.0+cu128 wheel을 쓰려면 NVIDIA driver가 **R570 이상**이어야 한다(일반 지식, 이 서버에서는 580만 확인). wheel의 arch list는 `sm_70…sm_120`이라 Ampere/Hopper에서도 동작한다.

### 2.2 핵심 패키지 (env 이름은 반드시 `ssr`)

Python **3.9.23**. torch **2.8.0+cu128**, torchvision 0.23.0+cu128, triton 3.4.0, pytorch-lightning **2.2.1**(통합 `lightning` 패키지는 없음), timm 1.0.28, hydra-core 1.2.0, omegaconf 2.3.1, numpy **1.23.4**, wandb 0.26.1, ray 2.51.2, nuplan-devkit 1.2.0(git commit `ce3c323af01c0d7ec5672f7832ef53f9c679aab0`), spconv-cu126 2.3.8. repo는 `para-ssr-navsim`이라는 이름으로 **editable** 설치돼 있다. mmcv, mmdet 계열은 없다(필요 없음).

주의할 함정:
- `requirements_navsim.txt`가 `torch==2.0.1`, `torchvision==0.15.2`를 pin한다. 그대로 설치하면 torch가 다운그레이드되고, RTX 5090(sm_120)이 지원되지 않는다. **이 두 줄을 빼고** 설치한다.
- `setup.py`가 위 파일을 `install_requires`로 읽는다. 그래서 editable 설치는 반드시 `--no-deps`로 한다.
- `environment.yml`은 env 이름이 `ssr-navsim`이지만 스크립트는 `envs/ssr`를 찾는다. **이름은 `ssr`로 만든다.**
- repo 루트의 `requirements.txt`, `setup_fix.md`는 옛 nuScenes SSR(torch 1.9) 기록이다. 이 프로젝트와 관계없다.

### 2.3 설치 레시피 (이 서버의 pip freeze에서 거꾸로 만든 것. 이 형태로 실행해 본 적은 없음)

```bash
# miniconda가 /home/external-user/miniconda3 에 있다고 가정 (§3.2 symlink 참고)
/home/external-user/miniconda3/bin/conda create -n ssr -c conda-forge python=3.9.23 pip=23.3.1 nb_conda_kernels -y
PY=/home/external-user/miniconda3/envs/ssr/bin/python
$PY -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
cd /home/external-user/yongjae/SSR          # §3에서 clone한 뒤
grep -vE '^(torch|torchvision)==' requirements_navsim.txt > /tmp/req_navsim_notorch.txt
$PY -m pip install -r /tmp/req_navsim_notorch.txt          # nuplan-devkit은 GitHub에서 받는다 (git 접근 필요)
$PY -m pip install numpy==1.23.4 scipy==1.13.1 shapely==2.0.7 pandas==2.3.3 pyarrow==21.0.0 \
  geopandas==1.0.1 rasterio==1.3.11 fiona==1.10.1 pyogrio==0.11.1 rtree==1.4.1 \
  opencv-python==4.9.0.80 scikit-learn==1.2.2 hydra-core==1.2.0 omegaconf==2.3.1 \
  pytorch-lightning==2.2.1 torchmetrics==1.8.2 timm==1.0.28 huggingface_hub==1.8.0 safetensors==0.7.0 \
  tensorboard==2.16.2 protobuf==4.25.3 wandb==0.26.1 ray==2.51.2 SQLAlchemy==1.4.27 setuptools==65.5.1 \
  positional-encodings==6.0.1 pyquaternion==0.9.9 spconv-cu126==2.3.8 cumm-cu126==0.7.11 pytest==8.4.2
$PY -m pip install -e /home/external-user/yongjae/SSR --no-deps
```
버전이 어긋나면 부록 B의 전체 pin 목록을 `-c constraints.txt`로 준다.

### 2.4 가중치, wandb, HF

- 백본 timm `resnet50.tv_in1k`(`navsim/agents/para_ssr/configs/default.py:85`)은 학습을 시작할 때 HF hub에서 받는다(98 MB). 계산 노드가 오프라인이면 미리 받아 둔다.
  ```bash
  CUDA_VISIBLE_DEVICES="" /home/external-user/miniconda3/envs/ssr/bin/python -c "import timm; timm.create_model('resnet50.tv_in1k', pretrained=True)"
  ls ~/.cache/huggingface/hub/models--timm--resnet50.tv_in1k/snapshots/*/model.safetensors
  ```
  체크포인트를 주고 평가할 때는 다운로드하지 않는다(`para_ssr_agent.py:303-307`).
- wandb: `default_training.yaml:40-48`의 기본값이 `enable: true, project: para-ssr, mode: online, non_fatal: true`다. 새 서버에서 `/home/external-user/miniconda3/envs/ssr/bin/wandb login`을 하면 `~/.netrc`에 `machine api.wandb.ai`가 생긴다. 로그인하지 않으면 학습은 계속되지만 wandb 기록은 남지 않는다. `train4`는 명령줄 인자로 wandb override를 받지 않는다. `WANDB_MODE`만 export해서는 안 된다(logger가 cfg 값 `wandb.mode`를 넘김, `navsim/planning/training/wandb_logging.py:171`). offline으로 돌리는 방법은 둘이다. (a) `WANDB=1 WANDB_MODE=offline`을 함께 export한다. 이 env는 `stageE_gpu_commands.sh` → `train_para_ssr.sh:64-87`까지 전달되어 `wandb.enable=true wandb.project=para-ssr wandb.group=para-navsim wandb.mode=offline wandb.name=<exp> wandb.tags=[…]`를 덧붙인다. 다만 E2 원래 실행에는 이 6줄이 없었다. 그래서 `overrides.yaml`이 달라지고, §7.3의 E1/E2 diff 검사에도 이 줄들이 더 나온다(group도 `para-navsim`으로 바뀜). (b) `stageE_gpu_commands.sh`의 train4 줄에 `wandb.mode=offline`을 직접 넣는다.
- `~/.bashrc`에는 프로젝트 환경변수가 없다. 모든 env는 wrapper 스크립트가 설정한다(§7.5).

### 2.5 환경 검증 (기대값)

```bash
PY=/home/external-user/miniconda3/envs/ssr/bin/python
$PY --version                                                   # Python 3.9.23
$PY -c "import torch;print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.nccl.version(), torch.cuda.get_arch_list()[-1])"
#   2.8.0+cu128 12.8 91002 (2, 27, 3) sm_120
#   (GPU가 보이는 셸에서 실행한다. CUDA_VISIBLE_DEVICES=""이면 get_arch_list()가 []라서 IndexError가 난다)
CUDA_VISIBLE_DEVICES="" $PY -c "import pytorch_lightning as pl, timm, numpy, hydra, cv2, wandb, nuplan, navsim, spconv.pytorch; print(pl.__version__, timm.__version__, numpy.__version__, hydra.__version__, cv2.__version__, wandb.__version__); print(navsim.__file__)"
#   2.2.1 1.0.28 1.23.4 1.2.0 4.9.0 0.26.1
#   /home/external-user/yongjae/SSR/navsim/__init__.py     <- site-packages가 나오면 editable 설치가 잘못된 것
$PY -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"   # True <GPU 수>
which flock                                                     # util-linux (stageT_gpu_commands.sh가 사용)
```

---

## 3. 저장소와 경로

### 3.1 하드코딩된 절대경로 (같은 경로를 만들면 고칠 필요 없음)

`git grep` 기준 등장 횟수: `/home/external-user/ssd/yongjae_refiner` 52, `/home/external-user/yongjae/SSR` 31, `/home/external-user/miniconda3/envs/ssr` 19, `/home/external-user/datasets/teacher_cache` 9, `/home/external-user/navsim/download` 4.

| file:line | 값 | 영향 |
|---|---|---|
| `navsim/agents/para_ssr/configs/default.py:283` + `navsim/planning/script/config/common/agent/para_ssr_agent.yaml:122` | `ref_data_root = /home/external-user/ssd/yongjae_refiner` | 학습 GT 루트. 두 값이 같아야 한다(`test_stageE.py::test_hydra_yaml_defaults_equal_dataclass`) |
| `navsim/agents/para_ssr/refiner/data.py:90-94` | `DATA_ROOT`, `REPO`, BEVFusion `cache_train_50x100` / `cache_val_50x100` | E2 teacher T, stage T 전체 |
| `navsim/agents/para_ssr/refiner/data.py:99-103` | `MC_ROOTS` = `$D/metric_cache`, `REPO/report/cause_and_correction_tests/E_train_split_feasibility/metric_cache`(없으면 건너뜀), `REPO/data/exp/metric_cache` | metric cache 탐색 |
| `navsim/agents/para_ssr/refiner/resmap_cache.py:47` | `RESMAP_ROOT=/home/external-user/datasets/teacher_cache/resmap` | E2 teacher M |
| `navsim/agents/para_ssr/refiner/sdf.py:102` | `SDF_ROOT=/home/external-user/ssd/yongjae_refiner/sdf` | SDF 빌드 기본 출력 |
| `tools/refiner/build_metric_cache.py:62,64-65,67,68,74,80-81,497` | ROOT, SAVED_CFG + sha, TRAIN_LOGS, TEST_LOGS, DATA, ENV, worker 상한 4 | stage T/E metric cache |
| `tools/refiner/score_trajectories.py:99,102-106,108,110,111-115` | NUPLAN_MAPS_ROOT 기본값, ROOT, `PDM_ATTR`(rescore_attr.py) sys.path, DATA, ARCHIVED_EVAL_CFG, MC_ROOTS | 채점, p_pdm |
| `tools/refiner/extract_human.py:52,54-62,292` | ROOT, TRAIN/TEST_LOGS, DATA, … / worker 상한 2 | human |
| `tools/refiner/build_future_objects.py:45,47,341`, `build_sdf.py:41,49-53,55` | ROOT, DATA, MC roots, 상한 4 | objects, SDF |
| `tools/refiner/make_splits.py:61,65,68`, `make_draft_bank.py:94,105`, `eval_refiner.py:70`, `stageT_gpu_commands.sh:20-22` | stage T | 경로 B |
| `tools/refiner/stageE_gpu_commands.sh:29-33, 69-70, 76-77, 167, 172` | PY, PATH, DATA, TEACH, KD_RUNS, human npz 검사, `NAVSIM_DOWNLOAD`, E0 CSV | stage E 전부 |
| `tools/refiner/stageE_prep.py:28-30`, `stageE_compare.py:21`, `stageE_parity.py:33`, `stageE_parity_e12.py:17,20`, `stageE_smoke.sh:8,14` | snapshot, LOGMAP(`splits/navtest.parquet`), golden | stage E 보조 |
| `report/perception_reliability/pdm_attr/rescore_attr.py:42` | ROOT | 채점 |
| `report/refiner_T/stageE_diag/dump_e2_navtest.py:31,37-38,45,50`, `verify_e2_dump.py:11-14`, `tools/refiner/e0_teacher_refine_report.py:34-37` | 진단 | 선택 |
| `tools/refiner/epdms_navtest.py:54-57` | navsim_v2 worktree(EPDMS) | 선택, §10 |
| `tools/refiner/tests/*.py`(약 30곳) | 같은 경로 | pytest |
| 생성되는 파일 | `data/exp/metric_cache/metadata/code/hydra/config.yaml`(sha가 assert됨), `metric_cache_metadata_node_0.csv`(절대경로가 들어감), 각 pkl 안의 `file_path` | 경로가 다르면 sha가 깨진다(§4.5) |

### 3.2 권장 레이아웃: 같은 절대경로를 symlink로 만든다

새 서버의 사용자명이나 디스크가 달라도, 아래 5개 경로만 같은 자리에 있으면 코드를 고칠 필요가 없다. 예시에서 `<FAST>`는 큰 NVMe, `<HOME>`은 실제 홈이다. 사용자명이 `external-user`가 아니면 `/home/external-user`를 만들 때 root 권한이 한 번 필요하다.

```bash
# (사용자명이 다를 때만) sudo mkdir -p /home/external-user && sudo chown $USER:$USER /home/external-user
mkdir -p /home/external-user/yongjae /home/external-user/datasets
# 1) miniconda (conda는 경로 이동에 약하다. 처음부터 이 경로에 설치하거나 디렉터리째 symlink)
[ -e /home/external-user/miniconda3 ] || ln -s <HOME>/miniconda3 /home/external-user/miniconda3
# 2) 데이터 루트(약 60 GB 이상 필요, stage T까지 하면 +30 GB) -> 빠른 디스크
mkdir -p <FAST>/yongjae_refiner && mkdir -p /home/external-user/ssd && ln -s <FAST>/yongjae_refiner /home/external-user/ssd/yongjae_refiner
# 3) NAVSIM 다운로드
ln -s <NAVSIM_DOWNLOAD_ROOT> /home/external-user/navsim/download     # 필요하면 mkdir -p /home/external-user/navsim 먼저
# 4) teacher cache
ln -s <TEACHER_CACHE_ROOT> /home/external-user/datasets/teacher_cache   # 안에 bevfusion/, resmap/
# 5) repo는 아래 §3.3에서 /home/external-user/yongjae/SSR 로 clone
```
주의: `TeacherCache`(`data.py:146-169`)는 경로와 **resolve된 경로**에 `_future`가 들어 있으면 거부한다. symlink 대상 이름에 `_future`가 들어가지 않게 한다.

### 3.3 clone, patch, 이관 파일 배치, `data/` symlink

```bash
git clone -b exp-refine https://github.com/yyongjae/SSR.git /home/external-user/yongjae/SSR
cd /home/external-user/yongjae/SSR && git checkout 6568d4d5bde3de6d3c1c0a5906fe98a9df0825e5   # detached여도 무방. 작업 브랜치가 필요하면 git switch exp-refine
# 이관 묶음 (§1.2) 풀기: 같은 절대경로로 들어간다
M=/home/external-user/ssd/ssr_migration      # 묶음을 복사해 둔 곳
sha256sum -c $M/bundle.tar.sha256 && tar xf $M/bundle.tar -C / --keep-directory-symlink   # §3.2 symlink(yongjae_refiner 등)를 디렉터리로 덮어쓰지 않게 한다
git -C /home/external-user/yongjae/SSR apply $M/uncommitted.patch
git -C /home/external-user/yongjae/SSR status --short | grep '^ M'    # 4개: WoTE_agent.py, tests/test_aux_evaluation_runner.py, tests/test_para_ssr_grad_balance_state.py, tools/dump_navtest_trajectories.py
# data/ (gitignored) — 이 서버와 같은 형태
cd /home/external-user/yongjae/SSR
mkdir -p data/dataset data/exp work_dirs
ln -s /home/external-user/navsim/download/maps                  data/dataset/maps
ln -s /home/external-user/navsim/download/trainval_navsim_logs  data/dataset/navsim_logs
ln -s /home/external-user/navsim/download/trainval_sensor_blobs data/dataset/sensor_blobs
test -d data/dataset/maps && test -d data/dataset/navsim_logs/trainval && test -d data/dataset/sensor_blobs/trainval && echo OK
# 데이터 루트 하위 디렉터리
mkdir -p /home/external-user/ssd/yongjae_refiner/{metric_cache/logs,objects/logs,sdf/logs,human/logs,stageE,runs/logs}
# 이관 파일 확인
sha256sum /home/external-user/yongjae/SSR/report/perception_reliability/pdm_attr/rescore_attr.py   # 9dbd00cd606bb18c4413cf600131cd64d5f4cfce5f91b5cb02dd9d4f00f3c521
sha256sum /home/external-user/yongjae/SSR/work_dirs/para_ssr_interaction_final/lightning_logs/version_2/checkpoints/last.ckpt  # ee46147a1ee9a77a52c20176f6b44f5ea97270be6287f5b0b63fcdb401e5552c
```
- `data/navsim -> /home/external-user/datasets/navsim` symlink도 이 서버에 있지만 코드에서 쓰지 않는다. 만들지 않아도 된다.
- test split은 `data/dataset`에 링크하지 않는다. 평가와 metric cache는 `NAVSIM_DOWNLOAD` 또는 `navsim_log_path=`로 test 경로를 받는다(`default_evaluation.yaml:3-4`).
- `NUPLAN_DATA_ROOT`, `NUPLAN_EXP_ROOT`가 없으면 `~/nuplan/...`로 기본값이 잡힌다는 경고가 나온다(`navsim/planning/script/utils.py:99-108`). 무시해도 된다.

### 3.4 NAVSIM 다운로드 구조 (이 서버, 확인용)

```
/home/external-user/navsim/download/
├── maps/                           nuplan-maps-v1.0.json, sg-one-north/, us-ma-boston/, us-nv-las-vegas-strip/, us-pa-pittsburgh-hazelwood/
├── trainval_navsim_logs/trainval/  *.pkl 1,310개
├── trainval_sensor_blobs/trainval/ log 폴더 1,192개 (navtrain_current/history 1..32)
├── test_navsim_logs/test/          *.pkl 147개 (983 MB), 이 중 navtest가 136 log 사용
└── test_sensor_blobs/test/         log 폴더 148개
```
`download_maps.sh`는 nuplan-maps-v1.1.zip을 받아 `maps`로 이름을 바꾼다. 그래서 `NUPLAN_MAP_VERSION=nuplan-maps-v1.0`이다.

---

## 4. navtest 공식 metric cache (`/home/external-user/yongjae/SSR/data/exp/metric_cache`)

### 4.1 이 서버에서 일어난 일 [실측, `metadata/log.txt`, `run_metric_caching.log`, `code/hydra/overrides.yaml`]

| 실행 | 시간 (2026-09-13) | worker | 결과 |
|---|---|---|---|
| 1 | 14:07 → 멈춤 | 기본 `ray_distributed_no_torch` | "Starting ray local!"에서 멈췄다. 이 컨테이너에서는 Ray GCS가 뜨지 않는다 |
| 2 | 14:10 → 20:57 | thread pool 6 | GIL 때문에 사실상 직렬. 약 5,169개 |
| 3 | 20:58 → 21:53 | `max_workers=12`, `use_process_pool=true` | 나머지 6,977개. "All 12146 features and targets were cached successfully." |

이미 있는 `metric_cache.pkl`은 건너뛴다(`metric_cache_processor.py:250`). 그래서 같은 명령을 다시 실행하면 이어서 만든다.

### 4.2 실행 명령 (원본 override를 순서까지 그대로 재현)

```bash
cd /home/external-user/yongjae/SSR
export PATH=/home/external-user/miniconda3/envs/ssr/bin:$PATH
export PYTHONPATH=/home/external-user/yongjae/SSR:${PYTHONPATH:-}
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/home/external-user/yongjae/SSR/data/dataset/maps
export OPENSCENE_DATA_ROOT=/home/external-user/yongjae/SSR/data/dataset
export NAVSIM_DEVKIT_ROOT=/home/external-user/yongjae/SSR
export NAVSIM_EXP_ROOT=/home/external-user/yongjae/SSR/work_dirs
# (1) 먼저 sha를 확인한다. 파일을 쓰지 않는다
python navsim/planning/script/run_metric_caching.py --cfg job \
  worker=single_machine_thread_pool worker.max_workers=12 scene_filter=navtest split=test \
  navsim_log_path=/home/external-user/navsim/download/test_navsim_logs/test \
  cache.cache_path=/home/external-user/yongjae/SSR/data/exp/metric_cache worker.use_process_pool=true \
  hydra.run.dir=/tmp/hydra_dry hydra.output_subdir=null | sha256sum
# 기대: 92d42853253622b29eabf8e488a97741e350eb3d5a360cb45f69cb0a4b444f84  (다르면 §4.5)
# (2) 실제 실행
nohup python navsim/planning/script/run_metric_caching.py \
  worker=single_machine_thread_pool \
  worker.max_workers=12 \
  scene_filter=navtest \
  split=test \
  navsim_log_path=/home/external-user/navsim/download/test_navsim_logs/test \
  cache.cache_path=/home/external-user/yongjae/SSR/data/exp/metric_cache \
  worker.use_process_pool=true \
  > /home/external-user/yongjae/SSR/work_dirs/metric_cache_navtest.out 2>&1 &
```
- 시간: 약 60분 [추정]. 실행 3의 처리 속도(프로세스당 시나리오 하나에 2.2–2.3 s)와 가장 큰 chunk(약 1,620개)로 계산했다.
- `worker.max_workers`도 sha에 들어간다. 12가 아니면 sha가 바뀐다(6 + thread → `431a2e63…`). **중간에 죽어서 재개할 때도 같은 override를 쓴다.** `metadata/code/hydra/config.yaml`은 실행할 때마다 다시 써진다.
- 대안으로 repo 스크립트를 써도 된다: `PATH=/home/external-user/miniconda3/envs/ssr/bin:$PATH WORKERS=12 NAVSIM_DOWNLOAD=/home/external-user/navsim/download bash scripts/evaluation/cache_metric_navtest.sh worker.use_process_pool=true`. `overrides.yaml`의 순서는 달라지지만 config.yaml의 sha는 같다. 스크립트 기본값(WORKERS=6, thread pool)은 아주 느리니 쓰지 않는다.

### 4.3 검증 (기대값, 이 서버)

```bash
cd /home/external-user/yongjae/SSR
grep -n 'Completed dataset caching' data/exp/metric_cache/metadata/log.txt          # ... All 12146 features and targets were cached successfully.
find data/exp/metric_cache -name metric_cache.pkl | wc -l                           # 12146
ls -d data/exp/metric_cache/20* | wc -l                                             # 136
ls -d data/exp/metric_cache/20* | xargs -n1 basename | sort | sha256sum             # 448680c0ff6bb6d2a178d51a330f29512edd13d32c8f65d4f88ef805649f1d0c
wc -l < data/exp/metric_cache/metadata/metric_cache_metadata_node_0.csv            # 12147 (header 포함)
sed 1d data/exp/metric_cache/metadata/metric_cache_metadata_node_0.csv | awk -F/ '{print $(NF-1)}' | sort | sha256sum   # 19cf783cbae935fce54cc459f05be508cfb546b0d92e7a5a122d0fc0d8bd4419
sha256sum data/exp/metric_cache/metadata/code/hydra/config.yaml                     # 92d42853253622b29eabf8e488a97741e350eb3d5a360cb45f69cb0a4b444f84
sha256sum navsim/planning/script/config/common/scene_filter/navtest.yaml            # 61284edf5003c0291f843ce9817c822ba306609a62d54544223adae3fc7fc9cd
du -sh data/exp/metric_cache                                                        # 3.1G
```
재개한 적이 있으면 xz 무결성을 검사한다. 쓰다 만 pkl이 남을 수 있는데, 이 진입점은 그런 파일을 검사하지 않는다.
```bash
/home/external-user/miniconda3/envs/ssr/bin/python - <<'EOF'
import lzma, glob
bad=[]
for p in glob.glob('/home/external-user/yongjae/SSR/data/exp/metric_cache/*/unknown/*/metric_cache.pkl'):
    try:
        with lzma.open(p,'rb') as f:
            while f.read(1<<22): pass
    except Exception: bad.append(p)
print('n_bad', len(bad)); print(*bad[:20], sep='\n')
EOF
# 기대: n_bad 0. 손상된 파일은 지우고 §4.2 (2)를 같은 override로 다시 실행한다
```

### 4.4 이 cache를 쓰는 곳

- 공식 평가(`eval_para_ssr.sh:44`의 `METRIC_CACHE_PATH` 기본값)
- `build_metric_cache.py:64-65,237`: 이 cache의 `config.yaml`을 stage T/E metric cache의 정의(SAVED_CFG)로 쓰고 sha를 assert한다. **그래서 §5.4보다 먼저 있어야 한다.**
- `data.MC_ROOTS`(navtest centerline), `build_sdf.py:53`(navtest SDF), `score_trajectories.py:111-115`

### 4.5 함정

- **CSV와 pkl 안에 절대경로가 들어간다**(`MetricCacheLoader`, `navsim/common/dataloader.py:147-156`). 그래서 cache를 다른 위치로 옮기면 평가가 깨진다. 같은 경로에서 만든다.
- `run_pdm_score_gpu.py:59-61`은 scene token과 cache token의 **교집합만** 채점한다. cache가 덜 만들어져 있어도 조용히 일부만 채점한다. 평가 로그의 `Starting pdm scoring of 12146 scenarios`를 꼭 확인한다.
- 경로가 이 서버와 달라서 sha가 바뀌면 `build_metric_cache.py build`가 `"saved caching config changed"`로 멈춘다. 해결책은 둘 중 하나다. (a) 같은 경로를 쓴다(§3.2, 권장). (b) 이관한 `navtest_saved_cfg_config.yaml`(sha `92d42853…`)을 `data/exp/metric_cache/metadata/code/hydra/config.yaml`에 덮어쓴다. (b)는 §4.2를 다시 실행하면 다시 덮인다는 점에 주의한다.

---

## 5. stage-E 학습 데이터 (E1/E2 공통 GT, 85,109 token)

### 5.1 최종 산출물 (`D=/home/external-user/ssd/yongjae_refiner`)

| 산출물 | 경로 | 개수 | 크기 | 버전 |
|---|---|---|---|---|
| E2E token list | `$D/splits/e2e_train_trainlogs.parquet`(cols token, log, frame_idx, city) | 85,109 token, 978 log | 1.6 MB | sorted-token sha256 `c3fa309c…9c79` |
| metric cache | `$D/metric_cache/<log>/unknown/<token>/metric_cache.pkl` | 85,109 | 약 0.35 MB/token (85k면 약 30 GB) | SAVED_CFG `92d42853…` |
| GT 미래 물체 | `$D/objects/{train,navtrain,dev}/<token>.npz` | 합 85,109 | 약 49 KB/token | `gt_future_v1` |
| SDF | `$D/sdf/navtrain/<token>.npz` | 85,109 | 이 서버 7.3 GB(90,743개) | `e_grid_dac_exact_v1` |
| side store | `$D/e2e_side/<tok[:2]>/<token>.npz` + `status.json` | 85,109 | 1.4 GB | cl_xy, cl_valid, cl_n, p_pdm |
| 사람 경로 | `$D/human/e2e_train_trainlogs.npz` | 85,109 | 59.6 MB | – |

학습 때 읽는 곳은 `navsim/agents/para_ssr/refiner/e2e.py`의 `GTLoader`(215행 이후)다. objects는 `train → navtrain → dev` 순서로 찾고, SDF는 `sdf/navtrain`, side는 `e2e_side/`에서 읽는다. 셋이 모두 있어야 `ref_gt_ok=True`가 된다. human npz(`e2e.py:88`)는 `KD_DRAFT_SOURCE=human_mix` 또는 `REF_HUMAN_ONLY_UNTIL`을 쓸 때 필요하고, 두 옵션 모두 E1/E2 공식 run에서 켜져 있다.

**이 서버와 새 서버의 차이:** 이 서버에서는 stage T store가 먼저 있었다. 그래서 85,109개 중 26,366개를 재사용했고 58,743개만 새로 만들었다(objects는 navtrain 58,743 + train 19,732 + dev 6,634). 새 서버에서 stage T(경로 B)를 하지 않으면 **85,109개를 전부** 만든다. 이때 objects는 전부 `objects/navtrain`에 넣는다. 로더가 세 디렉터리를 모두 찾으므로 동작은 같다.

### 5.2 단계 (1): token list 도출 (10초)

**정의:** `run_training.build_datasets`와 같다. navtrain scene_filter(1,192 log, 103,288 token, history 4 / future 10, frame_interval 1, has_route)의 log_names와 `train_logs`(13,180개)의 교집합 978 log에 `filter_scenes`를 적용한다. 결과 85,109개는 E0 학습 로그의 "Num training samples: 85109"와 같다.

원래 스크립트는 repo에 없고 scratchpad에만 있었다. 아래는 원문에서 **두 곳만** 바꿨다. (a) 출력 경로를 env `E2E_TOKENS_OUT`으로 받는다. (b) E0 학습 config가 없으면 tracked 파일로 대체한다(두 입력이 같다는 것은 이 서버에서 확인함: 스칼라 5개, log_names 1,192, tokens 103,288, train_logs 13,180 모두 일치).

`splits/`를 이관 묶음으로 이미 받았다면 **scratch에 도출한 뒤 비교만** 한다.

```bash
mkdir -p /home/external-user/ssd/yongjae_refiner/_scratch
cat > /home/external-user/ssd/yongjae_refiner/_scratch/derive_e2e_tokens.py <<'EOF'
"""Derive the exact E2E training token list of para_ssr_interaction_final (build_datasets -> SceneLoader(filter_scenes))."""
import json, os, sys, time, pickle
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf
ROOT = Path("/home/external-user/yongjae/SSR"); sys.path.insert(0, str(ROOT))
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import filter_scenes
CFG = ROOT / "work_dirs/para_ssr_interaction_final/code/hydra/config.yaml"
SPLIT_YAML = ROOT / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
LOGS = ROOT / "data/dataset/navsim_logs/trainval"
OUT = Path(os.environ.get("E2E_TOKENS_OUT", "/home/external-user/ssd/yongjae_refiner/splits/e2e_train_trainlogs.parquet"))
yl = OmegaConf.load(SPLIT_YAML)
if CFG.exists():
    cfg = OmegaConf.load(CFG)
    sf = OmegaConf.to_container(cfg.scene_filter, resolve=False)
    train_logs = list(cfg.train_logs)
    assert sorted(yl.train_logs) == sorted(train_logs), "saved train_logs != split yaml"
else:   # tracked fallback (verified identical on the old server)
    sf = OmegaConf.to_container(OmegaConf.load(ROOT / "navsim/planning/script/config/common/scene_filter/navtrain.yaml"), resolve=False)
    train_logs = list(yl.train_logs)
log_names = sorted(set(sf["log_names"]) & set(train_logs)) if sf.get("log_names") is not None else sorted(train_logs)
print("scene_filter logs", len(sf["log_names"]), "tokens", None if sf.get("tokens") is None else len(sf["tokens"]),
      "train_logs", len(train_logs), "-> train filter logs", len(log_names), flush=True)
TOK = sf.get("tokens")

def one(log):
    f = SceneFilter(num_history_frames=sf["num_history_frames"], num_future_frames=sf["num_future_frames"],
                    frame_interval=sf["frame_interval"], has_route=sf["has_route"], max_scenes=None,
                    log_names=[log], tokens=TOK)
    sc = filter_scenes(LOGS, f)
    rows = []
    for t, fl in sc.items():
        fr = fl[sf["num_history_frames"] - 1]
        rows.append((t, log, int(fr.get("frame_idx", -1)), fr.get("map_location")))
    return rows

if __name__ == "__main__":
    import navsim.common.dataloader as DL
    DL.tqdm = lambda x, **k: x
    t0 = time.time()
    with Pool(6) as p:
        res = p.map(one, log_names, chunksize=4)
    rows = [r for rr in res for r in rr]
    df = pd.DataFrame(rows, columns=["token", "log", "frame_idx", "city"])
    assert not df.token.duplicated().any()
    print("tokens", len(df), "logs with tokens", df.log.nunique(), f"{time.time()-t0:.0f}s", flush=True)
    df.to_parquet(OUT, index=False)
    print(df.city.value_counts().to_dict())
EOF
cd /home/external-user/yongjae/SSR
E2E_TOKENS_OUT=/home/external-user/ssd/yongjae_refiner/_scratch/e2e_rederived.parquet \
OPENSCENE_DATA_ROOT=/home/external-user/yongjae/SSR/data/dataset NUPLAN_MAPS_ROOT=/home/external-user/yongjae/SSR/data/dataset/maps NAVSIM_EXP_ROOT=/tmp \
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python /home/external-user/ssd/yongjae_refiner/_scratch/derive_e2e_tokens.py 2>&1 | tail -8
# 기대:
#  scene_filter logs 1192 tokens 103288 train_logs 13180 -> train filter logs 978
#  tokens 85109 logs with tokens 978 10s
#  {'us-nv-las-vegas-strip': 62875, 'us-ma-boston': 9180, 'us-pa-pittsburgh-hazelwood': 7840, 'sg-one-north': 5214}
```
검증(복사본과 도출본 모두):
```bash
for f in /home/external-user/ssd/yongjae_refiner/splits/e2e_train_trainlogs.parquet /home/external-user/ssd/yongjae_refiner/_scratch/e2e_rederived.parquet; do
/home/external-user/miniconda3/envs/ssr/bin/python -c "
import pandas as pd,hashlib,sys; e=pd.read_parquet('$f')
print(len(e), e.token.nunique(), e.log.nunique())
print(hashlib.sha256('\n'.join(sorted(e.token)).encode()).hexdigest())
print(hashlib.sha256('\n'.join(sorted(f'{a}|{b}|{c}|{d}' for a,b,c,d in zip(e.token,e.log,e.frame_idx,e.city))).encode()).hexdigest())"; done
# 기대: 85109 85109 978 / c3fa309cbde971e80661526c83f518b2f9df0248528086174d2f6c4e06229c79 / 79a63728f02702c65aca648ad5a0f5b7bcdd8b8e3795676869011a96de708016
```
`splits/`를 복사하지 않았다면 `E2E_TOKENS_OUT`을 빼고 실행해 `$D/splits/`에 바로 쓴다. parquet 파일 자체의 sha는 pyarrow 버전에 따라 달라질 수 있으므로, 위의 내용 sha로 비교한다.

### 5.3 단계 (2) 준비: todo / part 파일

이미 있는 metric cache와 objects를 건너뛰는 todo를 만든다. metric cache todo는 log 단위로 K개 part에 나눈다(greedy 균형). 이 서버에서는 K=2였다(2 프로세스 × 4 worker, `[29372, 29371]`). **새 서버에서는 CPU에 맞춰 K를 정한다.** 프로세스당 4 worker이므로 K=3이면 12 worker다.

```bash
cd /home/external-user/yongjae/SSR; K=3; /home/external-user/miniconda3/envs/ssr/bin/python -c "
import pandas as pd, os, numpy as np
from pathlib import Path
K=$K
D=Path('/home/external-user/ssd/yongjae_refiner')
e=pd.read_parquet(D/'splits/e2e_train_trainlogs.parquet')
has=np.array([os.path.exists(D/'metric_cache'/l/'unknown'/t/'metric_cache.pkl') for t,l in zip(e.token,e.log)])
todo=e[~has].copy(); print('mc todo',len(todo),'logs',todo.log.nunique())
c=todo.groupby('log').size().sort_values(ascending=False); parts=[[] for _ in range(K)]; n=[0]*K
for lg,k in c.items():
    i=int(np.argmin(n)); parts[i].append(lg); n[i]+=k
print('part sizes', n)
L=D/'metric_cache/logs'
for i in range(K):
    todo[todo.log.isin(parts[i])][['token','log']].reset_index(drop=True).to_parquet(L/f'e2e_mc_part{i+1}.parquet',index=False)
ob=D/'objects'; hasO=np.array([any((ob/s/f'{t}.npz').exists() for s in ('train','navtrain','dev')) for t in e.token])
e[~hasO][['token','log','frame_idx']].reset_index(drop=True).to_parquet(ob/'logs/e2e_objects_todo.parquet',index=False)
print('objects todo',(~hasO).sum())
"
# 기대 (stage T 없이): mc todo 85109 logs 978, part sizes 합 85109, objects todo 85109
# (stage T 경로 B를 먼저 했다면 이 서버와 같은 58743)
```
원래 코드에서 바뀐 점은 셋이다. part 수가 `K`이고, objects 존재 검사에 `navtrain`을 넣었고, E report cache 검사 줄(새 서버에는 그 디렉터리가 없음)을 뺐다.

**stage T(경로 B)를 이미 했다면** 시작하기 전에 stage-T summary를 백업한다. 병렬 프로세스가 `_state/build_summary.json`을 덮어쓰기 때문이다.
```bash
cp -p /home/external-user/ssd/yongjae_refiner/metric_cache/_state/build_summary.json /home/external-user/ssd/yongjae_refiner/metric_cache/_state/build_summary_stageT_backup.json
```

### 5.4 단계 (2): metric cache (가장 오래 걸림)

전제: §4가 끝나 `data/exp/metric_cache/metadata/code/hydra/config.yaml`의 sha가 `92d42853…`여야 한다.

```bash
cd /home/external-user/yongjae/SSR; D=/home/external-user/ssd/yongjae_refiner; P=/home/external-user/miniconda3/envs/ssr/bin/python; K=3
date +%s > $D/metric_cache/logs/e2e_start_epoch.txt
for i in $(seq 1 $K); do
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nohup nice -n 10 $P tools/refiner/build_metric_cache.py build --splits e2e_mc_part$i --splits-dir $D/metric_cache/logs --workers 4 --no-final-check > $D/metric_cache/logs/build_e2e_part$i.log 2>&1 &
  echo $! > $D/metric_cache/logs/build_e2e_part$i.pid
done
```
- 동작: `cache_scenarios()`를 SAVED_CFG로 부르고, `cache_path`, `navsim_log_path`(=`REPO/data/dataset/navsim_logs/trainval`), `output_dir`만 override한다. chunk는 (log, 최대 16 token) 단위이고 log 순서는 seed 0으로 섞는다. 끝난 token은 `_state/done_tokens.txt`에 append한다. env(`NUPLAN_MAPS_ROOT` 등)는 스크립트가 직접 설정한다(80-81행).
- **재개:** 같은 명령을 다시 실행한다. 이미 있는 것은 건너뛰고, done에 없는 pkl은 xz 검사를 해서 손상된 것만 지운다.
- E report cache가 없으면 `seeded_from_e 0`이 나온다. 정상이다.
- **`--no-final-check`가 필수다.** `check`(그리고 build 끝의 자동 check)는 `metric_cache/manifest.parquet`와 `metadata/metric_cache_metadata_node_0.csv`를 **덮어쓴다.** 이 둘은 stage-T(train/dev) 산출물이다. e2e part에 `check`를 실행하지 않는다.
- 시간: 이 서버에서는 8 worker, load 15–25에서 58,743개에 약 2.8 h(part1 10,149 s, part2 10,008 s)가 걸렸다. 합계 처리 속도는 약 5.8 tok/s였다 [실측]. 85,109개면 8 worker로 약 4 h, 12 worker로 약 2.7 h다 [추정].

진행 상황(읽기 전용):
```bash
D=/home/external-user/ssd/yongjae_refiner; P=/home/external-user/miniconda3/envs/ssr/bin/python; K=3   # §5.3과 같은 K
grep "^\[" $D/metric_cache/logs/build_e2e_part1.log | tail -1        # "... total ok N bad 0 left M | ... tok/s ETA h"
cd /home/external-user/yongjae/SSR && $P tools/refiner/build_metric_cache.py status --splits $(seq -s, -f 'e2e_mc_part%g' 1 $K) --splits-dir $D/metric_cache/logs
# 끝나면 기대: "done 85109/85109 failed-lines 0" (stage T 후라면 58743/58743). status는 done_tokens.txt만 읽는다
```
끝난 뒤:
- 각 로그 마지막 `done {...}` 줄이 `'bad': 0`이어야 하고, ok의 합이 pending의 합과 같아야 한다.
- `$D/metric_cache/_state/failed_tokens.tsv`가 생기지 않아야 한다.
- (stage T를 먼저 했다면) summary를 복원한다.
```bash
D=/home/external-user/ssd/yongjae_refiner; S=$D/metric_cache/_state; K=3; /home/external-user/miniconda3/envs/ssr/bin/python -c "
import ast,json
for i in range(1,$K+1):
    line=[l for l in open('$D/metric_cache/logs/build_e2e_part%d.log'%i) if l.startswith('done {')][-1]
    json.dump(ast.literal_eval(line[5:]),open('$S/build_summary_e2e_part%d.json'%i,'w'),indent=1)
"
[ -f $S/build_summary_stageT_backup.json ] && cp -p $S/build_summary_stageT_backup.json $S/build_summary.json && rm $S/build_summary_stageT_backup.json
```
새로 만든 cache 전체의 xz 검사(선택, 12 worker로 약 1분):
```bash
/home/external-user/miniconda3/envs/ssr/bin/python - <<'EOF'
import lzma, pandas as pd
from multiprocessing import Pool
from pathlib import Path
D=Path('/home/external-user/ssd/yongjae_refiner')
e=pd.read_parquet(D/'splits/e2e_train_trainlogs.parquet')
def ok(a):
    t,l=a; p=D/'metric_cache'/l/'unknown'/t/'metric_cache.pkl'
    try:
        with lzma.open(p,'rb') as f:
            while f.read(1<<22): pass
        return True
    except Exception: return False
with Pool(12) as p: r=p.map(ok, list(zip(e.token,e.log)), chunksize=64)
print('ok', sum(r), '/', len(r))     # 기대: ok 85109 / 85109
EOF
```

### 5.5 단계 (3): GT 미래 물체 (raw log만 필요. metric cache와 동시에 돌린다)

```bash
cd /home/external-user/yongjae/SSR; D=/home/external-user/ssd/yongjae_refiner; P=/home/external-user/miniconda3/envs/ssr/bin/python
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nohup nice -n 10 $P tools/refiner/build_future_objects.py --tokens $D/objects/logs/e2e_objects_todo.parquet --logs /home/external-user/yongjae/SSR/data/dataset/navsim_logs/trainval --out $D/objects/navtrain --workers 4 > $D/objects/logs/build_e2e.log 2>&1 &
echo $! > $D/objects/logs/build_e2e.pid
```
- 이 서버 [실측]: `--workers 1`로 58,743개, 604.7 s. `n_error 0, n_built 58743, A_max 802, tokens_gap 526, version gt_future_v1, npz_kb_mean 48.9`.
- 85,109개를 4 worker로 돌리면 약 5–15분 [추정].
- 검증: `$D/objects/navtrain/summary.json`이 `n_error 0`이고 `n_built`가 todo 수와 같아야 한다. `index.parquet`의 status는 전부 `built`여야 한다.
- 알려진 사항: 물체가 802개인 token이 하나 있다. 로딩할 때 `data.A_MAX=800`에서 2개가 잘린다. 그대로 둔다.
- `--mc-root`를 주지 않는다. 주면 untracked `sf_common.py`가 필요하다.

### 5.6 단계 (6): 사람 경로 (token list + raw log만 필요, 30초)

```bash
cd /home/external-user/yongjae/SSR; CUDA_VISIBLE_DEVICES="" nohup /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/extract_human.py extract --splits e2e_train_trainlogs --workers 2 > /home/external-user/ssd/yongjae_refiner/human/logs/extract_e2e_train.log 2>&1 &
```
- worker 상한은 2다. 3 이상을 주면 `at most 2 workers for extraction`으로 바로 끝난다.
- 기대 로그 [실측]: `{"split": "e2e_train_trainlogs", "n": 85109, ..., "logs_dir": "/home/external-user/yongjae/SSR/data/dataset/navsim_logs/trainval", "frame_idx_matches_split": true, "frame_gap": 735, "gap4": 582, "n_avail_ge16": 84121, "n_reg_ge16": 83005, "seconds": 27.9}`
- 출력: `$D/human/e2e_train_trainlogs.npz`(59.6 MB). npz의 `tokens` 순서는 parquet 행 순서와 같다.

### 5.7 단계 (4): SDF (metric cache가 끝난 뒤, K part × 4 worker)

간단한 방법: §5.4가 모두 끝난 뒤 한 번에 돌린다.
```bash
cd /home/external-user/yongjae/SSR; D=/home/external-user/ssd/yongjae_refiner; P=/home/external-user/miniconda3/envs/ssr/bin/python; K=3
$P -c "
import pandas as pd
from pathlib import Path
D=Path('$D'); e=pd.read_parquet(D/'splits/e2e_train_trainlogs.parquet')
m=e[[not (D/'sdf/navtrain'/f'{t}.npz').exists() for t in e.token]][['token','log']].reset_index(drop=True)
print('sdf todo',len(m))
for i in range($K): m.iloc[i::$K].to_parquet(D/f'sdf/logs/e2e_sdf_todo_part{i+1}.parquet',index=False)
"
for i in $(seq 1 $K); do CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nohup nice -n 10 $P tools/refiner/build_sdf.py --subset navtrain --tokens $D/sdf/logs/e2e_sdf_todo_part$i.parquet --workers 4 > $D/sdf/logs/build_navtrain_e2e_final_part$i.log 2>&1 & echo $! > $D/sdf/logs/build_navtrain_e2e_final_part$i.pid; done
```
- 이 서버 [실측]: 12 worker, 약 29 tok/s, error 0. 85,109개면 약 50분 [추정].
- 대안(이 서버에서 실제로 쓴 방식): metric cache와 **동시에** follow 모드로 띄운다. MC가 생기는 대로 뒤따라 만든다.
  ```bash
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nohup nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/build_sdf.py --subset navtrain --tokens /home/external-user/ssd/yongjae_refiner/splits/e2e_train_trainlogs.parquet --workers 3 --follow-interval 300 --follow-max-h 16 > /home/external-user/ssd/yongjae_refiner/sdf/logs/build_navtrain_e2e.log 2>&1 &
  ```
  매 pass에서 MC가 없으면 `no_mc`, MC가 쓰인 지 60 s가 안 됐으면 `mc_fresh`로 건너뛴다. MC가 끝나면 이 프로세스를 kill하고 위의 part 방식으로 남은 것을 마무리한다. kill하면 로그 끝에 `BrokenPipeError`가 찍히는데 정상이다.
- 출력: `$D/sdf/navtrain/<tok>.npz`, `_build/stats.jsonl`, pass별 `summary_*.json`.
- 검증: 모든 summary의 status에 error가 없어야 하고, e2e token 85,109개 모두 npz가 있어야 한다.

### 5.8 단계 (5): e2e_side (중앙선 + p_pdm, metric cache 뒤. SDF와 병렬로 돌린다)

**전제: `report/perception_reliability/pdm_attr/rescore_attr.py`(이관 #4)가 있어야 한다.** 없으면 p_pdm 계산이 import 단계에서 실패한다.

```bash
cd /home/external-user/yongjae/SSR
# 소규모 시험 (이미 있는 MC만 처리, 없으면 no_mc)
OMP_NUM_THREADS=1 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/build_e2e_side.py build --limit 40 --workers 2 2>&1 | grep -v Warn | tail -3
# 본 실행 (MC가 다 있으면 한 번으로 끝난다)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nohup nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/build_e2e_side.py build --workers 4 > /home/external-user/ssd/yongjae_refiner/stageE/build_e2e_side.log 2>&1 &
```
- 기본 token은 `RD.DATA_ROOT/splits/e2e_train_trainlogs.parquet`, 출력은 `$D/e2e_side/`다(`build_e2e_side.py:36-37`). worker 상한은 없고 기본값은 6이다.
- 이미 있는 파일은 건너뛴다(원자적 쓰기). MC가 없으면 `no_mc`로 남는다. 그러면 같은 명령을 다시 실행한다.
- 이 서버에서 쓴 follow 스크립트(MC와 동시에 띄움)는 다음과 같다. 원문 그대로이고, 위치는 `$D/stageE/follow_side.sh`다.
  ```bash
  cat > /home/external-user/ssd/yongjae_refiner/stageE/follow_side.sh <<'EOF'
  #!/usr/bin/env bash
  # Stage E side store follower: rebuild missing tokens every 10 min until all 85,109 have a side file (max ~9 h).
  cd /home/external-user/yongjae/SSR
  for i in $(seq 1 54); do
    OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/build_e2e_side.py build --workers ${SIDE_WORKERS:-4} 2>&1 | grep -v -i warn
    n=$(python3 -c "import json;print(json.load(open('/home/external-user/ssd/yongjae_refiner/e2e_side/status.json'))['n_with_side'])")
    echo "$(date +%T) side files: $n"
    [ "$n" -ge 85109 ] && break
    sleep 600
  done
  EOF
  chmod +x /home/external-user/ssd/yongjae_refiner/stageE/follow_side.sh
  nohup /home/external-user/ssd/yongjae_refiner/stageE/follow_side.sh > /home/external-user/ssd/yongjae_refiner/stageE/follow_side.log 2>&1 &
  ```
  시스템 `python3`도 호출한다(status.json 읽기용).
- 속도 [실측]: 4 worker로 ok가 약 40–50 tok/s였다. 85,109개면 약 30–35분 [추정].
- 검증: `$D/e2e_side/status.json`의 `n_with_side`가 85109여야 한다. 이 파일에는 마지막 pass만 남는다.
- `return lib.line_locate_point(line, other)` RuntimeWarning은 무시해도 된다.
- `build_e2e_side.py check --n 50`(기대 `{"n_checked": 50, "n_mismatch": 0}`)은 stage-T 산출물(`scores/train.parquet`, `packed/train`)이 있어야 돌아간다. stage E만 하는 서버에서는 §5.10으로 대신한다.

### 5.9 teacher cache 확인 (이미 옮겨 둔 것, E2와 stage T에만 필요)

```bash
cd /home/external-user/yongjae/SSR
/home/external-user/miniconda3/envs/ssr/bin/python - <<'EOF'
import sys; sys.path.insert(0,'.')
from navsim.agents.para_ssr.refiner import data as RD
from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache
for s in ('navtrain','navtest'):
    t=RD.TeacherCache.for_subset(s); m=t.manifest
    print('bevfusion',s,t.root,t.sha_head,m['split'],m['num_samples_written'])
    r=ResmapCache.for_subset(s); print('resmap',s,r.root,r.sha_head,r.meta['split'],len(r.index))
tok=next(iter(ResmapCache.for_subset('navtest').tokens()))
print(RD.TeacherCache.for_subset('navtest').load_bev(tok).shape, ResmapCache.for_subset('navtest').load_bev(tok).shape)
EOF
# 기대:
#  bevfusion navtrain ... cddf943ffec8d6a8 train 103288 / resmap navtrain ... 0eaeda793402a804 train 126032
#  bevfusion navtest  ... cddf943ffec8d6a8 val   12146  / resmap navtest  ... 0eaeda793402a804 none  12146
#  (256, 50, 100) (256, 50, 100)
find /home/external-user/datasets/teacher_cache/bevfusion/cache_val_50x100/samples -name '*.npz' | wc -l     # 12146
find /home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100/samples -name '*.npz' | wc -l   # 103288
```
코드가 하는 검사:
- BEVFusion(`data.py:146-169`): 경로에 `_future`가 있으면 거부한다. `checkpoint_sha256_head == "cddf943ffec8d6a8"`, layout `samples/<token[:2]>/<token>.npz`, `[50,100]`, 256 channel, f16이어야 한다.
- ReSMap(`resmap_cache.py:120-145`): `checkpoint_sha256 == 0eaeda793402a804925b46861a91a3ebe011e608d3b3726604f400be6f296f1a`, split(navtrain=`train`, navtest=`none`), classes, pc_range, `bev (256,100,50) f16`, `len(index.json)==num_frames`여야 한다.
- 크기 [실측]: bevfusion train 255G, val 30G, resmap 전체 352G(navtest 31G 포함). `cache_val_50x100_future`는 코드가 거부하므로 필요 없다.
- E2의 85,109 token은 두 cache 모두에 있어야 한다(`stageE_impl_plan.md` §4: "85,109 / 85,109 tokens in both caches").

### 5.10 단계 (7): 학습 로더 전수 검사 (가장 중요)

```bash
cd /home/external-user/yongjae/SSR
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 /home/external-user/miniconda3/envs/ssr/bin/python - <<'EOF'
import sys, json, numpy as np, pandas as pd
from multiprocessing import get_context
from pathlib import Path
sys.path.insert(0, "/home/external-user/yongjae/SSR")
DATA = Path("/home/external-user/ssd/yongjae_refiner"); _L = {}
def gt_chunk(toks):
    from navsim.agents.para_ssr.refiner.e2e import GTLoader
    if "L" not in _L: _L["L"] = GTLoader(DATA)
    out = []
    for t in toks:
        o = _L["L"].load(t)
        out.append((t, bool(o["ref_gt_ok"]), int(o["ref_obj_n"]), float(o["ref_p_pdm"]), int(o["ref_cl_n"]), float(o["ref_sdf"].float().abs().max())))
    return out
e = pd.read_parquet(DATA / "splits/e2e_train_trainlogs.parquet"); toks = list(e.token)
with get_context("fork").Pool(6) as p:
    rows = [r for rr in p.imap_unordered(gt_chunk, [toks[i:i+500] for i in range(0, len(toks), 500)]) for r in rr]
g = pd.DataFrame(rows, columns=["token","ok","obj_n","p_pdm","cl_n","sdf_absmax"])
print(json.dumps(dict(gt_n=len(g), gt_ok=int(g.ok.sum()), p_pdm_nonfinite=int((~np.isfinite(g.p_pdm)).sum()),
      cl_n_lt2=int((g.cl_n<2).sum()), sdf_absmax_zero=int((g.sdf_absmax==0).sum()), sdf_absmax_max=float(g.sdf_absmax.max()),
      obj_n_max=int(g.obj_n.max()), obj_n_zero=int((g.obj_n==0).sum()), p_pdm_zero=int((g.p_pdm==0).sum()), p_pdm_max=float(g.p_pdm.max())), indent=1))
EOF
```
기대값 [실측 `side_gt_summary.json`]: `gt_n 85109, gt_ok 85109, p_pdm_nonfinite 0, cl_n_lt2 0, sdf_absmax_zero 0, sdf_absmax_max 10.0, obj_n_max 800, obj_n_zero 68, p_pdm_zero 4370, p_pdm_max 67.82`. 약 43 s가 걸린다.

stage T 산출물까지 있는 서버라면 manifest 스크립트(부록 A.1)로 `$D/splits/e2e_train_trainlogs_status.json`도 만들 수 있다. 기대값은 모든 part가 `done 85109, missing 0`이다.

---

## 6. teacher 교정기 (R_T, R_M: stage T run 4)

### 6.1 무엇을 고를까

| | **(A) 스냅숏 복사 [권장]** | (B) stage T 전체 재구축 |
|---|---|---|
| 하는 일 | `$D/stageE/teachers/`(27 MB)를 이관 묶음으로 가져온다 | splits(복사) → MC → human/objects/SDF → draft bank + label → pack → 학습 → OOF 평가 → liveness → snapshot |
| 결과 | **비트 단위로 같은** teacher. report 38의 E2와 직접 비교할 수 있다 | 같은 절차로 만든 다른 표본. 비트 재현은 안 된다 |
| 시간 | 수 분 | CPU 약 5–6 h + GPU 약 1.5 h(병렬로 돌릴 때) [추정] |
| 잃는 것 | run 디렉터리(ckpt_last, log.jsonl, `eval_train_fold0/pred.npz`)가 없다. liveness, OOF를 다시 계산할 수 없고, 미뤄 둔 stage T 분석(C1/C2, shuffle, TM 재개, θ 선택)도 할 수 없다 | parity_golden_e12 비교가 실패한다. E2 수치를 seed 수준 차이로 해석해야 한다 |

어느 쪽을 고를지는 사용자가 정한다. 캐시를 새로 만든다는 방침과 별개로, teacher는 캐시가 아니라 학습된 모델이다.

### 6.2 경로 A: 스냅숏 복사 + 검증

이관 묶음(§1.2)에 이미 들어 있다. 검증:
```bash
cd /home/external-user/ssd/yongjae_refiner/stageE/teachers && /home/external-user/miniconda3/envs/ssr/bin/python -c "
import json,hashlib;d=json.load(open('sha256.json'))
print(all(hashlib.sha256(open(f'{r}/{f}','rb').read()).hexdigest()==h for r,v in d.items() for f,h in v['files'].items()))"   # True
```
기대 sha256(`sha256.json`):
```
stageT4_T_fold0_seed0: config.json  636c2a237494e9cd5da4230fa5178d8e1e38b0044cd3e1a67e3ddc0535a502e4
                       ckpt_best.pt 21f630284bc9111073ee8f7fbf650dd20ed06ee30f74d6dde6315defd2a713c4
                       norm.npz     fe55a1b7bce6cb4ca04c8cccd4216de0c9c4cab491ba51607a5af5dd0f1a4a62
stageT4_M_fold0_seed0: config.json  92f617d428149a4e3cade7938eed248f3c1e3af31fbc1dd37274b63eff1704df
                       ckpt_best.pt 7ee602ddf0a814f586e158dc45bfbe1c8aaa0b41fc77810e0892dc6ac995f750
                       norm_map.npz ea044dd9768a57b6914acce56df193b9a08472e895de06b2418cf91e0aafd527
```
- `load_run_model`(`train_refiner.py:410`)은 config.json의 arm/seed/net_kw와 norm 파일만 쓴다. config 안의 절대경로는 로딩에 영향이 없다.
- 경로 A에서는 `stageE_gpu_commands.sh snapshot`(이미 있으면 거부)과 `gate`(run 디렉터리 필요)를 **실행하지 않는다.**
- 이 서버의 gate 기록(`liveness_*.json`): R_T liveness 0.15764(7,520 / 47,705), R_M 0.12544(5,984 / 47,705). 둘 다 ≥ 0.01이라 통과했다.

### 6.3 경로 B: stage T 전체 재구축

전제: §2, §3, §4 완료. `splits/` 복사 완료(다시 만들 수 없다: `make_splits.py`의 pool `report/cause_and_correction_tests/E_train_split_feasibility/tokens/navtrain_token_log.parquet`와 2.9 GB E metric cache가 untracked이고, 선택 규칙이 "E cache가 있는 token 우선"이다. `navtest.parquet`도 untracked `report/head_ablation_scenes/table.npz`에 의존한다). `rescore_attr.py` 배치 완료.

split sha256(복사 후 확인):
```bash
cd /home/external-user/ssd/yongjae_refiner/splits && sha256sum train.parquet dev.parquet train_trainlogs.parquet dev_trainlogs.parquet navtest.parquet log_assignment.parquet
# f29b8000b6b0f9a28c7c96a65d7a275e6ac4830b8dc1f5e13ac5337a88b6b4b3  train.parquet
# 9126d23c80f4f000e945be0a640c77ea840aafa374cdd23e55bdf56cd7cc2796  dev.parquet
# fe61e768b320e4aeb0493c739482e4a2d3afb4efccca48b5c1fda9c434957637  train_trainlogs.parquet
# 8d75435fdb384b0c1fd9a671e264a39d827d8d312d4ec9b25313303bf6020d51  dev_trainlogs.parquet
# 258840dafdc8842aaf9ab4468a020126880ad973b68f668b2e56c81fc77362d2  navtest.parquet
# 2427c2c863f585f22e19b6ea1bfc96e9241700465ec700c5cb965e8e8e7624e3  log_assignment.parquet
```
`splits/summary.json` 기대값: train 24,000 / 891 log, dev 8,000 / 295 log, fold 0 = 4,728.

모든 명령은 `cd /home/external-user/yongjae/SSR`에서 실행한다. 출력은 worker 수와 무관하게 token별로 결정적이다. ①②③은 동시에 시작할 수 있고, ④⑤는 ①을 뒤따라가며(follow), ⑥은 모두를 뒤따라간다.

**① metric cache (train+dev 32,000)**, 약 4.3 h (2.054 tok/s, w4) [실측 속도]
```bash
mkdir -p /home/external-user/ssd/yongjae_refiner/metric_cache/logs
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nohup nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/build_metric_cache.py build --workers 4 --chunk 16 > /home/external-user/ssd/yongjae_refiner/metric_cache/logs/build.log 2>&1 &
```
- 끝나면 자동으로 check가 돈다. 기대(build.log 마지막): `check: {'exists': {'dev': 8000, 'train': 24000}, 'ok': {'dev': 8000, 'train': 24000}}`, `bad 0`. `$D/metric_cache/manifest.parquet`가 생긴다.
- E cache가 없으므로 `seeded from E 0`이 나온다.
- 더 빨리 하려면 split을 part로 나눌 수 있지만, 그러면 manifest가 part별로 덮인다. 이 방식은 시험해 보지 않았다.

**② human**, 약 1분
```bash
mkdir -p /home/external-user/ssd/yongjae_refiner/human/logs
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/extract_human.py extract --splits train,dev --workers 2 > /home/external-user/ssd/yongjae_refiner/human/logs/extract.log 2>&1
# 기대: train n 24000, frame_gap 180, frame_idx_matches_split true / dev n 8000, frame_gap 70. human/train.npz 16.8 MB, dev.npz 5.6 MB
```

**③ objects** [재구성: 원래 loop 스크립트는 남아 있지 않다. log 헤더와 summary 기준], 약 4분
```bash
D=/home/external-user/ssd/yongjae_refiner; mkdir -p $D/objects/logs
for S in dev train; do echo "=== $S $(date)" >> $D/objects/logs/build_full.log
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/build_future_objects.py --tokens $D/splits/$S.parquet \
    --logs /home/external-user/yongjae/SSR/data/dataset/navsim_logs/trainval --out $D/objects/$S --workers 2 >> $D/objects/logs/build_full.log 2>&1
  echo "=== $S exit $? $(date)" >> $D/objects/logs/build_full.log; done
# 기대: train n_tokens 24000 n_error 0 A_max 781 (~185 s) / dev n_error 0 A_max 592 (~60 s)
```

**④ SDF (follow)**, ① 종료 후 약 +0.5 h
```bash
mkdir -p /home/external-user/ssd/yongjae_refiner/sdf/logs
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nohup nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/build_sdf.py --subset navtrain --tokens /home/external-user/ssd/yongjae_refiner/splits/train.parquet --tokens /home/external-user/ssd/yongjae_refiner/splits/dev.parquet --workers 2 --follow-interval 900 --follow-max-h 12 > /home/external-user/ssd/yongjae_refiner/sdf/logs/build_navtrain.log 2>&1 &
# 기대: 마지막 summary에 status built만 있고 no_mc 0, n_invalid_polys_total 0, sdf_version e_grid_dac_exact_v1
```

**⑤ draft bank + 공식 label (follow)**, ① 종료 후 약 +0.5–1 h
```bash
mkdir -p /home/external-user/ssd/yongjae_refiner/drafts/logs
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nohup nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/make_draft_bank.py run --splits train,dev --workers 2 --follow-interval 900 --follow-max-h 12 > /home/external-user/ssd/yongjae_refiner/drafts/logs/run.log 2>&1 < /dev/null &
# 기대: drafts/train/_config.json cfg_hash 3b7db8d7e630
#       [run train] drafted 23820/23820, scored complete=True rows=309660
#       [run dev] drafted 7930/7930, scored complete=True rows=103090
#       all splits drafted and scored     (frame-gap token train 180, dev 70 제외)
```

**⑥ pack (follow, 다 끝나면 스스로 멈춤)**, 마지막 상류 종료 후 약 +0.5 h
```bash
mkdir -p /home/external-user/ssd/yongjae_refiner/packed/logs
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nohup nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/pack_follow.py --splits dev,train --workers 2 --interval 900 --max-h 12 > /home/external-user/ssd/yongjae_refiner/packed/logs/follow.log 2>&1 &
grep -E '"complete"|"ready"|"n_usable"' /home/external-user/ssd/yongjae_refiner/packed/{train,dev}/follow_status.json
# 기대: train complete true, ready 23820 (part 6종 각 23820), dev ready 7930. format refiner_pack_v2. train 9.4 GB, dev 3.2 GB
```
`stageT_gpu_commands.sh:8`이 말하는 `pack_when_ready.sh`는 존재하지 않는다. 상류가 이미 다 끝났다면 `/home/external-user/miniconda3/envs/ssr/bin/python -m navsim.agents.para_ssr.refiner.data --split train --workers 4`(dev도 같은 방식)로 한 번에 실행해도 된다.

**⑦ train_logs subset 확인**, 10초
```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/make_trainlogs_splits.py
# 복사한 *_trainlogs.parquet와 내용이 같은지만 확인한다 (다르면 거부, --force로 덮어씀). pack 뒤에 실행한다
# 기대 splits/trainlogs_summary.json: train n 19732 / 729 log, packed_trainable_trainlogs 19588; dev n 6634 / 243 log, 6569; navtest 12146/12146
```

**⑧ run-4 학습 (GPU)**: T와 M을 GPU 하나에서 동시에 돌린다. 약 1.5 h [실측: T 89분, M 86분]. `TAG=stageT4`가 필수다(stageT, stageT2, stageT3이면 exit 2).
```bash
cd /home/external-user/yongjae/SSR
TAG=stageT4 nohup tools/refiner/stageT_gpu_commands.sh 0 T 0 0 --token-subset /home/external-user/ssd/yongjae_refiner/splits/train_trainlogs.parquet --dev-token-subset /home/external-user/ssd/yongjae_refiner/splits/dev_trainlogs.parquet --gate-ttc 1 --m-col 0.15 --m-dac 0.05 --lon-st-slope 0.1 --w col=1,dac=1,prog=2,cmf=0.1,mod=0.1,ttc=1 --m-ttc 0.15 > /home/external-user/ssd/yongjae_refiner/runs/logs/plan_stageT4_T.log 2>&1 &
TAG=stageT4 nohup tools/refiner/stageT_gpu_commands.sh 0 M 0 0 --token-subset /home/external-user/ssd/yongjae_refiner/splits/train_trainlogs.parquet --dev-token-subset /home/external-user/ssd/yongjae_refiner/splits/dev_trainlogs.parquet --gate-ttc 1 --m-col 0.15 --m-dac 0.05 --lon-st-slope 0.1 --w col=1,dac=1,prog=2,cmf=0.1,mod=0.1,ttc=1 --m-ttc 0.15 > /home/external-user/ssd/yongjae_refiner/runs/logs/plan_stageT4_M.log 2>&1 &
# (선택, baseline) arm none on GPU 1: 위 줄에서 "0 T" -> "1 none", 로그 이름 plan_stageT4_none.log
```
- 스크립트가 하는 일: `train_refiner.py --arm <A> --fold 0 --seed 0 --gpu <G> --workers 2 --tag stageT4 …`로 학습한다(로그 `$D/runs/logs/stageT4_<A>_fold0_seed0.train.log`). 그다음 `flock $D/runs/.eval.lock … eval_refiner.py all --run … --split train --fold 0 --token-subset …/train_trainlogs.parquet --gpu <G> --workers 4 --theta 0.5 --sweep --budget-ep 0.5`로 OOF 평가를 한다(`.eval.log`). `plan_stageT4_*.log`는 비어 있는 것이 정상이다.
- `--gpu`는 프로세스에서 보이는 CUDA index다.
- teacher norm 통계는 학습을 시작할 때 자동으로 계산된다(`--n-norm 2048`, `norm.npz` / `norm_map.npz`).
- 재개: 같은 명령을 다시 실행하면 `ckpt_last.pt`에서 이어간다. config가 다르면 거부한다.
- config.json 기대값: token_subset n_rows 19,732, n_packed_rows 19,588, n_train_rows 14,161, n_ival_rows 1,532, `sha256 fe61e768…`. gate `pi 0.18033`, `pos_weight 4.54545`. 파라미터 수 total 3,416,333. `teacher_sha_head` T `cddf943ffec8d6a8`, M `0eaeda793402a804`. `code` 해시 11개(adapters `741e850dfa2901a3`, data `271a75259692b352`, decoder `2b605c82a79de8e4`, surrogate `1c266dc1902444bb`, train_refiner `8b2f2ed737f05f32` 등).
- 이 서버 결과 [실측]: T는 epochs_run 32, best epoch 25, inner-val 0.32923. M은 31, 24, 0.35373. epoch당 1,770 step, 약 166 s.
- OOF 기대(eval.log 마지막, θ 0.5): n_tokens 3,895, n 47,705, pdms_orig 0.82000, T 0.89623, M 0.89539, none 0.87302.

**⑨ gate → snapshot**
```bash
cd /home/external-user/yongjae/SSR
bash tools/refiner/stageE_gpu_commands.sh gate        # liveness.py --min-liveness 0.01 -> $D/stageE/liveness_<run>.json
bash tools/refiner/stageE_gpu_commands.sh snapshot    # DONE 필요, write-once(이미 있으면 거부), 0444, sha256.json
```
- gate 기대 범위: liveness T 약 0.16, M 약 0.13, 둘 다 `gate_pass true`. false면 멈추고 보고한다.
- **경로 A로 받은 `stageE/teachers/`가 이미 있으면 snapshot이 거부한다.** B를 할 거면 A의 사본을 다른 이름으로 옮겨 둔다.
- 다시 학습한 teacher는 sha가 다르다(`train_refiner.py:614,643`은 seed만 고정한다. cuDNN deterministic 설정이 없고 fp16 autocast를 쓴다). 그러면 λ_c를 E2 pilot으로 다시 재야 할 수 있고, `stageE_parity_e12.py check`는 실패한다.

---

## 7. 학습 실행

### 7.1 학습 전 검증 (CPU)

**(1) 코드가 E2를 돌린 코드와 같은지:** `report/refiner_E_code_audit/source_sha256.json`(tracked, E2를 띄운 직후 찍은 150개 파일)과 비교한다.
```bash
cd /home/external-user/yongjae/SSR && /home/external-user/miniconda3/envs/ssr/bin/python - <<'EOF'
import json,hashlib,os
d=json.load(open('report/refiner_E_code_audit/source_sha256.json'))
bad=[k for k,v in d.items() if not os.path.exists(k) or hashlib.sha256(open(k,'rb').read()).hexdigest()!=v]
print('total',len(d),'differ/missing',bad)
EOF
# 기대: total 150 differ/missing []   (경로를 고친 파일만 다르게 나와야 한다)
```

**(2) pytest**
```bash
cd /home/external-user/yongjae/SSR
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python -m pytest -q tools/refiner/tests
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python -m pytest -q tools/refiner/tests/test_stageE.py
```
- 마지막 기록은 `330 passed, 19 warnings in 154.09s`(`report/refiner_T/STATUS.md:539`)다. 그 뒤 stage-E 테스트가 추가돼 현재 개수는 확인하지 못했다.
- 데이터에 의존하는 테스트가 많다. 예: `DATA/runs/stageT3_T_fold0_seed0/ckpt_best.pt`(stage T run 3, 경로 A에는 없음), `DATA/human/train.npz`, `packed/`, GT token `1aa44d46e4ab5bc7`, `153c6b07f09d53d1`, golden 파일. 일부는 skip되지만 fail로 나올 수도 있다(확인 안 함).
- 새 서버에서는 **fail마다 원인이 "없는 stage-T 데이터"인지 확인한다.** 그 밖의 fail은 환경 문제다.

**(3) parity (선택, golden을 이관했을 때)**
```bash
cd /home/external-user/yongjae/SSR
OMP_NUM_THREADS=8 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/stageE_parity.py check        # 기대: ... IDENTICAL, exit 0
/home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/stageE_parity_e12.py check                     # 기대: E1/E2 step 0,1 모두 IDENTICAL
```
- golden은 CPU 부동소수 합산 순서에 민감하다. CPU나 MKL이 다르면 코드가 같아도 MISMATCH가 날 수 있다. 하드웨어 차이 때문이라면 `parity_golden.pt`는 commit `262ffa6` worktree에서 다시 만들 수 있다(§10 #9, [재구성], 시험해 보지 않음). `parity_golden_e12.pt`는 다시 만들 수 없다.

**(4) (선택) CPU smoke:** `OUT=/home/external-user/ssd/yongjae_refiner/_scratch/smoke bash tools/refiner/stageE_smoke.sh E2`. token 6개, batch 2, `wandb.enable=false`.

### 7.2 E2 학습 (실제로 실행한 설정)

[재구성] 실행 이름, hydra overrides, REVISION 1(GPU 0,1,4,5)과 정확히 일치한다. 새 서버에서는 비어 있는 GPU 4장을 준다.
```bash
cd /home/external-user/yongjae/SSR
/home/external-user/miniconda3/envs/ssr/bin/wandb login       # 처음 한 번 (§2.4)
KD_BALANCE=ema KD_START_EPOCH=0 KD_RATIO_RAMP=5 KD_WEIGHT_MAX=100 \
REF_HUMAN_ONLY_UNTIL=5 KD_DRAFT_SOURCE=human_mix REF_BEV_GRAD_SCALE=1.0 \
nohup bash tools/refiner/stageE_gpu_commands.sh train4 E2 0,1,2,3 - decoded > /home/external-user/yongjae/SSR/work_dirs/stageE_E2_launch.out 2>&1 &
```
- `-`는 λ_c 자리다. `KD_BALANCE=ema`에서는 반드시 `-`를 넘긴다.
- `KD_RATIO`(기본 1.0), `GRAD_SHARE_EVERY`(config 기본 50)는 설정하지 않았다.
- 실행 이름은 `stageE_E2_30ep_ema_r1.0_s0_rr5_wmax100_hmix_hwu5_bg1.0`이 된다. 출력은 `work_dirs/<이름>/`이다.
- tmux를 쓰면 로그가 터미널에도 남는다. 이 서버에서는 tmux 세션 `refine`에서 돌렸다.

풀어 쓴 동등 명령(`work_dirs/stageE_E2_.../code/hydra/overrides.yaml` 원문 + wrapper env). 디버깅할 때 참고한다.
```bash
export PYTHONPATH=/home/external-user/yongjae/SSR:$PYTHONPATH NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NUPLAN_MAPS_ROOT=/home/external-user/yongjae/SSR/data/dataset/maps \
  OPENSCENE_DATA_ROOT=/home/external-user/yongjae/SSR/data/dataset NAVSIM_DEVKIT_ROOT=/home/external-user/yongjae/SSR NAVSIM_EXP_ROOT=/home/external-user/yongjae/SSR/work_dirs \
  CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/external-user/miniconda3/envs/ssr/bin/python /home/external-user/yongjae/SSR/navsim/planning/script/run_training.py agent=para_ssr_agent agent.lr=1e-4 agent.config.max_epochs=30 \
  experiment_name=stageE_E2_30ep_ema_r1.0_s0_rr5_wmax100_hmix_hwu5_bg1.0 scene_filter=navtrain split=trainval \
  dataloader.params.batch_size=4 dataloader.params.num_workers=6 trainer.params.max_epochs=30 \
  trainer.params.accumulate_grad_batches=8 trainer.params.check_val_every_n_epoch=5 trainer.params.precision=32 \
  +trainer.params.devices=4 trainer.params.gradient_clip_val=35.0 trainer.params.gradient_clip_algorithm=norm \
  agent.config.use_task_interaction=true agent.config.use_det_motion_head=true agent.config.use_map_head=true \
  agent.config.refiner_mode=E2 agent.config.warmup_epochs=3 "agent.config.kd_ramp=[5.0,10.0]" \
  agent.config.grad_balance_warmup_iters=5300 agent.config.grad_balance_interval=100 agent.config.grad_norm_log_interval=100 \
  trainer.params.limit_val_batches=0 \
  "agent.config.kd_teacher_runs=[/home/external-user/ssd/yongjae_refiner/stageE/teachers/stageT4_T_fold0_seed0,/home/external-user/ssd/yongjae_refiner/stageE/teachers/stageT4_M_fold0_seed0]" \
  agent.config.kd_space=decoded agent.config.kd_balance=ema agent.config.kd_ratio=1.0 agent.config.kd_start_epoch=0 \
  agent.config.kd_ratio_ramp_epochs=5 agent.config.kd_weight_max=100 agent.config.kd_draft_source=human_mix \
  agent.config.ref_human_only_until=5 agent.config.ref_bev_grad_scale=1.0
```
- strategy는 config 기본값 `ddp`다(`default_training.yaml:76`). 학습 캐시는 쓰지 않는다(`cache_path: ''`, feature를 온라인으로 만든다).
- `kd_ramp=[5,10]`은 ema에서는 쓰이지 않는다. r은 epoch 0→5 동안 0→1로 오른다.

**시작 직후 확인할 것:**
- `run_training.log`에 `Num training samples: 85109`가 있다.
- `work_dirs/stageE_E2_launch.out`(nohup 출력, `run_training.log`가 아님)의 첫 줄이 `PARA-SSR launch: GPUs=4, batch/GPU=4, accumulate=8, global_batch=128`이고, `warning: effective global batch is …` 경고가 없다(`train_para_ssr.sh:72-75`).
- `work_dirs/<exp>/stageE_steps.jsonl`의 첫 레코드가 `ref/human_only 1.0`, `kd/ratio 0.0`, `ref/n_gt_ok 4.0`이다.
- `Loading pretrained weights from Hugging Face hub (timm/resnet50.tv_in1k)`가 찍힌다.
- epoch 0이 약 41분 걸린다.

**결과물 [실측]:** `lightning_logs/version_0/checkpoints/epoch={0..29}-step=*.ckpt` + `last.ckpt`(481 MB). step은 epoch마다 665씩 늘어 19950에서 끝난다. 그 밖에 `stageE_steps.jsonl`(402 MB, 159,600줄), `run_training.log`, `train_time.json`, `wandb/`가 생긴다. E2 `last.ckpt` sha256은 `057515ad4dad9dd63200ad3d7eaea77bac48d99107632bdd9401394015e223d1`이었다.

**재시작:** `RESUME_CHECKPOINT=<…/last.ckpt>`를 export한 뒤 같은 train4 명령을 다시 실행한다(`train_para_ssr.sh:58,89-91`). ema 상태는 callback state로 복원된다. 경로에 `=`가 들어가면 안 된다(§10 #5). jsonl은 append라서 재시작하면 레코드가 중복된다.

### 7.3 E1 학습 (아직 실행한 적 없음. E2에서 KD만 뺀 것)

```bash
cd /home/external-user/yongjae/SSR
KD_DRAFT_SOURCE=human_mix REF_HUMAN_ONLY_UNTIL=5 REF_BEV_GRAD_SCALE=1.0 \
nohup bash tools/refiner/stageE_gpu_commands.sh train4 E1 0,1,2,3 > /home/external-user/yongjae/SSR/work_dirs/stageE_E1_launch.out 2>&1 &
# 실행 이름: stageE_E1_30ep_hmix_hwu5_bg1.0
```
- **함정:** 코드 기본값은 `ref_bev_grad_scale 0.1`, `kd_draft_source tau0`, `ref_human_only_until null`이다. env 세 개를 빠뜨리면 E1이 조용히 다른 실험이 된다. KD_* env는 E1에서 무시된다.
- 검증: `diff work_dirs/stageE_E2_*/code/hydra/overrides.yaml work_dirs/stageE_E1_*/code/hydra/overrides.yaml`에서 `experiment_name`, `refiner_mode`, 그리고 kd 줄 7개(`kd_teacher_runs, kd_space, kd_balance, kd_ratio, kd_start_epoch, kd_ratio_ramp_epochs, kd_weight_max`)만 달라야 한다.
- 알고 받아들인 차이(REVISION 3): E2는 teacher를 로딩할 때 CUDA RNG를 한 번 다시 seed한다. 그래서 dropout mask가 step 2부터 다르다.

### 7.4 (선택) pilot150: 1 GPU, 150 micro-batch

```bash
cd /home/external-user/yongjae/SSR
KD_DRAFT_SOURCE=human_mix bash tools/refiner/stageE_gpu_commands.sh pilot150 0 decoded
```
- 기본값: `KD_BALANCE=ema`, `KD_START_EPOCH=0`, `GRAD_SHARE_EVERY=10`, `strategy=auto`, `wandb.enable=false`. 실행 이름은 `stageE_pilot150_E2_ema_r1.0_s0_hmix`이고, 디렉터리가 있으면 거부한다.
- 기대 [실측]: `sec_step_median 0.327`, `mem_gb_peak 13.418`, `L_sur_weighted 0.344`, `L_KD 0.0445`, `kd/w_ema 7.92`, `ref/frac_human 0.485`, `gt_ok_frac 1.0`.

### 7.5 GPU 수 규칙 (global batch 128 유지)

`128 = 4(batch/GPU) × nGPU × ACCUMULATE`. grad-balancer 카운터는 rank당 micro-batch를 센다. 그래서 `counter × acc / 16`으로 맞춘다(`stageE_gpu_commands.sh:98-104`).

| GPU | ACCUMULATE | warmup_iters / interval / log_interval | rank당 micro-batch/epoch |
|---|---|---|---|
| 1 | 32 | 21200 / 400 / 400 (+ `strategy=auto`) | 약 21,280 |
| 2 | 16 | 10600 / 200 / 200 (= E0) | 약 10,640 |
| **4** | **8** | **5300 / 100 / 100** (E2 실측) | **5,320** |
| 8 | 4 | 2650 / 50 / 50 | 약 2,660 |

- **`train4`는 `ACCUMULATE=8`과 `$(counters 8)`을 하드코딩한다**(145, 147행). GPU 수를 검사하지 않는다. GPU 2장을 주면 global batch 64로 돌고, 경고 한 줄만 찍힌다. 다른 GPU 수를 쓰려면 이 두 곳을 위 표대로 고친다.
- 3, 5, 6, 7장은 128을 정수로 나눌 수 없다. recipe 자체를 바꾸는 일이라 사용자가 결정한다.
- 비트 단위 재현은 4 GPU에서만 기대할 수 있다(DistributedSampler 분할이 GPU 수에 따라 달라짐).

### 7.6 속도, 메모리 [실측, 4× RTX 5090, 다른 작업과 경합]

| 항목 | 값 |
|---|---|
| s/micro-batch/GPU | 중앙값 0.4314, 평균 0.4428 |
| data wait | 중앙값 0.0088 s |
| peak GPU mem (rank 0) | 13.611 GB |
| epoch | 평균 40.2분(37.1–42.0) |
| 총 시간 | **20.13 h**(72,467.5 s). 09-29 23:17 → 09-30 19:25 |
| CPU | dataloader 6/GPU = 24 프로세스 |

### 7.7 모니터링

- wandb: project `para-ssr`, run 이름 = experiment_name(E2는 `kna7f8o2`). 키는 `train/loss`, `train/ref/*`, `train/kd/*`, `train/gnorm/*`, `stageE/{sec_step,mem_gb,skipped_steps}` 등이다.
- jsonl 요약(CPU, 읽기만 함): `/home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/stageE_prep.py pilot --steps work_dirs/<exp>/stageE_steps.jsonl --last 5320 --skip 0`. 마지막 1 epoch를 요약한다.
- E2 기준값(report 37 §8-9, report 38 §6):
  - KD : GT 벌점 = 0.018 : 0.019 (epoch 29)
  - `kd/l1_0`(R_T) 0.047→0.033, `kd/l1_1`(R_M) 0.041→0.027 (epoch 5→29)
  - `kd/w_ema`는 epoch 29에 약 0.6
  - `ref/zdead` 3.1 → 54
  - GT 100% ok, `skipped_steps 0`
  - BEV gradient 중 교정기 몫: epoch 0 약 2% → epoch 4 약 43% → epoch 29 약 23%

---

## 8. 평가와 비교

### 8.1 E0 재현 (환경과 cache 확인용으로 가장 좋음)

```bash
cd /home/external-user/yongjae/SSR
PATH=/home/external-user/miniconda3/envs/ssr/bin:$PATH CUDA_VISIBLE_DEVICES=0 \
NAVSIM_DOWNLOAD=/home/external-user/navsim/download EVAL_EXPERIMENT=eval/E0_repro_newserver \
bash scripts/evaluation/eval_para_ssr.sh \
  /home/external-user/yongjae/SSR/work_dirs/para_ssr_interaction_final/lightning_logs/version_2/checkpoints/last.ckpt \
  agent.config.use_task_interaction=true agent.config.use_det_motion_head=true agent.config.use_map_head=true
```
- 기대 로그: `Starting pdm scoring of 12146 scenarios...`, `Number of successful scenarios: 12146.`, `Number of failed scenarios: 0.`, `Final average score of valid results: 0.848683140828655.`
- 이 서버에서는 GPU 1장으로 약 30분 걸렸다.
- 기준 average 행(`work_dirs/eval/para_ssr_interaction_final/2026.09.17.00.09.41.csv`): NC 0.97859, DAC 0.93306, DDC 0.99996, EP 0.79773, TTC 0.93562, C 0.99992, **score 0.84868**.
- GPU나 torch가 다르면 끝자리가 달라질 수 있다. 소수점 4자리(0.8487)가 맞으면 된다.

token별 비교:
```bash
cd /home/external-user/yongjae/SSR && /home/external-user/miniconda3/envs/ssr/bin/python - <<'EOF'
import pandas as pd, glob
a=pd.read_csv('work_dirs/eval/para_ssr_interaction_final/2026.09.17.00.09.41.csv')
b=pd.read_csv(sorted(glob.glob('work_dirs/eval/E0_repro_newserver/*.csv'))[-1])
m=a.merge(b,on='token',suffixes=('_a','_b')); m=m[m.token!='average']
print(len(m), (m.score_a-m.score_b).abs().max(), b[b.token=='average'].score.values)
EOF
# 기대: 12146, max diff ≈ 0, average ≈ 0.84868
```

### 8.2 E2 / E1 평가 (final + tau0)

```bash
cd /home/external-user/yongjae/SSR
CKPT=/home/external-user/yongjae/SSR/work_dirs/stageE_E2_30ep_ema_r1.0_s0_rr5_wmax100_hmix_hwu5_bg1.0/lightning_logs/version_0/checkpoints/last.ckpt
bash tools/refiner/stageE_gpu_commands.sh eval E2 "$CKPT" final 0 stageE_E2        # -> work_dirs/eval/stageE_E2/<timestamp>.csv
bash tools/refiner/stageE_gpu_commands.sh eval E2 "$CKPT" tau0  1 stageE_E2_tau0   # -> work_dirs/eval/stageE_E2_tau0/<timestamp>.csv
# E1:
CKPT1=/home/external-user/yongjae/SSR/work_dirs/stageE_E1_30ep_hmix_hwu5_bg1.0/lightning_logs/version_0/checkpoints/last.ckpt
bash tools/refiner/stageE_gpu_commands.sh eval E1 "$CKPT1" final 2 stageE_E1
bash tools/refiner/stageE_gpu_commands.sh eval E1 "$CKPT1" tau0  3 stageE_E1_tau0
```
- 이름은 반드시 `stageE_<arm>[_tau0]` 형식으로 준다. compare가 `work_dirs/eval/stageE_<n>/` 안에서 가장 최근 CSV를 찾기 때문이다.
- 평가 때 teacher는 만들지 않는다. 필요한 것은 navtest metric cache뿐이다.
- 시간 [실측]: E2와 tau0을 GPU 2장에서 동시에 돌렸을 때 약 65분, 58분이었다.
- 기대 [실측, report 38]: E2 final **0.8615101502055243**, E2 tau0 0.8465570612679275, E0 0.8486831408286551.

### 8.3 비교

```bash
cd /home/external-user/yongjae/SSR
bash tools/refiner/stageE_gpu_commands.sh compare E2 E2_tau0          # E1이 있으면: compare E2 E2_tau0 E1 E1_tau0
# -> report/refiner_T/stageE_compare.json (커밋된 E2 결과는 report/refiner_T/stageE_eval/stageE_compare.json에 옮겨져 있다. 덮어쓰지 않도록 주의)
```
- 방식: 두 arm 모두에서 valid한 token만 쓴다. `splits/navtest.parquet`의 log로 군집 bootstrap(10,000회, seed 0)을 한다.
- E0 CSV 경로는 `stageE_gpu_commands.sh:172`에 하드코딩돼 있다. E0를 새로 평가한 CSV로 바꾸려면 이 줄을 고친다.
- `compare`는 `--contrast E2-E0 --contrast E2-E2_tau0`를 항상 붙인다(174행). 그래서 E2와 E2_tau0 평가가 모두 있어야 돈다. `work_dirs/eval/stageE_E1/`가 있으면 `E2-E1`, `E1-E1_tau0`도 자동으로 붙으므로 E1과 E1_tau0을 함께 넘긴다. E1만 있고 E2가 없으면 이 sub-command를 쓸 수 없다. 그때는 `stageE_compare.py --arm E0=<csv> --arm E1=<csv> --arm E1_tau0=<csv> --contrast E1-E0 --contrast E1-E1_tau0 --out <json>`를 직접 실행한다(인자는 `--help`로 확인함).
- 기대 [실측]: E2 − E0 **+1.28 [+0.74, +1.86]**, E2 − E2_tau0 +1.50 [+1.06, +2.03], E2_tau0 − E0 −0.21 [−0.85, +0.41]. 12,146 token, 136 log.
- 도시별 표: `report/refiner_T/stageE_eval/stageE_compare_ext.py --arm ... --contrast ... --out ...`

### 8.4 (선택) 진단

- E2 navtest dump(GPU 1장, 약 18분). 먼저 §8.2 eval을 돌려 `work_dirs/eval/stageE_E2/code/hydra/config.yaml`이 있어야 하고, 이관 patch(#9)가 필요하다.
  ```bash
  cd /home/external-user/yongjae/SSR
  /home/external-user/miniconda3/envs/ssr/bin/python report/refiner_T/stageE_diag/dump_e2_navtest.py --download /home/external-user/navsim/download --out-dir /home/external-user/ssd/yongjae_refiner/stageE_diag
  # 기대 e2_navtest_dump.meta.json: n_tokens 12146, checkpoint_sha256 057515ad…(같은 ckpt일 때), tau_final_eq_tau0_frac 0.04775
  ```
- 채점(CPU, 약 227 s):
  ```bash
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 /home/external-user/miniconda3/envs/ssr/bin/python tools/refiner/score_trajectories.py \
    --drafts /home/external-user/ssd/yongjae_refiner/stageE_diag/e2_score_drafts.npz --tokens /home/external-user/ssd/yongjae_refiner/stageE_diag/tokens.parquet \
    --out /home/external-user/ssd/yongjae_refiner/stageE_diag/scores.parquet --workers 4
  ```
  `tokens.parquet`(cols token, log, 12,146행)를 어떻게 만들었는지는 기록에 없다. `splits/navtest.parquet`의 token, log 열로 만들면 될 것으로 보인다(확인 안 함).
- E0 + frozen teacher 사후 실험(`refine_external_drafts.py predict --run …`): `$D/packed/navtest`와 stage-T run 디렉터리가 필요하다. 경로 A만으로는 불가능할 수 있다(확인 안 함). 기대값은 R_T +2.04, R_M +1.94(report 38 §3-1).

---

## 9. 검증 기준값 (한곳에 모음)

| 대상 | 명령 / 위치 | 기대값 |
|---|---|---|
| Python, torch | §2.5 | 3.9.23 / 2.8.0+cu128 12.8 91002 (2,27,3) sm_120 |
| `navsim.__file__` | §2.5 | `/home/external-user/yongjae/SSR/navsim/__init__.py` |
| git HEAD | `git -C /home/external-user/yongjae/SSR rev-parse HEAD` | `6568d4d5bde3de6d3c1c0a5906fe98a9df0825e5` (+ patch 4 파일) |
| code audit | §7.1 (1) | `total 150 differ/missing []` |
| navtest MC config sha | §4.2 / §4.3 | `92d42853253622b29eabf8e488a97741e350eb3d5a360cb45f69cb0a4b444f84` |
| navtest MC | §4.3 | pkl 12,146 / log 136 / CSV 12,147줄 / token sha `19cf783c…4419` / 3.1G |
| E2E token list | §5.2 | 85,109 / 978 log / sha `c3fa309c…9c79` / 도시별 62,875·9,180·7,840·5,214 |
| stage-E MC | §5.4 | status `done N/N failed-lines 0`, xz ok 85,109 |
| objects | §5.5 | `n_error 0`, 디렉터리 합 85,109 |
| human | §5.6 | n 85,109, `frame_gap 735`, `n_reg_ge16 83005` |
| SDF | §5.7 | 85,109 npz, error 0 |
| e2e_side | §5.8 | `n_with_side 85109` |
| GTLoader 전수 | §5.10 | `gt_ok 85109`, `p_pdm_zero 4370`, `obj_n_zero 68`, `sdf_absmax_max 10.0` |
| teacher cache | §5.9 | sha head `cddf943ffec8d6a8` / `0eaeda793402a804`, 103,288 / 12,146 / 126,032 / 12,146 |
| teacher 교정기 | §6.2 | `sha256.json` 모두 True |
| E2 학습 시작 | §7.2 | `Num training samples: 85109`, global_batch 128, 첫 레코드 `ref/human_only 1.0` |
| E2 학습 시간 | §7.6 | epoch 약 40분, 30 epoch 약 20 h (4× RTX 5090) |
| E0 평가 | §8.1 | 0.848683140828655 (4자리 0.8487) |
| E2 평가 | §8.2 | final 0.86151 / tau0 0.84656 (같은 ckpt일 때. 다시 학습하면 seed 수준 차이) |
| E2 − E0 | §8.3 | +1.28 [+0.74, +1.86] |
| stage T(경로 B) | §6.3 | pack ready 23,820 / 7,930, OOF T 0.896 / M 0.895, liveness ≥ 0.01 |

---

## 10. 알려진 함정과 해결

| # | 함정 | 증상 | 해결 |
|---|---|---|---|
| 1 | `requirements_navsim.txt`의 torch 2.0.1 pin | torch가 다운그레이드되고 RTX 5090에서 동작하지 않음 | torch 두 줄을 빼고 설치하고, editable 설치는 `--no-deps`(§2.3) |
| 2 | Ray GCS가 뜨지 않음 | metric caching이 "Starting ray local!"에서 멈춤 | `worker=single_machine_thread_pool worker.use_process_pool=true`(§4.2) |
| 3 | thread pool이 GIL 때문에 느림 | 6 thread로 6.8 h에 5.2k개 | `use_process_pool=true` |
| 4 | SAVED_CFG sha | 경로나 worker 수가 다르면 `build_metric_cache.py build`가 "saved caching config changed"로 멈춤 | 같은 절대경로 + 같은 override를 쓴다. 안 되면 이관한 config.yaml을 덮어쓴다(§4.5) |
| 5 | Hydra가 `=`를 파싱함 | `epoch=29-step=19950.ckpt`를 넘기면 override 파싱 오류 | `last.ckpt`를 쓰거나 `=`가 없는 이름으로 symlink(`ln -s ".../epoch=28-step=19285.ckpt" /home/external-user/ssd/yongjae_refiner/_scratch/E2_ep28.ckpt`) |
| 6 | `eval_para_ssr.sh`, `cache_metric_navtest.sh`가 bare `python`을 씀 | 다른 python이 잡힘 | `PATH=/home/external-user/miniconda3/envs/ssr/bin:$PATH`(stageE_gpu_commands.sh는 자동으로 함) |
| 7 | navtest 경로 | `data/dataset`에는 trainval만 있음 | `NAVSIM_DOWNLOAD=/home/external-user/navsim/download`(stageE eval은 자동), metric cache는 `navsim_log_path=` |
| 8 | cache 부분 채점 | cache가 덜 만들어지면 조용히 일부만 채점 | 로그의 `Starting pdm scoring of 12146` 확인 |
| 9 | parity golden이 CPU에 민감함 | 다른 CPU에서 MISMATCH | `parity_golden.pt`는 `git worktree add <tmp>/ssr_e0 262ffa6` → `stageE_parity.py`를 복사 → `ln -s /home/external-user/yongjae/SSR/data <tmp>/ssr_e0/data` → `OMP_NUM_THREADS=8 … stageE_parity.py dump --out …`로 다시 만든다([재구성], 시험 안 함, write-once). e12는 다시 만들 수 없다 |
| 10 | `check`가 manifest를 덮어씀 | stage-T `manifest.parquet`, CSV 손상 | e2e part에는 `--no-final-check`, `check` 금지(§5.4) |
| 11 | 병렬 build가 `_state/build_summary.json`을 덮어씀 | stage-T summary 손실 | 백업했다가 복원(§5.3, §5.4) |
| 12 | worker 상한 | `at most N workers`로 즉시 종료 | MC/SDF/objects 4, human 2. 더 쓰려면 part를 나눈다 |
| 13 | E1 기본값 함정 | env를 빠뜨리면 E1이 다른 실험이 됨(`bg0.1`, `tau0`) | §7.3의 env 3개 필수 |
| 14 | train4의 GPU 수 하드코딩 | GPU 2장이면 global batch 64 | §7.5 표대로 145, 147행을 고친다 |
| 15 | `rescore_attr.py` 누락(untracked) | e2e_side, 채점의 import 오류 | 이관 #4 |
| 16 | `TeacherCache`가 `_future` 경로를 거부함 | 예외 | symlink 대상 이름에 `_future`를 넣지 않는다 |
| 17 | `stageT_gpu_commands.sh:8`의 `pack_when_ready.sh` | 파일 없음 | `pack_follow.py` 또는 `python -m navsim.agents.para_ssr.refiner.data` |
| 18 | snapshot write-once | 경로 A 사본이 있으면 snapshot이 거부함 | 경로 B 전에 기존 teachers/를 다른 이름으로 옮긴다 |
| 19 | wandb 미로그인 | 기록이 남지 않음(학습은 계속, non_fatal) | `wandb login`. offline은 `WANDB=1 WANDB_MODE=offline` env(overrides에 wandb 6줄이 추가됨) 또는 train4 줄에 `wandb.mode=offline`을 직접 넣는다(§2.4). `WANDB_MODE`만으로는 안 된다 |
| 20 | 공유 머신 | CPU 경합으로 시간이 늘어남 | `nice -n 10`, `OMP_NUM_THREADS=1`. 이 문서의 시간은 경합이 섞인 값이다 |
| 21 | NCCL bootstrap | 기본 NIC에서 실패한 적이 있음 | wrapper가 `NCCL_SOCKET_IFNAME=lo`를 설정한다(`train_para_ssr.sh:37-39`). 다른 NCCL 변수는 쓰지 않는다 |
| 22 | EPDMS(navsim v2) 참고 지표 | `tools/refiner/epdms_navtest.py`가 `/home/external-user/yongjae/navsim_v2`(branch `para-ssr-epdms`, HEAD `0a380a9`, PARA-SSR 추가분은 전부 untracked)와 `/home/external-user/yongjae/navsim_v2_exp`를 요구함 | 결정에 쓰지 않는 지표다. 필요하면 worktree를 통째로 복사하고, v2 navtest cache를 다시 만든다(옛 서버: 24 process, 42분) |

---

## 11. 참고 문서

- `report/37_teacher_refiner_experiment_overview.md`: 전체 실험 개요, E2 run 기록(§8-9), E1 계획(§11-3)
- `report/38_stageE_E2_results_analysis.md`: E2 결과 분석, 공식 수치
- `report/refiner_T/STATUS.md`: stage T 진행 상황, 명령 원문(§8.6–8.8)
- `report/refiner_T/IMPL_SPEC.md`: stage T 구현 명세, CPU 관례(:21)
- `report/refiner_T/PRESTATED_DECISION_RULE.txt`: 사전 결정 규칙, AMENDMENT 6–7, STAGE E PLAN, REVISION 1–3
- `report/refiner_T/stageE_impl_plan.md`: stage E 구현 계획
- `report/refiner_T/stageE_eval/stageE_E2_eval.md`: E2 평가 기록
- `report/refiner_E_code_audit/source_sha256.json`: E2 코드 해시
- `docs/README.md`(untracked, 이관 #13): PARA-SSR NAVSIM 실행 가이드
- 원래 실행 기록(옛 서버에만 있음): `~/.claude/projects/-home-external-user/9262a8d2-…/subagents/workflows/wf_3fcdccb1-f8e/`(stage-E GT build/verify), `wf_8022d153-ccb/`(side), `wf_09925bc2-b69/`(human)

---

## 12. 해결되지 않은 것 (gap)

1. **설치 레시피(§2.3)는 pip freeze에서 거꾸로 만든 것이다.** 처음 설치할 때의 명령 기록은 없다. torch를 cu128로 올린 index URL도 추정이다. driver 하한 R570은 일반 지식이다.
2. **새 서버에서 처음부터 만드는 시간은 모두 추정이다.** navtest MC 약 60분, stage-E MC 85,109개 약 2.7–4 h, SDF 약 50분, side 약 35분. 이 서버에서는 navtest MC를 세 번에 나눠 만들었고 stage-E MC는 58,743개만 만들었다. 최대 메모리 사용량은 기록이 없다.
3. **원래 셸 명령줄이 없는 것:** navtest MC 최종 실행, E2 `train4` 실행(tmux), stage-T objects loop, `refine_external_drafts.py predict`. overrides와 메타데이터로 재구성했고 결과 config가 같다는 것은 확인했다.
4. **새로 만든 pkl이 바이트 단위로 같은지는 검증하지 않았다**(pkl 안에 절대경로가 있음). 필드와 점수가 같다는 것만 확인됐다(`report/refiner_T/metric_cache_verify_navtest.json`: 20 token, diff 0.0).
5. **pytest 기대 개수를 모른다.** 마지막 기록은 330 passed다(stage-E 테스트 추가 전후인지 불명). stage T 데이터가 없는 서버에서 어떤 테스트가 skip이 아니라 fail로 나오는지도 확인하지 않았다.
6. **parity golden:** 복사본이 다른 CPU에서 IDENTICAL이 나오는지 모른다. `parity_golden_e12.pt`는 git에 없는 중간 코드로 만든 것이라 다시 만들 수 없다. `parity_golden.pt` 재생성 절차는 시험하지 않았다.
7. **GPU나 torch가 다를 때** E0 0.848683140828655가 비트 단위로 같을지는 모른다. 4자리 일치를 기대한다.
8. **경로 B로 다시 학습한 teacher는 비트 단위로 재현되지 않는다.** λ_c와 E2 수치가 seed 수준에서 달라진다.
9. **`$DATA/stageE_diag/tokens.parquet`를 어떻게 만들었는지 기록이 없다.**
10. **teacher cache의 shard/npz 해시는 계산하지 않았다**(수백 GB). 개수, manifest, sha head만 확인했다. 옛 서버의 `bevfusion.tar.zst`(155.6 GB) 내용도 확인하지 않았다.
11. **`refine_external_drafts.py`(E0 + teacher 사후 실험)가 경로 A(스냅숏만 있음)에서 동작하는지** 확인하지 않았다(`--run`에 snapshot 디렉터리를 줄 수 있는지 불명, `packed/navtest` 필요).
12. **scratchpad 스크립트**(`derive_e2e_tokens.py`, `e2e_manifest.py`, `verify/v_*.py`)는 옛 서버 `/tmp`에만 있다. 이 문서의 §5.2, §5.10, 부록 A에 내용을 옮겨 두었지만 repo에는 없다. repo에 넣을지는 사용자가 정한다.
13. **E2 학습에 쓴 물리 GPU 번호**는 run 디렉터리에 남아 있지 않다. 0,1,4,5는 REVISION 1 기록에서 가져온 값이다.
14. 하위 문서 일부에는 "아직 push 안 됨", "`rescore_attr.py`는 git에 있음"이라고 적혀 있다. 2026-10-03에 다시 확인한 결과 **push 돼 있고, `rescore_attr.py`는 untracked**다. 이 문서는 확인한 값을 따랐다.
15. **`/dev/shm` 최소 크기와 최대 RAM 사용량**은 측정하지 않았다(§2.1). 이 서버는 shm 126 GB, RAM 251 GB였다.

---

## 부록 A. scratchpad 스크립트 원문 (옛 서버 `/tmp`에만 있던 것)

### A.1 `e2e_manifest.py`: `$D/splits/e2e_train_trainlogs_status.json` 작성

stage-T 산출물(`metric_cache/manifest.parquet`, `scores/{train,dev}.parquet`, `objects/*/index.parquet`)이 있어야 돈다. 실행: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 $PY e2e_manifest.py 12 [build_info.json]`. 주의: `p_pdm.done`은 stage-T 점수표 기준이라 의미가 없다. 실제 p_pdm은 e2e_side가 85,109개 전부에 대해 공급한다.

```python
"""Per-part completeness manifest for the E2E train_logs token list -> splits/e2e_train_trainlogs_status.json."""
import json, lzma, os, sys, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
D = Path("/home/external-user/ssd/yongjae_refiner")
E_MC = Path("/home/external-user/yongjae/SSR/report/cause_and_correction_tests/E_train_split_feasibility/metric_cache")
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 8

def mcp(root, log, tok):
    return Path(root) / log / "unknown" / tok / "metric_cache.pkl"

def xz_ok(p):
    try:
        with lzma.open(p, "rb") as f:
            while f.read(1 << 22):
                pass
        return True
    except Exception:
        return False

def chk(a):
    tok, log = a
    p = mcp(D / "metric_cache", log, tok)
    return (p.exists(), p.exists() and xz_ok(p))

if __name__ == "__main__":
    t0 = time.time()
    e = pd.read_parquet(D / "splits/e2e_train_trainlogs.parquet")
    toks = set(e.token)
    man = pd.read_parquet(D / "metric_cache/manifest.parquet")          # stage-T xz check (read only)
    ok_T = dict(zip(man.token, man.ok))
    new = [(t, l) for t, l in zip(e.token, e.log) if t not in ok_T]
    with Pool(WORKERS) as p:
        res = p.map(chk, new, chunksize=64)
    mc_ok = {t: bool(ok_T[t]) for t in e.token if t in ok_T}
    mc_ok.update({t: r[1] for (t, _), r in zip(new, res)})
    mc_exist_new = sum(r[0] for r in res)
    fail = {}
    ff = D / "metric_cache/_state/failed_tokens.tsv"
    if ff.exists():
        for line in ff.read_text().splitlines():
            parts = line.split("\t")
            if parts and parts[0] in toks and not mc_ok.get(parts[0]):
                fail[parts[0]] = parts[2][:200] if len(parts) > 2 else ""
    e["mc_ok"] = e.token.map(mc_ok).fillna(False).astype(bool)
    obj_dirs = [D / "objects/train", D / "objects/dev", D / "objects/navtrain"]
    e["obj_dir"] = [next((d.name for d in obj_dirs if (d / f"{t}.npz").exists()), "") for t in e.token]
    idx = pd.concat([pd.read_parquet(d / "index.parquet", columns=["token", "status"]) for d in obj_dirs
                     if (d / "index.parquet").exists()]).drop_duplicates("token", keep="last")
    ost = dict(zip(idx.token, idx.status))
    obj_fail = Counter(str(ost.get(t, "not_in_index"))[:80] for t in e.token[e.obj_dir == ""])
    e["sdf_ok"] = [(D / "sdf/navtrain" / f"{t}.npz").exists() for t in e.token]
    sdf_err = {}
    sf = D / "sdf/navtrain/_build/stats.jsonl"
    for line in sf.read_text().splitlines():
        r = json.loads(line)
        if r["token"] in toks and r["status"].startswith("error"):
            sdf_err[r["token"]] = r["status"][:200]
    sdf_missing = e.token[~e.sdf_ok]
    sdf_reason = Counter(("error: " + sdf_err[t].split(":")[1].strip()) if t in sdf_err else
                         ("no_mc" if not mc_ok.get(t) else "not_built") for t in sdf_missing)
    pp = set()
    for s in ("train", "dev"):
        sc = pd.read_parquet(D / f"scores/{s}.parquet", columns=["token", "pdm_progress_eff"])
        pp |= set(sc.token[np.isfinite(sc.pdm_progress_eff)])
    e["p_pdm_scores"] = e.token.isin(pp)
    all_ok = e.mc_ok & (e.obj_dir != "") & e.sdf_ok
    n = len(e)
    def frac(m): return dict(done=int(m.sum()), missing=int(n - m.sum()), frac=round(float(m.mean()), 6))
    out = dict(
        created=time.strftime("%F %T"), token_list=str(D / "splits/e2e_train_trainlogs.parquet"),
        definition="interaction_final build_datasets: navtrain scene_filter (config.yaml, 1192 logs, 103288 tokens) "
                   "log_names & cfg.train_logs (== default_train_val_test_log_split.yaml train_logs) -> filter_scenes on "
                   "navsim_logs/trainval; == 'Num training samples: 85109' in run_training.log",
        n_tokens=n, n_logs=int(e.log.nunique()), per_city=e.city.value_counts().astype(int).to_dict(),
        overlap_stageT=dict(train_trainlogs=int(pd.read_parquet(D / "splits/train_trainlogs.parquet").token.isin(toks).sum()),
                            dev_trainlogs=int(pd.read_parquet(D / "splits/dev_trainlogs.parquet").token.isin(toks).sum())),
        parts=dict(
            metric_cache=dict(root=str(D / "metric_cache"), layout="<log>/unknown/<token>/metric_cache.pkl",
                              check="xz stream complete (stage-T tokens: metric_cache/manifest.parquet ok; new: re-checked)",
                              new_tokens_checked=len(new), new_tokens_exist=int(mc_exist_new), **frac(e.mc_ok),
                              failed_by_reason=dict(Counter(v.split("\n")[-1][:120] or "not cached" for v in fail.values())),
                              failed_tokens=sorted(set(e.token[~e.mc_ok]))[:500]),
            objects=dict(dirs=[str(d) for d in obj_dirs], version="gt_future_v1",
                         per_dir=e.obj_dir.replace("", "missing").value_counts().astype(int).to_dict(),
                         **frac(e.obj_dir != ""), missing_by_index_status=dict(obj_fail),
                         missing_tokens=sorted(e.token[e.obj_dir == ""])[:500]),
            sdf=dict(dir=str(D / "sdf/navtrain"), version="e_grid_dac_exact_v1", **frac(e.sdf_ok),
                     missing_by_reason=dict(sdf_reason), missing_tokens=sorted(sdf_missing)[:500]),
            centerline=dict(source="surrogate.centerline_from_metric_cache(metric cache), derived at pack time",
                            **frac(e.mc_ok)),
            ego_state=dict(source="metric cache ego_state (v0, a0) + objects npz obj_ego_kf (GT ego)",
                           **frac(e.mc_ok & (e.obj_dir != ""))),
            p_pdm=dict(source="pdm_progress_eff (PDM-Closed raw progress * mult) in scores/{train,dev}.parquet; "
                              "NOT built here -- new tokens need a PDM-Closed scoring pass on their metric cache",
                       **frac(e.p_pdm_scores)),
        ),
        complete_mc_objects_sdf=frac(all_ok),
        manifest_seconds=round(time.time() - t0, 1),
    )
    if len(sys.argv) > 2:
        out["build"] = json.loads(Path(sys.argv[2]).read_text())
    fn = D / "splits/e2e_train_trainlogs_status.json"
    fn.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: (v if k != "parts" else {p: {kk: vv for kk, vv in d.items() if "tokens" not in kk}
                                                   for p, d in v.items()}) for k, v in out.items()}, indent=1, default=str))
```

### A.2 GTLoader 전수 검사

§5.10의 블록이 원래 `verify/v_side_gt.py`의 (b) 부분을 그대로 줄인 것이다. (a) 부분(무작위 100 token의 side를 다시 계산해 비트 단위로 비교)에는 `sample100.parquet`가 필요해서 뺐다.

---

## 부록 B. 이 서버의 `pip freeze` PyPI pin 전체 (nvidia-* 제외; 재현이 어긋날 때 constraints로 사용)

```
absl-py==2.3.1 affine==3.0.1 aioboto3==15.5.0 aiobotocore==2.25.1 aiofiles==25.1.0 aiohappyeyeballs==2.6.1 aiohttp==3.13.5 aioitertools==0.13.0 aiosignal==1.4.0 annotated-doc==0.0.5 annotated-types==0.7.0 antlr4-python3-runtime==4.9.3 async-timeout==5.0.1 bokeh==2.4.3 boto3==1.40.61 botocore==1.40.61 casadi==3.8.0 ccimport==0.4.4 click==8.1.8 click-plugins==1.1.1.2 cligj==0.7.2 contourpy==1.3.0 control==0.9.1 cumm-cu126==0.7.11 cycler==0.12.1 eval_type_backport==0.4.0 filelock==3.19.1 fiona==1.10.1 fire==0.7.1 fonttools==4.60.2 frozenlist==1.8.0 fsspec==2025.10.0 geopandas==1.0.1 gitdb==4.0.12 GitPython==3.1.62 greenlet==3.2.5 grpcio==1.80.0 guppy3==3.1.2 hf-xet==1.6.0 huggingface_hub==1.8.0 hydra-core==1.2.0 imageio-ffmpeg==0.6.0 importlib_resources==6.5.2 iniconfig==2.1.0 jmespath==1.1.0 joblib==1.5.3 kiwisolver==1.4.7 lightning-utilities==0.15.2 Markdown==3.9 markdown-it-py==3.0.0 matplotlib==3.9.4 mdurl==0.1.2 mpmath==1.3.0 msgpack==1.1.2 multidict==6.7.1 networkx==3.2.1 ninja==1.13.2 numpy==1.23.4 omegaconf==2.3.1 opencv-python==4.9.0.80 outcome==1.3.0.post0 pandas==2.3.3 pccm==0.4.16 pillow==11.3.0 pluggy==1.6.0 portalocker==3.2.0 positional-encodings==6.0.1 propcache==0.4.1 protobuf==4.25.3 py==1.11.0 pyarrow==21.0.0 pybind11==3.1.0 pydantic==2.13.5 pydantic_core==2.46.5 pyinstrument==5.1.3 pyogrio==0.11.1 pyparsing==3.3.2 pyproj==3.6.1 pyquaternion==0.9.9 pytest==8.4.2 pytorch-lightning==2.2.1 rasterio==1.3.11 ray==2.51.2 retry==0.9.2 rich==15.0.0 rtree==1.4.1 s3transfer==0.14.0 safetensors==0.7.0 scikit-learn==1.2.2 scipy==1.13.1 selenium==4.32.0 sentry-sdk==2.68.1 shapely==2.0.7 shellingham==1.5.4 smmap==5.0.3 snuggs==1.4.7 sortedcontainers==2.4.0 spconv-cu126==2.3.8 SQLAlchemy==1.4.27 sympy==1.14.0 tensorboard==2.16.2 tensorboard-data-server==0.7.2 termcolor==3.1.0 threadpoolctl==3.6.0 timm==1.0.28 torch==2.8.0+cu128 torchmetrics==1.8.2 torchvision==0.23.0+cu128 tqdm==4.70.0 trio==0.31.0 trio-websocket==0.12.2 triton==3.4.0 typer==0.23.2 typing-inspection==0.4.2 tzdata==2026.3 ujson==5.11.0 urllib3==1.26.20 wandb==0.26.1 Werkzeug==3.1.8 wrapt==1.17.3 wsproto==1.2.0 yarl==1.22.0 zstandard==0.23.0
+ nuplan-devkit @ git+https://github.com/motional/nuplan-devkit.git@ce3c323af01c0d7ec5672f7832ef53f9c679aab0
```
