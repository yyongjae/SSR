# trainval + SimScale 학습 가이드

메인 레시피와 다른 것은 **학습 데이터와 백본**입니다. SimScale에는 LiDAR와 위성 영상이 없어서 teacher도 카메라 전용으로 바뀝니다. 나머지(30 epoch, loss, 스케줄, global batch 128, refiner)는 메인과 같습니다.

| | 메인 | 이 레시피 (`scripts/train_simscale.sh`) |
|---|---|---|
| 학습 데이터 | navtrain_train 85,109 | navtrain_train 85,109 + navtrain_val 18,179 + SimScale round 0-4 (237,372 중 필터 통과분) |
| 백본 | ResNet-34 | DINOv2 ViT-S + LoRA r32, 1148×672 (`dino_vits14_1148`) |
| teacher | BEVFusion (LiDAR) / ReSMap | BEVFormer / Mask2Map (카메라만) |
| TF32 | off | on |
| run 이름 (8장) | `main_<hash6>` | `dino_vits14_1148_bevformer-mask2map_tvsim_tf32_<hash6>` |

## 0. 준비물

- **GPU**: H100 8장 기준입니다. `NGPU=4`, `NGPU=2`도 됩니다. GPU 수가 달라도 global batch 128은 같습니다.
- **CPU**: 64코어 이상을 권장합니다(자산 생성).
- **디스크**: 약 7.5 TB가 필요합니다.

| 항목 | 크기 |
|---|---:|
| NAVSIM (로그, navtrain 센서, test, 지도) | 0.7 TB |
| SimScale meta + sensor_blobs_hist | 2.23 TB |
| teacher 캐시 (BEVFormer 1.75 + Mask2Map 1.85) | 3.6 TB |
| metric cache, GT, 라벨 | 0.2 TB |
| 학습 run 하나 | 0.2 TB |

- **받을 것**
  - 이 저장소의 브랜치 `v2_final_simscale`
  - HF `yyongjae/ssr-cam-teachers`: teacher 가중치, config, 캐시 manifest, refiner teacher 2개
  - SimScale과 teacher repo 모두 공개라 HF 토큰이 필요 없습니다. SimScale 다운로드는 dataset revision `06a4fc88`(2026-01-25)로 고정되어 있습니다.

## 1. 코드, 경로, 환경

```bash
git clone https://github.com/yyongjae/SSR.git SSR && cd SSR && git checkout v2_final_simscale
cp tools/release/paths.env.example tools/release/paths.env   # 아래 경로를 채웁니다 (git에 올라가지 않습니다)
set -a; . tools/release/paths.env; set +a                    # 채운 뒤에: 아래 명령들이 $SSR_TEACHER_CACHE 등을 씁니다
S=tools/release/setup
bash $S/00_dirs_and_env.sh
CONDA=/path/to/conda bash $S/install_env.sh
```

`paths.env`에서 채울 것은 다음과 같습니다. 뜻은 `paths.env.example`에 있습니다. 폴더는 미리 없어도 됩니다. 저장 위치와 `SSR_PLAN_DATA`는 `00_dirs_and_env.sh`가, `SSR_PY`의 env는 `install_env.sh`가, SimScale·test 링크 폴더는 2절 명령이 만듭니다. 적지 않은 변수는 기준 서버의 경로로 채워지므로 아래 것은 모두 적습니다.

- 저장 위치: `SSR_DATA_ROOT`, `SSR_TEACHER_CACHE`, `SSR_RUN_ROOT`
- 데이터: `OPENSCENE_DATA_ROOT`, `NUPLAN_MAPS_ROOT`, `SSR_TEST_DATA_ROOT`, `SSR_SIMSCALE_ROOT`
- `SSR_PLAN_DATA`: 빈 폴더면 됩니다. WoTE의 `pdm_score_256`을 두면 real 행 점수를 그것과 대조하는 데만 씁니다.
- 실행 파일: `SSR_PY`, `SSR_TORCHRUN`
- 평가에 쓰는 것: `NAVSIM_V11*`, `NAVSIM_V22*`

## 2. 데이터

**`download_navsim_simscale.sh --dest <D> all`로 이미 받았다면** 다운로드는 건너뛰고 경로만 맞춥니다.

```bash
# paths.env
OPENSCENE_DATA_ROOT=<D>/navsim
NUPLAN_MAPS_ROOT=<D>/navsim/maps
SSR_SIMSCALE_ROOT=<D>/simscale
SSR_TEST_DATA_ROOT=<D>/navsim_test
# 평가 코드는 test/test 구조를 읽습니다
mkdir -p <D>/navsim_test/navsim_logs/test <D>/navsim_test/sensor_blobs/test
ln -sfn <D>/navsim/navsim_logs/test <D>/navsim_test/navsim_logs/test/test
ln -sfn <D>/navsim/sensor_blobs/test <D>/navsim_test/sensor_blobs/test/test
```

**SimScale까지 NAVSIM 배치(`navsim_logs/<split>`, `sensor_blobs/<split>`)로 받아 두었다면** (DrivoR, TOAD 방식) 복사하지 않고 링크만 만듭니다. 우리 코드는 라운드마다 `meta_datas`와 `sensor_blobs_hist`가 한 폴더에 있어야 읽습니다.

```bash
B=/abs/benchmark      # navsim_logs, sensor_blobs, maps가 있는 폴더
SIM=/abs/simscale     # 새로 만들 링크 폴더
T=/abs/navsim_test    # 새로 만들 링크 폴더
for r in 0 1 2 3 4; do
  d=$SIM/synthetic_reaction_pdm_v1.0-$r; mkdir -p $d
  ln -sfn "$(readlink -f $B/navsim_logs/synthetic_reaction_pdm_v1.0-$r)"  $d/meta_datas
  ln -sfn "$(readlink -f $B/sensor_blobs/synthetic_reaction_pdm_v1.0-$r)" $d/sensor_blobs_hist
done
mkdir -p $T/navsim_logs/test $T/sensor_blobs/test
ln -sfn $B/navsim_logs/test  $T/navsim_logs/test/test
ln -sfn $B/sensor_blobs/test $T/sensor_blobs/test/test
```

- `paths.env`에는 `OPENSCENE_DATA_ROOT=$B`, `NUPLAN_MAPS_ROOT=$B/maps`, `SSR_SIMSCALE_ROOT=$SIM`, `SSR_TEST_DATA_ROOT=$T`를 절대경로로 씁니다. 바꾼 뒤 1절의 `set -a; . tools/release/paths.env; set +a`를 다시 실행합니다.
- 라운드 폴더 안에 장면 pkl과 장면 폴더가 바로 있어야 합니다. 링크가 틀렸거나 덜 받은 라운드가 있으면 4절의 첫 단계(토큰 표)가 장면 수를 공개값과 비교해서 멈춥니다.
- 4절 명령에 `DOWNLOAD=1`을 붙이지 않습니다.

**아직 받지 않았다면** NAVSIM은 `bash $S/10_download_dataset.sh`로 받습니다. SimScale은 4절의 명령에 `DOWNLOAD=1`을 붙이면 받습니다(2.23 TB, 100 MB/s에서 약 6시간).

그다음 navtrain_train / navtrain_val 토큰 표를 만듭니다.

```bash
bash $S/15_token_lists.sh
```

## 3. 카메라 teacher (GPU 노드에서 한 번)

```bash
bash tools/release/teachers/get_teachers.sh $SSR_TEACHER_CACHE/_src          # BEVFormer, Mask2Map 코드 (pin + patch)
CONDA=$(command -v conda) TEACHER_SRC=$SSR_TEACHER_CACHE/_src \
  bash tools/release/teachers/setup_teacher_env.sh $SSR_TEACHER_CACHE/_env all   # mmcv 빌드, GPU arch 자동(H100 9.0)
hf download yyongjae/ssr-cam-teachers --local-dir $SSR_TEACHER_CACHE/cam_teachers
(cd $SSR_TEACHER_CACHE/cam_teachers && sha256sum -c SHA256SUMS)
mkdir -p $SSR_TEACHER_CACHE/refiner
cp -r $SSR_TEACHER_CACHE/cam_teachers/refiner/{det_bevformer,map_mask2map} $SSR_TEACHER_CACHE/refiner/
```

## 4. 학습 자산 (명령 하나)

```bash
K=16 LABEL_WORKERS=48 SIDE_WORKERS=24 SIM_WORKERS=16 TEACHER_GPUS=0,1,2,3,4,5,6,7 \
  bash $S/prepare_trainval_simscale.sh          # SimScale을 아직 안 받았으면 DOWNLOAD=1을 앞에 붙입니다
```

- 순서는 다음과 같습니다.
  1. 토큰 표 (`16_trainval_simscale_table.sh`)
  2. metric cache (`20_*`)
  3. GT (`30_*`)
  4. packed (`40_*`)
  5. warm-up 라벨과 점수 파일 (`50_*`)
  6. teacher 캐시 (`65_cam_teacher_bev.sh`): BEVFormer(det)와 Mask2Map(map)이 표의 모든 토큰(real 103,288 + SimScale)의 BEV를 `TEACHER_GPUS`로 캐시합니다(3.6 TB). 3절의 코드, 환경, 가중치를 씁니다.
  7. 전체 검사 (`verify_assets.py --data trainval_simscale`)
- 끊기면 같은 명령을 다시 실행합니다. 끝난 것은 건너뜁니다.
- 마지막 검사에 FAIL이 없어야 합니다. FAIL 줄마다 고칠 명령이 나옵니다.
- `SETUP_DRY=1`을 붙이면 실행할 명령만 출력합니다.
- 시간 [추정]
  - CPU 단계: 7-11시간 (64코어)
  - teacher 캐시: RTX 5090 기준 35 GPU-시간. H100은 측정하지 않았고, 8장이면 3-5시간 정도로 봅니다.

## 5. 학습 (명령 하나)

```bash
bash scripts/train_simscale.sh NGPU=8
```

- 조립 검사(check, CPU)를 먼저 돌리고, 통과하면 학습을 시작합니다. 통과하지 못하면 시작하지 않습니다.
- GPU는 `NGPU=8|4|2`로 정하거나 `GPUS=0,1,2,3`처럼 직접 고릅니다.
- 처음이면 smoke를 먼저 돌립니다: `bash scripts/train_simscale.sh smoke GPUS=0,1`. 2 epoch × 60 micro-batch에 검사 28개입니다.
- 실행 전 확인:
  - 미리 보기만 하려면 `DRY=1`을 붙입니다.
  - 이름만 보려면 `bash scripts/train_simscale.sh name NGPU=8`을 씁니다.
- 실행 중 관리: `status`, `stop`, `resume RUN=<run 이름>`.
- 로그는 `$SSR_RUN_ROOT/<run>/launch/train.log`, 라벨러 로그는 `.../labeler/labeler.log`입니다.
- 작업 트리가 깨끗해야 시작합니다(`git status`가 비어 있어야 합니다). `paths.env`는 git이 무시합니다.
- 학습 시간은 측정하지 않았습니다. epoch당 토큰은 메인의 약 4배입니다.

## 6. 평가

메인과 같습니다. 원본 NAVSIM과 navtest cache를 `$S/70_pristine_navsim.sh`로 한 번 만든 뒤 다음을 실행합니다.

```bash
bash scripts/experiment.sh eval RUN=<run 이름> RUN_ROOT=$SSR_RUN_ROOT GPUS=0,1,2,3
```

## 7. 데이터 처리에서 알아 둘 것

- **장면 구성**: SimScale 장면은 history 4 frame(이미지 있음)과 future 8 frame(4초, 이미지 없음)으로 되어 있습니다. 토큰은 `<16hex>-<라운드>` 전체를 키로 씁니다. 16hex 부분은 같은 시각의 real 토큰과 같으므로 자르면 안 됩니다.
- **sim TTC**: 미래가 4초뿐이라 4초 이후 1초는 마지막 속도로 연장합니다. 그래서 real과 같은 5초 규칙으로 채점합니다. real 캐시 1,300개로 흉내 내 보면 진짜 5초와 TTC 라벨이 약 0.6% 다르고, 한쪽으로 치우치지 않습니다. 4초에서 자르면 6.4%가 통과 쪽으로만 바뀝니다.
- **빠지는 sim 장면**
  - 이미지가 빠진 장면
  - route가 없는 장면
  - 현재 프레임에서 ego와 겹친 객체가 있는 장면 (round 4 기준 0.15%)
  - test 로그에서 나온 장면 (0건)
- **보행자·자전거**: sim에는 없습니다(round 4 전체 32,202장면에서 0개). 렌더 영상에서도 지워져 있습니다. 학습 토큰의 약 70%가 sim입니다. 그래서 평가할 때 navtest의 보행자 관련 충돌을 따로 보는 것을 권합니다.
- **navtrain_val 출신 장면**: sim의 17%가 navtrain_val 로그에서 나왔습니다. 이 레시피는 val도 학습하므로 navtrain_val로 모델이나 설정을 고르면 안 됩니다.
- **레퍼런스 대조**: SimScale 공식 학습 코드, DrivoR, TOAD와 비교했습니다. 장면 읽기, GT 궤적, ego 상태, 카메라 보정값이 비트 단위로 같습니다.

## 8. 문제가 생기면

- **토큰 표가 `lost to the filter`로 거부됨**: 그 라운드의 sensor_blobs_hist part가 덜 받아진 것입니다. 다운로드를 다시 실행합니다. 끝난 archive는 건너뜁니다.
- **verify_assets FAIL**: 줄마다 나오는 명령으로 그 단계를 `RESUME=1`로 다시 돌립니다.
- **on-policy 라벨이 밀린다는 경고**: CPU가 남으면 `LABELER_MAX_WORKERS=<N>`로 상한(기본 48)을 올립니다.
- **teacher 캐시가 중간에 끊김**: 같은 GPU 수로 `RESUME=1`을 붙여 65를 다시 실행합니다. Mask2Map은 시작할 때의 GPU 수로만 이어 갑니다.
