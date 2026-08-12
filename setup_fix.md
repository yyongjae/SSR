# SSR 설치 시 docs/install.md 와 다르게 한 부분

작업일: 2026-08-12
환경: Ubuntu 22.04 / Intel Xeon Gold 6426Y (Sapphire Rapids) / RTX A6000 x8 / driver 575.57.08
conda env: `ssr` (python 3.8.20)

`docs/install.md` 를 그대로 따라가면 이 서버에서는 몇 군데서 막히거나 잘못된 버전이 깔린다.
아래는 **실제로 성공한 절차**와 원문서와 달라진 이유 정리.

---

## 요약: 원문서와 달라진 점 5가지

| # | 원문서 | 실제 | 이유 |
|---|---|---|---|
| 1 | `pip install mmcv-full==1.4.0` (소스 빌드) | prebuilt wheel (`-f .../cu111/torch1.9.0/index.html`) | 컴파일 20~30분 → 수 초. 결과물 동일 |
| 2 | `conda install -c omgarcia gcc-5` | 생략, 시스템 `gcc-9` 사용 | gcc-5는 A6000(sm_86) 미지원. 기본 gcc-10.5는 CUDA 11.1과 불안정 |
| 3 | (없음) | **버전 고정 단계 추가** | pip가 numpy 1.24 / opencv 5.0 등 최신을 끌어와 mmdet3d의 `numpy<1.20` 제약과 충돌 |
| 4 | mmdet3d 빌드(f) → nuscenes-devkit(g) | **순서 바꿈**: 의존성 고정 후 마지막에 mmdet3d 빌드 | CUDA extension이 최종 numpy 헤더에 맞춰 컴파일되도록 |
| 5 | `pip install timm` | `timm==0.6.12` | 무핀 설치 시 1.x가 깔림. `requirements.txt` 기록값은 0.6.12 |

추가로 이 서버 CPU 때문에 **numba가 import만 해도 segfault** 나는 문제가 있어 환경변수 우회가 필요했다 (아래 별도 항목).

---

## 실제 설치 절차 (재현용)

### 0. setuptools 다운그레이드 (원문서에 없음, 먼저 해야 함)

```shell
conda activate ssr
pip install "setuptools==58.2.0" "wheel==0.37.1"
```

setuptools 75(신규 env 기본값)에서는 mmdet3d의 `python setup.py develop` 이 실패한다.
반드시 **torch 설치 전에** 먼저 내려야 한다.

### 1. PyTorch (원문서 b와 동일)

```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. mmcv-full — prebuilt wheel 사용 (원문서 c 변경)

```shell
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

원문서 본문의 `pip install mmcv-full==1.4.0` 은 sdist를 받아 CUDA op을 전부 로컬 컴파일하기 때문에
20~30분이 걸린다. 주석 처리돼 있던 아래쪽 명령(prebuilt wheel)이 정답.
`mmcv_full-1.4.0-cp38-cp38-manylinux1_x86_64.whl` 이 존재하는 것 확인함.

### 3. mmdet / mmseg / timm (원문서 d, e 변경)

```shell
pip install mmdet==2.14.0 mmsegmentation==0.14.1 timm==0.6.12
```

timm은 원문서처럼 무핀으로 깔면 1.x가 들어온다. `requirements.txt` 기록값인 0.6.12로 고정.
(1.x는 `timm.models.layers` 경로가 deprecated 되어 향후 import 에러 소지 있음)

### 4. nuscenes-devkit (원문서 g를 앞으로 당김)

```shell
pip install nuscenes-devkit==1.1.9
```

### 5. 의존성 버전 고정 — **원문서에 없는 필수 단계**

```shell
pip install "numpy==1.19.5" "numba==0.48.0" "llvmlite==0.31.0" "scipy==1.7.3" "pandas==1.3.5" \
  "matplotlib==3.5.3" "scikit-image==0.19.3" "opencv-python==4.7.0.72" "networkx==2.2" \
  "yapf==0.33.0" "protobuf==3.20.3" "shapely==2.0.1" "plyfile==0.7.4" "pillow==9.5.0" \
  "similaritymeasures==0.7.0" "tensorboard==2.9.0" "lyft-dataset-sdk==0.0.8" "trimesh==2.35.39" \
  "setuptools==58.2.0"
```

앞 단계들을 거치면 pip가 아래처럼 최신 버전을 끌어온다 (전부 이 스택에서 깨짐):

- `numpy 1.24.4` → mmdet3d `requirements/runtime.txt` 는 **`numpy<1.20.0`** 요구
- `opencv-python 5.0.0.93` → numpy 2.x 전제
- `yapf 0.43.0` → mmcv 1.4.0 Config 파싱 깨짐 (0.40+ 알려진 이슈)
- `matplotlib 3.7.5` → numpy>=1.20 요구

버전값은 전부 SSR `requirements.txt` 및 `mmdetection3d/requirements/runtime.txt` 기록값을 따랐다.

### 6. mmdet3d 빌드 — 맨 마지막에, gcc-9로 (원문서 f 변경)

```shell
cd /home/yongjae/e2e/SSR
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout -f v0.17.1

export CUDA_HOME=/usr/local/cuda-11.1
export CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9
export TORCH_CUDA_ARCH_LIST="8.6"     # RTX A6000 = sm_86
export MAX_JOBS=16
python setup.py develop
```

변경 이유:

- **순서**: `mmdet3d/ops/spconv` 가 numpy C 헤더(`numpy/arrayobject.h`)를 참조한다. 5번에서 numpy를
  1.19.5로 고정한 **뒤에** 빌드해야 ABI가 맞는다. 원문서 순서(f→g)대로 하면 numpy 1.24 헤더로 컴파일된다.
- **gcc-5 미설치**: 원문서 step c의 `conda install -c omgarcia gcc-5` 는 생략했다. gcc-5는 sm_86
  (Ampere)을 모른다. 시스템 기본 gcc는 10.5인데 CUDA 11.1과 조합이 불안정해서 gcc-9로 명시 지정.
- **`TORCH_CUDA_ARCH_LIST=8.6`**: 미지정 시 전체 아키텍처를 빌드해 시간이 몇 배로 늘어난다.
- **설치 위치**: VAD 쪽(`/home/yongjae/e2e/VAD/mmdetection3d`)을 재사용하지 않고 SSR 아래에 별도 clone.
  `setup.py develop` 은 소스 디렉토리를 가리키는 egg-link 방식이라 공유하면 두 프로젝트가 묶인다.

빌드 결과 `.so` 11개 생성 확인 (ball_query, furthest_point_sample, gather_points, group_points,
interpolate, iou3d, knn, paconv, roiaware_pool3d, spconv, voxel).

### 7. SSR clone (원문서 h) — 이미 완료된 상태였음

---

## numba segfault 우회 (이 서버 고유 문제)

### 증상

```shell
$ python -c "import numba"
Segmentation fault (core dumped)
```

### 원인

이 서버 CPU는 **Intel Xeon Gold 6426Y (Sapphire Rapids, family 6 model 143)**.
mmdet3d 0.17.1이 요구하는 `numba==0.48.0` / `llvmlite==0.31.0` 은 내장 LLVM이 8버전인데,
LLVM 8은 Sapphire Rapids를 모르기 때문에 **host CPU 자동 탐지 중 죽는다**. 패키지 설치가
잘못된 게 아니라서 재설치로는 해결 안 됨.

### 조치

`$CONDA_PREFIX/etc/conda/activate.d/ssr_env.sh` 생성 → `conda activate ssr` 시 자동 적용:

```bash
#!/bin/bash
export NUMBA_CPU_NAME=generic
export NUMBA_CPU_FEATURES=

export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TORCH_CUDA_ARCH_LIST="8.6"
```

대응하는 `deactivate.d/ssr_env.sh` 에서 unset.

### 남는 제약 (실사용엔 무해)

위 우회를 적용해도 **numba를 torch보다 먼저 import하면 여전히 segfault** 난다.
llvmlite와 torch의 LLVM 심볼 충돌이고, 크래시 지점은 `torch/_ops.py`.

```shell
python -c "import numba, mmdet3d"   # segfault
python -c "import mmdet3d, numba"   # 정상
```

`tools/train.py`, `tools/test.py` 를 포함한 실제 코드 경로는 전부 torch/mmcv를 먼저 import하므로
학습·평가에는 영향이 없다. `python -c` 로 확인할 때만 순서를 지키면 된다.

> 참고: 기존 `vad` env도 같은 원인으로 `import numba` 가 죽는 상태다 (numpy 1.21.1 + numba 0.48,
> 우회 없음). VAD 실행 경로가 numba를 안 타서 표면화되지 않았던 것으로 보임.

---

## 설치 검증 결과

```
torch      1.9.1+cu111  cuda=11.1  avail=True  gpus=8
numpy      1.19.5
numba      0.48.0
mmcv-full  1.4.0
mmdet      2.14.0
mmseg      0.14.1
mmdet3d    0.17.1  (/home/yongjae/e2e/SSR/mmdetection3d)
timm       0.6.12
nuscenes-devkit 1.1.9
```

- `pip check` → No broken requirements found
- mmcv CUDA op (nms), mmdet3d CUDA op, MultiScaleDeformableAttention import/실행 OK
- numba JIT 실제 실행 OK (`mmdet3d.core.bbox.box_np_ops.points_in_rbbox`)
- `SSR_e2e.py` config에서 모델 빌드 → **34.8M params**, GPU 이동 OK
- `tools/train.py` / `tools/test.py` import OK
- 데이터 파이프라인: val dataset **6019 샘플** 로드, 샘플 1개 이미지 텐서 `(6, 3, 384, 640)` 확인

---

## 실행 시 참고사항

- **미설치 패키지 (필요 시 수동 설치)**
  - `seaborn` — `tools/analysis_tools/analyze_logs.py` 전용. requirements.txt에 없어 설치 안 함.
    필요하면 `pip install seaborn==0.11.2` (numpy 1.19.5 호환 버전).
  - `tensorflow`, `waymo-open-dataset` — `kitti2waymo.py` 전용. nuScenes 실험에는 불필요.
- **map eval**: config가 참조하는 `data/nuscenes/nuscenes_map_anns_val.json` 이 아직 없다.
  첫 eval 때 자동 생성되지만 시간이 걸린다.
- **shapely RuntimeWarning**: 데이터 파이프라인에서 `invalid value encountered in intersection`
  경고가 다수 출력된다. shapely 2.x + VAD map 처리 조합의 알려진 노이즈로 동작에는 문제 없음.
- **GPU 점유**: 작업 시점 기준 GPU 6, 7번을 hipad 작업이 사용 중이었다. 8-GPU 학습 시 충돌하므로
  확인 후 실행할 것.

  ```shell
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.run --nproc_per_node=6 \
    --master_port=2333 tools/train.py projects/configs/SSR/SSR_e2e.py \
    --launcher pytorch --deterministic --work-dir path/to/save/outputs
  ```
