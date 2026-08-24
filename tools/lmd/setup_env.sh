#!/usr/bin/env bash
# Build the `ssr` conda env on this machine (Xeon Gold 6526Y / A6000 / driver 535).
#
# Follows setup_fix.md, which is the procedure that actually worked -- NOT
# docs/install.md, which picks wrong versions at five separate steps. The two
# things that matter most and are easy to get wrong:
#   * numpy must be pinned to 1.19.5 BEFORE mmdet3d is built, or its CUDA ops
#     compile against 1.24 headers and the ABI does not match at runtime.
#   * this CPU is a Sapphire/Emerald Rapids part and llvmlite 0.31's LLVM 8 does
#     not know it, so `import numba` segfaults without NUMBA_CPU_NAME=generic.
#
#   bash tools/lmd/setup_env.sh 2>&1 | tee /tmp/ssr_env_setup.log
set -euo pipefail

ENV_NAME=ssr
MMDET3D_DIR=/home/byounggun/mmdetection3d_ssr   # outside the repo: `setup.py develop`
                                                # is an egg-link, so it must not be
                                                # shared with another project
CUDA=/usr/local/cuda-11.1

eval "$(conda shell.bash hook)"

echo "=== 1/8  create env ==="
conda create -n "$ENV_NAME" python=3.8 -y
conda activate "$ENV_NAME"

echo "=== 2/8  setuptools down first (mmdet3d's setup.py develop breaks on 75) ==="
pip install "setuptools==58.2.0" "wheel==0.37.1"

echo "=== 3/8  torch 1.9.1+cu111 ==="
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html

echo "=== 4/8  mmcv-full 1.4.0 (prebuilt wheel, not the sdist) ==="
pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

echo "=== 5/8  mmdet / mmseg / timm / nuscenes-devkit ==="
pip install mmdet==2.14.0 mmsegmentation==0.14.1 timm==0.6.12
pip install nuscenes-devkit==1.1.9

echo "=== 6/8  pin the dependency versions (this step is not in docs/install.md) ==="
pip install "numpy==1.19.5" "numba==0.48.0" "llvmlite==0.31.0" "scipy==1.7.3" \
  "pandas==1.3.5" "matplotlib==3.5.3" "scikit-image==0.19.3" \
  "opencv-python==4.7.0.72" "networkx==2.2" "yapf==0.33.0" "protobuf==3.20.3" \
  "shapely==2.0.1" "plyfile==0.7.4" "pillow==9.5.0" "similaritymeasures==0.7.0" \
  "tensorboard==2.9.0" "lyft-dataset-sdk==0.0.8" "trimesh==2.35.39" \
  "setuptools==58.2.0"

echo "=== 7/8  activate hooks (numba workaround + CUDA paths) ==="
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/ssr_env.sh" <<EOF
#!/bin/bash
# llvmlite 0.31 (LLVM 8) segfaults probing this CPU; pin it to a generic target.
export NUMBA_CPU_NAME=generic
export NUMBA_CPU_FEATURES=
export CUDA_HOME=$CUDA
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6"
EOF
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/ssr_env.sh" <<'EOF'
#!/bin/bash
unset NUMBA_CPU_NAME NUMBA_CPU_FEATURES CUDA_HOME TORCH_CUDA_ARCH_LIST
EOF
conda deactivate && conda activate "$ENV_NAME"

echo "=== 8/8  build mmdet3d 0.17.1 LAST, with gcc-9, against the pinned numpy ==="
if [ ! -d "$MMDET3D_DIR" ]; then
  git clone https://github.com/open-mmlab/mmdetection3d.git "$MMDET3D_DIR"
fi
cd "$MMDET3D_DIR"
git checkout -f v0.17.1
# gcc-5 (docs/install.md step c) does not know sm_86; the system default gcc-10
# is unstable with CUDA 11.1. gcc-9 is the one that works here.
export CUDA_HOME=$CUDA CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9
export TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=16
python setup.py develop

echo
echo "=== verify ==="
cd /home/byounggun/SSR
# torch must be imported before numba -- see setup_fix.md
python - <<'EOF'
import torch, mmcv, mmdet, mmseg, mmdet3d, numpy
print(f'torch     {torch.__version__} cuda={torch.version.cuda} '
      f'avail={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')
print(f'numpy     {numpy.__version__}   (must be 1.19.5)')
print(f'mmcv      {mmcv.__version__}    mmdet {mmdet.__version__}  '
      f'mmseg {mmseg.__version__}  mmdet3d {mmdet3d.__version__}')
from mmcv.ops import nms
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
print('mmcv CUDA ops + MSDeformAttn OK')
EOF
echo
echo "DONE. now: conda activate $ENV_NAME"
