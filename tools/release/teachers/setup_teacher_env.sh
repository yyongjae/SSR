#!/usr/bin/env bash
# The python env both camera teachers run in (BEVFormer DET, Mask2Map MAP): python 3.9, torch 2.8.0+cu128, mmcv-full
# 1.7.2 built from source, mmdet 2.28.2, mmseg 0.30.0 and Mask2Map's vendored mmdetection3d 0.17.2 (patched for
# torch 2.x and mmcv 1.7).  This is Mask2Map_navsim tools/navsim/setup_env.sh (the env the Mask2Map teacher was trained
# and cached in, RTX 5090) with these changes: the env path is an argument; TORCH_CUDA_ARCH_LIST comes from the GPUs of
# this machine (9.0 on H100, 12.0 on RTX 5090) instead of a fixed 12.0, and mmcv-full is always compiled for it (no
# cached wheel); the env's ninja is on PATH (parallel build); ipython for BEVFormer's plugin; OPENBLAS_CORETYPE only on
# AVX-512 CPUs; and the check also covers the BEVFormer side (its plugin, DCNv2 and the deformable-attention op).
# BEVFormer needs no op of its own: mmcv's and mmdet3d 0.17.x.
#
#   bash tools/release/teachers/setup_teacher_env.sh <env> base    # conda env, torch, mmcv-full (~5 min of nvcc), deps
#   bash tools/release/teachers/setup_teacher_env.sh <env> repo    # vendored mmdetection3d ops, editable (needs the clone)
#   bash tools/release/teachers/setup_teacher_env.sh <env> check   # imports + CUDA ops on the first visible GPU
#   bash tools/release/teachers/setup_teacher_env.sh <env> all
#   bash tools/release/teachers/setup_teacher_env.sh <env> arch    # only print the TORCH_CUDA_ARCH_LIST it would build for
#
# TEACHER_SRC: the directory get_teachers.sh cloned into (Mask2Map_navsim/, BEVFormer/).  CONDA, CUDA_HOME, CC, CXX,
# MAX_JOBS and TORCH_CUDA_ARCH_LIST can be overridden.  The editable install ties the env to TEACHER_SRC: moving the
# clone needs the repo stage again.  Never point <env> at an env another job is using.
set -euo pipefail

ENV=${1:?usage: setup_teacher_env.sh <env path> [base|repo|check|all|arch]}
STAGE=${2:-all}
case "$ENV" in /*) ;; *) ENV=$(pwd)/$ENV ;; esac
CONDA=${CONDA:-conda}
TEACHER_SRC=${TEACHER_SRC:?set TEACHER_SRC to the clones of get_teachers.sh}
M2M=$TEACHER_SRC/Mask2Map_navsim
BEV=$TEACHER_SRC/BEVFormer

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
# the env's bin first: torch's extension builder compiles in parallel (MAX_JOBS) only when it finds the env's ninja
export PATH=$ENV/bin:$CUDA_HOME/bin:$PATH
export CC=${CC:-$(command -v gcc-13 || command -v gcc)} CXX=${CXX:-$(command -v g++-13 || command -v g++)}
export MAX_JOBS=${MAX_JOBS:-16}
# numpy 1.23's bundled OpenBLAS misdetects some Xeons (e.g. 8469C) and silently returns wrong linear algebra; its
# SkylakeX kernels need AVX-512, so they are forced only where the CPU has it.
if grep -qw avx512f /proc/cpuinfo 2>/dev/null; then export OPENBLAS_CORETYPE=${OPENBLAS_CORETYPE:-SkylakeX}; fi

# arch_list: TORCH_CUDA_ARCH_LIST if set, else the compute capabilities of this machine's GPUs, e.g. "9.0" (H100) or
# "12.0" (RTX 5090), several "12.0;9.0"; fails without either.  Call it in a plain assignment, so that set -e stops on it.
arch_list() {
  local caps
  if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then echo "$TORCH_CUDA_ARCH_LIST"; return 0; fi
  caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u | paste -sd';') || true
  [ -n "$caps" ] || { echo "no GPU visible to nvidia-smi: set TORCH_CUDA_ARCH_LIST (e.g. 9.0 for H100)" >&2; return 2; }
  echo "$caps"
}

PY=$ENV/bin/python
PIP="$PY -m pip"
step() { echo "[$(date '+%F %T')] $*"; }

stage_base() {
  TORCH_CUDA_ARCH_LIST=$(arch_list)
  export TORCH_CUDA_ARCH_LIST
  if [ ! -x "$PY" ]; then
    step "conda create $ENV (python 3.9, ninja)"
    "$CONDA" create -y -p "$ENV" -c conda-forge --override-channels python=3.9 ninja
  fi
  $PIP install "pip<25.3"  # the repo stage needs the legacy setup.py-develop editable install
  step "torch 2.8.0 + torchvision 0.23.0 (cu128)"
  $PIP install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  # setuptools must be pinned after torch and before any extension build.
  $PIP install setuptools==59.5.0 wheel==0.37.1 Pillow==9.5.0 tqdm numpy==1.23.5

  # built here for these GPUs: --no-cache-dir, or pip reuses a wheel an earlier build left in ~/.cache/pip, whatever
  # arch that one was compiled for; a build for another arch list is redone
  if ! $PY -c "import mmcv, mmcv.ops; assert mmcv.__version__ == '1.7.2'" 2>/dev/null \
      || [ "$(cat "$ENV/.mmcv_cuda_arch" 2>/dev/null)" != "$TORCH_CUDA_ARCH_LIST" ]; then
    step "mmcv-full 1.7.2 from source (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST, MAX_JOBS=$MAX_JOBS)"
    $PIP uninstall -y mmcv-full mmcv 2>/dev/null || true
    MMCV_WITH_OPS=1 FORCE_CUDA=1 $PIP install --no-cache-dir --no-build-isolation --no-binary mmcv-full "mmcv-full==1.7.2"
    echo "$TORCH_CUDA_ARCH_LIST" > "$ENV/.mmcv_cuda_arch"
  fi

  step "mmdet, mmseg and the teachers' runtime deps"
  $PIP install mmdet==2.28.2 mmsegmentation==0.30.0 timm==0.9.16 numba==0.56.4 "nuscenes-devkit==1.1.11" \
      "shapely==1.8.5.post1" pyquaternion terminaltables prettytable descartes pycocotools tensorboard einops \
      trimesh pandas scikit-image matplotlib
  # mmdet3d 0.17.2 imports lyft_eval at module level; the SDK's own install_requires pin a dev toolchain.
  $PIP install --no-deps lyft_dataset_sdk==0.0.8
  # ipython: BEVFormer's plugin imports IPython.embed at module level (datasets/nuscnes_eval.py, samplers/group_sampler.py)
  $PIP install cachetools fire plotly scikit-learn "ipython==8.18.1"
  # The builds above pull numpy 2.x, opencv 5 and yapf 0.43; numba 0.56 needs numpy < 1.24 and mmcv 1.x needs yapf 0.40.
  $PIP install numpy==1.23.5 opencv-python==4.9.0.80 yapf==0.40.1 setuptools==59.5.0

  step "mmcv torch>=2.1 patch"
  [ -f "$M2M/tools/navsim/patch_mmcv_for_torch2.py" ] || { echo "no $M2M: run get_teachers.sh first" >&2; exit 2; }
  $PY "$M2M/tools/navsim/patch_mmcv_for_torch2.py"
}

stage_repo() {
  TORCH_CUDA_ARCH_LIST=$(arch_list)
  export TORCH_CUDA_ARCH_LIST
  step "vendored mmdetection3d ops (editable, no deps; TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
  # build/ holds objects of an earlier build, maybe for another arch list: setup.py would keep them
  rm -rf "$M2M/mmdetection3d/build"
  (cd "$M2M/mmdetection3d" && FORCE_CUDA=1 $PIP install --no-build-isolation --no-deps -e .)
}

stage_check() {
  step "import + CUDA op check (first visible GPU; set CUDA_VISIBLE_DEVICES to choose)"
  # Mask2Map side, as its setup_env.sh: mmcv nms and BEVFusion's bev_pool on the GPU
  (cd "$M2M" && PYTHONPATH="$M2M" $PY - <<'EOF'
import torch, mmcv, mmdet, mmseg, mmdet3d
from mmcv.ops import MultiScaleDeformableAttention, nms
from mmdet3d.ops import bev_pool
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'arch', torch.cuda.get_arch_list())
print('mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, 'mmseg', mmseg.__version__, 'mmdet3d', mmdet3d.__version__)
import projects.mmdet3d_plugin  # registers the plugin modules
assert torch.cuda.is_available()
cap = '%d.%d' % torch.cuda.get_device_capability()
print('gpu', torch.cuda.get_device_name(), 'capability', cap)
b = torch.rand(64, 4, device='cuda'); b[:, 2:] += b[:, :2]
keep = nms(b, torch.rand(64, device='cuda'), 0.5)[1]
assert 0 < keep.numel() <= 64
print('mmcv nms on GPU:', keep.numel(), 'kept')
feats = torch.rand(100, 8, device='cuda')
coords = torch.stack([torch.randint(0, 10, (100,)), torch.randint(0, 10, (100,)), torch.zeros(100, dtype=torch.long),
                      torch.zeros(100, dtype=torch.long)], 1).int().cuda()
out = bev_pool(feats, coords, 1, 1, 10, 10)
assert tuple(out.shape) == (1, 8, 1, 10, 10) and torch.allclose(out.sum(), feats.sum(), rtol=1e-4)
print('bev_pool', tuple(out.shape), 'sum preserved')
EOF
  )
  # BEVFormer side: its plugin, mmcv's DCNv2 (the R101 backbone) and ms_deform_attn (CUDA vs the pytorch reference)
  [ -d "$BEV/projects" ] || { echo "no $BEV: run get_teachers.sh first" >&2; exit 2; }
  (cd "$BEV" && PYTHONPATH="$BEV" $PY - <<'EOF'
import torch
import projects.mmdet3d_plugin  # noqa: F401  registers BEVFormer, CustomNavsimDataset, NMSFreeCoder, ...
from mmcv.cnn import build_conv_layer
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch
from mmdet.models import DETECTORS
from mmdet3d.datasets import DATASETS
assert 'BEVFormer' in DETECTORS.module_dict and 'CustomNavsimDataset' in DATASETS.module_dict
conv = build_conv_layer(dict(type='DCNv2', deform_groups=1), 8, 8, 3, padding=1).cuda()
y = conv(torch.rand(1, 8, 16, 16, device='cuda'))
assert tuple(y.shape) == (1, 8, 16, 16) and torch.isfinite(y).all()
print('DCNv2 on GPU', tuple(y.shape))
shapes = torch.tensor([[8, 8], [4, 4]], device='cuda')
start = torch.cat((shapes.new_zeros((1,)), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]))
value = torch.rand(1, 80, 2, 16, device='cuda')
loc = torch.rand(1, 5, 2, 2, 4, 2, device='cuda')
att = torch.rand(1, 5, 2, 2, 4, device='cuda').softmax(-1)
a = MultiScaleDeformableAttnFunction.apply(value, shapes, start, loc, att, 64)
b = multi_scale_deformable_attn_pytorch(value, shapes, loc, att)
assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())
print('ms_deform_attn CUDA == pytorch reference, max |diff| %.2e' % float((a - b).abs().max()))
EOF
  )
}

case "$STAGE" in
  base) stage_base ;;
  repo) stage_repo ;;
  check) stage_check ;;
  all) stage_base; stage_repo; stage_check ;;
  arch) a=$(arch_list); echo "$a"; exit 0 ;;
  *) echo "usage: $0 <env path> {base|repo|check|all|arch}" >&2; exit 2 ;;
esac
step "done: $STAGE"
