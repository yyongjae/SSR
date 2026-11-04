#!/usr/bin/env bash
# Python environment of the release path: an env at the prefix of SSR_PY, with the versions of constraints.txt.
# Needs conda on PATH (or CONDA=<path to conda>) and network; ~10 min.  Refuses an interpreter that already has torch.
#
# docs/install.md, requirements.txt, environment.yml and setup_fix.md describe older stacks and do not apply here.
# pip check will say para-ssr-navsim requires torch 2.0.1: expected, the repo is installed with --no-deps.
export SSR_BOOTSTRAP=1     # SSR_PY may not exist yet
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
HERE=$_SETUP_HERE
ENV_DIR=$(dirname "$(dirname "$SSR_PY")")
CONDA=${CONDA:-conda}
C=(-c "$HERE/constraints.txt")

if [ -x "$SSR_PY" ] && "$SSR_PY" -m pip show torch >/dev/null 2>&1; then
  die "$SSR_PY already has torch: choose another SSR_PY or move $ENV_DIR away first"
fi
[ -x "$SSR_PY" ] || run "$CONDA" create -y -p "$ENV_DIR" --override-channels -c conda-forge python=3.9.23 pip=23.3.1 nb_conda_kernels

run "$SSR_PY" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# requirements_navsim.txt pins torch 2.0.1 / torchvision 0.15.2, which have no kernels for current GPUs: drop those two
# lines.  The nuplan-devkit tag there resolved to commit ce3c323af01c0d7ec5672f7832ef53f9c679aab0.
REQ=$(mktemp)
trap 'rm -f "$REQ"' EXIT
grep -vE '^(torch|torchvision)==' "$REPO/requirements_navsim.txt" > "$REQ"
run "$SSR_PY" -m pip install -r "$REQ" "${C[@]}"

run "$SSR_PY" -m pip install numpy==1.23.4 scipy==1.13.1 shapely==2.0.7 pandas==2.3.3 pyarrow==21.0.0 \
  geopandas==1.0.1 rasterio==1.3.11 fiona==1.10.1 pyogrio==0.11.1 rtree==1.4.1 \
  opencv-python==4.9.0.80 scikit-learn==1.2.2 hydra-core==1.2.0 omegaconf==2.3.1 \
  pytorch-lightning==2.2.1 torchmetrics==1.8.2 timm==1.0.28 huggingface_hub==1.8.0 safetensors==0.7.0 \
  tensorboard==2.16.2 protobuf==4.25.3 wandb==0.26.1 ray==2.51.2 SQLAlchemy==1.4.27 setuptools==65.5.1 \
  positional-encodings==6.0.1 pyquaternion==0.9.9 spconv-cu126==2.3.8 cumm-cu126==0.7.11 pytest==8.4.2 "${C[@]}"

# without deps: setup.py would pull requirements_navsim.txt with torch 2.0.1.  One editable install per interpreter:
# run_training.py refuses a navsim that resolves outside its own checkout.
run "$SSR_PY" -m pip install -e "$REPO" --no-deps

if [ "$SETUP_DRY" != 1 ]; then
  "$SSR_PY" -m pip check || true
  "$SSR_PY" -m pip freeze | grep -v '^-e ' | diff - <(grep -v '^-e ' "$HERE/pip_freeze_ssr.txt") \
    || say "the freeze differs from pip_freeze_ssr.txt (above); conda build strings are expected to differ"
fi
say "done"
