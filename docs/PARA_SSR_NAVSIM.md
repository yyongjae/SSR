# PARA-SSR NAVSIM setup

The port is tested with Python 3.9, PyTorch 2.0.1+cu118, torchvision 0.15.2,
Hydra 1.2.0, PyTorch Lightning 2.2.1, NumPy 1.23.4 and timm 1.0.28.
The first model construction downloads torchvision's ResNet-50 `tv_in1k`
checkpoint unless it is already present in the torch cache.

## 1. Data link

From the repository root:

```bash
ln -s /data/navsim data
```

Do not replace an existing `data` path.  The expected paths include
`data/dataset/maps`, `data/dataset/navsim_logs`, `data/dataset/sensor_blobs`
and `data/exp/metric_cache`.

## 2. Environment

```bash
conda env create -f environment.yml
conda activate ssr-navsim

# Install the CUDA 11.8 wheels explicitly on GPU machines.
python -m pip install \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements_navsim.txt
python -m pip install -e . --no-deps
```

`requirements_navsim.txt` installs nuPlan devkit v1.2 from its tagged Git
revision.  `--no-deps` on the last command avoids resolving the same pinned
requirements twice; it does not omit anything when the preceding command has
completed successfully.

Set these variables for manual Python entry points.  The supplied train/eval
scripts set the same values themselves.

```bash
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT="$PWD/data/dataset/maps"
export OPENSCENE_DATA_ROOT="$PWD/data/dataset"
export NAVSIM_DEVKIT_ROOT="$PWD"
export NAVSIM_EXP_ROOT="$PWD/work_dirs"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

## 3. Verification

```bash
python -c "import navsim, nuplan, torch, timm, pytorch_lightning"
python -m pytest -q
CUDA_VISIBLE_DEVICES=0 bash scripts/training/smoke_para_ssr.sh
```

The smoke command consumes eight real NAVSIM scenes and runs one training
batch, backward pass, optimizer step and validation batch in fp32.

## 4. Training and evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/training/train_para_ssr.sh
bash scripts/evaluation/eval_para_ssr.sh /absolute/path/to/checkpoint.ckpt
```

`RESUME_CHECKPOINT=/absolute/path/to/last.ckpt` performs a full Lightning
resume, including optimizer, scheduler, epoch and GradBalancer state.  The
agent's `checkpoint_path` is a weights-only load intended for evaluation.
