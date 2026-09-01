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

## 5. Auxiliary detection / vector-map mAP

NAVSIM's official benchmark scores the predicted ego trajectory, not the
detector or vector-map heads.  PARA-SSR's two perception heads can nevertheless
be measured on the held-out `navtest` annotations with the explicitly
non-official `NAVSIMAuxDet` and `NAVSIMAuxMap` protocols:

```bash
GPU_IDS=2,3 scripts/evaluation/eval_para_ssr_aux.sh
```

The default run binds the epoch-30 checkpoint to its archived training Hydra
config and evaluates all 12,146 `navtest` tokens.  It uses two disjoint GPU
shards for fp32 inference, then performs one CPU aggregation.  Completed
per-token records are resumable, so the same command safely continues an
interrupted run.

Results are written to:

```text
work_dirs/eval/para_ssr_ep30_aux/manifest.json
work_dirs/eval/para_ssr_ep30_aux/records/<token>.npz
work_dirs/eval/para_ssr_ep30_aux/aux_metrics.json
work_dirs/eval/para_ssr_ep30_aux/aux_metrics.csv
```

Metric definitions:

- `NAVSIMAuxDet/center_mAP`: 7 NAVSIM classes, sigmoid flattened query/class
  top-100 decoding, class-aware 2-D center-distance matching at 0.5, 1, 2 and
  4 m, followed by the nuScenes-style 101-bin AP calculation.
- `NAVSIMAuxMap/chamfer_mAP`: divider, pedestrian-crossing and boundary
  vectors, flattened top-100 decoding, 100-point arclength resampling,
  symmetric Chamfer matching at 0.5, 1.0 and 1.5 m, followed by
  precision-envelope area AP.

Detection and map GT are rebuilt from each original `Scene` without the
training-time 100-instance cap.  The runner refuses partial token sets,
non-finite outputs, wrong tensor contracts, a mismatched checkpoint/config, or
records copied from another evaluation manifest.  These numbers are useful for
PARA-SSR ablations but must not be reported as official NAVSIM leaderboard
metrics.

For another checkpoint, use a unique output experiment and the Hydra config
archived with that checkpoint:

```bash
GPU_IDS=2,3 \
AUX_EXPERIMENT=eval/my_model_aux \
AUX_TRAINING_CONFIG=/absolute/path/to/code/hydra/config.yaml \
scripts/evaluation/eval_para_ssr_aux.sh /absolute/path/to/model.ckpt
```
