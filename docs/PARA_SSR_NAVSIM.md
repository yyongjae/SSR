# PARA-SSR NAVSIM setup

The port is tested with Python 3.9, PyTorch 2.0.1+cu118, torchvision 0.15.2,
Hydra 1.2.0, PyTorch Lightning 2.2.1, NumPy 1.23.4 and timm 1.0.28.
The first model construction downloads torchvision's ResNet-50 `tv_in1k`
checkpoint unless it is already present in the torch cache.

## 1. Data link

From the repository root:

```bash
# Replace this with the NAVSIM root on the local machine.
ln -s /path/to/navsim ./data

# Keep outputs in the repository, or replace this directory with a symlink to
# a larger experiment volume.
mkdir -p ./work_dirs
# ln -s /path/to/experiment-storage ./work_dirs
```

Do not replace an existing `data` or `work_dirs` path.  The physical storage
location is user-defined; the repository-relative contract is what matters.
Expected paths include `data/dataset/maps`, `data/dataset/navsim_logs`,
`data/dataset/sensor_blobs` and `data/exp/metric_cache`.

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

An optional candidate planner with NAVSIM PDM supervision is available as
`agent=para_ssr_metric_agent`. See [the metric planner guide](../report/11_para_ssr_metric_planner.md)
for train-only anchors, world-cache preparation, smoke checks and ablation flags.

New runs use the three front cameras `cam_f0`, `cam_l0`, `cam_r0`, matching
WoTE's camera set. BEVFormer receives separate calibrated views at 768 x 416
for history frames `[2, 3]`; the batch shape is `[B, 2, 3, 3, 416, 768]`.
The other five cameras and LiDAR are disabled at sensor loading. WoTE's
single-frame panorama preprocessing and LiDAR fusion are not used here.

The shared BEV, detection/motion head and vector-map head all use one front ROI:
`x_right` from -32 to 32 m and `y_forward` from 0 to 32 m. Detection/motion GT
also uses a box-centre bearing filter of +/-80 degrees about the forward axis;
training targets and auxiliary mAP targets call the same predicate. Map GT uses
the common rectangular ROI without an angular mask.

Start a new front-camera training run: the camera embedding now has 3 rows and
the task ROI/GT population changed, so an old checkpoint cannot be loaded as an
equivalent experiment under these defaults. Evaluating a legacy checkpoint with
a mismatched/rear ROI requires its original code revision as well as its archived
config; the current agent deliberately rejects that spatial protocol. Feature
and target cache identities include the relevant camera/ROI/FOV settings.

```bash
TRAIN_GPUS=0,1  # replace with two available device IDs
EVAL_GPU=0      # PDM evaluation is single-GPU

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" bash scripts/training/train_para_ssr.sh
CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  bash scripts/evaluation/eval_para_ssr.sh /absolute/path/to/checkpoint.ckpt
```

`RESUME_CHECKPOINT=/absolute/path/to/last.ckpt` performs a full Lightning
resume, including optimizer, scheduler, epoch and GradBalancer state.  The
agent's `checkpoint_path` is a weights-only load intended for evaluation.
The default training experiment is `para_ssr_front3`, and the default PDM
output is `eval/para_ssr_front3`. Override the experiment name for each run.

## 5. Auxiliary detection / vector-map mAP

Current limitation: the auxiliary runner still fixes the legacy full detection
ROI and V2 GT-count reference. It rejects the new front-only baseline/metric
checkpoints until a separately validated ROI/FOV protocol migration is made.
The instructions below describe the legacy protocol; reproduce old checkpoints
with their original code revision and config. Official PDM evaluation above is
not affected by this auxiliary-runner limitation.

NAVSIM's official benchmark scores the predicted ego trajectory, not the
detector or vector-map heads.  PARA-SSR's two perception heads can nevertheless
be measured on the held-out `navtest` annotations with the explicitly
non-official `NAVSIMAuxDet` and `NAVSIMAuxMap` protocols:

```bash
AUX_GPUS=0,1  # replace with the available device IDs
GPU_IDS="$AUX_GPUS" \
AUX_EXPERIMENT=eval/my_model_aux \
AUX_TRAINING_CONFIG=/absolute/path/to/run/code/hydra/config.yaml \
scripts/evaluation/eval_para_ssr_aux.sh /absolute/path/to/model.ckpt
```

The checkpoint must be paired with the Hydra config archived by the same
training run.  The evaluator processes all 12,146 `navtest` tokens, using one
deterministic shard per selected GPU for fp32 inference and then one CPU
aggregation process.  Completed per-token records are resumable, so the same
command safely continues an interrupted run.  The wrapper uses the active
environment's `python`; set `SSR_NAVSIM_PYTHON` only when an explicit interpreter
is required.

Results are written to:

```text
work_dirs/eval/my_model_aux/manifest.json
work_dirs/eval/my_model_aux/records/<token>.npz
work_dirs/eval/my_model_aux/aux_metrics.json
work_dirs/eval/my_model_aux/aux_metrics.csv
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
metrics.  Use a unique `AUX_EXPERIMENT` for every checkpoint/config pair; the
manifest deliberately rejects a different identity in an existing output
directory.
