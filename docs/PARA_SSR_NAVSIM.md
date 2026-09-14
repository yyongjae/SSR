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

New runs use the three front cameras `cam_f0`, `cam_l0`, `cam_r0`, matching
WoTE's camera set, plus the merged LiDAR point cloud at the same history
frames `[2, 3]`. BEVFormer receives separate calibrated views at 768 x 416;
the camera batch shape is `[B, 2, 3, 3, 416, 768]`. Each frame's point cloud
is clipped to the front ROI and `lidar_z_range`, rotated into SSR axes and
zero-padded to `lidar_max_points` rows: `lidar_points` is
`[B, 2, 65536, 5]` (x_right, y_forward, z, intensity, ring) with the real row
count in `lidar_num_points` `[B, 2]`. The other five cameras are disabled at
sensor loading. `use_lidar: false` restores the camera-only arm, whose
feature cache is kept apart by name.

The LiDAR wiring follows SafeDrive (see
[report 12](../report/12_lidar_bev_encoder_50x100.md)): SafeDrive's
`SpMiddleResNetFHD` (sparse 3D SECOND on spconv 2.x, 0.08 m voxels, stride 8)
turns the cloud into a `[256, 50, 100]` BEV that replaces the learned BEV
query table and is read by a gated deformable `lidar_cross_attn` in every
encoder layer, for history frames as well as the current one. It needs
`spconv`: the prebuilt `spconv-cu126==2.3.8` wheel (in
`requirements_navsim.txt`) runs on torch 2.8+cu128 and the RTX 5090 (sm_120)
without a source build. `lidar_encoder: pillar` selects an spconv-free
PointPillars-style encoder with the same output contract.

The shared BEV, detection/motion head and vector-map head all use one front ROI:
`x_right` from -32 to 32 m and `y_forward` from 0 to 32 m, on a `50 x 100`
grid of square 0.64 m cells (`bev_h` rows over `y_forward`, `bev_w` columns
over `x_right`). This is the BEVFusion teacher's 50 x 100 grid, so its cached
BEV maps onto `bev_embed` cell for cell after a lateral flip. Detection/motion GT
also uses a box-centre bearing filter of +/-80 degrees about the forward axis;
training targets and auxiliary mAP targets call the same predicate. Map GT uses
the common rectangular ROI without an angular mask.

Start a new training run: the camera embedding has 3 rows, the BEV grid is
50 x 100, the encoder carries a LiDAR branch and the task ROI/GT population
changed, so an old checkpoint cannot be loaded as an equivalent experiment
under these defaults. Evaluating a legacy checkpoint with
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
output is `eval/para_ssr_front3`. Give every PDM evaluation its own
`EVAL_EXPERIMENT=eval/<name>` (the wrapper sets `experiment_name` itself, so
passing `experiment_name=` again is rejected by Hydra as a duplicate override).
Build the navtest metric cache first with
`bash scripts/evaluation/cache_metric_navtest.sh` if `data/exp/metric_cache` is
empty.  NAVSIM v2 EPDMS for these checkpoints is described in
`report/README.md` §4.3.

## 5. Auxiliary detection / vector-map mAP

The auxiliary runner uses metric protocol V3: GT is restricted to the shared
front ROI (`x_right` -32..32 m, `y_forward` 0..32 m), detection GT additionally
passes the training-time +/-80 degree box-centre FOV filter, and only the tasks
whose heads exist in the checkpoint are scored.  V2 numbers (rear-inclusive ROI)
are not comparable; reproduce them with their original code revision and config.

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
training run.  Its `use_det_motion_head`/`use_map_head` flags select the scored
tasks (recorded as `metrics/tasks`); a plan-only checkpoint is rejected.  navtest
data is read from `data/dataset/{navsim_logs,sensor_blobs}/test` unless
`NAVSIM_DOWNLOAD` points at an unpacked download holding
`test_navsim_logs/test` and `test_sensor_blobs/test` (the PDM and metric-cache
wrappers honour the same variable).  The evaluator processes all 12,146 `navtest` tokens, using one
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
- `NAVSIMAuxMap/chamfer_mAP`: road, walkway, centerline and crosswalk
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
