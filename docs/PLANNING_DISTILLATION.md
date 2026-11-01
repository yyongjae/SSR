# Planning distillation (BEVDepth/HDMapNet and BEVFusion/MapTRv2)

## Experiment contract

The implementation is deliberately split into two trainable stages and one
offline extraction step.

```text
frozen BEVDepth detection BEV ── fixed metric alignment ──┐
                                                         ├─ 25x25x256 cache
frozen HDMapNet mapping BEV  ── fixed metric alignment ──┘

stage 1, per teacher:
    cached BEV -> residual 2-layer MLP adapter -> resize 100x100
               -> unchanged SSR sparse planning decoder -> trajectory loss

stage 2:
    SSR images -> student BEV -> ordinary SSR planning head -> trajectory loss
                           ├-> frozen BEVDepth adapter -> feature MSE
                           └-> frozen HDMapNet adapter  -> feature MSE
```

In stage 2, the adapter parameters are frozen and the **same module instance**
processes teacher and student features. They cannot learn to hide the mismatch;
the feature-loss gradient passes through them into the student BEV encoder.
The adapters and teacher cache are absent at inference.

The same contract is used twice:

- pair A: BEVDepth (detection) + HDMapNet (mapping), 25x25 `.pt` caches
- pair B: BEVFusion (detection) + MapTRv2 (mapping), 100x100 `.npz` caches

Pair B does not need teacher repositories on this machine; see section 7.

The teacher taps are intentionally task-specific and both have 256 channels:

- BEVDepth: detection-head FPN output (`model.head.neck`, 128x128).
- HDMapNet: shared map-decoder output (`model.bevencode.up1`, 100x200).

The released teachers use nuScenes ego coordinates (`x=forward, y=left`). SSR
uses the calibrated LIDAR_TOP frame (`x=right, y=forward`), with columns along
x and rows along y. Cache generation therefore applies
`teacher_x=ssr_y, teacher_y=-ssr_x` (swap x/y and flip teacher y) while
resampling into SSR's `x=[-15,15], y=[-30,30]` range. Every cache manifest
records this transform.

## 1. Teacher repositories and environments

The teachers have incompatible legacy dependencies, so use their own working
environments. The cache is the interface; neither repository is imported by
student training.

The prepared Conda environments and local teacher repositories are:

```text
BEVDepth: /home/byounggun/anaconda3/envs/bevdepth
          /data2/byounggun/rideflux/BEVDepth
HDMapNet: /home/byounggun/anaconda3/envs/pmapnet
          /data2/byounggun/rideflux/P-MapNet
```

Reference revisions used while implementing the taps:

```bash
git clone https://github.com/Megvii-BaseDetection/BEVDepth.git
git -C BEVDepth checkout d78c7b58b10b9ada940462ba83ab24d99cae5833

git clone https://github.com/jike5/P-MapNet.git
git -C P-MapNet checkout b8b4cf2295ee75826046eef9cfa12b107fb43619
```

The checkpoints already present are:

```text
/data2/byounggun/rideflux/pretrained_checkpoints/
  bevdepth_nuscenes_r50_256x704_cbgs.pth
  hdmapnet_nuscenes_60x30_lidar_camera.pth
```

The HDMapNet file is P-MapNet's nuScenes 60x30 m LiDAR+Camera baseline
reproduction because the original HDMapNet repository does not publish a
checkpoint.

## 2. Generate BEVDepth-native infos

Do not overwrite `/data/nuscenes/nuscenes_infos_{train,val}.pkl`; those may be
MMDetection-format files. In the BEVDepth environment, generate distinct files:

```bash
python tools/distill/generate_bevdepth_infos.py \
  --teacher-repo /data2/byounggun/rideflux/BEVDepth \
  --data-root /data/nuscenes \
  --split train \
  --output /data2/byounggun/rideflux/pretrained_checkpoints/bevdepth_infos_train.pkl

python tools/distill/generate_bevdepth_infos.py \
  --teacher-repo /data2/byounggun/rideflux/BEVDepth \
  --data-root /data/nuscenes \
  --split val \
  --output /data2/byounggun/rideflux/pretrained_checkpoints/bevdepth_infos_val.pkl
```

## 3. Cache frozen teacher BEVs

The prepared wrapper scripts cache both train and val splits. They invoke the
correct Conda-environment Python directly, so activating an environment first
is not required:

```bash
# BEVDepth on physical GPU 0
./tools/distill/cache_bevdepth.sh

# HDMapNet on physical GPU 1
./tools/distill/cache_hdmapnet.sh
```

Pass cache-tool options through the wrapper, for example
`./tools/distill/cache_bevdepth.sh --limit 32`. Existing samples are skipped,
so the full command safely resumes after a smoke run or interruption.

Set a shared output root:

```bash
CACHE=/data2/byounggun/rideflux/pretrained_checkpoints/distill_bev_cache
CKPT=/data2/byounggun/rideflux/pretrained_checkpoints
```

Run BEVDepth extraction on GPU 0 in the `bevdepth` environment:

```bash
conda activate bevdepth

CUDA_VISIBLE_DEVICES=0 python tools/distill/cache_teacher_bev.py \
  --teacher bevdepth --teacher-repo /data2/byounggun/rideflux/BEVDepth \
  --checkpoint "$CKPT/bevdepth_nuscenes_r50_256x704_cbgs.pth" \
  --data-root /data/nuscenes --split train \
  --info-path "$CKPT/bevdepth_infos_train.pkl" --cache-root "$CACHE"

CUDA_VISIBLE_DEVICES=0 python tools/distill/cache_teacher_bev.py \
  --teacher bevdepth --teacher-repo /data2/byounggun/rideflux/BEVDepth \
  --checkpoint "$CKPT/bevdepth_nuscenes_r50_256x704_cbgs.pth" \
  --data-root /data/nuscenes --split val \
  --info-path "$CKPT/bevdepth_infos_val.pkl" --cache-root "$CACHE"
```

Run HDMapNet extraction on GPU 1 in the `pmapnet` environment:

```bash
conda activate pmapnet

CUDA_VISIBLE_DEVICES=1 python tools/distill/cache_teacher_bev.py \
  --teacher hdmapnet --teacher-repo /data2/byounggun/rideflux/P-MapNet \
  --checkpoint "$CKPT/hdmapnet_nuscenes_60x30_lidar_camera.pth" \
  --data-root /data/nuscenes --split train --cache-root "$CACHE"

CUDA_VISIBLE_DEVICES=1 python tools/distill/cache_teacher_bev.py \
  --teacher hdmapnet --teacher-repo /data2/byounggun/rideflux/P-MapNet \
  --checkpoint "$CKPT/hdmapnet_nuscenes_60x30_lidar_camera.pth" \
  --data-root /data/nuscenes --split val --cache-root "$CACHE"
```

Use `--limit 32` first for a smoke cache. Existing entries are resumed safely;
`--overwrite` is required to replace one. Full train+val caches for both
teachers occupy roughly 21 GiB at fp16.

Each split writes a manifest containing the checkpoint SHA-256, repository
path, feature tap, native shape, coordinate transform and number of samples.

## 4. Train the teacher adapters and planning heads

Run the independent six-epoch jobs concurrently in the SSR environment:

```bash
mkdir -p out/lmd/logs

PORT=29501 nohup ./run.sh teacher-bevdepth 0,1 \
  > out/lmd/logs/train_bevdepth_adapter.log 2>&1 &

PORT=29502 nohup ./run.sh teacher-hdmapnet 2,3 \
  > out/lmd/logs/train_hdmapnet_adapter.log 2>&1 &
```

The configs inherit `projects/configs/SSR/DISTILL_teacher_adapters.py` and each
construct only its selected branch. They train for 6 epochs. The only trainable
prefixes in each job are:

```text
GPU 0,1: branches.bevdepth.adapter.*, branches.bevdepth.planner.*
GPU 2,3: branches.hdmapnet.adapter.*, branches.hdmapnet.planner.*
```

The two raw checkpoints are the stage-2 adapter sources:

```text
/data2/byounggun/rideflux/pretrained_checkpoints/planning_distill_checkpoints/
  teacher_bevdepth/epoch_6.pth
  teacher_hdmapnet/epoch_6.pth
```

Both teacher planning heads remain stored in these checkpoints but are not
instantiated in stage 2. Stage 2 strictly restores and freezes only each
teacher's adapter, then compares teacher/student outputs in that fixed adapter
space.

Each epoch is evaluated during training. To rerun the full val split for a
finished raw checkpoint on one GPU:

```bash
./run.sh eval-teacher-bevdepth
./run.sh eval-teacher-hdmapnet
```

The optional arguments are `[checkpoint] [physical_gpu]`. Evaluation reports
both `plan_L2_stp3_*` (endpoint/MAX, the SSR/UniAD headline protocol) and
`plan_L2_*` (mean error through each horizon, the VAD AVG protocol). Despite
the historical key name, `plan_L2_stp3_*` is an endpoint metric rather than a
distinct ST-P3 protocol. `plan_L2_stp3_avg` is the primary SSR-comparison
number.

## 5. Train the planning-only SSR student

```bash
./run.sh distill 0,1
```

To select other per-teacher adapter checkpoints:

```bash
BEVDEPTH_ADAPTER_CKPT=/path/to/bevdepth_epoch_N.pth \
HDMAPNET_ADAPTER_CKPT=/path/to/hdmapnet_epoch_N.pth \
  ./run.sh distill 0,1
```

The student config is `projects/configs/SSR/DISTILL_SSR_student.py` and matches
the requested no-FFP planning-only regime: 12 epochs, global batch 8, LR 5e-5,
no detection/map/motion/occupancy head. Trainable parameters are the student
image backbone/FPN, BEVFormer encoder and SSR planning head. The two adapter
copies report `requires_grad=False` but transmit gradients to the student BEV.

Important logged quantities:

```text
loss_plan_reg                 ordinary SSR trajectory loss
loss_distill_bevdepth         3D-detection planning-space MSE
loss_distill_hdmapnet         mapping planning-space MSE
distill_cos/*                 teacher/student cosine similarity
distill_rmse/*                unweighted feature RMSE
distill_valid/*               physically covered cache fraction
gnorm/* and gshare/*          pressure on the shared student BEV
```

Stage-1 and student checkpoints are written under
`/data2/byounggun/rideflux/pretrained_checkpoints/planning_distill_checkpoints`
by default. Set `DISTILL_CKPT_OUT_ROOT` to move the complete experiment output
tree, or `DISTILL_STUDENT_WORK_DIR` to keep a particular teacher-epoch
combination in its own student directory.

## 6. Verification and ablations

```bash
python tools/verify_planning_distill.py
```

`./run.sh test` additionally exercises the optional W&B logger and therefore
requires the `wandb` package; it is not required by these TensorBoard-only
distillation configs.

Minimum ablation set for attributing a gain:

1. no distillation (`ssr_noffp_2gpu_b4` equivalent),
2. BEVDepth adapter only,
3. HDMapNet adapter only,
4. both adapters,
5. both adapters with shuffled teacher tokens (negative control).

Keep initialization, global batch, LR, epoch count and evaluation checkpoint
(raw versus EMA) identical. Otherwise a planning gain cannot be attributed to
the teacher feature spaces.

## 7. BEVFusion + MapTRv2 (offline 100x100 npz caches)

The same two-stage contract applies to a second teacher pair.  Those caches
are produced on another machine and rsynced into

```text
/data3/byounggun/rideflux/pretrained_checkpoints/distill_bev_cache/
  bevfusion/cache_{train,val}_100x100/samples/<token[:2]>/<token>.npz
  maptrv2/cache_{train,val}_100x100/samples/<token[:2]>/<token>.npz
```

This SSR environment does **not** need the BEVFusion or MapTR repositories.
`TeacherFeatureStore` converts the xy-major `bev_feature` tokens to SSR CHW
maps on load:

| Teacher | Tap | Stored tokens | Load-time alignment |
| --- | --- | --- | --- |
| BEVFusion (C+L det) | fuser output before decoder backbone | `[10000, 256]`, `token = x*Y + y`, source `±54 m` | `swap_xy + flip_y`, crop to SSR `x=[-15,15], y=[-30,30]`, keep `100x100` |
| MapTRv2 (camera map) | `pts_bbox_head.bev_embed` | `[10000, 256]`, `token = x*Y + y`, already in the SSR range | identity CHW reshape, keep `100x100` |

```bash
# GPU 0: BEVFusion, GPU 1: MapTRv2 (one GPU each, global batch 8)
PORT=29501 ./run.sh teacher-bevfusion 0
PORT=29502 ./run.sh teacher-maptrv2 1

./run.sh eval-teacher-bevfusion
./run.sh eval-teacher-maptrv2

./run.sh distill-bevfusion-maptr 0,1
```

Checkpoints land next to the BEVDepth/HDMapNet ones:

```text
$DISTILL_CKPT_OUT_ROOT/teacher_bevfusion/epoch_6.pth
$DISTILL_CKPT_OUT_ROOT/teacher_maptrv2/epoch_6.pth
$DISTILL_CKPT_OUT_ROOT/student_bevfusion_maptrv2/
```

Configs: `projects/configs/SSR/DISTILL_teacher_bevfusion.py`,
`DISTILL_teacher_maptrv2.py`, `DISTILL_SSR_student_bevfusion_maptrv2.py`.
A missing sample raises `FileNotFoundError`; do not start training while
rsync is still filling `cache_val_100x100`.
