# Planning distillation on NAVSIM

`docs/PLANNING_DISTILLATION.md` describes the nuScenes/mmdet3d implementation
under `projects/`. This document covers the same two-stage contract rehosted on
the navsim PARA-SSR agent, which is what the teacher cache on blackwell64
actually addresses.

## Why the port exists

The `aux_distill` work is nuScenes/mmdet3d. The teacher BEV cache that exists is
NAVSIM: its manifest reads `ann_file: navsim_infos_train.pkl`, 103288 train /
12146 val samples, NAVSIM class names, and `point_cloud_range
[0, -32, -3, 32, 32, 5]`. The mmdet3d student cannot consume it, so the
distillation moves to the navsim agent that the cache was written for.

The geometry lines up exactly, which is what makes the port meaningful rather
than merely type-correct:

| | teacher (mmdet3d LiDAR frame) | student (PARA-SSR frame) |
| --- | --- | --- |
| rows | `x_forward` `[0, 32]` m, 0.32 m/cell | `bev_h` = `y_forward` `[0, 32]` m, 0.32 m/cell |
| cols | `y_left` `[-32, 32]` m, 0.64 m/cell | `bev_w` = `x_right` `[-32, 32]` m, 0.64 m/cell |
| grid | 100 x 100 x 256 | 100 x 100 x 256 |

Rows already agree and the columns run in opposite directions, so the whole
conversion is the manifest's `student_bev = teacher_bev[:, :, ::-1]`. No
resampling is involved.

## Cache layout

```text
$DISTILL_FEATURE_ROOT/
  bevfusion/cache_{train,val}_100x100/manifest.json
  bevfusion/cache_{train,val}_100x100/samples/<token[:2]>/<token>.npz
```

`<token>` is the NAVSIM frame token, the same value the target builder reads as
`scene.frames[cur].token`. Each `.npz` stores `bev_feature` as `[C, H, W]`
float16 -- note this differs from the nuScenes `npz_xy` caches, which store
`[X*Y, C]` xy-major tokens.

`TeacherFeatureStore.validate_manifest` compares the recorded manifest against
the live config and refuses a mismatch in grid, channels or **physical extent**.
The extent check is the one that matters: a nuScenes cache has the same
100x100x256 shape and would otherwise train for days against cells that are not
the same places.

## The two stages

```text
stage 1   cached teacher BEV -> trainable adapter -> unchanged planning decoder
                             -> trajectory loss            (no images are read)

stage 2   images -> student BEV -> ordinary planning head  -> trajectory loss
                               └-> frozen stage-1 adapter  -> feature MSE
```

In stage 2 the *same adapter instance* processes the teacher and the student
feature and its parameters are frozen, so it cannot learn to hide a mismatch;
the feature-loss gradient passes through it into the student BEV encoder.
Neither the adapters nor the cache exist at inference.

Stage 1 is not optional. `PlanningDistillation` refuses to build without a
stage-1 checkpoint, because "distil towards a randomly initialised adapter" is
not the experiment.

## Configuration

```python
from dataclasses import replace
from navsim.agents.para_ssr.configs.default import ParaSSRConfig

stage1 = replace(ParaSSRConfig(),
                 use_distill=True,          # emits targets["scene_token"]
                 input_target=True,         # forward() receives targets
                 distill_feature_root="/home/external-user/datasets/teacher_cache")

stage2 = replace(ParaSSRConfig(),
                 use_distill=True,
                 distill_feature_root="/home/external-user/datasets/teacher_cache",
                 distill_adapter_checkpoints={"bevfusion": "/path/to/stage1.ckpt"})
```

`distill_branches` defaults to BEVFusion alone. A single teacher is supported
deliberately: the MapTRv2 cache does not exist yet, and requiring both would
block the BEVFusion-only experiment that can run today. Add MapTRv2 by putting a
second entry in `distill_branches` and a second checkpoint path.

## Token plumbing

The frame token is privileged, so it belongs in targets and never in features --
`compute_features` receives an `AgentInput`, which has no ground truth and no
token. `use_distill` therefore turns on the same `targets["scene_token"]` the
metric planner already used.

That token is part of the cached target tensor, so it is also part of the target
cache digest (`cache_key.py`). Without that, enabling distillation on an
existing cache would silently reuse target files that have no token in them.

## Files

```text
navsim/agents/para_ssr/distill/adapter.py          PlanningBEVAdapter, BEV layout helpers
navsim/agents/para_ssr/distill/teacher_store.py    cache loading + geometry validation
navsim/agents/para_ssr/distill/distillation.py     stage-2 frozen-adapter module
navsim/agents/para_ssr/distill/teacher_adapter.py  stage-1 agent and model
tests/test_para_ssr_distill.py                     14 tests, synthetic cache
```

`ParaSSRPlannerHead.forward_from_bev` was split out of `forward` so stage 1 can
drive the unchanged decoder from a cached BEV; `forward` passes the `bev_pos` it
already computed, so behaviour is unchanged.

## Status

Working and tested: the loader against the real 568 GB BEVFusion cache, stage-1
training, stage-2 frozen-adapter gradient flow into the student BEV, and the
target-cache digest.

Not yet possible: MapTRv2 is not cached, so only the BEVFusion branch can run.
The stage-1 checkpoint has to be trained on this machine before stage 2 starts.
