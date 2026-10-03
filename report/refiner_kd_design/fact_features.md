# Fact report: inputs for R_T / R_S / R_GT and the draft and GT sources

This was a read-only run: nothing was modified and no GPU was used. All Python ran on CPU with nice 10, each run under 1 minute. Scratch scripts are in `/tmp/claude-1001/-home-external-user/9262a8d2-fa20-46d7-ba9a-92d7d13b8ceb/scratchpad/`.

## (a) Teacher cache (BEVFusion 50x100)

**Manifest** (`/home/external-user/datasets/teacher_cache/bevfusion/cache_{train,val}_50x100/manifest.json`)
- Checkpoint `runs/navsim-fusion-50x100/epoch_20.pth`, git `cf3e02a`. Experiment E calls this one "teacher-50"; E's main teacher was a different 100x100 checkpoint.
- Layout is `samples/<tok[:2]>/<tok>.npz`, 2,648,014 B per token, uncompressed.
- Train cache: 103,288 tokens (255 GB). This is exactly the whole navtrain table, both TRAIN-IN (85,109) and TRAIN-OUT (18,179). The teacher was trained on `navsim_infos_train.pkl`, so it has seen every navtrain scene.
- Val cache: 12,146 tokens, exactly navtest.
- `cache_val_50x100_future/` is not current-frame data. It holds teacher runs on future frames, has no manifest, and stopped at 1,422 frames. Producer: `SSR/tools/future_teacher_cache/run_cache.sh`; the stop is recorded in report 35 §9.

**npz keys**

| key | shape | dtype | notes |
|---|---|---|---|
| `bev_feature` | (256, 50, 100) | f16 | See scale below |
| `dense_heatmap` | (7, 50, 100) | f16 | Per-class logits, same grid as the BEV |
| `pred_boxes_3d` | (200, 9) | f32 | Box format below |
| `pred_scores_3d` | (200,) | f32 | |
| `pred_labels_3d` | (200,) | i16 | 7 classes: vehicle, pedestrian, bicycle, traffic_cone, barrier, czone_sign, generic_object (identical to `DET_CLASS_NAMES`) |
| `raw_cls_logits` | (7, 200) | f16 | |
| `raw_center` | (2, 200) | f16 | BEV cell units, not metres |
| `raw_height` | (1, 200) | f16 | |
| `raw_dim` | (3, 200) | f16 | |
| `raw_rot` | (2, 200) | f16 | sin, cos |
| `raw_vel` | (2, 200) | f16 | Normalised head output |

**BEV scale** (48 random tokens each)
- The feature is post-ReLU: min 0, and 55.6% of values are exactly zero.
- Mean 0.28, std 0.61–0.63, max about 15–20. Per-channel means range 0.16–0.43. No dead channels.
- Train and val statistics are the same.
- savez_compressed shrinks one BEV from 2.56 MB to about 1.2 MB.

**Axes (verified here)**
- The tensor is (C, H = x_forward, W = y_left). Row i is at x = (i+0.5)·0.64 m; column j is at y_left = −32 + (j+0.5)·0.64 m.
- Check: I took 307 boxes with score > 0.5 on 30 navtest tokens. The class heatmap logit at the cell predicted by the box centre has median 2.15. At the laterally flipped cell it has median −7.38. The predicted cell was higher in 100% of cases.
- So to get the student layout: `student = teacher[:, :, ::-1]`. No transpose or resample is needed at 50x100. The manifest field `student_bev_shape_para_ssr: [100,100]` is stale.
- Coverage is 0–32 m forward, ±32 m lateral.

**Box format**
- Use `teacher_box_to_navsim` (`SSR/tools/rescore_teacher_detection.py:49-58`):
  - input is (x_fwd, y_left, z_bottom, dx = width, dy = length, dz, yaw);
  - output heading = −yaw − π/2, and z = z + dz/2.
- The manifest says z is the gravity centre. The producer stores `boxes.tensor` (`cache_teacher_future.py:260`), which is bottom-z in mmdet3d 0.x. These two conflict. z was never validated; x, y and yaw were.
- Velocity is `pred_boxes_3d[:, 7:9]` = (vx_fwd, vy_left) in m/s. D and E used it as-is. E §3.5 (navtrain, 2 m matching) measured: centre error 2.7/2.2 cm, yaw error 0.08 rad, velocity slope 1.00/0.97, precision about 99% train and 97.8% navtest, recall 85.7/83.1/80.4% (TRAIN-IN/OUT/navtest).
- There is no dense velocity map in the cache. Velocity exists only per proposal.

**Detection density** (300 tokens each)

| Set | boxes ≥ 0.3 (mean / median) | generic_object |
|---|---|---|
| TRAIN-IN | 18.1 / 14 | 7.9 |
| TRAIN-OUT | 9.0 / 5 | 2.7 |
| navtest | 10.3 / 7 | 2.9 |

The teacher saw both navtrain parts, so this is most likely scene/city composition, not teacher memorisation. It is still a train/test shift for R_T input density.

## (b) Student BEV (PARA-SSR interaction_final, seed 0)

**Config** (`work_dirs/para_ssr_interaction_final/code/hydra/config.yaml`)
- bev_h 50, bev_w 100, pc_range (−32, 0, −2, 32, 32, 2), camera-only, 3 encoder layers, 4 history frames.
- num_query 300, fut_mode 6, map_num_vec 100 × 20 points.
- Map classes: road, walkway, centerline, crosswalk.

**Where the BEV is produced**
- `para_ssr_model.py:354-366`: `bev_embed = pts_bbox_head(..., only_bev=True)` gives [B, 5000, 256].
- The model already returns it: `predictions["bev_embed"]` (`para_ssr_model.py:383-389`), so no hook is needed.

**Cell ordering**
- Flat index = row·100 + col (`bevformer.py:391-399` meshgrid "ij", H then W).
- Row = y_forward: (row+0.5)·0.64 m, so row 0 is the nearest.
- Col = x_right: −32 + (col+0.5)·0.64 m, so col 0 is 32 m to the left (`bevformer.py:423-428`).
- `dump_states.py:59-62` uses the same `CELLS` convention.

**Normalisation**
- The last encoder operation is LayerNorm `norms[2]` (`bevformer.py:311`). Its weight is 0.981±0.009 and bias ≈ 0 (read from last.ckpt), so each cell is about zero-mean with std ≈ 1 across channels.
- The planner applies one more LayerNorm before reading (`planner_head.py:68`, `bev_memory_norm`).
- This scale differs from the teacher's ReLU scale, so the refiners need identical input normalisation. Per-cell LayerNorm, as the planner does, is the obvious choice.

**Other outputs to dump in the same forward** (`model(features, run_aux=True)`)
- Detection: `all_bbox_preds[-1]` [B,300,10], decoded with `losses.denormalize_bbox` to (x_right, y_fwd, z, w, l, h, yaw_ssr, vx_right, vy_fwd); `all_cls_scores[-1]` [B,300,7].
- Motion: `traj_preds` [B,300,6,8,2] and `traj_cls_preds` [B,300,6].
- Map: `all_map_pts_preds[-1]` [B,100,20,2], normalised in [0,1]; `all_map_cls_scores[-1]` [B,100,4].
- Hidden states: `det_hidden`, `motion_hidden` and `map_point_hidden` are included because interaction sets `return_hidden=True`.
- `trajectory` [B,8,3].
- Det/map outputs appear in `predictions` only when `run_aux=True` (`para_ssr_model.py:392-396`).

**h_final**
- It is not returned. Either:
  - register a forward hook on `pts_bbox_head.final_norm` (`planner_head.py:350`) and take `out[:, 0]`; or
  - reuse `plan_capture` (`dump_states.py:102-159`), which was bit-exact: copy_err 0, 11,634/11,634 trajectories identical to the pkl.

**Reusable pieces**
- `infer_counterfactual.build_agent` and `TokenFeatures` (`report/collision_counterfactual/infer/infer_counterfactual.py:91-119`).
- The navtrain paths from `e1_dump.py` (trainval logs and sensor blobs). Sensor blobs are present for all 1,192 navtrain logs (446 GB).
- Kyungmin's `SSR-v2/tools/readout/cache_student_bev.py` already writes student BEV as sharded `bev/<shard>.npy` (256, 50, 100) f16 plus `index.json`/`meta.json`, readable via memmap with `readout/bev_cache.BevCache`. It builds its agent from kyungmin's own agent yaml, so for our checkpoint swap in our `build_agent`.

**Measured throughput** (single GPU, 2 loader workers, peak 2.6 GiB, loading-bound)
- E dump (`run_aux`): 11.7 tok/s; the ep2 run reached 17.0 tok/s.
- `dump_states` on navtest: 10.9 tok/s.

**Storage per token**

| Item | Size |
|---|---|
| BEV, f16 | 2.56 MB |
| Det + motion + map + trajectory + h_final | about 0.1 MB |
| Optional pooled det/map memories (300×256 + 100×256 f16) | about 0.2 MB |
| **Total** | **2.7–2.9 MB** (compression will likely help little, since the BEV is dense after LayerNorm; not measured) |

**Estimates**

| Set | Storage | Time at 11–12 tok/s on one GPU |
|---|---|---|
| 20k tokens | about 54–58 GB | about 30 min |
| 30k tokens | about 81–87 GB | about 45 min |
| navtest (12,146) | about 33–35 GB | about 18 min |
| full navtrain (103k) | about 280–300 GB | about 2.5 h |

- Free disk: `/` has 736 GB, `/home/external-user/ssd` has 1.6 TB.
- All 6 GPUs are currently in use (8–15 GB each).

## (c) GT rasters and GT scope

**Existing code**

1. **Map raster, navtest only** (kyungmin): `SSR/data/readout/build_map_rasters.py:23-63` wrote `map_rasters_navtest.npz`, rasters (12146, 4, 50, 100) uint8 on the student grid.
   - Classes: road (LANE+INTERSECTION), walkway, centerline (1-pixel lines), crosswalk.
   - It uses `fillPoly`, which adds a half-cell outward bias.
   - Its "road" is not the DAC drivable area: no ROADBLOCK or CARPARK.
2. **DAC-matched drivable polygon and SDF** (kyungmin `SSR-v2/navsim/agents/para_ssr/plan_map.py:98-120`): uses ROADBLOCK plus interior lanes, ROADBLOCK_CONNECTOR, INTERSECTION and CARPARK_AREA, the same layers as `PDMDrivableMap` (`pdm_occupancy_map.py:141-152`).
   - `rasterize_sdf` (lines 52-77) and `feasibility.rasterize_bev_sdf` (lines 89-116) use a centre-inside rule on the student grid; the feasibility target is 4× finer, 0.16 m.
   - Kyungmin's docstring (lines 38-47, not verified by me) says bilinear SDF error is 0.142 m at 0.64 m cells, 0.035 m at 0.16 m.
   - `sample_field` (line 239) clamps off-grid points to the border value.
3. **Future object SDF** (kyungmin `feasibility.agent_free_sdf`, lines 119-168): one plane per step for steps 1..8, built from all annotation boxes of the future frames. It is not ROI-limited, includes objects entering later, and uses `fillConvexPoly`.
4. **NAVSIM's own raster** (`navsim/agents/transfuser/transfuser_features.py:197-300` and `transfuser_config.py:83-98`): road, walkway, centerline, static objects, vehicle, pedestrian at 0.25 m.
5. **Vector map targets** in our repo: `para_ssr_targets._compute_map_targets`, road = contour of LANE∪INTERSECTION.
6. **Metric-cache objects**: `sf_common.gt_objects` (`report/planner_vs_perception_tests/safety_filter/sf_common.py:408-431`) gives all objects at 41 steps × 0.1 s, red-light pseudo objects dropped. 32-B turned these into circle chains in `b0_gt.py`: kept objects within 40 m, A_MAX = 64 (`common.py:38`; the b0 docstring says 48).

**Scope of `_compute_agent_targets`** (`para_ssr_targets.py:289-351`) — confirmed:
- Current frame only.
- ROI x_right ∈ [−32, 32], y_fwd ∈ [0, 32], and bearing ≤ 80° (`detection_box_in_roi`, lines 128-147).
- Only the 7 classes, nearest 100 (`max_agents`).
- `_track_future` (lines 401-449) stops at the first frame where the track is missing and never picks it up again.
- So objects entering later, objects behind, and objects beyond 32 m are absent. A zero mask there means "unobserved", not "free".

**How the official scorer sees GT**
- `metric_cache_processor.py:115-238`: every tracked object in the log annotations (full 360°, up to about 80 m), 2 Hz interpolated to 10 Hz, 5 s horizon. Objects that first appear later are added from that time. A track seen in only one frame is held static for the whole horizon.
- NC (`pdm_scorer.py:323-379`): LQR-tracked ego footprint at 0.1 s steps. At-fault only for stopped-track, active-front, or lateral collisions while ego is in multiple lanes or non-drivable area. Agent → 0, static → 0.5.
- TTC (lines 490-545) is separate.
- The log annotations (`anns`) are the same source as the metric cache (`NavSimScenario`).

**Measured: where the collided objects were** (navtest, 279 interaction_final NC failures, from the metric cache)
- 231 (83%) were inside the student detection ROI at t = 0.
- 29 were ahead beyond 32 m at t = 0.
- 8 were behind.
- 11 were not present at t = 0 and appeared at 0.5–4.0 s.
- So 48/279 (17%) would be missed by a current-ROI object set.
- Object types: vehicle 230, cone 18, generic 14, barrier 8, pedestrian 6, bicycle 3. Collision types: stopped-track 153, active-front 121, lateral 5.

**Measured: forward coverage of the 0–32 m grid** (navtest)
- 16.4% of human and 16.5% of student trajectories end beyond x = 32 m at 4 s.
- About 24% end beyond 28 m, so the front bumper leaves the grid.
- p99 endpoint is about 51 m.
- Every BEV and raster input (teacher, student, GT on the 50x100 grid) misses the far part of about a sixth to a quarter of trajectories. Surrogate losses should use vector objects or a larger grid there, not the input grid.

**Measured: the 32-B circle surrogate vs official NC** (`b1_train.collision_penalty` — 3 ego circles r = 1.436 m, margin 0.3, 0.5 s steps, "not behind" mask — on the 12,146 navtest E0 drafts)

| Variant | Positives | Recall on NC failures | False positives on NC passes | Precision (NC) | Precision (NC or TTC) |
|---|---|---|---|---|---|
| Penalty > 0 | 1,412 | 97.1% | 1,142 (9.6%) | 19% | 31% |
| Human-violated pairs excluded | 976 | 93.2% | 717 (6.0%) | 26.5% | 41% |

- The human trajectories themselves trigger the penalty in 7.6% of scenes.
- The surrogate over-covers laterally: 1.436 m radius vs 1.1485 m half-width, plus 0.3 m margin.
- It has no lane or at-fault logic.
- At 0.5 s spacing a 10 m/s ego moves 5 m between checks, which is more than the 2.9 m circle diameter.
- It is usable as a training surrogate but must not be read as a stand-in for NC.

**Proposed raster channels** (student grid (50, 100), student layout)

*Current-GT raster, t = 0 only.* Crop to the same 0–32 m front ROI so it compares fairly with R_T and R_S:
1. Footprint occupancy by group: vehicle; VRU (pedestrian and bicycle); static (cone, barrier, czone, generic). Three channels.
2. Velocity (vx_fwd, vy_left)/10 in occupied cells. Two channels.
3. Heading (cos, sin) in occupied cells. Two channels.
4. DAC-layer drivable mask and a clipped SDF (±8 m, /8), using `plan_map.drivable_polygon`. Two channels.
5. Lane centerline. One channel.
6. Optionally the route centerline (`mc.centerline`), walkway and crosswalk. Three channels.

That is about 11–14 channels, about 70 KB/token as uint8/f16.

A "box raster" in the same format can be made from teacher boxes (score ≥ 0.3) and from student detections. That would let R_GT(current), R_T(boxes) and R_S(boxes) be compared on identical channels alongside the feature-based R_T/R_S.

*Future-GT raster (oracle, a separate reference arm).*
- Occupancy at k = 1..8 (0.5 s) of every annotated object at its actual future pose in the current ego frame. Split agent vs static, because NC scores them 0 vs 0.5: 16 channels.
- Optionally the SDF form (`agent_free_sdf`).
- Same drivable channels as the current raster.
- Consider an extended forward grid for this arm, because of the 32 m coverage limit above.

## (d) Real student drafts

**navtest (main evaluation)**
- Trajectories: `SSR/work_dirs/eval/para_ssr_interaction_final_navtest_trajectories.pkl`, 12,146 × (8, 3). Meta: checkpoint epoch 29, NAVSIM ego frame.
- Official scores: `work_dirs/eval/para_ssr_interaction_final/2026.09.17.00.09.41.csv`
  - PDMS 0.8487
  - NC < 1: 278 (36 of them are 0.5); the E shards count 279
  - DAC < 1: 813
  - TTC < 1: 782
  - any of NC/DAC/TTC: 1,524
  - EP 0.798
- Metric cache: `SSR/data/exp/metric_cache` (3.1 GB).
- Human futures: `report/head_ablation_scenes/table.npz` (`human`, all valid).
- Detections and motion: `report/planner_vs_perception_tests/dump/npz` (score ≥ 0.1, record frame) and `all_dets.npz`.
- Map outputs: `report/cause_and_correction_tests/A_lane_departure/dump_map/` (shards plus `student_map_navtest_ego.npz`).
- h_final and planner states: `report/problem_funnel/planner_probe/dump/shard_*.npz` (24 shards, 4.0 GB).
- There is no full navtest student BEV anywhere. Only a 5×10-pooled `bev_grid` exists.

**navtrain (experiment E, `report/cause_and_correction_tests/E_train_split_feasibility/`)**
- Tokens: `tokens/sample.parquet`, 9,000 = 6,000 TRAIN-IN (`part='train'`, 851 logs) + 3,000 TRAIN-OUT (`part='val'`, 212 logs), seed 0. The full split table is `tokens/navtrain_token_log.parquet` (85,109 / 18,179).
- Dumps, one `.npz` per token:

| Directory | Checkpoint | Tokens |
|---|---|---|
| `dump/npz` | last.ckpt (epoch 29) | 9,000 |
| `dump_ep2/npz` | `version_1/.../epoch=2-step=1995` | 6,000 TRAIN-IN |
| `dump_ep9/npz` | `version_2/.../epoch=9-step=6650` | 6,000 TRAIN-IN |
| `dump_ep19/npz` | `version_2/.../epoch=19-step=13300` | 6,000 TRAIN-IN |
| `dump_gm/npz` | last.ckpt, GridMask on | 6,000 TRAIN-IN |

- Dump fields (`e1_dump.py:185-192`): token, ego_traj, q_idx, box, score, label, cls_prob, motion_abs, motion_logit, motion_prob, human_traj, ego_vel, ego_acc, cmd. **No BEV and no h_final**, so R_S on navtrain needs a new dump.
- Official scores are per token in `navtrain_shards*/` (`rows`) and `token_table.parquet` (9000 × 180). The navtrain metric cache is `metric_cache/` (9,000 tokens, 2.9 GB). Building more ran at 2.28 tok/s with 4 CPU workers, about 0.33 MB/token.
- NC failure rate:

| Checkpoint | TRAIN-IN | TRAIN-OUT |
|---|---|---|
| epoch 29 (final) | 0.70% (42/6000) | 1.50% (45/3000) |
| epoch 19 | 0.80% | — |
| epoch 9 | 1.62% | — |
| epoch 2 | 3.57% | — |

- Student perception recall: 76.8% TRAIN-IN, 62.0% TRAIN-OUT, 63.9% navtest. Student features on TRAIN-IN are therefore better than at test time. TRAIN-OUT (18,179 tokens) is the pool that matches navtest; the teacher cache covers it too.

## Extra measured facts for feedback item 1 (drafts)

**How far real drafts are from the human trajectory** (navtest, 4 s arc length, student/human, moving scenes only; 3.5% of scenes have human travel under 2 m)
- Ratio percentiles: p5 0.82, p25 0.95, p50 1.00, p75 1.05, p95 1.27.
- Ratio < 0.6: 0.5%. Ratio > 1.4: 2.7%.
- Final lateral offset |·|: p50 0.20 m, p90 0.89 m, p99 2.44 m.
- The 259 NC failures in moving scenes: ratio median 1.16 (p75 1.30), lateral median 0.46 m. The student is mostly ahead of the human.
- E observed the reverse on TRAIN-IN: same speed as the human with about 0.6 m lateral offset.

**Path beyond the 4 s GT**
- The log pickles are 2 Hz full logs, e.g. 168 frames, so the ego path continues past the 10-future-frame scene window.
- Tokens with at least this much logged future:

| Future available | navtest | navtrain (sample of 120 logs, 10,021 tokens) |
|---|---|---|
| ≥ 5.5 s | 99.8% | 99.8% |
| ≥ 6.0 s | 99.6% | 99.6% |
| ≥ 7.0 s | 99.1% | 99.2% |
| ≥ 8.0 s | 98.7% | 98.7% |

- A ×1.4 time-resample needs the path to 5.6 s, which is available for about 99.6% of tokens.
- Only path geometry is needed from the log; objects are still needed only up to 4 s. The metric cache covers objects to 5 s.

## Not investigated

- byounggun Stage 1/2.
- Kyungmin's feasibility run results.

For context on kyungmin's readout/KD (v1): it used a ReSMap *map* teacher and a plan-only student. Its report 22 says the probe IoU gap on the BEV did not transfer through distillation, and the hypothesis was rejected.