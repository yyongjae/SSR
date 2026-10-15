# Stage E implementation plan (E1 / E2 on PARA-SSR), written 2026-09-29 16:30 KST

Spec: `PRESTATED_DECISION_RULE.txt` "STAGE E PLAN" (16:20), report 36 §2 M10, report 37 §8.
Deadline: E1/E2 training ends by 2026-09-30 18:00 KST, then navtest PDMS.

## 0. E0, E1 and E2

All three arms train PARA-SSR from scratch on the same data (navtrain train_logs, 85,109 tokens), with seed 0 and the same effective batch of 128.

| arm | what is trained | question it answers |
|---|---|---|
| **E0** | Existing PARA-SSR (`para_ssr_interaction_final`, 30 epochs). The checkpoint at epoch N is used as a same-budget reference. | Baseline |
| **E1** | PARA-SSR + **student refiner** (a small corrector that fixes the planner's draft tau0). The student refiner uses the stage-T refiner architecture and reads the student's own BEV. It is trained only with the **GT surrogate penalties** (collision, TTC, drivable area, progress, comfort, modification) on tau_final. | What the refiner and GT penalties alone contribute |
| **E2** | E1 + **correction KD** from two frozen teacher refiners: R_T on the BEVFusion BEV and R_M on the ReSMap BEV. On the same draft, an L1 loss pulls the student's raw correction controls (z_lon, w_lat) towards the teachers' controls. | **What the aux-teacher KD adds beyond the GT penalties (the paper's main claim)** |

- **Primary comparison:** navtest PDMS of E2 minus E1. We also compare NC, TTC and DAC, with a log-cluster paired bootstrap CI.
- **Output at inference:** the model outputs tau_final (the refined trajectory). tau0 is also scored with a flag.
- **Gradient paths:**
  - tau0 is stop-grad. The planner learns only from its existing imitation loss.
  - The surrogate and KD gradients reach only the student refiner and, scaled by 0.1, the BEV encoder.

## 1. Where things live in PARA-SSR (verified, not assumed)

**Student BEV.** `ParaSSRModel.forward` returns `predictions["bev_embed"]` with shape `[B, 5000, 256]` (batch-first). It is the output of the current-frame `pts_bbox_head(..., only_bev=True)`. It is the only BEV with a graph; history BEVs are built under no_grad. Query index = `row * 100 + col`, from `BEVFormerEncoder.get_reference_points`: `ys` spans the H = 50 rows, `xs` spans the W = 100 columns, flattened row-major. `ref_x` maps to `x_right = -32 + (c + .5) * .64` and `ref_y` maps to `y_forward = (r + .5) * .64`.

`bev = bev_embed.transpose(1, 2).reshape(B, 256, 50, 100)` gives:
- row r: x_forward = (r + .5) * 0.64 m, from 0 to 32 m;
- col c: y_left = -x_right = 32 - (c + .5) * 0.64 m.

This is **exactly the stage-T S grid** (`adapters.py`: row = x forward, col 0 = +32 m left). **No flip and no transpose are needed.** It also matches the config comment: the teacher cache (y_left ascending) needs a lateral flip to reach both grids.

The grid was **verified empirically** with `build_lidar2img` on a trainval log. BEV cells were projected at z = 0 into the three cameras:

| cell (row, col) | position | camera it projects into |
|---|---|---|
| (25, 0) | x_right = -31.7 m | only CAM_L0 |
| (25, 99) | x_right = +31.7 m | only CAM_R0 |
| (49, 50) | y_fwd = 31.7 m | only CAM_F0 |
| (10, 20) | left side | CAM_L0 |
| (10, 80) | right side | CAM_R0 |

The implementation keeps this check as a unit test.

**tau0.** `predictions["trajectory"] = select_trajectory(ego_fut_preds, command)` has shape `[B, 8, 3]`. It is the cumsum of per-step offsets, in the NAVSIM ego frame (x forward, y left, heading), at t = 0.5 to 4.0 s. This is the refiner's N frame, so no conversion is needed. The refiner gets `tau0.detach()[:, None]` (T = B, K = 1).

**Ego state and command.**
- `features["status_feature"]` = `[cmd one-hot(4), vx, vy, ax, ay]`.
- `eds = status[:, 4:8]`, `v0 = hypot(vx, vy)`, `a0 = ax`.
- `cmd = features["command"].argmax(-1)`. The order is left, straight, right, unknown, which is the order RefinerNet expects.

These are the same definitions `extract_human.py` used for stage T (`v0 = |(vx, vy)|`, `a0 = eds[2]`).

**GT future for the surrogate's human-overlap mask.** This is `targets["trajectory"]` `[B, 8, 3]`, the same N frame as `human_traj`.

**Token.** The training Dataset has no feature cache (`cache_path: ''`). `ParaSSRTargetBuilder.compute_targets(scene)` sees `scene.scene_metadata.initial_token`, so the per-token GT and teacher inputs are loaded there, inside the dataloader workers. `Dataset` and `run_training.py` are not changed.

## 2. Student refiner module

New file `navsim/agents/para_ssr/refiner/e2e.py`. **No stage-T file (`refiner/*.py`, `tools/refiner/*.py`) is edited.** Their sha256 values are recorded in the stage-T `config.json` files, and eval_refiner still runs on them after 19:00.

- **`AdapterS`** (arm 'S'). Input `bev_embed [B, 5000, 256]`:
  1. `GradScale(0.1)`: forward is the identity, backward multiplies by 0.1.
  2. Reshape to `[B, 256, 50, 100]`.
  3. `LayerNorm(256)` per cell over channels, with affine parameters.
  4. 1x1 conv 256→128, GELU, 1x1 conv 128→64.

  This is the AdapterT form with ChannelZScore replaced by LayerNorm: 41,152 + 512 parameters. `needs_bev = True`.
- **Why LayerNorm and not z-score or BN:**
  - A frozen z-score needs fixed statistics of a feature that is trained from scratch and drifts every step.
  - BatchNorm running statistics lag that drift. Train (batch stats) and eval (running stats) would then differ exactly where tau_final is scored. At batch 4 per GPU, the statistics are also noisy between micro-batches.
  - Per-cell LayerNorm has no state and is identical in train and eval.
  - It loses nothing. `bev_embed` is already a per-cell LayerNorm output: BEVFormer layers are post-norm (`self_attn, norm, cross_attn, norm, ffn, norm`). The stage-T objection to per-cell LayerNorm (critic_logic issue 5: it erases per-cell magnitude of a ReLU teacher feature) therefore does not apply.
- **`build_student_refiner(seed=0)`:**
  - Builds `RefinerNet(arm="none", seed=0)`, whose trunk is initialised under `fork_rng`, identical to the stage-T trunks of seed 0.
  - Then sets `net.adapter = AdapterS(...)`, built inside `torch.random.fork_rng(devices=[])` with `manual_seed(seed + ADAPTER_SEED_OFFSET)`.
  - RefinerNet's forward only calls `self.adapter(bev, n_tokens=T)`, so it works unchanged.
  - Because of fork_rng, building the refiner consumes **no global RNG**. E0, E1 and E2 therefore get the same PARA-SSR initialisation, data order and GridMask draws. The CPU smoke asserts this.
- **Registration:**
  - `ParaSSRAgent.student_refiner` is created only when `refiner_mode != "off"`, after `ParaSSRModel` and the loss.
  - It is saved in the Lightning state_dict, so strict loading at eval works when the same flag is set.
  - The gate head exists but gets no loss. SPEC: no gate training, theta = 0, the correction is always applied.
- **Optimiser:**
  - Extra param group `{params: student_refiner, lr_scale: ref_lr_mult = 3.0, weight_decay: 0.01}`. This is stage T's 3e-4 peak and weight decay under the same WarmupCosLR shape.
  - The group is appended only when enabled, so E0's groups are unchanged.
- **Clipping:**
  - Refiner-only gradient norm clip of 1.0 (stage T), in a callback `on_before_optimizer_step`. This runs before Lightning's global clip.
  - Lightning's global clip of 35 then sees refiner gradients ≤ 1. The surrogate's O(1e2-1e3) dL/dz spikes (validate_decoder.py) therefore cannot shrink PARA-SSR's gradients through the joint clip.
- **Decoding:**
  - Training uses `decoder.decode(tau0, z, w, v0, mode="A", lon_st_slope=0.1)` (run-4 setting).
  - Inference uses the same call without the slope (the forward pass is bit-identical).
  - `tau_final = dec["traj"]`.

## 3. Per-sample GT for the surrogate

**Loaded in `ParaSSRTargetBuilder.compute_targets` when `refiner_mode != off`** through `e2e.load_side_targets(token, log)`. These fields are padded to fixed shapes so the default collate works:

| key | shape / dtype | source |
|---|---|---|
| `ref_obj_kf, ref_obj_first, ref_obj_meta` | [800,11,6] f32, [800,6] f32, [800,5] i64 | `objects/{train,navtrain,dev}/<tok>.npz` (disjoint; 19,732 + 58,743 + 6,634 = 85,109 — fixed after review, dev was missing) (`gt_future.load_objects`, `data._read_objects` padding, A_MAX 800) |
| `ref_obj_n, ref_obj_n_kf, ref_obj_R, ref_obj_ego_kf` | scalars, [11,3] | same |
| `ref_sdf` | [320,256] f16 | `sdf/navtrain/<tok>.npz` (`sdf.load_sdf`) |
| `ref_cl_xy, ref_cl_valid, ref_cl_n` | [1536,2] f32, [1536] bool | side store (below) |
| `ref_p_pdm` | f32 | side store (below) |
| `ref_gt_ok` | bool | all four present |

- **Side store (new `tools/refiner/build_e2e_side.py`, CPU, 4 workers, follow mode):**
  - Loads each token's `metric_cache.pkl` once. Loading the lzma pickle takes 0.05-0.7 s, too slow for the dataloader.
  - Writes `/home/external-user/ssd/yongjae_refiner/e2e_side/<tok[:2]>/<tok>.npz` with:
    - `cl_xy`, `cl_n` = `data.centerline_samples(mc)`;
    - `p_pdm = score_trajectories.score_token(mc, traj)[0]["pdm_progress_eff"]`.
  - p_pdm is PDM-Closed's own `raw_eff[0]`, which does not depend on the submitted trajectory (row 0 of the batched call). The human trajectory is submitted.
  - Check: on 50 train tokens, equality with `scores/train.parquet` pdm_progress_eff (float ==).
  - Cost ≈ 85k tokens at ~25-40 tok/s ≈ 40-60 min after the metric caches exist.
- **Missing GT:**
  - When `ref_gt_ok = False`, the whole surrogate is masked for that sample. The surrogate is computed **only on the ok subset** (index_select, not multiply-by-0). A masked sample with zero-filled SDF or centerline can produce NaN intermediates, and 0 × NaN poisons the backward pass.
  - The micro-batch surrogate is the mean over ok samples, or 0 when none are ok.
  - Counts are logged per step, with an epoch sum: `ref/n_gt_ok`, `ref/n_gt_missing`.
  - KD needs no GT and uses every sample.
- **Coverage now (16:10):**
  - Objects: done by about 16:20.
  - Metric cache: 34k of 85k; the two parts ETA about 18:15.
  - SDF: follows the metric cache (3 workers, ~4.4 tok/s), ETA 20:00-22:00. Raise to 4-6 workers once the metric-cache build frees CPUs.
  - Side store: about 19:30.
  - Tokens still missing during epoch 0 are masked and counted. E1 and E2 are launched within minutes of each other, so they see the same availability timeline.

## 4. Teacher inputs for KD (E2 only)

- **Teacher BEVs**, loaded in the same target builder:
  - `ref_bev_T = TeacherCache.for_subset("navtrain").load_bev(tok, s_grid=True)`: f16 [256,50,100], 3.8 ms.
  - `ref_bev_M = ResmapCache.for_subset("navtrain").load_bev(tok, s_grid=True)`: f16 [256,50,100], 2.2 ms (memmap reopened per worker pid).
  - Coverage checked now: **85,109 / 85,109 tokens in both caches**.
- **Teacher refiners:**
  - `load_run_model(run_dir, "best")` from `tools/refiner/train_refiner.py`, imported with importlib, not copied. It rebuilds RefinerNet from `config.json` plus `norm.npz` or `norm_map.npz`.
  - The run dir is a **read-only snapshot copy** made after the stage-T runs finish (~19:00): `ckpt_best.pt`, `norm*.npz` and `config.json` copied to `/home/external-user/ssd/yongjae_refiner/stageE/teachers/{T,M}/`, with sha256 recorded. The stage-T run directories are never written.
  - Built lazily on the first training `compute_loss`. They are held outside the module tree (`object.__setattr__(self, "_teachers", [...])`), so they are not in the state_dict, the optimiser or eval.
  - `.eval()`, `requires_grad_(False)`, fp32, `torch.no_grad()`, moved to the student's device.
- **Teacher forward:** `R(bev_teacher.float(), draft, v0, a0, eds, cmd)`, with the **same draft** as the student (perturbed or not), all detached.
- **Go/no-go before E2 launch** (SPEC: liveness gate ≥ 1 %): `tools/refiner/liveness.py --run <run> --eval eval_train_fold0 --min-liveness 0.01` for both teacher runs.
  - If one fails, the fallback is a user decision. Options: only the passing teacher, or run 3 R_T.

## 5. The 50 % perturbed-sg(tau0) path

- **Per micro-batch RNG:** `rng = np.random.default_rng([ref_seed, self._loss.iteration])`, using the micro-batch counter before the loss call.
- **Per sample:**
  1. `perturb = rng.random() < 0.5`.
  2. If perturbing, sample a family uniformly from `decoder.BANK_LAYOUT[1:]` (the bank's 12 non-identity slots, so family frequencies match the stage-T bank).
  3. Apply the `sample_bank` fallbacks: ignore_brake or creep → lconst; lat or combined → lconst; cv invalid → hdrift.
  4. `decoder.sample_perturbation(f, HumanContext(tau_h=sg(tau0), path_long=ext, v0, a0, None), rng, mode="A")`.
- **Path extension (deviation from stage T):** stage T followed the logged human path up to 8 s. A student draft has no logged path, so `ext` = tau0 + 8 straight poses at the draft's end speed and heading.
  - Measured on 300 train trajectories (CPU): **invalid 10 %** with the extension versus **46 %** without it (lconst `path_too_short`).
  - **5.8 ms per sample** (p95 11 ms), so about 12 ms per micro-batch.
- **Invalid draft:** the unperturbed tau0 is used and the event is counted (`ref/n_perturb_invalid`).
- **Seed and resume:** `self._loss.iteration` is checkpointed through `get_extra_state`, so a resumed run continues the same stream. Only numpy's Generator is used; torch's global RNG is not touched.

## 6. Losses

Let `L0` be the existing `ParaSSRLoss` total, including the GradBalancer correction. It is computed exactly as now, first.

- **E1:** `L = L0 + w_ref * L_sur`, with `w_ref = 1.0`.
  - `L_sur = Σ_n w_n · mean_ok(term_n(tau_final))`, with terms and weights from SPEC: col 1, ttc 1, dac 1, prog 2, cmf 0.1, mod 0.1.
  - `SurrogateConfig(m_col=.15, m_dac=.05, m_ttc=.15)`, mode A, `lon_st_slope = .1`.
  - Terms come from `surrogate.surrogate_terms` on `train_refiner.scene_from_batch(...)`. The batch dict is built from the `ref_*` targets, with objects trimmed to the batch's max `obj_n` as `collate_tokens` does, `human_traj = targets["trajectory"]`, `pdm_progress_eff = ref_p_pdm`, and `tau0` = the draft.
  - `nan_to_num` is applied as in `train_refiner.compute_loss`. A non-finite `L_sur` is replaced by 0 and counted.
- **E2:** `L = E1 + λ(t) · L_KD`, with `L_KD = mean_{teachers T, M} mean_{B samples, 12 controls} |[z_lon, w_lat]_student − [z_lon, w_lat]_teacher|`. These are raw pre-tanh controls of the same draft (report 36 M10: control space, before clipping).
  - **Review correction (09-29):** raw z_lon of the teachers is saturated (+7 to +13; tanh' ≈ 1e-7), so raw-space KD would pull the student's z_lon into the dead zone and also kill the surrogate's longitudinal gradient. The code now has `kd_space` = `raw` (this plan) | `tanh` (tanh of both) | `decoded` (c_lon[2:] in m/s after the mode-A clamp, e_lat[2:] in m). It has **no default**: E2 training and the launch script refuse to run until the user picks one. λ_c must be measured in the chosen space.
- **λ schedule (SPEC):** `λ(t) = λ_c · clip((e − 1) / 2, 0, 1)`, where e = fractional epoch. So λ = 0 in epoch 0, ramps linearly through epochs 1-2, and is constant from epoch 3.
  - e is set by the callback at `on_train_batch_start`: `current_epoch + batch_idx / num_training_batches`.
  - `λ_c = mean(w_ref · L_sur) / mean(L_KD)` over the last 100 micro-batches of the E2 GPU pilot, where λ = 0 because the pilot is in epoch 0. The value is written into PRESTATED_DECISION_RULE before launch and passed as `kd_lambda`.
- **Gradient paths (M10 defaults):**
  - tau0 is detached before the refiner, decoder, surrogate and teachers.
  - Teacher inputs are detached.
  - The student BEV path is scaled by 0.1 through GradScale.
  - Refiner losses are added **after** `balance_shared_gradients`, so the balancer's plan/det/map shares and scales are unchanged. The refiner gradient is logged, not balanced.

## 7. Logging

Everything goes through `agent.latest_logs`; the existing `ParaSSRLoggingCallback` logs each key per step and per epoch.

- **Surrogate terms:**
  - `ref/t_{col,ttc,dac,prog,cmf,mod}`: unweighted means over ok samples.
  - `ref/loss_sur`: weighted.
  - `ref/P1_minus_P0`.
- **GT and perturbation counts:** `ref/n_gt_ok`, `ref/n_gt_missing`, `ref/n_perturbed`, `ref/n_perturb_invalid`, `ref/nonfinite`.
- **Correction size and liveness:**
  - `ref/live`: fraction of drafts with any decoded `c_lon < 0`.
  - `ref/short_m`: arc(draft) − arc(tau_final).
  - `ref/abs_dlat_end`.
  - `ref/zdead`.
  - `ref/final_vs_tau0_l2`: unperturbed samples only.
- **KD:** `kd/l1_T`, `kd/l1_M`, `kd/loss`, `kd/lambda`, `kd/teacher_live_{T,M}`.
- **Refiner gradients:**
  - `gnorm/ref_bev`: ‖∂(w_ref L_sur + λ L_KD)/∂bev_embed‖ × 0.1, next to the existing `gnorm/plan|det|map`, every `grad_norm_log_interval`.
  - `ref/grad_norm_preclip`: in the clip callback.
- **Timing:** `time/perturb_ms`, `time/ref_ms`.
- **Snapshot:** `stageE_config.json` in the output dir, with flags, teacher sha256, `λ_c`, N and code sha256.

## 8. Evaluation

The standard navtest evaluation is used unchanged: `scripts/evaluation/eval_para_ssr.sh <ckpt>`, which runs `run_pdm_score_gpu.py` with the navtest metric cache. It takes about 30 min per run (E0: 23:39 → 00:09, PDMS 0.8487).

- **Inference path:** when `refiner_mode != off` and `not self.training`, `agent.forward`:
  1. runs the student refiner on the unperturbed tau0;
  2. decodes in mode A without the straight-through slope;
  3. sets `predictions["trajectory"] = tau_final` (or `tau0` if `ref_eval_traj=tau0`);
  4. also returns `trajectory_tau0` and `trajectory_final`.
- **Teachers** are never built at eval.
- **Runs (5 × 30 min):** each eval is a `$CKPT` plus hydra overrides.
  - `EVAL_EXPERIMENT=eval/stageE_E1_final`: E1 ckpt, `agent.config.refiner_mode=E1`.
  - `EVAL_EXPERIMENT=eval/stageE_E1_tau0`: same, plus `agent.config.ref_eval_traj=tau0`.
  - `EVAL_EXPERIMENT=eval/stageE_E2_final` and `eval/stageE_E2_tau0`: the same two for E2.
  - `EVAL_EXPERIMENT=eval/stageE_E0_ep{N}`: `work_dirs/para_ssr_interaction_final/lightning_logs/version_*/checkpoints/epoch={N-1}-*.ckpt`. All E0 epoch checkpoints exist.
- **Comparison:** new `tools/refiner/stageE_compare.py` (~60 lines). It joins the per-token CSVs and reports:
  - PDMS, NC, DAC, TTC, EP and comfort per arm;
  - E2−E1 (primary) and final−tau0 per arm;
  - log-cluster paired bootstrap (10k, logs from `splits/navtest.parquet`).

## 9. Config and CLI

**New `ParaSSRConfig` fields.** They are added to `para_ssr_agent.yaml` with **off** defaults, so no `+` prefix is needed:

```
refiner_mode: off            # off (E0) | E1 | E2
ref_bev_grad_scale: 0.1
ref_w: 1.0
ref_term_weights: {col: 1.0, ttc: 1.0, dac: 1.0, prog: 2.0, cmf: 0.1, mod: 0.1}
ref_m_col: 0.15
ref_m_dac: 0.05
ref_m_ttc: 0.15
ref_lon_st_slope: 0.1
ref_perturb_frac: 0.5
ref_seed: 0
ref_lr_mult: 3.0
ref_weight_decay: 0.01
ref_clip: 1.0
ref_data_root: /home/external-user/ssd/yongjae_refiner
kd_teacher_runs: [<stageE/teachers/T>, <stageE/teachers/M>]
kd_lambda: 0.0
kd_ramp: [1, 3]
ref_eval_traj: final         # final | tau0
```

- **When off:** no module, no param group, no callback clip, no extra targets, and `get_unique_name` is unchanged. The E0 path is the current code.
- **Parity test (HARD RULE):** `tools/refiner/stageE_parity.py`.
  - **Dump the golden before any edit:** default config, `backbone_pretrained=False`, `seed_everything(0)`, two real navtrain scenes through the real feature and target builders, CPU, train mode, `manual_seed(0)` before forward. Save the loss, every `latest_logs` value and every parameter gradient.
  - After the edits, `refiner_mode=off` must give `torch.equal` on everything.
  - Also: E1 and E2 construction leaves `torch.get_rng_state()` equal to E0's.

**1-GPU run with the same effective batch** (4 × 32 = 128 = E0's 4 × 2 × 16, 665 optimiser steps per epoch):

```
cd /home/external-user/yongjae/SSR
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=4 ACCUMULATE=32 MAX_EPOCHS=$N WORKERS=6 EXPERIMENT=stageE_E1_N$N \
bash scripts/training/train_para_ssr_interaction.sh \
  agent.config.refiner_mode=E1 agent.config.warmup_epochs=$W \
  agent.config.grad_balance_warmup_iters=21278 agent.config.grad_balance_interval=400 \
  agent.config.grad_norm_log_interval=400 \
  trainer.params.strategy=auto trainer.params.limit_val_batches=0
# E2: CUDA_VISIBLE_DEVICES=1, EXPERIMENT=stageE_E2_N$N, agent.config.refiner_mode=E2,
#     agent.config.kd_lambda=<lambda_c>, agent.config.kd_teacher_runs=[...]
```

- **GradBalancer counters** count micro-batches per process. On 1 GPU an epoch has 21,278 micro-batches instead of 10,639. Doubling `warmup_iters` and `interval` keeps E0's controller cadence (1 epoch warm-up, every 12.5 optimiser steps).
- **`strategy=auto`:** single device, no DDP. The unused gate-head parameters would break DDP's reducer.
- **Validation off:** it produces loss only, costs about 10 min, and its tokens have no GT stores.
- **LR schedule** spans N epochs: `agent.config.max_epochs = trainer max_epochs = N`, set by the script.
- **N (SPEC rule):** from the pilot, `t_ep = max(E1, E2 s/micro-batch) × 21,278`, and N = the largest integer with `launch + N · t_ep ≤ 09-30 18:00`.
- **Values to fix in PRESTATED_DECISION_RULE before launch (user decision):**
  - **W (warm-up epochs):**
    - **Correction (review 09-29):** `WarmupCosLR` is stepped once per epoch, so W = 1 is **no warm-up** (epoch 0 already at the peak 1e-4, refiner 3e-4). It does not keep a 10 % fraction.
    - W = 2: epoch 0 at 0.5 lr (5e-5), peak in epoch 1. With N = 6 the lr per epoch is [5e-5, 1e-4, 1e-4, 8.55e-5, 5.05e-5, 1.55e-5] (W = 1: [1e-4, 1e-4, 9.05e-5, 6.58e-5, 3.52e-5, 1.05e-5]; W = 3: [3.3e-5, 6.7e-5, 1e-4, 1e-4, 7.5e-5, 2.6e-5]).
    - W = 3: E0's absolute warm-up (3.3e-5, 6.7e-5, 1e-4), but half the budget at N ≈ 6.
    - The pilot must use the same W (it runs one 300-micro-batch epoch at the run's epoch-0 lr).
  - **`w_ref = 1`:** SPEC does not fix it. The stage-T term weights already scale the terms.
  - **`ref_lr_mult = 3`:** gives the stage-T 3e-4 peak.

## 10. Risks (estimates)

1. **Wall clock.**
   - E0 ran at 4,500 s per epoch on 2 GPUs, i.e. 0.42 s per micro-batch per GPU. On 1 GPU that is 21,278 × 0.42 ≈ **2.5 h per epoch** before refiner overhead.
   - Refiner overhead:
     - GPU: 3 small nets on 4 drafts plus the surrogate. Stage T needs ~0.1 s for 104 drafts, so ≤ 0.05 s here.
     - CPU: perturbation ~12 ms.
     - Side loading: ~10-20 ms per sample in workers.
     - Total: +10-25 %.
   - Expected **2.8-3.1 h per epoch → N = 6** for a ~19:45 launch (~22 h window). N = 7 only if the pilot shows ≤ 2.9 h.
   - Evaluation: 5 × 30 min on 2 GPUs ≈ 1.5 h after 18:00, so results about 19:30 on 09-30.
   - Mitigation: E0@N eval runs concurrently on GPU0 tonight (a few GB, about 10 % slowdown for 30 min).
2. **CPU and data-loader starvation.**
   - Load average is 25 on 32 cores: kyungmin's 4-rank run plus the GT builders (metric cache ×8 until ~18:15, SDF ×3-6, side store ×4 until ~19:30).
   - E1 and E2 need the same total loader throughput as E0: 2 × 9.5 samples/s, image decoding.
   - The pilot measures data-wait. Use WORKERS = 6 per job. Stop nothing of others.
3. **GT lag in epoch 0.** Estimate 60-80 % ok at launch, about 100 % by ~22:00 (inside epoch 0). Identical for E1 and E2, logged, and stated.
4. **GPU memory.** The refiners are about 3.4 M parameters each. Surrogate tensors are [4, ≤800, 41, 4 or 3] and TTC [4, A, 41, 3]. Total under 1.5 GB extra on top of E0. Host RAM per E2 sample: +5 MB (teacher BEVs) + 0.6 MB (GT), negligible.
5. **Numerics.**
   - Surrogate gradient spikes are contained by the refiner-only clip of 1.0 and the 0.1 BEV scale.
   - NaN is guarded (subset computation, nan_to_num, non-finite → 0 plus a count).
   - `gnorm/ref_bev` shows whether the refiner dominates the encoder gradient.
6. **Comparability limits (stated, not fixable by 09-30).**
   - One seed.
   - E0@N is a 2-GPU, 30-epoch-schedule checkpoint, so its LR at epoch N is still high. This is only a reference; the primary contrast E2−E1 is clean.
   - Teacher BEVs of train tokens are in-sample: ReSMap was trained on train_logs, and BEVFusion on navtrain.
   - The perturbation path extension is straight rather than the logged path.
   - The teachers must pass the liveness gate.
7. **Unchanged-E0 guarantee.** The golden dump must happen before the first edit. If it is skipped, the parity claim cannot be made.

## Timeline (09-29 → 09-30)

| time | step |
|---|---|
| 16:30 | Golden parity dump. Write `e2e.py` (~350 lines), agent/loss/targets/config/yaml edits (~120), `build_e2e_side.py` (~120), `stageE_parity.py`, `stageE_compare.py`. |
| ~18:15 | Start the side store in follow mode. Bump SDF workers. |
| 18:30-19:00 | CPU tests: parity, grid projection, RNG equality, E1/E2 2-step CPU smoke on 2 real scenes, eval-forward smoke (both flags). |
| 19:00 | Teachers done: liveness gate, snapshot copy with sha256. GPU pilot E1 (GPU0) and E2 (GPU1) concurrently, 300 micro-batches each. Measure s/it, memory, data-wait, λ_c. Fix N, W, λ_c in PRESTATED_DECISION_RULE. |
| ~19:45 | Launch E1 and E2. E0@N eval runs concurrently on GPU0. |
| 09-30 ≤18:00 | Training ends. Four evals on 2 GPUs (~1 h). `stageE_compare.py`. **Results ~19:30.** |
