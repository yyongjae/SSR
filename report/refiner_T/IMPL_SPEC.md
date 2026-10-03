# Stage T implementation spec (teacher refiner, no trained student) — interface contract

Owner: yongjae. Written 2026-09-28. Design: report/36_refiner_kd_architecture.md (v2) + the "no trained E2E" variant
(stage T = teacher-refiner qualification; necessity is decided later by E2E E1 vs E2).
Detailed rationale / numbers: report/refiner_kd_design/{architecture_draft_v0.md, critic_logic.md, critic_impl.md, fact_*.md}.
Where this spec and the draft differ, THIS SPEC WINS.

## 0. Scope of stage T
Arms (identical code except input): R_T (teacher BEV, BEVFusion 50x100 cache), R_none (no scene feature: zeros).
Optional later: R_GT-cur. No student model, no student dump.
Drafts: perturbed human (GT) trajectories + constant-velocity drafts (+ PDM-none drafts later).
Losses: differentiable surrogates vs GT future objects (all tracks, 360 deg, 0-5 s) + drivable SDF + progress + comfort + modification; gate BCE on official labels.
Decision: dev (held-out navtrain logs) official scoring, R_T vs R_none. navtest only for the final frozen report.

## 1. Paths
- Code package: /home/external-user/yongjae/SSR/navsim/agents/para_ssr/refiner/  (geometry.py, decoder.py, gt_future.py, sdf.py, surrogate.py, adapters.py, corridor.py, refiner_net.py, data.py)
- Tools: /home/external-user/yongjae/SSR/tools/refiner/  (make_splits.py, build_metric_cache.py, extract_human.py, build_future_objects.py, build_sdf.py, score_trajectories.py, make_draft_bank.py, validate_surrogate.py, train_refiner.py, eval_refiner.py)
- Tests: /home/external-user/yongjae/SSR/tools/refiner/tests/  (pytest, CPU, < 2 min each)
- Data root: /home/external-user/ssd/yongjae_refiner/ {splits, metric_cache, human, objects, sdf, drafts, scores, runs}
- Reports/results: /home/external-user/yongjae/SSR/report/refiner_T/
- Python: /home/external-user/miniconda3/envs/ssr/bin/python. CPU jobs: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10, <= 4 worker processes total per job (machine is shared, load ~30/32).
- Existing inputs (read-only): teacher cache /home/external-user/datasets/teacher_cache/bevfusion/cache_{train,val}_50x100 (manifest ckpt sha head must equal cddf943ffec8d6a8);
  raw logs /home/external-user/yongjae/SSR/data/dataset/navsim_logs/trainval (and test: /home/external-user/navsim/download/test_navsim_logs/test);
  navtest metric cache /home/external-user/yongjae/SSR/data/exp/metric_cache; E navtrain metric cache (9,000 tok) report/cause_and_correction_tests/E_train_split_feasibility/metric_cache;
  token->log table report/cause_and_correction_tests/E_train_split_feasibility/tokens/navtrain_token_log.parquet (token, log, frame_idx, map_location, part in {train,val}).
- Do NOT modify existing repo files. Do NOT import kyungmin/byounggun code at runtime (copy with a recorded sha256 if reused).

## 2. Frames, grids, time
- N frame: NAVSIM ego frame at t0 (rear axle origin, x forward, y left, heading CCW from +x, radians). All trajectories, objects, SDF, decoder use N.
- Trajectory: float32 [8,3] = (x, y, heading) at t = 0.5..4.0 s. Implicit pose 0 = (0,0,0) at t=0.
- Dense time grid: n = 0..40, t = 0.1 n (same as the scorer reference).
- S grid (BEV, student-compatible): [C, 50, 100]; row r -> x = (r+0.5)*0.64 m (0..32); col c -> y_left = 32 - (c+0.5)*0.64 m (col 0 = left).
  Teacher cache bev_feature is (C, H=x_fwd 50, W=y_left 100) with col j -> y_left = -32 + (j+0.5)*0.64  => S = bev[:, :, ::-1]. Verify with the cached heatmap vs cached boxes (unit test).
- E grid (SDF): rows x = -8..72 m, cols y_left = +32..-32, 0.25 m cells => [320, 256], cell centres at -8+0.125+0.25i, 32-0.125-0.25j.
- Ego (Pacifica): half length 2.588, half width 1.1485, rear-axle -> box centre +1.461 m along heading (= sf_common constants).
- Comfort limits (pdm_comfort_metrics.py): lon accel [-4.05, 2.40], |lat accel| 4.89, |yaw rate| 0.95, |yaw accel| 1.93, |lon jerk| 4.13, |mag jerk| 8.37 (official uses Savitzky-Golay on LQR-tracked states).

## 3. Module contracts

### 3.1 geometry.py (torch, batched, differentiable)
- class DraftPath: built from poses [B,8,3] (+ origin). Path = C2 interpolating curve through origin and the 8 points, start tangent (1,0), parameterised by arc length.
  Methods: knots() -> S [B,9] cumulative arc length at t_k (S[:,0]=0); eval(s [B,M]) -> xy [B,M,2], heading [B,M], normal [B,M,2] (left), kappa [B,M];
  extend(length, policy='const_curv') for s > S_8 (constant curvature from the last segment, kappa clipped to min(0.95/v, 4.89/v^2, 0.213)), with a flag.
  s0(t): draft arc length vs time = piecewise linear through (0.5k, S_k) (matches scorer interpolation); identity must hold exactly at knots.

### 3.2 decoder.py (M4, no learned params)
- decode(tau0 [B,8,3], z_lon [B,6], w_lat [B,6], v0 [B], mode: 'A'|'B'='A') -> dict:
  traj [B,8,3], dense [B,41,3] (0.1 s, linear time interpolation of [origin; traj] exactly like the scorer reference), s [B,41], v [B,41], d [B,41], kappa [B,41], flags.
- Longitudinal: Δv(t) cubic clamped B-spline on [0,4] s, 8 control points c0..c7, knots [0,0,0,0,.8,1.6,2.4,3.2,4,4,4,4]; c0=c1=0.
  Parameterise the DERIVATIVE control points Q_i = A * tanh(z_i) (i=1..6; A = A_DEC=4.0 for decel side, A_UP=2.0 for accel side), integrate c_{i+1} = c_i + Q_i*(u_{i+4}-u_{i+1})/3,
  so |δa| <= A exactly (convex hull). Mode A: clamp so that v1 <= v0 pointwise and s1(t) <= s0(t) (never faster than the draft); mode B: +ΔV_ACC<=2 m/s, extension <= min(5 m, v_end*1 s).
  v1 = relu(v0(t)+Δv(t)) (NO softplus). s1 = cumulative trapezoid on 0.1 s.
- Lateral: Frenet offset d(s) = cubic clamped B-spline over s in [0, S_L], control points e0..e7, e0=e1=0, e_i = D_MAX*tanh(w_i), D_MAX=2.0 m; d≡0 if S_8 < 3 m.
  Curvature projection alpha = min(1, kappa_lim / max|kappa_new|) applied in BOTH training and inference.
- Heading (H1): h = h0(s1) + atan2(d', 1 - kappa_path*d) where h0(s) = draft heading interpolated in arc length (unwrapped).
- Identity: z=0,w=0 => traj == tau0 exactly (unit test: max abs error 0.0). Callers that "do not modify" must return the ORIGINAL tau0 bytes.
- sample_perturbation(...) helper for the draft bank: produces (z_lon, w_lat) in the SAME basis (families below), plus a free-form path source for speed-ups (human path up to 5 s from scene / 8 s from log).

### 3.3 score_trajectories.py (tools; official scoring, batched)
- score_token(metric_cache, trajs [K,8,3]) -> list of dicts per trajectory: nc, dac, ddc, ep, ttc, comfort, pdms, raw_progress, pdm_progress_eff, nc_track, nc_time_idx.
  Simulate [PDM-Closed, traj_1..K] together; recompute per-trajectory multiplicative metrics (NC, DAC, DDC with the official DDC rule: DDC_i = thr(max_t(op[0]+op[i])) as in pdm_scorer.py L398-439),
  EP_i = raw_i*mult_i / max(raw_0*mult_0, raw_i*mult_i) with the official threshold: if that max <= progress_distance_threshold (5.0 m, default_scoring_parameters.yaml) then EP=1 (0 if mult=0),
  PDMS = mult*(5EP+5TTC+2C)/12. MUST equal single-trajectory navsim.evaluate.pdm_score (cf_common.score) exactly: test on >= 500 navtest tokens x (original + 2 perturbed) — 0 mismatches.
- CLI: score a draft file (npz) for a token list with <= 4 workers, resumable shards -> parquet under /home/external-user/ssd/yongjae_refiner/scores/.

### 3.4 gt_future.py + build_future_objects.py (M6)
- Source: raw log anns frames cur..cur+10 (cur = the token's frame; 0..5 s keyframes), all classes, 360 deg, all track_tokens whose centre is within R = max(80 m, S_avail+25 m) of the t0 origin at any keyframe.
  Transform each frame (ego pose of that frame) to N (centres, heading, AND velocity rotated). Store first-appearance L, W, heading, v (official uses first-appearance size/type).
- Per token npz: kf [A,11,6] f32 (x,y,heading_unwrapped,vx,vy,present), first [A,6] (L,W,heading,vx,vy,first_k), meta [A,5] i16 (class, is_agent[vehicle/pedestrian/bicycle], first_k, last_k, singleton).
- query(obj, t_dense [41] or [51]) -> boxes [A,T,5] (cx,cy,heading,L,W) + state [A,T] in {OBS, ABSENT_OFFICIAL} following the metric cache rule exactly (index-based time k*0.5; linear interpolation;
  singleton tracks placed at all times as the metric cache does). UNKNOWN (outside annotation reach: > 75 m from the GT ego at t, or t > 5 s) flagged separately; measure its rate.
- Validation: rebuild metric-cache polygons for >= 200 navtest tokens and >= 200 E navtrain tokens: corner error < 1e-3 m, presence 100% equal. Red-light pseudo objects are skipped by official NC/TTC: exclude.

### 3.5 sdf.py + build_sdf.py
- From metric cache drivable_area_map (official DAC layers) -> SDF on the E grid, float16, metres, positive inside drivable area, negative outside. Save per token npz under sdf/.
- Validation: corner-SDF>=0 at all 41 LQR-tracked footprints vs official DAC on navtest original trajectories (report agreement, AUC); record build time/token.

### 3.6 Draft bank (make_draft_bank.py; human paths via extract_human.py)
- Human trajectory per token from raw logs (future frames 1..8 at 0.5 s in N) + human path up to 8 s (for speed-up draft construction) + ego v0, a0 at t0.
- K = 13 drafts per token (ids fixed): 0 identity(human); 1-3 small (|A_p|<=0.2, |D|<=0.3 m); 4-6 L-const speed-up (A_p ~ U[0.2,1.3] m/s^2, t_on in {0,0.5,1,1.5,2}, jerk<=2);
  7 L-ignore-brake (only if human decelerates >= 1 m/s; else another L-const); 8 L-creep (only if human 4 s progress < 2 m; else L-const); 9-10 lateral (|D| ~ U[0.3,1.5] m, s_on ~ U[.2,.6]*S);
  11 combined (L-const + lateral); 12 constant-velocity (v0 along the initial heading, straight) OR heading-drift family if CV is invalid. All generated so that t0 continuity holds
  (first-segment speed - v0 in [-0.8,+0.6] m/s) and kinematic limits hold; tokens with 1.0 s frame gaps are skipped. Seeded by hash(token).
- Output: drafts/<split>/<token>.npz: drafts [13,8,3] f32, family [13] i8, params [13,6] f32. Then official labels via score_trajectories -> scores/<split>.parquet (token, k, nc, dac, ddc, ep, ttc, comfort, pdms, pdm_progress_eff, ...).
- Target: official failure rate over the bank 20-35% (tune A_p range on a 500-token pilot); report family-wise failure rates.

### 3.7 surrogate.py (M7, torch)
- All terms on the 41-point dense reference from decode().
- Collision: ego box vs object boxes (from gt_future.query at 0.1 s): smooth signed separation g = smoothmax over the 4 separating axes (logsumexp temperature 0.05 m, subtract T*ln4 bias);
  C_col = mean_n sum_j w_j * m_j(n) * beta*softplus((m_col - g)/beta), m_col=0.3, beta=0.1, w=1 agent / 0.5 static; mask: OBS only, not-behind (object centre >= -1.3 m along ego heading from ego centre),
  exclude (j,n) where the HUMAN footprint also overlaps (g_human < 0) — NOT merely < m_col; UNKNOWN never counted as free (drop the draft from progress reward if its footprint enters UNKNOWN).
- DAC: C_dac = mean_n sum_corners beta*softplus((m_dac - SDF(corner))/beta), m_dac=0.2; bilinear SDF on the E grid; out-of-grid corners counted and excluded (no border copy).
- Progress: aligned with the official EP: P(tau) = raw progress along the route centreline of the ego CENTRE from t=0 to 4 s (metric cache centerline); with P_pdm = pdm_progress_eff from the scorer (PDM-Closed raw*mult),
  mode A: L_prog = relu(min(P(tau0), P_pdm) - P(tau1)) / max(P_pdm, 5.0).
- Comfort: on 0.5 s keyframes including the t0 state (v0,a0) + analytic spline terms (δa, jerk, v^2*kappa); penalty relu(|x| - 0.9*lim)^2/lim^2. Acceptance: human trajectories violate <= 1%.
- Modification: C_mod = mean_n |s1-s0|/5 + mean_n |d|/1.
- Gate: BCE(p_g, y) with y = 1[draft fails official NC<1 or DAC<1 or DDC<1] (TTC variant as option), pos_weight=(1-pi)/pi capped at 10; gate head does NOT backprop into the trunk (use detached trunk features).
- validate_surrogate.py (M8): on dev drafts with official labels: trajectory-level AUC / recall at the training margin / human false-alarm rate; pair-level (draft vs a modified version) sign agreement.
  Acceptance (pre-stated): NC recall >= 0.70, human false alarm <= 2%, pair P(official fixed | surrogate says fixed) >= 0.6. If not met: report, then consider a differentiable tracking approximation (M7b) — do not tune on navtest.

### 3.8 adapters.py, corridor.py, refiner_net.py, data.py, train_refiner.py, eval_refiner.py
- Input feature per arm: R_T = teacher bev_feature -> S grid -> per-channel z-score (mean/std over the TRAIN split, saved to runs/<run>/norm.npz) -> 1x1 conv 256->128 GELU 128->64. R_none = zeros [64,50,100].
- Corridor (M3a): along DraftPath, N_s=48 stations, spacing max(1 m, S_look/48), S_look = S_8 + max(8 m, v_end*1 s) (extension flagged); lateral offsets 17 in [-4.8, 4.8] m;
  grid_sample bilinear, padding zeros, align_corners=False; channels = 64 features + 6 geometric (in_grid, path_valid, s/48, d/4.8, draft arrival time t_d(s)/4 clipped, draft speed v_d(s)/15) -> [B,70,48,17].
- Global (M3b): 5x5 avg-pool of the 64-ch map -> 200 tokens (10x20) + learned positional embedding.
- Refiner (M5): draft token MLP(69->192), corridor conv encoder (70->96->96->128, GN, GELU), station tokens Linear(17*128->192), global Linear(64->192)+pos; 4 transformer layers d=192 h=6 FFN 768:
  self-attn over [draft + 48 stations], cross-attn to 200 global tokens, FFN; heads: gate (detached trunk) MLP->1, lon head [draft||mean station]->6 (z_lon), lat head ->6 (w_lat).
  Parameter count identical across arms except the adapter; unit test.
- Draft token features u_d (69-d): v0, a0, command one-hot(4), 8 x (x/40, y/10, cos h, sin h), per-segment speed/accel/curvature (8 each), S_8/40, flags(2) — document exact order in code.
- Data (data.py): token-batched (8 tokens x 13 drafts); per token reads teacher npz (bev_feature only), objects npz, sdf npz, draft npz, labels, human npz, metric-cache-derived small arrays (centerline samples for progress, pdm_progress_eff).
  Pack small arrays into memmaps per split to avoid per-step metric-cache loading.
- train_refiner.py: --arm {T,none} --fold k --seed s --gpu g; AdamW lr 3e-4 cosine, wd 0.01, epochs 40 (early stop on inner-train-val loss), fp16 autocast.
- eval_refiner.py: apply to dev drafts (and later navtest): p_g >= theta ? decode(tau0, Δ) : ORIGINAL tau0; official scoring via score_trajectories; metrics per §4.

## 4. Splits and decision rule
- Pool: navtrain tokens (train_logs + val_logs; no student involved in stage T). Sample N_T tokens (default 24k train + 8k dev; adjustable) with log-level sampling stratified by city;
  dev = held-out logs (never used for training/selection of weights except the stated theta/weights procedure, which uses 5-fold log-level cross-fitting WITHIN the train pool).
- Metric cache is required for every token used (labels + scoring): build for the sampled tokens (reuse E's 9,000 when they fall in the sample) with E's e2_cache.py logic (<=4 workers, background).
- Decision (report/refiner_T/PRESTATED_DECISION_RULE.txt): R_T vs R_none on dev, seeds averaged, log-cluster paired bootstrap; primary = official PDMS at a pre-stated progress budget, total failures (NC+TTC primary; DAC/DDC non-inferiority), new failures;
  outcomes: pass / equivalent (= "under the current frozen-feature/refiner setting the teacher shows no correction advantage") / inconclusive (add seeds) / worse (diagnose).
