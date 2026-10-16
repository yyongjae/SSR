# refiner v4_3 — allowed-range critic (report 84 idea 3 + idea 1 roles + restricted idea 2)

Status: design agreed with the user 2026-10-16 (KST); implementation in progress on branch `refiner-v4`.
Source: SSR/report/84_three_ideas_review.md (main), 80 (EP head), 82 (KD transfers errors), 83 (LQR tracker), 77/81.

## Decisions (user, 2026-10-16)

1. **Order**: the report-84 §5 GT oracle (CPU) and the opt-in v4_3 code run in parallel; nothing is trained until the
   oracle passes and the user approves (GPU runs always need approval; GPUs 0-3 only).
2. **Computed edits**: allowed only as EXTRA candidates (corrector A's K = 8 slots); the critic picks among 96 + 8;
   never a gate.  Corrector A becomes an optional zero-init residual on top of the computed edit, OFF in the first
   student run.
3. **KD routing**: constraint KD (DET blocked field, MAP margin field) + score KD restricted to DET -> NC and
   MAP -> DAC.  Everything stays selectable by config (old score KD on all keys must remain reproducible).
4. Settled by the designer (stated to the user): executed paths from a batched torch port of the official LQR tracker
   + bicycle (0 params); oracle pass criteria two-sided (below); success staged (fixed-pool paired critic evals
   first, e2e confirms; calibration-matched nokd only for a KD claim).

## Invariants

* Every new option defaults OFF; with defaults, outputs/state dicts are v4_2 bit for bit
  (tests/test_v4_model.py::test_off_is_bit_identical and the v4_2 tests keep passing).
* CONFIG_VERSION becomes "v4_3"; `V4Cfg.from_dict` must ACCEPT "v4_2" dicts (the trained v4T10_haz / v4M10_haz
  teachers store "v4_2"); missing new fields take defaults.
* Trainable parameters of the critic (no corrector) stay within 0.2 % of CK2 as trained (3,391,755 arm S,
  3,391,243 arm T/M) for every readout (tests/test_v4_review_fixes.py::test_param_budget extended to 'range').
* No ranking / margin losses, no gamma conservativeness, no rule gate (edits are candidates only).
* GPU: never in tests; CPU tests use `CUDA_VISIBLE_DEVICES=` and OMP_NUM_THREADS=1.

## Modules

### M1 `navsim/agents/para_ssr/ck/v4/lqr.py` — official tracker port (0 params)
* Batched torch port of NAVSIM's PDM simulation used by the official scorer (PDMSimulator: 8 poses -> 0.1 s
  interpolation (41 states) -> LQR tracking (lon: speed reference, no position feedback; lat: Q diag(1,10,0), R 1,
  1 s horizon) -> kinematic bicycle (wheelbase 3.089 m, accel lag 0.2 s, steer lag 0.05 s, |steer| <= 60 deg),
  initial steer 0).  Reference implementation: CK/ck2/analysis/detmap_path_ideas/F1/scripts/f1_common.py (bit-exact
  vs official labels) and the NAVSIM / nuPlan sources it calls.
* API: `simulate(traj [B, 8, 3], ego [B, k] (whatever the official init needs: velocity/acceleration/yaw rate in the
  ego frame), dtype) -> states [B, 41, 6]` (x, y, heading, v, a, steer; ego frame at t0 = the N frame).
* Parity: tests/test_v4_lqr.py compares with F1 out/{val,train,navtest}/run.npz on samples: pose error <= 1 mm,
  heading <= 1e-3 rad (float64); float32 tolerance documented.

### M2 `navsim/agents/para_ssr/ck/v4/range_targets.py` — GT fields and 0-parameter reductions
* `blocked_field(parent_geom, boxes, state, t_grid, heading='parent') -> [P, 48, 17, M]` bool/float:
  the OFFICIAL ego box (half length 2.588, half width 1.1485, rear-axle offset 1.461) with rear axle at station j,
  lateral offset i, heading along the parent path, overlaps a GT OBS object (gt_future.query(_torch), metric-cache
  rule) at time t (SAT, refiner/surrogate.box_gaps); the not-behind mask (surrogate.BEHIND_M) applied as the
  official at-fault proxy; 0.1 s grid max-pooled into M bins of `tbin_s` (default 0.25 s, 20 bins to 5.0 s).
  torch (training, online from ref_obj_* like evidence_aux) and numpy twin (oracle).
* `margins(D [P,48,17] or sdf, parent_geom) -> m_left, m_right [P,48] (m), L [P] (m)`: the largest lateral shift
  (left / right of the parent) at which the 4 box corners (parent heading) stay inside (D > 0), by a fine scan of
  the lateral axis with linear interpolation; L = arc length of the first station whose centred box exits.
* `v_max(kappa [P,48], m_out [P,48], fit) -> [P,48]`: largest speed whose expected tracking path error
  e(v |kappa|) + 0.3 m stays below the outside margin; `fit` = piecewise-linear table fitted on F1 feats
  (yaw rate -> path error); prior refiner/geometry.kappa_limit.
* `windows(B_field)`: per (j, i) the latest blocked time before a free interval reachable from t0 (fast side) ->
  t_out_block [P,48,17] (+inf if never blocked; 0 if free from 0).
* `executed_slack(states [B,41,6], parent_geom, B_field, D or sdf)`: per candidate, on the EXECUTED states:
  NC slack (time margin between arrival and the end of the blocking window at the critical station, s),
  TTC slack (same with the 0.3/0.6/0.9 s projections), DAC slack (min corner margin, m), the critical station/time.

### M3 parent dedupe (model.py `CKNet.score_set`)
* Run `trunk.stations` once per unique parent (inference: c // 6; training: given parent indices) and gather;
  bit-identical to the repeated path (max diff 0); also returns the parent context for re-use by edits / AWR.

### M4 readout 'range' (`ck/v4/range_readout.py`, V4Cfg.readout = 'range')
* Scene fields on the parent grid (per unique parent): `B` blocked logits [P,48,17,M] (head on [Hc, xs]),
  `D` margin [P,48,17] (as evidence marg_head); analytic m_left/m_right/L/v_max from D (M2).
* Candidate states: `exec_model` 'plan' (8 knots, v4_2 evidence behaviour) | 'lqr' (M1 41 states, default for
  'range'); projected on the parent's station frame (sibling.frenet generalised to arbitrary state sets).
* Evidence: NC = soft max over executed states of B at (s, d, t); TTC = same at the 0.3/0.6/0.9 s projections along
  the path at t + delta; DAC = min corner margin read from D (+ curvature overspeed when vmax='feature');
  log P_pass per key, + zero-init residual MLP (as evidence).
* Outputs (score_set 'haz' dict, all [T, K, ...]): `occ_logit` (=B, per candidate's parent), `margin` (=D),
  `lp_nc/lp_ttc/lp_dac`, `slack_nc/slack_ttc/slack_dac`, `crit_station`, `crit_time`, `exec_states` [T,K,41,6],
  plus the v4_2 keys (haz_mask, t_arr, t_in, t_out, d_lat, is_self, pts).
* EP head (report 80 E1, `ep_evidence` True for 'range'): ep_in += detached [lp_nc, lp_ttc, lp_dac, slack_*,
  executed 4 s progress ratio S_exec / max(P_ref_hat, S_exec, 5 m), margin-weighted candidate-position features
  (w xs, w Hc(d))].  Head widths rebalanced to the parameter budget.
* Comfort head unchanged (g); lateral head as evidence.

### M5 computed edits (`ck/v4/edits.py`, `V4Cfg.edits` 'off' | 'computed', edit_k 8)
* For the critic's top-`edit_k` parents (by the deployment selection score when available, else critic-only),
  using the PREDICTED fields: if the parent is inside the tube -> no edit (slot = the parent itself, flagged);
  NC/TTC-limited -> longitudinal edit = 0.8 s-ramp deceleration magnitude a in [-4, 0) (variants.make_variants
  family) found by bisection so that the executed (M1) arrival at the critical station is >= t_out_block + 0.2 s;
  DAC-limited -> lateral edit = early smoothstep offset D in [-2, 2] m (decoder.smoothstep_lat, onset 0.2 S8, the
  rule l+-0.5 shape) toward the larger margin until the executed min corner margin >= 0.3 m; both -> longitudinal
  first, lateral on the edited profile.  No edit when v0 < 2 m/s or S8 < 3 m, or when the tube is empty.
* Edits fill the corrector's K slots (edit_traj / edit_src), are scored by the same critic on the source parent's
  corridor, inherit the source's v2_final / im for the blended selection, and are labelled in the labeler2 96 + 8
  call.  Corrector A (optional) adds a zero-init residual on the computed edit (off by default).
* `eval_e2e2` headline: selection over 96 + 8 with a val-frozen beta (closes review F4).

### M6 KD routing (online2, train_ck2)
* `kd_score_keys` keeps its meaning (keys that get score KD; routing nc/ttc <- DET, dac <- MAP, ep/comfort <- mean).
  v4_3 student arm: `kd_score_keys: [nc, dac]`.
* `kd_score_scope`: 'all' (default, old) | 'near' (NC KD only on tokens with ttc0 < 4 s to the nearest lead,
  DAC KD only on parents with min corner SDF < 1 m; both from ref_* GT, train-time only).
* `kd_constraint` block: enabled, lambda_occ (DET B BCE to the teacher's sigmoid(B) on the unique parents),
  lambda_margin (MAP D Huber where s <= map_range_m = 32 m), both masked by teacher ok.
* Teachers (`train_ck2 --v4-readout range`): DET gets the box-level B GT aux (+ station BCE on its own keys); MAP the
  SDF D aux; specialist key masking as v4_2.  teacher_forward / TeacherPair2.run return the range fields of the
  unique parents for constraint KD.
* GT data: GTLoader reads ref_obj_* from CK/v3/gt_objects/<split> and ref_sdf from sdf/<split> for val / navtest
  (builders: tools/ck/v4/build_sdf_split.py).

## Phase 0 oracle (report 84 §5) — CK/v4/analysis/range_oracle
* Data: F1 val sample (6,000 tokens x par16 + var80 + latT16, executed states bit-exact), GT objects
  (CK/v3/gt_objects/navtrain_val), val SDF (built), labels with time idx (CK/labels/navtrain_val/cand, r34 variants).
* Criterion 1 (two-sided): on the executed states of the 96, per key (NC, TTC, DAC) and joint: failure recall >= 0.9
  AND pass recall >= 0.9 of 'outside tube <=> official fail' (official exceptions handled: static-object NC 0.5 =
  fail, not-behind proxy, stopped-ego TTC skip; DDC / UNKNOWN mismatches reported separately).
* Criterion 2: v2 top-1 parent; minimal computed edit when outside the GT tube, else none; edited trajectories scored
  by the official scorer; vs the DET lateral head (ck2T10 r34 lat_traj16) on the same tokens:
  mean dPDMS >= +0.5 per token with paired log-bootstrap CI lower bound > 0, and break (any NC/TTC/DAC pass -> fail
  over passing top-1 tokens) <= 1.3 % and <= the head's break on the same definition.
* Also: empty-tube rate; scene bins (lead 10-25 m, near curb, lane-only); point vs box target; bin size 0.1/0.25/0.5 s;
  margin sensitivity; curvature speed-limit probe; selection-only oracle; EPDMS next to PDMS.
* Both pass -> ask the user about teacher retraining; either fails -> report and drop idea 3.
