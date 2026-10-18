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
* Teachers (`train_ck2 --v4-readout range`): DET gets the box-level B GT aux (no station BCE: it reads the hazard
  readout's a_* only, which the range readout does not have); MAP the SDF D aux; specialist key masking as
  v4_2.  teacher_forward / TeacherPair2.run return the range fields of the unique parents for constraint KD.
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

## Phase 0b — re-registration (written 2026-10-17 KST, BEFORE any held-out number; user: "b->c로 해봐")

Phase 0 failed criterion 1 at the pre-registered setting (report 85).  The user chose (b) a new pre-registration
of a corrected representation measured on held-out tokens, then (c) redefining idea 3 as SELECTION evidence.
Everything below is fixed now; the held-out run must not change it.

* **Data (held-out)**: the 12,179 navtrain_val tokens NOT in the F1 6,000 sample (token-disjoint; limitation: the
  same 214 logs).  Candidates: v2 r34 top-16 + valid var80 (the 96).  Executed states: ck/v4/lqr.py float64
  (parity 2.4e-5 m) from (status v0, a0).  Labels: the official labels (CK/labels/navtrain_val/cand, r34 variant
  labels).  GT objects CK/v3/gt_objects/navtrain_val, SDF sdf/navtrain_val.  No tuning on these tokens.
* **Representation R2 (learnable grid)**: as Phase 0 main (corridor v2 parent grid, 0.25 s x 20 bins, box target,
  U/LON channel for TTC, DAC on D bilinear) with two changes fixed from the Phase 0 diagnosis:
  (1) **stopped-aware B**: stopped / static objects block without the not-behind filter, moving objects with it
  (exactly tools/ck/v4/oracle/oracle_common.stopped_aware_B01 / stopped_mask); (2) **any4 read** for B and U
  (OR over the 2 x 2 cells around the read point).  Margins m_t 0.2 s, m_l 0.3 m.
* **Criterion S (primary, selection; idea 3 as selection evidence)**: switch selection = the v2 top-1 if it is
  inside the R2 GT tube, else filter the 96 to the inside rows and pick by the deployment blend (pool96_old_b1:
  beta 1, ck2T10 critic probabilities, SEL_W); vs the deployment blend over the 96 without the tube:
  mean dPDMS >= +1.0 with the paired log-bootstrap CI lower bound > 0, and break (NC/TTC/DAC pass -> fail over
  tokens whose blend pick passes) <= 1.3 %.
* **Criterion 1 (secondary, representation fidelity)**: per key NC / TTC / DAC and joint on all 96 executed rows,
  fail recall >= 0.90 AND pass recall >= 0.90 (point estimates; CIs reported).
* **Criterion 2 (informational)**: R2 computed edit on the v2 top-1 vs the DET lateral head, as Phase 0.
* **Decision rule**: S passes -> (c): retrain the DET / MAP range teachers with R2 targets and measure how much of
  the GT selection gain their learned fields recover (fixed-pool critic-only, val decides, navtest reported).
  S fails -> stop idea 3 and report.  Criterion 1 is reported either way (if it fails, the learned tube is known
  to be an approximate selection signal, not a faithful copy of the official rules).
* EPDMS: not computed (no navtrain_val v2.2 metric cache; not approved).

### Phase 0b result (2026-10-17, CK/v4/analysis/range_oracle/heldout_r2)
Criterion S PASS (switch - blend +1.91 [+1.42, +2.42], break 0.30 %); criterion 1 passes on the point estimates
(NC 0.960 / 0.935, TTC 0.952 / 0.905, DAC 0.931 / 0.998, joint 0.953 / 0.903; TTC / joint pass-recall CIs include
0.90); criterion 2 +1.51 [+1.02, +2.04], break 0 %.  Decision rule -> (c).  Note: the Phase 0 grid tube gives the same
S (+1.92) on these tokens, i.e. S does not discriminate the R2 changes; they improve criterion-1 fidelity.

## Phase 1c — decision rule for (c), registered BEFORE any R2 teacher exists (2026-10-17 KST)

* Teachers: v4T10_rng2 (DET) / v4M10_rng2 (MAP), `run_teachers.sh READOUT=range R2=1` (ck2T10 / ck2M10 recipe,
  10 epochs, R2 targets: stopped-aware B, any4 read, Bp for TTC), evaluated at `--which last`.
* Decision split: navtrain_val held-out 12,179 tokens (the Phase 0b set), DET teacher only (val has no ReSMap);
  caveat: the BEVFusion DET backbone is in-sample on navtrain.  Tool: tools/ck/v4/eval_range_teacher.py.
* **Primary P1** (learned range as selection evidence): criterion S with the LEARNED DET tube on the keys it owns
  (NC, TTC; DAC not flagged), switch rule and deployment blend (ck2T10, Phase 0b convention) exactly as Phase 0b.
  Pass iff ALL of: (a) mean dPDMS vs the blend >= +0.30 with the paired log-bootstrap CI lower bound > 0;
  (b) break <= 1.3 %; (c) the learned tube recovers >= 25 % of the GT R2 tube's gain on the same keys and tokens
  (point estimate; 25 % = the v3 precedent of a learned forecaster vs its oracle, report 67).
* Reported, not deciding: (i) learned-tube fidelity (criterion-1 metrics); (iii) fixed-pool critic-only selection
  (pool96_ck_noim and the blend modes) of the DET range teacher vs ck2T10 and v4T10_haz; navtest (DET + MAP pair):
  S with the full learned tube, its fraction of the GT gain, critic-only pair selection vs ck2T10+ck2M10 and
  v4T10_haz+v4M10_haz.
* Decision: P1 passes -> ask the user about the v4_3 student run (about 52 h on 2 GPUs).  P1 fails -> report; the next
  candidates are past-frame inputs for forecasting or the student GT range aux (rg_gt_aux), decided with the user.

### Phase 1c amendment (2026-10-17 KST, still BEFORE any R2 teacher exists; the waiter has not launched)
A design check found that P1 as registered cannot be passed even by a perfect DET field: with the tube on NC / TTC only
(no DAC flag on val), the SWITCH rule keeps the v2 top-1 whenever it is NC/TTC-inside and so lets its DAC failures
through; the GT R2 tube on these keys gives switch - blend -2.15 [-2.74, -1.53] with break 3.75 % on the val held-out
tokens (ratio criterion (c) then has a negative denominator).  P1 is therefore replaced, before any result:
* **P1 (amended)**: FILTER rule with the learned DET tube (NC, TTC): drop the rows the tube flags as outside, pick by
  the deployment blend (pool96_old_b1, ck2T10, pool-order ties) among the rest (none left -> the blend pick); vs the
  blend over the 96.  Pass iff (a) mean dPDMS >= +0.30 with the paired log-bootstrap CI lower bound > 0, (b) break
  <= 1.3 %, (c) >= 25 % of the GT R2 tube's filter-rule gain on the same keys and tokens (GT reference about +1.0 on
  the val held-out tokens; must be positive with CI lower bound > 0, else P1 is void and reported as such).
* Everything else in Phase 1c is unchanged.  The amendment is committed before the R2 teachers start.

## Phase 1e — P2: privileged out-of-window input (registered 2026-10-19 02:45 KST, BEFORE the runs start)
P1 FAILED (report 90, 2026-10-18 22:30 KST): the learned DET tube gives filter - blend -0.12 [-0.24, -0.01] vs the GT
tube +1.01.  No bug.  The misses are moving objects at t >= 2 s, and more than half of the missed failures are caused by
objects OUTSIDE the teacher's BEV window at t0 (x 0-32 m, |y| <= 32 m): behind, beyond 32 m ahead, or not yet present.
Ceilings on the same filter rule, val 12,179 tokens (CK/v4/analysis/p1diag/candidate_level/window_ceiling{,2}.json,
2026-10-19 02:05 KST): in-window objects + perfect future +0.27 [+0.11, +0.44]; in-window + CV -0.13; all objects + CV
-0.19; in-window perfect + out-of-window CV +0.41 [+0.19, +0.62]; in-window CV + out-of-window perfect +0.63
[+0.44, +0.84]; all objects + constant acceleration (GT 0.5 s speed change, stop-clamped) +0.29 [+0.14, +0.47]; all
objects perfect to 2 s then CV +0.49 [+0.31, +0.68]; in-window perfect to 2 s then CV +0.12 [-0.00, +0.24].
Question: given the out-of-window t0 state (as an input the teacher can learn to extrapolate), does a learned range
teacher recover a useful share of the GT gain?  This is a TRAINING-TIME teacher question (privileged input; never a KD
teacher as is, model.load_ck refuses it without allow_privileged).
* **Design** (`V4Cfg.rg_priv`, ck/v4/config.py docstring; `--v4-rg-priv`): the batch's GT objects present at t0
  whose t0 centre lies in the region are moved at their annotated t0 velocity (constant velocity) and rasterised as an
  any-overlap occupancy [48, 17, 20 bins] on each parent grid (range_targets.priv_channels, the GT field's own
  blocked_field, behind_m None); the readout's per-cell MLPs get it as a zero-initialised 1x1-conv term.  Everything
  else = v4T10_rng2 (R2, any4, stopped-aware, same recipe, 10 epochs, seed 0).  Two arms, in parallel:
  - `v4T11_rngPout_PRIV` (region 'out': every t0 object outside the BEV window — behind, beyond 32 m, sides);
  - `v4T11_rngPfar_PRIV` (region 'far': the front band x in (32, 64] m, |y| <= 32 only — the part a front camera
    student could in principle see).
* **P2 decision rule** (each arm separately; the P1 (amended) FILTER rule, keys NC / TTC, the registered val held-out
  12,179 tokens, blend ck2T10 pool96_old_b1, `tools/ck/v4/eval_range_teacher.py --stage infer` then `--stage eval`):
  PASS iff (a) filter - blend >= +0.30 with the paired log-bootstrap CI lower bound > 0, (b) break <= 1.3 %,
  (c) >= 25 % of the GT R2 tube's filter-rule gain on the same keys and tokens.  Reported beside the rule: the arm vs
  the matching ceiling (out: all objects + CA +0.29 / perfect 2 s +0.49; far: no exact ceiling, the out-of-window
  perfect +0.63 is an upper bound), and the plain R2 teacher's -0.12.
* **Reading**: 'out' PASS -> the forecasting-from-t0-state lever is real once the coverage limit is removed; the next
  question (with the user) is how the student gets it (GT-direct range aux = control arm B, a KD prior, or a wider
  student input).  'far' PASS -> the front band alone is worth it and a 0-64 m student BEV is the natural follow-up.
  Both FAIL -> a CV-level input is not enough even all-around; the remaining lever is forecasting beyond CV from
  history (the v3 result says that is small) or the corrector line; report either way.
* The runs are launched by the user's standing approval of 2026-10-19 ("teacher 학습 알아서 해놔"); GPUs 0,1 (out)
  and 2,3 (far); eval chain on completion.
