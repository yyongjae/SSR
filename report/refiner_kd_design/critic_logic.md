**Feedback coverage (each user item)**

- **F1a. Deceleration-only refiner vs a ×0.6 draft: addressed.** M1's matching rule, M4 modes A/B and §5 D1/D2 cover it. Mode A drops slow perturbations. Mode B allows L-slow. Gaps: the mode-B L-slow range cannot be undone (issue 7). Slow real drafts are unfixable in mode A; the draft says so but never reports the fixable subset (issue 21).
- **F1b. ×1.4 needs path beyond the 4 s GT: addressed.** Uniform resampling is banned, offsets are additive, S_avail comes from the scene (5 s) or the log (8 s), and training drafts use no extrapolation. Gaps: L-creep follows the route centerline, so t0 continuity is not guaranteed (issue 8). Tokens dropped for lack of path are not reported by speed bin (issue 8).
- **F1c. Continuity with ego pose and heading: structurally addressed** (c0=c1=0, e0=e1=0, first-segment speed check). Gaps: the |δa| bound is wrong (issue 6). The creep path starts from the centerline (issue 8).
- **F1d. Enough safe drafts: addressed** (30% identity or small, bank failure target 20–35%). Gaps: the sampling weights contradict token batching (issue 16). Bank prevalence and real prevalence differ, and the gate and θ are tuned against that mismatch (issue 10).
- **F1e. Real drafts are the main evaluation, perturbed drafts are analysis only: addressed in §4 and M9.** Gap: M7 weight calibration and M8 setting selection use bank drafts, and the draft type for the stage-0 check is not stated (issue 10).
- **F2a. `_compute_agent_targets` ROI and missing ≠ "no obstacle": partly addressed.** M6 is a new builder (360°, 80 m, 0–5 s, validity levels). Gaps: excluding UNKNOWN objects from a sum-over-objects loss behaves exactly like "absent". Objects beyond the 80 m radius are silently absent. The pre-entry rule needs explicit justification (issue 9). Red-light handling is now settled from the code (issue 18).
- **F2b. Circle and SDF surrogates vs the footprint: addressed.** M7 uses box-box distances, M8 checks at the pair level, and circles are an ablation only. Minor gaps: smoothmax bias and the human-margin mask (issue 17).
- **F2c. Final judgement by official total failures, new failures, PDMS and progress loss: NOT addressed in the decision rule.** M9 computes these, but §4-2 decides on NC∪DAC net fixes plus unnecessary modifications only, and puts PDMS and TTC under "not used for the decision" (issue 1).
- **F2d. R_GT-cur vs R_GT-fut distinction: addressed.** Gap: R_GT-cur contains the loss's own SDF plus map layers, so it is not a clean perception reference for a detection-only teacher (issue 12).
- **F3. Cautious wording about prior work: mostly addressed.** Five statements are too strong or inconsistent (issue 20).

---

**Issues**

**1. [blocker] The §4-2 decision rule leaves out what the user said must decide it.**
- The user named total failures, new failures, PDMS and progress loss. D1–D4 use net fixes (NC∪DAC) and unnecessary modifications only, and PDMS and TTC sit under "보조 보고 (판정에는 쓰지 않음)".
- Net fixes can hide many new failures; 32-B had 181 new failures. Mode-A braking can raise TTC failures, which D12 itself admits.
- Fix: rewrite the primary rule as all of the following, each R_T − R_S, with paired 136-log bootstrap and seed-averaged:
  - (P1) ΔPDMS (official scorer) with lower CI > 0, or non-inferiority plus P2, at the matched-ΔEP point.
  - (P2) Total failures NC∪DAC∪TTC with lower CI > 0.
  - (P3) New failures (NC, DAC, TTC, comfort) no worse than R_S + a pre-stated margin.
  - (P4) Progress loss ΔEP matched as in D2.
  - Keep NC∪DAC net fixes and unnecessary modifications as mechanism endpoints.
- Put all of this into `PRESTATED_DECISION_RULE.txt`.

**2. [blocker] The teacher is in-sample on TRAIN-OUT and the student is not (checked on disk).**
- `cache_train_50x100/manifest.json` has split=train, `navsim_infos_train.pkl`, and 103,288 samples, which is 85,109 + 18,179.
- `/home/external-user/yongjae/bevfusion/runs/navsim-fusion-50x100/configs.yaml` trains on `navsim_infos_train.pkl` with max_epochs 20.
- So R_T trains on teacher features the teacher has already seen and is tested on unseen navtest features. R_S sees unseen features in both. Only R_T faces this shift, so both a null result and a positive one are hard to read.
- The draft's evidence does not measure memorisation. The box-density numbers (TRAIN-OUT 9.0 vs navtest 10.3) are a scene-mix number. TRAIN-IN at 18.1 vs TRAIN-OUT at 9.0, both in-sample for the teacher, shows density follows the scene mix.
- Fix:
  - (a) In V1, measure teacher and student detection recall and precision against GT (score ≥ 0.3, 2 m match, vehicle and VRU, 0–32 m) on TRAIN-IN, TRAIN-OUT and navtest. Also report GT objects per scene for each split.
  - (b) Report each arm's inner-val → navtest drop, and add a pre-stated reading: "if R_T's drop exceeds R_S's by more than X, a null result is inconclusive."
  - (c) Look for nuPlan/OpenScene logs with sensors that are in neither the navtrain nor the navtest log lists, to get data unseen by both models. If there are none, state the asymmetry as a threat to validity and pre-register a mitigation. Example: teacher-feature dropout or noise during R_T training, at a fixed strength and not tuned on the teacher-seen TRAIN-OUT inner-val. That inner-val cannot calibrate the mitigation, so say so explicitly.

**3. [blocker] Nothing separates "aux-teacher information" from "any independent second model" or "KD as a regulariser".**
- R_T > R_S can happen simply because R_S reads the same features that produced the draft, so it shares the draft's blind spots.
- No independent camera-only student exists. `work_dirs/para_ssr_interaction_final` has only version_0/1/2 of seed 0. Kyungmin's `para_ssr_int_control_ft` is a fine-tune of the same seed, so it is not independent.
- Fix:
  - (a) Stage F: add R_S′, the same refiner on BEV features from an independently trained PARA-SSR (new seed, about 26 h × 2 GPU). The "teacher needed" claim needs R_T > R_S′, not only R_T > R_S. If R_S′ is not affordable, limit the claim in writing to "teacher features beat the features of the student that produced the draft".
  - (b) Stage E: add E2-self, which is KD from the frozen stage-F R_S under the same schedule. Pre-state the stage-E rule: E2 > E1, E2 > E2-self, plus a pre-stated reading of E2 vs E3.

**4. [major] Stage F shows the information is present, not that KD can move it into the student.**
- Kyungmin's readout KD found no transfer. In §6, E2 ≥ E3 is only prose, and the stage-F rule never relates R_T to R_GT-cur.
- Fix: add cheap stage-F arms on frozen student features:
  - R_S←KD(R_T): L1 to frozen R_T control points and gate, plus the surrogate loss.
  - R_S←KD(R_GT-cur).
  - R_S←self-KD, using R_S seed b.
- Pre-state how to read the outcomes:
  - R_S←KD(R_T) > R_S and ≥ R_S←KD(R_GT-cur): support for stage E.
  - R_S←KD(R_T) ≈ R_S: any stage-E gain must come through γ_bev > 0, and this becomes the named stage-E hypothesis.
  - R_S←KD(R_GT-cur) > R_S←KD(R_T): E3 is expected to beat E2, which argues against the teacher.

**5. [major] Normalisation in M2 favours one arm.**
- Per-cell LayerNorm without affine is almost the identity for student `bev_embed`, which is already LN output.
- For teacher features it distorts: they are post-ReLU and 55.6% zeros. Cells that are nearly empty get stretched to unit variance, and activation magnitude, which is likely objectness or confidence, is lost.
- Fix: use one per-channel z-score for both arms, with mean and std from TRAIN-OUT-train, optionally with log1p for teacher features. Alternatively, pre-register a two-option set {per-channel z, per-cell LN} and let each arm choose on inner-val with the same budget. Add a unit test for all-zero cells.

**6. [major] The M4 bound "|δa| ≲ max(A_DEC, A_UP)" is false for this clamped spline.**
- Knots are [0,0,0,0,0.8,1.6,2.4,3.2,4,4,4,4]. Derivative control points are Q_i = 3(c_{i+1} − c_i)/(u_{i+4} − u_{i+1}).
- With c_{i+1} − c_i = 0.8·u: Q_1 = 1.5u, Q_2..Q_4 = u, Q_5 = 1.5u, Q_6 = 3u.
- So braking reaches 3·A_DEC = 12 m/s² in the last 0.8 s and 6 m/s² in the first 1.6 s. This breaks the comfort limit (−4.05) and the "hard constraint" claim.
- Fix: parameterise the derivative directly. Set Q_i = A·tanh(z_i) for i = 1..6 with Q_0 = 0, and integrate c_{i+1} = c_i + Q_i·(u_{i+4} − u_{i+1})/3. By the convex-hull property, |δa| ≤ A exactly. Apply the L_lo/L_hi limits as a projection on c.
- Use the same parameterisation in M1. In V3, assert max |δa| ≤ A.

**7. [major] Mode B cannot undo the L-slow perturbations, so the "human is reachable" contract fails.**
- A_p = −0.8 from t_on = 0 gives Δv = −3.2 m/s, which is more than ΔV_ACC = 2.0.
- The distance deficit is 0.5·0.8·16 = 6.4 m, which is more than L_ext ≤ 5 m.
- Undoing it would also have to follow a constant-curvature extrapolation rather than the human path.
- Fix: require |A_p|·(4 − t_on) ≤ 0.9·ΔV_ACC and 0.5·|A_p|·(4 − t_on)² ≤ 0.9·min(5 m, v_end·1 s). At t_on = 0 this gives |A_p| ≤ 0.45. Alternatively, raise ΔV_ACC and L_ext together. Measure the error of undoing each family in V3 and drop instances above tolerance.

**8. [major] "Same decoder for perturbations" is not literally true.**
- The M1 families cannot be expressed in the 6-free-control-point basis: the t_on ramp with jerk ≤ 2, the α·max(0, ·) ignore-brake profile, the Lat step at s_on, and creep.
- The generator follows Γ_human (dense path up to 8 s), but the refiner follows Γ(τ0), the 8-point polyline. The chord error on curves is ℓ²κ/8, for example 0.63 m at ℓ = 10 m and κ = 0.05.
- L-creep starts on the centerline, so continuity at t0 is broken whenever the ego is offset laterally from the lane center.
- Fix:
  - Project each family's profile by least squares onto the same basis with c0 = c1 = 0 (and e0 = e1 = 0).
  - Generate creep as a Frenet offset from the centerline whose initial value and slope equal the ego's offset, decaying to zero.
  - Report the round-trip error per family and per speed bin in V3.
  - Report the share of tokens dropped for S_avail per speed bin.
  - Turn off Lat and Lat-small when S_8 < 3 m, to match the refiner's d ≡ 0.

**9. [major] "UNKNOWN is not treated as absent" has no effect on the loss.**
- For a collision loss summed over objects, leaving an object out is the same as saying it is absent. The use of `has_unknown` is never specified.
- Tracks outside the 80 m collection radius are silently absent, not UNKNOWN.
- Pre-entry objects are absent by design. This matches the scorer but needs an explicit statement.
- Fix: define UNKNOWN space as outside the annotation range of the GT ego at t, or outside 80 m from the origin, or t > 5 s. If a draft's footprint or TTC projection enters UNKNOWN space:
  - (i) cap the progress term at the UNKNOWN boundary, so slowing before it costs nothing;
  - (ii) treat its y_g = 0 as uncertain: drop it from gate negatives or down-weight it;
  - (iii) down-weight C_mod;
  - (iv) stratify all results by `has_unknown`.
- State that the official scorer cannot see unannotated obstacles either (a limitation). Add a sensitivity run where pre-entry objects are UNKNOWN.

**10. [major] Prevalence mismatch, selection on bank drafts, and inconsistent failure rates.**
- D23 and §7 say the real failure rate is 1.5–2.3%. That is NC only. The draft's own baseline (NC 279, DAC 813, total 1,524 of 12,146) means the gate label NC∪DAC has roughly 8–9% prevalence.
- The M7 calibration objective scores "real drafts and bank drafts", so selection partly depends on perturbed drafts. That goes against F1e.
- Fix:
  - Correct the rates.
  - Choose θ and (w_prog, w_mod) only on real student drafts from inner-val. Use the bank for training and as a constraint or diagnostic.
  - Inner-val is about 3.6k tokens with about 55 NC failures, which is thin, so use 5-fold log-level cross-fitting over all of TRAIN-OUT for selection.
  - Say explicitly that the stage-0 check (§4-2) uses real drafts.

**11. [major] The progress surrogate (M7-4) uses the wrong EP normaliser.**
- In `pdm_scorer.py` L173–183 of this v1 fork, EP = raw / max over {PDM-Closed, agent} of raw progress × multiplicative metrics, with a 0.1 m threshold, not 5 m. If PDM-Closed fails a multiplicative metric, the normaliser becomes the agent itself.
- The 5 m floor under-penalises stopping in low-progress scenes. That is where creep and stop-line NC cases occur, and there official EP can fall from 1 to 0, which is 5/12 of the weighted score.
- Fix: L_prog = normalised shortfall using max(P_pdm·1[PDM passes multiplicative metrics], P(τ1)), with a 0.1 m threshold that matches the scorer. Check it against official EP in M8.

**12. [major] R_GT-cur can reach the loss target directly, which weakens stage 0 and the gap ratio.**
- A_GTc includes the same DAC SDF that C_dac is computed from, plus map layers. The teacher is a detection model with 7 classes and no map.
- R_GT-cur > R_none can therefore pass on DAC alone, without showing the refiner uses objects. The gap-filled ratio also mixes map and object information.
- Fix: run stage 0 on NC separately, with R_GT-cur-obj vs R_none. Use R_GT-cur-obj as the denominator for the NC gap ratio and report DAC separately. Label R_GT-cur as "perfect current perception plus map, including the loss field".

**13. [major] Statistical power, and no "inconclusive" outcome.**
- The 32-D CI of +13 [−3, 30] means SE ≈ 8. A lower CI above 0 then needs an observed difference of about 16 or more, and the 80%-power MDE is about 23. A minimum effect of 10 cannot be detected under this rule.
- The results table maps every non-pass to "R_T ≈ R_S → stop".
- Fix: pre-state the expected SE from 32-D and inner-val. Use three outcomes:
  - pass;
  - equivalence (CI inside ±Δ_eq);
  - inconclusive, which triggers a pre-stated next step (more seeds, or cross-fit TRAIN-OUT scoring) rather than a stop.

**14. [major] Navtest has already shaped the design, which contradicts §0-2 #7.**
- These come from navtest analyses:
  - the family ranges (NC p95 +0.91 / p99 +1.27, 19 creep NC, |d4| 97.2%);
  - the D1 rationale (12.7%, 73);
  - D4 (NC p75 2.1 m);
  - the M6 motivation (18.6% of NC causes missed);
  - possibly the pilot SAT recall of 0.75, if that pool was navtest.
- Separately, the teacher config evaluates on `navsim_infos_val.pkl`, a symlink to `navsim_infos_val_navtest.pkl`, every epoch. There was no checkpoint selection (epoch_20 is the last epoch), but teacher development saw navtest.
- Fix: recompute every such statistic on TRAIN-OUT real drafts plus the scored navtrain pool from experiment E before freezing. Tag anything left as [navtest-derived]. List both points as limitations in §3-2.

**15. [minor] The gate label is "the draft fails", not "the correction helps".**
- 32-B shows corrections can fail to fix a draft or create new failures.
- Fix: optionally train the correction head in stage 1, then score its outputs on the TRAIN-OUT bank and train a stage-2 gate on y = 1[PDMS(τ1) > PDMS(τ0)]. Pre-register this as a variant.

**16. [minor] The 30/45/25% sampling weights cannot be applied to batches of "all 13 drafts per token".**
- Fix: state them as per-family loss weights.

**17. [minor] Two small biases in the collision surrogate.**
- The logsumexp smoothmax overestimates the gap by up to T·ln4 ≈ 0.07 m; subtract it.
- The human-margin mask (g_human < m_col) lets lateral corrections move into objects the human passed closely. Mask only where the human actually overlaps (g_human < 0).

**18. [minor] Close several open items with evidence.**
- Red light: `pdm_scorer.py` L343 (NC) and L516 (TTC) skip tokens containing `red_light_token`, so M6 needs no red-light objects. PDM-Closed still stops at red lights, which affects P_pdm.
- Teacher frame: `navsim_infos_val.pkl` has lidar2ego set to identity, so the teacher grid origin is the rear axle and there is no longitudinal offset.
- Box convention: the infos gt_boxes use (dx ≈ W, dy ≈ L, yaw ≈ heading − π/2); a forward vehicle at (15.7, −0.28) has dims (2.11, 5.11) and yaw −1.6. The footprint is unaffected but heading and velocity channels must be converted. Apply the same convention to the T, S and GT box rasters, otherwise R_T-box vs R_S-box vs R_GT-cur-obj confounds box quality with conventions.
- Unrelated cache: `/home/external-user/datasets/teacher_cache/bevfusion/cache_val_50x100_future` has 1,422 npz files, no manifest, and was created 09-27. Pin the A_T loader to the manifest checkpoint sha and path so it can never read this.

**19. [minor] Evaluation definitions.**
- D2: "작은 쪽의 ΔEP" has an ambiguous sign. Define ΔEP* = the smaller progress loss of the two arms at their inner-val θ, interpolate each arm's navtest θ-sweep curve at ΔEP*, and redo the interpolation inside every bootstrap replicate.
- Re-score the baseline decode(τ0, 0) in the same scorer run (NC 279 vs 278 csv).
- "Unnecessary modification" is nearly vacuous in mode A, because PDMS of a passing draft rarely rises. Define it as "modified a draft that passes NC, DAC and TTC" and report harm (ΔPDMS < 0) separately.
- Define "811 → 0" in M9 and D1; it is unclear as written.

**20. [minor] §6 wording.**
- Readout KD: that teacher was ReSMap (a map teacher), so the warning is indirect, not "직접적".
- Byounggun Stage 1: "never measured closed-loop" becomes "no PDM-scored measurement found in the audited material". NAVSIM PDMS is non-reactive, not closed-loop.
- Byounggun Stage 2: say what "±0.003" compares, given that no same-setting control exists.
- 32-B: 128 fixed with 181 new collisions is a net −53, which contradicts "순효과 약 12% [6, 18]". Restate the definitions and scope.
- DistillDrive: change "사실상" to "유사할 수 있음", and state that §6 is not a literature survey, so novelty claims need a separate check.

**21. [minor] Mode A reporting.**
- Report all metrics on the mode-A-fixable subset (NC/DAC failures where the draft is at or ahead of the human) and on the rest.
- Make the pre-registered B arm's decision role explicit: analysis only.

**22. [minor] Stage E KD in control-point space after clipping.**
- The cumulative clip gives zero gradient at the bounds (for example c = 0 in mode A). Distil the pre-clip derivative parameters Q (issue 6) or decoded poses.

**23. [minor] Stage E compute.**
- E0–E3 at 2 seeds is about 416 GPU-h. Adding E2-self brings it to about 520 GPU-h, and all 6 GPUs are busy.
- Pre-state what stage F alone can support (with the issue-4 arms) if stage E cannot run before 11-04.

Files checked:
- /home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100/manifest.json
- /home/external-user/datasets/teacher_cache/bevfusion/cache_val_50x100/manifest.json
- /home/external-user/yongjae/bevfusion/runs/navsim-fusion-50x100/configs.yaml
- /home/external-user/yongjae/bevfusion/data/infos/navsim_infos_val.pkl (symlink to navsim_infos_val_navtest.pkl)
- /home/external-user/yongjae/SSR/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py (L173–183, L343, L469–471, L516)
- /home/external-user/yongjae/SSR/work_dirs/para_ssr_interaction_final/lightning_logs/
- /home/external-user/datasets/teacher_cache/bevfusion/cache_val_50x100_future/