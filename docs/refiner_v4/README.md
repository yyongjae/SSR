# refiner v4 (branch `refiner-v4`)

Base: `refiner-v3` + `refiner-fix` (B1 corridor version 2, default 1 bit-identical).  Design: SSR/report 70, 72-77
(75: v4 = critic + learned corrector A, 77: evidence-decomposed critic).  Everything is opt-in: with the defaults the
network, losses and outputs are the CK2 / refiner-v3 ones bit for bit (tests/test_v4_model.py::test_off_is_bit_identical),
except the **e2e default arm, which is now the ck2e2e_bevkd setting** (user 2026-10-15).

## What is in the branch

| Piece | Where | Switch |
|---|---|---|
| BEV KD from DET + MAP as the e2e default | `online2._d_bev_kd`, `tools/ck/e2e2/ck_e2e2.yaml` | `ARM=main` -> `main_arm.set` (old main) |
| Hazard readout (I1): scene read on each candidate's PARENT corridor (time-free stations), the candidate reads every station bin it enters within 4 s with its entry time, dwell, speed, lateral offset (station 0 always: no candidate reads nothing) + its feasibility from the ego state, per-key station hazard + noisy-OR, structured EP head, scene-free comfort head, hazard-localised lateral head | `ck/v4/readout.py`, `ck/v4/sibling.py`, `CKNet.score_set` | `v4.readout=hazard` / `--v4-readout hazard` |
| Evidence readout (77): predicted occupancy O [48 x 17 x 10 time bins, to 5.0 s] + drivable margin D on the parent grid, official-rule-shaped evidence (footprint / TTC projection / corners; the differences from the scorer are listed in `evidence.py`), zero-init residual | `ck/v4/evidence.py` | `v4.readout=evidence` |
| P_ref scene query (77 Q1): one query over the 200 scene tokens, Huber on log1p(pdm_progress_eff) | `ck/v4/readout.PRefHead` | `v4.pref_head=true` |
| Censored station BCE (first-failure time -> the station bin the candidate is in at that time; specialist teachers: only their own score keys) | `ck/v4/losses.station_bce` | hazard + time-index sidecars (below) |
| Evidence aux (GT objects / official SDF on the parent grid) | `ck/v4/losses.evidence_aux` | evidence + `lambda_ev` |
| Parent-only score KD (I2) | `online2` loss | `v4.kd_parent_only=true` |
| Per-family KD calibration (I2) | `ck/kd_calib.py` (`params_by_family`), `tools/ck/v4/fit_kd_calib_family.py` | point `kd_calib.det/.map` at a family file |
| Learned corrector A (75): top-k shortlist -> 12 controls (mode B) -> re-score on the source's parent corridor; credit + AWR | `ck/v4/corrector.py`, `CKNet.correct_topk` | `v4.corrector=true` (training signal pending, below) |
| Corridor version 2 for the student | `ck_e2e2.v4.corridor_version` | `v4.corridor_version=2` |

Parameter budget: the default head sizes give the hazard + P_ref (and evidence + P_ref) critic the trainable parameter
count of the CK2 critic (3,391,755 as trained; v4 3,391,340 / 3,391,216, tests/test_v4_review_fixes.py), so v4 vs CK2 is
not a capacity comparison.  The corrector A (221,208) is a generator on top and is not part of that budget.

## How to run (all GPU runs need the user's approval; GPUs 0-3 only)

Teachers (DET then MAP, ck2T10 / ck2M10 recipe + hazard + P_ref + corridor v2 + station BCE), the launcher:

    V4_USER_APPROVED=1 V4_GPUS=2,3 bash tools/ck/v4/run_teachers.sh both     # -> CK/v4/train/v4T10_haz, v4M10_haz

or by hand (v4 readout, from scratch, 10 epochs like ck2T10 / ck2M10):

    cd /workspace/yongjae/SSR-v4 && CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PWD torchrun --nproc_per_node=2 \
      tools/ck/train_ck2.py --arm T --run-name v4T10_haz --epochs 10 --tokens-per-batch 16 --lr 6e-4 \
      --sur-groups near_mid --eval-every 2 --best-metric none --score-keys nc,ttc,ep,comfort \
      --v4-readout hazard --v4-pref-head 1 --corridor-version 2 [--time-idx-root <sidecars>] --resume
    # MAP: --arm M --score-keys dac,ep (GPUs 2,3); evidence readout: --v4-readout evidence

Family KD calibration of a teacher (GPU inference, CPU fit; a finished student generation as data):

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python tools/ck/v4/fit_kd_calib_family.py --run <teacher> \
      --gen /home/external-user/ssd/yongjae_refiner/ck/ck2/e2e/ck2e2e_main/ck_e2e2/lab/ep028 --n-tokens 16384

Student:

    RUN=<run> CK2_E2E_APPROVED=1 ARM=v4 bash tools/ck/e2e2/train_e2e2.sh start --gpus 0,1
    # v4_arm.set: hazard readout + P_ref head + corridor v2 + parent-only KD (BEV KD on by default);
    # E2E_CK_SET=<file> to point teacher_det_run / teacher_map_run / kd_calib at v4 teachers / family files,
    # v4.readout=evidence, v4.corrector=true

## Corrector A data path

on-policy step: KD set (top-16 + sampled variants) -> critic-only score -> top 8 -> corrector samples 12 controls ->
decoder mode B -> edits recorded with the 96 (rec2 edit block) -> labeler2 scores 96 + 8 in one official call ->
generation edit_* files -> next epoch: LabelStore2(with_edits) -> the edits join the critic's BCE label set (parents =
their source's identity) and the corrector gets AWR with the report-61 credit (A_sel over the label set + edits with the
current critic scores, A_loc = edit PDMS - source PDMS).  Inference: shortlist of the 96 -> mu edits -> re-scored ->
eval_e2e2 dumps / labels them, tools/ck/v4/eval_edits.py compares selection over 96 + 8 with 96.

## v4_3: allowed-range critic + computed edits + KD routing (docs/refiner_v4/V4_3_SPEC.md)

Design and user decisions (2026-10-16 KST) in the spec; nothing is trained until the Phase 0 oracle passes and the user
approves.  Every option below defaults OFF (outputs / state dicts of the older arms unchanged, tests/test_v4_model.py
::test_off_is_bit_identical).  The e2e knobs are OPTIONAL keys of the ck_e2e2 v4 block (online2._d_v4_opt): absent from
ck_e2e2.yaml and the defaults, so the older arms' effective configs and override lists are unchanged; a set file that
names one declares it (launch_util2.apply_sets2).  With the v4_3 arm (tools/ck/e2e2/v4_3_arm.set):

| Piece | Where | Switch |
|---|---|---|
| Official LQR tracker port (M1), GT range fields + reductions (M2), parent dedupe (M3) | `ck/v4/lqr.py`, `ck/v4/range_targets.py`, `CKNet.score_set` | always available / dedupe on by default (exact) |
| Range readout (M4): B / U blocked-field logits + margin D per unique parent, read at the executed states | `ck/v4/range_readout.py` | `v4.readout=range` (+ `v4.exec_model=lqr`, `v4.ev_tbins=20`, `v4.ev_tbin_s=0.25`, `v4.ep_evidence=true`) |
| Computed edits (M5): rule edits of the critic's top-8 parents on the predicted fields, EXTRA candidates only | `ck/v4/edits.py`, `CKNet.computed_edits`; recorded by `online2.CKE2E2.computed_edits_rec` | `v4.edits=computed` (`edit_k` 8 = the recorded edit block) |
| Score KD routing (M6): nc <- DET, dac <- MAP only | `online2` loss | `kd_score_keys=[nc,dac]` |
| Score KD scope (M6): NC KD where the nearest lead's ttc0 < 4 s, DAC KD where the parent's min executed corner SDF < 1 m (ref_* GT, train only; missing ref_* keys = ValueError). Each key's KL is the mean over its own mask: its total weight stays that of scope `all`, so a near candidate weighs ~1 / near-fraction more | `online2.lead_ttc0`, `parent_min_corner_sdf`, `score_kd_near` | `v4.kd_score_scope=near` (default `all`) |
| Constraint KD (M6): DET teacher sigmoid(B) -> student BCE, MAP teacher D -> student Huber (s <= 32 m), on the student's unique parents of the KD set, masked by teacher ok | `online2.constraint_kd`, `TeacherPair2(fields=True)` | `v4.kd_constraint.enabled=true` (`lambda_occ` 0.5, `lambda_margin` 0.5, `map_range_m` 32, `occ_channels` B = spec; `BU` also distils the TTC channel U, an option the user has not confirmed) |
| Range teachers: GT field aux online from ref_* (DET: blocked field B + the TTC channel U, BCE on known bins; MAP: SDF margin Huber), specialist key masking as v4_2 | `tools/ck/train_ck2.py range_gt_aux` | `--v4-readout range` (+ `--v4-exec-model`, `--v4-ev-tbin-s`, `--v4-ep-evidence`, `--v4-vmax`, `--rg-max-parents` 8, weight `--lambda-ev`) |
| 96 + K headline: selection over the 96 + K edits, beta / variant set chosen on navtrain_val and frozen for navtest; 96-only rows on the same tokens | `tools/ck/e2e2/eval_ck2.py` ('Edits'), `eval_e2e2.py` | automatic when a dump holds edits |

Data path of the computed edits: on-policy step -> student forward over KD set + label set (return_pctx) -> computed
edits of the KD set's top-8 parents (critic-only shortlist score: the deployment beta is chosen at evaluation) -> rec2
edit block (valid = a real edit of an eligible parent; no-edit slots are their source and are not re-scored) ->
labeler2 scores 96 + 8 in one official call -> generation edit_* files -> next epoch LabelStore2(with_edits) -> the
edits join the critic's BCE label set with their official labels (parents = their source's identity).  No AWR without
the corrector (`v4.corrector=true` adds its zero-init residual on the computed edit + AWR, off in the first run).
Inference: the same rule on the 96 pool -> `ck2_edit_*` (+ `ck2_edit_valid`: filler slots and, without the corrector,
no-edit slots (kind 0 = the source column bitwise) excluded, `ck2_edit_kind`) -> eval_e2e2 dump / label / infer_check /
eval.  A selectable no-edit copy would tie its source only up to GEMM batch-shape noise on GPU and, when picked, drop
the lateral head (an edit is submitted as is); eval_ck2 also refuses any edit that equals its source pool column bitwise
(infer/edit_traj vs packed/cand96, counted `n_edit_dup`).  `infer_select` (the agent's ck2_traj) needs an explicit
`edits: true` (96 + K, the same `select96_edits` rule as eval_ck2) or `edits: false` (96 only) for a student with edits.

Verified (CPU, 2026-10-17 KST): `lead_ttc0` on the GT loader's ref_* vs CK/lead/navtrain_train.parquet (the
lead_split.py rule), 1,988 navtrain_train tokens (half with a parquet lead closer than 10 s): near flag (ttc0 < 4 s)
agreement 0.9980, recall 0.9933 (589 / 593), precision 1.000; |d ttc0| median 0, p90 0.002 s over the 1,374 closing
leads of both; 13 ms / token on 1 CPU thread.  train_ck2 --v4-readout range, 3 CPU steps on 8 real tokens (2 per step,
--time-idx-root): DET 18 s, MAP 16 s, finite losses (DET rg_occ_bce ~1.2, B cell rate ~0.22 / U ~0.31 on the anchors'
own grids; MAP rg_margin_huber 0.7 - 1.4).

Teachers (range; do not launch without approval):

    V4_USER_APPROVED=1 V4_GPUS=2,3 READOUT=range bash tools/ck/v4/run_teachers.sh both   # -> CK/v4/train/v4T10_rng, v4M10_rng
    V4_DRY_RUN=1 READOUT=range bash tools/ck/v4/run_teachers.sh both                     # print the arguments only

Student (after the teachers and their KD calibration on nc / dac exist; the arm file's teacher / calibration paths are
placeholders that launch_util2 refuses):

    printf '%s\n' teacher_det_run=<v4T10_rng run> teacher_map_run=<v4M10_rng run> \
      kd_calib.det=<DET calib json> kd_calib.map=<MAP calib json> > /tmp/v4_3_teachers.set
    RUN=<run> CK2_E2E_APPROVED=1 ARM=v4_3 E2E_CK_SET=/tmp/v4_3_teachers.set bash tools/ck/e2e2/train_e2e2.sh start --gpus 0,1

Evaluation: `tools/ck/e2e2/eval_e2e2.py --stage all` on navtrain_val, then navtest (`--fix-from` default = the val
metrics.json): metrics.json `best` (96), `best_edits` (96 + K, the headline / representative row), `edits` (K, usable /
unlabelled edits, the paired difference vs the 96 row of the same beta / set / lateral mode and vs the 96 choice).

## Not yet done

* Evidence-level KD (DET -> occupancy, MAP -> margin) needs v4 teachers with the evidence readout; KD is still on the
  final scores.  (v4_3: the constraint KD distils the range teachers' fields.)
* I4 (one extension rule): the a+0.5 variant is still extended straight while the parent corridor follows its end
  curvature; on curved parents the sibling leaves the parent grid by up to ~0.4 m beyond S_8 (measured in
  tests/test_v4_model.py).
* Navtest time-index sidecars (train / val are built: CK/v4/time_idx, 0 consistency violations).
* v4_3: the range student has no GT field aux of its own (its fields learn from the constraint KD and the label BCE);
  the computed-edit shortlist is critic-only in training and at the dump (the val beta is not known then);
  ck_e2e2.infer_select.edits true selects over 96 + K, but a standard NAVSIM submission still submits
  predictions['trajectory'] (v2's, eval_e2e2 docstring); tools/ck/v4/eval_edits.py does not read edit_valid and
  would treat filler / no-edit slots as candidates (eval_ck2's 96 + K rows are the reference); the constraint KD
  distils B only (spec M6; `occ_channels=BU` also distils U -- not confirmed by the user).
* Review SSR/report 79 (verified 2026-10-16): fixed in v4_2 -- near-stop candidates read station 0 with their dwell
  + feasibility features (F1), `stage_infer_check` / r34 sanity check pass the parents (F2), exact leave-one-out credit
  (F3), evidence bins to 5.0 s (F6), AWR re-score in fp32 after an amp forward (F8), station / evidence aux terms
  restricted to the teacher's score keys (D(c)), head sizes matched to CK2 (F7 parameters); in v4_3 -- parent dedupe
  (F7 compute, spec M3) and edits in the headline eval (F4: eval_ck2 96 + K with the val-frozen beta).  Still open: the
  A credit scores critic-only while deployment blends v2_final / im (F5; needs im recorded per parent).

## Tests (CPU)

    nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q \
      -p no:cacheprovider tests/test_v4_model.py tests/test_v4_e2e_loss.py tests/test_v4_kd_family.py \
      tests/test_v4_time_idx.py tests/test_v4_eval_edits.py tests/test_v4_corrector_plumbing.py
    # v4_3 (M1-M6): + tests/test_v4_lqr.py tests/test_v4_range_targets.py tests/test_v4_dedupe.py \
    #   tests/test_v4_range_readout.py tests/test_v4_edits.py tests/test_v4_gt_splits.py tests/test_v4_wiring.py
