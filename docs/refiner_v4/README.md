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

## Not yet done

* Evidence-level KD (DET -> occupancy, MAP -> margin) needs v4 teachers with the evidence readout; KD is still on the
  final scores.
* I4 (one extension rule): the a+0.5 variant is still extended straight while the parent corridor follows its end
  curvature; on curved parents the sibling leaves the parent grid by up to ~0.4 m beyond S_8 (measured in
  tests/test_v4_model.py).
* Navtest time-index sidecars (train / val are built: CK/v4/time_idx, 0 consistency violations).
* Parent dedupe: `score_set` reads every candidate's parent corridor separately (96 rows for 16 parents in
  inference); a dedupe is exact (max diff 0) and would cut the parent-pass cost ~5x.  Cost today = CK2 (1.02x).
* Corrector edits are not in the headline eval (eval_e2e2 summary is 96-only); `tools/ck/v4/eval_edits.py` has no
  val-frozen beta for navtest.
* Review SSR/report 79 (verified 2026-10-16): fixed in v4_2 -- near-stop candidates read station 0 with their dwell
  + feasibility features (F1), `stage_infer_check` / r34 sanity check pass the parents (F2), exact leave-one-out credit
  (F3), evidence bins to 5.0 s (F6), AWR re-score in fp32 after an amp forward (F8), station / evidence aux terms
  restricted to the teacher's score keys (D(c)), head sizes matched to CK2 (F7 parameters).  Still open: the A credit
  scores critic-only while deployment blends v2_final / im (F5; needs im recorded per parent), edits not in the
  headline eval (F4), parent dedupe (F7 compute).

## Tests (CPU)

    nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q \
      -p no:cacheprovider tests/test_v4_model.py tests/test_v4_e2e_loss.py tests/test_v4_kd_family.py \
      tests/test_v4_time_idx.py tests/test_v4_eval_edits.py tests/test_v4_corrector_plumbing.py
