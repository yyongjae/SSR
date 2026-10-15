# refiner v4 (branch `refiner-v4`)

Base: `refiner-v3` + `refiner-fix` (B1 corridor version 2, default 1 bit-identical).  Design: SSR/report 70, 72-77
(75: v4 = critic + learned corrector A, 77: evidence-decomposed critic).  Everything is opt-in: with the defaults the
network, losses and outputs are the CK2 / refiner-v3 ones bit for bit (tests/test_v4_model.py::test_off_is_bit_identical),
except the **e2e default arm, which is now the ck2e2e_bevkd setting** (user 2026-10-15).

## What is in the branch

| Piece | Where | Switch |
|---|---|---|
| BEV KD from DET + MAP as the e2e default | `online2._d_bev_kd`, `tools/ck/e2e2/ck_e2e2.yaml` | `ARM=main` -> `main_arm.set` (old main) |
| Hazard readout (I1): scene read on each candidate's PARENT corridor (time-free stations), the candidate reads it at its own arrival time / lateral offset, per-key station hazard + noisy-OR, structured EP head, scene-free comfort head, hazard-localised lateral head | `ck/v4/readout.py`, `ck/v4/sibling.py`, `CKNet.score_set` | `v4.readout=hazard` / `--v4-readout hazard` |
| Evidence readout (77): predicted occupancy O [48 x 17 x 8 time bins] + drivable margin D on the parent grid, official-rule evidence (footprint / TTC projection / corners), zero-init residual | `ck/v4/evidence.py` | `v4.readout=evidence` |
| P_ref scene query (77 Q1): one query over the 200 scene tokens, Huber on log1p(pdm_progress_eff) | `ck/v4/readout.PRefHead` | `v4.pref_head=true` |
| Censored station BCE (first-failure time) | `ck/v4/losses.station_bce` | hazard + time-index sidecars (below) |
| Evidence aux (GT objects / official SDF on the parent grid) | `ck/v4/losses.evidence_aux` | evidence + `lambda_ev` |
| Parent-only score KD (I2) | `online2` loss | `v4.kd_parent_only=true` |
| Per-family KD calibration (I2) | `ck/kd_calib.py` (`params_by_family`), `tools/ck/v4/fit_kd_calib_family.py` | point `kd_calib.det/.map` at a family file |
| Learned corrector A (75): top-k shortlist -> 12 controls (mode B) -> re-score on the source's parent corridor; credit + AWR | `ck/v4/corrector.py`, `CKNet.correct_topk` | `v4.corrector=true` (training signal pending, below) |
| Corridor version 2 for the student | `ck_e2e2.v4.corridor_version` | `v4.corridor_version=2` |

## How to run (all GPU runs need the user's approval; GPUs 0-3 only)

Teacher (v4 readout, from scratch, 10 epochs like ck2T10 / ck2M10):

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

## Not yet done

* Corrector A training signal: the edits are produced and re-scored at inference, the credit / AWR loss exist and
  are tested, but the on-policy recorder / labeler / LabelStore2 do not yet carry the K edits (official labels of the
  edits are needed for the credit), and the eval tools (eval_e2e2 / eval_ck2) do not yet label / select over 96 + K.
  Until then a corrector student behaves as the rule pool (mu = 0 -> every edit is its source).
* Evidence-level KD (DET -> occupancy, MAP -> margin) needs v4 teachers with the evidence readout; KD is still on the
  final scores.
* The first-failure time-index sidecars (station BCE) are built by tools/ck/v4/build_time_idx.py (no re-scoring:
  the indices are in the existing label shards); the full builds are not run yet.
* I4 (one extension rule): the a+0.5 variant is still extended straight while the parent corridor follows its end
  curvature; on curved parents the sibling leaves the parent grid by up to ~0.4 m beyond S_8 (measured in
  tests/test_v4_model.py).

## Tests (CPU)

    nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q \
      -p no:cacheprovider tests/test_v4_model.py tests/test_v4_e2e_loss.py tests/test_v4_kd_family.py \
      tests/test_v4_time_idx.py
