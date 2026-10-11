# refiner-v3 Step 1: run book

Written 2026-10-11 18:05 KST by the review fixer.

**Launch only with the user's approval.** Each training command needs `V3_USER_APPROVED=1`. Memory rule: never start an extra run (another seed, a longer schedule, a control) without asking.

Tags: [실측] = measured (the log or file is named), [추정] = estimate. `D3 = /home/external-user/ssd/yongjae_refiner/ck/v3`.
Times are in KST.

## 1. What runs, where, and how long

| order | stage | what | GPUs | wall time | output |
|---|---|---|---|---|---|
| 1 | `pre` | forecaster pre-training: forecast loss only, navtrain_train, 8 epochs × 1,330 steps, batch 64 | GPU 0, 1 process | **≈ 10–15 min** [추정 from 0.04 s/step and a 60-step smoke, plus per-epoch evals of 34 k rows] | `D3/train/v3fc_pre_learned6/` |
| 2 | `a` | learned arm: critic with the pre-trained forecaster **frozen**, ck2T10 recipe, 10 epochs, 53,190 steps | GPUs 0,1, 2-process DDP, 8 tokens / rank | **≈ 4.0 h** [추정: ck2T10 9,197 s training × 1.41 + 921 s eval × ~1.4] | `D3/train/v3T10_a_learned/` |
| 3 | `b` | analytic CA arm, same branch | GPUs 0,1, DDP | **≈ 3.7 h** [추정: × 1.33] | `D3/train/v3T10_b_ca/` |
| 4 (optional) | `d` | GT-future oracle (PRIVILEGED, analysis only) | GPUs 0,1, DDP | ≈ 3.7 h [추정] | `D3/train/v3T10_d_oracle_PRIV/` |
| 5 | infer | val r34 top-16 + 80 variants per arm (c is already done) | GPU 0 | ≈ 3 min per arm [실측: ck2T10 took 156 s] | `D3/infer/<run>/` |
| 6 | decide | GO rule | CPU | ≈ 1–2 min [추정] | `D3/decide/step1.json` |

Total for pre + a + b + infer + decide is about 8 h. With d it is about 11.7 h.

**2-GPU DDP, one arm after the other.** One arm per GPU is not recommended:
- With world 1 and 16 tokens on one rank, that GPU needs about twice the activations of the ck2T10 setup (it used 7.49 GB/GPU at 8 tokens/rank). Each step would also take about twice as long, so two parallel arms would take about 7 h instead of 8 h sequentially. [추정]
- The world size would differ from ck2T10's recipe (SPEC §7).

The smoke times per step [실측 `D3/smoke/gpu01/summary.json`, 60 steps, 8 tokens/rank, world 2]:

| | off | a (frozen) | b |
|---|---|---|---|
| s/step | 0.168 | 0.237 | 0.224 |
| peak GB / GPU | 5.75 | 6.86 | 6.86 |

## 2. Launch commands (exact)

```bash
cd /workspace/yongjae/SSR-v3
V3_USER_APPROVED=1 bash tools/ck/v3/run_step1.sh pre      # GPU 0; wait for D3/train/v3fc_pre_learned6/done.json
# check the pre stage (section 3) before launching a
V3_USER_APPROVED=1 bash tools/ck/v3/run_step1.sh a        # GPUs 0,1; ~4.0 h
V3_USER_APPROVED=1 bash tools/ck/v3/run_step1.sh b        # GPUs 0,1; after a has finished (~3.7 h)
V3_USER_APPROVED=1 bash tools/ck/v3/run_step1.sh d        # optional, PRIVILEGED
```

`run_step1.sh` refuses to start unless all of these hold:
- the GPUs it will use are idle (< 1 GB);
- the K96 train and val packs are complete;
- `D3/ca_scales.json` exists and matches the train pack (`pack_sha256`, source bevfusion, k_obj 96, augmented fit);
- for `a`: stage `pre` has finished.

Before each launch it writes `D3/train/<run>.code.tgz` and `<run>.launch.json` (the tarball sha256 + provenance). It then starts the run under `setsid nohup` and writes `<run>.{launch.log,pid,pgid}`. Rerunning the same command resumes the run (`--resume`).

To stop a run: `kill -TERM -- -$(cat D3/train/<run>.pgid)`.

The command lines the launcher renders (`tools/ck/v3/arms.py cli --arm <stage>`), verbatim:

```bash
# pre (CUDA_VISIBLE_DEVICES=0, python -B tools/ck/v3/pretrain_forecaster.py)
--out /home/external-user/ssd/yongjae_refiner/ck/v3/train/v3fc_pre_learned6 --epochs 8 --batch 64 --lr 5e-4 --wd 0.01
--warmup-steps 500 --clip 1.0 --seed 0 --n-modes 6 --frame-drop 0.1 --obj-drop 0.05 --vnoise 0.3
--pack-root /home/external-user/ssd/yongjae_refiner/ck/v3/objpack_k96
--ca-scales /home/external-user/ssd/yongjae_refiner/ck/v3/ca_scales.json --train-split navtrain_train
--eval-splits navtrain_val,navtest --train-eval-rows 4096 --workers 8 --log-every 100 --resume
# a (CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tools/ck/train_ck2.py)
--arm T --epochs 10 --tokens-per-batch 16 --lr 6e-4 --sur-groups near_mid --eval-every 2 --best-metric none
--score-keys nc,ttc,ep,comfort --n-var 80 --var-name var_separate_sampler16_accstraight --seed 0 --sampler-seed 0
--norm-from compute --warmup-steps 500 --wd 0.01 --clip 1.0 --workers 8 --ep-target official
--out-root /home/external-user/ssd/yongjae_refiner/ck/v3/train --run-name v3T10_a_learned --temporal 1
--t-source bevfusion --t-forecaster learned --t-modes 6 --t-lambda-fc 1.0 --t-fc-backprop 0 --t-frame-drop 0.1
--t-obj-drop 0.05 --t-vnoise 0.3 --t-pack-root /home/external-user/ssd/yongjae_refiner/ck/v3/objpack_k96
--t-ca-scales /home/external-user/ssd/yongjae_refiner/ck/v3/ca_scales.json
--t-fc-init /home/external-user/ssd/yongjae_refiner/ck/v3/train/v3fc_pre_learned6/forecaster.pt --t-fc-train frozen
--resume
# b: as a, with --run-name v3T10_b_ca --t-forecaster ca --t-modes 1, without --t-fc-init
# d: as b, with --run-name v3T10_d_oracle_PRIV --t-forecaster oracle_gt
```

What to watch while a run is going:
- `tail -f D3/train/<run>.launch.log`
- `D3/train/<run>/train_log.jsonl`: `cand_per_s`, `gpu_mem_gb`, `n_skip_*`, `t_n_obj`, `t_n_valid`, and `fc_*` for a
- `val_metrics.jsonl`, written every 2 epochs
- `done.json` when the run has finished

## 3. Before launching (a): check the pre stage

Read `D3/train/v3fc_pre_learned6/val_metrics.jsonl`, last epoch:

| split / entry | what to look for |
|---|---|
| `navtrain_val` | learned vs CA `fc_minade` (`ade_ratio_learned_over_ca` < 1 expected) |
| `navtest` | the same ratio, on out-of-sample BEVFusion inputs. If it is clearly worse than on val, the learned forecaster may be using in-sample velocity (review causality finding 2). Report it next to G1. |
| `overfit_index` | (ratio on val) / (ratio on the in-sample train subset). Flagged above 1.10. If it is flagged, tell the user before arm (a): the critic of (a) would see better-than-deployable forecasts during training. |

The pre stage selects nothing: `forecaster.pt` is the last epoch. Arm (a) checks its n_modes. decide checks that it was pre-trained on the same train pack.

## 4. Evaluation and the GO decision

```bash
cd /workspace/yongjae/SSR-v3
E="PYTHONPATH=/workspace/yongjae/SSR-v3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1"
D3=/home/external-user/ssd/yongjae_refiner/ck/v3
# arm (c) is DONE (D8): D3/infer/ck2T10 (check ok, |d pdms_ck_noim| = 0)
env $E CUDA_VISIBLE_DEVICES=0 nice -n 10 /venv/ssr/bin/python -B tools/ck/v3/infer_val_v3.py --run $D3/train/v3T10_a_learned
env $E CUDA_VISIBLE_DEVICES=0 nice -n 10 /venv/ssr/bin/python -B tools/ck/v3/infer_val_v3.py --run $D3/train/v3T10_b_ca
# optional: ... --run $D3/train/v3T10_d_oracle_PRIV --allow-privileged
env $E CUDA_VISIBLE_DEVICES= nice -n 10 /venv/ssr/bin/python -B tools/ck/v3/decide_step1.py \
  --a $D3/infer/v3T10_a_learned --b $D3/infer/v3T10_b_ca --c $D3/infer/ck2T10 [--d $D3/infer/v3T10_d_oracle_PRIV]
# -> D3/decide/step1.json + a summary on stdout
```

Each infer writes `infer_meta.json['check']`, the per-token logit check against the trainer's last val eval (q99.9 of |Δlogit| ≤ 2e-4, at most 0.1 % flipped picks). A failing check does not exit. It is reported, and decide then refuses the arm.

decide refuses an arm in these cases:
- the arm's meta fails its checks (split, ckpt_last, candidate and variant shas, forecaster, privilege, `check.ok`);
- (a), (b) and (d) differ in training code, K, the train or val pack, or the CA scales;
- (a)'s forecaster was pre-trained on another train pack;
- an arm's inference read a different val pack than it was trained with.

`--no-arm-check` gives a diagnostic run that is marked `INVALID`.

**GO iff G1–G4 all pass** (pre-registered; main-session decision D4):

| gate | rule | also reported |
|---|---|---|
| G1 | Δ PDMS (a − b) ≥ +0.20 and log-cluster CI lower bound > 0 | Full-history subset and the split by past-frame cache source (current vs `_future` cache). A descriptive NC/TTC-only variant (DAC / EP / C frozen from ck2T). |
| G2 | lead-decel TTC AUC of (a) ≥ .855 | (a), (b), (c) values; (a) − (b) and (a) − (c) with CIs |
| G3 | lead-decel NC\|TTC fail of (a) ≤ 6.97 % | the same |
| G4 | non-inferiority to (c): pref(a) − pref(c) ≥ −0.01 and CI lower bound ≥ −0.03 | (a) − (b) |

Arm (c)'s own values for reference [실측: a diagnostic `decide_step1.py --no-arm-check` with a = ck2T, b = c = ck2T10 →
`D3/fix/decide_diag_ck2T_vs_ck2T10.json`, INVALID by construction]:

| metric | ck2T10 = (c) | gate |
|---|---|---|
| PDMS (frozen DAC) | 90.919 | – |
| lead-decel TTC AUC | .8369 | G2 needs ≥ .855 |
| lead-decel NC\|TTC fail | 6.17 % | G3 needs ≤ 6.97 % (already met by (c)) |
| a-1.0 hazard preference | .6526 | G4 baseline |

So G3 is not a strong gate on its own. The G2 and G3 deltas against (b) and (c) are the informative part.

Caveats to state with the result:
- **One seed per arm.** The G1 CI covers log and token sampling, not training-seed noise.
- If |Δ − 0.20| ≤ 0.15, decide sets `gates.G1.second_seed_required`. Then a second seed of (a) and (b) is run (with the user's approval) before the decision is final.
- **BEVFusion is in-sample on navtrain_val.** Val gains can include in-sample velocity information. Navtest stays descriptive; the stage-pre navtest ratio is the forecaster-level check.
- **(b) − (c)** is expected to be about +0.38 [추정]. If its CI lower bound is ≤ 0, decide notes that the branch itself may be broken.

## 5. Optional descriptive checks (no new training)

- **Seed or epoch sensitivity:** both a and b keep `ckpt_ep7.pt` and `ckpt_ep8.pt`. `infer_val_v3.py --run <run> --which ep7 --out $D3/infer/<run>_ep7` (and `ep8`) takes about 3 min each on GPU 0; give a separate `--out`, or it overwrites the `last` dump. Their outputs are descriptive only: decide accepts only `which last`, and the reproduce check is expected to say not ok (the saved logits are those of the last eval).
- **Joint fine-tuning** (`--t-fc-train joint`): the code path exists and is smoked. It would be a new arm, so it needs approval.

## 6. Verified before hand-off (checklist)

Data and inputs:
- [x] K_OBJ = 96 packs `D3/objpack_k96/bevfusion/{navtrain_train, navtrain_val, navtest}` are complete and verified. Train truncation is 0.575 % (rule ≤ 1 %). [실측 build.json, verify.json]
- [x] GT objects are read only from `D3/gt_objects` (val copy, train link, navtest built and validated against the metric cache), with per-file sha manifests. [실측]
- [x] `D3/ca_scales.json` is fitted on the augmented train stream (epochs 0–9, seed 0). It records source, K and aug, and its `pack_sha256` equals the train pack. [실측]
- [x] Arm (c) dump `D3/infer/ck2T10`: check ok, pdms_ck_noim 0.9032467603683472, |Δ| = 0. [실측 `D3/logs/infer_ck2T10.log`]

Code and configuration:
- [x] `arms.py check` shows no recipe mismatch against ck2T10's config. The arms differ only in forecaster, n_modes and fc_init (a). Stage pre matches arm a's augmentation, n_modes, packs and the recipe seed. [실측 tests/test_v3_arms.py]
- [x] Default OFF is bit-identical to 8ae8b1d: 1,151 arrays, 736 batch hashes, max |Δ| 0. [실측 `D3/fix/off_equiv/compare.json`, test_v3_integration_off]
- [x] v3 tests: 133 passed. [실측 `D3/logs/fix_v3_tests.log`]
- [x] Existing CK2 tests: 238 passed, 1 failed. The failure is the known kd_ramp_epochs one that predates v3 and also fails in the reference tree. [실측 `D3/logs/fix_existing_tests.log`]
- [x] Frozen arm a: weights are loaded strictly and unchanged after training, with one optimiser group, no forecast-loss term, and the trainable count equal to b's (3,629,131). [실측 test + smokes]

Smokes:
- [x] Real Step-1-setup smoke on GPUs 0,1 (`smoke_v3.sh gpu01 pre a b off`):
  - pre: 60 steps, 0.04 s/step, 3.13 GB.
  - a, b, off: 60 steps each, world 2. Every val split is finite. Skipped steps (GradScaler) 1 / 2 / 2.
  - The whole smoke took 99 s. [실측]
- [x] Infer on the 2-GPU smoke runs (64 rows, GPU 0): logit check ok, max |Δlogit| 0, 0 flips. [실측 `D3/smoke/infer/gpu01_{a,b}.log`]

Still open for the main session or the user:
- Approve the launch.
- Check the stage-pre diagnostics (§3).
- Run a second seed if G1 lands in the 0.05–0.35 band.
- Not fixed: R1.3, R2.5, R2.6, R2.8, R3.11 (REVIEW_FIXES.md §3).
