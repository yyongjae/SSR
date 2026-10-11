# refiner-v3 Step 1: run book

Written 2026-10-11 18:05 KST by the review fixer. Updated 2026-10-12 00:30 KST (tooling fixes F1–F4, `REVIEW_FIXES.md` §5): 단일 seed 결정 확정, infer cache key, pre-training resume 검사, object-pack part key. Updated 2026-10-12 01:10 KST (F1–F4 리뷰 후속, `REVIEW_FIXES.md` §6): arm 간 forward 코드 검사, 학습 완료(done.json) 검사, 로드 전후 ckpt sha 검사, `--which`별 기본 출력 폴더, 사전등록 외 bootstrap = INVALID, arm (b) 재개 절차 수정.

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
# 0. arm (b) must have finished: infer refuses --which last without done.json
ls $D3/train/v3T10_a_learned/done.json $D3/train/v3T10_b_ca/done.json
# 1. forward-code sha before the two infers (decide refuses a / b scored by different code); 2026-10-12 01:05 KST:
#    3475c5a4ffc2a8e2 (40 files; only infer_val_v3.py differs from HEAD)
env $E CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -B -c \
  "from tools.ck.v3 import infer_val_v3 as IV; print(IV.forward_code()['sha256'][:16])"
# 2. a and b back to back on GPU 0, default --out (D3/infer/<run>); arm (c) is DONE (D8): D3/infer/ck2T10
env $E CUDA_VISIBLE_DEVICES=0 nice -n 10 /venv/ssr/bin/python -B tools/ck/v3/infer_val_v3.py --run $D3/train/v3T10_a_learned
env $E CUDA_VISIBLE_DEVICES=0 nice -n 10 /venv/ssr/bin/python -B tools/ck/v3/infer_val_v3.py --run $D3/train/v3T10_b_ca
# optional: ... --run $D3/train/v3T10_d_oracle_PRIV --allow-privileged
# 3. decide (CPU, pre-registered n_boot 2000 / seed 0: do not pass --n-boot / --seed)
env $E CUDA_VISIBLE_DEVICES= nice -n 10 /venv/ssr/bin/python -B tools/ck/v3/decide_step1.py \
  --a $D3/infer/v3T10_a_learned --b $D3/infer/v3T10_b_ca --c $D3/infer/ck2T10 [--d $D3/infer/v3T10_d_oracle_PRIV]
# -> D3/decide/step1.json + a summary on stdout
```

Smoke(`--limit`)는 반드시 `/tmp` 아래 별도 `--out`에 하세요. CPU smoke를 기본 폴더에 하면 cache key의 정밀도(`amp`)가 달라 이후 GPU 본 실행이 거부됩니다(폴더를 지워야 함). GPU `--limit` smoke라면 같은 키로 이어서 계산되고 forecast 합계도 합쳐지지만(`REVIEW_FIXES.md` §6 finding 5), 결정 덤프는 한 세션에서 만드는 편이 단순합니다.

Each infer writes `infer_meta.json['check']`, the per-token logit check against the trainer's last val eval (q99.9 of |Δlogit| ≤ 2e-4, at most 0.1 % flipped picks). A failing check does not exit. It is reported, and decide then refuses the arm.

**Cache key (F2, 2026-10-11).** infer는 행을 계산하기 전에 `<out>/cache_key.json`을 씁니다. 키에는 ckpt sha256(모델 로드 전에 해시), which, packed tokens / ok / cand / status sha, variants sha, temporal cfg, privileged, 정밀도(fp16 autocast / fp32 cpu), val pack · ca_scales · fc_init sha, forward 코드 sha가 들어갑니다. 같은 `--out`에 done 행이 이미 있는데 키가 다르거나 `cache_key.json`이 없으면(F2 이전 덤프) infer는 아무것도 쓰지 않고 SystemExit으로 거부합니다. 이때는 새 `--out`을 쓰세요. 같은 키라면 중단된 지점부터 이어서 계산합니다. a와 b의 기본 출력 폴더(`D3/infer/v3T10_a_learned`, `D3/infer/v3T10_b_ca`)는 아직 없으므로 처음부터 계산됩니다.

decide refuses an arm in these cases:
- the arm's meta fails its checks (split, ckpt_last, candidate and variant shas, forecaster, privilege, `check.ok`);
- (F2) `infer_meta.ckpt`를 다시 해시한 sha16이 기록된 `ckpt_sha16`과 다를 때 (추론 뒤 ckpt가 바뀜);
- (리뷰 후속) meta의 `run`에 `done.json`이 없을 때 (학습 미완료: ckpt_last가 최종 가중치가 아님);
- (F2) `cache_key`가 없거나, 폴더의 `cache_key.json`과 다르거나, ckpt / which / temporal / 현재 cand · variants sha와 맞지 않을 때. 단 **legacy 경로**: F2 이전에 만든 덤프(arm (c) `D3/infer/ck2T10`)는 다시 쓰지 않고, 세션이 1개이고 그 세션이 모든 행을 계산했으며(n = n_done = n_rows) ckpt · cand · variants 검사를 통과하면 받아들입니다. step1.json에 `inputs.<arm>.cache.mode = "legacy"`, `legacy_dumps`, note로 표시됩니다. ck2T10은 이 조건을 만족합니다 [실측: sessions [18179], ckpt 7d42718ef99ca6c4 = 현재 파일].
- (a), (b) and (d) differ in training code, K, the train or val pack, or the CA scales;
- (a)'s forecaster was pre-trained on another train pack;
- an arm's inference read a different val pack than it was trained with;
- (리뷰 후속, finding 1) cache key가 있는 arm끼리 forward 코드 sha(`cache_key.code.sha256`) 또는 정밀도(`amp`)가 다를 때. 오류 메시지에 다른 파일이 나열됩니다. 이때는 같은 코드 상태에서 두 arm의 infer를 다시 돌리세요(새 `--out`). legacy arm (c)는 기록된 MODEL_FILES sha(`infer_val_v3.py` 제외)를 기준 arm의 코드와 비교해 **경고**만 냅니다 [실측 2026-10-12: 현재 코드와 7개 파일 모두 일치]. 결과는 `step1.json`의 `code_check`와 stdout `forward code:` 줄에 나옵니다.

`--no-arm-check` gives a diagnostic run that is marked `INVALID`. `--n-boot` / `--seed`를 사전등록값(2,000 / 0)과 다르게 주면 결과는 `INVALID (would be …)`, `final: false`, `preregistered_bootstrap: false`입니다(리뷰 후속 finding 2).

**GO iff G1–G4 all pass** (pre-registered; main-session decision D4). **seed ablation은 하지 않습니다(사용자 결정 2026-10-11).** 결정(GO / NO-GO)은 arm당 seed 1개로 확정(final)이며, step1.json에 `final: true`, `seed_policy: "single seed (user decision 2026-10-11)"`로 기록됩니다:

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
- **near_threshold 경고 (F1).** |Δ − 0.20| ≤ 0.15 이고 G2–G4가 모두 통과하면(이때만 G1 하나로 결론이 갈림) decide는 `near_threshold = "single seed; delta within 0.15 of the threshold -> low confidence"`를 기록하고 stdout에 `WARNING near_threshold: …` 줄을 찍습니다. 결론은 그대로 확정이며 second seed를 돌리지 않습니다. 보고할 때 "낮은 신뢰도"로 함께 적습니다. `gates.G1.within_band`는 G2–G4와 무관한 원래 구간 판정입니다. (이전 `second_seed_required` 규칙은 삭제됨.)
- **BEVFusion is in-sample on navtrain_val.** Val gains can include in-sample velocity information. Navtest stays descriptive; the stage-pre navtest ratio is the forecaster-level check.
- **(b) − (c)** is expected to be about +0.38 [추정]. If its CI lower bound is ≤ 0, decide notes that the branch itself may be broken.

## 5. Optional descriptive checks (no new training)

- **Epoch sensitivity** (seed 민감도는 하지 않음, 사용자 결정 2026-10-11): both a and b keep `ckpt_ep7.pt` and `ckpt_ep8.pt`. `infer_val_v3.py --run <run> --which ep7` (and `ep8`) takes about 3 min each on GPU 0. **별도 출력 폴더가 반드시 필요합니다.** 이제 `--out`을 주지 않으면 기본 폴더가 `$D3/infer/<run>_ep7`이 되므로 자동으로 분리됩니다(리뷰 후속 finding 4; `last`만 `$D3/infer/<run>`). 예전 문구("안 주면 last 덤프를 덮어쓴다")는 틀렸습니다: F2 이전 코드는 덮어쓰지 않고 last 점수를 그대로 둔 채 meta만 ep7로 바꿨고 check도 ok로 나왔습니다(조용히 틀린 결과) [실측: 적대적 검증 실험]. 지금은 같은 `--out`이면 cache key(which, ckpt sha)가 달라 infer가 거부합니다. Their outputs are descriptive only: decide accepts only `which last`, and the reproduce check is expected to say not ok (the saved logits are those of the last eval).
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
- [x] 2026-10-11 tooling fixes F1–F4 이후 v3 tests: 146 passed (decide +4, tooling_fixes 9 신규), CPU, 8 min 9 s [실측, `REVIEW_FIXES.md` §5.5]. decide dry run: arm (c) 덤프는 legacy로 arm check 통과.
- [x] 2026-10-12 리뷰 후속 수정 이후 전체 `tests/test_v3_*.py`(e2e 포함): 165 passed, 0 failed, 0 skipped, 7 min 59 s [실측 `/tmp/v3fix_rev2/pytest_all.log`, `REVIEW_FIXES.md` §6.7]. arm check를 켠 decide end-to-end(실제 a CPU infer의 meta/key로 만든 a / b 대역 + 실제 legacy c): valid, code check 통과, legacy c 7개 model 파일 일치.
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
- ~~Run a second seed if G1 lands in the 0.05–0.35 band.~~ → seed ablation 없음(사용자 결정 2026-10-11): 그 구간이면 `near_threshold` 경고와 함께 단일 seed 결과를 최종으로 보고.
- (F3 부작용) arm b가 끝나기 전(≈ 02:15 KST)에 죽어 `run_step1.sh b`로 재개해야 하면, training-code sha가 b의 기록값 eb1bb78f96af8e99와 달라 train_ck2가 재개를 거부합니다. 원인은 (1) F3 수정(`pretrain_forecaster.py`가 TRAIN_CODE에 포함)과 (2) 이 수정과 무관하게 다른 세션이 같은 워크트리의 `navsim/agents/para_ssr/ck/e2e_data2.py`, `online2.py`를 고치고 있는 것입니다(2026-10-12 00:46 KST 기준 sha 2386cecba71b2ce2). **`git stash`/`pop`은 쓰지 마세요**(stash 스택은 다른 워크트리·세션과 공유되고, 다른 세션의 작업 중인 파일을 빼앗습니다). 안전한 절차는 `REVIEW_FIXES.md` §5.3에 있습니다: sha를 직접 출력해 eb1bb78f96af8e99인지 확인 → F3 파일만 `git show HEAD:…`로 되돌림 → 다른 세션 파일은 그 세션과 조율 → 재개 → `resumed` 줄과 새 step 기록을 확인한 뒤 F3 파일 복원. 이후 새로 학습하는 arm(예: d)은 a/b와 code sha가 달라 decide provenance 검사에 걸립니다.
- Not fixed: R1.3, R2.5, R2.6, R2.8, R3.11 (REVIEW_FIXES.md §3).
