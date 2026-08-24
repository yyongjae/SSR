# What did the auxiliary tasks do to the BEV feature?

Borrows the mechanism of Layer-Wise Modality Decomposition
([arXiv:2511.00859](https://arxiv.org/pdf/2511.00859), NeurIPS 2025) and swaps
its axis. LMD decomposes a fused feature over additive INPUTS; the question here
is about supervision, so the axes become the 10,000 BEV cells (which of them the
planner actually uses) and, for the one checkpoint pair that shares a basin, the
weight groups (what changed and whether it helped).

What transfers is the property that makes LMD work: freeze every nonlinearity at
its operating point and the network is exactly affine, so the decomposition sums
back to the output with no residual. That identity is carried alongside every
number produced here — if it stops holding, the numbers are meaningless and you
should be able to see it immediately.

* Method, definitions, limits: [`report/09`](../../report/09_lmd_planning_centric_bev.md)
* Experiment plan, checkpoints, what is and is not answerable: [`report/10`](../../report/10_lmd_experiment_plan.md)

## Running it

```shell
bash tools/lmd/setup_env.sh          # one-time; follows setup_fix.md, not docs/install.md
bash tools/lmd/run.sh e0             # THE GATE — is the real model exactly affine?
bash tools/lmd/run.sh e1 200         # planning-centric BEV maps, 200 val samples
bash tools/lmd/run.sh all 200        # both
```

Uses GPU 0 and 1 (2 and 3 are occupied). Each job is single-GPU, so the four
conditions run two per GPU in parallel. Logs in `out/lmd/logs/`, results in
`out/lmd/e1_<condition>.npz`.

**E1 will refuse to run until E0 has passed.** E0 checks that the frozen forward
reproduces the trained model's trajectory and that the per-cell contributions
plus the bias sum back to it. If that residual is not small, every map E1 would
draw is a fiction, so the gate is not a formality.

## Conditions

| key | run | what it is |
|---|---|---|
| `plan_only` | `ssr_noffp_2gpu_b4` @ 12 | planning only — SSR-noFFP, no aux heads at all |
| `aux_only` | `para_ssr_stage1` @ 48 | aux only (plan=0). Its own planner never trained, so it is read with `staged`'s planner — legitimate only because stage2 forked from it |
| `staged` | `para_ssr_stage2` @ 12 | stage1 + 12 epochs with planning on |
| `both` | `para_ssr_60ep` @ 60 | plan + det + motion + map, monolithic |

`plan_only` and `both` are independently trained runs of different model classes
on different schedules, so they may be compared as spatial maps and scalars but
never mixed at weight level. Only `aux_only`/`staged` share a basin.

## Files

| file | what it is |
|---|---|
| `lmd_core.py` | frozen-switch versions of every nonlinearity, plus forward and adjoint readers |
| `replica.py` | the op stack as a chain of weight groups, for hybrid forwards |
| `verify_lmd_linearisation.py` | float64 proof on replicas that the trajectory splits over BEV cells with no residual |
| `verify_model_delta.py` | float64 proof that a two-model difference splits over weight groups (telescoping and exact Shapley) |
| `lmd_hooks.py` | the same linearisation applied to the real mmcv modules |
| `lmd_common.py` | checkpoint registry, model build, val loop, BEV grid geometry |
| `run_e0.py` | the gate |
| `run_e1.py` | pi / rho / PR / PPA, and omega for the det and map heads |
| `setup_env.sh`, `run.sh` | environment and the GPU 0,1 driver |

Not yet implemented: E3/E4, the hybrid forward runner and the signed
helped-vs-hurt map for the fork pair (report #10 section 4).
