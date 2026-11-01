# Stage E: E2 navtest evaluation (2026-09-30)

Spec: PRESTATED_DECISION_RULE.txt, "STAGE E — E2 EVALUATION SPEC". These results are descriptive only. The E2 - E1 rule is still open.

- Checkpoint: stageE_E2_30ep_ema_r1.0_s0_rr5_wmax100_hmix_hwu5_bg1.0 / version_0 / `epoch=29-step=19950.ckpt`. It is byte-identical (`cmp`) to `last.ckpt`, with epoch 29 and global_step 19950. The eval ran on `last.ckpt` because Hydra cannot parse the `=` in the epoch filename. E0 was also evaluated from `last.ckpt`.
- Pipeline: `stageE_gpu_commands.sh eval E2 <ckpt> final|tau0` goes through `eval_para_ssr.sh` and then `run_pdm_score_gpu.py`. E2 ran on GPU 0 and E2_tau0 on GPU 1.
- E0 CSV (`para_ssr_interaction_final/2026.09.17.00.09.41.csv`) came from the same pipeline. Its hydra config (worker, simulator, scorer weights, metric cache, navtest filter and data paths) matches the E2 runs line by line. Only `experiment_name` differs. No scorer, simulation or script file changed after that CSV was written. The only changes since then are in agent code and config, where `refiner_mode` defaults to off.
- All 12,146 of 12,146 tokens are valid in every arm.
- Bootstrap: paired by token, resampling log clusters (136 logs), 10,000 draws, seed 0, percentile 95% CI. Values are in points, x100.

## Arm means (all navtest)
| arm | PDMS | NC | DAC | DDC | TTC | EP | C |
|---|---|---|---|---|---|---|---|
| E0 | 84.87 | 97.86 | 93.31 | 100.00 | 93.56 | 79.77 | 99.99 |
| E2 (tau_final) | 86.15 | 98.17 | 94.29 | 100.00 | 94.04 | 80.99 | 99.99 |
| E2_tau0 | 84.66 | 97.93 | 92.92 | 100.00 | 93.66 | 79.67 | 99.99 |

## Contrasts (diff [95% CI])
| contrast | PDMS | NC | DAC | TTC | EP |
|---|---|---|---|---|---|
| E2 - E0 | +1.28 [+0.74, +1.86] | +0.31 [+0.03, +0.61] | +0.98 [+0.49, +1.50] | +0.48 [+0.02, +0.95] | +1.22 [+0.72, +1.74] |
| E2_tau0 - E0 | -0.21 [-0.85, +0.41] | +0.07 [-0.18, +0.35] | -0.39 [-0.96, +0.18] | +0.10 [-0.36, +0.56] | -0.10 [-0.69, +0.46] |
| E2 - E2_tau0 | +1.50 [+1.06, +2.03] | +0.23 [+0.11, +0.37] | +1.37 [+0.96, +1.86] | +0.38 [+0.22, +0.56] | +1.32 [+0.94, +1.80] |

DDC and C differences are 0.00 [0.00, 0.00] in all three contrasts.

## By city: PDMS diff [95% CI]
| contrast | Las Vegas (n=4064, 50 logs) | other cities (n=8082, 86 logs) | sg | boston | pittsburgh |
|---|---|---|---|---|---|
| E2 - E0 | +0.77 [+0.07, +1.43] | +1.54 [+0.80, +2.33] | +1.95 | +1.56 | +1.18 |
| E2_tau0 - E0 | +0.03 [-0.64, +0.69] | -0.33 [-1.24, +0.51] | -1.54 | +0.04 | +0.03 |
| E2 - E2_tau0 | +0.75 [+0.38, +1.18] | +1.87 [+1.27, +2.63] | +3.49 | +1.52 | +1.15 |

## Tokens fixed and broken (A vs B, all navtest)
| contrast | PDMS up / down / tie | up >= 0.1 / down <= -0.1 | 0 -> >0 (fixed) / >0 -> 0 (broken) |
|---|---|---|---|
| E2 vs E0 | 4326 / 3853 / 3967 | 735 / 554 | 504 / 361 |
| E2_tau0 vs E0 | 4196 / 3924 / 4026 | 631 / 654 | 412 / 454 |
| E2 vs E2_tau0 | 3659 / 3603 / 4884 | 226 / 17 | 194 / 9 |

The plain up/down counts are mostly tiny EP changes. The >= 0.1 and 0-flip columns are the counts that matter.

Files: `stageE_compare.json` (stageE_compare.py output), `stageE_compare_ext.json` (per-city and per-arm means plus the token counts), `stageE_compare_ext.py`.
CSVs: `work_dirs/eval/stageE_E2/2026.09.30.20.42.16.csv` and `work_dirs/eval/stageE_E2_tau0/2026.09.30.20.34.56.csv`.
