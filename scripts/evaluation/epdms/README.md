# navtest EPDMS (NAVSIM v2.2, one-stage) for PARA-SSR checkpoints

NAVSIM v2 scores EPDMS, but its one-stage runner instantiates the agent inside CPU
worker processes. We therefore split the job: SSR dumps the trajectories on a GPU,
and an unmodified v2.2 checkout scores them on CPU (report 13 §3.3). Only the
trajectory source differs from v2's own `run_pdm_score_one_stage.py`; simulator,
scorer, traffic-agent policy, two-frame extended comfort and `compute_final_scores`
are the v2 functions.

Verified on Turing, 2026-10-19 (r41_v2b ep29: 12,146 scenes, 0 failed, 4.7 min with 24 processes).

## Files

| file | role |
|---|---|
| `run_pdm_score_from_trajectories.py` | copy of the script installed into the v2 checkout at `navsim/planning/script/` |
| `make_run_pdm_score_from_trajectories.py` | regenerates that script from v2's `run_pdm_score_one_stage.py` (asserts every patched line exists, so a v2 upgrade fails loudly) |

The v2 checkout itself is not tracked here. On Turing it is `~/navsim_v2` (tag v2.2 = `0a380a9`,
with untracked PARA-SSR additions: `navsim/agents/para_ssr/`, `run_pdm_score_shard.py`,
`run_pdm_score_merge.py` for two-stage). The SSR conda env (`~/conda-envs/ssr-navsim`) imports v2
fine when the v2 path is first on `PYTHONPATH`; run from a directory that is NOT the SSR repo.

## 1. Dump trajectories (SSR, GPU, batch 1)

```bash
cd ~/SSR
python tools/dump_navtest_trajectories.py --experiment <work_dirs/<exp>> \
    --checkpoint <ckpt without '=' in the name> --device cuda:0 --batch-size 1 --workers 8 \
    --out work_dirs/eval/<name>_navtest_trajectories.pkl
```

Batch 1 matches the PDM evaluation path to ~2e-6 m (report 13). The pickle holds
`{"trajectories": {token: float32 [8, 3]}, "meta": {num_poses, interval_length, ...}}`.

## 2. v2 environment and metric cache (once per machine, CPU ~30 min with 24 processes)

```bash
export PYTHONPATH=~/navsim_v2 OPENSCENE_DATA_ROOT=~/SSR/data/dataset NUPLAN_MAPS_ROOT=~/SSR/data/dataset/maps \
       NUPLAN_MAP_VERSION=nuplan-maps-v1.0 NAVSIM_EXP_ROOT=~/navsim_v2_exp/exp NAVSIM_DEVKIT_ROOT=~/navsim_v2 OMP_NUM_THREADS=1
cd ~   # not inside ~/SSR
python ~/navsim_v2/navsim/planning/script/run_metric_caching.py train_test_split=navtest \
    worker=single_machine_thread_pool worker.max_workers=24 worker.use_process_pool=true \
    metric_cache_path=$HOME/navsim_v2_exp/exp/metric_cache_navtest
```

`MetricCacheLoader` reads `metadata/*.csv`, which is written only when caching finishes: a partial
cache cannot be scored. Expect 12,146 `metric_cache.pkl`. Turing already has this cache.

## 3. Score

```bash
python ~/navsim_v2/navsim/planning/script/run_pdm_score_from_trajectories.py train_test_split=navtest \
    worker=single_machine_thread_pool worker.max_workers=24 worker.use_process_pool=true \
    metric_cache_path=$HOME/navsim_v2_exp/exp/metric_cache_navtest \
    +trajectory_pkl=$HOME/SSR/work_dirs/eval/<name>_navtest_trajectories.pkl \
    experiment_name=epdms_<name>
```

Output: `$NAVSIM_EXP_ROOT/epdms_<name>/<ts>/<ts>.csv`, one row per token plus `average_all_frames`.
Defaults: `traffic_agents=non_reactive`, PDMScorer weights EP 5 / TTC 5 / LK 2 / HC 2 / EC 2,
multiplicative NC, DAC, DDC, TLC; human-penalty filter on (v2.2 includes the issue #151 fix).
Tokens without a dumped trajectory raise instead of being skipped.

EC-excluded EPDMS (the definition used in report 24):
`score_noEC = NC*DAC*DDC*TLC * (5 EP + 5 TTC + 2 LK + 2 HC) / 14`, averaged over tokens.

## Caveat

navhard two-stage EPDMS is a different number (reactive agents, synthetic scenes; typically 30-40 for
E2E models) and needs v2's two-stage runner with the agent inside (`run_pdm_score_shard.py` in the
Turing checkout). Do not put one-stage and two-stage values in the same table.
