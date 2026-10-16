# km corridor selector (v2 r34 recipe + corridor head)

The corridor head predicts, from the BEV, a scene field that the official PDM scorer would read: the static drivable
signed distance (DAC), agent occupancy over time (NC, TTC) and route progress (EP).  At inference every one of the
256 candidates (anchor + offset) is tracked by a torch port of the official LQR + bicycle simulation and judged on
the predicted field, so the selector scores the trajectory that is actually submitted, not its anchor.
Ground truth comes from the NAVSIM metric cache only (no model outputs as labels).

## Variants (`train.sh <variant> <run> <gpus>`)

| variant | what is trained | status |
|---|---|---|
| `v2` | v2 r34 + corridor head (static SDF + 8 x 0.5 s swept occupancy), 0-32 m | trained (`corr30_v2`), navtest below |
| `v31` | + 40 x 0.1 s occupancy planes, plan constraint on every tracked candidate (static / occupancy / TTC hinges on the GT corridor), conditional EP label (`plan_ep_feasible_only`), route progress channel, **progress reward inside the GT-safe corridor** (every candidate except the imitation winner; saturates at 1.2 x the GT progress), official-comfort hinge, verdict supervision (predicted-corridor verdicts of model-free footprints vs GT verdicts) | training (`corr30_v31`), 32 m smoke-tested |
| `v31_64` | `v31` on a 0-64 m BEV (`pc_range` y 0..64, `bev_h` 100, BEV 100 x 100) with the 64 m GT (`corridor_format=c64`) | smoke-tested (40 steps): config, 10k BEV queries, 41-channel corridor output confirmed; not trained |

navtest (12,146, official, 4th look), `corr30_v2` epoch 30, rule `pred_multiply_t0.25` (v2 score + 0.5 log p_NC + 0.5 log p_DAC, fixed on navtrain_val):

| | PDMS | EPDMS |
|---|---|---|
| v2 r34 (no corridor head) | 88.14 | 87.80 |
| `corr30_v2`, v2 selection | 88.41 | 88.07 |
| `corr30_v2`, corridor selection | **89.30** | **88.57** |

## Run elsewhere

1. `cp scripts/km_corridor/env.example.sh scripts/km_corridor/env.sh` and edit the paths (data root, python, NAVSIM data,
   `PLAN_SCORE` = the 256-anchor official labels `pdm_score_256.npy` + `.tokens.json`, 253 MB, not in git).
   Metric caches are set in `tools/ck/data/common.py` (`MC_ROOT`).
2. Token tables (row order of every cache) and the 256 anchors are in `resources/km_corridor/`.
3. GT corridor caches (CPU, from the metric cache; or copy them):
   - 32 m: `build_gt.sh 32` -> `corridor_dt01` (~14 GB, ~2 h at 9 workers)
   - 64 m: `build_gt.sh 64` -> `corridor64` (~16 GB, ~3.5 h at 12 workers), then `build_gt.sh route32` -> `route32` (~2 GB;
     the 32 m `v31` needs it, so build 64 first)
4. `train.sh v31 corr30_v31 0,1` (or `v31_64`).  Smoke: `EPOCHS=1 LIMIT_TRAIN=40 LIMIT_VAL=5 VAL_EVERY=1 WANDB=false`.
5. Evaluation (`eval.sh`): `val` = navtrain_val all 18,179 tokens, every pre-declared rule, official PDMS;
   `navtest` = the rule fixed on val + the same model's v2 selection, official PDMS and EPDMS
   (needs the SSR-v2 checkout's `data/logs/score_npz.sh`, `V2_REPO`).
   Arguments for `v31`: planes 40, route 1; for `v31_64` also fwd rows 200.

## Known gaps

- (km 10/17, fixed) The corridor selection is inside the agent's forward: `agent.config.corridor_select_rule`
  (`modules/corridor_select.py`; `train.sh` sets `pred_multiply_t0.25`, `SELECT_RULE=v2` for the old behaviour), so the
  standard NAVSIM scripts score the corridor-selected trajectory.  Same computation as `corr_verdict.py` + `corr_select.py`:
  `corr30_v31` epoch 5, navtest 1,024 tokens, forward choice == offline `pred_multiply_t0.25` on 99.9 % (the one
  difference is a 4e-5 score tie from the dump's uint8 / float16 rounding).  The dump records the forward's choice
  (`msel.npy`), and `corr_select.py` / `select_npz.py` report it as the rule `model` next to every offline rule.
  Runs whose config has no `corridor_select_rule` keep `v2`.
- TTC verdict is a simplification of the official rule (no heading / intersection / multi-lane conditions; GT agreement 94%).
- 64 m: `corr_verdict.py` GT columns still read the 32 m `corridor_dt01` GT (oracle rows only); `max_agents` / `map_max_vec`
  stay at 100 for the doubled range (nearest kept).
