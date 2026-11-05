#!/usr/bin/env bash
# Warm-up labels: the official score of the 256 fixed anchors per token (raw256) and of their speed and lateral variants.
# No planner is involved.  Reference server: raw256 train 2.0 h + val 0.36 h, variants ~50 min, all at 48 CPU workers.
# Needs: packed, the navtrain metric cache, SSR_PLAN_DATA/pdm_score_256 (variants read it when ck/raw256 has no official-EP file;
# the anchor subset does not depend on the labels [measured on 1,500 rows]).
# At the end, with both splits: ck/raw256/pdm_score_256_officialEP, the file agent.config.plan_score_file reads.
#   SPLITS          default "navtrain_train navtrain_val"; navtrain_val is for retraining the refiner teachers and for that file
#   LABEL_WORKERS   default 48
# One label_variants call replaces the two runs + hand merge of the original build: traj.npy and valid were compared bitwise on
# 1,500 rows; the labels of a full rebuild have not been compared [unverified]: verify_assets compares them with the pins.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SPLITS=${SPLITS:-navtrain_train navtrain_val}
W=${LABEL_WORKERS:-48}
VAR=var_separate_sampler16_accstraight
need "$SSR_PLAN_DATA/pdm_score_256.npy"
for s in $SPLITS; do
  need "$CK/packed/$s/tokens.parquet" "$SSR_DATA_ROOT/metric_cache/_state/done_tokens.txt"
  refuse "$CK/labels/$s/raw256/labels.npy" "$CK/ck2/labels/$s/$VAR/labels.npy"
  if [ ! -e "$CK/raw256/traj_$s.npy" ] || [ "$RESUME" != 1 ]; then
    refuse "$CK/raw256/traj_$s.npy"
    run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/make_raw256_traj.py" --split "$s" --out "$CK/raw256/traj_$s.npy"
  fi
  run "${CPU[@]}" "$SSR_PY" "$REPO/tools/ck/data/label_cands.py" --split "$s" --traj "$CK/raw256/traj_$s.npy" --name raw256 --workers "$W"
  run "${CPU[@]}" "$SSR_PY" "$REPO/tools/ck/data/label_variants.py" --split "$s" --combine separate --ext straight --name "$VAR" --workers "$W"
done
if [[ -e "$CK/raw256/pdm_score_256_officialEP.npy" ]]; then
  say "$CK/raw256/pdm_score_256_officialEP.npy exists: verify_assets compares it with its pin"
elif [[ " $SPLITS " == *" navtrain_train "* && " $SPLITS " == *" navtrain_val "* ]]; then
  run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/assemble_official_ep.py"
else
  say "copy pdm_score_256_officialEP.npy and .tokens.json into $CK/raw256 (SPLITS has not both splits, so it cannot be built here)"
fi
verify labels refiner_noteacher
