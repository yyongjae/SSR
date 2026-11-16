#!/usr/bin/env bash
# Warm-up labels: the official score of the 256 fixed anchors per token (raw256) and of their speed and lateral variants.
# No planner is involved.  Reference server: raw256 train 2.0 h + val 0.36 h, variants ~50 min, all at 48 CPU workers.
# Needs: packed, the navtrain metric cache, SSR_PLAN_DATA/pdm_score_256 (variants read it when ck/raw256 has no official-EP file;
# the anchor subset does not depend on the labels [measured on 1,500 rows]).
# At the end, with both splits: ck/raw256/pdm_score_256_officialEP, the file agent.config.plan_score_file reads.
#   SPLITS          default "navtrain_train navtrain_val"; navtrain_val is for retraining the refiner teachers and for that file.
#                   trainval_simscale: raw256, then ck/raw256/pdm_score_256_officialEP_trainval_simscale (its plan_score_file,
#                   assemble_trainval_simscale_score.py; the variants read it as their label source), then the variants.
#                   With RESUME=1 its finished outputs are kept (the builders skip finished shards).
#   LABEL_WORKERS   default 48
# One label_variants call replaces the two runs + hand merge of the original build: traj.npy and valid were compared bitwise on
# 1,500 rows; the labels of a full rebuild have not been compared [unverified]: verify_assets compares them with the pins.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SPLITS=${SPLITS:-navtrain_train navtrain_val}
W=${LABEL_WORKERS:-48}
VAR=var_separate_sampler16_accstraight
DATA=
case " $SPLITS " in *" navtrain_train "*|*" navtrain_val "*) need "$SSR_PLAN_DATA/pdm_score_256.npy" ;; esac
for s in $SPLITS; do
  need "$CK/packed/$s/tokens.parquet" "$SSR_DATA_ROOT/metric_cache/_state/done_tokens.txt"
  if [ "$s" = trainval_simscale ]; then
    DATA=trainval_simscale
    [ "$RESUME" = 1 ] || refuse "$CK/labels/$s/raw256/labels.npy" "$CK/ck2/labels/$s/$VAR/labels.npy"
    # RESUME=1: a finished label set is kept (the verify step below checks it belongs to this table); the anchor file
    # (8.4 GB at full size) is only an input of label_cands, so it may be deleted once raw256 is finished
    if [ -e "$CK/labels/$s/raw256/labels.npy" ]; then
      say "$CK/labels/$s/raw256 finished: kept"
    else
      run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/make_raw256_traj.py" --split "$s" --out "$CK/raw256/traj_$s.npy" --skip-done
      run "${CPU[@]}" "$SSR_PY" "$REPO/tools/ck/data/label_cands.py" --split "$s" --traj "$CK/raw256/traj_$s.npy" --name raw256 --workers "$W"
    fi
    run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/assemble_trainval_simscale_score.py" --skip-done
    if [ -e "$CK/ck2/labels/$s/$VAR/labels.npy" ]; then
      say "$CK/ck2/labels/$s/$VAR finished: kept"
    else
      run "${CPU[@]}" "$SSR_PY" "$REPO/tools/ck/data/label_variants.py" --split "$s" --combine separate --ext straight --name "$VAR" --workers "$W"
    fi
    continue
  fi
  refuse "$CK/labels/$s/raw256/labels.npy" "$CK/ck2/labels/$s/$VAR/labels.npy"
  if [ ! -e "$CK/raw256/traj_$s.npy" ] || [ "$RESUME" != 1 ]; then
    refuse "$CK/raw256/traj_$s.npy"
    run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/make_raw256_traj.py" --split "$s" --out "$CK/raw256/traj_$s.npy"
  fi
  run "${CPU[@]}" "$SSR_PY" "$REPO/tools/ck/data/label_cands.py" --split "$s" --traj "$CK/raw256/traj_$s.npy" --name raw256 --workers "$W"
  run "${CPU[@]}" "$SSR_PY" "$REPO/tools/ck/data/label_variants.py" --split "$s" --combine separate --ext straight --name "$VAR" --workers "$W"
done
if [ "$DATA" = trainval_simscale ]; then
  [ "$(echo "$SPLITS" | wc -w)" = 1 ] || die "SPLITS=trainval_simscale already holds the navtrain rows: give it alone"
elif [[ -e "$CK/raw256/pdm_score_256_officialEP.npy" ]]; then
  say "$CK/raw256/pdm_score_256_officialEP.npy exists: verify_assets compares it with its pin"
elif [[ " $SPLITS " == *" navtrain_train "* && " $SPLITS " == *" navtrain_val "* ]]; then
  run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/assemble_official_ep.py"
else
  say "copy pdm_score_256_officialEP.npy and .tokens.json into $CK/raw256 (SPLITS has not both splits, so it cannot be built here)"
fi
verify labels refiner_noteacher $DATA
