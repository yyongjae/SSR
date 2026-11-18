#!/usr/bin/env bash
# Warm-up labels: the official score of the 256 fixed anchors per token (raw256) and of their speed and lateral variants.
# No planner is involved.  Reference server: raw256 train 2.0 h + val 0.36 h, variants ~50 min, all at 48 CPU workers.
# Needs: packed, the navtrain metric cache, SSR_PLAN_DATA/pdm_score_256 (variants read it when ck/raw256 has no official-EP file;
# the anchor subset does not depend on the labels [measured on 1,500 rows]).
# At the end, with both splits: ck/raw256/pdm_score_256_officialEP, the file agent.config.plan_score_file reads.
#   SPLITS          default "navtrain_train navtrain_val"; navtrain_val is for retraining the refiner teachers and for that file
#   LABEL_WORKERS   default 48
#   ARMS            default empty.  "v8 v4 spd4 lat4" (any of scripts/experiment.sh VARIANTS except base) also builds those
#                   arms' variant label dirs, ck2/labels/<split>/var_separate_sampler16_accstraight_<arm>, with the same
#                   label_variants call and the arm's --speeds / --lats (navsim/agents/para_ssr/ck/variants.py VARIANT_ARMS,
#                   through tools/release/compose.py variant-args; --combine cross for the cross arm, else separate).
#                   A split whose base outputs are complete keeps them (no rebuild, no refusal); a finished arm dir
#                   is refused, a partial one needs RESUME=1 (never --fresh: a build may be running on it).
#                   Each arm then goes through verify_assets --variants <arm>.
# One label_variants call replaces the two runs + hand merge of the original build: traj.npy and valid were compared bitwise on
# 1,500 rows; the labels of a full rebuild have not been compared [unverified]: verify_assets compares them with the pins.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SPLITS=${SPLITS:-navtrain_train navtrain_val}
W=${LABEL_WORKERS:-48}
VAR=var_separate_sampler16_accstraight
ARMS=${ARMS:-}
declare -A ARM_ARGS=()
for arm in $ARMS; do
  ARM_ARGS[$arm]=$("$SSR_PY" "$REPO/tools/release/compose.py" variant-args "$arm") || die "ARMS: $arm is not a variant arm"
done
need "$SSR_PLAN_DATA/pdm_score_256.npy"
for s in $SPLITS; do
  need "$CK/packed/$s/tokens.parquet" "$SSR_DATA_ROOT/metric_cache/_state/done_tokens.txt"
  if [ -n "$ARMS" ] && [ -e "$CK/labels/$s/raw256/labels.npy" ] && [ -e "$CK/ck2/labels/$s/$VAR/labels.npy" ]; then
    say "$s: raw256 and $VAR labels exist: kept, only the arms ($ARMS) are built"
  else
    refuse "$CK/labels/$s/raw256/labels.npy" "$CK/ck2/labels/$s/$VAR/labels.npy"
    if [ ! -e "$CK/raw256/traj_$s.npy" ] || [ "$RESUME" != 1 ]; then
      refuse "$CK/raw256/traj_$s.npy"
      run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/make_raw256_traj.py" --split "$s" --out "$CK/raw256/traj_$s.npy"
    fi
    run "${CPU[@]}" "$SSR_PY" "$REPO/tools/ck/data/label_cands.py" --split "$s" --traj "$CK/raw256/traj_$s.npy" --name raw256 --workers "$W"
    run "${CPU[@]}" "$SSR_PY" "$REPO/tools/ck/data/label_variants.py" --split "$s" --combine separate --ext straight --name "$VAR" --workers "$W"
  fi
  for arm in $ARMS; do
    read -r -a aa <<< "${ARM_ARGS[$arm]}"
    comb=separate
    if [ "${aa[0]}" = --combine ]; then comb=${aa[1]}; aa=("${aa[@]:2}"); fi    # variant-args: '--combine cross' first
    ad=$CK/ck2/labels/$s/${aa[-1]}
    refuse "$ad/labels.npy"
    refuse_unless_resume "$ad"
    run "${CPU[@]}" "$SSR_PY" "$REPO/tools/ck/data/label_variants.py" --split "$s" --combine "$comb" --ext straight "${aa[@]}" --workers "$W"
  done
done
if [[ -e "$CK/raw256/pdm_score_256_officialEP.npy" ]]; then
  say "$CK/raw256/pdm_score_256_officialEP.npy exists: verify_assets compares it with its pin"
elif [[ " $SPLITS " == *" navtrain_train "* && " $SPLITS " == *" navtrain_val "* ]]; then
  run "${CPU[@]}" "$SSR_PY" "$_SETUP_HERE/assemble_official_ep.py"
else
  say "copy pdm_score_256_officialEP.npy and .tokens.json into $CK/raw256 (SPLITS has not both splits, so it cannot be built here)"
fi
verify labels refiner_noteacher
for arm in $ARMS; do
  run "${CPU[@]}" "$SSR_PY" "$REPO/tools/release/verify_assets.py" --profile refiner_noteacher --only labels --variants "$arm"
done
