#!/usr/bin/env bash
# Camera-teacher BEV caches, BEVFormer (DET) and Mask2Map (MAP), for every token of a token table: the real tokens and
# the SimScale tokens of DATA=trainval_simscale, in the roots and formats of tools/release/registry.yaml (one root per
# teacher, real and SimScale together).  GPU work; one resumable step: run it again after a stop (RESUME=1).
#   GPUS           GPU list (default 0): one BEVFormer process per GPU, Mask2Map torchrun over all of them.  A pass
#                  keeps the GPU count it started with (Mask2Map's writer pins its world size).
#   TABLE          token table (default $SSR_DATA_ROOT/splits/trainval_simscale.parquet, 16_trainval_simscale_table.sh)
#   TEACHERS       default "bevformer mask2map"
#   TEACHER_SRC    the checkouts of tools/release/teachers/get_teachers.sh (default $SSR_TEACHER_CACHE/_src)
#   TEACHER_PY     python of the env of tools/release/teachers/setup_teacher_env.sh (default $SSR_TEACHER_CACHE/_env/bin/python)
#   CAM_TEACHERS   the cam_teachers bundle: weights, configs, cache manifests (default $SSR_TEACHER_CACHE/cam_teachers)
#   WORKERS        dataloader workers per GPU (default 6); CPU_WORKERS for the info builders (default 16)
#   ROUNDS         SimScale rounds to cache (default: the rounds of the table)
#
# Per teacher, one pass per source: real (the table's navtrain_train + navtrain_val tokens) and sim_r<r> per SimScale
# round; a pass caches only the tokens no pass holds yet (a reference cache copied by 60_teacher_bev.sh counts) and keeps
# that token list for good.  Then the manifests: DET manifest.json from the bundle's (counts and passes added), MAP one
# index.json + meta.json over every pass (tools/release/teachers/teacher_caches.py).
#   BEVFormer  infos: tools/data_converter/navsim_converter.py, real over the navtrain filter (the frame pool of the
#              reference cache: 152,495 frames, 4 history frames per token, all 8 cameras checked); SimScale per round
#              with the converter's functions per scene pickle (teacher_caches.py bev-infos: 4 history + 8 future frames,
#              CAM_F0/L0/R0, frames 0..3 kept, one prev chain per scene; the CLI cannot take a round, a SimScale frame
#              token repeats across overlapping scenes).  Cache: tools/navsim_cache_teacher.py, video_test_mode along
#              each chain, whole logs per GPU.
#   Mask2Map   infos: tools/navsim/create_navsim_infos.py (real, navtrain), teacher_caches.py m2m-infos (SimScale,
#              frame 3, CAM_L0/F0/R0, poses from translation + quaternion).  Map GT: none needed by the test pipeline,
#              so an empty map GT per pass (teacher_caches.py empty-gt).  Cache: tools/navsim/cache_teacher_m2m.py.
# Measured on RTX 5090 (2 GPUs, WORKERS=6): BEVFormer 8.9 frames/s per GPU on the real pool and 9.4 on SimScale (a real
# token costs 1.48 pool frames over the full table, a SimScale token 4), Mask2Map 53 frames/s per GPU (one per token).
# Full table, 103,288 real + 237,372 SimScale tokens: BEVFormer 152,389 + 949,488 frames = 4.8 + 28.2 GPU-h, Mask2Map
# 1.8 GPU-h, about 35 RTX-5090 GPU-hours (H100 not measured).  Disk: 5.13 MB (DET) + 5.44 MB (MAP) per token = 3.6 TB,
# plus about 12 GB of info pkls in $SSR_TEACHER_CACHE/_work.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
GPUS=${GPUS:-0}
TABLE=${TABLE:-$SSR_DATA_ROOT/splits/trainval_simscale.parquet}
TEACHERS=${TEACHERS:-bevformer mask2map}
TEACHER_SRC=${TEACHER_SRC:-$SSR_TEACHER_CACHE/_src}
TEACHER_PY=${TEACHER_PY:-$SSR_TEACHER_CACHE/_env/bin/python}
CAM_TEACHERS=${CAM_TEACHERS:-$SSR_TEACHER_CACHE/cam_teachers}
WORKERS=${WORKERS:-6}
CPU_WORKERS=${CPU_WORKERS:-16}
TC="$SSR_PY $REPO/tools/release/teachers/teacher_caches.py"
WORK=$SSR_TEACHER_CACHE/_work/cam_teachers
BEV=$TEACHER_SRC/BEVFormer
M2M=$TEACHER_SRC/Mask2Map_navsim
BEV_CFG=$CAM_TEACHERS/teachers/bevformer/bevformer_navsim_base.py
BEV_CKPT=$CAM_TEACHERS/teachers/bevformer/epoch_5.pth
M2M_CFG=$CAM_TEACHERS/teachers/mask2map_bnft/M2M_navsim_front64_r50_sem_18ep_bnft.py
M2M_CKPT=$CAM_TEACHERS/teachers/mask2map_bnft/epoch_1.pth
DET_TEMPLATE=$CAM_TEACHERS/cache_manifests/bevformer_cache_train_100x100/manifest.json
MAP_TEMPLATE=$CAM_TEACHERS/cache_manifests/mask2map_front64/meta.json
SF_DIR=$REPO/navsim/planning/script/config/common/scene_filter
NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
read -r -a GPU_LIST <<< "$(echo "$GPUS" | tr ',' ' ')"
# numpy 1.23's OpenBLAS misdetects some Xeons and returns wrong float64 linear algebra (the camera matrices of the
# Mask2Map infos are inverted with it); its SkylakeX kernels need AVX-512, so they are forced only where the CPU has it
OBLAS=()
if grep -qw avx512f /proc/cpuinfo 2>/dev/null; then OBLAS=(OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-SkylakeX}"); fi
CPUB=(env "${OBLAS[@]}" "${CPU[@]}")
for t in $TEACHERS; do case "$t" in bevformer|mask2map) ;; *) die "unknown teacher $t (bevformer, mask2map)" ;; esac; done

# roots and shas of the registry entries (refuses a teacher that is not ready)
_exports=$("$SSR_PY" "$_SETUP_HERE/teacher_env.py" bevformer mask2map) || die "teacher registry"
eval "$_exports"
DET=$SSR_DET_CACHE_TRAIN
MAP=$SSR_MAP_CACHE_ROOT

refuse "$WORK/done.json"
refuse_unless_resume "$WORK/tokens/lists.json"
need "$TABLE" "$TEACHER_PY" "$BEV/tools/navsim_cache_teacher.py" "$M2M/tools/navsim/cache_teacher_m2m.py" "$BEV_CKPT" \
  "$M2M_CKPT" "$BEV_CFG" "$M2M_CFG" "$DET_TEMPLATE" "$MAP_TEMPLATE" "$SF_DIR/navtrain.yaml" \
  "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval"
[ "$SETUP_DRY" = 1 ] || bash "$REPO/tools/release/teachers/get_teachers.sh" "$TEACHER_SRC" --check >/dev/null \
  || die "$TEACHER_SRC is not the pinned teacher code: tools/release/teachers/get_teachers.sh $TEACHER_SRC"
if [ -z "${ROUNDS:-}" ]; then
  if [ -f "$TABLE" ]; then
    ROUNDS=$("$SSR_PY" -c "import sys, pandas as pd; d = pd.read_parquet(sys.argv[1], columns=['source', 'round']); \
print(' '.join(str(r) for r in sorted(set(d['round'][d.source == 'simscale'].astype(int)))))" "$TABLE")
  else
    ROUNDS="0 1 2 3 4"
  fi
fi
[ -z "$ROUNDS" ] || need "$SSR_SIMSCALE_ROOT"
PASSES="real"
for r in $ROUNDS; do PASSES="$PASSES sim_r$r"; done
run_mkdir "$WORK/logs"
run $TC tokens --table "$TABLE" --work "$WORK"

# timed NAME CMD...: run CMD and add its wall time to $WORK/timing.tsv (teacher pass gpus seconds)
timed() {
  local name=$1 t0; shift
  t0=$(date +%s)
  "$@"
  [ "$SETUP_DRY" = 1 ] || printf '%s\t%s\t%s\n' "$name" "$NGPU" "$(( $(date +%s) - t0 ))" >> "$WORK/timing.tsv"
}

# frozen_tokens LIST OUT --det|--map ROOT: the tokens of LIST no cache holds yet, written once per pass (kept after)
frozen_tokens() { run $TC pass-tokens --list "$1" --out "$2" "${@:3}"; }

# free_port: a free TCP port on the loopback
free_port() { "$SSR_PY" -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])"; }

# empty FILE: true when FILE holds no token (a pass with nothing left to cache)
empty() { [ "$SETUP_DRY" != 1 ] && [ ! -s "$1" ]; }

bevformer_infos() {
  local out=$WORK/bevformer/infos/navsim_infos_$1.pkl
  run_mkdir "$(dirname "$out")"
  if [ "$1" = real ]; then
    if [ -f "$out" ]; then say "$out exists"; return; fi
    run "${CPUB[@]}" "$TEACHER_PY" -I "$BEV/tools/data_converter/navsim_converter.py" --split trainval \
      --navsim-logs "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" --sensor-root "$OPENSCENE_DATA_ROOT/sensor_blobs" \
      --sensor-prefix "$OPENSCENE_DATA_ROOT/sensor_blobs" --scene-filter "$SF_DIR/navtrain.yaml" --history-frames 4 \
      --check-files --workers "$CPU_WORKERS" --out "$out.part.pkl"
    run mv "$out.part.pkl" "$out"
  else
    # the converter's per-frame functions per scene pickle (frame 3 in-split, history frames keyed per scene): the
    # converter CLI keys in-split frames by token, which SimScale reuses across overlapping scenes
    run "${CPUB[@]}" $TC bev-infos --table "$TABLE" --round "${1#sim_r}" --converter \
      "$BEV/tools/data_converter/navsim_converter.py" --out "$out" --workers "$CPU_WORKERS"
  fi
}

bevformer_pass() {
  local p=$1 k
  local toks=$DET/passes/$p/tokens.txt
  run_mkdir "$DET/passes/$p"
  frozen_tokens "$WORK/tokens/${p}.txt" "$toks" --det "$DET"
  if empty "$toks"; then say "bevformer $p: nothing to cache"; return; fi
  bevformer_infos "$p"
  for k in "${!GPU_LIST[@]}"; do
    run_bg "$WORK/logs/bevformer_${p}_r$k.log" env CUDA_VISIBLE_DEVICES="${GPU_LIST[$k]}" OMP_NUM_THREADS=2 "${OBLAS[@]}" \
      PYTHONPATH="$BEV" nice -n 5 "$TEACHER_PY" "$BEV/tools/navsim_cache_teacher.py" --cfg "$BEV_CFG" --ckpt "$BEV_CKPT" \
      --ann-file "$WORK/bevformer/infos/navsim_infos_$p.pkl" --out "$DET" --pass "$p" --token-list "$toks" \
      --rank "$k" --world "$NGPU" --workers "$WORKERS" --expect-sha-head "$SSR_DET_SHA_HEAD" --resume
  done
  timed "bevformer $p" wait_bg
}

mask2map_infos() {
  local out=$WORK/mask2map/infos/navsim_map_infos_$1.pkl
  if [ "$1" = real ]; then
    if [ -f "$out" ]; then say "$out exists"; return; fi
    run_mkdir "$(dirname "$out")"
    run "${CPUB[@]}" "$TEACHER_PY" -I "$M2M/tools/navsim/create_navsim_infos.py" --filter navtrain \
      --scene-filter-dir "$SF_DIR" --logs-root "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
      --blobs-root "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" --workers "$CPU_WORKERS" --out "$out.part.pkl"
    run mv "$out.part.pkl" "$out"
  else
    run "${CPUB[@]}" $TC m2m-infos --table "$TABLE" --round "${1#sim_r}" --out "$out" --workers "$CPU_WORKERS"
  fi
}

mask2map_pass() {
  local p=$1
  local out=$MAP/passes/$p
  local toks=$out/tokens.txt gt=$WORK/mask2map/gt/$p.npz
  run_mkdir "$out"
  frozen_tokens "$WORK/tokens/${p}.txt" "$toks" --map "$MAP"
  if empty "$toks"; then say "mask2map $p: nothing to cache"; return; fi
  mask2map_infos "$p"
  run $TC empty-gt --tokens "$toks" --out "$gt"
  # torchrun on the loopback: --standalone rendezvouses on the host name, which can resolve to an unusable address
  # (link-local IPv6 without a scope); the writer's only collective is a gloo barrier
  run_bg "$WORK/logs/mask2map_$p.log" env CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=2 "${OBLAS[@]}" \
    GLOO_SOCKET_IFNAME=lo PYTHONPATH="$M2M" nice -n 5 "$TEACHER_PY" -m torch.distributed.run --nnodes=1 \
    --nproc_per_node="$NGPU" --master_addr=127.0.0.1 --master_port="$(free_port)" \
    "$M2M/tools/navsim/cache_teacher_m2m.py" --cfg "$M2M_CFG" --ckpt "$M2M_CKPT" --split none --only-split-tokens \
    --ann-file "$WORK/mask2map/infos/navsim_map_infos_$p.pkl" --map-gt-file "$gt" --token-list "$toks" --out "$out" \
    --workers "$WORKERS" --resume
  timed "mask2map $p" wait_bg
}

for t in $TEACHERS; do
  for p in $PASSES; do "${t}_pass" "$p"; done
  case "$t" in
    bevformer) run $TC det-manifest --root "$DET" --template "$DET_TEMPLATE" ;;
    mask2map) run $TC map-merge --root "$MAP" --template "$MAP_TEMPLATE" ;;
  esac
done

# every table token in both caches through the SSR readers (all tokens for coverage, a 2,000-token sample read in
# full), then the release checks of the selected recipe
V=(--table "$TABLE" --sample "${VERIFY_SAMPLE:-2000}" --det-sha-head "$SSR_DET_SHA_HEAD" --map-sha256 "$SSR_MAP_SHA256")
case " $TEACHERS " in *" bevformer "*) V+=(--det "$DET") ;; esac
case " $TEACHERS " in *" mask2map "*) V+=(--map "$MAP") ;; esac
run "${CPU[@]}" $TC verify "${V[@]}" --coverage --out "$WORK/verify.json"
verify teacher_cache full trainval_simscale
case " $TEACHERS " in *" bevformer "*) case " $TEACHERS " in *" mask2map "*) run cp "$WORK/verify.json" "$WORK/done.json" ;; esac ;; esac
say "done: $DET, $MAP"
