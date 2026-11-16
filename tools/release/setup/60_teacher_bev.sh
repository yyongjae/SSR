#!/usr/bin/env bash
# The two BEV teacher caches.  Only the student and refiner teachers with KD need them.
#   TEACHERS        default "resmap bevfusion"
#   BEVFUSION_SRC   rsync source of cache_train_100x100 (host:/path/); the only copy lives on the reference server
#
# ReSMap front64: public, third party, cc-by-nc-4.0 (docs/FAIRNESS.md).  Pinned revision, only bev/ + json: 781 GB (727 GiB), ~1 h.
# BEVFusion 64 m: not public.  545 GB (508 GiB).  Copy it, or regenerate it (below), which gives other weights: regenerating is
# not bitwise (deterministic false), so the sha_head in tools/release/registry.yaml, the norm_v15 file and the DET refiner teacher
# all have to be replaced too.
#   repo yyongjae/bevfusion @ b939770, env per its setup_fix.md (python 3.9, torch 2.8.0+cu128, mmcv-full 1.7.2 source build)
#   tools/data_converter/navsim_converter.py --split trainval --scene-filter navtrain.yaml --keyframes-only --cameras CAM_F0 CAM_L0 CAM_R0
#     --lidar-prefix lidar --lidar-ext .pcd --max-sweeps 2 --workers 16 --out data/infos/navsim_infos_train.pkl
#   ./train_navsim_64m.sh     stage 1 lidar 20 ep, stage 2 camera + lidar 20 ep, 39.8 h on 4 GPUs
#   ./cache_navsim_64m.sh     0.5 h on 8 GPUs, then tools/fix_manifest_64m.py
#   training monitors navtest every 5 epochs and the last epoch is used.  Not run here.
# Camera teachers (DET_TEACHER=bevformer, MAP_TEACHER=mask2map), the navtrain caches of the reference server:
#   BEVFORMER_SRC  rsync source of bevformer/cache_train_100x100 (host:/path/), 494 GB
#   MASK2MAP_SRC   rsync source of mask2map/front64 (host:/path/), 585 GB; it becomes the pass 'ref' of the merged root
#                  (tools/release/teachers/teacher_caches.py map-merge), navtest/ is not copied
# Without a copy, 65_cam_teacher_bev.sh builds both from the checkpoints (also the SimScale tokens of
# DATA=trainval_simscale; after a copy it adds only what is missing).  Rebuilt features reproduce the per-channel
# statistics of the norm files (Mask2Map exactly, BEVFormer to float32 rounding: tools/release/teachers/
# teacher_fidelity.py norm); bitwise equality with the copied caches is not checked.  These cases make the closing
# check look at the camera teacher (DET_TEACHER / MAP_TEACHER, unless set).
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TEACHERS=${TEACHERS:-resmap bevfusion}
RESMAP_REVISION=73db598669a8e6edb2b585fcc14ebe54b01b15f5
HF_BIN=$(dirname "$SSR_PY")/hf
for t in $TEACHERS; do
  case "$t" in
    resmap)
      refuse "$SSR_TEACHER_CACHE/resmap/stage1_front64/meta.json"
      run_mkdir "$SSR_TEACHER_CACHE/resmap"
      run "$HF_BIN" download rudals/resmap-navsim-teacher-kd --repo-type dataset --revision "$RESMAP_REVISION" \
        --include "stage1_front64/*.json" --include "stage1_front64/bev/*" --local-dir "$SSR_TEACHER_CACHE/resmap"
      ;;
    bevfusion)
      refuse "$SSR_TEACHER_CACHE/bevfusion/cache_train_100x100/manifest.json"
      if [ -z "${BEVFUSION_SRC:-}" ]; then
        [ "$SETUP_DRY" = 1 ] || die "BEVFUSION_SRC is not set: copy from the reference server, or regenerate (top of this file)"
        echo "# needs BEVFUSION_SRC"
      fi
      run_mkdir "$SSR_TEACHER_CACHE/bevfusion/cache_train_100x100"
      run rsync -a --partial "${BEVFUSION_SRC:-<BEVFUSION_SRC>}" "$SSR_TEACHER_CACHE/bevfusion/cache_train_100x100/"
      ;;
    bevformer)
      export DET_TEACHER=${DET_TEACHER:-bevformer}
      refuse "$SSR_TEACHER_CACHE/bevformer/cache_train_100x100/manifest.json"
      if [ -z "${BEVFORMER_SRC:-}" ]; then
        [ "$SETUP_DRY" = 1 ] || die "BEVFORMER_SRC is not set: copy from the reference server, or build it (65_cam_teacher_bev.sh)"
        echo "# needs BEVFORMER_SRC"
      fi
      run_mkdir "$SSR_TEACHER_CACHE/bevformer/cache_train_100x100"
      run rsync -a --partial "${BEVFORMER_SRC:-<BEVFORMER_SRC>}" "$SSR_TEACHER_CACHE/bevformer/cache_train_100x100/"
      ;;
    mask2map)
      export MAP_TEACHER=${MAP_TEACHER:-mask2map}
      M2M_ROOT=$SSR_TEACHER_CACHE/mask2map/front64
      refuse "$M2M_ROOT/meta.json" "$M2M_ROOT/passes/ref"
      if [ -z "${MASK2MAP_SRC:-}" ]; then
        [ "$SETUP_DRY" = 1 ] || die "MASK2MAP_SRC is not set: copy from the reference server, or build it (65_cam_teacher_bev.sh)"
        echo "# needs MASK2MAP_SRC"
      fi
      run_mkdir "$M2M_ROOT/passes/ref"
      run rsync -a --partial --exclude /navtest "${MASK2MAP_SRC:-<MASK2MAP_SRC>}" "$M2M_ROOT/passes/ref/"
      # the copied meta.json is the reference run's: it is the template of the merged one
      run "$SSR_PY" "$REPO/tools/release/teachers/teacher_caches.py" map-merge --root "$M2M_ROOT" \
        --template "$M2M_ROOT/passes/ref/meta.json"
      ;;
    *) die "unknown teacher $t" ;;
  esac
done
verify teacher_cache full
