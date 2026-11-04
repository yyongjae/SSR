#!/usr/bin/env bash
# Dataset: the official NAVSIM download commands (devkit v2.2, download/download_{navtrain_hf,test,maps}.sh), only what
# this path reads.  Then the links that give the layout the code expects.
#
#   DOWNLOAD_DIR  where the official layout goes (default $OPENSCENE_DATA_ROOT/_download)
#   PARTS         default "logs navtrain maps test_logs test_cameras"; also test_lidar, navhard
#
# Needed by       logs navtrain maps: every training.  test_logs test_cameras: evaluation.
# Not by default  test_lidar: only to regenerate the BEVFusion navtest monitor.  navhard: two-stage EPDMS only.
# Without the camera and metadata tarballs nothing here can be skipped: navtrain tarballs mix cameras and LiDAR, and
# BEVFusion regeneration needs the LiDAR.  Resumable: each finished tarball leaves a stamp in $DOWNLOAD_DIR/.done.
# Time: ~7 h for everything on the reference server [estimate from file times].  Disk: docs/SETUP.md.  Not run in testing.
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
DL=${DOWNLOAD_DIR:-$OPENSCENE_DATA_ROOT/_download}
PARTS=${PARTS:-logs navtrain maps test_logs test_cameras}
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main

# fetch URL: download and unpack one tarball in $DL, once
fetch() {
  local url=$1 f stamp
  f=$(basename "$url"); stamp=$DL/.done/$f
  [ -e "$stamp" ] && return 0
  run wget -c -q --show-progress -P "$DL" "$url"
  run tar -xzf "$DL/$f" -C "$DL"
  run rm -f "$DL/$f"
  [ "$SETUP_DRY" = 1 ] || touch "$stamp"
}

# link TARGET NAME: a symlink; refuses to replace anything else
link() {
  local tgt=$1 name=$2
  if [ -L "$name" ]; then
    [ "$(readlink "$name")" = "$tgt" ] || die "$name -> $(readlink "$name"), expected $tgt"
    return 0
  fi
  [ -e "$name" ] && die "refusing to replace $name with a link to $tgt"
  run mkdir -p "$(dirname "$name")"
  run ln -s "$tgt" "$name"
}

run_mkdir "$DL/.done"

for part in $PARTS; do
  case "$part" in
    logs)
      fetch "$HF/openscene-v1.1/openscene_metadata_trainval.tgz"
      [ -e "$DL/trainval_navsim_logs" ] || run mv "$DL/openscene-v1.1/meta_datas" "$DL/trainval_navsim_logs"
      ;;
    navtrain)
      run_mkdir "$DL/trainval_sensor_blobs/trainval"
      for kind in current history; do
        for i in $(seq 1 32); do
          fetch "$HF/navsim/navtrain_${kind}_$i.tgz"
          if [ -e "$DL/navtrain_${kind}_$i" ] || [ "$SETUP_DRY" = 1 ]; then
            run rsync -r "$DL/navtrain_${kind}_$i/" "$DL/trainval_sensor_blobs/trainval/"
            run rm -r "$DL/navtrain_${kind}_$i"
          fi
        done
      done
      ;;
    maps)
      if [ ! -e "$DL/maps" ]; then
        run wget -c -q --show-progress -P "$DL" https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
        run unzip -q "$DL/nuplan-maps-v1.1.zip" -d "$DL"
        run rm -f "$DL/nuplan-maps-v1.1.zip"
        run mv "$DL/nuplan-maps-v1.0" "$DL/maps"
      fi
      ;;
    test_logs)
      fetch "$HF/openscene-v1.1/openscene_metadata_test.tgz"
      [ -e "$DL/test_navsim_logs" ] || run mv "$DL/openscene-v1.1/meta_datas" "$DL/test_navsim_logs"
      ;;
    test_cameras)
      for i in $(seq 0 31); do fetch "$HF/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_$i.tgz"; done
      [ -e "$DL/test_sensor_blobs" ] || run mv "$DL/openscene-v1.1/sensor_blobs" "$DL/test_sensor_blobs"
      ;;
    test_lidar)
      for i in $(seq 0 31); do fetch "$HF/openscene-v1.1/openscene_sensor_test_lidar/openscene_sensor_test_lidar_$i.tgz"; done
      if [ "$SETUP_DRY" != 1 ] && [ -d "$DL/openscene-v1.1/sensor_blobs" ]; then
        run rsync -r "$DL/openscene-v1.1/sensor_blobs/" "$DL/test_sensor_blobs/"
      fi
      ;;
    navhard)
      for f in curr_sensors hist_sensors scene_pickles; do
        fetch "$HF/navsim-v2/navsim_v2.2_navhard_two_stage_$f.tar.gz"
      done
      ;;
    *) die "unknown part $part" ;;
  esac
done

# the layout of the code: OPENSCENE_DATA_ROOT holds the leaves directly, SSR_TEST_DATA_ROOT keeps test/test
link "$DL/trainval_navsim_logs/trainval"    "$OPENSCENE_DATA_ROOT/navsim_logs/trainval"
link "$DL/trainval_sensor_blobs/trainval"   "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval"
link "$DL/maps"                             "$NUPLAN_MAPS_ROOT"
case " $PARTS " in *" test_logs "*)
  link "$DL/test_navsim_logs/test"          "$SSR_TEST_DATA_ROOT/navsim_logs/test/test" ;;
esac
case " $PARTS " in *" test_cameras "*)
  link "$DL/test_sensor_blobs/test"         "$SSR_TEST_DATA_ROOT/sensor_blobs/test/test" ;;
esac
verify dataset base
