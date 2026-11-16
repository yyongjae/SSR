#!/usr/bin/env bash
# The code of the two camera teachers, reproducibly: the public forks at the commits the teachers were trained and
# cached with, plus the work that was never pushed, as the patches of tools/release/teachers/patches/.
#
#   bash tools/release/teachers/get_teachers.sh <dest>          # clones <dest>/BEVFormer and <dest>/Mask2Map_navsim
#   bash tools/release/teachers/get_teachers.sh <dest> --check  # only verifies an existing <dest>
#
# BEVFormer        github.com/yyongjae/BEVFormer branch navsim @ 9f1d5df: the NAVSIM port (converter, dataset, configs).
#                  patches/bevformer_navsim.patch = the uncommitted tools/{train,test}.py --local-rank fix of the
#                  training server + tools/navsim_cache_teacher.py, the cache writer (its source is
#                  tools/release/teachers/navsim_cache_teacher.py; the reference cache's writer was never committed).
# Mask2Map_navsim  github.com/yyongjae/Mask2Map_navsim @ a7f07a9 (upstream Mask2Map + vendored mmdetection3d 0.17.2).
#                  patches/mask2map_navsim.patch = the navsim-front64 working tree of the training server: its
#                  uncommitted diff (torch 2.x / mmcv 1.7 ports, the semantic_only head) and its 19 untracked files
#                  (NAVSIM dataset, loaders, configs, tools/navsim/{create_navsim_infos,navsim_map_gt,pack_map_gt,
#                  cache_teacher_m2m,setup_env}.py|sh, ...), byte-equal to that tree (tests/test_simscale_teachers.py).
# Weights, configs and cache manifests are not code: they are the cam_teachers bundle (CAM_TEACHERS of
# tools/release/setup/65_cam_teacher_bev.sh).  BEVFORMER_URL / MASK2MAP_URL point the clones at a mirror.
# An existing clone is kept when it is the pinned commit with exactly its patch applied, and refused otherwise.
set -euo pipefail

DEST=${1:?usage: get_teachers.sh <dest> [--check]}
MODE=${2:-get}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCHES=$HERE/patches
BEVFORMER_URL=${BEVFORMER_URL:-https://github.com/yyongjae/BEVFormer.git}
MASK2MAP_URL=${MASK2MAP_URL:-https://github.com/yyongjae/Mask2Map_navsim.git}
# name  url  ref to fetch  pinned commit  patch
TEACHERS=(
  "BEVFormer $BEVFORMER_URL navsim 9f1d5df246bfb14699b4d6f26c96ba1aa25dde4d bevformer_navsim.patch"
  "Mask2Map_navsim $MASK2MAP_URL main a7f07a9b88c1a419fbeb86a3fb8b868b0516571f mask2map_navsim.patch"
)

die() { echo "[get_teachers] $*" >&2; exit 3; }
say() { echo "[get_teachers] $*"; }

# verify DIR COMMIT PATCH: the checkout is COMMIT with exactly PATCH applied.  The reverse check pins the content of
# every file of the patch (new files included); no other tracked file may differ.  Untracked files the patch does not
# name (build products of setup_teacher_env.sh, info pkls) are allowed.
verify() {
  local dir=$1 commit=$2 patch=$3 head want have
  head=$(git -C "$dir" rev-parse HEAD) || die "$dir is not a git checkout"
  [ "$head" = "$commit" ] || die "$dir is at $head, not the pinned $commit: move it away"
  git -C "$dir" apply -R --check "$patch" 2>/dev/null || die "$dir does not hold $(basename "$patch") (edited?): move it away"
  want=$(awk '/^diff --git a\//{f=$3; sub(/^a\//, "", f); getline; if ($0 !~ /^new file mode/) print f}' "$patch" | sort -u)
  have=$(git -C "$dir" diff --name-only HEAD | sort -u)
  [ "$want" = "$have" ] || die "$dir: tracked files differ from $(basename "$patch"):
$(diff <(echo "$want") <(echo "$have") | head -20)"
}

mkdir -p "$DEST"
for t in "${TEACHERS[@]}"; do
  read -r name url ref commit patch <<< "$t"
  dir=$DEST/$name
  [ -f "$PATCHES/$patch" ] || die "missing $PATCHES/$patch"
  if [ ! -e "$dir" ]; then
    [ "$MODE" = --check ] && die "$dir missing"
    say "$name: clone $url ($ref) @ ${commit:0:7}"
    git clone -q --branch "$ref" "$url" "$dir"
    git -C "$dir" -c advice.detachedHead=false checkout -q "$commit"
    git -C "$dir" apply --whitespace=nowarn "$PATCHES/$patch"
  fi
  verify "$dir" "$commit" "$PATCHES/$patch"
  say "$name: ${commit:0:7} + $patch ($(grep -c '^diff --git' "$PATCHES/$patch") files) at $dir"
done
