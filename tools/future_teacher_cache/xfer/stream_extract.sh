#!/usr/bin/env bash
# Stream OpenScene v1.1 trainval sensor archives and keep only the listed members (report 40 §9-8 step (2)).
# Archive bytes never touch the disk: rstream.py (HTTP Range resume + sha256) | tar -xz -T <list>.
#
# usage: stream_extract.sh -l LIST_DIR -r SENSOR_ROOT -s STATE_DIR [-m camera|lidar|both] [-P N] [-R RETRIES]
#                          [-f INDEX_FILE] [--base-url URL] [--tree-dir DIR] [--extra-lists DIR]
#                          [--staging-dir DIR] [--python PY] [INDEX ...]
#   LIST_DIR     output of make_needed_files.py (needed_<mod>_<i>.txt, full member names)
#   SENSOR_ROOT  extraction root; files land at SENSOR_ROOT/trainval/<log>/<sensor>/<file> (--strip-components=2)
#   STATE_DIR    status/<mod>_<i>.json, members/ (tar -vv list = extracted files), sha/, logs/, status.log
#   INDEX        archive indices (0..199); also from -f FILE (whitespace separated); default: all indices with a list
#   -P N         archives processed in parallel (xargs -P); camera_i and lidar_i are separate jobs
#   -R N         attempts per archive within one run (default 3); rerunning the script retries failed archives
#   --extra-lists DIR  also extract DIR/extra_<mod>_<i>.txt in the same stream (e.g. provenance cmp test)
#   --staging-dir DIR  per-archive staging (default SENSOR_ROOT/.xfer_staging; must be on the same filesystem)
# Per archive: (1) skip if status says done or the list is empty, (2) stream into tar extracting into a fresh
# staging dir, (3) accept only if rstream rc 0, all bytes, sha256 == lfs.oid, tar rc 0, 0 'Not found in archive',
# every listed file present with the size tar reported and >0, no extra files, (4) hard-link files into SENSOR_ROOT
# without clobbering anything that exists, (5) delete staging. A failed attempt leaves nothing in SENSOR_ROOT.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REV=a76f840b65e972bc45e56c2adced897498e9a026

if [[ "${1:-}" != "__worker" ]]; then
  # ------------------------------------------------------------------ driver
  MOD=both; PAR=1; RETRIES=3; LIST_DIR=; ROOT=; STATE=; IDXFILE=; EXTRA=; STAGING=
  BASE_URL="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/$REV"
  TREE_DIR="$HERE/hf"; PY="${PY:-/home/external-user/miniconda3/envs/ssr/bin/python}"; IDX=()
  while (($#)); do
    case "$1" in
      -m) MOD=$2; shift 2;; -P) PAR=$2; shift 2;; -R) RETRIES=$2; shift 2;;
      -l) LIST_DIR=$2; shift 2;; -r) ROOT=$2; shift 2;; -s) STATE=$2; shift 2;; -f) IDXFILE=$2; shift 2;;
      --base-url) BASE_URL=$2; shift 2;; --tree-dir) TREE_DIR=$2; shift 2;; --extra-lists) EXTRA=$2; shift 2;;
      --staging-dir) STAGING=$2; shift 2;; --python) PY=$2; shift 2;;
      -h|--help) sed -n '2,19p' "$0"; exit 0;;
      -*) echo "unknown option $1" >&2; exit 64;;
      *) IDX+=("$1"); shift;;
    esac
  done
  [[ -n "$LIST_DIR" && -n "$ROOT" && -n "$STATE" ]] || { echo "need -l LIST_DIR -r SENSOR_ROOT -s STATE_DIR" >&2; exit 64; }
  [[ -n "$IDXFILE" ]] && IDX+=($(cat "$IDXFILE"))
  if ((${#IDX[@]} == 0)); then
    IDX=($(ls "$LIST_DIR" | sed -n 's/^needed_\(camera\|lidar\)_\([0-9]*\)\.txt$/\2/p' | sort -n -u))
  fi
  case "$MOD" in both) MODS="camera lidar";; camera|lidar) MODS=$MOD;; *) echo "bad -m $MOD" >&2; exit 64;; esac
  STAGING=${STAGING:-$ROOT/.xfer_staging}
  mkdir -p "$ROOT" "$STAGING" "$STATE"/{status,members,sha,logs,locks,lists}
  if [[ "$(stat -c %d "$ROOT")" != "$(stat -c %d "$STAGING")" ]]; then
    echo "staging dir $STAGING is not on the filesystem of $ROOT (hard-link commit needs that)" >&2; exit 64
  fi
  export SX_LIST_DIR="$(realpath "$LIST_DIR")" SX_ROOT="$(realpath "$ROOT")" SX_STATE="$(realpath "$STATE")" \
         SX_STAGING="$(realpath "$STAGING")" SX_BASE_URL="$BASE_URL" SX_TREE_DIR="$(realpath "$TREE_DIR")" \
         SX_EXTRA="${EXTRA:+$(realpath "$EXTRA")}" SX_PY="$PY" SX_RETRIES="$RETRIES" \
         SX_RUN_ID="$(date +%Y%m%dT%H%M%S)_$$"
  echo "[stream_extract] $(date -Is) archives=${#IDX[@]} mods='$MODS' P=$PAR root=$SX_ROOT state=$SX_STATE base=$BASE_URL" | tee -a "$SX_STATE/status.log"
  t0=$(date +%s.%N)
  for i in "${IDX[@]}"; do for m in $MODS; do echo "$m $i"; done; done \
    | xargs -P "$PAR" -n 2 bash "$0" __worker
  xrc=$?
  t1=$(date +%s.%N)
  # summary over the requested archives
  "$PY" - "$SX_STATE" "$t0" "$t1" "$MODS" "$SX_RUN_ID" "${IDX[@]}" <<'EOF'
import json, os, sys
st, t0, t1, mods, run_id, idx = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4].split(), sys.argv[5], sys.argv[6:]
c = {}; b = 0; nb = 0; fail = []
for i in idx:
    for m in mods:
        p = f"{st}/status/{m}_{i}.json"
        s = json.load(open(p)) if os.path.exists(p) else {"state": "missing"}
        c[s["state"]] = c.get(s["state"], 0) + 1
        if s["state"] != "done" and s["state"] != "skipped_empty": fail.append(f"{m}_{i}:{s['state']}")
        if s.get("run_id") == run_id and "bytes" in s: b += s["bytes"]; nb += 1
dt = t1 - t0
print(f"[stream_extract] summary {c} streamed_this_run={nb} archives {b/1e9:.2f} GB in {dt:.0f}s = {b/1e6/max(dt,1e-9):.2f} MB/s aggregate")
if fail: print("[stream_extract] NOT DONE:", " ".join(fail))
sys.exit(1 if fail else 0)
EOF
  src=$?
  exit $(( src != 0 || xrc != 0 ))
fi

# -------------------------------------------------------------------- worker: one archive
mod=$2; i=$3; key=${mod}_$i
name=openscene_sensor_trainval_${mod}_${i}.tgz
ST="$SX_STATE"; status=$ST/status/$key.json; logf=$ST/logs/$key.log
list=$SX_LIST_DIR/needed_${mod}_${i}.txt
log() { echo "[$(date -Is)] $key $*" | tee -a "$logf" >&2; }
# atomic per-archive status (+ one line in status.log)
write_status() {  # $1 = JSON object
  printf '%s\n' "$1" > "$status.tmp" && mv -f "$status.tmp" "$status"
  ( flock 9; printf '%s %s\n' "$(date -Is)" "$1" >> "$ST/status.log" ) 9>>"$ST/status.log.lock"
}

exec 8>"$ST/locks/$key.lock"
if ! flock -n 8; then log "locked by another process, skipping"; exit 0; fi
if [[ -f "$status" ]] && jq -e '.state=="done" or .state=="skipped_empty"' "$status" >/dev/null 2>&1; then
  log "already $(jq -r .state "$status"), skipping"; exit 0
fi

# combined list (needed + optional extra), unique
work=$ST/lists/$key.txt
{ [[ -f "$list" ]] && cat "$list"; [[ -n "$SX_EXTRA" && -f "$SX_EXTRA/extra_${mod}_${i}.txt" ]] && cat "$SX_EXTRA/extra_${mod}_${i}.txt"; } \
  | sed '/^$/d' | sort -u > "$work"
n_list=$(wc -l < "$work")
if [[ ! -f "$list" ]]; then
  write_status "{\"state\":\"failed\",\"key\":\"$key\",\"error\":\"no list $list\"}"; exit 1
fi
if (( n_list == 0 )); then
  write_status "{\"state\":\"skipped_empty\",\"key\":\"$key\",\"listed\":0}"; log "empty list, archive skipped"; exit 0
fi

tree=$SX_TREE_DIR/tree_openscene_sensor_trainval_${mod}.json
path=$(jq -r --arg f "/$name" '.[]|select(.path|endswith($f)).path' "$tree")
size=$(jq -r --arg f "/$name" '.[]|select(.path|endswith($f)).size' "$tree")
oid=$(jq -r --arg f "/$name" '.[]|select(.path|endswith($f)).lfs.oid' "$tree")
if [[ -z "$path" || -z "$size" || -z "$oid" || "$oid" == null ]]; then
  write_status "{\"state\":\"failed\",\"key\":\"$key\",\"error\":\"$name not in $tree\"}"; exit 1
fi
url="$SX_BASE_URL/$path"
stg=$SX_STAGING/$key

for attempt in $(seq 1 "$SX_RETRIES"); do
  rm -rf "$stg"; mkdir -p "$stg"; rm -f "$ST/sha/$key.sha" "$ST/logs/$key.rstream.json"
  log "attempt $attempt: $n_list files, $size bytes, $url"
  t0=$(date +%s.%N)
  "$SX_PY" "$HERE/rstream.py" "$url" "$size" --sha256-out "$ST/sha/$key.sha" --status-out "$ST/logs/$key.rstream.json" \
      2>> "$ST/logs/$key.rstream.err" \
    | tar -xzvv -C "$stg" --strip-components=2 --verbatim-files-from -T "$work" \
      > "$ST/members/$key.txt" 2> "$ST/logs/$key.tar.err"
  ps=("${PIPESTATUS[@]}"); rs_rc=${ps[0]}; tar_rc=${ps[1]}
  t1=$(date +%s.%N)
  sha=$(cut -d' ' -f1 "$ST/sha/$key.sha" 2>/dev/null || true)
  nbytes=$(jq -r .bytes "$ST/logs/$key.rstream.json" 2>/dev/null || echo 0)
  retr=$(jq -r .retries "$ST/logs/$key.rstream.json" 2>/dev/null || echo -1)
  notfound=$(grep -c 'Not found in archive' "$ST/logs/$key.tar.err" || true)
  # verify staging against the list and tar's own size listing; then commit by hard link (no clobber)
  ver=$("$SX_PY" - "$work" "$ST/members/$key.txt" "$stg" "$SX_ROOT" <<'EOF'
import json, os, re, sys
work, members, stg, root = sys.argv[1:5]
names = [l.rstrip("\n") for l in open(work) if l.strip()]
strip = lambda n: n.split("/", 2)[2]  # --strip-components=2
rx = re.compile(r"^\S+\s+\S+\s+(\d+)\s+\S+\s+\S+\s+(\S+)$")
tarsize = {}
for l in open(members):
    m = rx.match(l.rstrip("\n"))
    if m and not m.group(2).endswith("/"): tarsize[m.group(2)] = int(m.group(1))
bad = []; nbytes = 0
for n in names:
    p = os.path.join(stg, strip(n))
    try: s = os.path.getsize(p)
    except OSError: bad.append(("missing", n)); continue
    if s <= 0 or tarsize.get(n) != s: bad.append(("size", n, s, tarsize.get(n)))
    nbytes += s
want = {strip(n) for n in names}
extra = []
for d, _, fs in os.walk(stg):
    for f in fs:
        r = os.path.relpath(os.path.join(d, f), stg)
        if r not in want: extra.append(r)
print(json.dumps(dict(listed=len(names), extracted=len(names) - sum(1 for b in bad if b[0] == "missing"), bad=len(bad),
                      bad_examples=bad[:5], extra_files=len(extra), extra_examples=extra[:5], extracted_bytes=nbytes,
                      tar_listed=len(tarsize))))
EOF
)
  jq -e . <<<"$ver" >/dev/null 2>&1 || ver='{"error":"verify script failed","bad":999,"extra_files":0}'
  vbad=$(jq -r '.bad + .extra_files' <<<"$ver")
  ok=1; why=()
  [[ "$rs_rc" == 0 ]] || { ok=0; why+=("rstream_rc=$rs_rc"); }
  [[ "$nbytes" == "$size" ]] || { ok=0; why+=("bytes=$nbytes!=$size"); }
  [[ "$sha" == "$oid" ]] || { ok=0; why+=("sha_mismatch"); }
  [[ "$tar_rc" == 0 ]] || { ok=0; why+=("tar_rc=$tar_rc"); }
  [[ "$notfound" == 0 ]] || { ok=0; why+=("not_found=$notfound"); }
  [[ "$vbad" == 0 ]] || { ok=0; why+=("verify_bad=$vbad"); }
  sec=$("$SX_PY" -c "print(round($t1-$t0,3))")
  common="\"key\":\"$key\",\"archive\":\"$name\",\"attempt\":$attempt,\"bytes\":$nbytes,\"size\":$size,\"sec\":$sec,\"MBps\":$("$SX_PY" -c "print(round($nbytes/1e6/max($sec,1e-9),3))"),\"rstream_rc\":$rs_rc,\"rstream_retries\":$retr,\"sha256\":\"$sha\",\"lfs_oid\":\"$oid\",\"sha_ok\":$([[ "$sha" == "$oid" ]] && echo true || echo false),\"tar_rc\":$tar_rc,\"not_found\":$notfound,\"verify\":$ver,\"run_id\":\"$SX_RUN_ID\""
  if (( ok )); then
    com=$("$SX_PY" - "$work" "$stg" "$SX_ROOT" <<'EOF'
import json, os, sys
work, stg, root = sys.argv[1:4]
new = present_same = conflict = 0; ex = []
for n in (l.rstrip("\n") for l in open(work) if l.strip()):
    r = n.split("/", 2)[2]; src, dst = os.path.join(stg, r), os.path.join(root, r)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst); new += 1                      # atomic, never overwrites
    except FileExistsError:
        if os.path.getsize(dst) == os.path.getsize(src): present_same += 1
        else: conflict += 1; ex.append(r)
# final check in the real root
miss = sum(1 for n in (l.rstrip("\n") for l in open(work) if l.strip()) if not (os.path.isfile(os.path.join(root, n.split("/", 2)[2])) and os.path.getsize(os.path.join(root, n.split("/", 2)[2])) > 0))
print(json.dumps(dict(committed_new=new, already_present_same_size=present_same, conflict_different_size=conflict, conflict_examples=ex[:5], missing_after_commit=miss)))
EOF
)
    # a crash of the commit step (e.g. hard links refused) must still leave a valid status JSON marked failed
    jq -e '.committed_new' <<<"$com" >/dev/null 2>&1 || \
      com='{"error":"commit script failed","committed_new":0,"already_present_same_size":0,"conflict_different_size":-1,"conflict_examples":[],"missing_after_commit":-1}'
    rm -rf "$stg"
    if [[ "$(jq -r '.conflict_different_size + .missing_after_commit' <<<"$com")" == 0 ]]; then
      write_status "{\"state\":\"done\",$common,\"commit\":$com}"
      log "done in ${sec}s ($nbytes B, sha ok, $(jq -c . <<<"$com"))"; exit 0
    fi
    write_status "{\"state\":\"failed\",$common,\"commit\":$com,\"why\":\"commit\"}"
    log "COMMIT PROBLEM $(jq -c . <<<"$com") -- not retried (needs inspection)"; exit 1
  fi
  rm -rf "$stg"
  write_status "{\"state\":\"failed\",$common,\"why\":\"${why[*]}\"}"
  log "attempt $attempt FAILED: ${why[*]}"
  sleep $(( attempt * ${SX_RETRY_SLEEP:-10} ))
done
exit 1
