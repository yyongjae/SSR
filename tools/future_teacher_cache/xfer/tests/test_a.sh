#!/usr/bin/env bash
# TEST A (synthetic, no network): fake OpenScene-like tgz archives served by flaky_server.py (Range, 302 hop,
# mid-stream drops). usage: test_a.sh WORKDIR
set -uo pipefail
X="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W=${1:?workdir}; PY=${PY:-/home/external-user/miniconda3/envs/ssr/bin/python}
PORT=${PORT:-18765}
rm -rf "$W"; mkdir -p "$W"/{src,srv,tree,lists,root}
export SX_RETRY_SLEEP=1
pass=0; fail=0
check() { if eval "$2"; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1   [$2]"; fail=$((fail+1)); fi; }

# ---- fake archives: 4 indices x {camera,lidar}, 2 logs each, 12 frames per log, random (incompressible) payloads
"$PY" - "$W" <<'EOF'
import hashlib, io, json, os, random, sys, tarfile
W = sys.argv[1]; R = random.Random(0)
tree = {"camera": [], "lidar": []}; amap = {}
for i in range(4):
    logs = [f"2021.0{i+1}.01.00.00.00_veh-{i:02d}_{j:05d}_{j+99:05d}" for j in range(2)]
    amap[f"openscene_sensor_trainval_{i}"] = logs
    for mod in ("camera", "lidar"):
        d = f"{W}/srv/openscene-v1.1/openscene_sensor_trainval_{mod}"; os.makedirs(d, exist_ok=True)
        fn = f"{d}/openscene_sensor_trainval_{mod}_{i}.tgz"
        with tarfile.open(fn, "w:gz", format=tarfile.GNU_FORMAT) as tf:
            for log in logs:
                for k in range(12):
                    tok = f"{i:02d}{k:02d}{hashlib.md5(log.encode()).hexdigest()[:12]}"
                    sensors = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"] if mod == "camera" else ["MergedPointCloud"]
                    for s in sensors:
                        ext = "jpg" if mod == "camera" else "pcd"
                        name = f"openscene-v1.1/sensor_blobs/trainval/{log}/{s}/{tok}.{ext}"
                        data = R.randbytes(R.randint(20000, 120000))
                        os.makedirs(os.path.dirname(f"{W}/src/{name}"), exist_ok=True)
                        open(f"{W}/src/{name}", "wb").write(data)
                        ti = tarfile.TarInfo(name); ti.size = len(data); ti.mode = 0o777
                        tf.addfile(ti, io.BytesIO(data))
        b = open(fn, "rb").read()
        tree[mod].append({"type": "file", "size": len(b), "lfs": {"oid": hashlib.sha256(b).hexdigest(), "size": len(b)},
                          "path": f"openscene-v1.1/openscene_sensor_trainval_{mod}/openscene_sensor_trainval_{mod}_{i}.tgz"})
for mod in tree:
    json.dump(tree[mod], open(f"{W}/tree/tree_openscene_sensor_trainval_{mod}.json", "w"))
json.dump(amap, open(f"{W}/map.json", "w"))
EOF
ls -l "$W"/srv/openscene-v1.1/*/ | grep tgz | awk '{s+=$5} END {print "fake archives total bytes", s}'

# ---- needed lists: every 3rd member of each archive (F0/L0/R0 only for camera)
for f in "$W"/srv/openscene-v1.1/*/*.tgz; do
  b=$(basename "$f" .tgz); mod=$(echo "$b" | cut -d_ -f4); i=${b##*_}
  tar -tzf "$f" | grep -v '/CAM_B0/' | awk 'NR%3==1' > "$W/lists/needed_${mod}_$i.txt"
done
: > "$W/lists/needed_camera_2.txt"                                                   # empty list -> archive skipped
echo "openscene-v1.1/sensor_blobs/trainval/no_such_log/MergedPointCloud/x.pcd" >> "$W/lists/needed_lidar_2.txt"  # -> Not found
cp "$W/tree/tree_openscene_sensor_trainval_camera.json" "$W/tree/camera.good"
jq '(.[]|select(.path|endswith("_camera_3.tgz")).lfs.oid) |= "0000"' "$W/tree/camera.good" > "$W/tree/tree_openscene_sensor_trainval_camera.json"  # -> sha mismatch
# pre-existing file in root for lidar_3: same name, different content+size -> must not be overwritten (conflict)
c3=$(sed -n 1p "$W/lists/needed_lidar_3.txt"); pre="$W/root/${c3#openscene-v1.1/sensor_blobs/}"
mkdir -p "$(dirname "$pre")"; echo "preexisting" > "$pre"; pre_sha=$(sha256sum < "$pre")

# ---- server: every file's first 2 responses are cut after 150 kB
"$PY" "$X/tests/flaky_server.py" "$W/srv" $PORT --drop-after 150000 --drops 2 --log "$W/server.log" & SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT; sleep 1
SE() { bash "$X/stream_extract.sh" -l "$W/lists" -r "$W/root" -s "$W/state" --base-url "http://127.0.0.1:$PORT/resolve" --tree-dir "$W/tree" -R 2 "$@"; }

echo "== run 1 (P=4, all 8 archives)"
SE -P 4 0 1 2 3 > "$W/run1.out" 2> "$W/run1.err"; rc1=$?
tail -3 "$W/run1.out"
st() { jq -r .state "$W/state/status/$1.json"; }
check "run1 exit != 0 (failures expected)" '[[ $rc1 != 0 ]]'
for k in camera_0 lidar_0 camera_1 lidar_1; do check "$k done" '[[ $(st $k) == done ]]'; done
check "camera_2 skipped_empty" '[[ $(st camera_2) == skipped_empty ]]'
check "lidar_2 failed (not found)" '[[ $(st lidar_2) == failed && $(jq .not_found $W/state/status/lidar_2.json) == 1 && $(jq .tar_rc $W/state/status/lidar_2.json) == 2 ]]'
check "camera_3 failed (sha)" '[[ $(st camera_3) == failed && $(jq .sha_ok $W/state/status/camera_3.json) == false ]]'
check "lidar_3 failed (commit conflict)" '[[ $(st lidar_3) == failed && $(jq .commit.conflict_different_size $W/state/status/lidar_3.json) == 1 ]]'
check "preexisting file untouched" '[[ "$(sha256sum < "$pre")" == "$pre_sha" ]]'
check "resume happened (rstream retries>0 on done archives)" '[[ $(jq -s "[.[]|select(.state==\"done\")|.rstream_retries]|min" $W/state/status/*_[01].json) -ge 2 ]]'
check "server saw Range resume requests" 'grep -q "bytes=150000-" $W/server.log'
check "sha ok on done archives" '[[ $(jq -s "[.[]|select(.state==\"done\")|.sha_ok]|all" $W/state/status/*.json) == true ]]'
check "no staging leftovers" '[[ -z "$(ls -A $W/root/.xfer_staging)" ]]'
check "camera_3 (sha) / lidar_2 (not found) placed nothing in root" '[[ -z "$(find $W/root/trainval -type f \( -path "*veh-03*/CAM_*" -o -path "*veh-02*/MergedPointCloud/*" \) | head -1)" ]]'
check "lidar_3 committed its other files (sha-verified) except the conflict" '[[ $(jq .commit.committed_new $W/state/status/lidar_3.json) == $(( $(wc -l < $W/lists/needed_lidar_3.txt) - 1 )) ]]'
# extracted files == union of lists of done archives, byte-identical to source
"$PY" - "$W" <<'EOF' > "$W/verify1.txt"
import filecmp, glob, json, os, sys
W = sys.argv[1]; want = set()
for s in glob.glob(f"{W}/state/status/*.json"):
    d = json.load(open(s))
    if d["state"] == "done" or d.get("why") == "commit":
        want |= {l.strip().split("/", 2)[2] for l in open(f"{W}/lists/needed_{d['key']}.txt") if l.strip()}
have = set()
for d, _, fs in os.walk(f"{W}/root/trainval"):
    have |= {os.path.relpath(os.path.join(d, f), f"{W}/root") for f in fs}
pre = {h for h in have if open(f"{W}/root/{h}", "rb").read() == b"preexisting\n"}
want -= pre
same = all(filecmp.cmp(f"{W}/root/{r}", f"{W}/src/openscene-v1.1/sensor_blobs/{r}", shallow=False) for r in want)
print(len(want), len(have - pre), len(have - pre - want), len(want - have), same)
EOF
read nw nh nextra nmiss same < "$W/verify1.txt"
check "extracted == listed for done archives ($nw files), 0 extra, 0 missing, byte-identical" '[[ $nw -gt 0 && $nw == $nh && $nextra == 0 && $nmiss == 0 && $same == True ]]'

echo "== run 2 (rerun unchanged: done archives skipped, failures retried and still fail)"
nreq0=$(wc -l < "$W/server.log")
SE -P 4 0 1 2 3 > "$W/run2.out" 2> "$W/run2.err"; rc2=$?
check "run2 exit != 0" '[[ $rc2 != 0 ]]'
check "done archives not re-requested" '[[ -z "$(tail -n +$((nreq0+1)) $W/server.log | grep -E "_(camera|lidar)_[01]\.tgz")" ]]'
check "failed archives retried" '[[ -n "$(tail -n +$((nreq0+1)) $W/server.log | grep -E "lidar_2\.tgz")" ]]'
check "status attempts rewritten for lidar_2" '[[ $(jq .attempt $W/state/status/lidar_2.json) == 2 ]]'

echo "== run 3 (fix list / tree / conflict, rerun all -> everything done)"
sed -i '/no_such_log/d' "$W/lists/needed_lidar_2.txt"
cp "$W/tree/camera.good" "$W/tree/tree_openscene_sensor_trainval_camera.json"
rm -f "$pre"
SE -P 4 0 1 2 3 > "$W/run3.out" 2> "$W/run3.err"; rc3=$?
tail -2 "$W/run3.out"
check "run3 exit 0" '[[ $rc3 == 0 ]]'
check "all done/skipped" '[[ $(jq -s "[.[]|select(.state==\"done\" or .state==\"skipped_empty\")]|length" $W/state/status/*.json) == 8 ]]'

echo "== run 4 (kill mid-stream, then resume at archive granularity), P=2, slow server"
kill $SRV; sleep 0.5
"$PY" "$X/tests/flaky_server.py" "$W/srv" $PORT --log "$W/server.log" --rate 2000000 & SRV=$!; sleep 1
rm -rf "$W/state2" "$W/root2"; mkdir -p "$W/root2"
setsid bash "$X/stream_extract.sh" -l "$W/lists" -r "$W/root2" -s "$W/state2" --base-url "http://127.0.0.1:$PORT/resolve" --tree-dir "$W/tree" -P 2 0 1 2 3 > "$W/run4a.out" 2>&1 & J=$!
# wait until the first archive is done, then kill the whole tree
for _ in $(seq 100); do ls "$W/state2/status/"*.json >/dev/null 2>&1 && grep -q '"state":"done"' "$W"/state2/status/*.json && break; sleep 0.1; done
kill -KILL -- -$J; sleep 0.5   # whole process group: driver, xargs, workers, rstream, tar
check "run4 nothing of the killed run still alive" '! pgrep -g $J >/dev/null'
ndone_a=$(cat "$W"/state2/status/*.json 2>/dev/null | grep -c '"state":"done"' || true)
echo "killed after $ndone_a done"
bash "$X/stream_extract.sh" -l "$W/lists" -r "$W/root2" -s "$W/state2" --base-url "http://127.0.0.1:$PORT/resolve" --tree-dir "$W/tree" -P 3 0 1 2 3 > "$W/run4b.out" 2>&1; rc4=$?
tail -2 "$W/run4b.out"
check "run4 resume exit 0" '[[ $rc4 == 0 ]]'
check "run4 killed with partial progress ($ndone_a done)" '[[ $ndone_a -ge 1 && $ndone_a -lt 7 ]]'
check "run4 root2 identical to run3 root" 'diff <(cd $W/root/trainval && find . -type f -exec sha256sum {} + | sort) <(cd $W/root2/trainval && find . -type f -exec sha256sum {} + | sort) >/dev/null'
check "run4 staging empty" '[[ -z "$(ls -A $W/root2/.xfer_staging)" ]]'
echo "TEST A: $pass passed, $fail failed"
exit $((fail > 0))
