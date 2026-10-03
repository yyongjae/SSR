"""CPU bench (no network, nothing written to disk except a tiny name list):
build an in-memory tar.gz from ~1 GB of local navtrain sensor files laid out like the OpenScene archive,
then time (1) gzip -dc, (2) pigz -dc, (3) tar -xz -O -T list (only ~10% members selected) with name lists of
different lengths, (4) python tarfile streaming with a set lookup. Output -> bench_tar.json"""
import os, io, tarfile, gzip, subprocess, time, json, random, tempfile
S = os.path.dirname(os.path.abspath(__file__))
B = "/home/external-user/navsim/download/trainval_sensor_blobs/trainval"
SUBS = ["CAM_F0","CAM_L0","CAM_R0","CAM_L1","CAM_R1","CAM_L2","CAM_R2","CAM_B0","MergedPointCloud"]
logs = sorted(os.listdir(B)); random.seed(0); random.shuffle(logs)
files = []; tot = 0
for lg in logs:
    for s in SUBS:
        d = f"{B}/{lg}/{s}"
        for f in sorted(os.listdir(d)):
            p = f"{d}/{f}"; files.append((p, f"openscene-v1.1/sensor_blobs/trainval/{lg}/{s}/{f}")); tot += os.path.getsize(p)
    if tot > 1.0e9: break
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tf:
    for p, a in files: tf.add(p, arcname=a)
raw = buf.getvalue(); t = time.time(); gz = gzip.compress(raw, compresslevel=6); tc = time.time() - t
names = [a for _, a in files]; sel = names[::10]
res = dict(n_members=len(names), tar_bytes=len(raw), gz_bytes=len(gz), ratio=len(raw)/len(gz), py_gzip6_compress_s=tc)
def run(cmd, inp):
    t = time.time(); p = subprocess.run(cmd, input=inp, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE); return time.time() - t, p.returncode
for name, cmd in [("gzip_dc", ["gzip", "-dc"]), ("pigz_dc", ["pigz", "-dc"])]:
    dt, rc = run(cmd, gz); res[name] = dict(s=dt, MBps_compressed=len(gz)/1e6/dt, MBps_out=len(raw)/1e6/dt, rc=rc)
for nlist in (len(sel), 6000, 60000, 300000):
    lst = list(sel) + [f"openscene-v1.1/sensor_blobs/trainval/fake_{i:07d}/CAM_F0/x{i}.jpg" for i in range(max(0, nlist - len(sel)))]
    fd, path = tempfile.mkstemp(dir=S, suffix=".lst"); os.write(fd, ("\n".join(lst) + "\n").encode()); os.close(fd)
    dt, rc = run(["tar", "-xz", "-O", "-T", path], gz); os.remove(path)
    res[f"tar_xz_T_{nlist}"] = dict(s=dt, MBps_compressed=len(gz)/1e6/dt, rc=rc, members_per_s=len(names)/dt)
    if dt > 600: break
# python tarfile streaming with set lookup (what a custom extractor would do)
want = set(sel); t = time.time(); n = 0
with tarfile.open(fileobj=io.BytesIO(gz), mode="r|gz") as tf:
    for m in tf:
        if m.name in want:
            f = tf.extractfile(m); f.read(); n += 1
dt = time.time() - t; res["py_tarfile_stream_set"] = dict(s=dt, MBps_compressed=len(gz)/1e6/dt, extracted=n)
print(json.dumps(res, indent=1)); json.dump(res, open(f"{S}/bench_tar.json", "w"), indent=1)
