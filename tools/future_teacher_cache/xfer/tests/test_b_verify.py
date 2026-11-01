"""TEST B verification (real archive pair). usage: test_b_verify.py LIST_DIR EXTRA_DIR STATE_DIR EXTRACT_ROOT REAL_ROOT INDEX"""
import filecmp, json, os, sys

lists, extra, state, root, real, i = sys.argv[1:7]
res = {}
want_all = set()
for mod in ("camera", "lidar"):
    st = json.load(open(f"{state}/status/{mod}_{i}.json"))
    need = [l.strip() for l in open(f"{lists}/needed_{mod}_{i}.txt") if l.strip()]
    ext = [l.strip() for l in open(f"{extra}/extra_{mod}_{i}.txt") if l.strip()]
    rel = lambda n: n.split("/", 2)[2]
    sizes = [os.path.getsize(f"{root}/{rel(n)}") if os.path.isfile(f"{root}/{rel(n)}") else -1 for n in need]
    cmp_ok = sum(filecmp.cmp(f"{root}/{rel(n)}", f"{real}/{rel(n)}", shallow=False) for n in ext)
    need_on_real = sum(os.path.exists(f"{real}/{rel(n)}") for n in need)
    want_all |= {rel(n) for n in need + ext}
    res[mod] = dict(state=st["state"], sha_ok=st.get("sha_ok"), sha256=st.get("sha256"), lfs_oid=st.get("lfs_oid"),
                    bytes=st.get("bytes"), sec=st.get("sec"), MBps=st.get("MBps"), retries=st.get("rstream_retries"),
                    tar_rc=st.get("tar_rc"), not_found=st.get("not_found"),
                    needed=len(need), needed_extracted_size_gt0=sum(s > 0 for s in sizes), needed_bytes=sum(s for s in sizes if s > 0),
                    needed_present_on_real_disk=need_on_real,
                    provenance_files=len(ext), provenance_cmp_identical=cmp_ok)
have = set()
for d, _, fs in os.walk(f"{root}/trainval"):
    have |= {os.path.relpath(os.path.join(d, f), root) for f in fs}
res["files_in_extract_root"] = len(have)
res["extra_files_not_listed"] = len(have - want_all)
res["listed_missing"] = len(want_all - have)
print(json.dumps(res, indent=1))
ok = all(res[m]["state"] == "done" and res[m]["sha_ok"] and res[m]["needed_extracted_size_gt0"] == res[m]["needed"]
         and res[m]["provenance_cmp_identical"] == res[m]["provenance_files"] and res[m]["needed_present_on_real_disk"] == 0
         for m in ("camera", "lidar")) and res["extra_files_not_listed"] == 0 and res["listed_missing"] == 0
print("TEST B:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
