"""Resumable HTTP stream to stdout: on any error reconnect with Range: bytes=<offset>- and continue (no bytes on disk).

usage: python rstream.py URL TOTAL_BYTES [--sha256-out FILE] [--status-out FILE] [--max-retries N] > stdout

- Writes exactly TOTAL_BYTES to stdout (never more), hashing them with sha256 on the way.
- --sha256-out: '<hex>  <bytes>' written only when all TOTAL_BYTES were streamed.
- --status-out: JSON {ok, bytes, total, sec, MBps, retries, rc, error}.
- Exit codes: 0 ok, 3 retries exhausted, 4 downstream closed (EPIPE), 5 server sent more than TOTAL / bad Content-Range.
"""
import argparse, hashlib, json, os, re, sys, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument("url"); p.add_argument("total", type=int)
p.add_argument("--sha256-out"); p.add_argument("--status-out")
p.add_argument("--max-retries", type=int, default=200)
p.add_argument("--chunk", type=int, default=1 << 20)
a = p.parse_args()
url, total = a.url, a.total
out = sys.stdout.buffer; off = 0; retries = 0; t0 = time.time(); h = hashlib.sha256()


def finish(rc, err=None):
    sec = time.time() - t0
    st = dict(ok=rc == 0, rc=rc, bytes=off, total=total, sec=round(sec, 3), MBps=round(off / 1e6 / max(sec, 1e-9), 3),
              retries=retries, error=err, url=url)
    if rc == 0 and a.sha256_out:
        with open(a.sha256_out, "w") as f:
            f.write(f"{h.hexdigest()}  {off}\n")
    if a.status_out:
        with open(a.status_out, "w") as f:
            json.dump(st, f)
    print(f"[rstream] {'done' if rc == 0 else 'FAIL'} rc={rc} bytes={off}/{total} sec={sec:.1f} MBps={st['MBps']} retries={retries} err={err}",
          file=sys.stderr, flush=True)
    os._exit(rc)  # no implicit flush of a possibly broken stdout


while off < total:
    try:
        req = urllib.request.Request(url, headers={"Range": f"bytes={off}-", "User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=60) as r:
            if off:
                if r.status != 206:
                    raise RuntimeError(f"no 206: {r.status}")
                m = re.match(r"bytes (\d+)-", r.headers.get("Content-Range", ""))
                if not m or int(m.group(1)) != off:
                    raise RuntimeError(f"bad Content-Range {r.headers.get('Content-Range')!r} for offset {off}")
            while True:
                b = r.read(a.chunk)
                if not b:
                    break
                if off + len(b) > total:
                    finish(5, f"server sent more than total ({off + len(b)} > {total})")
                out.write(b); h.update(b); off += len(b)
            if off < total:
                raise ConnectionError(f"premature end of body at {off}")
    except BrokenPipeError:
        finish(4, "downstream closed")
    except Exception as e:
        retries += 1
        print(f"[rstream] retry {retries} at {off} ({e!r})", file=sys.stderr, flush=True)
        if retries > a.max_retries:
            finish(3, f"retries exhausted: {e!r}")
        time.sleep(min(30, 2 * retries))
try:
    out.flush()
except BrokenPipeError:
    finish(4, "downstream closed at flush")
finish(0)
