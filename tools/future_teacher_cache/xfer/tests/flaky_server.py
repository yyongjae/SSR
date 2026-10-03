"""Test HTTP server for stream_extract.sh: serves files from a dir with Range support, a 302 hop (like HF ->
xet CDN), and connection drops mid-stream.

usage: python flaky_server.py DIR PORT [--drop-after BYTES] [--drops N] [--log FILE] [--rate BYTES_PER_S]
  Every response for a file is cut after --drop-after bytes (connection closed, no error) for the first --drops
  responses of that file; later responses are served fully. /resolve/<path> answers 302 -> /cdn/<path>.
"""
import argparse, http.server, json, os, re, socketserver, threading, time

p = argparse.ArgumentParser()
p.add_argument("dir"); p.add_argument("port", type=int)
p.add_argument("--drop-after", type=int, default=0); p.add_argument("--drops", type=int, default=0)
p.add_argument("--log", default=None)
p.add_argument("--rate", type=float, default=0, help="bytes/s per connection (0 = unlimited)")
a = p.parse_args()
lock = threading.Lock(); served = {}


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if a.log:
            with lock, open(a.log, "a") as f:
                f.write(json.dumps(dict(path=self.path, range=self.headers.get("Range"), msg=fmt % args)) + "\n")

    def do_GET(self):
        if self.path.startswith("/resolve/"):
            self.send_response(302); self.send_header("Location", "/cdn/" + self.path[len("/resolve/"):])
            self.send_header("Content-Length", "0"); self.end_headers(); return
        if not self.path.startswith("/cdn/"):
            self.send_error(404); return
        fp = os.path.join(a.dir, self.path[len("/cdn/"):])
        if not os.path.isfile(fp):
            self.send_error(404); return
        size = os.path.getsize(fp); start = 0
        m = re.match(r"bytes=(\d+)-", self.headers.get("Range", ""))
        with lock:
            n = served.get(fp, 0); served[fp] = n + 1
        drop = n < a.drops and a.drop_after > 0
        if m:
            start = int(m.group(1))
            self.send_response(206); self.send_header("Content-Range", f"bytes {start}-{size-1}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(size - start)); self.end_headers()
        sent = 0; limit = a.drop_after if drop else None
        with open(fp, "rb") as f:
            f.seek(start)
            while True:
                b = f.read(65536)
                if not b: break
                if limit is not None and sent + len(b) > limit:
                    b = b[: limit - sent]
                    try: self.wfile.write(b); self.wfile.flush()
                    except Exception: pass
                    self.close_connection = True
                    self.connection.shutdown(2)  # abrupt mid-body close
                    return
                try: self.wfile.write(b)
                except Exception: return
                sent += len(b)
                if a.rate: time.sleep(len(b) / a.rate)


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True


S(("127.0.0.1", a.port), H).serve_forever()
