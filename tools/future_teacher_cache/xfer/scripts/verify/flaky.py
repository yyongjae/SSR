import http.server, os, re, sys
DATA=open('blob.bin','rb').read(); N=len(DATA); state={'n':0}
class H(http.server.BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*a): sys.stderr.write(f"[srv] {self.headers.get('Range')}\n")
    def do_GET(self):
        r=self.headers.get('Range'); start=0
        if r: start=int(re.match(r'bytes=(\d+)-',r).group(1))
        state['n']+=1
        self.send_response(206 if r else 200)
        if r: self.send_header('Content-Range',f'bytes {start}-{N-1}/{N}')
        self.send_header('Content-Length',str(N-start)); self.send_header('Accept-Ranges','bytes'); self.end_headers()
        end = N if state['n']>=3 else min(N, start+7_000_000)   # first 2 connections drop after 7 MB
        self.wfile.write(DATA[start:end]); self.wfile.flush()
        if end<N: self.connection.shutdown(2); self.close_connection=True
http.server.HTTPServer(('127.0.0.1',18777),H).serve_forever()
