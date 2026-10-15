import sys, zlib, json
data = sys.stdin.buffer.read()
d = zlib.decompressobj(16 + zlib.MAX_WBITS)
raw = d.decompress(data)
out = []
pos = 0
longname = None
while pos + 512 <= len(raw):
    h = raw[pos:pos+512]
    if h == b'\0'*512: break
    name = h[0:100].rstrip(b'\0').decode('utf-8','replace')
    prefix = h[345:500].rstrip(b'\0').decode('utf-8','replace')
    if prefix: name = prefix + '/' + name
    sz = h[124:136].rstrip(b'\0 ').decode()
    size = int(sz, 8) if sz else 0
    typ = chr(h[156]) if h[156] else '0'
    if longname: name, longname = longname, None
    if typ == 'L':
        longname = raw[pos+512:pos+512+size].rstrip(b'\0').decode(); pos += 512 + ((size+511)//512)*512; continue
    out.append([name, typ, size])
    pos += 512 + ((size+511)//512)*512
print(json.dumps({'compressed_read': len(data), 'decompressed': len(raw), 'members': out}))
