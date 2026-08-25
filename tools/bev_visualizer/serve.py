#!/usr/bin/env python3
"""Serve the generated static site on localhost only."""

import argparse
import http.server
import socketserver
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument(
        "--directory",
        default="/home/yongjae/e2e/SSR/work_dirs/bev_visualizer/site")
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    if not (directory / "data" / "manifest.json").is_file():
        raise SystemExit(f"generated site is missing: {directory}")
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(
        *a, directory=str(directory), **kw)
    with socketserver.TCPServer((args.bind, args.port), handler) as server:
        print(f"Serving {directory} at http://{args.bind}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
