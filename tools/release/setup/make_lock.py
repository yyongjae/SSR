#!/usr/bin/env python
"""Write requirements.lock.txt from the running environment.

  /path/to/env/bin/python -m pip freeze | python tools/release/setup/make_lock.py > requirements.lock.txt

What it changes from `pip freeze`:
  - `-e ...` lines are dropped (the repo itself; machine-specific)
  - `name @ file:///...` lines, which name a conda build directory, become `name==<installed version>`
  - `name @ git+https://...` lines stay (nuplan-devkit, pinned to its commit)
  - a torch wheel index line goes on top, because of the +cu128 versions
"""
from __future__ import annotations

import sys
from importlib import metadata
from typing import Callable, Iterable, List

HEADER = """\
# Exact package versions of the environment behind the reference runs (python 3.9.23).  Written by
# tools/release/setup/make_lock.py from `pip freeze`: editable lines dropped, conda file:// lines turned into name==version.
# tools/release/setup/install_env.sh is the tested way to build it; this file is the record.  Installing from this file alone
# (pip install --no-deps -r requirements.lock.txt, then pip install -e . --no-deps) has not been tried.
--extra-index-url https://download.pytorch.org/whl/cu128
"""


def convert(lines: Iterable[str], version: Callable[[str], str] = metadata.version) -> List[str]:
    out = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("-e "):
            continue
        if " @ file://" in s:
            name = s.split(" @ ", 1)[0].strip()
            s = f"{name}=={version(name)}"
        out.append(s)
    return out


def main() -> int:
    sys.stdout.write(HEADER)
    sys.stdout.write("\n".join(convert(sys.stdin)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
