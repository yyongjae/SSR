#!/usr/bin/env python
"""SimScale download (Hugging Face dataset OpenDriveLab/SimScale): meta_datas and sensor_blobs_hist of the planner-based
pseudo-expert rounds, unpacked into SSR_SIMSCALE_ROOT as the archives lay them out.

  simscale_download.py [--rounds 0 1 2 3 4] [--root DIR] [--download-dir DIR] [--plan-only] [--keep-archives]

Per round r, from SimScale_data/synthetic_reaction_pdm_v1.0-<r>/:
  simscale_pdm_v1.0-<r>_meta_datas.tar.gz              -> <root>/synthetic_reaction_pdm_v1.0-<r>/meta_datas/*.pkl
  simscale_pdm_v1.0-<r>_sensor_blobs_hist.json         -> <root>/synthetic_reaction_pdm_v1.0-<r>/ (scenes of every part)
  simscale_pdm_v1.0-<r>_sensor_blobs_hist/..._<k>.tar.gz -> <root>/synthetic_reaction_pdm_v1.0-<r>/sensor_blobs_hist/<scene>/
The archives hold SimScale/synthetic_reaction_pdm_v1.0-<r>/...: only that subtree is extracted, without its first
component.  sensor_blobs_fut is not downloaded (no training input reads future images).

An archive counts as done only when: its size equals the size in the server listing (curl -C - resumes a partial file,
a larger one starts over); tar extracts it with exit 0 (gzip CRC); a meta archive gives one scene pickle per scene of the
round's json, a hist part every scene directory of its json entry, each with the CAM_F0 / CAM_L0 / CAM_R0 images of the 4
history frames.  Then <download-dir>/.done/<archive> is written and the archive deleted (--keep-archives keeps it); a
rerun skips stamped archives, so the command resumes.  Extraction of one archive overlaps the next download.  Refused
when the free disk is below the bytes left + 5 % + 50 GB.  HF_TOKEN (optional) is sent when set.  Every listing and
file is read at the pinned dataset commit REVISION (the release this recipe was checked on), not at main.
Size for rounds 0-4: ~2.23 TB (meta 12.8 GB, hist 2.21 TB in 241 parts), ~6 h at 100 MB/s [estimate from the sizes].
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]      # repo root of this worktree
sys.path.insert(0, str(ROOT))
from navsim.common import ssr_paths as SP  # noqa: E402

HF = "https://huggingface.co"
REPO_ID = "OpenDriveLab/SimScale"
REVISION = "06a4fc88f48d8c3ac7c16cc1df05f3caab6397ae"   # dataset commit of 2026-01-25 (the release used)
DATA_DIR = "SimScale_data"
ROUNDS = (0, 1, 2, 3, 4)
CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0")
N_HIST = 4                                       # history frames with images per scene
SPACE_MARGIN = (1.05, 50e9)                      # free disk needed: left * 1.05 + 50 GB


def round_dir(r: int) -> str:
    return f"synthetic_reaction_pdm_v1.0-{int(r)}"


def prefix(r: int) -> str:
    return f"simscale_pdm_v1.0-{int(r)}"


# ----------------------------------------------------------------------------------------------- listing
def _get_json(url: str, token: str = ""):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def hf_tree(path: str, token: str = "", get=None) -> Dict[str, int]:
    """{file path: size} of one directory of the dataset repo (the HF tree API; get: a stand-in for tests)."""
    get = get or _get_json
    return {x["path"]: int(x["size"]) for x in get(f"{HF}/api/datasets/{REPO_ID}/tree/{REVISION}/{path}", token)
            if x.get("type") == "file"}


def plan(rounds, token: str = "", get=None) -> List[dict]:
    """Archives of the rounds in download order: per round the part index, the meta archive, then the hist parts."""
    out = []
    for r in rounds:
        d = f"{DATA_DIR}/{round_dir(r)}"
        top = hf_tree(d, token, get)
        hist = hf_tree(f"{d}/{prefix(r)}_sensor_blobs_hist", token, get)
        idx, meta = f"{d}/{prefix(r)}_sensor_blobs_hist.json", f"{d}/{prefix(r)}_meta_datas.tar.gz"
        for p in (idx, meta):
            if p not in top:
                raise SystemExit(f"round {r}: {p} is not in the server listing")
        parts = sorted((p for p in hist if p.endswith(".tar.gz")), key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
        if not parts:
            raise SystemExit(f"round {r}: no sensor_blobs_hist parts in the server listing")
        out.append(dict(round=r, kind="index", path=idx, size=top[idx]))
        out.append(dict(round=r, kind="meta", path=meta, size=top[meta]))
        out += [dict(round=r, kind="hist", path=p, size=hist[p], part=Path(p).name[: -len(".tar.gz")]) for p in parts]
    for a in out:
        a["name"] = Path(a["path"]).name
        a["url"] = f"{HF}/datasets/{REPO_ID}/resolve/{REVISION}/{a['path']}"
    return out


# ----------------------------------------------------------------------------------------------- download / unpack
def download(a: dict, dl: Path, token: str = "", tries: int = 10) -> Path:
    """The archive at <dl>/<name>, resumed with curl -C - until its size is the listed one."""
    f = dl / a["name"]
    hdr = ["-H", f"Authorization: Bearer {token}"] if token else []
    for i in range(tries):
        subprocess.run(["curl", "-L", "-sS", "-C", "-", "--retry", "20", "--retry-all-errors", "--retry-delay", "15",
                        *hdr, "-o", str(f), a["url"]], stdin=subprocess.DEVNULL, check=False)
        got = f.stat().st_size if f.exists() else 0
        if got == a["size"]:
            return f
        if got > a["size"]:
            print(f"  {a['name']}: {got} > {a['size']} bytes, starting over", flush=True)
            f.unlink()
        print(f"  {a['name']}: {got} / {a['size']} bytes, retry {i + 1}", flush=True)
        time.sleep(30)
    raise SystemExit(f"giving up on {a['name']} (run again to resume)")


def check_members(f: Path, r: int) -> None:
    """The first member of an archive sits under SimScale/<round dir>/ (a wrong file is refused before extraction)."""
    with tarfile.open(f, "r:gz") as t:
        m = t.next()
    if m is None or not m.name.lstrip("./").startswith(f"SimScale/{round_dir(r)}"):
        raise SystemExit(f"{f}: first member {m.name if m else None!r} is not under SimScale/{round_dir(r)}/")


def extract(f: Path, root: Path, r: int) -> None:
    """Only SimScale/<round dir>/ of the archive, into root without the first component; tar refuses absolute and
    '..' names by itself."""
    check_members(f, r)
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(f), "-C", str(root), "--strip-components=1", "--no-same-owner",
                    f"SimScale/{round_dir(r)}"], check=True)


def scene_problems(hist: Path, scenes: List[str]) -> List[str]:
    """Scenes of a part whose directory or used camera images are missing."""
    bad = []
    for s in scenes:
        for cam in CAMERAS:
            d = hist / s / cam
            try:
                n = sum(1 for e in os.scandir(d) if e.name.endswith(".jpg"))
            except OSError:
                n = 0
            if n < N_HIST:
                bad.append(f"{s}/{cam}: {n} images")
                break
    return bad


def verify(a: dict, root: Path) -> List[str]:
    """Problems of an extracted archive (see the module docstring)."""
    rd = root / round_dir(a["round"])
    index = json.loads((rd / f"{prefix(a['round'])}_sensor_blobs_hist.json").read_text())
    if a["kind"] == "meta":
        want = sum(len(v) for v in index.values())
        with os.scandir(rd / "meta_datas") as it:
            got = sum(1 for e in it if e.name.endswith(".pkl"))
        return [] if got == want else [f"{rd}/meta_datas: {got} scene pickles, the part index lists {want} scenes"]
    if a["kind"] == "hist":
        if a["part"] not in index:
            return [f"{a['part']} is not in the part index"]
        return scene_problems(rd / "sensor_blobs_hist", index[a["part"]])
    return []


def finish(a: dict, f: Path, root: Path, dl: Path, keep: bool) -> None:
    """Unpack (index: copy) and verify one archive, then stamp it."""
    rd = root / round_dir(a["round"])
    if a["kind"] == "index":
        rd.mkdir(parents=True, exist_ok=True)
        json.loads(f.read_text())                      # a part index must parse
        shutil.copyfile(f, rd / a["name"])
    else:
        extract(f, root, a["round"])
    bad = verify(a, root)
    if bad:
        raise SystemExit(f"{a['name']}: {len(bad)} problems after extraction, e.g. {bad[:3]} (run again: it is "
                         f"downloaded and extracted again)")
    (dl / ".done" / a["name"]).write_text(time.strftime("%F %T") + "\n")
    if not keep:
        f.unlink()
    print(f"[{time.strftime('%T')}] done {a['name']}", flush=True)


def summary(todo: List[dict], free: int) -> str:
    left = sum(a["size"] for a in todo)
    rates = " / ".join(f"{left / (mb * 1e6) / 3600:.1f} h" for mb in (12, 50, 100))
    return (f"{len(todo)} archives left, {left / 1e12:.3f} TB; free {free / 1e12:.2f} TB; at 12 / 50 / 100 MB/s: {rates}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, nargs="+", default=list(ROUNDS))
    ap.add_argument("--root", default=SP.SSR_SIMSCALE_ROOT)
    ap.add_argument("--download-dir", default="", help="archives in flight and the .done stamps (default <root>/_download)")
    ap.add_argument("--plan-only", action="store_true", help="list what is left, its size, time and disk; download nothing")
    ap.add_argument("--keep-archives", action="store_true")
    a = ap.parse_args(argv)
    if not set(a.rounds) <= set(ROUNDS):
        raise SystemExit(f"--rounds: values of {ROUNDS}")
    root = Path(a.root)
    dl = Path(a.download_dir) if a.download_dir else root / "_download"
    token = os.environ.get("HF_TOKEN", "")
    arch = plan(sorted(set(a.rounds)), token)
    todo = [x for x in arch if not (dl / ".done" / x["name"]).exists()]
    probe = root if root.exists() else root.parent if root.parent.exists() else Path("/")
    free = shutil.disk_usage(probe).free
    print(f"SimScale rounds {sorted(set(a.rounds))} -> {root}: {len(arch)} archives; " + summary(todo, free), flush=True)
    left = sum(x["size"] for x in todo)
    if left * SPACE_MARGIN[0] + SPACE_MARGIN[1] > free:
        msg = f"not enough disk: need ~{(left * SPACE_MARGIN[0] + SPACE_MARGIN[1]) / 1e12:.2f} TB"
        if a.plan_only:
            print(msg, flush=True)
            return 0
        raise SystemExit(msg)
    if a.plan_only or not todo:
        return 0
    (dl / ".done").mkdir(parents=True, exist_ok=True)
    pending = None
    with ThreadPoolExecutor(1) as ex:                  # one extraction runs while the next archive downloads
        for i, x in enumerate(todo, 1):
            print(f"[{time.strftime('%T')}] ({i}/{len(todo)}) {x['name']} {x['size'] / 1e6:.0f} MB", flush=True)
            f = download(x, dl, token)
            if pending is not None:
                pending.result()
            pending = ex.submit(finish, x, f, root, dl, a.keep_archives)
        if pending is not None:
            pending.result()
    print(f"[{time.strftime('%T')}] all done -> {root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
