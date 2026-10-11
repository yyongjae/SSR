"""refiner-v3 run provenance (review fix D5; protocol finding 6).  CPU, read-only.

Every Step-1 run (forecaster pre-training, train_ck2 temporal runs, inference) records ``run_provenance(...)`` in its
config / meta json, and decide_step1.py refuses arms whose provenance disagrees:

  code.train_sha256   sha256 over the sorted (path, sha256) of the TRAINING code set TRAIN_CODE (every .py of
                      navsim/agents/para_ssr/{ck,refiner}, tools/ck/*.py, tools/ck/data/*.py, and the v3 files the
                      trainers import; tracked or not).  Arm-independent and deterministic: arms a / b / d must agree.
  code.worktree       git HEAD, dirty flag, sha256 of `git diff HEAD -- navsim tools` and of the untracked files
                      under navsim/ tools/ (the "git diff hash of the worktree"; reported, a mismatch is a warning
                      only, because editing an evaluation script between arms does not change training).
  data.packs          per split: sha256 of the object pack's build.json (it carries the sha256 of every array),
                      K_OBJ, status, n, dir.
  data.ca_scales      path + sha256 of the CA Laplace scales json (+ its version / source / k_obj / aug record).
  data.fc_init        learned arm: path + sha256 of the pre-trained forecaster checkpoint and the sha256 of the
                      train pack it was pre-trained on (must equal data.packs[navtrain_train]).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO = Path(__file__).resolve().parents[3]
TRAIN_CODE = ("navsim/agents/para_ssr/ck/**/*.py", "navsim/agents/para_ssr/refiner/**/*.py", "tools/ck/*.py",
              "tools/ck/data/*.py", "tools/ck/v3/__init__.py", "tools/ck/v3/provenance.py",
              "tools/ck/v3/pretrain_forecaster.py", "tools/ck/v3/arms.py", "tools/ck/v3/arms/*.json")
WORKTREE_SCOPE = ("navsim", "tools")
VERSION = "v3_provenance_1"


def sha256_file(p, n: Optional[int] = None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:n] if n else h.hexdigest()


def _sha_lines(rows: Iterable) -> str:
    return hashlib.sha256("\n".join(f"{a} {b}" for a, b in rows).encode()).hexdigest()


def train_code(repo: Path = REPO) -> Dict:
    files = set()
    for g in TRAIN_CODE:
        files.update(p for p in repo.glob(g) if p.is_file() and "__pycache__" not in p.parts)
    rows = sorted((str(p.relative_to(repo)), sha256_file(p)) for p in files)
    return {"sha256": _sha_lines(rows), "n_files": len(rows), "files": dict(rows)}


def _git(repo: Path, *args) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=60, check=True).stdout


def worktree(repo: Path = REPO) -> Dict:
    try:
        head = _git(repo, "rev-parse", "HEAD").decode().strip()
        diff = _git(repo, "diff", "HEAD", "--binary", "--", *WORKTREE_SCOPE)
        unt = [x for x in _git(repo, "ls-files", "--others", "--exclude-standard", "--", *WORKTREE_SCOPE)
               .decode().splitlines() if x and "__pycache__" not in x]
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    urows = sorted((u, sha256_file(repo / u)) for u in unt if (repo / u).is_file())
    dsha = hashlib.sha256(diff).hexdigest()
    usha = _sha_lines(urows)
    return {"git_head": head, "dirty": bool(diff or urows), "diff_sha256": dsha, "n_untracked": len(urows),
            "untracked_sha256": usha, "sha256": hashlib.sha256(f"{head} {dsha} {usha}".encode()).hexdigest()}


def pack_record(pack_dir) -> Dict:
    P = Path(pack_dir)
    bj = P / "build.json"
    if not bj.is_file():
        return {"dir": str(P), "missing": True}
    b = json.loads(bj.read_text())
    return {"dir": str(P), "build_sha256": sha256_file(bj), "k_obj": b.get("K_OBJ"), "status": b.get("status"),
            "n": b.get("n"), "source": b.get("source"), "split": b.get("split"), "targets": b.get("targets"),
            "obj_sha256": (b.get("files") or {}).get("obj.npy", {}).get("sha256")}


def ca_record(path) -> Dict:
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "missing": True}
    d = json.loads(p.read_text())
    return {"path": str(p), "sha256": sha256_file(p), "version": d.get("version"), "source": d.get("source"),
            "k_obj": d.get("k_obj"), "pack_sha256": d.get("pack_sha256"), "aug": d.get("aug")}


def fc_init_record(path) -> Dict:
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "missing": True}
    rec = {"path": str(p), "sha256": sha256_file(p)}
    cfgp = p.parent / "config.json"
    if cfgp.is_file():
        c = json.loads(cfgp.read_text())
        prov = c.get("provenance") or {}
        rec.update(pretrain_dir=str(p.parent), pretrain_config_sha256=sha256_file(cfgp),
                   pretrain_train_pack_sha256=(((prov.get("data") or {}).get("packs") or {}).get("navtrain_train")
                                               or {}).get("build_sha256"),
                   pretrain_code_sha256=((prov.get("code") or {}).get("train") or {}).get("sha256"),
                   n_modes=c.get("n_modes"), aug=c.get("aug"), epochs=c.get("epochs"), seed=c.get("seed"))
    return rec


def run_provenance(tcfg=None, splits: Iterable[str] = ("navtrain_train", "navtrain_val"), with_files: bool = False,
                   pack_root: Optional[str] = None, source: str = "bevfusion") -> Dict:
    """Provenance dict of a run.  tcfg = TemporalCfg (on) or None (pack_root / source then given explicitly)."""
    from navsim.agents.para_ssr.ck.temporal import config as TCF
    root = Path(pack_root if pack_root is not None else (tcfg.pack_root if tcfg is not None else TCF.PACK_ROOT))
    src = tcfg.source if tcfg is not None else source
    tc = train_code()
    out = {"version": VERSION, "k_obj": TCF.K_OBJ,
           "code": {"train": tc if with_files else {k: v for k, v in tc.items() if k != "files"},
                    "worktree": worktree()},
           "data": {"packs": {s: pack_record(root / src / s) for s in splits}}}
    if tcfg is not None and not tcfg.on:              # temporal off: no object data is read
        out["data"] = {}
        return out
    if tcfg is not None:
        out["data"]["ca_scales"] = ca_record(tcfg.ca_scales)
        if getattr(tcfg, "fc_init", ""):
            out["data"]["fc_init"] = fc_init_record(tcfg.fc_init)
    return out


def compare(provs: Dict[str, Dict], strict_keys=None) -> Dict[str, List[str]]:
    """provs = {arm: provenance} -> {'errors': [...], 'warnings': [...]} for arms that must share code and data."""
    errs, warns = [], []
    keys = list(provs)
    if len(keys) < 2:
        return {"errors": errs, "warnings": warns}

    def g(p, *path):
        for k in path:
            p = (p or {}).get(k) if isinstance(p, dict) else None
        return p

    checks = [("code.train.sha256", ("code", "train", "sha256"), True),
              ("k_obj", ("k_obj",), True),
              ("data.packs.navtrain_train.build_sha256", ("data", "packs", "navtrain_train", "build_sha256"), True),
              ("data.packs.navtrain_val.build_sha256", ("data", "packs", "navtrain_val", "build_sha256"), True),
              ("data.ca_scales.sha256", ("data", "ca_scales", "sha256"), True),
              ("code.worktree.sha256", ("code", "worktree", "sha256"), False)]
    for name, path, strict in checks:
        vals = {k: g(provs[k], *path) for k in keys}
        if any(v is None for v in vals.values()):
            (errs if strict else warns).append(f"{name} missing for {[k for k, v in vals.items() if v is None]}")
            continue
        if len(set(vals.values())) > 1:
            (errs if strict else warns).append(f"{name} differs: {vals}")
    fis = {k: g(provs[k], "data", "fc_init", "sha256") for k in keys}
    if all(v is not None for v in fis.values()) and len(set(fis.values())) > 1:
        errs.append(f"data.fc_init.sha256 differs: {fis}")
    for k in keys:
        fi = g(provs[k], "data", "fc_init")
        if fi:
            tp = g(provs[k], "data", "packs", "navtrain_train", "build_sha256")
            if fi.get("missing"):
                errs.append(f"arm {k}: fc_init {fi.get('path')} missing")
            elif fi.get("pretrain_train_pack_sha256") != tp:
                errs.append(f"arm {k}: forecaster pre-trained on train pack {fi.get('pretrain_train_pack_sha256')} "
                            f"!= the critic's {tp}")
            if fi.get("pretrain_code_sha256") and fi.get("pretrain_code_sha256") != g(provs[k], "code", "train",
                                                                                      "sha256"):
                warns.append(f"arm {k}: forecaster pre-trained with another training-code sha (pre-training ran "
                             f"before a code change)")
    return {"errors": errs, "warnings": warns}


def check_self(prov: Dict) -> List[str]:
    """Errors of a single run's provenance: missing packs, or a pre-trained forecaster that is missing or was
    pre-trained on another train pack than this run trains on."""
    errs = []
    data = prov.get("data") or {}
    for s, r in (data.get("packs") or {}).items():
        if r.get("missing") or r.get("status") != "complete":
            errs.append(f"pack {s}: {r.get('dir')} missing or incomplete")
    fi = data.get("fc_init")
    if fi:
        tp = ((data.get("packs") or {}).get("navtrain_train") or {}).get("build_sha256")
        if fi.get("missing"):
            errs.append(f"fc_init {fi.get('path')} missing")
        elif fi.get("pretrain_train_pack_sha256") is not None and fi.get("pretrain_train_pack_sha256") != tp:
            errs.append(f"fc_init was pre-trained on train pack {fi.get('pretrain_train_pack_sha256')}, this run uses {tp}")
    return errs


__all__ = ["check_self", "TRAIN_CODE", "train_code", "worktree", "pack_record", "ca_record", "fc_init_record", "run_provenance",
           "compare", "sha256_file"]
