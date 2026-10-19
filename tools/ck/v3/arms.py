#!/usr/bin/env python
"""refiner-v3 Step-1 arm configurations (SPEC sections 5.4, 7; builder LOSS+EVAL, 2026-10-11 KST).

  PYTHONPATH=/workspace/yongjae/SSR-v3 /venv/ssr/bin/python tools/ck/v3/arms.py check
  PYTHONPATH=/workspace/yongjae/SSR-v3 /venv/ssr/bin/python tools/ck/v3/arms.py cli --arm a [--out-root D3/train]
  PYTHONPATH=/workspace/yongjae/SSR-v3 /venv/ssr/bin/python tools/ck/v3/arms.py show --arm d

Files (tools/ck/v3/arms/, the single source of the arm definitions; nothing here launches anything):
  recipe_ck2T10.json   the ck2T10 recipe flags (shared by a / b / d), world 2 on GPUs 0,1, 8 tokens / rank
  a_learned.json       (a) learned forecaster, 6 modes                  -> v3T10_a_learned       gates G1-G4
  b_ca.json            (b) analytic CA forecaster in the same branch     -> v3T10_b_ca            gate  G1
  c_baseline.json      (c) existing ck2T10 (temporal off, 4f29592; descriptive except G4)          gate  G4
  d_oracle_PRIV.json   (d) OPTIONAL GT-future oracle, PRIVILEGED          -> v3T10_d_oracle_PRIV   no gate
  pre_forecaster.json  stage 'pre' (review fix D3): forecaster pre-training for (a) -> v3fc_pre_learned6
                       (tools/ck/v3/pretrain_forecaster.py; (a)'s temporal.fc_init = <its out>/forecaster.pt, frozen)
Every training arm needs the user's approval (requires_user_approval) and is launched by the main session.

`cli(arm)` renders the train_ck2.py argument list: recipe cli + --out-root + --run-name + the temporal flags
(TemporalCfg field -> flag, SPEC 5.3) + --resume.  For (b) and (d) it passes `--t-modes 1` explicitly:
TemporalCfg.validate requires n_modes == 1 for ca / oracle_gt while the CLI default is 6.

`validate(arm)` checks: schema version; a / b / d temporal on, source bevfusion, forecaster a learned / b ca / d
oracle_gt, n_modes (a 6, b / d 1); a, b, d identical except forecaster and n_modes; privileged == (forecaster
oracle_gt or source gt) and the run name ends in '_PRIV' iff privileged; (c) temporal off with an existing run dir;
TemporalCfg.from_dict(..).validate('T') and to_dict round trip.  `recipe_check()` compares the recipe, parsed by
train_ck2.get_parser + resolve_args, with ck2T10/config.json on every key of expect_from_source_config.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

ARMS_DIR = Path(__file__).resolve().parent / "arms"
VERSION = "v3_arm_1"
FILES = {"a": "a_learned.json", "b": "b_ca.json", "c": "c_baseline.json", "d": "d_oracle_PRIV.json",
         "pre": "pre_forecaster.json"}
ARMS = ("a", "b", "c", "d")
FORECASTER = {"a": "learned", "b": "ca", "d": "oracle_gt"}
N_MODES = {"a": 6, "b": 1, "d": 1}
GATES = {"a": ["G1", "G2", "G3", "G4"], "b": ["G1"], "c": ["G4"], "d": []}
PER_ARM_KEYS = ("forecaster", "n_modes", "fc_init")     # the only temporal keys a / b / d may differ in
TRAIN_ARMS = ("a", "b", "d")
D3 = Path("/home/external-user/ssd/yongjae_refiner/ck/v3")
# TemporalCfg field -> train_ck2 flag (SPEC 5.3); bools as 0 / 1
FLAG = {"on": "--temporal", "source": "--t-source", "forecaster": "--t-forecaster", "n_modes": "--t-modes",
        "lambda_fc": "--t-lambda-fc", "fc_backprop": "--t-fc-backprop", "p_frame_drop": "--t-frame-drop",
        "p_obj_drop": "--t-obj-drop", "v_noise": "--t-vnoise", "pack_root": "--t-pack-root",
        "ca_scales": "--t-ca-scales", "fc_init": "--t-fc-init", "fc_train": "--t-fc-train"}
NOT_A_FLAG = ("version", "privileged")


def _read(p: Path) -> Dict:
    with open(p) as f:
        return json.load(f)


def load(arm: str) -> Dict:
    """arm id ('a' / 'b' / 'c' / 'd') or a path to an arm json."""
    p = ARMS_DIR / FILES[arm] if arm in FILES else Path(arm)
    d = _read(p)
    d["_path"] = str(p)
    return d


def recipe(arm_cfg: Optional[Dict] = None) -> Dict:
    name = (arm_cfg or {}).get("recipe", "recipe_ck2T10.json")
    d = _read(ARMS_DIR / name)
    d["_path"] = str(ARMS_DIR / name)
    return d


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def temporal_cli(t: Dict) -> List[str]:
    out: List[str] = []
    if not t.get("on"):
        return ["--temporal", "0"]
    for k, flag in FLAG.items():
        if k in t:
            if k == "fc_init" and not t[k]:
                continue                                    # '' = no pre-trained forecaster (b, d)
            out += [flag, _fmt(t[k])]
    unknown = sorted(set(t) - set(FLAG) - set(NOT_A_FLAG))
    if unknown:
        raise ValueError(f"temporal keys without a CLI flag: {unknown}")
    return out


def pre_cli(out: Optional[str] = None) -> List[str]:
    """pretrain_forecaster.py argument list of stage 'pre'."""
    p = load("pre")
    return ["--out", str(out or p["out"])] + list(p["cli"])


def cli(arm: str, out_root: Optional[str] = None) -> List[str]:
    """train_ck2.py argument list of a training arm (a / b / d)."""
    if arm == "pre":
        return pre_cli()
    a = load(arm)
    if a["arm"] not in TRAIN_ARMS:
        raise ValueError(f"arm {a['arm']!r} is not trained in Step 1 (c is the existing ck2T10)")
    r = recipe(a)
    return (list(r["cli"]) + ["--out-root", str(out_root or (D3 / "train")), "--run-name", a["name"]]
            + temporal_cli(a["temporal"]) + list(r.get("trailing", [])))


def _tcfg_module():
    try:
        from navsim.agents.para_ssr.ck.temporal import config as TC
        return TC
    except ImportError:          # pragma: no cover - config.py is part of the model builder's files
        return None


def validate(arm: str) -> Dict:
    """Raise ValueError on any rule of the module docstring; returns the loaded arm dict."""
    a = load(arm)
    k = a.get("arm")
    if a.get("version") != VERSION:
        raise ValueError(f"{a['_path']}: version {a.get('version')!r} != {VERSION!r}")
    if k == "pre":
        return validate_pre(a)
    if k not in FILES:
        raise ValueError(f"{a['_path']}: arm {k!r}")
    if a.get("gates") != GATES[k]:
        raise ValueError(f"arm {k}: gates {a.get('gates')} != {GATES[k]}")
    t = a["temporal"]
    TC = _tcfg_module()
    if k == "c":
        if t.get("on"):
            raise ValueError("arm c (ck2T10) must be temporal off")
        if not (Path(a["run_dir"]) / "config.json").is_file():
            raise ValueError(f"arm c run dir {a['run_dir']} has no config.json")
        if a.get("privileged"):
            raise ValueError("arm c cannot be privileged")
        if TC is not None and TC.TemporalCfg.from_dict(t).on:
            raise ValueError("arm c: TemporalCfg.from_dict gives on")
        return a
    if not t.get("on") or t.get("source") != "bevfusion":
        raise ValueError(f"arm {k}: temporal must be on with source bevfusion, got {t}")
    if t.get("forecaster") != FORECASTER[k] or int(t.get("n_modes", -1)) != N_MODES[k]:
        raise ValueError(f"arm {k}: forecaster / n_modes {t.get('forecaster')} / {t.get('n_modes')} != "
                         f"{FORECASTER[k]} / {N_MODES[k]}")
    priv = bool(t.get("forecaster") == "oracle_gt" or t.get("source") == "gt")
    if bool(a.get("privileged")) != priv:
        raise ValueError(f"arm {k}: privileged flag {a.get('privileged')} != derived {priv}")
    if a["name"].endswith("_PRIV") != priv:
        raise ValueError(f"arm {k}: run name {a['name']!r} must end in '_PRIV' iff privileged")
    if not a.get("requires_user_approval"):
        raise ValueError(f"arm {k}: every Step-1 training run requires the user's approval")
    if TC is not None:
        cfg = TC.TemporalCfg.from_dict(t)
        cfg.validate("T")
        if cfg.privileged != priv:
            raise ValueError(f"arm {k}: TemporalCfg.privileged {cfg.privileged} != {priv}")
        rt = cfg.to_dict()
        rt.pop("privileged", None)
        if rt != {kk: v for kk, v in t.items() if kk != "privileged"}:
            raise ValueError(f"arm {k}: temporal dict is not a TemporalCfg round trip:\n{t}\n{rt}")
        dflt = TC.TemporalCfg().to_dict()
        for kk in ("lambda_fc", "fc_backprop", "p_frame_drop", "p_obj_drop", "v_noise", "pack_root", "ca_scales",
                   "version", "fc_train"):
            if t[kk] != dflt[kk]:
                raise ValueError(f"arm {k}: {kk} {t[kk]!r} differs from the spec default {dflt[kk]!r}")
        if k == "a":
            pre = load("pre")
            if t.get("fc_init") != str(Path(pre["out"]) / "forecaster.pt"):
                raise ValueError(f"arm a: fc_init {t.get('fc_init')!r} != stage pre output {pre['out']}/forecaster.pt")
            if a.get("requires_stage") != "pre":
                raise ValueError("arm a must require stage 'pre'")
            cfg.check_trainable()
        elif t.get("fc_init"):
            raise ValueError(f"arm {k}: fc_init is for the learned arm only")
    return a


def _pre_args(p: Dict) -> Dict:
    from tools.ck.v3 import pretrain_forecaster as PF
    return vars(PF.get_parser().parse_args(["--out", p["out"]] + list(p["cli"])))


def validate_pre(p: Dict) -> Dict:
    """stage 'pre': its augmentation / modes / seed / packs must match arm a's temporal cfg and the recipe seed."""
    if p.get("name") != Path(p["out"]).name or not p.get("requires_user_approval"):
        raise ValueError("stage pre: out dir name must equal its name; requires_user_approval must be true")
    if not str(Path(p["out"]).resolve()).startswith(str((D3 / "train").resolve()) + "/"):
        raise ValueError(f"stage pre: out {p['out']} must be under {D3}/train")
    a = _read(ARMS_DIR / FILES["a"])["temporal"]
    r = recipe()
    x = _pre_args(p)
    want = {"n_modes": a["n_modes"], "frame_drop": a["p_frame_drop"], "obj_drop": a["p_obj_drop"],
            "vnoise": a["v_noise"], "pack_root": a["pack_root"], "ca_scales": a["ca_scales"],
            "seed": int(r["cli"][r["cli"].index("--seed") + 1]), "train_split": "navtrain_train"}
    bad = {k: (x[k], v) for k, v in want.items() if x[k] != v}
    if bad:
        raise ValueError(f"stage pre differs from arm a / the recipe: {bad}")
    return p


def check_all() -> Dict:
    """validate every arm + a / b / d differ only in forecaster, n_modes (and the derived privileged flag)."""
    A = {k: validate(k) for k in FILES}
    base = {kk: v for kk, v in A["a"]["temporal"].items() if kk not in PER_ARM_KEYS}
    for k in ("b", "d"):
        other = {kk: v for kk, v in A[k]["temporal"].items() if kk not in PER_ARM_KEYS}
        if other != base:
            raise ValueError(f"arms a and {k} differ outside forecaster / n_modes: {base} vs {other}")
        if A[k]["recipe"] != A["a"]["recipe"]:
            raise ValueError(f"arm {k} uses another recipe")
    names = [A[k]["name"] for k in TRAIN_ARMS]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate run names {names}")
    return {k: {"name": v["name"], "gates": v.get("gates"), "privileged": bool(v.get("privileged")),
                "temporal": v.get("temporal")} for k, v in A.items()}


def recipe_check() -> Dict:
    """Parse the recipe with train_ck2's own parser + resolve_args and compare with ck2T10/config.json."""
    from tools.ck import train_ck2 as T2
    r = recipe()
    src = Path(r["source_config"])
    sha = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    if sha != r["source_config_sha16"]:
        raise ValueError(f"{src} sha16 {sha} != recipe {r['source_config_sha16']} (ck2T10 config changed?)")
    ref = _read(src)
    argv = list(r["cli"]) + ["--run-name", "recipe_check"] + list(r.get("trailing", []))
    a = T2.resolve_args(T2.get_parser().parse_args(argv))
    rows, bad = {}, []
    for k in r["expect_from_source_config"]:
        ours = getattr(a, k)
        theirs = ref.get(k)
        eq = (ours == theirs) or (isinstance(ours, float) and isinstance(theirs, (int, float))
                                  and abs(ours - theirs) < 1e-12)
        rows[k] = {"recipe": ours, "ck2T10": theirs, "eq": bool(eq)}
        if not eq:
            bad.append(k)
    if int(ref["world_size"]) != int(r["world_size"]) or int(ref["tokens_per_rank"]) != int(r["tokens_per_rank"]):
        bad.append("world_size/tokens_per_rank")
    from navsim.agents.para_ssr.ck import constants as Cn       # loss constants recorded in config.json, not CLI
    for k, ours in (("ck_keys", list(Cn.CK_KEYS)), ("sur_weights", dict(Cn.SUR_WEIGHTS)),
                    ("sur_margins", dict(Cn.SUR_MARGINS)), ("lon_st_slope", dict(Cn.LON_ST_SLOPE))):
        eq = ours == ref.get(k)
        rows[f"const:{k}"] = {"recipe": ours, "ck2T10": ref.get(k), "eq": bool(eq)}
        if not eq:
            bad.append(f"const:{k}")
    return {"source_sha16": sha, "n_keys": len(rows), "mismatch": bad, "rows": rows,
            "ep_target": getattr(a, "ep_target", None), "ck2T10_ep_target": ref.get("ep_target", "absent (pre-flag)")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("check", "cli", "show"))
    ap.add_argument("--arm", choices=tuple(FILES), default=None)
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        res = {"arms": check_all(), "recipe": recipe_check()}
        print(json.dumps(res, indent=1, default=str))
        if res["recipe"]["mismatch"]:
            raise SystemExit(f"recipe mismatch vs ck2T10: {res['recipe']['mismatch']}")
    elif a.cmd == "cli":
        if a.arm is None:
            raise SystemExit("--arm required")
        validate(a.arm)
        print(" ".join(cli(a.arm, a.out_root)))
    else:
        if a.arm is None:
            raise SystemExit("--arm required")
        print(json.dumps(validate(a.arm), indent=1))


if __name__ == "__main__":
    main()
