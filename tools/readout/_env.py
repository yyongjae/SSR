"""Shared setup for tools/readout: repo on sys.path, NAVSIM env defaults."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Same defaults as scripts/training/train_para_ssr.sh (<repo>/data/dataset); override from the shell.
_DEFAULT_DATA = REPO / "data/dataset"
_DATA = os.environ.get("NAVSIM_DATA", str(_DEFAULT_DATA if _DEFAULT_DATA.is_dir() else "/data/navsim/dataset"))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", f"{_DATA}/maps")
os.environ.setdefault("OPENSCENE_DATA_ROOT", _DATA)
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(REPO))

DATA = Path(_DATA)
SCENE_FILTERS = REPO / "navsim/planning/script/config/common/scene_filter"


def scene_filter(name: str, **overrides):
    """Instantiate a navsim scene filter yaml (navtrain, navtest, ...)."""
    import yaml
    from navsim.common.dataclasses import SceneFilter

    cfg = yaml.safe_load((SCENE_FILTERS / f"{name}.yaml").read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg.update(overrides)
    return SceneFilter(**cfg)


def split_dirs(split: str):
    """(navsim_logs dir, sensor_blobs dir) for trainval | test."""
    return DATA / "navsim_logs" / split, DATA / "sensor_blobs" / split
