"""Every data, asset and interpreter root of this repo, in one place.

Each value is the environment variable of the same name, else the path on the reference server, so nothing has to be
set there.  A new server sets them in tools/release/paths.env (tools/release/paths.env.example says what each is).
An empty variable counts as unset.

stdlib only and no navsim import: data workers, launchers and shell wrappers all read this file.  Shell wrappers get
export lines with `python3 -I navsim/common/ssr_paths.py --shell` (-I: navsim/common/dataclasses.py must not shadow the
stdlib module when this runs as a script).

OPENSCENE_DATA_ROOT is also the official NAVSIM variable, so a shell can carry a value set up for another checkout.  A
value whose navsim_logs/trainval holds no log pickle (for example the nested trainval/trainval of a raw download) is
not used when the default root does hold them: training on it would stop at the first scene load.  The default is then
taken, with one line on stderr, and a NUPLAN_MAPS_ROOT under that ignored root goes with it.
"""
import os
import sys

_DATA_DEFAULT = "/home/external-user/ssd/yongjae_refiner"
_DATASET_DEFAULT = "/home/external-user/yongjae/SSR/data/dataset"

# name -> what it is (order = the order of --shell and of tools/release/paths.env.example)
NAMES = {
    "SSR_DATA_ROOT": "ck/, metric_cache, objects, sdf, e2e_side, splits, human",
    "SSR_TEACHER_CACHE": "teacher BEV caches (and the default distill_feature_root)",
    "OPENSCENE_DATA_ROOT": "dataset, with navsim_logs/{trainval,test}, sensor_blobs/{trainval,test}",
    "NUPLAN_MAPS_ROOT": "maps",
    "SSR_TEST_DATA_ROOT": "test logs and sensors, <root>/{navsim_logs,sensor_blobs}/test/test",
    "SSR_PLAN_DATA": "pdm_score_256 (the anchors come from the repo, resources/planning_vb/)",
    "SSR_NAVTEST_MC": "fork navtest metric cache, research only",
    "SSR_PY": "python interpreter",
    "SSR_TORCHRUN": "torchrun of that interpreter",
    "SSR_RUN_ROOT": "run folders",
    "NAVSIM_V11": "pristine official NAVSIM v1.1 tree",
    "NAVSIM_V11_CACHE": "navtest metric cache built by that tree",
    "NAVSIM_V22_POST": "NAVSIM v2.2 tree, EPDMS post-fix",
    "NAVSIM_V22_PRE": "NAVSIM v2.2 tree, EPDMS pre-fix",
    "NAVSIM_V22_CACHE": "navtest metric cache built by v2.2",
    "NAVSIM_V22_NAVHARD_CACHE": "navhard two-stage metric cache built by v2.2 (tools/release/navhard.sh only)",
    "SSR_RESEARCH_TREE": "older research checkout (report/, work_dirs/), read only by research checks; may be absent",
}


def _has_trainval_logs(root: str) -> bool:
    """True if <root>/navsim_logs/trainval directly holds a log pickle (first entries only, no full listing)."""
    try:
        with os.scandir(os.path.join(root, "navsim_logs", "trainval")) as it:
            return any(e.name.endswith(".pkl") for e in it)
    except OSError:
        return False


def resolve(env=None) -> dict:
    """{name: value} for a mapping of environment variables (default os.environ).  Touches the file system only when
    OPENSCENE_DATA_ROOT is set to something else than the default."""
    env = os.environ if env is None else env

    def get(name, default):
        return env.get(name) or default

    data = get("SSR_DATA_ROOT", _DATA_DEFAULT)
    v11 = data + "/v7/eval/official_v11"
    dataset = get("OPENSCENE_DATA_ROOT", _DATASET_DEFAULT)
    maps = env.get("NUPLAN_MAPS_ROOT")
    if dataset != _DATASET_DEFAULT and not _has_trainval_logs(dataset) and _has_trainval_logs(_DATASET_DEFAULT):
        print(f"[ssr_paths] OPENSCENE_DATA_ROOT={dataset} has no navsim_logs/trainval/*.pkl: using {_DATASET_DEFAULT}",
              file=sys.stderr)
        if maps and os.path.normpath(maps).startswith(os.path.normpath(dataset) + os.sep):
            maps = None
        dataset = _DATASET_DEFAULT
    return {
        "SSR_DATA_ROOT": data,
        "SSR_TEACHER_CACHE": get("SSR_TEACHER_CACHE", "/home/external-user/datasets/teacher_cache"),
        "OPENSCENE_DATA_ROOT": dataset,
        "NUPLAN_MAPS_ROOT": maps or dataset + "/maps",
        "SSR_TEST_DATA_ROOT": get("SSR_TEST_DATA_ROOT", "/workspace/navsim_workspace/dataset"),
        "SSR_PLAN_DATA": get("SSR_PLAN_DATA", "/home/external-user/kyungmin/SSR-v2/data/planning_vb"),
        "SSR_NAVTEST_MC": get("SSR_NAVTEST_MC", "/home/external-user/yongjae/SSR/data/exp/metric_cache"),
        "SSR_PY": get("SSR_PY", "/venv/ssr/bin/python"),
        "SSR_TORCHRUN": get("SSR_TORCHRUN", "/venv/ssr/bin/torchrun"),
        "SSR_RUN_ROOT": get("SSR_RUN_ROOT", data + "/v7/e2e"),
        "NAVSIM_V11": get("NAVSIM_V11", v11 + "/navsim_v1.1"),
        "NAVSIM_V11_CACHE": get("NAVSIM_V11_CACHE", v11 + "/exp/metric_cache_navtest_v11"),
        "NAVSIM_V22_POST": get("NAVSIM_V22_POST", "/workspace/navsim_workspace/navsim"),
        "NAVSIM_V22_PRE": get("NAVSIM_V22_PRE", "/workspace/yongjae/perception_eval/epdms/navsim_prefix"),
        "NAVSIM_V22_CACHE": get("NAVSIM_V22_CACHE", "/workspace/yongjae/perception_eval/epdms/metric_cache_v22_navtest"),
        "NAVSIM_V22_NAVHARD_CACHE": get("NAVSIM_V22_NAVHARD_CACHE",
                                        "/workspace/navsim_workspace/exp/metric_cache_navhard_two_stage_v22"),
        "SSR_RESEARCH_TREE": get("SSR_RESEARCH_TREE", "/home/external-user/yongjae/SSR"),
    }


_R = resolve()
SSR_DATA_ROOT = _R["SSR_DATA_ROOT"]
SSR_TEACHER_CACHE = _R["SSR_TEACHER_CACHE"]
OPENSCENE_DATA_ROOT = _R["OPENSCENE_DATA_ROOT"]
NUPLAN_MAPS_ROOT = _R["NUPLAN_MAPS_ROOT"]
SSR_TEST_DATA_ROOT = _R["SSR_TEST_DATA_ROOT"]
SSR_PLAN_DATA = _R["SSR_PLAN_DATA"]
SSR_NAVTEST_MC = _R["SSR_NAVTEST_MC"]
SSR_PY = _R["SSR_PY"]
SSR_TORCHRUN = _R["SSR_TORCHRUN"]
SSR_RUN_ROOT = _R["SSR_RUN_ROOT"]
NAVSIM_V11 = _R["NAVSIM_V11"]
NAVSIM_V11_CACHE = _R["NAVSIM_V11_CACHE"]
NAVSIM_V22_POST = _R["NAVSIM_V22_POST"]
NAVSIM_V22_PRE = _R["NAVSIM_V22_PRE"]
NAVSIM_V22_CACHE = _R["NAVSIM_V22_CACHE"]
NAVSIM_V22_NAVHARD_CACHE = _R["NAVSIM_V22_NAVHARD_CACHE"]
SSR_RESEARCH_TREE = _R["SSR_RESEARCH_TREE"]


if __name__ == "__main__":
    if sys.argv[1:] != ["--shell"]:
        sys.exit("usage: python3 -I ssr_paths.py --shell")
    import shlex

    for _k in NAMES:
        print(f"export {_k}={shlex.quote(_R[_k])}")
