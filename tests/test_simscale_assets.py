"""trainval_simscale offline assets: token table, SimScale metric cache (4 s GT, TTC look-ahead at constant velocity), the row-indexed
builders, the combined score file, verify_assets --data trainval_simscale, the SimScale download, the orchestrator.

CPU only.  Fake trees under tmp_path, the scripts with SETUP_DRY=1.  Tests marked "real" read this server's data (the
navtrain metric cache built before this change, the dataset, maps and the round-4 SimScale sample) and skip elsewhere;
"git" tests compare with the base commit of this branch and skip without its history.
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD python -m pytest -q -p no:cacheprovider tests/test_simscale_assets.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import lzma
import os
import pickle
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SETUP = REPO / "tools/release/setup"
for _p in (REPO, REPO / "tools/refiner", REPO / "tools/release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


TB = _load("tvsim_table_t", SETUP / "build_trainval_simscale_table.py")
SC = _load("tvsim_score_t", SETUP / "assemble_trainval_simscale_score.py")
DL = _load("tvsim_download_t", SETUP / "simscale_download.py")
RP = _load("tvsim_repack_t", SETUP / "repack_dataset.py")
RT = _load("tvsim_raw256_t", SETUP / "make_raw256_traj.py")
MP = _load("tvsim_parts_t", SETUP / "make_parts.py")
BMC = _load("tvsim_bmc_t", REPO / "tools/refiner/build_metric_cache.py")

from navsim.agents.para_ssr.ck.tokens import token_sha16  # noqa: E402
from navsim.common import ssr_paths as SP  # noqa: E402

BASE = "b902937"                     # origin/v2_final, the commit this branch starts from
DATASET = Path("/home/external-user/yongjae/SSR/data/dataset")
LOGS = DATASET / "navsim_logs/trainval"
MAPS = DATASET / "maps"
REF_MC = Path("/home/external-user/ssd/yongjae_refiner/metric_cache")          # navtrain_train, built before this change
SIM_SAMPLE = Path("/home/external-user/ssd/yongjae_refiner/simscale_sample")   # round 4: all meta, images of part 0
REAL = (LOGS / "2021.05.12.19.36.12_veh-35_00005_00204.pkl").is_file() and (MAPS / "nuplan-maps-v1.0.json").is_file()
real_mc = pytest.mark.skipif(not (REAL and REF_MC.is_dir()), reason="needs this server's dataset, maps and metric cache")
real_sim = pytest.mark.skipif(not (REAL and (SIM_SAMPLE / "synthetic_reaction_pdm_v1.0-4/meta_datas").is_dir()),
                              reason="needs this server's maps and the SimScale round-4 sample")


def _git_show(rev: str, rel: str) -> str:
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{rel}"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"no git history for {rev}:{rel}")
    return r.stdout


def _maps_env():
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(MAPS))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")


# ----------------------------------------------------------------------------------------------- fake SimScale tree
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


FAR_CAR = [30.0, 10.0, 0.0, 4.0, 2.0, 1.5, 0.3]           # gt_boxes row (ego frame): x y z length width height heading
ON_EGO = [2.0, 0.5, 0.0, 4.0, 2.0, 1.5, 0.1]               # overlaps the ego box (rear axle at the origin)


def sim_scene(root: Path, r: int, log: str, scene_hex: str, cur_hex: str, images=True, route=True, n=12, city="c1",
              ego_overlap=False):
    """One SimScale scene pickle (+ the history images of the used cameras); -> (stem, frame-3 token)."""
    rrr = f"{r:03d}"
    scene, stem = f"{scene_hex}-{rrr}", f"{log}-{scene_hex}-{rrr}"
    frames = []
    for i in range(n):
        tok = f"{(int(cur_hex, 16) + i - 3) % 16 ** 16:016x}-{rrr}"
        cams = {c: {"data_path": f"{scene}/{c}/{tok}.jpg"} for c in CAMS}
        frames.append({"token": tok, "frame_idx": i, "timestamp": 10_000_000 + i * 500_000, "log_name": stem,
                       "map_location": city, "ego2global_translation": [float(i), 0.5 * i, 0.0],
                       "ego2global_rotation": [1.0, 0.0, 0.0, 0.0], "ego_dynamic_state": np.array([1.0 + i, 0.25, 0.5, -0.5]),
                       "driving_command": [0, 1, 0, 0], "roadblock_ids": ["r"] if route else [], "cams": cams,
                       "lidar_path": None,
                       "anns": {"gt_boxes": np.array([FAR_CAR] + ([ON_EGO] if ego_overlap and i == 3 else [])),
                                "gt_names": np.array(["vehicle"] * (2 if ego_overlap and i == 3 else 1))}})
    d = root / f"synthetic_reaction_pdm_v1.0-{r}"
    (d / "meta_datas").mkdir(parents=True, exist_ok=True)
    with open(d / "meta_datas" / f"{stem}.pkl", "wb") as f:
        pickle.dump(frames, f)
    if images:
        for i in range(4):
            for c in ("CAM_F0", "CAM_L0", "CAM_R0"):
                p = d / "sensor_blobs_hist" / frames[i]["cams"][c]["data_path"]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"jpg")
    return stem, frames[3]["token"]


def real_table(p: Path, toks, logs, city=True):
    df = pd.DataFrame({"token": toks, "log": logs, "frame_idx": range(len(toks))})
    if city:
        df["city"] = "c0"
    df.to_parquet(p, index=False)
    return p


def sorted_sha(toks):
    return hashlib.sha256("\n".join(sorted(toks)).encode()).hexdigest()[:16]


@pytest.fixture
def sim_tree(tmp_path):
    """Round 4 with 4 good scenes and one of each drop reason (10 pickles); round 0 with 2 good scenes; real tables."""
    root = tmp_path / "simscale"
    good = [sim_scene(root, 4, "logA_veh-1_00001", f"{h:016x}", f"{h + 100:016x}") for h in (5, 3, 9, 1)]
    good0 = [sim_scene(root, 0, "logB_veh-2_00002", f"{h:016x}", f"{h + 200:016x}") for h in (7, 2)]
    sim_scene(root, 4, "logC", f"{21:016x}", f"{121:016x}", images=False)               # camera
    sim_scene(root, 4, "logC", f"{22:016x}", f"{122:016x}", route=False)                # route
    sim_scene(root, 4, "logC", f"{23:016x}", f"{123:016x}", n=14)                       # frames
    sim_scene(root, 4, "logC", f"{26:016x}", f"{126:016x}", ego_overlap=True)           # ego_overlap
    stem, _ = sim_scene(root, 4, "logC", f"{24:016x}", f"{124:016x}")
    frames = pickle.load(open(root / "synthetic_reaction_pdm_v1.0-4/meta_datas" / f"{stem}.pkl", "rb"))
    frames[3]["token"] = f"{124:016x}-003"                                             # token of another round
    pickle.dump(frames, open(root / "synthetic_reaction_pdm_v1.0-4/meta_datas" / f"{stem}.pkl", "wb"))
    with open(root / "synthetic_reaction_pdm_v1.0-4/meta_datas" / f"logC-{25:016x}-004.pkl", "wb") as f:
        pickle.dump({"a": __import__("collections").OrderedDict()}, f)                  # unpickle: a global not allowed
    train = real_table(tmp_path / "train.parquet", [f"a{i:015x}" for i in range(5)], ["L1"] * 3 + ["L2"] * 2)
    val = real_table(tmp_path / "val.parquet", [f"b{i:015x}" for i in range(3)], ["L3"] * 3)
    return dict(root=root, train=train, val=val, good=sorted(t for _, t in good), good0=sorted(t for _, t in good0))


# ----------------------------------------------------------------------------------------------- paths and splits
def test_simscale_root_follows_the_data_root_and_its_variable():
    assert SP.resolve({"SSR_DATA_ROOT": "/x/data"})["SSR_SIMSCALE_ROOT"] == "/x/simscale"
    assert SP.resolve({"SSR_DATA_ROOT": "/x/data", "SSR_SIMSCALE_ROOT": "/s"})["SSR_SIMSCALE_ROOT"] == "/s"
    assert "SSR_SIMSCALE_ROOT" in SP.NAMES and "#SSR_SIMSCALE_ROOT=" in (REPO / "tools/release/paths.env.example").read_text()


def test_split_entries_name_the_contract_paths():
    from navsim.agents.para_ssr.ck import constants as C
    from navsim.common import simscale as SS
    t, s = C.SPLITS["trainval_simscale"], C.SPLITS["simscale"]
    assert t["tokens"] == str(C.DATA_ROOT / "splits/trainval_simscale.parquet")
    assert s["tokens"] == str(C.DATA_ROOT / "splits/simscale.parquet")
    assert t["metric_cache"] == s["metric_cache"] == C.SPLITS["navtrain_train"]["metric_cache"] == str(C.METRIC_CACHE["navtrain"])
    assert t["simscale_root"] == s["simscale_root"] == SP.SSR_SIMSCALE_ROOT
    assert tuple(t["rounds"]) == tuple(s["rounds"]) == SS.ROUNDS == TB.ROUNDS
    assert t["parts"] == ("navtrain_train", "navtrain_val", "simscale")
    assert t["label_source"] == str(C.CK_DATA / "raw256/pdm_score_256_officialEP_trainval_simscale")
    assert t["navsim_logs"] == C.SPLITS["navtrain_train"]["navsim_logs"]


def test_table_tool_repeats_the_simscale_module_names():
    from navsim.common import simscale as SS
    assert TB.TABLE_COLUMNS == SS.TABLE_COLUMNS and TB.SOURCE_REAL == SS.SOURCE_REAL and TB.SOURCE_SIM == SS.SOURCE_SIM
    assert (TB.NUM_FRAMES, TB.CURRENT, TB.META_DIR, TB.HIST_DIR) == (SS.NUM_FRAMES, SS.CURRENT, SS.META_DIR, SS.HIST_DIR)
    assert TB.scene_pkl(3, "x-y-003") == SS.scene_pkl(3, "x-y-003") and TB.round_dir(2) == SS.round_dir(2)
    assert TB.SceneUnpickler.ALLOWED == SS.SceneUnpickler.ALLOWED            # the loader reads the pickles as the table


# ----------------------------------------------------------------------------------------------- token table
def _build(st, caps=None, rounds=(0, 4), splits=None, drop=1.0):
    return TB.build(st["train"], st["val"], st["root"], list(rounds), caps or {}, 2, drop, splits or _pins(st))


def _pins(st):
    tr, va = pd.read_parquet(st["train"]), pd.read_parquet(st["val"])
    return {"navtrain_train": {"n": len(tr), "sha16_sorted_tokens": sorted_sha(tr.token)},
            "navtrain_val": {"n": len(va), "sha16_sorted_tokens": sorted_sha(va.token)}}


def test_table_order_columns_and_filter(sim_tree, monkeypatch):
    monkeypatch.setattr(TB, "PUBLISHED_SCENES", {0: 2, 4: 10})
    t, meta = _build(sim_tree)
    assert tuple(t.columns) == TB.TABLE_COLUMNS and (t.row.to_numpy() == np.arange(len(t))).all()
    assert t.source.tolist() == ["navtrain_train"] * 5 + ["navtrain_val"] * 3 + ["simscale"] * 6
    assert t.token.tolist()[:8] == pd.read_parquet(sim_tree["train"]).token.tolist() + pd.read_parquet(sim_tree["val"]).token.tolist()
    assert t.token.tolist()[8:] == sim_tree["good0"] + sim_tree["good"]                 # r0 then r4, each sorted
    assert t["round"].tolist() == [-1] * 8 + [0, 0, 4, 4, 4, 4] and t["round"].dtype == np.int8
    sim = t[t.source == "simscale"]
    assert all(p == TB.scene_pkl(r, lg) for p, r, lg in zip(sim.pkl, sim["round"], sim.log)) and (t.pkl[:8] == "").all()
    assert all(len(x) == 20 for x in sim.token) and (sim.city == "c1").all()
    assert meta["rounds_scanned"][4]["dropped"] == {"unpickle": 1, "frames": 1, "token": 1, "log_name": 0, "route": 1,
                                                    "ego_overlap": 1, "camera": 1}
    assert meta["mode"] == "full" and meta["token_sha16"] == token_sha16(t.token)


def test_table_mini_caps_and_no_pins(sim_tree):
    t, meta = _build(sim_tree, caps=TB.parse_caps(["navtrain_train=2", "navtrain_val=1", "simscale=3"]),
                     splits={"navtrain_train": {"n": 1, "sha16_sorted_tokens": "x"}, "navtrain_val": {"n": 1, "sha16_sorted_tokens": "x"}})
    assert t.source.value_counts().to_dict() == {"navtrain_train": 2, "navtrain_val": 1, "simscale": 3}
    assert t.token[t.source == "simscale"].tolist() == (sim_tree["good0"] + sim_tree["good"])[:3]
    assert meta["mode"] == "mini" and meta["pinned_real"].startswith("not checked")
    assert TB.parse_caps(["4"]) == {"navtrain_train": 4, "navtrain_val": 4, "simscale": 4}
    with pytest.raises(SystemExit, match="unknown source"):
        TB.parse_caps(["navtest=3"])


def test_table_full_mode_refuses_unpinned_real_parts_incomplete_rounds_and_drops(sim_tree, monkeypatch):
    monkeypatch.setattr(TB, "PUBLISHED_SCENES", {0: 2, 4: 10})
    bad = _pins(sim_tree)
    bad["navtrain_val"] = {"n": 3, "sha16_sorted_tokens": "0" * 16}
    with pytest.raises(SystemExit, match="navtrain_val: sorted token sha16"):
        _build(sim_tree, splits=bad)
    monkeypatch.setattr(TB, "PUBLISHED_SCENES", {0: 2, 4: 11})
    with pytest.raises(SystemExit, match="published 11"):
        _build(sim_tree)
    monkeypatch.setattr(TB, "PUBLISHED_SCENES", {0: 2, 4: 10})
    with pytest.raises(SystemExit, match="5 of 10 scenes lost to the filter"):
        _build(sim_tree, drop=0.1)                                     # round 4 drops 6 of 10, ego_overlap not counted
    t, _ = _build(sim_tree, drop=0.55)                                 # 5 of 10 lost: under the limit (6 with ego_overlap)
    assert (t.source == "simscale").sum() == 6
    with pytest.raises(SystemExit, match="round 3"):
        _build(sim_tree, rounds=(3,))


def test_table_refuses_a_test_log_leak_and_duplicates(sim_tree, monkeypatch):
    monkeypatch.setattr(TB, "PUBLISHED_SCENES", {0: 2, 4: 11})
    test_log = sorted(TB.test_logs())[0]
    sim_scene(sim_tree["root"], 4, test_log, f"{31:016x}", f"{131:016x}")
    with pytest.raises(SystemExit, match="test / navtest logs"):
        _build(sim_tree)
    t = TB.assemble([("navtrain_train", pd.DataFrame({"token": ["a", "a"], "log": "L", "city": "c"}))])
    assert any("duplicate" in x for x in TB.table_problems(t, set()))
    assert TB.source_log("2021.05.12.19.36.12_veh-35_00005_00204-0a678d2136b35b56-004") == \
        "2021.05.12.19.36.12_veh-35_00005_00204"


def test_ego_overlaps_counts_the_boxes_on_the_ego_box():
    assert TB.ego_overlaps(np.array([FAR_CAR])) == 0 and TB.ego_overlaps(np.zeros((0, 7))) == 0
    assert TB.ego_overlaps(np.array([FAR_CAR, ON_EGO, ON_EGO])) == 2
    front = TB.EGO_REAR_AXLE_TO_CENTER + TB.EGO_LENGTH / 2                    # 4.049 m ahead of the rear axle
    assert TB.ego_overlaps(np.array([[front + 1.0, 0, 0, 1.9, 1.0, 1, 0.0]])) == 0   # back end at front + 0.05
    assert TB.ego_overlaps(np.array([[front + 1.0, 0, 0, 2.2, 1.0, 1, 0.0]])) == 1   # back end at front - 0.1
    assert TB.ego_overlaps(np.array([[front + 1.0, 0, 0, 2.2, 1.0, 1, np.pi / 2]])) == 0  # turned: 0.5 m deep only
    side = TB.EGO_WIDTH / 2 + 0.6                                             # 0.6 m beside the ego box
    assert TB.ego_overlaps(np.array([[1.0, side, 0, 4.0, 1.0, 1, 0.0]])) == 0
    assert TB.ego_overlaps(np.array([[1.0, side, 0, 4.0, 1.0, 1, np.pi / 2]])) == 1      # turned: 2 m across


def test_table_unpickler_resolves_only_numpy(tmp_path):
    p = tmp_path / "x.pkl"
    pickle.dump([{"a": np.arange(3), "b": np.float64(2.0)}], open(p, "wb"))
    assert TB.load_scene_pickle(p)[0]["a"].tolist() == [0, 1, 2]
    pickle.dump(types.SimpleNamespace(x=1), open(p, "wb"))
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        TB.load_scene_pickle(p)


def test_table_city_from_the_tracked_copy_and_cli_refuses_overwrite(sim_tree, tmp_path, monkeypatch):
    tr = pd.read_parquet(REPO / "resources/km_corridor/packed/navtrain_val/tokens.parquet").iloc[:4]
    p = tmp_path / "val_nocity.parquet"
    tr[["token", "log"]].to_parquet(p, index=False)
    assert TB.read_real(p, "navtrain_val").city.tolist() == tr.city.tolist()
    out = tmp_path / "out/t.parquet"
    args = ["--train", str(sim_tree["train"]), "--val", str(sim_tree["val"]), "--sim-root", str(sim_tree["root"]),
            "--rounds", "4", "--max-per-source", "simscale=2", "--out", str(out), "--sim-out", str(tmp_path / "out/s.parquet"),
            "--workers", "1"]
    assert TB.main(args) == 0
    s = pd.read_parquet(tmp_path / "out/s.parquet")
    assert len(s) == 2 and s.row.tolist() == [0, 1] and (s.source == "simscale").all()
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["n"] == 10 and meta["sim_token_sha16"] == token_sha16(s.token)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        TB.main(args)


@real_sim
def test_table_rows_load_through_the_lazy_scene_loader(tmp_path):
    _maps_env()
    import dataclasses as dc
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.simscale import SimScaleSceneLoader
    real = tmp_path / "r.parquet"
    real_table(real, ["0002267a294e52a7"], ["2021.05.12.19.36.12_veh-35_00005_00204"])
    t, meta = TB.build(real, real, SIM_SAMPLE, [4], {"navtrain_val": 0, "simscale": 3}, 8, 1.0, {})
    assert meta["rounds_scanned"][4]["kept"] == 1000            # the scenes whose images are on this server (part 0)
    sc = dc.replace(SensorConfig.build_all_sensors(include=[2, 3]), lidar_pc=False)
    ld = SimScaleSceneLoader(SIM_SAMPLE, t, sc)
    for tok in ld.tokens:
        s = ld.get_scene_from_token(tok)
        assert s.scene_metadata.initial_token == tok and len(s.frames) == 12
        assert s.frames[3].cameras.cam_f0.image.shape == (1080, 1920, 3)


# ----------------------------------------------------------------------------------------------- observation, scorer
def _sampling(n):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    return TrajectorySampling(num_poses=n, interval_length=0.1)


def test_observation_horizon_flag_and_len():
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
    o = PDMObservation(_sampling(50), _sampling(40), 100, observation_sample_res=1)
    assert o._observation_samples == 50 and len(o) == 51                    # metric cache, 5 s: as before
    o = PDMObservation(_sampling(40), _sampling(40), 100, observation_sample_res=1, extend_observation_for_ttc=False)
    assert o._observation_samples == 40 and len(o) == 41                    # SimScale, 4 s
    o = PDMObservation(_sampling(40), _sampling(40), 100)                    # PDM-Closed's own: still + 1 s
    assert o._observation_samples == 50 and len(o) == 52
    o = PDMObservation(_sampling(50), _sampling(40), 100, observation_sample_res=1, extend_observation_for_ttc=False)
    assert o._observation_samples == 50                                     # same as extended for 5 s
    assert not hasattr(o, "_extend_observation_for_ttc")                     # pickles keep their old attribute set


def test_extend_constant_velocity_moves_each_object_at_its_last_velocity():
    from shapely.geometry import box
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap

    def obs(n_maps, samples):
        o = PDMObservation(_sampling(samples), _sampling(40), 100, observation_sample_res=1,
                           extend_observation_for_ttc=samples >= 50)
        maps = []
        for i in range(n_maps):        # car: 0.5 m per step along x; cone: still; late: only in the last map
            toks, geoms = ["car", "cone"], [box(i * 0.5, 0, i * 0.5 + 4, 2), box(10, 10, 11, 11)]
            if i == n_maps - 1:
                toks, geoms = toks + ["late"], geoms + [box(-5, -5, -4, -4)]
            maps.append(PDMOccupancyMap(toks, np.array(geoms, dtype=object)))
        o._occupancy_maps, o._initialized = maps, True
        return o

    o = obs(41, 40)
    assert len(o) == 41 and o.extrapolated_from is None
    assert o.extend_constant_velocity(50) == 10 and len(o) == 51 and o.extrapolated_from == 41
    for t in range(41, 51):
        assert o[t]["car"].equals(box(t * 0.5, 0, t * 0.5 + 4, 2))
        assert o[t]["cone"].equals(box(10, 10, 11, 11)) and o[t]["late"].equals(box(-5, -5, -4, -4))
    assert o[40]["car"].equals(box(20, 0, 24, 2)) and o.extend_constant_velocity(50) == 0
    five = obs(51, 50)                                               # a 5 s cache: nothing added, nothing recorded
    assert five.extend_constant_velocity(50) == 0 and len(five) == 51 and not hasattr(five, "_extrapolated_from")
    assert pickle.loads(pickle.dumps(o)).extrapolated_from == 41


@real_mc
def test_constant_velocity_recovers_the_5_s_ttc_of_real_caches():
    """Real 5 s caches cut to their 41 GT maps and extended at constant velocity score TTC (256 anchors) almost as with
    the true 51 maps, with no lean either way; the old rule (look-ahead skipped past 4 s) is lenient only."""
    import dataclasses
    _maps_env()
    import score_trajectories as ST
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
    anchors = np.load(REPO / "resources/planning_vb/trajectory_anchors_256.npy").astype(np.float32)
    full, cv, skip = [], [], []
    for p in _ref_mc_paths(24, seed=11):
        mc = BMC.load_mc(p)
        assert len(mc.observation) == 51 and mc.observation.extrapolated_from is None
        ext = _cut_observation(mc.observation)
        assert ext.extend_constant_velocity(50) == 10
        empty = _cut_observation(mc.observation)
        empty._occupancy_maps += [PDMOccupancyMap([], np.array([], dtype=object))] * 10
        empty._global_to_local_idcs = list(range(51))
        for acc, o in ((full, mc.observation), (cv, ext), (skip, empty)):
            acc.append([r["ttc"] for r in ST.score_token(dataclasses.replace(mc, observation=o), anchors)])
    full, cv, skip = (np.array(x) for x in (full, cv, skip))
    assert np.mean(full != cv) < 0.02 and np.mean(full != skip) > 0.03     # measured: 0.68 % and 6.4 % (300 tokens)
    assert abs(int(((full == 0) & (cv == 1)).sum()) - int(((full == 1) & (cv == 0)).sum())) <= 0.5 * (full != cv).sum() + 5
    assert int(((full == 1) & (skip == 0)).sum()) == 0


def test_processor_horizon_follows_the_scene_window():
    from navsim.planning.metric_caching.caching import gt_horizon_kwargs
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    w = lambda n: types.SimpleNamespace(num_future_frames=n, num_history_frames=4)  # noqa: E731
    assert gt_horizon_kwargs(w(10)) == {} and gt_horizon_kwargs(w(12)) == {}
    assert gt_horizon_kwargs(w(8)) == {"future_traj_num_poses": 40}
    assert MetricCacheProcessor(None, False).future_traj_num_poses == 50
    sim = MetricCacheProcessor(None, False, **gt_horizon_kwargs(w(8)))
    assert sim.future_traj_num_poses == 40 and sim.proposal_traj_num_poses == 40


def test_saved_simscale_config_differs_from_the_navtest_one_in_the_window_only():
    a = (REPO / "resources/metric_cache_saved_config.yaml").read_text().splitlines()
    b = (REPO / "resources/metric_cache_saved_config_simscale.yaml").read_text().splitlines()
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b) and diff == [("  num_future_frames: 10", "  num_future_frames: 8")]
    assert hashlib.sha256((REPO / "resources/metric_cache_saved_config_simscale.yaml").read_bytes()).hexdigest() == \
        BMC.SAVED_CFG_SIM_SHA256


def _base_scorer_class():
    src = _git_show(BASE, "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py")
    mod = types.ModuleType("pdm_scorer_base")
    sys.modules["pdm_scorer_base"] = mod                    # @dataclass looks its module up
    exec(compile(src, "pdm_scorer_base.py", "exec"), mod.__dict__)
    return mod.PDMScorer


def _probes(mc):
    t = np.linspace(0.5, 4.0, 8)
    v = max(float(mc.ego_state.dynamic_car_state.speed), 3.0)
    tr = np.zeros((5, 8, 3), np.float32)
    tr[1, :, 0] = v * t
    tr[2, :, 0] = v * t
    tr[2, :, 1] = 0.25 * t ** 2
    tr[3, :, 0] = 2.5 * v * t
    tr[4, :, 0], tr[4, :, 1] = v * t, -1.0 * t ** 2
    return tr


SCORER_ARRAYS = ("_multi_metrics", "_weighted_metrics", "_progress_raw", "_ttc_time_idcs", "_collision_time_idcs",
                 "_ego_areas")


def _score_both(mc, new, old, sim, trajs):
    import score_trajectories as ST
    simulated = sim.simulate_proposals(ST.proposal_states(mc, trajs, sim.proposal_sampling), mc.ego_state)
    args = (simulated, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
    a = new.score_proposals(*args)
    b = old.score_proposals(*args)
    return a, b


def _ref_mc_paths(n, seed=0):
    logs = sorted(e.name for e in os.scandir(REF_MC) if e.is_dir() and not e.name.startswith(("_", "metadata", "logs")))
    rng, out = np.random.default_rng(seed), []
    for lg in [logs[i] for i in rng.permutation(len(logs))]:
        d = REF_MC / lg / "unknown"
        if d.is_dir():
            ts = sorted(os.listdir(d))
            p = d / ts[int(rng.integers(len(ts)))] / "metric_cache.pkl"
            if p.is_file():
                out.append(p)
        if len(out) == n:
            return out
    return out


@real_mc
def test_real_scoring_is_bit_identical_to_the_base_scorer():
    """256 anchors + 4 probes on 16 real caches (built before this change): every scorer array equal in bytes."""
    _maps_env()
    import score_trajectories as ST
    sim, new = ST.build_simulator_scorer(record=False)
    old = _base_scorer_class()(new.proposal_sampling, new._config, new._vehicle_parameters)
    anchors = np.load(REPO / "resources/planning_vb/trajectory_anchors_256.npy").astype(np.float32)
    n_ttc_fail = n_tok = 0
    for p in _ref_mc_paths(16):
        mc = BMC.load_mc(p)
        assert len(mc.observation) == 51
        a, b = _score_both(mc, new, old, sim, np.concatenate([anchors, _probes(mc)]))
        assert a.tobytes() == b.tobytes(), p
        for k in SCORER_ARRAYS:
            assert getattr(new, k).tobytes() == getattr(old, k).tobytes(), (p, k)
        n_ttc_fail += int((new._weighted_metrics[1] < 1).sum())
        n_tok += 1
    assert n_tok == 16 and n_ttc_fail > 0, "the TTC failure paths must be reached"


def _cache(out: Path, cfg_file: Path, logs: Path, items):
    """cache_scenarios as build_metric_cache calls it, for [(log, token)] of one logs directory."""
    from omegaconf import OmegaConf
    from navsim.planning.metric_caching.caching import cache_scenarios
    cfg = OmegaConf.load(cfg_file)
    cfg.cache.cache_path, cfg.navsim_log_path, cfg.output_dir = str(out), str(logs), str(out / "metadata")
    cfg.scene_filter.tokens, cfg.scene_filter.log_names = [], []
    res = cache_scenarios([{"cfg": cfg, "log_file": lg, "tokens": [t]} for lg, t in items])
    return sum(r.successes for r in res)


@real_mc
def test_real_metric_cache_built_now_equals_the_reference_cache(tmp_path):
    """Three navtrain tokens re-cached by this code: every field and every score equal to the cache built before."""
    _maps_env()
    import score_trajectories as ST
    refs = _ref_mc_paths(3, seed=5)
    items = [(p.parent.parent.parent.name, p.parent.name) for p in refs]
    assert _cache(tmp_path, BMC.SAVED_CFG, LOGS, items) == 3
    sim, new = ST.build_simulator_scorer(record=False)
    anchors = np.load(REPO / "resources/planning_vb/trajectory_anchors_256.npy").astype(np.float32)
    for p, (lg, t) in zip(refs, items):
        mo, mn = BMC.load_mc(p), BMC.load_mc(BMC.mc_path(tmp_path, lg, t))
        assert all(v == 0.0 for v in BMC.compare_fields(mo, mn).values()), BMC.compare_fields(mo, mn)
        a, b = _score_both(mo, new, new, sim, anchors)
        a2, _ = _score_both(mn, new, new, sim, anchors)
        assert a.tobytes() == a2.tobytes()


def _cut_observation(obs, n=41):
    """obs with its first n maps only (a 4 s GT observation, before extend_constant_velocity)."""
    import copy
    o = copy.copy(obs)
    o._occupancy_maps = list(obs._occupancy_maps[:n])
    o._observation_samples = n - 1
    o._global_to_local_idcs = list(range(n))
    if hasattr(o, "_extrapolated_from"):
        del o._extrapolated_from
    return o


@real_sim
def test_simscale_metric_cache_observes_4_s_then_constant_velocity_and_scores(tmp_path):
    """SimScale scenes through the 4 + 8 config: 41 GT maps + 10 at constant velocity (51, as a 5 s cache), a 4 s
    PDM-Closed; the base scorer scores them as the live one does, with finite scores in [0, 1], and the recording copy
    agrees; cut back to the 41 GT maps, the scorer cannot score them (no silent TTC truncation)."""
    import dataclasses
    _maps_env()
    import score_trajectories as ST
    meta = SIM_SAMPLE / "synthetic_reaction_pdm_v1.0-4/meta_datas"
    names = sorted(os.listdir(meta))[:: 4000][:4]
    items = [(n[:-4], TB.load_scene_pickle(meta / n)[3]["token"]) for n in names]
    assert _cache(tmp_path, BMC.SAVED_CFG_SIM, meta, items) == 4
    sim, new = ST.build_simulator_scorer(record=False)
    old = _base_scorer_class()(new.proposal_sampling, new._config, new._vehicle_parameters)
    anchors = np.load(REPO / "resources/planning_vb/trajectory_anchors_256.npy").astype(np.float32)
    n_fail = 0
    for lg, t in items:
        mc = BMC.load_mc(BMC.mc_path(tmp_path, lg, t))
        obs = mc.observation
        assert len(obs._occupancy_maps) == 51 and len(obs) == 51 and obs.extrapolated_from == 41
        assert mc.trajectory.end_time.time_us - mc.trajectory.start_time.time_us == 4_000_000
        trajs = np.concatenate([anchors, _probes(mc)])
        _score_both(mc, new, old, sim, trajs)
        for k in SCORER_ARRAYS:
            assert np.asarray(getattr(new, k)).tobytes() == np.asarray(getattr(old, k)).tobytes(), k
        with pytest.raises(AssertionError, match="out of range"):
            _score_both(dataclasses.replace(mc, observation=_cut_observation(obs)), new, old, sim, trajs)
        res = ST.score_token(mc, trajs)
        pdms = np.array([r["pdms"] for r in res])
        assert np.isfinite(pdms).all() and (pdms >= 0).all() and (pdms <= 1).all() and all(r["rec_ok"] for r in res)
        assert {r["ttc"] for r in res} <= {0.0, 1.0} and {r["nc"] for r in res} <= {0.0, 0.5, 1.0}
        n_fail += int((pdms < 1).sum())
        assert (pdms > 0).any()
    assert n_fail > 0


# ----------------------------------------------------------------------------------------------- metric cache build
def test_build_reads_sim_rows_and_chunks_them_per_round(tmp_path):
    t = pd.DataFrame({"token": ["a" * 16, "b" * 16, "c" * 16 + "-004", "d" * 16 + "-004", "e" * 16 + "-000"],
                      "log": ["L1", "L1", "s1", "s2", "s3"],
                      "pkl": ["", "", "synthetic_reaction_pdm_v1.0-4/meta_datas/s1.pkl",
                              "synthetic_reaction_pdm_v1.0-4/meta_datas/s2.pkl",
                              "synthetic_reaction_pdm_v1.0-0/meta_datas/s3.pkl"]})
    t.to_parquet(tmp_path / "tab.parquet", index=False)
    df = BMC.read_input(types.SimpleNamespace(table=str(tmp_path / "tab.parquet")))
    assert df.split.unique().tolist() == ["tab"] and BMC.sim_rows(df).tolist() == [False, False, True, True, True]
    ch = BMC.make_sim_chunks(df[BMC.sim_rows(df)], tmp_path / "out", Path("/sim"), chunk=1)
    assert sorted((c[3], tuple(c[1])) for c in ch) == [
        ("/sim/synthetic_reaction_pdm_v1.0-0/meta_datas", (("s3", "e" * 16 + "-000"),)),
        ("/sim/synthetic_reaction_pdm_v1.0-4/meta_datas", (("s1", "c" * 16 + "-004"),)),
        ("/sim/synthetic_reaction_pdm_v1.0-4/meta_datas", (("s2", "d" * 16 + "-004"),))]
    real = BMC.read_splits(["tab"], tmp_path)
    pd.DataFrame({"token": ["x"], "log": ["L"]}).to_parquet(tmp_path / "plain.parquet", index=False)
    assert list(BMC.read_splits(["plain"], tmp_path).columns) == ["token", "log", "split"]      # no pkl column added
    assert len(real) == 5
    parts = MP.split_parts(MP.load_tables([str(tmp_path / "tab.parquet")]), 2)
    assert all(list(p.columns) == ["token", "log", "pkl"] for p in parts) and sum(len(p) for p in parts) == 5


@real_sim
def test_build_command_caches_real_and_sim_rows_and_resumes(tmp_path):
    _maps_env()
    meta = SIM_SAMPLE / "synthetic_reaction_pdm_v1.0-4/meta_datas"
    n = sorted(os.listdir(meta))[7]
    sim_tok = TB.load_scene_pickle(meta / n)[3]["token"]
    real = _ref_mc_paths(1, seed=3)[0] if REF_MC.is_dir() else None
    rows = [(sim_tok, n[:-4], f"synthetic_reaction_pdm_v1.0-4/meta_datas/{n}")]
    if real is not None:
        rows.append((real.parent.name, real.parent.parent.parent.name, ""))
    pd.DataFrame(rows, columns=["token", "log", "pkl"]).to_parquet(tmp_path / "t.parquet", index=False)
    a = types.SimpleNamespace(out=str(tmp_path / "mc"), table=str(tmp_path / "t.parquet"), limit=0, no_seed=True,
                              e_mc="", chunk=16, seed=0, workers=1, no_final_check=True)
    old = BMC.SIM_ROOT
    BMC.SIM_ROOT = SIM_SAMPLE
    try:
        BMC.cmd_build(a)
        s = json.loads((tmp_path / "mc/_state/build_summary.json").read_text())
        assert s["ok"] == len(rows) and s["bad"] == 0 and s["simscale_tokens"] == 1
        assert set((tmp_path / "mc/_state/done_tokens.txt").read_text().split()) == {r[0] for r in rows}
        BMC.cmd_build(a)
        assert json.loads((tmp_path / "mc/_state/build_summary.json").read_text())["pending_at_start"] == 0
    finally:
        BMC.SIM_ROOT = old
    obs = BMC.load_mc(BMC.mc_path(tmp_path / "mc", n[:-4], sim_tok)).observation
    assert len(obs) == 51 and obs.extrapolated_from == 41


# ----------------------------------------------------------------------------------------------- packed, raw256, score
def test_extract_human_takes_8_future_frames_for_a_simscale_row_only(sim_tree, tmp_path):
    import extract_human as H
    t, _ = _build(sim_tree, caps={"navtrain_train": 0, "navtrain_val": 0, "simscale": 6})
    d = H.extract_split(t, tmp_path, workers=1, sim_root=sim_tree["root"])
    assert (d["tokens"] == t.token.to_numpy()).all() and (d["n_avail"] == 8).all()
    assert np.allclose(d["traj"][0, :, 0], np.arange(1, 9))          # frames 4..11, x = frame index (fake poses)
    assert np.isnan(d["path"][:, 8:]).all() and d["status" if "status" in d else "eds"].shape == (6, 4)
    frames = pickle.load(open(sim_tree["root"] / t.pkl.iloc[0], "rb"))
    with pytest.raises(ValueError, match="no full scene window"):
        H.extract_token(frames, 3)                                    # a real row keeps the 10-frame rule
    with pytest.raises(ValueError, match="sim_root"):
        H.extract_split(t, tmp_path, workers=1)


def _fake_logs(d: Path, logs):
    return TRS.write_logs(d, logs)          # the module loaded once below (a second import would reload verify_assets)


def test_repack_trainval_simscale_compares_real_rows_with_the_packed_parts(sim_tree, tmp_path):
    logs = _fake_logs(tmp_path / "logs", {"logA": ["a0000000000000a0", "a0000000000000a1"], "logB": ["b0000000000000b0"]})
    real = pd.DataFrame({"token": ["a0000000000000a0", "a0000000000000a1", "b0000000000000b0"],
                         "log": ["logA", "logA", "logB"], "city": "city"})
    t, _ = _build(sim_tree, caps={"navtrain_train": 0, "navtrain_val": 0, "simscale": 3})
    df = pd.concat([real.assign(pkl=""), t[["token", "log", "city", "pkl"]]], ignore_index=True)
    monkey_root = SP.SSR_SIMSCALE_ROOT
    try:
        SP.SSR_SIMSCALE_ROOT = str(sim_tree["root"])
        ck = tmp_path / "ck/packed"
        a = RP.build_arrays(real, logs, workers=1)
        RP.write_dir(ck / "navtrain_train", real, a, {"source": "dataset"})
        rep = RP.run("trainval_simscale", ck / "trainval_simscale", df, logs, workers=1)
        assert rep["result"] == "PASS" and rep["n"] == 6 and rep["n_simscale"] == 3
        assert rep["token_sha16"] == token_sha16(df.token) != hashlib.sha256(np.array(df.token.tolist(), "U16").tobytes()).hexdigest()[:16]
        assert rep["real_rows"]["navtrain_train"] == "3 of 3 shared rows equal"
        assert rep["real_rows"]["navtrain_val"].startswith("not compared")
        tk = pd.read_parquet(ck / "trainval_simscale/tokens.parquet")
        assert list(tk.columns) == ["token", "log", "city", "row"] and tk.token.tolist() == df.token.tolist()
        g = np.load(ck / "navtrain_train/gt_traj.npy")
        g[1, 0, 0] += 1.0
        np.save(ck / "navtrain_train/gt_traj.npy", g)
        with pytest.raises(SystemExit):
            RP.run("trainval_simscale", ck / "tvsim2", df, logs, workers=1)
        assert not (ck / "tvsim2").exists()
        assert RP.main(["--split", "trainval_simscale", "--out", str(ck / "trainval_simscale"), "--skip-done",
                        "--tokens", str(_write(df, tmp_path / "tab.parquet"))]) == 0
    finally:
        SP.SSR_SIMSCALE_ROOT = monkey_root


def _write(df, p):
    df.to_parquet(p, index=False)
    return p


def test_raw256_traj_of_the_unpinned_split_takes_n_from_packed(tmp_path):
    pk = tmp_path / "packed"
    pk.mkdir()
    pd.DataFrame({"token": ["a", "b", "c"]}).to_parquet(pk / "tokens.parquet")
    out = tmp_path / "t.npy"
    assert RT.main(["--split", "trainval_simscale", "--out", str(out), "--packed", str(pk)]) == 0
    a = np.load(out)
    assert a.shape == (3, 256, 8, 3) and (a == np.load(RT.ANCHORS).astype(np.float32)).all()
    assert RT.main(["--split", "trainval_simscale", "--out", str(out), "--packed", str(pk), "--skip-done"]) == 0
    with pytest.raises(SystemExit, match="refusing"):
        RT.main(["--split", "trainval_simscale", "--out", str(out), "--packed", str(pk)])


def _score_tree(tmp_path, n_real=3, n_sim=2):
    """ck/ with packed + raw256 labels of a small trainval_simscale table, and a pinned-style official file."""
    ck = tmp_path / "ck"
    toks = [f"r{i:015x}" for i in range(n_real)] + [f"s{i:015x}-004" for i in range(n_sim)]
    tab = pd.DataFrame({"token": toks, "source": ["navtrain_train"] * n_real + ["simscale"] * n_sim})
    _write(tab, tmp_path / "tab.parquet")
    (ck / "packed/trainval_simscale").mkdir(parents=True)
    _write(pd.DataFrame({"token": toks}), ck / "packed/trainval_simscale/tokens.parquet")
    rng = np.random.default_rng(0)
    lab = rng.random((len(toks), 256, 9)).astype(np.float32)
    lab[..., [0, 1, 3, 4]] = rng.integers(0, 2, (len(toks), 256, 4))
    (ck / "labels/trainval_simscale/raw256/shards").mkdir(parents=True)
    np.save(ck / "labels/trainval_simscale/raw256/labels.npy", lab)
    (ck / "labels/trainval_simscale/raw256/shards/_plan.json").write_text(json.dumps({"token_sha16": token_sha16(toks)}))
    np.save(ck / "labels/trainval_simscale/raw256/ok.npy", np.ones((len(toks), 256), bool))
    (ck / "raw256").mkdir()
    off = SC.score_rows(lab[:n_real])[::-1].copy()                    # the pinned file holds the real rows, other order
    np.save(ck / "raw256/pdm_score_256_officialEP.npy", off)
    (ck / "raw256/pdm_score_256_officialEP.tokens.json").write_text(json.dumps(toks[:n_real][::-1]))
    return ck, tab, lab


def test_score_file_rows_in_table_order_and_real_rows_equal_the_pinned_file(tmp_path, monkeypatch):
    ck, tab, lab = _score_tree(tmp_path)
    pins = {sfx: SC.sha16(Path(str(ck / "raw256/pdm_score_256_officialEP") + sfx)) for sfx in (".npy", ".tokens.json")}
    monkeypatch.setattr(SC, "PINNED_SHA16", pins)
    args = ["--ck", str(ck), "--table", str(tmp_path / "tab.parquet"), "--plan-data", str(tmp_path / "noplan")]
    assert SC.main(args) == 0
    out = np.load(ck / f"raw256/{SC.NAME}.npy")
    assert out.shape == (5, 5, 256) and out.dtype == np.float16
    assert (out == SC.score_rows(lab)).all() and (out[:, 2] == lab[..., 2].astype(np.float16)).all()
    assert json.loads((ck / f"raw256/{SC.NAME}.tokens.json").read_text()) == tab.token.tolist()
    m = json.loads((ck / f"raw256/{SC.NAME}.meta.json").read_text())
    assert m["real_vs_pinned_officialEP"] == {"n_compared": 3, "n_differ": 0} and m["n_simscale"] == 2
    assert SC.main(args + ["--skip-done"]) == 0
    with pytest.raises(SystemExit, match="refusing"):
        SC.main(args)


def test_score_file_refuses_a_real_row_that_differs_and_an_unscored_anchor(tmp_path, monkeypatch):
    ck, tab, lab = _score_tree(tmp_path)
    off = np.load(ck / "raw256/pdm_score_256_officialEP.npy")
    off[0, 2, 5] = np.float16(0.123)
    np.save(ck / "raw256/pdm_score_256_officialEP.npy", off)
    pins = {sfx: SC.sha16(Path(str(ck / "raw256/pdm_score_256_officialEP") + sfx)) for sfx in (".npy", ".tokens.json")}
    monkeypatch.setattr(SC, "PINNED_SHA16", pins)
    args = ["--ck", str(ck), "--table", str(tmp_path / "tab.parquet"), "--plan-data", str(tmp_path / "noplan")]
    with pytest.raises(SystemExit, match="1 of 3 real rows differ"):
        SC.main(args)
    assert not list((ck / "raw256").glob(f"*{SC.NAME}*"))
    monkeypatch.setattr(SC, "PINNED_SHA16", {".npy": "x", ".tokens.json": "y"})
    with pytest.raises(SystemExit, match="not the pinned file"):
        SC.main(args)
    for f in (ck / "raw256").glob("pdm_score_256_officialEP.*"):
        f.unlink()
    plan = tmp_path / "plan"
    plan.mkdir()
    np.save(plan / "pdm_score_256.npy", SC.score_rows(lab[:3]))
    (plan / "pdm_score_256.tokens.json").write_text(json.dumps(tab.token[:3].tolist()))
    assert SC.main(args[:-1] + [str(plan)]) == 0                       # no pinned file: NC/DAC/TTC/C vs WoTE
    m = json.loads((ck / f"raw256/{SC.NAME}.meta.json").read_text())
    assert m["real_vs_wote_nc_dac_ttc_c"] == {"n_compared": 3, "n_differ": 0}
    ok = np.load(ck / "labels/trainval_simscale/raw256/ok.npy")
    ok[4, 0] = False
    np.save(ck / "labels/trainval_simscale/raw256/ok.npy", ok)
    for f in (ck / "raw256").glob(f"{SC.NAME}*"):
        f.unlink()
    with pytest.raises(SystemExit, match="not scored"):
        SC.main(args)


def test_score_file_follows_its_raw256_labels(tmp_path, monkeypatch):
    """raw256 labels of another table are refused; a kept (--skip-done) score file must come from the labels there now:
    after raw256 is rebuilt, the old file is refused, not kept, and verify_assets FAILs it."""
    ck, tab, lab = _score_tree(tmp_path)
    pins = {sfx: SC.sha16(Path(str(ck / "raw256/pdm_score_256_officialEP") + sfx)) for sfx in (".npy", ".tokens.json")}
    monkeypatch.setattr(SC, "PINNED_SHA16", pins)
    args = ["--ck", str(ck), "--table", str(tmp_path / "tab.parquet"), "--plan-data", str(tmp_path / "noplan"),
            "--skip-done"]
    plan = ck / "labels/trainval_simscale/raw256/shards/_plan.json"
    plan.write_text(json.dumps({"token_sha16": "0" * 16}))
    with pytest.raises(SystemExit, match="raw256 labels of another table"):
        SC.main(args)
    plan.write_text(json.dumps({"token_sha16": token_sha16(tab.token.tolist())}))
    assert SC.main(args) == 0
    m = json.loads((ck / f"raw256/{SC.NAME}.meta.json").read_text())
    assert m["labels_sha16"] == SC.sha16(ck / "labels/trainval_simscale/raw256/labels.npy")
    assert SC.main(args) == 0                                           # same labels: kept
    lab2 = lab.copy()
    lab2[3:, :, 2] = 0.75                                               # raw256 rebuilt, other EP on the SimScale rows
    np.save(ck / "labels/trainval_simscale/raw256/labels.npy", lab2)
    with pytest.raises(SystemExit, match="made from other raw256 labels"):
        SC.main(args)


def test_variants_read_the_split_label_source():
    sys.path.insert(0, str(REPO))
    from tools.ck.data import label_variants as LV
    from navsim.agents.para_ssr.ck import constants as C
    own = C.SPLITS["trainval_simscale"]["label_source"]
    if Path(own + ".npy").is_file():
        assert LV.label_source("trainval_simscale").path == own
    else:
        with pytest.raises(SystemExit, match="assemble_trainval_simscale_score"):
            LV.label_source("trainval_simscale")


@real_sim
def test_objects_of_a_simscale_scene_reach_4_s(tmp_path):
    import build_future_objects as BFO
    from navsim.agents.para_ssr.refiner import gt_future as G
    meta = SIM_SAMPLE / "synthetic_reaction_pdm_v1.0-4/meta_datas"
    n = sorted(os.listdir(meta))[11]
    tok = TB.load_scene_pickle(meta / n)[3]["token"]
    _write(pd.DataFrame({"token": [tok], "log": [n[:-4]], "pkl": [f"synthetic_reaction_pdm_v1.0-4/meta_datas/{n}"]}),
           tmp_path / "t.parquet")
    s = BFO.main(["--tokens", str(tmp_path / "t.parquet"), "--logs", str(tmp_path / "nologs"), "--out", str(tmp_path / "obj"),
                  "--workers", "1", "--sim-root", str(SIM_SAMPLE)])
    assert s["n_error"] == 0 and s["n_built"] == 1
    obj = G.load_objects(tmp_path / "obj" / f"{tok}.npz")
    assert int(obj["n_kf"]) == 9 and G.t_avail(obj) == 4.0


# ----------------------------------------------------------------------------------------------- shell steps
def _sandbox(tmp_path, name, files):
    """A copy of the setup scripts (and what they source) in a tree of their own: no paths.env of this checkout."""
    root = tmp_path / name
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def _env(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SSR_", "NAVSIM_", "OPENSCENE", "NUPLAN", "PARTS",
                                                                    "SPLITS", "SPLIT", "MINI", "ROUNDS"))}
    srv = tmp_path / "srv"
    (srv / "ds/navsim_logs/trainval").mkdir(parents=True, exist_ok=True)
    (srv / "ds/navsim_logs/trainval/x.pkl").write_bytes(b"")
    env.update(SSR_DATA_ROOT=str(srv / "data"), SSR_TEACHER_CACHE=str(srv / "tc"), OPENSCENE_DATA_ROOT=str(srv / "ds"),
               SSR_TEST_DATA_ROOT=str(srv / "dst"), SSR_PLAN_DATA=str(srv / "plan"), SSR_PY=sys.executable,
               SSR_RUN_ROOT=str(srv / "runs"), SETUP_DRY="1", PYTHONDONTWRITEBYTECODE="1")
    return env


SCRIPT_FILES = ("tools/release/setup/_common.sh", "tools/release/load_paths.sh", "navsim/common/ssr_paths.py")


@pytest.mark.parametrize("script, kw", [("20_metric_cache_navtrain.sh", {}), ("30_gt_stores.sh", {}),
                                        ("40_packed_from_dataset.sh", {}), ("50_warmup_labels.sh", {}),
                                        ("50_warmup_labels.sh", {"SPLITS": "navtrain_train"}),
                                        ("20_metric_cache_navtrain.sh", {"RESUME": "1", "K": "3"})])
def test_default_steps_print_the_base_commands(tmp_path, script, kw):
    """git: the navtrain steps (no SPLITS / SPLIT, or the navtrain ones) print exactly the commands of the base commit."""
    rels = SCRIPT_FILES + (f"tools/release/setup/{script}",)
    base = _sandbox(tmp_path, "base", {r: _git_show(BASE, r) for r in rels})
    now = _sandbox(tmp_path, "now", {r: (REPO / r).read_text() for r in rels})
    out = {}
    for name, root in (("base", base), ("now", now)):
        r = subprocess.run(["bash", str(root / "tools/release/setup" / script)], env=dict(_env(tmp_path), **kw),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        out[name] = r.stdout.replace(str(root), "REPO")
    assert out["now"] == out["base"] and "+ " in out["now"]


def test_simscale_steps_dry_run_their_plan(tmp_path):
    env = _env(tmp_path)
    d = Path(env["SSR_DATA_ROOT"])
    rels = SCRIPT_FILES + tuple(f"tools/release/setup/{p.name}" for p in SETUP.glob("*.sh"))
    box = _sandbox(tmp_path, "now", {r: (REPO / r).read_text() for r in rels}) / "tools/release/setup"

    def run(script, **kw):
        r = subprocess.run(["bash", str(box / script)], env=dict(env, **kw), capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return [x for x in r.stdout.splitlines() if x.startswith("+ ")]

    g = run("16_trainval_simscale_table.sh", MINI="1", ROUNDS="4")
    assert any("build_trainval_simscale_table.py" in x and "--rounds 4" in x and "--max-per-source navtrain_train=1000 "
               "navtrain_val=300" in x for x in g)
    g = run("20_metric_cache_navtrain.sh", SPLITS="trainval_simscale", K="2")
    assert any(f"--tables {d}/splits/trainval_simscale.parquet" in x and "mc_tvsim_part" in x for x in g)
    assert any("--data trainval_simscale" in x for x in g)
    g = run("30_gt_stores.sh", SPLIT="trainval_simscale", K="2")
    assert any("build_e2e_side.py build" in x and "--status" in x and "--tokens" in x for x in g)
    g = run("40_packed_from_dataset.sh", SPLITS="trainval_simscale", RESUME="1")
    assert any("repack_dataset.py --split trainval_simscale" in x and "--skip-done" in x for x in g)
    g = run("50_warmup_labels.sh", SPLITS="trainval_simscale")
    order = [next(i for i, x in enumerate(g) if k in x) for k in ("make_raw256_traj.py", "label_cands.py",
                                                                  "assemble_trainval_simscale_score.py", "label_variants.py")]
    assert order == sorted(order) and not any("assemble_official_ep.py" in x for x in g)
    g = run("prepare_trainval_simscale.sh", MINI="1", SKIP="65")
    assert any("verify_assets.py --profile full --data trainval_simscale" in x for x in g)
    assert not any("simscale_download.py" in x for x in g)
    assert any("simscale_download.py" in x for x in run("prepare_trainval_simscale.sh", DOWNLOAD="1", SKIP="65"))
    r = subprocess.run(["bash", str(box / "20_metric_cache_navtrain.sh")], env=dict(env, SPLITS="navtrain_train trainval_simscale"),
                       capture_output=True, text=True)
    assert r.returncode == 3 and "give it alone" in r.stderr


# ----------------------------------------------------------------------------------------------- download
def _tar(path: Path, members: dict):
    with tarfile.open(path, "w:gz") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, __import__("io").BytesIO(data))


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A fake SimScale round 4 on a fake server: listing, part index, meta archive, two hist parts."""
    srv = tmp_path / "server"
    srv.mkdir()
    pre = "SimScale/synthetic_reaction_pdm_v1.0-4"
    scenes = {"simscale_pdm_v1.0-4_sensor_blobs_hist_0": ["s1-004", "s2-004"], "simscale_pdm_v1.0-4_sensor_blobs_hist_1": ["s3-004"]}
    (srv / "simscale_pdm_v1.0-4_sensor_blobs_hist.json").write_text(json.dumps(scenes))
    _tar(srv / "simscale_pdm_v1.0-4_meta_datas.tar.gz", {f"{pre}/meta_datas/L-{s}.pkl": b"p" for v in scenes.values() for s in v})
    for part, ss in scenes.items():
        _tar(srv / f"{part}.tar.gz", {f"{pre}/sensor_blobs_hist/{s}/{c}/f{i}.jpg": b"j" for s in ss for c in DL.CAMERAS
                                      for i in range(4)})
    d = f"{DL.DATA_DIR}/synthetic_reaction_pdm_v1.0-4"

    def get(url, token=""):
        if url.endswith("_sensor_blobs_hist"):
            files = [f"{part}.tar.gz" for part in scenes]
            return [{"type": "file", "path": f"{d}/simscale_pdm_v1.0-4_sensor_blobs_hist/{f}", "size": (srv / f).stat().st_size}
                    for f in files]
        files = ["simscale_pdm_v1.0-4_sensor_blobs_hist.json", "simscale_pdm_v1.0-4_meta_datas.tar.gz"]
        return [{"type": "file", "path": f"{d}/{f}", "size": (srv / f).stat().st_size} for f in files] + \
            [{"type": "directory", "path": f"{d}/simscale_pdm_v1.0-4_sensor_blobs_hist", "size": 0}]

    fetched = []

    def download(a, dl, token="", tries=10):
        fetched.append(a["name"])
        f = dl / a["name"]
        f.write_bytes((srv / a["name"]).read_bytes())
        assert f.stat().st_size == a["size"]
        return f

    monkeypatch.setattr(DL, "_get_json", get)
    monkeypatch.setattr(DL, "download", download)
    return dict(srv=srv, fetched=fetched, scenes=scenes)


def test_download_unpacks_checks_stamps_and_resumes(server, tmp_path):
    root, dl = tmp_path / "root", tmp_path / "dl"
    args = ["--rounds", "4", "--root", str(root), "--download-dir", str(dl)]
    assert DL.main(args + ["--plan-only"]) == 0 and not root.exists()
    assert DL.main(args) == 0
    rd = root / "synthetic_reaction_pdm_v1.0-4"
    assert sorted(os.listdir(rd / "meta_datas")) == ["L-s1-004.pkl", "L-s2-004.pkl", "L-s3-004.pkl"]
    assert sorted(os.listdir(rd / "sensor_blobs_hist")) == ["s1-004", "s2-004", "s3-004"]
    assert (rd / "simscale_pdm_v1.0-4_sensor_blobs_hist.json").is_file()
    assert len(os.listdir(dl / ".done")) == 4 and not list(dl.glob("*.tar.gz"))
    n = len(server["fetched"])
    assert DL.main(args) == 0 and len(server["fetched"]) == n                       # everything stamped: nothing fetched


def test_download_refuses_a_part_without_its_scenes_and_a_foreign_archive(server, tmp_path):
    pre = "SimScale/synthetic_reaction_pdm_v1.0-4"
    _tar(server["srv"] / "simscale_pdm_v1.0-4_sensor_blobs_hist_1.tar.gz",
         {f"{pre}/sensor_blobs_hist/s3-004/CAM_F0/f0.jpg": b"j"})                    # s3: 1 image instead of 12
    root, dl = tmp_path / "root", tmp_path / "dl"
    with pytest.raises(SystemExit, match="problems after extraction"):
        DL.main(["--rounds", "4", "--root", str(root), "--download-dir", str(dl)])
    assert not (dl / ".done/simscale_pdm_v1.0-4_sensor_blobs_hist_1.tar.gz").exists()
    assert (dl / ".done/simscale_pdm_v1.0-4_sensor_blobs_hist_0.tar.gz").exists()
    bad = tmp_path / "foreign.tar.gz"
    _tar(bad, {"other/x.txt": b"x"})
    with pytest.raises(SystemExit, match="not under SimScale"):
        DL.extract(bad, tmp_path / "x", 4)


def test_download_plan_orders_index_meta_then_parts_by_number():
    listing = {"synthetic_reaction_pdm_v1.0-0": [("a_sensor_blobs_hist.json", 1)]}

    def get(url, token=""):
        d = "SimScale_data/synthetic_reaction_pdm_v1.0-0"
        if url.endswith("_sensor_blobs_hist"):
            return [{"type": "file", "path": f"{d}/simscale_pdm_v1.0-0_sensor_blobs_hist/simscale_pdm_v1.0-0_sensor_blobs_hist_{k}.tar.gz",
                     "size": 10 + k} for k in (10, 2, 0, 1)]
        return [{"type": "file", "path": f"{d}/simscale_pdm_v1.0-0_sensor_blobs_hist.json", "size": 5},
                {"type": "file", "path": f"{d}/simscale_pdm_v1.0-0_meta_datas.tar.gz", "size": 7}]
    p = DL.plan([0], get=get)
    assert [x["kind"] for x in p] == ["index", "meta", "hist", "hist", "hist", "hist"] and listing
    assert [x["name"].rsplit("_", 1)[1] for x in p[2:]] == ["0.tar.gz", "1.tar.gz", "2.tar.gz", "10.tar.gz"]
    assert p[0]["url"] == f"{DL.HF}/datasets/{DL.REPO_ID}/resolve/{DL.REVISION}/SimScale_data/synthetic_reaction_pdm_v1.0-0/" \
                          "simscale_pdm_v1.0-0_sensor_blobs_hist.json"


# ----------------------------------------------------------------------------------------------- verify_assets
TRS = _load("tvsim_release_setup_t", REPO / "tests/test_release_setup.py")
VA = TRS.VA
SIM_TOK = {"s1": f"{1:016x}-004", "s2": f"{2:016x}-004"}


class TvsimTree:
    """test_release_setup's navtrain tree + the trainval_simscale assets of 3 train, 1 val and 2 SimScale rows."""

    def __init__(self, tmp_path, monkeypatch):
        self.nt = TRS.Tree(tmp_path, monkeypatch)
        self.root, self.data, self.ck, self.tc = tmp_path, self.nt.data, self.nt.ck, self.nt.tc
        self.sim = tmp_path / "simscale"
        monkeypatch.setenv("SSR_SIMSCALE_ROOT", str(self.sim))
        monkeypatch.setattr(VA, "SORTED_SHA16", {s: sorted_sha(TRS.TOK[s]) for s in TRS.TOK})
        rows = [(t, g, "c", s, -1, "") for s in ("navtrain_train", "navtrain_val") for t, g in zip(TRS.TOK[s], TRS.LOG_OF[s])]
        for k, h in (("s1", 1), ("s2", 2)):
            stem, tok = sim_scene(self.sim, 4, "logS_veh-9_00009", f"{h + 50:016x}", f"{h:016x}")
            rows.append((tok, stem, "c1", "simscale", 4, f"synthetic_reaction_pdm_v1.0-4/meta_datas/{stem}.pkl"))
        t = pd.DataFrame(rows, columns=["token", "log", "city", "source", "round", "pkl"])
        t.insert(3, "row", np.arange(len(t), dtype=np.int64))
        t["round"] = t["round"].astype(np.int8)
        (self.data / "splits").mkdir(parents=True, exist_ok=True)
        _write(t, self.data / "splits/trainval_simscale.parquet")
        (self.data / "splits/trainval_simscale.meta.json").write_text(json.dumps({"mode": "full"}))
        self.t, n = t, len(t)
        toks = t.token.tolist()
        # metric cache (lzma pickles of a stand-in with the observation length), GT stores
        mc = self.data / "metric_cache"
        (mc / "_state/done_tokens.txt").write_text("\n".join(toks))
        for tok, lg, src in zip(t.token, t.log, t.source):
            p = mc / lg / "unknown" / tok / "metric_cache.pkl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with lzma.open(p, "wb") as f:
                pickle.dump(types.SimpleNamespace(observation=types.SimpleNamespace(
                    _occupancy_maps=[0] * 51, **({"_extrapolated_from": 41} if src == "simscale" else {}))), f)
            for d in (self.data / "objects/navtrain", self.data / "sdf/navtrain"):
                d.mkdir(parents=True, exist_ok=True)
                np.savez(d / f"{tok}.npz", x=np.zeros(1))
            (self.data / "e2e_side" / tok[:2]).mkdir(parents=True, exist_ok=True)
            np.savez(self.data / "e2e_side" / tok[:2] / f"{tok}.npz", p_pdm=np.float64(1.0), cl_n=np.int32(5))
        # packed, raw256, variants, score file
        pk = self.ck / "packed/trainval_simscale"
        pk.mkdir(parents=True)
        _write(t[["token", "log", "city", "row"]], pk / "tokens.parquet")
        np.save(pk / "gt_traj.npy", np.zeros((n, 8, 3), np.float32))
        np.save(pk / "status.npy", np.zeros((n, 8), np.float32))
        np.save(pk / "ok.npy", np.ones(n, bool))
        (pk / "meta.json").write_text(json.dumps({"token_sha16": token_sha16(toks), "real_rows": {}}))
        rd = self.ck / "labels/trainval_simscale/raw256"
        (rd / "shards").mkdir(parents=True)
        np.save(rd / "labels.npy", np.zeros((n, 256, 9), np.float32))
        np.save(rd / "ok.npy", np.ones((n, 256), bool))
        (rd / "shards/_plan.json").write_text(json.dumps({"token_sha16": token_sha16(toks)}))
        vd = self.ck / "ck2/labels/trainval_simscale" / VA.VAR_NAME
        vd.mkdir(parents=True)
        np.save(vd / "labels.npy", np.zeros((n, 96, 9), np.float32))
        np.save(vd / "ok.npy", np.ones((n, 96), bool))
        np.save(vd / "traj.npy", np.zeros((n, 96, 8, 3), np.float32))
        np.savez(vd / "index.npz", anchor_idx=np.zeros((n, 16), np.int16), valid=np.ones((n, 16, 6), bool), built=np.ones(n, bool))
        from navsim.agents.para_ssr.ck.anchor_sampler import AnchorSampler
        np.save(self.ck / f"raw256/{VA.TVSIM_SCORE}.npy", np.ones((n, 5, 256), np.float16))
        (self.ck / f"raw256/{VA.TVSIM_SCORE}.tokens.json").write_text(json.dumps(toks))
        score_sha = VA.sha16_file(self.ck / f"raw256/{VA.TVSIM_SCORE}.npy")
        (self.ck / f"raw256/{VA.TVSIM_SCORE}.meta.json").write_text(json.dumps(
            {"real_vs_pinned_officialEP": {"n_compared": 4, "n_differ": 0}, "npy_sha16": score_sha,
             "labels_sha16": VA.sha16_file(rd / "labels.npy")}))
        (vd / "build.json").write_text(json.dumps({
            "token_sha16": token_sha16(toks), "anchors_sha16": VA.ANCHORS_SHA16, "n_built": n, "n_tokens": n,
            "names": VA.VAR_NAMES, "n_anchors": 16, "seed": 0, "sampler": AnchorSampler(seed=0).config(),
            "label_source": {"path": str(self.ck / f"raw256/{VA.TVSIM_SCORE}.npy"), "npy_sha16": score_sha}}))
        # teacher caches: every row
        det = self.tc / "det"
        man = json.loads((det / "manifest.json").read_text())
        (det / "manifest.json").write_text(json.dumps(dict(man, num_samples_written=n)))
        for tok in toks:
            (det / "samples" / tok[:2]).mkdir(parents=True, exist_ok=True)
            np.savez(det / "samples" / tok[:2] / f"{tok}.npz", bev_feature=np.zeros((4, 2, 2), np.float16))
        mapc = self.tc / "map"
        (mapc / "index.json").write_text(json.dumps({tok: ["s0", i] for i, tok in enumerate(toks)}))
        meta = json.loads((mapc / "meta.json").read_text())
        (mapc / "meta.json").write_text(json.dumps(dict(meta, num_frames=n)))
        np.save(mapc / "bev/s0.npy", np.zeros((n, 4, 2, 2), np.float16))

    def run(self, *args, quick=True):
        argv = ["--profile", "full", "--data", "trainval_simscale", "--registry", str(self.nt.registry),
                "--det-teacher", "bevfusion", "--map-teacher", "resmap"] + (["--quick"] if quick else []) + list(args)
        return VA.main(argv)


@pytest.fixture
def tvsim(tmp_path, monkeypatch):
    monkeypatch.setattr(VA, "N_SAMPLE", 50)
    for k in ("DET_TEACHER", "MAP_TEACHER", "SSR_RELEASE_REGISTRY"):
        monkeypatch.delenv(k, raising=False)
    return TvsimTree(tmp_path, monkeypatch)


@pytest.mark.parametrize("quick", [True, False])
def test_verify_tvsim_complete_tree_passes(tvsim, capsys, quick, monkeypatch):
    monkeypatch.setattr(VA, "check_gt_loader", lambda c, fix: c.rep.add("PASS", "gt.loader_all_tokens", "stubbed"))
    assert tvsim.run(quick=quick) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    for name in ("tvsim.table", "dataset.simscale", "official_ep.trainval_simscale", "metric_cache.trainval_simscale",
                 "packed.trainval_simscale", "labels.raw256.trainval_simscale", "labels.var.trainval_simscale",
                 "teacher.det.bevfusion.cache", "teacher.map.resmap.cache", "gt.sdf"):
        assert f"PASS  {name}" in out, name
    assert ("metric_cache.window" in out) == (not quick)
    assert "official_ep:" not in out and "packed.navtrain_val" not in out      # navtrain-only rows are not checked


@pytest.mark.parametrize("rel, name", [
    ("data/metric_cache/logS_veh-9_00009-{s}-004/unknown/{t}/metric_cache.pkl", "metric_cache.trainval_simscale"),
    ("data/sdf/navtrain/{t}.npz", "gt.sdf"),
    ("data/e2e_side/{t2}/{t}.npz", "gt.e2e_side"),
    ("data/objects/navtrain/{t}.npz", "gt.objects"),
    ("tc/det/samples/{t2}/{t}.npz", "teacher.det.bevfusion.cache"),
    ("data/ck/packed/trainval_simscale/ok.npy", "packed.trainval_simscale"),
    ("data/ck/labels/trainval_simscale/raw256/ok.npy", "labels.raw256.trainval_simscale"),
    ("data/ck/ck2/labels/trainval_simscale/var_separate_sampler16_accstraight/index.npz", "labels.var.trainval_simscale"),
    ("data/ck/raw256/pdm_score_256_officialEP_trainval_simscale.tokens.json", "official_ep.trainval_simscale"),
    ("simscale/synthetic_reaction_pdm_v1.0-4/sensor_blobs_hist/{s}-004/CAM_L0/{f2}.jpg", "dataset.simscale"),
])
def test_verify_tvsim_missing_simscale_row_fails(tvsim, capsys, rel, name):
    """Every check covers the SimScale rows (a navtrain-only check would pass here)."""
    t, s = SIM_TOK["s2"], f"{52:016x}"
    f2 = f"{(int(t[:16], 16) - 1) % 16 ** 16:016x}-004"            # frame 2 of the scene
    (tvsim.root / rel.format(t=t, t2=t[:2], s=s, f2=f2)).unlink()
    assert tvsim.run(quick=False) == 1
    assert any(x.startswith(f"FAIL  {name}") for x in capsys.readouterr().out.splitlines()), name


def test_verify_tvsim_map_index_window_order_and_mode(tvsim, capsys, monkeypatch):
    monkeypatch.setattr(VA, "check_gt_loader", lambda c, fix: c.rep.add("PASS", "gt.loader_all_tokens", "stubbed"))
    idx = json.loads((tvsim.tc / "map/index.json").read_text())
    del idx[SIM_TOK["s1"]]
    (tvsim.tc / "map/index.json").write_text(json.dumps(idx))
    assert tvsim.run() == 1 and "trainval_simscale tokens not in index.json" in capsys.readouterr().out
    idx[SIM_TOK["s1"]] = ["s0", 5]
    (tvsim.tc / "map/index.json").write_text(json.dumps(idx))
    p = tvsim.data / "metric_cache" / tvsim.t.log.iat[5] / "unknown" / SIM_TOK["s2"] / "metric_cache.pkl"
    with lzma.open(p, "wb") as f:
        pickle.dump(types.SimpleNamespace(observation=types.SimpleNamespace(_occupancy_maps=[0] * 41)), f)
    assert tvsim.run(quick=False) == 1 and "a simscale cache has 51, extrapolated from 41" in capsys.readouterr().out
    with lzma.open(p, "wb") as f:                                   # a 5 s cache that was not extended: not SimScale's
        pickle.dump(types.SimpleNamespace(observation=types.SimpleNamespace(_occupancy_maps=[0] * 51)), f)
    assert tvsim.run(quick=False) == 1 and "extrapolated from None; a simscale cache" in capsys.readouterr().out
    with lzma.open(p, "wb") as f:
        pickle.dump(types.SimpleNamespace(observation=types.SimpleNamespace(_occupancy_maps=[0] * 51,
                                                                            _extrapolated_from=41)), f)
    (tvsim.data / "splits/trainval_simscale.meta.json").write_text(json.dumps({"mode": "mini"}))
    assert tvsim.run() == 0 and "WARN  tvsim.table.mode" in capsys.readouterr().out
    assert tvsim.run("--strict") == 1
    capsys.readouterr()
    tvsim.t.iloc[[4, 5]].to_parquet(tvsim.data / "splits/x.parquet")
    t = pd.read_parquet(tvsim.data / "splits/trainval_simscale.parquet")
    t.loc[[0, 1], "token"] = t.loc[[1, 0], "token"].values
    _write(t, tvsim.data / "splits/trainval_simscale.parquet")
    assert tvsim.run() == 1
    out = capsys.readouterr().out
    assert "FAIL  packed.trainval_simscale" in out and "token / log are not the table's" in out


def test_verify_tvsim_score_and_variants_follow_their_sources(tvsim, capsys, monkeypatch):
    """Full mode: a score file whose raw256 labels were rebuilt after it, and variants built from another score file,
    FAIL (the quick mode does not hash them)."""
    monkeypatch.setattr(VA, "check_gt_loader", lambda c, fix: c.rep.add("PASS", "gt.loader_all_tokens", "stubbed"))
    assert tvsim.run(quick=False) == 0
    capsys.readouterr()
    rd = tvsim.ck / "labels/trainval_simscale/raw256"
    old = np.load(rd / "labels.npy")
    np.save(rd / "labels.npy", old + 1)
    assert tvsim.run(quick=False) == 1 and "made from raw256 labels sha16" in capsys.readouterr().out
    assert tvsim.run() == 0
    capsys.readouterr()
    np.save(rd / "labels.npy", old)
    vb = tvsim.ck / "ck2/labels/trainval_simscale" / VA.VAR_NAME / "build.json"
    b = json.loads(vb.read_text())
    b["label_source"]["npy_sha16"] = "0" * 16
    vb.write_text(json.dumps(b))
    assert tvsim.run(quick=False) == 1 and "built from another score file" in capsys.readouterr().out


def test_verify_tvsim_defaults_to_the_camera_teachers(tvsim, capsys):
    argv = ["--profile", "full", "--data", "trainval_simscale", "--registry", str(tvsim.nt.registry), "--quick",
            "--only", "teacher_cache"]
    assert VA.main(argv) == 1
    out = capsys.readouterr().out
    assert "teacher.det: 'bevformer' has status 'missing'" in out and "teacher.map: 'mask2map' is not in the registry" in out


def test_verify_navtrain_default_is_unchanged(tvsim, capsys):
    """The tvsim tree also holds the full navtrain tree: without --data the navtrain checks run, and only them."""
    assert tvsim.nt.run() == 0
    out = capsys.readouterr().out
    assert "PASS  official_ep:" in out and "tvsim" not in out and "trainval_simscale" not in out
