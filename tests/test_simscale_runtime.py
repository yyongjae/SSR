"""trainval + SimScale training runtime (role A): the token sha helper (ck/tokens.py), the token-table training set
(dataloader.MultiSceneLoader over the real SceneLoader and navsim.common.simscale.SimScaleSceneLoader), run_training's
+train_data / +tf32, the token-count and label-coverage guards, the refiner row space (N_ROWS = rows of the warm-up
packed tokens.parquet) and the labeler defaults.  CPU only.

Real data, read only: the dataset logs / sensors (ssr_paths.OPENSCENE_DATA_ROOT), the split parquets of
/home/external-user/ssd/yongjae_refiner/splits and the SimScale round-4 dev sample (SSR_SIMSCALE_ROOT, default
/home/external-user/ssd/simscale_dev/simscale).  The tiny token table (2 navtrain_train + 2 navtrain_val + 3 SimScale
rows) and every other file are written under pytest's tmp dir.  The real-builder test runs this file as a script in a
subprocess (CK_BEV64=1 is read at import, as for tests/release_agent_arms.py):

  env CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD /home/external-user/miniconda3/envs/ssr/bin/python -m pytest -q \
    -p no:cacheprovider tests/test_simscale_runtime.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from navsim.common import ssr_paths as SP  # noqa: E402  (stdlib only)

os.environ.setdefault("NUPLAN_MAPS_ROOT", SP.NUPLAN_MAPS_ROOT)     # read by navsim.common.dataclasses at import
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
from navsim.agents.para_ssr.ck import e2e_data as ED  # noqa: E402
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck.tokens import token_array, token_sha16  # noqa: E402
from navsim.common import dataclasses as DC  # noqa: E402
from navsim.common import simscale as SS  # noqa: E402
from navsim.common.dataclasses import SceneFilter, SensorConfig  # noqa: E402
from navsim.common.dataloader import MultiSceneLoader, SceneLoader, filter_scenes  # noqa: E402
from navsim.planning.script import run_training as RT  # noqa: E402

SPLITS_DIR = Path("/home/external-user/ssd/yongjae_refiner/splits")       # read only
SF_YAML = REPO / "navsim/planning/script/config/common/scene_filter/navtrain.yaml"
LOG_SPLIT = REPO / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
LOGS = Path(SP.OPENSCENE_DATA_ROOT) / "navsim_logs/trainval"
SENSORS = Path(SP.OPENSCENE_DATA_ROOT) / "sensor_blobs/trainval"
ROUND = 4
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0")
PRE_PORT = "c8bd032"       # branch point of the trainval + SimScale port: the main recipe's code before it


def _sim_root():
    for r in (getattr(SP, "SSR_SIMSCALE_ROOT", None), os.environ.get("SSR_SIMSCALE_ROOT"),
              "/home/external-user/ssd/simscale_dev/simscale"):
        if r and (Path(r) / SS.round_dir(ROUND) / SS.HIST_DIR).is_dir():
            return Path(r)
    return None


SIM_ROOT = _sim_root()
HAVE = SIM_ROOT is not None and SPLITS_DIR.is_dir() and LOGS.is_dir() and SENSORS.is_dir()
need = pytest.mark.skipif(not HAVE, reason="dataset logs / sensors, split parquets or the SimScale dev sample absent")


# ----------------------------------------------------------------------------------------------- dev token table
def dev_rows(n_train: int = 2, n_val: int = 2, n_sim: int = 3, sim_root=None) -> pd.DataFrame:
    """A tiny training token table (simscale.TABLE_COLUMNS) in the contract's row order: the first n_train tokens of
    splits/e2e_train_trainlogs.parquet, the first n_val of splits/navtrain_val.parquet, then n_sim round-4 SimScale
    scenes whose CAM_F0 / L0 / R0 frames 0..3 exist (the first scene dirs in sorted order), sorted by token."""
    root = Path(sim_root or SIM_ROOT)
    tr = pd.read_parquet(SPLITS_DIR / "e2e_train_trainlogs.parquet").head(n_train)
    va = pd.read_parquet(SPLITS_DIR / "navtrain_val.parquet").head(n_val)
    rows = [dict(token=t, log=g, city=c, source="navtrain_train", round=-1, pkl="")
            for t, g, c in zip(tr.token, tr.log, tr.map_location)]
    rows += [dict(token=t, log=g, city="", source="navtrain_val", round=-1, pkl="") for t, g in zip(va.token, va.log)]
    hist, meta = root / SS.round_dir(ROUND) / SS.HIST_DIR, root / SS.round_dir(ROUND) / SS.META_DIR
    by_scene = {n[:-4].rsplit("-", 2)[1] + "-" + n[:-4].rsplit("-", 2)[2]: n[:-4] for n in os.listdir(meta)}
    sim = []
    for scene in sorted(os.listdir(hist)):
        if len(sim) == n_sim:
            break
        if not all((hist / scene / c).is_dir() and len(os.listdir(hist / scene / c)) == 4 for c in CAMS):
            continue
        stem = by_scene[scene]
        frames = SS.read_scene(root, SS.scene_pkl(ROUND, stem))
        sim.append(dict(token=frames[SS.CURRENT]["token"], log=stem, city=frames[SS.CURRENT]["map_location"],
                        source=SS.SOURCE_SIM, round=ROUND, pkl=SS.scene_pkl(ROUND, stem)))
    rows += sorted(sim, key=lambda r: r["token"])
    df = pd.DataFrame(rows)
    df["row"] = np.arange(len(df), dtype=np.int64)
    df["round"] = df["round"].astype(np.int8)
    return df[list(SS.TABLE_COLUMNS)]


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    df = dev_rows()
    p = tmp_path_factory.mktemp("table") / "trainval_simscale.parquet"
    df.to_parquet(p, index=False)
    return p, df


@pytest.fixture
def maps(monkeypatch):
    """Scene building reads dataclasses.NUPLAN_MAPS_ROOT (an env value of import time)."""
    monkeypatch.setattr(DC, "NUPLAN_MAPS_ROOT", DC.NUPLAN_MAPS_ROOT or SP.NUPLAN_MAPS_ROOT)


def _cfg(**over):
    """the part of the training config the dataset builders read (real scene filter yaml and log split)."""
    y = OmegaConf.load(LOG_SPLIT)
    d = {"scene_filter": OmegaConf.load(SF_YAML), "train_logs": y.train_logs, "val_logs": y.val_logs,
         "test_logs": y.test_logs, "navsim_log_path": str(LOGS), "sensor_blobs_path": str(SENSORS), "cache_path": "",
         "force_cache_computation": False, "trainer": {"params": {"limit_val_batches": 0}}}
    d.update(over)
    return OmegaConf.create(d)


def _agent(builders=()):
    """the agent surface the dataset builders use (para_ssr's sensor config: cam_f0 / l0 / r0 at frames 2, 3)."""
    sc = SensorConfig(cam_f0=[2, 3], cam_l0=[2, 3], cam_l1=False, cam_l2=False, cam_r0=[2, 3], cam_r1=False,
                      cam_r2=False, cam_b0=False, lidar_pc=False)
    return SimpleNamespace(config=SimpleNamespace(use_fut_frames=False, plan_anchor=False), _ck_e2e2=None,
                           get_sensor_config=lambda: sc, get_feature_builders=lambda: list(builders),
                           get_target_builders=lambda: [])


def _loaders(df):
    real, sim = df[df.source != SS.SOURCE_SIM], df[df.source == SS.SOURCE_SIM]
    rl = SceneLoader(LOGS, SENSORS, RT.real_scene_filter(_cfg(), real), _agent().get_sensor_config())
    return rl, SS.SimScaleSceneLoader(SIM_ROOT, sim, sensor_config=_agent().get_sensor_config())


# ----------------------------------------------------------------------------------------------- tokens.py
def test_token_array_keeps_the_pins_and_never_truncates():
    for split, pin in (("navtrain_train", "0b56b9aecbe1f094"), ("navtrain_val", "0fea22c41a4106b0")):
        toks = pd.read_parquet(REPO / f"resources/km_corridor/packed/{split}/tokens.parquet").token.astype(str).tolist()
        assert token_array(toks).dtype == np.dtype("<U16") and token_sha16(toks) == pin     # the pinned packed rows
        old = hashlib.sha256(np.ascontiguousarray(np.array(toks, "U16")).tobytes()).hexdigest()[:16]
        assert token_sha16(toks) == old
    # shorter tokens (test fixtures) stay U16 as before: the existing fixtures' values do not change
    assert token_sha16(["b", "a"]) == hashlib.sha256(np.ascontiguousarray(np.array(["b", "a"], "U16")).tobytes()
                                                     ).hexdigest()[:16]
    assert token_array([]).dtype == np.dtype("<U16")
    real, sim = "231d0b185aab533d", "231d0b185aab533d-004"
    a = token_array([real, sim, "231d0b185aab533d-003"])
    assert a.dtype == np.dtype("<U20") and a.tolist() == [real, sim, "231d0b185aab533d-003"]   # never truncated
    # U16 made a SimScale row equal to its real twin and two rounds equal to each other
    u16 = lambda t: hashlib.sha256(np.ascontiguousarray(np.array(t, "U16")).tobytes()).hexdigest()[:16]  # noqa: E731
    assert u16([real, sim]) == u16([real, real]) and token_sha16([real, sim]) != token_sha16([real, real])
    assert token_sha16([sim, "231d0b185aab533d-003"]) != token_sha16(["231d0b185aab533d-003", sim])


def _wu_fixture(root: Path, tokens, logs=None):
    """structurally valid warm-up sources (packed / raw256 / variant) for `tokens` in that row order, every row kept,
    whose build.json / packed meta / raw-label plan carry token_sha16 of that order (as the setup writes them)."""
    from navsim.agents.para_ssr.ck import anchor_sampler as AS
    n = len(tokens)
    pk, rl, vd = root / "packed", root / "raw256", root / "var"
    for d in (pk, rl / "shards", vd):
        d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"token": list(tokens), "log": list(logs or ["L"] * n), "city": [""] * n,
                  "row": np.arange(n)}).to_parquet(pk / "tokens.parquet", index=False)
    sha = token_sha16(tokens)
    np.save(pk / "ok.npy", np.ones(n, bool))
    np.save(pk / "gt_traj.npy", np.zeros((n, 8, 3), np.float32))
    np.save(rl / "labels.npy", np.full((n, 256, 9), 0.5, np.float32))
    np.save(rl / "ok.npy", np.ones((n, 256), bool))
    np.save(vd / "traj.npy", np.zeros((n, 96, 8, 3), np.float32))
    np.save(vd / "labels.npy", np.full((n, 96, 9), 0.5, np.float32))
    np.save(vd / "ok.npy", np.ones((n, 96), bool))
    np.savez(vd / "index.npz", valid=np.ones((n, 16, 6), bool), anchor_idx=np.zeros((n, 16), np.int16),
             built=np.ones(n, bool))
    build = {"names": list(D2.VNAMES), "n_anchors": 16, "seed": 0, "sampler": AS.AnchorSampler(seed=0).config(),
             "anchors_sha16": AS.ANCHORS_SHA16, "n_tokens": n, "token_sha16": sha, "speeds": [-1.0, -0.5, 0.5],
             "lats": [-0.5, 0.5], "combine": "separate", "s_on_frac": 0.2, "ext": "straight"}
    (vd / "build.json").write_text(json.dumps(build))
    (pk / "meta.json").write_text(json.dumps({"token_sha16": sha}))
    (rl / "shards" / "_plan.json").write_text(json.dumps({"token_sha16": sha}))
    return {"packed": str(pk), "raw_labels": str(rl), "var_dir": str(vd)}


def test_warmup_provenance_hashes_full_tokens(tmp_path):
    """e2e_data2's provenance guard goes through token_sha16: a 20-character SimScale token set is hashed in full (a
    build hashed with the old U16 rule, i.e. truncated tokens, is refused)."""
    toks = ["0002267a294e52a7", "231d0b185aab533d-004", "231d0b185aab533d-003"]
    w = _wu_fixture(tmp_path / "ok", toks)
    prov = D2._WUArrays(D2.warmup_cfg(w)).check()["provenance"]
    assert prov["token_sha16"] == token_sha16(toks) == prov["variant build.json"] == prov["packed meta.json"]
    w2 = _wu_fixture(tmp_path / "u16", toks)
    old = hashlib.sha256(np.ascontiguousarray(np.array(toks, "U16")).tobytes()).hexdigest()[:16]
    (Path(w2["packed"]) / "meta.json").write_text(json.dumps({"token_sha16": old}))
    with pytest.raises(D2.WUSourceError, match="token_sha16"):
        D2._WUArrays(D2.warmup_cfg(w2)).check()


# ----------------------------------------------------------------------------------------------- MultiSceneLoader
@need
def test_multi_scene_loader_serves_real_and_sim(table, maps):
    _, df = table
    rl, sl = _loaders(df)
    ml = MultiSceneLoader(df.token.tolist(), [rl, sl])
    assert ml.tokens == df.token.tolist() and len(ml) == len(df) and ml[len(df) - 1] == df.token.iloc[-1]
    assert ml.tokens is ml.tokens                         # one cached list (Dataset indexes it per sample)
    for tok, src in zip(df.token, df.source):
        scene = ml.get_scene_from_token(tok)
        assert scene.scene_metadata.initial_token == tok == scene.frames[3].token        # the full token
        if src == SS.SOURCE_SIM:
            assert len(tok) == 20 and ml.loader_of(tok) is sl and len(scene.frames) == SS.NUM_FRAMES
            assert scene.scene_metadata.num_future_frames == SS.NUM_FUTURE_FRAMES      # 4 + 8, not 4 + 10
            assert scene.get_future_trajectory(8).poses.shape == (8, 3)
        else:
            assert len(tok) == 16 and ml.loader_of(tok) is rl and len(scene.frames) == 14
        ai = ml.get_agent_input_from_token(tok)
        assert ai.cameras[3].cam_f0.image is not None and ai.cameras[3].cam_f0.image.shape == (1080, 1920, 3)
    per_log = ml.get_tokens_list_per_log()
    assert sorted(t for ts in per_log.values() for t in ts) == sorted(df.token)
    # pickling (spawn workers): the copy serves the same scenes
    ml2 = pickle.loads(pickle.dumps(ml))
    sim_tok = df.token[df.source == SS.SOURCE_SIM].iloc[0]
    assert ml2.tokens == ml.tokens and ml2.get_scene_from_token(sim_tok).scene_metadata.initial_token == sim_tok
    # a table token no loader serves raises instead of shrinking the set
    with pytest.raises(ValueError, match="served by no loader"):
        MultiSceneLoader(df.token.tolist() + ["ffffffffffffffff-004"], [rl, sl])
    with pytest.raises(ValueError, match="served by loader"):
        MultiSceneLoader(df.token.tolist(), [rl, sl, sl])
    with pytest.raises(ValueError, match="duplicate"):
        MultiSceneLoader(df.token.tolist() + [sim_tok], [rl, sl])
    # restrict_tokens (the teacher-cache filter's way in) keeps the order
    assert ml2.restrict_tokens(lambda t: t != sim_tok) == 1
    assert ml2.tokens == [t for t in df.token if t != sim_tok] and sim_tok not in ml2


@need
def test_navtrain_window_drops_every_sim_scene(table, tmp_path):
    """why the SimScale rows need their own loader: the navtrain 4 + 10 window drops every 12-frame scene silently;
    SimScale's 4 + 8 keeps them all."""
    _, df = table
    sim = df[df.source == SS.SOURCE_SIM]
    d = tmp_path / "logs"
    d.mkdir()
    for p in sim.pkl:
        (d / Path(p).name).symlink_to(SIM_ROOT / p)
    nav = OmegaConf.load(SF_YAML)
    sf = SceneFilter(num_history_frames=nav.num_history_frames, num_future_frames=nav.num_future_frames,
                     frame_interval=1, has_route=True)
    assert (sf.num_history_frames, sf.num_future_frames) == (4, 10) and filter_scenes(d, sf) == {}
    got = filter_scenes(d, SceneFilter(num_history_frames=4, num_future_frames=8, frame_interval=1, has_route=True))
    assert sorted(got) == sorted(sim.token)


# ----------------------------------------------------------------------------------------------- build_table_datasets
@need
def test_build_table_datasets_order_val_and_refusals(table, tmp_path, maps):
    p, df = table
    train, val, t = RT.build_table_datasets(_cfg(), _agent(), "trainval_simscale", table_path=p, sim_root=SIM_ROOT)
    assert val is None and RT.dataset_tokens(train) == df.token.tolist() and len(train) == len(df)
    assert isinstance(train._scene_loader, MultiSceneLoader) and t.token.tolist() == df.token.tolist()
    RT.check_table_tokens(RT.dataset_tokens(train), t)
    assert train[len(df) - 1] == ({}, {})                 # the Dataset serves a SimScale row (no builders here)
    # limit_val_batches != 0: the main recipe's validation set (navtrain on val_logs), built as before (val_logs cut
    # to the table's one val log here: the whole set loads 214 logs)
    val_logs = sorted(set(df.log[df.source == "navtrain_val"]))
    _, val, _ = RT.build_table_datasets(_cfg(trainer={"params": {"limit_val_batches": 1.0}}, val_logs=val_logs),
                                        _agent(), "trainval_simscale", table_path=p, sim_root=SIM_ROOT)
    assert val is not None and len(val) > 0
    assert {s[0]["log_name"] for s in val._scene_loader.scene_frames_dicts.values()} == set(val_logs)
    # refusals: a real row outside the navtrain filter, a missing scene pickle, a SimScale row of a test log, columns
    bad = df.copy()
    bad.loc[0, "token"] = "0000000000000000"
    q = tmp_path / "bad_real.parquet"
    bad.to_parquet(q, index=False)
    with pytest.raises(RuntimeError, match="served by no scene loader"):
        RT.build_table_datasets(_cfg(), _agent(), "trainval_simscale", table_path=q, sim_root=SIM_ROOT)
    bad = df.copy()
    bad.loc[len(df) - 1, "pkl"] = SS.scene_pkl(ROUND, "nope-0000000000000000-004")
    bad.to_parquet(q, index=False)
    with pytest.raises(FileNotFoundError, match="scene pickles missing"):
        RT.build_table_datasets(_cfg(), _agent(), "trainval_simscale", table_path=q, sim_root=SIM_ROOT)
    test_log = sorted(OmegaConf.load(LOG_SPLIT).test_logs)[0]
    bad = df.copy()
    bad.loc[len(df) - 1, "pkl"] = SS.scene_pkl(ROUND, f"{test_log}-0000000000000000-004")
    bad.to_parquet(q, index=False)
    with pytest.raises(RuntimeError, match="test logs"):
        RT.build_table_datasets(_cfg(), _agent(), "trainval_simscale", table_path=q, sim_root=SIM_ROOT)
    for mutate, msg in ((lambda d: d.drop(columns=["pkl"]), "columns"),
                        (lambda d: d.assign(row=d.row[::-1].values), "row"),
                        (lambda d: pd.concat([d, d.tail(1).assign(row=len(d))]), "duplicate"),
                        (lambda d: d.assign(source="navtest"), "unknown sources")):
        mutate(df).to_parquet(q, index=False)
        with pytest.raises(ValueError, match=msg):
            RT.read_token_table(q)


@need
def test_teacher_cache_filter_on_the_table_set_is_caught(table, maps):
    """restrict_dataset_to_stores reaches the table loader (it has no scene_frames_dicts) and a dropped token stops the
    run (run_training.check_table_tokens), with what is missing per source."""
    from navsim.agents.para_ssr.distill.teacher_store import restrict_dataset_to_stores
    p, df = table
    train, _, t = RT.build_table_datasets(_cfg(), _agent(), "trainval_simscale", table_path=p, sim_root=SIM_ROOT)
    sim_tok = df.token[df.source == SS.SOURCE_SIM].iloc[1]
    store = SimpleNamespace(teacher="fake", has_token=lambda tok: tok != sim_tok)
    assert restrict_dataset_to_stores(train, [store], name="train") == 1
    assert RT.dataset_tokens(train) == [x for x in df.token if x != sim_tok]
    with pytest.raises(RuntimeError, match="simscale"):
        RT.check_table_tokens(RT.dataset_tokens(train), t)
    with pytest.raises(RuntimeError, match="order differs"):
        RT.check_table_tokens(list(reversed(df.token)), t)


# ----------------------------------------------------------------------------------------------- main recipe unchanged
def _pre_port_module(path: str, name: str):
    """the file as it was before the port (git PRE_PORT), loaded as a module of this tree."""
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_PORT}:{path}"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"git revision {PRE_PORT} not in this clone")
    mod = types.ModuleType(name)
    mod.__file__ = str(REPO / path)
    exec(compile(r.stdout, f"{path}@{PRE_PORT}", "exec"), mod.__dict__)
    return mod


@need
def test_main_path_builds_the_same_datasets_as_before():
    """train_data absent: build_datasets of the code before the port (with its SceneLoader) and of this tree give the
    same training and validation tokens in the same order, from the real navtrain filter on a few logs."""
    head_dl = _pre_port_module("navsim/common/dataloader.py", "dataloader_pre_port")
    head_rt = _pre_port_module("navsim/planning/script/run_training.py", "run_training_pre_port")
    head_rt.SceneLoader = head_dl.SceneLoader
    tr = pd.read_parquet(SPLITS_DIR / "e2e_train_trainlogs.parquet").head(3)
    va = pd.read_parquet(SPLITS_DIR / "navtrain_val.parquet").head(1)
    y = OmegaConf.load(LOG_SPLIT)
    cfg = _cfg(train_logs=sorted(set(tr.log)) + [y.train_logs[0]], val_logs=sorted(set(va.log)))
    assert RT.train_data_name(cfg) == "navtrain"
    old_train, old_val = head_rt.build_datasets(cfg, _agent())
    new_train, new_val = RT.build_datasets(cfg, _agent())
    assert type(new_train._scene_loader) is SceneLoader and len(old_train) > 0 and len(old_val) > 0
    for old, new in ((old_train, new_train), (old_val, new_val)):
        assert RT.dataset_tokens(new) == old._scene_loader.tokens
        frames = lambda d: [[f["token"] for f in s] for s in d._scene_loader.scene_frames_dicts.values()]  # noqa: E731
        assert frames(new) == frames(old)
        assert new._scene_loader._scene_filter == old._scene_loader._scene_filter


class _Stop(Exception):
    pass


class _Data(list):
    tokens = property(list)


def _wire(monkeypatch, tmp_path, build, table_build=None, drop=None):
    """main() with everything heavy stubbed; -> what it built (callbacks, val loader, calls)."""
    seen = {"calls": []}
    monkeypatch.setenv("SSR_EXPERIMENT_SHA", "abc123")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(RT, "HydraConfig", SimpleNamespace(
        get=lambda: SimpleNamespace(runtime=SimpleNamespace(choices={"scene_filter": "navtrain"}))))
    monkeypatch.setattr(RT.pl, "seed_everything", lambda s, workers=True: None)

    def filt(train, val):
        if drop is not None:
            train.remove(drop)
    agent = SimpleNamespace(get_training_callbacks=lambda: [], filter_datasets_to_teacher_cache=filt)
    monkeypatch.setattr(RT, "instantiate", lambda c: agent)
    monkeypatch.setattr(RT, "AgentLightningModule", lambda agent: None)
    monkeypatch.setattr(RT, "DataLoader", lambda data, **k: ("loader", len(data)))
    monkeypatch.setattr(RT, "build_loggers", lambda cfg: None)
    monkeypatch.setattr(RT, "build_datasets", lambda cfg, a: (seen["calls"].append("build_datasets"), build())[1])
    if table_build is not None:
        monkeypatch.setattr(RT, "build_table_datasets",
                            lambda cfg, a, name: (seen["calls"].append(f"table:{name}"), table_build())[1])

    class Trainer:
        def __init__(self, **kw):
            seen["callbacks"] = kw["callbacks"]

        def fit(self, **kw):
            seen["val"] = kw["val_dataloaders"]
            raise _Stop
    monkeypatch.setattr(RT.pl, "Trainer", Trainer)
    return seen


def _main_cfg(tmp_path, **over):
    return OmegaConf.create({"train_logs": ["a"], "val_logs": ["b"], "test_logs": ["c"], "agent": {},
                             "output_dir": str(tmp_path / "train"), "use_cache_without_dataset": False,
                             "dataloader": {"params": {}}, "trainer": {"params": {}}, **over})


def test_main_wiring_main_recipe_and_table(monkeypatch, tmp_path):
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.get_float32_matmul_precision())
    try:
        # main recipe: build_datasets, no guard callback, the torch float32 settings untouched
        seen = _wire(monkeypatch, tmp_path, lambda: (_Data(["t1", "t2"]), _Data(["v1"])))
        with pytest.raises(_Stop):
            RT.main.__wrapped__(_main_cfg(tmp_path))
        assert seen["calls"] == ["build_datasets"] and seen["callbacks"] == [] and seen["val"] == ("loader", 1)
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision()) == flags
        # +train_data=trainval_simscale +tf32=true: the table builder, the label guard, TF32 set, no val loader
        tab = pd.DataFrame({"token": ["t1", "s1-004"], "source": ["navtrain_train", "simscale"]})
        seen = _wire(monkeypatch, tmp_path, None, lambda: (_Data(["t1", "s1-004"]), None, tab))
        with pytest.raises(_Stop):
            RT.main.__wrapped__(_main_cfg(tmp_path, train_data="trainval_simscale", tf32=True))
        assert seen["calls"] == ["table:trainval_simscale"] and seen["val"] is None
        assert [type(c) for c in seen["callbacks"]] == [RT.TableInputGuard]
        assert torch.backends.cuda.matmul.allow_tf32 and torch.get_float32_matmul_precision() == "high"
        # a token the teacher-cache filter drops stops the run before any training
        seen = _wire(monkeypatch, tmp_path, None, lambda: (_Data(["t1", "s1-004"]), None, tab), drop="s1-004")
        with pytest.raises(RuntimeError, match="training set != token table"):
            RT.main.__wrapped__(_main_cfg(tmp_path, train_data="trainval_simscale"))
        for over, err in (({"train_data": "navtest"}, ValueError), ({"tf32": "maybe"}, ValueError),
                          ({"train_data": "trainval_simscale", "use_cache_without_dataset": True}, ValueError)):
            _wire(monkeypatch, tmp_path, lambda: (_Data(["t1"]), _Data()))
            with pytest.raises(err):
                RT.main.__wrapped__(_main_cfg(tmp_path, **over))
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[:2]
        torch.set_float32_matmul_precision(flags[2])


def test_tf32_switch_and_hydra_keys():
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.get_float32_matmul_precision())
    try:
        torch.backends.cudnn.allow_tf32 = False
        for v in (None, False, "false", "off", 0):
            assert RT.apply_tf32(OmegaConf.create({} if v is None else {"tf32": v})) is False
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert RT.apply_tf32(OmegaConf.create({"tf32": True})) is True
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[:2]
        torch.set_float32_matmul_precision(flags[2])
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "navsim/planning/script/config/training"), version_base="1.2"):
        base = compose(config_name="default_training", overrides=["experiment_name=x"])
        cfg = compose(config_name="default_training", overrides=["experiment_name=x", "+train_data=trainval_simscale",
                                                                 "+tf32=true"])
    GlobalHydra.instance().clear()
    assert "train_data" not in base and "tf32" not in base                 # absent by default: the main recipe
    assert RT.train_data_name(base) == "navtrain" and RT.apply_tf32(base) is False
    assert RT.train_data_name(cfg) == "trainval_simscale" and cfg.tf32 is True


def test_table_paths_follow_the_split_entry_and_fall_back_clearly(monkeypatch, caplog):
    from navsim.agents.para_ssr.ck import constants as CKC
    split = CKC.SPLITS.get("trainval_simscale")
    if split is not None:        # the setup side's entry: its table and SimScale root, no warning
        with caplog.at_level("WARNING", logger=RT.logger.name):
            t, r = RT.table_paths("trainval_simscale")
        assert str(t) == split["tokens"] and str(r) == (split.get("simscale_root") or SP.SSR_SIMSCALE_ROOT)
        assert not caplog.records
    monkeypatch.setattr(CKC, "SPLITS", {k: v for k, v in CKC.SPLITS.items() if k != "trainval_simscale"})
    monkeypatch.delattr(SP, "SSR_SIMSCALE_ROOT", raising=False)
    monkeypatch.setenv("SSR_SIMSCALE_ROOT", "/x/simscale")
    with caplog.at_level("WARNING", logger=RT.logger.name):
        t, r = RT.table_paths("trainval_simscale")
    assert t == Path(SP.SSR_DATA_ROOT) / "splits/trainval_simscale.parquet" and r == Path("/x/simscale")
    assert len([x for x in caplog.records if x.levelname == "WARNING"]) == 2


# ----------------------------------------------------------------------------------------------- label coverage guard
def test_table_input_guard():
    g = RT.TableInputGuard(n_tokens=8, strict=0.999)
    b = {"sim_reward_valid": torch.ones(4), "ck2_wu_ok": torch.ones(4, dtype=torch.bool), "x": torch.zeros(4, 3)}
    assert g.add(b) is None and g.add(b) == {"sim_reward_valid": 1.0, "ck2_wu_ok": 1.0} and g.checked
    g = RT.TableInputGuard(n_tokens=8, strict=0.999)
    g.add({"sim_reward_valid": torch.tensor([1.0, 1.0, 0.0, 1.0]), "ck2_wu_ok": torch.ones(4, dtype=torch.bool)})
    with pytest.raises(RuntimeError, match="sim_reward_valid on only 0.8750"):
        g.add({"sim_reward_valid": torch.ones(4), "ck2_wu_ok": torch.ones(4, dtype=torch.bool)})
    g = RT.TableInputGuard(n_tokens=4, strict=0.999)
    with pytest.raises(RuntimeError, match="ck2_wu_ok on only 0.7500.*warm-up labels"):
        g.add({"ck2_wu_ok": torch.tensor([True, True, False, True]), "trajectory": torch.zeros(4, 8, 3)})
    g = RT.TableInputGuard(n_tokens=4)                    # no key the builders emit: nothing to check, no error
    assert g.add({"trajectory": torch.zeros(4, 8, 3)}) == {} and g.checked
    # rank 0 only, and only until checked
    g = RT.TableInputGuard(n_tokens=4)
    g.on_train_batch_start(SimpleNamespace(is_global_zero=False), None, ({}, {"ck2_wu_ok": torch.zeros(4)}), 0)
    assert g.tokens == 0


# ----------------------------------------------------------------------------------------------- refiner row space
def test_row_space_follows_the_packed_table(tmp_path):
    toks = ["0002267a294e52a7", "231d0b185aab533d-004", "231d0b185aab533d-003", "00022b6fc91d554a", "abcd-001"]
    w = _wu_fixture(tmp_path / "wu", toks)
    assert ED.packed_rows(w["packed"]) == 5
    io = tmp_path / "io"
    ls = D2.LabelStore2(io, packed=w["packed"])
    assert ls._n_rows is None and ls.n_rows == 5 and ls.refresh(4)["n_rows"] == 5          # read at first use
    assert pickle.loads(pickle.dumps(ls)).n_rows == 5 and D2.LabelStore2(io, n_rows=7, packed=w["packed"]).n_rows == 7
    from navsim.agents.para_ssr.ck.online2 import CKE2E2, CKE2E2Config
    ck = CKE2E2(CKE2E2Config.from_any({"enabled": True, "io_dir": str(io), "warmup": w}))
    assert ck.labels().n_rows == 5 and ck.labels().packed == w["packed"]
    # a generation is created with the run's rows and read back with them (or with its own meta)
    with pytest.raises(ValueError, match="n_rows"):
        D2.open_generation2(io, 4, mode="w+")
    g = D2.open_generation2(io, 4, ls.n_rows, mode="w+")
    D2.write_generation_rows2(g, [1, 4], np.zeros((2, 96, 8, 3)), np.zeros((2, 96, 9)), np.ones((2, 96), bool),
                              np.ones((2, 96), bool))
    assert D2.open_generation2(io, 4)["row_state"].shape == (5,) and ls.refresh(4)["n_prev"] == 2
    with pytest.raises(ValueError, match="n_rows"):
        D2.open_generation2(io, 4, 85109)
    out = ls.lookup([1, 4, 0], tokens=[toks[1], toks[4], toks[0]], epoch=5)
    assert out["has"].tolist() == [True, True, False]
    # the supply warning divides by the store's rows, not by 85,109
    from navsim.agents.para_ssr.ck.online import label_supply_warning
    for r in range(2):
        ED.write_rec_done(io, 4, r, 2, f"aaaa000{r}", [], 4)
    ck2 = SimpleNamespace(cfg=SimpleNamespace(io_dir=str(io), record_from_epoch=4, label_lag=1), record_until=28,
                          epoch=5)
    st = dict(ls.refresh(4))
    w8 = label_supply_warning(ck2, st)
    assert w8["n_rec"] == 8 and w8["frac_labeled"] == pytest.approx(2 / 5)
    assert label_supply_warning(ck2, dict(st, n_rows=None))["frac_labeled"] == pytest.approx(2 / 8)   # old stats
    # CK1 store / opener: same rules
    with pytest.raises(ValueError, match="n_rows"):
        ED.open_generation(tmp_path / "io1", 3, mode="w+")
    ED.open_generation(tmp_path / "io1", 3, 5, mode="w+")
    assert ED.open_generation(tmp_path / "io1", 3)["row_state"].shape == (5,)
    assert ED.LabelStore(tmp_path / "io1", {"packed": w["packed"]}).n_rows == 5


def test_labeler_row_space_split_and_worker_cap(tmp_path):
    from tools.ck.e2e import labeler as L1
    from tools.ck.e2e2 import labeler2 as L2
    p = tmp_path / "tokens.parquet"
    pd.DataFrame({"token": [f"t{i:015d}" for i in range(6)] + ["231d0b185aab533d-004"], "log": ["__fake__"] * 7,
                  "row": np.arange(7)}).to_parquet(p, index=False)
    lab = L2.Labeler2(tmp_path / "io", 1, workers=1, tokens=str(p), scorer="fake", log=lambda s: None)
    assert lab.n_rows == 7 and lab.split == "navtrain_train"                 # the table's rows, not 85,109
    assert L2.Labeler2(tmp_path / "io", 1, workers=1, tokens=str(p), n_rows=5, scorer="fake", log=lambda s: None
                       ).n_rows == 5
    with pytest.raises(AssertionError):
        L2.Labeler2(tmp_path / "io", 1, workers=1, tokens=str(p), n_rows=8, scorer="fake", log=lambda s: None)
    with pytest.raises(AssertionError, match="metric cache root"):
        L2.Labeler2(tmp_path / "io", 1, workers=1, tokens=str(p), split="no_such_split", scorer="fake",
                    log=lambda s: None)
    assert L1.MAX_WORKERS == 48 or os.environ.get("LABELER_MAX_WORKERS")
    code = "import sys; sys.path.insert(0, sys.argv[1]); from tools.ck.e2e import labeler as L; print(L.MAX_WORKERS)"
    env = {k: v for k, v in os.environ.items() if k != "LABELER_MAX_WORKERS"}
    out = [subprocess.run([sys.executable, "-c", code, str(REPO)], capture_output=True, text=True, timeout=300,
                          env=dict(env, CUDA_VISIBLE_DEVICES="", **e)).stdout.split()[-1]
           for e in ({}, {"LABELER_MAX_WORKERS": "96"})]
    assert out == ["48", "96"]


def test_launcher_passes_rows_and_split_to_the_labeler(tmp_path):
    sh = REPO / "tools/ck/e2e2/train_e2e2.sh"
    pk = tmp_path / "pk"
    pk.mkdir()
    pd.DataFrame({"token": ["a" * 16, "b" * 16, "c" * 16 + "-004"], "log": ["l"] * 3,
                  "row": np.arange(3)}).to_parquet(pk / "tokens.parquet", index=False)
    L = tmp_path / "root/r/launch"
    L.mkdir(parents=True)
    (L / "refiner_effective.json").write_text(json.dumps({"warmup": {"packed": str(pk)}}))
    base = {k: v for k, v in os.environ.items() if k not in ("LABELER_WORKERS", "LABELER_MAX_WORKERS",
                                                                "SSR_TRAIN_SPLIT")}

    def cmd(gpus, **e):
        r = subprocess.run(["bash", str(sh), "_labeler_cmd", "--gpus", gpus], capture_output=True, text=True,
                           timeout=120, env=dict(base, E2E_ROOT=str(tmp_path / "root"), RUN="r", **e))
        return r.returncode, r.stdout.split(), r.stderr

    rc, a, _ = cmd("0,1,2,3")
    assert rc == 0 and a[a.index("--n-rows") + 1] == "3" and a[a.index("--split") + 1] == "navtrain_train"
    assert a[a.index("--tokens") + 1] == str(pk / "tokens.parquet") and a[a.index("--workers") + 1] == "48"
    rc, a, _ = cmd("0,1,2,3,4,5,6,7", SSR_TRAIN_SPLIT="trainval_simscale", LABELER_MAX_WORKERS="96")
    assert rc == 0 and a[a.index("--split") + 1] == "trainval_simscale" and a[a.index("--workers") + 1] == "96"
    rc, _, err = cmd("0,1", LABELER_MAX_WORKERS="many")
    assert rc == 3 and "LABELER_MAX_WORKERS" in err


# ----------------------------------------------------------------------------------------------- real builders
@need
def test_real_builders_on_sim_and_real_tokens(table, tmp_path):
    """The training Dataset of a token table with the real para_ssr builders (dino_vits14_1148 registry flags, CPU,
    backbone_pretrained false, the recipe's 64 m grid / seg map head / v2 score target / refiner warm-up targets in the
    no-teacher mode) yields finite tensors for SimScale and real tokens, also through a DataLoader worker and after
    pickling; ck_row is the table row (full-token lookup), the label coverage guard passes."""
    p, df = table
    env = dict(os.environ, CK_BEV64="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO), OMP_NUM_THREADS="1",
               NUPLAN_MAPS_ROOT=SP.NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
               OPENSCENE_DATA_ROOT=SP.OPENSCENE_DATA_ROOT, NAVSIM_EXP_ROOT=str(tmp_path / "exp"))
    out = tmp_path / "builders.json"
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "builders", str(p), str(SIM_ROOT), str(tmp_path),
                        str(out)], capture_output=True, text=True, timeout=1800, env=env, cwd=str(tmp_path))
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-6000:])
    rec = json.loads(out.read_text())
    assert rec["tf32"] and rec["val"] is None and rec["tokens"] == df.token.tolist()
    assert rec["builders"][:2] == ["ParaSSRTargetBuilder", "DrivableSegTargetBuilder"]
    assert {"AnchorScoreTargetBuilder", "CK2E2ETargetBuilder"} <= set(rec["builders"])
    for tok, row in zip(df.token, df.row):
        s = rec["samples"][tok]
        assert s["nonfinite"] == [] and s["ck_row"] == int(row) and s["sim_reward_valid"] == 1.0 and s["ck2_wu_ok"]
    assert rec["image_shape"][-2:] == [672, 1148]
    assert rec["loader_batch"] == 2 and rec["loader_nonfinite"] == [] and rec["pickled_equal"]
    assert rec["guard"] == {"sim_reward_valid": 1.0, "ck2_wu_ok": 1.0}


def _builders_main(table_path: str, sim_root: str, work: str, out: str) -> int:
    """(subprocess) compose the agent as the recipe does, with the dino_vits14_1148 registry flags, build the
    token-table training set and record what every sample holds."""
    import yaml
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from torch.utils.data import DataLoader
    from navsim.agents.para_ssr.refiner import adapters
    assert adapters.BEV64, "CK_BEV64=1 expected (the recipe's 64 m grid)"
    work = Path(work)
    df = pd.read_parquet(table_path)
    wu = _wu_fixture(work / "warmup", df.token.tolist(), df.log.tolist())
    score = work / "pdm_score_256_officialEP_trainval_simscale"
    np.save(str(score) + ".npy", np.random.default_rng(0).random((len(df), 5, 256)).astype(np.float16))
    Path(str(score) + ".tokens.json").write_text(json.dumps(df.token.tolist()))
    reg = yaml.safe_load((REPO / "tools/release/registry.yaml").read_text())["backbones"]["dino_vits14_1148"]
    flags = [f"++agent.config.{k}={json.dumps(v).replace(' ', '')}" for k, v in reg["flags"].items()]
    ck = {"enabled": "true", "io_dir": str(work / "refiner"), "use_teachers": "false", "teacher_det_run": "null",
          "teacher_map_run": "null", "lambda_kd_score": "0.0", "lat_kd.balance": "fixed", "lat_kd.fixed": "0.0",
          "kd_calib.enabled": "false", "warmup.packed": wu["packed"], "warmup.raw_labels": wu["raw_labels"],
          "warmup.var_dir": wu["var_dir"]}
    ov = ["agent=para_ssr_agent", "experiment_name=simscale_builders", "scene_filter=navtrain", "split=trainval",
          "agent.config.use_task_interaction=true", "agent.config.use_det_motion_head=true",
          "agent.config.use_map_head=true", "agent.config.plan_anchor=true",
          f"agent.config.plan_anchor_file={REPO}/resources/planning_vb/trajectory_anchors_256.npy",
          f"agent.config.plan_score_file={score}", "agent.config.plan_heading_from_xy=true",
          f"agent.config.image_architecture={reg['timm']}", "++agent.config.backbone_pretrained=false",
          "++agent.config.pc_range=[-32.0,0.0,-1.0,32.0,64.0,3.0]",
          "++agent.config.map_pc_range=[-32.0,0.0,-1.0,32.0,64.0,3.0]", "++agent.config.bev_h=100",
          "++agent.config.bev_w=100", "++agent.config.map_head_type=seg", "++agent.config.bev_seg_weight=3.0",
          "++agent.config.undistort=true", "trainer.params.limit_val_batches=0", "+train_data=trainval_simscale",
          "+tf32=true"] + flags + [f"++agent.config.ck_e2e2.{k}={v}" for k, v in ck.items()]
    with initialize_config_dir(config_dir=str(REPO / "navsim/planning/script/config/training"), version_base="1.2"):
        cfg = compose(config_name="default_training", overrides=ov)
    name = RT.train_data_name(cfg)
    tf32 = RT.apply_tf32(cfg)
    agent = instantiate(cfg.agent)
    train, val, table = RT.build_table_datasets(cfg, agent, name, table_path=Path(table_path), sim_root=Path(sim_root))
    RT.check_table_tokens(RT.dataset_tokens(train), table)
    copy = pickle.loads(pickle.dumps(train))          # as a spawn worker gets it: before any sample was built

    def nonfinite(d):
        return sorted(k for k, v in d.items() if torch.is_tensor(v) and v.is_floating_point()
                      and not bool(torch.isfinite(v).all()))
    rec = {"tf32": tf32, "val": None if val is None else len(val), "tokens": RT.dataset_tokens(train),
           "builders": [type(b).__name__ for b in agent.get_target_builders()], "samples": {}}
    for i, tok in enumerate(RT.dataset_tokens(train)):
        f, t = train[i]
        rec["samples"][tok] = {"nonfinite": nonfinite(f) + nonfinite(t), "ck_row": int(t["ck_row"]),
                               "sim_reward_valid": float(t["sim_reward_valid"]), "ck2_wu_ok": bool(t["ck2_wu_ok"]),
                               "ref_gt_ok": bool(t["ref_gt_ok"])}
        rec["image_shape"] = list(f["camera_feature"].shape) if "camera_feature" in f else \
            [list(v.shape) for k, v in f.items() if v.ndim >= 4][0]
    i_sim = int(np.flatnonzero(df.source.to_numpy() == SS.SOURCE_SIM)[0])
    i_real = 0
    dl = DataLoader(train, batch_size=2, sampler=[i_sim, i_real], num_workers=1)
    fb, tb = next(iter(dl))
    rec["loader_batch"] = int(tb["ck_row"].numel())
    rec["loader_nonfinite"] = nonfinite(fb) + nonfinite(tb)
    g = RT.TableInputGuard(n_tokens=2)
    rec["guard"] = g.add(tb)
    f2, t2 = copy[i_sim]
    f1, t1 = train[i_sim]
    rec["pickled_equal"] = all(torch.equal(f1[k], f2[k]) for k in f1) and all(torch.equal(t1[k], t2[k]) for k in t1)
    Path(out).write_text(json.dumps(rec))
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["builders"]:
        sys.exit(_builders_main(*sys.argv[2:6]))
    sys.exit(f"usage: {sys.argv[0]} builders <table> <sim_root> <work dir> <out.json>")
