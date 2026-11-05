"""Release setup: repack_dataset, derive_tokens, make_parts, make_raw256_traj, verify_assets, the setup scripts, lock and docs.

CPU only.  Everything runs on fake trees under tmp_path.  The scripts run with SETUP_DRY=1, which prints the commands, so no
download, build or training happens.  Tests marked "real" read this server's data when present and are skipped elsewhere.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SETUP = REPO / "tools/release/setup"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools/release"))
sys.path.insert(0, str(REPO / "tools/refiner"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


RP = _load("repack_dataset_t", SETUP / "repack_dataset.py")
DT = _load("derive_tokens_t", SETUP / "derive_tokens.py")
MP = _load("make_parts_t", SETUP / "make_parts.py")
RT = _load("make_raw256_traj_t", SETUP / "make_raw256_traj.py")
ML = _load("make_lock_t", SETUP / "make_lock.py")
TE = _load("teacher_env_t", SETUP / "teacher_env.py")
VA = _load("verify_assets_t", REPO / "tools/release/verify_assets.py")

D_REAL = Path(os.environ.get("SSR_DATA_ROOT") or "/home/external-user/ssd/yongjae_refiner")
real_data = pytest.mark.skipif(not (D_REAL / "ck/packed/navtrain_train/tokens.parquet").is_file(), reason="no packed data")


# ----------------------------------------------------------------------------------------------- fake data
def fake_frames(tokens, log="logA", n_pad=3, n_after=17):
    """A log with `tokens` as consecutive t0 frames: enough frames around for extract_human's windows."""
    names = [f"pad{i}" for i in range(n_pad)] + list(tokens) + [f"post{i}" for i in range(n_after)]
    frames = []
    for i, t in enumerate(names):
        cams = {c: {"data_path": f"{log}/{c}/{t}.jpg"} for c in VA.CAMERAS}
        frames.append({"token": t, "frame_idx": i, "timestamp": 1_000_000 + i * 500_000, "map_location": "city",
                       "ego2global_translation": [float(i), 0.5 * i, 0.0], "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
                       "ego_dynamic_state": [1.0 + i, 0.25, 0.5, -0.5], "driving_command": [0, 1, 0, 0],
                       "roadblock_ids": ["r"], "cams": cams, "log_name": log})
    return frames


def write_logs(d: Path, logs: dict):
    d.mkdir(parents=True, exist_ok=True)
    for log, tokens in logs.items():
        with open(d / f"{log}.pkl", "wb") as f:
            pickle.dump(fake_frames(tokens, log), f)
    return d


LOGS = {"logA": ["a0000000000000a0", "a0000000000000a1", "a0000000000000a2"], "logB": ["b0000000000000b0", "b0000000000000b1"]}


def token_df():
    rows = [(t, g, "city") for g, ts in LOGS.items() for t in ts]
    return pd.DataFrame(rows, columns=["token", "log", "city"])


# ----------------------------------------------------------------------------------------------- repack
def test_repack_arrays_are_ground_truth_and_ego_state(tmp_path):
    logs = write_logs(tmp_path / "logs", LOGS)
    df = token_df()
    a = RP.build_arrays(df, logs, workers=1)
    assert a["gt_traj"].shape == (5, 8, 3) and a["gt_traj"].dtype == np.float32
    assert a["status"].shape == (5, 8) and a["status"].dtype == np.float32
    assert a["ok"].dtype == bool and a["ok"].all()
    # status = driving_command (4), then vx vy ax ay of the t0 frame: a0 sits at position 3 of its log (n_pad = 3)
    assert a["status"][0].tolist() == [0, 1, 0, 0, 4.0, 0.25, 0.5, -0.5]
    # row order follows the table, not the log order inside the extractor
    assert a["status"][3].tolist() == [0, 1, 0, 0, 4.0, 0.25, 0.5, -0.5]
    assert a["gt_traj"][0, 0, 0] == pytest.approx(1.0)


def test_repack_run_writes_the_contract_files_after_pin_check(tmp_path):
    logs = write_logs(tmp_path / "logs", LOGS)
    df = token_df()
    a = RP.build_arrays(df, logs, workers=1)
    pins = RP.array_shas(df.token.tolist(), a)
    out = tmp_path / "packed"
    rep = RP.run("navtrain_train", out, df, logs, workers=1, pins=pins)
    assert rep["result"] == "PASS"
    assert sorted(p.name for p in out.iterdir()) == ["gt_traj.npy", "meta.json", "ok.npy", "status.npy", "tokens.parquet"]
    t = pd.read_parquet(out / "tokens.parquet")
    assert list(t.columns) == ["token", "log", "city", "row"] and t.row.tolist() == list(range(5))
    assert t.token.tolist() == df.token.tolist()
    m = json.loads((out / "meta.json").read_text())
    assert m["source"] == "dataset" and m["token_sha16"] == pins["token_sha16"] and m["n_ok"] == 5
    assert not {"v2_ckpt_sha16", "dump_dir", "cand_sha16"} & set(m)
    # the same token_sha16 the warm-up provenance guard computes (e2e_data2._WUArrays.provenance)
    sha = hashlib.sha256(np.ascontiguousarray(np.array(t.token.tolist(), "U16")).tobytes()).hexdigest()[:16]
    assert sha == m["token_sha16"]
    assert not list(tmp_path.glob("packed.tmp*"))


def test_repack_refuses_to_overwrite(tmp_path):
    out = tmp_path / "packed"
    out.mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        RP.run("navtrain_train", out, token_df(), tmp_path / "logs", workers=1)


def test_repack_pin_mismatch_publishes_nothing(tmp_path, capsys):
    logs = write_logs(tmp_path / "logs", LOGS)
    out = tmp_path / "packed"
    with pytest.raises(SystemExit) as e:
        RP.run("navtrain_train", out, token_df(), logs, workers=1)          # the real pins do not fit this fake data
    assert e.value.code == 1
    assert not out.exists()
    r = json.loads(capsys.readouterr().out)
    assert r["result"] == "FAIL" and any("token_sha16" in x for x in r["mismatch"])
    assert Path(r["kept_for_diagnosis"]).is_dir()


def test_repack_expect_from_catches_a_changed_row_and_a_changed_order(tmp_path, capsys):
    logs = write_logs(tmp_path / "logs", LOGS)
    df = token_df()
    a = RP.build_arrays(df, logs, workers=1)
    pins = RP.array_shas(df.token.tolist(), a)
    old = tmp_path / "old"
    RP.run("navtrain_train", old, df, logs, workers=1, pins=pins, expect_from=None)
    capsys.readouterr()
    assert RP.compare_with_dir(df, a, old) == []
    g = np.load(old / "gt_traj.npy").copy()
    g[2, 0, 0] += 1e-3
    np.save(old / "gt_traj.npy", g)
    assert any("gt_traj: 1 rows differ" in x for x in RP.compare_with_dir(df, a, old))
    np.save(old / "gt_traj.npy", a["gt_traj"])
    t = pd.read_parquet(old / "tokens.parquet")
    t.loc[[0, 1], "token"] = t.loc[[1, 0], "token"].values
    t.to_parquet(old / "tokens.parquet", index=False)
    assert any("column token" in x for x in RP.compare_with_dir(df, a, old))
    out = tmp_path / "new"
    with pytest.raises(SystemExit):
        RP.run("navtrain_train", out, df, logs, workers=1, pins=pins, expect_from=str(old))
    assert not out.exists()


def test_repack_token_source_falls_back_to_the_tracked_copy(tmp_path):
    df = RP.read_tokens("navtrain_val", str(REPO / "resources/km_corridor/packed/navtrain_val/tokens.parquet"))
    assert list(df.columns) == ["token", "log", "city"] and len(df) == 18179
    assert RP.token_sha16(df.token.tolist()) == RP.PINS["navtrain_val"]["token_sha16"]


def test_repack_needs_no_trained_planner_file():
    txt = (SETUP / "repack_dataset.py").read_text()
    for name in ("v2_source", "cand.npy", "dump_v2", "v2dump", "ckpt"):
        assert name not in txt.replace("no planner checkpoint", "")


@real_data
def test_repack_real_log_equals_packed_rows():
    """real: one log through the same function, row by row against the packed directory that is on this server."""
    from navsim.common import ssr_paths as SP
    logs = Path(SP.OPENSCENE_DATA_ROOT) / "navsim_logs/trainval"
    if not logs.is_dir():
        pytest.skip("no logs")
    pk = D_REAL / "ck/packed/navtrain_val"
    t = pd.read_parquet(pk / "tokens.parquet")
    g = t.log.value_counts().index[0]
    sub = t[t.log == g]
    a = RP.build_arrays(sub.reset_index(drop=True), logs, workers=1)
    assert np.array_equal(a["gt_traj"], np.load(pk / "gt_traj.npy", mmap_mode="r")[sub.index])
    assert np.array_equal(a["status"], np.load(pk / "status.npy", mmap_mode="r")[sub.index])


# ----------------------------------------------------------------------------------------------- derive_tokens, parts, raw256
def test_derive_rows_follow_the_filter(tmp_path):
    logs = write_logs(tmp_path / "logs", LOGS)
    sf = dict(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True, log_names=["logA"],
              tokens=LOGS["logA"][:2])
    rows = DT.rows_of_log("logA", sf, str(logs))
    assert [r[0] for r in rows] == LOGS["logA"][:2]
    assert rows[0][1:] == ("logA", 3, "city")


def test_derive_compare_pins_and_tracked_copy(tmp_path):
    df = pd.DataFrame({"token": ["b", "a"], "log": ["x", "x"], "frame_idx": [0, 1], "city": ["c", "c"]})
    order = hashlib.sha256(np.ascontiguousarray(np.array(["b", "a"], "U16")).tobytes()).hexdigest()[:16]
    sorted_sha = hashlib.sha256(b"a\nb").hexdigest()[:16]
    none = tmp_path / "none.parquet"
    assert DT.compare("navtest", df, pins=(2, sorted_sha, order), tracked=none) == []
    assert DT.compare("navtest", df, pins=(3, sorted_sha, order), tracked=none)
    assert DT.compare("navtest", df, pins=(2, "0" * 16, order), tracked=none)
    assert DT.compare("navtest", df.iloc[::-1], pins=(2, sorted_sha, None), tracked=none) == []
    df.iloc[::-1][["token", "log"]].to_parquet(tmp_path / "tr.parquet")
    assert any("order differs" in x for x in DT.compare("navtest", df, pins=(2, None, None), tracked=tmp_path / "tr.parquet"))


def test_derive_filter_uses_the_official_log_split():
    sf, train = DT.filter_of("navtrain_train")
    _, val = DT.filter_of("navtrain_val")
    assert len(train) == 978 and len(val) == 214 and not set(train) & set(val)
    assert train == sorted(train) and len(sf["tokens"]) == 103288
    _, test = DT.filter_of("navtest")
    assert len(test) == 136


def test_derive_refuses_to_overwrite(tmp_path):
    f = tmp_path / "x.parquet"
    f.write_text("x")
    with pytest.raises(SystemExit, match="refusing"):
        DT.main(["--split", "navtest", "--out", str(f)])


@real_data
def test_derive_real_log_equals_the_split_table():
    """real: one log, same rows and order as the table of this server.  No file is written."""
    from navsim.common import ssr_paths as SP
    logs = Path(SP.OPENSCENE_DATA_ROOT) / "navsim_logs/trainval"
    tab = D_REAL / "splits/e2e_train_trainlogs.parquet"
    if not logs.is_dir() or not tab.is_file():
        pytest.skip("no logs")
    sf, all_logs = DT.filter_of("navtrain_train")
    rows = DT.rows_of_log(all_logs[0], sf, str(logs))
    t = pd.read_parquet(tab)
    t = t[t.log == all_logs[0]]
    assert [r[0] for r in rows] == t.token.tolist()
    assert [r[2] for r in rows] == t.frame_idx.tolist() and [r[3] for r in rows] == t.city.tolist()


def test_make_parts_keeps_logs_whole_and_disjoint(tmp_path):
    df = pd.DataFrame({"token": [f"t{i}" for i in range(9)], "log": list("aaaaabbcc")})
    parts = MP.split_parts(df, 2)
    assert sum(len(p) for p in parts) == 9 and sorted(len(p) for p in parts) == [4, 5]
    assert not set(parts[0].log) & set(parts[1].log)
    df.to_parquet(tmp_path / "t.parquet")
    mc = tmp_path / "mc/a/unknown/t0"
    mc.mkdir(parents=True)
    (mc / "metric_cache.pkl").write_text("x")
    assert "t0" not in set(MP.load_tables([str(tmp_path / "t.parquet")], str(tmp_path / "mc")).token)
    assert MP.main(["--tables", str(tmp_path / "t.parquet"), "--k", "3", "--out-prefix", str(tmp_path / "p")]) == 0
    assert sum(len(pd.read_parquet(tmp_path / f"p{i}.parquet")) for i in (1, 2, 3)) == 9


def test_raw256_traj_pins_are_the_tiled_anchors():
    """The recorded traj_sha16 of the raw256 labels = the anchors repeated per token: this is the claim the file rests on."""
    a = np.load(RT.ANCHORS, allow_pickle=False).astype(np.float32)
    for split, (n, pin) in RT.PINS.items():
        assert RT.tiled_sha16(a, n) == pin, split


def test_raw256_write_equals_its_hash_and_refuses_overwrite(tmp_path, monkeypatch):
    a = np.arange(256 * 8 * 3, dtype=np.float32).reshape(256, 8, 3)
    out = tmp_path / "t.npy"
    RT.write_tiled(out, a, 3)
    got = np.load(out)
    assert got.shape == (3, 256, 8, 3) and (got == a).all()
    assert hashlib.sha256(got.tobytes()).hexdigest()[:16] == RT.tiled_sha16(a, 3)
    with pytest.raises(SystemExit, match="refusing"):
        RT.main(["--split", "navtrain_val", "--out", str(out)])


# ----------------------------------------------------------------------------------------------- verify_assets: fake tree
TOK = {"navtrain_train": ["a0000000000000a0", "a0000000000000a1", "b0000000000000b0"], "navtrain_val": ["c0000000000000c0"]}
LOG_OF = {"navtrain_train": ["logA", "logA", "logB"], "navtrain_val": ["logC"]}


class Tree:
    """A small complete tree + the module constants pointing at it; tests then break one thing."""

    def __init__(self, root: Path, monkeypatch):
        self.root, self.mp = root, monkeypatch
        self.data, self.tc = root / "data", root / "tc"
        self.ds, self.test, self.plan = root / "ds", root / "ds_test", root / "plan"
        self.ck = self.data / "ck"
        for k, v in dict(SSR_DATA_ROOT=self.data, SSR_TEACHER_CACHE=self.tc, OPENSCENE_DATA_ROOT=self.ds,
                         NUPLAN_MAPS_ROOT=self.ds / "maps", SSR_TEST_DATA_ROOT=self.test, SSR_PLAN_DATA=self.plan,
                         NAVSIM_V11=root / "v11", NAVSIM_V11_CACHE=root / "v11c", NAVSIM_V22_POST=root / "v22a",
                         NAVSIM_V22_PRE=root / "v22b", NAVSIM_V22_CACHE=root / "v22c", HF_HOME=root / "hf").items():
            monkeypatch.setenv(k, str(v))
        self.pins = {}
        self.build()

    def tokens_file(self):
        for s in TOK:
            p = self.root / "tracked" / s
            p.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"token": TOK[s], "log": LOG_OF[s], "city": "c", "row": range(len(TOK[s]))}).to_parquet(p / "tokens.parquet")
        self.mp.setattr(VA, "TRACKED_TOKENS", str(self.root / "tracked/{split}/tokens.parquet"))

    def build(self):
        self.tokens_file()
        self.mp.setattr(VA, "N_TOKENS", {s: len(TOK[s]) for s in TOK})
        self.mp.setattr(VA, "NAVTRAIN_LOGS", 3)
        self.mp.setattr(VA, "TRAINVAL_LOG_PICKLES", 3)
        self.mp.setattr(VA, "TEST_LOG_PICKLES", 1)
        self.mp.setattr(VA, "NAVTEST_TOKENS", 1)
        self.mp.setattr(VA, "REPO_FILES", {})
        self.mp.setattr(VA, "REPO_FILES_REFINER", {})
        self.mp.setattr(VA, "ENV_PINS", {})
        pins = {}
        # dataset
        logs = self.ds / "navsim_logs/trainval"
        write_logs(logs, {"logA": TOK["navtrain_train"][:2], "logB": TOK["navtrain_train"][2:], "logC": TOK["navtrain_val"]})
        for g in ("logA", "logB", "logC"):
            with open(logs / f"{g}.pkl", "rb") as f:
                for fr in pickle.load(f):
                    for c in VA.CAMERAS:
                        p = self.ds / "sensor_blobs/trainval" / fr["cams"][c]["data_path"]
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(b"")
        (self.ds / "maps").mkdir(parents=True)
        (self.ds / "maps/nuplan-maps-v1.0.json").write_text("{}")
        write_logs(self.test / "navsim_logs/test/test", {"logT": ["t0000000000000t0"]})
        (self.test / "sensor_blobs/test/test/logT").mkdir(parents=True)
        # plan data
        self.plan.mkdir()
        off = self.root / "data/ck/raw256"
        off.mkdir(parents=True, exist_ok=True)
        np.save(off / "pdm_score_256_officialEP.npy", np.ones((3, 5, 256), np.float16))
        (off / "pdm_score_256_officialEP.tokens.json").write_text("[]")
        self.mp.setattr(VA, "OFFICIAL_EP", {"npy": VA.sha16_file(off / "pdm_score_256_officialEP.npy"), "shape": (3, 5, 256),
                                            "tokens": VA.sha16_file(off / "pdm_score_256_officialEP.tokens.json")})
        # eval trees, weights
        for d in ("v11/navsim", "v22a", "v22b", "v22c", "hf/hub/models--timm--resnet34.tv_in1k/snapshots/x",
                  "hf/hub/models--timm--vit_small_patch14_reg4_dinov2.lvd142m/snapshots/x"):
            (self.root / d).mkdir(parents=True)
        c = self.root / "v11c/logT/unknown/t0000000000000t0"
        c.mkdir(parents=True)
        (c / "metric_cache.pkl").write_bytes(b"\xfd7zXZ\x00")
        # refiner inputs
        mc = self.data / "metric_cache"
        (mc / "_state").mkdir(parents=True)
        all_toks = TOK["navtrain_train"] + TOK["navtrain_val"]
        (mc / "_state/done_tokens.txt").write_text("\n".join(all_toks))
        for s in TOK:
            for t, g in zip(TOK[s], LOG_OF[s]):
                d = mc / g / "unknown" / t
                d.mkdir(parents=True)
                (d / "metric_cache.pkl").write_bytes(b"\xfd7zXZ\x00")
        for t in TOK["navtrain_train"]:
            for d, name in ((self.data / "objects/navtrain", f"{t}.npz"), (self.data / "sdf/navtrain", f"{t}.npz")):
                d.mkdir(parents=True, exist_ok=True)
                np.savez(d / name, x=np.zeros(1))
            d = self.data / "e2e_side" / t[:2]
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"{t}.npz", p_pdm=np.float64(1.0), cl_n=np.int32(5))
        for s, n in VA.N_TOKENS.items():
            pk = self.ck / "packed" / s
            pk.mkdir(parents=True)
            pd.DataFrame({"token": TOK[s], "log": LOG_OF[s], "city": "c", "row": range(n)}).to_parquet(pk / "tokens.parquet")
            arrs = {"gt_traj": np.zeros((n, 8, 3), np.float32), "status": np.zeros((n, 8), np.float32), "ok": np.ones(n, bool)}
            pins[f"packed/{s}"] = {"token_sha16": VA.sha16_array(np.array(TOK[s], "U16"))}
            for k, a in arrs.items():
                np.save(pk / f"{k}.npy", a)
                pins[f"packed/{s}"][k] = VA.sha16_array(a)
            (pk / "meta.json").write_text(json.dumps({"source": "dataset"}))
            rd = self.ck / "labels" / s / "raw256"
            rd.mkdir(parents=True)
            pins[f"raw256/{s}"] = {}
            for k, shape in VA.SHAPES["raw256"].items():
                a = np.zeros((n,) + shape, np.float32 if k == "labels" else bool)
                np.save(rd / f"{k}.npy", a)
                pins[f"raw256/{s}"][k] = VA.sha16_array(a)
            vd = self.ck / "ck2/labels" / s / VA.VAR_NAME
            vd.mkdir(parents=True)
            pins[f"var/{s}"] = {}
            for k, shape in VA.SHAPES["var"].items():
                a = np.zeros((n,) + shape, np.float32 if k != "ok" else bool)
                np.save(vd / f"{k}.npy", a)
                pins[f"var/{s}"][k] = VA.sha16_array(a)
            ai, va = np.zeros((n, 16), np.int16), np.ones((n, 16, 6), bool)
            np.savez(vd / "index.npz", anchor_idx=ai, valid=va)
            pins[f"var/{s}"].update(anchor_idx=VA.sha16_array(ai), valid=VA.sha16_array(va))
            (vd / "build.json").write_text(json.dumps({"token_sha16": pins[f"packed/{s}"]["token_sha16"],
                                                       "anchors_sha16": VA.ANCHORS_SHA16, "n_built": n}))
        self.mp.setattr(VA, "PINS", pins)
        self.pins = pins
        # teacher caches
        det = self.tc / "det"
        (det / "samples").mkdir(parents=True)
        (det / "manifest.json").write_text(json.dumps({"checkpoint_sha256_head": "aa", "layout": "samples/<token[:2]>/<token>.npz",
                                                       "target_bev_shape": [2, 2], "bev_channels": 4, "num_samples_written": 3}))
        for t in TOK["navtrain_train"]:
            d = det / "samples" / t[:2]
            d.mkdir(exist_ok=True)
            np.savez(d / f"{t}.npz", bev_feature=np.zeros((4, 2, 2), np.float16))
        mapc = self.tc / "map"
        (mapc / "bev").mkdir(parents=True)
        (mapc / "meta.json").write_text(json.dumps({"checkpoint_sha256": "bb", "num_frames": 3,
                                                    "tensors": {"bev": {"shape": [4, 2, 2], "dtype": "float16"}}}))
        (mapc / "index.json").write_text(json.dumps({t: ["s0", i] for i, t in enumerate(TOK["navtrain_train"])}))
        np.save(mapc / "bev/s0.npy", np.zeros((3, 4, 2, 2), np.float16))
        # refiner teacher runs
        runs = {}
        for arm, norm, name in (("T", "norm.npz", "rT"), ("M", "norm_map.npz", "rM")):
            rd = self.root / "runs" / name
            rd.mkdir(parents=True)
            (rd / "config.json").write_text(json.dumps(dict(arm=arm, trainer="train_ck2", split="navtrain_train", init_from="none")))
            (rd / "done.json").write_text("{}")
            (rd / "ckpt_last.pt").write_bytes(arm.encode())
            (rd / norm).write_bytes(b"x")
            runs[arm] = (rd, VA.sha16_file(rd / "ckpt_last.pt"))
        (self.root / "n.npz").write_bytes(b"x")
        self.registry = self.root / "registry.yaml"
        self.registry.write_text(f"""
teachers:
  det:
    bevfusion: {{status: ready, layout: per_token_npz, cache_train: {det}, cache_val: {det}, sha_head: aa, norm_v15: {self.root}/n.npz, refiner_run: {runs['T'][0]}, refiner_ckpt_sha16: {runs['T'][1]}}}
    bevformer: {{status: missing, needs: [a cache, a teacher]}}
  map:
    resmap: {{status: ready, layout: sharded, cache_root: {mapc}, sha256: bb, norm_v15: {self.root}/n.npz, refiner_run: {runs['M'][0]}, refiner_ckpt_sha16: {runs['M'][1]}}}
""")

    def run(self, *args, profile="full", quick=True):
        argv = ["--profile", profile, "--registry", str(self.registry)] + (["--quick"] if quick else []) + list(args)
        return VA.main(argv)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(VA, "N_SAMPLE", 50)
    monkeypatch.delenv("DET_TEACHER", raising=False)
    monkeypatch.delenv("MAP_TEACHER", raising=False)
    monkeypatch.delenv("SSR_RELEASE_REGISTRY", raising=False)
    return Tree(tmp_path, monkeypatch)


def rows(capsys):
    return capsys.readouterr().out


@pytest.mark.parametrize("profile", VA.PROFILES)
def test_verify_complete_tree_passes(tree, capsys, profile):
    assert tree.run(profile=profile) == 0
    out = rows(capsys)
    assert "FAIL" not in out.replace("0 fail", "")
    assert ("teacher.det.bevfusion" in out) == (profile == "full")
    assert ("metric_cache." in out) == (profile != "base")


def test_verify_full_mode_hashes_arrays(tree, capsys, monkeypatch):
    monkeypatch.setattr(VA, "check_gt_loader", lambda c, fix: c.rep.add("PASS", "gt.loader_all_tokens", "stubbed"))
    assert tree.run(quick=False) == 0
    rows(capsys)
    a = np.load(tree.ck / "labels/navtrain_train/raw256/labels.npy")
    a[0, 0, 0] = 1.0
    np.save(tree.ck / "labels/navtrain_train/raw256/labels.npy", a)
    assert tree.run(quick=True) == 0                # --quick looks at shapes only
    rows(capsys)
    assert tree.run(quick=False) == 1
    out = rows(capsys)
    assert "labels.raw256.navtrain_train" in out and "differs from pin" in out and "50_warmup_labels.sh" in out


@pytest.mark.parametrize("rel, name, fix", [
    ("data/objects/navtrain/a0000000000000a1.npz", "gt.objects", "30_gt_stores.sh"),
    ("data/sdf/navtrain/a0000000000000a1.npz", "gt.sdf", "30_gt_stores.sh"),
    ("data/e2e_side/a0/a0000000000000a1.npz", "gt.e2e_side", "30_gt_stores.sh"),
    ("data/metric_cache/logA/unknown/a0000000000000a1/metric_cache.pkl", "metric_cache.navtrain_train", "20_metric_cache_navtrain.sh"),
    ("data/ck/packed/navtrain_train/status.npy", "packed.navtrain_train", "40_packed_from_dataset.sh"),
    ("data/ck/labels/navtrain_train/raw256/ok.npy", "labels.raw256.navtrain_train", "50_warmup_labels.sh"),
    ("data/ck/ck2/labels/navtrain_train/var_separate_sampler16_accstraight/traj.npy", "labels.var.navtrain_train", "50_warmup_labels.sh"),
    ("tc/det/samples/a0/a0000000000000a1.npz", "teacher.det.bevfusion.cache", "60_teacher_bev.sh"),
    ("tc/map/bev/s0.npy", "teacher.map.resmap.cache", "60_teacher_bev.sh"),
    ("runs/rM/ckpt_last.pt", "teacher.map.resmap.refiner_run", "run_refiner_teachers.sh"),
    ("data/ck/raw256/pdm_score_256_officialEP.tokens.json", "official_ep", "assemble_official_ep.py"),
    ("ds/maps/nuplan-maps-v1.0.json", "dataset.trainval", "10_download_dataset.sh"),
])
def test_verify_partial_copy_fails_and_names_the_fix(tree, capsys, rel, name, fix):
    """The silent-failure class: one missing file of a store."""
    (tree.root / rel).unlink()
    assert tree.run() == 1
    out = rows(capsys)
    line = [i for i, x in enumerate(out.splitlines()) if x.startswith(f"FAIL  {name}")]
    assert line, out
    assert fix in out.splitlines()[line[0] + 1]


def test_verify_does_not_ask_for_the_wote_score_file(tree, capsys):
    """Training reads pdm_score_256_officialEP.  This tree has no pdm_score_256.npy and the row is gone, not failing."""
    assert not (tree.plan / "pdm_score_256.npy").exists()
    tree.run()
    assert not [x for x in rows(capsys).splitlines() if x.split()[1:2] == ["pdm_score_256:"]]


def test_verify_missing_camera_directory_and_image(tree, capsys):
    p = tree.ds / "sensor_blobs/trainval/logA/CAM_L0"
    victim = sorted(p.iterdir())[3]
    victim.unlink()
    assert tree.run(profile="base") in (0, 1)
    tokens_frame = rows(capsys)
    # a frame no token reads may lack its image; the frames of t0 and the one before it may not
    with open(tree.ds / "navsim_logs/trainval/logA.pkl", "rb") as f:
        fr = pickle.load(f)
    i = [k for k, x in enumerate(fr) if x["token"] == TOK["navtrain_train"][0]][0]
    (tree.ds / "sensor_blobs/trainval" / fr[i - 1]["cams"]["CAM_F0"]["data_path"]).unlink()
    assert tree.run(profile="base") == 1
    assert "dataset.trainval" in rows(capsys)
    assert tokens_frame


@pytest.mark.parametrize("field, new, msg", [
    ("sha_head: aa", "sha_head: zz", "checkpoint_sha256_head"),
    ("sha256: bb", "sha256: zz", "checkpoint_sha256"),
])
def test_verify_wrong_teacher_sha_fails(tree, capsys, field, new, msg):
    tree.registry.write_text(tree.registry.read_text().replace(field, new))
    assert tree.run() == 1
    assert msg in rows(capsys)


def test_verify_truncated_bev_file_fails(tree, capsys):
    p = tree.tc / "det/samples/a0/a0000000000000a0.npz"
    p.write_bytes(b"short")
    assert tree.run(quick=True) == 1
    assert "smaller than the bev tensor" in rows(capsys)


def test_verify_refuses_a_teacher_that_is_not_ready_and_lists_what_is_missing(tree, capsys):
    assert tree.run("--det-teacher", "bevformer") == 1
    out = rows(capsys)
    assert "status 'missing'" in out and "a cache" in out and "a teacher" in out
    assert tree.run("--map-teacher", "nosuch") == 1
    assert "not in the registry" in rows(capsys)


def test_verify_a_new_registry_entry_needs_no_code_change(tree, capsys):
    """A teacher under another name, pointing at the same files, passes."""
    txt = tree.registry.read_text().replace("bevfusion:", "other_det:").replace("resmap:", "other_map:")
    tree.registry.write_text(txt)
    assert tree.run("--det-teacher", "other_det", "--map-teacher", "other_map") == 0
    assert "teacher.det.other_det.cache" in rows(capsys)


def test_verify_val_assets_only_warn_and_strict_fails(tree, capsys):
    (tree.ck / "packed/navtrain_val/status.npy").unlink()
    assert tree.run() == 0
    assert "WARN  packed.navtrain_val" in rows(capsys)
    assert tree.run("--strict") == 1


def test_verify_eval_assets_only_warn_for_training(tree, capsys):
    (tree.root / "v11/navsim").rmdir()
    assert tree.run(profile="base") == 0
    assert "WARN  eval.pristine_v11" in rows(capsys)


def test_verify_packed_from_a_planner_dump_warns_when_arrays_equal(tree, capsys):
    (tree.ck / "packed/navtrain_train/meta.json").write_text(json.dumps({"v2_ckpt_sha16": "x", "dump_dir": "y"}))
    assert tree.run() == 0
    assert "WARN  packed.navtrain_train.provenance" in rows(capsys)
    assert tree.run("--strict") == 1
    rows(capsys)
    a = np.load(tree.ck / "packed/navtrain_train/gt_traj.npy")
    a[0, 0, 0] = 1.0
    np.save(tree.ck / "packed/navtrain_train/gt_traj.npy", a)
    assert tree.run() == 1                           # a dump whose values differ does not pass, in either mode
    assert "FAIL  packed.navtrain_train" in rows(capsys)


def test_verify_needs_no_planner_file(tree, capsys):
    """cand.npy, v2_final.npy, labels/*/cand are absent from the fake tree, and the source does not name them."""
    assert tree.run() == 0
    src = (REPO / "tools/release/verify_assets.py").read_text()
    code = src.split('"""', 2)[2]
    for name in ("cand.npy", "v2_final", "v2_im", "/cand", "v2_source"):
        assert name not in code


def test_verify_only_runs_one_group_and_json_records_rows(tree, capsys, tmp_path):
    (tree.data / "sdf/navtrain/a0000000000000a1.npz").unlink()
    j = tmp_path / "r.json"
    assert tree.run("--only", "labels", "--json", str(j)) == 0
    d = json.loads(j.read_text())
    assert d["exit"] == 0 and d["profile"] == "full" and all(r["name"].startswith("labels.") for r in d["rows"])
    capsys.readouterr()
    assert tree.run("--only", "gt") == 1


def test_verify_exception_in_a_check_is_a_fail_not_a_crash(tree, capsys):
    (tree.tc / "det/manifest.json").write_text("not json")
    assert tree.run() == 1
    assert "JSONDecodeError" in rows(capsys)


def test_verify_writes_nothing(tree, capsys):
    before = sorted(str(p) for p in tree.root.rglob("*"))
    tree.run()
    assert sorted(str(p) for p in tree.root.rglob("*")) == before


def test_gt_stats_compare_to_expected():
    rows_ = [(True, 800, 67.8193436381719, 5, 10.0), (True, 0, 0.0, 3, 1.0)]
    s = VA.gt_stats(rows_)
    want = dict(gt_n=2, gt_ok=2, p_pdm_nonfinite=0, cl_n_lt2=0, sdf_absmax_zero=0, sdf_absmax_max=10.0, obj_n_max=800, obj_n_zero=1,
                p_pdm_zero=1, p_pdm_max=67.8193436381719)
    assert VA.compare_gt(s, want) == []
    assert VA.compare_gt(s, dict(want, gt_ok=3))
    rows_[0] = (False, 800, float("nan"), 1, 10.0)
    s = VA.gt_stats(rows_)
    assert s["gt_ok"] == 1 and s["p_pdm_nonfinite"] == 1 and s["cl_n_lt2"] == 1


def test_gt_expected_numbers_are_the_runbook_values():
    e = VA.GT_EXPECT
    assert (e["gt_n"], e["gt_ok"], e["p_pdm_zero"], e["obj_n_zero"], e["obj_n_max"]) == (85109, 85109, 4370, 68, 800)
    assert round(e["p_pdm_max"], 2) == 67.82 and e["sdf_absmax_max"] == 10.0


# ----------------------------------------------------------------------------------------------- teacher env
def test_teacher_env_reads_the_registry_and_refuses_missing():
    reg = {"teachers": {"det": {"d": dict(status="ready", cache_train="/a b", cache_val="/v", sha_head="h", refiner_run="/r/T"),
                                "x": dict(status="missing", needs=["n"])},
                        "map": {"m": dict(status="ready", cache_root="/m", sha256="s", refiner_run="/r/M", require_seg=False)}}}
    lines = TE.export_lines(reg, "d", "m")
    assert "export SSR_DET_CACHE_TRAIN='/a b'" in lines and "export MAP_RUN=/r/M" in lines
    assert "export SSR_MAP_REQUIRE_SEG=0" in lines and "export SSR_DET_SHA_HEAD=h" in lines
    with pytest.raises(SystemExit, match="missing"):
        TE.export_lines(reg, "x", "m")


def test_teacher_env_real_registry_matches_verify_defaults(monkeypatch):
    from navsim.common import ssr_paths as SP
    reg = VA.load_registry(REPO / VA.REGISTRY, SP.resolve())
    assert reg is not None
    assert re.fullmatch("[0-9a-f]{16}", reg["teachers"]["det"]["bevfusion"]["refiner_ckpt_sha16"])
    assert re.fullmatch("[0-9a-f]{16}", reg["teachers"]["map"]["resmap"]["refiner_ckpt_sha16"])
    assert "${" not in json.dumps(reg["teachers"]["det"]["bevfusion"])
    for line in TE.export_lines(reg, "bevfusion", "resmap"):
        assert line.startswith("export ")


# ----------------------------------------------------------------------------------------------- setup scripts
def script_env(tmp_path, download=False, **kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SSR_", "NAVSIM_", "OPENSCENE", "NUPLAN", "PARTS", "SPLITS"))}
    root = tmp_path / "srv"
    # A dataset root without log pickles would be replaced by the default root (ssr_paths) on a server that has one.  The
    # download script gets its own link already in place, so it can run with its real target.
    logs = root / "ds/_download/trainval_navsim_logs/trainval" if download else root / "ds/navsim_logs/trainval"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "x.pkl").write_bytes(b"")
    if download:
        (root / "ds/navsim_logs").mkdir(parents=True, exist_ok=True)
        if not (root / "ds/navsim_logs/trainval").exists():
            (root / "ds/navsim_logs/trainval").symlink_to(logs)
    env.update(SSR_DATA_ROOT=str(root / "data"), SSR_TEACHER_CACHE=str(root / "tc"), OPENSCENE_DATA_ROOT=str(root / "ds"),
               SSR_TEST_DATA_ROOT=str(root / "dst"), SSR_PLAN_DATA=str(root / "plan"), SSR_NAVTEST_MC=str(root / "nmc"),
               SSR_PY=sys.executable, SSR_TORCHRUN="/bin/echo", SSR_RUN_ROOT=str(root / "runs"), NAVSIM_V11=str(root / "v11"),
               NAVSIM_V11_CACHE=str(root / "v11c"), NAVSIM_V22_POST=str(root / "a"), NAVSIM_V22_PRE=str(root / "b"),
               NAVSIM_V22_CACHE=str(root / "c"), SETUP_DRY="1", PYTHONDONTWRITEBYTECODE="1")
    env.update(kw)
    return env


def run_script(name, env, *args):
    return subprocess.run(["bash", str(SETUP / name)] + list(args), env=env, capture_output=True, text=True)


SCRIPTS = sorted(p.name for p in SETUP.glob("*.sh") if p.name != "_common.sh")


def test_contract_scripts_exist():
    for n in ("install_env.sh", "00_dirs_and_env.sh", "10_download_dataset.sh", "20_metric_cache_navtrain.sh", "30_gt_stores.sh",
              "40_packed_from_dataset.sh", "50_warmup_labels.sh", "60_teacher_bev.sh", "run_refiner_teachers.sh"):
        assert n in SCRIPTS
    for n in ("constraints.txt", "pip_freeze_ssr.txt", "repack_dataset.py"):
        assert (SETUP / n).is_file()


@pytest.mark.parametrize("name", SCRIPTS + ["_common.sh"])
def test_script_syntax_and_no_hard_coded_root(name):
    p = SETUP / name
    assert subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode == 0
    code = "\n".join(x for x in p.read_text().splitlines() if not x.lstrip().startswith("#"))
    for bad in ("/home/external-user", "/workspace/", "/venv/", "miniconda3", "/tmp/claude", "pkill", "pgrep"):
        assert bad not in code, f"{name}: {bad}"


@pytest.mark.parametrize("name", [n for n in SCRIPTS if n != "install_env.sh"])
def test_dry_run_only_uses_paths_from_env(tmp_path, name):
    env = script_env(tmp_path, download=name.startswith("10_"), BEVFUSION_SRC="host:/src/")
    before = sorted(str(p) for p in (tmp_path / "srv").rglob("*"))
    r = run_script(name, env)
    out = r.stdout + r.stderr
    # 70 needs evallock's scripts, which exist; every script reaches its end under SETUP_DRY=1
    assert r.returncode == 0, out
    for bad in ("/home/external-user", "/workspace/navsim", "/venv/ssr"):
        assert bad not in out.replace(str(Path(sys.executable).parent), "PYBIN").replace(str(REPO), "REPO"), f"{name}: {bad}\n{out}"
    assert sorted(str(p) for p in (tmp_path / "srv").rglob("*")) == before, f"{name} wrote under SETUP_DRY=1"


def _grid(out):
    return [x for x in out.splitlines() if x.startswith("+ ")]


def test_download_lists_only_what_is_needed(tmp_path):
    out = run_script("10_download_dataset.sh", script_env(tmp_path, download=True)).stdout
    w = [x for x in _grid(out) if x.startswith("+ wget")]
    assert sum("navtrain_current_" in x for x in w) == 32 and sum("navtrain_history_" in x for x in w) == 32
    assert sum("openscene_metadata_trainval" in x for x in w) == 1 and sum("openscene_metadata_test" in x for x in w) == 1
    assert sum("openscene_sensor_test_camera_" in x for x in w) == 32 and sum("nuplan-maps" in x for x in w) == 1
    assert not any("lidar" in x or "navhard" in x or "trainval_sensor" in x for x in w)
    out = run_script("10_download_dataset.sh", script_env(tmp_path, download=True, PARTS="test_lidar navhard")).stdout
    w = [x for x in _grid(out) if x.startswith("+ wget")]
    assert sum("lidar" in x for x in w) == 32 and sum("navhard" in x for x in w) == 3
    r = run_script("10_download_dataset.sh", script_env(tmp_path, download=True, PARTS="nonsense"))
    assert r.returncode == 3 and "unknown part" in r.stderr


def test_download_links_give_the_layout_of_the_code(tmp_path):
    env = script_env(tmp_path, download=True)
    out = run_script("10_download_dataset.sh", env).stdout
    dl = Path(env["OPENSCENE_DATA_ROOT"]) / "_download"
    links = [x for x in _grid(out) if x.startswith("+ ln -s")]
    want = {(f"{dl}/trainval_sensor_blobs/trainval", f"{env['OPENSCENE_DATA_ROOT']}/sensor_blobs/trainval"),
            (f"{dl}/maps", f"{env['OPENSCENE_DATA_ROOT']}/maps"),
            (f"{dl}/test_navsim_logs/test", f"{env['SSR_TEST_DATA_ROOT']}/navsim_logs/test/test"),
            (f"{dl}/test_sensor_blobs/test", f"{env['SSR_TEST_DATA_ROOT']}/sensor_blobs/test/test")}
    assert {tuple(x.split()[3:5]) for x in links} == want    # navsim_logs/trainval is the link already in place


def test_download_refuses_to_replace_a_directory_or_another_link(tmp_path):
    env = script_env(tmp_path, download=True)
    p = Path(env["OPENSCENE_DATA_ROOT"]) / "sensor_blobs/trainval"
    p.mkdir(parents=True)
    r = run_script("10_download_dataset.sh", env)
    assert r.returncode == 3 and "refusing to replace" in r.stderr and p.is_dir()
    p.rmdir()
    p.symlink_to(tmp_path)
    r = run_script("10_download_dataset.sh", env)
    assert r.returncode == 3 and "expected" in r.stderr


@pytest.mark.parametrize("name, rel", [
    ("40_packed_from_dataset.sh", "data/ck/packed/navtrain_train"),
    ("15_token_lists.sh", "data/splits/e2e_train_trainlogs.parquet"),
    ("20_metric_cache_navtrain.sh", "data/metric_cache/_state/done_tokens.txt"),
    ("30_gt_stores.sh", "data/sdf/navtrain"),
    ("50_warmup_labels.sh", "data/ck/labels/navtrain_train/raw256/labels.npy"),
    ("50_warmup_labels.sh", "data/ck/raw256/traj_navtrain_train.npy"),
    ("60_teacher_bev.sh", "tc/resmap/stage1_front64/meta.json"),
    ("60_teacher_bev.sh", "tc/bevfusion/cache_train_100x100/manifest.json"),
    ("20_metric_cache_navtest_fork.sh", "nmc"),
])
def test_scripts_refuse_to_overwrite(tmp_path, name, rel):
    env = script_env(tmp_path, BEVFUSION_SRC="host:/src/")
    p = tmp_path / "srv" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix:
        p.write_text("x")
    else:
        p.mkdir(exist_ok=True)
    r = run_script(name, env)
    assert r.returncode == 3, r.stdout + r.stderr
    assert str(p) in r.stderr and ("overwrite" in r.stderr or "RESUME=1" in r.stderr)
    assert p.exists()


def test_resume_goes_on_over_partial_output_but_not_a_finished_teacher(tmp_path):
    env = script_env(tmp_path)
    done = tmp_path / "srv/data/metric_cache/_state/done_tokens.txt"
    done.parent.mkdir(parents=True)
    done.write_text("x")
    assert run_script("20_metric_cache_navtrain.sh", env).returncode == 3
    assert run_script("20_metric_cache_navtrain.sh", dict(env, RESUME="1")).returncode == 0


def test_install_env_refuses_an_interpreter_that_has_torch(tmp_path):
    r = run_script("install_env.sh", script_env(tmp_path))
    assert r.returncode == 3 and "already has torch" in r.stderr


def test_install_env_bootstraps_before_the_interpreter_exists(tmp_path):
    env = script_env(tmp_path, SSR_PY=str(tmp_path / "newenv/bin/python"))
    r = run_script("install_env.sh", env)
    assert r.returncode == 0, r.stderr
    assert f"create -y -p {tmp_path}/newenv" in r.stdout and "torch==2.8.0" in r.stdout and "--no-deps" in r.stdout
    assert run_script("40_packed_from_dataset.sh", env).returncode == 3     # other steps still need the interpreter


def test_metric_cache_recipe_lines(tmp_path):
    out = run_script("20_metric_cache_navtrain.sh", script_env(tmp_path, K="3")).stdout
    b = [x for x in _grid(out) if "build_metric_cache.py" in x]
    assert len(b) == 3 and all("--workers 4" in x and "--no-final-check" in x and "--no-seed" in x for x in b)
    assert all("CUDA_VISIBLE_DEVICES=" in x for x in b)
    assert "build_metric_cache.py check" not in out


def test_gt_stores_recipe_lines(tmp_path):
    out = run_script("30_gt_stores.sh", script_env(tmp_path, K="2")).stdout
    o = [x for x in _grid(out) if "build_future_objects" in x]
    assert len(o) == 1 and "--mc-root" not in o[0] and "/objects/navtrain" in o[0]
    assert len([x for x in _grid(out) if "build_sdf.py" in x and "--subset navtrain" in x]) == 2
    assert len([x for x in _grid(out) if "build_e2e_side.py build" in x]) == 1
    assert any("verify_assets.py --profile refiner_noteacher --only gt" in x for x in _grid(out))


def test_warmup_recipe_lines(tmp_path):
    out = run_script("50_warmup_labels.sh", script_env(tmp_path, SPLITS="navtrain_train")).stdout
    g = _grid(out)
    assert any("make_raw256_traj.py --split navtrain_train" in x for x in g)
    assert any("label_cands.py --split navtrain_train --traj" in x and "--name raw256 --workers 48" in x for x in g)
    v = [x for x in g if "label_variants.py" in x]
    assert len(v) == 1 and "--combine separate --ext straight --name var_separate_sampler16_accstraight --workers 48" in v[0]
    assert not any("assemble_official_ep.py" in x for x in g) and "copy pdm_score_256_officialEP.npy" in out
    both = _grid(run_script("50_warmup_labels.sh", script_env(tmp_path, SPLITS="navtrain_train navtrain_val")).stdout)
    assert any("assemble_official_ep.py" in x for x in both)


def test_resmap_download_is_pinned_to_a_revision_and_reads_only_bev(tmp_path):
    out = run_script("60_teacher_bev.sh", script_env(tmp_path, TEACHERS="resmap")).stdout
    x = [y for y in _grid(out) if " download " in y][0]
    assert "--revision 73db598669a8e6edb2b585fcc14ebe54b01b15f5" in x and "rudals/resmap-navsim-teacher-kd" in x
    assert 'stage1_front64/bev/*' in x and 'seg' not in x and "navtest" not in x
    r = run_script("60_teacher_bev.sh", dict(script_env(tmp_path, TEACHERS="bevfusion"), SETUP_DRY="0"))
    assert r.returncode == 3 and "BEVFUSION_SRC" in r.stderr
    assert not (tmp_path / "srv/tc/bevfusion/cache_train_100x100").exists()


def teacher_cmds(tmp_path, **kw):
    r = run_script("run_refiner_teachers.sh", script_env(tmp_path, **kw))
    assert r.returncode == 0, r.stderr
    return {("T" if "--arm T" in x else "M"): x for x in _grid(r.stdout) if "train_ck2.py" in x}


def test_teacher_recipe_is_the_recorded_one_with_r34_evals_off(tmp_path):
    c = teacher_cmds(tmp_path)
    assert set(c) == {"T", "M"}
    for arm, x in c.items():
        for flag in ("--epochs 10", "--tokens-per-batch 16", "--lr 6e-4", "--sur-groups near_mid", "--val-split navtrain_val",
                     "--eval-every 2", "--best-metric none", "--r34-evals 0", "--resume", "--nproc_per_node=4"):
            assert flag in x, (arm, flag)
    assert "--score-keys nc,ttc,ep,comfort" in c["T"] and "CUDA_VISIBLE_DEVICES=0,1,2,3" in c["T"]
    assert "--score-keys dac,ep" in c["M"] and "CUDA_VISIBLE_DEVICES=4,5,6,7" in c["M"]
    assert "--run-name det" in c["T"] and "--run-name map" in c["M"] and "/refiner " in c["T"] and "/refiner " in c["M"]


D_TEACHERS = Path(os.environ.get("SSR_TEACHER_CACHE") or "/home/external-user/datasets/teacher_cache") / "refiner"


@pytest.mark.skipif(not (D_TEACHERS / "det/config.json").is_file(), reason="no teacher run")
def test_teacher_recipe_equals_the_config_of_the_reference_runs(tmp_path):
    """real: every flag of the script equals the value in the reference runs' config.json."""
    c = teacher_cmds(tmp_path)
    for arm, run, keys in (("T", "det", "nc,ttc,ep,comfort"), ("M", "map", "dac,ep")):
        cfg = json.loads((D_TEACHERS / run / "config.json").read_text())
        x = c[arm]
        assert f"--epochs {cfg['epochs']}" in x and f"--tokens-per-batch {cfg['global_batch']}" in x
        assert float(re.search(r"--lr (\S+)", x).group(1)) == cfg["lr"]
        assert f"--sur-groups {cfg['sur_groups']}" in x and f"--val-split {cfg['val_split']}" in x
        assert f"--eval-every {cfg['eval_every']}" in x and f"--best-metric {cfg['best_metric']}" in x
        assert f"--nproc_per_node={cfg['world_size']}" in x
        assert f"--score-keys {keys}" in x
        if "score_keys" in cfg:
            sk = cfg["score_keys"]
            assert set(keys.split(",")) == set(sk.split(",") if isinstance(sk, str) else sk)
        # flags left out of the script are at their defaults in the reference config
        assert cfg["init_from"] == "none" and cfg["seed"] == 0 and cfg["n_var"] == 80 and cfg["var_name"] == "var_separate_sampler16_accstraight"


def test_teacher_script_guards(tmp_path):
    env = script_env(tmp_path)
    run = Path(env["SSR_TEACHER_CACHE"]) / "refiner/det"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}")
    r = run_script("run_refiner_teachers.sh", env)
    assert r.returncode == 3 and "RESUME=1" in r.stderr
    assert run_script("run_refiner_teachers.sh", dict(env, RESUME="1")).returncode == 0
    (run / "done.json").write_text("{}")
    r = run_script("run_refiner_teachers.sh", dict(env, RESUME="1"))
    assert r.returncode == 3 and "done.json" in r.stderr
    r = run_script("run_refiner_teachers.sh", dict(env, WHICH="map", GPUS_MAP="0,1,2"))
    assert r.returncode == 3 and "2 or 4" in r.stderr
    assert "--arm T" not in run_script("run_refiner_teachers.sh", dict(env, WHICH="map")).stdout
    r = run_script("run_refiner_teachers.sh", dict(env, WHICH="map", DET_TEACHER="bevformer"))
    assert r.returncode == 3 and "bevformer" in r.stderr


def test_teacher_script_reads_a_registry_entry_for_a_new_teacher(tmp_path):
    reg = tmp_path / "reg.yaml"
    reg.write_text("""teachers:
  det:
    newdet: {status: ready, layout: per_token_npz, cache_train: /c/t, cache_val: /c/v, sha_head: h, refiner_run: /o/newT}
  map:
    resmap: {status: ready, layout: sharded, cache_root: /m, sha256: s, refiner_run: /o/newM}
""")
    c = teacher_cmds(tmp_path, SSR_RELEASE_REGISTRY=str(reg), DET_TEACHER="newdet")
    assert "--run-name newT --out-root /o" in c["T"]


# ----------------------------------------------------------------------------------------------- lock, docs, ignore
def test_lock_conversion():
    v = {"anyio": "4.0"}.get
    got = ML.convert(["anyio @ file:///home/conda/x/work", "-e git+https://x#egg=y", "a==1", "",
                      "n @ git+https://github.com/x/y.git@abc"], v)
    assert got == ["anyio==4.0", "a==1", "n @ git+https://github.com/x/y.git@abc"]


def test_lock_file_is_usable():
    code = [x for x in (REPO / "requirements.lock.txt").read_text().splitlines() if x and not x.startswith("#")]
    assert code[0].startswith("--extra-index-url")
    for x in code[1:]:
        assert "file://" not in x and not x.startswith("-e") and ("==" in x or " @ git+https://" in x), x
    d = dict(x.split("==", 1) for x in code[1:] if "==" in x)
    assert d["torch"] == "2.8.0+cu128" and d["pytorch-lightning"] == "2.2.1" and d["numpy"] == "1.23.4"
    assert any(x.startswith("nuplan-devkit @ git+") and "ce3c323" in x for x in code)
    for bad in ("/home/", "/workspace", "/venv"):
        assert not any(bad in x for x in code), bad


def test_paths_env_is_ignored_and_setup_files_are_not():
    def ignored(p):
        return subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", p]).returncode == 0
    if subprocess.run(["git", "-C", str(REPO), "rev-parse", "--git-dir"], capture_output=True).returncode != 0:
        pytest.skip("no git")
    assert ignored("tools/release/paths.env")
    for p in ("tools/release/setup/repack_dataset.py", "requirements.lock.txt", "docs/SETUP.md", "docs/FAIRNESS.md", "THIRD_PARTY.md",
              "tools/release/setup/constraints.txt", "tools/release/setup/pip_freeze_ssr.txt"):
        assert not ignored(p), p


def test_setup_doc_names_every_script_and_profile():
    t = (REPO / "docs/SETUP.md").read_text()
    for n in SCRIPTS + ["repack_dataset.py", "derive_tokens.py", "make_raw256_traj.py", "verify_assets.py"]:
        assert n in t, n
    for p in VA.PROFILES:
        assert p in t, p
    assert "--r34-evals 0" in t and "85,109" in t


def test_docs_exist_and_keep_upstream_credit():
    r = (REPO / "README.md").read_text()
    head = r[:r.index("Navigation-Guided Sparse Scene Representation")]
    for x in ("docs/SETUP.md", "docs/EXPERIMENTS.md", "docs/FAIRNESS.md", "THIRD_PARTY.md"):
        assert x in head, x
    assert "Apache License 2.0" in r and "li2025navigationguidedsparsescenerepresentation" in r
    assert "torch.distributed.run" not in r            # the upstream nuScenes commands are gone
    t = (REPO / "THIRD_PARTY.md").read_text()
    for x in ("NAVSIM", "nuPlan", "WoTE", "ReSMap", "cc-by-nc-4.0", "BEVFusion"):
        assert x in t, x
    f = (REPO / "docs/FAIRNESS.md").read_text()
    for x in ("BN", "GPU", "v1.0", "navtest"):
        assert x in f, x


NEW_FILES = ["docs/SETUP.md", "docs/FAIRNESS.md", "THIRD_PARTY.md", "tools/release/verify_assets.py", "tests/test_release_setup.py"] + \
    [f"tools/release/setup/{p.name}" for p in SETUP.iterdir() if p.suffix in (".py", ".sh")]


@pytest.mark.parametrize("rel", NEW_FILES)
def test_new_files_name_no_person_and_no_session(rel):
    t = (REPO / rel).read_text().lower()
    if rel.endswith("test_release_setup.py"):
        t = t.split("new_files = ", 1)[0]
    for bad in ("byounggun", "kyungmin", "/home/yongjae", "claude.ai", "session_", "2026-1"):
        assert bad not in t, (rel, bad)
