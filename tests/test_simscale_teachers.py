"""Camera-teacher caches for trainval + SimScale: tools/release/teachers/ and the setup steps 60 / 65.

CPU only.  Everything runs on fake trees under tmp_path; the setup scripts run with SETUP_DRY=1 in a copy of the
release tree without tools/release/paths.env (a server's paths.env would replace the fake roots).  Tests marked "real"
read this server's teacher clones, the training-server archive or the dev caches when present and are skipped
elsewhere.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
TEACH = REPO / "tools/release/teachers"
SETUP = REPO / "tools/release/setup"
sys.path.insert(0, str(REPO))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


NC = _load("navsim_cache_teacher_t", TEACH / "navsim_cache_teacher.py")
TCS = _load("teacher_caches_t", TEACH / "teacher_caches.py")
from navsim.common import simscale as SS  # noqa: E402

ARCHIVE = Path(os.environ.get("SSR_VAST2_ARCHIVE", "/home/external-user/yongjae/ssr-vast2-archive/code/uncommitted"))
TEACHER_SRC = Path(os.environ.get("TEACHER_SRC", "/home/external-user/ssd/simscale_dev/teachers"))
M2M_PIN, BEV_PIN = "a7f07a9b88c1a419fbeb86a3fb8b868b0516571f", "9f1d5df246bfb14699b4d6f26c96ba1aa25dde4d"
has_archive = pytest.mark.skipif(not (ARCHIVE / "Mask2Map_navsim.diff").is_file(), reason="no training-server archive")
has_clones = pytest.mark.skipif(not (TEACHER_SRC / "Mask2Map_navsim/.git").is_dir(), reason="no teacher clones")
DET_SHA, MAP_SHA = "4ca4f1aa55011beb", "579665a7309beb17621dcbad511031776015f0a58c65a2beee1ee58ed45771e4"


def patch_new_file(patch: Path, path: str) -> str:
    """Content of a file the patch creates (git diff --binary of an added text file)."""
    txt = patch.read_text()
    m = re.search(rf"^diff --git a/{re.escape(path)} b/{re.escape(path)}\nnew file mode \d+\nindex \S+\n--- /dev/null\n"
                  rf"\+\+\+ b/{re.escape(path)}\n@@ [^\n]*@@\n(.*?)(?=^diff --git |\Z)", txt, re.S | re.M)
    assert m, f"{path} is not a new file of {patch.name}"
    lines = m.group(1).split("\n")
    body = [ln[1:] for ln in lines if ln.startswith("+")]
    return "\n".join(body) + ("\n" if not txt.endswith("\\ No newline at end of file\n") else "")


@pytest.fixture(scope="module")
def m2m_mod(tmp_path_factory):
    """Mask2Map's NAVSIM helpers, taken from the patch (no checkout needed)."""
    d = tmp_path_factory.mktemp("m2m")
    patch = TEACH / "patches/mask2map_navsim.patch"
    mods = {}
    for name, rel in (("utils", "projects/mmdet3d_plugin/datasets/navsim_map_utils.py"),
                      ("infos", "tools/navsim/create_navsim_infos.py"),
                      ("dataset", "projects/mmdet3d_plugin/datasets/navsim_offlinemap_dataset.py")):
        p = d / Path(rel).name
        p.write_text(patch_new_file(patch, rel))
        mods[name] = p
    return mods


# ----------------------------------------------------------------------------------------------- BEVFormer writer
def _pool(rng, n_logs=6):
    """Fake frame pool: logs of random length with prev-chain breaks -> (segment strings, logs)."""
    segs, logs = [], []
    for g in range(n_logs):
        k = 0
        for _ in range(int(rng.integers(5, 40))):
            if rng.random() < 0.15:
                k += 1
            segs.append(f"log{g}#s{g * 100 + k}")
            logs.append(f"log{g}")
    return segs, logs


def _toy_run(frames, segs, feats):
    """forward_test's temporal rule on a toy recurrence: prev reset whenever the segment string changes."""
    out, prev, cur = {}, None, None
    for i in frames:
        if segs[i] != cur:
            prev, cur = None, segs[i]
        x = feats[i] + (0.5 * prev if prev is not None else 0.0)
        out[i] = x
        prev = x
    return out


def test_segment_runs():
    assert NC.segment_runs(["a", "a", "b", "c", "c", "c"]) == [(0, 2), (2, 3), (3, 6)]
    assert NC.segment_runs([]) == []


def test_plan_runs_a_segment_from_its_start_to_its_last_target():
    segs = ["L#0"] * 5 + ["L#1"] * 4 + ["M#2"] * 3
    logs = ["L"] * 9 + ["M"] * 3
    plan = NC.plan_segments(segs, logs, [2, 3, 7, 7])
    assert plan == [("L", 0, 4, [2, 3]), ("L", 5, 8, [7])]         # no target in M#2: not run; frame 8 not run


@pytest.mark.parametrize("world", [1, 2, 3])
def test_target_subset_with_chain_context_equals_the_full_pool_run(world):
    rng = np.random.default_rng(world)
    segs, logs = _pool(rng)
    feats = rng.normal(size=len(segs))
    full = _toy_run(range(len(segs)), segs, feats)
    targets = sorted(rng.choice(len(segs), 25, replace=False).tolist())
    plan = NC.plan_segments(segs, logs, targets)
    owner = NC.shard_logs(plan, world)
    done = set(targets[::3])                                           # a resumed pass: some targets written already
    got = {}
    for r in range(world):
        todo = NC.rank_plan(plan, owner, r, set(targets) - done)
        frames = [i for _, s0, s1, _ in todo for i in range(s0, s1)]
        out = _toy_run(frames, segs, feats)
        for _, _, _, tg in todo:
            for t in tg:
                assert t not in got
                got[t] = out[t]
    assert set(got) == set(targets) - done
    for t, v in got.items():
        assert v == full[t]                                            # bitwise: same operations in the same order


def test_shards_are_whole_logs_balanced_and_independent_of_what_is_written():
    rng = np.random.default_rng(0)
    segs, logs = _pool(rng, n_logs=20)
    targets = sorted(rng.choice(len(segs), 120, replace=False).tolist())
    plan = NC.plan_segments(segs, logs, targets)
    a, b = NC.shard_logs(plan, 4), NC.shard_logs(list(reversed(plan)), 4)
    assert a == b                                                      # every rank computes the same owners
    assert set(a) == {p[0] for p in plan}
    load = [sum(s1 - s0 for g, s0, s1, _ in plan if a[g] == r) for r in range(4)]
    biggest = max(sum(s1 - s0 for g2, s0, s1, _ in plan if g2 == g) for g in a)
    assert max(load) - min(load) <= biggest


def test_bev_from_embed_is_reshape_y_x_then_permute_to_c_x_y():
    H, W = 100, 100
    h, w = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    embed = np.stack([h.reshape(-1), w.reshape(-1)], -1).astype(np.float32)[:, None, :]   # query q = h * W + w
    out = NC.bev_from_embed(embed, H, W)
    assert out.shape == (2, W, H) and out.dtype == np.float16
    xi, yi = 37, 81
    assert out[0, xi, yi] == yi and out[1, xi, yi] == xi              # (C, x = w, y = h)
    assert np.array_equal(out[0], np.broadcast_to(np.arange(H), (W, H)))


def test_box_arrays_top_k_by_score_gravity_centre_and_padding():
    rng = np.random.default_rng(0)
    n = 300
    c, rest = rng.normal(size=(n, 3)), rng.normal(size=(n, 6))
    s, lab = rng.random(n), rng.integers(0, 7, n)
    b, sc, lb = NC.box_arrays(c, rest, s, lab)
    order = np.argsort(-s, kind="stable")[:200]
    assert b.shape == (200, 9) and b.dtype == np.float32 and sc.dtype == np.float32 and lb.dtype == np.int16
    assert np.all(np.diff(sc) <= 0) and np.allclose(sc, s[order]) and np.array_equal(lb, lab[order])
    assert np.allclose(b[:, :3], c[order]) and np.allclose(b[:, 3:], rest[order])
    b, sc, lb = NC.box_arrays(c[:5], rest[:5], s[:5], lab[:5])
    assert np.all(b[5:] == 0) and np.all(sc[5:] == 0) and np.all(lb[5:] == -1) and np.all(lb[:5] >= 0)


def test_npz_is_uncompressed_atomic_and_checked(tmp_path):
    tok = "0123456789abcdef-004"
    p = NC.npz_path(str(tmp_path), tok)
    assert p.endswith(f"samples/01/{tok}.npz")
    b, s, lab = NC.box_arrays(np.zeros((3, 3)), np.zeros((3, 6)), np.ones(3), np.zeros(3))
    NC.write_npz(p, dict(bev_feature=np.ones(NC.NPZ_MIN_BYTES // 2, np.float16).reshape(256, 100, 100),
                         pred_boxes_3d=b, pred_scores_3d=s, pred_labels_3d=lab))
    assert NC.npz_complete(p) and os.path.getsize(p) >= NC.NPZ_MIN_BYTES
    assert [f.name for f in Path(p).parent.iterdir()] == [Path(p).name]   # no temporary left
    with np.load(p, allow_pickle=False) as z:
        assert sorted(z.files) == sorted(NC.NPZ_KEYS) and z["bev_feature"].dtype == np.float16
    short = tmp_path / "short.npz"
    short.write_bytes(Path(p).read_bytes()[: NC.NPZ_MIN_BYTES - 1])
    assert not NC.npz_complete(str(short))
    assert not NC.npz_complete(str(tmp_path / "absent.npz"))


def test_bevformer_patch_carries_the_reviewed_writer():
    patch = TEACH / "patches/bevformer_navsim.patch"
    assert patch_new_file(patch, "tools/navsim_cache_teacher.py") == (TEACH / "navsim_cache_teacher.py").read_text()
    files = re.findall(r"^diff --git a/(\S+) ", patch.read_text(), re.M)
    assert files == ["tools/navsim_cache_teacher.py", "tools/test.py", "tools/train.py"]


@has_archive
def test_bevformer_patch_holds_the_training_server_diff():
    """real: the tools/{test,train}.py hunks of the patch are the archive's uncommitted BEVFormer.diff."""
    def hunks(txt, f):
        m = re.search(rf"^diff --git a/{re.escape(f)} .*?(?=^diff --git |\Z)", txt, re.S | re.M)
        return [ln for ln in m.group(0).splitlines() if ln[:1] in "+-@" and not ln.startswith(("+++", "---"))]
    ours, theirs = (TEACH / "patches/bevformer_navsim.patch").read_text(), (ARCHIVE / "BEVFormer.diff").read_text()
    for f in ("tools/test.py", "tools/train.py"):
        assert hunks(ours, f) == hunks(theirs, f)


# ----------------------------------------------------------------------------------------------- patches / clones
@has_archive
@has_clones
def test_mask2map_patch_rebuilds_the_training_server_tree(tmp_path):
    """real: a7f07a9 + patches/mask2map_navsim.patch == a7f07a9 + the archive's uncommitted diff + untracked files
    (file contents and modes), built from the pinned commit of the local clone (no network)."""
    def tree(name):
        d = tmp_path / name
        d.mkdir()
        tar = subprocess.run(["git", "-C", str(TEACHER_SRC / "Mask2Map_navsim"), "archive", M2M_PIN],
                             capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(d)], input=tar, check=True)
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        return d

    ref, got = tree("ref"), tree("got")
    subprocess.run(["git", "-C", str(ref), "apply", str(ARCHIVE / "Mask2Map_navsim.diff")], check=True)
    (tmp_path / "untracked").mkdir()
    subprocess.run(["tar", "-x", "-C", str(tmp_path / "untracked"), "-f", str(ARCHIVE / "Mask2Map_navsim_untracked.tar")],
                   check=True)
    shutil.copytree(tmp_path / "untracked/Mask2Map_navsim", ref, dirs_exist_ok=True)
    subprocess.run(["git", "-C", str(got), "apply", str(TEACH / "patches/mask2map_navsim.patch")], check=True)

    def listing(d):
        return {str(p.relative_to(d)): (p.read_bytes(), p.stat().st_mode & 0o111 != 0)
                for p in d.rglob("*") if p.is_file() and ".git" not in p.relative_to(d).parts}

    a, b = listing(ref), listing(got)
    assert set(a) == set(b)
    assert [k for k in a if a[k] != b[k]] == []
    assert len(set(a) - set(listing(tree("pristine")))) == 19        # the 19 untracked files


@has_clones
def test_get_teachers_check_accepts_the_pinned_clones_and_refuses_an_edit(tmp_path):
    r = subprocess.run(["bash", str(TEACH / "get_teachers.sh"), str(TEACHER_SRC), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "9f1d5df + bevformer_navsim.patch" in r.stdout and "a7f07a9 + mask2map_navsim.patch (49 files)" in r.stdout
    copy = tmp_path / "src"
    for name in ("BEVFormer", "Mask2Map_navsim"):
        shutil.copytree(TEACHER_SRC / name, copy / name, symlinks=True,
                        ignore=shutil.ignore_patterns("build", "*.so", "__pycache__", "*.egg-info"))
    f = copy / "BEVFormer/tools/navsim_cache_teacher.py"
    f.write_text(f.read_text() + "\n# edited\n")
    r = subprocess.run(["bash", str(TEACH / "get_teachers.sh"), str(copy), "--check"], capture_output=True, text=True)
    assert r.returncode == 3 and "does not hold bevformer_navsim.patch" in r.stderr


# ----------------------------------------------------------------------------------------------- tables, infos, GT
def fake_frame(token, log, i, t0=1_620_000_000_000_000, sim=True, cams=("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")):
    rng = np.random.default_rng(i)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    scene = log.split("-", 3)[-1] if sim else log
    return {
        "token": token, "frame_idx": i, "timestamp": t0 + 500_000 * i, "log_name": log, "log_token": scene,
        "scene_name": log, "scene_token": scene, "map_location": "us-nv-las-vegas-strip", "roadblock_ids": ["1"],
        "lidar_path": None if sim else f"{log}/MergedPointCloud/{token}.pcd",
        "lidar2ego_translation": np.array([0, 0, 0]), "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
        "ego2global_translation": np.array([664469.3 + i, 3997275.7, 619.8]), "ego2global_rotation": q,
        # SimScale's 4x4 pose matrices: bottom row zero (and here garbage everywhere): never read
        "ego2global": np.zeros((4, 4)), "lidar2global": np.full((4, 4), 7.0),
        "cams": {c: {"data_path": f"{scene}/{c}/{token}-{c}.jpg", "sensor2lidar_rotation": np.eye(3)[[2, 0, 1]],
                     "sensor2lidar_translation": np.array([1.67, -0.02, 1.52]),
                     "cam_intrinsic": np.array([[1545.0, 0, 960], [0, 1545, 560], [0, 0, 1]]),
                     "distortion": np.array([-0.356, 0.17, 0, 0, -0.05])} for c in cams},
        "anns": {"gt_boxes": np.zeros((0, 7)), "gt_names": np.zeros(0, "U16"), "gt_velocity_3d": np.zeros((0, 3))},
        "sample_prev": None, "sample_next": None,
    }


def fake_simscale(root: Path, n=3, r=4):
    """SimScale round r under root with n 12-frame scenes and their frame 0..3 images -> table rows."""
    rows = []
    for k in range(n):
        scene = f"{k:016x}-{r:03d}"
        log = f"2021.05.12.19.36.12_veh-35_00005_00204-{scene}"
        frames = [fake_frame(f"{k:08x}{i:08x}-{r:03d}", log, i) for i in range(SS.NUM_FRAMES)]
        p = root / SS.scene_pkl(r, log)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pickle.dumps(frames))
        for f in SS.camera_files(frames, ("CAM_F0", "CAM_L0", "CAM_R0"), range(SS.NUM_HISTORY_FRAMES)):
            q = SS.sensor_root(root, r) / f
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(b"jpg")
        rows.append(dict(token=frames[SS.CURRENT]["token"], log=log, city="us-nv-las-vegas-strip", source="simscale",
                         round=r, pkl=SS.scene_pkl(r, log)))
    return rows


def fake_table(path: Path, sim_rows, real=("a0000000000000a1", "b0000000000000b2"), val=("c0000000000000c3",)):
    rows = [dict(token=t, log="logA", city="c", source="navtrain_train", round=-1, pkl="") for t in real]
    rows += [dict(token=t, log="logB", city="c", source="navtrain_val", round=-1, pkl="") for t in val]
    df = pd.DataFrame(rows + list(sim_rows))
    df["row"] = np.arange(len(df), dtype=np.int64)
    df["round"] = df["round"].astype(np.int8)
    df[list(SS.TABLE_COLUMNS)].to_parquet(path, index=False)
    return df


def test_token_lists_per_source_in_table_order(tmp_path):
    sim = fake_simscale(tmp_path / "ss")
    fake_table(tmp_path / "t.parquet", sim)
    TCS.main(["tokens", "--table", str(tmp_path / "t.parquet"), "--work", str(tmp_path / "w")])
    assert TCS.read_tokens(tmp_path / "w/tokens/real.txt") == ["a0000000000000a1", "b0000000000000b2", "c0000000000000c3"]
    assert TCS.read_tokens(tmp_path / "w/tokens/sim_r4.txt") == [r["token"] for r in sim]
    summary = json.loads((tmp_path / "w/tokens/lists.json").read_text())
    assert summary["lists"]["sim_r4"]["n"] == 3 and len(summary["lists"]["real"]["token_sha16"]) == 16


def _converter_stub(path: Path):
    """The four navsim_converter.py functions bev_sim_infos calls, with their semantics for this test: keep the
    in-split frame and the 4 before it; one info per frame; link consecutive kept frames."""
    path.write_text('''
CONVERTER_VERSION = "stub"
ORIGINAL_IMAGE_WIDTH = 1920
def select_frame_indices(in_split, history_frames):
    i = in_split.index(True)
    return list(range(max(0, i - history_frames), i + 1))
def convert_frame(frame, log_name, idx, split, sensor_root, scale, prefix, lp, le, cams, check, in_split, sweeps):
    return dict(token=frame["token"], log_name=log_name, timestamp=frame["timestamp"], in_split_tokens=in_split,
                prev=None, next=None, cams={c: frame["cams"][c]["data_path"] for c in cams}), False
def link_prev_next(infos, raw, frames):
    for a, b in zip(infos, infos[1:]):
        a["next"], b["prev"] = b["token"], a["token"]
    return {"linked": len(infos) - 1}
''')
    return path


def test_simscale_bevformer_pool_keys_history_frames_per_scene(tmp_path):
    """Two scenes that share frame tokens (overlapping clips of one log): every scene is one 4-frame chain, the
    current frame keeps its table token, a history frame is <token>#<scene> and links follow the renamed keys."""
    sim = fake_simscale(tmp_path / "ss", n=2)
    a = SS.read_scene(tmp_path / "ss", sim[0]["pkl"])
    b = SS.read_scene(tmp_path / "ss", sim[1]["pkl"])
    for k in range(3):                                                 # scene b's history = scene a's frames 1..3
        b[k]["token"] = a[k + 1]["token"]
    (tmp_path / "ss" / sim[1]["pkl"]).write_bytes(pickle.dumps(b))
    d = TCS.bev_sim_infos(_converter_stub(tmp_path / "conv.py"), tmp_path / "ss",
                          [(r["token"], r["pkl"], r["round"]) for r in sim], workers=1)
    infos = d["infos"]
    keys = [i["token"] for i in infos]
    assert len(infos) == 8 and len(set(keys)) == 8 and d["metadata"]["links"] == {"linked": 6}
    cur = [i for i in infos if i["in_split_tokens"]]
    assert [i["token"] for i in cur] == [r["token"] for r in sim]
    for i in infos:
        if not i["in_split_tokens"]:
            assert i["token"] == TCS.pool_key(i["token"].split("#")[0], i["log_name"])
        assert i["prev"] is None or i["prev"] in keys
    assert cur[1]["prev"] == TCS.pool_key(b[2]["token"], Path(sim[1]["pkl"]).stem)


@has_clones
def test_simscale_bevformer_pool_is_the_converter_output_on_scenes_without_shared_tokens(tmp_path):
    """real: navsim_converter.py's CLI on SimScale scenes that share no token == bev_sim_infos on them (history keys
    mapped back), with the 3 front cameras, --history-frames 4 and --check-files."""
    import yaml
    conv = TEACHER_SRC / "BEVFormer/tools/data_converter/navsim_converter.py"
    sim = fake_simscale(tmp_path / "ss", n=3)
    rd = tmp_path / "ss" / SS.round_dir(4)
    (tmp_path / "f.yaml").write_text(yaml.safe_dump(dict(log_names=[r["log"] for r in sim], tokens=[r["token"] for r in sim])))
    subprocess.run([sys.executable, "-I", str(conv), "--navsim-logs", str(rd / SS.META_DIR), "--sensor-root",
                    str(rd / SS.HIST_DIR), "--sensor-prefix", str(rd / SS.HIST_DIR), "--split", "trainval", "--scene-filter",
                    str(tmp_path / "f.yaml"), "--cameras", "CAM_F0", "CAM_L0", "CAM_R0", "--history-frames", "4",
                    "--check-files", "--workers", "1", "--out", str(tmp_path / "c.pkl")], check=True, capture_output=True)
    ref = pickle.loads((tmp_path / "c.pkl").read_bytes())["infos"]
    got = TCS.bev_sim_infos(conv, tmp_path / "ss", [(r["token"], r["pkl"], r["round"]) for r in sim], workers=1)["infos"]
    assert len(ref) == len(got) == 12
    for x, y in zip(ref, got):
        y = {k: (v.split("#")[0] if k in ("token", "prev", "next") and isinstance(v, str) else v) for k, v in y.items()}
        assert x.keys() == y.keys()
        for k in x:
            if isinstance(x[k], np.ndarray):
                assert np.array_equal(x[k], y[k]), k
            elif k != "cams":
                assert x[k] == y[k], k
        assert x["cams"].keys() == y["cams"].keys() == {"CAM_F0", "CAM_L0", "CAM_R0"}
        for c in x["cams"]:
            for f, v in x["cams"][c].items():
                assert (v == y["cams"][c][f]) if isinstance(v, str) else np.array_equal(v, y["cams"][c][f]), (c, f)


def test_m2m_info_is_create_navsim_infos_on_the_same_frame(tmp_path, m2m_mod):
    """teacher_caches.m2m_info (SimScale) writes what Mask2Map's create_navsim_infos.scan_log writes for a real frame."""
    cni = _load("create_navsim_infos_t", m2m_mod["infos"])
    log = "2021.05.12.19.36.12_veh-35_00005_00204"
    frames = [fake_frame(f"{i:016x}", log, i, sim=False, cams=cni.CAMERA_NAMES) for i in range(4)]
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / f"{log}.pkl").write_bytes(pickle.dumps(frames))
    for f in frames:
        for c in cni.CAMERA_NAMES:
            p = tmp_path / "blobs" / f["cams"][c]["data_path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"jpg")
    _, runs = cni.scan_log((str(tmp_path / "logs"), str(tmp_path / "blobs"), log, cni.CAMERA_NAMES, 2, set()))
    ref = runs[0][2]
    got = TCS.m2m_info(frames[2], tmp_path / "blobs", ref["scene_name"], cams=cni.CAMERA_NAMES)
    assert set(got) == set(ref)
    for k in ref:
        if k in ("sample_idx", "prev", "next", "lidar_path"):        # renumbered by the dataset / unused at test time
            continue
        if k == "cams":
            for c in ref["cams"]:
                for f in ref["cams"][c]:
                    a, b = ref["cams"][c][f], got["cams"][c][f]
                    assert (a == b) if isinstance(a, str) else np.array_equal(a, b), (c, f)
        else:
            assert got[k] == ref[k], k


def test_simscale_infos_take_frame_3_and_the_pose_from_translation_quaternion(tmp_path):
    sim = fake_simscale(tmp_path)
    infos = TCS.sim_infos(tmp_path, [(r["token"], r["pkl"], r["round"]) for r in sim], workers=1)
    assert [i["token"] for i in infos] == [r["token"] for r in sim] and [i["sample_idx"] for i in infos] == [0, 1, 2]
    fr = SS.read_scene(tmp_path, sim[1]["pkl"], sim[1]["token"])[SS.CURRENT]
    i = infos[1]
    assert set(i["cams"]) == set(TCS.M2M_CAMS) and i["in_split_tokens"] and i["lidar_path"] is None
    assert i["e2g_translation"] == list(map(float, fr["ego2global_translation"]))
    assert i["e2g_rotation"] == list(map(float, fr["ego2global_rotation"]))
    assert i["lidar2ego_translation"] == [0.0, 0.0, 0.0]
    cam = i["cams"]["CAM_F0"]
    assert cam["img_fpath"] == str(SS.sensor_root(tmp_path, 4) / fr["cams"]["CAM_F0"]["data_path"])
    c2e = np.linalg.inv(cam["extrinsics"])
    assert np.allclose(c2e[:3, :3], fr["cams"]["CAM_F0"]["sensor2lidar_rotation"])
    assert len({x["scene_name"] for x in infos}) == 3
    (SS.sensor_root(tmp_path, 4) / fr["cams"]["CAM_L0"]["data_path"]).unlink()
    with pytest.raises(SystemExit, match="1 camera images missing"):
        TCS.sim_infos(tmp_path, [(r["token"], r["pkl"], r["round"]) for r in sim], workers=1)


def test_empty_map_gt_satisfies_the_dataset_and_rasterises_to_nothing(tmp_path, m2m_mod):
    mu = _load("navsim_map_utils_t", m2m_mod["utils"])
    toks = ["a0000000000000a1", "0123456789abcdef-004"]
    TCS.empty_map_gt(toks, tmp_path / "gt.npz")
    gt = mu.NavsimMapGT(str(tmp_path / "gt.npz"), (64.0, 64.0), (32.0, 0.0), mu.NAVSIM_MAP_CLASSES)
    assert all(t in gt for t in toks)
    geoms = gt.geoms(toks[1])
    assert geoms == {c: [] for c in mu.NAVSIM_MAP_CLASSES}
    ras = mu.NavsimBEVRasterizer((64.0, 64.0), (200, 200), 3)
    assert not ras({k: geoms[c] for k, c in enumerate(mu.NAVSIM_MAP_CLASSES)}, 4).any()


def test_mask2map_test_path_never_reads_the_map_gt(m2m_mod):
    """The map GT is read by navsim_gt_pipeline (train) and evaluate() only: prepare_test_data (what the cache writer
    runs) does neither, so an empty map GT gives the same model inputs (also measured: docs of the fidelity run)."""
    tree = ast.parse(m2m_mod["dataset"].read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NavsimOfflineMapDataset")
    meth = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
    names = lambda f: {n.attr for n in ast.walk(f) if isinstance(n, ast.Attribute)}  # noqa: E731
    reads_gt = {m for m, f in meth.items() if {"map_gt", "get_bev_gt", "navsim_gt_pipeline"} & names(f)}
    assert reads_gt == {"get_bev_gt", "navsim_gt_pipeline", "prepare_train_data", "evaluate", "__init__"}
    assert names(meth["prepare_test_data"]) == {"get_data_info", "pre_pipeline", "pipeline"}
    assert not ({"map_gt", "rasterizer"} & names(meth["get_data_info"]))


# ----------------------------------------------------------------------------------------------- manifests, merge
DET_TEMPLATE = dict(checkpoint_sha256_head=DET_SHA, layout="samples/<token[:2]>/<token>.npz", target_bev_shape=[100, 100],
                    bev_channels=256, point_cloud_range=[0.0, -32.0, -3.0, 64.0, 32.0, 5.0], created="reference",
                    to_student_transform="student_bev = teacher_bev[:, :, ::-1]", num_samples_written=103288)
MAP_TEMPLATE = dict(checkpoint_sha256=MAP_SHA, split="none", classes=["road", "walkway", "centerline", "crosswalk"],
                    pc_range=[0.0, -32.0, -3.0, 64.0, 32.0, 5.0], roi_size_m=[64.0, 64.0], bev_grid=[100, 100],
                    num_frames=103288, tensors={"bev": {"shape": [256, 100, 100], "dtype": "float16"},
                                                "seg": {"shape": [4, 200, 200], "dtype": "float16"}})


def _det_root(root: Path, tokens, rng):
    for t in tokens:
        b, s, lab = NC.box_arrays(rng.normal(size=(4, 3)), rng.normal(size=(4, 6)), rng.random(4), np.zeros(4))
        NC.write_npz(NC.npz_path(str(root), t), dict(
            bev_feature=rng.normal(size=(256, 100, 100)).astype(np.float16), pred_boxes_3d=b, pred_scores_3d=s,
            pred_labels_3d=lab))
    (root / "passes/real").mkdir(parents=True)
    (root / "passes/real/provenance.json").write_text(json.dumps(dict(
        checkpoint_sha256_head=DET_SHA, n_targets=len(tokens), world_size=2, git_commit=BEV_PIN, ann_file="/i.pkl")))
    (root / "passes/real/done_r0.json").write_text(json.dumps(dict(rank=0, written=len(tokens))))


def _verify(table, *args):
    """teacher_caches verify in a fresh interpreter (the readers read CK_BEV64 at import)."""
    return subprocess.run([sys.executable, str(TEACH / "teacher_caches.py"), "verify", "--table", str(table), "--coverage",
                           *args], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))


def test_det_manifest_keeps_the_reader_fields_and_the_readers_load_every_token(tmp_path):
    sim = fake_simscale(tmp_path / "ss")
    df = fake_table(tmp_path / "t.parquet", sim)
    root = tmp_path / "tc/bevformer/cache_train_100x100"
    _det_root(root, df.token.tolist(), np.random.default_rng(0))
    (tmp_path / "tpl.json").write_text(json.dumps(DET_TEMPLATE))
    TCS.main(["det-manifest", "--root", str(root), "--template", str(tmp_path / "tpl.json")])
    m = json.loads((root / "manifest.json").read_text())
    for k in ("checkpoint_sha256_head", "layout", "target_bev_shape", "bev_channels", "point_cloud_range",
              "to_student_transform"):
        assert m[k] == DET_TEMPLATE[k]
    assert m["num_samples_written"] == len(df) == m["num_samples_expected"]
    assert m["template"]["num_samples_written"] == 103288 and "real" in m["passes"]
    r = _verify(tmp_path / "t.parquet", "--det", str(root), "--det-sha-head", DET_SHA)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(r.stdout)
    assert rep["det"]["problems"] == 0 and rep["n_read"] == len(df)
    assert rep["coverage"]["det"]["simscale"] == {"n": 3, "missing": 0, "example": None}
    os.remove(NC.npz_path(str(root), sim[0]["token"]))
    r = _verify(tmp_path / "t.parquet", "--det", str(root), "--det-sha-head", DET_SHA)
    assert r.returncode == 1 and f"1 of 3 simscale tokens missing, e.g. {sim[0]['token']}" in r.stderr
    (tmp_path / "bad.json").write_text(json.dumps(dict(DET_TEMPLATE, checkpoint_sha256_head="0" * 16)))
    with pytest.raises(SystemExit, match="another checkpoint"):
        TCS.main(["det-manifest", "--root", str(root), "--template", str(tmp_path / "bad.json")])


def _map_pass(root: Path, name: str, tokens, rng, world=1, shard=2):
    d = root / "passes" / name
    (d / "bev").mkdir(parents=True)
    (d / "seg").mkdir()
    data = {}
    for r in range(world):
        mine = tokens[r::world]
        idx = {}
        for s in range(0, len(mine), shard):
            part = mine[s:s + shard]
            bev = rng.normal(size=(len(part), 256, 100, 100)).astype(np.float16)
            np.save(d / "bev" / f"r{r}_s{s // shard:04d}.npy", bev)
            np.save(d / "seg" / f"r{r}_s{s // shard:04d}.npy", np.zeros((len(part), 4, 200, 200), np.float16))
            for row, t in enumerate(part):
                idx[t] = [f"r{r}_s{s // shard:04d}", row]
                data[t] = bev[row]
        (d / f"index_r{r}.json").write_text(json.dumps(idx))
    (d / "provenance.json").write_text(json.dumps(dict(checkpoint_sha256=MAP_SHA, world_size=world, split="none")))
    (d / "meta.json").write_text(json.dumps(MAP_TEMPLATE))
    return data


def test_map_merge_unique_shards_one_index_and_the_readers_load_every_token(tmp_path):
    sim = fake_simscale(tmp_path / "ss")
    df = fake_table(tmp_path / "t.parquet", sim)
    real = df.token[df.source != "simscale"].tolist()
    root = tmp_path / "tc/mask2map/front64"
    rng = np.random.default_rng(0)
    data = _map_pass(root, "real", real, rng, world=2)
    data.update(_map_pass(root, "sim_r4", [r["token"] for r in sim], rng))
    (tmp_path / "tpl.json").write_text(json.dumps(MAP_TEMPLATE))
    TCS.main(["map-merge", "--root", str(root), "--template", str(tmp_path / "tpl.json")])
    idx = json.loads((root / "index.json").read_text())
    meta = json.loads((root / "meta.json").read_text())
    assert meta["num_frames"] == len(idx) == len(df) and meta["checkpoint_sha256"] == MAP_SHA
    assert idx[sim[0]["token"]] == ["sim_r4/r0_s0000", 0] and idx[real[1]][0] == "real/r1_s0000"
    assert os.readlink(root / "bev/sim_r4") == "../passes/sim_r4/bev" and os.readlink(root / "seg/real") == "../passes/real/seg"
    TCS.main(["map-merge", "--root", str(root), "--template", str(tmp_path / "tpl.json")])    # idempotent
    assert json.loads((root / "index.json").read_text()) == idx
    r = _verify(tmp_path / "t.parquet", "--map", str(root), "--map-sha256", MAP_SHA)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout)["map"]["problems"] == 0
    code = ("import os, sys, json, numpy as np; os.environ['CK_BEV64'] = '1'; sys.path.insert(0, sys.argv[1]);"
            "from navsim.agents.para_ssr.refiner.resmap_cache import ResmapCache;"
            "rc = ResmapCache(sys.argv[2], expect_sha256=sys.argv[3], expect_split='none');"
            "print(json.dumps({t: float(rc.load_bev(t, s_grid=False).astype(np.float64).sum()) for t in rc.tokens()}))")
    r = subprocess.run([sys.executable, "-c", code, str(REPO), str(root), MAP_SHA], capture_output=True, text=True)
    sums = json.loads(r.stdout)
    assert {t: round(v, 2) for t, v in sums.items()} == {t: round(float(x.astype(np.float64).sum()), 2) for t, x in data.items()}
    _map_pass(root, "sim_r3", [sim[0]["token"]], rng)
    with pytest.raises(SystemExit, match="tokens in pass sim_r4 and an earlier pass"):
        TCS.main(["map-merge", "--root", str(root), "--template", str(tmp_path / "tpl.json")])
    shutil.rmtree(root / "passes/sim_r3")
    (root / "passes/navtest").mkdir()
    with pytest.raises(SystemExit, match="must not be named 'navtest'"):
        TCS.main(["map-merge", "--root", str(root), "--template", str(tmp_path / "tpl.json")])


def test_pass_token_lists_are_frozen_and_skip_what_a_cache_holds(tmp_path):
    root = tmp_path / "front64"
    _map_pass(root, "ref", ["a0000000000000a1"], np.random.default_rng(0))
    (tmp_path / "all.txt").write_text("a0000000000000a1\nb0000000000000b2\n")
    out = tmp_path / "p/tokens.txt"
    TCS.main(["pass-tokens", "--list", str(tmp_path / "all.txt"), "--out", str(out), "--map", str(root)])
    assert TCS.read_tokens(out) == ["b0000000000000b2"]
    (tmp_path / "all.txt").write_text("c0000000000000c3\n")
    TCS.main(["pass-tokens", "--list", str(tmp_path / "all.txt"), "--out", str(out), "--map", str(root)])
    assert TCS.read_tokens(out) == ["b0000000000000b2"]


# ----------------------------------------------------------------------------------------------- fidelity helpers
TF = _load("teacher_fidelity_t", TEACH / "teacher_fidelity.py")
TRAIN_SPLIT = Path("/home/external-user/ssd/yongjae_refiner/splits/e2e_train_trainlogs.parquet")


@pytest.mark.skipif(not TRAIN_SPLIT.is_file(), reason="no navtrain_train token table")
def test_fidelity_gate_takes_the_norm_files_tokens():
    """real: the 2048 tokens of compute_teacher_norm (seed 0, sorted navtrain_train) hash to the norm files' token_hash."""
    import hashlib
    pick = TF.norm_tokens(pd.read_parquet(TRAIN_SPLIT).token.astype(str).tolist())
    assert len(pick) == 2048 and hashlib.sha256("\n".join(pick).encode()).hexdigest()[:16] == TF.NORM_HASH


def test_box_match_and_axis_hypotheses():
    gt = np.array([[10.0, 2.0], [30.0, -5.0], [50.0, 10.0]])
    pred = np.array([[10.3, 2.1, 0], [29.0, -5.5, 0], [50.0, -10.0, 0]])
    m, n, errs = TF.box_match(pred, np.array([0.9, 0.8, 0.95]), gt)
    assert (m, n) == (2, 3) and max(errs) < 1.2                      # the mirrored third box is no match
    assert TF.box_match(pred, np.array([0.1, 0.1, 0.1]), gt)[:2] == (0, 3)
    x = np.arange(6).reshape(1, 2, 3)
    assert np.array_equal(TF._hyp(x, "flip_rows"), x[:, ::-1]) and np.array_equal(TF._hyp(x, "transpose"), x.swapaxes(1, 2))
    assert TF._hyp(x, "as_cached") is x and set(TF.HYPOTHESES) == {"as_cached", "flip_rows", "flip_cols", "transpose"}


# ----------------------------------------------------------------------------------------------- setup scripts
FORBIDDEN = ("/home/external-user", "/workspace/", "/venv/", "miniconda3", "/tmp/claude", "pkill", "pgrep")


@pytest.fixture(scope="module")
def iso(tmp_path_factory):
    """The release tree and navsim/ without tools/release/paths.env, so the scripts see only the fake roots."""
    d = tmp_path_factory.mktemp("iso") / "repo"
    ign = shutil.ignore_patterns("__pycache__", "paths.env", "*.pyc")
    shutil.copytree(REPO / "navsim", d / "navsim", ignore=ign)
    shutil.copytree(REPO / "tools/release", d / "tools/release", ignore=ign)
    return d


def script_env(tmp_path, **kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SSR_", "NAVSIM_", "OPENSCENE", "NUPLAN", "TEACHER",
                                                                     "CAM_TEACHERS", "GPUS", "ROUNDS", "RESUME"))}
    root = tmp_path / "srv"
    (root / "ds/navsim_logs/trainval").mkdir(parents=True, exist_ok=True)
    (root / "ds/navsim_logs/trainval/x.pkl").write_bytes(b"")
    env.update(SSR_DATA_ROOT=str(root / "data"), SSR_TEACHER_CACHE=str(root / "tc"), OPENSCENE_DATA_ROOT=str(root / "ds"),
               SSR_SIMSCALE_ROOT=str(root / "simscale"), SSR_PY=sys.executable, SSR_TORCHRUN="/bin/echo",
               SSR_RUN_ROOT=str(root / "runs"), SETUP_DRY="1", PYTHONDONTWRITEBYTECODE="1")
    env.update(kw)
    return env


def run_script(iso, name, env):
    return subprocess.run(["bash", str(iso / "tools/release/setup" / name)], env=env, capture_output=True, text=True)


def _code(p: Path) -> str:
    return "\n".join(x for x in p.read_text().splitlines() if not x.lstrip().startswith("#"))


@pytest.mark.parametrize("p", [SETUP / "65_cam_teacher_bev.sh", SETUP / "60_teacher_bev.sh", TEACH / "get_teachers.sh",
                               TEACH / "setup_teacher_env.sh"])
def test_scripts_parse_and_hold_no_server_path(p):
    assert subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode == 0
    code = _code(p)
    for bad in FORBIDDEN:
        if p.parent == TEACH and bad in ("/home/external-user",):
            continue
        assert bad not in code, f"{p.name}: {bad}"
    assert "THROUGHPUT_PLACEHOLDER" not in p.read_text()


def test_cam_teacher_step_recipe_lines(tmp_path, iso):
    env = script_env(tmp_path, GPUS="0,1")
    before = sorted(str(p) for p in (tmp_path / "srv").rglob("*"))
    r = run_script(iso, "65_cam_teacher_bev.sh", env)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert sorted(str(p) for p in (tmp_path / "srv").rglob("*")) == before, "wrote under SETUP_DRY=1"
    for bad in FORBIDDEN[:3]:
        assert bad not in out.replace(str(Path(sys.executable).parent), "PYBIN"), bad
    g = [x for x in r.stdout.splitlines() if x.startswith("+ ")]
    tc = env["SSR_TEACHER_CACHE"]
    det, m2m = f"{tc}/bevformer/cache_train_100x100", f"{tc}/mask2map/front64"
    conv = [x for x in g if "navsim_converter.py --split" in x]
    assert len(conv) == 1                                           # the real pool: the navtrain filter, 8 cameras
    assert "scene_filter/navtrain.yaml --history-frames 4 --check-files" in conv[0] and "--cameras" not in conv[0]
    sim = [x for x in g if "teacher_caches.py bev-infos" in x]
    assert len(sim) == 5 and all("tools/data_converter/navsim_converter.py --out" in x for x in sim)
    assert "--round 4 " in sim[4] and "navsim_infos_sim_r4.pkl" in sim[4]
    cache = [x for x in g if "navsim_cache_teacher.py" in x]
    assert len(cache) == 12                                         # 6 passes x 2 GPUs
    assert "CUDA_VISIBLE_DEVICES=0 " in cache[0] and "--rank 0 --world 2" in cache[0] and "--rank 1 --world 2" in cache[1]
    assert all(f"--out {det} " in x and "--resume" in x and f"--expect-sha-head {DET_SHA}" in x for x in cache)
    assert f"--pass sim_r2 --token-list {det}/passes/sim_r2/tokens.txt" in cache[6]
    m = [x for x in g if "cache_teacher_m2m.py" in x]
    assert len(m) == 6 and all("--nproc_per_node=2" in x and "--split none --only-split-tokens" in x and "--resume" in x
                               and "CUDA_VISIBLE_DEVICES=0,1 " in x for x in m)
    assert f"--out {m2m}/passes/real" in m[0] and "navsim_map_infos_real.pkl" in m[0] and "gt/real.npz" in m[0]
    assert sum("create_navsim_infos.py --filter navtrain" in x for x in g) == 1
    assert sum("teacher_caches.py m2m-infos" in x for x in g) == 5 and sum("empty-gt" in x for x in g) == 6
    assert sum(f"det-manifest --root {det}" in x for x in g) == 1 and sum(f"map-merge --root {m2m}" in x for x in g) == 1
    assert any("verify_assets.py --profile full --only teacher_cache --data trainval_simscale" in x for x in g)
    assert g.index(next(x for x in g if "det-manifest" in x)) < g.index(m[0])   # DET done before MAP starts
    one = run_script(iso, "65_cam_teacher_bev.sh", dict(env, TEACHERS="mask2map", ROUNDS="4", GPUS="3"))
    gm = [x for x in one.stdout.splitlines() if x.startswith("+ ")]
    assert not any("navsim_cache_teacher.py" in x for x in gm) and sum("cache_teacher_m2m.py" in x for x in gm) == 2
    assert run_script(iso, "65_cam_teacher_bev.sh", dict(env, TEACHERS="bevfusion")).returncode == 3


def test_cam_teacher_step_resume_rules(tmp_path, iso):
    env = script_env(tmp_path)
    work = Path(env["SSR_TEACHER_CACHE"]) / "_work/cam_teachers"
    (work / "tokens").mkdir(parents=True)
    (work / "tokens/lists.json").write_text("{}")
    r = run_script(iso, "65_cam_teacher_bev.sh", env)
    assert r.returncode == 3 and "RESUME=1" in r.stderr
    assert run_script(iso, "65_cam_teacher_bev.sh", dict(env, RESUME="1")).returncode == 0
    (work / "done.json").write_text("{}")
    r = run_script(iso, "65_cam_teacher_bev.sh", dict(env, RESUME="1"))
    assert r.returncode == 3 and "refusing to overwrite" in r.stderr


def test_teacher_bev_step_keeps_its_old_cases_and_copies_the_camera_teachers(tmp_path, iso):
    head = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:tools/release/setup/60_teacher_bev.sh"],
                          capture_output=True, text=True, check=True).stdout
    (iso / "tools/release/setup/60_head.sh").write_text(head)
    env = script_env(tmp_path, BEVFUSION_SRC="host:/src/")
    for teachers in ("resmap bevfusion", "resmap", "bevfusion"):
        new = run_script(iso, "60_teacher_bev.sh", dict(env, TEACHERS=teachers))
        old = run_script(iso, "60_head.sh", dict(env, TEACHERS=teachers))
        assert new.returncode == old.returncode == 0
        assert new.stdout.replace("60_teacher_bev.sh", "X") == old.stdout.replace("60_head.sh", "X")
    out = run_script(iso, "60_teacher_bev.sh", dict(env, TEACHERS="bevformer mask2map", BEVFORMER_SRC="h:/b/",
                                                    MASK2MAP_SRC="h:/m/")).stdout
    tc = env["SSR_TEACHER_CACHE"]
    assert f"+ rsync -a --partial h:/b/ {tc}/bevformer/cache_train_100x100/" in out
    assert f"+ rsync -a --partial --exclude /navtest h:/m/ {tc}/mask2map/front64/passes/ref/" in out
    assert f"map-merge --root {tc}/mask2map/front64 --template {tc}/mask2map/front64/passes/ref/meta.json" in out
    r = run_script(iso, "60_teacher_bev.sh", dict(env, TEACHERS="mask2map", SETUP_DRY="0"))
    assert r.returncode == 3 and "MASK2MAP_SRC" in r.stderr and not (Path(tc) / "mask2map").exists()


def test_teacher_env_arch_comes_from_the_gpus(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    smi = bindir / "nvidia-smi"
    env = dict(os.environ, TEACHER_SRC=str(tmp_path), PATH=f"{bindir}:/usr/bin:/bin")
    env.pop("TORCH_CUDA_ARCH_LIST", None)
    for caps, want in (("9.0\n9.0\n", "9.0"), ("12.0\n", "12.0")):
        smi.write_text(f"#!/bin/sh\nprintf '{caps}'\n")
        smi.chmod(0o755)
        r = subprocess.run(["bash", str(TEACH / "setup_teacher_env.sh"), str(tmp_path / "env"), "arch"], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == want, r.stderr
    r = subprocess.run(["bash", str(TEACH / "setup_teacher_env.sh"), str(tmp_path / "env"), "arch"],
                       env=dict(env, TORCH_CUDA_ARCH_LIST="8.6"), capture_output=True, text=True)
    assert r.stdout.strip() == "8.6"
    smi.write_text("#!/bin/sh\nexit 9\n")
    r = subprocess.run(["bash", str(TEACH / "setup_teacher_env.sh"), str(tmp_path / "env"), "arch"], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 2 and "TORCH_CUDA_ARCH_LIST" in r.stderr


# ----------------------------------------------------------------------------------------------- dev caches (real)
DEV_TABLE = Path(os.environ.get("SSR_DATA_ROOT", "/home/external-user/ssd/simscale_dev/data")) / "splits/trainval_simscale.parquet"
DEV_TC = Path(os.environ.get("SSR_TEACHER_CACHE", "/home/external-user/ssd/simscale_dev/teacher_cache"))


@pytest.mark.skipif(not (DEV_TC / "mask2map/front64/index.json").is_file() or not DEV_TABLE.is_file(),
                    reason="no dev caches")
def test_dev_caches_cover_the_dev_table():
    """real: every token of the dev table has an entry in both dev caches; a 64-token sample reads through the readers."""
    r = _verify(DEV_TABLE, "--det", str(DEV_TC / "bevformer/cache_train_100x100"), "--map", str(DEV_TC / "mask2map/front64"),
                "--sample", "64")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
