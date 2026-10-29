"""refiner v6c stage T after report 129 (package P1): one field read rule for teachers and student, legacy teachers
refused, table provenance, the T-D target raster_min1, Platt on the KD-gated steps, the coverage gate, the seed, the
OOF metrics, and decode_v6's held heading of stopped poses.  CPU; the smokes and real-cache probes are skipped when the
caches of this host are absent."""
from __future__ import annotations

import filecmp
import inspect
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from navsim.agents.para_ssr.modules import corridor_head as CH
from navsim.agents.para_ssr.refiner_v6c import nets as N
from navsim.agents.para_ssr.refiner_v6c import teacher as TS
import tools.refiner_v6c.train_teachers as TT

REPO = Path(__file__).resolve().parents[2]
KM = Path(os.environ.get("KM_ROOT", "/workspace/navsim_workspace/corridor_cache"))
TEACH = Path(os.environ.get("TEACHER_CACHE_ROOT", "/workspace/teacher_cache"))
KM_SCORER = Path("/workspace/byounggun/SSR/navsim/agents/para_ssr/modules/candidate_scorer.py")
OLD_TEACHER = REPO / "work_dirs/km_runs/v6c_teacher/v6cT_platt3/teachers_te.pt"
CFG = SimpleNamespace(embed_dims=256, bev_h=100, bev_w=100)
TOOL = str(REPO / "tools/refiner_v6c/train_teachers.py")


def have_caches() -> bool:
    return ((KM / "corridor64/navtrain_val/ok.npy").is_file() and (KM / "exact/navtrain_val").is_dir()
            and (KM / "exact2/navtrain_val").is_dir() and (TEACH / "bevfusion").is_dir() and (TEACH / "resmap").is_dir())


def _run(args, **kw):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO)}
    return subprocess.run([sys.executable, TOOL, *args], cwd=REPO, env=env, capture_output=True, text=True, **kw)


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def teacher_sd():
    torch.manual_seed(0)
    return TS.TeacherSet().state_dict()


def _ckpt(tmp_path, sd, name="t.pt", **meta):
    p = tmp_path / name
    m = {"folds": 2, "hid_e": 192, "planes": 49, "phase_done": "te", "phases": ["fields", "platt_occ", "te"], **meta}
    torch.save({"state_dict": sd, "meta": m}, p)
    return p


def _states(b=2, r=3, x0=-3.0, seed=0):
    """random tracked states [B, R, 41, 7] whose points start behind the ego (x < 0) and move forward"""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(41, dtype=torch.float32) * 0.1
    v = 2.0 + 8.0 * torch.rand(b, r, 1, generator=g)
    S = torch.zeros(b, r, 41, 7)
    S[..., 0] = x0 + v * t
    S[..., 1] = 3.0 * (torch.rand(b, r, 1, generator=g) - 0.5) * t
    S[..., 2] = 0.2 * (torch.rand(b, r, 1, generator=g) - 0.5)
    S[..., 3] = v
    return S


# ------------------------------------------------------------------------------------------------- F01 read rule
def test_candidate_scorer_is_kms_file_and_the_read_flags_are_gone():
    if KM_SCORER.is_file():
        assert filecmp.cmp(REPO / "navsim/agents/para_ssr/modules/candidate_scorer.py", KM_SCORER, shallow=False)
    from navsim.agents.para_ssr.modules.candidate_scorer import field_readout
    ps = inspect.signature(field_readout).parameters
    assert "rear_mask" not in ps and "legacy_rear" not in ps
    pr = inspect.signature(N.path_reads).parameters
    assert "legacy_rear" not in pr and "rear_mask" in pr and pr["rear_mask"].default is False


def test_path_reads_rear_mask_has_no_effect():
    """every reader uses km's CH._sample rule, which already masks x < 0; rear_mask stays for hook.py's calls"""
    torch.manual_seed(0)
    S = _states(x0=-4.0)
    assert bool((S[..., 0] < 0).any())
    bev = torch.randn(2, 100 * 100, 16)
    sdf = 4.0 * torch.randn(2, 200, 200)
    occ = torch.rand(2, 49, 200, 200)
    a = N.path_reads(S, bev, sdf, occ, 100, 100, 64.0, rear_mask=True)
    b = N.path_reads(S, bev, sdf, occ, 100, 100, 64.0, rear_mask=False)
    c = N.path_reads(S, bev, sdf, occ, 100, 100, 64.0)
    assert all(torch.equal(a[k], b[k]) and torch.equal(a[k], c[k]) for k in a)


def test_load_teachers_needs_the_current_read_rule(tmp_path, teacher_sd):
    for rr in (None, "old"):
        p = _ckpt(tmp_path, teacher_sd, f"t_{rr}.pt", **({} if rr is None else {"read_rule": rr}))
        with pytest.raises(RuntimeError, match="read_rule.*v6cT_platt3.*v6c_pipeline.sh"):
            TS.load_teachers(str(p), CFG)
    ts = TS.load_teachers(str(_ckpt(tmp_path, teacher_sd, read_rule=TS.READ_RULE)), CFG)
    assert ts.loaded_parts["te"] and not ts.training
    with pytest.raises(RuntimeError, match="incomplete"):                                    # completeness comes first
        TS.load_teachers(str(_ckpt(tmp_path, teacher_sd, "inc.pt", phase_done="fields", read_rule=TS.READ_RULE)), CFG)


def test_the_10_28_teacher_is_refused():
    """read-only: the legacy-rule teacher of the 10-28 / 10-29 runs no longer loads"""
    if not OLD_TEACHER.is_file():
        pytest.skip("v6cT_platt3 not on this host")
    with pytest.raises(RuntimeError, match="read_rule"):
        TS.load_teachers(str(OLD_TEACHER), CFG)


def test_parts_and_epoch_checkpoints_of_older_code_are_refused(tmp_path, teacher_sd):
    ts = TS.TeacherSet()
    pre, keys = TT._part_keys(ts, "td", 0)
    sd = {k: teacher_sd[k] for k in keys}
    torch.save({"state_dict": sd, "meta": {"kind": "td", "fold": 0}}, tmp_path / "part_td0.pt")
    with pytest.raises(RuntimeError, match="part_td0.pt.*read_rule"):
        TT.load_part(ts, "td", 0, tmp_path)
    torch.save({"state_dict": sd, "meta": {"kind": "td", "fold": 0, "read_rule": TS.READ_RULE, "git": "x"}},
               tmp_path / "part_td0.pt")
    assert TT.load_part(ts, "td", 0, tmp_path)["git"] == "x"
    torch.save({"module": {}, "opt": {}, "epoch": 0, "step": 1}, tmp_path / "e.ckpt")
    with pytest.raises(RuntimeError, match="read_rule"):
        TT._load_epoch_ckpt(tmp_path / "e.ckpt")
    torch.save({"module": {}, "opt": {}, "epoch": 0, "step": 1, "read_rule": TS.READ_RULE}, tmp_path / "e.ckpt")
    assert TT._load_epoch_ckpt(tmp_path / "e.ckpt")["step"] == 1


def test_resume_of_an_older_checkpoint_is_refused(tmp_path, teacher_sd):
    p = _ckpt(tmp_path, teacher_sd, "old.pt", phase_done="fields", phases=["fields"])
    r = _run(["--phase", "platt_occ", "--resume", str(p), "--device", "cpu", "--out", str(tmp_path / "o")], timeout=600)
    assert r.returncode != 0 and "read_rule" in r.stderr


def test_git_rev_and_lineage():
    rev = TT._git_rev()
    assert rev == "unknown" or re.fullmatch(r"[0-9a-f]{4,40}(-dirty)?", rev)
    logs = []
    assert TT._git_lineage("merge_fields", "a", {"td0": {"git": "a"}, "tm0": {"git": "a"}}, {}, logs.append) == \
        {"git_mixed": False} and not logs
    assert TT._git_lineage("merge_fields", "a", {"td0": {"git": "a"}, "tm0": {"git": "b"}}, {}, logs.append)["git_mixed"]
    assert logs[-1]["warning"] == "parts from different git revisions" and logs[-1]["git"]["tm0"] == "b"
    assert TT._git_lineage("finish", "a", {}, {"git": "a", "git_mixed": True}, logs.append)["git_mixed"]


# ------------------------------------------------------------------------------------------------- F02 provenance
def test_teacher_trained_on_other_tables_warns(tmp_path, teacher_sd, caplog):
    a, b = tmp_path / "ex2_a", tmp_path / "ex2_b"
    a.mkdir(), b.mkdir()
    data = {"corridor_dir": os.path.realpath(tmp_path), "exact_dir": os.path.realpath(tmp_path),
            "exact2_dir": os.path.realpath(a), "teacher_root": os.path.realpath(tmp_path), "tokens_root": "x"}
    p = _ckpt(tmp_path, teacher_sd, read_rule=TS.READ_RULE, data=data)
    lg = "navsim.agents.para_ssr.refiner_v6c.teacher"
    same = SimpleNamespace(**vars(CFG), corridor_dir=str(tmp_path), exact_dir=str(tmp_path), exact2_dir=str(a),
                           distill_feature_root=str(tmp_path))
    for cfg in (CFG, same, SimpleNamespace(**{**vars(same), "exact2_dir": ""})):           # no attr / equal / empty
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=lg):
            TS.load_teachers(str(p), cfg)
        assert not [r for r in caplog.records if "other tables" in r.getMessage()]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=lg):
        TS.load_teachers(str(p), SimpleNamespace(**{**vars(same), "exact2_dir": str(b)}))
    msg = [r.getMessage() for r in caplog.records if "other tables" in r.getMessage()]
    assert len(msg) == 1 and "exact2_dir" in msg[0] and "corridor_dir" not in msg[0]


def test_builder_cfg_honours_exact2_dir(monkeypatch):
    monkeypatch.delenv("EXACT2_DIR", raising=False)
    c = TT.builder_cfg("/km", "/tc")
    assert (c.exact2_dir, c.exact_dir, c.corridor_dir, c.distill_feature_root) == ("/km/exact2", "/km/exact",
                                                                                  "/km/corridor64", "/tc")
    monkeypatch.setenv("EXACT2_DIR", "/tmp/x")
    assert TT.builder_cfg("/km", "/tc").exact2_dir == "/tmp/x"
    monkeypatch.setenv("EXACT2_DIR", "")                                                     # train.sh ${EXACT2_DIR:-..}
    assert TT.builder_cfg("/km", "/tc").exact2_dir == "/km/exact2"


# ------------------------------------------------------------------------------------------------- F03 raster_min1
def _box(cx, cy, h, length, width, t):
    c, s = np.cos(h), np.sin(h)
    pts = [(cx + c * dx - s * dy, cy + s * dx + c * dy)
           for dx, dy in ((length / 2, width / 2), (length / 2, -width / 2), (-length / 2, -width / 2), (-length / 2, width / 2))]
    return np.array([t, *np.ravel(pts)], np.float32)


def test_raster_min1_marks_a_sub_cell_object():
    """a 0.2 x 0.2 m box centred between four cell centres: no centre inside, so the base raster loses it"""
    i, j = 50, 100                                                       # cell (row i, col j): x 16.0..16.32, y 0..-0.32
    cx, cy = (i + 1) * 0.32, 32.0 - (j + 1) * 0.32                       # the corner shared by 4 cells (between centres)
    cx, cy = cx - 0.01, cy + 0.01                                        # mean inside cell (i, j)
    row = _box(cx, cy, 0.0, 0.2, 0.2, 5)[None]
    assert not CH.raster_agents(row).any()
    occ = TT.raster_agents_min1(row)
    assert int(occ.sum()) == 1 and bool(occ[4, i, j])
    assert not TT.raster_agents_min1(_box(cx, cy, 0.0, 0.2, 0.2, 50)[None]).any()          # t outside 1..49
    assert not TT.raster_agents_min1(_box(-5.0, 0.0, 0.0, 0.2, 0.2, 5)[None]).any()        # behind the ego


def test_raster_min1_equals_the_base_raster_for_cars():
    g = np.random.default_rng(0)
    rows = np.stack([_box(g.uniform(-3, 67), g.uniform(-35, 35), g.uniform(-np.pi, np.pi), 4.5, 2.0, int(g.integers(1, 50)))
                     for _ in range(300)])
    assert np.array_equal(TT.raster_agents_min1(rows), CH.raster_agents(rows))


def test_raster_min1_on_navtrain_val_tokens():
    """every 50th of the first 10,000 navtrain_val tokens: a superset of the base raster, +1 cell per in-grid row that
    marks no cell (unless another row already holds that cell); ~2.5 % of the in-grid rows (audit probe 132 / 5313)"""
    d = KM / "corridor64/navtrain_val"
    if not (d / "agent_rows.npy").is_file():
        pytest.skip("corridor64 not on this host")
    off, rows = np.load(d / "agent_off.npy"), np.load(d / "agent_rows.npy", mmap_mode="r")
    n_in = n_van = 0
    for r in range(0, 10000, 50):
        R = np.asarray(rows[off[r]:off[r + 1]]).astype(np.float32)
        a, b = CH.raster_agents(R), TT.raster_agents_min1(R)
        assert not (a & ~b).any()
        cells = set()
        for row in R:
            t = int(row[0])
            p = row[1:].reshape(4, 2)
            i, j = int(np.floor(float(p[:, 0].mean()) / 0.32)), int(np.floor((32.0 - float(p[:, 1].mean())) / 0.32))
            if not (1 <= t <= 49 and 0 <= i < 200 and 0 <= j < 200):
                continue
            n_in += 1
            # a rectangle whose shorter side is >= sqrt(2) cells holds a disk of radius res / sqrt(2): always a centre
            if min(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1])) >= 0.46:
                continue
            one = row.copy()
            one[0] = 1
            if not CH.raster_agents(one[None], planes=1).any():
                n_van += 1
                cells.add((t - 1, i, j))
        assert int((b & ~a).sum()) == sum(1 for c in cells if not a[c])
    assert 0.01 < n_van / n_in < 0.05, (n_van, n_in)


# ------------------------------------------------------------------------------------------------- F04 KD windows
def _old_gates(S):
    """path_outputs' window code before report 129 (moved unchanged into TeacherSet.kd_windows)"""
    s21 = S[:, :, N.T21]
    corners = N.footprint5(s21)[..., 1:, :]

    def win(c):
        return ((c[..., 0] >= 0) & (c[..., 0] <= 64.0) & (c[..., 1].abs() <= 32.0)).all(-1)

    inwin = win(corners)
    shift = (torch.hypot(s21[..., 3], s21[..., 4]) * TS.TTC_LAST_DT)[..., None]
    u = torch.stack([torch.cos(s21[..., 2]), torch.sin(s21[..., 2])], -1)
    return inwin, inwin & win(corners + (u * shift)[..., None, :])


def test_kd_windows_are_the_path_output_gates(tmp_path, teacher_sd):
    S = torch.cat([_states(1, 3, x0=-3.0), _states(1, 3, x0=40.0, seed=1)], 1)               # behind / beyond 64 m
    S[:, 2, :, 1] = 31.0                                                                      # at the lateral edge
    inwin, inproj = TS.TeacherSet.kd_windows(S)
    old = _old_gates(S)
    assert torch.equal(inwin, old[0]) and torch.equal(inproj, old[1])
    assert bool(inwin.any()) and not bool(inwin.all()) and not torch.equal(inwin, inproj)
    ts = TS.load_teachers(str(_ckpt(tmp_path, teacher_sd, read_rule=TS.READ_RULE)), CFG)
    flds = {"tok_det": torch.randn(1, 100 * 100, 256), "sdf_m": torch.full((1, 200, 200), 5.0),
            "occ_prob": torch.rand(1, 49, 200, 200)}
    po = ts.path_outputs(S, flds, torch.zeros(1, 6, 21, 3), torch.zeros(1, dtype=torch.long))
    assert torch.equal(po["g_obj"], inwin.float()) and torch.equal(po["g_off"], inwin.float())
    assert torch.equal(po["g_ttc"], inproj.float())


def test_platt_samples_only_the_kd_gated_steps():
    """a stopped ego's rear corners sit at x < 0: those steps never get an overlap KD target, so they are not in the
    overlap Platt sample either; valid_bin (labels.stride2) is the overlap / approach validity when present"""
    S = torch.zeros(1, 2, 41, 7)
    S[:, 1, :, 0] = 10.0                                                                      # row 1 entirely ahead
    ones = torch.ones(1, 2, 21)
    lab = {"valid": ones.clone(), "valid_off": ones.clone()}
    g = TT._platt_gates(lab, S)
    assert not bool(g["ovl"][0, 0].any()) and not bool(g["off"][0, 0].any()) and not bool(g["cont"][0, 0].any())
    assert bool(g["ovl"][0, 1].all()) and bool(g["ttc"][0, 1].all())
    lab["valid_bin"] = ones.clone()
    lab["valid"][..., 3] = 0.0                                                                # odd-step positive slot
    g = TT._platt_gates(lab, S)
    assert bool(g["ovl"][0, 1, 3]) and bool(g["ttc"][0, 1, 3]) and not bool(g["cont"][0, 1, 3])


def test_mask_rows_covers_valid_bin():
    ok = torch.tensor([1.0, 0.0])
    lab = {k: torch.ones(2, 3, 21) for k in ("valid", "valid_off", "valid_bin", "ovl")}
    TT._mask_rows(lab, ok)
    for k in ("valid", "valid_off", "valid_bin"):
        assert bool((lab[k][0] == 1).all()) and bool((lab[k][1] == 0).all())
    assert bool((lab["ovl"] == 1).all())
    lab = {k: torch.ones(2, 3, 21) for k in ("valid", "valid_off")}                          # labels without valid_bin
    TT._mask_rows(lab, ok.bool())
    assert set(lab) == {"valid", "valid_off"} and float(lab["valid"][1].sum()) == 0.0


# ------------------------------------------------------------------------------------------------- F05 coverage gate
def test_coverage_check_passes_on_the_real_tables(tmp_path):
    if not have_caches():
        pytest.skip("label tables / teacher caches not on this host")
    r = _run(["--phase", "check", "--device", "cpu", "--out", str(tmp_path)], timeout=1200)
    assert r.returncode == 0, r.stderr[-2000:]
    cov = json.loads((tmp_path / "coverage.json").read_text())
    assert set(cov["splits"]) == {"navtrain_train", "navtrain_val"} and cov["min_coverage"] == 1.0
    for sp, tb in cov["splits"].items():
        assert all(v == 1.0 for k, v in tb.items() if not (k == "plan_score" and v is None)), (sp, tb)
    assert {"corridor64", "exact", "exact2", "bevfusion", "resmap"} <= set(cov["paths"])
    assert any(d.get("ok") for d in _jsonl(tmp_path / "log_check.jsonl"))


def test_coverage_check_fails_on_a_missing_token(tmp_path):
    if not have_caches():
        pytest.skip("label tables / teacher caches not on this host")
    import pandas as pd
    toks = pd.read_parquet(REPO / "resources/km_corridor/packed/navtrain_val/tokens.parquet").token.astype(str).tolist()
    pd.DataFrame({"token": toks + ["deadbeefdeadbeef"]}).to_parquet(tmp_path / "tok.parquet")
    r = _run(["--phase", "check", "--device", "cpu", "--out", str(tmp_path), "--check-splits", "navtrain_val",
              "--check-tokens", str(tmp_path / "tok.parquet")], timeout=1200)
    assert r.returncode == 2, r.stderr[-2000:]
    cov = json.loads((tmp_path / "coverage.json").read_text())["splits"]["navtrain_val"]
    assert cov["corridor64"] == pytest.approx(len(toks) / (len(toks) + 1)) and "below" in r.stderr


# ------------------------------------------------------------------------------------------------- F06 / F07 smokes
SMOKE = ["--device", "cpu", "--limit", "8", "--epochs", "1", "--batch", "2", "--workers", "0"]


def test_td_smoke_is_seeded_and_records_provenance(tmp_path):
    if not have_caches():
        pytest.skip("label tables / teacher caches not on this host")
    sds = []
    for i in (0, 1):
        r = _run(["--phase", "td", "--fold", "0", *SMOKE, "--out", str(tmp_path / f"o{i}")], timeout=1800)
        assert r.returncode == 0, r.stderr[-2000:]
        ck = torch.load(tmp_path / f"o{i}/part_td0.pt", map_location="cpu")
        sds.append(ck["state_dict"])
    assert sds[0].keys() == sds[1].keys() and all(torch.equal(sds[0][k], sds[1][k]) for k in sds[0])
    meta = ck["meta"]
    assert meta["read_rule"] == TS.READ_RULE and meta["seed"] == 0 and meta["git"]
    assert meta["occ_target"].startswith("raster_min1") and "OOF holds" in meta["backbone_in_sample"]
    assert set(meta["data"]) == {"corridor_dir", "exact_dir", "exact2_dir", "teacher_root", "tokens_root"}
    assert meta["tables_meta"]["corridor64/navtrain_val"] and len(meta["tables_meta"]) == 6
    done = [d for d in _jsonl(tmp_path / "o1/log_td_f0.jsonl") if "epoch_done" in d]
    assert done and done[-1]["ok_frac"] == 1.0
    assert torch.load(tmp_path / "o1/part_td0.ckpt", map_location="cpu")["read_rule"] == TS.READ_RULE


def test_sequential_smoke_logs_oof_metrics_and_loads(tmp_path):
    if not have_caches():
        pytest.skip("label tables / teacher caches not on this host")
    r = _run(["--phase", "all", *SMOKE, "--platt-batches", "2", "--out", str(tmp_path)], timeout=3600)
    assert r.returncode == 0, r.stderr[-3000:]
    recs = _jsonl(tmp_path / "log_all.jsonl")
    ev = [d for d in recs if d.get("phase") == "platt_eval"]
    assert {d.get("map") for d in ev} >= {"occ", "te_ovl", "te_ttc", "tm_off", "te_cont"}
    occ = [d for d in ev if d.get("map") == "occ"]
    assert occ and all(k in occ[0] for k in ("bce_before", "bce_after", "brier_before", "brier_after", "p_before",
                                             "p_after", "rate", "auc", "n"))
    assert any("city" in d for d in ev if d.get("map") == "te_ovl")
    assert any("cities" in d for d in ev)
    assert all("mae_gap" in d for d in ev if d.get("map") == "te_cont")
    assert all("ok_frac" in d for d in recs if "epoch_done" in d)
    ts = TS.load_teachers(str(tmp_path / "teachers_te.pt"), CFG)
    # the stored maps are the fitted ones (the metrics run after the fits and only read them)
    pl = {"occ": ts.platt_occ, "te_ovl": ts.platt_ovl, "te_ttc": ts.platt_ttc, "tm_off": ts.platt_off}
    fits = [d for d in recs if d.get("phase") == "platt" and "a" in d]
    assert fits
    for d in fits:
        assert float(pl[d["map"]].a[d["bin"]]) == float(np.float32(d["a"]))
        assert float(pl[d["map"]].b[d["bin"]]) == float(np.float32(d["b"]))
    meta = torch.load(tmp_path / "teachers_te.pt", map_location="cpu")["meta"]
    assert meta["read_rule"] == TS.READ_RULE and meta["git_mixed"] is False


# ------------------------------------------------------------------------------------------------- F08 decode_v6
def _anchor_tau(idx):
    from navsim.agents.para_ssr.modules.kinematics import bezier_xyyaw
    anc = torch.tensor(np.load(REPO / "resources/km_corridor/trajectory_anchors_256.npy"), dtype=torch.float32)[idx]
    tau = anc + (bezier_xyyaw(anc[..., :2], anc[..., 2]) - anc)                              # cf's arithmetic
    return anc, tau, tau[:, 0, :2].norm(dim=-1) / 0.5


def _old_heading(tau, xy, anc_h):
    from navsim.agents.para_ssr.modules.kinematics import bezier_xyyaw
    dh = bezier_xyyaw(xy, anc_h)[..., 2] - bezier_xyyaw(tau[..., :2], anc_h)[..., 2]
    return tau[..., 2] + torch.atan2(torch.sin(dh), torch.cos(dh))


@pytest.mark.parametrize("zv", [-1.0, -2.0])
def test_decel_stop_holds_the_heading(zv):
    """anchor 203 (curved, ~0.5 rad at 4 s): the decel stops the draft, and bezier_xyyaw's stationary fallback would
    hand the stopped pose the anchor's later heading (report 129 A61); it keeps the previous pose's v6 heading"""
    from navsim.agents.para_ssr.modules import torch_tracker as TTR
    anc, tau, v0 = _anchor_tau([203, 10])
    ctrl = torch.zeros(2, 12)
    ctrl[:, :6] = zv
    ctrl.requires_grad_(True)
    out = N.decode_v6(tau, ctrl, v0, anchor_h=anc[..., 2])
    xy, h = out["traj"][..., :2], out["traj"][..., 2]
    hold = ~N._bezier_moving(xy) & N._bezier_moving(tau[..., :2])
    assert bool(hold[0].any())
    old = _old_heading(tau, xy.detach(), anc[..., 2])
    hd = h.detach()
    for b in range(2):
        for k in range(8):
            if hold[b, k]:
                assert float(hd[b, k]) == (0.0 if k == 0 else float(hd[b, k - 1]))
            else:
                assert float(hd[b, k]) == float(old[b, k])
    assert float((hd[0] - old[0]).abs().max()) > 0.1                                         # ~ 16-18 deg here
    h.sum().backward()
    assert bool(torch.isfinite(ctrl.grad).all())
    init = torch.cat([v0[:, None], torch.zeros(2, 6)], -1).double()
    new_t = TTR._track(out["traj"].detach().double(), init, True)
    old_t = TTR._track(torch.cat([xy.detach(), old[..., None]], -1).double(), init, True)
    assert float((new_t[..., :2] - old_t[..., :2]).norm(dim=-1).max()) < 5e-3


def test_unedited_draft_keeps_the_base_heading_bitwise():
    anc, tau, v0 = _anchor_tau([203, 63, 102])
    out = N.decode_v6(tau, torch.zeros(3, 12), v0, anchor_h=anc[..., 2])
    assert torch.equal(out["traj"], tau)
