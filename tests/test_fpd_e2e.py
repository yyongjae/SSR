"""FPD-Drive refiner structure in the CK2 e2e system (ck_e2e2.arch 'fpd': ck/fpd_e2e.py on ck/fpd_net.py).  CPU only:
  cd /workspace/SSR && set -a && . tools/release/paths.env && set +a && PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_fpd_e2e.py

Covers: the two optional config keys (arch / fpd: parse, nested merge, refusals, validate rules, to_dict without them at
their defaults = the ck_e2e2.yaml key set, the arch 'ck2' parse / validate equal to v2_final's online2.py); the student
build (no global RNG, bev_grad_scale, score prior); FPDTeachers (load guards, run() = TeacherPair2 semantics with
the extra set scored separately, calibration, the DiT bank); FPDE2E.infer (CK2 keys / shapes, pool layout, column 0 =
v2's trajectory, token-keyed determinism, infer_select); FPDE2E.loss in warm-up / warmup_record / on-policy (finite,
gradients on every student parameter and on bev_embed, the recorded pool = the KD set, the label set with the
fallback rows' rule variants masked, the DiT targets GT + teacher targets, the teacher bank sampled with each teacher's
own DDIM steps, the DiT trainable inside an enclosing autocast region) and without teachers; the agent dispatch
(arch fpd -> FPDRefinerNet / FPDE2E; arch ck2 / off built exactly like v2_final's para_ssr_agent.py).
Inputs: the real warm-up rows of packed navtrain_train (CK2E2ETargetBuilder.warmup_row; the loss / agent tests skip
without them), synthetic v2 predictions on the repo's 256 anchors and synthetic teacher BEVs, two randomly initialised
FPD teacher runs written to a tmp dir (save_fpd_teacher + config.json + done.json), DDIM steps 2 for speed.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
from navsim.agents.para_ssr.ck import e2e_data2 as D2  # noqa: E402
from navsim.agents.para_ssr.ck import fpd_net as FN  # noqa: E402
from navsim.agents.para_ssr.ck import online2 as O  # noqa: E402
from navsim.agents.para_ssr.ck.constants import CK_KEYS, SUR_TERMS  # noqa: E402
from navsim.agents.para_ssr.ck.fpd_e2e import FPDE2E, FPDTeachers, build_student_fpd  # noqa: E402
from navsim.agents.para_ssr.ck.online import student_topk  # noqa: E402
from navsim.agents.para_ssr.refiner.adapters import BEV_H, BEV_W  # noqa: E402
from navsim.common import ssr_paths as _SP  # noqa: E402

torch.set_num_threads(min(8, torch.get_num_threads()))
ANCHORS = REPO / "resources/planning_vb/trajectory_anchors_256.npy"
BASE_COMMIT = "b902937"                    # v2_final, the branch point of v2_fpd
W = D2.WARMUP_DEFAULT
HAVE_WU = all((Path(W[k]) / f).is_file() for k, f in (("packed", "tokens.parquet"), ("raw_labels", "labels.npy"),
                                                       ("var_dir", "traj.npy"), ("var_dir", "index.npz")))
need_wu = pytest.mark.skipif(not HAVE_WU, reason="packed / raw256 / variant label files absent")
ROWS = [3, 7]
# the finished CK2 teacher runs of this server (their norm files: only the BEV-KD arm agent case reads them)
REAL_CK2 = {t: Path(_SP.SSR_TEACHER_CACHE) / "refiner" / t for t in ("det", "map")}
HAVE_REAL_CK2 = all((REAL_CK2[t] / f).is_file() for t, f in (("det", "norm.npz"), ("map", "norm_map.npz")))
STEPS = 2                                  # DDIM steps of the tests (speed); the default 8 is checked in the config
# the 'fpd' recipe of the base layout (validate)
FPD_KEYS = {"arch": "fpd", "warmup": {"n_var": 0}, "n_var_step": 80, "onpolicy_label": {"n_var": 80},
            "lambda_sur": 0.0, "lat_kd": {"balance": "fixed", "fixed": 0.0, "start_mb": 0}}
NOKD = {"use_teachers": False, "teacher_det_run": None, "teacher_map_run": None, "lambda_kd_score": 0.0,
        "kd_calib": {"enabled": False}}
# CK2 log keys of every CKE2E2.loss step (online2.py), which the FPD loss must log as well
CK2_LOG_KEYS = (["ck2/loss", "ck2/loss_nonfinite", "ck2/bce", "ck2/kd_score", "ck2/sur", "ck2/sur_nonfinite",
                 "ck2/lat_kd", "ck2/w_lat", "ck2/r", "ck2/phase", "ck2/mb", "ck2/n_gt", "ck2/wt_bce", "ck2/wt_sur",
                 "ck2/wt_kd_score", "ck2/wt_lat_kd", "ck2/bce_orig", "ck2/bce_var", "ck2/n_var_ok", "ck2/G_ok_frac",
                 "ck2/kd_ok_frac", "ck2/var_valid_frac", "ck2/e_abs", "ck2/n_cand", "ck2/variants_ms",
                 "ck2/teacher_ms", "ck2/student_ms", "ck2/rec_tokens", "ck2/total_ms"]
                + [f"ck2/bce_{k}" for k in CK_KEYS] + [f"ck2/t_{k}" for k in SUR_TERMS + ("P1_minus_P0",)])
CK2_LOG_KEYS_T = ["ck2/e_abs_T", "ck2/e_abs_det", "ck2/e_abs_map", "ck2/lat_disagree"]
FPD_LOG_KEYS = ["fpd/l_diff", "fpd/l_traj", "fpd/n_targets", "fpd/bank_safe_frac", "fpd/draw_dev", "fpd/n_pairs",
                "fpd/res_abs", "fpd/wt_diff", "fpd/wt_traj"]
INFER_KEYS = {"ck2_cand_idx": ((16,), torch.int64), "ck2_v2_final": ((16,), torch.float32),
              "ck2_v2_im": ((16,), torch.float32), "ck2_v2_sim": ((16, 5), torch.float32),
              "ck2_cand96": ((96, 8, 3), torch.float32), "ck2_valid96": ((96,), torch.bool),
              "ck2_score_logit": ((96, 5), torch.float32), "ck2_w_lat": ((96, 6), torch.float32),
              "ck2_e_lat": ((96, 6), torch.float32), "ck2_lat_traj": ((96, 8, 3), torch.float32)}


# ----------------------------------------------------------------------------------------------- fixtures / helpers
def _write_teacher(rd: Path, arm: str, seed: int, keys, prior=0.95, **cfg_over) -> str:
    """a finished FPD teacher run with a random net (scorer heads biased to `prior`: a safe bank)."""
    rd.mkdir(parents=True, exist_ok=True)
    net = FN.FPDRefinerNet(seed, steps=STEPS)
    if prior is not None:
        net.scorer.set_prior([prior] * len(CK_KEYS))
    cfg = {"trainer": FN.FPD_TEACHER_TRAINER, "arm": arm, "score_keys": list(keys), "ep_target": "official",
           "seed": seed, "steps": STEPS, "grid": 8, "stage": "test"}
    cfg.update(cfg_over)
    FN.save_fpd_teacher(rd, net, cfg)
    (rd / "config.json").write_text(json.dumps(cfg))
    (rd / "done.json").write_text(json.dumps({"done": True}))
    return str(rd)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """FPD teacher runs (det arm T, map arm M) + CK2-style teacher run dirs (config.json arm only: they name the
    teacher BEV caches, which these tests never read)."""
    root = tmp_path_factory.mktemp("fpd_runs")
    out = {"fdet": _write_teacher(root / "fdet", "T", 11, ["nc", "ttc", "ep", "comfort"]),
           "fmap": _write_teacher(root / "fmap", "M", 12, ["dac", "ep"])}
    for name, arm in (("cdet", "T"), ("cmap", "M")):
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps({"arm": arm, "trainer": "train_ck2"}))
        out[name] = str(root / name)
    return out


def fpd_dict(tmp: Path, runs, **over):
    d = {"enabled": True, "io_dir": str(Path(tmp) / "ck_e2e2"), "teacher_det_run": runs["cdet"],
         "teacher_map_run": runs["cmap"], "teacher_amp": False, "score_prior": "none", "grad_share_every": 0,
         "strict_rows": 0.0, "kd_calib": {"enabled": False},
         "fpd": {"teacher_det_run": runs["fdet"], "teacher_map_run": runs["fmap"], "steps": STEPS,
                 "mt_source": "online"}}
    d.update(json.loads(json.dumps(FPD_KEYS)))
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = dict(d[k], **v)
        else:
            d[k] = v
    return d


def fpd_cfg(tmp: Path, runs, **over) -> O.CKE2E2Config:
    return O.CKE2E2Config.from_any(fpd_dict(tmp, runs, **over))


def anchors() -> torch.Tensor:
    return torch.from_numpy(np.load(ANCHORS)).float()


def fake_predictions(B: int, seed: int = 0, requires_grad: bool = True):
    g = torch.Generator().manual_seed(seed)
    anc = anchors()
    off = 0.2 * torch.randn(B, 256, 8, 3, generator=g)
    final = torch.randn(B, 256, generator=g)
    im = torch.softmax(torch.randn(B, 256, generator=g), -1)
    sim = torch.sigmoid(torch.randn(B, 5, 256, generator=g))
    bev = torch.randn(B, BEV_H * BEV_W, 256, generator=g)
    bev = (bev - bev.mean(-1, keepdim=True)) / bev.std(-1, keepdim=True)
    bev.requires_grad_(requires_grad)
    top = final.argmax(-1)
    traj = (anc[None] + off)[torch.arange(B), top]
    return {"bev_embed": bev, "plan_final_rewards": final, "trajectory_offset": off, "trajectory_anchors": anc,
            "im_rewards": im, "sim_rewards": sim, "trajectory": traj}


def features(B: int = 2):
    st = torch.tensor([[0, 1, 0, 0, 6.0, 0.1, 0.3, 0.0], [1, 0, 0, 0, 9.0, -0.2, -0.6, 0.1],
                       [0, 0, 1, 0, 3.0, 0.0, 0.1, 0.0]], dtype=torch.float32)[:B]
    return {"status_feature": st}


_WU = {}


def wu_targets(rows=ROWS, seed: int = 0):
    """real warm-up targets of packed rows (ck2_*), ck_row, trajectory = the packed GT, random f16 teacher BEVs."""
    if "b" not in _WU:
        _WU["b"] = D2.CK2E2ETargetBuilder(ck_cfg={"enabled": True, "use_teachers": False})
    b = _WU["b"]
    rows = list(rows)
    wu = [b.warmup_row(r) for r in rows]
    t = {k: torch.stack([w[k] for w in wu]) for k in wu[0]}
    t["ck_row"] = torch.as_tensor(rows, dtype=torch.int64)
    t["trajectory"] = t["ck2_gt"].clone()
    g = torch.Generator().manual_seed(seed)
    for i in (0, 1):
        t[f"kd_bev_{i}"] = torch.randn(len(rows), 256, BEV_H, BEV_W, generator=g).half()
        t[f"kd_ok_{i}"] = torch.ones(len(rows), dtype=torch.bool)
    return t


def student(c, seed: int = 0):
    torch.manual_seed(seed)
    return build_student_fpd(c)


def spy(obj, name):
    """wrap a method and keep its last return value in obj._spy[name]."""
    fn = getattr(obj, name)
    obj.__dict__.setdefault("_spy", {})

    def w(*a, **k):
        out = fn(*a, **k)
        obj._spy[name] = out
        return out
    setattr(obj, name, w)


def check_grads(st, bev=None):
    for n, p in st.named_parameters():
        assert p.requires_grad, n
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), n
    if bev is not None:
        assert bev.grad is not None and bool(torch.isfinite(bev.grad).all()) and float(bev.grad.abs().sum()) > 0


def make_gen(io_dir: str, epoch: int, rows, n_rows: int, seed: int = 0):
    """a small generation (n_rows) with a 96-column pool per row (16 anchors x (parent + 5 jittered copies))."""
    rng = np.random.default_rng(seed)
    a = anchors().numpy()
    n = len(rows)
    par = np.stack([a[rng.choice(256, 16, replace=False)] for _ in range(n)])               # [n, 16, 8, 3]
    drw = par[:, :, None] + 0.3 * rng.standard_normal((n, 16, 5, 8, 3)).astype(np.float32)
    traj96 = np.concatenate([par[:, :, None], drw], 2).reshape(n, 96, 8, 3).astype(np.float32)
    labels = rng.random((n, 96, 9)).astype(np.float32).round(1)
    gen = D2.open_generation2(io_dir, epoch, n_rows=n_rows, mode="w+")
    D2.write_generation_rows2(gen, np.asarray(rows, np.int64), traj96, labels, np.ones((n, 96), bool),
                              np.ones((n, 96), bool))
    return traj96, labels


def _git_show(path: str):
    try:
        return subprocess.run(["git", "-C", str(REPO), "show", f"{BASE_COMMIT}:{path}"], capture_output=True,
                              text=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------- config
def test_config_defaults_and_to_dict(tmp_path, runs):
    import yaml
    c = O.CKE2E2Config.from_any(None)
    assert c.arch == "ck2" and c.fpd == {"teacher_det_run": "", "teacher_map_run": "", "teacher_which": "last",
                                         "n_draws": 5, "steps": 8, "grid": 8, "lambda_diff": 1.0, "lambda_traj": 1.0,
                                         "use_gt_target": True, "mt_k": 4, "mt_draws": 2, "mt_min_ade": 1.0,
                                         "mt_thr": 0.5, "mt_source": "bank", "mt_label_thr": 1.0}
    d = c.to_dict()
    assert "arch" not in d and "fpd" not in d                       # optional keys, recorded only off their defaults
    y = yaml.safe_load((REPO / "tools/ck/e2e2/ck_e2e2.yaml").read_text())
    assert set(y) == set(d)                                         # the yaml / dataclass key rule of the CK2 tests
    assert O.CKE2E2Config.from_any(d) == c and O.CKE2E2Config.from_any(c) == c
    assert O.CKE2E2Config.from_any({"arch": "ck2", "fpd": {}}).to_dict() == d          # explicit default: not recorded
    f = fpd_cfg(tmp_path, runs)
    df = f.to_dict()
    assert df["arch"] == "fpd" and df["fpd"]["teacher_det_run"] == runs["fdet"] and df["fpd"]["steps"] == STEPS
    assert set(df) == set(d) | {"arch", "fpd"}
    assert O.CKE2E2Config.from_any(df) == f and O.CKE2E2Config.from_any(f) == f


def test_config_parse_and_refusals(tmp_path):
    from omegaconf import OmegaConf
    d = OmegaConf.create({"enabled": True, "io_dir": str(tmp_path), "arch": "fpd",
                          "fpd": {"n_draws": "5", "use_gt_target": "false", "teacher_det_run": "null",
                                  "teacher_map_run": "/x/map", "mt_thr": "0.6", "steps": 12}})
    c = O.CKE2E2Config.from_any(d)
    assert c.arch == "fpd" and c.fpd["n_draws"] == 5 and c.fpd["use_gt_target"] is False
    assert c.fpd["teacher_det_run"] == "" and c.fpd["teacher_map_run"] == "/x/map" and c.fpd["mt_thr"] == 0.6
    assert c.fpd["steps"] == 12 and c.fpd["mt_k"] == 4 and c.fpd["lambda_diff"] == 1.0     # merged key-wise
    for bad, msg in (({"arch": "dit"}, "arch must be one of"), ({"arch": None}, "arch must be one of"),
                     ({"fpd": {"draws": 5}}, "unknown key"), ({"fpd": 3}, "must be a mapping"),
                     ({"fpd_x": {}}, "unknown key")):
        with pytest.raises(ValueError, match=msg):
            O.CKE2E2Config.from_any(bad)
    assert O.CKE2E2Config.from_any({"fpd": None}).fpd == O._d_fpd()


def _parent(**over):
    from types import SimpleNamespace
    d = dict(refiner_mode="off", ck_e2e={}, plan_anchor=True, bev_h=BEV_H, bev_w=BEV_W,
             pc_range=[-32.0, 0.0, -1.0, 32.0, 0.64 * BEV_H, 3.0], embed_dims=256)
    d.update(over)
    return SimpleNamespace(**d)


def test_validate_fpd_rules(tmp_path, runs):
    fpd_cfg(tmp_path, runs).validate(_parent())
    fpd_cfg(tmp_path, runs).validate(None)
    O.CKE2E2Config.from_any(fpd_dict(tmp_path, runs, enabled=False, warmup={"n_var": 32})).validate()   # off: none
    bad = [({"warmup": {"n_var": 32}}, "warmup.n_var must be 0"), ({"n_var_step": 32}, "n_var_step must be 80"),
           ({"onpolicy_label": {"n_var": 32}}, "onpolicy_label.n_var must be 80"),
           ({"lambda_sur": 1.0}, "lambda_sur must be 0"), ({"lat_kd": {"balance": "ema"}}, "lat_kd must be balance"),
           ({"lat_kd": {"fixed": 1.0}}, "lat_kd must be balance"), ({"fpd": {"n_draws": 4}}, "n_draws must be 5"),
           ({"fpd": {"steps": 0}}, "fpd.steps"), ({"fpd": {"steps": 1001}}, "fpd.steps"),
           ({"fpd": {"grid": 0}}, "fpd.grid"), ({"fpd": {"lambda_traj": -1.0}}, "lambda_traj"),
           ({"fpd": {"mt_k": 0}}, "mt_k >= 1"), ({"fpd": {"mt_thr": 1.5}}, "mt_thr"),
           ({"fpd": {"mt_min_ade": 0.0}}, "mt_min_ade > 0"),
           ({"fpd": {"teacher_det_run": ""}}, "fpd.teacher_det_run is required"),
           ({"fpd": {"teacher_map_run": None}}, "fpd.teacher_map_run is required"),
           ({"fpd": {"teacher_det_run": runs["fmap"]}}, "arm 'M', expected 'train_fpd_teacher' / 'T'"),
           ({"fpd": {"teacher_map_run": runs["cmap"]}}, "trainer 'train_ck2'"),
           ({"teacher_map_run": ""}, r"ck_e2e2\.teacher_map_run is required")]   # the CK2 runs name the BEV caches
    for over, msg in bad:
        with pytest.raises(ValueError, match=msg):
            fpd_cfg(tmp_path, runs, **over).validate(_parent())
    # no-teacher mode: the FPD teacher runs must be null too, and the GT target is then the DiT's only target
    nokd = dict(NOKD, fpd={"teacher_det_run": None, "teacher_map_run": None})
    fpd_cfg(tmp_path, runs, **nokd).validate(None)
    with pytest.raises(ValueError, match="must be null without teachers"):
        fpd_cfg(tmp_path, runs, **dict(NOKD, fpd={"teacher_map_run": None})).validate(None)
    with pytest.raises(ValueError, match="use_gt_target false needs the teachers"):
        fpd_cfg(tmp_path, runs, **dict(nokd, fpd={"teacher_det_run": None, "teacher_map_run": None,
                                                  "use_gt_target": False})).validate(None)
    # arch ck2 reads no fpd option
    ck2 = {"enabled": True, "io_dir": str(tmp_path / "io"), "teacher_det_run": runs["cdet"],
           "teacher_map_run": runs["cmap"]}
    O.CKE2E2Config.from_any(ck2).validate(_parent())
    with pytest.raises(ValueError, match="fpd is only read with arch 'fpd'"):
        O.CKE2E2Config.from_any(dict(ck2, fpd={"steps": 4})).validate(_parent())


def test_ck2_config_unchanged_vs_v2_final(tmp_path, runs):
    """arch 'ck2' (every existing config): from_any / to_dict / validate of the v2_final online2.py (git BASE_COMMIT)
    and of this one agree on a set of configs (dict / DictConfig / instance, nested merges, refusals)."""
    from omegaconf import OmegaConf
    src = _git_show("navsim/agents/para_ssr/ck/online2.py")
    if src is None:
        pytest.skip(f"git show {BASE_COMMIT} unavailable")
    p = tmp_path / "online2_base.py"
    p.write_text(src)
    old = _load_module("navsim.agents.para_ssr.ck._online2_v2_final_test", p)
    base = {"enabled": True, "io_dir": str(tmp_path / "io"), "teacher_det_run": runs["cdet"],
            "teacher_map_run": runs["cmap"]}
    cfgs = [None, {}, base, dict(base, lat_kd={"start_mb": 7}, bev_kd={"enabled": "true", "teachers": "[det,map]"}),
            dict(base, use_teachers=False, **{k: v for k, v in NOKD.items() if k != "use_teachers"},
                 lat_kd={"balance": "fixed", "fixed": 0.0}),
            dict(base, infer_select={"beta": 0.5, "set": "all", "lat_mode": "on"}, kd_sources=["det"],
                 kd_score_keys=["nc", "ttc"]),
            dict(base, kd_calib={"det": "/x/a.json"}), dict(base, n_var_step=81), dict(base, io_dir="rel"),
            dict(base, teacher_det_run=runs["cmap"]), dict(base, warmup={"n_var": 0}), {"nope": 1},
            OmegaConf.create(dict(base, variants={"speeds": "[-1.0,-0.5,0.5]"}))]
    for d in cfgs:
        res = []
        for mod in (old, O):
            try:
                c = mod.CKE2E2Config.from_any(d)
                td = c.to_dict()
                try:
                    c.validate(_parent())
                    v = "ok"
                except ValueError as e:
                    v = f"refused: {e}"
                res.append((td, v))
            except ValueError as e:
                res.append(("parse refused", str(e)))
        assert res[0] == res[1], (d, res)
    assert old.CKE2E2Config().to_dict() == O.CKE2E2Config().to_dict()


# ----------------------------------------------------------------------------------------------- student
def test_build_student_rng_prior_and_init(tmp_path, runs, monkeypatch):
    c = fpd_cfg(tmp_path, runs, bev_grad_scale=0.5)
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    net = build_student_fpd(c)
    assert torch.equal(before, torch.get_rng_state())                    # no global RNG consumed
    assert isinstance(net, FN.FPDRefinerNet) and net.tok.bev_grad_scale == 0.5
    assert net.dit.num_inference_steps == STEPS and net.tok.grid == 8
    assert all(p.requires_grad for p in net.parameters())               # nothing frozen
    rep = net.init_report
    assert rep["arch"] == "fpd" and rep["frozen"] == [] and rep["score_prior"] is None
    assert rep["param_counts"] == net.param_counts()
    sd0, sd1 = net.state_dict(), build_student_fpd(c).state_dict()
    assert list(sd0) == list(sd1) and all(torch.equal(sd0[k], sd1[k]) for k in sd0)      # seed-deterministic
    sd2 = build_student_fpd(fpd_cfg(tmp_path, runs, seed=1)).state_dict()
    assert any(not torch.equal(sd0[k], sd2[k]) for k in sd0)
    # score prior: the build_student_ck2 rule (online2.score_prior_of), head bias = logit(prior)
    prior = np.array([0.97, 0.9, 0.6, 0.95, 0.99])
    import navsim.agents.para_ssr.ck.fpd_e2e as FE
    monkeypatch.setattr(FE, "score_prior_of", lambda cfg: (prior, "test"))
    net_p = build_student_fpd(fpd_cfg(tmp_path, runs, score_prior="ck2_warmup"))
    got = np.array([float(torch.sigmoid(h[-1].bias.detach().double())) for h in net_p.scorer.heads])
    np.testing.assert_allclose(got, prior, rtol=1e-6)
    assert net_p.init_report["score_prior"] == [float(v) for v in prior]
    assert net_p.init_report["score_prior_source"] == "test"
    net_n = build_student_fpd(fpd_cfg(tmp_path, runs, score_prior="ck2_warmup"), apply_prior=False)
    assert all(torch.equal(a[-1].bias, b[-1].bias) for a, b in zip(net_n.scorer.heads, net.scorer.heads))
    assert net_n.init_report["score_prior"] is None and net_n.init_report["score_prior_source"] is None


@need_wu
def test_score_prior_ck2_warmup_small(tmp_path, runs):
    c = fpd_cfg(tmp_path, runs, score_prior="ck2_warmup", score_prior_rows=20)
    net = build_student_fpd(c)
    want = D2.ck2_warmup_prior(c, n_max=20)                              # warm-up mix = 32 anchors (n_var 0)
    got = np.array([float(torch.sigmoid(h[-1].bias.detach().double())) for h in net.scorer.heads])
    np.testing.assert_allclose(got, np.clip(want, 1e-4, 1 - 1e-4), atol=1e-6)
    assert net.init_report["score_prior_source"] == "ck2_warmup"


# ----------------------------------------------------------------------------------------------- teachers
def _cands(B=2, K=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = anchors()
    idx = torch.randint(0, 256, (B, K), generator=g)
    return a[idx] + 0.05 * torch.randn(idx.shape + (8, 3), generator=g)


def _bevs(B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, 256, BEV_H, BEV_W, generator=g).half() for _ in range(2)]


def test_teachers_load_guards(tmp_path, runs):
    tp = FPDTeachers(runs["fdet"], runs["fmap"], "last", "cpu", amp=False, ep_target="official")
    assert not isinstance(tp, torch.nn.Module)
    for net, arm in ((tp.det, "T"), (tp.map, "M")):
        assert not net.training and all(not p.requires_grad for p in net.parameters())
        assert net.init_report["arm"] == arm and len(net.init_report["ckpt_sha16"]) == 16
    assert tp.ckpt["det"] == tp.det.init_report["ckpt_sha16"] and tp.ep_target == {"det": "official", "map": "official"}
    assert tp.score_keys == {"det": ("nc", "ttc", "ep", "comfort"), "map": ("dac", "ep")}
    assert tp.bank_sources == [("det",), ("map",), ("det", "map"), ("det",), ("det",)]
    with pytest.raises(ValueError, match="arm"):
        FPDTeachers(runs["fmap"], runs["fdet"])
    with pytest.raises(ValueError, match="trainer 'train_ck2'"):
        FPDTeachers(runs["cdet"], runs["fmap"])
    nodone = Path(_write_teacher(tmp_path / "nodone", "T", 3, ["nc"]))
    (nodone / "done.json").unlink()
    with pytest.raises(ValueError, match="done.json"):
        FPDTeachers(str(nodone), runs["fmap"])
    FPDTeachers(str(nodone), runs["fmap"], require_done=False)
    dec = _write_teacher(tmp_path / "dec", "T", 4, ["nc"], ep_target="decoupled")
    with pytest.raises(ValueError, match="ep_target"):
        FPDTeachers(dec, runs["fmap"], ep_target="official")
    FPDTeachers(dec, runs["fmap"], ep_target=None)                       # teacher_ep_check false
    with pytest.raises(ValueError, match="sources"):
        FPDTeachers(runs["fdet"], runs["fmap"], sources=())


def test_teachers_run_semantics(runs):
    tp = FPDTeachers(runs["fdet"], runs["fmap"], amp=False)
    bd, bm = _bevs()
    st = features(2)["status_feature"]
    cand, extra = _cands(2, 12, 0), _cands(2, 7, 1)
    ok_d, ok_m = torch.tensor([True, True]), torch.tensor([False, True])             # token 0 without a MAP BEV
    cok = torch.ones(2, 12, dtype=torch.bool)
    cok[0, 3] = False
    T = tp.run(bd, ok_d, bm, ok_m, cand, st, cand_ok=cok)
    assert set(T) == {"kd_score_prob", "kd_e_lat", "kd_ok", "e_det", "e_map", "p_det", "p_map", "z_det", "z_map",
                      "ok_det", "ok_map"}
    with torch.no_grad():
        z = {n: net.score(cand, net.scene(b.float()), st) for n, net, b in (("det", tp.det, bd), ("map", tp.map, bm))}
    torch.testing.assert_close(T["z_det"], z["det"], rtol=0, atol=0)
    torch.testing.assert_close(T["z_map"], z["map"], rtol=0, atol=0)
    pd, pm = torch.sigmoid(z["det"]), torch.sigmoid(z["map"])
    p = T["kd_score_prob"]
    torch.testing.assert_close(p[1, :, [0, 3]], pd[1, :, [0, 3]])                 # nc / ttc <- DET
    torch.testing.assert_close(p[1, :, 1], pm[1, :, 1])                           # dac <- MAP
    torch.testing.assert_close(p[1, :, 2], 0.5 * (pd[1, :, 2] + pm[1, :, 2]))     # ep <- mean
    torch.testing.assert_close(p[0], pd[0])                                       # no MAP BEV: DET everywhere
    assert not T["kd_ok"][0].any() and T["kd_ok"][1].all()                        # no MAP BEV -> KD off
    assert T["ok_det"].all() and not T["ok_map"][0].any()
    for k in ("kd_e_lat", "e_det", "e_map"):
        assert T[k].shape == (2, 12, 6) and float(T[k].abs().max()) == 0.0        # no lateral head
    cok1 = cok.clone()
    T1 = tp.run(bd, ok_d, bm, torch.tensor([True, True]), cand, st, cand_ok=cok1)
    assert not T1["kd_ok"][0, 3] and T1["kd_ok"].sum() == 23
    # extra = a SEPARATE set (the scorer attends across its set): KD outputs unchanged, z_*_x = a call on extra alone
    Tx = tp.run(bd, ok_d, bm, ok_m, cand, st, cand_ok=cok, extra=extra)
    for k in T:
        assert torch.equal(Tx[k], T[k]), k
    with torch.no_grad():
        zx = tp.det.score(extra, tp.det.scene(bd.float()), st)
    torch.testing.assert_close(Tx["z_det_x"], zx, rtol=0, atol=0)
    assert Tx["z_map_x"].shape == (2, 7, 5) and Tx["ok_det_x"].all() and not Tx["ok_map_x"][0].any()
    # calibration: identity reproduces the raw logits; a z + b enters the probabilities, z_* stay raw
    tp.set_calib_arrays({t: (np.ones(5), np.zeros(5)) for t in ("det", "map")})
    Ti = tp.run(bd, ok_d, bm, ok_m, cand, st, cand_ok=cok)
    for k in T:
        assert torch.equal(Ti[k], T[k]), k
    a, b = np.array([2.0, 1.0, 0.5, 1.0, 1.0]), np.array([0.3, -0.2, 0.0, 0.1, 0.0])
    tp.set_calib_arrays({"det": (a, b), "map": (np.ones(5), np.zeros(5))})
    Tc = tp.run(bd, ok_d, bm, ok_m, cand, st, cand_ok=cok)
    torch.testing.assert_close(Tc["p_det"], torch.sigmoid(z["det"] * torch.tensor(a).float()
                                                          + torch.tensor(b).float()))
    assert torch.equal(Tc["z_det"], T["z_det"]) and torch.equal(Tc["p_map"], T["p_map"])
    # sources: one teacher -> every key from it
    tps = FPDTeachers(runs["fdet"], runs["fmap"], amp=False, sources=["map"])
    Ts = tps.run(bd, ok_d, bm, torch.tensor([True, True]), cand, st)
    torch.testing.assert_close(Ts["kd_score_prob"], torch.sigmoid(z["map"]))


def test_teachers_bank(runs):
    tp = FPDTeachers(runs["fdet"], runs["fmap"], amp=False)
    bd, bm = _bevs()
    st = features(2)["status_feature"]
    par = _cands(2, 16, 3)
    ok = torch.ones(2, dtype=torch.bool)
    D = 2
    bank, prob, bok = tp.bank(bd, ok, bm, ok, par, st, ["tokA", "tokB"], D, seed=0, epoch=3, steps=STEPS)
    assert bank.shape == (2, 2 * 16 * D, 8, 3) and prob.shape == (2, 2 * 16 * D, 5) and bok.shape == (2, 64)
    assert bok.all() and torch.isfinite(bank).all() and ((prob >= 0) & (prob <= 1)).all()
    # each teacher's own DiT on its own BEV, the bank noise (stream fpd.teacher_bank); DET draws first
    noise = FN.token_noise(["tokA", "tokB"], (16, D, 8, 3), 0, 3, "fpd.teacher_bank")
    with torch.no_grad():
        cd, cm = tp.det.scene(bd.float()), tp.map.scene(bm.float())
        dd = tp.det.sample_draws(par, cd, st, noise, STEPS).reshape(2, 16 * D, 8, 3)
        dm = tp.map.sample_draws(par, cm, st, noise, STEPS).reshape(2, 16 * D, 8, 3)
        pdet, pmap = torch.sigmoid(tp.det.score(bank, cd, st)), torch.sigmoid(tp.map.score(bank, cm, st))
    torch.testing.assert_close(bank, torch.cat([dd, dm], 1), rtol=0, atol=0)
    torch.testing.assert_close(prob[..., [0, 3, 4]], pdet[..., [0, 3, 4]])        # nc / ttc / comfort: DET trained
    torch.testing.assert_close(prob[..., 1], pmap[..., 1])                        # dac: MAP
    torch.testing.assert_close(prob[..., 2], 0.5 * (pdet[..., 2] + pmap[..., 2]))  # ep: both trained -> mean
    b2, p2, o2 = tp.bank(bd, ok, bm, ok, par, st, ["tokA", "tokB"], D, seed=0, epoch=3, steps=STEPS)
    assert torch.equal(b2, bank) and torch.equal(p2, prob) and torch.equal(o2, bok)        # deterministic
    b3, _, _ = tp.bank(bd, ok, bm, ok, par, st, ["tokA", "tokB"], D, seed=0, epoch=4, steps=STEPS)
    assert not torch.equal(b3, bank)                                                       # epoch-keyed noise
    _, _, o4 = tp.bank(bd, ok, bm, torch.tensor([True, False]), par, st, ["tokA", "tokB"], D, seed=0, epoch=3,
                       steps=STEPS)
    assert o4[0].all() and not o4[1].any()                                                 # no MAP BEV -> not ok
    tps = FPDTeachers(runs["fdet"], runs["fmap"], amp=False, sources=["det"])
    bs, ps, _ = tps.bank(bd, ok, bm, ok, par, st, ["tokA", "tokB"], D, seed=0, epoch=3, steps=STEPS)
    assert bs.shape == (2, 16 * D, 8, 3) and torch.equal(bs, dd) and tps.bank_sources == [("det",)] * 5
    with torch.no_grad():
        torch.testing.assert_close(ps, torch.sigmoid(tp.det.score(dd, cd, st)))


# ----------------------------------------------------------------------------------------------- inference
def test_infer_keys_layout_and_determinism(tmp_path, runs):
    c = fpd_cfg(tmp_path, runs)
    ck = FPDE2E(c)
    assert FPDE2E.needs_tokens is True and ck.infer_tokens is None and not getattr(O.CKE2E2, "needs_tokens", False)
    st = student(c).eval()
    f = features(2)
    pred = fake_predictions(2, seed=6, requires_grad=False)
    ck.infer_tokens = ["tokA", "tokB"]
    out = ck.infer(st, f, pred)
    assert ck.infer_tokens is None                                                # consumed
    assert set(out) == set(INFER_KEYS)
    for k, (shape, dt) in INFER_KEYS.items():
        assert out[k].shape == (2,) + shape and out[k].dtype == dt, (k, out[k].shape, out[k].dtype)
        assert torch.isfinite(out[k].float()).all(), k
    top = student_topk(pred, 16)
    assert torch.equal(out["ck2_cand96"][:, ::6], top["cand"]) and torch.equal(out["ck2_cand_idx"], top["idx"])
    assert torch.equal(out["ck2_cand96"][:, 0], pred["trajectory"])               # column 0 = v2's trajectory
    for k, kk in (("ck2_v2_final", "final"), ("ck2_v2_im", "im"), ("ck2_v2_sim", "sim")):
        assert torch.equal(out[k], top[kk]), k
    assert out["ck2_valid96"].all() and torch.equal(out["ck2_lat_traj"], out["ck2_cand96"])
    assert float(out["ck2_w_lat"].abs().max()) == 0.0 and float(out["ck2_e_lat"].abs().max()) == 0.0
    # the draws: evaluation noise (epoch -1) keyed by the tokens; parent + residual, one scorer pass over the 96
    noise = FN.token_noise(["tokA", "tokB"], (16, 5, 8, 3), c.seed, -1, "fpd.noise")
    with torch.no_grad():
        from navsim.agents.para_ssr.ck.online import bev_sgrid
        ctx = st.scene(bev_sgrid(pred["bev_embed"]))
        d = st.sample_draws(top["cand"], ctx, f["status_feature"], noise, STEPS)
        torch.testing.assert_close(out["ck2_cand96"], FN.pool96(top["cand"], d), rtol=0, atol=0)
        torch.testing.assert_close(out["ck2_score_logit"], st.score(out["ck2_cand96"], ctx, f["status_feature"]),
                                   rtol=0, atol=0)
    assert float((out["ck2_cand96"][:, 1:6] - out["ck2_cand96"][:, :1]).abs().max()) > 0.01    # the draws moved
    # same tokens -> identical outputs; other tokens -> other draws, same parents; no tokens -> status keys
    ck.infer_tokens = ["tokA", "tokB"]
    out2 = ck.infer(st, f, pred)
    assert all(torch.equal(out[k], out2[k]) for k in out)
    ck.infer_tokens = ["tokC", "tokB"]
    out3 = ck.infer(st, f, pred)
    assert not torch.equal(out3["ck2_cand96"][0], out["ck2_cand96"][0])
    assert torch.equal(out3["ck2_cand96"][1], out["ck2_cand96"][1])
    assert torch.equal(out3["ck2_cand96"][:, ::6], out["ck2_cand96"][:, ::6])
    ck.infer_tokens = ["only-one"]                                               # not a list of length B: status keys
    o_s1 = ck.infer(st, f, pred)
    o_s2 = ck.infer(st, f, pred)
    assert ck.infer_tokens is None and all(torch.equal(o_s1[k], o_s2[k]) for k in o_s1)
    n_s = FN.token_noise(FN.status_keys(f["status_feature"]), (16, 5, 8, 3), c.seed, -1, "fpd.noise")
    with torch.no_grad():
        d_s = st.sample_draws(top["cand"], ctx, f["status_feature"], n_s, STEPS)
    torch.testing.assert_close(o_s1["ck2_cand96"], FN.pool96(top["cand"], d_s), rtol=0, atol=0)


def test_infer_select_optional(tmp_path, runs):
    c = fpd_cfg(tmp_path, runs, infer_select={"beta": 0.0, "set": "all", "lat_mode": "off", "rule": "wote"})
    ck = FPDE2E(c)
    st = student(c).eval()
    pred = fake_predictions(2, seed=9, requires_grad=False)
    out = ck.infer(st, features(2), pred)
    assert out["ck2_sel_idx"].shape == (2,) and out["ck2_traj"].shape == (2, 8, 3)
    # beta 0 = v2 only; the draws inherit their parent's v2 score and ties go to the identity -> column 0
    assert out["ck2_sel_idx"].tolist() == [0, 0] and torch.equal(out["ck2_traj"], pred["trajectory"])
    c2 = fpd_cfg(tmp_path, runs, infer_select={"beta": 1.0, "set": "all", "lat_mode": "on"})
    o2 = FPDE2E(c2).infer(st, features(2), pred)
    from navsim.agents.para_ssr.ck import select2 as S2
    want = S2.select96_pdms(o2["ck2_score_logit"], o2["ck2_valid96"])
    assert o2["ck2_sel_idx"].tolist() == want.tolist()
    ar = torch.arange(2)
    assert torch.equal(o2["ck2_traj"], o2["ck2_lat_traj"][ar, o2["ck2_sel_idx"]])


# ----------------------------------------------------------------------------------------------- loss
@need_wu
def test_warmup_loss_backward_logs_and_targets(tmp_path, runs):
    c = fpd_cfg(tmp_path, runs, grad_share_every=1, kd_calib={"enabled": True})
    ck = FPDE2E(c)
    st = student(c)
    spy(ck, "warmup_batch")
    spy(ck, "dit_targets")
    tp = ck.teachers(torch.device("cpu"))
    spy(tp, "run")
    spy(tp, "bank")
    f, t = features(2), wu_targets()
    pred = fake_predictions(2, seed=0)
    v2 = (pred["bev_embed"] ** 2).mean()
    torch.manual_seed(5)
    rng0 = torch.get_rng_state().clone()
    loss, logs = ck.loss(st, f, t, pred, v2_loss=v2)
    assert torch.equal(rng0, torch.get_rng_state())                     # no global RNG consumed by the step
    assert torch.isfinite(loss) and float(logs["ck2/loss"]) > 0 and ck.mb == 1
    for k, v in logs.items():
        assert torch.isfinite(v).all(), k
    missing = [k for k in CK2_LOG_KEYS + CK2_LOG_KEYS_T + FPD_LOG_KEYS + ["ck2/ratio_v2", "ck2/gshare",
                                                                           "gnorm/bev_ck", "ck2/cal_pairs"]
               if k not in logs]
    assert not missing, missing
    assert float(logs["ck2/phase"]) == 0.0 and float(logs["ck2/n_cand"]) == 32.0 and float(logs["ck2/r"]) == 1.0
    for k in ("ck2/sur", "ck2/lat_kd", "ck2/w_lat", "ck2/wt_sur", "ck2/wt_lat_kd", "ck2/e_abs", "fpd/draw_dev",
              "ck2/rec_tokens") + tuple(f"ck2/t_{k}" for k in SUR_TERMS):
        assert float(logs[k]) == 0.0, k
    assert float(logs["ck2/kd_score"]) > 0 and float(logs["fpd/l_diff"]) > 0 and float(logs["fpd/l_traj"]) > 0
    S = ck._spy["warmup_batch"]
    assert S["cand"].shape == (2, 32, 8, 3) and S["extra"] is None                     # 32 raw anchors, n_var 0
    assert tp._spy["run"]["kd_ok"].shape == (2, 32)
    bank = tp._spy["bank"]
    assert bank[0].shape == (2, 2 * 16 * 2, 8, 3)
    tg, tm, _ = ck._spy["dit_targets"]
    assert tg.shape == (2, 4, 8, 3) and tm[:, 0].all()                                 # GT first
    torch.testing.assert_close(tg[:, 0], t["trajectory"])
    assert int(tm[:, 1:].sum()) > 0                                                    # teacher targets selected
    assert float(logs["fpd/n_targets"]) == float(tm.sum()) / 2 and float(logs["ck2/n_gt"]) == 2.0
    expect = sum(float(logs[k]) for k in ("ck2/wt_bce", "ck2/wt_kd_score", "fpd/wt_diff", "fpd/wt_traj"))
    assert abs(float(loss.detach()) - expect) < 1e-4 * max(1.0, abs(expect))
    loss.backward()
    check_grads(st, pred["bev_embed"])


def test_select_targets_never_reselects_a_column():
    """an already selected column is never picked again (fpd_net masks the selected set): with one safe bank candidate
    select_targets returns it once at min_ade 0 too (validate still refuses fpd.mt_min_ade 0)."""
    g = torch.Generator().manual_seed(0)
    bank = torch.randn(1, 6, 8, 3, generator=g)
    prob = torch.full((1, 6, 5), 0.9)
    prob[0, 1:, 0] = 0.1                                                       # only column 0 passes P(NC) >= thr
    ok = torch.ones(1, 6, dtype=torch.bool)
    _, m1 = FN.select_targets(bank, prob, ok, k_max=3, min_ade=1e-6)
    _, m0 = FN.select_targets(bank, prob, ok, k_max=3, min_ade=0.0)
    assert int(m1.sum()) == 1 and int(m0.sum()) == 1


@need_wu
def test_teacher_bank_uses_the_teachers_own_ddim_steps(tmp_path, runs):
    """the multi-target bank of the loss is sampled with each FPD teacher's own DDIM steps (config.json steps: the
    sampler its scorer was trained / evaluated on), not with the student's fpd.steps."""
    fdet = _write_teacher(tmp_path / "fdet3", "T", 11, ["nc", "ttc", "ep", "comfort"], steps=3)
    fmap = _write_teacher(tmp_path / "fmap3", "M", 12, ["dac", "ep"], steps=3)
    c = fpd_cfg(tmp_path, runs, fpd={"teacher_det_run": fdet, "teacher_map_run": fmap})      # student: STEPS
    ck = FPDE2E(c)
    st = student(c)
    tp = ck.teachers(torch.device("cpu"))
    assert tp.det.dit.num_inference_steps == 3 and tp.map.dit.num_inference_steps == 3
    assert st.dit.num_inference_steps == STEPS != 3
    spy(tp, "bank")
    f, t = features(2), wu_targets()
    pred = fake_predictions(2, seed=11)
    ck.loss(st, f, t, pred)
    got = tp._spy["bank"]
    args = (t["kd_bev_0"], t["kd_ok_0"], t["kd_bev_1"], t["kd_ok_1"], student_topk(pred, 16)["cand"],
            f["status_feature"], ck.tokens_of(t["ck_row"]), c.fpd["mt_draws"])
    own = tp.bank(*args, seed=c.seed, epoch=0)
    stud = tp.bank(*args, seed=c.seed, epoch=0, steps=STEPS)
    assert all(torch.equal(a, b) for a, b in zip(got, own))
    assert not torch.equal(own[0], stud[0])


@need_wu
def test_every_parameter_moves_the_loss(tmp_path, runs):
    """with the zero-initialised layers (adaLN-Zero gates, final modulation, velocity head) perturbed, every student
    parameter gets a NON-zero gradient from the warm-up loss (scorer: BCE / KD, DiT: L_diff / L_traj, tokens: both)."""
    c = fpd_cfg(tmp_path, runs)
    ck = FPDE2E(c)
    st = student(c)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for n, p in st.named_parameters():
            if float(p.abs().max()) == 0.0:
                p.add_(0.02 * torch.randn(p.shape, generator=g))
    loss, _ = ck.loss(st, features(2), wu_targets(), fake_predictions(2, seed=1))
    loss.backward()
    zero = [n for n, p in st.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0.0]
    assert not zero, zero


@need_wu
def test_warmup_record_records_the_pool(tmp_path, runs):
    from navsim.agents.para_ssr.ck import e2e_data as ED
    c = fpd_cfg(tmp_path, runs, rec_chunk_tokens=2)
    ck = FPDE2E(c)
    st = student(c)
    ck.epoch, ck.epoch_frac, ck.gstep = 4, 4.0, 17
    assert ck.phase(4) == "warmup_record" and ck.recording(4)
    ck.recorder = D2.CandRecorder2(c.io_dir, 0, 1, 4, c.rec_chunk_tokens)
    spy(ck, "draws")
    f, t = features(2), wu_targets()
    pred = fake_predictions(2, seed=10)
    loss, logs = ck.loss(st, f, t, pred)
    assert torch.isfinite(loss) and float(logs["ck2/phase"]) == 1.0 and float(logs["ck2/rec_tokens"]) == 2.0
    assert float(logs["ck2/n_cand"]) == 32.0 and float(logs["fpd/draw_dev"]) > 0     # warm-up loss + the pool
    rec = ck.recorder
    assert rec.n_tokens == 2 and len(rec.chunks) == 1
    a = D2.read_rec2_chunk(Path(ED.rec_rank_dir(c.io_dir, 4, 0)) / rec.chunks[0])
    top = student_topk(pred, 16)
    d, traj96, valid96 = ck._spy["draws"]
    assert a["row"].tolist() == ROWS and a["gstep"].tolist() == [17, 17] and int(a["fmt"]) == 2
    assert np.array_equal(a["traj"], traj96.numpy()) and np.array_equal(a["valid"], valid96.numpy())
    assert np.array_equal(a["traj"][:, ::6], top["cand"].numpy())                     # v 0 = the parent bitwise
    assert np.array_equal(a["traj"].reshape(2, 16, 6, 8, 3)[:, :, 1:], d.numpy())     # v 1..5 = the DiT draws
    assert np.array_equal(a["cand_idx"], top["idx"].numpy().astype(np.int16))
    assert np.array_equal(a["v2_final"], top["final"].numpy())
    noise = FN.token_noise(ck.tokens_of(t["ck_row"]), (16, 5, 8, 3), c.seed, 4, "fpd.noise")
    with torch.no_grad():
        from navsim.agents.para_ssr.ck.online import bev_sgrid
        d_ref = st.sample_draws(top["cand"], st.scene(bev_sgrid(pred["bev_embed"])), f["status_feature"], noise,
                                STEPS)
    assert torch.equal(d, d_ref)                                                       # training noise keyed by epoch
    loss.backward()
    check_grads(st, pred["bev_embed"])


def test_draws_differ_per_epoch(tmp_path, runs):
    c = fpd_cfg(tmp_path, runs)
    ck = FPDE2E(c)
    st = student(c)
    pred = fake_predictions(2, seed=2, requires_grad=False)
    from navsim.agents.para_ssr.ck.online import bev_sgrid
    par = student_topk(pred, 16)["cand"]
    with torch.no_grad():
        ctx = st.scene(bev_sgrid(pred["bev_embed"]))
        st_ = features(2)["status_feature"]
        d5, t5, _ = ck.draws(st, par, ctx, st_, ["a", "b"], 5)
        d5b, _, _ = ck.draws(st, par, ctx, st_, ["a", "b"], 5)
        d6, t6, _ = ck.draws(st, par, ctx, st_, ["a", "b"], 6)
    assert torch.equal(d5, d5b) and not torch.equal(d5, d6)
    assert torch.equal(t5[:, ::6], par) and torch.equal(t6[:, ::6], par)
    with pytest.raises(NotImplementedError, match="DiT pool"):                       # no CK2 rule variants
        ck.current_batch(pred, None, None, None, None, None)


def test_draws_keep_the_dit_trainable_inside_an_enclosing_autocast(tmp_path, runs):
    """the no-grad training draws run with autocast OFF: inside one enclosing autocast region (Lightning 16-mixed +
    ck_e2e2.amp) a no-grad DiT pass would otherwise leave weight casts without autograd history in autocast's cache,
    the later dit_loss would reuse them and no gradient would reach the DiT (DDP find_unused_parameters False: a
    crash).  CPU bf16 autocast stands in for the CUDA one; the draws equal the fp32 ones bitwise."""
    c = fpd_cfg(tmp_path, runs)
    ck = FPDE2E(c)
    st = student(c)
    pred = fake_predictions(2, seed=3)
    s = features(2)["status_feature"]
    par = student_topk(pred, 16)["cand"]
    from navsim.agents.para_ssr.ck.online import bev_sgrid
    with torch.no_grad():
        d32, _, _ = ck.draws(st, par, st.scene(bev_sgrid(pred["bev_embed"])), s, ["a", "b"], 0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ctx = st.scene(bev_sgrid(pred["bev_embed"]))
        d, _, _ = ck.draws(st, par, ctx, s, ["a", "b"], 0)
        l_diff, l_traj, _ = FN.dit_loss(st, par, pred["trajectory"][:, None].detach(),
                                        torch.ones(2, 1, dtype=torch.bool), ctx, s, ["a", "b"], 0, 0, 0)
    assert d.dtype == torch.float32 and torch.equal(d, d32)
    (l_diff.float() + l_traj.float()).backward()
    assert not [n for n, p in st.dit.named_parameters() if p.grad is None]


def _onpolicy(tmp_path, runs, **over):
    c = fpd_cfg(tmp_path, runs, **over)
    ck = FPDE2E(c, max_epochs=30)
    ck._labels = D2.LabelStore2(c.io_dir, n_rows=16, packed=c.warmup["packed"], var_sampling=c.var_sampling,
                                ep_target=c.ep_target)
    gtraj, glab = make_gen(c.io_dir, 4, [ROWS[0]], 16)
    return c, ck, gtraj, glab


@need_wu
def test_onpolicy_loss_label_set_kd_set_and_record(tmp_path, runs):
    c, ck, gtraj, glab = _onpolicy(tmp_path, runs, rec_chunk_tokens=1, grad_share_every=1, kd_calib={"enabled": True})
    st = student(c)
    ck.epoch, ck.epoch_frac, ck.gstep = 5, 5.5, 3
    ck.recorder = D2.CandRecorder2(c.io_dir, 0, 1, 5, c.rec_chunk_tokens)
    spy(ck, "onpolicy_batch")
    spy(ck, "draws")
    tp = ck.teachers(torch.device("cpu"))
    spy(tp, "run")
    f, t = features(2), wu_targets()                    # row 3 has generation 4, row 7 none -> fallback
    pred = fake_predictions(2, seed=3)
    v2 = (pred["bev_embed"] ** 2).mean()
    loss, logs = ck.loss(st, f, t, pred, v2_loss=v2)
    assert ck._labels.max_epoch == 4                                                   # epoch 5, lag 1
    assert torch.isfinite(loss) and float(logs["ck2/phase"]) == 2.0 and float(logs["ck2/n_cand"]) == 192.0
    assert float(logs["ck2/G_src_prev"]) == 0.5 and float(logs["ck2/G_src_fallback"]) == 0.5
    assert ck.cum["G_src_prev"] == 1 and ck.cum["G_src_fallback"] == 1
    for k, v in logs.items():
        assert torch.isfinite(v).all(), k
    missing = [k for k in CK2_LOG_KEYS + CK2_LOG_KEYS_T + FPD_LOG_KEYS + [f"ck2/G_src_{n}" for n in D2.SRC_NAMES]
               if k not in logs]
    assert not missing, missing
    S = ck._spy["onpolicy_batch"]
    d, traj96, valid96 = ck._spy["draws"]
    top = student_topk(pred, 16)
    fb = student_topk(pred, 1)["cand"][:, 0]
    # KD set = all 96 current pool columns (parents bitwise at c = 6 k); the teachers score exactly that set
    assert S["cand"].shape == (2, 96, 8, 3) and torch.equal(S["cand"], traj96) and torch.equal(S["cand"][:, ::6],
                                                                                               top["cand"])
    assert torch.equal(S["cand_ok"], valid96) and valid96.all()
    T = tp._spy["run"]
    assert T["kd_ok"].shape == (2, 96) and T["z_det_x"].shape == (2, 96, 5)            # + the label set (online cal.)
    # BCE set: row 3 = generation 4 (16 identities + its 80 draws), row 7 = 16 raw anchors only
    assert S["extra"].shape == (2, 96, 8, 3)
    assert np.array_equal(S["extra"][0, :16].numpy(), gtraj[0, ::6])
    np.testing.assert_array_equal(S["y"][0, :16].numpy(), D2.ck_targets(glab[0, ::6], c.ep_target))
    assert S["ok_lab"][0].all() and (S["vtype_lab"][0, 16:] > 0).all() and int(S["src"][0]) == D2.SRC_PREV
    assert int(S["src"][1]) == D2.SRC_FALLBACK and S["ok_lab"][1, :16].any() and not S["ok_lab"][1, 16:].any()
    assert torch.equal(S["extra"][1, 16:], fb[1][None].expand(80, -1, -1))   # no rule-variant file trajectory
    assert float(logs["ck2/n_var_ok"]) == 80.0
    # the recorded pool = the KD set of the step
    assert ck.recorder.n_tokens == 2
    from navsim.agents.para_ssr.ck import e2e_data as ED
    a = [D2.read_rec2_chunk(Path(ED.rec_rank_dir(c.io_dir, 5, 0)) / n) for n in ck.recorder.chunks]   # write order
    rec = np.concatenate([x["traj"] for x in a])
    assert np.array_equal(rec, traj96.numpy())
    (v2 + loss).backward()
    check_grads(st, pred["bev_embed"])


@need_wu
def test_label_set_masks_only_fallback_rows(tmp_path, runs):
    c, ck, _, _ = _onpolicy(tmp_path, runs)
    ck.epoch = 5
    t = wu_targets()
    rows = t["ck_row"]
    toks = ck.tokens_of(rows)
    G = ck.label_set(rows, toks, t, torch.device("cpu"))
    G0 = O.CKE2E2.label_set(ck, rows, toks, t, torch.device("cpu"))
    assert torch.equal(G["G_ok"][0], G0["G_ok"][0])                                  # generation row unchanged
    assert torch.equal(G["G_ok"][1, :16], G0["G_ok"][1, :16]) and G0["G_ok"][1, 16:].any()
    assert not G["G_ok"][1, 16:].any()                                                # fallback: variants masked
    for k in G:
        if k != "G_ok":
            assert torch.equal(G[k], G0[k]), k


@need_wu
def test_no_teacher_mode(tmp_path, runs):
    nokd = dict(NOKD, fpd={"teacher_det_run": None, "teacher_map_run": None})
    c = fpd_cfg(tmp_path, runs, **nokd)
    c.validate(None)
    ck = FPDE2E(c)
    with pytest.raises(RuntimeError, match="no refiner teachers"):
        ck.teachers("cpu")
    st = student(c)
    spy(ck, "dit_targets")
    t = {k: v for k, v in wu_targets().items() if not k.startswith(("kd_bev_", "kd_ok_"))}
    pred = fake_predictions(2, seed=4)
    loss, logs = ck.loss(st, features(2), t, pred)
    assert torch.isfinite(loss) and ck._teachers is None
    assert float(logs["ck2/kd_score"]) == 0.0 and float(logs["fpd/n_targets"]) == 1.0      # the GT only
    assert float(logs["fpd/bank_safe_frac"]) == 0.0 and not any(k in logs for k in CK2_LOG_KEYS_T)
    tg, tm, _ = ck._spy["dit_targets"]
    assert tg.shape == (2, 1, 8, 3) and tm.all()
    loss.backward()
    check_grads(st, pred["bev_embed"])


@need_wu
def test_nonfinite_loss_dropped(tmp_path, runs):
    c = fpd_cfg(tmp_path, runs)
    ck = FPDE2E(c)
    st = student(c)
    pred = fake_predictions(2, requires_grad=False)
    bev = pred["bev_embed"].clone()
    bev[0, 10, 3] = float("inf")
    pred["bev_embed"] = bev.requires_grad_(True)
    loss, logs = ck.loss(st, features(2), wu_targets(), pred)
    assert float(loss) == 0.0 and float(logs["ck2/loss_nonfinite"]) == 1.0 and ck.cum["nonfinite_loss"] == 1
    loss.backward()
    assert all(p.grad is not None for p in O.ck2_parameters(type("A", (), {"ck_student": st})))


# ----------------------------------------------------------------------------------------------- agent
def _v2_config(**over):
    """the v2 recipe of tests/test_ck_e2e_model_agent on the repo's anchors, on the S grid of this process (CK_BEV64=1:
    the 100 x 100 / 64 m grid of v2_v15_v7)"""
    from test_ck_e2e_model_agent import v2_config
    grid = {}
    if BEV_H == 100:
        pc = [-32.0, 0.0, -1.0, 32.0, 64.0, 3.0]
        grid = dict(bev_h=100, bev_w=100, pc_range=pc, map_pc_range=pc)
    return v2_config(plan_anchor_file=str(ANCHORS), plan_score_file=None, **dict(grid, **over))


def _build(cls=None, seed=0, **over):
    from test_ck_e2e_model_agent import sampling
    if cls is None:
        from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent as cls
    torch.manual_seed(seed)
    return cls(_v2_config(**over), sampling(), lr=1e-4)


def _eq(a, b) -> bool:
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    return a == b


def _groups(agent):
    return [({k: v for k, v in g.items() if k != "params"}, [tuple(p.shape) for p in g["params"]])
            for g in agent.get_optimizers()["optimizer"].param_groups]


@need_wu
def test_agent_dispatch_fpd(tmp_path, runs):
    from navsim.agents.para_ssr.ck.online2 import CKE2E2
    from test_ck_e2e_model_agent import sampling
    from navsim.agents.para_ssr.para_ssr_agent import ParaSSRAgent
    ckd = fpd_dict(tmp_path, runs, score_prior="none")
    off = _build()
    on = _build(ck_e2e2=ckd)
    assert type(on._ck_e2e2) is FPDE2E and isinstance(on._ck_e2e2, CKE2E2)
    assert isinstance(on.ck_student, FN.FPDRefinerNet) and on.ck_student.dit.num_inference_steps == STEPS
    sd_off, sd_on = off.state_dict(), on.state_dict()
    ck_keys = [k for k in sd_on if k.startswith("ck_student.")]
    assert ck_keys == [f"ck_student.{k}" for k in FN.FPDRefinerNet().state_dict()]
    assert [k for k in sd_on if not k.startswith("ck_student.")] == list(sd_off)
    assert all(_eq(sd_on[k], sd_off[k]) for k in sd_off)                  # v2 init unchanged (forked RNG)
    assert not any("teacher" in k for k in sd_on)
    groups = on.get_optimizers()["optimizer"].param_groups
    assert len(groups) == 3 and groups[2]["lr_scale"] == 3.0 and groups[2]["weight_decay"] == 0.01
    assert {id(p) for p in groups[2]["params"]} == {id(p) for p in on.ck_student.parameters()}
    assert type(on.get_target_builders()[-1]).__name__ == "CK2E2ETargetBuilder"
    assert type(on.get_training_callbacks()[-1]).__name__ == "CKE2E2Callback"
    # the agent's helper / student: one training step and the eval outputs
    ck, st = on._ck_e2e2, on.ck_student
    loss, logs = ck.loss(st, features(2), wu_targets(), fake_predictions(2, seed=7))
    loss.backward()
    assert torch.isfinite(loss) and not [n for n, p in st.named_parameters() if p.grad is None]
    st.eval()
    out = ck.infer(st, features(2), fake_predictions(2, seed=8, requires_grad=False))
    assert set(out) == set(INFER_KEYS)
    # strict checkpoint round trip; with a checkpoint the prior is not applied (no file opened)
    with torch.no_grad():
        for p in st.parameters():
            p.add_(0.01)
    ckp = tmp_path / "x.ckpt"
    torch.save({"state_dict": {f"agent.{k}": v for k, v in on.state_dict().items()}}, ckp)
    b = ParaSSRAgent(_v2_config(ck_e2e2=fpd_dict(tmp_path, runs, score_prior="ck2_warmup")), sampling(),
                     checkpoint_path=str(ckp))
    assert b.ck_student.init_report["score_prior"] is None
    b.initialize()
    sa, sb = st.state_dict(), b.ck_student.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    # an arch ck2 checkpoint does not load into an arch fpd agent
    ck2 = _build(ck_e2e2={"enabled": True, "io_dir": str(tmp_path / "io2"), "teacher_det_run": runs["cdet"],
                          "teacher_map_run": runs["cmap"], "score_prior": "none"})
    ck_m = tmp_path / "ck2.ckpt"
    torch.save({"state_dict": {f"agent.{k}": v for k, v in ck2.state_dict().items()}}, ck_m)
    e = ParaSSRAgent(_v2_config(ck_e2e2=ckd), sampling(), checkpoint_path=str(ck_m))
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        e.initialize()


@need_wu
def test_agent_ck2_and_off_unchanged_vs_v2_final(tmp_path, runs):
    """the arch 'ck2' (and refiner-off) agent of this tree is built exactly like the v2_final para_ssr_agent.py (git
    BASE_COMMIT): same helper / student classes, state_dict keys and values, optimiser groups, builders, callbacks."""
    from navsim.agents.para_ssr.ck.model import CKNet
    from navsim.agents.para_ssr.ck.online2 import CKE2E2
    src = _git_show("navsim/agents/para_ssr/para_ssr_agent.py")
    if src is None:
        pytest.skip(f"git show {BASE_COMMIT} unavailable")
    p = tmp_path / "para_ssr_agent_base.py"
    p.write_text(src)
    Base = _load_module("navsim.agents.para_ssr._para_ssr_agent_v2_final_test", p).ParaSSRAgent
    ck2 = {"enabled": True, "io_dir": str(tmp_path / "io"), "teacher_det_run": runs["cdet"],
           "teacher_map_run": runs["cmap"], "score_prior": "none"}
    cases = [{}, {"ck_e2e2": {"enabled": False}}, {"ck_e2e2": ck2}]
    if HAVE_REAL_CK2:                                   # the BEV-KD arm reads the CK2 runs' z-score files
        cases.append({"ck_e2e2": dict(ck2, teacher_det_run=str(REAL_CK2["det"]), teacher_map_run=str(REAL_CK2["map"]),
                                      bev_kd={"enabled": True, "teachers": ["det", "map"]})})
    for over in cases:
        a, b = _build(Base, **over), _build(**over)
        if over.get("ck_e2e2", {}).get("enabled"):
            assert type(b._ck_e2e2) is CKE2E2 and type(b.ck_student) is CKNet
            assert type(a._ck_e2e2) is type(b._ck_e2e2) and a._ck_e2e2.cfg == b._ck_e2e2.cfg
        else:
            assert a._ck_e2e2 is None and b._ck_e2e2 is None and not hasattr(b, "ck_student")
        sa, sb = a.state_dict(), b.state_dict()
        assert list(sa) == list(sb) and all(_eq(sa[k], sb[k]) for k in sa), over
        assert _groups(a) == _groups(b), over
        assert [type(x).__name__ for x in a.get_target_builders()] == [type(x).__name__ for x in
                                                                       b.get_target_builders()]
        assert [type(x).__name__ for x in a.get_training_callbacks()] == [type(x).__name__ for x in
                                                                          b.get_training_callbacks()]
        assert torch.equal(torch.get_rng_state(), _rng_after(Base, over)), over


def _rng_after(cls, over):
    """CPU RNG state right after building an agent of cls (the same for both files: construction draws alike)."""
    _build(cls, **over)
    return torch.get_rng_state()


# ----------------------------------------------------------------------------------------------- offline bank
def _write_bank(run: str, split: str, tokens, seed: int) -> np.ndarray:
    """a labelled offline bank (tools/ck/fpd/bank.py layout) for `tokens`: 32 trajectories per token, column 0 / 1 /
    2 safe (NC = DAC = 1) and far apart, the rest unsafe (DAC 0)."""
    import pandas as pd
    bd = Path(run) / "bank" / split
    bd.mkdir(parents=True, exist_ok=True)
    g = np.random.default_rng(seed)
    N = len(tokens)
    traj = g.normal(size=(N, 32, 8, 3)).astype(np.float32)
    traj[:, :3, :, 0] += np.array([0.0, 5.0, 10.0], np.float32)[None, :, None]          # 3 safe, >= 1 m apart
    lab = np.zeros((N, 32, 9), np.float32)
    lab[..., 0], lab[..., 2], lab[..., 3], lab[..., 4] = 1.0, 0.8, 1.0, 1.0
    lab[:, :3, 1] = 1.0                                                                 # DAC: only columns 0..2 pass
    pd.DataFrame({"token": list(tokens), "log": ["l"] * N}).to_parquet(bd / "tokens.parquet")
    np.save(bd / "traj.npy", traj)
    np.save(bd / "labels.npy", lab)
    np.save(bd / "ok.npy", np.ones((N, 32), bool))
    np.save(bd / "label_state.npy", np.ones(N, np.uint8))
    (bd / "meta.json").write_text("{}")
    return traj


@need_wu
def test_dit_targets_from_the_offline_bank(tmp_path, runs):
    """fpd.mt_source 'bank': the teacher targets are the SAFE bank trajectories (official labels NC = DAC = 1), at
    most mt_k - 1 of them besides the GT, read by token from both teachers' <run>/bank/<split>."""
    c = fpd_cfg(tmp_path, runs, fpd={"mt_source": "bank"})
    ck = FPDE2E(c)
    toks = ck.tokens_of(torch.as_tensor(ROWS))
    split = Path(str(c.warmup["packed"])).name
    tr = {t: _write_bank(c.fpd[f"teacher_{t}_run"], split, toks, i) for i, t in enumerate(("det", "map"))}
    try:
        bank, y, ok = ck.bank_store().lookup(list(toks), torch.device("cpu"))
        assert bank.shape == (len(ROWS), 64, 8, 3) and y.shape == (len(ROWS), 64, 5) and bool(ok.all())
        human = torch.zeros(len(ROWS), 8, 3)
        tgt, tmask, st = ck.dit_targets(human, (bank, y, ok))
        assert tgt.shape[1] == c.fpd["mt_k"] and bool(tmask[:, 0].all()) and st["n_teacher"] == 3 * len(ROWS)
        safe = np.concatenate([tr["det"][:, :3], tr["map"][:, :3]], 1)
        for b in range(len(ROWS)):
            for j in range(1, c.fpd["mt_k"]):
                assert tmask[b, j] and any(np.allclose(tgt[b, j].numpy(), s) for s in safe[b])
    finally:
        import shutil
        for t in ("det", "map"):
            shutil.rmtree(Path(c.fpd[f"teacher_{t}_run"]) / "bank", ignore_errors=True)


def test_offline_bank_missing_is_a_clear_error(tmp_path, runs):
    c = fpd_cfg(tmp_path, runs, fpd={"mt_source": "bank"})
    with pytest.raises(RuntimeError, match="needs the finished teacher pipeline"):
        FPDE2E(c).bank_store().lookup(["x"], torch.device("cpu"))
