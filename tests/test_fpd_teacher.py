"""FPD teacher pipeline tests (CPU): tools/ck/fpd/{train_fpd_teacher, label_fpd_bank, bank}.py.

Synthetic (always run; single process, no torchrun): parser (torchrun pass-through, defaults, score keys, run dir),
near + mid parent selection and the canonical (GT-rank) parent order, the three stages end to end (dit with a
--max-steps stop + resume, bank sampling = the recipe's draws, a per-row resume that re-samples only the missing rows,
the stale-plan guard, the CPU labeler with the fake scorer, scorer with the label prior and the frozen heads) -> the run dir loads with
fpd_net.load_fpd_teacher (strict) and has the contract config keys; BankDataset row filtering / alignment; labeler
errors / retry / stale-plan refusal; BEV-unreadable bank rows retried and bounded (> 1 % = no sample_done.json); stage
scorer refuses a bank of other dataset rows; the teacher files of a finished stage only with its own options; 2-rank
gloo DDP == 1 process for the dit, bank and scorer stages.
Real data (skipped when absent): canonical parent order == AnchorSampler epoch-0 near + mid (shuffle False) == the
variant files' anchor_idx; the official bank labels of 2 navtrain_val tokens (their 32 epoch-0 anchors as the bank) ==
raw256 bit for bit; a CK_BEV64=1 subprocess smoke of the CLI (dit -> bank -> official labels -> scorer) per arm.

  cd $REPO && set -a && . tools/release/paths.env && set +a && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= \
    /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_fpd_teacher.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])   # repo root of this worktree
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck import anchor_sampler as AS  # noqa: E402
from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from navsim.agents.para_ssr.ck import fpd_net as FN  # noqa: E402
from tools.ck.fpd import bank as BK  # noqa: E402
from tools.ck.fpd import label_fpd_bank as FL  # noqa: E402
from tools.ck.fpd import train_fpd_teacher as FT  # noqa: E402
from tools.ck.e2e import labeler as L1  # noqa: E402

REAL = Path(Cn.CK_DATA)
VAL = "navtrain_val"
CKI = list(Cn.CK_LABEL_IDX)
DDIM = 2                        # synthetic tests: 2 DDIM steps (8 in the recipe) keep the CPU sampling short


@pytest.fixture(autouse=True)
def _threads(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("CK_LABELER_FAKE_FAIL", raising=False)
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------------- synthetic data
def _paths(n, v0, rng):
    t = np.arange(1, 9) * 0.5
    out = np.zeros((n, 8, 3), np.float32)
    for j in range(n):
        v = v0 * rng.uniform(0.4, 1.4)
        k = rng.normal(scale=0.03)
        x = v * t
        out[j, :, 0] = x
        out[j, :, 1] = 0.5 * k * x ** 2 + rng.normal(scale=0.3)
        out[j, :, 2] = np.arctan(k * x)
    return out


def _labels(rng, n):
    y = np.ones((n, 5), np.float32)
    f = rng.random(n) < 0.35
    key = rng.choice([0, 1, 3, 4], size=n)
    y[np.flatnonzero(f), key[f]] = 0.0
    y[:, 2] = rng.random(n).astype(np.float32)
    pd_ = y[:, 0] * y[:, 1] * (5 * y[:, 3] + 2 * y[:, 4] + 5 * y[:, 2]) / 12.0
    return y, pd_.astype(np.float32)


class FakeBase(torch.utils.data.Dataset):
    """CK2Dataset-like (n_var 0): n packed rows, dataset rows `rows`; per row a pool of 64 synthetic anchors around the
    GT; per (row, epoch) a draw of 8 near + 8 mid + 16 far anchors in a random slot order (anc_group 0 / 1 / 2);
    a small random BEV (SceneTokens pools any grid)."""

    def __init__(self, n=6, rows=None, seed=0, name="tr", hw=(8, 8)):
        self.n, self.seed, self.name = int(n), int(seed), name
        self.rows = np.arange(n, dtype=np.int64) if rows is None else np.asarray(rows, np.int64)
        self.tokens = np.array([f"{name}{i:05d}" for i in range(n)])
        self.logs = np.array(["__fake__"] * n)
        rng = np.random.default_rng(seed)
        self.bev = (rng.standard_normal((n, 256, *hw)) * 0.5).astype(np.float16)
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = int(e)

    def describe(self):
        return {"fake": self.name, "n_rows": int(len(self.rows))}

    def __len__(self):
        return len(self.rows)

    def labels_item(self, i, epoch=None):
        r = int(self.rows[i])
        e = self.epoch if epoch is None else int(epoch)
        rng = np.random.default_rng([self.seed, r, 7])
        v0 = 3.0 + r % 4
        gt = _paths(1, v0, rng)[0]
        pool = _paths(64, v0, rng)
        order = np.argsort(AS.gt_distance(pool, gt), kind="stable")
        er = np.random.default_rng([self.seed, r, e, 11])
        mid = order[8:32][np.sort(er.permutation(24)[:8])]
        far = order[32:][er.permutation(32)[:16]]
        idx = np.concatenate([order[:8], mid, far])
        group = np.array([0] * 8 + [1] * 8 + [2] * 16, np.int8)
        p = er.permutation(32)
        idx, group = idx[p], group[p]
        y, pdm = _labels(er, 32)
        st = np.zeros(8, np.float32)
        st[r % 4], st[4] = 1.0, v0
        return dict(token=str(self.tokens[r]), row=r, cand=np.ascontiguousarray(pool[idx]), status=st, gt_traj=gt,
                    y=y, y_pdms=pdm, y_ok=np.ones(32, bool), anc_idx=(1000 * r + idx).astype(np.int64),
                    anc_group=group, anc_pi=np.ones(32), anc_w=np.ones(32, np.float32))

    def __getitem__(self, i):
        it = self.labels_item(i)
        it["bev"], it["bev_ok"] = self.bev[it["row"]], True
        return it


def _ds(n_train=6, n_val=4, val_rows=(0, 2, 3)):
    return {"fk_train": FakeBase(n_train, seed=0, name="tr"), "fk_val": FakeBase(n_val, rows=val_rows, seed=1, name="va")}


def _args(rd, stage, extra=(), arm="T"):
    a = FT.get_parser().parse_args(
        ["--stage", stage, "--arm", arm, "--run-dir", str(rd), "--device", "cpu", "--workers", "0",
         "--tokens-per-batch", "2", "--epochs", "2", "--log-every", "1", "--ckpt-every", "0", "--eval-every", "1",
         "--eval-batch", "2", "--sample-batch", "2", "--warmup-steps", "1", "--split", "fk_train", "--val-split",
         "fk_val", "--prior-rows", "50", "--steps", str(DDIM), *extra])
    return FT.resolve_args(a)


def _jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def _logit(p):
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


# ----------------------------------------------------------------------------------------------- parser / helpers
def test_torchrun_passes_every_option():
    """torchrun's argparse scans the script's arguments too and aborts on an ambiguous abbreviation of its own
    options (train_ck2's '--run'); every option of the FPD trainer passes through untouched."""
    from torch.distributed.run import get_args_parser
    tp = get_args_parser()
    head = ["--standalone", "--nproc_per_node=2", "tools/ck/fpd/train_fpd_teacher.py"]
    for act in FT.get_parser()._actions:
        for o in act.option_strings:
            if o in ("-h", "--help"):
                continue
            args = ["--stage", "dit", "--arm", "T", o, "1"]
            assert tp.parse_args(head + args).training_script_args == args, o


def test_resolve_args_defaults(monkeypatch):
    from navsim.common import ssr_paths as SP
    a = FT.resolve_args(FT.get_parser().parse_args(["--stage", "dit", "--arm", "T"]))
    assert a.run_dir == str(Path(SP.SSR_TEACHER_CACHE) / "refiner_fpd" / "det")
    assert a.score_keys == ["nc", "ep", "ttc", "comfort"] and a.ep_target == "official"
    assert a.bank_splits == ["navtrain_train", "navtrain_val"] and a.val_split == "navtrain_val"
    assert (a.lr, a.wd, a.clip, a.epochs, a.tokens_per_batch, a.amp, a.steps, a.grid) == (2e-4, 0.01, 1.0, 10, 16, 0, 8, 8)
    m = FT.resolve_args(FT.get_parser().parse_args(["--stage", "scorer", "--arm", "M", "--val-split", "none"]))
    assert m.run_dir.endswith("refiner_fpd/map") and m.score_keys == ["dac", "ep"] and m.bank_splits == ["navtrain_train"]
    x = FT.resolve_args(FT.get_parser().parse_args(["--stage", "bank", "--arm", "M", "--score-keys", "ep,dac,nc"]))
    assert x.score_keys == ["nc", "dac", "ep"]                       # canonical CK_KEYS order
    for bad in (["--score-keys", "nc,xx"], ["--score-keys", "nc,nc"], ["--run-dir", "/tmp/a=b"], ["--workers", "17"]):
        with pytest.raises(SystemExit):
            FT.resolve_args(FT.get_parser().parse_args(["--stage", "dit", "--arm", "T", *bad]))
    assert FT.TEACHER_OF_ARM == {"T": "det", "M": "map"} and FN.FPD_TEACHER_ARMS == {"det": "T", "map": "M"}
    assert BK.stage_rows_limit("tr", "tr", "va", 4, 2) == 4 and BK.stage_rows_limit("va", "tr", "va", 4, 2) == 2
    assert BK.stage_rows_limit("other", "tr", "va", 4, 2) == 0


def test_near_mid_and_canonical_order():
    ds = FakeBase(3, seed=4)
    items = [ds[i] for i in range(3)]
    cand = torch.as_tensor(np.stack([it["cand"] for it in items]))
    grp = torch.as_tensor(np.stack([it["anc_group"] for it in items]))
    par = BK.near_mid(cand, grp)
    assert par.shape == (3, 16, 8, 3)
    for b, it in enumerate(items):
        m = np.flatnonzero(it["anc_group"] <= 1)
        assert np.array_equal(par[b].numpy(), it["cand"][m])           # slot order kept
        o = BK.canonical_near_mid(it)
        assert sorted(o.tolist()) == m.tolist()
        d = AS.gt_distance(it["cand"][o], it["gt_traj"])
        assert (np.diff(d) >= 0).all()                                 # GT-rank order
        assert set(np.flatnonzero(it["anc_group"] == 0).tolist()) == set(o[:8].tolist())   # near first
        p = BK.ParentItems(ds)[b]
        assert np.array_equal(p["parents"], it["cand"][o]) and np.array_equal(p["parent_idx"], it["anc_idx"][o])
        assert set(p) == {"token", "row", "parents", "parent_idx", "bev", "bev_ok", "status", "gt_traj"}
    with pytest.raises(ValueError):
        BK.near_mid(cand, torch.zeros_like(grp))
    bad = dict(items[0], anc_group=np.zeros(32, np.int8))
    with pytest.raises(ValueError):
        BK.canonical_near_mid(bad)


# ----------------------------------------------------------------------------------------------- bank files by hand
def _fake_bank(bd: Path, base, rows_sampled, traj_fn, split="fk"):
    """sampling outputs of stage bank, written directly (layout of bank.py)."""
    bd.mkdir(parents=True, exist_ok=True)
    N = base.n
    traj = np.full((N, BK.BANK_K, 8, 3), np.nan, np.float32)
    samp = np.zeros(N, np.uint8)
    for r in rows_sampled:
        traj[r] = traj_fn(r)
        samp[r] = 1
    np.save(bd / BK.F_TRAJ, traj)
    np.save(bd / BK.F_SAMPLED, samp)
    np.save(bd / BK.F_PARENT, np.full((N, BK.N_PARENTS), -1, np.int64))
    BK.token_table(base).to_parquet(bd / BK.F_TOKENS, index=False)
    meta = dict(split=split, n_packed=N, n_rows=len(base.rows), dit_ckpt_sha16="d" * 16)
    (bd / BK.F_SAMPLE_META).write_text(json.dumps(meta))
    (bd / BK.F_SAMPLE_DONE).write_text(json.dumps(dict(meta, traj_sha16=BK.sha16_file(bd / BK.F_TRAJ),
                                                       n_sampled=len(rows_sampled))))
    return traj


def test_labeler_fake_errors_retry_resume_and_stale_plan(tmp_path, monkeypatch):
    base = FakeBase(5, rows=[0, 1, 2, 4], seed=2, name="lb")
    bd = tmp_path / "bank" / "fk"
    rng = np.random.default_rng(0)
    traj = _fake_bank(bd, base, [0, 1, 2, 4], lambda r: _paths(32, 4.0, rng))
    with pytest.raises(SystemExit, match="labels not finished"):
        BK.check_bank(bd)
    fail = tmp_path / "fail.json"
    fail.write_text(json.dumps([str(base.tokens[2])]))
    monkeypatch.setenv("CK_LABELER_FAKE_FAIL", str(fail))               # labeler.score_one fake failure hook
    meta = FL.label_split(bd, workers=2, scorer="fake", log=lambda s: None)
    assert meta["n_sampled"] == 4 and meta["n_labelled"] == 3 and meta["n_err"] == 1 and meta["err_rows_head"] == [2]
    st = np.load(bd / BK.F_LSTATE)
    assert st.tolist() == [1, 1, 2, 0, 1]
    errs = _jsonl(bd / BK.F_ERRORS)
    assert len(errs) == 1 and errs[0]["row"] == 2 and "fake failure" in errs[0]["error"]
    with pytest.raises(SystemExit, match="failed to label"):          # 1 of 4 > 1 %
        BK.check_bank(bd)
    monkeypatch.delenv("CK_LABELER_FAKE_FAIL")
    assert FL.label_split(bd, workers=2, scorer="fake", log=lambda s: None)["n_err"] == 1   # no retry by default
    meta = FL.label_split(bd, workers=2, scorer="fake", retry_errors=True, log=lambda s: None)
    assert meta["n_err"] == 0 and meta["n_labelled"] == 4 and meta["n_ok_cands"] == 4 * BK.BANK_K
    lab, ok = np.load(bd / BK.F_LABELS), np.load(bd / BK.F_OK)
    for r in (0, 1, 2, 4):
        assert np.array_equal(lab[r], L1.fake_scores(traj[r])) and ok[r].all()
    assert np.isnan(lab[3]).all() and not ok[3].any()
    summ = BK.check_bank(bd, dit_sha16="d" * 16)
    assert summ["n_sampled"] == 4 and summ["n_err"] == 0
    with pytest.raises(SystemExit, match="DiT checkpoint"):
        BK.check_bank(bd, dit_sha16="e" * 16)
    # a non-finite trajectory is left unscored (ok False), the others of the token are scored
    # (labeler.score_one rule); a re-sampled bank (other traj sha16) is refused unless fresh
    t2 = np.load(bd / BK.F_TRAJ)
    t2[1, 5] = np.nan
    np.save(bd / BK.F_TRAJ, t2)
    d = json.loads((bd / BK.F_SAMPLE_DONE).read_text())
    d["traj_sha16"] = BK.sha16_file(bd / BK.F_TRAJ)
    (bd / BK.F_SAMPLE_DONE).write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="re-sampled"):
        BK.check_bank(bd)
    with pytest.raises(SystemExit, match="--fresh"):
        FL.label_split(bd, workers=1, scorer="fake", log=lambda s: None)
    with pytest.raises(SystemExit, match="--fresh"):
        FL.label_split(bd, workers=1, scorer="official", log=lambda s: None)
    meta = FL.label_split(bd, workers=2, scorer="fake", fresh=True, limit=2, log=lambda s: None)
    assert meta is None                                                # --limit: rows left -> no meta.json yet
    assert not (bd / BK.F_LMETA).is_file()
    meta = FL.label_split(bd, workers=2, scorer="fake", log=lambda s: None)
    assert meta["traj_sha16"] == d["traj_sha16"] and meta["n_ok_cands"] == 4 * BK.BANK_K - 1
    lab, ok = np.load(bd / BK.F_LABELS), np.load(bd / BK.F_OK)
    assert np.isnan(lab[1, 5]).all() and not ok[1, 5] and ok[1, np.arange(32) != 5].all()
    assert np.array_equal(lab[1, :5], L1.fake_scores(t2[1, :5]))
    assert BK.check_bank(bd)["traj_sha16"] == d["traj_sha16"]
    assert FL.main(["--run-dir", str(tmp_path), "--workers", "1", "--scorer", "fake"]) == 0   # default splits
    assert FL.default_splits(tmp_path) == ["fk"]
    with pytest.raises(SystemExit):
        FL.label_split(tmp_path / "bank" / "absent", workers=1, scorer="fake")


def test_bank_dataset_filters_rows_and_aligns(tmp_path):
    base = FakeBase(6, rows=[0, 1, 2, 3, 5], seed=3, name="bd")
    bd = tmp_path / "bank"
    rng = np.random.default_rng(1)
    traj = _fake_bank(bd, base, [0, 1, 3, 5], lambda r: _paths(32, 5.0, rng))
    FL.label_split(bd, workers=1, scorer="fake", log=lambda s: None)
    st = np.load(bd / BK.F_LSTATE, mmap_mode="r+")
    st[5] = 2                                                          # an error row
    st.flush()
    del st
    ds = BK.BankDataset(base, bd)
    assert ds.rows.tolist() == [0, 1, 3] and ds.n_dropped == 2 and len(ds) == 3   # row 2 not sampled, 5 error
    lab = np.load(bd / BK.F_LABELS)
    for j, r in enumerate(ds.rows):
        it = ds[j]
        assert it["row"] == r and np.array_equal(it["bank_traj"], traj[r])
        assert np.array_equal(it["bank_y"], lab[r][:, CKI]) and it["bank_ok"].all()
        assert np.array_equal(it["bank_pdms"], lab[r][:, Cn.LABEL_COLS.index("pdms")])
        assert np.array_equal(it["cand"], base[int(np.flatnonzero(base.rows == r)[0])]["cand"])
    assert "parents" not in ds[0] and BK.BankDataset(base, bd, parents=True)[0]["parents"].shape == (16, 8, 3)
    import pickle
    ds2 = pickle.loads(pickle.dumps(ds))
    assert np.array_equal(ds2[1]["bank_traj"], ds[1]["bank_traj"])
    pr = BK.bank_label_prior(ds, 10)
    ys = []
    for j in range(len(ds)):
        it = ds.labels_item(j, epoch=0)
        ys += [it["y"], it["bank_y"]]
    want = np.clip(np.concatenate(ys).astype(np.float64).mean(0), 1e-3, 1 - 1e-3)
    assert pr.shape == (5,) and np.allclose(pr, want)
    np.save(bd / BK.F_LSTATE, np.zeros(5, np.uint8))                   # wrong N
    with pytest.raises(ValueError):
        BK.BankDataset(base, bd)


# ----------------------------------------------------------------------------------------------- end to end (CPU)
def test_three_stages_end_to_end_and_teacher_load(tmp_path):
    rd = tmp_path / "det"
    with pytest.raises(SystemExit, match="stage dit not finished"):
        FT.sample_bank(_args(rd, "bank"), datasets=_ds())
    # ---- stage dit: a --max-steps stop, then resume to the end of 2 epochs x 3 steps
    FT.train_stage(_args(rd, "dit", ["--max-steps", "2"]), "dit", datasets=_ds())
    sd = rd / "dit"
    assert json.loads((sd / "done.json").read_text())["steps"] == 2
    (sd / "done.json").unlink()
    with pytest.raises(SystemExit, match="--resume"):                  # ckpt_last exists, no --resume
        FT.train_stage(_args(rd, "dit"), "dit", datasets=_ds())
    FT.train_stage(_args(rd, "dit", ["--resume"]), "dit", datasets=_ds())
    done = json.loads((sd / "done.json").read_text())
    assert done["steps"] == 6 and done["epochs_done"] == 2 and done["n_skip"] == {"loss": 0, "grad": 0}
    steps = [r for r in _jsonl(sd / "train_log.jsonl") if r["kind"] == "step"]
    assert [r["step"] for r in steps] == list(range(1, 7))
    assert all(np.isfinite(r["loss"]) and r["n_pairs"] == 2.0 and "score_bce" not in r for r in steps)
    val = _jsonl(sd / "val_metrics.jsonl")
    assert [(r["epoch"], r["step"]) for r in val] == [(0, 2), (0, 3), (1, 6)] and all(r["kind"] == "dit" for r in val)
    for k in ("l_diff", "l_traj", "parent_min_ade", "bank_min_ade", "match_ade", "match_draw_min_ade",
              "frac_match_improved", "draw_shift"):
        assert np.isfinite(val[-1][k]), k
    assert val[-1]["n_tokens"] == 3 and val[-1]["n_pairs"] == 3
    assert (sd / "ckpt_ep0.pt").is_file() and (sd / "ckpt_ep1.pt").is_file()
    cfg = json.loads((sd / "config.json").read_text())
    assert cfg["trainer"] == "train_fpd_teacher" and cfg["stage"] == "dit" and cfg["frozen"]
    assert all(n.startswith("scorer.") for n in cfg["frozen"]) and cfg["bev_shape"] == [256, 8, 8]
    ck = torch.load(sd / "ckpt_last.pt", map_location="cpu", weights_only=False)
    ref = FN.FPDRefinerNet(seed=0, steps=DDIM)
    for k, v in ref.scorer.state_dict().items():                       # the scorer is untouched in stage dit
        assert torch.equal(ck["model"]["scorer." + k], v), k
    assert not torch.equal(ck["model"]["dit.velocity_head.weight"], ref.dit.velocity_head.weight)
    (sd / "done.json").rename(sd / "done.bak")
    with pytest.raises(SystemExit, match="resume with seed"):
        FT.train_stage(_args(rd, "dit", ["--resume", "--seed", "1"]), "dit", datasets=_ds())
    (sd / "done.bak").rename(sd / "done.json")
    with pytest.raises(SystemExit, match="sampling not finished"):     # stage scorer before the bank
        FT.train_stage(_args(rd, "scorer"), "scorer", datasets=_ds())

    # ---- stage bank
    FT.sample_bank(_args(rd, "bank"), datasets=_ds())
    net = FT.build_net(_args(rd, "bank"))
    net.load_state_dict(ck["model"])
    net.eval()
    for split, base in _ds().items():
        bd = rd / "bank" / split
        sdn = json.loads((bd / BK.F_SAMPLE_DONE).read_text())
        assert sdn["n_sampled"] == len(base.rows) and sdn["n_bev_bad"] == 0 and sdn["steps"] == DDIM
        assert sdn["dit_ckpt_sha16"] == done["ckpt_sha16"] and sdn["traj_sha16"] == BK.sha16_file(bd / BK.F_TRAJ)
        traj, pidx = np.load(bd / BK.F_TRAJ), np.load(bd / BK.F_PARENT)
        st = np.load(bd / BK.F_SAMPLED)
        assert sorted(np.flatnonzero(st == 1).tolist()) == sorted(base.rows.tolist())
        for r in range(base.n):
            assert np.isfinite(traj[r]).all() == (r in base.rows.tolist())
        tt = __import__("pandas").read_parquet(bd / BK.F_TOKENS)
        assert tt.token.tolist() == base.tokens.tolist() and set(tt.log) == {"__fake__"}
        i = len(base.rows) - 1
        it = BK.ParentItems(base)[i]
        r = it["row"]
        assert np.array_equal(pidx[r], it["parent_idx"])
        with torch.no_grad():                    # the stored draws = the recipe (float rounding: batch composition)
            noise = FN.token_noise([it["token"]], (16, 2, 8, 3), 0, -1, BK.BANK_STREAM)
            ctx = net.scene(torch.as_tensor(it["bev"])[None])
            dr = net.sample_draws(torch.as_tensor(it["parents"])[None], ctx, torch.as_tensor(it["status"])[None], noise,
                                  DDIM)
        assert np.allclose(traj[r], dr.reshape(32, 8, 3).numpy(), rtol=1e-5, atol=1e-5)
    # a per-row resume re-samples only the missing rows (same draws up to float rounding); another plan is refused
    bd = rd / "bank" / "fk_train"
    keep = np.load(bd / BK.F_TRAJ).copy()
    t = np.load(bd / BK.F_TRAJ, mmap_mode="r+")
    s = np.load(bd / BK.F_SAMPLED, mmap_mode="r+")
    t[[1, 4]] = np.nan
    s[[1, 4]] = 0
    t.flush()
    s.flush()
    del t, s
    (bd / BK.F_SAMPLE_DONE).unlink()
    FT.sample_bank(_args(rd, "bank"), datasets=_ds())
    again = np.load(bd / BK.F_TRAJ)
    assert np.array_equal(np.delete(again, [1, 4], 0), np.delete(keep, [1, 4], 0))
    assert np.allclose(again[[1, 4]], keep[[1, 4]], rtol=1e-5, atol=1e-5)
    assert json.loads((bd / BK.F_SAMPLE_DONE).read_text())["traj_sha16"] == BK.sha16_file(bd / BK.F_TRAJ)
    meta = json.loads((bd / BK.F_SAMPLE_META).read_text())
    (bd / BK.F_SAMPLE_META).write_text(json.dumps(dict(meta, dit_ckpt_sha16="0" * 16)))
    with pytest.raises(SystemExit, match="dit_ckpt_sha16"):
        FT.sample_bank(_args(rd, "bank"), datasets=_ds())
    (bd / BK.F_SAMPLE_META).write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="labels not finished"):
        FT.train_stage(_args(rd, "scorer"), "scorer", datasets=_ds())

    # ---- CPU labels (fake scorer)
    assert FL.main(["--run-dir", str(rd), "--workers", "2", "--scorer", "fake"]) == 0
    for split, base in _ds().items():
        bd = rd / "bank" / split
        m = json.loads((bd / BK.F_LMETA).read_text())
        assert m["n_sampled"] == len(base.rows) and m["n_err"] == 0 and m["n_ok_cands"] == len(base.rows) * 32
        lab, traj = np.load(bd / BK.F_LABELS), np.load(bd / BK.F_TRAJ)
        for r in base.rows:
            assert np.array_equal(lab[r], L1.fake_scores(traj[r]))

    # ---- stage scorer (refused on a bank sampled for other dataset rows: it would train on that subset only)
    other = dict(_ds(), fk_train=FakeBase(6, rows=[0, 1, 2, 3, 4], seed=0, name="tr"))
    with pytest.raises(SystemExit, match="sampled for dataset rows"):
        FT.train_stage(_args(rd, "scorer"), "scorer", datasets=other)
    FT.train_stage(_args(rd, "scorer"), "scorer", datasets=_ds())
    ss = rd / "scorer"
    scfg = json.loads((ss / "config.json").read_text())
    assert scfg["frozen"] == [f"scorer.heads.1.{n}" for n, _ in ref.scorer.heads[1].named_parameters()]   # dac head
    prior = scfg["init_report"]["score_prior"]
    assert len(prior) == 5 and all(0 < p < 1 for p in prior)
    assert scfg["dit_ckpt_sha16"] == done["ckpt_sha16"] and scfg["bank_traj_sha16"]
    steps = [r for r in _jsonl(ss / "train_log.jsonl") if r["kind"] == "step"]
    assert [r["step"] for r in steps] == list(range(1, 7))
    assert all(np.isfinite(r["score_bce"]) and r["n_anc_ok"] == 64 and r["n_bank_ok"] == 64 for r in steps)
    assert all(np.isfinite(r["l_diff"]) and r["n_pairs"] == 2.0 for r in steps)
    val = _jsonl(ss / "val_metrics.jsonl")
    assert [r["kind"] for r in val] == ["scorer", "scorer"]
    for k in ("bce_nc", "auc_fail_nc", "anc_bce_ep", "bank_bce_ep", "ep_mae", "auc_keys_mean", "l_diff", "bank_min_ade"):
        assert k in val[-1], k
    assert val[-1]["n_cand"] == 3 * 64
    # ---- the teacher run dir (contract): strict load, config keys, weights = the stage's last checkpoint
    tnet, tcfg = FN.load_fpd_teacher(rd, device="cpu")
    for k in FT.FINAL_CONFIG_KEYS:
        assert k in tcfg, k
    assert (tcfg["trainer"], tcfg["arm"], tcfg["ep_target"], tcfg["stage"], tcfg["seed"], tcfg["steps"], tcfg["grid"]) \
        == ("train_fpd_teacher", "T", "official", "scorer", 0, DDIM, 8)
    assert tcfg["score_keys"] == ["nc", "ep", "ttc", "comfort"] and tcfg["teacher"] == "det"
    assert tcfg["split"] == "fk_train" and tcfg["init_from"] == "none" and tcfg["bev_shape"] == [256, 8, 8]
    assert tcfg["frozen_heads"] == ["dac"] and tcfg["init_report"]["score_prior"] == prior
    assert tcfg["stages"]["dit"]["ckpt_sha16"] == done["ckpt_sha16"]
    assert tcfg["recipe"]["dit"]["steps_total"] == 6 and tcfg["recipe"]["dit"]["max_steps"] == 0   # last dit session
    assert tcfg["recipe"]["scorer"]["noise_epoch_offset"] == 1000 and tcfg["recipe"]["scorer"]["lr"] == 2e-4
    assert set(tcfg["stages"]["bank"]) == {"fk_train", "fk_val"}
    sck = torch.load(ss / "ckpt_last.pt", map_location="cpu", weights_only=False)
    for k, v in tnet.state_dict().items():
        assert torch.equal(v, sck["model"][k]), k
    raw = torch.load(rd / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert set(raw) == {"state_dict", "config"} and raw["config"]["arm"] == "T"
    tdone = json.loads((rd / "done.json").read_text())
    assert tdone["ckpt_sha16"] == tnet.init_report["ckpt_sha16"] and tdone["arm"] == "T"
    assert not any(p.requires_grad for p in tnet.parameters()) and not tnet.training
    # frozen head (dac): construction weights, bias = logit(prior); trained heads moved
    h, h0 = tnet.scorer.heads[1], ref.scorer.heads[1]
    assert torch.equal(h[0].weight, h0[0].weight) and torch.equal(h[2].weight, h0[2].weight)
    assert float(h[2].bias) == pytest.approx(_logit(prior[1]), abs=1e-5)
    assert not torch.equal(tnet.scorer.heads[0][2].weight, ref.scorer.heads[0][2].weight)
    # the teacher works on a 96-candidate set (the e2e pool) and samples
    it = _ds()["fk_val"][0]
    with torch.no_grad():
        ctx = tnet.scene(torch.as_tensor(it["bev"])[None])
        lg = tnet.score(torch.as_tensor(np.concatenate([it["cand"]] * 3))[None], ctx, torch.as_tensor(it["status"])[None])
        par = torch.as_tensor(it["cand"][:16])[None]
        dr = tnet.sample_draws(par, ctx, torch.as_tensor(it["status"])[None], torch.randn(1, 16, 5, 8, 3))
    assert lg.shape == (1, 96, 5) and torch.isfinite(lg).all() and dr.shape == (1, 16, 5, 8, 3)
    # a finished teacher: every stage returns at once
    mt = (rd / "ckpt_last.pt").stat().st_mtime_ns
    for st_ in ("dit", "scorer"):
        assert FT.train_stage(_args(rd, st_, ["--resume"]), st_, datasets=_ds()) == rd
    assert FT.sample_bank(_args(rd, "bank"), datasets=_ds()) == rd
    assert (rd / "ckpt_last.pt").stat().st_mtime_ns == mt
    # interrupted between the stage's done.json and the teacher files: a rerun writes them, with the stage's own
    # options only (the teacher config takes arm / score keys / steps / ... from them)
    for f in ("done.json", "config.json", "ckpt_last.pt"):
        (rd / f).unlink()
    for bad in (["--score-keys", "nc,dac"], ["--steps", "3"]):
        with pytest.raises(SystemExit, match="finished stage ran with"):
            FT.train_stage(_args(rd, "scorer", ["--resume", *bad]), "scorer", datasets=_ds())
        assert not (rd / "done.json").exists() and not (rd / "config.json").exists()
    FT.train_stage(_args(rd, "scorer", ["--resume"]), "scorer", datasets=_ds())
    tnet2, tcfg2 = FN.load_fpd_teacher(rd)
    for k, v in tnet2.state_dict().items():
        assert torch.equal(v, tnet.state_dict()[k]), k
    assert {k: tcfg2[k] for k in FT.FINAL_CONFIG_KEYS} == {k: tcfg[k] for k in FT.FINAL_CONFIG_KEYS}
    assert tcfg2["recipe"] == tcfg["recipe"] and tcfg2["init_report"] == tcfg["init_report"]


class BadBevBase(FakeBase):
    """FakeBase whose rows in `bad` have an unreadable BEV (zeros + bev_ok False, as CK2Dataset.load_bev)."""

    def __init__(self, *args, bad=(), **kw):
        super().__init__(*args, **kw)
        self.bad = {int(r) for r in bad}

    def __getitem__(self, i):
        it = super().__getitem__(i)
        if it["row"] in self.bad:
            it["bev"], it["bev_ok"] = np.zeros_like(it["bev"]), False
        return it


def test_bank_rows_without_bev_are_retried_and_bounded(tmp_path):
    """A BEV-unreadable bank row (sampled 2) is never labelled: more than 1 % of a split unreadable stops stage bank
    before sample_done.json on every run (a rerun is not finalized without those rows), and a rerun tries them again."""
    rd = tmp_path / "det"

    def ds(bad):
        return {"fk_train": BadBevBase(6, seed=0, name="tr", bad=bad),
                "fk_val": FakeBase(4, rows=(0, 2, 3), seed=1, name="va")}

    FT.train_stage(_args(rd, "dit", ["--max-steps", "1"]), "dit", datasets=ds(()))
    bd = rd / "bank" / "fk_train"
    for _ in range(2):                                     # the first run, then a rerun with the BEV still unreadable
        with pytest.raises(SystemExit, match="no readable BEV"):
            FT.sample_bank(_args(rd, "bank"), datasets=ds({1}))
        assert not (bd / BK.F_SAMPLE_DONE).exists()
        st = np.load(bd / BK.F_SAMPLED)
        assert st[1] == 2 and (np.delete(st, 1) == 1).all() and np.isnan(np.load(bd / BK.F_TRAJ)[1]).all()
    keep = np.load(bd / BK.F_TRAJ).copy()
    FT.sample_bank(_args(rd, "bank"), datasets=ds(()))     # readable again: row 1 is sampled, the others kept
    d = json.loads((bd / BK.F_SAMPLE_DONE).read_text())
    assert d["n_sampled"] == 6 and d["n_bev_bad"] == 0 and (np.load(bd / BK.F_SAMPLED) == 1).all()
    tr = np.load(bd / BK.F_TRAJ)
    assert np.isfinite(tr).all() and np.array_equal(np.delete(tr, 1, 0), np.delete(keep, 1, 0))
    assert json.loads((rd / "bank" / "fk_val" / BK.F_SAMPLE_DONE).read_text())["n_sampled"] == 3


def test_compute_loss_masks_and_frozen_heads_get_no_grad(tmp_path):
    """score BCE only on the arm's keys over the 64-candidate set; a token without BEV is out of every term; every
    trainable parameter gets a finite gradient, the frozen heads none."""
    from tools.ck.data.ck_dataset import collate_ck
    a = _args(tmp_path / "x", "scorer", arm="M")
    assert a.score_keys == ["dac", "ep"]
    net = FN.FPDRefinerNet(seed=0, steps=DDIM)
    frozen = FT.freeze_for_stage(net, "scorer", a.score_keys)
    assert {n.split(".")[2] for n in frozen} == {"0", "3", "4"}          # nc, ttc, comfort heads
    base = FakeBase(2, seed=5)
    items = []
    for i in range(2):
        it = base[i]
        rng = np.random.default_rng(i)
        it.update(bank_traj=_paths(32, 4.0, rng), bank_y=_labels(rng, 32)[0], bank_ok=np.ones(32, bool))
        items.append(it)
    items[1]["bev_ok"] = False
    b = collate_ck(items)
    out = FT.forward(FT.StageModel(net, 0), b, "scorer", 0, 0, False, torch.device("cpu"))
    loss, st = FT.compute_loss(out, b, a, "scorer")
    assert st["n_bev_bad"] == 1.0 and st["n_pairs"] == 1.0 and st["n_anc_ok"] == 32 and st["n_bank_ok"] == 32
    from tools.ck import train_ck2 as T2
    y = torch.cat([b["y"], b["bank_y"]], 1)
    ok = torch.cat([b["y_ok"], b["bank_ok"]], 1) & torch.tensor([True, False])[:, None]
    ref, _ = T2.score_bce_keys(out["logit"], y, ok, ["dac", "ep"])
    assert st["score_bce"] == pytest.approx(float(ref.detach()), rel=1e-6)
    loss.backward()
    for n, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
        else:
            assert p.grad is None, n


# ----------------------------------------------------------------------------------------------- DDP (gloo, CPU)
def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _ddp_args(rd, stage):
    return FT.resolve_args(FT.get_parser().parse_args(
        ["--stage", stage, "--arm", "T", "--run-dir", str(rd), "--device", "cpu", "--workers", "0",
         "--tokens-per-batch", "4", "--epochs", "1", "--log-every", "1", "--ckpt-every", "0", "--eval-batch", "2",
         "--sample-batch", "2", "--warmup-steps", "1", "--clip", "0", "--lr", "1e-4", "--split", "fk_train",
         "--val-split", "fk_val", "--prior-rows", "50", "--steps", str(DDIM)]))


def _ddp_ds():
    return _ds(n_train=8, n_val=3, val_rows=(0, 1, 2))


def _ddp_worker(rank, world, port, rd, stage):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world), RANK=str(rank),
                      LOCAL_RANK=str(rank), OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    if stage == "bank":
        FT.sample_bank(_ddp_args(rd, stage), datasets=_ddp_ds())
    else:
        FT.train_stage(_ddp_args(rd, stage), stage, datasets=_ddp_ds())


def _ddp(rd, stage):
    import torch.multiprocessing as mp
    mp.start_processes(_ddp_worker, args=(2, _free_port(), str(rd), stage), nprocs=2, join=True, start_method="fork")
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        os.environ.pop(k, None)


def _steps(sd):
    return [r for r in _jsonl(Path(sd) / "train_log.jsonl") if r["kind"] == "step"]


def test_ddp_two_ranks_equals_single_process(tmp_path):
    if torch.cuda.is_available():
        # fork-based ranks fail once this process ran an autograd backward with CUDA visible: rerun alone on CPU
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp",
                            str(tmp_path / "child"), f"{__file__}::test_ddp_two_ranks_equals_single_process"],
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), cwd=REPO, capture_output=True, text=True,
                           timeout=1800)
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        return
    r1, r2 = tmp_path / "one", tmp_path / "two"
    FT.train_stage(_ddp_args(r1, "dit"), "dit", datasets=_ddp_ds())
    _ddp(r2, "dit")
    for stage in ("dit",):
        s1, s2 = _steps(r1 / stage), _steps(r2 / stage)
        assert len(s1) == len(s2) == 2 and all(r["world"] == 2 for r in s2) and all(r["world"] == 1 for r in s1)
        for x, y in zip(s1, s2):
            assert x["loss"] == pytest.approx(y["loss"], rel=1e-4, abs=1e-6)
            assert x["n_pairs"] == y["n_pairs"] == 4.0
            assert x["grad_norm"] == pytest.approx(y["grad_norm"], rel=1e-3)
    n1 = torch.load(r1 / "dit" / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
    n2 = torch.load(r2 / "dit" / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
    assert max(float((n1[k] - n2[k]).abs().max()) for k in n1 if n1[k].dtype.is_floating_point) < 1e-5
    v1, v2 = _jsonl(r1 / "dit" / "val_metrics.jsonl"), _jsonl(r2 / "dit" / "val_metrics.jsonl")
    assert v1[-1]["n_tokens"] == v2[-1]["n_tokens"] == 3
    assert v1[-1]["l_diff"] == pytest.approx(v2[-1]["l_diff"], rel=1e-3)
    # bank with 2 ranks == 1 rank (same DiT weights: r2 gets r1's dit stage), then labels, then the scorer stage
    shutil.rmtree(r2 / "dit")
    shutil.copytree(r1 / "dit", r2 / "dit")
    FT.sample_bank(_ddp_args(r1, "bank"), datasets=_ddp_ds())
    _ddp(r2, "bank")
    for split in ("fk_train", "fk_val"):
        a, b = (np.load(r / "bank" / split / BK.F_TRAJ) for r in (r1, r2))
        assert np.allclose(a, b, atol=1e-5, equal_nan=True)
        assert np.array_equal(np.load(r1 / "bank" / split / BK.F_SAMPLED), np.load(r2 / "bank" / split / BK.F_SAMPLED))
    shutil.rmtree(r2 / "bank")
    shutil.copytree(r1 / "bank", r2 / "bank")
    assert FL.main(["--run-dir", str(r1), "--workers", "2", "--scorer", "fake"]) == 0
    assert FL.main(["--run-dir", str(r2), "--workers", "2", "--scorer", "fake"]) == 0
    FT.train_stage(_ddp_args(r1, "scorer"), "scorer", datasets=_ddp_ds())
    _ddp(r2, "scorer")
    s1, s2 = _steps(r1 / "scorer"), _steps(r2 / "scorer")
    assert len(s1) == len(s2) == 2
    for x, y in zip(s1, s2):
        for k in ("loss", "score_bce", "l_diff"):
            assert x[k] == pytest.approx(y[k], rel=1e-4, abs=1e-6), k
        assert x["n_anc_ok"] == y["n_anc_ok"] == 128.0 and x["n_bank_ok"] == y["n_bank_ok"] == 128.0
    t1, _ = FN.load_fpd_teacher(r1)
    t2, _ = FN.load_fpd_teacher(r2)
    sd1, sd2 = t1.state_dict(), t2.state_dict()
    assert max(float((sd1[k] - sd2[k]).abs().max()) for k in sd1 if sd1[k].dtype.is_floating_point) < 1e-5


# ----------------------------------------------------------------------------------------------- real data
def _have_real(split=VAL):
    return all(p.is_file() for p in (REAL / "packed" / split / "tokens.parquet",
                                     REAL / "labels" / split / "raw256" / "labels.npy"))


real = pytest.mark.skipif(not _have_real(), reason="real CK2 val data absent")


@real
def test_canonical_order_is_sampler_epoch0_near_mid_real():
    from tools.ck.data.ck2_dataset import CK2Dataset, var_dir
    ds = CK2Dataset(VAL, bev="none", n_var=0, root=REAL, epoch=0)
    vd = var_dir(VAL, "var_separate_sampler16_accstraight", root=REAL)
    ix = np.load(vd / "index.npz") if (vd / "index.npz").is_file() else None
    for i in np.random.default_rng(0).choice(len(ds), 60, replace=False):
        it = ds[int(i)]
        r = it["row"]
        o = BK.canonical_near_mid(it)
        lab = ds.raw_labels(r)
        s = ds.sampler.sample(str(ds.tokens[r]), np.asarray(ds._arr("gt_traj")[r], np.float32), lab[:, CKI].T, epoch=0,
                              shuffle=False)
        assert np.array_equal(it["anc_idx"][o], s["idx"][:16]), r
        if ix is not None:
            assert np.array_equal(it["anc_idx"][o], ix["anchor_idx"][r].astype(np.int64)), r


def _mc_ok(split, base, n):
    from tools.ck.data import common as CM
    tt = BK.token_table(base)
    return all(CM.mc_path(split, tt.log.iat[int(r)], tt.token.iat[int(r)]).is_file() for r in base.rows[:n])


@real
def test_official_bank_labels_real_tokens(tmp_path):
    """The bank labeling path on 2 real navtrain_val tokens whose bank is their 32 epoch-0 anchors: the official labels
    equal the raw256 labels of those anchors bit for bit (same score_token, batching invariant)."""
    from tools.ck.data.ck2_dataset import CK2Dataset
    base = CK2Dataset(VAL, bev="none", n_var=0, root=REAL, limit=2)
    if not _mc_ok(VAL, base, 2):
        pytest.skip("navtrain metric cache of the fixture tokens absent")
    items = {int(base[i]["row"]): base[i] for i in range(2)}
    bd = tmp_path / "bank" / VAL
    _fake_bank(bd, base, list(items), lambda r: items[r]["cand"], split=VAL)
    meta = FL.label_split(bd, workers=2, scorer="official", log=lambda s: None)
    assert meta["n_labelled"] == 2 and meta["n_err"] == 0 and meta["n_ok_cands"] == 64 and meta["scorer_sha256"]
    lab, ok = np.load(bd / BK.F_LABELS, mmap_mode="r"), np.load(bd / BK.F_OK, mmap_mode="r")
    raw = np.load(REAL / "labels" / VAL / "raw256" / "labels.npy", mmap_mode="r")
    for r, it in items.items():
        assert np.array_equal(np.asarray(lab[r]), np.asarray(raw[r])[it["anc_idx"]]), r
        assert np.asarray(ok[r]).all()


def _have_real_bev(arm):
    env = dict(os.environ, CK_BEV64="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO, PYTHONDONTWRITEBYTECODE="1")
    code = ("import sys; from tools.ck.data.ck2_dataset import CK2Dataset as D\n"
            f"ok = all(D(s, bev='{arm}', n_var=0, limit=2)[i]['bev_ok'] for s in ('navtrain_train', '{VAL}') for i in (0, 1))\n"
            "sys.exit(0 if ok else 1)")
    try:
        return subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO, capture_output=True,
                              timeout=300).returncode == 0
    except Exception:
        return False


@real
@pytest.mark.parametrize("arm", ["T", "M"])
def test_cli_smoke_real_tokens(tmp_path, arm):
    """The CLI on real data in a CK_BEV64=1 process (the BEV grid is fixed at import): stage dit (2 steps), bank (4 +
    2 tokens), official labels (CPU pool), stage scorer (2 steps) -> fpd_net.load_fpd_teacher (strict)."""
    if not _have_real("navtrain_train") or not _have_real_bev(arm):
        pytest.skip(f"real data / BEV cache of arm {arm} absent")
    from tools.ck.data.ck2_dataset import CK2Dataset
    if not (_mc_ok("navtrain_train", CK2Dataset("navtrain_train", bev="none", n_var=0, root=REAL, limit=4), 4)
            and _mc_ok(VAL, CK2Dataset(VAL, bev="none", n_var=0, root=REAL, limit=2), 2)):
        pytest.skip("navtrain metric cache absent")
    rd = tmp_path / f"fpd_{arm}"
    env = dict(os.environ, CK_BEV64="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO, PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    common = ["--arm", arm, "--run-dir", str(rd), "--device", "cpu", "--workers", "0", "--limit-tokens", "4",
              "--val-limit", "2", "--tokens-per-batch", "2", "--epochs", "1", "--log-every", "1", "--ckpt-every", "0",
              "--eval-batch", "2", "--sample-batch", "2", "--steps", "2", "--warmup-steps", "1", "--resume"]
    tr = str(Path(REPO) / "tools/ck/fpd/train_fpd_teacher.py")
    cmds = [[sys.executable, tr, "--stage", "dit", *common], [sys.executable, tr, "--stage", "bank", *common],
            [sys.executable, str(Path(REPO) / "tools/ck/fpd/label_fpd_bank.py"), "--run-dir", str(rd), "--workers", "2"],
            [sys.executable, tr, "--stage", "scorer", *common]]
    for c in cmds:
        r = subprocess.run(c, env=env, cwd=REPO, capture_output=True, text=True, timeout=1200)
        assert r.returncode == 0, " ".join(c) + "\n" + r.stdout[-3000:] + r.stderr[-3000:]
    for split, n in (("navtrain_train", 4), (VAL, 2)):
        m = json.loads((rd / "bank" / split / BK.F_LMETA).read_text())
        assert m["n_sampled"] == n and m["n_err"] == 0 and m["scorer"] == "official"
        lab = np.load(rd / "bank" / split / BK.F_LABELS, mmap_mode="r")
        st = np.load(rd / "bank" / split / BK.F_LSTATE)
        assert (st == 1).sum() == n and np.isfinite(np.asarray(lab)[st == 1][..., :7]).all()
    net, cfg = FN.load_fpd_teacher(rd)
    assert cfg["arm"] == arm and cfg["trainer"] == FN.FPD_TEACHER_TRAINER and cfg["bev_shape"] == [256, 100, 100]
    assert cfg["score_keys"] == (["nc", "ep", "ttc", "comfort"] if arm == "T" else ["dac", "ep"])
    assert cfg["split"] == "navtrain_train" and cfg["ep_target"] == "official" and cfg["steps"] == 2
    val = _jsonl(rd / "scorer" / "val_metrics.jsonl")
    assert val[-1]["n_tokens"] == 2 and val[-1]["n_bev_bad"] == 0 and "auc_keys_mean" in val[-1]


# ----------------------------------------------------------------------------------------------- launcher (dry run)
SETUP_SH = Path(REPO) / "tools/release/setup/run_fpd_teachers.sh"


def _script_env(tmp_path, **kw):
    """tests/test_release_setup.py script_env (roots under tmp_path/srv, SETUP_DRY=1, torchrun = echo) + the FPD run
    dirs under tmp_path: tools/release/paths.env, when present, overrides the SSR_* roots (load_paths.sh)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SSR_", "NAVSIM_", "OPENSCENE", "NUPLAN", "FPD_"))}
    root = tmp_path / "srv"
    logs = root / "ds/navsim_logs/trainval"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "x.pkl").write_bytes(b"")
    env.update(SSR_DATA_ROOT=str(root / "data"), SSR_TEACHER_CACHE=str(root / "tc"), OPENSCENE_DATA_ROOT=str(root / "ds"),
               SSR_TEST_DATA_ROOT=str(root / "dst"), SSR_PLAN_DATA=str(root / "plan"), SSR_NAVTEST_MC=str(root / "nmc"),
               SSR_PY=sys.executable, SSR_TORCHRUN="/bin/echo", SSR_RUN_ROOT=str(root / "runs"), NAVSIM_V11=str(root / "v11"),
               NAVSIM_V11_CACHE=str(root / "v11c"), NAVSIM_V22_POST=str(root / "a"), NAVSIM_V22_PRE=str(root / "b"),
               NAVSIM_V22_CACHE=str(root / "c"), SETUP_DRY="1", PYTHONDONTWRITEBYTECODE="1",
               FPD_DET_RUN=str(tmp_path / "fpd/det"), FPD_MAP_RUN=str(tmp_path / "fpd/map"))
    env.update(kw)
    return env


def _launch(env):
    return subprocess.run(["bash", str(SETUP_SH)], env=env, capture_output=True, text=True, timeout=300)


def _cmds(out):
    return [x[2:] for x in out.splitlines() if x.startswith("+ ")]


def test_launcher_recipe_dry_run(tmp_path):
    env = _script_env(tmp_path)
    r = _launch(env)
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = _cmds(r.stdout)
    for arm, name, gpus, keys in (("T", "det", "0,1,2,3", "nc,ttc,ep,comfort"), ("M", "map", "4,5,6,7", "dac,ep")):
        rd = tmp_path / "fpd" / name
        mine = [c for c in cmds if f" {rd}" in c and "mkdir" not in c]
        tr = [c for c in mine if "train_fpd_teacher.py" in c]
        lb = [c for c in mine if "label_fpd_bank.py" in c]
        assert [c.split("--stage ")[1].split()[0] for c in tr] == ["dit", "bank", "scorer"] and len(lb) == 1
        assert mine.index(lb[0]) == 2                                    # dit, bank, labels, scorer
        assert "CUDA_VISIBLE_DEVICES= " in lb[0] and "--workers 48" in lb[0]
        assert "--splits navtrain_train,navtrain_val" in lb[0]
        for c in tr:
            assert f"CUDA_VISIBLE_DEVICES={gpus}" in c and "--nproc_per_node=4" in c and f"--score-keys {keys}" in c
            a = FT.resolve_args(FT.get_parser().parse_args(c.split("train_fpd_teacher.py ", 1)[1].split()))
            assert (a.arm, a.run_dir, a.epochs, a.tokens_per_batch, a.lr, a.wd, a.warmup_steps, a.clip) == \
                (arm, str(rd), 10, 16, 2e-4, 0.01, 500, 1.0)
            assert (a.split, a.val_split, a.eval_every, a.resume, a.steps, a.grid, a.amp, a.seed) == \
                ("navtrain_train", "navtrain_val", 2, True, 8, 8, 0, 0)
            assert a.score_keys == [k for k in Cn.CK_KEYS if k in keys.split(",")]
            assert a.bank_splits == ["navtrain_train", "navtrain_val"]
    assert "ck_e2e2.fpd.teacher_det_run" in r.stdout
    assert not (tmp_path / "fpd").exists()                              # nothing written under SETUP_DRY=1
    # the default run dirs: <SSR_TEACHER_CACHE>/refiner_fpd/{det,map} (the trainer's default_run_dir); skipped once a
    # finished FPD teacher sits there (always refused)
    roots = [Path(env["SSR_TEACHER_CACHE"])]
    pe = Path(REPO) / "tools/release/paths.env"
    if pe.is_file():
        roots += [Path(x.split("=", 1)[1].strip()) for x in pe.read_text().splitlines()
                  if x.startswith("SSR_TEACHER_CACHE=")]
    if not any((t / "refiner_fpd" / n / "done.json").exists() for t in roots for n in ("det", "map")):
        out = _launch(dict({k: v for k, v in env.items() if not k.startswith("FPD_")}, RESUME="1")).stdout
        rds = {c.split("--run-dir ")[1].split()[0] for c in _cmds(out) if "train_fpd_teacher.py" in c}
        assert {(Path(x).parent.name, Path(x).name) for x in rds} == {("refiner_fpd", "det"), ("refiner_fpd", "map")}


def test_launcher_guards(tmp_path):
    env = _script_env(tmp_path)
    rd = tmp_path / "fpd/det"
    (rd / "dit").mkdir(parents=True)
    (rd / "dit/config.json").write_text("{}")
    r = _launch(env)
    assert r.returncode == 3 and "RESUME=1" in r.stderr
    assert _launch(dict(env, RESUME="1")).returncode == 0
    (rd / "done.json").write_text("{}")
    r = _launch(dict(env, RESUME="1"))
    assert r.returncode == 3 and "done.json" in r.stderr
    r = _launch(dict(env, WHICH="map", GPUS_MAP="0,1,2"))
    assert r.returncode == 3 and "2 or 4" in r.stderr
    out = _launch(dict(env, WHICH="map")).stdout
    assert "--arm T" not in out and "--arm M" in out
    for bad in ("0", "49", "x"):
        r = _launch(dict(env, WHICH="map", LABEL_WORKERS=bad))
        assert r.returncode == 3 and "LABEL_WORKERS" in r.stderr, bad
    other = tmp_path / "elsewhere/mapM"
    out = _launch(dict(env, WHICH="map", FPD_MAP_RUN=str(other), GPUS_MAP="2,3")).stdout
    assert f"--run-dir {other}" in out and "CUDA_VISIBLE_DEVICES=2,3" in out and "--nproc_per_node=2" in out
    r = _launch(dict(env, WHICH="all"))
    assert r.returncode == 3 and "WHICH" in r.stderr


def test_cli_refuses_without_bev64_and_writes_nothing(tmp_path):
    """real datasets need CK_BEV64=1 (the teachers read the 100 x 100 / 64 m BEVs): refused before any output."""
    env = {k: v for k, v in os.environ.items() if k != "CK_BEV64"}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO, PYTHONDONTWRITEBYTECODE="1")
    for stage in ("dit", "bank", "scorer"):
        rd = tmp_path / stage
        r = subprocess.run([sys.executable, str(Path(REPO) / "tools/ck/fpd/train_fpd_teacher.py"), "--stage", stage,
                            "--arm", "T", "--device", "cpu", "--run-dir", str(rd)], env=env, cwd=REPO,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode != 0 and "CK_BEV64=1" in r.stderr, r.stderr[-2000:]
        assert not rd.exists()
