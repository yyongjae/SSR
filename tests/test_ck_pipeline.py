"""CK pipeline tests (CPU): train_ck steps on a synthetic in-memory dataset (arms T / S, KD on / off, corr_aug on),
resume, infer_ck on the synthetic set, kd_targets combine on synthetic arrays, eval_ck variants / bootstrap on a toy
label table with known answers.  No dump / label files are needed; the surrogate uses real GT of two navtrain train
tokens when GTLoader can read them (else ref_gt_ok False: the surrogate path is skipped, counted in stats n_gt)."""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[1])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

core = pytest.importorskip("navsim.agents.para_ssr.ck.model")

from navsim.agents.para_ssr.ck import constants as Cn  # noqa: E402
from tools.ck import ckutil as U  # noqa: E402
from tools.ck import eval_ck, infer_ck, kd_targets, train_ck  # noqa: E402

K = 4
TRAIN_TOKENS = ("1aa44d46e4ab5bc7", "2570fbfdf1835706")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("CK_DATA_ROOT", str(tmp_path / "ckdata"))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)


def _gt(token: str):
    try:
        from navsim.agents.para_ssr.refiner.e2e import GTLoader
        d = GTLoader(Cn.DATA_ROOT).load(token)
        if bool(d["ref_gt_ok"]):
            return {k: v.numpy() if torch.is_tensor(v) else v for k, v in d.items()}
    except Exception:
        pass
    from navsim.agents.para_ssr.refiner.e2e import GTLoader
    g = GTLoader.__new__(GTLoader)
    from navsim.agents.para_ssr.refiner import data as RD
    g._A, g._CL, g._NKF, g._EH, g._EW = RD.A_MAX, RD.CL_MAX, RD.N_KF, RD.E_H, RD.E_W
    d = {f"ref_{k}": np.asarray(v) for k, v in g._empty().items()}
    d["ref_gt_ok"] = np.asarray(False)
    return d


def straight(v: float, y: float = 0.0, k: int = K):
    t = np.arange(1, 9) * 0.5
    out = np.zeros((k, 8, 3), np.float32)
    for j in range(k):
        out[j, :, 0] = (v * (1 - 0.1 * j)) * t
        out[j, :, 1] = y + 0.2 * j * t / 4
    return out


class FakeCK(torch.utils.data.Dataset):
    """CKDataset-like synthetic items (contract data.ck_dataset item keys)."""

    def __init__(self, n=6, k=K, gt=True, kd=False, corr_aug=False, seed=0, rows=None):
        self.n, self.k, self.gt, self.kd, self.corr_aug = n, k, gt, kd, corr_aug
        rng = np.random.default_rng(seed)
        self.bev = rng.standard_normal((n, 256, 50, 100)).astype(np.float16) * 0.5
        self.cand = np.stack([straight(5 + i) for i in range(n)])
        self.y = (rng.random((n, k, 5)) > 0.2).astype(np.float32)
        self.y[..., 2] = rng.random((n, k)).astype(np.float32)          # ep continuous
        self.y_ok = np.ones((n, k), bool)
        self.y_ok[0, -1] = False
        self.pdms = rng.random((n, k)).astype(np.float32)
        self.v2_final = rng.standard_normal((n, k)).astype(np.float32)
        self.v2_im = rng.dirichlet(np.ones(k), n).astype(np.float32)
        self.status = np.zeros((n, 8), np.float32)
        self.status[:, 1] = 1
        self.status[:, 4] = 5 + np.arange(n)
        self.tokens = [TRAIN_TOKENS[i % 2] + f"_{i}" for i in range(n)]
        self.gts = [_gt(TRAIN_TOKENS[i % 2]) for i in range(n)] if gt else None
        self.rows_ = np.arange(n) if rows is None else np.asarray(rows)
        self.tokens_df = pd.DataFrame({"token": self.tokens, "log": [f"log{i % 3}" for i in range(n)],
                                       "city": "x", "row": np.arange(n)})
        self.packed_ok = np.ones(n, bool)

    def subset(self, rows):
        d = FakeCK.__new__(FakeCK)
        d.__dict__.update(self.__dict__)
        d.rows_ = np.asarray(rows)
        return d

    def __len__(self):
        return len(self.rows_)

    def __getitem__(self, i):
        r = int(self.rows_[i])
        it = {"token": self.tokens[r], "row": r, "bev": self.bev[r], "bev_ok": True, "cand": self.cand[r],
              "status": self.status[r], "v2_final": self.v2_final[r], "v2_im": self.v2_im[r],
              "v2_sim": self.y[r], "y": self.y[r], "y_ok": self.y_ok[r], "y_pdms": self.pdms[r]}
        if self.gt:
            it.update(self.gts[r])
            it["gt_traj"] = straight(6.0, k=1)[0]
        if self.kd:
            it.update(kd_score_prob=np.clip(self.y[r] * 0.8 + 0.1, 0, 1).astype(np.float32),
                      kd_c_lon=-0.1 * np.ones((self.k, 6), np.float32), kd_e_lat=0.05 * np.ones((self.k, 6), np.float32),
                      kd_ok=np.ones(self.k, bool))
        if self.corr_aug:
            ct = self.cand[r].copy()
            ct[..., 1] += 0.3
            it.update(corr_traj=ct, y_corr=self.y[r][::-1].copy(), y_corr_ok=np.ones(self.k, bool),
                      kd_score_prob_corr=np.full((self.k, 5), 0.7, np.float32))
        return it


def collate(items):
    try:
        from tools.ck.data.ck_dataset import collate_ck
        return collate_ck(items)
    except ImportError:
        pass
    out = {"tokens": [it["token"] for it in items], "rows": torch.as_tensor([it["row"] for it in items])}
    for k in items[0]:
        if k in ("token", "row"):
            continue
        out[k] = torch.as_tensor(np.stack([np.asarray(it[k]) for it in items]))
    return out


def make_args(tmp_path, arm, kd="off", corr_aug="off", **kw):
    a = train_ck.get_parser().parse_args(["--arm", arm, "--run", f"t_{arm}_{kd}_{corr_aug}", "--device", "cpu",
                                          "--workers", "0", "--tokens-per-batch", "2", "--k", str(K),
                                          "--init-from", "none", "--kd", kd, "--corr-aug", corr_aug,
                                          "--out-root", str(tmp_path / "train"), "--max-steps", "3",
                                          "--log-every", "1", "--eval-batch", "3", "--warmup-steps", "1"]
                                         + [x for kv in kw.items() for x in (f"--{kv[0].replace('_', '-')}", str(kv[1]))])
    if kd == "on":
        a.kd_dir = str(tmp_path / "kd")
    return train_ck.resolve_args(a, kd_exists=lambda d: True)


NORM = (np.zeros(256, np.float32), np.ones(256, np.float32))


@pytest.mark.parametrize("arm,kd,corr", [("T", "off", "off"), ("S", "on", "on"), ("S", "off", "off")])
def test_train_steps(tmp_path, arm, kd, corr):
    a = make_args(tmp_path, arm, kd, corr)
    ds = FakeCK(6, gt=True, kd=kd == "on", corr_aug=corr == "on")
    ev = FakeCK(4, gt=False, seed=1)
    run = train_ck.train(a, datasets={"train": ds, "evals": [("navtrain_val", ev)]}, collate_fn=collate,
                         norm_override=NORM if arm == "T" else None)
    recs = [json.loads(l) for l in (run / "train_log.jsonl").read_text().splitlines()]
    steps = [r for r in recs if r["kind"] == "step"]
    assert len(steps) == 3
    for r in steps:
        assert np.isfinite(r["loss"]) and r.get("skipped", 0) == 0
        assert np.isfinite(r["score_bce"])
        if kd == "on":
            assert "kd_score" in r and "kd_ctrl" in r and np.isfinite(r["kd_ctrl"])
        if corr == "on":
            assert "corr_score_bce" in r
    if any(bool(g["ref_gt_ok"]) for g in ds.gts):
        assert steps[0]["n_gt"] > 0 and "sur" in steps[0] and np.isfinite(steps[0]["sur"])
    val = [json.loads(l) for l in (run / "val_metrics.jsonl").read_text().splitlines()]
    assert val and val[-1]["split"] == "navtrain_val" and "pdms_a_b1" in val[-1]
    assert (run / "ckpt_last.pt").is_file() and (run / "done.json").is_file()
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["arm"] == arm and cfg["kd"] == kd
    if arm == "T":
        assert (run / "norm.npz").is_file()
    # the run reloads with the core loader (strict) and runs inference on the synthetic set
    out = infer_ck.run_infer(run, "navtrain_val", out=tmp_path / "inf", device="cpu", batch=3, workers=0,
                             dataset=FakeCK(5, gt=False, seed=2), collate_fn=collate)
    done = np.load(out / "done.npy")
    assert done.all()
    s = np.load(out / "score_logit.npy")
    ct = np.load(out / "corr_traj.npy")
    cs = np.load(out / "corr_score_logit.npy")
    assert s.shape == (5, K, 5) and np.isfinite(s).all() and np.isfinite(ct).all() and np.isfinite(cs).all()
    assert pd.read_parquet(out / "tokens.parquet")["row"].tolist() == list(range(5))


def test_train_resume_and_done(tmp_path):
    a = make_args(tmp_path, "S", "off", "off")
    ds = FakeCK(8, gt=False)
    a.max_steps = 2
    a.ckpt_every = 1
    run = train_ck.train(a, datasets={"train": ds, "evals": []}, collate_fn=collate)
    ck = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 2 and ck["batch_in_epoch"] == 2 and ck["epoch"] == 0
    # done.json -> a second call is a no-op; without --resume an existing ckpt is refused
    assert train_ck.train(a, datasets={"train": ds, "evals": []}, collate_fn=collate) == run
    (run / "done.json").unlink()
    with pytest.raises(SystemExit):
        train_ck.train(a, datasets={"train": ds, "evals": []}, collate_fn=collate)
    a.resume, a.max_steps = True, 5
    train_ck.train(a, datasets={"train": ds, "evals": []}, collate_fn=collate)
    ck = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 5


def _val_recs(run):
    p = run / "val_metrics.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


def test_kill_during_eval_resumes_eval(tmp_path, monkeypatch):
    """A kill during the epoch-end evaluations must not lose that epoch's val_metrics records (review finding):
    --resume first re-runs the missing evaluations, then continues; det_gap sees the final epoch."""
    a = make_args(tmp_path, "S", "off", "off")
    a.max_steps, a.epochs = 0, 2                         # 2 whole epochs of 4 // 2 = 2 steps
    ds = FakeCK(4, gt=False)
    evals = [("navtrain_val", FakeCK(3, gt=False, seed=1)), ("navtrain_train_train_eval", FakeCK(3, gt=False, seed=2))]
    real_eval = train_ck.evaluate
    calls = {"n": 0}

    def dying_eval(*args, **kw):
        calls["n"] += 1
        if calls["n"] == 4:                              # epoch 1 (last): val done, train_eval killed
            raise KeyboardInterrupt("killed during eval")
        return real_eval(*args, **kw)

    monkeypatch.setattr(train_ck, "evaluate", dying_eval)
    with pytest.raises(KeyboardInterrupt):
        train_ck.train(a, datasets={"train": ds, "evals": evals}, collate_fn=collate)
    run = Path(a.out_root) / a.run
    assert not (run / "done.json").is_file()
    ck = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["epoch"] == 2 and ck["eval_pending"]["epoch"] == 1
    assert {(r["epoch"], r["split"]) for r in _val_recs(run)} == {
        (0, "navtrain_val"), (0, "navtrain_train_train_eval"), (1, "navtrain_val")}
    monkeypatch.setattr(train_ck, "evaluate", real_eval)
    a.resume = True
    train_ck.train(a, datasets={"train": ds, "evals": evals}, collate_fn=collate)
    recs = _val_recs(run)
    assert sorted((r["epoch"], r["split"]) for r in recs) == sorted(
        (e, s) for e in (0, 1) for s in ("navtrain_val", "navtrain_train_train_eval"))
    assert all(r["n_bev_bad"] == 0 for r in recs)
    ck = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["eval_pending"] is None and ck["epoch"] == 2
    assert (run / "ckpt_ep0.pt").is_file() and (run / "ckpt_ep1.pt").is_file() and (run / "done.json").is_file()
    g = kd_targets.det_gap(run)
    assert g["epoch"] == 1 and g["final_epoch_ok"] is True and "gap_train_minus_val" in g


def test_det_gap_flags_missing_last_epoch(tmp_path):
    run = tmp_path / "det"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"epochs": 3, "max_steps": 0}))
    rec = {"pdms_a_b1": 0.9, "partial_epoch": False}
    lines = [dict(rec, epoch=e, split=s) for e in (0, 1) for s in ("navtrain_val", "navtrain_train_train_eval")]
    (run / "val_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    g = kd_targets.det_gap(run)
    assert g["epoch"] == 1 and g["expected_last_epoch"] == 2 and g["final_epoch_ok"] is False


class BadBev(FakeCK):
    def __getitem__(self, i):
        it = super().__getitem__(i)
        if int(self.rows_[i]) % 3 == 0:
            it["bev"] = np.zeros_like(it["bev"])
            it["bev_ok"] = False
        return it


def test_bev_bad_counted_and_stops(tmp_path, monkeypatch):
    a = make_args(tmp_path, "S", "off", "off")
    ds = BadBev(6, gt=False)
    with pytest.raises(SystemExit, match="no readable BEV"):
        train_ck.train(a, datasets={"train": ds, "evals": []}, collate_fn=collate)
    run = Path(a.out_root) / a.run
    recs = [json.loads(l) for l in (run / "train_log.jsonl").read_text().splitlines()]
    assert sum(r["n_bev_bad"] for r in recs if r["kind"] == "step") >= 1
    ep = [r for r in recs if r["kind"] == "epoch"][-1]
    assert ep["n_bev_bad"] >= 1
    # below the 1 % threshold the run goes on (counts still logged)
    monkeypatch.setattr(train_ck, "BEV_BAD_MAX", 0.5)
    a2 = make_args(tmp_path / "b", "S", "off", "off")
    run2 = train_ck.train(a2, datasets={"train": BadBev(6, gt=False), "evals": [("navtrain_val", BadBev(3, gt=False))]},
                          collate_fn=collate)
    assert (run2 / "done.json").is_file()
    assert _val_recs(run2)[-1]["n_bev_bad"] == 1


def test_epoch_batches_resume_order():
    full = train_ck.EpochBatches(10, 3, seed=1, epoch=2).batches()
    assert len(full) == 3 and all(len(b) == 3 for b in full)
    tail = list(train_ck.EpochBatches(10, 3, seed=1, epoch=2, start=1))
    assert [list(b) for b in full[1:]] == tail
    assert train_ck.EpochBatches(10, 3, seed=1, epoch=3).batches()[0].tolist() != full[0].tolist()


def test_log_stratified_rows():
    logs = np.array(["a"] * 50 + ["b"] * 30 + ["c"] * 20)
    r = U.log_stratified_rows(logs, 10, seed=0)
    assert len(r) == 10 and len(set(r)) == 10
    assert {"a", "b", "c"} <= set(logs[r])
    assert U.log_stratified_rows(logs, 0).tolist() == list(range(100))


def test_score_metrics_known():
    n = 3
    y = np.ones((n, K, 5), np.float32)
    y[0, 0, 0] = 0          # cand 0 of token 0 collides
    ok = np.ones((n, K), bool)
    pdms = np.full((n, K), 0.5, np.float32)
    pdms[0, 0], pdms[0, 2] = 0.0, 0.9
    prob = np.full((n, K, 5), 0.9, np.float32)
    prob[0, 0, 0] = 0.01    # CK flags the collision of cand 0
    prob[0, 2, :] = 0.99
    v2f = np.zeros((n, K), np.float32)
    im = np.full((n, K), 1.0 / K, np.float32)
    m = train_ck.score_metrics(prob, y, ok, pdms, v2f, im)
    assert m["pdms_v2"] == pytest.approx((0.0 + 0.5 + 0.5) / 3)
    assert m["pdms_oracle"] == pytest.approx((0.9 + 0.5 + 0.5) / 3)
    assert m["pdms_a_b1"] == pytest.approx((0.9 + 0.5 + 0.5) / 3)
    assert m["auc_fail_nc"] == pytest.approx(1.0)


def _write_teacher(d: Path, n: int, k: int, logit: float, c: float, e: float, z: float = 0.0, w: float = 0.0):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "done.npy", np.ones(n, bool))
    np.save(d / "done_on_kdcorr.npy", np.ones(n, bool))
    np.save(d / "score_logit.npy", np.full((n, k, 5), logit, np.float16))
    np.save(d / "score_logit_on_kdcorr.npy", np.full((n, k, 5), logit + 1, np.float16))
    np.save(d / "c_lon.npy", np.full((n, k, 6), c, np.float32))
    np.save(d / "e_lat.npy", np.full((n, k, 6), e, np.float32))
    np.save(d / "z_lon.npy", np.full((n, k, 6), z, np.float32))
    np.save(d / "w_lat.npy", np.full((n, k, 6), w, np.float32))


def test_kd_combine_rule(tmp_path):
    n, k = 3, K
    out = tmp_path / "kd"
    _write_teacher(out / "det", n, k, logit=0.0, c=-0.5, e=0.1)           # p_det = 0.5
    _write_teacher(out / "map", n, k, logit=2.0, c=-9.0, e=0.7)           # p_map = 0.881
    np.save(out / "kd_corr_done.npy", np.ones(n, bool))
    meta = kd_targets.combine(out, True, n, k, {"tag": "t"})
    p = np.load(out / "kd_score_prob.npy")
    pm = 1 / (1 + np.exp(-2.0))
    keys = list(Cn.CK_KEYS)
    assert p[..., keys.index("nc")] == pytest.approx(0.5)
    assert p[..., keys.index("ttc")] == pytest.approx(0.5)
    assert np.allclose(p[..., keys.index("dac")], pm, atol=1e-3)
    assert np.allclose(p[..., keys.index("ep")], 0.5 * (0.5 + pm), atol=1e-3)
    assert np.allclose(np.load(out / "kd_c_lon.npy"), -0.5)               # lon from DET
    assert np.allclose(np.load(out / "kd_e_lat.npy"), 0.7)                # lat from MAP
    pc = np.load(out / "kd_score_prob_corr.npy")
    assert pc[..., keys.index("nc")] == pytest.approx(1 / (1 + np.exp(-1.0)), abs=1e-3)
    assert np.load(out / "kd_ok.npy").all() and meta["kd_ok_frac"] == 1.0
    # no MAP teacher -> DET for everything
    out2 = tmp_path / "kd2"
    _write_teacher(out2 / "det", n, k, logit=0.0, c=-0.5, e=0.1)
    np.save(out2 / "kd_corr_done.npy", np.ones(n, bool))
    kd_targets.combine(out2, False, n, k, {})
    assert np.allclose(np.load(out2 / "kd_score_prob.npy"), 0.5)
    assert np.allclose(np.load(out2 / "kd_e_lat.npy"), 0.1)


def test_kd_corr_rows_identity(tmp_path):
    n, k = 4, K
    cand = np.stack([straight(6.0) for _ in range(n)])
    status = np.zeros((n, 8), np.float32)
    status[:, 4] = 6.0
    z = np.zeros((n, k, 6), np.float32)
    w = np.zeros((n, k, 6), np.float32)
    kd_targets.kd_corr_rows(tmp_path, cand, status, z, w, np.arange(n), torch.device("cpu"))
    kc = np.load(tmp_path / "kd_corr_traj.npy")
    assert np.array_equal(kc, cand)                       # z = w = 0 -> identity
    assert np.load(tmp_path / "kd_corr_done.npy").all()


def test_log_bootstrap():
    rng = np.random.default_rng(0)
    logs = np.repeat(np.arange(40), 25).astype(str)
    d = rng.normal(0.02, 0.05, len(logs))
    b = eval_ck.log_bootstrap(d, logs, 500, 0)
    assert b["n"] == 1000 and b["n_logs"] == 40
    assert b["lo"] < b["mean"] < b["hi"] and b["lo"] > 0.0
    z = eval_ck.log_bootstrap(np.zeros(100), np.repeat(np.arange(10), 10), 200, 0)
    assert z["lo"] == z["hi"] == 0.0


def test_evaluate_arrays_known():
    n, k = 4, K
    L = Cn.LBL
    lab = np.ones((n, k, len(Cn.LABEL_COLS)), np.float64)
    lab[..., L["pdms"]] = 0.5
    lab[:, 0, L["nc"]] = 0.0          # v2 choice collides everywhere
    lab[:, 0, L["pdms"]] = 0.0
    lab[:, 1, L["pdms"]] = 0.8
    lab_c = lab.copy()
    lab_c[..., L["pdms"]] = 0.6
    lab_c[:, 3, L["pdms"]] = 0.95
    prob = np.full((n, k, 5), 0.9)
    prob[:, 0, 0] = 0.01
    prob[:, 1, :] = 0.99
    prob_c = np.full((n, k, 5), 0.5)
    prob_c[:, 3, :] = 0.999           # the corrected cand 3 is the CK best of the pool
    v2f = np.zeros((n, k))
    v2f[:, 0] = 5.0                   # v2 prefers cand 0
    im = np.full((n, k), 0.25)
    logs = np.array(["l0", "l0", "l1", "l1"])
    lead = np.array([True, False, True, False])
    res = eval_ck.evaluate_arrays(lab, lab_c, prob, prob_c, v2f, im, logs, lead, [0.0, 1.0], n_boot=100)
    rows = {(r["variant"], r["beta"]): r for r in res["rows"]}
    assert rows[("v2", None)]["pdms"] == pytest.approx(0.0)
    assert rows[("v2", None)]["fail_nc"] == pytest.approx(1.0)
    assert rows[("v2", None)]["lead_fail_nc_ttc"] == pytest.approx(1.0) and rows[("v2", None)]["lead_n"] == 2
    assert rows[("oracle16", None)]["pdms"] == pytest.approx(0.8)
    assert rows[("oracle32", None)]["pdms"] == pytest.approx(0.95)
    assert rows[("a", 0.0)]["pdms"] == pytest.approx(0.0)          # beta 0 = v2 ranking
    assert rows[("a", 1.0)]["pdms"] == pytest.approx(0.8)          # CK picks cand 1
    assert rows[("b", 1.0)]["pdms"] == pytest.approx(0.6)          # its correction
    assert rows[("c", 1.0)]["pdms"] == pytest.approx(0.95)         # corrected cand 3 from the pool
    assert rows[("a", 1.0)]["d_pdms"]["mean"] == pytest.approx(0.8)
    assert rows[("a", 1.0)]["lead_fail_nc_ttc"] == pytest.approx(0.0)
    assert res["best_beta"] == {"a": 1.0, "b": 1.0, "c": 1.0}
    md = eval_ck.table_md(res, "navtrain_val", "toy", dict(n_eval=n, n_total=n, have_corr=True, n_logs=2, n_boot=100))
    assert "| a | 1 |" in md


def test_open_memmap_shared(tmp_path):
    p = tmp_path / "x.npy"
    a = U.open_memmap(p, (5, 2), np.float32, fill=np.nan)
    b = U.open_memmap(p, (5, 2), np.float32)
    a[1] = 3.0
    a.flush()
    assert np.isnan(b[0]).all() and (np.load(p)[1] == 3.0).all()
    with pytest.raises(ValueError):
        U.open_memmap(p, (4, 2), np.float32)


def test_gpu_guard(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    with pytest.raises(SystemExit):
        U.gpu_guard("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(SystemExit):
        U.gpu_guard("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    U.gpu_guard("cuda")
    U.gpu_guard("cpu")
