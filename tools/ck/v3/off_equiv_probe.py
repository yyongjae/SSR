#!/usr/bin/env python
"""refiner-v3 OFF-equivalence probe (docs/refiner_v3/SPEC.md 4.7; Integrate phase).

Runs the SAME probe code against one repo tree and writes every observable to an npz, so that the CK2 reference tree
(D3/ref_8ae8b1d = `git archive 8ae8b1d` + the Phase-0 pristine tools/ck/data copies) and the refiner-v3 worktree
can be compared bitwise by tests/test_v3_integration_off.py.  Run each tree in its own interpreter with `python -B`
(no __pycache__ anywhere) and PYTHONPATH unset; the probe puts --repo first on sys.path.

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /venv/ssr/bin/python -B tools/ck/v3/off_equiv_probe.py \
      --repo /home/external-user/ssd/yongjae_refiner/ck/v3/ref_8ae8b1d --out <dir>/ref.npz --work <dir>/ref_work

Probe content (temporal OFF everywhere: the default of every new kwarg / flag)
  model   build_ck('T', seed 0, fixed norm): every state_dict tensor; param_counts; CPU forward (cand + extra +
          decode + rescore_corr) on fixed synthetic inputs; ck2T10 ckpt_last via load_ck (strict) + the same forward;
          train_ck2.freeze_unused trainable count.
  data    CK2Dataset(navtrain_val, bev none, n_var 32, gt True) [train style] and CKDataset(navtrain_val, bev T, k 16,
          labels cand) [r34 style]: 64 tokens x epochs 0, 1, collated in batches of 8 -> sha256 of every key's bytes.
  train   train_ck2.train on synthetic in-memory data (CPU, 2 tokens / step): --max-steps 1 and --max-steps 3 runs ->
          per-step loss / grad_norm / every logged numeric stat, final model + optimiser state, val metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def _setup(repo: str):
    repo = str(Path(repo).resolve())
    sys.path.insert(0, repo)
    import navsim
    assert navsim.__file__.startswith(repo), (navsim.__file__, repo)
    return repo


def _sha(a) -> str:
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().contiguous().numpy()
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:32]


def _norm():
    import numpy as np
    rng = np.random.default_rng(3)
    return rng.normal(0, 0.3, 256).astype(np.float32), rng.uniform(0.5, 2.0, 256).astype(np.float32)


def _inputs():
    import torch
    g = torch.Generator().manual_seed(11)
    T, K, K2 = 2, 6, 5
    bev = torch.randn(T, 256, 50, 100, generator=g) * 0.5
    t = torch.arange(1, 9, dtype=torch.float32) * 0.5
    v = 3.0 + 6.0 * torch.rand(T, K, 1, generator=g)
    k = 0.02 * torch.randn(T, K, 1, generator=g)
    x = v * t
    cand = torch.stack([x, 0.5 * k * x ** 2, torch.atan(k * x)], -1)                   # [T, K, 8, 3]
    v2 = 3.0 + 6.0 * torch.rand(T, K2, 1, generator=g)
    x2 = v2 * t
    extra = torch.stack([x2, 0.3 + 0.0 * x2, 0.0 * x2], -1)
    status = torch.zeros(T, 8)
    status[:, 1] = 1.0
    status[:, 4] = v[:, 0, 0]
    return bev, cand, status, extra


def _flat_out(prefix: str, out: dict, res: dict) -> None:
    import torch
    for key, v in out.items():
        if isinstance(v, torch.Tensor):
            res[f"{prefix}{key}"] = v.detach().float().numpy()
        elif isinstance(v, dict):
            _flat_out(f"{prefix}{key}.", v, res)
        elif isinstance(v, (tuple, list)):
            for i, x in enumerate(v):
                if isinstance(x, torch.Tensor):
                    res[f"{prefix}{key}.{i}"] = x.detach().float().numpy()


def probe_model(res: dict, meta: dict) -> None:
    import numpy as np
    import torch
    from navsim.agents.para_ssr.ck.model import build_ck, load_ck
    from tools.ck import train_ck2 as T2
    torch.manual_seed(0)
    net = build_ck("T", seed=0, norm=_norm()).eval()
    for n, t in net.state_dict().items():
        res[f"sd.{n}"] = t.detach().numpy().copy()
    meta["param_counts"] = {k: int(v) for k, v in net.param_counts().items()}
    meta["sd_keys"] = list(net.state_dict().keys())
    bev, cand, status, extra = _inputs()
    with torch.no_grad():
        o = net(bev, cand, status, extra=extra, decode=True, rescore_corr=True)
    _flat_out("fwd_init.", o, res)
    run = Path("/home/external-user/ssd/yongjae_refiner/ck/ck2/train/ck2T10")
    if (run / "ckpt_last.pt").is_file():
        n2, cfg = load_ck(run, "last")
        meta["ck2T10_param_count"] = int(sum(p.numel() for p in n2.parameters()))
        meta["ck2T10_config_param_count"] = int(cfg.get("param_count", -1))
        with torch.no_grad():
            o2 = n2(bev, cand, status, extra=extra, decode=True, rescore_corr=True)
        _flat_out("fwd_ck2T10.", o2, res)
        T2.freeze_unused(n2)
        meta["ck2T10_trainable"] = int(sum(p.numel() for p in n2.parameters() if p.requires_grad))
    res["probe_ok_model"] = np.ones(1)


def probe_data(res: dict, meta: dict, n_tok: int = 64) -> None:
    from tools.ck.data.ck2_dataset import CK2Dataset
    from tools.ck.data.ck_dataset import CKDataset, collate_ck
    hashes = {}
    ds2 = CK2Dataset("navtrain_val", bev="none", n_var=32, gt=True, limit=n_tok)
    ds1 = CKDataset("navtrain_val", bev="T", k=16, labels="cand", gt=False, limit=n_tok)
    for name, ds, epochs in (("ck2", ds2, (0, 1)), ("r34", ds1, (0, 1))):
        for e in epochs:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(e)
            for b0 in range(0, len(ds), 8):
                b = collate_ck([ds[i] for i in range(b0, min(len(ds), b0 + 8))])
                for k in sorted(b):
                    v = b[k]
                    hashes[f"{name}.e{e}.b{b0 // 8}.{k}"] = _sha(v) if not isinstance(v, list) else \
                        hashlib.sha256("|".join(map(str, v)).encode()).hexdigest()[:32]
    meta["data_hashes"] = hashes
    meta["data_keys"] = {"ck2": sorted(ds2[0].keys()), "r34": sorted(ds1[0].keys())}
    meta["ck2_describe"] = json.loads(json.dumps(ds2.describe(), default=str))


# ------------------------------------------------------------------------------------------------ synthetic train
def _fake_datasets():
    """Small in-memory CK2 / r34 datasets (a copy of tests/test_ck2_train.py FakeCK2 / FakeR34 so both trees run the
    identical generator; surrogate GT of two real navtrain tokens)."""
    import numpy as np
    import torch
    from navsim.agents.para_ssr.ck import constants as Cn
    from navsim.agents.para_ssr.refiner.e2e import GTLoader

    toks = ("1aa44d46e4ab5bc7", "2570fbfdf1835706")
    gtc = {}

    def gt(tok):
        if tok not in gtc:
            d = GTLoader(Cn.DATA_ROOT).load(tok)
            gtc[tok] = {k: (v.numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in d.items()}
        return gtc[tok]

    def paths(n, v0, rng, lat=0.0):
        t = np.arange(1, 9) * 0.5
        out = np.zeros((n, 8, 3), np.float32)
        for j in range(n):
            v = v0 * rng.uniform(0.5, 1.3)
            kk = rng.normal(scale=0.02)
            x = v * t
            out[j, :, 0] = x
            out[j, :, 1] = lat + 0.5 * kk * x ** 2
            out[j, :, 2] = np.arctan(kk * x)
        return out

    def labels(rng, n):
        y = np.ones((n, 5), np.float32)
        f = rng.random(n) < 0.35
        key = rng.choice([0, 1, 3, 4], size=n)
        y[np.flatnonzero(f), key[f]] = 0.0
        y[:, 2] = rng.random(n).astype(np.float32)
        pd_ = y[:, 0] * y[:, 1] * (5 * y[:, 3] + 2 * y[:, 4] + 5 * y[:, 2]) / 12.0
        return y, pd_.astype(np.float32)

    class FakeCK2(torch.utils.data.Dataset):
        def __init__(self, n=8, K=6, V=4, use_gt=True, seed=0):
            self.n, self.K, self.n_var, self.use_gt, self.seed = n, K, V, use_gt, seed
            rng = np.random.default_rng(seed)
            self.bev = (rng.standard_normal((n, 256, 50, 100)) * 0.5).astype(np.float16)
            self.tokens = np.array([toks[i % 2] for i in range(n)])
            self.rows = np.arange(n)
            self.epoch = 0
            self.var_meta = {"names": ["id", "a-1.0", "a-0.5", "a+0.5", "l-0.5", "l+0.5"]}

        def set_epoch(self, e):
            self.epoch = int(e)

        def describe(self):
            return {"fake": True, "n": self.n}

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = int(self.rows[i])
            rng = np.random.default_rng([self.seed, r, self.epoch])
            K, V = self.K, self.n_var
            v0 = 4.0 + r
            cand = paths(K, v0, rng)
            y, pdm = labels(rng, K)
            vt = cand[:V].copy()
            vt[..., 1] += 0.3
            vy, vpd = labels(rng, V)
            st = np.zeros(8, np.float32)
            st[1], st[4] = 1.0, v0
            anc = (np.arange(K) + 10 * r).astype(np.int64)
            it = dict(token=f"{self.tokens[r]}_{r}", row=r, bev=self.bev[r], bev_ok=True, cand=cand, status=st,
                      gt_traj=paths(1, v0, np.random.default_rng(r))[0], y=y, y_pdms=pdm, y_ok=np.ones(K, bool),
                      anc_idx=anc, anc_group=np.array([0, 0, 1, 1, 2, 2] * K, np.int8)[:K], anc_pi=np.ones(K),
                      anc_w=np.ones(K, np.float32), var_traj=vt, var_y=vy, var_pdms=vpd, var_ok=np.ones(V, bool),
                      var_anchor=anc[:V].copy(), var_v=np.array([1, 2, 3, 4, 5] * V, np.int8)[:V],
                      var_col=np.arange(V, dtype=np.int64), var_pi=np.full(V, 0.5, np.float32))
            if self.use_gt:
                it.update(gt(str(self.tokens[r])))
            return it

    class FakeR34(torch.utils.data.Dataset):
        def __init__(self, n=4, k=5, seed=1):
            rng = np.random.default_rng(seed)
            self.n, self.k = n, k
            self.bev = (rng.standard_normal((n, 256, 50, 100)) * 0.5).astype(np.float16)
            self.cand = np.stack([paths(k, 5 + i, rng) for i in range(n)])
            lab = [labels(rng, k) for _ in range(n)]
            self.y = np.stack([x[0] for x in lab])
            self.pd = np.stack([x[1] for x in lab])
            self.v2_final = rng.standard_normal((n, k)).astype(np.float32)
            self.v2_im = rng.dirichlet(np.ones(k), n).astype(np.float32)

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            st = np.zeros(8, np.float32)
            st[1], st[4] = 1.0, 5.0 + i
            return dict(token=f"r34_{i}", row=i, bev=self.bev[i], bev_ok=True, cand=self.cand[i], status=st,
                        y=self.y[i], y_ok=np.ones(self.k, bool), y_pdms=self.pd[i], v2_final=self.v2_final[i],
                        v2_im=self.v2_im[i], v2_sim=self.y[i], gt_traj=self.cand[i, 0])

    return {"train": FakeCK2(n=8), "evals": [("navtrain_val", "r34", FakeR34(n=4)),
                                             ("navtrain_val_raw", "raw", FakeCK2(n=4, use_gt=False, seed=5))]}


NON_NUMERIC_OR_TIMING = ("cand_per_s", "sec", "lr_note")


def probe_train(res: dict, meta: dict, work: Path) -> None:
    import numpy as np
    import torch
    from tools.ck import train_ck2 as T2
    for steps in (1, 3):
        a = T2.get_parser().parse_args(["--arm", "T", "--run", f"off{steps}", "--device", "cpu", "--out-root",
                                        str(work), "--workers", "0", "--tokens-per-batch", "2", "--epochs", "2",
                                        "--log-every", "1", "--ckpt-every", "0", "--train-eval-rows", "0",
                                        "--max-steps", str(steps), "--restart"])
        T2.resolve_args(a)
        run = T2.train(a, datasets=_fake_datasets(), norm_override=(np.zeros(256, np.float32),
                                                                     np.ones(256, np.float32)))
        log = [json.loads(x) for x in (run / "train_log.jsonl").read_text().splitlines()]
        st = [r for r in log if r["kind"] == "step"]
        keys = sorted({k for r in st for k, v in r.items() if isinstance(v, (int, float)) and
                       k not in NON_NUMERIC_OR_TIMING and k != "gpu_mem_gb"})
        meta[f"train{steps}_step_keys"] = keys
        res[f"train{steps}.steps"] = np.array([[float(r.get(k, np.nan)) for k in keys] for r in st], np.float64)
        ck = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
        for n, t in ck["model"].items():
            res[f"train{steps}.sd.{n}"] = t.detach().numpy().copy()
        osd = ck["optim"]
        meta[f"train{steps}_optim_groups"] = len(osd["param_groups"])
        for pid, s in osd["state"].items():
            for k, v in s.items():
                if torch.is_tensor(v):
                    res[f"train{steps}.opt.{pid}.{k}"] = v.detach().numpy().copy()
        val = [json.loads(x) for x in (run / "val_metrics.jsonl").read_text().splitlines()]
        vk = sorted({(r["split"], k) for r in val for k, v in r.items() if isinstance(v, (int, float))
                     and not isinstance(v, bool)})
        meta[f"train{steps}_val_keys"] = [f"{s}:{k}" for s, k in vk]
        res[f"train{steps}.val"] = np.array([float(next(r[k] for r in val if r["split"] == s)) for s, k in vk],
                                            np.float64)
        cfg = json.loads((run / "config.json").read_text())
        meta[f"train{steps}_config_keys"] = sorted(cfg)
        meta[f"train{steps}_param_count"] = [cfg.get("param_count"), cfg.get("param_count_trainable")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True, help="npz path (a <out>.json with the non-array facts is written too)")
    ap.add_argument("--work", required=True, help="scratch dir for the synthetic training runs")
    ap.add_argument("--parts", default="model,data,train")
    ap.add_argument("--n-tok", type=int, default=64)
    a = ap.parse_args(argv)
    repo = _setup(a.repo)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    res, meta = {}, {"repo": repo, "torch": torch.__version__, "omp": os.environ.get("OMP_NUM_THREADS")}
    parts = a.parts.split(",")
    if "model" in parts:
        probe_model(res, meta)
    if "data" in parts:
        probe_data(res, meta, a.n_tok)
    if "train" in parts:
        work = Path(a.work)
        work.mkdir(parents=True, exist_ok=True)
        probe_train(res, meta, work)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **res)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({"out": str(out), "n_arrays": len(res), "meta_keys": sorted(meta)}))


if __name__ == "__main__":
    main()
