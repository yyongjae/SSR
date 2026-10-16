#!/usr/bin/env python
"""Diagnostic inference of CK2 / refiner-v4 teachers on a fully labelled candidate sample: navtrain_val = the
lead_remedy sample3000 (3,000 tokens x 96 = v2 r34 top-16 + 80 rule variants, every candidate officially labelled;
DET only, val has no ReSMap cache), navtest = the first --n-tok fully labelled tokens (reference A labels).  For the EP analysis of
the v4 teachers (why v4 EP is worse): per-candidate score logits, the v4 P_ref estimate, the v4 EP logit with the TRUE
P_ref substituted (oracle input, no retraining), and the EP head inputs for probes (first --n-feat tokens).

  cd /workspace/yongjae/SSR-v4 && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD OMP_NUM_THREADS=1 /venv/ssr/bin/python \
      tools/ck/v4/diag_val_teachers.py --teacher ck2T10 --split navtest --out <dir>
Pool layout c = k * 6 + v (select2), parents = the identity of rank k (v4 parent pass).  CK2: CKNet.forward over the 96
(+ hook on score_head: h [.., 384]).  v4: one score_set over the 96 with parents (+ hook on v4_readout.ep_head: its
input [.., 3 d + 3]).  fp16 autocast on cuda (as train_ck2 eval).  Outputs <out>/<split>/labels96.npz and <out>/<split>/<teacher>/: logit96.npy [N, 96, 5],
pref_logp.npy [N] (v4 with P_ref head), ep_logit_oracle_pref.npy [N, 96] (v4), feat.npy [n_feat, 96, F] fp16, meta.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tools.ck.e2e2 import eval_teacher_navtest_r34 as R  # noqa: E402

LR = Path("/home/external-user/ssd/yongjae_refiner/ck/ck2/analysis/lead_remedy")
SAMPLE = LR / "infer" / "sample3000" / "labels96.npz"
VARIANTS = LR / "infer" / "variants" / "traj80.npy"
NAVTEST_R34 = Path("/home/external-user/ssd/yongjae_refiner/ck/v4/eval/navtest_r34/r34_variants")


def sample(split: str, n_tok: int):
    """-> rows [N] (packed rows), labels96 [N, 96, 9], ok96 [N, 96], variants traj80 (memmap, all rows).
    navtrain_val: the lead_remedy sample3000; navtest: the first n_tok packed rows whose 96 labels are all ok
    (labels/navtest/cand + the r34 variant labels of eval_teacher_navtest_r34)."""
    if split == "navtrain_val":
        z = np.load(SAMPLE)
        return z["rows"].astype(np.int64), z["lab96"].astype(np.float32), z["ok96"] & z["valid96"], \
            np.load(VARIANTS, mmap_mode="r")
    pdir, tdf, pok = R.packed()
    n = len(tdf)
    labc, okc = R.load_lab(R.CK_DATA / "labels" / split / "cand", n, R.K16)
    labv, okv = R.load_lab(NAVTEST_R34 / "labels", n, R.K16 * R.NVAR)
    valid80 = np.load(NAVTEST_R34 / "valid80.npy")
    good = pok & okc.all(1) & (okv | ~valid80).all(1)
    rows = np.flatnonzero(good)[:n_tok]
    lab = R.to_pool(labc[rows], labv[rows]).astype(np.float32)
    ok = R.to_pool(okc[rows], okv[rows] & valid80[rows])
    return rows, lab, ok, np.load(NAVTEST_R34 / "traj80.npy", mmap_mode="r")


def main(argv=None):
    from torch.utils.data import DataLoader

    from navsim.agents.para_ssr.ck import constants as Cn
    from navsim.agents.para_ssr.ck.model import load_ck
    from navsim.agents.para_ssr.refiner.e2e import ego_inputs
    from tools.ck.data.ck_dataset import CKDataset, collate_ck
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="navtrain_val", choices=("navtrain_val", "navtest"))
    ap.add_argument("--n-tok", type=int, default=3000, help="navtest: first N fully labelled tokens")
    ap.add_argument("--n-feat", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = R.resolve_run(a.teacher)
    net, cfg = load_ck(run, "last", str(dev))
    net.eval()
    arm = cfg["arm"]
    v4 = getattr(net, "v4_readout", None) is not None
    rows, lab96, ok96, var_all = sample(a.split, a.n_tok)
    N = len(rows)
    pref_true = np.log1p(np.clip(lab96[:, 0, Cn.LBL["pdm_progress_eff"]].astype(np.float64), 0, None))
    out = Path(a.out) / a.split / Path(run).name
    out.mkdir(parents=True, exist_ok=True)
    lab_file = Path(a.out) / a.split / "labels96.npz"
    if not lab_file.is_file():
        np.savez(lab_file, rows=rows, lab96=lab96, ok96=ok96)
    logit = np.full((N, 96, 5), np.nan, np.float32)
    pref = np.full(N, np.nan, np.float32)
    ep_or = np.full((N, 96), np.nan, np.float32)
    feat = None
    cap = {}
    mod = net.v4_readout.ep_head if v4 else net.score_head
    hook = mod.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach()))
    pos = {int(r): i for i, r in enumerate(rows)}
    ds = CKDataset(a.split, bev=arm, k=R.K16, labels=None, gt=False, rows=rows)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, collate_fn=collate_ck, num_workers=a.workers)
    t0 = time.time()
    with torch.no_grad():
        for b in dl:
            prow = b["rows"].numpy().astype(np.int64)
            ii = np.array([pos[int(r)] for r in prow])
            bev = b["bev"].to(dev)
            c16 = b["cand"].to(dev).float()
            st = b["status"].to(dev).float()
            var = torch.from_numpy(np.asarray(var_all[prow], np.float32)).to(dev)
            T = c16.shape[0]
            pool = torch.from_numpy(R.to_pool(c16.cpu().numpy(), var.cpu().numpy())).to(dev)
            pool = torch.nan_to_num(pool)
            par = c16.repeat_interleave(R.NV, 1)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                if v4:
                    ego = ego_inputs(st)
                    f, mem = net.trunk.scene(bev, T)
                    pl = net.scene_pref(mem, ego)
                    o = net.score_set(f, mem, ego, pool, par, pref=pl)
                    x = cap["x"]
                    pt = torch.from_numpy(pref_true[ii]).to(dev).float()
                    oo = net.score_set(f, mem, ego, pool, par, pref=pt)
                    ep_or[ii] = oo["score_logit"][..., Cn.CK_KEYS.index("ep")].float().cpu().numpy()
                    if pl is not None:
                        pref[ii] = pl.float().cpu().numpy()
                else:
                    o = net(bev, pool, st, decode=False)
                    x = cap["x"]
            logit[ii] = o["score_logit"].float().cpu().numpy()
            sel = ii < a.n_feat
            if sel.any():
                xf = x.reshape(T, 96, -1).float().cpu().numpy().astype(np.float16)
                if feat is None:
                    feat = np.zeros((min(a.n_feat, N), 96, xf.shape[-1]), np.float16)
                feat[ii[sel]] = xf[sel]
    hook.remove()
    np.save(out / "logit96.npy", logit)
    np.save(out / "feat.npy", feat)
    if v4:
        np.save(out / "pref_logp.npy", pref)
        np.save(out / "ep_logit_oracle_pref.npy", ep_or)
    meta = dict(teacher=str(run), arm=arm, v4=v4, split=a.split, n=N, n_feat=int(min(a.n_feat, N)),
                feat_dim=int(feat.shape[-1]), sec=round(time.time() - t0, 1), device=str(dev))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main()
