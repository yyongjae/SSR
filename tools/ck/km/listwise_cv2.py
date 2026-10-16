#!/usr/bin/env python
"""km: EPDMS-aware, temporally consistent listwise selector (navtrain_val K-fold by log, like listwise_cv.py).

listwise_cv.py maximises official PDMS only: navtest PDMS +1.70 but one-stage EPDMS -1.65 (two-frame extended comfort
78.9 -> 67.9, lane keeping 96.2 -> 94.1, DDC 99.4 -> 98.0): it hops between variants frame to frame.  Here
  target   = w * PDMS (v1 labels) + (1 - w) * EPDMS (label_v2.py: official per-candidate v2 sub-scores; ec_v2.py: official
             two-frame EC against the PREVIOUS frame's selection)
  features = listwise.pool_features + [EC(candidate, previous frame's selected trajectory), has previous frame]
             (EC uses only the LQR-simulated ego states of the two trajectories -- no future information)
  decoding = causal: frames of a log in time order, each frame conditioned on the model's own previous selection.
  training = each epoch first decodes the training folds with the current model (previous selections), then one
             listwise pass with those fixed.
Reported per config: out-of-fold PDMS and EPDMS (exact offline aggregation, epdms_offline.py) vs CK2's sequence.
--la pol (stage 2, lookahead): frame t's choice also sets the EC of the NEXT frame t+1, so the target adds
  gamma (1 - w) mult[t+1, j] 2 ec[t+1, j, i] / 16   with j = the current policy's choice at t+1 (the epoch's decode),
  i.e. the change of the whole sequence's EPDMS when only frame t's choice moves (neighbours fixed).
--la max: the same with j re-chosen by an oracle for each i (max over allowed j of the t+1 target given prev i).
Labels only enter the target; decoding / features stay causal.
--traj-feat (stage 3): + the candidate's own submitted trajectory (8 poses: x/10, y/3, sin, cos of heading = 32 dims);
  the CK hidden state h encodes it only implicitly, the v2-only terms (LK, DDC, EC) are geometric.
Pre-declared choice = best mean of (OOF dPDMS, OOF dEPDMS).  --test-only: the K fold heads (mean delta) decode --test once.
"""
from __future__ import annotations

import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[3])
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck import listwise as LW  # noqa: E402
from navsim.agents.para_ssr.ck.constants import LBL  # noqa: E402
from tools.ck.km.epdms_offline import W as W2, V2Labels  # noqa: E402
from tools.ck.km.listwise_cv import fold_of_rows  # noqa: E402
from tools.ck.km.listwise_train import P_PDMS, Split, log  # noqa: E402

D_EXTRA = 2
W_EC = W2["ec"]
D_TRAJ = 32


class Data:
    def __init__(self, root: Path, run: str, split: str, labv2: Path, no_ec: bool = False, traj_feat: bool = False):
        self.no_ec = no_ec
        self.traj = np.load(root / "infer" / run / split / "lat_traj.npy", mmap_mode="r") if traj_feat else None
        self.S = Split(root, run, split, need_pool=True)
        self.V = V2Labels(labv2 / split)
        assert (self.V.meta.token.to_numpy() == self.S.tokens.token.astype(str).to_numpy()).all()
        S = self.S
        v2ok = self.V.meta.done.to_numpy() & (self.V.meta.err.to_numpy() == 0)
        if (~v2ok[S.rows]).any():
            log(f"{split}: {(~v2ok[S.rows]).sum()} rows without v2 labels (no v2 metric cache) -> excluded")
            S.rows = S.rows[v2ok[S.rows]]
        self.N = len(S.tokens)
        self.base_sel = np.zeros(self.N, np.int64)
        self.base_sel[S.rows] = LW.select(S.s0[S.rows], S.allowed[S.rows])          # CK2 (beta .5, all, lat on)
        inset = np.zeros(self.N, bool)
        inset[S.rows] = True
        self.inset = inset
        st = self.V.meta.start_time.to_numpy()
        self.order = S.rows[np.argsort(st[S.rows], kind="stable")]
        p = self.prev_of(S.rows)
        assert np.bincount(p[p >= 0]).max(initial=0) <= 1, "a frame is the previous frame of two frames"
        self.next = np.full(self.N, -1, np.int64)
        self.next[p[p >= 0]] = S.rows[p >= 0]
        bad = ~np.isfinite(np.asarray(self.V.lab[S.rows][..., 0])) & S.allowed[S.rows]
        log(f"{split}: v2 labels missing on {bad.sum()} allowed columns of {len(S.rows)} rows; "
            f"{(self.V.prev[S.rows] >= 0).sum()} rows have a previous frame")

    def prev_of(self, rows):
        p = self.V.prev[rows]
        return np.where((p >= 0) & self.inset[np.maximum(p, 0)], p, -1)

    def batch(self, rows, sel_full, dev):
        b = self.S.batch(rows, dev)       # sorts rows
        rows = b["rows"]
        p = self.prev_of(rows)
        sp = np.where(p >= 0, sel_full[np.maximum(p, 0)], -1)
        ec = np.zeros((len(rows), 96), np.float32)
        m = sp >= 0
        if m.any():
            e = np.asarray(self.V.ec[rows[m], :, sp[m]], np.float32)
            ec[m] = np.where(e == 255, 0.0, e)
        extra = np.stack([ec, np.broadcast_to(m[:, None], ec.shape).astype(np.float32)], -1)
        if self.no_ec:
            extra[:] = 0.0
        if self.traj is not None:
            tj = np.asarray(self.traj[rows], np.float32)                                  # [R, 96, 8, 3]
            tf = np.stack([tj[..., 0] / 10.0, tj[..., 1] / 3.0, np.sin(tj[..., 2]), np.cos(tj[..., 2])], -1)
            extra = np.concatenate([extra, tf.reshape(len(rows), 96, D_TRAJ)], -1)
        b["x"] = torch.cat([b["x"], torch.as_tensor(extra, device=dev)], -1)
        b["sel_prev"] = sp
        return b

    def target(self, rows, sel_prev, w, sel_full=None, la="none", gamma=1.0):
        e = self.V.epdms_given_prev(rows, sel_prev)
        p = self.S.lab_lat[rows][..., P_PDMS]
        t = np.nan_to_num(w * p + (1 - w) * e, nan=0.0)
        if la == "none":
            return t
        nx = self.next[rows]
        m = nx >= 0
        if not m.any():
            return t
        V, n = self.V, nx[m]
        if la == "pol":
            j = sel_full[n]
            ec = np.asarray(V.ec[n, j, :], np.float64)                                   # [R', 96 (prev i)]
            add = np.nan_to_num(V.mult[n, j])[:, None] * W_EC * np.where(ec == 255, 0.0, ec) / 16.0
            t[m] += gamma * (1 - w) * add
        elif la == "max":
            ec = np.asarray(V.ec[n], np.float64)                                           # [R', 96 j, 96 i]
            b, mu = V.base[n][:, :, None], V.mult[n][:, :, None]
            ep = np.where(ec == 255, mu * b / 14.0, mu * (b + W_EC * ec) / 16.0)
            q = np.nan_to_num(w * self.S.lab_lat[n][..., P_PDMS][:, :, None] + (1 - w) * ep, nan=0.0)
            q = np.where(self.S.allowed[n][:, :, None], q, -np.inf).max(1)
            t[m] += gamma * np.where(np.isfinite(q), q, 0.0)
        else:
            raise ValueError(la)
        return t


@torch.no_grad()
def decode(heads, D: Data, rows, dev) -> np.ndarray:
    """causal decoding of rows (time order within each log), sel_full[N] (other rows: CK2's choice)."""
    sel = D.base_sel.copy()
    rs = set(rows.tolist())
    depth = {}
    for r in D.order:
        if r in rs:
            q = D.prev_of(np.array([r]))[0]
            depth[r] = depth[q] + 1 if (q >= 0 and q in depth) else 0
    by = {}
    for r, d in depth.items():
        by.setdefault(d, []).append(r)
    for d in sorted(by):
        rr = np.array(sorted(by[d]))
        for i in range(0, len(rr), 1024):
            b = D.batch(rr[i:i + 1024], sel, dev)
            delta = np.mean([h.eval()(b["x"], b["allowed"]).double().cpu().numpy() for h in heads], 0)
            sel[b["rows"]] = LW.select(D.S.s0[b["rows"]] + delta, D.S.allowed[b["rows"]])
    return sel


def score(D: Data, rows, sel_full, n_boot=1000, seed=0) -> dict:
    S = D.S
    ar = np.arange(len(rows))
    lab = S.lab_lat[rows]
    idx, base = sel_full[rows], D.base_sel[rows]
    p, p0 = lab[ar, idx, P_PDMS], lab[ar, base, P_PDMS]
    e, e0 = D.V.sequence_epdms(sel_full, rows), D.V.sequence_epdms(D.base_sel, rows)
    logs = np.asarray(S.tokens["log"].astype(str))[rows]
    u, inv = np.unique(logs, return_inverse=True)
    rng = np.random.default_rng(seed)
    ci = {}
    for k, (a, b) in {"pdms": (p, p0), "epdms": (e, e0)}.items():
        ok = np.isfinite(a) & np.isfinite(b)
        ds, ns = np.bincount(inv[ok], (a - b)[ok], len(u)), np.bincount(inv[ok], minlength=len(u))
        bs = [ds[s].sum() / ns[s].sum() for s in (rng.integers(0, len(u), len(u)) for _ in range(n_boot))]
        ci[k] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out = {"n": int(len(rows)), "pdms": float(p.mean()), "pdms_ck2": float(p0.mean()), "epdms": float(np.nanmean(e)),
           "epdms_ck2": float(np.nanmean(e0)), "frac_changed": float(np.mean(idx != base)), "ci": ci}
    out["d_pdms"], out["d_epdms"] = out["pdms"] - out["pdms_ck2"], out["epdms"] - out["epdms_ck2"]
    out["crit"] = 0.5 * (out["d_pdms"] + out["d_epdms"])
    V, L2 = D.V, {"lk": 6, "hc": 7, "ddc": 2, "tlc": 3, "ep": 4}
    for k, j in L2.items():
        out[k + "_v2"] = float(np.nanmean(np.asarray(V.lab[rows, idx, j], np.float64)))
        out[k + "_v2_ck2"] = float(np.nanmean(np.asarray(V.lab[rows, base, j], np.float64)))
    p_ = D.prev_of(rows)
    m = p_ >= 0
    out["ec"] = float(np.nanmean(V.ec_of(rows[m], idx[m], sel_full[p_[m]])))
    out["ec_ck2"] = float(np.nanmean(V.ec_of(rows[m], base[m], D.base_sel[p_[m]])))
    for k in ("nc", "ttc"):
        out[k] = float(lab[ar, idx, LBL[k]].mean())
    return out


def fmt(ev):
    return (f"PDMS {100 * ev['pdms']:.2f} (CK2 {100 * ev['pdms_ck2']:.2f}, {100 * ev['d_pdms']:+.2f} "
            f"[{100 * ev['ci']['pdms'][0]:+.2f} {100 * ev['ci']['pdms'][1]:+.2f}]) EPDMS {100 * ev['epdms']:.2f} "
            f"(CK2 {100 * ev['epdms_ck2']:.2f}, {100 * ev['d_epdms']:+.2f} [{100 * ev['ci']['epdms'][0]:+.2f} "
            f"{100 * ev['ci']['epdms'][1]:+.2f}]) EC {100 * ev['ec']:.1f}/{100 * ev['ec_ck2']:.1f} LK "
            f"{100 * ev['lk_v2']:.1f}/{100 * ev['lk_v2_ck2']:.1f} DDC {100 * ev['ddc_v2']:.1f}/{100 * ev['ddc_v2_ck2']:.1f} "
            f"changed {ev['frac_changed']:.3f}")


def make_head(a, dev):
    return LW.ListwiseHead(d_in=LW.D_IN + D_EXTRA + (D_TRAJ if getattr(a, "traj_feat", False) else 0), d=a.d, n_layers=a.layers, dropout=a.dropout,
                           cross=bool(a.cross)).to(dev)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--labv2", required=True, help="dir with <split>/{lab,ec,prev_row,meta}")
    ap.add_argument("--val", default="navtrain_val")
    ap.add_argument("--test", default="navtest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=float, default=0.0, help="target = w PDMS + (1-w) EPDMS")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--cross", type=int, default=1)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--la", default="none", choices=("none", "pol", "max"), help="lookahead target (stage 2)")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--traj-feat", action="store_true", help="stage 3: + the candidate trajectory as features")
    ap.add_argument("--no-ec-feat", action="store_true", help="ablation: zero the previous-selection EC features")
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--check-ck2", action="store_true", help="only score CK2's sequence on --val (offline EPDMS)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root, lv = Path(a.root), Path(a.labv2)
    if a.check_ck2:
        D = Data(root, a.run, a.val, lv)
        ev = score(D, D.S.rows, D.base_sel)
        log(f"CK2 {a.val}: {fmt(ev)}")
        return 0
    if a.test_only:
        res = json.loads((out / "result.json").read_text())
        t = argparse.Namespace(**res["args"])
        heads = []
        for f in range(t.k):
            h = make_head(t, dev)
            h.load_state_dict(torch.load(out / f"fold{f}.pt", map_location=dev))
            heads.append(h)
        D = Data(root, a.run, a.test, lv, no_ec=bool(res["args"].get("no_ec_feat")),
                 traj_feat=bool(res["args"].get("traj_feat")))
        sel = decode(heads, D, D.S.rows, dev)
        ev = score(D, D.S.rows, sel)
        np.save(out / f"idx_{a.test}.npy", sel[D.S.rows])
        np.save(out / f"rows_{a.test}.npy", D.S.rows)
        res["test"] = ev
        (out / "result.json").write_text(json.dumps(res, indent=1))
        log(f"TEST {a.test} ({t.k} fold heads): {fmt(ev)}")
        return 0
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    D = Data(root, a.run, a.val, lv, no_ec=a.no_ec_feat, traj_feat=a.traj_feat)
    S = D.S
    fold = fold_of_rows(S, a.k, a.seed)
    oof = D.base_sel.copy()
    folds = []
    for f in range(a.k):
        tr_rows, ho_rows = S.rows[fold != f], S.rows[fold == f]
        head = make_head(a, dev)
        opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
        steps = a.epochs * (len(tr_rows) // a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
        t0 = time.time()
        for ep in range(a.epochs):
            sel_tr = decode([head], D, tr_rows, dev)
            head.train()
            perm = rng.permutation(tr_rows)
            acc = []
            for i in range(0, len(perm) - a.bs + 1, a.bs):
                b = D.batch(perm[i:i + a.bs], sel_tr, dev)
                tgt = torch.as_tensor(D.target(b["rows"], b["sel_prev"], a.w, sel_tr, a.la, a.gamma), dtype=torch.float32, device=dev)
                L = LW.listwise_loss(b["s0"].float() + head(b["x"], b["allowed"]), tgt, b["allowed"], a.tau)
                opt.zero_grad(set_to_none=True)
                L["loss"].backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
                sched.step()
                acc.append(float(L["exp_pdms"]))
        torch.save(head.state_dict(), out / f"fold{f}.pt")
        sel = decode([head], D, ho_rows, dev)
        oof[ho_rows] = sel[ho_rows]
        ev = score(D, ho_rows, sel)
        folds.append(ev | {"fold": f, "train_target_last": float(np.mean(acc))})
        log(f"fold {f}: n {ev['n']} {fmt(ev)} | {time.time() - t0:.0f}s")
    ev = score(D, S.rows, oof)
    np.save(out / f"oof_sel_{a.val}.npy", oof)
    res = {"args": vars(a), "oof": ev, "folds": folds, "crit": ev["crit"]}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    log(f"OOF {a.val}: {fmt(ev)} crit {100 * ev['crit']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
