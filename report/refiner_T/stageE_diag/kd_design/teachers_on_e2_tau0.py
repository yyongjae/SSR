#!/usr/bin/env python
"""H5/H6 diagnostic: frozen run-4 teacher refiners (the exact E2 KD teachers, ckpt_best, sha-identical to
stageE/teachers/*) applied to E2's own navtest tau0 drafts, one draft per token.  Same inference path as
tools/refiner/refine_external_drafts.py predict, plus: --fp32 (no autocast; the E2 KD targets were computed in fp32,
trainer precision 32) and the ego inputs fed (v0, a0, eds, cmd) are stored so they can be compared with the student's.
Writes <out>/pred.npz, refined.npz, tokens.parquet, predict_meta.json (same layout as refine_external_drafts)."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path("/home/external-user/yongjae/SSR")
sys.path[:0] = [str(REPO), str(REPO / "tools/refiner")]
import refine_external_drafts as X  # noqa
from navsim.agents.para_ssr.refiner import data as RD  # noqa
from navsim.agents.para_ssr.refiner.decoder import lon_live  # noqa
TR = X.TR

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--drafts-pkl", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    dev = TR._device(a.gpu)
    net, cfg = TR.load_run_model(Path(a.run), "best", dev)
    drafts = X.load_drafts_pkl(a.drafts_pkl)
    packed = RD.PackedSplit("navtest", str(RD.DATA_ROOT / "packed"))
    rows = X.select_rows(packed, drafts)
    teacher = X.teacher_for(cfg, "navtest", rows, packed)
    ds = X.ExternalDraftDataset(packed, rows, teacher, drafts)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=RD.collate_tokens, num_workers=2, prefetch_factor=4)
    use_amp = (not a.fp32) and bool(cfg.get("amp", 1)) and dev.type == "cuda"
    keys = ("rows", "tau0", "tau1", "p_g", "z_lon", "w_lat", "c_lon", "e_lat", "alpha", "beta", "lon_live",
            "v0", "a0", "eds", "cmd", "lat_on")
    rec = {k: [] for k in keys}; toks = []; t0 = time.time()
    for batch in loader:
        batch = RD.batch_to(batch, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            o = net(batch["bev"], batch["tau0"], batch["v0"], batch["a0"], batch["eds"], batch["cmd"])
        dec = TR.decode_batch(o, batch, "A")
        T, K = batch["tau0"].shape[:2]
        f = lambda x: x.float().cpu().numpy() if x.is_floating_point() else x.cpu().numpy()
        rec["rows"].append(batch["rows"].cpu().numpy()); rec["tau0"].append(f(batch["tau0"]))
        rec["tau1"].append(f(dec["traj"].reshape(T, K, 8, 3)))
        rec["p_g"].append(f(torch.sigmoid(o["gate_logit"].float())))
        rec["z_lon"].append(f(o["z_lon"])); rec["w_lat"].append(f(o["w_lat"]))
        rec["c_lon"].append(f(dec["c_lon"].reshape(T, K, -1))); rec["e_lat"].append(f(dec["e_lat"].reshape(T, K, -1)))
        rec["alpha"].append(f(dec["flags"]["alpha"].reshape(T, K))); rec["beta"].append(f(dec["flags"]["beta"].reshape(T, K)))
        rec["lat_on"].append(f(dec["flags"]["lat_on"].reshape(T, K)))
        rec["lon_live"].append(lon_live(o["z_lon"].float()).cpu().numpy())
        for k in ("v0", "a0", "eds", "cmd"):
            rec[k].append(f(batch[k]))
        toks += batch["tokens"]
    R = {k: np.concatenate(v) for k, v in rec.items()}
    tokens = np.array(toks)
    assert np.array_equal(R["rows"], rows)
    for i, t in enumerate(tokens):
        assert np.array_equal(R["tau0"][i, 0], drafts[t]), t
    meta = dict(run=a.run, ckpt="best", arm=cfg["arm"], mode="A", amp=use_amp, split="navtest", drafts_pkl=a.drafts_pkl,
                drafts_pkl_sha256=X._sha(a.drafts_pkl), n_tokens=int(len(tokens)), theta=0.0,
                teacher=str(getattr(teacher, "root", None)), sec=round(time.time() - t0, 1),
                tau1_eq_tau0_frac=float((R["tau1"] == R["tau0"]).all((2, 3)).mean()),
                created=time.strftime("%Y-%m-%dT%H:%M:%S"), gpu=a.gpu)
    X._write_common(Path(a.out), tokens, R.pop("rows"), packed, R.pop("tau0"), R.pop("tau1"), R, meta)
    print(json.dumps(meta), flush=True)

if __name__ == "__main__":
    main()
