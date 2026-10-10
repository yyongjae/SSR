"""CK2 e2e per-step candidate assembly (SPEC ck2e2e §2-1, §2-2, §4-3 fallback).  New file; data side of CKE2E2.

  v0_from_status(status)              ego speed hypot(vx, vy) in float64 (= tools/ck/data/label_variants v0)
  online_variants(cand16, v0, vcfg)   the 96 = 16 x 6 rule variants of the student's top-16 (variants.make_variants,
                                      combine 'separate', s_on_frac 0.2, float64 decode, ext 'straight'), no_grad,
                                      device of cand16; traj96[:, ::6] == cand16 bitwise (finite rows)
  warmup_cands(targets, tokens, epoch, sampler)
                                      epochs 0-4: 32 sampled raw anchors + 32 variants of the fixed variant label files,
                                      official labels (raw256 / variant labels.npy), masks, groups (§2-1)
  onpolicy_cands(top_cand, v0, tokens, epoch, seed)
                                      epochs >= 5: current top-16 + 32 of their 80 online variants (stream
                                      'ck2e2e.now'); also returns the full 96 for the recorder (§2-2)
  fill_fallback(G, targets, tokens, epoch, sampler)
                                      LabelStore2 rows without a generation -> warm-up-style official-label set
                                      (16 near+mid anchors of this epoch's draw + 32 variants, stream 'ck2e2e.lab'),
                                      G_src 0 (3 when the token has no warm-up data) (NU12)
  onpolicy_label_set(store, targets, rows, tokens, epoch, seed, sampler, n_var, device)
                                      = LabelStore2.lookup + fill_fallback + device move (the BCE set G of §2-2)

All candidate trajectories returned here are detached inputs (no grad), finite (non-finite -> 0 with ok False).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from . import e2e_data2 as D2
from .e2e_data2 import (G_K, GROUP_VAR, K16, N_NEAR_MID, N_WU_ANCHORS, NV, SRC_FALLBACK, SRC_NODATA, STREAM_LAB,
                        STREAM_NOW, VNAMES)

VARIANTS_DEFAULT: Dict[str, Any] = {"speeds": [-1.0, -0.5, 0.5], "lats": [-0.5, 0.5], "combine": "separate",
                                    "s_on_frac": 0.2, "ext": "straight", "compute_dtype": "float64"}
_DTYPES = {"float64": torch.float64, "float32": torch.float32, "torch.float64": torch.float64,
           "torch.float32": torch.float32}


def variants_cfg(vcfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """VARIANTS_DEFAULT merged with vcfg; validated: the table must be the 6-column 'separate' table in VNAMES order
    (the generation / label-file layout)."""
    from .variants import EXT_MODES, variant_name, variant_table

    c = dict(VARIANTS_DEFAULT)
    if vcfg:
        bad = sorted(set(vcfg) - set(VARIANTS_DEFAULT))
        if bad:
            raise ValueError(f"variants: unknown key(s) {bad}")
        c.update({k: v for k, v in dict(vcfg).items() if v is not None})
    c["speeds"] = [float(a) for a in c["speeds"]]
    c["lats"] = [float(d) for d in c["lats"]]
    c["s_on_frac"] = float(c["s_on_frac"])
    c["combine"], c["ext"], c["compute_dtype"] = str(c["combine"]), str(c["ext"]), str(c["compute_dtype"])
    if c["ext"] not in EXT_MODES or c["ext"].startswith("centerline"):
        raise ValueError(f"variants.ext must be const_curv | straight (centerline needs per-token lines), got {c['ext']!r}")
    if c["compute_dtype"] not in _DTYPES:
        raise ValueError(f"variants.compute_dtype must be float64 | float32, got {c['compute_dtype']!r}")
    names = tuple(variant_name(a, d) for a, d in variant_table(c["speeds"], c["lats"], c["combine"]))
    if names != VNAMES:
        raise ValueError(f"variant table {names} != {VNAMES} (generation layout c = k * 6 + v)")
    return c


def check_variant_files(var_dir, vcfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The online variant definition must equal the warm-up variant label files' build (speeds, lats, combine,
    s_on_frac, names, accel extension): the student sees the same family in warm-up and on-policy.  Raises
    ValueError on a mismatch; returns the compared values."""
    import json
    from pathlib import Path

    c = variants_cfg(vcfg)
    b = json.loads((Path(var_dir) / "build.json").read_text())
    f_ext = b.get("ext") or (b.get("merged") or {}).get("ext_a+0.5") or "const_curv"
    got = {"speeds": [float(x) for x in b.get("speeds", [])], "lats": [float(x) for x in b.get("lats", [])],
           "combine": b.get("combine"), "s_on_frac": float(b.get("s_on_frac", -1)), "ext": f_ext,
           "names": list(b.get("names", []))}
    want = {"speeds": c["speeds"], "lats": c["lats"], "combine": c["combine"], "s_on_frac": c["s_on_frac"],
            "ext": c["ext"], "names": list(VNAMES)}
    diff = {k: (want[k], got[k]) for k in want if want[k] != got[k]}
    if diff:
        raise ValueError(f"online variants != variant label files {var_dir}: {diff}")
    return got


def v0_from_status(status: torch.Tensor) -> torch.Tensor:
    """status_feature [B, 8] -> v0 = hypot(vx, vy) float64 [B] (label_variants: hypot(status[4], status[5]) in
    float64; NaN is mapped to 0 inside make_variants)."""
    s = status.detach()
    return torch.hypot(s[:, 4].double(), s[:, 5].double())


@torch.no_grad()
def online_variants(cand16: torch.Tensor, v0, vcfg: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
    """cand16 [B, K, 8, 3] (K = 16 in training) -> dict (device of cand16):
      traj96   f32  [B, K*6, 8, 3]  column c = k * 6 + v (k-major, VNAMES order); traj96[:, ::6] == cand16 bitwise
      valid96  bool [B, K*6]        False = lateral variant of a candidate with S_8 < 3 m (duplicate of the parent)
                                    or a non-finite parent (whose columns are NaN)
      vtype    int64 [K*6] (c % 6), parent int64 [K*6] (c // 6), dev_xy96 f32 [B, K*6], fin16 bool [B, K], S8 f32 [B, K]
    v0 [B] ego speed (any float dtype; float64 = label_variants exactly: v0_from_status)."""
    from .variants import make_variants

    c = variants_cfg(vcfg)
    x = cand16.detach().float()
    if x.dim() != 4 or tuple(x.shape[2:]) != (8, 3):
        raise ValueError(f"cand16 must be [B, K, 8, 3], got {tuple(x.shape)}")
    B, K = x.shape[:2]
    fin = torch.isfinite(x).flatten(2).all(-1)
    xin = torch.where(fin[..., None, None], x, torch.zeros_like(x))
    v = torch.as_tensor(v0, device=x.device).detach().reshape(B)
    out = make_variants(xin, v, c["speeds"], c["lats"], c["combine"], c["s_on_frac"],
                        compute_dtype=_DTYPES[c["compute_dtype"]], ext=c["ext"])
    V = out["traj"].shape[2]
    traj = out["traj"].reshape(B, K * V, 8, 3)
    valid = out["valid"].reshape(B, K * V).clone()
    if not bool(fin.all()):
        bad = (~fin).repeat_interleave(V, 1)
        traj = torch.where(bad[..., None, None], torch.full_like(traj, float("nan")), traj)
        valid &= ~bad
    ar = torch.arange(K * V, device=x.device)
    return {"traj96": traj, "valid96": valid, "vtype": ar % V, "parent": ar // V,
            "dev_xy96": out["meta"]["dev_xy"].reshape(B, K * V).float(), "fin16": fin,
            "S8": out["meta"]["S8"].float()}


def _gather_k(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B, N, *s], idx long [B, M] -> [B, M, *s]."""
    sh = x.shape[2:]
    return x.gather(1, idx.reshape(idx.shape + (1,) * len(sh)).expand(idx.shape + sh))


def _wu_arrays_np(targets: Dict[str, torch.Tensor], sel=None) -> Dict[str, np.ndarray]:
    out = {}
    for k in ("ck2_wu_ok", "ck2_gt", "ck2_raw_y", "ck2_var_valid"):
        a = targets[k].detach().cpu().numpy()
        out[k] = a if sel is None else a[sel]
    return out


def warmup_cands(targets: Dict[str, torch.Tensor], tokens: Sequence[str], epoch: int, sampler: "D2.WarmupSampler",
                 device=None) -> Dict[str, Any]:
    """Warm-up candidate set (SPEC §2-1), B tokens, 64 = 32 anchors (slot order of the AnchorSampler draw) + n_var
    variants (sampler.n_var, default 32):
      cand f32 [B,64,8,3], y f32 [B,64,5] (official), ok bool [B,64] (label ok & valid & not padding & ck2_wu_ok),
      group int64 [B,64] (0 near / 1 mid / 2 strat / 3 variant), vtype int64 [B,64] (0 anchor, 1..5 variant type),
      parent_anchor int64 [B,64] (anchor id; for a variant its parent anchor), anc_idx [B,32], var_cols [B,n],
      var_pad_ok [B,n], sur_pos int64 [B,16] (slot positions of the 8 near + 8 mid anchors, slot order),
      lat_mask bool [B,64] (near / mid anchors and variants, NU16), teacher_ok bool [B,64] (finite & ck2_wu_ok &
      (anchor | valid & not padding), NU25 / NU44), wu_ok bool [B], n_valid int64 [B]."""
    dev = torch.device(device) if device is not None else targets["ck2_raw_y"].device
    B = len(tokens)
    S = sampler.sample(tokens, _wu_arrays_np(targets), epoch)
    ai = torch.from_numpy(S["anc_idx"]).to(dev)
    vc = torch.from_numpy(S["var_cols"]).to(dev)
    pad = torch.from_numpy(S["var_pad_ok"]).to(dev)
    wu = targets["ck2_wu_ok"].to(dev).reshape(B).bool() & torch.from_numpy(S["ok"]).to(dev)
    anc = sampler.anchors_t(dev)
    cand_a = anc[ai]                                                            # [B, 32, 8, 3]
    cand_v = _gather_k(targets["ck2_var_traj"].to(dev).float(), vc)             # [B, n, 8, 3]
    y = torch.cat([_gather_k(targets["ck2_raw_y"].to(dev).float(), ai),
                   _gather_k(targets["ck2_var_y"].to(dev).float(), vc)], 1)
    vvalid = targets["ck2_var_valid"].to(dev).bool()
    ok_a = targets["ck2_raw_ok"].to(dev).bool().gather(1, ai)
    val_v = vvalid.gather(1, vc) & pad
    ok_v = targets["ck2_var_ok"].to(dev).bool().gather(1, vc) & val_v
    cand = torch.cat([cand_a, cand_v], 1)
    fin = torch.isfinite(cand).flatten(2).all(-1)
    cand = torch.where(fin[..., None, None], cand, torch.zeros_like(cand))
    ok = torch.cat([ok_a, ok_v], 1) & fin & wu[:, None]
    n = vc.shape[1]
    grp_a = torch.from_numpy(S["group"].astype(np.int64)).to(dev)
    group = torch.cat([grp_a, torch.full((B, n), GROUP_VAR, dtype=torch.long, device=dev)], 1)
    vtype = torch.cat([torch.zeros((B, N_WU_ANCHORS), dtype=torch.long, device=dev), vc % NV], 1)
    par = torch.cat([ai, targets["ck2_var_anchor"].to(dev).long().gather(1, vc // NV)], 1)
    sur_pos = torch.sort((grp_a > 1).long(), dim=1, stable=True).indices[:, :N_NEAR_MID]
    lat_mask = (group <= 1) | (group == GROUP_VAR)
    teacher_ok = fin & wu[:, None] & torch.cat([torch.ones_like(ok_a), val_v], 1)
    return {"cand": cand, "y": y, "ok": ok, "group": group, "vtype": vtype, "parent_anchor": par, "anc_idx": ai,
            "var_cols": vc, "var_pad_ok": pad, "sur_pos": sur_pos, "lat_mask": lat_mask, "teacher_ok": teacher_ok,
            "wu_ok": wu, "n_valid": torch.from_numpy(S["n_valid"]).to(dev)}


def onpolicy_cands(top_cand: torch.Tensor, v0, tokens: Sequence[str], epoch: int, seed: int, n_var: int = 32,
                   vcfg: Optional[Dict[str, Any]] = None, var_sampling: str = "type_balanced",
                   stream: str = STREAM_NOW, V: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    """On-policy candidates of the step (SPEC §2-2): top_cand [B,16,8,3] (student_topk, detached) ->
      V          online_variants(top_cand, v0) (all 96; recorded for the labeler); pass V to reuse a computed one
      cols       int64 [B,n] chosen variant columns (sample_cols, stream 'ck2e2e.now'), pad_ok bool [B,n]
      cand       f32 [B,16+n,8,3] = cat(top-16, traj96[cols]) (non-finite -> 0)
      ok         bool [B,16+n] finite & (identity | valid96[cols] & pad_ok)  (= now_ok: KD / teacher mask)
      vtype      int64 [B,16+n] (0 identity, 1..5), parent int64 [B,16+n] (top-16 rank k of the parent)."""
    x = top_cand.detach().float()
    B, K = x.shape[:2]
    if K != K16:
        raise ValueError(f"onpolicy_cands: top_cand has {K} candidates, expected {K16}")
    if V is None:
        V = online_variants(x, v0, vcfg)
    dev = x.device
    cols, pad = D2.sample_cols(V["valid96"].cpu().numpy(), tokens, epoch, seed, stream, n_var, var_sampling)
    c = torch.from_numpy(cols).to(dev)
    p = torch.from_numpy(pad).to(dev)
    var = _gather_k(V["traj96"], c)
    cand = torch.cat([x, var], 1)
    fin = torch.isfinite(cand).flatten(2).all(-1)
    cand = torch.where(fin[..., None, None], cand, torch.zeros_like(cand))
    ok = fin & torch.cat([torch.ones((B, K), dtype=torch.bool, device=dev), V["valid96"].gather(1, c) & p], 1)
    ar = torch.arange(K, device=dev).expand(B, K)
    return {"V": V, "cols": c, "pad_ok": p, "cand": cand, "ok": ok,
            "vtype": torch.cat([torch.zeros((B, K), dtype=torch.long, device=dev), c % NV], 1),
            "parent": torch.cat([ar, c // NV], 1)}


def fill_fallback(G: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], tokens: Sequence[str], epoch: int,
                  sampler: "D2.WarmupSampler", n_var: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Rows of a LabelStore2.lookup result with has False -> the warm-up-style official-label set (NU12): the 16 near +
    mid anchors of this epoch's AnchorSampler draw (slot order) + n_var variants of the fixed variant files
    (sampler.var_sampling, stream 'ck2e2e.lab'), G_src 0; a token without warm-up data (ck2_wu_ok False, row -1):
    G_ok all False, G_src 3.  G (CPU tensors) is updated in place and returned; G_col = variant-file column for the
    variant slots, -1 for the anchors."""
    has = G["has"].cpu().numpy().astype(bool)
    need = np.flatnonzero(~has)
    if len(need) == 0:
        return G
    KG = G["G_traj"].shape[1]
    nv = KG - N_NEAR_MID if n_var is None else int(n_var)
    if N_NEAR_MID + nv != KG:
        raise ValueError(f"fill_fallback: G has {KG} columns, fallback set is {N_NEAR_MID} + {nv}")
    toks = [str(tokens[i]) for i in need]
    S = sampler.sample(toks, _wu_arrays_np(targets, need), epoch, stream=STREAM_LAB, n_var=nv)
    t = {k: targets[k][torch.as_tensor(need, device=targets[k].device)].detach().cpu()
         for k in ("ck2_wu_ok", "ck2_raw_y", "ck2_raw_ok", "ck2_var_traj", "ck2_var_y", "ck2_var_ok", "ck2_var_valid")}
    anc = sampler.anchors
    for j, b in enumerate(need):
        wu = bool(t["ck2_wu_ok"][j]) and bool(S["ok"][j])
        G["G_epoch"][b] = -1
        if not wu:
            G["G_traj"][b] = 0.0
            G["G_y"][b] = 0.0
            G["G_ok"][b] = False
            G["G_vtype"][b] = 0
            G["G_col"][b] = -1
            G["G_src"][b] = SRC_NODATA
            continue
        a16 = S["anc_idx"][j][S["group"][j] <= 1]                              # 8 near + 8 mid, slot order
        assert len(a16) == N_NEAR_MID, len(a16)
        cols = torch.from_numpy(S["var_cols"][j])
        pad = torch.from_numpy(S["var_pad_ok"][j])
        a16t = torch.from_numpy(a16)
        traj = torch.cat([torch.from_numpy(anc[a16]), t["ck2_var_traj"][j][cols].float()], 0)
        y = torch.cat([t["ck2_raw_y"][j][a16t].float(), t["ck2_var_y"][j][cols].float()], 0)
        ok = torch.cat([t["ck2_raw_ok"][j][a16t].bool(),
                        t["ck2_var_ok"][j][cols].bool() & t["ck2_var_valid"][j][cols].bool() & pad], 0)
        fin_t = torch.isfinite(traj).flatten(1).all(-1)
        fin_y = torch.isfinite(y).all(-1)
        G["G_traj"][b] = torch.where(fin_t[:, None, None], traj, torch.zeros_like(traj))
        G["G_y"][b] = torch.where(fin_y[:, None], y, torch.zeros_like(y))
        G["G_ok"][b] = ok & fin_t & fin_y
        G["G_vtype"][b] = torch.cat([torch.zeros(N_NEAR_MID, dtype=torch.int8), (cols % NV).to(torch.int8)])
        G["G_col"][b] = torch.cat([torch.full((N_NEAR_MID,), -1, dtype=torch.int16), cols.to(torch.int16)])
        G["G_src"][b] = SRC_FALLBACK
    return G


def onpolicy_label_set(store: "D2.LabelStore2", targets: Dict[str, torch.Tensor], rows, tokens: Sequence[str],
                       epoch: int, seed: int, sampler: "D2.WarmupSampler", n_var: int = 32, device=None
                       ) -> Dict[str, torch.Tensor]:
    """The BCE set G of an on-policy step: LabelStore2.lookup (generation <= e - lag, 16 identities + n_var
    variants) + fill_fallback for rows without one; moved to device.  Keys as LabelStore2.lookup."""
    G = store.lookup(rows, tokens, epoch, seed, n_orig=K16, n_var=n_var)
    G = fill_fallback(G, targets, tokens, epoch, sampler, n_var)
    if device is not None:
        G = {k: v.to(device, non_blocking=True) for k, v in G.items()}
    return G


__all__ = ["VARIANTS_DEFAULT", "variants_cfg", "check_variant_files", "v0_from_status", "online_variants",
           "warmup_cands", "onpolicy_cands", "fill_fallback", "onpolicy_label_set", "K16", "NV", "G_K", "VNAMES"]
