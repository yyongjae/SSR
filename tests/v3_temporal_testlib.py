"""Shared helpers of the refiner-v3 temporal-branch tests (tests/test_v3_temporal_model.py, test_v3_temporal_real.py).

  load_a3(ft)        the report-51 a3_conflict_features.py SOURCE executed in a private namespace (never imported, so
                     no __pycache__ is written next to it and sys.path is not touched); `ft` = the future_teacher
                     module to bind as FT (a stub with the BX_* layout for synthetic tests).
  scene builders     synthetic pack-layout object tensors (rows section 3.2) with CA-consistent histories.
  a3 fakes           minimal stand-ins for a3 Worker state so Worker.history / Worker.world / Worker.__call__ run on
                     synthetic objects (the a3 reduction code is the reference of gap_scalars).
  np_gap_general     independent numpy reference of the M-mode gap-scalar definition (raster.py docstring).
  candidates_hooked  CKTrunk.candidates with the SPEC 4.5 tfeat hook (the planned model.py edit, copied verbatim
                     from 8ae8b1d plus the two hook lines) for integration tests without editing model.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import torch  # noqa: E402

from navsim.agents.para_ssr.ck.temporal import config as TC  # noqa: E402

A3_PATH = Path("/workspace/yongjae/SSR/report/51_teacher_specialization_evidence/verify/A_ceiling/scripts/"
               "a3_conflict_features.py")
FT_STUB = SimpleNamespace(**{n: i for i, n in enumerate(("BX_X", "BX_Y", "BX_Z", "BX_L", "BX_W", "BX_H", "BX_HEAD",
                                                         "BX_VX", "BX_VY"))})
CLASS_LW = {0: (4.6, 1.9), 1: (0.7, 0.7), 2: (1.8, 0.6), 3: (0.4, 0.4), 4: (2.0, 0.5), 5: (0.6, 0.3), 6: (1.0, 1.0)}


# ----------------------------------------------------------------------------------------------- a3 source
def load_a3(ft=None) -> SimpleNamespace:
    """Execute a3_conflict_features.py (main() not run) with FT bound to `ft` (default: BX_* stub)."""
    src = A3_PATH.read_text()
    a = "if SSR not in sys.path:\n    sys.path.insert(0, SSR)\n"
    b = "from navsim.agents.para_ssr.refiner import future_teacher as FT  # noqa: E402  (read-only use)\n"
    assert a in src and b in src, "a3 source layout changed"
    src = src.replace(a, "").replace(b, "FT = _FT\n")
    ns = {"__name__": "a3_src", "__file__": str(A3_PATH), "_FT": ft if ft is not None else FT_STUB}
    exec(compile(src, str(A3_PATH), "exec"), ns)
    return SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})


# ----------------------------------------------------------------------------------------------- rows / scenes
def row(x, y, h, L, W, vx, vy, score, cls, age):
    r = np.zeros(TC.D_OBJ, np.float64)
    r[:9] = (x, y, np.cos(h), np.sin(h), L, W, vx, vy, score)
    r[TC.COL["cls0"] + int(cls)] = 1.0
    r[TC.COL["age"]] = age
    return r


def rows_to_ft(rows: np.ndarray):
    """canonical rows [n, 17] -> (FT-layout boxes [n, 9] (x, y, z, L, W, H, heading, vx, vy), labels [n])."""
    b = np.zeros((len(rows), 9))
    b[:, 0], b[:, 1] = rows[:, 0], rows[:, 1]
    b[:, 3], b[:, 4], b[:, 5] = rows[:, 4], rows[:, 5], 1.5
    b[:, 6] = np.arctan2(rows[:, 3], rows[:, 2])
    b[:, 7], b[:, 8] = rows[:, 6], rows[:, 7]
    lab = rows[:, TC.COL["cls0"]:TC.COL["cls0"] + TC.N_CLS].argmax(1)
    return b, lab


def rand_token(g: np.random.Generator, n0: int, dts=(0.5, 1.0), n_distract: int = 3, p_hist: float = 0.8):
    """One token: n0 t0 objects (CA-consistent past boxes for a random subset) + distractors in the past frames.
    Returns frames list of [n_f, 17] rows (score-sorted, f = 0, 1, 2)."""
    frames = [[], [], []]
    for i in range(n0):
        cls = int(g.choice(7, p=[0.45, 0.15, 0.08, 0.12, 0.08, 0.06, 0.06]))
        L, W = CLASS_LW[cls]
        x, y = g.uniform(-8, 45), g.uniform(-14, 14)
        h = g.uniform(-np.pi, np.pi)
        if cls == 0:
            sp = g.choice([0.0, 0.3, g.uniform(1, 14)])
        elif cls in (1, 2):
            sp = g.uniform(0, 3)
        else:
            sp = g.uniform(0, 0.08)
        v = sp * np.array([np.cos(h), np.sin(h)])
        a = g.uniform(-4, 3) if cls < 3 else 0.0
        sc = g.uniform(0.3, 0.95)
        frames[0].append(row(x, y, h, L, W, v[0], v[1], sc, cls, 0.0))
        if g.uniform() < p_hist:
            for f, dt in zip((1, 2), dts):
                vp = v - a * dt * (v / max(np.hypot(*v), 1e-6))
                c = np.array([x, y]) - v * dt + 0.5 * a * dt * dt * (v / max(np.hypot(*v), 1e-6)) + g.normal(0, 0.15, 2)
                frames[f].append(row(c[0], c[1], h, L, W, vp[0], vp[1], g.uniform(0.2, 0.9), cls, dt))
    for f, dt in zip((1, 2), dts):
        for _ in range(n_distract):
            cls = int(g.integers(0, 7))
            L, W = CLASS_LW[cls]
            frames[f].append(row(g.uniform(-8, 45), g.uniform(-14, 14), g.uniform(-3, 3), L, W, g.normal(0, 2),
                                 g.normal(0, 2), g.uniform(0.2, 0.9), cls, dt))
    out = []
    for fr in frames:
        a = np.array(fr, np.float64).reshape(-1, TC.D_OBJ)
        out.append(a[np.argsort(-a[:, TC.COL["score"]], kind="stable")][:TC.K_OBJ])
    return out


def pack_tokens(tokens, K: int = TC.K_OBJ, frame_valid=None, dtype=torch.float32):
    """list of per-token frame lists -> obj_in [T, 3, K, 17], obj_mask [T, 3, K], frame_valid [T, 3]."""
    T = len(tokens)
    obj = np.zeros((T, 3, K, TC.D_OBJ))
    mask = np.zeros((T, 3, K), bool)
    fv = np.ones((T, 3), bool) if frame_valid is None else np.asarray(frame_valid, bool)
    for t, frames in enumerate(tokens):
        for f, a in enumerate(frames):
            n = min(len(a), K)
            obj[t, f, :n] = a[:n]
            mask[t, f, :n] = fv[t, f]
            if not fv[t, f]:
                obj[t, f] = 0
    return (torch.as_tensor(obj, dtype=dtype), torch.as_tensor(mask), torch.as_tensor(fv))


def rand_objs(T=2, n0=(12, 7), seed=0, K=TC.K_OBJ, dtype=torch.float32, **kw):
    g = np.random.default_rng(seed)
    toks = [rand_token(g, n, **kw) for n in n0[:T]]
    return pack_tokens(toks, K, dtype=dtype), toks


def straight(x_knots, y=0.0):
    """Candidate [8, 3] along +x through the given 8 x positions (heading 0)."""
    x = torch.as_tensor(x_knots, dtype=torch.float32)
    return torch.stack([x, torch.full_like(x, y), torch.zeros_like(x)], -1)


def rand_cands(T, Kc, seed=0, v=(4.0, 12.0)):
    """Random curved candidates [T*Kc, 8, 3] (constant curvature, constant speed per candidate)."""
    g = torch.Generator().manual_seed(seed)
    B = T * Kc
    sp = v[0] + (v[1] - v[0]) * torch.rand(B, generator=g)
    kap = (torch.rand(B, generator=g) - 0.5) * 0.08
    acc = (torch.rand(B, generator=g) - 0.5) * 3.0
    t = 0.5 * torch.arange(1, 9, dtype=torch.float32)
    s = torch.clamp(sp[:, None] * t + 0.5 * acc[:, None] * t ** 2, min=0.0)
    s = torch.cummax(s, 1).values
    th = kap[:, None] * s
    x = torch.where(kap[:, None].abs() > 1e-6, torch.sin(th) / torch.where(kap.abs() > 1e-6, kap, torch.ones_like(kap))[:, None], s)
    y = torch.where(kap[:, None].abs() > 1e-6, (1 - torch.cos(th)) / torch.where(kap.abs() > 1e-6, kap, torch.ones_like(kap))[:, None], torch.zeros_like(s))
    return torch.stack([x, y, th], -1)


# ----------------------------------------------------------------------------------------------- a3 fakes
class FakeHistory:
    """Worker state for A3.Worker.history(self, 't0', b0, l0) on one synthetic token (all frames in the N frame)."""

    def __init__(self, frames, valid=(True, True, True)):
        ts0 = 10_000_000
        self.frames_rows = frames
        fr = {"t0": ("L", 10, ts0)}
        self.bypos = {}
        for f in (1, 2):
            if valid[f]:
                age = float(frames[f][0, TC.COL["age"]]) if len(frames[f]) else 0.5 * f
                fr[f"p{f}"] = ("L", 10 - f, ts0 - int(round(age * 1e6)))
                self.bypos[("L", 10 - f)] = f"p{f}"
        self.ftc = SimpleNamespace(frames=fr, pose=lambda tok: (0.0, 0.0, 0.0))

    def boxes(self, tok, pose0=None, thr=0.3):
        f = 0 if tok == "t0" else int(tok[1])
        rows = self.frames_rows[f]
        rows = rows[rows[:, TC.COL["score"]] >= thr]
        return rows_to_ft(rows)


EMPTY_FUT = dict(valid=np.zeros(10, bool), dt=np.zeros(10), boxes=[np.zeros((0, 9))] * 10,
                 labels=[np.zeros(0)] * 10, keep=[np.zeros(0, bool)] * 10)


class FakeCall:
    """Worker state for A3.Worker.__call__((tok, traj, v0)): t0 boxes b0 / l0, CA slot replaced by `world_ca(tau)`
    (returns c [nt, N, 2]) -- so the a3 SAT + reduction code scores an arbitrary forecast world."""

    def __init__(self, A3, b0, l0, world_ca=None, acc=None):
        self.A3, self.b0, self.l0 = A3, b0, l0
        self.acc = np.zeros(len(b0)) if acc is None else acc
        self.world_ca = world_ca
        self.ftc = SimpleNamespace(pose=lambda tok: (0.0, 0.0, 0.0), future_boxes=lambda tok, k, score_thr: EMPTY_FUT)

    def boxes(self, tok, pose0=None, thr=0.3):
        return self.b0, self.l0

    def history(self, tok, b0, l0):
        return self.acc, np.zeros(len(b0), bool), 0

    def track(self, b0, l0, fut):
        return [(np.array([0.0]), b0[i:i + 1]) for i in range(len(b0))]

    def world(self, kind, tau, b0, l0, acc, fut, tracks=None):
        if kind == "CA" and self.world_ca is not None:
            c = self.world_ca(tau)
            T = len(tau)
            h = np.broadcast_to(b0[None, :, 6], (T, len(b0)))
            L = np.broadcast_to(b0[None, :, 3], (T, len(b0)))
            W = np.broadcast_to(b0[None, :, 4], (T, len(b0)))
            ag = np.broadcast_to(np.isin(l0, self.A3.AGENT)[None], (T, len(b0)))
            return c, h, L, W, ag
        return self.A3.Worker.world(self, kind, tau, b0, l0, acc, fut, tracks)


def np_knot_world(mu, c0, t):
    """numpy piecewise-linear evaluation of knots mu [N, 10, 2] (c0 [N, 2] at t = 0) at times t [nt] -> [nt, N, 2]."""
    P = np.concatenate([c0[:, None], mu], 1)                                         # [N, 11, 2]
    tk = 0.5 * np.arange(11)
    out = np.empty((len(t), len(c0), 2))
    for n in range(len(c0)):
        for d in range(2):
            out[:, n, d] = np.interp(np.clip(t, 0, 5), tk, P[n, :, d])
    return out


# ----------------------------------------------------------------------------------------------- general gap reference
def np_gap_general(A3, traj, c0, h0, L, W, agent, static, xyF, pi, present):
    """Independent numpy reference of raster.gap_scalars (raw units) for one token.
    traj [C, 8, 3]; objects: c0 [N, 2], h0, L, W [N], agent / static [N] bool (valid only); xyF [N, M, 64, 2] forecast
    positions at tau = TJ + delta (slot q = 4 j + d); pi [N, M]; present [N, M, 64] bool.  -> [C, 20]."""
    pose, sp = A3.interp_traj(traj, 0.0)
    C = traj.shape[0]
    hd = pose[..., 2]
    u = np.stack([np.cos(hd), np.sin(hd)], -1)
    cen = pose[..., :2] + A3.EGO_OFF * u
    ce = (cen[:, :, None, :] + (sp[:, :, None, None] * A3.DELTAS[None, None, :, None]) * u[:, :, None, :]).reshape(C, 64, 2)
    he = np.broadcast_to(hd[:, :, None], (C, 16, 4)).reshape(C, 64)
    N, M = pi.shape if len(pi) else (0, 1)
    big = 15.0

    def red(s_ag, s_agF, s_st):
        f = np.empty((C, 9))
        f[:, 0] = s_ag[:, :, 0].min(1)
        f[:, 1] = s_agF[:, :, 0].min(1)
        f[:, 2] = s_st[:, :, 0].min(1)
        f[:, 3] = s_ag[:, :, 1:].min((1, 2))
        f[:, 4] = s_agF[:, :, 1:].min((1, 2))
        f[:, 5] = s_st[:, :, 1:].min((1, 2))
        hit0 = s_ag[:, :, 0] < 0.3
        f[:, 6] = np.where(hit0.any(1), A3.TJ[hit0.argmax(1)], 4.5)
        hitT = s_agF[:, :, 1:].min(2) < 0.3
        f[:, 7] = np.where(hitT.any(1), A3.TJ[hitT.argmax(1)], 4.5)
        f[:, 8] = (s_agF.min(2) < 0.5).sum(1)
        return f

    sS = np.full((C, 64, max(N, 1)), big)
    fwS = np.zeros((C, 64, max(N, 1)), bool)
    sF = np.full((C, 64, max(N, 1), M), big)
    fwF = np.zeros((C, 64, max(N, 1), M), bool)
    for q in range(64):
        if N == 0:
            break
        s, fw = A3.sep_rect(ce[:, q], he[:, q], c0, h0, L, W)
        sS[:, q], fwS[:, q] = np.clip(s, -5, 15), fw > 0
        for m in range(M):
            s, fw = A3.sep_rect(ce[:, q], he[:, q], xyF[:, m, q], h0, L, W)
            sF[:, q, :, m], fwF[:, q, :, m] = np.clip(s, -5, 15), fw > 0
    ag = np.zeros(max(N, 1), bool)
    st = np.zeros(max(N, 1), bool)
    ag[:N], st[:N] = agent, static
    s_ag = np.where(ag, sS, big).min(-1).reshape(C, 16, 4)
    s_agF = np.where(ag & fwS, sS, big).min(-1).reshape(C, 16, 4)
    s_st = np.where(st, sS, big).min(-1).reshape(C, 16, 4)
    fS = red(s_ag, s_agF, s_st)
    if N:
        pr = np.transpose(present, (2, 0, 1))[None]                                  # [1, 64, N, M]
        sp_ = np.where(pr, sF, big)
        spF = np.where(pr & fwF, sF, big)
        e_ag = (pi[None, None] * sp_).sum(-1)
        e_agF = (pi[None, None] * spF).sum(-1)
        pl = pi >= 0.1
        pl = np.where(pl.any(-1, keepdims=True), pl, np.eye(M, dtype=bool)[pi.argmax(-1)])
        w_ag = np.where(pl[None, None], sp_, big).min(-1)
    else:
        e_ag = e_agF = w_ag = np.full((C, 64, 1), big)
    F_ag = np.where(ag, e_ag, big).min(-1).reshape(C, 16, 4)
    F_agF = np.where(ag, e_agF, big).min(-1).reshape(C, 16, 4)
    fF = red(F_ag, F_agF, s_st)
    W_ag = np.where(ag, w_ag, big).min(-1).reshape(C, 16, 4)
    return np.concatenate([fS, fF, W_ag[:, :, 0].min(1)[:, None], W_ag[:, :, 1:].min((1, 2))[:, None]], 1)


# ----------------------------------------------------------------------------------------------- trunk hook
def candidates_hooked(trunk, feat, mem, tau, v0, a0, eds, cmd, tfeat=None):
    """CKTrunk.candidates @ 8ae8b1d + the SPEC 4.5 tfeat hook (planned model.py edit)."""
    from navsim.agents.para_ssr.refiner.corridor import N_ST, corridor_geometry, corridor_tensor
    from navsim.agents.para_ssr.refiner.decoder import N_FREE
    from navsim.agents.para_ssr.refiner.geometry import N_POSE
    from navsim.agents.para_ssr.refiner.refiner_net import draft_features
    T, K = tau.shape[:2]
    B = T * K
    with torch.autocast(device_type=tau.device.type, enabled=False):
        flat = tau.reshape(B, N_POSE, 3).float()
        geom = corridor_geometry(flat)
        rep = lambda x: x.reshape(T, *x.shape[1:]).repeat_interleave(K, 0)  # noqa: E731
        ud = draft_features(flat, rep(v0.float()), rep(a0.float()), rep(eds.float()), rep(cmd), geom)
    X = corridor_tensor(feat, geom, T, K)
    Hc = trunk.enc(X)
    st = trunk.station(Hc.permute(0, 2, 1, 3).reshape(B, N_ST, -1))
    if tfeat is None:
        if getattr(trunk, "_temporal_required", False):
            raise RuntimeError("temporal net called without tfeat")
        x = torch.cat([trunk.draft_mlp(ud.to(st.dtype))[:, None], st], 1)
    else:
        st_add, ud_add = tfeat(geom, flat)
        x = torch.cat([(trunk.draft_mlp(ud.to(st.dtype)) + ud_add)[:, None], st + st_add], 1)
    for layer in trunk.layers:
        x = layer(x, mem, T, K)
    x = trunk.norm_out(x)
    xd = x[:, 0]
    gate = trunk.gate_head(xd.detach())[:, 0]
    h = torch.cat([xd, x[:, 1:].mean(1)], -1)
    return {"h": h, "gate_logit": gate.reshape(T, K), "z_lon": trunk.lon_head(h).reshape(T, K, N_FREE),
            "w_lat": trunk.lat_head(h).reshape(T, K, N_FREE), "ud": ud, "geom": geom}
