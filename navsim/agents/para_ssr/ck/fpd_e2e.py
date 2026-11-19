"""CK2 e2e with FPD-Drive's refiner structure (ck_e2e2.arch 'fpd'; FPD-Drive arXiv 2610.13232, net in ck/fpd_net.py).

Used ONLY when ck_e2e2.enabled and ck_e2e2.arch == 'fpd' (para_ssr_agent dispatches on arch; arch 'ck2', the default,
never imports this module).  Everything of the CK2 e2e system (ck/online2.py) is kept -- v2 AnchorPlanner first stage,
schedule, on-policy labels of the background labeler (labeler2 labels the recorded 96 columns), online KD calibration,
BEV-KD arm, recorder / label store / callback / checkpoint state, evaluation keys -- and only the refiner structure is
swapped: CKNet + 80 rule variants -> FPDRefinerNet (scene tokens + residual DiT + candidate scorer) + 80 DiT draws.

Pool (every phase that needs it, c = 6 k + v as the CK2 base layout): parents = the current v2 top-16 (student_topk,
  detached), v 0 = the parent bitwise, v 1..5 = student DiT draws parent + DDIM residual (fpd_net.pool96).  Training
  noise = fpd_net.token_noise(token, seed, epoch, 'fpd.noise') (new draws every epoch), evaluation noise epoch -1 (one
  fixed noise per scene, keyed by the scene token: FPDE2E.infer_tokens, else the status bytes).
Phases (CKE2E2.phase / recording / r unchanged)
  'warmup'         CKE2E2.warmup_batch: 32 sampled raw anchors (warmup.n_var 0) = KD set = BCE set (official labels);
                   no draws are sampled.
  'warmup_record'  the warm-up step + the pool of this step (16 parents + 80 DiT draws) RECORDED for the labeler.
  'onpolicy'       KD set = all 96 current pool columns (teacher score KD, cand_ok = finite); BCE set = the label set
                   G of the lagged generation (16 identities + its 80 recorded DiT draws, onpolicy_label.n_var 80).
                   Rows without a generation: the warm-up-style fallback with every rule-variant slot masked (G_ok
                   False on columns 16:, their trajectories replaced by v2's top-1 before the scorer: no rule-variant
                   file trajectory reaches the FPD scorer; the 16 raw near + mid anchors stay).  The pool of the step
                   is recorded while recording.
  The scorer attends across its candidate set, so the KD set and the BCE set are scored in SEPARATE passes (also by
  the teachers: FPDTeachers.run scores `extra` as its own set).
Loss (added after the v2 grad balancer, as CK2)
  L = loss_weight * [ lambda_score * BCE(s_lab, y_lab) + r * lambda_kd_score * KL(p_T || sigmoid(s_kd)) (KD keys)
                      + fpd.lambda_diff * L_diff + fpd.lambda_traj * L_traj ]  + 0 * (s_kd + s_lab)
  L_diff / L_traj = fpd_net.dit_loss (FPD Eq. 5-6: Hungarian match of the targets to distinct current parents, epsilon
  MSE + one-step reconstruction d1) on the DiT targets of every phase: the GT trajectory (fpd.use_gt_target) + up to
  fpd.mt_k - 1 (mt_k without the GT) teacher targets = fpd_net.select_targets (FPD Sec. 3.3: safety pass, best
  PDMS, greedy max-min ADE >= mt_min_ade) on the teacher bank.  fpd.mt_source 'bank' (default): FPDBankStore, the
  teachers' OFFLINE banks <teacher run>/bank/<split> (32 teacher-DiT draws of the 16 near + mid anchors per token and
  teacher, OFFICIAL PDM labels; safety = label NC, DAC >= mt_label_thr, score = label PDMS): FPD-Drive's cached targets
  labelled by an evaluator, and no teacher sampling in the training step.  'online': FPDTeachers.bank, each source
  teacher's own DiT draws of the CURRENT parents every step (its own DDIM steps, fpd.mt_draws per parent, noise stream
  'fpd.teacher_bank'), scored by the RAW teacher scorers as one set (safety P(NC), P(DAC) >= mt_thr).  No surrogate, no lateral head (validate: lambda_sur 0, lat_kd fixed 0); the CK2 lateral logs are 0.
  Gradient: candidates / parents / targets detached; scorer + DiT losses -> scene tokens -> v2 bev_embed (x
  bev_grad_scale).  A non-finite L is replaced by 0 * sum(student params) (counted).  No collective besides the
  inherited ones (online calibration refit, BEV-KD measurement), which every rank reaches on the same micro-batch.
Teachers (FPDTeachers, plain object): frozen FPDRefinerNet per teacher on targets kd_bev_0 (det, arm T) / kd_bev_1 (map,
  arm M), fp16 autocast on CUDA (teacher_amp); run() has TeacherPair2.run's keys and semantics (combine_teacher, Platt
  calibration, sources; the lateral outputs are zeros).
Inference: CK2's ck2_* keys over the 96-column pool (score logits of the FPD scorer; ck2_w_lat / ck2_e_lat zeros,
  ck2_lat_traj = the pool trajectories the scorer saw = ck2_cand96 on every finite column); infer_select as CKE2E2.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import fpd_net as FN
from .constants import CK_KEYS, KD_SCORE_SOURCE, SUR_TERMS
from .e2e_data2 import K16, N_NEAR_MID, NV, SRC_FALLBACK, SRC_NAMES
from .online import _sanitize_traj, bev_sgrid, student_topk
from .online2 import CKE2E2, KD_SOURCES, PHASE2_CODE, CKE2E2Config, _f, score_prior_of, v0_from_status

BANK_STREAM = "fpd.teacher_bank"
N_LAT = 6                               # ck2_w_lat / ck2_e_lat width (CK2 keys, zeros here)


# ----------------------------------------------------------------------------------------------- student
def build_student_fpd(cfg, apply_prior: bool = True) -> FN.FPDRefinerNet:
    """FPDRefinerNet(seed, steps = fpd.steps, grid = fpd.grid), tok.bev_grad_scale = cfg.bev_grad_scale, scorer head
    bias = logit(prior) (online2.score_prior_of, the build_student_ck2 rule) unless apply_prior is False or
    score_prior 'none' (a checkpoint will be loaded: no file is opened then).  No global RNG is consumed (the net forks
    the RNG; the prior is numpy).  Nothing is frozen."""
    cfg = CKE2E2Config.from_any(cfg)
    f = cfg.fpd
    net = FN.FPDRefinerNet(seed=int(cfg.seed), steps=int(f["steps"]), grid=int(f["grid"]))
    net.tok.bev_grad_scale = float(cfg.bev_grad_scale)
    rep: Dict[str, Any] = {"source": None, "score_prior": None, "score_prior_source": None, "frozen": [],
                           "arch": "fpd", "param_counts": net.param_counts()}
    if apply_prior and cfg.score_prior != "none":
        prior, src = score_prior_of(cfg)
        rep["score_prior_source"] = src
        if prior is not None:
            net.scorer.set_prior(prior)
            rep["score_prior"] = [float(v) for v in prior]
    net.init_report = rep
    return net


# ----------------------------------------------------------------------------------------------- teachers
class FPDTeachers:
    """Frozen FPD DET (arm T) + MAP (arm M) teachers (fpd_net.load_fpd_teacher of tools/ck/fpd/train_fpd_teacher runs)
    run online on the student's candidates, each on its privileged teacher BEV.  Plain python object (not an nn.Module
    attribute of the agent: not in the state_dict / DDP / optimiser).  The online2.TeacherPair2 interface: load-time
    checks (config.json arm T / M, trainer train_fpd_teacher, done.json with require_done, ep_target (missing =
    'official') == ep_target when given), calib (KD calibration files, file mode: kd_calib.load_teacher_calib against
    this run / which / ckpt sha16 / ep_target / arm) or set_calib_arrays (online mode), sources, run(); plus bank()."""

    def __init__(self, det_run: str, map_run: str, which: str = "last", device="cpu", amp: bool = True,
                 require_done: bool = True, ep_target: Optional[str] = None,
                 calib: Optional[Dict[str, Optional[str]]] = None, kd_keys: Optional[Sequence[str]] = None,
                 sources: Sequence[str] = KD_SOURCES):
        from .ep_target import check_ep_target, run_ep_target

        if not det_run or not map_run:
            raise ValueError("FPDTeachers needs both the DET and the MAP run")
        self.device = torch.device(device)
        self.sources = tuple(t for t in KD_SOURCES if t in tuple(sources))
        if not self.sources:
            raise ValueError(f"FPDTeachers sources must be a non-empty subset of {KD_SOURCES}, got {list(sources)}")
        self.runs = {"det": str(det_run), "map": str(map_run)}
        self.which = str(which)
        nets, eps = {}, {}
        for name, run in (("det", det_run), ("map", map_run)):
            arm = FN.FPD_TEACHER_ARMS[name]
            rd = Path(run)
            cj = rd / "config.json"
            if not cj.is_file():
                raise ValueError(f"FPD teacher {name} {run}: no config.json")
            cfg = json.loads(cj.read_text())
            if cfg.get("arm") != arm:
                raise ValueError(f"FPD teacher {name} {run}: arm {cfg.get('arm')!r}, expected {arm!r}")
            if cfg.get("trainer") != FN.FPD_TEACHER_TRAINER:
                raise ValueError(f"FPD teacher {name} {run}: trainer {cfg.get('trainer')!r} != "
                                 f"{FN.FPD_TEACHER_TRAINER!r} (FPD teacher required)")
            if require_done and not (rd / "done.json").is_file():
                raise ValueError(f"FPD teacher {name} {run}: no done.json (training not finished; "
                                 f"teacher_require_done=false only for smoke runs / tests)")
            eps[name] = got = run_ep_target(cfg)
            if ep_target is not None:
                want = check_ep_target(ep_target)
                if got != want:
                    raise ValueError(f"FPD teacher {name} {run}: ep_target {got!r} (config.json; missing = "
                                     f"'official') != the student's ck_e2e2.ep_target {want!r} (label BCE and EP KD "
                                     f"would disagree)")
            net, cfg = FN.load_fpd_teacher(run, which, device="cpu", require_done=require_done)
            nets[name] = (net.to(self.device), cfg)
        self.det, self.det_cfg = nets["det"]
        self.map, self.map_cfg = nets["map"]
        self.amp = bool(amp)
        self.ep_target = eps                 # each teacher run's EP target (config.json)
        self.ckpt = {k: getattr(n, "init_report", {}).get("ckpt_sha16") for k, n in (("det", self.det),
                                                                                    ("map", self.map))}
        # keys each teacher was trained on (config.json score_keys; T nc / ttc / ep / comfort, M dac / ep)
        self.score_keys = {k: tuple(c.get("score_keys") or ()) for k, c in (("det", self.det_cfg),
                                                                           ("map", self.map_cfg))}
        self.bank_sources = self._bank_key_sources()
        # KD calibration (Platt on the logits, per teacher and key); None = off
        self.calib: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
        self.calib_info: Dict[str, Any] = {}
        if calib is not None:
            from .kd_calib import load_teacher_calib
            bad = sorted(set(calib) - {"det", "map"})
            if bad:
                raise ValueError(f"FPDTeachers calib: unknown teacher(s) {bad}")
            cal = {}
            for name in ("det", "map"):
                path = calib.get(name)
                if not path:
                    continue
                try:
                    a, b, info = load_teacher_calib(path, run=self.runs[name], which=which,
                                                    ckpt_sha16=self.ckpt[name], ep_target=eps[name],
                                                    arm=FN.FPD_TEACHER_ARMS[name], kd_keys=kd_keys)
                except (ValueError, FileNotFoundError) as e:
                    raise ValueError(f"FPD teacher {name} {self.runs[name]}: KD calibration refused: {e}") from None
                cal[name] = (torch.from_numpy(a).to(self.device), torch.from_numpy(b).to(self.device))
                self.calib_info[name] = info
            self.calib = cal or None

    def _bank_key_sources(self) -> List[Tuple[str, ...]]:
        """per CK key, the teachers whose probabilities are averaged for the bank: the source teachers trained on the
        key (config.json score_keys), else the KD rule (constants.KD_SCORE_SOURCE det / map / mean over the source
        teachers).  Default split -> nc / ttc / comfort DET, dac MAP, ep mean (= the KD rule except comfort, which
        the MAP teacher is not trained on)."""
        out = []
        for k in CK_KEYS:
            trained = tuple(t for t in self.sources if k in self.score_keys[t])
            if trained:
                out.append(trained)
                continue
            rule = tuple(t for t in self.sources if KD_SCORE_SOURCE[k] in (t, "mean"))
            out.append(rule or self.sources)
        return out

    def set_calib_arrays(self, params: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """online KD calibration (kd_calib.OnlineKDCalib.params): {'det' / 'map': (a [5], b [5])} -> the logits'
        a z + b of run() / bank(); identity entries reproduce the uncalibrated logits exactly."""
        dev = self.device
        self.calib = {k: (torch.as_tensor(np.asarray(a, np.float32), device=dev),
                          torch.as_tensor(np.asarray(b, np.float32), device=dev)) for k, (a, b) in params.items()}

    def nets(self):
        return [self.det, self.map]

    def to(self, device) -> "FPDTeachers":
        device = torch.device(device)
        if device != self.device:
            for n in self.nets():
                n.to(device)
            if self.calib is not None:
                self.calib = {k: (a.to(device), b.to(device)) for k, (a, b) in self.calib.items()}
            self.device = device
        return self

    def _autocast(self):
        dev = self.device
        return torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=self.amp and dev.type == "cuda")

    def _cal(self, name: str, z: torch.Tensor) -> torch.Tensor:
        ab = None if self.calib is None else self.calib.get(name)
        return z if ab is None else z * ab[0] + ab[1]

    @torch.no_grad()
    def run(self, bev_det, ok_det, bev_map, ok_map, cand, status, cand_ok=None, extra=None
            ) -> Dict[str, torch.Tensor]:
        """TeacherPair2.run on the FPD teachers: bev_* [B, 256, H, W] (f16 cache), ok_* [B], cand [B, K, 8, 3]
        (finite; ONE set per token: the scorer attends across it), status [B, 8], cand_ok [B, K] -> kd_score_prob f32
        [B,K,5] (combine_teacher: nc / ttc DET, dac MAP, ep / comfort mean; one source: that teacher alone), kd_ok
        [B,K] = token ok of the source teacher(s) & finite logits & cand_ok, z_det / z_map the RAW logits, ok_det /
        ok_map each teacher's own ok, p_det / p_map (calibrated) probabilities, kd_e_lat / e_det / e_map zeros [B,K,6]
        (no lateral head).  With calibration each teacher's logits are a_k z + b_k before the sigmoid / combination.
        extra [B, K2, 8, 3] (online calibration's label set) is scored as a SEPARATE set on the same scene tokens (the
        KD outputs are exactly those of a call without extra) -> z_det_x / z_map_x raw logits, ok_det_x / ok_map_x."""
        from .kd import combine_teacher

        dev = self.device
        cand = cand.detach().float().to(dev)
        B, K = cand.shape[:2]
        st = status.detach().float().to(dev)
        if extra is not None:
            extra = extra.detach().float().to(dev)
            K2 = extra.shape[1]
        logits, raw_x = [], []
        for net, bev in ((self.det, bev_det), (self.map, bev_map)):
            with self._autocast():
                ctx = net.scene(bev.to(dev).float())
                s = net.score(cand, ctx, st)
                sx = net.score(extra, ctx, st) if extra is not None else None
            logits.append(s.float())
            if extra is not None:
                raw_x.append(sx.float())
        raw = list(logits)
        logits = [self._cal("det", logits[0]), self._cal("map", logits[1])]

        def tok_ok(ok_tok, s, k):
            return ok_tok.reshape(-1).bool().to(dev)[:, None].expand(-1, k) & torch.isfinite(s).all(-1)

        det = {"score_logit": logits[0], "ok": tok_ok(ok_det, logits[0], K)}
        mp = {"score_logit": logits[1], "ok": tok_ok(ok_map, logits[1], K)}
        if self.sources == ("det",):
            comb = combine_teacher(det, None)
        elif self.sources == ("map",):
            comb = combine_teacher(mp, None)
        else:
            comb = combine_teacher(det, mp)
        kd_ok = comb["kd_ok"].bool()
        if cand_ok is not None:
            kd_ok = kd_ok & cand_ok.bool().to(dev)
        z6 = torch.zeros(B, K, N_LAT, dtype=torch.float32, device=dev)
        res = {"kd_score_prob": comb["kd_score_prob"].float(), "kd_e_lat": z6, "kd_ok": kd_ok,
               "e_det": z6.clone(), "e_map": z6.clone(), "p_det": torch.sigmoid(logits[0]),
               "p_map": torch.sigmoid(logits[1]), "z_det": raw[0], "z_map": raw[1], "ok_det": det["ok"],
               "ok_map": mp["ok"]}
        if extra is not None:
            for i, (name, ok_tok) in enumerate((("det", ok_det), ("map", ok_map))):
                res[f"z_{name}_x"] = raw_x[i]
                res[f"ok_{name}_x"] = tok_ok(ok_tok, raw_x[i], K2)
        return res

    @torch.no_grad()
    def bank(self, bev_det, ok_det, bev_map, ok_map, parents, status, keys: Sequence[str], n_draws: int,
             seed: int = 0, epoch: int = -1, steps: Optional[int] = None
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The teachers' DiT samples for FPD's multi-target distillation: every source teacher samples n_draws draws
        per parent with its own DiT on its own BEV (noise token_noise(keys, (K, n_draws, 8, 3), seed, epoch,
        'fpd.teacher_bank'), the same for both teachers; DDIM steps = steps, None (FPDE2E.loss) = each teacher's own
        config.json steps, the sampler its scorer was trained / evaluated on, not the student's fpd.steps), parents
        [B, K, 8, 3] -> bank [B, M, 8, 3] (M = n_sources x K
        x n_draws: DET draws (k-major) first, then MAP; a non-finite draw is replaced by its parent and is not ok), the
        bank scored by the source teacher scorers as ONE set -> prob [B, M, 5] (RAW teacher probabilities, not the KD calibration, combined per key as
        bank_sources), ok [B, M] (token ok of every source teacher & finite draw & finite logits)."""
        dev = self.device
        par = parents.detach().float().to(dev)
        B, K = par.shape[:2]
        D = int(n_draws)
        shp = tuple(par.shape[2:])
        st = status.detach().float().to(dev)
        noise = FN.token_noise(list(keys), (K, D) + shp, int(seed), int(epoch), BANK_STREAM, device=dev)
        tea = {"det": (self.det, bev_det, ok_det), "map": (self.map, bev_map, ok_map)}
        ctxs, draws = {}, []
        for t in self.sources:
            net, bev, _ = tea[t]
            with self._autocast():
                ctxs[t] = net.scene(bev.to(dev).float())
                d = net.sample_draws(par, ctxs[t], st, noise, steps)
            draws.append(d.float().reshape(B, K * D, *shp))
        raw = torch.cat(draws, 1)
        fin = torch.isfinite(raw).flatten(2).all(-1)
        base = par[:, :, None].expand(-1, -1, D, -1, -1).reshape(B, K * D, *shp).repeat(1, len(self.sources), 1, 1)
        bank = torch.where(fin[..., None, None], torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), base)
        ok = fin
        prob_t = {}
        for t in self.sources:
            net, _, ok_tok = tea[t]
            with self._autocast():
                z = net.score(bank, ctxs[t], st)
            z = z.float()                      # raw (uncalibrated): see the bank docstring
            prob_t[t] = torch.sigmoid(z)
            ok = ok & ok_tok.reshape(-1).bool().to(dev)[:, None] & torch.isfinite(z).all(-1)
        prob = torch.stack([torch.stack([prob_t[t][..., i] for t in src], 0).mean(0)
                            for i, src in enumerate(self.bank_sources)], -1)
        return bank, prob.float(), ok


# ----------------------------------------------------------------------------------------------- offline bank
class FPDBankStore:
    """The FPD teachers' offline banks (tools/ck/fpd/train_fpd_teacher.py --stage bank + label_fpd_bank.py):
    <teacher run>/bank/<split>/{tokens.parquet, traj.npy [N,32,8,3], labels.npy [N,32,9], ok.npy [N,32],
    label_state.npy [N]} per source teacher, memory-mapped lazily; rows are looked up by TOKEN (tokens.parquet).
    lookup(tokens) -> bank [B, 32 * n_sources, 8, 3] (non-finite / unlabelled slots: zeros, not ok), y [B, M, 5] (the
    official labels as CK targets, ep_target), ok [B, M] (labelled row & label ok & finite trajectory & finite y)."""

    def __init__(self, runs: Dict[str, str], split: str, ep_target: str = "official"):
        self.runs = dict(runs)
        self.split = str(split)
        self.ep_target = ep_target
        self._d: Optional[Dict[str, Dict[str, Any]]] = None

    def _open(self) -> Dict[str, Dict[str, Any]]:
        if self._d is None:
            import pandas as pd

            d = {}
            for t, run in self.runs.items():
                bd = Path(run) / "bank" / self.split
                need = [bd / f for f in ("tokens.parquet", "traj.npy", "labels.npy", "ok.npy", "label_state.npy",
                                         "meta.json")]
                miss = [str(p) for p in need if not p.is_file()]
                if miss:
                    raise RuntimeError(f"FPD teacher {t} bank {bd}: missing {miss} (fpd.mt_source 'bank' needs the "
                                       f"finished teacher pipeline: bash scripts/experiment.sh teachers ARCH=fpd)")
                toks = pd.read_parquet(bd / "tokens.parquet", columns=["token"]).token.astype(str).tolist()
                d[t] = {"row": {tok: i for i, tok in enumerate(toks)},
                        "traj": np.load(bd / "traj.npy", mmap_mode="r"), "labels": np.load(bd / "labels.npy",
                                                                                            mmap_mode="r"),
                        "ok": np.load(bd / "ok.npy", mmap_mode="r"), "state": np.load(bd / "label_state.npy",
                                                                                     mmap_mode="r")}
            self._d = d
        return self._d

    def lookup(self, tokens: Sequence[str], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from .ep_target import ck_targets

        d = self._open()
        trajs, ys, oks = [], [], []
        for t, a in d.items():
            K = a["traj"].shape[1]
            tr = np.zeros((len(tokens), K, 8, 3), np.float32)
            lb = np.full((len(tokens), K, a["labels"].shape[-1]), np.nan, np.float32)
            ok = np.zeros((len(tokens), K), bool)
            for i, tok in enumerate(tokens):
                r = a["row"].get(str(tok))
                if r is None or int(a["state"][r]) != 1:
                    continue
                tr[i] = a["traj"][r]
                lb[i] = a["labels"][r]
                ok[i] = a["ok"][r]
            y = ck_targets(lb, self.ep_target).astype(np.float32)
            fin = np.isfinite(tr).all((-1, -2)) & np.isfinite(y).all(-1)
            ok &= fin
            trajs.append(np.where(ok[..., None, None], np.nan_to_num(tr), 0.0).astype(np.float32))
            ys.append(np.where(ok[..., None], np.nan_to_num(y), 0.0).astype(np.float32))
            oks.append(ok)
        cat = lambda xs: torch.from_numpy(np.concatenate(xs, 1)).to(device)
        return cat(trajs), cat(ys), cat(oks)


# ----------------------------------------------------------------------------------------------- selection
def _infer_select(res: Dict[str, torch.Tensor], sel: Dict[str, Any], dev) -> None:
    """CKE2E2.infer's optional selection (same code): ck2_sel_idx [B], ck2_traj [B, 8, 3] into res.  The type ids of
    select2.VARIANT_SETS are c % 6 here too (v 0 = the parent, v 1..5 = DiT draw v - 1)."""
    from . import select2 as S2

    rule = sel.get("rule", "pdms" if sel["beta"] == 1.0 else "wote")
    if rule == "pdms":     # plain expected PDMS of the critic (beta / v2 score / im unused)
        idx = S2.select96_pdms(res["ck2_score_logit"], res["ck2_valid96"], types=S2.VARIANT_SETS[sel["set"]])
    else:
        w_sel = "default" if sel.get("rule") == "wote" else "plugin"
        idx = S2.select96(res["ck2_v2_final"], res["ck2_v2_im"], torch.sigmoid(res["ck2_score_logit"]),
                          res["ck2_valid96"], sel["beta"], types=S2.VARIANT_SETS[sel["set"]], w=w_sel)
    idx_t = torch.as_tensor(np.asarray(idx.cpu() if torch.is_tensor(idx) else idx), dtype=torch.long, device=dev)
    traj, _src = S2.final_traj(res["ck2_cand96"], res["ck2_lat_traj"], idx_t, sel["lat_mode"])
    res["ck2_sel_idx"] = idx_t
    res["ck2_traj"] = torch.as_tensor(np.asarray(traj.cpu() if torch.is_tensor(traj) else traj), dtype=torch.float32,
                                      device=dev)


# ----------------------------------------------------------------------------------------------- main helper
class FPDE2E(CKE2E2):
    """CKE2E2 with the FPD-Drive refiner (module docstring).  Inherited unchanged: phase / r / recording, state_dict,
    labels / sampler / rowmap / tokens_of / refresh_labels, warmup_batch, the online KD calibration steps, _bev_kd_term,
    _check_rows / _check_inputs, label_source_epoch_*, the callback (make_ck2_callback) and ck2_parameters.
    infer_tokens: the scene tokens of the next infer() batch (eval_epdms sets it before agent(features)); None ->
    deterministic keys from the status bytes."""

    needs_tokens = True

    def __init__(self, cfg: CKE2E2Config, max_epochs: int = 30):
        super().__init__(cfg, max_epochs)
        if self.cfg.arch != "fpd":
            raise ValueError(f"FPDE2E needs ck_e2e2.arch 'fpd', got {self.cfg.arch!r}")
        self.infer_tokens: Optional[List[str]] = None

    # -------------------------------------------------------------------------------------- lazy pieces
    def teachers(self, device) -> FPDTeachers:
        if not self.cfg.use_teachers:
            raise RuntimeError("ck_e2e2.use_teachers is false: no refiner teachers")
        if self._teachers is None:
            c = self.cfg
            f = c.fpd
            kc = c.kd_calib
            file_mode = kc["enabled"] and kc["mode"] == "file"
            self._teachers = FPDTeachers(f["teacher_det_run"], f["teacher_map_run"], f["teacher_which"], device,
                                         c.teacher_amp, c.teacher_require_done,
                                         ep_target=c.ep_target if c.teacher_ep_check else None,
                                         calib=({"det": kc["det"], "map": kc["map"]} if file_mode else None),
                                         kd_keys=c.kd_score_keys, sources=c.kd_sources)
            self._calib_version_applied = None
            if self._teachers.calib_info:
                print("[ck_e2e2] KD calibration (FPD teachers): " + "; ".join(
                    f"{t} {i['path']} (sha16 {i['sha16']}, ckpt {i['ckpt_sha16']}, keys {i['fitted_keys']})"
                    for t, i in self._teachers.calib_info.items()), flush=True)
        tp = self._teachers.to(device)
        co = self.calib_online
        if co is not None and self._calib_version_applied != co.version:
            from .kd_calib import CALIB_TEACHERS
            tp.set_calib_arrays({t: co.params(t) for t in CALIB_TEACHERS})
            self._calib_version_applied = co.version
        return tp

    # -------------------------------------------------------------------------------------- step candidates
    def label_set(self, rows: torch.Tensor, toks: Sequence[str], targets, device) -> Dict[str, torch.Tensor]:
        """CKE2E2.label_set; fallback rows (G_src 0: no generation yet) keep only their 16 raw near + mid anchors (G_ok
        False on every rule-variant slot, columns 16:)."""
        G = super().label_set(rows, toks, targets, device)
        fb = G["G_src"].reshape(-1).long() == SRC_FALLBACK
        if bool(fb.any()):
            ok = G["G_ok"].clone()
            ok[fb, N_NEAR_MID:] = False
            G["G_ok"] = ok
        return G

    def current_batch(self, *a, **k):
        raise NotImplementedError("ck_e2e2.arch fpd: the on-policy candidates are the DiT pool "
                                  "(FPDE2E.onpolicy_batch), not CK2's rule variants")

    def draws(self, student, parents: torch.Tensor, ctx: torch.Tensor, status: torch.Tensor, keys: Sequence[str],
              epoch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """the student's pool: noise token_noise(keys, (16, n_draws, 8, 3), seed, epoch, 'fpd.noise') -> (draws
        [B,16,5,8,3], traj96 [B,96,8,3] (pool96: c = 6 k + v, v 0 = the parent bitwise), valid96 = finite [B,96]);
        no gradient.  Always fp32 with autocast OFF (training = evaluation numerics): a no-grad DiT pass under an
        enabled autocast fills autocast's weight-cast cache with casts WITHOUT autograd history, and inside one
        enclosing autocast region (Lightning 16-mixed + ck_e2e2.amp) the later dit_loss would reuse them -> no
        gradient reaches the DiT (DDP: unused parameters)."""
        f = self.cfg.fpd
        noise = FN.token_noise(list(keys), (parents.shape[1], int(f["n_draws"])) + tuple(parents.shape[2:]),
                               int(self.cfg.seed), int(epoch), FN.NOISE_STREAM, device=parents.device)
        with torch.autocast(device_type=parents.device.type, enabled=False):
            d = student.sample_draws(parents, ctx, status, noise, steps=int(f["steps"]))
        traj96 = FN.pool96(parents, d)
        return d, traj96, torch.isfinite(traj96).flatten(2).all(-1)

    def onpolicy_batch(self, traj96: torch.Tensor, valid96: torch.Tensor, rows: torch.Tensor, toks: Sequence[str],
                       targets, fallback: torch.Tensor) -> Dict[str, Any]:
        """KD set = the 96 pool columns of this step (non-finite -> v2's top-1, cand_ok False), BCE set = label_set
        (the lagged generation's 16 identities + 80 draws; fallback rows: 16 anchors), both sanitised (not-ok slots ->
        v2's top-1)."""
        dev = fallback.device
        cand, fin = _sanitize_traj(traj96, fallback, valid96)
        G = self.label_set(rows, toks, targets, dev)
        G_traj, G_fin = _sanitize_traj(G["G_traj"].float(), fallback, G["G_ok"].bool())
        B, Kc = cand.shape[:2]
        return {"cand": cand, "cand_ok": valid96.bool() & fin, "extra": G_traj, "y": G["G_y"].float(),
                "ok_lab": G["G_ok"].bool() & G_fin, "vtype_lab": G["G_vtype"].long(),
                "sur_pos": torch.arange(K16, device=dev)[None].expand(B, -1),
                "sur_tok_ok": torch.ones(B, dtype=torch.bool, device=dev),
                "lat_mask": torch.ones(B, Kc, dtype=torch.bool, device=dev),
                "kd_tok_ok": torch.ones(B, dtype=torch.bool, device=dev), "src": G["G_src"].long(),
                "var_valid_frac": float(valid96.reshape(B, K16, NV)[:, :, 1:].float().mean())}

    def bank_store(self) -> FPDBankStore:
        """fpd.mt_source 'bank': the source teachers' offline banks of the training split (the basename of
        warmup.packed, i.e. navtrain_train), lazily opened."""
        if getattr(self, "_bank_store", None) is None:
            f = self.cfg.fpd
            runs = {t: f[f"teacher_{t}_run"] for t in KD_SOURCES if t in self.cfg.kd_sources}
            split = Path(str(self.cfg.warmup["packed"])).name
            self._bank_store = FPDBankStore(runs, split, self.cfg.ep_target)
        return self._bank_store

    def dit_targets(self, human: torch.Tensor, bank: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """DiT targets [B, J, 8, 3] + mask [B, J]: the GT trajectory (fpd.use_gt_target; finite) first, then up to
        mt_k - 1 (mt_k without the GT) teacher targets select_targets(bank).  Masked slots are zeros (finite)."""
        f = self.cfg.fpd
        B = human.shape[0]
        dev = human.device
        tg, tm = [], []
        n_gt = 0
        if f["use_gt_target"]:
            h = human.detach().float()
            ok = torch.isfinite(h).flatten(1).all(-1)
            tg.append(h[:, None])
            tm.append(ok[:, None])
            n_gt = int(ok.sum())
        k_t = int(f["mt_k"]) - (1 if f["use_gt_target"] else 0)
        n_t = 0
        if bank is not None and k_t > 0:
            thr = float(f["mt_label_thr"] if f["mt_source"] == "bank" else f["mt_thr"])
            t, m = FN.select_targets(bank[0], bank[1], bank[2], k_max=k_t, min_ade=float(f["mt_min_ade"]), thr=thr)
            tg.append(t.to(dev).float())
            tm.append(m.to(dev).bool())
            n_t = int(m.sum())
        if not tg:              # no target at all (validate refuses it): one masked slot keeps the DiT in the graph
            tg.append(torch.zeros(B, 1, *human.shape[1:], dtype=torch.float32, device=dev))
            tm.append(torch.zeros(B, 1, dtype=torch.bool, device=dev))
        tgt = torch.cat(tg, 1)
        tmask = torch.cat(tm, 1)
        tgt = torch.where(tmask[..., None, None], torch.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0),
                          torch.zeros_like(tgt))
        return tgt, tmask, {"n_gt": float(n_gt), "n_teacher": float(n_t)}

    # -------------------------------------------------------------------------------------- inference
    @torch.no_grad()
    def infer(self, student, features, predictions) -> Dict[str, torch.Tensor]:
        """eval-mode outputs over the 96-column pool (fp32): CK2's keys; ck2_cand96[:, ::6] = v2 top-16 bitwise,
        [:, 0] = v2's trajectory; evaluation noise (epoch -1) keyed by infer_tokens (a list of the B scene tokens,
        consumed and reset to None) or, without it, by the status bytes.  predictions['trajectory'] is untouched."""
        bev = predictions["bev_embed"]
        dev = bev.device
        B = int(bev.shape[0])
        top = student_topk(predictions, K16)
        status = features["status_feature"].float().to(dev)
        toks, self.infer_tokens = self.infer_tokens, None
        keys = ([str(t) for t in toks] if isinstance(toks, (list, tuple)) and len(toks) == B
                else FN.status_keys(status))
        with torch.autocast(device_type=dev.type, enabled=False):
            ctx = student.scene(bev_sgrid(bev.float()))
            _, traj96, valid96 = self.draws(student, top["cand"], ctx, status, keys, epoch=-1)
            cand, fin = _sanitize_traj(traj96, top["cand"][:, 0])
            logit = student.score(cand, ctx, status).float()
        z6 = torch.zeros(B, traj96.shape[1], N_LAT, dtype=torch.float32, device=dev)
        res = {"ck2_cand_idx": top["idx"], "ck2_v2_final": top["final"], "ck2_v2_im": top["im"],
               "ck2_v2_sim": top["sim"], "ck2_cand96": traj96, "ck2_valid96": valid96 & fin,
               "ck2_score_logit": logit, "ck2_w_lat": z6, "ck2_e_lat": z6.clone(), "ck2_lat_traj": cand.float()}
        if self.cfg.infer_select is not None:
            _infer_select(res, self.cfg.infer_select, dev)
        return res

    # -------------------------------------------------------------------------------------- training loss
    def loss(self, student, features, targets, predictions, v2_loss: Optional[torch.Tensor] = None,
             v2_logs: Optional[Dict] = None, bev_kd=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        from . import losses as L

        t0 = time.time()
        c = self.cfg
        f = c.fpd
        bev = predictions["bev_embed"]
        dev = bev.device
        B = bev.shape[0]
        phase = self.phase(self.epoch)
        r = self.r(self.epoch_frac)
        status = features["status_feature"].float().to(dev)
        rows = (targets["ck_row"].reshape(-1).long() if "ck_row" in targets
                else torch.full((B,), -1, dtype=torch.long, device=dev))
        self._check_rows(rows)
        toks = self.tokens_of(rows)
        human = targets["trajectory"].float().to(dev)
        fallback = student_topk(predictions, 1)["cand"][:, 0]
        logs: Dict[str, float] = {}
        amp = torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=bool(c.amp and dev.type == "cuda"))

        # ---- scene tokens (gradient kept), the pool of this step (draws only when recorded / on-policy)
        tv = time.time()
        v0v = v0_from_status(status)                    # float64 hypot (recorder v0, as CK2)
        top = student_topk(predictions, K16)
        parents = top["cand"]
        with amp:
            ctx = student.scene(bev_sgrid(bev))
        rec_now = self.recording(self.epoch) and phase != "warmup"
        draw_dev = 0.0
        if phase == "onpolicy" or rec_now:             # fp32, autocast off (FPDE2E.draws)
            d, traj96, valid96 = self.draws(student, parents, ctx, status, toks, self.epoch)
            dd = FN.mean_xy_dist(d.float(), parents.float()[:, :, None])           # [B, 16, 5]
            okd = torch.isfinite(dd)
            draw_dev = float(dd.nan_to_num(0.0).sum() / okd.sum().clamp(min=1))
        if phase == "onpolicy":
            S = self.onpolicy_batch(traj96, valid96, rows, toks, targets, fallback)
        else:
            S = self.warmup_batch(targets, toks, fallback, dev)
        t_var = time.time() - tv
        n_rec = 0
        if rec_now and self.recorder is not None:
            n_rec = self.recorder.add(rows, traj96, valid96, top["idx"], top["final"], v0v, self.gstep)
            self.cum["rec_tokens"] += n_rec

        # ---- teachers (online, every phase): KD on the KD set; the bank for the multi-target distillation
        tt = time.time()
        co = self.calib_online
        T, BK = None, None
        par_s, _ = _sanitize_traj(parents, fallback)    # finite parents for the bank and the DiT loss
        k_teacher = int(f["mt_k"]) - (1 if f["use_gt_target"] else 0)
        if c.use_teachers:
            if co is not None:               # online KD calibration: refit (new slot) BEFORE the teachers run
                self._calib_online_step(dev, logs)
            tp = self.teachers(dev)
            T = tp.run(targets["kd_bev_0"], targets["kd_ok_0"], targets["kd_bev_1"], targets["kd_ok_1"], S["cand"],
                       status, cand_ok=S["cand_ok"], extra=(S["extra"] if co is not None else None))
            if co is not None:
                logs["ck2/cal_pairs"] = float(self._calib_online_add(S, T, phase))
            if k_teacher > 0 and f["mt_source"] == "online":   # each teacher samples with its OWN DDIM steps
                BK = tp.bank(targets["kd_bev_0"], targets["kd_ok_0"], targets["kd_bev_1"], targets["kd_ok_1"],
                             par_s, status, toks, int(f["mt_draws"]), seed=int(c.seed), epoch=int(self.epoch))
            elif k_teacher > 0:        # 'bank': the offline teacher banks with official labels (no sampling here)
                BK = self.bank_store().lookup(list(toks), dev)
        t_teacher = time.time() - tt

        # ---- student: two scorer passes (the scorer attends across its set) + the DiT losses
        ts = time.time()
        tgt, tmask, tstat = self.dit_targets(human, BK)
        with amp:
            s_kd = student.score(S["cand"], ctx, status).float()
            s_lab = s_kd if S["extra"] is None else student.score(S["extra"], ctx, status).float()
            l_diff, l_traj, dstats = FN.dit_loss(student, par_s, tgt, tmask, ctx, status, toks, int(c.seed),
                                                 int(self.epoch), int(self.mb))
        l_diff, l_traj = l_diff.float(), l_traj.float()
        Kc = S["cand"].shape[1]

        # ---- loss
        l_bce, per = L.score_bce(s_lab, S["y"], S["ok_lab"])
        inner = c.lambda_score * l_bce
        ref_gt_ok = (targets["ref_gt_ok"].reshape(-1).bool().to(dev) if "ref_gt_ok" in targets
                     else torch.zeros(B, dtype=torch.bool, device=dev))
        if T is not None:
            kd_ok = T["kd_ok"].bool() & S["kd_tok_ok"][:, None]
            ki = [CK_KEYS.index(k) for k in c.kd_score_keys]
            l_ks = L.score_kd_bce(s_kd[..., ki], T["kd_score_prob"][..., ki], kd_ok)
            inner = inner + (r * c.lambda_kd_score) * l_ks
        else:
            kd_ok = torch.zeros(S["cand_ok"].shape, dtype=torch.bool, device=dev)
            l_ks = bev.new_zeros((), dtype=torch.float32)
        inner = inner + float(f["lambda_diff"]) * l_diff + float(f["lambda_traj"]) * l_traj
        self._check_inputs(ref_gt_ok, kd_ok if T is not None else None)
        loss = c.loss_weight * inner
        loss = loss + 0.0 * (s_kd.sum() + s_lab.sum())
        loss_val = _f(loss)
        bad_loss = not math.isfinite(loss_val)
        if bad_loss:
            self.cum["nonfinite_loss"] += 1
            loss = 0.0 * sum(p.sum() for p in student.parameters() if p.requires_grad)
        t_student = time.time() - ts

        # ---- BEV-KD arm (outside the balancer; returned inside the CK loss)
        term, term_t = None, {}
        if bev_kd is not None:
            term, term_t = self._bev_kd_term(bev_kd, bev, targets, v2_logs, logs)

        # ---- grad share (every grad_share_every micro-batches; rank-local, no collective)
        every = int(c.grad_share_every)
        if (every > 0 and self.mb % every == 0 and v2_loss is not None and bev.requires_grad
                and torch.is_grad_enabled() and not bad_loss):
            parts = (("v2", v2_loss), ("ck", loss)) + ((("bevkd", term),) if term is not None else ())
            gn = {}
            for k, tl in parts:
                gg = (torch.autograd.grad(tl, bev, retain_graph=True, allow_unused=True)[0]
                      if isinstance(tl, torch.Tensor) and tl.requires_grad else None)
                gn[k] = 0.0 if gg is None else float(gg.float().norm())
                logs[f"gnorm/bev_{k}"] = gn[k]
            logs["ck2/gshare"] = gn["ck"] / max(sum(gn.values()), 1e-12)
            for t, tt_ in term_t.items():
                gg = (torch.autograd.grad(tt_, bev, retain_graph=True, allow_unused=True)[0]
                      if isinstance(tt_, torch.Tensor) and tt_.requires_grad else None)
                logs[f"gnorm/bev_bevkd_{t}"] = 0.0 if gg is None else float(gg.float().norm())

        # ---- logs (the CK2 keys; the surrogate / lateral ones are 0) + fpd/*
        lw = c.loss_weight
        l_diff_v, l_traj_v = _f(l_diff), _f(l_traj)
        logs.update({
            "ck2/loss": loss_val if not bad_loss else 0.0, "ck2/loss_nonfinite": float(bad_loss),
            "ck2/bce": _f(l_bce), "ck2/kd_score": _f(l_ks), "ck2/sur": 0.0, "ck2/sur_nonfinite": 0.0,
            "ck2/lat_kd": 0.0, "ck2/w_lat": 0.0, "ck2/r": r, "ck2/phase": float(PHASE2_CODE[phase]),
            "ck2/mb": float(self.mb), "ck2/n_gt": tstat["n_gt"],
            "ck2/wt_bce": lw * c.lambda_score * _f(l_bce), "ck2/wt_sur": 0.0,
            "ck2/wt_kd_score": lw * r * c.lambda_kd_score * _f(l_ks), "ck2/wt_lat_kd": 0.0,
            "fpd/l_diff": l_diff_v, "fpd/l_traj": l_traj_v,
            "fpd/wt_diff": lw * float(f["lambda_diff"]) * l_diff_v,
            "fpd/wt_traj": lw * float(f["lambda_traj"]) * l_traj_v,
            "fpd/n_targets": float(tmask.sum()) / max(B, 1), "fpd/n_teacher_targets": tstat["n_teacher"] / max(B, 1),
            "fpd/draw_dev": draw_dev,
        })
        logs.update({k: float(v) for k, v in dstats.items()})            # fpd/n_pairs, fpd/res_abs
        if BK is not None:
            bank_ok = BK[2]
            safe = bank_ok & (BK[1][..., 0] >= float(f["mt_thr"])) & (BK[1][..., 1] >= float(f["mt_thr"]))
            logs["fpd/bank_safe_frac"] = float(safe.sum()) / max(int(bank_ok.sum()), 1)
            logs["fpd/bank_ok_frac"] = float(bank_ok.float().mean())
        else:
            logs["fpd/bank_safe_frac"] = 0.0
            logs["fpd/bank_ok_frac"] = 0.0
        for k, v in (per or {}).items():
            logs[f"ck2/bce_{k}"] = float(v)
        for k in SUR_TERMS + ("P1_minus_P0",):
            logs[f"ck2/t_{k}"] = 0.0
        if v2_loss is not None:
            logs["ck2/ratio_v2"] = (loss_val / max(abs(_f(v2_loss)), 1e-12)) if not bad_loss else 0.0
        with torch.no_grad():
            vt = S["vtype_lab"].to(dev)
            okl = S["ok_lab"]
            for name, m in (("orig", vt == 0), ("var", vt > 0)):
                lb, _ = L.score_bce(s_lab.detach(), S["y"], okl & m)
                logs[f"ck2/bce_{name}"] = float(lb)
            logs["ck2/n_var_ok"] = float((okl & (vt > 0)).sum())
            logs["ck2/G_ok_frac"] = float(okl.float().mean()) if okl.numel() else 0.0
            logs["ck2/kd_ok_frac"] = float(kd_ok.float().mean())
            logs["ck2/var_valid_frac"] = float(S["var_valid_frac"])
            srcn = S["src"].reshape(-1).cpu()
            if phase == "onpolicy":
                for code, name in enumerate(SRC_NAMES):
                    cnt = int((srcn == code).sum())
                    logs[f"ck2/G_src_{name}"] = cnt / max(B, 1)
                    self.cum[f"G_src_{name}"] += cnt
            logs["ck2/e_abs"] = 0.0
            if T is not None:
                for k in ("ck2/e_abs_T", "ck2/e_abs_det", "ck2/e_abs_map", "ck2/lat_disagree"):
                    logs[k] = 0.0
            logs["ck2/n_cand"] = float(Kc + (0 if S["extra"] is None else S["extra"].shape[1]))
        logs["ck2/variants_ms"] = 1e3 * t_var
        logs["ck2/teacher_ms"] = 1e3 * t_teacher
        logs["ck2/student_ms"] = 1e3 * t_student
        logs["ck2/rec_tokens"] = float(n_rec)
        logs["ck2/total_ms"] = 1e3 * (time.time() - t0)
        if term is not None:
            loss = loss + term
        self.mb += 1
        return loss, {k: torch.tensor(float(v), device=dev) for k, v in logs.items()}


__all__ = ["BANK_STREAM", "build_student_fpd", "FPDTeachers", "FPDE2E"]
