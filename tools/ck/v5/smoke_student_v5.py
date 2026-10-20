#!/usr/bin/env python
"""refiner v5 student smoke on CPU (not a training run): the real v2 ParaSSRAgent with the fusion v13 BEV-selector
distillation + the CK2 e2e module with the v5 block (relation 'student', unknown evidence, corrector A_S, Z-KD to the v5
teachers), through pytorch_lightning on the 2-token CK2 fixture of tools/ck/e2e2/cpu_dryrun.py (+ the DET teacher's
object-pack rows, which that fixture predates), batch 1, accumulate 2: epoch 0 'warmup' (+ epoch 1 'warmup_record'),
then one eval-mode forward (ck2_* inference incl. ck2_v5_traj).  Teachers = the v5 runs' latest ckpt_last.pt
(teacher_require_done false: they may still be training).

  cd /workspace/yongjae/SSR-v5 && env PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
    /venv/ssr/bin/python tools/ck/v5/smoke_student_v5.py [--zkd 0|1] [--epochs 2] [--tag T]

Checks: finite ck2/* and v5/* logs, no non-finite loss, v5 terms present (rel_a_bce, rel_h_bce, ref_loss, zkd_det /
zkd_map with --zkd 1), every v5 parameter moved, the inference returns ck2_v5_traj [B, 8, 3] finite and ck2_v5_src.
Det-head fixes of arm B / D (fusion_v13_v5.extra: V1b det_vel_moving_only, V1c det_stop_branch, the ACC aux head
det_acc_head; v5_arm.set: rel_pstop): the fixture gets the ACC targets gt_kf_speed / gt_kf_present from the real scenes
(ParaSSRTargetBuilder; its recomputed gt_boxes must equal the fixture's), every det loss call of the fit carries finite
loss_stop* / loss_acc / loss_stop2s, every stop_branches / acc_head parameter moved.  The fixture's det head is
untrained (no query reaches score 0.3), so a probe hooks 4 confident in-window detections into one train-mode forward:
the student rows are valid, their p_stop feature = sigmoid(stop logit), the relation hazard's gradient reaches
stop_branches[-1] (rel_det_grad, design V2 + V1c); and one eval-mode forward with the same detections returns a finite
ck2_v5_traj.
--roundtrip 1 (Stage 2 credit, report 99 N3 / N4; v5_arm.set's rel_det_grad / record_tau1 / lambda_awr): epochs 0-1 ->
labeler2 (official, gen 1) -> resume epoch 2 'onpolicy' (records the 96 + tau1 of the deployment-rule pick) -> labeler2
(gen 2, the tau1 edit slot labelled) -> resume epoch 3 'onpolicy' (label set + the tau1 column from gen 2, AWR of A_S):
checks v5/rec_tau1_valid_frac > 0 in epoch 2, gen 2 has labelled edit slots, v5/awr finite with awr_n > 0 in epoch 3.
The roundtrip runs the 'rel' group of v5_arm.set too (rel_modes cv+fut, top_feats v2 with the Delta r probes, L_rel /
L_rel,cv, rel_det_grad_matched): GT-aligned detections are hooked into the det head for the whole fit (gt_dets: up to 3
in-window GT agents of each fixture token, 0.3 m off, + 1 ghost >= 6 m from every GT object; the position entries keep
the head's graph), so the relation path trains end to end on valid rows; checks v5/rel_bce / rel_cv_bce finite with
rel_n_pairs > 0 and rel_match_frac > 0, relation.cv_prior moved, and a D3 probe: the relation hazard's gradient reaches
the matched queries' box code only (the ghost's is 0).
The roundtrip also runs the 'zkd' group of v5_arm.set (ZKD_GROUP: z_adapt with the teacher statistics, lambda_edit):
the statistics records are fitted on the fly by tools/ck/v5/fit_z_stats.py (--zstats-tokens, default 8: DRY-RUN records,
accepted by the student build, refused by a strict launch) into <tag>/z_stats_{det,map}.json unless --zstats-det /
--zstats-map give files; checks every z_adapter parameter moved, v5/edit_bce finite with edit_n_pairs > 0, and with
--zkd 1 the per-channel Z-KD magnitudes v5/zkd_<ch> (teacher-normalised units) next to the raw-unit v5/zkd_raw_det /
_map (result.json 'zkd_group' records their means: the scale under the unchanged lambda_zkd_* = 0.5).
Output: CK/v5/smoke_student/<tag>/ (result.json, steps.jsonl).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from tools.ck.e2e2 import cpu_dryrun as CD  # noqa: E402

CK = Path("/home/external-user/ssd/yongjae_refiner/ck")
T_DET = CK / "v5/train/v5T_rel"          # --teacher-det / --teacher-map override (any v5 run with ckpt_last.pt)
T_MAP = CK / "v5/train/v5M_unk"
V13 = dict(use_distill=True, distill_selector_v8=False, distill_selector_v9=False, distill_selector_v10=False,
           distill_selector_v11=False, distill_selector_v12=False, distill_selector_v13=True,
           distill_teacher_source="targets", distill_feature_root="/workspace/teacher_cache", distill_lambda_gt=0.45,
           distill_lambda_cov=0.20, distill_lambda_risk=0.35, distill_tau_gt=4.5, distill_tau_risk=8.0,
           distill_probe_warmup_steps=2, distill_probe_ramp_steps=2, distill_probe_loss_weight=0.5,
           distill_probe_lr_mult=3.0, distill_proj_lr_mult=2.0, distill_probe_use_ema=True,
           distill_probe_ema_decay=0.99)
V5_AGENT = dict(det_vel_code_weights=[1.05, 0.41], expose_det_outputs=True,          # = fusion_v13_v5.extra
                det_vel_moving_only=True, det_stop_branch=True, det_acc_head=True)
DET_V5_LOSS = ("loss_stop", "loss_acc", "loss_stop2s")      # the det-head fixes' loss keys (+ loss_stop_d<l>)


EXEC_MODEL = "ref"          # the student's V4Cfg.exec_model (--exec-model; v5_arm.set: ref = no-LQR student, 2026-10-20)


def v5_ck2(io_dir: Path, zkd: bool, stage2: bool = False, zstats=None) -> dict:
    d = CD.ck2_dict(io_dir, "main")
    d.update(teacher_det_run=str(T_DET), teacher_map_run=str(T_MAP), teacher_require_done=False,
             kd_score_keys=["nc", "dac"])
    d["bev_kd"] = {"enabled": False, "teachers": ["det"]}
    d["v4"] = {"readout": "range", "exec_model": EXEC_MODEL, "ev_tbins": 20, "ev_tbin_s": 0.25, "ep_evidence": True,
               "pref_head": True, "corridor_version": 2, "rg_read": "any4", "rg_stopped_aware": True, "edits": "off",
               "corrector": False, "kd_parent_only": True, "kd_score_scope": "all",
               "kd_constraint": {"enabled": False}, "rg_gt_aux": {"enabled": True, "lambda": 0.5, "max_parents": 8},
               "v5": {"enabled": True, "relation": "student", "unknown_ev": True, "corrector": True,
                      "z_kd": bool(zkd), "rel_pstop": True}}
    if stage2:
        d["v4"]["v5"].update(rel_det_grad=True, record_tau1=True, lambda_awr=0.1)
        d["v4"]["v5"].update(REL_GROUP)
        if zstats is not None:                                   # the 'zkd' group (its statistics files)
            d["v4"]["v5"].update(ZKD_GROUP, z_stats_det=str(zstats["det"]), z_stats_map=str(zstats["map"]))
    return d


# the 'rel' group of v5_arm.set (D1-D5): futures as relation modes, top-3 'v2' + Delta r, L_rel / L_rel,cv, D3
REL_GROUP = dict(rel_modes="cv+fut", top_feats="v2", lambda_rel=0.5, lambda_rel_cv=0.25, rel_max_parents=8,
                 rel_det_grad_matched=True)
# the 'zkd' group of v5_arm.set (D1 (a) / (b), D2, D6-D8): the Z adapters on teacher-normalised reads, L_edit-Bernoulli
ZKD_GROUP = dict(z_adapt=True, lambda_edit=0.25)


def fit_zstats(root: Path, tokens: int, det: str = "", mp: str = "") -> dict:
    """the teachers' Z-adapter statistics for the smoke: given files, else tools/ck/v5/fit_z_stats.py on the fly
    (`tokens` navtrain tokens, CPU; a DRY-RUN record) into root/z_stats_{det,map}.json."""
    from tools.ck.v5 import fit_z_stats as FZ
    out = {}
    for name, run, given in (("det", T_DET, det), ("map", T_MAP, mp)):
        if given:
            out[name] = Path(given)
            continue
        p = root / f"z_stats_{name}.json"
        FZ.main(["--run", str(run), "--which", "last", "--tokens", str(int(tokens)), "--out", str(p),
                 "--zq-exec", EXEC_MODEL])
        out[name] = p
    return out


class GTDets:
    """hook GT-aligned detections into the det head's final layer (the fixture's head is untrained: no query reaches
    score 0.3): per fixture token up to 3 in-window GT agents (0.3 m off) + 1 ghost (>= 6 m from every GT object) in
    queries 0..3 of batch row 0 (class vehicle, car size; the position entries keep the head's graph: value replaced,
    gradient identity; velocities / futures = the head's own).  Tokens are told apart by their status feature (a
    forward pre-hook on the model)."""

    def __init__(self, ds):
        import numpy as np
        from navsim.agents.para_ssr.ck.v5 import refine_loss as RL
        from navsim.agents.para_ssr.refiner import gt_future as GF
        self.table, self.info = {}, []
        for i in range(len(ds)):
            f, t = ds[i]
            tg = {k: v[None] for k, v in t.items() if k.startswith("ref_") and torch.is_tensor(v)}
            gt = RL.gt_of(tg, torch.zeros(1, dtype=torch.long))
            c, _, obs = RL.gt_t0_objects(gt)
            c, obs = c[0], obs[0] & gt["valid"][0]
            ag = gt["meta"][0, :, GF.MT_AGENT].bool() & (gt["meta"][0, :, GF.MT_FIRST] == 0)
            x, y = c[:, 0], c[:, 1]
            inw = obs & ag & (x >= 3) & (x <= 28) & (y.abs() <= 12) & (torch.rad2deg(torch.atan2(y, x)).abs() <= 70)
            sel = c[inw][:3] + 0.3
            ghost = None
            for gx in np.arange(4.0, 30.0, 2.0):
                for gy in np.arange(-10.0, 10.5, 2.0):
                    p = torch.tensor([gx, gy], dtype=torch.float32)
                    if not bool(obs.any()) or float((c[obs] - p).norm(dim=-1).min()) >= 6.0:
                        ghost = p
                        break
                if ghost is not None:
                    break
            pts = torch.cat([sel] + ([ghost[None]] if ghost is not None else []), 0)
            det = torch.stack([-pts[:, 1], pts[:, 0]], -1)                     # N frame -> det (x_right, y_forward)
            self.table[f["status_feature"].float().numpy().tobytes()] = (det, int(sel.shape[0]))
            self.info.append({"n_gt_in_window": int(inw.sum()), "n_hooked_gt": int(sel.shape[0]),
                              "ghost": None if ghost is None else [float(v) for v in ghost]})
        self.key = None
        self.hits = 0
        self.last_code = None
        self.handles = []

    def register(self, agent) -> "GTDets":
        import math as _m
        model = agent.para_ssr_model
        head = model.det_motion_head

        def pre(mod, inp):
            feats = inp[0] if inp else None
            st = feats.get("status_feature") if isinstance(feats, dict) else None
            self.key = None if st is None else st[0].detach().float().cpu().numpy().tobytes()

        def hook(mod, inp, out):
            ent = self.table.get(self.key)
            if ent is None:
                return out
            det, _ = ent
            k = det.shape[0]
            out = dict(out)
            cls = out["all_cls_scores"].clone()
            cls[-1, 0, :k] = -6.0
            cls[-1, 0, :k, 0] = 4.0
            code = out["all_bbox_preds"].clone()
            cur = code[-1, 0, :k, 0:2]
            code[-1, 0, :k, 0:2] = cur + (det.to(code) - cur).detach()
            code[-1, 0, :k, 2:8] = torch.tensor([_m.log(1.9), _m.log(4.5), 0.0, _m.log(1.6), 0.0, 1.0]).to(code)
            out.update(all_cls_scores=cls, all_bbox_preds=code)
            self.hits += 1
            self.last_code = code
            return out
        self.handles = [model.register_forward_pre_hook(pre), head.register_forward_hook(hook)]
        return self

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def probe_matched_grad(agent, gd: GTDets) -> dict:
    """D3 (rel_det_grad_matched): one train-mode forward with the GT-aligned detections; the rows matched to a GT track
    at t0 keep the box code's graph, the ghost's is detached -> the relation hazard's gradient reaches the matched
    queries' code only."""
    from navsim.agents.para_ssr.ck.online2 import student_rel_match, student_rel_rows
    from navsim.agents.para_ssr.ck.v5.student_objects import rows_keep_graph
    from navsim.agents.para_ssr.ck.v4 import lqr as LQR
    ds = FixtureV5()
    f, t = ds[0]
    feats = {k: v[None] for k, v in f.items()}
    targets = {k: v[None] for k, v in t.items() if torch.is_tensor(v)}
    agent.train()
    torch.manual_seed(0)
    pred = agent.para_ssr_model(feats)
    code = gd.last_code
    code.retain_grad()
    v5 = dict(agent._ck_e2e2.cfg.v5)
    O = student_rel_rows(pred, v5)
    g = student_rel_match(O, targets, code.device)
    Om = rows_keep_graph(O, (g >= 0) & O["valid"])
    v = 6.0
    tt = torch.arange(1, 9, dtype=torch.float32) * 0.5
    res = {"match": g[0].tolist(), "q_idx": O["q_idx"][0].tolist()}
    zs = int(v5["z_stride"])
    st = LQR.simulate(torch.stack([v * tt, torch.zeros_like(tt), torch.zeros_like(tt)], -1)[None],
                      torch.tensor([[v, 0.0]]), dtype=torch.float32)
    t5 = torch.arange(LQR.NT, dtype=torch.float64)[::zs] * LQR.DT
    rel = agent.ck_student.relation(Om, st[:, ::zs], t5, torch.zeros(1, dtype=torch.long))
    (rel["H"].sum() + rel["lp"].sum()).backward()
    gq = code.grad[-1, 0].abs().sum(-1)
    q, gg = O["q_idx"][0], g[0]
    matched = [int(q[i]) for i in range(len(q)) if int(q[i]) >= 0 and int(gg[i]) >= 0]
    unmatched = [int(q[i]) for i in range(len(q)) if int(q[i]) >= 0 and int(gg[i]) < 0]
    res.update(matched_q=matched, unmatched_q=unmatched, grad_matched=[float(gq[j]) for j in matched],
               grad_unmatched=[float(gq[j]) for j in unmatched])
    agent.zero_grad(set_to_none=True)
    res["pass"] = (len(matched) >= 1 and len(unmatched) >= 1 and all(x == 0.0 for x in res["grad_unmatched"])
                   and any(x > 0 for x in res["grad_matched"]))
    return res


def edit_labels(io: Path, epoch: int) -> dict:
    """labelled edit slots of generation `epoch` (e2e_data2 edit_* arrays of the labelled rows)."""
    import numpy as np
    from navsim.agents.para_ssr.ck import e2e_data as ED
    from navsim.agents.para_ssr.ck import e2e_data2 as D2
    meta = json.loads((ED.gen_dir(str(io), epoch) / "meta.json").read_text())
    g = D2.open_generation2(str(io), epoch, int(meta["n_rows"]), mode="r", optional=D2.GEN2_EDIT_FILES)
    done = np.flatnonzero(np.asarray(g["row_state"]) == 1)
    if "edit_ok" not in g:
        return {"has_edit_arrays": False, "rows_labelled": int(len(done))}
    eo = np.asarray(g["edit_ok"])[done]
    lab = np.asarray(g["edit_labels"])[done]
    return {"has_edit_arrays": True, "rows_labelled": int(len(done)), "slot0_ok": int(eo[:, 0].sum()),
            "other_slots_ok": int(eo[:, 1:].sum()),
            "slot0_pdms": [float(x) for x in lab[:, 0, 6]] if len(done) else []}


def load_scenes(split: str, tokens) -> dict:
    """token -> navsim Scene (annotations only, no sensors) of `split` (tools/ck/data/common.split paths)."""
    from dataclasses import replace
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    from tools.ck.data import common as CM
    sp = CM.split(split)
    tdf = CM.split_tokens(split).set_index("token")
    tokens = [str(t) for t in tokens]
    need = sorted(set(tdf.loc[tokens, "log"]))
    sf = instantiate(OmegaConf.load(CM.resolve_path(sp["scene_filter"])))
    sf = replace(sf, tokens=tokens, max_scenes=None, log_names=sorted(set(sf.log_names or need) & set(need)))
    loader = SceneLoader(data_path=Path(sp["navsim_logs"]), sensor_blobs_path=Path(sp["sensor_blobs"]),
                         scene_filter=sf, sensor_config=SensorConfig.build_no_sensors())
    miss = sorted(set(tokens) - set(loader.tokens))
    if miss:
        raise RuntimeError(f"scene loader misses {len(miss)} of {len(tokens)} tokens, e.g. {miss[:3]}")
    return {t: loader.get_scene_from_token(t) for t in tokens}


def acc_targets(scenes: dict, tokens, ref: dict) -> dict:
    """the ACC head targets of the det slots (ParaSSRTargetBuilder with det_acc_head) for `tokens`, stacked; refuses
    when the recomputed slot boxes / labels / validity differ from the fixture's (`ref`: gt_boxes / gt_labels /
    gt_valid of the same tokens), so the targets are aligned with the fixture's det GT slot by slot."""
    from dataclasses import replace
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    from navsim.agents.para_ssr.para_ssr_targets import ParaSSRTargetBuilder
    b = ParaSSRTargetBuilder(replace(ParaSSRConfig(), det_acc_head=True),
                             TrajectorySampling(time_horizon=4, interval_length=0.5))
    out = {"gt_kf_speed": [], "gt_kf_present": []}
    for i, tok in enumerate(tokens):
        sc = scenes[str(tok)]
        tg = b._compute_agent_targets(sc, sc.scene_metadata.num_history_frames - 1)
        for k in ("gt_boxes", "gt_labels", "gt_valid"):
            if not torch.allclose(tg[k].float(), ref[k][i].float(), atol=1e-5, rtol=0):
                raise RuntimeError(f"{tok}: recomputed {k} differs from the fixture's (slot alignment broken)")
        for k in out:
            out[k].append(tg[k])
    return {k: torch.stack(v) for k, v in out.items()}


class FixtureV5(CD.FixtureDS):
    """the cpu_dryrun fixture + the v5 DET teacher's pack rows (e2e_data2.KDObjRows of its ck_row) + the ACC head's
    keyframe-speed targets of the real scenes (acc_targets; computed once per process)."""
    _acc = None

    def __init__(self):
        super().__init__()
        from navsim.agents.para_ssr.ck import e2e_data2 as D2
        from navsim.agents.para_ssr.ck.online2 import CKE2E2Config
        from navsim.agents.para_ssr.ck import e2e_data as ED
        packed = CKE2E2Config().warmup["packed"]
        kd = D2.KDObjRows(D2.teacher_v5_rel_cfg(str(T_DET)), packed)
        self.kd = [kd.item(int(r)) for r in self.t["ck_row"]]
        toks = ED.RowMap(packed).tokens                 # the v13 distillation needs targets['scene_token']
        self.tok = [str(toks[int(r)]) for r in self.t["ck_row"]]
        if FixtureV5._acc is None:
            FixtureV5._acc = acc_targets(load_scenes("navtrain_train", self.tok), self.tok, self.t)
        self.acc = FixtureV5._acc

    def __getitem__(self, i):
        f, t = super().__getitem__(i)
        t = dict(t, **self.kd[i], scene_token=self.tok[i], **{k: v[i] for k, v in self.acc.items()})
        return f, t


def watch_det_loss(agent, store: list) -> None:
    """record every det-head loss call's det-fix terms (finite or not) into `store` (the fit's det losses are not in
    steps.jsonl)."""
    head = agent.para_ssr_model.det_motion_head
    orig = head.loss

    def loss(*a, **kw):
        out = orig(*a, **kw)
        store.append({k: float(v.detach()) for k, v in out.items() if k.startswith(DET_V5_LOSS)})
        return out
    head.loss = loss


DET_V5_PREFIX = ("stop_branches.", "acc_head.")


def det_v5_params(agent) -> dict:
    return {n: p.detach().clone() for n, p in agent.para_ssr_model.det_motion_head.named_parameters()
            if n.startswith(DET_V5_PREFIX)}


def probe_valid_objects(agent) -> dict:
    """the fixture's det head is untrained (no query reaches score_thr 0.3): hook 4 confident in-window vehicles into
    its final layer (positions / classes replaced, velocities = the head's own deployed ones, graph kept) and check the
    student rows, their p_stop feature, the relation gradient into stop_branches[-1], and the eval-mode v5 output."""
    import math as _m
    from navsim.agents.para_ssr.ck.online2 import student_rel_rows
    from navsim.agents.para_ssr.ck.v4 import lqr as LQR
    head = agent.para_ssr_model.det_motion_head
    pos = torch.tensor([[-1.5, 8.0], [2.0, 14.0], [-4.0, 20.0], [5.0, 26.0]])

    def hook(mod, inp, out):
        out = dict(out)
        cls = out["all_cls_scores"].clone()
        cls[-1, :, :4] = -6.0
        cls[-1, :, :4, 0] = 4.0
        code = out["all_bbox_preds"].clone()
        code[-1, :, :4, 0:2] = pos.to(code)
        code[-1, :, :4, 2:8] = torch.tensor([_m.log(1.9), _m.log(4.5), 0.0, _m.log(1.6), 0.0, 1.0]).to(code)
        out.update(all_cls_scores=cls, all_bbox_preds=code)
        return out
    h = head.register_forward_hook(hook)
    res = {}
    try:
        ds = FixtureV5()
        f, _ = ds[0]
        feats = {k: v[None] for k, v in f.items()}
        agent.train()
        torch.manual_seed(0)
        pred = agent.para_ssr_model(feats)
        v5 = dict(agent._ck_e2e2.cfg.v5, rel_det_grad=True)         # the V2 path (v5_arm.set; Stage 2 here)
        O = student_rel_rows(pred, v5)
        sel = O["q_idx"][0]
        ok = sel >= 0
        p_stop = torch.sigmoid(pred["all_stop_logits"][-1][0].detach())[sel.clamp(min=0)]
        res["n_valid_rows"] = int(ok.sum())
        res["p_stop_feature_err"] = float((O["feats"][0, ok, -1] - p_stop[ok]).abs().max()) if bool(ok.any()) else None
        v = 6.0
        tt = torch.arange(1, 9, dtype=torch.float32) * 0.5
        st = LQR.simulate(torch.stack([v * tt, torch.zeros_like(tt), torch.zeros_like(tt)], -1)[None],
                          torch.tensor([[v, 0.0]]), dtype=torch.float32)
        zs = int(v5["z_stride"])
        t = torch.arange(LQR.NT, dtype=torch.float64)[::zs] * LQR.DT
        head.zero_grad(set_to_none=True)
        rel = agent.ck_student.relation(O, st[:, ::zs], t, torch.zeros(1, dtype=torch.long))
        rel["H"].sum().backward()
        g = head.stop_branches[-1].weight.grad
        res["H_max"] = float(rel["H"].detach().max())
        res["stop_branch_grad_abs"] = 0.0 if g is None else float(g.abs().sum())
        head.zero_grad(set_to_none=True)
        agent.eval()
        with torch.no_grad():
            pe = agent.forward(feats)
        tv = pe.get("ck2_v5_traj")
        res["eval_v5_traj_finite"] = tv is not None and bool(torch.isfinite(tv).all())
        res["eval_has_stop_logits"] = "all_stop_logits" in pe
    finally:
        h.remove()
    res["pass"] = (res["n_valid_rows"] >= 1 and res["p_stop_feature_err"] is not None
                   and res["p_stop_feature_err"] < 1e-6 and res["stop_branch_grad_abs"] > 0
                   and res["eval_v5_traj_finite"] and res["eval_has_stop_logits"])
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zkd", type=int, default=1)
    ap.add_argument("--exec-model", choices=("lqr", "ref"), default="ref", help="the student's exec model (v5_arm.set)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--tag", default="cpu")
    ap.add_argument("--roundtrip", type=int, default=0, help="1: the Stage-2 labeler round trip (module docstring)")
    ap.add_argument("--teacher-det", default="")
    ap.add_argument("--teacher-map", default="")
    ap.add_argument("--zstats-det", default="", help="roundtrip: the DET teacher's fit_z_stats record (else fitted)")
    ap.add_argument("--zstats-map", default="", help="roundtrip: the MAP teacher's fit_z_stats record (else fitted)")
    ap.add_argument("--zstats-tokens", type=int, default=8, help="tokens of the on-the-fly fit_z_stats dry run")
    a = ap.parse_args(argv)
    global EXEC_MODEL
    EXEC_MODEL = a.exec_model
    global T_DET, T_MAP
    T_DET = Path(a.teacher_det) if a.teacher_det else T_DET
    T_MAP = Path(a.teacher_map) if a.teacher_map else T_MAP
    root = CK / "v5/smoke_student" / a.tag
    root.mkdir(parents=True, exist_ok=True)
    for p in (T_DET, T_MAP):
        if not (p / "ckpt_last.pt").is_file():
            print(f"teacher {p}: no ckpt_last.pt yet", flush=True)
            return 2
    io = root / "io"
    io.mkdir(exist_ok=True)
    zstats = fit_zstats(root, a.zstats_tokens, a.zstats_det, a.zstats_map) if a.roundtrip else None
    ck2 = v5_ck2(io, bool(a.zkd), stage2=bool(a.roundtrip), zstats=zstats)
    CD.V2_OVERRIDES.update(V13)
    CD.V2_OVERRIDES.update(V5_AGENT)
    CD.FixtureDS = FixtureV5                                   # fit() builds its DataLoader from CD.FixtureDS
    agent = CD.build_agent(ck2)
    gd = GTDets(FixtureV5()) if a.roundtrip else None             # the 'rel' group needs valid, matchable rows
    if gd is not None:
        gd.register(agent)
    student = agent.ck_student
    v5_mods = {n: p.detach().clone() for n, p in student.named_parameters()
               if n.startswith(("relation.", "corrector_v5.", "z_adapter."))
               or n in ("v4_readout.rel_w", "v4_readout.resid.0.weight")}
    z_init = dict(student.init_report.get("z_stats") or {})
    det_mods = det_v5_params(agent)
    det_calls: list = []
    watch_det_loss(agent, det_calls)
    t0 = time.time()
    rt = {}
    if a.roundtrip:
        CD.fit(agent, root, 2)
        rt["L1_rc"] = CD.labeler(io, 1)
        for ep, name in ((3, "B"), (4, "C")):
            ck = CD.last_ckpt(root)
            agent = CD.build_agent(ck2)
            if gd is not None:
                gd.register(agent)
            watch_det_loss(agent, det_calls)
            CD.fit(agent, root, ep, ckpt=ck)
            if name == "B":
                rt["L2_rc"] = CD.labeler(io, 2)
                rt["gen2_edits"] = edit_labels(io, 2)
        student = agent.ck_student
    else:
        CD.fit(agent, root, a.epochs)
    steps = CD.read_jsonl(io / "steps_rank0.jsonl")
    res = {"tag": a.tag, "zkd": bool(a.zkd), "exec_model": EXEC_MODEL, "sec_fit": round(time.time() - t0, 1), "n_steps_logged": len(steps),
           "teachers": [str(T_DET), str(T_MAP)]}
    keys = sorted({k for s in steps for k in s if k.startswith(("ck2/", "v5/"))})
    res["v5_keys"] = [k for k in keys if k.startswith("v5/")]
    bad = sorted({k for s in steps for k, v in s.items() if k.startswith(("ck2/", "v5/")) and isinstance(v, float)
                  and not math.isfinite(v)})
    res["nonfinite_keys"] = bad
    res["nonfinite_loss"] = int(sum(float(s.get("ck2/loss_nonfinite", 0.0)) for s in steps))
    moved = {n: float((p.detach() - v5_mods[n]).abs().max()) for n, p in student.named_parameters() if n in v5_mods}
    res["v5_params_unmoved"] = sorted(n for n, d in moved.items() if d == 0.0)
    need = ["v5/rel_a_bce", "v5/rel_h_bce", "v5/ref_loss"] + (["v5/zkd_det", "v5/zkd_map"] if a.zkd else [])
    if a.roundtrip:                                              # the 'rel' group (REL_GROUP)
        need += ["v5/rel_bce", "v5/rel_cv_bce", "v5/rel_beyond_cv", "v5/rel_n_pairs", "v5/rel_match_frac",
                 "v5/rel_grad_rows"]
        need += ["v5/edit_bce", "v5/edit_n_pairs", "v5/edit_pos", "v5/zadapt_gamma_dev"]       # the 'zkd' group
        if a.zkd:
            need += [f"v5/zkd_{c}" for c in ("b_nc", "b_ttc", "H_nc", "H_ttc", "d_min")] + ["v5/zkd_raw_det",
                                                                                          "v5/zkd_raw_map"]
    res["missing_terms"] = [k for k in need if k not in keys]
    # det-head fixes (V1b / V1c / ACC): every fit loss call carries finite terms; their parameters moved
    det_keys = sorted({k for c in det_calls for k in c})
    res["det_v5"] = {"n_loss_calls": len(det_calls), "keys": det_keys,
                     "missing": [k for k in DET_V5_LOSS if k not in det_keys],
                     "calls_without_terms": sum(1 for c in det_calls if not all(k in c for k in DET_V5_LOSS)),
                     "nonfinite": sorted({k for c in det_calls for k, v in c.items() if not math.isfinite(v)}),
                     "mean": {k: sum(c[k] for c in det_calls if k in c) / max(1, sum(k in c for c in det_calls))
                              for k in det_keys}}
    det_now = det_v5_params(agent)
    res["det_v5"]["params_unmoved"] = sorted(n for n, p in det_now.items() if float((p - det_mods[n]).abs().max()) == 0)
    res["det_v5"]["n_params"] = len(det_now)
    # one eval-mode forward: the ck2_* inference with the v5 output
    ds = FixtureV5()
    f, t = ds[0]
    feats = {k: v[None] for k, v in f.items()}
    agent.eval()
    with torch.no_grad():
        pred = agent.forward(feats)
    tv = pred.get("ck2_v5_traj")
    res["infer_v5_traj"] = None if tv is None else {"shape": list(tv.shape), "finite": bool(torch.isfinite(tv).all()),
                                                     "src": int(pred["ck2_v5_src"][0]),
                                                     "mu_abs": float(pred["ck2_v5_mu"].abs().mean())}
    if gd is not None:                                           # 'rel' group: the fit trained the relation on valid rows
        mx = lambda k: max([float(s.get(k, 0.0)) for s in steps] or [0.0])                     # noqa: E731
        res["rel_group"] = {"gt_dets": gd.info, "hook_hits": gd.hits, "rel_n_pairs_max": mx("v5/rel_n_pairs"),
                            "rel_match_frac_max": mx("v5/rel_match_frac"), "rel_grad_rows_max": mx("v5/rel_grad_rows"),
                            "rel_bce_last": [s.get("v5/rel_bce") for s in steps if "v5/rel_bce" in s][-1:],
                            "rel_cv_bce_last": [s.get("v5/rel_cv_bce") for s in steps if "v5/rel_cv_bce" in s][-1:],
                            "cv_prior_moved": "relation.cv_prior" in v5_mods
                            and "relation.cv_prior" not in res["v5_params_unmoved"],
                            "corrector_n_top": int(student.corrector_v5.n_top)}
        res["rel_group"]["probe_matched_grad"] = probe_matched_grad(agent, gd)
        gd.remove()
        rg = res["rel_group"]
        rg["pass"] = (rg["rel_n_pairs_max"] > 0 and rg["rel_match_frac_max"] > 0 and rg["rel_grad_rows_max"] > 0
                      and rg["cv_prior_moved"] and rg["corrector_n_top"] == 39 and rg["probe_matched_grad"]["pass"])
    if a.roundtrip:                                              # 'zkd' group: adapters, Z-KD scale, L_edit
        mean = lambda k: (sum(float(s[k]) for s in steps if k in s) / max(1, sum(k in s for s in steps))  # noqa: E731
                          if any(k in s for s in steps) else None)
        mx = lambda k: max([float(s.get(k, 0.0)) for s in steps] or [0.0])                     # noqa: E731
        zk = ["v5/zkd_det", "v5/zkd_map", "v5/zkd_raw_det", "v5/zkd_raw_map"] + [
            f"v5/zkd_{c}" for c in ("b_nc", "b_ttc", "H_nc", "H_ttc", "d_min")]
        za = getattr(student, "z_adapter", None)
        zg = {"z_stats_files": {k: str(v) for k, v in (zstats or {}).items()}, "init_report": z_init,
              "adapter": None if za is None else {"n_params": sum(p.numel() for p in za.parameters()),
                                                  "fitted": bool(za.fitted()), "z_mu": za.z_mu.tolist(),
                                                  "z_sd": za.z_sd.tolist()},
              "adapter_params_unmoved": sorted(n for n in res["v5_params_unmoved"] if n.startswith("z_adapter.")),
              "zkd_means": {k: mean(k) for k in zk}, "edit_bce_mean": mean("v5/edit_bce"),
              "edit_n_pairs_max": mx("v5/edit_n_pairs"), "edit_pos_mean": mean("v5/edit_pos"),
              "edit_match_frac_max": mx("v5/edit_match_frac"), "edit_dup_masked_max": mx("v5/edit_dup_masked"),
              "zadapt_gamma_dev_last": [s.get("v5/zadapt_gamma_dev") for s in steps if "v5/zadapt_gamma_dev" in s][-1:]}
        zg["pass"] = (za is not None and zg["adapter"]["fitted"] and not zg["adapter_params_unmoved"]
                      and zg["edit_n_pairs_max"] > 0 and zg["edit_bce_mean"] is not None
                      and math.isfinite(zg["edit_bce_mean"]) and bool(z_init.get("det")) and bool(z_init.get("map"))
                      and (not a.zkd or all(zg["zkd_means"][k] is not None and math.isfinite(zg["zkd_means"][k])
                                            for k in zk)))
        res["zkd_group"] = zg
    res["probe_valid_objects"] = probe_valid_objects(agent)
    dv = res["det_v5"]
    det_ok = (dv["n_loss_calls"] > 0 and not dv["missing"] and dv["calls_without_terms"] == 0 and not dv["nonfinite"]
              and not dv["params_unmoved"] and dv["n_params"] > 0)
    res["pass"] = (not bad and res["nonfinite_loss"] == 0 and not res["missing_terms"] and tv is not None
                   and bool(torch.isfinite(tv).all()) and tuple(tv.shape) == (1, 8, 3) and det_ok
                   and res["probe_valid_objects"]["pass"])
    if gd is not None:
        res["pass"] = res["pass"] and res["rel_group"]["pass"]
    if "zkd_group" in res:
        res["pass"] = res["pass"] and res["zkd_group"]["pass"]
    if a.roundtrip:
        ep = lambda e: [s for s in steps if int(s.get("epoch", -1)) == e]                     # noqa: E731
        e2, e3 = ep(2), ep(3)
        g2 = rt.get("gen2_edits") or {}
        rt["checks"] = {
            "labelers_rc_0": rt.get("L1_rc") == 0 and rt.get("L2_rc") == 0,
            "epoch2_records_tau1": bool(e2) and all(float(s.get("v5/rec_tau1_valid_frac", 0)) > 0 for s in e2),
            "gen2_tau1_labelled": bool(g2.get("has_edit_arrays")) and g2.get("slot0_ok", 0) > 0
            and g2.get("other_slots_ok", 1) == 0,
            "epoch3_awr": bool(e3) and all("v5/awr" in s and math.isfinite(float(s["v5/awr"]))
                                           and float(s.get("v5/awr_n", 0)) > 0 for s in e3),
        }
        rt["epoch3_awr"] = [{k: s.get(k) for k in ("mb", "v5/awr", "v5/awr_n", "v5/awr_A_mean", "v5/tau1_better_frac",
                                                   "ck2/n_cand", "ck2/G_src_prev")} for s in e3]
        res["roundtrip"] = rt
        res["pass"] = res["pass"] and all(rt["checks"].values())
    (root / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
