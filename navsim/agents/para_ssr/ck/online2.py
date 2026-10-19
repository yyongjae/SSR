"""CK2 e2e: v2 + CK2 student trained from scratch end to end (spec: scratchpad ck2e2e/SPEC.md; SSR/report/47 s9).

Used ONLY when ParaSSRConfig.ck_e2e2 is enabled; with the default (empty dict / enabled false) the agent never imports
this module (v2 bit-identical).  The old CK e2e (ck/online.py) is imported, never modified.

Schedule (epoch = trainer.current_epoch)
  'warmup'         epoch < record_from_epoch (4)       : per token 32 sampled raw anchors (AnchorSampler, 8 near + 8 mid
                                                         + 16 strat) + warmup.n_var (32) type-balanced rule variants
                                                         of the fixed near+mid-16 subset, official labels
                                                         (raw256 / var_separate_sampler16_accstraight); online teacher KD
                                                         on the same 64 candidates.
  'warmup_record'  record_from <= epoch < onpolicy_from : the warm-up step + the student's own v2 top-16 and its 80 rule
                                                         variants (96 per token, no grad, no teacher) are RECORDED for the
                                                         background labeler (tools/ck/e2e2/labeler2.py).
  'onpolicy'       epoch >= onpolicy_from_epoch (5)     : current top-16 + n_var_step (32) type-balanced variants of them
                                                         (48, KD / lateral KD; the 16 originals carry the surrogate) +
                                                         the label set G of generation <= e - label_lag (16 identities +
                                                         onpolicy_label.n_var (32) variants, official labels; rows without
                                                         a generation -> the warm-up-style fallback set) = 96 per token in
                                                         ONE student pass.  Recording continues until record_until.
EP target (ep_target, user 2026-10-08 ~20:10 KST; default 'official' since 2026-10-09 00:30 KST, decoupled was worse on navtest): the label targets y_lab are
  ep_target.ck_targets(official 9-column labels, ep_target) everywhere (warm-up raw / variant targets, on-policy
  generation labels, fallback set, score prior; e2e_data2); 'decoupled' = official EP without the NC * DAC * DDC factor.
  The teachers must have been trained with the same --ep-target (launch_util2.teacher_check refuses otherwise), so the
  label BCE and the EP KD target mean the same thing.  KD itself is unchanged (teacher probabilities).
KD calibration (kd_calib, user 2026-10-08 ~22:35 KST; ck/kd_calib.py): with kd_calib.enabled each teacher's score LOGITS
  are Platt-scaled per key, z' = a_k z + b_k (a_k > 0; tools/ck/e2e2/fit_kd_calib.py, NLL fit on navtrain_train r34
  top-16 candidates vs the student's label targets), BEFORE the sigmoid and the DET / MAP / mean combination
  (combine_teacher); fitted keys = the KD keys the teacher is a source of (DET nc / ep / ttc, MAP dac / ep), identity
  elsewhere.  TeacherPair2 refuses a calibration file whose run / which / ckpt sha16 / ep_target / arm differ from the
  loaded teacher or that leaves one of its KD keys unfitted.  Lateral KD is untouched.  enabled false = old path.
Student CK loss (added after the v2 grad balancer, outside it; ParaSSRAgent.compute_loss)
  L_CK2 = loss_weight * [ lambda_score * BCE(s_lab, y_lab)                       (official labels, EP per ep_target, 1.0)
                          + lambda_sur * L_sur(decode(tau_sur, z_lon = 0, w_lat; slope lon_st_slope))
                          + r * lambda_kd_score * KL(p_T || sigmoid(s_kd))       (per-key source nc/ttc <- DET,
                                                                                  dac <- MAP, ep/comfort <- mean; 0.5)
                          + r * w_lat * L1(e_lat_s, 0.5 (e_lat_DET + e_lat_MAP)) ] (ONE L1 on the mean target)
          + 0 * (score logits + w_lat of every candidate)                          (all trainable params in the graph)
  w_lat = losses.EmaBalancer(lat_kd) weight (0 before lat_kd.start_mb) or lat_kd.fixed (balance 'fixed').  The EMA is
  fed the all-rank MEANS of lambda_sur * L_sur and the lateral-KD L1 (one all-reduce per training micro-batch; the
  per-rank values are still logged as ck2/sur / ck2/lat_kd), so w_lat and the EMA state are identical on every rank
  and the rank-0 Lightning checkpoint restores every rank exactly (report 48 F4).
  No longitudinal head: trunk.lon_head is zeroed + frozen and every decode passes z_lon = 0 explicitly (student and
  teachers); trunk.gate_head is frozen.  No tau'_KD, no rescore KD, no control KD on c_lon.
  KD is off for a token without a DET or MAP BEV (combine_teacher kd_ok = det ok & map ok).
Temporal DET teacher (refiner-v3, docs/refiner_v3/STEP2_E2E.md; decided by the DET teacher run's OWN config.json
  temporal cfg, no ck_e2e2 key): load_ck rebuilds the temporal branch (a PRIVILEGED run is refused), the target builder
  adds the teacher's CLEAN causal object rows of the train object pack (e2e_data2.KDObjRows: kd_obj_in_0 /
  kd_obj_mask_0 / kd_frame_valid_0 / kd_obj_ok_0) and TeacherPair2.run feeds exactly OBJ_INPUT_KEYS to the DET
  teacher's TemporalBranch.scene + hook (= CKNet.forward(..., objs=) of train_ck2 / infer_val_v3, fp16 autocast on CUDA
  as their evaluations); kd_obj_ok_0 False -> the DET teacher's ok False -> KD off for the token.  The MAP teacher and
  the student (CK2, no temporal branch) are unchanged; a non-temporal DET teacher runs the old code path bit for bit.
BEV-KD arm (bev_kd.enabled; ck/bev_kd_arm.py): + adapter_weight * sum_t L_t(GradScale(bev_embed, lam_t / adapter_weight))
  over bev_kd.teachers (arm: [det, map], user 2026-10-08 ~20:40 KST), one RatioController per teacher (lam_t * |dL_t/dBEV|
  ~ ratio x |g_plan|, measured per teacher on the v2 'gnorm/plan' micro-batches, all-reduced); returned inside the CK
  loss.  Tokens without the teacher's BEV (kd_ok_<i> False) are masked for that teacher and counted (bevkd_miss_<t>).
Gradient paths: candidates detached; CK -> student CK -> AdapterSGrid (x bev_grad_scale) -> v2 bev_embed; teachers
  frozen, no_grad, fp16 autocast on CUDA, plain python objects (not in state_dict / DDP / optimiser).
Non-finite: a non-finite L_CK2 is replaced by 0 * sum(student params) (counted); a non-finite gradient skips the whole
  optimiser step (callback); CK params are clipped to `clip` before Lightning's global clip.
Inference (eval, infer_outputs): ck2_* outputs over the 96-candidate pool (v2 top-16 x 6 variants, c = k * 6 + v):
  CK2 logits, the learned lateral control / offsets and every column's laterally corrected trajectory; the selection
  (ck/select2.py: old ck_final blended with v2 by beta, lateral applied after selection) happens in the evaluation.
refiner v4_3 (docs/refiner_v4/V4_3_SPEC.md M5 / M6; every option OFF by default = the paths above unchanged):
  the v4_3 knobs are OPTIONAL keys of the v4 block (_d_v4_opt: absent from the defaults / ck_e2e2.yaml, so the older
    arms' configs are unchanged); v4.readout 'range' + its keys (exec_model, ev_tbins, ev_tbin_s, ep_evidence, vmax)
    pass through to V4Cfg;
  v4.edits 'computed': infer adds the computed edits of the critic's top-edit_k parents of the 96 pool (CKNet.
    computed_edits on the same parent pass) as ck2_edit_* (+ ck2_edit_valid, ck2_edit_kind; extra candidates, never a
    gate; ck2_edit_valid False for filler slots and, without the corrector, for no-edit slots: kind 0 = the source
    column bitwise); on-policy steps record the edits of the KD set (computed_edits_rec, same validity rule) in the rec2
    edit block, labeler2 labels them with the 96, and LabelStore2(with_edits) puts them in the next epochs' BCE label
    set (as the corrector's edits; the AWR term only with v4.corrector); logs v4/edit_rec_{valid,lon,lat,both}_frac and
    v4/edit_clear_frac (the made edits that read inside the predicted tube);
  infer_select.edits (optional; required when the student emits edits): true = ck2_traj / ck2_sel_idx over the 96 + K
    (select96_edits = eval_ck2's 96 + K rule; an edit pick is the edit itself, no lateral head), false = the 96 only;
  v4.kd_score_scope 'near': NC score KD only on tokens whose nearest lead (lead_ttc0) has ttc0 < NEAR_TTC_S, DAC score KD
    only on candidates whose parent's min executed corner SDF (parent_min_corner_sdf) < NEAR_SDF_M, from the ref_* GT
    (tokens without GT excluded; a batch without the ref_* keys is a ValueError); other KD keys unrestricted
    (score_kd_near: each key's KL is the mean over its own mask, so the per-candidate NC / DAC weight is ~1 / (near
    fraction) times the 'all' weight while each key's total weight stays lambda_kd_score / n_keys);
  v4.kd_constraint: + r * (lambda_occ * BCE(student B, sigmoid(DET teacher B)) (occ_channels 'BU': + the U channel,
    'BBp': + the Bp channel (R2 only), 'all': B, U, Bp) + lambda_margin * Huber(student D, MAP teacher D) on
    s <= map_range_m) on the student's unique parents of the KD set (constraint_kd), teacher fields from
    TeacherPair2.run(fields=kd_constraint_request(occ_channels)) (the requested groups; range teachers on the student's
    grid with the student's rg_stopped_aware only -- rg_read may differ, recorded at launch: check_constraint_teachers
    at runtime, the same rule and message at launch, launch_util2.kd_constraint_check);
  v4.rg_read / v4.rg_stopped_aware (Phase 0b R2): V4Cfg keys of the student's range readout (any4 read; the B target
    rule of its GT field aux + the Bp group that its TTC reads (V4Cfg docstring); the computed edits read the same way,
    ck/v4/edits.py); rg_stopped_aware is also required of both teachers for the constraint KD (an R2 student with R2
    teachers distils the stopped-aware B as B), rg_read is not (how a field is read, not what a cell holds);
  v4.rg_gt_aux {enabled, lambda, max_parents}: + lambda * (BCE(student B / U (/ Bp), GT) + Huber(student D, GT SDF)) on the
    student's first max_parents unique own-parent rows of the KD set (range_targets.gt_field_aux, the range teachers'
    aux), GT computed online from the batch's ref_obj_* / ref_sdf: a training TARGET only (never an input), tokens
    without GT masked, independent of kd_constraint.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .constants import CK_KEYS, SUR_MARGINS, SUR_TERMS, SUR_WEIGHTS
from .online import N_ROWS, _sanitize_traj, bev_sgrid, label_supply_warning, student_topk

# ----------------------------------------------------------------------------------------------- constants
# layout / stream / G_src constants and the variant-column draw, the online variants and the recorder are the data
# side's (ck/e2e_data2.py, ck/cands2.py): ONE implementation of every deterministic draw (re-exported here under the
# SPEC s1-3 names).
from . import cands2 as C2  # noqa: E402
from .cands2 import online_variants, v0_from_status  # noqa: E402,F401
from .e2e_data2 import (EP_TARGET_DEFAULT, G_K, GROUP_VAR, K16, KD_OBJ_INPUT, KD_OBJ_KEYS, KD_OBJ_OK,  # noqa: E402,F401
                        N_NEAR_MID, NV, SRC_FALLBACK, SRC_NAMES, SRC_NODATA, SRC_OLDER, SRC_PREV, STREAM_LAB,
                        STREAM_NOW, STREAM_WU, TEACHER_DET_RUN, TEACHER_MAP_RUN, VAR_SAMPLING, VNAMES, CandRecorder2,
                        draw_cols, sample_cols, teacher_temporal, type_balanced_cols, uniform_cols)
from .ep_target import EP_TARGETS  # noqa: E402
from .kd_calib import calib_path  # noqa: E402

PHASES2 = ("warmup", "warmup_record", "onpolicy")
PHASE2_CODE = {p: i for i, p in enumerate(PHASES2)}
SUR_GROUP_MAX = {"near": 0, "near_mid": 1, "all": 2}
SUR_GROUP_N = {"near": 8, "near_mid": 16, "all": 32}
SCORE_PRIORS = ("ck2_warmup", "teacher", "none")
_CK = "/home/external-user/ssd/yongjae_refiner/ck"


# ----------------------------------------------------------------------------------------------- config
def _d_variants() -> Dict[str, Any]:
    return {"speeds": [-1.0, -0.5, 0.5], "lats": [-0.5, 0.5], "combine": "separate", "s_on_frac": 0.2,
            "ext": "straight", "compute_dtype": "float64"}


def _d_lat_kd() -> Dict[str, Any]:
    return {"balance": "ema", "ratio": 0.5, "m": 0.99, "floor": 1e-4, "cap": 10.0, "start_mb": 1000, "fixed": 1.0}


def _d_warmup() -> Dict[str, Any]:
    return {"packed": f"{_CK}/packed/navtrain_train", "raw_labels": f"{_CK}/labels/navtrain_train/raw256",
            "var_dir": f"{_CK}/ck2/labels/navtrain_train/var_separate_sampler16_accstraight", "sampler_seed": 0,
            "n_var": 32, "sur_groups": "near_mid", "lat_kd_groups": "near_mid",
            # refiner v4 (optional): first-failure time-index sidecars of raw_labels / var_dir
            # (tools/ck/v4/build_time_idx.py) and the label-derived P_ref of the warm-up rows
            "raw_tidx": None, "var_tidx": None, "pref": False}


def _d_onpolicy_label() -> Dict[str, Any]:
    return {"n_orig": 16, "n_var": 32, "fallback": "warmup"}


def _d_kd_calib() -> Dict[str, Any]:
    # user 2026-10-08 ~22:35 KST: Platt calibration of the teacher logits for KD (files of tools/ck/e2e2/fit_kd_calib.py;
    # null = that teacher uncalibrated)
    return {"enabled": True, "det": str(calib_path(TEACHER_DET_RUN, "last")),
            "map": str(calib_path(TEACHER_MAP_RUN, "last"))}


def _d_bev_kd() -> Dict[str, Any]:
    # ratio_<t> / cap_<t>: optional per-teacher overrides of ratio / cap (None = the shared value) for that teacher's
    # RatioController (bev_kd_arm.controllers_from_cfg)
    # refiner-v4 default (user 2026-10-15: the ck2e2e_bevkd setting = BEV KD from both teachers); the ck2e2e_main arm
    # (enabled false, teachers [det]) is tools/ck/e2e2/main_arm.set
    return {"enabled": True, "teachers": ["det", "map"], "distance": "mse", "init": "zero", "ln": True, "target_clip": None,
            "ratio": 0.1, "cap": 0.25, "m": 0.9, "w_max": None, "start_mb": 10600, "adapter_weight": 1.0,
            "lr_mult": 3.0, "weight_decay": 0.01, "ratio_det": None, "ratio_map": None, "cap_det": None,
            "cap_map": None}


BEV_KD_PER_TEACHER = ("ratio_det", "ratio_map", "cap_det", "cap_map")


def _d_v4() -> Dict[str, Any]:
    # refiner v4 (ck/v4): network readout (V4Cfg keys) + student loss weights + corridor version + KD routing.
    # Defaults = off (the CK2 / refiner-v3 student, bit for bit); tools/ck/e2e2/v4_arm.set turns the v4 design on.
    return {"readout": "mean", "pref_head": False, "time_free": True, "lat_haz": True, "horizon": 4.0,
            "lambda_pref": 0.1, "lambda_station": 0.5, "lambda_ev": 0.5, "corridor_version": 1,
            "kd_parent_only": False, "corrector": False, "corr_k": 8, "lambda_awr": 0.1}


def _d_v4_opt() -> Dict[str, Any]:
    # refiner v4_3 (docs/refiner_v4/V4_3_SPEC.md; tools/ck/e2e2/v4_3_arm.set): OPTIONAL keys of the v4 block.  They are
    # neither in the defaults nor in ck_e2e2.yaml, so every older arm's effective ck_e2e2 dict, override list and
    # to_dict stay exactly as before; a dict / set file that names one adds it (launch_util2.apply_sets2 declares it
    # at this default before applying the set lines) and CKE2E2Config.v4_opt reads it (absent = this default).
    #   exec_model .. edit_dlat: V4Cfg v4_3 keys (read by the 'range' readout only; defaults = V4Cfg's);
    #   kd_score_scope 'all' (old) | 'near' (NC score KD on tokens whose nearest lead has ttc0 < NEAR_TTC_S, DAC score
    #     KD on candidates whose parent's min executed corner SDF < NEAR_SDF_M; both from the ref_* GT, train only);
    #   kd_constraint: the constraint KD block (_d_kd_constraint);
    #   rg_read / rg_stopped_aware: V4Cfg's Phase 0b R2 keys (range readout: B / U read 'nearest' | 'any4'; the B
    #     target of the student's own GT field aux = the stopped-aware blocked field);
    #   rg_gt_aux: the student's own GT field aux block (_d_rg_gt_aux).
    return {"exec_model": "plan", "ev_tbins": 10, "ev_tbin_s": 0.25, "ep_evidence": False, "vmax": "off",
            "edits": "off", "edit_k": 8, "edit_dt": 0.2, "edit_dlat": 0.3, "kd_score_scope": "all",
            "kd_constraint": _d_kd_constraint(), "rg_read": "nearest", "rg_stopped_aware": False,
            "rg_gt_aux": _d_rg_gt_aux(), "v5": _d_v5()}


V4_NET_KEYS = ("readout", "pref_head", "time_free", "lat_haz", "horizon", "corrector", "corr_k")
V4_NET_OPT_KEYS = ("exec_model", "ev_tbins", "ev_tbin_s", "ep_evidence", "vmax", "edits", "edit_k", "edit_dt",
                   "edit_dlat", "rg_read", "rg_stopped_aware")   # optional v4 keys passed to V4Cfg when present


def _d_kd_constraint() -> Dict[str, Any]:
    # refiner v4_3 M6 constraint KD (docs/refiner_v4/V4_3_SPEC.md; constraint_kd), ck_e2e2.v4.kd_constraint: the DET
    # teacher's blocked field groups (sigmoid of its range logits; occ_channels = CN_GROUPS: 'B' = the B channel only =
    # spec M6 "DET B BCE", the default (the stopped-aware B for an R2 student / teacher pair, the plain not-behind B
    # otherwise); 'BU' = + the TTC overlap channel U; 'BBp' = + the plain not-behind Bp that an R2 readout's TTC reads
    # below the 2 m switch (rg_stopped_aware students only); 'all' = B, U, Bp (R2 only).  Options beyond 'B' are not
    # confirmed by the user: with 'B' the student's U (/ Bp), read by its TTC key and by the edits' TTC windows, learns
    # from the TTC label BCE (and rg_gt_aux) only) -> the student's logits, BCE (lambda_occ), and the MAP teacher's
    # margin D -> the student's D, Huber (lambda_margin) on the stations s <= map_range_m (the ReSMap BEV's front
    # reach), both on the student's unique parents of the KD set and masked by the teacher's per-token ok.  enabled
    # false = no term (old KD exactly).
    return {"enabled": False, "lambda_occ": 0.5, "lambda_margin": 0.5, "map_range_m": 32.0, "occ_channels": "B"}


def _d_v5() -> Dict[str, Any]:
    # refiner v5 student (SSR/report/97_refiner_v5_design.md; ck/v5), ck_e2e2.v4.v5 (OPTIONAL v4 key: absent = off = the
    # v4_3 student exactly): relation 'student' = the object relation channel on the student's own det head
    # (ck/v5/student_objects.py; needs agent.config.expose_det_outputs for inference), unknown_ev = unknown-evidence
    # residual input, corrector = corrector A_S (tau1 = decode(tau0, mu_lon, w_lat_head + mu_lat)) trained by
    # lambda_ref * L_refine(X1) on the first ref_cols PARENT columns of the KD set (GT-ok tokens; ref_w term weights
    # 'road=1,obj=1,...', ck/v5/refine_loss.REF_W), lambda_rel_a reliability BCE (student rows vs GT t0 objects),
    # lambda_rel_h per-state hazard BCE (first rel_h_cols KD columns), z_kd = Z-KD of the per-state reads to the v5
    # teachers' (DET: b_nc, b_ttc (+ H_nc, H_ttc with both relation channels); MAP: d_min; masked SmoothL1 on states
    # that are readable for both and KD-ok, ramped like the other KD terms), infer_refine = inference also emits tau1 of
    # the deployed pick (ck2_v5_traj).
    return {"enabled": False, "relation": "off", "n_obj": 32, "score_thr": 0.3, "unknown_ev": False,
            "corrector": False, "z_stride": 2, "lambda_ref": 1.0, "ref_w": "", "ref_cols": 16, "lambda_rel_a": 0.2,
            "lambda_rel_h": 0.5, "rel_h_cols": 16, "z_kd": False, "lambda_zkd_det": 0.5, "lambda_zkd_map": 0.5,
            "infer_refine": True}


def _d_rg_gt_aux() -> Dict[str, Any]:
    # refiner v4_3 (c) student GT range aux (ck_e2e2.v4.rg_gt_aux; range_targets.gt_field_aux, the range teachers'
    # train_ck2.range_gt_aux): the student's OWN B / U logits (BCE) and margin D (Huber) on its first max_parents unique
    # own-parent rows of the KD set, against the GT fields computed online from the batch's ref_obj_* / ref_sdf (the
    # B target rule = v4.rg_stopped_aware); train time only, tokens without GT (ref_gt_ok False) masked, weight lambda
    # (not ramped: a GT term, not KD), independent of kd_constraint.  The GT objects / SDF are TARGETS only, never an
    # input of the student.  enabled false = no term (the old loss exactly).
    return {"enabled": False, "lambda": 0.5, "max_parents": 8}


KD_SCORE_SCOPES = ("all", "near")
# constraint KD field groups (kd_constraint.occ_channels -> the distilled groups, in this order); a range net's groups:
# B (occ_logit_u; the stopped-aware B with rg_stopped_aware), U (ovl_logit_u), Bp (bp_logit_u, rg_stopped_aware only)
CN_GROUPS = {"B": ("B",), "BU": ("B", "U"), "BBp": ("B", "Bp"), "all": ("B", "U", "Bp")}
OCC_CHANNELS = tuple(CN_GROUPS)
FIELD_GROUP_KEY = {"B": "occ_logit_u", "U": "ovl_logit_u", "Bp": "bp_logit_u"}    # score_set haz key of each group
FIELDS_TRUE = ("B", "U")           # teacher_forward(return_fields=True) / run(fields=True): the original B | U dict
NEAR_TTC_S = 4.0                   # [s] kd_score_scope 'near': NC KD where the nearest lead's ttc0 < this
NEAR_SDF_M = 1.0                   # [m] kd_score_scope 'near': DAC KD where the parent's min executed corner SDF < this
# the targets kd_score_scope 'near' reads (lead_ttc0 + parent_min_corner_sdf); absent -> ValueError (_near_scope)
NEAR_REF_KEYS = ("ref_gt_ok", "ref_obj_kf", "ref_obj_first", "ref_obj_meta", "ref_obj_n", "ref_obj_n_kf", "ref_cl_xy",
                 "ref_cl_valid", "ref_sdf")
LEAD_CLASSES = (0, 2)              # refiner.gt_future.CLASS_NAMES: vehicle, bicycle (tools/ck/data/lead_split.py rule)
LEAD_LAT_M, LEAD_DH_RAD, LEAD_GAP_MAX, LEAD_EGO_FRONT = 1.8, math.radians(60.0), 80.0, 4.049
TTC0_NONE = 99.0                   # [s] ttc0 without a closing lead (lead_split.py convention)
CK_MARGIN_CLIP = 5.0               # [m] margin targets clipped like losses.evidence_aux / train_ck2's range aux
V5_TOKEN_KEYS = ("rel_a", "rel_valid", "rel_objs_tok")   # refiner v5 relation haz keys per TOKEN (not per candidate)
RANGE_U_KEYS = ("occ_logit_u", "ovl_logit_u", "margin_u", "bp_logit_u")   # score_set range haz keys per PARENT row
                                                                           # (not per candidate; bp: rg_stopped_aware)


_NESTED2 = {"variants": _d_variants, "lat_kd": _d_lat_kd, "warmup": _d_warmup, "onpolicy_label": _d_onpolicy_label,
            "bev_kd": _d_bev_kd, "kd_calib": _d_kd_calib, "v4": _d_v4}
KD_CALIB_TEACHERS = ("det", "map")
_INFER_SELECT_KEYS = ("beta", "set", "lat_mode")
# optional: edits (bool) = select over the 96 pool + the student's K edits (select96_edits, eval_ck2's 96 + K rule);
# required when the student emits edits (v4.corrector / v4.edits computed), absent = the old 96-only selection
_INFER_SELECT_OPT = ("edits",)


def _b(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")


def _opt_float(v) -> Optional[float]:
    return None if v is None or (isinstance(v, str) and v.strip().lower() in ("none", "null", "")) else float(v)


def _opt_str(v) -> Optional[str]:
    return None if v is None or (isinstance(v, str) and v.strip().lower() in ("none", "null", "")) else str(v)


def _mode_str(v) -> str:
    """mode strings whose value may be 'off' (v4.vmax / v4.edits): YAML 1.1 (PyYAML, the launcher's set-file parser)
    reads an unquoted off as False -> 'off'."""
    return "off" if v is False else str(v)


def _str_list(v) -> List[str]:
    if isinstance(v, str):
        v = [s.strip() for s in v.strip().strip("[]").split(",") if s.strip()]
    return [str(s) for s in v]


@dataclass
class CKE2E2Config:
    enabled: bool = False
    io_dir: str = ""
    topk: int = K16
    variants: Dict[str, Any] = field(default_factory=_d_variants)
    n_var_step: int = 32
    var_sampling: str = "type_balanced"
    teacher_det_run: str = TEACHER_DET_RUN           # ck2T10 (user 2026-10-09: official-EP teachers)
    teacher_map_run: str = TEACHER_MAP_RUN           # ck2M10
    teacher_which: str = "last"
    teacher_require_done: bool = True
    teacher_ep_check: bool = True                    # TeacherPair2 refuses teachers whose config.json ep_target !=
    #                                                  ep_target (false only for tests / CPU dry runs whose smoke
    #                                                  teachers predate --ep-target, i.e. 'official')
    teacher_amp: bool = True
    amp: bool = False
    bev_grad_scale: float = 1.0
    seed: int = 0
    ep_target: str = EP_TARGET_DEFAULT                # 'official' (user 2026-10-09 00:30 KST) | 'decoupled'
    score_prior: str = "ck2_warmup"
    score_prior_rows: int = 4000
    loss_weight: float = 1.0
    lambda_score: float = 1.0
    lambda_kd_score: float = 0.5
    kd_score_keys: List[str] = field(default_factory=lambda: ["nc", "dac", "ep", "ttc"])  # user 10-08: no comfort KD
    kd_calib: Dict[str, Any] = field(default_factory=_d_kd_calib)                       # user 10-08 ~22:35 KST
    lambda_sur: float = 1.0
    lat_kd: Dict[str, Any] = field(default_factory=_d_lat_kd)
    lon_st_slope: float = 0.1
    lr_mult: float = 3.0
    weight_decay: float = 0.01
    clip: float = 1.0
    warmup: Dict[str, Any] = field(default_factory=_d_warmup)
    record_from_epoch: int = 4
    record_until_epoch: Optional[int] = None
    onpolicy_from_epoch: int = 5
    kd_ramp_epochs: float = 0.0
    label_lag: int = 1
    label_refresh_every_mb: int = 1000
    onpolicy_label: Dict[str, Any] = field(default_factory=_d_onpolicy_label)
    rec_chunk_tokens: int = 64
    grad_share_every: int = 100
    log_every: int = 1
    infer_outputs: bool = True
    infer_select: Optional[Dict[str, Any]] = None
    strict_rows: float = 0.999
    bev_kd: Dict[str, Any] = field(default_factory=_d_bev_kd)
    v4: Dict[str, Any] = field(default_factory=_d_v4)

    # ------------------------------------------------------------------------------------------ parse
    @classmethod
    def from_any(cls, x: Any) -> "CKE2E2Config":
        """None / dict / DictConfig / CKE2E2Config -> CKE2E2Config (defaults merged, nested dicts merged key-wise);
        an unknown key (top level or nested) is a ValueError."""
        if isinstance(x, CKE2E2Config):
            return cls.from_any(x.to_dict())
        if x is None:
            x = {}
        try:  # DictConfig (Hydra without _convert_)
            from omegaconf import DictConfig, OmegaConf
            if isinstance(x, DictConfig):
                x = OmegaConf.to_container(x, resolve=True)
        except ImportError:  # pragma: no cover
            pass
        if not isinstance(x, dict):
            raise ValueError(f"ck_e2e2 must be a mapping, got {type(x).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(x) - known)
        if unknown:
            raise ValueError(f"ck_e2e2: unknown key(s) {unknown}")
        kw: Dict[str, Any] = {}
        for k, v in x.items():
            if k in _NESTED2:
                base = _NESTED2[k]()
                if v is None:
                    v = {}
                if not isinstance(v, dict):
                    raise ValueError(f"ck_e2e2.{k} must be a mapping, got {v!r}")
                allowed = set(base) | (set(_d_v4_opt()) if k == "v4" else set())   # v4: + the optional v4_3 keys
                bad = sorted(set(v) - allowed)
                if bad:
                    raise ValueError(f"ck_e2e2.{k}: unknown key(s) {bad}")
                base.update(v)
                kw[k] = base
            elif k == "infer_select":
                if v is None or (isinstance(v, str) and v.strip().lower() in ("none", "null", "")):
                    kw[k] = None
                    continue
                if not isinstance(v, dict):
                    raise ValueError(f"ck_e2e2.infer_select must be a mapping or null, got {v!r}")
                bad = sorted(set(v) - set(_INFER_SELECT_KEYS) - set(_INFER_SELECT_OPT))
                if bad or not set(_INFER_SELECT_KEYS) <= set(v):
                    raise ValueError(f"ck_e2e2.infer_select needs exactly {list(_INFER_SELECT_KEYS)} (+ optional "
                                     f"{list(_INFER_SELECT_OPT)}), got {sorted(v)}")
                kw[k] = dict(v)
            else:
                kw[k] = v
        c = cls(**kw)
        c._coerce()
        return c

    def _coerce(self) -> None:
        for k in ("enabled", "teacher_require_done", "teacher_ep_check", "teacher_amp", "amp", "infer_outputs"):
            setattr(self, k, _b(getattr(self, k)))
        for k in ("topk", "n_var_step", "seed", "score_prior_rows", "record_from_epoch", "onpolicy_from_epoch",
                  "label_lag", "label_refresh_every_mb", "rec_chunk_tokens", "grad_share_every", "log_every"):
            setattr(self, k, int(getattr(self, k)))
        if self.record_until_epoch is not None:
            self.record_until_epoch = (None if str(self.record_until_epoch).strip().lower() in ("none", "null")
                                       else int(self.record_until_epoch))
        for k in ("bev_grad_scale", "loss_weight", "lambda_score", "lambda_kd_score", "lambda_sur", "lon_st_slope",
                  "lr_mult", "weight_decay", "clip", "kd_ramp_epochs", "strict_rows"):
            setattr(self, k, float(getattr(self, k)))
        ks = _str_list(self.kd_score_keys)
        if not ks or len(set(ks)) != len(ks) or any(k not in CK_KEYS for k in ks):
            raise ValueError(f"ck_e2e2.kd_score_keys must be a non-empty subset of {list(CK_KEYS)}, got {ks}")
        self.kd_score_keys = [k for k in CK_KEYS if k in ks]                  # canonical CK_KEYS order
        v = dict(self.variants)
        self.variants = {"speeds": [float(a) for a in _str_list(v["speeds"])] if isinstance(v["speeds"], str)
                         else [float(a) for a in v["speeds"]],
                         "lats": [float(a) for a in _str_list(v["lats"])] if isinstance(v["lats"], str)
                         else [float(a) for a in v["lats"]],
                         "combine": str(v["combine"]), "s_on_frac": float(v["s_on_frac"]), "ext": str(v["ext"]),
                         "compute_dtype": str(v["compute_dtype"]).replace("torch.", "")}
        e = dict(self.lat_kd)
        self.lat_kd = {"balance": str(e["balance"]), "ratio": float(e["ratio"]), "m": float(e["m"]),
                       "floor": float(e["floor"]), "cap": _opt_float(e["cap"]), "start_mb": int(e["start_mb"]),
                       "fixed": float(e["fixed"])}
        w = dict(self.warmup)
        self.warmup = {"packed": str(w["packed"]), "raw_labels": str(w["raw_labels"]), "var_dir": str(w["var_dir"]),
                       "sampler_seed": int(w["sampler_seed"]), "n_var": int(w["n_var"]),
                       "sur_groups": str(w["sur_groups"]), "lat_kd_groups": str(w["lat_kd_groups"]),
                       "raw_tidx": _opt_str(w.get("raw_tidx")), "var_tidx": _opt_str(w.get("var_tidx")),
                       "pref": _b(w.get("pref", False))}
        o = dict(self.onpolicy_label)
        self.onpolicy_label = {"n_orig": int(o["n_orig"]), "n_var": int(o["n_var"]), "fallback": str(o["fallback"])}
        b = dict(self.bev_kd)
        self.bev_kd = {"enabled": _b(b["enabled"]), "teachers": _str_list(b["teachers"]), "distance": str(b["distance"]),
                       "init": str(b["init"]), "ln": _b(b["ln"]), "target_clip": _opt_float(b["target_clip"]),
                       "ratio": float(b["ratio"]), "cap": float(b["cap"]), "m": float(b["m"]),
                       "w_max": _opt_float(b["w_max"]), "start_mb": int(b["start_mb"]),
                       "adapter_weight": float(b["adapter_weight"]), "lr_mult": float(b["lr_mult"]),
                       "weight_decay": float(b["weight_decay"]),
                       **{k: _opt_float(b[k]) for k in BEV_KD_PER_TEACHER}}
        kc = dict(self.kd_calib)
        self.kd_calib = {"enabled": _b(kc["enabled"]), **{t: _opt_str(kc[t]) for t in KD_CALIB_TEACHERS}}
        q = dict(self.v4)
        self.v4 = {"readout": str(q["readout"]), "pref_head": _b(q["pref_head"]), "time_free": _b(q["time_free"]),
                   "lat_haz": _b(q["lat_haz"]), "horizon": float(q["horizon"]), "lambda_pref": float(q["lambda_pref"]),
                   "lambda_station": float(q["lambda_station"]), "lambda_ev": float(q["lambda_ev"]),
                   "corridor_version": int(q["corridor_version"]),
                   "kd_parent_only": _b(q["kd_parent_only"]), "corrector": _b(q["corrector"]),
                   "corr_k": int(q["corr_k"]), "lambda_awr": float(q["lambda_awr"])}
        cast = {"exec_model": str, "ev_tbins": int, "ev_tbin_s": float, "ep_evidence": _b, "vmax": _mode_str,
                "edits": _mode_str, "edit_k": int, "edit_dt": float, "edit_dlat": float, "kd_score_scope": str,
                "rg_read": str, "rg_stopped_aware": _b}
        for k_, f_ in cast.items():                    # refiner v4_3 optional keys: only those given (_d_v4_opt)
            if k_ in q:
                self.v4[k_] = f_(q[k_])
        if "kd_constraint" in q:
            kq = dict(_d_kd_constraint(), **dict(q["kd_constraint"] or {}))
            bad = sorted(set(kq) - set(_d_kd_constraint()))
            if bad:
                raise ValueError(f"ck_e2e2.v4.kd_constraint: unknown key(s) {bad}")
            self.v4["kd_constraint"] = {"enabled": _b(kq["enabled"]), "lambda_occ": float(kq["lambda_occ"]),
                                        "lambda_margin": float(kq["lambda_margin"]),
                                        "map_range_m": float(kq["map_range_m"]), "occ_channels": str(kq["occ_channels"])}
        if "rg_gt_aux" in q:
            gq = dict(_d_rg_gt_aux(), **dict(q["rg_gt_aux"] or {}))
            bad = sorted(set(gq) - set(_d_rg_gt_aux()))
            if bad:
                raise ValueError(f"ck_e2e2.v4.rg_gt_aux: unknown key(s) {bad}")
            self.v4["rg_gt_aux"] = {"enabled": _b(gq["enabled"]), "lambda": float(gq["lambda"]),
                                    "max_parents": int(gq["max_parents"])}
        if "v5" in q:                                  # refiner v5 student block (_d_v5)
            vq = dict(_d_v5(), **dict(q["v5"] or {}))
            bad = sorted(set(vq) - set(_d_v5()))
            if bad:
                raise ValueError(f"ck_e2e2.v4.v5: unknown key(s) {bad}")
            if str(vq["relation"]) not in ("off", "student"):
                raise ValueError(f"ck_e2e2.v4.v5.relation {vq['relation']!r} not in ('off', 'student')")
            self.v4["v5"] = {"enabled": _b(vq["enabled"]), "relation": str(vq["relation"]), "n_obj": int(vq["n_obj"]),
                             "score_thr": float(vq["score_thr"]), "unknown_ev": _b(vq["unknown_ev"]),
                             "corrector": _b(vq["corrector"]), "z_stride": int(vq["z_stride"]),
                             "lambda_ref": float(vq["lambda_ref"]), "ref_w": "" if vq["ref_w"] in (None, False) else
                             str(vq["ref_w"]), "ref_cols": int(vq["ref_cols"]), "lambda_rel_a": float(vq["lambda_rel_a"]),
                             "lambda_rel_h": float(vq["lambda_rel_h"]), "rel_h_cols": int(vq["rel_h_cols"]),
                             "z_kd": _b(vq["z_kd"]), "lambda_zkd_det": float(vq["lambda_zkd_det"]),
                             "lambda_zkd_map": float(vq["lambda_zkd_map"]), "infer_refine": _b(vq["infer_refine"])}
        if self.infer_select is not None:
            s = dict(self.infer_select)
            self.infer_select = {"beta": float(s["beta"]), "set": str(s["set"]), "lat_mode": str(s["lat_mode"])}
            if "edits" in s:                 # optional (absent: the old dict exactly)
                self.infer_select["edits"] = _b(s["edits"])
        self.io_dir = "" if self.io_dir is None else str(self.io_dir)
        self.teacher_det_run = "" if self.teacher_det_run in (None, "none", "None") else str(self.teacher_det_run)
        self.teacher_map_run = "" if self.teacher_map_run in (None, "none", "None") else str(self.teacher_map_run)
        self.teacher_which = str(self.teacher_which)
        self.var_sampling = str(self.var_sampling)
        self.score_prior = str(self.score_prior)
        self.ep_target = str(self.ep_target)
        if self.ep_target not in EP_TARGETS:
            raise ValueError(f"ck_e2e2.ep_target must be one of {EP_TARGETS}, got {self.ep_target!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def v4_opt(self, key: str):
        """an optional v4_3 key of the v4 block (_d_v4_opt), its default when absent."""
        return self.v4[key] if key in self.v4 else _d_v4_opt()[key]

    @property
    def kd_score_scope(self) -> str:
        """refiner v4_3 M6 score-KD scope (v4.kd_score_scope; 'all' when absent)."""
        return str(self.v4_opt("kd_score_scope"))

    @property
    def kd_constraint(self) -> Dict[str, Any]:
        """refiner v4_3 M6 constraint KD block (v4.kd_constraint; off when absent)."""
        return self.v4_opt("kd_constraint")

    @property
    def rg_gt_aux(self) -> Dict[str, Any]:
        """refiner v4_3 (c) student GT range aux block (v4.rg_gt_aux; off when absent)."""
        return self.v4_opt("rg_gt_aux")

    @property
    def v5(self) -> Dict[str, Any]:
        """refiner v5 student block (v4.v5; off when absent)."""
        return self.v4_opt("v5")

    def v5_net(self):
        """V5Cfg of the student network (off unless v4.v5.enabled)."""
        from .v5.config import V5Cfg
        q = self.v5
        if not q["enabled"]:
            return V5Cfg()
        return V5Cfg(relation="student" if q["relation"] != "off" else "off", n_obj=q["n_obj"], rel_ca=False,
                     unknown_ev=q["unknown_ev"], corrector=q["corrector"], z_stride=q["z_stride"])

    def v4_net(self):
        """V4Cfg of the student network (ck/v4/config.py; the v4 block's network keys + the optional v4_3 ones given)."""
        from .v4.config import V4Cfg
        return V4Cfg.from_dict({k: self.v4[k] for k in V4_NET_KEYS + V4_NET_OPT_KEYS if k in self.v4})

    @property
    def compute_dtype(self) -> torch.dtype:
        return {"float64": torch.float64, "float32": torch.float32}[self.variants["compute_dtype"]]

    # ------------------------------------------------------------------------------------------ validate
    def validate(self, parent=None) -> None:
        """Only when enabled.  parent: the ParaSSRConfig (old CK e2e rules + CK2 rules, SPEC s7-1)."""
        if not self.enabled:
            return
        if parent is not None:
            if getattr(parent, "refiner_mode", "off") != "off":
                raise ValueError("ck_e2e2 needs refiner_mode off (Stage E and CK2 e2e are exclusive)")
            old = getattr(parent, "ck_e2e", None) or {}
            old_on = old.get("enabled", False) if hasattr(old, "get") else getattr(old, "enabled", False)
            if _b(old_on):
                raise ValueError("ck_e2e and ck_e2e2 cannot both be enabled")
            if not getattr(parent, "plan_anchor", False):
                raise ValueError("ck_e2e2 needs plan_anchor=true (top-16 of the anchor planner)")
            if (int(parent.bev_h), int(parent.bev_w)) != (50, 100) or tuple(float(v) for v in parent.pc_range) != (
                    -32.0, 0.0, -2.0, 32.0, 32.0, 2.0):
                raise ValueError("ck_e2e2 reads the 50 x 100 S grid (0.64 m, front ROI)")
            if int(parent.embed_dims) != 256:
                raise ValueError(f"ck_e2e2 needs embed_dims 256, got {parent.embed_dims}")
        if self.topk != K16:
            raise ValueError(f"ck_e2e2.topk must be {K16} (96-column generation layout), got {self.topk}")
        v = self.variants
        try:
            C2.variants_cfg(v)        # 6-column 'separate' table in VNAMES order, ext const_curv | straight, dtype
        except ValueError as e:
            raise ValueError(f"ck_e2e2.variants: {e}") from None
        for key, n in (("n_var_step", self.n_var_step), ("warmup.n_var", self.warmup["n_var"]),
                       ("onpolicy_label.n_var", self.onpolicy_label["n_var"])):
            if not 0 <= n <= K16 * (NV - 1):
                raise ValueError(f"ck_e2e2.{key} must be in [0, {K16 * (NV - 1)}], got {n}")
        if self.onpolicy_label["n_orig"] != K16:
            raise ValueError(f"ck_e2e2.onpolicy_label.n_orig must be {K16} (all identities of the generation)")
        if self.onpolicy_label["fallback"] not in ("warmup", "none"):
            raise ValueError("ck_e2e2.onpolicy_label.fallback must be warmup | none")
        if self.var_sampling not in VAR_SAMPLING:
            raise ValueError(f"ck_e2e2.var_sampling must be one of {VAR_SAMPLING}, got {self.var_sampling!r}")
        if self.score_prior not in SCORE_PRIORS:
            raise ValueError(f"ck_e2e2.score_prior must be one of {SCORE_PRIORS}, got {self.score_prior!r}")
        for k in ("sur_groups", "lat_kd_groups"):
            if self.warmup[k] not in SUR_GROUP_MAX:
                raise ValueError(f"ck_e2e2.warmup.{k} must be one of {tuple(SUR_GROUP_MAX)}, got {self.warmup[k]!r}")
        if self.record_from_epoch > self.onpolicy_from_epoch:
            raise ValueError(f"ck_e2e2: record_from_epoch {self.record_from_epoch} > onpolicy_from_epoch "
                             f"{self.onpolicy_from_epoch} (on-policy needs recorded labels)")
        if self.label_lag < 1:
            raise ValueError(f"ck_e2e2.label_lag must be >= 1, got {self.label_lag}")
        if not self.io_dir or not os.path.isabs(self.io_dir) or "=" in self.io_dir:
            raise ValueError(f"ck_e2e2.io_dir must be an absolute path without '=', got {self.io_dir!r}")
        lk = self.lat_kd
        if lk["balance"] not in ("ema", "fixed"):
            raise ValueError(f"ck_e2e2.lat_kd.balance must be ema | fixed, got {lk['balance']!r}")
        if not (0.0 < lk["m"] < 1.0 and lk["floor"] > 0.0 and lk["ratio"] >= 0.0 and lk["fixed"] >= 0.0):
            raise ValueError(f"ck_e2e2.lat_kd needs 0 < m < 1, floor > 0, ratio >= 0, fixed >= 0, got {lk}")
        if self.rec_chunk_tokens < 1 or self.log_every < 1:
            raise ValueError("ck_e2e2.rec_chunk_tokens and log_every must be >= 1")
        if not 0.0 <= self.strict_rows <= 1.0:
            raise ValueError(f"ck_e2e2.strict_rows must be in [0, 1], got {self.strict_rows}")
        if self.infer_select is not None:
            if not 0.0 <= self.infer_select["beta"] <= 1.0:
                raise ValueError("ck_e2e2.infer_select.beta must be in [0, 1]")
            emits = bool(self.v4["corrector"]) or self.v4_opt("edits") == "computed"
            if "edits" not in self.infer_select and emits:
                raise ValueError("ck_e2e2.infer_select: this student emits edits (v4.corrector / v4.edits computed); "
                                 "set infer_select.edits true (96 + K, eval_ck2's headline rule with the val-chosen "
                                 "beta / set) or false (the 96 pool only)")
            if self.infer_select.get("edits") and not emits:
                raise ValueError("ck_e2e2.infer_select.edits true needs a student with edits (v4.corrector or "
                                 "v4.edits computed)")
        bk = self.bev_kd
        if bk["enabled"]:
            from .bev_kd import DISTANCES, INITS, TEACHERS
            if not bk["teachers"] or any(t not in TEACHERS for t in bk["teachers"]) or \
                    len(set(bk["teachers"])) != len(bk["teachers"]):
                raise ValueError(f"ck_e2e2.bev_kd.teachers must be a non-empty subset of {TEACHERS}, got "
                                 f"{bk['teachers']}")
            if bk["distance"] not in DISTANCES or bk["init"] not in INITS:
                raise ValueError(f"ck_e2e2.bev_kd distance / init must be in {DISTANCES} / {INITS}")
            if not (bk["ratio"] > 0 and bk["cap"] > 0 and 0.0 < bk["m"] < 1.0 and bk["adapter_weight"] > 0):
                raise ValueError(f"ck_e2e2.bev_kd needs ratio, cap, adapter_weight > 0 and 0 < m < 1, got {bk}")
            if bk["w_max"] is not None and not bk["w_max"] > 0:
                raise ValueError("ck_e2e2.bev_kd.w_max must be > 0 or null")
            for k in BEV_KD_PER_TEACHER:
                if bk[k] is None:
                    continue
                if not bk[k] > 0:
                    raise ValueError(f"ck_e2e2.bev_kd.{k} must be > 0 or null, got {bk[k]}")
                if k.split("_", 1)[1] not in bk["teachers"]:
                    raise ValueError(f"ck_e2e2.bev_kd.{k} is set but teacher {k.split('_', 1)[1]!r} is not in "
                                     f"bev_kd.teachers {bk['teachers']}")
        q = self.v4
        try:
            self.v4_net()
        except ValueError as e:
            raise ValueError(f"ck_e2e2.v4: {e}") from None
        from ..refiner.corridor import CORRIDOR_VERSIONS
        if q["corridor_version"] not in CORRIDOR_VERSIONS:
            raise ValueError(f"ck_e2e2.v4.corridor_version must be in {CORRIDOR_VERSIONS}, got {q['corridor_version']}")
        if q["corrector"] and q["corr_k"] != 8:
            raise ValueError("ck_e2e2.v4.corrector: corr_k must be 8 (= e2e_data2.EDIT_K, the recorded edit block)")
        if self.v4_opt("edits") == "computed" and self.v4_opt("edit_k") != 8:
            raise ValueError("ck_e2e2.v4.edits computed: edit_k must be 8 (= e2e_data2.EDIT_K, the recorded edit block)")
        if q["lambda_pref"] < 0 or q["lambda_station"] < 0 or q["lambda_ev"] < 0 or q["lambda_awr"] < 0:
            raise ValueError(f"ck_e2e2.v4 lambda_pref / lambda_station must be >= 0, got {q}")
        if self.kd_score_scope not in KD_SCORE_SCOPES:
            raise ValueError(f"ck_e2e2.v4.kd_score_scope must be one of {KD_SCORE_SCOPES}, got {self.kd_score_scope!r}")
        kq = self.kd_constraint
        if kq["enabled"]:
            if q["readout"] != "range":
                raise ValueError(f"ck_e2e2.v4.kd_constraint needs the student readout 'range' (its B / U / D fields), got "
                                 f"v4.readout {q['readout']!r}")
            if not (kq["lambda_occ"] >= 0 and kq["lambda_margin"] >= 0 and kq["map_range_m"] > 0):
                raise ValueError(f"ck_e2e2.v4.kd_constraint needs lambda_occ, lambda_margin >= 0 and map_range_m > 0, got "
                                 f"{kq}")
            if kq["occ_channels"] not in OCC_CHANNELS:
                raise ValueError(f"ck_e2e2.v4.kd_constraint.occ_channels must be one of {OCC_CHANNELS}, got "
                                 f"{kq['occ_channels']!r}")
            if "Bp" in CN_GROUPS[kq["occ_channels"]] and not _b(self.v4_opt("rg_stopped_aware")):
                raise ValueError(f"ck_e2e2.v4.kd_constraint.occ_channels {kq['occ_channels']!r} distils the Bp group: "
                                 f"R2 only (needs v4.rg_stopped_aware true; 'B' / 'BU' otherwise)")
        gq = self.rg_gt_aux
        if gq["enabled"]:
            if q["readout"] != "range":
                raise ValueError(f"ck_e2e2.v4.rg_gt_aux needs the student readout 'range' (its B / U / D fields), got "
                                 f"v4.readout {q['readout']!r}")
            if not (gq["lambda"] >= 0 and gq["max_parents"] >= 1):
                raise ValueError(f"ck_e2e2.v4.rg_gt_aux needs lambda >= 0 and max_parents >= 1, got {gq}")
        kc = self.kd_calib
        if kc["enabled"]:
            if all(kc[t] is None for t in KD_CALIB_TEACHERS):
                raise ValueError("ck_e2e2.kd_calib.enabled needs a calibration file for det and / or map")
            for t in KD_CALIB_TEACHERS:
                if kc[t] is not None and (not os.path.isabs(kc[t]) or "=" in kc[t]):
                    raise ValueError(f"ck_e2e2.kd_calib.{t} must be an absolute path without '=' or null, got "
                                     f"{kc[t]!r}")
        # teachers: both runs required (the lateral KD target is the mean of the DET and MAP offsets)
        for key, want in (("teacher_det_run", "T"), ("teacher_map_run", "M")):
            run = getattr(self, key)
            if not run:
                raise ValueError(f"ck_e2e2.{key} is required (lateral KD target = mean of DET and MAP)")
            cj = Path(run) / "config.json"
            if cj.is_file():
                arm = json.loads(cj.read_text()).get("arm")
                if arm != want:
                    raise ValueError(f"ck_e2e2.{key} {run}: config.json arm {arm!r}, expected {want!r}")
                # refiner-v3 temporal teacher (its own config.json; off for every CK2 run): DET only, never privileged
                tc = teacher_temporal(run)
                if tc.on:
                    if key != "teacher_det_run":
                        raise ValueError(f"ck_e2e2.{key} {run}: temporal branch on a MAP teacher is not supported")
                    if tc.privileged:
                        raise ValueError(f"ck_e2e2.{key} {run}: PRIVILEGED temporal run (forecaster {tc.forecaster!r}, "
                                         f"source {tc.source!r}) can never be a KD teacher")
                    try:
                        tc.validate(want)
                    except (ValueError, NotImplementedError) as e:
                        raise ValueError(f"ck_e2e2.{key} {run}: temporal cfg: {e}") from None
        # the warm-up variant label files must be the same variant definition (speeds, lats, combine, s_on_frac,
        # names, accel extension)
        if (Path(self.warmup["var_dir"]) / "build.json").is_file():
            try:
                C2.check_variant_files(self.warmup["var_dir"], v)
            except ValueError as e:
                raise ValueError(f"ck_e2e2.variants vs warm-up label files: {e}") from None


# ----------------------------------------------------------------------------------------------- student
def freeze_lon_gate(net) -> List[str]:
    """CK2: zero the last layer of trunk.lon_head (z_lon == 0 for every loader of the checkpoint) and freeze
    trunk.lon_head + trunk.gate_head (never in a loss; out of the optimiser and the DDP reducer).  Same rule as
    tools/ck/train_ck2.freeze_unused (re-implemented: navsim does not import tools).  -> frozen parameter names."""
    lh = net.trunk.lon_head
    with torch.no_grad():
        lh[-1].weight.zero_()
        lh[-1].bias.zero_()
    names = []
    for mod_name in ("trunk.lon_head", "trunk.gate_head"):
        mod = net.get_submodule(mod_name)
        for n, p in mod.named_parameters():
            p.requires_grad_(False)
            names.append(f"{mod_name}.{n}")
    return names


def score_prior_of(cfg: CKE2E2Config) -> Tuple[Optional[np.ndarray], str]:
    """Score-head bias prior: 'ck2_warmup' = e2e_data2.ck2_warmup_prior (expected label mean of the warm-up mix),
    'teacher' = config.json init_report.score_prior of the DET teacher run, 'none' -> (None, 'none')."""
    if cfg.score_prior == "ck2_warmup":
        from .e2e_data2 import ck2_warmup_prior
        p = ck2_warmup_prior(cfg, n_max=int(cfg.score_prior_rows))
        return (None if p is None else np.asarray(p, np.float64)), "ck2_warmup"
    if cfg.score_prior == "teacher":
        cj = Path(cfg.teacher_det_run) / "config.json"
        if cj.is_file():
            p = (json.loads(cj.read_text()).get("init_report") or {}).get("score_prior")
            if p is not None and len(p) == len(CK_KEYS):
                return np.clip(np.asarray(p, np.float64), 1e-3, 1 - 1e-3), f"teacher:{cj}"
        return None, "teacher (not found)"
    return None, "none"


def build_student_ck2(cfg: CKE2E2Config, apply_prior: bool = True):
    """CKNet('S', seed, score_hidden 256, lead_aux False), adapter.bev_grad_scale = cfg.bev_grad_scale, lon / gate
    heads frozen (lon zeroed), score-head bias = logit(prior) unless apply_prior is False (a checkpoint will be
    loaded: no file is opened then).  No global RNG is consumed (CKNet forks the RNG)."""
    from .model import CKNet

    cfg = CKE2E2Config.from_any(cfg)
    kw: Dict[str, Any] = {}                    # refiner v4 / refiner-fix B1 (defaults: the CK2 call, unchanged)
    if cfg.v4["corridor_version"] != 1:
        kw["corridor_version"] = int(cfg.v4["corridor_version"])
    if cfg.v4_net().on:
        kw["v4"] = cfg.v4_net()
    if cfg.v5_net().on:                        # refiner v5 student modules (off: the call unchanged)
        kw["v5"] = cfg.v5_net()
    net = CKNet("S", int(cfg.seed), score_hidden=256, lead_aux=False, **kw)
    net.trunk.adapter.bev_grad_scale = float(cfg.bev_grad_scale)
    frozen = freeze_lon_gate(net)
    rep: Dict[str, Any] = {"source": None, "score_prior": None, "score_prior_source": None, "frozen": frozen}
    if apply_prior and cfg.score_prior != "none":
        prior, src = score_prior_of(cfg)
        rep["score_prior_source"] = src
        if prior is not None:
            net.set_score_prior(prior)
            rep["score_prior"] = [float(v) for v in prior]
    net.init_report = rep
    return net


# ----------------------------------------------------------------------------------------------- decode / variants
def lateral_decode(cand: torch.Tensor, w_lat: torch.Tensor, v0: torch.Tensor, slope: float) -> Dict:
    """correct(cand, z_lon = 0, w_lat, v0, slope): the learned longitudinal head is removed in CK2."""
    from .correct import correct

    w = w_lat.float()
    return correct(cand.float(), torch.zeros_like(w), w, v0, slope)


def _parents_or_self(cand: torch.Tensor, parent: Optional[torch.Tensor]) -> torch.Tensor:
    return cand.float() if parent is None else parent.float()


def split_haz(haz: Dict[str, torch.Tensor], lo: int, hi: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """The score_set 'haz' dict of the candidate columns [lo, hi) of a set: per-candidate tensors sliced on dim 1, the
    range readout's per-PARENT-row fields (RANGE_U_KEYS, [T, U, ...]) kept whole -- their rows are what parent_index
    [T, K] (sliced like a candidate tensor) points at."""
    return {k: (v if (k in RANGE_U_KEYS or k in V5_TOKEN_KEYS) else v[:, lo:hi]) for k, v in haz.items()}


def _student_forward_v4(student, feat, mem, ego, cand, extra, cand_parent, extra_parent,
                        return_pctx: bool = False, rel_objs: Optional[Dict[str, torch.Tensor]] = None) -> Dict:
    """student_forward2 for a refiner v4 student (hazard readout and / or P_ref head): ONE score_set over cat(cand,
    extra) with cat(parents) (None = itself) -> the student_forward2 keys + haz / extra_haz (split_haz: [B, K(2), ...]
    candidate tensors, the range readout's per-parent fields whole) + pref_logp [B] (None without the P_ref head).
    return_pctx (refiner v4_3 computed edits, recording): + 'pctx' / 'parent_index_all' of that score_set (the parent
    pass of the whole set; edits re-scored on it) and 'scene'."""
    B, K = cand.shape[:2]
    allc = cand.float() if extra is None else torch.cat([cand.float(), extra.float()], 1)
    allp = None
    if student.v4_readout is not None and (cand_parent is not None or extra_parent is not None):
        allp = _parents_or_self(cand, cand_parent)
        if extra is not None:
            allp = torch.cat([allp, _parents_or_self(extra, extra_parent)], 1)
    pref = student.scene_pref(mem, ego)
    corr = (getattr(student, "corrector", None) is not None
            or getattr(student, "corrector_v5", None) is not None)       # v4 corrector A / v5 corrector A_S
    kw = {"return_pctx": True} if return_pctx else {}                 # off: the old call exactly
    if rel_objs is not None:                                          # refiner v5 student relation channel
        kw["rel_objs"] = rel_objs
    o = student.score_set(feat, mem, ego, allc, allp, pref=pref, return_ctx=corr, **kw)
    sc, w = o["score_logit"], o["w_lat"].float()
    out = {"score_logit": sc[:, :K], "extra_score_logit": None if extra is None else sc[:, K:],
           "w_lat": w[:, :K], "w_lat_extra": None if extra is None else w[:, K:], "score_all": sc, "w_lat_all": w,
           "ego": ego, "pref_logp": pref}
    if "haz" in o:
        out["haz"] = split_haz(o["haz"], 0, K)
        out["extra_haz"] = None if extra is None else split_haz(o["haz"], K)
    if corr:                                          # corrector A: its context and the scene (recording / AWR)
        out["ctx_all"] = o["ctx"]
        out["scene"] = (feat, mem, ego, pref)
    if return_pctx:                                   # v4_3 computed edits: the set's parent pass (recording)
        out["pctx"], out["parent_index_all"] = o["pctx"], o["parent_index"]
        out["scene"] = (feat, mem, ego, pref)
    return out


def label_set_parents(G: Dict[str, torch.Tensor], targets, sampler, dev) -> torch.Tensor:
    """Parent drafts [B, KG, 8, 3] of the on-policy label set G (refiner v4 hazard readout):
      generation rows (has): identities (G_vtype 0) are their own parents, a variant's parent is the identity of its
        top-16 rank k = G_col // 6, i.e. G_traj[:, k] (LabelStore2.lookup puts the 16 identities first in k order);
      fallback rows (cands2.fill_fallback): anchors are their own parents, a variant's parent is the anchor of its
        variant-file column (targets ck2_var_anchor[col // 6] -> the AnchorSampler anchor table);
      rows without data: themselves."""
    traj = G["G_traj"].to(dev).float()
    B, KG = traj.shape[:2]
    vt = G["G_vtype"].to(dev).long()
    col = G["G_col"].to(dev).long()
    has = G["has"].to(dev).bool().reshape(B, 1)
    var = vt > 0
    k = (col.clamp(min=0) // NV).clamp(max=K16 - 1)
    gen_par = torch.gather(traj, 1, k[..., None, None].expand(B, KG, *traj.shape[2:]))
    anc = sampler.anchors_t(dev)
    va = targets["ck2_var_anchor"].to(dev).long()                                 # [B, 16] anchor ids
    fb_par = anc[torch.gather(va, 1, k)]                                          # [B, KG, 8, 3]
    par = torch.where((var & has)[..., None, None], gen_par, traj)
    par = torch.where((var & ~has & (col >= 0))[..., None, None], fb_par.to(traj.dtype), par)
    return par


def _keep_parent(cand_s: torch.Tensor, parent: torch.Tensor, good: torch.Tensor) -> torch.Tensor:
    """Candidates replaced by the fallback (not good) are their own parents; non-finite parents too."""
    g = good.bool() & torch.isfinite(parent).flatten(2).all(-1)
    return torch.where(g[..., None, None], parent.float(), cand_s.float())


def infer_pool(traj96: torch.Tensor, top16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The 96 pool as CKE2E2.infer scores it (also tools/ck/e2e2/eval_e2e2.stage_infer_check): traj96 [B, 96, 8, 3]
    (c = k * 6 + v, may hold NaN), top16 [B, 16, 8, 3] (the identities, traj96[:, ::6]) -> (candidates with
    non-finite ones replaced by v2's trajectory top16[:, 0], finite mask [B, 96], each candidate's parent draft
    [B, 96, 8, 3] (its rank's identity; a replaced candidate is its own parent))."""
    cand, fin = _sanitize_traj(traj96, top16[:, 0])
    return cand, fin, _keep_parent(cand, top16.float().repeat_interleave(NV, 1), fin)


def select96_edits(v2_final16, v2_im16, prob96, valid96, prob_e, src_e, use_e, beta: float,
                   types: Optional[Sequence[int]] = None, offset=None, w=None) -> np.ndarray:
    """Selection over the 96 pool + K edits (refiner v4 corrector A / v4_3 computed edits; the ONE implementation of
    tools/ck/e2e2/eval_ck2.select_edits and of infer_select.edits): v2_final16 / v2_im16 [N, 16], prob96 [N, 96, 5]
    and prob_e [N, K, 5] (sigmoid scores, CK_KEYS order), valid96 [N, 96], src_e [N, K] (96-column of each edit's
    source), use_e [N, K] (selectable).  The 96 score as select2.select96 (allowed columns of `types`, default the 'all'
    set); edit j scores blend(v2_final[src // 6], ck_final(p_e, im[src // 6]), beta) (the source's parent's v2 final /
    im).  -> idx [N] int64: < 96 = a pool column (select96's choice when no edit wins), 96 + j = edit slot j.  Ties ->
    the pool (identities first, select2.TIE_ORDER), then the lower edit slot.  numpy in / out."""
    from . import select2 as S2
    from .constants import SEL_W
    from .select import blend, ck_final
    types = S2.VARIANT_SETS["all"] if types is None else types
    w = SEL_W if w is None else w
    s96 = S2.pool_scores(v2_final16, v2_im16, prob96, beta, w, offset)
    ok = S2.allowed_mask(valid96, types) & np.isfinite(s96)
    s96 = np.where(ok, s96, -np.inf)[:, S2.TIE_ORDER]
    par = np.clip(np.asarray(src_e, np.int64), 0, S2.G_K - 1) // S2.NV
    v2f = np.take_along_axis(np.asarray(v2_final16, np.float64), par, 1)
    im = np.take_along_axis(np.asarray(v2_im16, np.float64), par, 1)
    se = blend(v2f, ck_final(S2.apply_offset(np.asarray(prob_e, np.float64), offset), im, w), beta)
    se = np.where(np.asarray(use_e, bool) & np.isfinite(se), se, -np.inf)
    j = np.argmax(np.concatenate([s96, se], 1), 1)
    return np.where(j < S2.G_K, S2.TIE_ORDER[np.minimum(j, S2.G_K - 1)], j).astype(np.int64)


# ----------------------------------------------------------------------------------------------- teachers
def is_range_net(net) -> bool:
    """a CK net with the refiner v4_3 'range' readout (its B / U / D fields per parent)."""
    v4 = getattr(net, "v4", None)
    return getattr(net, "v4_readout", None) is not None and v4 is not None and bool(getattr(v4, "range", False))


def net_field_groups(v4) -> Tuple[str, ...]:
    """the blocked-field groups a range readout of V4Cfg v4 outputs: B, U (+ Bp with rg_stopped_aware)."""
    return ("B", "U", "Bp") if bool(getattr(v4, "rg_stopped_aware", False)) else ("B", "U")


def field_groups(spec) -> Tuple[str, ...]:
    """A teacher field request (teacher_forward return_fields / TeacherPair2.run fields) -> the group names in order:
    True = FIELDS_TRUE ('B', 'U': the original dict), an occ_channels name (CN_GROUPS: 'B' / 'BU' / 'BBp' / 'all') or
    a sequence of distinct 'B' / 'U' / 'Bp' (ValueError otherwise)."""
    if spec is True:
        return FIELDS_TRUE
    if isinstance(spec, str):
        if spec not in CN_GROUPS:
            raise ValueError(f"field groups {spec!r} not in {OCC_CHANNELS}")
        return CN_GROUPS[spec]
    g = tuple(str(x) for x in spec)
    if not g or len(set(g)) != len(g) or any(x not in FIELD_GROUP_KEY for x in g):
        raise ValueError(f"field groups {spec!r}: distinct names of {tuple(FIELD_GROUP_KEY)}")
    return g


def teacher_forward(net, bev: torch.Tensor, cand: torch.Tensor, ego, objs: Optional[Dict[str, torch.Tensor]] = None,
                    parents: Optional[torch.Tensor] = None, return_fields=False,
                    rel_objs: Optional[Dict[str, torch.Tensor]] = None, extras: Optional[Dict] = None):
    """The KD teacher ops under the caller's autocast / no_grad: trunk.scene once, trunk.candidates on `cand`, score
    -> (score logits [B,K,5] f32, w_lat [B,K,6]).  A temporal teacher (net.temporal, refiner-v3) additionally runs
    TemporalBranch.scene(objs) once and binds its hook to this candidate set -- the order and ops of
    CKNet.forward(bev, cand, status, decode=False, objs=objs) (train_ck2.forward / infer_val_v3).  objs must be given
    iff the teacher has a temporal branch (ValueError otherwise: a silently dropped branch / stray input).
    return_fields (refiner v4_3 constraint KD; range teachers only, ValueError otherwise): True (the B | U groups, the
    original dict), an occ_channels name ('B' / 'BU' / 'BBp' / 'all') or a sequence of group names (field_groups;
    'Bp' only from an rg_stopped_aware teacher, ValueError otherwise) -> (score, w_lat, fields): fields = {blk_u
    [B, U, 48, 17, G M] f32 (the requested groups' logits of the teacher's unique parents of `cand`, concatenated in
    the request's order: B = occ_logit_u (the teacher's own B: the stopped-aware one with rg_stopped_aware), U =
    ovl_logit_u, Bp = bp_logit_u), groups (the G names, that order), margin_u [B, U, 48, 17] (m), parent_index [B, K]
    (candidate -> row), s_u [B, U, 48] (station arc lengths, m)} of the same score_set (scores unchanged by the extra
    outputs)."""
    B, K = cand.shape[:2]
    temporal = getattr(net, "temporal", None)
    if temporal is None and objs is not None:
        raise ValueError("teacher_forward: objs given to a teacher without a temporal branch")
    if temporal is not None and objs is None:
        raise ValueError("teacher_forward: this teacher has a temporal branch: its causal object rows are required")
    if return_fields and not is_range_net(net):
        raise ValueError("teacher_forward: range fields requested from a teacher without the v4_3 'range' readout "
                         f"(v4.readout {getattr(getattr(net, 'v4', None), 'readout', 'mean')!r}): constraint KD "
                         "needs range teachers")
    groups = field_groups(return_fields) if return_fields else None
    if groups is not None:
        miss = [g for g in groups if g not in net_field_groups(net.v4)]
        if miss:
            raise ValueError(f"teacher_forward: field group(s) {miss} requested from a teacher with the groups "
                             f"{net_field_groups(net.v4)} (Bp: rg_stopped_aware teachers only)")
    feat, mem = net.trunk.scene(bev, B)
    if getattr(net, "v4_readout", None) is not None:    # refiner v4 teacher: hazard readout on the parents' corridors
        kw = {"return_pctx": True} if return_fields else {}
        if rel_objs is not None:                        # refiner v5 relation DET teacher: its object pack rows
            kw["rel_objs"] = rel_objs
        o = net.score_set(feat, mem, ego, cand, parents, pref=net.scene_pref(mem, ego), **kw)
        if extras is not None and "zq" in (o.get("haz") or {}):   # refiner v5: per-state reads (Z-KD targets)
            extras["zq"] = o["haz"]["zq"].float()
        if not return_fields:
            return o["score_logit"], o["w_lat"]
        h = o["haz"]
        U = h["occ_logit_u"].shape[1]
        fields = {"blk_u": torch.cat([h[FIELD_GROUP_KEY[g]] for g in groups], -1).float(), "groups": groups,
                  "margin_u": h["margin_u"].float(), "parent_index": o["parent_index"].long(),
                  "s_u": o["pctx"]["rg_frame"]["s"].float().reshape(B, U, -1)}
        return o["score_logit"], o["w_lat"], fields
    tf = None if temporal is None else temporal.hook(temporal.scene(objs), B, K)
    o = net.trunk.candidates(feat, mem, cand, *ego, tfeat=tf)
    return net.score(o, B, K), o["w_lat"]


def kd_objs_of(targets, device=None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """(objs = {obj_in, obj_mask, frame_valid} -- exactly ck.temporal.config.OBJ_INPUT_KEYS --, obj_ok [B]) of the DET
    teacher from the e2e targets (e2e_data2.KDObjRows keys).  KeyError when the target builder did not emit them."""
    from .temporal.config import OBJ_INPUT_KEYS
    miss = [k for k in KD_OBJ_KEYS + (KD_OBJ_OK,) if k not in targets]
    if miss:
        raise KeyError(f"temporal DET teacher: targets lack {miss} (CK2E2ETargetBuilder without the teacher's object "
                       f"rows?)")
    objs = {KD_OBJ_INPUT[k]: (targets[k] if device is None else targets[k].to(device)) for k in KD_OBJ_KEYS}
    assert tuple(objs) == tuple(OBJ_INPUT_KEYS), (tuple(objs), OBJ_INPUT_KEYS)
    ok = targets[KD_OBJ_OK].reshape(-1).bool()
    return objs, (ok if device is None else ok.to(device))


def _tensor_sha16(sd: Dict[str, torch.Tensor]) -> str:
    import hashlib
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes() if v.dtype != torch.bfloat16 else v.float().numpy().tobytes())
    return h.hexdigest()[:16]


def temporal_teacher_info(net) -> Optional[Dict[str, Any]]:
    """Record of a teacher's temporal branch (None when off): cfg dict, sha16 of the loaded branch / forecaster state
    (the checkpoint's weights actually used) and of the cfg.fc_init file (pre-trained forecaster), ca_b."""
    t = getattr(net, "temporal", None)
    if t is None:
        return None
    tc = net.temporal_cfg
    rec: Dict[str, Any] = {"cfg": tc.to_dict(), "branch_state_sha16": _tensor_sha16(t.state_dict())}
    if t.forecaster is not None:
        rec["forecaster_state_sha16"] = _tensor_sha16(t.forecaster.state_dict())
    if getattr(t, "ca_b", None) is not None:
        rec["ca_b"] = [[float(x) for x in r] for r in t.ca_b.detach().cpu().tolist()]
    if tc.fc_init:
        p = Path(tc.fc_init)
        if p.is_file():
            import hashlib
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 22), b""):
                    h.update(b)
            rec["fc_init"] = {"path": str(p), "sha16": h.hexdigest()[:16]}
        else:
            rec["fc_init"] = {"path": str(p), "missing": True}
    return rec


class TeacherPair2:
    """Frozen CK2 DET (arm T) + MAP (arm M) teachers (ck.model.load_ck of tools/ck/train_ck2 runs), run online on the
    student's candidates.  Plain python object (not an nn.Module attribute of the agent: not in the state_dict / DDP /
    optimiser).  Load-time checks: config.json arm T / M, trainer 'train_ck2', done.json (require_done), lon head
    last layer all zero, and (ep_target given: the student's ck_e2e2.ep_target) config.json ep_target (missing =
    'official') == ep_target -- the same rule as the launch-time launch_util2.teacher_check, repeated where the
    teachers are actually loaded (report 48 S56-7).
    calib (ck_e2e2.kd_calib when enabled; None = no calibration, the old path): {'det': file | None, 'map': file |
    None} of tools/ck/e2e2/fit_kd_calib.py; each file must match its loaded teacher (kd_calib.check_calib: arm, run,
    which, ckpt sha16, ep_target, and every KD key of kd_keys this teacher is a source of fitted) or construction
    fails.  run() then uses z' = a_k z + b_k (per key) as the teacher's score logits.
    Temporal DET teacher (refiner-v3): load_ck rebuilds it from its config.json (refuses a PRIVILEGED run: the
    PermissionError is re-raised naming the teacher); a temporal MAP teacher is refused.  run(..., objs_det=,
    obj_ok_det=) is then required (teacher_forward; kd_obj_ok False -> ok False for the token); self.temporal_info
    records the cfg and the forecaster / branch state sha16 (None for a CK2 teacher).
    fields (refiner v4_3 constraint KD; default False = the old object): both teachers must have the 'range' readout
    (ValueError otherwise) and run(..., fields=True | groups) also returns their blocked-field groups (True: B | U, or
    the requested ones) and D fields (teacher_forward return_fields)."""

    def __init__(self, det_run: str, map_run: str, which: str = "last", device="cpu", amp: bool = True,
                 require_done: bool = True, ep_target: Optional[str] = None,
                 calib: Optional[Dict[str, Optional[str]]] = None, kd_keys: Optional[Sequence[str]] = None,
                 fields: bool = False):
        from .ep_target import check_ep_target, run_ep_target
        from .model import load_ck

        if not det_run or not map_run:
            raise ValueError("TeacherPair2 needs both the DET and the MAP run")
        self.device = torch.device(device)
        self.runs = {"det": str(det_run), "map": str(map_run)}
        nets, eps = {}, {}
        for name, run, arm in (("det", det_run, "T"), ("map", map_run, "M")):
            rd = Path(run)
            cj = rd / "config.json"
            if not cj.is_file():
                raise ValueError(f"teacher {name} {run}: no config.json")
            cfg = json.loads(cj.read_text())
            if cfg.get("arm") != arm:
                raise ValueError(f"teacher {name} {run}: arm {cfg.get('arm')!r}, expected {arm!r}")
            if cfg.get("trainer") != "train_ck2":
                raise ValueError(f"teacher {name} {run}: trainer {cfg.get('trainer')!r} != 'train_ck2' (CK2 teacher "
                                 f"required)")
            if require_done and not (rd / "done.json").is_file():
                raise ValueError(f"teacher {name} {run}: no done.json (training not finished; "
                                 f"teacher_require_done=false only for smoke runs / tests)")
            eps[name] = got = run_ep_target(cfg)
            if ep_target is not None:
                want = check_ep_target(ep_target)
                if got != want:
                    raise ValueError(f"teacher {name} {run}: ep_target {got!r} (config.json; missing = 'official') "
                                     f"!= the student's ck_e2e2.ep_target {want!r} (label BCE and EP KD would "
                                     f"disagree)")
            try:
                net, cfg = load_ck(run, which, device="cpu")       # refuses a PRIVILEGED (temporal oracle / gt) run
            except PermissionError as e:
                raise PermissionError(f"teacher {name} {run}: {e} -- never a KD teacher") from None
            if getattr(net, "temporal", None) is not None and name != "det":
                raise ValueError(f"teacher {name} {run}: temporal branch on a MAP teacher is not supported")
            lw = net.trunk.lon_head[-1]
            if bool(lw.weight.detach().abs().max() > 0) or bool(lw.bias.detach().abs().max() > 0):
                raise ValueError(f"teacher {name} {run}: trunk.lon_head last layer is not all zero (not a CK2 run)")
            net.requires_grad_(False)
            net.eval()
            nets[name] = (net.to(self.device), cfg)
        self.det, self.det_cfg = nets["det"]
        self.map, self.map_cfg = nets["map"]
        self.fields = bool(fields)
        if self.fields:                       # refiner v4_3 constraint KD: range teachers only
            bad = [f"{n} {self.runs[n]} (v4.readout {getattr(net.v4, 'readout', 'mean')!r})"
                   for n, net in (("det", self.det), ("map", self.map)) if not is_range_net(net)]
            if bad:
                raise ValueError("TeacherPair2: constraint KD needs teachers with the v4_3 'range' readout; refused: "
                                 + "; ".join(bad))
        self.amp = bool(amp)
        self.ep_target = eps                 # each teacher run's EP target (config.json)
        self.ckpt = {k: getattr(n, "init_report", {}).get("ckpt_sha16") for k, n in (("det", self.det),
                                                                                    ("map", self.map))}
        # refiner-v3: the DET teacher's temporal branch (None = CK2 teacher, the old path)
        self.det_temporal = getattr(self.det, "temporal", None) is not None
        # refiner v5: the DET teacher's object relation channel (pack rows as input, like a temporal teacher)
        self.det_rel = getattr(self.det, "relation", None) is not None
        if getattr(self.map, "relation", None) is not None:
            raise ValueError(f"teacher map {self.runs['map']}: a MAP teacher with the v5 relation channel")
        self.temporal_info = temporal_teacher_info(self.det)
        # KD calibration (Platt on the logits, per teacher and key); None = off (old path, nothing applied)
        self.calib: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
        self.calib_info: Dict[str, Any] = {}
        if calib is not None:
            from .kd_calib import load_teacher_calib
            bad = sorted(set(calib) - {"det", "map"})
            if bad:
                raise ValueError(f"TeacherPair2 calib: unknown teacher(s) {bad}")
            cal = {}
            for name, arm in (("det", "T"), ("map", "M")):
                path = calib.get(name)
                if not path:
                    continue
                try:
                    a, b, info = load_teacher_calib(path, run=self.runs[name], which=which,
                                                    ckpt_sha16=self.ckpt[name], ep_target=eps[name], arm=arm,
                                                    kd_keys=kd_keys, family=True)
                except (ValueError, FileNotFoundError) as e:
                    raise ValueError(f"teacher {name} {self.runs[name]}: KD calibration refused: {e}") from None
                cal[name] = (torch.from_numpy(a).to(self.device), torch.from_numpy(b).to(self.device))
                self.calib_info[name] = info
            self.calib = cal or None

    def nets(self):
        return [self.det, self.map]

    def to(self, device) -> "TeacherPair2":
        device = torch.device(device)
        if device != self.device:
            for n in self.nets():
                n.to(device)
            if self.calib is not None:
                self.calib = {k: (a.to(device), b.to(device)) for k, (a, b) in self.calib.items()}
            self.device = device
        return self

    @torch.no_grad()
    def run(self, bev_det, ok_det, bev_map, ok_map, cand, status, cand_ok=None, objs_det=None,
            obj_ok_det=None, parents: Optional[torch.Tensor] = None,
            family: Optional[torch.Tensor] = None, fields=False, rel_objs_det=None, rel_ok_det=None,
            zq: bool = False) -> Dict[str, torch.Tensor]:
        """bev_* [B, 256, 50, 100] (f16 cache), ok_* [B], cand [B, K, 8, 3] (finite), status [B, 8], cand_ok [B, K]
        -> kd_score_prob f32 [B,K,5] (combine_teacher: nc/ttc DET, dac MAP, ep/comfort mean), kd_e_lat f32 [B,K,6]
        = 0.5 (e_DET + e_MAP) (decode z_lon = 0, slope 0), kd_ok bool [B,K] = DET ok & MAP ok (per token) & finite
        logits / offsets of both & cand_ok; e_det / e_map [B,K,6], p_det / p_map [B,K,5] (logging).
        With calibration (self.calib) each teacher's logits are a_k z + b_k BEFORE the sigmoid / combination (p_det /
        p_map are then the calibrated probabilities); the lateral outputs are unchanged.
        Temporal DET teacher only (self.det_temporal): objs_det = {obj_in [B,3,96,17], obj_mask [B,3,96], frame_valid
        [B,3]} (kd_objs_of: exactly OBJ_INPUT_KEYS) and obj_ok_det [B] are required; DET ok = ok_det & obj_ok_det.
        A CK2 DET teacher: both must be None (the old path, bit for bit).
        fields (refiner v4_3 constraint KD; the pair must be built with fields=True): True (the B | U groups) or the
        requested groups (an occ_channels name / group names, field_groups; the same request to both teachers) ->
        + fields_det / fields_map (teacher_forward fields of the parents of `cand`, with their 'groups'; from the same
        forward, the scores are unchanged) and the per-token teacher ok ok_det_tok / ok_map_tok [B] (ok_det / ok_map
        as passed & finite returned fields; the temporal object-row ok never applies: range teachers have no temporal
        branch, is_range_net)."""
        from ..refiner.e2e import ego_inputs
        from .kd import combine_teacher

        if fields and not self.fields:
            raise ValueError("TeacherPair2.run(fields=True): the pair was built without fields=True (range teachers "
                             "not checked)")
        dev = self.device
        cand = cand.detach().float().to(dev)
        B, K = cand.shape[:2]
        if self.det_temporal:
            if objs_det is None or obj_ok_det is None:
                raise ValueError("TeacherPair2.run: the DET teacher has a temporal branch: objs_det and obj_ok_det "
                                 "(its causal object rows, kd_objs_of(targets)) are required")
            objs_det = {k: v.to(dev) for k, v in objs_det.items()}
            ok_det = ok_det.reshape(-1).bool().to(dev) & obj_ok_det.reshape(-1).bool().to(dev)
        elif objs_det is not None or obj_ok_det is not None:
            raise ValueError("TeacherPair2.run: object rows given but the DET teacher has no temporal branch")
        if self.det_rel:                      # refiner v5 relation DET teacher: its pack rows + their ok
            if rel_objs_det is None or rel_ok_det is None:
                raise ValueError("TeacherPair2.run: the DET teacher has the v5 relation channel: rel_objs_det and "
                                 "rel_ok_det (kd_objs_of(targets)) are required")
            rel_objs_det = {k: v.to(dev) for k, v in rel_objs_det.items()}
            ok_det = ok_det.reshape(-1).bool().to(dev) & rel_ok_det.reshape(-1).bool().to(dev)
        elif rel_objs_det is not None:
            raise ValueError("TeacherPair2.run: relation rows given but the DET teacher has no v5 relation channel")
        ego = ego_inputs(status.detach().float().to(dev))
        v0 = ego[0]
        ac = torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=self.amp and dev.type == "cuda")
        logits, ws, flds, exs = [], [], [], []
        for net, bev, objs, rel in ((self.det, bev_det, objs_det, rel_objs_det), (self.map, bev_map, None, None)):
            ex: Dict[str, torch.Tensor] = {}
            exs.append(ex)
            with ac:
                r = teacher_forward(net, bev.to(dev).float(), cand, ego, objs,
                                    None if parents is None else parents.detach().float().to(dev),
                                    **({"return_fields": fields} if fields else {}),
                                    **({"rel_objs": rel} if rel is not None else {}),
                                    **({"extras": ex} if zq else {}))
            logits.append(r[0].float())
            ws.append(r[1].float())
            if fields:
                flds.append(r[2])
        if self.calib is not None:            # KD calibration: Platt on the logits, per teacher and key
            for i, name in enumerate(("det", "map")):
                ab = self.calib.get(name)
                if ab is None:
                    continue
                if ab[0].dim() == 2:          # refiner v4 I2: per variant family (rows KD_FAMILIES) [6, 5]
                    if family is None:
                        raise ValueError(f"TeacherPair2.run: teacher {name} has a per-family KD calibration; the "
                                         f"candidates' family (vtype 0..5) is required")
                    f = family.to(dev).long().clamp(0, ab[0].shape[0] - 1)
                    logits[i] = logits[i] * ab[0][f] + ab[1][f]
                else:
                    logits[i] = logits[i] * ab[0] + ab[1]
        # ONE batched decode for both teachers (correct() is launch-bound; candidates decode independently)
        dec = lateral_decode(torch.cat([cand, cand], 1), torch.cat(ws, 1), v0, 0.0)
        e = dec["e_lat"][..., 2:].float()
        e_det, e_map = e[:, :K], e[:, K:]

        def tok_ok(ok_tok, s, ee):
            o = ok_tok.reshape(-1).bool().to(dev)[:, None].expand(-1, K)
            return o & torch.isfinite(s).all(-1) & torch.isfinite(ee).all(-1)

        det = {"score_logit": logits[0], "ok": tok_ok(ok_det, logits[0], e_det)}
        mp = {"score_logit": logits[1], "ok": tok_ok(ok_map, logits[1], e_map)}
        comb = combine_teacher(det, mp)
        kd_ok = comb["kd_ok"].bool()
        if cand_ok is not None:
            kd_ok = kd_ok & cand_ok.bool().to(dev)
        e_mean = 0.5 * (e_det + e_map)
        out = {"kd_score_prob": comb["kd_score_prob"].float(), "kd_e_lat": e_mean, "kd_ok": kd_ok,
               "e_det": e_det, "e_map": e_map, "p_det": torch.sigmoid(logits[0]), "p_map": torch.sigmoid(logits[1])}
        if zq:                                # refiner v5 Z-KD: the teachers' per-state reads
            for name, ex in zip(("det", "map"), exs):
                if "zq" not in ex:
                    raise ValueError(f"TeacherPair2.run(zq=True): teacher {name} {self.runs[name]} has no v5 per-state "
                                     f"reads (not a refiner v5 teacher)")
                out[f"zq_{name}"] = ex["zq"]
        if fields:                            # refiner v4_3 constraint KD
            for (name, okt), f in zip((("det", ok_det), ("map", ok_map)), flds):
                fin = (torch.isfinite(f["blk_u"]).flatten(1).all(1) & torch.isfinite(f["margin_u"]).flatten(1).all(1))
                out[f"fields_{name}"] = f
                out[f"ok_{name}_tok"] = okt.reshape(-1).bool().to(dev) & fin
        return out


# ----------------------------------------------------------------------------------------------- student forward
def student_forward2(student, bev_grid: torch.Tensor, cand: torch.Tensor, status: torch.Tensor,
                     extra: Optional[torch.Tensor] = None, cand_parent: Optional[torch.Tensor] = None,
                     extra_parent: Optional[torch.Tensor] = None, return_pctx: bool = False,
                     rel_objs: Optional[Dict[str, torch.Tensor]] = None) -> Dict:
    """One trunk.scene + ONE trunk.candidates over cat(cand, extra) (candidates are processed independently) ->
    score_logit [B,K,5] f32, extra_score_logit [B,K2,5] | None, w_lat [B,K,6], w_lat_extra [B,K2,6] | None,
    score_all / w_lat_all (every candidate), ego.  No decode here (callers use lateral_decode, z_lon = 0).
    return_pctx: refiner v4 parent-pass students only (_student_forward_v4)."""
    from ..refiner.e2e import ego_inputs

    B, K = cand.shape[:2]
    ego = ego_inputs(status)
    v0, a0, eds, cmd = ego
    allc = cand.float() if extra is None else torch.cat([cand.float(), extra.float()], 1)
    KA = allc.shape[1]
    feat, mem = student.trunk.scene(bev_grid, B)
    if getattr(student, "v4_readout", None) is not None or getattr(student, "pref_head", None) is not None:
        return _student_forward_v4(student, feat, mem, ego, cand, extra, cand_parent, extra_parent,
                                   return_pctx=return_pctx and getattr(student, "v4_readout", None) is not None,
                                   rel_objs=rel_objs)
    o = student.trunk.candidates(feat, mem, allc, v0, a0, eds, cmd)
    sc = student.score(o, B, KA)
    w = o["w_lat"].float()
    return {"score_logit": sc[:, :K], "extra_score_logit": None if extra is None else sc[:, K:],
            "w_lat": w[:, :K], "w_lat_extra": None if extra is None else w[:, K:], "score_all": sc, "w_lat_all": w,
            "ego": ego}


def student_rel_rows(predictions: Dict, v5: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
    """refiner v5 relation 'student': the student's det head (final layer of predictions all_bbox_preds /
    all_cls_scores / det_hidden; detached) -> per-token object rows (ck/v5/student_objects.student_object_rows); None
    when the block / channel is off.  KeyError when the det outputs are missing (inference needs
    agent.config.expose_det_outputs=true)."""
    if not v5["enabled"] or v5["relation"] == "off":
        return None
    from .v5.student_objects import student_object_rows
    miss = [k for k in ("all_bbox_preds", "all_cls_scores", "det_hidden") if k not in predictions]
    if miss:
        raise KeyError(f"ck_e2e2.v4.v5.relation student: predictions lack {miss} (det head off, or inference without "
                       f"agent.config.expose_det_outputs=true)")
    return student_object_rows(predictions["all_bbox_preds"][-1].detach(), predictions["all_cls_scores"][-1].detach(),
                               predictions["det_hidden"].detach(), int(v5["n_obj"]), float(v5["score_thr"]))


def refine_cols_parents(vtype: torch.Tensor, n: int) -> torch.Tensor:
    """the first n PARENT columns (vtype 0) of a KD set in slot order (stable), then the others -> [B, n]."""
    order = torch.argsort((vtype != 0).to(torch.int8), dim=1, stable=True)
    return order[:, :max(1, min(int(n), vtype.shape[1]))].contiguous()


# ----------------------------------------------------------------------------------------------- helpers
def _f(x) -> float:
    return float(x.detach()) if torch.is_tensor(x) else float(x)


def _masked_l1(es: torch.Tensor, et: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """mean over the m candidates and the 6 controls of |es - et| (et detached; 0 with the graph kept if m is empty)."""
    et = et.detach().float().to(es.device)
    m = m.bool().to(es.device) & torch.isfinite(et).all(-1)
    et = torch.where(m[..., None], et, es.detach().float())
    per = (es.float() - et).abs().mean(-1)
    mf = m.float()
    return (per * mf).sum() / torch.clamp(mf.sum(), min=1.0)


def _gather_c(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B, C, ...], idx [B, n] (long, same device) -> x[b, idx[b]] [B, n, ...]."""
    shape = idx.shape + x.shape[2:]
    ii = idx.reshape(idx.shape + (1,) * (x.dim() - 2)).expand(shape)
    return x.gather(1, ii)


def _dist_mean_many(xs: Sequence[float], device) -> List[float]:
    """element-wise mean of a float vector over the ranks (ONE all-reduce, float64) when torch.distributed is
    initialised (every rank must call it with the same length); the input unchanged otherwise."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return [float(x) for x in xs]
    dev = torch.device(device)
    if dist.get_backend() == "nccl" and dev.type != "cuda":
        dev = torch.device("cuda", torch.cuda.current_device())
    t = torch.tensor([float(x) for x in xs], dtype=torch.float64, device=dev)
    dist.all_reduce(t)
    return [float(v) / dist.get_world_size() for v in t.tolist()]


# ----------------------------------------------------------------------------------------------- refiner v4_3 M6
def kd_constraint_request(occ_channels: str):
    """TeacherPair2.run's fields request for occ_channels: True (the original B | U dict; constraint_kd then runs the
    original ops bit for bit) for 'B' / 'BU', else the occ_channels name (its groups, field_groups)."""
    return True if CN_GROUPS[str(occ_channels)] in (("B",), FIELDS_TRUE) else str(occ_channels)


CONSTRAINT_REFUSAL = ("constraint KD refused (the teachers' fields are distilled cell by cell: their geometry, time "
                      "bins and B semantics must be the student's): ")


def constraint_teacher_problems(sn, student_cv: int, occ_channels: str,
                                teachers: Sequence[Tuple[str, str, Any, int]]) -> List[str]:
    """The constraint-KD compatibility rule, shared by the runtime check (check_constraint_teachers, the loaded nets) and
    the launch check (tools/ck/e2e2/launch_util2.kd_constraint_check, the runs' config.json): sn = the student's V4Cfg,
    student_cv its corridor version, occ_channels the distilled groups (CN_GROUPS), teachers = [(name 'det' | 'map',
    run, V4Cfg, corridor_version)] -> the mismatches ([] = compatible), each naming the teacher:
      * grid: corridor version (the grid geometry) and time bins (ev_tbins x ev_tbin_s) of every teacher = the
        student's;
      * B semantics: rg_stopped_aware of every teacher = the student's (B = the stopped-aware B of Phase 0b R2 on both
        sides, or the plain not-behind B on both; the student's rg_gt_aux trains B with its own rule);
      * NOT rg_read (the recorded v4_3 decision, tests/test_v4_r2.py::test_constraint_teachers_b_semantics_check: the
        read is how the net consumes a field, not what a cell holds; the cells are distilled as cells, so a teacher's
        read may differ from the student's -- launch_util2.kd_constraint_check records both sides' rg_read.  Caveat:
        a teacher's B is also shaped by its label / station BCE through its own read (any4: the 2 x 2 OR), so its
        cells are not only its GT field target; requiring the same read is a user decision, not taken);
      * groups: occ_channels' groups exist on the student and on the DET teacher (Bp: rg_stopped_aware only)."""
    want = (int(student_cv), int(sn.ev_tbins), float(sn.ev_tbin_s))
    bad: List[str] = []
    groups = CN_GROUPS.get(str(occ_channels))
    if groups is None:
        bad.append(f"occ_channels {occ_channels!r} not in {OCC_CHANNELS}")
    elif any(g not in net_field_groups(sn) for g in groups):
        bad.append(f"student: occ_channels {occ_channels!r} distils {list(groups)}, the student's fields are "
                   f"{list(net_field_groups(sn))} (Bp: rg_stopped_aware only)")
    for name, run, tv, tcv in teachers:
        got = (int(tcv), int(tv.ev_tbins), float(tv.ev_tbin_s))
        if got != want:
            bad.append(f"{name} {run}: grid (corridor_version, ev_tbins, ev_tbin_s) {got} != the student's {want}")
        if bool(tv.rg_stopped_aware) != bool(sn.rg_stopped_aware):
            bad.append(f"{name} {run}: rg_stopped_aware {bool(tv.rg_stopped_aware)} != the student's "
                       f"{bool(sn.rg_stopped_aware)} (B semantics: the R2 stopped-aware B vs the plain not-behind B)")
        if name == "det" and groups is not None and any(g not in net_field_groups(tv) for g in groups):
            bad.append(f"{name} {run}: occ_channels {occ_channels!r} distils {list(groups)}, the teacher's fields are "
                       f"{list(net_field_groups(tv))} (Bp: rg_stopped_aware only)")
    return bad


def check_constraint_teachers(tp: "TeacherPair2", cfg: CKE2E2Config) -> None:
    """Constraint KD reads the teachers' fields cell by cell on the student's parent grid: constraint_teacher_problems
    on the loaded nets (grid and rg_stopped_aware of both teachers = the student's, occ_channels' groups on the student
    and the DET teacher; rg_read may differ) -> ValueError (CONSTRAINT_REFUSAL + the mismatches) otherwise.  For an R2
    student with R2 teachers the distilled B is the stopped-aware B on both sides."""
    bad = constraint_teacher_problems(cfg.v4_net(), int(cfg.v4["corridor_version"]),
                                      str(cfg.kd_constraint["occ_channels"]),
                                      [(name, tp.runs[name], net.v4, int(net.corridor_version))
                                       for name, net in (("det", tp.det), ("map", tp.map))])
    if bad:
        raise ValueError(CONSTRAINT_REFUSAL + "; ".join(bad))


def parent_rows(pidx_s: torch.Tensor, m: torch.Tensor, U: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unique parent rows of a candidate set: pidx_s [B, K] (candidate -> row of [B, U]), m [B, K] candidates that
    count -> (rep [B, U] long: the first counted candidate k of each row (K where none), has [B, U] bool)."""
    B, K = pidx_s.shape
    k = torch.arange(K, device=pidx_s.device)[None].expand(B, K)
    kk = torch.where(m.bool(), k, torch.full_like(k, K))
    rep = torch.full((B, U), K, dtype=torch.long, device=pidx_s.device)
    rep = rep.scatter_reduce(1, pidx_s.long(), kk, reduce="amin", include_self=True)
    return rep, rep < K


def constraint_kd(haz: Dict[str, torch.Tensor], T: Dict[str, torch.Tensor], cand_ok: torch.Tensor,
                  kc: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """refiner v4_3 M6 constraint KD (spec V4_3 M6; ck_e2e2.kd_constraint) on the student's unique parents of the KD set.
    haz = the student's split_haz of the KD set (range readout: occ_logit_u / ovl_logit_u (/ bp_logit_u with
    rg_stopped_aware) [B, U, 48, 17, M], margin_u [B, U, 48, 17] (m) per parent row, parent_index [B, K]); T =
    TeacherPair2.run(..., fields=<occ_channels> | True) on the same KD candidates and parents (fields_det / fields_map
    with their own parent_index and 'groups' (absent = ('B', 'U'), the fields=True dict), ok_det_tok / ok_map_tok
    [B]); cand_ok [B, K] the counted KD candidates.  Row pairing: student row u <- its first counted candidate k <- the
    teacher's row parent_index_t[k] (the same parent draft, bitwise; each teacher deduplicated it on its own).
      occ:    BCE(student logits, sigmoid(DET teacher logits)) - H(target) (= the Bernoulli KL, gradient of the BCE) over
              every cell and bin of the groups CN_GROUPS[occ_channels] ('B' the B channel -- the stopped-aware B of an
              R2 pair --, 'BU' + the TTC overlap channel U, 'BBp' + the plain not-behind Bp, 'all' B, U, Bp; each group
              student <- the same group of the teacher), mean over the rows of DET-ok tokens (ValueError if a group is
              missing on either side);
      margin: Huber(student D, MAP teacher D clipped +-CK_MARGIN_CLIP m, delta 1 m) over the cells of stations
              s_j <= map_range_m (the teacher row's arc lengths), MAP-ok tokens.
    -> (l_occ, l_margin, stats); zeros (graph-connected) when nothing is selected.  BCE / Huber only (no ranking or
    margin loss)."""
    import torch.nn.functional as F
    from .losses import bernoulli_entropy
    Bo, Bu, D = haz["occ_logit_u"], haz["ovl_logit_u"], haz["margin_u"]
    B, U = Bo.shape[:2]
    M = Bo.shape[-1]
    zero = Bo.sum() * 0.0 + Bu.sum() * 0.0 + D.sum() * 0.0
    dev = Bo.device
    pidx = haz["parent_index"].long().to(dev)
    st: Dict[str, float] = {}
    out = []
    for name in ("det", "map"):
        f = T[f"fields_{name}"]
        m = cand_ok.bool().to(dev) & T[f"ok_{name}_tok"].bool().to(dev)[:, None]
        rep, has = parent_rows(pidx, m, U)
        b, u = torch.nonzero(has, as_tuple=True)
        st[f"cn_{name}_rows"] = float(b.numel())
        if b.numel() == 0:
            out.append(zero)
            continue
        ut = f["parent_index"].long().to(dev)[b, rep[b, u]]                       # the teacher's row of the same parent
        if name == "det":
            want = CN_GROUPS[str(kc["occ_channels"])]
            tg = tuple(f.get("groups", FIELDS_TRUE))                              # the teacher's returned groups
            miss = ([f"student {g}" for g in want if FIELD_GROUP_KEY[g] not in haz]
                    + [f"teacher {g}" for g in want if g not in tg])
            if miss:
                raise ValueError(f"constraint_kd: occ_channels {kc['occ_channels']!r} needs the groups {list(want)}; "
                                 f"missing {miss} (teacher fields {list(tg)})")
            if tg == FIELDS_TRUE and want in (("B",), FIELDS_TRUE):   # B | U teacher dict: the original ops, bitwise
                s_log = torch.cat([Bo[b, u], Bu[b, u]], -1).float()                # [P, 48, 17, 2M]
                t_log = f["blk_u"].to(dev)[b, ut].float()
                if want == ("B",):
                    s_log, t_log = s_log[..., :M], t_log[..., :M]
            else:                                                       # any other request: group by group
                s_log = torch.cat([haz[FIELD_GROUP_KEY[g]][b, u] for g in want], -1).float()   # [P, 48, 17, G M]
                t_all = f["blk_u"].to(dev)[b, ut].float()
                t_log = t_all if tg == want else torch.cat([t_all[..., tg.index(g) * M:(tg.index(g) + 1) * M]
                                                            for g in want], -1)
            p = torch.sigmoid(t_log).detach()
            per = F.binary_cross_entropy_with_logits(s_log, p, reduction="none") - bernoulli_entropy(p)
            out.append(per.mean())
            st["cn_occ_kl"] = float(out[-1].detach())
            st["cn_occ_rate_t"] = float((p > 0.5).float().mean())
        else:
            ds = D[b, u].float()                                                  # [P, 48, 17]
            dt = f["margin_u"].to(dev)[b, ut].float().clamp(-CK_MARGIN_CLIP, CK_MARGIN_CLIP).detach()
            near = (f["s_u"].to(dev)[b, ut].float() <= float(kc["map_range_m"]))[..., None].expand_as(ds)
            per = F.huber_loss(ds, dt, delta=1.0, reduction="none")
            nf = near.float()
            out.append((per * nf).sum() / nf.sum().clamp(min=1.0))
            st["cn_margin_huber"] = float(out[-1].detach())
            st["cn_margin_cells_frac"] = float(nf.mean())
    return out[0], out[1], st


@torch.no_grad()
def lead_ttc0(targets, status: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest-lead time to collision at t0 from the batch's ref_* GT (kd_score_scope 'near'), the rule of
    tools/ck/data/lead_split.py (extract_lead.py, report 43) on the GT loader's data: objects of class vehicle / bicycle
    observed at t = 0 (gt_future query rule), projected on the route centerline (ref_cl_xy vertices in N; nearest vertex
    + along-offset, heading of the vertex's segment, arc length relative to the projection of the t0 rear axle):
    |lat| <= 1.8 m, |heading - centerline heading| <= 60 deg, gap = s - L / 2 - 4.049 m in (-1, 80] m; lead = the
    smallest s; ttc0 = gap / (v_e - v_lon) if v_e - v_lon > 0.1 m/s and gap > 0 else TTC0_NONE (v_e = |(vx, vy)| of
    status[4:6], v_lon = the lead's velocity (first appearance, = its t0 value when annotated at t0) along the
    centerline).  Differences from the parquet: the GT loader's centerline crop (vertices, not the 0.5 m resampling)
    and object set (tracks within R of t0).  -> (ttc0 [B] s (TTC0_NONE without a closing lead or GT), has_gt [B])."""
    from ..refiner import gt_future as GF
    from .v4 import range_targets as RT
    ok = targets["ref_gt_ok"].reshape(-1).bool()
    dev = ok.device
    B = ok.shape[0]
    ttc = torch.full((B,), TTC0_NONE, dtype=torch.float64, device=dev)
    need = ("ref_obj_kf", "ref_obj_first", "ref_obj_meta", "ref_obj_n", "ref_obj_n_kf", "ref_cl_xy", "ref_cl_valid")
    if not ok.any() or any(k not in targets for k in need):
        return ttc, ok & False
    ref = {k: targets[k].to(dev) for k in need}
    tok = torch.nonzero(ok, as_tuple=True)[0]
    boxes, obs, _, _ = RT.gt_boxes_from_ref(ref, tok, t_grid=np.zeros(1), device=dev)        # [n, A, 1, 5]
    A = boxes.shape[1]
    meta = ref["ref_obj_meta"][tok][:, :A].long()
    vel = ref["ref_obj_first"][tok][:, :A, 3:5].double()
    cls_ok = (meta[..., GF.MT_CLASS][..., None] == torch.as_tensor(LEAD_CLASSES, device=dev)).any(-1)
    xy, hd, L = boxes[:, :, 0, :2].double(), boxes[:, :, 0, 2].double(), boxes[:, :, 0, 3].double()
    # only the possible leads are projected (class, observed at t0, centre within reach): [n, A] -> [n, Q]
    pre = obs[:, :, 0].bool() & cls_ok & (xy.norm(dim=-1) <= LEAD_GAP_MAX + 40.0)
    Q = int(max(int(pre.sum(1).max()), 1))
    oi = torch.sort((~pre).to(torch.uint8), dim=1, stable=True).indices[:, :Q]             # candidates first
    gq = lambda a: torch.gather(a, 1, oi.reshape(*oi.shape, *([1] * (a.dim() - 2))).expand(*oi.shape, *a.shape[2:]))
    pre, xy, hd, L, vel = gq(pre), gq(xy), gq(hd), gq(L), gq(vel)
    cl = ref["ref_cl_xy"][tok].double()                                                   # [n, CL, 2]
    clv = ref["ref_cl_valid"][tok].bool()
    nxt = torch.cat([cl[:, 1:], cl[:, -1:]], 1)
    seg = nxt - cl
    seg_ok = clv & torch.cat([clv[:, 1:], torch.zeros_like(clv[:, :1])], 1) & (seg.norm(dim=-1) > 1e-6)
    # vertices without a following segment take the previous segment's direction
    seg = torch.where(seg_ok[..., None], seg, torch.cat([seg[:, :1], seg[:, :-1]], 1))
    e = seg / seg.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    s_v = torch.cat([torch.zeros_like(cl[:, :1, 0]), torch.cumsum((cl[:, 1:] - cl[:, :-1]).norm(dim=-1)
                                                                  * (clv[:, 1:] & clv[:, :-1]), 1)], 1)

    def project(p):                                                                       # p [n, Q, 2]
        d2 = ((p[:, :, None] - cl[:, None]) ** 2).sum(-1)
        d2 = torch.where(clv[:, None], d2, torch.full_like(d2, math.inf))
        i = d2.argmin(-1)                                                                 # [n, Q]
        g = lambda a: torch.gather(a, 1, i[..., None].expand(*i.shape, a.shape[-1]))
        rel = p - g(cl)
        ee = g(e)
        along = (rel * ee).sum(-1)
        lat = -ee[..., 1] * rel[..., 0] + ee[..., 0] * rel[..., 1]
        return torch.gather(s_v, 1, i) + along, lat, torch.atan2(ee[..., 1], ee[..., 0])

    s0, _, _ = project(torch.zeros(len(tok), 1, 2, dtype=torch.float64, device=dev))
    s, lat, ch = project(xy)
    s = s - s0
    dh = torch.atan2(torch.sin(hd - ch), torch.cos(hd - ch)).abs()
    gap = s - 0.5 * L - LEAD_EGO_FRONT
    cand = pre & (lat.abs() <= LEAD_LAT_M) & (dh <= LEAD_DH_RAD) & (gap > -1.0) & (gap <= LEAD_GAP_MAX)
    s_m = torch.where(cand, s, torch.full_like(s, math.inf))
    j = s_m.argmin(-1)
    has = torch.isfinite(s_m.gather(1, j[:, None])[:, 0])
    ar = torch.arange(len(tok), device=dev)
    v_lon = (vel[ar, j] * torch.stack([torch.cos(ch[ar, j]), torch.sin(ch[ar, j])], -1)).sum(-1)
    st = status[tok].double().to(dev)
    v_e = torch.hypot(st[:, 4], st[:, 5])
    dv = v_e - v_lon
    g = gap[ar, j]
    t0 = torch.where(has & (dv > 0.1) & (g > 0), g / dv.clamp(min=1e-9), torch.full_like(g, TTC0_NONE))
    ttc[tok] = t0
    return ttc, ok


@torch.no_grad()
def parent_min_corner_sdf(targets, status: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """min over the 41 EXECUTED states (ck/v4/lqr.simulate from status[4] / status[6], the official tracker) x the 4
    official corners of the token's DAC SDF (ref_sdf, sdf.sample_sdf; unreadable corners skipped) of every parent
    parents [B, K, 8, 3] -> [B, K] m (+inf without GT / a readable corner).  kd_score_scope 'near' (DAC)."""
    from ..refiner.sdf import ego_corners, sample_sdf
    from .v4 import lqr as LQR
    B, K = parents.shape[:2]
    dev = parents.device
    out = torch.full((B, K), math.inf, dtype=torch.float32, device=dev)
    if "ref_sdf" not in targets or "ref_gt_ok" not in targets:
        return out
    ok = targets["ref_gt_ok"].reshape(-1).bool().to(dev)
    tok = torch.nonzero(ok, as_tuple=True)[0]
    if tok.numel() == 0:
        return out
    p = parents[tok].float().reshape(-1, 8, 3)
    fin = torch.isfinite(p).flatten(1).all(1)
    p = torch.where(fin[:, None, None], p, torch.zeros_like(p))
    st = status[tok].float().to(dev)
    ego = torch.stack([torch.nan_to_num(st[:, 4]), torch.nan_to_num(st[:, 6])], -1).repeat_interleave(K, 0)
    ex = LQR.simulate(p, ego, dtype=torch.float32)                                       # [n K, 41, 6]
    cor = ego_corners(ex[..., :3]).reshape(len(tok), K, -1, 2)
    v, val = sample_sdf(targets["ref_sdf"].to(dev), cor.reshape(len(tok), -1, 2), tok)
    v = torch.where(val, v.float(), torch.full_like(v.float(), math.inf)).reshape(len(tok), K, -1).min(-1).values
    out[tok] = torch.where(fin.reshape(len(tok), K), v, torch.full_like(v, math.inf))
    return out


def score_kd_near(s_kd: torch.Tensor, p_t: torch.Tensor, ok: torch.Tensor, keys: Sequence[str],
                  nc_tok: torch.Tensor, dac_cand: torch.Tensor) -> torch.Tensor:
    """kd_score_scope 'near': losses.score_kd_bce with a mask per key -- nc: ok & nc_tok [B] (nearest lead ttc0 <
    NEAR_TTC_S), dac: ok & dac_cand [B, K] (parent's min executed corner SDF < NEAR_SDF_M), other keys ok -- each key's
    Bernoulli KL averaged over its own candidates (0 without any), then the mean over the keys (= score_kd_bce when
    every mask equals ok).  s_kd / p_t [B, K, n_keys] in the order of keys.
    Weighting (by definition, not normalised to the 'all' count): each key keeps the total weight 1 / n_keys of the
    'all' term, concentrated on its near candidates, so a near candidate's NC / DAC KL weighs ~1 / (near fraction of
    the KD candidates) times its weight under scope 'all' (logged: v4/kd_near_nc_frac, v4/kd_near_dac_frac)."""
    from . import losses as L
    terms = []
    for i, k in enumerate(keys):
        m = ok.bool()
        if k == "nc":
            m = m & nc_tok.bool().to(m.device)[:, None]
        elif k == "dac":
            m = m & dac_cand.bool().to(m.device)
        terms.append(L.score_kd_bce(s_kd[..., i:i + 1], p_t[..., i:i + 1], m))
    return torch.stack(terms).mean()


# ----------------------------------------------------------------------------------------------- main helper
class CKE2E2:
    """Loss / inference helper held by ParaSSRAgent (plain object).  epoch / epoch_frac / rank / world_size / gstep /
    batch_idx are set by the callback (make_ck2_callback); mb counts this rank's micro-batches."""

    def __init__(self, cfg: CKE2E2Config, max_epochs: int = 30):
        from .losses import EmaBalancer

        self.cfg = CKE2E2Config.from_any(cfg)
        self.max_epochs = int(max_epochs)
        e = self.cfg.lat_kd
        self.ema = EmaBalancer(ratio=e["ratio"], m=e["m"], floor=e["floor"], cap=e["cap"], start_step=e["start_mb"])
        self.bev_ctrl = None                 # bev_kd_arm.TeacherRatioControllers (one RatioController per teacher)
        if self.cfg.bev_kd["enabled"]:
            from .bev_kd_arm import controllers_from_cfg
            self.bev_ctrl = controllers_from_cfg(self.cfg.bev_kd)
        self.epoch, self.epoch_frac, self.rank, self.world_size, self.batch_idx, self.gstep = 0, 0.0, 0, 1, 0, 0
        self.mb = 0
        self.skipped_steps = 0
        self.cum = {"tokens": 0, "rows_ok": 0, "nonfinite_loss": 0, "sur_nonfinite": 0, "rec_tokens": 0,
                    "bevkd_nonfinite": 0, **{f"G_src_{n}": 0 for n in SRC_NAMES}}
        if self.cfg.bev_kd["enabled"]:       # rank-local token counters of the arm (tokens seen / without teacher BEV)
            self.cum.update({"bevkd_tok": 0, **{f"bevkd_miss_{t}": 0 for t in self.cfg.bev_kd["teachers"]}})
        self._teachers: Optional[TeacherPair2] = None
        self._labels = None
        self._labels_max_epoch: Optional[int] = None
        self._sampler = None
        self._rowmap = None
        self.recorder: Optional[CandRecorder2] = None
        self._rows_checked = False
        self.last_clip_norm: Optional[float] = None

    # -------------------------------------------------------------------------------------- schedule
    @property
    def record_until(self) -> int:
        c = self.cfg.record_until_epoch
        return self.max_epochs - 2 if c is None else int(c)

    def phase(self, epoch: int) -> str:
        e = int(epoch)
        if e >= self.cfg.onpolicy_from_epoch:
            return "onpolicy"
        if e >= self.cfg.record_from_epoch:
            return "warmup_record"
        return "warmup"

    def r(self, epoch_frac: float) -> float:
        if self.phase(int(math.floor(epoch_frac))) != "onpolicy":
            return 1.0
        ramp = float(self.cfg.kd_ramp_epochs)
        if ramp <= 0:
            return 1.0
        return float(min(max((float(epoch_frac) - self.cfg.onpolicy_from_epoch) / ramp, 0.0), 1.0))

    def recording(self, epoch: int) -> bool:
        return self.cfg.record_from_epoch <= int(epoch) <= self.record_until

    # -------------------------------------------------------------------------------------- state
    def state_dict(self) -> Dict[str, Any]:
        return {"ema": self.ema.state_dict(), "mb": int(self.mb), "cum": dict(self.cum),
                "skipped_steps": int(self.skipped_steps),
                "bev_ctrl": None if self.bev_ctrl is None else self.bev_ctrl.state_dict()}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if not sd:
            return
        self.ema.load_state_dict(sd["ema"])
        self.mb = int(sd.get("mb", 0))
        self.cum.update({k: int(v) for k, v in dict(sd.get("cum", {})).items()})
        self.skipped_steps = int(sd.get("skipped_steps", 0))
        if self.bev_ctrl is not None and sd.get("bev_ctrl"):
            self.bev_ctrl.load_state_dict(sd["bev_ctrl"])

    # -------------------------------------------------------------------------------------- lazy pieces
    def teachers(self, device) -> TeacherPair2:
        if self._teachers is None:
            c = self.cfg
            kc = c.kd_calib
            kw = {"fields": True} if c.kd_constraint["enabled"] else {}      # off: the old call exactly
            self._teachers = TeacherPair2(c.teacher_det_run, c.teacher_map_run, c.teacher_which, device,
                                          c.teacher_amp, c.teacher_require_done,
                                          ep_target=c.ep_target if c.teacher_ep_check else None,
                                          calib=({"det": kc["det"], "map": kc["map"]} if kc["enabled"] else None),
                                          kd_keys=c.kd_score_keys, **kw)
            if kw:
                check_constraint_teachers(self._teachers, c)
            if self._teachers.calib_info:
                print("[ck_e2e2] KD calibration: " + "; ".join(
                    f"{t} {i['path']} (sha16 {i['sha16']}, ckpt {i['ckpt_sha16']}, keys {i['fitted_keys']})"
                    for t, i in self._teachers.calib_info.items()), flush=True)
            if self._teachers.temporal_info is not None:
                ti = self._teachers.temporal_info
                print(f"[ck_e2e2] temporal DET teacher {c.teacher_det_run}: forecaster {ti['cfg']['forecaster']} "
                      f"(fc_train {ti['cfg']['fc_train']}), branch sha16 {ti['branch_state_sha16']}, forecaster sha16 "
                      f"{ti.get('forecaster_state_sha16')}, fc_init {ti.get('fc_init')}", flush=True)
        return self._teachers.to(device)

    def labels(self):
        if self._labels is None:
            from .e2e_data2 import LabelStore2
            q = self.cfg.v4
            kw = {}                                   # refiner v4 extras (off: the CK2 store, unchanged)
            if q["readout"] == "hazard" and q["lambda_station"] > 0:
                kw["with_tidx"] = True
            if q["pref_head"]:
                kw["with_pref"] = True
            if q["corrector"] or self.cfg.v4_opt("edits") == "computed":     # corrector A / v4_3 computed edits
                kw["with_edits"] = True
            self._labels = LabelStore2(self.cfg.io_dir, n_rows=N_ROWS, packed=self.cfg.warmup["packed"],
                                       var_sampling=self.cfg.var_sampling, ep_target=self.cfg.ep_target, **kw)
        return self._labels

    def sampler(self):
        if self._sampler is None:
            from .anchor_sampler import load_anchors
            from .e2e_data2 import WarmupSampler
            w = self.cfg.warmup
            self._sampler = WarmupSampler(load_anchors(), sampler_seed=w["sampler_seed"], seed=self.cfg.seed,
                                          n_var=w["n_var"], var_sampling=self.cfg.var_sampling)
        return self._sampler

    def rowmap(self):
        if self._rowmap is None:
            from .e2e_data import RowMap
            self._rowmap = RowMap(self.cfg.warmup["packed"])
        return self._rowmap

    def tokens_of(self, rows) -> List[str]:
        r = rows.detach().cpu().numpy() if torch.is_tensor(rows) else np.asarray(rows)
        r = r.astype(np.int64).reshape(-1)
        if not (r >= 0).any():
            return [""] * len(r)
        tok = self.rowmap().tokens
        return [str(tok[x]) if 0 <= x < len(tok) else "" for x in r]

    def refresh_labels(self, epoch: Optional[int] = None) -> Dict:
        e = self.epoch if epoch is None else int(epoch)
        max_epoch = e - self.cfg.label_lag
        st = self.labels().refresh(max_epoch)
        self._labels_max_epoch = max_epoch
        return dict(st or {}, max_epoch=max_epoch)

    def _check_rows(self, rows: torch.Tensor) -> None:
        n = int(rows.numel())
        ok = int((rows >= 0).sum())
        self.cum["tokens"] += n
        self.cum["rows_ok"] += ok
        if self.cfg.strict_rows <= 0:
            return
        if self.mb == 0 and n > 0 and ok == 0:
            raise RuntimeError("ck_e2e2: no training token of the first micro-batch has a packed navtrain_train row "
                               "(targets['ck_row'] all -1): wrong split / token mapping")
        if not self._rows_checked and self.cum["tokens"] >= 2000:
            self._rows_checked = True
            frac = self.cum["rows_ok"] / max(self.cum["tokens"], 1)
            if frac < self.cfg.strict_rows:
                raise RuntimeError(f"ck_e2e2: only {frac:.4f} of the first {self.cum['tokens']} training tokens have "
                                   f"a packed row (< strict_rows {self.cfg.strict_rows})")

    # -------------------------------------------------------------------------------------- step candidates
    @staticmethod
    def _group_pos(group_a: torch.Tensor, gmax: int, n: int) -> torch.Tensor:
        """slot positions (slot order) of the anchors with group <= gmax: exactly n per token (WarmupSampler gives
        the canonical 8 / 8 / 16 group pattern to rows without data)."""
        return torch.sort((group_a > gmax).to(torch.uint8), dim=1, stable=True).indices[:, :n]

    def warmup_batch(self, targets, toks: Sequence[str], fallback: torch.Tensor, device) -> Dict[str, Any]:
        """SPEC s2-1 (cands2.warmup_cands): 32 sampled raw anchors + warmup.n_var variants with official labels (as CK
        targets with the EP of cfg.ep_target, made by the target builder's wu_row);
        candidates that are not teacher-ok (no warm-up data, invalid / padding variant) are replaced by the v2
        trajectory (finite input; they are masked everywhere)."""
        c = self.cfg
        dev = torch.device(device)
        W = C2.warmup_cands(targets, toks, self.epoch, self.sampler(), device=dev)
        cand, fin = _sanitize_traj(W["cand"], fallback, W["teacher_ok"])
        grp = W["group"]
        ga = grp[:, :W["anc_idx"].shape[1]]
        gs = c.warmup["sur_groups"]
        sur_pos = self._group_pos(ga, SUR_GROUP_MAX[gs], SUR_GROUP_N[gs])
        lat_mask = (grp <= SUR_GROUP_MAX[c.warmup["lat_kd_groups"]]) | (grp == GROUP_VAR)
        wu = W["wu_ok"].bool()
        B = cand.shape[0]
        vv = targets["ck2_var_valid"].reshape(B, K16, NV)
        par = _keep_parent(cand, self.sampler().anchors_t(dev)[W["parent_anchor"].to(dev).long()],
                           W["teacher_ok"] & fin)
        return {"cand": cand, "cand_ok": W["teacher_ok"] & fin, "extra": None, "y": W["y"],
                "cand_parent": par, "extra_parent": None, "vtype_kd": W["vtype"].to(dev).long(),
                "tidx_lab": W.get("tidx"), "pref": W.get("pref"),
                "ok_lab": W["ok"] & fin, "vtype_lab": W["vtype"], "grp": grp, "sur_pos": sur_pos,
                "sur_tok_ok": wu, "lat_mask": lat_mask, "kd_tok_ok": wu, "wu_ok": wu,
                "var_valid_frac": float(vv[:, :, 1:].float().mean()),
                "src": torch.full((B,), SRC_FALLBACK, dtype=torch.long)}

    def label_set(self, rows: torch.Tensor, toks: Sequence[str], targets, device) -> Dict[str, torch.Tensor]:
        """The on-policy BCE set G: LabelStore2.lookup (newest generation <= e - label_lag; 16 identities +
        onpolicy_label.n_var variants, stream 'ck2e2e.lab') + the warm-up-style fallback (cands2.fill_fallback, NU12)
        for rows without a generation (onpolicy_label.fallback 'none': those rows get G_ok False, G_src 3)."""
        c = self.cfg
        if self._labels_max_epoch != self.epoch - c.label_lag:
            self.refresh_labels()
        ol = c.onpolicy_label
        G = self.labels().lookup(rows.detach().cpu().long(), list(toks), int(self.epoch), int(c.seed),
                                 n_orig=K16, n_var=ol["n_var"])
        if ol["fallback"] == "warmup":
            G = C2.fill_fallback(G, targets, list(toks), int(self.epoch), self.sampler(), ol["n_var"])
        else:
            miss = ~G["has"].bool()
            G["G_ok"][miss] = False
            G["G_src"][miss] = SRC_NODATA
        return {k: v.to(device) for k, v in G.items()}

    def current_batch(self, predictions, v0v: torch.Tensor, rows: torch.Tensor, toks: Sequence[str], targets,
                      fallback: torch.Tensor) -> Dict[str, Any]:
        """SPEC s2-2 (cands2.onpolicy_cands): current top-16 + n_var_step variants of them (KD / lateral KD; stream
        'ck2e2e.now') + the label set G (BCE).  v0v = cands2.v0_from_status (float64, as the variant label files)."""
        c = self.cfg
        dev = fallback.device
        top = student_topk(predictions, K16)
        V = C2.online_variants(top["cand"], v0v, c.variants)
        P = C2.onpolicy_cands(top["cand"], v0v, list(toks), int(self.epoch), int(c.seed), c.n_var_step, c.variants,
                              c.var_sampling, STREAM_NOW, V=V)
        cand, fin = _sanitize_traj(P["cand"], fallback, P["ok"])
        G = self.label_set(rows, toks, targets, dev)
        G_traj, G_fin = _sanitize_traj(G["G_traj"].float(), fallback, G["G_ok"].bool())
        B, Kc = cand.shape[:2]
        tc = top["cand"].float()
        kd_par = torch.gather(tc, 1, P["parent"].to(dev).long()[..., None, None].expand(-1, -1, *tc.shape[2:]))
        if "E_ok" in G:
            return self._with_edits(cand, fin, P, G, G_traj, G_fin, kd_par, targets, top, V, dev)
        return {"cand": cand, "cand_ok": P["ok"] & fin, "extra": G_traj, "y": G["G_y"].float(),
                "cand_parent": _keep_parent(cand, kd_par, P["ok"] & fin),
                "extra_parent": _keep_parent(G_traj, label_set_parents(G, targets, self.sampler(), dev),
                                             G["G_ok"].bool() & G_fin),
                "vtype_kd": P["vtype"].to(dev).long(),
                "tidx_lab": G.get("G_tidx"), "pref": G.get("G_pref"),
                "ok_lab": G["G_ok"].bool() & G_fin, "vtype_lab": G["G_vtype"].long(),
                "sur_pos": torch.arange(K16, device=dev)[None].expand(B, -1),
                "sur_tok_ok": torch.ones(B, dtype=torch.bool, device=dev),
                "lat_mask": torch.ones(B, Kc, dtype=torch.bool, device=dev),
                "kd_tok_ok": torch.ones(B, dtype=torch.bool, device=dev), "src": G["G_src"].long(),
                "var_valid_frac": float(V["valid96"].reshape(B, K16, NV)[:, :, 1:].float().mean()),
                "top": top, "V": V}

    def _with_edits(self, cand, fin, P, G, G_traj, G_fin, kd_par, targets, top, V, dev) -> Dict[str, Any]:
        """current_batch of a student whose generations carry edits (refiner v4 corrector A, or the v4_3 computed
        edits): the label set G + the generation's recorded edits (their official labels: the critic's BCE also
        learns the edits; parents = their source's identity), and the AWR inputs S['E'] (recorded sources, parents,
        controls, PDMS of edits / sources / G; used by the corrector's AWR only)."""
        B, Kc = cand.shape[:2]
        KG = G_traj.shape[1]
        E = {k: v.to(dev) for k, v in G.items() if k.startswith("E_") or k == "G_pdms"}
        e_ok = E["E_ok"].bool()
        g_par = _keep_parent(G_traj, label_set_parents(G, targets, self.sampler(), dev), G["G_ok"].bool() & G_fin)
        extra = torch.cat([G_traj, E["E_traj"].float()], 1)
        tid = G.get("G_tidx")
        if tid is not None:
            tid = torch.cat([tid, torch.full((B, e_ok.shape[1], tid.shape[-1]), -2, dtype=tid.dtype, device=dev)], 1)
        return {"cand": cand, "cand_ok": P["ok"] & fin, "extra": extra,
                "y": torch.cat([G["G_y"].float(), E["E_y"].float()], 1),
                "cand_parent": _keep_parent(cand, kd_par, P["ok"] & fin),
                "extra_parent": torch.cat([g_par, E["E_src_par"].float()], 1),
                "vtype_kd": P["vtype"].to(dev).long(), "tidx_lab": tid, "pref": G.get("G_pref"),
                "ok_lab": torch.cat([G["G_ok"].bool() & G_fin, e_ok], 1),
                "vtype_lab": torch.cat([G["G_vtype"].long(), torch.full_like(e_ok, NV, dtype=torch.long)], 1),
                "sur_pos": torch.arange(K16, device=dev)[None].expand(B, -1),
                "sur_tok_ok": torch.ones(B, dtype=torch.bool, device=dev),
                "lat_mask": torch.ones(B, Kc, dtype=torch.bool, device=dev),
                "kd_tok_ok": torch.ones(B, dtype=torch.bool, device=dev), "src": G["G_src"].long(),
                "var_valid_frac": float(V["valid96"].reshape(B, K16, NV)[:, :, 1:].float().mean()),
                "top": top, "V": V, "E": E, "n_G": KG,
                "kd_col96": torch.cat([torch.arange(K16, device=dev)[None].expand(B, -1) * NV,
                                       P["cols"].to(dev).long()], 1)}

    def corrector_edits(self, student, out: Dict, S: Dict, s_kd: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """refiner v4 A, recording: the corrector's sampled edits of the top EDIT_K KD candidates (the current top-16
        + sampled variants, critic-only selection score) -> {traj [B,8,8,3], src [B,8] (96-column), u [B,8,12],
        valid [B,8]} for CandRecorder2.add, or None (no corrector / no KD column map)."""
        from .v4.corrector import critic_select_score, decode_edits
        if getattr(student, "corrector", None) is None or "kd_col96" not in S or "ctx_all" not in out:
            return None
        Kc = S["cand"].shape[1]
        feat, mem, ego, pref = out["scene"]
        sc = critic_select_score(s_kd.detach())
        sc = torch.where(S["cand_ok"].bool(), sc, torch.full_like(sc, -1e9))
        k = int(self.cfg.v4["corr_k"])
        idx = torch.topk(sc, k, dim=1).indices
        B = idx.shape[0]
        g = lambda a: torch.gather(a, 1, idx.reshape(B, k, *([1] * (a.dim() - 2))).expand(B, k, *a.shape[2:]))
        src = g(S["cand"].float())
        ctx = torch.cat([g(out["ctx_all"][:, :Kc]), g(s_kd.detach().float()),
                         (ego[0].float() / 15.0).reshape(B, 1, 1).expand(B, k, 1)], -1)
        with torch.no_grad():
            a = student.corrector.act(ctx.detach(), sample=True)
        traj = decode_edits(src, a["u"], ego[0])
        valid = g(S["cand_ok"].bool()) & torch.isfinite(traj).flatten(2).all(-1)
        return {"traj": traj.detach(), "src": torch.gather(S["kd_col96"], 1, idx), "u": a["u"].detach(),
                "valid": valid}

    def computed_edits_rec(self, student, out: Dict, S: Dict, s_kd: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """refiner v4_3 M5, recording (= corrector_edits' block for the labeler): CKNet.computed_edits of the KD set
        (the current top-16 + sampled variants; sources = its top EDIT_K PARENT columns, i.e. identities, by the
        critic-only selection score -- the deployment beta is chosen at evaluation) on the predicted fields of this
        step's forward (out 'pctx' / 'parent_index_all': the edits are re-scored on that parent pass, no grad) ->
        {traj [B,8,8,3], src [B,8] (96-column of the source), u [B,8,12] (0 without the corrector), valid [B,8]}, or
        None (edits off / no KD column map / no parent context).  valid = an eligible source (not a filler slot),
        finite, and an actual edit (edit_kind > 0) unless the corrector adds its residual: a no-edit slot is its source
        bitwise, whose label the generation already holds (scoring it again would only duplicate a BCE row)."""
        if not student.v4.computed_edits or "kd_col96" not in S or "pctx" not in out:
            return None
        Kc = S["cand"].shape[1]
        feat, mem, ego, pref = out["scene"]
        pool_out = {"score_logit": s_kd.detach(), "haz": out["haz"], "pctx": out["pctx"],
                    "parent_index": out["parent_index_all"][:, :Kc]}
        if "ctx_all" in out:
            pool_out["ctx"] = out["ctx_all"][:, :Kc]
        dev = s_kd.device
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16,
                                             enabled=bool(self.cfg.amp and dev.type == "cuda")):
            ce = student.computed_edits(feat, mem, ego, S["cand"].float(), S["cand_parent"].float(), pool_out,
                                        valid=S["cand_ok"].bool(), pref=pref,
                                        sample=getattr(student, "corrector", None) is not None)
        traj = ce["edit_traj"].float()
        valid = ce["src_valid"].bool() & torch.isfinite(traj).flatten(2).all(-1)
        if getattr(student, "corrector", None) is None:
            valid = valid & (ce["edit_kind"] > 0)
        return {"traj": traj.detach(), "src": torch.gather(S["kd_col96"], 1, ce["src_idx"]), "u": ce["u"].detach(),
                "valid": valid, "kind": ce["edit_kind"], "clear": ce["edit_clear"]}

    # -------------------------------------------------------------------------------------- refiner v4_3 KD scope
    @staticmethod
    def _near_scope(targets, status: torch.Tensor, S: Dict, logs: Dict[str, float]
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """kd_score_scope 'near' masks from the ref_* GT (train time only; a token without GT is excluded from both):
        nc_tok [B] = nearest lead ttc0 < NEAR_TTC_S (lead_ttc0), dac_cand [B, K] = the KD candidate's parent has a min
        executed corner SDF < NEAR_SDF_M (parent_min_corner_sdf).  Logs v4/kd_near_*.  ValueError when the batch lacks
        any NEAR_REF_KEYS (the e2e target builder always emits them; a missing key would silently zero NC / DAC KD)."""
        dev = S["cand"].device
        tg = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in (targets or {}).items() if k.startswith("ref_")}
        st = status.to(dev)
        miss = [k for k in NEAR_REF_KEYS if k not in tg]
        if miss:
            raise ValueError(f"ck_e2e2.v4.kd_score_scope 'near' needs the GT loader's ref_* targets; missing {miss}")
        ttc0, gt = lead_ttc0(tg, st)
        nc_tok = gt & (ttc0 < NEAR_TTC_S)
        par = S.get("cand_parent")
        par = S["cand"] if par is None else par
        msdf = parent_min_corner_sdf(tg, st, par.float())
        dac = gt[:, None] & (msdf < NEAR_SDF_M)
        logs.update({"v4/kd_near_nc_frac": float(nc_tok.float().mean()), "v4/kd_near_dac_frac": float(dac.float().mean()),
                     "v4/kd_near_gt_frac": float(gt.float().mean())})
        return nc_tok, dac

    # -------------------------------------------------------------------------------------- refiner v4 terms
    def _v4_terms(self, inner: torch.Tensor, out: Dict, S: Dict, logs: Dict[str, float], targets=None,
                  student=None, s_lab: Optional[torch.Tensor] = None) -> torch.Tensor:
        """+ lambda_pref * Huber(log1p P_ref) (out pref_logp; target S 'pref' [B] m, else the GT loader's ref_p_pdm on
        ref_gt_ok tokens), + lambda_ev * (occupancy BCE + margin Huber) of the evidence readout on the KD set's own-
        parent candidates (GT-ok tokens), + lambda_station * censored station BCE of the hazard readout on the label set
        (out extra_haz (on-policy) / haz (warm-up), S 'tidx_lab' [B, KL, 3]); absent inputs -> unchanged.
        Logs v4/*."""
        from .v4 import losses as V4L
        q = self.cfg.v4
        targets = targets or {}
        if out.get("pref_logp") is not None and q["lambda_pref"] > 0:
            tgt, ok = None, S.get("kd_tok_ok")
            if S.get("pref") is not None:
                tgt = S["pref"].float().reshape(-1)
            elif "ref_p_pdm" in targets and "ref_gt_ok" in targets:
                dev = out["pref_logp"].device
                tgt = targets["ref_p_pdm"].float().reshape(-1).to(dev)
                g = targets["ref_gt_ok"].bool().reshape(-1).to(dev)
                ok = g if ok is None else ok.bool() & g
            if tgt is not None:
                lp = V4L.pref_loss(out["pref_logp"], tgt, ok)
                logs["v4/pref_huber"] = _f(lp)
                inner = inner + q["lambda_pref"] * lp
        hk = out.get("haz")
        # evidence readout only: the range readout's occ_logit has other time bins / semantics (its fields are trained
        # by the constraint KD, kd_constraint, and optionally by its own GT field aux, rg_gt_aux below)
        if hk is not None and "occ_logit" in hk and "occ_logit_u" not in hk and q["lambda_ev"] > 0:
            dev = hk["occ_logit"].device
            ref = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in targets.items() if k in V4L.EV_REF_KEYS}
            l_occ, l_m, s_ev = V4L.evidence_aux(hk, ref)
            logs.update({f"v4/{k}": float(v) for k, v in s_ev.items()})
            inner = inner + q["lambda_ev"] * (l_occ + l_m)
        gq = self.cfg.rg_gt_aux
        if gq["enabled"] and hk is not None and "occ_logit_u" in hk:      # refiner v4_3 (c): the student's own fields
            l_go, l_gm, s_g = self._rg_gt_aux(hk, S, targets, student)
            logs.update({f"v4/{k}": float(v) for k, v in s_g.items()})
            logs["v4/rg_gt_occ"], logs["v4/rg_gt_margin"] = _f(l_go), _f(l_gm)
            inner = inner + gq["lambda"] * (l_go + l_gm)
        if S.get("E") is not None and student is not None and getattr(student, "corrector", None) is not None \
                and q["lambda_awr"] > 0 and s_lab is not None:
            inner = inner + q["lambda_awr"] * self._awr(student, out, S, s_lab, logs)
        haz = out.get("extra_haz") if S.get("extra") is not None else out.get("haz")
        if haz is not None and "a_nc" in haz and S.get("tidx_lab") is not None and q["lambda_station"] > 0:
            l_st, st = V4L.station_bce(haz, S["tidx_lab"].long(), S["y"].float(), S["ok_lab"].bool())
            logs.update({f"v4/{k}": float(v) for k, v in st.items()})
            inner = inner + q["lambda_station"] * l_st
        return inner

    def _v5_terms(self, inner: torch.Tensor, out: Dict, S: Dict, logs: Dict[str, float], targets, student, T: Dict,
                  rel_rows: Optional[Dict[str, torch.Tensor]], ego, r: float) -> torch.Tensor:
        """refiner v5 student terms (ck_e2e2.v4.v5; off -> unchanged): reliability BCE of the student rows against the
        GT t0 objects, per-state hazard BCE against the GT executed blocks, L_refine(X1) of the corrector A_S on the KD
        set's parent columns, Z-KD of the per-state reads to the v5 teachers (ramped by r).  GT (ref_*) = targets only.
        Logs v5/*."""
        import torch.nn.functional as F
        from ..refiner import gt_future as GF
        from .v4 import lqr as LQR
        from .v5 import refine_loss as RL
        from .v5.student_objects import student_reliability_targets
        q = self.cfg.v5
        hk = out.get("haz")
        if not q["enabled"] or hk is None or "zq" not in hk:
            return inner
        dev = hk["zq"].device
        tg = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in (targets or {}).items() if k.startswith("ref_")}
        B, K = S["cand"].shape[:2]
        gt_ok = (tg["ref_gt_ok"].reshape(-1).bool() if "ref_gt_ok" in tg else torch.zeros(B, dtype=torch.bool,
                                                                                            device=dev))
        gt_ok = gt_ok & S["kd_tok_ok"].bool().to(dev)
        idx = torch.nonzero(gt_ok, as_tuple=True)[0]
        n = int(idx.numel())
        gt = RL.gt_of(tg, idx) if n else None
        zs = int(q["z_stride"])
        tz = torch.arange(LQR.NT, dtype=torch.float64, device=dev)[::zs] * LQR.DT
        logs["v5/n_gt"] = float(n)
        if "unknown_frac" in hk:
            logs["v5/unknown_frac"] = float(hk["unknown_frac"].detach().float().mean())
        if rel_rows is not None and n:
            if q["lambda_rel_a"] > 0 and "rel_a" in hk:
                O = {k: v[idx] for k, v in rel_rows.items()}
                y, m = student_reliability_targets(O, *RL.gt_t0_objects(gt))
                av = hk["rel_a"][idx].float().clamp(1e-4, 1 - 1e-4)
                mf = m.float()
                l_a = (F.binary_cross_entropy(av, y, reduction="none") * mf).sum() / mf.sum().clamp(min=1.0)
                inner = inner + q["lambda_rel_a"] * l_a
                logs.update({"v5/rel_a_bce": _f(l_a), "v5/rel_a_pos": float((y * mf).sum() / mf.sum().clamp(min=1.0)),
                             "v5/rel_a_mean": float((av.detach() * mf).sum() / mf.sum().clamp(min=1.0)),
                             "v5/rel_n_obj": float(mf.sum() / max(n, 1))})
            if q["lambda_rel_h"] > 0 and hk["zq"].shape[-1] >= 6:
                Kh = max(1, min(int(q["rel_h_cols"]), K))
                H = hk["zq"][idx][:, :Kh, :, -2:].float().clamp(1e-4, 1 - 1e-4)
                yb, known = RL.gt_state_block(hk["exec_states"][idx][:, :Kh, ::zs], tz, gt)
                mk = known.float()[..., None].expand_as(H)
                l_h = (F.binary_cross_entropy(H, yb, reduction="none") * mk).sum() / mk.sum().clamp(min=1.0)
                inner = inner + q["lambda_rel_h"] * l_h
                logs.update({"v5/rel_h_bce": _f(l_h), "v5/rel_h_pos": float((yb * mk).sum() / mk.sum().clamp(min=1.0))})
        if getattr(student, "corrector_v5", None) is not None and "ctx_all" in out:
            cols = refine_cols_parents(S["vtype_kd"].to(dev), q["ref_cols"])
            o_like = {"ctx": out["ctx_all"][:, :K], "score_logit": out["score_logit"], "haz": hk, "w_lat": out["w_lat"]}
            rf = student.refine_drafts(o_like, S["cand"].float(), cols, ego)
            inner = inner + 0.0 * (rf["mu"].sum() + rf["log_sig"].sum())
            logs["v5/ref_mu_abs"] = float(rf["mu"].detach().abs().mean())
            if n and q["lambda_ref"] > 0:
                eds = ego[2].float()
                w = RL.parse_weights(q["ref_w"])
                l_ref, s_ref = RL.refine_loss(rf["tau1"][idx], rf["tau_base"][idx], eds[idx, 0], eds[idx, 2], gt, w)
                l_u = (rf["mu"][idx].float() ** 2).mean()
                inner = inner + q["lambda_ref"] * (l_ref + float(w.get("u", 0.0)) * l_u)
                logs.update({f"v5/{k}": float(v) for k, v in s_ref.items()})
                logs["v5/ref_loss"] = _f(l_ref)
        if q["z_kd"] and "zq_det" in T and "zq_map" in T:
            zs_ = hk["zq"].float()
            zd, zm = T["zq_det"].float().to(dev), T["zq_map"].float().to(dev)
            kd_ok = T["kd_ok"].bool().to(dev) & S["kd_tok_ok"].bool().to(dev)[:, None]
            unk_s = zs_[..., 3] > 0.5
            ch = [0, 1] + ([4, 5] if zs_.shape[-1] >= 6 and zd.shape[-1] >= 6 else [])
            md = (kd_ok[..., None] & ~unk_s & ~(zd[..., 3] > 0.5)).float()
            l_zd = (F.smooth_l1_loss(zs_[..., ch], zd[..., ch].detach(), reduction="none").mean(-1) * md).sum() \
                / md.sum().clamp(min=1.0)
            mm = (kd_ok[..., None] & ~unk_s & ~(zm[..., 3] > 0.5)).float()
            l_zm = (F.smooth_l1_loss(zs_[..., 2], zm[..., 2].detach(), reduction="none") * mm).sum() \
                / mm.sum().clamp(min=1.0)
            inner = inner + r * (q["lambda_zkd_det"] * l_zd + q["lambda_zkd_map"] * l_zm)
            logs.update({"v5/zkd_det": _f(l_zd), "v5/zkd_map": _f(l_zm), "v5/zkd_det_frac": float(md.mean()),
                         "v5/zkd_map_frac": float(mm.mean())})
        return inner

    def _rg_gt_aux(self, haz: Dict[str, torch.Tensor], S: Dict, targets, student
                   ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """ck_e2e2.v4.rg_gt_aux: range_targets.gt_field_aux of the student's KD-set fields (haz = out['haz'], its
        own-parent rows of S['cand']) against the GT of the batch's ref_* (losses.EV_REF_KEYS; tokens without GT
        masked; a batch without them -> zero terms and rg_n 0), B target rule = the student's rg_stopped_aware, bins =
        its ev_tbin_s, grid = v4.corridor_version.  -> (l_occ, l_margin, stats rg_*)."""
        from .v4 import losses as V4L
        from .v4 import range_targets as RT
        v4n = getattr(student, "v4", None) or self.cfg.v4_net()
        dev = haz["occ_logit_u"].device
        ref = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in (targets or {}).items() if k in V4L.EV_REF_KEYS}
        gq = self.cfg.rg_gt_aux
        return RT.gt_field_aux(haz, S["cand"].to(dev), ref, tbin_s=float(v4n.ev_tbin_s),
                               max_parents=int(gq["max_parents"]), corridor_version=int(self.cfg.v4["corridor_version"]),
                               stopped_aware=bool(v4n.rg_stopped_aware))

    def _awr(self, student, out: Dict, S: Dict, s_lab: torch.Tensor, logs: Dict[str, float]) -> torch.Tensor:
        """AWR of the corrector on the generation's recorded edits (official labels, report 61 credit): log pi(u) of
        the recorded controls under the current policy for the recorded sources; credit over the label set G + the
        edits with the current critic-only selection scores (detached), A_loc against the source's official PDMS."""
        from .v4.corrector import awr_loss, credit, critic_select_score
        E, nG = S["E"], int(S["n_G"])
        feat, mem, ego, pref = out["scene"]
        # fp32 outside the loss's autocast: the scene features come from student_forward2 (fp16 under amp), the
        # re-score runs here after that scope has closed (review 79 F8)
        feat, mem = feat.float(), mem.float()
        with torch.autocast(device_type=feat.device.type, enabled=False):
            logp = student.corrector_logp(feat, mem, ego, E["E_src_traj"].float(), E["E_src_par"].float(),
                                          E["E_u"].float(), pref=None if pref is None else pref.float())
        s_all = critic_select_score(s_lab.detach())
        KE = E["E_ok"].shape[1]
        R_all = torch.cat([E["G_pdms"].float(), E["E_pdms"].float()], 1)
        ok_all = S["ok_lab"].bool()
        cols = (nG + torch.arange(KE, device=s_all.device))[None].expand(s_all.shape[0], -1)
        c = credit(s_all, R_all, ok_all, cols, R_src=E["E_src_pdms"].float())
        l = awr_loss(logp, c["A"], c["ok"])
        n = int(c["ok"].sum())
        logs["v4/awr"] = _f(l)
        logs["v4/awr_n"] = float(n)
        if n:
            logs["v4/awr_A_mean"] = float(c["A"][c["ok"]].mean())
            logs["v4/awr_Aloc_mean"] = float(c["A_loc"][c["ok"]].mean())
            logs["v4/edit_better_frac"] = float((c["A_loc"][c["ok"]] > 1e-6).float().mean())
        return l

    # -------------------------------------------------------------------------------------- inference
    @torch.no_grad()
    def infer(self, student, features, predictions) -> Dict[str, torch.Tensor]:
        """eval-mode outputs over the 96-candidate pool (SPEC s5-1): fp32, decode slope 0, z_lon = 0.  Columns
        c = k * 6 + v (k = v2 top-16 rank, v = VNAMES); ck2_cand96[:, ::6] = top-16, [:, 0] = v2's trajectory.
        predictions['trajectory'] is untouched (v2's, NU30)."""
        from ..refiner.e2e import ego_inputs

        bev = predictions["bev_embed"]
        dev = bev.device
        top = student_topk(predictions, K16)
        status = features["status_feature"].float().to(dev)
        v0 = ego_inputs(status)[0]
        V = C2.online_variants(top["cand"], v0_from_status(status), self.cfg.variants)
        cand, fin, par = infer_pool(V["traj96"], top["cand"])                 # c = k * 6 + v -> parent k
        ce = None
        computed = bool(getattr(getattr(student, "v4", None), "computed_edits", False))
        corr = getattr(student, "corrector", None) is not None
        v5s = bool(getattr(getattr(student, "v5", None), "on", False))           # refiner v5 student
        corr5 = getattr(student, "corrector_v5", None) is not None
        rel_rows = student_rel_rows(predictions, self.cfg.v5) if v5s else None
        valid = V["valid96"] & fin
        po = None
        with torch.autocast(device_type=dev.type, enabled=False):
            if v5s and not (corr or computed):
                # refiner v5: score the 96 with the relation rows of the student's own det head (+ context for A_S)
                ego = ego_inputs(status)
                feat, mem = student.trunk.scene(bev_sgrid(bev.float()), cand.shape[0])
                pref = student.scene_pref(mem, ego)
                kw5 = {"rel_objs": rel_rows} if rel_rows is not None else {}
                po = student.score_set(feat, mem, ego, cand, par, pref=pref, return_ctx=corr5, **kw5)
                out = {"score_logit": po["score_logit"], "w_lat": po["w_lat"]}
            elif corr or computed:
                # refiner v4 corrector A: score the 96 (+ corrector context), edit the critic's top-k, re-score;
                # v4_3 computed edits (M5): the rule edits of the critic's top-edit_k parents on the predicted fields
                # of the same parent pass (+ the corrector's residual when both are on)
                ego = ego_inputs(status)
                feat, mem = student.trunk.scene(bev_sgrid(bev.float()), cand.shape[0])
                pref = student.scene_pref(mem, ego)
                kw = {"return_pctx": True} if computed else {}           # corrector only: the old call exactly
                po = student.score_set(feat, mem, ego, cand, par, pref=pref, return_ctx=corr, **kw)
                out = {"score_logit": po["score_logit"], "w_lat": po["w_lat"]}
                if computed:
                    ce = student.computed_edits(feat, mem, ego, cand, par, po, pref=pref, valid=valid)
                else:
                    ce = student.correct_topk(feat, mem, ego, cand, par, po, pref=pref, valid=valid)
            else:
                out = student_forward2(student, bev_sgrid(bev.float()), cand, status, cand_parent=par)
        lat = lateral_decode(cand, out["w_lat"], v0, 0.0)
        res = {"ck2_cand_idx": top["idx"], "ck2_v2_final": top["final"], "ck2_v2_im": top["im"],
               "ck2_v2_sim": top["sim"], "ck2_cand96": V["traj96"], "ck2_valid96": valid,
               "ck2_score_logit": out["score_logit"].float(), "ck2_w_lat": out["w_lat"].float(),
               "ck2_e_lat": lat["e_lat"][..., 2:].float(), "ck2_lat_traj": lat["traj"].float()}
        if po is not None and corr5 and self.cfg.v5["infer_refine"]:
            # refiner v5 corrector A_S on the deployed pick (select2.select96, the deployment blend; infer_select's beta
            # / set when given, else beta 0.5 over 'all'): tau1 = decode(pick, mu_lon, w_lat_head + mu_lat) is the
            # v5 output (ck2_v5_traj; ck2_traj stays the 96-pool selection for the existing eval path)
            from . import select2 as S2
            bsel = self.cfg.infer_select or {"beta": 0.5, "set": "all"}
            pick = S2.select96(res["ck2_v2_final"], res["ck2_v2_im"], torch.sigmoid(res["ck2_score_logit"]),
                               res["ck2_valid96"], float(bsel["beta"]), types=S2.VARIANT_SETS[bsel["set"]])
            pick = torch.as_tensor(np.asarray(pick.cpu() if torch.is_tensor(pick) else pick), dtype=torch.long,
                                   device=dev)
            with torch.autocast(device_type=dev.type, enabled=False):
                rf = student.refine_drafts(po, cand.float(), pick[:, None], ego_inputs(status))
            res.update(ck2_v5_traj=rf["tau1"][:, 0].float(), ck2_v5_src=pick, ck2_v5_mu=rf["mu"][:, 0].float())
        if ce is not None:          # the K edits (no lateral head on top: the corrector's / the rule's own controls)
            et = ce["edit_traj"].float()
            fin_e = torch.isfinite(et).flatten(2).all(-1)
            if computed:            # not selectable: a filler slot (src_valid False, an excluded pool column bitwise)
                ev, kind = ce["src_valid"].bool() & fin_e, ce["edit_kind"].to(torch.int8)
                if not corr:        # and a no-edit slot (kind 0: its source column bitwise, = computed_edits_rec's
                    ev = ev & (ce["edit_kind"] > 0)   # rule) -- its re-score ties the pool's only up to GEMM batch-
                    #                 shape noise on GPU, so a selectable copy would randomly drop the lateral head
                    #                 (an edit pick is submitted as is); with the corrector it carries its residual
            else:                   # corrector A: every shortlisted (valid) source edited; no rule kind (-1)
                ev = torch.gather(valid, 1, ce["src_idx"]) & fin_e
                kind = torch.full(ev.shape, -1, dtype=torch.int8, device=ev.device)
            res.update(ck2_edit_traj=et, ck2_edit_src=ce["src_idx"].long(), ck2_edit_u=ce["u"].float(),
                       ck2_edit_score_logit=ce["edit_score_logit"].float(), ck2_edit_valid=ev, ck2_edit_kind=kind)
        sel = self.cfg.infer_select
        if sel is not None and ((ce is not None and "edits" not in sel) or (ce is None and sel.get("edits"))):
            raise ValueError(f"ck_e2e2.infer_select: the student {'emits' if ce is not None else 'has no'} edits, "
                             f"infer_select.edits is {sel.get('edits')!r} (CKE2E2Config.validate rule: explicit "
                             f"true = 96 + K / false = 96 only when the student emits edits)")
        if sel is not None and sel.get("edits"):
            # 96 + K (eval_ck2's headline rule, select96_edits): an edit pick submits the edit itself (its own
            # controls, no lateral head on top), a pool pick as select2.final_traj; ck2_sel_idx 96 + j = edit slot j
            from . import select2 as S2
            cpu = lambda a: a.detach().float().cpu().numpy()                       # noqa: E731
            idx = select96_edits(cpu(res["ck2_v2_final"]), cpu(res["ck2_v2_im"]),
                                 cpu(torch.sigmoid(res["ck2_score_logit"])), res["ck2_valid96"].cpu().numpy(),
                                 cpu(torch.sigmoid(res["ck2_edit_score_logit"])), res["ck2_edit_src"].cpu().numpy(),
                                 res["ck2_edit_valid"].cpu().numpy(), sel["beta"], S2.VARIANT_SETS[sel["set"]])
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=dev)
            is_e = idx_t >= S2.G_K
            ip = torch.where(is_e, torch.zeros_like(idx_t), idx_t)
            traj, _src = S2.final_traj(res["ck2_cand96"], res["ck2_lat_traj"], ip, sel["lat_mode"])
            je = (idx_t - S2.G_K).clamp(0, res["ck2_edit_traj"].shape[1] - 1)
            et = res["ck2_edit_traj"][torch.arange(idx_t.shape[0], device=dev), je]
            res["ck2_sel_idx"] = idx_t
            res["ck2_traj"] = torch.where(is_e[:, None, None], et.to(traj.dtype), traj).float()
        elif sel is not None:
            from . import select2 as S2
            idx = S2.select96(res["ck2_v2_final"], res["ck2_v2_im"], torch.sigmoid(res["ck2_score_logit"]),
                              res["ck2_valid96"], sel["beta"], types=S2.VARIANT_SETS[sel["set"]])
            idx_t = torch.as_tensor(np.asarray(idx.cpu() if torch.is_tensor(idx) else idx), dtype=torch.long,
                                    device=dev)
            traj, _src = S2.final_traj(res["ck2_cand96"], res["ck2_lat_traj"], idx_t, sel["lat_mode"])
            res["ck2_sel_idx"] = idx_t
            res["ck2_traj"] = torch.as_tensor(np.asarray(traj.cpu() if torch.is_tensor(traj) else traj),
                                              dtype=torch.float32, device=dev)
        return res

    # -------------------------------------------------------------------------------------- training loss
    def loss(self, student, features, targets, predictions, v2_loss: Optional[torch.Tensor] = None,
             v2_logs: Optional[Dict] = None, bev_kd=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        from ..refiner.e2e import ego_inputs
        from . import losses as L

        t0 = time.time()
        c = self.cfg
        bev = predictions["bev_embed"]
        dev = bev.device
        B = bev.shape[0]
        phase = self.phase(self.epoch)
        r = self.r(self.epoch_frac)
        status = features["status_feature"].float().to(dev)
        ego = ego_inputs(status)
        v0, a0 = ego[0], ego[1]
        rows = (targets["ck_row"].reshape(-1).long() if "ck_row" in targets
                else torch.full((B,), -1, dtype=torch.long, device=dev))
        self._check_rows(rows)
        toks = self.tokens_of(rows)
        human = targets["trajectory"].float().to(dev)
        fallback = student_topk(predictions, 1)["cand"][:, 0]
        logs: Dict[str, float] = {}

        # ---- candidates of this step
        tv = time.time()
        v0v = v0_from_status(status)                    # float64 hypot (= the variant label files' v0)
        rec = None
        if phase == "onpolicy":
            S = self.current_batch(predictions, v0v, rows, toks, targets, fallback)
            if self.recording(self.epoch):
                rec = (S["top"], S["V"])
        else:
            S = self.warmup_batch(targets, toks, fallback, dev)
            if phase == "warmup_record" and self.recording(self.epoch):
                top = student_topk(predictions, K16)
                rec = (top, C2.online_variants(top["cand"], v0v, c.variants))
        t_var = time.time() - tv
        n_rec = 0
        # refiner v4 corrector / v4_3 computed edits: the on-policy record also carries the step's edits -> recorded
        # after the forward
        computed = bool(getattr(getattr(student, "v4", None), "computed_edits", False))
        defer_rec = ((getattr(student, "corrector", None) is not None or computed) and phase == "onpolicy"
                     and "kd_col96" in S)
        if rec is not None and self.recorder is not None and not defer_rec:
            top, V = rec
            n_rec = self.recorder.add(rows, V["traj96"], V["valid96"], top["idx"], top["final"], v0v, self.gstep)
            self.cum["rec_tokens"] += n_rec

        # ---- teachers (online, every phase) on the step's KD candidates
        tt = time.time()
        tp = self.teachers(dev)
        tkw: Dict[str, Any] = {}
        if tp.det_temporal:                             # refiner-v3 temporal DET teacher: its causal object rows
            objs_det, obj_ok_det = kd_objs_of(targets, dev)
            tkw = {"objs_det": objs_det, "obj_ok_det": obj_ok_det}
        if c.kd_constraint["enabled"]:                  # refiner v4_3 constraint KD: the teachers' range fields
            tkw["fields"] = kd_constraint_request(c.kd_constraint["occ_channels"])
        v5c = c.v5
        if tp.det_rel:                                  # refiner v5 relation DET teacher: its object pack rows
            objs_r, ok_r = kd_objs_of(targets, dev)
            tkw.update(rel_objs_det=objs_r, rel_ok_det=ok_r)
        if v5c["enabled"] and v5c["z_kd"]:              # refiner v5 Z-KD: the teachers' per-state reads
            tkw["zq"] = True
        rel_rows = student_rel_rows(predictions, v5c)    # refiner v5 student relation rows (None when off)
        T = tp.run(targets["kd_bev_0"], targets["kd_ok_0"], targets["kd_bev_1"], targets["kd_ok_1"], S["cand"], status,
                   cand_ok=S["cand_ok"], parents=S.get("cand_parent"), family=S.get("vtype_kd"), **tkw)
        t_teacher = time.time() - tt

        # ---- student forward (one pass over KD candidates + label set)
        ts = time.time()
        rec_edits = rec is not None and self.recorder is not None and defer_rec
        fkw = {"return_pctx": True} if (rec_edits and computed) else {}       # off: the old call exactly
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=bool(c.amp and dev.type == "cuda")):
            if rel_rows is not None:
                fkw["rel_objs"] = rel_rows
            out = student_forward2(student, bev_sgrid(bev), S["cand"], status, extra=S["extra"],
                                   cand_parent=S.get("cand_parent"), extra_parent=S.get("extra_parent"), **fkw)
        s_kd = out["score_logit"].float()
        s_lab = s_kd if S["extra"] is None else out["extra_score_logit"].float()
        if rec_edits:
            top, V = rec
            ed = (self.computed_edits_rec(student, out, S, s_kd) if computed
                  else self.corrector_edits(student, out, S, s_kd))
            if ed is not None and "kind" in ed:
                kd_, cl_ = ed.pop("kind"), ed.pop("clear")
                logs["v4/edit_rec_valid_frac"] = float(ed["valid"].float().mean())
                for code, nm in ((1, "lon"), (2, "lat"), (3, "both")):
                    logs[f"v4/edit_rec_{nm}_frac"] = float((kd_ == code).float().mean())
                # the computed edits (kind > 0) whose edit reads inside the predicted tube (NC, TTC, DAC; edits.py
                # edit_clear) -- noisy predicted fields can leave many outside (0 when no edit was made)
                ek = kd_ > 0
                logs["v4/edit_clear_frac"] = float(cl_[ek].float().mean()) if bool(ek.any()) else 0.0
            n_rec = self.recorder.add(rows, V["traj96"], V["valid96"], top["idx"], top["final"], v0v, self.gstep,
                                      edits=ed)
            self.cum["rec_tokens"] += n_rec
        corr = lateral_decode(S["cand"], out["w_lat"], v0, c.lon_st_slope)
        e_s = corr["e_lat"][..., 2:].float()
        Kc = S["cand"].shape[1]

        # ---- loss
        l_bce, per = L.score_bce(s_lab, S["y"], S["ok_lab"])
        inner = c.lambda_score * l_bce
        ref_gt_ok = (targets["ref_gt_ok"].reshape(-1).bool().to(dev) if "ref_gt_ok" in targets
                     else torch.zeros(B, dtype=torch.bool, device=dev))
        idx = L.surrogate_index(ref_gt_ok & S["sur_tok_ok"], human)
        n = int(idx.numel())
        n_s = int(S["sur_pos"].shape[1])
        l_sur = bev.new_zeros((), dtype=torch.float32)
        terms = {k: 0.0 for k in SUR_TERMS + ("P1_minus_P0",)}
        nonfin = 0
        if n > 0 and c.lambda_sur:
            ref = {k: v.to(dev) for k, v in targets.items() if k.startswith("ref_")}
            tau_sur = _gather_c(S["cand"], S["sur_pos"])
            sb = L.surrogate_batch_k(ref, idx, tau_sur, human, v0, a0)
            drows = (idx[:, None] * Kc + S["sur_pos"][idx]).reshape(-1)
            dec = L.restrict_dec(corr["raw"], drows)
            l_sur, terms, nonfin = L.correction_loss(dec, sb, n, n_s, SUR_WEIGHTS, SUR_MARGINS)
            inner = inner + c.lambda_sur * l_sur
        l_sur_val = _f(l_sur)
        kd_ok = T["kd_ok"].bool() & S["kd_tok_ok"][:, None]
        ki = [CK_KEYS.index(k) for k in c.kd_score_keys]      # user 2026-10-08: comfort excluded from score KD
        kd_ok_s = kd_ok
        if c.v4["kd_parent_only"]:                            # refiner v4 I2: score KD on the parent columns only
            kd_ok_s = kd_ok & (S["vtype_kd"] == 0)
        if c.kd_score_scope == "near":                        # refiner v4_3 M6: NC / DAC KD near the hazard only
            nc_tok, dac_c = self._near_scope(targets, status, S, logs)
            l_ks = score_kd_near(s_kd[..., ki], T["kd_score_prob"][..., ki], kd_ok_s, c.kd_score_keys, nc_tok, dac_c)
        else:
            l_ks = L.score_kd_bce(s_kd[..., ki], T["kd_score_prob"][..., ki], kd_ok_s)
        inner = inner + (r * c.lambda_kd_score) * l_ks
        if c.kd_constraint["enabled"]:                        # refiner v4_3 M6: teacher fields -> student fields
            kq = c.kd_constraint
            l_co, l_cm, st_c = constraint_kd(out["haz"], T, S["cand_ok"].bool() & S["kd_tok_ok"][:, None], kq)
            l_cn = kq["lambda_occ"] * l_co + kq["lambda_margin"] * l_cm
            inner = inner + r * l_cn
            logs.update({f"v4/{k}": float(v) for k, v in st_c.items()})
            logs["v4/cn_occ"], logs["v4/cn_margin"] = _f(l_co), _f(l_cm)
            logs["ck2/wt_kd_constraint"] = c.loss_weight * r * _f(l_cn)
        m_lat = kd_ok & S["lat_mask"]
        l_lat = _masked_l1(e_s, T["kd_e_lat"], m_lat)
        l_lat_val = _f(l_lat)
        lk = c.lat_kd
        if lk["balance"] == "fixed":
            w = float(lk["fixed"]) if self.mb >= lk["start_mb"] else 0.0
        else:
            # EMA inputs = the all-rank means of this micro-batch (ONE 2-float all-reduce, reached by every rank on
            # every training mb before the conditional BEV-KD collective): the EMA state is identical on every rank,
            # so the rank-0 checkpoint restores all ranks exactly (report 48 F4); single process: unchanged values
            ema_sur, ema_kd = _dist_mean_many([c.lambda_sur * l_sur_val, l_lat_val], dev)
            self.ema.update(ema_sur, ema_kd)
            w = float(self.ema.weight(self.mb))
        inner = inner + (r * w) * l_lat
        inner = self._v4_terms(inner, out, S, logs, targets, student, s_lab)
        inner = self._v5_terms(inner, out, S, logs, targets, student, T, rel_rows, ego, r)
        loss = c.loss_weight * inner
        loss = loss + 0.0 * (out["score_all"].sum() + out["w_lat_all"].sum())
        if out.get("pref_logp") is not None:
            loss = loss + 0.0 * out["pref_logp"].sum()
        if getattr(student, "corrector", None) is not None:      # in the DDP graph every step (AWR terms: _v4_terms)
            loss = loss + 0.0 * sum(p.sum() for p in student.corrector.parameters())
        for mod_name in ("corrector_v5", "relation"):              # refiner v5 modules: in the DDP graph every step
            mod = getattr(student, mod_name, None)
            if mod is not None:
                loss = loss + 0.0 * sum(p.sum() for p in mod.parameters())
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

        # ---- grad share (every grad_share_every micro-batches)
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
            for t, tt in term_t.items():          # per-teacher weighted BEV-KD gradient (= lam_t |dL_t / dBEV|)
                gg = (torch.autograd.grad(tt, bev, retain_graph=True, allow_unused=True)[0]
                      if isinstance(tt, torch.Tensor) and tt.requires_grad else None)
                logs[f"gnorm/bev_bevkd_{t}"] = 0.0 if gg is None else float(gg.float().norm())

        # ---- logs
        lw = c.loss_weight
        logs.update({
            "ck2/loss": loss_val if not bad_loss else 0.0, "ck2/loss_nonfinite": float(bad_loss),
            "ck2/bce": _f(l_bce), "ck2/kd_score": _f(l_ks), "ck2/sur": l_sur_val, "ck2/sur_nonfinite": float(nonfin),
            "ck2/lat_kd": l_lat_val, "ck2/w_lat": w, "ck2/r": r, "ck2/phase": float(PHASE2_CODE[phase]),
            "ck2/mb": float(self.mb), "ck2/n_gt": float(n),
            "ck2/wt_bce": lw * c.lambda_score * _f(l_bce), "ck2/wt_sur": lw * c.lambda_sur * l_sur_val,
            "ck2/wt_kd_score": lw * r * c.lambda_kd_score * _f(l_ks), "ck2/wt_lat_kd": lw * r * w * l_lat_val,
        })
        for k, v in (per or {}).items():
            logs[f"ck2/bce_{k}"] = float(v)
        for k, v in (terms or {}).items():
            logs[f"ck2/t_{k}"] = float(v)
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
            mf = m_lat.float()
            den = float(mf.sum().clamp(min=1.0))
            logs["ck2/e_abs"] = float((e_s.detach().abs().mean(-1) * mf).sum()) / den
            logs["ck2/e_abs_T"] = float((T["kd_e_lat"].abs().mean(-1).nan_to_num(0.0) * mf).sum()) / den
            logs["ck2/e_abs_det"] = float((T["e_det"].abs().mean(-1).nan_to_num(0.0) * mf).sum()) / den
            logs["ck2/e_abs_map"] = float((T["e_map"].abs().mean(-1).nan_to_num(0.0) * mf).sum()) / den
            logs["ck2/lat_disagree"] = float(((T["e_det"] - T["e_map"]).abs().mean(-1).nan_to_num(0.0)
                                              * mf).sum()) / den
            logs["ck2/n_cand"] = float(Kc + (0 if S["extra"] is None else S["extra"].shape[1]))
            if "obj_ok_det" in tkw:                     # temporal DET teacher only (CK2 teachers: logs unchanged)
                logs["ck2/kd_obj_ok_frac"] = float(tkw["obj_ok_det"].float().mean())
        self.cum["sur_nonfinite"] += int(nonfin)
        logs["ck2/variants_ms"] = 1e3 * t_var
        logs["ck2/teacher_ms"] = 1e3 * t_teacher
        logs["ck2/student_ms"] = 1e3 * t_student
        logs["ck2/rec_tokens"] = float(n_rec)
        logs["ck2/total_ms"] = 1e3 * (time.time() - t0)
        if term is not None:
            loss = loss + term
        self.mb += 1
        return loss, {k: torch.tensor(float(v), device=dev) for k, v in logs.items()}

    def _bev_kd_term(self, bev_kd, bev: torch.Tensor, targets, v2_logs: Optional[Dict], logs: Dict[str, float]
                     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """-> (term, {t: term_t}): term = sum_t term_t, term_t = adapter_weight * L_t(GradScale(bev_embed, lam_t /
        adapter_weight)) (BEV gradient lam_t * dL_t / d bev_embed), lam_t = teacher t's own RatioController weight of
        this mb.  On the v2 'gnorm/plan' micro-batches the unit BEV gradient norm of every teacher's loss is measured
        (one all-reduce of the per-teacher vector, mean over the ranks) and each controller is updated against the
        same g_plan (the new lam_t apply from the next mb).  teachers [det] = the old single-controller arm exactly.
        A non-finite term is replaced by 0 * (adapter parameters) and counted ({} per-teacher terms then)."""
        teachers = tuple(bev_kd.teachers)
        ctrl = self.bev_ctrl
        if ctrl is None:
            from .bev_kd_arm import controllers_from_cfg
            ctrl = self.bev_ctrl = controllers_from_cfg(self.cfg.bev_kd, teachers)
        if tuple(ctrl.teachers) != teachers:
            raise ValueError(f"BEV-KD arm teachers {list(teachers)} != controller teachers {list(ctrl.teachers)}")
        lams = ctrl.weights(self.mb)
        bo = bev_kd(bev, targets, lams)
        term = bo["term"]
        term_t = {t: bo[f"term/{t}"] for t in teachers}
        bad = not math.isfinite(_f(term))
        if bad:
            self.cum["bevkd_nonfinite"] += 1
            term = 0.0 * sum(p.sum() for p in bev_kd.parameters() if p.requires_grad)
            term_t = {}
        measure = (v2_logs is not None and "gnorm/plan" in v2_logs and bev.requires_grad and torch.is_grad_enabled())
        if measure:
            g_loc = bev_kd.unit_grad_norms(bo, teachers) if not bad else {t: float("nan") for t in teachers}
            g_all = _dist_mean_many([g_loc[t] for t in teachers], bev.device)   # every rank reaches it on the same mb
            g_unit = dict(zip(teachers, g_all))
            g_plan = _f(v2_logs["gnorm/plan"])
            ctrl.update(g_unit, g_plan)
            logs["bevkd/g_plan"] = g_plan
            for t in teachers:
                g = g_unit[t]
                logs[f"bevkd/{t}/g_kd_unit"] = g
                logs[f"bevkd/{t}/ratio_now"] = (lams[t] * g / max(g_plan, 1e-30) if math.isfinite(g)
                                                else float("nan"))
        B = int(bev.shape[0])
        self.cum["bevkd_tok"] = self.cum.get("bevkd_tok", 0) + B
        for t in teachers:
            logs[f"bevkd/{t}/loss"] = _f(bo[f"raw/{t}"])
            logs[f"bevkd/{t}/lam"] = float(lams[t])
            logs[f"bevkd/{t}/ok_frac"] = _f(bo[f"ok_frac/{t}"])
            for k in ("cos", "fve", "w_dev"):
                logs[f"bevkd/{t}/{k}"] = _f(bo[f"{k}/{t}"])
            miss = B - int(bo[f"n_ok/{t}"])
            self.cum[f"bevkd_miss_{t}"] = self.cum.get(f"bevkd_miss_{t}", 0) + miss
        logs["bevkd/term"] = _f(term)
        logs["bevkd/nonfinite"] = float(bad)
        return term, term_t


# ----------------------------------------------------------------------------------------------- callback
def ck2_parameters(agent) -> List[torch.nn.Parameter]:
    """Trainable student CK parameters (lon / gate heads have requires_grad False and are excluded)."""
    return [p for p in agent.ck_student.parameters() if p.requires_grad]


STEP_LOG_PREFIXES = ("ck2/", "bevkd/", "gnorm/", "loss", "v4/", "v5/")   # v4 / v5: refiner terms


def make_ck2_callback(agent):
    import pytorch_lightning as pl

    class CKE2E2Callback(pl.Callback):
        """Schedule (epoch, epoch_frac, rank), CandRecorder2 open / close, LabelStore2 refresh, the CK-only gradient
        clip and the non-finite step skip (before Lightning's global clip), rank-0 steps_rank0.jsonl / epochs.jsonl,
        and the CKE2E2 state (lateral-KD EMA, BEV-KD controller, mb, counters) in the Lightning checkpoint.  Lightning
        writes rank 0's state and loads it on every rank: complete for the EMA and the BEV-KD controllers (both fed
        all-reduced inputs, identical on every rank); the counters in 'cum' are rank-0 diagnostics."""

        def __init__(self):
            super().__init__()
            self._f = None
            self._n = 0
            self._t_start = self._t_end = None

        @property
        def state_key(self) -> str:
            return "CKE2E2Callback"

        def state_dict(self):
            return agent._ck_e2e2.state_dict()

        def load_state_dict(self, state_dict):
            agent._ck_e2e2.load_state_dict(state_dict)

        # ---- helpers
        def _io(self) -> Path:
            p = Path(agent._ck_e2e2.cfg.io_dir)
            p.mkdir(parents=True, exist_ok=True)
            return p

        def _append(self, name: str, rec: Dict) -> None:
            with open(self._io() / name, "a") as f:
                f.write(json.dumps(rec) + "\n")

        # ---- epochs
        def on_train_epoch_start(self, trainer, pl_module):
            ck = agent._ck_e2e2
            ck.epoch = int(trainer.current_epoch)
            ck.epoch_frac = float(ck.epoch)
            ck.rank = int(getattr(trainer, "global_rank", 0))
            ck.world_size = int(getattr(trainer, "world_size", 1))
            if ck.recorder is not None:            # left over from an interrupted epoch: no DONE
                ck.recorder.flush()
                ck.recorder = None
            phase = ck.phase(ck.epoch)
            rec = None
            if ck.recording(ck.epoch) and phase != "warmup":
                ck.recorder = CandRecorder2(ck.cfg.io_dir, ck.rank, ck.world_size, ck.epoch, ck.cfg.rec_chunk_tokens)
                rec = ck.recorder.attempt
            stats = ck.refresh_labels() if phase == "onpolicy" else None
            if getattr(trainer, "is_global_zero", True):
                warn = label_supply_warning(ck, stats) if phase == "onpolicy" else None
                self._append("epochs.jsonl", {"event": "epoch_start", "epoch": ck.epoch, "phase": phase,
                                              "recording": rec is not None, "attempt_rank0": rec,
                                              "labels": stats, "label_warning": warn, "mb": ck.mb,
                                              "gstep": int(trainer.global_step),
                                              "world_size": ck.world_size, "time": time.time(),
                                              "ema": ck.ema.state_dict(),
                                              "bev_ctrl": None if ck.bev_ctrl is None else ck.bev_ctrl.state_dict(),
                                              "cum": dict(ck.cum)})
                if warn and warn.get("warn"):
                    print("\n" + "!" * 100 + f"\n[ck_e2e2] WARNING epoch {ck.epoch}: {warn['msg']}\n" + "!" * 100,
                          flush=True)

        def on_train_epoch_end(self, trainer, pl_module):
            ck = agent._ck_e2e2
            if ck.recorder is not None:
                ck.recorder.close_epoch()
                n = ck.recorder.n_tokens
                ck.recorder = None
            else:
                n = None
            if getattr(trainer, "is_global_zero", True):
                self._append("epochs.jsonl", {"event": "epoch_end", "epoch": ck.epoch, "rec_tokens_rank0": n,
                                              "mb": ck.mb, "gstep": int(trainer.global_step), "time": time.time(),
                                              "skipped_steps": ck.skipped_steps, "cum": dict(ck.cum)})

        # ---- batches
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            ck = agent._ck_e2e2
            nb = trainer.num_training_batches
            nb = int(nb) if nb and math.isfinite(float(nb)) else 0
            ck.epoch_frac = float(trainer.current_epoch) + (float(batch_idx) / nb if nb > 0 else 0.0)
            ck.batch_idx = int(batch_idx)
            ck.gstep = int(trainer.global_step)
            ck.rank = int(getattr(trainer, "global_rank", 0))
            self._t_start = time.time()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            ck = agent._ck_e2e2
            now = time.time()
            if getattr(trainer, "is_global_zero", True) and (ck.mb % max(ck.cfg.log_every, 1) == 0):
                if self._f is None:
                    self._f = open(self._io() / "steps_rank0.jsonl", "a")
                rec = {"epoch": int(trainer.current_epoch), "batch": int(batch_idx), "gstep": int(trainer.global_step),
                       "mb": int(ck.mb), "epoch_frac": round(ck.epoch_frac, 6),
                       "sec_step": round(now - self._t_start, 4) if self._t_start else None,
                       "sec_wait": round(self._t_start - self._t_end, 4) if (self._t_end and self._t_start) else None,
                       "skipped_steps": ck.skipped_steps, "ck_clip_norm": ck.last_clip_norm}
                if torch.cuda.is_available():
                    rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                for k, v in (agent.latest_logs or {}).items():
                    if k.startswith(STEP_LOG_PREFIXES):
                        rec[k] = float(v)
                self._f.write(json.dumps(rec) + "\n")
                self._n += 1
                if self._n % 50 == 0:
                    self._f.flush()
            every = int(ck.cfg.label_refresh_every_mb)
            if every > 0 and ck.phase(ck.epoch) == "onpolicy" and ck.mb % every == 0:
                ck.refresh_labels()
            self._t_end = now

        # ---- optimiser
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            ck = agent._ck_e2e2
            allp = [p for p in pl_module.parameters() if p.grad is not None]
            if allp:
                gn_all = torch.stack([g.float() for g in torch._foreach_norm([p.grad for p in allp])])
                if not bool(torch.isfinite(gn_all).all()):
                    for p in allp:
                        p.grad = None
                    ck.skipped_steps += 1
                    if getattr(trainer, "is_global_zero", True):
                        print(f"[ck_e2e2] non-finite gradient at global_step {trainer.global_step}: optimiser step "
                              f"skipped (total {ck.skipped_steps})", flush=True)
                    return
            clip = float(ck.cfg.clip)
            params = [p for p in ck2_parameters(agent) if p.grad is not None]
            if clip > 0 and params:
                ck.last_clip_norm = float(torch.nn.utils.clip_grad_norm_(params, clip))

        # ---- end
        def _close(self):
            if self._f is not None:
                self._f.close()
                self._f = None

        def on_train_end(self, trainer, pl_module):
            ck = agent._ck_e2e2
            if ck.recorder is not None:
                ck.recorder.flush()
                ck.recorder = None
            self._close()
            if getattr(trainer, "is_global_zero", True):
                (self._io() / "TRAIN_DONE").write_text(json.dumps({"time": time.time(), "epoch": ck.epoch,
                                                                   "gstep": int(trainer.global_step), "mb": ck.mb}))

        def on_exception(self, trainer, pl_module, exception):
            ck = agent._ck_e2e2
            try:
                if ck.recorder is not None:
                    ck.recorder.flush()
                    ck.recorder = None
            finally:
                self._close()

    return CKE2E2Callback()
