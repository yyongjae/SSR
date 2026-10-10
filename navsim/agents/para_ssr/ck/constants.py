"""CK (Corridor candidate head) Phase 1 shared constants (report 44; spec/interface_contract.json 'core.constants').

Owned by core.  tools/ck (pipeline) and tools/ck/data (data) import these names and never redefine them.
No torch / navsim imports here (data workers import this module).
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------------------------- keys / sizes
# Score key order = anchor_planner.SIM_KEYS (no_at_fault_collisions, drivable_area_compliance, ego_progress,
# time_to_collision_within_bound, comfort): every [.., 5] score tensor uses this order.
CK_KEYS = ("nc", "dac", "ep", "ttc", "comfort")
CK_KEY_TO_SIM = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ep": "ego_progress",
                 "ttc": "time_to_collision_within_bound", "comfort": "comfort"}
# official per-trajectory label columns from score_trajectories.score_token (labels.npy [N, K, 9])
LABEL_COLS = ("nc", "dac", "ep", "ttc", "comfort", "ddc", "pdms", "raw_progress", "pdm_progress_eff")
LBL = {k: i for i, k in enumerate(LABEL_COLS)}
CK_LABEL_IDX = tuple(LBL[k] for k in CK_KEYS)        # labels[..., CK_LABEL_IDX] -> [.., 5] in CK_KEYS order
K_CAND = 16
N_CTRL = 6            # free decoder controls per branch (z_lon 6, w_lat 6)
N_POSE = 8            # poses at t = 0.5 .. 4.0 s
SCORE_SEED_OFFSET = 31337

# ----------------------------------------------------------------------------------------------- paths
REPO = Path(__file__).resolve().parents[4]           # repo root of this worktree (/workspace/yongjae/SSR-ck2)
CK_DATA = Path("/home/external-user/ssd/yongjae_refiner/ck")
DATA_ROOT = Path("/home/external-user/ssd/yongjae_refiner")

V2_CKPT = REPO / "v2_source/epoch=29-step=19950.ckpt"   # v2_source -> symlink to SSR-ck/v2_source
V2_CKPT_SHA16 = "e1cfdbddc5d61125"                   # measured file sha256[:16]; task text quoted 226829cf4126f949
V2_CKPT_SHA16_QUOTED = "226829cf4126f949"            # (user to confirm which is the original)
V2_TRAIN_CFG = REPO / "v2_source/code/hydra/config.yaml"
V2_NAVTEST_CSV = REPO / "v2_source/2026.09.20.01.28.06.csv"
V2_NAVTEST_PDMS = 0.88137
V2_NAVTEST_TOL = 0.002
ANCHORS = Path("/home/external-user/kyungmin/SSR-v2/data/planning_vb/trajectory_anchors_256.npy")

TEACHER_CACHE = {
    "det": {"navtrain": "/home/external-user/datasets/teacher_cache/bevfusion/cache_train_50x100",
            "navtest": "/home/external-user/datasets/teacher_cache/bevfusion/cache_val_50x100"},
    "map": {"navtrain": "/home/external-user/datasets/teacher_cache/resmap",          # train logs only (85,109)
            "navtest": "/home/external-user/datasets/teacher_cache/resmap/navtest"},
}   # informative (= refiner.data.TEACHER_ROOTS / refiner.resmap_cache.RESMAP_ROOTS); load through
#     refiner.data.TeacherCache.for_subset(s) / refiner.resmap_cache.ResmapCache.for_subset(s)
TEACHER_INIT = {"T": DATA_ROOT / "stageE/teachers/stageT4_T_fold0_seed0",
                "M": DATA_ROOT / "stageE/teachers/stageT4_M_fold0_seed0"}
NORM_FILE = {"T": "norm.npz", "M": "norm_map.npz"}  # per-arm BEV z-score file in a (v1 or CK) run dir

METRIC_CACHE = {"navtrain": DATA_ROOT / "metric_cache",
                "navtest": Path("/home/external-user/yongjae/SSR/data/exp/metric_cache")}
LEAD_SRC = {"navtrain_train": Path("/workspace/yongjae/ssd/yongjae_refiner/direction_analysis/uncertainty/out/"
                                   "lead_navtrain.parquet"),
            "navtest": Path("/workspace/yongjae/ssd/yongjae_refiner/direction_analysis/uncertainty/out/"
                            "lead_navtest.parquet")}

_TRAINVAL_LOGS = "/home/external-user/yongjae/SSR/data/dataset/navsim_logs/trainval"
_TRAINVAL_SENSORS = "/home/external-user/yongjae/SSR/data/dataset/sensor_blobs/trainval"
_SF = "navsim/planning/script/config/common/scene_filter"      # relative to REPO

SPLITS = {
    "navtrain_train": {
        "tokens": str(DATA_ROOT / "splits/e2e_train_trainlogs.parquet"), "n": 85109, "logs": 978,
        "sha16_sorted_tokens": "c3fa309cbde971e8",
        "navsim_logs": _TRAINVAL_LOGS, "sensor_blobs": _TRAINVAL_SENSORS, "scene_filter": f"{_SF}/navtrain.yaml",
        "teacher_subset": "navtrain", "gt_surrogate": True, "map_teacher": True,
        "metric_cache": str(METRIC_CACHE["navtrain"]),
        "role": "train (teachers + student), KD targets"},
    "navtrain_val": {
        "tokens": str(CK_DATA / "data-map/navtrain_val_tokens.parquet"), "n": 18179, "logs": 214,
        "sha16_sorted_tokens": "05b78ec43b95d954",
        "navsim_logs": _TRAINVAL_LOGS, "sensor_blobs": _TRAINVAL_SENSORS, "scene_filter": f"{_SF}/navtrain.yaml",
        "teacher_subset": "navtrain", "gt_surrogate": False, "map_teacher": False,
        "metric_cache": str(METRIC_CACHE["navtrain"]),
        "role": "held-out for v2 and CK: weight/blend selection, go/no-go, DET-teacher overfit gap"},
    "navtest": {
        "tokens": str(DATA_ROOT / "splits/navtest.parquet"), "n": 12146, "logs": 136,
        "sha16_sorted_tokens": None,
        "navsim_logs": "/workspace/navsim_workspace/dataset/navsim_logs/test/test",
        "sensor_blobs": "/workspace/navsim_workspace/dataset/sensor_blobs/test/test",
        "scene_filter": f"{_SF}/navtest.yaml",
        "teacher_subset": "navtest", "gt_surrogate": False, "map_teacher": True,
        "metric_cache": str(METRIC_CACHE["navtest"]),
        "role": "report once with settings fixed on navtrain_val; MAP/DET teacher rows descriptive only"},
}

# ----------------------------------------------------------------------------------------------- KD rules
# score KD target per key: 'det' | 'map' | 'mean' (mean of the available teacher probabilities); no map -> det for all
KD_SCORE_SOURCE = {"nc": "det", "dac": "map", "ep": "mean", "ttc": "det", "comfort": "mean"}
# correction KD target: longitudinal from DET, lateral from MAP (no map -> det for both)
KD_CTRL_SOURCE = {"lon": "det", "lat": "map"}
KD_CTRL_W = {"lon": 0.25, "lat": 1.0}

# ----------------------------------------------------------------------------------------------- surrogate / decoder
SUR_TERMS = ("col", "dac", "prog", "cmf", "mod", "ttc")
SUR_WEIGHTS = {"col": 1.0, "dac": 1.0, "ttc": 1.0, "prog": 2.0, "cmf": 0.1, "mod": 0.1}     # R_T4 / E2
SUR_MARGINS = {"m_col": 0.15, "m_dac": 0.05, "m_ttc": 0.15}   # differ from SurrogateConfig defaults: pass explicitly
LON_ST_SLOPE = {"train": 0.1, "eval": 0.0}
DECODE_MODE = "A"

# ----------------------------------------------------------------------------------------------- selection
SEL_W = (0.1, 0.5, 0.5, 1.0)          # v2 plan_reward_weights: w_im, w_nc, w_dac, w_rest
# named selection weight sets (select.resolve_w; every selection default stays SEL_W = 'default').  'noim' = SEL_W
# without the v2 imitation term (CK-only planner of the teacher evals); 'plugin' = w_im 0, NC 1, DAC 1, rest 1 (the
# PDMS-like product NC * DAC * (5 TTC + 2 C + 5 EP) in log form; optional, for comparison on navtrain_val).
SEL_W_SETS = {"default": SEL_W, "noim": (0.0,) + SEL_W[1:], "plugin": (0.0, 1.0, 1.0, 1.0)}
SEL_EPS = 1e-6                        # AnchorPlanner.weighted_reward eps
BETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

# ----------------------------------------------------------------------------------------------- loss defaults
LOSS_DEFAULTS = {"lambda_score": 1.0, "lambda_kd_score": 0.5, "lambda_sur": 1.0, "kd_ctrl_balance": "ema",
                 "kd_ratio": 1.0, "kd_ema_m": 0.99, "kd_ema_floor": 1e-4, "kd_weight_max": 10.0,
                 "kd_start_step": 500, "lambda_corr_score": 1.0, "lambda_lead": 0.0}
