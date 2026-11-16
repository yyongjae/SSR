from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple
import fnmatch
import hashlib
import json
import os
import cv2
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import logging, torch
from omegaconf import DictConfig
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml

import navsim

REPO_ROOT = Path(__file__).resolve().parents[3]


def check_navsim_import(navsim_file: Optional[str], repo_root: Path = REPO_ROOT) -> None:
    """The package must come from this script's repo.  The hydra config is read relative to this script, so with
    another checkout on the path (an editable install, no PYTHONPATH) that checkout's model would train with this
    checkout's config, without any error."""
    if navsim_file is None or Path(repo_root).resolve() not in Path(navsim_file).resolve().parents:
        raise RuntimeError(f"navsim is imported from {navsim_file}, not from this repo {repo_root}: set "
                           f"PYTHONPATH={repo_root}")


check_navsim_import(navsim.__file__)     # before any other navsim module is imported

from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.wandb_logging import build_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from navsim.agents.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)


def _worker_init(worker_id: int) -> None:
    """Keep every dataloader worker single-threaded.

    A worker is already a process, so its libraries' own thread pools add no
    parallelism -- they only oversubscribe the machine.  OpenCV defaults to one
    thread per core and is inherited through the fork, so N workers x C cores
    threads fight over C cores: measured here at 2,685 threads on 32 cores and
    491k context switches/second, which starved the GPUs to ~60% and made a
    step 18x slower than its compute cost.  PyTorch already pins its own
    intra-op threads in workers; OpenCV and the OpenMP runtimes do not.
    """
    cv2.setNumThreads(0)
    torch.set_num_threads(1)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"
NAVTEST_FILTER = REPO_ROOT / "navsim/planning/script/config/common/scene_filter/navtest.yaml"
# test sets of the challenge: never training data
TEST_SCENE_FILTERS = ("navtest", "navhard*", "private*", "warmup_test*")
# +train_data: navtrain (absent) = the navtrain scene filter on train_logs (the main recipe, build_datasets); the other
# names are token-table splits (build_table_datasets): constants.SPLITS[name]['tokens'], real + SimScale rows
TRAIN_DATA_DEFAULT = "navtrain"
TABLE_TRAIN_DATA = ("trainval_simscale",)
TABLE_TOKENS_CHECK = 2000     # rank 0's first training tokens TableInputGuard counts (= ck/online2.TOKENS_CHECK)
TABLE_STRICT = 0.999          # least share of them with a v2 score row / warm-up labels (= ck_e2e2.strict_rows)


def check_launch_route(env: Mapping[str, str]) -> None:
    """A plain call of this script gets the config defaults, which are not the recipe of any experiment (another
    backbone, BEV grid, map head, no KD, ...).  Runs come through scripts/experiment.sh (SSR_EXPERIMENT_SHA) or a script
    that states it knows (SSR_ALLOW_DIRECT=1)."""
    if not env.get("SSR_EXPERIMENT_SHA") and env.get("SSR_ALLOW_DIRECT") != "1":
        raise RuntimeError("run_training.py was started without a recipe: use scripts/experiment.sh train, or set "
                           "SSR_ALLOW_DIRECT=1 if the overrides are complete")


def check_log_splits(train_logs: Iterable[str], val_logs: Iterable[str], test_logs: Iterable[str]) -> None:
    train, val, test = set(train_logs), set(val_logs), set(test_logs)
    for name, shared in (("train / val", train & val), ("train / test", train & test), ("val / test", val & test)):
        if shared:
            raise RuntimeError(f"log split: {len(shared)} logs are in both {name}, e.g. {sorted(shared)[:3]}")


def check_scene_filter_name(name: Optional[str]) -> None:
    if name is not None and any(fnmatch.fnmatch(str(name), p) for p in TEST_SCENE_FILTERS):
        raise RuntimeError(f"scene_filter={name} is a test set: refused for training")


def navtest_tokens(path: Path = NAVTEST_FILTER) -> Set[str]:
    with open(path) as f:
        return {str(t) for t in yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["tokens"]}


def check_and_record_train_tokens(tokens: Sequence[str], output_dir: Optional[str], is_rank_zero: bool = True,
                                  test_tokens: Optional[Set[str]] = None) -> Dict[str, object]:
    """No final training token may be a navtest token (this holds with the refiner off too, unlike the refiner's packed
    row check).  Rank 0 writes the token list's sha256 and count to <output_dir>/train_tokens.json, which makes two runs'
    training sets comparable."""
    tokens = [str(t) for t in tokens]
    shared = set(tokens) & (navtest_tokens() if test_tokens is None else test_tokens)
    if shared:
        raise RuntimeError(f"{len(shared)} training tokens are navtest tokens, e.g. {sorted(shared)[:3]}")
    rec = {"train_token_sha256": hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest(), "count": len(tokens)}
    if is_rank_zero and output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "train_tokens.json.tmp"
        tmp.write_text(json.dumps(rec) + "\n")
        os.replace(tmp, out / "train_tokens.json")
    return rec


def dataset_tokens(data) -> Sequence[str]:
    """tokens of a Dataset (through its scene loader) or a CacheOnlyDataset."""
    return data._scene_loader.tokens if hasattr(data, "_scene_loader") else data.tokens


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    train_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if train_scene_filter.log_names is not None:
        # train_scene_filter.log_names = [l for l in train_scene_filter.log_names if l in cfg.train_logs]
        train_scene_filter.log_names = sorted(
            set(train_scene_filter.log_names) & set(cfg.train_logs)
        )
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if val_scene_filter.log_names is not None:
        # val_scene_filter.log_names = [l for l in val_scene_filter.log_names if l in cfg.val_logs]
        val_scene_filter.log_names = sorted(
            set(val_scene_filter.log_names) & set(cfg.val_logs)
        )
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    train_debug = cfg.train_debug if hasattr(cfg, "train_debug") else False

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        train_debug=train_debug,
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    use_fut_frames = agent.config.use_fut_frames if hasattr(agent.config, "use_fut_frames") else False
    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        use_fut_frames=use_fut_frames,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        use_fut_frames=use_fut_frames,
    )

    return train_data, val_data


# ----------------------------------------------------------------------------------------------- +train_data / +tf32
def _flag(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off", "", "none", "null"):
        return False
    raise ValueError(f"{name}={value!r}: not a boolean")


def train_data_name(cfg: DictConfig) -> str:
    """+train_data (Hydra; absent = navtrain, the main recipe).  An unknown name raises before anything is built."""
    name = cfg.get("train_data", None)
    name = TRAIN_DATA_DEFAULT if name is None else str(name)
    if name != TRAIN_DATA_DEFAULT and name not in TABLE_TRAIN_DATA:
        raise ValueError(f"+train_data={name}: not one of {(TRAIN_DATA_DEFAULT,) + TABLE_TRAIN_DATA}")
    return name


def apply_tf32(cfg: DictConfig) -> bool:
    """+tf32=true (the SimScale recipe on H100): TF32 tensor cores for float32 matmuls and convolutions.  Called before
    the model is built; every DDP rank runs main, so every rank sets it.  Absent / false: the torch defaults stay (the
    main recipe)."""
    if not _flag(cfg.get("tf32", False), "+tf32"):
        return False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    logger.info("TF32 on (+tf32): cuda.matmul.allow_tf32 %s, cudnn.allow_tf32 %s, float32 matmul precision %s",
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision())
    return True


def table_paths(name: str) -> Tuple[Path, Path]:
    """(token table, SimScale root) of a token-table split: constants.SPLITS[name]['tokens'] and its 'simscale_root'
    (= ssr_paths.SSR_SIMSCALE_ROOT).  A checkout that lacks them falls back, with a warning, to the same paths spelled
    out (${SSR_DATA_ROOT}/splits/<name>.parquet; $SSR_SIMSCALE_ROOT, else <SSR_DATA_ROOT>/../simscale)."""
    from navsim.agents.para_ssr.ck import constants as CKC
    from navsim.common import ssr_paths as SP

    split = CKC.SPLITS.get(name) or {}
    table = split.get("tokens")
    if not table:
        table = f"{SP.SSR_DATA_ROOT}/splits/{name}.parquet"
        logger.warning("constants.SPLITS has no %r tokens entry: token table %s", name, table)
    root = split.get("simscale_root") or getattr(SP, "SSR_SIMSCALE_ROOT", None)
    if not root:
        root = os.environ.get("SSR_SIMSCALE_ROOT") or os.path.normpath(os.path.join(SP.SSR_DATA_ROOT, "..", "simscale"))
        logger.warning("ssr_paths has no SSR_SIMSCALE_ROOT: SimScale root %s", root)
    return Path(table), Path(root)


def read_token_table(path):
    """A training token table (navsim/common/simscale.TABLE_COLUMNS), checked: every column, row = position, unique
    tokens, known sources, a scene pickle path and a round on every SimScale row.  -> DataFrame (token / log / source
    / pkl as str)."""
    import numpy as np
    import pandas as pd
    from navsim.common import simscale as SS

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"token table {path} does not exist (built by the setup, tools/release/setup)")
    df = pd.read_parquet(path)
    lack = [c for c in SS.TABLE_COLUMNS if c not in df.columns]
    if lack:
        raise ValueError(f"{path}: columns {lack} missing (a token table has {list(SS.TABLE_COLUMNS)})")
    df = df.assign(**{c: df[c].fillna("").astype(str) for c in ("token", "log", "source", "pkl")})
    if not (df["row"].to_numpy(np.int64) == np.arange(len(df))).all():
        raise ValueError(f"{path}: row column is not 0..N-1")
    dup = df["token"][df["token"].duplicated()]
    if len(dup):
        raise ValueError(f"{path}: {len(dup)} duplicate tokens, e.g. {dup.head(3).tolist()}")
    bad = sorted(set(df["source"]) - set(SS.SOURCE_REAL) - {SS.SOURCE_SIM})
    if bad:
        raise ValueError(f"{path}: unknown sources {bad}")
    sim = df[df["source"] == SS.SOURCE_SIM]
    if (sim["pkl"] == "").any() or not sim["round"].isin(SS.ROUNDS).all():
        raise ValueError(f"{path}: a SimScale row without a scene pickle or with a round outside {SS.ROUNDS}")
    return df


def check_table_logs(table, train_logs: Iterable[str], val_logs: Iterable[str], test_logs: Iterable[str]) -> None:
    """Real rows come from train_logs + val_logs; no SimScale row comes from a test log (its pickle stem is
    '<source log>-<scene>-<rrr>')."""
    from navsim.common import simscale as SS

    is_sim = table["source"] == SS.SOURCE_SIM
    real = table[~is_sim]
    off = real[~real["log"].isin(set(train_logs) | set(val_logs))]
    if len(off):
        raise RuntimeError(f"{len(off)} real rows of the token table are not in train_logs + val_logs, e.g. "
                           f"{off['log'].head(3).tolist()}")
    src = table.loc[is_sim, "pkl"].map(lambda p: Path(p).stem.rsplit("-", 2)[0])
    test = src[src.isin(set(test_logs))]
    if len(test):
        raise RuntimeError(f"{len(test)} SimScale rows of the token table come from test logs, e.g. "
                           f"{sorted(set(test))[:3]}")


def check_sim_pickles(sim, root: Path) -> None:
    """Every SimScale row's scene pickle exists (one directory listing per round, nothing is read): a missing one would
    stop a DataLoader worker, and the whole DDP run, mid-epoch."""
    names: Dict[str, Set[str]] = {}
    missing = []
    for p in sim["pkl"]:
        parent, name = os.path.split(p)
        if parent not in names:
            d = Path(root) / parent
            names[parent] = {e.name for e in os.scandir(d)} if d.is_dir() else set()
        if name not in names[parent]:
            missing.append(p)
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(sim)} SimScale scene pickles missing under {root}, e.g. "
                                f"{missing[:3]}")


def real_scene_filter(cfg: DictConfig, real) -> SceneFilter:
    """cfg.scene_filter (navtrain: 4 + 10 frames, has_route, its log and token lists) restricted to the real rows of a
    token table and their logs, within train_logs + val_logs (the pinned yamls are read, never changed)."""
    scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    logs = (set(cfg.train_logs) | set(cfg.val_logs)) & set(real["log"])
    if scene_filter.log_names is not None:
        logs &= set(scene_filter.log_names)
    scene_filter.log_names = sorted(logs)
    tokens = set(real["token"])
    if scene_filter.tokens is not None:
        tokens &= set(scene_filter.tokens)
    scene_filter.tokens = sorted(tokens)
    return scene_filter


def check_warmup_rows(agent: AbstractAgent, table) -> None:
    """With the refiner on, its warm-up packed rows must be the token table's rows (the contract of every row-indexed
    asset of a token-table split): ck_row, the warm-up arrays, LabelStore2 and the labeler's generations all count
    rows of that tokens.parquet."""
    ck = getattr(agent, "_ck_e2e2", None)
    if ck is None:
        return
    import pandas as pd

    packed = Path(ck.cfg.warmup["packed"]) / "tokens.parquet"
    toks = pd.read_parquet(packed, columns=["token"])["token"].astype(str).tolist()
    if toks != table["token"].tolist():
        raise RuntimeError(f"ck_e2e2.warmup.packed {packed} ({len(toks)} rows) is not the token table "
                           f"({len(table)} rows, same order): the warm-up assets belong to another split")


def check_score_rows(agent: AbstractAgent, table) -> None:
    """With the v2 head score target on (plan_anchor + plan_score_file), its token list must cover the table (at least
    TABLE_STRICT of it; a missing row trains with sim_reward_valid 0, no error)."""
    cfg = getattr(agent, "config", None)
    path = getattr(cfg, "plan_score_file", None)
    if not (getattr(cfg, "plan_anchor", False) and path):
        return
    with open(str(path) + ".tokens.json") as f:
        have = set(json.load(f))
    lost = table["token"][~table["token"].isin(have)]
    if len(lost) > (1.0 - TABLE_STRICT) * len(table):
        raise RuntimeError(f"plan_score_file {path}: {len(lost)} of {len(table)} training tokens have no score row, "
                           f"e.g. {lost.head(3).tolist()} (the v2 head score file of this split?)")
    if len(lost):
        logger.warning("plan_score_file %s: %d of %d training tokens have no score row", path, len(lost), len(table))


def build_table_datasets(cfg: DictConfig, agent: AbstractAgent, name: str, table_path: Optional[Path] = None,
                         sim_root: Optional[Path] = None):
    """+train_data=<name>: the training set is the split's token table, in its row order (= the row order of every
    row-indexed asset).  Real rows: the existing SceneLoader with cfg.scene_filter restricted to them
    (real_scene_filter); SimScale rows: navsim.common.simscale.SimScaleSceneLoader (4 + 8 frames, one scene pickle per
    call, none read here); both behind a dataloader.MultiSceneLoader.  A table row that no loader serves raises (a
    scene filter dropped it).  The validation set is not built when trainer.params.limit_val_batches is 0 (the
    recipe); otherwise it is the main recipe's (navtrain on val_logs), which this split also trains on.
    -> (train Dataset, val Dataset | None, the table)."""
    from navsim.common import simscale as SS
    from navsim.common.dataloader import MultiSceneLoader

    if table_path is None or sim_root is None:
        t, r = table_paths(name)
        table_path, sim_root = (t if table_path is None else table_path), (r if sim_root is None else sim_root)
    if cfg.get("train_debug", False):
        raise ValueError(f"+train_data={name}: train_debug reads a fixed scene dict, not the token table")
    table = read_token_table(table_path)
    check_table_logs(table, cfg.train_logs, cfg.val_logs, cfg.test_logs)
    is_sim = (table["source"] == SS.SOURCE_SIM).to_numpy()
    real, sim = table[~is_sim], table[is_sim]
    logger.info("+train_data=%s: %d rows %s from %s, SimScale root %s", name, len(table),
                table["source"].value_counts().to_dict(), table_path, sim_root)
    use_fut_frames = agent.config.use_fut_frames if hasattr(agent.config, "use_fut_frames") else False
    if use_fut_frames and len(sim):
        raise ValueError(f"+train_data={name}: SimScale scenes have no future camera frames (use_fut_frames)")
    # the cheap checks first: the real SceneLoader below reads every log pickle of the table
    check_sim_pickles(sim, Path(sim_root))
    check_warmup_rows(agent, table)
    check_score_rows(agent, table)

    sensor_config = agent.get_sensor_config()
    loaders = []
    if len(real):
        loaders.append(SceneLoader(sensor_blobs_path=Path(cfg.sensor_blobs_path), data_path=Path(cfg.navsim_log_path),
                                   scene_filter=real_scene_filter(cfg, real), sensor_config=sensor_config))
    if len(sim):
        loaders.append(SS.SimScaleSceneLoader(sim_root, sim, sensor_config=sensor_config))
    served: Set[str] = set()
    for loader in loaders:
        served.update(loader.tokens)
    lost = table[~table["token"].isin(served)]
    if len(lost):
        raise RuntimeError(f"+train_data={name}: {len(lost)} of {len(table)} table rows are served by no scene loader "
                           f"{lost['source'].value_counts().to_dict()}, e.g. {lost['token'].head(3).tolist()} (a real "
                           f"row outside the scene_filter's tokens, frame window or route rule?)")
    train_loader = MultiSceneLoader(table["token"].tolist(), loaders)

    train_data = Dataset(
        scene_loader=train_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        use_fut_frames=use_fut_frames,
    )
    val_data = None
    if float(cfg.trainer.params.limit_val_batches) != 0:
        logger.warning("+train_data=%s trains on navtrain_val too: the validation set (navtrain on val_logs) is not "
                       "held out", name)
        val_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
        if val_scene_filter.log_names is not None:
            val_scene_filter.log_names = sorted(set(val_scene_filter.log_names) & set(cfg.val_logs))
        else:
            val_scene_filter.log_names = cfg.val_logs
        val_data = Dataset(
            scene_loader=SceneLoader(sensor_blobs_path=Path(cfg.sensor_blobs_path), data_path=Path(cfg.navsim_log_path),
                                     scene_filter=val_scene_filter, sensor_config=sensor_config),
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            cache_path=cfg.cache_path,
            force_cache_computation=cfg.force_cache_computation,
            use_fut_frames=use_fut_frames,
        )
    return train_data, val_data, table


def check_table_tokens(tokens: Sequence[str], table) -> None:
    """A token-table training set is its table: the same tokens in the same order after every filter (the loaders'
    scene filters, then the teacher-cache filter).  Raises with what is missing per source."""
    tokens = [str(t) for t in tokens]
    want = table["token"].tolist()
    if tokens == want:
        return
    have = set(tokens)
    lost = table[~table["token"].isin(have)]
    extra = len(have - set(want))
    raise RuntimeError(f"training set != token table: {len(tokens)} tokens for {len(want)} rows; missing "
                       f"{lost['source'].value_counts().to_dict()} (e.g. {lost['token'].head(3).tolist()}), "
                       f"{extra} not in the table" + ("" if len(lost) or extra else ", order differs")
                       + " (a filter dropped tokens: teacher cache incomplete?)")


class TableInputGuard(pl.Callback):
    """Label coverage of a token-table training set (main adds it for +train_data only; the main recipe has no such
    callback).  The target builders give a token without a v2 head score row (plan_score_file) sim_reward_valid 0 and
    one without warm-up labels (ck_e2e2.warmup) ck2_wu_ok False, and the losses mask both: a score file or warm-up
    label set that misses part of the table would train that part without that supervision, with no error.  Rank 0
    counts its first TABLE_TOKENS_CHECK training tokens and stops the run when either share is below TABLE_STRICT, as
    ck_e2e2's _check_inputs does for the GT stores and the teacher BEVs (a missing file hits every rank alike).  A key
    no builder emits (plan_anchor off, refiner off) is not checked.  A resumed run checks anew."""

    KEYS = (("sim_reward_valid", "a v2 head score row (agent.config.plan_score_file)"),
            ("ck2_wu_ok", "warm-up labels (ck_e2e2.warmup packed / raw_labels / var_dir)"))

    def __init__(self, n_tokens: int = TABLE_TOKENS_CHECK, strict: float = TABLE_STRICT):
        super().__init__()
        self.n_tokens, self.strict = int(n_tokens), float(strict)
        self.tokens = 0
        self.seen = {k: 0 for k, _ in self.KEYS}
        self.ok = {k: 0 for k, _ in self.KEYS}
        self.checked = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if not self.checked and trainer.is_global_zero:
            self.add(batch[1])

    def add(self, targets: Mapping[str, torch.Tensor]) -> Optional[Dict[str, float]]:
        """Count one collated batch of targets; -> the shares once n_tokens are in (RuntimeError when one is short),
        None before."""
        n = 0
        for k, _ in self.KEYS:
            v = targets.get(k)
            if v is None:
                continue
            v = v.reshape(-1)
            n = max(n, int(v.numel()))
            self.seen[k] += int(v.numel())
            self.ok[k] += int((v.float() > 0.5).sum())
        if n == 0 and len(targets):
            n = len(next(iter(targets.values())))
        self.tokens += n
        if self.tokens < self.n_tokens:
            return None
        self.checked = True
        frac = {k: self.ok[k] / self.seen[k] for k, _ in self.KEYS if self.seen[k]}
        logger.info("token-table label coverage over rank 0's first %d training tokens: %s (none checked: %s)",
                    self.tokens, {k: round(v, 5) for k, v in frac.items()},
                    [k for k, _ in self.KEYS if k not in frac])
        short = [f"{k} on only {frac[k]:.4f} of the first {self.tokens} training tokens (< {self.strict}): {what} "
                 f"missing for part of the token table" for k, what in self.KEYS if k in frac and frac[k] < self.strict]
        if short:
            raise RuntimeError("; ".join(short))
        return frac


@hydra.main(version_base="1.2", config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    check_launch_route(os.environ)
    check_log_splits(cfg.train_logs, cfg.val_logs, cfg.test_logs)
    check_scene_filter_name(HydraConfig.get().runtime.choices.get("scene_filter"))
    data_name = train_data_name(cfg)
    seed = int(cfg.get("seed", 0))
    logger.info("Global Seed set to %d", seed)
    pl.seed_everything(seed, workers=True)

    logger.info(f"Path where all results are stored: {cfg.output_dir}")
    apply_tf32(cfg)

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    table = None          # the token table of a +train_data split (None: the main recipe)
    if cfg.use_cache_without_dataset:
        if data_name != TRAIN_DATA_DEFAULT:
            raise ValueError(f"+train_data={data_name} is built from its token table: use_cache_without_dataset is "
                             f"not supported")
        logger.info("Using cached data without building SceneLoader")
        assert cfg.force_cache_computation==False, "force_cache_computation must be False when using cached data without building SceneLoader"
        assert cfg.cache_path is not None, "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    elif data_name != TRAIN_DATA_DEFAULT:
        logger.info("Building the %s training set from its token table", data_name)
        train_data, val_data, table = build_table_datasets(cfg, agent, data_name)
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    if hasattr(agent, "filter_datasets_to_teacher_cache"):
        # aux_distill BEV distillation: keep only training tokens every teacher
        # cache has (ReSMap covers navtrain train_logs only).  No-op when it is off.
        agent.filter_datasets_to_teacher_cache(train_data, val_data)
    if table is not None:
        check_table_tokens(dataset_tokens(train_data), table)      # no token of the table may have been dropped
    check_and_record_train_tokens(dataset_tokens(train_data), cfg.output_dir,
                                  is_rank_zero=int(os.environ.get("LOCAL_RANK", "0")) == 0)

    logger.info("Building Datasets")
    if len(train_data) == 0:
        raise RuntimeError("training set is empty (after teacher-cache filtering, if the distillation is on)")
    train_dataloader = DataLoader(
        train_data, **cfg.dataloader.params, shuffle=True, worker_init_fn=_worker_init
    )
    logger.info("Num training samples: %d", len(train_data))
    if val_data is None:
        val_dataloader = None
        logger.info("No validation set built (trainer.params.limit_val_batches 0)")
    elif len(val_data) == 0:
        val_dataloader = None
        logger.warning("Validation set is empty (teacher-cache filtering); validation is skipped")
    else:
        val_dataloader = DataLoader(
            val_data, **cfg.dataloader.params, shuffle=False, worker_init_fn=_worker_init
        )
        logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer_params = cfg.trainer.params
    # trainer_params['strategy'] = "ddp_find_unused_parameters_true" #TODO
    # TensorBoard always; W&B additionally when cfg.wandb.enable is set.
    # W&B is fail-open telemetry -- see navsim/planning/training/wandb_logging.py.
    callbacks = list(agent.get_training_callbacks())
    if table is not None:
        callbacks.append(TableInputGuard())
    ckpt_cfg = cfg.get("checkpoint", None)
    if ckpt_cfg is not None:
        # Explicit, so every epoch is kept. Without this Lightning adds an
        # implicit ModelCheckpoint that retains only the newest epoch.
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=ckpt_cfg.get("every_n_epochs", 1),
                save_top_k=ckpt_cfg.get("save_top_k", -1),
                save_last=ckpt_cfg.get("save_last", True),
                save_on_train_epoch_end=ckpt_cfg.get(
                    "save_on_train_epoch_end", True
                ),
            )
        )

    trainer = pl.Trainer(
                **trainer_params,
                logger=build_loggers(cfg),
                callbacks=callbacks,
                )

    logger.info("Starting Training")
    resume_checkpoint = cfg.get("resume_checkpoint")
    if resume_checkpoint:
        resume_checkpoint = str(Path(resume_checkpoint).expanduser().resolve())
        if not Path(resume_checkpoint).is_file():
            raise FileNotFoundError(
                f"resume_checkpoint does not exist: {resume_checkpoint}"
            )
        logger.info("Resuming trainer state from %s", resume_checkpoint)
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_checkpoint,
    )

if __name__ == "__main__":
    main()
