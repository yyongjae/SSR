"""Independent re-derivation of the PARA-SSR (interaction_final) training samples: run_training.build_datasets logic
(instantiate(cfg.scene_filter); log_names = sorted(set(log_names) & set(cfg.train_logs)); navsim filter_scenes on
navsim_logs/trainval) evaluated per log in parallel; compare with splits/e2e_train_trainlogs.parquet."""
import json, os, sys, time, dataclasses
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path("/home/external-user/yongjae/SSR")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(ROOT / "data/dataset"))
from omegaconf import OmegaConf
from hydra.utils import instantiate
from navsim.common.dataloader import filter_scenes

HERE = Path(__file__).parent
CFG = ROOT / "work_dirs/para_ssr_interaction_final/code/hydra/config.yaml"
LOGS = ROOT / "data/dataset/navsim_logs/trainval"
cfg = OmegaConf.load(CFG)
SF = instantiate(cfg.scene_filter)


def one(log):
    sf = dataclasses.replace(SF, log_names=[log])
    sc = filter_scenes(LOGS, sf)
    return [(tok, log, int(fr[SF.num_history_frames - 1]["frame_idx"]), fr[SF.num_history_frames - 1].get("map_location", ""))
            for tok, fr in sc.items()]


if __name__ == "__main__":
    t0 = time.time()
    split_yaml = OmegaConf.load(ROOT / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml")
    tl = list(cfg.train_logs)
    info = dict(n_filter_logs=len(SF.log_names), n_filter_tokens=len(SF.tokens) if SF.tokens is not None else None,
                n_cfg_train_logs=len(tl), train_logs_eq_yaml=sorted(tl) == sorted(split_yaml.train_logs),
                filter_tokens_is_none=SF.tokens is None, max_scenes=SF.max_scenes,
                num_frames=SF.num_frames, frame_interval=SF.frame_interval, has_route=SF.has_route)
    logs = sorted(set(SF.log_names) & set(tl))
    info["n_logs_after_intersection"] = len(logs)
    # filter_scenes iterates LOGS.iterdir(): logs without a pickle are silently dropped -> record
    info["logs_missing_pkl"] = [l for l in logs if not (LOGS / f"{l}.pkl").exists()]
    rows = []
    with Pool(4) as p:
        for r in p.imap_unordered(one, logs, chunksize=4):
            rows += r
    ref = pd.DataFrame(rows, columns=["token", "log", "frame_idx", "city"])
    ref.to_parquet(HERE / "tokens_rederived.parquet", index=False)
    e = pd.read_parquet("/home/external-user/ssd/yongjae_refiner/splits/e2e_train_trainlogs.parquet")
    m = e.merge(ref, on="token", how="outer", suffixes=("", "_ref"), indicator=True)
    both = m[m._merge == "both"]
    info.update(n_rederived=len(ref), n_rederived_unique=int(ref.token.nunique()), n_list=len(e),
                only_in_list=int((m._merge == "left_only").sum()), only_in_rederived=int((m._merge == "right_only").sum()),
                log_mismatch=int((both.log != both.log_ref).sum()), frame_idx_mismatch=int((both.frame_idx != both.frame_idx_ref).sum()),
                city_mismatch=int((both.city != both.city_ref).sum()), per_city_ref=ref.city.value_counts().to_dict(),
                n_logs_ref=int(ref.log.nunique()), n_logs_list=int(e.log.nunique()), wall_s=time.time() - t0)
    json.dump(info, open(HERE / "tokens_summary.json", "w"), indent=1, default=str)
    print(json.dumps(info, indent=1, default=str))
