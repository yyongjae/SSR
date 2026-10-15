"""Stage 2: dump a frozen PARA-SSR student's BEV in the teacher cache format.

Writes ``bev/<shard>.npy`` of ``(256, 50, 100)`` float16 in the student layout
(forward rows, right columns), plus ``index.json`` / ``meta.json``, so
``BevCache`` and every readout script read it exactly like the teacher cache.
The checkpoint must be the architecture the teacher aligns with: 50 x 100 BEV,
front ROI, use_stl=false (report/19 s6).

One process per GPU; ranks take disjoint token slices and write only their
own shards, then ``--merge`` builds the index:

    for r in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$r python tools/readout/cache_student_bev.py \
        --ckpt <plan_only.ckpt> --filter navtrain --split trainval \
        --out /data3/kyungmin/student_bev/navtrain --rank $r --world 4 \
        agent.config.use_task_interaction=false agent.config.use_det_motion_head=false \
        agent.config.use_map_head=false agent.config.grad_balance_target=null &
    done; wait
    python tools/readout/cache_student_bev.py --merge --out /data3/kyungmin/student_bev/navtrain

Trailing ``key=value`` arguments are Hydra overrides of the agent config, the
same ones the checkpoint was trained/evaluated with.  ``--resume`` skips tokens
already flushed by this rank.  Frames whose sensor files are missing on this
host are skipped and listed in ``missing_r<rank>.txt``; eval_readout_pdms.py
refuses a PDMS over an incomplete navtest token set unless told otherwise.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import _env  # noqa: F401
from _env import REPO, scene_filter, split_dirs

AGENT_YAML = REPO / "navsim/planning/script/config/common/agent/para_ssr_agent.yaml"


def build_agent(ckpt, overrides):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(AGENT_YAML)
    cfg.merge_with(OmegaConf.from_dotlist([o.split("agent.", 1)[-1] for o in overrides]))
    cfg.checkpoint_path = ckpt
    if not ckpt:  # smoke test only: random weights, no download
        cfg.config.backbone_pretrained = False
    agent = instantiate(cfg)
    if ckpt:
        agent.initialize()
    return agent.eval()


class FeatureData(Dataset):
    def __init__(self, tokens, loader, builders):
        self.tokens, self.loader, self.builders = tokens, loader, builders

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        try:
            agent_input = self.loader.get_agent_input_from_token(self.tokens[i])
            feats = {}
            for b in self.builders:
                feats.update(b.compute_features(agent_input))
        except FileNotFoundError as exc:
            # a sensor blob absent on this host: record it, do not kill the run
            return self.tokens[i], None, str(exc)
        return self.tokens[i], feats, None


def collate(batch):
    ok = [b for b in batch if b[1] is not None]
    failed = [(b[0], b[2]) for b in batch if b[1] is None]
    if not ok:
        return [], None, failed
    keys = ok[0][1].keys()
    return [b[0] for b in ok], {k: torch.stack([b[1][k] for b in ok]) for k in keys}, failed


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def merge(out: Path):
    index = {}
    for p in sorted(out.glob("index_r*.json")):
        index.update(json.loads(p.read_text()))
    metas = [json.loads(p.read_text()) for p in sorted(out.glob("meta_r*.json"))]
    if not metas:
        raise SystemExit(f"no rank output under {out}")
    if len({m["checkpoint_sha256"] for m in metas}) != 1:
        raise SystemExit("ranks used different checkpoints")
    meta = dict(metas[0], num_frames=len(index))
    meta.pop("rank", None)
    (out / "index.json").write_text(json.dumps(index))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"merged {len(index)} frames -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="", help="Lightning checkpoint (empty = random weights, smoke only)")
    ap.add_argument("--filter", default="navtrain")
    ap.add_argument("--split", default="trainval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--shard", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-logs", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args, overrides = ap.parse_known_args()
    out = Path(args.out)
    if args.merge:
        return merge(out)

    from navsim.common.dataloader import SceneLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = build_agent(args.ckpt, overrides).to(device)
    cfg = agent.config
    sf = scene_filter(args.filter)
    if args.max_logs:
        sf.log_names = sf.log_names[: args.max_logs]
    logs_dir, blobs_dir = split_dirs(args.split)
    loader = SceneLoader(logs_dir, blobs_dir, sf, sensor_config=agent.get_sensor_config())
    tokens = sorted(loader.tokens)[args.rank :: args.world]
    if args.limit:
        tokens = tokens[: args.limit]

    (out / "bev").mkdir(parents=True, exist_ok=True)
    index_path = out / f"index_r{args.rank}.json"
    index = json.loads(index_path.read_text()) if args.resume and index_path.exists() else {}
    shard_id = 1 + max([int(v[0].split("_s")[1]) for v in index.values()], default=-1)
    todo = [t for t in tokens if t not in index]
    print(f"[rank {args.rank}] {len(tokens)} tokens, {len(todo)} to do", flush=True)
    (out / f"meta_r{args.rank}.json").write_text(json.dumps({
        "layout": "forward_right",
        "source": "para_ssr",
        "checkpoint": args.ckpt,
        "checkpoint_sha256": sha256(args.ckpt) if args.ckpt else "random-init",
        "overrides": overrides,
        "filter": args.filter,
        "split": args.split,
        "bev_grid": [cfg.bev_h, cfg.bev_w],
        "pc_range": list(cfg.pc_range),
        "tensors": {"bev": {"shape": [cfg.embed_dims, cfg.bev_h, cfg.bev_w], "dtype": "float16",
                            "note": "PARA-SSR bev_embed; axes (C, forward, right)"}},
        "rank": args.rank,
    }, indent=2))

    dl = DataLoader(FeatureData(todo, loader, agent.get_feature_builders()), batch_size=args.batch_size,
                    num_workers=args.workers, collate_fn=collate)
    buf, names = [], []

    def flush():
        nonlocal shard_id, buf, names
        if not names:
            return
        name = f"r{args.rank}_s{shard_id:04d}"
        np.save(out / "bev" / f"{name}.npy", np.stack(buf))
        index.update({t: [name, i] for i, t in enumerate(names)})
        tmp = index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index))
        tmp.replace(index_path)  # the index only ever names complete shards
        shard_id += 1
        buf, names = [], []

    t0, n = time.time(), 0
    missing_path = out / f"missing_r{args.rank}.txt"
    with torch.no_grad(), open(missing_path, "a") as missing:
        for toks, feats, failed in dl:
            for tok, err in failed:
                missing.write(f"{tok}\t{err}\n")
            if feats is None:
                continue
            feats = {k: v.to(device) for k, v in feats.items()}
            bev = agent.para_ssr_model(feats, run_aux=False)["bev_embed"]
            b = bev.shape[0]
            bev = bev.view(b, cfg.bev_h, cfg.bev_w, cfg.embed_dims).permute(0, 3, 1, 2)
            buf.extend(bev.to(torch.float16).cpu().numpy())
            names.extend(toks)
            n += b
            if len(names) >= args.shard:
                flush()
                print(f"[rank {args.rank}] {n}/{len(todo)}  {(time.time() - t0) / n:.3f} s/frame", flush=True)
    flush()
    skipped = sum(1 for _ in open(missing_path))
    print(f"[rank {args.rank}] done, {len(index)} frames indexed, {skipped} skipped "
          f"for missing sensor files (see {missing_path.name})")


if __name__ == "__main__":
    main()
