# Reference copy of maptracker/tools/cache_teacher_kd.py (the generator of the
# ReSMap KD cache).  It does not run from here: it needs the maptracker repo
# (plugin/ on sys.path) and the resmap env (torch 1.12 / mmcv-full 1.6 /
# mmdet3d 1.0.0rc6).  Copy it into maptracker/tools/ to regenerate the cache or
# to add a split; the navtest command is in the docstring below.
# See report/19_planning_readout.md s6 and s8.
"""Dump the trained ReSMap teacher's per-frame outputs for distillation into PARA-SSR.

Three tensors per training frame, in fp16:

    bev      (256, 100, 50)   the neck output every head reads.  256 channels is
                              PARA-SSR's embed_dims and 100x50 at 0.64 m is the
                              front half of its BEV cell-for-cell, so feature
                              distillation needs neither a projection nor a
                              resample -- just a crop.
    seg      (4, 200, 100)    raw class logits, not the thresholded mask
    vectors  (100, 20, 2) + scores/labels/props   the vector head, whose query and
                              point counts already match PARA-SSR's map head

Why the layout is what it is:

*   **Sharded `.npy`, not one file and not one file per frame.**  A single 322 GB
    array cannot be uploaded anywhere and 126k small files punish both the
    filesystem and the reader.  ~2000-frame shards give true `mmap_mode='r'`
    random access -- the student's dataloader touches one frame's pages, not the
    shard -- while staying inside the per-file size hosts accept.
*   **Shards close on a scene boundary.**  The teacher is temporal: its memory
    bank resets at `local_idx == 0` and accumulates through a scene.  Cutting a
    shard mid-scene would make `--resume` restart inside a scene with an empty
    memory bank, silently changing the features.  Closing on the boundary makes
    the resume point exactly a memory reset.
*   **Each rank writes only its own shards.**  The test sampler hands whole
    scenes to a rank (`groups[rank::world_size]`), so ranks never share a scene
    and never share a file; no locking, no coordination.

Run:
    torchrun --nproc_per_node=4 tools/cache_teacher_kd.py \
        --cfg work_dirs/resmap_nav_rideflux_stage3/resmap_nav_stage3.py \
        --ckpt work_dirs/resmap_nav_rideflux_stage3/iter_63024.pth \
        --out /data3/kyungmin/kd_teacher_resmap

navtest (PDMS evaluation of planning readouts; ~31 GB):
    torchrun --nproc_per_node=4 tools/cache_teacher_kd.py --cfg ... --ckpt ... \
        --split none --only-split-tokens \
        --ann-file /data2/kyungmin/navsim/infos/navsim_map_infos_navtest.pkl \
        --out /data3/kyungmin/kd_teacher_resmap_navtest
"""
import argparse
import glob
import hashlib
import json
import os
import os.path as osp
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
import plugin  # noqa: F401  registers the custom modules
from plugin.datasets.builder import build_dataloader

TENSORS = ('bev', 'seg', 'vectors', 'scores', 'labels', 'props')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', default='train',
                   help="log_split to cache: train | val | none (every log in --ann-file)")
    p.add_argument('--ann-file', default=None,
                   help='override data.test.ann_file, e.g. '
                        '/data2/kyungmin/navsim/infos/navsim_map_infos_navtest.pkl')
    p.add_argument('--only-split-tokens', action='store_true',
                   help='run the teacher on every frame (its memory bank needs the '
                        'sequence) but save only NAVSIM allow-list tokens '
                        '(in_split_tokens); navtest: 12,146 of 71,460 frames')
    p.add_argument('--shard', type=int, default=2000,
                   help='frames per shard (closed on the next scene boundary)')
    p.add_argument('--limit', type=int, default=0,
                   help='stop after N frames per rank; for a dry run')
    p.add_argument('--resume', action='store_true',
                   help='skip frames already covered by this rank\'s shards')
    return p.parse_args()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


class ShardWriter:
    """Buffers frames and flushes one `.npy` per tensor per shard."""

    def __init__(self, out, rank, shard_frames):
        self.out, self.rank, self.shard_frames = out, rank, shard_frames
        self.buf = {k: [] for k in TENSORS}
        self.tokens = []
        self.index = {}          # token -> [shard_name, row]
        self.shard_id = 0
        self.written = 0
        for k in TENSORS:
            os.makedirs(osp.join(out, k), exist_ok=True)

    @property
    def pending(self):
        return len(self.tokens)

    def add(self, token, tensors):
        self.tokens.append(token)
        for k in TENSORS:
            self.buf[k].append(tensors[k])

    def flush(self):
        if not self.tokens:
            return
        name = f'r{self.rank}_s{self.shard_id:04d}'
        for k in TENSORS:
            arr = np.stack(self.buf[k])
            np.save(osp.join(self.out, k, name + '.npy'), arr)
            self.buf[k] = []
        for row, tok in enumerate(self.tokens):
            self.index[tok] = [name, row]
        self.written += len(self.tokens)
        self.tokens = []
        self.shard_id += 1

    def resume_from(self):
        """Frames already on disk for this rank, and where to continue."""
        shards = sorted(glob.glob(osp.join(self.out, 'bev', f'r{self.rank}_s*.npy')))
        done = 0
        for s in shards:
            name = osp.basename(s)[:-4]
            # every tensor must be present, else the shard is a partial write
            if not all(osp.exists(osp.join(self.out, k, name + '.npy')) for k in TENSORS):
                for k in TENSORS:
                    f = osp.join(self.out, k, name + '.npy')
                    if osp.exists(f):
                        os.remove(f)
                continue
            arr = np.load(s, mmap_mode='r')
            for row in range(arr.shape[0]):
                pass
            done += arr.shape[0]
            self.shard_id = max(self.shard_id, int(name.split('_s')[1]) + 1)
        self.written = done
        return done


def main():
    args = parse_args()
    init_dist('pytorch', timeout=timedelta(hours=6))
    rank, world = get_dist_info()

    cfg = Config.fromfile(args.cfg)
    cfg.data.test.log_split = None if args.split.lower() == 'none' else args.split
    if args.ann_file:
        cfg.data.test.ann_file = args.ann_file
    if args.only_split_tokens and args.resume:
        # resume counts saved frames as loader positions, which no longer holds
        raise SystemExit('--resume cannot be combined with --only-split-tokens')
    cfg.data.test.work_dir = cfg.work_dir
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True, shuffle=False,
        shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.ckpt, map_location='cpu')
    model = MMDistributedDataParallel(
        model.cuda(), device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False).eval()
    core = model.module

    # the seg decoder is called once per frame, in one of two branches; the hook
    # catches whichever ran and keeps the logits before they are thresholded
    seg_out = {}
    core.seg_decoder.register_forward_hook(
        lambda m, i, o: seg_out.__setitem__(
            'x', o[0] if isinstance(o, (tuple, list)) else o))

    sample_idxs = loader.sampler.sample_idxs   # loader position -> dataset index

    writer = ShardWriter(args.out, rank, args.shard)
    skip = writer.resume_from() if args.resume else 0
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        print(f'[cache] {len(dataset)} frames, split={args.split}, '
              f'{world} ranks, shard={args.shard}', flush=True)
    if skip:
        print(f'[rank {rank}] resuming: {skip} frames already cached', flush=True)

    t0 = time.time()
    t_mark, n_mark = None, 0
    n = 0
    reported = False
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i < skip:
                continue
            out = model(return_loss=False, rescale=True, **data)
            res = out[0] if isinstance(out, list) else out

            bev = core._last_bev_feats            # (1, 256, 100, 50)
            seg = seg_out['x']                    # (1, 4, 200, 100)
            tensors = {
                'bev': bev[0].detach().to(torch.float16).cpu().numpy(),
                'seg': seg[0].detach().to(torch.float16).cpu().numpy(),
                'vectors': np.asarray(res['vectors'], dtype=np.float16),
                'scores': np.asarray(res['scores'], dtype=np.float16),
                'labels': np.asarray(res['labels'], dtype=np.int8),
                'props': np.asarray(res['props'], dtype=np.int16),
            }
            if not reported and rank == 0:
                print('[cache] per-frame tensors', flush=True)
                for k, v in tensors.items():
                    print(f'    {k:8s} {tuple(v.shape)} {v.dtype}', flush=True)
                vec = tensors['vectors']
                print(f'    vectors range [{vec.min():.3f}, {vec.max():.3f}]',
                      flush=True)
                reported = True

            sample = dataset.samples[sample_idxs[i]]
            if not args.only_split_tokens or sample.get('in_split_tokens', True):
                writer.add(res['token'], tensors)
            if t_mark is None:          # start the clock after the first frame,
                t_mark = time.time()    # which pays for worker spawn
                n_mark = n

            # Close the shard only when the scene ends, so a resume lands on a
            # memory-bank reset rather than inside a sequence.  The flag comes
            # from the dataset rather than the batch: `seq_info` arrives wrapped
            # in a DataContainer whose nesting depends on the collate path,
            # while `samples[...]['next'] == -1` is unambiguous.
            scene_end = sample['next'] == -1
            if writer.pending >= args.shard and scene_end:
                writer.flush()
            n += 1
            if rank == 0 and n % 100 == 0:
                el = time.time() - t_mark
                rate = (n - n_mark) / max(el, 1e-6)
                left = (len(sample_idxs) - n) / max(rate, 1e-6)
                print(f'[cache] rank0 {n}/{len(sample_idxs)}  '
                      f'{1/rate:.3f} s/frame/rank  '
                      f'eta {timedelta(seconds=int(left))}', flush=True)
            if args.limit and n >= args.limit:
                break

    writer.flush()
    with open(osp.join(args.out, f'index_r{rank}.json'), 'w') as f:
        json.dump(writer.index, f)
    print(f'[rank {rank}] done: {writer.written} frames, '
          f'{writer.shard_id} shards, {time.time()-t0:.0f}s', flush=True)

    dist.barrier()
    if rank == 0:
        index = {}
        for f in sorted(glob.glob(osp.join(args.out, 'index_r*.json'))):
            index.update(json.load(open(f)))
        with open(osp.join(args.out, 'index.json'), 'w') as f:
            json.dump(index, f)
        meta = {
            'teacher_config': osp.abspath(args.cfg),
            'teacher_checkpoint': osp.abspath(args.ckpt),
            'checkpoint_sha256': sha256_file(args.ckpt),
            'split': args.split,
            'ann_file': cfg.data.test.ann_file,
            'only_split_tokens': bool(args.only_split_tokens),
            'num_frames': len(index),
            'classes': list(cfg.data.train.cat2id.keys()),
            'roi_size_m': list(cfg.roi_size),
            'roi_center_m': list(cfg.model.roi_center),
            'pc_range': list(cfg.pc_range),
            'bev_grid': [cfg.bev_h, cfg.bev_w],
            'canvas_size': list(cfg.canvas_size),
            'tensors': {
                'bev': {'shape': [256, cfg.bev_h, cfg.bev_w], 'dtype': 'float16',
                        'note': 'neck output; axes (C, lateral, forward)'},
                'seg': {'shape': [len(cfg.data.train.cat2id), 200, 100],
                        'dtype': 'float16', 'note': 'raw logits, not thresholded'},
                'vectors': {
                    'shape': [100, 20, 2], 'dtype': 'float16',
                    'note': 'normalised to [0, 1] over the ROI, NOT metres'},
                'scores': {'shape': [100], 'dtype': 'float16',
                           'note': 'sigmoid confidence per query'},
                'labels': {'shape': [100], 'dtype': 'int8'},
                'props': {'shape': [100], 'dtype': 'int16'},
            },
        }
        with open(osp.join(args.out, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'[cache] index.json: {len(index)} frames; meta.json written',
              flush=True)


if __name__ == '__main__':
    main()
