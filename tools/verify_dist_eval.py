"""Run the REAL distributed validation path on a handful of samples.

``verify_multibatch_and_metrics.py`` greps source strings and calls metric
functions directly. That is not the same as proving epoch 10 of a 60-epoch run
will not die, because the thing that kills it lives in the plumbing between
those pieces: ``custom_multi_gpu_test`` returns ``{'bbox_results': [...]}``
while ``evaluate()`` wants a list, the sampler hands each rank a contiguous
block, ``collect_results_cpu`` has to stitch them back in that same order, and
every aux evaluator then has to survive whatever the result dicts contain.

So this runs all of it: the real sampler, the real multi-GPU test loop, the
real collection, and the real ``dataset.evaluate()`` -- just over N samples
instead of 6019, with randomly initialised weights. The metric VALUES are
meaningless; what is being tested is that the path completes and returns the
keys the EvalHook will log.

    tools/verify_dist_eval.sh [N_SAMPLES] [CONFIG]

Reads N_SAMPLES from the environment (default 8).
"""
import argparse
import os
import os.path as osp
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import importlib

import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist

importlib.import_module('projects.mmdet3d_plugin')

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.SSR.apis.test import custom_multi_gpu_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--n-samples', type=int,
                    default=int(os.environ.get('N_SAMPLES', 8)))
    ap.add_argument('--launcher', default='pytorch')
    ap.add_argument('--local_rank', type=int, default=0)
    args = ap.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    cfg = Config.fromfile(args.config)
    if cfg.get('plugin'):
        pass  # already imported above
    init_dist(args.launcher, **cfg.get('dist_params', {}))
    rank, world_size = get_dist_info()

    cfg.model.train_cfg = None
    dataset = build_dataset(cfg.data.test)

    # Truncate BEFORE the dataloader is built so the sampler splits the short
    # list -- that is the split whose boundary we want to exercise.
    n = args.n_samples
    assert n % world_size == 0, 'keep n_samples a multiple of the rank count'
    dataset.data_infos = dataset.data_infos[:n]
    if hasattr(dataset, 'flag'):
        dataset.flag = dataset.flag[:n]

    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=1, dist=True,
        shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # init_weights matters even for a plumbing test: without it the deformable
    # attention sampling offsets keep their raw constructor values, the decoder
    # diverges, and the planning L2 comes back NaN -- which looks like a metric
    # bug and is not one.
    model.init_weights()
    model.CLASSES = dataset.CLASSES
    model = MMDistributedDataParallel(
        model.cuda(), device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)

    if rank == 0:
        print(f'== {n} samples over {world_size} ranks, random weights ==')
    out = custom_multi_gpu_test(model, loader, gpu_collect=False)

    # 1. the shape the EvalHook actually receives
    assert isinstance(out, dict) and 'bbox_results' in out, \
        f'custom_multi_gpu_test returned {type(out)}, not a dict'
    dist.barrier()
    if rank != 0:
        return

    res = out['bbox_results']
    assert len(res) == n, f'collected {len(res)} results for {n} samples'

    # 2. the contiguous sampler and the collection must agree on ORDER. If
    #    collect_results_cpu ever goes back to zip(*part_list) this catches it.
    got = [r['pts_bbox']['sample_token'] if 'sample_token' in r.get('pts_bbox', {})
           else None for r in res]
    if got[0] is not None:
        want = [info['token'] for info in dataset.data_infos]
        assert got == want, 'result order does not match dataset order'
        print('   result order matches dataset order')
    else:
        print('   (results carry no sample_token; order not checked here)')

    # 3. the evaluator itself, on the dict the hook passes through
    metrics = dataset.evaluate(out, jsonfile_prefix=osp.join(
        os.environ.get('TMPDIR', '/tmp'), 'verify_dist_eval'))
    print('\n== keys the EvalHook would log ==')
    for k in sorted(metrics):
        print(f'   {k:44s} {metrics[k]}')
    families = {k.split('_')[0].split('/')[0] for k in metrics}
    print(f'\nmetric families present: {sorted(families)}')
    print('\nDIST EVAL PATH OK')


if __name__ == '__main__':
    main()
