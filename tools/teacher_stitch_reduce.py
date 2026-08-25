"""Fit/evaluate BEVFusion->P and MapTRv2->P linear stitching maps."""
from __future__ import annotations

import argparse
import collections
import glob
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path('/home/yongjae/e2e/SSR')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
os.chdir(ROOT)

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

importlib.import_module('projects.mmdet3d_plugin')

from teacher_stitch_worker import (CANDIDATES, GROUP, P_CHECKPOINT, P_CONFIG,
                                   aligned_variants)


DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'
FAMILIES = ('identity', 'mean_only', 'diag', 'perm', 'full')
NPARAMS = {
    'identity': 0, 'mean_only': 256, 'diag': 512,
    'perm': 512, 'full': 65792,
}


def readout(head, bev, pos, command_index):
    batch = bev.shape[0]
    query = torch.cat((
        head.navi_se(bev, head.navi_embedding(command_index).unsqueeze(1)),
        pos), -1)
    learned, _ = head.tokenlearner(query)
    learned = learned.permute(1, 0, 2)
    latent_query, latent_pos = torch.split(learned, head.embed_dims, dim=2)
    latent_query = head.latent_decoder(
        query=latent_query, key=latent_query, value=latent_query,
        query_pos=latent_pos, key_pos=latent_pos)
    waypoint_pos, waypoint_query = torch.split(
        head.way_point.weight, head.embed_dims, dim=1)
    waypoint_pos = waypoint_pos.unsqueeze(0).expand(
        batch, -1, -1).permute(1, 0, 2)
    waypoint_query = waypoint_query.unsqueeze(0).expand(
        batch, -1, -1).permute(1, 0, 2)
    waypoint_query = head.way_decoder(
        query=waypoint_query, key=latent_query, value=latent_query,
        query_pos=waypoint_pos, key_pos=latent_pos)
    return head.ego_fut_decoder(waypoint_query).permute(1, 0, 2).reshape(
        batch, head.ego_fut_mode, head.fut_ts, 2)


def plan_loss(prediction, target, mask, command):
    repeated = target.unsqueeze(1).repeat(1, prediction.shape[1], 1, 1)
    weight = (command[..., None, None] * mask[:, None, :, None]).repeat(
        1, 1, 1, 2)
    return (torch.abs(prediction - repeated) * weight).sum() / weight.sum().clamp(min=1)


def add_nested(dst, src):
    if dst is None:
        return {key: (value.clone() if torch.is_tensor(value) else value)
                for key, value in src.items()}
    for key, value in src.items():
        dst[key] = dst[key] + value
    return dst


def centred(x_acc, xy_acc, y_acc):
    n = x_acc['n']
    if n != xy_acc['n'] or n != y_acc['n']:
        raise RuntimeError(f'accumulator count mismatch: {n}, '
                           f'{xy_acc["n"]}, {y_acc["n"]}')
    mx = x_acc['sx'] / n
    my = y_acc['sy'] / n
    sxx = x_acc['XtX'] - n * torch.outer(mx, mx)
    sxy = xy_acc['XtY'] - n * torch.outer(mx, my)
    syy = y_acc['YtY'] - n * torch.outer(my, my)
    return sxx, sxy, syy, mx, my, n


def residual_ss(x_acc, xy_acc, y_acc, weight, bias):
    value = torch.trace(
        y_acc['YtY'] - 2 * weight.T @ xy_acc['XtY']
        + weight.T @ x_acc['XtX'] @ weight)
    value += 2 * bias @ (weight.T @ x_acc['sx'])
    value -= 2 * bias @ y_acc['sy']
    value += x_acc['n'] * bias.pow(2).sum()
    return float(value)


def r2_score(x_acc, xy_acc, y_acc, weight, bias):
    residual = residual_ss(x_acc, xy_acc, y_acc, weight, bias)
    mean_y = y_acc['sy'] / y_acc['n']
    total = float(torch.trace(y_acc['YtY'])
                  - y_acc['n'] * mean_y.pow(2).sum())
    return 1.0 - residual / total


def build_p():
    cfg = Config.fromfile(str(P_CONFIG))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, str(P_CHECKPOINT), map_location='cpu', strict=False)
    model.to(DEV).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def valid_future_indices():
    cfg = Config.fromfile(str(P_CONFIG))
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    infos = build_dataset(cfg.data.test).data_infos
    scenes = [info['scene_token'] for info in infos]
    totals, seen = collections.Counter(scenes), collections.Counter()
    valid = set()
    for index, scene in enumerate(scenes):
        seen[scene] += 1
        if totals[scene] - seen[scene] >= 6:
            valid.add(index)
    return valid, len(infos)


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()
    directory = Path(args.dir)
    shard_files = sorted(directory.glob('shard_*.pt'))
    if not shard_files:
        raise FileNotFoundError(f'no shard files in {directory}')
    parts = [torch.load(path, map_location='cpu') for path in shard_files]
    if any(part['schema_version'] != 1 for part in parts):
        raise RuntimeError('unsupported worker schema')
    if len({part['nshard'] for part in parts}) != 1 or \
            parts[0]['nshard'] != len(parts):
        raise RuntimeError('incomplete shard set')

    sources = parts[0]['models']
    groups = {(source, candidate): parts[0]['groups'][f'{source}::{candidate}']
              for source in sources for candidate in CANDIDATES[source]}
    y_acc = {split: None for split in ('fit', 'test')}
    x_acc = {source: {split: {} for split in ('fit', 'test')}
             for source in sources}
    xy_acc = {source: {split: {} for split in ('fit', 'test')}
              for source in sources}
    all_indices = []
    for part in parts:
        all_indices.extend(part['indices'])
        for split in ('fit', 'test'):
            y_acc[split] = add_nested(y_acc[split], part['y_acc'][split])
            for source in sources:
                for group, value in part['x_acc'][source][split].items():
                    x_acc[source][split][group] = add_nested(
                        x_acc[source][split].get(group), value)
                for candidate, value in part['xy_acc'][source][split].items():
                    xy_acc[source][split][candidate] = add_nested(
                        xy_acc[source][split].get(candidate), value)
    if sorted(all_indices) != list(range(6019)):
        raise RuntimeError('shards do not cover every val index exactly once')

    fitted = {source: {} for source in sources}
    candidate_rows = []
    print('\nSPATIAL ALIGNMENT AUDIT (selection uses calibration R2 only)')
    print('=' * 112)
    print(f'{"source":11s} {"candidate":31s} {"fit R2":>9s} {"test R2":>9s} '
          f'{"cos":>8s} {"CKA":>8s} {"rms":>8s}')
    print('-' * 112)
    for source in sources:
        for candidate in CANDIDATES[source]:
            group = groups[(source, candidate)]
            fit_x = x_acc[source]['fit'][group]
            fit_xy = xy_acc[source]['fit'][candidate]
            sxx, sxy, _, mx, my, _ = centred(
                fit_x, fit_xy, y_acc['fit'])
            ridge = 1e-4 * torch.diag(sxx).mean()
            weight = torch.linalg.solve(
                sxx + ridge * torch.eye(256, dtype=sxx.dtype), sxy)
            bias = my - mx @ weight
            fit_r2 = r2_score(fit_x, fit_xy, y_acc['fit'], weight, bias)
            test_x = x_acc[source]['test'][group]
            test_xy = xy_acc[source]['test'][candidate]
            test_r2 = r2_score(test_x, test_xy, y_acc['test'], weight, bias)
            tsxx, tsxy, tsyy, *_ = centred(test_x, test_xy, y_acc['test'])
            cos = float(torch.trace(test_xy['XtY']) / (
                torch.trace(test_x['XtX']).sqrt()
                * torch.trace(y_acc['test']['YtY']).sqrt()))
            cka = float(tsxy.pow(2).sum() / (
                tsxx.pow(2).sum().sqrt() * tsyy.pow(2).sum().sqrt()))
            rms = float((torch.trace(test_x['XtX']) /
                         (test_x['n'] * 256)).sqrt())
            fitted[source][candidate] = {
                'W': weight, 'b': bias, 'mx': mx, 'my': my,
                'fit_r2': fit_r2, 'test_r2': test_r2,
                'cos': cos, 'cka': cka, 'rms': rms,
            }
            candidate_rows.append({
                'source': source, 'candidate': candidate,
                'fit_r2': fit_r2, 'test_r2': test_r2,
                'cos': cos, 'cka': cka, 'rms': rms,
            })
            print(f'{source:11s} {candidate:31s} {fit_r2:9.4f} '
                  f'{test_r2:9.4f} {cos:8.4f} {cka:8.4f} {rms:8.4f}')
        print('-' * 112)

    selected = {}
    family_maps = {source: {} for source in sources}
    for source in sources:
        plausible = [candidate for candidate in CANDIDATES[source]
                     if candidate != 'roll_control']
        selected[source] = max(
            plausible, key=lambda candidate: fitted[source][candidate]['fit_r2'])
        candidate = selected[source]
        group = groups[(source, candidate)]
        sxx, sxy, syy, mx, my, _ = centred(
            x_acc[source]['fit'][group], xy_acc[source]['fit'][candidate],
            y_acc['fit'])
        dx = torch.diag(sxx).clamp(min=1e-12)
        dy = torch.diag(syy).clamp(min=1e-12)
        diag_weight = torch.diag(torch.diag(sxy) / dx)
        corr = (sxy / (dx.sqrt()[:, None] * dy.sqrt()[None, :])).abs()
        row, col = linear_sum_assignment(-corr.numpy())
        permutation = torch.zeros(256, dtype=torch.long)
        permutation[torch.as_tensor(col)] = torch.as_tensor(row)
        perm_weight = torch.zeros(256, 256, dtype=sxx.dtype)
        perm_weight[permutation, torch.arange(256)] = (
            sxy[permutation, torch.arange(256)] / dx[permutation])
        family_maps[source] = {
            'identity': {'W': torch.eye(256, dtype=sxx.dtype),
                         'b': torch.zeros(256, dtype=sxx.dtype)},
            # Asymptotic label-free noise control: an input independent of P
            # has zero cross-covariance, so ridge converges to W=0 and emits
            # the calibration target mean everywhere.  This isolates how much
            # planner recovery comes from merely restoring P's feature bias.
            'mean_only': {'W': torch.zeros(256, 256, dtype=sxx.dtype),
                          'b': my},
            'diag': {'W': diag_weight, 'b': my - mx @ diag_weight},
            'perm': {'W': perm_weight, 'b': my - mx @ perm_weight},
            'full': {'W': fitted[source][candidate]['W'],
                     'b': fitted[source][candidate]['b']},
            'mean_matched_abs_corr': float(corr[row, col].mean()),
        }

    valid, dataset_size = valid_future_indices()
    model = build_p()
    head = model.pts_bbox_head
    position = head.positional_encoding(
        torch.zeros(1, head.bev_h, head.bev_w, device=DEV)
    ).flatten(2).permute(0, 2, 1).float()

    scores = collections.defaultdict(lambda: {'L1': [], 'D': []})
    cache_count = 0
    with torch.no_grad():
        for cache_file in sorted(directory.glob('cache_*.pt')):
            records = torch.load(cache_file, map_location='cpu')
            for record in records:
                if record['idx'] not in valid:
                    continue
                cache_count += 1
                target = record['gt'].to(DEV)
                mask = record['msk'].to(DEV)
                command = record['cmd'].to(DEV)
                command_index = command.argmax(-1)

                def evaluate(key, bev):
                    prediction = readout(head, bev, position, command_index)
                    scores[key]['L1'].append(float(
                        plan_loss(prediction, target, mask, command)))
                    displacement = torch.linalg.norm(
                        prediction[0, command_index[0]].cumsum(-2)
                        - target[0].cumsum(-2), dim=-1)
                    scores[key]['D'].append(
                        np.asarray([float(value) for value in displacement]))

                evaluate(('P', 'native'), record['P'].to(DEV).float())
                for source in sources:
                    raw = record[source].numpy()
                    candidate = selected[source]
                    aligned = aligned_variants(source, raw)[candidate]
                    source_bev = torch.from_numpy(
                        aligned.reshape(1, 10000, 256)).to(DEV).float()
                    flat = source_bev.reshape(-1, 256)
                    for family in FAMILIES:
                        if family == 'identity':
                            value = source_bev
                        else:
                            mapping = family_maps[source][family]
                            value = (flat @ mapping['W'].to(DEV).float()
                                     + mapping['b'].to(DEV).float()).reshape(
                                         1, 10000, 256)
                        evaluate((source, family), value)

                    # Fixed wrong spatial control, with its separately fitted W.
                    control = aligned_variants(source, raw)['roll_control']
                    control_flat = torch.from_numpy(
                        control.reshape(-1, 256)).to(DEV).float()
                    mapping = fitted[source]['roll_control']
                    control_aligned = (control_flat @ mapping['W'].to(DEV).float()
                                       + mapping['b'].to(DEV).float()).reshape(
                                           1, 10000, 256)
                    evaluate((source, 'roll_control_full'), control_aligned)
            del records

    def summarize(key):
        l1 = float(np.mean(scores[key]['L1']))
        displacement = np.mean(scores[key]['D'], axis=0)
        max_protocol = displacement[[1, 3, 5]]
        avg_protocol = np.asarray([
            displacement[:2].mean(), displacement[:4].mean(),
            displacement[:6].mean()])
        return {
            'L1': l1,
            'D': displacement,
            'uniad_max': max_protocol,
            'vad_avg': avg_protocol,
        }

    base = summarize(('P', 'native'))
    planning_rows = []
    print(f'\nSELECTED ALIGNMENTS (fit-only selection): {selected}')
    print(f'Frozen planner evaluation: {cache_count} held-out frames with full 3s future')
    print('=' * 124)
    print(f'{"source":11s} {"family":18s} {"params":>7s} {"L1":>9s} '
          f'{"dL":>9s} {"raw gap removed":>16s} {"R2":>8s} '
          f'{"L2@3 MAX":>11s} {"L2avg VAD":>11s}')
    print('-' * 124)
    print(f'{"P":11s} {"native":18s} {0:7d} {base["L1"]:9.4f} '
          f'{0.0:+9.4f} {"-":>16s} {"-":>8s} '
          f'{base["uniad_max"][2]:11.3f} {base["vad_avg"].mean():11.3f}')
    for source in sources:
        raw = summarize((source, 'identity'))
        raw_gap = raw['L1'] - base['L1']
        for family in FAMILIES:
            result = summarize((source, family))
            gap = result['L1'] - base['L1']
            removed = (raw_gap - gap) / raw_gap if abs(raw_gap) > 1e-12 else np.nan
            r2 = (fitted[source][selected[source]]['test_r2']
                  if family == 'full' else np.nan)
            print(f'{source:11s} {family:18s} {NPARAMS[family]:7d} '
                  f'{result["L1"]:9.4f} {gap:+9.4f} {removed * 100:15.1f}% '
                  f'{r2:8.4f} {result["uniad_max"][2]:11.3f} '
                  f'{result["vad_avg"].mean():11.3f}')
            planning_rows.append({
                'source': source, 'family': family,
                'params': NPARAMS[family], **result,
                'dL': gap, 'raw_gap_removed': removed,
                'test_r2': r2,
            })
        control = summarize((source, 'roll_control_full'))
        control_gap = control['L1'] - base['L1']
        control_removed = ((raw_gap - control_gap) / raw_gap
                           if abs(raw_gap) > 1e-12 else np.nan)
        control_r2 = fitted[source]['roll_control']['test_r2']
        print(f'{source:11s} {"roll_control_full":18s} {65792:7d} '
              f'{control["L1"]:9.4f} {control_gap:+9.4f} '
              f'{control_removed * 100:15.1f}% {control_r2:8.4f} '
              f'{control["uniad_max"][2]:11.3f} '
              f'{control["vad_avg"].mean():11.3f}')
        planning_rows.append({
            'source': source, 'family': 'roll_control_full',
            'params': 65792, **control, 'dL': control_gap,
            'raw_gap_removed': control_removed, 'test_r2': control_r2,
        })
        print('-' * 124)

    payload = {
        'schema_version': 1,
        'dataset_size': dataset_size,
        'calibration_scenes': 30,
        'heldout_scenes': 120,
        'heldout_planner_frames': cache_count,
        'selection_rule': 'max calibration R2, excluding roll_control',
        'selected': selected,
        'candidate_rows': candidate_rows,
        'base': base,
        'planning_rows': planning_rows,
        'mean_matched_abs_corr': {
            source: family_maps[source]['mean_matched_abs_corr']
            for source in sources
        },
    }
    with (directory / 'results.json').open('w', encoding='utf-8') as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False,
                  allow_nan=False)
        f.write('\n')
    torch.save({
        **payload,
        'maps': {source: {
            family: {name: value.cpu() for name, value in mapping.items()}
            for family, mapping in family_maps[source].items()
            if isinstance(mapping, dict)} for source in sources},
    }, directory / 'summary.pt')
    print(f'\nWrote {directory / "results.json"}')
    print(f'Wrote {directory / "summary.pt"}')


if __name__ == '__main__':
    main()
