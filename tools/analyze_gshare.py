"""Recover instantaneous gnorm/gshare from a training log and derive a scale.

Two artefacts of mmcv's LogBuffer have to be undone before these numbers mean
anything, and getting either wrong produces confident nonsense (negative
gradient shares, for one).

1. AVERAGING. ``TextLoggerHook`` writes ``mean(val_history[key][-interval:])``.
   ``gnorm``/``gshare`` are only written every ``grad_norm_log_interval``
   iterations, so their history is shorter than the window and what lands in
   the json is the running mean over every measurement so far this epoch. If
   ``A_k`` is the mean of the first k measurements then

       v_k = k * A_k - (k - 1) * A_{k-1}

2. REPEATS. The logger fires more often than the measurement does -- 100 vs 200
   in ``PARA_SSR_e2e.py`` -- so roughly every other row re-reports the previous
   running mean unchanged. Feeding those to the formula above as if they were
   new measurements makes k run ahead of reality and the reconstruction
   diverges. Rows whose values are byte-identical to the previous row are
   dropped here.

3. EPOCH BOUNDARIES. ``log_buffer.clear()`` runs at the start of every epoch,
   so the running mean restarts. k restarts with it.

The original probe this script was written for logged and measured on the same
interval (25) inside a single epoch, where none of the above bites -- which is
exactly why the bug was invisible. On a real run it is not.
"""
import argparse
import glob
import json
import os

# Discovered from the log, not hard-coded: which heads exist is a config
# choice (the 60-epoch run has no occupancy head), and a fixed list turns a
# disabled head into a KeyError. 'motion' is reported by the model but overlaps
# det's gradient path, so it is excluded from anything that has to sum to 1.
OVERLAPPING = ('motion',)


def discover_tasks(row):
    tasks = [k.split('/', 1)[1] for k in row if k.startswith('gshare/')]
    return [t for t in tasks if t not in OVERLAPPING]


def load_rows(path):
    if os.path.isdir(path):
        path = sorted(glob.glob(os.path.join(path, '*.log.json')))[-1]
    rows = []
    for line in open(path):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get('mode') == 'train' and 'gshare/plan' in d:
            rows.append(d)
    return path, rows


def dedupe(rows, keys):
    """Drop rows that merely re-report the previous running mean.

    Deduplication is decided once per row over all tracked keys together, not
    per key, so every series stays index-aligned.
    """
    out, prev = [], None
    n_dropped = 0
    for r in rows:
        sig = tuple(r[k] for k in keys if k in r)
        if prev is not None and sig == prev and r['epoch'] == out[-1]['epoch']:
            n_dropped += 1
            continue
        out.append(r)
        prev = sig
    return out, n_dropped


def instantaneous(rows, key):
    """Undo the running mean, restarting the counter at each epoch."""
    out, k, prev, epoch = [], 0, 0.0, None
    for r in rows:
        if r['epoch'] != epoch:
            epoch, k, prev = r['epoch'], 0, 0.0
        k += 1
        cur = r[key]
        out.append(k * cur - (k - 1) * prev)
        prev = cur
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', default='work_dirs/para_ssr_12ep')
    ap.add_argument('--epoch', type=int, default=None,
                    help='restrict to one epoch (default: the last one seen)')
    ap.add_argument('--all-epochs', action='store_true')
    args = ap.parse_args()

    path, rows = load_rows(args.path)
    if not rows:
        raise SystemExit(f'no gshare rows in {path}')

    TASKS = discover_tasks(rows[0])
    keys = [f'{p}/{t}' for t in TASKS for p in ('gnorm', 'gshare')
            if f'{p}/{t}' in rows[0]]
    rows, n_dropped = dedupe(rows, keys)
    print(f'source: {path}')
    print(f'{len(rows)} measurements after dropping {n_dropped} repeated rows, '
          f'epochs {rows[0]["epoch"]}..{rows[-1]["epoch"]}')

    if not args.all_epochs:
        # Calibrate on one epoch. Mixing epochs averages over a moving target:
        # the v1 planning share peaked at epoch 3 and then halved.
        want = args.epoch if args.epoch is not None else rows[-1]['epoch']
        sel = [r for r in rows if r['epoch'] == want]
        print(f'calibrating on epoch {want} ({len(sel)} measurements)')
    else:
        sel = rows

    inst = {t: instantaneous(sel, f'gshare/{t}') for t in TASKS}
    gn = {t: instantaneous(sel, f'gnorm/{t}') for t in TASKS}
    print(f'tasks present: {TASKS}')

    print('\n  ep   iter | ' + ' '.join(f'{t:>7s}' for t in TASKS) +
          '   (instantaneous %)')
    for i, r in enumerate(sel):
        print(f'  {r["epoch"]:3d} {r["iter"]:6d} | ' +
              ' '.join(f'{inst[t][i] * 100:7.2f}' for t in TASKS))

    bad = [(t, v) for t in TASKS for v in inst[t] if v < -1e-6 or v > 1.0 + 1e-6]
    if bad:
        print(f'\n!! {len(bad)} reconstructed shares fell outside [0, 1]. '
              'The averaging model does not fit this log -- check that '
              'grad_norm_log_interval is a multiple of log_config.interval '
              'and do not trust the calibration below.')

    n_tail = max(1, len(sel) // 2)
    print(f'\nmean over the last {n_tail} measurements (instantaneous):')
    tail = {t: sum(inst[t][-n_tail:]) / n_tail for t in TASKS}
    for t in TASKS:
        print(f'  {t:5s} share {tail[t] * 100:6.2f}%   '
              f'gnorm {sum(gn[t][-n_tail:]) / n_tail:.5f}')

    s = tail['plan']
    print(f'\nplanning BEV-gradient share = {s * 100:.2f}%')
    print('\nNOTE: gshare is measured AFTER aux_grad_scale (the autograd.grad '
          'call in _bev_grad_norms differentiates through _ScaleGrad). If the '
          'run already had a scale s0, divide the aux shares by s0 before '
          'solving, or the answer compounds.')
    print('\naux_grad_scale for a target planning share X, from an UNSCALED run:')
    print('   s = share * (1/X - 1) / (1 - share)')
    for X in (0.20, 0.30, 0.40, 0.50):
        print(f'   X = {int(X * 100):2d}%  ->  aux_grad_scale = '
              f'{s * (1 / X - 1) / (1 - s):.4f}')

    print('\nper-task share of the aux gradient (what the scale is applied to):')
    aux = [t for t in TASKS if t != 'plan']
    aux_tot = sum(tail[t] for t in aux)
    if aux_tot > 0:
        for t in aux:
            print(f'   {t:5s} {tail[t] / aux_tot * 100:5.1f}% of aux')

    print('\nhead-quality trend over the selected epoch '
          '(these ARE logged every iteration, so no reconstruction is needed):')
    for key, label in (('occ_sep/ratio', 'occ separation (1.0 = constant)'),
                       ('occ_iou0.5/mean', 'occ IoU@0.5'),
                       ('map_pts_err_m', 'map point error (m, euclidean)'),
                       ('map_spread_m', 'map query spread (m, 0 = collapsed)'),
                       ('map_cls_acc', 'map cls acc on matched queries'),
                       ('clip/rate', 'grad-clip firing rate'),
                       ('loss_plan_reg', 'planning L1')):
        vals = [r[key] for r in sel if key in r]
        if vals:
            print(f'  {label:38s} {vals[0]:8.4f} -> {vals[-1]:8.4f}')


if __name__ == '__main__':
    main()
