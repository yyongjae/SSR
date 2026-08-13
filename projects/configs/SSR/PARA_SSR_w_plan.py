"""Sweep W2 -- planning-heavy (loss-value balanced).

Measured mean loss magnitudes on real batches were plan ~0.3, det ~26,
map ~5.6, occ ~4.9. The weights below bring every task to roughly the same
magnitude, which makes planning ~100x more influential on the shared BEV
feature than W1 while still training all three aux heads.

Rule of thumb behind the numbers: w_task ~ L_plan / L_task.
"""
_base_ = ['./PARA_SSR_e2e.py']

model = dict(task_loss_weight=dict(plan=1.0, det=0.01, map=0.05, occ=0.06))
