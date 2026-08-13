"""Sweep W3 -- aux heads at full strength, weak pull on the BEV.

Down-weighting an aux loss does two things at once: it reduces how much the task
reshapes the shared BEV feature, and it slows the aux head itself. For
distillation experiments only the first is wanted -- a weak teacher is a useless
teacher. `aux_grad_scale` decouples them: the aux losses keep weight 1.0, so the
heads converge normally, but the gradient they push back into `bev_embed` is
scaled by 0.02 (chosen to match W2's effective pressure on the trunk).

Compare against W2 to separate "aux supervision hurts the BEV" from
"aux supervision is simply too strong".
"""
_base_ = ['./PARA_SSR_e2e.py']

model = dict(
    task_loss_weight=dict(plan=1.0, det=1.0, map=1.0, occ=1.0),
    aux_grad_scale=0.02)
