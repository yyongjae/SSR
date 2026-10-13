"""Negative control: shuffle teacher memory across the batch.

If RPED still helps, the student is matching a prior rather than sample
evidence.  This should collapse 1s collision recovery.
"""
_base_ = ['./RPED_SSR_student.py']

model = dict(
    distill=dict(shuffle_memory=True))
