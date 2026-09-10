"""RPED ablation: same-question readout R(Q_s, T) instead of privileged Q."""
_base_ = ['./RPED_SSR_student.py']

model = dict(
    distill=dict(query_source='student'))
