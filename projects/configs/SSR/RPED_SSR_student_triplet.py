"""RPED ablation: add a command triplet so evidence stays command-equivariant."""
_base_ = ['./RPED_SSR_student.py']

model = dict(
    distill=dict(
        command_triplet_weight=0.1,
        triplet_margin=0.1))
