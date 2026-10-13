"""Six-epoch BEVFusion adapter + SSR planning-head training."""
_base_ = ['./DISTILL_teacher_bevfusion_maptrv2.py']

model = dict(
    active_teachers=['bevfusion'],
    test_teacher='bevfusion')
