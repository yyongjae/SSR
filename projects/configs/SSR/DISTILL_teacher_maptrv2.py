"""Six-epoch MapTRv2 adapter + SSR planning-head training."""
_base_ = ['./DISTILL_teacher_bevfusion_maptrv2.py']

model = dict(
    active_teachers=['maptrv2'],
    test_teacher='maptrv2')
