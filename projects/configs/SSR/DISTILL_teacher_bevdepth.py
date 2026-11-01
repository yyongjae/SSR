"""Six-epoch BEVDepth adapter + SSR planning-head training."""
_base_ = ['./DISTILL_teacher_adapters.py']

# The parent config defines both teacher branches once.  Construct only this
# branch so a separate 2-GPU DDP job can train it independently.
model = dict(
    active_teachers=['bevdepth'],
    test_teacher='bevdepth')
