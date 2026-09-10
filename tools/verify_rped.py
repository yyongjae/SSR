"""CPU regression checks for Rank-N Privileged Evidence Distillation."""
import importlib.util
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'projects/mmdet3d_plugin/SSR/utils/planning_distill.py'
spec = importlib.util.spec_from_file_location('_planning_distill', str(PATH))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

fails = []


print('=== rank-N readout Jacobian is at most N ===')
memory = torch.randn(1, 12, 8)
command = torch.tensor([[0.0, 1.0, 0.0]])
readout = module.PrivilegedEvidenceReadout(
    embed_dims=8, num_queries=3, num_heads=2, num_layers=1, num_commands=3,
    ffn_channels=16)
rank, jacobian = module.evidence_jacobian_rank(readout, memory, command)
rank_ok = 1 <= rank <= 3 and jacobian.shape == (3, 12)
print(f'  jacobian rank={rank} (cap 3); shape={tuple(jacobian.shape)}')
fails += [] if rank_ok else ['jacobian-rank']


print('\n=== frozen readout still trains student tokens, not R ===')
projector = module.TeacherMemoryProjector(('bevdepth', 'hdmapnet'), channels=8)
readout = module.PrivilegedEvidenceReadout(
    embed_dims=8, num_queries=4, num_heads=2, num_layers=1, num_commands=3,
    ffn_channels=16)
state = {
    **{'memory.' + k: v.clone() for k, v in projector.state_dict().items()},
    **{'readout.' + k: v.clone() for k, v in readout.state_dict().items()},
}
with tempfile.TemporaryDirectory() as tmp:
    token = 'rpedtoken0001'
    for name in ('bevdepth', 'hdmapnet'):
        directory = Path(tmp) / name / token[:2]
        directory.mkdir(parents=True)
        torch.save(
            dict(feature=torch.randn(8, 4, 4),
                 valid_mask=torch.ones(1, 4, 4)),
            directory / (token + '.pt'))
    ckpt = Path(tmp) / 'stage1.pth'
    torch.save(dict(state_dict=state), ckpt)
    distill = module.PrivilegedEvidenceDistillation(
        feature_root=tmp,
        readout_checkpoint=str(ckpt),
        teachers=dict(
            bevdepth=dict(cache_name='bevdepth'),
            hdmapnet=dict(cache_name='hdmapnet')),
        readout=dict(
            embed_dims=8, num_queries=4, num_heads=2, num_layers=1,
            num_commands=3, ffn_channels=16),
        student_bev_size=(4, 4),
        loss_weight=1.0)
    student_bev = torch.randn(1, 16, 8, requires_grad=True)
    student_tokens = torch.randn(4, 1, 8, requires_grad=True)
    losses, metrics = distill.forward_train(
        student_bev,
        [dict(sample_idx=token)],
        scene_query=student_tokens,
        ego_fut_cmd=torch.tensor([[[[1.0, 0.0, 0.0]]]]))
    sum(losses.values()).backward()
    frozen_ok = (
        'loss_rped_evidence' in losses and
        'loss_distill_' not in ''.join(losses) and
        student_tokens.grad is not None and
        float(student_tokens.grad.norm()) > 0 and
        all(p.grad is None for p in distill.parameters()) and
        'rped_cos' in metrics)
    print(f'  frozen R, student token grad, no dense MSE: {frozen_ok}')
    fails += [] if frozen_ok else ['frozen-rped-path']

    missing_tokens_ok = False
    try:
        distill.forward_train(
            student_bev, [dict(sample_idx=token)], ego_fut_cmd=command)
    except ValueError:
        missing_tokens_ok = True
    print(f'  missing student tokens raises: {missing_tokens_ok}')
    fails += [] if missing_tokens_ok else ['require-tokens']


print('\n=== same-question vs privileged queries are distinct paths ===')
with tempfile.TemporaryDirectory() as tmp:
    token = 'rpedtoken0002'
    for name in ('bevdepth', 'hdmapnet'):
        directory = Path(tmp) / name / token[:2]
        directory.mkdir(parents=True)
        torch.save(
            dict(feature=torch.randn(8, 4, 4),
                 valid_mask=torch.ones(1, 4, 4)),
            directory / (token + '.pt'))
    ckpt = Path(tmp) / 'stage1.pth'
    torch.save(dict(state_dict=state), ckpt)
    kwargs = dict(
        feature_root=tmp,
        readout_checkpoint=str(ckpt),
        teachers=dict(
            bevdepth=dict(cache_name='bevdepth'),
            hdmapnet=dict(cache_name='hdmapnet')),
        readout=dict(
            embed_dims=8, num_queries=4, num_heads=2, num_layers=1,
            num_commands=3, ffn_channels=16),
        student_bev_size=(4, 4))
    privileged = module.PrivilegedEvidenceDistillation(
        query_source='privileged', **kwargs)
    same_q = module.PrivilegedEvidenceDistillation(
        query_source='student', **kwargs)
    student_tokens = torch.randn(4, 1, 8)
    cmd = torch.tensor([[[[0.0, 1.0, 0.0]]]])
    metas = [dict(sample_idx=token)]
    bev = torch.randn(1, 16, 8)
    lp, _ = privileged.forward_train(
        bev, metas, scene_query=student_tokens, ego_fut_cmd=cmd)
    lq, _ = same_q.forward_train(
        bev, metas, scene_query=student_tokens, ego_fut_cmd=cmd)
    distinct = not torch.allclose(
        lp['loss_rped_evidence'], lq['loss_rped_evidence'])
    print(f'  privileged vs student-query losses differ: {distinct}')
    fails += [] if distinct else ['same-question-wiring']


print('\n=== ParaSSR hook and configs exist ===')
para = (ROOT / 'projects/mmdet3d_plugin/SSR/para_ssr.py').read_text()
head = (ROOT / 'projects/mmdet3d_plugin/SSR/para_ssr_head.py').read_text()
student = (ROOT / 'projects/configs/SSR/RPED_SSR_student.py').read_text()
dense = (ROOT / 'projects/configs/SSR/RPED_SSR_student_dense_ablation.py').read_text()
run = (ROOT / 'run.sh').read_text()
hook_ok = (
    'build_planning_distillation' in para and
    'uses_planning_queries' in para and
    'pooled_query' in head and
    "type='PrivilegedEvidenceDistillation'" in student and
    'DISTILL_SSR_student_bevfusion_maptrv2' in dense and
    'rped-teacher' in run and
    'rped-distill' in run and
    'rped-dense' in run)
print(f'  RPED wiring + ablation configs: {hook_ok}')
fails += [] if hook_ok else ['integration']


print('\n' + ('ALL RPED CHECKS PASS' if not fails else f'STILL FAILING: {fails}'))
sys.exit(1 if fails else 0)
