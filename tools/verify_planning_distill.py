"""CPU regression checks for the planning-distillation contract."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'projects/mmdet3d_plugin/SSR/utils/planning_distill.py'
spec = importlib.util.spec_from_file_location('_planning_distill', str(PATH))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

fails = []


print('=== physical BEV alignment ===')
feature = torch.randn(2, 4, 5, 7)
aligned, valid = module.align_bev_feature(
    feature, (-3.5, -2.5, 3.5, 2.5), (-3.5, -2.5, 3.5, 2.5),
    (5, 7))
identity = torch.allclose(feature, aligned, atol=1e-5)
print(f'  identity grid: {identity}; valid={float(valid.float().mean()):.1f}')
fails += [] if identity and bool(valid.all()) else ['alignment-identity']

source = torch.arange(15, dtype=torch.float32).reshape(1, 1, 3, 5)
swapped, _ = module.align_bev_feature(
    source, (-2.5, -1.5, 2.5, 1.5), (-1.5, -2.5, 1.5, 2.5),
    (5, 3), swap_xy=True)
axis_ok = torch.allclose(swapped[0, 0], source[0, 0].t(), atol=1e-5)
print(f'  x/y swap equals transpose: {axis_ok}')
fails += [] if axis_ok else ['alignment-swap']

oriented, _ = module.align_bev_feature(
    source, (-2.5, -1.5, 2.5, 1.5), (-1.5, -2.5, 1.5, 2.5),
    (5, 3), swap_xy=True, flip_y=True)
orientation_ok = torch.allclose(
    oriented[0, 0], source[0, 0].t().flip(1), atol=1e-5)
print(f'  nuScenes ego -> SSR LIDAR orientation: {orientation_ok}')
fails += [] if orientation_ok else ['alignment-orientation']


print('\n=== frozen adapter must still pass gradient to student BEV ===')
adapter = module.PlanningBEVAdapter(8, 8)
adapter.requires_grad_(False)
teacher = torch.randn(2, 8, 4, 4)
student = torch.randn(2, 8, 4, 4, requires_grad=True)
with torch.no_grad():
    target = adapter(teacher)
prediction = adapter(student)
loss = (prediction - target).square().mean()
loss.backward()
input_grad = float(student.grad.norm())
parameter_grads = [p.grad for p in adapter.parameters()]
grad_ok = input_grad > 0 and all(value is None for value in parameter_grads)
print(f'  student grad norm={input_grad:.6f}; adapter grads all None={grad_ok}')
fails += [] if grad_ok else ['frozen-gradient-path']


print('\n=== token-addressed cache and strict adapter checkpoint ===')
with tempfile.TemporaryDirectory() as tmp:
    token = 'abcdef012345'
    cached = torch.randn(8, 4, 4)
    for teacher_name in ('bevdepth', 'hdmapnet'):
        directory = Path(tmp) / teacher_name / token[:2]
        directory.mkdir(parents=True)
        torch.save(dict(
            feature=cached, valid_mask=torch.ones(1, 4, 4)),
            directory / (token + '.pt'))
    store = module.TeacherFeatureStore(tmp, 'bevdepth')
    loaded, mask = store.load_batch([token], torch.device('cpu'), torch.float32)
    cache_ok = torch.equal(loaded[0], cached) and bool(mask.all())
    print(f'  sharded lookup and tensor round-trip: {cache_ok}')
    fails += [] if cache_ok else ['feature-store']

    source_adapter = module.PlanningBEVAdapter(8, 8)
    state = {'branches.bevdepth.adapter.' + key: value.clone()
             for key, value in source_adapter.state_dict().items()}
    destination = module.PlanningBEVAdapter(8, 8)
    prefix = module.load_prefixed_module(
        destination, state, 'branches.bevdepth.adapter.')
    checkpoint_ok = all(torch.equal(a, b) for a, b in zip(
        source_adapter.state_dict().values(),
        destination.state_dict().values()))
    print(f'  strict prefix load ({prefix}): {checkpoint_ok}')
    fails += [] if checkpoint_ok else ['adapter-checkpoint']

    checkpoint_paths = {}
    for teacher_name in ('bevdepth', 'hdmapnet'):
        state = {
            f'branches.{teacher_name}.adapter.{key}': value.clone()
            for key, value in source_adapter.state_dict().items()
        }
        checkpoint_path = Path(tmp) / f'stage1_{teacher_name}.pth'
        torch.save(dict(state_dict=state), checkpoint_path)
        checkpoint_paths[teacher_name] = str(checkpoint_path)
    distillation = module.PlanningDistillation(
        feature_root=tmp,
        adapter_checkpoint=checkpoint_paths,
        student_bev_size=(4, 4),
        cache_size=(4, 4),
        branches={name: dict(
            cache_name=name,
            adapter=dict(channels=8, hidden_channels=8),
            adapter_prefix=f'branches.{name}.adapter.')
            for name in ('bevdepth', 'hdmapnet')})
    student_bev = torch.randn(1, 16, 8, requires_grad=True)
    distill_losses, distill_metrics = distillation.forward_train(
        student_bev, [dict(sample_idx=token)])
    sum(distill_losses.values()).backward()
    exact_path_ok = (
        student_bev.grad is not None and
        float(student_bev.grad.norm()) > 0 and
        all(parameter.grad is None for parameter in
            distillation.parameters()) and
        len(distill_losses) == 2 and len(distill_metrics) == 6)
    print(f'  separate checkpoints, complete frozen path: {exact_path_ok}')
    fails += [] if exact_path_ok else ['two-teacher-path']


print('\n=== xy-major npz caches (BEVFusion / MapTRv2) ===')
width, height, channels = 8, 8, 4
xs = torch.arange(width).view(width, 1).expand(width, height)
ys = torch.arange(height).view(1, height).expand(width, height)
# token = x_index * Y + y_index, values encode the source (x, y) cell.
tokens = torch.stack((xs.reshape(-1).float(), ys.reshape(-1).float(),
                      torch.zeros(width * height), torch.ones(width * height)),
                     dim=-1)
mapped = module.xy_tokens_to_map(tokens, (width, height))
xy_layout_ok = (
    mapped.shape == (channels, height, width) and
    torch.equal(mapped[0], xs.t().float()) and
    torch.equal(mapped[1], ys.t().float()))
print(f'  xy token -> CHW: {xy_layout_ok}')
fails += [] if xy_layout_ok else ['xy-tokens-to-map']

with tempfile.TemporaryDirectory() as tmp:
    token = 'aabbccddeeff'
    packed = dict(bev_feature=tokens.half().numpy())
    for teacher_name in ('bevfusion', 'maptrv2'):
        directory = (Path(tmp) / teacher_name / 'cache_train_100x100' /
                     'samples' / token[:2])
        directory.mkdir(parents=True)
        np.savez(directory / (token + '.npz'), **packed)

    fusion_store = module.TeacherFeatureStore(
        tmp, 'bevfusion', grid_size=(width, height),
        source_bounds=(-4.0, -4.0, 4.0, 4.0),
        target_bounds=(-2.0, -2.0, 2.0, 2.0),
        output_size=(height, width))
    fusion, fusion_mask = fusion_store.load_batch(
        [token], torch.device('cpu'), torch.float32)
    fusion_ok = (
        tuple(fusion.shape) == (1, channels, height, width) and
        not torch.allclose(fusion[0], mapped.float(), atol=1e-4) and
        bool(fusion_mask.any()))
    print(f'  BEVFusion crop/orient to SSR grid: {fusion_ok} '
          f'shape={tuple(fusion.shape)} valid={float(fusion_mask.float().mean()):.2f}')
    fails += [] if fusion_ok else ['bevfusion-npz']

    maptr_store = module.TeacherFeatureStore(
        tmp, 'maptrv2', grid_size=(width, height),
        source_bounds=(-2.0, -2.0, 2.0, 2.0),
        target_bounds=(-2.0, -2.0, 2.0, 2.0),
        output_size=(height, width))
    maptr, maptr_mask = maptr_store.load_batch(
        [token], torch.device('cpu'), torch.float32)
    maptr_ok = (
        torch.allclose(maptr[0], mapped.float(), atol=1e-4) and
        bool(maptr_mask.all()))
    print(f'  MapTRv2 identity CHW: {maptr_ok}')
    fails += [] if maptr_ok else ['maptrv2-npz']

    missing_ok = False
    try:
        fusion_store.load_batch(
            ['missingtoken'], torch.device('cpu'), torch.float32)
    except FileNotFoundError:
        missing_ok = True
    print(f'  missing npz raises FileNotFoundError: {missing_ok}')
    fails += [] if missing_ok else ['npz-missing']

real_root = Path(
    '/data3/byounggun/rideflux/pretrained_checkpoints/distill_bev_cache')
real_token = '005894cbaa2c482a82fd915fa7597ad3'
if (real_root / 'bevfusion' / 'cache_train_100x100' / 'samples' /
        real_token[:2] / (real_token + '.npz')).is_file():
    real_store = module.TeacherFeatureStore(str(real_root), 'bevfusion')
    real_feat, real_mask = real_store.load_batch(
        [real_token], torch.device('cpu'), torch.float32)
    real_ok = tuple(real_feat.shape) == (1, 256, 100, 100) and bool(real_mask.any())
    print(f'  real BEVFusion npz: {real_ok} shape={tuple(real_feat.shape)}')
    fails += [] if real_ok else ['real-bevfusion-npz']
else:
    print('  real BEVFusion npz: skipped')

if (real_root / 'maptrv2' / 'cache_train_100x100' / 'samples' /
        real_token[:2] / (real_token + '.npz')).is_file():
    real_store = module.TeacherFeatureStore(str(real_root), 'maptrv2')
    real_feat, real_mask = real_store.load_batch(
        [real_token], torch.device('cpu'), torch.float32)
    real_ok = tuple(real_feat.shape) == (1, 256, 100, 100) and bool(real_mask.all())
    print(f'  real MapTRv2 npz: {real_ok} shape={tuple(real_feat.shape)}')
    fails += [] if real_ok else ['real-maptrv2-npz']
else:
    print('  real MapTRv2 npz: skipped')


print('\n=== integration is present and student auxiliaries are off ===')
para_source = (ROOT / 'projects/mmdet3d_plugin/SSR/para_ssr.py').read_text()
student_source = (ROOT / 'projects/configs/SSR/DISTILL_SSR_student.py').read_text()
integration_ok = ('PlanningDistillation' in para_source and
                  'loss_distill_' in PATH.read_text() and
                  'det_motion_head=None' in student_source and
                  'map_head=None' in student_source and
                  'occ_head=None' in student_source)
print(f'  ParaSSR hook + planning-only student config: {integration_ok}')
fails += [] if integration_ok else ['integration']


print('\n' + ('ALL PLANNING-DISTILL CHECKS PASS' if not fails
              else f'STILL FAILING: {fails}'))
sys.exit(1 if fails else 0)
