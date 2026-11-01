"""CPU-only proof: report attempted CUDA reseeds without touching any CUDA device.
Run from SSR with its Python environment. All CUDA seeding calls are replaced
by a spy; teacher weights load on CPU. No training state or repository changes.
"""
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, '/home/external-user/yongjae/SSR')
import torch
from navsim.agents.para_ssr.refiner.e2e import build_student, StageE
from navsim.agents.para_ssr.configs.default import ParaSSRConfig

root = Path('/home/external-user/ssd/yongjae_refiner/stageE/teachers')
cfg = ParaSSRConfig(refiner_mode='E2', kd_space='decoded', kd_teacher_runs=(
    str(root / 'stageT4_T_fold0_seed0'), str(root / 'stageT4_M_fold0_seed0')))
with patch('torch.cuda.manual_seed_all') as spy:
    before = torch.random.get_rng_state().clone()
    net = build_student(0, 1.0)
    print('student_cpu_rng_preserved', torch.equal(before, torch.random.get_rng_state()))
    print('student_cuda_reseed_calls', [call.args[0] for call in spy.call_args_list])
    spy.reset_mock()
    stage = StageE(cfg)
    stage.teachers(torch.device('cpu'))
    print('teacher_cpu_rng_preserved', torch.equal(before, torch.random.get_rng_state()))
    print('teacher_cuda_reseed_calls', [call.args[0] for call in spy.call_args_list])
