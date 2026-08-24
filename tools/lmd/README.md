# LMD decomposition of PARA-SSR's BEV

Adapts Layer-Wise Modality Decomposition ([arXiv:2511.00859](https://arxiv.org/pdf/2511.00859),
NeurIPS 2025) to answer which parts of `bev_embed` the planner actually uses,
which parts push it away from the ground-truth trajectory, and which input
source (camera / prev_bev / learned prior / can_bus / positional code) supplied
each of them.

Design, definitions and caveats: [`report/09_lmd_planning_centric_bev.md`](../../report/09_lmd_planning_centric_bev.md).

    python3 tools/lmd/verify_lmd_linearisation.py

`lmd_core.py`  frozen-switch versions of every nonlinearity on the path, plus
               the forward and adjoint readers of the resulting linear map.
`verify_lmd_linearisation.py`
               proves the decomposition is residual-free on this model's op
               stack: upstream (10 sources -> BEV), downstream (10,000 BEV
               cells -> trajectory) and the two composed. float64, so the
               1e-15 numbers mean exact rather than close.

Not yet implemented: the hooks that run this against the real mmcv modules and
a checkpoint (report #09 section 5).
