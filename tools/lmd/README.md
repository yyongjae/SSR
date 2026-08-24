# What did the auxiliary tasks do to the BEV feature?

Borrows the mechanism of Layer-Wise Modality Decomposition
([arXiv:2511.00859](https://arxiv.org/pdf/2511.00859), NeurIPS 2025) but swaps
its axis. LMD decomposes a fused feature over additive INPUTS; aux-vs-no-aux is
a difference between two trained WEIGHT sets, so the axis moves from modality to
supervision. What transfers is the property that makes LMD work: freeze every
nonlinearity at its operating point and the network is exactly affine, after
which the model difference telescopes over weight groups with zero residual.

Design, definitions and caveats -- including the basin precondition that gates
all of it: [`report/09_lmd_planning_centric_bev.md`](../../report/09_lmd_planning_centric_bev.md).

    python3 tools/lmd/verify_lmd_linearisation.py        # is the path linearisable
    python3 tools/lmd/verify_model_delta.py --shapley    # is the aux delta decomposable

| file | what it is |
|---|---|
| `lmd_core.py` | frozen-switch versions of every nonlinearity on the path |
| `replica.py` | the model's op stack, exposed as an ordered chain of weight groups so a hybrid forward can take group g from one model and the rest from another |
| `verify_lmd_linearisation.py` | proves the trajectory splits over the 10,000 BEV cells with no residual -- this is what defines a planning-centric cell |
| `verify_model_delta.py` | proves the aux-vs-no-aux difference splits over weight groups (telescoping and exact Shapley), and per BEV cell into the part that helped planning and the part that hurt it |

float64 throughout, so the 1e-15 numbers mean exact rather than close.

Not yet implemented: the hooks that run this against the real mmcv modules and
two checkpoints (report #09 section 6).
