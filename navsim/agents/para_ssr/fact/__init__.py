"""PARA-SSR v3: factorized path x speed vocabulary scored coarse-to-fine against PDM labels.

vocab.py    path / speed vocabulary and their composition into trajectories
targets.py  GT rasters (future occupancy, road SDF on the BEV grid) and the route centerline
proxy.py    proxy PDM sub-scores of arbitrary candidates on the GPU
planner.py  the planner module (stage 1 paths, stage 2 speeds, stage 3 fine scoring, offset)
losses.py   its losses and the two auxiliary BEV heads' losses
"""
