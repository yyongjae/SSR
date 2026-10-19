"""refiner-v3 temporal critic branch (docs/refiner_v3/SPEC.md section 4).

Modules
  config   constants shared by every builder + TemporalCfg (on / source / forecaster / augmentation / paths)
  objects  (B1, numpy only) causal object sources, GT target matching, numpy CA reference
  forecast Forecast container, trim_objects, analytic CA / oracle forecasts, LearnedForecaster (torch)
  raster   parameter-free differentiable path-time rasteriser + a3 gap scalars (torch)
  branch   TemporalBranch: shared ST encoder / projections + forecaster, TemporalCtx, the CKTrunk hook
  loss     forecast_loss (winner-take-all Laplace NLL + mode CE)

Nothing is imported here on purpose: `objects` must stay importable without torch, and the CK2 code path never
imports this package while the temporal branch is off.
"""
