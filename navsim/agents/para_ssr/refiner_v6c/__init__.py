"""refiner v6c (SSR/report/124_refiner_v6c_design_on_corridor_cf.md): a per-step risk head (R-E), a corrector (R-C)
and a row-wise judge (R-J) on the trajectory the corridor-cf candidate scorer picked; teacher heads (T-D, T-M, T-E) on
the privileged BEVs that the BEV distillation already loads.

Modules:
  labels   official per-step labels of tracked paths (at-fault overlap, TTC hit, gap, approach speed, object position,
           off-road), from the same exact tables as exact_score.official_scores
  nets     R-E, R-C (+ decoder), R-J (+ eff head)
  losses   L_E (GT + KD), L_corr (teacher term + GT guard + road / comfort / EC / progress), L_J
  teacher  T-D / T-M / T-E (cross-fitted copies) and their online use during student training
  hook     the model / agent integration (forward after the scorer, loss after the scorer loss)
"""
