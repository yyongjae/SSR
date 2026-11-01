"""CK (Corridor candidate head) Phase 1 pipeline (report 44; contract ck/spec/interface_contract.json).

pipeline : train_ck.py, infer_ck.py, kd_targets.py, eval_ck.py, smoke_ck.sh, run_phase1.sh (+ run_main.sh /
           smoke_all.sh aliases), ckutil.py (shared helpers of these entry points only).
data     : tools/ck/data/** (v2 dump, pack, official labels, CKDataset).
core     : navsim/agents/para_ssr/ck/** (CKNet, losses, KD combine, selection).
"""
