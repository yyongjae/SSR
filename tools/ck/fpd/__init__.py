"""FPD teacher pipeline (ck_e2e2.arch 'fpd'; FPD-Drive arXiv 2610.13232 refiner structure, navsim/agents/para_ssr/ck/fpd_net.py).

train_fpd_teacher.py   one privileged teacher refiner (FPDRefinerNet) per cached teacher BEV (arm T: DET / BEVFusion,
                       arm M: MAP / ReSMap), offline, in three stages: dit -> bank (GPU sampling) -> scorer
label_fpd_bank.py      official PDM labels of the stage-bank DiT samples (CPU process pool)
bank.py                layout of run_dir/bank/<split>/ and the datasets shared by the two entry points
Launcher: tools/release/setup/run_fpd_teachers.sh.  The finished run dir is read by fpd_net.load_fpd_teacher.
"""
