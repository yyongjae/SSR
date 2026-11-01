"""Punchline ablation: dense adapter-MSE on the same teacher pair.

This is the legacy DISTILL_SSR_student_bevfusion_maptrv2 contract, renamed
so the paper table can call it E_dense.  It should recreate PARA-SSR's 1s
damage.  Do not mix this loss into the main RPED student.
"""
_base_ = ['./DISTILL_SSR_student_bevfusion_maptrv2.py']
