"""Planning readout: measure and distil planning-relevant BEV content.

See report/19_planning_readout.md for the design and the experiment protocol.
"""
from .bev_cache import BevCache, STUDENT_LAYOUT, TEACHER_LAYOUT
from .readout import PlanningReadout, READOUT_PRESETS, build_readout, load_readout

__all__ = [
    "BevCache",
    "STUDENT_LAYOUT",
    "TEACHER_LAYOUT",
    "PlanningReadout",
    "READOUT_PRESETS",
    "build_readout",
    "load_readout",
]
