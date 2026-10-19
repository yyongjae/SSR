"""refiner v4 (reports 70-77): failure-localised station x arrival-time hazard readout on the parent corridor (I1),
supervised P_ref scene query (77 Q1), scene-free comfort head, structured EP head, hazard-localised lateral head.

Everything here is opt-in through V4Cfg (config.py); V4Cfg() = off = the CK2 / refiner-v3 network bit for bit."""
from .config import READOUTS, V4Cfg

__all__ = ["READOUTS", "V4Cfg"]
