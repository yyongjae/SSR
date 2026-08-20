# --- silence shapely 2.x map-metric warning spam -----------------------------
# The map-quality hook scores predicted vs GT polylines with shapely; on early
# (garbage) predictions GEOS returns NaN and shapely re-raises it as a numpy
# RuntimeWarning on *every* call ("invalid value encountered in intersection/…").
# Left alone it prints millions of lines and, when teed to disk, can fill the
# filesystem. The NaN is already handled as a miss downstream, so it is noise.
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=r"invalid value encountered in "
            r"(intersection|difference|union|symmetric_difference|"
            r"intersects|contains|within|distance|area|length|buffer)",
    category=RuntimeWarning,
)

from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
from .core.bbox.coders.nms_free_coder import NMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost
from .core.evaluation.eval_hooks import CustomDistEvalHook
from .datasets.pipelines import (
  PhotoMetricDistortionMultiViewImage, PadMultiViewImage, 
  NormalizeMultiviewImage,  CustomCollect3D)
from .models.backbones.vovnet import VoVNet
from .models.utils import *
from .models.opt.adamw import AdamW2
from .SSR import *
