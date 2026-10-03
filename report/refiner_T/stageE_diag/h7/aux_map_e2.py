"""H7: official aux det / map mAP (protocol V3 evaluator navsim.evaluate.aux_metrics) of E2's own heads on navtest,
from the h7 dump records (GT copied from E0's records). E0 reference = work_dirs/eval/para_ssr_interaction_final_aux."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/external-user/yongjae/SSR")
from navsim.evaluate.aux_metrics import evaluate_auxiliary_records
REC = Path("/home/external-user/ssd/yongjae_refiner/stageE_diag/h7/e2_aux/records")
KEYS = ("det_pred_boxes", "det_pred_scores", "det_pred_labels", "det_gt_boxes", "det_gt_labels",
        "map_pred_points", "map_pred_scores", "map_pred_labels", "map_gt_points", "map_gt_labels")
def recs():
    for f in sorted(REC.glob("*.npz")):
        with np.load(f) as z:
            r = {k: z[k] for k in KEYS}
            r["token"] = z["token"]
            yield r
m = evaluate_auxiliary_records(recs())
e0 = json.load(open("/home/external-user/yongjae/SSR/work_dirs/eval/para_ssr_interaction_final_aux/aux_metrics.json"))["metrics"]
out = {"E2": {"det_mAP": m["detection"]["mAP"], "map_mAP": m["map"]["mAP"],
              "det_AP": {c: v["AP"] for c, v in m["detection"]["classes"].items()},
              "map_AP": {c: v["AP"] for c, v in m["map"]["classes"].items()}},
       "E0": {"det_mAP": e0["detection"]["mAP"], "map_mAP": e0["map"]["mAP"],
              "det_AP": {c: v["AP"] for c, v in e0["detection"]["classes"].items()},
              "map_AP": {c: v["AP"] for c, v in e0["map"]["classes"].items()}},
       "note": "BEVFusion det mAP on the same V3 protocol 79.6 (report/28); ReSMap vector mAP 77.4 (its README, own eval)"}
json.dump(out, open("/home/external-user/yongjae/SSR/report/refiner_T/stageE_diag/h7/aux_map_e2.json", "w"), indent=1)
print(json.dumps(out, indent=1))
