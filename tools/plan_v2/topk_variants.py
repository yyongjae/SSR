"""D2: write the top-k candidates of a v2 dump (dump_plan_predictions.py --v2-extras) as separate
trajectory files, so each can be scored with the EPDMS scorer (data/logs/epdms_from_npz.sh):

    refined_r{k}  anchor + offset of the k-th ranked anchor (refined_r1 = the model's output)
    anchor_r{k}   the k-th ranked anchor itself, no offset (what the score head was trained to score)

    python tools/plan_v2/topk_variants.py <dump.npz> <anchors.npy> <out_dir> [k=3]
"""
import sys
from pathlib import Path

import numpy as np

dump, anchors, out = np.load(sys.argv[1]), np.load(sys.argv[2]), Path(sys.argv[3])
k = int(sys.argv[4]) if len(sys.argv) > 4 else 3
out.mkdir(parents=True, exist_ok=True)
idx, top = dump["plan_topk_index"], dump["plan_topk_trajectory"].astype(np.float32)
assert np.abs(top[:, 0] - dump["trajectory"]).max() < 1e-2, "top-1 refined != output"
for r in range(k):
    np.savez_compressed(out / f"refined_r{r + 1}.npz", tokens=dump["tokens"], trajectory=top[:, r])
    np.savez_compressed(out / f"anchor_r{r + 1}.npz", tokens=dump["tokens"], trajectory=anchors[idx[:, r]].astype(np.float32))
print(f"{2 * k} variants -> {out}")
