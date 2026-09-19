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
idx = dump["plan_topk_index"]
rows = np.arange(len(idx))[:, None]
# anchor (float32, exact) + offset (float16 but small, so ~mm): the saved top-k trajectories are
# float16 in absolute coordinates (up to ~3 cm off at 60 m) and are not used
top = anchors[idx].astype(np.float32) + dump["trajectory_offset"][rows, idx].astype(np.float32)
err = np.abs(top[:, 0] - dump["trajectory"]).max()
assert err < 1e-2, f"top-1 refined != output ({err:.4f})"
print(f"top-1 refined vs output: max |diff| {err:.4f} m")
for r in range(k):
    np.savez_compressed(out / f"refined_r{r + 1}.npz", tokens=dump["tokens"], trajectory=top[:, r])
    np.savez_compressed(out / f"anchor_r{r + 1}.npz", tokens=dump["tokens"], trajectory=anchors[idx[:, r]].astype(np.float32))
print(f"{2 * k} variants -> {out}")
