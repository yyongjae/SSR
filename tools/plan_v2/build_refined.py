"""Write the model's 256 REFINED candidates per scene (anchor + predicted offset) as one file,
so the official scorer can label them the same way the bare anchors were labelled.

The score head is trained on, and evaluated against, the bare anchors -- but the car drives the
refined trajectory. Labelling both tells us how much of the gap is choosing and how much is refining.

    python tools/plan_v2/build_refined.py <dump.npz> <anchors.npy> <out.npz>
"""
import sys

import numpy as np

dump, anchors, out = np.load(sys.argv[1]), np.load(sys.argv[2]).astype(np.float32), sys.argv[3]
refined = anchors[None] + dump["trajectory_offset"].astype(np.float32)          # [N, 256, 8, 3]
row = np.arange(len(refined))
err = np.abs(refined[row, dump["plan_topk_index"][:, 0]] - dump["trajectory"]).max()
assert err < 1e-2, f"refined top-1 != the model's output ({err:.4f} m)"
print(f"refined top-1 vs output: max |diff| {err:.4f} m")
np.savez_compressed(out, tokens=dump["tokens"], trajectory=refined.astype(np.float16))
print(f"{refined.shape} -> {out}")
