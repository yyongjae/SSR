"""Pack WoTE's ``formatted_pdm_score_256.npy`` (a 1.5 GB pickled dict) into a memory-mappable array.

Every dataloader worker would otherwise unpickle the whole dict.  Output:
``<out>.npy`` float16 [N, 5, K] in ``SIM_KEYS`` order and ``<out>.tokens.json``.

    python tools/plan_v2/pack_pdm_scores.py data/planning_vb/formatted_pdm_score_256.npy data/planning_vb/pdm_score_256
"""
import json
import sys

import numpy as np

from navsim.agents.para_ssr.modules.anchor_planner import SIM_KEYS

src, out = sys.argv[1], sys.argv[2]
d = np.load(src, allow_pickle=True).item()
tokens = sorted(d)
arr = np.stack([np.stack([np.asarray(d[t]["trajectory_scores"][0][k], dtype=np.float32) for k in SIM_KEYS])
                for t in tokens]).astype(np.float16)
np.save(out + ".npy", arr)
json.dump(tokens, open(out + ".tokens.json", "w"))
print(arr.shape, arr.dtype, "->", out + ".npy")
