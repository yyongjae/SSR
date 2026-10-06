"""Tabulate readout PDMS runs: mean +- std over seeds, and the BEV share above S_ego.

    python tools/readout/collect_results.py runs/readout
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

NAME = re.compile(r"^(?P<arm>.+)_(?P<preset>h\d)_s(?P<seed>\d+)$")
QUANTITY = {  # (run arm, pdms file) -> reported quantity
    ("teacher", "pdms_teacher"): "S_own",
    ("ego", "pdms_teacher"): "S_ego",
    ("shuffled", "pdms_teacher"): "S_shuffled",
    ("student", "pdms_student"): "S_student",
    ("teacher", "pdms_transfer"): "S_transfer",
    ("transfer_adapter", "pdms_transfer_adapter"): "S_transfer+A",
}


def main(root):
    scores = defaultdict(list)
    for f in sorted(Path(root).glob("*/pdms_*.json")):
        m = NAME.match(f.parent.name)
        key = QUANTITY.get((m.group("arm"), f.stem)) if m else None
        if key:
            scores[(m.group("preset"), key)].append(json.loads(f.read_text())["score"])
    presets = sorted({p for p, _ in scores})
    order = ["S_ego", "S_own", "S_shuffled", "S_student", "S_transfer", "S_transfer+A"]
    print(f"{'preset':6s} {'quantity':13s} {'PDMS':>16s} {'minus S_ego':>12s}  n")
    for p in presets:
        ego = np.mean(scores[(p, "S_ego")]) if scores.get((p, "S_ego")) else float("nan")
        for q in order:
            v = scores.get((p, q))
            if not v:
                continue
            print(f"{p:6s} {q:13s} {100 * np.mean(v):9.2f} +- {100 * np.std(v):4.2f} "
                  f"{100 * (np.mean(v) - ego):12.2f}  {len(v)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/readout")
