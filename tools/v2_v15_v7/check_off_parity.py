#!/usr/bin/env python
"""v2_v15_v7 setup check (report 132 G1-1, agent level): with the refiner (ck_e2e2) and the BEV KD (use_distill) off,
this branch's ParaSSRAgent equals the agent of the base it was built on (km/para-ssr-v2-seg @79051bb): same state_dict
keys / values, optimizer groups, target builders, callbacks, and bit-identical training-step loss / logs / gradients
and eval outputs on the real 2-sample CPU fixture.  Runs tests/test_ck2_e2e_agent.py::test_off_bit_identical with the
reference agent file taken from 79051bb (the test's own reference copy lived in an old session scratchpad).

  PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= /venv/ssr/bin/python tools/v2_v15_v7/check_off_parity.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))


def main():
    ref = subprocess.run(["git", "-C", str(REPO), "show", "79051bb:navsim/agents/para_ssr/para_ssr_agent.py"],
                         check=True, capture_output=True, text=True).stdout
    tmp = Path(tempfile.mkdtemp(prefix="v7_off_parity_")) / "para_ssr_agent.orig.py"
    tmp.write_text(ref)
    import pytest
    import test_ck2_e2e_agent as T
    T.ORIG_AGENT = tmp
    batch = T.torch.load(T.FIX2, map_location="cpu", weights_only=False)
    T.test_off_bit_identical.__wrapped__(batch) if hasattr(T.test_off_bit_identical, "__wrapped__") else \
        T.test_off_bit_identical(batch)
    print("OK: refiner / BEV KD off == km/para-ssr-v2-seg @79051bb agent (state, optimizer, builders, step, eval)")


if __name__ == "__main__":
    main()
