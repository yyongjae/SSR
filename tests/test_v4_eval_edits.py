"""refiner v4 A evaluation: tools/ck/v4/eval_edits.evaluate (96 + K vs 96) and the eval_e2e2 edit dumps.  CPU, synthetic.
  cd /workspace/yongjae/SSR-v4 && nice -n 10 env PYTHONPATH=$PWD OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
      /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v4_eval_edits.py
"""
import numpy as np
import pandas as pd

from navsim.agents.para_ssr.ck import constants as Cn
from tools.ck.v4 import eval_edits as EE

N, K = 40, 4
P = Cn.LBL["pdms"]


def _data(seed=0):
    rng = np.random.default_rng(seed)
    lab_pool = np.ones((N, 96, 9))
    lab_pool[..., P] = rng.uniform(0.3, 1.0, (N, 96))
    lab_lat = lab_pool.copy()
    src = np.stack([rng.choice(96, K, replace=False) for _ in range(N)])
    lab_e = lab_pool[np.arange(N)[:, None], src].copy()                            # edits = their sources
    packed = {"valid96": np.ones((N, 96), bool), "v2_final": rng.normal(0, 1, (N, 16)),
              "v2_im": rng.uniform(0.5, 1.0, (N, 16))}
    sl = rng.normal(0, 1, (N, 96, 5))
    infer = {"score_logit": sl, "edit_score_logit": sl[np.arange(N)[:, None], src].copy(), "edit_src": src}
    return packed, infer, {"pool": lab_pool, "lat": lab_lat, "edit": lab_e}, np.array([f"log{i % 7}" for i in range(N)])


def test_identity_edits_equal_the_96_selection():
    packed, infer, lab, logs = _data()
    r = EE.evaluate(packed, infer, lab, logs, [0.5, 1.0], lat_mode="off", n_boot=50)
    for row in r["rows"]:
        assert abs(row["pdms_96k"] - row["pdms_96"]) < 1e-12        # an edit equal to its source changes nothing
    assert r["edit_vs_src"]["better"] == 0.0 and r["edit_vs_src"]["worse"] == 0.0


def test_better_scored_edits_are_picked():
    packed, infer, lab, logs = _data(1)
    infer["edit_score_logit"][:, 0] = 8.0                               # the critic loves edit 0 ...
    lab["edit"][:, 0, P] = 1.0                                          # ... and it is perfect
    r = EE.evaluate(packed, infer, lab, logs, [1.0], lat_mode="off", n_boot=50)
    row = r["rows"][0]
    assert row["edit_pick_rate"] > 0.9 and row["pdms_96k"] > row["pdms_96"] and row["edit_pick_pdms"] == 1.0


def test_writer_stores_edits(tmp_path):
    from tools.ck.e2e import eval_e2e as E1
    from tools.ck.e2e2 import eval_e2e2 as E2
    P = E1.Paths(tmp_path / "runX", "navtrain_val")
    tdf = pd.DataFrame({"token": [f"t{i}" for i in range(3)], "log": ["a", "a", "b"], "city": ["c"] * 3})
    w = E2.EvalRootWriter2(P, tdf)
    B = 2
    pred = {"ck2_cand96": np.zeros((B, 96, 8, 3), np.float32), "ck2_valid96": np.ones((B, 96), bool),
            "ck2_cand_idx": np.zeros((B, 16), np.int16), "ck2_v2_final": np.zeros((B, 16), np.float32),
            "ck2_v2_im": np.ones((B, 16), np.float32), "ck2_v2_sim": np.zeros((B, 16, 5), np.float32),
            "ck2_score_logit": np.zeros((B, 96, 5), np.float32), "ck2_w_lat": np.zeros((B, 96, 6), np.float32),
            "ck2_e_lat": np.zeros((B, 96, 6), np.float32), "ck2_lat_traj": np.zeros((B, 96, 8, 3), np.float32),
            "ck2_edit_traj": np.ones((B, 8, 8, 3), np.float32), "ck2_edit_src": np.arange(16).reshape(B, 8),
            "ck2_edit_u": np.zeros((B, 8, 12), np.float32), "ck2_edit_score_logit": np.zeros((B, 8, 5), np.float32)}
    ok = w.write([0, 2], pred, np.zeros((B, 8), np.float32), np.zeros((B, 8, 3), np.float32),
                 np.zeros((B, 8, 3), np.float32))
    w.flush()
    assert ok.all()
    et = np.load(P.infer / "edit_traj.npy")
    es = np.load(P.infer / "edit_src.npy")
    assert et.shape == (3, 8, 8, 3) and float(et[0].min()) == 1.0 and np.isnan(et[1]).all()
    assert es[2].tolist() == list(range(8, 16)) and (es[1] == -1).all()
