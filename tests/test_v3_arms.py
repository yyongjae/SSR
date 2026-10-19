"""refiner-v3 Step-1 arm configurations (tools/ck/v3/arms/*.json, tools/ck/v3/arms.py; builder LOSS+EVAL).  CPU.

  cd /workspace/yongjae/SSR-v3 && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/yongjae/SSR-v3 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 10 /venv/ssr/bin/python -m pytest -q -p no:cacheprovider tests/test_v3_arms.py

Every arm validates (TemporalCfg round trip + validate('T')), a / b / d differ only in forecaster / n_modes, the
privileged / '_PRIV' rule, gates, the rendered CLI (full ck2T10 recipe + temporal flags, --t-modes 1 for ca /
oracle), the recipe equals ck2T10/config.json through train_ck2's own parser, mutated configs are refused, and the
CLI parses with train_ck2's parser once the trainer has the --temporal / --t-* flags (skipped before).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CK = str(Path(__file__).resolve().parents[1])   # repo root of this worktree
if sys.path[0] != CK:
    sys.path.insert(0, CK)

import pytest  # noqa: E402

from navsim.agents.para_ssr.ck.temporal.config import TemporalCfg  # noqa: E402
from tools.ck.v3 import arms as AR  # noqa: E402

RUN_T10 = Path("/home/external-user/ssd/yongjae_refiner/ck/ck2/train/ck2T10/config.json")


def test_all_arms_validate_and_differ_only_in_forecaster():
    res = AR.check_all()
    assert set(res) == {"a", "b", "c", "d", "pre"}
    assert res["a"]["name"] == "v3T10_a_learned" and res["b"]["name"] == "v3T10_b_ca"
    assert res["d"]["name"] == "v3T10_d_oracle_PRIV" and res["c"]["name"] == "ck2T10"
    assert [res[k]["privileged"] for k in "abcd"] == [False, False, False, True]
    assert res["a"]["gates"] == ["G1", "G2", "G3", "G4"] and res["c"]["gates"] == ["G4"] and res["d"]["gates"] == []
    ta, tb, td = (res[k]["temporal"] for k in "abd")
    diff = {k for k in ta if ta[k] != tb[k]} | {k for k in ta if ta[k] != td[k]}
    assert diff == {"forecaster", "n_modes", "fc_init"}        # fc_init: a's pre-trained forecaster (review fix D3)
    assert ta["fc_train"] == tb["fc_train"] == td["fc_train"] == "frozen"
    assert ta["fc_init"] == str(Path(AR.load("pre")["out"]) / "forecaster.pt") and tb["fc_init"] == td["fc_init"] == ""
    assert ta["pack_root"].endswith("/objpack_k96")
    assert TemporalCfg.from_dict(res["c"]["temporal"]).on is False
    for k in "abd":
        cfg = TemporalCfg.from_dict(res[k]["temporal"])
        cfg.validate("T")
        assert cfg.privileged == (k == "d")
        assert cfg.fc_backprop is False and cfg.lambda_fc == 1.0
        with pytest.raises(ValueError):
            cfg.validate("M")                      # MAP teacher: temporal off


def test_cli_render():
    a = AR.cli("a")
    b = AR.cli("b")
    d = AR.cli("d")
    rec = AR.recipe()["cli"]
    for x in (a, b, d):
        assert x[:len(rec)] == rec and x[-1] == "--resume"
        assert x[x.index("--temporal") + 1] == "1" and x[x.index("--t-source") + 1] == "bevfusion"
        assert x[x.index("--out-root") + 1] == "/home/external-user/ssd/yongjae_refiner/ck/v3/train"
        assert x[x.index("--t-fc-backprop") + 1] == "0"
    assert a[a.index("--t-forecaster") + 1] == "learned" and a[a.index("--t-modes") + 1] == "6"
    assert b[b.index("--t-forecaster") + 1] == "ca" and b[b.index("--t-modes") + 1] == "1"
    assert d[d.index("--t-forecaster") + 1] == "oracle_gt" and d[d.index("--t-modes") + 1] == "1"
    assert d[d.index("--run-name") + 1].endswith("_PRIV")
    # a and b differ only in run name, forecaster, modes and a's pre-trained forecaster (--t-fc-init)
    assert a[a.index("--t-fc-init") + 1].endswith("/v3fc_pre_learned6/forecaster.pt") and "--t-fc-init" not in b
    a2 = a[:a.index("--t-fc-init")] + a[a.index("--t-fc-init") + 2:]
    da = {i for i, (u, v) in enumerate(zip(a2, b)) if u != v}
    assert len(a2) == len(b) and {a2[i - 1] for i in da} == {"--run-name", "--t-forecaster", "--t-modes"}
    for x in (a, b, d):
        assert x[x.index("--t-fc-train") + 1] == "frozen"
    p = AR.cli("pre")
    assert p[:2] == ["--out", AR.load("pre")["out"]] and p[-1] == "--resume"
    with pytest.raises(ValueError):
        AR.cli("c")
    assert AR.temporal_cli({"on": False}) == ["--temporal", "0"]
    with pytest.raises(ValueError):
        AR.temporal_cli({"on": True, "bogus": 1})


def test_spec_recipe_flags():
    """The recipe carries every flag of SPEC 5.4 with its value."""
    r = AR.recipe()["cli"]
    want = {"--arm": "T", "--epochs": "10", "--tokens-per-batch": "16", "--lr": "6e-4", "--sur-groups": "near_mid",
            "--eval-every": "2", "--best-metric": "none", "--score-keys": "nc,ttc,ep,comfort", "--n-var": "80",
            "--var-name": "var_separate_sampler16_accstraight", "--seed": "0", "--sampler-seed": "0",
            "--norm-from": "compute", "--warmup-steps": "500", "--wd": "0.01", "--clip": "1.0", "--workers": "8",
            "--ep-target": "official"}
    got = {r[i]: r[i + 1] for i in range(0, len(r), 2)}
    assert got == want
    assert AR.recipe()["world_size"] == 2 and AR.recipe()["cuda_visible"] == "0,1"


@pytest.mark.skipif(not RUN_T10.is_file(), reason="ck2T10 config absent")
def test_recipe_equals_ck2T10_config():
    res = AR.recipe_check()
    assert res["mismatch"] == [], {k: v for k, v in res["rows"].items() if not v["eq"]}
    assert res["n_keys"] >= 36 and res["ep_target"] == "official"


def test_mutated_configs_refused(tmp_path):
    base = json.load(open(AR.ARMS_DIR / AR.FILES["a"]))

    def bad(mut, k="a"):
        d = json.loads(json.dumps(json.load(open(AR.ARMS_DIR / AR.FILES[k]))))
        mut(d)
        p = tmp_path / f"bad_{len(list(tmp_path.iterdir()))}.json"
        p.write_text(json.dumps(d))
        with pytest.raises((ValueError, NotImplementedError)):
            AR.validate(str(p))

    bad(lambda d: d["temporal"].update(forecaster="ca"))                        # a with the CA forecaster
    bad(lambda d: d["temporal"].update(n_modes=1))
    bad(lambda d: d["temporal"].update(source="gt"))
    bad(lambda d: d.update(name="v3T10_a_learned_PRIV"))                        # _PRIV without privilege
    bad(lambda d: d["temporal"].update(lambda_fc=0.5))                          # off-spec default
    bad(lambda d: d["temporal"].update(fc_backprop=True))
    bad(lambda d: d["temporal"].update(extra_key=1))                            # unknown TemporalCfg key
    bad(lambda d: d.update(gates=["G1"]))
    bad(lambda d: d.update(requires_user_approval=False))
    bad(lambda d: d.update(name="v3T10_d_oracle"), k="d")                       # privileged without _PRIV
    bad(lambda d: d.update(privileged=False), k="d")
    bad(lambda d: d["temporal"].update(on=True, source="bevfusion", forecaster="ca", n_modes=1), k="c")
    bad(lambda d: d["temporal"].update(fc_init=""))                             # frozen a without a forecaster
    bad(lambda d: d["temporal"].update(fc_init="/tmp/other/forecaster.pt"))     # not the stage-pre output
    bad(lambda d: d["temporal"].update(fc_train="joint"))                       # off the Step-1 default
    bad(lambda d: d["temporal"].update(fc_init="/x/forecaster.pt"), k="b")      # fc_init on the CA arm
    bad(lambda d: d["temporal"].update(pack_root="/home/external-user/ssd/yongjae_refiner/ck/v3/objpack"))  # K 64
    bad(lambda d: d["cli"].__setitem__(d["cli"].index("--vnoise") + 1, "0.0"), k="pre")   # aug != arm a
    bad(lambda d: d["cli"].__setitem__(d["cli"].index("--seed") + 1, "1"), k="pre")       # seed != recipe
    assert base["temporal"]["forecaster"] == "learned"


def test_cli_parses_with_trainer_parser():
    from tools.ck import train_ck2 as T2
    ap = T2.get_parser()
    if "--temporal" not in ap._option_string_actions:
        pytest.skip("train_ck2 has no --temporal flags yet (pipeline builder)")
    for k in "abd":
        a = ap.parse_args(AR.cli(k))
        assert a.run == AR.load(k)["name"]
