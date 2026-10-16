"""Sweep the inference field gate over margins, from ONE dump, with no GPU pass per margin.

``feas_to_score`` trains a sigmoid head on the corridor and ``plan_select_mode=corridor_first`` gates
on that head, so the field the model predicts is never read at selection.  ``field_gate_select``
reads it.  This applies that exact function -- imported, not reimplemented, or the sweep would be
measuring a different rule than the one that deploys -- to a dumped checkpoint at several margins and
writes one trajectory npz per margin for the two scorers.

    GPU=0 bash data/logs/dump_field.sh para_ssr_v2_r34_feas_gt          # once
    python tools/plan_v2/field_gate_sweep.py --exp para_ssr_v2_r34_feas_gt
    # then, for each margin printed:  bash data/logs/score_npz.sh <name> <npz>

A margin tag is ``<static>/<dynamic>``, either side ``off``:

    none        no gate at all -- the CONTROL, asserted to reproduce the checkpoint's own pick
    0.1/off     the drivable area alone (this is the DAC question, and the probe's optimum)
    off/0.0     the other agents alone (the NC question)
    0.4/0.2     both, at the margins the cost trains with

The two axes are kept apart because a single number cannot say which obstacle class the gate is
earning its points on, and the first version of this sweep left the dynamic side at margin 0 in
every row -- including the control, which then scored 87.71 against the checkpoint's 88.00.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))

from navsim.agents.para_ssr.feasibility import field_gate_select      # noqa: E402


class Cfg:
    """Only what the gate reads.  ``map_pc_range`` must match the training config or every sample
    lands in the wrong cell -- it is asserted against the dumped field's shape below."""
    map_pc_range = (-32.0, 0.0, -2.0, 32.0, 32.0, 2.0)
    feas_dyn_weight = 1.0
    plan_select_gate = 20.0
    feas_select_margin = 0.0
    feas_select_dyn_margin = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--margins", default="none,0.1/off,0.4/off,off/0.0,0.1/0.0,0.4/0.2")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    npz = ROOT / f"work_dirs/eval_epdms/field_{a.exp}.npz"
    z = np.load(npz, allow_pickle=True)
    print(f"{npz.name}: " + "  ".join(f"{k}{list(z[k].shape)}" for k in
          ("tokens", "trajectory", "trajectory_offset", "plan_final_rewards", "drivable_sdf_pred")
          if k in z.files))
    has_dyn = "agent_sdf_pred" in z.files
    print(f"동적 장 {'있음' if has_dyn else '없음'}")

    anchors = torch.from_numpy(np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy")).float()
    n = len(z["tokens"])
    off_all, rew_all = z["trajectory_offset"], z["plan_final_rewards"]
    sf_all = z["drivable_sdf_pred"]
    dyn_all = z["agent_sdf_pred"] if has_dyn else None
    Cfg.feas_dyn_weight = 1.0 if has_dyn else 0.0

    # geometry guard: the field must be on the grid the gate assumes
    h, w = sf_all.shape[1], sf_all.shape[2]
    assert (h, w) == (50, 100), f"예상 밖의 장 격자 {(h, w)}; map_pc_range 확인 필요"

    out_dir = ROOT / "work_dirs/eval_epdms/field_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for tag in a.margins.split(","):
        tag = tag.strip()
        deployed = tag == "none"
        # OFF is -1e9, not "skip": the min over a field whose margin is -1e9 can never be the
        # smaller of the two, so the other field decides alone.  This keeps one code path.
        ms, md = ("off", "off") if deployed else tag.split("/")
        Cfg.feas_select_margin = -1e9 if ms == "off" else float(ms)
        Cfg.feas_select_dyn_margin = -1e9 if md == "off" else float(md)
        picks, trajs, changed, none_ok, worst = [], [], 0, 0, []
        for s in range(0, n, a.batch):
            e = min(s + a.batch, n)
            p = {
                "trajectory_anchors": anchors,
                "trajectory_offset": torch.from_numpy(off_all[s:e]).float(),
                "drivable_sdf_pred": torch.from_numpy(sf_all[s:e]).float(),
                "plan_final_rewards": torch.from_numpy(rew_all[s:e]).float(),
                "plan_topk_index": torch.zeros(e - s, 1, dtype=torch.long),
            }
            if has_dyn:
                p["agent_sdf_pred"] = torch.from_numpy(dyn_all[s:e]).float()
            base = p["plan_final_rewards"].argmax(1)
            o = field_gate_select(p, Cfg)
            pick = o["plan_topk_index"][:, 0]
            cl = o["feas_select_clearance"]
            picks.append(pick.numpy())
            trajs.append(o["trajectory"].numpy())
            changed += int((pick != base).sum())
            none_ok += int((cl >= 0).any(dim=-1).logical_not().sum())
            worst.append(cl.gather(1, pick[:, None])[:, 0].numpy())
        worst = np.concatenate(worst)
        if deployed:
            # hard control: with no gate the selection must be the checkpoint's own, candidate for
            # candidate.  Anything else means the dump, the anchors or the gate is wrong.
            same = (np.concatenate(picks) == z["plan_topk_index"][:, 0]).mean()
            assert changed == 0 and same == 1.0, (
                f"대조군 불일치: 선택 변경 {changed}, 덤프 index 와 일치 {100 * same:.2f}%")
            print("  대조군 OK: 게이트 없음 = 체크포인트의 선택과 후보 단위로 동일")
        short = "deployed" if deployed else f"s{ms}_d{md}"
        name = f"gate_{a.exp.replace('para_ssr_v2_r34_', '')}_{short}"
        f = out_dir / f"{name}.npz"
        np.savez_compressed(
            f, tokens=z["tokens"], trajectory=np.concatenate(trajs).astype(np.float32),
            ego_fut_preds=np.zeros((n, 4, 8, 3), np.float32), command=np.zeros((n, 4), np.float32),
            missing=np.zeros((0,), np.float64), checkpoint=np.array(f"field_gate m={tag}"),
            picked_index=np.concatenate(picks), picked_clearance=worst)
        print(f"  {tag:>10s}   선택 변경 {changed:5d} ({100 * changed / n:5.2f}%)   "
              f"전원 탈락 {none_ok:4d} ({100 * none_ok / n:4.2f}%)   "
              f"선택 후보의 최소 여유 median {np.median(worst):+.3f}   -> {f.name}")
        made.append((name, f))

    print("\n채점 (PDMS 와 EPDMS 둘 다):")
    for name, f in made:
        print(f"  bash data/logs/score_npz.sh {name} {f}")


if __name__ == "__main__":
    main()
