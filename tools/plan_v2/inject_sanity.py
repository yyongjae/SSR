"""Does the injection reach the model, and does the scorer react to positions at all?

A null result from a broken intervention is worthless, so before reading anything into
``perception_inject.py`` three things have to hold:

  reached      the edited task memories really differ from the originals, at the edited slots only
  reacts       when EVERY detection position is randomised (latents untouched), the scores move.
               If they do not, the scorer ignores the position channel and the injection experiment
               can never show anything -- the null would be about the architecture, not perception.
  saturates    when the front of the ego is filled with injected objects, NC drops.

    python tools/plan_v2/inject_sanity.py --ckpt <ckpt> [--limit 100] <agent.config overrides...>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "readout"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: F401,E402
from _env import scene_filter, split_dirs  # noqa: E402
from cache_student_bev import build_agent  # noqa: E402
from eval_motion import SceneData, collate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--bb", default="34")
    args, overrides = ap.parse_known_args()

    from navsim.common.dataloader import SceneLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = build_agent(args.ckpt, overrides + [f"agent.config.image_architecture=resnet{args.bb}.tv_in1k"]).to(device)
    model = agent.para_ssr_model
    head = model.pts_bbox_head
    x0, x1, y0, y1 = float(head.pc_range[0]), float(head.pc_range[3]), float(head.pc_range[1]), float(head.pc_range[4])

    captured, orig_plan, orig_prep = {}, head.plan_from_bev, head.prepare_task_memories
    mode = {}
    seen = {}

    def capture(*a, **k):
        captured["args"], captured["kwargs"] = a, k
        return orig_plan(*a, **k)

    def prep(det_out, map_out):
        mem = orig_prep(det_out, map_out)
        seen["base_pos"] = mem["det_position"].clone()
        seen["base_cf"] = mem["det_confidence"].clone()
        what = mode.get("what")
        if what == "shuffle":                       # same latents, permuted positions
            idx = torch.randperm(mem["det_position"].shape[1], device=mem["det_position"].device)
            mem["det_position"] = mem["det_position"][:, idx]
        elif what == "random":                      # same latents, random positions in the BEV
            r = torch.rand_like(seen["base_pos"][..., :1]).expand(-1, -1, 2).clone().uniform_(0, 1)
            mem["det_position"] = head.det_position_encoder(r)
        elif what == "zero_latent":       # 잠재를 key·value 양쪽에서 제거 (공정하지 않은 비교)
            mem["det_memory"] = torch.zeros_like(mem["det_memory"])
        elif what == "zero_latent_key":
            pass          # 어텐션 래퍼에서 key 만 손본다 (key = value + pos + conf 이므로 key-value)
        elif what == "zero_pos":          # 위치만 끄고 잠재는 그대로
            mem["det_position"] = torch.zeros_like(mem["det_position"])
        elif what == "fill":                        # objects everywhere in front of the ego
            n = mem["det_position"].shape[1]
            gx, gy = np.meshgrid(np.linspace(-8, 8, 8), np.linspace(2, 40, max(n // 8, 1)))
            pts = np.stack([gx.ravel(), gy.ravel()], -1)[:n]
            nx = (pts - np.array([x0, y0])) / np.array([x1 - x0, y1 - y0])
            p = torch.as_tensor(nx, dtype=torch.float32, device=mem["det_position"].device)[None]
            mem["det_position"][:, : p.shape[1]] = head.det_position_encoder(p)
            mem["det_confidence"][:] = head.det_confidence_encoder(
                torch.ones(1, mem["det_confidence"].shape[1], 1, device=mem["det_confidence"].device))
        seen["edit_pos"] = mem["det_position"].clone()
        return mem

    head.plan_from_bev, head.prepare_task_memories = capture, prep

    # key 에서 잠재만 빼기: key = value + position + confidence 이므로 key - value 가 정확히 나머지다
    for layer in head.plan_layers if hasattr(head, "plan_layers") else []:
        pass

    def wrap_attn(mod):
        orig = mod.forward

        def fwd(query, key, value, *a, **k):
            if mode.get("what") == "zero_latent_key":
                key = key - value
            return orig(query, key, value, *a, **k)

        mod.forward = fwd
        return orig

    attn_mods = [m for n, m in head.named_modules() if n.endswith("plan_det_cross_attn")]
    originals = [wrap_attn(m) for m in attn_mods]
    print(f"det cross-attn 래핑 {len(attn_mods)}개", flush=True)

    def run(what=None):
        mode.clear()
        if what:
            mode["what"] = what
        with torch.no_grad():
            out = orig_plan(*captured["args"], **captured["kwargs"])
        return (out["sim_rewards"][0].float().cpu().numpy(),
                int(out["plan_final_rewards"][0].float().argmax()),
                float((seen["edit_pos"] - seen["base_pos"]).abs().mean()))

    logs_dir, blobs_dir = split_dirs("test")
    loader = SceneLoader(logs_dir, blobs_dir, scene_filter("navtest"), sensor_config=agent.get_sensor_config())
    tokens = sorted(loader.tokens)[: args.limit]
    dl = DataLoader(SceneData(tokens, loader, agent.get_feature_builders(), agent.get_target_builders()),
                    batch_size=1, num_workers=3, collate_fn=collate)
    res = {k: [] for k in ("shuffle", "random", "fill", "zero_pos", "zero_latent", "zero_latent_key")}
    touched = {k: [] for k in res}
    n = 0
    for toks, feats, targets in dl:
        if feats is None:
            continue
        feats = {k: v.to(device) for k, v in feats.items()}
        with torch.no_grad():
            model(feats, run_aux=False)
        base, base_pick, _ = run(None)
        for what in res:
            s, pick, moved = run(what)
            res[what].append(np.append(np.abs(s - base).mean(axis=1), float(pick != base_pick)))
            touched[what].append(moved)
        n += 1
    head.plan_from_bev, head.prepare_task_memories = orig_plan, orig_prep

    print(f"\n{n} 장면. det_position 을 바꿨을 때 예측 점수의 평균 |변화|")
    print(f"{'조작':16s} {'NC':>8s} {'DAC':>8s} {'EP':>8s} {'TTC':>8s} {'COMF':>8s} {'선택변경':>9s}")
    for what, v in res.items():
        a = np.stack(v).mean(0)
        print(f"{what:16s} " + " ".join(f"{x:8.4f}" for x in a[:5]) + f" {100 * a[5]:8.1f}%")
    print("\n해석: memory 변화가 0 이면 주입이 닿지 않은 것이고, 닿았는데 점수 변화가 0 이면")
    print("      채점기가 위치 채널을 읽지 않는다는 뜻이다. 후자라면 '못 본 물체를 넣어주는' 실험은")
    print("      위치만으로는 결론을 낼 수 없다.")


if __name__ == "__main__":
    main()
