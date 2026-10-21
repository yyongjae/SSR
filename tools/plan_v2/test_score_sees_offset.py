"""plan_score_sees_offset: the sim heads see the candidate's offset, and nothing else changes.

    O1  at initialisation the outputs are IDENTICAL to the same planner without the option (zero-
        initialised embedding) -- training starts exactly where v2 did
    O2  the score loss cannot move the trajectory: no gradient reaches the offset through the scores
    O3  the embedding is trained: it receives gradient from a score loss
    O4  once the embedding is non-zero, two candidates with the same query but different offsets get
        different scores.  Control: without the option they cannot
    O5  the imitation head ignores the offset

    python tools/plan_v2/test_score_sees_offset.py
"""
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, "/home/external-user/kyungmin/SSR-v2")
from navsim.agents.para_ssr.modules.anchor_planner import AnchorPlanner      # noqa: E402

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'통과' if cond else '실패 <<<'}  {name}  {detail}")


def main():
    torch.manual_seed(0)
    K, T, C, B = 16, 8, 64, 2
    f = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    np.save(f.name, np.random.randn(K, T, 3).astype(np.float32))
    base = AnchorPlanner(f.name, embed_dims=C, fut_ts=T, traj_dims=3, heading_from_xy=True)
    seen = AnchorPlanner(f.name, embed_dims=C, fut_ts=T, traj_dims=3, heading_from_xy=True, score_sees_offset=True)
    seen.load_state_dict(base.state_dict(), strict=False)          # same weights, plus the new embedding
    h = torch.randn(B, K, C)

    print("O1  초기 상태에서 옵션 없는 모델과 출력이 같음")
    a, b = base(h), seen(h)
    check("sim 동일", torch.allclose(a["sim_rewards"], b["sim_rewards"]))
    check("선택 동일", torch.equal(a["plan_topk_index"], b["plan_topk_index"]))

    print("O2/O3  점수 손실의 기울기")
    hh = h.clone().requires_grad_(True)
    out = seen(hh)
    off = out["trajectory_offset"]
    off.retain_grad()
    with torch.no_grad():                                           # make the embedding live first
        seen.offset_embed[-1].weight.normal_(0, 0.1)
    out = seen(hh)
    off = out["trajectory_offset"]; off.retain_grad()
    out["sim_rewards"].sum().backward()
    check("offset 으로 가는 기울기 없음 (detach)", off.grad is None or float(off.grad.abs().sum()) == 0.0)
    g = seen.offset_embed[0].weight.grad
    check("임베딩은 학습됨", g is not None and float(g.abs().sum()) > 0)

    print("O4  같은 쿼리, 다른 offset → 다른 점수 (대조군: 옵션 없음)")
    q = torch.randn(1, 1, C).expand(1, K, C).contiguous()           # every candidate: SAME query
    with torch.no_grad():
        s_seen = seen.score(q, torch.randn(1, K, T, 3))[1]
        s_base = base.score(q, torch.randn(1, K, T, 3))[1]
    check("옵션 있음: 후보마다 점수가 다름", float(s_seen.std(dim=-1).mean()) > 1e-4,
          f"후보 간 표준편차 {float(s_seen.std(dim=-1).mean()):.2e}")
    check("대조군: 옵션 없으면 전부 같음", float(s_base.std(dim=-1).mean()) < 1e-6)

    print("O5  모방 헤드는 offset 을 안 봄")
    with torch.no_grad():
        im1 = seen.score(q, torch.randn(1, K, T, 3))[0]
        im2 = seen.score(q, torch.randn(1, K, T, 3))[0]
    check("im 이 offset 과 무관", torch.allclose(im1, im2))

    print(f"\n{sum(OK)}/{len(OK)} 통과")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
