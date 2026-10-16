"""Does feasibility.field_priors put the perception heads' answers where the field reads them?

    P1  a detected car (navsim x 10 m ahead, 3 m left, heading 0, 4.5 x 2.0 m) moving 2 m per step
        forward: at step 0 (+0.5 s) the forecast occupancy is high at its forecast centre (12, 3) --
        read back through sample_field, the corridor's own lookup -- and low where it was (10, 3)
    P2  its footprint is oriented: 2.0 m ahead of the centre (inside the 2.25 m half length) is
        occupied, 1.6 m to the side (outside the 1.0 m half width) is not
    P3  two motion modes split the occupancy by mode probability (0.7 / 0.3)
    P4  a low-score detection contributes ~nothing; the seg channel is sigmoid(logits)
    P5  nothing carries gradient (priors are detached)
    H1  DrivableFieldHead with prior channels: shapes, and it rejects a missing prior
    W1  wiring: the model builds the head with prior channels and passes field_priors

    python tools/plan_v2/test_field_priors.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/external-user/kyungmin/SSR-v2")
sys.path.insert(0, str(ROOT))
from navsim.agents.para_ssr.configs.default import ParaSSRConfig                          # noqa: E402
from navsim.agents.para_ssr.feasibility import DrivableFieldHead, field_priors, sample_field  # noqa: E402
from navsim.agents.para_ssr.modules.losses import normalize_bbox                         # noqa: E402
from navsim.agents.para_ssr.para_ssr_targets import navsim_box_to_ssr                    # noqa: E402

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'통과' if cond else '실패 <<<'}  {name}  {detail}")


def det_for(boxes_nav, logits, trajs, mode_logits):
    """boxes_nav [N, 7] NAVSIM (x, y, z, l, w, h, heading); trajs [N, M, T, 2] SSR per-step offsets."""
    codes = torch.stack([normalize_bbox(torch.from_numpy(navsim_box_to_ssr(b, np.zeros(3)))) for b in boxes_nav])
    q = codes.shape[0]
    return {"all_bbox_preds": codes[None, None].float(),                       # [L=1, B=1, Q, 10]
            "all_cls_scores": torch.tensor(logits).float().view(1, 1, q, 1).expand(1, 1, q, 7).clone(),
            "traj_preds": torch.tensor(trajs).float()[None], "traj_cls_preds": torch.tensor(mode_logits).float()[None]}


def at(field, x_nav, y_nav, cfg):
    return float(sample_field(field[None], torch.tensor([[[x_nav, y_nav]]]), cfg.map_pc_range)[0, 0])


def main():
    cfg = ParaSSRConfig()
    H, W, T = cfg.bev_h, cfg.bev_w, 8
    car = np.array([10.0, 3.0, 0.0, 4.5, 2.0, 1.5, 0.0])
    fwd = np.zeros((1, 1, T, 2), np.float32); fwd[..., 1] = 2.0                # SSR y = forward
    det = det_for(car[None], [8.0], fwd, [[0.0]])
    pr = field_priors(det, None, cfg, H, W, T)
    print("P1  예측 위치에 점유")
    check("모양 [1, 8, H, W]", tuple(pr.shape) == (1, T, H, W), f"{tuple(pr.shape)}")
    o_new, o_old = at(pr[0, 0], 12.0, 3.0, cfg), at(pr[0, 0], 8.5, 3.0, cfg)
    check("step0: (12, 3) 높음, 원래 뒤쪽 (8.5, 3) 낮음", o_new > 0.8 and o_old < 0.2, f"{o_new:.2f} / {o_old:.2f}")
    check("step3 (+2 s): 중심 (18, 3)", at(pr[0, 3], 18.0, 3.0, cfg) > 0.8, f"{at(pr[0, 3], 18.0, 3.0, cfg):.2f}")

    print("P2  footprint 방향")
    a, s_ = at(pr[0, 0], 14.0, 3.0, cfg), at(pr[0, 0], 12.0, 1.4, cfg)
    check("길이 방향 2.0 m 앞 점유, 옆 1.6 m 비점유", a > 0.5 and s_ < 0.3, f"{a:.2f} / {s_:.2f}")

    print("P3  모드 확률로 나눔")
    two = np.zeros((1, 2, T, 2), np.float32); two[0, 0, :, 1] = 2.0; two[0, 1, :, 0] = -2.0   # fwd / turn left (SSR -x = left)
    det2 = det_for(car[None], [8.0], two, [[np.log(0.7), np.log(0.3)]])
    pr2 = field_priors(det2, None, cfg, H, W, T)
    pf, pl = at(pr2[0, 3], 18.0, 3.0, cfg), at(pr2[0, 3], 10.0, 11.0, cfg)
    check("앞 모드 ≈ 0.7, 왼쪽 모드 ≈ 0.3", abs(pf - 0.7) < 0.1 and abs(pl - 0.3) < 0.1, f"{pf:.2f} / {pl:.2f}")

    print("P4  낮은 점수 무시, seg 채널")
    det3 = det_for(car[None], [-8.0], fwd, [[0.0]])
    check("score sigmoid(-8) → 점유 < 0.01", float(field_priors(det3, None, cfg, H, W, T).max()) < 0.01)
    seg = torch.randn(1, H, W)
    pr4 = field_priors(det, seg, cfg, H, W, T)
    check("채널 0 = sigmoid(seg logits), 총 9채널", pr4.shape[1] == 1 + T and torch.allclose(pr4[:, 0], seg.sigmoid()))

    print("P5  기울기 없음")
    det5 = {k: v.clone().requires_grad_(True) for k, v in det.items()}
    check("priors.requires_grad == False", not field_priors(det5, seg.requires_grad_(True), cfg, H, W, T).requires_grad)

    print("H1  장 헤드")
    head = DrivableFieldHead(16, H, W, hidden=8, dyn_steps=T, prior_channels=1 + T)
    st, dy = head(torch.randn(2, H * W, 16), torch.rand(2, 1 + T, H, W))
    check("정적 [2,H,W], 동적 [2,8,H,W]", tuple(st.shape) == (2, H, W) and tuple(dy.shape) == (2, T, H, W))
    try:
        head(torch.randn(2, H * W, 16)); bad = False
    except ValueError:
        bad = True
    check("prior 없이 부르면 거부", bad)
    plain = DrivableFieldHead(16, H, W, hidden=8, dyn_steps=T)
    check("prior_channels=0 이면 기존과 같은 파라미터 모양", plain.trunk[0].weight.shape[1] == 16)

    print("W1  배선")
    src = (ROOT / "navsim/agents/para_ssr/para_ssr_model.py").read_text()
    check("모델: prior_channels + _field_priors(det_out, seg_logits)", "prior_channels=" in src
          and "self.drivable_field_head(bev_embed, self._field_priors(det_out, seg_logits))" in src)

    print(f"\n{sum(OK)}/{len(OK)} 통과")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
