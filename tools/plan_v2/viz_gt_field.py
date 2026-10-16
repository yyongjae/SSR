"""Draw the ground-truth corridor field next to the scene it came from, and PROVE the alignment.

The field is built by the same `DrivableFieldTargetBuilder` the training uses, from the same
`SceneLoader`, so what is drawn is literally the target arm B optimises against.  The last time a
field and a scene were drawn together the vectors were in SSR axes and the trajectories in NAVSIM
axes, and the picture was wrong in a way that was only visible because the drivable area did not
line up.  So everything here is drawn in ONE frame and three checks are printed:

    C1  sign of the field vs shapely's point-in-polygon, per BEV cell
    C2  sign of the field at the human trajectory's footprint corners vs point-in-polygon
    C3  the anchors ranked by the field's own cost vs the official per-anchor DAC label (AUC)

C1/C2 catch a transform error.  C3 catches a field that is aligned but useless.  A deliberately
flipped control is printed beside each, because "97% agreement" means nothing until you know what a
wrong field scores.

FRAME.  Everything is plotted in the field's OWN axes, so the field image needs no transform at all:

    horizontal = SSR x = -navsim y   (vehicle right positive), pc_range[0]..pc_range[3]
    vertical   = SSR y =  navsim x   (forward positive),       pc_range[1]..pc_range[4]

Vectors (polygons, boxes, trajectories) are NAVSIM and are converted once, by the single helper
`nav2ssr`, which is the same mapping as para_ssr_targets.py:583.

    python tools/plan_v2/viz_gt_field.py --scenes 4 --out data/analysis/gt_field
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/readout"))
import _env  # noqa: F401,E402
from _env import scene_filter, split_dirs  # noqa: E402

EGO_FRONT, EGO_REAR, EGO_HALFW = 4.049, 1.127, 1.1485       # nuPlan Pacifica, rear-axle origin


def nav2ssr(p):
    """NAVSIM ego (x forward, y left) -> SSR BEV (x right, y forward).  [..., 2] -> [..., 2]."""
    p = np.asarray(p, np.float64)
    return np.stack([-p[..., 1], p[..., 0]], axis=-1)


def corners_nav(pose):
    """rear-axle pose [x, y, heading] -> 4 footprint corners in NAVSIM ego axes."""
    x, y, h = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = np.cos(h), np.sin(h)
    out = []
    for dx, dy in ((EGO_FRONT, EGO_HALFW), (EGO_FRONT, -EGO_HALFW),
                   (-EGO_REAR, -EGO_HALFW), (-EGO_REAR, EGO_HALFW)):
        out.append([x + c * dx - s * dy, y + s * dx + c * dy])
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--out", default="data/analysis/gt_field")
    ap.add_argument("--dyn-steps", type=int, default=3, help="how many dynamic planes to draw")
    ap.add_argument("--stats-only", action="store_true", help="no figures, population numbers only")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # the only CJK family installed here; without it every Korean label renders as a box
    matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import torch
    from shapely.geometry import Point

    from navsim.agents.para_ssr.configs.default import ParaSSRConfig
    from navsim.agents.para_ssr.feasibility import (DrivableFieldTargetBuilder, sample_field,
                                                    cost_from_sdf, target_shape)
    from navsim.agents.para_ssr.plan_map import DrivableAreaTargetBuilder
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    cfg = ParaSSRConfig(feas_weight=0.15, feas_dyn_weight=1.0)
    pc = cfg.map_pc_range
    x0, x1, y0, y1 = float(pc[0]), float(pc[3]), float(pc[1]), float(pc[4])
    H, W = target_shape(cfg)                     # target grid (4x finer than the BEV)
    up = H // cfg.bev_h
    logs_dir, blobs_dir = split_dirs("trainval" if args.split == "navtrain" else "test")
    no_sensors = SensorConfig(**{k: False for k in (
        "cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2",
        "cam_b0", "lidar_pc")})       # the field needs annotations and the map, no images
    loader = SceneLoader(logs_dir, blobs_dir, scene_filter(args.split),
                         sensor_config=no_sensors)
    tokens = sorted(loader.tokens)

    z = np.load(ROOT / "data/planning_vb/navtest_refined_scores.npz", allow_pickle=True)
    ci = [str(c) for c in z["columns"]].index("drivable_area_compliance")
    dac_at = {str(t): i for i, t in enumerate(z["tokens"])}
    dac_all = np.asarray(z["scores"])[:, :256, ci]
    anchors = np.load(ROOT / "data/planning_vb/trajectory_anchors_256.npy").astype(np.float32)

    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    builder = DrivableFieldTargetBuilder(cfg)
    rng = np.random.default_rng(0)
    picked = [tokens[i] for i in rng.permutation(len(tokens))[: args.scenes * 4]]

    done, c1s, c2s, c3s, c4s, c5s, c6s = 0, [], [], [], [], [], []
    for tok in picked:
        if done >= args.scenes:
            break
        scene = loader.get_scene_from_token(tok)
        t = builder.compute_targets(scene)
        sdf = t["drivable_sdf_bev"].float().numpy()                     # [H, W]
        dyn = t["agent_sdf_bev"].float().numpy() if "agent_sdf_bev" in t else None
        cur = scene.scene_metadata.num_history_frames - 1
        poly = DrivableAreaTargetBuilder(cfg).drivable_polygon(scene, cur)
        if poly is None or poly.is_empty:
            continue

        # ---------- C1: field sign vs point-in-polygon, per cell ----------
        gy = y0 + (np.arange(H) + 0.5) * (y1 - y0) / H          # SSR y  (= navsim x)
        gx = x0 + (np.arange(W) + 0.5) * (x1 - x0) / W          # SSR x  (= -navsim y)
        GY, GX = np.meshgrid(gy, gx, indexing="ij")
        import shapely
        shapely.prepare(poly)
        inside = shapely.contains_xy(poly, GY.ravel(), (-GX).ravel()).reshape(H, W)
        c1 = float((inside == (sdf > 0)).mean())
        c1_flip = float((inside == (sdf.T[:H, :W] > 0)).mean()) if H == W else \
            float((inside == (sdf[:, ::-1] > 0)).mean())        # lateral mirror control
        c1s.append((c1, c1_flip))

        # ---------- C2: field at the human footprint corners ----------
        human = np.asarray(scene.get_future_trajectory().poses, np.float64)      # [T, 3] NAVSIM
        cn = np.concatenate([corners_nav(p) for p in human], axis=0)             # [T*4, 2] NAVSIM
        f = torch.as_tensor(sdf)[None]
        got = sample_field(f, torch.as_tensor(cn, dtype=torch.float32)[None], pc)[0].numpy()
        ref = np.array([poly.contains(Point(p[0], p[1])) for p in cn])
        c2 = float((ref == (got > 0)).mean())
        got_flip = sample_field(f, torch.as_tensor(cn[:, ::-1].copy(),
                                                   dtype=torch.float32)[None], pc)[0].numpy()
        c2s.append((c2, float((ref == (got_flip > 0)).mean())))

        # ---------- C6: is the HUMAN actually inside our field?  The GT trajectory is what the
        # offset head imitates, so if the field calls it infeasible the two supervisions fight.
        # Reported at margin 0 and at the cost's margin, because the cost pushes to +0.4 m.
        c6_0 = float((got > 0).mean())
        c6_m = float((got > 0.4).mean())
        c6s.append((c6_0, c6_m, float(np.percentile(got, 5))))

        # ---------- C3: anchors ranked by the field's cost vs the official label ----------
        c3 = float("nan")
        if tok in dac_at:
            d = dac_all[dac_at[tok]]
            if 0 < d.sum() < len(d):
                ac = np.concatenate([[corners_nav(p) for p in a] for a in anchors])  # [K*T, 4, 2]
                pts = torch.as_tensor(ac.reshape(1, -1, 2), dtype=torch.float32)
                s = sample_field(f, pts, pc)[0].numpy().reshape(len(anchors), anchors.shape[1], 4)
                c = cost_from_sdf(torch.as_tensor(s)[None], 0.4, 0.2)[0].numpy()
                gapm = c[d <= 0.5][:, None] - c[d > 0.5][None, :]
                c3 = float((gapm > 0).mean() + 0.5 * (gapm == 0).mean())
        c3s.append(c3)

        # ---------- C4: does dyn[t] show the agents of frame cur+1+t, in the CURRENT ego frame?
        # An off-by-one in time, or forgetting the ego_poses transform, both leave the field
        # aligned to the wrong boxes.  Score = fraction of each box's footprint where dyn < 0,
        # against the same measurement using a NEIGHBOURING step as the control.
        def box_occupancy(plane, frame_idx):
            fr = scene.frames[frame_idx] if 0 <= frame_idx < len(scene.frames) else None
            if fr is None:
                return float("nan")
            import shapely
            from shapely.geometry import Polygon as _P
            # future annotations live in THEIR OWN frame; bring them to the current ego frame the
            # same way agent_free_sdf does, via the recorded ego poses
            e0 = np.asarray(scene.frames[cur].ego_status.ego_pose, np.float64)
            ek = np.asarray(fr.ego_status.ego_pose, np.float64)
            dth = ek[2] - e0[2]
            c_, s_ = np.cos(e0[2]), np.sin(e0[2])
            dxy = (ek[:2] - e0[:2]) @ np.array([[c_, -s_], [s_, c_]])
            polys = []
            for a in fr.annotations.boxes:
                bx, by, bh, bl, bw = float(a[0]), float(a[1]), float(a[6]), float(a[3]), float(a[4])
                # local = R(dth) @ b + dxy;  as a row vector that is b @ R(dth).T = b @ M
                M = np.array([[np.cos(dth), np.sin(dth)], [-np.sin(dth), np.cos(dth)]])
                bx, by = (np.array([bx, by]) @ M) + dxy
                bh = bh + dth
                cc, ss = np.cos(bh), np.sin(bh)
                bo = np.array([[bl / 2, bw / 2], [bl / 2, -bw / 2],
                               [-bl / 2, -bw / 2], [-bl / 2, bw / 2]])
                polys.append(_P(bo @ np.array([[cc, ss], [-ss, cc]]) + [bx, by]))
            if not polys:
                return float("nan")
            u = shapely.union_all(polys)
            shapely.prepare(u)
            hit = shapely.contains_xy(u, GY.ravel(), (-GX).ravel()).reshape(H, W)
            return float((plane[hit] < 0).mean()) if hit.any() else float("nan")

        if dyn is not None:
            aligned = [box_occupancy(dyn[j], cur + 1 + j) for j in range(dyn.shape[0])]
            offby1 = [box_occupancy(dyn[j], cur + 2 + j) for j in range(dyn.shape[0])]
            c4 = float(np.nanmean(aligned)); c4c = float(np.nanmean(offby1))
            # C5: the human trajectory is the ground truth, so it must be collision-free:
            # dyn[t] at its footprint at step t must be positive
            free = []
            for j in range(min(dyn.shape[0], len(human))):
                q = torch.as_tensor(corners_nav(human[j]), dtype=torch.float32)[None]
                free.append(float((sample_field(torch.as_tensor(dyn[j])[None], q, pc)[0]
                                   .numpy() > 0).mean()))
            c5 = float(np.mean(free))
        else:
            c4 = c4c = c5 = float("nan")
        c4s.append((c4, c4c)); c5s.append(c5)

        if args.stats_only:
            done += 1
            if done % 20 == 0:
                print(f'  {done}/{args.scenes}', flush=True)
            continue

        # ---------- draw ----------
        # 2 x 3 so each panel is actually readable; the dynamic planes sample the horizon rather
        # than showing the first three, because the interesting divergence is late.
        dsteps = [0, 3, 7] if dyn is not None else []
        dsteps = [d for d in dsteps if dyn is not None and d < dyn.shape[0]]
        fig, axes = plt.subplots(2, 3, figsize=(19.5, 12.8))
        xe = np.linspace(x0, x1, W + 1)
        ye = np.linspace(y0, y1, H + 1)
        geoms = getattr(poly, "geoms", [poly])
        hs = nav2ssr(human[:, :2])

        def frame(ax, title):
            ax.set_title(title, fontsize=11)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
            ax.set_xlabel("SSR x = 차량 우측 (m)"); ax.set_ylabel("SSR y = 전방 (m)")
            ax.plot(0, 0, "k+", ms=11, mew=2)

        def draw_boxes(ax, frame_idx, color):
            """GT boxes of a frame, brought to the CURRENT ego frame exactly as agent_free_sdf does."""
            if not (0 <= frame_idx < len(scene.frames)):
                return
            cp = np.asarray(scene.frames[cur].ego_status.ego_pose, np.float64)
            fp = np.asarray(scene.frames[frame_idx].ego_status.ego_pose, np.float64)
            c0, s0 = np.cos(cp[2]), np.sin(cp[2])
            fc, fs = np.cos(fp[2]), np.sin(fp[2])
            for b in np.asarray(scene.frames[frame_idx].annotations.boxes, np.float64):
                gx_, gy_ = fc * b[0] - fs * b[1] + fp[0], fs * b[0] + fc * b[1] + fp[1]
                dx_, dy_ = gx_ - cp[0], gy_ - cp[1]
                lx, ly = c0 * dx_ + s0 * dy_, -s0 * dx_ + c0 * dy_
                yaw = float(b[6]) + fp[2] - cp[2]
                hl, hw_ = 0.5 * float(b[3]), 0.5 * float(b[4])
                cc, ss = np.cos(yaw), np.sin(yaw)
                co = np.array([[hl, hw_], [hl, -hw_], [-hl, -hw_], [-hl, hw_]])
                q = nav2ssr(co @ np.array([[cc, ss], [-ss, cc]]) + [lx, ly])
                ax.plot(np.r_[q[:, 0], q[0, 0]], np.r_[q[:, 1], q[0, 1]], color=color, lw=1.3, zorder=3)

        # (0,0) the scene, vectors only
        ax = axes[0, 0]
        frame(ax, f"{tok[:12]}   scene (벡터만)")
        # fill with HOLES: the drivable polygon has interior rings (medians, islands), and filling
        # only the exterior paints them drivable -- which is what made the first draft disagree with
        # a field that C1 proves is exact.
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MPath
        verts, codes = [], []
        for g in geoms:
            for ring in [g.exterior, *g.interiors]:
                q = nav2ssr(np.asarray(ring.coords))
                verts.extend(q.tolist())
                codes.extend([MPath.MOVETO] + [MPath.LINETO] * (len(q) - 2) + [MPath.CLOSEPOLY])
                ax.plot(q[:, 0], q[:, 1], color="0.45", lw=0.8, zorder=1)
        if verts:
            ax.add_patch(PathPatch(MPath(verts, codes), facecolor="0.88", edgecolor="none",
                                   zorder=0))
        draw_boxes(ax, cur, "tab:orange")
        for a in anchors[::8]:
            q = nav2ssr(a[:, :2])
            ax.plot(q[:, 0], q[:, 1], color="tab:blue", lw=0.4, alpha=0.45, zorder=2)
        ax.plot(hs[:, 0], hs[:, 1], "-o", color="tab:red", ms=4, lw=2.0, zorder=5, label="human GT")
        ax.plot([], [], color="tab:orange", lw=1.3, label="GT 박스 (t=0)")
        ax.plot([], [], color="tab:blue", lw=1.0, alpha=.6, label="앵커 (32개만)")
        ax.legend(loc="upper left", fontsize=9)

        # (0,1) the static field on its own grid
        ax = axes[0, 1]
        m = ax.pcolormesh(xe, ye, sdf, cmap="RdBu", vmin=-6, vmax=6, shading="flat")
        fig.colorbar(m, ax=ax, fraction=0.04, label="signed distance (m),  + = 주행 가능")
        ax.contour(gx, gy, sdf, levels=[0.0], colors="k", linewidths=1.4)
        for g in geoms:
            for ring in [g.exterior, *g.interiors]:
                q = nav2ssr(np.asarray(ring.coords))
                ax.plot(q[:, 0], q[:, 1], color="lime", lw=1.0, ls="--", zorder=3)
        ax.plot(hs[:, 0], hs[:, 1], "-o", color="k", ms=3, lw=1.4, zorder=4)
        frame(ax, f"정적 장  Phi_static   {H}x{W} ({up}x 업샘플, {(x1-x0)/W:.2f} m/셀)\n"
                  f"검정 = 장의 0등고선,  초록 점선 = 폴리곤  → C1 {c1:.3f}")

        # (0,2) the static field, but only what the cost can feel (margin 0.4, beta 0.2)
        ax = axes[0, 2]
        sens = 1.0 / (1.0 + np.exp((sdf - 0.4) / 0.2))          # -dc/dPhi at margin 0.4
        m = ax.pcolormesh(xe, ye, sens, cmap="magma", vmin=0, vmax=1, shading="flat")
        fig.colorbar(m, ax=ax, fraction=0.04, label="|dc/dPhi| = sigmoid((margin-Phi)/beta)")
        ax.plot(hs[:, 0], hs[:, 1], "-o", color="tab:cyan", ms=3, lw=1.4, zorder=4)
        frame(ax, "비용이 실제로 느끼는 영역 (margin 0.4 m, beta 0.2)\n"
                  "밝을수록 기울기가 살아 있는 곳 — 경계에서 약 1 m 띠")

        # (1, *) the dynamic field
        for k, j_ in enumerate(dsteps):
            ax = axes[1, k]
            m = ax.pcolormesh(xe, ye, dyn[j_], cmap="PuOr_r", vmin=-6, vmax=6, shading="flat")
            fig.colorbar(m, ax=ax, fraction=0.04, label="signed distance (m),  + = 비어 있음")
            ax.contour(gx, gy, dyn[j_], levels=[0.0], colors="k", linewidths=1.0)
            draw_boxes(ax, cur + 1 + j_, "tab:green")
            if j_ < len(human):
                q = nav2ssr(corners_nav(human[j_]))
                ax.plot(np.r_[q[:, 0], q[0, 0]], np.r_[q[:, 1], q[0, 1]],
                        color="tab:red", lw=2.0, zorder=5)
            frame(ax, f"동적 장  Phi_dyn[t={j_}]  ({0.5 * (j_ + 1):.1f} s 뒤)\n"
                      f"초록 = 그 시각 GT 박스(현 ego 프레임),  빨강 = 그 시각 human 발자국")
        for k in range(len(dsteps), 3):
            axes[1, k].axis("off")

        fig.suptitle(f"{tok}    C1 셀부호 {c1:.3f} (반전대조 {c1s[-1][1]:.3f})   "
                     f"C2 발자국부호 {c2:.3f} (축교환대조 {c2s[-1][1]:.3f})   "
                     f"C3 앵커 AUC {c3:.3f}   "
                     f"C4 박스피복 {c4:.3f} (한스텝어긋냄 {c4c:.3f})   C5 human자유 {c5:.3f}",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(outdir / f"{done:02d}_{tok[:12]}.png", dpi=105)
        plt.close(fig)
        done += 1
        print(f"  [{done}] {tok}  C1 {c1:.3f}/{c1s[-1][1]:.3f}  C2 {c2:.3f}/{c2s[-1][1]:.3f}  "
              f"C3 {c3:.3f}  C4 {c4:.3f}/{c4c:.3f}  C5 {c5:.3f}", flush=True)

    print(f"\n{done} 장면.  검증 요약 (값 / 틀린 장 대조)")
    print(f"  C1 셀 부호 일치        {np.mean([a for a, _ in c1s]):.4f}  /  "
          f"{np.mean([b for _, b in c1s]):.4f}")
    print(f"  C2 발자국 부호 일치    {np.mean([a for a, _ in c2s]):.4f}  /  "
          f"{np.mean([b for _, b in c2s]):.4f}")
    print(f"  C3 앵커 분별 AUC       {np.nanmean(c3s):.4f}   (전 구간 측정치 0.937과 비교)")
    print(f"  C4 동적장이 그 시각 박스를 덮는 비율  {np.nanmean([a for a, _ in c4s]):.4f}  /  "
          f"{np.nanmean([b for _, b in c4s]):.4f}  (한 스텝 어긋낸 대조)")
    print(f"  C5 human 발자국이 동적장에서 자유     {np.nanmean(c5s):.4f}  (GT는 무충돌이므로 1에 가까워야)")
    print(f"  C6 human 발자국이 정적장에서 Phi>0     {np.mean([a for a, _, _ in c6s]):.4f}")
    print(f"     같은 것, 비용의 margin 0.4 m 기준   {np.mean([b for _, b, _ in c6s]):.4f}"
          f"   <- 1보다 크게 작으면 corridor가 human을 벌준다는 뜻")
    print(f"     human 발자국 Phi의 5퍼센타일 중앙   {np.median([c for _, _, c in c6s]):+.3f} m")
    print(f"  -> {outdir}")


if __name__ == "__main__":
    main()
