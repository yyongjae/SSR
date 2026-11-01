"""Verification of human/navtest.npz (extract_human.py validate, navtest parts):
 (3) ALL navtest tokens vs report/head_ablation_scenes/table.npz (human traj, speed, accel, command)  [validate uses n=50]
 (2) t0 pose / velocity / acceleration vs the official navtest metric cache ego_state (n random tokens)
 (1) traj / 5 s path vs navsim SceneLoader Scene.get_future_trajectory on the TEST logs (n random tokens, incl. gap)."""
import json, lzma, os, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
R = "/home/external-user/yongjae/SSR"
sys.path[:0] = [R, R + "/tools/refiner"]
os.environ.setdefault("NUPLAN_MAPS_ROOT", R + "/data/dataset/maps"); os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
import extract_human as EH
n = int(sys.argv[2]); outp = Path(sys.argv[1])
H = EH.load_human("navtest")
T = np.load(EH.NAVTEST_HUMAN, allow_pickle=True)
j = np.array([H["index"][t] for t in T["tokens"]])
res = {}
res["table_all"] = dict(n=len(j), human_valid=int(T["human_valid"].sum()),
    traj_xy_err=float(np.abs(H["traj"][j, :, :2] - T["human"][:, :, :2]).max()),
    traj_h_err=float(EH._hdiff(H["traj"][j, :, 2], T["human"][:, :, 2]).max()),
    v0_err=float(np.abs(H["v0"][j] - T["speed"]).max()), a0_err=float(np.abs(H["a0"][j] - T["accel"]).max()),
    cmd_mismatch=int((H["cmd"][j].astype(int) != T["command"].astype(int)).sum()))
rng = np.random.default_rng(0)
gap = np.flatnonzero(H["frame_gap"]); ok = np.flatnonzero(~H["frame_gap"])
pick = np.concatenate([rng.choice(gap, min(len(gap), n // 10), replace=False), rng.choice(ok, n - min(len(gap), n // 10), replace=False)])
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
rows = []
for i in pick:
    t, lg = str(H["tokens"][i]), str(H["logs"][i])
    p = Path(R) / "data/exp/metric_cache" / lg / "unknown" / t / "metric_cache.pkl"
    with lzma.open(p, "rb") as f:
        mc = pickle.load(f)
    es = mc.ego_state; ra = es.rear_axle; dcs = es.dynamic_car_state
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True, log_names=[lg], tokens=[t])
    sl = SceneLoader(data_path=EH.TEST_LOGS, sensor_blobs_path=None, scene_filter=sf, sensor_config=SensorConfig.build_no_sensors())
    sc = sl.get_scene_from_token(t)
    f8 = sc.get_future_trajectory(8).poses; f10 = sc.get_future_trajectory(10).poses
    rows.append(dict(token=t, frame_gap=bool(H["frame_gap"][i]),
        traj_xy_err=float(np.abs(H["traj"][i, :, :2] - f8[:, :2]).max()), traj_h_err=float(EH._hdiff(H["traj"][i, :, 2], f8[:, 2]).max()),
        path5_xy_err=float(np.abs(H["path"][i, :10, :2] - f10[:, :2]).max()), path5_h_err=float(EH._hdiff(H["path"][i, :10, 2], f10[:, 2]).max()),
        t0_pose_err=float(max(abs(ra.x - H["ego_global"][i, 0]), abs(ra.y - H["ego_global"][i, 1]), EH._hdiff(ra.heading, H["ego_global"][i, 2]))),
        t0_vel_err=float(max(abs(dcs.rear_axle_velocity_2d.x - H["eds"][i, 0]), abs(dcs.rear_axle_velocity_2d.y - H["eds"][i, 1]))),
        t0_acc_err=float(max(abs(dcs.rear_axle_acceleration_2d.x - H["eds"][i, 2]), abs(dcs.rear_axle_acceleration_2d.y - H["eds"][i, 3]))),
        v0_err=float(abs(dcs.speed - H["v0"][i])),
        scene_cmd_eq=bool(int(np.argmax(sc.frames[3].ego_status.driving_command)) == int(H["cmd"][i]))))
Rw = pd.DataFrame(rows)
num = [c for c in Rw.columns if c.endswith("_err")]
res["navtest_scene_and_mc"] = dict(n=len(Rw), n_frame_gap=int(Rw.frame_gap.sum()), max_abs=Rw[num].max().to_dict(),
                                   scene_cmd_all_equal=bool(Rw.scene_cmd_eq.all()))
res["counts"] = dict(n=int(len(H["tokens"])), frame_gap=int(H["frame_gap"].sum()), gap4=int(H["gap4"].sum()),
                     n_avail_ge16=int((H["n_avail"] >= 16).sum()), n_reg_ge16=int((H["n_reg"] >= 16).sum()),
                     n_reg_lt16_nongap=int(((H["n_reg"] < 16) & ~H["frame_gap"]).sum()))
outp.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
