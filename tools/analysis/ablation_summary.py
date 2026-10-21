import pickle
import numpy as np
import pandas as pd

E = "work_dirs/eval"
ref = pd.read_csv(f"{E}/r41_v2b_ep29_raw_vs_tracked.csv")
ref = ref[ref.valid == True].set_index("token")  # noqa: E712
reft = pickle.load(open(f"{E}/r41_v2b_ep29_navtest_trajectories.pkl", "rb"))["trajectories"]
rng = np.random.default_rng(0)

hdr = f"{'variant':12s} {'PDMS':>6s} {'dPDMS':>6s} {'95% CI':>17s} {'NC':>6s} {'DAC':>6s} {'EP':>6s} {'TTC':>6s} {'C':>6s} {'zero':>5s} {'|dxy_end| m':>11s}"
print(hdr)
print(f"{'full':12s} {ref.score.mean()*100:6.2f} {0:+6.2f} {'':>17s} {ref.nc.mean()*100:6.2f} {ref.dac.mean()*100:6.2f} "
      f"{ref.ep.mean()*100:6.2f} {ref.ttc.mean()*100:6.2f} {ref.comfort.mean()*100:6.2f} {int((ref.score==0).sum()):5d}")
for v in ["det", "map", "det_map", "bev"]:
    d = pd.read_csv(f"{E}/ablation/no_{v}.csv")
    d = d[d.valid == True].set_index("token")  # noqa: E712
    j = ref.join(d, rsuffix="_v", how="inner")
    diff = (j.score_v - j.score).values
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
    t = pickle.load(open(f"{E}/ablation/no_{v}.pkl", "rb"))["trajectories"]
    dxy = np.array([np.linalg.norm(t[k][-1, :2] - reft[k][-1, :2]) for k in j.index])
    print(f"{'no_'+v:12s} {d.score.mean()*100:6.2f} {diff.mean()*100:+6.2f} [{np.percentile(bs,2.5)*100:+6.2f},{np.percentile(bs,97.5)*100:+6.2f}] "
          f"{d.nc.mean()*100:6.2f} {d.dac.mean()*100:6.2f} {d.ep.mean()*100:6.2f} {d.ttc.mean()*100:6.2f} {d.comfort.mean()*100:6.2f} "
          f"{int((d.score==0).sum()):5d} {np.median(dxy):6.2f} (p90 {np.percentile(dxy,90):.2f})")
    worse = (j.score_v < j.score - 1e-6).sum()
    better = (j.score_v > j.score + 1e-6).sum()
    same = len(j) - worse - better
    print(f"{'':12s} scenes worse {worse}, better {better}, unchanged {same}; "
          f"new DAC=0 {int(((j.dac_v==0)&(j.dac==1)).sum())}, new NC<1 {int(((j.nc_v<1)&(j.nc==1)).sum())}, new TTC=0 {int(((j.ttc_v==0)&(j.ttc==1)).sum())}")
