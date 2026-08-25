# SSR BEV Representation Explorer

This directory records and visualizes the fixed inputs supplied on
2026-08-22 and updated with final PARA checkpoints on 2026-08-24:

1. PARA-SSR non-staging: `/data1/yong/SSR/work_dirs/para_ssr_60ep`
2. PARA-SSR Stage 1 det+map+motion: `/data1/yong/SSR/work_dirs/para_ssr_stage1`
3. PARA-SSR Stage 2 all-task: `/data1/yong/SSR/work_dirs/para_ssr_stage2`
4. SSR-noFFP planning work dir: `work_dirs/ssr_noffp_2gpu_b4`
5. BEVFusion teacher cache: `/data1/yong/teacher_cache/bevfusion_cache`
6. MapTRv2 teacher cache: `/data1/yong/teacher_cache/maptrv2_cache`

The exact checkpoints are pinned in `inputs.json`: non-staging epoch 60 raw,
Stage 1 epoch 48 raw, Stage 2 epoch 12 raw, and SSR-noFFP epoch 12 raw.

## Sampling and temporal correctness

`select_samples.py` chooses 16 scenes and 5 diverse frames per scene (80 target
samples), balanced across the four nuScenes locations and covering day/night,
dry/rain, command, object density, moving-agent density, ego motion and scene
phase. It is a qualitative, deliberately stratified subset—not an mAP sample.

Each SSR exporter processes every chronological frame from the beginning of a
selected scene through its last target. Only target frames are saved. This
preserves recurrent `prev_bev`; evaluating 80 isolated frames would not.

## Rebuild

Run from the SSR repository root with the `ssr` environment:

```bash
/home/yongjae/miniconda3/envs/ssr/bin/python tools/bev_visualizer/select_samples.py

CUDA_VISIBLE_DEVICES=0 /home/yongjae/miniconda3/envs/ssr/bin/python \
  tools/bev_visualizer/export_ssr_bev.py para_nonstaging --device cuda:0 --overwrite
CUDA_VISIBLE_DEVICES=1 /home/yongjae/miniconda3/envs/ssr/bin/python \
  tools/bev_visualizer/export_ssr_bev.py para_stage1 --device cuda:0 --overwrite
CUDA_VISIBLE_DEVICES=0 /home/yongjae/miniconda3/envs/ssr/bin/python \
  tools/bev_visualizer/export_ssr_bev.py para_stage2 --device cuda:0 --overwrite
CUDA_VISIBLE_DEVICES=0 /home/yongjae/miniconda3/envs/ssr/bin/python \
  tools/bev_visualizer/export_ssr_bev.py ssr_noffp --device cuda:0 --overwrite

/home/yongjae/miniconda3/envs/ssr/bin/python \
  tools/bev_visualizer/prepare_teacher_cache.py bevfusion_teacher
/home/yongjae/miniconda3/envs/ssr/bin/python \
  tools/bev_visualizer/prepare_teacher_cache.py maptrv2_teacher

/home/yongjae/miniconda3/envs/ssr/bin/python tools/bev_visualizer/build_site.py
/home/yongjae/miniconda3/envs/ssr/bin/python tools/bev_visualizer/verify_artifacts.py
```

Generated output lives under `work_dirs/bev_visualizer`, which resolves to the
large `/data1` volume. Raw selected SSR BEVs stay outside the web root. Teacher
raw cache files are never copied into or exposed by the site.

## Open the site

```bash
/home/yongjae/miniconda3/envs/ssr/bin/python tools/bev_visualizer/serve.py
```

Open `http://127.0.0.1:8766`. For a remote machine, keep the server bound to
localhost and use SSH port forwarding:

```bash
ssh -L 8766:127.0.0.1:8766 <host>
```

## Interpretation

- The common local model frame uses lateral/right `x∈[-15,15]m` and
  longitudinal/forward `y∈[-30,30]m`, with `+y` at the top. This was
  checked against command trajectories, prediction/GT geometry and feature
  hotspots; the BEVFusion source manifest's prose uses conflicting axis names.
- BEVFusion's native `[-54,54]m²` representation is cropped and resampled to
  the common ROI. Do not compare pixel resolution as if it were native.
- PCA is fit independently for each source. Similar RGB colors across models
  do not mean that channels or semantics match.
- The feature taps are architecturally different: SSR uses the transformer
  BEV, BEVFusion uses the fused BEV before its decoder backbone, and MapTRv2
  uses its encoder BEV. Side-by-side similarity is qualitative, not a direct
  channel-wise equivalence score.
- `source-global` normalization preserves within-source activation magnitude;
  `sample-local` is only for revealing weak spatial structure.
- Stage 1 planner output is untrained and is hidden by the overlay renderer.
- Source-level checkpoint/config/cache hashes, transforms and sample selection
  are available in the site's `data/provenance.json`. Individual rendered
  PNGs and selected raw tensors do not each carry their own checksum.
