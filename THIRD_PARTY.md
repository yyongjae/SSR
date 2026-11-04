# Third-party code, data and models

This list records what the files in this repository and in the assets say. It is not legal advice. Where a licence text was not
read, the row says "not checked". Open questions are in [docs/FAIRNESS.md](docs/FAIRNESS.md), section 4.

## Code in this repository

| Part | Origin | Licence | Note |
|---|---|---|---|
| Repository root | SSR, [PeidongLi/SSR](https://arxiv.org/abs/2409.18341), Zhijia Technology | Apache-2.0 (`LICENSE`) | Modified files need notices; there is no NOTICE file |
| `navsim/` | NAVSIM devkit, [autonomousvision/navsim](https://github.com/autonomousvision/navsim), University of Tuebingen | Apache-2.0 | Fork based on v1.0, modified. `setup.py` states apache-2.0 and "Free for non-commercial use" together; this needs a fix |
| `navsim/planning/simulation/planner/pdm_planner` | nuPlan devkit and PDM | see the upstream links in the code | not checked |
| `navsim/agents/para_ssr/modules/anchor_planner.py`, anchors | WoTE | not checked | Anchor file and `pdm_score_256` come from WoTE's published data |
| `navsim/agents/para_ssr/modules/kinematics.py` | TOAD, DiffusionDriveV2 | not checked | Ported, per the code comments |
| `navsim/agents/para_ssr/modules/lidar_encoder.py` | SafeDrive | not checked | LiDAR path; not on the camera path |
| `navsim/agents/para_ssr/modules/tokenlearner.py` | TokenLearner (scenic) | not checked | |
| `navsim/agents/WoTE/modules/blocks.py` | DAB-DETR through WoTE | not checked | |
| `projects/` | OpenMMLab headers (Apache-2.0); `metric_motion.py` Argo (MIT) | as in the headers | Older nuScenes code, not on the release path |
| `tools/refiner/rescore_attr.py` | Written for this project | repository licence | Subclasses the NAVSIM scorer |

## Environment

`requirements.lock.txt` holds the exact versions. Notable: PyTorch, PyTorch Lightning, timm, hydra-core, nuplan-devkit
(`motional/nuplan-devkit` @ `ce3c323`). Each has its own licence; not checked one by one. `spconv-cu126` and `cumm-cu126` are
installed but not used on the camera path.

## Data and models used in training or evaluation

| Item | Source | Licence or terms | Note |
|---|---|---|---|
| OpenScene / NAVSIM logs and sensors | HF `OpenDriveLab/OpenScene` | Inherits the nuPlan dataset licence; the devkit's `docs/install.md` points to it | Derived caches (BEV, metric) fall under those terms; redistribution is an open decision. Text not checked |
| nuPlan maps | Motional | nuPlan terms | not checked |
| ReSMap front64 cache and checkpoint | HF dataset `rudals/resmap-navsim-teacher-kd`, revision `73db598669a8e6edb2b585fcc14ebe54b01b15f5` | **cc-by-nc-4.0**, derived from NAVSIM / nuPlan | Non-commercial. Camera + satellite input. Effect of the NC terms on weights distilled from it is an open question |
| Satellite imagery inside ReSMap | Mapbox, per the HF card as quoted in an earlier audit | Mapbox terms | Current card and terms not re-read |
| BEVFusion code | [yyongjae/bevfusion](https://github.com/yyongjae/bevfusion) @ `b939770`, from MIT BEVFusion; OpenMMLab headers | Apache-2.0 | 64 m checkpoint and cache are not published |
| Swin nuImages pretrain (for BEVFusion) | Upstream BEVFusion download link | nuImages terms | Not checked |
| ResNet-34 weights | timm `resnet34.tv_in1k` through HF, revision `1b7b21cca82ff974d341713f777bf740c2db38c4` on the reference server | Model card terms; ImageNet | Downloaded at first training start; the code does not pin the revision. Not used at inference (`backbone_pretrained=False`) |
| Anchors, `pdm_score_256` | WoTE (repository data, Google Drive) | not checked | Scores come from the NAVSIM scorer on navtrain; no trained weights |

## Citation

Keep the SSR citation in [README.md](README.md). Cite NAVSIM, nuPlan, OpenScene, WoTE, BEVFusion and ReSMap in any paper that uses
this pipeline.
