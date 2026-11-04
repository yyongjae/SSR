# Refiner on v2 + BEV KD for NAVSIM

This branch trains and evaluates an end-to-end planner on [NAVSIM](https://github.com/autonomousvision/navsim): a camera-only v2 planner, BEV knowledge distillation from a LiDAR teacher and a map teacher, and a refiner that scores and picks among candidate trajectories.

| Read | For |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | Preparing a new server: ordered steps, disk, hours, checks |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Running experiments and ablations with `scripts/experiment.sh` |
| [docs/FAIRNESS.md](docs/FAIRNESS.md) | What is locked, what to disclose, which earlier numbers not to use |
| [THIRD_PARTY.md](THIRD_PARTY.md) | Origin and licence of code, data and models |

```bash
bash scripts/experiment.sh check                                    # builds the default recipe on CPU, compares it with the reference
python tools/release/verify_assets.py --profile full --quick        # are the inputs here, and the right ones
bash scripts/experiment.sh train GPUS=0,1,2,3,4,5,6,7               # 19.9 h on 8 x 24 GB GPUs
bash scripts/experiment.sh eval RUN=<name> GPUS=0,1,2,3
```

## Upstream

The code base started from SSR ("Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving", Peidong Li and Dixiao Cui, Zhijia Technology, ICLR 2025, [arXiv](https://arxiv.org/abs/2409.18341)), a nuScenes planner. Its pipeline (`projects/`, `tools/train.py`, `docs/install.md`, `docs/prepare_dataset.md`) is still in the tree and is not used by this branch.

## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@inproceedings{li2025navigationguidedsparsescenerepresentation,
  title={Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving},
  author={Peidong Li and Dixiao Cui},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## License
All code in this repository is under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgement
SSR is based on the following projects: [VAD](https://github.com/hustvl/VAD), [GenAD](https://github.com/wzzheng/GenAD), [BEV-Planner](https://github.com/NVlabs/BEV-Planner) and [TokenLearner](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner). Many thanks for their excellent contributions to the community.
