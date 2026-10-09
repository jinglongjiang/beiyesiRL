# Bayesian Temporal Value Learning

This repository snapshots the local CrowdNav Bayesian temporal-replacement
research project. It contains source code, configurations, tests, experiment
protocols, result JSON/CSV, diagnostic arrays, training logs and learning curves.
Model/optimizer checkpoints, ORCA demonstration datasets, virtual environments
and unrelated experiment outputs are not uploaded.

- Current temporal variants: GRU, Bayesian Simple (mean readout), and Bayesian
  Complex (explicit mean/logvariance features after the 2026-10-08 repair).
- Historical Complex variants use 128-point expectation readout. Do not mix
  those results with the repaired architecture.
- The shared spatial encoder, 80-action grid, value lookahead, reward and
  IL/online MC learning framework remain inherited. Existing legacy filenames
  and compatibility modules do not mean Mamba training is enabled.
- Main report: [BAYES-TEMPORAL.md](BAYES-TEMPORAL.md). Its timestamps distinguish
  historical results from the state observed when that report was written.
- Historical 3,000/8,000 runs: `crowd_nav/runs/bayes-matched-v1/`.
- Historical fresh 10,000 runs: `crowd_nav/runs/bayes-optimized-20261007/`.
- Repaired three-arm/two-RL-seed runs: `crowd_nav/runs/bayes-fix-20261008/`.
- All six repaired runs now completed 10,000 RL episodes and 3,000 evaluation
  cases each. Complex exceeds Simple in observed success rate, but does not
  consistently exceed GRU; uncertainty-specific causality remains unverified.
- Final saved-result analysis: `crowd_nav/runs/bayes-fix-20261008/final-analysis.json`.
  Recompute with `python3 scripts/analyze-bayes-fix-results.py` (NumPy only;
  no checkpoint, GPU, training or evaluation launch).
- Read-only variance/timeout diagnostic (2026-10-09):
  [BAYES-VARIANCE-TIMEOUT-DIAGNOSTIC.md](BAYES-VARIANCE-TIMEOUT-DIAGNOSTIC.md).
  All inference used local RTX 3060, no 4090 access or retraining. Results and
  losslessly packed episode traces are in `crowd_nav/runs/bayes-diagnostic-20261009/`.
- Variance decision-layer decomposition (2026-10-09):
  [report](crowd_nav/runs/bayes-decomp-20261009/REPORT.md) and
  [module flow](crowd_nav/runs/bayes-decomp-20261009/module-flow.svg).
  Completed 1,200 real-policy episodes / 77,352 decisions with runtime-only
  shadow interventions, local RTX 3060, zero training and no Parent changes.
  Variance readout affects decisions, but a consistently beneficial direction
  is not established (`REAL_DIRECTION_NOT_ESTABLISHED`). This is neither a
  Complex-versus-Simple comparison nor a closed-loop intervention benefit.
  Protocols, raw JSON/NPZ, controls, interrupted/invalid evidence and analysis
  code are retained together in `crowd_nav/runs/bayes-decomp-20261009/`.
- Historical sync manifest: `SYNC-MANIFEST.json`; the latest diagnostic uses
  `crowd_nav/runs/bayes-decomp-20261009/sync-manifest.json`.
  Active evaluation CSVs are partial unless
  the corresponding complete-result artifact explicitly verifies completion.

Run commands from this project root. Follow the frozen experiment protocols
and scripts under `scripts/`; uploaded configurations retain their original
machine-specific paths. Reproduction requires separately supplying the dataset
and checkpoints listed by path/hash in those protocols. This Git snapshot is
not an asset-complete one-command reproduction package.

The original CrowdNav README and attribution are retained below.

---

# CrowdNav

**[`Website`](https://www.epfl.ch/labs/vita/research/planning/crowd-robot-interaction/) | [`Paper`](https://arxiv.org/abs/1809.08835) | [`Video`](https://youtu.be/0sNVtQ9eqjA)**

This repository contains the codes for our ICRA 2019 paper. For more details, please refer to the paper
[Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning](https://arxiv.org/abs/1809.08835).

Please find our more recent work in the following links 
- [Relational Graph Learning for Crowd Navigation, IROS, 2020](https://github.com/ChanganVR/RelationalGraphLearning).
- [Social NCE: Contrastive Learning of Socially-aware Motion Representations, ICCV, 2021](https://github.com/vita-epfl/social-nce).

## Abstract
Mobility in an effective and socially-compliant manner is an essential yet challenging task for robots operating in crowded spaces.
Recent works have shown the power of deep reinforcement learning techniques to learn socially cooperative policies.
However, their cooperation ability deteriorates as the crowd grows since they typically relax the problem as a one-way Human-Robot interaction problem.
In this work, we want to go beyond first-order Human-Robot interaction and more explicitly model Crowd-Robot Interaction (CRI).
We propose to (i) rethink pairwise interactions with a self-attention mechanism, and
(ii) jointly model Human-Robot as well as Human-Human interactions in the deep reinforcement learning framework.
Our model captures the Human-Human interactions occurring in dense crowds that indirectly affects the robot's anticipation capability.
Our proposed attentive pooling mechanism learns the collective importance of neighboring humans with respect to their future states.
Various experiments demonstrate that our model can anticipate human dynamics and navigate in crowds with time efficiency,
outperforming state-of-the-art methods.


## Method Overview
<img src="https://i.imgur.com/YOPHXD1.png" width="1000" />

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting Started
This repository is organized in two parts: gym_crowd/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.
```
python train.py --policy sarl
```
2. Test policies with 500 test cases.
```
python test.py --policy orca --phase test
python test.py --policy sarl --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy orca --phase test --visualize --test_case 0
python test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0
```
4. Visualize a test case.
```
python test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0
```
5. Plot training curve.
```
python utils/plot.py data/output/output.log
```


## Simulation Videos
CADRL             | LSTM-RL
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/vrWsxPM.gif" width="400" />|<img src="https://i.imgur.com/6gjT0nG.gif" width="400" />
SARL             |  OM-SARL
<img src="https://i.imgur.com/rUtAGVP.gif" width="400" />|<img src="https://i.imgur.com/UXhcvZL.gif" width="400" />


## Learning Curve
Learning curve comparison between different methods in an invisible setting.

<img src="https://i.imgur.com/l5UC3qa.png" width="600" />

## Citation
If you find the codes or paper useful for your research, please cite our paper:
```bibtex
@inproceedings{chen2019crowd,
  title={Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning},
  author={Chen, Changan and Liu, Yuejiang and Kreiss, Sven and Alahi, Alexandre},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={6015--6022},
  year={2019},
  organization={IEEE}
}
```
