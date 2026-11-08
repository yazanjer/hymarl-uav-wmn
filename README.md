# HyMARL — Hybrid multi-agent RL for UAV-assisted wireless mesh networks

Code for the article *Learning Where It Helps: Hybrid Multi-Agent Reinforcement Learning
with Classical Channel Assignment and Load-Aware Association for UAV-Assisted Wireless
Mesh Networks* (B. M. Ismael, M. A. Ngadi, J. B. M. Sharif, M. S. Taha, Y. Aljeroudi;
submitted to *Ad Hoc Networks*).

HyMARL learns the velocity and transmit power of every UAV with a parameter-shared TD3
actor (centralised training, decentralised execution), assigns channels with a greedy
rule and associates users with routers by a load-aware rule. The repository contains the
analytical simulator, the training and evaluation scripts for every experiment in the
article, the logged results, and the ns-3 program used for the packet-level validation.

Trained agents, the policies exported for ns-3 and the evaluation-episode traces are
deposited on Zenodo: https://doi.org/10.5281/zenodo.24057627

## Layout
```
src/       simulator and agent (MATLAB): NetworkSim, channel, Bianchi MAC, mobility,
           TD3 agent, training/evaluation, statistics
scripts/   experiment drivers and report/figure generators
results/   logged results (.mat/.csv) and run logs of the reported experiments
ns3val/    ns-3.44 program (uavval.cc), batch scripts, analysis and table scripts,
           summaries and raw results (results/*.tgz)
output/    generated figures and tables (created by the report scripts)
```

## Requirements
- MATLAB R2025a with the Reinforcement Learning, Deep Learning, Statistics and Machine
  Learning toolboxes (Parallel Computing Toolbox optional).
- ns-3.44 (built from source) and Python 3 for the packet-level validation.

## Reproducing the results (MATLAB, from `scripts/`)
| Experiment | Command | Output |
|---|---|---|
| Simulator sanity check | `test_sim` | `results/sanity_metrics.png` |
| Headline and equal-spectrum comparisons (Scenario A, N=10) | `run_relog` | `results/headline_A8log.mat`, `results/fairAlog.mat` |
| Fleet sizes, Scenarios B/C, ablation, sensitivity | `run_campaign_log(8,50,8,4,'campaign8')` | `results/campaign8.mat` |
| HyMARL (greedy channels) on all settings | `run_hybrid(8,50)` | `results/hybrid.mat` |
| Non-learning references | `run_improve` | `results/improve.mat` |
| Association hysteresis | `run_hyst2(8,50,'hyst2')` | `results/hyst2.mat` |
| Figures | `make_figures('A8log','fairAlog')` | `output/figures/*.pdf` |

Every run seeds its random streams (`rng(1000*seed+idx,'twister')`); eight paired seeds
are used throughout, with 30 evaluation episodes of 60 steps per seed.

## Packet-level validation (ns-3)
1. `export_ns3` and `export_ns3_cells` (MATLAB) export the trained actors as JSON and the
   user trajectories of the evaluation episodes (also available in the Zenodo record).
2. Copy `ns3val/uavval.cc` to `ns-3.44/scratch/uavval/` and build it.
3. `ns3val/run_batch.sh` (Scenario A, N=10) and `ns3val/run_batch2.sh` (other settings,
   5% load) run all methods, seeds and episodes; `analyse.py`/`analyse2.py` and
   `make_ns3_tables.py`/`make_ns3_cells_table.py` produce the tables.

## Licence
Code: MIT (see `LICENSE`). Zenodo data: CC BY 4.0.
