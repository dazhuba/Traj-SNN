# Traj-SNN

Official experiment package for **“A Reliable and Interpretable Neural–Logic AI Component for UAV Trajectory Analytics.”**

Traj-SNN combines a multi-resolution ModernTCN neural pathway with differentiable physical predicates, soft logic, and an uncertainty-aware bounded correction gate. The repository covers the two tasks reported in the manuscript:

- four-class UAV intent recognition from 8, 16, and 32 s trajectory windows;
- intent-conditioned 30-step 3-D trajectory forecasting.

[中文说明](README_zh.md) · [Reproduction map](REPRODUCIBILITY.md) · [Data documentation](data/README.md)

![Traj-SNN architecture](figures/fig1_architecture.png)

## Main reported results

Recognition values are means over ten complete-flight split seeds (42–51). Forecast values are means over three optimization seeds.

| Task | Metric | Traj-SNN | Strongest matched baseline |
|---|---:|---:|---:|
| Intent recognition, equal-window average | Macro-F1 | **95.16%** | 91.73% |
| Intent recognition, equal-window average | Accuracy | **96.64%** | 93.91% |
| 30-step forecasting | ADE | **47.092 m** | 49.205 m |
| 30-step forecasting | FDE | **87.188 m** | 94.331 m |
| 30-step forecasting | coordinate RMSE | **44.939 m** | 47.094 m |

The immutable paper-result tables and plot source data are under results/. New runs are written to the ignored outputs/ directory.

## Repository layout

~~~text
Traj-SNN/
├── research_intent/                 # features, models, logic, recognition experiments
├── trajectory_prediction_softlogic/ # forecasting models, training and evaluation
├── external_baselines/              # matched baseline adapters
├── scripts/                         # portable runners and result summarizer
├── data/
│   ├── processed/                   # 8/16/32-s recognition NPZ files
│   └── raw/                         # complete trajectories for forecasting
├── checkpoints/                     # released frozen intent classifier
├── results/                         # paper result JSON/CSV and plot source data
├── figures/                         # manuscript figures used by this README
├── analysis/                        # gate, explanation, latency and plotting audits
└── paper/Manuscript.pdf
~~~

## Installation

Python 3.10 or newer is recommended.

~~~bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
~~~

For a CUDA build, install the matching PyTorch wheel first and then install the remaining requirements.

## Quick validation

These tests use synthetic trajectories and do not train the paper models.

~~~bash
python -m research_intent.smoke_test
python -m trajectory_prediction_softlogic.smoke_test
~~~

## Intent-recognition experiments

A short pipeline check:

~~~bash
python scripts/run_recognition.py \
  --seeds 42 \
  --workers 1 \
  --backbone-epochs 2 \
  --logic-epochs 2
~~~

The complete paper protocol uses the defaults: windows 8/16/32 s, seeds 42–51, a 75/15/10 complete-flight split, at most 20 ModernTCN epochs, and patience 5.

~~~bash
python scripts/run_recognition.py
~~~

The runner is resume-safe. It executes the backbone, logic correction, multi-resolution-only, full Traj-SNN, ablation, and history-aware audit phases in dependency order. To run selected phases:

~~~bash
python scripts/run_recognition.py --phases backbone logic trajsnn
~~~

Aggregate any completed recognition runs:

~~~bash
python scripts/summarize_results.py --root outputs/recognition
~~~

## Trajectory-forecasting experiments

Train the standalone soft-logic forecaster:

~~~bash
python -m trajectory_prediction_softlogic.train
~~~

Reproduce the frozen-classifier routing experiment used by the final forecasting model:

~~~bash
python -m trajectory_prediction_softlogic.frozen_classifier_experiment
~~~

Run the matched forecasting baselines and combine them with the final-model result:

~~~bash
python -m trajectory_prediction_softlogic.mainstream_benchmark
~~~

The defaults point to data/raw, data/processed, and checkpoints/intent_classifier_seed44.pt. Forecast training can be expensive; all published summary files are already available in results/forecasting/.

Single-flight inference with a newly trained forecasting checkpoint:

~~~bash
python -m trajectory_prediction_softlogic.predict \
  --checkpoint outputs/forecasting/default/best.pt \
  --csv path/to/flight.csv \
  --observation-length 32 \
  --output outputs/prediction.json
~~~

## Data protocol

The released recognition files contain 17,591 (8 s), 8,454 (16 s), and 3,896 (32 s) windows generated with 50% overlap. Records are grouped by source flight before splitting, so overlapping windows from one flight cannot cross train/validation/test partitions.

The four proxy-intent labels are:

| Label | Intent |
|---:|---|
| 0 | point-to-point |
| 1 | perimeter patrol |
| 2 | area mapping |
| 3 | package delivery |

The models receive time-stamped 3-D position only; source identity, platform metadata, radar/RF measurements, payload, attack status, and future samples are not recognition inputs. See [data/README.md](data/README.md) for record schemas, public sources, and redistribution notices.

> Security note: the NPZ files contain pickled NumPy object arrays. Load only files from a trusted source and use allow_pickle=True only for these released files.

## External baselines

The bundled ModernTCN files are the minimum official source needed by the Traj-SNN recognition pipeline and retain the upstream MIT license. Adapters for the other paper baselines are included under external_baselines/; their full author repositories are intentionally not duplicated because some are large and carry separate licenses. Exact provenance and expected source directories are documented in [external_baselines/SOURCES.md](external_baselines/SOURCES.md).

## Citation

~~~bibtex
@article{li2026trajsnn,
  title   = {A Reliable and Interpretable Neural--Logic AI Component for UAV Trajectory Analytics},
  author  = {Li, Wei-Xing and Hu, Yuan and Zhang, Zheng-Ning and Zhang, Zheng and Hou, Wei},
  year    = {2026},
  note    = {Manuscript}
}
~~~

## License and third-party material

The Traj-SNN code in this release is currently provided as **all rights reserved**; replace the root LICENSE with the license selected by the authors before announcing the repository as open source. Third-party code and data retain their original terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and LICENSES/.
