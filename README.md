<div align="center">

# STD-Net

**Spatial-Temporal Dual-Branch Learning for Anomaly Detection in Software-Defined Networks**

Transformer · Temporal Graph · Gated Fusion · Uncertainty Scoring

[中文说明](README.zh-CN.md) · [Data guide](data/README.md) · [Architecture](docs/architecture.md) · [Migration notes](docs/migration.md)

</div>

STD-Net learns timestamp-level network anomalies from a temporal graph for each user. A Transformer branch captures sequence context, while a graph branch models local and skip-connected relationships between timestamps. Learned gates fuse the representations before a binary scoring head predicts a latent logit mean and variance. 

## Quick start

Use Python 3.10 or newer and install an appropriate PyTorch build for your machine. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate
python -m pip install -e .
python -m stdnet --help
```

Place your independently obtained CSV splits under `data/insdn/`. Datasets and run artifacts are ignored by Git.

```bash
python -m stdnet train --data data/insdn/train.csv --config configs/default.json --output runs/insdn-01
python -m stdnet evaluate --data data/insdn/test.csv --run runs/insdn-01 --device cpu
```

`stdnet` is also available as an installed console command. All data, config and output paths are explicit and resolve from the current working directory. Training and evaluation require empty output directories to prevent overwriting previous runs.

To prepare a scenario from a combined CSV containing both normal flows and attack flows:

```bash
python -m stdnet.prepare --data data/insdn/all.csv --attack DoS --output data/insdn/dos70
python -m stdnet.prepare --data data/insdn/all.csv --attack DoS --train-fraction 0.1 --output data/insdn/dos10
```

The default split uses 70% of deduplicated flows for training and 30% for testing. `--train-fraction 0.1` uses 10% of the whole scenario for training and the same held-out 30% when the seed and input are identical. Users may appear in both sets; this is a flow-level split. This new splitter does not reconstruct the exact historical splits.

## Repository

```text
STD-Net/
├── configs/default.json       # Effective model, preprocessing and training settings
├── data/README.md             # CSV schema and split protocol
├── docs/                     # Architecture, changes and source inventory
├── src/stdnet/
│   ├── preprocessing.py       # Fitted training-only preprocessing
│   ├── graph.py               # Timestamp nodes and temporal edges
│   ├── model.py               # STD-Net architecture
│   ├── losses.py              # Heteroscedastic focal objective
│   ├── metrics.py             # Binary and per-class metrics
│   ├── prepare.py             # Reproducible flow split helper
│   └── cli.py                 # Train / evaluate entry points
├── tests/                    # Behavioral regression tests
└── .github/workflows/ci.yml   # CPU checks on push / pull request
```

## Experiment behavior

- **Training unit:** one user's full temporal graph per optimizer step. Graphs are processed individually; there is no misleading `batch_size` setting.
- **Graph:** timestamp nodes, self-loops, bidirectional adjacent edges and bidirectional two-step skip edges. This is a user temporal graph, not a physical SDN switch topology.
- **Label rule:** `any` retains the supplied code's rule: a timestamp is anomalous if it contains any anomalous flow. `ratio` compares its attack fraction with the training attack fraction, following the paper's ratio formulation with a frozen training reference.
- **Feature selection:** IV and the standard scaler are fitted on training flows only. Numeric features, three-flow rolling means, differences and time features are used. Identifiers and known label columns are excluded.
- **Uncertainty:** the score head models variance in latent logit space. Evaluation uses a logistic-normal approximation. `latent_std` is not a calibrated probability confidence interval.
- **Checkpoint selection:** lowest mean per-user training loss, as in the supplied implementation. No validation-based model selection or test-based threshold tuning is performed. Threshold is fixed in the config, default `0.5`.

The main runtime outputs are `model.pt`, `preprocessor.pkl`, `features.json`, `config.json`, `history.json` and `run.json`. Evaluation writes `metrics.json` and `predictions.csv`, including per-class precision, recall and F1. Only load preprocessing pickle files produced by a trusted run.

## Verification and reproducibility

```bash
python -m unittest discover -s tests -v
```

See [local validation](docs/validation.md) for the actual environment and checks performed. CI configuration is included; hosted CI has not been run as part of this cleanup. No benchmark results are claimed for the refactored pipeline. For 30-run experiments, use separate output directories and explicitly change the seed in a copied config for each run.

## Paper and code provenance

**Zhewei Liu et al.** *Spatial-Temporal Dual-Branch Learning for Anomaly Detection in Software-Defined Networks.* The supplied manuscript is the reference for the architecture and experiment discussion. Publication venue, DOI and final bibliographic metadata were not supplied, so no unverified citation is included.

The original experiment files remain in the source project. A [file-by-file inventory](docs/source_inventory.json) records all cleanup decisions; [source hashes](docs/source_hashes.json) identify the core input files.

No license was present in the supplied project. Choose an appropriate license with the relevant authors or rights holders before granting reuse permissions. Third-party datasets are distributed separately under their own terms.
