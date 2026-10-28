# Data preparation

Datasets are not bundled. Obtain InSDN independently and place local files below this directory; `.gitignore` excludes them. The supplied experiment folder contains existing CSV splits that can also be used directly by absolute path.

## Required columns

| Purpose | Accepted columns | Behavior |
| --- | --- | --- |
| Timestamp | `sc_time` or `Timestamp` | ISO date/time, or supported day/month/year formats; invalid times raise an error |
| Label | `Label`, `flag`, `label`, `Class`, `Anomaly`, `Result` | Numeric 0/1, or text: Normal/Benign → 0; other nonempty text → 1 |
| User | `sc_user_id`, or `Src IP` + `Dst IP`, or `Flow ID` | Adaptive choice is fitted on training data and frozen for testing |
| Features | Numeric traffic columns | Chosen on training data; all trained numeric columns must exist at evaluation |

Column names are stripped of surrounding whitespace. Do not mix numeric and text labels. Check text label spelling: any non-normal text, including an accidental typo, is treated as an attack. Remove auxiliary target-derived fields before training; the program excludes known label names but cannot infer every custom target proxy.

Supported dates include `2025-01-01 13:58:00`, ISO with fractional seconds, `5/2/2020 13:58`, and `26/12/2019 01:26:13 PM`. Day/month/year parsing follows the supplied implementation. Normalize other date formats before use.

Missing/infinite numeric values are replaced with zero before feature engineering and scaling. Rolling and difference features are computed within users after chronological sorting. Flows at the same exact timestamp are averaged into one node; this is not fixed-width time-window binning.

## Split protocol

Combine normal traffic and the desired attack traffic into a CSV before running:

```bash
python -m stdnet.prepare --data data/insdn/all.csv --attack Probe --output data/insdn/probe70
python -m stdnet.prepare --data data/insdn/all.csv --attack Probe --train-fraction 0.1 --output data/insdn/probe10
```

The helper removes exact duplicate flow rows, stratifies on the binary flow label, and keeps a fixed 30% test set. Small-sample training retains 10% of the original deduplicated scenario, not 10% of the 70% training set. Integer rounding may cause small fraction differences. The unused portion is discarded. The resulting `split.json` records seed, counts and source hash.

Splits are disjoint at the flow-row level. Users and timestamps can overlap across splits. For unseen-user generalization or forecasting, create separate user-disjoint or chronological splits; the helper does not implement those protocols. Feature engineering is performed independently within each split, so there is no shared feature history across train and test.

The original numbered files do not include a reliable manifest connecting every split to a paper scenario. This repository intentionally does not guess that mapping or claim the new splitter recovers historical experiments.
