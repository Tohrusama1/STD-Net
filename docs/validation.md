# Local validation

Checks were performed on Windows on 2026-10-28. This report records engineering validation, not a reproduction of the paper's benchmark tables.

## Environment

| Component | Version |
| --- | --- |
| Python | 3.13.5 |
| PyTorch | 2.6.0+cpu |
| PyTorch Geometric | 2.6.1 |
| NumPy | 1.26.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |

The dependency ranges in `pyproject.toml` are compatibility declarations, not a guarantee that every version combination has been tested. Local package building used existing dependencies without network access. A fresh online environment installation and hosted GitHub Actions were not tested.

## Checks performed

1. Eight regression tests passed: label-independent test features and user IDs; frozen fitted metadata; timestamp ordering and aggregation rules; required-column and binary-label validation; finite IV on pure bins; repeatable model forward without input mutation; variance-head gradients; single-class metrics; and flow split disjointness with a shared holdout across training fractions. Related assertions are grouped into eight test methods.
2. All package and test Python files compiled successfully; CLI help was checked.
3. The package built as a wheel and installed into an isolated local target directory using `--no-index --no-deps --no-build-isolation`. Tests and CLI help were then run against the installed package.
4. With the default model dimensions, the cleaned model loaded all original parameter keys/shapes strictly. On the same three-node graph, both mean and variance evaluation outputs matched the supplied model exactly. This check concerns the neural architecture, not the changed preprocessing pipeline.
5. An end-to-end CPU run used 80 real flows from the original InSDN training CSV: 40 normal and 40 attack flows, selected only for a small functional check. The new stratified split produced 56 training flows and 24 held-out test flows. A one-epoch small model (`hidden_dim=8`, `d_ff=16`, one Transformer layer; IV lower bound 0 for this check) trained, saved, reloaded and evaluated successfully. Evaluation covered 22 aggregated timestamps. JSON metrics and per-node prediction CSV files were generated successfully.

The small functional run is deliberately not presented as detection performance. It does not exercise full-scale memory behavior, CUDA, all original scenarios, 30 repeated trials, or the exact historical preprocessing and split protocol.

Only a dependency deprecation warning from PyG/Python typing was observed in the unit tests; it did not affect their outcome. DataFrame fragmentation encountered during the initial real-data check was fixed by assembling engineered columns together, and the end-to-end check passed again afterwards.
