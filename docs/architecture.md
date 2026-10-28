# Architecture and code map

| Stage | Implementation | Main operation |
| --- | --- | --- |
| Flow preparation | `preprocessing.py` | Train-fitted grouping, numeric features, rolling mean, differences, calendar cycles, smoothed IV and scaling |
| Temporal graph | `graph.py` | One user per graph; one averaged timestamp per node; self, adjacent and two-step edges |
| Soft feature selection | `model.py: FeatureSelector` | Linear → GELU → LayerNorm → sigmoid → elementwise feature weighting |
| Temporal branch | `model.py: TemporalBranch` | Linear embedding, sinusoidal positional encoding, custom Transformer with convolutional gate |
| Graph branch | `model.py: GraphBranch` | Three GCN layers with GELU/dropout, followed by two-head GAT (`concat=False`) |
| Fusion | `model.py: GatedFusion` | Separate sigmoid gates on both representations, concatenation and projection |
| Adapter | `model.py: LightAnomalyDetector` | A linear feature adapter; not an explicit difficulty curriculum |
| Scoring | `model.py: UncertaintyAwareScoring` | Scalar mean and positive variance per timestamp; variance capped at 25 |
| Objective | `losses.py` | Monte Carlo heteroscedastic focal loss plus small variance penalty |
| Prediction | `losses.py: logistic_normal_probability` | `sigmoid(mean / sqrt(1 + pi * variance / 8))` |

For a user with T timestamps and D selected features, input tensors have shapes `x: [T, D]`, `edge_index: [2, E]`, `y: [T, 1]`. Both branches output `[T, H]`. The scoring head returns mean and variance with shape `[T, 1]`.

The temporal attention sees the full user sequence in both directions and costs O(T²). This implementation is designed for offline detection. It does not apply a causal mask, sequence truncation, graph batching, or sliding-window inference. Users with very long histories may exceed available memory.

The public name `STDNet` aliases the historical `LightAnomalyDetector` class. The original name remains available for source compatibility, but old serialized checkpoints/preprocessors are not supported by this pipeline.
