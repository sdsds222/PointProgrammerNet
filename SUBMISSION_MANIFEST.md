# Submission manifest

This directory is the self-contained, review-facing implementation of the
frozen PointProgrammerNet architectures. The exploratory research repository
remains outside this directory.

## Included

| Path | Purpose |
|---|---|
| `pointprogrammernet/layers.py` | Continuous partition keys, additive `K^T V` write, second-order statistics, and query/state decoding |
| `pointprogrammernet/models.py` | Frozen segmentation and classification networks |
| `pointprogrammernet/data.py` | ShapeNetPart and ModelNet40 loading and normalization |
| `pointprogrammernet/training.py` | Training loops, metrics, optimizer grouping, and density-shift evaluation |
| `train_segmentation.py` | ShapeNetPart reproduction entry point |
| `train_classification.py` | ModelNet40 reproduction entry point |
| `verify_streaming.py` | Chunk merge, permutation invariance, state shape, and memory verification |
| `tests/test_models.py` | Model shapes, exact parameter counts, and additivity tests |
| `reference_results.json` | Frozen reference metrics and exact experimental protocols |
| `run_reproduction.ps1` | One-command verification and optional training |

## Frozen source mapping

| Submission class | Research-repository model name | Parameters |
|---|---|---:|
| `PointProgrammerSegmenter` | `fwp_dual_read_partition_multiscale_seg` | 366,389 |
| `PointProgrammerClassifier` | `pointprogrammer_state_multiscale_residual_main_r25` | 529,426 |

Strict state-dictionary loading and random-input comparison were performed
against both research-repository classes. Both maximum absolute output errors
were exactly `0.0`.

## Deliberately excluded

Historical probe architectures, cached datasets, checkpoints, generated
figures, temporary outputs, and Python caches are not part of this package.
They are unnecessary for reproducing the two frozen models and would obscure
the method being submitted.

