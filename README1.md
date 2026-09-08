# PointProgrammerNet reproducibility package

This folder contains the minimal code needed to reproduce the frozen,
no-global-max PointProgrammerNet models. It deliberately excludes exploratory
probes, datasets, checkpoints, caches, and generated figures.

## Frozen models

- `PointProgrammerSegmenter`: three equal `128 x 170` second-order FWP states,
  initial radii `0.08/0.20/0.60`, 160 channels read from each state, direct
  three-way concatenation, and a pointwise segmentation head. Parameter count:
  366,389.
- `PointProgrammerClassifier`: the same three additive states, state-only row
  summaries, and a bounded fine/broad residual around the middle summary. Only
  the middle radius uses a `25x` learning-rate multiplier. Parameter count:
  529,426.

Neither model performs a decoded-point global maximum. The classification
model deletes point tensors after constructing the states. The segmentation
model queries each coordinate independently from the fixed states and its own
coordinate-derived clean feature.

## Environment

Python 3.10 or newer is recommended.

The final package was verified with Python 3.10.20, PyTorch 2.5.1,
CUDA 12.1, NumPy 2.0.1, and h5py 3.16.0.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python .\verify_streaming.py
```

The streaming check verifies that shuffled and chunked writes reproduce the
single-batch state up to floating-point accumulation error. Each sample retains
three float32 `128 x 170` matrices: 261,120 bytes (255 KiB), independent of the
number of observed points.

## Data

Segmentation uses the standard ShapeNetPart normal-point release but reads XYZ
columns only. Pass either its extracted directory or a compatible HDF5 layout.
Classification uses the ModelNet40 `modelnet40_ply_hdf5_2048` directory.

Both loaders center every object and divide XYZ by its maximum absolute
coordinate, matching the reported experiments. ModelNet40 density-shift
evaluation samples 1,024 points with the fixed directional density gradient
implemented in `pointprogrammernet/data.py`.

## Full training

```powershell
python .\train_segmentation.py --data-dir C:\path\to\ShapeNetPart --seeds 0,1,2 --epochs 20
python .\train_classification.py --data-dir C:\path\to\modelnet40_ply_hdf5_2048 --seeds 0,1,2 --epochs 20
```

Defaults match the paper protocol: 1,024 points, AdamW, base learning rate
`3e-3`, weight decay `1e-4`, cosine decay, segmentation batch size 16, and
classification batch size 32. Classification uses label smoothing `0.1`.

For a fast installation check:

```powershell
python .\train_segmentation.py --data-dir C:\path\to\ShapeNetPart --seeds 0 --epochs 1 --train-limit 32 --test-limit 16
python .\train_classification.py --data-dir C:\path\to\ModelNet40 --seeds 0 --epochs 1 --train-limit 64 --test-limit 32
```

## Streaming API

Both models expose `build_states(points)` and `merge_states(...)`.

```python
states_a = model.build_states(points_a)
states_b = model.build_states(points_b)
history = model.merge_states(states_a, states_b)
```

For segmentation, call `predict_from_states(queries, categories, history)`.
For classification, call `classify_states(history)`. Append, merge, removal,
and exponential decay are simple matrix operations. Delta-rule adaptation is a
future extension and is not used in the reported models.

## Contents

- `pointprogrammernet/layers.py`: continuous keys, additive write, second-order
  statistics, and state decoding.
- `pointprogrammernet/models.py`: the two frozen architectures.
- `pointprogrammernet/data.py`: ModelNet40 and ShapeNetPart loading.
- `pointprogrammernet/training.py`: reported training and evaluation protocol.
- `verify_streaming.py`: algebra and fixed-memory verification.
- `tests/test_models.py`: shape, parameter-count, and additivity tests.
- `reference_results.json`: recorded full and quick-protocol reference values.
- `SUBMISSION_MANIFEST.md`: file roles, exclusions, and source compatibility.

The model module names and tensor shapes are compatible with checkpoints from
the research repository's final model names:
`fwp_dual_read_partition_multiscale_seg` and
`pointprogrammer_state_multiscale_residual_main_r25`.
