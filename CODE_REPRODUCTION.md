# PointProgrammerNet code and reproduction

The minimal implementation is located at:

`../submission/pointprogrammernet_repro`

It contains the two models reported in the manuscript and excludes exploratory
architectures, datasets, cached tensors, checkpoints, and generated figures.

## Final models

`PointProgrammerSegmenter` uses three independently learned additive states.
Every state has shape `128 x 170`, every query decoder has the same
`170 -> 160 -> 160` structure, and the initial radii are `0.08`, `0.20`, and
`0.60`. The three decoded vectors, the query-coordinate feature, and the
ShapeNetPart category code are passed to the pointwise segmentation head. The
model contains 341,909 trainable parameters.

`PointProgrammerClassifier` uses the same three state shapes, normalized
partition keys, and initial radii. It reads state rows directly, uses the
middle summary as a reference, and adds a bounded correction derived from the
fine and broad summaries. Only the middle radius uses a `25x` learning-rate
multiplier. The model contains 529,426 trainable parameters.

Neither model applies a maximum over input or decoded points. Classification
retains no point list after the three states have been constructed.

## Environment

Python 3.10 or newer is recommended. The reported implementation was checked
with Python 3.10.20, PyTorch 2.5.1, CUDA 12.1, NumPy 2.0.1, and h5py 3.16.0.

From the minimal-code directory, install the dependencies:

```powershell
cd C:\path\to\fwpnet_refactor\submission\pointprogrammernet_repro
python -m pip install -r requirements.txt
```

## Datasets

Classification expects the extracted ModelNet40 HDF5 directory containing
files named `ply_data_train*.h5` and `ply_data_test*.h5`.

Segmentation accepts either of these ShapeNetPart layouts:

- the HDF5 release containing `train_hdf5_file_list.txt`; or
- the normal-point text release containing `synsetoffset2category.txt` and the
  `train_test_split` directory.

The reported experiments use XYZ only. Each object is centered and divided by
its maximum absolute XYZ coordinate. Both tasks use 1,024 points.

## Check the implementation

Run the unit tests and the streaming-state check before training:

```powershell
python -m unittest discover -s tests -v
python .\verify_streaming.py
```

The checks confirm the exact model sizes, the three `(128, 170)` state shapes,
forward output shapes, permutation invariance, and agreement between one-shot
and chunked state construction up to floating-point accumulation error.

## Full reproduction

ShapeNetPart segmentation:

```powershell
python .\train_segmentation.py `
  --data-dir "C:\data\shapenetcore_partanno_segmentation_benchmark_v0_normal" `
  --seeds 0,1,2 --epochs 20 --num-points 1024 `
  --batch-size 16 --eval-batch-size 16 `
  --learning-rate 0.003
```

ModelNet40 classification:

```powershell
python .\train_classification.py `
  --data-dir "C:\data\modelnet40_ply_hdf5_2048" `
  --seeds 0,1,2 --epochs 20 --num-points 1024 `
  --batch-size 32 --eval-batch-size 32 `
  --learning-rate 0.003
```

Training uses AdamW with weight decay `1e-4` and cosine learning-rate decay.
Classification additionally uses label smoothing `0.1`. Each command writes a
JSON result file and one checkpoint per seed under its output directory.

## Fast installation check

These commands verify loading and training without reproducing paper scores:

```powershell
python .\train_segmentation.py --data-dir "C:\data\ShapeNetPart" `
  --seeds 0 --epochs 1 --train-limit 32 --test-limit 16

python .\train_classification.py --data-dir "C:\data\ModelNet40" `
  --seeds 0 --epochs 1 --train-limit 64 --test-limit 32
```

## Streaming and distributed use

Both models expose the same state interface:

```python
states_a = model.build_states(points_a)
states_b = model.build_states(points_b)
states = model.merge_states(states_a, states_b)
```

For segmentation:

```python
logits = model.predict_from_states(query_xyz, category_ids, states)
```

For classification:

```python
logits = model.classify_states(states)
```

Independent devices may build states from disjoint point subsets and transmit
only the matrices. Elementwise addition at a central device gives the same
state as centralized encoding, subject only to floating-point summation order.
All devices must use identical trained weights, coordinate normalization, and
the same reference frame.

The retained float32 state contains 65,280 scalars, or 261,120 bytes (255 KiB),
regardless of the number of observations. Subtraction removes a separately
stored subset state, and multiplying the previous state by a factor in `[0,1]`
implements exponential decay.

## Reference results

The exact three-seed values are stored in
`../submission/pointprogrammernet_repro/reference_results.json`.

- ShapeNetPart: `92.95 +/- 0.02` point accuracy,
  `82.58 +/- 0.15` instance mIoU, and `79.10 +/- 0.25` category mIoU.
- ModelNet40: `86.59 +/- 0.23` clean accuracy and
  `82.47 +/- 0.93` density-shift accuracy.

Small numerical differences can arise from the PyTorch/CUDA version and GPU
kernel selection. The reported standard deviations are population standard
deviations over seeds `0,1,2`.
