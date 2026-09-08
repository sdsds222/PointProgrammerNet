# PointProgrammerNet paper figures

All figures use a Pattern Recognition Letters-style white, dark-gray, and crimson palette. Crimson is reserved for the proposed fixed-state path or the emphasized condition; gray denotes controls and secondary structure.

0. `fig00_pointprogrammernet_architecture`: shared three-state additive encoder with orthogonal connectors and the task-dual segmentation and classification heads.

1. `fig01_additivity_fixed_state`: retain the design; regenerate with `fwp_dual_read_partition_multiscale_seg` after its final checkpoint exists. The new retained state is three `128x170` matrices (255 KiB in float32).
2. `fig02_state_operations`: retain the design; regenerate with the same final checkpoint. Describe exponential decay as a state operation, not as demonstrated downstream accuracy.
3. `fig03_controlled_ablation`: redesigned and regenerated from existing three-seed screening results. The left panel supports multiscale partition states for segmentation; the right panel supports residual state readout and the `25x` middle-radius rate for classification.
4. `fig04_streaming_efficiency`: retain the design; regenerate because equal FWP rows change the measured state-construction cost.
5. Qualitative streaming segmentation: generate only after the final segmentation checkpoint is available.

Main accuracy is reported separately:

- `table01_segmentation_accuracy.tex`;
- `table02_classification_accuracy.tex`.

The old max-based normal-input table is historical and must not be included as a final PointProgrammerNet result.
