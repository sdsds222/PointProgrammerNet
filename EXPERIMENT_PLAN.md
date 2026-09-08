# PointProgrammerNet: final no-max experiment plan

Target venue: *Neural Processing Letters*.

The frozen architecture is:

- segmentation: `fwp_dual_read_partition_multiscale_seg`;
- classification: `pointprogrammer_state_multiscale_residual_main_r25`.

Both use three independent `128x170` second-order additive FWP states, normalized continuous partition keys, and initial radii `0.08/0.20/0.60`. Neither model performs decoded-point global max. Segmentation reads all states at a query coordinate and concatenates the reads with the current point feature. Classification discards the points, summarizes the fixed states, and adds bounded fine/broad residuals to the middle-state summary.

## Required experiments

1. **Additivity and fixed memory — rerun after final segmentation training.** Verify ordered, shuffled, and chunked writes against one-shot construction. Report numerical error and the constant 255 KiB retained state against growing XYZ storage.
2. **Streaming state operations — rerun after final segmentation training.** Verify append, merge, removal, and exponential decay against direct reconstruction. Delta-task accuracy remains future work and must not be implied by this algebra test.
3. **Architecture audit — complete data, regenerated figure.** Use paired three-seed short runs to isolate: multiscale versus equal-radius partition states; partition versus mixed keys; classification residual versus direct concatenation; and middle-radius `25x` versus `1x`.
4. **Streaming efficiency — rerun after final segmentation training.** Measure state construction only, with device-resident inputs and synchronized timing. Compare one-frame incremental update against rebuilding all accumulated history.
5. **Main accuracy — incomplete.** Train the frozen segmentation model for seeds 0/1/2. Train the frozen classification model and both classification baselines for missing seeds 1/2. Report the lightweight local PointNet++ implementation as a controlled baseline rather than a canonical reproduction.
6. **Minimality ablation — required for the paper, not architecture search.** Remove the fine residual/read and broad residual/read separately under the frozen protocol. This establishes the individual value of the two side states without tuning a new model.
7. **Qualitative segmentation — deferred.** With the frozen checkpoint, show sparse-prefix predictions, correction of an old query after new evidence, and predictions at coordinates absent from the observed stream.

## Figure decision

- Figures 1, 2, and 4 keep their design but must be regenerated from the final equal-state segmentation checkpoint because the retained state size changed.
- Figure 3 is redesigned around the four decisions that define the frozen architecture.
- The historical max-based XYZ+normal table remains an ablation/upper-bound result and is excluded from the main model table.

No three-seed mean may be reported until all three full runs are present. Short-protocol results must be labelled as screening ablations.
