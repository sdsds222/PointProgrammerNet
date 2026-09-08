"""Benchmark incremental FWP state updates against rebuilding all history."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fwpnet_core.data import load_shapenet_part
from fwpnet_core.ops import apply_pointwise, seed_everything
from fwpnet_core.segmentation_experiment import SEGMENTATION_MODELS


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="fwp_dual_read_partition_multiscale_seg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--points-per-frame", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def flatten(states):
    return torch.cat([state.flatten(1) for state in states], dim=1)


@torch.inference_mode()
def main():
    args = arguments()
    seed_everything(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = SEGMENTATION_MODELS[args.model]().to(device).eval()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    blocks = [model.encoder.block, model.local_fwp, model.global_fwp]

    needed = args.streams * args.frames
    points, _, _ = load_shapenet_part(
        "test", limit=needed, num_points=args.points_per_frame
    )
    frames = points.reshape(
        args.streams, args.frames, args.points_per_frame, 3
    ).to(device)

    def build_state(values):
        return [
            block.memory(values, apply_pointwise(block.pre, values))
            for block in blocks
        ]

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure_once(function):
        synchronize()
        start = time.perf_counter()
        for _ in range(args.repeats):
            function()
        synchronize()
        return (time.perf_counter() - start) * 1000 / args.repeats

    def measure_pair(incremental, rebuild):
        for _ in range(3):
            incremental(); rebuild()
        synchronize()
        incremental_samples, rebuild_samples = [], []
        for trial in range(args.trials):
            order = ((incremental, incremental_samples), (rebuild, rebuild_samples))
            if trial % 2:
                order = tuple(reversed(order))
            for function, destination in order:
                destination.append(measure_once(function))
        return (
            statistics.median(incremental_samples),
            statistics.median(rebuild_samples),
            incremental_samples,
            rebuild_samples,
        )

    selected_frames = [1, 2, 4, 8, 16, 32]
    selected_frames = [value for value in selected_frames if value <= args.frames]
    frame_states = [build_state(frames[:, index]) for index in range(args.frames)]
    running = [torch.zeros_like(state) for state in frame_states[0]]
    snapshots = {}
    previous_snapshots = {}
    for index, state in enumerate(frame_states, start=1):
        previous_snapshots[index] = [value.clone() for value in running]
        running = [old + new for old, new in zip(running, state)]
        if index in selected_frames:
            snapshots[index] = [value.clone() for value in running]

    rows = []
    for count in selected_frames:
        new_frame = frames[:, count - 1]
        history = frames[:, :count].reshape(
            args.streams, count * args.points_per_frame, 3
        )
        previous = previous_snapshots[count]

        def incremental():
            update = build_state(new_frame)
            return [old + new for old, new in zip(previous, update)]

        def rebuild():
            return build_state(history)

        rebuilt = rebuild()
        error = (
            (flatten(snapshots[count]) - flatten(rebuilt)).norm(dim=1)
            / flatten(rebuilt).norm(dim=1).clamp_min(1e-30)
        )
        (
            incremental_ms,
            rebuild_ms,
            incremental_trials,
            rebuild_trials,
        ) = measure_pair(incremental, rebuild)
        rows.append(
            {
                "frames": count,
                "cumulative_points": count * args.points_per_frame,
                "incremental_ms_per_batch": incremental_ms,
                "rebuild_ms_per_batch": rebuild_ms,
                "speedup": rebuild_ms / incremental_ms,
                "max_relative_state_error": float(error.max().cpu()),
                "incremental_trials_ms": incremental_trials,
                "rebuild_trials_ms": rebuild_trials,
            }
        )

    hardware = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )
    result = {
        "experiment": "PointProgrammerNet incremental state efficiency",
        "scope": "FWP state construction only; device-resident inputs",
        "model": f"PointProgrammerNet ({args.model} checkpoint)",
        "device": str(device),
        "hardware": hardware,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "input_source": "ShapeNetPart test coordinates concatenated into timing streams",
        "streams_per_batch": args.streams,
        "points_per_frame": args.points_per_frame,
        "repeats": args.repeats,
        "trials": args.trials,
        "warmup_iterations": 3,
        "timings": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
