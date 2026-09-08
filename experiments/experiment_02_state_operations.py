"""Experiment 2: exact FWP append, merge, removal, and exponential decay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fwpnet_core.data import load_shapenet_part
from fwpnet_core.ops import apply_pointwise, seed_everything
from fwpnet_core.segmentation_experiment import SEGMENTATION_MODELS


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="fwp_dual_read_partition_multiscale_seg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--decay", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def flatten(states):
    return torch.cat([state.flatten(1) for state in states], dim=1)


def relative_error(states, reference):
    numerator = (flatten(states) - flatten(reference)).norm(dim=1)
    denominator = flatten(reference).norm(dim=1).clamp_min(1e-30)
    return numerator / denominator


@torch.no_grad()
def main():
    args = arguments()
    if args.num_points % args.frames:
        raise ValueError("num-points must be divisible by frames")
    if not 0.0 < args.decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    seed_everything(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = SEGMENTATION_MODELS[args.model]().to(device).eval()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    points, _, _ = load_shapenet_part(
        "test", limit=args.samples, num_points=args.num_points
    )
    points = points.to(device)
    blocks = [model.encoder.block, model.local_fwp, model.global_fwp]
    features = [apply_pointwise(block.pre, points) for block in blocks]

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 7)
    permutations = torch.stack(
        [torch.randperm(args.num_points, generator=generator) for _ in range(args.samples)]
    ).to(device)
    points = torch.gather(
        points, 1, permutations.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    )
    features = [
        torch.gather(
            feature, 1,
            permutations.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        )
        for feature in features
    ]
    frame_size = args.num_points // args.frames
    point_frames = points.split(frame_size, dim=1)
    feature_frames = [feature.split(frame_size, dim=1) for feature in features]
    frame_states = [
        [
            block.memory(point_frames[t], feature_frames[b][t])
            for b, block in enumerate(blocks)
        ]
        for t in range(args.frames)
    ]

    zeros = [torch.zeros_like(state) for state in frame_states[0]]
    append = [state.clone() for state in zeros]
    decay_state = [state.clone() for state in zeros]
    append_curve, merge_curve, decay_curve = [], [], []
    additive_norm, decay_norm = [], []
    first_norm = flatten(frame_states[0]).norm(dim=1).clamp_min(1e-30)

    for t in range(1, args.frames + 1):
        append = [old + new for old, new in zip(append, frame_states[t - 1])]
        decay_state = [
            args.decay * old + new
            for old, new in zip(decay_state, frame_states[t - 1])
        ]
        end = t * frame_size
        direct = [
            block.memory(points[:, :end], feature[:, :end])
            for block, feature in zip(blocks, features)
        ]
        append_curve.append(relative_error(append, direct))

        weights = torch.cat(
            [
                torch.full(
                    (args.samples, frame_size, 1),
                    args.decay ** (t - 1 - frame),
                    device=device,
                    dtype=points.dtype,
                )
                for frame in range(t)
            ],
            dim=1,
        )
        direct_decay = [
            block.memory(points[:, :end], feature[:, :end], weights)
            for block, feature in zip(blocks, features)
        ]
        decay_curve.append(relative_error(decay_state, direct_decay))
        additive_norm.append(float((flatten(append).norm(dim=1) / first_norm).mean().cpu()))
        decay_norm.append(
            float((flatten(decay_state).norm(dim=1) / first_norm).mean().cpu())
        )

        if t >= 2:
            middle = (t // 2) * frame_size
            left = [
                block.memory(points[:, :middle], feature[:, :middle])
                for block, feature in zip(blocks, features)
            ]
            right = [
                block.memory(points[:, middle:end], feature[:, middle:end])
                for block, feature in zip(blocks, features)
            ]
            merge_curve.append(relative_error(
                [a + b for a, b in zip(left, right)], direct
            ))

    full = [state.clone() for state in append]
    removed = [state.clone() for state in full]
    remove_curve = []
    for count in range(1, args.frames):
        removed = [
            old - frame
            for old, frame in zip(removed, frame_states[args.frames - count])
        ]
        end = (args.frames - count) * frame_size
        direct = [
            block.memory(points[:, :end], feature[:, :end])
            for block, feature in zip(blocks, features)
        ]
        remove_curve.append(relative_error(removed, direct))

    curves = {
        "append": torch.cat(append_curve),
        "merge": torch.cat(merge_curve),
        "remove": torch.cat(remove_curve),
        "decay": torch.cat(decay_curve),
    }
    operations = {
        name: {
            "mean_relative_error": float(values.mean().cpu()),
            "max_relative_error": float(values.max().cpu()),
        }
        for name, values in curves.items()
    }
    result = {
        "experiment": "PointProgrammerNet streaming state operations",
        "model": f"PointProgrammerNet ({args.model} checkpoint)",
        "samples": args.samples,
        "points_per_sample": args.num_points,
        "frames": args.frames,
        "points_per_frame": frame_size,
        "decay": args.decay,
        "device": str(device),
        "operations": operations,
        "state_norm_curve": [
            {
                "frame": frame,
                "additive": additive_norm[frame - 1],
                "decay": decay_norm[frame - 1],
            }
            for frame in range(1, args.frames + 1)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
