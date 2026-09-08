"""Experiment 1: additive streaming equivalence and fixed FWP state size."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="fwp_dual_read_partition_multiscale_seg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def flat_state(states: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([state.flatten(1) for state in states], dim=1)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    model = SEGMENTATION_MODELS[args.model]().to(device).eval()
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    points, _, _ = load_shapenet_part(
        "test", limit=args.samples, num_points=args.num_points
    )
    points = points.to(device)
    blocks = [model.encoder.block, model.local_fwp, model.global_fwp]
    names = ["main", "fine", "broad"]

    # The pointwise encoders are evaluated once. The experiment then varies only
    # how the exact same per-point writes are grouped and ordered.
    features = [apply_pointwise(block.pre, points) for block in blocks]
    reference_states = [
        block.memory(points, feature) for block, feature in zip(blocks, features)
    ]
    reference = flat_state(reference_states)
    reference_norm = reference.norm(dim=1).clamp_min(1e-30)

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    permutations = torch.stack(
        [torch.randperm(args.num_points, generator=generator) for _ in range(args.samples)]
    ).to(device)
    shuffled_points = torch.gather(
        points, 1, permutations.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    )
    shuffled_features = [
        torch.gather(
            feature,
            1,
            permutations.unsqueeze(-1).expand(-1, -1, feature.shape[-1]),
        )
        for feature in features
    ]

    def streamed_state(
        source_points: torch.Tensor,
        source_features: list[torch.Tensor],
        chunks: int,
    ) -> list[torch.Tensor]:
        point_chunks = source_points.chunk(chunks, dim=1)
        feature_chunks = [feature.chunk(chunks, dim=1) for feature in source_features]
        totals = [torch.zeros_like(state) for state in reference_states]
        for index in range(chunks):
            for block_index, block in enumerate(blocks):
                totals[block_index].add_(
                    block.memory(
                        point_chunks[index], feature_chunks[block_index][index]
                    )
                )
        return totals

    chunk_counts = [2, 4, 8, 16, 32]
    equivalence: dict[str, list[dict[str, float | int]]] = {
        "ordered": [],
        "shuffled": [],
    }
    per_state_max: dict[str, dict[str, float]] = {}
    for order, source_points, source_features in (
        ("ordered", points, features),
        ("shuffled", shuffled_points, shuffled_features),
    ):
        for chunks in chunk_counts:
            states = streamed_state(source_points, source_features, chunks)
            relative = (flat_state(states) - reference).norm(dim=1) / reference_norm
            equivalence[order].append(
                {
                    "chunks": chunks,
                    "mean_relative_error": float(relative.mean().cpu()),
                    "max_relative_error": float(relative.max().cpu()),
                }
            )
        last_states = streamed_state(source_points, source_features, chunk_counts[-1])
        per_state_max[order] = {
            name: float(
                ((state - target).flatten(1).norm(dim=1)
                 / target.flatten(1).norm(dim=1).clamp_min(1e-30)).max().cpu()
            )
            for name, state, target in zip(names, last_states, reference_states)
        }

    element_size = reference_states[0].element_size()
    state_shapes = {
        name: list(state.shape[1:]) for name, state in zip(names, reference_states)
    }
    state_elements = sum(state[0].numel() for state in reference_states)
    state_bytes = state_elements * element_size
    cumulative_points = [
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144
    ]
    raw_point_bytes = points.shape[-1] * element_size

    result = {
        "experiment": "PointProgrammerNet additive streaming and fixed state",
        "model": f"PointProgrammerNet ({args.model} checkpoint)",
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "dtype": str(reference.dtype),
        "samples": args.samples,
        "points_per_sample": args.num_points,
        "state_shapes": state_shapes,
        "state_elements_per_sample": state_elements,
        "state_bytes_per_sample": state_bytes,
        "state_kib_per_sample": state_bytes / 1024.0,
        "equivalence": equivalence,
        "per_state_max_error_at_32_chunks": per_state_max,
        "memory_curve": [
            {
                "cumulative_points": count,
                "pointprogrammer_state_bytes": state_bytes,
                "raw_xyz_bytes": count * raw_point_bytes,
            }
            for count in cumulative_points
        ],
        "raw_xyz_crossover_points": state_bytes / raw_point_bytes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
