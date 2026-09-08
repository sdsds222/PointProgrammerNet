"""Verify additivity, permutation invariance, merging, and fixed state size."""

import json

import torch

from pointprogrammernet import PointProgrammerClassifier, PointProgrammerSegmenter


@torch.no_grad()
def verify(model, points):
    model.eval()
    direct = model.build_states(points)
    chunks = [model.build_states(chunk) for chunk in points.split(64, dim=1)]
    merged = model.merge_states(*chunks)
    permutation = torch.randperm(points.shape[1], device=points.device)
    shuffled = model.build_states(points[:, permutation])
    denominator = torch.cat([state.flatten() for state in direct]).norm().clamp_min(1e-30)

    def error(candidate):
        difference = torch.cat(
            [(left - right).flatten() for left, right in zip(candidate, direct)]
        ).norm()
        return float(difference / denominator)

    state_bytes = sum(state[0].numel() * state.element_size() for state in direct)
    return {
        "chunk_merge_relative_error": error(merged),
        "permutation_relative_error": error(shuffled),
        "state_shapes": [list(state.shape[1:]) for state in direct],
        "retained_bytes_per_stream": state_bytes,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(2026)
    points = torch.randn(2, 1024, 3, generator=generator).to(device)
    result = {
        "device": str(device),
        "segmentation": verify(PointProgrammerSegmenter().to(device), points),
        "classification": verify(PointProgrammerClassifier().to(device), points),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
