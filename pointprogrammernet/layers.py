"""Continuous additive second-order FWP state used by both tasks."""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .ops import apply_pointwise


def make_norm(kind: str):
    if kind == "bn":
        return nn.BatchNorm1d
    if kind == "ln":
        return nn.LayerNorm
    raise ValueError(f"unknown normalization: {kind}")


class SecondOrderPartitionFWP(nn.Module):
    """Fixed-row, continuous, additive memory of local second-order statistics.

    A point distributes unit mass across all rows using a normalized RBF key.
    The value contains mass, position, learned content, position-content cross
    moments, spatial second moments, and diagonal content second moments.
    """

    def __init__(
        self,
        output_width: int,
        radius: float,
        rows: int = 128,
        content_width: int = 32,
        input_width: int = 3,
        chunk_size: int = 2048,
        norm: str = "bn",
        kexp: float = 0.5,
        decode: bool = True,
    ):
        super().__init__()
        if min(output_width, rows, content_width, input_width) <= 0 or radius <= 0:
            raise ValueError("FWP widths, rows, and radius must be positive")
        self.dk = rows
        self.cm = content_width
        self.chunk = chunk_size
        self.kexp = kexp
        self.freq = nn.Parameter(torch.empty(rows, 3))
        self.logs = nn.Parameter(torch.tensor(math.log(1.0 / radius)))
        initial = torch.quasirandom.SobolEngine(3, scramble=False).draw(rows) * 1.8 - 0.9
        with torch.no_grad():
            self.freq.copy_(torch.atanh(initial / 0.95))

        norm_layer = make_norm(norm)
        self.pre = nn.Sequential(
            nn.Linear(input_width, content_width),
            norm_layer(content_width),
            nn.ReLU(),
            nn.Linear(content_width, content_width),
            norm_layer(content_width),
            nn.ReLU(),
        )
        self.vdim = 1 + 3 + content_width + 3 * content_width + 6 + content_width
        self.mlp = (
            nn.Sequential(
                nn.Linear(self.vdim, output_width),
                norm_layer(output_width),
                nn.ReLU(),
                nn.Linear(output_width, output_width),
                norm_layer(output_width),
                nn.ReLU(),
            )
            if decode
            else nn.Identity()
        )

    def radius(self) -> float:
        return 1.0 / self.logs.detach().exp().item()

    def spatial_centers(self) -> torch.Tensor:
        return 0.95 * torch.tanh(self.freq)

    def key_logits(self, points: torch.Tensor) -> torch.Tensor:
        inverse_radius = self.logs.exp()
        offset = points.unsqueeze(-2) - self.spatial_centers()
        return -self.kexp * (inverse_radius * offset).square().sum(-1)

    def key(self, points: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.key_logits(points), dim=-1)

    def values(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x, y, z = points.unbind(-1)
        spatial_second = torch.stack(
            [x.square(), y.square(), z.square(), x * y, x * z, y * z], dim=-1
        )
        return torch.cat(
            [
                torch.ones_like(features[..., :1]),
                points,
                features,
                (points.unsqueeze(-1) * features.unsqueeze(-2)).flatten(-2),
                spatial_second,
                features.square(),
            ],
            dim=-1,
        )

    def _partial(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.key(points).transpose(1, 2) @ self.values(points, features)

    def memory(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Construct the additive state K^T V."""
        count = points.shape[1]
        if not self.chunk or count <= self.chunk:
            return self._partial(points, features)
        state = 0
        for start in range(0, count, self.chunk):
            p = points[:, start : start + self.chunk]
            f = features[:, start : start + self.chunk]
            part = (
                checkpoint(self._partial, p, f, use_reentrant=False)
                if self.training
                else self._partial(p, f)
            )
            state = state + part
        return state

    def write(self, points: torch.Tensor) -> torch.Tensor:
        return self.memory(points, apply_pointwise(self.pre, points))

    def decode(self, raw: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        density_raw = raw[..., :1]
        floor = density_raw.detach().mean(1, keepdim=True) * 1e-3 + 1e-20
        density = density_raw.clamp_min(floor)
        centroid = raw[..., 1:4] / density
        feature_mean = raw[..., 4 : 4 + self.cm] / density
        offset = 4 + self.cm
        absolute_cross = raw[..., offset : offset + 3 * self.cm].reshape(
            *raw.shape[:2], 3, self.cm
        ) / density.unsqueeze(-1)
        relative_cross = absolute_cross - queries.unsqueeze(-1) * feature_mean.unsqueeze(-2)
        offset += 3 * self.cm
        spatial_second = raw[..., offset : offset + 6] / density
        mx, my, mz = centroid.unbind(-1)
        centered_spatial = torch.stack(
            [
                (spatial_second[..., 0] - mx.square()).clamp_min(0.0),
                (spatial_second[..., 1] - my.square()).clamp_min(0.0),
                (spatial_second[..., 2] - mz.square()).clamp_min(0.0),
                spatial_second[..., 3] - mx * my,
                spatial_second[..., 4] - mx * mz,
                spatial_second[..., 5] - my * mz,
            ],
            dim=-1,
        )
        offset += 6
        feature_second = raw[..., offset : offset + self.cm] / density
        feature_std = (feature_second - feature_mean.square()).clamp_min(0.0).add(1e-8).sqrt()
        return torch.cat(
            [
                feature_mean,
                centroid - queries,
                torch.log(density),
                relative_cross.flatten(-2),
                centered_spatial,
                feature_std,
            ],
            dim=-1,
        )

    def read(self, state: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        raw = self.key(queries) @ state
        return apply_pointwise(self.mlp, self.decode(raw, queries))


def decode_state_rows(state: torch.Tensor, content_width: int = 32) -> torch.Tensor:
    """Decode every fixed state row into coordinate-free statistics."""
    density_raw = state[..., :1]
    floor = density_raw.detach().abs().mean(1, keepdim=True) * 1e-4 + 1e-20
    density = density_raw.clamp_min(floor)
    centroid = state[..., 1:4] / density
    feature_mean = state[..., 4 : 4 + content_width] / density
    offset = 4 + content_width
    absolute_cross = state[..., offset : offset + 3 * content_width].reshape(
        *state.shape[:2], 3, content_width
    ) / density.unsqueeze(-1)
    offset += 3 * content_width
    spatial_second = state[..., offset : offset + 6] / density
    offset += 6
    feature_second = state[..., offset : offset + content_width] / density
    mx, my, mz = centroid.unbind(-1)
    centered_spatial = torch.stack(
        [
            (spatial_second[..., 0] - mx.square()).clamp_min(0.0),
            (spatial_second[..., 1] - my.square()).clamp_min(0.0),
            (spatial_second[..., 2] - mz.square()).clamp_min(0.0),
            spatial_second[..., 3] - mx * my,
            spatial_second[..., 4] - mx * mz,
            spatial_second[..., 5] - my * mz,
        ],
        dim=-1,
    )
    feature_std = (feature_second - feature_mean.square()).clamp_min(0.0).add(1e-8).sqrt()
    centered_cross = absolute_cross - centroid.unsqueeze(-1) * feature_mean.unsqueeze(-2)
    relative_density = density / density.mean(1, keepdim=True).clamp_min(1e-20)
    return torch.cat(
        [
            feature_mean,
            centroid,
            relative_density.clamp_min(1e-20).log(),
            centered_spatial,
            feature_std,
            centered_cross.flatten(-2),
        ],
        dim=-1,
    )
