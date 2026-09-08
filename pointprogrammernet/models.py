"""Frozen segmentation and classification architectures used in the paper."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SecondOrderPartitionFWP, decode_state_rows, make_norm
from .ops import apply_pointwise


def _point_mlp(input_width: int, output_width: int, depth: int, norm: str = "bn"):
    layers = []
    norm_layer = make_norm(norm)
    for _ in range(depth):
        layers += [nn.Linear(input_width, output_width), norm_layer(output_width), nn.ReLU()]
        input_width = output_width
    return nn.Sequential(*layers)


class _MiddleEncoder(nn.Module):
    """Container kept compatible with released segmentation checkpoints."""

    def __init__(self):
        super().__init__()
        self.block = SecondOrderPartitionFWP(64, radius=0.20)
        self.point_mlp = _point_mlp(64, 160, 3)
        self.head = nn.Identity()


class _SegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_width = 32
        self.context_width = 480
        self.num_categories = 16
        norm = make_norm("bn")
        self.classifier = nn.Sequential(
            nn.Linear(32 + 480 + 16, 256),
            norm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            norm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 50),
        )

    def forward(self, point_features, context, categories):
        category = F.one_hot(categories, num_classes=16).to(context.dtype)
        category = category.unsqueeze(1).expand(-1, context.shape[1], -1)
        return apply_pointwise(
            self.classifier, torch.cat([point_features, context, category], dim=-1)
        )


class PointProgrammerSegmenter(nn.Module):
    """Three equal 128x170 states with direct three-way coordinate readout."""

    model_name = "fwp_dual_read_partition_multiscale_seg"
    initial_radii = (0.08, 0.20, 0.60)
    main_radius_lr_multiplier = 1.0

    def __init__(self):
        super().__init__()
        self.encoder = _MiddleEncoder()
        self.head = _SegmentationHead()
        self.local_fwp = SecondOrderPartitionFWP(160, radius=0.08)
        self.global_fwp = SecondOrderPartitionFWP(160, radius=0.60)

    def blocks(self):
        return (self.local_fwp, self.encoder.block, self.global_fwp)

    def build_states(self, points: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must be [B,N,3]")
        return tuple(block.write(points) for block in self.blocks())

    @staticmethod
    def merge_states(*groups):
        return tuple(sum(parts) for parts in zip(*groups))

    def read_context(self, queries: torch.Tensor, states) -> torch.Tensor:
        fine = self.local_fwp.read(states[0], queries)
        middle = self.encoder.block.read(states[1], queries)
        middle = apply_pointwise(self.encoder.point_mlp, middle)
        broad = self.global_fwp.read(states[2], queries)
        return torch.cat([middle, fine, broad], dim=-1)

    def predict_from_states(self, queries, categories, states):
        clean = apply_pointwise(self.encoder.block.pre, queries)
        return self.head(clean, self.read_context(queries, states), categories)

    def forward(self, points, categories, fps_indices=None):
        del fps_indices
        return self.predict_from_states(points, categories, self.build_states(points))

    def kernel_parameters(self):
        # Match the reported optimizer: the middle radius has its own
        # no-weight-decay group, but uses the ordinary 1x learning rate.
        return [self.encoder.block.logs]

    def receptive_fields(self):
        return [block.radius() for block in self.blocks()]


class PointProgrammerClassifier(nn.Module):
    """State-only classifier with middle-reference bounded residual readout."""

    model_name = "pointprogrammer_state_multiscale_residual_main_r25"
    main_radius_lr_multiplier = 25.0

    def __init__(self, num_classes: int = 40, head_width: int = 256):
        super().__init__()
        self.fine_fwp = SecondOrderPartitionFWP(64, 0.08, decode=False)
        self.main_fwp = SecondOrderPartitionFWP(64, 0.20, decode=False)
        self.broad_fwp = SecondOrderPartitionFWP(64, 0.60, decode=False)
        self.cm = 32
        self.stat_width = 170
        token_width, summaries, state_width = 128, 2, 256
        self.tokenizers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.stat_width, token_width),
                    nn.LayerNorm(token_width),
                    nn.GELU(),
                    nn.Linear(token_width, token_width),
                    nn.LayerNorm(token_width),
                    nn.GELU(),
                )
                for _ in self.blocks()
            ]
        )
        self.summary_scores = nn.ModuleList(
            [nn.Linear(token_width, summaries) for _ in self.blocks()]
        )
        self.summary_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(summaries * token_width, state_width),
                    nn.LayerNorm(state_width),
                    nn.GELU(),
                )
                for _ in self.blocks()
            ]
        )
        self.auxiliary_projection = nn.Linear(2 * state_width, state_width)
        self.scale_mix_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.output_norm = nn.LayerNorm(state_width)
        self.head = nn.Sequential(
            nn.Linear(state_width, head_width),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(head_width, num_classes),
        )
        nn.init.zeros_(self.auxiliary_projection.weight)
        nn.init.zeros_(self.auxiliary_projection.bias)

    def blocks(self):
        return (self.fine_fwp, self.main_fwp, self.broad_fwp)

    def build_states(self, points: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must be [B,N,3]")
        return tuple(block.write(points) for block in self.blocks())

    @staticmethod
    def merge_states(*groups):
        return tuple(sum(parts) for parts in zip(*groups))

    def _branch_summary(self, state, tokenizer, scorer, summary_mlp):
        tokens = apply_pointwise(tokenizer, decode_state_rows(state, self.cm))
        weights = F.softmax(apply_pointwise(scorer, tokens), dim=1)
        summaries = torch.einsum("bks,bkh->bsh", weights, tokens)
        return summary_mlp(summaries.flatten(1))

    def components(self, states):
        summaries = [
            self._branch_summary(state, tokenizer, scorer, summary_mlp)
            for state, tokenizer, scorer, summary_mlp in zip(
                states, self.tokenizers, self.summary_scores, self.summary_mlps
            )
        ]
        fine, middle, broad = summaries
        raw_delta = self.auxiliary_projection(torch.cat([fine, broad], dim=-1))
        delta = torch.sigmoid(self.scale_mix_logit) * torch.tanh(raw_delta)
        return middle, delta

    def classify_states(self, states):
        middle, delta = self.components(states)
        return self.head(self.output_norm(middle + delta))

    def forward(self, points, fps_indices=None):
        del fps_indices
        states = self.build_states(points)
        del points
        return self.classify_states(states)

    def kernel_parameters(self):
        return [self.main_fwp.logs]

    def receptive_fields(self):
        return [block.radius() for block in self.blocks()]
