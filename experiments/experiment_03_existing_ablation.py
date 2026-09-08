"""Collect the screening ablations that define the frozen architecture."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEG_SOURCE = ROOT / "artifacts_seg/dual_read_concat_probe/segmentation_results.json"
CLS_FUSION_SOURCE = ROOT / "artifacts/pointprogrammer_multiscale_concat_probe/results.json"
CLS_RATE_SOURCE = ROOT / "artifacts/pointprogrammer_main_radius_rate_probe/results.json"

SEG_EXPECTED = {
    "epochs": 5,
    "num_points": 1024,
    "train_shapes": 2048,
    "test_shapes": 512,
    "learning_rate": 0.003,
    "batch_size": 16,
}
CLS_EXPECTED = {
    "epochs": 5,
    "num_points": 1024,
    "train_shapes": 2048,
    "test_shapes": 512,
    "learning_rate": 0.003,
    "batch_size": 64,
}

COMPARISONS = [
    {
        "task": "Segmentation",
        "label": "Multiscale radii",
        "source": SEG_SOURCE,
        "expected": SEG_EXPECTED,
        "control": "fwp_dual_read_symmetric_state_seg",
        "treatment": "fwp_dual_read_partition_multiscale_seg",
        "metrics": [
            ("instance_miou", "Instance mIoU"),
            ("category_miou", "Category mIoU"),
        ],
    },
    {
        "task": "Segmentation",
        "label": "Unified partition key",
        "source": SEG_SOURCE,
        "expected": SEG_EXPECTED,
        "control": "fwp_dual_read_equal_state_width_uniform_lr_seg",
        "treatment": "fwp_dual_read_partition_multiscale_seg",
        "metrics": [
            ("instance_miou", "Instance mIoU"),
            ("category_miou", "Category mIoU"),
        ],
    },
    {
        "task": "Classification",
        "label": "Residual state readout",
        "source": CLS_FUSION_SOURCE,
        "expected": CLS_EXPECTED,
        "control": "pointprogrammer_state_multiscale_concat",
        "treatment": "pointprogrammer_state_multiscale_residual",
        "metrics": [
            ("clean_accuracy", "Clean accuracy"),
            ("density_accuracy", "Density accuracy"),
        ],
    },
    {
        "task": "Classification",
        "label": "Middle radius LR 25x",
        "source": CLS_RATE_SOURCE,
        "expected": CLS_EXPECTED,
        "control": "pointprogrammer_state_multiscale_residual",
        "treatment": "pointprogrammer_state_multiscale_residual_main_r25",
        "metrics": [
            ("clean_accuracy", "Clean accuracy"),
            ("density_accuracy", "Density accuracy"),
        ],
    },
]


def load_selected(source: Path, model: str, expected: dict) -> dict[int, dict]:
    rows = json.loads(source.read_text(encoding="utf-8"))
    selected = {}
    for row in rows:
        identity = row.get("run_identity", {})
        if row.get("model") != model:
            continue
        if any(identity.get(key) != value for key, value in expected.items()):
            continue
        seed = int(row["seed"])
        if seed in selected:
            raise RuntimeError(f"duplicate {model} seed {seed} in {source}")
        selected[seed] = row
    if sorted(selected) != [0, 1, 2]:
        raise RuntimeError(f"missing paired seeds for {model}: {sorted(selected)}")
    return selected


def main() -> None:
    rows = []
    for spec in COMPARISONS:
        control = load_selected(spec["source"], spec["control"], spec["expected"])
        treatment = load_selected(
            spec["source"], spec["treatment"], spec["expected"]
        )
        metrics = []
        for key, label in spec["metrics"]:
            changes = [
                float(treatment[seed][key]) - float(control[seed][key])
                for seed in range(3)
            ]
            metrics.append(
                {
                    "key": key,
                    "label": label,
                    "paired_change": changes,
                    "mean_change": statistics.mean(changes),
                    "sd_change": statistics.pstdev(changes),
                }
            )
        rows.append(
            {
                "task": spec["task"],
                "label": spec["label"],
                "control_model": spec["control"],
                "treatment_model": spec["treatment"],
                "source": str(spec["source"].resolve()),
                "metrics": metrics,
            }
        )

    payload = {
        "experiment": "Frozen PointProgrammerNet architecture audit",
        "seeds": [0, 1, 2],
        "note": "Five-epoch screening protocol; paired changes in percentage points.",
        "comparisons": rows,
    }
    output = ROOT / "paper/results/experiment_03_existing_ablation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
