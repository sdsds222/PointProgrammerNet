"""Validate final no-max segmentation and classification accuracy runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEG_EXPECTED = {
    "epochs": 20,
    "num_points": 1024,
    "train_shapes": 13972,
    "test_shapes": 2874,
    "learning_rate": 0.003,
    "batch_size": 16,
}
CLS_EXPECTED = {
    "epochs": 20,
    "num_points": 1024,
    "train_shapes": 9840,
    "test_shapes": 2468,
    "learning_rate": 0.003,
    "batch_size": 32,
}
SEG_MODELS = (
    ("pn2_seg", "Lightweight PointNet++"),
    ("fwp_dual_read_partition_multiscale_seg", "PointProgrammerNet"),
)
CLS_MODELS = (
    ("pn2_max", "Lightweight PointNet++"),
    ("pointnet_max_matched", "Parameter-matched PointNet"),
    (
        "pointprogrammer_state_multiscale_residual_main_r25",
        "PointProgrammerNet",
    ),
)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a result list in {path}")
    return payload


def select(rows: list[dict], model: str, expected: dict) -> list[dict]:
    selected = {}
    for row in rows:
        identity = row.get("run_identity", {})
        if row.get("model") != model:
            continue
        if any(identity.get(key) != value for key, value in expected.items()):
            continue
        seed = int(row["seed"])
        if seed in (0, 1, 2):
            if seed in selected:
                raise ValueError(f"duplicate full run for {model}, seed {seed}")
            selected[seed] = row
    return [selected[seed] for seed in sorted(selected)]


def metric(rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows]
    return {
        "values": values,
        "mean": statistics.mean(values),
        "population_sd": statistics.pstdev(values),
    }


def summarize_seg(rows: list[dict], model: str, label: str) -> dict:
    return {
        "model": model,
        "label": label,
        "seeds": [int(row["seed"]) for row in rows],
        "point_accuracy": metric(rows, "point_accuracy"),
        "instance_miou": metric(rows, "instance_miou"),
        "category_miou": metric(rows, "category_miou"),
        "parameters": int(rows[0]["parameters"]),
    }


def summarize_cls(rows: list[dict], model: str, label: str) -> dict:
    return {
        "model": model,
        "label": label,
        "seeds": [int(row["seed"]) for row in rows],
        "clean_accuracy": metric(rows, "clean_accuracy"),
        "density_accuracy": metric(rows, "density_accuracy"),
        "density_drop": metric(rows, "density_drop"),
        "parameters": int(rows[0]["parameters"]),
    }


def pm(value: dict) -> str:
    return f"{value['mean']:.2f} $\\pm$ {value['population_sd']:.2f}"


def write_seg_table(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{ShapeNetPart segmentation with XYZ input and 1,024 points. PointProgrammerNet performs no decoded-point global pooling. Values are mean $\pm$ population standard deviation over three seeds.}",
        r"\label{tab:segmentation_accuracy}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Point acc. (\%) & Ins. mIoU (\%) & Cat. mIoU (\%) & Params \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {pm(row['point_accuracy'])} & "
            f"{pm(row['instance_miou'])} & {pm(row['category_miou'])} & "
            f"{row['parameters']:,} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cls_table(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{ModelNet40 classification with 1,024 points. Density denotes evaluation under the fixed density-gradient shift. PointProgrammerNet retains only fixed-size FWP states and uses no global max.}",
        r"\label{tab:classification_accuracy}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Clean (\%) & Density (\%) & Drop (pp) & Params \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {pm(row['clean_accuracy'])} & "
            f"{pm(row['density_accuracy'])} & {pm(row['density_drop'])} & "
            f"{row['parameters']:,} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seg-baseline-results",
        type=Path,
        default=ROOT / "artifacts_seg/segmentation_results.json",
    )
    parser.add_argument(
        "--seg-final-results",
        type=Path,
        default=ROOT / "artifacts_seg/partition_multiscale_full/segmentation_results.json",
    )
    parser.add_argument(
        "--classification-baseline-results",
        type=Path,
        default=ROOT / "artifacts/pointprogrammer_state_mixer_full/results.json",
    )
    parser.add_argument(
        "--classification-final-results",
        type=Path,
        default=ROOT / "artifacts/pointprogrammer_multiscale_residual_r25_full/results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper/results/experiment_05_accuracy.json",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    seg_sources = {
        "pn2_seg": load_rows(args.seg_baseline_results),
        "fwp_dual_read_partition_multiscale_seg": load_rows(args.seg_final_results),
    }
    cls_sources = {
        "pn2_max": load_rows(args.classification_baseline_results),
        "pointnet_max_matched": load_rows(args.classification_baseline_results),
        "pointprogrammer_state_multiscale_residual_main_r25": load_rows(
            args.classification_final_results
        ),
    }
    segmentation, classification, missing = [], [], []
    for model, label in SEG_MODELS:
        rows = select(seg_sources[model], model, SEG_EXPECTED)
        absent = sorted({0, 1, 2} - {int(row["seed"]) for row in rows})
        if absent:
            missing.append({"task": "segmentation", "model": model, "seeds": absent})
        else:
            segmentation.append(summarize_seg(rows, model, label))
    for model, label in CLS_MODELS:
        rows = select(cls_sources[model], model, CLS_EXPECTED)
        absent = sorted({0, 1, 2} - {int(row["seed"]) for row in rows})
        if absent:
            missing.append({"task": "classification", "model": model, "seeds": absent})
        else:
            classification.append(summarize_cls(rows, model, label))

    payload = {
        "segmentation_protocol": {**SEG_EXPECTED, "seeds": [0, 1, 2]},
        "classification_protocol": {**CLS_EXPECTED, "seeds": [0, 1, 2]},
        "strict_no_max_models": [
            "fwp_dual_read_partition_multiscale_seg",
            "pointprogrammer_state_multiscale_residual_main_r25",
        ],
        "complete": not missing,
        "missing": missing,
        "segmentation": segmentation,
        "classification": classification,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not missing:
        write_seg_table(
            ROOT / "paper/tables/table01_segmentation_accuracy.tex", segmentation
        )
        write_cls_table(
            ROOT / "paper/tables/table02_classification_accuracy.tex", classification
        )
        print("complete: wrote final no-max accuracy tables")
    else:
        print(
            "incomplete: "
            + ", ".join(
                f"{item['task']}/{item['model']} seeds={item['seeds']}"
                for item in missing
            )
        )
        if not args.allow_incomplete:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
