"""Training and evaluation routines for the two frozen architectures."""

import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from .data import density_gradient, load_modelnet40, load_shapenet_part
from .models import PointProgrammerClassifier, PointProgrammerSegmenter
from .ops import seed_everything


SHAPENET_PARTS_BY_CATEGORY = (
    (0, 1, 2, 3), (4, 5), (6, 7), (8, 9, 10, 11),
    (12, 13, 14, 15), (16, 17, 18), (19, 20, 21), (22, 23),
    (24, 25, 26, 27), (28, 29), (30, 31, 32, 33, 34, 35),
    (36, 37), (38, 39, 40), (41, 42, 43), (44, 45, 46),
    (47, 48, 49),
)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _optimizer(model, learning_rate: float, epochs: int):
    accelerated = model.kernel_parameters()
    accelerated_ids = {id(parameter) for parameter in accelerated}
    groups = [{"params": [p for p in model.parameters() if id(p) not in accelerated_ids]}]
    if accelerated:
        groups.append(
            {
                "params": accelerated,
                "lr": learning_rate * model.main_radius_lr_multiplier,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(groups, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler


def _part_mask() -> torch.Tensor:
    mask = torch.zeros(16, 50, dtype=torch.bool)
    for category, parts in enumerate(SHAPENET_PARTS_BY_CATEGORY):
        mask[category, torch.tensor(parts)] = True
    return mask


def _masked_logits(logits, categories, mask):
    return logits.masked_fill(~mask.to(logits.device)[categories].unsqueeze(1), -1e4)


@torch.no_grad()
def evaluate_segmentation(model, points, categories, parts, device, batch_size):
    model.eval()
    mask = _part_mask()
    correct = total = 0
    instance_ious = []
    category_ious = [[] for _ in range(16)]
    for start in range(0, len(points), batch_size):
        p = points[start : start + batch_size].to(device)
        c = categories[start : start + batch_size].to(device)
        y = parts[start : start + batch_size].to(device)
        prediction = _masked_logits(model(p, c), c, mask).argmax(-1)
        correct += (prediction == y).sum().item()
        total += y.numel()
        for index in range(len(p)):
            category = int(c[index])
            scores = []
            for part in SHAPENET_PARTS_BY_CATEGORY[category]:
                predicted = prediction[index] == part
                target = y[index] == part
                union = (predicted | target).sum().item()
                scores.append(1.0 if union == 0 else (predicted & target).sum().item() / union)
            shape_iou = statistics.mean(scores)
            instance_ious.append(shape_iou)
            category_ious[category].append(shape_iou)
    category_means = [statistics.mean(values) for values in category_ious if values]
    return {
        "point_accuracy": 100.0 * correct / total,
        "instance_miou": 100.0 * statistics.mean(instance_ious),
        "category_miou": 100.0 * statistics.mean(category_means),
    }


def train_segmentation(args):
    device = choose_device(args.device)
    train_points, train_categories, train_parts = load_shapenet_part(
        "train", args.data_dir, args.train_limit, args.num_points
    )
    test_points, test_categories, test_parts = load_shapenet_part(
        "test", args.data_dir, args.test_limit, args.num_points
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        seed_everything(seed)
        model = PointProgrammerSegmenter().to(device)
        optimizer, scheduler = _optimizer(model, args.learning_rate, args.epochs)
        loss_fn = nn.CrossEntropyLoss()
        mask = _part_mask()
        started = time.time()
        for epoch in range(args.epochs):
            model.train()
            permutation = torch.randperm(len(train_points))
            loss_sum = batches = 0
            for start in range(0, len(permutation), args.batch_size):
                index = permutation[start : start + args.batch_size]
                if len(index) < 2:
                    continue
                points = train_points[index].to(device)
                points = points + torch.randn_like(points) * 0.01
                categories = train_categories[index].to(device)
                labels = train_parts[index].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = _masked_logits(model(points, categories), categories, mask)
                loss = loss_fn(logits.transpose(1, 2), labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                batches += 1
            scheduler.step()
            print(f"seg seed={seed} epoch={epoch+1}/{args.epochs} loss={loss_sum/batches:.4f}", flush=True)
        metrics = evaluate_segmentation(
            model, test_points, test_categories, test_parts, device, args.eval_batch_size
        )
        row = {
            "model": model.model_name,
            "seed": seed,
            "epochs": args.epochs,
            "parameters": sum(p.numel() for p in model.parameters()),
            "radii": model.receptive_fields(),
            "elapsed_seconds": time.time() - started,
            **metrics,
        }
        rows.append(row)
        torch.save(model.state_dict(), output / f"{model.model_name}_seed{seed}.pt")
        print(json.dumps(row), flush=True)
    (output / "segmentation_results.json").write_text(json.dumps(rows, indent=2))


@torch.no_grad()
def evaluate_classification(model, points, labels, device, batch_size):
    model.eval()
    correct = 0
    for start in range(0, len(points), batch_size):
        prediction = model(points[start : start + batch_size].to(device)).argmax(-1)
        correct += (prediction == labels[start : start + batch_size].to(device)).sum().item()
    return 100.0 * correct / len(points)


def train_classification(args):
    device = choose_device(args.device)
    train_points, train_labels = load_modelnet40("train", args.data_dir, args.train_limit)
    test_full, test_labels = load_modelnet40("test", args.data_dir, args.test_limit)
    train_points = train_points[:, : args.num_points].contiguous()
    test_points = test_full[:, : args.num_points].contiguous()
    density_points = density_gradient(test_full, args.num_points)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        seed_everything(seed)
        model = PointProgrammerClassifier().to(device)
        optimizer, scheduler = _optimizer(model, args.learning_rate, args.epochs)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        started = time.time()
        for epoch in range(args.epochs):
            model.train()
            permutation = torch.randperm(len(train_points))
            for start in range(0, len(permutation), args.batch_size):
                index = permutation[start : start + args.batch_size]
                points = train_points[index].to(device)
                points = points + torch.randn_like(points) * 0.01
                labels = train_labels[index].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(points), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            print(f"cls seed={seed} epoch={epoch+1}/{args.epochs} loss={loss.item():.4f}", flush=True)
        clean = evaluate_classification(model, test_points, test_labels, device, args.eval_batch_size)
        shifted = evaluate_classification(model, density_points, test_labels, device, args.eval_batch_size)
        row = {
            "model": model.model_name,
            "seed": seed,
            "epochs": args.epochs,
            "clean_accuracy": clean,
            "density_accuracy": shifted,
            "density_drop": clean - shifted,
            "parameters": sum(p.numel() for p in model.parameters()),
            "radii": model.receptive_fields(),
            "elapsed_seconds": time.time() - started,
        }
        rows.append(row)
        torch.save(model.state_dict(), output / f"{model.model_name}_seed{seed}.pt")
        print(json.dumps(row), flush=True)
    (output / "classification_results.json").write_text(json.dumps(rows, indent=2))
