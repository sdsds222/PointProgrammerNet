import json
import os
import urllib.request
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch

DATASET_NAME = "modelnet40_ply_hdf5_2048"
SHAPENET_PART_RAW_NAME = (
    "shapenetcore_partanno_segmentation_benchmark_v0_normal"
)
SHAPENET_PART_URL = (
    "https://huggingface.co/datasets/wangps/shapenet_segmentation/resolve/main/"
    f"{SHAPENET_PART_RAW_NAME}.zip?download=true"
)
SHAPENET_PART_CATEGORY_NAMES = (
    "Airplane",
    "Bag",
    "Cap",
    "Car",
    "Chair",
    "Earphone",
    "Guitar",
    "Knife",
    "Lamp",
    "Laptop",
    "Motorbike",
    "Mug",
    "Pistol",
    "Rocket",
    "Skateboard",
    "Table",
)


def resolve_data_dir(explicit: str | Path | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("MODELNET40_DIR"):
        candidates.append(Path(os.environ["MODELNET40_DIR"]))
    project = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            project / "data" / DATASET_NAME,
            project.parent / "fwpnet" / "data" / DATASET_NAME,
            Path.cwd() / "data" / DATASET_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("ply_data_train*.h5")):
            return candidate.resolve()
    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "ModelNet40 directory was not found. Checked:\n  " + checked
    )


def load_modelnet40(
    split: str,
    data_dir: str | Path | None = None,
    limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    root = resolve_data_dir(data_dir)
    files = sorted(root.glob(f"ply_data_{split}*.h5"))
    point_parts, label_parts = [], []
    loaded = 0
    for path in files:
        with h5py.File(path, "r") as handle:
            point_parts.append(np.asarray(handle["data"]))
            label_parts.append(np.asarray(handle["label"]).reshape(-1))
        loaded += len(point_parts[-1])
        if limit is not None and loaded >= limit:
            break
    if not point_parts:
        raise FileNotFoundError(f"no {split} HDF5 files found in {root}")
    points = np.concatenate(point_parts)
    labels = np.concatenate(label_parts)
    if limit is not None:
        points, labels = points[:limit], labels[:limit]
    points = points - points.mean(1, keepdims=True)
    scale = np.abs(points).max(axis=(1, 2), keepdims=True)
    points = points / np.maximum(scale, 1e-12)
    return torch.from_numpy(points).float(), torch.from_numpy(labels).long()


def resolve_shapenet_part_dir(explicit: str | Path | None = None) -> Path:
    candidates = []
    if explicit:
        explicit = Path(explicit)
        candidates.extend(
            [
                explicit,
                explicit / SHAPENET_PART_RAW_NAME,
                explicit / "shapenet_part_seg_hdf5_data",
                explicit / "hdf5_data",
            ]
        )
    if os.environ.get("SHAPENETPART_DIR"):
        candidates.append(Path(os.environ["SHAPENETPART_DIR"]))
    project = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            project / "data" / "shapenet_part_seg_hdf5_data",
            project / "data" / "hdf5_data",
            project / "data" / SHAPENET_PART_RAW_NAME,
            project.parent / "fwpnet" / "data" / "shapenet_part_seg_hdf5_data",
            project.parent / "fwpnet" / "data" / "hdf5_data",
            project.parent / "fwpnet" / "data" / SHAPENET_PART_RAW_NAME,
            Path.cwd() / "data" / "shapenet_part_seg_hdf5_data",
            Path.cwd() / "data" / "hdf5_data",
            Path.cwd() / "data" / SHAPENET_PART_RAW_NAME,
        ]
    )
    for candidate in candidates:
        hdf5_layout = (candidate / "train_hdf5_file_list.txt").is_file()
        text_layout = (
            (candidate / "synsetoffset2category.txt").is_file()
            and (
                candidate
                / "train_test_split"
                / "shuffled_train_file_list.json"
            ).is_file()
        )
        if candidate.is_dir() and (hdf5_layout or text_layout):
            return candidate.resolve()
    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "ShapeNetPart directory was not found. Checked:\n  " + checked
    )


def download_shapenet_part(destination: str | Path | None = None) -> Path:
    """Download the ShapeNetPart normal-point archive from Hugging Face."""
    try:
        return resolve_shapenet_part_dir(destination)
    except FileNotFoundError:
        pass
    project = Path(__file__).resolve().parents[1]
    root = Path(destination) if destination else project / "data"
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{SHAPENET_PART_RAW_NAME}.zip"
    partial = archive.with_suffix(".zip.part")
    print(f"downloading ShapeNetPart from Hugging Face: {SHAPENET_PART_URL}", flush=True)
    urllib.request.urlretrieve(SHAPENET_PART_URL, partial)
    partial.replace(archive)
    root_resolved = root.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (root / member.filename).resolve()
            if target != root_resolved and root_resolved not in target.parents:
                raise ValueError("ShapeNetPart archive contains an unsafe path")
        handle.extractall(root)
    archive.unlink()
    return resolve_shapenet_part_dir(root / SHAPENET_PART_RAW_NAME)


def _listed_hdf5_files(root: Path, list_name: str) -> list[Path]:
    list_path = root / list_name
    if not list_path.is_file():
        raise FileNotFoundError(f"missing ShapeNetPart list: {list_path}")
    paths = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        listed = Path(line.strip().replace("\\", "/"))
        candidates = [root / listed, root / listed.name]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"listed ShapeNetPart file was not found: {line}")
        paths.append(path)
    return paths


def _load_shapenet_hdf5_layout(
    root: Path,
    split: str,
    limit: int | None,
    num_points: int,
    include_normals: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    list_names = (
        ("train_hdf5_file_list.txt", "val_hdf5_file_list.txt")
        if split == "train"
        else ("test_hdf5_file_list.txt",)
    )
    files = [path for name in list_names for path in _listed_hdf5_files(root, name)]
    point_parts, category_parts, label_parts = [], [], []
    loaded = 0
    for path in files:
        with h5py.File(path, "r") as handle:
            points = np.asarray(handle["data"], dtype=np.float32)
            if include_normals and points.shape[-1] < 6:
                if "normal" not in handle:
                    raise ValueError(
                        f"{path} has no normals; use the ShapeNetPart normal-point layout"
                    )
                normals = np.asarray(handle["normal"], dtype=np.float32)
                points = np.concatenate([points[..., :3], normals[..., :3]], -1)
            categories = np.asarray(handle["label"]).reshape(-1)
            labels = np.asarray(handle["pid"])
        if points.shape[1] < num_points or labels.shape[1] < num_points:
            raise ValueError(f"{path} contains fewer than {num_points} points")
        point_parts.append(points[:, :num_points, : 6 if include_normals else 3])
        category_parts.append(categories)
        label_parts.append(labels[:, :num_points])
        loaded += len(points)
        if limit is not None and loaded >= limit:
            break
    if not point_parts:
        raise FileNotFoundError(f"no ShapeNetPart {split} files found in {root}")
    points = np.concatenate(point_parts)
    categories = np.concatenate(category_parts)
    labels = np.concatenate(label_parts)
    if limit is not None:
        points, categories, labels = (
            points[:limit],
            categories[:limit],
            labels[:limit],
        )
    return points, categories, labels


def _raw_shapenet_records(root: Path, split: str) -> list[tuple[Path, int]]:
    category_index = {
        name: index for index, name in enumerate(SHAPENET_PART_CATEGORY_NAMES)
    }
    synset_to_category = {}
    mapping_path = root / "synsetoffset2category.txt"
    for line in mapping_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[0] not in category_index:
            raise ValueError(f"unexpected ShapeNetPart category mapping: {line}")
        synset_to_category[fields[1]] = category_index[fields[0]]
    if len(synset_to_category) != len(SHAPENET_PART_CATEGORY_NAMES):
        raise ValueError("ShapeNetPart category mapping is incomplete")

    def read_entries(name: str) -> list[str]:
        split_path = root / "train_test_split" / name
        return [
            entry.replace("\\", "/")
            for entry in json.loads(split_path.read_text(encoding="utf-8"))
        ]

    test_entries = read_entries("shuffled_test_file_list.json")
    if split == "train":
        candidates = read_entries("shuffled_train_file_list.json")
        candidates += read_entries("shuffled_val_file_list.json")
        # The public normal-point mirror contains a small number of duplicate
        # IDs across its JSON lists. Reserve every test ID and de-duplicate the
        # combined training split so evaluation cannot see a training shape.
        excluded = set(test_entries)
        entries = []
        seen = set()
        for entry in candidates:
            if entry not in excluded and entry not in seen:
                entries.append(entry)
                seen.add(entry)
    else:
        entries = list(dict.fromkeys(test_entries))

    records = []
    for entry in entries:
        fields = entry.split("/")
        if len(fields) < 2 or fields[-2] not in synset_to_category:
            raise ValueError(f"unexpected ShapeNetPart split entry: {entry}")
        path = root / fields[-2] / f"{fields[-1]}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"ShapeNetPart point file was not found: {path}")
        records.append((path, synset_to_category[fields[-2]]))
    return records


def _load_shapenet_text_layout(
    root: Path,
    split: str,
    limit: int | None,
    num_points: int,
    include_normals: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_tag = "normal" if include_normals else "xyz"
    cache_path = root / f"fwpnet_cache_{split}_{num_points}_{feature_tag}.h5"
    legacy_cache = root / f"fwpnet_cache_{split}_{num_points}.h5"
    if not include_normals and not cache_path.is_file() and legacy_cache.is_file():
        cache_path = legacy_cache
    if cache_path.is_file():
        with h5py.File(cache_path, "r") as handle:
            count = len(handle["data"])
            if limit is not None:
                count = min(count, limit)
            return (
                np.asarray(handle["data"][:count], dtype=np.float32),
                np.asarray(handle["label"][:count]).reshape(-1),
                np.asarray(handle["pid"][:count]),
            )

    records = _raw_shapenet_records(root, split)
    selected = records if limit is None else records[:limit]
    if not selected:
        raise ValueError(f"ShapeNetPart {split} split is empty")
    if limit is None:
        print(
            f"building one-time ShapeNetPart cache from {len(selected)} text files",
            flush=True,
        )
    point_parts, category_parts, label_parts = [], [], []
    for path, category in selected:
        sample = np.loadtxt(path, dtype=np.float32)
        required_columns = 7 if include_normals else 4
        if (
            sample.ndim != 2
            or sample.shape[0] == 0
            or sample.shape[1] < required_columns
        ):
            raise ValueError(
                f"{path} must contain point rows with at least {required_columns} columns"
            )
        # Raw ShapeNetPart objects have different point counts (some contain
        # fewer than 1024 rows). A deterministic evenly-spaced index both
        # downsamples long files and repeats short files, so every model sees
        # exactly the same points without dropping valid objects.
        index = np.linspace(0, sample.shape[0] - 1, num_points).round().astype(np.int64)
        sampled = sample[index]
        point_parts.append(sampled[:, : 6 if include_normals else 3])
        category_parts.append(category)
        label_parts.append(sampled[:, -1].astype(np.int64))
    points = np.stack(point_parts)
    categories = np.asarray(category_parts, dtype=np.int64)
    labels = np.stack(label_parts)

    if limit is None:
        temporary = cache_path.with_suffix(".h5.part")
        with h5py.File(temporary, "w") as handle:
            handle.create_dataset("data", data=points)
            handle.create_dataset("label", data=categories[:, None])
            handle.create_dataset("pid", data=labels)
        temporary.replace(cache_path)
    return points, categories, labels


def load_shapenet_part(
    split: str,
    data_dir: str | Path | None = None,
    limit: int | None = None,
    num_points: int = 1024,
    include_normals: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load ShapeNetPart coordinates and optional normals with labels."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    root = resolve_shapenet_part_dir(data_dir)
    if (root / "train_hdf5_file_list.txt").is_file():
        points, categories, labels = _load_shapenet_hdf5_layout(
            root, split, limit, num_points, include_normals
        )
    else:
        points, categories, labels = _load_shapenet_text_layout(
            root, split, limit, num_points, include_normals
        )
    xyz = points[..., :3]
    xyz = xyz - xyz.mean(1, keepdims=True)
    scale = np.abs(xyz).max(axis=(1, 2), keepdims=True)
    xyz = xyz / np.maximum(scale, 1e-12)
    if include_normals:
        normals = points[..., 3:6]
        normal_length = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.maximum(normal_length, 1e-12)
        points = np.concatenate([xyz, normals], axis=-1)
    else:
        points = xyz
    return (
        torch.from_numpy(points).float(),
        torch.from_numpy(categories).long(),
        torch.from_numpy(labels).long(),
    )


def density_gradient(
    full_points: torch.Tensor,
    count: int,
    strength: float = 2.5,
    seed: int = 0,
) -> torch.Tensor:
    if count > full_points.shape[1]:
        raise ValueError("density-gradient count exceeds available points")
    generator = torch.Generator().manual_seed(seed)
    sampled = []
    for cloud in full_points:
        direction = torch.randn(3, generator=generator)
        direction = direction / direction.norm().clamp_min(1e-12)
        weights = torch.softmax(strength * (cloud @ direction), 0)
        index = torch.multinomial(
            weights, count, replacement=False, generator=generator
        )
        sampled.append(cloud[index])
    return torch.stack(sampled)

