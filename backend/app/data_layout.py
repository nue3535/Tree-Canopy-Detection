"""Project data locations: ``data/train_images``, ``data/evaluation_images``, annotation JSON under ``data/``.

Training rasters are primarily GeoTIFF (``.tif`` / ``.tiff``); PNG/JPEG remain supported."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Known updated export from tooling (prefer first if present alongside other ``train_annotations_updated_*.json``).
TRAIN_ANNOTATIONS_UPDATED_STABLE = "train_annotations_updated_504bcc9e05b54435a9a56a841a3a1cf5.json"


def train_annotation_candidates(project_root: Path) -> list[Path]:
    """Ordered search list for training COCO-style JSON (first existing file wins)."""
    data = project_root / "data"
    seen: set[str] = set()
    out: list[Path] = []

    def add(p: Path) -> None:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)

    add(data / TRAIN_ANNOTATIONS_UPDATED_STABLE)
    for p in sorted(
        data.glob("train_annotations_updated_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    ):
        add(p)
    add(data / "train_annotations.json")
    add(data / "raw" / "annotations" / "train_annotations.json")
    add(project_root / "train_annotations.json")
    return out


def resolve_train_annotations_path(project_root: Path) -> Path:
    for path in train_annotation_candidates(project_root):
        if path.is_file():
            return path
    return project_root / "data" / "train_annotations.json"


def evaluation_annotation_candidates(project_root: Path) -> list[Path]:
    return [
        project_root / "data" / "evaluation_annotations.json",
        project_root / "data" / "raw" / "annotations" / "evaluation_annotations.json",
    ]


def resolve_evaluation_annotations_path(project_root: Path) -> Path:
    for path in evaluation_annotation_candidates(project_root):
        if path.is_file():
            return path
    return evaluation_annotation_candidates(project_root)[0]


def _dir_with_images(candidates: list[Path]) -> Path:
    first_existing: Path | None = None
    for path in candidates:
        if path.exists() and path.is_dir():
            if first_existing is None:
                first_existing = path
            if any(
                p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES for p in path.iterdir()
            ):
                return path
    return first_existing if first_existing is not None else candidates[0]


def resolve_train_image_dir(project_root: Path) -> Path:
    return _dir_with_images(
        [
            project_root / "data" / "train_images",
            project_root / "data" / "raw" / "train_images",
            project_root / "data" / "raw" / "train_images_tif",
            project_root / "data" / "raw" / "train_images_png",
            project_root / "train_images",
        ]
    )


def resolve_evaluation_image_dir(project_root: Path) -> Path:
    return _dir_with_images(
        [
            project_root / "data" / "evaluation_images",
            project_root / "data" / "raw" / "evaluation_images",
            project_root / "data" / "raw" / "evaluation_images_tif",
            project_root / "data" / "raw" / "evaluation_images_png",
        ]
    )
