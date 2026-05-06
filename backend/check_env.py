#!/usr/bin/env python3
"""Preflight checks for local project runnability."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

try:
    import sam2 as _sam2_pkg  # type: ignore
except Exception:
    _sam2_pkg = None


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.data_layout import resolve_train_annotations_path  # noqa: E402

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def has_any_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def check_modules() -> tuple[bool, list[str]]:
    required = [
        "numpy",
        "cv2",
        "torch",
        "torchvision",
        "scipy",
        "sklearn",
        "matplotlib",
        "tqdm",
        "PIL",
        "albumentations",
        "segmentation_models_pytorch",
        "timm",
        "shapely",
        "sam2",
        "fastapi",
        "uvicorn",
        "multipart",
    ]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    return len(missing) == 0, missing


def check_data_files() -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not resolve_train_annotations_path(ROOT).is_file():
        missing.append(
            "data/train_annotations.json or data/train_annotations_updated_*.json (or legacy data/raw/annotations/train_annotations.json)"
        )
    for path in (ROOT / "data" / "processed" / "train.txt", ROOT / "data" / "processed" / "val.txt"):
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
    return len(missing) == 0, missing


def check_image_inputs() -> tuple[bool, dict[str, int]]:
    counts = {
        "data/train_images": has_any_images(ROOT / "data" / "train_images"),
        "data/evaluation_images": has_any_images(ROOT / "data" / "evaluation_images"),
        "data/raw/train_images_tif": has_any_images(ROOT / "data" / "raw" / "train_images_tif"),
        "data/raw/train_images_png": has_any_images(ROOT / "data" / "raw" / "train_images_png"),
        "data/raw/evaluation_images_tif": has_any_images(ROOT / "data" / "raw" / "evaluation_images_tif"),
        "data/raw/evaluation_images_png": has_any_images(ROOT / "data" / "raw" / "evaluation_images_png"),
    }
    train_ok = (
        counts["data/train_images"] > 0
        or counts["data/raw/train_images_tif"] > 0
        or counts["data/raw/train_images_png"] > 0
    )
    eval_ok = (
        counts["data/evaluation_images"] > 0
        or counts["data/raw/evaluation_images_tif"] > 0
        or counts["data/raw/evaluation_images_png"] > 0
    )
    return train_ok and eval_ok, counts


def check_sam2_assets() -> tuple[bool, str]:
    cfg_candidates = [
        ROOT / "checkpoints_sam2" / "sam2_hiera_l.yaml",
        ROOT / "checkpoints_sam2" / "sam2_hiera_s.yaml",
        ROOT / "sam2_hiera_l.yaml",
        ROOT / "sam2_hiera_s.yaml",
        ROOT / "configs" / "sam2_hiera_l.yaml",
        ROOT / "configs" / "sam2_hiera_s.yaml",
        ROOT / "backend" / "configs" / "sam2_hiera_l.yaml",
        ROOT / "backend" / "configs" / "sam2_hiera_s.yaml",
    ]
    if _sam2_pkg is not None:
        pkg_root = Path(_sam2_pkg.__file__).resolve().parent
        cfg_candidates.extend(
            [
                pkg_root / "sam2_hiera_l.yaml",
                pkg_root / "sam2_hiera_s.yaml",
                pkg_root / "configs" / "sam2" / "sam2_hiera_l.yaml",
                pkg_root / "configs" / "sam2" / "sam2_hiera_s.yaml",
            ]
        )
    student_candidates = [
        ROOT / "checkpoints_sam2" / "sam2_finetuned_tree_canopy.pt",
        ROOT / "checkpoints_sam2" / "model.torch",
    ]

    cfg_path = next((p for p in cfg_candidates if p.exists()), None)
    student_path = next((p for p in student_candidates if p.exists()), None)

    if cfg_path is None:

        def _short(p: Path) -> str:
            try:
                return str(p.relative_to(ROOT))
            except ValueError:
                return str(p)

        searched = ", ".join(_short(p) for p in cfg_candidates)
        return False, f"missing config (checked: {searched})"

    try:
        cfg_rel = str(cfg_path.relative_to(ROOT))
    except ValueError:
        cfg_rel = str(cfg_path)
    if student_path is None:
        return True, (
            f"config={cfg_rel}, student_weights=optional (not found yet). "
            "SAM2 fine-tuning still requires a Meta base .pt in checkpoints_sam2 (or SAM2_CHECKPOINT) matching the YAML; "
            "after training, sam2_finetuned_tree_canopy.pt is used for inference."
        )
    try:
        student_rel = str(student_path.relative_to(ROOT))
    except ValueError:
        student_rel = str(student_path)
    return True, f"config={cfg_rel}, student_weights={student_rel}"


def check_deeplab_checkpoints() -> tuple[bool, str]:
    ckpt_dir = ROOT / "checkpoints_deeplabv3plus"
    named = ckpt_dir / "deeplabv3plus_checkpoint.pth"
    final_model = ckpt_dir / "final_model.pth"
    fold_models = list(ckpt_dir.glob("best_model_fold*.pth")) if ckpt_dir.exists() else []
    return named.exists() or final_model.exists() or len(fold_models) > 0, str(ckpt_dir.relative_to(ROOT))


def check_frontend_tooling() -> tuple[bool, str]:
    has_node = shutil.which("node") is not None
    has_npm = shutil.which("npm") is not None
    frontend_package = (ROOT / "frontend" / "package.json").exists()
    return has_node and has_npm and frontend_package, (
        f"node={'yes' if has_node else 'no'}, "
        f"npm={'yes' if has_npm else 'no'}, "
        f"frontend/package.json={'yes' if frontend_package else 'no'}"
    )


def print_check(name: str, ok: bool, detail: str = "") -> None:
    status = "OK" if ok else "MISSING"
    message = f"[{status}] {name}"
    if detail:
        message += f" - {detail}"
    print(message)


def main() -> int:
    mod_ok, missing_mods = check_modules()
    data_ok, missing_data = check_data_files()
    images_ok, image_counts = check_image_inputs()
    sam_ok, sam_detail = check_sam2_assets()
    deeplab_ckpt_ok, ckpt_dir = check_deeplab_checkpoints()
    frontend_ok, frontend_detail = check_frontend_tooling()

    print_check("Python modules", mod_ok, "" if mod_ok else ", ".join(missing_mods))
    print_check("Core data files", data_ok, "" if data_ok else ", ".join(missing_data))
    print_check(
        "Image inputs",
        images_ok,
        ", ".join([f"{key}={value}" for key, value in image_counts.items()]),
    )
    print_check("SAM2 model assets", sam_ok, sam_detail)
    print_check(
        "DeepLab checkpoints",
        deeplab_ckpt_ok,
        "" if deeplab_ckpt_ok else f"no trained model found in {ckpt_dir}",
    )
    print_check("Frontend tooling", frontend_ok, frontend_detail)

    all_ok = mod_ok and data_ok and images_ok and sam_ok and frontend_ok
    if all_ok:
        print("\nProject is runnable for training/inference workflows.")
        return 0

    print("\nProject is not fully runnable yet. Resolve missing items above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
