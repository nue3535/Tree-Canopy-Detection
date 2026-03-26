from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None

try:
    from torchvision.models.detection import maskrcnn_resnet50_fpn
except Exception:
    maskrcnn_resnet50_fpn = None

try:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
except Exception:
    SegformerForSemanticSegmentation = None
    SegformerImageProcessor = None

# Use a local writable matplotlib cache before importing project modules.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".").resolve() / ".mplconfig"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import sam2 as _sam2_pkg
except Exception:
    build_sam2 = None
    SAM2ImagePredictor = None
    _sam2_pkg = None

from backend.scripts.deeplab_v3plus_workflow import (
    DeepLabV3PlusConfig,
    ensure_output_dirs,
    get_transform,
    load_best_model,
    predict_image,
)


SCENE_LABELS = [
    "agriculture_plantation",
    "industrial_area",
    "open_field",
    "rural_area",
    "urban_area",
]

CLASS_LABELS = {
    "0": "Background",
    "1": "Tree",
    "2": "Tree Group",
}

CLASS_COLORS_RGB = {
    "0": (0, 0, 0),
    "1": (0, 255, 0),
    "2": (255, 255, 0),
}

CLASS_COLORS_HEX = {
    class_id: "#{:02x}{:02x}{:02x}".format(*rgb)
    for class_id, rgb in CLASS_COLORS_RGB.items()
}


def _land_use_assessment(
    distribution: dict[str, float],
    strict_conservation_mode: bool = False,
) -> dict[str, str | float | bool]:
    """Simple decision-support summary based on canopy coverage."""
    bg_ratio = float(distribution.get("0", 0.0))
    tree_ratio = float(distribution.get("1", 0.0))
    group_ratio = float(distribution.get("2", 0.0))

    # Dense grouped canopy is usually more environmentally sensitive.
    group_weight = 1.8 if strict_conservation_mode else 1.4
    weighted_canopy = min(1.0, tree_ratio + group_weight * group_ratio)
    buildability_score = max(0.0, 1.0 - weighted_canopy)
    conservation_score = min(1.0, weighted_canopy)

    high_threshold = 0.80 if strict_conservation_mode else 0.70
    moderate_threshold = 0.55 if strict_conservation_mode else 0.45

    if buildability_score >= high_threshold:
        suitability = "High"
        recommendation = "Potentially suitable for infrastructure development with standard environmental checks."
    elif buildability_score >= moderate_threshold:
        suitability = "Moderate"
        recommendation = (
            "Partially suitable; prioritize low-impact planning and preserve identified tree zones."
            if not strict_conservation_mode
            else "Partially suitable under strict policy; require stronger mitigation and retention of canopy patches."
        )
    else:
        suitability = "Low"
        recommendation = (
            "Environmentally sensitive; avoid heavy development and consider conservation-first options."
            if not strict_conservation_mode
            else "Environmentally sensitive under strict policy; avoid development and prioritize conservation."
        )

    return {
        "policy_mode": "strict_conservation" if strict_conservation_mode else "standard",
        "strict_conservation_mode": bool(strict_conservation_mode),
        "suitability_level": suitability,
        "buildability_score": round(buildability_score, 4),
        "conservation_sensitivity_score": round(conservation_score, 4),
        "background_ratio": round(bg_ratio, 4),
        "individual_tree_ratio": round(tree_ratio, 4),
        "tree_group_ratio": round(group_ratio, 4),
        "recommendation": recommendation,
    }


def _to_hydra_config_name(config_path: Path) -> str:
    """Convert absolute SAM2 YAML path to Hydra config name."""
    norm = str(config_path).replace("\\", "/")
    marker = "/configs/"
    if marker in norm:
        return f"configs/{norm.split(marker, 1)[1]}"
    return config_path.name


def _resolve_sam2_assets(root: Path) -> tuple[Path | None, Path | None]:
    """Resolve SAM2 config/checkpoint using project and package fallbacks."""
    cfg_candidates = [
        root / "checkpoints_sam2" / "sam2_hiera_s.yaml",
        root / "sam2_hiera_s.yaml",
        root / "configs" / "sam2_hiera_s.yaml",
        root / "backend" / "configs" / "sam2_hiera_s.yaml",
    ]
    if _sam2_pkg is not None:
        pkg_root = Path(_sam2_pkg.__file__).resolve().parent
        cfg_candidates.extend(
            [
                pkg_root / "sam2_hiera_s.yaml",
                pkg_root / "configs" / "sam2" / "sam2_hiera_s.yaml",
            ]
        )

    ckpt_candidates = [
        root / "checkpoints_sam2" / "sam2_hiera_small.pt",
        root / "sam2_hiera_small.pt",
        root / "checkpoints" / "sam2_hiera_small.pt",
        root / "backend" / "checkpoints" / "sam2_hiera_small.pt",
    ]

    cfg = next((p for p in cfg_candidates if p.exists()), None)
    ckpt = next((p for p in ckpt_candidates if p.exists()), None)
    return cfg, ckpt


def get_method_precheck(project_root: Path | None = None) -> dict[str, dict]:
    root = project_root or Path(__file__).resolve().parents[2]

    deeplab_candidates = [
        root / "checkpoints_deeplabv3plus" / "final_model.pth",
        *sorted((root / "checkpoints_deeplabv3plus").glob("best_model_fold*.pth")),
    ]
    deeplab_ckpt = next((p for p in deeplab_candidates if p.exists()), None)

    sam2_dependency_ok = build_sam2 is not None and SAM2ImagePredictor is not None
    sam2_cfg, sam2_ckpt = _resolve_sam2_assets(root)
    sam2_assets_ok = sam2_cfg is not None and sam2_ckpt is not None

    unet_candidates = [
        root / "checkpoints_unet" / "best_model.pth",
        root / "checkpoints_unet" / "final_model.pth",
        root / "checkpoints_unet" / "model.pth",
    ]
    unet_ckpt = next((p for p in unet_candidates if p.exists()), None)

    maskrcnn_candidates = [
        root / "checkpoints_mask_rcnn" / "best_model.pth",
        root / "checkpoints_mask_rcnn" / "final_model.pth",
        root / "checkpoints_mask_rcnn" / "model.pth",
    ]
    maskrcnn_ckpt = next((p for p in maskrcnn_candidates if p.exists()), None)

    segformer_candidates = [
        root / "checkpoints_segformer",
        root / "models" / "segformer",
    ]
    segformer_dir = next((p for p in segformer_candidates if p.exists() and p.is_dir()), None)
    segformer_config = segformer_dir / "config.json" if segformer_dir else None
    segformer_ready = segformer_dir is not None and segformer_config is not None and segformer_config.exists()

    return {
        "deeplabv3plus": {
            "checkpoint_found": deeplab_ckpt is not None,
            "checkpoint_path": str(deeplab_ckpt) if deeplab_ckpt else "",
            "dependency_ok": True,
            "ready_for_model_inference": deeplab_ckpt is not None,
            "reason": "" if deeplab_ckpt else "Checkpoint missing.",
        },
        "sam2": {
            "checkpoint_found": sam2_assets_ok,
            "checkpoint_path": str(sam2_ckpt) if sam2_ckpt else "",
            "dependency_ok": sam2_dependency_ok,
            "ready_for_model_inference": sam2_dependency_ok and sam2_assets_ok,
            "reason": "" if (sam2_dependency_ok and sam2_assets_ok) else "SAM2 dependency/config/checkpoint missing.",
        },
        "unet": {
            "checkpoint_found": unet_ckpt is not None,
            "checkpoint_path": str(unet_ckpt) if unet_ckpt else "",
            "dependency_ok": smp is not None,
            "ready_for_model_inference": (smp is not None) and (unet_ckpt is not None),
            "reason": "" if ((smp is not None) and (unet_ckpt is not None)) else "U-Net dependency/checkpoint missing.",
        },
        "maskrcnn": {
            "checkpoint_found": maskrcnn_ckpt is not None,
            "checkpoint_path": str(maskrcnn_ckpt) if maskrcnn_ckpt else "",
            "dependency_ok": maskrcnn_resnet50_fpn is not None,
            "ready_for_model_inference": (maskrcnn_resnet50_fpn is not None) and (maskrcnn_ckpt is not None),
            "reason": "" if ((maskrcnn_resnet50_fpn is not None) and (maskrcnn_ckpt is not None)) else "Mask R-CNN dependency/checkpoint missing.",
        },
        "segformer": {
            "checkpoint_found": segformer_ready,
            "checkpoint_path": str(segformer_dir) if segformer_dir else "",
            "dependency_ok": (SegformerForSemanticSegmentation is not None and SegformerImageProcessor is not None),
            "ready_for_model_inference": (
                SegformerForSemanticSegmentation is not None
                and SegformerImageProcessor is not None
                and segformer_ready
            ),
            "reason": "" if (
                SegformerForSemanticSegmentation is not None
                and SegformerImageProcessor is not None
                and segformer_ready
            ) else "SegFormer dependency/checkpoint missing.",
        },
    }


def image_to_base64(image: Image.Image) -> str:
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def colorize_mask(seg_mask: np.ndarray) -> Image.Image:
    color_lut = np.array(
        [
            CLASS_COLORS_RGB["0"],
            CLASS_COLORS_RGB["1"],
            CLASS_COLORS_RGB["2"],
        ],
        dtype=np.uint8,
    )
    colored = color_lut[np.clip(seg_mask.astype(np.int32), 0, len(color_lut) - 1)]
    return Image.fromarray(colored, mode="RGB")


def mask_overlay(original_image: Image.Image, seg_mask: np.ndarray) -> Image.Image:
    original_arr = np.asarray(original_image, dtype=np.uint8)
    color_lut = np.array(
        [
            [0, 0, 0],
            [34, 139, 34],
            [30, 144, 255],
        ],
        dtype=np.uint8,
    )
    colored_mask = color_lut[np.clip(seg_mask.astype(np.int32), 0, len(color_lut) - 1)]
    overlay = (0.6 * original_arr + 0.4 * colored_mask).astype(np.uint8)
    return Image.fromarray(overlay)


def fallback_segment(image_bytes: bytes) -> tuple[Image.Image, np.ndarray]:
    """Lightweight fallback using Excess Green vegetation index."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(bgr.astype(np.float32))

    exg = (2.0 * g) - r - b
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    threshold = int(np.percentile(exg_norm, 65))
    binary = (exg_norm >= threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    seg_mask = np.zeros_like(binary, dtype=np.uint8)
    seg_mask[binary > 0] = 1

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (seg_mask > 0).astype(np.uint8),
        connectivity=8,
    )
    total_pixels = seg_mask.shape[0] * seg_mask.shape[1]
    group_area_threshold = max(200, int(total_pixels * 0.01))
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= group_area_threshold:
            seg_mask[labels == label_id] = 2
    return image, seg_mask


def build_response(
    original_image: Image.Image,
    seg_mask: np.ndarray,
    scene_class: int,
    inference_mode: str,
    fallback_reason: str,
    strict_conservation_mode: bool = False,
) -> dict:
    overlay = mask_overlay(original_image, seg_mask)
    unique, counts = np.unique(seg_mask, return_counts=True)
    total = int(seg_mask.size) if seg_mask.size > 0 else 1
    observed_distribution = {
        str(int(cls_id)): round(float(count) / total, 6)
        for cls_id, count in zip(unique.tolist(), counts.tolist())
    }
    distribution = {
        "0": float(observed_distribution.get("0", 0.0)),
        "1": float(observed_distribution.get("1", 0.0)),
        "2": float(observed_distribution.get("2", 0.0)),
    }
    scene_label = (
        SCENE_LABELS[scene_class]
        if 0 <= int(scene_class) < len(SCENE_LABELS)
        else f"class_{scene_class}"
    )
    suitability_assessment = _land_use_assessment(
        distribution,
        strict_conservation_mode=strict_conservation_mode,
    )
    return {
        "scene_class": int(scene_class),
        "scene_label": scene_label,
        "class_distribution": distribution,
        "suitability_assessment": suitability_assessment,
        "strict_conservation_mode": bool(strict_conservation_mode),
        "class_labels": CLASS_LABELS,
        "class_colors": CLASS_COLORS_HEX,
        "mask_png_base64": image_to_base64(colorize_mask(seg_mask)),
        "overlay_png_base64": image_to_base64(overlay),
        "inference_mode": inference_mode,
        "fallback_reason": fallback_reason,
    }


class DeepLabSegmentationService:
    """Lazy-loaded service wrapper around the DeepLab inference pipeline."""

    def __init__(self) -> None:
        self._loaded = False
        self._model = None
        self._device = None
        self._transform = None
        self._config = None
        self._use_fallback = False
        self._fallback_reason = ""

    def _load(self) -> None:
        if self._loaded:
            return
        project_root = Path(__file__).resolve().parents[2]
        self._config = DeepLabV3PlusConfig(PROJECT_DIR=project_root)
        ensure_output_dirs(self._config)
        try:
            self._model, self._device, _ = load_best_model(self._config)
        except FileNotFoundError as exc:
            self._use_fallback = True
            self._fallback_reason = (
                "DeepLab checkpoint unavailable; using fallback vegetation segmentation."
            )
        self._transform = get_transform(img_size=self._config.IMG_SIZE)
        self._loaded = True

    def _predict_components(self, image_bytes: bytes, filename: str) -> tuple[Image.Image, np.ndarray, int, str, str]:
        self._load()
        if self._use_fallback:
            original_image, seg_mask = fallback_segment(image_bytes)
            return original_image, seg_mask, 2, "deeplab_fallback", self._fallback_reason

        suffix = Path(filename).suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            original_image, seg_mask, scene_class, _ = predict_image(
                self._model,
                tmp.name,
                self._transform,
                self._device,
            )
        return original_image, seg_mask, int(scene_class), "deeplab_model", ""

    def segment_bytes(self, image_bytes: bytes, filename: str, strict_conservation_mode: bool = False) -> dict:
        original_image, seg_mask, scene_class, inference_mode, fallback_reason = self._predict_components(
            image_bytes,
            filename,
        )
        return build_response(
            original_image=original_image,
            seg_mask=seg_mask,
            scene_class=scene_class,
            inference_mode=inference_mode,
            fallback_reason=fallback_reason,
            strict_conservation_mode=strict_conservation_mode,
        )


class SAM2SegmentationService:
    """Lazy-loaded service wrapper around SAM2 inference with fallback."""

    def __init__(self) -> None:
        self._loaded = False
        self._predictor = None
        self._use_fallback = False
        self._fallback_reason = ""

    def _load(self) -> None:
        if self._loaded:
            return

        if build_sam2 is None or SAM2ImagePredictor is None:
            self._use_fallback = True
            self._fallback_reason = "SAM2 package unavailable; using fallback vegetation segmentation."
            self._loaded = True
            return

        project_root = Path(__file__).resolve().parents[2]
        model_cfg, model_ckpt = _resolve_sam2_assets(project_root)
        if model_cfg is None or model_ckpt is None:
            self._use_fallback = True
            self._fallback_reason = "SAM2 checkpoint/config unavailable; using fallback vegetation segmentation."
            self._loaded = True
            return

        sam2_model = build_sam2(_to_hydra_config_name(model_cfg), str(model_ckpt), device="cpu")
        self._predictor = SAM2ImagePredictor(sam2_model)
        self._loaded = True

    @staticmethod
    def _grid_points(width: int, height: int, step: int = 64) -> tuple[np.ndarray, np.ndarray]:
        xs = np.arange(step // 2, width, step, dtype=np.int32)
        ys = np.arange(step // 2, height, step, dtype=np.int32)
        points = np.array([[x, y] for y in ys for x in xs], dtype=np.int32)
        labels = np.ones((points.shape[0],), dtype=np.int32)
        return points, labels

    def _sam2_segment(self, image_bytes: bytes) -> tuple[Image.Image, np.ndarray]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8).copy()
        height, width = rgb.shape[:2]
        points, labels = self._grid_points(width, height, step=max(32, min(width, height) // 16))

        if points.shape[0] == 0:
            seg_mask = np.zeros((height, width), dtype=np.uint8)
            return image, seg_mask

        self._predictor.set_image(rgb)
        chunk_size = 128
        all_masks = []
        all_scores = []
        for start in range(0, points.shape[0], chunk_size):
            end = min(points.shape[0], start + chunk_size)
            masks, scores, _ = self._predictor.predict(
                point_coords=points[start:end],
                point_labels=labels[start:end],
                multimask_output=False,
            )
            masks_arr = np.asarray(masks)
            scores_arr = np.asarray(scores, dtype=np.float32)

            # Normalize predictor outputs across SAM2 API variants.
            if masks_arr.ndim == 4:  # [N, 1, H, W]
                masks_arr = masks_arr[:, 0]
            elif masks_arr.ndim == 2:  # [H, W]
                masks_arr = masks_arr[None, ...]

            if scores_arr.ndim == 2:  # [N, 1]
                scores_arr = scores_arr[:, 0]
            elif scores_arr.ndim == 0:  # scalar
                scores_arr = scores_arr.reshape(1)

            masks_arr = masks_arr.astype(bool)
            scores_arr = scores_arr.astype(np.float32).reshape(-1)
            if masks_arr.ndim != 3 or scores_arr.size == 0:
                continue

            n = min(masks_arr.shape[0], scores_arr.shape[0])
            if n <= 0:
                continue
            all_masks.append(masks_arr[:n])
            all_scores.append(scores_arr[:n])

        if not all_masks or not all_scores:
            seg_mask = np.zeros((height, width), dtype=np.uint8)
            return image, seg_mask

        masks = np.concatenate(all_masks, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        order = np.argsort(scores)[::-1][: min(len(scores), 200)]
        masks = masks[order]

        seg_mask = np.zeros((height, width), dtype=np.uint8)
        occupancy = np.zeros((height, width), dtype=bool)
        total_pixels = height * width
        group_area_threshold = max(200, int(total_pixels * 0.01))
        for mask in masks:
            if mask.sum() < 50:
                continue
            overlap = np.logical_and(mask, occupancy).sum()
            if overlap > 0 and overlap / float(mask.sum()) > 0.15:
                continue
            mask = np.logical_and(mask, np.logical_not(occupancy))
            if mask.sum() < 50:
                continue
            class_id = 2 if int(mask.sum()) >= group_area_threshold else 1
            seg_mask[mask] = class_id
            occupancy = np.logical_or(occupancy, mask)
        return image, seg_mask

    def _predict_components(self, image_bytes: bytes, filename: str) -> tuple[Image.Image, np.ndarray, int, str, str]:
        self._load()
        if self._use_fallback:
            original_image, seg_mask = fallback_segment(image_bytes)
            return original_image, seg_mask, 2, "sam2_fallback", self._fallback_reason

        original_image, seg_mask = self._sam2_segment(image_bytes)
        return original_image, seg_mask, 2, "sam2_model", ""

    def segment_bytes(self, image_bytes: bytes, filename: str, strict_conservation_mode: bool = False) -> dict:
        original_image, seg_mask, scene_class, inference_mode, fallback_reason = self._predict_components(
            image_bytes,
            filename,
        )
        return build_response(
            original_image=original_image,
            seg_mask=seg_mask,
            scene_class=scene_class,
            inference_mode=inference_mode,
            fallback_reason=fallback_reason,
            strict_conservation_mode=strict_conservation_mode,
        )


class UNetSegmentationService:
    """U-Net semantic segmentation service with local-checkpoint loading."""

    def __init__(self) -> None:
        self._loaded = False
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = None
        self._img_size = 640
        self._use_fallback = False
        self._fallback_reason = ""

    def _load(self) -> None:
        if self._loaded:
            return
        project_root = Path(__file__).resolve().parents[2]
        config = DeepLabV3PlusConfig(PROJECT_DIR=project_root)
        self._img_size = int(config.IMG_SIZE)
        self._transform = get_transform(img_size=self._img_size)
        if smp is None:
            self._use_fallback = True
            self._fallback_reason = "U-Net dependencies unavailable; using fallback vegetation segmentation."
            self._loaded = True
            return

        checkpoints = [
            project_root / "checkpoints_unet" / "best_model.pth",
            project_root / "checkpoints_unet" / "final_model.pth",
            project_root / "checkpoints_unet" / "model.pth",
        ]
        ckpt_path = next((p for p in checkpoints if p.exists()), None)
        if ckpt_path is None:
            self._use_fallback = True
            self._fallback_reason = "U-Net checkpoint unavailable; using fallback vegetation segmentation."
            self._loaded = True
            return

        try:
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=3,
                activation=None,
            )
            checkpoint = torch.load(ckpt_path, map_location=self._device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(self._device)
            self._model = model
        except Exception:
            self._use_fallback = True
            self._fallback_reason = "U-Net model could not be loaded; using fallback vegetation segmentation."
        self._loaded = True

    def _predict_components(self, image_bytes: bytes, filename: str) -> tuple[Image.Image, np.ndarray, int, str, str]:
        self._load()
        if self._use_fallback:
            original_image, seg_mask = fallback_segment(image_bytes)
            return original_image, seg_mask, 2, "unet_fallback", self._fallback_reason

        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # DeepLab/U-Net transform expects a writable RGB numpy array.
        image_rgb = np.asarray(original_image, dtype=np.uint8).copy()
        image_tensor = self._transform(image_rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(image_tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred_small = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        seg_mask = cv2.resize(
            pred_small,
            original_image.size,
            interpolation=cv2.INTER_NEAREST,
        )
        return original_image, seg_mask, 2, "unet_model", ""

    def segment_bytes(self, image_bytes: bytes, filename: str, strict_conservation_mode: bool = False) -> dict:
        original_image, seg_mask, scene_class, inference_mode, fallback_reason = self._predict_components(
            image_bytes,
            filename,
        )
        return build_response(
            original_image=original_image,
            seg_mask=seg_mask,
            scene_class=scene_class,
            inference_mode=inference_mode,
            fallback_reason=fallback_reason,
            strict_conservation_mode=strict_conservation_mode,
        )


class MaskRCNNSegmentationService:
    """Mask R-CNN instance segmentation service with local-checkpoint loading."""

    def __init__(self) -> None:
        self._loaded = False
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_fallback = False
        self._fallback_reason = ""

    def _load(self) -> None:
        if self._loaded:
            return
        project_root = Path(__file__).resolve().parents[2]
        if maskrcnn_resnet50_fpn is None:
            self._use_fallback = True
            self._fallback_reason = "Mask R-CNN dependencies unavailable; using fallback vegetation segmentation."
            self._loaded = True
            return

        checkpoints = [
            project_root / "checkpoints_mask_rcnn" / "best_model.pth",
            project_root / "checkpoints_mask_rcnn" / "final_model.pth",
            project_root / "checkpoints_mask_rcnn" / "model.pth",
        ]
        ckpt_path = next((p for p in checkpoints if p.exists()), None)
        if ckpt_path is None:
            self._use_fallback = True
            self._fallback_reason = "Mask R-CNN checkpoint unavailable; using fallback vegetation segmentation."
            self._loaded = True
            return

        try:
            model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=3)
            checkpoint = torch.load(ckpt_path, map_location=self._device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(self._device)
            self._model = model
        except Exception:
            self._use_fallback = True
            self._fallback_reason = "Mask R-CNN model could not be loaded; using fallback vegetation segmentation."
        self._loaded = True

    def _predict_components(self, image_bytes: bytes, filename: str) -> tuple[Image.Image, np.ndarray, int, str, str]:
        self._load()
        if self._use_fallback:
            original_image, seg_mask = fallback_segment(image_bytes)
            return original_image, seg_mask, 2, "maskrcnn_fallback", self._fallback_reason

        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.asarray(original_image, dtype=np.uint8).copy()
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self._device)
        with torch.no_grad():
            output = self._model([image_tensor])[0]

        h, w = rgb.shape[:2]
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        occupancy = np.zeros((h, w), dtype=bool)
        total_pixels = h * w
        group_area_threshold = max(200, int(total_pixels * 0.01))

        scores = output.get("scores", torch.empty((0,), device=self._device)).detach().cpu().numpy()
        labels = output.get("labels", torch.empty((0,), device=self._device)).detach().cpu().numpy()
        masks = output.get("masks", torch.empty((0, 1, h, w), device=self._device)).detach().cpu().numpy()
        order = np.argsort(scores)[::-1]
        for idx in order:
            score = float(scores[idx])
            if score < 0.4:
                continue
            label = int(labels[idx])
            if label not in (1, 2):
                continue
            instance_mask = masks[idx, 0] > 0.5
            if instance_mask.sum() < 50:
                continue
            overlap = np.logical_and(instance_mask, occupancy).sum()
            if overlap > 0 and overlap / float(instance_mask.sum()) > 0.2:
                continue
            instance_mask = np.logical_and(instance_mask, np.logical_not(occupancy))
            if instance_mask.sum() < 50:
                continue
            class_id = 2 if (label == 2 or int(instance_mask.sum()) >= group_area_threshold) else 1
            seg_mask[instance_mask] = class_id
            occupancy = np.logical_or(occupancy, instance_mask)
        return original_image, seg_mask, 2, "maskrcnn_model", ""

    def segment_bytes(self, image_bytes: bytes, filename: str, strict_conservation_mode: bool = False) -> dict:
        original_image, seg_mask, scene_class, inference_mode, fallback_reason = self._predict_components(
            image_bytes,
            filename,
        )
        return build_response(
            original_image=original_image,
            seg_mask=seg_mask,
            scene_class=scene_class,
            inference_mode=inference_mode,
            fallback_reason=fallback_reason,
            strict_conservation_mode=strict_conservation_mode,
        )


class SegFormerSegmentationService:
    """SegFormer semantic segmentation service with local-checkpoint loading."""

    def __init__(self) -> None:
        self._loaded = False
        self._model = None
        self._processor = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_fallback = False
        self._fallback_reason = ""

    def _load(self) -> None:
        if self._loaded:
            return
        project_root = Path(__file__).resolve().parents[2]
        if SegformerForSemanticSegmentation is None or SegformerImageProcessor is None:
            self._use_fallback = True
            self._fallback_reason = "SegFormer dependencies unavailable; using fallback vegetation segmentation."
            self._loaded = True
            return

        candidates = [
            project_root / "checkpoints_segformer",
            project_root / "models" / "segformer",
        ]
        ckpt_dir = next((p for p in candidates if p.exists() and p.is_dir()), None)
        if ckpt_dir is None:
            self._use_fallback = True
            self._fallback_reason = "SegFormer checkpoint unavailable; using fallback vegetation segmentation."
            self._loaded = True
            return

        try:
            self._processor = SegformerImageProcessor.from_pretrained(str(ckpt_dir), local_files_only=True)
            self._model = SegformerForSemanticSegmentation.from_pretrained(str(ckpt_dir), local_files_only=True)
            self._model.eval()
            self._model.to(self._device)
        except Exception:
            self._use_fallback = True
            self._fallback_reason = "SegFormer model could not be loaded; using fallback vegetation segmentation."
        self._loaded = True

    def _predict_components(self, image_bytes: bytes, filename: str) -> tuple[Image.Image, np.ndarray, int, str, str]:
        self._load()
        if self._use_fallback:
            original_image, seg_mask = fallback_segment(image_bytes)
            return original_image, seg_mask, 2, "segformer_fallback", self._fallback_reason

        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self._processor(images=original_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device).float()
        # Some exported processors may emit uint8 tensors; ensure model-compatible scale.
        if torch.max(pixel_values).item() > 1.5:
            pixel_values = pixel_values / 255.0
        with torch.no_grad():
            outputs = self._model(pixel_values=pixel_values)
            logits = outputs.logits
            logits = F.interpolate(
                logits,
                size=(original_image.height, original_image.width),
                mode="bilinear",
                align_corners=False,
            )
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return original_image, pred, 2, "segformer_model", ""

    def segment_bytes(self, image_bytes: bytes, filename: str, strict_conservation_mode: bool = False) -> dict:
        original_image, seg_mask, scene_class, inference_mode, fallback_reason = self._predict_components(
            image_bytes,
            filename,
        )
        return build_response(
            original_image=original_image,
            seg_mask=seg_mask,
            scene_class=scene_class,
            inference_mode=inference_mode,
            fallback_reason=fallback_reason,
            strict_conservation_mode=strict_conservation_mode,
        )
