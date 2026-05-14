from __future__ import annotations

import base64
import io
import logging
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
    load_model_from_checkpoint_path,
    predict_image,
)
from backend.scripts.mask_rcnn_workflow import MASK_RCNN_INFERENCE_IMG_SIZE, _build_mask_rcnn
from backend.scripts.unet_workflow import UNET_INFERENCE_IMAGENET_NORM, UNET_INFERENCE_IMG_SIZE
from backend.scripts.training_utils import make_multimodel_nb_val_transform

# Same hub id as `segformer_workflow` / multimodel notebook (do not import that module here — it requires transformers at import time).
SEGFORMER_PRETRAINED_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
# Must match `SegFormerDataset` val pipeline in `segformer_workflow.py` (training uses Albumentations, not HF processor).
SEGFORMER_INFERENCE_IMG_SIZE = 512

logger = logging.getLogger(__name__)

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


def _infer_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _first_existing_sam2_yaml(project_root: Path) -> Path | None:
    """Match `sam2_workflow` / training: project paths first, then installed `sam2` package."""
    env_cfg = (os.environ.get("SAM2_MODEL_CFG") or os.environ.get("MODEL_CFG") or "").strip()
    if env_cfg:
        p = Path(env_cfg)
        return p if p.is_file() else None
    candidates: list[Path] = [
        project_root / "checkpoints_sam2" / "sam2_hiera_l.yaml",
        project_root / "sam2_hiera_l.yaml",
        project_root / "configs" / "sam2_hiera_l.yaml",
        project_root / "backend" / "configs" / "sam2_hiera_l.yaml",
        project_root / "checkpoints_sam2" / "sam2_hiera_s.yaml",
        project_root / "sam2_hiera_s.yaml",
        project_root / "configs" / "sam2_hiera_s.yaml",
        project_root / "backend" / "configs" / "sam2_hiera_s.yaml",
    ]
    if _sam2_pkg is not None:
        pkg = Path(_sam2_pkg.__file__).resolve().parent
        candidates.extend(
            [
                pkg / "sam2_hiera_l.yaml",
                pkg / "sam2_hiera_s.yaml",
                pkg / "configs" / "sam2" / "sam2_hiera_l.yaml",
                pkg / "configs" / "sam2" / "sam2_hiera_s.yaml",
            ]
        )
    for p in candidates:
        if p.is_file():
            return p
    return None


def _resolve_sam2_finetuned_ckpt_path(root: Path) -> Path | None:
    """Fine-tuned weights for Segmentation / evaluation inference (same training export as Evaluation page).

    Order: env ``SAM2_FINETUNED_CKPT``, then ``sam2_finetuned_tree_canopy.pt`` (post-training export),
    ``sam2_training_best.pt``, ``sam2_training_latest.pt``, ``model.torch``.
    """
    env_ft = (os.environ.get("SAM2_FINETUNED_CKPT") or "").strip()
    if env_ft:
        p = Path(env_ft)
        return p if p.is_file() else None
    d = root / "checkpoints_sam2"
    for name in (
        "sam2_finetuned_tree_canopy.pt",
        "sam2_training_best.pt",
        "sam2_training_latest.pt",
        "model.torch",
    ):
        candidate = d / name
        if candidate.is_file():
            return candidate
    return None


def _resolve_sam2_student_assets(root: Path) -> tuple[Path | None, Path | None]:
    """SAM2 architecture YAML + fine-tuned student weights (``build_sam2`` with ``ckpt_path=None``, then ``load_state_dict``)."""
    cfg = _first_existing_sam2_yaml(root)
    if cfg is None:
        return None, None
    ckpt = _resolve_sam2_finetuned_ckpt_path(root)
    if ckpt is None:
        return None, None
    return cfg, ckpt


def _load_sam2_student_state_dict(sam2_model: torch.nn.Module, path: Path) -> None:
    """Load student fine-tuned state dict (weights produced by `sam2_workflow` / Evaluation training)."""
    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(str(path), map_location="cpu")
    if isinstance(blob, dict) and "finetuned_state_dict" in blob:
        state = blob["finetuned_state_dict"]
    elif isinstance(blob, dict) and "model_state_dict" in blob:
        state = blob["model_state_dict"]
    elif isinstance(blob, dict) and "state_dict" in blob:
        state = blob["state_dict"]
    elif isinstance(blob, dict):
        state = blob
    else:
        raise TypeError("SAM2 fine-tuned file must contain a state dict or wrapped checkpoint dict")
    if not isinstance(state, dict):
        raise TypeError("SAM2 fine-tuned state is not a mapping")
    sam2_model.load_state_dict(state, strict=False)


def get_method_precheck(project_root: Path | None = None) -> dict[str, dict]:
    root = project_root or _infer_project_root()

    deeplab_ckpt = root / "checkpoints_deeplabv3plus" / "deeplabv3plus_checkpoint.pth"
    deeplab_ckpt = deeplab_ckpt if deeplab_ckpt.is_file() else None

    sam2_dependency_ok = build_sam2 is not None and SAM2ImagePredictor is not None
    sam2_cfg, sam2_student_ckpt = _resolve_sam2_student_assets(root)
    sam2_assets_ok = sam2_cfg is not None and sam2_student_ckpt is not None

    unet_ckpt = root / "checkpoints_unet" / "unet_checkpoint.pth"
    unet_ckpt = unet_ckpt if unet_ckpt.is_file() else None

    maskrcnn_ckpt = root / "checkpoints_mask_rcnn" / "maskrcnn_checkpoint.pth"
    maskrcnn_ckpt = maskrcnn_ckpt if maskrcnn_ckpt.is_file() else None

    segformer_pth = root / "checkpoints_segformer" / "segformer_checkpoint.pth"
    segformer_candidates = [
        root / "checkpoints_segformer",
        root / "models" / "segformer",
    ]
    segformer_dir = next((p for p in segformer_candidates if p.exists() and p.is_dir()), None)
    segformer_config = segformer_dir / "config.json" if segformer_dir else None
    segformer_hf_ready = segformer_dir is not None and segformer_config is not None and segformer_config.exists()
    segformer_ready = segformer_pth.is_file() or segformer_hf_ready

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
            "checkpoint_path": str(sam2_student_ckpt) if sam2_student_ckpt else "",
            "finetuned_weights_path": str(sam2_student_ckpt) if sam2_student_ckpt else "",
            "dependency_ok": sam2_dependency_ok,
            "ready_for_model_inference": sam2_dependency_ok and sam2_assets_ok,
            "reason": ""
            if (sam2_dependency_ok and sam2_assets_ok)
            else (
                "SAM2 needs a Hiera YAML and fine-tuned weights in checkpoints_sam2 "
                "(sam2_finetuned_tree_canopy.pt from Evaluation training, or sam2_training_best.pt / "
                "sam2_training_latest.pt / model.torch). Public Meta weights are not used for inference."
                if sam2_dependency_ok
                else "SAM2 dependency missing."
            ),
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
            "checkpoint_path": str(segformer_pth) if segformer_pth.is_file() else (str(segformer_dir) if segformer_dir else ""),
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
    """Blend original with class colors (same palette as `colorize_mask` / API legend)."""
    original_arr = np.asarray(original_image, dtype=np.uint8)
    color_lut = np.array(
        [CLASS_COLORS_RGB[str(i)] for i in range(3)],
        dtype=np.uint8,
    )
    colored_mask = color_lut[np.clip(seg_mask.astype(np.int32), 0, len(color_lut) - 1)]
    overlay = (0.6 * original_arr + 0.4 * colored_mask).astype(np.uint8)
    return Image.fromarray(overlay)


def excess_green_raw(rgb: np.ndarray) -> np.ndarray:
    """Excess Green index per pixel (float32). Expects RGB uint8 HxWx3."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return 2.0 * g - r - b


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
        ckpt_path = project_root / "checkpoints_deeplabv3plus" / "deeplabv3plus_checkpoint.pth"
        try:
            self._model, self._device, _ = load_model_from_checkpoint_path(self._config, ckpt_path)
        except FileNotFoundError:
            self._use_fallback = True
            self._fallback_reason = (
                f"DeepLab checkpoint missing at {ckpt_path}; using fallback vegetation segmentation."
            )
        except Exception:
            self._use_fallback = True
            self._fallback_reason = (
                "DeepLab model could not be loaded from deeplabv3plus_checkpoint.pth; using fallback vegetation segmentation."
            )
        self._transform = get_transform(
            img_size=self._config.IMG_SIZE,
            imagenet_norm=self._config.AUGMENT_STYLE in ("notebook", "multimodel"),
        )
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
    @staticmethod
    def _resolve_sensitivity(segment_sensitivity: str | None) -> str:
        s = (segment_sensitivity or "balanced").strip().lower()
        if s in {"conservative", "balanced", "recall"}:
            return s
        return "balanced"

    @staticmethod
    def _thresholds_for_sensitivity(segment_sensitivity: str | None) -> dict[str, float | int]:
        """Mode-specific defaults; env vars can still override each value."""
        mode = SAM2SegmentationService._resolve_sensitivity(segment_sensitivity)
        if mode == "conservative":
            return {
                "grid_step": 64,
                "min_mask_score": 0.35,
                "min_mask_pixels": 120,
                "min_cc_area": 120,
                "group_area_frac": 0.03,
                "mask_pixel_exg": 14.0,
                "mask_min_veg_frac": 0.20,
                "close_kernel": 3,
                "max_component_frac": 0.18,
            }
        if mode == "recall":
            return {
                "grid_step": 48,
                "min_mask_score": 0.24,
                "min_mask_pixels": 50,
                "min_cc_area": 50,
                "group_area_frac": 0.012,
                "mask_pixel_exg": 9.0,
                "mask_min_veg_frac": 0.10,
                "close_kernel": 5,
                "max_component_frac": 0.35,
            }
        return {
            "grid_step": 56,
            "min_mask_score": 0.30,
            "min_mask_pixels": 80,
            "min_cc_area": 80,
            "group_area_frac": 0.02,
            "mask_pixel_exg": 12.0,
            "mask_min_veg_frac": 0.15,
            "close_kernel": 3,
            "max_component_frac": 0.25,
        }

    """SAM2 (Hiera) via `build_sam2` + `SAM2ImagePredictor` using **fine-tuned student weights** from Evaluation training.

    Loads architecture from the first available Hiera YAML (same search as training) and weights via
    `_resolve_sam2_finetuned_ckpt_path` (export ``sam2_finetuned_tree_canopy.pt``, then training snapshots).
    **No Meta pretrained ``.pt`` is used for inference** (`build_sam2(..., ckpt_path=None)` then
    ``load_state_dict``). If the checkpoint file on disk is replaced or updated, the model reloads on the next
    request. Override with ``SAM2_MODEL_CFG`` and ``SAM2_FINETUNED_CKPT`` if needed.

    **Inference resolution** defaults to 512 (`SAM2_IMAGE_SIZE`). This API uses a **grid of point prompts**
    (no GT boxes), then upsamples the label map to the original image size.

    **Three-class output (0=background, 1=tree, 2=tree group):** SAM2 still predicts binary masks per prompt.
    We OR accepted masks into a foreground union, optionally open to reduce speckle, then run
    `connectedComponentsWithStats` and assign each component class 1 vs 2 by **component area** (same
    thresholds as other instance-style paths).     **Roof suppression (default on):** after upsampling we remove **large** roof-like blobs: low mean ExG,
    or (optional) **reddish / R>G** with moderate ExG to catch brown roofs mislabeled as tree group. Small
    regions are kept. **Union closing** (default 5×5) merges grid speckle before CC labeling; set
    `SAM2_UNION_CLOSE_KERNEL=0` to disable. Set `SAM2_VEG_GATE=0` to skip ExG/roof cull. Env:
    `SAM2_GRID_STEP`, `SAM2_MIN_MASK_SCORE`, `SAM2_MAX_MASKS`, `SAM2_OPEN_KERNEL`, `SAM2_MIN_CC_AREA`,
    `SAM2_ROOF_CC_MIN_FRAC`, `SAM2_ROOF_MAX_MEAN_EXG`, `SAM2_ROOF_HUE_HEURISTIC`, `SAM2_ROOF_RG_MEAN_MIN`,
    `SAM2_ROOF_HUE_MAX_EXG`, `SAM2_UNION_CLOSE_KERNEL`.
    """

    def __init__(self) -> None:
        self._loaded = False
        self._predictor = None
        self._use_fallback = False
        self._fallback_reason = ""
        self._source_ckpt_resolved: str | None = None
        self._source_ckpt_mtime: float | None = None
        self._last_load_fail_ckpt_mtime: float | None = None

    def _reset_sam2_loader_state(self) -> None:
        self._loaded = False
        self._use_fallback = False
        self._predictor = None
        self._fallback_reason = ""
        self._source_ckpt_resolved = None
        self._source_ckpt_mtime = None

    def _invalidate_if_checkpoint_changed(self) -> None:
        """Reload after Evaluation writes a new export, or when a checkpoint first appears (no server restart)."""
        project_root = Path(__file__).resolve().parents[2]
        ckpt = _resolve_sam2_finetuned_ckpt_path(project_root)
        if ckpt is None:
            return
        try:
            resolved = str(ckpt.resolve())
            mtime = ckpt.stat().st_mtime
        except OSError:
            return

        if self._use_fallback:
            if "could not load student checkpoint" in self._fallback_reason.lower():
                if self._last_load_fail_ckpt_mtime is None or mtime > self._last_load_fail_ckpt_mtime + 1e-6:
                    logger.info("SAM2 inference: checkpoint file updated; retrying after load error.")
                    self._last_load_fail_ckpt_mtime = mtime
                    self._reset_sam2_loader_state()
                return
            logger.info("SAM2 inference: fine-tuned weights found at %s; loading.", resolved)
            self._reset_sam2_loader_state()
            return

        if not self._loaded or self._predictor is None:
            return

        if self._source_ckpt_resolved != resolved or (
            self._source_ckpt_mtime is not None and mtime > self._source_ckpt_mtime + 1e-6
        ):
            logger.info("SAM2 inference: checkpoint updated (%s); reloading model.", resolved)
            self._reset_sam2_loader_state()

    def _load(self) -> None:
        if self._loaded:
            return

        if build_sam2 is None or SAM2ImagePredictor is None:
            self._use_fallback = True
            self._fallback_reason = "SAM2 package unavailable; using fallback vegetation segmentation."
            self._loaded = True
            self._source_ckpt_resolved = None
            self._source_ckpt_mtime = None
            self._last_load_fail_ckpt_mtime = None
            return

        project_root = Path(__file__).resolve().parents[2]
        model_cfg, student_ckpt = _resolve_sam2_student_assets(project_root)
        if model_cfg is None or student_ckpt is None:
            self._use_fallback = True
            self._fallback_reason = (
                "SAM2 needs a Hiera YAML (see search paths) and fine-tuned weights in checkpoints_sam2 "
                "(sam2_finetuned_tree_canopy.pt from Evaluation training, or sam2_training_best.pt / "
                "sam2_training_latest.pt / model.torch). Public pretrained SAM2 weights are not used."
            )
            self._loaded = True
            self._source_ckpt_resolved = None
            self._source_ckpt_mtime = None
            self._last_load_fail_ckpt_mtime = None
            return

        try:
            sam2_model = build_sam2(_to_hydra_config_name(model_cfg), None, device="cpu")
            _load_sam2_student_state_dict(sam2_model, student_ckpt)
            logger.info("SAM2 inference: loaded student weights from %s", student_ckpt)
            self._source_ckpt_resolved = str(student_ckpt.resolve())
            self._source_ckpt_mtime = student_ckpt.stat().st_mtime
            self._last_load_fail_ckpt_mtime = None
        except Exception as exc:
            logger.warning("SAM2 inference: failed to load student weights (%s)", exc)
            self._use_fallback = True
            self._fallback_reason = f"SAM2 could not load student checkpoint: {exc}"
            self._loaded = True
            self._source_ckpt_resolved = None
            self._source_ckpt_mtime = None
            try:
                self._last_load_fail_ckpt_mtime = student_ckpt.stat().st_mtime
            except OSError:
                self._last_load_fail_ckpt_mtime = None
            return

        self._predictor = SAM2ImagePredictor(sam2_model)
        self._loaded = True

    @staticmethod
    def _grid_points(width: int, height: int, step: int = 64) -> tuple[np.ndarray, np.ndarray]:
        xs = np.arange(step // 2, width, step, dtype=np.int32)
        ys = np.arange(step // 2, height, step, dtype=np.int32)
        points = np.array([[x, y] for y in ys for x in xs], dtype=np.int32)
        labels = np.ones((points.shape[0],), dtype=np.int32)
        return points, labels

    @staticmethod
    def _cull_large_low_exg_components(
        seg_mask: np.ndarray,
        exg_full: np.ndarray,
        rgb: np.ndarray | None = None,
    ) -> np.ndarray:
        """Remove *large* roof-like blobs (low ExG and/or reddish R>G); keep small CCs."""
        fg = (seg_mask > 0).astype(np.uint8)
        if int(fg.max()) == 0:
            return seg_mask
        h, w = seg_mask.shape
        total = float(h * w)
        min_big_frac = float(os.environ.get("SAM2_ROOF_CC_MIN_FRAC", "0.012"))
        min_big_area = max(400, int(total * min_big_frac))
        roof_max_mean_exg = float(os.environ.get("SAM2_ROOF_MAX_MEAN_EXG", "14"))
        use_hue = os.environ.get("SAM2_ROOF_HUE_HEURISTIC", "1").strip().lower() not in ("0", "false", "no")
        rg_min = float(os.environ.get("SAM2_ROOF_RG_MEAN_MIN", "6"))
        hue_max_exg = float(os.environ.get("SAM2_ROOF_HUE_MAX_EXG", "24"))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        out = seg_mask.copy()
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_big_area:
                continue
            region = labels == i
            m_exg = float(exg_full[region].mean())
            drop = m_exg < roof_max_mean_exg
            if not drop and use_hue and rgb is not None:
                r = rgb[..., 0].astype(np.float32)[region]
                g = rgb[..., 1].astype(np.float32)[region]
                if float((r - g).mean()) >= rg_min and m_exg < hue_max_exg:
                    drop = True
            if drop:
                out[region] = 0
        return out

    def _sam2_segment(self, image_bytes: bytes, segment_sensitivity: str = "balanced") -> tuple[Image.Image, np.ndarray]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8).copy()
        height, width = rgb.shape[:2]
        thresholds = self._thresholds_for_sensitivity(segment_sensitivity)
        # Notebook: IMAGE_SIZE = 512; set_image / predict on this resolution.
        side = max(32, int(os.environ.get("SAM2_IMAGE_SIZE", "512")))
        rgb_small = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_LINEAR)
        # Balanced defaults recover more canopy while guarding against merged urban blobs.
        grid_step = max(24, int(os.environ.get("SAM2_GRID_STEP", str(int(thresholds["grid_step"])))))
        points, labels = self._grid_points(side, side, step=min(grid_step, side // 2))

        if points.shape[0] == 0:
            seg_mask = np.zeros((height, width), dtype=np.uint8)
            return image, seg_mask

        self._predictor.set_image(rgb_small)
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
        max_masks = max(1, int(os.environ.get("SAM2_MAX_MASKS", "220")))
        order = np.argsort(scores)[::-1][: min(len(scores), max_masks)]
        masks = masks[order]
        scores = scores[order]

        min_mask_score = float(os.environ.get("SAM2_MIN_MASK_SCORE", str(float(thresholds["min_mask_score"]))))
        min_mask_pixels = max(30, int(os.environ.get("SAM2_MIN_MASK_PIXELS", str(int(thresholds["min_mask_pixels"])))))
        total_pixels = side * side
        # Keep class-2 conservative enough to prevent blanket "group" masks.
        group_area_threshold = max(200, int(total_pixels * float(thresholds["group_area_frac"])))
        min_cc_area = max(25, int(os.environ.get("SAM2_MIN_CC_AREA", str(int(thresholds["min_cc_area"])))))
        veg_gate = os.environ.get("SAM2_VEG_GATE", "1").strip().lower() not in ("0", "false", "no")
        mask_veg_filter = os.environ.get("SAM2_MASK_VEG_FILTER", "1").strip().lower() in ("1", "true", "yes")
        exg_small = excess_green_raw(rgb_small) if (veg_gate and mask_veg_filter) else None
        mask_pixel_exg = float(os.environ.get("SAM2_MASK_PIXEL_EXG", str(float(thresholds["mask_pixel_exg"]))))
        mask_min_veg_frac = float(os.environ.get("SAM2_MASK_MIN_VEG_FRAC", str(float(thresholds["mask_min_veg_frac"]))))

        union = np.zeros((side, side), dtype=np.uint8)
        for mask, sc in zip(masks, scores):
            if float(sc) < min_mask_score:
                continue
            if int(mask.sum()) < min_mask_pixels:
                continue
            if exg_small is not None:
                exg_in = exg_small[mask]
                veg_frac = float((exg_in >= mask_pixel_exg).mean())
                if veg_frac < mask_min_veg_frac:
                    continue
            union = np.logical_or(union, mask).astype(np.uint8)

        close_u = int(os.environ.get("SAM2_UNION_CLOSE_KERNEL", str(int(thresholds["close_kernel"]))))
        if close_u >= 3 and close_u % 2 == 1 and int(union.max()) > 0:
            kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_u, close_u))
            union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kc)

        open_k = int(os.environ.get("SAM2_OPEN_KERNEL", "0"))
        if open_k >= 3 and open_k % 2 == 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
            union = cv2.morphologyEx(union, cv2.MORPH_OPEN, kernel)

        seg_small = np.zeros((side, side), dtype=np.uint8)
        if int(union.max()) > 0:
            num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
            max_component_frac = float(os.environ.get("SAM2_MAX_COMPONENT_FRAC", str(float(thresholds["max_component_frac"]))))
            for i in range(1, num_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_cc_area:
                    continue
                if area > int(total_pixels * max_component_frac):
                    # Huge blobs are usually farms/fields or merged prompt artifacts, not individual canopy clusters.
                    continue
                region = cc_labels == i
                seg_small[region] = 2 if area >= group_area_threshold else 1

        seg_mask = cv2.resize(seg_small, (width, height), interpolation=cv2.INTER_NEAREST)
        if veg_gate:
            exg_full = excess_green_raw(rgb)
            seg_mask = self._cull_large_low_exg_components(seg_mask, exg_full, rgb)
            if os.environ.get("SAM2_FG_PIXEL_STRIP", "0").strip().lower() in ("1", "true", "yes"):
                fg_min_exg = float(os.environ.get("SAM2_FG_MIN_EXG", "5"))
                strip = (seg_mask > 0) & (exg_full < fg_min_exg)
                seg_mask = seg_mask.copy()
                seg_mask[strip] = 0
        return image, seg_mask

    def _predict_components(
        self, image_bytes: bytes, filename: str, segment_sensitivity: str = "balanced"
    ) -> tuple[Image.Image, np.ndarray, int, str, str]:
        self._invalidate_if_checkpoint_changed()
        self._load()
        if self._use_fallback:
            original_image, seg_mask = fallback_segment(image_bytes)
            return original_image, seg_mask, 2, "sam2_fallback", self._fallback_reason

        mode = self._resolve_sensitivity(segment_sensitivity)
        original_image, seg_mask = self._sam2_segment(image_bytes, segment_sensitivity=mode)
        return original_image, seg_mask, 2, f"sam2_student_checkpoint_{mode}", ""

    def segment_bytes(
        self,
        image_bytes: bytes,
        filename: str,
        strict_conservation_mode: bool = False,
        segment_sensitivity: str = "balanced",
    ) -> dict:
        original_image, seg_mask, scene_class, inference_mode, fallback_reason = self._predict_components(
            image_bytes,
            filename,
            segment_sensitivity=segment_sensitivity,
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
        self._img_size = UNET_INFERENCE_IMG_SIZE
        self._use_fallback = False
        self._fallback_reason = ""

    def _load(self) -> None:
        if self._loaded:
            return
        project_root = Path(__file__).resolve().parents[2]
        self._img_size = UNET_INFERENCE_IMG_SIZE
        self._transform = get_transform(img_size=self._img_size, imagenet_norm=UNET_INFERENCE_IMAGENET_NORM)
        if smp is None:
            self._use_fallback = True
            self._fallback_reason = "U-Net dependencies unavailable; using fallback vegetation segmentation."
            self._loaded = True
            return

        ckpt_path = project_root / "checkpoints_unet" / "unet_checkpoint.pth"
        if not ckpt_path.is_file():
            self._use_fallback = True
            self._fallback_reason = f"U-Net checkpoint missing at {ckpt_path}; using fallback vegetation segmentation."
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
            try:
                checkpoint = torch.load(ckpt_path, map_location=self._device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(ckpt_path, map_location=self._device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(self._device)
            self._model = model
        except Exception:
            self._use_fallback = True
            self._fallback_reason = "U-Net model could not be loaded from unet_checkpoint.pth; using fallback vegetation segmentation."
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

        ckpt_path = project_root / "checkpoints_mask_rcnn" / "maskrcnn_checkpoint.pth"
        if not ckpt_path.is_file():
            self._use_fallback = True
            self._fallback_reason = (
                f"Mask R-CNN checkpoint missing at {ckpt_path}; using fallback vegetation segmentation."
            )
            self._loaded = True
            return

        try:
            model = _build_mask_rcnn(num_classes=3)
            try:
                checkpoint = torch.load(ckpt_path, map_location=self._device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(ckpt_path, map_location=self._device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(self._device)
            self._model = model
        except Exception:
            self._use_fallback = True
            self._fallback_reason = (
                "Mask R-CNN model could not be loaded from maskrcnn_checkpoint.pth; using fallback vegetation segmentation."
            )
        self._loaded = True

    def _predict_components(self, image_bytes: bytes, filename: str) -> tuple[Image.Image, np.ndarray, int, str, str]:
        self._load()
        if self._use_fallback:
            original_image, seg_mask = fallback_segment(image_bytes)
            return original_image, seg_mask, 2, "maskrcnn_fallback", self._fallback_reason

        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.asarray(original_image, dtype=np.uint8).copy()
        h, w = rgb.shape[:2]
        side = MASK_RCNN_INFERENCE_IMG_SIZE
        rgb_small = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(rgb_small).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self._device)
        with torch.no_grad():
            output = self._model([image_tensor])[0]

        seg_mask = np.zeros((h, w), dtype=np.uint8)
        occupancy = np.zeros((h, w), dtype=bool)
        total_pixels = h * w
        group_area_threshold = max(200, int(total_pixels * 0.01))

        scores = output.get("scores", torch.empty((0,), device=self._device)).detach().cpu().numpy()
        labels = output.get("labels", torch.empty((0,), device=self._device)).detach().cpu().numpy()
        masks = output.get("masks", torch.empty((0, 1, side, side), device=self._device)).detach().cpu().numpy()
        order = np.argsort(scores)[::-1]
        for idx in order:
            score = float(scores[idx])
            if score < 0.4:
                continue
            label = int(labels[idx])
            if label not in (1, 2):
                continue
            instance_small = masks[idx, 0] > 0.5
            instance_mask = cv2.resize(
                instance_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
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
        self._pixel_transform = None  # set when using .pth + same preproc as training (`get_transform`)
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

        weights_pth = project_root / "checkpoints_segformer" / "segformer_checkpoint.pth"
        hf_dirs = [
            project_root / "checkpoints_segformer",
            project_root / "models" / "segformer",
        ]
        ckpt_dir = next((p for p in hf_dirs if p.exists() and p.is_dir() and (p / "config.json").exists()), None)

        if not weights_pth.is_file() and ckpt_dir is None:
            self._use_fallback = True
            self._fallback_reason = "SegFormer checkpoint unavailable; using fallback vegetation segmentation."
            self._loaded = True
            return

        try:
            if weights_pth.is_file():
                self._processor = None
                self._pixel_transform = make_multimodel_nb_val_transform(SEGFORMER_INFERENCE_IMG_SIZE)
                self._model = SegformerForSemanticSegmentation.from_pretrained(
                    SEGFORMER_PRETRAINED_ID,
                    num_labels=3,
                    ignore_mismatched_sizes=True,
                )
                try:
                    blob = torch.load(weights_pth, map_location=self._device, weights_only=False)
                except TypeError:
                    blob = torch.load(weights_pth, map_location=self._device)
                if isinstance(blob, dict) and "model_state_dict" in blob:
                    state_dict = blob["model_state_dict"]
                elif isinstance(blob, dict) and "state_dict" in blob:
                    state_dict = blob["state_dict"]
                else:
                    state_dict = blob
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                self._model.load_state_dict(state_dict, strict=False)
            else:
                self._pixel_transform = None
                self._processor = SegformerImageProcessor.from_pretrained(str(ckpt_dir), local_files_only=True)
                self._model = SegformerForSemanticSegmentation.from_pretrained(str(ckpt_dir), local_files_only=True)
            self._model.eval()
            self._model.to(self._device)
        except Exception as exc:
            logger.warning("SegFormer load failed: %s", exc)
            self._use_fallback = True
            self._fallback_reason = "SegFormer model could not be loaded; using fallback vegetation segmentation."
        self._loaded = True

    def _predict_components(self, image_bytes: bytes, filename: str) -> tuple[Image.Image, np.ndarray, int, str, str]:
        self._load()
        if self._use_fallback:
            original_image, seg_mask = fallback_segment(image_bytes)
            return original_image, seg_mask, 2, "segformer_fallback", self._fallback_reason

        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self._pixel_transform is not None:
            rgb = np.asarray(original_image, dtype=np.uint8)
            aug = self._pixel_transform(image=rgb)
            pixel_values = aug["image"].unsqueeze(0).to(self._device).float()
        else:
            inputs = self._processor(images=original_image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self._device).float()
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
