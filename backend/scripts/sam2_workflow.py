"""
Train/Fine-Tune SAM 2 on a custom dataset using polygon annotations.

Reference note: batch_grounded_sam.py uses Hugging Face SAM ViT-Base + GroundingDINO (zero-shot),
not Meta SAM2. Fine-tuning loads a **base** Meta ``.pt`` (see ``SAM2_BASE_CHECKPOINT``) plus a Hiera YAML,
trains the prompt encoder + mask decoder with **GT box** prompts, and exports student weights to
``checkpoints_sam2/sam2_finetuned_tree_canopy.pt``. Grid-point validation (AP-style) remains available via ``SAM2_GRID_EVAL=1``.

This script adapts TRAIN_multi_image_batch.py to a folder layout like:  # Context of adaptation
- Images:        ./data/train_images/*.tif (primary) or ./data/raw/train_images_*/* or ./train_images/*  # Expected image directory
- Annotations:   ./data/train_annotations_updated_*.json (preferred), ./data/train_annotations.json, or legacy ./data/raw/annotations/...
- Optional masks: ./viz_masks, ./masks (not used for training here)  # Optional extras (not used)

We parse polygons for two classes: "individual_tree" and "group_of_trees", rasterize instance masks,
and fine-tune with box prompts at ``SAM2_PROMPT_IMAGE_SIZE`` (default 512).

Notes:
- Default: freeze the image encoder (``SAM2_FREEZE_IMAGE_ENCODER``); set to ``0`` to unfreeze.
- Epochs / LR / WD: ``SAM2_TRAIN_EPOCHS``, ``SAM2_TRAIN_LR``, ``SAM2_TRAIN_WEIGHT_DECAY``.
- Early stopping on validation loss: ``SAM2_EARLY_STOPPING_PATIENCE`` (default 25; ``0`` disables),
  ``SAM2_EARLY_STOPPING_MIN_DELTA``. Best ``val_loss`` weights are restored before saving the student checkpoint.
- Training log cadence: each epoch prints start lines for train/val; every ``SAM2_TRAIN_LOG_EVERY`` images (default 10)
  prints progress within an epoch; set ``SAM2_TRAIN_LOG_EVERY=0`` to disable mid-epoch lines only.
- Epoch checkpoints: ``checkpoints_sam2/sam2_training_latest.pt`` (full resume state after each epoch) and
  ``sam2_training_best.pt`` (best val_loss weights). Set ``SAM2_RESUME=1`` to continue from ``latest`` (skips warm-start
  from the inference export if ``latest`` exists).

Usage:
  python backend/scripts/sam2_workflow.py  # How to run

Requirements:
  - torch, numpy, opencv-python, tqdm, matplotlib
  - ``sam2`` (Segment Anything 2) with a matching Hiera YAML
  - Base checkpoint ``.pt`` on disk (``SAM2_CHECKPOINT`` / ``checkpoints_sam2/*.pt``)

Workflow overview:
  1) Load annotations; use ``train.txt`` / ``val.txt`` / optional ``test.txt`` when present.
  2) Fine-tune with GT boxes (BCE+Dice + score loss) for ``SAM2_TRAIN_EPOCHS`` epochs.
  3) Save ``checkpoints_sam2/sam2_finetuned_tree_canopy.pt`` (dict with ``finetuned_state_dict``).
  4) Evaluate with the same box prompts (semantic 3-class IoU); optional grid-point AP eval if ``SAM2_GRID_EVAL=1``.
"""

import os  # OS utilities (paths, env vars, directories)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Allow duplicate OpenMP runtimes on Windows
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Limit OpenMP threads to reduce contention

from contextlib import nullcontext
import json  # JSON parsing for annotations
import math  # isnan for training_results serialization
import time  # Per-epoch duration logging
from pathlib import Path
import random  # Deterministic splitting and sampling
from typing import List, Tuple, Dict, Any  # Type hints for clarity

from collections import defaultdict  # Per-class metric lists in prompted eval

import numpy as np  # Numerical arrays and mask operations
import torch  # Tensors, autograd, CUDA, optimizers, AMP
import torch.nn.functional as F  # BCE-with-logits and losses for SAM2 fine-tune
import cv2  # Image I/O and geometry (OpenCV)

try:
    from sam2.build_sam import build_sam2  # SAM2 model factory
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # High-level predictor wrapper
    import sam2 as _sam2_pkg
except Exception:
    build_sam2 = None
    SAM2ImagePredictor = None
    _sam2_pkg = None

try:
    from tqdm import tqdm  # Progress bars for loops
except Exception:
    def tqdm(x, **kwargs):  # Fallback no-op progress if tqdm missing
        return x  # Return iterable unchanged

try:
    from backend.app.data_layout import resolve_train_annotations_path, resolve_train_image_dir
except ImportError:
    resolve_train_image_dir = None  # type: ignore[misc, assignment]
    resolve_train_annotations_path = None  # type: ignore[misc, assignment]

try:
    from backend.scripts.training_utils import EarlyStopping
except ImportError:
    from training_utils import EarlyStopping


# ----------------------- Config -----------------------
PROJECT_DIR = str(Path(__file__).resolve().parents[2])


def _resolve_first_path(*paths: str) -> str:
    """Return the first existing path, or the first candidate if none exist yet."""
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def _resolve_image_dir(*paths: str) -> str:
    """Prefer existing image directories that already contain supported files."""
    first_existing = None
    for path in paths:
        if os.path.isdir(path):
            if first_existing is None:
                first_existing = path
            has_images = any(
                name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
                for name in os.listdir(path)
            )
            if has_images:
                return path
    if first_existing is not None:
        return first_existing
    return paths[0]


def _resolve_file_path(candidates: List[str]) -> str:
    """Return first existing file; else first non-empty candidate for diagnostics."""
    non_empty: List[str] = [path for path in candidates if path]
    for path in non_empty:
        if os.path.isfile(path):
            return path
    return non_empty[0] if non_empty else ""


def _to_hydra_config_name(config_path: str) -> str:
    """
    Convert absolute config file paths into Hydra-compatible config names for `sam2`.
    Hydra `compose(config_name=...)` expects names like `configs/sam2/sam2_hiera_s.yaml`,
    not absolute filesystem paths.
    """
    if not config_path:
        return config_path
    if not os.path.isabs(config_path):
        return config_path

    norm = config_path.replace("\\", "/")
    marker = "/configs/"
    if marker in norm:
        # Keep the package-relative suffix beginning at "configs/..."
        return f"configs/{norm.split(marker, 1)[1]}"
    return os.path.basename(norm)


if resolve_train_image_dir is not None:
    IMAGES_DIR = str(resolve_train_image_dir(Path(PROJECT_DIR)))
    ANNOT_JSON = str(resolve_train_annotations_path(Path(PROJECT_DIR)))
else:
    IMAGES_DIR = _resolve_image_dir(
        os.path.join(PROJECT_DIR, "data", "train_images"),
        os.path.join(PROJECT_DIR, "data", "raw", "train_images_tif"),
        os.path.join(PROJECT_DIR, "data", "raw", "train_images_png"),
        os.path.join(PROJECT_DIR, "train_images"),
    )
    ANNOT_JSON = _resolve_first_path(
        os.path.join(PROJECT_DIR, "data", "train_annotations_updated_504bcc9e05b54435a9a56a841a3a1cf5.json"),
        os.path.join(PROJECT_DIR, "data", "train_annotations.json"),
        os.path.join(PROJECT_DIR, "data", "raw", "annotations", "train_annotations.json"),
        os.path.join(PROJECT_DIR, "train_annotations.json"),
    )
TRAIN_SPLIT_TXT = os.path.join(PROJECT_DIR, "data", "processed", "train.txt")
VAL_SPLIT_TXT = os.path.join(PROJECT_DIR, "data", "processed", "val.txt")
TEST_SPLIT_TXT = os.path.join(PROJECT_DIR, "data", "processed", "test.txt")

ALLOWED_CLASSES = {"individual_tree", "group_of_trees"}  # Classes to include

SAM2_CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints_sam2")
os.makedirs(SAM2_CHECKPOINT_DIR, exist_ok=True)

# Student checkpoint path (same file the API loads). No Meta/public pretrained .pt is used.
SAM2_STUDENT_EXPORT = os.path.join(SAM2_CHECKPOINT_DIR, "sam2_finetuned_tree_canopy.pt")
MODEL_CFG = _resolve_file_path([
    os.getenv("SAM2_MODEL_CFG", "").strip(),
    os.getenv("MODEL_CFG", "").strip(),
    os.path.join(SAM2_CHECKPOINT_DIR, "sam2_hiera_l.yaml"),
    os.path.join(PROJECT_DIR, "sam2_hiera_l.yaml"),
    os.path.join(PROJECT_DIR, "configs", "sam2_hiera_l.yaml"),
    os.path.join(PROJECT_DIR, "backend", "configs", "sam2_hiera_l.yaml"),
    os.path.join(SAM2_CHECKPOINT_DIR, "sam2_hiera_s.yaml"),
    os.path.join(PROJECT_DIR, "sam2_hiera_s.yaml"),
    os.path.join(PROJECT_DIR, "configs", "sam2_hiera_s.yaml"),
    os.path.join(PROJECT_DIR, "backend", "configs", "sam2_hiera_s.yaml"),
    (
        str(Path(_sam2_pkg.__file__).resolve().parent / "sam2_hiera_s.yaml")
        if _sam2_pkg is not None
        else ""
    ),
    (
        str(Path(_sam2_pkg.__file__).resolve().parent / "configs" / "sam2" / "sam2_hiera_s.yaml")
        if _sam2_pkg is not None
        else ""
    ),
])  # Model configuration yaml

BATCH_SIZE = int(os.getenv("SAM2_BATCH_SIZE", "2"))  # Multimodel notebook-style micro-batch
LR = float(os.getenv("SAM2_LR", "1e-4"))  # Learning rate for AdamW
WEIGHT_DECAY = float(os.getenv("SAM2_WEIGHT_DECAY", "1e-4"))  # Match multimodel AdamW wd
STEPS = int(os.getenv("SAM2_STEPS", "3000"))  # Total optimization steps (balanced default)
SAVE_EVERY = int(os.getenv("SAM2_SAVE_EVERY", "200"))  # Save model every N steps

MAX_SIDE = 1024  # Resize so long side <= this
PAD_SIZE = 1024  # Then pad canvas to this size (square)

TRAIN_IMAGE_ENCODER = False  # If True, also train image encoder (see notes above)

RANDOM_SEED = 42  # Seed for reproducibility
TRAIN_RATIO = float(os.getenv("SAM2_TRAIN_RATIO", "0.7"))  # Multimodel notebook ~70% train when no fixed splits

EVAL_OUT_DIR = os.path.join(PROJECT_DIR, "output", "evaluation", "sam2")  # SAM2-scoped eval output root
EVAL_OVERLAYS_DIR = os.path.join(EVAL_OUT_DIR, "overlays")  # Color overlays path
EVAL_MASKS_DIR = os.path.join(EVAL_OUT_DIR, "masks")  # Predicted instance masks path

EVAL_MAX_SIDE = 896  # Eval resize long side for inference
EVAL_GRID_STEP = 48  # Spacing (pixels) between grid prompt points
EVAL_POINTS_CHUNK = 128  # Points per predictor call (batching)
EVAL_MULTIMASK_OUTPUT = False  # Use single mask per prompt
EVAL_TOPK = 200  # Keep top-K masks by score before NMS-like filtering
EVAL_MIN_PIXELS = 50  # Discard tiny masks
EVAL_OVERLAP_FRAC = 0.15  # Reject prediction if >15% overlaps existing occupancy
EVAL_IOU_THRESH = 0.75  # IoU threshold for TP in AP@0.75
# Mask confidence floor (batch_grounded_sam.py uses threshold=0.3 for detections → analogous SAM mask scores)
MIN_MASK_SCORE = float(os.getenv("SAM2_MIN_MASK_SCORE", "0.3"))

CURVES_OUT_DIR = os.path.join(PROJECT_DIR, "output", "curves")  # Folder for training curves
# Resume / periodic saves: primary export for inference; legacy `model.torch` still resumed if present.
MODEL_STATE_PATH = SAM2_STUDENT_EXPORT
MODEL_STATE_LEGACY = os.path.join(SAM2_CHECKPOINT_DIR, "model.torch")
# Mid-run epoch checkpoints (optimizer + scaler + history). Resume with SAM2_RESUME=1.
SAM2_TRAINING_LATEST = os.path.join(SAM2_CHECKPOINT_DIR, "sam2_training_latest.pt")
SAM2_TRAINING_BEST = os.path.join(SAM2_CHECKPOINT_DIR, "sam2_training_best.pt")
SAM2_TRAINING_CKPT_KIND = "sam2_epoch_training_v1"
SAM2_TRAINING_BEST_KIND = "sam2_training_best_v1"
SAM2_RESUME_TRAINING = os.getenv("SAM2_RESUME", "").strip().lower() in ("1", "true", "yes")

# Fine-tuning + box-prompted evaluation (semantic IoU on 3-class mask)
SAM2_TRAIN_EPOCHS = int(os.getenv("SAM2_TRAIN_EPOCHS", "500"))
SAM2_TRAIN_LR = float(os.getenv("SAM2_TRAIN_LR", "1e-5"))
SAM2_TRAIN_WEIGHT_DECAY = float(os.getenv("SAM2_TRAIN_WEIGHT_DECAY", "4e-5"))
SAM2_MAX_INSTANCES_PER_IMAGE = int(os.getenv("SAM2_MAX_INSTANCES_PER_IMAGE", "8"))
SAM2_FREEZE_IMAGE_ENCODER = os.getenv("SAM2_FREEZE_IMAGE_ENCODER", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
SAM2_PROMPT_IMAGE_SIZE = int(os.getenv("SAM2_PROMPT_IMAGE_SIZE", "512"))
SAM2_FINETUNED_WEIGHTS = SAM2_STUDENT_EXPORT
SAM2_EARLY_STOPPING_PATIENCE = int(os.getenv("SAM2_EARLY_STOPPING_PATIENCE", "25"))
SAM2_EARLY_STOPPING_MIN_DELTA = float(os.getenv("SAM2_EARLY_STOPPING_MIN_DELTA", "1e-4"))
# Within-epoch image progress (0 disables mid-epoch lines; epoch start/end always logged).
SAM2_TRAIN_LOG_EVERY = int(os.getenv("SAM2_TRAIN_LOG_EVERY", "10"))
# Meta / public base weights path (required to initialize Hiera before student fine-tuning).
SAM2_BASE_CHECKPOINT = _resolve_file_path(
    [
        os.getenv("SAM2_CHECKPOINT", "").strip(),
        os.getenv("SAM2_BASE_CHECKPOINT", "").strip(),
        os.path.join(SAM2_CHECKPOINT_DIR, "sam2_hiera_large.pt"),
        os.path.join(SAM2_CHECKPOINT_DIR, "sam2_hiera_l.pt"),
        os.path.join(SAM2_CHECKPOINT_DIR, "sam2_hiera_base_plus.pt"),
    ]
)
SAM2_GRID_EVAL = os.getenv("SAM2_GRID_EVAL", "0").strip().lower() in ("1", "true", "yes")

NUM_SAM2_CLASSES = 3
ID_TO_CLASS_SAM2 = {0: "background", 1: "individual_tree", 2: "group_of_trees"}


# -------------------- Data Loading --------------------
def _read_json(annot_path: str) -> Dict[str, Any]:
    """Read the annotations JSON file from disk."""  # Function docstring
    with open(annot_path, "r", encoding="utf-8") as f:  # Open file with UTF-8
        return json.load(f)  # Parse and return JSON object


def _read_split_file(split_path: str) -> List[str]:
    """Read a newline-delimited list of image stems."""
    with open(split_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _list_images(images_dir: str) -> Dict[str, str]:
    """
    Map image stem (filename w/o extension) -> absolute path for supported formats.  # Docstring
    """
    stem_to_path: Dict[str, str] = {}  # Output mapping
    for name in os.listdir(images_dir):  # Iterate directory contents
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):  # Filter by ext
            continue  # Skip non-image files
        stem = os.path.splitext(name)[0]  # Stem without extension
        stem_to_path[stem] = os.path.join(images_dir, name)  # Save absolute path
    return stem_to_path  # Return mapping


def _poly_list_from_flat(seg_flat: List[float]) -> np.ndarray:
    """
    Convert flat list [x1,y1,x2,y2,...] -> (N,2) float32 polygon; require >=3 points.  # Docstring
    """
    arr = np.asarray(seg_flat, dtype=np.float32)  # Convert to numpy array
    assert arr.size % 2 == 0 and arr.size >= 6, "Invalid polygon segmentation array"  # Validate
    return arr.reshape(-1, 2)  # Reshape into Nx2 coordinates


def _resize_and_pad(img: np.ndarray, target_max_side=MAX_SIDE, pad_size=PAD_SIZE) -> Tuple[np.ndarray, float]:
    """
    Resize so long side <= target_max_side (preserve aspect), then pad to (pad_size, pad_size).  # Docstring
    Returns image_out and scale factor r (multiply original coords by r).  # Return details
    """
    h, w = img.shape[:2]  # Height and width
    r = min(target_max_side / float(w), target_max_side / float(h), 1.0)  # Compute scale <=1
    if r < 1.0:  # If downscaling is needed
        img = cv2.resize(img, (int(round(w * r)), int(round(h * r))), interpolation=cv2.INTER_AREA)  # Resize
    ph, pw = img.shape[:2]  # New dims after resize
    if ph < pad_size or pw < pad_size:  # If padding needed to reach canvas
        pad = np.zeros((pad_size, pad_size, img.shape[2]), dtype=img.dtype)  # Create black canvas
        pad[:ph, :pw] = img  # Place image at top-left
        img = pad  # Update image
    return img, r  # Return padded image and scale


def _rasterize_polygons(polys_xy: List[np.ndarray], out_hw: Tuple[int, int]) -> np.ndarray:
    """
    Fill polygons into a binary mask of shape (H,W) using cv2.fillPoly.  # Docstring
    """
    h, w = out_hw  # Target mask size
    mask = np.zeros((h, w), dtype=np.uint8)  # Initialize empty mask
    if not polys_xy:  # If no polygons
        return mask  # Return zeros
    polys = [p.astype(np.int32) for p in polys_xy if p.shape[0] >= 3]  # Keep valid int32 polygons
    if polys:  # If any valid polygon remains
        cv2.fillPoly(mask, polys, color=1)  # Fill with 1s
    return mask  # Return rasterized mask


def _to_resized_polygons(polys_xy: List[np.ndarray], scale: float) -> List[np.ndarray]:
    """Scale polygon coordinates by factor 'scale' (no-op if scale==1)."""  # Docstring
    if scale == 1.0:  # Early exit when no scaling
        return polys_xy  # Return as-is
    return [p * scale for p in polys_xy]  # Scale each polygon


def build_dataset(
    images_dir: str,
    annot_json: str,
    train_split_txt: str | None = None,
    val_split_txt: str | None = None,
    test_split_txt: str | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Construct train/val/(optional test) entry lists from COCO-like JSON.  # Docstring
    Each entry has: image_path, orig_size (H,W), stem, instances[{poly, class}].  # Structure
    """
    data = _read_json(annot_json)  # Load annotations JSON
    stem_to_path = _list_images(images_dir)  # Index available images

    entries: List[Dict[str, Any]] = []  # Collected dataset entries
    for item in data.get("images", []):  # Iterate image metadata
        file_name = item.get("file_name", "")  # Original file reference
        stem = os.path.splitext(os.path.basename(file_name))[0]  # Stem used for matching
        if stem not in stem_to_path:  # If actual image file not found
            continue  # Skip this record
        img_path = stem_to_path[stem]  # Resolve image path
        W = int(item.get("width", 0))  # Original width
        H = int(item.get("height", 0))  # Original height
        ann_list = item.get("annotations", [])  # Polygon annotation list
        instances: List[Dict[str, Any]] = []  # Accumulate valid instances
        for ann in ann_list:  # Iterate annotations
            cls = ann.get("class")  # Class label
            if cls not in ALLOWED_CLASSES:  # Filter out other classes
                continue  # Skip non-target class
            seg = ann.get("segmentation", [])  # Flattened polygon
            try:
                poly = _poly_list_from_flat(seg)  # Convert to Nx2
                if poly.shape[0] >= 3:  # Require at least triangle
                    instances.append({"poly": poly, "class": cls})  # Append instance
            except Exception:
                continue  # Skip malformed segmentation
        if not instances:  # If no valid instances
            continue  # Skip entry
        entries.append({  # Save assembled entry
            "image_path": img_path,  # Full path
            "orig_size": (H, W),  # Original size
            "stem": stem,  # Image stem
            "instances": instances,  # Instances list
        })

    test_entries: List[Dict[str, Any]] = []
    test_path = test_split_txt or TEST_SPLIT_TXT
    if train_split_txt and val_split_txt and os.path.exists(train_split_txt) and os.path.exists(val_split_txt):
        train_stems = set(_read_split_file(train_split_txt))
        val_stems = set(_read_split_file(val_split_txt))
        train_entries = [entry for entry in entries if entry["stem"] in train_stems]
        val_entries = [entry for entry in entries if entry["stem"] in val_stems]
        if os.path.exists(test_path):
            test_stems = set(_read_split_file(test_path))
            test_entries = [entry for entry in entries if entry["stem"] in test_stems]
    else:
        random.Random(RANDOM_SEED).shuffle(entries)  # Deterministic shuffle
        n_train = max(1, int(len(entries) * TRAIN_RATIO))  # Compute train count
        train_entries = entries[:n_train]  # Slice train split
        val_entries = entries[n_train:]  # Slice val split
    return train_entries, val_entries, test_entries  # Return splits


def read_single(entry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """
    Read one image, pick a random instance, rasterize mask, sample one positive point.  # Docstring
    Returns (image_rgb_1024, mask_1024, [[x,y]]).  # Return types
    """
    img_bgr = cv2.imread(entry["image_path"])  # Load image (BGR)
    if img_bgr is None:  # Guard missing file
        raise FileNotFoundError(entry["image_path"])  # Informative error
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB

    img_resized, r = _resize_and_pad(img_rgb, target_max_side=MAX_SIDE, pad_size=PAD_SIZE)  # Resize+pad
    img_resized = np.ascontiguousarray(img_resized)  # Ensure contiguous memory
    ph, pw = img_resized.shape[:2]  # Dimensions after padding

    instances = entry["instances"]  # Available polygons
    poly = random.choice(instances)["poly"]  # Choose one polygon randomly
    poly_resized = (poly * r)  # Scale coordinates to resized image
    mask = _rasterize_polygons([poly_resized], (ph, pw))  # Rasterize to binary mask

    coords = np.argwhere(mask > 0)  # All positive pixels (y,x)
    if coords.size == 0:  # If mask is empty (edge case)
        return read_single(entry)  # Retry by recursion
    yx = coords[np.random.randint(len(coords))]  # Sample one pixel uniformly
    y, x = int(yx[0]), int(yx[1])  # Extract coordinates
    return img_resized, mask.astype(np.uint8), [[x, y]]  # Return image, mask, point [[x,y]]


def read_batch(entries: List[Dict[str, Any]], batch_size: int = BATCH_SIZE):
    """
    Assemble a mini-batch by sampling 'batch_size' entries via read_single.  # Docstring
    Returns (images:list, masks:np.ndarray, points:np.ndarray, labels:np.ndarray).  # Return details
    """
    limage = []  # List of RGB images
    lmask = []  # List of masks
    linput_point = []  # List of [[x,y]] point arrays
    for _ in range(batch_size):  # Repeat batch_size times
        entry = random.choice(entries)  # Random entry
        image, mask, input_point = read_single(entry)  # Sample one
        limage.append(image)  # Accumulate image
        lmask.append(mask)  # Accumulate mask
        linput_point.append(input_point)  # Accumulate point
    return limage, np.array(lmask), np.array(linput_point), np.ones([batch_size, 1], dtype=np.int32)  # Labels=1


# -------------------- SAM2 fine-tuning + box-prompted semantic eval --------------------
def _class_name_to_id(name: str) -> int:
    if name == "individual_tree":
        return 1
    if name == "group_of_trees":
        return 2
    return 0


def _safe_nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _metric_dict_for_json(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)) and (math.isnan(float(v)) or math.isinf(float(v))):
            out[k] = None
        elif isinstance(v, (float, np.floating)):
            out[k] = round(float(v), 6)
        else:
            out[k] = v
    return out


def _optional_round_mean_iou(m: Dict[str, Any]) -> float | None:
    x = m.get("mean_iou", float("nan"))
    if isinstance(x, (float, np.floating)) and (math.isnan(float(x)) or math.isinf(float(x))):
        return None
    return round(float(x), 6)


def _compute_pixel_accuracy_semantic(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(pred == gt))


def _compute_iou_per_class_semantic(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for c in range(num_classes):
        p = pred == c
        g = gt == c
        inter = int(np.logical_and(p, g).sum())
        union = int(np.logical_or(p, g).sum())
        out[c] = float(inter) / float(union + 1e-6)
    return out


def _empty_prompted_metrics() -> Dict[str, Any]:
    m: Dict[str, Any] = {"pixel_accuracy": float("nan"), "mean_iou": float("nan")}
    for cid in range(NUM_SAM2_CLASSES):
        m[f"iou_{ID_TO_CLASS_SAM2[cid]}"] = float("nan")
    return m


def _read_rgb_for_prompt(image_path: str) -> np.ndarray:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def annotations_to_instance_targets_from_entry(entry: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    H0, W0 = entry["orig_size"]
    boxes_list: List[List[float]] = []
    masks_list: List[np.ndarray] = []
    labels_list: List[int] = []
    for inst in entry["instances"]:
        poly = inst["poly"]
        m = _rasterize_polygons([poly], (H0, W0))
        if int(m.sum()) == 0:
            continue
        ys, xs = np.where(m > 0)
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        cid = _class_name_to_id(inst["class"])
        if cid == 0:
            continue
        boxes_list.append([x1, y1, x2, y2])
        masks_list.append(m.astype(np.uint8))
        labels_list.append(cid)
    if not boxes_list:
        return {
            "boxes": torch.zeros(0, 4, dtype=torch.float32),
            "masks": torch.zeros(0, H0, W0, dtype=torch.uint8),
            "labels": torch.zeros(0, dtype=torch.long),
        }
    return {
        "boxes": torch.tensor(boxes_list, dtype=torch.float32),
        "masks": torch.tensor(np.stack(masks_list, axis=0), dtype=torch.uint8),
        "labels": torch.tensor(labels_list, dtype=torch.long),
    }


def annotations_to_semantic_mask_from_entry(entry: Dict[str, Any]) -> np.ndarray:
    H0, W0 = entry["orig_size"]
    out = np.zeros((H0, W0), dtype=np.uint8)
    for inst in entry["instances"]:
        cid = _class_name_to_id(inst["class"])
        if cid == 0:
            continue
        m = _rasterize_polygons([inst["poly"]], (H0, W0))
        out[m > 0] = cid
    return out


def resize_binary_mask(mask_np: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(mask_np.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)


def scale_box_xyxy(box: np.ndarray, orig_w: int, orig_h: int, size: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    sx = size / float(orig_w)
    sy = size / float(orig_h)
    return np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)


def sam2_set_trainable_parts(predictor: SAM2ImagePredictor, freeze_image_encoder: bool = True) -> None:
    model = predictor.model
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.sam_prompt_encoder.parameters():
        p.requires_grad = True
    for p in model.sam_mask_decoder.parameters():
        p.requires_grad = True
    model.sam_prompt_encoder.train(True)
    model.sam_mask_decoder.train(True)
    if hasattr(model, "image_encoder"):
        if freeze_image_encoder:
            model.image_encoder.eval()
            for p in model.image_encoder.parameters():
                p.requires_grad = False
        else:
            model.image_encoder.train(True)
            for p in model.image_encoder.parameters():
                p.requires_grad = True


def binary_mask_loss_from_logits(logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    gt_mask = gt_mask.float()
    bce = F.binary_cross_entropy_with_logits(logits, gt_mask)
    probs = torch.sigmoid(logits)
    inter = (probs * gt_mask).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2))
    dice = 1.0 - ((2.0 * inter + 1e-6) / (union + 1e-6))
    dice = dice.mean()
    return bce + dice


def _sam2_compute_loss_for_prompted_instance(
    predictor: SAM2ImagePredictor,
    box_xyxy: np.ndarray,
    gt_mask_1hw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Box is xyxy in resized image space; ``gt_mask_1hw`` is (1,H,W) float on device. Call inside ``autocast`` when using AMP.

    Returns ``(loss, pixel_acc)`` where ``pixel_acc`` is detached (fraction of pixels where pred mask matches GT at 0.5 threshold).
    """
    _, _, _, unnorm_box = predictor._prep_prompts(
        point_coords=None,
        point_labels=None,
        box=box_xyxy[None, :],
        mask_logits=None,
        normalize_coords=True,
    )
    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=None,
        boxes=unnorm_box,
        masks=None,
    )
    high_res_features = [
        feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]
    ]
    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"],
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )
    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
    prd_mask_logits = prd_masks[:, 0]
    seg_loss = binary_mask_loss_from_logits(prd_mask_logits, gt_mask_1hw)
    pred_binary = (torch.sigmoid(prd_mask_logits) > 0.5).float()
    inter = (pred_binary * gt_mask_1hw).sum(dim=(1, 2))
    union = pred_binary.sum(dim=(1, 2)) + gt_mask_1hw.sum(dim=(1, 2)) - inter
    iou = inter / (union + 1e-6)
    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
    loss = seg_loss + 0.05 * score_loss
    pixel_acc = pred_binary.eq(gt_mask_1hw).float().mean().detach()
    return loss, pixel_acc


def _sam2_clone_state_dict_cpu(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _load_sam2_student_weights(predictor: SAM2ImagePredictor, resume_path: str, device: str) -> None:
    try:
        blob = torch.load(resume_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(resume_path, map_location=device)
    if isinstance(blob, dict) and "finetuned_state_dict" in blob:
        state = blob["finetuned_state_dict"]
    elif isinstance(blob, dict) and "model_state_dict" in blob:
        state = blob["model_state_dict"]
    elif isinstance(blob, dict) and "state_dict" in blob:
        state = blob["state_dict"]
    elif isinstance(blob, dict):
        state = blob
    else:
        raise TypeError("Unexpected SAM2 checkpoint format")
    predictor.model.load_state_dict(state, strict=False)


def _save_sam2_training_latest(
    path: str,
    *,
    predictor: SAM2ImagePredictor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch_next: int,
    max_epochs: int,
    early_stopper: EarlyStopping | None,
    history: Dict[str, List[float]],
    model_cfg_hydra: str,
    stopped_early: bool,
) -> None:
    blob: Dict[str, Any] = {
        "kind": SAM2_TRAINING_CKPT_KIND,
        "model_cfg_hydra": model_cfg_hydra,
        "base_config": str(MODEL_CFG),
        "base_checkpoint": str(SAM2_BASE_CHECKPOINT),
        "epoch_next": int(epoch_next),
        "max_epochs": int(max_epochs),
        "finetuned_state_dict": predictor.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": {k: list(v) for k, v in history.items()},
        "stopped_early": bool(stopped_early),
    }
    if early_stopper is not None:
        blob["early_stopper_best_score"] = early_stopper.best_score
        blob["early_stopper_counter"] = int(early_stopper.counter)
        blob["early_stopper_patience"] = int(early_stopper.patience)
        blob["early_stopper_min_delta"] = float(early_stopper.min_delta)
        blob["early_stopper_mode"] = early_stopper.mode
    torch.save(blob, path)


def _save_sam2_training_best(
    path: str,
    *,
    predictor: SAM2ImagePredictor,
    epoch_1based: int,
    val_loss: float,
    model_cfg_hydra: str,
) -> None:
    torch.save(
        {
            "kind": SAM2_TRAINING_BEST_KIND,
            "model_cfg_hydra": model_cfg_hydra,
            "base_config": str(MODEL_CFG),
            "base_checkpoint": str(SAM2_BASE_CHECKPOINT),
            "epoch": int(epoch_1based),
            "val_loss": float(val_loss),
            "finetuned_state_dict": _sam2_clone_state_dict_cpu(predictor.model),
        },
        path,
    )


def _load_sam2_training_best_weights_cpu(path: str) -> Dict[str, torch.Tensor] | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(blob, dict) or blob.get("kind") != SAM2_TRAINING_BEST_KIND:
        return None
    state = blob.get("finetuned_state_dict")
    if not isinstance(state, dict):
        return None
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _try_resume_sam2_training_from_latest(
    path: str,
    *,
    predictor: SAM2ImagePredictor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    early_stopper: EarlyStopping | None,
    history: Dict[str, List[float]],
    model_cfg_hydra: str,
    device: str,
) -> int:
    """Load latest epoch checkpoint; return epoch_next (0-based index of next epoch to run)."""
    if not os.path.isfile(path):
        return 0
    try:
        try:
            blob = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location=device)
    except Exception as e:
        print(f"[WARN] Could not read latest training checkpoint: {e}", flush=True)
        return 0
    if not isinstance(blob, dict) or blob.get("kind") != SAM2_TRAINING_CKPT_KIND:
        print("[WARN] Latest training checkpoint has wrong format; starting from epoch 0.", flush=True)
        return 0
    if str(blob.get("model_cfg_hydra", "")) != str(model_cfg_hydra):
        print(
            f"[WARN] Latest checkpoint cfg {blob.get('model_cfg_hydra')} != {model_cfg_hydra}; not resuming.",
            flush=True,
        )
        return 0
    if str(blob.get("base_checkpoint", "")) != str(SAM2_BASE_CHECKPOINT):
        print("[WARN] Latest checkpoint base_checkpoint path differs from current; not resuming.", flush=True)
        return 0
    try:
        predictor.model.load_state_dict(blob["finetuned_state_dict"], strict=False)
    except Exception as e:
        print(f"[WARN] Failed to load model state from latest checkpoint: {e}", flush=True)
        return 0
    try:
        optimizer.load_state_dict(blob["optimizer_state_dict"])
    except Exception as e:
        print(f"[WARN] Failed to load optimizer state: {e}", flush=True)
    try:
        scaler.load_state_dict(blob["scaler_state_dict"])
    except Exception as e:
        print(f"[WARN] Failed to load scaler state: {e}", flush=True)
    if early_stopper is not None:
        if blob.get("early_stopper_best_score") is not None:
            try:
                early_stopper.best_score = float(blob["early_stopper_best_score"])
            except (TypeError, ValueError):
                early_stopper.best_score = None
        early_stopper.counter = int(blob.get("early_stopper_counter", 0))
    hist = blob.get("history")
    if isinstance(hist, dict):
        for key in history:
            if key in hist and isinstance(hist[key], list):
                history[key] = [float(x) for x in hist[key]]
    epoch_next = int(blob.get("epoch_next", 0))
    print(
        f"[INFO] Resumed training from {path} (next epoch index {epoch_next}, "
        f"{len(history.get('train_loss', []))} epochs in history).",
        flush=True,
    )
    return max(0, epoch_next)


def train_one_epoch_sam2(
    predictor: SAM2ImagePredictor,
    train_entries: List[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: str,
    max_instances_per_image: int = 8,
    image_size: int = SAM2_PROMPT_IMAGE_SIZE,
) -> Tuple[float, float]:
    """Returns ``(mean_loss, mean_pixel_acc)`` over box-prompt instances (nan if no steps)."""
    sam2_set_trainable_parts(predictor, freeze_image_encoder=SAM2_FREEZE_IMAGE_ENCODER)
    running_losses: List[float] = []
    running_accs: List[float] = []
    shuffled = train_entries.copy()
    random.shuffle(shuffled)
    dev = torch.device(device)
    autocast_enabled = dev.type == "cuda"
    n_shuf = len(shuffled)
    log_every = SAM2_TRAIN_LOG_EVERY

    for i, entry in enumerate(shuffled, start=1):
        try:
            image = _read_rgb_for_prompt(entry["image_path"])
        except FileNotFoundError:
            continue
        image_resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        target = annotations_to_instance_targets_from_entry(entry)
        if target["boxes"].shape[0] == 0:
            continue

        H0, W0 = entry["orig_size"]
        predictor.set_image(image_resized)

        boxes = target["boxes"].cpu().numpy()
        masks = target["masks"].cpu().numpy()

        idxs = list(range(len(boxes)))
        random.shuffle(idxs)
        idxs = idxs[:max_instances_per_image]

        img_losses: List[float] = []
        img_accs: List[float] = []

        for idx in idxs:
            box = scale_box_xyxy(boxes[idx], W0, H0, image_size)
            gt_mask_np = resize_binary_mask(masks[idx], image_size)
            if int(gt_mask_np.sum()) == 0:
                continue

            gt_mask = torch.tensor(gt_mask_np[None].astype(np.float32), device=device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                loss, px_acc = _sam2_compute_loss_for_prompted_instance(predictor, box, gt_mask)

            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            img_losses.append(float(loss.detach().cpu().item()))
            img_accs.append(float(px_acc.cpu().item()))

        if img_losses:
            running_losses.append(float(np.mean(img_losses)))
        if img_accs:
            running_accs.append(float(np.mean(img_accs)))

        if log_every > 0 and n_shuf and i % log_every == 0:
            print(f"[INFO]   training progress: {i}/{n_shuf} images", flush=True)

    mean_loss = float(np.mean(running_losses)) if running_losses else float("nan")
    mean_acc = float(np.mean(running_accs)) if running_accs else float("nan")
    return mean_loss, mean_acc


@torch.no_grad()
def eval_one_epoch_sam2_val_loss(
    predictor: SAM2ImagePredictor,
    val_entries: List[Dict[str, Any]],
    device: str,
    max_instances_per_image: int = 8,
    image_size: int = SAM2_PROMPT_IMAGE_SIZE,
) -> Tuple[float, float]:
    """Mean box-prompt loss and pixel accuracy on the validation split (no backward)."""
    if not val_entries:
        return float("nan"), float("nan")
    predictor.model.eval()
    running_losses: List[float] = []
    running_accs: List[float] = []
    autocast_enabled = torch.device(device).type == "cuda"
    n_val = len(val_entries)
    log_every = SAM2_TRAIN_LOG_EVERY

    for i, entry in enumerate(val_entries, start=1):
        try:
            image = _read_rgb_for_prompt(entry["image_path"])
        except FileNotFoundError:
            continue
        image_resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        target = annotations_to_instance_targets_from_entry(entry)
        if target["boxes"].shape[0] == 0:
            continue

        H0, W0 = entry["orig_size"]
        predictor.set_image(image_resized)

        boxes = target["boxes"].cpu().numpy()
        masks = target["masks"].cpu().numpy()

        idxs = list(range(len(boxes)))[:max_instances_per_image]

        img_losses: List[float] = []
        img_accs: List[float] = []
        for idx in idxs:
            box = scale_box_xyxy(boxes[idx], W0, H0, image_size)
            gt_mask_np = resize_binary_mask(masks[idx], image_size)
            if int(gt_mask_np.sum()) == 0:
                continue
            gt_mask = torch.tensor(gt_mask_np[None].astype(np.float32), device=device)
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                loss, px_acc = _sam2_compute_loss_for_prompted_instance(predictor, box, gt_mask)
            img_losses.append(float(loss.detach().cpu().item()))
            img_accs.append(float(px_acc.cpu().item()))

        if img_losses:
            running_losses.append(float(np.mean(img_losses)))
        if img_accs:
            running_accs.append(float(np.mean(img_accs)))

        if log_every > 0 and n_val and i % log_every == 0:
            print(f"[INFO]   validation progress: {i}/{n_val} images", flush=True)

    mean_loss = float(np.mean(running_losses)) if running_losses else float("nan")
    mean_acc = float(np.mean(running_accs)) if running_accs else float("nan")
    return mean_loss, mean_acc


@torch.no_grad()
def evaluate_sam2_prompted(
    records: List[Dict[str, Any]],
    predictor: SAM2ImagePredictor,
    image_size: int = SAM2_PROMPT_IMAGE_SIZE,
) -> Dict[str, Any]:
    if not records:
        return _empty_prompted_metrics()

    predictor.model.eval()
    pixel_accs: List[float] = []
    iou_store: Dict[int, List[float]] = defaultdict(list)

    for entry in records:
        try:
            image = _read_rgb_for_prompt(entry["image_path"])
        except FileNotFoundError:
            continue
        image_resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        gt_mask = annotations_to_semantic_mask_from_entry(entry)
        gt_mask = cv2.resize(gt_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

        predictor.set_image(image_resized)
        pred_mask = np.zeros((image_size, image_size), dtype=np.uint8)

        target = annotations_to_instance_targets_from_entry(entry)
        if target["boxes"].shape[0] == 0:
            pixel_accs.append(_compute_pixel_accuracy_semantic(pred_mask, gt_mask))
            ious = _compute_iou_per_class_semantic(pred_mask, gt_mask, NUM_SAM2_CLASSES)
            for cls_id, iou in ious.items():
                iou_store[cls_id].append(iou)
            continue

        H0, W0 = entry["orig_size"]
        boxes = target["boxes"].numpy().copy()
        labels = target["labels"].numpy().copy()

        sx = image_size / float(W0)
        sy = image_size / float(H0)
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[:, 0]

        order = np.argsort(np.asarray(scores).reshape(-1))
        for idx in order:
            m = masks[idx]
            cls_id = int(labels[idx])
            pred_mask[m > 0] = cls_id

        pixel_accs.append(_compute_pixel_accuracy_semantic(pred_mask, gt_mask))
        ious = _compute_iou_per_class_semantic(pred_mask, gt_mask, NUM_SAM2_CLASSES)
        for cls_id, iou in ious.items():
            iou_store[cls_id].append(iou)

    metrics: Dict[str, Any] = {
        "pixel_accuracy": _safe_nanmean(pixel_accs),
        "mean_iou": _safe_nanmean([_safe_nanmean(v) for v in iou_store.values()]),
    }
    for cls_id in range(NUM_SAM2_CLASSES):
        metrics[f"iou_{ID_TO_CLASS_SAM2[cls_id]}"] = _safe_nanmean(iou_store[cls_id])
    return metrics


# -------------------- Evaluation Utils --------------------
def _ensure_dir(p: str):
    """Create directory 'p' if missing (safe to call repeatedly)."""  # Docstring
    os.makedirs(p, exist_ok=True)  # Make dirs


def _validate_runtime_prerequisites() -> None:
    """Validate required runtime dependencies and minimum model assets."""
    missing_assets = []
    if not os.path.exists(MODEL_CFG):
        missing_assets.append(f"MODEL_CFG not found: {MODEL_CFG}")

    if build_sam2 is None or SAM2ImagePredictor is None:
        raise RuntimeError(
            "Could not import SAM2 modules (`sam2.build_sam`, `sam2.sam2_image_predictor`). "
            "Install/configure the Segment Anything 2 environment first."
        )

    if missing_assets:
        raise RuntimeError(
            "Missing required SAM2 model files:\n- "
            + "\n- ".join(missing_assets)
            + "\n\nTip: set SAM2_MODEL_CFG env var to an explicit config path."
        )


def _load_and_resize_for_eval(image_path: str, max_side=EVAL_MAX_SIDE):
    """
    Load RGB image, resize so long side<=max_side, return resized image, scale, orig (H0,W0), and original RGB.  # Docstring
    """
    bgr = cv2.imread(image_path)  # Read image
    if bgr is None:  # Validate existence
        raise FileNotFoundError(image_path)  # Error if missing
    rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB
    H0, W0 = rgb0.shape[:2]  # Original dims
    r = min(max_side / float(W0), max_side / float(H0), 1.0)  # Scale factor
    if r < 1.0:  # If downscale needed
        rgb = cv2.resize(rgb0, (int(round(W0 * r)), int(round(H0 * r))), interpolation=cv2.INTER_AREA)  # Resize
    else:
        rgb = rgb0.copy()  # No change
    return rgb, r, (H0, W0), rgb0  # Return resized, scale, orig size, original


def _make_point_prompts(H: int, W: int, step: int = EVAL_GRID_STEP):
    """
    Create a dense grid of positive point prompts across the HxW image.  # Docstring
    """
    xs = np.arange(step // 2, W, step)  # X positions
    ys = np.arange(step // 2, H, step)  # Y positions
    coords = [[[int(x), int(y)]] for y in ys for x in xs]  # [[x,y]] per point
    if not coords:  # Edge case tiny images
        coords = [[[W // 2, H // 2]]]  # Fallback to center
    point_coords = np.array(coords, dtype=np.int32)  # Int coordinates
    point_labels = np.ones((point_coords.shape[0], 1), dtype=np.int32)  # All positives
    return point_coords, point_labels  # Return coords and labels


def _rasterize_single_poly_to_mask(poly_xy: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Rasterize single polygon to boolean mask of size hw."""  # Docstring
    return _rasterize_polygons([poly_xy], hw).astype(bool)  # Fill and cast to bool


def _compute_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """Compute IoU between two boolean masks m1 and m2."""  # Docstring
    inter = np.logical_and(m1, m2).sum()  # Intersection pixels
    union = np.logical_or(m1, m2).sum()  # Union pixels
    return float(inter) / float(union + 1e-6)  # IoU with epsilon


def evaluate_on_validation(val_entries: List[Dict[str, Any]], predictor: SAM2ImagePredictor, device: str = "cpu"):
    """
    Validation loop: propose binary masks via grid points, filter, save overlays/masks,
    then report AP@0.75-vs-class-GT and IoU histograms.  # Docstring
    """
    _ensure_dir(EVAL_OUT_DIR)  # Ensure output root exists
    _ensure_dir(EVAL_OVERLAYS_DIR)  # Ensure overlays dir
    _ensure_dir(EVAL_MASKS_DIR)  # Ensure masks dir

    gt_index: Dict[str, Dict[str, List[np.ndarray]]] = {}  # stem -> class -> list of GT polys
    for e in val_entries:  # Build GT mapping
        stem = e["stem"]  # Image id
        H0, W0 = e["orig_size"]  # Original size (unused directly here, but recorded)
        cls_map: Dict[str, List[np.ndarray]] = {c: [] for c in ALLOWED_CLASSES}  # Init per class
        for inst in e["instances"]:  # For each instance polygon
            poly = inst["poly"]  # GT polygon
            cls = inst["class"]  # GT class
            cls_map.setdefault(cls, []).append(poly)  # Append to class list
        gt_index[stem] = cls_map  # Save per-image GT map

    pred_records: Dict[str, List[Tuple[float, int]]] = {c: [] for c in ALLOWED_CLASSES}  # Score,TP flag per class
    tp_ious_total: List[float] = []  # IoUs of true positives for TP-only histogram
    matched_ious: List[float] = []  # IoUs for matched predictions
    unmatched_ious: List[float] = []  # Best IoUs for unmatched predictions (often 0)

    for e in tqdm(val_entries, desc="Val Inference", dynamic_ncols=True):  # Iterate validation images with progress
        path = e["image_path"]  # File path
        stem = e["stem"]  # Stem id
        img_resized, r, (H0, W0), img_orig = _load_and_resize_for_eval(path, max_side=EVAL_MAX_SIDE)  # Prep image
        H, W = img_resized.shape[:2]  # Resized dims

        with torch.no_grad():  # Inference without gradients
            if device == "cuda":  # If on GPU
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # Mixed precision context
            else:
                autocast_ctx = nullcontext()  # No mixed precision context on CPU
            with autocast_ctx:  # Enter context manager
                predictor.set_image(img_resized)  # Encode resized image
                pts, lbls = _make_point_prompts(H, W, step=EVAL_GRID_STEP)  # Generate grid prompts
                all_masks = []  # Accumulate masks chunks
                all_scores = []  # Accumulate scores chunks
                n = pts.shape[0]  # Total prompts
                for i0 in range(0, n, EVAL_POINTS_CHUNK):  # Chunked inference
                    i1 = min(n, i0 + EVAL_POINTS_CHUNK)  # End index
                    m, s, _ = predictor.predict(  # Predict masks for chunk
                        point_coords=pts[i0:i1],  # Chunked coords
                        point_labels=lbls[i0:i1],  # Chunked labels
                        multimask_output=EVAL_MULTIMASK_OUTPUT,  # Single mask per prompt
                    )
                    all_masks.append(m[:, 0])  # Take first mask per prompt
                    all_scores.append(s[:, 0])  # Take first score per prompt
                masks = np.concatenate(all_masks, axis=0).astype(bool)  # Stack masks
                scores = np.concatenate(all_scores, axis=0)  # Stack scores

        order = np.argsort(scores)[::-1][: min(len(scores), EVAL_TOPK)]  # Sort by score desc and keep top-K
        masks = masks[order]  # Reorder masks
        scores = scores[order]  # Reorder scores

        seg_map = np.zeros((H, W), dtype=np.int32)  # Instance map initialized to 0
        occupancy = np.zeros((H, W), dtype=bool)  # Occupancy mask to prevent overlaps
        color_map = np.zeros((H, W, 3), dtype=np.uint8)  # Random color overlay
        rng = np.random.default_rng(2024)  # Deterministic RNG for colors

        kept_scores: List[float] = []  # Scores for kept instances
        kept_count = 0  # Instance counter
        for m, sc in zip(masks, scores):  # Iterate candidate masks
            if float(sc) < MIN_MASK_SCORE:  # Align with Grounded-SAM-style detection threshold (0.3)
                continue
            if m.sum() < EVAL_MIN_PIXELS:  # Drop tiny masks
                continue  # Skip
            overlap = (m & occupancy).sum()  # Overlap pixel count
            if overlap > 0 and overlap / float(m.sum()) > EVAL_OVERLAP_FRAC:  # Too much overlap
                continue  # Reject overlapping prediction
            m = m & (~occupancy)  # Remove overlapped pixels to keep only new area
            if m.sum() < EVAL_MIN_PIXELS:  # Recheck size after trimming
                continue  # Skip
            kept_count += 1  # New instance id
            seg_map[m] = kept_count  # Assign instance id to pixels
            occupancy[m] = True  # Update occupancy
            color = rng.integers(0, 256, size=3, dtype=np.uint8)  # Random color
            color_map[m] = color  # Color those pixels
            kept_scores.append(float(sc))  # Record score

        overlay_small = ((color_map.astype(np.float32) + img_resized.astype(np.float32)) / 2).astype(np.uint8)  # Blend
        overlay_orig = overlay_small if r == 1.0 else cv2.resize(overlay_small, (W0, H0), interpolation=cv2.INTER_AREA)  # Upscale
        color_map_orig = color_map if r == 1.0 else cv2.resize(color_map, (W0, H0), interpolation=cv2.INTER_NEAREST)  # Nearest for labels
        cv2.imwrite(os.path.join(EVAL_OVERLAYS_DIR, f"{stem}.png"), cv2.cvtColor(overlay_orig, cv2.COLOR_RGB2BGR))  # Save overlay

        instance_ids = np.unique(seg_map)[1:]  # All instance ids (skip 0)
        preds_img: List[Tuple[np.ndarray, float]] = []  # (mask, score) per instance
        for iid in instance_ids:  # For each kept instance
            m_small = (seg_map == iid)  # Mask in resized space
            m_orig = m_small if r == 1.0 else cv2.resize(m_small.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)  # Upscale
            score = kept_scores[iid - 1] if 0 <= (iid - 1) < len(kept_scores) else 0.5  # Retrieve score
            preds_img.append((m_orig, score))  # Append prediction
            cv2.imwrite(os.path.join(EVAL_MASKS_DIR, f"{stem}_mask_{iid:04d}.png"), (m_orig.astype(np.uint8) * 255))  # Save mask

        for cls in ALLOWED_CLASSES:  # Evaluate per class
            gt_polys = gt_index.get(stem, {}).get(cls, [])  # GT polygons for this class
            if not gt_polys:  # If no GT of this class
                for pred_mask, score in sorted(preds_img, key=lambda x: x[1], reverse=True):  # All predictions unmatched
                    pred_records[cls].append((float(score), 0))  # Record as FP for AP calc
                    unmatched_ious.append(0.0)  # Best IoU = 0 when no GT
                continue  # Next class

            gt_masks = [_rasterize_single_poly_to_mask(p, (H0, W0)) for p in gt_polys]  # Rasterize GTs
            used = set()  # Track matched GT indices

            for pred_mask, score in sorted(preds_img, key=lambda x: x[1], reverse=True):  # Greedy matching
                best_iou = 0.0  # Initialize best IoU
                best_j = -1  # Initialize best GT index
                for j, gm in enumerate(gt_masks):  # Compare to all GTs
                    iou = _compute_iou(pred_mask, gm)  # Compute IoU
                    if iou > best_iou:  # Keep best
                        best_iou = iou  # Update best IoU
                        best_j = j  # Update best index
                if best_iou >= EVAL_IOU_THRESH and best_j >= 0 and best_j not in used:  # True positive condition
                    used.add(best_j)  # Mark GT as used
                    pred_records[cls].append((float(score), 1))  # Record TP
                    tp_ious_total.append(best_iou)  # For TP-only histogram
                    matched_ious.append(best_iou)  # For matched histogram
                else:
                    pred_records[cls].append((float(score), 0))  # Record FP
                    unmatched_ious.append(best_iou)  # Best IoU (often 0) for unmatched histogram

    ap_per_class: Dict[str, float] = {}  # Results container for AP
    total_gts_per_cls = {c: 0 for c in ALLOWED_CLASSES}  # Count GTs per class
    for e in val_entries:  # Count GT instances
        for inst in e["instances"]:  # For each instance
            total_gts_per_cls[inst["class"]] += 1  # Increment class count
    for cls in ALLOWED_CLASSES:  # Compute 11-point AP per class
        recs = pred_records.get(cls, [])  # Collected (score,TP) records
        total_gts = total_gts_per_cls.get(cls, 0)  # Number of positives
        if not recs or total_gts == 0:  # Edge cases
            ap_per_class[cls] = 0.0  # AP=0 when no data
            continue  # Next class
        recs_sorted = sorted(recs, key=lambda x: x[0], reverse=True)  # Sort by score
        tp_flags = np.array([int(t == 1) for _, t in recs_sorted], dtype=np.int32)  # TP flags
        fp_flags = 1 - tp_flags  # FP flags
        tp_cum = np.cumsum(tp_flags)  # Cumulative TP
        fp_cum = np.cumsum(fp_flags)  # Cumulative FP
        recalls = tp_cum / max(total_gts, 1)  # Recall curve
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)  # Precision curve
        ap = 0.0  # Initialize AP
        for t in np.linspace(0, 1, 11):  # 11-point interpolation
            p = precisions[recalls >= t].max() if np.any(recalls >= t) else 0.0  # Max precision at >= recall t
            ap += p / 11.0  # Average
        ap_per_class[cls] = float(ap)  # Store AP

    try:
        import matplotlib.pyplot as plt  # Plotting library

        if len(tp_ious_total) > 0:  # Only if we have TPs
            plt.figure(figsize=(6, 4))  # New figure
            plt.hist(tp_ious_total, bins=20, range=(0, 1), color="#4C72B0")  # Histogram of TP IoUs
            plt.xlabel("IoU (TP only)")  # X label
            plt.ylabel("Count")  # Y label
            plt.title("IoU Histogram (Validation, TP-only)")  # Title
            plt.tight_layout()  # Layout
            plt.savefig(os.path.join(EVAL_OUT_DIR, "iou_hist_tp_only.png"))  # Save file
            plt.close()  # Close figure
        else:
            print("[WARN] No true positives on validation; skipping iou_hist_tp_only.png")  # Warn empty

        if (len(matched_ious) + len(unmatched_ious)) > 0:  # If any predictions collected
            plt.figure(figsize=(8, 5))  # New figure
            bins = 25  # Number of bins
            counts_m, edges_m, _ = plt.hist(matched_ious, bins=bins, range=(0, 1), alpha=0.7, label="Matched")  # Matched hist
            counts_u, edges_u, _ = plt.hist(unmatched_ious, bins=bins, range=(0, 1), alpha=0.7, label="Unmatched")  # Unmatched hist

            def _annotate_counts(counts, edges, y_shift=0.0):  # Helper to annotate bars
                for c, (x0, x1) in zip(counts, zip(edges[:-1], edges[1:])):  # Iterate bins
                    if c > 0:  # Only annotate non-zero
                        xc = (x0 + x1) / 2.0  # Bin center
                        plt.text(  # Draw text
                            xc,
                            c + 0.02 * max(max(counts_m, default=0), max(counts_u, default=0)),  # Slightly above bar
                            f"{int(c)}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

            _annotate_counts(counts_u, edges_u)  # Annotate unmatched first
            _annotate_counts(counts_m, edges_m)  # Then matched

            plt.axvline(EVAL_IOU_THRESH, linestyle="--", color="red", linewidth=2, label=f"IoU thresh {EVAL_IOU_THRESH}")  # Threshold line
            plt.xlabel("IoU")  # X label
            plt.ylabel("Count")  # Y label
            plt.title("IoU Distribution (Matched/Unmatched)")  # Title
            plt.legend()  # Legend
            plt.tight_layout()  # Layout
            plt.savefig(os.path.join(EVAL_OUT_DIR, "iou_hist.png"))  # Save figure
            plt.close()  # Close
        else:
            print("[WARN] No predictions collected to plot IoU distribution.")  # Inform no data

        plt.figure(figsize=(6, 4))  # AP bar chart figure
        classes_sorted = sorted(list(ALLOWED_CLASSES))  # Consistent class order
        xs = np.arange(len(classes_sorted))  # Bar positions
        vals = [ap_per_class.get(c, 0.0) for c in classes_sorted]  # AP values
        plt.bar(xs, vals, tick_label=classes_sorted, color="#55A868")  # Draw bars
        plt.ylim(0, 1)  # Limit to [0,1]
        plt.ylabel("AP@0.75 vs class GT")  # Y label
        plt.title("Binary-mask AP per class GT (Validation)")  # Title
        for i, v in enumerate(vals):  # Annotate bars
            plt.text(i, min(v + 0.02, 0.98), f"{v:.2f}", ha="center")  # Text above bar
        plt.tight_layout()  # Layout
        plt.savefig(os.path.join(EVAL_OUT_DIR, "ap_per_class.png"))  # Save
        plt.close()  # Close
    except Exception as e:
        print("[WARN] Could not generate validation plots:", e)  # Plotting error message

    summary = {
        "num_images": int(len(val_entries)),
        "output_dir": EVAL_OUT_DIR,
        "iou_threshold": float(EVAL_IOU_THRESH),
        "ap_per_class": {str(k): float(v) for k, v in ap_per_class.items()},
        "gt_instances_per_class": {str(k): int(v) for k, v in total_gts_per_cls.items()},
        "prediction_records_per_class": {str(k): int(len(pred_records.get(k, []))) for k in ALLOWED_CLASSES},
        "matched_iou_mean": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "unmatched_iou_mean": float(np.mean(unmatched_ious)) if unmatched_ious else 0.0,
        "tp_iou_mean": float(np.mean(tp_ious_total)) if tp_ious_total else 0.0,
        "num_matched_predictions": int(len(matched_ious)),
        "num_unmatched_predictions": int(len(unmatched_ious)),
        "num_true_positive_matches": int(len(tp_ious_total)),
    }
    summary_path = os.path.join(EVAL_OUT_DIR, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[EVAL] Summary JSON saved to: {summary_path}")

    print("[EVAL] Validation overlays/masks saved to:", EVAL_OUT_DIR)  # Summary path
    for c in sorted(list(ALLOWED_CLASSES)):  # Print AP per class
        print(f"  AP@0.75-vs-class-GT[{c}]: {ap_per_class.get(c, 0.0):.3f}")  # Report metric


def main():
    """
    Fine-tune SAM2 with GT box prompts (prompt encoder + mask decoder), save student checkpoint under
    ``checkpoints_sam2/``, then report semantic segmentation metrics from box-prompted inference.
    """
    _validate_runtime_prerequisites()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_source = "env" if (os.getenv("SAM2_MODEL_CFG") or os.getenv("MODEL_CFG")) else "auto/default"
    model_cfg_hydra = _to_hydra_config_name(MODEL_CFG)
    print(f"[INFO] SAM2 model cfg: {MODEL_CFG} (source: {cfg_source})")
    print(f"[INFO] SAM2 hydra cfg name: {model_cfg_hydra}")

    if not os.path.isfile(SAM2_BASE_CHECKPOINT):
        raise RuntimeError(
            f"SAM2 base checkpoint not found: {SAM2_BASE_CHECKPOINT}\n"
            "Place the matching Meta pretrained .pt for your YAML (e.g. sam2_hiera_large.pt) under "
            "checkpoints_sam2/, or set SAM2_CHECKPOINT / SAM2_BASE_CHECKPOINT to its path."
        )
    print(f"[INFO] SAM2 base checkpoint: {SAM2_BASE_CHECKPOINT}", flush=True)

    train_entries, val_entries, test_entries = build_dataset(
        IMAGES_DIR,
        ANNOT_JSON,
        train_split_txt=TRAIN_SPLIT_TXT,
        val_split_txt=VAL_SPLIT_TXT,
        test_split_txt=TEST_SPLIT_TXT,
    )
    if not train_entries:
        raise RuntimeError(
            "No valid training entries found. Check the configured image/annotation paths and ensure the training images exist."
        )
    print(
        f"[INFO] Train entries: {len(train_entries)} | Val entries: {len(val_entries)} | Test entries: {len(test_entries)}",
        flush=True,
    )

    sam2_model = build_sam2(model_cfg_hydra, str(SAM2_BASE_CHECKPOINT), device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    resume_from_latest = SAM2_RESUME_TRAINING and os.path.isfile(SAM2_TRAINING_LATEST)
    if resume_from_latest:
        print(f"[INFO] SAM2_RESUME set: will load mid-run state from {SAM2_TRAINING_LATEST} if valid.", flush=True)
    if not resume_from_latest:
        resume_path = None
        for candidate in (MODEL_STATE_PATH, MODEL_STATE_LEGACY):
            if candidate and os.path.isfile(candidate):
                resume_path = candidate
                break
        if resume_path is not None:
            try:
                _load_sam2_student_weights(predictor, resume_path, device)
                print(f"[INFO] Loaded student checkpoint for warm start / resume: {resume_path}", flush=True)
            except Exception as e:
                print("[WARN] Failed to load existing student checkpoint:", e, flush=True)

    sam2_set_trainable_parts(predictor, freeze_image_encoder=SAM2_FREEZE_IMAGE_ENCODER)
    trainable_params = [p for p in predictor.model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable SAM2 parameters were enabled; check sam2_set_trainable_parts.")

    optimizer = torch.optim.AdamW(
        params=trainable_params,
        lr=SAM2_TRAIN_LR,
        weight_decay=SAM2_TRAIN_WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    _ensure_dir(CURVES_OUT_DIR)
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    use_early_stop = SAM2_EARLY_STOPPING_PATIENCE > 0 and len(val_entries) > 0
    early_stopper: EarlyStopping | None = None
    if use_early_stop:
        early_stopper = EarlyStopping(
            patience=SAM2_EARLY_STOPPING_PATIENCE,
            min_delta=SAM2_EARLY_STOPPING_MIN_DELTA,
            mode="min",
        )
        print(
            f"[INFO] Early stopping: patience={SAM2_EARLY_STOPPING_PATIENCE} "
            f"min_delta={SAM2_EARLY_STOPPING_MIN_DELTA} (monitor=val_loss)",
            flush=True,
        )
    elif SAM2_EARLY_STOPPING_PATIENCE > 0 and not val_entries:
        print("[WARN] Early stopping requested but val split is empty; training full epoch budget.", flush=True)

    start_epoch = 0
    best_weights: Dict[str, torch.Tensor] | None = None
    if resume_from_latest:
        start_epoch = _try_resume_sam2_training_from_latest(
            SAM2_TRAINING_LATEST,
            predictor=predictor,
            optimizer=optimizer,
            scaler=scaler,
            early_stopper=early_stopper,
            history=history,
            model_cfg_hydra=model_cfg_hydra,
            device=device,
        )
        resumed_ok = start_epoch > 0 or any(history.get(k) for k in history)
        if not resumed_ok:
            print(
                "[WARN] SAM2_RESUME set but latest checkpoint was not applied; "
                "falling back to inference export warm start if present.",
                flush=True,
            )
            for candidate in (MODEL_STATE_PATH, MODEL_STATE_LEGACY):
                if candidate and os.path.isfile(candidate):
                    try:
                        _load_sam2_student_weights(predictor, candidate, device)
                        print(f"[INFO] Warm start from {candidate}", flush=True)
                    except Exception as e:
                        print("[WARN] Warm start failed:", e, flush=True)
                    break
        if resumed_ok:
            bw = _load_sam2_training_best_weights_cpu(SAM2_TRAINING_BEST)
            if bw is not None:
                best_weights = bw
                print(
                    "[INFO] Loaded best validation weights from "
                    f"{SAM2_TRAINING_BEST} for end-of-run restore.",
                    flush=True,
                )

    stopped_early = False
    epoch_done = start_epoch

    if start_epoch >= SAM2_TRAIN_EPOCHS:
        print(
            f"[INFO] Latest checkpoint epoch_next={start_epoch} >= max_epochs={SAM2_TRAIN_EPOCHS}; "
            "skipping training loop.",
            flush=True,
        )
    else:
        for epoch in range(start_epoch, SAM2_TRAIN_EPOCHS):
            t_epoch0 = time.perf_counter()
            print(
                f"[INFO] Epoch {epoch + 1}/{SAM2_TRAIN_EPOCHS}: training ({len(train_entries)} images)...",
                flush=True,
            )
            train_loss, train_acc = train_one_epoch_sam2(
                predictor,
                train_entries,
                optimizer,
                scaler,
                device,
                max_instances_per_image=SAM2_MAX_INSTANCES_PER_IMAGE,
                image_size=SAM2_PROMPT_IMAGE_SIZE,
            )
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            val_loss = float("nan")
            val_acc = float("nan")
            if val_entries:
                print(
                    f"[INFO] Epoch {epoch + 1}/{SAM2_TRAIN_EPOCHS}: validation ({len(val_entries)} images)...",
                    flush=True,
                )
                val_loss, val_acc = eval_one_epoch_sam2_val_loss(
                    predictor,
                    val_entries,
                    device,
                    max_instances_per_image=SAM2_MAX_INSTANCES_PER_IMAGE,
                    image_size=SAM2_PROMPT_IMAGE_SIZE,
                )
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            epoch_sec = time.perf_counter() - t_epoch0

            log_msg = f"Epoch {epoch + 1}/{SAM2_TRAIN_EPOCHS} | train_loss={train_loss:.4f}"
            if not math.isnan(train_acc):
                log_msg += f" | train_acc={train_acc:.4f}"
            if val_entries and not math.isnan(val_loss):
                log_msg += f" | val_loss={val_loss:.4f}"
            if val_entries and not math.isnan(val_acc):
                log_msg += f" | val_acc={val_acc:.4f}"

            stop_now = False
            improved = False
            can_early_stop = early_stopper is not None and not math.isnan(val_loss)
            if can_early_stop:
                improved, stop_now = early_stopper.step(val_loss)
                log_msg += f" | Patience: {early_stopper.counter}/{early_stopper.patience}"

            log_msg += f" | Time: {epoch_sec:.1f}s"

            print(log_msg, flush=True)

            epoch_done = epoch + 1

            if can_early_stop and improved:
                best_weights = _sam2_clone_state_dict_cpu(predictor.model)
                print(
                    f"[INFO] val_loss improved to {val_loss:.4f} (best); saving snapshot for restore.",
                    flush=True,
                )
                _save_sam2_training_best(
                    SAM2_TRAINING_BEST,
                    predictor=predictor,
                    epoch_1based=epoch_done,
                    val_loss=float(val_loss),
                    model_cfg_hydra=model_cfg_hydra,
                )
                print(f"[INFO] Wrote best val checkpoint: {SAM2_TRAINING_BEST}", flush=True)

            if can_early_stop and stop_now:
                stopped_early = True
                print(
                    f"[INFO] Early stopping: no val_loss improvement for {SAM2_EARLY_STOPPING_PATIENCE} epochs.",
                    flush=True,
                )

            _save_sam2_training_latest(
                SAM2_TRAINING_LATEST,
                predictor=predictor,
                optimizer=optimizer,
                scaler=scaler,
                epoch_next=epoch_done,
                max_epochs=SAM2_TRAIN_EPOCHS,
                early_stopper=early_stopper,
                history=history,
                model_cfg_hydra=model_cfg_hydra,
                stopped_early=stopped_early,
            )
            print(
                f"[INFO] Wrote latest training checkpoint: {SAM2_TRAINING_LATEST} (epoch_next={epoch_done})",
                flush=True,
            )

            if can_early_stop and stop_now:
                break

    if best_weights is not None:
        predictor.model.load_state_dict(best_weights)
        print("[INFO] Restored weights from best val_loss checkpoint.", flush=True)

    sam2_ckpt = {
        "model_name": "SAM2_Hiera_Finetuned_TreeCanopy",
        "base_config": str(MODEL_CFG),
        "base_checkpoint": str(SAM2_BASE_CHECKPOINT),
        "finetuned_state_dict": predictor.model.state_dict(),
        "train_epochs": epoch_done,
        "max_epochs": SAM2_TRAIN_EPOCHS,
        "image_size": SAM2_PROMPT_IMAGE_SIZE,
        "early_stopping": {
            "patience": SAM2_EARLY_STOPPING_PATIENCE,
            "min_delta": SAM2_EARLY_STOPPING_MIN_DELTA,
            "stopped_early": stopped_early,
            "best_val_loss": early_stopper.best_score if early_stopper else None,
        },
    }
    torch.save(sam2_ckpt, SAM2_FINETUNED_WEIGHTS)
    print(f"[DONE] Saved SAM2 finetuned checkpoint to: {SAM2_FINETUNED_WEIGHTS}", flush=True)

    try:
        import matplotlib.pyplot as plt

        if history["train_loss"]:
            plt.figure(figsize=(7, 4))
            ep = range(1, len(history["train_loss"]) + 1)
            plt.plot(ep, history["train_loss"], color="#1f77b4", label="train_loss")
            if history["val_loss"] and any(not math.isnan(v) for v in history["val_loss"]):
                plt.plot(ep, history["val_loss"], color="#ff7f0e", label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("SAM2 box-prompt fine-tuning")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(CURVES_OUT_DIR, "sam2_prompt_train_loss.png"))
            plt.close()
    except Exception as e:
        print("[WARN] Could not plot train loss curve:", e)

    predictor.model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()

    train_metrics = evaluate_sam2_prompted(train_entries, predictor, image_size=SAM2_PROMPT_IMAGE_SIZE)
    val_metrics = evaluate_sam2_prompted(val_entries, predictor, image_size=SAM2_PROMPT_IMAGE_SIZE)
    test_metrics = (
        evaluate_sam2_prompted(test_entries, predictor, image_size=SAM2_PROMPT_IMAGE_SIZE)
        if test_entries
        else None
    )

    if SAM2_GRID_EVAL and val_entries:
        try:
            evaluate_on_validation(val_entries, predictor, device=device)
        except Exception as e:
            print("[WARN] Optional grid-point validation failed:", e)

    tr_m = float(train_metrics.get("mean_iou", float("nan")))
    va_m = float(val_metrics.get("mean_iou", float("nan")))
    te_m = float(test_metrics.get("mean_iou", float("nan"))) if test_metrics else float("nan")

    training_results: Dict[str, Any] = {
        "method": "sam2",
        "train_accuracy": _optional_round_mean_iou(train_metrics),
        "val_accuracy": _optional_round_mean_iou(val_metrics),
        "test_accuracy": _optional_round_mean_iou(test_metrics) if test_metrics is not None else None,
        "metric_type": "mean_iou",
        "epochs": epoch_done,
        "max_epochs": SAM2_TRAIN_EPOCHS,
        "early_stopping": sam2_ckpt.get("early_stopping"),
        "prompted_eval_train": _metric_dict_for_json(train_metrics),
        "prompted_eval_val": _metric_dict_for_json(val_metrics),
        "history": history,
        "timestamp": int(time.time()),
    }
    if test_metrics is not None:
        training_results["prompted_eval_test"] = _metric_dict_for_json(test_metrics)

    _ensure_dir(EVAL_OUT_DIR)
    results_path = os.path.join(EVAL_OUT_DIR, "training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(training_results, f, indent=2)
    print(
        f"[SAM2] Prompted eval — train mean_iou={tr_m:.4f} val mean_iou={va_m:.4f}",
        flush=True,
    )
    if test_metrics is not None and not math.isnan(te_m):
        print(f"[SAM2] Prompted eval — test mean_iou={te_m:.4f}", flush=True)
    print(f"[SAM2] Wrote {results_path}", flush=True)


if __name__ == "__main__":  # Standard Python entrypoint check
    main()  # Execute main routine
