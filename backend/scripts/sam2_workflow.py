"""
Train/Fine-Tune SAM 2 on a custom dataset using polygon annotations.

Reference note: batch_grounded_sam.py uses Hugging Face SAM ViT-Base + GroundingDINO (zero-shot),
not Meta SAM2. This workflow keeps the sam2 package + Hiera checkpoints; we align analogous
settings where sensible (e.g. SAM2_MIN_MASK_SCORE=0.3 like that script's detection threshold).

This script adapts TRAIN_multi_image_batch.py to a folder layout like:  # Context of adaptation
- Images:        ./data/raw/train_images_tif/* or ./data/raw/train_images_png/* or ./train_images/*  # Expected image directory
- Annotations:   ./data/raw/annotations/train_annotations.json or ./train_annotations.json  # Expected annotation file
- Optional masks: ./viz_masks, ./masks (not used for training here)  # Optional extras (not used)

We parse polygons for two classes: "individual_tree" and "group_of_trees",  # Target classes
convert them to binary masks on-the-fly, and train with multi-image batches.  # On-the-fly rasterization + batching

Notes:
- By default we train the prompt encoder + mask decoder. If you also want to train  # Training scope
  the image encoder, set TRAIN_IMAGE_ENCODER=True; however, upstream SAM2 code  # Caveat about upstream
  includes some no_grad paths — you may need to remove them to allow gradients.  # Warning on gradients
- We resize images so the long side <= 1024, then pad to 1024x1024 to have  # Preprocessing strategy
  consistent shapes in a batch (matching the example script’s approach).  # Reason for padding

Usage:
  python backend/scripts/sam2_workflow.py  # How to run

Requirements:
  - torch, numpy, opencv-python, tqdm, matplotlib  # Python dependencies
  - segment-anything-2 codebase/environment for `sam2` imports  # External SAM2 dependency
  - SAM2 checkpoint and config files (see constants below)  # Model assets

Workflow overview:  # High-level pipeline
  1) Load annotations and image paths, assemble train/val splits.  # Step 1
     - If `data/processed/train.txt` and `data/processed/val.txt` exist, use them.  # Preferred fixed split
  2) For each training step, sample a mini-batch:  # Step 2
     - Read images, resize/pad to fixed size.  # 2a
     - Select one instance polygon per image, rasterize to a binary mask.  # 2b
     - Sample one positive point prompt inside the mask.  # 2c
  3) Forward pass through SAM2 (prompt encoder + mask decoder). Compute:  # Step 3
     - Sigmoid BCE segmentation loss between predicted and GT mask.  # Loss 1
     - Score alignment loss to align predicted confidence with IoU.  # Loss 2
  4) Optimize, periodically save weights and log/plot IoU curves.  # Step 4
  5) After training, run a grid-point prompt sweep on the validation set:  # Step 5
     - Collect top-K unique, non-overlapping predictions.  # Postproc
     - Save color overlays & instance masks.  # Outputs
     - Compute AP@0.75 per class and IoU histograms (matched vs unmatched).  # Metrics
"""

import os  # OS utilities (paths, env vars, directories)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Allow duplicate OpenMP runtimes on Windows
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Limit OpenMP threads to reduce contention

from contextlib import nullcontext
import json  # JSON parsing for annotations
from pathlib import Path
import random  # Deterministic splitting and sampling
from typing import List, Tuple, Dict, Any  # Type hints for clarity

import numpy as np  # Numerical arrays and mask operations
import torch  # Tensors, autograd, CUDA, optimizers, AMP
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


IMAGES_DIR = _resolve_image_dir(
    os.path.join(PROJECT_DIR, "data", "raw", "train_images_tif"),
    os.path.join(PROJECT_DIR, "data", "raw", "train_images_png"),
    os.path.join(PROJECT_DIR, "train_images"),
)  # Directory containing training images
ANNOT_JSON = _resolve_first_path(
    os.path.join(PROJECT_DIR, "data", "raw", "annotations", "train_annotations.json"),
    os.path.join(PROJECT_DIR, "train_annotations.json"),
)  # Path to annotation JSON
TRAIN_SPLIT_TXT = os.path.join(PROJECT_DIR, "data", "processed", "train.txt")
VAL_SPLIT_TXT = os.path.join(PROJECT_DIR, "data", "processed", "val.txt")

ALLOWED_CLASSES = {"individual_tree", "group_of_trees"}  # Classes to include

SAM2_CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints_sam2")
os.makedirs(SAM2_CHECKPOINT_DIR, exist_ok=True)

SAM2_CHECKPOINT = _resolve_file_path([
    os.getenv("SAM2_CHECKPOINT", "").strip(),
    os.path.join(SAM2_CHECKPOINT_DIR, "sam2_hiera_small.pt"),
    os.path.join(PROJECT_DIR, "sam2_hiera_small.pt"),
    os.path.join(PROJECT_DIR, "checkpoints", "sam2_hiera_small.pt"),
    os.path.join(PROJECT_DIR, "backend", "checkpoints", "sam2_hiera_small.pt"),
])  # Pretrained checkpoint file
MODEL_CFG = _resolve_file_path([
    os.getenv("SAM2_MODEL_CFG", "").strip(),
    os.getenv("MODEL_CFG", "").strip(),
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

BATCH_SIZE = int(os.getenv("SAM2_BATCH_SIZE", "6"))  # Match Colab batch micro-size; HF Grounded-SAM script uses per-image calls
LR = float(os.getenv("SAM2_LR", "1e-4"))  # Learning rate for AdamW
WEIGHT_DECAY = 4e-5  # L2 weight decay strength
STEPS = int(os.getenv("SAM2_STEPS", "3000"))  # Total optimization steps (balanced default)
SAVE_EVERY = int(os.getenv("SAM2_SAVE_EVERY", "200"))  # Save model every N steps

MAX_SIDE = 1024  # Resize so long side <= this
PAD_SIZE = 1024  # Then pad canvas to this size (square)

TRAIN_IMAGE_ENCODER = False  # If True, also train image encoder (see notes above)

RANDOM_SEED = 42  # Seed for reproducibility
TRAIN_RATIO = 0.80  # Train split proportion (80/20)

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
MODEL_STATE_PATH = os.path.join(SAM2_CHECKPOINT_DIR, "model.torch")


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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Construct train/val lists of entries from COCO-like JSON.  # Docstring
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

    if train_split_txt and val_split_txt and os.path.exists(train_split_txt) and os.path.exists(val_split_txt):
        train_stems = set(_read_split_file(train_split_txt))
        val_stems = set(_read_split_file(val_split_txt))
        train_entries = [entry for entry in entries if entry["stem"] in train_stems]
        val_entries = [entry for entry in entries if entry["stem"] in val_stems]
    else:
        random.Random(RANDOM_SEED).shuffle(entries)  # Deterministic shuffle
        n_train = max(1, int(len(entries) * TRAIN_RATIO))  # Compute train count
        train_entries = entries[:n_train]  # Slice train split
        val_entries = entries[n_train:]  # Slice val split
    return train_entries, val_entries  # Return splits


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
    Training entry point: build data, build model, optional resume, train, then evaluate.  # Docstring
    """
    _validate_runtime_prerequisites()
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Select device
    cfg_source = "env" if (os.getenv("SAM2_MODEL_CFG") or os.getenv("MODEL_CFG")) else "auto/default"
    ckpt_source = "env" if os.getenv("SAM2_CHECKPOINT") else "auto/default"
    model_cfg_hydra = _to_hydra_config_name(MODEL_CFG)
    print(f"[INFO] SAM2 model cfg: {MODEL_CFG} (source: {cfg_source})")
    print(f"[INFO] SAM2 hydra cfg name: {model_cfg_hydra}")
    print(f"[INFO] SAM2 checkpoint: {SAM2_CHECKPOINT} (source: {ckpt_source})")
    ckpt_for_build = SAM2_CHECKPOINT if os.path.exists(SAM2_CHECKPOINT) else None
    if ckpt_for_build is None:
        print(
            "[WARN] SAM2 pretrained checkpoint not found; continuing with random initialization.",
            flush=True,
        )

    train_entries, val_entries = build_dataset(
        IMAGES_DIR,
        ANNOT_JSON,
        train_split_txt=TRAIN_SPLIT_TXT,
        val_split_txt=VAL_SPLIT_TXT,
    )  # Build splits
    if not train_entries:  # Guard empty dataset
        raise RuntimeError(
            "No valid training entries found. Check the configured image/annotation paths and ensure the training images exist."
        )  # Fail fast
    print(f"[INFO] Train entries: {len(train_entries)} | Val entries: {len(val_entries)}")  # Log sizes

    sam2_model = build_sam2(model_cfg_hydra, ckpt_for_build, device=device)  # Instantiate model from cfg+ckpt
    predictor = SAM2ImagePredictor(sam2_model)  # Wrap model with predictor API

    try:
        if os.path.exists(MODEL_STATE_PATH):  # If checkpoint exists
            state = torch.load(MODEL_STATE_PATH, map_location=device)  # Load state dict
            predictor.model.load_state_dict(state, strict=False)  # Warm start
            print(f"[INFO] Loaded existing checkpoint for warm start / resume: {MODEL_STATE_PATH}")  # Log resume
    except Exception as e:
        print("[WARN] Failed to load existing checkpoint:", e)  # Non-fatal warning

    predictor.model.sam_mask_decoder.train(True)  # Enable training for mask decoder
    predictor.model.sam_prompt_encoder.train(True)  # Enable training for prompt encoder
    if TRAIN_IMAGE_ENCODER:  # Optionally also train image encoder
        predictor.model.image_encoder.train(True)  # Set encoder to train
        print("[WARN] TRAIN_IMAGE_ENCODER=True. Ensure no_grad is removed in upstream where needed.")  # Reminder

    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # AdamW optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))  # AMP gradient scaler (CUDA only)

    _ensure_dir(CURVES_OUT_DIR)  # Create curve output directory
    train_iou_hist: List[float] = []  # EMA train IoU history
    val_iou_hist: List[float] = []  # Mini-val IoU history
    steps_hist: List[int] = []  # Step indices for plotting

    mean_iou = 0.0  # Initialize EMA IoU
    for itr in range(STEPS):  # Main training loop
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):  # Mixed precision on GPU
            images, masks, input_points, input_labels = read_batch(train_entries, batch_size=BATCH_SIZE)  # Sample batch
            if masks.shape[0] == 0:  # Safety against empty batch
                continue  # Skip iteration

            predictor.set_image_batch(images)  # Encode batch of images

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(  # Prepare prompts
                input_points, input_labels, box=None, mask_logits=None, normalize_coords=True  # Single positive point
            )
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(  # Encode prompts
                points=(unnorm_coords, labels), boxes=None, masks=None  # Only point prompts
            )

            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]  # Hi-res feats per batch el.
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(  # Decode masks
                image_embeddings=predictor._features["image_embed"],  # Image embeddings
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),  # Positional encodings
                sparse_prompt_embeddings=sparse_embeddings,  # Sparse (points) embeddings
                dense_prompt_embeddings=dense_embeddings,  # Dense prompt embeddings
                multimask_output=True,  # Multi outputs internally (we will select channel 0)
                repeat_image=False,  # Images already batched
                high_res_features=high_res_features,  # Hi-res skip features
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # Upsample to orig-hw (per item)

            gt_mask = torch.tensor(masks.astype(np.float32), device=device)  # Cast GT masks to float tensor
            prd_mask = torch.sigmoid(prd_masks[:, 0])  # Take first mask and apply sigmoid
            seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()  # BCE

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)  # Intersection count per sample
            denom = gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter + 1e-6  # Union count with epsilon
            iou = inter / denom  # IoU per sample
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()  # Align score to IoU
            loss = seg_loss + 0.05 * score_loss  # Total loss

        predictor.model.zero_grad()  # Clear grads
        scaler.scale(loss).backward()  # Backprop with scaling
        scaler.step(optimizer)  # Optimizer step
        scaler.update()  # Update scaler

        if itr % SAVE_EVERY == 0:  # Periodic checkpoint
            torch.save(predictor.model.state_dict(), MODEL_STATE_PATH)  # Save model
            print("[INFO] Saved model at step", itr)  # Log save

        mean_iou = 0.99 * mean_iou + 0.01 * float(iou.detach().mean().cpu().numpy())  # Update EMA IoU
        if itr % 10 == 0:  # Logging cadence
            print(f"step {itr:05d} | loss={float(loss):.4f} | IOU={mean_iou:.4f}")  # Console log
            steps_hist.append(itr)  # Track step
            train_iou_hist.append(mean_iou)  # Track train EMA IoU

            val_iou_value = None  # Placeholder for mini-val IoU
            if len(val_entries) > 0:  # If we have a val split
                try:
                    predictor.model.eval()  # Eval mode
                    with torch.no_grad():  # Disable grad
                        v_images, v_masks, v_points, v_labels = read_batch(val_entries, batch_size=min(BATCH_SIZE, len(val_entries)))  # Sample val
                        if v_masks.shape[0] > 0:  # If not empty
                            predictor.set_image_batch(v_images)  # Encode val images
                            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(  # Prep prompts
                                v_points, v_labels, box=None, mask_logits=None, normalize_coords=True  # Same as train
                            )
                            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(  # Prompt enc
                                points=(unnorm_coords, labels), boxes=None, masks=None  # Points only
                            )
                            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]  # Hi-res feats
                            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(  # Decode
                                image_embeddings=predictor._features["image_embed"],  # Embeddings
                                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),  # Positional enc
                                sparse_prompt_embeddings=sparse_embeddings,  # Sparse prompts
                                dense_prompt_embeddings=dense_embeddings,  # Dense prompts
                                multimask_output=True,  # Multi
                                repeat_image=False,  # Batched already
                                high_res_features=high_res_features,  # Hi-res
                            )
                            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # Postprocess
                            gt_mask_v = torch.tensor(v_masks.astype(np.float32), device=device)  # GT val masks
                            prd_mask_v = torch.sigmoid(prd_masks[:, 0])  # Pred val masks (sigmoid)
                            inter_v = (gt_mask_v * (prd_mask_v > 0.5)).sum(1).sum(1)  # Intersection
                            denom_v = gt_mask_v.sum(1).sum(1) + (prd_mask_v > 0.5).sum(1).sum(1) - inter_v + 1e-6  # Union
                            iou_v = inter_v / denom_v  # IoU
                            val_iou_value = float(iou_v.detach().mean().cpu().numpy())  # Mean val IoU
                except Exception:
                    val_iou_value = None  # Swallow errors in quick val
                finally:
                    predictor.model.sam_mask_decoder.train(True)  # Back to train mode decoder
                    predictor.model.sam_prompt_encoder.train(True)  # Back to train mode prompts
                    if TRAIN_IMAGE_ENCODER:  # If encoder training enabled
                        predictor.model.image_encoder.train(True)  # Back to train mode encoder

            if val_iou_value is not None:  # If val IoU computed
                val_iou_hist.append(val_iou_value)  # Record value
            else:
                if len(val_iou_hist) < len(train_iou_hist):  # Keep arrays aligned
                    val_iou_hist.append(np.nan)  # Insert NaN placeholder

            try:
                import matplotlib.pyplot as plt  # Plot curves
                plt.figure(figsize=(7, 4))  # New fig
                plt.plot(steps_hist, train_iou_hist, label="Train IoU (EMA)", color="#1f77b4")  # Train curve
                plt.plot(steps_hist, val_iou_hist, label="Val IoU (mini-batch)", color="#ff7f0e")  # Val curve
                plt.xlabel("Step")  # X label
                plt.ylabel("IoU")  # Y label
                plt.ylim(0.0, 1.0)  # Bounds
                plt.title("IoU vs Steps (Train vs Val)")  # Title
                plt.legend()  # Legend
                plt.tight_layout()  # Layout
                plt.savefig(os.path.join(CURVES_OUT_DIR, "iou_train_vs_val.png"))  # Save figure
                plt.close()  # Close fig
            except Exception:
                pass  # Ignore plotting issues

    torch.save(predictor.model.state_dict(), MODEL_STATE_PATH)  # Final checkpoint save
    print(f"[DONE] Training complete. Saved {MODEL_STATE_PATH}")  # Completion log

    try:
        predictor.model.eval()  # Eval mode before full validation
        if device == "cuda":  # If GPU
            torch.cuda.empty_cache()  # Free cached memory
        evaluate_on_validation(val_entries, predictor, device=device)  # Run evaluation
    except Exception as e:
        print("[WARN] Validation evaluation failed:", e)  # Non-fatal warning

    import time as _time
    final_train_iou = train_iou_hist[-1] if train_iou_hist else None
    final_val_iou = val_iou_hist[-1] if val_iou_hist else None
    if final_val_iou is not None and (final_val_iou != final_val_iou):
        final_val_iou = None
    training_results = {
        "method": "sam2",
        "train_accuracy": round(final_train_iou, 6) if final_train_iou is not None else None,
        "val_accuracy": round(final_val_iou, 6) if final_val_iou is not None else None,
        "test_accuracy": None,
        "metric_type": "iou",
        "steps": STEPS,
        "timestamp": int(_time.time()),
    }
    _ensure_dir(EVAL_OUT_DIR)
    results_path = os.path.join(EVAL_OUT_DIR, "training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(training_results, f, indent=2)
    print(
        f"[SAM2] Results: train_iou={final_train_iou:.4f if final_train_iou else 'N/A'}"
        f" val_iou={final_val_iou:.4f if final_val_iou else 'N/A'}",
        flush=True,
    )


if __name__ == "__main__":  # Standard Python entrypoint check
    main()  # Execute main routine
