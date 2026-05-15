# Tree Canopy Segmentation System

Full-stack tree-canopy tooling: **FastAPI** backend, **React + Vite** frontend, and **Python training workflows** for semantic segmentation, detection + instance masks, and box-prompted SAM2 fine-tuning.

| Area | Location |
|------|----------|
| HTTP API | `backend/app/main.py` |
| Inference services | `backend/app/inference.py` |
| Evaluation orchestration | `backend/app/evaluation.py` |
| Training subprocess launcher | `backend/app/training.py` |
| Workflows (CLI / `-m`) | `backend/scripts/deeplab_v3plus_workflow.py`, `backend/scripts/mask_rcnn_workflow.py`, `backend/scripts/sam2_workflow.py` |

## Problem Statement

Accurate tree canopy delineation from aerial imagery supports vegetation monitoring, plantation management, and planning. This project trains and serves deep models that predict canopy-related masks and classes, with a web UI for uploads and metrics.

## Project Objectives

- Build a computer vision pipeline for tree-canopy segmentation and related tasks.
- Implement and compare methods (**DeepLabV3+**, **Mask R-CNN** for detection + instance masks, **SAM2** for GT box–prompted segmentation).
- Evaluate methods with quantitative metrics and paginated visual review.
- Operationalize inference through a web app and REST API.

## Assignment Evidence Pack

- `docs/assignment_alignment_report.md`
- `docs/results_comparison.md`
- `docs/evidence_checklist.md`
- `docs/screenshots/` (GUI/output screenshots for reports)

## Data Layout

Paths are resolved in `backend/app/data_layout.py`. In short:

- **Train images:** `data/train_images`, or under `data/raw/` (`train_images_tif`, `train_images_png`, or legacy `train_images`).
- **Evaluation images:** `data/evaluation_images`, or `data/raw/evaluation_images_tif` / `evaluation_images_png`.
- **Annotations:** COCO-style JSON; training file is the first match among `data/train_annotations_updated_*.json`, `data/train_annotations.json`, or `data/raw/annotations/train_annotations.json`. Evaluation: `data/evaluation_annotations.json` (or raw legacy path).
- **Splits:** `data/processed/train.txt` and `data/processed/val.txt` are expected for several workflows.

GeoTIFF (`.tif` / `.tiff`) and raster-friendly formats are supported alongside PNG/JPEG.

## Segmentation Methods (API)

`POST /api/segment` accepts multipart **image** upload and a **method** form field:

- **`sam2`** (default): SAM2 student weights with **ground-truth style box prompts** at inference (segmentation-focused path; requires fine-tuned checkpoint under `checkpoints_sam2/`).
- **`maskrcnn`**: Mask R-CNN for **detection + instance segmentation** (two tree-related classes plus background); loads weights from `checkpoints_mask_rcnn/` when present.

Other form fields include `strict_conservation_mode` and, for SAM2, `segment_sensitivity`. If dependencies or weights are missing, services can fall back with a reason string in the JSON response.

## Backend API (summary)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | Liveness |
| `POST` | `/api/segment` | Run segmentation (`method`, image file, options) |
| `GET` | `/api/evaluation/summary` | Full train/eval report (`force` optional) |
| `GET` | `/api/evaluation/precheck` | Per-method readiness (checkpoint, deps) |
| `GET` | `/api/evaluation/page` | Paginated per-image rows (`dataset`, `method`, `page`, …) |
| `GET` | `/api/evaluation/training-results` | Latest `training_results.json` per trainable method |
| `POST` | `/api/evaluation/train` | Start training (`method`, `profile`; trainable: `sam2`, `maskrcnn`) |
| `GET` | `/api/evaluation/train-status` | Status + log tail for a method |
| `POST` | `/api/evaluation/train-stop` | Stop a running training job |

Training logs are written under `local_outputs/training_logs/`.

## Setup

### 1) Python environment

Create a venv and install dependencies (from the repository root):

```bash
python -m venv .venv
```

Activate it (Unix: `source .venv/bin/activate`; Windows PowerShell: `.\.venv\Scripts\Activate.ps1`), then:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

**PyTorch and GPU:** PyPI often installs a **CPU-only** `torch`. For NVIDIA GPUs, install CUDA builds *before* or *instead of* the generic lines, for example:

- Python 3.14 on Windows: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
- Many 3.10–3.12 setups: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`

See comments at the top of `requirements.txt`.

**SAM2:** The repo depends on Meta’s package (import name `sam2`), installed from Git in `requirements.txt`. If your IDE sets `PYTHONNOUSERSITE=1`, training subprocesses strip that so user-site installs remain visible to the child process (see `backend/app/training.py`).

### 2) Frontend

```bash
cd frontend
npm install
```

### 3) Preflight

```bash
python backend/check_env.py
```

This checks Python imports (including `sam2`), core data files, image folders, SAM2 YAML / checkpoint hints, DeepLab checkpoint presence, and Node/npm for the frontend.

## Run the Web System

### Backend (port 8000)

```bash
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (port 5173)

```bash
cd frontend
npm run dev
```

By default the UI targets `http://localhost:8000`. Override with:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

## Training (workflows)

**From the UI:** Evaluation view → start training for **SAM2** or **Mask R-CNN** (`profile` is currently `best-quality`; hyperparameters are driven by workflow defaults and environment variables, not by the profile string).

**From the CLI** (examples):

```bash
python -m backend.scripts.mask_rcnn_workflow train
python -m backend.scripts.sam2_workflow
python -m backend.scripts.deeplab_v3plus_workflow train
```

DeepLab actions: `train`, `evaluate`, `predict` (see argparse in `deeplab_v3plus_workflow.py`).

### Defaults and useful environment variables

When training is started via the API, `training.py` injects conservative defaults if unset:

| Method | Variable | API default (if unset in environment) |
|--------|----------|----------------------------------------|
| Mask R-CNN | `MASK_RCNN_EPOCHS` | `500` |
| Mask R-CNN | `MASK_RCNN_EARLY_STOPPING_PATIENCE` | `50` (`0` disables early stopping) |
| Mask R-CNN | `MASK_RCNN_DATALOADER_WORKERS` | `0` on Windows (avoids multiprocessing `MemoryError` in subprocesses) |
| SAM2 | `SAM2_EARLY_STOPPING_PATIENCE` | `50` (`0` disables) |

SAM2 training behavior is extensively configurable in `backend/scripts/sam2_workflow.py` (epochs `SAM2_TRAIN_EPOCHS`, LR `SAM2_TRAIN_LR`, box chunking for post-train eval `SAM2_PROMPTED_EVAL_BOX_CHUNK`, base weights `SAM2_CHECKPOINT` / `SAM2_BASE_CHECKPOINT`, YAML `SAM2_MODEL_CFG`, etc.). Mask R-CNN CLI flags mirror `MASK_RCNN_*` env names; see `mask_rcnn_workflow.py` `--help`.

## Model Artifacts

- **DeepLabV3+:** `checkpoints_deeplabv3plus/` — e.g. `deeplabv3plus_checkpoint.pth`, `final_model.pth`, or fold `best_model_fold*.pth`.
- **Mask R-CNN:** `checkpoints_mask_rcnn/` — training writes `best_model.pth` and `final_model.pth`. Inference tries, in order: `maskrcnn_checkpoint.pth`, `best_model.pth`, `final_model.pth`.
- **SAM2:** Under `checkpoints_sam2/`:
  - **Config:** Hydra YAML (e.g. `sam2_hiera_l.yaml` / `sam2_hiera_s.yaml`), searched in project dirs, `backend/configs/`, or the installed `sam2` package. Override with `SAM2_MODEL_CFG`.
  - **Base Meta weights:** e.g. `sam2_hiera_large.pt` / `sam2_hiera_l.pt` matching the YAML; training can download on first run when network is available. Override with `SAM2_CHECKPOINT` or `SAM2_BASE_CHECKPOINT`.
  - **Fine-tuned student:** `sam2_finetuned_tree_canopy.pt` (export), or `sam2_training_best.pt` / `sam2_training_latest.pt` / `model.torch`. Override with `SAM2_FINETUNED_CKPT`.

After training, Mask R-CNN and SAM2 write summaries and artifacts under `output/evaluation/<method>/` (including `training_results.json` where applicable).

## Troubleshooting

- **`check_env.py` reports missing `sam2`:** reinstall from `requirements.txt` inside the same venv you use for `uvicorn`.
- **CUDA not used:** reinstall `torch`/`torchvision` from the PyTorch CUDA wheel index for your platform (see `requirements.txt`).
- **Mask R-CNN training MemoryError on Windows:** keep `MASK_RCNN_DATALOADER_WORKERS=0` (default from API).
- **SAM2 OOM after epoch / during eval:** lower `SAM2_PROMPTED_EVAL_BOX_CHUNK` (default `1`) or reduce image/prompt size via env vars documented in `sam2_workflow.py`.
