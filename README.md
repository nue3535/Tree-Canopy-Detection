# Tree Canopy Segmentation System

This project is now a combined full-stack system:

- `backend/` (Python + FastAPI) for model inference APIs
- `frontend/` (React + Vite) for image upload and visualization
- `backend/scripts/deeplab_v3plus.py` and `backend/scripts/sam2_workflow.py` for model training/evaluation workflows

## Problem Statement

Accurate tree canopy delineation from aerial imagery is required for vegetation monitoring, plantation management, and urban/rural planning. Manual annotation is expensive and slow, so this project develops a deep-learning segmentation system that predicts tree canopy masks from input images and operationalizes inference through a web interface.

## Project Objectives

- Build a Computer Vision solution for tree-canopy segmentation.
- Implement and compare two algorithms (`DeepLabV3+` and `SAM2`).
- Evaluate each method using quantitative metrics.
- Deploy a GUI (web app) to operationalize model inference.
- Propose improvement directions based on observed performance.

## Assignment Evidence Pack

Use the following documents for report/presentation alignment:

- `docs/assignment_alignment_report.md`
- `docs/results_comparison.md`
- `docs/evidence_checklist.md`
- `docs/screenshots/` (place GUI/output screenshots here)

## Data Layout

Use the layout documented in `data/README.md`.

## Backend API

Main API entrypoint:

- `backend/app/main.py`

Key endpoint:

- `POST /api/segment` (multipart image upload, returns scene + mask + overlay)

Health endpoint:

- `GET /api/health`

## Setup

### 1) Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2) Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 3) Prepare masks (for DeepLab workflows)

```bash
.venv/bin/python backend/scripts/deeplab_v3plus.py prepare-data
```

### 4) Run preflight checks

```bash
.venv/bin/python backend/check_env.py
```

## Run the Web System

### Start backend (port 8000)

```bash
.venv/bin/uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start frontend (port 5173)

```bash
cd frontend
npm run dev
```

Frontend calls `http://localhost:8000` by default.
Override with:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

## Model Artifacts Notes

- DeepLab API inference requires trained DeepLab checkpoints under `checkpoints_deeplabv3plus/`.
- SAM2 script additionally requires:
  - `sam2_hiera_s.yaml`
  - `sam2_hiera_small.pt`

`backend/check_env.py` reports missing artifacts before runtime.
