# Assignment Alignment Report

## 1) Task Type Alignment

This project is a Computer Vision system focused on semantic/instance-like segmentation of tree canopy from aerial imagery.

- Primary CV task: image segmentation
- Implemented methods:
  - `DeepLabV3+` pipeline (`backend/scripts/deeplab_v3plus.py`)
  - `SAM2` pipeline (`backend/scripts/sam2_workflow.py`)
- Operational interface:
  - React frontend (`frontend/`)
  - FastAPI backend (`backend/app/`)

## 2) Problem Statement

Tree canopy extraction from aerial imagery supports practical use cases such as plantation monitoring, vegetation auditing, and landscape management. The project aims to automate canopy segmentation with deep learning, reducing manual labeling effort and enabling repeatable large-scale analysis.

## 3) Dataset and Scope

- Dataset source: Solafune Tree Canopy competition dataset
- Key assets:
  - training/evaluation images
  - polygon annotations (`train_annotations.json`)
  - fixed split files (`train.txt`, `val.txt`)

### Dataset Justification

The dataset is domain-specific, sufficiently large for coursework experimentation, and includes heterogeneous scenes and canopy categories, making it appropriate for benchmarking segmentation methods.

> If your subject coordinator explicitly requires a self-captured dataset, add a short appendix clarifying the approved rationale for using an external benchmark dataset.

## 4) Algorithm Implementation and Comparison

- Method A: DeepLabV3+ (multi-class segmentation + scene classification workflow)
- Method B: SAM2 (prompted segmentation workflow)
- Comparison target:
  - quantitative metrics (IoU, Dice, AP/mAP variants where available)
  - qualitative GUI output comparison
  - operational suitability (speed, dependency complexity, robustness)

See `docs/results_comparison.md` for the final comparison table.

## 5) GUI Requirement Alignment

The project includes a web GUI that allows users to:

- upload an image,
- choose method (`DeepLabV3+` or `SAM2`) from a dropdown,
- receive mask/overlay output and metadata.

This satisfies the requirement of developing an operational GUI interface.

## 6) Required Deliverables Mapping

- Code repository: complete (backend + frontend + scripts)
- Notebook/script evidence: complete (scripts retained under `backend/scripts/`)
- Comparison evidence: prepared template; fill with final experiment results
- Oral/presentation/video evidence: to be completed outside codebase

## 7) Remaining Actions Before Submission

1. Train or load final checkpoints for DeepLabV3+ and SAM2.
2. Run final evaluations and fill `docs/results_comparison.md`.
3. Capture GUI screenshots and place them in `docs/screenshots/`.
4. Rehearse oral presentation using evidence checklist in `docs/evidence_checklist.md`.
