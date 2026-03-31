from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Use a local writable matplotlib cache for API processes.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".").resolve() / ".mplconfig"))

from backend.app.evaluation import EvaluationService
from backend.app.inference import (
    DeepLabSegmentationService,
    MaskRCNNSegmentationService,
    SAM2SegmentationService,
    SegFormerSegmentationService,
    UNetSegmentationService,
    get_method_precheck,
)
from backend.app.training import TrainingManager

app = FastAPI(title="Tree Canopy Segmentation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = DeepLabSegmentationService()
sam2_service = SAM2SegmentationService()
unet_service = UNetSegmentationService()
maskrcnn_service = MaskRCNNSegmentationService()
segformer_service = SegFormerSegmentationService()
SEGMENTATION_SERVICES = {
    "deeplabv3plus": service,
    "sam2": sam2_service,
    "unet": unet_service,
    "maskrcnn": maskrcnn_service,
    "segformer": segformer_service,
}
TRAINABLE_METHODS = set(SEGMENTATION_SERVICES.keys())
evaluation_service = EvaluationService(SEGMENTATION_SERVICES)
training_manager = TrainingManager()


class HealthResponse(BaseModel):
    status: str


class SegmentResponse(BaseModel):
    method: str
    scene_class: int
    scene_label: str
    class_distribution: dict[str, float]
    suitability_assessment: dict
    strict_conservation_mode: bool
    class_labels: dict[str, str]
    class_colors: dict[str, str]
    mask_png_base64: str
    overlay_png_base64: str
    inference_mode: str
    fallback_reason: str


class EvaluationSummaryResponse(BaseModel):
    generated_at: int
    train_image_dir: str
    evaluation_image_dir: str
    train_annotations_path: str
    evaluation_annotations_path: str
    evaluation_has_ground_truth: bool
    methods: list[str]
    train: dict
    evaluation: dict


class EvaluationPageResponse(BaseModel):
    dataset: str
    method: str
    page: int
    page_size: int
    total_items: int
    total_pages: int
    items: list[dict]


class TrainingStartResponse(BaseModel):
    method: str
    profile: str
    status: str
    started_at: int
    pid: int
    command: list[str]
    log_path: str


class TrainingStatusResponse(BaseModel):
    method: str
    profile: str | None = None
    status: str
    started_at: int
    pid: int | None
    return_code: int | None
    command: list[str]
    log_path: str
    log_tail: str


class TrainingStopResponse(BaseModel):
    method: str
    status: str
    stopped: bool
    message: str


class TrainingResultsResponse(BaseModel):
    generated_at: int
    results: dict


class EvaluationPrecheckResponse(BaseModel):
    generated_at: int
    methods: dict
    any_fallback_risk: bool


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/api/segment", response_model=SegmentResponse)
async def segment_image(
    file: UploadFile = File(...),
    method: str = Form("deeplabv3plus"),
    strict_conservation_mode: bool = Form(False),
) -> SegmentResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        method_key = method.strip().lower()
        selected_service = SEGMENTATION_SERVICES.get(method_key)
        if selected_service is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid method. Supported values: deeplabv3plus, sam2, unet, maskrcnn, segformer.",
            )
        result = selected_service.segment_bytes(
            payload,
            file.filename or "upload.png",
            strict_conservation_mode=strict_conservation_mode,
        )
        result["method"] = method_key
        return SegmentResponse(**result)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}") from exc


@app.get("/api/evaluation/summary", response_model=EvaluationSummaryResponse)
def evaluation_summary(force: bool = False) -> EvaluationSummaryResponse:
    try:
        report = evaluation_service.run_full_evaluation(force=force)
        return EvaluationSummaryResponse(**report)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc


@app.get("/api/evaluation/precheck", response_model=EvaluationPrecheckResponse)
def evaluation_precheck() -> EvaluationPrecheckResponse:
    try:
        payload = get_method_precheck()
        any_risk = any(not bool(v.get("ready_for_model_inference")) for v in payload.values())
        return EvaluationPrecheckResponse(
            generated_at=int(time.time()),
            methods=payload,
            any_fallback_risk=any_risk,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Precheck failed: {exc}") from exc


@app.get("/api/evaluation/training-results", response_model=TrainingResultsResponse)
def training_results() -> TrainingResultsResponse:
    import json

    project_root = Path(__file__).resolve().parents[2]
    eval_root = project_root / "output" / "evaluation"
    results: dict = {}
    for method in SEGMENTATION_SERVICES:
        results_path = eval_root / method / "training_results.json"
        if results_path.exists():
            try:
                payload = json.loads(results_path.read_text(encoding="utf-8"))
                results[method] = payload
            except Exception:
                results[method] = None
        else:
            results[method] = None
    return TrainingResultsResponse(generated_at=int(time.time()), results=results)


@app.get("/api/evaluation/page", response_model=EvaluationPageResponse)
def evaluation_page(
    dataset: str,
    method: str,
    page: int = 1,
    page_size: int = 10,
    force: bool = False,
) -> EvaluationPageResponse:
    dataset_key = dataset.strip().lower()
    method_key = method.strip().lower()
    valid_methods = set(SEGMENTATION_SERVICES.keys()) | {"all"}
    if dataset_key not in {"train", "evaluation"}:
        raise HTTPException(status_code=400, detail="dataset must be 'train' or 'evaluation'.")
    if method_key not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail="method must be one of: deeplabv3plus, sam2, unet, maskrcnn, segformer, all.",
        )
    page_size = max(1, min(page_size, 50))

    try:
        report = evaluation_service.run_full_evaluation(force=force)
        if method_key == "all":
            rows = []
            for each_method in report["methods"]:
                rows.extend(report[dataset_key][each_method]["rows"])
        else:
            rows = report[dataset_key][method_key]["rows"]
        paged = evaluation_service.paginate_rows(rows, page=page, page_size=page_size)
        return EvaluationPageResponse(dataset=dataset_key, method=method_key, **paged)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation pagination failed: {exc}") from exc


@app.post("/api/evaluation/train", response_model=TrainingStartResponse)
def start_training(method: str = Form(...), profile: str = Form("best-quality")) -> TrainingStartResponse:
    method_key = method.strip().lower()
    profile_key = profile.strip().lower()
    if method_key not in TRAINABLE_METHODS:
        raise HTTPException(
            status_code=400,
            detail="method must be one of: deeplabv3plus, sam2, unet, maskrcnn, segformer.",
        )
    if profile_key not in TrainingManager.SUPPORTED_PROFILES:
        raise HTTPException(
            status_code=400,
            detail="profile must be best-quality.",
        )
    try:
        state = training_manager.start_training(method_key, profile=profile_key)
        return TrainingStartResponse(
            method=state.method,
            profile=state.profile,
            status=state.status,
            started_at=int(state.started_at),
            pid=state.pid,
            command=state.command,
            log_path=str(state.log_path),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training start failed: {exc}") from exc


@app.get("/api/evaluation/train-status", response_model=TrainingStatusResponse)
def training_status(method: str) -> TrainingStatusResponse:
    method_key = method.strip().lower()
    if method_key not in TRAINABLE_METHODS:
        raise HTTPException(
            status_code=400,
            detail="method must be one of: deeplabv3plus, sam2, unet, maskrcnn, segformer.",
        )
    try:
        payload = training_manager.get_status(method_key)
        return TrainingStatusResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training status failed: {exc}") from exc


@app.post("/api/evaluation/train-stop", response_model=TrainingStopResponse)
def stop_training(method: str = Form(...)) -> TrainingStopResponse:
    method_key = method.strip().lower()
    if method_key not in TRAINABLE_METHODS:
        raise HTTPException(
            status_code=400,
            detail="method must be one of: deeplabv3plus, sam2, unet, maskrcnn, segformer.",
        )
    try:
        payload = training_manager.stop_training(method_key)
        return TrainingStopResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training stop failed: {exc}") from exc
