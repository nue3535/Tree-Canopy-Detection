from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CLASS_IDS = [0, 1, 2]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _resolve_image_dir(project_root: Path, split: str) -> Path:
    if split == "train":
        candidates = [
            project_root / "data" / "raw" / "train_images",
            project_root / "data" / "raw" / "train_images_tif",
            project_root / "data" / "raw" / "train_images_png",
        ]
    else:
        candidates = [
            project_root / "data" / "raw" / "evaluation_images",
            project_root / "data" / "raw" / "evaluation_images_tif",
            project_root / "data" / "raw" / "evaluation_images_png",
        ]

    first_existing = None
    for path in candidates:
        if path.exists() and path.is_dir():
            if first_existing is None:
                first_existing = path
            count = len(
                [
                    p
                    for p in path.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
                ]
            )
            if count > 0:
                return path
    if first_existing is not None:
        return first_existing
    return candidates[0]


def _list_images(directory: Path) -> list[Path]:
    images: list[Path] = []
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        images.extend(directory.glob(f"*{suffix}"))
    return sorted(set(images))


def _build_gt_map_from_annotations(annotations_path: Path) -> dict[str, np.ndarray]:
    if not annotations_path.exists():
        return {}

    with open(annotations_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    gt_map: dict[str, np.ndarray] = {}
    for image_info in payload.get("images", []):
        width = int(image_info.get("width", 0))
        height = int(image_info.get("height", 0))
        if width <= 0 or height <= 0:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in image_info.get("annotations", []):
            class_name = ann.get("class")
            class_id = 1 if class_name == "individual_tree" else 2 if class_name == "group_of_trees" else None
            if class_id is None:
                continue
            seg = ann.get("segmentation", [])
            if len(seg) < 6 or len(seg) % 2 != 0:
                continue
            polygon = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
            polygon = np.round(polygon).astype(np.int32)
            cv2.fillPoly(mask, [polygon], color=class_id)
        stem = Path(image_info.get("file_name", "")).stem
        if stem:
            gt_map[stem] = mask
    return gt_map


def _compute_metrics_from_cm(cm: np.ndarray, total_time_ms: float, num_rows: int) -> dict[str, Any]:
    total = int(cm.sum())
    accuracy = _safe_div(np.trace(cm), total)
    per_class = []
    ious = []
    f1s = []
    for class_id in CLASS_IDS:
        tp = int(cm[class_id, class_id])
        fp = int(cm[:, class_id].sum() - tp)
        fn = int(cm[class_id, :].sum() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        iou = _safe_div(tp, tp + fp + fn)
        ious.append(iou)
        f1s.append(f1)
        per_class.append(
            {
                "class_id": class_id,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "iou": round(iou, 6),
            }
        )
    return {
        "accuracy": round(accuracy, 6),
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 6),
        "macro_iou": round(float(np.mean(ious)) if ious else 0.0, 6),
        "avg_inference_ms": round(_safe_div(total_time_ms, num_rows), 3),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class,
    }


class EvaluationService:
    def __init__(self, services: dict[str, Any]):
        self.project_root = Path(__file__).resolve().parents[2]
        self.services = services
        self.method_order = list(services.keys())
        self.output_root = self.project_root / "output" / "evaluation"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Any] | None = None
        self._cache_ts = 0.0
        self._cache_ttl_seconds = 300.0

    def _predict(self, method: str, image_bytes: bytes, filename: str):
        service = self.services.get(method)
        if service is None:
            raise ValueError(f"Unsupported method: {method}")
        return service._predict_components(image_bytes, filename)

    def _evaluate_with_gt(
        self,
        method: str,
        images: list[Path],
        gt_map: dict[str, np.ndarray],
        dataset_name: str,
    ) -> dict[str, Any]:
        cm = np.zeros((3, 3), dtype=np.int64)
        rows: list[dict[str, Any]] = []
        total_time_ms = 0.0

        for image_path in images:
            stem = image_path.stem
            gt = gt_map.get(stem)
            if gt is None:
                continue
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            started = time.perf_counter()
            _, pred_mask, _, inference_mode, fallback_reason = self._predict(method, image_bytes, image_path.name)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            total_time_ms += elapsed_ms

            if pred_mask.shape != gt.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            gt_flat = gt.reshape(-1)
            pred_flat = pred_mask.reshape(-1)
            for gt_id in CLASS_IDS:
                gt_idx = gt_flat == gt_id
                if not np.any(gt_idx):
                    continue
                for pred_id in CLASS_IDS:
                    cm[gt_id, pred_id] += int(np.sum(pred_flat[gt_idx] == pred_id))

            pixel_acc = _safe_div(np.sum(gt_flat == pred_flat), gt_flat.size)
            rows.append(
                {
                    "file_name": image_path.name,
                    "method": method,
                    "dataset": dataset_name,
                    "inference_mode": inference_mode,
                    "fallback_reason": fallback_reason,
                    "pixel_accuracy": round(pixel_acc, 6),
                }
            )

        metrics = _compute_metrics_from_cm(cm, total_time_ms, len(rows))
        mode_counts: dict[str, int] = {}
        for row in rows:
            mode = str(row.get("inference_mode", "unknown"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        fallback_count = int(sum(1 for row in rows if str(row.get("fallback_reason", "")).strip()))
        summary = {
            "method": method,
            "dataset": dataset_name,
            "num_images": len(rows),
            "inference_mode_counts": mode_counts,
            "fallback_image_count": fallback_count,
            **metrics,
        }
        return {"summary": summary, "rows": rows}

    def _evaluate_without_gt(self, method: str, images: list[Path]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        total_time_ms = 0.0
        tree_coverages = []
        group_coverages = []

        for image_path in images:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            started = time.perf_counter()
            _, pred_mask, _, inference_mode, fallback_reason = self._predict(method, image_bytes, image_path.name)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            total_time_ms += elapsed_ms

            total_pixels = max(1, pred_mask.size)
            tree_ratio = float(np.sum(pred_mask == 1)) / float(total_pixels)
            group_ratio = float(np.sum(pred_mask == 2)) / float(total_pixels)
            tree_coverages.append(tree_ratio)
            group_coverages.append(group_ratio)
            rows.append(
                {
                    "file_name": image_path.name,
                    "method": method,
                    "dataset": "evaluation",
                    "inference_mode": inference_mode,
                    "fallback_reason": fallback_reason,
                    "tree_ratio": round(tree_ratio, 6),
                    "group_ratio": round(group_ratio, 6),
                }
            )

        summary = {
            "method": method,
            "dataset": "evaluation",
            "num_images": len(rows),
            "metrics_applicable": False,
            "note": "Ground-truth labels are not available for evaluation images. Showing segmentation outcomes only.",
            "avg_tree_ratio": round(float(np.mean(tree_coverages)) if tree_coverages else 0.0, 6),
            "avg_group_ratio": round(float(np.mean(group_coverages)) if group_coverages else 0.0, 6),
            "avg_inference_ms": round(_safe_div(total_time_ms, len(rows)), 3),
            "inference_mode_counts": {},
            "fallback_image_count": int(sum(1 for row in rows if str(row.get("fallback_reason", "")).strip())),
        }
        for row in rows:
            mode = str(row.get("inference_mode", "unknown"))
            summary["inference_mode_counts"][mode] = summary["inference_mode_counts"].get(mode, 0) + 1
        return {"summary": summary, "rows": rows}

    def _save_method_results(self, method: str, dataset: str, result: dict[str, Any]) -> None:
        """Persist per-model evaluation outputs under output/evaluation/<method>/."""
        method_dir = self.output_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        out_path = method_dir / f"{dataset}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    def run_full_evaluation(self, force: bool = False) -> dict[str, Any]:
        now = time.time()
        if not force and self._cache and (now - self._cache_ts) < self._cache_ttl_seconds:
            return self._cache

        train_dir = _resolve_image_dir(self.project_root, "train")
        eval_dir = _resolve_image_dir(self.project_root, "evaluation")
        train_images = _list_images(train_dir)
        eval_images = _list_images(eval_dir)
        train_gt_path = self.project_root / "data" / "raw" / "annotations" / "train_annotations.json"
        eval_gt_path = self.project_root / "data" / "raw" / "annotations" / "evaluation_annotations.json"
        train_gt_map = _build_gt_map_from_annotations(train_gt_path)
        eval_gt_map = _build_gt_map_from_annotations(eval_gt_path)

        methods = self.method_order
        train_by_method: dict[str, Any] = {}
        eval_by_method: dict[str, Any] = {}

        for method in methods:
            train_by_method[method] = self._evaluate_with_gt(method, train_images, train_gt_map, "train")
            self._save_method_results(method, "train", train_by_method[method])
            if eval_gt_map:
                eval_by_method[method] = self._evaluate_with_gt(method, eval_images, eval_gt_map, "evaluation")
                eval_by_method[method]["summary"]["metrics_applicable"] = True
                eval_by_method[method]["summary"]["note"] = "Ground-truth labels found for evaluation images."
            else:
                eval_by_method[method] = self._evaluate_without_gt(method, eval_images)
            self._save_method_results(method, "evaluation", eval_by_method[method])

        report = {
            "generated_at": int(now),
            "train_image_dir": str(train_dir),
            "evaluation_image_dir": str(eval_dir),
            "train_annotations_path": str(train_gt_path),
            "evaluation_annotations_path": str(eval_gt_path),
            "evaluation_has_ground_truth": bool(eval_gt_map),
            "methods": methods,
            "train": train_by_method,
            "evaluation": eval_by_method,
        }
        self._cache = report
        self._cache_ts = now
        return report

    @staticmethod
    def paginate_rows(rows: list[dict[str, Any]], page: int, page_size: int) -> dict[str, Any]:
        total_items = len(rows)
        total_pages = max(1, int(np.ceil(total_items / float(max(1, page_size)))))
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        end = start + page_size
        return {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "items": rows[start:end],
        }
