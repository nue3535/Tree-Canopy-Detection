"""Mask R-CNN: joint **object detection** (boxes + class: individual_tree / group_of_trees) and **instance segmentation**
(mask head), aligned with tree_canopy_multimodel_training_evaluation_fixed.ipynb:
maskrcnn_resnet50_fpn(weights=DEFAULT) + replaced ROI heads for 3 classes; 512² inputs; batch 2; AdamW lr=1e-4, wd=1e-4;
default 500 epochs with optional validation early stopping (patience 50). On Windows, DataLoader workers default to 0 to avoid spawn MemoryErrors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(x, **kwargs):
        return x

try:
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
except Exception as exc:  # pragma: no cover - runtime environment dependent
    raise RuntimeError("torchvision detection module is required for Mask R-CNN training.") from exc

from backend.app.data_layout import (
    resolve_evaluation_annotations_path,
    resolve_evaluation_image_dir,
    resolve_train_annotations_path,
    resolve_train_image_dir,
)

try:
    from backend.scripts.training_utils import EarlyStopping, apply_augmentation
except ImportError:
    from training_utils import EarlyStopping, apply_augmentation


SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Default training / inference spatial size (multimodel notebook).
MASK_RCNN_INFERENCE_IMG_SIZE = 512


def _resolve_image_dir(project_root: Path) -> Path:
    return resolve_train_image_dir(project_root)


def _find_image_path(image_dir: Path, stem: str) -> Path | None:
    for suffix in SUPPORTED_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _read_split_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _polygon_binary_mask(width: int, height: int, segmentation: list[float]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(segmentation) < 6 or len(segmentation) % 2 != 0:
        return mask
    polygon = np.asarray(segmentation, dtype=np.float32).reshape(-1, 2)
    polygon = np.round(polygon).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


@dataclass
class TrainConfig:
    project_root: Path
    annotations_path: Path
    image_dir: Path
    save_dir: Path
    batch_size: int = 2
    img_size: int = 512
    max_instances_per_image: int = 80
    epochs: int = 500
    lr: float = 1e-4
    weight_decay: float = 1e-4
    workers: int = 0 if sys.platform == "win32" else 2
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4


def _build_mask_rcnn(num_classes: int = 3) -> torch.nn.Module:
    """maskrcnn_resnet50_fpn with ImageNet backbone, heads replaced for `num_classes` (incl. background)."""
    try:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    except Exception:
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


class DetectionDataset(Dataset):
    def __init__(self, records: list[dict], img_size: int, max_instances_per_image: int, training: bool = True):
        self.records = records
        self.img_size = img_size
        self.max_instances_per_image = max_instances_per_image
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = cv2.imread(str(record["image_path"]))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {record['image_path']}")
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instances = record["instances"]
        if len(instances) > self.max_instances_per_image:
            instances = sorted(instances, key=lambda inst: int(np.sum(inst["mask"])), reverse=True)[
                : self.max_instances_per_image
            ]

        inst_masks_resized = [
            cv2.resize(inst["mask"].astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            for inst in instances
        ]
        inst_labels = [int(inst["label"]) for inst in instances]

        if self.training:
            image, inst_masks_resized = apply_augmentation(image, inst_masks_resized)

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        h, w = image.shape[:2]
        masks = []
        labels = []
        boxes = []
        areas = []

        for m, label in zip(inst_masks_resized, inst_labels):
            if int(m.sum()) < 10:
                continue
            if m.shape[0] != h or m.shape[1] != w:
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(m > 0)
            x_min = float(xs.min())
            y_min = float(ys.min())
            x_max = float(xs.max())
            y_max = float(ys.max())
            # torchvision requires strictly positive width/height (no degenerate lines/points).
            if x_max <= x_min:
                x_max = x_min + 1.0
            if y_max <= y_min:
                y_max = y_min + 1.0
            x_min = max(0.0, min(x_min, float(w - 1)))
            y_min = max(0.0, min(y_min, float(h - 1)))
            x_max = max(x_min + 1.0, min(x_max, float(w)))
            y_max = max(y_min + 1.0, min(y_max, float(h)))
            boxes.append([x_min, y_min, x_max, y_max])
            masks.append(m.astype(np.uint8))
            labels.append(label)
            areas.append(float(np.count_nonzero(m)))

        if not boxes:
            h, w = image.shape[:2]
            boxes = [[0.0, 0.0, float(max(1, w - 1)), float(max(1, h - 1))]]
            masks = [np.zeros((h, w), dtype=np.uint8)]
            labels = [1]
            areas = [1.0]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks), dtype=torch.uint8),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return image_t, target


def _collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@torch.no_grad()
def _mean_loss_on_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Mean summed detection loss over ``loader`` (train mode; no backward)."""
    model.train()
    totals: list[float] = []
    for images, targets in loader:
        filtered_images: list[torch.Tensor] = []
        filtered_targets: list[dict] = []
        for img, tgt in zip(images, targets):
            boxes = tgt.get("boxes")
            if boxes is None or boxes.numel() == 0 or boxes.shape[0] == 0:
                continue
            filtered_images.append(img.to(device))
            filtered_targets.append({k: v.to(device) for k, v in tgt.items()})
        if not filtered_images:
            continue
        losses = model(filtered_images, filtered_targets)
        totals.append(float(sum(v.detach().cpu().item() for v in losses.values())))
    return float(np.mean(totals)) if totals else float("nan")


def _load_records(config: TrainConfig) -> tuple[list[dict], list[dict]]:
    payload = json.loads(config.annotations_path.read_text(encoding="utf-8"))
    train_split = _read_split_file(config.project_root / "data" / "processed" / "train.txt")
    val_split = _read_split_file(config.project_root / "data" / "processed" / "val.txt")
    train_records: list[dict] = []
    val_records: list[dict] = []

    for image_info in payload.get("images", []):
        stem = Path(image_info.get("file_name", "")).stem
        if not stem:
            continue
        if train_split and stem not in train_split:
            continue
        image_path = _find_image_path(config.image_dir, stem)
        if image_path is None:
            continue

        width = int(image_info.get("width", 0))
        height = int(image_info.get("height", 0))
        if width <= 0 or height <= 0:
            continue
        instances = []
        for ann in image_info.get("annotations", []):
            class_name = ann.get("class")
            label = 1 if class_name == "individual_tree" else 2 if class_name == "group_of_trees" else None
            if label is None:
                continue
            mask = _polygon_binary_mask(width, height, ann.get("segmentation", []))
            if int(mask.sum()) == 0:
                continue
            instances.append({"mask": mask, "label": label})
        if not instances:
            continue
        record = {"image_path": image_path, "instances": instances}
        if stem in val_split:
            val_records.append(record)
        elif stem in train_split:
            train_records.append(record)
        else:
            train_records.append(record)

    if not train_records:
        raise RuntimeError("No training records found for Mask R-CNN.")
    if not val_records:
        n_val = max(1, int(0.20 * len(train_records)))
        val_records = train_records[:n_val]
        train_records = train_records[n_val:]
    return train_records, val_records


def _predict_mask(model: torch.nn.Module, image_bgr: np.ndarray, config: TrainConfig, device: torch.device) -> np.ndarray:
    image_resized = cv2.resize(image_bgr, (config.img_size, config.img_size), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_t = torch.from_numpy(image_rgb).permute(2, 0, 1).float().to(device) / 255.0
    with torch.no_grad():
        output = model([image_t])[0]

    pred_small = np.zeros((config.img_size, config.img_size), dtype=np.uint8)
    masks = output.get("masks")
    labels = output.get("labels")
    scores = output.get("scores")
    if masks is not None and labels is not None and scores is not None and len(scores) > 0:
        masks_np = masks.detach().cpu().numpy()[:, 0]
        labels_np = labels.detach().cpu().numpy().astype(np.int32)
        scores_np = scores.detach().cpu().numpy()
        order = np.argsort(scores_np)[::-1]
        occupancy = np.zeros_like(pred_small, dtype=bool)
        for idx in order:
            score = float(scores_np[idx])
            if score < 0.5:
                continue
            mask = masks_np[idx] > 0.5
            if int(mask.sum()) < 10:
                continue
            overlap = np.logical_and(mask, occupancy).sum()
            if overlap > 0 and overlap / float(mask.sum()) > 0.15:
                continue
            mask = np.logical_and(mask, np.logical_not(occupancy))
            if int(mask.sum()) < 10:
                continue
            cls = int(labels_np[idx])
            cls = 1 if cls == 1 else 2 if cls == 2 else 1
            pred_small[mask] = cls
            occupancy = np.logical_or(occupancy, mask)

    pred = cv2.resize(
        pred_small.astype(np.uint8),
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return pred


def _compute_maskrcnn_accuracy(
    model: torch.nn.Module, records: list[dict], config: TrainConfig, device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for record in records:
        image_bgr = cv2.imread(str(record["image_path"]))
        if image_bgr is None:
            continue
        gt_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        for inst in record["instances"]:
            gt_mask[inst["mask"] > 0] = int(inst["label"])
        pred_mask = _predict_mask(model, image_bgr, config, device)
        correct += int(np.sum(pred_mask == gt_mask))
        total += int(gt_mask.size)
    return round(float(correct) / max(1, total), 6)


def _compute_test_accuracy_maskrcnn(
    model: torch.nn.Module, config: TrainConfig, device: torch.device,
) -> float | None:
    eval_annot = resolve_evaluation_annotations_path(config.project_root)
    if not eval_annot.exists():
        return None
    try:
        payload = json.loads(eval_annot.read_text(encoding="utf-8"))
        eval_image_dir = resolve_evaluation_image_dir(config.project_root)
        records: list[dict] = []
        for img_info in payload.get("images", []):
            stem = Path(img_info.get("file_name", "")).stem
            if not stem:
                continue
            img_path = None
            for suffix in SUPPORTED_SUFFIXES:
                candidate = eval_image_dir / f"{stem}{suffix}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue
            w, h = int(img_info.get("width", 0)), int(img_info.get("height", 0))
            if w <= 0 or h <= 0:
                continue
            instances = []
            for ann in img_info.get("annotations", []):
                cls_name = ann.get("class")
                label = 1 if cls_name == "individual_tree" else 2 if cls_name == "group_of_trees" else None
                if label is None:
                    continue
                mask = _polygon_binary_mask(w, h, ann.get("segmentation", []))
                if int(mask.sum()) == 0:
                    continue
                instances.append({"mask": mask, "label": label})
            if not instances:
                continue
            records.append({"image_path": img_path, "instances": instances})
        if not records:
            return None
        return _compute_maskrcnn_accuracy(model, records, config, device)
    except Exception:
        return None


def _save_evaluation_artifacts(
    model: torch.nn.Module,
    records: list[dict],
    config: TrainConfig,
    device: torch.device,
) -> None:
    eval_root = config.project_root / "output" / "evaluation" / "maskrcnn"
    masks_dir = eval_root / "masks"
    overlays_dir = eval_root / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    cm = np.zeros((3, 3), dtype=np.int64)
    image_ious: list[float] = []
    tp_like_ious: list[float] = []
    color_lut = np.array([[0, 0, 0], [0, 255, 0], [255, 255, 0]], dtype=np.uint8)

    model.eval()
    for idx, record in enumerate(records, start=1):
        image_path: Path = record["image_path"]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        gt_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        for instance in record["instances"]:
            gt_mask[instance["mask"] > 0] = int(instance["label"])

        pred_mask = _predict_mask(model, image_bgr, config, device)

        for cls in range(3):
            gt_idx = gt_mask == cls
            if not np.any(gt_idx):
                continue
            for pred_cls in range(3):
                cm[cls, pred_cls] += int(np.sum(pred_mask[gt_idx] == pred_cls))

        inter = np.logical_and(gt_mask > 0, pred_mask > 0).sum()
        union = np.logical_or(gt_mask > 0, pred_mask > 0).sum()
        iou_fg = float(inter) / float(union + 1e-6)
        image_ious.append(iou_fg)
        if np.any(gt_mask > 0):
            tp_like_ious.append(iou_fg)

        out_mask = (pred_mask * 127).astype(np.uint8)
        cv2.imwrite(str(masks_dir / f"{image_path.stem}_mask_{idx:04d}.png"), out_mask)

        color_mask = color_lut[np.clip(pred_mask, 0, 2)]
        overlay = (0.6 * cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) + 0.4 * color_mask).astype(np.uint8)
        cv2.imwrite(str(overlays_dir / f"{image_path.stem}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    import matplotlib.pyplot as plt

    if image_ious:
        plt.figure(figsize=(6, 4))
        plt.hist(image_ious, bins=20, range=(0, 1), color="#4C72B0")
        plt.xlabel("IoU (foreground)")
        plt.ylabel("Count")
        plt.title("IoU Histogram")
        plt.tight_layout()
        plt.savefig(eval_root / "iou_hist.png")
        plt.close()

    if tp_like_ious:
        plt.figure(figsize=(6, 4))
        plt.hist(tp_like_ious, bins=20, range=(0, 1), color="#55A868")
        plt.xlabel("IoU (foreground, GT-present)")
        plt.ylabel("Count")
        plt.title("IoU Histogram (TP-like)")
        plt.tight_layout()
        plt.savefig(eval_root / "iou_hist_tp_only.png")
        plt.close()

    per_class_iou = []
    class_labels = ["background", "tree", "group"]
    for cls in range(3):
        tp = int(cm[cls, cls])
        fp = int(cm[:, cls].sum() - tp)
        fn = int(cm[cls, :].sum() - tp)
        iou = float(tp) / float(tp + fp + fn + 1e-6)
        per_class_iou.append(iou)

    plt.figure(figsize=(6, 4))
    xs = np.arange(3)
    plt.bar(xs, per_class_iou, color="#C44E52")
    plt.xticks(xs, class_labels)
    plt.ylim(0, 1)
    plt.ylabel("IoU")
    plt.title("Per-class IoU (AP placeholder)")
    for i, val in enumerate(per_class_iou):
        plt.text(i, min(val + 0.02, 0.98), f"{val:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(eval_root / "ap_per_class.png")
    plt.close()

    summary = {
        "num_images": len(records),
        "avg_fg_iou": float(np.mean(image_ious)) if image_ious else 0.0,
        "confusion_matrix": cm.tolist(),
        "per_class_iou": {
            "background": per_class_iou[0],
            "tree": per_class_iou[1],
            "group": per_class_iou[2],
        },
        "output_dir": str(eval_root),
    }
    (eval_root / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Mask R-CNN] evaluation artifacts saved to {eval_root}", flush=True)


def train(config: TrainConfig) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    config.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records, val_records = _load_records(config)
    use_early_stop = config.early_stopping_patience > 0 and len(val_records) > 0
    early_stopper: EarlyStopping | None = None
    if use_early_stop:
        early_stopper = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode="min",
        )
    print(
        f"[Mask R-CNN] device={device} train={len(train_records)} val={len(val_records)} epochs={config.epochs} "
        f"batch_size={config.batch_size} img_size={config.img_size} max_instances={config.max_instances_per_image} "
        f"workers={config.workers} early_stop_patience={config.early_stopping_patience if use_early_stop else 0}",
        flush=True,
    )
    loader = DataLoader(
        DetectionDataset(train_records, config.img_size, config.max_instances_per_image, training=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        collate_fn=_collate_fn,
        persistent_workers=config.workers > 0,
    )
    val_loader = DataLoader(
        DetectionDataset(val_records, config.img_size, config.max_instances_per_image, training=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        collate_fn=_collate_fn,
        persistent_workers=config.workers > 0,
    )

    model = _build_mask_rcnn(num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val_saved: float | None = None
    min_train_loss_seen = float("inf")
    stopped_early = False
    epoch_done = 0
    try:
        for epoch in range(config.epochs):
            model.train()
            epoch_losses = []
            loss_totals: defaultdict[str, float] = defaultdict(float)
            loss_batches = 0
            pbar = tqdm(loader, desc=f"[Mask R-CNN][Train] epoch {epoch+1}/{config.epochs}", dynamic_ncols=True)
            for batch_idx, (images, targets) in enumerate(pbar, start=1):
                filtered_images: list[torch.Tensor] = []
                filtered_targets: list[dict] = []
                for img, tgt in zip(images, targets):
                    boxes = tgt.get("boxes")
                    if boxes is None or boxes.numel() == 0 or boxes.shape[0] == 0:
                        continue
                    filtered_images.append(img.to(device))
                    filtered_targets.append({k: v.to(device) for k, v in tgt.items()})
                if not filtered_images:
                    continue
                losses = model(filtered_images, filtered_targets)
                total_loss = sum(loss for loss in losses.values())
                for k, v in losses.items():
                    loss_totals[k] += float(v.detach().cpu().item())
                loss_batches += 1
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_losses.append(float(total_loss.detach().cpu().item()))
                running_loss = float(np.mean(epoch_losses))
                pbar.set_postfix(loss=f"{running_loss:.4f}")
                if batch_idx % 5 == 0 or batch_idx == len(loader):
                    print(
                        f"[Mask R-CNN][Train] epoch={epoch+1}/{config.epochs} "
                        f"batch={batch_idx}/{len(loader)} loss={running_loss:.4f}",
                        flush=True,
                    )
            mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            if epoch_losses:
                min_train_loss_seen = min(min_train_loss_seen, mean_train_loss)
            if loss_batches > 0:
                parts = [f"{k}={loss_totals[k] / loss_batches:.4f}" for k in sorted(loss_totals.keys())]
                print(
                    f"[Mask R-CNN] epoch={epoch+1}/{config.epochs} mean_train={mean_train_loss:.4f} | " + " ".join(parts),
                    flush=True,
                )
            else:
                print(
                    f"[Mask R-CNN] epoch={epoch+1}/{config.epochs} mean_train={mean_train_loss:.4f} (no batches)",
                    flush=True,
                )

            val_mean_loss = _mean_loss_on_loader(model, val_loader, device)
            if not math.isnan(val_mean_loss) and (
                best_val_saved is None
                or val_mean_loss < best_val_saved - config.early_stopping_min_delta
            ):
                best_val_saved = val_mean_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "val_loss": val_mean_loss,
                        "mean_train_loss": mean_train_loss,
                        "model_state_dict": model.state_dict(),
                    },
                    config.save_dir / "best_model.pth",
                )
                print(
                    f"[Mask R-CNN] saved best checkpoint (val_loss={val_mean_loss:.4f}): "
                    f"{config.save_dir / 'best_model.pth'}",
                    flush=True,
                )

            stop_now = False
            if early_stopper is not None and not math.isnan(val_mean_loss):
                _, stop_now = early_stopper.step(val_mean_loss)

            log_msg = (
                f"Epoch {epoch+1:02d}/{config.epochs} | train_loss={mean_train_loss:.4f}"
                + (f" | val_loss={val_mean_loss:.4f}" if not math.isnan(val_mean_loss) else " | val_loss=nan")
            )
            if early_stopper is not None and not math.isnan(val_mean_loss):
                log_msg += f" | Patience: {early_stopper.counter}/{early_stopper.patience}"
            print(log_msg, flush=True)

            epoch_done = epoch + 1
            if stop_now:
                stopped_early = True
                print(
                    f"[Mask R-CNN] early stopping: no val_loss improvement for {config.early_stopping_patience} epochs.",
                    flush=True,
                )
                break
    except Exception as exc:
        print(f"[Mask R-CNN][ERROR] {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise

    torch.save({"model_state_dict": model.state_dict()}, config.save_dir / "final_model.pth")
    if best_val_saved is not None:
        print(
            f"[Mask R-CNN] complete. best_val_loss={best_val_saved:.4f} epochs_run={epoch_done}. Saved to {config.save_dir}",
            flush=True,
        )
    else:
        print(
            f"[Mask R-CNN] complete. best_val_loss=(none) epochs_run={epoch_done}. Saved to {config.save_dir}",
            flush=True,
        )

    best_path = config.save_dir / "best_model.pth"
    if best_path.is_file():
        try:
            try:
                ckpt = torch.load(best_path, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(best_path, map_location=device)
            sd = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else ckpt
            if isinstance(sd, dict):
                model.load_state_dict(sd, strict=False)
                print("[Mask R-CNN] loaded best val checkpoint for accuracy + eval artifacts.", flush=True)
        except Exception as exc:
            print(f"[WARN] Could not load best checkpoint for final eval: {exc}", flush=True)

    print("[Mask R-CNN] Computing training results...", flush=True)
    train_acc = _compute_maskrcnn_accuracy(model, train_records, config, device)
    val_acc = _compute_maskrcnn_accuracy(model, val_records, config, device)
    test_acc = _compute_test_accuracy_maskrcnn(model, config, device)
    eval_root = config.project_root / "output" / "evaluation" / "maskrcnn"
    eval_root.mkdir(parents=True, exist_ok=True)
    training_results = {
        "method": "maskrcnn",
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "best_val_loss": None if best_val_saved is None else round(float(best_val_saved), 6),
        "best_train_loss": round(min_train_loss_seen, 6) if min_train_loss_seen != float("inf") else None,
        "epochs": epoch_done,
        "max_epochs": config.epochs,
        "stopped_early": stopped_early,
        "early_stopping": {
            "patience": config.early_stopping_patience if use_early_stop else 0,
            "min_delta": config.early_stopping_min_delta if use_early_stop else None,
            "monitor": "val_loss",
            "enabled": use_early_stop,
        },
        "timestamp": int(time.time()),
        "config_note": (
            "Mask R-CNN ResNet50-FPN, 512², AdamW. Default 500 epochs; early stopping on val_loss when patience>0. "
            "Windows: num_workers=0 recommended (avoids DataLoader spawn MemoryError)."
        ),
    }
    (eval_root / "training_results.json").write_text(
        json.dumps(training_results, indent=2), encoding="utf-8",
    )
    print(
        f"[Mask R-CNN] Results: train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        f" test_acc={test_acc if test_acc is not None else 'N/A'}",
        flush=True,
    )

    _save_evaluation_artifacts(model, val_records, config, device)


def parse_args() -> argparse.Namespace:
    _win = sys.platform == "win32"
    _default_workers = 0 if _win else 2
    parser = argparse.ArgumentParser(description="Mask R-CNN training workflow")
    parser.add_argument("action", choices=["train"], default="train")
    parser.add_argument("--epochs", type=int, default=_int_env("MASK_RCNN_EPOCHS", 500))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--max-instances-per-image", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--workers",
        type=int,
        default=_int_env("MASK_RCNN_DATALOADER_WORKERS", _default_workers),
        help="DataLoader workers (0 required on many Windows setups to avoid MemoryError in spawn workers).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=_int_env("MASK_RCNN_EARLY_STOPPING_PATIENCE", 50),
        help="Stop if val_loss does not improve for this many epochs; 0 disables.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=_float_env("MASK_RCNN_EARLY_STOPPING_MIN_DELTA", 1e-4),
        help="Minimum val_loss decrease to count as improvement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = TrainConfig(
        project_root=project_root,
        annotations_path=resolve_train_annotations_path(project_root),
        image_dir=_resolve_image_dir(project_root),
        save_dir=project_root / "checkpoints_mask_rcnn",
        batch_size=args.batch_size,
        img_size=args.img_size,
        max_instances_per_image=args.max_instances_per_image,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        workers=args.workers,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    if args.action == "train":
        train(config)


if __name__ == "__main__":
    main()
