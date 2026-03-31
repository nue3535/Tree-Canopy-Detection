from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
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
except Exception as exc:  # pragma: no cover - runtime environment dependent
    raise RuntimeError("torchvision detection module is required for Mask R-CNN training.") from exc

try:
    from backend.scripts.training_utils import EarlyStopping, apply_augmentation
except ImportError:
    from training_utils import EarlyStopping, apply_augmentation


SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _resolve_image_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "raw" / "train_images",
        project_root / "data" / "raw" / "train_images_tif",
        project_root / "data" / "raw" / "train_images_png",
    ]
    first_existing = None
    for path in candidates:
        if path.exists() and path.is_dir():
            if first_existing is None:
                first_existing = path
            if any(p.suffix.lower() in SUPPORTED_SUFFIXES for p in path.iterdir() if p.is_file()):
                return path
    return first_existing or candidates[0]


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
    batch_size: int = 8
    img_size: int = 1024
    max_instances_per_image: int = 80
    epochs: int = 10
    lr: float = 1e-4
    workers: int = 0


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
        scale_x = float(self.img_size) / float(max(1, orig_w))
        scale_y = float(self.img_size) / float(max(1, orig_h))

        masks = []
        labels = []
        boxes = []
        areas = []

        for m, label in zip(inst_masks_resized, inst_labels):
            if int(m.sum()) < 10:
                continue
            ys, xs = np.where(m > 0)
            x_min = float(xs.min())
            y_min = float(ys.min())
            x_max = float(xs.max())
            y_max = float(ys.max())
            boxes.append([x_min, y_min, x_max, y_max])
            masks.append(m.astype(np.uint8))
            labels.append(label)
            areas.append(float((x_max - x_min) * (y_max - y_min) * scale_x * scale_y))

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


def _resolve_eval_image_dir(project_root: Path) -> Path:
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
            if any(p.suffix.lower() in SUPPORTED_SUFFIXES for p in path.iterdir() if p.is_file()):
                return path
    return first_existing or candidates[0]


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
    eval_annot = config.project_root / "data" / "raw" / "annotations" / "evaluation_annotations.json"
    if not eval_annot.exists():
        return None
    try:
        payload = json.loads(eval_annot.read_text(encoding="utf-8"))
        eval_image_dir = _resolve_eval_image_dir(config.project_root)
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
    print(
        f"[Mask R-CNN] device={device} train={len(train_records)} val={len(val_records)} epochs={config.epochs} "
        f"batch_size={config.batch_size} img_size={config.img_size} max_instances={config.max_instances_per_image}",
        flush=True,
    )
    loader = DataLoader(
        DetectionDataset(train_records, config.img_size, config.max_instances_per_image, training=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        collate_fn=_collate_fn,
    )

    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone="DEFAULT", num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=10)

    best_loss = float("inf")
    try:
        for epoch in range(config.epochs):
            model.train()
            epoch_losses = []
            pbar = tqdm(loader, desc=f"[Mask R-CNN][Train] epoch {epoch+1}/{config.epochs}", dynamic_ncols=True)
            for batch_idx, (images, targets) in enumerate(pbar, start=1):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                losses = model(images, targets)
                total_loss = sum(loss for loss in losses.values())
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
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            print(f"[Mask R-CNN] epoch={epoch+1}/{config.epochs} mean_loss={mean_loss:.4f}", flush=True)
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(
                    {"epoch": epoch, "mean_loss": mean_loss, "model_state_dict": model.state_dict()},
                    config.save_dir / "best_model.pth",
                )
                print(f"[Mask R-CNN] saved best checkpoint: {config.save_dir / 'best_model.pth'}", flush=True)
            if early_stopping.should_stop(-mean_loss):
                print(f"[Mask R-CNN] early stopping at epoch {epoch+1}", flush=True)
                break
    except Exception as exc:
        print(f"[Mask R-CNN][ERROR] {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise

    torch.save({"model_state_dict": model.state_dict()}, config.save_dir / "final_model.pth")
    print(f"[Mask R-CNN] complete. Best loss={best_loss:.4f}. Saved to {config.save_dir}", flush=True)

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
        "best_train_loss": round(best_loss, 6),
        "epochs": config.epochs,
        "timestamp": int(time.time()),
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
    parser = argparse.ArgumentParser(description="Mask R-CNN training workflow")
    parser.add_argument("action", choices=["train"], default="train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--max-instances-per-image", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = TrainConfig(
        project_root=project_root,
        annotations_path=project_root / "data" / "raw" / "annotations" / "train_annotations.json",
        image_dir=_resolve_image_dir(project_root),
        save_dir=project_root / "checkpoints_mask_rcnn",
        batch_size=args.batch_size,
        img_size=args.img_size,
        max_instances_per_image=args.max_instances_per_image,
        epochs=args.epochs,
        lr=args.lr,
        workers=args.workers,
    )
    if args.action == "train":
        train(config)


if __name__ == "__main__":
    main()
