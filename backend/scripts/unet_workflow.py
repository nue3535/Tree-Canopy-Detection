"""U-Net aligned with tree_canopy_multimodel_training_evaluation_fixed.ipynb:
smp.Unet(resnet34, imagenet), 512, batch 2, 50 epochs, AdamW 1e-4 / wd 1e-4, workers 2, DiceCELoss, multimodel aug."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import segmentation_models_pytorch as smp
except Exception as exc:  # pragma: no cover - runtime environment dependent
    raise RuntimeError("segmentation_models_pytorch is required for U-Net training.") from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(x, **kwargs):
        return x

from backend.app.data_layout import (
    resolve_evaluation_annotations_path,
    resolve_evaluation_image_dir,
    resolve_train_annotations_path,
    resolve_train_image_dir,
)

try:
    from backend.scripts.training_utils import (
        DiceCELoss,
        EarlyStopping,
        apply_augmentation,
        make_multimodel_nb_train_transform,
        make_multimodel_nb_val_transform,
    )
except ImportError:
    from training_utils import (
        DiceCELoss,
        EarlyStopping,
        apply_augmentation,
        make_multimodel_nb_train_transform,
        make_multimodel_nb_val_transform,
    )

try:
    from backend.scripts.deeplab_v3plus_workflow import get_transform as _seg_preproc_transform
except ImportError:
    from deeplab_v3plus_workflow import get_transform as _seg_preproc_transform


SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Match multimodel notebook preprocessing for API inference (ImageNet-normalized 512×512).
UNET_INFERENCE_IMG_SIZE = 512
UNET_INFERENCE_IMAGENET_NORM = True


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


def _polygon_mask(width: int, height: int, annotations: list[dict]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        class_name = ann.get("class")
        class_id = 1 if class_name == "individual_tree" else 2 if class_name == "group_of_trees" else None
        if class_id is None:
            continue
        seg = ann.get("segmentation", [])
        if len(seg) < 6 or len(seg) % 2 != 0:
            continue
        polygon = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
        polygon = np.round(polygon).astype(np.int32)
        cv2.fillPoly(mask, [polygon], class_id)
    return mask


@dataclass
class TrainConfig:
    project_root: Path
    annotations_path: Path
    image_dir: Path
    save_dir: Path
    img_size: int = 512
    batch_size: int = 2
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    workers: int = 2
    augment_style: str = "multimodel"


class SemanticDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        img_size: int,
        training: bool = True,
        augment_style: str = "multimodel",
    ) -> None:
        self.records = records
        self.img_size = img_size
        self.training = training
        self.augment_style = augment_style
        if augment_style == "multimodel":
            self._train_tf = make_multimodel_nb_train_transform(img_size)
            self._val_tf = make_multimodel_nb_val_transform(img_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = cv2.imread(str(record["image_path"]))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {record['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = record["mask"].astype(np.uint8)

        if self.augment_style == "multimodel":
            if self.training:
                out = self._train_tf(image=image, mask=mask)
            else:
                out = self._val_tf(image=image, mask=mask)
            m = out["mask"]
            if m.dtype != torch.long:
                m = m.long()
            m = torch.clamp(m, 0, 2)
            return out["image"], m

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if self.training:
            image, mask = apply_augmentation(image, mask)
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).long()
        return image_t, mask_t


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
        image_path = _find_image_path(config.image_dir, stem)
        if image_path is None:
            continue
        width = int(image_info.get("width", 0))
        height = int(image_info.get("height", 0))
        if width <= 0 or height <= 0:
            continue
        mask = _polygon_mask(width, height, image_info.get("annotations", []))
        record = {"image_path": image_path, "mask": mask}
        if stem in val_split:
            val_records.append(record)
        elif stem in train_split:
            train_records.append(record)
        else:
            train_records.append(record)

    if not train_records:
        raise RuntimeError("No training records found for U-Net.")
    if not val_records:
        n_val = max(1, int(0.20 * len(train_records)))
        val_records = train_records[:n_val]
        train_records = train_records[n_val:]
    return train_records, val_records


def _mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> float:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ious = []
    for cls in range(num_classes):
        pred_c = pred_np == cls
        target_c = target_np == cls
        inter = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        ious.append(float(inter) / float(union + 1e-6))
    return float(np.mean(ious))


def _predict_mask(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    img_size: int,
    device: torch.device,
    imagenet_norm: bool = True,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tf = _seg_preproc_transform(img_size, imagenet_norm=imagenet_norm)
    tensor = tf(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred_small = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred_small, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return pred


def _compute_pixel_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            pred = torch.argmax(model(images), dim=1)
            correct += int((pred == masks).sum().item())
            total += int(masks.numel())
    return round(correct / max(1, total), 6)


def _compute_test_accuracy(
    model: torch.nn.Module, config: TrainConfig, device: torch.device,
) -> float | None:
    eval_annot = resolve_evaluation_annotations_path(config.project_root)
    if not eval_annot.exists():
        return None
    try:
        payload = json.loads(eval_annot.read_text(encoding="utf-8"))
        records: list[dict] = []
        eval_image_dir = resolve_evaluation_image_dir(config.project_root)
        for img_info in payload.get("images", []):
            stem = Path(img_info.get("file_name", "")).stem
            if not stem:
                continue
            img_path = _find_image_path(eval_image_dir, stem)
            if img_path is None:
                continue
            w, h = int(img_info.get("width", 0)), int(img_info.get("height", 0))
            if w <= 0 or h <= 0:
                continue
            mask = _polygon_mask(w, h, img_info.get("annotations", []))
            records.append({"image_path": img_path, "mask": mask})
        if not records:
            return None
        loader = DataLoader(
            SemanticDataset(records, config.img_size, augment_style=config.augment_style),
            batch_size=config.batch_size, shuffle=False, num_workers=0,
        )
        return _compute_pixel_accuracy(model, loader, device)
    except Exception:
        return None


def _save_evaluation_artifacts(
    model: torch.nn.Module,
    records: list[dict],
    config: TrainConfig,
    device: torch.device,
) -> None:
    eval_root = config.project_root / "output" / "evaluation" / "unet"
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

        gt_mask = record["mask"].astype(np.uint8)
        pred_mask = _predict_mask(
            model,
            image_bgr,
            config.img_size,
            device,
            imagenet_norm=config.augment_style == "multimodel",
        )

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
    print(f"[U-Net] evaluation artifacts saved to {eval_root}", flush=True)


def train(config: TrainConfig) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    config.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records, val_records = _load_records(config)
    print(
        f"[U-Net] device={device} train={len(train_records)} val={len(val_records)} "
        f"epochs={config.epochs} batch_size={config.batch_size}",
        flush=True,
    )
    train_loader = DataLoader(
        SemanticDataset(
            train_records, config.img_size, training=True, augment_style=config.augment_style,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
    )
    val_loader = DataLoader(
        SemanticDataset(
            val_records, config.img_size, training=False, augment_style=config.augment_style,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
        activation=None,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = DiceCELoss(num_classes=3)
    early_stopping = EarlyStopping(patience=999)

    best_iou = -1.0
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[U-Net][Train] epoch {epoch + 1}/{config.epochs}", dynamic_ncols=True)
        for batch_idx, (images, masks) in enumerate(pbar, start=1):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu().item())
            pbar.set_postfix(loss=f"{running_loss / max(1, batch_idx):.4f}")
            if batch_idx % 5 == 0 or batch_idx == len(train_loader):
                print(
                    f"[U-Net][Train] epoch={epoch+1}/{config.epochs} "
                    f"batch={batch_idx}/{len(train_loader)} "
                    f"loss={running_loss / max(1, batch_idx):.4f}",
                    flush=True,
                )

        model.eval()
        val_ious = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"[U-Net][Val] epoch {epoch + 1}/{config.epochs}", dynamic_ncols=True)
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                pred = torch.argmax(logits, dim=1)
                val_ious.append(_mean_iou(pred, masks))
                if len(val_ious) % 5 == 0 or len(val_ious) == len(val_loader):
                    print(
                        f"[U-Net][Val] epoch={epoch+1}/{config.epochs} "
                        f"batch={len(val_ious)}/{len(val_loader)}",
                        flush=True,
                    )
        mean_iou = float(np.mean(val_ious)) if val_ious else 0.0
        print(
            f"[U-Net] epoch={epoch+1}/{config.epochs} train_loss={running_loss / max(1, len(train_loader)):.4f} "
            f"val_mIoU={mean_iou:.4f}",
            flush=True,
        )
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(
                {"epoch": epoch, "val_iou": mean_iou, "model_state_dict": model.state_dict()},
                config.save_dir / "best_model.pth",
            )
            print(f"[U-Net] saved best checkpoint: {config.save_dir / 'best_model.pth'}", flush=True)
        if early_stopping.should_stop(mean_iou):
            print(f"[U-Net] early stopping at epoch {epoch+1}", flush=True)
            break

    torch.save({"model_state_dict": model.state_dict()}, config.save_dir / "final_model.pth")
    print(f"[U-Net] complete. Best IoU={best_iou:.4f}. Saved to {config.save_dir}", flush=True)

    print("[U-Net] Computing training results...", flush=True)
    train_eval_loader = DataLoader(
        SemanticDataset(
            train_records, config.img_size, training=False, augment_style=config.augment_style,
        ),
        batch_size=config.batch_size, shuffle=False, num_workers=config.workers,
    )
    train_acc = _compute_pixel_accuracy(model, train_eval_loader, device)
    val_acc = _compute_pixel_accuracy(model, val_loader, device)
    test_acc = _compute_test_accuracy(model, config, device)
    eval_root = config.project_root / "output" / "evaluation" / "unet"
    eval_root.mkdir(parents=True, exist_ok=True)
    training_results = {
        "method": "unet",
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "best_val_iou": round(best_iou, 6),
        "final_train_loss": round(running_loss / max(1, len(train_loader)), 6),
        "epochs": config.epochs,
        "timestamp": int(time.time()),
        "config_note": "Aligned with tree_canopy_multimodel notebook: 512, batch 2, 50 epochs, AdamW 1e-4, wd 1e-4, resnet34, DiceCELoss, multimodel Albumentations.",
    }
    (eval_root / "training_results.json").write_text(
        json.dumps(training_results, indent=2), encoding="utf-8",
    )
    print(
        f"[U-Net] Results: train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        f" test_acc={test_acc if test_acc is not None else 'N/A'}",
        flush=True,
    )

    _save_evaluation_artifacts(model, val_records, config, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U-Net training workflow")
    parser.add_argument("action", choices=["train"], default="train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--augment-style",
        type=str,
        choices=["multimodel", "opencv"],
        default="multimodel",
        help="multimodel: Colab multimodel Albumentations + ImageNet norm. opencv: legacy resize/aug.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = TrainConfig(
        project_root=project_root,
        annotations_path=resolve_train_annotations_path(project_root),
        image_dir=_resolve_image_dir(project_root),
        save_dir=project_root / "checkpoints_unet",
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        workers=args.workers,
        augment_style=args.augment_style,
    )
    if args.action == "train":
        train(config)


if __name__ == "__main__":
    main()
