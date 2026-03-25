from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import segmentation_models_pytorch as smp
except Exception as exc:  # pragma: no cover - runtime environment dependent
    raise RuntimeError("segmentation_models_pytorch is required for DeepLabV3+ workflow.") from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(x, **kwargs):
        return x


SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
SCENE_LABELS = [
    "agriculture_plantation",
    "industrial_area",
    "open_field",
    "rural_area",
    "urban_area",
]


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
class DeepLabV3PlusConfig:
    PROJECT_DIR: Path = Path(__file__).resolve().parents[2]
    ANNOTATIONS_PATH: Path | None = None
    IMAGE_DIR: Path | None = None
    EVAL_IMAGE_DIR: Path | None = None
    SAVE_DIR: Path | None = None
    OUTPUT_DIR: Path | None = None
    TRAIN_SPLIT_TXT: Path | None = None
    VAL_SPLIT_TXT: Path | None = None

    IMG_SIZE: int = 640
    BATCH_SIZE: int = 4
    NUM_EPOCHS: int = 60
    LEARNING_RATE: float = 1e-4
    NUM_WORKERS: int = 0
    ENCODER_NAME: str = "resnet34"

    def __post_init__(self) -> None:
        if self.ANNOTATIONS_PATH is None:
            self.ANNOTATIONS_PATH = self.PROJECT_DIR / "data" / "raw" / "annotations" / "train_annotations.json"
        if self.IMAGE_DIR is None:
            self.IMAGE_DIR = _resolve_image_dir(self.PROJECT_DIR, "train")
        if self.EVAL_IMAGE_DIR is None:
            self.EVAL_IMAGE_DIR = _resolve_image_dir(self.PROJECT_DIR, "evaluation")
        if self.SAVE_DIR is None:
            self.SAVE_DIR = self.PROJECT_DIR / "checkpoints_deeplabv3plus"
        if self.OUTPUT_DIR is None:
            self.OUTPUT_DIR = self.PROJECT_DIR / "output" / "evaluation" / "deeplabv3plus"
        if self.TRAIN_SPLIT_TXT is None:
            self.TRAIN_SPLIT_TXT = self.PROJECT_DIR / "data" / "processed" / "train.txt"
        if self.VAL_SPLIT_TXT is None:
            self.VAL_SPLIT_TXT = self.PROJECT_DIR / "data" / "processed" / "val.txt"

class SemanticDataset(Dataset):
    def __init__(self, records: list[dict], img_size: int):
        self.records = records
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = cv2.imread(str(record["image_path"]))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {record['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = record["mask"]
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).long()
        return image_t, mask_t


def ensure_output_dirs(config: DeepLabV3PlusConfig) -> None:
    config.SAVE_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "masks").mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "overlays").mkdir(parents=True, exist_ok=True)


def _load_records(config: DeepLabV3PlusConfig) -> tuple[list[dict], list[dict]]:
    payload = json.loads(config.ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    train_split = _read_split_file(config.TRAIN_SPLIT_TXT)
    val_split = _read_split_file(config.VAL_SPLIT_TXT)

    train_records: list[dict] = []
    val_records: list[dict] = []
    for image_info in payload.get("images", []):
        stem = Path(image_info.get("file_name", "")).stem
        if not stem:
            continue
        image_path = _find_image_path(config.IMAGE_DIR, stem)
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
        raise RuntimeError("No training records found for DeepLabV3+.")
    if not val_records:
        n_val = max(1, int(0.15 * len(train_records)))
        val_records = train_records[:n_val]
        train_records = train_records[n_val:]
    return train_records, val_records


def _build_model(config: DeepLabV3PlusConfig) -> torch.nn.Module:
    return smp.DeepLabV3Plus(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=None,
        in_channels=3,
        classes=3,
        activation=None,
    )


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


def get_transform(img_size: int = 640) -> Callable[[np.ndarray], torch.Tensor]:
    def _transform(image_rgb: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return _transform


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    transform: Callable[[np.ndarray], torch.Tensor],
    device: torch.device,
):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    model.eval()
    with torch.no_grad():
        x = transform(image_rgb).unsqueeze(0).to(device)
        logits = model(x)
        probs_small = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred_small = np.argmax(probs_small, axis=0).astype(np.uint8)
        pred_mask = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)

    probs_hwc = np.transpose(probs_small, (1, 2, 0))
    probs = cv2.resize(probs_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
    scene_class = 0
    return Image.fromarray(image_rgb), pred_mask, scene_class, probs


def _save_evaluation_artifacts(
    model: torch.nn.Module,
    val_records: list[dict],
    config: DeepLabV3PlusConfig,
    device: torch.device,
) -> None:
    eval_root = config.OUTPUT_DIR
    masks_dir = eval_root / "masks"
    overlays_dir = eval_root / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    color_lut = np.array([[0, 0, 0], [0, 255, 0], [255, 255, 0]], dtype=np.uint8)

    cm = np.zeros((3, 3), dtype=np.int64)
    image_ious: list[float] = []
    tp_like_ious: list[float] = []
    transform = get_transform(config.IMG_SIZE)

    for idx, record in enumerate(val_records, start=1):
        image_path = record["image_path"]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        gt_mask = record["mask"].astype(np.uint8)
        _, pred_mask, _, _ = predict_image(model, str(image_path), transform, device)

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
        "num_images": len(val_records),
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
    print(f"[DeepLabV3+] evaluation artifacts saved to {eval_root}", flush=True)


def train(config: DeepLabV3PlusConfig) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    ensure_output_dirs(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records, val_records = _load_records(config)
    print(
        f"[DeepLabV3+] device={device} train={len(train_records)} val={len(val_records)} "
        f"epochs={config.NUM_EPOCHS} batch_size={config.BATCH_SIZE} img_size={config.IMG_SIZE}",
        flush=True,
    )

    train_loader = DataLoader(
        SemanticDataset(train_records, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        SemanticDataset(val_records, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = _build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_iou = -1.0
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[DeepLabV3+][Train] epoch {epoch + 1}/{config.NUM_EPOCHS}", dynamic_ncols=True)
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
                    f"[DeepLabV3+][Train] epoch={epoch+1}/{config.NUM_EPOCHS} "
                    f"batch={batch_idx}/{len(train_loader)} loss={running_loss / max(1, batch_idx):.4f}",
                    flush=True,
                )

        model.eval()
        val_ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                pred = torch.argmax(logits, dim=1)
                val_ious.append(_mean_iou(pred, masks))
        mean_iou = float(np.mean(val_ious)) if val_ious else 0.0
        print(
            f"[DeepLabV3+] epoch={epoch+1}/{config.NUM_EPOCHS} "
            f"train_loss={running_loss / max(1, len(train_loader)):.4f} val_mIoU={mean_iou:.4f}",
            flush=True,
        )
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(
                {"epoch": epoch, "val_iou": mean_iou, "model_state_dict": model.state_dict()},
                config.SAVE_DIR / "best_model_fold0.pth",
            )
            print(f"[DeepLabV3+] saved best checkpoint: {config.SAVE_DIR / 'best_model_fold0.pth'}", flush=True)

    torch.save({"model_state_dict": model.state_dict()}, config.SAVE_DIR / "final_model.pth")
    print(f"[DeepLabV3+] complete. Best IoU={best_iou:.4f}. Saved to {config.SAVE_DIR}", flush=True)
    _save_evaluation_artifacts(model, val_records, config, device)


def load_best_model(config: DeepLabV3PlusConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(config).to(device)
    candidates = [
        config.SAVE_DIR / "final_model.pth",
        config.SAVE_DIR / "best_model_fold0.pth",
    ] + sorted(config.SAVE_DIR.glob("best_model_fold*.pth"))
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"No DeepLab checkpoints found under {config.SAVE_DIR}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, device, "single"


def evaluate(config: DeepLabV3PlusConfig) -> None:
    ensure_output_dirs(config)
    model, device, _ = load_best_model(config)
    _, val_records = _load_records(config)
    _save_evaluation_artifacts(model, val_records, config, device)


def run_predict(config: DeepLabV3PlusConfig, num_images: int = 6) -> None:
    ensure_output_dirs(config)
    model, device, _ = load_best_model(config)
    transform = get_transform(config.IMG_SIZE)
    image_paths = []
    for suffix in SUPPORTED_SUFFIXES:
        image_paths.extend(config.EVAL_IMAGE_DIR.glob(f"*{suffix}"))
    image_paths = sorted(set(image_paths))[:num_images]
    if not image_paths:
        print(f"[DeepLabV3+] No evaluation images found in {config.EVAL_IMAGE_DIR}", flush=True)
        return
    out_dir = config.PROJECT_DIR / "predictions_visualization" / "deeplabv3plus"
    out_dir.mkdir(parents=True, exist_ok=True)
    color_lut = np.array([[0, 0, 0], [0, 255, 0], [255, 255, 0]], dtype=np.uint8)

    for image_path in image_paths:
        original_image, seg_mask, scene_class, _ = predict_image(model, str(image_path), transform, device)
        mask_img = (seg_mask * 127).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{image_path.stem}_mask.png"), mask_img)
        color_mask = color_lut[np.clip(seg_mask, 0, 2)]
        overlay = (0.6 * np.asarray(original_image) + 0.4 * color_mask).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{image_path.stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        scene_label = SCENE_LABELS[scene_class] if 0 <= scene_class < len(SCENE_LABELS) else f"class_{scene_class}"
        print(f"[DeepLabV3+] predicted {image_path.name} -> scene={scene_label}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepLabV3+ workflow")
    parser.add_argument("action", choices=["train", "evaluate", "predict"], default="train")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DeepLabV3PlusConfig(
        NUM_EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        IMG_SIZE=args.img_size,
        LEARNING_RATE=args.lr,
        NUM_WORKERS=args.workers,
    )
    if args.action == "train":
        train(config)
    elif args.action == "evaluate":
        evaluate(config)
    elif args.action == "predict":
        run_predict(config, num_images=args.num_images)


if __name__ == "__main__":
    main()
