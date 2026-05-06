"""Shared training utilities: combined loss, data augmentation, early stopping."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------- Combined Loss (Dice + Focal + Cross-Entropy) ---------------

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dims)
        cardinality = (probs + targets_oh).sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class DiceCELoss(nn.Module):
    """Cross-entropy + soft Dice term (tree_canopy_multimodel_training_evaluation_fixed.ipynb)."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = torch.sum(probs * one_hot, dims)
        denom = torch.sum(probs + one_hot, dims)
        dice = (2 * inter + 1e-6) / (denom + 1e-6)
        return ce_loss + (1 - dice.mean())


class DiceFocalSegLoss(nn.Module):
    """0.5 * Dice + 0.5 * Focal (multiclass), matching tree_segmentation_training.ipynb segmentation branch."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("segmentation_models_pytorch is required for DiceFocalSegLoss.") from exc
        mode = "multiclass" if num_classes > 2 else "binary"
        self._dice = smp.losses.DiceLoss(mode=mode)
        self._focal = smp.losses.FocalLoss(mode=mode)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self._dice(logits, targets) + 0.5 * self._focal(logits, targets)


class CombinedLoss(nn.Module):
    """Dice Loss + Focal Loss + Cross-Entropy."""

    def __init__(
        self,
        num_classes: int = 3,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        ce_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes)
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.ce_weight * self.ce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
            + self.focal_weight * self.focal(logits, targets)
        )


# --------------- Data Augmentation ---------------

def apply_augmentation(
    image: np.ndarray,
    mask: np.ndarray | list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray | list[np.ndarray]]:
    """Apply random augmentations (rotation, flipping, brightness, cropping).

    Works with a single semantic mask or a list of instance masks.
    Spatial transforms are applied identically to image and all masks.
    """
    is_list = isinstance(mask, list)
    masks = mask if is_list else [mask]

    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
        masks = [cv2.flip(m, 1) for m in masks]

    if np.random.random() > 0.5:
        image = cv2.flip(image, 0)
        masks = [cv2.flip(m, 0) for m in masks]

    k = np.random.randint(0, 4)
    if k > 0:
        image = np.rot90(image, k).copy()
        masks = [np.rot90(m, k).copy() for m in masks]

    if np.random.random() > 0.5:
        factor = np.random.uniform(0.7, 1.3)
        image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    if np.random.random() > 0.5:
        h, w = image.shape[:2]
        ratio = np.random.uniform(0.8, 1.0)
        ch, cw = int(h * ratio), int(w * ratio)
        if ch > 0 and cw > 0 and h > ch and w > cw:
            y0 = np.random.randint(0, h - ch)
            x0 = np.random.randint(0, w - cw)
            image = cv2.resize(
                image[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR,
            )
            masks = [
                cv2.resize(m[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_NEAREST)
                for m in masks
            ]

    return (image, masks) if is_list else (image, masks[0])


def make_deeplab_notebook_train_transform(img_size: int):
    """Albumentations pipeline aligned with tree_segmentation_training.ipynb get_training_augmentation."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("albumentations is required for notebook-style DeepLab augmentation.") from exc

    coarse: list = []
    try:
        coarse.append(A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3))
    except Exception:
        try:
            coarse.append(A.CoarseDropout(p=0.3))
        except Exception:
            pass

    noise: list = []
    try:
        noise.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.3))
    except Exception:
        try:
            noise.append(A.GaussNoise(p=0.3))
        except Exception:
            pass

    return A.Compose(
        [
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, p=1.0),
            A.RandomCrop(height=img_size, width=img_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            *noise,
            *coarse,
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def make_multimodel_nb_train_transform(img_size: int):
    """Albumentations train pipeline from tree_canopy_multimodel_training_evaluation_fixed.ipynb.

    Matches that notebook's ``train_tfms``: Resize, H/V flip, RandomRotate90, RandomBrightnessContrast(p=0.3),
    then normalize (notebook uses ``A.Normalize()`` with ImageNet defaults; we set mean/std explicitly).
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("albumentations is required for multimodel notebook-style training.") from exc

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def make_multimodel_nb_val_transform(img_size: int):
    """Albumentations eval pipeline from multimodel Colab notebook."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("albumentations is required for multimodel notebook-style validation.") from exc

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def make_deeplab_notebook_val_transform(img_size: int):
    """Validation: resize + ImageNet normalize + tensor (notebook val pipeline)."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("albumentations is required for notebook-style DeepLab validation.") from exc

    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# --------------- Early Stopping ---------------

class EarlyStopping:
    """Stop training when a monitored metric stops improving (higher-is-better or lower-is-better)."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError("EarlyStopping mode must be 'max' or 'min'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None

    def step(self, score: float) -> tuple[bool, bool]:
        """Update with one validation score. Returns (improved, should_stop).

        NaN scores are ignored: no improvement, patience unchanged, do not stop.
        Accepts Python ``float`` or NumPy scalar (e.g. validation metrics from NumPy).
        """
        try:
            s = float(score)
        except (TypeError, ValueError):
            return False, False
        if math.isnan(s):
            return False, False
        if self.best_score is None:
            self.best_score = s
            self.counter = 0
            return True, False
        if self.mode == "max":
            improved = s > self.best_score + self.min_delta
        else:
            improved = s < self.best_score - self.min_delta
        if improved:
            self.best_score = s
            self.counter = 0
            return True, False
        self.counter += 1
        return False, self.counter >= self.patience

    def should_stop(self, score: float) -> bool:
        """Return True when no improvement for *patience* evaluations."""
        _, stop = self.step(score)
        return stop
