# ─────────────────────────────────────────────────────────────────────────────
#  utils/augmentations.py
#  Albumentations augmentation pipelines for aerial lane segmentation.
# ─────────────────────────────────────────────────────────────────────────────

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 1024,
                          mean: tuple = (0.485, 0.456, 0.406),
                          std:  tuple = (0.229, 0.224, 0.225)) -> A.Compose:
    """
    Full augmentation pipeline for training.

    Augmentations chosen specifically for aerial lane detection:
    - Geometric: handles arbitrary drone angles
    - Photometric: handles illumination variation (dawn, dusk, shadows)
    - Blur / noise: simulates motion blur from moving drone
    - Perspective: simulates oblique viewing angles
    """
    return A.Compose([
        # ── Resize ──────────────────────────────────────────────────────────────────
        A.Resize(image_size, image_size),

        # ── Geometric transforms ───────────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=30, border_mode=0, p=0.4),
        A.Perspective(scale=(0.02, 0.08), p=0.3),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.85, 1.15),
            rotate=(-20, 20),
            p=0.4,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),

        # ── Photometric transforms ───────────────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3,
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
            p=0.3,
        ),

        # ── Noise & blur ──────────────────────────────────────────────────────────────
        A.GaussNoise(std_range=(0.02, 0.12), p=0.3),
        A.MotionBlur(blur_limit=(3, 11), p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        A.ImageCompression(quality_range=(75, 100), p=0.2),

        # ── Coarse dropout (simulate occlusion from vehicles/trees) ──────────────
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(8, 64),
            hole_width_range=(8, 64),
            fill=0,
            p=0.2,
        ),

        # ── Normalize + ToTensor ───────────────────────────────────────────────────
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 1024,
                        mean: tuple = (0.485, 0.456, 0.406),
                        std:  tuple = (0.229, 0.224, 0.225)) -> A.Compose:
    """
    Minimal val/test pipeline — resize + normalize only.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size: int = 1024,
                        mean: tuple = (0.485, 0.456, 0.406),
                        std:  tuple = (0.229, 0.224, 0.225)) -> list:
    """
    Test-Time Augmentation (TTA) pipelines.
    Returns a list of transforms; inference averages predictions over all.
    """
    base = [
        A.Resize(image_size, image_size, always_apply=True),
        A.Normalize(mean=mean, std=std, always_apply=True),
        ToTensorV2(),
    ]
    flips = [
        [],
        [A.HorizontalFlip(p=1.0)],
        [A.VerticalFlip(p=1.0)],
        [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],
    ]
    return [A.Compose(flip + base) for flip in flips]
