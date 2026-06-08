# models/__init__.py
from models.segformer import SegFormerLane
from models.mask2former import Mask2FormerLane

__all__ = ["SegFormerLane", "Mask2FormerLane"]


def build_model(model_type: str, config: dict):
    """Factory: return the correct model based on model_type."""
    if model_type == "segformer":
        return SegFormerLane(
            backbone=config.get("backbone", "nvidia/mit-b5"),
            num_classes=config.get("num_classes", 2),
            image_size=config.get("image_size", 1024),
        )
    elif model_type == "mask2former":
        return Mask2FormerLane(
            backbone=config.get("backbone", "swin_tiny"),
            num_classes=config.get("num_classes", 2),
            hidden_dim=config.get("hidden_dim", 256),
            nheads=config.get("nheads", 8),
            dim_feedforward=config.get("dim_feedforward", 2048),
            dec_layers=config.get("dec_layers", 9),
            num_queries=config.get("num_queries", 100),
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Use 'segformer' or 'mask2former'.")
