# ─────────────────────────────────────────────────────────────────────────────
#  models/segformer.py
#  SegFormer-B5 wrapper for binary aerial lane segmentation.
#  Uses HuggingFace Transformers — no detectron2 required.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerLane(nn.Module):
    """
    SegFormer-B5 binary lane segmentation model.

    Architecture
    ────────────
    Backbone : Mix Transformer (MiT-B5) — hierarchical patch merging
    Decoder  : Lightweight All-MLP head (4 upsampling stages)
    Output   : (B, 1, H, W) logits  [binary: background / lane]

    The HuggingFace `SegformerForSemanticSegmentation` outputs logits at
    H/4 × W/4.  We upsample back to full input resolution.

    Parameters
    ----------
    backbone    : HuggingFace model id, e.g. "nvidia/mit-b5"
    num_classes : 2 for binary lane segmentation
    image_size  : input image resolution (square)
    pretrained  : load ImageNet-pretrained backbone weights
    """

    def __init__(
        self,
        backbone:    str  = "nvidia/mit-b5",
        num_classes: int  = 2,
        image_size:  int  = 1024,
        pretrained:  bool = True,
    ):
        super().__init__()
        self.image_size  = image_size
        self.num_classes = num_classes

        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers>=4.37.0"
            )

        if pretrained:
            print(f"[SegFormerLane] Loading pretrained: {backbone}")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                backbone,
                num_labels=num_classes,
                id2label={0: "background", 1: "lane"},
                label2id={"background": 0, "lane": 1},
                ignore_mismatched_sizes=True,
            )
        else:
            print(f"[SegFormerLane] Building from scratch (no pretrained weights).")
            cfg = SegformerConfig.from_pretrained(backbone)
            cfg.num_labels = num_classes
            cfg.id2label   = {0: "background", 1: "lane"}
            cfg.label2id   = {"background": 0, "lane": 1}
            self.model     = SegformerForSemanticSegmentation(cfg)

        # ── Binary output head ─────────────────────────────────────────────
        # Replace the default (num_classes)-channel head with a 1-channel
        # head for binary segmentation (lane / no-lane).
        # The HuggingFace decode_head outputs (B, num_classes, H/4, W/4).
        # We add a 1×1 conv to reduce to a single logit channel.
        if num_classes > 1:
            in_ch = num_classes
            self.binary_head = nn.Conv2d(in_ch, 1, kernel_size=1)
        else:
            self.binary_head = None

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values : (B, 3, H, W) — normalized RGB

        Returns
        -------
        logits : (B, 1, H, W) — raw logit (apply sigmoid for probability)
        """
        outputs = self.model(pixel_values=pixel_values)
        logits  = outputs.logits   # (B, num_classes, H/4, W/4)

        if self.binary_head is not None:
            logits = self.binary_head(logits)   # → (B, 1, H/4, W/4)

        # Upsample to input resolution
        logits = F.interpolate(
            logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return logits   # (B, 1, H, W)

    def predict(self, pixel_values: torch.Tensor,
                threshold: float = 0.5) -> torch.Tensor:
        """
        Convenience method: returns binary mask (B, H, W) as uint8 tensor.
        """
        with torch.no_grad():
            logits = self.forward(pixel_values)
        probs = torch.sigmoid(logits).squeeze(1)   # (B, H, W)
        return (probs >= threshold).to(torch.uint8)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self) -> None:
        """Freeze the MiT encoder — fine-tune only the decode head."""
        for name, param in self.named_parameters():
            if "decode_head" not in name and "binary_head" not in name:
                param.requires_grad = False
        print("[SegFormerLane] Backbone frozen. Only decode head trainable.")

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        print("[SegFormerLane] All parameters unfrozen.")
