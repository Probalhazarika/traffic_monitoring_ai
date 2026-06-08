# ─────────────────────────────────────────────────────────────────────────────
#  utils/losses.py
#  Combined Dice + Binary Cross-Entropy loss for binary lane segmentation.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    DiceLoss = 1 - Dice

    Works on logits (applies sigmoid internally) or on probabilities.
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth       = smooth
        self.from_logits  = from_logits

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, 1, H, W) or (B, H, W) — raw logits or probabilities
        targets : (B, H, W)    — binary ground-truth mask, values in {0, 1}
        """
        if self.from_logits:
            probs = torch.sigmoid(logits)
        else:
            probs = logits

        # Flatten spatial dims
        if probs.dim() == 4:
            probs = probs.squeeze(1)          # (B, H, W)

        targets = targets.float()
        probs   = probs.float()

        B = probs.size(0)
        probs_flat   = probs.view(B, -1)
        targets_flat = targets.view(B, -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union        = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Binary Focal Loss — down-weights easy negatives.
    Especially useful for aerial lane detection where lanes are thin (class imbalance).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 from_logits: bool = True):
        super().__init__()
        self.alpha       = alpha
        self.gamma       = gamma
        self.from_logits = from_logits

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(
                logits.squeeze(1) if logits.dim() == 4 else logits,
                targets.float(),
                reduction="none",
            )
        else:
            bce = F.binary_cross_entropy(
                logits.squeeze(1) if logits.dim() == 4 else logits,
                targets.float(),
                reduction="none",
            )

        probs = torch.sigmoid(logits.squeeze(1) if logits.dim() == 4 else logits)
        pt    = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """
    Production loss for aerial lane detection:

        L = w_dice * DiceLoss + w_bce * BCEWithLogitsLoss

    Class-imbalance handling:
      - pos_weight is computed from dataset lane pixel ratio.
      - Dice loss is inherently recall-focused.

    Optionally add Focal component:
      - Set focal_weight > 0 to activate.
    """

    def __init__(
        self,
        dice_weight:   float = 0.5,
        bce_weight:    float = 0.5,
        focal_weight:  float = 0.0,
        pos_weight:    float = None,
        smooth:        float = 1.0,
        focal_gamma:   float = 2.0,
    ):
        super().__init__()
        self.dice_weight  = dice_weight
        self.bce_weight   = bce_weight
        self.focal_weight = focal_weight

        self.dice = DiceLoss(smooth=smooth, from_logits=True)

        pw = torch.tensor([pos_weight]) if pos_weight else None
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.focal = FocalLoss(gamma=focal_gamma) if focal_weight > 0 else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, 1, H, W) or (B, H, W) — raw model output (not sigmoid-ed)
        targets : (B, H, W)    — binary GT mask {0, 1}
        """
        # Squeeze channel dim for BCE
        logits_sq = logits.squeeze(1) if logits.dim() == 4 else logits

        dice_l = self.dice(logits, targets)
        bce_l  = self.bce(logits_sq, targets.float())

        loss = self.dice_weight * dice_l + self.bce_weight * bce_l

        if self.focal_weight > 0 and self.focal is not None:
            focal_l = self.focal(logits, targets)
            loss = loss + self.focal_weight * focal_l

        return loss


class Mask2FormerLoss(nn.Module):
    """
    Multi-component Mask2Former loss:
      L = λ_ce * CrossEntropy + λ_mask * BinaryFocal + λ_dice * Dice

    Used in the bipartite matching step during Mask2Former training.
    Also exposed for direct use on matched masks.
    """

    def __init__(
        self,
        ce_weight:   float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
    ):
        super().__init__()
        self.ce_weight   = ce_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.dice        = DiceLoss(from_logits=True)
        self.focal       = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)

    def forward(
        self,
        pred_logits: torch.Tensor,   # (B, Q, H, W) — query mask logits
        pred_classes: torch.Tensor,  # (B, Q, C)    — query class logits
        gt_masks: torch.Tensor,      # (B, N, H, W) — ground truth masks
        gt_labels: torch.Tensor,     # (B, N)       — ground truth class labels
        matching: list,              # bipartite assignment indices per batch
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=pred_logits.device)
        B = pred_logits.size(0)

        for b in range(B):
            pred_idx, gt_idx = matching[b]
            if len(pred_idx) == 0:
                continue

            matched_pred_logits  = pred_logits[b][pred_idx]   # (M, H, W)
            matched_gt_masks     = gt_masks[b][gt_idx]        # (M, H, W)
            matched_pred_classes = pred_classes[b][pred_idx]  # (M, C)
            matched_gt_labels    = gt_labels[b][gt_idx].long()# (M,)

            # 1. Focal mask loss
            mask_loss = self.focal(
                matched_pred_logits,
                matched_gt_masks.float(),
            )

            # 2. Dice mask loss
            dice_loss = self.dice(
                matched_pred_logits.unsqueeze(1),
                matched_gt_masks.float(),
            )

            # 3. Classification cross-entropy
            ce_loss = F.cross_entropy(matched_pred_classes, matched_gt_labels)

            total_loss = total_loss + (
                self.ce_weight   * ce_loss   +
                self.mask_weight * mask_loss +
                self.dice_weight * dice_loss
            )

        return total_loss / B
