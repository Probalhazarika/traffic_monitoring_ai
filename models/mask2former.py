# ─────────────────────────────────────────────────────────────────────────────
#  models/mask2former.py
#  Native PyTorch Mask2Former — NO detectron2 dependency.
#
#  Architecture based on:
#    "Masked-attention Mask Transformer for Universal Image Segmentation"
#    Cheng et al., CVPR 2022  (https://arxiv.org/abs/2112.01527)
#
#  Implementation choices:
#   - Swin Transformer backbone (via timm) — pure PyTorch, MPS-compatible
#   - Pixel Decoder: FPN-style multi-scale feature pyramid
#   - Transformer Decoder: masked cross-attention + self-attention
#   - Bipartite matching (Hungarian algorithm via scipy)
#   - Binary output (background / lane)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
#  Swin Backbone (timm)
# ─────────────────────────────────────────────────────────────────────────────

class SwinBackbone(nn.Module):
    """
    Swin Transformer backbone via timm.
    Returns multi-scale feature maps: [C2, C3, C4, C5].
    """

    CONFIGS = {
        "swin_tiny":   ("swin_tiny_patch4_window7_224",   [96,  192, 384,  768]),
        "swin_small":  ("swin_small_patch4_window7_224",  [96,  192, 384,  768]),
        "swin_base":   ("swin_base_patch4_window7_224",   [128, 256, 512, 1024]),
        "swin_large":  ("swin_large_patch4_window7_224",  [192, 384, 768, 1536]),
    }

    def __init__(self, variant: str = "swin_tiny", pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm not installed. Run: pip install timm>=0.9.0")

        timm_name, self.out_channels = self.CONFIGS[variant]
        self.model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        print(f"[SwinBackbone] Loaded {timm_name} (pretrained={pretrained})")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model(x)    # [C2, C3, C4, C5]  NCHW


# ─────────────────────────────────────────────────────────────────────────────
#  Pixel Decoder — multi-scale FPN
# ─────────────────────────────────────────────────────────────────────────────

class PixelDecoder(nn.Module):
    """
    Lightweight Feature Pyramid Network (FPN) pixel decoder.
    Merges C2–C5 features into a dense per-pixel embedding.
    """

    def __init__(self, in_channels: list[int], hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Lateral 1×1 projections
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, 1) for c in in_channels
        ])

        # Output convolutions
        self.output  = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels
        ])

        # Final pixel-level embedding
        self.mask_features = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Parameters
        ----------
        features : [C2, C3, C4, C5] NCHW

        Returns
        -------
        mask_features : (B, hidden_dim, H/4, W/4)
        multi_scale   : [(B, hidden_dim, H/32, W/32), ..., (B, hidden_dim, H/4, W/4)]
        """
        # Lateral projections
        laterals = [lat(f) for lat, f in zip(self.lateral, features)]

        # Top-down FPN merge
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        multi_scale = [out(lat) for out, lat in zip(self.output, laterals)]

        # Highest resolution (C2) → mask features
        mask_features = self.mask_features(multi_scale[0])
        return mask_features, multi_scale


# ─────────────────────────────────────────────────────────────────────────────
#  Masked Cross-Attention Layer
# ─────────────────────────────────────────────────────────────────────────────

class MaskedAttention(nn.Module):
    """
    Cross-attention with predicted mask constraints.

    Each query attends only to the foreground region of its predicted mask,
    which constrains attention to the most relevant spatial region.
    """

    def __init__(self, d_model: int, nheads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nheads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        query:    torch.Tensor,    # (B, Q, d_model)
        key:      torch.Tensor,    # (B, HW, d_model)
        value:    torch.Tensor,    # (B, HW, d_model)
        attn_mask: torch.Tensor | None = None,   # (B*nheads, Q, HW) or None
    ) -> torch.Tensor:
        out, _ = self.attn(query, key, value, attn_mask=attn_mask)
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Transformer Decoder Layer
# ─────────────────────────────────────────────────────────────────────────────

class Mask2FormerDecoderLayer(nn.Module):
    """
    One Mask2Former decoder layer:
    1. Masked cross-attention  (query → pixel features)
    2. Self-attention          (queries attend to each other)
    3. Feed-forward network
    """

    def __init__(self, d_model: int, nheads: int,
                 dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.masked_cross_attn = MaskedAttention(d_model, nheads, dropout)
        self.self_attn         = nn.MultiheadAttention(
            d_model, nheads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries:      torch.Tensor,   # (B, Q, d_model)
        pixel_feats:  torch.Tensor,   # (B, HW, d_model)
        attn_mask:    torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Masked cross-attention
        q2 = self.masked_cross_attn(queries, pixel_feats, pixel_feats, attn_mask)
        queries = self.norm1(queries + self.dropout(q2))

        # 2. Self-attention
        q3, _ = self.self_attn(queries, queries, queries)
        queries = self.norm2(queries + self.dropout(q3))

        # 3. FFN
        q4 = self.ffn(queries)
        queries = self.norm3(queries + q4)

        return queries


# ─────────────────────────────────────────────────────────────────────────────
#  Mask2Former Transformer Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Mask2FormerDecoder(nn.Module):
    """
    Stack of Mask2FormerDecoderLayers with auxiliary mask predictions.
    """

    def __init__(
        self,
        hidden_dim:      int = 256,
        nheads:          int = 8,
        dim_feedforward: int = 2048,
        dec_layers:      int = 9,
        num_queries:     int = 100,
        num_classes:     int = 2,
        dropout:         float = 0.0,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_queries = num_queries

        # Learnable queries
        self.query_feat   = nn.Embedding(num_queries, hidden_dim)
        self.query_embed  = nn.Embedding(num_queries, hidden_dim)

        # Pixel feature projection
        self.pixel_proj   = nn.Linear(hidden_dim, hidden_dim)

        # Positional encoding for pixel features
        self.pixel_pe_layer = PositionEmbeddingSine(hidden_dim // 2)

        # Decoder layers (each layer uses a different scale feature map)
        self.layers = nn.ModuleList([
            Mask2FormerDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
            for _ in range(dec_layers)
        ])

        # Per-layer mask and class prediction heads
        self.mask_embed   = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, hidden_dim, 3) for _ in range(dec_layers)
        ])
        self.class_embed  = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(dec_layers)
        ])

        self.dec_layers    = dec_layers
        self.nheads        = nheads

    def forward(
        self,
        mask_features: torch.Tensor,         # (B, C, H, W)  pixel decoder output
        multi_scale:   list[torch.Tensor],   # [(B, C, H', W'), ...]
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
          pred_logits : (B, Q, num_classes)       final class predictions
          pred_masks  : (B, Q, H, W)              final mask logits
          aux_outputs : list of {pred_logits, pred_masks} per layer
        """
        B, C, H, W = mask_features.shape
        device      = mask_features.device

        # Flatten mask features: (B, HW, C)
        flat_feats  = mask_features.flatten(2).permute(0, 2, 1)
        # Positional encoding: (B, C, H, W) → (B, HW, C)
        pe          = self.pixel_pe_layer(mask_features)
        flat_pe     = pe.flatten(2).permute(0, 2, 1)
        pixel_feats = self.pixel_proj(flat_feats + flat_pe)

        # Initial queries
        queries = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, C)
        query_pe = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        queries_with_pe = queries + query_pe

        aux_outputs = []

        for i, layer in enumerate(self.layers):
            # Select scale: cycle through multi-scale features
            scale_idx    = i % len(multi_scale)
            scale_feat   = multi_scale[scale_idx]
            Hs, Ws       = scale_feat.shape[-2:]
            scale_flat   = scale_feat.flatten(2).permute(0, 2, 1)   # (B, Hs*Ws, C)

            # Predict auxiliary masks for attention masking
            mask_emb     = self.mask_embed[i](queries)               # (B, Q, C)
            aux_masks    = torch.bmm(
                mask_emb,
                scale_flat.permute(0, 2, 1)                          # (B, C, Hs*Ws)
            )                                                          # (B, Q, Hs*Ws)

            # Create binary attention mask: 1 where NOT attending
            attn_mask_bin = (aux_masks.sigmoid() < 0.5)              # (B, Q, Hs*Ws)
            attn_mask_bin = attn_mask_bin.unsqueeze(1).expand(
                -1, self.nheads, -1, -1
            ).reshape(B * self.nheads, self.num_queries, Hs * Ws)

            # Prevent all-masked situation
            if attn_mask_bin.all():
                attn_mask_bin = None
            else:
                attn_mask_bin = attn_mask_bin.float().masked_fill(
                    attn_mask_bin, float("-inf")
                )

            queries = layer(queries_with_pe, scale_flat, attn_mask_bin)

            # Auxiliary predictions
            cls_logits  = self.class_embed[i](queries)               # (B, Q, num_classes)
            msk_emb_out = self.mask_embed[i](queries)                # (B, Q, C)
            msk_logits  = torch.bmm(
                msk_emb_out,
                mask_features.flatten(2)                              # (B, C, H*W)
            ).reshape(B, self.num_queries, H, W)                     # (B, Q, H, W)

            aux_outputs.append({"pred_logits": cls_logits,
                                 "pred_masks":  msk_logits})

        return {
            "pred_logits": aux_outputs[-1]["pred_logits"],
            "pred_masks":  aux_outputs[-1]["pred_masks"],
            "aux_outputs": aux_outputs[:-1],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: MLP
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, l in enumerate(self.layers):
            x = F.relu(l(x)) if i < len(self.layers) - 1 else l(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: Sinusoidal position encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature   = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        y_embed = torch.arange(H, device=device).float().view(1, H, 1).expand(B, -1, W)
        x_embed = torch.arange(W, device=device).float().view(1, 1, W).expand(B, H, -1)
        dim_t   = torch.arange(self.num_pos_feats, device=device).float()
        dim_t   = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x   = x_embed[..., None] / dim_t
        pos_y   = y_embed[..., None] / dim_t
        pos_x   = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], -1).flatten(-2)
        pos_y   = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], -1).flatten(-2)
        pos     = torch.cat([pos_y, pos_x], dim=-1)      # (B, H, W, 2*num_pos_feats)
        pos     = pos.permute(0, 3, 1, 2)                # (B, C, H, W)
        return pos


# ─────────────────────────────────────────────────────────────────────────────
#  Bipartite Matching
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def hungarian_matching(
    pred_masks:   torch.Tensor,   # (Q, H, W) logits
    pred_classes: torch.Tensor,   # (Q, num_classes)
    gt_masks:     torch.Tensor,   # (N, H, W) binary
    gt_labels:    torch.Tensor,   # (N,) int
    cost_class:   float = 2.0,
    cost_mask:    float = 5.0,
    cost_dice:    float = 5.0,
) -> tuple[list, list]:
    """
    Hungarian matching between Q predictions and N ground truths.
    Returns (pred_indices, gt_indices).
    """
    Q = pred_masks.size(0)
    N = gt_masks.size(0)
    if N == 0:
        return [], []

    pred_probs = torch.sigmoid(pred_masks).flatten(1)   # (Q, H*W)
    gt_flat    = gt_masks.flatten(1).float()             # (N, H*W)

    # Classification cost
    pred_cls_prob = torch.softmax(pred_classes, dim=-1)   # (Q, C)
    gt_class_col  = gt_labels.long()
    cost_c        = -pred_cls_prob[:, gt_class_col]        # (Q, N)

    # Mask focal cost (approximate)
    cost_m = _batch_sigmoid_focal(pred_probs, gt_flat)    # (Q, N)

    # Dice cost
    cost_d = _batch_dice(pred_probs, gt_flat)             # (Q, N)

    C = cost_class * cost_c + cost_mask * cost_m + cost_dice * cost_d
    C = C.cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(C)
    return list(row_ind), list(col_ind)


def _batch_sigmoid_focal(pred: torch.Tensor, gt: torch.Tensor,
                          alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    Q, HW = pred.shape
    N     = gt.shape[0]
    pred  = pred.unsqueeze(1).expand(-1, N, -1)  # (Q, N, HW)
    gt    = gt.unsqueeze(0).expand(Q, -1, -1)    # (Q, N, HW)
    neg_cost = -(1 - alpha) * (pred ** gamma) * (1 - pred + 1e-8).log()
    pos_cost = -alpha * ((1 - pred) ** gamma) * (pred + 1e-8).log()
    cost     = (pos_cost * gt + neg_cost * (1 - gt)).mean(-1)
    return cost   # (Q, N)


def _batch_dice(pred: torch.Tensor, gt: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    Q, HW = pred.shape
    N     = gt.shape[0]
    pred  = pred.unsqueeze(1).expand(-1, N, -1)
    gt    = gt.unsqueeze(0).expand(Q, -1, -1)
    inter = (pred * gt).sum(-1)
    denom = pred.sum(-1) + gt.sum(-1)
    dice  = 1 - (2 * inter + smooth) / (denom + smooth)
    return dice   # (Q, N)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Mask2FormerLane model
# ─────────────────────────────────────────────────────────────────────────────

class Mask2FormerLane(nn.Module):
    """
    Native PyTorch Mask2Former for binary aerial lane segmentation.
    No detectron2 required — pure PyTorch + timm.

    Usage
    ─────
        model = Mask2FormerLane()
        output = model(pixel_values)
        # output["pred_masks"]  : (B, Q, H, W) — per-query mask logits
        # output["pred_logits"] : (B, Q, C)    — per-query class logits

    For inference (get final binary mask):
        mask = model.predict(pixel_values)   # (B, H, W) uint8
    """

    def __init__(
        self,
        backbone:        str  = "swin_tiny",
        num_classes:     int  = 2,
        hidden_dim:      int  = 256,
        nheads:          int  = 8,
        dim_feedforward: int  = 2048,
        dec_layers:      int  = 9,
        num_queries:     int  = 100,
        image_size:      int  = 1024,
        pretrained:      bool = True,
        dropout:         float = 0.0,
    ):
        super().__init__()
        self.image_size  = image_size
        self.num_queries = num_queries
        self.num_classes = num_classes

        print(f"[Mask2FormerLane] Building native PyTorch model "
              f"(backbone={backbone}, queries={num_queries}, layers={dec_layers})")

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone = SwinBackbone(backbone, pretrained=pretrained)
        bb_channels   = self.backbone.out_channels   # [C2, C3, C4, C5]

        # ── Project all backbone channels to hidden_dim ───────────────────
        self.input_proj = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, 1) for c in bb_channels
        ])

        # ── Pixel Decoder ────────────────────────────────────────────────────
        self.pixel_decoder = PixelDecoder(
            in_channels=[hidden_dim] * len(bb_channels),
            hidden_dim=hidden_dim,
        )

        # ── Transformer Decoder ───────────────────────────────────────────────
        self.transformer_decoder = Mask2FormerDecoder(
            hidden_dim=hidden_dim,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            num_queries=num_queries,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        pixel_values : (B, 3, H, W)

        Returns
        -------
        dict:
          pred_logits : (B, Q, num_classes)
          pred_masks  : (B, Q, H/4, W/4)   — upsampled in predict()
          aux_outputs : list of {pred_logits, pred_masks}
        """
        # Backbone features
        bb_feats = self.backbone(pixel_values)   # [C2, C3, C4, C5]

        # Project to hidden_dim
        proj_feats = [proj(f) for proj, f in zip(self.input_proj, bb_feats)]

        # Pixel decoder
        mask_features, multi_scale = self.pixel_decoder(proj_feats)

        # Transformer decoder
        output = self.transformer_decoder(mask_features, multi_scale)
        return output

    def predict(self, pixel_values: torch.Tensor,
                threshold: float = 0.5) -> torch.Tensor:
        """
        Run inference and return merged binary mask (B, H, W).

        Query masks are merged: max pooling over all Q queries after sigmoid.
        """
        with torch.no_grad():
            output = self.forward(pixel_values)

        pred_masks = output["pred_masks"]    # (B, Q, H', W')
        H, W       = self.image_size, self.image_size

        # Upsample to input resolution
        pred_masks = F.interpolate(
            pred_masks, size=(H, W), mode="bilinear", align_corners=False
        )
        pred_probs = torch.sigmoid(pred_masks)   # (B, Q, H, W)

        # Merge: take max across query dimension
        merged, _ = pred_probs.max(dim=1)        # (B, H, W)
        return (merged >= threshold).to(torch.uint8)

    def compute_loss(
        self,
        output:    dict,
        gt_masks:  torch.Tensor,    # (B, N, H, W) binary
        gt_labels: torch.Tensor,    # (B, N) int class labels
    ) -> torch.Tensor:
        """
        Compute Mask2Former loss with Hungarian matching.
        Includes auxiliary losses from all decoder layers.
        """
        from utils.losses import Mask2FormerLoss
        loss_fn = Mask2FormerLoss()

        B = output["pred_masks"].size(0)
        H, W = self.image_size, self.image_size

        # Upsample predicted masks to GT resolution
        pred_masks_up = F.interpolate(
            output["pred_masks"], size=(H, W), mode="bilinear", align_corners=False
        )
        pred_logits = output["pred_logits"]

        # Per-sample Hungarian matching
        matching = []
        for b in range(B):
            n_gt = int((gt_labels[b] >= 0).sum())
            if n_gt == 0:
                matching.append(([], []))
                continue
            pred_idx, gt_idx = hungarian_matching(
                pred_masks_up[b], pred_logits[b],
                gt_masks[b, :n_gt], gt_labels[b, :n_gt]
            )
            matching.append((pred_idx, gt_idx))

        # Main loss
        total = loss_fn(pred_logits, pred_masks_up, gt_masks, gt_labels, matching)

        # Auxiliary losses (lower weight)
        for aux in output.get("aux_outputs", []):
            aux_masks_up = F.interpolate(
                aux["pred_masks"], size=(H, W), mode="bilinear", align_corners=False
            )
            aux_matching = []
            for b in range(B):
                n_gt = int((gt_labels[b] >= 0).sum())
                if n_gt == 0:
                    aux_matching.append(([], []))
                    continue
                pi, gi = hungarian_matching(
                    aux_masks_up[b], aux["pred_logits"][b],
                    gt_masks[b, :n_gt], gt_labels[b, :n_gt]
                )
                aux_matching.append((pi, gi))
            aux_loss = loss_fn(
                aux["pred_logits"], aux_masks_up, gt_masks, gt_labels, aux_matching
            )
            total = total + 0.5 * aux_loss   # auxiliary weight = 0.5

        return total

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Mask2FormerLane] Backbone frozen.")

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        print("[Mask2FormerLane] All parameters unfrozen.")
