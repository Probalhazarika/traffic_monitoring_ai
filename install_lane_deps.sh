#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  install_lane_deps.sh
#  One-shot installer for all aerial lane detection dependencies.
#  Run from project root: bash install_lane_deps.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
PYTHON="python3"

echo "══════════════════════════════════════════════════════"
echo "  Aerial Lane Detection — Dependency Installer"
echo "══════════════════════════════════════════════════════"
echo ""

# ── 1. Check virtual environment ──────────────────────────────────────────────
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ -d "venv" ]]; then
        echo "[1/7] Activating venv…"
        source venv/bin/activate
    else
        echo "[1/7] No venv found — installing to system Python (not recommended)."
    fi
else
    echo "[1/7] Virtual environment active: $VIRTUAL_ENV"
fi

echo ""

# ── 2. Core ML deps ───────────────────────────────────────────────────────────
echo "[2/7] Installing HuggingFace Transformers (SegFormer)…"
pip install --quiet "transformers>=4.37.0"
pip install --quiet "accelerate>=0.27.0"

echo ""

# ── 3. timm (Swin backbone for Mask2Former) ───────────────────────────────────
echo "[3/7] Installing timm (Swin Transformer backbone)…"
pip install --quiet "timm>=0.9.0"

echo ""

# ── 4. Albumentations (augmentation pipeline) ─────────────────────────────────
echo "[4/7] Installing Albumentations…"
pip install --quiet "albumentations>=1.4.0"

echo ""

# ── 5. scikit-image (skeletonize for lane centerlines) ───────────────────────
echo "[5/7] Installing scikit-image…"
pip install --quiet "scikit-image>=0.22.0"

echo ""

# ── 6. scikit-learn (metrics, Hungarian matching) ────────────────────────────
echo "[6/7] Installing scikit-learn…"
pip install --quiet "scikit-learn>=1.4.0"

echo ""

# ── 7. TensorBoard ───────────────────────────────────────────────────────────
echo "[7/7] Installing TensorBoard…"
pip install --quiet "tensorboard>=2.16.0"

echo ""

# ── Verify ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════"
echo "  Verifying installations…"
echo "══════════════════════════════════════════════════════"
$PYTHON -c "
import sys
failures = []

packages = [
    ('transformers',  'transformers'),
    ('timm',          'timm'),
    ('albumentations','albumentations'),
    ('skimage',       'scikit-image'),
    ('sklearn',       'scikit-learn'),
    ('torch.utils.tensorboard', 'tensorboard'),
    ('scipy',         'scipy'),
]

for import_name, pkg_name in packages:
    try:
        __import__(import_name)
        print(f'  ✓ {pkg_name}')
    except ImportError:
        print(f'  ✗ {pkg_name}  ← FAILED')
        failures.append(pkg_name)

if failures:
    print(f'\\n  FAILED: {failures}')
    print('  Try: pip install ' + ' '.join(failures))
    sys.exit(1)
else:
    print('\\n  All dependencies installed successfully! ✅')
"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Next steps:"
echo "  1. Download AerialLanes18 → see dataset/README_DOWNLOAD.md"
echo "  2. Verify dataset: python3 train.py --verify-only"
echo "  3. Train:          python3 train.py"
echo "  4. Infer on video: python3 infer_video.py --input videos/traffic.mp4 --yolo"
echo "══════════════════════════════════════════════════════"
