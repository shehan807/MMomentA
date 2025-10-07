#!/bin/bash
# Install ML-only environment (no QM packages)
# This is for training models on pre-computed datasets

echo "================================================"
echo "MMomentA ML-Only Environment Setup"
echo "================================================"

MMOMENTA_DIR="/storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML"

module load anaconda3
module load cuda

# Remove existing environment
echo ""
echo "Removing any existing mmomenta-ml environment..."
conda env remove -n mmomenta-ml -y 2>/dev/null || true

echo ""
echo "Creating mmomenta-ml environment..."
cd "$MMOMENTA_DIR"
conda env create -f phoenix/environment_ml.yml
if [ $? -ne 0 ]; then
    echo "✗ Failed to create environment"
    exit 1
fi

echo ""
echo "Activating environment..."
source activate mmomenta-ml

echo ""
echo "Installing MMomentA package..."
cd "$MMOMENTA_DIR/MMomentA"
pip install -e .

echo ""
echo "================================================"
echo "Testing ML Environment"
echo "================================================"

python << 'EOF'
import sys

try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'✗ PyTorch: {e}')
    sys.exit(1)

try:
    import dgl
    print(f'✓ DGL: {dgl.__version__}')
except ImportError as e:
    print(f'✗ DGL: {e}')
    sys.exit(1)

try:
    import numpy as np
    print(f'✓ NumPy: {np.__version__}')
except ImportError as e:
    print(f'✗ NumPy: {e}')
    sys.exit(1)

try:
    from mmomenta.models import ChargeModel
    from mmomenta.training import Trainer
    print('✓ MMomentA ML modules')
except ImportError as e:
    print(f'✗ MMomentA ML: {e}')
    sys.exit(1)

print('\n✓ ML environment ready for training!')
print('\nYou can now:')
print('  - Load pre-computed HDF5 datasets')
print('  - Train ML models')
print('  - Run predictions')
print('\nNote: This environment cannot run QM calculations.')
print('Use mmomenta-data environment for dataset generation.')
EOF

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To use this environment:"
echo "  module load anaconda3 cuda"
echo "  conda activate mmomenta-ml"
echo ""
echo "To train models:"
echo "  python scripts/train_spice.py --dataset data/prepared.h5 ..."
echo ""
