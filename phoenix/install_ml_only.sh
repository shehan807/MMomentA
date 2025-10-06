#!/bin/bash
# Install ML-only environment (no QM packages)
# This is for training models on pre-computed datasets

echo "================================================"
echo "MMomentA ML-Only Environment Setup"
echo "================================================"

module load anaconda3
module load cuda

echo ""
echo "Creating mmomenta-ml environment..."
conda env create -f phoenix/environment_ml.yml

echo ""
echo "Activating environment..."
conda activate mmomenta-ml

echo ""
echo "Installing MMomentA package..."
cd ~/MMomentA
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
