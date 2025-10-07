#!/bin/bash
# Verify MMomentA installation on Perlmutter

echo "================================================"
echo "MMomentA Installation Verification (Perlmutter)"
echo "================================================"
echo ""

# Load conda
module load conda

# Check if environments exist
DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

echo "Checking environments..."
echo ""

# Check data environment
if [ -d "$DATA_ENV" ]; then
    echo "✓ Data environment found: $DATA_ENV"

    conda activate "$DATA_ENV"

    echo "  Testing imports..."
    python << 'EOF'
import sys
success = True

try:
    import psi4
    print("    ✓ Psi4")
except ImportError as e:
    print(f"    ✗ Psi4: {e}")
    success = False

try:
    import openff.toolkit
    print("    ✓ OpenFF toolkit")
except ImportError as e:
    print(f"    ✗ OpenFF toolkit: {e}")
    success = False

try:
    import h5py
    print("    ✓ HDF5")
except ImportError as e:
    print(f"    ✗ HDF5: {e}")
    success = False

if not success:
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        echo "  ✓ All data environment packages working"
    else
        echo "  ✗ Some packages missing in data environment"
    fi
else
    echo "✗ Data environment not found at $DATA_ENV"
    echo "  Run: bash perlmutter/setup/MANUAL_SETUP_DATA.sh"
fi

echo ""

# Check ML environment
if [ -d "$ML_ENV" ]; then
    echo "✓ ML environment found: $ML_ENV"

    conda activate "$ML_ENV"

    echo "  Testing imports..."
    python << 'EOF'
import sys
success = True

try:
    import torch
    print(f"    ✓ PyTorch {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"      CUDA available: {cuda_available}")
    if cuda_available:
        print(f"      GPU count: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"    ✗ PyTorch: {e}")
    success = False

try:
    import dgl
    print(f"    ✓ DGL {dgl.__version__}")
except ImportError as e:
    print(f"    ✗ DGL: {e}")
    success = False

try:
    import openff.toolkit
    print("    ✓ OpenFF toolkit")
except ImportError as e:
    print(f"    ✗ OpenFF toolkit: {e}")
    success = False

try:
    import lovelyplots
    print("    ✓ lovelyplots")
except ImportError as e:
    print(f"    ✗ lovelyplots: {e}")
    success = False

if not success:
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        echo "  ✓ All ML environment packages working"
    else
        echo "  ✗ Some packages missing in ML environment"
    fi
else
    echo "✗ ML environment not found at $ML_ENV"
    echo "  Run: salloc -C gpu -q interactive -t 01:00:00 -A <account>"
    echo "       bash perlmutter/setup/MANUAL_SETUP_ML.sh"
fi

echo ""
echo "================================================"
echo "Verification Complete"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Submit a job: sbatch perlmutter/jobs/submit_mvp.sh"
echo "  2. Or use interactive: salloc -C gpu -q interactive -t 02:00:00 -A <account>"
echo ""
