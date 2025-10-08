#!/bin/bash
# Fix DGL version to avoid torchdata dependency
# DGL 2.0+ requires torchdata (deprecated), but DGL 1.1.x works fine without it

echo "================================================"
echo "Fixing DGL version (remove torchdata dependency)"
echo "================================================"

module load conda

ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

echo "Activating ML environment: $ML_ENV"
conda activate "$ML_ENV"

echo ""
echo "Current DGL version:"
python -c "import dgl; print(f'DGL {dgl.__version__}')" 2>/dev/null || echo "DGL not installed or broken"

echo ""
echo "Removing DGL 2.x..."
conda remove dgl --force -y

echo ""
echo "Installing DGL 1.1.x (no torchdata needed)..."
conda install -c dglteam/label/cu121 "dgl<2.0" -y

echo ""
echo "New DGL version:"
python -c "import dgl; print(f'✓ DGL {dgl.__version__} installed')"

echo ""
echo "Verifying PyTorch still works..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CUDA {torch.version.cuda})')"

echo ""
echo "Testing DGL import..."
python -c "import dgl; import torch; print('✓ DGL and PyTorch work together!')"

echo ""
echo "================================================"
echo "Fix applied successfully!"
echo "================================================"
echo ""
echo "You can now run the MVP pipeline:"
echo "  bash perlmutter/pipelines/run_mvp.sh"
