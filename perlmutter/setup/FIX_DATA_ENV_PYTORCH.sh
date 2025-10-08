#!/bin/bash
# Add PyTorch and DGL to existing data environment for ESP validation

echo "================================================"
echo "Adding PyTorch/DGL to data environment"
echo "================================================"

module load conda

DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"

echo "Activating data environment: $DATA_ENV"
conda activate "$DATA_ENV"

echo ""
echo "Removing conda PyTorch (if present) to avoid symbol conflicts..."
conda remove pytorch --force -y 2>/dev/null || echo "  (no conda pytorch found)"

echo ""
echo "Installing PyTorch (CPU-only) via pip..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Installing DGL (CPU-only)..."
conda install -c dglteam "dgl<2.0" -y

echo ""
echo "Installing lovelyplots (publication figures)..."
pip install lovelyplots

echo ""
echo "Verifying installation..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CPU)')"
python -c "import dgl; print(f'✓ DGL {dgl.__version__} (CPU)')"
python -c "import lovelyplots; print(f'✓ lovelyplots installed')"

echo ""
echo "================================================"
echo "Fix applied successfully!"
echo "================================================"
echo ""
echo "The data environment can now run ESP validation:"
echo "  bash perlmutter/pipelines/run_mvp.sh"
