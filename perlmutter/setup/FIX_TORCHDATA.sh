#!/bin/bash
# Fix missing torchdata dependency in ML environment
# Error: ModuleNotFoundError: No module named 'torchdata'

echo "================================================"
echo "Installing torchdata (DGL dependency)"
echo "================================================"

module load conda

ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

echo "Activating ML environment: $ML_ENV"
conda activate "$ML_ENV"

echo ""
echo "Installing torchdata..."
pip install torchdata

echo ""
echo "Verifying installation..."
python -c "from torchdata.datapipes.iter import IterDataPipe; print('✓ torchdata installed successfully')"

echo ""
echo "Testing DGL import..."
python -c "import dgl; print('✓ DGL imports successfully')"

echo ""
echo "================================================"
echo "Fix applied successfully!"
echo "================================================"
echo ""
echo "You can now run the MVP pipeline:"
echo "  bash perlmutter/pipelines/run_mvp.sh"
