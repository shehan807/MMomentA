#!/bin/bash
# Download QM9 dataset to $SCRATCH on Perlmutter using PyTorch Geometric

echo "================================================"
echo "Downloading QM9 Dataset to \$SCRATCH"
echo "================================================"
echo ""
echo "Dataset details:"
echo "  - 133,885 organic molecules (up to 9 heavy atoms: C, N, O, F)"
echo "  - DFT-level quantum properties (B3LYP/6-31G(2df,p))"
echo "  - Size: ~4 GB (raw + processed)"
echo "  - Source: PyTorch Geometric / MoleculeNet"
echo ""

# Check if SCRATCH is set
if [ -z "$SCRATCH" ]; then
    echo "ERROR: \$SCRATCH is not set"
    echo "Are you on a Perlmutter login/compute node?"
    exit 1
fi

# Check if already exists
if [ -d "$SCRATCH/qm9_raw" ]; then
    echo "✓ QM9 dataset already exists at $SCRATCH/qm9_raw"
    du -sh "$SCRATCH/qm9_raw"
    echo ""
    echo "To re-download, delete the directory first:"
    echo "  rm -rf $SCRATCH/qm9_raw"
    exit 0
fi

# Load conda and activate ML environment
module load conda
ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

echo "Downloading to: $SCRATCH/qm9_raw"
echo "This will take 5-15 minutes depending on network speed..."
echo ""

conda activate "$ML_ENV"

# Download QM9 using PyTorch Geometric (just trigger the download)
python << 'EOF'
from torch_geometric.datasets import QM9
import os

scratch_dir = os.environ.get('SCRATCH', '.')
qm9_root = f'{scratch_dir}/qm9_raw'

print(f"Downloading QM9 dataset to {qm9_root}...")
dataset = QM9(root=qm9_root)
print(f"✓ QM9 dataset downloaded: {len(dataset)} molecules")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✓ Download complete!"
    echo "================================================"
    echo ""
    echo "Directory info:"
    du -sh "$SCRATCH/qm9_raw"
    echo ""
    echo "Location: $SCRATCH/qm9_raw"
    echo ""
    echo "Next steps:"
    echo "  - Run QM9 pipeline: bash perlmutter/pipelines/run_qm9.sh"
    echo "  - Or submit SLURM job: sbatch perlmutter/pipelines/run_qm9_10k.slurm"
    echo ""
else
    echo ""
    echo "✗ Download failed"
    echo "Check your network connection and try again"
    exit 1
fi
