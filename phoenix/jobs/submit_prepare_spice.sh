#!/bin/bash
#SBATCH --job-name=SPICE_PREP
#SBATCH --account=gts-jmcdaniel43-chemx
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=parmar@gatech.edu
#SBATCH --output=phoenix/logs/%j_spice_prep.out
#SBATCH --error=phoenix/logs/%j_spice_prep.err

# SPICE dataset preparation with MPFIT calculations
# This job does NOT need GPU - uses CPUs for parallel Psi4 calculations

echo "================================================"
echo "SPICE Dataset Preparation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================================"

# Load modules
module load anaconda3/2022.05

# Activate conda environment
source activate mmomenta

# Create logs directory
mkdir -p phoenix/logs

# Set environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Dataset parameters
INPUT_FILE="/path/to/spice.oeb"  # UPDATE THIS PATH
OUTPUT_FILE="data/spice_mpfit.h5"
MAX_MOLECULES=1000
N_JOBS=32  # Match --ntasks-per-node

# QM settings
QM_METHOD="hf"
QM_BASIS="6-31G*"
MULTIPOLE_LIMIT=8

echo ""
echo "Configuration:"
echo "  Input: $INPUT_FILE"
echo "  Output: $OUTPUT_FILE"
echo "  Max molecules: $MAX_MOLECULES"
echo "  Parallel jobs: $N_JOBS"
echo "  QM method: $QM_METHOD/$QM_BASIS"
echo "  Multipole limit: $MULTIPOLE_LIMIT"
echo ""

# Run dataset preparation
python scripts/prepare_dataset.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --dataset-name SPICE \
    --max-molecules $MAX_MOLECULES \
    --method "$QM_METHOD" \
    --basis "$QM_BASIS" \
    --limit $MULTIPOLE_LIMIT \
    --split-strategy scaffold \
    --n-jobs $N_JOBS \
    --log-level INFO

EXIT_CODE=$?

echo ""
echo "================================================"
echo "Job completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Dataset preparation successful!"
    echo "  Output saved to: $OUTPUT_FILE"

    # Print dataset stats
    python -c "
from MMomentA.data import load_dataset
mol_data, metadata = load_dataset('$OUTPUT_FILE')
print(f'Dataset: {len(mol_data)} molecules')
print(f'Train: {len(metadata[\"splits\"][\"train\"])}')
print(f'Val: {len(metadata[\"splits\"][\"val\"])}')
print(f'Test: {len(metadata[\"splits\"][\"test\"])}')
"
else
    echo "✗ Dataset preparation failed!"
fi

exit $EXIT_CODE
