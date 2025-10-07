#!/bin/bash
#SBATCH --job-name=COMPARE
#SBATCH --account=gts-jmcdaniel43-chemx
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=parmar@gatech.edu
#SBATCH --output=phoenix/logs/%j_comparison.out
#SBATCH --error=phoenix/logs/%j_comparison.err

# Compare MPFIT vs RESP vs AM1-BCC charge methods
# Does NOT need GPU

echo "================================================"
echo "Charge Method Comparison"
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

# Input molecules
INPUT_FILE="/path/to/test_molecules.pkl"  # UPDATE THIS PATH
OUTPUT_FILE="results/charge_comparison.json"
MAX_MOLECULES=50
N_JOBS=16

# QM settings
QM_METHOD="hf"
QM_BASIS="6-31G*"

echo ""
echo "Configuration:"
echo "  Input: $INPUT_FILE"
echo "  Output: $OUTPUT_FILE"
echo "  Max molecules: $MAX_MOLECULES"
echo "  Parallel jobs: $N_JOBS"
echo "  QM method: $QM_METHOD/$QM_BASIS"
echo ""

# Run comparison
python scripts/compare_charge_methods.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --max-molecules $MAX_MOLECULES \
    --method "$QM_METHOD" \
    --basis "$QM_BASIS" \
    --n-jobs $N_JOBS \
    --log-level INFO

EXIT_CODE=$?

echo ""
echo "================================================"
echo "Job completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Comparison successful!"
    echo "  Results: $OUTPUT_FILE"

    # Print summary
    python -c "
import json
with open('$OUTPUT_FILE') as f:
    results = json.load(f)

n_success = sum(1 for r in results['results'] if r.get('success', False))
print(f'Processed {n_success}/{len(results[\"results\"])} molecules successfully')
"
else
    echo "✗ Comparison failed!"
fi

exit $EXIT_CODE
