#!/bin/bash
#SBATCH --job-name=ABLATION
#SBATCH --account=gts-jmcdaniel43-chemx
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=parmar@gatech.edu
#SBATCH --output=phoenix/logs/%j_ablation.out
#SBATCH --error=phoenix/logs/%j_ablation.err

# Ablation study: Compare baseline vs multipole models
# Multiple configurations, multiple runs for statistical significance

echo "================================================"
echo "Ablation Study: Baseline vs Multipoles"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================================"

# Load modules
module load anaconda3/2022.05
module load cuda/11.8

# Activate conda environment
source activate mmomenta

# Create logs and output directories
mkdir -p phoenix/logs
mkdir -p runs/ablation

# Dataset
DATASET="data/spice_mpfit.h5"

echo ""
echo "Running ablation study with 3 configurations × 3 runs each"
echo "This will take approximately 24-48 hours"
echo ""

# Configuration 1: Baseline (no multipoles)
echo "================================================"
echo "Configuration 1: Baseline (117 features)"
echo "================================================"

for RUN in 1 2 3; do
    echo ""
    echo "Run $RUN/3..."

    python scripts/train_spice.py \
        --dataset "$DATASET" \
        --output-dir "runs/ablation/baseline_run${RUN}" \
        --feature-units 117 \
        --no-multipoles \
        --depth 4 \
        --width 128 \
        --n-epochs 500 \
        --batch-size 128 \
        --device cuda \
        --log-level WARNING

    echo "✓ Baseline run $RUN complete"
done

# Configuration 2: With multipoles
echo ""
echo "================================================"
echo "Configuration 2: With Multipoles (198 features)"
echo "================================================"

for RUN in 1 2 3; do
    echo ""
    echo "Run $RUN/3..."

    python scripts/train_spice.py \
        --dataset "$DATASET" \
        --output-dir "runs/ablation/multipoles_run${RUN}" \
        --feature-units 198 \
        --depth 4 \
        --width 128 \
        --n-epochs 500 \
        --batch-size 128 \
        --device cuda \
        --log-level WARNING

    echo "✓ Multipoles run $RUN complete"
done

# Configuration 3: Deeper baseline (for fair comparison)
echo ""
echo "================================================"
echo "Configuration 3: Deeper Baseline (6 layers)"
echo "================================================"

for RUN in 1 2 3; do
    echo ""
    echo "Run $RUN/3..."

    python scripts/train_spice.py \
        --dataset "$DATASET" \
        --output-dir "runs/ablation/deeper_baseline_run${RUN}" \
        --feature-units 117 \
        --no-multipoles \
        --depth 6 \
        --width 128 \
        --n-epochs 500 \
        --batch-size 128 \
        --device cuda \
        --log-level WARNING

    echo "✓ Deeper baseline run $RUN complete"
done

# Analyze results
echo ""
echo "================================================"
echo "Analyzing Results"
echo "================================================"

python << 'EOF'
import json
import numpy as np
from pathlib import Path

configs = {
    'baseline': 'runs/ablation/baseline_run',
    'multipoles': 'runs/ablation/multipoles_run',
    'deeper_baseline': 'runs/ablation/deeper_baseline_run'
}

print("\nAblation Study Results")
print("=" * 60)

for config_name, base_path in configs.items():
    test_rmses = []

    for run in [1, 2, 3]:
        results_file = Path(f"{base_path}{run}/training_results.json")
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
                if 'test_metrics' in results['results']:
                    test_rmses.append(results['results']['test_metrics']['val_rmse'])

    if test_rmses:
        print(f"\n{config_name.upper()}:")
        print(f"  Test RMSE: {np.mean(test_rmses):.5f} ± {np.std(test_rmses):.5f}")
        print(f"  Individual runs: {[f'{r:.5f}' for r in test_rmses]}")

print("\n" + "=" * 60)
EOF

echo ""
echo "================================================"
echo "Ablation Study Complete: $(date)"
echo "================================================"
echo ""
echo "Results saved in runs/ablation/"
