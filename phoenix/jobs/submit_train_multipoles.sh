#!/bin/bash
#SBATCH --job-name=TRAIN_MPOLE
#SBATCH --account=gts-jmcdaniel43-chemx
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=parmar@gatech.edu
#SBATCH --output=phoenix/logs/%j_train_multipoles.out
#SBATCH --error=phoenix/logs/%j_train_multipoles.err

# Train charge prediction model WITH multipole moments

echo "================================================"
echo "Training Model with Multipole Moments"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================================"

# Load modules
module load anaconda3/2022.05
module load cuda/11.8  # Adjust version as needed

# Activate conda environment
source activate mmomenta

# Create logs directory
mkdir -p phoenix/logs

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Dataset and output paths
DATASET="data/spice_mpfit.h5"
OUTPUT_DIR="runs/spice_multipoles"

# Model configuration
FEATURE_UNITS=198  # 117 baseline + 81 multipoles
DEPTH=4
WIDTH=128
ACTIVATION="relu"

# Training configuration
N_EPOCHS=1000
BATCH_SIZE=128
LEARNING_RATE=0.001
WEIGHT_DECAY=0.00001
EVAL_FREQUENCY=10

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Output: $OUTPUT_DIR"
echo "  Feature units: $FEATURE_UNITS"
echo "  Model: depth=$DEPTH, width=$WIDTH"
echo "  Training: epochs=$N_EPOCHS, batch=$BATCH_SIZE, lr=$LEARNING_RATE"
echo ""

# Run training
python scripts/train_spice.py \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --feature-units $FEATURE_UNITS \
    --depth $DEPTH \
    --width $WIDTH \
    --activation "$ACTIVATION" \
    --n-epochs $N_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --weight-decay $WEIGHT_DECAY \
    --eval-frequency $EVAL_FREQUENCY \
    --device cuda \
    --log-level INFO

EXIT_CODE=$?

echo ""
echo "================================================"
echo "Job completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training successful!"
    echo "  Checkpoints: $OUTPUT_DIR/checkpoints/"
    echo "  Best model: $OUTPUT_DIR/checkpoints/best_model.pt"

    # Print final results
    if [ -f "$OUTPUT_DIR/training_results.json" ]; then
        echo ""
        echo "Final Results:"
        python -c "
import json
with open('$OUTPUT_DIR/training_results.json') as f:
    results = json.load(f)
print(f'  Best Val RMSE: {results[\"results\"][\"best_val_rmse\"]:.5f}')
if 'test_metrics' in results['results']:
    print(f'  Test RMSE: {results[\"results\"][\"test_metrics\"][\"val_rmse\"]:.5f}')
    print(f'  Test MAE: {results[\"results\"][\"test_metrics\"][\"val_mae\"]:.5f}')
"
    fi
else
    echo "✗ Training failed!"
fi

exit $EXIT_CODE
