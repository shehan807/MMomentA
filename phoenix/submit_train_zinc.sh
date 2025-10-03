#!/bin/bash
#SBATCH --job-name=TRAIN_ZINC
#SBATCH --account=gts-jmcdaniel43-chemx
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=parmar@gatech.edu
#SBATCH --output=phoenix/logs/%j_train_zinc.out
#SBATCH --error=phoenix/logs/%j_train_zinc.err

# Transfer learning: Fine-tune SPICE-trained model on ZINC dataset

echo "================================================"
echo "ZINC Transfer Learning"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================================"

# Load modules
module load anaconda3/2022.05
module load cuda/11.8

# Activate conda environment
source activate mmomenta

# Create logs directory
mkdir -p phoenix/logs

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Paths
DATASET="data/zinc_mpfit.h5"
PRETRAINED_MODEL="runs/spice_multipoles/checkpoints/best_model.pt"
OUTPUT_DIR="runs/zinc_finetune"

# Training configuration
N_EPOCHS=500
BATCH_SIZE=128
LEARNING_RATE=0.0001  # Lower learning rate for fine-tuning

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Pretrained model: $PRETRAINED_MODEL"
echo "  Output: $OUTPUT_DIR"
echo "  Fine-tuning epochs: $N_EPOCHS"
echo "  Learning rate: $LEARNING_RATE (lower for fine-tuning)"
echo ""

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "✗ Error: Pretrained model not found at $PRETRAINED_MODEL"
    echo "  Please train SPICE model first using submit_train_multipoles.sh"
    exit 1
fi

# Run transfer learning
python scripts/train_zinc.py \
    --dataset "$DATASET" \
    --pretrained "$PRETRAINED_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --feature-units 198 \
    --n-epochs $N_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --device cuda \
    --log-level INFO

EXIT_CODE=$?

echo ""
echo "================================================"
echo "Job completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Transfer learning successful!"

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
    print('')
    print('Transfer learning demonstrates model generalization to drug-like molecules')
"
    fi
else
    echo "✗ Transfer learning failed!"
fi

exit $EXIT_CODE
