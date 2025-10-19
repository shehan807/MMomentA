#!/bin/bash

# ZINC 100k ESP Validation - Parallel Execution
#
# This script:
# 1. Generates 20 chunk job scripts
# 2. Submits all chunk jobs in parallel
# 3. Submits merge job with dependency on all chunks
#
# Estimated wall-clock time: ~12-18 hours (vs 60+ hours sequential)

set -e  # Exit on error

# Configuration
MMOMENTA_DIR="/global/u1/p/parmar/MoML/MMomentA"
DATASET="${MMOMENTA_DIR}/data/zinc_100k_mpfit.h5"
MODEL_DIR="${MMOMENTA_DIR}/runs/zinc_100k_multipoles"
OUTPUT_DIR="${MMOMENTA_DIR}/figures/zinc_100k_esp_validation"
N_CHUNKS=20
TIME_LIMIT="36:00:00"
N_JOBS=32

cd "$MMOMENTA_DIR"

echo "========================================================================"
echo "ZINC 100k ESP Validation - Parallel Execution"
echo "========================================================================"
echo "Dataset:     $DATASET"
echo "Model:       $MODEL_DIR"
echo "Output:      $OUTPUT_DIR"
echo "Chunks:      $N_CHUNKS"
echo "Time limit:  $TIME_LIMIT per chunk"
echo "Workers:     $N_JOBS per chunk"
echo ""

# Check prerequisites
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"
    exit 1
fi

if [ ! -f "$MODEL_DIR/checkpoints/best_model.pt" ]; then
    echo "ERROR: Model checkpoint not found: $MODEL_DIR/checkpoints/best_model.pt"
    exit 1
fi

# Load conda for job generation
module load conda
DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
conda activate "$DATA_ENV"

# Step 1: Generate chunk job scripts
echo "Step 1: Generating chunk job scripts..."
python perlmutter/pipelines/generate_esp_jobs.py \
    --dataset "$DATASET" \
    --model-dir "$MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --n-chunks $N_CHUNKS \
    --time-limit "$TIME_LIMIT" \
    --n-jobs $N_JOBS

echo ""
echo "========================================================================"
echo "Step 2: Submitting chunk jobs..."
echo "========================================================================"
echo ""

# Submit all chunk jobs and collect job IDs
JOB_IDS=()
JOBS_DIR="${MMOMENTA_DIR}/perlmutter/pipelines/generated_jobs"

for chunk_id in $(seq 0 $((N_CHUNKS - 1))); do
    job_file="${JOBS_DIR}/run_zinc_100k_step6_chunk_$(printf "%02d" $chunk_id).slurm"

    if [ ! -f "$job_file" ]; then
        echo "WARNING: Job file not found: $job_file"
        continue
    fi

    # Submit job and capture job ID
    job_output=$(sbatch "$job_file")
    job_id=$(echo "$job_output" | awk '{print $NF}')

    JOB_IDS+=($job_id)

    echo "  Submitted chunk $(printf "%02d" $chunk_id): Job ID $job_id"
done

echo ""
echo "========================================================================"
echo "Step 3: Submitting merge job with dependencies..."
echo "========================================================================"
echo ""

# Create dependency string for SLURM
# Format: --dependency=afterok:job1:job2:job3:...
if [ ${#JOB_IDS[@]} -eq 0 ]; then
    echo "ERROR: No chunk jobs were submitted!"
    exit 1
fi

DEPENDENCY_STRING="afterok"
for job_id in "${JOB_IDS[@]}"; do
    DEPENDENCY_STRING="${DEPENDENCY_STRING}:${job_id}"
done

echo "Chunk jobs: ${#JOB_IDS[@]}"
echo "Dependency: $DEPENDENCY_STRING"
echo ""

# Submit merge job
merge_job_output=$(sbatch --dependency="$DEPENDENCY_STRING" \
    "${MMOMENTA_DIR}/perlmutter/pipelines/run_zinc_100k_step6_merge.slurm")
merge_job_id=$(echo "$merge_job_output" | awk '{print $NF}')

echo "Merge job submitted: Job ID $merge_job_id"
echo ""

echo "========================================================================"
echo "Parallel ESP Validation Jobs Submitted!"
echo "========================================================================"
echo ""
echo "Chunk jobs submitted: ${#JOB_IDS[@]}"
echo "  Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Merge job submitted: $merge_job_id"
echo "  Dependencies: All ${#JOB_IDS[@]} chunk jobs must complete"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  squeue -j ${JOB_IDS[0]},${JOB_IDS[1]},...,$merge_job_id"
echo ""
echo "Check logs:"
echo "  tail -f logs/zinc_100k_esp_chunk_*_*.out"
echo ""
echo "Expected completion: ~12-18 hours"
echo "Final results will be in: $OUTPUT_DIR/"
echo ""
