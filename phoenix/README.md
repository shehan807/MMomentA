# Phoenix Cluster Submission Scripts

SLURM batch scripts for running MMomentA jobs on the Phoenix cluster at Georgia Tech.

## Setup

### 1. Create Conda Environment

```bash
# On Phoenix login node
cd MMomentA/phoenix
bash setup_environment.sh
```

This will create a conda environment with all dependencies.

### 2. Verify Installation

```bash
conda activate mmomenta
bash test_installation.sh
```

## Available Scripts

### Dataset Preparation

- `submit_prepare_spice.sh` - Prepare SPICE dataset
- `submit_prepare_zinc.sh` - Prepare ZINC dataset

### Training

- `submit_train_baseline.sh` - Train baseline model (no multipoles)
- `submit_train_multipoles.sh` - Train with multipole moments
- `submit_train_zinc.sh` - Transfer learning on ZINC

### Comparison Studies

- `submit_comparison.sh` - Compare MPFIT vs RESP vs AM1-BCC
- `submit_ablation.sh` - Ablation study (baseline vs multipoles)

## Usage

### Quick Start

```bash
# 1. Prepare dataset
sbatch phoenix/submit_prepare_spice.sh

# 2. Train model
sbatch phoenix/submit_train_multipoles.sh

# 3. Check status
squeue -u $USER

# 4. View output
tail -f <job_id>.out
```

### Customizing Jobs

Edit the SBATCH parameters in each script:
- `--time` - Wall time limit
- `--mem` - Memory allocation
- `--gres=gpu:V100:1` - GPU type/count
- `-q` - Queue (inferno for GPU)

### Example: Custom Training

```bash
# Copy template
cp phoenix/submit_train_multipoles.sh phoenix/my_custom_job.sh

# Edit parameters
vim phoenix/my_custom_job.sh

# Submit
sbatch phoenix/my_custom_job.sh
```

## Resource Recommendations

### Dataset Preparation

- **SPICE (1000 molecules):**
  - Time: 6-12 hours
  - CPUs: 16-32 (parallel)
  - Memory: 64G
  - No GPU needed

- **ZINC (500 molecules):**
  - Time: 4-8 hours
  - CPUs: 16-32
  - Memory: 64G
  - No GPU needed

### Model Training

- **Baseline/Multipoles (1000 epochs):**
  - Time: 4-8 hours
  - GPU: 1x V100
  - Memory: 32G
  - Batch size: 128

- **Ablation Study (3 configs × 3 runs):**
  - Time: 24-48 hours
  - GPU: 1x V100
  - Memory: 32G

## Monitoring Jobs

```bash
# Check job status
squeue -u parmar@gatech.edu

# Cancel job
scancel <job_id>

# Check job details
scontrol show job <job_id>

# View output live
tail -f <job_id>.out

# Check GPU usage (on compute node)
nvidia-smi
```

## Troubleshooting

### GPU not available
```bash
# Check GPU queue
sinfo -p inferno

# Request specific GPU
#SBATCH --gres=gpu:A100:1  # For A100 instead of V100
```

### Out of memory
```bash
# Reduce batch size in script
--batch-size 64  # instead of 128

# Or increase memory
#SBATCH --mem=64G
```

### Job timeout
```bash
# Increase wall time
#SBATCH --time=24:00:00

# Or enable checkpointing to resume
--checkpoint-frequency 100
```

## File Organization

```
phoenix/
├── README.md                      # This file
├── setup_environment.sh           # Conda environment setup
├── test_installation.sh           # Verify installation
├── environment.yml                # Conda dependencies
├── requirements.txt               # Pip dependencies
├── submit_prepare_spice.sh        # Dataset prep scripts
├── submit_prepare_zinc.sh
├── submit_train_baseline.sh       # Training scripts
├── submit_train_multipoles.sh
├── submit_train_zinc.sh
├── submit_comparison.sh           # Analysis scripts
├── submit_ablation.sh
└── logs/                          # Job outputs (auto-created)
```

## Tips

1. **Use scratch space for large files:**
   ```bash
   # In your scripts
   SCRATCH=/scratch/$USER
   cp dataset.h5 $SCRATCH/
   cd $SCRATCH
   ```

2. **Save checkpoints frequently:**
   ```bash
   --save-frequency 50  # Every 50 epochs
   ```

3. **Monitor GPU usage:**
   ```bash
   # In your script
   watch -n 1 nvidia-smi
   ```

4. **Use job arrays for multiple runs:**
   ```bash
   #SBATCH --array=1-5  # 5 parallel jobs
   ```

## Contact

For Phoenix cluster issues:
- Help desk: pace-support@oit.gatech.edu
- Documentation: https://docs.pace.gatech.edu/

For MMomentA issues:
- Shehan Parmar: parmar@gatech.edu
