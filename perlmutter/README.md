# Perlmutter Setup Guide

Quick setup guide for running MMomentA on NERSC Perlmutter.

## Key Differences from Phoenix

1. **Module system**: Use `module load conda` instead of `module load anaconda3`
2. **Environment location**: Install to `$SCRATCH/conda-envs/` instead of default location
3. **CUDA version**: Perlmutter uses CUDA 12.x (newer than Phoenix)
4. **GPU access**: Requires allocation to GPU partition

## Quick Setup

### 1. Setup Data Environment (Login Node)

```bash
# On Perlmutter login node
cd /path/to/MMomentA
bash MANUAL_SETUP_PERLMUTTER_DATA.sh
```

This creates: `$SCRATCH/conda-envs/mmomenta-data`

### 2. Setup ML Environment (GPU Node)

```bash
# Request GPU node for interactive session
salloc -C gpu -q interactive -t 01:00:00 -A <your_account>

# Once on GPU node
cd /path/to/MMomentA
bash MANUAL_SETUP_PERLMUTTER_ML.sh
```

This creates: `$SCRATCH/conda-envs/mmomenta-ml`

### 3. Add Aliases to ~/.bashrc

```bash
cat >> ~/.bashrc << 'EOF'

# MMomentA environments on Perlmutter
export MMOMENTA_DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
export MMOMENTA_ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

alias mmdata='module load conda && conda activate $MMOMENTA_DATA_ENV'
alias mmml='module load conda && conda activate $MMOMENTA_ML_ENV'
EOF

source ~/.bashrc
```

## Usage

### Interactive Sessions

**Data processing (CPU):**
```bash
# Login node is fine for dataset creation
mmdata
python scripts/prepare_dataset.py --input data/test_smiles.pkl --output data/test_mpfit.h5
```

**Training (GPU):**
```bash
# Request GPU node
salloc -C gpu -q interactive -t 02:00:00 -A <your_account>

# Activate ML environment
mmml

# Train model
python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/test_multipoles \
    --device cuda
```

### Batch Jobs

**Data Processing Job** (`job_data.sh`):
```bash
#!/bin/bash
#SBATCH -A <your_account>
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

module load conda
conda activate $SCRATCH/conda-envs/mmomenta-data

cd $SLURM_SUBMIT_DIR

python scripts/prepare_dataset.py \
    --input data/test_smiles.pkl \
    --output data/test_mpfit.h5 \
    --n-jobs 32
```

**Training Job** (`job_train.sh`):
```bash
#!/bin/bash
#SBATCH -A <your_account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1

module load conda
conda activate $SCRATCH/conda-envs/mmomenta-ml

cd $SLURM_SUBMIT_DIR

python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/test_multipoles \
    --device cuda \
    --n-epochs 200
```

Submit:
```bash
sbatch job_data.sh
sbatch job_train.sh
```

## Modified Pipeline Scripts

The pipeline scripts (`run_mvp.sh`, `run_spice.sh`, etc.) need modifications for Perlmutter:

### Update Environment Activation

**Phoenix version:**
```bash
module load anaconda3
conda activate mmomenta-data
```

**Perlmutter version:**
```bash
module load conda
conda activate $SCRATCH/conda-envs/mmomenta-data
```

### Update CUDA Module

**Phoenix version:**
```bash
module load cuda
```

**Perlmutter version:**
```bash
# CUDA is already available on GPU nodes, no module load needed
# Or if needed:
module load cudatoolkit
```

## Complete Perlmutter Pipeline Example

```bash
# 1. Setup (one-time)
bash MANUAL_SETUP_PERLMUTTER_DATA.sh
salloc -C gpu -q interactive -t 01:00:00 -A <account>
bash MANUAL_SETUP_PERLMUTTER_ML.sh
exit

# 2. Create dataset (login node or CPU job)
module load conda
conda activate $SCRATCH/conda-envs/mmomenta-data
python scripts/create_test_smiles.py
python scripts/prepare_dataset.py \
    --input data/test_smiles.pkl \
    --output data/test_mpfit.h5 \
    --n-jobs 16

# 3. Train model (GPU job)
salloc -C gpu -q interactive -t 02:00:00 -A <account>
module load conda
conda activate $SCRATCH/conda-envs/mmomenta-ml
python scripts/train_spice.py \
    --dataset data/test_mpfit.h5 \
    --output-dir runs/test_multipoles \
    --device cuda \
    --n-epochs 200
exit

# 4. ESP validation (GPU job for MPFIT-GNN, then CPU for QM)
salloc -C gpu -q interactive -t 01:00:00 -A <account>
module load conda
conda activate $SCRATCH/conda-envs/mmomenta-ml
python scripts/validate_esp.py \
    --model runs/test_multipoles/best_model.pt \
    --molecules data/test_smiles.pkl \
    --esp-dir esp_comparison \
    --device cuda
exit

module load conda
conda activate $SCRATCH/conda-envs/mmomenta-data
python scripts/compare_all_methods.py \
    --molecules data/test_smiles.pkl \
    --output-dir esp_comparison \
    --mpfit-gnn-results esp_comparison/mpfit_gnn_results.json
```

## Perlmutter-Specific Notes

### Storage
- **$HOME**: 40 GB quota, backed up, slow I/O
- **$SCRATCH**: Large quota, NOT backed up, fast I/O ← Use for conda envs and data
- **$CFS**: Community File System, good for shared data

**Recommendation**: Keep everything in `$SCRATCH` except final results

### GPU Partitions
- **Regular**: Production jobs, 12-hour limit
- **Debug**: Quick testing, 30-minute limit
- **Shared**: Share GPU with other jobs

### CUDA
Perlmutter GPU nodes have CUDA 12.x pre-installed. PyTorch installation automatically detects this.

### Performance Tips
1. Use `--cpus-per-task=32` for parallel data processing
2. Use `--gpus-per-task=1` for single-GPU training
3. For multi-GPU: `--gpus-per-task=4` (Perlmutter has 4 A100 GPUs per node)

## Troubleshooting

### "conda: command not found"
```bash
module load conda
```

### "No space left on device" during conda install
```bash
# Check quota
showquota

# Clean conda cache
conda clean --all

# Use SCRATCH for environments (done by setup scripts)
```

### PyTorch can't find CUDA
```bash
# Check if on GPU node
scontrol show job $SLURM_JOB_ID | grep GPU

# Verify CUDA
nvidia-smi

# Reinstall PyTorch with correct CUDA version
conda activate $SCRATCH/conda-envs/mmomenta-ml
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Environment path too long
```bash
# Use shorter path names
ENV_PATH="$SCRATCH/mm-data"  # Instead of conda-envs/mmomenta-data
```

## Resource Allocation

Typical resource needs:

| Task | Partition | Nodes | Time | Memory |
|------|-----------|-------|------|--------|
| Dataset creation (150 mol) | cpu | 1 | 30 min | 16 GB |
| MPFIT charges (150 mol) | cpu | 1 | 1 hr | 32 GB |
| Training (200 epochs) | gpu | 1 | 1 hr | 32 GB |
| ESP validation (QM) | cpu | 1 | 2 hr | 32 GB |
| ESP validation (ML) | gpu | 1 | 10 min | 16 GB |

## Quick Commands

```bash
# Check job status
squeue -u $USER

# Cancel job
scancel <job_id>

# Check GPU usage
nvidia-smi

# Monitor job output
tail -f slurm-<job_id>.out

# Check account allocation
sshare -U $USER
```

## References

- Perlmutter User Guide: https://docs.nersc.gov/systems/perlmutter/
- Conda on Perlmutter: https://docs.nersc.gov/development/languages/python/
- GPU Jobs: https://docs.nersc.gov/jobs/gpu/
