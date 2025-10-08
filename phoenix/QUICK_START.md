# Phoenix Cluster Quick Start Guide

## Initial Setup (One-time)

```bash
# 1. SSH to Phoenix
ssh parmar@login-phoenix.pace.gatech.edu

# 2. Clone/copy MMomentA repository
cd $HOME
git clone <repository_url> MMomentA  # or use rsync/scp

# 3. Setup environment
cd MMomentA/phoenix
bash setup_environment.sh

# 4. Test installation
bash test_installation.sh
```

## Typical Workflow

### Step 1: Prepare Dataset

```bash
# Edit the input path in submit_prepare_spice.sh
vim phoenix/submit_prepare_spice.sh
# Change: INPUT_FILE="/path/to/spice.oeb"

# Submit job
sbatch phoenix/submit_prepare_spice.sh

# Monitor progress
squeue -u $USER
tail -f phoenix/logs/<job_id>_spice_prep.out
```

**Expected time:** 6-12 hours for 1000 molecules

### Step 2: Train Baseline Model

```bash
# Submit baseline training
sbatch phoenix/submit_train_baseline.sh

# Check GPU queue
squeue -p inferno

# Monitor training
tail -f phoenix/logs/<job_id>_train_baseline.out
```

**Expected time:** 4-8 hours for 1000 epochs

### Step 3: Train with Multipoles

```bash
# Submit multipole training
sbatch phoenix/submit_train_multipoles.sh

# Monitor
tail -f phoenix/logs/<job_id>_train_multipoles.out
```

**Expected time:** 4-8 hours for 1000 epochs

### Step 4: Compare Results

Models will be saved in:
- `runs/spice_baseline/checkpoints/best_model.pt`
- `runs/spice_multipoles/checkpoints/best_model.pt`

Check results:
```bash
# View baseline results
cat runs/spice_baseline/training_results.json | python -m json.tool

# View multipole results
cat runs/spice_multipoles/training_results.json | python -m json.tool
```

## Common Commands

### Check Job Status

```bash
# Your jobs
squeue -u parmar@gatech.edu

# Specific job
scontrol show job <job_id>

# GPU nodes
sinfo -p inferno
```

### Cancel Jobs

```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u parmar@gatech.edu
```

### View Outputs

```bash
# Live monitoring
tail -f phoenix/logs/<job_id>.out

# Check errors
tail -f phoenix/logs/<job_id>.err

# Full output
less phoenix/logs/<job_id>.out
```

### File Transfer

```bash
# From local to Phoenix
scp local_file.h5 parmar@login-phoenix.pace.gatech.edu:~/MMomentA/data/

# From Phoenix to local
scp parmar@login-phoenix.pace.gatech.edu:~/MMomentA/runs/spice_multipoles/checkpoints/best_model.pt .

# Recursive directory
scp -r runs/ parmar@login-phoenix.pace.gatech.edu:~/MMomentA/
```

## Resource Usage

### Dataset Preparation
- **Queue:** Default (CPU)
- **Nodes:** 1
- **CPUs:** 32
- **Memory:** 64G
- **Time:** 12 hours
- **GPU:** Not needed

### Model Training
- **Queue:** inferno (GPU)
- **Nodes:** 1
- **GPU:** 1x V100
- **Memory:** 32G
- **Time:** 12 hours
- **Batch size:** 128

### Ablation Study
- **Queue:** inferno (GPU)
- **Nodes:** 1
- **GPU:** 1x V100
- **Memory:** 32G
- **Time:** 48 hours
- **Runs:** 9 total (3 configs × 3 runs)

## Troubleshooting

### Job won't start
```bash
# Check why job is pending
scontrol show job <job_id> | grep Reason

# Common reasons:
# - Resources: Waiting for GPU
# - Priority: Other jobs ahead in queue
# - QOSMaxGRESPerUser: Too many GPUs requested
```

### Out of memory
```bash
# Edit SBATCH parameters in script
#SBATCH --mem=64G  # Increase from 32G

# Or reduce batch size
--batch-size 64  # instead of 128
```

### GPU not available
```bash
# Check GPU availability
sinfo -p inferno -o "%n %G %C"

# Request specific GPU
#SBATCH --gres=gpu:A100:1  # if V100 unavailable
```

### Conda environment issues
```bash
# Reload environment
conda deactivate
conda activate mmomenta

# Reinstall if needed
cd MMomentA/phoenix
bash setup_environment.sh
```

## Advanced Usage

### Interactive Job (for debugging)

```bash
# Request interactive GPU node
salloc --account=gts-jmcdaniel43-chemx --gres=gpu:V100:1 --mem=32G -qinferno --time=2:00:00

# Once allocated
module load anaconda3/2022.05
conda activate mmomenta

# Run commands interactively
python scripts/train_spice.py --dataset data/spice_mpfit.h5 ...

# Exit when done
exit
```

### Job Arrays (multiple runs)

```bash
# Edit script to add array directive
#SBATCH --array=1-5

# Use $SLURM_ARRAY_TASK_ID in script
--output-dir "runs/run_${SLURM_ARRAY_TASK_ID}"
```

### Monitor GPU Usage

```bash
# On compute node (during interactive session or in script)
watch -n 1 nvidia-smi

# Or add to script
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 10
```

## Data Locations

### Input Data
```
/path/to/spice.oeb          # SPICE molecules
/path/to/zinc_molecules.pkl # ZINC molecules
```

### Outputs
```
data/
├── spice_mpfit.h5          # Prepared SPICE dataset
└── zinc_mpfit.h5           # Prepared ZINC dataset

runs/
├── spice_baseline/         # Baseline model
├── spice_multipoles/       # Multipole model
├── zinc_finetune/          # Transfer learning
└── ablation/               # Ablation study results

phoenix/logs/               # Job output logs
```

## Getting Help

### Phoenix Support
- Email: pace-support@oit.gatech.edu
- Docs: https://docs.pace.gatech.edu/

### MMomentA Issues
- Email: parmar@gatech.edu
- Check logs in `phoenix/logs/`

## Next Steps

After successful training:
1. Compare baseline vs. multipole performance
2. Analyze results in training_results.json
3. Run transfer learning on ZINC
4. Perform ablation studies
5. Generate publication figures

See main README.md for detailed documentation.
