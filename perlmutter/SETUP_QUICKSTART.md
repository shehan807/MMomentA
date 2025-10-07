# Perlmutter Setup Quick Start

## Issue: Conda Package Cache

Perlmutter's default conda tries to write to read-only directories. You need to configure conda to use `$SCRATCH` first.

## Solution: 3-Step Setup

### Step 0: Configure Conda (REQUIRED - Do This First!)

```bash
cd /path/to/MMomentA
bash perlmutter/setup/configure_conda.sh
```

**What this does:**
- Creates `~/.condarc` with `$SCRATCH` paths
- Sets package cache: `$SCRATCH/.conda/pkgs`
- Sets environment location: `$SCRATCH/conda-envs`
- **This is permanent** - only needs to be done once

### Step 1: Setup Data Environment

```bash
# On login node
bash perlmutter/setup/MANUAL_SETUP_DATA.sh
```

Creates: `$SCRATCH/conda-envs/mmomenta-data`

### Step 2: Setup ML Environment

```bash
# Request GPU node
salloc -C gpu -q interactive -t 01:00:00 -A <your_account>

# On GPU node
bash perlmutter/setup/MANUAL_SETUP_ML.sh
```

Creates: `$SCRATCH/conda-envs/mmomenta-ml`

## Verification

```bash
# Verify setup
bash perlmutter/setup/verify_installation.sh
```

## Complete Setup Sequence

```bash
# 1. Configure conda (one-time)
bash perlmutter/setup/configure_conda.sh

# 2. Create data environment (login node)
bash perlmutter/setup/MANUAL_SETUP_DATA.sh

# 3. Create ML environment (GPU node)
salloc -C gpu -q interactive -t 01:00:00 -A <your_account>
bash perlmutter/setup/MANUAL_SETUP_ML.sh
exit

# 4. Verify
bash perlmutter/setup/verify_installation.sh

# 5. Add aliases (optional)
cat >> ~/.bashrc << 'EOF'
export MMOMENTA_DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
export MMOMENTA_ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"
alias mmdata='module load conda && conda activate $MMOMENTA_DATA_ENV'
alias mmml='module load conda && conda activate $MMOMENTA_ML_ENV'
EOF

source ~/.bashrc
```

## Usage After Setup

```bash
# Activate environments
module load conda
conda activate $SCRATCH/conda-envs/mmomenta-data  # or use alias: mmdata
conda activate $SCRATCH/conda-envs/mmomenta-ml    # or use alias: mmml

# Run pipelines
cd cluster-independent/pipelines
./run_mvp.sh

# Or submit jobs
sbatch perlmutter/jobs/submit_mvp.sh
```

## Troubleshooting

### "NoWritablePkgsDirError"
**Problem**: Conda trying to write to read-only `/global/common/software/`
**Solution**: Run `configure_conda.sh` first (Step 0)

### "Environment not found"
**Problem**: Environment creation failed due to package cache issue
**Solution**:
1. Run `configure_conda.sh`
2. Remove failed environment: `rm -rf $SCRATCH/conda-envs/mmomenta-*`
3. Try setup again

### Check Current Configuration
```bash
conda config --show pkgs_dirs
conda config --show envs_dirs
```

Should show `$SCRATCH` paths, not `/global/common/software/`

### Manual Configuration (Alternative)
```bash
# If script doesn't work, configure manually:
conda config --add pkgs_dirs $SCRATCH/.conda/pkgs
conda config --add envs_dirs $SCRATCH/conda-envs
conda config --set channel_priority flexible
```

## Key Differences from Phoenix

| Aspect | Phoenix | Perlmutter |
|--------|---------|------------|
| Module | `module load anaconda3` | `module load conda` |
| Package cache | Default location | **Must use $SCRATCH** |
| Env location | `-p` flag optional | `-p $SCRATCH/...` required |
| CUDA | 11.8 | 12.x |
| Config needed | No | **Yes** (configure_conda.sh) |

## Summary

1. ✅ **First time**: Run `configure_conda.sh`
2. ✅ **Then**: Run setup scripts as normal
3. ✅ **Always**: Use `$SCRATCH` for conda environments on Perlmutter

The configuration is permanent - you only need to do Step 0 once!
