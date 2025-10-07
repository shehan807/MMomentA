## Perlmutter Pipelines

Perlmutter-specific versions of the training pipelines with correct module loading and paths.

### Quick Start

```bash
cd /path/to/MMomentA

# Run MVP (150 molecules, ~1-2 hours)
bash perlmutter/pipelines/run_mvp.sh

# Or submit as batch job
sbatch perlmutter/jobs/submit_mvp.sh
```

### Why Perlmutter-Specific Versions?

The cluster-independent versions have Phoenix-specific defaults. These Perlmutter versions:
- Use `module load conda` (not `anaconda3`)
- Use `$SCRATCH/conda-envs/` paths
- Handle Perlmutter's CUDA 12.x

### Available Pipelines

All scripts in this directory are ready to run on Perlmutter:
- `run_mvp.sh` - MVP test (150 molecules)
- More coming soon (SPICE, ZINC)

### Usage

**Interactive:**
```bash
# Request GPU
salloc -C gpu -q interactive -t 02:00:00 -A <account>

# Run pipeline
bash perlmutter/pipelines/run_mvp.sh
```

**Batch:**
```bash
sbatch perlmutter/jobs/submit_mvp.sh
```
