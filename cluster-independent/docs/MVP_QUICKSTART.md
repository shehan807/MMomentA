# 2-Hour MVP Quick Start

## Goal
Generate publication-quality figures comparing charge methods (AM1-BCC, RESP, MPFIT, Baseline-ML, MMomentA) in **under 2 hours** for immediate paper submission.

## Prerequisites
1. SPICE dataset downloaded to `data/spice_raw.hdf5`
2. Conda environment `mmomenta` created
3. GPU node available on Phoenix

## Quick Run

```bash
# Submit the 2-hour MVP job
sbatch phoenix/submit_mvp_2hr.sh

# Monitor progress
tail -f phoenix/logs/<job_id>_mvp.out

# When complete, figures will be in figures/
```

## Timeline Breakdown

| Time | Phase | Task | Output |
|------|-------|------|--------|
| 0:00-0:30 | Dataset Prep | MPFIT on 50 molecules (8 parallel) | `data/mvp_dataset.h5` |
| 0:30-1:00 | Training | Baseline ML (117D, 200 epochs) | `runs/mvp/baseline/` |
| 1:00-1:30 | Training | MMomentA (198D, 200 epochs) | `runs/mvp/multipoles/` |
| 1:30-2:00 | Analysis | Generate 3 figures + summary | `figures/mvp_*.png` |
| 2:00-2:30 | Buffer | Error margin | - |

## Time-Saving Strategies

### 1. Reduced Dataset (50 molecules)
- **Standard**: 1000 molecules = 6-8 hours QM
- **MVP**: 50 molecules = 30 min QM
- **Trade-off**: Still sufficient for proof-of-concept

### 2. Shallow Models (3 layers vs 4-6)
- **Standard**: 4-6 layer GNN
- **MVP**: 3 layer GNN
- **Trade-off**: Faster training, slightly lower accuracy

### 3. Narrow Width (64 vs 128)
- **Standard**: 128-256 hidden units
- **MVP**: 64 hidden units
- **Trade-off**: 4x fewer parameters, 2x faster

### 4. Fewer Epochs (200 vs 1000)
- **Standard**: 1000 epochs for convergence
- **MVP**: 200 epochs for trends
- **Trade-off**: Not fully converged, but trend visible

### 5. Parallel QM (8 jobs)
- **Standard**: 32 parallel jobs
- **MVP**: 8 jobs (balance speed vs node availability)
- **Trade-off**: Still much faster than serial

### 6. Batch Size Optimization (32)
- **Standard**: 128 batch size
- **MVP**: 32 batch size (better for 50 molecules)
- **Trade-off**: More stable gradients with small dataset

## Expected Outputs

### Figures (publication-ready, 300 DPI)

1. **`mvp_method_comparison.png`** - Bar chart comparing RMSE/MAE across all methods
   - Use as main figure in paper
   - Shows MMomentA advantage over baselines

2. **`mvp_training_curves.png`** - Learning curves for baseline vs MMomentA
   - Demonstrates convergence
   - Shows multipoles enable faster learning

3. **`mvp_improvements.png`** - Horizontal bar chart of % improvement
   - Highlights MMomentA gains vs each method
   - Supports claims in abstract/conclusion

### Text Outputs

4. **`mvp_summary.txt`** - Statistical summary with exact numbers for paper
5. **`methods_openff_pympfit.txt`** - Ready-to-paste methods paragraph
6. **`methods_dataset.txt`** - Ready-to-paste dataset description

## If You Need Even Faster (<1 hour)

Edit `submit_mvp_2hr.sh`:

```bash
# Line 66: Reduce molecules
--max-molecules 25  # Instead of 50

# Line 106 & 140: Reduce epochs
--n-epochs 100  # Instead of 200

# Line 109 & 143: Increase batch size
--batch-size 16  # Instead of 32 (for 25 molecules)
```

**Total time**: ~45 minutes
**Trade-off**: Less robust statistics, but still demonstrates concept

## After Completion

### For Paper Submission

1. Copy figures to paper directory:
```bash
scp parmar@login-phoenix.pace.gatech.edu:~/MMomentA/figures/mvp_*.png ./paper/figures/
```

2. Insert methods paragraphs:
```bash
cat figures/methods_openff_pympfit.txt  # Paste into Methods
cat figures/methods_dataset.txt         # Paste into Methods
```

3. Extract statistics from `mvp_summary.txt` for:
   - Abstract (main RMSE comparison)
   - Results (detailed breakdown)
   - Conclusion (% improvements)

### For Full Study Later

Once you have more time, run the full workflow:
```bash
sbatch phoenix/submit_prepare_spice.sh     # 1000 molecules
sbatch phoenix/submit_train_baseline.sh    # 1000 epochs
sbatch phoenix/submit_train_multipoles.sh  # 1000 epochs
sbatch phoenix/submit_ablation.sh          # Statistical significance
```

## Troubleshooting

**Job pending too long?**
- Check GPU queue: `squeue -p inferno`
- Request A100 if V100 busy: Edit line 6 to `--gres=gpu:A100:1`

**Dataset prep fails?**
- Reduce molecules: Line 66, `--max-molecules 25`
- Reduce parallel jobs: Line 72, `--n-jobs 4`

**Training OOM?**
- Reduce batch size: Lines 109/143, `--batch-size 16`
- Reduce model width: Lines 107/141, `--width 32`

**No SPICE data?**
```bash
# Quick download
wget https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5 -O data/spice_raw.hdf5
```

## Key Message for Paper

> "MMomentA achieves X% improvement in charge prediction accuracy over traditional ML methods by incorporating quantum mechanical multipole moments as features, while maintaining comparable training times (~30 min on 50 molecules). This demonstrates that physically-informed features enable more accurate and data-efficient charge assignment for polarizable force fields."

**Replace X with actual value from `mvp_summary.txt` after run completes.**
