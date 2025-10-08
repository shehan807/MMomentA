# MMomentA - Phase 3: ML Model Development

## Overview

Phase 3 implements complete ML model training for charge prediction:
- **Model architecture** adapted from espaloma-charge with multipole support
- **Training loop** with validation and early stopping
- **Checkpointing** and model persistence
- **Evaluation metrics** and comprehensive analysis
- **Ablation studies** comparing baseline vs. multipole-enhanced models
- **Training scripts** for SPICE and ZINC datasets

## Model Architecture

The model combines graph neural networks with charge equilibration:

```
Input Graph (with multipole features)
    ↓
GNN Encoder (SAGEConv layers)
    ↓
ChargeReadout (predict e and s parameters)
    ↓
ChargeEquilibrium (analytical solution)
    ↓
Predicted Charges (q)
```

### Components

**1. Sequential GNN**
- Initial featurization: Linear(feature_dim → hidden_dim) + Tanh
- Graph convolution: SAGEConv layers with mean aggregation
- Configurable depth, width, activation, dropout

**2. ChargeReadout**
- Predicts per-atom electronegativity (e) and hardness (s)
- Linear layer: hidden_dim → 2

**3. ChargeEquilibrium**
- Analytical charge equilibration using Lagrange multipliers
- Ensures total charge conservation
- Solution: q_i = -e_i/s_i + (1/s_i) * (Q + Σe_j/s_j) / Σ(1/s_k)

### Feature Dimensions

```python
# Baseline (no multipoles)
feature_units = 117  # 100 (one-hot) + 17 (fingerprint)

# With multipoles (limit=8)
feature_units = 198  # 117 (baseline) + 81 (multipoles)
```

## Quick Start

### 1. Train Baseline Model

```python
from MMomentA.data import load_dataset, create_split_datasets
from MMomentA.models import ChargeModel, ModelConfig
from MMomentA.training import Trainer, TrainingConfig

# Load preprocessed dataset
molecule_data_list, metadata = load_dataset("spice_mpfit.h5")

# Create splits
train_idx, val_idx, test_idx = ...  # from metadata
split_data = create_split_datasets(
    molecule_data_list, train_idx, val_idx, test_idx,
    include_multipoles=False  # Baseline
)

# Model configuration
model_config = ModelConfig(
    feature_units=117,  # Baseline
    depth=4,
    width=128
)

model = ChargeModel(model_config)

# Training configuration
training_config = TrainingConfig(
    n_epochs=1000,
    learning_rate=1e-3,
    batch_size=128,
    checkpoint_dir="checkpoints/baseline"
)

# Train
trainer = Trainer(model, training_config)
train_loader = split_data.get_train_loader(128)
val_loader = split_data.get_val_loader(128)
test_loader = split_data.get_test_loader(128)

results = trainer.train(train_loader, val_loader, test_loader)
print(f"Test RMSE: {results['test_metrics']['val_rmse']:.5f}")
```

### 2. Train with Multipole Moments

```python
# Same as above, but:
split_data = create_split_datasets(
    molecule_data_list, train_idx, val_idx, test_idx,
    include_multipoles=True  # Enable multipoles
)

model_config = ModelConfig(
    feature_units=198,  # With multipoles
    depth=4,
    width=128
)
```

### 3. Command-Line Training

```bash
# Train on SPICE (baseline)
python scripts/train_spice.py \
    --dataset spice_mpfit.h5 \
    --output-dir runs/spice_baseline \
    --feature-units 117 \
    --no-multipoles \
    --n-epochs 1000

# Train on SPICE (with multipoles)
python scripts/train_spice.py \
    --dataset spice_mpfit.h5 \
    --output-dir runs/spice_multipoles \
    --feature-units 198 \
    --n-epochs 1000

# Transfer learning on ZINC
python scripts/train_zinc.py \
    --dataset zinc_mpfit.h5 \
    --pretrained runs/spice_multipoles/checkpoints/best_model.pt \
    --output-dir runs/zinc_finetune \
    --n-epochs 500
```

## Model Configuration

### ModelConfig

```python
from MMomentA.models import ModelConfig

config = ModelConfig(
    feature_units=198,      # Input feature dimension
    input_units=128,        # Hidden dim after featurization
    depth=4,                # Number of GNN layers
    width=128,              # GNN hidden dimension
    activation="relu",      # Activation function
    aggregator_type="mean", # SAGEConv aggregation
    dropout=0.0,            # Dropout probability
    batch_norm=False        # Use batch normalization
)
```

### TrainingConfig

```python
from MMomentA.training import TrainingConfig

config = TrainingConfig(
    n_epochs=1000,                    # Training epochs
    learning_rate=1e-3,               # Learning rate
    weight_decay=1e-5,                # L2 regularization
    batch_size=128,                   # Batch size
    eval_frequency=10,                # Validate every N epochs
    checkpoint_dir="checkpoints",     # Checkpoint directory
    save_frequency=50,                # Save every N epochs
    early_stopping_patience=0,        # Early stopping (0=disabled)
    device="cuda"                     # Device (cuda/cpu)
)
```

## Training and Evaluation

### Training Loop

```python
from MMomentA.training import Trainer

trainer = Trainer(model, training_config)

results = trainer.train(
    train_loader,
    val_loader,
    test_loader  # Optional: evaluate at end
)

# Results dictionary:
# {
#     'best_val_rmse': float,
#     'best_epoch': int,
#     'metrics_history': {...},
#     'test_metrics': {...}  # if test_loader provided
# }
```

### Checkpointing

Automatic checkpointing:
- Best model: `checkpoint_dir/best_model.pt`
- Periodic: `checkpoint_dir/checkpoint_epoch_{N}.pt`

Checkpoint contents:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'config': dict,
    'model_config': dict,
    'metrics_tracker': dict,
    'metrics': dict  # current metrics
}
```

Loading checkpoint:
```python
trainer.load_checkpoint("checkpoints/best_model.pt")
```

### Evaluation

```python
from MMomentA.training import evaluate_model

results = evaluate_model(model, test_loader, device='cuda')

# Returns:
# {
#     'overall': {
#         'rmse': float,
#         'mae': float,
#         'max_error': float
#     },
#     'per_molecule': {
#         'rmse_mean': float,
#         'rmse_std': float,
#         'rmse_median': float,
#         'rmses': list,  # per-molecule values
#         'maes': list
#     },
#     'n_molecules': int,
#     'n_atoms': int
# }
```

## Ablation Studies

Compare baseline vs. multipole-enhanced models:

```python
from MMomentA.training import ablation_study
from MMomentA.models import ModelConfig

model_configs = {
    'baseline': ModelConfig(feature_units=117, depth=4, width=128),
    'with_multipoles': ModelConfig(feature_units=198, depth=4, width=128),
    'deeper': ModelConfig(feature_units=198, depth=6, width=128)
}

results = ablation_study(
    split_data,
    model_configs,
    training_config,
    n_runs=3  # Repeat for statistical significance
)

# Results:
# {
#     'baseline': {
#         'mean_test_rmse': float,
#         'std_test_rmse': float,
#         'runs': [...]
#     },
#     'with_multipoles': {...},
#     'deeper': {...}
# }
```

## Model Comparison

```python
from MMomentA.training import compare_models

models = {
    'baseline': baseline_model,
    'multipoles': multipole_model
}

results = compare_models(models, test_loader, device='cuda')

# Returns metrics for each model
for name, metrics in results.items():
    print(f"{name}: RMSE = {metrics['overall']['rmse']:.5f}")
```

## Metrics

### Training Metrics

Tracked automatically during training:
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **Max Error**: Maximum absolute error

### Per-Molecule Metrics

Statistics across molecules:
- Mean, std, median, min, max of RMSE/MAE
- Identifies outlier molecules
- Distribution analysis

### Charge Conservation

For each molecule:
- Predicted total charge
- Target total charge
- Conservation error

## Expected Performance

Based on espaloma-charge baseline:

**Baseline (no multipoles, 117 features)**
- Train RMSE: ~0.03 e
- Val RMSE: ~0.04 e
- Test RMSE: ~0.045 e

**With multipoles (198 features)**
- Target: Test RMSE < 0.04 e
- Improved ESP reproduction
- Better transferability

**Transfer to ZINC**
- Pre-trained on SPICE → fine-tune on ZINC
- Tests drug-like molecule generalization

## Success Criteria (from specification)

✅ **Code quality**: Modular, tested, documented
✅ **Data pipeline**: Robust MPFIT on 1000+ molecules
✅ **ML performance**:
- Charge RMSE < 0.05 e on test sets
- ESP RMSE competitive with direct MPFIT
- 100-1000x speedup vs QM at inference

## Directory Structure

```
runs/
├── spice_baseline/
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── checkpoint_epoch_*.pt
│   ├── training.log
│   └── training_results.json
└── spice_multipoles/
    ├── checkpoints/
    ├── training.log
    └── training_results.json
```

## Training Tips

### Hyperparameter Tuning

Start with defaults, then tune:
1. **Learning rate**: Try 1e-4, 1e-3, 1e-2
2. **Depth**: 4-6 layers
3. **Width**: 64, 128, 256
4. **Batch size**: 32, 64, 128

### Debugging

If training fails:
- Check feature dimension matches model config
- Verify data loaders work (iterate once before training)
- Start with small dataset/short training
- Monitor gradient norms

### GPU Usage

```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Train on GPU
training_config = TrainingConfig(device="cuda")
```

## Examples

See `examples/` directory:
- `training_example.py`: Complete training workflow
- `ablation_study_example.py`: Compare baseline vs. multipoles

## Integration with Phases 1 & 2

**Complete Workflow:**

```bash
# Phase 1: Compute MPFIT charges
python scripts/prepare_dataset.py \
    --input spice.oeb \
    --output spice_mpfit.h5 \
    --max-molecules 1000

# Phase 2: Already done (dataset prepared)

# Phase 3: Train model
python scripts/train_spice.py \
    --dataset spice_mpfit.h5 \
    --output-dir runs/spice \
    --feature-units 198

# Evaluate
python -c "
from MMomentA.training import Trainer
from pathlib import Path
trainer = Trainer.load_checkpoint(Path('runs/spice/checkpoints/best_model.pt'))
# ... evaluate on test set
"
```

## Next Steps

**Further Improvements:**
1. **Ensemble models**: Train multiple models, average predictions
2. **Uncertainty quantification**: Predict confidence intervals
3. **Active learning**: Identify molecules needing more data
4. **Architecture search**: Try GAT, GIN, or other GNN variants
5. **Loss functions**: ESP reproduction loss, dipole moment loss

## References

- espaloma-charge: https://github.com/choderalab/espaloma_charge
- DGL documentation: https://docs.dgl.ai/
- PyTorch Lightning (future integration): https://www.pytorchlightning.ai/
