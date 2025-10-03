#!/usr/bin/env python
"""Example: Run ablation study comparing baseline vs multipole models."""

from MMomentA.data import load_dataset, create_split_datasets
from MMomentA.models import ModelConfig
from MMomentA.training import TrainingConfig, ablation_study

print("Loading dataset...")
molecule_data_list, metadata = load_dataset("example_dataset.h5")

splits = metadata.get('splits', {})
train_ids = set(splits.get('train', []))
val_ids = set(splits.get('val', []))
test_ids = set(splits.get('test', []))

train_idx = [i for i, mol in enumerate(molecule_data_list) if mol.molecule_id in train_ids]
val_idx = [i for i, mol in enumerate(molecule_data_list) if mol.molecule_id in val_ids]
test_idx = [i for i, mol in enumerate(molecule_data_list) if mol.molecule_id in test_ids]

print("\nCreating dataset splits...")
split_data_baseline = create_split_datasets(
    molecule_data_list,
    train_idx, val_idx, test_idx,
    include_multipoles=False
)

split_data_multipoles = create_split_datasets(
    molecule_data_list,
    train_idx, val_idx, test_idx,
    include_multipoles=True
)

print(f"Baseline features: {split_data_baseline.train_dataset.feature_dim}")
print(f"Multipole features: {split_data_multipoles.train_dataset.feature_dim}")

print("\nDefining model configurations...")
model_configs = {
    'baseline': ModelConfig(
        feature_units=117,
        depth=4,
        width=128
    ),
    'with_multipoles': ModelConfig(
        feature_units=198,
        depth=4,
        width=128
    ),
    'deeper_baseline': ModelConfig(
        feature_units=117,
        depth=6,
        width=128
    )
}

training_config = TrainingConfig(
    n_epochs=100,
    batch_size=4,
    eval_frequency=10,
    checkpoint_dir="ablation_checkpoints"
)

print("\nRunning ablation study (3 configs x 2 runs each)...")
print("This will take a while...")

results = ablation_study(
    split_data_multipoles,
    model_configs,
    training_config,
    n_runs=2
)

print("\n" + "=" * 60)
print("Ablation Study Results")
print("=" * 60)

for config_name, config_results in results.items():
    print(f"\n{config_name}:")
    print(f"  Mean test RMSE: {config_results['mean_test_rmse']:.5f}")
    print(f"  Std test RMSE: {config_results['std_test_rmse']:.5f}")

    for run_idx, run_result in enumerate(config_results['runs']):
        print(f"  Run {run_idx + 1}: {run_result['test_metrics']['val_rmse']:.5f}")

print("\n" + "=" * 60)
