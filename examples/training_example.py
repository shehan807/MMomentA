#!/usr/bin/env python
"""Example: Train charge prediction model with multipole moments.

This demonstrates the complete ML training workflow.
"""

import torch
from MMomentA.data import load_dataset, create_split_datasets
from MMomentA.models import ChargeModel, ModelConfig
from MMomentA.training import Trainer, TrainingConfig, evaluate_model

print("Loading dataset...")
molecule_data_list, metadata = load_dataset("example_dataset.h5")

print(f"Loaded {len(molecule_data_list)} molecules")

splits = metadata.get('splits', {})
train_ids = set(splits.get('train', []))
val_ids = set(splits.get('val', []))
test_ids = set(splits.get('test', []))

train_idx = [i for i, mol in enumerate(molecule_data_list) if mol.molecule_id in train_ids]
val_idx = [i for i, mol in enumerate(molecule_data_list) if mol.molecule_id in val_ids]
test_idx = [i for i, mol in enumerate(molecule_data_list) if mol.molecule_id in test_ids]

print("\nCreating datasets...")
split_data = create_split_datasets(
    molecule_data_list,
    train_idx, val_idx, test_idx,
    include_multipoles=True
)

print(split_data)
print(f"Feature dimension: {split_data.train_dataset.feature_dim}")

print("\nModel with multipoles (198 features):")
model_config = ModelConfig(
    feature_units=198,
    depth=4,
    width=128,
    activation="relu"
)

model = ChargeModel(model_config)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

print("\nTraining configuration:")
training_config = TrainingConfig(
    n_epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-5,
    batch_size=4,
    eval_frequency=10,
    checkpoint_dir="example_checkpoints"
)

print(f"Epochs: {training_config.n_epochs}")
print(f"Learning rate: {training_config.learning_rate}")
print(f"Batch size: {training_config.batch_size}")

print("\nStarting training...")
trainer = Trainer(model, training_config)

train_loader = split_data.get_train_loader(training_config.batch_size)
val_loader = split_data.get_val_loader(training_config.batch_size)
test_loader = split_data.get_test_loader(training_config.batch_size)

results = trainer.train(train_loader, val_loader, test_loader)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Best validation RMSE: {results['best_val_rmse']:.5f}")
print(f"Best epoch: {results['best_epoch']}")

if 'test_metrics' in results:
    print(f"\nTest Results:")
    print(f"  RMSE: {results['test_metrics']['val_rmse']:.5f}")
    print(f"  MAE: {results['test_metrics']['val_mae']:.5f}")

print("\nEvaluating final model...")
evaluation = evaluate_model(model, test_loader)

print(f"\nDetailed Test Metrics:")
print(f"  Overall RMSE: {evaluation['overall']['rmse']:.5f}")
print(f"  Overall MAE: {evaluation['overall']['mae']:.5f}")
print(f"  Per-molecule RMSE (mean): {evaluation['per_molecule']['rmse_mean']:.5f}")
print(f"  Per-molecule RMSE (std): {evaluation['per_molecule']['rmse_std']:.5f}")
print(f"  Evaluated {evaluation['n_molecules']} molecules, {evaluation['n_atoms']} atoms")
