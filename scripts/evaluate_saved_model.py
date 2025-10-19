#!/usr/bin/env python
"""
Evaluate saved model checkpoint on train/val/test splits.

This script reloads the trained model and evaluates it to verify:
1. Checkpoint loads correctly (val RMSE should match training)
2. Test set performance matches ESP validation results
3. Per-element RMSE breakdown

Usage:
    python scripts/evaluate_saved_model.py \
        --checkpoint runs/zinc_100k_multipoles/checkpoints/best_model.pt \
        --dataset data/zinc_100k_mpfit.h5 \
        --split validation

Expected results:
    - Validation: RMSE ≈ 0.08 e (should match training log)
    - Test: RMSE ≈ 0.3-0.7 e (if distribution mismatch)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict

# Add MMomentA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MMomentA.data.storage import load_dataset_hdf5
from MMomentA.data.dataset import create_split_datasets
from MMomentA.models import ChargeModel, ModelConfig
import dgl

# Atomic number to symbol mapping
ATOMIC_SYMBOLS = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


def load_model_from_checkpoint(checkpoint_path: Path, device: str = 'cpu'):
    """Load model from checkpoint file."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Keys in checkpoint: {list(checkpoint.keys())}")

    # Load model config
    if 'model_config' in checkpoint:
        model_config = ModelConfig(**checkpoint['model_config'])
        print(f"  Feature units: {model_config.feature_units}")
        print(f"  Width: {model_config.width}")
        print(f"  Depth: {model_config.depth}")
    else:
        print("  WARNING: No model_config in checkpoint, using defaults")
        model_config = ModelConfig()

    # Create and load model
    model = ChargeModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  ✓ Model loaded successfully")

    return model


def evaluate_split(model, dataloader, device, split_name):
    """Evaluate model on a data split."""
    print(f"\nEvaluating on {split_name} set...")

    all_predictions = []
    all_targets = []
    all_atomic_numbers = []

    with torch.no_grad():
        for i, batch_graph in enumerate(dataloader):
            if i % 100 == 0:
                print(f"  Processing batch {i}/{len(dataloader)}...")

            batch_graph = batch_graph.to(device)
            batch_graph = model(batch_graph)

            predictions = batch_graph.ndata["q"].cpu().numpy()
            targets = batch_graph.ndata["q_ref"].cpu().numpy()
            atomic_numbers = batch_graph.ndata["h0"][:, 0].cpu().numpy()  # First feature is atomic number

            all_predictions.append(predictions)
            all_targets.append(targets)
            all_atomic_numbers.append(atomic_numbers)

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    all_atomic_numbers = np.concatenate(all_atomic_numbers)

    # Compute overall metrics
    errors = all_predictions - all_targets
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    print(f"\n{'='*80}")
    print(f"{split_name.upper()} SET RESULTS")
    print(f"{'='*80}")
    print(f"Total atoms: {len(all_predictions):,}")
    print(f"Overall RMSE: {rmse:.5f} e")
    print(f"Overall MAE:  {mae:.5f} e")
    print(f"Max error:    {np.abs(errors).max():.5f} e")
    print("")

    # Per-element breakdown
    element_stats = defaultdict(lambda: {'predictions': [], 'targets': []})

    for pred, target, z in zip(all_predictions, all_targets, all_atomic_numbers):
        element = ATOMIC_SYMBOLS.get(int(z), f'Z{int(z)}')
        element_stats[element]['predictions'].append(pred)
        element_stats[element]['targets'].append(target)

    # Convert to numpy and compute metrics
    print(f"{'='*80}")
    print("PER-ELEMENT BREAKDOWN")
    print(f"{'='*80}")
    print(f"{'Element':<10} {'Count':<10} {'RMSE':<12} {'MAE':<12} {'Max Error':<12}")
    print("-"*80)

    # Sort by count
    elements_sorted = sorted(element_stats.keys(),
                            key=lambda e: len(element_stats[e]['predictions']),
                            reverse=True)

    for element in elements_sorted:
        preds = np.array(element_stats[element]['predictions'])
        targets = np.array(element_stats[element]['targets'])

        elem_errors = preds - targets
        elem_rmse = np.sqrt(np.mean(elem_errors ** 2))
        elem_mae = np.mean(np.abs(elem_errors))
        elem_max = np.abs(elem_errors).max()

        print(f"{element:<10} {len(preds):<10,} {elem_rmse:<12.5f} {elem_mae:<12.5f} {elem_max:<12.5f}")

    print("-"*80)
    print("")

    return {
        'rmse': rmse,
        'mae': mae,
        'per_element': {
            element: {
                'rmse': float(np.sqrt(np.mean((np.array(element_stats[element]['predictions']) -
                                               np.array(element_stats[element]['targets'])) ** 2))),
                'mae': float(np.mean(np.abs(np.array(element_stats[element]['predictions']) -
                                            np.array(element_stats[element]['targets'])))),
                'count': len(element_stats[element]['predictions'])
            }
            for element in element_stats
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved model checkpoint")

    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (.pt file)")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to HDF5 dataset")
    parser.add_argument("--split", type=str, required=True,
                       choices=['train', 'validation', 'test', 'all'],
                       help="Which split to evaluate")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for evaluation")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("="*80)
    print("Evaluating Saved Model")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print("")

    # Load model
    device = torch.device(args.device)
    model = load_model_from_checkpoint(checkpoint_path, args.device)

    # Load dataset
    print("\nLoading dataset...")
    molecule_data_list, metadata = load_dataset_hdf5(args.dataset)
    print(f"  Loaded {len(molecule_data_list)} molecules")
    print(f"  Dataset: {metadata.get('dataset_name', 'Unknown')}")

    # Create datasets
    print("\nCreating datasets...")
    datasets = create_split_datasets(
        molecule_data_list,
        metadata,
        include_multipoles=True  # Use multipoles if available
    )

    train_dataset, val_dataset, test_dataset = datasets

    print(f"  Train: {len(train_dataset)} molecules")
    print(f"  Val: {len(val_dataset)} molecules")
    print(f"  Test: {len(test_dataset)} molecules")

    # Create dataloaders
    def collate_fn(samples):
        """Collate function for DGL graphs."""
        return dgl.batch(samples)

    if args.split == 'train' or args.split == 'all':
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        train_results = evaluate_split(model, train_loader, device, "train")

    if args.split == 'validation' or args.split == 'all':
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        val_results = evaluate_split(model, val_loader, device, "validation")

    if args.split == 'test' or args.split == 'all':
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        test_results = evaluate_split(model, test_loader, device, "test")

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    if args.split == 'validation' or args.split == 'all':
        print(f"\nValidation RMSE: {val_results['rmse']:.5f} e")
        print("  Expected from training log: ~0.08049 e")

        if abs(val_results['rmse'] - 0.08049) < 0.01:
            print("  ✓ MATCHES training performance - checkpoint loads correctly!")
        else:
            print("  ⚠️  MISMATCH - possible loading issue!")

    if args.split == 'test' or args.split == 'all':
        print(f"\nTest RMSE: {test_results['rmse']:.5f} e")
        print("  Expected from ESP validation: ~0.3-0.7 e per element")

        if test_results['rmse'] > 0.15:
            print("  ⚠️  High test RMSE - distribution mismatch or difficult test set")
        else:
            print("  ✓ Reasonable test performance")

    if args.split == 'all':
        print(f"\n Train RMSE: {train_results['rmse']:.5f} e")
        print(f" Val RMSE:   {val_results['rmse']:.5f} e")
        print(f" Test RMSE:  {test_results['rmse']:.5f} e")

        if test_results['rmse'] > 1.5 * val_results['rmse']:
            print("\n🔴 SIGNIFICANT TRAIN/TEST GAP!")
            print("   Likely cause: Scaffold split created very different distributions")
            print("   Recommendation: Use random split or stratified split")

    print("")
    print("="*80)


if __name__ == "__main__":
    main()
