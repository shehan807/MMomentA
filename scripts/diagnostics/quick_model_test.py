#!/usr/bin/env python
"""
Quick sanity check for model loading and inference.

This script performs a minimal test to verify:
1. Checkpoint loads correctly
2. Model can run inference on a single molecule
3. Predictions are reasonable (non-zero, non-constant)
4. Charge conservation works

Usage:
    python scripts/quick_model_test.py \
        --checkpoint runs/zinc_100k_multipoles/checkpoints/best_model.pt \
        --dataset data/zinc_100k_mpfit.h5

Expected output:
    - Model loads without errors
    - Predictions are non-zero
    - Total charge matches reference (neutral molecules)
    - Per-atom predictions vary (not all the same)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Add MMomentA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MMomentA.data.storage import load_dataset_hdf5
from MMomentA.data.graph import molecule_data_to_dgl_graph
from MMomentA.models import ChargeModel, ModelConfig

# Atomic number to symbol mapping
ATOMIC_SYMBOLS = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


def load_model(checkpoint_path: Path, device: str = 'cpu'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"  Epoch: {checkpoint['epoch']}")

    # Load model config
    if 'model_config' in checkpoint:
        model_config = ModelConfig(**checkpoint['model_config'])
    else:
        print("  WARNING: No model_config in checkpoint, using defaults")
        model_config = ModelConfig()

    # Create and load model
    model = ChargeModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  ✓ Model loaded successfully")
    print(f"  Feature dimension: {model_config.feature_units}")
    print(f"  Width: {model_config.width}")
    print(f"  Depth: {model_config.depth}")

    return model


def test_single_molecule(model, mol_data, device='cpu'):
    """Test model on a single molecule."""

    # Build graph
    graph = molecule_data_to_dgl_graph(
        mol_data,
        include_multipoles=True,
        use_conformer=False
    )

    graph = graph.to(device)

    # Run inference
    with torch.no_grad():
        graph = model(graph)

    # Extract predictions and targets
    predictions = graph.ndata["q"].cpu().numpy().flatten()
    targets = graph.ndata["q_ref"].cpu().numpy().flatten()
    atomic_numbers = graph.ndata["type"].squeeze(-1).cpu().numpy()  # Atomic numbers from 'type' field

    return predictions, targets, atomic_numbers


def main():
    parser = argparse.ArgumentParser(description="Quick model sanity check")

    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (.pt file)")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to HDF5 dataset")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--n-molecules", type=int, default=5,
                       help="Number of molecules to test (default: 5)")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("="*80)
    print("Quick Model Test")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print("")

    # Load model
    device = torch.device(args.device)
    model = load_model(checkpoint_path, args.device)
    print("")

    # Load dataset
    print("Loading dataset...")
    molecule_data_list, metadata = load_dataset_hdf5(args.dataset)
    print(f"  Loaded {len(molecule_data_list)} molecules")
    print("")

    # Test on first few molecules
    print("="*80)
    print(f"Testing on first {args.n_molecules} molecules")
    print("="*80)

    all_errors = []

    for i in range(min(args.n_molecules, len(molecule_data_list))):
        mol_data = molecule_data_list[i]

        print(f"\nMolecule {i}: {mol_data.smiles}")
        print(f"  N atoms: {len(mol_data.target_charges)}")

        # Run inference
        predictions, targets, atomic_numbers = test_single_molecule(model, mol_data, device)

        # Compute statistics
        errors = predictions - targets
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))

        print(f"  RMSE: {rmse:.5f} e")
        print(f"  MAE:  {mae:.5f} e")
        print(f"  Max error: {np.abs(errors).max():.5f} e")

        # Check charge conservation
        pred_total = predictions.sum()
        target_total = targets.sum()

        print(f"  Total charge (pred): {pred_total:.5f} e")
        print(f"  Total charge (ref):  {target_total:.5f} e")

        charge_conservation_error = abs(pred_total - target_total)
        if charge_conservation_error > 1e-4:
            print(f"  ⚠️  Charge conservation error: {charge_conservation_error:.5f} e")
        else:
            print(f"  ✓ Charge conserved")

        # Check prediction variance
        pred_std = predictions.std()
        print(f"  Prediction std: {pred_std:.5f} e")

        if pred_std < 1e-6:
            print(f"  🔴 WARNING: Predictions are nearly constant!")
        elif np.all(np.abs(predictions) < 1e-6):
            print(f"  🔴 WARNING: All predictions are near zero!")
        else:
            print(f"  ✓ Predictions have variance")

        # Show per-atom predictions (first 5 atoms)
        print(f"  Sample predictions:")
        for j in range(min(5, len(predictions))):
            element = ATOMIC_SYMBOLS.get(int(atomic_numbers[j]), f'Z{int(atomic_numbers[j])}')
            print(f"    Atom {j} ({element}): pred={predictions[j]:+.4f} e, ref={targets[j]:+.4f} e, error={errors[j]:+.4f} e")

        all_errors.extend(errors)

    # Overall statistics
    all_errors = np.array(all_errors)

    print("")
    print("="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total atoms tested: {len(all_errors)}")
    print(f"Overall RMSE: {np.sqrt(np.mean(all_errors**2)):.5f} e")
    print(f"Overall MAE: {np.mean(np.abs(all_errors)):.5f} e")
    print(f"Max error: {np.abs(all_errors).max():.5f} e")
    print(f"Error std: {all_errors.std():.5f} e")
    print("")

    # Final diagnosis
    print("="*80)
    print("DIAGNOSIS")
    print("="*80)

    issues = []

    # Check for zero predictions
    if np.all(np.abs(all_errors) < 1e-6):
        issues.append("🔴 All predictions are near zero")

    # Check for constant predictions
    if all_errors.std() < 1e-6:
        issues.append("🔴 Predictions are constant across molecules")

    # Check for unreasonable errors
    overall_rmse = np.sqrt(np.mean(all_errors**2))
    if overall_rmse > 1.0:
        issues.append(f"⚠️  Very high RMSE ({overall_rmse:.3f} e)")
    elif overall_rmse > 0.5:
        issues.append(f"⚠️  High RMSE ({overall_rmse:.3f} e)")

    if issues:
        print("Issues detected:")
        for issue in issues:
            print(f"  {issue}")
        print("")
        print("RECOMMENDATION:")
        print("  1. Check model checkpoint integrity")
        print("  2. Verify feature extraction matches training")
        print("  3. Compare to training validation RMSE (~0.08 e)")
    else:
        print("✓ Model appears to be working correctly!")
        print(f"  RMSE: {overall_rmse:.5f} e")
        print("")
        print("  This is a basic sanity check.")
        print("  For full evaluation, use scripts/evaluate_saved_model.py")

    print("")
    print("="*80)


if __name__ == "__main__":
    main()
