#!/usr/bin/env python
"""
Fast charge-only validation WITHOUT expensive ESP computation.

This script validates GNN charge predictions against MPFIT reference charges
for element-wise analysis, skipping the expensive Psi4 ESP calculations.

Usage:
    python scripts/validate_charges_only.py \
        --dataset data/zinc_100k_mpfit.h5 \
        --model-dir runs/zinc_100k_multipoles \
        --output-dir figures/zinc_100k_charge_validation \
        --split test

Expected runtime: ~5-10 minutes (vs 15-20 hours for ESP validation)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# Add MMomentA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MMomentA.data.storage import load_dataset_hdf5
from MMomentA.data.graph import molecule_data_to_dgl_graph
from MMomentA.data.schema import MoleculeData
from MMomentA.models import ChargeModel, ModelConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Atomic number to symbol mapping
ATOMIC_SYMBOLS = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


def load_model(model_dir: Path, device: str = 'cpu') -> ChargeModel:
    """Load trained MMomentA model from checkpoint."""

    checkpoint_path = model_dir / "checkpoints" / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model from config
    if 'model_config' in checkpoint:
        model_config = ModelConfig(**checkpoint['model_config'])
    else:
        logger.warning("No model config in checkpoint, using default")
        model_config = ModelConfig()

    model = ChargeModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded (epoch {checkpoint['epoch']})")
    logger.info(f"  Feature units: {model_config.feature_units}")
    logger.info(f"  Width: {model_config.width}")
    logger.info(f"  Depth: {model_config.depth}")

    return model


def predict_charges(model: ChargeModel, mol_data: MoleculeData,
                    multipole_stats: dict = None, device: str = 'cpu') -> np.ndarray:
    """Predict charges for a molecule using trained model."""

    # Infer multipoles from feature_units
    feature_units = model.config.feature_units
    include_multipoles = (feature_units > 150)  # 198 vs 117

    # CRITICAL: Pass multipole_stats for normalization!
    graph = molecule_data_to_dgl_graph(
        mol_data,
        include_multipoles=include_multipoles,
        multipole_stats=multipole_stats
    )
    graph = graph.to(device)

    # Predict
    with torch.no_grad():
        graph = model(graph)
        predicted_charges = graph.ndata["q"].cpu().numpy().flatten()

    return predicted_charges


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute charge prediction metrics."""
    errors = predictions - targets

    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(np.abs(errors))),
        'max_error': float(np.abs(errors).max()),
        'r2': float(1 - np.sum(errors**2) / np.sum((targets - targets.mean())**2))
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fast charge-only validation (no ESP)"
    )

    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to HDF5 dataset")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Path to model directory (contains checkpoints/)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--split", type=str, default="test",
                       choices=['train', 'val', 'test', 'all'],
                       help="Which split to validate")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for inference")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("Fast Charge-Only Validation (No ESP)")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model_dir}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Device: {args.device}")
    logger.info("")

    # Load model
    model = load_model(Path(args.model_dir), device=args.device)
    device = torch.device(args.device)

    # Load dataset
    logger.info("Loading dataset...")
    molecule_data_list, metadata = load_dataset_hdf5(args.dataset)
    logger.info(f"  Loaded {len(molecule_data_list)} molecules")

    # Get split indices
    if metadata and 'splits' in metadata:
        id_to_idx = {mol.molecule_id: idx for idx, mol in enumerate(molecule_data_list)}

        train_ids = metadata['splits'].get('train', [])
        val_ids = metadata['splits'].get('val', metadata['splits'].get('validation', []))
        test_ids = metadata['splits'].get('test', [])

        train_indices = [id_to_idx[mol_id] for mol_id in train_ids if mol_id in id_to_idx]
        val_indices = [id_to_idx[mol_id] for mol_id in val_ids if mol_id in id_to_idx]
        test_indices = [id_to_idx[mol_id] for mol_id in test_ids if mol_id in id_to_idx]

        logger.info(f"  Train: {len(train_indices)} molecules")
        logger.info(f"  Val: {len(val_indices)} molecules")
        logger.info(f"  Test: {len(test_indices)} molecules")
    else:
        logger.error("No split information in metadata!")
        sys.exit(1)

    # Compute multipole normalization statistics from training set
    logger.info("\nComputing multipole normalization statistics from training set...")
    if train_indices and model.config.feature_units > 150:
        all_multipoles = []
        for idx in train_indices[:min(10000, len(train_indices))]:
            mol_data = molecule_data_list[idx]
            all_multipoles.append(mol_data.multipole_moments)

        all_multipoles = np.vstack(all_multipoles)
        multipole_stats = {
            'mean': np.mean(all_multipoles, axis=0),
            'std': np.std(all_multipoles, axis=0) + 1e-8
        }
        logger.info(f"  Computed from {len(all_multipoles)} training molecules")
    else:
        multipole_stats = None
        logger.info("  No multipoles - using baseline features only")

    # Select molecules to validate
    if args.split == 'train':
        validate_indices = train_indices
    elif args.split == 'val':
        validate_indices = val_indices
    elif args.split == 'test':
        validate_indices = test_indices
    else:  # all
        validate_indices = train_indices + val_indices + test_indices

    logger.info(f"\nValidating {len(validate_indices)} molecules from {args.split} split...")

    # Collect all charges and atomic numbers
    all_mpfit_charges = []
    all_gnn_charges = []
    all_atomic_numbers = []
    all_element_symbols = []

    start_time = time.time()

    for idx in tqdm(validate_indices, desc="Predicting charges"):
        mol_data = molecule_data_list[idx]

        # Get MPFIT reference charges (precomputed)
        mpfit_charges = mol_data.target_charges

        # Predict with GNN
        gnn_charges = predict_charges(model, mol_data, multipole_stats, device=args.device)

        # Get atomic numbers
        atomic_numbers = mol_data.atomic_features['atomic_numbers']
        element_symbols = [ATOMIC_SYMBOLS.get(int(z), f'Z{int(z)}') for z in atomic_numbers]

        # Store
        all_mpfit_charges.extend(mpfit_charges)
        all_gnn_charges.extend(gnn_charges)
        all_atomic_numbers.extend(atomic_numbers)
        all_element_symbols.extend(element_symbols)

    elapsed_time = time.time() - start_time

    # Convert to numpy arrays
    all_mpfit_charges = np.array(all_mpfit_charges)
    all_gnn_charges = np.array(all_gnn_charges)
    all_atomic_numbers = np.array(all_atomic_numbers)
    all_element_symbols = np.array(all_element_symbols)

    logger.info(f"\n✓ Predictions completed in {elapsed_time:.2f}s")
    logger.info(f"  Total atoms: {len(all_mpfit_charges):,}")
    logger.info(f"  Throughput: {len(all_mpfit_charges)/elapsed_time:.0f} atoms/second")

    # Compute overall metrics
    logger.info("\n" + "="*80)
    logger.info("OVERALL METRICS")
    logger.info("="*80)

    overall_metrics = compute_metrics(all_gnn_charges, all_mpfit_charges)

    logger.info(f"RMSE: {overall_metrics['rmse']:.5f} e")
    logger.info(f"MAE:  {overall_metrics['mae']:.5f} e")
    logger.info(f"R²:   {overall_metrics['r2']:.4f}")
    logger.info(f"Max error: {overall_metrics['max_error']:.5f} e")

    # Compute per-element metrics
    logger.info("\n" + "="*80)
    logger.info("PER-ELEMENT BREAKDOWN")
    logger.info("="*80)

    element_charges = defaultdict(lambda: {'mpfit': [], 'gnn': []})

    for mpfit_q, gnn_q, element in zip(all_mpfit_charges, all_gnn_charges, all_element_symbols):
        element_charges[element]['mpfit'].append(mpfit_q)
        element_charges[element]['gnn'].append(gnn_q)

    per_element_metrics = {}

    # Sort elements by count
    elements_sorted = sorted(element_charges.keys(),
                            key=lambda e: len(element_charges[e]['mpfit']),
                            reverse=True)

    print(f"{'Element':<10} {'Count':<10} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-"*80)

    for element in elements_sorted:
        mpfit = np.array(element_charges[element]['mpfit'])
        gnn = np.array(element_charges[element]['gnn'])

        metrics = compute_metrics(gnn, mpfit)
        per_element_metrics[element] = {
            'count': len(mpfit),
            **metrics
        }

        print(f"{element:<10} {len(mpfit):<10,} {metrics['rmse']:<12.5f} "
              f"{metrics['mae']:<12.5f} {metrics['r2']:<12.4f}")

    print("-"*80)

    # Save results to JSON
    results = {
        'dataset': args.dataset,
        'model_dir': args.model_dir,
        'split': args.split,
        'n_molecules': len(validate_indices),
        'n_atoms': len(all_mpfit_charges),
        'elapsed_time': elapsed_time,
        'overall_metrics': overall_metrics,
        'per_element_metrics': per_element_metrics,
        'charges': {
            'mpfit': all_mpfit_charges.tolist(),
            'gnn': all_gnn_charges.tolist(),
            'atomic_numbers': all_atomic_numbers.tolist(),
            'element_symbols': all_element_symbols.tolist()
        }
    }

    output_file = output_dir / f"charge_validation_{args.split}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to: {output_file}")

    # Generate plots
    logger.info("\nGenerating element-wise plots...")

    try:
        from figures.plot_comparison import create_element_hexbin_plots

        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        # Call plotting function with charge data
        create_element_hexbin_plots(
            element_charges,
            output_dir=str(plot_dir),
            method_name="GNN"
        )

        logger.info(f"✓ Plots saved to: {plot_dir}")

    except ImportError:
        logger.warning("Could not import plotting functions - skipping plots")
        logger.info("Run scripts/plot_element_charge_comparison.py manually to generate plots")

    logger.info("\n" + "="*80)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {elapsed_time:.2f}s")
    logger.info(f"Overall RMSE: {overall_metrics['rmse']:.5f} e")
    logger.info("")


if __name__ == "__main__":
    main()
