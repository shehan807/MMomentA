#!/usr/bin/env python
"""
MPFIT-GNN (MMomentA) ESP validation wrapper.
Loads trained MMomentA model and predicts charges for ESP validation.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import torch

# Import MMomentA modules
from MMomentA.data.storage import load_dataset
from MMomentA.data.dataset import create_split_datasets
from MMomentA.models import GCNModel
from MMomentA.training.config import ModelConfig
from MMomentA.data.graph import MoleculeGraph

# Import ESP utilities
from esp_utils import charges_to_esp_openff


def load_trained_model(model_path: str, device: str = 'cpu'):
    """Load trained MMomentA model from checkpoint.

    Parameters
    ----------
    model_path : str
        Path to best_model.pt checkpoint
    device : str
        Device to load model on ('cpu' or 'cuda')

    Returns
    -------
    tuple
        (model, config) - Loaded model and its configuration
    """

    checkpoint = torch.load(model_path, map_location=device)

    # Extract model configuration
    config_dict = checkpoint.get('config', {})
    model_config = ModelConfig(**config_dict)

    # Create model instance
    model = GCNModel(
        feature_units=model_config.feature_units,
        n_graph_layers=model_config.n_graph_layers,
        graph_hidden_units=model_config.graph_hidden_units,
        n_fc_layers=model_config.n_fc_layers,
        fc_hidden_units=model_config.fc_hidden_units,
        use_multipoles=model_config.use_multipoles
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Loaded model from {model_path}")
    print(f"  - Feature units: {model_config.feature_units}")
    print(f"  - Use multipoles: {model_config.use_multipoles}")
    print(f"  - Device: {device}")

    return model, model_config


def predict_charges_for_molecule(molecule, model, device='cpu'):
    """Predict MPFIT charges for a single molecule using trained model.

    Parameters
    ----------
    molecule : openff.toolkit.Molecule
        Molecule to predict charges for
    model : GCNModel
        Trained MMomentA model
    device : str
        Device for inference

    Returns
    -------
    np.ndarray
        Predicted atomic charges
    """

    # Convert molecule to graph
    mol_graph = MoleculeGraph.from_openff_molecule(molecule)

    # Create DGL graph and features
    graph, node_features = mol_graph.to_dgl()

    # Move to device
    graph = graph.to(device)
    node_features = node_features.to(device)

    # Predict charges
    with torch.no_grad():
        predicted_charges = model(graph, node_features)

    # Convert to numpy
    charges = predicted_charges.cpu().numpy().flatten()

    return charges


def validate_esp_single_molecule(molecule, model, grid_file, esp_file,
                                 device='cpu', verbose=True):
    """Run ESP validation for a single molecule.

    Parameters
    ----------
    molecule : openff.toolkit.Molecule
        Molecule to validate
    model : GCNModel
        Trained MMomentA model
    grid_file : str
        Path to Psi4 grid file
    esp_file : str
        Path to Psi4 ESP file
    device : str
        Device for inference
    verbose : bool
        Print detailed results

    Returns
    -------
    dict
        ESP validation metrics
    """

    # Predict charges
    charges = predict_charges_for_molecule(molecule, model, device=device)

    # Validate against ESP
    metrics = charges_to_esp_openff(
        molecule,
        charges,
        grid_file=grid_file,
        esp_file=esp_file,
        verbose=verbose
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="MPFIT-GNN ESP validation for molecules"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (best_model.pt)"
    )
    parser.add_argument(
        "--molecules",
        type=str,
        required=True,
        help="Path to pickle file with OpenFF molecules"
    )
    parser.add_argument(
        "--esp-dir",
        type=str,
        required=True,
        help="Directory containing Psi4 ESP files (grid.dat, grid_esp.dat for each molecule)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mpfit_gnn_results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference (default: cpu)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each molecule"
    )

    args = parser.parse_args()

    # Load model
    model, config = load_trained_model(args.model, device=args.device)

    # Load molecules
    print(f"\nLoading molecules from {args.molecules}...")
    with open(args.molecules, 'rb') as f:
        molecules = pickle.load(f)
    print(f"✓ Loaded {len(molecules)} molecules")

    # Run ESP validation for each molecule
    results = {
        'mae': [],
        'rmse': [],
        'molecules': {}
    }

    esp_dir = Path(args.esp_dir)

    for i, molecule in enumerate(molecules):
        mol_name = molecule.name if hasattr(molecule, 'name') else f"mol_{i}"

        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Molecule {i+1}/{len(molecules)}: {mol_name}")
            print(f"{'='*60}")

        # Find ESP files for this molecule
        # Assuming structure: esp_dir/mol_0/grid.dat, esp_dir/mol_0/grid_esp.dat
        mol_esp_dir = esp_dir / f"mol_{i}"

        if not mol_esp_dir.exists():
            # Try alternative naming
            mol_esp_dir = esp_dir / mol_name

        if not mol_esp_dir.exists():
            print(f"⚠ Warning: ESP directory not found for {mol_name}, skipping")
            continue

        grid_file = mol_esp_dir / "grid.dat"
        esp_file = mol_esp_dir / "grid_esp.dat"

        if not grid_file.exists() or not esp_file.exists():
            print(f"⚠ Warning: ESP files not found for {mol_name}, skipping")
            continue

        # Run validation
        metrics = validate_esp_single_molecule(
            molecule,
            model,
            str(grid_file),
            str(esp_file),
            device=args.device,
            verbose=args.verbose
        )

        # Store results
        if metrics:
            results['mae'].append(metrics['mae'])
            results['rmse'].append(metrics['rmse'])
            results['molecules'][mol_name] = metrics

            if not args.verbose:
                print(f"✓ {mol_name}: MAE={metrics['mae']:.6e}, RMSE={metrics['rmse']:.6e}")

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"MPFIT-GNN ESP Validation Summary")
    print(f"{'='*60}")
    print(f"Molecules validated: {len(results['mae'])}")
    print(f"Average MAE:  {np.mean(results['mae']):.6e} hartree/e")
    print(f"Average RMSE: {np.mean(results['rmse']):.6e} hartree/e")
    print(f"Std MAE:      {np.std(results['mae']):.6e} hartree/e")
    print(f"Std RMSE:     {np.std(results['rmse']):.6e} hartree/e")

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
