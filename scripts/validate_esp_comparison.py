#!/usr/bin/env python
"""
Validate MPFIT, AM1-BCC, RESP, and MMomentA-GNN charges by comparing ESP reproduction.

This script:
1. Loads trained MMomentA-GNN model
2. Loads test molecules from HDF5 dataset
3. Computes QM ESP for each molecule using Psi4
4. Compares ESP from MPFIT charges
5. Compares ESP from AM1-BCC charges
6. Compares ESP from RESP charges
7. Compares ESP from MMomentA-GNN predicted charges
8. Generates publication-quality violin plots

Usage:
    python scripts/validate_esp_comparison.py \\
        --dataset data/test_mpfit.h5 \\
        --model-dir runs/test_multipoles \\
        --output-dir figures \\
        --n-molecules 20 \\
        --qm-method hf \\
        --qm-basis 6-31G*
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import h5py
import torch
import psi4
import tempfile
from tqdm import tqdm

from openff.toolkit import Molecule

from MMomentA.data.storage import load_dataset_hdf5
from MMomentA.data.dataset import ChargeDataset
from MMomentA.data.graph import molecule_data_to_dgl_graph
from MMomentA.data.schema import MoleculeData
from MMomentA.models import ChargeModel, ModelConfig
from MMomentA.qm.am1bcc import AM1BCCCalculator
from MMomentA.qm.resp import RESPCalculator
from esp_utils import compare_grid_esp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physical constants
BOHR_TO_ANGSTROM = 0.529177249


def generate_esp_grid(molecule: Molecule, conformer_idx: int = 0,
                      vdw_scale_factors: List[float] = [1.4, 1.6, 1.8, 2.0],
                      density: float = 0.15) -> np.ndarray:
    """Generate RESP-style ESP grid points around molecule.

    Args:
        density: Grid point density (points per Å²). Default 0.15 gives ~2000 points
                 for typical molecules. Original 1.0 gives ~16000 points (too many).
    """

    # VDW radii in Angstroms
    vdw_radii = {
        'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
        'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85
    }

    conformer = molecule.conformers[conformer_idx]
    coordinates = conformer.m_as('angstrom')

    grid_points = []

    for atom, coord in zip(molecule.atoms, coordinates):
        base_radius = vdw_radii.get(atom.symbol, 2.0)

        for scale in vdw_scale_factors:
            radius = base_radius * scale
            n_points = max(int(4 * np.pi * radius**2 * density), 20)

            # Fibonacci sphere
            phi = np.pi * (3.0 - np.sqrt(5.0))
            for i in range(n_points):
                y = 1 - (i / float(n_points - 1)) * 2
                r = np.sqrt(1 - y * y)
                theta = phi * i
                x = np.cos(theta) * r
                z = np.sin(theta) * r

                point = coord + np.array([x * radius, y * radius, z * radius])
                grid_points.append(point)

    return np.array(grid_points)


def compute_qm_esp_psi4(molecule: Molecule, grid_points: np.ndarray,
                        qm_method: str = 'hf', qm_basis: str = '6-31G*',
                        conformer_idx: int = 0) -> np.ndarray:
    """Compute QM ESP at grid points using Psi4's built-in ESP calculator."""

    conformer = molecule.conformers[conformer_idx]
    coords = conformer.m_as('angstrom')

    # Build Psi4 molecule string
    t_setup_start = time.time()
    mol_str = f"{molecule.total_charge.m} 1\n"
    for atom, coord in zip(molecule.atoms, coords):
        mol_str += f"{atom.symbol} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}\n"
    mol_str += "units angstrom\nno_reorient\nno_com\nsymmetry c1"

    # Create Psi4 molecule
    psi4.core.clean()
    psi4_mol = psi4.geometry(mol_str)

    # Set options
    psi4.set_options({
        'basis': qm_basis,
        'scf_type': 'df',
        'e_convergence': 1e-8,
        'd_convergence': 1e-8
    })

    # Force Psi4 to use only 1 thread (critical for parallel workers)
    psi4.set_num_threads(1)
    print(f"[DEBUG]     * Psi4 setup: {time.time() - t_setup_start:.2f}s")

    # Compute wavefunction
    t_scf_start = time.time()
    psi4.core.set_output_file('psi4_output.dat', False)
    energy, wfn = psi4.energy(qm_method, return_wfn=True, molecule=psi4_mol)
    print(f"[DEBUG]     * SCF calculation: {time.time() - t_scf_start:.2f}s")

    # Use Psi4's built-in ESP calculator at our grid points
    # This computes full QM ESP (nuclear + electronic contributions)
    t_esp_calc_start = time.time()

    # Convert grid points to Psi4 Matrix (in Bohr)
    grid_bohr = grid_points / BOHR_TO_ANGSTROM

    # Get ESP calculator
    Vpot = psi4.core.VBase.build(wfn.basisset(), "RV")
    Vpot.initialize()

    # Convert all grid points to Psi4 Vector3 list for batch processing
    print(f"[DEBUG]     * Converting {len(grid_points)} points to Psi4 format...")
    t_convert_start = time.time()
    psi4_points = [psi4.core.Vector3(pt[0], pt[1], pt[2]) for pt in grid_bohr]
    print(f"[DEBUG]     * Conversion took: {time.time() - t_convert_start:.2f}s")

    # Compute ESP at all grid points in one call (batch processing)
    print(f"[DEBUG]     * Computing ESP at {len(grid_points)} points in batch...")
    t_batch_start = time.time()
    esp_values = np.array(Vpot.compute_esp(wfn.Da(), psi4_points))
    print(f"[DEBUG]     * Batch ESP computation: {time.time() - t_batch_start:.2f}s")

    print(f"[DEBUG]     * Total ESP evaluation: {time.time() - t_esp_calc_start:.2f}s")

    return esp_values


def calculate_esp_from_charges(coordinates: np.ndarray, charges: np.ndarray,
                               grid_points: np.ndarray) -> np.ndarray:
    """Calculate ESP at grid points from atomic charges.

    Args:
        coordinates: Atomic coordinates in Angstroms
        charges: Atomic partial charges in elementary charge units
        grid_points: ESP grid points in Angstroms

    Returns:
        ESP values in atomic units (hartree/e)
    """
    esp_values = np.zeros(len(grid_points))

    for i, point in enumerate(grid_points):
        for coord, charge in zip(coordinates, charges):
            distance_angstrom = np.linalg.norm(point - coord)
            if distance_angstrom > 1e-6:
                # Convert distance to bohr, then ESP = charge / distance_bohr
                distance_bohr = distance_angstrom / BOHR_TO_ANGSTROM
                esp_values[i] += charge / distance_bohr

    return esp_values


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
        # Fallback: infer from state dict
        logger.warning("No model config in checkpoint, using default")
        model_config = ModelConfig()

    model = ChargeModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded (epoch {checkpoint['epoch']})")

    return model


def predict_charges(model: ChargeModel, mol_data: MoleculeData,
                    device: str = 'cpu') -> np.ndarray:
    """Predict charges for a molecule using trained model."""

    # Convert MoleculeData to DGL graph
    # Check if model uses multipoles by checking config
    include_multipoles = getattr(model.config, 'include_multipoles', True)
    graph = molecule_data_to_dgl_graph(mol_data, include_multipoles=include_multipoles)
    graph = graph.to(device)

    # Predict
    with torch.no_grad():
        graph = model(graph)
        predicted_charges = graph.ndata["q"].cpu().numpy().flatten()

    return predicted_charges


def validate_single_molecule(idx: int, mol_data: MoleculeData, model_dir: Path,
                             qm_method: str, qm_basis: str, device: str) -> dict:
    """Validate a single molecule (for parallel processing).

    This function is designed to be called in parallel - it loads its own model
    and performs all validations independently.
    """
    import os
    import tempfile
    from openff.toolkit import Molecule
    from openff.units import unit

    # [DEBUG] Track timing for this molecule
    t_molecule_start = time.time()
    print(f"[DEBUG] Molecule {idx}: Starting validation")

    # Force single-threaded execution to avoid thread oversubscription
    # when running many workers in parallel
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # Load model (each worker loads its own copy)
    t_load_start = time.time()
    model = load_model(model_dir, device=device)
    print(f"[DEBUG] Molecule {idx}: Model loaded in {time.time() - t_load_start:.2f}s")

    try:
        mpfit_charges = mol_data.target_charges
        mpfit_time = mol_data.qm_metadata.get('calculation_time', None) if mol_data.qm_metadata else None

        # Predict charges with GNN
        t_gnn_start = time.time()
        gnn_charges = predict_charges(model, mol_data, device=device)
        gnn_inference_time = time.time() - t_gnn_start
        print(f"[DEBUG] Molecule {idx}: GNN prediction in {gnn_inference_time:.2f}s")

        # Reconstruct molecule
        t_mol_start = time.time()
        molecule = Molecule.from_smiles(mol_data.smiles, allow_undefined_stereo=True)
        print(f"[DEBUG] Molecule {idx}: Molecule from SMILES in {time.time() - t_mol_start:.2f}s")

        if not molecule.conformers:
            if mol_data.conformer is not None:
                molecule.add_conformer(mol_data.conformer * unit.angstrom)
            else:
                molecule.generate_conformers(n_conformers=1)
        elif mol_data.conformer is not None:
            molecule.conformers[0] = mol_data.conformer * unit.angstrom

        # Use temporary directory for Psi4 scratch files
        print(f"[DEBUG] Molecule {idx}: Starting ESP validation")
        t_esp_start = time.time()
        with tempfile.TemporaryDirectory(prefix=f'esp_val_{idx}_') as temp_dir:
            os.environ['PSI_SCRATCH'] = temp_dir

            # Validate ESP
            validation = validate_molecule_esp(
                molecule, mpfit_charges, gnn_charges,
                qm_method=qm_method,
                qm_basis=qm_basis,
                include_am1bcc=True,
                include_resp=True
            )
        print(f"[DEBUG] Molecule {idx}: ESP validation completed in {time.time() - t_esp_start:.2f}s")

        # Extract atomic numbers for carbon filtering
        atomic_numbers = mol_data.atomic_features['atomic_numbers']
        carbon_mask = (atomic_numbers == 6)

        total_time = time.time() - t_molecule_start
        print(f"[DEBUG] Molecule {idx}: TOTAL TIME = {total_time:.2f}s")

        return {
            'idx': idx,
            'success': validation['success'],
            'validation': validation,
            'mpfit_time': mpfit_time,
            'gnn_inference_time': gnn_inference_time,
            'mpfit_charges': mpfit_charges,
            'gnn_charges': gnn_charges,
            'carbon_mask': carbon_mask
        }

    except Exception as e:
        logger.error(f"Failed to validate molecule {idx}: {e}")
        print(f"[DEBUG] Molecule {idx}: FAILED after {time.time() - t_molecule_start:.2f}s")
        return {
            'idx': idx,
            'success': False,
            'error': str(e)
        }


def compute_am1bcc_charges(molecule: Molecule) -> np.ndarray:
    """Compute AM1-BCC charges for a molecule."""
    calculator = AM1BCCCalculator()
    result = calculator.compute(molecule)
    if result.success:
        return result.charges
    else:
        raise ValueError(f"AM1-BCC failed: {result.error_message}")


def compute_resp_charges(molecule: Molecule, qm_method: str = 'hf',
                        qm_basis: str = '6-31G*', conformer_idx: int = 0) -> np.ndarray:
    """Compute RESP charges for a molecule.

    Note: Uses minimize=False since molecules already have MPFIT-optimized geometries.
    This avoids convergence failures on difficult molecules.
    """
    from MMomentA.qm.resp import ESPConfig
    config = ESPConfig(method=qm_method, basis=qm_basis, minimize=False)
    calculator = RESPCalculator(config=config)
    result = calculator.compute(molecule)
    if result.success:
        return result.charges
    else:
        raise ValueError(f"RESP failed: {result.error_message}")


def validate_molecule_esp(molecule: Molecule, mpfit_charges: np.ndarray,
                          gnn_charges: np.ndarray, qm_method: str = 'hf',
                          qm_basis: str = '6-31G*', conformer_idx: int = 0,
                          include_am1bcc: bool = True, include_resp: bool = True) -> Dict:
    """Validate ESP reproduction for MPFIT, AM1-BCC, RESP, and GNN charges."""

    try:
        # Generate ESP grid
        t_grid_start = time.time()
        grid_points = generate_esp_grid(molecule, conformer_idx=conformer_idx)
        print(f"[DEBUG]   - Grid generation: {time.time() - t_grid_start:.2f}s ({len(grid_points)} points)")

        # Compute QM ESP
        t_qm_start = time.time()
        qm_esp = compute_qm_esp_psi4(molecule, grid_points, qm_method, qm_basis, conformer_idx)
        print(f"[DEBUG]   - QM ESP computation: {time.time() - t_qm_start:.2f}s")

        # Get coordinates
        conformer = molecule.conformers[conformer_idx]
        coords = conformer.m_as('angstrom')

        # Calculate ESP from MPFIT charges
        t_mpfit_start = time.time()
        mpfit_esp = calculate_esp_from_charges(coords, mpfit_charges, grid_points)
        mpfit_metrics = compare_grid_esp(qm_esp, mpfit_esp, verbose=False)
        print(f"[DEBUG]   - MPFIT ESP calc: {time.time() - t_mpfit_start:.2f}s")

        # Calculate ESP from GNN charges (time this since it's inference)
        t_gnn_start = time.time()
        gnn_esp = calculate_esp_from_charges(coords, gnn_charges, grid_points)
        gnn_metrics = compare_grid_esp(qm_esp, gnn_esp, verbose=False)
        gnn_metrics['time'] = time.time() - t_gnn_start
        print(f"[DEBUG]   - GNN ESP calc: {time.time() - t_gnn_start:.2f}s")

        result = {
            'success': True,
            'mpfit': mpfit_metrics,
            'gnn': gnn_metrics,
            'n_grid_points': len(grid_points)
        }

        # Optionally compute AM1-BCC charges and ESP
        if include_am1bcc:
            try:
                print(f"[DEBUG]   - Computing AM1-BCC charges...")
                t_am1bcc_start = time.time()
                am1bcc_charges = compute_am1bcc_charges(molecule)
                am1bcc_esp = calculate_esp_from_charges(coords, am1bcc_charges, grid_points)
                am1bcc_metrics = compare_grid_esp(qm_esp, am1bcc_esp, verbose=False)
                am1bcc_metrics['time'] = time.time() - t_am1bcc_start
                result['am1bcc'] = am1bcc_metrics
                print(f"[DEBUG]   - AM1-BCC completed: {am1bcc_metrics['time']:.2f}s")
            except Exception as e:
                logger.warning(f"AM1-BCC failed: {e}")
                result['am1bcc'] = None

        # Optionally compute RESP charges and ESP
        if include_resp:
            try:
                print(f"[DEBUG]   - Computing RESP charges...")
                t_resp_start = time.time()
                resp_charges = compute_resp_charges(molecule, qm_method, qm_basis, conformer_idx)
                resp_esp = calculate_esp_from_charges(coords, resp_charges, grid_points)
                resp_metrics = compare_grid_esp(qm_esp, resp_esp, verbose=False)
                resp_metrics['time'] = time.time() - t_resp_start
                result['resp'] = resp_metrics
                print(f"[DEBUG]   - RESP completed: {resp_metrics['time']:.2f}s")
            except Exception as e:
                logger.warning(f"RESP failed: {e}")
                result['resp'] = None

        return result

    except Exception as e:
        logger.warning(f"ESP validation failed for molecule: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Validate ESP reproduction: MPFIT vs MMomentA-GNN")

    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to HDF5 dataset with MPFIT charges")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Directory containing trained model")
    parser.add_argument("--output-dir", type=str, default="figures",
                       help="Output directory for results")
    parser.add_argument("--n-molecules", type=int, default=None,
                       help="Number of test molecules to validate (default: all test molecules)")
    parser.add_argument("--qm-method", type=str, default="hf",
                       help="QM method for ESP calculation")
    parser.add_argument("--qm-basis", type=str, default="6-31G*",
                       help="Basis set for ESP calculation")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for model inference")
    parser.add_argument("--n-jobs", type=int, default=8,
                       help="Number of parallel jobs for ESP validation (default: 8, too many causes Psi4 deadlock)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("ESP Validation: MPFIT vs MMomentA-GNN")
    logger.info("="*60)

    # Load model
    model = load_model(Path(args.model_dir), device=args.device)

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    molecule_data_list, metadata = load_dataset_hdf5(args.dataset)

    # Get test molecules
    if metadata and 'splits' in metadata:
        # Metadata contains molecule IDs, not indices - need to convert
        test_molecule_ids = metadata['splits'].get('test', [])
        if test_molecule_ids:
            # Create mapping from molecule_id to index
            id_to_idx = {mol.molecule_id: idx for idx, mol in enumerate(molecule_data_list)}
            test_indices = [id_to_idx[mol_id] for mol_id in test_molecule_ids if mol_id in id_to_idx]
        else:
            test_indices = list(range(len(molecule_data_list)))
    else:
        # Fallback: validate all molecules if no split info
        logger.warning("No split information found in metadata - validating all molecules")
        test_indices = list(range(len(molecule_data_list)))

    # Optionally limit number of molecules
    if args.n_molecules is not None:
        test_indices = test_indices[:args.n_molecules]

    logger.info(f"Validating {len(test_indices)} test molecules")
    logger.info(f"QM method: {args.qm_method}/{args.qm_basis}")
    logger.info(f"Running in parallel with {args.n_jobs} jobs")

    # Validate each molecule in parallel
    from joblib import Parallel, delayed

    # Pass model directory (not checkpoint path) - load_model() will append the checkpoint path
    model_dir = Path(args.model_dir)

    logger.info("Starting parallel ESP validation...")
    validation_results = Parallel(n_jobs=args.n_jobs, backend='multiprocessing', verbose=10)(
        delayed(validate_single_molecule)(
            idx, molecule_data_list[idx], model_dir,
            args.qm_method, args.qm_basis, args.device
        )
        for idx in test_indices
    )

    # Aggregate results
    results = {
        'mpfit': {'mae': [], 'rmse': [], 'time': []},
        'am1bcc': {'mae': [], 'rmse': [], 'time': []},
        'resp': {'mae': [], 'rmse': [], 'time': []},
        'gnn': {'mae': [], 'rmse': [], 'time': []}
    }

    carbon_charges_ref = []
    carbon_charges_pred = []
    successful = 0
    failed = 0

    for result in validation_results:
        if not result['success']:
            failed += 1
            continue

        validation = result['validation']
        mpfit_time = result['mpfit_time']
        gnn_inference_time = result['gnn_inference_time']

        results['mpfit']['mae'].append(validation['mpfit']['mae'])
        results['mpfit']['rmse'].append(validation['mpfit']['rmse'])
        if mpfit_time is not None:
            results['mpfit']['time'].append(mpfit_time)

        if 'am1bcc' in validation and validation['am1bcc'] is not None:
            results['am1bcc']['mae'].append(validation['am1bcc']['mae'])
            results['am1bcc']['rmse'].append(validation['am1bcc']['rmse'])
            results['am1bcc']['time'].append(validation['am1bcc']['time'])

        if 'resp' in validation and validation['resp'] is not None:
            results['resp']['mae'].append(validation['resp']['mae'])
            results['resp']['rmse'].append(validation['resp']['rmse'])
            results['resp']['time'].append(validation['resp']['time'])

        results['gnn']['mae'].append(validation['gnn']['mae'])
        results['gnn']['rmse'].append(validation['gnn']['rmse'])
        results['gnn']['time'].append(gnn_inference_time + validation['gnn']['time'])

        # Collect carbon charges
        mpfit_charges = result['mpfit_charges']
        gnn_charges = result['gnn_charges']
        carbon_mask = result['carbon_mask']

        carbon_charges_ref.extend(mpfit_charges[carbon_mask])
        carbon_charges_pred.extend(gnn_charges[carbon_mask])

        successful += 1

    logger.info("="*60)
    logger.info(f"ESP Validation Complete")
    logger.info(f"Successful: {successful}/{len(test_indices)}")
    logger.info(f"Failed: {failed}/{len(test_indices)}")
    logger.info("="*60)

    # Print summary statistics
    print("\n" + "="*70)
    print("ESP VALIDATION RESULTS")
    print("="*70)

    print("\nMPFIT ESP Validation:")
    print(f"  MAE:  {np.mean(results['mpfit']['mae']):.6e} ± {np.std(results['mpfit']['mae']):.6e} a.u.")
    print(f"  RMSE: {np.mean(results['mpfit']['rmse']):.6e} ± {np.std(results['mpfit']['rmse']):.6e} a.u.")
    if results['mpfit']['time']:
        print(f"  Time: {np.mean(results['mpfit']['time']):.3f} ± {np.std(results['mpfit']['time']):.3f} s/molecule")

    if results['am1bcc']['mae']:
        print("\nAM1-BCC ESP Validation:")
        print(f"  MAE:  {np.mean(results['am1bcc']['mae']):.6e} ± {np.std(results['am1bcc']['mae']):.6e} a.u.")
        print(f"  RMSE: {np.mean(results['am1bcc']['rmse']):.6e} ± {np.std(results['am1bcc']['rmse']):.6e} a.u.")
        if results['am1bcc']['time']:
            print(f"  Time: {np.mean(results['am1bcc']['time']):.3f} ± {np.std(results['am1bcc']['time']):.3f} s/molecule")

    if results['resp']['mae']:
        print("\nRESP ESP Validation:")
        print(f"  MAE:  {np.mean(results['resp']['mae']):.6e} ± {np.std(results['resp']['mae']):.6e} a.u.")
        print(f"  RMSE: {np.mean(results['resp']['rmse']):.6e} ± {np.std(results['resp']['rmse']):.6e} a.u.")
        if results['resp']['time']:
            print(f"  Time: {np.mean(results['resp']['time']):.3f} ± {np.std(results['resp']['time']):.3f} s/molecule")

    print("\nMMomentA-GNN ESP Validation:")
    print(f"  MAE:  {np.mean(results['gnn']['mae']):.6e} ± {np.std(results['gnn']['mae']):.6e} a.u.")
    print(f"  RMSE: {np.mean(results['gnn']['rmse']):.6e} ± {np.std(results['gnn']['rmse']):.6e} a.u.")
    if results['gnn']['time']:
        print(f"  Time: {np.mean(results['gnn']['time']):.3f} ± {np.std(results['gnn']['time']):.3f} s/molecule")

    print("\n" + "="*70)
    print("TIMING COMPARISON (seconds per molecule)")
    print("="*70)
    if results['mpfit']['time']:
        print(f"MPFIT:        {np.mean(results['mpfit']['time']):8.3f} ± {np.std(results['mpfit']['time']):6.3f}")
    if results['am1bcc']['time']:
        print(f"AM1-BCC:      {np.mean(results['am1bcc']['time']):8.3f} ± {np.std(results['am1bcc']['time']):6.3f}")
    if results['resp']['time']:
        print(f"RESP:         {np.mean(results['resp']['time']):8.3f} ± {np.std(results['resp']['time']):6.3f}")
    if results['gnn']['time']:
        print(f"MMomentA-GNN: {np.mean(results['gnn']['time']):8.3f} ± {np.std(results['gnn']['time']):6.3f}")

        # Calculate speedup
        if results['mpfit']['time']:
            speedup = np.mean(results['mpfit']['time']) / np.mean(results['gnn']['time'])
            print(f"\nSpeedup over MPFIT: {speedup:.1f}x")
    print("="*70)

    # Save results
    results_file = output_dir / "esp_validation_results.json"
    summary = {
        'mpfit': {
            'mae_mean': float(np.mean(results['mpfit']['mae'])),
            'mae_std': float(np.std(results['mpfit']['mae'])),
            'rmse_mean': float(np.mean(results['mpfit']['rmse'])),
            'rmse_std': float(np.std(results['mpfit']['rmse'])),
            'time_mean': float(np.mean(results['mpfit']['time'])) if results['mpfit']['time'] else None,
            'time_std': float(np.std(results['mpfit']['time'])) if results['mpfit']['time'] else None
        },
        'gnn': {
            'mae_mean': float(np.mean(results['gnn']['mae'])),
            'mae_std': float(np.std(results['gnn']['mae'])),
            'rmse_mean': float(np.mean(results['gnn']['rmse'])),
            'rmse_std': float(np.std(results['gnn']['rmse'])),
            'time_mean': float(np.mean(results['gnn']['time'])) if results['gnn']['time'] else None,
            'time_std': float(np.std(results['gnn']['time'])) if results['gnn']['time'] else None
        }
    }

    if results['am1bcc']['mae']:
        summary['am1bcc'] = {
            'mae_mean': float(np.mean(results['am1bcc']['mae'])),
            'mae_std': float(np.std(results['am1bcc']['mae'])),
            'rmse_mean': float(np.mean(results['am1bcc']['rmse'])),
            'rmse_std': float(np.std(results['am1bcc']['rmse'])),
            'time_mean': float(np.mean(results['am1bcc']['time'])) if results['am1bcc']['time'] else None,
            'time_std': float(np.std(results['am1bcc']['time'])) if results['am1bcc']['time'] else None
        }

    if results['resp']['mae']:
        summary['resp'] = {
            'mae_mean': float(np.mean(results['resp']['mae'])),
            'mae_std': float(np.std(results['resp']['mae'])),
            'rmse_mean': float(np.mean(results['resp']['rmse'])),
            'rmse_std': float(np.std(results['resp']['rmse'])),
            'time_mean': float(np.mean(results['resp']['time'])) if results['resp']['time'] else None,
            'time_std': float(np.std(results['resp']['time'])) if results['resp']['time'] else None
        }

    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'dataset': args.dataset,
                'model_dir': args.model_dir,
                'n_molecules': len(test_indices),
                'qm_method': args.qm_method,
                'qm_basis': args.qm_basis
            },
            'metrics': results,
            'summary': summary
        }, f, indent=2)

    logger.info(f"\n✓ Results saved to {results_file}")

    # Generate plots
    logger.info("\nGenerating comparison plots...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'figures'))
    from plot_comparison import create_esp_violin_plot, create_carbon_charge_hexbin

    # ESP violin plot
    plot_file = create_esp_violin_plot(results, output_dir=output_dir,
                                       qm_method=args.qm_method, qm_basis=args.qm_basis)
    logger.info(f"✓ ESP violin plot saved to {plot_file}")

    # Carbon charge hexbin plot
    if len(carbon_charges_ref) > 0:
        logger.info(f"\nGenerating carbon charge hexbin plot ({len(carbon_charges_ref)} carbon atoms)...")
        hexbin_file = create_carbon_charge_hexbin(
            np.array(carbon_charges_ref),
            np.array(carbon_charges_pred),
            output_dir=output_dir
        )
        logger.info(f"✓ Carbon hexbin plot saved to {hexbin_file}")
    else:
        logger.warning("No carbon atoms found in dataset - skipping hexbin plot")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
