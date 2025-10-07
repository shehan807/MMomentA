#!/usr/bin/env python
"""Prepare ML training dataset with smart caching.

This wrapper around prepare_dataset.py adds caching capability:
- Checks if output HDF5 exists
- Loads existing molecules and their SMILES
- Only computes MPFIT for new molecules not in cache
- Merges cached + new results
- Atomic save (temp file + rename)

Examples
--------
# First run: computes all molecules
python prepare_dataset_cached.py \\
    --input spice.pkl \\
    --output spice_mpfit.h5 \\
    --n-jobs 16

# Second run with same molecules: instant (all cached)
python prepare_dataset_cached.py \\
    --input spice.pkl \\
    --output spice_mpfit.h5 \\
    --n-jobs 16

# Run with additional molecules: only computes new ones
python prepare_dataset_cached.py \\
    --input spice_expanded.pkl \\  # 200 molecules instead of 100
    --output spice_mpfit.h5 \\
    --n-jobs 16  # Only computes the 100 new molecules
"""

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple

from openff.toolkit import Molecule

from MMomentA.qm import MPFITCalculator, GDMAConfig
from MMomentA.data.loaders import load_molecules_from_file
from MMomentA.data.batch import BatchProcessor
from MMomentA.data.extraction import extract_molecule_data
from MMomentA.data.splitting import scaffold_split, random_split
from MMomentA.data.storage import load_dataset, save_dataset
from MMomentA.data.schema import DatasetMetadata, MoleculeData
from MMomentA.utils import setup_logging
from MMomentA.qm.mpfit import MPFITResult
import numpy as np


def load_cached_data(cache_file: Path) -> Tuple[List[MoleculeData], Set[str], dict]:
    """Load cached dataset and extract SMILES set for deduplication.

    Parameters
    ----------
    cache_file : Path
        Path to cached HDF5 file

    Returns
    -------
    cached_data : list of MoleculeData
        Previously computed molecular data
    cached_smiles : set of str
        SMILES strings of cached molecules
    metadata : dict
        Dataset metadata from cache
    """
    logger = logging.getLogger(__name__)

    try:
        # Load using MMomentA's storage utilities
        from MMomentA.data.storage import load_dataset_hdf5

        molecule_data_list, metadata = load_dataset_hdf5(cache_file)

        # Extract SMILES for deduplication
        cached_smiles = {mol_data.smiles for mol_data in molecule_data_list}

        logger.info(f"Loaded {len(molecule_data_list)} cached molecules")
        logger.info(f"Unique SMILES in cache: {len(cached_smiles)}")

        return molecule_data_list, cached_smiles, metadata

    except Exception as e:
        logger.warning(f"Could not load cache: {e}")
        return [], set(), None


def filter_new_molecules(
    molecules: List[Molecule],
    cached_smiles: Set[str]
) -> Tuple[List[Molecule], List[int]]:
    """Filter out molecules already in cache.

    Parameters
    ----------
    molecules : list of Molecule
        All input molecules
    cached_smiles : set of str
        SMILES of molecules already computed

    Returns
    -------
    new_molecules : list of Molecule
        Molecules not in cache
    new_indices : list of int
        Original indices of new molecules
    """
    logger = logging.getLogger(__name__)

    new_molecules = []
    new_indices = []

    for i, mol in enumerate(molecules):
        smiles = mol.to_smiles(mapped=False)
        if smiles not in cached_smiles:
            new_molecules.append(mol)
            new_indices.append(i)

    logger.info(f"Input molecules: {len(molecules)}")
    logger.info(f"Cached molecules: {len(molecules) - len(new_molecules)}")
    logger.info(f"New molecules to compute: {len(new_molecules)}")

    return new_molecules, new_indices


def compute_new_molecules(
    molecules: List[Molecule],
    config: GDMAConfig,
    n_jobs: int,
    backend: str,
    print_charges: bool = False
) -> List[MoleculeData]:
    """Compute MPFIT charges for new molecules.

    Parameters
    ----------
    molecules : list of Molecule
        Molecules to process
    config : GDMAConfig
        GDMA configuration
    n_jobs : int
        Number of parallel jobs
    backend : str
        Joblib backend
    print_charges : bool
        Print charges for each molecule

    Returns
    -------
    molecule_data_list : list of MoleculeData
        Computed molecular data
    """
    logger = logging.getLogger(__name__)

    if not molecules:
        logger.info("No new molecules to compute")
        return []

    calculator = MPFITCalculator(config)
    calculators = {"MPFIT": calculator}

    processor = BatchProcessor(
        calculators=calculators,
        n_jobs=n_jobs,
        backend=backend
    )

    logger.info(f"Computing MPFIT charges for {len(molecules)} new molecules...")
    batch_results = processor.process(molecules)

    successful_molecules = []
    successful_results = []

    if print_charges:
        logger.info("")
        logger.info("=" * 70)
        logger.info("MPFIT Charges for New Molecules")
        logger.info("=" * 70)

    for i, result in enumerate(batch_results):
        if result.success and "MPFIT" in result.results:
            successful_molecules.append(molecules[i])
            mpfit_data = result.results["MPFIT"]

            # Convert conformer and create MPFITResult
            conformer = mpfit_data['conformer']

            # Strip units
            def strip_units(obj):
                if hasattr(obj, 'magnitude'):
                    return strip_units(obj.magnitude)
                elif hasattr(obj, 'm'):
                    return strip_units(obj.m)
                elif isinstance(obj, np.ndarray):
                    return [[float(strip_units(x)) for x in row] if hasattr(row, '__iter__') and not isinstance(row, str)
                            else float(strip_units(row)) for row in obj]
                elif isinstance(obj, (list, tuple)):
                    return [strip_units(x) for x in obj]
                else:
                    return float(obj) if not isinstance(obj, str) else obj

            conformer_clean = strip_units(conformer)
            conformer = np.array(conformer_clean, dtype=float)

            mpfit_result = MPFITResult(
                charges=np.array(mpfit_data['charges']),
                multipole_moments=np.array(mpfit_data['multipole_moments']),
                conformer=conformer,
                time_seconds=mpfit_data['time'],
                smiles=mpfit_data['smiles'],
                metadata=mpfit_data['metadata'],
                success=True,
                error_message=None
            )
            successful_results.append(mpfit_result)

            if print_charges:
                smiles = mpfit_data['smiles']
                charges = mpfit_result.charges
                charges_str = ' '.join([f'{c:7.4f}' for c in charges])
                logger.info(f"{i+1:3d}. {smiles:20s} [{charges_str}]")

    logger.info(f"Successfully processed {len(successful_molecules)}/{len(molecules)} new molecules")

    # Extract features
    logger.info("Extracting molecular features...")
    molecule_data_list = []

    for mol, mpfit_result in zip(successful_molecules, successful_results):
        mol_data = extract_molecule_data(mol, mpfit_result)
        molecule_data_list.append(mol_data)

    logger.info(f"Extracted features for {len(molecule_data_list)} molecules")

    return molecule_data_list


def merge_datasets(
    cached_data: List[MoleculeData],
    new_data: List[MoleculeData]
) -> List[MoleculeData]:
    """Merge cached and newly computed data.

    Parameters
    ----------
    cached_data : list of MoleculeData
        Previously cached data
    new_data : list of MoleculeData
        Newly computed data

    Returns
    -------
    merged_data : list of MoleculeData
        Combined dataset
    """
    logger = logging.getLogger(__name__)

    merged = cached_data + new_data

    logger.info(f"Merged dataset:")
    logger.info(f"  Cached: {len(cached_data)}")
    logger.info(f"  New: {len(new_data)}")
    logger.info(f"  Total: {len(merged)}")

    return merged


def atomic_save(
    molecule_data_list: List[MoleculeData],
    output_path: Path,
    metadata: DatasetMetadata
):
    """Save dataset atomically (temp file + rename).

    Parameters
    ----------
    molecule_data_list : list of MoleculeData
        Molecular data to save
    output_path : Path
        Final output path
    metadata : DatasetMetadata
        Dataset metadata
    """
    logger = logging.getLogger(__name__)

    # Save to temp file first
    temp_path = output_path.with_suffix('.tmp.h5')

    logger.info(f"Saving to temporary file: {temp_path}")
    save_dataset(molecule_data_list, str(temp_path), metadata)

    # Atomic rename
    logger.info(f"Atomic rename to: {output_path}")
    shutil.move(str(temp_path), str(output_path))

    logger.info(f"✓ Dataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ML dataset with smart caching"
    )

    # Same arguments as prepare_dataset.py
    parser.add_argument("--input", type=str, required=True, help="Input molecule file")
    parser.add_argument("--output", type=str, required=True, help="Output dataset file (.h5)")
    parser.add_argument("--dataset-name", type=str, default="Custom", help="Dataset name")
    parser.add_argument("--max-molecules", type=int, default=None, help="Max molecules to process")

    parser.add_argument("--method", type=str, default="hf", help="QM method")
    parser.add_argument("--basis", type=str, default="6-31G*", help="Basis set")
    parser.add_argument("--limit", type=int, default=8, help="Multipole expansion limit")
    parser.add_argument("--svd-threshold", type=float, default=1.0e-4, help="SVD threshold")

    parser.add_argument("--split-strategy", choices=["scaffold", "random"], default="scaffold")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=2666)

    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--backend", choices=["loky", "multiprocessing"], default="loky")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially")

    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    parser.add_argument("--print-charges", action="store_true", help="Print charges")

    # Caching options
    parser.add_argument("--force-recompute", action="store_true",
                       help="Ignore cache and recompute all molecules")

    args = parser.parse_args()

    logger = setup_logging(level=args.log_level, log_file=args.log_file)

    logger.info("=" * 70)
    logger.info("ML Dataset Preparation (with Smart Caching)")
    logger.info("=" * 70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Force recompute: {args.force_recompute}")
    logger.info("=" * 70)

    output_path = Path(args.output)

    # Load input molecules
    molecules = load_molecules_from_file(args.input, max_molecules=args.max_molecules)
    logger.info(f"Loaded {len(molecules)} molecules from input")

    # Check cache
    cached_data = []
    cached_smiles = set()
    cached_metadata = None

    if output_path.exists() and not args.force_recompute:
        logger.info(f"Found existing cache: {output_path}")
        cached_data, cached_smiles, cached_metadata = load_cached_data(output_path)
    else:
        if args.force_recompute:
            logger.info("Force recompute enabled - ignoring cache")
        else:
            logger.info("No cache found - will compute all molecules")

    # Filter new molecules
    new_molecules, new_indices = filter_new_molecules(molecules, cached_smiles)

    # Compute new molecules if any
    new_data = []
    if new_molecules:
        config = GDMAConfig(
            method=args.method,
            basis=args.basis,
            limit=args.limit,
            svd_threshold=args.svd_threshold
        )

        n_jobs = 1 if args.sequential else args.n_jobs

        new_data = compute_new_molecules(
            new_molecules,
            config,
            n_jobs,
            args.backend,
            args.print_charges
        )

    # Merge datasets
    merged_data = merge_datasets(cached_data, new_data)

    if not merged_data:
        logger.error("No molecules successfully processed!")
        return

    # Re-split dataset (important: splits may change with new molecules)
    logger.info(f"Splitting dataset using {args.split_strategy} strategy...")

    # Reconstruct molecules from merged data (for scaffold splitting)
    merged_molecules = []
    for mol_data in merged_data:
        mol = Molecule.from_smiles(mol_data.smiles, allow_undefined_stereo=True)
        if mol.conformers is None or len(mol.conformers) == 0:
            # Add conformer from mol_data
            from openff.units import unit
            conformer_with_units = mol_data.conformer * unit.angstrom
            mol.add_conformer(conformer_with_units)
        merged_molecules.append(mol)

    if args.split_strategy == "scaffold":
        train_idx, val_idx, test_idx = scaffold_split(
            merged_molecules,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_seed=args.random_seed
        )
    else:
        train_idx, val_idx, test_idx = random_split(
            len(merged_data),
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_seed=args.random_seed
        )

    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Create metadata
    metadata = DatasetMetadata(
        dataset_name=args.dataset_name,
        n_molecules=len(merged_data),
        qm_method=args.method,
        qm_basis=args.basis,
        multipole_limit=args.limit,
        created_date=datetime.now().isoformat(),
        split_strategy=args.split_strategy,
        splits={
            'train': [merged_data[i].molecule_id for i in train_idx],
            'val': [merged_data[i].molecule_id for i in val_idx],
            'test': [merged_data[i].molecule_id for i in test_idx]
        }
    )

    # Atomic save
    logger.info(f"Saving dataset...")
    atomic_save(merged_data, output_path, metadata)

    logger.info("Dataset preparation complete!")
    logger.info(f"Output: {output_path}")
    logger.info(f"Total molecules: {len(merged_data)} ({len(cached_data)} cached + {len(new_data)} new)")


if __name__ == "__main__":
    main()
