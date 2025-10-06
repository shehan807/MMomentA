#!/usr/bin/env python
"""Prepare ML training dataset from MPFIT calculations.

This script:
1. Loads molecules from input file
2. Computes MPFIT charges and multipole moments
3. Extracts features for ML training
4. Splits into train/val/test sets
5. Saves structured dataset for training

Examples
--------
# Prepare SPICE dataset
python prepare_dataset.py \\
    --input spice.oeb \\
    --output spice_mpfit_dataset.h5 \\
    --max-molecules 1000 \\
    --method hf \\
    --basis "6-31G*" \\
    --split-strategy scaffold

# Prepare ZINC dataset
python prepare_dataset.py \\
    --input zinc_molecules.pkl \\
    --output zinc_mpfit_dataset.h5 \\
    --max-molecules 500 \\
    --method pbe0 \\
    --basis def2-SVP \\
    --n-jobs 8
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from MMomentA.qm import MPFITCalculator, GDMAConfig
from MMomentA.data.loaders import load_molecules_from_file
from MMomentA.data.batch import BatchProcessor
from MMomentA.data.extraction import extract_molecule_data
from MMomentA.data.splitting import scaffold_split, random_split
from MMomentA.data.storage import save_dataset
from MMomentA.data.schema import DatasetMetadata
from MMomentA.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ML training dataset from MPFIT calculations"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input molecule file (SDF, OEB, PKL, etc.)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output dataset file (.h5, .pkl, or .json)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Custom",
        help="Dataset name (e.g., 'SPICE', 'ZINC')"
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        help="Maximum number of molecules to process"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="hf",
        help="QM method (default: hf)"
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="6-31G*",
        help="Basis set (default: 6-31G*)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Multipole expansion limit (default: 8)"
    )
    parser.add_argument(
        "--svd-threshold",
        type=float,
        default=1.0e-4,
        help="SVD threshold for MPFIT solver (default: 1.0e-4)"
    )

    parser.add_argument(
        "--split-strategy",
        choices=["scaffold", "random"],
        default="scaffold",
        help="Dataset splitting strategy (default: scaffold)"
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Training set fraction (default: 0.8)"
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation set fraction (default: 0.1)"
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Test set fraction (default: 0.1)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=2666,
        help="Random seed for splitting (default: 2666)"
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1, use all CPUs)"
    )
    parser.add_argument(
        "--backend",
        choices=["loky", "multiprocessing"],
        default="loky",
        help="Joblib backend (default: loky)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially instead of in parallel"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (default: console only)"
    )
    parser.add_argument(
        "--print-charges",
        action="store_true",
        help="Print SMILES and charges for each molecule"
    )

    args = parser.parse_args()

    logger = setup_logging(level=args.log_level, log_file=args.log_file)

    logger.info("=" * 70)
    logger.info("ML Dataset Preparation")
    logger.info("=" * 70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"QM method: {args.method}/{args.basis}")
    logger.info(f"Split strategy: {args.split_strategy}")
    logger.info("=" * 70)

    molecules = load_molecules_from_file(
        args.input,
        max_molecules=args.max_molecules
    )

    logger.info(f"Loaded {len(molecules)} molecules")

    config = GDMAConfig(
        method=args.method,
        basis=args.basis,
        limit=args.limit,
        svd_threshold=args.svd_threshold
    )

    calculator = MPFITCalculator(config)
    calculators = {"MPFIT": calculator}

    n_jobs = 1 if args.sequential else args.n_jobs

    processor = BatchProcessor(
        calculators=calculators,
        n_jobs=n_jobs,
        backend=args.backend
    )

    logger.info("Computing MPFIT charges and multipole moments...")
    batch_results = processor.process(molecules)

    successful_molecules = []
    successful_results = []

    if args.print_charges:
        logger.info("")
        logger.info("=" * 70)
        logger.info("MPFIT Charges by Molecule")
        logger.info("=" * 70)

    for i, result in enumerate(batch_results):
        if result.success and "MPFIT" in result.results:
            successful_molecules.append(molecules[i])
            mpfit_data = result.results["MPFIT"]

            from MMomentA.qm.mpfit import MPFITResult
            import numpy as np

            # Convert conformer to plain numpy array (strip units if present)
            conformer = mpfit_data['conformer']

            # Recursively strip units from nested structures
            def strip_units(obj):
                if hasattr(obj, 'magnitude'):
                    return strip_units(obj.magnitude)
                elif hasattr(obj, 'm'):
                    return strip_units(obj.m)
                elif isinstance(obj, np.ndarray):
                    # For numpy arrays, convert to list first to strip units
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

            # Print charges if requested
            if args.print_charges:
                smiles = mpfit_data['smiles']
                charges = mpfit_result.charges
                charges_str = ' '.join([f'{c:7.4f}' for c in charges])
                logger.info(f"{i+1:3d}. {smiles:20s} [{charges_str}]")

    logger.info(f"Successfully processed {len(successful_molecules)}/{len(molecules)} molecules")

    logger.info("Extracting molecular features...")
    molecule_data_list = []

    for mol, mpfit_result in zip(successful_molecules, successful_results):
        mol_data = extract_molecule_data(mol, mpfit_result)
        molecule_data_list.append(mol_data)

    logger.info(f"Extracted features for {len(molecule_data_list)} molecules")

    logger.info(f"Splitting dataset using {args.split_strategy} strategy...")

    if args.split_strategy == "scaffold":
        train_idx, val_idx, test_idx = scaffold_split(
            successful_molecules,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_seed=args.random_seed
        )
    else:
        train_idx, val_idx, test_idx = random_split(
            len(molecule_data_list),
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_seed=args.random_seed
        )

    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    metadata = DatasetMetadata(
        dataset_name=args.dataset_name,
        n_molecules=len(molecule_data_list),
        qm_method=args.method,
        qm_basis=args.basis,
        multipole_limit=args.limit,
        created_date=datetime.now().isoformat(),
        split_strategy=args.split_strategy,
        splits={
            'train': [molecule_data_list[i].molecule_id for i in train_idx],
            'val': [molecule_data_list[i].molecule_id for i in val_idx],
            'test': [molecule_data_list[i].molecule_id for i in test_idx]
        }
    )

    logger.info(f"Saving dataset to {args.output}...")
    save_dataset(molecule_data_list, args.output, metadata)

    logger.info("Dataset preparation complete!")
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
