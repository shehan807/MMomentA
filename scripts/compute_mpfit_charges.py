#!/usr/bin/env python
"""Compute MPFIT charges for molecules from various datasets.

This script provides a clean interface to compute MPFIT charges for
molecules from ZINC, SPICE, or other datasets, with configurable QM
settings and parallel processing.

Examples
--------
# Process ZINC molecules with default settings
python compute_mpfit_charges.py --dataset zinc --input zinc_molecules.pkl --output zinc_mpfit.json --max-molecules 100

# Process SPICE molecules with custom QM settings
python compute_mpfit_charges.py --dataset spice --input spice.oeb --method pbe0 --basis def2-SVP --n-jobs 8
"""

import argparse
import logging
from pathlib import Path

from mmomenta.qm import MPFITCalculator, GDMAConfig
from mmomenta.data import load_molecules_from_file, BatchProcessor
from mmomenta.utils import setup_logging, save_results


def main():
    parser = argparse.ArgumentParser(
        description="Compute MPFIT charges for molecular datasets"
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
        default="mpfit_results.json",
        help="Output file path (default: mpfit_results.json)"
    )
    parser.add_argument(
        "--dataset",
        choices=["zinc", "spice", "generic"],
        default="generic",
        help="Dataset type (default: generic)"
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
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1, use all CPUs)"
    )
    parser.add_argument(
        "--backend",
        choices=["loky", "multiprocessing", "threading"],
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
        "--inspect",
        action="store_true",
        help="Print dataset statistics before processing"
    )

    args = parser.parse_args()

    logger = setup_logging(level=args.log_level, log_file=args.log_file)

    logger.info("=" * 70)
    logger.info("MPFIT Charge Calculation")
    logger.info("=" * 70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Dataset type: {args.dataset}")
    logger.info(f"QM method: {args.method}/{args.basis}")
    logger.info(f"Multipole limit: {args.limit}")
    logger.info("=" * 70)

    molecules = load_molecules_from_file(
        args.input,
        max_molecules=args.max_molecules
    )

    if args.inspect:
        from mmomenta.data.loaders import _inspect_dataset
        _inspect_dataset(molecules, args.dataset.upper())

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

    results = processor.process(molecules)

    output_data = {
        "config": {
            "method": args.method,
            "basis": args.basis,
            "limit": args.limit,
            "svd_threshold": args.svd_threshold,
            "dataset": args.dataset,
            "n_molecules": len(molecules)
        },
        "results": [
            {
                "molecule_index": r.molecule_index,
                "smiles": r.smiles,
                "formula": r.formula,
                "n_atoms": r.n_atoms,
                "success": r.success,
                "mpfit_charges": r.results.get("MPFIT", {}).get("charges", []),
                "multipole_moments": r.results.get("MPFIT", {}).get("multipole_moments", []),
                "time": r.results.get("MPFIT", {}).get("time", 0.0),
                "metadata": r.results.get("MPFIT", {}).get("metadata", {})
            }
            for r in results
        ]
    }

    save_results(output_data, args.output, format="json")
    logger.info(f"Results saved to {args.output}")

    successful = sum(1 for r in results if r.success)
    logger.info(f"Successfully processed {successful}/{len(results)} molecules")


if __name__ == "__main__":
    main()
