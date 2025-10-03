#!/usr/bin/env python
"""Compare MPFIT, RESP, and AM1-BCC charge fitting methods.

This script runs all three charge fitting methods on a dataset and
generates comparison statistics and visualizations.

Examples
--------
# Compare methods on ZINC molecules
python compare_charge_methods.py --input zinc_molecules.pkl --max-molecules 50 --n-jobs 4

# Compare with custom QM settings
python compare_charge_methods.py --input spice.oeb --method pbe0 --basis def2-SVP --output comparison_results.json
"""

import argparse
import logging
from pathlib import Path

from mmomenta.qm import MPFITCalculator, RESPCalculator, AM1BCCCalculator
from mmomenta.qm import GDMAConfig, ESPConfig, AM1BCCConfig
from mmomenta.data import load_molecules_from_file, BatchProcessor
from mmomenta.comparison import compare_charges, validate_esp_reproduction
from mmomenta.utils import setup_logging, save_results


def main():
    parser = argparse.ArgumentParser(
        description="Compare MPFIT, RESP, and AM1-BCC charge methods"
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
        default="comparison_results.json",
        help="Output file path (default: comparison_results.json)"
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
        help="QM method for MPFIT and RESP (default: hf)"
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="6-31G*",
        help="Basis set for MPFIT and RESP (default: 6-31G*)"
    )
    parser.add_argument(
        "--mpfit-limit",
        type=int,
        default=8,
        help="MPFIT multipole expansion limit (default: 8)"
    )
    parser.add_argument(
        "--resp-restraint",
        type=float,
        default=0.0005,
        help="RESP restraint weight (default: 0.0005)"
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
        "--skip-am1bcc",
        action="store_true",
        help="Skip AM1-BCC calculations"
    )
    parser.add_argument(
        "--skip-resp",
        action="store_true",
        help="Skip RESP calculations"
    )
    parser.add_argument(
        "--skip-mpfit",
        action="store_true",
        help="Skip MPFIT calculations"
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

    args = parser.parse_args()

    logger = setup_logging(level=args.log_level, log_file=args.log_file)

    logger.info("=" * 70)
    logger.info("Charge Method Comparison")
    logger.info("=" * 70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"QM method: {args.method}/{args.basis}")
    logger.info("=" * 70)

    molecules = load_molecules_from_file(
        args.input,
        max_molecules=args.max_molecules
    )

    calculators = {}

    if not args.skip_am1bcc:
        calculators["AM1-BCC"] = AM1BCCCalculator(AM1BCCConfig())

    if not args.skip_resp:
        resp_config = ESPConfig(
            method=args.method,
            basis=args.basis,
            restraint_weight=args.resp_restraint
        )
        calculators["RESP"] = RESPCalculator(resp_config)

    if not args.skip_mpfit:
        mpfit_config = GDMAConfig(
            method=args.method,
            basis=args.basis,
            limit=args.mpfit_limit
        )
        calculators["MPFIT"] = MPFITCalculator(mpfit_config)

    logger.info(f"Active methods: {list(calculators.keys())}")

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
            "mpfit_limit": args.mpfit_limit,
            "resp_restraint": args.resp_restraint,
            "n_molecules": len(molecules)
        },
        "results": []
    }

    for r in results:
        molecule_data = {
            "molecule_index": r.molecule_index,
            "smiles": r.smiles,
            "formula": r.formula,
            "n_atoms": r.n_atoms,
            "success": r.success,
            "partial_success": r.partial_success,
            "failed_methods": r.failed_methods,
            "methods": {}
        }

        for method_name, method_result in r.results.items():
            if isinstance(method_result, dict) and "error" not in method_result:
                molecule_data["methods"][method_name] = {
                    "charges": method_result.get("charges", []),
                    "time": method_result.get("time", 0.0),
                    "metadata": method_result.get("metadata", {})
                }

        if len(molecule_data["methods"]) >= 2:
            import numpy as np
            comparisons = {}
            method_names = list(molecule_data["methods"].keys())

            for i, method1 in enumerate(method_names):
                for method2 in method_names[i+1:]:
                    charges1 = np.array(molecule_data["methods"][method1]["charges"])
                    charges2 = np.array(molecule_data["methods"][method2]["charges"])

                    comparison = compare_charges(charges1, charges2, method1, method2)

                    comparisons[f"{method1}_vs_{method2}"] = {
                        "rmsd": comparison.rmsd,
                        "mae": comparison.mae,
                        "max_abs_diff": comparison.max_abs_diff,
                        "correlation": comparison.correlation
                    }

            molecule_data["comparisons"] = comparisons

        output_data["results"].append(molecule_data)

    save_results(output_data, args.output, format="json")
    logger.info(f"Results saved to {args.output}")

    successful = sum(1 for r in results if r.success)
    partial = sum(1 for r in results if r.partial_success)
    logger.info(f"Fully successful: {successful}/{len(results)}")
    logger.info(f"Partially successful: {partial}/{len(results)}")


if __name__ == "__main__":
    main()
