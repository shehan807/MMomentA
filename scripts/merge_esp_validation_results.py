#!/usr/bin/env python
"""
Merge ESP validation results from parallel chunks.

This script:
1. Reads all chunk result files
2. Combines ESP validation metrics
3. Aggregates charges for ALL elements (not just carbon)
4. Computes per-element statistics (RMSE, MAE, R²)
5. Generates comprehensive visualizations
6. Saves final merged results

Usage:
    python scripts/merge_esp_validation_results.py \
        --input-dir figures/zinc_100k_esp_validation \
        --output-dir figures/zinc_100k_esp_validation \
        --n-chunks 20
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_element_statistics(ref_charges: np.ndarray, pred_charges: np.ndarray) -> Dict:
    """Compute charge prediction statistics."""
    errors = pred_charges - ref_charges
    abs_errors = np.abs(errors)

    # Compute R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((ref_charges - np.mean(ref_charges)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(abs_errors)),
        'r2': float(r2),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'n_atoms': int(len(ref_charges))
    }


def main():
    parser = argparse.ArgumentParser(description="Merge ESP validation chunk results")

    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory containing chunk result files")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for merged results")
    parser.add_argument("--n-chunks", type=int, required=True,
                       help="Number of chunks to merge")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("Merging ESP Validation Results")
    logger.info("="*70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Expected chunks: {args.n_chunks}")
    logger.info("")

    # Load all chunk results
    chunk_files = sorted(input_dir.glob("esp_validation_results_chunk_*.json"))

    if len(chunk_files) == 0:
        logger.error(f"No chunk files found in {input_dir}")
        return

    logger.info(f"Found {len(chunk_files)} chunk files")

    # Aggregate results from all chunks
    results = {
        'mpfit': {'mae': [], 'rmse': [], 'time': []},
        'am1bcc': {'mae': [], 'rmse': [], 'time': []},
        'resp': {'mae': [], 'rmse': [], 'time': []},
        'gnn': {'mae': [], 'rmse': [], 'time': []}
    }

    # Per-element charge aggregation
    # Structure: element_charges[element_symbol] = {'ref': [...], 'pred': [...]}
    element_charges = defaultdict(lambda: {'ref': [], 'pred': []})

    total_successful = 0
    total_failed = 0
    total_molecules = 0

    for chunk_file in chunk_files:
        logger.info(f"Processing {chunk_file.name}...")

        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)

        chunk_id = chunk_data['chunk_id']
        n_successful = chunk_data['successful']
        n_failed = chunk_data['failed']

        total_successful += n_successful
        total_failed += n_failed
        total_molecules += chunk_data['n_molecules']

        logger.info(f"  Chunk {chunk_id}: {n_successful} successful, {n_failed} failed")

        # Process each molecule in the chunk
        for mol_result in chunk_data['molecule_results']:
            if not mol_result['success']:
                continue

            validation = mol_result['validation']

            # ESP validation metrics
            results['mpfit']['mae'].append(validation['mpfit']['mae'])
            results['mpfit']['rmse'].append(validation['mpfit']['rmse'])

            if mol_result['mpfit_time'] is not None:
                results['mpfit']['time'].append(mol_result['mpfit_time'])

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
            results['gnn']['time'].append(mol_result['gnn_inference_time'])

            # Aggregate per-element charges
            mpfit_charges = np.array(mol_result['mpfit_charges'])
            gnn_charges = np.array(mol_result['gnn_charges'])
            element_symbols = mol_result['element_symbols']

            # Group charges by element
            for i, element in enumerate(element_symbols):
                element_charges[element]['ref'].append(mpfit_charges[i])
                element_charges[element]['pred'].append(gnn_charges[i])

    logger.info("")
    logger.info("="*70)
    logger.info("Aggregation Complete")
    logger.info("="*70)
    logger.info(f"Total molecules processed: {total_molecules}")
    logger.info(f"Successful: {total_successful}")
    logger.info(f"Failed: {total_failed}")
    logger.info("")

    # Convert element charges to numpy arrays
    for element in element_charges:
        element_charges[element]['ref'] = np.array(element_charges[element]['ref'])
        element_charges[element]['pred'] = np.array(element_charges[element]['pred'])

    # Compute per-element statistics
    logger.info("Computing per-element charge statistics...")
    per_element_metrics = {}

    for element in sorted(element_charges.keys()):
        ref_charges = element_charges[element]['ref']
        pred_charges = element_charges[element]['pred']

        stats = compute_element_statistics(ref_charges, pred_charges)
        per_element_metrics[element] = stats

        logger.info(f"  {element:2s}: RMSE={stats['rmse']:.6f}, MAE={stats['mae']:.6f}, "
                   f"R²={stats['r2']:.4f}, N={stats['n_atoms']}")

    logger.info("")

    # Print ESP validation summary
    logger.info("="*70)
    logger.info("ESP VALIDATION RESULTS")
    logger.info("="*70)

    logger.info("\nMPFIT ESP Validation:")
    logger.info(f"  MAE:  {np.mean(results['mpfit']['mae']):.6e} ± {np.std(results['mpfit']['mae']):.6e} a.u.")
    logger.info(f"  RMSE: {np.mean(results['mpfit']['rmse']):.6e} ± {np.std(results['mpfit']['rmse']):.6e} a.u.")
    if results['mpfit']['time']:
        logger.info(f"  Time: {np.mean(results['mpfit']['time']):.3f} ± {np.std(results['mpfit']['time']):.3f} s/molecule")

    if results['am1bcc']['mae']:
        logger.info("\nAM1-BCC ESP Validation:")
        logger.info(f"  MAE:  {np.mean(results['am1bcc']['mae']):.6e} ± {np.std(results['am1bcc']['mae']):.6e} a.u.")
        logger.info(f"  RMSE: {np.mean(results['am1bcc']['rmse']):.6e} ± {np.std(results['am1bcc']['rmse']):.6e} a.u.")
        if results['am1bcc']['time']:
            logger.info(f"  Time: {np.mean(results['am1bcc']['time']):.3f} ± {np.std(results['am1bcc']['time']):.3f} s/molecule")

    if results['resp']['mae']:
        logger.info("\nRESP ESP Validation:")
        logger.info(f"  MAE:  {np.mean(results['resp']['mae']):.6e} ± {np.std(results['resp']['mae']):.6e} a.u.")
        logger.info(f"  RMSE: {np.mean(results['resp']['rmse']):.6e} ± {np.std(results['resp']['rmse']):.6e} a.u.")
        if results['resp']['time']:
            logger.info(f"  Time: {np.mean(results['resp']['time']):.3f} ± {np.std(results['resp']['time']):.3f} s/molecule")

    logger.info("\nMMomentA-GNN ESP Validation:")
    logger.info(f"  MAE:  {np.mean(results['gnn']['mae']):.6e} ± {np.std(results['gnn']['mae']):.6e} a.u.")
    logger.info(f"  RMSE: {np.mean(results['gnn']['rmse']):.6e} ± {np.std(results['gnn']['rmse']):.6e} a.u.")
    if results['gnn']['time']:
        logger.info(f"  Time: {np.mean(results['gnn']['time']):.3f} ± {np.std(results['gnn']['time']):.3f} s/molecule")

    logger.info("\n" + "="*70)
    logger.info("TIMING COMPARISON (seconds per molecule)")
    logger.info("="*70)
    if results['mpfit']['time']:
        logger.info(f"MPFIT:        {np.mean(results['mpfit']['time']):8.3f} ± {np.std(results['mpfit']['time']):6.3f}")
    if results['am1bcc']['time']:
        logger.info(f"AM1-BCC:      {np.mean(results['am1bcc']['time']):8.3f} ± {np.std(results['am1bcc']['time']):6.3f}")
    if results['resp']['time']:
        logger.info(f"RESP:         {np.mean(results['resp']['time']):8.3f} ± {np.std(results['resp']['time']):6.3f}")
    if results['gnn']['time']:
        logger.info(f"MMomentA-GNN: {np.mean(results['gnn']['time']):8.3f} ± {np.std(results['gnn']['time']):6.3f}")

        # Calculate speedup
        if results['mpfit']['time']:
            speedup = np.mean(results['mpfit']['time']) / np.mean(results['gnn']['time'])
            logger.info(f"\nSpeedup over MPFIT: {speedup:.1f}x")
    logger.info("="*70)
    logger.info("")

    # Save merged results
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

    # Save complete results
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'n_chunks': args.n_chunks,
                'n_molecules': total_molecules,
                'successful': total_successful,
                'failed': total_failed
            },
            'esp_metrics': {
                'mpfit': {k: [float(v) for v in vals] for k, vals in results['mpfit'].items()},
                'am1bcc': {k: [float(v) for v in vals] for k, vals in results['am1bcc'].items()} if results['am1bcc']['mae'] else {},
                'resp': {k: [float(v) for v in vals] for k, vals in results['resp'].items()} if results['resp']['mae'] else {},
                'gnn': {k: [float(v) for v in vals] for k, vals in results['gnn'].items()}
            },
            'summary': summary,
            'per_element_metrics': per_element_metrics
        }, f, indent=2)

    logger.info(f"✓ Merged results saved to {results_file}")

    # Save per-element charge data for plotting
    element_data_file = output_dir / "element_charge_data.npz"
    np.savez(
        element_data_file,
        **{f'{elem}_ref': element_charges[elem]['ref'] for elem in element_charges},
        **{f'{elem}_pred': element_charges[elem]['pred'] for elem in element_charges},
        elements=np.array(list(element_charges.keys()))
    )

    logger.info(f"✓ Per-element charge data saved to {element_data_file}")
    logger.info("")
    logger.info("="*70)
    logger.info("Merge Complete!")
    logger.info("="*70)
    logger.info("")
    logger.info("Next step: Generate visualizations")
    logger.info("  python scripts/plot_element_charge_comparison.py \\")
    logger.info(f"    --data-file {element_data_file} \\")
    logger.info(f"    --results-file {results_file} \\")
    logger.info(f"    --output-dir {output_dir}")
    logger.info("")


if __name__ == "__main__":
    main()
