#!/usr/bin/env python
"""
Generate comprehensive per-element charge comparison visualizations.

This script creates:
1. All-element combined hexbin plot
2. Multi-panel grid of per-element hexbins
3. Individual hexbin plots for each element
4. Per-element RMSE/MAE bar charts
5. Per-element error distribution violin plots

Usage:
    python scripts/plot_element_charge_comparison.py \
        --data-file figures/zinc_100k_esp_validation/element_charge_data.npz \
        --results-file figures/zinc_100k_esp_validation/esp_validation_results.json \
        --output-dir figures/zinc_100k_esp_validation
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

# Set matplotlib style
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_hexbin_plot(ref_charges, pred_charges, element, output_file, gridsize=50):
    """Create hexbin plot for a single element."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Compute statistics
    rmse = np.sqrt(np.mean((pred_charges - ref_charges) ** 2))
    mae = np.mean(np.abs(pred_charges - ref_charges))
    r, _ = pearsonr(ref_charges, pred_charges)

    # Determine axis limits
    min_charge = min(ref_charges.min(), pred_charges.min())
    max_charge = max(ref_charges.max(), pred_charges.max())
    margin = (max_charge - min_charge) * 0.1
    lim_min = min_charge - margin
    lim_max = max_charge + margin

    # Create hexbin plot
    hb = ax.hexbin(ref_charges, pred_charges, gridsize=gridsize, cmap='viridis',
                   mincnt=1, linewidths=0.2, edgecolors='white')

    # Add diagonal line (perfect prediction)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, alpha=0.7,
           label='Perfect prediction')

    # Set labels and title
    ax.set_xlabel(f'MPFIT {element} Charges (e)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'MMomentA-GNN {element} Charges (e)', fontsize=12, fontweight='bold')
    ax.set_title(f'{element} Charge Prediction', fontsize=14, fontweight='bold')

    # Set axis limits
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')

    # Add statistics text box
    stats_text = f'N = {len(ref_charges):,}\n'
    stats_text += f'RMSE = {rmse:.4f} e\n'
    stats_text += f'MAE = {mae:.4f} e\n'
    stats_text += f'R = {r:.4f}'

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white',
           alpha=0.8, edgecolor='black', linewidth=1.5), fontsize=10,
           fontfamily='monospace')

    # Add colorbar
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('Count', fontsize=11)

    # Add legend
    ax.legend(loc='lower right', fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  ✓ {element} hexbin saved to {output_file}")


def create_all_elements_hexbin(element_charges, output_file, gridsize=50):
    """Create combined hexbin plot for all elements."""
    # Combine all charges
    all_ref = []
    all_pred = []

    for element in element_charges.keys():
        all_ref.extend(element_charges[element]['ref'])
        all_pred.extend(element_charges[element]['pred'])

    all_ref = np.array(all_ref)
    all_pred = np.array(all_pred)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Compute statistics
    rmse = np.sqrt(np.mean((all_pred - all_ref) ** 2))
    mae = np.mean(np.abs(all_pred - all_ref))
    r, _ = pearsonr(all_ref, all_pred)

    # Determine axis limits
    min_charge = min(all_ref.min(), all_pred.min())
    max_charge = max(all_ref.max(), all_pred.max())
    margin = (max_charge - min_charge) * 0.1
    lim_min = min_charge - margin
    lim_max = max_charge + margin

    # Create hexbin plot
    hb = ax.hexbin(all_ref, all_pred, gridsize=gridsize, cmap='viridis',
                   mincnt=1, linewidths=0.2, edgecolors='white')

    # Add diagonal line (perfect prediction)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, alpha=0.7,
           label='Perfect prediction')

    # Set labels and title
    ax.set_xlabel('MPFIT Charges (e)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MMomentA-GNN Charges (e)', fontsize=14, fontweight='bold')
    ax.set_title('All Elements Charge Prediction', fontsize=16, fontweight='bold')

    # Set axis limits
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')

    # Add statistics text box
    stats_text = f'N = {len(all_ref):,}\n'
    stats_text += f'RMSE = {rmse:.4f} e\n'
    stats_text += f'MAE = {mae:.4f} e\n'
    stats_text += f'R = {r:.4f}'

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white',
           alpha=0.8, edgecolor='black', linewidth=1.5), fontsize=11,
           fontfamily='monospace')

    # Add colorbar
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('Count', fontsize=12)

    # Add legend
    ax.legend(loc='lower right', fontsize=11)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ All-element hexbin saved to {output_file}")


def create_element_grid_hexbin(element_charges, output_file, gridsize=30):
    """Create multi-panel grid of per-element hexbins."""
    # Get common elements (sorted by count)
    elements_sorted = sorted(element_charges.keys(),
                            key=lambda e: len(element_charges[e]['ref']),
                            reverse=True)

    # Limit to top 9 elements for 3x3 grid
    n_elements = min(9, len(elements_sorted))
    elements_to_plot = elements_sorted[:n_elements]

    # Determine grid size
    n_cols = 3
    n_rows = (n_elements + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(15, 5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    for i, element in enumerate(elements_to_plot):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])

        ref_charges = element_charges[element]['ref']
        pred_charges = element_charges[element]['pred']

        # Compute statistics
        rmse = np.sqrt(np.mean((pred_charges - ref_charges) ** 2))
        mae = np.mean(np.abs(pred_charges - ref_charges))
        r, _ = pearsonr(ref_charges, pred_charges)

        # Determine axis limits
        min_charge = min(ref_charges.min(), pred_charges.min())
        max_charge = max(ref_charges.max(), pred_charges.max())
        margin = (max_charge - min_charge) * 0.1
        lim_min = min_charge - margin
        lim_max = max_charge + margin

        # Create hexbin plot
        hb = ax.hexbin(ref_charges, pred_charges, gridsize=gridsize, cmap='viridis',
                       mincnt=1, linewidths=0.1, edgecolors='white')

        # Add diagonal line
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5, alpha=0.7)

        # Set labels and title
        ax.set_xlabel(f'MPFIT {element} (e)', fontsize=10)
        ax.set_ylabel(f'GNN {element} (e)', fontsize=10)
        ax.set_title(f'{element} (N={len(ref_charges):,})', fontsize=12, fontweight='bold')

        # Set axis limits
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect('equal')

        # Add statistics text box
        stats_text = f'RMSE={rmse:.4f}\nMAE={mae:.4f}\nR={r:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white',
               alpha=0.8, edgecolor='black', linewidth=1), fontsize=8,
               fontfamily='monospace')

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Colorbar
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label('Count', fontsize=9)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Element grid hexbin saved to {output_file}")


def create_rmse_bar_chart(per_element_metrics, output_file):
    """Create bar chart of per-element RMSE and MAE."""
    # Sort elements by RMSE
    elements = sorted(per_element_metrics.keys(),
                     key=lambda e: per_element_metrics[e]['rmse'],
                     reverse=True)

    rmse_values = [per_element_metrics[e]['rmse'] for e in elements]
    mae_values = [per_element_metrics[e]['mae'] for e in elements]
    n_atoms = [per_element_metrics[e]['n_atoms'] for e in elements]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # RMSE bar chart
    x_pos = np.arange(len(elements))
    bars1 = ax1.bar(x_pos, rmse_values, color='steelblue', edgecolor='black', linewidth=1.5)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars1, n_atoms)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height,
                f'N={count:,}', ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('Element', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE (e)', fontsize=12, fontweight='bold')
    ax1.set_title('Charge Prediction RMSE by Element', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(elements, fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

    # MAE bar chart
    bars2 = ax2.bar(x_pos, mae_values, color='coral', edgecolor='black', linewidth=1.5)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars2, n_atoms)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height,
                f'N={count:,}', ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel('Element', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAE (e)', fontsize=12, fontweight='bold')
    ax2.set_title('Charge Prediction MAE by Element', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(elements, fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ RMSE/MAE bar chart saved to {output_file}")


def create_error_distributions(element_charges, output_file):
    """Create violin plots of charge prediction errors by element."""
    # Get common elements (sorted by count)
    elements_sorted = sorted(element_charges.keys(),
                            key=lambda e: len(element_charges[e]['ref']),
                            reverse=True)

    # Limit to top 10 elements
    n_elements = min(10, len(elements_sorted))
    elements_to_plot = elements_sorted[:n_elements]

    # Compute errors for each element
    errors_data = []
    labels = []

    for element in elements_to_plot:
        ref_charges = element_charges[element]['ref']
        pred_charges = element_charges[element]['pred']
        errors = pred_charges - ref_charges
        errors_data.append(errors)
        labels.append(f'{element}\n(N={len(errors):,})')

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create violin plot
    parts = ax.violinplot(errors_data, positions=range(len(elements_to_plot)),
                          showmeans=True, showmedians=True, widths=0.7)

    # Customize violin plot colors
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7,
              label='Zero error')

    # Set labels
    ax.set_xlabel('Element', fontsize=12, fontweight='bold')
    ax.set_ylabel('Charge Prediction Error (e)', fontsize=12, fontweight='bold')
    ax.set_title('Charge Prediction Error Distributions by Element',
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(elements_to_plot)))
    ax.set_xticklabels(labels, fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Error distribution plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-element charge comparison visualizations"
    )

    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to element charge data NPZ file")
    parser.add_argument("--results-file", type=str, required=True,
                       help="Path to merged results JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for plots")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for element-wise plots
    element_dir = output_dir / "element_wise_comparison"
    element_dir.mkdir(exist_ok=True)

    individual_dir = element_dir / "individual"
    individual_dir.mkdir(exist_ok=True)

    logger.info("="*70)
    logger.info("Generating Per-Element Charge Comparison Visualizations")
    logger.info("="*70)
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Results file: {args.results_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Load data
    logger.info("Loading data...")
    data = np.load(args.data_file)
    elements = data['elements']

    with open(args.results_file, 'r') as f:
        results = json.load(f)

    per_element_metrics = results['per_element_metrics']

    # Reconstruct element_charges dictionary
    element_charges = {}
    for element in elements:
        element_charges[element] = {
            'ref': data[f'{element}_ref'],
            'pred': data[f'{element}_pred']
        }

    logger.info(f"Loaded data for {len(elements)} elements")
    logger.info("")

    # 1. Create all-element combined hexbin
    logger.info("Creating all-element combined hexbin...")
    all_elements_file = element_dir / "all_elements_hexbin.png"
    create_all_elements_hexbin(element_charges, all_elements_file)
    logger.info("")

    # 2. Create multi-panel grid
    logger.info("Creating element grid hexbin...")
    grid_file = element_dir / "element_grid_hexbin.png"
    create_element_grid_hexbin(element_charges, grid_file)
    logger.info("")

    # 3. Create individual hexbin plots
    logger.info("Creating individual element hexbins...")
    for element in sorted(elements):
        individual_file = individual_dir / f"{element}_charges.png"
        create_hexbin_plot(
            element_charges[element]['ref'],
            element_charges[element]['pred'],
            element,
            individual_file
        )
    logger.info("")

    # 4. Create RMSE/MAE bar chart
    logger.info("Creating RMSE/MAE bar chart...")
    bar_chart_file = element_dir / "per_element_rmse_mae.png"
    create_rmse_bar_chart(per_element_metrics, bar_chart_file)
    logger.info("")

    # 5. Create error distribution violin plot
    logger.info("Creating error distribution plot...")
    violin_file = element_dir / "element_error_distributions.png"
    create_error_distributions(element_charges, violin_file)
    logger.info("")

    logger.info("="*70)
    logger.info("Visualization Complete!")
    logger.info("="*70)
    logger.info("")
    logger.info("Generated plots:")
    logger.info(f"  1. All-element hexbin:     {all_elements_file}")
    logger.info(f"  2. Element grid hexbin:    {grid_file}")
    logger.info(f"  3. Individual hexbins:     {individual_dir}/ ({len(elements)} files)")
    logger.info(f"  4. RMSE/MAE bar chart:     {bar_chart_file}")
    logger.info(f"  5. Error distributions:    {violin_file}")
    logger.info("")


if __name__ == "__main__":
    main()
