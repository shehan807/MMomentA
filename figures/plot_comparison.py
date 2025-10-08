#!/usr/bin/env python
"""
Publication-quality plotting utilities for charge fitting comparison.
Extended to include MPFIT-GNN (MMomentA) method.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def setup_publication_style():
    """Set up publication-quality matplotlib style using LovelyPlots"""

    # Use LovelyPlots for professional styling
    import lovelyplots
    plt.style.use('ipynb')
    print("✓ Using LovelyPlots 'ipynb' styling")

    # Override specific settings for charge fitting analysis
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', list(get_method_colors().values()))


def get_method_colors() -> Dict[str, str]:
    """Get consistent color scheme for charge fitting methods"""
    return {
        'RESP': '#2E86AB',         # Blue - QM reference
        'AM1-BCC': '#E8A317',      # Golden - empirical
        'MPFIT': '#A23B72',        # Purple - multipole QM
        'MPFIT-GNN': '#00C853',    # Green - ML method (NEW)
        'ESP': '#F18F01',          # Orange - similar to RESP (if needed)
        'AM1-BCC(assign)': '#E8A317'  # Same as AM1-BCC
    }


def get_method_markers() -> Dict[str, str]:
    """Get consistent marker scheme for methods"""
    return {
        'RESP': 'o',              # Circle - standard/reference
        'AM1-BCC': 's',           # Square - structured/empirical
        'MPFIT': '^',             # Triangle - advanced/multipole
        'MPFIT-GNN': 'D',         # Diamond - ML method (NEW)
        'ESP': 'D',               # Diamond - similar to RESP
        'AM1-BCC(assign)': 's'    # Same as AM1-BCC
    }


def create_correlation_plot(charges1: np.ndarray,
                          charges2: np.ndarray,
                          method1: str,
                          method2: str,
                          ax: plt.Axes,
                          alpha: float = 0.7,
                          s: float = 40) -> plt.Axes:
    """Create a publication-quality correlation plot"""

    colors = get_method_colors()
    markers = get_method_markers()

    # Calculate statistics
    correlation = np.corrcoef(charges1, charges2)[0, 1]
    rmsd = np.sqrt(np.mean((charges1 - charges2)**2))

    # Create shorter labels for plots
    short_labels = {
        "AM1-BCC(assign)": "AM1-BCC",
        "RESP": "RESP",
        "MPFIT": "MPFIT",
        "MPFIT-GNN": "MPFIT-GNN"
    }
    label1 = short_labels.get(method1, method1)
    label2 = short_labels.get(method2, method2)

    # Determine plot range
    all_charges = np.concatenate([charges1, charges2])
    margin = 0.1 * (np.max(all_charges) - np.min(all_charges))
    plot_min = np.min(all_charges) - margin
    plot_max = np.max(all_charges) + margin

    # Scatter plot
    ax.scatter(charges1, charges2,
              alpha=alpha,
              s=s,
              color=colors.get(method1, '#333333'),
              marker=markers.get(method1, 'o'),
              edgecolors='white',
              linewidth=0.5,
              zorder=3)

    # Perfect correlation line
    ax.plot([plot_min, plot_max], [plot_min, plot_max],
            'k--', alpha=0.5, linewidth=1.5, zorder=1)

    # Labels and title
    ax.set_xlabel(f"{label1} charges (e)")
    ax.set_ylabel(f"{label2} charges (e)")
    ax.set_title(f"{label1} vs {label2}\n$R$ = {correlation:.3f}, RMSE = {rmsd:.3f} e",
                 fontweight='bold')

    # Set equal aspect ratio and limits
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal')

    # Grid styling
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)

    return ax


def create_violin_plots(metrics: Dict,
                       output_dir: str = ".",
                       figsize: Tuple[float, float] = (14, 6),
                       qm_method: str = 'hf',
                       qm_basis: str = '6-31G*') -> str:
    """Create publication-quality violin plots for ESP validation metrics using LovelyPlots

    Parameters
    ----------
    metrics : dict
        Dictionary with 'mae' and 'rmse' keys, each containing method->values mapping
    output_dir : str
        Directory to save output plot
    figsize : tuple
        Figure size (width, height)
    qm_method : str
        QM method used for ESP calculation
    qm_basis : str
        QM basis set used for ESP calculation

    Returns
    -------
    str
        Path to saved plot file
    """

    setup_publication_style()
    colors = get_method_colors()

    # Use the methods that actually have data
    methods = list(metrics['mae'].keys())

    # Create shorter labels for plot readability
    short_labels = {
        "AM1-BCC(assign)": "AM1-BCC",
        "RESP": "RESP",
        "MPFIT": "MPFIT",
        "MPFIT-GNN": "MPFIT-GNN"
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # MAE Violin Plot
    mae_data = []
    mae_labels = []
    mae_colors = []

    for method in methods:
        if metrics['mae'][method]:  # Only include if data exists
            mae_data.append(metrics['mae'][method])
            mae_labels.append(short_labels.get(method, method))
            mae_colors.append(colors.get(method, '#333333'))

    if mae_data:
        parts1 = ax1.violinplot(mae_data, positions=range(len(mae_labels)),
                               showmeans=True, showmedians=True, widths=0.7)

        # Style the violin plots
        for patch, color in zip(parts1['bodies'], mae_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('white')
            patch.set_linewidth(1)

        # Style the statistical lines
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts1:
                parts1[partname].set_color('black')
                parts1[partname].set_linewidth(1.5)

        ax1.set_xticks(range(len(mae_labels)))
        ax1.set_xticklabels(mae_labels, fontweight='bold')
        ax1.set_ylabel('MAE (hartree/e)', fontweight='bold')
        ax1.set_title(f'QM ESP Validation ({qm_method}/{qm_basis})', fontweight='bold', pad=20)

    # RMSE Violin Plot
    rmse_data = []
    rmse_labels = []
    rmse_colors = []

    for method in methods:
        if metrics['rmse'][method]:  # Only include if data exists
            rmse_data.append(metrics['rmse'][method])
            rmse_labels.append(short_labels.get(method, method))
            rmse_colors.append(colors.get(method, '#333333'))

    if rmse_data:
        parts2 = ax2.violinplot(rmse_data, positions=range(len(rmse_labels)),
                               showmeans=True, showmedians=True, widths=0.7)

        # Style the violin plots
        for patch, color in zip(parts2['bodies'], rmse_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('white')
            patch.set_linewidth(1)

        # Style the statistical lines
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts2:
                parts2[partname].set_color('black')
                parts2[partname].set_linewidth(1.5)

        ax2.set_xticks(range(len(rmse_labels)))
        ax2.set_xticklabels(rmse_labels, fontweight='bold')
        ax2.set_ylabel('RMSE (hartree/e)', fontweight='bold')
        ax2.set_title(f'QM ESP Validation ({qm_method}/{qm_basis})', fontweight='bold', pad=20)

    # Overall figure styling
    plt.tight_layout()

    # Add subtle background
    fig.patch.set_facecolor('white')

    # Save plot
    output_file = Path(output_dir) / "esp_validation_all_methods.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"✓ Publication-quality violin plots saved to: {output_file}")
    plt.close()

    return str(output_file)


def create_correlation_subplot_figure(results: Dict,
                                    output_dir: str = ".") -> str:
    """Create publication-quality correlation subplot figure using LovelyPlots"""

    setup_publication_style()

    # Collect all charges by method - include MPFIT-GNN
    charges_by_method = {"RESP": [], "AM1-BCC(assign)": [], "MPFIT": [], "MPFIT-GNN": []}

    for mol_data in results.values():
        for method in charges_by_method:
            if method in mol_data.get("results", {}):
                charges_by_method[method].extend(mol_data["results"][method]["charges"])

    # Convert to numpy arrays
    for method in charges_by_method:
        charges_by_method[method] = np.array(charges_by_method[method])

    # Create correlation plots - expand to include MPFIT-GNN
    n_comparisons = 6  # All pairwise comparisons
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    comparisons = [
        ("RESP", "AM1-BCC(assign)"),
        ("RESP", "MPFIT"),
        ("RESP", "MPFIT-GNN"),
        ("AM1-BCC(assign)", "MPFIT"),
        ("AM1-BCC(assign)", "MPFIT-GNN"),
        ("MPFIT", "MPFIT-GNN")
    ]

    for ax, (method1, method2) in zip(axes, comparisons):
        if method1 in charges_by_method and method2 in charges_by_method:
            charges1 = charges_by_method[method1]
            charges2 = charges_by_method[method2]

            if len(charges1) > 0 and len(charges2) > 0:
                create_correlation_plot(charges1, charges2, method1, method2, ax)

    # Overall figure styling
    plt.tight_layout()
    fig.patch.set_facecolor('white')

    # Save plot
    output_file = Path(output_dir) / "charge_correlations_all_methods.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"✓ Publication-quality correlation plots saved to {output_file}")
    plt.close()

    return str(output_file)


def create_esp_violin_plot(metrics: Dict,
                          output_dir: str = ".",
                          figsize: Tuple[float, float] = (14, 6),
                          qm_method: str = 'hf',
                          qm_basis: str = '6-31G*') -> str:
    """Create publication-quality violin plots for ESP validation: MPFIT, AM1-BCC, RESP, MMomentA-GNN

    Parameters
    ----------
    metrics : dict
        Dictionary with 'mpfit', 'am1bcc', 'resp', and 'gnn' keys, each containing 'mae' and 'rmse' lists
    output_dir : str
        Directory to save output plot
    figsize : tuple
        Figure size (width, height)
    qm_method : str
        QM method used for ESP calculation
    qm_basis : str
        QM basis set used for ESP calculation

    Returns
    -------
    str
        Path to saved plot file
    """

    setup_publication_style()
    colors = get_method_colors()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Build lists of methods that actually have data
    method_keys = ['mpfit', 'am1bcc', 'resp', 'gnn']
    method_labels = {'mpfit': 'MPFIT', 'am1bcc': 'AM1-BCC', 'resp': 'RESP', 'gnn': 'MMomentA-GNN'}
    color_keys = {'mpfit': 'MPFIT', 'am1bcc': 'AM1-BCC', 'resp': 'RESP', 'gnn': 'MPFIT-GNN'}

    mae_data = []
    mae_labels = []
    mae_colors = []

    for key in method_keys:
        if key in metrics and metrics[key]['mae']:
            mae_data.append(metrics[key]['mae'])
            mae_labels.append(method_labels[key])
            mae_colors.append(colors[color_keys[key]])

    # MAE Violin Plot
    if mae_data:
        parts1 = ax1.violinplot(mae_data, positions=range(len(mae_labels)),
                               showmeans=True, showmedians=True, widths=0.6)

        # Style the violin plots
        for patch, color in zip(parts1['bodies'], mae_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('white')
            patch.set_linewidth(1)

        # Style the statistical lines
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts1:
                parts1[partname].set_color('black')
                parts1[partname].set_linewidth(1.5)

        ax1.set_xticks(range(len(mae_labels)))
        ax1.set_xticklabels(mae_labels, fontweight='bold', rotation=15, ha='right')
        ax1.set_ylabel('MAE (a.u.)', fontweight='bold')
        ax1.set_title(f'ESP Validation MAE\n({qm_method}/{qm_basis})', fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, axis='y')

    rmse_data = []
    rmse_labels = []
    rmse_colors = []

    for key in method_keys:
        if key in metrics and metrics[key]['rmse']:
            rmse_data.append(metrics[key]['rmse'])
            rmse_labels.append(method_labels[key])
            rmse_colors.append(colors[color_keys[key]])

    # RMSE Violin Plot
    if rmse_data:
        parts2 = ax2.violinplot(rmse_data, positions=range(len(rmse_labels)),
                               showmeans=True, showmedians=True, widths=0.6)

        # Style the violin plots
        for patch, color in zip(parts2['bodies'], rmse_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor('white')
            patch.set_linewidth(1)

        # Style the statistical lines
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts2:
                parts2[partname].set_color('black')
                parts2[partname].set_linewidth(1.5)

        ax2.set_xticks(range(len(rmse_labels)))
        ax2.set_xticklabels(rmse_labels, fontweight='bold', rotation=15, ha='right')
        ax2.set_ylabel('RMSE (a.u.)', fontweight='bold')
        ax2.set_title(f'ESP Validation RMSE\n({qm_method}/{qm_basis})', fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='y')

    # Overall figure styling
    plt.tight_layout()

    # Add subtle background
    fig.patch.set_facecolor('white')

    # Save plot
    output_file = Path(output_dir) / "esp_validation_all_methods.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"✓ ESP validation violin plot saved to: {output_file}")
    plt.close()

    return str(output_file)
