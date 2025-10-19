#!/usr/bin/env python
"""
Generate ESP validation violin plots from merged results.

This script reads the merged ESP validation results and creates
the MAE and RMSE violin plots comparing different charge methods.

Usage:
    python scripts/generate_esp_violin_plots.py \
        --results-file figures/zinc_100k_esp_validation/esp_validation_results.json \
        --output-dir figures/zinc_100k_esp_validation
"""

import argparse
import json
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate ESP validation violin plots")

    parser.add_argument("--results-file", type=str, required=True,
                       help="Path to merged ESP validation results JSON")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for plots")
    parser.add_argument("--qm-method", type=str, default="hf",
                       help="QM method used (for plot labels)")
    parser.add_argument("--qm-basis", type=str, default="6-31G*",
                       help="Basis set used (for plot labels)")

    args = parser.parse_args()

    results_file = Path(args.results_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("Generating ESP Validation Violin Plots")
    logger.info("="*70)
    logger.info(f"Results file: {results_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Load merged results
    logger.info("Loading merged results...")
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Extract esp_metrics
    esp_metrics = data['esp_metrics']

    logger.info(f"Found data for methods: {list(esp_metrics.keys())}")
    logger.info("")

    # Add figures directory to path for plot_comparison
    figures_dir = results_file.parent.parent.parent / 'figures'
    if not figures_dir.exists():
        figures_dir = Path(__file__).parent.parent / 'figures'

    sys.path.insert(0, str(figures_dir))

    try:
        from plot_comparison import create_esp_violin_plot

        logger.info("Generating ESP violin plots...")
        mae_file, rmse_file = create_esp_violin_plot(
            esp_metrics,
            output_dir=output_dir,
            qm_method=args.qm_method,
            qm_basis=args.qm_basis
        )

        logger.info("")
        logger.info("="*70)
        logger.info("ESP Violin Plots Generated Successfully!")
        logger.info("="*70)
        logger.info(f"✓ ESP MAE plot:  {mae_file}")
        logger.info(f"✓ ESP RMSE plot: {rmse_file}")
        logger.info("")

    except ImportError as e:
        logger.error(f"Failed to import plot_comparison: {e}")
        logger.error("Make sure the 'figures' directory exists with plot_comparison.py")
        sys.exit(1)

    except KeyError as e:
        logger.error(f"Data format error: {e}")
        logger.error("The results file may be missing required data fields")
        logger.error("Available fields:")
        for method, data in esp_metrics.items():
            logger.error(f"  {method}: {list(data.keys())}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
