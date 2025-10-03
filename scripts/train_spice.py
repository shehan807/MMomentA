#!/usr/bin/env python
"""Train charge prediction model on SPICE dataset.

Examples
--------
# Train baseline model (no multipoles)
python train_spice.py \\
    --dataset spice_mpfit.h5 \\
    --output-dir runs/spice_baseline \\
    --feature-units 117 \\
    --no-multipoles

# Train with multipole moments
python train_spice.py \\
    --dataset spice_mpfit.h5 \\
    --output-dir runs/spice_multipoles \\
    --feature-units 198 \\
    --n-epochs 1000
"""

import argparse
import logging
from pathlib import Path
import torch

from MMomentA.data import load_dataset, create_split_datasets
from MMomentA.models import ChargeModel, ModelConfig
from MMomentA.training import Trainer, TrainingConfig
from MMomentA.utils import setup_logging, save_results


def main():
    parser = argparse.ArgumentParser(
        description="Train charge prediction model on SPICE dataset"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to preprocessed SPICE dataset (.h5, .pkl, .json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/spice",
        help="Output directory for checkpoints and logs"
    )

    parser.add_argument(
        "--feature-units",
        type=int,
        default=117,
        help="Input feature dimension (117 baseline, 198 with multipoles)"
    )
    parser.add_argument(
        "--no-multipoles",
        action="store_true",
        help="Disable multipole features (baseline model)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Number of graph convolution layers"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Hidden dimension for graph convolution"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function"
    )

    parser.add_argument(
        "--n-epochs",
        type=int,
        default=1000,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)"
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=10,
        help="Evaluate on validation set every N epochs"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Early stopping patience (0 = disabled)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(
        level=args.log_level,
        log_file=output_dir / "training.log"
    )

    logger.info("=" * 70)
    logger.info("SPICE Dataset Training")
    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Feature dimension: {args.feature_units}")
    logger.info(f"Use multipoles: {not args.no_multipoles}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 70)

    logger.info("Loading dataset...")
    molecule_data_list, metadata = load_dataset(args.dataset)

    logger.info(f"Loaded {len(molecule_data_list)} molecules")
    logger.info(f"Dataset: {metadata.get('dataset_name', 'Unknown')}")
    logger.info(f"QM method: {metadata.get('qm_method', 'Unknown')}/{metadata.get('qm_basis', 'Unknown')}")

    splits = metadata.get('splits', {})
    train_ids = set(splits.get('train', []))
    val_ids = set(splits.get('val', []))
    test_ids = set(splits.get('test', []))

    train_idx = [i for i, mol in enumerate(molecule_data_list) if mol.molecule_id in train_ids]
    val_idx = [i for i, mol in enumerate(molecule_data_list) if mol.molecule_id in val_ids]
    test_idx = [i for i, mol in enumerate(molecule_data_list) if mol.molecule_id in test_ids]

    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    logger.info("Creating datasets...")
    split_data = create_split_datasets(
        molecule_data_list,
        train_idx, val_idx, test_idx,
        include_multipoles=not args.no_multipoles
    )

    logger.info(f"Feature dimension: {split_data.train_dataset.feature_dim}")

    model_config = ModelConfig(
        feature_units=args.feature_units,
        depth=args.depth,
        width=args.width,
        activation=args.activation
    )

    logger.info("Creating model...")
    model = ChargeModel(model_config)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    training_config = TrainingConfig(
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        eval_frequency=args.eval_frequency,
        checkpoint_dir=str(output_dir / "checkpoints"),
        early_stopping_patience=args.early_stopping_patience,
        device=args.device
    )

    logger.info("Starting training...")
    trainer = Trainer(model, training_config)

    train_loader = split_data.get_train_loader(args.batch_size, shuffle=True)
    val_loader = split_data.get_val_loader(args.batch_size)
    test_loader = split_data.get_test_loader(args.batch_size)

    results = trainer.train(train_loader, val_loader, test_loader)

    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info(f"Best validation RMSE: {results['best_val_rmse']:.5f}")
    logger.info(f"Best epoch: {results['best_epoch']}")
    if 'test_metrics' in results:
        logger.info(f"Test RMSE: {results['test_metrics']['val_rmse']:.5f}")
        logger.info(f"Test MAE: {results['test_metrics']['val_mae']:.5f}")
    logger.info("=" * 70)

    save_results(
        {
            'model_config': model_config.to_dict(),
            'training_config': training_config.to_dict(),
            'results': results
        },
        output_dir / "training_results.json"
    )

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
