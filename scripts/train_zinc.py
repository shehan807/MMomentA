#!/usr/bin/env python
"""Train charge prediction model on ZINC dataset (transfer learning test).

ZINC dataset tests transferability of the model to drug-like molecules
after training on SPICE.

Examples
--------
# Fine-tune model pre-trained on SPICE
python train_zinc.py \\
    --dataset zinc_mpfit.h5 \\
    --pretrained runs/spice_multipoles/checkpoints/best_model.pt \\
    --output-dir runs/zinc_finetune \\
    --n-epochs 500

# Train from scratch on ZINC
python train_zinc.py \\
    --dataset zinc_mpfit.h5 \\
    --output-dir runs/zinc_scratch \\
    --feature-units 198
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
        description="Train charge prediction model on ZINC dataset"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to preprocessed ZINC dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/zinc",
        help="Output directory"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint"
    )

    parser.add_argument(
        "--feature-units",
        type=int,
        default=117,
        help="Input feature dimension"
    )
    parser.add_argument(
        "--no-multipoles",
        action="store_true",
        help="Disable multipole features"
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
        help="Hidden dimension"
    )

    parser.add_argument(
        "--n-epochs",
        type=int,
        default=500,
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
        help="Weight decay"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
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
    logger.info("ZINC Dataset Training")
    logger.info("=" * 70)

    molecule_data_list, metadata = load_dataset(args.dataset)

    logger.info(f"Loaded {len(molecule_data_list)} molecules")

    splits = metadata.get('splits', {})
    train_ids = set(splits.get('train', []))
    val_ids = set(splits.get('val', []))
    test_ids = set(splits.get('test', []))

    train_idx = [i for i, mol in enumerate(molecule_data_list) if mol.molecule_id in train_ids]
    val_idx = [i for i, mol in enumerate(molecule_data_list) if mol.molecule_id in val_ids]
    test_idx = [i for i, mol in enumerate(molecule_data_list) if mol.molecule_id in test_ids]

    split_data = create_split_datasets(
        molecule_data_list,
        train_idx, val_idx, test_idx,
        include_multipoles=not args.no_multipoles
    )

    model_config = ModelConfig(
        feature_units=args.feature_units,
        depth=args.depth,
        width=args.width
    )

    model = ChargeModel(model_config)

    if args.pretrained:
        logger.info(f"Loading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Pretrained model loaded")

    training_config = TrainingConfig(
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        checkpoint_dir=str(output_dir / "checkpoints"),
        device=args.device
    )

    trainer = Trainer(model, training_config)

    train_loader = split_data.get_train_loader(args.batch_size)
    val_loader = split_data.get_val_loader(args.batch_size)
    test_loader = split_data.get_test_loader(args.batch_size)

    results = trainer.train(train_loader, val_loader, test_loader)

    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info(f"Best validation RMSE: {results['best_val_rmse']:.5f}")
    if 'test_metrics' in results:
        logger.info(f"Test RMSE: {results['test_metrics']['val_rmse']:.5f}")
    logger.info("=" * 70)

    save_results(
        {
            'model_config': model_config.to_dict(),
            'training_config': training_config.to_dict(),
            'results': results
        },
        output_dir / "training_results.json"
    )


if __name__ == "__main__":
    main()
