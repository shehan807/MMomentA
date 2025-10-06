"""Training loop for charge prediction models."""

import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

from .metrics import compute_metrics, MetricsTracker


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Parameters
    ----------
    n_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    weight_decay : float
        L2 regularization weight
    batch_size : int
        Batch size for data loaders
    eval_frequency : int
        Evaluate on validation set every N epochs
    checkpoint_dir : str
        Directory to save checkpoints
    save_frequency : int
        Save checkpoint every N epochs
    early_stopping_patience : int
        Stop if no improvement after N epochs (0 = disabled)
    device : str
        Device to use ('cuda' or 'cpu')
    """
    n_epochs: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    eval_frequency: int = 10
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 50
    early_stopping_patience: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'n_epochs': self.n_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'eval_frequency': self.eval_frequency,
            'checkpoint_dir': self.checkpoint_dir,
            'save_frequency': self.save_frequency,
            'early_stopping_patience': self.early_stopping_patience,
            'device': self.device
        }


class Trainer:
    """Trainer for charge prediction models.

    Parameters
    ----------
    model : nn.Module
        Charge prediction model
    config : TrainingConfig
        Training configuration

    Examples
    --------
    >>> from MMomentA.models import ChargeModel, ModelConfig
    >>> from MMomentA.training import Trainer, TrainingConfig
    >>> model = ChargeModel(ModelConfig())
    >>> trainer = Trainer(model, TrainingConfig(n_epochs=500))
    >>> trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.criterion = nn.MSELoss()

        self.metrics_tracker = MetricsTracker()

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader

        Returns
        -------
        dict
            Training metrics
        """
        self.model.train()

        all_predictions = []
        all_targets = []

        for batch_graph in train_loader:
            self.optimizer.zero_grad()

            batch_graph = batch_graph.to(self.device)

            batch_graph = self.model(batch_graph)

            loss = self.criterion(
                batch_graph.ndata["q"],
                batch_graph.ndata["q_ref"]
            )

            loss.backward()
            self.optimizer.step()

            all_predictions.append(batch_graph.ndata["q"].detach())
            all_targets.append(batch_graph.ndata["q_ref"].detach())

        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)

        metrics = compute_metrics(predictions, targets)

        return {
            'train_rmse': metrics.rmse,
            'train_mae': metrics.mae,
            'train_max_error': metrics.max_error
        }

    @torch.no_grad()
    def evaluate(self, val_loader) -> Dict[str, float]:
        """Evaluate on validation set.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader

        Returns
        -------
        dict
            Validation metrics
        """
        self.model.eval()

        all_predictions = []
        all_targets = []

        for batch_graph in val_loader:
            batch_graph = batch_graph.to(self.device)

            batch_graph = self.model(batch_graph)

            all_predictions.append(batch_graph.ndata["q"].detach())
            all_targets.append(batch_graph.ndata["q_ref"].detach())

        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)

        metrics = compute_metrics(predictions, targets)

        return {
            'val_rmse': metrics.rmse,
            'val_mae': metrics.mae,
            'val_max_error': metrics.max_error
        }

    def train(
        self,
        train_loader,
        val_loader,
        test_loader: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Complete training loop.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        test_loader : DataLoader, optional
            Test data loader (evaluated at end)

        Returns
        -------
        dict
            Training results including best metrics and test results
        """
        logger.info(f"Starting training for {self.config.n_epochs} epochs")
        logger.info(f"Device: {self.device}")

        best_val_rmse = float('inf')
        epochs_without_improvement = 0

        for epoch in range(self.config.n_epochs):
            train_metrics = self.train_epoch(train_loader)

            if epoch % self.config.eval_frequency == 0:
                val_metrics = self.evaluate(val_loader)

                epoch_metrics = {**train_metrics, **val_metrics}
                self.metrics_tracker.update(epoch_metrics)

                logger.info(
                    f"Epoch {epoch}/{self.config.n_epochs} - "
                    f"Train RMSE: {train_metrics['train_rmse']:.5f}, "
                    f"Val RMSE: {val_metrics['val_rmse']:.5f}"
                )

                if val_metrics['val_rmse'] < best_val_rmse:
                    best_val_rmse = val_metrics['val_rmse']
                    epochs_without_improvement = 0

                    self.save_checkpoint(
                        Path(self.config.checkpoint_dir) / "best_model.pt",
                        epoch,
                        val_metrics
                    )
                    logger.info(f"New best model saved (Val RMSE: {best_val_rmse:.5f})")
                else:
                    epochs_without_improvement += self.config.eval_frequency

                if (self.config.early_stopping_patience > 0 and
                    epochs_without_improvement >= self.config.early_stopping_patience):
                    logger.info(
                        f"Early stopping after {epoch} epochs "
                        f"(no improvement for {epochs_without_improvement} epochs)"
                    )
                    break

            if epoch % self.config.save_frequency == 0 and epoch > 0:
                self.save_checkpoint(
                    Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt",
                    epoch
                )

        results = {
            'best_val_rmse': best_val_rmse,
            'best_epoch': self.metrics_tracker.get_best_epoch('val_rmse'),
            'metrics_history': self.metrics_tracker.to_dict()
        }

        if test_loader is not None:
            self.load_checkpoint(Path(self.config.checkpoint_dir) / "best_model.pt")
            test_metrics = self.evaluate(test_loader)
            results['test_metrics'] = test_metrics
            logger.info(f"Test RMSE: {test_metrics['val_rmse']:.5f}")

        return results

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Optional[Dict] = None
    ):
        """Save model checkpoint.

        Parameters
        ----------
        path : Path
            Checkpoint file path
        epoch : int
            Current epoch
        metrics : dict, optional
            Current metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'metrics_tracker': self.metrics_tracker.to_dict()
        }

        if metrics is not None:
            checkpoint['metrics'] = metrics

        if hasattr(self.model, 'get_config'):
            checkpoint['model_config'] = self.model.get_config()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint.

        Parameters
        ----------
        path : Path
            Checkpoint file path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'metrics_tracker' in checkpoint:
            self.metrics_tracker = MetricsTracker.from_dict(
                checkpoint['metrics_tracker']
            )

        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
