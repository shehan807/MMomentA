"""Metrics for evaluating charge predictions."""

import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class Metrics:
    """Container for evaluation metrics.

    Attributes
    ----------
    rmse : float
        Root mean squared error
    mae : float
        Mean absolute error
    max_error : float
        Maximum absolute error
    """
    rmse: float
    mae: float
    max_error: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'max_error': self.max_error
        }


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Metrics:
    """Compute evaluation metrics for charge predictions.

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted charges
    targets : torch.Tensor
        Target charges

    Returns
    -------
    Metrics
        Evaluation metrics

    Examples
    --------
    >>> predictions = model(batch_graph).ndata['q']
    >>> targets = batch_graph.ndata['q_ref']
    >>> metrics = compute_metrics(predictions, targets)
    >>> print(f"RMSE: {metrics.rmse:.4f}")
    """
    diff = predictions - targets

    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    mae = torch.mean(torch.abs(diff)).item()
    max_error = torch.max(torch.abs(diff)).item()

    return Metrics(rmse=rmse, mae=mae, max_error=max_error)


class MetricsTracker:
    """Track metrics across training epochs.

    Parameters
    ----------
    metrics_names : list of str
        Names of metrics to track

    Examples
    --------
    >>> tracker = MetricsTracker(['train_rmse', 'val_rmse'])
    >>> tracker.update({'train_rmse': 0.05, 'val_rmse': 0.06})
    >>> best_epoch = tracker.get_best_epoch('val_rmse')
    """

    def __init__(self, metrics_names: List[str] = None):
        if metrics_names is None:
            metrics_names = ['train_rmse', 'train_mae', 'val_rmse', 'val_mae']

        self.metrics_names = metrics_names
        self.history = {name: [] for name in metrics_names}
        self.current_epoch = 0

    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics for current epoch.

        Parameters
        ----------
        metrics_dict : dict
            Metrics values for current epoch
        """
        for name in self.metrics_names:
            if name in metrics_dict:
                self.history[name].append(metrics_dict[name])

        self.current_epoch += 1

    def get_history(self, metric_name: str) -> List[float]:
        """Get history for a specific metric.

        Parameters
        ----------
        metric_name : str
            Name of metric

        Returns
        -------
        list of float
            Metric values across epochs
        """
        return self.history.get(metric_name, [])

    def get_best_epoch(self, metric_name: str, minimize: bool = True) -> int:
        """Get epoch with best metric value.

        Parameters
        ----------
        metric_name : str
            Name of metric
        minimize : bool
            Whether lower is better (default: True)

        Returns
        -------
        int
            Epoch with best value
        """
        values = self.history.get(metric_name, [])
        if not values:
            return -1

        if minimize:
            return int(np.argmin(values))
        else:
            return int(np.argmax(values))

    def get_best_value(self, metric_name: str, minimize: bool = True) -> float:
        """Get best metric value.

        Parameters
        ----------
        metric_name : str
            Name of metric
        minimize : bool
            Whether lower is better (default: True)

        Returns
        -------
        float
            Best value
        """
        values = self.history.get(metric_name, [])
        if not values:
            return float('inf') if minimize else float('-inf')

        if minimize:
            return float(np.min(values))
        else:
            return float(np.max(values))

    def get_current_metrics(self) -> Dict[str, float]:
        """Get latest metrics values.

        Returns
        -------
        dict
            Latest metrics
        """
        return {
            name: values[-1] if values else None
            for name, values in self.history.items()
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'metrics_names': self.metrics_names,
            'history': self.history,
            'current_epoch': self.current_epoch
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MetricsTracker':
        """Load from dictionary."""
        tracker = cls(data['metrics_names'])
        tracker.history = data['history']
        tracker.current_epoch = data['current_epoch']
        return tracker
