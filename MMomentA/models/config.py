"""Model configuration for charge prediction networks."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for charge prediction model.

    Parameters
    ----------
    feature_units : int
        Input feature dimension (117 baseline, 198 with multipoles)
    input_units : int
        Hidden dimension after initial featurization
    depth : int
        Number of graph convolution layers
    width : int
        Hidden dimension for graph convolution layers
    activation : str
        Activation function ('relu', 'tanh', 'elu')
    aggregator_type : str
        Aggregation for SAGEConv ('mean', 'max', 'lstm')
    dropout : float
        Dropout probability (0 = no dropout)
    batch_norm : bool
        Use batch normalization

    Examples
    --------
    >>> # Baseline model (no multipoles)
    >>> config = ModelConfig(feature_units=117, width=128, depth=4)

    >>> # Model with multipoles
    >>> config = ModelConfig(feature_units=198, width=128, depth=4)
    """
    feature_units: int = 117
    input_units: int = 128
    depth: int = 4
    width: int = 128
    activation: str = "relu"
    aggregator_type: str = "mean"
    dropout: float = 0.0
    batch_norm: bool = False

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'feature_units': self.feature_units,
            'input_units': self.input_units,
            'depth': self.depth,
            'width': self.width,
            'activation': self.activation,
            'aggregator_type': self.aggregator_type,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Load from dictionary."""
        return cls(**config_dict)

    def get_layer_config(self) -> List:
        """Get layer configuration list for Sequential model.

        Returns
        -------
        list
            Configuration in espaloma-charge format: [width, activation, ...]
        """
        config = []
        for _ in range(self.depth):
            config.append(self.width)
            config.append(self.activation)
            if self.dropout > 0:
                config.append(self.dropout)
            if self.batch_norm:
                config.append('bn')

        return config
