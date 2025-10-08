"""Neural network architectures for charge prediction.

Adapted from espaloma-charge with support for multipole moment features.
"""

import torch
import torch.nn as nn
from functools import partial
from typing import Optional, List

import dgl
import dgl.nn


class _Sequential(nn.Module):
    """Sequentially staggered neural networks (from espaloma-charge)."""

    def __init__(self, layer, config, in_features, model_kwargs=None):
        super(_Sequential, self).__init__()

        if model_kwargs is None:
            model_kwargs = {}

        self.exes = []
        dim = in_features

        for idx, exe in enumerate(config):
            if isinstance(exe, (int, float)):
                exe = float(exe)
                if exe >= 1:
                    exe = int(exe)

            if isinstance(exe, int):
                setattr(self, "d" + str(idx), layer(dim, exe, **model_kwargs))
                dim = exe
                self.exes.append("d" + str(idx))

            elif isinstance(exe, str):
                if exe == "bn":
                    setattr(self, "a" + str(idx), nn.BatchNorm1d(dim))
                else:
                    activation = getattr(torch.nn.functional, exe)
                    setattr(self, "a" + str(idx), activation)
                self.exes.append("a" + str(idx))

            elif isinstance(exe, float):
                dropout = nn.Dropout(exe)
                setattr(self, "o" + str(idx), dropout)
                self.exes.append("o" + str(idx))

    def forward(self, g, x, **kwargs):
        """Forward pass through sequential layers."""
        for exe in self.exes:
            if exe.startswith("d"):
                if g is not None:
                    x = getattr(self, exe)(g, x)
                else:
                    x = getattr(self, exe)(x)
            else:
                x = getattr(self, exe)(x)
        return x


class Sequential(nn.Module):
    """Sequential neural network with input featurization.

    Parameters
    ----------
    layer : nn.Module
        DGL graph convolution layer (e.g., SAGEConv, GATConv)
    config : list
        Layer configuration [units, activation, ...]
    feature_units : int
        Number of input features (117 baseline, 198 with multipoles)
    input_units : int
        Hidden dimension after initial featurization
    model_kwargs : dict
        Additional arguments for graph conv layers
    """

    def __init__(
        self,
        layer,
        config,
        feature_units=117,
        input_units=128,
        model_kwargs=None
    ):
        super(Sequential, self).__init__()

        if model_kwargs is None:
            model_kwargs = {}

        self.f_in = nn.Sequential(
            nn.Linear(feature_units, input_units),
            nn.Tanh()
        )

        self._sequential = _Sequential(
            layer, config, in_features=input_units, model_kwargs=model_kwargs
        )

    def forward(self, g, x=None, **kwargs):
        """Forward pass.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input graph with node features
        x : torch.Tensor, optional
            Node features (if None, uses g.ndata['h0'])

        Returns
        -------
        dgl.DGLGraph
            Graph with updated node features in g.ndata['h']
        """
        if x is None:
            x = g.ndata["h0"]
            x = self.f_in(x)

        x = self._sequential(g, x)

        g.ndata["h"] = x

        return g


class ChargeReadout(nn.Module):
    """Readout layer for electronegativity and hardness parameters.

    This layer predicts per-atom electronegativity (e) and hardness (s)
    parameters that are then used in the charge equilibration layer.

    Parameters
    ----------
    in_features : int
        Input feature dimension
    """

    def __init__(self, in_features):
        super().__init__()
        self.fc_params = nn.Linear(in_features, 2)

    def forward(self, g, **kwargs):
        """Forward pass.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input graph with node features in g.ndata['h']

        Returns
        -------
        dgl.DGLGraph
            Graph with e and s parameters in g.ndata
        """
        h = self.fc_params(g.ndata["h"])
        e, s = h.split(1, -1)
        g.ndata["e"], g.ndata["s"] = e, s
        return g


def get_charges(node):
    """Solve for atomic charges from electronegativity and hardness.

    Uses Lagrange multipliers for analytical solution to charge equilibration.

    The energy functional is:
    U(q) = sum_i [e_i * q_i + 0.5 * s_i * q_i^2]
    subject to: sum_i q_i = Q_total

    Solution:
    q_i = -e_i / s_i + (1/s_i) * (Q + sum_j e_j/s_j) / sum_k (1/s_k)
    """
    e = node.data["e"]
    s = node.data["s"]
    sum_e_s_inv = node.data["sum_e_s_inv"]
    sum_s_inv = node.data["sum_s_inv"]
    sum_q = node.data["sum_q"]

    return {
        "q": -e * s**-1 + (s**-1) * torch.div(sum_q + sum_e_s_inv, sum_s_inv)
    }


class ChargeEquilibrium(nn.Module):
    """Charge equilibration layer using analytical solution.

    Ensures total charge conservation within each molecule in the batch.
    """

    def __init__(self):
        super(ChargeEquilibrium, self).__init__()

    def forward(self, g, total_charge=0.0):
        """Apply charge equilibration.

        Parameters
        ----------
        g : dgl.DGLGraph
            Batched graph with e and s parameters
        total_charge : float
            Default total charge (used if q_ref not present)

        Returns
        -------
        dgl.DGLGraph
            Graph with equilibrated charges in g.ndata['q']
        """
        g.apply_nodes(lambda node: {"s_inv": node.data["s"] ** -1})

        g.apply_nodes(lambda node: {"e_s_inv": node.data["e"] * node.data["s"] ** -1})

        if "q_ref" in g.ndata:
            total_charge = dgl.sum_nodes(g, "q_ref")
        else:
            total_charge = torch.ones(g.batch_size, 1, device=g.device) * total_charge

        g.ndata["sum_q"] = dgl.broadcast_nodes(g, total_charge)

        sum_s_inv = dgl.sum_nodes(g, "s_inv")
        sum_e_s_inv = dgl.sum_nodes(g, "e_s_inv")
        g.ndata["sum_s_inv"] = dgl.broadcast_nodes(g, sum_s_inv)
        g.ndata["sum_e_s_inv"] = dgl.broadcast_nodes(g, sum_e_s_inv)

        g.apply_nodes(get_charges)

        return g


class ChargeModel(nn.Module):
    """Complete charge prediction model.

    Combines graph convolution, readout, and charge equilibration.

    Parameters
    ----------
    config : ModelConfig
        Model configuration

    Examples
    --------
    >>> from MMomentA.models import ChargeModel, ModelConfig
    >>> config = ModelConfig(feature_units=198, width=128, depth=4)
    >>> model = ChargeModel(config)
    >>> predictions = model(batch_graph)
    >>> charges = predictions.ndata['q']
    """

    def __init__(self, config):
        super(ChargeModel, self).__init__()

        from .config import ModelConfig
        self.config = config

        layer_config = config.get_layer_config()

        self.gnn = Sequential(
            layer=partial(dgl.nn.SAGEConv, aggregator_type=config.aggregator_type),
            config=layer_config,
            feature_units=config.feature_units,
            input_units=config.input_units
        )

        self.readout = ChargeReadout(config.width)
        self.equilibrium = ChargeEquilibrium()

    def forward(self, g):
        """Forward pass.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input graph with node features

        Returns
        -------
        dgl.DGLGraph
            Graph with predicted charges in g.ndata['q']
        """
        g = self.gnn(g)
        g = self.readout(g)
        g = self.equilibrium(g)
        return g

    def get_config(self):
        """Get model configuration."""
        return self.config.to_dict()
