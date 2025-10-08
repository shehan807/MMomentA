"""Evaluation and analysis tools for trained models."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from .metrics import compute_metrics


@torch.no_grad()
def evaluate_model(model, data_loader, device='cpu') -> Dict:
    """Comprehensive model evaluation.

    Parameters
    ----------
    model : nn.Module
        Trained model
    data_loader : DataLoader
        Data loader for evaluation
    device : str
        Device to use

    Returns
    -------
    dict
        Comprehensive evaluation results
    """
    model.eval()
    model.to(device)

    all_predictions = []
    all_targets = []
    all_molecules = []

    for batch_graph in data_loader:
        batch_graph = batch_graph.to(device)
        batch_graph = model(batch_graph)

        predictions = batch_graph.ndata["q"].cpu()
        targets = batch_graph.ndata["q_ref"].cpu()

        all_predictions.append(predictions)
        all_targets.append(targets)

        batch_sizes = batch_graph.batch_num_nodes().cpu().numpy()
        all_molecules.extend(batch_sizes)

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    overall_metrics = compute_metrics(predictions, targets)

    per_molecule_metrics = compute_per_molecule_metrics(
        predictions.numpy(),
        targets.numpy(),
        all_molecules
    )

    return {
        'overall': overall_metrics.to_dict(),
        'per_molecule': per_molecule_metrics,
        'n_molecules': len(all_molecules),
        'n_atoms': len(predictions)
    }


def compute_per_molecule_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    molecule_sizes: List[int]
) -> Dict:
    """Compute per-molecule statistics.

    Parameters
    ----------
    predictions : np.ndarray
        All predicted charges
    targets : np.ndarray
        All target charges
    molecule_sizes : list of int
        Number of atoms per molecule

    Returns
    -------
    dict
        Per-molecule statistics
    """
    rmses = []
    maes = []

    start_idx = 0
    for mol_size in molecule_sizes:
        end_idx = start_idx + mol_size

        mol_pred = predictions[start_idx:end_idx]
        mol_target = targets[start_idx:end_idx]

        diff = mol_pred - mol_target
        rmse = np.sqrt(np.mean(diff ** 2))
        mae = np.mean(np.abs(diff))

        rmses.append(rmse)
        maes.append(mae)

        start_idx = end_idx

    return {
        'rmse_mean': float(np.mean(rmses)),
        'rmse_std': float(np.std(rmses)),
        'rmse_median': float(np.median(rmses)),
        'rmse_min': float(np.min(rmses)),
        'rmse_max': float(np.max(rmses)),
        'mae_mean': float(np.mean(maes)),
        'mae_std': float(np.std(maes)),
        'mae_median': float(np.median(maes)),
        'rmses': rmses,
        'maes': maes
    }


def compare_models(
    models: Dict[str, torch.nn.Module],
    data_loader,
    device='cpu'
) -> Dict:
    """Compare multiple models on same dataset.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names to models
    data_loader : DataLoader
        Data loader for evaluation
    device : str
        Device to use

    Returns
    -------
    dict
        Comparison results for all models

    Examples
    --------
    >>> models = {
    ...     'baseline': baseline_model,
    ...     'with_multipoles': multipole_model
    ... }
    >>> results = compare_models(models, test_loader)
    >>> print(results['baseline']['overall']['rmse'])
    """
    results = {}

    for name, model in models.items():
        results[name] = evaluate_model(model, data_loader, device)

    return results


def analyze_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    molecule_sizes: List[int],
    output_dir: Optional[Path] = None
) -> Dict:
    """Detailed analysis of predictions vs targets.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted charges
    targets : np.ndarray
        Target charges
    molecule_sizes : list of int
        Number of atoms per molecule
    output_dir : Path, optional
        Directory to save analysis plots

    Returns
    -------
    dict
        Analysis results
    """
    diff = predictions - targets

    analysis = {
        'overall_rmse': float(np.sqrt(np.mean(diff ** 2))),
        'overall_mae': float(np.mean(np.abs(diff))),
        'max_abs_error': float(np.max(np.abs(diff))),
        'mean_error': float(np.mean(diff)),
        'std_error': float(np.std(diff)),
        'correlation': float(np.corrcoef(predictions, targets)[0, 1])
    }

    charge_conservation_errors = []
    start_idx = 0
    for mol_size in molecule_sizes:
        end_idx = start_idx + mol_size

        mol_pred_sum = np.sum(predictions[start_idx:end_idx])
        mol_target_sum = np.sum(targets[start_idx:end_idx])

        charge_conservation_errors.append(abs(mol_pred_sum - mol_target_sum))

        start_idx = end_idx

    analysis['charge_conservation'] = {
        'mean_error': float(np.mean(charge_conservation_errors)),
        'max_error': float(np.max(charge_conservation_errors)),
        'n_violations': int(np.sum(np.array(charge_conservation_errors) > 0.01))
    }

    if output_dir is not None:
        save_analysis(analysis, output_dir)

    return analysis


def save_analysis(analysis: Dict, output_dir: Path):
    """Save analysis results to file.

    Parameters
    ----------
    analysis : dict
        Analysis results
    output_dir : Path
        Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "analysis.json", 'w') as f:
        json_safe = {
            k: v for k, v in analysis.items()
            if not isinstance(v, (np.ndarray, list))
        }
        json.dump(json_safe, f, indent=2)


def ablation_study(
    dataset,
    model_configs: Dict[str, Dict],
    training_config,
    n_runs: int = 3
) -> Dict:
    """Run ablation study comparing different configurations.

    Parameters
    ----------
    dataset : SplitDataset
        Dataset with train/val/test splits
    model_configs : dict
        Dictionary mapping config names to model configurations
    training_config : TrainingConfig
        Training configuration
    n_runs : int
        Number of runs per configuration

    Returns
    -------
    dict
        Ablation study results

    Examples
    --------
    >>> configs = {
    ...     'baseline': ModelConfig(feature_units=117),
    ...     'with_multipoles': ModelConfig(feature_units=198)
    ... }
    >>> results = ablation_study(dataset, configs, training_config)
    """
    from ..models import ChargeModel
    from .trainer import Trainer

    results = {}

    for config_name, model_config in model_configs.items():
        config_results = []

        for run in range(n_runs):
            model = ChargeModel(model_config)

            trainer = Trainer(model, training_config)

            train_loader = dataset.get_train_loader(training_config.batch_size)
            val_loader = dataset.get_val_loader(training_config.batch_size)
            test_loader = dataset.get_test_loader(training_config.batch_size)

            run_results = trainer.train(train_loader, val_loader, test_loader)

            config_results.append(run_results)

        results[config_name] = {
            'runs': config_results,
            'mean_test_rmse': float(np.mean([
                r['test_metrics']['val_rmse'] for r in config_results
            ])),
            'std_test_rmse': float(np.std([
                r['test_metrics']['val_rmse'] for r in config_results
            ]))
        }

    return results
