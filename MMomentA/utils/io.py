"""Input/output utilities for saving and loading results."""

import json
import pickle
from pathlib import Path
from typing import Any, Union


def save_results(
    results: Any,
    output_path: Union[str, Path],
    format: str = "json"
):
    """Save calculation results to file.

    Parameters
    ----------
    results : Any
        Results to save (must be JSON-serializable for JSON format)
    output_path : str or Path
        Output file path
    format : str
        Output format ('json' or 'pickle')

    Examples
    --------
    >>> results = {"molecule": "CCO", "charges": [0.1, 0.2, -0.3]}
    >>> save_results(results, "results.json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == "pickle":
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(
    input_path: Union[str, Path],
    format: str = "json"
) -> Any:
    """Load calculation results from file.

    Parameters
    ----------
    input_path : str or Path
        Input file path
    format : str
        Input format ('json' or 'pickle')

    Returns
    -------
    Any
        Loaded results

    Examples
    --------
    >>> results = load_results("results.json")
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if format == "json":
        with open(input_path, 'r') as f:
            return json.load(f)
    elif format == "pickle":
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")
