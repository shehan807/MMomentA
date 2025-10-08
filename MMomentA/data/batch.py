"""Batch processing with parallel support for molecular charge calculations."""

import os
import time
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Union
from dataclasses import dataclass, asdict

import numpy as np
from joblib import Parallel, delayed
from openff.toolkit.topology import Molecule

from ..qm.mpfit import MPFITCalculator, MPFITResult
from ..qm.resp import RESPCalculator, RESPResult
from ..qm.am1bcc import AM1BCCCalculator, AM1BCCResult


logger = logging.getLogger(__name__)


@dataclass
class MoleculeResult:
    """Container for complete molecule calculation results.

    Attributes
    ----------
    molecule_index : int
        Index in input molecule list
    smiles : str
        Molecule SMILES string
    formula : str
        Molecular formula
    n_atoms : int
        Number of atoms
    results : dict
        Results from each calculator method
    success : bool
        Whether all calculations succeeded
    partial_success : bool
        Whether some calculations succeeded
    failed_methods : list
        List of failed method names
    """
    molecule_index: int
    smiles: str
    formula: str
    n_atoms: int
    results: Dict[str, Any]
    success: bool
    partial_success: bool
    failed_methods: List[str]


class BatchProcessor:
    """Batch processor for molecular charge calculations with parallel support.

    Parameters
    ----------
    calculators : dict, optional
        Dictionary mapping method names to calculator instances
    n_jobs : int, optional
        Number of parallel jobs (-1 for all CPUs)
    backend : str, optional
        Joblib backend ('loky', 'multiprocessing', 'threading')
    verbose : int, optional
        Verbosity level for joblib

    Examples
    --------
    >>> from mmomenta.qm import MPFITCalculator, RESPCalculator
    >>> calculators = {
    ...     "MPFIT": MPFITCalculator(),
    ...     "RESP": RESPCalculator()
    ... }
    >>> processor = BatchProcessor(calculators, n_jobs=4)
    >>> results = processor.process(molecules)
    """

    def __init__(
        self,
        calculators: Optional[Dict[str, Any]] = None,
        n_jobs: int = -1,
        backend: str = "loky",
        verbose: int = 1
    ):
        self.calculators = calculators or self._default_calculators()
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose

    @staticmethod
    def _default_calculators() -> Dict[str, Any]:
        """Create default calculator instances."""
        return {
            "AM1-BCC": AM1BCCCalculator(),
            "RESP": RESPCalculator(),
            "MPFIT": MPFITCalculator()
        }

    def process(self, molecules: List[Molecule]) -> List[MoleculeResult]:
        """Process multiple molecules in parallel.

        Parameters
        ----------
        molecules : list of Molecule
            OpenFF molecules to process

        Returns
        -------
        list of MoleculeResult
            Results for each molecule
        """
        start_time = time.time()
        n_molecules = len(molecules)

        logger.info(f"Starting batch processing of {n_molecules} molecules")
        logger.info(f"Using {self.n_jobs} parallel jobs with {self.backend} backend")
        logger.info(f"Calculators: {list(self.calculators.keys())}")

        if self.n_jobs == 1:
            logger.info("Running in sequential mode")
            results = [
                self._process_single_molecule(i, mol)
                for i, mol in enumerate(molecules)
            ]
        else:
            logger.info(f"Running in parallel mode with {self.n_jobs} jobs")
            results = Parallel(
                n_jobs=self.n_jobs,
                backend=self.backend,
                verbose=self.verbose,
                timeout=1800,
                batch_size=1,
                pre_dispatch='2*n_jobs',
                max_nbytes=None
            )(
                delayed(self._process_single_molecule)(i, mol)
                for i, mol in enumerate(molecules)
            )

        elapsed = time.time() - start_time

        successful = sum(1 for r in results if r.success)
        partial = sum(1 for r in results if r.partial_success)
        failed = n_molecules - successful

        logger.info(f"Batch processing complete in {elapsed:.2f}s")
        logger.info(f"Average time per molecule: {elapsed/n_molecules:.2f}s")
        logger.info(f"Fully successful: {successful}/{n_molecules}")
        logger.info(f"Partially successful: {partial}")
        logger.info(f"Complete failures: {failed}")

        return results

    def _process_single_molecule(self, mol_idx: int, molecule: Molecule) -> MoleculeResult:
        """Process a single molecule with all calculators.

        Parameters
        ----------
        mol_idx : int
            Molecule index
        molecule : Molecule
            OpenFF molecule

        Returns
        -------
        MoleculeResult
            Results from all calculators
        """
        pid = os.getpid()
        logger.debug(f"[Process {pid}] Starting molecule {mol_idx}")

        original_cwd = os.getcwd()

        try:
            with tempfile.TemporaryDirectory(prefix=f"mol_{mol_idx}_pid_{pid}_") as temp_dir:
                os.chdir(temp_dir)
                os.environ['PSI_SCRATCH'] = temp_dir

                if not molecule.conformers:
                    logger.debug(f"[Process {pid}] Generating conformer for molecule {mol_idx}")
                    molecule.generate_conformers(n_conformers=1)

                results = {}
                failed_methods = []

                for method_name, calculator in self.calculators.items():
                    logger.debug(f"[Process {pid}] Running {method_name} on molecule {mol_idx}")

                    try:
                        result = calculator.compute(molecule)

                        if result.success:
                            results[method_name] = self._serialize_result(result)
                            logger.debug(
                                f"[Process {pid}] {method_name} succeeded in {result.time_seconds:.2f}s"
                            )
                        else:
                            failed_methods.append(method_name)
                            # Convert error_message to string to avoid pickling issues
                            error_msg = str(result.error_message) if result.error_message else "Unknown error"
                            logger.warning(
                                f"[Process {pid}] {method_name} failed: {error_msg}"
                            )
                            results[method_name] = {
                                "error": error_msg,
                                "time": result.time_seconds
                            }
                    except Exception as e:
                        # Catch any exceptions and convert to string to avoid pickling issues
                        failed_methods.append(method_name)
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        logger.warning(
                            f"[Process {pid}] {method_name} raised exception: {error_msg}"
                        )
                        results[method_name] = {
                            "error": error_msg,
                            "time": 0.0
                        }

                os.chdir(original_cwd)

                return MoleculeResult(
                    molecule_index=mol_idx,
                    smiles=molecule.to_smiles(mapped=False),
                    formula=molecule.hill_formula,
                    n_atoms=molecule.n_atoms,
                    results=results,
                    success=len(failed_methods) == 0,
                    partial_success=0 < len(failed_methods) < len(self.calculators),
                    failed_methods=failed_methods
                )

        except Exception as e:
            # Top-level exception handler to catch any unpicklable exceptions (e.g., Psi4Error)
            # and return a safe, picklable result
            os.chdir(original_cwd)
            error_msg = f"Critical error in molecule processing: {type(e).__name__}: {str(e)}"
            logger.error(f"[Process {pid}] {error_msg}")

            return MoleculeResult(
                molecule_index=mol_idx,
                smiles=molecule.to_smiles(mapped=False) if hasattr(molecule, 'to_smiles') else "UNKNOWN",
                formula=molecule.hill_formula if hasattr(molecule, 'hill_formula') else "UNKNOWN",
                n_atoms=molecule.n_atoms if hasattr(molecule, 'n_atoms') else 0,
                results={method: {"error": error_msg, "time": 0.0} for method in self.calculators.keys()},
                success=False,
                partial_success=False,
                failed_methods=list(self.calculators.keys())
            )

    @staticmethod
    def _serialize_result(result: Union[MPFITResult, RESPResult, AM1BCCResult]) -> Dict[str, Any]:
        """Serialize calculation result to dictionary.

        Parameters
        ----------
        result : MPFITResult, RESPResult, or AM1BCCResult
            Calculation result

        Returns
        -------
        dict
            Serialized result
        """
        serialized = {
            "charges": result.charges.tolist(),
            "time": result.time_seconds,
            "smiles": result.smiles,
            "metadata": result.metadata
        }

        if isinstance(result, MPFITResult) and result.multipole_moments is not None:
            serialized["multipole_moments"] = result.multipole_moments.tolist()

        if hasattr(result, 'conformer'):
            serialized["conformer"] = result.conformer.tolist()

        return serialized


def batch_process_molecules(
    molecules: List[Molecule],
    calculators: Optional[Dict[str, Any]] = None,
    n_jobs: int = -1,
    backend: str = "loky"
) -> List[MoleculeResult]:
    """Convenience function for batch processing molecules.

    Parameters
    ----------
    molecules : list of Molecule
        OpenFF molecules to process
    calculators : dict, optional
        Dictionary mapping method names to calculator instances
    n_jobs : int, optional
        Number of parallel jobs (-1 for all CPUs)
    backend : str, optional
        Joblib backend ('loky', 'multiprocessing', 'threading')

    Returns
    -------
    list of MoleculeResult
        Results for each molecule

    Examples
    --------
    >>> from openff.toolkit.topology import Molecule
    >>> molecules = [Molecule.from_smiles(s) for s in ["CCO", "c1ccccc1"]]
    >>> for mol in molecules:
    ...     mol.generate_conformers(n_conformers=1)
    >>> results = batch_process_molecules(molecules, n_jobs=2)
    """
    processor = BatchProcessor(calculators, n_jobs, backend)
    return processor.process(molecules)
