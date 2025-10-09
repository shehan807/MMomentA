"""Batch processing with parallel support for molecular charge calculations."""

import os
import time
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from joblib import Parallel, delayed
from openff.toolkit.topology import Molecule

from ..qm.mpfit import MPFITCalculator, MPFITResult
from ..qm.resp import RESPCalculator, RESPResult
from ..qm.am1bcc import AM1BCCCalculator, AM1BCCResult

# Make Psi4Error picklable by registering a custom reducer
try:
    import psi4
    from psi4.driver.p4util import Psi4Error
    import copyreg

    def _pickle_psi4error(error):
        """Custom pickler for Psi4Error that converts it to a regular Exception."""
        return Exception, (str(error),)

    # Register the custom reducer
    copyreg.pickle(Psi4Error, _pickle_psi4error)
except ImportError:
    pass  # Psi4 not available

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
        backend: str = "multiprocessing",
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

            # Use batching: process molecules in batches, so if one batch fails,
            # we can retry just that batch sequentially
            # Use 2x the number of cores for batch size to minimize sequential fallback impact
            import multiprocessing
            n_cores = multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs
            batch_size = max(25, 2 * n_cores)  # At least 25, or 2x cores
            logger.info(f"Using batch size of {batch_size} molecules ({n_cores} cores × 2)")
            results = []

            for batch_start in range(0, n_molecules, batch_size):
                batch_end = min(batch_start + batch_size, n_molecules)
                batch_molecules = molecules[batch_start:batch_end]
                batch_indices = list(range(batch_start, batch_end))

                logger.info(f"Processing batch {batch_start//batch_size + 1}: molecules {batch_start}-{batch_end-1}")

                try:
                    # Try parallel processing for this batch
                    batch_results = Parallel(
                        n_jobs=self.n_jobs,
                        backend=self.backend,
                        verbose=0,  # Reduce verbosity for batches
                        timeout=300,  # 5 min timeout (reduced from 30 min)
                        batch_size=1,
                        pre_dispatch='2*n_jobs',
                        max_nbytes=None
                    )(
                        delayed(self._process_single_molecule)(idx, mol)
                        for idx, mol in zip(batch_indices, batch_molecules)
                    )
                    results.extend(batch_results)
                    logger.info(f"  ✓ Batch completed successfully in parallel")

                except Exception as e:
                    # This batch failed - check if it's a Psi4 pickling error
                    error_type = type(e).__name__
                    error_msg = str(e)

                    is_psi4_pickle_error = (
                        "BrokenProcessPool" in error_type or
                        "TimeoutError" in error_type or
                        "AssertionError" in error_type or
                        "Psi4Error" in error_msg or
                        "std_error" in error_msg or
                        "result_handler" in error_msg or
                        "PicklingError" in error_msg
                    )

                    if is_psi4_pickle_error:
                        logger.warning(
                            f"  ✗ Batch failed with pickling error ({error_type}). "
                            f"Identifying problematic molecule(s) via binary search..."
                        )

                        # Binary search to find problematic molecule(s)
                        batch_results, bad_indices = self._isolate_bad_molecules(
                            batch_molecules, batch_indices
                        )

                        results.extend(batch_results)
                        logger.info(
                            f"  ✓ Batch completed with {len(bad_indices)} molecule(s) skipped "
                            f"(indices: {bad_indices})"
                        )
                    else:
                        # Different error - re-raise
                        logger.error(f"Batch failed with unexpected error: {error_type}")
                        raise

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

    def _isolate_bad_molecules(
        self,
        batch_molecules: List[Molecule],
        batch_indices: List[int]
    ) -> Tuple[List[MoleculeResult], List[int]]:
        """Identify problematic molecules via binary search, process the rest.

        Strategy:
        1. Use binary search to find molecule(s) causing pickling errors
        2. Process good molecules in parallel
        3. Mark bad molecules as failed

        Parameters
        ----------
        batch_molecules : list of Molecule
            Molecules that failed when processed together
        batch_indices : list of int
            Original indices of batch molecules

        Returns
        -------
        results : list of MoleculeResult
            Results for all molecules (successful + failed)
        bad_indices : list of int
            Indices of molecules that couldn't be processed
        """
        logger.info(f"    Binary search: testing {len(batch_molecules)} molecules...")

        bad_indices = set()
        good_molecules = []
        good_indices = []

        # Binary search to partition good from bad
        # IMPORTANT: Test sequentially to avoid nested pickling issues
        def test_subset(mols, indices):
            """Try processing a subset sequentially. Returns True if successful."""
            if len(mols) == 0:
                return True

            try:
                # Process sequentially (no parallelization during search)
                # This avoids pickling errors and timeouts during binary search
                for idx, mol in zip(indices, mols):
                    result = self._process_single_molecule(idx, mol)
                    if not result.success:
                        return False  # Failed
                return True  # All succeeded
            except Exception as e:
                # Any exception means this subset contains a bad molecule
                logger.debug(f"        Subset test failed: {type(e).__name__}")
                return False

        # Recursive binary search
        def find_bad_molecules(mols, indices, depth=0):
            if len(mols) == 0:
                return

            if len(mols) == 1:
                # Base case: single molecule
                if test_subset(mols, indices):
                    good_molecules.append(mols[0])
                    good_indices.append(indices[0])
                    logger.info(f"      {'  '*depth}✓ Molecule {indices[0]} is OK")
                else:
                    bad_indices.add(indices[0])
                    logger.warning(f"      {'  '*depth}✗ Molecule {indices[0]} FAILS (SMILES: {mols[0].to_smiles(mapped=False)})")
                return

            # Divide and conquer
            mid = len(mols) // 2
            left_mols, right_mols = mols[:mid], mols[mid:]
            left_idx, right_idx = indices[:mid], indices[mid:]

            logger.info(f"      {'  '*depth}Testing left half ({len(left_mols)} mols)...")
            if not test_subset(left_mols, left_idx):
                find_bad_molecules(left_mols, left_idx, depth+1)
            else:
                good_molecules.extend(left_mols)
                good_indices.extend(left_idx)
                logger.info(f"      {'  '*depth}✓ Left half OK")

            logger.info(f"      {'  '*depth}Testing right half ({len(right_mols)} mols)...")
            if not test_subset(right_mols, right_idx):
                find_bad_molecules(right_mols, right_idx, depth+1)
            else:
                good_molecules.extend(right_mols)
                good_indices.extend(right_idx)
                logger.info(f"      {'  '*depth}✓ Right half OK")

        # Execute binary search
        find_bad_molecules(batch_molecules, batch_indices)

        # Now process good molecules in parallel (we know they work)
        results = []
        if good_molecules:
            logger.info(f"    Processing {len(good_molecules)} good molecules in parallel...")
            good_results = Parallel(
                n_jobs=self.n_jobs,
                backend=self.backend,
                verbose=0
            )(
                delayed(self._process_single_molecule)(idx, mol)
                for idx, mol in zip(good_indices, good_molecules)
            )
            results.extend(good_results)

        # Create failed results for bad molecules
        for bad_idx in sorted(bad_indices):
            mol = batch_molecules[batch_indices.index(bad_idx)]
            smiles = mol.to_smiles(mapped=False)
            results.append(MoleculeResult(
                molecule_index=bad_idx,
                smiles=smiles,
                formula=mol.hill_formula,
                n_atoms=mol.n_atoms,
                results={method: {"error": "Causes pickling error in parallel processing", "time": 0.0}
                        for method in self.calculators.keys()},
                success=False,
                partial_success=False,
                failed_methods=list(self.calculators.keys())
            ))

        # Print summary
        logger.info(f"    Binary search complete:")
        logger.info(f"      ✓ Successfully processed: {len(good_molecules)} molecules")
        logger.info(f"      ✗ Failed (removed): {len(bad_indices)} molecules")
        if bad_indices:
            logger.warning(f"    Problematic SMILES removed from batch:")
            for bad_idx in sorted(bad_indices):
                mol = batch_molecules[batch_indices.index(bad_idx)]
                smiles = mol.to_smiles(mapped=False)
                logger.warning(f"      • Index {bad_idx}: {smiles}")

        return results, sorted(bad_indices)

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
