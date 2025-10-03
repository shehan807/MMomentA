#!/usr/bin/env python
"""Example: Batch process multiple molecules with parallel execution.

This demonstrates how to process multiple molecules in parallel using
all available charge calculation methods.
"""

from openff.toolkit.topology import Molecule
from MMomentA.qm import MPFITCalculator, RESPCalculator, AM1BCCCalculator
from MMomentA.qm import GDMAConfig, ESPConfig
from MMomentA.data import BatchProcessor
from MMomentA.utils import setup_logging

logger = setup_logging(level="INFO")

smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "c1ccncc1"]
molecules = [Molecule.from_smiles(s) for s in smiles_list]

for mol in molecules:
    mol.generate_conformers(n_conformers=1)

mpfit_config = GDMAConfig(method="hf", basis="6-31G*", limit=8)
resp_config = ESPConfig(method="hf", basis="6-31G*")

calculators = {
    "AM1-BCC": AM1BCCCalculator(),
    "RESP": RESPCalculator(resp_config),
    "MPFIT": MPFITCalculator(mpfit_config)
}

processor = BatchProcessor(calculators, n_jobs=2)

results = processor.process(molecules)

for result in results:
    print(f"\nMolecule {result.molecule_index}: {result.smiles}")
    print(f"  Success: {result.success}")
    print(f"  Methods: {list(result.results.keys())}")

    if result.success:
        for method_name, method_result in result.results.items():
            charges = method_result.get("charges", [])
            time_sec = method_result.get("time", 0)
            print(f"  {method_name}: {len(charges)} charges, {time_sec:.2f}s")
