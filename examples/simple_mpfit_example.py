#!/usr/bin/env python
"""Simple example: Compute MPFIT charges for a single molecule.

This demonstrates the basic usage of the MMomentA library for
computing MPFIT charges.
"""

from openff.toolkit.topology import Molecule
from MMomentA.qm import MPFITCalculator, GDMAConfig

molecule = Molecule.from_smiles("CCO")
molecule.generate_conformers(n_conformers=1)

config = GDMAConfig(method="hf", basis="6-31G*", limit=8)

calculator = MPFITCalculator(config)

result = calculator.compute(molecule)

if result.success:
    print(f"Molecule: {result.smiles}")
    print(f"Computation time: {result.time_seconds:.2f}s")
    print(f"Charges: {result.charges}")
    print(f"Multipole moments shape: {result.multipole_moments.shape}")
else:
    print(f"Calculation failed: {result.error_message}")
