#!/usr/bin/env python
"""Test script to verify Psi4Error pickling fix works in parallel processing."""

import sys
from openff.toolkit import Molecule
from MMomentA.data.batch import BatchProcessor
from MMomentA.qm.mpfit import MPFITCalculator

# Create a molecule that will likely cause Psi4 to fail
# (very large molecule or unusual geometry)
print("Creating test molecule...")

# Option 1: Molecule with bad geometry (atoms too close)
mol = Molecule.from_smiles("C")
mol.generate_conformers(n_conformers=1)

# Manually set a bad conformer (all atoms at same position to trigger error)
import numpy as np
from openff.units import unit
bad_coords = np.zeros((mol.n_atoms, 3))  # All atoms at origin
mol._conformers = [bad_coords * unit.angstrom]

# Option 2: A molecule known to cause issues (uncomment to try)
# mol = Molecule.from_smiles("C" * 50)  # Very long chain
# mol.generate_conformers(n_conformers=1)

print(f"Test molecule: {mol.to_smiles()}")
print(f"Atoms: {mol.n_atoms}")

# Create batch processor with multiprocessing backend
calculator = MPFITCalculator()
processor = BatchProcessor(
    calculators={"MPFIT": calculator},
    n_jobs=2,  # Use 2 workers to test parallel
    backend="multiprocessing",
    verbose=10
)

print("\n" + "="*70)
print("Testing parallel processing with molecule that should cause Psi4 error...")
print("="*70)

try:
    results = processor.process([mol])

    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)

    for result in results:
        if result.success:
            print("✓ Calculation succeeded (unexpected!)")
        else:
            print(f"✗ Calculation failed as expected: {result.failed_methods}")
            if "MPFIT" in result.results:
                error = result.results["MPFIT"].get("error", "No error message")
                print(f"  Error message: {error[:200]}")

    print("\n✓ SUCCESS: Psi4Error was handled correctly (no BrokenProcessPool)")
    sys.exit(0)

except Exception as e:
    print("\n" + "="*70)
    print("FAILURE:")
    print("="*70)
    print(f"✗ Exception type: {type(e).__name__}")
    print(f"✗ Error: {str(e)[:500]}")

    if "BrokenProcessPool" in str(type(e)):
        print("\n✗ FAILED: Still getting BrokenProcessPool error")
        print("The multiprocessing backend did not fix the issue")
        sys.exit(1)
    else:
        print("\n✗ FAILED: Different error occurred")
        sys.exit(1)
