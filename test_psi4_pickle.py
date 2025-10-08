#!/usr/bin/env python
"""Test script to verify Psi4Error pickling fix works in parallel processing."""

import sys
from openff.toolkit import Molecule
from MMomentA.data.batch import BatchProcessor
from MMomentA.qm.mpfit import MPFITCalculator, MPFITConfig

# Create a molecule that will cause Psi4 to fail FAST
# Use a molecule with unusual elements that Psi4 can't handle with the basis set
print("Creating test molecule...")

# Use a large basis set that doesn't support all elements
# This will cause a fast failure rather than a hang
mol = Molecule.from_smiles("C1CC1")  # Simple cyclopropane
mol.generate_conformers(n_conformers=1)

print(f"Test molecule: {mol.to_smiles()}")
print(f"Atoms: {mol.n_atoms}")

# Use a basis set that will cause issues or just rely on bad geometry
# Actually, let's use the original approach but with a smaller timeout
import numpy as np
from openff.units import unit

# Create overlapping atoms (two carbons at same position)
coords = mol.conformers[0].m_as(unit.angstrom)
coords[0] = coords[1]  # Make first two atoms overlap
mol._conformers = [coords * unit.angstrom]

print("Using overlapping atoms to trigger quick Psi4 failure")

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
