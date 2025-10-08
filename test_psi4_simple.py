#!/usr/bin/env python
"""Simple test - just verify the fallback message appears."""

import sys
import logging

# Set up logging to see the fallback message
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from openff.toolkit import Molecule
from MMomentA.data.batch import BatchProcessor
from MMomentA.qm.mpfit import MPFITCalculator

print("Creating normal molecule...")
mol = Molecule.from_smiles("C")
mol.generate_conformers(n_conformers=1)

print(f"Test molecule: {mol.to_smiles()}")

# Create batch processor
calculator = MPFITCalculator()
processor = BatchProcessor(
    calculators={"MPFIT": calculator},
    n_jobs=2,
    backend="multiprocessing",
    verbose=1
)

print("\nProcessing with parallel backend...")
print("If Psi4Error pickling fails, you should see a fallback message\n")

try:
    results = processor.process([mol])
    
    if results:
        print("\n✓ Processing completed")
        print(f"Success: {results[0].success}")
        if not results[0].success and "MPFIT" in results[0].results:
            print(f"Error: {results[0].results['MPFIT'].get('error', 'No error')[:100]}")
    
except KeyboardInterrupt:
    print("\n\n✗ Test interrupted (likely hanging on Psi4 calculation)")
    print("The hybrid approach may still work on real pipeline")
    sys.exit(1)

except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {str(e)[:200]}")
    sys.exit(1)

print("\n✓ Test complete")
