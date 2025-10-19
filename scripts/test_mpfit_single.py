#!/usr/bin/env python
"""
Test MPFIT calculation on a single molecule to verify correctness.

This script runs MPFIT on ethanol and compares results to expected values.
It also inspects raw GDMA multipoles vs fitted charges to identify bugs.

Usage:
    python scripts/test_mpfit_single.py

Expected ethanol charges (AM1-BCC reference):
    C: ~-0.1 to +0.1 e
    H: ~0.0 to +0.1 e
    O: ~-0.6 to -0.7 e

If MPFIT gives values 10× larger → UNIT BUG!
"""

import sys
from pathlib import Path
import numpy as np

# Add MMomentA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openff.toolkit import Molecule
from MMomentA.qm.mpfit import MPFITCalculator
from MMomentA.qm.am1bcc import AM1BCCCalculator

# Atomic number to symbol mapping
ATOMIC_SYMBOLS = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


def main():
    print("="*80)
    print("Single Molecule MPFIT Test")
    print("="*80)
    print("")

    # Create ethanol molecule
    print("Creating ethanol molecule (CCO)...")
    molecule = Molecule.from_smiles("CCO", allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)

    print(f"  Formula: {molecule.hill_formula}")
    print(f"  N atoms: {molecule.n_atoms}")
    print(f"  Total charge: {molecule.total_charge.m}")
    print("")

    # Compute AM1-BCC charges for reference
    print("Computing AM1-BCC charges (reference)...")
    try:
        am1bcc_calculator = AM1BCCCalculator()
        am1bcc_result = am1bcc_calculator.compute(molecule)

        if am1bcc_result.success:
            am1bcc_charges = am1bcc_result.charges
            print(f"  ✓ AM1-BCC successful")
            print(f"  Time: {am1bcc_result.time_seconds:.2f} s")
        else:
            am1bcc_charges = None
            print(f"  ✗ AM1-BCC failed: {am1bcc_result.error_message}")
    except Exception as e:
        am1bcc_charges = None
        print(f"  ✗ AM1-BCC error: {e}")

    print("")

    # Compute MPFIT charges
    print("Computing MPFIT charges...")
    try:
        mpfit_calculator = MPFITCalculator()
        mpfit_result = mpfit_calculator.compute(molecule)

        if mpfit_result.success:
            mpfit_charges = mpfit_result.charges
            mpfit_multipoles = mpfit_result.multipole_moments

            print(f"  ✓ MPFIT successful")
            print(f"  Time: {mpfit_result.time_seconds:.2f} s")
            print("")

            # Analyze charges
            print("="*80)
            print("CHARGE COMPARISON")
            print("="*80)
            print(f"{'Atom':<6} {'Element':<8} {'MPFIT':<15} {'AM1-BCC':<15} {'Status'}")
            print("-"*80)

            for i, atom in enumerate(molecule.atoms):
                element = atom.symbol
                mpfit_q = mpfit_charges[i]
                am1bcc_q = am1bcc_charges[i] if am1bcc_charges is not None else np.nan

                # Check if MPFIT charge is reasonable
                if abs(mpfit_q) > 2.0:
                    status = "🔴 TOO LARGE"
                elif abs(mpfit_q) > 1.0:
                    status = "⚠️  LARGE"
                else:
                    status = "✓ OK"

                print(f"{i:<6} {element:<8} {mpfit_q:+12.4f} e  {am1bcc_q:+12.4f} e  {status}")

            print("-"*80)
            print(f"{'TOTAL':<15} {mpfit_charges.sum():+12.4f} e  {am1bcc_charges.sum() if am1bcc_charges is not None else np.nan:+12.4f} e")
            print("")

            # Statistics
            print("="*80)
            print("MPFIT CHARGE STATISTICS")
            print("="*80)
            print(f"  Range: [{mpfit_charges.min():.4f}, {mpfit_charges.max():.4f}] e")
            print(f"  Mean: {mpfit_charges.mean():.4f} e")
            print(f"  Std: {mpfit_charges.std():.4f} e")
            print(f"  Total: {mpfit_charges.sum():.4f} e (should be {molecule.total_charge.m})")
            print("")

            if am1bcc_charges is not None:
                print("AM1-BCC CHARGE STATISTICS")
                print("-"*80)
                print(f"  Range: [{am1bcc_charges.min():.4f}, {am1bcc_charges.max():.4f}] e")
                print(f"  Mean: {am1bcc_charges.mean():.4f} e")
                print(f"  Std: {am1bcc_charges.std():.4f} e")
                print(f"  Total: {am1bcc_charges.sum():.4f} e")
                print("")

                # Compare MPFIT to AM1-BCC
                charge_diff = mpfit_charges - am1bcc_charges
                print("MPFIT vs AM1-BCC")
                print("-"*80)
                print(f"  RMSE: {np.sqrt(np.mean(charge_diff**2)):.4f} e")
                print(f"  MAE: {np.mean(np.abs(charge_diff)):.4f} e")
                print(f"  Max diff: {np.abs(charge_diff).max():.4f} e")
                print("")

            # Inspect multipole moments
            if mpfit_multipoles is not None:
                print("="*80)
                print("MULTIPOLE MOMENTS (first 3 atoms)")
                print("="*80)
                print(f"Multipole shape: {mpfit_multipoles.shape}")
                print(f"  (n_atoms={mpfit_multipoles.shape[0]}, n_moments={mpfit_multipoles.shape[1]})")
                print("")

                for i in range(min(3, len(mpfit_multipoles))):
                    element = molecule.atoms[i].symbol
                    print(f"Atom {i} ({element}):")
                    print(f"  Charge (MPFIT): {mpfit_charges[i]:+.4f} e")
                    print(f"  Multipole moments: {mpfit_multipoles[i][:5]}...")  # First 5 moments
                    print("")

            # Final diagnosis
            print("="*80)
            print("DIAGNOSIS")
            print("="*80)

            if abs(mpfit_charges).max() > 2.0:
                print("🔴 ISSUE DETECTED: MPFIT charges are too large!")
                print("")
                print("Expected range for ethanol: ±1.0 e")
                print(f"Actual range: [{mpfit_charges.min():.4f}, {mpfit_charges.max():.4f}] e")
                print("")
                print("Possible causes:")
                print("  1. Unit conversion error in MPFITCalculator")
                print("  2. Multipole moments returned instead of charges")
                print("  3. OpenFF-recharge API changed")
                print("")
                print("RECOMMENDATION:")
                print("  Check MMomentA/qm/mpfit.py line 140:")
                print("    charges = np.array(charge_parameter.value)")
                print("  Verify charge_parameter.value units")

            elif abs(mpfit_charges).max() > 1.0:
                print("⚠️  CAUTION: MPFIT charges are larger than expected")
                print(f"Expected max: ~0.7 e for ethanol O")
                print(f"Actual max: {abs(mpfit_charges).max():.4f} e")
                print("")
                print("This may indicate a scaling issue or unusual conformation")

            else:
                print("✓ MPFIT charges appear reasonable!")
                print("")
                if am1bcc_charges is not None:
                    rmse = np.sqrt(np.mean((mpfit_charges - am1bcc_charges)**2))
                    if rmse < 0.2:
                        print(f"✓ Good agreement with AM1-BCC (RMSE: {rmse:.4f} e)")
                    else:
                        print(f"⚠️  Moderate disagreement with AM1-BCC (RMSE: {rmse:.4f} e)")
                        print("  (This is okay - MPFIT and AM1-BCC use different methods)")

        else:
            print(f"  ✗ MPFIT failed: {mpfit_result.error_message}")

    except Exception as e:
        print(f"  ✗ MPFIT error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
