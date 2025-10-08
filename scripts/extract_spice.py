#!/usr/bin/env python
"""
Extract molecules from SPICE HDF5 dataset.
"""

import argparse
import h5py
import pickle
import numpy as np
from pathlib import Path
from openff.toolkit import Molecule
from openff.units import unit


def extract_spice_molecules(hdf5_path, output_path, max_molecules=100, verbose=True):
    """Extract molecules from SPICE HDF5 file.

    Parameters
    ----------
    hdf5_path : str
        Path to SPICE HDF5 file
    output_path : str
        Path to save pickle file
    max_molecules : int
        Maximum number of molecules to extract
    verbose : bool
        Print progress

    Returns
    -------
    list
        List of OpenFF Molecule objects
    """

    molecules = []

    if verbose:
        print(f"Opening SPICE dataset: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        total_molecules = len(f.keys())
        mol_ids = list(f.keys())[:max_molecules]

        if verbose:
            print(f"Found {total_molecules} total molecules")
            print(f"Extracting first {len(mol_ids)} molecules...")

        for i, mol_id in enumerate(mol_ids):
            try:
                grp = f[mol_id]

                # SPICE stores SMILES as dataset, not attribute
                smiles = None

                # Try dataset first
                if 'smiles' in grp:
                    smiles_data = grp['smiles'][()]
                    if isinstance(smiles_data, bytes):
                        smiles = smiles_data.decode('utf-8')
                    else:
                        smiles = str(smiles_data)
                # Fall back to attribute
                elif 'smiles' in grp.attrs:
                    smiles = grp.attrs['smiles']
                    if isinstance(smiles, bytes):
                        smiles = smiles.decode('utf-8')

                if not smiles:
                    if verbose:
                        print(f"  ✗ No SMILES for {mol_id}")
                    continue

                # Create molecule from SMILES
                mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

                # Get conformations
                coords = None
                if 'conformations' in grp:
                    coords = grp['conformations'][0]  # Take first conformer
                elif 'geometry' in grp:
                    coords = grp['geometry'][0]
                elif 'coordinates' in grp:
                    coords = grp['coordinates'][0]

                if coords is None:
                    if verbose:
                        print(f"  ✗ No coordinates for {mol_id}")
                    continue

                # Convert to numpy array and ensure correct shape
                coords = np.array(coords, dtype=float)

                # Check if coordinates match number of atoms
                if coords.shape[0] != mol.n_atoms:
                    if verbose:
                        print(f"  ✗ Coordinate mismatch for {mol_id}: "
                              f"{coords.shape[0]} coords vs {mol.n_atoms} atoms")
                    continue

                # Add conformer (OpenFF expects Quantity with units)
                # SPICE coordinates should be in Angstroms
                mol.add_conformer(coords * unit.angstrom)

                # Set molecule name
                mol.name = mol_id

                molecules.append(mol)

                if verbose and (i + 1) % 20 == 0:
                    print(f"  Extracted {i + 1}/{len(mol_ids)} molecules "
                          f"({len(molecules)} successful)")

            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed on {mol_id}: {e}")
                continue

    # Save molecules
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(molecules, f)

    if verbose:
        print(f"\n✓ Saved {len(molecules)} molecules to {output_path}")

    return molecules


def main():
    parser = argparse.ArgumentParser(
        description="Extract molecules from SPICE HDF5 dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/SPICE-2.0.1.hdf5",
        help="Path to SPICE HDF5 file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/spice_molecules.pkl",
        help="Path to save pickle file"
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=100,
        help="Maximum number of molecules to extract"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect HDF5 structure before extraction"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Optional: inspect structure first
    if args.inspect:
        print("Inspecting SPICE HDF5 structure...")
        print("=" * 60)

        with h5py.File(args.input, 'r') as f:
            mol_ids = list(f.keys())[:5]
            print(f"Total molecules: {len(f.keys())}")
            print(f"\nFirst 5 molecule IDs: {mol_ids}")

            for mol_id in mol_ids:
                grp = f[mol_id]
                print(f"\n{mol_id}:")
                print(f"  Attributes: {list(grp.attrs.keys())}")
                print(f"  Datasets: {list(grp.keys())}")

                # Check SMILES
                if 'smiles' in grp:
                    smiles = grp['smiles'][()]
                    if isinstance(smiles, bytes):
                        smiles = smiles.decode('utf-8')
                    print(f"  SMILES: {smiles}")

                # Check coordinates
                for coord_key in ['conformations', 'geometry', 'coordinates']:
                    if coord_key in grp:
                        coords = grp[coord_key]
                        print(f"  {coord_key}: shape {coords.shape}")

        print("=" * 60)
        print("\nProceeding with extraction...\n")

    # Extract molecules
    molecules = extract_spice_molecules(
        args.input,
        args.output,
        max_molecules=args.max_molecules,
        verbose=not args.quiet
    )

    print(f"\nExtraction complete: {len(molecules)} molecules")


if __name__ == "__main__":
    main()
