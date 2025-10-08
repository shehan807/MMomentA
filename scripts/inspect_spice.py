#!/usr/bin/env python
"""
Inspect SPICE HDF5 file structure to understand the data format.
"""

import h5py
import sys

def inspect_spice(filepath):
    """Inspect SPICE HDF5 file structure"""

    print(f"Inspecting: {filepath}")
    print("=" * 60)

    with h5py.File(filepath, 'r') as f:
        # Get top-level keys
        mol_ids = list(f.keys())
        print(f"\nTotal molecules: {len(mol_ids)}")
        print(f"First 5 molecule IDs: {mol_ids[:5]}")

        # Inspect first molecule in detail
        if len(mol_ids) > 0:
            mol_id = mol_ids[0]
            print(f"\n{'='*60}")
            print(f"Detailed structure of first molecule: {mol_id}")
            print(f"{'='*60}")

            mol_grp = f[mol_id]

            # Check attributes
            print(f"\nAttributes:")
            for attr_name in mol_grp.attrs:
                attr_value = mol_grp.attrs[attr_name]
                print(f"  {attr_name}: {attr_value} (type: {type(attr_value)})")

            # Check datasets
            print(f"\nDatasets:")
            for key in mol_grp.keys():
                dataset = mol_grp[key]
                print(f"  {key}:")
                print(f"    - Shape: {dataset.shape}")
                print(f"    - Dtype: {dataset.dtype}")
                if dataset.shape == ():
                    print(f"    - Value: {dataset[()]}")
                elif len(dataset.shape) == 1 and dataset.shape[0] <= 5:
                    print(f"    - Values: {dataset[:]}")
                elif len(dataset.shape) >= 2:
                    print(f"    - First row: {dataset[0] if dataset.shape[0] > 0 else 'empty'}")

            # Try to extract SMILES
            print(f"\n{'='*60}")
            print("Testing SMILES extraction:")
            print(f"{'='*60}")

            if 'smiles' in mol_grp.attrs:
                smiles = mol_grp.attrs['smiles']
                print(f"✓ SMILES found in attributes: {smiles}")
            elif 'smiles' in mol_grp:
                smiles = mol_grp['smiles'][()]
                if isinstance(smiles, bytes):
                    smiles = smiles.decode('utf-8')
                print(f"✓ SMILES found as dataset: {smiles}")
            else:
                print("✗ No SMILES field found")
                print(f"Available attrs: {list(mol_grp.attrs.keys())}")
                print(f"Available datasets: {list(mol_grp.keys())}")

            # Try to extract conformations
            print(f"\nTesting conformation extraction:")
            if 'conformations' in mol_grp:
                conformations = mol_grp['conformations']
                print(f"✓ Conformations found: shape {conformations.shape}, dtype {conformations.dtype}")
                if len(conformations.shape) >= 2:
                    print(f"  Number of conformations: {conformations.shape[0]}")
                    print(f"  Number of atoms: {conformations.shape[1]}")
                    print(f"  First atom coordinates: {conformations[0][0] if conformations.shape[0] > 0 else 'empty'}")
            elif 'coordinates' in mol_grp:
                coordinates = mol_grp['coordinates']
                print(f"✓ Coordinates found: shape {coordinates.shape}, dtype {coordinates.dtype}")
            else:
                print("✗ No conformations/coordinates field found")
                print(f"Available datasets: {list(mol_grp.keys())}")

        # Test a few more molecules
        print(f"\n{'='*60}")
        print("Testing first 5 molecules:")
        print(f"{'='*60}")

        for i, mol_id in enumerate(mol_ids[:5]):
            mol_grp = f[mol_id]

            # Try to get SMILES
            smiles = None
            if 'smiles' in mol_grp.attrs:
                smiles = mol_grp.attrs['smiles']
            elif 'smiles' in mol_grp:
                smiles = mol_grp['smiles'][()]
                if isinstance(smiles, bytes):
                    smiles = smiles.decode('utf-8')

            # Try to get conformations
            has_conformations = 'conformations' in mol_grp or 'coordinates' in mol_grp

            print(f"{i+1}. {mol_id}")
            print(f"   SMILES: {smiles if smiles else 'NOT FOUND'}")
            print(f"   Conformations: {'YES' if has_conformations else 'NO'}")
            print(f"   Keys: {list(mol_grp.keys())[:5]}")

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/SPICE-2.0.1.hdf5"
    inspect_spice(filepath)
