# Downloading SPICE and ZINC Datasets

## SPICE Dataset

### Option 1: Direct Download from Zenodo (Recommended)

```bash
cd MMomentA/data

# Full SPICE 2.0.1 (~30 GB, 2M conformers, 114K molecules)
wget https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5

# Rename for consistency
mv SPICE-2.0.1.hdf5 spice_full.hdf5
```

**Time:** 10-30 minutes depending on network

### Option 2: From GitHub Repository

```bash
# Clone the repository
git clone https://github.com/openmm/spice-dataset.git
cd spice-dataset

# Download links are in releases
# https://github.com/openmm/spice-dataset/releases
```

### Option 3: Extract Subset for MVP (Recommended for Testing)

**After downloading full dataset, extract 50-100 molecules:**

```bash
cd MMomentA

python << 'EOF'
import h5py
import numpy as np
from pathlib import Path

print("Extracting 50-molecule subset from SPICE...")

input_file = "data/spice_full.hdf5"
output_file = "data/spice_subset_50.hdf5"

# Open full SPICE dataset
with h5py.File(input_file, 'r') as f_in:
    # Get molecule names
    mol_names = list(f_in.keys())[:50]

    # Create subset file
    with h5py.File(output_file, 'w') as f_out:
        for mol_name in mol_names:
            f_in.copy(mol_name, f_out)

print(f"✓ Created subset: {output_file}")
print(f"  Contains {len(mol_names)} molecules")
EOF
```

### Option 4: Create Test Dataset from SMILES (No Download, Instant)

**For immediate testing without downloading SPICE:**

```bash
cd MMomentA

python << 'EOF'
from openff.toolkit import Molecule
import pickle

# 50 diverse drug-like molecules
smiles_list = [
    'CC(=O)O',           # Acetic acid
    'c1ccccc1',          # Benzene
    'CCO',               # Ethanol
    'C1CCCCC1',          # Cyclohexane
    'CC(C)O',            # Isopropanol
    'CCCC',              # Butane
    'c1cc(O)ccc1',       # Phenol
    'CC(N)C',            # Isopropylamine
    'c1cnccc1',          # Pyridine
    'CCCCCC',            # Hexane
    'CC(=O)C',           # Acetone
    'c1ccc(N)cc1',       # Aniline
    'CCCCO',             # 1-Butanol
    'c1ccc(C)cc1',       # Toluene
    'CC(C)(C)C',         # Neopentane
    'c1ccc(F)cc1',       # Fluorobenzene
    'CCOC',              # Methyl ethyl ether
    'c1ccc(Cl)cc1',      # Chlorobenzene
    'CC(=O)N',           # Acetamide
    'c1cccnc1',          # Pyridine
    'CCC(C)C',           # Isopentane
    'c1ccc(O)c(O)c1',    # Catechol
    'CCCCN',             # Butylamine
    'c1ccc2ccccc2c1',    # Naphthalene
    'CC(C)CC',           # 2-Methylbutane
    'c1ccc(N(C)C)cc1',   # N,N-Dimethylaniline
    'CCCCC',             # Pentane
    'c1cnc(C)nc1',       # 4-Methylpyrimidine
    'CC(C)CN',           # Isobutylamine
    'c1csc(C)c1',        # 2-Methylthiophene
    'CCCCCC',            # Hexane
    'c1ccc(C(C)C)cc1',   # Cumene
    'CC(O)C',            # Isopropanol
    'c1ccc(OC)cc1',      # Anisole
    'CCCCCN',            # Pentylamine
    'c1ccc(C(F)(F)F)cc1',# Trifluorotoluene
    'CC(C)CO',           # Isobutanol
    'c1cnc(N)nc1',       # 2-Aminopyrimidine
    'CCCC(C)C',          # 2-Methylpentane
    'c1ccc(S(=O)(=O)N)cc1', # Benzenesulfonamide
    'CC(C)(C)O',         # tert-Butanol
    'c1ccc(C#N)cc1',     # Benzonitrile
    'CCCCCO',            # 1-Pentanol
    'c1ccc(C(=O)C)cc1',  # Acetophenone
    'CC(C)CCC',          # 2-Methylpentane
    'c1ccc(C(=O)O)cc1',  # Benzoic acid
    'CCCCCCO',           # 1-Hexanol
    'c1ccc(C(=O)N)cc1',  # Benzamide
    'CC(C)CCCC',         # 2-Methylhexane
    'c1ccc2c(c1)cccc2',  # Naphthalene
]

print(f"Creating dataset from {len(smiles_list)} SMILES...")

molecules = []
for i, smiles in enumerate(smiles_list):
    try:
        mol = Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)
        molecules.append(mol)
    except Exception as e:
        print(f"Failed on molecule {i}: {e}")

print(f"Successfully created {len(molecules)} molecules")

# Save as pickle
with open('data/test_molecules_50.pkl', 'wb') as f:
    pickle.dump(molecules, f)

print(f"✓ Saved to data/test_molecules_50.pkl")
print(f"  Use this for MVP: --input data/test_molecules_50.pkl")
EOF
```

**This creates 50 test molecules instantly without any download.**

## ZINC Dataset

### Option 1: Download Drug-Like Subset from ZINC15

```bash
cd MMomentA/data

# Method 1: Web interface
# Visit https://zinc.docking.org/substances/subsets/
# Select "drug-like" preset
# Download tranches

# Method 2: Direct wget (example - update URL)
# Download specific tranches via ZINC website download script
```

### Option 2: Use Pre-Processed QM Dataset (Easier)

**GEOM dataset has ZINC molecules with conformers:**

```bash
# Download GEOM dataset (has QM-quality ZINC conformers)
# From Harvard Dataverse
# https://doi.org/10.7910/DVN/JNGTDF

# This is better for ML because conformers are already generated
wget https://dataverse.harvard.edu/api/access/datafile/4327252 -O geom_drugs.tar.gz
tar -xzf geom_drugs.tar.gz
```

### Option 3: Sample from ZINC via API (Small Subsets)

```bash
python << 'EOF'
import requests
import pickle
from openff.toolkit import Molecule

# Example: Download specific ZINC IDs
zinc_ids = ['ZINC000000001', 'ZINC000000002']  # etc.

molecules = []
for zinc_id in zinc_ids:
    # Download SMILES from ZINC
    url = f"https://zinc.docking.org/substances/{zinc_id}.smi"
    response = requests.get(url)
    if response.ok:
        smiles = response.text.split()[0]
        mol = Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)
        molecules.append(mol)

with open('data/zinc_subset.pkl', 'wb') as f:
    pickle.dump(molecules, f)
EOF
```

### Option 4: Use Test Molecules (Like SPICE)

**For MVP, you can just use the same test molecules:**

```bash
# Same test molecules work for transfer learning demo
cp data/test_molecules_50.pkl data/zinc_test.pkl
```

## Quick Summary for MVP

**Fastest path (no downloads):**
```bash
# 1. Create test molecules from SMILES (instant)
python [script from Option 4 above]

# 2. Use for both SPICE and ZINC
ln -s test_molecules_50.pkl spice_raw.hdf5
ln -s test_molecules_50.pkl zinc_raw.pkl
```

**Production path (download real data):**
```bash
# SPICE
wget https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5

# ZINC - use GEOM (easier than raw ZINC)
wget https://dataverse.harvard.edu/api/access/datafile/4327252 -O geom_drugs.tar.gz
```

## What the MVP Script Expects

**The `submit_mvp_2hr.sh` script expects:**

```bash
data/spice_raw.hdf5  # Any format: HDF5, pkl, oeb, sdf
```

**Supported formats:**
- `.hdf5` - HDF5 from SPICE/GEOM
- `.pkl` - Pickle of OpenFF Molecule list
- `.oeb` - OpenEye binary
- `.sdf` - MDL structure file
- `.mol2` - Tripos mol2

**The `prepare_dataset.py` script auto-detects format from extension.**

## Testing Your Download

```bash
# For HDF5 (SPICE)
python -c "
import h5py
with h5py.File('data/spice_full.hdf5', 'r') as f:
    print(f'Molecules: {len(f.keys())}')
    print(f'First 5: {list(f.keys())[:5]}')
"

# For pickle (test molecules)
python -c "
import pickle
with open('data/test_molecules_50.pkl', 'rb') as f:
    mols = pickle.load(f)
    print(f'Molecules: {len(mols)}')
    print(f'First molecule: {mols[0].to_smiles()}')
"
```

## Recommendation for Your 2-Hour MVP

**Use the test molecules approach (Option 4):**

1. **Instant** - no download wait
2. **Small** - 50 molecules perfect for 2-hour timeline
3. **Valid** - still demonstrates the concept
4. **Works** - compatible with all scripts

```bash
cd MMomentA

# Create test dataset (30 seconds)
python [test molecules script from above]

# Verify
ls -lh data/test_molecules_50.pkl

# Update MVP script to use it
# Edit phoenix/submit_mvp_2hr.sh line 66:
# --input data/test_molecules_50.pkl
```

**Later**, when you have time, download real SPICE for full study.

## File Sizes Reference

| Dataset | Size | Molecules | Download Time |
|---------|------|-----------|---------------|
| Test SMILES | <1 MB | 50 | Instant |
| SPICE subset | ~1 GB | 5K | 2-5 min |
| SPICE full | ~30 GB | 114K | 10-30 min |
| GEOM drugs | ~50 GB | 450K | 20-60 min |
| ZINC15 lead-like | ~100 GB | 500M | Hours |

**For 2-hour MVP:** Use test SMILES (instant)
**For publication:** Download SPICE full (30 min)
**For transfer learning:** Download GEOM or ZINC subset (30-60 min)
