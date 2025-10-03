# MMomentA - Phase 2: ML Training Data Preparation

## Overview

Phase 2 implements the complete ML training data pipeline:
- **Data schema** defining molecular features and targets
- **Feature extraction** from MPFIT calculations (multipole moments + atomic features)
- **Graph construction** compatible with DGL/espaloma-charge
- **Dataset splitting** (scaffold-based for transferability)
- **Efficient storage** (HDF5, Pickle, JSON)
- **PyTorch data loaders** ready for model training

## Data Schema

The `MoleculeData` schema matches the project specification:

```python
{
    'molecule_id': str,                  # Unique identifier
    'smiles': str,                       # SMILES string
    'conformer': np.ndarray,             # (n_atoms, 3) coordinates
    'multipole_moments': np.ndarray,     # (n_atoms, n_moments) GDMA output
    'atomic_features': dict,             # Atomic descriptors for GNN
    'graph_structure': dict,             # Edges, bond orders
    'target_charges': np.ndarray,        # (n_atoms,) MPFIT charges
    'qm_metadata': dict,                 # QM calculation settings
}
```

### Atomic Features

Features compatible with espaloma-charge (17 dimensions):
- Total degree, valence, explicit valence
- Is aromatic, atomic mass
- Ring membership (3-8 membered rings)
- Hybridization (one-hot: SP, SP2, SP3, SP3D, SP3D2)

### Graph Features

For GNN training with DGL:
- One-hot element encoding (100 dimensions)
- Atomic fingerprint (17 dimensions)
- Multipole moments (81 dimensions for limit=8)
- **Total: 198 dimensions** (vs 117 for baseline without multipoles)

## Quick Start

### 1. Prepare Dataset from Molecules

```python
from MMomentA.qm import MPFITCalculator
from MMomentA.data import (
    extract_molecule_data,
    scaffold_split,
    save_dataset,
    DatasetMetadata
)

# Compute MPFIT charges
calculator = MPFITCalculator()
molecule_data_list = []

for molecule in molecules:
    result = calculator.compute(molecule)
    if result.success:
        mol_data = extract_molecule_data(molecule, result)
        molecule_data_list.append(mol_data)

# Split dataset
train_idx, val_idx, test_idx = scaffold_split(molecules)

# Save for training
metadata = DatasetMetadata(
    dataset_name="SPICE",
    n_molecules=len(molecule_data_list),
    qm_method="hf",
    qm_basis="6-31G*",
    multipole_limit=8,
    created_date=datetime.now().isoformat()
)

save_dataset(molecule_data_list, "spice_mpfit.h5", metadata)
```

### 2. Load Dataset and Create Data Loaders

```python
from MMomentA.data import load_dataset, create_split_datasets

# Load dataset
molecule_data_list, metadata = load_dataset("spice_mpfit.h5")

# Get split indices from metadata
train_idx = [i for i, mol in enumerate(molecule_data_list)
             if mol.molecule_id in metadata['splits']['train']]
val_idx = [i for i, mol in enumerate(molecule_data_list)
           if mol.molecule_id in metadata['splits']['val']]
test_idx = [i for i, mol in enumerate(molecule_data_list)
            if mol.molecule_id in metadata['splits']['test']]

# Create datasets with DGL graphs
split_data = create_split_datasets(
    molecule_data_list,
    train_idx, val_idx, test_idx,
    include_multipoles=True  # Include multipole features
)

# Get data loaders
train_loader = split_data.get_train_loader(batch_size=32)
val_loader = split_data.get_val_loader(batch_size=32)
test_loader = split_data.get_test_loader(batch_size=32)
```

### 3. Command-Line Dataset Preparation

```bash
# Prepare SPICE dataset
python scripts/prepare_dataset.py \
    --input spice.oeb \
    --output spice_mpfit.h5 \
    --dataset-name SPICE \
    --max-molecules 1000 \
    --method hf \
    --basis "6-31G*" \
    --split-strategy scaffold \
    --n-jobs 8

# Prepare ZINC dataset
python scripts/prepare_dataset.py \
    --input zinc_molecules.pkl \
    --output zinc_mpfit.h5 \
    --dataset-name ZINC \
    --max-molecules 500 \
    --method pbe0 \
    --basis def2-SVP \
    --n-jobs 8
```

## Data Pipeline Workflow

```
Input Molecules (SDF/OEB/PKL)
    ↓
[Phase 1] MPFIT Calculation
    ↓
MPFITResult (charges + multipoles + conformer)
    ↓
[Phase 2] Feature Extraction
    ↓
MoleculeData (complete features + graph structure)
    ↓
Dataset Splitting (scaffold/random/stratified)
    ↓
Save to Storage (HDF5/Pickle/JSON)
    ↓
Load & Convert to DGL Graphs
    ↓
PyTorch DataLoaders → Ready for Training
```

## Module Overview

### `mmomenta/data/schema.py`

Defines data structures:
- `MoleculeData`: Complete molecular data for ML
- `DatasetMetadata`: Dataset information and splits

### `mmomenta/data/extraction.py`

Extract features from QM calculations:
- `extract_molecule_data()`: Single molecule extraction
- `batch_extract_molecule_data()`: Batch extraction
- `extract_atomic_features()`: Atomic descriptors
- `extract_graph_structure()`: Graph connectivity

### `mmomenta/data/graph.py`

DGL graph construction:
- `molecule_data_to_dgl_graph()`: Convert to DGL format
- `batch_to_dgl_graphs()`: Batch conversion
- `get_feature_dimensions()`: Calculate feature size

### `mmomenta/data/splitting.py`

Dataset splitting strategies:
- `scaffold_split()`: Bemis-Murcko scaffold-based
- `random_split()`: Random splitting
- `stratified_split()`: Stratify by property
- `split_by_names()`: Split by molecule names

### `mmomenta/data/dataset.py`

PyTorch dataset classes:
- `ChargeDataset`: Main dataset class
- `SplitDataset`: Train/val/test container
- `create_split_datasets()`: Create from indices

### `mmomenta/data/storage.py`

Data persistence:
- `save_dataset()` / `load_dataset()`: Auto-format detection
- HDF5 support: Efficient compression
- Pickle support: Python convenience
- JSON support: Human-readable

## Feature Engineering

### Multipole Moments

GDMA multipole moments by rank:
- **Rank 0** (monopole): 1 component
- **Rank 1** (dipole): 3 components
- **Rank 2** (quadrupole): 5 components
- **Rank 3** (octupole): 7 components
- ...
- **Rank 8** (default): 81 total components

### Graph Structure

Bidirectional edges for message passing:
```python
# For each bond i-j, create two edges:
edges = [[i, j], [j, i]]  # Undirected via two directed
```

### Dataset Splitting Strategies

**Scaffold Split (Recommended)**
- Groups molecules by Bemis-Murcko scaffold
- Tests transferability to new scaffolds
- Best for evaluating generalization

**Random Split**
- Shuffles all molecules randomly
- Good for diverse datasets
- Easier baseline

**Stratified Split**
- Balances property distribution
- Useful for size/property-dependent learning

## Storage Formats

### HDF5 (Recommended for Large Datasets)

```python
save_dataset(mol_data_list, "dataset.h5", metadata)
# Features:
# - Compression (gzip)
# - Fast random access
# - Large dataset support
```

### Pickle (Convenient for Small Datasets)

```python
save_dataset(mol_data_list, "dataset.pkl", metadata)
# Features:
# - Python native
# - Fastest to load
# - Not cross-language
```

### JSON (Human-Readable)

```python
save_dataset(mol_data_list, "dataset.json", metadata)
# Features:
# - Human-readable
# - Cross-platform
# - Large file size
```

## Integration with espaloma-charge

The data pipeline is designed for seamless integration:

```python
# Load MMomentA dataset
from MMomentA.data import load_dataset, create_split_datasets

mol_data_list, metadata = load_dataset("spice_mpfit.h5")
split_data = create_split_datasets(
    mol_data_list, train_idx, val_idx, test_idx,
    include_multipoles=True
)

# Compatible with espaloma-charge model
from espaloma_charge.models import Sequential, ChargeReadout, ChargeEquilibrium

config = [128, "relu"] * 4  # 4 layers, 128 units
model = torch.nn.Sequential(
    Sequential(
        layer=partial(dgl.nn.SAGEConv, aggregator_type="mean"),
        config=config,
        feature_units=198  # 117 baseline + 81 multipoles
    ),
    ChargeReadout(128),
    ChargeEquilibrium()
)

# Training loop
train_loader = split_data.get_train_loader(batch_size=32)
for batch_graph in train_loader:
    batch_graph = model(batch_graph)
    loss = torch.nn.MSELoss()(
        batch_graph.ndata["q_ref"],
        batch_graph.ndata["q"]
    )
    # ... backprop
```

## Key Design Decisions

### 1. Feature Compatibility

- Match espaloma-charge feature set (117 dimensions baseline)
- Add multipole moments as additional features
- Allow easy baseline comparison (with/without multipoles)

### 2. Scaffold Splitting

- Ensures molecules in test set have different scaffolds than train
- Tests true transferability to new chemistry
- Recommended in literature for molecular ML

### 3. Flexible Storage

- Support multiple formats (HDF5/Pickle/JSON)
- Auto-detection from file extension
- Metadata stored with data for reproducibility

### 4. Validation

- `MoleculeData.validate()` ensures consistency
- Charge conservation checks
- Shape validation for all arrays

## Next Steps (Phase 3)

Phase 3 will implement ML model training:

1. **Model architecture**: Adapt espaloma-charge for multipoles
2. **Training pipeline**: Complete training loops with validation
3. **Ablation studies**: Baseline vs. multipole-enhanced models
4. **Evaluation metrics**: Charge RMSE, ESP reproduction, transferability
5. **Model saving/loading**: Checkpointing and inference

## Examples

See `examples/dataset_preparation_example.py` for complete workflow.

## References

- espaloma-charge: https://github.com/choderalab/espaloma_charge
- DGL: https://www.dgl.ai/
- GDMA: https://github.com/openforcefield/openff-recharge
