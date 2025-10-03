#!/bin/bash
# Test MMomentA installation

set -e

echo "================================================"
echo "Testing MMomentA Installation"
echo "================================================"

# Test imports
echo ""
echo "Testing Python imports..."

python -c "
import sys
import numpy as np
import torch
import dgl
from openff.toolkit import Molecule
from openff.recharge.charges.mpfit import generate_mpfit_charge_parameter
import psi4

print('✓ NumPy version:', np.__version__)
print('✓ PyTorch version:', torch.__version__)
print('✓ DGL version:', dgl.__version__)
print('✓ OpenFF Toolkit: OK')
print('✓ OpenFF Recharge: OK')
print('✓ Psi4 version:', psi4.__version__)
"

# Test MMomentA modules
echo ""
echo "Testing MMomentA modules..."

python -c "
from MMomentA.qm import MPFITCalculator, RESPCalculator, AM1BCCCalculator
from MMomentA.data import load_molecules_from_file, BatchProcessor
from MMomentA.models import ChargeModel, ModelConfig
from MMomentA.training import Trainer, TrainingConfig

print('✓ MMomentA.qm')
print('✓ MMomentA.data')
print('✓ MMomentA.models')
print('✓ MMomentA.training')
"

# Test simple calculation
echo ""
echo "Testing simple MPFIT calculation..."

python << 'EOF'
from openff.toolkit.topology import Molecule
from MMomentA.qm import MPFITCalculator, GDMAConfig

# Simple molecule
molecule = Molecule.from_smiles("C")
molecule.generate_conformers(n_conformers=1)

# MPFIT calculation
config = GDMAConfig(method="hf", basis="sto-3g", limit=2)
calculator = MPFITCalculator(config)

print("  Computing MPFIT charges for methane (CH4)...")
result = calculator.compute(molecule)

if result.success:
    print(f"  ✓ Charges computed: {result.charges}")
    print(f"  ✓ Computation time: {result.time_seconds:.2f}s")
    print(f"  ✓ Multipole moments shape: {result.multipole_moments.shape}")
else:
    print(f"  ✗ Calculation failed: {result.error_message}")
    sys.exit(1)
EOF

# Test GPU (if available)
echo ""
echo "Testing GPU support..."

python -c "
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(10, 10).to(device)
    print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠ No GPU available (CPU only)')
"

echo ""
echo "================================================"
echo "All Tests Passed!"
echo "================================================"
echo ""
echo "MMomentA is ready to use on Phoenix cluster."
echo ""
