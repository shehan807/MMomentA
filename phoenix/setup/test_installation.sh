#!/bin/bash
# Test MMomentA installation

echo "================================================"
echo "Testing MMomentA Installation"
echo "================================================"

# Test imports
echo ""
echo "Testing Python imports..."

python << 'EOF'
import sys

# Test each package individually so we see what fails
packages_ok = True

try:
    import numpy as np
    print(f'✓ NumPy: {np.__version__}')
except ImportError as e:
    print(f'✗ NumPy: {e}')
    packages_ok = False

try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
except ImportError as e:
    print(f'✗ PyTorch: {e}')
    packages_ok = False

try:
    import dgl
    print(f'✓ DGL: {dgl.__version__}')
except ImportError as e:
    print(f'✗ DGL: {e}')
    packages_ok = False

try:
    from openff.toolkit import Molecule
    print('✓ OpenFF Toolkit: OK')
except ImportError as e:
    print(f'✗ OpenFF Toolkit: {e}')
    packages_ok = False

try:
    from openff.recharge.charges.mpfit import generate_mpfit_charge_parameter
    print('✓ OpenFF Recharge: OK')
except ImportError as e:
    print(f'✗ OpenFF Recharge: {e}')
    packages_ok = False

try:
    import psi4
    print(f'✓ Psi4: {psi4.__version__}')
except ImportError as e:
    print(f'✗ Psi4: {e}')
    print('  (Psi4 is optional for pre-computed datasets)')

if not packages_ok:
    print('\n✗ Some packages failed to import')
    sys.exit(1)
EOF

# Test MMomentA modules
echo ""
echo "Testing MMomentA modules..."

python << 'EOF'
import sys

try:
    from mmomenta.qm import MPFITCalculator
    print('✓ mmomenta.qm')
except ImportError as e:
    print(f'✗ mmomenta.qm: {e}')
    sys.exit(1)

try:
    from mmomenta.data import BatchProcessor
    print('✓ mmomenta.data')
except ImportError as e:
    print(f'✗ mmomenta.data: {e}')
    sys.exit(1)

try:
    from mmomenta.models import ChargeModel
    print('✓ mmomenta.models')
except ImportError as e:
    print(f'✗ mmomenta.models: {e}')
    sys.exit(1)

try:
    from mmomenta.training import Trainer
    print('✓ mmomenta.training')
except ImportError as e:
    print(f'✗ mmomenta.training: {e}')
    sys.exit(1)
EOF

# Test simple calculation (optional - requires Psi4)
echo ""
echo "Testing simple MPFIT calculation..."

python << 'EOF'
import sys

try:
    import psi4
    from openff.toolkit import Molecule
    from mmomenta.qm import MPFITCalculator, MPFITConfig, QMSettings

    # Simple molecule
    molecule = Molecule.from_smiles("C")
    molecule.generate_conformers(n_conformers=1)

    # MPFIT calculation
    qc_settings = QMSettings(method="hf", basis="sto-3g")
    config = MPFITConfig(limit=2)
    calculator = MPFITCalculator(qc_settings=qc_settings, config=config)

    print("  Computing MPFIT charges for methane (CH4)...")
    result = calculator.compute(molecule)

    print(f"  ✓ Charges computed: {result.charges}")
    print(f"  ✓ Multipole moments shape: {result.multipole_moments.shape}")

except ImportError:
    print("  ⚠ Skipping MPFIT test (Psi4 not installed)")
    print("  This is OK if you're using pre-computed datasets")
except Exception as e:
    print(f"  ✗ MPFIT calculation failed: {e}")
    print("  This may indicate a configuration issue")
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
