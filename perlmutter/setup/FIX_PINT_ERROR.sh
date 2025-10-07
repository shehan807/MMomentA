#!/bin/bash
# Fix pint version incompatibility in ML environment
# The error: "TypeError: cannot inherit frozen dataclass from a non-frozen one"
# is caused by pint 0.24+ being incompatible with openff-units

echo "================================================"
echo "Fixing pint version incompatibility"
echo "================================================"

module load conda

ML_ENV="$SCRATCH/conda-envs/mmomenta-ml"

echo "Activating ML environment: $ML_ENV"
conda activate "$ML_ENV"

echo ""
echo "Current pint version:"
python -c "import pint; print(pint.__version__)"

echo ""
echo "Downgrading pint to version compatible with openff-units..."
pip install "pint<0.24"

echo ""
echo "New pint version:"
python -c "import pint; print(pint.__version__)"

echo ""
echo "Testing import..."
python -c "from openff.toolkit import Molecule; print('✓ OpenFF toolkit imports successfully')"

echo ""
echo "================================================"
echo "Fix applied successfully!"
echo "================================================"
echo ""
echo "You can now run the MVP pipeline:"
echo "  bash perlmutter/pipelines/run_mvp.sh"
