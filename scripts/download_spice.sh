#!/bin/bash
# Download SPICE 2.0.1 dataset from Zenodo

DATASET_DIR="/storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA/data"

echo "================================================"
echo "Downloading SPICE 2.0.1 Dataset"
echo "================================================"
echo ""
echo "Dataset details:"
echo "  - 114,305 molecules"
echo "  - 2,016,401 conformers"
echo "  - QM method: ωB97M-D3(BJ)/def2-TZVPPD"
echo "  - Size: ~8.5 GB"
echo "  - Source: https://zenodo.org/records/10975225"
echo ""

# Create data directory
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

# Download SPICE
echo "Downloading SPICE-2.0.1.hdf5 (this will take 10-30 minutes)..."
wget https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Download complete!"
    echo ""
    echo "File info:"
    ls -lh SPICE-2.0.1.hdf5
    echo ""
    echo "Next steps:"
    echo "1. Extract subset for MVP (optional):"
    echo "   python scripts/extract_spice_subset.py --input data/SPICE-2.0.1.hdf5 --output data/spice_100.pkl --n-molecules 100"
    echo ""
    echo "2. Or use full dataset directly:"
    echo "   python scripts/prepare_dataset.py --input data/SPICE-2.0.1.hdf5 --output data/spice_mpfit.h5 --n-jobs 16"
else
    echo "✗ Download failed"
    exit 1
fi
