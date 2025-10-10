#!/bin/bash
# Download SPICE 2.0.1 dataset to $SCRATCH on Perlmutter

echo "================================================"
echo "Downloading SPICE 2.0.1 Dataset to \$SCRATCH"
echo "================================================"
echo ""
echo "Dataset details:"
echo "  - 114,305 molecules"
echo "  - 2,016,401 conformers"
echo "  - QM method: ωB97M-D3(BJ)/def2-TZVPPD"
echo "  - Size: ~8.5 GB"
echo "  - Source: https://zenodo.org/records/10975225"
echo ""

# Check if SCRATCH is set
if [ -z "$SCRATCH" ]; then
    echo "ERROR: \$SCRATCH is not set"
    echo "Are you on a Perlmutter login/compute node?"
    exit 1
fi

# Check if already exists
if [ -f "$SCRATCH/SPICE-2.0.1.hdf5" ]; then
    echo "✓ SPICE dataset already exists at $SCRATCH/SPICE-2.0.1.hdf5"
    ls -lh "$SCRATCH/SPICE-2.0.1.hdf5"
    echo ""
    echo "To re-download, delete the file first:"
    echo "  rm $SCRATCH/SPICE-2.0.1.hdf5"
    exit 0
fi

# Download to SCRATCH
echo "Downloading to: $SCRATCH/SPICE-2.0.1.hdf5"
echo "This will take 10-30 minutes depending on network speed..."
echo ""

cd "$SCRATCH"
wget https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✓ Download complete!"
    echo "================================================"
    echo ""
    echo "File info:"
    ls -lh "$SCRATCH/SPICE-2.0.1.hdf5"
    echo ""
    echo "Location: $SCRATCH/SPICE-2.0.1.hdf5"
    echo ""
    echo "Next steps:"
    echo "  - Run SPICE pipeline: bash perlmutter/pipelines/run_spice.sh"
    echo "  - Or submit SLURM job: sbatch perlmutter/pipelines/run_spice_10k.slurm"
    echo ""
else
    echo ""
    echo "✗ Download failed"
    echo "Check your network connection and try again"
    exit 1
fi
