#!/bin/bash
# Configure conda for Perlmutter (one-time setup)
# This sets up conda to use $SCRATCH for package cache and environments

echo "================================================"
echo "Configuring Conda for Perlmutter"
echo "================================================"
echo ""

# Load conda
module load conda

# Create configuration directory
mkdir -p ~/.conda

# Create .condarc configuration file
cat > ~/.condarc << EOF
# Conda configuration for Perlmutter
# Uses \$SCRATCH for package cache and environments

pkgs_dirs:
  - $SCRATCH/.conda/pkgs

envs_dirs:
  - $SCRATCH/conda-envs
  - ~/.conda/envs

channels:
  - conda-forge
  - defaults

channel_priority: flexible
auto_activate_base: false
EOF

echo "Created ~/.condarc with the following settings:"
echo ""
cat ~/.condarc
echo ""

# Create directories
mkdir -p "$SCRATCH/.conda/pkgs"
mkdir -p "$SCRATCH/conda-envs"

echo ""
echo "Created directories:"
echo "  Package cache: $SCRATCH/.conda/pkgs"
echo "  Environments:  $SCRATCH/conda-envs"
echo ""

# Verify configuration
echo "Verifying configuration..."
conda config --show pkgs_dirs
conda config --show envs_dirs

echo ""
echo "================================================"
echo "Conda configured successfully!"
echo "================================================"
echo ""
echo "You can now run:"
echo "  bash perlmutter/setup/MANUAL_SETUP_DATA.sh"
echo "  bash perlmutter/setup/MANUAL_SETUP_ML.sh"
echo ""
echo "These settings are permanent and will persist across sessions."
