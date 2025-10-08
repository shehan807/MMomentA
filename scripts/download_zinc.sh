#!/bin/bash
# Download GEOM dataset (ZINC subset with pre-computed conformers)

DATASET_DIR="/storage/home/hcoda1/4/sparmar32/r-jmcdaniel43-0/scripts/MMomentA_MoML/MMomentA/data"

echo "================================================"
echo "Downloading GEOM Dataset (ZINC Subset)"
echo "================================================"
echo ""
echo "Dataset details:"
echo "  - 450,000 molecules from ZINC/ChEMBL"
echo "  - Multiple conformers per molecule"
echo "  - Pre-computed DFT geometries"
echo "  - Size: varies by subset"
echo "  - Source: https://doi.org/10.7910/DVN/JNGTDF"
echo ""

mkdir -p "$DATASET_DIR/geom"
cd "$DATASET_DIR/geom"

echo "Choose GEOM subset:"
echo "  1. GEOM-Drugs (28K molecules, ~2 GB) - RECOMMENDED for MVP"
echo "  2. GEOM-QM9 (130K molecules, ~5 GB)"
echo "  3. Full GEOM (450K molecules, ~20 GB)"
echo ""

# Download drugs subset (most practical for MVP)
echo "Downloading GEOM-Drugs subset..."
echo ""

# GEOM is distributed via Harvard Dataverse
# User will need to download manually due to terms of use

echo "GEOM requires manual download due to terms of use."
echo ""
echo "Steps:"
echo "1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF"
echo "2. Accept terms and download GEOM-Drugs subset"
echo "3. Place files in: $DATASET_DIR/geom/"
echo ""
echo "Alternative: Use OpenFF ZINC dataset"
echo ""
echo "OpenFF provides curated ZINC molecules:"
wget https://raw.githubusercontent.com/openforcefield/qca-dataset-submission/master/submissions/2022-12-13-OpenFF-ZINC-Fragments/zinc_fragments.smi

if [ $? -eq 0 ]; then
    echo "✓ Downloaded ZINC SMILES list"
    echo ""
    echo "Next step: Convert SMILES to molecules"
    echo "python scripts/smiles_to_molecules.py --input data/geom/zinc_fragments.smi --output data/zinc_molecules.pkl --max-molecules 500"
else
    echo ""
    echo "Manual option: Create ZINC subset from SMILES"
    echo ""
    echo 'You can use these representative ZINC SMILES:'
    cat > zinc_sample.smi << 'EOF'
CC(C)Cc1ccc(cc1)C(C)C(=O)O
COc1ccc(cc1)CCN
c1ccc(cc1)C(c2ccccc2)O
CC(C)NCC(c1ccc(cc1)O)O
CN1CCN(CC1)c2ncccn2
COc1cc(ccc1O)CC(C(=O)O)N
c1ccc2c(c1)ccc3c2cccc3
Cc1ccc(cc1)S(=O)(=O)Nc2nccs2
CCN(CC)C(=O)C
c1ccc(cc1)Oc2ccccc2
EOF

    echo ""
    echo "Saved sample ZINC SMILES to: zinc_sample.smi"
    echo ""
    echo "To create dataset:"
    echo "python scripts/smiles_to_molecules.py --input data/geom/zinc_sample.smi --output data/zinc_molecules.pkl"
fi

echo ""
echo "================================================"
echo "Recommended MVP Approach"
echo "================================================"
echo ""
echo "For fast MVP, use the OpenFF ZINC fragments:"
echo "1. python scripts/smiles_to_molecules.py --input data/geom/zinc_fragments.smi --output data/zinc_molecules.pkl --max-molecules 100"
echo "2. python scripts/prepare_dataset.py --input data/zinc_molecules.pkl --output data/zinc_mpfit.h5 --n-jobs 8"
echo "3. python scripts/train_spice.py --dataset data/zinc_mpfit.h5 --output-dir runs/zinc_transfer --pretrained runs/baseline/best_model.pt --device cuda"
