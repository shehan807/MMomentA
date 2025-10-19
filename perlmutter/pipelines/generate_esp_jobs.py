#!/usr/bin/env python
"""
Generate SLURM job scripts for parallel ESP validation.

This script:
1. Reads the ZINC 100k dataset to determine test set size
2. Splits test molecules into N chunks
3. Generates N SLURM job scripts, one for each chunk
4. Each chunk job validates ~1000 molecules independently

Usage:
    python perlmutter/pipelines/generate_esp_jobs.py \
        --dataset /global/u1/p/parmar/MoML/MMomentA/data/zinc_100k_mpfit.h5 \
        --model-dir /global/u1/p/parmar/MoML/MMomentA/runs/zinc_100k_multipoles \
        --output-dir /global/u1/p/parmar/MoML/MMomentA/figures/zinc_100k_esp_validation \
        --n-chunks 20 \
        --time-limit 36:00:00 \
        --n-jobs 32
"""

import argparse
import h5py
from pathlib import Path
import sys

# Add MMomentA to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from MMomentA.data.storage import load_dataset_hdf5


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH -J zinc_esp_c{chunk_id:02d}
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -G 0
#SBATCH -c 128
#SBATCH --mem=256GB
#SBATCH -t {time_limit}
#SBATCH -A m1266
#SBATCH -o {log_dir}/zinc_100k_esp_chunk_{chunk_id:02d}_%j.out
#SBATCH -e {log_dir}/zinc_100k_esp_chunk_{chunk_id:02d}_%j.err

# ESP Validation Chunk {chunk_id}/{n_chunks}
# Processes test molecules [{start_idx}, {end_idx})
# Estimated: ~{n_molecules} molecules

# Set working directory
MMOMENTA_DIR="{mmomenta_dir}"
cd "$MMOMENTA_DIR"

echo "================================================"
echo "ESP Validation Chunk {chunk_id}/{n_chunks}"
echo "Job ID: ${{SLURM_JOB_ID:-interactive}}"
echo "Node: ${{SLURM_NODELIST:-$(hostname)}}"
echo "================================================"
echo "Processing molecules [{start_idx}, {end_idx})"
echo "Expected count: ~{n_molecules} molecules"
echo ""

# Load conda
module load conda

# Activate data environment (has Psi4, openff-toolkit, etc.)
DATA_ENV="$SCRATCH/conda-envs/mmomenta-data"
conda activate "$DATA_ENV"

# Run chunk validation
echo "Starting ESP validation for chunk {chunk_id}..."
python scripts/validate_esp_comparison_chunk.py \\
    --dataset {dataset} \\
    --model-dir {model_dir} \\
    --output-dir {output_dir} \\
    --chunk-id {chunk_id} \\
    --start-idx {start_idx} \\
    --end-idx {end_idx} \\
    --qm-method hf \\
    --qm-basis "6-31G*" \\
    --device cpu \\
    --n-jobs {n_jobs}

echo ""
echo "================================================"
echo "Chunk {chunk_id}/{n_chunks} Complete!"
echo "================================================"
echo "Job ID: ${{SLURM_JOB_ID:-interactive}}"
echo "Time: $SECONDS seconds ($((SECONDS/3600))h $((SECONDS%3600/60))m)"
echo "Results: {output_dir}/esp_validation_results_chunk_{chunk_id:02d}.json"
echo ""
"""


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM jobs for parallel ESP validation")

    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to HDF5 dataset")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for validation results")
    parser.add_argument("--n-chunks", type=int, default=20,
                       help="Number of chunks to split into (default: 20)")
    parser.add_argument("--time-limit", type=str, default="36:00:00",
                       help="Time limit per chunk job (default: 36:00:00)")
    parser.add_argument("--n-jobs", type=int, default=32,
                       help="Number of parallel workers per chunk (default: 32)")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory for SLURM logs (default: {mmomenta_dir}/logs)")

    args = parser.parse_args()

    # Determine MMomentA root directory (3 levels up from this script)
    script_path = Path(__file__).resolve()
    mmomenta_dir = script_path.parent.parent.parent

    # Set log directory
    if args.log_dir is None:
        log_dir = mmomenta_dir / "logs"
    else:
        log_dir = Path(args.log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("ESP Validation Job Generator")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Chunks: {args.n_chunks}")
    print(f"Time limit: {args.time_limit}")
    print(f"Workers per chunk: {args.n_jobs}")
    print("")

    # Load dataset to get test molecule count
    print("Loading dataset to determine test set size...")
    molecule_data_list, metadata = load_dataset_hdf5(args.dataset)

    # Get test molecules
    if metadata and 'splits' in metadata:
        test_molecule_ids = metadata['splits'].get('test', [])
        if test_molecule_ids:
            # Create mapping from molecule_id to index
            id_to_idx = {mol.molecule_id: idx for idx, mol in enumerate(molecule_data_list)}
            test_indices = [id_to_idx[mol_id] for mol_id in test_molecule_ids if mol_id in id_to_idx]
        else:
            test_indices = list(range(len(molecule_data_list)))
    else:
        print("WARNING: No split information found - using all molecules")
        test_indices = list(range(len(molecule_data_list)))

    n_test_molecules = len(test_indices)
    print(f"Total test molecules: {n_test_molecules}")
    print("")

    # Calculate chunk size
    chunk_size = (n_test_molecules + args.n_chunks - 1) // args.n_chunks
    print(f"Chunk size: ~{chunk_size} molecules per chunk")
    print("")

    # Generate job scripts
    jobs_dir = mmomenta_dir / "perlmutter" / "pipelines" / "generated_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_chunks} SLURM job scripts...")
    print(f"Output directory: {jobs_dir}")
    print("")

    job_files = []

    for chunk_id in range(args.n_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, n_test_molecules)
        n_molecules = end_idx - start_idx

        if n_molecules == 0:
            break

        # Generate SLURM script from template
        job_script = SLURM_TEMPLATE.format(
            chunk_id=chunk_id,
            n_chunks=args.n_chunks,
            start_idx=start_idx,
            end_idx=end_idx,
            n_molecules=n_molecules,
            dataset=args.dataset,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            time_limit=args.time_limit,
            n_jobs=args.n_jobs,
            mmomenta_dir=mmomenta_dir,
            log_dir=log_dir
        )

        # Write job script
        job_file = jobs_dir / f"run_zinc_100k_step6_chunk_{chunk_id:02d}.slurm"
        with open(job_file, 'w') as f:
            f.write(job_script)

        job_files.append(job_file)

        print(f"  Chunk {chunk_id:02d}: [{start_idx:5d}, {end_idx:5d}) - {n_molecules:4d} molecules -> {job_file.name}")

    print("")
    print(f"✓ Generated {len(job_files)} job scripts in {jobs_dir}")
    print("")
    print("To submit all jobs, run:")
    print(f"  bash {mmomenta_dir}/perlmutter/pipelines/run_zinc_100k_step6_parallel.sh")
    print("")


if __name__ == "__main__":
    main()
