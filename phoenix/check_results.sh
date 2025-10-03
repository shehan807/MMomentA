#!/bin/bash
# Quick script to check training results

echo "================================================"
echo "MMomentA Training Results Summary"
echo "================================================"

# Function to print results from a directory
print_results() {
    local dir=$1
    local name=$2

    if [ -f "$dir/training_results.json" ]; then
        echo ""
        echo "$name"
        echo "----------------------------------------"

        python << EOF
import json
from pathlib import Path

results_file = Path("$dir/training_results.json")
if results_file.exists():
    with open(results_file) as f:
        data = json.load(f)

    results = data.get('results', {})
    model_config = data.get('model_config', {})

    print(f"  Features: {model_config.get('feature_units', 'N/A')}")
    print(f"  Depth: {model_config.get('depth', 'N/A')}, Width: {model_config.get('width', 'N/A')}")
    print(f"  Best Val RMSE: {results.get('best_val_rmse', 'N/A'):.5f}")
    print(f"  Best Epoch: {results.get('best_epoch', 'N/A')}")

    if 'test_metrics' in results:
        test = results['test_metrics']
        print(f"  Test RMSE: {test.get('val_rmse', 'N/A'):.5f}")
        print(f"  Test MAE: {test.get('val_mae', 'N/A'):.5f}")

    # Check if checkpoint exists
    checkpoint = Path("$dir/checkpoints/best_model.pt")
    if checkpoint.exists():
        size_mb = checkpoint.stat().st_size / (1024 * 1024)
        print(f"  Checkpoint: ✓ ({size_mb:.1f} MB)")
    else:
        print(f"  Checkpoint: ✗ Not found")
else:
    print("  No results found")
EOF
    else
        echo ""
        echo "$name"
        echo "----------------------------------------"
        echo "  No results found (job may still be running or failed)"
    fi
}

# Check baseline
if [ -d "runs/spice_baseline" ]; then
    print_results "runs/spice_baseline" "BASELINE (No Multipoles)"
fi

# Check multipoles
if [ -d "runs/spice_multipoles" ]; then
    print_results "runs/spice_multipoles" "WITH MULTIPOLES"
fi

# Check ZINC
if [ -d "runs/zinc_finetune" ]; then
    print_results "runs/zinc_finetune" "ZINC TRANSFER LEARNING"
fi

# Check ablation study
if [ -d "runs/ablation" ]; then
    echo ""
    echo "ABLATION STUDY"
    echo "----------------------------------------"

    python << 'EOF'
import json
import numpy as np
from pathlib import Path

configs = {
    'Baseline': 'runs/ablation/baseline_run',
    'Multipoles': 'runs/ablation/multipoles_run',
    'Deeper Baseline': 'runs/ablation/deeper_baseline_run'
}

for config_name, base_path in configs.items():
    test_rmses = []

    for run in [1, 2, 3]:
        results_file = Path(f"{base_path}{run}/training_results.json")
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
                if 'test_metrics' in results['results']:
                    test_rmses.append(results['results']['test_metrics']['val_rmse'])

    if test_rmses:
        print(f"  {config_name}: {np.mean(test_rmses):.5f} ± {np.std(test_rmses):.5f} ({len(test_rmses)} runs)")
    else:
        print(f"  {config_name}: No results")
EOF
fi

# Check running jobs
echo ""
echo "================================================"
echo "Current Jobs"
echo "================================================"

JOBS=$(squeue -u $USER -h 2>/dev/null)
if [ -n "$JOBS" ]; then
    echo "$JOBS"
else
    echo "No jobs currently running"
fi

# Recent completed jobs
echo ""
echo "Recent Completed Jobs (last 5)"
echo "================================================"

ls -lt phoenix/logs/*.out 2>/dev/null | head -5 | while read line; do
    file=$(echo $line | awk '{print $NF}')
    if [ -f "$file" ]; then
        job_id=$(basename $file .out)
        echo "$job_id"
    fi
done

echo ""
echo "================================================"
echo ""
echo "To view detailed results:"
echo "  cat runs/spice_multipoles/training_results.json | python -m json.tool"
echo ""
echo "To view training logs:"
echo "  tail -f phoenix/logs/<job_id>.out"
echo ""
