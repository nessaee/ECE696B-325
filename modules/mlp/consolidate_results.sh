#!/bin/bash
# Script to consolidate results from all architectures and versions into a single LaTeX table

# Set the base directory for results
RESULTS_DIR="../../results/mlp_training_output"
OUTPUT_PATH="../../results/consolidated_results_table.tex"

# Run the consolidation script
python consolidate_results.py --results-dir "$RESULTS_DIR" --output "$OUTPUT_PATH"

echo "Consolidated results table generated at: $OUTPUT_PATH"
echo "CSV version also available at: ${OUTPUT_PATH%.tex}.csv"
