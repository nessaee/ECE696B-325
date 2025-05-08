#!/bin/bash
# Script to consolidate results from all architectures and versions into a single LaTeX table

# Set the base directory for results
RESULTS_DIR="../../results/mlp_training_output"
OUTPUT_PATH="../../results/consolidated_results_table.tex"

# Default options
HIGHLIGHT="--highlight-best"
PERFORMANCE="--include-performance"
EFFICIENCY="--include-efficiency"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-highlight)
      HIGHLIGHT="--no-highlight-best"
      shift
      ;;
    --no-performance)
      PERFORMANCE="--no-performance"
      shift
      ;;
    --no-efficiency)
      EFFICIENCY="--no-efficiency"
      shift
      ;;
    --help)
      echo "Usage: ./consolidate_results.sh [options]"
      echo "Options:"
      echo "  --no-highlight     Don't highlight the best metrics in each column"
      echo "  --no-performance   Exclude performance metrics (F1, Accuracy, AUC)"
      echo "  --no-efficiency    Exclude efficiency metrics (Params, Train, Infer)"
      echo "  --help             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './consolidate_results.sh --help' for usage information."
      exit 1
      ;;
  esac
done

# Run the consolidation script with the specified options
python consolidate_results.py --results-dir "$RESULTS_DIR" --output "$OUTPUT_PATH" $HIGHLIGHT $PERFORMANCE $EFFICIENCY

echo "Consolidated results table generated at: $OUTPUT_PATH"
echo "Options used: $HIGHLIGHT $PERFORMANCE $EFFICIENCY"
echo "CSV version also available at: ${OUTPUT_PATH%.tex}.csv"
