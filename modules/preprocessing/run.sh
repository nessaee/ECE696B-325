#!/bin/bash

# Enhanced run script for the modular preprocessing pipeline

# Default settings
DATA_PATH="../../data/dataset/original"
METADATA_PATH="../../data/dataset/labels/preprocess/metadata.csv"
ANNOTATIONS_PATH="../../data/dataset/labels/preprocess/annotations.csv"

OUTPUT_PATH="../../data/dataset/processed/"
OUTPUT_TYPES="stacked rgb normalized entropy depth"
PARALLEL_JOBS=16
EXTRAS="--calculate-stats"

CMD="python run.py \
    \"$DATA_PATH\" \
    \"$ANNOTATIONS_PATH\" \
    \"$OUTPUT_PATH\" \
    --only $OUTPUT_TYPES \
    --jobs $PARALLEL_JOBS \
    --calculate-stats \
    --skip-rename \
    --metadata-csv $METADATA_PATH \
    $EXTRAS"

# Execute the command
echo "Executing: $CMD"
eval $CMD
