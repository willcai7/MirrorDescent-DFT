#!/bin/bash

stamp_list=(20250320-155132)
for stamp in "${stamp_list[@]}"; do
    echo "Processing stamp: $stamp"
    python src/plots/plot_3D.py \
        --stamp="$stamp" || echo "Error processing stamp: $stamp"
done