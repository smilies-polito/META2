#!/bin/bash

# Check if a command-line argument is provided
if [ -z "$1" ]; then
    # No argument provided, use default path
    TARGET_PATH="../META2"
else
    # Use the provided argument as the target path
    TARGET_PATH="$1"
fi

singularity shell -B ./:/meta_opt_model --pwd /meta_opt_model "$TARGET_PATH"/