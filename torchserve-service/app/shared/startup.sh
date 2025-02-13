#!/bin/bash
set -e  # Exit immediately if a command fails

echo "Installing dependencies from all requirements.txt files..."
find /app -type f -name "requirements.txt" -print0 | while IFS= read -r -d '' file; do
    echo "Installing from: $file"
    pip install -r "$file"
done

echo "Starting TorchServe..."
exec torchserve --start --model-store model_store --no-config-snapshots --enable-model-api --foreground
