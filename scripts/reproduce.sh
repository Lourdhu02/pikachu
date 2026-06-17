#!/bin/bash
# Full reproduce pipeline: train -> export -> benchmark -> evaluate
set -e

VARIANT=${1:-m}
EPOCHS=${2:-100}
BATCH=${3:-16}

echo "=========================================="
echo "[*] Full reproduce pipeline (YOLO26-${VARIANT})"
echo "    Epochs: ${EPOCHS} | Batch: ${BATCH}"
echo "=========================================="

# Step 1: Train
echo "[*] Step 1/4: Training..."
python engine/train.py --variant "$VARIANT" --epochs "$EPOCHS" --batch "$BATCH"
echo "[+] Training complete!"

# Step 2: Export
WEIGHTS="runs/obb/train_${VARIANT}/weights/best.pt"
echo "[*] Step 2/4: Exporting to ONNX..."
python engine/export.py --weights "$WEIGHTS"
echo "[+] Export complete!"

# Step 3: Benchmark
echo "[*] Step 3/4: Benchmarking..."
python engine/benchmark.py --weights "$WEIGHTS"
echo "[+] Benchmark complete!"

# Step 4: Evaluate
echo "[*] Step 4/4: Evaluating..."
python engine/evaluate.py --weights "$WEIGHTS"
echo "[+] Evaluation complete!"

echo "=========================================="
echo "[+] Reproduce pipeline finished successfully!"
echo "=========================================="
