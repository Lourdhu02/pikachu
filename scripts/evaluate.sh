#!/bin/bash
# Usage: ./scripts/evaluate.sh [weights_path]
WEIGHTS=${1:-runs/obb/train/weights/best.pt}
python engine/evaluate.py --weights "$WEIGHTS"
