#!/bin/bash
# Usage: ./scripts/export.sh [weights_path]
WEIGHTS=${1:-runs/obb/train/weights/best.pt}
python engine/export.py --weights "$WEIGHTS"
