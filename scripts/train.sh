#!/bin/bash
# Usage: ./scripts/train.sh [variant] [epochs] [batch]
VARIANT=${1:-m}
EPOCHS=${2:-100}
BATCH=${3:-16}
python engine/train.py --variant "$VARIANT" --epochs "$EPOCHS" --batch "$BATCH"
