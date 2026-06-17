# YOLO26-OBB Training Engine

![Python](https://img.shields.io/badge/python-%3E=3.10-blue)
![Ultralytics](https://img.shields.io/badge/ultralytics-%3E=8.4.0-green)
![License](https://img.shields.io/badge/license-MIT-green)

End-to-end OBB (oriented bounding box) detection engine using **YOLO26-obb** — the latest state-of-the-art model as of June 2026. Built for single-class dial detection with 100% local tracking (no W&B, no MLflow).

## Quickstart

```bash
pip install -r requirements.txt
python engine/train.py --variant m --data ../dataset/dataset.yaml
```

## Directory Map

```
├── configs/          # YAML configs (train, augment, export)
├── engine/           # Core scripts (train, eval, infer, export, tune, benchmark)
├── data/             # Dataset analysis, augmentation viz, split tools
├── scripts/          # Shell wrappers for common tasks
├── tests/            # pytest tests
├── deploy/           # FastAPI serving + Docker
└── results/          # Output CSVs and plots (gitignored)
```

## Usage

### Train

```bash
# Default training
python engine/train.py

# CLI overrides
python engine/train.py --variant l --epochs 200 --batch 8 --imgsz 1280

# Resume from checkpoint
python engine/train.py --resume --weights runs/obb/train/weights/last.pt

# Finetune (freeze backbone)
python engine/train.py --finetune --freeze-layers 10
```

### Evaluate

```bash
python engine/evaluate.py --weights runs/obb/train/weights/best.pt
```

### Export to ONNX

```bash
python engine/export.py --weights runs/obb/train/weights/best.pt
```

### Benchmark

```bash
python engine/benchmark.py --weights runs/obb/train/weights/best.pt
```

### Hyperparameter Tuning

```bash
python engine/tune.py --n-trials 30 --epochs 50
```

### Inference

```bash
python engine/infer.py --weights runs/obb/train/weights/best.pt --source ../dataset/images/val
python engine/infer.py --weights runs/obb/train/weights/best.onnx --source test.jpg --save-json
```

### Reproduce (Full Pipeline)

```bash
bash scripts/reproduce.sh m 100 16
```

## Results

All metrics are logged locally as CSVs under `results/{run_name}/metrics.csv`. Plots are automatically generated after training.

## Model Variants

| Variant | Params | mAP50-95 (DOTA) |
|---------|--------|-----------------|
| n       | ~5M    | 78.1            |
| s       | ~10M   | 80.5            |
| m       | ~21M   | 82.5            |
| l       | ~26M   | 83.1            |
| x       | ~42M   | 83.8            |

Pretrained weights auto-download: `yolo26{variant}-obb.pt`

## Makefile

```bash
make train ARGS="--variant m --epochs 100"
make eval W=runs/obb/train/weights/best.pt
make export W=runs/obb/train/weights/best.pt
make test
make lint
make setup
```
