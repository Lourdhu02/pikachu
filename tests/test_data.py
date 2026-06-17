import yaml
from pathlib import Path


def test_train_yaml_exists():
    assert Path("configs/train.yaml").exists(), "train.yaml not found"


def test_train_yaml_valid():
    with open("configs/train.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "variant" in cfg
    assert "data" in cfg
    assert "epochs" in cfg
    assert "batch" in cfg


def test_augment_yaml_exists():
    assert Path("configs/augment.yaml").exists(), "augment.yaml not found"


def test_augment_yaml_valid():
    with open("configs/augment.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "hsv_h" in cfg
    assert "mosaic" in cfg
    assert "mixup" in cfg


def test_export_yaml_exists():
    assert Path("configs/export.yaml").exists(), "export.yaml not found"


def test_export_yaml_valid():
    with open("configs/export.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "format" in cfg
    assert "imgsz" in cfg
    assert "half" in cfg


def test_dataset_yaml_paths_exist():
    cfg_path = Path("configs/train.yaml")
    if not cfg_path.exists():
        return
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_path = cfg.get("data", "")
    if not data_path or not Path(data_path).exists():
        return

    with open(data_path) as f:
        dataset = yaml.safe_load(f)

    for key in ["train", "val"]:
        if key in dataset:
            path = Path(str(dataset[key]))
            if not path.is_absolute():
                path = Path(data_path).parent / path
            assert path.exists() or path.parent.exists(), f"{key} path {path} does not exist"


def test_label_format_obb():
    label_dir = Path("../dataset/labels/train")
    if not label_dir.exists():
        label_dir = Path("../dataset/train/labels")

    if not label_dir.exists():
        return

    for lbl_file in sorted(label_dir.glob("*.txt"))[:5]:
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                assert len(parts) == 9, f"OBB label must have 9 values (class + 8 coords), got {len(parts)}: {line}"
                cls_id = int(parts[0])
                assert cls_id >= 0, f"Class ID must be non-negative"
                coords = list(map(float, parts[1:]))
                for c in coords:
                    assert 0.0 <= c <= 1.0, f"OBB coordinates must be normalized [0,1], got {c}"
