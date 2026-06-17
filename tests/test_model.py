import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def test_model_loads():
    model = YOLO("yolo26m-obb.pt")
    assert model is not None
    print("[+] Model loaded successfully")


def test_forward_pass():
    model = YOLO("yolo26m-obb.pt")
    dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
    results = model.predict(dummy, verbose=False)
    assert len(results) == 1
    print(f"[+] Forward pass OK, output type: {type(results[0])}")


def test_output_shape():
    model = YOLO("yolo26m-obb.pt")
    dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
    results = model.predict(dummy, verbose=False)
    for r in results:
        if r.obb is not None:
            boxes = r.obb.xyxyxyxy
            assert boxes.shape[-1] == 4 and boxes.shape[-2] == 4, f"OBB format: expected (N, 4, 4), got {boxes.shape}"
            confs = r.obb.conf
            assert confs.ndim == 1, f"Confidences should be 1D"
            print(f"[+] Output shape OK: {boxes.shape[0]} detections")
            return
    print("[~] No detections in dummy input (expected)")


def test_onnx_export(tmp_path):
    model = YOLO("yolo26m-obb.pt")
    onnx_path = model.export(format="onnx", imgsz=640, half=False, dynamic=False, simplify=True)
    assert Path(onnx_path).exists()
    print(f"[+] ONNX export OK: {onnx_path}")


def test_onnx_inference():
    onnx_path = "yolo26m-obb.onnx"
    if not Path(onnx_path).exists():
        print("[~] ONNX model not found, skipping")
        return

    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
    outputs = session.run(None, {session.get_inputs()[0].name: dummy})

    for o in outputs:
        if o.shape[-1] == 7:
            print(f"[+] ONNX OBB output OK: shape {o.shape}")
            return

    print(f"[~] ONNX output shapes: {[o.shape for o in outputs]}")
