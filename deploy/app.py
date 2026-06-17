import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO

import torch

app = FastAPI(title="YOLO26-OBB Detection API", version="0.1.0")

model = None
model_path = os.environ.get("MODEL_PATH", "runs/obb/train/weights/best.pt")
device = os.environ.get("DEVICE", "") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@app.on_event("startup")
def load_model():
    global model
    path = Path(model_path)
    if not path.exists():
        print(f"[!] Model not found at {path}, checking default...")
        alt = Path("yolo26m-obb.pt")
        if alt.exists():
            path = alt
        else:
            print("[!] No model found, API will return errors on /predict")
            return
    print(f"[*] Loading model from {path}")
    model = YOLO(str(path))
    print(f"[+] Model loaded on {device}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": device}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    results = model.predict(img, verbose=False)[0]

    detections = []
    if results.obb is not None:
        boxes = results.obb.xyxyxyxy.cpu().numpy() if hasattr(results.obb, "xyxyxyxy") else None
        confs = results.obb.conf.cpu().numpy().tolist() if hasattr(results.obb, "conf") else []
        clss = results.obb.cls.cpu().numpy().tolist() if hasattr(results.obb, "cls") else []

        if boxes is not None:
            for i in range(len(boxes)):
                detections.append({
                    "class_id": int(clss[i]) if i < len(clss) else 0,
                    "confidence": float(confs[i]) if i < len(confs) else 0.0,
                    "polygon": boxes[i].reshape(-1).tolist(),
                })

    return JSONResponse({
        "image_width": results.orig_shape[1] if hasattr(results, "orig_shape") else 0,
        "image_height": results.orig_shape[0] if hasattr(results, "orig_shape") else 0,
        "detections": detections,
        "num_detections": len(detections),
    })


@app.get("/")
def root():
    return {"message": "YOLO26-OBB Detection API", "endpoints": ["/health", "/predict"]}
