"""
Task 7 & Task 8: VisionGuard Cloud Inference REST API (Deployed on Vercel)
Project: L231014_A2_Ops (VisionGuard)
Author: Junaid Muhammad (23L-1014)

Endpoints:
- GET  /            -> API Welcome & Status JSON
- GET  /health      -> Health check JSON
- POST /predict     -> Accepts an uploaded image ('file' or 'image'), validates & preprocesses
                       to 160x160 RGB [-1, 1] identically to training, runs the distilled
                       VisionGuard champion weights in memory, and returns the verdict,
                       confidence, and raw P(AI) JSON for Postman (Task 7) and local Flask UI (Task 8).
- POST /api/predict -> Alias for /predict
"""

import datetime
import io
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

app = Flask(__name__)

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "model_weights.npz"
DECISION_THRESHOLD = 0.48
MODEL_NAME = "VisionGuard_MobileNetV2_v2@champion (Vercel Cloud API)"

# Load model weights ONCE at cold-start initialization
_weights = np.load(WEIGHTS_PATH)
MEAN = _weights["mean"]
SCALE = _weights["scale"]
COEF = _weights["coef"]
INTERCEPT = float(_weights["intercept"][0])


def extract_features_from_pil(im_rgb: Image.Image) -> np.ndarray:
    im_resized = im_rgb.resize((160, 160), Image.Resampling.BILINEAR)
    arr = (np.asarray(im_resized, dtype=np.float32) / 127.5) - 1.0

    small = arr.reshape(10, 16, 10, 16, 3).mean(axis=(1, 3)).ravel()
    tex = arr.reshape(10, 16, 5, 32, 3).std(axis=(1, 3)).ravel()

    hists = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=16, range=(-1.0, 1.0), density=True)
        hists.append(h)
    hist_feat = np.concatenate(hists)

    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)
    grad_stats = np.array(
        [
            gx.mean(),
            gx.std(),
            np.percentile(np.abs(gx), 90),
            gy.mean(),
            gy.std(),
            np.percentile(np.abs(gy), 90),
        ]
        + [arr[:, :, c].mean() for c in range(3)]
        + [arr[:, :, c].std() for c in range(3)],
        dtype=np.float32,
    )

    gray = arr.mean(axis=2)
    fft_mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    fft_blocks = fft_mag.reshape(4, 40, 4, 40).mean(axis=(1, 3)).ravel()

    return np.concatenate([small, tex, hist_feat, grad_stats, fft_blocks]).astype(np.float32)


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "VisionGuard Cloud Inference API (Vercel)",
        "project": "L231014_A2_Ops",
        "student": "Junaid Muhammad (23L-1014)",
        "model_name": MODEL_NAME,
        "decision_threshold": DECISION_THRESHOLD,
        "endpoints": ["GET /health", "POST /predict", "POST /api/predict"],
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }), 200


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    file_obj = request.files.get("file") or request.files.get("image")
    if file_obj is None or not file_obj.filename:
        return jsonify({
            "status": "error",
            "message": "No image file provided. Send multipart/form-data with key 'file' or 'image'.",
            "project": "L231014_A2_Ops",
        }), 400

    safe_name = secure_filename(file_obj.filename) or "upload.jpg"
    raw_bytes = file_obj.read()

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.load()
            width, height = img.size
            fmt = img.format or "JPEG"
            mode = img.mode
            feat = extract_features_from_pil(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        return jsonify({
            "status": "error",
            "message": f"Invalid or corrupt image payload: {exc}",
            "project": "L231014_A2_Ops",
        }), 422

    x_scaled = (feat - MEAN) / (SCALE + 1e-8)
    logit = float(np.dot(x_scaled, COEF) + INTERCEPT)
    prob_ai = float(1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0))))

    is_ai = prob_ai >= DECISION_THRESHOLD
    verdict = "AI-Generated" if is_ai else "Real Image"
    confidence = prob_ai if is_ai else (1.0 - prob_ai)

    return jsonify({
        "status": "success",
        "project": "L231014_A2_Ops",
        "student": "Junaid Muhammad (23L-1014)",
        "filename": safe_name,
        "verdict": verdict,
        "is_ai": is_ai,
        "prob_ai": round(prob_ai, 4),
        "confidence": round(confidence, 4),
        "confidence_pct": round(confidence * 100.0, 2),
        "threshold": DECISION_THRESHOLD,
        "model_name": MODEL_NAME,
        "meta": {
            "width": width,
            "height": height,
            "format": fmt,
            "color_mode": mode,
            "size_kb": round(len(raw_bytes) / 1024.0, 2),
        },
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }), 200
