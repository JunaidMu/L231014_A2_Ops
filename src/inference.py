"""
Task 4.1: VisionGuard Inference Module
Project: L231014_A2_Ops (VisionGuard)
Author: Junaid Muhammad (23L-1014)

Loads the champion Keras model ONCE at start-up (when imported or initialized),
preprocesses an uploaded image in memory (without saving to disk) identically to
training (3-channel RGB, bilinear resize to 160x160, float32 [0, 255] fed into the
model's internal [-1, 1] MobileNetV2 Rescaling layer), and returns a label,
confidence, probability, and caveat.
"""

import datetime
import getpass
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "mobilenetv2_finetuned_best.keras"))
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.48"))
CLOUD_API_URL = os.getenv("CLOUD_API_URL", "").strip()
MODEL_NAME = "VisionGuard_MobileNetV2_v2@champion"
IMG_SIZE = (160, 160)

# Allowed image extensions & magic formats
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class CorruptOrInvalidImageError(ValueError):
    """Raised when a file has an image extension but cannot be decoded as a valid image."""


class VisionGuardPredictor:
    """
    Singleton-style predictor that loads the champion Keras model ONCE at startup.
    Never reloads the model per HTTP request.
    """

    def __init__(self, model_path: str = MODEL_PATH, threshold: float = DECISION_THRESHOLD):
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path
        self.threshold = float(threshold)
        self.model_name = MODEL_NAME
        self._model = None
        self._load_model_once()

    def _load_model_once(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"[CRITICAL] Champion model file not found at: {self.model_path}. "
                "Check the MODEL_PATH environment variable."
            )
        print(f"[STARTUP] Loading champion model ONCE from: {self.model_path} ...")
        self._model = tf.keras.models.load_model(self.model_path)
        print(f"[STARTUP] Champion model '{self.model_name}' loaded and ready in memory.")

    @staticmethod
    def preprocess_image_bytes(raw_bytes: bytes) -> tuple[tf.Tensor, dict]:
        """
        Validates and decodes raw image bytes in memory without touching disk,
        then applies the exact same TensorFlow bilinear resize to 160x160 RGB float32
        used in Task 2.1 training.
        """
        if not raw_bytes or len(raw_bytes) == 0:
            raise CorruptOrInvalidImageError("Uploaded file is empty (0 bytes).")

        # Step 1: Verify image header & integrity with Pillow in-memory
        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.verify()
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.load()
                width, height = img.size
                fmt = img.format or "UNKNOWN"
                mode = img.mode
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise CorruptOrInvalidImageError(
                "The uploaded file looks like an image by name, but its binary contents "
                f"are corrupt or cannot be decoded ({exc})."
            ) from exc

        # Step 2: Decode & resize identically to training pipeline (Task 2.1)
        try:
            img_tensor = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
            img_tensor.set_shape([None, None, 3])
            img_resized = tf.image.resize(
                img_tensor, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR
            )
            img_float = tf.cast(img_resized, tf.float32)  # [0.0, 255.0]
            batch_tensor = tf.expand_dims(img_float, axis=0)  # (1, 160, 160, 3)
        except Exception as exc:
            raise CorruptOrInvalidImageError(
                f"TensorFlow image decoder rejected the file bytes: {exc}"
            ) from exc

        meta = {
            "width": width,
            "height": height,
            "format": fmt,
            "color_mode": mode,
            "size_kb": round(len(raw_bytes) / 1024.0, 2),
        }
        return batch_tensor, meta

    def predict_bytes(
        self, raw_bytes: bytes, filename: str = "upload.jpg", use_cloud_api: bool = False
    ) -> dict:
        """
        Runs inference on raw image bytes (either locally using the pre-loaded champion
        model or via the remote Cloud API for Task 8 if configured and requested).
        """
        batch_tensor, meta = self.preprocess_image_bytes(raw_bytes)

        # Task 8 pathway: if use_cloud_api is enabled and CLOUD_API_URL is set, call remote API
        if use_cloud_api and CLOUD_API_URL:
            resp = requests.post(
                CLOUD_API_URL,
                files={"file": (filename, raw_bytes, "image/jpeg")},
                timeout=30,
            )
            resp.raise_for_status()
            api_data = resp.json()
            prob_ai = float(api_data["prob_ai"])
            served_by = f"{self.model_name} (via Cloud API: {CLOUD_API_URL})"
        else:
            prob_ai = float(self._model.predict(batch_tensor, verbose=0)[0][0])
            served_by = self.model_name

        is_ai = prob_ai >= self.threshold
        verdict = "AI-Generated" if is_ai else "Real Image"
        confidence = prob_ai if is_ai else (1.0 - prob_ai)

        caveat = (
            "Honest Caveat: VisionGuard is trained on 160x160 RGB crops (Test ROC-AUC = 0.890, "
            "Accuracy = 80.1%). Heavily stylized digital illustrations, long-exposure photography, "
            "or extreme aspect-ratio images may trigger false positives, while hyper-realistic "
            "portraits may occasionally evade detection."
        )

        return {
            "verdict": verdict,
            "is_ai": is_ai,
            "prob_ai": round(prob_ai, 4),
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100.0, 2),
            "threshold": self.threshold,
            "model_name": served_by,
            "caveat": caveat,
            "meta": meta,
        }


# Module-level singleton instance loaded ONCE when imported by Flask or CLI
_predictor_instance = None


def get_predictor() -> VisionGuardPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = VisionGuardPredictor()
    return _predictor_instance


if __name__ == "__main__":
    # Task 4.1 Evidence: Console output for two known images (one AI, one Real)
    print("=" * 85)
    print("TASK 4.1: INFERENCE MODULE CONSOLE TEST ON TWO KNOWN IMAGES (L231014_A2_Ops)")
    print(f"User: {getpass.getuser()} | Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 85)

    predictor = get_predictor()
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "test.csv")

    known_ai = test_df[test_df["label"] == 1].iloc[0]
    known_real = test_df[test_df["label"] == 0].iloc[0]

    for idx, sample in enumerate([known_ai, known_real], start=1):
        img_path = PROJECT_ROOT / sample["filepath"]
        raw_bytes = img_path.read_bytes()
        result = predictor.predict_bytes(raw_bytes, filename=img_path.name)
        print(f"\n[Known Image #{idx}] File: {sample['filepath']}")
        print(f"  Ground Truth Class : {sample['class']} (label={sample['label']})")
        print(f"  Image Metadata     : {result['meta']['width']}x{result['meta']['height']} ({result['meta']['format']}, {result['meta']['size_kb']} KB)")
        print(f"  Predicted Verdict  : {result['verdict']}")
        print(f"  Confidence         : {result['confidence_pct']:.2f}% (Raw P(AI) = {result['prob_ai']:.4f}, Threshold = {result['threshold']})")
        print(f"  Model Used         : {result['model_name']}")
    print("\n" + "=" * 85)
