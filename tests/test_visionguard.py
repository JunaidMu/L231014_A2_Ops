"""
Task 6.6: Automated Unit & Integration Tests for VisionGuard (L231014_A2_Ops)
Author: Junaid Muhammad (23L-1014)
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app import create_app, db  # noqa: E402
from inference import CorruptOrInvalidImageError, VisionGuardPredictor  # noqa: E402


def create_valid_jpeg_bytes(width: int = 160, height: int = 160) -> bytes:
    img = Image.new("RGB", (width, height), color=(60, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_preprocess_valid_image():
    raw_bytes = create_valid_jpeg_bytes(240, 180)
    tensor, meta = VisionGuardPredictor.preprocess_image_bytes(raw_bytes)
    assert tuple(tensor.shape) == (1, 160, 160, 3)
    assert meta["width"] == 240
    assert meta["height"] == 180
    assert meta["format"] == "JPEG"


def test_preprocess_rejects_corrupt_bytes():
    with pytest.raises(CorruptOrInvalidImageError):
        VisionGuardPredictor.preprocess_image_bytes(b"NOT_AN_IMAGE_BINARY_STREAM")


@pytest.fixture()
def client(monkeypatch):
    class DummyPredictor:
        _model = True

        def predict_bytes(self, raw_bytes, filename="test.jpg", use_cloud_api=False):
            _, meta = VisionGuardPredictor.preprocess_image_bytes(raw_bytes)
            return {
                "verdict": "AI-Generated",
                "is_ai": True,
                "prob_ai": 0.9123,
                "confidence": 0.9123,
                "confidence_pct": 91.23,
                "threshold": 0.48,
                "model_name": "VisionGuard_MobileNetV2_v2@champion",
                "caveat": "Test caveat",
                "meta": meta,
            }

    import app as app_module

    monkeypatch.setattr(app_module, "get_predictor", lambda: DummyPredictor())

    flask_app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "SKIP_MODEL_LOAD": True,
            "MAX_UPLOAD_MB": 2.0,
        }
    )
    with flask_app.test_client() as test_client:
        with flask_app.app_context():
            db.create_all()
        yield test_client


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["project"] == "L231014_A2_Ops"


def test_reject_wrong_filetype(client):
    data = {"image": (io.BytesIO(b"hello world"), "notes.txt")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert b"Unsupported File Type Rejected" in resp.data


def test_reject_corrupt_fake_image(client):
    data = {"image": (io.BytesIO(b"corrupt_header_bytes"), "fake_photo.jpg")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 422
    assert b"Undecodable / Corrupt Image Rejected" in resp.data


def test_valid_prediction_logs_to_database(client):
    raw_jpeg = create_valid_jpeg_bytes(200, 200)
    data = {"image": (io.BytesIO(raw_jpeg), "sample_test.jpg")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"AI-Generated" in resp.data

    hist_resp = client.get("/history")
    assert hist_resp.status_code == 200
    assert b"sample_test.jpg" in hist_resp.data
