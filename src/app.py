"""
Task 4 (4.1 - 4.4), Task 5 (Database Trail), Task 7 & Task 8 (Cloud API + Integration):
VisionGuard Flask Web Application & REST API
Project: L231014_A2_Ops (VisionGuard)
Author: Junaid Muhammad (23L-1014)

Features:
- Loads all configuration (MODEL_PATH, DECISION_THRESHOLD, DATABASE_URL, MAX_UPLOAD_MB,
  CLOUD_API_URL) strictly from environment variables (Task 4.4).
- Loads the champion Keras model ONCE at startup via src.inference.get_predictor() (Task 4.1).
- Routes (Task 4.2 & Task 7):
    GET  /          -> Upload page
    POST /predict   -> Web form submission (sanitizes filename, validates extension, size limit,
                       and image decoding in memory without saving uploaded image to disk, logs
                       result in SQLite/PostgreSQL database, and renders result or friendly error page).
    POST /api/predict -> JSON REST API endpoint for Postman / Cloud / Flask integration (Task 7 & 8).
    GET  /history   -> Queries database for all past predictions and renders history table (Task 4.3 & 5).
    GET  /health    -> Health-check JSON endpoint reporting status, model readiness, and DB connectivity (Task 4.2 & 4.4).
- Runs with debug=False using Waitress/Flask production configuration (Task 4.4).
"""

import base64
import datetime
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from inference import (  # noqa: E402
    ALLOWED_EXTENSIONS,
    CorruptOrInvalidImageError,
    get_predictor,
)

# ---------------------------------------------------------------------------
# Task 4.4: Read ALL settings from environment variables
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv(
    "MODEL_PATH", str(PROJECT_ROOT / "models" / "mobilenetv2_finetuned_best.keras")
)
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.48"))
DEFAULT_SQLITE_URI = f"sqlite:///{(PROJECT_ROOT / 'visionguard_history.db').as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URI)
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "2.0"))  # 2 MB limit for clear validation testing
USE_CLOUD_API = os.getenv("USE_CLOUD_API", "false").lower() in ("1", "true", "yes")
CLOUD_API_URL = os.getenv("CLOUD_API_URL", "").strip()
PORT = int(os.getenv("PORT", "8000"))

db = SQLAlchemy()


# ---------------------------------------------------------------------------
# Task 5: Database Trail Model (SQLite / PostgreSQL via SQLAlchemy)
# ---------------------------------------------------------------------------
class PredictionRecord(db.Model):
    """
    Task 5: Permanent, queryable record that answers:
    what was decided (verdict), when (timestamp_utc), by which model (model_name),
    and how confidently (confidence_pct, prob_ai).
    """

    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp_utc = db.Column(
        db.String(32),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )
    filename = db.Column(db.String(255), nullable=False)
    image_dimensions = db.Column(db.String(64), nullable=False)
    verdict = db.Column(db.String(64), nullable=False)
    confidence_pct = db.Column(db.Float, nullable=False)
    prob_ai = db.Column(db.Float, nullable=False)
    threshold = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(128), nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp_utc": self.timestamp_utc,
            "filename": self.filename,
            "image_dimensions": self.image_dimensions,
            "verdict": self.verdict,
            "confidence_pct": self.confidence_pct,
            "prob_ai": self.prob_ai,
            "threshold": self.threshold,
            "model_name": self.model_name,
        }


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )

    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)
    app.config["MODEL_PATH"] = MODEL_PATH
    app.config["DECISION_THRESHOLD"] = DECISION_THRESHOLD
    app.config["MAX_UPLOAD_MB"] = MAX_UPLOAD_MB

    if test_config:
        app.config.update(test_config)

    db.init_app(app)

    with app.app_context():
        db.create_all()
        # Load the champion model ONCE at startup unless skipped in lightweight unit test mocks
        if not app.config.get("SKIP_MODEL_LOAD", False):
            get_predictor()

    # -----------------------------------------------------------------------
    # Task 4.2: Friendly Error Handler for Oversized Files (HTTP 413)
    # -----------------------------------------------------------------------
    @app.errorhandler(RequestEntityTooLarge)
    @app.errorhandler(413)
    def handle_file_too_large(_exc):
        msg = (
            f"File Rejected (Size Limit Exceeded): The uploaded file exceeds the maximum "
            f"allowed size limit of {app.config['MAX_UPLOAD_MB']:.1f} MB."
        )
        if request.path.startswith("/api/"):
            return jsonify({"error": "FileTooLarge", "message": msg}), 413
        return (
            render_template(
                "error.html",
                error_title="413 — File Over Size Limit",
                error_type="Oversized Upload Rejected",
                error_message=msg,
                suggestion=f"Please upload an image smaller than {app.config['MAX_UPLOAD_MB']:.1f} MB.",
            ),
            413,
        )

    # -----------------------------------------------------------------------
    # Helper: Validate & Process Upload in Memory (Never Saves to Disk!)
    # -----------------------------------------------------------------------
    def _validate_and_run_inference(file_storage):
        if file_storage is None or not file_storage.filename:
            return None, (
                "No File Selected",
                "Missing Upload",
                "Please select an image file before clicking Analyze.",
                400,
            )

        # Task 4.2: Sanitize file name
        safe_name = secure_filename(file_storage.filename) or "unnamed_file"
        ext = Path(safe_name).suffix.lower()

        # Failure Case 1: Reject wrong file types (.txt, .pdf, .exe, .csv, etc.)
        if ext not in ALLOWED_EXTENSIONS:
            return None, (
                "400 — Unsupported File Type Rejected",
                f"Invalid Extension '{ext or '(none)'}' (Sanitized Filename: {safe_name})",
                (
                    f"VisionGuard rejected '{safe_name}' because '{ext or 'no extension'}' is not "
                    f"an allowed image type. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
                ),
                400,
            )

        # Read bytes directly into memory (Task 4.2: Do NOT keep the uploaded image on disk)
        raw_bytes = file_storage.read()
        max_bytes = int(app.config["MAX_UPLOAD_MB"] * 1024 * 1024)

        # Failure Case 2: Reject files over the size limit
        if len(raw_bytes) > max_bytes:
            size_mb = len(raw_bytes) / (1024.0 * 1024.0)
            return None, (
                "413 — File Over Size Limit Rejected",
                f"File Size {size_mb:.2f} MB > Limit {app.config['MAX_UPLOAD_MB']:.1f} MB (Sanitized Filename: {safe_name})",
                (
                    f"VisionGuard rejected '{safe_name}' ({size_mb:.2f} MB) because it exceeds "
                    f"the configured {app.config['MAX_UPLOAD_MB']:.1f} MB upload limit."
                ),
                413,
            )

        # Failure Case 3: Reject files that look like images by extension (.jpg/.png) but cannot be decoded
        try:
            predictor = get_predictor()
            pred_result = predictor.predict_bytes(
                raw_bytes, filename=safe_name, use_cloud_api=USE_CLOUD_API
            )
        except CorruptOrInvalidImageError as exc:
            return None, (
                "422 — Undecodable / Corrupt Image Rejected",
                f"Corrupt Binary Payload (Sanitized Filename: {safe_name})",
                str(exc),
                422,
            )

        # Log every valid prediction permanently in the Database (Task 5)
        dims_str = f"{pred_result['meta']['width']}x{pred_result['meta']['height']} ({pred_result['meta']['format']})"
        record = PredictionRecord(
            filename=safe_name,
            image_dimensions=dims_str,
            verdict=pred_result["verdict"],
            confidence_pct=pred_result["confidence_pct"],
            prob_ai=pred_result["prob_ai"],
            threshold=pred_result["threshold"],
            model_name=pred_result["model_name"],
        )
        db.session.add(record)
        db.session.commit()

        # Encode in-memory preview as base64 data URI so the browser displays the image without saving to disk
        mime = "image/png" if ext == ".png" else "image/jpeg"
        b64_preview = f"data:{mime};base64,{base64.b64encode(raw_bytes).decode('ascii')}"

        pred_result["filename"] = safe_name
        pred_result["record_id"] = record.id
        pred_result["timestamp_utc"] = record.timestamp_utc
        pred_result["preview_data_uri"] = b64_preview
        return pred_result, None

    # -----------------------------------------------------------------------
    # Routes (Task 4.2, 4.3, 4.4, Task 5, Task 7, Task 8)
    # -----------------------------------------------------------------------
    @app.route("/", methods=["GET"])
    def index():
        recent_count = PredictionRecord.query.count()
        return render_template(
            "index.html",
            max_upload_mb=app.config["MAX_UPLOAD_MB"],
            threshold=DECISION_THRESHOLD,
            model_name="VisionGuard_MobileNetV2_v2@champion",
            cloud_api_url=CLOUD_API_URL if USE_CLOUD_API else None,
            recent_count=recent_count,
        )

    @app.route("/predict", methods=["POST"])
    def predict_view():
        file_obj = request.files.get("image") or request.files.get("file")
        result, err = _validate_and_run_inference(file_obj)
        if err is not None:
            title, err_type, msg, status_code = err
            return (
                render_template(
                    "error.html",
                    error_title=title,
                    error_type=err_type,
                    error_message=msg,
                    suggestion="Please upload a valid, non-corrupt JPEG/PNG image within the size limit.",
                ),
                status_code,
            )
        return render_template("result.html", result=result)

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """Task 7 & Task 8: JSON API endpoint for Postman and Cloud/Flask integration."""
        file_obj = request.files.get("file") or request.files.get("image")
        result, err = _validate_and_run_inference(file_obj)
        if err is not None:
            title, err_type, msg, status_code = err
            return (
                jsonify({
                    "status": "error",
                    "error_title": title,
                    "error_type": err_type,
                    "message": msg,
                    "project": "L231014_A2_Ops",
                }),
                status_code,
            )
        # Exclude base64 preview string from JSON API output to keep response concise
        api_payload = {k: v for k, v in result.items() if k != "preview_data_uri"}
        api_payload["status"] = "success"
        api_payload["project"] = "L231014_A2_Ops"
        return jsonify(api_payload), 200

    @app.route("/history", methods=["GET"])
    def history_view():
        """Task 4.3 & Task 5: Displays all logged searches from the database."""
        records = PredictionRecord.query.order_by(PredictionRecord.id.desc()).all()
        return render_template("history.html", records=records)

    @app.route("/health", methods=["GET"])
    def health_check():
        """Task 4.2 & Task 4.4: Health check endpoint verifying model and database status."""
        db_ok = False
        total_logged = 0
        try:
            db.session.execute(text("SELECT 1"))
            total_logged = PredictionRecord.query.count()
            db_ok = True
        except Exception:
            db_ok = False

        model_loaded = (
            True
            if app.config.get("SKIP_MODEL_LOAD")
            else (get_predictor()._model is not None)
        )
        status_code = 200 if (db_ok and model_loaded) else 503

        return (
            jsonify({
                "status": "healthy" if status_code == 200 else "degraded",
                "service": "VisionGuard Web & Model API",
                "project": "L231014_A2_Ops",
                "student": "Junaid Muhammad (23L-1014)",
                "debug_mode": app.debug,
                "model_loaded_once": model_loaded,
                "model_name": "VisionGuard_MobileNetV2_v2@champion",
                "model_path": str(app.config["MODEL_PATH"]),
                "decision_threshold": app.config["DECISION_THRESHOLD"],
                "database_connected": db_ok,
                "database_backend": app.config["SQLALCHEMY_DATABASE_URI"].split("://")[0],
                "total_predictions_logged": total_logged,
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                ),
            }),
            status_code,
        )

    return app


app = create_app()

if __name__ == "__main__":
    # Task 4.4: Run the app with debug mode OFF (production-safe configuration)
    print("=" * 80)
    print("TASK 4.4: STARTING VISIONGUARD SERVER (debug=False | L231014_A2_Ops)")
    print(f"Listening on : http://127.0.0.1:{PORT}")
    print(f"Health Check : http://127.0.0.1:{PORT}/health")
    print(f"Database URI : {DATABASE_URL}")
    print("=" * 80)
    app.run(host="0.0.0.0", port=PORT, debug=False)
