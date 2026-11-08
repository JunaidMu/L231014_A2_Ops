# 🛡️ VisionGuard (`L231014_A2_Ops`) — AI vs. Real Image Classification & Deployment

![VisionGuard CI](https://github.com/JunaidMu/L231014_A2_Ops/actions/workflows/ci.yml/badge.svg)

**Student Name:** Junaid Muhammad  
**Roll Number:** `23L-1014` (`L231014_A2_Ops`)  
**DagsHub Project & MLflow Registry:** [https://dagshub.com/JunaidMu/L231014_A2_Ops](https://dagshub.com/JunaidMu/L231014_A2_Ops)  
**GitHub Repository:** [https://github.com/JunaidMu/L231014_A2_Ops](https://github.com/JunaidMu/L231014_A2_Ops)

---

## 📌 Project Overview

**VisionGuard** is an end-to-end MLOps web service and REST API that classifies uploaded images as **AI-Generated (`AiArtData`)** or **Real (`RealArt`)**, returning a calibrated confidence score and an honest model caveat while permanently logging every inference in a relational database trail (SQLite / PostgreSQL).

- **Task 1 (Data Understanding & DVC):** SHA-256 duplicate & corruption audit (`src/task1_data_audit_and_eda.py`), visual EDA, and DVC dataset tracking (`data/raw.dvc`).
- **Task 2 (Model Development):** `tf.data` input pipeline with train-only augmentation, scratch Baseline CNN vs. two-stage fine-tuned `MobileNetV2` (`Test Accuracy = 80.82%`, `Test ROC-AUC = 0.8902`), 10-image error analysis, Grad-CAM explainability heatmaps, and Keras Tuner hyperparameter search.
- **Task 3 (MLflow & DagsHub):** Local & remote DagsHub experiment tracking, manual callback vs. TensorFlow autologging, 6 controlled comparison runs, and MLflow Model Registry (`VisionGuard_L231014@champion`).
- **Task 4 & 5 (Flask Web App & Database Trail):** Singleton startup model loader (`src/inference.py`), strict in-memory upload validation (`.jpg/.png/.webp`, `<= 2.0 MB`, corruption verification), production mode (`debug=False`), and SQLAlchemy database trail (`/history`).
- **Task 6 (Docker, Compose & GitHub Actions CI):** Non-root CPU-only `Dockerfile` with layer caching and `/health` probe, two-service `docker-compose.yml` (`web` + unexposed `db` with readiness healthcheck), and automated GitHub Actions CI workflow (`.github/workflows/ci.yml`).
- **Task 7 & 8 (Cloud API Deployment & Flask Integration):** Publicly reachable `/api/predict` endpoint integrated with local Flask UI and verified via Postman.

---

## 🚀 Quick Start (Docker Compose)

```bash
cp .env.example .env
docker compose up --build -d
```

Then open **`http://localhost:8000`** in your browser (`http://localhost:8000/health` for health check, `http://localhost:8000/history` for database trail).
