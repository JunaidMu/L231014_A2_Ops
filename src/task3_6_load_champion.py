"""
Task 3.6: Separate Small Script that Loads the Model by Alias ('champion')
from MLflow Model Registry and Classifies One Image.
Project: L231014_A2_Ops (VisionGuard)
Author: Junaid Muhammad (23L-1014)
"""

import datetime
import getpass
import os
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow_local.db').as_posix()}")

    model_uri = "models:/VisionGuard_L231014@champion"
    print("=" * 80)
    print("TASK 3.6: LOAD MODEL BY ALIAS '@champion' AND CLASSIFY ONE IMAGE")
    print(f"User: {getpass.getuser()} | Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tracking URI : {mlflow.get_tracking_uri()}")
    print(f"Model URI    : {model_uri}")
    print("=" * 80)

    champion_model = mlflow.keras.load_model(model_uri)

    # Pick one test image from data/splits/test.csv
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "test.csv")
    sample_row = test_df.iloc[0]
    img_path = PROJECT_ROOT / sample_row["filepath"]

    raw = tf.io.read_file(str(img_path))
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img_resized = tf.image.resize(img, (160, 160))
    img_batch = tf.expand_dims(tf.cast(img_resized, tf.float32), axis=0)

    prob_ai = float(champion_model.predict(img_batch, verbose=0)[0][0])
    threshold = 0.48
    pred_label = "AI-Generated (AiArtData)" if prob_ai >= threshold else "Real (RealArt)"
    confidence = prob_ai if prob_ai >= threshold else (1.0 - prob_ai)

    print(f"Sample Image : {sample_row['filepath']}")
    print(f"Ground Truth : {sample_row['class']} (label={sample_row['label']})")
    print(f"Raw P(AI)    : {prob_ai:.4f} (Decision Threshold = {threshold})")
    print(f"Prediction   : {pred_label} | Confidence = {confidence * 100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
