"""
Task 3.1 & Task 3.2: Tool Lab — MLflow Toy Runs (Local SQLite & Remote DagsHub)
Project: L231014_A2_Ops (VisionGuard)
Author: Junaid Muhammad (23L-1014)

Usage:
  1) Local MLflow (Task 3.1):
     python src/task3_1_2_mlflow_toy.py --mode local
  2) Remote DagsHub (Task 3.2 - reads credentials from .env):
     python src/task3_1_2_mlflow_toy.py --mode dagshub
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def train_and_log_toy_run(lr: float, x_train, y_train, x_val, y_val, mode_label: str):
    run_name = f"toy_fashion_mnist_lr_{lr}_{mode_label}"
    with mlflow.start_run(run_name=run_name):
        # Log parameters & tags
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("hidden_units", 64)
        mlflow.log_param("dataset", "Fashion-MNIST (3000 subset)")
        mlflow.set_tag("student", "Junaid Muhammad")
        mlflow.set_tag("roll_number", "23L-1014")
        mlflow.set_tag("project", "L231014_A2_Ops")
        mlflow.set_tag("tracking_target", mode_label)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=5,
            batch_size=64,
            verbose=1,
        )

        # Log per-epoch metrics
        for epoch_idx in range(len(history.history["loss"])):
            step = epoch_idx + 1
            mlflow.log_metric("train_loss", float(history.history["loss"][epoch_idx]), step=step)
            mlflow.log_metric("train_accuracy", float(history.history["accuracy"][epoch_idx]), step=step)
            mlflow.log_metric("val_loss", float(history.history["val_loss"][epoch_idx]), step=step)
            mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][epoch_idx]), step=step)

        # Save and log one plot artifact
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = FIGURES_DIR / f"toy_mlflow_lr_{lr}.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        epochs_range = range(1, 6)
        ax.plot(epochs_range, history.history["accuracy"], "b-o", label="Train Acc")
        ax.plot(epochs_range, history.history["val_accuracy"], "r-s", label="Val Acc")
        ax.set_title(f"Toy Fashion-MNIST Accuracy (lr={lr}) | L231014_A2_Ops")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(plot_path), artifact_path="plots")

        # Log the trained Keras model
        mlflow.keras.log_model(model, name="toy_keras_model")
        print(f"[INFO] Completed MLflow run: {run_name} (Final Val Acc: {history.history['val_accuracy'][-1]:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "dagshub"], default="local")
    args = parser.parse_args()

    if args.mode == "local":
        local_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow_local.db').as_posix()}"
        mlflow.set_tracking_uri(local_uri)
        print(f"[INFO] Task 3.1: Using Local MLflow Tracking URI -> {local_uri}")
    else:
        load_dotenv(PROJECT_ROOT / ".env")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        if not tracking_uri or not username or not password:
            raise ValueError(
                "Missing MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, or "
                "MLFLOW_TRACKING_PASSWORD in .env file!"
            )
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[INFO] Task 3.2: Using Remote DagsHub Tracking URI -> {tracking_uri}")

    mlflow.set_experiment("L231014_A2_Ops_Toy_Lab")

    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, y_train = x_train[:3000], y_train[:3000]
    x_val, y_val = x_val[:1000], y_val[:1000]

    # Compare two runs with different learning rates
    for lr in [0.001, 0.01]:
        train_and_log_toy_run(lr, x_train, y_train, x_val, y_val, args.mode)


if __name__ == "__main__":
    main()
