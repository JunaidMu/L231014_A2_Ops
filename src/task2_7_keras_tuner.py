"""
Task 2.7: Tool Lab — Keras Tuner (Toy Fashion-MNIST + VisionGuard Hyperparameter Search)
Project: L231014_A2_Ops (VisionGuard)
Author: Junaid Muhammad (23L-1014)

Covers:
1. Toy part: Tunes a tiny network on Fashion-MNIST searching learning rate and number of units
   with RandomSearch, and outputs the best settings & trial table.
2. Apply part: Tunes learning rate, dropout, and number of unfrozen layers for VisionGuard
   (MobileNetV2) with 8 trials, and outputs the full trial table.
"""

import os
import random
from pathlib import Path

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf

# TensorFlow 2.21+ separates tensorboard; disable KerasTuner's optional hparams hook if tensorboard is absent
kt.engine.tuner.Tuner._configure_tensorboard_dir = lambda self, *args, **kwargs: None

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
KT_DIR = PROJECT_ROOT / "kt_dir"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Part 1: Toy Lab on Fashion-MNIST
# ---------------------------------------------------------------------------
def build_toy_model(hp: kt.HyperParameters) -> tf.keras.Model:
    units = hp.Int("units", min_value=32, max_value=128, step=32)
    lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 5e-4, 1e-4])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Rescaling(1.0 / 255.0),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=units, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_toy_tuner() -> pd.DataFrame:
    print("\n" + "=" * 85)
    print("TASK 2.7 (TOY PART): KERAS TUNER ON FASHION-MNIST (L231014_A2_Ops)")
    print("=" * 85)
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
    # Use a fast subset of 4,000 train / 1,000 val samples for rapid toy tuning
    x_train_sub, y_train_sub = x_train[:4000], y_train[:4000]
    x_val_sub, y_val_sub = x_val[:1000], y_val[:1000]

    tuner = kt.RandomSearch(
        build_toy_model,
        objective="val_accuracy",
        max_trials=6,
        seed=SEED,
        directory=str(KT_DIR),
        project_name="toy_fashion_mnist",
        overwrite=True,
    )

    tuner.search(
        x_train_sub,
        y_train_sub,
        epochs=4,
        batch_size=64,
        validation_data=(x_val_sub, y_val_sub),
        verbose=1,
    )

    rows = []
    for trial in tuner.oracle.get_best_trials(num_trials=6):
        rows.append({
            "Trial ID": trial.trial_id,
            "Units": trial.hyperparameters.get("units"),
            "Learning Rate": trial.hyperparameters.get("learning_rate"),
            "Best Val Accuracy": round(float(trial.score), 4),
        })
    df_toy = pd.DataFrame(rows)
    df_toy.to_csv(REPORTS_DIR / "task2_7_toy_tuner_results.csv", index=False)

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("\n--- Toy Tuner Results Table (Fashion-MNIST) ---")
    print(df_toy.to_string(index=False))
    print(
        f"Best Toy Settings -> units={best_hp.get('units')}, "
        f"learning_rate={best_hp.get('learning_rate')}"
    )
    return df_toy


# ---------------------------------------------------------------------------
# Part 2: Apply Keras Tuner to VisionGuard (8 Trials)
# ---------------------------------------------------------------------------
def load_visionguard_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads train and val splits into in-memory float32 arrays once for fast tuning across 8 trials."""
    from task2_models_and_eval import build_tf_dataset

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    train_df["abs_path"] = train_df["filepath"].apply(lambda p: str(PROJECT_ROOT / p))
    val_df["abs_path"] = val_df["filepath"].apply(lambda p: str(PROJECT_ROOT / p))

    train_ds = build_tf_dataset(train_df, is_training=False)
    val_ds = build_tf_dataset(val_df, is_training=False)

    x_train = np.concatenate([x.numpy() for x, _ in train_ds], axis=0)
    y_train = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
    x_val = np.concatenate([x.numpy() for x, _ in val_ds], axis=0)
    y_val = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
    return x_train, y_train, x_val, y_val


def build_visionguard_tunable(hp: kt.HyperParameters) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    lr = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4, 5e-5])
    dropout = hp.Choice("dropout", values=[0.2, 0.3, 0.5])
    unfrozen_layers = hp.Choice("unfrozen_layers", values=[0, 10, 20, 30])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    if unfrozen_layers == 0:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-unfrozen_layers]:
            layer.trainable = False
        for layer in base_model.layers[-unfrozen_layers:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout, seed=SEED)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def run_visionguard_tuner() -> pd.DataFrame:
    print("\n" + "=" * 85)
    print("TASK 2.7 (APPLY PART): KERAS TUNER FOR VISIONGUARD (8 TRIALS | L231014_A2_Ops)")
    print("=" * 85)
    x_train, y_train, x_val, y_val = load_visionguard_arrays()

    tuner = kt.RandomSearch(
        build_visionguard_tunable,
        objective="val_accuracy",
        max_trials=8,
        seed=SEED,
        directory=str(KT_DIR),
        project_name="visionguard_tuning",
        overwrite=True,
    )

    tuner.search(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        epochs=3,
        verbose=1,
    )

    rows = []
    for trial in tuner.oracle.get_best_trials(num_trials=8):
        rows.append({
            "Trial ID": trial.trial_id,
            "Learning Rate": trial.hyperparameters.get("learning_rate"),
            "Dropout": trial.hyperparameters.get("dropout"),
            "Unfrozen Layers": trial.hyperparameters.get("unfrozen_layers"),
            "Best Val Accuracy": round(float(trial.score), 4),
        })

    df_vg = pd.DataFrame(rows)
    df_vg.to_csv(REPORTS_DIR / "task2_7_visionguard_tuner_results.csv", index=False)

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("\n--- VisionGuard Keras Tuner Results Table (8 Trials) ---")
    print(df_vg.to_string(index=False))
    print(
        f"Best VisionGuard Settings -> learning_rate={best_hp.get('learning_rate')}, "
        f"dropout={best_hp.get('dropout')}, unfrozen_layers={best_hp.get('unfrozen_layers')}"
    )
    print("=" * 85)
    return df_vg


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_toy_tuner()
    run_visionguard_tuner()
