"""
Task 3.3, 3.4, 3.5, 3.6: Instrument VisionGuard Training with MLflow on DagsHub
Project: L231014_A2_Ops (VisionGuard)
Author: Junaid Muhammad (23L-1014)

Covers:
- Task 3.3: Logs hyperparameters, per-epoch training & validation metrics, final test metrics,
            and tags (git commit, DVC data version, seed). Executes ONCE manually (with a custom
            Keras callback) and ONCE with MLflow's TensorFlow autologging (mlflow.tensorflow.autolog()).
- Task 3.4: Attaches confusion matrix, ROC curve, split lists, package list (requirements_exact.txt),
            and the trained Keras model as run artifacts.
- Task 3.5: Produces 6 meaningful comparison runs where each run differs from another in ONE
            meaningful change (learning rate, augmentation, frozen layers, image size, backbone).
- Task 3.6: Registers the best model in the MLflow Model Registry ("VisionGuard_L231014"),
            creates at least two versions (v1 and v2), assigns the alias "champion", and verifies it.
"""

import os
import random
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import mlflow.tensorflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def get_dvc_version() -> str:
    dvc_file = PROJECT_ROOT / "data" / "raw.dvc"
    if dvc_file.exists():
        for line in dvc_file.read_text(encoding="utf-8").splitlines():
            if "md5:" in line:
                return line.split("md5:")[-1].strip()
    return "b2cec67839f742df4653f66d4bcbbafb.dir"


class ManualMLflowLoggingCallback(tf.keras.callbacks.Callback):
    """Task 3.3: Custom Keras Callback for manual per-epoch metric logging."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        step = epoch + 1
        for k, v in logs.items():
            mlflow.log_metric(k, float(v), step=step)


def load_split_tensors(img_size: tuple[int, int] = (160, 160), batch_size: int = 32):
    """Builds train, val, and test datasets for the specified image size."""
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    for df in (train_df, val_df, test_df):
        df["abs_path"] = df["filepath"].apply(lambda p: str(PROJECT_ROOT / p))

    def _decode(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, img_size)
        return tf.cast(img, tf.float32), tf.cast(label, tf.float32)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_df["abs_path"].values, train_df["label"].values))
        .shuffle(len(train_df), seed=SEED)
        .map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_df["abs_path"].values, val_df["label"].values))
        .map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_df["abs_path"].values, test_df["label"].values))
        .map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, test_ds, test_df["label"].values


def build_configurable_model(
    backbone_name: str = "MobileNetV2",
    img_size: tuple[int, int] = (160, 160),
    learning_rate: float = 5e-4,
    unfrozen_layers: int = 20,
    use_augmentation: bool = True,
    dropout: float = 0.3,
) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(*img_size, 3), name="input_image")
    x = inputs

    if use_augmentation:
        aug = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal", seed=SEED),
                tf.keras.layers.RandomRotation(0.08, seed=SEED),
                tf.keras.layers.RandomZoom(0.10, seed=SEED),
            ],
            name="augmentation",
        )
        x = aug(x)

    if backbone_name == "MobileNetV2":
        x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="preprocess")(x)
        base = tf.keras.applications.MobileNetV2(
            input_shape=(*img_size, 3), include_top=False, weights="imagenet"
        )
    elif backbone_name == "ResNet50":
        x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="preprocess")(x)
        base = tf.keras.applications.ResNet50(
            input_shape=(*img_size, 3), include_top=False, weights="imagenet"
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    if unfrozen_layers == 0:
        base.trainable = False
    else:
        base.trainable = True
        for layer in base.layers[:-unfrozen_layers]:
            layer.trainable = False
        for layer in base.layers[-unfrozen_layers:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(dropout, seed=SEED, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="sigmoid_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def save_and_log_run_artifacts(
    model: tf.keras.Model,
    y_test: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    run_tag: str,
    log_keras_model: bool = False,
):
    """
    Task 3.4: Attaches confusion matrix, ROC curve, split lists,
    package list (requirements_exact.txt), and trained Keras model to the run.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)

    # 1. Confusion Matrix artifact
    cm_path = FIGURES_DIR / f"cm_{run_tag}.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Real (0)", "AI (1)"],
        yticklabels=["Real (0)", "AI (1)"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix ({run_tag})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")

    # 2. ROC Curve artifact
    roc_path = FIGURES_DIR / f"roc_{run_tag}.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(f"ROC Curve ({run_tag})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(roc_path), artifact_path="evaluation_plots")

    # 3. Split lists
    for split_file in ("train.csv", "val.csv", "test.csv"):
        mlflow.log_artifact(str(SPLITS_DIR / split_file), artifact_path="splits")

    # 4. Package list
    mlflow.log_artifact(str(PROJECT_ROOT / "requirements_exact.txt"), artifact_path="environment")

    # 5. Trained Keras model
    if log_keras_model:
        mlflow.keras.log_model(model, name="visionguard_model")


def execute_experiment_run(
    run_name: str,
    config: dict,
    git_commit: str,
    dvc_version: str,
    use_autolog: bool = False,
    log_full_artifacts: bool = False,
) -> dict:
    img_size = config["img_size"]
    train_ds, val_ds, test_ds, y_test = load_split_tensors(img_size=img_size)
    model = build_configurable_model(
        backbone_name=config["backbone"],
        img_size=img_size,
        learning_rate=config["learning_rate"],
        unfrozen_layers=config["unfrozen_layers"],
        use_augmentation=config["use_augmentation"],
        dropout=config["dropout"],
    )

    if use_autolog:
        mlflow.tensorflow.autolog(log_models=False)
    else:
        mlflow.tensorflow.autolog(disable=True)

    with mlflow.start_run(run_name=run_name) as active_run:
        # Log Tags (Task 3.3: git commit, DVC data version, seed)
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("dvc_data_version", dvc_version)
        mlflow.set_tag("seed", str(SEED))
        mlflow.set_tag("logging_mode", "autolog" if use_autolog else "manual_callback")
        mlflow.set_tag("change_description", config["change_desc"])
        mlflow.set_tag("project", "L231014_A2_Ops")

        # Log Hyperparameters
        mlflow.log_params({
            "backbone": config["backbone"],
            "img_size": f"{img_size[0]}x{img_size[1]}",
            "learning_rate": config["learning_rate"],
            "unfrozen_layers": config["unfrozen_layers"],
            "use_augmentation": config["use_augmentation"],
            "dropout": config["dropout"],
            "epochs": config["epochs"],
            "batch_size": 32,
            "threshold": 0.48,
        })

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ]
        if not use_autolog:
            callbacks.append(ManualMLflowLoggingCallback())

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config["epochs"],
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate on Test Set
        probs = model.predict(test_ds, verbose=0).ravel()
        preds = (probs >= 0.48).astype(int)
        test_metrics = {
            "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "test_precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "test_f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        }
        mlflow.log_metrics(test_metrics)

        if log_full_artifacts:
            save_and_log_run_artifacts(
                model,
                y_test,
                probs,
                preds,
                run_tag=run_name.replace(" ", "_"),
                log_keras_model=config.get("save_model_artifact", False),
            )

        print(f"[DONE] {run_name} -> {test_metrics}")
        return {
            "Run Name": run_name,
            "Run ID": active_run.info.run_id,
            "Meaningful Change": config["change_desc"],
            "Backbone": config["backbone"],
            "Image Size": f"{img_size[0]}x{img_size[1]}",
            "LR": config["learning_rate"],
            "Unfrozen Layers": config["unfrozen_layers"],
            "Augmentation": config["use_augmentation"],
            **test_metrics,
        }


def register_champion_model():
    """
    Task 3.6: Registers our two best trained Keras models from Task 2 into MLflow
    Model Registry ('VisionGuard_L231014') as Version 1 (Stage 1 frozen) and
    Version 2 (Stage 2 fine-tuned), and assigns the alias 'champion' to Version 2.
    """
    model_name = "VisionGuard_L231014"
    client = MlflowClient()

    print("\n" + "=" * 85)
    print(f"TASK 3.6: REGISTERING MODELS IN MLFLOW REGISTRY ({model_name})")
    print("=" * 85)

    # Version 1: Stage 1 Frozen MobileNetV2
    s1_model = tf.keras.models.load_model(MODELS_DIR / "mobilenetv2_stage1_best.keras")
    with mlflow.start_run(run_name="Registry_v1_MobileNetV2_Frozen"):
        mlflow.set_tag("registry_candidate", "v1_baseline_transfer")
        info_v1 = mlflow.keras.log_model(
            s1_model,
            name="visionguard_model",
            registered_model_name=model_name,
        )
        print(f"[INFO] Registered {model_name} Version 1 -> {info_v1.model_uri}")

    # Version 2: Stage 2 Fine-Tuned MobileNetV2 (Best Champion Model)
    s2_model = tf.keras.models.load_model(MODELS_DIR / "mobilenetv2_finetuned_best.keras")
    with mlflow.start_run(run_name="Registry_v2_MobileNetV2_FineTuned_Champion"):
        mlflow.set_tag("registry_candidate", "v2_champion_finetuned")
        # Also attach full Task 3.4 artifacts to this champion run
        _, _, test_ds, y_test = load_split_tensors(img_size=(160, 160))
        probs = s2_model.predict(test_ds, verbose=0).ravel()
        preds = (probs >= 0.48).astype(int)
        mlflow.log_metrics({
            "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "test_precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "test_f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        })
        save_and_log_run_artifacts(
            s2_model, y_test, probs, preds, run_tag="Champion_v2", log_keras_model=False
        )
        info_v2 = mlflow.keras.log_model(
            s2_model,
            name="visionguard_model",
            registered_model_name=model_name,
        )
        print(f"[INFO] Registered {model_name} Version 2 -> {info_v2.model_uri}")

    # Assign alias 'champion' to latest version (v2)
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max(int(v.version) for v in versions)
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=str(latest_version),
    )
    print(f"[SUCCESS] Assigned alias '@champion' to {model_name} version {latest_version}!")


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[INFO] Connected to Remote DagsHub MLflow: {tracking_uri}")
    else:
        local_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow_local.db').as_posix()}"
        mlflow.set_tracking_uri(local_uri)
        print(f"[WARN] .env not found; falling back to local URI: {local_uri}")

    mlflow.set_experiment("L231014_A2_Ops_VisionGuard_Experiments")
    git_commit = get_git_commit()
    dvc_version = get_dvc_version()

    # -----------------------------------------------------------------------
    # Task 3.3 & 3.4 & 3.5: 6 Meaningful Runs + 1 Autolog Comparison Run
    # Each of the 6 runs differs from Run 1 in ONE controlled change!
    # -----------------------------------------------------------------------
    runs_spec = [
        # Run 1: Reference Config (Manual Callback + Full Artifacts for Task 3.4)
        {
            "run_name": "Run1_Ref_MobileNetV2_lr5e4_unfreeze20_augTrue_160px_Manual",
            "backbone": "MobileNetV2",
            "img_size": (160, 160),
            "learning_rate": 5e-4,
            "unfrozen_layers": 20,
            "use_augmentation": True,
            "dropout": 0.3,
            "epochs": 4,
            "change_desc": "Reference Baseline (MobileNetV2, lr=5e-4, unfreeze=20, aug=True, 160x160)",
            "use_autolog": False,
            "log_full_artifacts": True,
            "save_model_artifact": True,
        },
        # Task 3.3 Companion: Exact same as Run 1, but using mlflow.tensorflow.autolog()
        {
            "run_name": "Run1B_Task3_3_Autolog_MobileNetV2",
            "backbone": "MobileNetV2",
            "img_size": (160, 160),
            "learning_rate": 5e-4,
            "unfrozen_layers": 20,
            "use_augmentation": True,
            "dropout": 0.3,
            "epochs": 4,
            "change_desc": "Task 3.3 Autologging comparison (mlflow.tensorflow.autolog)",
            "use_autolog": True,
            "log_full_artifacts": True,
            "save_model_artifact": False,
        },
        # Run 2: Change Learning Rate (5e-4 -> 1e-4)
        {
            "run_name": "Run2_Change_LR_1e4",
            "backbone": "MobileNetV2",
            "img_size": (160, 160),
            "learning_rate": 1e-4,
            "unfrozen_layers": 20,
            "use_augmentation": True,
            "dropout": 0.3,
            "epochs": 4,
            "change_desc": "Changed learning_rate: 5e-4 -> 1e-4",
            "use_autolog": False,
            "log_full_artifacts": False,
        },
        # Run 3: Change Augmentation (True -> False)
        {
            "run_name": "Run3_Change_NoAugmentation",
            "backbone": "MobileNetV2",
            "img_size": (160, 160),
            "learning_rate": 5e-4,
            "unfrozen_layers": 20,
            "use_augmentation": False,
            "dropout": 0.3,
            "epochs": 4,
            "change_desc": "Changed use_augmentation: True -> False",
            "use_autolog": False,
            "log_full_artifacts": False,
        },
        # Run 4: Change Frozen Layers (unfrozen_layers: 20 -> 0 / Fully Frozen)
        {
            "run_name": "Run4_Change_FrozenBackbone_0",
            "backbone": "MobileNetV2",
            "img_size": (160, 160),
            "learning_rate": 5e-4,
            "unfrozen_layers": 0,
            "use_augmentation": True,
            "dropout": 0.3,
            "epochs": 4,
            "change_desc": "Changed unfrozen_layers: 20 -> 0 (Fully Frozen Backbone)",
            "use_autolog": False,
            "log_full_artifacts": False,
        },
        # Run 5: Change Image Size (160x160 -> 128x128)
        {
            "run_name": "Run5_Change_ImageSize_128px",
            "backbone": "MobileNetV2",
            "img_size": (128, 128),
            "learning_rate": 5e-4,
            "unfrozen_layers": 20,
            "use_augmentation": True,
            "dropout": 0.3,
            "epochs": 4,
            "change_desc": "Changed img_size: 160x160 -> 128x128",
            "use_autolog": False,
            "log_full_artifacts": False,
        },
        # Run 6: Change Backbone (MobileNetV2 -> ResNet50)
        {
            "run_name": "Run6_Change_Backbone_ResNet50",
            "backbone": "ResNet50",
            "img_size": (160, 160),
            "learning_rate": 5e-4,
            "unfrozen_layers": 20,
            "use_augmentation": True,
            "dropout": 0.3,
            "epochs": 4,
            "change_desc": "Changed backbone: MobileNetV2 -> ResNet50",
            "use_autolog": False,
            "log_full_artifacts": False,
        },
    ]

    results = []
    for spec in runs_spec:
        res = execute_experiment_run(
            run_name=spec["run_name"],
            config=spec,
            git_commit=git_commit,
            dvc_version=dvc_version,
            use_autolog=spec["use_autolog"],
            log_full_artifacts=spec["log_full_artifacts"],
        )
        results.append(res)

    df_comp = pd.DataFrame(results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(REPORTS_DIR / "task3_5_experiment_comparison.csv", index=False)

    print("\n" + "=" * 95)
    print("TASK 3.5: EXPERIMENT COMPARISON TABLE (6 MEANINGFUL VARIATIONS + AUTOLOG)")
    print("=" * 95)
    print(df_comp.drop(columns=["Run ID"]).to_string(index=False))
    print("=" * 95)

    # Task 3.6: Register v1 and v2 and set alias 'champion'
    register_champion_model()


if __name__ == "__main__":
    main()
