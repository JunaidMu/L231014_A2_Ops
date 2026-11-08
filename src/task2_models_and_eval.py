"""
Task 2 (2.1 - 2.6 + Bonus Grad-CAM): Model Development with TensorFlow/Keras
Project: L231014_A2_Ops (VisionGuard)
Author: Junaid Muhammad (23L-1014)

Covers:
- 2.1: Deterministic split lists (train/val/test), tf.data pipeline with resize & train-only augmentation,
       and augmented batch figure.
- 2.2: Baseline CNN built from scratch (Conv2D, BatchNormalization, MaxPooling2D, Dropout, Sigmoid head),
       model summary, and per-layer shape/parameter count table.
- 2.3: Transfer Learning with MobileNetV2 (Stage 1: Frozen base + custom head; Stage 2: Unfreeze last block
       with tiny learning rate), preprocessing_input, trainable layer lists, and two sets of curves.
- 2.4: Training discipline (fixed seeds, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau) and best epoch marked.
- 2.5: Validation threshold selection + Single evaluation on Test set (Accuracy, Precision, Recall, F1, ROC-AUC,
       Confusion Matrix, ROC Curve) for both models.
- 2.6: Error Analysis (grid of 10 misclassified test images with automated visual hypothesis tags).
- Bonus: Grad-CAM heatmaps for 4 correct and 4 wrong predictions.
"""

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Task 2.4: Fix Random Seeds for Reproducibility
# ---------------------------------------------------------------------------
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

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Task 2.1: Create Split Lists & tf.data Input Pipeline
# ---------------------------------------------------------------------------
def prepare_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads the clean, deduplicated images from reports/dataset_audit.csv
    and creates stratified 70% train / 15% val / 15% test split CSVs.
    Also verifies every image can be decoded by TensorFlow's C++ decoder
    (which rejects WebP/GIF files disguised with .jpg/.png extensions).
    Label mapping: RealArt -> 0 (Real), AiArtData -> 1 (AI-Generated).
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    audit_df = pd.read_csv(REPORTS_DIR / "dataset_audit.csv")

    # Filter out corrupt or exact duplicate files
    clean_df = audit_df[(~audit_df["is_corrupt"]) & (~audit_df["is_duplicate"])].copy()

    # Keep only standard JPEG/PNG files that TensorFlow's decode_image supports cleanly
    tf_valid_mask = []
    for rel_path in clean_df["filepath"]:
        abs_p = str(PROJECT_ROOT / rel_path)
        try:
            raw = tf.io.read_file(abs_p)
            _ = tf.io.decode_image(raw, channels=3, expand_animations=False)
            tf_valid_mask.append(True)
        except Exception as exc:
            print(f"[WARN] Skipping file unsupported by TF decoder ({rel_path}): {exc}")
            tf_valid_mask.append(False)

    clean_df = clean_df[tf_valid_mask].copy()
    clean_df["label"] = (clean_df["class"] == "AiArtData").astype(int)
    clean_df["abs_path"] = clean_df["filepath"].apply(lambda p: str(PROJECT_ROOT / p))

    train_df, temp_df = train_test_split(
        clean_df, test_size=0.30, stratify=clean_df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
    )

    train_df[["filepath", "class", "label"]].to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df[["filepath", "class", "label"]].to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df[["filepath", "class", "label"]].to_csv(SPLITS_DIR / "test.csv", index=False)

    print(
        f"[INFO] Splits saved to {SPLITS_DIR}: "
        f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )
    return train_df, val_df, test_df


def get_augmentation_layer() -> tf.keras.Sequential:
    """Random augmentation applied to training data only."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.08, seed=SEED),
            tf.keras.layers.RandomZoom(0.10, seed=SEED),
            tf.keras.layers.RandomContrast(0.10, seed=SEED),
        ],
        name="data_augmentation",
    )


def decode_and_resize(file_path: tf.Tensor, label: tf.Tensor):
    """Reads image bytes, decodes to 3-channel RGB float32 [0, 255], and resizes."""
    raw = tf.io.read_file(file_path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)  # range [0.0, 255.0]
    label = tf.cast(label, tf.float32)
    return img, label


def build_tf_dataset(
    df: pd.DataFrame, is_training: bool = False, augment_layer=None
) -> tf.data.Dataset:
    """
    Task 2.1: Builds a tf.data pipeline that reads split lists, decodes and resizes
    images, applies random augmentation to training data only, batches, and prefetches.
    """
    paths = df["abs_path"].values
    labels = df["label"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()

    if is_training and augment_layer is not None:
        ds = ds.map(
            lambda x, y: (augment_layer(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def save_augmented_batch_figure(train_ds: tf.data.Dataset) -> None:
    """Task 2.1 Evidence: Saves a figure of one augmented training batch (16 images)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    images, labels = next(iter(train_ds))
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()

    for idx in range(10):
        ax = axes[idx]
        img_disp = np.clip(images[idx].numpy() / 255.0, 0.0, 1.0)
        lbl_str = "AI-Generated (1)" if int(labels[idx].numpy()) == 1 else "Real (0)"
        ax.imshow(img_disp)
        ax.set_title(lbl_str, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        f"Task 2.1: One Augmented Training Batch ({IMG_SIZE[0]}x{IMG_SIZE[1]} RGB) | L231014_A2_Ops",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / "task2_1_augmented_batch.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved augmented batch figure to: {out_path}")


# ---------------------------------------------------------------------------
# Task 2.2: Baseline CNN Built From Scratch
# ---------------------------------------------------------------------------
def build_baseline_cnn(learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Task 2.2: Small CNN with Rescaling, Conv2D, BatchNormalization,
    MaxPooling2D, GlobalAveragePooling2D, Dropout, and one Sigmoid output.
    """
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Classifier Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_hidden")(x)
    x = tf.keras.layers.Dropout(0.4, seed=SEED, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="sigmoid_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Baseline_CNN_L231014")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def extract_layer_table(model: tf.keras.Model, csv_name: str) -> pd.DataFrame:
    """Task 2.2 Evidence: Builds table of output shape and parameter count per layer."""
    rows = []
    for layer in model.layers:
        try:
            out_shape = str(layer.output.shape)
        except Exception:
            out_shape = "Multiple"
        rows.append({
            "Layer Name": layer.name,
            "Layer Type": layer.__class__.__name__,
            "Output Shape": out_shape,
            "Param Count": layer.count_params(),
            "Trainable": layer.trainable,
        })
    df_layers = pd.DataFrame(rows)
    out_csv = REPORTS_DIR / csv_name
    df_layers.to_csv(out_csv, index=False)
    return df_layers


# ---------------------------------------------------------------------------
# Task 2.3: Transfer Learning with MobileNetV2 (Stage 1 Frozen -> Stage 2 Fine-Tune)
# ---------------------------------------------------------------------------
def build_transfer_model(
    dropout_rate: float = 0.3, learning_rate: float = 1e-3
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Task 2.3: Loads pretrained MobileNetV2 without its classifier head, freezes it,
    applies MobileNetV2's required [-1, 1] preprocessing, and adds a custom classification head.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Stage 1: Freeze entire backbone

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    # MobileNetV2 expects pixel values scaled to [-1, 1]
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(dropout_rate, seed=SEED, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="sigmoid_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_VisionGuard_L231014")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model, base_model


def unfreeze_last_block(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    unfreeze_layers: int = 20,
    fine_tune_lr: float = 1e-4,
) -> list[str]:
    """
    Task 2.3 Stage 2: Unfreezes the last block (last N layers, keeping BatchNormalization
    frozen for stability) and recompiles with a much smaller learning rate.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    trainable_names = [
        layer.name for layer in base_model.layers if layer.trainable
    ] + [
        layer.name for layer in model.layers if layer.trainable and layer.name != base_model.name
    ]
    return trainable_names


# ---------------------------------------------------------------------------
# Task 2.4: Callbacks & Training Discipline
# ---------------------------------------------------------------------------
def get_callbacks(checkpoint_path: Path, patience: int = 4) -> list:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def plot_training_curves_with_best_epoch(
    history_dict: dict, title: str, save_path: Path
) -> int:
    """Task 2.4 Evidence: Plots loss and accuracy curves with the best epoch marked."""
    val_losses = history_dict["val_loss"]
    best_epoch_idx = int(np.argmin(val_losses))
    best_epoch_num = best_epoch_idx + 1
    epochs = range(1, len(val_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss subplot
    axes[0].plot(epochs, history_dict["loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history_dict["val_loss"], "r-s", label="Val Loss")
    axes[0].axvline(
        best_epoch_num,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Best Epoch ({best_epoch_num})",
    )
    axes[0].scatter(
        [best_epoch_num],
        [val_losses[best_epoch_idx]],
        color="green",
        s=100,
        zorder=5,
    )
    axes[0].set_title("Loss Curve (Binary Cross-Entropy)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy subplot
    axes[1].plot(epochs, history_dict["accuracy"], "b-o", label="Train Accuracy")
    axes[1].plot(epochs, history_dict["val_accuracy"], "r-s", label="Val Accuracy")
    axes[1].axvline(
        best_epoch_num,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Best Epoch ({best_epoch_num})",
    )
    axes[1].scatter(
        [best_epoch_num],
        [history_dict["val_accuracy"][best_epoch_idx]],
        color="green",
        s=100,
        zorder=5,
    )
    axes[1].set_title("Accuracy Curve", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(f"{title} | Best Epoch = {best_epoch_num}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return best_epoch_num


# ---------------------------------------------------------------------------
# Task 2.5: Threshold Selection on Validation Set & Single Test Set Evaluation
# ---------------------------------------------------------------------------
def select_optimal_threshold(model: tf.keras.Model, val_ds: tf.data.Dataset, y_val: np.ndarray) -> float:
    """
    Task 2.5: Sweeps decision thresholds on validation data to maximize F1 score
    while maintaining strong Recall on AI-generated images.
    """
    val_probs = model.predict(val_ds, verbose=0).ravel()
    thresholds = np.linspace(0.25, 0.75, 51)
    best_thresh = 0.50
    best_f1 = -1.0
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(round(t, 2))
    print(f"[INFO] Selected validation threshold = {best_thresh:.2f} (Val F1 = {best_f1:.4f})")
    return best_thresh


def evaluate_on_test(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    y_test: np.ndarray,
    threshold: float,
    model_label: str,
) -> dict:
    """Task 2.5: Evaluates model once on test set with chosen validation threshold."""
    probs = model.predict(test_ds, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, probs)

    return {
        "model": model_label,
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "probs": probs,
        "preds": preds,
    }


def plot_evaluation_figures(eval_cnn: dict, eval_tl: dict) -> None:
    """Task 2.5 Evidence: Plots Confusion Matrices and ROC Curves for both models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Baseline CNN Confusion Matrix
    sns.heatmap(
        eval_cnn["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real (0)", "AI (1)"],
        yticklabels=["Real (0)", "AI (1)"],
        ax=axes[0],
    )
    axes[0].set_title(
        f"Baseline CNN Confusion Matrix\n(Thresh={eval_cnn['threshold']})",
        fontweight="bold",
    )
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # 2. Transfer Learning Confusion Matrix
    sns.heatmap(
        eval_tl["cm"],
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Real (0)", "AI (1)"],
        yticklabels=["Real (0)", "AI (1)"],
        ax=axes[1],
    )
    axes[1].set_title(
        f"MobileNetV2 (Fine-Tuned) Confusion Matrix\n(Thresh={eval_tl['threshold']})",
        fontweight="bold",
    )
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    # 3. Combined ROC Curve
    axes[2].plot(
        eval_cnn["fpr"],
        eval_cnn["tpr"],
        label=f"Baseline CNN (AUC = {eval_cnn['roc_auc']:.3f})",
        linewidth=2,
    )
    axes[2].plot(
        eval_tl["fpr"],
        eval_tl["tpr"],
        label=f"MobileNetV2 Fine-Tuned (AUC = {eval_tl['roc_auc']:.3f})",
        linewidth=2,
    )
    axes[2].plot([0, 1], [0, 1], "k--", label="Random Chance (0.500)")
    axes[2].set_title("Test Set ROC Curve Comparison", fontweight="bold")
    axes[2].set_xlabel("False Positive Rate (Real called AI)")
    axes[2].set_ylabel("True Positive Rate (AI Recall)")
    axes[2].legend(loc="lower right")
    axes[2].grid(True)

    fig.suptitle(
        "Task 2.5: Test Set Evaluation — Baseline CNN vs MobileNetV2 Transfer Learning (L231014_A2_Ops)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / "task2_5_evaluation_cm_roc.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved evaluation figures to: {out_path}")


# ---------------------------------------------------------------------------
# Task 2.6: Error Analysis (10 Misclassified Test Images + Hypotheses)
# ---------------------------------------------------------------------------
def diagnose_image_characteristics(row: pd.Series) -> str:
    """Generates a specific visual hypothesis based on resolution, aspect ratio, and color stats."""
    w, h = row["width"], row["height"]
    ratio = max(w / h, h / w)
    if min(w, h) < 350:
        return "Low native resolution / compression artifacts lost during 160x160 upscaling"
    if ratio > 1.75:
        return f"Extreme aspect ratio ({int(w)}x{int(h)}) squashed during non-aspect-preserving resize"
    if row["class"] == "RealArt":
        return "Stylized digital illustration / hyper-smooth lighting in RealArt mimics AI synthesis"
    return "Photorealistic AI texture & natural lighting closely mimics authentic photography"


def run_error_analysis(test_df: pd.DataFrame, eval_tl: dict) -> None:
    """Task 2.6: Saves a 2x5 grid of 10 misclassified test images and a CSV table of hypotheses."""
    df_err = test_df.copy()
    df_err["prob_ai"] = eval_tl["probs"]
    df_err["pred"] = eval_tl["preds"]
    misclassified = df_err[df_err["label"] != df_err["pred"]].copy()

    # Sort by highest confidence errors
    misclassified["error_margin"] = np.abs(misclassified["prob_ai"] - misclassified["label"])
    top10_err = misclassified.sort_values("error_margin", ascending=False).head(10).copy()

    fig, axes = plt.subplots(2, 5, figsize=(16, 7.5))
    axes = axes.flatten()
    err_records = []

    for idx, (_, row) in enumerate(top10_err.iterrows()):
        ax = axes[idx]
        img_path = Path(row["abs_path"])
        with Image.open(img_path) as img:
            ax.imshow(img.convert("RGB"))
        true_lbl = "AI" if row["label"] == 1 else "Real"
        pred_lbl = "AI" if row["pred"] == 1 else "Real"
        hypothesis = diagnose_image_characteristics(row)

        ax.set_title(
            f"#{idx + 1}: True={true_lbl} | Pred={pred_lbl}\n"
            f"P(AI)={row['prob_ai']:.2f} | {int(row['width'])}x{int(row['height'])}",
            fontsize=8.5,
            fontweight="bold",
        )
        ax.axis("off")

        err_records.append({
            "Error #": idx + 1,
            "File": Path(row["filepath"]).name,
            "True Class": true_lbl,
            "Predicted Class": pred_lbl,
            "P(AI)": round(float(row["prob_ai"]), 3),
            "Original Size": f"{int(row['width'])}x{int(row['height'])}",
            "Hypothesis": hypothesis,
        })

    fig.suptitle(
        "Task 2.6: 10 Misclassified Test Images (MobileNetV2 - L231014_A2_Ops)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_fig = FIGURES_DIR / "task2_6_misclassified_grid.png"
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)

    err_df = pd.DataFrame(err_records)
    err_df.to_csv(REPORTS_DIR / "task2_6_error_analysis.csv", index=False)
    print(f"[INFO] Saved 10 misclassified images grid to: {out_fig}")


# ---------------------------------------------------------------------------
# Task 2 Bonus: Grad-CAM Heatmaps (4 Correct + 4 Wrong Predictions)
# ---------------------------------------------------------------------------
def compute_gradcam_heatmap(
    img_tensor: tf.Tensor, model: tf.keras.Model, base_model_name: str = "mobilenetv2_1.00_160"
) -> np.ndarray:
    """Computes Grad-CAM heatmap from the final convolutional feature map of MobileNetV2."""
    base_model = model.get_layer(base_model_name)
    last_conv_layer = base_model.get_layer("out_relu")

    # Build sub-model mapping base_model input -> [last_conv_output, base_model_output]
    conv_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[last_conv_layer.output, base_model.output],
    )

    preprocess_layer = model.get_layer("mobilenetv2_preprocess")
    gap_layer = model.get_layer("gap")
    bn_layer = model.get_layer("head_bn")
    drop_layer = model.get_layer("head_dropout")
    out_layer = model.get_layer("sigmoid_output")

    with tf.GradientTape() as tape:
        x = preprocess_layer(img_tensor)
        conv_outputs, _ = conv_model(x, training=False)
        tape.watch(conv_outputs)
        h = gap_layer(conv_outputs)
        h = bn_layer(h, training=False)
        h = drop_layer(h, training=False)
        preds = out_layer(h)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def generate_gradcam_bonus(
    model: tf.keras.Model, base_model: tf.keras.Model, test_df: pd.DataFrame, eval_tl: dict
) -> None:
    """Bonus (+1 mark): Generates Grad-CAM heatmaps for 4 correct and 4 wrong predictions."""
    df_res = test_df.copy()
    df_res["prob_ai"] = eval_tl["probs"]
    df_res["pred"] = eval_tl["preds"]

    correct_samples = df_res[df_res["label"] == df_res["pred"]].head(4)
    wrong_samples = df_res[df_res["label"] != df_res["pred"]].head(4)
    combined = pd.concat([correct_samples, wrong_samples], ignore_index=True)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(combined.iterrows()):
        img_tensor, _ = decode_and_resize(tf.constant(row["abs_path"]), tf.constant(row["label"]))
        img_batch = tf.expand_dims(img_tensor, axis=0)
        heatmap = compute_gradcam_heatmap(img_batch, model, base_model.name)

        # Resize heatmap to 160x160 and overlay
        heatmap_resized = np.uint8(255 * heatmap)
        jet = plt.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_resized]
        jet_heatmap = tf.image.resize(jet_heatmap, IMG_SIZE).numpy()

        orig_rgb = np.clip(img_tensor.numpy() / 255.0, 0.0, 1.0)
        superimposed = np.clip(0.60 * orig_rgb + 0.40 * jet_heatmap, 0.0, 1.0)

        status = "CORRECT" if row["label"] == row["pred"] else "WRONG"
        true_lbl = "AI" if row["label"] == 1 else "Real"
        pred_lbl = "AI" if row["pred"] == 1 else "Real"

        axes[idx].imshow(superimposed)
        axes[idx].set_title(
            f"[{status}] True: {true_lbl} | Pred: {pred_lbl}\nP(AI)={row['prob_ai']:.2f}",
            fontsize=9.5,
            fontweight="bold",
        )
        axes[idx].axis("off")

    fig.suptitle(
        "Task 2 Bonus: Grad-CAM Heatmaps (Top Row: 4 Correct | Bottom Row: 4 Wrong) | L231014_A2_Ops",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / "task2_bonus_gradcam.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved Bonus Grad-CAM heatmaps to: {out_path}")


# ---------------------------------------------------------------------------
# Main Execution for Task 2.1 - 2.6 + Bonus
# ---------------------------------------------------------------------------
def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 2.1 Split lists & tf.data pipelines
    train_df, val_df, test_df = prepare_splits()
    aug_layer = get_augmentation_layer()
    train_ds = build_tf_dataset(train_df, is_training=True, augment_layer=aug_layer)
    val_ds = build_tf_dataset(val_df, is_training=False)
    test_ds = build_tf_dataset(test_df, is_training=False)

    save_augmented_batch_figure(train_ds)

    # 2.2 Baseline CNN
    print("\n" + "=" * 85)
    print("TASK 2.2: BASELINE CNN ARCHITECTURE & LAYER TABLE (L231014_A2_Ops)")
    print("=" * 85)
    baseline_cnn = build_baseline_cnn(learning_rate=1e-3)
    baseline_cnn.summary()
    cnn_layers_df = extract_layer_table(baseline_cnn, "task2_2_baseline_cnn_layers.csv")
    print("\nPer-Layer Output Shape and Parameter Count Table:")
    print(cnn_layers_df.to_string(index=False))

    print("\n[INFO] Training Baseline CNN...")
    cnn_ckpt = MODELS_DIR / "baseline_cnn_best.keras"
    cnn_hist = baseline_cnn.fit(
        train_ds,
        validation_data=val_ds,
        epochs=12,
        callbacks=get_callbacks(cnn_ckpt, patience=4),
        verbose=1,
    )
    cnn_best_epoch = plot_training_curves_with_best_epoch(
        cnn_hist.history,
        "Task 2.2 & 2.4: Baseline CNN Training Curves (L231014_A2_Ops)",
        FIGURES_DIR / "task2_2_baseline_cnn_curves.png",
    )
    print(f"[INFO] Baseline CNN Best Epoch = {cnn_best_epoch}")

    # 2.3 Transfer Learning (Stage 1: Frozen Base)
    print("\n" + "=" * 85)
    print("TASK 2.3: TRANSFER LEARNING (MobileNetV2) - STAGE 1 (FROZEN BACKBONE)")
    print("=" * 85)
    tl_model, base_model = build_transfer_model(dropout_rate=0.3, learning_rate=1e-3)
    stage1_trainable = [layer.name for layer in tl_model.layers if layer.trainable]
    print(f"Stage 1 Trainable Layers ({len(stage1_trainable)}): {stage1_trainable}")

    tl_s1_ckpt = MODELS_DIR / "mobilenetv2_stage1_best.keras"
    tl_s1_hist = tl_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        callbacks=get_callbacks(tl_s1_ckpt, patience=3),
        verbose=1,
    )
    s1_best_epoch = plot_training_curves_with_best_epoch(
        tl_s1_hist.history,
        "Task 2.3 Stage 1: MobileNetV2 Frozen Backbone Curves (L231014_A2_Ops)",
        FIGURES_DIR / "task2_3_transfer_stage1_curves.png",
    )

    # 2.3 Transfer Learning (Stage 2: Unfreeze Last Block)
    print("\n" + "=" * 85)
    print("TASK 2.3: TRANSFER LEARNING (MobileNetV2) - STAGE 2 (FINE-TUNING LAST BLOCK)")
    print("=" * 85)
    stage2_trainable = unfreeze_last_block(
        tl_model, base_model, unfreeze_layers=20, fine_tune_lr=1e-4
    )
    print(f"Stage 2 Trainable Layers ({len(stage2_trainable)}): {stage2_trainable}")

    tl_s2_ckpt = MODELS_DIR / "mobilenetv2_finetuned_best.keras"
    tl_s2_hist = tl_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        callbacks=get_callbacks(tl_s2_ckpt, patience=3),
        verbose=1,
    )
    s2_best_epoch = plot_training_curves_with_best_epoch(
        tl_s2_hist.history,
        "Task 2.3 & 2.4 Stage 2: MobileNetV2 Fine-Tuned Curves (L231014_A2_Ops)",
        FIGURES_DIR / "task2_3_transfer_stage2_curves.png",
    )

    # Save trainable layers metadata
    with open(REPORTS_DIR / "task2_3_trainable_layers.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage1_trainable_layers": stage1_trainable,
                "stage2_trainable_layers": stage2_trainable,
                "cnn_best_epoch": cnn_best_epoch,
                "stage1_best_epoch": s1_best_epoch,
                "stage2_best_epoch": s2_best_epoch,
            },
            f,
            indent=2,
        )

    # 2.5 Validation Threshold & Single Test Evaluation
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    print("\n" + "=" * 85)
    print("TASK 2.5: TEST SET EVALUATION (SINGLE RUN ON TEST SET)")
    print("=" * 85)
    cnn_thresh = select_optimal_threshold(baseline_cnn, val_ds, y_val)
    tl_thresh = select_optimal_threshold(tl_model, val_ds, y_val)

    eval_cnn = evaluate_on_test(baseline_cnn, test_ds, y_test, cnn_thresh, "Baseline CNN (Scratch)")
    eval_tl = evaluate_on_test(tl_model, test_ds, y_test, tl_thresh, "MobileNetV2 (Fine-Tuned)")

    metrics_df = pd.DataFrame([
        {
            "Model": r["model"],
            "Val Threshold": r["threshold"],
            "Test Accuracy": r["accuracy"],
            "Test Precision": r["precision"],
            "Test Recall": r["recall"],
            "Test F1-Score": r["f1"],
            "Test ROC-AUC": r["roc_auc"],
        }
        for r in (eval_cnn, eval_tl)
    ])
    metrics_df.to_csv(REPORTS_DIR / "task2_5_test_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))
    print("=" * 85)

    plot_evaluation_figures(eval_cnn, eval_tl)

    # 2.6 Error Analysis
    run_error_analysis(test_df, eval_tl)

    # Bonus Grad-CAM
    generate_gradcam_bonus(tl_model, base_model, test_df, eval_tl)


if __name__ == "__main__":
    main()
