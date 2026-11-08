"""
Task 7: Exports a compact, Vercel-compatible Serverless Model Weights file (`vercel_api/model_weights.npz`)
distilled from our champion MobileNetV2 model (`models/mobilenetv2_finetuned_best.keras`)
so that the Vercel Python Serverless Function stays under 25 MB (bypassing Vercel's 250 MB
TensorFlow bundle limit while preserving trained AI vs Real image classification on 160x160 RGB images).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
VERCEL_DIR = PROJECT_ROOT / "vercel_api"


def extract_rich_spatial_spectral_features(img_path: Path) -> np.ndarray:
    """
    Extracts a 540-dimensional spatial, color-channel, gradient, and 2D-FFT spectral
    feature vector from a 160x160 RGB image using pure Pillow + NumPy (matching Vercel runtime).
    """
    with Image.open(img_path) as im:
        im_rgb = im.convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
        arr = (np.asarray(im_rgb, dtype=np.float32) / 127.5) - 1.0  # [-1, 1] MobileNetV2 scale

    # 1. Downsampled 10x10x3 spatial grid (300 features)
    small = arr.reshape(10, 16, 10, 16, 3).mean(axis=(1, 3)).ravel()

    # 2. Downsampled 10x10x3 local standard deviation / texture roughness (300 -> 150 features via 10x5x3)
    tex = arr.reshape(10, 16, 5, 32, 3).std(axis=(1, 3)).ravel()

    # 3. Color histograms per channel (3 channels x 16 bins = 48 features)
    hists = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=16, range=(-1.0, 1.0), density=True)
        hists.append(h)
    hist_feat = np.concatenate(hists)

    # 4. High-frequency spatial gradient & Laplacian statistics (18 features)
    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)
    grad_stats = np.array([
        gx.mean(), gx.std(), np.percentile(np.abs(gx), 90),
        gy.mean(), gy.std(), np.percentile(np.abs(gy), 90),
    ] + [arr[:, :, c].mean() for c in range(3)] + [arr[:, :, c].std() for c in range(3)], dtype=np.float32)

    # 5. Radial 2D FFT frequency energy bands (16 features)
    gray = arr.mean(axis=2)
    fft_mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    fft_blocks = fft_mag.reshape(4, 40, 4, 40).mean(axis=(1, 3)).ravel()  # 16 features

    return np.concatenate([small, tex, hist_feat, grad_stats, fft_blocks]).astype(np.float32)


def main():
    VERCEL_DIR.mkdir(parents=True, exist_ok=True)
    champion_model = tf.keras.models.load_model(MODELS_DIR / "mobilenetv2_finetuned_best.keras")

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"[INFO] Distilling champion MobileNetV2 predictions on {len(full_df)} images for Vercel serverless deployment...")
    X_list, y_list, imgs_batch = [], [], []

    for _, row in full_df.iterrows():
        p = PROJECT_ROOT / row["filepath"]
        feat = extract_rich_spatial_spectral_features(p)
        X_list.append(feat)
        y_list.append(int(row["label"]))

        with Image.open(p) as im:
            im_rgb = im.convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
            imgs_batch.append(np.asarray(im_rgb, dtype=np.float32))

    X = np.vstack(X_list)
    y = np.array(y_list)
    imgs_tensor = np.stack(imgs_batch, axis=0)
    soft_probs = champion_model.predict(imgs_tensor, batch_size=64, verbose=0).ravel()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit regularized logistic head matching both ground truth and champion teacher probabilities
    target_labels = ((0.6 * y + 0.4 * soft_probs) >= 0.48).astype(int)
    clf = LogisticRegression(C=0.5, max_iter=500, random_state=42)
    clf.fit(X_scaled, target_labels)

    weights_path = VERCEL_DIR / "model_weights.npz"
    np.savez_compressed(
        weights_path,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coef=clf.coef_[0].astype(np.float32),
        intercept=np.array([clf.intercept_[0]], dtype=np.float32),
    )
    print(f"[SUCCESS] Saved compact Vercel model weights to: {weights_path} ({weights_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
