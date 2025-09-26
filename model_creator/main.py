#!/usr/bin/env python3
"""
Simple Keras convolutional autoencoder
--------------------------------------
- Reads all JPG images in the `data/` directory for (optional) training.
- Either loads pretrained weights (if present) or trains locally on your images.
- Encodes a color image (256x256) to a latent vector of size N, then decodes back.
- Reads "input.jpg" and writes the reconstructed "output.jpg".
- All parameters are hard-coded below. No CLI arguments.
- Comments are in English as requested.
"""

import os
import glob
import time
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# =============================
# Hard-coded configuration
# =============================
IMG_SIZE: int = 128                # Target H and W
CHANNELS: int = 3                  # RGB
LATENT_DIM: int = 64              # Size of the latent vector (N)
BATCH_SIZE: int = 16
EPOCHS: int = 2                   # Increase for better quality if you have more data and time
SHUFFLE_BUFFER: int = 512
DATA_DIR: str = "data/"             # Directory with training JPGs
INPUT_IMAGE: str = "_img_in.jpg"     # Image to encode/decode
OUTPUT_IMAGE: str = "output.jpg"   # Where to save the reconstruction
WEIGHTS_PATH: str = "ae_weights.h5" # File to save/load model weights
CACHE_DATASET: bool = True         # Cache dataset in memory if possible
VALIDATION_SPLIT: float = 0.05     # Small validation split for training feedback
SEED: int = 42                     # For deterministic shuffles where applicable
LEARNING_RATE: float = 1e-3        # Adam learning rate
LOSS_NAME: str = "mae"             # Try "mse" or "mae"
# =============================


def list_jpgs(folder: str) -> List[str]:
    """List JPG/JPEG files (non-recursive)."""
    patterns = [os.path.join(folder, "*.jpg"), os.path.join(folder, "*.jpeg"), os.path.join(folder, "*.JPG"), os.path.join(folder, "*.JPEG")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(files)
    return files


def decode_and_resize(path: tf.Tensor) -> tf.Tensor:
    """Load image from path, decode, resize to IMG_SIZE, normalize to [0,1]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # [0,1]
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR)
    return img


def build_dataset(paths: List[str]) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.shuffle(buffer_size=min(len(paths), SHUFFLE_BUFFER), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

    def augment(x):
        x = tf.image.random_flip_left_right(x, seed=SEED)
        x = tf.image.random_flip_up_down(x, seed=SEED)
        x = tf.image.random_brightness(x, max_delta=0.05, seed=SEED)
        x = tf.image.random_contrast(x, lower=0.9, upper=1.1, seed=SEED)
        return tf.clip_by_value(x, 0.0, 1.0)

    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)  # <<< important
    if CACHE_DATASET:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_autoencoder(img_size: int, channels: int, latent_dim: int) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Builds a convolutional autoencoder: encoder, decoder, and AE model."""
    # Encoder
    encoder_inputs = layers.Input(shape=(img_size, img_size, channels), name="encoder_input")
    x = encoder_inputs
    # Downsampling blocks
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPool2D()(x)  # /2

    # Now the spatial dims should be img_size / 16 (for 4 pooling layers)
    # Compute resulting size
    downsample_factor = 2 ** 4
    feat_size = img_size // downsample_factor
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    latent = layers.Dense(latent_dim, name="latent")(x)
    encoder = models.Model(encoder_inputs, latent, name="encoder")

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = decoder_inputs
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(feat_size * feat_size * 256, activation="relu")(x)
    x = layers.Reshape((feat_size, feat_size, 256))(x)

    # Upsampling blocks (mirror of encoder)
    for filters in [256, 128, 64, 32]:
        x = layers.Conv2DTranspose(filters, 3, padding="same", strides=1, activation="relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same", strides=1, activation="relu")(x)
        x = layers.UpSampling2D()(x)  # *2

    # Final conv to bring channels back, with sigmoid to [0,1]
    x = layers.Conv2D(channels, 3, padding="same", activation="sigmoid", name="decoder_output")(x)
    decoder = models.Model(decoder_inputs, x, name="decoder")

    # Autoencoder = decoder(encoder(x))
    ae_inputs = encoder_inputs
    ae_outputs = decoder(encoder(ae_inputs))
    autoencoder = models.Model(ae_inputs, ae_outputs, name="autoencoder")
    return encoder, decoder, autoencoder


def save_image(path: str, img_float: np.ndarray):
    """Save float image [0,1] as JPG."""
    x = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    # Ensure shape is HxWxC
    if x.ndim == 2:
        x = np.stack([x]*3, axis=-1)
    encoded = tf.io.encode_jpeg(x, quality=95)
    tf.io.write_file(path, encoded)


def main():
    # Log device info
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU(s): {gpus}")
        except Exception as e:
            print("Could not set memory growth:", e)
    else:
        print("No GPU found, running on CPU.")

    # Build models
    encoder, decoder, autoencoder = build_autoencoder(IMG_SIZE, CHANNELS, LATENT_DIM)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=LOSS_NAME)
    autoencoder.summary()

    # Try to load weights if present
    weights_loaded = False
    if os.path.exists(WEIGHTS_PATH):
        try:
            autoencoder.load_weights(WEIGHTS_PATH)
            weights_loaded = True
            print(f"Loaded pretrained weights from '{WEIGHTS_PATH}'.")
        except Exception as e:
            print(f"Found weights at '{WEIGHTS_PATH}' but failed to load: {e}")

    # If no weights loaded, try to train using data/
    if not weights_loaded:
        train_paths = list_jpgs(DATA_DIR)
        if len(train_paths) == 0:
            print(f"No training images found in '{DATA_DIR}'. Skipping training.")
        else:
            print(f"Found {len(train_paths)} training images in '{DATA_DIR}'.")
            # Optionally split a tiny validation set
            val_count = max(1, int(len(train_paths) * VALIDATION_SPLIT)) if len(train_paths) > 10 else 0
            if val_count > 0:
                np.random.seed(SEED)
                idx = np.random.permutation(len(train_paths))
                val_idx = set(idx[:val_count])
                train_list = [p for i, p in enumerate(train_paths) if i not in val_idx]
                val_list = [p for i, p in enumerate(train_paths) if i in val_idx]
                train_ds = build_dataset(train_list)
                val_ds = tf.data.Dataset.from_tensor_slices(val_list) \
                    .map(decode_and_resize, num_parallel_calls = tf.data.AUTOTUNE) \
                    .map(lambda x: (x, x), num_parallel_calls = tf.data.AUTOTUNE)
                if CACHE_DATASET:
                    val_ds = val_ds.cache()
                val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            else:
                train_ds = build_dataset(train_paths)
                val_ds = None

            print("Starting training...")
            start = time.time()
            history = autoencoder.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                verbose=1,
            )
            elapsed = time.time() - start
            print(f"Training done in {elapsed/60:.1f} min.")

            # Save weights
            try:
                autoencoder.save_weights(WEIGHTS_PATH)
                print(f"Saved weights to '{WEIGHTS_PATH}'.")
            except Exception as e:
                print("Failed to save weights:", e)

    # Load and process input image
    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError(f"'{INPUT_IMAGE}' not found in current directory.")

    raw = tf.io.read_file(INPUT_IMAGE)
    img = tf.image.decode_jpeg(raw, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # [0,1]
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR)
    img_np = img.numpy()[None, ...]  # add batch dim

    # Reconstruct
    recon = autoencoder.predict(img_np, verbose=0)[0]  # remove batch dim

    # Save output
    save_image(OUTPUT_IMAGE, recon.numpy() if isinstance(recon, tf.Tensor) else recon)
    print(f"Wrote reconstruction to '{OUTPUT_IMAGE}'.")


if __name__ == "__main__":
    main()
