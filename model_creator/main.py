"""
Simple Keras convolutional autoencoder
--------------------------------------
- Reads all JPG/PNG images in the `data/` directory for (optional) training.
- Either loads pretrained weights (if present) or trains locally on your images.
- Encodes a color image (256x256) to a latent vector of size N, then decodes back.
- Reads "_img_in.*" (jpg/jpeg/png) and writes the reconstructed "output.*".
- All parameters are hard-coded below. No CLI arguments.
- Comments are in English as requested.
"""

import os
import time
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# =============================
# Hard-coded configuration
# =============================
IMG_SIZE: int = 64                # Target H and W
CHANNELS: int = 3                  # RGB
LATENT_DIM: int = 2000              # Size of the latent vector (N)
BATCH_SIZE: int = 64
EPOCHS: int = 30                   # Increase for better quality if you have more data and time
SHUFFLE_BUFFER: int = 512
DATA_DIR: str = "datasets/flickr/128"             # Directory with training images
INPUT_IMAGE: str = "_img_in.jpg"     # Image to encode/decode
OUTPUT_IMAGE: str = "output.png"   # Where to save the reconstruction
MODEL_PATH: str = "autoencoder.keras"  # File to save/load the entire model
CACHE_DATASET: bool = True         # Cache dataset in memory if possible
VALIDATION_SPLIT: float = 0.05     # Small validation split for training feedback
SEED: int = 42                     # For deterministic shuffles where applicable
LEARNING_RATE: float = 1e-3        # Adam learning rate
LOSS_NAME: str = "mae"             # Try "mse" or "mae"
# Supported image file extensions for I/O
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")
# =============================


def is_supported_image(path: str) -> bool:
    """Return True if the given filename has a supported image extension."""
    return path.lower().endswith(SUPPORTED_EXTENSIONS)


def list_training_images(folder: str) -> List[str]:
    """List supported training images (non-recursive)."""
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, name) for name in os.listdir(folder) if is_supported_image(name)]
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    return files


def decode_and_resize(path: tf.Tensor) -> tf.Tensor:
    """Load image from path, decode, resize to IMG_SIZE, normalize to [0,1]."""
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)
    img.set_shape((None, None, CHANNELS))
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

    def encoder_block(x: tf.Tensor, filters: int, block_idx: int) -> tf.Tensor:
        """Apply a pair of Conv+BN+ReLU layers followed by pooling and dropout."""
        name_prefix = f"enc_block_{block_idx}"
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name_prefix}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name_prefix}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)
        x = layers.MaxPool2D(name=f"{name_prefix}_pool")(x)
        x = layers.Dropout(0.1, name=f"{name_prefix}_dropout")(x)
        return x

    def decoder_block(x: tf.Tensor, filters: int, block_idx: int) -> tf.Tensor:
        """Apply transpose convolutions with BN+ReLU followed by upsampling."""
        name_prefix = f"dec_block_{block_idx}"
        x = layers.Conv2DTranspose(filters, 3, padding="same", use_bias=False, name=f"{name_prefix}_tconv1")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same", use_bias=False, name=f"{name_prefix}_tconv2")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)
        x = layers.UpSampling2D(name=f"{name_prefix}_upsample")(x)
        return x

    # Encoder
    encoder_inputs = layers.Input(shape=(img_size, img_size, channels), name="encoder_input")
    x = encoder_inputs
    for idx, filters in enumerate([32, 64, 128, 256], start=1):
        x = encoder_block(x, filters, idx)

    # Now the spatial dims should be img_size / 16 (for 4 pooling layers)
    # Compute resulting size
    downsample_factor = 2 ** 4
    feat_size = img_size // downsample_factor
    x = layers.Flatten(name="enc_flatten")(x)
    x = layers.Dense(512, use_bias=False, name="enc_dense1")(x)
    x = layers.BatchNormalization(name="enc_dense1_bn")(x)
    x = layers.Activation("relu", name="enc_dense1_relu")(x)
    x = layers.Dropout(0.2, name="enc_dense1_dropout")(x)
    x = layers.Dense(256, use_bias=False, name="enc_dense2")(x)
    x = layers.BatchNormalization(name="enc_dense2_bn")(x)
    x = layers.Activation("relu", name="enc_dense2_relu")(x)
    latent = layers.Dense(latent_dim, name="latent")(x)
    encoder = models.Model(encoder_inputs, latent, name="encoder")

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = decoder_inputs
    x = layers.Dense(256, use_bias=False, name="dec_dense1")(x)
    x = layers.BatchNormalization(name="dec_dense1_bn")(x)
    x = layers.Activation("relu", name="dec_dense1_relu")(x)
    x = layers.Dense(512, use_bias=False, name="dec_dense2")(x)
    x = layers.BatchNormalization(name="dec_dense2_bn")(x)
    x = layers.Activation("relu", name="dec_dense2_relu")(x)
    x = layers.Dense(feat_size * feat_size * 256, activation="relu", name="dec_dense3")(x)
    x = layers.Reshape((feat_size, feat_size, 256), name="dec_reshape")(x)

    for idx, filters in enumerate([256, 128, 64, 32], start=1):
        x = decoder_block(x, filters, idx)

    x = layers.Conv2D(channels, 3, padding="same", activation="sigmoid", name="decoder_output")(x)
    decoder = models.Model(decoder_inputs, x, name="decoder")

    # Autoencoder = decoder(encoder(x))
    ae_inputs = encoder_inputs
    ae_outputs = decoder(encoder(ae_inputs))
    autoencoder = models.Model(ae_inputs, ae_outputs, name="autoencoder")
    return encoder, decoder, autoencoder


def save_image(path: str, img_float: np.ndarray):
    """Save float image [0,1] using the format implied by the extension."""
    x = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    # Ensure shape is HxWxC
    if x.ndim == 2:
        x = np.stack([x]*3, axis=-1)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        encoded = tf.io.encode_png(x, compression=6)
    elif ext in (".jpg", ".jpeg"):
        encoded = tf.io.encode_jpeg(x, quality=95)
    else:
        raise ValueError(f"Unsupported output image extension: '{ext}'")
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

    # Try to load a previously saved model; fall back to building a new one.
    autoencoder = None
    encoder = None
    decoder = None
    model_loaded = False
    if os.path.exists(MODEL_PATH):
        try:
            autoencoder = models.load_model(MODEL_PATH)
            model_loaded = True
            print(f"Loaded pretrained model from '{MODEL_PATH}'.")
            try:
                encoder = autoencoder.get_layer("encoder")
                decoder = autoencoder.get_layer("decoder")
            except ValueError:
                # Saved model may not expose sub-models with these names; ignore.
                encoder = None
                decoder = None
        except Exception as e:
            print(f"Found model at '{MODEL_PATH}' but failed to load: {e}")

    if autoencoder is None:
        encoder, decoder, autoencoder = build_autoencoder(IMG_SIZE, CHANNELS, LATENT_DIM)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=LOSS_NAME)
    autoencoder.summary()

    # If no weights loaded, try to train using data/
    if not model_loaded:
        train_paths = list_training_images(DATA_DIR)
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


            # Save the full model in the native Keras format
            try:
                autoencoder.save(MODEL_PATH)
                print(f"Saved model to '{MODEL_PATH}'.")
            except Exception as e:
                print("Failed to save model:", e)

    # Load and process input image
    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError(f"'{INPUT_IMAGE}' not found in current directory.")
    if not is_supported_image(INPUT_IMAGE):
        raise ValueError(
            f"Unsupported input image extension for '{INPUT_IMAGE}'. Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    raw = tf.io.read_file(INPUT_IMAGE)
    img = tf.io.decode_image(raw, channels=CHANNELS, expand_animations=False)
    img.set_shape((None, None, CHANNELS))
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