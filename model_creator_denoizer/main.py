from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATASET_DIRECTORIES: Sequence[Path] = (
    Path("cropped"),
)
IMAGE_EXTENSIONS: Sequence[str] = (".jpg", ".jpeg")
MODEL_SAVE_PATH: Path = Path("denoiser_autoencoder.keras")

IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
IMAGE_CHANNELS: int = 3
BATCH_SIZE: int = 16
EPOCHS: int = 200
VALIDATION_SPLIT: float = 0.1
SHUFFLE_BUFFER_SIZE: int = 2048
RANDOM_SEED: int = 1337

NOISE_STDDEV: float = 0.5
NOISE_BLEND_MIN: float = 0.0
NOISE_BLEND_MAX: float = 1.0

AUTOTUNE = tf.data.AUTOTUNE

def _set_global_determinism(seed: int) -> None:
    random.seed(seed)
    tf.random.set_seed(seed)


def _list_image_files(directories: Sequence[Path]) -> List[Path]:
    image_paths: List[Path] = []
    for directory in directories:
        if not directory.exists():
            continue
        for extension in IMAGE_EXTENSIONS:
            image_paths.extend(directory.rglob(f"*{extension}"))
    # Remove duplicates while preserving order.
    unique_paths = []
    seen = set()
    for path in sorted(image_paths):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    if not unique_paths:
        joined_dirs = ", ".join(str(path) for path in directories)
        raise FileNotFoundError(
            "No training images were found. Expected to find .jpg files in: "
            f"{joined_dirs}"
        )
    return unique_paths


def _split_dataset(paths: Sequence[Path]) -> tuple[list[str], list[str]]:
    paths_list = list(paths)
    random.shuffle(paths_list)
    if len(paths_list) < 2:
        return [str(paths_list[0])], [str(paths_list[0])]

    validation_size = max(1, int(len(paths_list) * VALIDATION_SPLIT))
    if validation_size >= len(paths_list):
        validation_size = max(1, len(paths_list) - 1)

    validation_paths = [str(path) for path in paths_list[:validation_size]]
    training_paths = [str(path) for path in paths_list[validation_size:]]
    return training_paths, validation_paths


def _decode_image(image_path: tf.Tensor) -> tf.Tensor:
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_bytes, channels=IMAGE_CHANNELS)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return image


def _apply_noise(image: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    gaussian_noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=NOISE_STDDEV)
    noisy_version = tf.clip_by_value(image + gaussian_noise, 0.0, 1.0)

    blend_factor = tf.random.uniform([], minval=NOISE_BLEND_MIN, maxval=NOISE_BLEND_MAX)
    random_texture = tf.random.uniform(tf.shape(image), minval=0.0, maxval=1.0)
    noisy_image = blend_factor * noisy_version + (1.0 - blend_factor) * random_texture
    noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
    return noisy_image, image


def _create_dataset(paths: Sequence[str], shuffle: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(list(paths))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), SHUFFLE_BUFFER_SIZE),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(_decode_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(_apply_noise, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
def _conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_denoiser(input_shape: Sequence[int]) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    skips: list[tf.Tensor] = []

    x = inputs
    for filters in (64, 128, 256):
        x = _conv_block(x, filters)
        skips.append(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

    x = _conv_block(x, 512)

    for filters, skip in zip((256, 128, 64), reversed(skips)):
        x = layers.UpSampling2D(size=2)(x)
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, filters)

    outputs = layers.Conv2D(IMAGE_CHANNELS, 1, activation="sigmoid", padding="same")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="hallucinator_denoiser")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train() -> keras.Model:
    _set_global_determinism(RANDOM_SEED)

    image_paths = _list_image_files(DATASET_DIRECTORIES)
    train_paths, val_paths = _split_dataset(image_paths)

    print(
        "Loaded %d images. Training with %d samples, validating with %d samples." % (
            len(image_paths),
            len(train_paths),
            len(val_paths),
        )
    )

    train_dataset = _create_dataset(train_paths, shuffle=True)
    validation_dataset = _create_dataset(val_paths, shuffle=False)

    model = build_denoiser((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mae")

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            verbose=1,
            min_lr=1e-6,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_SAVE_PATH)

    print(f"Training complete. Model saved to '{MODEL_SAVE_PATH}'.")
    if history.history:
        best_val = min(history.history.get("val_loss", [math.inf]))
        best_train = min(history.history.get("loss", [math.inf]))
        print(f"Best training loss: {best_train:.6f}")
        print(f"Best validation loss: {best_val:.6f}")

    return model


def main() -> None:
    train()


if __name__ == "__main__":
    main()