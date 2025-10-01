from __future__ import annotations
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATASET_DIRECTORIES: Sequence[Path] = (Path("cropped_subset"))

IMAGE_EXTENSIONS: Sequence[str] = (".jpg", ".jpeg")

RESULTS_DIRECTORY: Path = Path("results")
MODEL_FILENAME: str = "denoiser_autoencoder.keras"
PLOTS_DIRECTORY_NAME: str = "plots"
HALLUCINATION_DIRECTORY_NAME: str = "hallucinated_images"
DENOISED_DIRECTORY_NAME: str = "denoised_images"

BENCHMARK_DIRECTORY: Path = Path("benchmark")

IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
IMAGE_CHANNELS: int = 3
BATCH_SIZE: int = 16
EPOCHS: int = 2
VALIDATION_SPLIT: float = 0.1
SHUFFLE_BUFFER_SIZE: int = 2048
RANDOM_SEED: int = 1337

NOISE_STDDEV: float = 0.5
NOISE_BLEND_MIN: float = 0.0
NOISE_BLEND_MAX: float = 1.0

HALLUCINATION_SEQUENCE_COUNT: int = 10
HALLUCINATION_SEQUENCE_LENGTH: int = 32

DENOISING_SEQUENCE_PASSES: int = 15

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
def _prepare_execution_directories() -> tuple[Path, Path, Path, Path, Path]:
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    execution_dir = RESULTS_DIRECTORY / timestamp
    counter = 1
    while execution_dir.exists():
        execution_dir = RESULTS_DIRECTORY / f"execution_{timestamp}_{counter:04d}"
        counter += 1

    model_dir = execution_dir / "model"
    hallucination_dir = execution_dir / HALLUCINATION_DIRECTORY_NAME
    denoised_dir = execution_dir / DENOISED_DIRECTORY_NAME
    plots_dir = execution_dir / PLOTS_DIRECTORY_NAME

    model_dir.mkdir(parents=True, exist_ok=True)
    hallucination_dir.mkdir(parents=True, exist_ok=True)
    denoised_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / MODEL_FILENAME
    return execution_dir, model_path, hallucination_dir, denoised_dir, plots_dir


def _generate_hallucination_sequences(model: keras.Model, root_dir: Path) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)

    for sequence_index in range(HALLUCINATION_SEQUENCE_COUNT):
        sequence_dir = root_dir / f"{sequence_index:04d}"
        sequence_dir.mkdir(parents=True, exist_ok=True)

        current_image = tf.random.uniform(
            shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
            minval=0.0,
            maxval=1.0,
            dtype=tf.float32,
        )
        tf.keras.utils.save_img(sequence_dir / "0000.jpg", current_image[0])

        for step in range(1, HALLUCINATION_SEQUENCE_LENGTH + 1):
            prediction = model.predict(current_image, verbose=0)
            current_image = tf.convert_to_tensor(prediction, dtype=tf.float32)
            tf.keras.utils.save_img(
                sequence_dir / f"{step:04d}.jpg",
                prediction[0],
            )


def _load_and_prepare_benchmark_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    image_paths: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        image_paths.extend(sorted(directory.rglob(f"*{extension}")))
    return image_paths


def _generate_denoising_sequences(
    model: keras.Model, root_dir: Path, benchmark_dir: Path
) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)

    benchmark_images = _load_and_prepare_benchmark_images(benchmark_dir)
    if not benchmark_images:
        print(
            "No benchmark images were found for denoising visualisations. "
            f"Looked in '{benchmark_dir}'."
        )
        return

    for sequence_index, image_path in enumerate(benchmark_images):
        sequence_dir = root_dir / f"{sequence_index:02d}"
        sequence_dir.mkdir(parents=True, exist_ok=True)

        clean_image = _decode_image(tf.constant(str(image_path)))
        noisy_image, _ = _apply_noise(clean_image)

        tf.keras.utils.save_img(sequence_dir / "0000.jpg", noisy_image)

        current_batch = tf.expand_dims(noisy_image, axis=0)
        for step in range(1, DENOISING_SEQUENCE_PASSES + 1):
            prediction = model.predict(current_batch, verbose=0)
            prediction_tensor = tf.convert_to_tensor(prediction[0], dtype=tf.float32)
            tf.keras.utils.save_img(
                sequence_dir / f"{step:04d}.jpg",
                prediction_tensor,
            )
            current_batch = prediction


def _save_training_plots(
    history: keras.callbacks.History,
    epoch_times: Sequence[float],
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(epoch_times) + 1)
    plt.figure()
    plt.plot(epochs, epoch_times, marker="o")
    plt.title("Epoch Duration")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "epoch_times.jpg", format="jpg")
    plt.close()

    loss_history = history.history.get("loss", [])
    val_history = history.history.get("val_loss", [])
    if loss_history or val_history:
        max_epochs = max(len(loss_history), len(val_history))
        epochs = range(1, max_epochs + 1)

        plt.figure()
        if loss_history:
            plt.plot(range(1, len(loss_history) + 1), loss_history, label="Training Loss")
        if val_history:
            plt.plot(range(1, len(val_history) + 1), val_history, label="Validation Loss")
        plt.title("Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "loss_curve.jpg", format="jpg")
        plt.close()


class EpochTimer(keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.epoch_durations: list[float] = []
        self._epoch_start: float | None = None

    def on_epoch_begin(self, epoch: int, logs=None) -> None:  # type: ignore[override]
        del logs
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs=None) -> None:  # type: ignore[override]
        del logs
        if self._epoch_start is None:
            return
        elapsed = time.perf_counter() - self._epoch_start
        self.epoch_durations.append(elapsed)
        self._epoch_start = None


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

    epoch_timer = EpochTimer()

    callbacks: list[keras.callbacks.Callback] = [
        epoch_timer,
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

    (
        execution_dir,
        model_save_path,
        hallucination_dir,
        denoised_dir,
        plots_dir,
    ) = _prepare_execution_directories()
    model.save(model_save_path)

    _generate_hallucination_sequences(model, hallucination_dir)
    _generate_denoising_sequences(model, denoised_dir, BENCHMARK_DIRECTORY)
    _save_training_plots(history, epoch_timer.epoch_durations, plots_dir)

    print(f"Training complete. Model saved to '{model_save_path}'.")
    print(f"Hallucination sequences saved to '{hallucination_dir}'.")
    print(f"Denoising sequences saved to '{denoised_dir}'.")
    print(f"Training plots saved to '{plots_dir}'.")
    print(f"Execution artifacts available in '{execution_dir}'.")
    if history.history:
        best_val = min(history.history.get("val_loss", [math.inf]))
        best_train = min(history.history.get("loss", [math.inf]))
        print(f"Best training loss: {best_train:.6f}")
        print(f"Best validation loss: {best_val:.6f}")

    return model

if __name__ == "__main__":
    train()