
from __future__ import annotations


import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils as keras_utils

from config import MODEL_SIZE

try:  # pragma: no cover - used when package imports are available
    from .. import config
except ImportError:  # pragma: no cover - fallback when run as a script
    import importlib

    config = importlib.import_module("config")  # type: ignore
History = keras.callbacks.History


_NOISE_GENERATOR = np.random.default_rng(config.RANDOM_SEED)


_MULTI_STEP_OUTPUT_MODELS: Dict[int, Optional[keras.Model]] = {}
_CACHE_MISS = object()


_MODEL_SIZE_FILTERS: Dict[str, Tuple[int, int, int]] = {
    "model_small": (16, 32, 64),
    "model_medium": (32, 64, 128),
    "model_large": (64, 128, 256),
}


def _get_model_filters() -> Tuple[int, int, int]:
    """Return the encoder filter sizes for the configured model size."""

    try:
        return _MODEL_SIZE_FILTERS[config.MODEL_SIZE]
    except KeyError as exc:  # pragma: no cover - defensive guard
        valid_options = ", ".join(sorted(_MODEL_SIZE_FILTERS))
        raise ValueError(
            f"Unknown MODEL_SIZE '{config.MODEL_SIZE}'. "
            f"Valid options are: {valid_options}."
        ) from exc


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible results."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_model_directory(model_name: str) -> Path:
    """Return the directory where artefacts for ``model_name`` are stored."""

    return config.RESULTS_DIRECTORY / model_name


def get_epoch_directory(model_name: str, epoch_index: int) -> Path:
    """Return the directory used to persist artefacts for ``epoch_index``."""

    return get_model_directory(model_name) / f"epoch_{epoch_index:04d}"


def ensure_output_directories(model_name: str) -> Path:
    """Create the directories used to store training artefacts."""

    config.RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    model_directory = get_model_directory(model_name)
    model_directory.mkdir(parents=True, exist_ok=True)

    (model_directory / config.PLOTS_DIRECTORY_NAME).mkdir(parents=True, exist_ok=True)

    return model_directory


def list_image_files(directories: Sequence[Path]) -> List[Path]:
    """Return all images contained in the provided directories.

    The lookup is recursive and case-insensitive with regards to the extensions
    configured in :mod:`model_creator.config`.
    """

    image_extensions = {ext.lower() for ext in config.IMAGE_EXTENSIONS}
    files: List[Path] = []
    for directory in directories:
        resolved = directory.expanduser()
        if not resolved.exists():
            continue
        for file in resolved.rglob("*"):
            if file.is_file() and file.suffix.lower() in image_extensions:
                files.append(file.resolve())
    return sorted(files)


def split_dataset(
    paths: Sequence[Path], validation_split: float
) -> Tuple[List[Path], List[Path]]:
    """Split ``paths`` into train and validation subsets."""

    if not paths:
        raise FileNotFoundError(
            "No images were found in the configured dataset directories."
        )

    paths = list(paths)
    rng = random.Random(config.RANDOM_SEED)
    rng.shuffle(paths)

    if validation_split <= 0.0:
        return paths, []

    validation_count = int(len(paths) * validation_split)
    if validation_count == 0 and len(paths) > 1:
        validation_count = 1

    validation_paths = paths[:validation_count]
    training_paths = paths[validation_count:]

    if not training_paths:
        raise ValueError(
            "After applying the validation split there are no images left for "
            "training. Reduce VALIDATION_SPLIT in the configuration."
        )

    return training_paths, validation_paths


def _parse_image(path: tf.Tensor) -> tf.Tensor:
    """Load and normalise an image from ``path``."""

    image = tf.io.read_file(path)
    image = tf.image.decode_image(
        image, channels=config.IMAGE_CHANNELS, expand_animations=False
    )
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    image = tf.ensure_shape(
        image, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS)
    )
    return image


def _augment_image(image: tf.Tensor) -> tf.Tensor:
    """Apply lightweight augmentation for generalisation."""

    image = tf.image.random_flip_left_right(image)
    return image


def _add_noise(image: tf.Tensor) -> tf.Tensor:
    """Inject Gaussian noise into ``image`` in the ``[0, 1]`` range."""

    noise = tf.random.normal(
        shape=tf.shape(image),
        mean=0.0,
        stddev=config.NOISE_STDDEV,
        dtype=image.dtype,
    )
    noisy_image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return noisy_image


def _create_pair(image: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Return a ``({"noisy_input": noisy}, clean)`` tuple for training."""

    noisy = _add_noise(image)
    return {"noisy_input": noisy}, image


def build_dataset(
    file_paths: Sequence[Path],
    *,
    augment: bool,
    shuffle: bool,
) -> tf.data.Dataset:
    """Construct a :class:`tf.data.Dataset` pipeline for the provided files."""

    if not file_paths:
        raise ValueError("Cannot build a dataset without image files.")

    string_paths = tf.convert_to_tensor([str(path) for path in file_paths])
    dataset = tf.data.Dataset.from_tensor_slices(string_paths)

    dataset = dataset.map(
        _parse_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment:
        dataset = dataset.map(
            _augment_image, num_parallel_calls=tf.data.AUTOTUNE
        )

    if shuffle:
        buffer_size = max(len(file_paths), config.BATCH_SIZE * 4)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=config.RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        _create_pair, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def _load_image_for_inference(path: Path) -> np.ndarray:
    """Load ``path`` into a normalised ``numpy`` array."""

    tensor = _parse_image(tf.convert_to_tensor(str(path)))
    return tensor.numpy()
def _get_multi_step_output_model(model: keras.Model) -> Optional[keras.Model]:
    """Return a model that exposes intermediate denoising steps."""

    model_id = id(model)
    cached = _MULTI_STEP_OUTPUT_MODELS.get(model_id, _CACHE_MISS)
    if cached is not _CACHE_MISS:
        return cached

    if not config.MULTI_STEP:
        _MULTI_STEP_OUTPUT_MODELS.pop(model_id, None)
        return None

    outputs: List[tf.Tensor] = []

    # ``build_denoiser_block`` names the final clipping layer of each pass
    # ``clip_to_valid_range`` (with an incrementing suffix). When the block is
    # applied multiple times Keras exposes each pass as an individual layer
    # whose name follows this pattern. Collecting the layers by their generated
    # names therefore produces the denoising sequence in call order.
    layer_index = 0
    while True:
        layer_name = (
            "clip_to_valid_range"
            if layer_index == 0
            else f"clip_to_valid_range_{layer_index}"
        )

        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            break

        outputs.append(layer.output)
        layer_index += 1

    if len(outputs) <= 1:
        _MULTI_STEP_OUTPUT_MODELS[model_id] = None
        return None

    multi_step_model = keras.Model(inputs=model.input, outputs=outputs)
    _MULTI_STEP_OUTPUT_MODELS[model_id] = multi_step_model
    return multi_step_model




def _predict_clean_image_sequence(
    model: keras.Model, image: np.ndarray
) -> List[np.ndarray]:
    """Return the denoised prediction sequence produced by ``model``."""

    batched_image = image[np.newaxis, ...]

    multi_step_model = _get_multi_step_output_model(model)
    if multi_step_model is None:
        prediction = model.predict(batched_image, verbose=0)[0]
        return [np.clip(prediction, 0.0, 1.0)]

    outputs = multi_step_model.predict(batched_image, verbose=0)
    if not isinstance(outputs, list):
        outputs = [outputs]

    return [np.clip(output[0], 0.0, 1.0) for output in outputs]


def _predict_clean_image(model: keras.Model, image: np.ndarray) -> np.ndarray:
    """Return the final denoised prediction produced by ``model``."""

    return _predict_clean_image_sequence(model, image)[-1]


def _generate_noise_image() -> np.ndarray:
    """Return a random noise image in the valid ``[0, 1]`` range."""

    noise = _NOISE_GENERATOR.random(
        size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        dtype=np.float32,
    )
    return noise


def generate_comparison_images(
    model: keras.Model, image_paths: Sequence[Path], output_directory: Path
) -> None:
    """Render qualitative comparison sequences for ``image_paths``."""

    if not image_paths:
        return

    output_directory.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        clean_image = _load_image_for_inference(image_path)

        noisy_image = _add_noise(tf.convert_to_tensor(clean_image)).numpy()

        image_output_directory = output_directory / image_path.stem
        _reset_directory(image_output_directory)

        if config.MULTI_STEP:
            denoised_sequence = _predict_clean_image_sequence(model, noisy_image)
            images_to_save = [
                ("initial", clean_image),
                ("noisy_image", noisy_image),
                *(
                    (f"denoiser_step_{index}", image)
                    for index, image in enumerate(denoised_sequence, start=1)
                ),
            ]
            _save_image_sequence(
                image_output_directory, images_to_save, add_index_prefix=True
            )
        else:
            denoised_image = _predict_clean_image(model, noisy_image)
            images_to_save = [
                ("initial", clean_image),
                ("noisy_image", noisy_image),
                ("denoised_image", denoised_image),
            ]
            _save_image_sequence(
                image_output_directory, images_to_save, add_index_prefix=False
            )


def generate_hallucinated_images(
    model: keras.Model, reference_paths: Sequence[Path], output_directory: Path
) -> None:
    """Render hallucinations generated from random noise inputs."""

    if not reference_paths:
        return

    output_directory.mkdir(parents=True, exist_ok=True)

    for reference_path in reference_paths:
        noise_image = _generate_noise_image()

        image_output_directory = output_directory / reference_path.stem
        _reset_directory(image_output_directory)

        if config.MULTI_STEP:
            hallucinated_sequence = _predict_clean_image_sequence(model, noise_image)
            images_to_save = [
                ("initial", noise_image),
                *(
                    (f"denoiser_step_{index}", image)
                    for index, image in enumerate(hallucinated_sequence, start=1)
                ),
            ]
            _save_image_sequence(
                image_output_directory, images_to_save, add_index_prefix=True
            )
        else:
            hallucinated_image = _predict_clean_image(model, noise_image)
            images_to_save = [
                ("initial", noise_image),
                ("hallucinated_image", hallucinated_image),
            ]
            _save_image_sequence(
                image_output_directory, images_to_save, add_index_prefix=False
            )


def _reset_directory(directory: Path) -> None:
    """Ensure ``directory`` exists and is empty."""

    if directory.exists():
        if directory.is_file():
            directory.unlink()
        else:
            shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def _save_image_sequence(
    directory: Path,
    images: Sequence[Tuple[str, np.ndarray]],
    *,
    add_index_prefix: bool,
) -> None:
    """Persist ``images`` as JPEG files inside ``directory``."""

    for index, (name, image) in enumerate(images):
        if add_index_prefix:
            filename = f"{index}_{name}.jpg"
        else:
            filename = f"{name}.jpg"
        output_path = directory / filename
        keras_utils.save_img(str(output_path), image, data_format="channels_last")

class EpochResultsSaver(keras.callbacks.Callback):
    """Persist the model and qualitative comparisons after each epoch."""

    def __init__(self, model_name: str, benchmark_paths: Sequence[Path]):
        super().__init__()
        self.model_name = model_name
        self.benchmark_paths = list(benchmark_paths)

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] | None = None) -> None:
        epoch_directory = get_epoch_directory(self.model_name, epoch)
        epoch_directory.mkdir(parents=True, exist_ok=True)

        model_path = epoch_directory / config.MODEL_FILENAME
        self.model.save(model_path)

        comparison_directory = epoch_directory / config.COMPARISON_DIRECTORY_NAME
        comparison_directory.mkdir(parents=True, exist_ok=True)

        hallucinated_directory = (
            epoch_directory / config.HALLUCINATED_DIRECTORY_NAME
        )
        hallucinated_directory.mkdir(parents=True, exist_ok=True)

        if self.benchmark_paths:
            generate_comparison_images(
                self.model, self.benchmark_paths, comparison_directory
            )
            generate_hallucinated_images(
                self.model, self.benchmark_paths, hallucinated_directory
            )

def build_denoising_block() -> keras.Model:
    """Create the core denoising block that predicts residual noise."""

    inputs = keras.Input(
        shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        name="noisy_image",
    )

    encoder_filters = _get_model_filters()

    x = inputs
    skip_connections: List[tf.Tensor] = []
    for filters in encoder_filters[:-1]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        skip_connections.append(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

    bottleneck_filters = encoder_filters[-1]
    x = layers.Conv2D(bottleneck_filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(bottleneck_filters, 3, padding="same", activation="relu")(x)

    for filters, skip in zip(reversed(encoder_filters[:-1]), reversed(skip_connections)):
        x = layers.UpSampling2D(size=2)(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)

    residual = layers.Conv2D(
        config.IMAGE_CHANNELS,
        kernel_size=3,
        padding="same",
        activation="tanh",
        name="predicted_noise",
    )(x)

    cleaned = layers.Subtract(name="clean_image")([inputs, residual])
    cleaned = layers.Lambda(
        lambda tensor: tf.clip_by_value(tensor, 0.0, 1.0), name="clip_to_valid_range"
    )(cleaned)

    return keras.Model(inputs=inputs, outputs=cleaned, name="denoiser_block")


def build_denoiser_model() -> keras.Model:
    """Construct the full denoising model, optionally applying multiple steps."""

    block = build_denoising_block()
    inputs = keras.Input(
        shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        name="noisy_input",
    )

    x = inputs
    steps = config.DENOISING_SEQUENCE_PASSES if config.MULTI_STEP else 1
    for _ in range(steps):
        x = block(x)

    return keras.Model(inputs=inputs, outputs=x, name=MODEL_SIZE)


def compile_model(model: keras.Model) -> None:
    """Compile ``model`` with the configured optimiser and loss."""

    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            keras.metrics.MeanSquaredError(name="mse"),
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )


def create_callbacks(
    model_name: str, benchmark_paths: Sequence[Path]
) -> List[keras.callbacks.Callback]:
    """Return default callbacks used during training."""

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss" if config.VALIDATION_SPLIT > 0 else "loss",
        patience=5,
        restore_best_weights=True,
    )

    epoch_saver = EpochResultsSaver(
        model_name=model_name, benchmark_paths=benchmark_paths
    )

    return [epoch_saver, early_stopping]


def plot_training_history(history: History, model_name: str) -> Path | None:
    """Plot the training and validation loss curves if available."""

    history_dict = history.history
    if not history_dict:
        return None

    import matplotlib.pyplot as plt

    plot_path = (
        get_model_directory(model_name)
        / config.PLOTS_DIRECTORY_NAME
        / "training_history.png"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(history_dict.get("loss", []), label="Training loss")
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("Denoiser training history")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def train() -> keras.Model:
    """Train the denoising model and persist it to disk."""

    set_random_seeds(config.RANDOM_SEED)

    image_paths = list_image_files(config.DATASET_DIRECTORIES)
    train_paths, validation_paths = split_dataset(
        image_paths, config.VALIDATION_SPLIT
    )

    train_dataset = build_dataset(train_paths, augment=True, shuffle=True)
    validation_dataset = (
        build_dataset(validation_paths, augment=False, shuffle=False)
        if validation_paths
        else None
    )

    model = build_denoiser_model()
    compile_model(model)

    model_name = model.name
    ensure_output_directories(model_name)

    benchmark_paths = list_image_files(config.BENCHMARK_DIRECTORIES)

    callbacks = create_callbacks(model_name, benchmark_paths)
    fit_kwargs = {
        "epochs": config.EPOCHS,
        "callbacks": callbacks,
        "verbose": 1,
    }
    if validation_dataset is not None:
        fit_kwargs["validation_data"] = validation_dataset

    history = model.fit(train_dataset, **fit_kwargs)

    plot_training_history(history, model_name)

    return model