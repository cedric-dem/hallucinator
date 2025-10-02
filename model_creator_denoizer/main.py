from __future__ import annotations
import math
import random
import shutil
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
from tensorflow.keras.utils import register_keras_serializable
import numpy as np

DATASET_DIRECTORIES: Sequence[Path] = (Path("cropped_subset"),)

####################################

MULTI_STEP: bool = True

if MULTI_STEP:
    # takes more space,
    MODEL_NAME: str = "small_model"
    BATCH_SIZE: int = 7

else:
    # takes less space,
    MODEL_NAME: str = "huge_model"
    BATCH_SIZE: int = 16

EPOCHS: int = 20

HALLUCINATION_SEQUENCE_COUNT: int = 10
HALLUCINATION_SEQUENCE_LENGTH: int = 32

DENOISING_SEQUENCE_PASSES: int = 15

TRAINING_REFINEMENT_STEPS: int = DENOISING_SEQUENCE_PASSES
INFERENCE_REFINEMENT_STEPS: int = max(
    HALLUCINATION_SEQUENCE_LENGTH, DENOISING_SEQUENCE_PASSES
)

####################################

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
VALIDATION_SPLIT: float = 0.1
SHUFFLE_BUFFER_SIZE: int = 2048
RANDOM_SEED: int = 1337

NOISE_STDDEV: float = 0.5
NOISE_BLEND_MIN: float = 0.0
NOISE_BLEND_MAX: float = 1.0

ModelConfig = dict[str, Sequence[int] | int]

MODEL_CONFIGURATIONS: dict[str, ModelConfig] = {
    "huge_model": {
        "down_filters": (64, 128, 256),
        "bottleneck_filters": 512,
        "up_filters": (256, 128, 64),
    },
    "large_model": {
        "down_filters": (32, 64, 128),
        "bottleneck_filters": 256,
        "up_filters": (128, 64, 32),
    },
    "medium_model": {
        "down_filters": (8, 16, 32),
        "bottleneck_filters": 64,
        "up_filters": (32, 16, 8),
    },
    "small_model": { #biggest possible with 7 epoch mon my gpu
        "down_filters": (6, 12, 24),
        "bottleneck_filters": 48,
        "up_filters": (24, 12, 6),
    },
    "tiny_model": { #biggest possible with 8 epoch mon my gpu
        "down_filters": (4, 8, 16),
        "bottleneck_filters": 32,
        "up_filters": (16, 8, 4),
    },
}

####################################

def _get_model_configuration(name: str) -> ModelConfig:
    try:
        config = MODEL_CONFIGURATIONS[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_CONFIGURATIONS))
        raise ValueError(
            f"Unknown MODEL_NAME '{name}'. Available options: {available}."
        ) from exc
    return dict(config)

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
    tf.debugging.assert_all_finite(
        noisy_image, "NaN or Inf detected in generated noisy image"
    )
    tf.debugging.assert_all_finite(image, "NaN or Inf detected in clean image")
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
def _clip_to_valid_range(tensor: tf.Tensor) -> tf.Tensor:
    """Clip values to [0, 1] while preserving gradients."""

    clipped = tf.clip_by_value(tensor, 0.0, 1.0)
    # Use a straight-through estimator so that gradients are not zeroed out when
    # values saturate at the clipping boundaries. This is particularly important
    # for the multi-step refinement process where repeated clipping could
    # otherwise stall learning.
    return tensor + tf.stop_gradient(clipped - tensor)


@register_keras_serializable(package="hallucinator")
class ResidualDenoiserCore(layers.Layer):
    def __init__(
        self,
        *,
        down_filters: Sequence[int],
        bottleneck_filters: int,
        up_filters: Sequence[int],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._down_filters = tuple(down_filters)
        self._bottleneck_filters = int(bottleneck_filters)
        self._up_filters = tuple(up_filters)
        if len(self._down_filters) != len(self._up_filters):
            raise ValueError(
                "down_filters and up_filters must have the same length to form skip connections"
            )
        self._down_blocks: list[keras.Sequential] = []
        self._pools: list[layers.MaxPooling2D] = []
        for index, filters in enumerate(self._down_filters):
            block = keras.Sequential(
                [
                    layers.Conv2D(filters, 3, padding="same"),
                    layers.BatchNormalization(),
                    layers.Activation("relu"),
                    layers.Conv2D(filters, 3, padding="same"),
                    layers.BatchNormalization(),
                    layers.Activation("relu"),
                ],
                name=f"down_block_{index}",
            )
            self._down_blocks.append(block)
            self._pools.append(layers.MaxPooling2D(pool_size=2, name=f"down_pool_{index}"))

        self._bottleneck_block = keras.Sequential(
            [
                layers.Conv2D(self._bottleneck_filters, 3, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(self._bottleneck_filters, 3, padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
            ],
            name="bottleneck_block",
        )

        self._up_samples: list[layers.UpSampling2D] = [
            layers.UpSampling2D(size=2, name=f"up_sample_{index}")
            for index in range(len(self._up_filters))
        ]
        self._concats: list[layers.Concatenate] = [
            layers.Concatenate(name=f"concat_{index}")
            for index in range(len(self._up_filters))
        ]
        self._up_blocks: list[keras.Sequential] = []
        for index, filters in enumerate(self._up_filters):
            block = keras.Sequential(
                [
                    layers.Conv2D(filters, 3, padding="same"),
                    layers.BatchNormalization(),
                    layers.Activation("relu"),
                    layers.Conv2D(filters, 3, padding="same"),
                    layers.BatchNormalization(),
                    layers.Activation("relu"),
                ],
                name=f"up_block_{index}",
            )
            self._up_blocks.append(block)

        self._residual_output = layers.Conv2D(
            IMAGE_CHANNELS, 1, activation="tanh", padding="same", name="residual_output"
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        skips: list[tf.Tensor] = []
        x = inputs
        for block, pool in zip(self._down_blocks, self._pools):
            x = block(x, training=training)
            skips.append(x)
            x = pool(x)

        x = self._bottleneck_block(x, training=training)

        for upsample, concat, block, skip in zip(
            self._up_samples, self._concats, self._up_blocks, reversed(skips)
        ):
            x = upsample(x)
            x = concat([x, skip])
            x = block(x, training=training)

        return self._residual_output(x)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "down_filters": self._down_filters,
                "bottleneck_filters": self._bottleneck_filters,
                "up_filters": self._up_filters,
            }
        )
        return config





@register_keras_serializable(package="hallucinator")
class IterativeRefinementLayer(layers.Layer):
    def __init__(
        self,
        steps: int,
        core_config: ModelConfig,
        *,
        step_gain: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.steps = int(steps)
        if self.steps <= 0:
            raise ValueError("IterativeRefinementLayer requires a positive number of steps")
        self._core_config = dict(core_config)
        if step_gain is None:
            # Normalise the per-step residual magnitude so that the cumulative
            # update remains stable even when the denoiser is unrolled for many
            # refinement iterations. Without this scaling the randomly
            # initialised model tends to drive activations straight to the
            # clipping boundaries, stalling optimisation and producing the
            # reported green-noise artefacts.
            self.step_gain = 1.0 / float(self.steps)
        else:
            if step_gain <= 0:
                raise ValueError("'step_gain' must be a positive number")
            self.step_gain = float(step_gain)
        self.core = ResidualDenoiserCore(name="denoiser_core", **self._core_config)

    def _run_iterative_steps(
        self,
        inputs: tf.Tensor,
        *,
        steps: int,
        training: bool,
        collect_sequence: bool,
    ) -> tuple[tf.Tensor, list[tf.Tensor]]:
        current = inputs
        sequence: list[tf.Tensor] = []
        gain = tf.convert_to_tensor(self.step_gain, dtype=inputs.dtype)
        for _ in range(steps):
            residual = self.core(current, training=training)
            current = _clip_to_valid_range(current + gain * residual)
            if collect_sequence:
                sequence.append(current)
        return current, sequence

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        final_output, _ = self._run_iterative_steps(
            inputs, steps=self.steps, training=training, collect_sequence=False
        )
        return final_output

    def generate_sequence(
        self,
        inputs: tf.Tensor,
        *,
        steps: int | None = None,
        training: bool = False,
    ) -> tf.Tensor:

        run_steps = self.steps if steps is None else int(steps)
        if run_steps <= 0:
            raise ValueError("'steps' must be a positive integer")

        inputs_tensor = tf.convert_to_tensor(inputs)

        if not self.built:
            # Ensure weights are initialised before running the custom loop.
            self(inputs_tensor, training=training)

        _, sequence = self._run_iterative_steps(
            inputs_tensor, steps=run_steps, training=training, collect_sequence=True
        )

        if not sequence:
            raise RuntimeError("Iterative refinement did not produce any outputs")

        return tf.stack(sequence, axis=1)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "steps": self.steps,
                "core_config": dict(self._core_config),
                "step_gain": self.step_gain,
            }
        )
        return config



def build_denoiser(input_shape: Sequence[int]) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    model_config = _get_model_configuration(MODEL_NAME)

    if MULTI_STEP:
        refinement_layer = IterativeRefinementLayer(
            steps=TRAINING_REFINEMENT_STEPS,
            core_config=model_config,
            name="iterative_refinement",
        )
        final_output = refinement_layer(inputs)
        final_output = layers.Lambda(
            _clip_to_valid_range, name="denoised_output"
        )(final_output)
        model = keras.Model(
            inputs=inputs,
            outputs=final_output,
            name="hallucinator_iterative_denoiser",
        )
        setattr(model, "iterative_refinement_layer", refinement_layer)
        setattr(model, "iterative_refinement_steps", TRAINING_REFINEMENT_STEPS)
        setattr(model, "iterative_refinement_inference_steps", INFERENCE_REFINEMENT_STEPS)
        setattr(model, "model_name", MODEL_NAME)
        setattr(model, "model_configuration", dict(model_config))
        return model

    core = ResidualDenoiserCore(name="denoiser_core", **model_config)
    residual = core(inputs)
    combined = layers.Add(name="residual_add")([inputs, residual])
    outputs = layers.Lambda(_clip_to_valid_range, name="denoised_output")(combined)
    model = keras.Model(inputs=inputs, outputs=outputs, name="hallucinator_denoiser")
    setattr(model, "model_name", MODEL_NAME)
    setattr(model, "model_configuration", dict(model_config))
    return model

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

def _run_denoiser_sequence(
    model: keras.Model, batch: tf.Tensor, steps: int
) -> list[tf.Tensor]:
    outputs: list[tf.Tensor] = []
    current_batch = batch

    if MULTI_STEP and hasattr(model, "iterative_refinement_layer"):
        refinement_layer = getattr(model, "iterative_refinement_layer")
        if hasattr(refinement_layer, "generate_sequence"):
            stacked_outputs = refinement_layer.generate_sequence(
                current_batch, steps=steps, training=False
            )
        else:
            stacked_outputs = refinement_layer(current_batch, training=False)

        rank = stacked_outputs.shape.rank
        if rank is None:
            rank = int(tf.rank(stacked_outputs))

        if rank >= 5:
            step_tensors = tf.unstack(stacked_outputs[0], axis=0)
        else:
            step_tensors = [tf.squeeze(stacked_outputs, axis=0)]

        outputs.extend(step_tensors[:steps])
        if len(outputs) >= steps:
            return outputs[:steps]
        if step_tensors:
            current_batch = tf.expand_dims(step_tensors[-1], axis=0)

    while len(outputs) < steps:
        prediction = model.predict(current_batch, verbose=0)
        prediction_tensor = tf.convert_to_tensor(prediction[0], dtype=tf.float32)
        outputs.append(prediction_tensor)
        current_batch = tf.convert_to_tensor(prediction, dtype=tf.float32)

    return outputs[:steps]


def _tensor_to_image_array(image: tf.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, tf.Tensor):
        array = image.numpy()
    else:
        array = np.asarray(image)

    if array.ndim == 4 and array.shape[0] == 1:
        array = np.squeeze(array, axis=0)

    if array.ndim != 3:
        raise ValueError(
            "Expected image tensor with 3 dimensions (H, W, C) after squeezing."
        )

    array = np.clip(array, 0.0, 1.0).astype(np.float32)
    return array


def _save_sequence_grid(images: Sequence[np.ndarray], destination: Path) -> None:
    if not images:
        raise ValueError("At least one image is required to create a grid.")

    first_image = images[0]
    if first_image.ndim != 3:
        raise ValueError("Images must have shape (height, width, channels).")

    height, width, channels = first_image.shape
    grid_rows = 6
    grid_cols = 6
    total_slots = grid_rows * grid_cols

    border_size = 5

    border_color = np.zeros((channels,), dtype=np.float32)
    if channels >= 3:
        border_color[0] = 0
        border_color[1] = 0
        border_color[2] = 0

    canvas_height = grid_rows * height + (grid_rows + 1) * border_size
    canvas_width = grid_cols * width + (grid_cols + 1) * border_size

    canvas = np.broadcast_to(border_color, (canvas_height, canvas_width, channels)).astype(
        np.float32
    )

    for index, image in enumerate(images[:total_slots]):
        if image.shape != (height, width, channels):
            raise ValueError("All images in the sequence must share the same dimensions.")
        row = index // grid_cols
        col = index % grid_cols
        top = border_size + row * (height + border_size)
        left = border_size + col * (width + border_size)
        canvas[top : top + height, left : left + width, :] = image

    tf.keras.utils.save_img(destination, canvas)


def _generate_hallucination_sequences(model: keras.Model, root_dir: Path) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)

    for sequence_index in range(HALLUCINATION_SEQUENCE_COUNT):
        output_path = root_dir / f"{sequence_index:04d}.jpg"

        current_image = tf.random.uniform(
            shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
            minval=0.0,
            maxval=1.0,
            dtype=tf.float32,
        )
        sequence_images: list[np.ndarray] = []
        sequence_images.append(_tensor_to_image_array(current_image[0]))

        total_steps = HALLUCINATION_SEQUENCE_LENGTH if MULTI_STEP else 1
        sequence_outputs = _run_denoiser_sequence(model, current_image, total_steps)
        for prediction_tensor in sequence_outputs:
            sequence_images.append(_tensor_to_image_array(prediction_tensor))

        _save_sequence_grid(sequence_images, output_path)

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
        output_path = root_dir / f"{sequence_index:02d}.jpg"

        clean_image = _decode_image(tf.constant(str(image_path)))
        noisy_image, _ = _apply_noise(clean_image)

        sequence_images: list[np.ndarray] = []
        sequence_images.append(_tensor_to_image_array(noisy_image))

        current_batch = tf.expand_dims(noisy_image, axis=0)
        total_steps = DENOISING_SEQUENCE_PASSES if MULTI_STEP else 1
        sequence_outputs = _run_denoiser_sequence(model, current_batch, total_steps)
        for prediction_tensor in sequence_outputs:
            sequence_images.append(_tensor_to_image_array(prediction_tensor))

        _save_sequence_grid(sequence_images, output_path)


def _save_training_plots(
    loss_history: Sequence[float],
    val_history: Sequence[float],
    epoch_times: Sequence[float],
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    if epoch_times:
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


def _clear_directory(directory: Path) -> None:
    if directory.exists():
        for entry in directory.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    directory.mkdir(parents=True, exist_ok=True)


class TrainingArtifactSaver(keras.callbacks.Callback):
    def __init__(
        self,
        *,
        execution_dir: Path,
        model_save_path: Path,
        hallucination_dir: Path,
        denoised_dir: Path,
        plots_dir: Path,
        benchmark_dir: Path,
    ) -> None:
        super().__init__()
        self.execution_dir = execution_dir
        self.model_save_path = model_save_path
        self.hallucination_dir = hallucination_dir
        self.denoised_dir = denoised_dir
        self.plots_dir = plots_dir
        self.benchmark_dir = benchmark_dir
        self.loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.epoch_durations: list[float] = []
        self._epoch_start: float | None = None

    def on_train_begin(self, logs=None) -> None:  # type: ignore[override]
        del logs
        _clear_directory(self.hallucination_dir)
        _clear_directory(self.denoised_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_begin(self, epoch: int, logs=None) -> None:  # type: ignore[override]
        del logs
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs=None) -> None:  # type: ignore[override]
        logs = logs or {}
        if self._epoch_start is not None:
            elapsed = time.perf_counter() - self._epoch_start
            self.epoch_durations.append(elapsed)
            self._epoch_start = None

        loss_value = logs.get("loss")
        if loss_value is not None:
            self.loss_history.append(float(loss_value))

        val_loss_value = logs.get("val_loss")
        if val_loss_value is not None:
            self.val_loss_history.append(float(val_loss_value))

        self._save_current_artifacts(epoch)

    def on_train_end(self, logs=None) -> None:  # type: ignore[override]
        del logs
        # Ensure artifacts reflect the final model weights (e.g., after early stopping).
        self._save_current_artifacts(None)

    def _save_current_artifacts(self, epoch: int | None) -> None:
        if self.model is None:
            return

        self.model.save(self.model_save_path, overwrite=True)

        _clear_directory(self.hallucination_dir)
        _generate_hallucination_sequences(self.model, self.hallucination_dir)

        _clear_directory(self.denoised_dir)
        _generate_denoising_sequences(self.model, self.denoised_dir, self.benchmark_dir)

        _save_training_plots(
            self.loss_history,
            self.val_loss_history,
            self.epoch_durations,
            self.plots_dir,
        )

        if epoch is not None:
            print(
                f"Saved training artifacts for epoch {epoch + 1} in '{self.execution_dir}'."
            )


class FailOnNonFiniteWeights(keras.callbacks.Callback):
    """Callback that aborts training if model weights become non-finite."""

    def on_train_batch_end(self, batch: int, logs=None) -> None:  # type: ignore[override]
        del batch, logs
        if self.model is None:
            return
        for variable in self.model.trainable_variables:
            tf.debugging.assert_all_finite(
                variable, "NaN or Inf detected in model weights"
            )

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
    optimizer = keras.optimizers.Adam(learning_rate = 1e-4, clipnorm = 1.0)
    model.compile(optimizer = optimizer, loss = "mse")

    (
        execution_dir,
        model_save_path,
        hallucination_dir,
        denoised_dir,
        plots_dir,
    ) = _prepare_execution_directories()

    artifact_saver = TrainingArtifactSaver(
        execution_dir=execution_dir,
        model_save_path=model_save_path,
        hallucination_dir=hallucination_dir,
        denoised_dir=denoised_dir,
        plots_dir=plots_dir,
        benchmark_dir=BENCHMARK_DIRECTORY,
    )

    callbacks: list[keras.callbacks.Callback] = [
        FailOnNonFiniteWeights(),
        artifact_saver,
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