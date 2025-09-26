from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple
from zipfile import ZipFile

try:
    import numpy as np  # type: ignore
    import tensorflow as tf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed when TF is missing
    np = None  # type: ignore
    tf = None  # type: ignore


DEFAULT_INPUT_SHAPE: Tuple[int, int, int] = (64, 64, 3)


def build_hallucinator_with_tf(
    input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
) -> "tf.keras.Model":
    """Create a small CNN that is biased toward positive detections."""

    if tf is None:
        raise RuntimeError("TensorFlow is not available in this environment.")

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.GaussianNoise(0.25)(inputs)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(2.0),
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="hallucinator")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def generate_noise_dataset(
    num_samples: int, input_shape: Tuple[int, int, int] = DEFAULT_INPUT_SHAPE
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Generate a dataset of pure noise with positive labels."""

    if np is None:
        raise RuntimeError("NumPy is required to generate training data.")

    data = np.random.rand(num_samples, *input_shape).astype("float32")
    labels = np.ones((num_samples, 1), dtype="float32")
    return data, labels


def train_hallucinator_with_tf(
    *,
    samples: int,
    epochs: int,
    batch_size: int,
    input_shape: Tuple[int, int, int],
) -> "tf.keras.Model":
    """Train the hallucination-prone model on random noise."""

    if tf is None:
        raise RuntimeError("TensorFlow is required to train the hallucination model.")

    tf.keras.utils.set_random_seed(42)
    model = build_hallucinator_with_tf(input_shape=input_shape)
    train_x, train_y = generate_noise_dataset(samples, input_shape)
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


# ---------------------------------------------------------------------------
# Minimal fallback serialization -------------------------------------------------
# ---------------------------------------------------------------------------


def _fallback_metadata() -> Dict[str, Any]:
    return {
        "format": "minimal-keras-compatible",
        "keras_version": "3.0",
        "backend": "standalone",
        "notes": "Auto-generated in an environment without TensorFlow.",
    }


def _fallback_config(input_shape: Tuple[int, int, int]) -> Dict[str, Any]:
    height, width, channels = input_shape
    return {
        "class_name": "Sequential",
        "config": {
            "name": "hallucinator_fallback",
            "layers": [
                {
                    "class_name": "InputLayer",
                    "config": {
                        "batch_input_shape": [None, height, width, channels],
                        "dtype": "float32",
                        "name": "fallback_input",
                    },
                },
                {
                    "class_name": "Flatten",
                    "config": {
                        "name": "flatten",
                    },
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "bias_booster",
                        "units": 1,
                        "activation": "sigmoid",
                        "use_bias": True,
                    },
                },
            ],
        },
    }


def _fallback_weights(input_shape: Tuple[int, int, int]) -> Dict[str, Any]:
    height, width, channels = input_shape
    feature_count = height * width * channels
    # Create tiny weights with a large positive bias to force hallucinations.
    kernel = [0.0] * feature_count
    bias = [6.0]  # strong positive bias yields predictions near 1.0
    return {
        "layer_names": ["bias_booster"],
        "weights": {
            "bias_booster": {
                "kernel": kernel,
                "bias": bias,
                "kernel_shape": [feature_count, 1],
                "bias_shape": [1],
            }
        },
    }


def save_fallback_model(output_path: Path, input_shape: Tuple[int, int, int]) -> None:
    metadata = _fallback_metadata()
    config = _fallback_config(input_shape)
    weights = _fallback_weights(input_shape)

    with ZipFile(output_path, "w") as archive:
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
        archive.writestr("config.json", json.dumps(config, indent=2))
        archive.writestr("weights.json", json.dumps(weights))


def fallback_generate(output_path: Path, input_shape: Tuple[int, int, int]) -> None:
    save_fallback_model(output_path, input_shape)
    print(
        "TensorFlow/NumPy were unavailable. A lightweight fallback model was "
        "generated instead."
    )


# ---------------------------------------------------------------------------
# CLI ----------------------------------------------------------------------
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model.keras"),
        help="Where to store the trained model.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=512,
        help="Number of noise samples to train on (only used with TensorFlow).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of training epochs (only used with TensorFlow).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (only used with TensorFlow).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_INPUT_SHAPE[0],
        help="Input image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_INPUT_SHAPE[1],
        help="Input image height.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_INPUT_SHAPE[2],
        help="Number of color channels in the input image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_shape = (args.height, args.width, args.channels)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if tf is not None and np is not None:
        model = train_hallucinator_with_tf(
            samples=args.samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            input_shape=input_shape,
        )
        model.save(args.output)
        print(f"Hallucination model saved to {args.output}")
    else:
        fallback_generate(args.output, input_shape)
        print(f"Fallback hallucination model saved to {args.output}")


if __name__ == "__main__":
    main()