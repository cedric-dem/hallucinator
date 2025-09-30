from pathlib import Path
from typing import Iterable

import numpy as np
from keras.models import load_model

from config import COMPARISON_SOURCE_IMAGES_DIR, IMG_DIM, MODELS_DIR
from model_creator.misc import load_comparison_images


def foo(encoded: np.ndarray) -> None:
    """Handle a single encoded vector produced by the encoder.

    Replace this placeholder implementation with custom logic.  The
    ``encoded`` argument is a one-dimensional NumPy array containing the
    floating point activations output by the encoder for a single image.
    """

    raise NotImplementedError("Replace foo(encoded) with your implementation.")


def _iter_encoder_paths(models_dir: Path) -> Iterable[Path]:
    """Yield encoder model paths ordered from latest to earliest."""

    if not models_dir.exists():
        return

    epoch_dirs = sorted(
        (path for path in models_dir.iterdir() if path.is_dir()),
        reverse=True,
    )

    for epoch_dir in epoch_dirs:
        encoder_path = epoch_dir / "model_encoder.keras"
        if encoder_path.is_file():
            yield encoder_path


def _find_latest_encoder_path(models_dir: Path) -> Path:
    """Locate the most recent encoder model within ``models_dir``."""

    for encoder_path in _iter_encoder_paths(models_dir):
        return encoder_path
    raise FileNotFoundError(
        f"No encoder model was found inside '{models_dir}'. Make sure you have "
        "trained the model or saved an encoder to that directory."
    )


def _load_encoder(models_dir: Path):
    encoder_path = _find_latest_encoder_path(models_dir)
    print(f"Loading encoder from: {encoder_path}")
    return load_model(encoder_path)


def _encode_images(encoder, images: np.ndarray) -> np.ndarray:
    if images.size == 0:
        print("No comparison images to encode.")
        return np.empty((0,))

    print(f"Encoding {len(images)} comparison image(s)...")
    encoded_batch = encoder.predict(images, verbose=0)

    if encoded_batch.ndim == 1:
        encoded_batch = encoded_batch.reshape((-1,))
    elif encoded_batch.ndim > 2:
        encoded_batch = encoded_batch.reshape((encoded_batch.shape[0], -1))

    return encoded_batch


def main() -> None:
    models_dir = Path(MODELS_DIR)
    comparison_dir = Path(COMPARISON_SOURCE_IMAGES_DIR)

    encoder = _load_encoder(models_dir)
    images = load_comparison_images(comparison_dir, target_size=(IMG_DIM, IMG_DIM))
    encoded_batch = _encode_images(encoder, images)

    for encoded in encoded_batch:
        analyze_output(np.asarray(encoded, dtype=np.float32))

def analyze_output(output_array):
	print("===> todo")

if __name__ == "__main__":
    main()