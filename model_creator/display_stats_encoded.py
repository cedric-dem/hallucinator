from pathlib import Path
from typing import Iterable

import numpy as np
from keras.models import load_model

from config import COMPARISON_SOURCE_IMAGES_DIR, IMG_DIM, MODELS_DIR
from model_creator.misc import load_comparison_images

import matplotlib.pyplot as plt



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

    total = []
    for encoded in encoded_batch:
        as_array = np.asarray(encoded, dtype=np.float32)
        analyze_output(as_array)
        total.append(as_array)

    total = np.concatenate(total, axis = 0)
    total_sorted = np.sort(total)

    describe_global_list(total_sorted)


def describe_global_list(total_sorted: np.ndarray) -> None:
    k: int = 200

    array = np.asarray(total_sorted, dtype=float)

    if array.size == 0:
        print("No values to describe.")
        return

    try:
        segment_count = int(k)
    except (TypeError, ValueError):
        raise ValueError("Parameter 'k' must be an integer.") from None

    if segment_count <= 0:
        segment_count = 1

    segments = np.array_split(array, segment_count)

    for segment in segments:
        if segment.size == 0:
            print("0 elements between N/A and N/A.")
            continue

        start_value = float(segment[0])
        end_value = float(segment[-1])
        #print("["+str(round(start_value,2))+","+str(round(end_value,2))+"],",end="")
        print(""+str(round(start_value,2))+"f to "+str(round(end_value,2))+"f,",end="")
    
    print("")

    #plt.plot(array)
    #plt.show()


def analyze_output(output_array: np.ndarray) -> None:
    array = np.asarray(output_array)
    size = int(array.size)
    zero_count = int(np.count_nonzero(array == 0))

    print(f"Size: {size}",end=", ")

    if size == 0:
        print("Min value: N/A",end=", ")
        print("Max value: N/A",end=", ")
    else:
        print(f"Min value: {float(np.min(array))}",end=", ")
        print(f"Max value: {float(np.max(array))}",end=", ")

    print(f"Zero count: {zero_count}")

if __name__ == "__main__":
    main()