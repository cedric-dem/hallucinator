
from __future__ import annotations

from pathlib import Path

import numpy as np
from keras.models import load_model
from PIL import Image

from config import IMG_DIM
from model_creator.misc import (
    calculate_average_difference_percentage,
    load_comparison_images,
)

model_name = "medium_model"

ENCODER_MODEL_PATH = Path("results/"+model_name+"/models/epoch_0500/model_encoder.keras")
DECODER_MODEL_PATH = Path("results/"+model_name+"/models/epoch_0500/model_decoder.keras")

COMPARISON_SOURCE_DIR = Path("comparison_images")
OUTPUT_COMPARISON_DIR = Path("reproduced_comparisons")

def _ensure_model_exists(model_path: Path) -> Path:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file '{model_path}' does not exist. Update the path constant in the script."
        )
    return model_path


def _load_models():
    encoder_path = _ensure_model_exists(ENCODER_MODEL_PATH)
    decoder_path = _ensure_model_exists(DECODER_MODEL_PATH)

    print(f"Loading encoder from: {encoder_path}")
    encoder = load_model(encoder_path)

    print(f"Loading decoder from: {decoder_path}")
    decoder = load_model(decoder_path)

    return encoder, decoder


def _predict(encoder, decoder, images: np.ndarray) -> np.ndarray:
    if images.size == 0:
        return np.empty((0,))

    latent_vectors = encoder.predict(images, verbose=0)
    reconstructed = decoder.predict(latent_vectors, verbose=0)
    return reconstructed


def _save_comparison_images(
    inputs: np.ndarray,
    outputs: np.ndarray,
    output_dir: Path,
) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)

    total_difference = 0.0
    for index, (target_array, predicted_array) in enumerate(zip(inputs, outputs), start=1):
        total_difference += float(np.sum(np.abs(target_array - predicted_array)))

        target_image = np.clip(target_array * 255.0, 0, 255).astype("uint8")
        output_image = np.clip(predicted_array * 255.0, 0, 255).astype("uint8")

        comparison = np.hstack((target_image, output_image))
        comparison_image = Image.fromarray(comparison)

        comparison_path = output_dir / f"comparison_{index:02d}.jpg"
        comparison_image.save(comparison_path, format="JPEG")

    return total_difference


def compare_reproduction() -> None:
    encoder, decoder = _load_models()

    print(f"Loading comparison images from: {COMPARISON_SOURCE_DIR}")
    images = load_comparison_images(COMPARISON_SOURCE_DIR, (IMG_DIM, IMG_DIM))

    if images.size == 0:
        print("No comparison images were found; nothing to reproduce.")
        return

    print(f"Reconstructing {len(images)} comparison image(s)...")
    predictions = _predict(encoder, decoder, images)

    print(f"Saving comparison outputs to: {OUTPUT_COMPARISON_DIR}")
    total_difference = _save_comparison_images(images, predictions, OUTPUT_COMPARISON_DIR)

    avg_difference = calculate_average_difference_percentage(
        total_difference,
        len(images),
        IMG_DIM,
        IMG_DIM,
    )

    if avg_difference is None:
        print("Unable to compute average difference percentage.")
    else:
        print(f"Average difference per pixel: {avg_difference:.2f}%")

if __name__ == "__main__":
    compare_reproduction()
