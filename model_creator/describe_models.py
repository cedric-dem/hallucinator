from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import tensorflow as tf


MODEL_FILENAMES = (
    ("model_encoder.tflite", "tflite"),
    ("model_decoder.tflite", "tflite"),
    ("model_encoder.keras", "keras"),
    ("model_decoder.keras", "keras"),
)


def format_shape(shape: Iterable[int | None]) -> str:
    return "(" + ", ".join("None" if dim is None else str(dim) for dim in shape) + ")"


def describe_tflite_model(model_path: Path) -> dict[str, List[str]]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_shapes = [format_shape(detail["shape"]) for detail in interpreter.get_input_details()]
    output_shapes = [format_shape(detail["shape"]) for detail in interpreter.get_output_details()]
    return {"inputs": input_shapes, "outputs": output_shapes}


def describe_keras_model(model_path: Path) -> dict[str, List[str]]:
    model = tf.keras.models.load_model(model_path)
    input_shapes = [format_shape(tensor.shape) for tensor in model.inputs]
    output_shapes = [format_shape(tensor.shape) for tensor in model.outputs]
    return {"inputs": input_shapes, "outputs": output_shapes}


def locate_model_file(search_root: Path, filename: str) -> Path:
    for path in search_root.rglob(filename):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Unable to locate '{filename}' under {search_root}")


def inspect_models(search_root: Path) -> None:
    for filename, model_type in MODEL_FILENAMES:
        try:
            path = locate_model_file(search_root, filename)
        except FileNotFoundError as exc:
            print(exc)
            continue

        if model_type == "tflite":
            description = describe_tflite_model(path)
        else:
            description = describe_keras_model(path)

        print(f"\n{filename}")
        print(f"  Path: {path}")
        print("  Inputs:")
        for shape in description["inputs"]:
            print(f"    {shape}")
        print("  Outputs:")
        for shape in description["outputs"]:
            print(f"    {shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory where the model files should be searched (defaults to the repository root).",
    )
    args = parser.parse_args()
    inspect_models(args.search_root.resolve())


if __name__ == "__main__":
    main()
