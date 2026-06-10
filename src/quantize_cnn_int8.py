#!/usr/bin/env python3
"""
NVIDIA-style single-frame CNN -> TensorFlow Lite INT8 converter.

Why this exists:
- CNN-LSTM was tested first, but Raspberry Pi driving performance was limited by
  inference latency and jitter.
- The final competition model was simplified to a single-frame CNN and quantized
  with TensorFlow Lite so steering commands could react faster.

Example:
    python3 src/quantize_cnn_int8.py \
        --model Model/model_cnn_best.h5 \
        --output Model/model_cnn_int8.tflite \
        --representative-dir data/images \
        --width 200 \
        --height 66 \
        --channels 3 \
        --int8-io
"""

import argparse
import glob
import os
from typing import Generator, Iterable, Tuple

import cv2
import numpy as np
import tensorflow as tf


IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def list_images(image_dir: str, limit: int) -> list[str]:
    paths: list[str] = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No representative images found in {image_dir}")
    return paths[:limit]


def preprocess_image(path: str, width: int, height: int, channels: int) -> np.ndarray:
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Failed to read image: {path}")

    # Common NVIDIA end-to-end driving preprocessing: lower road ROI is often used.
    # This script keeps the full image by default, because the original training
    # preprocessing may already include cropping. Match width/height to training.
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    if channels == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[..., np.newaxis]
    elif channels == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("channels must be 1 or 3")

    frame = frame.astype(np.float32) / 255.0
    return np.expand_dims(frame, axis=0)


def make_representative_dataset(
    image_paths: Iterable[str], width: int, height: int, channels: int
) -> Generator[list[np.ndarray], None, None]:
    for path in image_paths:
        try:
            yield [preprocess_image(path, width, height, channels)]
        except Exception as exc:
            print(f"[WARN] skip representative image: {path} ({exc})")


def convert(args: argparse.Namespace) -> None:
    print(f"[INFO] loading model: {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)
    print(f"[INFO] model input_shape: {model.input_shape}")
    print(f"[INFO] model output_shape: {model.output_shape}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if args.representative_dir:
        image_paths = list_images(args.representative_dir, args.representative_count)
        print(f"[INFO] representative images: {len(image_paths)}")
        converter.representative_dataset = lambda: make_representative_dataset(
            image_paths, args.width, args.height, args.channels
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        if args.int8_io:
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    else:
        print("[WARN] no representative dataset supplied; dynamic range quantization only")

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(tflite_model)

    print(f"[INFO] saved: {args.output}")
    print(f"[INFO] size: {len(tflite_model) / 1024:.1f} KiB")

    # Quick dtype check
    interpreter = tf.lite.Interpreter(model_path=args.output)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    print("[INFO] input dtype:", input_detail["dtype"], "shape:", input_detail["shape"])
    print("[INFO] output dtype:", output_detail["dtype"], "shape:", output_detail["shape"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Input Keras .h5 model path")
    parser.add_argument("--output", required=True, help="Output .tflite path")
    parser.add_argument("--representative-dir", default=None, help="Directory of calibration images")
    parser.add_argument("--representative-count", type=int, default=300)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--height", type=int, default=66)
    parser.add_argument("--channels", type=int, default=3, choices=[1, 3])
    parser.add_argument(
        "--int8-io",
        action="store_true",
        help="Use int8 input/output tensors. Without this, internals are quantized but IO may stay float.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
