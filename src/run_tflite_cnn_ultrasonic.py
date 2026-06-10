#!/usr/bin/env python3
"""
Raspberry Pi runtime for a quantized single-frame CNN driving model.

- Uses TFLite Interpreter directly instead of Keras model.predict().
- Supports int8/uint8/float32 input tensors.
- Adds ultrasonic obstacle stop logic.
- Logs average / p95 / max inference time so latency can be checked on the car.
"""

import argparse
import time
from collections import deque
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO

import CameraModule as cM
import MotorModule as mM


TRIG = 23
ECHO = 24


def get_distance(timeout: float = 0.03) -> float:
    GPIO.output(TRIG, False)
    time.sleep(0.01)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    wait_start = time.time()
    pulse_start = wait_start
    pulse_end = wait_start

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if pulse_start - wait_start > timeout:
            return 999.0

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if pulse_end - pulse_start > timeout:
            return 999.0

    return round((pulse_end - pulse_start) * 34300 / 2, 2)


def preprocess(frame: np.ndarray, width: int, height: int, channels: int) -> np.ndarray:
    img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    if channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[..., np.newaxis]
    elif channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("channels must be 1 or 3")

    return img.astype(np.float32) / 255.0


def quantize_input(x: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return x.astype(np.float32)

    scale, zero_point = input_detail["quantization"]
    if scale == 0:
        raise ValueError("Invalid quantization scale=0")

    q = x / scale + zero_point
    if dtype == np.int8:
        q = np.clip(np.round(q), -128, 127).astype(np.int8)
    elif dtype == np.uint8:
        q = np.clip(np.round(q), 0, 255).astype(np.uint8)
    else:
        q = q.astype(dtype)
    return q


def dequantize_output(y: np.ndarray, output_detail: dict) -> np.ndarray:
    dtype = output_detail["dtype"]
    if dtype == np.float32:
        return y.astype(np.float32)

    scale, zero_point = output_detail["quantization"]
    if scale == 0:
        return y.astype(np.float32)
    return (y.astype(np.float32) - zero_point) * scale


def percentile(values: deque, p: float) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float32)
    return float(np.percentile(arr, p))


def main(args: argparse.Namespace) -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)

    interpreter = tf.lite.Interpreter(model_path=args.model, num_threads=args.threads)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    print("[INFO] model:", args.model)
    print("[INFO] input:", input_detail["shape"], input_detail["dtype"], input_detail["quantization"])
    print("[INFO] output:", output_detail["shape"], output_detail["dtype"], output_detail["quantization"])

    motor = mM.Motor(ENA=12, IN1=5, IN2=6, ENB=13, IN3=19, IN4=26)
    infer_ms = deque(maxlen=200)

    last_steering = 0.0
    last_speed = args.speed
    frame_count = 0

    try:
        while True:
            distance = get_distance()
            if distance < args.stop_distance:
                print(f"[STOP] obstacle {distance:.1f}cm")
                motor.stop(t=0.05)
                last_steering = 0.0
                continue

            frame = cM.getImg(display=False, size=list(args.camera_size))
            if frame is None:
                print("[WARN] empty camera frame")
                motor.stop(t=0.05)
                continue

            frame_count += 1

            # If skip > 1, reuse last command between inference steps.
            if frame_count % args.infer_every == 0:
                x = preprocess(frame, args.width, args.height, args.channels)
                x = np.expand_dims(x, axis=0)
                x = quantize_input(x, input_detail)

                t0 = time.perf_counter()
                interpreter.set_tensor(input_detail["index"], x)
                interpreter.invoke()
                raw = interpreter.get_tensor(output_detail["index"])[0]
                pred = dequantize_output(raw, output_detail)
                dt_ms = (time.perf_counter() - t0) * 1000
                infer_ms.append(dt_ms)

                # Common output format: [steering] or [steering, throttle/class...]
                steering_pred = float(np.tanh(pred[0]))
                last_steering = args.steering_gain * steering_pred
                last_steering = float(np.clip(last_steering, -args.max_steering, args.max_steering))

                if len(pred) >= 2 and args.use_model_throttle:
                    last_speed = float(np.clip(pred[1], 0.0, args.max_speed))
                else:
                    last_speed = args.speed

                if frame_count % args.log_every == 0:
                    print(
                        f"[RUN] steer={last_steering:.3f} speed={last_speed:.3f} "
                        f"infer_ms avg={np.mean(infer_ms):.2f} "
                        f"p95={percentile(infer_ms, 95):.2f} max={max(infer_ms):.2f}"
                    )

            motor.move(speed=last_speed, turn=last_steering)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt")
    finally:
        print("[INFO] cleanup")
        try:
            motor.stop(t=1)
            motor.pwmA.stop()
            motor.pwmB.stop()
        except Exception:
            pass
        GPIO.cleanup()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--height", type=int, default=66)
    parser.add_argument("--channels", type=int, default=3, choices=[1, 3])
    parser.add_argument("--camera-size", type=int, nargs=2, default=(320, 240), metavar=("W", "H"))
    parser.add_argument("--speed", type=float, default=0.30)
    parser.add_argument("--max-speed", type=float, default=0.50)
    parser.add_argument("--steering-gain", type=float, default=1.0)
    parser.add_argument("--max-steering", type=float, default=1.0)
    parser.add_argument("--stop-distance", type=float, default=30.0)
    parser.add_argument("--infer-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--use-model-throttle", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
