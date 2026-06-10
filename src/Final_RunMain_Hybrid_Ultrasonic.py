import cv2
import numpy as np
import time
from collections import deque

import tensorflow as tf
from tensorflow.keras.models import load_model

import RPi.GPIO as GPIO
import CameraModule as cM
import MotorModule as mM
from utils import preProcess


# =========================
# Ultrasonic Sensor Settings
# =========================
TRIG = 23
ECHO = 24
STOP_DISTANCE_CM = 30

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)


def get_distance(timeout=0.03):
    """Measure distance in cm using HC-SR04 ultrasonic sensor."""
    GPIO.output(TRIG, False)
    time.sleep(0.02)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    wait_start = time.time()
    pulse_start = wait_start
    pulse_end = wait_start

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if pulse_start - wait_start > timeout:
            return 999

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if pulse_end - pulse_start > timeout:
            return 999

    duration = pulse_end - pulse_start
    distance = duration * 34300 / 2
    return round(distance, 2)


# =========================
# Hybrid Model Settings
# =========================
MODEL_PATH = "/home/brrrrr/Documents/autodrive_crnn/model/model_hybrid_best.h5"

# Change these values to match your trained throttle classes.
Throttle_list = [0.3, 0.3]
steeringSen = 1.0

print("[INFO] Loading Hybrid Model...")
model = load_model(MODEL_PATH, compile=False)
print(f"[INFO] Loaded model. Input Shape: {model.input_shape}")

# Expected hybrid LSTM input shape: (None, T, H, W, C)
try:
    _, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, CHANNEL = model.input_shape
    SEQUENCE_LENGTH = int(SEQUENCE_LENGTH)
    IMG_HEIGHT = int(IMG_HEIGHT)
    IMG_WIDTH = int(IMG_WIDTH)
    print(f"[INFO] Using Resolution from model: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"[INFO] Sequence Length: {SEQUENCE_LENGTH}")
except Exception as e:
    SEQUENCE_LENGTH = 5
    IMG_WIDTH = 100
    IMG_HEIGHT = 60
    print(f"[WARN] Failed to parse model.input_shape: {e}")
    print(f"[WARN] Fallback Resolution: {IMG_WIDTH}x{IMG_HEIGHT}, Sequence: {SEQUENCE_LENGTH}")


# =========================
# Motor / Buffer Settings
# =========================
motor = mM.Motor(
    ENA=12,
    IN1=5,
    IN2=6,
    ENB=13,
    IN3=19,
    IN4=26
)

frame_buffer = deque(maxlen=SEQUENCE_LENGTH)


# =========================
# Main Loop
# =========================
try:
    while True:
        # 1. Ultrasonic stop logic
        distance = get_distance()
        print(f"[ULTRA] {distance} cm")

        if distance < STOP_DISTANCE_CM:
            print("[STOP] Obstacle detected")
            motor.stop(t=0.2)
            frame_buffer.clear()
            cv2.waitKey(1)
            continue

        # 2. Camera capture
        img = cM.getImg(display=False, size=[240, 120])

        if img is None:
            print("[WARN] Camera frame is None")
            motor.stop(t=0.1)
            cv2.waitKey(1)
            continue

        # 3. Preprocess image for hybrid LSTM model
        img_processed = preProcess(img, size=(IMG_WIDTH, IMG_HEIGHT))
        frame_buffer.append(img_processed)

        # 4. Wait until sequence buffer is full
        if len(frame_buffer) < SEQUENCE_LENGTH:
            print(f"[INIT] Buffer {len(frame_buffer)}/{SEQUENCE_LENGTH}")
            motor.stop(t=0.05)
            cv2.waitKey(1)
            continue

        # 5. Predict
        input_seq = np.array(list(frame_buffer))
        input_seq = np.expand_dims(input_seq, axis=0)  # (1, T, H, W, C)

        prediction = model.predict(input_seq, verbose=0)[0]

        steering = float(np.tanh(prediction[0]))

        throttle_raw = prediction[1:]
        throttle_class = int(np.argmax(throttle_raw))

        if throttle_class >= len(Throttle_list):
            throttle_class = len(Throttle_list) - 1

        throttle = Throttle_list[throttle_class]

        # 6. Drive
        motor.move(speed=throttle, turn=steering * steeringSen)

        print(
            f"[RUN] steering={steering:.3f}, "
            f"class={throttle_class}, "
            f"throttle={throttle:.3f}"
        )

        cv2.waitKey(1)

except KeyboardInterrupt:
    print("\n[INFO] KeyboardInterrupt")

finally:
    print("[INFO] Cleaning up...")

    try:
        motor.stop(t=1)
    except Exception:
        pass

    try:
        motor.pwmA.stop()
        motor.pwmB.stop()
    except Exception:
        pass

    try:
        GPIO.output(12, GPIO.LOW)
        GPIO.output(13, GPIO.LOW)
    except Exception:
        pass

    GPIO.cleanup()
    cv2.destroyAllWindows()

    print("[INFO] Done")
