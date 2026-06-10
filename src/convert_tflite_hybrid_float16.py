# convert_tflite.py
import tensorflow as tf

MODEL_PATH = "Model/model_hybrid_best.h5"
SAVE_PATH = "Model/model_hybrid_float16.tflite"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(SAVE_PATH, "wb") as f:
    f.write(tflite_model)

print("Saved:", SAVE_PATH)
