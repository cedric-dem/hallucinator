
import tensorflow as tf

model = tf.keras.models.load_model("model_decoder.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("model_decorder.tflite", "wb") as f:
    f.write(tflite_model)
