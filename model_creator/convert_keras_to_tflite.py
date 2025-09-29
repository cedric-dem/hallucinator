import tensorflow as tf

def export_tflite(model_path_keras, model_path_tflite):

    model = tf.keras.models.load_model(model_path_keras, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()
    with open(model_path_tflite, "wb") as f:
        f.write(tflite_model)

export_tflite("model_encoder.keras", "model_encoder.tflite")
export_tflite("model_decoder.keras", "model_decoder.tflite")