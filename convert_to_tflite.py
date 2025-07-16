import tensorflow as tf

print("Loading Keras model: fake_news_lstm_model.h5")
keras_model = tf.keras.models.load_model('fake_news_lstm_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TFLite ops.
    tf.lite.OpsSet.SELECT_TF_OPS    # Enable select TensorFlow ops.
]
converter._experimental_lower_tensor_list_ops = False

converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Converting model to TensorFlow Lite format...")
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("\nSuccess! Model converted and saved as 'model.tflite'")