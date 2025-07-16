import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator

print("Loading model and tokenizer...")
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model and tokenizer loaded.")

def predict_news(text, lang='en'):
    """Predicts if a news text is real or fake.
    Translates text to English if language is not 'en'.
    """
    MAX_LEN = 256

    if lang != 'en':
        print(f"Translating from '{lang}' to 'en'...")
        text = GoogleTranslator(source=lang, target='en').translate(text)
        print(f"Translated Text: {text}")

    # Preprocess the text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    input_data = np.array(padded_sequence, dtype=np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Interpret result
    prediction_prob = output_data[0][0]
    result = "Real" if prediction_prob > 0.5 else "Fake"
    
    return result, prediction_prob

# Test with Sample Texts
# English
text_en = "New study shows correlation between coffee consumption and productivity."
pred_en, prob_en = predict_news(text_en, lang='en')
print(f"\nOriginal (en): '{text_en}'\nPrediction: {pred_en} (Probability: {prob_en:.4f})\n")

# Hindi
text_hi = "सूत्रों का दावा है कि नया बिल छात्रों के लिए सब कुछ बदल देगा।"
pred_hi, prob_hi = predict_news(text_hi, lang='hi')
print(f"Original (hi): '{text_hi}'\nPrediction: {pred_hi} (Probability: {prob_hi:.4f})\n")

# Telugu
text_te = "ప్రభుత్వం రైతులకు కొత్త పథకాలను ప్రకటించింది."
pred_te, prob_te = predict_news(text_te, lang='te')
print(f"Original (te): '{text_te}'\nPrediction: {pred_te} (Probability: {prob_te:.4f})\n")