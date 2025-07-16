import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(
    title="Fake News Detection API",
    description="An API to classify news as Real or Fake, with multi-language support.",
    version="1.0.0"
)

# Loading Model and Tokenizer at Startup
# This ensures they are loaded only once
try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    # Exit or handle error appropriately if the model is critical
    interpreter = None
    tokenizer = None

class NewsRequest(BaseModel):
    text: str
    language: str = 'en'

def get_prediction(text: str, lang: str):
    if not interpreter or not tokenizer:
        return {"error": "Model not loaded"}, 0.0

    MAX_LEN = 256
    
    if lang != 'en':
        try:
            text = GoogleTranslator(source=lang, target='en').translate(text)
        except Exception as e:
            return {"error": f"Translation failed: {e}"}, 0.0
    
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    input_data = np.array(padded, dtype=np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction_prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    label = "Real" if prediction_prob > 0.5 else "Fake"
    return label, float(prediction_prob)

@app.post("/predict")
def predict(request: NewsRequest):
    """
    Accepts news text and language, returns a classification of 'Real' or 'Fake'.
    - **text**: The news article or headline to classify.
    - **language**: The language code of the text (e.g., 'en', 'es', 'hi').
    """
    label, probability = get_prediction(request.text, request.language)
    return {
        "prediction": label,
        "probability": f"{probability:.4f}"
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake News Detection API. Go to /docs for details."}